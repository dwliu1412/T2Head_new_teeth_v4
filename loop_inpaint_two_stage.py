"""Two-stage offline pseudo-GT refinement for a FLAME-bound Gaussian avatar.

This entry point replaces the online/random refresh schedule used by the
legacy ``loop_inpaint.py`` with two small, immutable target banks:

1. ``stage1_mouth`` selects a deterministic subset of *open-mouth* expression
   and pose pairs and generates teachers only at the frontal camera.  The fit
   is restricted to lips, teeth, and the oral cavity.
2. ``stage2_multiview`` starts from the Stage-1 model, selects a larger bank of
   diverse dynamic open-mouth poses/expressions, generates low-strength
   teachers at a controlled camera grid, and performs a shorter multi-view fit.

Targets are generated once before each fit.  They are never refreshed while
the optimizer is running, so the amount of supervision is bounded and
reproducible.  After each fit, the complete paired
``assets/test/{exp.npy,pose.npy}`` sequence is rendered at the calibrated front
camera.

The implementation reuses the current repository's validated target generator
and direct UVD fitter from ``tools/direct_loop_inpaint_targets.py``.  A normal
run is:

    python loop_inpaint_two_stage.py \
        --config configs/direct_loop_inpaint_targets.yaml

Useful overrides include:

    two_stage.stage1.pose_bank.count=4
    two_stage.stage2.pose_bank.count=16
    two_stage.stage2.config.teacher.source_azimuths=[225,240,270,300,315]
    two_stage.stage1.config.optimization.geometry_iterations=800
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import torch
import yaml

from tools import direct_loop_inpaint_targets as direct


_ORIGINAL_JAW_POSE_BANK = direct.jaw_pose_bank
_ACTIVE_TWO_STAGE_CONFIG: Optional[Mapping[str, Any]] = None


DEFAULT_TWO_STAGE: Dict[str, Any] = {
    # ``None`` derives "<input reconstruction>_two_stage_mouth_views".
    "name": None,
    "require_open_mouth": True,
    "test_view": {
        "source_azimuth": 270.0,
        "source_elevation": 0.0,
    },
    "common": {
        "config": {
            "data": {
                # MediaPipe ControlNet conditions are only needed by the
                # disabled global RV5.1 pass.
                "use_mediapipe_condition": False,
                "test_expression_path": "assets/test/exp.npy",
                "test_pose_path": "assets/test/pose.npy",
            },
            "teacher": {
                "include_reference_pose": False,
                "inpaint_reference_pose": False,
                "pose_quantiles": [],
                "include_validation_pose": False,
                # Select diffusion candidates independently for every pose,
                # then reuse the selected candidate across all views of that
                # pose.  A globally shared candidate can work for the widest
                # jaw but erase teeth in another expression.
                "seed_mode": "pose",
                "seed_overrides": {},
                "accepted_mouth_priors": {},
                "samples_per_view": 1,
                # Candidate selection runs once per pose; the selected layout
                # is reused for every calibrated view of that pose.  This
                # improves robustness without enlarging the training bank.
                "mouth_candidates_per_view": 4,
                # Global portrait diffusion is deliberately disabled here.
                # It is the main route by which a large target bank can blur
                # identity.  SDXL edits only the mouth mask.
                "rv51": {
                    "enabled": False,
                },
                "quality": {
                    # Keep QA measurements in manifest.json for inspection,
                    # but never reject a generated pseudo target.  Diffusion
                    # variance can otherwise stop a long target-bank run on a
                    # single unusually wide/bright mouth.
                    "enforce": False,
                    "min_non_mouth_delta": 0.0,
                    "min_ear_edit_coverage": 0.0,
                    "min_ear_delta": 0.0,
                    # The widest validation jaw can produce a legitimate
                    # frontal tooth span around 0.805.  Keep the denture gate,
                    # but give wide-open anchors a small tolerance.
                    "max_teeth_component_span": 0.85,
                },
            },
            "optimization": {
                "geometry_regions": [
                    "teeth_upper",
                    "teeth_lower",
                    "oral_cavity",
                ],
                "appearance_regions": [
                    "teeth_upper",
                    "teeth_lower",
                    "oral_cavity",
                    "lips",
                ],
                "opacity_regions": [
                    "teeth_upper",
                    "teeth_lower",
                    "oral_cavity",
                    "lips",
                ],
            },
            "loss": {
                # Retain the Stage-1 face exactly outside the local mouth edit.
                "identity_weight": 20.0,
            },
            "test_render": {
                "enabled": True,
            },
        },
    },
    "stage1": {
        "output_subdir": "stage1_mouth",
        "learning_rate_scale": 1.0,
        "pose_bank": {
            "source": "validation_open_mouth",
            "count": 6,
            "min_jaw_quantile": 0.80,
        },
        "config": {
            "teacher": {
                "source_elevation": 0.0,
                "source_azimuths": [270.0],
                "strength": 0.70,
                "num_inference_steps": 45,
                "guidance_scale": 6.5,
            },
            "optimization": {
                "geometry_iterations": 1500,
                "appearance_iterations": 500,
            },
        },
    },
    "stage2": {
        "output_subdir": "stage2_multiview",
        "learning_rate_scale": 0.5,
        "pose_bank": {
            # Retain several wide-open Stage-1 anchors, then add diverse
            # chemistry expressions/poses.  This expands supervision without
            # forgetting the strongest tooth/cavity observations.
            "source": "expanded_open_mouth",
            "count": 12,
            "validation_anchor_count": 3,
            "validation_min_jaw_quantile": 0.80,
            "chemistry_min_jaw_quantile": 0.10,
        },
        "config": {
            "bootstrap": {
                # The Stage-1 output already contains the seeded and fitted
                # oral topology.  Bootstrapping it a second time would erase
                # precisely what Stage 1 learned.
                "enabled": False,
            },
            "teacher": {
                "source_elevation": 0.0,
                "source_azimuths": [210.0, 240.0, 270.0, 300.0, 330.0],
                # Low-strength inpainting preserves the learned frontal tooth
                # layout while filling view-dependent evidence.
                "strength": 0.30,
                "num_inference_steps": 30,
                "guidance_scale": 5.5,
            },
            "optimization": {
                "geometry_iterations": 250,
                "appearance_iterations": 750,
            },
        },
    },
}


LEARNING_RATE_KEYS = (
    "feature_lr",
    "opacity_lr",
    "d_lr",
    "scale_lr",
    "rotation_lr",
)


def deep_merge(
    destination: MutableMapping[str, Any], source: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    """Recursively merge mappings while replacing scalar/list values."""

    for key, value in source.items():
        current = destination.get(key)
        if isinstance(current, MutableMapping) and isinstance(value, Mapping):
            deep_merge(current, value)
        else:
            destination[key] = copy.deepcopy(value)
    return destination


def safe_relative_path(value: Any, label: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"{label} must be a relative path without '..': {value}")
    if not path.parts:
        raise ValueError(f"{label} must not be empty")
    return path


def resolve_original_input_dir(
    reconstruction_dir: Path, resolved_config: Mapping[str, Any]
) -> Path:
    """Resolve the calibrated image/camera directory of the original model."""

    raw = resolved_config.get("input_dir")
    if not isinstance(raw, str) or not raw.strip():
        raise KeyError(
            f"{reconstruction_dir / 'resolved_config.yaml'} is missing input_dir"
        )
    candidate = Path(raw).expanduser()
    candidates = (
        [candidate.resolve()]
        if candidate.is_absolute()
        else [
            (direct.PROJECT_ROOT / candidate).resolve(),
            (reconstruction_dir / candidate).resolve(),
        ]
    )
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        "Cannot resolve the original calibrated input_dir. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def reconstruction_provenance(
    reconstruction_dir: Path,
) -> Dict[str, Any]:
    """Carry camera/image provenance through the intermediate Stage-1 model."""

    path = reconstruction_dir / "resolved_config.yaml"
    with path.open("r", encoding="utf-8") as file:
        resolved = yaml.safe_load(file)
    if not isinstance(resolved, Mapping):
        raise ValueError(f"{path}: expected a YAML mapping")
    provenance: Dict[str, Any] = {
        "input_dir": str(resolve_original_input_dir(reconstruction_dir, resolved))
    }
    model = resolved.get("model")
    if isinstance(model, Mapping):
        provenance["model"] = copy.deepcopy(dict(model))
    return provenance


def calibrated_front_camera(
    input_dir: Path, test_view: Mapping[str, Any]
) -> Dict[str, Any]:
    """Select the calibrated frame nearest the requested frontal direction."""

    camera_path = input_dir / "cameras.json"
    with camera_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    frames = metadata.get("frames") if isinstance(metadata, Mapping) else None
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{camera_path}: missing non-empty frames list")
    target_azimuth = float(test_view.get("source_azimuth", 270.0))
    target_elevation = float(test_view.get("source_elevation", 0.0))
    candidates: List[Tuple[float, int, float, float]] = []
    for fallback_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            continue
        azimuth_value = frame.get(
            "azimuth_degrees", frame.get("relative_azimuth_deg")
        )
        elevation_value = frame.get(
            "elevation_degrees", frame.get("elevation_deg")
        )
        if azimuth_value is None or elevation_value is None:
            continue
        azimuth = float(azimuth_value)
        elevation = float(elevation_value)
        azimuth_error = direct.circular_distance_degrees(
            azimuth, target_azimuth
        )
        elevation_error = abs(elevation - target_elevation)
        score = azimuth_error * azimuth_error + elevation_error * elevation_error
        candidates.append(
            (
                score,
                int(frame.get("frame_index", fallback_index)),
                azimuth,
                elevation,
            )
        )
    if not candidates:
        raise ValueError(
            f"{camera_path}: no frame exposes source azimuth/elevation metadata"
        )
    _, frame_index, azimuth, elevation = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    return {
        "frame_index": frame_index,
        "source_azimuth": azimuth,
        "source_elevation": elevation,
        "requested_source_azimuth": target_azimuth,
        "requested_source_elevation": target_elevation,
    }


def stage_output_name(run_name: Path, stage_settings: Mapping[str, Any]) -> str:
    subdir = safe_relative_path(
        stage_settings["output_subdir"], "two_stage.*.output_subdir"
    )
    return (run_name / subdir).as_posix()


def scale_learning_rates(config: MutableMapping[str, Any], scale: float) -> None:
    if scale <= 0.0:
        raise ValueError("learning_rate_scale must be positive")
    optimization = config["optimization"]
    for key in LEARNING_RATE_KEYS:
        if key in optimization:
            optimization[key] = float(optimization[key]) * scale


def build_stage_configs(
    base_config: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Create independent resolved configs for mouth-first and multi-view fits."""

    supplied = base_config.get("two_stage", {})
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, Mapping):
        raise ValueError("two_stage must be a mapping")
    settings = copy.deepcopy(DEFAULT_TWO_STAGE)
    deep_merge(settings, supplied)

    clean_base = copy.deepcopy(dict(base_config))
    clean_base.pop("two_stage", None)
    reconstruction_name = Path(
        str(clean_base["input"]["reconstruction_dir"])
    ).name
    requested_name = settings.get("name")
    run_name = safe_relative_path(
        requested_name or f"{reconstruction_name}_two_stage_mouth_views",
        "two_stage.name",
    )

    original_reconstruction = direct.resolve_path(
        clean_base["input"]["reconstruction_dir"]
    )
    provenance = reconstruction_provenance(original_reconstruction)
    output_root = direct.resolve_path(clean_base["output"]["root"])

    test_view = settings.get("test_view", {})
    if not isinstance(test_view, Mapping):
        raise ValueError("two_stage.test_view must be a mapping")
    front_camera = calibrated_front_camera(
        Path(provenance["input_dir"]), test_view
    )
    common_patch = copy.deepcopy(settings["common"].get("config", {}))

    def build_one(stage_name: str) -> Dict[str, Any]:
        stage_settings = settings[stage_name]
        if not isinstance(stage_settings, Mapping):
            raise ValueError(f"two_stage.{stage_name} must be a mapping")
        config = copy.deepcopy(clean_base)
        deep_merge(config, common_patch)
        patch = stage_settings.get("config", {})
        if not isinstance(patch, Mapping):
            raise ValueError(f"two_stage.{stage_name}.config must be a mapping")
        deep_merge(config, patch)
        pose_bank = stage_settings.get("pose_bank")
        if not isinstance(pose_bank, Mapping):
            raise ValueError(f"two_stage.{stage_name}.pose_bank must be a mapping")
        config["teacher"]["two_stage_pose_bank"] = copy.deepcopy(
            dict(pose_bank)
        )
        # Both stages must render the same paired assets/test animation from
        # the calibrated front camera after their respective fits.
        config["data"]["test_expression_path"] = "assets/test/exp.npy"
        config["data"]["test_pose_path"] = "assets/test/pose.npy"
        config["data"]["test_camera_frame_index"] = int(
            front_camera["frame_index"]
        )
        config["test_render"]["enabled"] = True
        config["output"]["root"] = str(output_root)
        config["output"]["name"] = stage_output_name(run_name, stage_settings)
        # Never let the two stages accidentally share a user-supplied cache.
        config["output"]["target_cache"] = None
        config["input_dir"] = provenance["input_dir"]
        if "model" in provenance:
            config["model"] = copy.deepcopy(provenance["model"])
        scale_learning_rates(
            config, float(stage_settings.get("learning_rate_scale", 1.0))
        )
        config["two_stage_runtime"] = {
            "stage": stage_name,
            "run_name": run_name.as_posix(),
            "source_reconstruction": str(original_reconstruction),
        }
        return config

    stage1 = build_one("stage1")
    stage1_directory = direct.output_directory(stage1)

    stage2 = build_one("stage2")
    stage2["input"]["reconstruction_dir"] = str(stage1_directory)
    stage1_pose_bank = stage1["teacher"].get("two_stage_pose_bank")
    stage2_pose_bank = stage2["teacher"].get("two_stage_pose_bank")

    run_root = output_root / run_name
    plan = {
        "run_root": str(run_root),
        "source_reconstruction": str(original_reconstruction),
        "pose_banks": {
            "stage1": copy.deepcopy(stage1_pose_bank),
            "stage2": copy.deepcopy(stage2_pose_bank),
        },
        "test_render": {
            "expression_path": str(
                direct.resolve_path(stage1["data"]["test_expression_path"])
            ),
            "pose_path": str(
                direct.resolve_path(stage1["data"]["test_pose_path"])
            ),
            "front_camera": front_camera,
            "stage1_video": str(stage1_directory / "test_render" / "test.mp4"),
            "stage2_video": str(
                direct.output_directory(stage2) / "test_render" / "test.mp4"
            ),
        },
        "stage1": {
            "directory": str(stage1_directory),
            "views": list(stage1["teacher"]["source_azimuths"]),
            "expected_target_count": int(stage1_pose_bank["count"])
            * len(stage1["teacher"]["source_azimuths"]),
            "iterations": int(stage1["optimization"]["geometry_iterations"])
            + int(stage1["optimization"]["appearance_iterations"]),
        },
        "stage2": {
            "directory": str(direct.output_directory(stage2)),
            "input_reconstruction": str(stage1_directory),
            "views": list(stage2["teacher"]["source_azimuths"]),
            "expected_target_count": int(stage2_pose_bank["count"])
            * len(stage2["teacher"]["source_azimuths"]),
            "iterations": int(stage2["optimization"]["geometry_iterations"])
            + int(stage2["optimization"]["appearance_iterations"]),
        },
        "policy": {
            "targets_are_offline_and_immutable": True,
            "stage2_expands_pose_expression_bank": (
                stage2_pose_bank != stage1_pose_bank
            ),
            "global_rv51_disabled": not bool(
                stage1["teacher"].get("rv51", {}).get("enabled", False)
            ),
            "mouth_candidates_per_view": int(
                stage1["teacher"]["mouth_candidates_per_view"]
            ),
            "candidate_seed_mode": str(stage1["teacher"]["seed_mode"]),
            "target_qa_enforced": bool(
                stage1["teacher"]["quality"].get("enforce", True)
            ),
            "max_teeth_component_span": float(
                stage1["teacher"]["quality"][
                    "max_teeth_component_span"
                ]
            ),
            "require_open_mouth": bool(settings["require_open_mouth"]),
            "render_assets_test_after_each_stage": True,
        },
    }
    return stage1, stage2, plan


def normalized_feature_block(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    center = np.median(value, axis=0, keepdims=True)
    scale = np.std(value, axis=0, keepdims=True)
    scale[scale < 1.0e-6] = 1.0
    normalized = (value - center) / scale
    return normalized / max(float(np.sqrt(value.shape[1])), 1.0)


def farthest_point_indices(
    features: np.ndarray, jaw_x: np.ndarray, count: int
) -> List[int]:
    """Deterministically choose diverse expressions, seeded by widest jaw."""

    if features.shape[0] == 0:
        return []
    count = min(max(int(count), 1), features.shape[0])
    selected = [int(np.argmax(jaw_x))]
    minimum_distance = np.full(features.shape[0], np.inf, dtype=np.float64)
    while len(selected) < count:
        latest = features[selected[-1]]
        distance = np.square(features - latest[None]).sum(axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[np.asarray(selected, dtype=np.int64)] = -1.0
        selected.append(int(np.argmax(minimum_distance)))
    return selected


def validation_open_mouth_pose_bank(
    assets: Any, bank_config: Mapping[str, Any]
) -> List[Tuple[str, Any]]:
    """Build a small, diverse bank from paired open-mouth validation arrays."""

    if _ACTIVE_TWO_STAGE_CONFIG is None:
        raise RuntimeError("The two-stage pose-bank context is not active")
    data = _ACTIVE_TWO_STAGE_CONFIG["data"]
    expression_path = direct.resolve_path(data["validation_expression_path"])
    pose_path = direct.resolve_path(data["validation_pose_path"])
    expression = np.asarray(
        np.load(expression_path, allow_pickle=False), dtype=np.float32
    )
    pose = np.asarray(np.load(pose_path, allow_pickle=False), dtype=np.float32)
    if expression.ndim != 2 or expression.shape[1] < 100:
        raise ValueError(
            f"{expression_path}: expected shape (T, >=100), got {expression.shape}"
        )
    if pose.ndim != 2 or pose.shape[1] < 15:
        raise ValueError(f"{pose_path}: expected shape (T, >=15), got {pose.shape}")
    frame_count = min(expression.shape[0], pose.shape[0])
    expression = expression[:frame_count, :100]
    pose = pose[:frame_count, :15]
    finite = np.isfinite(expression).all(axis=1) & np.isfinite(pose).all(axis=1)
    jaw_x = pose[:, 6]
    quantile = float(bank_config.get("min_jaw_quantile", 0.80))
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("two_stage_pose_bank.min_jaw_quantile must be in [0, 1]")
    valid_jaw = jaw_x[finite]
    if valid_jaw.size == 0:
        raise RuntimeError("The validation open-mouth pose bank has no finite rows")
    threshold = max(
        float(np.quantile(valid_jaw, quantile)),
        float(assets.open_mouth_threshold),
    )
    candidates = np.flatnonzero(finite & (jaw_x >= threshold))
    if candidates.size == 0:
        raise RuntimeError(
            "No validation pose passes the open-mouth threshold "
            f"(jaw-x >= {threshold:.6f})"
        )

    expression_features = normalized_feature_block(expression[candidates])
    jaw_eye_features = normalized_feature_block(pose[candidates, 6:15])
    features = np.concatenate(
        [expression_features, 2.0 * jaw_eye_features], axis=1
    )
    local = farthest_point_indices(
        features,
        jaw_x[candidates],
        int(bank_config.get("count", 6)),
    )
    selected = [int(candidates[index]) for index in local]

    bank: List[Tuple[str, Any]] = []
    for index in selected:
        item = SimpleNamespace(
            expression=expression[index : index + 1, :100].copy(),
            jaw_pose=pose[index : index + 1, 6:9].copy(),
            leye_pose=pose[index : index + 1, 9:12].copy(),
            reye_pose=pose[index : index + 1, 12:15].copy(),
            source_index=index,
            is_open_mouth=bool(jaw_x[index] >= assets.open_mouth_threshold),
            is_reference=False,
        )
        bank.append((f"open_{index:06d}", item))
    return bank


def chemistry_open_diverse_pose_bank(
    assets: Any, bank_config: Mapping[str, Any]
) -> List[Tuple[str, Any]]:
    """Select diverse dynamic expressions/poses from valid chemistry frames."""

    expression = np.asarray(assets.chemistry_expression, dtype=np.float32)
    jaw = np.asarray(assets.chemistry_jaw, dtype=np.float32)
    leye = np.asarray(assets.chemistry_leye, dtype=np.float32)
    reye = np.asarray(assets.chemistry_reye, dtype=np.float32)
    source_indices = np.asarray(assets.chemistry_source_indices, dtype=np.int64)
    is_open = np.asarray(assets.chemistry_is_open, dtype=bool)
    frame_count = expression.shape[0]
    if not all(
        value.shape[0] == frame_count
        for value in (jaw, leye, reye, source_indices, is_open)
    ):
        raise ValueError("Chemistry pose/expression arrays have inconsistent lengths")
    finite = (
        np.isfinite(expression).all(axis=1)
        & np.isfinite(jaw).all(axis=1)
        & np.isfinite(leye).all(axis=1)
        & np.isfinite(reye).all(axis=1)
    )
    open_indices = np.flatnonzero(finite & is_open)
    if open_indices.size == 0:
        raise RuntimeError("The chemistry bank contains no valid open-mouth poses")

    quantile = float(bank_config.get("min_jaw_quantile", 0.0))
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("two_stage_pose_bank.min_jaw_quantile must be in [0, 1]")
    jaw_x = jaw[:, 0]
    threshold = max(
        float(np.quantile(jaw_x[open_indices], quantile)),
        float(assets.open_mouth_threshold),
    )
    candidates = open_indices[jaw_x[open_indices] >= threshold]
    if candidates.size == 0:
        raise RuntimeError(
            "No chemistry pose passes the requested open-mouth jaw threshold "
            f"(jaw-x >= {threshold:.6f})"
        )

    expression_features = normalized_feature_block(expression[candidates])
    pose_features = normalized_feature_block(
        np.concatenate(
            [jaw[candidates], leye[candidates], reye[candidates]], axis=1
        )
    )
    features = np.concatenate(
        [expression_features, 2.0 * pose_features], axis=1
    )
    local = farthest_point_indices(
        features,
        jaw_x[candidates],
        int(bank_config.get("count", 12)),
    )
    selected = [int(candidates[index]) for index in local]

    bank: List[Tuple[str, Any]] = []
    for local_index in selected:
        source_index = int(source_indices[local_index])
        item = SimpleNamespace(
            expression=expression[local_index : local_index + 1, :100].copy(),
            jaw_pose=jaw[local_index : local_index + 1, :3].copy(),
            leye_pose=leye[local_index : local_index + 1, :3].copy(),
            reye_pose=reye[local_index : local_index + 1, :3].copy(),
            source_index=source_index,
            is_open_mouth=bool(is_open[local_index]),
            is_reference=False,
        )
        bank.append((f"chem_open_{source_index:06d}", item))
    return bank


def two_stage_pose_bank(
    assets: Any, teacher_config: Mapping[str, Any]
) -> List[Tuple[str, Any]]:
    bank_config = teacher_config.get("two_stage_pose_bank")
    if not isinstance(bank_config, Mapping):
        return _ORIGINAL_JAW_POSE_BANK(assets, teacher_config)
    source = str(bank_config.get("source", "validation_open_mouth")).lower()
    if source == "validation_open_mouth":
        return validation_open_mouth_pose_bank(assets, bank_config)
    if source == "chemistry_open_diverse":
        return chemistry_open_diverse_pose_bank(assets, bank_config)
    if source == "expanded_open_mouth":
        count = max(int(bank_config.get("count", 12)), 1)
        anchor_count = int(bank_config.get("validation_anchor_count", 3))
        anchor_count = min(max(anchor_count, 0), count)
        anchors: List[Tuple[str, Any]] = []
        if anchor_count:
            anchors = validation_open_mouth_pose_bank(
                assets,
                {
                    "count": anchor_count,
                    "min_jaw_quantile": float(
                        bank_config.get(
                            "validation_min_jaw_quantile", 0.80
                        )
                    ),
                },
            )
        chemistry_count = count - len(anchors)
        chemistry: List[Tuple[str, Any]] = []
        if chemistry_count:
            chemistry = chemistry_open_diverse_pose_bank(
                assets,
                {
                    "count": chemistry_count,
                    "min_jaw_quantile": float(
                        bank_config.get(
                            "chemistry_min_jaw_quantile", 0.10
                        )
                    ),
                },
            )
        return anchors + chemistry
    raise ValueError(
        "two_stage_pose_bank.source must be 'validation_open_mouth' or "
        "'chemistry_open_diverse' or 'expanded_open_mouth'"
    )


@contextmanager
def use_two_stage_pose_bank(config: Mapping[str, Any]) -> Iterator[None]:
    """Route the target generator to the configured deterministic pose bank."""

    global _ACTIVE_TWO_STAGE_CONFIG
    previous_config = _ACTIVE_TWO_STAGE_CONFIG
    original = direct.jaw_pose_bank
    _ACTIVE_TWO_STAGE_CONFIG = config
    direct.jaw_pose_bank = two_stage_pose_bank
    try:
        yield
    finally:
        direct.jaw_pose_bank = original
        _ACTIVE_TWO_STAGE_CONFIG = previous_config


def verify_pose_bank(
    assets: Any,
    config: Mapping[str, Any],
    require_open_mouth: bool,
) -> None:
    poses = two_stage_pose_bank(assets, config["teacher"])
    if not poses:
        raise RuntimeError("The configured two-stage pose bank is empty")
    invalid = [
        label
        for label, pose in poses
        if bool(pose.is_reference)
        or (require_open_mouth and not bool(pose.is_open_mouth))
    ]
    if invalid:
        raise RuntimeError(
            "Two-stage target bank contains non-open/reference poses: "
            + ", ".join(invalid)
        )
    jaw_values = [
        float(np.asarray(pose.jaw_pose).reshape(-1)[0]) for _, pose in poses
    ]
    print(
        "[pose-bank] "
        f"{len(poses)} open-mouth poses; jaw-x "
        f"{min(jaw_values):.5f}..{max(jaw_values):.5f}; "
        f"labels={[label for label, _ in poses]}"
    )


def verify_manifest(path: Path, require_open_mouth: bool) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    entries = manifest.get("entries", [])
    if not entries:
        raise RuntimeError(f"{path}: target manifest has no observations")
    invalid = [
        str(entry.get("key", "<unknown>"))
        for entry in entries
        if bool(entry.get("pose", {}).get("is_reference", False))
        or (
            require_open_mouth
            and not bool(entry.get("pose", {}).get("is_open_mouth", False))
        )
    ]
    if invalid:
        raise RuntimeError(
            f"{path}: non-open/reference observations are forbidden: "
            + ", ".join(invalid)
        )
    pose_ids = {str(entry["pose_id"]) for entry in entries}
    camera_ids = {int(entry["camera_index"]) for entry in entries}
    expected = len(pose_ids) * len(camera_ids)
    if len(entries) != expected:
        raise RuntimeError(
            f"{path}: incomplete Cartesian pose/view bank: "
            f"{len(entries)} observations, expected {expected}"
        )
    return {
        "poses": len(pose_ids),
        "views": len(camera_ids),
        "observations": len(entries),
    }


def generate_targets(
    config: Mapping[str, Any],
    directory: Path,
    overwrite: bool,
    require_open_mouth: bool,
) -> Path:
    with use_two_stage_pose_bank(config):
        generator = direct.MouthTargetGenerator(config, directory, overwrite)
        try:
            verify_pose_bank(generator.assets, config, require_open_mouth)
            manifest = generator.generate()
        finally:
            del generator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary = verify_manifest(manifest, require_open_mouth)
    print(
        f"[targets] {directory.name}: "
        f"{summary['poses']} poses x {summary['views']} views "
        f"= {summary['observations']} fixed observations"
    )
    return manifest


def train_stage(
    config: Mapping[str, Any],
    directory: Path,
    resume: Path | None,
    require_open_mouth: bool,
) -> None:
    manifest = direct.target_directory(config, directory) / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Missing {manifest}; generate this stage's fixed targets first"
        )
    verify_manifest(manifest, require_open_mouth)
    trainer = direct.DirectInpaintTrainer(config, directory, resume)
    try:
        trainer.train()
    finally:
        trainer.close()
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_stage(
    label: str,
    config: Dict[str, Any],
    generate: bool,
    train: bool,
    overwrite: bool,
    resume: Path | None,
    require_open_mouth: bool,
) -> None:
    directory = direct.output_directory(config)
    direct.validate_config(config, check_files=True)
    directory.mkdir(parents=True, exist_ok=True)
    direct.write_resolved_config(config, directory)
    direct.seed_everything(int(config.get("seed", 0)))
    print(f"\n=== {label}: {directory} ===")
    if generate:
        generate_targets(
            config,
            directory,
            overwrite=overwrite,
            require_open_mouth=require_open_mouth,
        )
    if train:
        train_stage(
            config,
            directory,
            resume=resume,
            require_open_mouth=require_open_mouth,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/direct_loop_inpaint_targets.yaml"),
        help="Base direct-inpaint YAML; two_stage.* overrides are optional",
    )
    parser.add_argument(
        "--mode",
        choices=(
            "all",
            "stage1",
            "stage1-targets",
            "stage1-train",
            "stage2",
            "stage2-targets",
            "stage2-train",
        ),
        default="all",
    )
    parser.add_argument(
        "--overwrite-targets",
        action="store_true",
        help="Regenerate selected stage target caches",
    )
    parser.add_argument("--resume-stage1", type=Path, default=None)
    parser.add_argument("--resume-stage2", type=Path, default=None)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the plan without loading CUDA or writing outputs",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Dotted YAML overrides, for example "
            "two_stage.stage2.pose_bank.count=16"
        ),
    )
    return parser.parse_args()


def selected_actions(mode: str) -> Tuple[Tuple[bool, bool], Tuple[bool, bool]]:
    actions = {
        "all": ((True, True), (True, True)),
        "stage1": ((True, True), (False, False)),
        "stage1-targets": ((True, False), (False, False)),
        "stage1-train": ((False, True), (False, False)),
        "stage2": ((False, False), (True, True)),
        "stage2-targets": ((False, False), (True, False)),
        "stage2-train": ((False, False), (False, True)),
    }
    return actions[mode]


def main() -> None:
    args = parse_args()
    config_path = direct.resolve_path(args.config)
    base_config = direct.load_config(config_path, args.overrides)
    stage1, stage2, plan = build_stage_configs(base_config)
    supplied_two_stage = base_config.get("two_stage")
    if not isinstance(supplied_two_stage, Mapping):
        supplied_two_stage = {}
    require_open_mouth = bool(
        supplied_two_stage.get(
            "require_open_mouth",
            DEFAULT_TWO_STAGE["require_open_mouth"],
        )
    )

    stage1_directory = direct.output_directory(stage1)
    stage2_directory = direct.output_directory(stage2)
    stage1_model_exists = (
        stage1_directory / "model" / "uvd.ply"
    ).is_file() and (
        stage1_directory / "model" / "reconstruction_params.npz"
    ).is_file()

    direct.validate_config(stage1, check_files=True)
    direct.validate_config(stage2, check_files=stage1_model_exists)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.validate_only:
        if not stage1_model_exists:
            print(
                "[validate] Stage 2 file checks are deferred until "
                f"Stage 1 creates {stage1_directory / 'model'}"
            )
        print("[validate] two-stage configuration is valid")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for rendering, diffusion, and fitting")

    run_root = Path(plan["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "two_stage_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    stage1_actions, stage2_actions = selected_actions(args.mode)
    resume_stage1 = (
        direct.resolve_path(args.resume_stage1) if args.resume_stage1 else None
    )
    resume_stage2 = (
        direct.resolve_path(args.resume_stage2) if args.resume_stage2 else None
    )

    if any(stage1_actions):
        run_stage(
            "Stage 1 / open-mouth frontal fit",
            stage1,
            generate=stage1_actions[0],
            train=stage1_actions[1],
            overwrite=bool(args.overwrite_targets),
            resume=resume_stage1,
            require_open_mouth=require_open_mouth,
        )

    if any(stage2_actions):
        # In --mode all, this check happens only after Stage 1 has saved its
        # model.  In stage2-only modes it gives a precise prerequisite error.
        if not (
            stage1_directory / "model" / "uvd.ply"
        ).is_file():
            raise FileNotFoundError(
                "Stage 2 requires the completed Stage-1 model: "
                f"{stage1_directory / 'model' / 'uvd.ply'}"
            )
        run_stage(
            "Stage 2 / controlled multi-view consolidation",
            stage2,
            generate=stage2_actions[0],
            train=stage2_actions[1],
            overwrite=bool(args.overwrite_targets),
            resume=resume_stage2,
            require_open_mouth=require_open_mouth,
        )

    completed_directory = (
        stage2_directory if any(stage2_actions) else stage1_directory
    )
    print(f"\nRequested two-stage work complete: {completed_directory}")


if __name__ == "__main__":
    main()
