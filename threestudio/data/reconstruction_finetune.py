"""Calibrated stage-two data for reconstruction-initialized head fine-tuning.

The first reconstruction stage optimizes in the FaceLift world coordinate
system.  Consequently this module deliberately keeps every calibrated camera
in its original OpenCV convention:

    x_camera = w2c @ x_facelift,  pixel ~ K @ x_camera

View-dependent prompt angles are a separate quantity.  They are computed after
mapping the camera centre through ``inv(facelift_from_training)`` so that the
prompt processor sees the original training-FLAME convention (front is +Y,
approximately 90 degrees).

FLAME animation parameters normally come from ``chemistry_exp.npy``.  The
mouth stage can instead sample the paired ``open_mouth_exp.npy`` / ``pose.npy``
sequence shipped with AnimPortrait3D, matching GSAvatar's training driver.
Shape is fixed to the reconstruction shape, and neck/global/translation motion
is always zero.  No neck parameter is exposed in a returned batch.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from flame_model.flame import (
    FLAME_MEDIAPIPE_468_PATH,
    FLAME_MEDIAPIPE_LMK_PATH,
    FlameHead,
)
from flame_model.flame_teeth import FlameHead as Stage1FlameHead
from threestudio import register
from threestudio.utils.config import parse_structured
from threestudio.utils.mediapipe_utils import draw_landmarks_468
from threestudio.utils.mediapipe_utils_v2 import draw_landmarks_105
from threestudio.utils.typing import DictConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ReconstructionFinetuneDataModuleConfig:
    """Configuration for calibrated reconstruction fine-tuning."""

    reconstruction_dir: str = "outputs/reconstruction/00000001"
    chemistry_path: str = "talkshow/collection/chemistry_exp.npy"

    height: int = 512
    width: int = 512
    batch_size: int = 4
    # Opt-in joint SDEdit batches retain a legacy-distributed anchor in row 0.
    surface_consistent_batch: bool = False
    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    n_val_views: int = 4
    num_workers: int = 0
    test_expression_path: str = "assets/test/exp.npy"
    test_pose_path: str = "assets/test/pose.npy"
    test_camera_frame_index: int = 21
    validation_expression_path: str = "assets/open_mouth_exp.npy"
    validation_pose_path: str = "assets/open_mouth_pose.npy"
    eval_open_mouth_index: int = -1  # < 0 selects the largest jaw-opening axis

    # A batch is sampled from one calibrated elevation ring.  The selected
    # views are evenly spaced in azimuth with a random cyclic phase.
    camera_elevation_tolerance: float = 1.0e-4

    # AnimPortrait3D first performs a mouth-only pass with open-mouth poses and
    # near-frontal cameras, then releases the complete avatar.  These options
    # make the same calibrated data module usable for both passes.  Angles are
    # expressed in the inverse-aligned training-FLAME convention used by the
    # prompt processor (front is approximately 90 degrees).  A wrapped
    # azimuth interval such as [300, 40] is supported; null keeps all cameras.
    # "animportrait3d_mouth" uniformly samples paired rows from the validation
    # expression/pose paths (the local copy of GSAvatar's 909-frame sequence).
    train_pose_mode: str = "mixed"
    # ``calibrated`` keeps the Stage-1 discrete cameras.  The continuous mode
    # keeps that exact FaceLift orbit/intrinsics and only samples its azimuth
    # and elevation continuously.
    train_camera_sampling: str = "calibrated"
    train_azimuth_range: Optional[List[float]] = None
    train_elevation_range: Optional[List[float]] = None
    # train_all.py draws 70% of cameras from yaw [0, 180] (the face
    # hemisphere) and 30% from [180, 360].  Calibrated camera rings are not
    # uniformly distributed between those halves, so uniform row sampling
    # does not reproduce that distribution.
    train_front_hemisphere_probability: Optional[float] = None

    # Dynamic-expression ablations.
    use_dynamic_expression: bool = True
    # Dynamic training samples are chemistry-only. Static identity supervision
    # is provided separately by the system's reference replay.
    reference_pose_probability: float = 0.0
    eval_pose_mode: str = "reference"  # "reference" or "chemistry"
    eval_chemistry_index: int = 0

    # Chemistry cleaning and open-mouth stratification.
    chemistry_jaw_outlier_quantile: float = 0.995
    chemistry_jaw_max_norm: float = 0.0  # <= 0 disables the absolute cutoff
    # Rare failed fits also occur in expression and eye coefficients.  These
    # independent absolute cutoffs are intentionally configurable so their
    # contribution can be ablated without changing the chemistry source.
    chemistry_expression_max_norm: float = 30.0  # <= 0 disables
    chemistry_eye_max_norm: float = 1.0  # <= 0 disables; applies to each eye
    chemistry_open_mouth_quantile: float = 0.80
    chemistry_open_mouth_oversample: bool = True
    chemistry_open_mouth_prob: float = 0.40

    # ControlNet condition. ``mediapipe`` preserves the previous landmark
    # control, while ``animportrait3d_normal_seg`` reproduces AnimPortrait3D's
    # four-channel face-normal + semantic-segmentation input.  ``use_condition``
    # supersedes the legacy, misleadingly named ``use_mediapipe_condition``;
    # leaving it null preserves existing configs.
    condition_type: str = "mediapipe"
    use_condition: Optional[bool] = None
    use_mediapipe_condition: bool = True
    condition_device: str = "cuda"
    projection_min_depth: float = 1.0e-5
    animportrait3d_assets_dir: str = (
        "ckpts/animportrait3d_normal_seg_flame_teeth"
    )


@dataclass(frozen=True)
class _CameraFrame:
    frame_index: int
    image_path: Path
    width: int
    height: int
    K: np.ndarray
    c2w: np.ndarray
    w2c: np.ndarray
    source_azimuth_deg: float
    source_elevation_deg: float
    prompt_camera_position: np.ndarray
    prompt_azimuth_deg: float
    prompt_elevation_deg: float
    prompt_distance: float


@dataclass(frozen=True)
class _Pose:
    expression: np.ndarray
    jaw_pose: np.ndarray
    leye_pose: np.ndarray
    reye_pose: np.ndarray
    source_index: int
    is_open_mouth: bool
    is_reference: bool


def _normalized(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1.0e-10:
        raise ValueError(f"Cannot normalize degenerate {name}")
    return vector / norm


def _calibrated_orbit_camera(
    azimuth_deg: float,
    elevation_deg: float,
    radius: float,
    pivot: Sequence[float],
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a continuous camera on the calibrated FaceLift orbit.

    FaceLift's camera world is z-up and uses OpenCV camera axes: the c2w
    columns are right, down, and forward.  At angles already present in
    ``cameras.json`` this construction reproduces the Stage-1 cameras; only
    the two source angles become continuous during refinement.
    """

    values = (azimuth_deg, elevation_deg, radius)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Continuous camera parameters must be finite")
    if radius <= 0.0:
        raise ValueError("Calibrated camera radius must be positive")
    pivot_array = np.asarray(pivot, dtype=np.float64).reshape(-1)
    if pivot_array.size != 3 or not np.isfinite(pivot_array).all():
        raise ValueError("Calibrated camera pivot must contain three finite values")
    K = np.asarray(intrinsics, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("Calibrated camera intrinsics must be a finite 3x3 matrix")
    if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        raise ValueError("Calibrated camera focal lengths must be positive")

    azimuth = math.radians(float(azimuth_deg))
    elevation = math.radians(float(elevation_deg))
    camera_offset = radius * np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=np.float64,
    )
    origin = pivot_array + camera_offset
    forward = _normalized(pivot_array - origin, "camera forward vector")
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = _normalized(np.cross(forward, world_up), "camera right vector")
    camera_down = _normalized(
        np.cross(forward, right), "camera down vector"
    )
    rotation = np.stack((right, camera_down, forward), axis=1)
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=2.0e-6
    ) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=2.0e-6
    ):
        raise ValueError("Continuous calibrated camera is not a proper rotation")

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = origin
    w2c = np.linalg.inv(c2w)
    return c2w.astype(np.float32), w2c.astype(np.float32)


def _fit_calibrated_orbit(c2ws: Sequence[np.ndarray]) -> tuple[np.ndarray, float]:
    """Fit the common look-at pivot and radius of calibrated cameras."""

    matrices = np.asarray(c2ws, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError("Calibrated camera poses must have shape (N, 4, 4)")
    if matrices.shape[0] < 2 or not np.isfinite(matrices).all():
        raise ValueError("At least two finite calibrated camera poses are required")
    origins = matrices[:, :3, 3]
    forwards = matrices[:, :3, 2].copy()
    forward_norms = np.linalg.norm(forwards, axis=1, keepdims=True)
    if not np.isfinite(forward_norms).all() or bool(
        (forward_norms <= 1.0e-10).any()
    ):
        raise ValueError("Calibrated camera has a degenerate forward axis")
    forwards /= forward_norms
    projectors = np.eye(3, dtype=np.float64)[None] - (
        forwards[:, :, None] * forwards[:, None, :]
    )
    normal = projectors.sum(axis=0)
    normal_condition = float(np.linalg.cond(normal))
    if not math.isfinite(normal_condition) or normal_condition > 1.0e8:
        raise ValueError("Calibrated camera optical axes do not define a stable orbit")
    pivot = np.linalg.solve(normal, np.einsum("nij,nj->i", projectors, origins))
    radii = np.linalg.norm(origins - pivot[None], axis=1)
    radius = float(np.median(radii))
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("Fitted calibrated camera radius is invalid")
    max_radius_error = float(np.max(np.abs(radii - radius)))
    if max_radius_error > max(1.0e-4, radius * 1.0e-4):
        raise ValueError(
            "Calibrated cameras are not on one spherical orbit: maximum "
            f"radius error is {max_radius_error:.6g}"
        )
    return pivot.astype(np.float32), radius


def _resolve_path(value: str, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _require_finite(name: str, value: np.ndarray, source: Path) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{source}: '{name}' contains NaN or infinity")


def _as_parameter(
    archive: Any,
    name: str,
    width: int,
    source: Path,
) -> np.ndarray:
    if name not in archive:
        raise KeyError(f"{source}: missing required array '{name}'")
    value = np.asarray(archive[name], dtype=np.float32).reshape(-1)
    if value.size < width:
        raise ValueError(
            f"{source}: '{name}' has {value.size} values; expected at least {width}"
        )
    value = value[:width][None].copy()
    _require_finite(name, value, source)
    return value


class _ReconstructionAssets:
    """Validated CPU-side reconstruction, camera, image, and pose assets."""

    def __init__(self, cfg: ReconstructionFinetuneDataModuleConfig) -> None:
        self.cfg = cfg
        self.reconstruction_dir = _resolve_path(str(cfg.reconstruction_dir))
        if not self.reconstruction_dir.is_dir():
            raise FileNotFoundError(
                "reconstruction_dir does not exist or is not a directory: "
                f"{self.reconstruction_dir}"
            )

        resolved_config_path = self.reconstruction_dir / "resolved_config.yaml"
        if not resolved_config_path.is_file():
            raise FileNotFoundError(
                "Stage-two data requires the first-stage resolved config: "
                f"{resolved_config_path}"
            )
        with resolved_config_path.open("r", encoding="utf-8") as file:
            resolved_config = yaml.safe_load(file)
        if not isinstance(resolved_config, dict):
            raise ValueError(
                f"{resolved_config_path}: expected a YAML mapping at the top level"
            )
        raw_input_dir = resolved_config.get("input_dir")
        if not isinstance(raw_input_dir, str) or not raw_input_dir.strip():
            raise KeyError(
                f"{resolved_config_path}: missing non-empty first-stage 'input_dir'"
            )
        self.input_dir = self._resolve_input_dir(raw_input_dir, resolved_config_path)

        self.camera_path = self.input_dir / "cameras.json"
        self.parameter_path = (
            self.reconstruction_dir / "model" / "reconstruction_params.npz"
        )
        if not self.camera_path.is_file():
            raise FileNotFoundError(
                "Cannot find the FaceLift camera calibration inferred from "
                f"{resolved_config_path}: {self.camera_path}"
            )
        if not self.parameter_path.is_file():
            raise FileNotFoundError(
                "Cannot find first-stage reconstruction parameters: "
                f"{self.parameter_path}"
            )

        self._load_reconstruction_parameters(resolved_config)
        self.frames = self._load_cameras()
        self.elevation_groups = self._group_camera_elevations()
        self._validate_requested_resolutions()
        self._load_test_sequence()
        self._load_validation_sequence()
        self._rgb_cache: Dict[int, torch.Tensor] = {}
        self._alpha_cache: Dict[int, torch.Tensor] = {}
        self.reference_alpha_paths = self._discover_reference_alpha()

        self.chemistry_path = self._resolve_chemistry_path(str(cfg.chemistry_path))
        (
            self.chemistry_expression,
            self.chemistry_jaw,
            self.chemistry_leye,
            self.chemistry_reye,
            self.chemistry_source_indices,
            self.chemistry_is_open,
            self.open_mouth_threshold,
            chemistry_stats,
        ) = self._load_chemistry()
        self.chemistry_open_indices = np.flatnonzero(
            self.chemistry_is_open
        ).astype(np.int64, copy=False)

        self.reference_pose = _Pose(
            expression=self.reference_expression,
            jaw_pose=self.reference_pose_vector[:, 3:6],
            leye_pose=self.reference_eyes[:, :3],
            reye_pose=self.reference_eyes[:, 3:6],
            source_index=-1,
            is_open_mouth=bool(
                float(self.reference_pose_vector[0, 3])
                >= float(self.open_mouth_threshold)
            ),
            is_reference=True,
        )
        self.validation_pose = _Pose(
            expression=self.validation_expression,
            jaw_pose=self.validation_jaw,
            leye_pose=self.validation_leye,
            reye_pose=self.validation_reye,
            source_index=self.validation_source_index,
            is_open_mouth=bool(
                float(self.validation_jaw[0, 0])
                >= float(self.open_mouth_threshold)
            ),
            is_reference=False,
        )
        threestudio.info(
            "Reconstruction fine-tune data: "
            f"{len(self.frames)} calibrated views in "
            f"{len(self.elevation_groups)} elevation rings; "
            f"{chemistry_stats['kept']}/{chemistry_stats['total']} valid "
            "chemistry frames "
            f"({chemistry_stats['open']} open-mouth, threshold "
            f"{self.open_mouth_threshold:.5f}); "
            f"{self.test_frame_count} test animation frames at calibrated "
            f"camera frame {self.frames[self.test_camera_index].frame_index}; "
            f"validation uses open-mouth frame "
            f"{self.validation_source_index} "
            f"(jaw-x={self.validation_jaw[0, 0]:.5f})."
        )
        reference_probability = float(self.cfg.reference_pose_probability)
        if not 0.0 <= reference_probability <= 1.0:
            raise ValueError("reference_pose_probability must be in [0, 1]")
        front_probability = self.cfg.train_front_hemisphere_probability
        if front_probability is not None and not 0.0 <= float(
            front_probability
        ) <= 1.0:
            raise ValueError(
                "train_front_hemisphere_probability must be in [0, 1]"
            )
        self.train_camera_sampling = str(
            self.cfg.train_camera_sampling
        ).strip().lower()
        if self.train_camera_sampling not in {
            "calibrated",
            "calibrated_continuous",
        }:
            raise ValueError(
                "train_camera_sampling must be 'calibrated' or "
                "'calibrated_continuous', got "
                f"{self.cfg.train_camera_sampling!r}"
            )
        pose_mode = str(self.cfg.train_pose_mode).lower()
        if pose_mode not in {
            "mixed",
            "open_mouth",
            "animportrait3d_mouth",
            "reference",
        }:
            raise ValueError(
                "train_pose_mode must be 'mixed', 'open_mouth', "
                "'animportrait3d_mouth', or 'reference', got "
                f"{self.cfg.train_pose_mode!r}"
            )
        # Validate camera settings before the infinite iterable dataset starts
        # sampling. Continuous training is fitted to the same calibrated rig.
        if self.train_camera_sampling == "calibrated":
            self._training_camera_groups(int(self.cfg.batch_size))
        else:
            self._setup_calibrated_continuous_orbit()
        min_depth = float(self.cfg.projection_min_depth)
        if not math.isfinite(min_depth) or min_depth <= 0.0:
            raise ValueError("projection_min_depth must be finite and positive")

    def _resolve_input_dir(
        self, raw_input_dir: str, resolved_config_path: Path
    ) -> Path:
        candidate = Path(raw_input_dir).expanduser()
        if candidate.is_absolute():
            resolved = candidate.resolve()
            if resolved.is_dir():
                return resolved
            raise FileNotFoundError(
                f"{resolved_config_path}: resolved input_dir does not exist: {resolved}"
            )

        candidates = [
            (PROJECT_ROOT / candidate).resolve(),
            (resolved_config_path.parent / candidate).resolve(),
        ]
        for resolved in candidates:
            if resolved.is_dir():
                return resolved
        raise FileNotFoundError(
            f"{resolved_config_path}: cannot resolve relative input_dir "
            f"'{raw_input_dir}'. Tried: "
            + ", ".join(str(path) for path in candidates)
        )

    def _resolve_chemistry_path(self, raw_path: str) -> Path:
        path = _resolve_path(raw_path)
        if path.is_dir():
            path = path / "chemistry_exp.npy"
        if not path.is_file():
            raise FileNotFoundError(
                "chemistry_path must name chemistry_exp.npy (or its containing "
                f"directory); resolved path is {path}"
            )
        return path

    def _load_animation_sequence(
        self,
        expression_value: str,
        pose_value: str,
        label: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        expression_path = _resolve_path(expression_value)
        pose_path = _resolve_path(pose_value)
        if not expression_path.is_file():
            raise FileNotFoundError(
                f"Cannot find {label} expression sequence: {expression_path}"
            )
        if not pose_path.is_file():
            raise FileNotFoundError(
                f"Cannot find {label} pose sequence: {pose_path}"
            )

        expression = np.asarray(
            np.load(expression_path, allow_pickle=False), dtype=np.float32
        )
        pose = np.asarray(np.load(pose_path, allow_pickle=False), dtype=np.float32)
        if expression.ndim != 2 or expression.shape[1] < 100:
            raise ValueError(
                f"{expression_path}: expected shape (T, >=100), "
                f"got {expression.shape}"
            )
        if pose.ndim != 2 or pose.shape[1] < 15:
            raise ValueError(
                f"{pose_path}: expected shape (T, >=15), got {pose.shape}"
            )
        frame_count = min(expression.shape[0], pose.shape[0])
        if frame_count == 0:
            raise ValueError(
                f"{label.capitalize()} expression/pose sequences must not be empty"
            )
        if expression.shape[0] != pose.shape[0]:
            threestudio.warn(
                f"{label.capitalize()} expression/pose lengths differ "
                f"({expression.shape[0]} vs {pose.shape[0]}); "
                f"using the first {frame_count} paired frames."
            )

        expression = expression[:frame_count, :100].copy()
        # assets/test/pose.npy layout follows the first-stage driver:
        # global[0:3], neck[3:6], jaw[6:9], left/right eye[9:15].
        # Global and neck are deliberately ignored.
        jaw = pose[:frame_count, 6:9].copy()
        leye = pose[:frame_count, 9:12].copy()
        reye = pose[:frame_count, 12:15].copy()
        _require_finite("expression", expression, expression_path)
        _require_finite("jaw/eyes", pose[:frame_count, 6:15], pose_path)
        return expression, jaw, leye, reye

    def _load_test_sequence(self) -> None:
        (
            self.test_expression,
            self.test_jaw,
            self.test_leye,
            self.test_reye,
        ) = self._load_animation_sequence(
            str(self.cfg.test_expression_path),
            str(self.cfg.test_pose_path),
            "test",
        )
        self.test_frame_count = self.test_expression.shape[0]

        requested_frame = int(self.cfg.test_camera_frame_index)
        matching = [
            index
            for index, frame in enumerate(self.frames)
            if frame.frame_index == requested_frame
        ]
        if not matching:
            available = [frame.frame_index for frame in self.frames]
            raise ValueError(
                f"test_camera_frame_index={requested_frame} is absent from "
                f"cameras.json; available frame indices are {available}"
            )
        self.test_camera_index = matching[0]

    def _load_validation_sequence(self) -> None:
        expression, jaw, leye, reye = self._load_animation_sequence(
            str(self.cfg.validation_expression_path),
            str(self.cfg.validation_pose_path),
            "validation",
        )
        # Keep all paired rows for the exact AnimPortrait3D mouth sampler.
        # Validation below still uses one deterministic maximum-opening row.
        self.open_mouth_expression = expression
        self.open_mouth_jaw = jaw
        self.open_mouth_leye = leye
        self.open_mouth_reye = reye
        index = int(self.cfg.eval_open_mouth_index)
        if index < 0:
            index = int(np.argmax(jaw[:, 0]))
        if index >= expression.shape[0]:
            raise IndexError(
                f"eval_open_mouth_index={index} is outside validation "
                f"sequence length {expression.shape[0]}"
            )
        self.validation_expression = expression[index : index + 1]
        self.validation_jaw = jaw[index : index + 1]
        self.validation_leye = leye[index : index + 1]
        self.validation_reye = reye[index : index + 1]
        self.validation_source_index = index

    def _load_reconstruction_parameters(self, resolved_config: Dict[str, Any]) -> None:
        with np.load(self.parameter_path, allow_pickle=False) as archive:
            self.shape = _as_parameter(
                archive, "shape", 300, self.parameter_path
            )
            self.reference_expression = _as_parameter(
                archive, "expression", 100, self.parameter_path
            )
            self.reference_pose_vector = _as_parameter(
                archive, "pose", 6, self.parameter_path
            )
            self.reference_eyes = _as_parameter(
                archive, "eyes", 6, self.parameter_path
            )
            if "facelift_from_training" not in archive:
                raise KeyError(
                    f"{self.parameter_path}: missing 'facelift_from_training'"
                )
            self.facelift_from_training = np.asarray(
                archive["facelift_from_training"], dtype=np.float32
            ).copy()
            if self.facelift_from_training.shape != (4, 4):
                raise ValueError(
                    f"{self.parameter_path}: facelift_from_training must be 4x4, "
                    f"got {self.facelift_from_training.shape}"
                )
            _require_finite(
                "facelift_from_training",
                self.facelift_from_training,
                self.parameter_path,
            )
            if not np.allclose(
                self.facelift_from_training[3],
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                atol=1.0e-6,
            ):
                raise ValueError(
                    f"{self.parameter_path}: facelift_from_training is not affine "
                    "(last row must be [0, 0, 0, 1])"
                )
            linear_condition = float(
                np.linalg.cond(self.facelift_from_training[:3, :3])
            )
            if not np.isfinite(linear_condition) or linear_condition > 1.0e6:
                raise ValueError(
                    f"{self.parameter_path}: alignment linear part is singular or "
                    f"ill-conditioned (condition number {linear_condition:.3e})"
                )
            self.training_from_facelift = np.linalg.inv(
                self.facelift_from_training.astype(np.float64)
            )

            if "flame_scale" not in archive:
                raise KeyError(f"{self.parameter_path}: missing 'flame_scale'")
            self.flame_scale = float(np.asarray(archive["flame_scale"]).reshape(()))
            if not math.isfinite(self.flame_scale):
                raise ValueError(
                    f"{self.parameter_path}: flame_scale must be finite"
                )

        model_cfg = resolved_config.get("model", {})
        if isinstance(model_cfg, dict) and "flame_scale" in model_cfg:
            configured_scale = float(model_cfg["flame_scale"])
            if not math.isclose(
                configured_scale, self.flame_scale, rel_tol=0.0, abs_tol=1.0e-6
            ):
                raise ValueError(
                    "flame_scale differs between resolved_config.yaml "
                    f"({configured_scale}) and reconstruction_params.npz "
                    f"({self.flame_scale})"
                )

    def _resolve_image_path(self, file_path: str) -> Path:
        relative = Path(file_path)
        candidates = [
            relative if relative.is_absolute() else self.input_dir / relative,
            self.input_dir / "rgb" / relative.name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            f"{self.camera_path}: cannot resolve image '{file_path}'. Tried: "
            + ", ".join(str(path) for path in candidates)
        )

    def _load_cameras(self) -> List[_CameraFrame]:
        with self.camera_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if not isinstance(metadata, dict):
            raise ValueError(f"{self.camera_path}: expected a JSON object")
        convention = str(metadata.get("camera_convention", "")).lower()
        if convention and "opencv" not in convention:
            raise ValueError(
                f"{self.camera_path}: expected OpenCV cameras, got convention "
                f"'{metadata.get('camera_convention')}'"
            )
        raw_frames = metadata.get("frames")
        if not isinstance(raw_frames, list) or not raw_frames:
            raise ValueError(f"{self.camera_path}: 'frames' must be a non-empty list")

        frames: List[_CameraFrame] = []
        seen_indices = set()
        for fallback_index, raw in enumerate(raw_frames):
            if not isinstance(raw, dict):
                raise ValueError(
                    f"{self.camera_path}: frame {fallback_index} is not an object"
                )
            required = ("file_path", "w", "h", "fx", "fy", "cx", "cy", "c2w")
            missing = [name for name in required if name not in raw]
            if missing:
                raise KeyError(
                    f"{self.camera_path}: frame {fallback_index} is missing "
                    + ", ".join(missing)
                )

            frame_index = int(raw.get("frame_index", fallback_index))
            if frame_index in seen_indices:
                raise ValueError(
                    f"{self.camera_path}: duplicate frame_index {frame_index}"
                )
            seen_indices.add(frame_index)
            width, height = int(raw["w"]), int(raw["h"])
            fx, fy = float(raw["fx"]), float(raw["fy"])
            cx, cy = float(raw["cx"]), float(raw["cy"])
            if width <= 0 or height <= 0 or fx <= 0.0 or fy <= 0.0:
                raise ValueError(
                    f"{self.camera_path}: frame {frame_index} has invalid image "
                    f"size/focal length ({width}x{height}, fx={fx}, fy={fy})"
                )
            K = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            c2w = np.asarray(raw["c2w"], dtype=np.float64)
            if c2w.shape != (4, 4):
                raise ValueError(
                    f"{self.camera_path}: frame {frame_index} c2w must be 4x4, "
                    f"got {c2w.shape}"
                )
            w2c = np.asarray(raw.get("w2c", np.linalg.inv(c2w)), dtype=np.float64)
            if w2c.shape != (4, 4):
                raise ValueError(
                    f"{self.camera_path}: frame {frame_index} w2c must be 4x4, "
                    f"got {w2c.shape}"
                )
            _require_finite(f"frame {frame_index} K", K, self.camera_path)
            _require_finite(f"frame {frame_index} c2w", c2w, self.camera_path)
            _require_finite(f"frame {frame_index} w2c", w2c, self.camera_path)
            inverse_error = max(
                float(np.abs(c2w @ w2c - np.eye(4)).max()),
                float(np.abs(w2c @ c2w - np.eye(4)).max()),
            )
            if inverse_error > 1.0e-5:
                raise ValueError(
                    f"{self.camera_path}: frame {frame_index} c2w/w2c inverse "
                    f"error is {inverse_error:.3e} (limit 1e-5)"
                )
            for matrix_name, matrix in (("c2w", c2w), ("w2c", w2c)):
                if not np.allclose(
                    matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1.0e-6
                ):
                    raise ValueError(
                        f"{self.camera_path}: frame {frame_index} {matrix_name} "
                        "last row must be [0, 0, 0, 1]"
                    )
                determinant = float(np.linalg.det(matrix[:3, :3]))
                if abs(determinant - 1.0) > 1.0e-4:
                    raise ValueError(
                        f"{self.camera_path}: frame {frame_index} {matrix_name} "
                        f"rotation determinant is {determinant:.6f}, expected +1"
                    )

            camera_facelift = c2w[:3, 3]
            camera_h = np.concatenate([camera_facelift, np.ones(1)])
            camera_training_h = self.training_from_facelift @ camera_h
            if abs(float(camera_training_h[3])) < 1.0e-10:
                raise ValueError(
                    f"{self.camera_path}: frame {frame_index} camera centre maps "
                    "to an invalid homogeneous point under alignment inverse"
                )
            camera_training = camera_training_h[:3] / camera_training_h[3]
            distance = float(np.linalg.norm(camera_training))
            if not math.isfinite(distance) or distance <= 1.0e-8:
                raise ValueError(
                    f"{self.camera_path}: frame {frame_index} has an invalid "
                    "training-coordinate camera distance"
                )
            azimuth = math.degrees(
                math.atan2(float(camera_training[1]), float(camera_training[0]))
            )
            elevation = math.degrees(
                math.atan2(
                    float(camera_training[2]),
                    float(np.linalg.norm(camera_training[:2])),
                )
            )
            source_azimuth = float(
                raw.get("azimuth_degrees", raw.get("relative_azimuth_deg", azimuth))
            )
            source_elevation = float(
                raw.get("elevation_degrees", raw.get("elevation_deg", elevation))
            )
            if not all(
                math.isfinite(value)
                for value in (source_azimuth, source_elevation, azimuth, elevation)
            ):
                raise ValueError(
                    f"{self.camera_path}: frame {frame_index} has non-finite angles"
                )

            frames.append(
                _CameraFrame(
                    frame_index=frame_index,
                    image_path=self._resolve_image_path(str(raw["file_path"])),
                    width=width,
                    height=height,
                    K=K.astype(np.float32),
                    c2w=c2w.astype(np.float32),
                    w2c=w2c.astype(np.float32),
                    source_azimuth_deg=source_azimuth,
                    source_elevation_deg=source_elevation,
                    prompt_camera_position=camera_training.astype(np.float32),
                    prompt_azimuth_deg=azimuth,
                    prompt_elevation_deg=elevation,
                    prompt_distance=distance,
                )
            )
        return frames

    def _group_camera_elevations(self) -> List[List[int]]:
        tolerance = float(self.cfg.camera_elevation_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("camera_elevation_tolerance must be finite and positive")
        grouped: Dict[int, List[int]] = {}
        for index, frame in enumerate(self.frames):
            key = int(round(frame.source_elevation_deg / tolerance))
            grouped.setdefault(key, []).append(index)
        groups = []
        for _, indices in sorted(grouped.items()):
            indices.sort(
                key=lambda index: self.frames[index].source_azimuth_deg % 360.0
            )
            groups.append(indices)
        if not groups:
            raise RuntimeError("No calibrated elevation groups were constructed")
        return groups

    def _setup_calibrated_continuous_orbit(self) -> None:
        """Recover and verify the Stage-1 orbit used for continuous sampling."""

        reference_K = self.frames[0].K.astype(np.float64)
        for frame in self.frames[1:]:
            if not np.allclose(frame.K, reference_K, rtol=0.0, atol=1.0e-4):
                raise ValueError(
                    "calibrated_continuous sampling requires identical "
                    "intrinsics for every Stage-1 camera"
                )
        pivot, radius = _fit_calibrated_orbit(
            [frame.c2w for frame in self.frames]
        )

        maximum_pose_error = 0.0
        for frame in self.frames:
            reconstructed, _ = _calibrated_orbit_camera(
                azimuth_deg=frame.source_azimuth_deg,
                elevation_deg=frame.source_elevation_deg,
                radius=radius,
                pivot=pivot,
                intrinsics=reference_K,
            )
            maximum_pose_error = max(
                maximum_pose_error,
                float(np.max(np.abs(reconstructed - frame.c2w))),
            )
        if maximum_pose_error > 2.0e-4:
            raise ValueError(
                "The fitted continuous orbit does not reproduce cameras.json: "
                f"maximum c2w error is {maximum_pose_error:.3e}"
            )

        eligible_groups = self._training_camera_groups(int(self.cfg.batch_size))
        eligible_indices = [index for group in eligible_groups for index in group]
        source_elevations = [
            self.frames[index].source_elevation_deg for index in eligible_indices
        ]
        self.continuous_camera_intrinsics = reference_K.astype(np.float32)
        self.continuous_camera_pivot = pivot
        self.continuous_camera_radius = radius
        self.continuous_source_elevation_range = (
            float(min(source_elevations)),
            float(max(source_elevations)),
        )
        threestudio.info(
            "Continuous FaceLift cameras: fitted Stage-1 orbit "
            f"radius={radius:.6f}, pivot={pivot.tolist()}, source elevation "
            f"range={self.continuous_source_elevation_range}, "
            f"max discrete-pose error={maximum_pose_error:.3e}."
        )

    def _discover_reference_alpha(self) -> Optional[List[Path]]:
        mask_dir = self.reconstruction_dir / "final_views" / "mask"
        paths: List[Path] = []
        missing: List[Path] = []
        for frame in self.frames:
            candidates = [
                mask_dir / frame.image_path.name,
                mask_dir / f"{frame.image_path.stem}.png",
            ]
            path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if path is None:
                missing.append(candidates[0])
            else:
                paths.append(path.resolve())
        if missing:
            threestudio.warn(
                "Exact stage-one reference alpha is incomplete; "
                "'reference_alpha' will be omitted from all batches. Missing "
                f"{len(missing)}/{len(self.frames)} masks, e.g. {missing[0]}"
            )
            return None
        return paths

    def _validate_requested_resolutions(self) -> None:
        requested = {
            (int(self.cfg.width), int(self.cfg.height)),
            (int(self.cfg.eval_width), int(self.cfg.eval_height)),
        }
        source = {(frame.width, frame.height) for frame in self.frames}
        if len(source) != 1:
            raise ValueError(
                f"{self.camera_path}: all calibrated frames must share one "
                f"resolution for batched training, got {sorted(source)}"
            )
        source_resolution = next(iter(source))
        if requested != {source_resolution}:
            raise ValueError(
                "Strict first-stage intrinsics require train/eval resolution to "
                f"match cameras.json exactly. Camera resolution is "
                f"{source_resolution[0]}x{source_resolution[1]}, requested "
                f"{sorted(requested)}"
            )
        if int(self.cfg.batch_size) <= 0 or int(self.cfg.eval_batch_size) <= 0:
            raise ValueError("batch_size and eval_batch_size must be positive")
        largest_ring = max(len(group) for group in self.elevation_groups)
        if int(self.cfg.batch_size) > largest_ring:
            raise ValueError(
                f"batch_size={self.cfg.batch_size} exceeds the largest calibrated "
                f"same-elevation ring ({largest_ring} views)"
            )

    @staticmethod
    def _normalise_chemistry_entries(loaded: np.ndarray, path: Path) -> List[dict]:
        if loaded.dtype != object:
            raise ValueError(
                f"{path}: expected an object array of dictionaries, got "
                f"dtype={loaded.dtype}, shape={loaded.shape}"
            )
        entries: List[dict] = []
        for index, raw in enumerate(loaded.reshape(-1)):
            if isinstance(raw, np.ndarray) and raw.shape == ():
                raw = raw.item()
            if not isinstance(raw, dict):
                raise ValueError(
                    f"{path}: entry {index} is {type(raw).__name__}, expected dict"
                )
            entries.append(raw)
        if not entries:
            raise ValueError(f"{path}: chemistry array is empty")
        return entries

    def _load_chemistry(self):
        loaded = np.load(self.chemistry_path, allow_pickle=True)
        entries = self._normalise_chemistry_entries(loaded, self.chemistry_path)
        required_widths = {
            "expression": 100,
            "jaw_pose": 3,
            "leye_pose": 3,
            "reye_pose": 3,
        }
        validated_entries: List[Dict[str, np.ndarray]] = []
        finite_masks: List[np.ndarray] = []
        jaw_norms: List[np.ndarray] = []
        expression_norms: List[np.ndarray] = []
        eye_norms: List[np.ndarray] = []
        frame_offsets: List[int] = []
        offset = 0
        for entry_index, entry in enumerate(entries):
            missing = [name for name in required_widths if name not in entry]
            if missing:
                raise KeyError(
                    f"{self.chemistry_path}: entry {entry_index} is missing "
                    + ", ".join(missing)
                )
            frame_count: Optional[int] = None
            validated: Dict[str, np.ndarray] = {}
            for name, width in required_widths.items():
                value = np.asarray(entry[name], dtype=np.float32)
                if value.ndim == 1:
                    value = value[None]
                if value.ndim != 2 or value.shape[1] < width:
                    raise ValueError(
                        f"{self.chemistry_path}: entry {entry_index} '{name}' "
                        f"must have shape (T, >= {width}), got {value.shape}"
                    )
                if frame_count is None:
                    frame_count = int(value.shape[0])
                elif value.shape[0] != frame_count:
                    raise ValueError(
                        f"{self.chemistry_path}: entry {entry_index} has "
                        "inconsistent frame counts across expression/jaw/eyes"
                    )
                validated[name] = value[:, :width]
            assert frame_count is not None
            finite = np.ones(frame_count, dtype=bool)
            for value in validated.values():
                finite &= np.isfinite(value).all(axis=1)
            validated_entries.append(validated)
            finite_masks.append(finite)
            jaw_norms.append(
                np.linalg.norm(validated["jaw_pose"], axis=1).astype(
                    np.float32, copy=False
                )
            )
            expression_norms.append(
                np.linalg.norm(validated["expression"], axis=1).astype(
                    np.float32, copy=False
                )
            )
            eye_norms.append(
                np.maximum(
                    np.linalg.norm(validated["leye_pose"], axis=1),
                    np.linalg.norm(validated["reye_pose"], axis=1),
                ).astype(np.float32, copy=False)
            )
            frame_offsets.append(offset)
            offset += frame_count

        total = int(offset)
        finite_count = int(sum(int(mask.sum()) for mask in finite_masks))
        if finite_count == 0:
            raise ValueError(
                f"{self.chemistry_path}: no finite expression/jaw/eye frames"
            )

        quantile = float(self.cfg.chemistry_jaw_outlier_quantile)
        if not 0.0 < quantile <= 1.0:
            raise ValueError(
                "chemistry_jaw_outlier_quantile must be in (0, 1]"
            )
        finite_jaw_norms = np.concatenate(
            [
                norm[finite]
                for norm, finite in zip(jaw_norms, finite_masks)
                if finite.any()
            ]
        )
        quantile_threshold = float(np.quantile(finite_jaw_norms, quantile))
        absolute_threshold = float(self.cfg.chemistry_jaw_max_norm)
        if absolute_threshold > 0.0:
            if not math.isfinite(absolute_threshold):
                raise ValueError("chemistry_jaw_max_norm must be finite")
            jaw_threshold = min(quantile_threshold, absolute_threshold)
        else:
            jaw_threshold = quantile_threshold
        expression_threshold = float(self.cfg.chemistry_expression_max_norm)
        eye_threshold = float(self.cfg.chemistry_eye_max_norm)
        for name, threshold in (
            ("chemistry_expression_max_norm", expression_threshold),
            ("chemistry_eye_max_norm", eye_threshold),
        ):
            if not math.isfinite(threshold):
                raise ValueError(f"{name} must be finite")
        keep_masks = []
        for finite, jaw_norm, expression_norm, eye_norm in zip(
            finite_masks, jaw_norms, expression_norms, eye_norms
        ):
            keep = finite & (jaw_norm <= jaw_threshold)
            if expression_threshold > 0.0:
                keep &= expression_norm <= expression_threshold
            if eye_threshold > 0.0:
                keep &= eye_norm <= eye_threshold
            keep_masks.append(keep)
        kept_count = int(sum(int(mask.sum()) for mask in keep_masks))
        if kept_count == 0:
            raise ValueError(
                f"{self.chemistry_path}: chemistry filtering rejected every finite frame "
                f"(quantile={quantile}, threshold={quantile_threshold:.6f}, "
                f"jaw_absolute={absolute_threshold}, "
                f"expression_max={expression_threshold}, eye_max={eye_threshold})"
            )

        # Allocate each retained array exactly once.  chemistry_exp.npy has more
        # than half a million frames, so concatenating all frames before
        # filtering would transiently duplicate hundreds of MB of expressions.
        expression = np.empty((kept_count, 100), dtype=np.float32)
        jaw = np.empty((kept_count, 3), dtype=np.float32)
        leye = np.empty((kept_count, 3), dtype=np.float32)
        reye = np.empty((kept_count, 3), dtype=np.float32)
        source_indices = np.empty((kept_count,), dtype=np.int64)
        output_offset = 0
        for values, keep, source_offset in zip(
            validated_entries, keep_masks, frame_offsets
        ):
            count = int(keep.sum())
            if count == 0:
                continue
            target = slice(output_offset, output_offset + count)
            expression[target] = values["expression"][keep]
            jaw[target] = values["jaw_pose"][keep]
            leye[target] = values["leye_pose"][keep]
            reye[target] = values["reye_pose"][keep]
            source_indices[target] = (
                np.flatnonzero(keep).astype(np.int64, copy=False) + source_offset
            )
            output_offset += count
        assert output_offset == kept_count

        open_quantile = float(self.cfg.chemistry_open_mouth_quantile)
        if not 0.0 <= open_quantile <= 1.0:
            raise ValueError("chemistry_open_mouth_quantile must be in [0, 1]")
        # FLAME's first jaw axis is the mouth opening/closing rotation.  Using
        # it instead of the full norm prevents lateral jaw/pose noise from
        # being mislabeled as an open mouth.
        open_score = jaw[:, 0]
        open_threshold = float(np.quantile(open_score, open_quantile))
        is_open = open_score >= open_threshold
        if not is_open.any():
            raise ValueError(
                f"{self.chemistry_path}: open-mouth quantile produced no samples"
            )

        probability = float(self.cfg.chemistry_open_mouth_prob)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("chemistry_open_mouth_prob must be in [0, 1]")
        return (
            expression,
            jaw,
            leye,
            reye,
            source_indices,
            is_open,
            open_threshold,
            {
                "total": total,
                "finite": finite_count,
                "kept": kept_count,
                "open": int(is_open.sum()),
                "jaw_threshold": jaw_threshold,
                "expression_threshold": expression_threshold,
                "eye_threshold": eye_threshold,
            },
        )

    @staticmethod
    def _range_pair(value: Any, name: str) -> Optional[tuple[float, float]]:
        if value is None:
            return None
        values = list(value)
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values")
        lower, upper = float(values[0]), float(values[1])
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError(f"{name} values must be finite")
        return lower, upper

    @staticmethod
    def _angle_in_range(
        value: float,
        bounds: Optional[tuple[float, float]],
        circular: bool,
    ) -> bool:
        if bounds is None:
            return True
        lower, upper = bounds
        if not circular:
            return lower <= value <= upper
        value = value % 360.0
        lower, upper = lower % 360.0, upper % 360.0
        if lower <= upper:
            return lower <= value <= upper
        return value >= lower or value <= upper

    def _training_camera_groups(self, batch_size: int) -> List[List[int]]:
        azimuth_range = self._range_pair(
            self.cfg.train_azimuth_range, "train_azimuth_range"
        )
        elevation_range = self._range_pair(
            self.cfg.train_elevation_range, "train_elevation_range"
        )
        eligible: List[List[int]] = []
        for group in self.elevation_groups:
            filtered = [
                index
                for index in group
                if self._angle_in_range(
                    self.frames[index].prompt_azimuth_deg,
                    azimuth_range,
                    circular=True,
                )
                and self._angle_in_range(
                    self.frames[index].prompt_elevation_deg,
                    elevation_range,
                    circular=False,
                )
            ]
            if len(filtered) >= batch_size:
                eligible.append(filtered)
        if not eligible:
            raise RuntimeError(
                "No calibrated same-elevation camera ring satisfies "
                f"batch_size={batch_size}, train_azimuth_range="
                f"{self.cfg.train_azimuth_range}, and "
                f"train_elevation_range={self.cfg.train_elevation_range}"
            )
        return eligible

    def sample_camera_indices(self, batch_size: int) -> List[int]:
        eligible = self._training_camera_groups(batch_size)
        if not eligible:
            raise RuntimeError(
                f"No same-elevation camera ring can provide batch_size={batch_size}"
            )
        front_probability = self.cfg.train_front_hemisphere_probability
        if front_probability is not None:
            sample_front = random.random() < float(front_probability)
            hemisphere_groups = []
            for candidate_group in eligible:
                filtered = [
                    index
                    for index in candidate_group
                    if (
                        0.0
                        <= self.frames[index].prompt_azimuth_deg % 360.0
                        <= 180.0
                    )
                    == sample_front
                ]
                if len(filtered) >= batch_size:
                    hemisphere_groups.append(filtered)
            if not hemisphere_groups:
                hemisphere = "front" if sample_front else "back"
                raise RuntimeError(
                    "No calibrated camera ring can satisfy the requested "
                    f"{hemisphere}-hemisphere sample for batch_size={batch_size}"
                )
            eligible = hemisphere_groups
        group = eligible[random.randrange(len(eligible))]
        count = len(group)
        phase = random.randrange(count)
        positions = [
            (phase + int(math.floor(index * count / batch_size))) % count
            for index in range(batch_size)
        ]
        selected = [group[position] for position in positions]
        random.shuffle(selected)
        return selected

    def sample_surface_camera_indices(
        self, batch_size: int
    ) -> List[int]:
        """Sample a legacy B=1 anchor, then same-ring surface companions."""

        count_requested = int(batch_size)
        if count_requested < 2:
            raise ValueError(
                "surface-consistent camera batches require at least two views"
            )
        # This call preserves the exact original elevation-ring,
        # front/back-mixture, and within-ring distribution for row zero.
        anchor = self.sample_camera_indices(1)[0]
        eligible = self._training_camera_groups(1)
        candidates = next(
            (group for group in eligible if anchor in group), None
        )
        if candidates is None:
            raise RuntimeError(
                "Surface-view anchor disappeared from its eligible camera ring"
            )
        # Prefer the configured azimuth window. If it is narrower than K,
        # widen only the companion set to the full same-elevation ring. The
        # anchor distribution remains unchanged and every view stays distinct.
        if len(candidates) < count_requested:
            candidates = next(
                (
                    group
                    for group in self.elevation_groups
                    if anchor in group
                ),
                None,
            )
        if candidates is None or len(candidates) < count_requested:
            available = 0 if candidates is None else len(candidates)
            raise RuntimeError(
                "The anchor's calibrated elevation ring cannot provide "
                f"{count_requested} distinct surface views (available="
                f"{available})"
            )
        anchor_position = candidates.index(anchor)
        candidate_count = len(candidates)
        positions = [
            (
                anchor_position
                + int(math.floor(index * candidate_count / count_requested))
            )
            % candidate_count
            for index in range(count_requested)
        ]
        selected = [candidates[position] for position in positions]
        if selected[0] != anchor or len(set(selected)) != count_requested:
            raise RuntimeError(
                "Surface-view sampler failed to retain a distinct row-zero anchor"
            )
        return selected

    def _continuous_camera_frame(
        self,
        source_azimuth_deg: float,
        source_elevation_deg: float,
        sample_index: int,
    ) -> _CameraFrame:
        width, height = int(self.cfg.width), int(self.cfg.height)
        K = self.continuous_camera_intrinsics.copy()
        c2w, w2c = _calibrated_orbit_camera(
            azimuth_deg=source_azimuth_deg,
            elevation_deg=source_elevation_deg,
            radius=self.continuous_camera_radius,
            pivot=self.continuous_camera_pivot,
            intrinsics=K,
        )
        camera_facelift_h = np.concatenate(
            (c2w[:3, 3].astype(np.float64), np.ones(1, dtype=np.float64))
        )
        camera_training_h = self.training_from_facelift @ camera_facelift_h
        homogeneous_scale = float(camera_training_h[3])
        if abs(homogeneous_scale) <= 1.0e-10:
            raise ValueError(
                "Continuous camera centre maps to an invalid training point"
            )
        training_origin = camera_training_h[:3] / homogeneous_scale
        prompt_distance = float(np.linalg.norm(training_origin))
        if not math.isfinite(prompt_distance) or prompt_distance <= 1.0e-8:
            raise ValueError("Continuous camera has an invalid prompt distance")
        prompt_azimuth = math.degrees(
            math.atan2(float(training_origin[1]), float(training_origin[0]))
        )
        prompt_elevation = math.degrees(
            math.atan2(
                float(training_origin[2]),
                float(np.linalg.norm(training_origin[:2])),
            )
        )
        return _CameraFrame(
            frame_index=int(sample_index),
            # Continuous training has no captured RGB frame.  The path is
            # retained only because calibrated/eval frames share this record
            # type; the batch builder never reads it in continuous mode.
            image_path=self.frames[0].image_path,
            width=width,
            height=height,
            K=K,
            c2w=c2w,
            w2c=w2c,
            source_azimuth_deg=float(source_azimuth_deg) % 360.0,
            source_elevation_deg=float(source_elevation_deg),
            prompt_camera_position=training_origin.astype(np.float32),
            prompt_azimuth_deg=float(prompt_azimuth),
            prompt_elevation_deg=float(prompt_elevation),
            prompt_distance=prompt_distance,
        )

    def _continuous_frame_is_eligible(
        self, frame: _CameraFrame, sample_front: Optional[bool]
    ) -> bool:
        azimuth_range = self._range_pair(
            self.cfg.train_azimuth_range, "train_azimuth_range"
        )
        elevation_range = self._range_pair(
            self.cfg.train_elevation_range, "train_elevation_range"
        )
        if not self._angle_in_range(
            frame.prompt_azimuth_deg, azimuth_range, circular=True
        ):
            return False
        if not self._angle_in_range(
            frame.prompt_elevation_deg, elevation_range, circular=False
        ):
            return False
        if sample_front is not None:
            is_front = 0.0 <= frame.prompt_azimuth_deg % 360.0 <= 180.0
            if is_front != sample_front:
                return False
        return True

    def _sample_continuous_camera_frame(
        self, sample_index: int
    ) -> _CameraFrame:
        front_probability = self.cfg.train_front_hemisphere_probability
        sample_front = (
            None
            if front_probability is None
            else random.random() < float(front_probability)
        )
        lower, upper = self.continuous_source_elevation_range
        for _ in range(4096):
            source_azimuth = random.random() * 360.0
            source_elevation = lower + random.random() * (upper - lower)
            frame = self._continuous_camera_frame(
                source_azimuth, source_elevation, sample_index
            )
            if self._continuous_frame_is_eligible(frame, sample_front):
                return frame
        hemisphere = (
            "either"
            if sample_front is None
            else ("front" if sample_front else "back")
        )
        raise RuntimeError(
            "Failed to sample a continuous calibrated camera after 4096 "
            f"attempts (hemisphere={hemisphere}, train_azimuth_range="
            f"{self.cfg.train_azimuth_range}, train_elevation_range="
            f"{self.cfg.train_elevation_range})"
        )

    def sample_continuous_camera_frames(
        self, batch_size: int
    ) -> List[_CameraFrame]:
        if self.train_camera_sampling != "calibrated_continuous":
            raise RuntimeError(
                "Continuous cameras requested while train_camera_sampling="
                f"{self.train_camera_sampling!r}"
            )
        return [
            self._sample_continuous_camera_frame(-(index + 1))
            for index in range(batch_size)
        ]

    def sample_pose(self) -> _Pose:
        pose_mode = str(self.cfg.train_pose_mode).lower()
        if pose_mode == "reference":
            return self.reference_pose
        if pose_mode == "animportrait3d_mouth":
            return self.animportrait3d_mouth_pose(
                random.randrange(self.open_mouth_expression.shape[0])
            )
        if pose_mode == "open_mouth":
            local_index = int(
                self.chemistry_open_indices[
                    random.randrange(len(self.chemistry_open_indices))
                ]
            )
            return self.chemistry_pose(local_index)

        reference_probability = float(self.cfg.reference_pose_probability)
        if (
            not bool(self.cfg.use_dynamic_expression)
            or random.random() < reference_probability
        ):
            return self.reference_pose

        if (
            bool(self.cfg.chemistry_open_mouth_oversample)
            and random.random() < float(self.cfg.chemistry_open_mouth_prob)
        ):
            local_index = int(
                self.chemistry_open_indices[
                    random.randrange(len(self.chemistry_open_indices))
                ]
            )
        else:
            local_index = random.randrange(self.chemistry_expression.shape[0])
        return self.chemistry_pose(local_index)

    def animportrait3d_mouth_pose(self, local_index: int) -> _Pose:
        """Return one paired row from GSAvatar's open-mouth sequence."""

        frame_count = self.open_mouth_expression.shape[0]
        if frame_count == 0:
            raise RuntimeError("No AnimPortrait3D open-mouth poses are available")
        local_index %= frame_count

        # GSAvatar ignores the stored eye pose here and independently samples
        # conjugate eye motion for every mouth-training render.
        eye_x = random.uniform(-0.3, 0.3)
        eye_y = random.uniform(-0.4, 0.4)
        leye = np.asarray([[eye_x, eye_y, 0.0]], dtype=np.float32)
        reye = leye.copy()
        return _Pose(
            expression=self.open_mouth_expression[
                local_index:local_index + 1
            ],
            jaw_pose=self.open_mouth_jaw[local_index:local_index + 1],
            leye_pose=leye,
            reye_pose=reye,
            source_index=local_index,
            is_open_mouth=True,
            is_reference=False,
        )

    def chemistry_pose(self, local_index: int) -> _Pose:
        if self.chemistry_expression.shape[0] == 0:
            raise RuntimeError("No valid chemistry poses are available")
        local_index %= self.chemistry_expression.shape[0]
        return _Pose(
            expression=self.chemistry_expression[local_index : local_index + 1],
            jaw_pose=self.chemistry_jaw[local_index : local_index + 1],
            leye_pose=self.chemistry_leye[local_index : local_index + 1],
            reye_pose=self.chemistry_reye[local_index : local_index + 1],
            source_index=int(self.chemistry_source_indices[local_index]),
            is_open_mouth=bool(self.chemistry_is_open[local_index]),
            is_reference=False,
        )

    def test_pose(self, index: int) -> _Pose:
        if not 0 <= index < self.test_frame_count:
            raise IndexError(
                f"Test pose index {index} is outside [0, {self.test_frame_count})"
            )
        return _Pose(
            expression=self.test_expression[index : index + 1],
            jaw_pose=self.test_jaw[index : index + 1],
            leye_pose=self.test_leye[index : index + 1],
            reye_pose=self.test_reye[index : index + 1],
            source_index=index,
            is_open_mouth=bool(
                float(self.test_jaw[index, 0]) >= float(self.open_mouth_threshold)
            ),
            is_reference=False,
        )

    def load_reference_rgb(self, camera_indices: Sequence[int]) -> torch.Tensor:
        images = []
        for camera_index in camera_indices:
            if camera_index not in self._rgb_cache:
                frame = self.frames[camera_index]
                with Image.open(frame.image_path) as image_file:
                    image = image_file.convert("RGB")
                    if image.size != (frame.width, frame.height):
                        raise ValueError(
                            f"{frame.image_path}: actual image size {image.size} "
                            f"does not match cameras.json "
                            f"({frame.width}, {frame.height})"
                        )
                    array = np.asarray(image, dtype=np.uint8).copy()
                self._rgb_cache[camera_index] = torch.from_numpy(array)
            images.append(self._rgb_cache[camera_index])
        return torch.stack(images, dim=0).to(dtype=torch.float32).div_(255.0)

    def load_reference_alpha(
        self, camera_indices: Sequence[int]
    ) -> Optional[torch.Tensor]:
        if self.reference_alpha_paths is None:
            return None
        masks = []
        for camera_index in camera_indices:
            if camera_index not in self._alpha_cache:
                frame = self.frames[camera_index]
                path = self.reference_alpha_paths[camera_index]
                with Image.open(path) as image_file:
                    image = image_file.convert("L")
                    if image.size != (frame.width, frame.height):
                        raise ValueError(
                            f"{path}: alpha size {image.size} does not match "
                            f"cameras.json ({frame.width}, {frame.height})"
                        )
                    array = np.asarray(image, dtype=np.uint8).copy()
                self._alpha_cache[camera_index] = torch.from_numpy(array)[..., None]
            masks.append(self._alpha_cache[camera_index])
        return torch.stack(masks, dim=0).to(dtype=torch.float32).div_(255.0)


class _MediapipeConditioner:
    """FLAME landmarks projected by the exact first-stage OpenCV cameras."""

    def __init__(
        self,
        assets: _ReconstructionAssets,
        cfg: ReconstructionFinetuneDataModuleConfig,
    ) -> None:
        self.assets = assets
        self.cfg = cfg
        requested_device = torch.device(str(cfg.condition_device))
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            threestudio.warn(
                "condition_device is CUDA but CUDA is unavailable; falling back "
                "to CPU for MediaPipe condition generation."
            )
            requested_device = torch.device("cpu")
        self.device = requested_device

        self.model = FlameHead(
            shape_params=300,
            expr_params=100,
            include_mask=True,
            add_teeth=True,
        ).to(self.device)
        self.model.eval()
        self.faces = torch.as_tensor(
            self.model.faces, dtype=torch.long, device=self.device
        )

        embedding = np.load(FLAME_MEDIAPIPE_LMK_PATH, allow_pickle=True)
        self.lmk105_faces = torch.as_tensor(
            np.asarray(embedding["lmk_face_idx"], dtype=np.int64),
            dtype=torch.long,
            device=self.device,
        )
        self.lmk105_bary = torch.as_tensor(
            np.asarray(embedding["lmk_b_coords"], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        lmk468 = np.asarray(
            np.load(FLAME_MEDIAPIPE_468_PATH), dtype=np.int64
        ).reshape(-1)
        lmk468 = np.concatenate(
            [lmk468, np.array([4597, 4051], dtype=np.int64)], axis=0
        )
        self.lmk468_vertices = torch.as_tensor(
            lmk468, dtype=torch.long, device=self.device
        )
        if int(self.lmk468_vertices.max()) >= int(self.faces.max()) + 1:
            raise ValueError(
                "FLAME MediaPipe vertex map contains an out-of-range vertex index"
            )

        zeros_shape = torch.zeros((1, 300), dtype=torch.float32, device=self.device)
        zeros_expression = torch.zeros(
            (1, 100), dtype=torch.float32, device=self.device
        )
        zeros3 = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        zeros6 = torch.zeros((1, 6), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            neutral = self.model(
                shape=zeros_shape,
                expr=zeros_expression,
                rotation=zeros3,
                neck=zeros3,
                jaw=zeros3,
                eyes=zeros6,
                translation=zeros3,
                zero_centered_at_root_node=False,
                return_landmarks=False,
            ).squeeze(0)
        self.center = ((neutral.amin(0) + neutral.amax(0)) / 2.0).detach()
        self.scale = (
            0.6 / (neutral.amax(0) - neutral.amin(0)).amax()
        ).detach()
        self.alignment = torch.as_tensor(
            assets.facelift_from_training,
            dtype=torch.float32,
            device=self.device,
        )

    @torch.inference_mode()
    def render(
        self,
        pose: _Pose,
        frames: Sequence[_CameraFrame],
    ) -> torch.Tensor:
        shape = torch.as_tensor(
            self.assets.shape, dtype=torch.float32, device=self.device
        )
        expression = torch.as_tensor(
            pose.expression, dtype=torch.float32, device=self.device
        )
        jaw = torch.as_tensor(
            pose.jaw_pose, dtype=torch.float32, device=self.device
        )
        leye = torch.as_tensor(
            pose.leye_pose, dtype=torch.float32, device=self.device
        )
        reye = torch.as_tensor(
            pose.reye_pose, dtype=torch.float32, device=self.device
        )
        zeros3 = torch.zeros((1, 3), dtype=torch.float32, device=self.device)

        # Global orientation, neck and translation remain exactly zero.  The
        # batch API intentionally exposes none of them.
        vertices = self.model(
            shape=shape,
            expr=expression,
            rotation=zeros3,
            neck=zeros3,
            jaw=jaw,
            eyes=torch.cat([leye, reye], dim=-1),
            translation=zeros3,
            zero_centered_at_root_node=False,
            return_landmarks=False,
        ).squeeze(0)
        vertices = (vertices - self.center) * self.scale
        vertices = vertices.clone()
        vertices[:, [1, 2]] = vertices[:, [2, 1]]
        vertices *= 1.1 ** (-float(self.assets.flame_scale))
        vertices = (
            vertices @ self.alignment[:3, :3].T
            + self.alignment[:3, 3][None]
        )

        triangles105 = vertices[self.faces[self.lmk105_faces]]
        landmarks105 = (
            triangles105 * self.lmk105_bary[:, :, None]
        ).sum(dim=1)
        landmarks468 = vertices[self.lmk468_vertices]
        landmarks = torch.cat([landmarks105, landmarks468], dim=0)

        K = torch.as_tensor(
            np.stack([frame.K for frame in frames]),
            dtype=torch.float32,
            device=self.device,
        )
        w2c = torch.as_tensor(
            np.stack([frame.w2c for frame in frames]),
            dtype=torch.float32,
            device=self.device,
        )
        camera_points = (
            torch.einsum("bij,nj->bni", w2c[:, :3, :3], landmarks)
            + w2c[:, None, :3, 3]
        )
        homogeneous_pixels = torch.einsum("bij,bnj->bni", K, camera_points)
        depth = camera_points[..., 2]
        projected = (
            homogeneous_pixels[..., :2]
            / homogeneous_pixels[..., 2:3].clamp_min(
                float(self.cfg.projection_min_depth)
            )
        )
        valid = (
            torch.isfinite(projected).all(dim=-1)
            & torch.isfinite(depth)
            & (depth > float(self.cfg.projection_min_depth))
        )
        projected = projected.masked_fill(~valid[..., None], float("nan"))
        projected_np = projected.detach().cpu().numpy()

        conditions = []
        count105 = int(landmarks105.shape[0])
        for batch_index, frame in enumerate(frames):
            canvas = np.ones(
                (1, frame.height, frame.width, 3), dtype=np.uint8
            )
            image = draw_landmarks_105(
                canvas, projected_np[batch_index : batch_index + 1, :count105]
            )[0]
            image = draw_landmarks_468(
                image, projected_np[batch_index, count105:]
            )
            conditions.append(np.ascontiguousarray(image))
        condition = np.stack(conditions, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(condition)


class _AnimPortrait3DNormalSegConditioner:
    """AnimPortrait3D's four-channel FLAME normal + segmentation control.

    The reference implementation renders camera-space normals on its fixed
    face-region topology, renders semantic vertex colours on the complete
    articulated FLAME mesh, then concatenates ``normal / 2 + 0.5`` with the
    blue segmentation channel.  This implementation keeps those exact assets
    and raster conventions, but poses the same generated-teeth FLAME topology
    used by Stage 1 so the condition remains registered to the reconstructed
    UVD avatar.
    """

    channels = 4
    _REFERENCE_BASE_VERTEX_COUNT = 5023

    def __init__(
        self,
        assets: _ReconstructionAssets,
        cfg: ReconstructionFinetuneDataModuleConfig,
    ) -> None:
        self.assets = assets
        self.cfg = cfg
        requested_device = torch.device(str(cfg.condition_device))
        if requested_device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "animportrait3d_normal_seg condition generation requires a "
                "CUDA condition_device because it uses the reference "
                "nvdiffrast CUDA rasterizer"
            )
        self.device = requested_device
        try:
            import nvdiffrast.torch as dr
        except ImportError as error:
            raise RuntimeError(
                "AnimPortrait3D normal+seg control requires nvdiffrast"
            ) from error
        self.dr = dr
        self.glctx = dr.RasterizeCudaContext(device=self.device)

        assets_dir = _resolve_path(str(cfg.animportrait3d_assets_dir))
        required = {
            "face_region_faces": assets_dir / "face_region_faces.npy",
            "face_region_verts_mask": assets_dir / "face_region_verts_mask.npy",
            "verts_seg": assets_dir / "verts_seg.npy",
            "verts_seg_idxs": assets_dir / "verts_seg_idxs.json",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "AnimPortrait3D normal+seg assets are incomplete; missing: "
                + ", ".join(missing)
            )
        self.assets_dir = assets_dir

        self.model = Stage1FlameHead(
            shape_params=300,
            expr_params=100,
            add_teeth=True,
        ).to(self.device)
        self.model.eval()
        vertex_count = int(self.model.v_template.shape[0])
        dental_count = int(getattr(self.model, "num_verts_teeth", 0))
        base_vertex_count = vertex_count - dental_count
        if base_vertex_count != self._REFERENCE_BASE_VERTEX_COUNT:
            raise ValueError(
                "Stage-1 FLAME base topology is incompatible with "
                f"AnimPortrait3D assets: expected {self._REFERENCE_BASE_VERTEX_COUNT} "
                f"vertices before dentition, got {base_vertex_count}"
            )

        def model_vertex_region(name: str) -> np.ndarray:
            return (
                torch.unique(
                    self.model.mask.get_vid_by_region(name).long()
                )
                .detach()
                .cpu()
                .numpy()
            )

        model_faces_cpu = (
            torch.as_tensor(self.model.faces, dtype=torch.long)
            .detach()
            .cpu()
            .numpy()
        )
        full_faces = torch.as_tensor(
            model_faces_cpu, dtype=torch.long, device=self.device
        )
        # AnimPortrait3D's NVDiffRenderer reverses the model faces at entry.
        self.full_faces = full_faces[:, [0, 2, 1]].to(torch.int32).contiguous()

        face_vertex_ids_np = np.asarray(
            np.load(required["face_region_verts_mask"], allow_pickle=False),
            dtype=np.int64,
        ).reshape(-1)
        face_faces_np = np.asarray(
            np.load(required["face_region_faces"], allow_pickle=False),
            dtype=np.int64,
        )
        if face_faces_np.ndim != 2 or face_faces_np.shape[1] != 3:
            raise ValueError(
                f"{required['face_region_faces']}: expected Fx3 faces, "
                f"got {face_faces_np.shape}"
            )
        if (
            face_vertex_ids_np.size == 0
            or face_vertex_ids_np.min() < 0
            or face_vertex_ids_np.max() >= base_vertex_count
        ):
            raise ValueError(
                f"{required['face_region_verts_mask']}: indices are outside "
                "the base FLAME topology"
            )
        if (
            face_faces_np.size == 0
            or face_faces_np.min() < 0
            or face_faces_np.max() >= face_vertex_ids_np.size
        ):
            raise ValueError(
                f"{required['face_region_faces']}: local face indices are invalid"
            )
        if np.unique(face_vertex_ids_np).size != face_vertex_ids_np.size:
            raise ValueError(
                f"{required['face_region_verts_mask']}: duplicate vertex indices"
            )
        expected_face_vertex_ids = np.union1d(
            model_vertex_region("face"), model_vertex_region("nose")
        )
        if not np.array_equal(face_vertex_ids_np, expected_face_vertex_ids):
            raise ValueError(
                "AnimPortrait3D face-region vertices must equal the current "
                "FLAME face + nose regions"
            )

        # The face-region files are local-indexed, so merely checking their
        # ranges is not enough: a file made for another FLAME revision could
        # still pass.  Map every triangle back to global indices and require
        # the exact oriented subset of the current shared 5023-vertex mesh.
        mapped_face_faces_np = face_vertex_ids_np[face_faces_np]
        base_faces_np = model_faces_cpu[
            np.all(
                (model_faces_cpu >= 0)
                & (model_faces_cpu < base_vertex_count),
                axis=1,
            )
        ]
        expected_face_faces_np = base_faces_np[
            np.all(np.isin(base_faces_np, face_vertex_ids_np), axis=1)
        ]
        mapped_face_rows = {tuple(row) for row in mapped_face_faces_np.tolist()}
        expected_face_rows = {
            tuple(row) for row in expected_face_faces_np.tolist()
        }
        if (
            len(mapped_face_rows) != mapped_face_faces_np.shape[0]
            or mapped_face_rows != expected_face_rows
        ):
            missing_count = len(expected_face_rows - mapped_face_rows)
            extra_count = len(mapped_face_rows - expected_face_rows)
            raise ValueError(
                "AnimPortrait3D face-region assets do not match the current "
                "base FLAME topology/orientation: "
                f"{missing_count} missing and {extra_count} extra triangles"
            )
        if not np.array_equal(
            np.unique(mapped_face_faces_np), np.sort(face_vertex_ids_np)
        ):
            raise ValueError(
                "AnimPortrait3D face-region faces do not use exactly the "
                "vertices in face_region_verts_mask.npy"
            )
        self.face_vertex_ids = torch.as_tensor(
            face_vertex_ids_np, dtype=torch.long, device=self.device
        )
        # helper.py loads [:, [0, 2, 1]] and render_from_camera reverses once
        # more, so the triangle order passed to dr.rasterize is the stored one.
        self.face_faces = torch.as_tensor(
            face_faces_np, dtype=torch.int32, device=self.device
        ).contiguous()

        reference_seg = np.asarray(
            np.load(required["verts_seg"], allow_pickle=False),
            dtype=np.float32,
        )
        if reference_seg.ndim == 3 and reference_seg.shape[0] == 1:
            reference_seg = reference_seg[0]
        if (
            reference_seg.ndim != 2
            or reference_seg.shape[1] != 3
            or reference_seg.shape[0] < base_vertex_count
        ):
            raise ValueError(
                f"{required['verts_seg']}: expected at least "
                f"({base_vertex_count}, 3), got {reference_seg.shape}"
            )
        if not np.isfinite(reference_seg).all():
            raise ValueError(f"{required['verts_seg']}: contains non-finite values")
        if reference_seg.min() < 0.0 or reference_seg.max() > 255.0:
            raise ValueError(
                f"{required['verts_seg']}: expected semantic RGB values in "
                "[0, 255]"
            )

        with required["verts_seg_idxs"].open("r", encoding="utf-8") as file:
            seg_indices = json.load(file)
        if not isinstance(seg_indices, dict) or not {
            "left_eye",
            "right_eye",
            "teeth",
        }.issubset(seg_indices):
            raise ValueError(
                f"{required['verts_seg_idxs']}: missing eye/teeth index groups"
            )

        def reference_indices(name: str) -> np.ndarray:
            raw_indices = seg_indices[name]
            if (
                not isinstance(raw_indices, list)
                or not raw_indices
                or any(
                    isinstance(index, bool) or not isinstance(index, int)
                    for index in raw_indices
                )
            ):
                raise ValueError(
                    f"{required['verts_seg_idxs']}: {name!r} must be a "
                    "non-empty list of integer vertex indices"
                )
            indices = np.unique(np.asarray(raw_indices, dtype=np.int64))
            if indices[0] < 0 or indices[-1] >= reference_seg.shape[0]:
                raise ValueError(
                    f"{required['verts_seg_idxs']}: {name!r} indices are "
                    "outside verts_seg.npy"
                )
            return indices

        reference_left_eye_ids = reference_indices("left_eye")
        reference_right_eye_ids = reference_indices("right_eye")
        reference_teeth_ids = reference_indices("teeth")
        if (
            np.intersect1d(
                reference_left_eye_ids, reference_right_eye_ids
            ).size
            or np.intersect1d(
                reference_left_eye_ids, reference_teeth_ids
            ).size
            or np.intersect1d(
                reference_right_eye_ids, reference_teeth_ids
            ).size
        ):
            raise ValueError(
                f"{required['verts_seg_idxs']}: semantic groups overlap"
            )

        # The first 5023 vertices (head plus eyeballs) really are shared by
        # both models.  Verify the JSON eye sets against the current mask before
        # copying their values; duplicate JSON entries are intentionally
        # collapsed because the reference lists overlapping iris/eye regions.
        current_left_eye_ids = model_vertex_region("left_eyeball")
        current_right_eye_ids = model_vertex_region("right_eyeball")
        if not np.array_equal(reference_left_eye_ids, current_left_eye_ids):
            raise ValueError(
                "AnimPortrait3D left-eye indices do not match the current "
                "base FLAME topology"
            )
        if not np.array_equal(reference_right_eye_ids, current_right_eye_ids):
            raise ValueError(
                "AnimPortrait3D right-eye indices do not match the current "
                "base FLAME topology"
            )
        current_left_iris_ids = model_vertex_region("left_iris")
        current_right_iris_ids = model_vertex_region("right_iris")
        if not np.all(np.isin(current_left_iris_ids, current_left_eye_ids)):
            raise ValueError("Current left iris is not inside the left eyeball")
        if not np.all(np.isin(current_right_iris_ids, current_right_eye_ids)):
            raise ValueError("Current right iris is not inside the right eyeball")

        expected_base_seg = np.zeros(
            (base_vertex_count, 3), dtype=np.float32
        )
        expected_base_seg[current_left_eye_ids] = 255.0
        expected_base_seg[current_right_eye_ids] = 255.0
        expected_base_seg[current_left_iris_ids] = 85.0
        expected_base_seg[current_right_iris_ids] = 85.0
        if not np.array_equal(
            reference_seg[:base_vertex_count], expected_base_seg
        ):
            raise ValueError(
                f"{required['verts_seg']}: shared-base segmentation does not "
                "match the required sclera=255, iris=85 palette"
            )

        reference_base_nonzero = np.flatnonzero(
            np.any(np.abs(reference_seg[:base_vertex_count]) > 1.0e-6, axis=1)
        )
        expected_base_nonzero = np.union1d(
            reference_left_eye_ids, reference_right_eye_ids
        )
        if not np.array_equal(reference_base_nonzero, expected_base_nonzero):
            raise ValueError(
                f"{required['verts_seg']}: non-zero base semantics are not "
                "exactly the two validated eyeball groups"
            )
        if np.any(reference_teeth_ids < base_vertex_count):
            raise ValueError(
                f"{required['verts_seg_idxs']}: reference teeth indices "
                "unexpectedly overlap the shared base FLAME vertices"
            )
        reference_appended_nonzero = np.flatnonzero(
            np.any(
                np.abs(reference_seg[base_vertex_count:]) > 1.0e-6,
                axis=1,
            )
        ) + base_vertex_count
        if not np.array_equal(reference_appended_nonzero, reference_teeth_ids):
            raise ValueError(
                f"{required['verts_seg']}: appended non-zero semantics do not "
                "exactly match the reference teeth index group"
            )
        reference_teeth_colours = np.unique(
            reference_seg[reference_teeth_ids], axis=0
        )
        if (
            reference_teeth_colours.shape != (1, 3)
            or reference_teeth_colours[0, 2] <= 0.0
            or not np.allclose(
                reference_teeth_colours[0], reference_teeth_colours[0, 2]
            )
        ):
            raise ValueError(
                f"{required['verts_seg']}: reference teeth must use one "
                "non-zero grayscale semantic value"
            )
        reference_teeth_colour = reference_teeth_colours[0] / 255.0

        # The appended dental topology is deliberately *not* shared:
        # AnimPortrait3D has 2986 dental vertices and this repository builds
        # 2504 watertight crown/gum vertices.  Treat the reference indices as a
        # semantic codebook, then transfer by the current model's named regions.
        crown_ids_np = model_vertex_region("teeth_crowns")
        gum_ids_np = model_vertex_region("gums")
        current_teeth_ids = model_vertex_region("teeth")
        oral_vertex_ids_np = model_vertex_region("oral_cavity")
        expected_dental_ids = np.arange(
            base_vertex_count, vertex_count, dtype=np.int64
        )
        if np.intersect1d(crown_ids_np, gum_ids_np).size:
            raise ValueError(
                "Current generated teeth_crowns and gums vertex regions overlap"
            )
        if not np.array_equal(
            np.union1d(crown_ids_np, gum_ids_np), expected_dental_ids
        ):
            raise ValueError(
                "Current generated crown/gum regions do not partition every "
                "appended dental vertex"
            )
        if not np.array_equal(current_teeth_ids, expected_dental_ids):
            raise ValueError(
                "Current 'teeth' mask does not cover exactly the generated "
                "dental topology"
            )
        if not np.all(np.isin(oral_vertex_ids_np, gum_ids_np)):
            raise ValueError(
                "Current oral-cavity vertices must be a subset of the zero-label "
                "gum region"
            )

        native_current_topology = reference_seg.shape[0] == vertex_count
        if native_current_topology:
            if not np.array_equal(reference_teeth_ids, crown_ids_np):
                raise ValueError(
                    "Current-topology condition assets must label exactly the "
                    "current teeth_crowns vertices; do not label gums or use "
                    "AnimPortrait3D dental indices"
                )
            # The local package was generated for this exact topology.  Its
            # full per-vertex segmentation can therefore be consumed directly.
            vertex_seg = torch.as_tensor(
                reference_seg / 255.0,
                dtype=torch.float32,
                device=self.device,
            )
            semantic_mapping_mode = "direct-current-topology"
        else:
            # Compatibility path for the original 8009-vertex AnimPortrait3D
            # package: copy the shared base and transfer its enamel code to the
            # named crown region of this repository's different dentition.
            vertex_seg = torch.zeros(
                (vertex_count, 3), dtype=torch.float32, device=self.device
            )
            vertex_seg[:base_vertex_count] = torch.as_tensor(
                reference_seg[:base_vertex_count] / 255.0,
                dtype=torch.float32,
                device=self.device,
            )
            crown_ids = torch.as_tensor(
                crown_ids_np, dtype=torch.long, device=self.device
            )
            vertex_seg[crown_ids] = torch.as_tensor(
                reference_teeth_colour,
                dtype=torch.float32,
                device=self.device,
            )
            semantic_mapping_mode = "legacy-region-transfer"
        self.vertex_seg = vertex_seg

        oral_face_ids_np = (
            torch.unique(
                self.model.mask.get_fid_by_region("oral_cavity").long()
            )
            .detach()
            .cpu()
            .numpy()
        )
        if (
            oral_face_ids_np.size == 0
            or oral_face_ids_np[0] < 0
            or oral_face_ids_np[-1] >= model_faces_cpu.shape[0]
        ):
            raise ValueError(
                "Current oral-cavity face mask contains invalid face indices"
            )
        oral_face_vertex_ids = np.unique(model_faces_cpu[oral_face_ids_np])
        if not np.all(np.isin(oral_face_vertex_ids, gum_ids_np)):
            raise ValueError(
                "Current oral-cavity faces must contain only zero-label gum "
                "vertices"
            )

        self.semantic_mapping_stats = {
            "mapping_mode": semantic_mapping_mode,
            "asset_vertices": int(reference_seg.shape[0]),
            "asset_teeth_vertices": int(reference_teeth_ids.size),
            "reference_vertices": int(reference_seg.shape[0]),
            "reference_teeth_vertices": int(reference_teeth_ids.size),
            "current_base_vertices": int(base_vertex_count),
            "current_crown_vertices": int(crown_ids_np.size),
            "current_gum_vertices": int(gum_ids_np.size),
            "current_oral_vertices": int(oral_vertex_ids_np.size),
            "teeth_semantic_value": float(reference_teeth_colour[2]),
        }
        threestudio.info(
            "AnimPortrait3D normal+seg topology validated: shared base "
            f"{base_vertex_count}, face region {face_vertex_ids_np.size}v/"
            f"{face_faces_np.shape[0]}f; condition-asset dentition "
            f"{reference_seg.shape[0] - base_vertex_count}v "
            f"({reference_teeth_ids.size} labelled), mode "
            f"{semantic_mapping_mode}; current crowns {crown_ids_np.size}v, gums "
            f"{gum_ids_np.size}v and oral cavity {oral_vertex_ids_np.size}v "
            "remain zero."
        )

        zeros_shape = torch.zeros((1, 300), dtype=torch.float32, device=self.device)
        zeros_expression = torch.zeros(
            (1, 100), dtype=torch.float32, device=self.device
        )
        zeros3 = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        zeros6 = torch.zeros((1, 6), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            neutral = self.model(
                shape=zeros_shape,
                expr=zeros_expression,
                rotation=zeros3,
                neck=zeros3,
                jaw=zeros3,
                eyes=zeros6,
                translation=zeros3,
                zero_centered_at_root_node=False,
                return_landmarks=False,
            ).squeeze(0)
        self.center = ((neutral.amin(0) + neutral.amax(0)) / 2.0).detach()
        self.scale = (
            0.6 / (neutral.amax(0) - neutral.amin(0)).amax()
        ).detach()
        self.alignment = torch.as_tensor(
            assets.facelift_from_training,
            dtype=torch.float32,
            device=self.device,
        )

    def _posed_vertices(self, pose: _Pose) -> torch.Tensor:
        shape = torch.as_tensor(
            self.assets.shape, dtype=torch.float32, device=self.device
        )
        expression = torch.as_tensor(
            pose.expression, dtype=torch.float32, device=self.device
        )
        jaw = torch.as_tensor(
            pose.jaw_pose, dtype=torch.float32, device=self.device
        )
        leye = torch.as_tensor(
            pose.leye_pose, dtype=torch.float32, device=self.device
        )
        reye = torch.as_tensor(
            pose.reye_pose, dtype=torch.float32, device=self.device
        )
        zeros3 = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        vertices = self.model(
            shape=shape,
            expr=expression,
            rotation=zeros3,
            neck=zeros3,
            jaw=jaw,
            eyes=torch.cat([leye, reye], dim=-1),
            translation=zeros3,
            zero_centered_at_root_node=False,
            return_landmarks=False,
        ).squeeze(0)
        vertices = (vertices - self.center) * self.scale
        vertices = vertices.clone()
        vertices[:, [1, 2]] = vertices[:, [2, 1]]
        vertices *= 1.1 ** (-float(self.assets.flame_scale))
        return (
            vertices @ self.alignment[:3, :3].T
            + self.alignment[:3, 3][None]
        )

    def _camera_geometry(
        self,
        vertices: torch.Tensor,
        frames: Sequence[_CameraFrame],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        K = torch.as_tensor(
            np.stack([frame.K for frame in frames]),
            dtype=torch.float32,
            device=self.device,
        )
        w2c = torch.as_tensor(
            np.stack([frame.w2c for frame in frames]),
            dtype=torch.float32,
            device=self.device,
        )
        camera_cv = (
            torch.einsum("bij,nj->bni", w2c[:, :3, :3], vertices)
            + w2c[:, None, :3, 3]
        )
        z = camera_cv[..., 2]
        eps = float(self.cfg.projection_min_depth)
        safe_z = torch.where(
            z.abs() >= eps,
            z,
            torch.where(z < 0.0, -torch.full_like(z, eps), torch.full_like(z, eps)),
        )
        width = float(frames[0].width)
        height = float(frames[0].height)
        u = K[:, None, 0, 0] * camera_cv[..., 0] / safe_z + K[:, None, 0, 2]
        v = K[:, None, 1, 1] * camera_cv[..., 1] / safe_z + K[:, None, 1, 2]
        x_ndc = 2.0 * u / width - 1.0
        y_ndc = 1.0 - 2.0 * v / height
        near, far = 0.1, 10.0
        z_ndc = (far + near) / (far - near) - (
            2.0 * far * near / (far - near)
        ) / safe_z
        clip = torch.stack(
            (x_ndc * z, y_ndc * z, z_ndc * z, z), dim=-1
        )
        # NVDiffRenderer converts OpenCV (x right, y down, z forward) to its
        # camera space (x right, y up, z toward the viewer) before normals.
        camera_gl = camera_cv * camera_cv.new_tensor([1.0, -1.0, -1.0])
        return clip.contiguous(), camera_gl

    @staticmethod
    def _vertex_normals(
        vertices: torch.Tensor, faces: torch.Tensor
    ) -> torch.Tensor:
        faces_long = faces.long()
        v0 = vertices[:, faces_long[:, 0]]
        v1 = vertices[:, faces_long[:, 1]]
        v2 = vertices[:, faces_long[:, 2]]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        normals = torch.zeros_like(vertices)
        for corner in range(3):
            indices = faces_long[:, corner][None, :, None].expand(
                vertices.shape[0], -1, 3
            )
            normals.scatter_add_(1, indices, face_normals)
        valid = normals.square().sum(dim=-1, keepdim=True) > 1.0e-20
        fallback = normals.new_tensor([0.0, 0.0, 1.0]).view(1, 1, 3)
        normals = torch.where(valid, normals, fallback)
        return torch.nn.functional.normalize(normals, dim=-1, eps=1.0e-12)

    @torch.inference_mode()
    def render(
        self,
        pose: _Pose,
        frames: Sequence[_CameraFrame],
    ) -> torch.Tensor:
        if not frames:
            raise ValueError("Cannot render an empty condition batch")
        image_sizes = {(frame.height, frame.width) for frame in frames}
        if len(image_sizes) != 1:
            raise ValueError(
                "AnimPortrait3D condition batches must share one image size"
            )
        height, width = next(iter(image_sizes))
        vertices = self._posed_vertices(pose)
        clip, camera_gl = self._camera_geometry(vertices, frames)

        full_rast, _ = self.dr.rasterize(
            self.glctx, clip, self.full_faces, (height, width)
        )
        batch_seg = self.vertex_seg[None].expand(
            len(frames), -1, -1
        ).contiguous()
        segment, _ = self.dr.interpolate(
            batch_seg, full_rast, self.full_faces
        )
        full_foreground = full_rast[..., 3:4] > 0.0
        segment = torch.where(
            full_foreground, segment, -torch.ones_like(segment)
        )
        seg = segment[..., 2:3]
        # Match helper.py exactly: the raster background is -1 and is changed
        # to zero before concatenation.
        seg = torch.where(seg == -1.0, torch.zeros_like(seg), seg)
        face_clip = clip[:, self.face_vertex_ids]
        face_camera = camera_gl[:, self.face_vertex_ids]
        face_normals = self._vertex_normals(face_camera, self.face_faces)
        face_rast, _ = self.dr.rasterize(
            self.glctx, face_clip, self.face_faces, (height, width)
        )
        normal, _ = self.dr.interpolate(
            face_normals, face_rast, self.face_faces
        )
        normal = torch.nn.functional.normalize(normal, dim=-1, eps=1.0e-12)
        face_foreground = face_rast[..., 3:4] > 0.0
        # black_normal_bg=False in the reference means raw normal background=1.
        normal = torch.where(face_foreground, normal, torch.ones_like(normal))
        normal = normal / 2.0 + 0.5

        # dr.rasterize stores scanlines bottom-to-top.  NVDiffRenderer returns
        # every buffer with flip(1), restoring the image/camera top-to-bottom
        # convention used by the Gaussian render and bbox crops.
        condition = torch.cat((normal, seg), dim=-1).flip(1).clamp(0.0, 1.0)
        if not torch.isfinite(condition).all():
            raise FloatingPointError(
                "AnimPortrait3D normal+seg condition contains non-finite values"
            )
        return condition.to(dtype=torch.float32).contiguous().cpu()


class _BatchBuilder:
    def __init__(
        self,
        assets: _ReconstructionAssets,
        conditioner: Optional[Any],
        cfg: ReconstructionFinetuneDataModuleConfig,
    ) -> None:
        self.assets = assets
        self.conditioner = conditioner
        self.cfg = cfg

    def build(
        self,
        camera_indices: Sequence[int],
        pose: _Pose,
    ) -> Dict[str, Any]:
        if not camera_indices:
            raise ValueError("Cannot build an empty calibrated camera batch")
        frames = [self.assets.frames[index] for index in camera_indices]
        return self._build_frames(frames, pose, camera_indices)

    def build_continuous(
        self,
        frames: Sequence[_CameraFrame],
        pose: _Pose,
    ) -> Dict[str, Any]:
        if not frames:
            raise ValueError("Cannot build an empty continuous camera batch")
        return self._build_frames(frames, pose, None)

    def _build_frames(
        self,
        frames: Sequence[_CameraFrame],
        pose: _Pose,
        camera_indices: Optional[Sequence[int]],
    ) -> Dict[str, Any]:
        batch_size = len(frames)

        c2w = torch.from_numpy(np.stack([frame.c2w for frame in frames])).float()
        w2c = torch.from_numpy(np.stack([frame.w2c for frame in frames])).float()
        K = torch.from_numpy(np.stack([frame.K for frame in frames])).float()
        camera_positions = c2w[:, :3, 3].clone()
        prompt_positions = torch.from_numpy(
            np.stack([frame.prompt_camera_position for frame in frames])
        ).float()
        elevation = torch.tensor(
            [frame.prompt_elevation_deg for frame in frames], dtype=torch.float32
        )
        azimuth = torch.tensor(
            [frame.prompt_azimuth_deg for frame in frames], dtype=torch.float32
        )
        distances = torch.tensor(
            [frame.prompt_distance for frame in frames], dtype=torch.float32
        )
        source_elevation = torch.tensor(
            [frame.source_elevation_deg for frame in frames], dtype=torch.float32
        )
        source_azimuth = torch.tensor(
            [frame.source_azimuth_deg for frame in frames], dtype=torch.float32
        )
        fovx = torch.tensor(
            [
                2.0 * math.atan(frame.width / (2.0 * float(frame.K[0, 0])))
                for frame in frames
            ],
            dtype=torch.float32,
        )
        fovy = torch.tensor(
            [
                2.0 * math.atan(frame.height / (2.0 * float(frame.K[1, 1])))
                for frame in frames
            ],
            dtype=torch.float32,
        )

        if camera_indices is None:
            reference_rgb = None
            reference_alpha = None
        else:
            reference_rgb = self.assets.load_reference_rgb(camera_indices)
            reference_alpha = self.assets.load_reference_alpha(camera_indices)
        if self.conditioner is None:
            condition_channels = (
                4
                if str(self.cfg.condition_type).lower()
                in {"animportrait3d", "animportrait3d_normal_seg", "normal_seg"}
                else 3
            )
            flame_conds = torch.zeros(
                (
                    batch_size,
                    int(self.cfg.height),
                    int(self.cfg.width),
                    condition_channels,
                ),
                dtype=torch.float32,
            )
        else:
            flame_conds = self.conditioner.render(pose, frames)

        expression = torch.from_numpy(pose.expression.copy()).float()
        jaw = torch.from_numpy(pose.jaw_pose.copy()).float()
        leye = torch.from_numpy(pose.leye_pose.copy()).float()
        reye = torch.from_numpy(pose.reye_pose.copy()).float()
        shape = torch.from_numpy(self.assets.shape.copy()).float()
        alignment = torch.from_numpy(
            self.assets.facelift_from_training.copy()
        ).float()
        frame_index = torch.tensor(
            [frame.frame_index for frame in frames], dtype=torch.long
        )
        batch = {
            # Exact first-stage calibrated OpenCV cameras.  No legacy
            # rectification or OpenGL axis conversion is applied.
            "c2w": c2w,
            "w2c": w2c,
            "K": K,
            "fx": K[:, 0, 0].clone(),
            "fy": K[:, 1, 1].clone(),
            "cx": K[:, 0, 2].clone(),
            "cy": K[:, 1, 2].clone(),
            "camera_positions": camera_positions,
            "camera_positions_training": prompt_positions,
            "light_positions": camera_positions.clone(),
            "fovx": fovx,
            "fovy": fovy,
            "height": int(self.cfg.height),
            "width": int(self.cfg.width),
            # Prompt angles/distances live in the inverse-aligned training
            # FLAME coordinate system (front=+Y~=90 degrees).
            "elevation": elevation,
            "azimuth": azimuth,
            "camera_distances": distances,
            "source_elevation": source_elevation,
            "source_azimuth": source_azimuth,
            "frame_index": frame_index,
            "index": frame_index.clone(),
            # Static FaceLift targets are added below only for calibrated
            # cameras. Continuous cameras use a same-view immutable Gaussian
            # snapshot inside the system, avoiding target/camera mismatch.
            "flame_conds": flame_conds,
            "expression": expression,
            "jaw_pose": jaw,
            "leye_pose": leye,
            "reye_pose": reye,
            "shape": shape,
            "facelift_from_training": alignment,
            "flame_scale": torch.tensor(
                self.assets.flame_scale, dtype=torch.float32
            ),
            "pose_index": torch.full(
                (batch_size,), pose.source_index, dtype=torch.long
            ),
            "is_open_mouth": torch.full(
                (batch_size,), pose.is_open_mouth, dtype=torch.bool
            ),
            "is_reference_pose": torch.full(
                (batch_size,), pose.is_reference, dtype=torch.bool
            ),
            "is_lower_face": torch.zeros((batch_size,), dtype=torch.bool),
        }
        if reference_rgb is not None:
            batch["reference_rgb"] = reference_rgb
            batch["rgb"] = reference_rgb
        if reference_alpha is not None:
            batch["reference_alpha"] = reference_alpha
        if camera_indices is None:
            batch["continuous_camera"] = torch.ones(
                (batch_size,), dtype=torch.bool
            )
        return batch


class ReconstructionFinetuneIterableDataset(IterableDataset):
    def __init__(
        self,
        assets: _ReconstructionAssets,
        builder: _BatchBuilder,
        cfg: ReconstructionFinetuneDataModuleConfig,
    ) -> None:
        super().__init__()
        self.assets = assets
        self.builder = builder
        self.cfg = cfg

    def __iter__(self):
        while True:
            yield {}

    def collate(self, _: Any) -> Dict[str, Any]:
        if self.assets.train_camera_sampling == "calibrated_continuous":
            frames = self.assets.sample_continuous_camera_frames(
                int(self.cfg.batch_size)
            )
            return self.builder.build_continuous(
                frames, self.assets.sample_pose()
            )
        if bool(self.cfg.surface_consistent_batch):
            camera_indices = self.assets.sample_surface_camera_indices(
                int(self.cfg.batch_size)
            )
        else:
            camera_indices = self.assets.sample_camera_indices(
                int(self.cfg.batch_size)
            )
        source_elevations = {
            round(self.assets.frames[index].source_elevation_deg, 6)
            for index in camera_indices
        }
        if len(source_elevations) != 1:
            raise RuntimeError(
                "Internal camera sampler error: a training batch crossed "
                f"elevation rings ({sorted(source_elevations)})"
            )
        return self.builder.build(camera_indices, self.assets.sample_pose())

class ReconstructionFinetuneEvalDataset(Dataset):
    def __init__(
        self,
        assets: _ReconstructionAssets,
        builder: _BatchBuilder,
        cfg: ReconstructionFinetuneDataModuleConfig,
        split: str,
    ) -> None:
        super().__init__()
        self.assets = assets
        self.builder = builder
        self.cfg = cfg
        self.split = split
        self.is_dynamic_test = split == "test"
        if self.is_dynamic_test:
            if int(cfg.eval_batch_size) != 1:
                raise ValueError(
                    "Dynamic test rendering requires eval_batch_size=1 so every "
                    "video frame uses its matching FLAME parameters"
                )
            return

        requested = int(cfg.n_val_views)
        if requested <= 0:
            requested = len(assets.frames)
        count = min(requested, len(assets.frames))
        if requested > len(assets.frames):
            threestudio.warn(
                f"Requested {requested} {split} views but cameras.json contains "
                f"{len(assets.frames)}; using every calibrated view once."
            )
        if count == len(assets.frames):
            self.camera_indices = list(range(len(assets.frames)))
        else:
            self.camera_indices = np.linspace(
                0, len(assets.frames) - 1, count, dtype=np.int64
            ).tolist()
        if assets.test_camera_index not in self.camera_indices:
            self.camera_indices[len(self.camera_indices) // 2] = (
                assets.test_camera_index
            )

        mode = str(cfg.eval_pose_mode).lower()
        if mode == "reference":
            self.pose = assets.reference_pose
        elif mode == "chemistry":
            self.pose = assets.chemistry_pose(int(cfg.eval_chemistry_index))
        elif mode == "open_mouth":
            self.pose = assets.validation_pose
        else:
            raise ValueError(
                "eval_pose_mode must be 'reference', 'chemistry', or "
                "'open_mouth', "
                f"got '{cfg.eval_pose_mode}'"
            )

    def __len__(self) -> int:
        if self.is_dynamic_test:
            return self.assets.test_frame_count
        return len(self.camera_indices)

    def __getitem__(self, index: int) -> int:
        if self.is_dynamic_test:
            return int(index)
        return int(self.camera_indices[index])

    def collate(self, indices: Sequence[int]) -> Dict[str, Any]:
        if self.is_dynamic_test:
            sequence_index = int(indices[0])
            batch = self.builder.build(
                [self.assets.test_camera_index],
                self.assets.test_pose(sequence_index),
            )
            batch["sequence_index"] = torch.tensor(
                [sequence_index], dtype=torch.long
            )
            return batch
        return self.builder.build(
            [int(index) for index in indices], self.pose
        )


@register("reconstruction-finetune-datamodule")
class ReconstructionFinetuneDataModule(pl.LightningDataModule):
    cfg: ReconstructionFinetuneDataModuleConfig

    def __init__(
        self, cfg: Optional[DictConfig] = None
    ) -> None:
        super().__init__()
        self.cfg = parse_structured(ReconstructionFinetuneDataModuleConfig, cfg)
        if int(self.cfg.num_workers) != 0:
            raise ValueError(
                "reconstruction-finetune-datamodule requires num_workers=0: "
                "the shared calibrated FLAME condition generator must remain in "
                "the trainer process"
            )
        self.assets: Optional[_ReconstructionAssets] = None
        self.conditioner: Optional[Any] = None
        self.builder: Optional[_BatchBuilder] = None

    def prepare_data(self) -> None:
        # Files are read and validated in setup so distributed ranks each build
        # their own correctly-local CUDA FLAME condition generator.
        return None

    def _ensure_assets(self) -> None:
        if self.assets is not None:
            return
        self.assets = _ReconstructionAssets(self.cfg)
        enabled = (
            bool(self.cfg.use_mediapipe_condition)
            if self.cfg.use_condition is None
            else bool(self.cfg.use_condition)
        )
        condition_type = str(self.cfg.condition_type).strip().lower()
        if not enabled:
            self.conditioner = None
        elif condition_type in {"mediapipe", "mediapipe_landmarks"}:
            self.conditioner = _MediapipeConditioner(self.assets, self.cfg)
        elif condition_type in {
            "animportrait3d",
            "animportrait3d_normal_seg",
            "normal_seg",
        }:
            self.conditioner = _AnimPortrait3DNormalSegConditioner(
                self.assets, self.cfg
            )
        else:
            raise ValueError(
                "condition_type must be 'mediapipe' or "
                f"'animportrait3d_normal_seg', got {self.cfg.condition_type!r}"
            )
        self.builder = _BatchBuilder(self.assets, self.conditioner, self.cfg)

    def setup(self, stage: Optional[str] = None) -> None:
        self._ensure_assets()
        assert self.assets is not None and self.builder is not None
        if stage in (None, "fit"):
            self.train_dataset = ReconstructionFinetuneIterableDataset(
                self.assets, self.builder, self.cfg
            )
        if stage in (None, "fit", "validate"):
            self.val_dataset = ReconstructionFinetuneEvalDataset(
                self.assets, self.builder, self.cfg, "val"
            )
        if stage in (None, "test", "predict"):
            self.test_dataset = ReconstructionFinetuneEvalDataset(
                self.assets, self.builder, self.cfg, "test"
            )

    @staticmethod
    def _loader(
        dataset: Any,
        batch_size: Optional[int],
        collate_fn: Any,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(
            self.train_dataset,
            batch_size=None,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return self._loader(
            self.val_dataset,
            batch_size=int(self.cfg.eval_batch_size),
            collate_fn=self.val_dataset.collate,
        )

    def test_dataloader(self) -> DataLoader:
        return self._loader(
            self.test_dataset,
            batch_size=int(self.cfg.eval_batch_size),
            collate_fn=self.test_dataset.collate,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
