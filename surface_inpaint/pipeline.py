"""Surface-coherent coarse-to-fine refinement for reconstructed UVD avatars.

This is the second stage after ``train_reconstruction.py``.  It borrows the
coarse-to-fine idea from AvatarMakeup (arXiv:2507.02419), but uses the
canonical UV already stored by every Gaussian:

1. Sanitize covariance streaks over a representative FLAME pose envelope.
2. Jointly refine detached multi-view renders with surface-aware denoising.
3. Fuse Base into one face atlas and four semantic oral residual atlases.
4. Train Base with arbitrary atlas queries plus matched direct teacher views.
5. Train Detail only against exact same-pose/same-camera teacher RGB/masks.
6. Fit only Gaussian appearance/opacity and preserve Stage-1 identity.

After the deterministic Stage-0 sanitation, UV, face binding, normal offset,
scale, rotation, shape, and topology stay frozen.  There is no densification
or pruning in the optimization stages.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import random
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm.auto import tqdm

from .attention import (
    SurfaceAttentionConfig,
    SurfaceAttentionController,
    install_surface_attention,
)
from .detail_bank import DetailTargetBank, DetailTargetBankWriter
from .layered_surface import (
    SURFACE_LAYER_NAMES,
    LayerSurfaceBuffers,
    compose_layered_surface,
    decode_surface_rgb_residual,
    encode_surface_rgb_residual,
    normalize_alpha_weighted,
)
from .stability import stabilize_uvd_covariances


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


def _parse_override_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _set_by_dotted_key(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cursor = config
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def apply_overrides(
    config: dict[str, Any], overrides: Sequence[str]
) -> dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"Invalid override {item!r}; expected a dotted key and value"
            )
        key, raw = item.split("=", 1)
        _set_by_dotted_key(config, key, _parse_override_value(raw))
    return config


def load_config(path: Path, overrides: Sequence[str]) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: the top-level YAML value must be a mapping")
    return apply_overrides(loaded, overrides)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def output_directory(config: Mapping[str, Any]) -> Path:
    root = resolve_path(config["output"]["root"])
    name = str(config["output"].get("name", "")).strip()
    if not name:
        name = Path(config["input"]["reconstruction_dir"]).name
    return root / name


def validate_sh0_ply(path: Path) -> None:
    properties: list[str] = []
    found_end = False
    with path.open("rb") as file:
        for _ in range(10000):
            raw = file.readline()
            if not raw:
                break
            line = raw.decode("ascii", errors="strict").strip()
            if line == "end_header":
                found_end = True
                break
            if line.startswith("property "):
                properties.append(line.split()[-1])
    if not found_end:
        raise ValueError(f"{path}: malformed PLY header")
    rest = [name for name in properties if name.startswith("f_rest_")]
    if rest:
        raise ValueError(
            f"{path}: Stage-2 supports SH degree 0 only, but the PLY contains "
            f"{len(rest)} f_rest properties"
        )


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section {key!r} must be a mapping")
    return value


def validate_config(config: Mapping[str, Any], check_files: bool = True) -> None:
    for section in (
        "input",
        "output",
        "pose_control",
        "stage_outputs",
        "data",
        "stability",
        "teacher",
        "detail_supervision",
        "surface_attention",
        "fusion",
        "optimization",
        "loss",
        "checkpoint",
        "test_render",
    ):
        _require_mapping(config, section)

    device = str(config.get("device", "cuda"))
    if device not in {"cuda", "cuda:0"}:
        raise ValueError(
            "This Stage-2 pipeline currently requires device 'cuda' or "
            "'cuda:0' because ControlNet follows LOCAL_RANK"
        )

    pose_control = config["pose_control"]
    forbidden_motion = [
        key
        for key in (
            "use_global_orient",
            "use_neck_pose",
            "use_translation",
        )
        if bool(pose_control.get(key, False))
    ]
    if forbidden_motion:
        raise ValueError(
            "This pipeline deliberately excludes global/neck/translation "
            "motion; these pose_control flags must remain false: "
            + ", ".join(forbidden_motion)
        )

    input_cfg = config["input"]
    reconstruction_dir = resolve_path(input_cfg["reconstruction_dir"])
    required = (
        reconstruction_dir / "model" / "uvd.ply",
        reconstruction_dir / "model" / "reconstruction_params.npz",
        reconstruction_dir / "resolved_config.yaml",
    )
    if check_files:
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Stage-1 reconstruction is incomplete:\n  " + "\n  ".join(missing)
            )
        validate_sh0_ply(required[0])
        chemistry = resolve_path(config["data"]["chemistry_path"])
        if not chemistry.is_file():
            raise FileNotFoundError(f"Missing chemistry pose file: {chemistry}")
        edit_allow_mask = config["fusion"].get("edit_allow_mask")
        if edit_allow_mask is not None:
            edit_allow_mask_path = resolve_path(edit_allow_mask)
            if not edit_allow_mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing canonical edit-allow mask: {edit_allow_mask_path}"
                )

    height = int(config["data"]["height"])
    width = int(config["data"]["width"])
    if height <= 0 or width <= 0 or height % 8 or width % 8:
        raise ValueError("data.height and data.width must be positive multiples of 8")

    total_steps = int(config["optimization"]["iterations"])
    refresh = int(config["teacher"]["refresh_interval"])
    coarse_refresh = int(
        config["teacher"].get("coarse_refresh_interval", refresh)
    )
    if total_steps <= 0 or refresh <= 0 or coarse_refresh <= 0:
        raise ValueError(
            "optimization.iterations, teacher.refresh_interval and "
            "teacher.coarse_refresh_interval must be positive"
        )
    views_per_pose = int(config["teacher"]["views_per_pose"])
    poses_per_refresh = int(config["teacher"]["poses_per_refresh"])
    coarse_poses_per_refresh = int(
        config["teacher"].get(
            "coarse_poses_per_refresh", poses_per_refresh
        )
    )
    if (
        views_per_pose <= 0
        or poses_per_refresh <= 0
        or coarse_poses_per_refresh <= 0
    ):
        raise ValueError(
            "teacher.views_per_pose, teacher.poses_per_refresh and "
            "teacher.coarse_poses_per_refresh must be positive"
        )
    view_sampling = str(
        config["teacher"].get("view_sampling", "horizontal_ring")
    ).lower()
    if view_sampling not in {
        "horizontal_ring",
        "stratified_all_rings",
    }:
        raise ValueError(
            "teacher.view_sampling must be 'horizontal_ring' or "
            "'stratified_all_rings'"
        )
    output = config["output"]
    if "teacher_previews" in output:
        raise ValueError(
            "output.teacher_previews is the removed global preview limit; "
            "use output.save_all_teacher_observations and "
            "output.teacher_previews_per_pose instead"
        )
    previews_per_pose = int(
        output.get("teacher_previews_per_pose", views_per_pose)
    )
    if previews_per_pose <= 0:
        raise ValueError(
            "output.teacher_previews_per_pose must be positive"
        )
    if (
        not bool(output.get("save_all_teacher_observations", True))
        and previews_per_pose > views_per_pose
    ):
        raise ValueError(
            "output.teacher_previews_per_pose cannot exceed "
            "teacher.views_per_pose"
        )
    if int(output.get("teacher_contact_sheet_tile_size", 160)) < 64:
        raise ValueError(
            "output.teacher_contact_sheet_tile_size must be at least 64"
        )
    coarse_steps = int(config["teacher"].get("coarse_iterations", 0))
    if not 0 < coarse_steps < total_steps:
        raise ValueError(
            "teacher.coarse_iterations must be in "
            "(0, optimization.iterations) because both Base and Detail "
            "stages are required"
        )

    timestep_min = int(config["teacher"]["timestep_min"])
    timestep_max = int(config["teacher"]["timestep_max"])
    if not 0 <= timestep_min <= timestep_max < 1000:
        raise ValueError("teacher timesteps must satisfy 0 <= min <= max < 1000")
    inference_steps = int(config["teacher"]["num_inference_steps"])
    if inference_steps <= 0:
        raise ValueError("teacher.num_inference_steps must be positive")
    if timestep_min < inference_steps:
        raise ValueError(
            "teacher.timestep_min must be >= teacher.num_inference_steps "
            "so the custom DDIM schedule has distinct effective transitions"
        )

    resolution = int(config["fusion"]["resolution"])
    if resolution < 64:
        raise ValueError("fusion.resolution must be at least 64")
    alpha_threshold = float(config["fusion"]["alpha_threshold"])
    if not 0.0 <= alpha_threshold <= 1.0:
        raise ValueError("fusion.alpha_threshold must be in [0, 1]")
    uv_variance_threshold = float(
        config["fusion"].get("uv_variance_threshold", float("inf"))
    )
    if uv_variance_threshold < 0.0:
        raise ValueError("fusion.uv_variance_threshold must be non-negative")
    min_support = int(config["fusion"]["min_view_support"])
    if min_support < 1:
        raise ValueError("fusion.min_view_support must be at least one")
    if views_per_pose < min_support:
        raise ValueError(
            "teacher.views_per_pose must be >= fusion.min_view_support"
        )
    layered = _require_mapping(config["fusion"], "layered_surface")
    expected_layers = [
        "lips",
        "teeth_upper",
        "teeth_lower",
        "oral_cavity",
    ]
    if not bool(layered.get("enabled", False)):
        raise ValueError(
            "fusion.layered_surface.enabled must be true for this pipeline"
        )
    if list(layered.get("layers", ())) != expected_layers:
        raise ValueError(
            "fusion.layered_surface.layers must be exactly "
            "[lips, teeth_upper, teeth_lower, oral_cavity]"
        )
    if not 0.0 <= float(layered["opacity_floor"]) <= 1.0:
        raise ValueError(
            "fusion.layered_surface.opacity_floor must be in [0,1]"
        )
    if not 0.0 <= float(layered["contribution_threshold"]) <= 1.0:
        raise ValueError(
            "fusion.layered_surface.contribution_threshold must be in [0,1]"
        )
    residual_floor = float(layered["residual_decomposition_floor"])
    if not 0.0 < residual_floor <= 1.0:
        raise ValueError(
            "fusion.layered_surface.residual_decomposition_floor must be "
            "in (0,1]"
        )
    if residual_floor < float(layered["contribution_threshold"]):
        raise ValueError(
            "fusion.layered_surface.residual_decomposition_floor must be "
            "no smaller than contribution_threshold"
        )
    required_effective_layers = list(
        layered["required_effective_layers"]
    )
    if (
        len(set(required_effective_layers))
        != len(required_effective_layers)
        or any(
            name not in expected_layers
            for name in required_effective_layers
        )
    ):
        raise ValueError(
            "fusion.layered_surface.required_effective_layers must contain "
            "unique configured oral layer names"
        )
    if int(layered["minimum_effective_gaussians"]) < 1:
        raise ValueError(
            "fusion.layered_surface.minimum_effective_gaussians must be "
            "positive"
        )
    if float(layered["dominance_ratio"]) < 1.0:
        raise ValueError(
            "fusion.layered_surface.dominance_ratio must be >= 1"
        )
    history_weight = float(config["fusion"].get("history_weight", 0.5))
    confidence_decay = float(
        config["fusion"].get("confidence_decay", 0.98)
    )
    edit_decay = float(config["fusion"].get("edit_decay", 0.95))
    if history_weight < 0.0:
        raise ValueError("fusion.history_weight must be non-negative")
    if not 0.0 <= confidence_decay <= 1.0:
        raise ValueError("fusion.confidence_decay must be in [0, 1]")
    if not 0.0 <= edit_decay <= 1.0:
        raise ValueError("fusion.edit_decay must be in [0, 1]")
    feature_lr = float(config["optimization"]["feature_lr"])
    opacity_lr = float(config["optimization"]["opacity_lr"])
    if feature_lr <= 0.0 or opacity_lr < 0.0:
        raise ValueError(
            "optimization.feature_lr must be positive and opacity_lr non-negative"
        )
    max_opacity_increase = float(
        config["optimization"].get("max_opacity_increase", 1.0)
    )
    if not 0.0 <= max_opacity_increase <= 1.0:
        raise ValueError("optimization.max_opacity_increase must be in [0, 1]")
    oral_max_opacity_increase = float(
        config["optimization"].get(
            "oral_max_opacity_increase", max_opacity_increase
        )
    )
    if not 0.0 <= oral_max_opacity_increase <= 1.0:
        raise ValueError(
            "optimization.oral_max_opacity_increase must be in [0, 1]"
        )

    stability = config["stability"]
    if int(stability.get("passes", 1)) <= 0:
        raise ValueError("stability.passes must be positive")
    if float(stability["absolute_max_scale"]) <= 0.0:
        raise ValueError("stability.absolute_max_scale must be positive")
    if float(stability["min_streak_scale"]) < 0.0:
        raise ValueError("stability.min_streak_scale must be non-negative")
    if float(stability["max_planar_aspect"]) < 1.0:
        raise ValueError("stability.max_planar_aspect must be at least one")
    repair_margin = float(stability.get("repair_margin", 1.0))
    if not 0.0 < repair_margin <= 1.0:
        raise ValueError("stability.repair_margin must be in (0, 1]")
    pose_quantiles = list(stability.get("pose_quantiles", ()))
    if not pose_quantiles or any(
        not 0.0 <= float(value) <= 1.0 for value in pose_quantiles
    ):
        raise ValueError(
            "stability.pose_quantiles must contain values in [0, 1]"
        )

    attention = config["surface_attention"]
    SurfaceAttentionConfig(
        atlas_resolution=int(attention["atlas_resolution"]),
        max_tokens=int(attention["max_tokens"]),
        min_views=int(attention["min_views"]),
        strength=float(attention["strength"]),
    )
    attention_alpha = float(attention.get("alpha_threshold", 0.0))
    if not 0.0 <= attention_alpha <= 1.0:
        raise ValueError("surface_attention.alpha_threshold must be in [0, 1]")
    layer_contribution_threshold = float(
        layered["contribution_threshold"]
    )
    if (
        bool(attention.get("enabled", True))
        and attention_alpha > layer_contribution_threshold
    ):
        raise ValueError(
            "surface_attention.alpha_threshold must be no greater than "
            "fusion.layered_surface.contribution_threshold; otherwise "
            "accepted teeth/cavity correspondences are discarded before "
            "denoising attention"
        )
    if float(attention.get("uv_jump_threshold", 0.0)) < 0.0:
        raise ValueError(
            "surface_attention.uv_jump_threshold must be non-negative"
        )
    if float(attention.get("uv_variance_threshold", 0.0)) < 0.0:
        raise ValueError(
            "surface_attention.uv_variance_threshold must be non-negative"
        )
    if bool(attention.get("enabled", True)) and int(
        config["teacher"]["batch_size"]
    ) < int(attention["min_views"]):
        raise ValueError(
            "teacher.batch_size must be >= surface_attention.min_views"
        )

    highpass_kernel = int(config["loss"].get("highpass_kernel", 5))
    if highpass_kernel <= 0 or highpass_kernel % 2 == 0:
        raise ValueError("loss.highpass_kernel must be a positive odd integer")
    if float(config["loss"].get("edit_highpass_weight", 0.0)) < 0.0:
        raise ValueError("loss.edit_highpass_weight must be non-negative")
    if float(config["loss"].get("layered_oral_weight", 0.0)) < 0.0:
        raise ValueError("loss.layered_oral_weight must be non-negative")

    detail = config["detail_supervision"]
    if str(detail.get("mode", "")).lower() != "direct_teacher":
        raise ValueError(
            "detail_supervision.mode must be 'direct_teacher'"
        )
    for key in ("base_direct_probability", "open_mouth_probability"):
        value = float(detail[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"detail_supervision.{key} must be in [0,1]")
    direct_batch_size = int(config["data"]["batch_size"])
    if not 0 < direct_batch_size <= views_per_pose:
        raise ValueError(
            "data.batch_size must be positive and no greater than "
            "teacher.views_per_pose for same-pose direct sampling"
        )

    test_render = config["test_render"]
    if bool(test_render.get("enabled", True)):
        for key in ("height", "width", "fps", "contact_sheet_frames"):
            if int(test_render[key]) <= 0:
                raise ValueError(f"test_render.{key} must be positive")

    stage_outputs = config["stage_outputs"]
    for key in (
        "diagnostic_views",
        "comparison_frames",
    ):
        if int(stage_outputs[key]) <= 0:
            raise ValueError(f"stage_outputs.{key} must be positive")
    stage_pose_quantiles = list(
        stage_outputs.get("diagnostic_pose_quantiles", ())
    )
    if not stage_pose_quantiles or any(
        not 0.0 <= float(value) <= 1.0
        for value in stage_pose_quantiles
    ):
        raise ValueError(
            "stage_outputs.diagnostic_pose_quantiles must contain values "
            "in [0, 1]"
        )


# -----------------------------------------------------------------------------
# Small tensor and image helpers
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in batch.items():
        output[key] = value.to(device) if torch.is_tensor(value) else value
    return output


def rescale_render_batch(
    batch: Mapping[str, Any], height: int, width: int
) -> dict[str, Any]:
    """Scale calibrated camera intrinsics for a different raster size."""

    source_height = int(torch.as_tensor(batch["height"]).reshape(-1)[0])
    source_width = int(torch.as_tensor(batch["width"]).reshape(-1)[0])
    height, width = int(height), int(width)
    if height <= 0 or width <= 0:
        raise ValueError("Render resolution must be positive")
    scaled = dict(batch)
    if (height, width) == (source_height, source_width):
        return scaled
    intrinsic = torch.as_tensor(batch["K"]).clone()
    intrinsic[..., 0, :] *= float(width) / float(source_width)
    intrinsic[..., 1, :] *= float(height) / float(source_height)
    scaled.update(
        {
            "K": intrinsic,
            "fx": intrinsic[..., 0, 0].clone(),
            "fy": intrinsic[..., 1, 1].clone(),
            "cx": intrinsic[..., 0, 2].clone(),
            "cy": intrinsic[..., 1, 2].clone(),
            "height": height,
            "width": width,
        }
    )
    return scaled


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_digest() -> str:
    """Hash the executable surface-coherent implementation files."""

    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("attention.py"),
        Path(__file__).resolve().with_name("detail_bank.py"),
        Path(__file__).resolve().with_name("layered_surface.py"),
        Path(__file__).resolve().with_name("stability.py"),
        PROJECT_ROOT / "loop_inpaint.py",
        PROJECT_ROOT / "threestudio" / "data" / "reconstruction_finetune.py",
        PROJECT_ROOT
        / "gaussiansplatting"
        / "scene"
        / "gaussian_flame_face.py",
        PROJECT_ROOT / "gaussiansplatting" / "gaussian_renderer" / "__init__.py",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def as_bchw(value: Any, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected a 4D image batch, got {tuple(tensor.shape)}")
    if tensor.shape[-1] in (1, 3, 4):
        tensor = tensor.permute(0, 3, 1, 2)
    if tensor.max().detach() > 1.5:
        tensor = tensor / 255.0
    return tensor.contiguous()


def masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(device=prediction.device, dtype=prediction.dtype)
    if mask.shape[-2:] != prediction.shape[-2:]:
        mask = F.interpolate(
            mask, prediction.shape[-2:], mode="bilinear", align_corners=False
        )
    if mask.shape[1] == 1 and prediction.shape[1] != 1:
        mask = mask.expand(-1, prediction.shape[1], -1, -1)
    numerator = ((prediction - target).abs() * mask).sum()
    return numerator / mask.sum().clamp_min(eps)


def lowpass(image: torch.Tensor, resolution: int, kernel: int) -> torch.Tensor:
    kernel = max(int(kernel), 1)
    if kernel % 2 == 0:
        kernel += 1
    if kernel > 1:
        image = F.avg_pool2d(
            image, kernel_size=kernel, stride=1, padding=kernel // 2
        )
    return F.interpolate(
        image,
        size=(int(resolution), int(resolution)),
        mode="bilinear",
        align_corners=False,
    )


def resize_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if mask.shape[-2:] == like.shape[-2:]:
        return mask
    return F.interpolate(
        mask, size=like.shape[-2:], mode="bilinear", align_corners=False
    )


def rgb_u8(image: torch.Tensor) -> np.ndarray:
    image = image.detach().clamp(0.0, 1.0)
    if image.ndim == 4:
        image = image[0]
    return (
        image.permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .cpu()
        .numpy()
    )


def gray_u8(image: torch.Tensor) -> np.ndarray:
    image = image.detach().squeeze().clamp(0.0, 1.0)
    return image.mul(255.0).round().byte().cpu().numpy()


def surface_layer_u8(layer_ids: torch.Tensor) -> np.ndarray:
    """Colorize stable semantic IDs; invalid correspondence is black."""

    ids = layer_ids.detach().squeeze().long().cpu().numpy()
    palette = np.asarray(
        [
            [120, 170, 235],  # face
            [235, 85, 130],   # lips
            [250, 245, 170],  # upper teeth
            [150, 245, 245],  # lower teeth
            [105, 45, 120],   # oral cavity
        ],
        dtype=np.uint8,
    )
    output = np.zeros((*ids.shape, 3), dtype=np.uint8)
    valid = (ids >= 0) & (ids < len(palette))
    output[valid] = palette[ids[valid]]
    return output


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(
        output,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def setup_logger(directory: Path) -> logging.Logger:
    logger = logging.getLogger("loop_inpaint")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(directory / "train.log", mode="a", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# -----------------------------------------------------------------------------
# Canonical UV projection and robust multi-view fusion
# -----------------------------------------------------------------------------


def surface_validity(
    surface_uv: torch.Tensor,
    alpha: torch.Tensor,
    alpha_threshold: float,
    uv_jump_threshold: float,
    uv_variance: Optional[torch.Tensor] = None,
    uv_variance_threshold: float = float("inf"),
) -> torch.Tensor:
    """Reject background and pixels that straddle a UV/occlusion discontinuity."""

    if surface_uv.ndim != 4 or surface_uv.shape[1] != 2:
        raise ValueError("surface_uv must have shape [B, 2, H, W]")
    if alpha.ndim == 3:
        alpha = alpha.unsqueeze(1)
    valid = (
        (alpha > float(alpha_threshold))
        & torch.isfinite(surface_uv).all(dim=1, keepdim=True)
        & (surface_uv >= 0.0).all(dim=1, keepdim=True)
        & (surface_uv <= 1.0).all(dim=1, keepdim=True)
    )
    if uv_variance is not None:
        if uv_variance.ndim == 3:
            uv_variance = uv_variance.unsqueeze(1)
        if uv_variance.shape != alpha.shape:
            raise ValueError("uv_variance and alpha must have the same shape")
        valid = (
            valid
            & torch.isfinite(uv_variance)
            & (uv_variance <= float(uv_variance_threshold))
        )
    if uv_jump_threshold <= 0.0:
        return valid.to(surface_uv.dtype)

    jump = torch.zeros_like(valid)
    dx = torch.linalg.vector_norm(
        surface_uv[:, :, :, 1:] - surface_uv[:, :, :, :-1], dim=1, keepdim=True
    )
    dy = torch.linalg.vector_norm(
        surface_uv[:, :, 1:, :] - surface_uv[:, :, :-1, :], dim=1, keepdim=True
    )
    jump_x = dx > float(uv_jump_threshold)
    jump_y = dy > float(uv_jump_threshold)
    jump[:, :, :, 1:] |= jump_x
    jump[:, :, :, :-1] |= jump_x
    jump[:, :, 1:, :] |= jump_y
    jump[:, :, :-1, :] |= jump_y
    return (valid & ~jump).to(surface_uv.dtype)


def screen_center_weight(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    power: float,
) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius2 = xx.square() + yy.square()
    weight = (1.0 - radius2.clamp(max=1.0)).clamp_min(0.0)
    return weight.pow(float(power)).view(1, 1, height, width)


def splat_to_uv(
    values: torch.Tensor,
    surface_uv: torch.Tensor,
    weights: torch.Tensor,
    resolution: int,
    flip_v: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinearly scatter image values into one shared canonical atlas."""

    if values.ndim != 4 or surface_uv.ndim != 4 or weights.ndim != 4:
        raise ValueError("values, surface_uv, and weights must be BCHW tensors")
    if surface_uv.shape[1] != 2 or weights.shape[1] != 1:
        raise ValueError("surface_uv must have 2 and weights exactly 1 channel")
    if values.shape[0] != surface_uv.shape[0] or values.shape[2:] != surface_uv.shape[2:]:
        raise ValueError("values and surface_uv batch/spatial shapes differ")

    batch, channels, _, _ = values.shape
    resolution = int(resolution)
    uv = surface_uv.detach().clamp(0.0, 1.0).clone()
    if flip_v:
        uv[:, 1] = 1.0 - uv[:, 1]

    u = uv[:, 0].reshape(-1)
    v = uv[:, 1].reshape(-1)
    pixel_weight = weights.detach().reshape(-1)
    color = values.detach().permute(0, 2, 3, 1).reshape(-1, channels)

    finite = (
        torch.isfinite(u)
        & torch.isfinite(v)
        & torch.isfinite(pixel_weight)
        & torch.isfinite(color).all(dim=1)
        & (pixel_weight > 0.0)
    )
    if not finite.any():
        return (
            torch.zeros(
                channels,
                resolution,
                resolution,
                device=values.device,
                dtype=values.dtype,
            ),
            torch.zeros(
                1,
                resolution,
                resolution,
                device=values.device,
                dtype=values.dtype,
            ),
        )

    u, v = u[finite], v[finite]
    pixel_weight, color = pixel_weight[finite], color[finite]
    x = u * (resolution - 1)
    y = v * (resolution - 1)
    x0 = x.floor().long().clamp(0, resolution - 1)
    y0 = y.floor().long().clamp(0, resolution - 1)
    x1 = (x0 + 1).clamp(0, resolution - 1)
    y1 = (y0 + 1).clamp(0, resolution - 1)
    dx = x - x0.to(x.dtype)
    dy = y - y0.to(y.dtype)

    atlas_sum = torch.zeros(
        channels,
        resolution * resolution,
        device=values.device,
        dtype=values.dtype,
    )
    atlas_weight = torch.zeros(
        1,
        resolution * resolution,
        device=values.device,
        dtype=values.dtype,
    )
    contributions = (
        (y0 * resolution + x0, (1.0 - dx) * (1.0 - dy)),
        (y1 * resolution + x0, (1.0 - dx) * dy),
        (y0 * resolution + x1, dx * (1.0 - dy)),
        (y1 * resolution + x1, dx * dy),
    )
    for flat_index, bilinear_weight in contributions:
        contribution = pixel_weight * bilinear_weight
        atlas_sum.scatter_add_(
            1,
            flat_index.unsqueeze(0).expand(channels, -1),
            (color * contribution.unsqueeze(1)).transpose(0, 1),
        )
        atlas_weight.scatter_add_(
            1, flat_index.unsqueeze(0), contribution.unsqueeze(0)
        )

    return (
        atlas_sum.view(channels, resolution, resolution),
        atlas_weight.view(1, resolution, resolution),
    )


def sample_uv_atlas(
    atlas: torch.Tensor,
    surface_uv: torch.Tensor,
    flip_v: bool = False,
) -> torch.Tensor:
    if atlas.ndim == 3:
        atlas = atlas.unsqueeze(0)
    if atlas.ndim != 4 or surface_uv.ndim != 4:
        raise ValueError("atlas and surface_uv must be BCHW tensors")
    uv = surface_uv.detach().clamp(0.0, 1.0).clone()
    if flip_v:
        uv[:, 1] = 1.0 - uv[:, 1]
    grid = uv.permute(0, 2, 3, 1) * 2.0 - 1.0
    if atlas.shape[0] == 1 and surface_uv.shape[0] > 1:
        atlas = atlas.expand(surface_uv.shape[0], -1, -1, -1)
    return F.grid_sample(
        atlas,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


@dataclass
class UVAtlas:
    rgb: torch.Tensor
    confidence: torch.Tensor
    edit: torch.Tensor
    support: torch.Tensor
    variance: torch.Tensor
    refresh_step: int
    teacher_timestep: int

    def state_dict(self) -> dict[str, Any]:
        return {
            "rgb": self.rgb.detach().cpu(),
            "confidence": self.confidence.detach().cpu(),
            "edit": self.edit.detach().cpu(),
            "support": self.support.detach().cpu(),
            "variance": self.variance.detach().cpu(),
            "refresh_step": int(self.refresh_step),
            "teacher_timestep": int(self.teacher_timestep),
        }

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any], device: torch.device
    ) -> "UVAtlas":
        return cls(
            rgb=torch.as_tensor(state["rgb"], device=device).float(),
            confidence=torch.as_tensor(state["confidence"], device=device).float(),
            edit=torch.as_tensor(state["edit"], device=device).float(),
            support=torch.as_tensor(state["support"], device=device).float(),
            variance=torch.as_tensor(state["variance"], device=device).float(),
            refresh_step=int(state["refresh_step"]),
            teacher_timestep=int(state["teacher_timestep"]),
        )


class UVAtlasAccumulator:
    """Store per-view observations on CPU and perform two-pass Huber fusion."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.resolution = int(config["resolution"])
        self.observations: list[torch.Tensor] = []
        self.coverages: list[torch.Tensor] = []
        self.view_ids: list[int] = []
        self._next_auto_view_id = 0

    @property
    def num_views(self) -> int:
        return len(self.observations)

    @torch.inference_mode()
    def add(
        self,
        teacher_rgb: torch.Tensor,
        surface_uv: torch.Tensor,
        surface_alpha: torch.Tensor,
        edit: torch.Tensor,
        composite_alpha: Optional[torch.Tensor] = None,
        view_ids: Optional[Sequence[int]] = None,
        surface_uv_variance: Optional[torch.Tensor] = None,
    ) -> None:
        if teacher_rgb.shape[0] != surface_uv.shape[0]:
            raise ValueError("Teacher and UV batches have different sizes")
        if composite_alpha is None:
            composite_alpha = surface_alpha
        composite_alpha = composite_alpha.detach().clamp(0.0, 1.0)
        visibility_alpha = torch.minimum(
            surface_alpha.detach().clamp(0.0, 1.0), composite_alpha
        )
        batch, _, height, width = teacher_rgb.shape
        if view_ids is None:
            resolved_view_ids = list(
                range(self._next_auto_view_id, self._next_auto_view_id + batch)
            )
            self._next_auto_view_id += batch
        else:
            resolved_view_ids = [int(value) for value in view_ids]
            if len(resolved_view_ids) != batch:
                raise ValueError("view_ids must contain one ID per image")
        validity = surface_validity(
            surface_uv,
            visibility_alpha,
            float(self.config["alpha_threshold"]),
            float(self.config["uv_jump_threshold"]),
            surface_uv_variance,
            float(self.config.get("uv_variance_threshold", float("inf"))),
        )
        center = screen_center_weight(
            height,
            width,
            teacher_rgb.device,
            teacher_rgb.dtype,
            float(self.config["screen_center_power"]),
        )
        weights = (
            visibility_alpha.pow(
                float(self.config["alpha_weight_power"])
            )
            * validity
            * center
        )
        background = torch.ones(
            1, 3, 1, 1, device=teacher_rgb.device, dtype=teacher_rgb.dtype
        )
        surface_rgb = (
            teacher_rgb.detach() - (1.0 - composite_alpha) * background
        ) / composite_alpha.clamp_min(1.0e-4)
        surface_rgb = torch.nan_to_num(
            surface_rgb, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        fields = torch.cat((surface_rgb, edit.detach()), dim=1)

        for index in range(batch):
            atlas_sum, atlas_weight = splat_to_uv(
                fields[index : index + 1],
                surface_uv[index : index + 1],
                weights[index : index + 1],
                self.resolution,
                bool(self.config.get("flip_v", False)),
            )
            observed = atlas_weight > 0.0
            atlas_value = atlas_sum / atlas_weight.clamp_min(1.0e-8)
            positive = atlas_weight[observed]
            if positive.numel() == 0:
                continue
            quantile = float(self.config["coverage_normalization_quantile"])
            scale = torch.quantile(positive.float(), quantile).clamp_min(1.0e-6)
            coverage = (atlas_weight / scale).clamp(0.0, 1.0)
            coverage = coverage * observed.to(coverage.dtype)
            self.observations.append(atlas_value.half().cpu())
            self.coverages.append(coverage.half().cpu())
            self.view_ids.append(resolved_view_ids[index])

    @torch.inference_mode()
    def finalize(
        self,
        device: torch.device,
        refresh_step: int,
        teacher_timestep: int,
    ) -> UVAtlas:
        if not self.observations:
            raise RuntimeError("No valid UV observations were collected")

        resolution = self.resolution
        rgb_sum = torch.zeros(3, resolution, resolution, device=device)
        edit_sum = torch.zeros(1, resolution, resolution, device=device)
        weight_sum = torch.zeros(1, resolution, resolution, device=device)
        chunk_size = max(int(self.config["fusion_chunk_size"]), 1)

        for start in range(0, self.num_views, chunk_size):
            values = torch.stack(
                self.observations[start : start + chunk_size]
            ).to(device=device, dtype=torch.float32)
            coverage = torch.stack(
                self.coverages[start : start + chunk_size]
            ).to(device=device, dtype=torch.float32)
            rgb_sum.add_((values[:, :3] * coverage).sum(dim=0))
            edit_sum.add_((values[:, 3:4] * coverage).sum(dim=0))
            weight_sum.add_(coverage.sum(dim=0))
        support_by_view: dict[int, torch.Tensor] = {}
        support_threshold = float(self.config["support_threshold"])
        for view_id, coverage in zip(self.view_ids, self.coverages):
            observed = coverage > support_threshold
            if view_id in support_by_view:
                support_by_view[view_id] |= observed
            else:
                support_by_view[view_id] = observed.clone()
        support = torch.zeros(1, resolution, resolution, device=device)
        support_maps = list(support_by_view.values())
        for start in range(0, len(support_maps), chunk_size):
            visible = torch.stack(
                support_maps[start : start + chunk_size]
            ).to(device=device, dtype=torch.float32)
            support.add_(visible.sum(dim=0))

        initial_rgb = rgb_sum / weight_sum.clamp_min(1.0e-8)
        initial_edit = edit_sum / weight_sum.clamp_min(1.0e-8)

        robust_rgb_sum = torch.zeros_like(rgb_sum)
        robust_edit_sum = torch.zeros_like(edit_sum)
        robust_weight_sum = torch.zeros_like(weight_sum)
        squared_error_sum = torch.zeros_like(weight_sum)
        huber_delta = float(self.config["huber_delta"])

        for start in range(0, self.num_views, chunk_size):
            values = torch.stack(
                self.observations[start : start + chunk_size]
            ).to(device=device, dtype=torch.float32)
            coverage = torch.stack(
                self.coverages[start : start + chunk_size]
            ).to(device=device, dtype=torch.float32)
            error = (
                (values[:, :3] - initial_rgb.unsqueeze(0))
                .square()
                .mean(dim=1, keepdim=True)
                .sqrt()
            )
            huber = torch.where(
                error <= huber_delta,
                torch.ones_like(error),
                huber_delta / error.clamp_min(1.0e-8),
            )
            robust_weight = coverage * huber
            robust_rgb_sum.add_((values[:, :3] * robust_weight).sum(dim=0))
            robust_edit_sum.add_((values[:, 3:4] * robust_weight).sum(dim=0))
            robust_weight_sum.add_(robust_weight.sum(dim=0))

        rgb = robust_rgb_sum / robust_weight_sum.clamp_min(1.0e-8)
        edit = robust_edit_sum / robust_weight_sum.clamp_min(1.0e-8)

        for start in range(0, self.num_views, chunk_size):
            values = torch.stack(
                self.observations[start : start + chunk_size]
            ).to(device=device, dtype=torch.float32)
            coverage = torch.stack(
                self.coverages[start : start + chunk_size]
            ).to(device=device, dtype=torch.float32)
            error2 = (values[:, :3] - rgb.unsqueeze(0)).square().mean(
                dim=1, keepdim=True
            )
            squared_error_sum.add_((error2 * coverage).sum(dim=0))

        variance = squared_error_sum / weight_sum.clamp_min(1.0e-8)
        min_support = float(self.config["min_view_support"])
        support_confidence = (support / max(min_support, 1.0)).clamp(0.0, 1.0)
        variance_confidence = torch.exp(
            -variance / max(float(self.config["variance_scale"]), 1.0e-8)
        )
        confidence = support_confidence * variance_confidence
        confidence = confidence * (support >= min_support).to(confidence.dtype)
        confidence = torch.where(
            robust_weight_sum > 0.0, confidence, torch.zeros_like(confidence)
        )

        fused_edit = initial_edit.lerp(edit, 0.5).clamp(0.0, 1.0)
        fused_edit = torch.where(
            confidence > 0.0, fused_edit, torch.zeros_like(fused_edit)
        )
        return UVAtlas(
            rgb=rgb.clamp(0.0, 1.0),
            confidence=confidence.clamp(0.0, 1.0),
            edit=fused_edit,
            support=support,
            variance=variance,
            refresh_step=int(refresh_step),
            teacher_timestep=int(teacher_timestep),
        )


@torch.inference_mode()
def merge_uv_atlas(
    previous: UVAtlas,
    current: UVAtlas,
    history_weight: float,
    confidence_decay: float,
    edit_decay: float,
    variance_scale: float,
) -> UVAtlas:
    """Carry coherent targets and the edit region across teacher refreshes."""

    if previous.rgb.shape != current.rgb.shape:
        raise ValueError("Cannot merge UV atlases with different resolutions")
    old_weight = previous.confidence * max(float(history_weight), 0.0)
    new_weight = current.confidence
    total_weight = old_weight + new_weight
    rgb = (
        previous.rgb * old_weight + current.rgb * new_weight
    ) / total_weight.clamp_min(1.0e-8)
    within_variance = (
        previous.variance * old_weight + current.variance * new_weight
    ) / total_weight.clamp_min(1.0e-8)
    between_variance = (
        old_weight
        * (previous.rgb - rgb).square().mean(dim=0, keepdim=True)
        + new_weight
        * (current.rgb - rgb).square().mean(dim=0, keepdim=True)
    ) / total_weight.clamp_min(1.0e-8)
    variance = within_variance + between_variance
    confidence = torch.maximum(
        previous.confidence * float(confidence_decay), current.confidence
    )
    confidence = confidence * torch.exp(
        -between_variance / max(float(variance_scale), 1.0e-8)
    )
    supported = total_weight > 0.0
    return UVAtlas(
        rgb=torch.where(supported.expand_as(rgb), rgb, current.rgb).clamp(
            0.0, 1.0
        ),
        confidence=confidence.clamp(0.0, 1.0),
        # Retain edits long enough that later low-t teachers cannot immediately
        # erase them, but decay transient diffusion errors because this project
        # infers its region from RGB differences rather than a parsing mask.
        edit=torch.maximum(
            previous.edit
            * (previous.confidence > 0.0).to(previous.edit.dtype)
            * float(edit_decay),
            current.edit
            * (current.confidence > 0.0).to(current.edit.dtype),
        ).clamp(0.0, 1.0),
        support=torch.maximum(previous.support, current.support),
        variance=torch.where(supported, variance, current.variance),
        refresh_step=int(current.refresh_step),
        teacher_timestep=int(current.teacher_timestep),
    )


# -----------------------------------------------------------------------------
# Stage-1 avatar and calibrated rendering
# -----------------------------------------------------------------------------


@dataclass
class SurfaceLayerRender:
    """One semantic FLAME/3DGS layer before cross-layer compositing."""

    uv: torch.Tensor
    alpha: torch.Tensor
    uv_variance: torch.Tensor
    depth: torch.Tensor
    # Stable correspondence contribution uses max(initial,current) opacity
    # plus the configured oral floor. Appearance contribution instead uses
    # the actual current RGB-render opacity and is required for residual
    # de-compositing.
    contribution: torch.Tensor
    appearance_contribution: Optional[torch.Tensor] = None


@dataclass
class RenderBatch:
    rgb: torch.Tensor
    alpha: torch.Tensor
    identity_rgb: Optional[torch.Tensor]
    identity_alpha: Optional[torch.Tensor]
    surface_uv: Optional[torch.Tensor]
    surface_alpha: Optional[torch.Tensor]
    surface_uv_variance: Optional[torch.Tensor]
    surface_layer_ids: Optional[torch.Tensor] = None
    surface_layers: Optional[dict[str, SurfaceLayerRender]] = None


class UVDAvatar:
    """Load Stage-1 UVD Gaussians and expose fixed-geometry batch rendering."""

    def __init__(
        self,
        reconstruction_dir: Path,
        device: torch.device,
        stability_config: Optional[Mapping[str, Any]] = None,
        stability_poses: Optional[Sequence[tuple[str, Any]]] = None,
        layered_surface_config: Optional[Mapping[str, Any]] = None,
    ):
        from gaussiansplatting.gaussian_renderer import render
        from gaussiansplatting.scene.gaussian_flame_face import (
            GaussianFlameUVModel,
        )
        from gaussiansplatting.utils.general_utils import strip_symmetric
        from gaussiansplatting.utils.sh_utils import SH2RGB
        from train_reconstruction import OpenCVCamera, aligned_geometry

        self._render = render
        self._strip_symmetric = strip_symmetric
        self._SH2RGB = SH2RGB
        self._OpenCVCamera = OpenCVCamera
        self._aligned_geometry_fn = aligned_geometry
        self.device = device
        self.reconstruction_dir = reconstruction_dir

        sidecar_path = reconstruction_dir / "model" / "reconstruction_params.npz"
        uvd_path = reconstruction_dir / "model" / "uvd.ply"
        validate_sh0_ply(uvd_path)
        with np.load(sidecar_path, allow_pickle=False) as archive:
            self.sidecar = {key: np.asarray(archive[key]) for key in archive.files}

        spatial_lr_scale = float(
            np.asarray(self.sidecar.get("spatial_lr_scale", 4.0)).reshape(-1)[0]
        )
        flame_scale = float(
            np.asarray(self.sidecar.get("flame_scale", -10.0)).reshape(-1)[0]
        )
        self.spatial_lr_scale = spatial_lr_scale
        self.flame_scale = flame_scale

        self.gaussian = GaussianFlameUVModel(sh_degree=0, device=str(device))
        self.gaussian.initialize_flame_state(spatial_lr_scale, flame_scale)
        self.gaussian.load_ply(str(uvd_path))
        if int(self.gaussian.max_sh_degree) != 0:
            raise ValueError(
                "loop_inpaint currently requires the Stage-1 SH degree to be 0"
            )

        shape = self._parameter(self.sidecar["shape"], 300)
        expression = self._parameter(self.sidecar["expression"], 100)
        eyes = self._parameter(self.sidecar["eyes"], 6)
        if "jaw_pose" in self.sidecar:
            jaw = self._parameter(self.sidecar["jaw_pose"], 3)
        else:
            jaw = self._parameter(self.sidecar["pose"], 6)[:, 3:6]
        self.alignment = torch.as_tensor(
            self.sidecar["facelift_from_training"],
            dtype=torch.float32,
            device=device,
        ).reshape(4, 4)
        self.reference_pose = (
            expression,
            jaw,
            eyes[:, :3],
            eyes[:, 3:6],
        )
        self.gaussian._shape.data.copy_(shape)
        self.gaussian._shape.requires_grad_(False)
        self.set_pose(*self.reference_pose)
        # Retain the exact Stage-1 covariance parameters so Stage-0 can emit a
        # directly comparable before/after render without loading a second
        # Gaussian model. They are never optimized or checkpointed.
        self.pre_stability_geometry = {
            "scale": self.gaussian._scaling.detach().clone(),
            "rotation": self.gaussian._rotation.detach().clone(),
        }
        self.stability_report: Optional[dict[str, Any]] = None
        if stability_config is not None:
            named_poses = list(
                stability_poses or (("reference", self.reference_pose),)
            )
            self.stability_report = stabilize_uvd_covariances(
                self.gaussian,
                named_poses,
                self._set_stability_pose,
                stability_config,
                reference_pose=self.reference_pose,
            )

        self.initial = {
            "uv": self.gaussian._uv.detach().clone(),
            "face_idx": self.gaussian._face_idx.detach().clone(),
            "d": self.gaussian._d.detach().clone(),
            "feature_dc": self.gaussian._features_dc.detach().clone(),
            "feature_rest": self.gaussian._features_rest.detach().clone(),
            "opacity": self.gaussian._opacity.detach().clone(),
            "scale": self.gaussian._scaling.detach().clone(),
            "rotation": self.gaussian._rotation.detach().clone(),
            "shape": self.gaussian._shape.detach().clone(),
        }
        self.initial_rgb = self._SH2RGB(
            self.initial["feature_dc"][:, 0, :]
        ).clamp_min(0.0)
        self.initial_opacity = torch.sigmoid(self.initial["opacity"])
        self.pipeline = SimpleNamespace(
            compute_cov3D_python=True,
            convert_SHs_python=False,
            debug=False,
        )
        self.white = torch.ones(3, device=device)
        self.black = torch.zeros(3, device=device)
        layered = dict(layered_surface_config or {})
        oral_layers = tuple(
            str(value)
            for value in layered.get(
                "layers",
                ("lips", "teeth_upper", "teeth_lower", "oral_cavity"),
            )
        )
        expected_oral_layers = (
            "lips",
            "teeth_upper",
            "teeth_lower",
            "oral_cavity",
        )
        if oral_layers != expected_oral_layers:
            raise ValueError(
                "fusion.layered_surface.layers must be exactly "
                "[lips, teeth_upper, teeth_lower, oral_cavity]"
            )
        self.surface_layer_names = SURFACE_LAYER_NAMES
        oral_masks = {
            name: self.gaussian.point_region_mask(name).detach()
            for name in oral_layers
        }
        oral_union = torch.zeros(
            int(self.gaussian.num_gs), dtype=torch.bool, device=self.device
        )
        for mask in oral_masks.values():
            if mask.shape != oral_union.shape:
                raise RuntimeError(
                    f"FLAME region {mask.shape} does not match Gaussian count"
                )
            oral_union |= mask
        self.surface_layer_masks = {
            "face": ~oral_union,
            **oral_masks,
        }
        assigned = torch.stack(
            [
                self.surface_layer_masks[name].to(torch.int32)
                for name in self.surface_layer_names
            ]
        ).sum(dim=0)
        if not torch.equal(assigned, torch.ones_like(assigned)):
            raise RuntimeError(
                "Semantic FLAME surface layers must partition every Gaussian"
            )
        self.layer_opacity_floor = float(layered.get("opacity_floor", 0.0))
        self.layer_contribution_threshold = float(
            layered.get("contribution_threshold", 0.0)
        )
        self.layer_dominance_ratio = float(
            layered.get("dominance_ratio", 1.0)
        )
        self.layer_variance_threshold = float(
            layered.get(
                "uv_variance_threshold",
                (layered_surface_config or {}).get(
                    "variance_threshold", float("inf")
                ),
            )
        )
        self._configure_trainable_parameters()

    def _parameter(self, value: Any, width: int) -> torch.Tensor:
        return torch.as_tensor(
            value, dtype=torch.float32, device=self.device
        ).reshape(1, -1)[:, :width]

    def _set_stability_pose(self, pose: Any) -> None:
        if isinstance(pose, tuple) and len(pose) == 4:
            self.set_pose(*pose)
            return
        self.set_pose(
            self._parameter(pose.expression, 100),
            self._parameter(pose.jaw_pose, 3),
            self._parameter(pose.leye_pose, 3),
            self._parameter(pose.reye_pose, 3),
        )

    def _configure_trainable_parameters(self) -> None:
        frozen = (
            self.gaussian._uv,
            self.gaussian._d,
            self.gaussian._features_rest,
            self.gaussian._scaling,
            self.gaussian._rotation,
            self.gaussian._shape,
        )
        for parameter in frozen:
            parameter.requires_grad_(False)
        self.gaussian._features_dc.requires_grad_(True)
        self.gaussian._opacity.requires_grad_(True)

    def set_pose(
        self,
        expression: torch.Tensor,
        jaw: torch.Tensor,
        leye: torch.Tensor,
        reye: torch.Tensor,
    ) -> None:
        zeros = torch.zeros((1, 3), device=self.device, dtype=torch.float32)
        self.gaussian._expression = expression.reshape(1, -1)[:, :100].float()
        self.gaussian._jaw_pose = jaw.reshape(1, -1)[:, :3].float()
        self.gaussian._leye_pose = leye.reshape(1, -1)[:, :3].float()
        self.gaussian._reye_pose = reye.reshape(1, -1)[:, :3].float()
        self.gaussian._global_orient = zeros
        self.gaussian._neck_pose = zeros
        self.gaussian._translation = zeros
        self._assert_root_motion_disabled()

    def _assert_root_motion_disabled(self) -> None:
        for name in ("_global_orient", "_neck_pose", "_translation"):
            value = getattr(self.gaussian, name)
            if int(torch.count_nonzero(value).item()) != 0:
                raise RuntimeError(
                    f"{name} must remain zero: root/neck motion is disabled"
                )

    @contextmanager
    def geometry_variant(self, variant: str):
        """Temporarily select raw Stage-1 or stabilized covariance geometry."""

        variant = str(variant).lower()
        if variant in {"stabilized", "current"}:
            yield
            return
        if variant != "raw":
            raise ValueError(
                "geometry variant must be 'raw', 'stabilized', or 'current'"
            )
        saved_scale = self.gaussian._scaling.detach().clone()
        saved_rotation = self.gaussian._rotation.detach().clone()
        try:
            with torch.no_grad():
                self.gaussian._scaling.copy_(
                    self.pre_stability_geometry["scale"]
                )
                self.gaussian._rotation.copy_(
                    self.pre_stability_geometry["rotation"]
                )
            yield
        finally:
            with torch.no_grad():
                self.gaussian._scaling.copy_(saved_scale)
                self.gaussian._rotation.copy_(saved_rotation)
            self.set_pose(*self.reference_pose)

    @contextmanager
    def appearance_variant(self, variant: str):
        """Temporarily restore immutable Stage-1 appearance for diagnostics."""

        variant = str(variant).lower()
        if variant == "current":
            yield
            return
        if variant != "initial":
            raise ValueError(
                "appearance variant must be 'initial' or 'current'"
            )
        saved_feature = self.gaussian._features_dc.detach().clone()
        saved_opacity = self.gaussian._opacity.detach().clone()
        try:
            with torch.no_grad():
                self.gaussian._features_dc.copy_(
                    self.initial["feature_dc"]
                )
                self.gaussian._opacity.copy_(self.initial["opacity"])
            yield
        finally:
            with torch.no_grad():
                self.gaussian._features_dc.copy_(saved_feature)
                self.gaussian._opacity.copy_(saved_opacity)

    def set_batch_pose(self, batch: Mapping[str, Any]) -> None:
        self.set_pose(
            self._parameter(batch["expression"], 100),
            self._parameter(batch["jaw_pose"], 3),
            self._parameter(batch["leye_pose"], 3),
            self._parameter(batch["reye_pose"], 3),
        )

    def cameras(self, batch: Mapping[str, Any]) -> list[Any]:
        w2c = torch.as_tensor(
            batch["w2c"], dtype=torch.float32, device=self.device
        )
        intrinsic = torch.as_tensor(
            batch["K"], dtype=torch.float32, device=self.device
        )
        if w2c.ndim == 2:
            w2c = w2c.unsqueeze(0)
            intrinsic = intrinsic.unsqueeze(0)
        width = int(torch.as_tensor(batch["width"]).reshape(-1)[0])
        height = int(torch.as_tensor(batch["height"]).reshape(-1)[0])
        return [
            self._OpenCVCamera(
                {
                    "w": width,
                    "h": height,
                    "fx": float(intrinsic[index, 0, 0]),
                    "fy": float(intrinsic[index, 1, 1]),
                    "w2c": w2c[index],
                },
                self.device,
            )
            for index in range(w2c.shape[0])
        ]

    @torch.no_grad()
    def packed_geometry(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.cuda.amp.autocast(enabled=False):
            means, covariance = self._aligned_geometry_fn(
                self.gaussian, self.alignment
            )
        return means.detach(), self._strip_symmetric(covariance.detach())

    @torch.no_grad()
    def _correspondence_opacity(self) -> torch.Tensor:
        """Opacity used only for correspondence buffers, never RGB training."""

        opacity = torch.maximum(
            self.gaussian.get_opacity.detach(), self.initial_opacity
        )
        if self.layer_opacity_floor > 0.0:
            oral = ~self.surface_layer_masks["face"]
            opacity = opacity.clone()
            opacity[oral] = opacity[oral].clamp_min(
                self.layer_opacity_floor
            )
        return opacity.clamp(0.0, 1.0)

    @torch.no_grad()
    def _semantic_contributions(
        self,
        camera: Any,
        packed: tuple[torch.Tensor, torch.Tensor],
        correspondence_opacity: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return occlusion-aware alpha contribution for every surface layer."""

        contributions: dict[str, torch.Tensor] = {}
        point_count = int(self.gaussian.num_gs)
        for start in range(0, len(self.surface_layer_names), 3):
            group = self.surface_layer_names[start : start + 3]
            colors = torch.zeros(
                point_count,
                3,
                device=self.device,
                dtype=correspondence_opacity.dtype,
            )
            for channel, name in enumerate(group):
                colors[self.surface_layer_masks[name], channel] = 1.0
            package = self._render(
                camera,
                self.gaussian,
                self.pipeline,
                self.black,
                override_color=colors,
                override_opacity=correspondence_opacity,
                precomputed_geometry=packed,
            )
            for channel, name in enumerate(group):
                contributions[name] = package["render"][
                    channel : channel + 1
                ].clamp(0.0, 1.0)
        return contributions

    @torch.no_grad()
    def _layer_moments(
        self,
        camera: Any,
        packed: tuple[torch.Tensor, torch.Tensor],
        correspondence_opacity: torch.Tensor,
        uv_color: torch.Tensor,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        layer_opacity = correspondence_opacity * self.surface_layer_masks[
            name
        ].to(correspondence_opacity.dtype).unsqueeze(1)
        package = self._render(
            camera,
            self.gaussian,
            self.pipeline,
            self.black,
            override_color=uv_color,
            override_opacity=layer_opacity,
            precomputed_geometry=packed,
        )
        alpha = package["alpha_3dgs"].clamp(0.0, 1.0)
        moments = normalize_alpha_weighted(
            package["render"].unsqueeze(0), alpha.unsqueeze(0)
        )[0]
        uv = moments[:2].clamp(0.0, 1.0)
        variance = (
            moments[2:3] - uv.square().sum(dim=0, keepdim=True)
        ).clamp_min(0.0)
        depth = normalize_alpha_weighted(
            package["depth_3dgs"].unsqueeze(0), alpha.unsqueeze(0)
        )[0]
        return uv, alpha, variance, depth

    def _select_surface_layer(
        self,
        layer_values: Mapping[
            str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        contributions: Mapping[str, torch.Tensor],
        composite_alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        composed = compose_layered_surface(
            {
                name: LayerSurfaceBuffers(
                    uv=layer_values[name][0].unsqueeze(0),
                    variance=layer_values[name][2].unsqueeze(0),
                    depth=layer_values[name][3].unsqueeze(0),
                    alpha=layer_values[name][1].unsqueeze(0),
                    contribution=contributions[name].unsqueeze(0),
                )
                for name in self.surface_layer_names
            },
            alpha_threshold=0.0,
            contribution_threshold=self.layer_contribution_threshold,
            variance_threshold=self.layer_variance_threshold,
            dominance_ratio=self.layer_dominance_ratio,
        )
        selected_alpha = torch.minimum(
            composed.surface_contribution[0],
            composite_alpha.clamp(0.0, 1.0),
        )
        return (
            composed.surface_uv[0],
            selected_alpha,
            composed.surface_variance[0],
            composed.layer_id[0],
        )

    def render_batch(
        self,
        batch: Mapping[str, Any],
        include_identity: bool,
        include_surface_uv: bool,
        include_surface_layers: bool = False,
        include_appearance_contributions: bool = False,
    ) -> RenderBatch:
        if include_surface_layers and not include_surface_uv:
            raise ValueError(
                "include_surface_layers requires include_surface_uv=True"
            )
        if include_appearance_contributions and not include_surface_layers:
            raise ValueError(
                "include_appearance_contributions requires "
                "include_surface_layers=True"
            )
        self.set_batch_pose(batch)
        cameras = self.cameras(batch)
        packed = self.packed_geometry()
        images, alphas = [], []
        identity_images, identity_alphas = [], []
        surface_uvs, surface_alphas, surface_uv_variances = [], [], []
        surface_layer_ids: list[torch.Tensor] = []
        layer_lists: dict[str, dict[str, list[torch.Tensor]]] = {
            name: {
                "uv": [],
                "alpha": [],
                "uv_variance": [],
                "depth": [],
                "contribution": [],
                "appearance_contribution": [],
            }
            for name in self.surface_layer_names
        }
        gaussian_uv = self.gaussian.get_uv.detach()
        uv_color = torch.cat(
            (gaussian_uv, gaussian_uv.square().sum(dim=1, keepdim=True)),
            dim=1,
        )
        correspondence_opacity = (
            self._correspondence_opacity() if include_surface_uv else None
        )

        for camera in cameras:
            with torch.cuda.amp.autocast(enabled=False):
                package = self._render(
                    camera,
                    self.gaussian,
                    self.pipeline,
                    self.white,
                    precomputed_geometry=packed,
                )
            images.append(package["render"])
            alphas.append(package["alpha_3dgs"].clamp(0.0, 1.0))

            if include_identity:
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                    identity = self._render(
                        camera,
                        self.gaussian,
                        self.pipeline,
                        self.white,
                        override_color=self.initial_rgb,
                        override_opacity=self.initial_opacity,
                        precomputed_geometry=packed,
                    )
                identity_images.append(identity["render"])
                identity_alphas.append(
                    identity["alpha_3dgs"].clamp(0.0, 1.0)
                )

            if include_surface_uv:
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                    if correspondence_opacity is None:
                        raise AssertionError(
                            "Correspondence opacity was not initialized"
                        )
                    contributions = self._semantic_contributions(
                        camera, packed, correspondence_opacity
                    )
                    appearance_contributions = (
                        self._semantic_contributions(
                            camera,
                            packed,
                            self.gaussian.get_opacity.detach().clamp(
                                0.0, 1.0
                            ),
                        )
                        if include_appearance_contributions
                        else None
                    )
                    dominant_stack = torch.stack(
                        [
                            contributions[name]
                            for name in self.surface_layer_names
                        ],
                        dim=0,
                    )
                    dominant_values, dominant_ids = dominant_stack[
                        :, 0
                    ].max(dim=0)
                    semantic_valid = (
                        dominant_values
                        >= self.layer_contribution_threshold
                    )
                    semantic_ids = torch.where(
                        semantic_valid.unsqueeze(0),
                        dominant_ids.unsqueeze(0),
                        torch.full_like(dominant_ids.unsqueeze(0), -1),
                    ).long()

                    if include_surface_layers:
                        layer_values = {
                            name: self._layer_moments(
                                camera,
                                packed,
                                correspondence_opacity,
                                uv_color,
                                name,
                            )
                            for name in self.surface_layer_names
                        }
                        (
                            surface_uv,
                            surface_alpha,
                            surface_uv_variance,
                            semantic_ids,
                        ) = self._select_surface_layer(
                            layer_values,
                            contributions,
                            package["alpha_3dgs"],
                        )
                        for name in self.surface_layer_names:
                            values = layer_values[name]
                            layer_lists[name]["uv"].append(values[0])
                            layer_lists[name]["alpha"].append(values[1])
                            layer_lists[name]["uv_variance"].append(values[2])
                            layer_lists[name]["depth"].append(values[3])
                            layer_lists[name]["contribution"].append(
                                contributions[name]
                            )
                            if appearance_contributions is not None:
                                layer_lists[name][
                                    "appearance_contribution"
                                ].append(appearance_contributions[name])
                    else:
                        surface = self._render(
                            camera,
                            self.gaussian,
                            self.pipeline,
                            self.black,
                            override_color=uv_color,
                            override_opacity=correspondence_opacity,
                            precomputed_geometry=packed,
                        )
                        surface_alpha = surface["alpha_3dgs"].clamp(
                            0.0, 1.0
                        )
                        surface_moments = (
                            surface["render"]
                            / surface_alpha.clamp_min(1.0e-6)
                        )
                        surface_uv = surface_moments[:2].clamp(0.0, 1.0)
                        surface_uv_variance = (
                            surface_moments[2:3]
                            - surface_uv.square().sum(
                                dim=0, keepdim=True
                            )
                        ).clamp_min(0.0)
                surface_uvs.append(surface_uv)
                surface_alphas.append(surface_alpha)
                surface_uv_variances.append(surface_uv_variance)
                surface_layer_ids.append(semantic_ids)

        return RenderBatch(
            rgb=torch.stack(images),
            alpha=torch.stack(alphas),
            identity_rgb=(
                torch.stack(identity_images) if identity_images else None
            ),
            identity_alpha=(
                torch.stack(identity_alphas) if identity_alphas else None
            ),
            surface_uv=torch.stack(surface_uvs) if surface_uvs else None,
            surface_alpha=(
                torch.stack(surface_alphas) if surface_alphas else None
            ),
            surface_uv_variance=(
                torch.stack(surface_uv_variances)
                if surface_uv_variances
                else None
            ),
            surface_layer_ids=(
                torch.stack(surface_layer_ids)
                if surface_layer_ids
                else None
            ),
            surface_layers=(
                {
                    name: SurfaceLayerRender(
                        uv=torch.stack(values["uv"]),
                        alpha=torch.stack(values["alpha"]),
                        uv_variance=torch.stack(values["uv_variance"]),
                        depth=torch.stack(values["depth"]),
                        contribution=torch.stack(values["contribution"]),
                        appearance_contribution=(
                            torch.stack(
                                values["appearance_contribution"]
                            )
                            if values["appearance_contribution"]
                            else None
                        ),
                    )
                    for name, values in layer_lists.items()
                }
                if include_surface_layers
                else None
            ),
        )

    def assert_fixed_geometry(self) -> None:
        self._assert_root_motion_disabled()
        checks = {
            "uv": self.gaussian._uv,
            "d": self.gaussian._d,
            "scale": self.gaussian._scaling,
            "rotation": self.gaussian._rotation,
            "shape": self.gaussian._shape,
        }
        for name, current in checks.items():
            if not torch.equal(current.detach(), self.initial[name]):
                raise RuntimeError(
                    f"Fixed Stage-2 parameter {name!r} changed unexpectedly"
                )
        if not torch.equal(
            self.gaussian._face_idx.detach(), self.initial["face_idx"]
        ):
            raise RuntimeError("Fixed Gaussian face binding changed unexpectedly")
        if int(self.gaussian.num_gs) != int(self.initial["uv"].shape[0]):
            raise RuntimeError("Stage-2 topology changed unexpectedly")

    def model_state(self) -> dict[str, torch.Tensor]:
        return {
            "feature_dc": self.gaussian._features_dc.detach().cpu(),
            "opacity": self.gaussian._opacity.detach().cpu(),
        }

    def load_model_state(self, state: Mapping[str, Any]) -> None:
        feature = torch.as_tensor(
            state["feature_dc"],
            dtype=self.gaussian._features_dc.dtype,
            device=self.device,
        )
        opacity = torch.as_tensor(
            state["opacity"],
            dtype=self.gaussian._opacity.dtype,
            device=self.device,
        )
        if feature.shape != self.gaussian._features_dc.shape:
            raise ValueError("Checkpoint feature tensor has incompatible shape")
        if opacity.shape != self.gaussian._opacity.shape:
            raise ValueError("Checkpoint opacity tensor has incompatible shape")
        self.gaussian._features_dc.data.copy_(feature)
        self.gaussian._opacity.data.copy_(opacity)

    @torch.no_grad()
    def clamp_opacity_increase(
        self,
        maximum_increase: float,
        oral_maximum_increase: Optional[float] = None,
    ) -> None:
        maximum_increase = float(maximum_increase)
        oral_maximum = (
            maximum_increase
            if oral_maximum_increase is None
            else float(oral_maximum_increase)
        )
        if maximum_increase >= 1.0 and oral_maximum >= 1.0:
            return
        current = torch.sigmoid(self.gaussian._opacity)
        maximum = (self.initial_opacity + maximum_increase).clamp(
            1.0e-6, 1.0 - 1.0e-6
        )
        oral_mask = (~self.surface_layer_masks["face"]).unsqueeze(1)
        oral_limit = (self.initial_opacity + oral_maximum).clamp(
            1.0e-6, 1.0 - 1.0e-6
        )
        maximum = torch.where(oral_mask, oral_limit, maximum)
        clamped = torch.minimum(current, maximum).clamp(
            1.0e-6, 1.0 - 1.0e-6
        )
        self.gaussian._opacity.copy_(
            torch.log(clamped) - torch.log1p(-clamped)
        )

    def save(self, directory: Path) -> None:
        from train_reconstruction import save_world_ply

        directory.mkdir(parents=True, exist_ok=True)
        self.set_pose(*self.reference_pose)
        self.assert_fixed_geometry()
        self.gaussian.save_ply(str(directory / "uvd.ply"))
        save_world_ply(
            self.gaussian, self.alignment, directory / "world.ply"
        )
        sidecar = dict(self.sidecar)
        sidecar["shape"] = self.gaussian._shape.detach().cpu().numpy()
        sidecar["expression"] = self.reference_pose[0].detach().cpu().numpy()
        sidecar["jaw_pose"] = self.reference_pose[1].detach().cpu().numpy()
        sidecar["eyes"] = (
            torch.cat((self.reference_pose[2], self.reference_pose[3]), dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        np.savez(directory / "reconstruction_params.npz", **sidecar)


# -----------------------------------------------------------------------------
# UV-correlated low-noise ControlNet SDEdit teacher
# -----------------------------------------------------------------------------


class SurfaceNoiseAtlas:
    """Canonical latent noise shared only within the same semantic surface."""

    def __init__(
        self,
        resolution: int,
        seed: int,
        device: torch.device,
        layer_count: int = 1,
        state: Optional[Mapping[str, Any]] = None,
    ):
        self.resolution = int(resolution)
        self.seed = int(seed)
        self.device = device
        self.layer_count = int(layer_count)
        if self.layer_count <= 0:
            raise ValueError("Surface-noise layer_count must be positive")
        self.background_counter = 0
        if state is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
            self.atlas = torch.randn(
                self.layer_count,
                4,
                self.resolution,
                self.resolution,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        else:
            atlas = torch.as_tensor(
                state["atlas"], dtype=torch.float32, device=device
            )
            expected = (
                self.layer_count,
                4,
                self.resolution,
                self.resolution,
            )
            if tuple(atlas.shape) != expected:
                raise ValueError(
                    "Surface-noise checkpoint shape differs from configuration: "
                    f"{tuple(atlas.shape)} vs {expected}"
                )
            self.atlas = atlas
            self.background_counter = int(state.get("background_counter", 0))

    def state_dict(self) -> dict[str, Any]:
        return {
            "atlas": self.atlas.detach().cpu(),
            "background_counter": int(self.background_counter),
            "seed": int(self.seed),
            "resolution": int(self.resolution),
            "layer_count": int(self.layer_count),
        }

    @torch.inference_mode()
    def sample(
        self,
        surface_uv: torch.Tensor,
        alpha: torch.Tensor,
        latent_size: tuple[int, int],
        layer_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent_alpha = F.interpolate(
            alpha, size=latent_size, mode="area"
        ).clamp(0.0, 1.0)
        if layer_ids is None:
            latent_layers = torch.zeros_like(
                latent_alpha, dtype=torch.long
            )
            # Preserve the legacy, premultiplied area-resize path for the
            # single-layer case.
            uv = F.interpolate(
                surface_uv * alpha,
                size=latent_size,
                mode="area",
            ) / latent_alpha.clamp_min(1.0e-6)
        else:
            if (
                layer_ids.ndim != 4
                or layer_ids.shape[1] != 1
                or layer_ids.shape[0] != surface_uv.shape[0]
                or layer_ids.shape[-2:] != surface_uv.shape[-2:]
            ):
                raise ValueError(
                    "layer_ids must have shape [B,1,H,W] matching surface_uv"
                )
            # Nearest resampling deliberately keeps UV and semantic ID from
            # one surface. Area filtering here would recreate the exact
            # lip/teeth averaging that layered correspondence is meant to
            # remove.
            latent_layers = F.interpolate(
                layer_ids.float(), size=latent_size, mode="nearest"
            ).long()
            uv = F.interpolate(
                surface_uv, size=latent_size, mode="nearest"
            )
        uv = uv.clamp(0.0, 1.0)
        grid = uv.permute(0, 2, 3, 1) * 2.0 - 1.0
        surface_noise = torch.zeros(
            uv.shape[0],
            4,
            *latent_size,
            device=self.device,
            dtype=self.atlas.dtype,
        )
        for layer_id in range(self.layer_count):
            sampled = F.grid_sample(
                self.atlas[layer_id : layer_id + 1].expand(
                    uv.shape[0], -1, -1, -1
                ),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            surface_noise = torch.where(
                (latent_layers == layer_id).expand_as(surface_noise),
                sampled,
                surface_noise,
            )

        # Bilinear interpolation lowers Gaussian variance.  Divide by the exact
        # interpolation standard deviation so the teacher still sees N(0, 1).
        x = uv[:, 0:1] * (self.resolution - 1)
        y = uv[:, 1:2] * (self.resolution - 1)
        dx = x - x.floor()
        dy = y - y.floor()
        variance = (
            ((1.0 - dx) * (1.0 - dy)).square()
            + ((1.0 - dx) * dy).square()
            + (dx * (1.0 - dy)).square()
            + (dx * dy).square()
        )
        surface_noise = surface_noise / variance.sqrt().clamp_min(1.0e-4)

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed + 1_000_003 + self.background_counter)
        background = torch.randn(
            surface_noise.shape,
            generator=generator,
            device=self.device,
            dtype=surface_noise.dtype,
        )
        self.background_counter += 1

        # This mixture preserves unit variance while softly transitioning at
        # the silhouette.
        surface_weight = latent_alpha * (
            (latent_layers >= 0) & (latent_layers < self.layer_count)
        ).to(latent_alpha.dtype)
        background_scale = (
            1.0 - surface_weight.square()
        ).clamp_min(0.0).sqrt()
        return surface_weight * surface_noise + background_scale * background


class DiffusionTeacher:
    """A single, lazily loaded SD1.5 + ControlNet SDEdit pipeline."""

    def __init__(
        self,
        config: Mapping[str, Any],
        device: torch.device,
        noise_state: Optional[Mapping[str, Any]] = None,
        attention_config: Optional[Mapping[str, Any]] = None,
        surface_layer_count: int = 1,
    ):
        os.environ.setdefault("THREESTUDIO_LAZY_IMPORT", "1")
        from threestudio.models.guidance.controlnet_guidance import (
            ControlNetGuidance,
        )

        self.config = config
        self.device = device
        self.attention_config = dict(attention_config or {})
        controlnet_config = dict(config["controlnet"])
        controlnet_config["vae_encode_mode"] = True
        controlnet_config["diffusion_steps"] = int(config["num_inference_steps"])
        controlnet_config["use_uvd_surface_flow"] = False
        controlnet_config["use_vsd"] = False
        controlnet_config["use_ism"] = False
        controlnet_config["use_nfsd"] = False
        controlnet_config["use_dsd"] = False
        # The project's guidance module contains local model paths relative to
        # the repository. Resolve them deterministically even when this script
        # is launched from another working directory.
        previous_cwd = Path.cwd()
        try:
            os.chdir(PROJECT_ROOT)
            self.guidance = ControlNetGuidance(controlnet_config)
        finally:
            os.chdir(previous_cwd)
        self.guidance.pipe.set_progress_bar_config(disable=True)
        self.noise = SurfaceNoiseAtlas(
            resolution=int(config["noise_atlas_resolution"]),
            seed=int(config["noise_seed"]),
            device=device,
            layer_count=int(surface_layer_count),
            state=noise_state,
        )
        self.surface_attention: Optional[SurfaceAttentionController] = None
        if bool(self.attention_config.get("enabled", False)):
            processor_config = SurfaceAttentionConfig(
                atlas_resolution=int(
                    self.attention_config["atlas_resolution"]
                ),
                max_tokens=int(self.attention_config["max_tokens"]),
                min_views=int(self.attention_config["min_views"]),
                strength=float(self.attention_config["strength"]),
            )
            self.surface_attention = install_surface_attention(
                self.guidance.unet, processor_config
            )

    def close(self) -> None:
        if self.surface_attention is not None:
            self.surface_attention.uninstall()
            self.surface_attention = None

    @staticmethod
    def _view_name(azimuth: float, elevation: float) -> str:
        azimuth = (float(azimuth) + 180.0) % 360.0 - 180.0
        if float(elevation) > 60.0:
            return "overhead view"
        if 45.0 < azimuth < 135.0:
            return "front view"
        if -135.0 < azimuth < -45.0:
            return "back view"
        return "side view"

    def prompts(self, batch: Mapping[str, Any]) -> list[str]:
        count = int(torch.as_tensor(batch["w2c"]).shape[0])
        base = str(self.config["prompt"]).strip()
        if not bool(self.config.get("append_view_prompt", True)):
            return [base] * count
        azimuth = torch.as_tensor(batch["azimuth"]).reshape(-1).tolist()
        elevation = torch.as_tensor(batch["elevation"]).reshape(-1).tolist()
        return [
            f"{base}, {self._view_name(azimuth[index], elevation[index])}"
            for index in range(count)
        ]

    @torch.inference_mode()
    def _text_embeddings(self, prompts: Sequence[str]) -> torch.Tensor:
        pipe = self.guidance.pipe
        negative = [str(self.config["negative_prompt"])] * len(prompts)
        texts = negative + list(prompts)
        tokens = pipe.tokenizer(
            texts,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return pipe.text_encoder(tokens.input_ids.to(self.device))[0]

    def _ddim_step(
        self,
        noise_prediction: torch.Tensor,
        timestep: int,
        previous_timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        scheduler = self.guidance.scheduler
        alpha_t = scheduler.alphas_cumprod[int(timestep)].to(sample)
        if previous_timestep >= 0:
            alpha_previous = scheduler.alphas_cumprod[
                int(previous_timestep)
            ].to(sample)
        else:
            alpha_previous = scheduler.final_alpha_cumprod.to(sample)
        beta_t = 1.0 - alpha_t
        prediction_type = str(scheduler.config.prediction_type)
        if prediction_type == "epsilon":
            predicted_x0 = (
                sample - beta_t.sqrt() * noise_prediction
            ) / alpha_t.sqrt()
            predicted_epsilon = noise_prediction
        elif prediction_type == "sample":
            predicted_x0 = noise_prediction
            predicted_epsilon = (
                sample - alpha_t.sqrt() * predicted_x0
            ) / beta_t.sqrt().clamp_min(1.0e-8)
        elif prediction_type == "v_prediction":
            predicted_x0 = (
                alpha_t.sqrt() * sample - beta_t.sqrt() * noise_prediction
            )
            predicted_epsilon = (
                alpha_t.sqrt() * noise_prediction + beta_t.sqrt() * sample
            )
        else:
            raise ValueError(
                f"Unsupported scheduler prediction_type: {prediction_type}"
            )
        if bool(scheduler.config.thresholding):
            predicted_x0 = scheduler._threshold_sample(predicted_x0)
        elif bool(scheduler.config.clip_sample):
            limit = float(scheduler.config.clip_sample_range)
            predicted_x0 = predicted_x0.clamp(-limit, limit)
        direction = (1.0 - alpha_previous).clamp_min(0.0).sqrt()
        return (
            alpha_previous.sqrt() * predicted_x0
            + direction * predicted_epsilon
        )

    @torch.inference_mode()
    def refine(
        self,
        image: torch.Tensor,
        condition: torch.Tensor,
        surface_uv: torch.Tensor,
        surface_alpha: torch.Tensor,
        surface_uv_variance: torch.Tensor,
        batch: Mapping[str, Any],
        timestep: int,
        anchor_image: Optional[torch.Tensor] = None,
        surface_layer_ids: Optional[torch.Tensor] = None,
        composite_alpha: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_size = image.shape[-2:]
        teacher_size = int(self.config["image_size"])
        image = F.interpolate(
            image.detach().clamp(0.0, 1.0),
            size=(teacher_size, teacher_size),
            mode="bilinear",
            align_corners=False,
        )
        if anchor_image is None:
            anchor = image
        else:
            anchor = F.interpolate(
                anchor_image.detach().clamp(0.0, 1.0),
                size=(teacher_size, teacher_size),
                mode="bilinear",
                align_corners=False,
            )
        condition = F.interpolate(
            condition.detach().clamp(0.0, 1.0),
            size=(teacher_size, teacher_size),
            mode="bilinear",
            align_corners=False,
        )
        surface_visibility = F.interpolate(
            surface_alpha.detach(),
            size=(teacher_size, teacher_size),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        if composite_alpha is None:
            composite_alpha = surface_alpha
        alpha = F.interpolate(
            composite_alpha.detach(),
            size=(teacher_size, teacher_size),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        surface_visibility = torch.minimum(surface_visibility, alpha)
        if surface_layer_ids is None:
            # Premultiply before resizing to avoid bleeding the background's
            # placeholder UV=(0, 0) into silhouette tokens.
            uv = F.interpolate(
                surface_uv.detach() * surface_alpha.detach(),
                size=(teacher_size, teacher_size),
                mode="bilinear",
                align_corners=False,
            ) / F.interpolate(
                surface_alpha.detach(),
                size=(teacher_size, teacher_size),
                mode="bilinear",
                align_corners=False,
            ).clamp_min(1.0e-6)
        else:
            # A nearest sample keeps a tooth UV on the tooth layer. Bilinear
            # filtering across a lip/tooth boundary would silently reintroduce
            # a cross-surface correspondence.
            uv = F.interpolate(
                surface_uv.detach(),
                size=(teacher_size, teacher_size),
                mode="nearest",
            )
        uv = uv.clamp(0.0, 1.0)
        uv_variance = F.interpolate(
            surface_uv_variance.detach(),
            size=(teacher_size, teacher_size),
            mode="bilinear",
            align_corners=False,
        ).clamp_min(0.0)
        layer_ids: Optional[torch.Tensor] = None
        if surface_layer_ids is not None:
            layer_ids = F.interpolate(
                surface_layer_ids.detach().float(),
                size=(teacher_size, teacher_size),
                mode="nearest",
            ).long()

        text_embeddings = self._text_embeddings(self.prompts(batch))
        latents = self.guidance.encode_images(image)
        noise = self.noise.sample(
            uv,
            surface_visibility,
            latents.shape[-2:],
            layer_ids=layer_ids,
        )
        t_batch = torch.full(
            (latents.shape[0],),
            int(timestep),
            device=self.device,
            dtype=torch.long,
        )
        latents = self.guidance.scheduler.add_noise(latents, noise, t_batch)

        step_count = int(self.config["num_inference_steps"])
        nodes = (
            torch.linspace(
                int(timestep),
                0,
                steps=step_count + 1,
                device=self.device,
                dtype=torch.float64,
            )
            .round()
            .long()
        )
        if torch.unique_consecutive(nodes).numel() != step_count + 1:
            raise ValueError(
                f"Cannot form {step_count} distinct DDIM transitions from "
                f"timestep {timestep}"
            )
        controller = self.surface_attention
        if controller is not None:
            attention_visibility = surface_visibility * surface_validity(
                uv,
                surface_visibility,
                float(self.attention_config.get("alpha_threshold", 0.0)),
                float(
                    self.attention_config.get("uv_jump_threshold", 0.0)
                ),
                uv_variance,
                float(
                    self.attention_config.get(
                        "uv_variance_threshold", float("inf")
                    )
                ),
            )
            controller.set_context(
                uv,
                attention_visibility,
                denoise_progress=0.0,
                cfg_branches=2,
                cfg_layout="chunked",
                layer_ids=layer_ids,
            )
        try:
            for index in range(step_count):
                current = int(nodes[index].item())
                previous = int(nodes[index + 1].item())
                if controller is not None:
                    controller.set_denoise_progress(
                        float(index + 1) / float(step_count)
                    )
                latent_input = torch.cat((latents, latents), dim=0)
                condition_input = torch.cat((condition, condition), dim=0)
                timestep_tensor = torch.tensor(
                    current, device=self.device, dtype=torch.long
                )
                down, middle = self.guidance.forward_controlnet(
                    latent_input,
                    timestep_tensor,
                    image_cond=condition_input,
                    condition_scale=float(self.config["condition_scale"]),
                    encoder_hidden_states=text_embeddings,
                )
                prediction = self.guidance.forward_control_unet(
                    latent_input,
                    timestep_tensor,
                    encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down,
                    mid_block_additional_residual=middle,
                )
                unconditioned, conditioned = prediction.chunk(2)
                prediction = unconditioned + float(
                    self.config["guidance_scale"]
                ) * (conditioned - unconditioned)
                latents = self._ddim_step(
                    prediction, current, previous, latents
                )
        finally:
            if controller is not None:
                controller.clear_context()

        refined = self.guidance.decode_latents(
            latents,
            latent_height=latents.shape[-2],
            latent_width=latents.shape[-1],
        ).clamp(0.0, 1.0)
        # The rasterizer uses white. Keep a standard alpha composite here so
        # atlas fusion can recover foreground surface color without white
        # silhouette contamination.
        background = torch.ones_like(refined)
        refined = alpha * refined + (1.0 - alpha) * background
        # Compare against the immutable Stage-1 render. Comparing with the
        # current render would make the edit region disappear after fitting a
        # teacher target and the identity loss would undo the edit.
        delta = (refined - anchor).abs().mean(dim=1, keepdim=True)
        mask_mode = str(self.config.get("edit_mask_mode", "difference")).lower()
        if mask_mode == "foreground":
            edit = alpha
        elif mask_mode == "difference":
            threshold = float(self.config["edit_threshold"])
            softness = max(float(self.config["edit_softness"]), 1.0e-6)
            edit = ((delta - threshold) / softness).clamp(0.0, 1.0) * alpha
        else:
            raise ValueError(
                "teacher.edit_mask_mode must be 'difference' or 'foreground'"
            )
        dilation = int(self.config.get("edit_dilation", 0))
        if dilation > 0:
            kernel = 2 * dilation + 1
            edit = F.max_pool2d(edit, kernel, stride=1, padding=dilation)
        feather = int(self.config.get("edit_feather", 0))
        if feather > 0:
            kernel = 2 * feather + 1
            edit = F.avg_pool2d(edit, kernel, stride=1, padding=feather)
        edit = edit.clamp(0.0, 1.0)

        if refined.shape[-2:] != source_size:
            refined = F.interpolate(
                refined,
                source_size,
                mode="bilinear",
                align_corners=False,
            )
            edit = F.interpolate(
                edit, source_size, mode="bilinear", align_corners=False
            )
        return refined, edit


# -----------------------------------------------------------------------------
# Reconstruction-aware data
# -----------------------------------------------------------------------------


def create_data_module(config: Mapping[str, Any]) -> Any:
    os.environ.setdefault("THREESTUDIO_LAZY_IMPORT", "1")
    from omegaconf import OmegaConf
    from threestudio.data.reconstruction_finetune import (
        ReconstructionFinetuneDataModule,
    )

    data_config = dict(config["data"])
    data_config["reconstruction_dir"] = str(
        resolve_path(config["input"]["reconstruction_dir"])
    )
    data_config["chemistry_path"] = str(
        resolve_path(data_config["chemistry_path"])
    )
    for key in (
        "test_expression_path",
        "test_pose_path",
        "validation_expression_path",
        "validation_pose_path",
    ):
        if key in data_config:
            data_config[key] = str(resolve_path(data_config[key]))
    module = ReconstructionFinetuneDataModule(OmegaConf.create(data_config))
    module.setup("fit")
    return module


def evenly_spaced_indices(indices: Sequence[int], count: int) -> list[int]:
    if count <= 0:
        raise ValueError("View count must be positive")
    if count > len(indices):
        raise ValueError(
            f"Requested {count} views from a group containing {len(indices)}"
        )
    if count == len(indices):
        return list(indices)
    positions = np.linspace(0, len(indices), count, endpoint=False)
    selected = [indices[int(round(position)) % len(indices)] for position in positions]
    if len(set(selected)) != len(selected):
        selected = [indices[int(position)] for position in positions]
    return selected


def teacher_preview_view_orders(
    view_count: int, output_config: Mapping[str, Any]
) -> list[int]:
    """Select preview positions independently for every teacher pose."""

    view_count = int(view_count)
    if view_count <= 0:
        raise ValueError("Teacher view count must be positive")
    if bool(output_config.get("save_all_teacher_observations", True)):
        return list(range(view_count))
    previews_per_pose = int(
        output_config.get("teacher_previews_per_pose", view_count)
    )
    return evenly_spaced_indices(
        list(range(view_count)), min(previews_per_pose, view_count)
    )


def teacher_observation_schedule(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe every scheduled teacher refresh and observation budget."""

    teacher = config["teacher"]
    total_steps = int(config["optimization"]["iterations"])
    coarse_steps = int(teacher.get("coarse_iterations", 0))
    detail_interval = int(teacher["refresh_interval"])
    coarse_interval = int(
        teacher.get("coarse_refresh_interval", detail_interval)
    )
    base_refresh_steps = (
        list(range(0, coarse_steps, coarse_interval))
        if coarse_steps > 0
        else []
    )
    detail_refresh_steps = list(
        range(coarse_steps, total_steps, detail_interval)
    )
    views = int(teacher["views_per_pose"])
    base_poses = int(
        teacher.get(
            "coarse_poses_per_refresh",
            teacher.get("poses_per_refresh", 1),
        )
    )
    detail_poses = int(teacher.get("poses_per_refresh", 1))
    return {
        "base_refresh_steps": base_refresh_steps,
        "detail_refresh_steps": detail_refresh_steps,
        "all_refresh_steps": base_refresh_steps + detail_refresh_steps,
        "base_refresh_count": len(base_refresh_steps),
        "detail_refresh_count": len(detail_refresh_steps),
        "views_per_pose": views,
        "base_poses_per_refresh": base_poses,
        "detail_poses_per_refresh": detail_poses,
        "base_observation_budget": (
            len(base_refresh_steps) * base_poses * views
        ),
        "detail_observation_budget": (
            len(detail_refresh_steps) * detail_poses * views
        ),
    }


def expected_active_refresh_step(
    config: Mapping[str, Any], completed_steps: int
) -> Optional[int]:
    """Refresh whose targets must be active after ``completed_steps``."""

    completed_steps = int(completed_steps)
    eligible = [
        int(step)
        for step in teacher_observation_schedule(config)[
            "all_refresh_steps"
        ]
        if int(step) < completed_steps
    ]
    return max(eligible) if eligible else None


def expected_base_atlas_refresh_step(
    config: Mapping[str, Any], completed_steps: int
) -> Optional[int]:
    """Last Base refresh that must back the frozen canonical atlases."""

    completed_steps = int(completed_steps)
    eligible = [
        int(step)
        for step in teacher_observation_schedule(config)[
            "base_refresh_steps"
        ]
        if int(step) < completed_steps
    ]
    return max(eligible) if eligible else None


def canonical_camera_indices(
    assets: Any,
    count: int,
    mode: str = "horizontal_ring",
) -> list[int]:
    mode = str(mode).lower()
    if mode == "stratified_all_rings":
        groups = [
            list(group) for group in assets.elevation_groups if len(group)
        ]
        if not groups:
            raise ValueError("No calibrated elevation rings are available")
        total = sum(len(group) for group in groups)
        if not 0 < int(count) <= total:
            raise ValueError(
                f"teacher.views_per_pose={count} exceeds the {total} "
                "available calibrated cameras"
            )

        elevation = [
            float(
                np.mean(
                    [
                        assets.frames[index].source_elevation_deg
                        for index in group
                    ]
                )
            )
            for group in groups
        ]
        allocations = [0 for _ in groups]
        # Allocate views round-robin, prioritizing rings nearest horizontal
        # whenever count is not divisible by the number of rings.
        priority = sorted(
            range(len(groups)), key=lambda index: abs(elevation[index])
        )
        for allocation_index in range(int(count)):
            available = [
                index
                for index in priority
                if allocations[index] < len(groups[index])
            ]
            if not available:
                break
            minimum = min(allocations[index] for index in available)
            candidates = [
                index
                for index in available
                if allocations[index] == minimum
            ]
            selected_group = candidates[
                allocation_index % len(candidates)
            ]
            allocations[selected_group] += 1

        selected: list[int] = []
        for group_index in sorted(
            range(len(groups)), key=lambda index: elevation[index]
        ):
            allocation = allocations[group_index]
            if allocation <= 0:
                continue
            ordered = sorted(
                groups[group_index],
                key=lambda index: (
                    assets.frames[index].source_azimuth_deg % 360.0
                ),
            )
            selected.extend(
                evenly_spaced_indices(ordered, allocation)
            )
        if len(selected) != int(count) or len(set(selected)) != int(count):
            raise RuntimeError(
                "Stratified camera selection did not produce the requested "
                "number of unique calibrated views"
            )
        return selected

    if mode != "horizontal_ring":
        raise ValueError(
            "teacher.view_sampling must be 'horizontal_ring' or "
            "'stratified_all_rings'"
        )
    eligible = [
        group for group in assets.elevation_groups if len(group) >= int(count)
    ]
    if not eligible:
        largest = max(len(group) for group in assets.elevation_groups)
        raise ValueError(
            f"teacher.views_per_pose={count} exceeds largest elevation ring "
            f"({largest})"
        )
    group = min(
        eligible,
        key=lambda values: abs(
            float(
                np.mean(
                    [
                        assets.frames[index].source_elevation_deg
                        for index in values
                    ]
                )
            )
        ),
    )
    ordered = sorted(
        group,
        key=lambda index: assets.frames[index].source_azimuth_deg % 360.0,
    )
    return evenly_spaced_indices(ordered, int(count))


def jaw_quantile_pose(assets: Any, quantile: float) -> Any:
    """Return the chemistry pose nearest a requested jaw-x quantile."""

    quantile = float(quantile)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("jaw quantile must be in [0, 1]")
    jaw = np.asarray(assets.chemistry_jaw[:, 0], dtype=np.float64)
    target = float(np.quantile(jaw, quantile))
    local_index = int(np.argmin(np.abs(jaw - target)))
    return assets.chemistry_pose(local_index)


def named_pose_envelope(
    assets: Any,
    quantiles: Sequence[float],
    include_validation: bool,
) -> list[tuple[str, Any]]:
    """Build a deterministic, de-duplicated FLAME pose envelope."""

    candidates: list[tuple[str, Any]] = [("reference", assets.reference_pose)]
    candidates.extend(
        (
            f"jaw_q{int(round(1000.0 * float(quantile))):03d}",
            jaw_quantile_pose(assets, float(quantile)),
        )
        for quantile in quantiles
    )
    if include_validation:
        candidates.append(("validation_open_mouth", assets.validation_pose))

    unique: list[tuple[str, Any]] = []
    signatures: set[str] = set()
    for label, pose in candidates:
        digest = hashlib.sha256()
        for value in (
            pose.expression,
            pose.jaw_pose,
            pose.leye_pose,
            pose.reye_pose,
        ):
            digest.update(
                np.ascontiguousarray(value, dtype=np.float32).tobytes()
            )
        signature = digest.hexdigest()
        if signature not in signatures:
            unique.append((label, pose))
            signatures.add(signature)
    return unique


def _artifact_label(value: str) -> str:
    """Return a filesystem-safe, human-readable artifact label."""

    cleaned = "".join(
        character
        if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in str(value)
    )
    cleaned = cleaned.strip("_")
    return cleaned or "unnamed"


def stratified_refresh_pose_envelope(
    assets: Any,
    count: int,
    *,
    include_reference: bool,
    require_open_mouth: bool,
) -> list[tuple[str, Any]]:
    """Sample dynamic poses across the complete jaw-opening rank range.

    ``count`` includes the optional reference pose.  Dynamic samples are drawn
    from equal-population jaw-x strata, so a refresh cannot accidentally spend
    all of its pose budget on nearly identical closed-mouth frames.
    """

    count = int(count)
    if count <= 0:
        raise ValueError("Teacher refresh pose count must be positive")
    named: list[tuple[str, Any]] = []
    if include_reference:
        named.append(("reference", assets.reference_pose))
    dynamic_count = count - len(named)
    if dynamic_count <= 0:
        return named[:count]

    jaw = np.asarray(assets.chemistry_jaw[:, 0], dtype=np.float64)
    if jaw.ndim != 1 or jaw.size == 0:
        raise RuntimeError("No chemistry jaw poses are available")
    if dynamic_count > jaw.size:
        raise ValueError(
            f"Requested {dynamic_count} dynamic teacher poses from only "
            f"{jaw.size} chemistry poses"
        )

    ordered = np.argsort(jaw, kind="stable")
    boundaries = np.linspace(0, jaw.size, dynamic_count + 1)
    selected_local_indices: list[int] = []
    for bin_index in range(dynamic_count):
        lower = int(math.floor(boundaries[bin_index]))
        upper = int(math.floor(boundaries[bin_index + 1]))
        upper = max(upper, lower + 1)
        candidates = [
            int(value)
            for value in ordered[lower:min(upper, jaw.size)]
            if int(value) not in selected_local_indices
        ]
        if not candidates:
            candidates = [
                int(value)
                for value in ordered
                if int(value) not in selected_local_indices
            ]
        local_index = candidates[random.randrange(len(candidates))]
        selected_local_indices.append(local_index)

    poses = [assets.chemistry_pose(index) for index in selected_local_indices]
    if (
        require_open_mouth
        and poses
        and not any(bool(pose.is_open_mouth) for pose in poses)
    ):
        open_candidates = [
            int(value)
            for value in np.asarray(
                assets.chemistry_open_indices, dtype=np.int64
            ).reshape(-1)
            if int(value) not in selected_local_indices[:-1]
        ]
        if not open_candidates:
            raise RuntimeError(
                "teacher.require_open_mouth_pose is true, but no unique "
                "open-mouth chemistry pose is available"
            )
        selected_local_indices[-1] = open_candidates[
            random.randrange(len(open_candidates))
        ]
        poses[-1] = assets.chemistry_pose(selected_local_indices[-1])

    for bin_index, pose in enumerate(poses):
        openness = "open" if bool(pose.is_open_mouth) else "closed"
        label = (
            f"jaw_bin_{bin_index + 1:02d}_of_{dynamic_count:02d}"
            f"_{openness}_src_{int(pose.source_index):06d}"
        )
        named.append((label, pose))
    return named[:count]


# -----------------------------------------------------------------------------
# Training, checkpointing, and export
# -----------------------------------------------------------------------------


STAGE_DEFINITIONS: tuple[tuple[str, str, str], ...] = (
    (
        "00_stage1_input",
        "Stage-1 input",
        "Unmodified reconstruction input before covariance stabilization.",
    ),
    (
        "01_geometry_stabilized",
        "geometry stabilized",
        "Same appearance after FLAME-pose-envelope covariance repair.",
    ),
    (
        "02_coherent_base",
        "coherent base",
        "Appearance fitted to an absolute face atlas, independent oral-layer "
        "residual atlases, and matched direct teacher views.",
    ),
    (
        "03_detail_refinement",
        "detail refinement",
        "Final low-timestep direct same-pose/same-view teacher refinement; "
        "the base atlas is frozen and is not used as detail pseudo truth.",
    ),
)


class LoopInpaintTrainer:
    def __init__(
        self,
        config: Mapping[str, Any],
        directory: Path,
        resume_path: Optional[Path] = None,
    ):
        self.config = config
        self.directory = directory
        self.device = torch.device(str(config.get("device", "cuda")))
        self.logger = setup_logger(directory)
        self.data_module = create_data_module(config)
        if self.data_module.assets is None or self.data_module.builder is None:
            raise RuntimeError("Reconstruction data module was not initialized")
        self.assets = self.data_module.assets
        self.builder = self.data_module.builder
        self.config_digest = hashlib.sha256(
            json.dumps(
                config,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.implementation_digest = implementation_digest()
        self.camera_digest = self._camera_digest()
        self.pose_data_digest = self._pose_data_digest()
        self.train_loader = self.data_module.train_dataloader()
        self.train_iterator = iter(self.train_loader)

        reconstruction_dir = resolve_path(
            config["input"]["reconstruction_dir"]
        )
        stability_config = config["stability"]
        stability_poses = named_pose_envelope(
            self.assets,
            stability_config.get("pose_quantiles", (0.0, 0.5, 0.95)),
            bool(
                stability_config.get("include_validation_pose", True)
            ),
        )
        self.avatar = UVDAvatar(
            reconstruction_dir,
            self.device,
            stability_config=stability_config,
            stability_poses=stability_poses,
            layered_surface_config={
                **dict(
                    config["fusion"].get("layered_surface", {})
                ),
                "uv_variance_threshold": float(
                    config["fusion"].get(
                        "uv_variance_threshold", float("inf")
                    )
                ),
            },
        )
        if self.avatar.stability_report is not None:
            stability_path = directory / "geometry_stability.json"
            stability_path.write_text(
                json.dumps(
                    self.avatar.stability_report,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            before = self.avatar.stability_report.get("before") or {}
            after = self.avatar.stability_report.get("after") or {}
            self.logger.info(
                "Covariance envelope: %d -> %d flagged Gaussians; "
                "%d unique repaired; report=%s",
                int(before.get("flagged_unique", 0)),
                int(after.get("flagged_unique", 0)),
                int(self.avatar.stability_report.get("unique_updated", 0)),
                stability_path,
            )
            if bool(stability_config.get("enabled", True)) and not bool(
                self.avatar.stability_report.get("converged", False)
            ):
                self.logger.warning(
                    "Covariance envelope did not fully converge after %d passes",
                    int(stability_config.get("passes", 1)),
                )
        self._verify_unique_uv()

        optimization = config["optimization"]
        groups = [
            {
                "params": [self.avatar.gaussian._features_dc],
                "lr": float(optimization["feature_lr"]),
                "name": "feature",
            }
        ]
        opacity_lr = float(optimization["opacity_lr"])
        self.avatar.gaussian._opacity.requires_grad_(opacity_lr > 0.0)
        if opacity_lr > 0.0:
            groups.append(
                {
                    "params": [self.avatar.gaussian._opacity],
                    "lr": opacity_lr,
                    "name": "opacity",
                }
            )
        self.optimizer = torch.optim.Adam(
            groups,
            betas=tuple(float(value) for value in optimization["betas"]),
            eps=float(optimization["eps"]),
        )

        from gaussiansplatting.utils.loss_utils import ssim

        self.ssim = ssim
        self.atlas: Optional[UVAtlas] = None
        self.layer_atlases: dict[str, UVAtlas] = {}
        self.layer_reference_rgb: Optional[torch.Tensor] = None
        self.direct_bank: Optional[DetailTargetBank] = None
        self.edit_allow_mask = self._load_edit_allow_mask()
        self.edit_mask_digest = self._edit_mask_digest()
        self.teacher: Optional[DiffusionTeacher] = None
        self.pending_noise_state: Optional[Mapping[str, Any]] = None
        self.pending_attention_diagnostics: Optional[
            Mapping[str, Any]
        ] = None
        self.start_step = 0
        self.canonical_indices = canonical_camera_indices(
            self.assets,
            int(config["teacher"]["views_per_pose"]),
            str(config["teacher"].get("view_sampling", "horizontal_ring")),
        )

        self.logs_dir = directory / "logs"
        self.atlas_dir = directory / "atlas"
        self.teacher_dir = directory / "teacher"
        self.preview_dir = directory / "previews"
        self.checkpoint_dir = directory / "checkpoints"
        # Stage directories are deliberately top-level.  This matches the
        # documented 00/01/02/03 tree and keeps them visible without requiring
        # callers to discover an extra, undocumented "stages" container.
        self.stages_dir = directory
        for path in (
            self.logs_dir,
            self.atlas_dir,
            self.teacher_dir,
            self.preview_dir,
            self.checkpoint_dir,
            self.stages_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer: Any = SummaryWriter(log_dir=str(self.logs_dir))
        except Exception as error:
            self.logger.warning("TensorBoard disabled: %s", error)
            self.writer = None

        self.metrics_file = (directory / "metrics.jsonl").open(
            "a", encoding="utf-8"
        )
        self.last_metric_record: Optional[dict[str, Any]] = None
        if resume_path is not None:
            self.load_checkpoint(resume_path)
        self.logger.info(
            "Loaded %d UVD Gaussians; trainable=%s; canonical views=%s",
            int(self.avatar.gaussian.num_gs),
            ", ".join(group["name"] for group in groups),
            [
                int(self.assets.frames[index].frame_index)
                for index in self.canonical_indices
            ],
        )

    def _camera_digest(self) -> str:
        digest = hashlib.sha256()
        for frame in self.assets.frames:
            digest.update(
                np.asarray(
                    [frame.frame_index, frame.width, frame.height],
                    dtype=np.int64,
                ).tobytes()
            )
            for value in (frame.K, frame.w2c):
                digest.update(
                    np.ascontiguousarray(value, dtype=np.float64).tobytes()
                )
        return digest.hexdigest()

    def _pose_data_digest(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.assets.chemistry_expression,
            self.assets.chemistry_jaw,
            self.assets.chemistry_leye,
            self.assets.chemistry_reye,
            self.assets.chemistry_source_indices,
            self.assets.chemistry_is_open,
            self.assets.validation_pose.expression,
            self.assets.validation_pose.jaw_pose,
            self.assets.validation_pose.leye_pose,
            self.assets.validation_pose.reye_pose,
        ):
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()

    def _edit_mask_digest(self) -> str:
        if self.edit_allow_mask is None:
            return "none"
        return hashlib.sha256(
            self.edit_allow_mask.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()

    def _fixed_source_state(self) -> dict[str, torch.Tensor]:
        state = {
            name: value.detach().cpu()
            for name, value in self.avatar.initial.items()
        }
        state["alignment"] = self.avatar.alignment.detach().cpu()
        state["flame_scale"] = torch.tensor(self.avatar.flame_scale)
        state["spatial_lr_scale"] = torch.tensor(
            self.avatar.spatial_lr_scale
        )
        expression, jaw, leye, reye = self.avatar.reference_pose
        state["reference_expression"] = expression.detach().cpu()
        state["reference_jaw"] = jaw.detach().cpu()
        state["reference_leye"] = leye.detach().cpu()
        state["reference_reye"] = reye.detach().cpu()
        return state

    def _load_edit_allow_mask(self) -> Optional[torch.Tensor]:
        value = self.config["fusion"].get("edit_allow_mask")
        if value is None:
            return None
        path = resolve_path(value)
        image = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        mask = torch.from_numpy(image).to(self.device).view(
            1, 1, image.shape[0], image.shape[1]
        )
        mask = mask / 255.0
        resolution = int(self.config["fusion"]["resolution"])
        if mask.shape[-2:] != (resolution, resolution):
            mask = F.interpolate(
                mask,
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
            )
        self.logger.info("Using canonical edit-allow mask: %s", path)
        return mask[0].clamp(0.0, 1.0)

    def _verify_unique_uv(self) -> None:
        uv = self.avatar.gaussian.get_uv.detach()
        unique = torch.unique(uv, dim=0).shape[0]
        if unique != uv.shape[0]:
            duplicates = int(uv.shape[0] - unique)
            raise ValueError(
                f"Stage-2 requires one canonical UV per Gaussian; found "
                f"{duplicates} duplicate UV coordinates"
            )

    def close(self) -> None:
        if self.teacher is not None:
            self.teacher.close()
        self.metrics_file.flush()
        self.metrics_file.close()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def _ensure_teacher(self) -> DiffusionTeacher:
        if self.teacher is None:
            self.teacher = DiffusionTeacher(
                self.config["teacher"],
                self.device,
                noise_state=self.pending_noise_state,
                attention_config=self.config["surface_attention"],
                surface_layer_count=len(self.avatar.surface_layer_names),
            )
            if (
                self.pending_attention_diagnostics is not None
                and self.teacher.surface_attention is not None
            ):
                self.teacher.surface_attention.load_diagnostics(
                    self.pending_attention_diagnostics
                )
            self.pending_noise_state = None
            self.pending_attention_diagnostics = None
        return self.teacher

    def _teacher_timestep(self, step: int) -> int:
        teacher = self.config["teacher"]
        lower = int(teacher["timestep_min"])
        upper = int(teacher["timestep_max"])
        coarse_steps = int(teacher.get("coarse_iterations", 0))
        if step < coarse_steps:
            return upper
        schedule = str(teacher.get("timestep_schedule", "linear")).lower()
        if schedule == "uniform":
            return random.randint(lower, upper)
        if schedule != "linear":
            raise ValueError(
                "teacher.timestep_schedule must be 'linear' or 'uniform'"
            )
        total_steps = int(self.config["optimization"]["iterations"])
        refinement_steps = max(total_steps - coarse_steps, 1)
        refresh_interval = int(teacher["refresh_interval"])
        last_refresh = (
            max(refinement_steps - 1, 0) // max(refresh_interval, 1)
        ) * max(refresh_interval, 1)
        progress = (
            min(
                max(float(step - coarse_steps) / float(last_refresh), 0.0),
                1.0,
            )
            if last_refresh > 0
            else 0.0
        )
        return int(round(upper + progress * (lower - upper)))

    def _refresh_poses(self, step: int) -> list[tuple[str, Any]]:
        teacher = self.config["teacher"]
        coarse_steps = int(teacher.get("coarse_iterations", 0))
        if step == 0 and step < coarse_steps:
            return named_pose_envelope(
                self.assets,
                teacher.get("coarse_pose_quantiles", (0.5, 0.8, 0.95)),
                bool(
                    teacher.get(
                        "include_validation_coarse_pose", True
                    )
                ),
            )
        if step < coarse_steps:
            return stratified_refresh_pose_envelope(
                self.assets,
                max(int(teacher.get("coarse_poses_per_refresh", 5)), 1),
                include_reference=bool(
                    teacher.get("include_reference_pose", True)
                ),
                require_open_mouth=bool(
                    teacher.get("require_open_mouth_pose", True)
                ),
            )
        if not bool(self.config["data"].get("use_dynamic_expression", True)):
            return [("reference", self.assets.reference_pose)]
        count = max(int(teacher.get("poses_per_refresh", 1)), 1)
        if bool(teacher.get("stratify_dynamic_poses_by_jaw", True)):
            return stratified_refresh_pose_envelope(
                self.assets,
                count,
                include_reference=bool(
                    teacher.get("include_reference_pose", True)
                ),
                require_open_mouth=bool(
                    teacher.get("require_open_mouth_pose", True)
                ),
            )

        named_poses: list[tuple[str, Any]] = []
        pose_keys: set[tuple[bool, int]] = set()
        if bool(teacher.get("include_reference_pose", True)):
            reference = self.assets.reference_pose
            named_poses.append(("reference", reference))
            pose_keys.add((True, int(reference.source_index)))
        while len(named_poses) < count:
            candidate = self.assets.sample_pose()
            key = (bool(candidate.is_reference), int(candidate.source_index))
            if key in pose_keys:
                candidate = self.assets.chemistry_pose(
                    random.randrange(self.assets.chemistry_expression.shape[0])
                )
                key = (False, int(candidate.source_index))
                if key in pose_keys:
                    continue
            label = (
                "reference"
                if bool(candidate.is_reference)
                else (
                    f"chemistry_{'open' if candidate.is_open_mouth else 'closed'}"
                    f"_src_{int(candidate.source_index):06d}"
                )
            )
            named_poses.append((label, candidate))
            pose_keys.add(key)
        return named_poses[:count]

    @staticmethod
    def _teacher_pose_metadata(
        pose_index: int, pose_label: str, pose: Any
    ) -> dict[str, Any]:
        expression = np.asarray(pose.expression, dtype=np.float32).reshape(-1)
        jaw = np.asarray(pose.jaw_pose, dtype=np.float32).reshape(-1)
        left_eye = np.asarray(pose.leye_pose, dtype=np.float32).reshape(-1)
        right_eye = np.asarray(pose.reye_pose, dtype=np.float32).reshape(-1)
        return {
            "pose_index": int(pose_index),
            "label": str(pose_label),
            "source_index": int(pose.source_index),
            "is_reference": bool(pose.is_reference),
            "is_open_mouth": bool(pose.is_open_mouth),
            "jaw_pose": [float(value) for value in jaw],
            "jaw_x": float(jaw[0]) if jaw.size else 0.0,
            "jaw_l2": float(np.linalg.norm(jaw)),
            "expression_l2": float(np.linalg.norm(expression)),
            "left_eye_l2": float(np.linalg.norm(left_eye)),
            "right_eye_l2": float(np.linalg.norm(right_eye)),
            "observations": [],
        }

    @staticmethod
    def _save_teacher_contact_sheet(
        rows: Sequence[Sequence[np.ndarray]], path: Path
    ) -> None:
        if not rows or any(not row for row in rows):
            raise ValueError("Teacher contact sheet has an empty pose row")
        column_counts = {len(row) for row in rows}
        if len(column_counts) != 1:
            raise ValueError(
                "Teacher contact-sheet pose rows have different view counts"
            )
        Image.fromarray(
            np.concatenate(
                [np.concatenate(list(row), axis=1) for row in rows],
                axis=0,
            )
        ).save(path, quality=92)

    @staticmethod
    def _prepare_refresh_attempt(refresh_dir: Path) -> Path:
        """Make a crashed completed-bank refresh safely retryable."""

        outer_success = refresh_dir / "_SUCCESS.json"
        if outer_success.is_file():
            suffix = 1
            while True:
                archived = (
                    refresh_dir
                    / f"_SUCCESS.superseded_{suffix:02d}.json"
                )
                if not archived.exists():
                    os.replace(outer_success, archived)
                    break
                suffix += 1

        primary = refresh_dir / "direct_bank"
        if not (primary / "_SUCCESS.json").is_file():
            return primary
        suffix = 1
        while True:
            candidate = refresh_dir / f"direct_bank_retry_{suffix:02d}"
            if not (candidate / "_SUCCESS.json").is_file():
                return candidate
            suffix += 1

    @torch.inference_mode()
    def refresh_atlas(self, step: int) -> None:
        teacher = self._ensure_teacher()
        teacher_timestep = self._teacher_timestep(step)
        named_poses = self._refresh_poses(step)
        coarse_steps = int(
            self.config["teacher"].get("coarse_iterations", 0)
        )
        build_canonical_atlases = int(step) < coarse_steps
        accumulator = (
            UVAtlasAccumulator(self.config["fusion"])
            if build_canonical_atlases
            else None
        )
        layer_accumulators = (
            {
                name: UVAtlasAccumulator(
                    {
                        **dict(self.config["fusion"]),
                        "alpha_threshold": float(
                            self.config["fusion"]["layered_surface"][
                                "contribution_threshold"
                            ]
                        ),
                    }
                )
                for name in self.avatar.surface_layer_names[1:]
            }
            if build_canonical_atlases
            else {}
        )
        refreshed_layer_reference = (
            torch.nan_to_num(
                self.avatar._SH2RGB(
                    self.avatar.gaussian._features_dc[
                        :, 0, :
                    ].detach()
                ),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            .clamp(0.0, 1.0)
            .clone()
            if build_canonical_atlases
            else None
        )
        batch_size = max(int(self.config["teacher"]["batch_size"]), 1)
        output_config = self.config["output"]
        save_all_observations = bool(
            output_config.get("save_all_teacher_observations", True)
        )
        previews_per_pose = int(
            output_config.get(
                "teacher_previews_per_pose", len(self.canonical_indices)
            )
        )
        preview_view_orders = set(
            teacher_preview_view_orders(
                len(self.canonical_indices), output_config
            )
        )
        tile_size = int(
            output_config.get("teacher_contact_sheet_tile_size", 160)
        )
        previews_saved = 0
        refresh_dir = self.teacher_dir / f"step_{step:06d}"
        refresh_dir.mkdir(parents=True, exist_ok=True)
        bank_directory = self._prepare_refresh_attempt(refresh_dir)
        bank_writer = DetailTargetBankWriter(
            bank_directory,
            refresh_step=step,
            teacher_timestep=teacher_timestep,
            config_sha256=self.config_digest,
            implementation_sha256=self.implementation_digest,
            metadata={
                "stage": (
                    "coherent_base"
                    if build_canonical_atlases
                    else "detail_refinement"
                ),
                "supervision": (
                    "canonical_face_and_layer_atlases_plus_direct_teacher"
                    if build_canonical_atlases
                    else "direct_teacher"
                ),
                "pose_control": {
                    "global_orient": "forced_zero",
                    "neck_pose": "forced_zero",
                    "translation": "forced_zero",
                },
            },
        )
        pose_records: list[dict[str, Any]] = []
        render_grid_rows: list[list[np.ndarray]] = []
        condition_grid_rows: list[list[np.ndarray]] = []
        teacher_grid_rows: list[list[np.ndarray]] = []
        edit_grid_rows: list[list[np.ndarray]] = []
        layer_grid_rows: list[list[np.ndarray]] = []
        oral_correspondence_grid_rows: list[list[np.ndarray]] = []
        oral_appearance_grid_rows: list[list[np.ndarray]] = []
        attention_before = dict(
            self._surface_attention_stage_metrics()["runtime"]
        )
        expected_observations = (
            len(named_poses) * len(self.canonical_indices)
        )

        self.logger.info(
            "Refreshing %s supervision at step %d: %d pose(s), "
            "%d views/pose, t=%d",
            (
                "Base layered-atlas + direct"
                if build_canonical_atlases
                else "Detail direct"
            ),
            step,
            len(named_poses),
            len(self.canonical_indices),
            teacher_timestep,
        )
        self.logger.info(
            "Teacher pose envelope: %s",
            [
                {
                    "label": label,
                    "source_index": int(pose.source_index),
                    "open_mouth": bool(pose.is_open_mouth),
                    "jaw_x": float(
                        np.asarray(pose.jaw_pose).reshape(-1)[0]
                    ),
                }
                for label, pose in named_poses
            ],
        )
        progress = tqdm(
            total=expected_observations,
            desc=f"Teacher refresh {step}",
            dynamic_ncols=True,
            leave=False,
        )
        for pose_index, (pose_label, pose) in enumerate(named_poses):
            pose_record = self._teacher_pose_metadata(
                pose_index, pose_label, pose
            )
            pose_records.append(pose_record)
            render_tiles: list[np.ndarray] = []
            condition_tiles: list[np.ndarray] = []
            teacher_tiles: list[np.ndarray] = []
            edit_tiles: list[np.ndarray] = []
            layer_tiles: list[np.ndarray] = []
            oral_correspondence_tiles: list[np.ndarray] = []
            oral_appearance_tiles: list[np.ndarray] = []
            pose_dir = (
                refresh_dir
                / f"pose_{pose_index:02d}_{_artifact_label(pose_label)}"
            )
            for start in range(0, len(self.canonical_indices), batch_size):
                camera_indices = self.canonical_indices[start : start + batch_size]
                batch = self.builder.build(camera_indices, pose)
                batch = to_device(batch, self.device)
                rendering = self.avatar.render_batch(
                    batch,
                    include_identity=True,
                    include_surface_uv=True,
                    include_surface_layers=True,
                    include_appearance_contributions=True,
                )
                if (
                    rendering.identity_rgb is None
                    or rendering.surface_uv is None
                    or rendering.surface_alpha is None
                    or rendering.surface_uv_variance is None
                    or rendering.surface_layer_ids is None
                    or rendering.surface_layers is None
                    or any(
                        layer.appearance_contribution is None
                        for layer in rendering.surface_layers.values()
                    )
                ):
                    raise RuntimeError(
                        "Teacher refresh is missing identity/layered surface "
                        "buffers"
                    )
                condition = as_bchw(batch["flame_conds"], self.device)[:, :3]
                refined, edit = teacher.refine(
                    rendering.rgb.clamp(0.0, 1.0),
                    condition,
                    rendering.surface_uv,
                    rendering.surface_alpha,
                    rendering.surface_uv_variance,
                    batch,
                    teacher_timestep,
                    anchor_image=rendering.identity_rgb,
                    surface_layer_ids=rendering.surface_layer_ids,
                    composite_alpha=rendering.alpha,
                )
                frame_ids = (
                    torch.as_tensor(batch["frame_index"])
                    .reshape(-1)
                    .tolist()
                )
                if accumulator is not None:
                    face = rendering.surface_layers["face"]
                    face_visibility = face.contribution * (
                        rendering.surface_layer_ids == 0
                    ).to(face.contribution.dtype)
                    accumulator.add(
                        refined,
                        face.uv,
                        face_visibility,
                        edit,
                        composite_alpha=rendering.alpha,
                        view_ids=frame_ids,
                        surface_uv_variance=face.uv_variance,
                    )
                    for layer_name, layer_accumulator in (
                        layer_accumulators.items()
                    ):
                        layer = rendering.surface_layers[layer_name]
                        layer_id = self.avatar.surface_layer_names.index(
                            layer_name
                        )
                        dominant_visibility = layer.contribution * (
                            rendering.surface_layer_ids == layer_id
                        ).to(layer.contribution.dtype)
                        # Store a signed teacher residual, encoded into [0,1],
                        # rather than the fully composited teacher RGB. The
                        # latter still contains other translucent layers and
                        # would turn a teeth atlas into another lip/background
                        # mixture. During optimization this residual is added
                        # to the layer's per-Gaussian RGB snapshot.
                        encoded_surface_residual = (
                            encode_surface_rgb_residual(
                                refined.detach(),
                                rendering.rgb.detach(),
                                layer.appearance_contribution,
                                contribution_floor=float(
                                    self.config["fusion"][
                                        "layered_surface"
                                    ]["residual_decomposition_floor"]
                                ),
                            )
                        )
                        layer_accumulator.add(
                            encoded_surface_residual,
                            layer.uv,
                            dominant_visibility,
                            edit,
                            composite_alpha=torch.ones_like(
                                layer.contribution
                            ),
                            view_ids=frame_ids,
                            surface_uv_variance=layer.uv_variance,
                        )
                validity = surface_validity(
                    rendering.surface_uv,
                    rendering.surface_alpha,
                    float(self.config["fusion"]["alpha_threshold"]),
                    float(self.config["fusion"]["uv_jump_threshold"]),
                    rendering.surface_uv_variance,
                    float(
                        self.config["fusion"]["uv_variance_threshold"]
                    ),
                )
                render_highpass = rendering.rgb - F.avg_pool2d(
                    rendering.rgb,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
                teacher_highpass = refined - F.avg_pool2d(
                    refined,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )

                for local_index, camera_index in enumerate(camera_indices):
                    view_order = start + local_index
                    frame = int(
                        torch.as_tensor(batch["frame_index"])
                        .reshape(-1)[local_index]
                        .item()
                    )
                    camera = self.assets.frames[camera_index]
                    current_image = rgb_u8(rendering.rgb[local_index])
                    condition_image = rgb_u8(condition[local_index])
                    teacher_image = rgb_u8(refined[local_index])
                    mask_rgb = edit[local_index].expand(3, -1, -1)
                    edit_image = rgb_u8(mask_rgb)
                    layer_image = surface_layer_u8(
                        rendering.surface_layer_ids[local_index]
                    )
                    oral_correspondence = torch.stack(
                        [
                            rendering.surface_layers[name].contribution[
                                local_index
                            ]
                            for name in self.avatar.surface_layer_names[1:]
                        ]
                    ).sum(dim=0).clamp(0.0, 1.0)
                    oral_appearance_values = [
                        rendering.surface_layers[
                            name
                        ].appearance_contribution
                        for name in self.avatar.surface_layer_names[1:]
                    ]
                    if any(
                        value is None for value in oral_appearance_values
                    ):
                        raise AssertionError(
                            "Teacher render lost actual appearance "
                            "contributions"
                        )
                    oral_appearance = torch.stack(
                        [
                            value[local_index]
                            for value in oral_appearance_values
                            if value is not None
                        ]
                    ).sum(dim=0).clamp(0.0, 1.0)
                    oral_correspondence_image = rgb_u8(
                        oral_correspondence.expand(3, -1, -1)
                    )
                    oral_appearance_image = rgb_u8(
                        oral_appearance.expand(3, -1, -1)
                    )
                    layer_evidence = {}
                    for layer_id, layer_name in enumerate(
                        self.avatar.surface_layer_names
                    ):
                        contribution = rendering.surface_layers[
                            layer_name
                        ].contribution[local_index]
                        appearance_contribution = (
                            rendering.surface_layers[
                                layer_name
                            ].appearance_contribution
                        )
                        if appearance_contribution is None:
                            raise AssertionError(
                                "Teacher render lost a layer appearance "
                                "contribution"
                            )
                        appearance_contribution = (
                            appearance_contribution[local_index]
                        )
                        dominant = (
                            rendering.surface_layer_ids[local_index]
                            == layer_id
                        )
                        layer_evidence[layer_name] = {
                            "dominant_pixels": int(dominant.sum().item()),
                            "dominant_fraction": float(
                                dominant.float().mean().item()
                            ),
                            "correspondence_contribution_mean": float(
                                contribution.mean().item()
                            ),
                            "correspondence_contribution_max": float(
                                contribution.max().item()
                            ),
                            "appearance_contribution_mean": float(
                                appearance_contribution.mean().item()
                            ),
                            "appearance_contribution_max": float(
                                appearance_contribution.max().item()
                            ),
                        }
                    pose_id = (
                        f"pose_{pose_index:02d}_"
                        f"{_artifact_label(pose_label)}"
                    )
                    bank_writer.add(
                        pose_id=pose_id,
                        pose=pose,
                        camera_index=int(camera_index),
                        frame_index=frame,
                        target=refined[local_index],
                        edit_mask=edit[local_index],
                        metadata={
                            "view_order": int(view_order),
                            "pose_label": str(pose_label),
                            "surface_layer_valid_fraction": float(
                                (
                                    rendering.surface_layer_ids[local_index]
                                    >= 0
                                )
                                .float()
                                .mean()
                                .item()
                            ),
                            "oral_correspondence_mean": float(
                                oral_correspondence.mean().item()
                            ),
                            "oral_appearance_contribution_mean": float(
                                oral_appearance.mean().item()
                            ),
                            "surface_layers": layer_evidence,
                        },
                    )
                    short_label = (
                        f"p{pose_index:02d} "
                        f"{_artifact_label(pose_label)[:12]} "
                        f"| v{view_order:02d} f{frame:03d}"
                    )

                    def contact_tile(image: np.ndarray) -> np.ndarray:
                        resized = np.asarray(
                            Image.fromarray(image).resize(
                                (tile_size, tile_size),
                                Image.Resampling.LANCZOS,
                            ),
                            dtype=np.uint8,
                        )
                        return add_label(resized, short_label)

                    render_tiles.append(contact_tile(current_image))
                    condition_tiles.append(contact_tile(condition_image))
                    teacher_tiles.append(contact_tile(teacher_image))
                    edit_tiles.append(contact_tile(edit_image))
                    layer_tiles.append(contact_tile(layer_image))
                    oral_correspondence_tiles.append(
                        contact_tile(oral_correspondence_image)
                    )
                    oral_appearance_tiles.append(
                        contact_tile(oral_appearance_image)
                    )

                    preview_path: Optional[Path] = None
                    if view_order in preview_view_orders:
                        pose_dir.mkdir(parents=True, exist_ok=True)
                        preview_path = (
                            pose_dir
                            / (
                                f"view_{view_order:02d}_frame_{frame:03d}"
                                f"_t{teacher_timestep:03d}.jpg"
                            )
                        )
                        comparison_strip = np.concatenate(
                            (
                                add_label(
                                    current_image,
                                    (
                                        f"current | {pose_label} | "
                                        f"view {view_order:02d} frame {frame:03d}"
                                    ),
                                ),
                                add_label(
                                    condition_image,
                                    (
                                        f"FLAME condition | {pose_label} | "
                                        f"view {view_order:02d} frame {frame:03d}"
                                    ),
                                ),
                                add_label(
                                    teacher_image,
                                    (
                                        f"surface teacher | {pose_label} | "
                                        f"view {view_order:02d} frame {frame:03d}"
                                    ),
                                ),
                                add_label(
                                    edit_image,
                                    (
                                        f"edit mask | {pose_label} | "
                                        f"view {view_order:02d} frame {frame:03d}"
                                    ),
                                ),
                                add_label(
                                    layer_image,
                                    (
                                        f"surface layer | {pose_label} | "
                                        f"view {view_order:02d} frame {frame:03d}"
                                    ),
                                ),
                                add_label(
                                    oral_correspondence_image,
                                    (
                                        "oral correspondence | "
                                        f"{pose_label} | "
                                        f"view {view_order:02d} frame {frame:03d}"
                                    ),
                                ),
                                add_label(
                                    oral_appearance_image,
                                    (
                                        "oral RGB contribution | "
                                        f"{pose_label} | "
                                        f"view {view_order:02d} frame {frame:03d}"
                                    ),
                                ),
                            ),
                            axis=1,
                        )
                        Image.fromarray(comparison_strip).save(
                            preview_path, quality=92
                        )
                        previews_saved += 1

                    pose_record["observations"].append(
                        {
                            "observation_index": int(
                                pose_index * len(self.canonical_indices)
                                + view_order
                            ),
                            "view_order": int(view_order),
                            "camera_data_index": int(camera_index),
                            "camera_frame_index": int(frame),
                            "source_azimuth_deg": float(
                                camera.source_azimuth_deg
                            ),
                            "source_elevation_deg": float(
                                camera.source_elevation_deg
                            ),
                            "prompt_azimuth_deg": float(
                                camera.prompt_azimuth_deg
                            ),
                            "prompt_elevation_deg": float(
                                camera.prompt_elevation_deg
                            ),
                            "preview": (
                                preview_path.relative_to(refresh_dir).as_posix()
                                if preview_path is not None
                                else None
                            ),
                            "surface_valid_fraction": float(
                                validity[local_index].mean().item()
                            ),
                            "surface_alpha_coverage": float(
                                (
                                    rendering.surface_alpha[local_index]
                                    > float(
                                        self.config["fusion"][
                                            "alpha_threshold"
                                        ]
                                    )
                                )
                                .float()
                                .mean()
                                .item()
                            ),
                            "edit_mask_mean": float(
                                edit[local_index].mean().item()
                            ),
                            "teacher_delta_l1": float(
                                (
                                    refined[local_index]
                                    - rendering.rgb[local_index]
                                )
                                .abs()
                                .mean()
                                .item()
                            ),
                            "teacher_highpass_delta_l1": float(
                                (
                                    teacher_highpass[local_index]
                                    - render_highpass[local_index]
                                )
                                .abs()
                                .mean()
                                .item()
                            ),
                            "surface_layer_valid_fraction": float(
                                (
                                    rendering.surface_layer_ids[local_index]
                                    >= 0
                                )
                                .float()
                                .mean()
                                .item()
                            ),
                            "oral_correspondence_mean": float(
                                oral_correspondence.mean().item()
                            ),
                            "oral_appearance_contribution_mean": float(
                                oral_appearance.mean().item()
                            ),
                            "surface_layers": layer_evidence,
                        }
                    )
                progress.update(len(camera_indices))
            if len(pose_record["observations"]) != len(
                self.canonical_indices
            ):
                raise RuntimeError(
                    f"Teacher pose {pose_label!r} produced "
                    f"{len(pose_record['observations'])} observations; "
                    f"expected {len(self.canonical_indices)}"
                )
            render_grid_rows.append(render_tiles)
            condition_grid_rows.append(condition_tiles)
            teacher_grid_rows.append(teacher_tiles)
            edit_grid_rows.append(edit_tiles)
            layer_grid_rows.append(layer_tiles)
            oral_correspondence_grid_rows.append(
                oral_correspondence_tiles
            )
            oral_appearance_grid_rows.append(oral_appearance_tiles)
        progress.close()

        processed_observations = sum(
            len(record["observations"]) for record in pose_records
        )
        if processed_observations != expected_observations:
            raise RuntimeError(
                f"Teacher refresh processed {processed_observations} "
                f"observations; expected {expected_observations}"
            )
        self._save_teacher_contact_sheet(
            render_grid_rows, refresh_dir / "00_current_render_grid.jpg"
        )
        self._save_teacher_contact_sheet(
            condition_grid_rows, refresh_dir / "01_flame_condition_grid.jpg"
        )
        self._save_teacher_contact_sheet(
            teacher_grid_rows, refresh_dir / "02_surface_teacher_grid.jpg"
        )
        self._save_teacher_contact_sheet(
            edit_grid_rows, refresh_dir / "03_edit_mask_grid.jpg"
        )
        self._save_teacher_contact_sheet(
            layer_grid_rows, refresh_dir / "04_surface_layer_grid.jpg"
        )
        self._save_teacher_contact_sheet(
            oral_correspondence_grid_rows,
            refresh_dir / "05_oral_correspondence_grid.jpg",
        )
        self._save_teacher_contact_sheet(
            oral_appearance_grid_rows,
            refresh_dir / "06_oral_appearance_contribution_grid.jpg",
        )
        self.direct_bank = bank_writer.finalize(verify_files=True)

        valid_uv_observations = 0
        coverage = 0.0
        confidence_mean = 0.0
        layer_atlas_metrics: dict[str, Any] = {}
        if accumulator is not None:
            if refreshed_layer_reference is None:
                raise AssertionError(
                    "Base refresh did not capture layer RGB reference"
                )
            self.layer_reference_rgb = refreshed_layer_reference
            refreshed_atlas = accumulator.finalize(
                self.device,
                refresh_step=step,
                teacher_timestep=teacher_timestep,
            )
            if self.edit_allow_mask is not None:
                refreshed_atlas.edit.mul_(self.edit_allow_mask)
            self.atlas = self._merge_refreshed_atlas(
                self.atlas, refreshed_atlas
            )
            valid_uv_observations = int(accumulator.num_views)
            required_layers = set(
                self.config["fusion"]["layered_surface"][
                    "required_effective_layers"
                ]
            )
            minimum_effective = int(
                self.config["fusion"]["layered_surface"][
                    "minimum_effective_gaussians"
                ]
            )
            for layer_name, layer_accumulator in (
                layer_accumulators.items()
            ):
                if layer_accumulator.num_views == 0:
                    raise RuntimeError(
                        f"No valid {layer_name} observations at Base "
                        f"refresh {step}; refusing to train with a missing "
                        "semantic oral atlas"
                    )
                refreshed_layer = layer_accumulator.finalize(
                    self.device,
                    refresh_step=step,
                    teacher_timestep=teacher_timestep,
                )
                # Residuals are relative to this refresh's appearance
                # snapshot, so they are replaced rather than EMA-merged with
                # residuals expressed in an older reference frame. Previous
                # edits already persist in the new current snapshot.
                self.layer_atlases[layer_name] = refreshed_layer
                layer_confidence = self.layer_atlases[
                    layer_name
                ].confidence
                layer_observed = layer_confidence > 0.0
                gaussian_metrics = (
                    self._layer_gaussian_supervision_metrics(
                        layer_name,
                        self.layer_atlases[layer_name],
                    )
                )
                layer_atlas_metrics[layer_name] = {
                    "valid_observations": int(
                        layer_accumulator.num_views
                    ),
                    "coverage": float(
                        layer_observed.float().mean().item()
                    ),
                    "confidence_mean_observed": (
                        float(
                            layer_confidence[layer_observed]
                            .mean()
                            .item()
                        )
                        if layer_observed.any()
                        else 0.0
                    ),
                    **gaussian_metrics,
                }
                if (
                    layer_name in required_layers
                    and gaussian_metrics["effective_gaussians"]
                    < minimum_effective
                ):
                    raise RuntimeError(
                        f"Base refresh {step} produced only "
                        f"{gaussian_metrics['effective_gaussians']} effective "
                        f"{layer_name} Gaussians; required at least "
                        f"{minimum_effective}. Refusing silent zero dental "
                        "supervision."
                    )
            self.save_atlas_visuals(step)
            if self.atlas is None:
                raise AssertionError("Base atlas merge returned None")
            observed = self.atlas.confidence > 0.0
            confidence_mean = (
                float(self.atlas.confidence[observed].mean().item())
                if observed.any()
                else 0.0
            )
            coverage = float(observed.float().mean().item())
        attention_after = dict(
            self._surface_attention_stage_metrics()["runtime"]
        )
        attention_delta = {
            key: float(attention_after[key]) - float(attention_before.get(key, 0))
            for key in attention_after
            if isinstance(attention_after[key], (int, float))
            and not isinstance(attention_after[key], bool)
        }
        attention_config = self.config["surface_attention"]
        if (
            bool(attention_config.get("enabled", True))
            and bool(
                attention_config.get("require_runtime_activity", True)
            )
            and (
                attention_delta.get("contexts_set", 0.0) <= 0.0
                or attention_delta.get("self_attention_calls", 0.0) <= 0.0
                or int(attention_after.get("maximum_joint_views", 0))
                < int(attention_config["min_views"])
            )
        ):
            raise RuntimeError(
                "Surface-correspondence attention was enabled but did not "
                "record valid multi-view denoising activity"
            )

        manifest = {
            "schema_version": 4,
            "config_sha256": self.config_digest,
            "implementation_sha256": self.implementation_digest,
            "refresh_step": int(step),
            "teacher_timestep": int(teacher_timestep),
            "training_stage": (
                "coherent_base"
                if build_canonical_atlases
                else "detail_refinement"
            ),
            "supervision_mode": (
                "canonical_face_and_layer_atlases_plus_direct_teacher"
                if build_canonical_atlases
                else "direct_teacher_same_pose_same_view"
            ),
            "pose_count": len(named_poses),
            "views_per_pose": len(self.canonical_indices),
            "expected_observations": int(expected_observations),
            "processed_observations": int(processed_observations),
            "valid_uv_observations": int(valid_uv_observations),
            "saved_previews": int(previews_saved),
            "save_all_teacher_observations": save_all_observations,
            "configured_previews_per_pose": int(previews_per_pose),
            "preview_view_orders": sorted(preview_view_orders),
            "pose_control": {
                **dict(self.config["pose_control"]),
                "applied_components": [
                    "expression",
                    "jaw_pose",
                    "left_eye_pose",
                    "right_eye_pose",
                ],
                "global_orient": "forced_zero",
                "neck_pose": "forced_zero",
                "translation": "forced_zero",
            },
            "canonical_camera_frame_indices": [
                int(self.assets.frames[index].frame_index)
                for index in self.canonical_indices
            ],
            "atlas": {
                "updated": bool(build_canonical_atlases),
                "training_role": (
                    "Base canonical supervision"
                    if build_canonical_atlases
                    else "frozen; not used as Detail pseudo-ground-truth"
                ),
                "detail_training_uses_atlas": False,
                "coverage": float(coverage),
                "confidence_mean_observed": float(confidence_mean),
                "face_encoding": "absolute RGB",
                    "semantic_layer_encoding": (
                        "encoded signed (teacher-current_render)/"
                        "max(actual_appearance_contribution,residual_floor) "
                        "RGB residual"
                ),
                "semantic_layer_residual_floor": float(
                    self.config["fusion"]["layered_surface"][
                        "residual_decomposition_floor"
                    ]
                ),
                "semantic_layer_reference": (
                    "per-Gaussian SH0 RGB snapshot from this Base refresh"
                    if build_canonical_atlases
                    else "frozen snapshot from the final Base refresh"
                ),
                "semantic_layers": layer_atlas_metrics,
            },
            "direct_bank": self.direct_bank.checkpoint_descriptor(
                relative_to=self.directory
            ),
            "surface_layers": {
                "ids": {
                    name: index
                    for index, name in enumerate(
                        self.avatar.surface_layer_names
                    )
                },
                "selection": "occlusion-aware contribution with ambiguity rejection",
            },
            "surface_attention": {
                "config": dict(attention_config),
                "runtime_before": attention_before,
                "runtime_after": attention_after,
                "runtime_delta": attention_delta,
            },
            "poses": pose_records,
            "artifacts": {
                "current_render_grid": "00_current_render_grid.jpg",
                "flame_condition_grid": "01_flame_condition_grid.jpg",
                "surface_teacher_grid": "02_surface_teacher_grid.jpg",
                "edit_mask_grid": "03_edit_mask_grid.jpg",
                "surface_layer_grid": "04_surface_layer_grid.jpg",
                "oral_correspondence_grid": (
                    "05_oral_correspondence_grid.jpg"
                ),
                "oral_appearance_contribution_grid": (
                    "06_oral_appearance_contribution_grid.jpg"
                ),
                "direct_targets": bank_directory.relative_to(
                    refresh_dir
                ).as_posix(),
            },
        }
        manifest_path = refresh_dir / "manifest.json"
        self._atomic_write_json(manifest, manifest_path)
        self._atomic_write_json(
            {
                "refresh_step": int(step),
                "teacher_timestep": int(teacher_timestep),
                "config_sha256": self.config_digest,
                "implementation_sha256": self.implementation_digest,
                "processed_observations": int(processed_observations),
                "valid_uv_observations": int(valid_uv_observations),
                "direct_observations": int(
                    len(self.direct_bank.observations)
                ),
                "manifest_sha256": file_digest(manifest_path),
            },
            refresh_dir / "_SUCCESS.json",
        )
        self.logger.info(
            "Teacher refresh complete: %d direct targets, %d canonical "
            "observations, coverage=%.3f, mean confidence=%.3f; evidence=%s",
            len(self.direct_bank.observations),
            valid_uv_observations,
            coverage,
            confidence_mean,
            manifest_path,
        )
        torch.cuda.empty_cache()

    def _merge_refreshed_atlas(
        self, previous: Optional[UVAtlas], refreshed: UVAtlas
    ) -> UVAtlas:
        if previous is None:
            return refreshed
        return merge_uv_atlas(
            previous,
            refreshed,
            history_weight=float(
                self.config["fusion"].get("history_weight", 0.5)
            ),
            confidence_decay=float(
                self.config["fusion"].get("confidence_decay", 0.98)
            ),
            edit_decay=float(
                self.config["fusion"].get("edit_decay", 0.95)
            ),
            variance_scale=float(
                self.config["fusion"]["variance_scale"]
            ),
        )

    @torch.inference_mode()
    def _layer_decoded_atlas_visuals(
        self,
        layer_name: str,
        atlas: UVAtlas,
    ) -> dict[str, torch.Tensor]:
        """Splat decoded per-Gaussian oral supervision back to canonical UV."""

        sampled = self._layer_gaussian_supervision(layer_name, atlas)
        point_uv = (
            self.avatar.gaussian.get_uv.detach()
            .transpose(0, 1)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        weight = sampled["weight"].reshape(1, 1, -1, 1)
        resolution = int(self.config["fusion"]["resolution"])
        flip_v = bool(self.config["fusion"].get("flip_v", False))

        def splat_field(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            atlas_sum, atlas_weight = splat_to_uv(
                field.transpose(0, 1)
                .unsqueeze(0)
                .unsqueeze(-1),
                point_uv,
                weight,
                resolution,
                flip_v,
            )
            return (
                atlas_sum / atlas_weight.clamp_min(1.0e-8),
                atlas_weight,
            )

        target, atlas_weight = splat_field(sampled["target"])
        reference, _ = splat_field(sampled["reference"])
        delta_encoded, _ = splat_field(
            (
                0.5
                + 0.5
                * (sampled["target"] - sampled["reference"])
            ).clamp(0.0, 1.0)
        )
        observed = atlas_weight > 0.0
        neutral = torch.full_like(target, 0.5)
        positive = atlas_weight[observed]
        if positive.numel() > 0:
            scale = torch.quantile(
                positive.float(),
                float(
                    self.config["fusion"].get(
                        "coverage_normalization_quantile", 0.9
                    )
                ),
            ).clamp_min(1.0e-8)
            weight_visual = (atlas_weight / scale).clamp(0.0, 1.0)
        else:
            weight_visual = torch.zeros_like(atlas_weight)
        return {
            "decoded_target": torch.where(
                observed.expand_as(target), target, neutral
            ),
            "reference_rgb": torch.where(
                observed.expand_as(reference), reference, neutral
            ),
            "target_delta_encoded": torch.where(
                observed.expand_as(delta_encoded),
                delta_encoded,
                neutral,
            ),
            "effective_weight": weight_visual,
        }

    def save_atlas_visuals(self, step: int) -> None:
        if self.atlas is None:
            return
        if self.layer_atlases:
            if self.layer_reference_rgb is None:
                raise RuntimeError(
                    "Cannot export semantic residual atlases without their "
                    "RGB reference"
                )
            torch.save(
                self.layer_reference_rgb.detach().cpu(),
                self.atlas_dir
                / f"semantic_layer_reference_rgb_{step:06d}.pt",
            )
            self._atomic_write_json(
                {
                    "refresh_step": int(step),
                    "face_encoding": "absolute RGB",
                    "oral_layer_encoding": (
                        "encoded signed RGB residual: encoded=0.5+0.5*"
                        "(teacher-current_render)/"
                        "max(actual_appearance_contribution,residual_floor)"
                    ),
                    "residual_floor": float(
                        self.config["fusion"]["layered_surface"][
                            "residual_decomposition_floor"
                        ]
                    ),
                    "decode": (
                        "target_gaussian_rgb=reference_rgb+"
                        "2*(encoded_residual-0.5)"
                    ),
                    "reference": (
                        "semantic_layer_reference_rgb_"
                        f"{step:06d}.pt"
                    ),
                },
                self.atlas_dir / f"encoding_{step:06d}.json",
            )
        atlases = {"face": self.atlas, **self.layer_atlases}
        for layer_name, atlas in atlases.items():
            directory = self.atlas_dir / layer_name
            directory.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb_u8(atlas.rgb)).save(
                directory
                / (
                    f"rgb_{step:06d}.png"
                    if layer_name == "face"
                    else f"residual_encoded_{step:06d}.png"
                )
            )
            Image.fromarray(gray_u8(atlas.confidence)).save(
                directory / f"confidence_{step:06d}.png"
            )
            Image.fromarray(gray_u8(atlas.edit)).save(
                directory / f"edit_{step:06d}.png"
            )
            support = atlas.support / max(
                float(self.config["fusion"]["min_view_support"]), 1.0
            )
            Image.fromarray(gray_u8(support)).save(
                directory / f"support_{step:06d}.png"
            )
            variance_scale = max(
                float(self.config["fusion"]["variance_scale"]), 1.0e-8
            )
            Image.fromarray(
                gray_u8(
                    (atlas.variance / variance_scale).clamp(0.0, 1.0)
                )
            ).save(directory / f"variance_{step:06d}.png")
            if layer_name != "face":
                decoded = self._layer_decoded_atlas_visuals(
                    layer_name, atlas
                )
                Image.fromarray(
                    rgb_u8(decoded["decoded_target"])
                ).save(
                    directory / f"decoded_target_{step:06d}.png"
                )
                Image.fromarray(
                    rgb_u8(decoded["reference_rgb"])
                ).save(
                    directory / f"reference_rgb_{step:06d}.png"
                )
                Image.fromarray(
                    rgb_u8(decoded["target_delta_encoded"])
                ).save(
                    directory
                    / f"target_delta_encoded_{step:06d}.png"
                )
                Image.fromarray(
                    gray_u8(decoded["effective_weight"])
                ).save(
                    directory / f"effective_weight_{step:06d}.png"
                )

    def atlas_targets(
        self,
        surface_uv: torch.Tensor,
        surface_alpha: torch.Tensor,
        composite_alpha: torch.Tensor,
        surface_uv_variance: torch.Tensor,
        surface_layer_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.atlas is None:
            raise RuntimeError("UV atlas has not been initialized")
        flip_v = bool(self.config["fusion"].get("flip_v", False))
        confidence = sample_uv_atlas(
            self.atlas.confidence, surface_uv, flip_v
        ).clamp(0.0, 1.0)
        surface_target = sample_uv_atlas(
            self.atlas.rgb * self.atlas.confidence, surface_uv, flip_v
        ) / confidence.clamp_min(1.0e-6)
        composite_alpha = composite_alpha.detach().clamp(0.0, 1.0)
        target_alpha = surface_alpha.detach().clamp(0.0, 1.0)
        target = (
            surface_target * target_alpha + (1.0 - target_alpha)
        )
        edit = (
            sample_uv_atlas(
                self.atlas.edit * self.atlas.confidence,
                surface_uv,
                flip_v,
            )
            / confidence.clamp_min(1.0e-6)
        ).clamp(0.0, 1.0)
        visibility_alpha = torch.minimum(
            target_alpha, composite_alpha
        )
        validity = surface_validity(
            surface_uv,
            visibility_alpha,
            float(self.config["fusion"]["alpha_threshold"]),
            float(self.config["fusion"]["uv_jump_threshold"]),
            surface_uv_variance,
            float(
                self.config["fusion"].get(
                    "uv_variance_threshold", float("inf")
                )
            ),
        )
        confidence = confidence.pow(
            float(self.config["fusion"].get("confidence_power", 1.0))
        )
        if surface_layer_ids is not None:
            if surface_layer_ids.shape != confidence.shape:
                raise ValueError(
                    "surface_layer_ids must match atlas confidence shape"
                )
            # The shared face atlas intentionally excludes all oral layers.
            # This guard is essential because cavity and scalp, and upper and
            # lower teeth, can occupy overlapping numeric UV coordinates.
            validity = validity * (surface_layer_ids == 0).to(
                validity.dtype
            )
        mask = confidence * edit * validity * visibility_alpha
        return target.detach(), mask.detach().clamp(0.0, 1.0), confidence.detach()

    def _next_batch(self) -> dict[str, Any]:
        try:
            batch = next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_loader)
            batch = next(self.train_iterator)
        return to_device(batch, self.device)

    def _next_direct_batch(
        self,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
        if self.direct_bank is None:
            raise RuntimeError(
                "Direct teacher supervision was requested before refresh"
            )
        sampled = self.direct_bank.sample(
            int(self.config["data"]["batch_size"]),
            open_mouth_probability=float(
                self.config["detail_supervision"][
                    "open_mouth_probability"
                ]
            ),
        )
        batch = self.builder.build(
            list(sampled.camera_indices), sampled.pose
        )
        actual_frames = tuple(
            int(value)
            for value in torch.as_tensor(batch["frame_index"])
            .reshape(-1)
            .tolist()
        )
        if actual_frames != sampled.frame_indices:
            raise RuntimeError(
                "Direct target camera/pose reconstruction changed frame "
                f"identity: {actual_frames} != {sampled.frame_indices}"
            )
        return (
            to_device(batch, self.device),
            sampled.targets.to(
                device=self.device, dtype=torch.float32
            ),
            sampled.edit_masks.to(
                device=self.device, dtype=torch.float32
            ),
        )

    def _deterministic_direct_preview(
        self, count: int
    ) -> tuple[
        str,
        Any,
        tuple[int, ...],
        tuple[int, ...],
        torch.Tensor,
        torch.Tensor,
    ]:
        """Choose a stable open-mouth pose and evenly spaced saved views."""

        if self.direct_bank is None:
            raise RuntimeError("No direct target bank is available")
        pose_ids = (
            self.direct_bank.open_pose_ids
            if self.direct_bank.open_pose_ids
            else self.direct_bank.pose_ids
        )

        def pose_key(pose_id: str) -> tuple[float, str]:
            jaw = np.asarray(
                self.direct_bank.poses[pose_id].jaw_pose,
                dtype=np.float32,
            ).reshape(-1)
            return (
                float(jaw[0]) if jaw.size else float("-inf"),
                pose_id,
            )

        pose_id = max(pose_ids, key=pose_key)
        observations = sorted(
            self.direct_bank.by_pose[pose_id],
            key=lambda item: (
                int(item.metadata.get("view_order", item.camera_index)),
                int(item.camera_index),
            ),
        )
        positions = evenly_spaced_indices(
            list(range(len(observations))),
            min(int(count), len(observations)),
        )
        selected = tuple(observations[index] for index in positions)
        return (
            pose_id,
            self.direct_bank.poses[pose_id],
            tuple(item.camera_index for item in selected),
            tuple(item.frame_index for item in selected),
            torch.stack(
                [item.target_float() for item in selected], dim=0
            ),
            torch.stack(
                [item.edit_mask_float() for item in selected], dim=0
            ),
        )

    def _layer_gaussian_supervision(
        self,
        layer_name: str,
        atlas: UVAtlas,
    ) -> dict[str, torch.Tensor]:
        """Sample one semantic residual atlas at its bound Gaussian UVs."""

        if self.layer_reference_rgb is None:
            raise RuntimeError(
                "Semantic residual atlases have no RGB reference snapshot"
            )
        reference_rgb = self.layer_reference_rgb.to(
            device=self.device, dtype=torch.float32
        )
        reference_rgb = torch.nan_to_num(
            reference_rgb,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        expected_shape = (int(self.avatar.gaussian.num_gs), 3)
        if reference_rgb.shape != expected_shape:
            raise RuntimeError(
                "Semantic layer RGB reference shape differs from Gaussians"
            )
        point_uv = (
            self.avatar.gaussian.get_uv.detach()
            .transpose(0, 1)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        flip_v = bool(self.config["fusion"].get("flip_v", False))
        confidence_power = float(
            self.config["fusion"].get("confidence_power", 1.0)
        )
        confidence = sample_uv_atlas(
            atlas.confidence, point_uv, flip_v
        ).reshape(-1).clamp(0.0, 1.0)
        encoded_residual = (
            sample_uv_atlas(
                atlas.rgb * atlas.confidence, point_uv, flip_v
            )[0, :, :, 0]
            .transpose(0, 1)
            / confidence.unsqueeze(1).clamp_min(1.0e-6)
        )
        encoded_residual = torch.where(
            (confidence > 0.0).unsqueeze(1),
            encoded_residual,
            torch.full_like(encoded_residual, 0.5),
        )
        target = decode_surface_rgb_residual(
            encoded_residual,
            reference_rgb,
        )
        edit = (
            sample_uv_atlas(
                atlas.edit * atlas.confidence, point_uv, flip_v
            ).reshape(-1)
            / confidence.clamp_min(1.0e-6)
        ).clamp(0.0, 1.0)
        region = self.avatar.surface_layer_masks[layer_name].to(
            dtype=torch.float32, device=self.device
        )
        weight = (
            confidence.pow(confidence_power) * edit * region
        ).detach()
        return {
            "encoded_residual": encoded_residual.detach(),
            "target": target.detach(),
            "reference": reference_rgb.detach(),
            "confidence": confidence.detach(),
            "edit": edit.detach(),
            "weight": weight,
            "region": region.detach(),
        }

    def _layer_gaussian_supervision_metrics(
        self,
        layer_name: str,
        atlas: UVAtlas,
    ) -> dict[str, Any]:
        sampled = self._layer_gaussian_supervision(layer_name, atlas)
        weight = sampled["weight"]
        effective = weight > 1.0e-6
        strong = weight >= 0.01
        region = sampled["region"] > 0.0
        encoded = sampled["encoded_residual"]
        target_delta = (
            sampled["target"] - sampled["reference"]
        ).abs().mean(dim=1)
        weight_sum = weight.sum()
        saturated = (
            (encoded <= 1.0e-4) | (encoded >= 1.0 - 1.0e-4)
        ).to(torch.float32)
        return {
            "gaussians": int(region.sum().item()),
            "confidence_count": int(
                ((sampled["confidence"] > 0.0) & region).sum().item()
            ),
            "effective_gaussians": int(effective.sum().item()),
            "strong_gaussians": int(strong.sum().item()),
            "effective_weight_sum": float(weight_sum.item()),
            "effective_weight_max": float(weight.max().item()),
            "weighted_target_delta_l1": float(
                (target_delta * weight).sum().item()
                / weight_sum.clamp_min(1.0e-8).item()
            ),
            "encoded_saturation_fraction": (
                float(saturated[effective].mean().item())
                if bool(effective.any())
                else 0.0
            ),
        }

    def _layered_oral_loss(
        self,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Supervise oral Gaussian color from four non-overlapping atlases."""

        current_rgb = self.avatar._SH2RGB(
            self.avatar.gaussian._features_dc[:, 0, :]
        )
        zero = current_rgb.sum() * 0.0
        if self.layer_reference_rgb is None:
            if self.layer_atlases:
                raise RuntimeError(
                    "Semantic residual atlases have no RGB reference snapshot"
                )
            return zero, {
                name: zero
                for name in self.avatar.surface_layer_names[1:]
            }
        total_numerator = zero
        total_weight = current_rgb.new_zeros(())
        per_layer: dict[str, torch.Tensor] = {}
        for layer_name in self.avatar.surface_layer_names[1:]:
            atlas = self.layer_atlases.get(layer_name)
            if atlas is None:
                per_layer[layer_name] = zero
                continue
            sampled = self._layer_gaussian_supervision(
                layer_name, atlas
            )
            target = sampled["target"].to(
                device=current_rgb.device, dtype=current_rgb.dtype
            )
            weight = sampled["weight"].to(
                device=current_rgb.device, dtype=current_rgb.dtype
            )
            numerator = (
                (current_rgb - target).abs().mean(dim=1) * weight
            ).sum()
            denominator = weight.sum()
            layer_loss = numerator / denominator.clamp_min(1.0e-8)
            layer_loss = torch.where(
                denominator > 0.0, layer_loss, zero
            )
            per_layer[layer_name] = layer_loss
            total_numerator = total_numerator + numerator
            total_weight = total_weight + denominator
        total = total_numerator / total_weight.clamp_min(1.0e-8)
        total = torch.where(total_weight > 0.0, total, zero)
        return total, per_layer

    def _losses(
        self,
        rendering: RenderBatch,
        direct_target: Optional[torch.Tensor] = None,
        direct_edit_mask: Optional[torch.Tensor] = None,
        include_layered_oral: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if rendering.identity_rgb is None or rendering.identity_alpha is None:
            raise RuntimeError("Training render is missing identity buffers")
        loss_cfg = self.config["loss"]
        direct = direct_target is not None or direct_edit_mask is not None
        if direct:
            if direct_target is None or direct_edit_mask is None:
                raise ValueError(
                    "Direct target and edit mask must be provided together"
                )
            target = direct_target.detach().clamp(0.0, 1.0)
            edit_mask = direct_edit_mask.detach().clamp(0.0, 1.0)
            if (
                target.shape != rendering.rgb.shape
                or edit_mask.shape
                != (
                    rendering.rgb.shape[0],
                    1,
                    rendering.rgb.shape[2],
                    rendering.rgb.shape[3],
                )
            ):
                raise ValueError(
                    "Direct target/mask shapes must match the training render"
                )
            atlas_confidence = torch.ones_like(edit_mask)
        else:
            if (
                rendering.surface_uv is None
                or rendering.surface_alpha is None
                or rendering.surface_uv_variance is None
                or rendering.surface_layer_ids is None
            ):
                raise RuntimeError(
                    "Canonical training render is missing surface buffers"
                )
            target, edit_mask, atlas_confidence = self.atlas_targets(
                rendering.surface_uv,
                rendering.surface_alpha,
                rendering.alpha,
                rendering.surface_uv_variance,
                rendering.surface_layer_ids,
            )
        prediction_raw = rendering.rgb
        prediction = prediction_raw.clamp(0.0, 1.0)
        identity = rendering.identity_rgb.detach()
        oral_protection = torch.zeros_like(edit_mask)
        if not direct:
            if rendering.surface_layer_ids is None:
                raise AssertionError(
                    "Canonical supervision requires semantic layer IDs"
                )
            oral_protection = (
                rendering.surface_layer_ids > 0
            ).to(edit_mask.dtype)
            dilation = max(
                int(self.config["teacher"].get("edit_dilation", 0)),
                0,
            )
            if dilation > 0:
                kernel = 2 * dilation + 1
                oral_protection = F.max_pool2d(
                    oral_protection,
                    kernel_size=kernel,
                    stride=1,
                    padding=dilation,
                )
            oral_protection = oral_protection.clamp(0.0, 1.0)

        edit_l1 = masked_l1(prediction_raw, target, edit_mask)
        edit_composite = (
            prediction * edit_mask + target * (1.0 - edit_mask)
        )
        edit_ssim = 1.0 - self.ssim(edit_composite, target)
        highpass_kernel = int(loss_cfg.get("highpass_kernel", 5))
        prediction_high = prediction_raw - F.avg_pool2d(
            prediction_raw,
            highpass_kernel,
            stride=1,
            padding=highpass_kernel // 2,
        )
        target_high = target - F.avg_pool2d(
            target,
            highpass_kernel,
            stride=1,
            padding=highpass_kernel // 2,
        )
        edit_highpass = masked_l1(
            prediction_high, target_high, edit_mask
        )
        edit_loss = (
            float(loss_cfg["edit_l1_weight"]) * edit_l1
            + float(loss_cfg["edit_ssim_weight"]) * edit_ssim
            + float(loss_cfg.get("edit_highpass_weight", 0.0))
            * edit_highpass
        )

        identity_mask = (
            rendering.identity_alpha.detach()
            * (1.0 - edit_mask)
            * (1.0 - oral_protection)
        ).clamp(0.0, 1.0)
        identity_resolution = int(loss_cfg["identity_resolution"])
        identity_kernel = int(loss_cfg["identity_kernel"])
        prediction_low = lowpass(
            prediction_raw, identity_resolution, identity_kernel
        )
        identity_low = lowpass(
            identity, identity_resolution, identity_kernel
        )
        identity_mask_low = resize_mask(identity_mask, prediction_low)
        identity_full = masked_l1(
            prediction_raw, identity, identity_mask
        )
        identity_lowpass = masked_l1(
            prediction_low, identity_low, identity_mask_low
        )
        identity_loss = (
            float(loss_cfg.get("identity_l1_weight", 1.0)) * identity_full
            + float(loss_cfg.get("identity_lowpass_weight", 0.25))
            * identity_lowpass
        )
        alpha_loss = masked_l1(
            rendering.alpha,
            rendering.identity_alpha.detach(),
            (1.0 - edit_mask) * (1.0 - oral_protection),
        )
        feature_proximal = F.smooth_l1_loss(
            self.avatar.gaussian._features_dc,
            self.avatar.initial["feature_dc"],
        )
        opacity_proximal = F.smooth_l1_loss(
            self.avatar.gaussian._opacity,
            self.avatar.initial["opacity"],
        )
        chroma = (
            F.relu(prediction_raw - 1.0).square()
            + F.relu(-prediction_raw).square()
        ).mean()
        if include_layered_oral:
            layered_oral, per_layer = self._layered_oral_loss()
        else:
            layered_oral = prediction_raw.sum() * 0.0
            per_layer = {
                name: layered_oral
                for name in self.avatar.surface_layer_names[1:]
            }

        total = (
            float(loss_cfg["edit_weight"]) * edit_loss
            + float(loss_cfg["identity_weight"]) * identity_loss
            + float(loss_cfg["alpha_weight"]) * alpha_loss
            + float(loss_cfg["feature_proximal_weight"]) * feature_proximal
            + float(loss_cfg["opacity_proximal_weight"]) * opacity_proximal
            + float(loss_cfg["chroma_weight"]) * chroma
            + float(loss_cfg.get("layered_oral_weight", 0.0))
            * layered_oral
        )
        losses = {
            "total": total,
            "edit": edit_loss,
            "edit_l1": edit_l1,
            "edit_ssim": edit_ssim,
            "edit_highpass": edit_highpass,
            "identity": identity_loss,
            "identity_l1": identity_full,
            "identity_lowpass": identity_lowpass,
            "alpha": alpha_loss,
            "feature_proximal": feature_proximal,
            "opacity_proximal": opacity_proximal,
            "chroma": chroma,
            "layered_oral": layered_oral,
            **{
                f"layered_{name}": value
                for name, value in per_layer.items()
            },
        }
        diagnostics = {
            "target": target,
            "edit_mask": edit_mask,
            "identity_mask": identity_mask,
            "oral_protection": oral_protection,
            "atlas_confidence": atlas_confidence,
            "direct_supervision": torch.full(
                (),
                1.0 if direct else 0.0,
                device=prediction_raw.device,
                dtype=prediction_raw.dtype,
            ),
        }
        return total, losses, diagnostics

    def _log(
        self,
        step: int,
        losses: Mapping[str, torch.Tensor],
        diagnostics: Mapping[str, torch.Tensor],
    ) -> None:
        record: dict[str, Any] = {
            "step": int(step),
            "num_gaussians": int(self.avatar.gaussian.num_gs),
            "atlas_refresh_step": (
                int(self.atlas.refresh_step) if self.atlas is not None else -1
            ),
            "teacher_timestep": (
                int(self.direct_bank.teacher_timestep)
                if self.direct_bank is not None
                else -1
            ),
            "direct_bank_refresh_step": (
                int(self.direct_bank.refresh_step)
                if self.direct_bank is not None
                else -1
            ),
            "supervision_mode": (
                "direct_teacher"
                if float(
                    diagnostics["direct_supervision"].item()
                )
                > 0.5
                else "canonical_face_atlas"
            ),
            "edit_mask_mean": float(
                diagnostics["edit_mask"].mean().item()
            ),
            "atlas_confidence_mean": float(
                diagnostics["atlas_confidence"].mean().item()
            ),
            "oral_protection_mean": float(
                diagnostics["oral_protection"].mean().item()
            ),
        }
        record.update(
            {name: float(value.detach().item()) for name, value in losses.items()}
        )
        self.last_metric_record = dict(record)
        self.metrics_file.write(json.dumps(record) + "\n")
        self.metrics_file.flush()
        if self.writer is not None:
            for name, value in losses.items():
                self.writer.add_scalar(
                    f"train/{name}", float(value.detach().item()), step
                )
            self.writer.add_scalar(
                "train/edit_mask_mean", record["edit_mask_mean"], step
            )
            self.writer.add_scalar(
                "train/atlas_confidence_mean",
                record["atlas_confidence_mean"],
                step,
            )
            self.writer.add_scalar(
                "train/oral_protection_mean",
                record["oral_protection_mean"],
                step,
            )
            for group in self.optimizer.param_groups:
                self.writer.add_scalar(
                    f"train/lr_{group['name']}", group["lr"], step
                )

    @torch.inference_mode()
    def save_preview(
        self,
        step: int,
        destination: Optional[Path] = None,
        force_direct: bool = False,
    ) -> None:
        count = min(
            int(self.config["output"]["preview_views"]),
            len(self.canonical_indices),
        )
        coarse_steps = int(
            self.config["teacher"].get("coarse_iterations", 0)
        )
        direct_mode = bool(force_direct) or int(step) > coarse_steps
        if direct_mode:
            if self.direct_bank is None:
                raise RuntimeError(
                    "Detail preview requires a direct target bank"
                )
            (
                pose_id,
                pose,
                camera_indices,
                frame_indices,
                preview_targets,
                preview_masks,
            ) = self._deterministic_direct_preview(
                count
            )
            batch = to_device(
                self.builder.build(
                    list(camera_indices), pose
                ),
                self.device,
            )
            actual_frames = tuple(
                int(value)
                for value in torch.as_tensor(batch["frame_index"])
                .reshape(-1)
                .tolist()
            )
            if actual_frames != frame_indices:
                raise RuntimeError(
                    "Detail preview camera/pose reconstruction changed "
                    f"frame identity: {actual_frames} != {frame_indices}"
                )
            rendering = self.avatar.render_batch(
                batch,
                include_identity=True,
                include_surface_uv=False,
            )
            if rendering.identity_rgb is None:
                raise RuntimeError(
                    "Detail preview render is missing identity RGB"
                )
            target = preview_targets.to(
                self.device, dtype=torch.float32
            )
            edit_mask = preview_masks.to(
                self.device, dtype=torch.float32
            )
            rows = []
            for index in range(rendering.rgb.shape[0]):
                mask_rgb = edit_mask[index].expand(3, -1, -1)
                error = (
                    rendering.rgb[index].clamp(0.0, 1.0)
                    - target[index]
                ).abs()
                rows.append(
                    np.concatenate(
                        (
                            add_label(
                                rgb_u8(rendering.identity_rgb[index]),
                                "Stage-1 identity",
                            ),
                            add_label(
                                rgb_u8(rendering.rgb[index]),
                                "current direct-fit render",
                            ),
                            add_label(
                                rgb_u8(target[index]),
                                "direct teacher (same pose/view)",
                            ),
                            add_label(
                                rgb_u8(mask_rgb),
                                "direct edit mask (no UV gate)",
                            ),
                            add_label(
                                rgb_u8(error),
                                "current-to-teacher error",
                            ),
                        ),
                        axis=1,
                    )
                )
            path = (
                self.preview_dir / f"step_{step:06d}.jpg"
                if destination is None
                else Path(destination)
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.concatenate(rows, axis=0)).save(
                path, quality=92
            )
            self._atomic_write_json(
                {
                    "supervision_mode": (
                        "direct_teacher_same_pose_same_view"
                    ),
                    "pose_id": pose_id,
                    "pose_source_index": int(pose.source_index),
                    "is_open_mouth": bool(pose.is_open_mouth),
                    "camera_data_indices": list(camera_indices),
                    "camera_frame_indices": list(frame_indices),
                    "direct_bank_refresh_step": int(
                        self.direct_bank.refresh_step
                    ),
                    "image": path.name,
                },
                path.with_suffix(".json"),
            )
            return

        camera_indices = evenly_spaced_indices(self.canonical_indices, count)
        batch = self.builder.build(camera_indices, self.assets.validation_pose)
        batch = to_device(batch, self.device)
        rendering = self.avatar.render_batch(
            batch,
            include_identity=True,
            include_surface_uv=True,
            include_surface_layers=True,
        )
        if (
            rendering.identity_rgb is None
            or rendering.surface_uv is None
            or rendering.surface_alpha is None
            or rendering.surface_uv_variance is None
        ):
            raise RuntimeError("Preview render is missing required buffers")
        target, edit_mask, _ = self.atlas_targets(
            rendering.surface_uv,
            rendering.surface_alpha,
            rendering.alpha,
            rendering.surface_uv_variance,
            rendering.surface_layer_ids,
        )
        rows = []
        for index in range(rendering.rgb.shape[0]):
            mask_rgb = edit_mask[index].expand(3, -1, -1)
            rows.append(
                np.concatenate(
                    (
                        add_label(
                            rgb_u8(rendering.identity_rgb[index]),
                            "Stage-1 identity",
                        ),
                        add_label(
                            rgb_u8(rendering.rgb[index]), "current Stage-2"
                        ),
                        add_label(rgb_u8(target[index]), "coherent UV target"),
                        add_label(rgb_u8(mask_rgb), "trusted edit mask"),
                    ),
                    axis=1,
                )
            )
        path = (
            self.preview_dir / f"step_{step:06d}.jpg"
            if destination is None
            else Path(destination)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.concatenate(rows, axis=0)).save(path, quality=92)

    @torch.inference_mode()
    def render_test_sequence(
        self,
        output_dir: Optional[Path] = None,
        stage_label: str = "detail refinement",
        save_frames: Optional[bool] = None,
        save_masks: Optional[bool] = None,
    ) -> Path:
        """Render the complete assets/test driver for one stage snapshot."""

        import imageio.v2 as imageio

        config = self.config["test_render"]
        stage_config = self.config["stage_outputs"]
        output_dir = (
            self.directory / "test_render"
            if output_dir is None
            else Path(output_dir)
        )
        frame_dir = output_dir / "frames"
        mask_dir = output_dir / "masks"
        if save_frames is None:
            save_frames = bool(
                stage_config.get("save_driving_frames", True)
            )
        if save_masks is None:
            save_masks = bool(
                stage_config.get(
                    "save_driving_masks",
                    config.get("save_masks", True),
                )
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        if save_frames:
            frame_dir.mkdir(parents=True, exist_ok=True)
        if save_masks:
            mask_dir.mkdir(parents=True, exist_ok=True)

        height = int(config["height"])
        width = int(config["width"])
        fps = int(config["fps"])
        frame_count = int(self.assets.test_frame_count)
        camera_index = int(self.assets.test_camera_index)
        expression_path = resolve_path(
            self.config["data"]["test_expression_path"]
        )
        pose_path = resolve_path(self.config["data"]["test_pose_path"])
        video_path = output_dir / "test.mp4"
        sample_count = min(
            int(config["contact_sheet_frames"]), frame_count
        )
        selected = np.unique(
            np.linspace(0, frame_count - 1, sample_count)
            .round()
            .astype(int)
        )
        selected_set = {int(index) for index in selected}
        selected_images: dict[int, np.ndarray] = {}
        writer = imageio.get_writer(str(video_path), fps=fps)
        try:
            for index in tqdm(
                range(frame_count),
                desc="Render assets/test",
                dynamic_ncols=True,
            ):
                pose = self.assets.test_pose(index)
                batch = rescale_render_batch(
                    self.builder.build([camera_index], pose),
                    height,
                    width,
                )
                rendering = self.avatar.render_batch(
                    to_device(batch, self.device),
                    include_identity=False,
                    include_surface_uv=False,
                )
                image = rgb_u8(rendering.rgb[0])
                frame_path = frame_dir / f"{index:06d}.png"
                if save_frames:
                    Image.fromarray(image).save(frame_path)
                if index in selected_set:
                    selected_images[index] = image
                writer.append_data(image)
                if save_masks:
                    Image.fromarray(gray_u8(rendering.alpha[0])).save(
                        mask_dir / f"{index:06d}.png"
                    )
        finally:
            writer.close()
            self.avatar.set_pose(*self.avatar.reference_pose)

        tiles: list[np.ndarray] = []
        for index in selected:
            tile = np.asarray(
                Image.fromarray(selected_images[int(index)]).resize(
                    (256, 256), Image.Resampling.LANCZOS
                ),
                dtype=np.uint8,
            )
            tiles.append(
                add_label(
                    tile,
                    f"{stage_label} | frame {int(index):03d}",
                )
            )
        columns = 4
        rows = int(math.ceil(len(tiles) / float(columns)))
        sheet = np.full(
            (rows * 256, columns * 256, 3), 255, dtype=np.uint8
        )
        for index, tile in enumerate(tiles):
            row, column = divmod(index, columns)
            sheet[
                row * 256 : (row + 1) * 256,
                column * 256 : (column + 1) * 256,
            ] = tile
        Image.fromarray(sheet).save(
            output_dir / "contact_sheet.jpg", quality=92
        )

        metadata = {
            "stage": str(stage_label),
            "frame_count": frame_count,
            "fps": fps,
            "resolution": [height, width],
            "camera_frame_index": int(
                self.assets.frames[camera_index].frame_index
            ),
            "expression_path": str(expression_path),
            "expression_sha256": file_digest(expression_path),
            "pose_path": str(pose_path),
            "pose_sha256": file_digest(pose_path),
            "pose_layout": {
                "global": "ignored pose[:,0:3]",
                "neck": "ignored pose[:,3:6]",
                "jaw": "applied pose[:,6:9]",
                "left_eye": "applied pose[:,9:12]",
                "right_eye": "applied pose[:,12:15]",
            },
            "video": video_path.name,
            "frames": "frames" if save_frames else None,
            "masks": "masks" if save_masks else None,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.logger.info(
            "Rendered stage %s: %d test frames at camera %03d: %s",
            stage_label,
            frame_count,
            metadata["camera_frame_index"],
            video_path,
        )
        return video_path

    @staticmethod
    def _atomic_write_json(value: Mapping[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                dict(value),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _stage_is_complete(
        self, stage_id: str, completed_steps: int
    ) -> bool:
        marker = self.stages_dir / stage_id / "_SUCCESS.json"
        if not marker.is_file():
            return False
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            value.get("stage_id") == stage_id
            and int(value.get("completed_steps", -1))
            == int(completed_steps)
            and value.get("config_sha256") == self.config_digest
            and value.get("implementation_sha256")
            == self.implementation_digest
        )

    def _write_stage_index(self) -> None:
        stages: list[dict[str, Any]] = []
        expected_steps = {
            "00_stage1_input": 0,
            "01_geometry_stabilized": 0,
            "02_coherent_base": int(
                self.config["teacher"].get("coarse_iterations", 0)
            ),
            "03_detail_refinement": int(
                self.config["optimization"]["iterations"]
            ),
        }
        for stage_id, stage_label, description in STAGE_DEFINITIONS:
            stage_dir = self.stages_dir / stage_id
            marker_path = stage_dir / "_SUCCESS.json"
            marker: Optional[dict[str, Any]] = None
            if marker_path.is_file():
                try:
                    loaded = json.loads(
                        marker_path.read_text(encoding="utf-8")
                    )
                    if isinstance(loaded, dict):
                        marker = loaded
                except (OSError, json.JSONDecodeError):
                    marker = None
            stages.append(
                {
                    "stage_id": stage_id,
                    "stage_label": stage_label,
                    "description": description,
                    "path": stage_id,
                    "status": (
                        "complete"
                        if self._stage_is_complete(
                            stage_id, expected_steps[stage_id]
                        )
                        else "pending"
                    ),
                    "completed_steps": (
                        int(marker["completed_steps"])
                        if marker is not None
                        and "completed_steps" in marker
                        else None
                    ),
                    "artifacts": {
                        "success": f"{stage_id}/_SUCCESS.json",
                        "metrics": f"{stage_id}/metrics.json",
                        "diagnostics": (
                            f"{stage_id}/diagnostics/render_grid.jpg"
                        ),
                        "surface_layers": (
                            f"{stage_id}/diagnostics/surface_layer_grid.jpg"
                        ),
                        "oral_correspondence": (
                            f"{stage_id}/diagnostics/"
                            "oral_correspondence_grid.jpg"
                        ),
                        "oral_appearance_contribution": (
                            f"{stage_id}/diagnostics/"
                            "oral_appearance_contribution_grid.jpg"
                        ),
                        "driving": f"{stage_id}/driving/test.mp4",
                        "model": f"{stage_id}/model",
                        "supervision": (
                            f"{stage_id}/supervision_grid.jpg"
                            if stage_id
                            in (
                                "02_coherent_base",
                                "03_detail_refinement",
                            )
                            else None
                        ),
                        "direct_targets": (
                            f"{stage_id}/direct_targets"
                            if stage_id == "03_detail_refinement"
                            else None
                        ),
                        "base_direct_supervision": (
                            f"{stage_id}/direct_supervision_grid.jpg"
                            if stage_id == "02_coherent_base"
                            else None
                        ),
                    },
                }
            )
        self._atomic_write_json(
            {
                "schema_version": 3,
                "layout": "numbered stage directories are direct children of the run root",
                "stage_order": [
                    stage_id for stage_id, _, _ in STAGE_DEFINITIONS
                ],
                "stages": stages,
                "teacher_evidence": "teacher/step_*/manifest.json",
                "teacher_schedule": teacher_observation_schedule(
                    self.config
                ),
                "comparisons": {
                    "geometry": "geometry_stability_comparison.jpg",
                    "driving": "driving_comparison.jpg",
                    "diagnostics": "diagnostic_comparison.jpg",
                    "metrics": "metrics_summary.json",
                },
                "run_success": "_RUN_SUCCESS.json",
            },
            self.directory / "stage_index.json",
        )

    def _copy_source_model(self, destination: Path) -> None:
        source = self.avatar.reconstruction_dir / "model"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("uvd.ply", "world.ply", "reconstruction_params.npz"):
            source_path = source / name
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Stage-1 source model is missing {source_path}"
                )
            shutil.copy2(source_path, destination / name)

    def _latest_metric_at_or_before(
        self, completed_steps: int
    ) -> Optional[dict[str, Any]]:
        if (
            self.last_metric_record is not None
            and int(self.last_metric_record.get("step", -1))
            <= int(completed_steps)
        ):
            return dict(self.last_metric_record)
        self.metrics_file.flush()
        path = self.directory / "metrics.jsonl"
        if not path.is_file():
            return None
        latest: Optional[dict[str, Any]] = None
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if int(record.get("step", -1)) <= int(completed_steps):
                    latest = record
        return latest

    @torch.inference_mode()
    def _model_stage_metrics(self) -> dict[str, Any]:
        feature = self.avatar.gaussian._features_dc.detach()
        feature_initial = self.avatar.initial["feature_dc"]
        opacity = torch.sigmoid(self.avatar.gaussian._opacity.detach())
        opacity_initial = self.avatar.initial_opacity
        opacity_increase = opacity - opacity_initial
        layer_metrics = {}
        for name in self.avatar.surface_layer_names:
            mask = self.avatar.surface_layer_masks[name]
            layer_metrics[name] = {
                "gaussians": int(mask.sum().item()),
                "feature_delta_l1": (
                    float(
                        (
                            feature[mask] - feature_initial[mask]
                        )
                        .abs()
                        .mean()
                        .item()
                    )
                    if bool(mask.any())
                    else 0.0
                ),
                "opacity_mean": (
                    float(opacity[mask].mean().item())
                    if bool(mask.any())
                    else 0.0
                ),
                "opacity_delta_l1": (
                    float(
                        (
                            opacity[mask] - opacity_initial[mask]
                        )
                        .abs()
                        .mean()
                        .item()
                    )
                    if bool(mask.any())
                    else 0.0
                ),
                "opacity_increase_max": (
                    float(opacity_increase[mask].max().item())
                    if bool(mask.any())
                    else 0.0
                ),
            }
        return {
            "num_gaussians": int(self.avatar.gaussian.num_gs),
            "feature_delta_l1": float(
                (feature - feature_initial).abs().mean().item()
            ),
            "feature_delta_rms": float(
                (feature - feature_initial).square().mean().sqrt().item()
            ),
            "opacity_mean": float(opacity.mean().item()),
            "opacity_delta_l1": float(
                (opacity - opacity_initial).abs().mean().item()
            ),
            "opacity_increase_max": float(
                opacity_increase.max().item()
            ),
            "opacity_increased_count": int(
                (opacity_increase > 1.0e-6).sum().item()
            ),
            "semantic_layers": layer_metrics,
            "root_motion_nonzero": {
                name: int(
                    torch.count_nonzero(
                        getattr(self.avatar.gaussian, name)
                    ).item()
                )
                for name in (
                    "_global_orient",
                    "_neck_pose",
                    "_translation",
                )
            },
        }

    @torch.inference_mode()
    def _atlas_stage_metrics(self) -> Optional[dict[str, Any]]:
        if self.atlas is None:
            return None
        def metrics(atlas: UVAtlas) -> dict[str, Any]:
            confidence = atlas.confidence.detach()
            observed = confidence > 0.0
            observed_count = int(observed.sum().item())
            highpass = atlas.rgb - F.avg_pool2d(
                atlas.rgb.unsqueeze(0),
                kernel_size=5,
                stride=1,
                padding=2,
            )[0]
            weight = confidence.expand_as(highpass)
            return {
                "refresh_step": int(atlas.refresh_step),
                "teacher_timestep": int(atlas.teacher_timestep),
                "coverage": float(observed.float().mean().item()),
                "observed_texels": observed_count,
                "confidence_mean_observed": (
                    float(confidence[observed].mean().item())
                    if observed_count
                    else 0.0
                ),
                "edit_mean_observed": (
                    float(atlas.edit[observed].mean().item())
                    if observed_count
                    else 0.0
                ),
                "support_mean_observed": (
                    float(atlas.support[observed].mean().item())
                    if observed_count
                    else 0.0
                ),
                "variance_mean_observed": (
                    float(atlas.variance[observed].mean().item())
                    if observed_count
                    else 0.0
                ),
                "weighted_highpass_l1": float(
                    (highpass.abs() * weight).sum().item()
                    / weight.sum().clamp_min(1.0e-8).item()
                ),
            }

        return {
            "supervision": (
                "absolute face atlas plus independent oral residual atlases"
            ),
            "face_encoding": "absolute RGB",
            "semantic_layer_encoding": (
                "encoded signed (teacher-current_render)/"
                "max(actual_appearance_contribution,residual_floor) "
                "RGB residual"
            ),
            "semantic_layer_residual_floor": float(
                self.config["fusion"]["layered_surface"][
                    "residual_decomposition_floor"
                ]
            ),
            "semantic_layer_reference": (
                "per-Gaussian SH0 RGB snapshot"
            ),
            "face": metrics(self.atlas),
            "semantic_layers": {
                name: {
                    **metrics(atlas),
                    **self._layer_gaussian_supervision_metrics(
                        name, atlas
                    ),
                }
                for name, atlas in self.layer_atlases.items()
            },
        }

    def _surface_attention_stage_metrics(self) -> dict[str, Any]:
        config = self.config["surface_attention"]
        metrics: dict[str, Any] = {
            "enabled": bool(config.get("enabled", True)),
            "strength": float(config["strength"]),
            "min_views": int(config["min_views"]),
            "atlas_resolution": int(config["atlas_resolution"]),
        }
        if (
            self.teacher is not None
            and self.teacher.surface_attention is not None
        ):
            metrics["runtime"] = (
                self.teacher.surface_attention.diagnostics()
            )
        elif self.pending_attention_diagnostics is not None:
            metrics["runtime"] = dict(
                self.pending_attention_diagnostics
            )
        else:
            metrics["runtime"] = {
                "installed": False,
                "wrapped_processors": 0,
                "contexts_set": 0,
                "self_attention_calls": 0,
                "denoise_progress_updates": 0,
                "maximum_joint_views": 0,
                "visible_surface_tokens": 0,
            }
        return metrics

    @torch.inference_mode()
    def _save_stage_atlas(self, directory: Path) -> None:
        if self.atlas is None:
            return
        directory.mkdir(parents=True, exist_ok=True)
        if self.layer_atlases:
            if self.layer_reference_rgb is None:
                raise RuntimeError(
                    "Cannot export semantic residual atlases without their "
                    "RGB reference"
                )
            torch.save(
                self.layer_reference_rgb.detach().cpu(),
                directory / "semantic_layer_reference_rgb.pt",
            )
            self._atomic_write_json(
                {
                    "face_encoding": "absolute RGB",
                    "oral_layer_encoding": (
                        "encoded signed RGB residual: encoded=0.5+0.5*"
                        "(teacher-current_render)/"
                        "max(actual_appearance_contribution,residual_floor)"
                    ),
                    "residual_floor": float(
                        self.config["fusion"]["layered_surface"][
                            "residual_decomposition_floor"
                        ]
                    ),
                    "decode": (
                        "target_gaussian_rgb=reference_rgb+"
                        "2*(encoded_residual-0.5)"
                    ),
                    "reference": "semantic_layer_reference_rgb.pt",
                },
                directory / "encoding.json",
            )
        for layer_name, atlas in {
            "face": self.atlas,
            **self.layer_atlases,
        }.items():
            layer_dir = directory / layer_name
            layer_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb_u8(atlas.rgb)).save(
                layer_dir
                / (
                    "rgb.png"
                    if layer_name == "face"
                    else "residual_encoded.png"
                )
            )
            Image.fromarray(gray_u8(atlas.confidence)).save(
                layer_dir / "confidence.png"
            )
            Image.fromarray(gray_u8(atlas.edit)).save(
                layer_dir / "edit.png"
            )
            support = atlas.support / max(
                float(self.config["fusion"]["min_view_support"]), 1.0
            )
            Image.fromarray(gray_u8(support)).save(
                layer_dir / "support.png"
            )
            variance_scale = max(
                float(self.config["fusion"]["variance_scale"]), 1.0e-8
            )
            Image.fromarray(
                gray_u8(
                    (atlas.variance / variance_scale).clamp(0.0, 1.0)
                )
            ).save(layer_dir / "variance.png")
            if layer_name != "face":
                decoded = self._layer_decoded_atlas_visuals(
                    layer_name, atlas
                )
                Image.fromarray(
                    rgb_u8(decoded["decoded_target"])
                ).save(layer_dir / "decoded_target.png")
                Image.fromarray(
                    rgb_u8(decoded["reference_rgb"])
                ).save(layer_dir / "reference_rgb.png")
                Image.fromarray(
                    rgb_u8(decoded["target_delta_encoded"])
                ).save(layer_dir / "target_delta_encoded.png")
                Image.fromarray(
                    gray_u8(decoded["effective_weight"])
                ).save(layer_dir / "effective_weight.png")
            torch.save(atlas.state_dict(), layer_dir / "state.pt")

    @torch.inference_mode()
    def save_stage_diagnostics(
        self, directory: Path, stage_label: str
    ) -> dict[str, Any]:
        """Render the same pose/view grid for every stage."""

        config = self.config["stage_outputs"]
        view_count = min(
            int(config["diagnostic_views"]),
            len(self.canonical_indices),
        )
        camera_indices = evenly_spaced_indices(
            self.canonical_indices, view_count
        )
        named_poses = named_pose_envelope(
            self.assets,
            config["diagnostic_pose_quantiles"],
            include_validation=True,
        )
        render_dir = directory / "renders"
        alpha_dir = directory / "alpha"
        validity_dir = directory / "surface_validity"
        layer_id_dir = directory / "surface_layer_id"
        layer_root = directory / "surface_layers"
        for path in (
            render_dir,
            alpha_dir,
            validity_dir,
            layer_id_dir,
            layer_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        layer_directories = {}
        for layer_name in self.avatar.surface_layer_names:
            layer_directories[layer_name] = {
                field: layer_root / layer_name / field
                for field in (
                    "uv",
                    "alpha",
                    "variance",
                    "depth",
                    "visibility",
                    "appearance_contribution",
                )
            }
            for path in layer_directories[layer_name].values():
                path.mkdir(parents=True, exist_ok=True)

        tile_size = 192
        rows: list[np.ndarray] = []
        layer_rows: list[np.ndarray] = []
        oral_correspondence_rows: list[np.ndarray] = []
        oral_appearance_rows: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        fusion = self.config["fusion"]
        for pose_label, pose in named_poses:
            batch = to_device(
                self.builder.build(camera_indices, pose), self.device
            )
            rendering = self.avatar.render_batch(
                batch,
                include_identity=False,
                include_surface_uv=True,
                include_surface_layers=True,
                include_appearance_contributions=True,
            )
            if (
                rendering.surface_uv is None
                or rendering.surface_alpha is None
                or rendering.surface_uv_variance is None
                or rendering.surface_layer_ids is None
                or rendering.surface_layers is None
                or any(
                    layer.appearance_contribution is None
                    for layer in rendering.surface_layers.values()
                )
            ):
                raise RuntimeError(
                    "Stage diagnostics require surface UV buffers"
                )
            validity = surface_validity(
                rendering.surface_uv,
                torch.minimum(rendering.surface_alpha, rendering.alpha),
                float(fusion["alpha_threshold"]),
                float(fusion["uv_jump_threshold"]),
                rendering.surface_uv_variance,
                float(fusion["uv_variance_threshold"]),
            )
            highpass = rendering.rgb - F.avg_pool2d(
                rendering.rgb,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            row_tiles: list[np.ndarray] = []
            row_layer_tiles: list[np.ndarray] = []
            row_oral_correspondence_tiles: list[np.ndarray] = []
            row_oral_appearance_tiles: list[np.ndarray] = []
            for local_index, camera_index in enumerate(camera_indices):
                frame_index = int(
                    self.assets.frames[camera_index].frame_index
                )
                stem = f"{pose_label}_frame_{frame_index:03d}"
                image = rgb_u8(rendering.rgb[local_index])
                Image.fromarray(image).save(render_dir / f"{stem}.png")
                Image.fromarray(
                    gray_u8(rendering.alpha[local_index])
                ).save(alpha_dir / f"{stem}.png")
                Image.fromarray(
                    gray_u8(validity[local_index])
                ).save(validity_dir / f"{stem}.png")
                layer_image = surface_layer_u8(
                    rendering.surface_layer_ids[local_index]
                )
                Image.fromarray(layer_image).save(
                    layer_id_dir / f"{stem}.png"
                )
                oral_correspondence = torch.stack(
                    [
                        rendering.surface_layers[
                            name
                        ].contribution[local_index]
                        for name in self.avatar.surface_layer_names[1:]
                    ]
                ).sum(dim=0).clamp(0.0, 1.0)
                oral_appearance_values = [
                    rendering.surface_layers[
                        name
                    ].appearance_contribution
                    for name in self.avatar.surface_layer_names[1:]
                ]
                if any(
                    value is None for value in oral_appearance_values
                ):
                    raise AssertionError(
                        "Stage diagnostics lost actual oral appearance "
                        "contributions"
                    )
                oral_appearance = torch.stack(
                    [
                        value[local_index]
                        for value in oral_appearance_values
                        if value is not None
                    ]
                ).sum(dim=0).clamp(0.0, 1.0)
                oral_correspondence_image = rgb_u8(
                    oral_correspondence.expand(3, -1, -1)
                )
                oral_appearance_image = rgb_u8(
                    oral_appearance.expand(3, -1, -1)
                )
                for layer_name, layer in rendering.surface_layers.items():
                    layer_dirs = layer_directories[layer_name]
                    uv_rgb = torch.cat(
                        (
                            layer.uv[local_index],
                            torch.zeros_like(
                                layer.uv[local_index, :1]
                            ),
                        ),
                        dim=0,
                    )
                    Image.fromarray(rgb_u8(uv_rgb)).save(
                        layer_dirs["uv"] / f"{stem}.png"
                    )
                    Image.fromarray(
                        gray_u8(layer.alpha[local_index])
                    ).save(layer_dirs["alpha"] / f"{stem}.png")
                    variance_scale = max(
                        float(fusion["uv_variance_threshold"]),
                        1.0e-8,
                    )
                    Image.fromarray(
                        gray_u8(
                            (
                                layer.uv_variance[local_index]
                                / variance_scale
                            ).clamp(0.0, 1.0)
                        )
                    ).save(
                        layer_dirs["variance"] / f"{stem}.png"
                    )
                    depth = layer.depth[local_index].detach()
                    depth_valid = layer.alpha[local_index] > 1.0e-6
                    if bool(depth_valid.any()):
                        depth_values = depth[depth_valid]
                        depth_min = depth_values.min()
                        depth_max = depth_values.max()
                        depth_visual = (
                            (depth - depth_min)
                            / (depth_max - depth_min).clamp_min(1.0e-6)
                        )
                    else:
                        depth_visual = torch.zeros_like(depth)
                    Image.fromarray(gray_u8(depth_visual)).save(
                        layer_dirs["depth"] / f"{stem}.png"
                    )
                    np.save(
                        layer_dirs["depth"] / f"{stem}.npy",
                        depth.cpu().numpy(),
                    )
                    Image.fromarray(
                        gray_u8(layer.contribution[local_index])
                    ).save(
                        layer_dirs["visibility"] / f"{stem}.png"
                    )
                    if layer.appearance_contribution is None:
                        raise AssertionError(
                            "Stage diagnostics lost a layer appearance "
                            "contribution"
                        )
                    Image.fromarray(
                        gray_u8(
                            layer.appearance_contribution[local_index]
                        )
                    ).save(
                        layer_dirs["appearance_contribution"]
                        / f"{stem}.png"
                    )
                tile = np.asarray(
                    Image.fromarray(image).resize(
                        (tile_size, tile_size),
                        Image.Resampling.LANCZOS,
                    ),
                    dtype=np.uint8,
                )
                row_tiles.append(
                    add_label(
                        tile,
                        f"{stage_label} | {pose_label} | {frame_index:03d}",
                    )
                )
                layer_tile = np.asarray(
                    Image.fromarray(layer_image).resize(
                        (tile_size, tile_size),
                        Image.Resampling.NEAREST,
                    ),
                    dtype=np.uint8,
                )
                row_layer_tiles.append(
                    add_label(
                        layer_tile,
                        f"surface ID | {pose_label} | {frame_index:03d}",
                    )
                )
                oral_correspondence_tile = np.asarray(
                    Image.fromarray(oral_correspondence_image).resize(
                        (tile_size, tile_size),
                        Image.Resampling.LANCZOS,
                    ),
                    dtype=np.uint8,
                )
                row_oral_correspondence_tiles.append(
                    add_label(
                        oral_correspondence_tile,
                        "oral correspondence | "
                        f"{pose_label} | {frame_index:03d}",
                    )
                )
                oral_appearance_tile = np.asarray(
                    Image.fromarray(oral_appearance_image).resize(
                        (tile_size, tile_size),
                        Image.Resampling.LANCZOS,
                    ),
                    dtype=np.uint8,
                )
                row_oral_appearance_tiles.append(
                    add_label(
                        oral_appearance_tile,
                        "oral RGB contribution | "
                        f"{pose_label} | {frame_index:03d}",
                    )
                )
                visible = rendering.surface_alpha[
                    local_index
                ] > float(fusion["alpha_threshold"])
                records.append(
                    {
                        "pose": pose_label,
                        "pose_source_index": int(pose.source_index),
                        "camera_frame_index": frame_index,
                        "alpha_coverage": float(
                            (
                                rendering.alpha[local_index]
                                > float(fusion["alpha_threshold"])
                            )
                            .float()
                            .mean()
                            .item()
                        ),
                        "surface_valid_fraction": float(
                            validity[local_index].mean().item()
                        ),
                        "uv_variance_mean_visible": (
                            float(
                                rendering.surface_uv_variance[
                                    local_index
                                ][visible].mean().item()
                            )
                            if bool(visible.any())
                            else 0.0
                        ),
                        "render_highpass_l1": float(
                            highpass[local_index].abs().mean().item()
                        ),
                        "surface_layer_valid_fraction": float(
                            (
                                rendering.surface_layer_ids[local_index]
                                >= 0
                            )
                            .float()
                            .mean()
                            .item()
                        ),
                        "layer_contributions": {
                            name: {
                                "correspondence_mean": float(
                                    layer.contribution[local_index]
                                    .mean()
                                    .item()
                                ),
                                "appearance_mean": float(
                                    layer.appearance_contribution[
                                        local_index
                                    ]
                                    .mean()
                                    .item()
                                ),
                            }
                            for name, layer in rendering.surface_layers.items()
                            if layer.appearance_contribution is not None
                        },
                    }
                )
            rows.append(np.concatenate(row_tiles, axis=1))
            layer_rows.append(np.concatenate(row_layer_tiles, axis=1))
            oral_correspondence_rows.append(
                np.concatenate(
                    row_oral_correspondence_tiles, axis=1
                )
            )
            oral_appearance_rows.append(
                np.concatenate(row_oral_appearance_tiles, axis=1)
            )

        contact_sheet = np.concatenate(rows, axis=0)
        Image.fromarray(contact_sheet).save(
            directory / "render_grid.jpg", quality=92
        )
        Image.fromarray(np.concatenate(layer_rows, axis=0)).save(
            directory / "surface_layer_grid.jpg", quality=92
        )
        Image.fromarray(
            np.concatenate(oral_correspondence_rows, axis=0)
        ).save(directory / "oral_correspondence_grid.jpg", quality=92)
        Image.fromarray(
            np.concatenate(oral_appearance_rows, axis=0)
        ).save(
            directory / "oral_appearance_contribution_grid.jpg",
            quality=92,
        )
        summary = {
            "stage": stage_label,
            "poses": [label for label, _ in named_poses],
            "camera_frame_indices": [
                int(self.assets.frames[index].frame_index)
                for index in camera_indices
            ],
            "observations": len(records),
            "alpha_coverage_mean": float(
                np.mean([record["alpha_coverage"] for record in records])
            ),
            "surface_valid_fraction_mean": float(
                np.mean(
                    [
                        record["surface_valid_fraction"]
                        for record in records
                    ]
                )
            ),
            "uv_variance_mean_visible": float(
                np.mean(
                    [
                        record["uv_variance_mean_visible"]
                        for record in records
                    ]
                )
            ),
            "render_highpass_l1_mean": float(
                np.mean(
                    [
                        record["render_highpass_l1"]
                        for record in records
                    ]
                )
            ),
            "surface_layers": {
                "ids": {
                    name: index
                    for index, name in enumerate(
                        self.avatar.surface_layer_names
                    )
                },
                "buffers": (
                    "surface_layers/<layer>/{uv,alpha,variance,depth,"
                    "visibility,appearance_contribution}"
                ),
                "layer_id_grid": "surface_layer_grid.jpg",
                "oral_correspondence_grid": (
                    "oral_correspondence_grid.jpg"
                ),
                "oral_appearance_contribution_grid": (
                    "oral_appearance_contribution_grid.jpg"
                ),
            },
            "records": records,
        }
        self._atomic_write_json(summary, directory / "metrics.json")
        self.avatar.set_pose(*self.avatar.reference_pose)
        return summary

    def export_stage(
        self,
        stage_id: str,
        stage_label: str,
        completed_steps: int,
        *,
        geometry_variant: str = "current",
        appearance_variant: str = "current",
        include_atlas: bool = True,
        source_model: bool = False,
    ) -> Path:
        """Persist one independently inspectable algorithm-stage snapshot."""

        stage_config = self.config["stage_outputs"]
        stage_dir = self.stages_dir / stage_id
        if not bool(stage_config.get("enabled", True)):
            return stage_dir
        if (
            bool(stage_config.get("skip_completed", True))
            and self._stage_is_complete(stage_id, completed_steps)
        ):
            self.logger.info(
                "Stage artifact already complete, skipping: %s", stage_dir
            )
            return stage_dir

        stage_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "Exporting stage artifact %s at completed step %d",
            stage_label,
            completed_steps,
        )
        diagnostic_metrics: Optional[dict[str, Any]] = None
        with self.avatar.appearance_variant(appearance_variant):
            with self.avatar.geometry_variant(geometry_variant):
                if bool(stage_config.get("save_models", True)):
                    model_dir = stage_dir / "model"
                    if source_model:
                        self._copy_source_model(model_dir)
                    else:
                        self.avatar.save(model_dir)
                if bool(
                    stage_config.get("render_diagnostics", True)
                ):
                    diagnostic_metrics = self.save_stage_diagnostics(
                        stage_dir / "diagnostics", stage_label
                    )
                if bool(stage_config.get("render_driving", True)):
                    self.render_test_sequence(
                        output_dir=stage_dir / "driving",
                        stage_label=stage_label,
                        save_frames=bool(
                            stage_config.get(
                                "save_driving_frames", True
                            )
                        ),
                        save_masks=bool(
                            stage_config.get(
                                "save_driving_masks", True
                            )
                        ),
                    )
                model_metrics = self._model_stage_metrics()
        model_dir = stage_dir / "model"
        model_metrics["files"] = {
            name: {
                "bytes": int((model_dir / name).stat().st_size),
                "sha256": file_digest(model_dir / name),
            }
            for name in (
                "uvd.ply",
                "world.ply",
                "reconstruction_params.npz",
            )
            if (model_dir / name).is_file()
        }

        atlas_metrics = (
            self._atlas_stage_metrics()
            if include_atlas
            else None
        )
        direct_stage = stage_id == "03_detail_refinement"
        if include_atlas and self.atlas is not None:
            self._save_stage_atlas(stage_dir / "atlas")
        if (
            (not direct_stage and include_atlas and self.atlas is not None)
            or (direct_stage and self.direct_bank is not None)
        ):
            self.save_preview(
                completed_steps,
                destination=stage_dir / "supervision_grid.jpg",
            )
        if (
            stage_id == "02_coherent_base"
            and self.direct_bank is not None
        ):
            self.save_preview(
                completed_steps,
                destination=stage_dir / "direct_supervision_grid.jpg",
                force_direct=True,
            )
        direct_supervision: Optional[dict[str, Any]] = None
        if direct_stage:
            if self.direct_bank is None:
                raise RuntimeError(
                    "Detail stage cannot be exported without direct targets"
                )
            copied_bank_dir = stage_dir / "direct_targets"
            shutil.copytree(
                self.direct_bank.directory,
                copied_bank_dir,
                dirs_exist_ok=True,
            )
            copied_bank = DetailTargetBank(
                copied_bank_dir / "manifest.json",
                expected_config_sha256=self.config_digest,
                expected_implementation_sha256=(
                    self.implementation_digest
                ),
                expected_manifest_sha256=(
                    self.direct_bank.manifest_sha256
                ),
                verify_files=True,
            )
            direct_supervision = {
                "mode": "direct_teacher_same_pose_same_view",
                "uv_atlas_used_as_pseudo_ground_truth": False,
                "edit_mask_uv_gated": False,
                "bank": copied_bank.checkpoint_descriptor(
                    relative_to=stage_dir
                ),
                "observation_count": len(
                    copied_bank.observations
                ),
                "pose_count": len(copied_bank.pose_ids),
                "base_atlas": "frozen reference only",
            }
            self._atomic_write_json(
                direct_supervision,
                stage_dir / "direct_supervision.json",
            )
        stability = self.avatar.stability_report or {}
        geometry_metrics = (
            stability.get("before")
            if geometry_variant == "raw"
            else stability.get("after")
        )
        attention_metrics = self._surface_attention_stage_metrics()
        if int(completed_steps) == 0:
            attention_metrics["runtime"] = {
                "installed": False,
                "wrapped_processors": 0,
                "contexts_set": 0,
                "self_attention_calls": 0,
                "denoise_progress_updates": 0,
                "maximum_joint_views": 0,
                "visible_surface_tokens": 0,
            }
        metrics = {
            "schema_version": 3,
            "stage_id": stage_id,
            "stage_label": stage_label,
            "completed_steps": int(completed_steps),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": self.config_digest,
            "implementation_sha256": self.implementation_digest,
            "geometry_variant": geometry_variant,
            "appearance_variant": appearance_variant,
            "pose_control": dict(self.config["pose_control"]),
            "model": model_metrics,
            "geometry_envelope": geometry_metrics,
            "atlas": atlas_metrics,
            "supervision": (
                direct_supervision
                if direct_stage
                else {
                    "mode": (
                        "canonical_face_and_independent_oral_residual_"
                        "atlases_plus_direct_teacher"
                        if include_atlas and self.atlas is not None
                        else "none"
                    ),
                    "base_direct_probability": (
                        float(
                            self.config["detail_supervision"][
                                "base_direct_probability"
                            ]
                        )
                        if stage_id == "02_coherent_base"
                        else 0.0
                    ),
                }
            ),
            "diagnostics": diagnostic_metrics,
            "latest_training_metric": self._latest_metric_at_or_before(
                completed_steps
            ),
            "surface_attention": attention_metrics,
            "artifacts": {
                "model": (
                    "model"
                    if bool(stage_config.get("save_models", True))
                    else None
                ),
                "diagnostics": (
                    "diagnostics"
                    if diagnostic_metrics is not None
                    else None
                ),
                "driving": (
                    "driving"
                    if bool(stage_config.get("render_driving", True))
                    else None
                ),
                "atlas": (
                    "atlas"
                    if include_atlas and self.atlas is not None
                    else None
                ),
                "atlas_role": (
                    "frozen_coherent_base_reference"
                    if direct_stage
                    and include_atlas
                    and self.atlas is not None
                    else (
                        "training_supervision"
                        if include_atlas and self.atlas is not None
                        else None
                    )
                ),
                "supervision_grid": (
                    "supervision_grid.jpg"
                    if (
                        (not direct_stage and include_atlas and self.atlas is not None)
                        or (direct_stage and self.direct_bank is not None)
                    )
                    else None
                ),
                "base_direct_supervision_grid": (
                    "direct_supervision_grid.jpg"
                    if stage_id == "02_coherent_base"
                    and self.direct_bank is not None
                    else None
                ),
                "direct_supervision": (
                    "direct_supervision.json" if direct_stage else None
                ),
                "direct_targets": (
                    "direct_targets" if direct_stage else None
                ),
            },
        }
        self._atomic_write_json(metrics, stage_dir / "metrics.json")
        marker = {
            "stage_id": stage_id,
            "stage_label": stage_label,
            "completed_steps": int(completed_steps),
            "config_sha256": self.config_digest,
            "implementation_sha256": self.implementation_digest,
            "metrics_sha256": file_digest(stage_dir / "metrics.json"),
        }
        self._atomic_write_json(marker, stage_dir / "_SUCCESS.json")
        self._write_stage_index()
        self.logger.info("Stage artifact complete: %s", stage_dir)
        torch.cuda.empty_cache()
        return stage_dir

    def _ensure_initial_stage_outputs(self) -> None:
        if not bool(self.config["stage_outputs"].get("enabled", True)):
            return
        self.export_stage(
            "00_stage1_input",
            "Stage-1 input",
            0,
            geometry_variant="raw",
            appearance_variant="initial",
            include_atlas=False,
            source_model=True,
        )
        self.export_stage(
            "01_geometry_stabilized",
            "geometry stabilized",
            0,
            geometry_variant="stabilized",
            appearance_variant="initial",
            include_atlas=False,
            source_model=False,
        )
        before_path = (
            self.stages_dir
            / "00_stage1_input"
            / "diagnostics"
            / "render_grid.jpg"
        )
        after_path = (
            self.stages_dir
            / "01_geometry_stabilized"
            / "diagnostics"
            / "render_grid.jpg"
        )
        if before_path.is_file() and after_path.is_file():
            with Image.open(before_path) as before_file:
                before = np.asarray(
                    before_file.convert("RGB"), dtype=np.uint8
                )
            with Image.open(after_path) as after_file:
                after = np.asarray(
                    after_file.convert("RGB"), dtype=np.uint8
                )
            if before.shape == after.shape:
                Image.fromarray(
                    np.concatenate((before, after), axis=1)
                ).save(
                    self.stages_dir
                    / "geometry_stability_comparison.jpg",
                    quality=92,
                )
        self._write_stage_index()

    def _recover_coherent_base_output(self, coarse_steps: int) -> None:
        if (
            not bool(self.config["stage_outputs"].get("enabled", True))
            or coarse_steps <= 0
            or self.start_step < coarse_steps
            or self._stage_is_complete(
                "02_coherent_base", coarse_steps
            )
        ):
            return
        if self.start_step == coarse_steps:
            self.export_stage(
                "02_coherent_base",
                "coherent base",
                coarse_steps,
            )
            return

        checkpoint_path = (
            self.checkpoint_dir / f"step_{coarse_steps:06d}.pt"
        )
        if not checkpoint_path.is_file():
            raise RuntimeError(
                "Cannot reconstruct the missing coherent-base stage output "
                f"while resuming step {self.start_step}: boundary checkpoint "
                f"is missing: {checkpoint_path}"
            )
        state = self._torch_load(checkpoint_path)
        if (
            int(state.get("version", -1)) != 10
            or state.get("config_sha256") != self.config_digest
            or state.get("implementation_sha256")
            != self.implementation_digest
            or int(state.get("completed_steps", -1)) != coarse_steps
        ):
            raise ValueError(
                f"Incompatible coherent-base boundary checkpoint: "
                f"{checkpoint_path}"
            )
        saved_model = self.avatar.model_state()
        saved_atlas = self.atlas
        saved_layer_atlases = self.layer_atlases
        saved_layer_reference = self.layer_reference_rgb
        saved_direct_bank = self.direct_bank
        saved_attention_diagnostics = (
            self.pending_attention_diagnostics
        )
        try:
            self.avatar.load_model_state(state["model"])
            atlas_state = state.get("atlas")
            if atlas_state is None:
                raise ValueError(
                    "Coherent-base checkpoint does not contain a UV atlas"
                )
            self.atlas = UVAtlas.from_state_dict(
                atlas_state, self.device
            )
            self.layer_atlases = {
                name: UVAtlas.from_state_dict(value, self.device)
                for name, value in state.get(
                    "layer_atlases", {}
                ).items()
            }
            reference_state = state.get("layer_reference_rgb")
            self.layer_reference_rgb = (
                torch.as_tensor(
                    reference_state,
                    dtype=torch.float32,
                    device=self.device,
                )
                if reference_state is not None
                else None
            )
            descriptor = state.get("direct_bank")
            self.direct_bank = (
                DetailTargetBank.from_checkpoint_descriptor(
                    descriptor,
                    root=self.directory,
                    expected_config_sha256=self.config_digest,
                    expected_implementation_sha256=(
                        self.implementation_digest
                    ),
                )
                if descriptor is not None
                else None
            )
            self.pending_attention_diagnostics = state.get(
                "surface_attention_diagnostics"
            )
            self._assert_active_supervision(coarse_steps)
            self.export_stage(
                "02_coherent_base",
                "coherent base",
                coarse_steps,
            )
        finally:
            self.avatar.load_model_state(saved_model)
            self.atlas = saved_atlas
            self.layer_atlases = saved_layer_atlases
            self.layer_reference_rgb = saved_layer_reference
            self.direct_bank = saved_direct_bank
            self.pending_attention_diagnostics = (
                saved_attention_diagnostics
            )
            self.avatar.set_pose(*self.avatar.reference_pose)

    def save_stage_comparison(self) -> None:
        """Create one fixed-frame grid spanning all completed stages."""

        stages = tuple(
            (stage_id, stage_label)
            for stage_id, stage_label, _ in STAGE_DEFINITIONS
        )
        summary: dict[str, Any] = {"schema_version": 1, "stages": {}}
        for stage_id, _ in stages:
            metrics_path = self.stages_dir / stage_id / "metrics.json"
            if metrics_path.is_file():
                summary["stages"][stage_id] = json.loads(
                    metrics_path.read_text(encoding="utf-8")
                )
        self._atomic_write_json(
            summary, self.stages_dir / "metrics_summary.json"
        )

        frame_count = int(self.assets.test_frame_count)
        sample_count = min(
            int(self.config["stage_outputs"]["comparison_frames"]),
            frame_count,
        )
        selected = np.unique(
            np.linspace(0, frame_count - 1, sample_count)
            .round()
            .astype(int)
        )
        tile_size = 192
        rows: list[np.ndarray] = []
        driving_ready = True
        for stage_id, label in stages:
            frame_dir = self.stages_dir / stage_id / "driving" / "frames"
            paths = [
                frame_dir / f"{int(index):06d}.png"
                for index in selected
            ]
            if not all(path.is_file() for path in paths):
                self.logger.warning(
                    "Cannot build driving stage comparison; missing frames "
                    "under %s",
                    frame_dir,
                )
                driving_ready = False
                break
            tiles = []
            for index, path in zip(selected, paths):
                with Image.open(path) as image_file:
                    tile = np.asarray(
                        image_file.convert("RGB").resize(
                            (tile_size, tile_size),
                            Image.Resampling.LANCZOS,
                        ),
                        dtype=np.uint8,
                    )
                tiles.append(
                    add_label(tile, f"{label} | {int(index):03d}")
                )
            rows.append(np.concatenate(tiles, axis=1))
        if driving_ready:
            Image.fromarray(np.concatenate(rows, axis=0)).save(
                self.stages_dir / "driving_comparison.jpg", quality=92
            )

        diagnostic_rows = []
        for stage_id, label in stages:
            path = (
                self.stages_dir
                / stage_id
                / "diagnostics"
                / "render_grid.jpg"
            )
            if not path.is_file():
                return
            with Image.open(path) as image_file:
                image = image_file.convert("RGB")
                target_width = 1536
                target_height = int(
                    round(image.height * target_width / image.width)
                )
                resized = np.asarray(
                    image.resize(
                        (target_width, target_height),
                        Image.Resampling.LANCZOS,
                    ),
                    dtype=np.uint8,
                )
            diagnostic_rows.append(
                add_label(resized, f"stage comparison | {label}")
            )
        Image.fromarray(np.concatenate(diagnostic_rows, axis=0)).save(
            self.stages_dir / "diagnostic_comparison.jpg", quality=92
        )
        self._write_stage_index()

    def _write_run_success(self, total_steps: int) -> None:
        coarse_steps = int(
            self.config["teacher"].get("coarse_iterations", 0)
        )
        expected_steps = {
            "00_stage1_input": 0,
            "01_geometry_stabilized": 0,
            "02_coherent_base": coarse_steps,
            "03_detail_refinement": int(total_steps),
        }
        missing = [
            stage_id
            for stage_id, completed_steps in expected_steps.items()
            if not self._stage_is_complete(stage_id, completed_steps)
        ]
        if missing:
            raise RuntimeError(
                "Cannot mark the run complete; stage artifacts are missing "
                "or stale: " + ", ".join(missing)
            )

        schedule = teacher_observation_schedule(self.config)
        expected_refresh_steps = [
            int(value) for value in schedule["all_refresh_steps"]
        ]
        teacher_markers = [
            self.teacher_dir / f"step_{step:06d}" / "_SUCCESS.json"
            for step in expected_refresh_steps
        ]
        missing_teacher_markers = [
            str(path) for path in teacher_markers if not path.is_file()
        ]
        if missing_teacher_markers:
            raise RuntimeError(
                "Cannot mark the run complete; scheduled teacher refresh "
                "markers are missing:\n  "
                + "\n  ".join(missing_teacher_markers)
            )
        teacher_refreshes = []
        for expected_step, marker_path in zip(
            expected_refresh_steps, teacher_markers
        ):
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if (
                int(marker.get("refresh_step", -1)) != expected_step
                or marker.get("config_sha256") != self.config_digest
                or marker.get("implementation_sha256")
                != self.implementation_digest
            ):
                raise RuntimeError(
                    "Cannot mark the run complete; stale or incompatible "
                    f"teacher marker: {marker_path}"
                )
            teacher_refreshes.append(
                {
                    **marker,
                    "path": marker_path.parent.relative_to(
                        self.directory
                    ).as_posix(),
                    "success_sha256": file_digest(marker_path),
                }
            )

        stage_markers = {
            stage_id: {
                "path": f"{stage_id}/_SUCCESS.json",
                "sha256": file_digest(
                    self.stages_dir / stage_id / "_SUCCESS.json"
                ),
            }
            for stage_id in expected_steps
        }
        comparison_paths = (
            "geometry_stability_comparison.jpg",
            "driving_comparison.jpg",
            "diagnostic_comparison.jpg",
            "metrics_summary.json",
        )
        missing_comparisons = [
            name
            for name in comparison_paths
            if not (self.directory / name).is_file()
        ]
        if missing_comparisons:
            raise RuntimeError(
                "Cannot mark the run complete; comparison artifacts are "
                "missing: " + ", ".join(missing_comparisons)
            )
        self._atomic_write_json(
            {
                "schema_version": 3,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "completed_steps": int(total_steps),
                "config_sha256": self.config_digest,
                "implementation_sha256": self.implementation_digest,
                "stage_layout": "run_root/00_stage1_input ... run_root/03_detail_refinement",
                "stages": stage_markers,
                "teacher_refresh_count": len(teacher_refreshes),
                "teacher_schedule": schedule,
                "teacher_refreshes": teacher_refreshes,
                "comparisons": {
                    name: file_digest(self.directory / name)
                    for name in comparison_paths
                },
            },
            self.directory / "_RUN_SUCCESS.json",
        )
        self._write_stage_index()

    def _assert_active_supervision(self, completed_steps: int) -> None:
        expected_bank = expected_active_refresh_step(
            self.config, completed_steps
        )
        actual_bank = (
            int(self.direct_bank.refresh_step)
            if self.direct_bank is not None
            else None
        )
        if actual_bank != expected_bank:
            raise RuntimeError(
                "Checkpoint active direct bank does not match the teacher "
                f"schedule at completed step {completed_steps}: "
                f"{actual_bank} != {expected_bank}"
            )

        expected_atlas = expected_base_atlas_refresh_step(
            self.config, completed_steps
        )
        actual_atlas = (
            int(self.atlas.refresh_step)
            if self.atlas is not None
            else None
        )
        if actual_atlas != expected_atlas:
            raise RuntimeError(
                "Checkpoint canonical Base atlas does not match its frozen "
                f"schedule at completed step {completed_steps}: "
                f"{actual_atlas} != {expected_atlas}"
            )
        expected_layers = set(self.avatar.surface_layer_names[1:])
        actual_layers = set(self.layer_atlases)
        if expected_atlas is None:
            if actual_layers or self.layer_reference_rgb is not None:
                raise RuntimeError(
                    "Semantic atlases/reference exist before the first Base "
                    "refresh"
                )
            return
        if actual_layers != expected_layers:
            raise RuntimeError(
                "Checkpoint semantic atlases are incomplete; expected "
                f"{sorted(expected_layers)}, got {sorted(actual_layers)}"
            )
        stale_layers = {
            name: int(atlas.refresh_step)
            for name, atlas in self.layer_atlases.items()
            if int(atlas.refresh_step) != expected_atlas
        }
        if stale_layers:
            raise RuntimeError(
                "Checkpoint semantic atlases are not synchronized with the "
                f"Base atlas refresh {expected_atlas}: {stale_layers}"
            )
        if (
            self.layer_reference_rgb is None
            or self.layer_reference_rgb.shape
            != (
                int(self.avatar.gaussian.num_gs),
                3,
            )
        ):
            raise RuntimeError(
                "Checkpoint semantic residual atlases are missing their "
                "per-Gaussian RGB reference"
            )

    def checkpoint_state(self, completed_steps: int) -> dict[str, Any]:
        self._assert_active_supervision(completed_steps)
        state: dict[str, Any] = {
            "version": 10,
            "completed_steps": int(completed_steps),
            "config_sha256": self.config_digest,
            "implementation_sha256": self.implementation_digest,
            "cameras_sha256": self.camera_digest,
            "pose_data_sha256": self.pose_data_digest,
            "edit_mask_sha256": self.edit_mask_digest,
            "model": self.avatar.model_state(),
            "optimizer": self.optimizer.state_dict(),
            "atlas": self.atlas.state_dict() if self.atlas is not None else None,
            "layer_atlases": {
                name: atlas.state_dict()
                for name, atlas in self.layer_atlases.items()
            },
            "layer_reference_rgb": (
                self.layer_reference_rgb.detach().cpu()
                if self.layer_reference_rgb is not None
                else None
            ),
            "direct_bank": (
                self.direct_bank.checkpoint_descriptor(
                    relative_to=self.directory
                )
                if self.direct_bank is not None
                else None
            ),
            "surface_noise": (
                self.teacher.noise.state_dict()
                if self.teacher is not None
                else self.pending_noise_state
            ),
            "surface_attention_diagnostics": (
                self._surface_attention_stage_metrics()["runtime"]
            ),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
            "fixed": self._fixed_source_state(),
        }
        return state

    @staticmethod
    def _atomic_torch_save(state: Mapping[str, Any], path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(dict(state), temporary)
        os.replace(temporary, path)

    def save_checkpoint(self, completed_steps: int) -> Path:
        state = self.checkpoint_state(completed_steps)
        path = self.checkpoint_dir / f"step_{completed_steps:06d}.pt"
        self._atomic_torch_save(state, path)
        self._atomic_torch_save(state, self.checkpoint_dir / "latest.pt")
        self.logger.info("Saved checkpoint: %s", path)
        return path

    @staticmethod
    def _torch_load(path: Path) -> Mapping[str, Any]:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def load_checkpoint(self, path: Path) -> None:
        state = self._torch_load(path)
        if int(state.get("version", -1)) != 10:
            raise ValueError(
                "Unsupported loop-inpaint checkpoint version; expected "
                "version 10"
            )
        if state.get("config_sha256") != self.config_digest:
            raise ValueError(
                "Checkpoint configuration differs from the current configuration"
            )
        if (
            state.get("implementation_sha256")
            != self.implementation_digest
        ):
            raise ValueError(
                "Checkpoint implementation differs from the current "
                "surface-coherent code"
            )
        if state.get("cameras_sha256") != self.camera_digest:
            raise ValueError(
                "Checkpoint calibrated cameras differ from the current data"
            )
        if state.get("pose_data_sha256") != self.pose_data_digest:
            raise ValueError(
                "Checkpoint chemistry/validation poses differ from current data"
            )
        if state.get("edit_mask_sha256") != self.edit_mask_digest:
            raise ValueError(
                "Checkpoint canonical edit mask differs from the current mask"
            )
        expected_fixed = self._fixed_source_state()
        checkpoint_fixed = state.get("fixed")
        if not isinstance(checkpoint_fixed, Mapping):
            raise ValueError("Checkpoint is missing its fixed Stage-1 state")
        for name, expected in expected_fixed.items():
            if name not in checkpoint_fixed:
                raise ValueError(
                    f"Checkpoint is missing fixed Stage-1 tensor {name!r}"
                )
            actual = torch.as_tensor(checkpoint_fixed[name])
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                raise ValueError(
                    "Checkpoint belongs to a different Stage-1 reconstruction: "
                    f"fixed tensor {name!r} does not match"
                )
        self.avatar.load_model_state(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)
        atlas_state = state.get("atlas")
        if atlas_state is not None:
            self.atlas = UVAtlas.from_state_dict(atlas_state, self.device)
            expected_resolution = int(self.config["fusion"]["resolution"])
            if self.atlas.rgb.shape[-2:] != (
                expected_resolution,
                expected_resolution,
            ):
                raise ValueError(
                    "Checkpoint UV-atlas resolution differs from fusion.resolution"
                )
        self.layer_atlases = {
            name: UVAtlas.from_state_dict(value, self.device)
            for name, value in state.get("layer_atlases", {}).items()
        }
        reference_state = state.get("layer_reference_rgb")
        self.layer_reference_rgb = (
            torch.as_tensor(
                reference_state,
                dtype=torch.float32,
                device=self.device,
            )
            if reference_state is not None
            else None
        )
        unexpected_layers = set(self.layer_atlases) - set(
            self.avatar.surface_layer_names[1:]
        )
        if unexpected_layers:
            raise ValueError(
                "Checkpoint contains unknown semantic atlases: "
                + ", ".join(sorted(unexpected_layers))
            )
        expected_resolution = int(self.config["fusion"]["resolution"])
        for name, atlas in self.layer_atlases.items():
            if atlas.rgb.shape[-2:] != (
                expected_resolution,
                expected_resolution,
            ):
                raise ValueError(
                    f"Checkpoint {name} atlas resolution differs from "
                    "fusion.resolution"
                )
        direct_descriptor = state.get("direct_bank")
        self.direct_bank = (
            DetailTargetBank.from_checkpoint_descriptor(
                direct_descriptor,
                root=self.directory,
                expected_config_sha256=self.config_digest,
                expected_implementation_sha256=(
                    self.implementation_digest
                ),
            )
            if direct_descriptor is not None
            else None
        )
        self._assert_active_supervision(
            int(state["completed_steps"])
        )
        self.pending_noise_state = state.get("surface_noise")
        self.pending_attention_diagnostics = state.get(
            "surface_attention_diagnostics"
        )
        rng = state.get("rng", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "cuda" in rng:
            torch.cuda.set_rng_state_all(rng["cuda"])
        self.start_step = int(state["completed_steps"])
        self.avatar.assert_fixed_geometry()
        self.logger.info(
            "Resumed %s at completed step %d", path, self.start_step
        )

    def train(self) -> None:
        optimization = self.config["optimization"]
        total_steps = int(optimization["iterations"])
        refresh_interval = int(self.config["teacher"]["refresh_interval"])
        coarse_refresh_interval = int(
            self.config["teacher"].get(
                "coarse_refresh_interval", refresh_interval
            )
        )
        coarse_steps = int(
            self.config["teacher"].get("coarse_iterations", 0)
        )
        log_interval = max(int(self.config["output"]["log_interval"]), 1)
        preview_interval = max(
            int(self.config["output"]["preview_interval"]), 1
        )
        checkpoint_interval = max(
            int(self.config["checkpoint"]["interval"]), 1
        )
        schedule = teacher_observation_schedule(self.config)
        self.logger.info(
            "Teacher schedule: %d base refreshes / %d observations; "
            "%d detail refreshes / %d observations",
            int(schedule["base_refresh_count"]),
            int(schedule["base_observation_budget"]),
            int(schedule["detail_refresh_count"]),
            int(schedule["detail_observation_budget"]),
        )
        self._write_stage_index()
        self._ensure_initial_stage_outputs()
        self._recover_coherent_base_output(coarse_steps)
        progress = tqdm(
            range(self.start_step, total_steps),
            desc="Loop inpaint",
            dynamic_ncols=True,
        )

        for step in progress:
            in_detail = step >= coarse_steps
            if not in_detail:
                needs_refresh = self.atlas is None or self.direct_bank is None
            else:
                needs_refresh = self.direct_bank is None
            if (
                not in_detail
                and self.atlas is not None
                and self.direct_bank is not None
            ):
                needs_refresh = (
                    step - int(self.atlas.refresh_step)
                    >= coarse_refresh_interval
                )
            elif in_detail and self.direct_bank is not None:
                needs_refresh = (
                    int(self.direct_bank.refresh_step) < coarse_steps
                    or step
                    - int(self.direct_bank.refresh_step)
                    >= refresh_interval
                )
            if needs_refresh:
                self.refresh_atlas(step)

            use_direct = in_detail or (
                random.random()
                < float(
                    self.config["detail_supervision"][
                        "base_direct_probability"
                    ]
                )
            )
            if use_direct:
                batch, direct_target, direct_edit_mask = (
                    self._next_direct_batch()
                )
            else:
                batch = self._next_batch()
                direct_target = None
                direct_edit_mask = None
            self.optimizer.zero_grad(set_to_none=True)
            rendering = self.avatar.render_batch(
                batch,
                include_identity=True,
                include_surface_uv=not use_direct,
                include_surface_layers=not use_direct,
            )
            total, losses, diagnostics = self._losses(
                rendering,
                direct_target=direct_target,
                direct_edit_mask=direct_edit_mask,
                include_layered_oral=not in_detail,
            )
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"Non-finite loss at step {step}: {float(total.detach())}"
                )
            total.backward()
            max_grad_norm = float(optimization["max_grad_norm"])
            if max_grad_norm > 0.0:
                parameters = [
                    parameter
                    for group in self.optimizer.param_groups
                    for parameter in group["params"]
                ]
                torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            self.optimizer.step()
            self.avatar.clamp_opacity_increase(
                float(optimization.get("max_opacity_increase", 1.0)),
                oral_maximum_increase=float(
                    optimization.get(
                        "oral_max_opacity_increase",
                        optimization.get("max_opacity_increase", 1.0),
                    )
                ),
            )
            self.avatar.assert_fixed_geometry()
            completed = step + 1

            if (
                completed % log_interval == 0
                or step == self.start_step
                or completed == coarse_steps
                or completed == total_steps
            ):
                self._log(completed, losses, diagnostics)
            progress.set_postfix(
                loss=f"{float(total.detach()):.5f}",
                edit=f"{float(losses['edit'].detach()):.4f}",
                identity=f"{float(losses['identity'].detach()):.4f}",
                mask=f"{float(diagnostics['edit_mask'].mean()):.3f}",
            )
            if completed % preview_interval == 0:
                self.save_preview(completed)
            checkpoint_saved = False
            if completed % checkpoint_interval == 0:
                self.save_checkpoint(completed)
                checkpoint_saved = True
            if coarse_steps > 0 and completed == coarse_steps:
                # Always retain the exact algorithm boundary, even when the
                # generic checkpoint interval is changed by an override.
                if not checkpoint_saved:
                    self.save_checkpoint(completed)
                self.export_stage(
                    "02_coherent_base",
                    "coherent base",
                    coarse_steps,
                )

        self.save_preview(total_steps)
        if total_steps % checkpoint_interval != 0:
            self.save_checkpoint(total_steps)
        self.avatar.save(self.directory / "model")
        if bool(self.config["stage_outputs"].get("enabled", True)):
            final_stage = self.export_stage(
                "03_detail_refinement",
                "detail refinement",
                total_steps,
            )
            driving = final_stage / "driving"
            if driving.is_dir():
                shutil.copytree(
                    driving,
                    self.directory / "test_render",
                    dirs_exist_ok=True,
                )
            self.save_stage_comparison()
            self._write_run_success(total_steps)
        elif bool(self.config["test_render"].get("enabled", True)):
            self.render_test_sequence()
        self.logger.info("Stage-2 refinement complete: %s", self.directory)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/loop_inpaint.yaml"),
        help="Standalone Stage-2 YAML configuration.",
    )
    parser.add_argument(
        "--mode",
        choices=("train",),
        default="train",
        help="Compatibility mode; the surface pipeline currently trains.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume an exact training checkpoint.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration and Stage-1 input files without loading CUDA.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional dotted overrides, e.g. optimization.iterations=10",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path, args.overrides)
    validate_config(config, check_files=True)
    if args.validate_only:
        print(f"Configuration is valid: {config_path}")
        print(
            "Stage-1 input:",
            resolve_path(config["input"]["reconstruction_dir"]),
        )
        print("Output:", output_directory(config))
        schedule = teacher_observation_schedule(config)
        print(
            "Teacher schedule:",
            f"{schedule['base_refresh_count']} base refreshes / "
            f"{schedule['base_observation_budget']} observations; "
            f"{schedule['detail_refresh_count']} detail refreshes / "
            f"{schedule['detail_observation_budget']} observations",
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Gaussian rendering and SDEdit")
    seed_everything(int(config.get("seed", 0)))
    directory = output_directory(config)
    resume_path = resolve_path(args.resume) if args.resume is not None else None
    if (
        resume_path is None
        and directory.is_dir()
        and any(directory.iterdir())
    ):
        raise FileExistsError(
            "Refusing to mix a fresh run with an existing non-empty output "
            f"directory: {directory}. Choose a new output.name, or pass "
            "--resume with an exact compatible checkpoint."
        )
    directory.mkdir(parents=True, exist_ok=True)

    trainer = LoopInpaintTrainer(config, directory, resume_path)
    try:
        # For resume, write only after the checkpoint has validated its source
        # avatar, cameras, and configuration. A failed resume must not overwrite
        # the existing run's provenance record.
        resolved = copy.deepcopy(config)
        resolved["input"]["reconstruction_dir"] = str(
            resolve_path(config["input"]["reconstruction_dir"])
        )
        resolved["data"]["chemistry_path"] = str(
            resolve_path(config["data"]["chemistry_path"])
        )
        resolved["resolved_output_dir"] = str(directory)
        resolved["implementation_sha256"] = trainer.implementation_digest
        resolved["teacher_observation_schedule"] = (
            teacher_observation_schedule(config)
        )
        (directory / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
