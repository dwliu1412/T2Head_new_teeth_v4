"""Archived direct-target repair pipeline kept for reproducibility.

The diffusion models are used once to build a small, jaw-stratified cache of
high-quality targets.  SDXL inpainting is used only where Stage 1 has no valid
oral anatomy.  Its result is then passed through a low-strength, full-foreground
RV5.1/ControlNet enhancement for the face, ears, hair, neck, and clothing.
An exact pose/view mouth target that was previously accepted can replace the
stochastic mouth proposal through a verified soft mask.  Other observations
prefer the RV mouth and use only a small inward SDXL fallback when conservative
anatomy gates fail.
One shared canonical UVD avatar is then fitted to those targets across
calibrated views and expressions.

This deliberately avoids the previous online UV-atlas loop.  UV remains useful
as the persistent identity of every Gaussian and for canonical smoothness, but
teeth, lips, and the oral cavity are not collapsed into one alpha-composited
texture atlas.
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
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_VERSION = 11
CHECKPOINT_VERSION = 10


# -----------------------------------------------------------------------------
# Configuration and small utilities
# -----------------------------------------------------------------------------


def _parse_override_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _set_by_dotted_key(config: Dict[str, Any], key: str, value: Any) -> None:
    cursor = config
    parts = key.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def load_config(path: Path, overrides: Sequence[str]) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("%s: top-level YAML value must be a mapping" % path)
    for item in overrides:
        if "=" not in item:
            raise ValueError("Invalid override %r; expected key=value" % item)
        key, raw = item.split("=", 1)
        _set_by_dotted_key(config, key, _parse_override_value(raw))
    return config


def resolve_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def output_directory(config: Mapping[str, Any]) -> Path:
    root = resolve_path(config["output"]["root"])
    name = str(config["output"].get("name", "")).strip()
    if not name:
        name = Path(str(config["input"]["reconstruction_dir"])).name
    return root / name


def target_directory(config: Mapping[str, Any], directory: Path) -> Path:
    value = config["output"].get("target_cache")
    return resolve_path(value) if value else directory / "targets"


def stable_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tensor_digest(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accepted_mouth_prior_paths(
    config: Mapping[str, Any],
) -> Dict[str, Path]:
    """Resolve manually accepted mouth teachers by stable observation key."""

    raw = config["teacher"].get("accepted_mouth_priors", {})
    if not isinstance(raw, Mapping):
        raise ValueError("teacher.accepted_mouth_priors must be a mapping")
    resolved: Dict[str, Path] = {}
    for raw_key, raw_path in raw.items():
        key = str(raw_key).strip()
        pose_label, separator, frame_text = key.rpartition("_frame_")
        if (
            not key
            or not separator
            or not pose_label
            or len(frame_text) != 3
            or not frame_text.isdigit()
        ):
            raise ValueError(
                "teacher.accepted_mouth_priors key %r must use "
                "'<pose_label>_frame_<three digits>'" % raw_key
            )
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(
                "teacher.accepted_mouth_priors.%s must be a non-empty path"
                % key
            )
        resolved[key] = resolve_path(raw_path)
    return resolved


def accepted_mouth_prior_source(path: Path) -> Path:
    """Return the Stage-1 source saved beside an accepted target."""

    suffix = "_target.png"
    if not path.name.endswith(suffix):
        raise ValueError(
            "Accepted mouth prior must end in %s so its source can be "
            "verified: %s" % (suffix, path)
        )
    return path.with_name(
        path.name[: -len(suffix)] + "_source.png"
    )


def target_signature(config: Mapping[str, Any]) -> str:
    reconstruction = resolve_path(config["input"]["reconstruction_dir"])
    dependency_digests: Dict[str, str] = {}
    for key in (
        "chemistry_path",
        "validation_expression_path",
        "validation_pose_path",
        "test_expression_path",
        "test_pose_path",
    ):
        value = config["data"].get(key)
        if value:
            path = resolve_path(value)
            if path.is_file():
                dependency_digests[key] = file_digest(path)
    accepted_prior_digests: Dict[str, Dict[str, str]] = {}
    for key, path in accepted_mouth_prior_paths(config).items():
        source_path = accepted_mouth_prior_source(path)
        accepted_prior_digests[key] = {
            "target": file_digest(path),
            "source": file_digest(source_path),
        }
    return stable_digest(
        {
            "version": MANIFEST_VERSION,
            "uvd_sha256": file_digest(reconstruction / "model" / "uvd.ply"),
            "sidecar_sha256": file_digest(
                reconstruction / "model" / "reconstruction_params.npz"
            ),
            "stage1_config_sha256": file_digest(
                reconstruction / "resolved_config.yaml"
            ),
            "data": config["data"],
            "data_file_sha256": dependency_digests,
            "accepted_mouth_prior_sha256": accepted_prior_digests,
            "bootstrap": config["bootstrap"],
            # QA belongs to the signature as well: a cache which has not passed
            # the currently configured acceptance gates must never be reused.
            "teacher": config["teacher"],
        }
    )


def validate_sh0_ply(path: Path) -> None:
    properties: List[str] = []
    found_end_header = False
    with path.open("rb") as file:
        for _ in range(10000):
            raw = file.readline()
            if not raw:
                break
            line = raw.decode("ascii", errors="strict").strip()
            if line == "end_header":
                found_end_header = True
                break
            if line.startswith("property "):
                properties.append(line.split()[-1])
    if not found_end_header:
        raise ValueError("%s: malformed PLY header" % path)
    rest = [name for name in properties if name.startswith("f_rest_")]
    if rest:
        raise ValueError(
            "%s: direct inpaint requires SH degree 0, found %d f_rest fields"
            % (path, len(rest))
        )


def validate_config(config: Mapping[str, Any], check_files: bool = True) -> None:
    sections = (
        "input",
        "output",
        "data",
        "bootstrap",
        "teacher",
        "test_render",
        "optimization",
        "loss",
        "checkpoint",
    )
    for section in sections:
        if not isinstance(config.get(section), Mapping):
            raise ValueError("Configuration section %r must be a mapping" % section)

    if str(config.get("device", "cuda")) not in {"cuda", "cuda:0"}:
        raise ValueError("This pipeline currently requires device cuda")

    reconstruction = resolve_path(config["input"]["reconstruction_dir"])
    accepted_priors = accepted_mouth_prior_paths(config)
    required = (
        reconstruction / "model" / "uvd.ply",
        reconstruction / "model" / "reconstruction_params.npz",
        reconstruction / "resolved_config.yaml",
    )
    if check_files:
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Incomplete Stage-1 reconstruction:\n  " + "\n  ".join(missing)
            )
        validate_sh0_ply(required[0])
        chemistry = resolve_path(config["data"]["chemistry_path"])
        if not chemistry.is_file():
            raise FileNotFoundError("Missing chemistry pose file: %s" % chemistry)
        model_path = resolve_path(config["teacher"]["model_path"])
        if not model_path.exists():
            raise FileNotFoundError("Missing inpainting model: %s" % model_path)
        rv51 = config["teacher"].get("rv51", {})
        if bool(rv51.get("enabled", False)):
            for key in ("model_path", "controlnet_path", "scheduler_path"):
                path = resolve_path(rv51[key])
                if not path.exists():
                    raise FileNotFoundError(
                        "Missing teacher.rv51.%s: %s" % (key, path)
                    )
        cache_path = target_directory(config, output_directory(config)).resolve()
        for key, path in accepted_priors.items():
            source_path = accepted_mouth_prior_source(path)
            for label, image_path in (
                ("target", path),
                ("source", source_path),
            ):
                if not image_path.is_file():
                    raise FileNotFoundError(
                        "Missing accepted mouth prior %s for %s: %s"
                        % (label, key, image_path)
                    )
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except Exception as error:
                    raise ValueError(
                        "Invalid accepted mouth prior %s for %s: %s"
                        % (label, key, image_path)
                    ) from error
            try:
                path.relative_to(cache_path)
            except ValueError:
                pass
            else:
                raise ValueError(
                    "Accepted mouth prior %s is inside the target cache "
                    "that may be overwritten: %s" % (key, path)
                )
        if bool(config["test_render"].get("enabled", True)):
            for key in ("test_expression_path", "test_pose_path"):
                path = resolve_path(config["data"][key])
                if not path.is_file():
                    raise FileNotFoundError(
                        "Missing data.%s for final test rendering: %s"
                        % (key, path)
                    )

    for key in ("height", "width"):
        value = int(config["data"][key])
        if value <= 0 or value % 8:
            raise ValueError("data.%s must be a positive multiple of 8" % key)
    image_size = int(config["teacher"]["image_size"])
    if image_size <= 0 or image_size % 8:
        raise ValueError("teacher.image_size must be a positive multiple of 8")
    for key, data_key in (
        ("render_height", "height"),
        ("render_width", "width"),
        ("target_height", "height"),
        ("target_width", "width"),
    ):
        value = int(config["teacher"].get(key, config["data"][data_key]))
        if value <= 0 or value % 8:
            raise ValueError("teacher.%s must be a positive multiple of 8" % key)
    if int(config["teacher"]["num_inference_steps"]) <= 0:
        raise ValueError("teacher.num_inference_steps must be positive")
    if int(config["teacher"].get("mouth_candidates_per_view", 1)) <= 0:
        raise ValueError("teacher.mouth_candidates_per_view must be positive")
    if int(config["teacher"].get("mask_reference_resolution", 1024)) <= 0:
        raise ValueError("teacher.mask_reference_resolution must be positive")
    strength = float(config["teacher"]["strength"])
    if not 0.0 < strength <= 1.0:
        raise ValueError("teacher.strength must be in (0, 1]")
    quantiles = [float(value) for value in config["teacher"]["pose_quantiles"]]
    if any(value < 0.0 or value > 1.0 for value in quantiles):
        raise ValueError("teacher.pose_quantiles values must be in [0, 1]")
    if not config["teacher"]["source_azimuths"]:
        raise ValueError("teacher.source_azimuths must not be empty")
    seed_mode = str(config["teacher"].get("seed_mode", "shared")).lower()
    if seed_mode not in {"shared", "pose", "observation"}:
        raise ValueError(
            "teacher.seed_mode must be shared, pose, or observation"
        )
    seed_overrides = config["teacher"].get("seed_overrides", {})
    if not isinstance(seed_overrides, Mapping):
        raise ValueError("teacher.seed_overrides must be a mapping")
    for key, value in seed_overrides.items():
        if not str(key).strip():
            raise ValueError("teacher.seed_overrides keys must not be empty")
        if int(value) < 0:
            raise ValueError(
                "teacher.seed_overrides values must be non-negative"
            )
    prior_alignment_limit = float(
        config["teacher"].get(
            "accepted_mouth_prior_max_source_mae", 0.01
        )
    )
    if not 0.0 <= prior_alignment_limit <= 1.0:
        raise ValueError(
            "teacher.accepted_mouth_prior_max_source_mae must be in [0, 1]"
        )
    rv51 = config["teacher"].get("rv51", {})
    if bool(rv51.get("enabled", False)):
        if not bool(config["data"].get("use_mediapipe_condition", False)):
            raise ValueError(
                "teacher.rv51 requires data.use_mediapipe_condition=true"
            )
        rv_strength = float(rv51.get("strength", 0.25))
        if not 0.0 < rv_strength <= 1.0:
            raise ValueError("teacher.rv51.strength must be in (0, 1]")
        if int(rv51.get("num_inference_steps", 50)) <= 0:
            raise ValueError(
                "teacher.rv51.num_inference_steps must be positive"
            )
        rv_image_size = int(
            rv51.get("image_size", config["teacher"]["image_size"])
        )
        if rv_image_size <= 0 or rv_image_size % 8:
            raise ValueError(
                "teacher.rv51.image_size must be a positive multiple of 8"
            )
        scope = str(rv51.get("composite_scope", "foreground")).lower()
        if scope not in {"foreground", "full"}:
            raise ValueError(
                "teacher.rv51.composite_scope must be foreground or full"
            )
        alpha_threshold = float(rv51.get("foreground_alpha_threshold", 0.01))
        if not 0.0 <= alpha_threshold <= 1.0:
            raise ValueError(
                "teacher.rv51.foreground_alpha_threshold must be in [0, 1]"
            )
        for key in (
            "foreground_dilation",
            "foreground_feather",
            "mouth_fallback_feather",
        ):
            if int(rv51.get(key, 0)) < 0:
                raise ValueError("teacher.rv51.%s must be non-negative" % key)
        fallback_alphas = [
            float(value)
            for value in rv51.get(
                "mouth_fallback_alphas", [0.0, 0.15, 0.30]
            )
        ]
        if not fallback_alphas:
            raise ValueError(
                "teacher.rv51.mouth_fallback_alphas must not be empty"
            )
        if fallback_alphas[0] != 0.0:
            raise ValueError(
                "teacher.rv51.mouth_fallback_alphas must start with 0.0"
            )
        if any(value < 0.0 or value > 1.0 for value in fallback_alphas):
            raise ValueError(
                "teacher.rv51.mouth_fallback_alphas values must be in [0, 1]"
            )
        if any(
            right <= left
            for left, right in zip(
                fallback_alphas[:-1], fallback_alphas[1:]
            )
        ):
            raise ValueError(
                "teacher.rv51.mouth_fallback_alphas must be strictly increasing"
            )

    quality = config["teacher"].get("quality", {})
    if not isinstance(quality, Mapping):
        raise ValueError("teacher.quality must be a mapping")
    if not isinstance(quality.get("enforce", True), bool):
        raise ValueError("teacher.quality.enforce must be a boolean")
    for key in (
        "max_outside_edit_delta",
    ):
        if float(quality.get(key, 0.0)) < 0.0:
            raise ValueError("teacher.quality.%s must be non-negative" % key)
    for key in (
        "min_teeth_coverage",
        "teeth_hint_threshold",
        "teeth_min_luminance",
        "teeth_max_saturation",
        "teeth_max_chroma",
        "max_edit_mask_fraction",
        "min_non_mouth_delta",
        "min_ear_edit_coverage",
        "min_ear_delta",
        "min_teeth_component_coverage",
        "max_teeth_component_coverage",
        "max_teeth_component_span",
        "min_teeth_span_front",
        "min_teeth_span_three_quarter",
        "min_mouth_dark_fraction",
        "preferred_teeth_component_coverage",
        "preferred_teeth_span",
        "preferred_mouth_dark_fraction",
    ):
        value = float(quality.get(key, 0.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError("teacher.quality.%s must be in [0, 1]" % key)
    if float(quality.get("min_teeth_component_aspect", 0.0)) < 0.0:
        raise ValueError(
            "teacher.quality.min_teeth_component_aspect must be non-negative"
        )

    for key in ("geometry_regions", "appearance_regions", "opacity_regions"):
        regions = config["optimization"].get(key)
        if not isinstance(regions, (list, tuple)) or not regions:
            raise ValueError(
                "optimization.%s must be a non-empty sequence" % key
            )
        if any(not str(region).strip() for region in regions):
            raise ValueError(
                "optimization.%s contains an empty region" % key
            )
    if any(
        str(region).lower() == "all"
        for region in config["optimization"]["geometry_regions"]
    ):
        raise ValueError(
            "optimization.geometry_regions cannot contain 'all'; "
            "global geometry optimization is intentionally unsupported"
        )

    geometry_steps = int(config["optimization"]["geometry_iterations"])
    appearance_steps = int(config["optimization"]["appearance_iterations"])
    if geometry_steps < 0 or appearance_steps <= 0:
        raise ValueError(
            "geometry_iterations must be non-negative and "
            "appearance_iterations must be positive"
        )
    if int(config["optimization"]["batch_size"]) <= 0:
        raise ValueError("optimization.batch_size must be positive")

    test_render = config["test_render"]
    if int(test_render.get("fps", 30)) <= 0:
        raise ValueError("test_render.fps must be positive")
    for key in ("height", "width"):
        value = int(test_render.get(key, config["data"]["eval_" + key]))
        if value <= 0 or value % 8:
            raise ValueError(
                "test_render.%s must be a positive multiple of 8" % key
            )
    if int(test_render.get("contact_sheet_frames", 12)) <= 0:
        raise ValueError("test_render.contact_sheet_frames must be positive")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    return value


def rescale_render_batch(
    batch: Mapping[str, Any], height: int, width: int
) -> Dict[str, Any]:
    """Scale calibrated intrinsics for a different raster resolution."""
    source_height = int(torch.as_tensor(batch["height"]).reshape(-1)[0].item())
    source_width = int(torch.as_tensor(batch["width"]).reshape(-1)[0].item())
    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError("Render resolution must be positive")
    scaled = dict(batch)
    if height == source_height and width == source_width:
        return scaled

    scale_x = float(width) / float(source_width)
    scale_y = float(height) / float(source_height)
    intrinsic = torch.as_tensor(batch["K"]).clone()
    intrinsic[..., 0, :] *= scale_x
    intrinsic[..., 1, :] *= scale_y
    scaled["K"] = intrinsic
    scaled["fx"] = intrinsic[..., 0, 0].clone()
    scaled["fy"] = intrinsic[..., 1, 1].clone()
    scaled["cx"] = intrinsic[..., 0, 2].clone()
    scaled["cy"] = intrinsic[..., 1, 2].clone()
    scaled["height"] = height
    scaled["width"] = width
    return scaled


def rgb_u8(image: torch.Tensor) -> np.ndarray:
    value = image.detach().clamp(0.0, 1.0)
    if value.ndim == 4:
        value = value[0]
    return (
        value.permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .cpu()
        .numpy()
    )


def gray_u8(mask: torch.Tensor) -> np.ndarray:
    value = mask.detach().clamp(0.0, 1.0)
    while value.ndim > 2:
        value = value[0]
    return value.mul(255.0).round().byte().cpu().numpy()


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 26), (18, 18, 18), -1)
    cv2.putText(
        output,
        label,
        (7, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def setup_logger(directory: Path) -> logging.Logger:
    logger = logging.getLogger("direct-mouth-inpaint:%s" % directory)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(
        directory / "train.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 1 and value.shape[1] != 1:
        mask = mask.expand(-1, value.shape[1], -1, -1)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def masked_l1(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return masked_mean((prediction - target).abs(), mask)


def circular_distance_degrees(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def square_bbox(
    mask: np.ndarray, scale: float, minimum: int
) -> Tuple[int, int, int, int]:
    height, width = mask.shape
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0:
        side = min(max(int(minimum), 8), min(height, width))
        cx, cy = width // 2, height // 2
    else:
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        side = int(math.ceil(max(x1 - x0, y1 - y0) * float(scale)))
        side = max(side, int(minimum))
        side = min(side, min(height, width))
    x0 = min(max(cx - side // 2, 0), width - side)
    y0 = min(max(cy - side // 2, 0), height - side)
    return x0, y0, x0 + side, y0 + side


def resolution_scaled_pixels(
    value: Any,
    height: int,
    width: int,
    reference_resolution: int = 1024,
) -> int:
    """Scale a configured morphology radius consistently at 512 or 1024."""

    pixels = int(value)
    if pixels <= 0:
        return 0
    reference = max(int(reference_resolution), 1)
    scale = min(int(height), int(width)) / float(reference)
    return max(int(round(pixels * scale)), 1)


def dental_coverage(
    image: np.ndarray,
    hint: np.ndarray,
    hint_threshold: float,
    minimum_luminance: float,
    maximum_saturation: float,
) -> float:
    """Estimate low-saturation bright tooth coverage in a rendered hint ROI."""

    rgb = image.astype(np.float32)
    if rgb.size and float(rgb.max()) > 1.0:
        rgb = rgb / 255.0
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    saturation = (maximum - minimum) / np.maximum(maximum, 1.0e-6)
    luminance = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )
    region = hint > float(hint_threshold)
    if not bool(region.any()):
        return 0.0
    dental = (
        (luminance >= float(minimum_luminance))
        & (saturation <= float(maximum_saturation))
    )
    return float(dental[region].mean())


def dental_structure(
    image: np.ndarray,
    aperture: np.ndarray,
    minimum_luminance: float,
    maximum_chroma: float,
) -> Dict[str, float]:
    """Measure a tooth-like horizontal component inside the mouth aperture."""

    rgb = image.astype(np.float32)
    if rgb.size and float(rgb.max()) > 1.0:
        rgb = rgb / 255.0
    luminance = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )
    aperture_binary = aperture > 0.5
    ys, xs = np.nonzero(aperture_binary)
    if xs.size == 0:
        return {
            "coverage": 0.0,
            "span": 0.0,
            "aspect": 0.0,
            "dark_fraction": 0.0,
        }
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    mouth_width = max(x1 - x0, 1)
    mouth_height = max(y1 - y0, 1)
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
    white = (
        aperture_binary
        & (luminance >= float(minimum_luminance))
        & (chroma <= float(maximum_chroma))
    ).astype(np.uint8)
    kernel_width = max(int(round(mouth_width * 0.05)), 1)
    kernel_height = max(int(round(mouth_height * 0.03)), 1)
    closed = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        np.ones((kernel_height, kernel_width), dtype=np.uint8),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )
    best_area = 0
    best_width = 0
    best_height = 1
    for index in range(1, count):
        component_width = int(stats[index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        component_area = int(stats[index, cv2.CC_STAT_AREA])
        if component_width > best_width or (
            component_width == best_width and component_area > best_area
        ):
            best_width = component_width
            best_height = max(component_height, 1)
            best_area = component_area
    aperture_area = max(int(aperture_binary.sum()), 1)
    return {
        "coverage": float(best_area / aperture_area),
        "span": float(best_width / mouth_width),
        "aspect": float(best_width / best_height),
        "dark_fraction": float(
            (luminance[aperture_binary] < 0.28).mean()
        ),
    }


def inward_aperture_weight(
    aperture: np.ndarray,
    feather: int,
) -> np.ndarray:
    """Build an inward-only feather which leaves the lip boundary RV-owned."""

    binary = np.ascontiguousarray(
        (aperture > 0.5).astype(np.uint8)
    )
    if not bool(binary.any()):
        return np.zeros_like(aperture, dtype=np.float32)
    if int(feather) <= 0:
        return binary.astype(np.float32)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    value = np.clip(
        (distance - 1.0) / float(feather),
        0.0,
        1.0,
    ).astype(np.float32)
    return value * value * (3.0 - 2.0 * value)


def tensor_mouth_crops(
    image: torch.Tensor, mask: torch.Tensor, size: int = 224
) -> torch.Tensor:
    crops = []
    for index in range(image.shape[0]):
        binary = gray_u8(mask[index]) > 8
        x0, y0, x1, y1 = square_bbox(binary.astype(np.uint8), 1.45, 64)
        crop = image[index : index + 1, :, y0:y1, x0:x1]
        crops.append(
            F.interpolate(
                crop, (size, size), mode="bilinear", align_corners=False
            )
        )
    return torch.cat(crops, dim=0)


# -----------------------------------------------------------------------------
# UVD avatar, deterministic mouth bootstrap, and differentiable rendering
# -----------------------------------------------------------------------------


@dataclass
class RenderBatch:
    rgb: torch.Tensor
    alpha: torch.Tensor
    region_alpha: Dict[str, torch.Tensor]
    world_scale: torch.Tensor


class UVDAvatar:
    def __init__(
        self,
        reconstruction_dir: Path,
        bootstrap: Mapping[str, Any],
        optimization: Mapping[str, Any],
        device: torch.device,
    ) -> None:
        from gaussiansplatting.gaussian_renderer import render
        from gaussiansplatting.scene.gaussian_flame_face import (
            GaussianFlameUVModel,
        )
        from gaussiansplatting.utils.sh_utils import RGB2SH, SH2RGB
        from train_reconstruction import (
            OpenCVCamera,
            aligned_scaling_rotation,
        )

        self._render = render
        self._RGB2SH = RGB2SH
        self._SH2RGB = SH2RGB
        self._OpenCVCamera = OpenCVCamera
        self._aligned_geometry = aligned_scaling_rotation
        self.device = device
        self.reconstruction_dir = reconstruction_dir

        sidecar_path = reconstruction_dir / "model" / "reconstruction_params.npz"
        uvd_path = reconstruction_dir / "model" / "uvd.ply"
        validate_sh0_ply(uvd_path)
        with np.load(sidecar_path, allow_pickle=False) as archive:
            self.sidecar = {key: np.asarray(archive[key]) for key in archive.files}

        self.spatial_lr_scale = float(
            np.asarray(self.sidecar.get("spatial_lr_scale", 4.0)).reshape(-1)[0]
        )
        self.flame_scale = float(
            np.asarray(self.sidecar.get("flame_scale", -10.0)).reshape(-1)[0]
        )
        self.gaussian = GaussianFlameUVModel(sh_degree=0, device=str(device))
        self.gaussian.initialize_flame_state(
            self.spatial_lr_scale, self.flame_scale
        )
        self.gaussian.load_ply(str(uvd_path))
        if int(self.gaussian.max_sh_degree) != 0:
            raise ValueError("Stage-2 requires an SH-degree-0 reconstruction")

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
        self.alignment_scale = float(
            self.alignment[:3, :3].square().sum().div(3.0).sqrt().item()
        )
        self.reference_pose = (expression, jaw, eyes[:, :3], eyes[:, 3:6])
        self.gaussian._shape.data.copy_(shape)
        self.gaussian._shape.requires_grad_(False)
        self.set_pose(*self.reference_pose)

        self.bootstrap_stats = self._bootstrap_mouth(bootstrap)
        self.region_masks = self._build_region_masks()
        self.geometry_region_names = tuple(
            str(name) for name in optimization["geometry_regions"]
        )
        self.appearance_region_names = tuple(
            str(name) for name in optimization["appearance_regions"]
        )
        self.opacity_region_names = tuple(
            str(name) for name in optimization["opacity_regions"]
        )
        self.geometry_mask = self.region_union(
            self.geometry_region_names
        )
        self.appearance_mask = self.region_union(
            self.appearance_region_names
        )
        self.opacity_mask = self.region_union(self.opacity_region_names)
        if not bool(self.geometry_mask.any()):
            raise ValueError("Configured geometry regions contain no Gaussians")
        if not bool(self.appearance_mask.any()):
            raise ValueError("Configured appearance regions contain no Gaussians")
        if not bool(self.opacity_mask.any()):
            raise ValueError("Configured opacity regions contain no Gaussians")

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
        self.pipeline = SimpleNamespace(
            compute_cov3D_python=True,
            convert_SHs_python=False,
            debug=False,
        )
        self.white = torch.ones(3, dtype=torch.float32, device=device)
        self.black = torch.zeros(3, dtype=torch.float32, device=device)
        self._configure_parameters()

    def _parameter(self, value: Any, width: int) -> torch.Tensor:
        return torch.as_tensor(
            value, dtype=torch.float32, device=self.device
        ).reshape(1, -1)[:, :width]

    def _filter_points(self, keep: torch.Tensor) -> None:
        keep = keep.to(device=self.device, dtype=torch.bool)
        names = (
            "_uv",
            "_d",
            "_features_dc",
            "_features_rest",
            "_opacity",
            "_scaling",
            "_rotation",
        )
        for name in names:
            value = getattr(self.gaussian, name)
            filtered = value.detach()[keep].clone()
            setattr(
                self.gaussian,
                name,
                torch.nn.Parameter(filtered.requires_grad_(True)),
            )
        self.gaussian._face_idx = self.gaussian._face_idx[keep].clone()
        self.gaussian.num_gs = int(keep.sum().item())
        self.gaussian._reset_densification_buffers()

    @torch.no_grad()
    def _limit_region_world_scale(
        self, mask: torch.Tensor, maximum: float
    ) -> int:
        if not bool(mask.any()) or maximum <= 0.0:
            return 0
        world = (
            self.gaussian.get_world_scale()[:, 0] * self.alignment_scale
        )
        ratio = (float(maximum) / world.clamp_min(1.0e-8)).clamp(max=1.0)
        changed = mask & (ratio < 1.0)
        self.gaussian._scaling.data[changed] = (
            self.gaussian._scaling.data[changed]
            + ratio[changed, None].log()
        )
        return int(changed.sum().item())

    @torch.no_grad()
    def _isotropize_region(
        self, mask: torch.Tensor, maximum: float
    ) -> int:
        indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if indices.numel() == 0:
            return 0
        vertices, _ = self.gaussian._flame_verts_and_normals()
        world_scale = (
            self.gaussian.get_world_scale()[indices] * self.alignment_scale
        )
        radius = world_scale.prod(dim=-1).clamp_min(1.0e-18).pow(1.0 / 3.0)
        radius = radius.clamp(
            min=float(self.gaussian.SIGMA_FLOOR),
            max=float(maximum),
        )
        _, face_scale = self.gaussian._face_properties(
            vertices, self.gaussian._face_idx[indices]
        )
        local_scale = (
            radius[:, None] / (self.alignment_scale * face_scale)
        ).expand(-1, 3).contiguous()
        local_rotation = torch.zeros(
            (indices.numel(), 4),
            dtype=self.gaussian._rotation.dtype,
            device=self.gaussian.device,
        )
        local_rotation[:, 0] = 1.0
        self.gaussian._scaling.data[indices] = local_scale.log()
        self.gaussian._rotation.data[indices] = local_rotation
        return int(indices.numel())

    @torch.no_grad()
    def _bootstrap_mouth(
        self, config: Mapping[str, Any]
    ) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "isotropized_streaks": 0,
            "removed_cavity": 0,
            "seeded_cavity": 0,
            "reset_teeth": 0,
            "clamped_scale": 0,
        }
        if not bool(config.get("enabled", True)):
            return stats

        if bool(config.get("isotropize_streaks", True)):
            world_scale = (
                self.gaussian.get_world_scale() * self.alignment_scale
            )
            aspect = world_scale[:, 0] / world_scale[:, -1].clamp_min(1.0e-8)
            streaks = (
                (
                    world_scale[:, 0]
                    > float(config["absolute_scale_threshold"])
                )
                | (
                    (
                        world_scale[:, 0]
                        > float(config["streak_scale_threshold"])
                    )
                    & (aspect > float(config["streak_aspect_threshold"]))
                )
            )
            stats["isotropized_streaks"] = self._isotropize_region(
                streaks, float(config["streak_max_world_scale"])
            )

        cavity = self.gaussian.point_region_mask("oral_cavity")
        if bool(config.get("remove_existing_cavity", True)) and bool(cavity.any()):
            stats["removed_cavity"] = int(cavity.sum().item())
            self._filter_points(~cavity)

        cavity_points = int(config.get("cavity_points", 0))
        if cavity_points > 0:
            rng_devices: List[int] = []
            if self.device.type == "cuda":
                rng_devices = [
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                ]
            with torch.random.fork_rng(devices=rng_devices):
                seed = int(config.get("seed", 0))
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                seeded = self.gaussian.seed_flame_region(
                    region="oral_cavity",
                    num_points=cavity_points,
                    rgb=tuple(float(v) for v in config["cavity_rgb"]),
                    opacity=float(config["cavity_opacity"]),
                    min_world_scale=float(config["cavity_scale_min"]),
                    max_world_scale=float(config["cavity_scale_max"]),
                )
            stats["seeded_cavity"] = int(seeded["added"])

        upper = self.gaussian.point_region_mask("teeth_upper")
        lower = self.gaussian.point_region_mask("teeth_lower")
        teeth = upper | lower
        if bool(config.get("reset_teeth", True)) and bool(teeth.any()):
            self.gaussian._d.data[teeth] = 0.0
            color = torch.as_tensor(
                config["teeth_rgb"], dtype=torch.float32, device=self.device
            ).reshape(1, 3)
            feature = self._RGB2SH(color).reshape(1, 1, 3)
            self.gaussian._features_dc.data[teeth] = feature
            opacity = float(config["teeth_opacity"])
            opacity = min(max(opacity, 1.0e-5), 1.0 - 1.0e-5)
            logit = math.log(opacity / (1.0 - opacity))
            self.gaussian._opacity.data[teeth] = logit
            stats["reset_teeth"] = int(teeth.sum().item())

        mouth = teeth | self.gaussian.point_region_mask("oral_cavity")
        stats["clamped_scale"] = self._limit_region_world_scale(
            mouth, float(config["max_initial_world_scale"])
        )
        return stats

    def _build_region_masks(self) -> Dict[str, torch.Tensor]:
        upper = self.gaussian.point_region_mask("teeth_upper")
        lower = self.gaussian.point_region_mask("teeth_lower")
        cavity = self.gaussian.point_region_mask("oral_cavity")
        lips = self.gaussian.point_region_mask("lips")
        ears = self.gaussian.point_region_mask("ears")
        return {
            "teeth_upper": upper,
            "teeth_lower": lower,
            "teeth": upper | lower,
            "cavity": cavity,
            "lips": lips & ~(upper | lower | cavity),
            "ears": ears,
        }

    def region_mask(self, name: str) -> torch.Tensor:
        key = str(name).strip()
        lower = key.lower()
        if lower == "all":
            return torch.ones(
                self.gaussian.num_gs,
                dtype=torch.bool,
                device=self.device,
            )
        aliases = {
            "oral_cavity": "cavity",
            "teeth_upper": "teeth_upper",
            "teeth_lower": "teeth_lower",
            "teeth": "teeth",
            "lips": "lips",
            "ears": "ears",
        }
        cached = aliases.get(lower)
        if cached is not None:
            return self.region_masks[cached]
        try:
            return self.gaussian.point_region_mask(key)
        except Exception as error:
            raise ValueError("Unknown FLAME repair region %r" % key) from error

    def region_union(self, names: Sequence[str]) -> torch.Tensor:
        output = torch.zeros(
            self.gaussian.num_gs,
            dtype=torch.bool,
            device=self.device,
        )
        for name in names:
            output |= self.region_mask(str(name))
        return output

    def _configure_parameters(self) -> None:
        for parameter in (
            self.gaussian._uv,
            self.gaussian._features_rest,
            self.gaussian._shape,
        ):
            parameter.requires_grad_(False)
        for parameter in (
            self.gaussian._d,
            self.gaussian._features_dc,
            self.gaussian._opacity,
            self.gaussian._scaling,
            self.gaussian._rotation,
        ):
            parameter.requires_grad_(True)

    def set_pose(
        self,
        expression: torch.Tensor,
        jaw: torch.Tensor,
        leye: torch.Tensor,
        reye: torch.Tensor,
    ) -> None:
        zeros = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        self.gaussian._expression = expression.reshape(1, -1)[:, :100].float()
        self.gaussian._jaw_pose = jaw.reshape(1, -1)[:, :3].float()
        self.gaussian._leye_pose = leye.reshape(1, -1)[:, :3].float()
        self.gaussian._reye_pose = reye.reshape(1, -1)[:, :3].float()
        self.gaussian._global_orient = zeros
        self.gaussian._neck_pose = zeros
        self.gaussian._translation = zeros

    def set_batch_pose(self, batch: Mapping[str, Any]) -> None:
        self.set_pose(
            self._parameter(batch["expression"], 100),
            self._parameter(batch["jaw_pose"], 3),
            self._parameter(batch["leye_pose"], 3),
            self._parameter(batch["reye_pose"], 3),
        )

    def cameras(self, batch: Mapping[str, Any]) -> List[Any]:
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

    def _current_geometry(
        self, differentiable: bool
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        context = torch.enable_grad() if differentiable else torch.no_grad()
        with context, torch.cuda.amp.autocast(enabled=False):
            means, scales, rotations = self._aligned_geometry(
                self.gaussian, self.alignment
            )
            world_scale = scales.amax(dim=-1)
            packed = (means, scales, rotations)
        return packed, world_scale

    def render_batch(
        self,
        batch: Mapping[str, Any],
        include_regions: bool = False,
        differentiable: bool = True,
        region_names: Optional[Sequence[str]] = None,
    ) -> RenderBatch:
        self.set_batch_pose(batch)
        cameras = self.cameras(batch)
        packed, world_scale = self._current_geometry(differentiable)

        images = []
        alphas = []
        names = (
            tuple(region_names)
            if region_names is not None
            else ("teeth", "cavity")
        )
        region_alphas: Dict[str, List[torch.Tensor]] = {
            name: [] for name in names
        }
        opacity = self.gaussian.get_opacity
        region_colors = torch.ones(
            (self.gaussian.num_gs, 3),
            dtype=torch.float32,
            device=self.device,
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

            if include_regions:
                for name in names:
                    mask = self.region_mask(name)[:, None].to(opacity.dtype)
                    with torch.cuda.amp.autocast(enabled=False):
                        region = self._render(
                            camera,
                            self.gaussian,
                            self.pipeline,
                            self.black,
                            override_color=region_colors,
                            override_opacity=opacity * mask,
                            precomputed_geometry=packed,
                        )
                    region_alpha = region["alpha_3dgs"].clamp(0.0, 1.0)
                    region_alphas[name].append(region_alpha)

        return RenderBatch(
            rgb=torch.stack(images),
            alpha=torch.stack(alphas),
            region_alpha={
                name: torch.stack(values)
                for name, values in region_alphas.items()
                if values
            },
            world_scale=world_scale,
        )

    @torch.no_grad()
    def semantic_masks(
        self,
        alpha: torch.Tensor,
        threshold: float,
        dilation: int,
        feather: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Turn rendered alpha into a dilated hard mask and feathered soft mask."""

        hard_masks, soft_masks = [], []
        for item in alpha:
            alpha_np = gray_u8(item).astype(np.float32) / 255.0
            hard = (alpha_np >= float(threshold)).astype(np.uint8) * 255
            if dilation > 0:
                kernel = np.ones(
                    (2 * int(dilation) + 1, 2 * int(dilation) + 1),
                    dtype=np.uint8,
                )
                hard = cv2.dilate(hard, kernel, iterations=1)
            soft = hard.astype(np.float32) / 255.0
            if feather > 0:
                kernel_size = 2 * int(feather) + 1
                soft = cv2.GaussianBlur(
                    soft, (kernel_size, kernel_size), 0
                )
                soft = np.clip(soft, 0.0, 1.0)
            hard_masks.append(
                torch.from_numpy(hard.astype(np.float32) / 255.0)
            )
            soft_masks.append(torch.from_numpy(soft))
        return (
            torch.stack(hard_masks)[:, None].to(self.device),
            torch.stack(soft_masks)[:, None].to(self.device),
        )

    @torch.no_grad()
    def foreground_masks(
        self,
        alpha: torch.Tensor,
        threshold: float,
        dilation: int,
        feather: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build an avatar-foreground mask without an image matting network."""

        return self.semantic_masks(
            alpha,
            threshold=threshold,
            dilation=dilation,
            feather=feather,
        )

    @torch.no_grad()
    def lip_masks(
        self,
        batch: Mapping[str, Any],
        dilation: int,
        feather: int,
        aperture_dilation: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.set_batch_pose(batch)
        vertices, _ = self.gaussian._flame_verts_and_normals()
        vertices = (
            vertices @ self.alignment[:3, :3].T
            + self.alignment[:3, 3][None]
        )
        hard_ids = self.gaussian.model.mask.get_vid_by_region(
            ["lips"],
            keep_order=True,
        ).to(self.device)
        inner_ids = self.gaussian.model.mask.get_vid_by_region(
            ["lip_inside_ring_upper", "lip_inside_ring_lower"],
            keep_order=True,
        ).to(self.device)

        w2c = torch.as_tensor(batch["w2c"], dtype=torch.float32, device=self.device)
        intrinsic = torch.as_tensor(
            batch["K"], dtype=torch.float32, device=self.device
        )
        if w2c.ndim == 2:
            w2c = w2c[None]
            intrinsic = intrinsic[None]
        width = int(torch.as_tensor(batch["width"]).reshape(-1)[0])
        height = int(torch.as_tensor(batch["height"]).reshape(-1)[0])

        def project(ids: torch.Tensor, index: int) -> np.ndarray:
            points = vertices[ids]
            camera = (
                points @ w2c[index, :3, :3].T
                + w2c[index, :3, 3][None]
            )
            homogeneous = camera @ intrinsic[index].T
            valid = (
                torch.isfinite(homogeneous).all(dim=-1)
                & (camera[:, 2] > 1.0e-5)
            )
            pixels = homogeneous[:, :2] / homogeneous[:, 2:3].clamp_min(1.0e-5)
            pixels = pixels[valid]
            if pixels.numel() == 0:
                return np.zeros((0, 2), dtype=np.int32)
            return pixels.round().long().cpu().numpy().astype(np.int32)

        hard_masks, soft_masks, apertures = [], [], []
        for index in range(w2c.shape[0]):
            hard = np.zeros((height, width), dtype=np.uint8)
            aperture = np.zeros_like(hard)
            outer = project(hard_ids, index)
            inner = project(inner_ids, index)
            if outer.shape[0] >= 3:
                cv2.fillConvexPoly(hard, cv2.convexHull(outer), 255)
            if inner.shape[0] >= 3:
                cv2.fillConvexPoly(aperture, cv2.convexHull(inner), 255)
            if dilation > 0:
                kernel = np.ones((2 * dilation + 1, 2 * dilation + 1), np.uint8)
                hard = cv2.dilate(hard, kernel, iterations=1)
            if aperture_dilation > 0:
                kernel = np.ones(
                    (2 * aperture_dilation + 1, 2 * aperture_dilation + 1),
                    np.uint8,
                )
                aperture = cv2.dilate(aperture, kernel, iterations=1)
            soft = hard.astype(np.float32) / 255.0
            if feather > 0:
                kernel = 2 * feather + 1
                soft = cv2.GaussianBlur(soft, (kernel, kernel), 0)
                soft = np.clip(soft, 0.0, 1.0)
            hard_masks.append(torch.from_numpy(hard.astype(np.float32) / 255.0))
            soft_masks.append(torch.from_numpy(soft))
            apertures.append(
                torch.from_numpy(aperture.astype(np.float32) / 255.0)
            )
        return (
            torch.stack(hard_masks)[:, None].to(self.device),
            torch.stack(soft_masks)[:, None].to(self.device),
            torch.stack(apertures)[:, None].to(self.device),
        )

    def topology_digest(self) -> str:
        return tensor_digest(
            self.initial["uv"],
            self.initial["face_idx"],
            self.initial["shape"],
        )

    def model_state(self) -> Dict[str, torch.Tensor]:
        return {
            "d": self.gaussian._d.detach().cpu(),
            "feature_dc": self.gaussian._features_dc.detach().cpu(),
            "opacity": self.gaussian._opacity.detach().cpu(),
            "scale": self.gaussian._scaling.detach().cpu(),
            "rotation": self.gaussian._rotation.detach().cpu(),
        }

    def load_model_state(self, state: Mapping[str, Any]) -> None:
        mapping = {
            "d": self.gaussian._d,
            "feature_dc": self.gaussian._features_dc,
            "opacity": self.gaussian._opacity,
            "scale": self.gaussian._scaling,
            "rotation": self.gaussian._rotation,
        }
        for name, parameter in mapping.items():
            value = torch.as_tensor(
                state[name], dtype=parameter.dtype, device=self.device
            )
            if value.shape != parameter.shape:
                raise ValueError(
                    "Checkpoint %s shape %s differs from %s"
                    % (name, tuple(value.shape), tuple(parameter.shape))
                )
            parameter.data.copy_(value)

    @torch.no_grad()
    def restore_frozen_rows(self) -> None:
        masks = {
            "d": self.geometry_mask,
            "scale": self.geometry_mask,
            "rotation": self.geometry_mask,
            "feature_dc": self.appearance_mask,
            "opacity": self.opacity_mask,
        }
        parameters = {
            "d": self.gaussian._d,
            "scale": self.gaussian._scaling,
            "rotation": self.gaussian._rotation,
            "feature_dc": self.gaussian._features_dc,
            "opacity": self.gaussian._opacity,
        }
        for name, parameter in parameters.items():
            keep = masks[name]
            parameter.data[~keep] = self.initial[name][~keep]
        rotation = self.gaussian._rotation.data
        rotation[self.geometry_mask] = F.normalize(
            rotation[self.geometry_mask], dim=-1
        )

    def assert_invariants(self) -> None:
        if not torch.equal(
            self.gaussian._uv.detach(), self.initial["uv"]
        ):
            raise RuntimeError("Canonical UV changed during direct inpaint")
        if not torch.equal(
            self.gaussian._face_idx.detach(), self.initial["face_idx"]
        ):
            raise RuntimeError("Gaussian face binding changed during direct inpaint")
        if not torch.equal(
            self.gaussian._shape.detach(), self.initial["shape"]
        ):
            raise RuntimeError("FLAME shape changed during direct inpaint")
        checks = (
            ("d", self.gaussian._d, self.geometry_mask),
            ("scale", self.gaussian._scaling, self.geometry_mask),
            ("rotation", self.gaussian._rotation, self.geometry_mask),
            ("feature_dc", self.gaussian._features_dc, self.appearance_mask),
            ("opacity", self.gaussian._opacity, self.opacity_mask),
        )
        for name, value, mask in checks:
            if not torch.equal(value.detach()[~mask], self.initial[name][~mask]):
                raise RuntimeError("Frozen Gaussian rows changed in %s" % name)

    @torch.no_grad()
    def clamp_editable_geometry(
        self, max_abs_d: float, max_world_scale: float
    ) -> None:
        mask = self.geometry_mask
        self.gaussian._d.data[mask] = self.gaussian._d.data[mask].clamp(
            -float(max_abs_d), float(max_abs_d)
        )
        self._limit_region_world_scale(mask, float(max_world_scale))
        rotation = self.gaussian._rotation.data
        rotation[mask] = F.normalize(rotation[mask], dim=-1)
        opacity_mask = self.opacity_mask
        self.gaussian._opacity.data[opacity_mask] = (
            self.gaussian._opacity.data[opacity_mask].clamp(-10.0, 10.0)
        )

    @torch.no_grad()
    def save(self, directory: Path) -> None:
        from train_reconstruction import save_world_ply

        directory.mkdir(parents=True, exist_ok=True)
        self.set_pose(*self.reference_pose)
        self.assert_invariants()
        self.gaussian.save_ply(str(directory / "uvd.ply"))
        save_world_ply(self.gaussian, self.alignment, directory / "world.ply")
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
# Calibrated data, jaw-stratified poses, and selected mouth-facing cameras
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


def selected_camera_indices(
    assets: Any, config: Mapping[str, Any]
) -> List[int]:
    target_elevation = float(config["source_elevation"])
    eligible = min(
        assets.elevation_groups,
        key=lambda group: abs(
            float(
                np.mean(
                    [
                        assets.frames[index].source_elevation_deg
                        for index in group
                    ]
                )
            )
            - target_elevation
        ),
    )
    selected: List[int] = []
    for azimuth in config["source_azimuths"]:
        available = [index for index in eligible if index not in selected]
        if not available:
            break
        chosen = min(
            available,
            key=lambda index: circular_distance_degrees(
                assets.frames[index].source_azimuth_deg, float(azimuth)
            ),
        )
        selected.append(chosen)
    if not selected:
        raise RuntimeError("No calibrated teacher cameras were selected")
    return selected


def jaw_pose_bank(
    assets: Any, config: Mapping[str, Any]
) -> List[Tuple[str, Any]]:
    bank: List[Tuple[str, Any]] = []
    if bool(config.get("include_reference_pose", True)):
        bank.append(("reference", assets.reference_pose))
    jaw_x = np.asarray(assets.chemistry_jaw[:, 0], dtype=np.float64)
    used: set = set()
    for quantile in config["pose_quantiles"]:
        value = float(np.quantile(jaw_x, float(quantile)))
        local_index = int(np.argmin(np.abs(jaw_x - value)))
        source = int(assets.chemistry_source_indices[local_index])
        if source in used:
            continue
        used.add(source)
        label = "jaw_q%02d" % int(round(100.0 * float(quantile)))
        bank.append((label, assets.chemistry_pose(local_index)))
    if bool(config.get("include_validation_pose", True)):
        bank.append(("validation", assets.validation_pose))
    return bank


def pose_to_json(pose: Any) -> Dict[str, Any]:
    return {
        "expression": np.asarray(pose.expression, dtype=np.float32).tolist(),
        "jaw_pose": np.asarray(pose.jaw_pose, dtype=np.float32).tolist(),
        "leye_pose": np.asarray(pose.leye_pose, dtype=np.float32).tolist(),
        "reye_pose": np.asarray(pose.reye_pose, dtype=np.float32).tolist(),
        "source_index": int(pose.source_index),
        "is_open_mouth": bool(pose.is_open_mouth),
        "is_reference": bool(pose.is_reference),
    }


def pose_from_json(value: Mapping[str, Any]) -> Any:
    return SimpleNamespace(
        expression=np.asarray(value["expression"], dtype=np.float32),
        jaw_pose=np.asarray(value["jaw_pose"], dtype=np.float32),
        leye_pose=np.asarray(value["leye_pose"], dtype=np.float32),
        reye_pose=np.asarray(value["reye_pose"], dtype=np.float32),
        source_index=int(value["source_index"]),
        is_open_mouth=bool(value["is_open_mouth"]),
        is_reference=bool(value["is_reference"]),
    )


# -----------------------------------------------------------------------------
# Offline SDXL mouth target generation
# -----------------------------------------------------------------------------


class MouthTargetGenerator:
    def __init__(
        self,
        config: Mapping[str, Any],
        directory: Path,
        overwrite: bool,
    ) -> None:
        self.config = config
        self.directory = directory
        self.device = torch.device(str(config.get("device", "cuda")))
        self.cache_dir = target_directory(config, directory)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(directory)
        self.data_module = create_data_module(config)
        if self.data_module.assets is None or self.data_module.builder is None:
            raise RuntimeError("Reconstruction fine-tune data was not initialized")
        self.assets = self.data_module.assets
        self.builder = self.data_module.builder
        teacher = config["teacher"]
        self.render_height = int(
            teacher.get("render_height", config["data"]["height"])
        )
        self.render_width = int(
            teacher.get("render_width", config["data"]["width"])
        )
        self.target_height = int(
            teacher.get("target_height", self.render_height)
        )
        self.target_width = int(
            teacher.get("target_width", self.render_width)
        )
        self.avatar = UVDAvatar(
            resolve_path(config["input"]["reconstruction_dir"]),
            config["bootstrap"],
            config["optimization"],
            self.device,
        )
        self.overwrite = overwrite
        self.signature = target_signature(config)
        self.accepted_mouth_priors = accepted_mouth_prior_paths(config)
        self.pipeline = None
        self.rv51_pipeline = None

    def _load_pipeline(self) -> Any:
        if self.pipeline is not None:
            return self.pipeline
        from diffusers import AutoPipelineForInpainting

        teacher = self.config["teacher"]
        kwargs: Dict[str, Any] = {
            "torch_dtype": torch.float16,
            "local_files_only": bool(teacher.get("local_files_only", True)),
        }
        variant = teacher.get("variant")
        if variant:
            kwargs["variant"] = str(variant)
        self.pipeline = AutoPipelineForInpainting.from_pretrained(
            str(resolve_path(teacher["model_path"])), **kwargs
        ).to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)
        if bool(teacher.get("attention_slicing", False)):
            self.pipeline.enable_attention_slicing()
        return self.pipeline

    def _load_rv51_pipeline(self) -> Any:
        if self.rv51_pipeline is not None:
            return self.rv51_pipeline
        from diffusers import (
            ControlNetModel,
            DDIMScheduler,
            StableDiffusionControlNetImg2ImgPipeline,
        )

        config = self.config["teacher"]["rv51"]
        dtype = (
            torch.float16
            if bool(config.get("half_precision", True))
            else torch.float32
        )
        controlnet = ControlNetModel.from_pretrained(
            str(resolve_path(config["controlnet_path"])),
            subfolder=str(config.get("controlnet_subfolder", "diffusion_sd15")),
            torch_dtype=dtype,
            local_files_only=bool(config.get("local_files_only", True)),
            use_safetensors=bool(
                config.get("controlnet_use_safetensors", False)
            ),
        )
        self.rv51_pipeline = (
            StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                str(resolve_path(config["model_path"])),
                controlnet=controlnet,
                torch_dtype=dtype,
                local_files_only=bool(config.get("local_files_only", True)),
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            ).to(self.device)
        )
        scheduler_path = resolve_path(
            config.get("scheduler_path", config["model_path"])
        )
        self.rv51_pipeline.scheduler = DDIMScheduler.from_pretrained(
            str(scheduler_path),
            subfolder="scheduler",
            local_files_only=bool(config.get("local_files_only", True)),
        )
        self.rv51_pipeline.set_progress_bar_config(disable=True)
        if bool(config.get("attention_slicing", False)):
            self.rv51_pipeline.enable_attention_slicing()
        return self.rv51_pipeline

    def _view_name(self, source_azimuth: float) -> str:
        difference = circular_distance_degrees(source_azimuth, 270.0)
        if difference <= 15.0:
            return "front view"
        if difference <= 55.0:
            return "three-quarter view"
        return "side view"

    def _load_accepted_mouth_prior(
        self,
        observation_key: str,
        source: np.ndarray,
        mouth_mask: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load and verify a manually accepted prior against this observation."""

        path = self.accepted_mouth_priors[observation_key]
        source_path = accepted_mouth_prior_source(path)
        height, width = source.shape[:2]
        with Image.open(path) as image:
            source_resolution = [int(image.height), int(image.width)]
            prior = np.asarray(
                image.convert("RGB").resize(
                    (width, height), Image.Resampling.LANCZOS
                ),
                dtype=np.float32,
            ) / 255.0
        with Image.open(source_path) as image:
            prior_source = np.asarray(
                image.convert("RGB").resize(
                    (width, height), Image.Resampling.LANCZOS
                ),
                dtype=np.float32,
            ) / 255.0
        current_source = source.astype(np.float32)
        if current_source.size and float(current_source.max()) > 1.0:
            current_source = current_source / 255.0
        weight = mouth_mask.astype(np.float32).clip(0.0, 1.0)
        source_mae = float(
            (
                np.abs(current_source - prior_source).mean(axis=-1)
                * weight
            ).sum()
            / max(float(weight.sum()), 1.0e-8)
        )
        maximum_mae = float(
            self.config["teacher"].get(
                "accepted_mouth_prior_max_source_mae", 0.01
            )
        )
        if source_mae > maximum_mae:
            raise RuntimeError(
                "%s accepted mouth prior source MAE %.6f > %.6f; "
                "the prior does not match this pose/view"
                % (observation_key, source_mae, maximum_mae)
            )
        return prior.astype(np.float32, copy=False), {
            "configured_path": str(
                self.config["teacher"]["accepted_mouth_priors"][
                    observation_key
                ]
            ),
            "resolved_path": str(path),
            "sha256": file_digest(path),
            "source_path": str(source_path),
            "source_sha256": file_digest(source_path),
            "source_resolution": source_resolution,
            "source_mouth_mae": source_mae,
        }

    def _refine_rv51(
        self,
        image: np.ndarray,
        source: np.ndarray,
        composite_mask: np.ndarray,
        condition: torch.Tensor,
        view: str,
        seed: int,
    ) -> np.ndarray:
        config = self.config["teacher"]["rv51"]
        pipeline = self._load_rv51_pipeline()
        rv_size = int(config.get("image_size", image.shape[0]))
        if condition.ndim == 3 and condition.shape[-1] == 3:
            condition = condition.permute(2, 0, 1)
        if condition.ndim == 3:
            condition = condition[None]
        condition = F.interpolate(
            condition.float().to(self.device),
            size=(rv_size, rv_size),
            mode="bilinear",
            align_corners=False,
        )
        prompt = str(config["prompt"]).strip()
        if bool(config.get("append_view_prompt", True)):
            prompt = "%s, %s" % (prompt, view)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed) + int(config.get("seed_offset", 0)))
        image_u8 = (
            image * 255.0
        ).round().clip(0, 255).astype(np.uint8)
        image_input = Image.fromarray(image_u8)
        if image_input.size != (rv_size, rv_size):
            image_input = image_input.resize(
                (rv_size, rv_size), Image.Resampling.LANCZOS
            )
        result = pipeline(
            prompt=prompt,
            negative_prompt=str(config["negative_prompt"]).strip(),
            image=image_input,
            control_image=condition,
            strength=float(config.get("strength", 0.25)),
            guidance_scale=float(config.get("guidance_scale", 7.5)),
            controlnet_conditioning_scale=float(
                config.get("controlnet_conditioning_scale", 1.5)
            ),
            num_inference_steps=int(
                config.get("num_inference_steps", 50)
            ),
            generator=generator,
        ).images[0]
        refined = np.asarray(result, dtype=np.float32) / 255.0
        original = source.astype(np.float32)
        if original.size and float(original.max()) > 1.0:
            original = original / 255.0
        if original.shape[:2] != (rv_size, rv_size):
            original = cv2.resize(
                original,
                (rv_size, rv_size),
                interpolation=cv2.INTER_LANCZOS4,
            )
        blend = cv2.resize(
            composite_mask.astype(np.float32),
            (rv_size, rv_size),
            interpolation=cv2.INTER_LINEAR,
        )[..., None].clip(0.0, 1.0)
        composed = np.clip(
            original * (1.0 - blend) + refined * blend,
            0.0,
            1.0,
        )
        if composed.shape[:2] != image.shape[:2]:
            composed = cv2.resize(
                composed,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        return composed.astype(np.float32, copy=False)

    def _inpaint_sample(
        self,
        source: np.ndarray,
        hard_mask: np.ndarray,
        soft_mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        prompt: str,
        negative_prompt: str,
        seed: int,
        settings: Optional[Mapping[str, Any]] = None,
    ) -> np.ndarray:
        pipeline = self._load_pipeline()
        teacher = self.config["teacher"]
        diffusion = teacher if settings is None else settings
        full_frame = bool(teacher.get("full_frame", True))
        if full_frame:
            size = int(teacher["image_size"])
            source_input = Image.fromarray(source).resize(
                (size, size), Image.Resampling.LANCZOS
            )
            mask_input = Image.fromarray(
                (hard_mask * 255.0).round().astype(np.uint8)
            ).resize((size, size), Image.Resampling.NEAREST)
        else:
            x0, y0, x1, y1 = bbox
            size = int(teacher["image_size"])
            source_input = Image.fromarray(source[y0:y1, x0:x1]).resize(
                (size, size), Image.Resampling.LANCZOS
            )
            mask_input = Image.fromarray(
                (hard_mask[y0:y1, x0:x1] * 255.0)
                .round()
                .astype(np.uint8)
            ).resize((size, size), Image.Resampling.NEAREST)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source_input,
            mask_image=mask_input,
            height=source_input.height,
            width=source_input.width,
            guidance_scale=float(diffusion["guidance_scale"]),
            num_inference_steps=int(diffusion["num_inference_steps"]),
            strength=float(diffusion["strength"]),
            generator=generator,
        ).images[0]
        if full_frame:
            result = np.asarray(
                result.resize(
                    (source.shape[1], source.shape[0]),
                    Image.Resampling.LANCZOS,
                ),
                dtype=np.float32,
            ) / 255.0
            generated = result
        else:
            result = np.asarray(
                result.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS),
                dtype=np.float32,
            ) / 255.0
            generated = source.astype(np.float32) / 255.0
            generated = generated.copy()
            generated[y0:y1, x0:x1] = result
        blend = soft_mask[..., None].astype(np.float32)
        original = source.astype(np.float32) / 255.0
        return np.clip(original * (1.0 - blend) + generated * blend, 0.0, 1.0)

    def _build_repair_masks(
        self,
        batch: Mapping[str, Any],
        rendering: RenderBatch,
    ) -> Dict[str, np.ndarray]:
        """Build the mouth inpaint mask and the global RV foreground mask."""

        teacher = self.config["teacher"]
        rv51 = teacher.get("rv51", {})
        height = self.render_height
        width = self.render_width
        reference = int(teacher.get("mask_reference_resolution", 1024))

        mouth_hard, mouth_soft, aperture = self.avatar.lip_masks(
            batch,
            dilation=resolution_scaled_pixels(
                teacher["mask_dilation"], height, width, reference
            ),
            feather=resolution_scaled_pixels(
                teacher["mask_feather"], height, width, reference
            ),
            aperture_dilation=resolution_scaled_pixels(
                teacher["aperture_dilation"], height, width, reference
            ),
        )
        _, foreground_soft = self.avatar.foreground_masks(
            rendering.alpha,
            threshold=float(
                rv51.get("foreground_alpha_threshold", 0.01)
            ),
            dilation=resolution_scaled_pixels(
                rv51.get("foreground_dilation", 2),
                height,
                width,
                reference,
            ),
            feather=resolution_scaled_pixels(
                rv51.get("foreground_feather", 4),
                height,
                width,
                reference,
            ),
        )

        def as_float(mask: torch.Tensor) -> np.ndarray:
            return gray_u8(mask[0]).astype(np.float32) / 255.0

        mouth_hard_np = as_float(mouth_hard)
        mouth_soft_np = as_float(mouth_soft)
        foreground_soft_np = as_float(foreground_soft)
        scope = str(rv51.get("composite_scope", "foreground")).lower()
        rv_mask = (
            foreground_soft_np.copy()
            if scope == "foreground"
            else np.ones_like(mouth_soft_np)
        )

        aperture_np = as_float(aperture)
        upper_hint = (
            as_float(rendering.region_alpha["teeth_upper"]) * aperture_np
        )
        lower_hint = (
            as_float(rendering.region_alpha["teeth_lower"]) * aperture_np
        )
        return {
            "mouth_hard": mouth_hard_np,
            "mouth_soft": mouth_soft_np,
            "rv": rv_mask,
            "foreground": foreground_soft_np,
            "aperture": aperture_np,
            "upper_hint": upper_hint,
            "lower_hint": lower_hint,
            "teeth_hint": np.maximum(upper_hint, lower_hint),
            "ear_hint": as_float(rendering.region_alpha["ears"]),
        }

    def _oral_metrics(
        self,
        image: np.ndarray,
        masks: Mapping[str, np.ndarray],
    ) -> Dict[str, Any]:
        quality = self.config["teacher"].get("quality", {})
        teeth_coverage = dental_coverage(
            image,
            masks["teeth_hint"],
            float(quality.get("teeth_hint_threshold", 0.10)),
            float(quality.get("teeth_min_luminance", 0.48)),
            float(quality.get("teeth_max_saturation", 0.42)),
        )
        structure = dental_structure(
            image,
            masks["aperture"],
            float(quality.get("teeth_min_luminance", 0.48)),
            float(quality.get("teeth_max_chroma", 0.20)),
        )
        return {
            "teeth_coverage": float(teeth_coverage),
            "structure": structure,
        }

    def _oral_anatomy_failures(
        self,
        pose_label: str,
        source_azimuth: float,
        metrics: Mapping[str, Any],
    ) -> List[str]:
        """Reject denture-like mouths without hiding valid oblique views."""

        quality = self.config["teacher"].get("quality", {})
        structure = metrics["structure"]
        failures: List[str] = []
        maximum_component = float(
            quality.get("max_teeth_component_coverage", 0.12)
        )
        if float(structure["coverage"]) > maximum_component:
            failures.append(
                "tooth component coverage %.4f > %.4f"
                % (float(structure["coverage"]), maximum_component)
            )
        maximum_span = float(
            quality.get("max_teeth_component_span", 0.78)
        )
        if float(structure["span"]) > maximum_span:
            failures.append(
                "tooth component span %.4f > %.4f"
                % (float(structure["span"]), maximum_span)
            )
        minimum_dark = float(
            quality.get("min_mouth_dark_fraction", 0.28)
        )
        if float(structure["dark_fraction"]) < minimum_dark:
            failures.append(
                "mouth dark fraction %.4f < %.4f"
                % (float(structure["dark_fraction"]), minimum_dark)
            )

        distance = circular_distance_degrees(source_azimuth, 270.0)
        # At q80 and at oblique cameras, a natural projection may show almost
        # no white tooth component.  Enforce visibility only for the wider
        # q95/validation front view; all views still retain the upper bound.
        require_visible_teeth = (
            distance <= 15.0 and pose_label != "jaw_q80"
        )
        if require_visible_teeth:
            minimum_teeth = float(
                quality.get("min_teeth_coverage", 0.10)
            )
            if float(metrics["teeth_coverage"]) < minimum_teeth:
                failures.append(
                    "teeth coverage %.4f < %.4f"
                    % (float(metrics["teeth_coverage"]), minimum_teeth)
                )
            minimum_component = float(
                quality.get("min_teeth_component_coverage", 0.005)
            )
            if float(structure["coverage"]) < minimum_component:
                failures.append(
                    "tooth component coverage %.4f < %.4f"
                    % (float(structure["coverage"]), minimum_component)
                )
            minimum_aspect = float(
                quality.get("min_teeth_component_aspect", 1.5)
            )
            if float(structure["aspect"]) < minimum_aspect:
                failures.append(
                    "tooth component aspect %.4f < %.4f"
                    % (float(structure["aspect"]), minimum_aspect)
                )
            minimum_span = float(
                quality.get("min_teeth_span_front", 0.15)
            )
            if float(structure["span"]) < minimum_span:
                failures.append(
                    "tooth span %.4f < %.4f"
                    % (float(structure["span"]), minimum_span)
                )
        return failures

    def _select_oral_refinement(
        self,
        pose_label: str,
        source_azimuth: float,
        rv_raw: np.ndarray,
        sdxl_target: np.ndarray,
        masks: Mapping[str, np.ndarray],
    ) -> Tuple[np.ndarray, float]:
        """Prefer raw RV and use the smallest accepted inward SDXL fallback."""

        quality = self.config["teacher"].get("quality", {})
        if not bool(quality.get("enforce", True)):
            return rv_raw.astype(np.float32, copy=False), 0.0
        aperture = masks["aperture"]
        if not bool((aperture > 0.5).any()):
            raise RuntimeError(
                "%s has an empty lip aperture" % pose_label
            )
        rv51 = self.config["teacher"].get("rv51", {})
        feather_reference = int(rv51.get("mouth_fallback_feather", 8))
        feather = (
            resolution_scaled_pixels(
                feather_reference,
                rv_raw.shape[0],
                rv_raw.shape[1],
                int(
                    self.config["teacher"].get(
                        "mask_reference_resolution", 1024
                    )
                ),
            )
            if feather_reference > 0
            else 0
        )
        weight = inward_aperture_weight(aperture, feather)
        failures_by_alpha: List[str] = []
        for raw_alpha in rv51.get(
            "mouth_fallback_alphas", [0.0, 0.15, 0.30]
        ):
            alpha = float(raw_alpha)
            blend = (alpha * weight)[..., None]
            candidate = np.clip(
                rv_raw * (1.0 - blend) + sdxl_target * blend,
                0.0,
                1.0,
            ).astype(np.float32, copy=False)
            failures = self._oral_anatomy_failures(
                pose_label,
                source_azimuth,
                self._oral_metrics(candidate, masks),
            )
            if not failures:
                return candidate, alpha
            failures_by_alpha.append(
                "alpha %.2f: %s" % (alpha, ", ".join(failures))
            )
        raise RuntimeError(
            "%s oral refinement rejected every fallback: %s"
            % (pose_label, "; ".join(failures_by_alpha))
        )

    def _target_quality(
        self,
        pose_label: str,
        frame_index: int,
        source_azimuth: float,
        source: np.ndarray,
        sdxl_target: np.ndarray,
        mouth_teacher: np.ndarray,
        rv_raw: np.ndarray,
        target: np.ndarray,
        masks: Mapping[str, np.ndarray],
        should_mouth_inpaint: bool,
        should_rv51_refine: bool,
        edit_mask: np.ndarray,
        mouth_fallback_alpha: float,
        mouth_source: str,
    ) -> Dict[str, Any]:
        """Calculate QA metrics and optionally enforce them before caching."""

        quality = self.config["teacher"].get("quality", {})
        enforce_quality = bool(quality.get("enforce", True))
        source_u8 = source.astype(np.uint8)
        target_u8 = (
            target * 255.0
        ).round().clip(0, 255).astype(np.uint8)
        sdxl_u8 = (
            sdxl_target * 255.0
        ).round().clip(0, 255).astype(np.uint8)
        mouth_teacher_u8 = (
            mouth_teacher * 255.0
        ).round().clip(0, 255).astype(np.uint8)
        edit_u8 = (
            edit_mask * 255.0
        ).round().clip(0, 255).astype(np.uint8)
        channel_delta = np.abs(
            target_u8.astype(np.int16) - source_u8.astype(np.int16)
        )
        outside = edit_u8 == 0
        outside_edit_max = (
            float(channel_delta[outside].max()) / 255.0
            if bool(outside.any())
            else 0.0
        )
        outside_mismatch_pixels = (
            int(np.any(channel_delta != 0, axis=-1)[outside].sum())
            if bool(outside.any())
            else 0
        )
        mouth_support = (
            masks["aperture"] * 255.0
        ).round().astype(np.uint8) > 0
        sdxl_mouth_delta_max = (
            float(
                np.abs(
                    target_u8.astype(np.int16)
                    - sdxl_u8.astype(np.int16)
                )[mouth_support].max()
            )
            / 255.0
            if should_mouth_inpaint and bool(mouth_support.any())
            else 0.0
        )
        source_float = source_u8.astype(np.float32) / 255.0
        pixel_delta = np.abs(target - source_float).mean(axis=-1)
        sdxl_oral = self._oral_metrics(sdxl_target, masks)
        mouth_teacher_oral = self._oral_metrics(mouth_teacher, masks)
        rv_raw_oral = self._oral_metrics(rv_raw, masks)
        target_oral = self._oral_metrics(target, masks)
        non_mouth_weight = (
            masks["foreground"].clip(0.0, 1.0)
            * (1.0 - masks["mouth_soft"].clip(0.0, 1.0))
        )
        non_mouth_delta = float(
            (pixel_delta * non_mouth_weight).sum()
            / max(float(non_mouth_weight.sum()), 1.0e-8)
        )
        ear_weight = masks["ear_hint"].clip(0.0, 1.0)
        ear_edit_coverage = float(
            (edit_mask.clip(0.0, 1.0) * ear_weight).sum()
            / max(float(ear_weight.sum()), 1.0e-8)
        )
        ear_delta = float(
            (pixel_delta * ear_weight).sum()
            / max(float(ear_weight.sum()), 1.0e-8)
        )
        failures: List[str] = []
        gates = (
            (
                outside_edit_max,
                float(quality.get("max_outside_edit_delta", 1.0e-6)),
                "outside edit delta",
            ),
        )
        for value, maximum, label in gates:
            if value > maximum:
                failures.append("%s %.8f > %.8f" % (label, value, maximum))
        if outside_mismatch_pixels:
            failures.append(
                "outside mismatch pixels %d" % outside_mismatch_pixels
            )
        if should_rv51_refine:
            minimum_delta = float(
                quality.get("min_non_mouth_delta", 0.0)
            )
            if non_mouth_delta < minimum_delta:
                failures.append(
                    "non-mouth RV delta %.6f < %.6f"
                    % (non_mouth_delta, minimum_delta)
                )
            minimum_ear_coverage = float(
                quality.get("min_ear_edit_coverage", 0.0)
            )
            if bool(ear_weight.any()) and (
                ear_edit_coverage < minimum_ear_coverage
            ):
                failures.append(
                    "ear RV coverage %.4f < %.4f"
                    % (ear_edit_coverage, minimum_ear_coverage)
                )
            minimum_ear_delta = float(
                quality.get("min_ear_delta", 0.0)
            )
            if bool(ear_weight.any()) and ear_delta < minimum_ear_delta:
                failures.append(
                    "ear RV delta %.6f < %.6f"
                    % (ear_delta, minimum_ear_delta)
                )
        if should_mouth_inpaint:
            failures.extend(
                self._oral_anatomy_failures(
                    pose_label,
                    source_azimuth,
                    target_oral,
                )
            )
        accepted_prior_core_delta = 0.0
        if mouth_source == "accepted_prior":
            prior_core = masks["mouth_soft"] >= 0.999
            if not bool(prior_core.any()):
                failures.append("accepted mouth prior has no solid mask core")
            else:
                accepted_prior_core_delta = (
                    float(
                        np.abs(
                            target_u8.astype(np.int16)
                            - mouth_teacher_u8.astype(np.int16)
                        )[prior_core].max()
                    )
                    / 255.0
                )
                maximum_prior_delta = 1.0 / 255.0
                if accepted_prior_core_delta > maximum_prior_delta:
                    failures.append(
                        "accepted mouth prior core delta %.8f > %.8f"
                        % (
                            accepted_prior_core_delta,
                            maximum_prior_delta,
                        )
                    )
        if failures and enforce_quality:
            raise RuntimeError(
                "%s frame %03d target QA failed: %s"
                % (pose_label, int(frame_index), "; ".join(failures))
            )
        return {
            "qa_enforced": enforce_quality,
            "qa_passed": not failures,
            "qa_failures": failures,
            "outside_edit_max": outside_edit_max,
            "outside_mismatch_pixels": outside_mismatch_pixels,
            "sdxl_mouth_delta_max": sdxl_mouth_delta_max,
            "mouth_fallback_alpha": float(mouth_fallback_alpha),
            "mouth_source": mouth_source,
            "accepted_mouth_prior_core_delta": (
                accepted_prior_core_delta
            ),
            "sdxl_teeth_coverage": sdxl_oral["teeth_coverage"],
            "mouth_teacher_teeth_coverage": mouth_teacher_oral[
                "teeth_coverage"
            ],
            "rv_raw_teeth_coverage": rv_raw_oral["teeth_coverage"],
            "target_teeth_coverage": target_oral["teeth_coverage"],
            "sdxl_teeth_structure": sdxl_oral["structure"],
            "mouth_teacher_teeth_structure": mouth_teacher_oral[
                "structure"
            ],
            "rv_raw_teeth_structure": rv_raw_oral["structure"],
            "target_teeth_structure": target_oral["structure"],
            "non_mouth_delta_mean": non_mouth_delta,
            "ear_edit_coverage": ear_edit_coverage,
            "ear_delta_mean": ear_delta,
        }

    def _write_observation(
        self,
        key: str,
        source: np.ndarray,
        sdxl_target: np.ndarray,
        mouth_teacher: np.ndarray,
        mouth_source: str,
        rv_raw: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
        mouth_mask: np.ndarray,
        foreground_mask: np.ndarray,
        aperture: np.ndarray,
        confidence: np.ndarray,
        upper_hint: np.ndarray,
        lower_hint: np.ndarray,
        ear_hint: np.ndarray,
    ) -> Dict[str, str]:
        teeth_hint = np.maximum(upper_hint, lower_hint)
        paths = {
            "source": "%s_source.png" % key,
            "sdxl_target": "%s_sdxl.png" % key,
            "mouth_teacher": "%s_mouth_teacher.png" % key,
            "rv_raw": "%s_rv_raw.png" % key,
            "target": "%s_target.png" % key,
            "mask": "%s_mask.png" % key,
            "mouth_mask": "%s_mouth_mask.png" % key,
            "foreground_mask": "%s_foreground_mask.png" % key,
            "aperture": "%s_aperture.png" % key,
            "confidence": "%s_confidence.png" % key,
            "teeth_hint": "%s_teeth_hint.png" % key,
            "ear_hint": "%s_ear_hint.png" % key,
            "preview": "%s_preview.jpg" % key,
        }
        Image.fromarray(source).save(self.cache_dir / paths["source"])
        Image.fromarray(
            (sdxl_target * 255.0).round().clip(0, 255).astype(np.uint8)
        ).save(self.cache_dir / paths["sdxl_target"])
        Image.fromarray(
            (mouth_teacher * 255.0)
            .round()
            .clip(0, 255)
            .astype(np.uint8)
        ).save(self.cache_dir / paths["mouth_teacher"])
        Image.fromarray(
            (rv_raw * 255.0).round().clip(0, 255).astype(np.uint8)
        ).save(self.cache_dir / paths["rv_raw"])
        Image.fromarray(
            (target * 255.0).round().clip(0, 255).astype(np.uint8)
        ).save(self.cache_dir / paths["target"])
        Image.fromarray((mask * 255.0).round().astype(np.uint8)).save(
            self.cache_dir / paths["mask"]
        )
        Image.fromarray(
            (mouth_mask * 255.0).round().astype(np.uint8)
        ).save(self.cache_dir / paths["mouth_mask"])
        Image.fromarray(
            (foreground_mask * 255.0).round().astype(np.uint8)
        ).save(self.cache_dir / paths["foreground_mask"])
        Image.fromarray((aperture * 255.0).round().astype(np.uint8)).save(
            self.cache_dir / paths["aperture"]
        )
        Image.fromarray((confidence * 255.0).round().astype(np.uint8)).save(
            self.cache_dir / paths["confidence"]
        )
        Image.fromarray((teeth_hint * 255.0).round().astype(np.uint8)).save(
            self.cache_dir / paths["teeth_hint"]
        )
        Image.fromarray((ear_hint * 255.0).round().astype(np.uint8)).save(
            self.cache_dir / paths["ear_hint"]
        )
        mask_rgb = np.repeat(
            (mask[..., None] * 255.0).round().astype(np.uint8), 3, axis=2
        )
        mouth_mask_rgb = np.repeat(
            (mouth_mask[..., None] * 255.0).round().astype(np.uint8),
            3,
            axis=2,
        )
        foreground_rgb = np.repeat(
            (foreground_mask[..., None] * 255.0).round().astype(np.uint8),
            3,
            axis=2,
        )
        ear_rgb = np.repeat(
            (ear_hint[..., None] * 255.0).round().astype(np.uint8),
            3,
            axis=2,
        )
        aperture_rgb = np.repeat(
            (aperture[..., None] * 255.0).round().astype(np.uint8), 3, axis=2
        )
        teeth_rgb = (
            np.stack(
                (upper_hint, lower_hint, np.zeros_like(upper_hint)), axis=-1
            )
            * 255.0
        ).round().astype(np.uint8)
        target_u8 = (target * 255.0).round().clip(0, 255).astype(np.uint8)
        preview = np.concatenate(
            (
                add_label(source, "bootstrapped render"),
                add_label(
                    (sdxl_target * 255.0)
                    .round()
                    .clip(0, 255)
                    .astype(np.uint8),
                    "SDXL proposal",
                ),
                add_label(
                    (mouth_teacher * 255.0)
                    .round()
                    .clip(0, 255)
                    .astype(np.uint8),
                    (
                        "accepted mouth teacher"
                        if mouth_source == "accepted_prior"
                        else "adaptive mouth teacher"
                    ),
                ),
                add_label(
                    (rv_raw * 255.0)
                    .round()
                    .clip(0, 255)
                    .astype(np.uint8),
                    "global RV raw",
                ),
                add_label(
                    target_u8,
                    (
                        "RV + accepted mouth"
                        if mouth_source == "accepted_prior"
                        else "adaptive oral result"
                    ),
                ),
                add_label(mask_rgb, "training edit mask"),
                add_label(mouth_mask_rgb, "SDXL mouth mask"),
                add_label(foreground_rgb, "RV foreground mask"),
                add_label(ear_rgb, "ear Gaussian alpha"),
                add_label(aperture_rgb, "lip aperture"),
                add_label(teeth_rgb, "upper(red) lower(green)"),
            ),
            axis=1,
        )
        Image.fromarray(preview).save(
            self.cache_dir / paths["preview"], quality=92
        )
        return paths

    @torch.inference_mode()
    def generate(self) -> Path:
        manifest_path = self.cache_dir / "manifest.json"
        if manifest_path.is_file() and not self.overwrite:
            with manifest_path.open("r", encoding="utf-8") as file:
                existing = json.load(file)
            if (
                int(existing.get("version", -1)) == MANIFEST_VERSION
                and existing.get("signature") == self.signature
                and existing.get("topology_sha256")
                == self.avatar.topology_digest()
            ):
                self.logger.info("Using existing target cache: %s", manifest_path)
                return manifest_path
            raise ValueError(
                "Existing target cache belongs to another configuration; "
                "use --overwrite-targets or change output.name"
            )

        teacher = self.config["teacher"]
        poses = jaw_pose_bank(self.assets, teacher)
        cameras = selected_camera_indices(self.assets, teacher)
        # The front view selects the shared mouth seed before side views are
        # generated, so all poses/views can reuse one tooth-layout candidate.
        cameras = sorted(
            cameras,
            key=lambda index: circular_distance_degrees(
                self.assets.frames[index].source_azimuth_deg, 270.0
            ),
        )
        entries: List[Dict[str, Any]] = []
        samples = max(int(teacher.get("samples_per_view", 1)), 1)
        mouth_candidates = max(
            int(teacher.get("mouth_candidates_per_view", 1)), 1
        )
        base_seed = int(teacher.get("seed", self.config.get("seed", 0)))
        rv51_config = teacher.get("rv51", {})
        rv51_enabled = bool(rv51_config.get("enabled", False))
        rv51_scope = str(
            rv51_config.get("composite_scope", "foreground")
        ).lower()
        quality = teacher.get("quality", {})
        render_height = self.render_height
        render_width = self.render_width
        selected_candidate_indices: Dict[str, int] = {}
        applied_prior_keys: set[str] = set()
        tau = max(float(teacher.get("confidence_tau", 0.002)), 1.0e-8)
        progress = tqdm(
            total=len(poses) * len(cameras),
            desc="Generate repair targets",
            dynamic_ncols=True,
        )

        for pose_index, (pose_label, pose) in enumerate(poses):
            for camera_slot, camera_index in enumerate(cameras):
                batch = self.builder.build([camera_index], pose)
                batch = rescale_render_batch(
                    batch, render_height, render_width
                )
                batch = to_device(batch, self.device)
                rendering = self.avatar.render_batch(
                    batch,
                    include_regions=True,
                    differentiable=False,
                    region_names=(
                        "teeth_upper",
                        "teeth_lower",
                        "ears",
                    ),
                )
                masks = self._build_repair_masks(batch, rendering)
                source = rgb_u8(rendering.rgb[0])
                hard_np = masks["mouth_hard"]
                soft_np = masks["mouth_soft"]
                rv51_mask_np = masks["rv"]
                aperture_np = masks["aperture"]
                upper_hint = masks["upper_hint"]
                lower_hint = masks["lower_hint"]
                ear_hint = masks["ear_hint"]
                bbox = square_bbox(
                    (hard_np > 0.0).astype(np.uint8),
                    float(teacher["crop_scale"]),
                    int(teacher["crop_min_size"]),
                )
                frame = self.assets.frames[camera_index]
                view = self._view_name(frame.source_azimuth_deg)
                should_mouth_inpaint = (
                    not bool(pose.is_reference)
                    or bool(teacher.get("inpaint_reference_pose", False))
                )
                should_rv51_refine = rv51_enabled and (
                    should_mouth_inpaint
                    or bool(rv51_config.get("refine_reference_pose", False))
                )
                mouth_edit_np = (
                    soft_np if should_mouth_inpaint
                    else np.zeros_like(soft_np)
                )
                rv_edit_np = (
                    rv51_mask_np if should_rv51_refine
                    else np.zeros_like(rv51_mask_np)
                )
                edit_mask_np = np.maximum(mouth_edit_np, rv_edit_np)
                rv_mask_fraction = float((rv_edit_np > 0.01).mean())
                edit_mask_fraction = float((edit_mask_np > 0.01).mean())
                if edit_mask_fraction > float(
                    quality.get("max_edit_mask_fraction", 0.50)
                ):
                    raise RuntimeError(
                        "%s frame %03d edit mask fraction %.4f is unsafe"
                        % (
                            pose_label,
                            int(frame.frame_index),
                            edit_mask_fraction,
                        )
                    )
                generated: List[np.ndarray] = []
                generated_sdxl: List[np.ndarray] = []
                generated_rv_raw: List[np.ndarray] = []
                sample_seeds: List[int] = []
                sample_fallback_alphas: List[float] = []
                mouth_candidate_seeds: List[List[int]] = []
                mouth_candidate_scores: List[List[float]] = []
                prompt = str(teacher["open_mouth_prompt"]).strip()
                if bool(teacher.get("append_view_prompt", False)):
                    prompt = "%s, %s" % (prompt, view)
                negative = str(teacher["negative_prompt"]).strip()
                observation_index = pose_index * len(cameras) + camera_slot
                override_key = "%s_frame_%03d" % (
                    pose_label,
                    int(frame.frame_index),
                )
                has_accepted_prior = (
                    override_key in self.accepted_mouth_priors
                )
                if has_accepted_prior and not should_mouth_inpaint:
                    raise RuntimeError(
                        "%s configures an accepted mouth prior for an "
                        "observation that is not mouth-inpainted" % override_key
                    )
                seed_override = teacher.get(
                    "seed_overrides", {}
                ).get(override_key)
                unchanged = source.astype(np.float32) / 255.0
                seed_mode = str(teacher.get("seed_mode", "shared")).lower()
                if seed_override is not None:
                    selection_key = "override:%s" % override_key
                elif seed_mode == "shared":
                    selection_key = "shared"
                elif seed_mode == "pose":
                    selection_key = "pose:%s" % pose_label
                else:
                    selection_key = "observation:%s" % override_key
                for sample_index in range(samples):
                    if seed_mode == "shared":
                        seed_group = 0
                    elif seed_mode == "pose":
                        seed_group = pose_index
                    else:
                        seed_group = observation_index
                    candidate_images: List[np.ndarray] = []
                    candidate_seeds: List[int] = []
                    candidate_scores: List[float] = []
                    known_candidate = selected_candidate_indices.get(
                        selection_key
                    )
                    if not should_mouth_inpaint:
                        candidate_indices = [0]
                    elif known_candidate is None:
                        candidate_indices = list(range(mouth_candidates))
                    else:
                        candidate_indices = [known_candidate]
                    for candidate_index in candidate_indices:
                        offset = (
                            sample_index * mouth_candidates
                            + candidate_index
                        )
                        if seed_override is not None:
                            candidate_seed = int(seed_override) + offset
                        else:
                            candidate_seed = (
                                base_seed
                                + seed_group
                                * samples
                                * mouth_candidates
                                + offset
                        )
                        if should_mouth_inpaint:
                            candidate = self._inpaint_sample(
                                source,
                                hard_np,
                                soft_np,
                                bbox,
                                prompt,
                                negative,
                                candidate_seed,
                            )
                            oral = self._oral_metrics(candidate, masks)
                            structure = oral["structure"]
                            preferred_component = float(
                                quality.get(
                                    "preferred_teeth_component_coverage",
                                    0.035,
                                )
                            )
                            preferred_span = float(
                                quality.get("preferred_teeth_span", 0.48)
                            )
                            preferred_dark = float(
                                quality.get(
                                    "preferred_mouth_dark_fraction", 0.40
                                )
                            )
                            maximum_component = float(
                                quality.get(
                                    "max_teeth_component_coverage", 0.12
                                )
                            )
                            # Rank around a natural center instead of rewarding
                            # the whitest and widest connected tooth strip.
                            score = -(
                                abs(
                                    float(structure["coverage"])
                                    - preferred_component
                                )
                                + 0.25
                                * abs(
                                    float(structure["span"])
                                    - preferred_span
                                )
                                + 0.15
                                * abs(
                                    float(structure["dark_fraction"])
                                    - preferred_dark
                                )
                                + 4.0
                                * max(
                                    float(structure["coverage"])
                                    - maximum_component,
                                    0.0,
                                )
                            )
                        else:
                            candidate = unchanged.copy()
                            score = 0.0
                        candidate_images.append(candidate)
                        candidate_seeds.append(candidate_seed)
                        candidate_scores.append(score)
                    best_candidate = (
                        int(np.argmax(candidate_scores))
                        if should_mouth_inpaint
                        else 0
                    )
                    chosen_candidate_index = candidate_indices[best_candidate]
                    if should_mouth_inpaint and known_candidate is None:
                        selected_candidate_indices[selection_key] = (
                            chosen_candidate_index
                        )
                    seed = candidate_seeds[best_candidate]
                    sdxl_target = candidate_images[best_candidate]
                    sample_seeds.append(seed)
                    mouth_candidate_seeds.append(candidate_seeds)
                    mouth_candidate_scores.append(candidate_scores)
                    generated_sdxl.append(sdxl_target)

                    if should_rv51_refine:
                        rv_raw = self._refine_rv51(
                            sdxl_target,
                            sdxl_target,
                            rv51_mask_np,
                            batch["flame_conds"][0],
                            view,
                            seed,
                        )
                    else:
                        rv_raw = sdxl_target.copy()
                    if (
                        should_mouth_inpaint
                        and should_rv51_refine
                        and not has_accepted_prior
                    ):
                        final_target, fallback_alpha = (
                            self._select_oral_refinement(
                                pose_label,
                                float(frame.source_azimuth_deg),
                                rv_raw,
                                sdxl_target,
                                masks,
                            )
                        )
                    else:
                        final_target = rv_raw.copy()
                        fallback_alpha = 0.0
                    generated_rv_raw.append(rv_raw)
                    sample_fallback_alphas.append(float(fallback_alpha))
                    generated.append(final_target)

                stack = np.stack(generated, axis=0)
                sample_teeth_scores = [
                    dental_coverage(
                        item,
                        masks["teeth_hint"],
                        float(
                            quality.get("teeth_hint_threshold", 0.10)
                        ),
                        float(
                            quality.get("teeth_min_luminance", 0.48)
                        ),
                        float(
                            quality.get("teeth_max_saturation", 0.42)
                        ),
                    )
                    for item in generated
                ]
                selected_sample = min(
                    range(len(generated)),
                    key=lambda index: (
                        sample_fallback_alphas[index],
                        index,
                    ),
                )
                sdxl_target = generated_sdxl[selected_sample]
                rv_raw = generated_rv_raw[selected_sample]
                target = stack[selected_sample].copy()
                fallback_alpha = sample_fallback_alphas[selected_sample]
                variance = stack.var(axis=0).mean(axis=-1)
                confidence = np.exp(-variance / tau).astype(np.float32)
                confidence = np.clip(
                    confidence,
                    float(teacher.get("confidence_min", 0.15)),
                    1.0,
                )
                confidence = (
                    confidence * edit_mask_np + (1.0 - edit_mask_np)
                )

                target_height = self.target_height
                target_width = self.target_width

                def resize_rgb(value: np.ndarray) -> np.ndarray:
                    if value.shape[:2] == (target_height, target_width):
                        return value.copy()
                    return cv2.resize(
                        value,
                        (target_width, target_height),
                        interpolation=cv2.INTER_AREA,
                    )

                def resize_mask(value: np.ndarray) -> np.ndarray:
                    if value.shape == (target_height, target_width):
                        return value.astype(np.float32, copy=True)
                    return cv2.resize(
                        value.astype(np.float32),
                        (target_width, target_height),
                        interpolation=cv2.INTER_AREA,
                    ).clip(0.0, 1.0)

                source = resize_rgb(source)
                sdxl_target = resize_rgb(sdxl_target).astype(
                    np.float32, copy=False
                )
                rv_raw = resize_rgb(rv_raw).astype(
                    np.float32, copy=False
                )
                target = resize_rgb(target).astype(
                    np.float32, copy=False
                )
                edit_mask_np = resize_mask(edit_mask_np)
                mouth_edit_np = resize_mask(mouth_edit_np)
                confidence = resize_mask(confidence)
                masks = {
                    name: resize_mask(value)
                    for name, value in masks.items()
                }
                mouth_source = "rv_adaptive"
                mouth_teacher = sdxl_target
                accepted_prior_metadata: Optional[Dict[str, Any]] = None
                if has_accepted_prior:
                    mouth_teacher, accepted_prior_metadata = (
                        self._load_accepted_mouth_prior(
                            override_key,
                            source,
                            mouth_edit_np,
                        )
                    )
                    prior_weight = mouth_edit_np[..., None].clip(0.0, 1.0)
                    target = np.clip(
                        target * (1.0 - prior_weight)
                        + mouth_teacher * prior_weight,
                        0.0,
                        1.0,
                    ).astype(np.float32, copy=False)
                    mouth_source = "accepted_prior"
                    applied_prior_keys.add(override_key)
                aperture_np = masks["aperture"]
                upper_hint = masks["upper_hint"]
                lower_hint = masks["lower_hint"]
                ear_hint = masks["ear_hint"]
                rv_mask_fraction = float(
                    (
                        masks["rv"] > 0.01
                    ).mean()
                    if should_rv51_refine
                    else 0.0
                )
                edit_mask_fraction = float((edit_mask_np > 0.01).mean())

                edit_mask_u8 = (
                    edit_mask_np * 255.0
                ).round().clip(0, 255).astype(np.uint8)
                outside_saved_mask = edit_mask_u8 == 0
                target_u8 = (
                    target * 255.0
                ).round().clip(0, 255).astype(np.uint8)
                # Saved PNGs are the actual training contract. Force exact
                # source identity wherever the saved mask is zero.
                target_u8[outside_saved_mask] = source[outside_saved_mask]
                target = target_u8.astype(np.float32) / 255.0
                metrics = self._target_quality(
                    pose_label=pose_label,
                    frame_index=int(frame.frame_index),
                    source_azimuth=float(frame.source_azimuth_deg),
                    source=source,
                    sdxl_target=sdxl_target,
                    mouth_teacher=mouth_teacher,
                    rv_raw=rv_raw,
                    target=target,
                    masks=masks,
                    should_mouth_inpaint=should_mouth_inpaint,
                    should_rv51_refine=should_rv51_refine,
                    edit_mask=edit_mask_np,
                    mouth_fallback_alpha=fallback_alpha,
                    mouth_source=mouth_source,
                )
                key = "%02d_%s_frame_%03d" % (
                    pose_index,
                    pose_label,
                    int(frame.frame_index),
                )
                paths = self._write_observation(
                    key=key,
                    source=source,
                    sdxl_target=sdxl_target,
                    mouth_teacher=mouth_teacher,
                    mouth_source=mouth_source,
                    rv_raw=rv_raw,
                    target=target,
                    mask=edit_mask_np,
                    mouth_mask=mouth_edit_np,
                    foreground_mask=masks["foreground"],
                    aperture=aperture_np,
                    confidence=confidence,
                    upper_hint=upper_hint,
                    lower_hint=lower_hint,
                    ear_hint=ear_hint,
                )
                entries.append(
                    {
                        "key": key,
                        "pose_id": pose_label,
                        "pose": pose_to_json(pose),
                        "camera_index": int(camera_index),
                        "frame_index": int(frame.frame_index),
                        "source_azimuth": float(frame.source_azimuth_deg),
                        "source_elevation": float(frame.source_elevation_deg),
                        "jaw_x": float(np.asarray(pose.jaw_pose)[0, 0]),
                        "inpainted": bool(should_mouth_inpaint),
                        "mouth_inpainted": bool(should_mouth_inpaint),
                        "rv51_refined": bool(should_rv51_refine),
                        "rv51_composite_scope": rv51_scope,
                        "edit_mask_fraction": edit_mask_fraction,
                        "rv_mask_fraction": rv_mask_fraction,
                        "sample_seeds": sample_seeds,
                        "mouth_candidate_selection_key": selection_key,
                        "mouth_candidate_index": (
                            selected_candidate_indices.get(selection_key, 0)
                        ),
                        "mouth_candidate_seeds": mouth_candidate_seeds,
                        "mouth_candidate_scores": mouth_candidate_scores,
                        "sample_teeth_scores": sample_teeth_scores,
                        "sample_fallback_alphas": sample_fallback_alphas,
                        "selected_sample": selected_sample,
                        "observation_key": override_key,
                        "mouth_source": mouth_source,
                        "accepted_mouth_prior_applied": bool(
                            accepted_prior_metadata is not None
                        ),
                        "accepted_mouth_prior": accepted_prior_metadata,
                        **metrics,
                        **paths,
                    }
                )
                progress.update(1)
        progress.close()

        unused_prior_keys = (
            set(self.accepted_mouth_priors) - applied_prior_keys
        )
        if unused_prior_keys:
            raise RuntimeError(
                "Configured accepted mouth priors were not used: %s"
                % ", ".join(sorted(unused_prior_keys))
            )

        front_open = [
            entry
            for entry in entries
            if bool(entry["inpainted"])
            and circular_distance_degrees(
                float(entry["source_azimuth"]), 270.0
            )
            <= 15.0
        ]
        if not front_open:
            raise RuntimeError(
                "Target bank has no inpainted front-view open-mouth target"
            )

        manifest = {
            "version": MANIFEST_VERSION,
            "signature": self.signature,
            "topology_sha256": self.avatar.topology_digest(),
            "bootstrap_stats": self.avatar.bootstrap_stats,
            "render_resolution": [render_height, render_width],
            "target_resolution": [
                self.target_height,
                self.target_width,
            ],
            "accepted_mouth_priors": {
                key: {
                    "configured_path": str(
                        teacher["accepted_mouth_priors"][key]
                    ),
                    "resolved_path": str(path),
                    "sha256": file_digest(path),
                    "source_path": str(
                        accepted_mouth_prior_source(path)
                    ),
                    "source_sha256": file_digest(
                        accepted_mouth_prior_source(path)
                    ),
                }
                for key, path in sorted(
                    self.accepted_mouth_priors.items()
                )
            },
            "entries": entries,
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
        self.logger.info(
            "Generated %d cached repair targets: %s",
            len(entries),
            manifest_path,
        )
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        if self.rv51_pipeline is not None:
            del self.rv51_pipeline
            self.rv51_pipeline = None
        torch.cuda.empty_cache()
        return manifest_path


# -----------------------------------------------------------------------------
# Cached observations and direct multi-view/multi-expression training
# -----------------------------------------------------------------------------


@dataclass
class CachedObservation:
    key: str
    pose_id: str
    pose: Any
    camera_index: int
    frame_index: int
    source: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor
    mouth_mask: torch.Tensor
    confidence: torch.Tensor


class ObservationBank:
    def __init__(
        self,
        cache_dir: Path,
        topology_digest_value: str,
        signature: str,
    ) -> None:
        self.cache_dir = cache_dir
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "Missing target manifest %s; run --mode generate-targets or all"
                % manifest_path
            )
        raw = manifest_path.read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw.decode("utf-8"))
        if int(manifest.get("version", -1)) != MANIFEST_VERSION:
            raise ValueError("Unsupported target manifest version")
        if manifest.get("signature") != signature:
            raise ValueError(
                "Target cache was generated from different input/configuration"
            )
        if manifest.get("topology_sha256") != topology_digest_value:
            raise ValueError(
                "Target cache topology differs from the bootstrapped avatar"
            )
        self.observations: List[CachedObservation] = []
        self.by_pose: Dict[str, List[CachedObservation]] = {}
        for entry in manifest["entries"]:
            observation = CachedObservation(
                key=str(entry["key"]),
                pose_id=str(entry["pose_id"]),
                pose=pose_from_json(entry["pose"]),
                camera_index=int(entry["camera_index"]),
                frame_index=int(entry["frame_index"]),
                source=self._rgb(entry["source"]),
                target=self._rgb(entry["target"]),
                mask=self._gray(entry["mask"]),
                mouth_mask=self._gray(entry["mouth_mask"]),
                confidence=self._gray(entry["confidence"]),
            )
            self.observations.append(observation)
            self.by_pose.setdefault(observation.pose_id, []).append(observation)
        if not self.observations:
            raise ValueError("Target manifest contains no observations")
        self.pose_ids = sorted(self.by_pose)

    def _rgb(self, relative: str) -> torch.Tensor:
        with Image.open(self.cache_dir / relative) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
        return torch.from_numpy(array).permute(2, 0, 1).div_(255.0)

    def _gray(self, relative: str) -> torch.Tensor:
        with Image.open(self.cache_dir / relative) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32).copy()
        return torch.from_numpy(array)[None].div_(255.0)

    def sample(
        self, batch_size: int
    ) -> Tuple[Any, List[int], List[CachedObservation]]:
        pose_id = random.choice(self.pose_ids)
        group = self.by_pose[pose_id]
        count = min(int(batch_size), len(group))
        observations = random.sample(group, count)
        return (
            observations[0].pose,
            [item.camera_index for item in observations],
            observations,
        )

    def preview_group(self) -> List[CachedObservation]:
        if "validation" in self.by_pose:
            return self.by_pose["validation"]
        return max(
            self.by_pose.values(),
            key=lambda values: float(
                np.asarray(values[0].pose.jaw_pose)[0, 0]
            ),
        )


class DirectInpaintTrainer:
    def __init__(
        self,
        config: Mapping[str, Any],
        directory: Path,
        resume_path: Optional[Path],
    ) -> None:
        self.config = config
        self.directory = directory
        self.device = torch.device(str(config.get("device", "cuda")))
        self.logger = setup_logger(directory)
        # MediaPipe/FLAME control images are required only while building the
        # RV5.1 teacher bank. Direct inverse rendering never consumes them, so
        # disable their per-batch construction during the 3,000-step fit.
        training_data_config = copy.deepcopy(config)
        training_data_config["data"]["use_mediapipe_condition"] = False
        self.data_module = create_data_module(training_data_config)
        if self.data_module.assets is None or self.data_module.builder is None:
            raise RuntimeError("Reconstruction fine-tune data was not initialized")
        self.assets = self.data_module.assets
        self.builder = self.data_module.builder
        teacher = config["teacher"]
        self.render_height = int(
            teacher.get(
                "target_height",
                teacher.get("render_height", config["data"]["height"]),
            )
        )
        self.render_width = int(
            teacher.get(
                "target_width",
                teacher.get("render_width", config["data"]["width"]),
            )
        )
        self.avatar = UVDAvatar(
            resolve_path(config["input"]["reconstruction_dir"]),
            config["bootstrap"],
            config["optimization"],
            self.device,
        )
        self.bank = ObservationBank(
            target_directory(config, directory),
            self.avatar.topology_digest(),
            target_signature(config),
        )
        cached_resolution = tuple(self.bank.observations[0].target.shape[-2:])
        expected_resolution = (self.render_height, self.render_width)
        if cached_resolution != expected_resolution:
            raise ValueError(
                "Cached target resolution %s differs from configured training "
                "resolution %s" % (cached_resolution, expected_resolution)
            )
        self.config_digest = stable_digest(config)
        self.start_step = 0
        self.geometry_steps = int(
            config["optimization"]["geometry_iterations"]
        )
        self.total_steps = self.geometry_steps + int(
            config["optimization"]["appearance_iterations"]
        )
        self.optimizer = self._build_optimizer()
        self.gradient_masks = {
            self.avatar.gaussian._d: self.avatar.geometry_mask,
            self.avatar.gaussian._scaling: self.avatar.geometry_mask,
            self.avatar.gaussian._rotation: self.avatar.geometry_mask,
            self.avatar.gaussian._features_dc: self.avatar.appearance_mask,
            self.avatar.gaussian._opacity: self.avatar.opacity_mask,
        }
        self.uv_edges = self._build_uv_edges(
            int(config["loss"].get("uv_neighbors", 6))
        )

        from gaussiansplatting.utils.loss_utils import ssim

        self.ssim = ssim
        self.perceptual = None
        if float(config["loss"].get("perceptual_weight", 0.0)) > 0.0:
            try:
                from threestudio.utils.perceptual.vgg_feature import (
                    VGGPerceptualLoss,
                )

                self.perceptual = VGGPerceptualLoss().to(self.device).eval()
            except Exception as error:
                self.logger.warning(
                    "Perceptual loss disabled because VGG could not load: %s",
                    error,
                )

        self.preview_dir = directory / "previews"
        self.checkpoint_dir = directory / "checkpoints"
        self.model_dir = directory / "model"
        for path in (self.preview_dir, self.checkpoint_dir, self.model_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.metrics_file = (directory / "metrics.jsonl").open(
            "a", encoding="utf-8"
        )
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer: Any = SummaryWriter(log_dir=str(directory / "logs"))
        except Exception as error:
            self.logger.warning("TensorBoard disabled: %s", error)
            self.writer = None
        if resume_path is not None:
            self.load_checkpoint(resume_path)

        self.logger.info(
            "Direct repair training: %d Gaussians, color=%d, opacity=%d, "
            "geometry=%d, ears=%d, cavity=%d, teeth=%d, "
            "cached observations=%d",
            int(self.avatar.gaussian.num_gs),
            int(self.avatar.appearance_mask.sum().item()),
            int(self.avatar.opacity_mask.sum().item()),
            int(self.avatar.geometry_mask.sum().item()),
            int(self.avatar.region_masks["ears"].sum().item()),
            int(self.avatar.region_masks["cavity"].sum().item()),
            int(self.avatar.region_masks["teeth"].sum().item()),
            len(self.bank.observations),
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        config = self.config["optimization"]
        groups = [
            {
                "params": [self.avatar.gaussian._features_dc],
                "lr": float(config["feature_lr"]),
                "base_lr": float(config["feature_lr"]),
                "name": "feature",
            },
            {
                "params": [self.avatar.gaussian._opacity],
                "lr": float(config["opacity_lr"]),
                "base_lr": float(config["opacity_lr"]),
                "name": "opacity",
            },
            {
                "params": [self.avatar.gaussian._d],
                "lr": float(config["d_lr"]),
                "base_lr": float(config["d_lr"]),
                "name": "d",
            },
            {
                "params": [self.avatar.gaussian._scaling],
                "lr": float(config["scale_lr"]),
                "base_lr": float(config["scale_lr"]),
                "name": "scale",
            },
            {
                "params": [self.avatar.gaussian._rotation],
                "lr": float(config["rotation_lr"]),
                "base_lr": float(config["rotation_lr"]),
                "name": "rotation",
            },
        ]
        return torch.optim.Adam(
            groups,
            betas=tuple(float(v) for v in config["betas"]),
            eps=float(config["eps"]),
        )

    def _build_uv_edges(self, neighbors: int) -> torch.Tensor:
        pairs = []
        uv = self.avatar.gaussian._uv.detach()
        for name in self.avatar.geometry_region_names:
            indices = torch.nonzero(
                self.avatar.region_mask(name), as_tuple=False
            ).squeeze(1)
            if indices.numel() < 2:
                continue
            count = min(max(int(neighbors), 1), int(indices.numel()) - 1)
            distances = torch.cdist(uv[indices], uv[indices])
            nearest = distances.topk(count + 1, largest=False).indices[:, 1:]
            source = indices[:, None].expand_as(nearest).reshape(-1)
            target = indices[nearest.reshape(-1)]
            pairs.append(torch.stack((source, target), dim=1))
        if not pairs:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
        return torch.cat(pairs, dim=0)

    def _set_phase_lrs(self, step: int) -> str:
        geometry = step < self.geometry_steps
        phase = "geometry" if geometry else "appearance"
        for group in self.optimizer.param_groups:
            if group["name"] in {"d", "scale", "rotation"} and not geometry:
                group["lr"] = 0.0
            else:
                group["lr"] = float(group["base_lr"])
        return phase

    def _batch(
        self,
    ) -> Tuple[
        Dict[str, Any],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        pose, camera_indices, observations = self.bank.sample(
            int(self.config["optimization"]["batch_size"])
        )
        batch = rescale_render_batch(
            self.builder.build(camera_indices, pose),
            self.render_height,
            self.render_width,
        )
        batch = to_device(batch, self.device)
        target = torch.stack([item.target for item in observations]).to(
            self.device
        )
        mask = torch.stack([item.mask for item in observations]).to(self.device)
        mouth_mask = torch.stack(
            [item.mouth_mask for item in observations]
        ).to(self.device)
        confidence = torch.stack(
            [item.confidence for item in observations]
        ).to(self.device)
        return batch, target, mask, mouth_mask, confidence

    def _geometry_regularizers(
        self, rendering: RenderBatch
    ) -> Dict[str, torch.Tensor]:
        gaussian = self.avatar.gaussian
        mask = self.avatar.geometry_mask
        appearance_mask = self.avatar.appearance_mask
        opacity_mask = self.avatar.opacity_mask
        initial = self.avatar.initial
        d_prox = F.smooth_l1_loss(gaussian._d[mask], initial["d"][mask])
        scale_prox = F.smooth_l1_loss(
            gaussian._scaling[mask], initial["scale"][mask]
        )
        feature_prox = F.smooth_l1_loss(
            gaussian._features_dc[appearance_mask],
            initial["feature_dc"][appearance_mask],
        )
        opacity_prox = F.smooth_l1_loss(
            gaussian._opacity[opacity_mask],
            initial["opacity"][opacity_mask],
        )
        current_rotation = F.normalize(gaussian._rotation[mask], dim=-1)
        initial_rotation = F.normalize(initial["rotation"][mask], dim=-1)
        rotation_prox = (
            1.0
            - (current_rotation * initial_rotation)
            .sum(dim=-1)
            .abs()
            .clamp(max=1.0)
        ).mean()
        if self.uv_edges.numel() > 0:
            source, target = self.uv_edges.unbind(dim=1)
            uv_laplacian = F.smooth_l1_loss(
                gaussian._d[source], gaussian._d[target]
            )
        else:
            uv_laplacian = torch.zeros((), device=self.device)
        maximum = float(self.config["optimization"]["max_world_scale"])
        scale_barrier = F.relu(
            rendering.world_scale[mask] - maximum
        ).square().mean()
        return {
            "d_prox": d_prox,
            "scale_prox": scale_prox,
            "feature_prox": feature_prox,
            "opacity_prox": opacity_prox,
            "rotation_prox": rotation_prox,
            "uv_laplacian": uv_laplacian,
            "scale_barrier": scale_barrier,
        }

    def _losses(
        self,
        rendering: RenderBatch,
        target: torch.Tensor,
        edit_mask: torch.Tensor,
        mouth_mask: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        config = self.config["loss"]
        prediction = rendering.rgb
        weight = (edit_mask * confidence).clamp(0.0, 1.0)
        photo = masked_l1(prediction, target, weight)
        composite_prediction = (
            prediction * weight + target * (1.0 - weight)
        )
        ssim_loss = 1.0 - self.ssim(composite_prediction, target)

        perceptual = torch.zeros((), device=self.device)
        mouth_rows = mouth_mask.flatten(1).amax(dim=1) > 0.0
        if self.perceptual is not None and bool(mouth_rows.any()):
            prediction_crop = tensor_mouth_crops(
                composite_prediction[mouth_rows], mouth_mask[mouth_rows]
            )
            target_crop = tensor_mouth_crops(
                target[mouth_rows], mouth_mask[mouth_rows]
            )
            perceptual = self.perceptual(
                prediction_crop, target_crop
            )

        outside = (1.0 - edit_mask).clamp(0.0, 1.0)
        identity = masked_l1(prediction, target, outside)

        regularizers = self._geometry_regularizers(rendering)
        total = (
            float(config["photo_weight"]) * photo
            + float(config["ssim_weight"]) * ssim_loss
            + float(config["perceptual_weight"]) * perceptual
            + float(config["identity_weight"]) * identity
            + float(config["d_proximal_weight"]) * regularizers["d_prox"]
            + float(config["scale_proximal_weight"])
            * regularizers["scale_prox"]
            + float(config["feature_proximal_weight"])
            * regularizers["feature_prox"]
            + float(config["opacity_proximal_weight"])
            * regularizers["opacity_prox"]
            + float(config["rotation_proximal_weight"])
            * regularizers["rotation_prox"]
            + float(config["uv_laplacian_weight"])
            * regularizers["uv_laplacian"]
            + float(config["scale_barrier_weight"])
            * regularizers["scale_barrier"]
        )
        losses = {
            "total": total,
            "photo": photo,
            "ssim": ssim_loss,
            "perceptual": perceptual,
            "identity": identity,
            **regularizers,
        }
        diagnostics = {
            "target": target,
            "edit_mask": edit_mask,
        }
        return total, losses, diagnostics

    def _mask_gradients(self) -> None:
        for parameter, mask in self.gradient_masks.items():
            if parameter.grad is None:
                continue
            shape = (mask.shape[0],) + (1,) * (parameter.grad.ndim - 1)
            parameter.grad.mul_(mask.reshape(shape).to(parameter.grad.dtype))

    def _log(
        self,
        step: int,
        phase: str,
        losses: Mapping[str, torch.Tensor],
        diagnostics: Mapping[str, torch.Tensor],
    ) -> None:
        record: Dict[str, Any] = {
            "step": int(step),
            "phase": phase,
            "num_gaussians": int(self.avatar.gaussian.num_gs),
            "edit_mask_mean": float(
                diagnostics["edit_mask"].mean().item()
            ),
        }
        record.update(
            {name: float(value.detach().item()) for name, value in losses.items()}
        )
        self.metrics_file.write(json.dumps(record) + "\n")
        self.metrics_file.flush()
        if self.writer is not None:
            for name, value in losses.items():
                self.writer.add_scalar(
                    "train/%s" % name, float(value.detach().item()), step
                )
            self.writer.add_scalar(
                "train/phase", 0 if phase == "geometry" else 1, step
            )
            for group in self.optimizer.param_groups:
                self.writer.add_scalar(
                    "train/lr_%s" % group["name"], group["lr"], step
                )

    @torch.inference_mode()
    def save_preview(self, step: int) -> None:
        observations = self.bank.preview_group()
        pose = observations[0].pose
        camera_indices = [item.camera_index for item in observations]
        batch = rescale_render_batch(
            self.builder.build(camera_indices, pose),
            self.render_height,
            self.render_width,
        )
        batch = to_device(batch, self.device)
        rendering = self.avatar.render_batch(
            batch,
            differentiable=False,
        )
        rows = []
        for index, observation in enumerate(observations):
            target = observation.target
            mask = observation.mask.expand(3, -1, -1)
            difference = (
                rendering.rgb[index] - target.to(self.device)
            ).abs()
            rows.append(
                np.concatenate(
                    (
                        add_label(
                            rgb_u8(observation.source),
                            "bootstrapped source",
                        ),
                        add_label(
                            rgb_u8(rendering.rgb[index]), "current UVD"
                        ),
                        add_label(rgb_u8(target), "cached teacher"),
                        add_label(rgb_u8(mask), "training edit mask"),
                        add_label(rgb_u8(difference), "absolute error"),
                    ),
                    axis=1,
                )
            )
        Image.fromarray(np.concatenate(rows, axis=0)).save(
            self.preview_dir / ("step_%06d.jpg" % step), quality=92
        )

    @torch.inference_mode()
    def render_test_sequence(self) -> Path:
        """Render assets/test exp+pose with the final trained UVD avatar."""

        import imageio.v2 as imageio

        config = self.config["test_render"]
        output_dir = self.directory / "test_render"
        frame_dir = output_dir / "frames"
        mask_dir = output_dir / "masks"
        frame_dir.mkdir(parents=True, exist_ok=True)
        save_masks = bool(config.get("save_masks", True))
        if save_masks:
            mask_dir.mkdir(parents=True, exist_ok=True)

        height = int(
            config.get("height", self.config["data"]["eval_height"])
        )
        width = int(
            config.get("width", self.config["data"]["eval_width"])
        )
        fps = int(config.get("fps", 30))
        frame_count = int(self.assets.test_frame_count)
        camera_index = int(self.assets.test_camera_index)
        expression_path = resolve_path(
            self.config["data"]["test_expression_path"]
        )
        pose_path = resolve_path(self.config["data"]["test_pose_path"])
        video_path = output_dir / "test.mp4"
        frame_paths: List[Path] = []
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
                batch = to_device(batch, self.device)
                rendering = self.avatar.render_batch(
                    batch,
                    differentiable=False,
                )
                image = rgb_u8(rendering.rgb[0])
                frame_path = frame_dir / ("%06d.png" % index)
                Image.fromarray(image).save(frame_path)
                frame_paths.append(frame_path)
                writer.append_data(image)
                if save_masks:
                    Image.fromarray(
                        gray_u8(rendering.alpha[0])
                    ).save(mask_dir / ("%06d.png" % index))
        finally:
            writer.close()
            self.avatar.set_pose(*self.avatar.reference_pose)

        sample_count = min(
            int(config.get("contact_sheet_frames", 12)),
            frame_count,
        )
        selected = np.unique(
            np.linspace(0, frame_count - 1, sample_count).round().astype(int)
        )
        tiles: List[np.ndarray] = []
        for index in selected:
            with Image.open(frame_paths[int(index)]) as image_file:
                tile = np.asarray(
                    image_file.convert("RGB").resize(
                        (256, 256), Image.Resampling.LANCZOS
                    ),
                    dtype=np.uint8,
                )
            tiles.append(add_label(tile, "test frame %03d" % int(index)))
        columns = 4
        rows = int(math.ceil(len(tiles) / float(columns)))
        sheet = np.full(
            (rows * 256, columns * 256, 3),
            255,
            dtype=np.uint8,
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
            "frames": "frames",
            "masks": "masks" if save_masks else None,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.logger.info(
            "Rendered %d assets/test frames at camera %03d: %s",
            frame_count,
            metadata["camera_frame_index"],
            video_path,
        )
        return video_path

    def checkpoint_state(self, completed_steps: int) -> Dict[str, Any]:
        return {
            "version": CHECKPOINT_VERSION,
            "completed_steps": int(completed_steps),
            "config_sha256": self.config_digest,
            "target_bank_sha256": self.bank.digest,
            "topology_sha256": self.avatar.topology_digest(),
            "model": self.avatar.model_state(),
            "optimizer": self.optimizer.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
        }

    @staticmethod
    def _atomic_save(state: Mapping[str, Any], path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(dict(state), temporary)
        os.replace(temporary, path)

    def save_checkpoint(self, completed_steps: int) -> Path:
        state = self.checkpoint_state(completed_steps)
        path = self.checkpoint_dir / ("step_%06d.pt" % completed_steps)
        self._atomic_save(state, path)
        self._atomic_save(state, self.checkpoint_dir / "latest.pt")
        self.logger.info("Saved checkpoint: %s", path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        if int(state.get("version", -1)) != CHECKPOINT_VERSION:
            raise ValueError("Unsupported direct-inpaint checkpoint version")
        if state.get("config_sha256") != self.config_digest:
            raise ValueError("Checkpoint configuration differs")
        if state.get("target_bank_sha256") != self.bank.digest:
            raise ValueError("Checkpoint target cache differs")
        if state.get("topology_sha256") != self.avatar.topology_digest():
            raise ValueError("Checkpoint topology differs")
        self.avatar.load_model_state(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)
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
        self.avatar.restore_frozen_rows()
        self.avatar.assert_invariants()
        self.logger.info("Resumed %s at step %d", path, self.start_step)

    def close(self) -> None:
        self.metrics_file.flush()
        self.metrics_file.close()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def train(self) -> None:
        optimization = self.config["optimization"]
        log_interval = max(int(self.config["output"]["log_interval"]), 1)
        preview_interval = max(
            int(self.config["output"]["preview_interval"]), 1
        )
        checkpoint_interval = max(
            int(self.config["checkpoint"]["interval"]), 1
        )
        progress = tqdm(
            range(self.start_step, self.total_steps),
            desc="Direct artifact repair",
            dynamic_ncols=True,
        )
        for step in progress:
            phase = self._set_phase_lrs(step)
            batch, target, mask, mouth_mask, confidence = self._batch()
            self.optimizer.zero_grad(set_to_none=True)
            rendering = self.avatar.render_batch(
                batch,
                differentiable=True,
            )
            total, losses, diagnostics = self._losses(
                rendering,
                target,
                mask,
                mouth_mask,
                confidence,
            )
            if not torch.isfinite(total):
                raise FloatingPointError(
                    "Non-finite loss at step %d: %s"
                    % (step, float(total.detach()))
                )
            total.backward()
            self._mask_gradients()
            maximum = float(optimization["max_grad_norm"])
            if maximum > 0.0:
                parameters = [
                    parameter
                    for group in self.optimizer.param_groups
                    for parameter in group["params"]
                ]
                torch.nn.utils.clip_grad_norm_(parameters, maximum)
            self.optimizer.step()
            self.avatar.restore_frozen_rows()
            self.avatar.clamp_editable_geometry(
                float(optimization["max_abs_d"]),
                float(optimization["max_world_scale"]),
            )
            self.avatar.assert_invariants()
            completed = step + 1

            if completed % log_interval == 0 or step == self.start_step:
                self._log(completed, phase, losses, diagnostics)
            progress.set_postfix(
                phase=phase,
                loss="%.4f" % float(total.detach()),
                photo="%.4f" % float(losses["photo"].detach()),
            )
            if completed % preview_interval == 0:
                self.save_preview(completed)
            if completed % checkpoint_interval == 0:
                self.save_checkpoint(completed)

        self.save_preview(self.total_steps)
        if self.total_steps % checkpoint_interval != 0:
            self.save_checkpoint(self.total_steps)
        self.avatar.save(self.model_dir)
        if bool(self.config["test_render"].get("enabled", True)):
            self.render_test_sequence()
        self.logger.info("Direct artifact repair complete: %s", self.directory)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/direct_loop_inpaint_targets.yaml"),
        help="Standalone Stage-2 YAML configuration",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "generate-targets", "train"),
        default="all",
        help="Generate cached teachers, train from cache, or run both",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume a direct-training checkpoint",
    )
    parser.add_argument(
        "--overwrite-targets",
        action="store_true",
        help="Regenerate an existing teacher cache",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate files/configuration without loading CUDA",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional dotted overrides, e.g. optimization.geometry_iterations=10",
    )
    return parser.parse_args()


def write_resolved_config(
    config: Mapping[str, Any], directory: Path
) -> None:
    resolved = copy.deepcopy(dict(config))
    resolved["input"]["reconstruction_dir"] = str(
        resolve_path(config["input"]["reconstruction_dir"])
    )
    resolved["data"]["chemistry_path"] = str(
        resolve_path(config["data"]["chemistry_path"])
    )
    resolved["teacher"]["model_path"] = str(
        resolve_path(config["teacher"]["model_path"])
    )
    for key, path in accepted_mouth_prior_paths(config).items():
        resolved["teacher"]["accepted_mouth_priors"][key] = str(path)
    rv51 = resolved["teacher"].get("rv51", {})
    if bool(rv51.get("enabled", False)):
        for key in ("model_path", "controlnet_path", "scheduler_path"):
            rv51[key] = str(resolve_path(config["teacher"]["rv51"][key]))
    resolved["resolved_output_dir"] = str(directory)
    resolved["resolved_target_cache"] = str(target_directory(config, directory))
    (directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path, args.overrides)
    validate_config(config, check_files=True)
    directory = output_directory(config)
    if args.validate_only:
        print("Configuration is valid:", config_path)
        print("Stage-1 input:", resolve_path(config["input"]["reconstruction_dir"]))
        print("Output:", directory)
        print("Target cache:", target_directory(config, directory))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Gaussian rendering and inpainting")

    seed_everything(int(config.get("seed", 0)))
    directory.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, directory)
    if args.mode in {"all", "generate-targets"}:
        generator = MouthTargetGenerator(
            config, directory, overwrite=bool(args.overwrite_targets)
        )
        generator.generate()
        del generator
        torch.cuda.empty_cache()
    if args.mode in {"all", "train"}:
        resume = resolve_path(args.resume) if args.resume else None
        trainer = DirectInpaintTrainer(config, directory, resume)
        try:
            trainer.train()
        finally:
            trainer.close()


if __name__ == "__main__":
    main()
