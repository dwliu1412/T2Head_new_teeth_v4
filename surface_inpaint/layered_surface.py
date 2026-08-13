"""Renderer-independent semantic surface correspondence utilities.

The Gaussian rasterizer can provide one UV/depth/alpha buffer per FLAME
semantic layer.  Keeping those buffers separate is essential for the oral
surfaces: upper and lower teeth deliberately occupy overlapping canonical UV
coordinates, but they must remain different correspondence keys.

This module contains only tensor validation and composition.  It has no CUDA,
renderer, diffusion, or configuration dependency, so its selection behavior
can be covered by CPU unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

import torch


INVALID_SURFACE_LAYER_ID = -1


class SurfaceLayerId(IntEnum):
    """Stable semantic IDs used by UV correspondence and attention."""

    FACE = 0
    LIPS = 1
    TEETH_UPPER = 2
    TEETH_LOWER = 3
    ORAL_CAVITY = 4


@dataclass(frozen=True)
class SurfaceLayerDefinition:
    """Describe one mutually exclusive FLAME/Gaussian semantic layer."""

    name: str
    layer_id: int
    flame_region: str
    # Regions removed from this layer after face-index classification.
    # Only the broad face region needs exclusions; the four oral regions are
    # already disjoint in the augmented FLAME topology.
    exclude_regions: Tuple[str, ...] = ()


SURFACE_LAYER_DEFINITIONS: Tuple[SurfaceLayerDefinition, ...] = (
    SurfaceLayerDefinition(
        name="face",
        layer_id=int(SurfaceLayerId.FACE),
        flame_region="face",
        exclude_regions=(
            "lips",
            "teeth_upper",
            "teeth_lower",
            "oral_cavity",
        ),
    ),
    SurfaceLayerDefinition(
        name="lips",
        layer_id=int(SurfaceLayerId.LIPS),
        flame_region="lips",
    ),
    SurfaceLayerDefinition(
        name="teeth_upper",
        layer_id=int(SurfaceLayerId.TEETH_UPPER),
        flame_region="teeth_upper",
    ),
    SurfaceLayerDefinition(
        name="teeth_lower",
        layer_id=int(SurfaceLayerId.TEETH_LOWER),
        flame_region="teeth_lower",
    ),
    SurfaceLayerDefinition(
        name="oral_cavity",
        layer_id=int(SurfaceLayerId.ORAL_CAVITY),
        flame_region="oral_cavity",
    ),
)

SURFACE_LAYER_NAMES: Tuple[str, ...] = tuple(
    definition.name for definition in SURFACE_LAYER_DEFINITIONS
)
SURFACE_LAYER_IDS = MappingProxyType(
    {
        definition.name: definition.layer_id
        for definition in SURFACE_LAYER_DEFINITIONS
    }
)
SURFACE_LAYER_BY_NAME = MappingProxyType(
    {
        definition.name: definition
        for definition in SURFACE_LAYER_DEFINITIONS
    }
)


def validate_layer_names(
    names: Sequence[str],
    *,
    require_all: bool = False,
) -> Tuple[str, ...]:
    """Validate names and return them in stable semantic-ID order."""

    if isinstance(names, (str, bytes)):
        raise TypeError("Layer names must be a sequence, not a string")
    resolved = tuple(names)
    if not resolved:
        raise ValueError("At least one surface layer is required")
    for name in resolved:
        if not isinstance(name, str):
            raise TypeError("Every surface layer name must be a string")
        if name not in SURFACE_LAYER_BY_NAME:
            expected = ", ".join(SURFACE_LAYER_NAMES)
            raise ValueError(
                f"Unknown surface layer {name!r}; expected one of: {expected}"
            )
    if len(set(resolved)) != len(resolved):
        raise ValueError("Surface layer names must be unique")
    if require_all and set(resolved) != set(SURFACE_LAYER_NAMES):
        missing = [
            name for name in SURFACE_LAYER_NAMES if name not in resolved
        ]
        raise ValueError(
            "All fixed surface layers are required; missing: "
            + ", ".join(missing)
        )
    return tuple(
        sorted(resolved, key=lambda name: SURFACE_LAYER_IDS[name])
    )


@dataclass(frozen=True)
class LayerSurfaceBuffers:
    """Normalized raster buffers for one semantic layer.

    ``alpha`` is the layer-only composited alpha. ``contribution`` is the
    occlusion-aware contribution obtained while all scene Gaussians still
    participate in front-to-back compositing.  Therefore contribution, rather
    than layer-only alpha, is the dominant-layer score.
    """

    uv: torch.Tensor
    variance: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor
    contribution: torch.Tensor


@dataclass(frozen=True)
class LayeredSurface:
    """A single, ambiguity-rejected semantic surface per screen pixel."""

    surface_uv: torch.Tensor
    surface_variance: torch.Tensor
    surface_depth: torch.Tensor
    surface_alpha: torch.Tensor
    surface_contribution: torch.Tensor
    layer_id: torch.Tensor
    validity: torch.Tensor
    ambiguous: torch.Tensor


def normalize_alpha_weighted(
    premultiplied: torch.Tensor,
    alpha: torch.Tensor,
    *,
    alpha_epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Safely normalize an alpha-weighted render.

    Pixels at or below ``alpha_epsilon`` are returned as exact zeros.  This
    avoids turning tiny rasterizer tails into extreme UV or depth values and
    makes the zero/threshold behavior explicit for callers and tests.
    """

    if not torch.is_tensor(premultiplied) or not torch.is_tensor(alpha):
        raise TypeError("premultiplied and alpha must be torch tensors")
    if premultiplied.ndim < 2 or alpha.ndim != premultiplied.ndim:
        raise ValueError(
            "premultiplied and alpha must have matching tensor ranks"
        )
    if alpha.shape[0] != premultiplied.shape[0]:
        raise ValueError("premultiplied and alpha batch sizes must match")
    if alpha.shape[1] != 1:
        raise ValueError("alpha must have exactly one channel")
    if alpha.shape[2:] != premultiplied.shape[2:]:
        raise ValueError("premultiplied and alpha spatial shapes must match")
    if premultiplied.device != alpha.device:
        raise ValueError("premultiplied and alpha must be on the same device")
    epsilon = float(alpha_epsilon)
    if not 0.0 < epsilon < 1.0:
        raise ValueError("alpha_epsilon must be in (0, 1)")

    finite_alpha = torch.isfinite(alpha)
    positive = finite_alpha & (alpha > epsilon)
    safe_alpha = torch.where(
        positive,
        alpha,
        torch.ones_like(alpha),
    )
    safe_value = torch.nan_to_num(
        premultiplied,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    normalized = safe_value / safe_alpha
    return torch.where(
        positive.expand_as(normalized),
        normalized,
        torch.zeros_like(normalized),
    )


def _validate_rgb_pair(
    first_name: str,
    first: torch.Tensor,
    second_name: str,
    second: torch.Tensor,
) -> None:
    for name, value in ((first_name, first), (second_name, second)):
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch tensor")
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} must use a floating dtype")
        channels_are_rgb = (
            value.ndim == 2
            and value.shape[-1] == 3
            or value.ndim >= 3
            and value.shape[-3] == 3
        )
        if not channels_are_rgb:
            raise ValueError(f"{name} must have an RGB channel dimension")
    if first.shape != second.shape:
        raise ValueError(
            f"{first_name} and {second_name} must have identical shapes"
        )
    if first.device != second.device:
        raise ValueError(
            f"{first_name} and {second_name} must be on the same device"
        )


def encode_surface_rgb_residual(
    teacher_rgb: torch.Tensor,
    current_rgb: torch.Tensor,
    surface_contribution: torch.Tensor,
    *,
    contribution_floor: float,
) -> torch.Tensor:
    """Encode a de-composited semantic-surface residual into ``[0, 1]``.

    Storing a full composited teacher pixel in a semantic teeth atlas would
    also store the contribution of lips and background.  A residual cancels
    pixels that the teacher did not change.  Dividing by the layer's
    occlusion-aware contribution converts a screen-space delta into the
    approximate intrinsic color delta required from that semantic surface.
    The result is later applied to a per-Gaussian appearance snapshot.
    """

    _validate_rgb_pair(
        "teacher_rgb",
        teacher_rgb,
        "current_rgb",
        current_rgb,
    )
    if teacher_rgb.ndim != 4:
        raise ValueError(
            "teacher_rgb and current_rgb must have shape B x 3 x H x W"
        )
    if not torch.is_tensor(surface_contribution):
        raise TypeError("surface_contribution must be a torch tensor")
    if not torch.is_floating_point(surface_contribution):
        raise TypeError("surface_contribution must use a floating dtype")
    if surface_contribution.shape != (
        teacher_rgb.shape[0],
        1,
        teacher_rgb.shape[2],
        teacher_rgb.shape[3],
    ):
        raise ValueError(
            "surface_contribution must have shape B x 1 x H x W "
            "matching the RGB tensors"
        )
    if surface_contribution.device != teacher_rgb.device:
        raise ValueError(
            "surface_contribution and RGB tensors must be on the same device"
        )
    floor = float(contribution_floor)
    if not torch.isfinite(torch.tensor(floor)).item() or not 0.0 < floor <= 1.0:
        raise ValueError("contribution_floor must be finite and in (0, 1]")
    teacher = torch.nan_to_num(
        teacher_rgb,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    current = torch.nan_to_num(
        current_rgb,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    contribution = torch.nan_to_num(
        surface_contribution,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    intrinsic_residual = (teacher - current) / contribution.clamp_min(floor)
    return (0.5 + 0.5 * intrinsic_residual).clamp(0.0, 1.0)


def decode_surface_rgb_residual(
    encoded_residual: torch.Tensor,
    reference_rgb: torch.Tensor,
) -> torch.Tensor:
    """Apply an encoded residual to its matching RGB reference snapshot."""

    _validate_rgb_pair(
        "encoded_residual",
        encoded_residual,
        "reference_rgb",
        reference_rgb,
    )
    encoded = torch.nan_to_num(
        encoded_residual,
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    reference = torch.nan_to_num(
        reference_rgb,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    return (reference + 2.0 * (encoded - 0.5)).clamp(0.0, 1.0)


def _validate_scalar(name: str, value: float, minimum: float) -> float:
    resolved = float(value)
    if not torch.isfinite(torch.tensor(resolved)).item():
        raise ValueError(f"{name} must be finite")
    if resolved < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return resolved


def _validate_layer_buffers(
    layer_name: str,
    buffers: LayerSurfaceBuffers,
    reference_shape: Optional[Tuple[int, int, int]],
    reference_device: Optional[torch.device],
) -> Tuple[Tuple[int, int, int], torch.device]:
    if not isinstance(buffers, LayerSurfaceBuffers):
        raise TypeError(
            f"Layer {layer_name!r} must contain LayerSurfaceBuffers"
        )
    fields = {
        "uv": buffers.uv,
        "variance": buffers.variance,
        "depth": buffers.depth,
        "alpha": buffers.alpha,
        "contribution": buffers.contribution,
    }
    for field_name, value in fields.items():
        if not torch.is_tensor(value):
            raise TypeError(
                f"{layer_name}.{field_name} must be a torch tensor"
            )
        if not torch.is_floating_point(value):
            raise TypeError(
                f"{layer_name}.{field_name} must use a floating dtype"
            )
    uv = buffers.uv
    if uv.ndim != 4 or uv.shape[1] != 2:
        raise ValueError(f"{layer_name}.uv must have shape [B,2,H,W]")
    shape = (int(uv.shape[0]), int(uv.shape[2]), int(uv.shape[3]))
    for field_name in ("variance", "depth", "alpha", "contribution"):
        value = fields[field_name]
        if value.ndim != 4 or value.shape[1] != 1:
            raise ValueError(
                f"{layer_name}.{field_name} must have shape [B,1,H,W]"
            )
        if (
            int(value.shape[0]),
            int(value.shape[2]),
            int(value.shape[3]),
        ) != shape:
            raise ValueError(
                f"{layer_name}.{field_name} must match its UV batch/spatial shape"
            )
    devices = {value.device for value in fields.values()}
    if len(devices) != 1:
        raise ValueError(
            f"All {layer_name!r} buffers must be on the same device"
        )
    device = uv.device
    if reference_shape is not None and shape != reference_shape:
        raise ValueError("All surface layers must share one B,H,W shape")
    if reference_device is not None and device != reference_device:
        raise ValueError("All surface layers must be on the same device")
    return shape, device


def compose_layered_surface(
    layers: Mapping[str, LayerSurfaceBuffers],
    *,
    alpha_threshold: float = 0.0,
    contribution_threshold: float = 0.0,
    variance_threshold: float = float("inf"),
    depth_tolerance: Optional[float] = None,
    dominance_ratio: float = 1.25,
    dominance_margin: float = 0.0,
) -> LayeredSurface:
    """Select one visible semantic layer and reject ambiguous pixels.

    Selection proceeds as follows:

    1. reject invalid UV/depth/variance and weak alpha/contribution;
    2. optionally retain candidates within ``depth_tolerance`` of the nearest
       normalized expected depth;
    3. rank the remaining candidates by occlusion-aware contribution;
    4. reject pixels whose first/second scores fail either dominance test.

    ``depth_tolerance=None`` disables the expected-depth gate.  This is useful
    when the rasterizer exposes alpha-composited expected depth rather than a
    strict z-buffer; occlusion-aware contribution still remains active.
    """

    if not isinstance(layers, Mapping):
        raise TypeError("layers must be a mapping")
    names = validate_layer_names(tuple(layers.keys()))
    alpha_limit = _validate_scalar(
        "alpha_threshold", alpha_threshold, 0.0
    )
    contribution_limit = _validate_scalar(
        "contribution_threshold", contribution_threshold, 0.0
    )
    if alpha_limit > 1.0 or contribution_limit > 1.0:
        raise ValueError("Alpha/contribution thresholds must not exceed 1")
    variance_limit = float(variance_threshold)
    if variance_limit < 0.0 or torch.isnan(
        torch.tensor(variance_limit)
    ).item():
        raise ValueError("variance_threshold must be non-negative")
    if depth_tolerance is not None:
        depth_limit = _validate_scalar(
            "depth_tolerance", depth_tolerance, 0.0
        )
    else:
        depth_limit = None
    ratio = _validate_scalar("dominance_ratio", dominance_ratio, 1.0)
    margin = _validate_scalar("dominance_margin", dominance_margin, 0.0)

    reference_shape: Optional[Tuple[int, int, int]] = None
    reference_device: Optional[torch.device] = None
    for name in names:
        reference_shape, reference_device = _validate_layer_buffers(
            name,
            layers[name],
            reference_shape,
            reference_device,
        )

    first_uv = layers[names[0]].uv
    dtype = first_uv.dtype
    device = first_uv.device
    uv = torch.stack(
        [layers[name].uv.to(dtype=dtype) for name in names],
        dim=1,
    )
    variance_raw = torch.stack(
        [layers[name].variance.to(dtype=dtype) for name in names],
        dim=1,
    )
    depth_raw = torch.stack(
        [layers[name].depth.to(dtype=dtype) for name in names],
        dim=1,
    )
    alpha_raw = torch.stack(
        [layers[name].alpha.to(dtype=dtype) for name in names],
        dim=1,
    )
    contribution_raw = torch.stack(
        [layers[name].contribution.to(dtype=dtype) for name in names],
        dim=1,
    )

    # Scalar fields are B,L,1,H,W; remove their known singleton channel for
    # ranking while retaining the original tensors for winner gathering.
    variance = variance_raw[:, :, 0]
    depth = depth_raw[:, :, 0]
    alpha = torch.nan_to_num(
        alpha_raw[:, :, 0], nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(0.0, 1.0)
    contribution = torch.nan_to_num(
        contribution_raw[:, :, 0],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    # For a common correspondence-opacity render, a layer's full-scene
    # contribution cannot exceed its layer-only alpha. Clamp tiny numerical
    # overshoots and make alpha=0 an unconditional rejection.
    contribution = torch.minimum(contribution, alpha)

    finite_uv = torch.isfinite(uv).all(dim=2)
    inside_uv = (
        (uv >= 0.0).all(dim=2) & (uv <= 1.0).all(dim=2)
    )
    candidate = (
        finite_uv
        & inside_uv
        & torch.isfinite(variance)
        & (variance >= 0.0)
        & (variance <= variance_limit)
        & torch.isfinite(depth)
        & (depth > 0.0)
        & (alpha > alpha_limit)
        & (contribution > contribution_limit)
    )

    if depth_limit is not None:
        infinity = torch.full_like(depth, float("inf"))
        nearest_depth = torch.where(candidate, depth, infinity).amin(
            dim=1, keepdim=True
        )
        candidate = candidate & (
            depth <= nearest_depth + depth_limit
        )

    scores = torch.where(
        candidate,
        contribution,
        torch.zeros_like(contribution),
    )
    top_score, winner_index = scores.max(dim=1)
    candidate_count = candidate.sum(dim=1)
    has_candidate = candidate_count > 0

    if len(names) > 1:
        top_two = torch.topk(scores, k=2, dim=1).values
        second_score = top_two[:, 1]
    else:
        second_score = torch.zeros_like(top_score)
    has_competitor = candidate_count > 1
    ambiguous = has_competitor & (
        (top_score < ratio * second_score)
        | ((top_score - second_score) < margin)
    )
    accepted = has_candidate & ~ambiguous

    uv_index = winner_index[:, None, None].expand(
        -1, 1, 2, -1, -1
    )
    selected_uv = torch.gather(uv, 1, uv_index).squeeze(1)
    scalar_index = winner_index[:, None, None]

    def gather_scalar(value: torch.Tensor) -> torch.Tensor:
        return torch.gather(value, 1, scalar_index).squeeze(1)

    selected_variance = gather_scalar(
        torch.nan_to_num(
            variance_raw,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
    )
    selected_depth = gather_scalar(
        torch.nan_to_num(
            depth_raw,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
    )
    selected_alpha = gather_scalar(alpha[:, :, None])
    selected_contribution = gather_scalar(contribution[:, :, None])

    accepted_channel = accepted[:, None]
    selected_uv = torch.where(
        accepted_channel.expand_as(selected_uv),
        torch.nan_to_num(
            selected_uv,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0),
        torch.zeros_like(selected_uv),
    )

    def zero_rejected(value: torch.Tensor) -> torch.Tensor:
        return torch.where(
            accepted_channel,
            value,
            torch.zeros_like(value),
        )

    layer_ids = torch.tensor(
        [SURFACE_LAYER_IDS[name] for name in names],
        dtype=torch.long,
        device=device,
    )
    selected_layer_id = layer_ids[winner_index]
    selected_layer_id = torch.where(
        accepted,
        selected_layer_id,
        torch.full_like(
            selected_layer_id, INVALID_SURFACE_LAYER_ID
        ),
    )[:, None]

    return LayeredSurface(
        surface_uv=selected_uv,
        surface_variance=zero_rejected(selected_variance),
        surface_depth=zero_rejected(selected_depth),
        surface_alpha=zero_rejected(selected_alpha),
        surface_contribution=zero_rejected(selected_contribution),
        layer_id=selected_layer_id,
        validity=accepted_channel.to(dtype=dtype),
        ambiguous=ambiguous[:, None],
    )


__all__ = [
    "INVALID_SURFACE_LAYER_ID",
    "LayerSurfaceBuffers",
    "LayeredSurface",
    "SURFACE_LAYER_BY_NAME",
    "SURFACE_LAYER_DEFINITIONS",
    "SURFACE_LAYER_IDS",
    "SURFACE_LAYER_NAMES",
    "SurfaceLayerDefinition",
    "SurfaceLayerId",
    "compose_layered_surface",
    "decode_surface_rgb_residual",
    "encode_surface_rgb_residual",
    "normalize_alpha_weighted",
    "validate_layer_names",
]
