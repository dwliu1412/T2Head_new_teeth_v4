"""Sparse FLAME-surface K/V memory for diffusers self-attention.

This module wraps processors already installed on a diffusers U-Net.  The
wrapped processor remains the complete intra-view attention path.  When a
surface context is active, a second path projects real Q/K/V tensors and lets
each query attend only to distinct-view K/V slots with the same
``(semantic layer, canonical UV texel)`` key.  The two paths are blended late
in denoising.  No dense UV atlas or token-by-token dense correspondence mask
is constructed.

The direct delegation in :class:`SurfaceMemoryAttnProcessor2_0` is
intentional: without a context (and for cross-attention) installing this
module is a bitwise no-op, including for non-standard base processors.
"""

from __future__ import annotations

import fnmatch
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SurfaceMemoryConfig:
    """Configuration for canonical-surface memory attention."""

    atlas_resolution: int = 64
    max_tokens: int = 65536
    min_views: int = 2
    max_memory_views: int = 4
    strength: float = 0.65
    start_progress: float = 0.45
    end_progress: float = 1.0
    exclude_self: bool = False
    processor_patterns: Tuple[str, ...] = (
        "up_blocks.2",
        "up_blocks.3",
    )

    def __post_init__(self) -> None:
        if int(self.atlas_resolution) <= 0:
            raise ValueError("atlas_resolution must be positive")
        if int(self.max_tokens) <= 0:
            raise ValueError("max_tokens must be positive")
        if int(self.min_views) <= 0:
            raise ValueError("min_views must be positive")
        if int(self.max_memory_views) < int(self.min_views):
            raise ValueError(
                "max_memory_views must be at least min_views"
            )
        strength = float(self.strength)
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be finite and in [0, 1]")
        start = float(self.start_progress)
        end = float(self.end_progress)
        if not (
            math.isfinite(start)
            and math.isfinite(end)
            and 0.0 <= start < end <= 1.0
        ):
            raise ValueError(
                "progress ramp must satisfy "
                "0 <= start_progress < end_progress <= 1"
            )
        patterns = self.processor_patterns
        if isinstance(patterns, (str, bytes)):
            patterns = (str(patterns),)
        else:
            patterns = tuple(str(value) for value in patterns)
        if not patterns or any(not value.strip() for value in patterns):
            raise ValueError(
                "processor_patterns must contain non-empty patterns"
            )
        object.__setattr__(self, "processor_patterns", patterns)


@dataclass
class _SurfaceMemoryContext:
    uv: torch.Tensor
    visibility: torch.Tensor
    layer_ids: torch.Tensor
    depth: torch.Tensor
    denoise_progress: float
    cfg_branches: int
    cfg_layout: str


class _SurfaceMemoryState:
    def __init__(self) -> None:
        self.context: Optional[_SurfaceMemoryContext] = None
        self.contexts_set = 0
        self.progress_updates = 0
        self.self_attention_calls = 0
        self.surface_attention_calls = 0
        self.memory_queries = 0
        self.memory_slots = 0
        self.maximum_views = 0
        self.visible_tokens = 0
        self.invalid_depth_tokens = 0


def _coerce_config(
    config: Optional[Union[SurfaceMemoryConfig, Mapping[str, Any]]],
) -> SurfaceMemoryConfig:
    if config is None:
        return SurfaceMemoryConfig()
    if isinstance(config, SurfaceMemoryConfig):
        return config
    if isinstance(config, Mapping):
        return SurfaceMemoryConfig(**dict(config))
    raise TypeError(
        "config must be SurfaceMemoryConfig, a mapping, or None"
    )


def _infer_spatial_shape(
    sequence_length: int,
    reference_height: int,
    reference_width: int,
) -> Tuple[int, int]:
    """Infer a factorization closest to the context map's aspect ratio."""

    tokens = int(sequence_length)
    if tokens <= 0:
        raise ValueError("sequence_length must be positive")
    target_ratio = float(reference_width) / max(
        float(reference_height), 1.0
    )
    best = (1, tokens)
    best_score = float("inf")
    best_size_error = float("inf")
    for divisor in range(1, math.isqrt(tokens) + 1):
        if tokens % divisor:
            continue
        quotient = tokens // divisor
        for height, width in ((divisor, quotient), (quotient, divisor)):
            ratio = float(width) / max(float(height), 1.0)
            score = abs(
                math.log(max(ratio, 1.0e-12) / target_ratio)
            )
            size_error = abs(height - reference_height) + abs(
                width - reference_width
            )
            if (score, size_error) < (best_score, best_size_error):
                best = (height, width)
                best_score = score
                best_size_error = size_error
    return best


def _branch_and_view_ids(
    batch_size: int,
    cfg_branches: int,
    cfg_layout: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if batch_size % cfg_branches:
        raise ValueError(
            "hidden batch %d is not divisible by cfg_branches=%d"
            % (batch_size, cfg_branches)
        )
    views = batch_size // cfg_branches
    branches = torch.arange(
        cfg_branches, device=device, dtype=torch.long
    )
    view_range = torch.arange(views, device=device, dtype=torch.long)
    if cfg_layout == "chunked":
        return (
            branches.repeat_interleave(views),
            view_range.repeat(cfg_branches),
        )
    return (
        branches.repeat(views),
        view_range.repeat_interleave(cfg_branches),
    )


def _expand_context_maps(
    context: _SurfaceMemoryContext,
    batch_size: int,
    device: torch.device,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    uv = context.uv.to(device=device, dtype=torch.float32)
    visibility = context.visibility.to(device=device, dtype=torch.float32)
    layer_ids = context.layer_ids.to(device=device, dtype=torch.long)
    depth = context.depth.to(device=device, dtype=torch.float32)
    branches, views = _branch_and_view_ids(
        batch_size,
        int(context.cfg_branches),
        context.cfg_layout,
        device,
    )
    map_batch = int(uv.shape[0])
    cfg_branches = int(context.cfg_branches)

    if map_batch == batch_size:
        return uv, visibility, layer_ids, depth, branches, views
    if map_batch * cfg_branches != batch_size:
        raise ValueError(
            "surface context has %d maps, but hidden batch is %d with "
            "cfg_branches=%d"
            % (map_batch, batch_size, cfg_branches)
        )

    if context.cfg_layout == "chunked":
        uv = torch.cat([uv] * cfg_branches, dim=0)
        visibility = torch.cat([visibility] * cfg_branches, dim=0)
        layer_ids = torch.cat([layer_ids] * cfg_branches, dim=0)
        depth = torch.cat([depth] * cfg_branches, dim=0)
    else:
        uv = (
            uv[:, None]
            .expand(-1, cfg_branches, -1, -1, -1)
            .reshape(batch_size, *uv.shape[1:])
        )
        visibility = (
            visibility[:, None]
            .expand(-1, cfg_branches, -1, -1, -1)
            .reshape(batch_size, *visibility.shape[1:])
        )
        layer_ids = (
            layer_ids[:, None]
            .expand(-1, cfg_branches, -1, -1, -1)
            .reshape(batch_size, *layer_ids.shape[1:])
        )
        depth = (
            depth[:, None]
            .expand(-1, cfg_branches, -1, -1, -1)
            .reshape(batch_size, *depth.shape[1:])
        )
    return uv, visibility, layer_ids, depth, branches, views


def _resize_context_maps(
    uv: torch.Tensor,
    visibility: torch.Tensor,
    layer_ids: torch.Tensor,
    depth: torch.Tensor,
    height: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    size = (int(height), int(width))
    if tuple(uv.shape[-2:]) != size:
        # Nearest sampling never interpolates across a UV or semantic seam.
        uv = F.interpolate(uv, size=size, mode="nearest")
    if tuple(visibility.shape[-2:]) != size:
        source_height, source_width = visibility.shape[-2:]
        if height <= source_height and width <= source_width:
            visibility = F.interpolate(visibility, size=size, mode="area")
        else:
            visibility = F.interpolate(
                visibility,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
    if tuple(layer_ids.shape[-2:]) != size:
        layer_ids = F.interpolate(
            layer_ids.to(dtype=torch.float32),
            size=size,
            mode="nearest",
        ).to(dtype=torch.long)
    if tuple(depth.shape[-2:]) != size:
        depth = F.interpolate(depth, size=size, mode="nearest")
    return uv, visibility, layer_ids, depth


def _surface_keys(
    uv: torch.Tensor,
    visibility: torch.Tensor,
    layer_ids: torch.Tensor,
    depth: torch.Tensor,
    atlas_resolution: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    confidence = torch.nan_to_num(
        visibility[:, 0], nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(0.0, 1.0)
    finite_uv = torch.isfinite(uv).all(dim=1)
    inside_uv = (
        (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= 1.0)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= 1.0)
    )
    valid_depth = torch.isfinite(depth[:, 0]) & (depth[:, 0] > 0.0)
    layers = layer_ids[:, 0].long()
    valid = (
        finite_uv
        & inside_uv
        & valid_depth
        & (confidence > 0.0)
        & (layers >= 0)
    )

    safe_uv = torch.nan_to_num(
        uv, nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    maximum = int(atlas_resolution) - 1
    u = (safe_uv[:, 0] * maximum).round().long()
    v = (safe_uv[:, 1] * maximum).round().long()
    texel = v * int(atlas_resolution) + u
    key = layers.clamp_min(0) * int(atlas_resolution) ** 2 + texel
    return key.flatten(1), confidence.flatten(1), valid.flatten(1)


def _working_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _attend_surface_branch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    surface_key: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
    view_ids: torch.Tensor,
    config: SurfaceMemoryConfig,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Attend within one CFG branch using sparse per-surface buckets.

    Inputs use ``views x tokens x heads x head_dim`` layout.  Each
    ``(view, surface_key)`` pair is visibility-weighted into exactly one K/V
    slot before target queries are gathered.
    """

    views, tokens, heads, head_dim = query.shape
    output = torch.zeros_like(value)
    read_mask = torch.zeros(
        (views, tokens), dtype=torch.bool, device=query.device
    )
    source_indices = torch.nonzero(
        valid.reshape(-1), as_tuple=False
    ).squeeze(1)
    if source_indices.numel() == 0:
        return output, read_mask, 0

    flat_surface = surface_key.reshape(-1)
    flat_confidence = confidence.reshape(-1)
    flat_views = (
        view_ids[:, None].expand(-1, tokens).reshape(-1)
    )
    selected_surface = flat_surface[source_indices]
    selected_views = flat_views[source_indices]
    selected_weights = flat_confidence[source_indices]
    key_span = int(selected_surface.max().item()) + 1
    pair_codes = selected_views * key_span + selected_surface
    unique_pairs, pair_inverse = torch.unique(
        pair_codes, sorted=True, return_inverse=True
    )
    pair_count = int(unique_pairs.numel())
    work_dtype = _working_dtype(query.dtype)

    source_key = key.reshape(-1, heads, head_dim)[source_indices].to(
        dtype=work_dtype
    )
    source_value = value.reshape(-1, heads, head_dim)[source_indices].to(
        dtype=work_dtype
    )
    weights = selected_weights.to(dtype=work_dtype)
    pair_key_sum = torch.zeros(
        (pair_count, heads, head_dim),
        dtype=work_dtype,
        device=query.device,
    )
    pair_value_sum = torch.zeros_like(pair_key_sum)
    weighted = weights[:, None, None]
    pair_key_sum.index_add_(0, pair_inverse, source_key * weighted)
    pair_value_sum.index_add_(
        0, pair_inverse, source_value * weighted
    )
    pair_weight_sum = torch.zeros(
        pair_count, dtype=work_dtype, device=query.device
    )
    pair_weight_sum.index_add_(0, pair_inverse, weights)
    denominator = pair_weight_sum.clamp_min(1.0e-8)[:, None, None]
    pair_key = pair_key_sum / denominator
    pair_value = pair_value_sum / denominator

    pair_samples = torch.zeros_like(pair_weight_sum)
    pair_samples.index_add_(
        0, pair_inverse, torch.ones_like(weights)
    )
    pair_confidence = pair_weight_sum / pair_samples.clamp_min(1.0)
    pair_view = torch.div(
        unique_pairs, key_span, rounding_mode="floor"
    )
    pair_surface = unique_pairs.remainder(key_span)

    # Bound sparse working memory after the required per-(view,key)
    # aggregation.  No atlas-sized tensor is ever allocated.
    if pair_count > int(config.max_tokens):
        keep = torch.topk(
            pair_confidence,
            k=int(config.max_tokens),
            largest=True,
            sorted=False,
        ).indices
        pair_key = pair_key[keep]
        pair_value = pair_value[keep]
        pair_view = pair_view[keep]
        pair_surface = pair_surface[keep]
        pair_confidence = pair_confidence[keep]
        pair_count = int(keep.numel())

    # Group slots by sparse surface key.  Stable sorting preserves distinct
    # view order inside a bucket and makes max_memory_views deterministic.
    order = torch.argsort(pair_surface, stable=True)
    pair_surface = pair_surface[order]
    pair_view = pair_view[order]
    pair_key = pair_key[order]
    pair_value = pair_value[order]
    pair_confidence = pair_confidence[order]

    target_surface = flat_surface
    starts = torch.searchsorted(pair_surface, target_surface, right=False)
    ends = torch.searchsorted(pair_surface, target_surface, right=True)
    support = ends - starts
    extra = 1 if bool(config.exclude_self) else 0
    candidate_width = min(
        int(config.max_memory_views) + extra, pair_count
    )
    if candidate_width <= 0:
        return output, read_mask, pair_count
    offsets = torch.arange(
        candidate_width, device=query.device, dtype=torch.long
    )
    positions = starts[:, None] + offsets[None, :]
    candidate_mask = offsets[None, :] < support[:, None]
    safe_positions = positions.clamp(max=pair_count - 1)
    candidate_views = pair_view[safe_positions]
    if bool(config.exclude_self):
        candidate_mask = candidate_mask & (
            candidate_views != flat_views[:, None]
        )
        # If self was not among the first slots, trim the extra non-self slot.
        ranks = candidate_mask.long().cumsum(dim=1)
        candidate_mask = candidate_mask & (
            ranks <= int(config.max_memory_views)
        )

    eligible = (
        valid.reshape(-1)
        & (support >= int(config.min_views))
        & candidate_mask.any(dim=1)
    )
    target_indices = torch.nonzero(
        eligible, as_tuple=False
    ).squeeze(1)
    if target_indices.numel() == 0:
        return output, read_mask, pair_count

    target_positions = safe_positions[target_indices]
    target_candidate_mask = candidate_mask[target_indices]
    memory_key = pair_key[target_positions]
    memory_value = pair_value[target_positions]
    memory_confidence = pair_confidence[target_positions]
    target_query = query.reshape(-1, heads, head_dim)[
        target_indices
    ].to(dtype=work_dtype)
    scores = torch.einsum(
        "nhd,nmhd->nhm", target_query, memory_key
    ) * (float(head_dim) ** -0.5)
    # Treat source visibility as a prior over otherwise matching view slots.
    # This is equivalent to multiplying exp(q.K) by the slot confidence.
    scores = scores + memory_confidence.clamp_min(1.0e-8).log()[
        :, None, :
    ]
    scores = scores.masked_fill(
        ~target_candidate_mask[:, None, :], float("-inf")
    )
    attention = torch.softmax(scores, dim=-1)
    attended = torch.einsum(
        "nhm,nmhd->nhd", attention, memory_value
    ).to(dtype=value.dtype)
    output.reshape(-1, heads, head_dim)[target_indices] = attended
    read_mask.reshape(-1)[target_indices] = True
    return output, read_mask, pair_count


def _project_surface_memory(
    attn: Any,
    hidden_states: torch.Tensor,
    temb: Optional[torch.Tensor],
    context: _SurfaceMemoryContext,
    config: SurfaceMemoryConfig,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, int]:
    """Build projected Q/K/V and return a sparse surface-attention path."""

    residual = hidden_states
    prepared = hidden_states
    spatial_norm = getattr(attn, "spatial_norm", None)
    if spatial_norm is not None:
        prepared = spatial_norm(prepared, temb)
    input_ndim = prepared.ndim
    if input_ndim == 4:
        batch_size, channels, height, width = prepared.shape
        prepared = prepared.view(
            batch_size, channels, height * width
        ).transpose(1, 2)
    elif input_ndim == 3:
        batch_size, sequence_length, _ = prepared.shape
        reference_height, reference_width = context.uv.shape[-2:]
        height, width = _infer_spatial_shape(
            sequence_length, reference_height, reference_width
        )
        channels = int(prepared.shape[-1])
    else:
        raise ValueError(
            "surface memory expects 3D or 4D hidden states, got %s"
            % (tuple(prepared.shape),)
        )

    group_norm = getattr(attn, "group_norm", None)
    if group_norm is not None:
        prepared = group_norm(prepared.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(prepared)
    key = attn.to_k(prepared)
    value = attn.to_v(prepared)
    heads = int(attn.heads)
    inner_dim = int(key.shape[-1])
    if inner_dim % heads:
        raise ValueError("attention K dimension is not divisible by heads")
    head_dim = inner_dim // heads
    if int(query.shape[-1]) != inner_dim or int(value.shape[-1]) != inner_dim:
        raise ValueError(
            "surface memory requires equal Q/K/V projection dimensions"
        )
    query = query.view(batch_size, -1, heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, heads, head_dim).transpose(1, 2)
    norm_q = getattr(attn, "norm_q", None)
    norm_k = getattr(attn, "norm_k", None)
    if norm_q is not None:
        query = norm_q(query)
    if norm_k is not None:
        key = norm_k(key)

    (
        uv,
        visibility,
        layer_ids,
        depth,
        branch_ids,
        view_ids,
    ) = _expand_context_maps(context, batch_size, query.device)
    uv, visibility, layer_ids, depth = _resize_context_maps(
        uv, visibility, layer_ids, depth, height, width
    )
    surface_key, confidence, valid = _surface_keys(
        uv,
        visibility,
        layer_ids,
        depth,
        int(config.atlas_resolution),
    )

    query_tokens = query.transpose(1, 2)
    key_tokens = key.transpose(1, 2)
    value_tokens = value.transpose(1, 2)
    memory_tokens = torch.zeros_like(value_tokens)
    read_mask = torch.zeros_like(valid)
    slot_count = 0
    for branch in range(int(context.cfg_branches)):
        batch_indices = torch.nonzero(
            branch_ids == branch, as_tuple=False
        ).squeeze(1)
        if batch_indices.numel() == 0:
            continue
        branch_memory, branch_read, branch_slots = _attend_surface_branch(
            query_tokens.index_select(0, batch_indices),
            key_tokens.index_select(0, batch_indices),
            value_tokens.index_select(0, batch_indices),
            surface_key.index_select(0, batch_indices),
            confidence.index_select(0, batch_indices),
            valid.index_select(0, batch_indices),
            view_ids.index_select(0, batch_indices),
            config,
        )
        memory_tokens[batch_indices] = branch_memory
        read_mask[batch_indices] = branch_read
        slot_count += int(branch_slots)
    if not bool(read_mask.any()):
        return None, read_mask, confidence, slot_count

    projected = memory_tokens.reshape(
        batch_size, -1, heads * head_dim
    ).to(dtype=query.dtype)
    projected = attn.to_out[0](projected)
    projected = attn.to_out[1](projected)
    if input_ndim == 4:
        projected = projected.transpose(-1, -2).reshape(
            batch_size, channels, height, width
        )
    if bool(getattr(attn, "residual_connection", False)):
        projected = projected + residual
    projected = projected / float(
        getattr(attn, "rescale_output_factor", 1.0)
    )
    return projected, read_mask, confidence, slot_count


def _progress_gate(
    context: _SurfaceMemoryContext, config: SurfaceMemoryConfig
) -> float:
    progress = float(context.denoise_progress)
    start = float(config.start_progress)
    end = float(config.end_progress)
    if progress <= start:
        return 0.0
    if progress >= end:
        return float(config.strength)
    return float(config.strength) * (progress - start) / (end - start)


def _blend_valid_tokens(
    base: torch.Tensor,
    surface: torch.Tensor,
    read_mask: torch.Tensor,
    confidence: torch.Tensor,
    gate: float,
) -> torch.Tensor:
    if base.shape != surface.shape:
        raise ValueError(
            "base and surface attention outputs have different shapes"
        )
    if base.ndim == 4:
        batch, channels, height, width = base.shape
        base_tokens = base.permute(0, 2, 3, 1).reshape(
            batch, height * width, channels
        )
        surface_tokens = surface.permute(0, 2, 3, 1).reshape(
            batch, height * width, channels
        )
    elif base.ndim == 3:
        base_tokens = base
        surface_tokens = surface
    else:
        raise ValueError("attention output must be a 3D or 4D tensor")

    token_gate = (
        confidence.to(device=base.device, dtype=base.dtype)
        * float(gate)
    ).unsqueeze(-1)
    blended_tokens = base_tokens + token_gate * (
        surface_tokens - base_tokens
    )
    # torch.where gives every failed correspondence the exact base value;
    # biases or residuals computed by the surface path cannot leak into it.
    output_tokens = torch.where(
        read_mask.to(device=base.device).unsqueeze(-1),
        blended_tokens,
        base_tokens,
    )
    if base.ndim == 4:
        return (
            output_tokens.reshape(batch, height, width, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
    return output_tokens


class SurfaceMemoryAttnProcessor2_0(nn.Module):
    """AttnProcessor2_0-compatible sparse surface-memory wrapper."""

    def __init__(
        self,
        base_processor: Any,
        state: _SurfaceMemoryState,
        config: SurfaceMemoryConfig,
        processor_name: str = "",
    ) -> None:
        super().__init__()
        self.base_processor = base_processor
        self._surface_memory_state = state
        self.config = config
        self.processor_name = str(processor_name)

    # Attention.forward in diffusers 0.34 inspects __call__ rather than
    # forward, so expose the complete processor signature explicitly.
    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return super().__call__(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            temb,
            *args,
            **kwargs,
        )

    def forward(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        base = self.base_processor(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            temb,
            *args,
            **kwargs,
        )
        context = self._surface_memory_state.context
        # The current SD U-Net self-attention blocks are unmasked. Preserve
        # exact processor semantics for any custom masked self-attention
        # instead of applying a surface path that cannot honor its mask.
        if (
            context is None
            or encoder_hidden_states is not None
            or attention_mask is not None
        ):
            return base
        self._surface_memory_state.self_attention_calls += 1
        gate = _progress_gate(context, self.config)
        if gate <= 0.0:
            return base
        if not isinstance(base, torch.Tensor):
            raise TypeError("base attention processor must return a tensor")

        surface, read_mask, confidence, slots = _project_surface_memory(
            attn, hidden_states, temb, context, self.config
        )
        self._surface_memory_state.memory_slots += int(slots)
        if surface is None:
            return base
        queries = int(read_mask.sum().item())
        self._surface_memory_state.surface_attention_calls += 1
        self._surface_memory_state.memory_queries += queries
        return _blend_valid_tokens(
            base, surface, read_mask, confidence, gate
        )


class SurfaceMemoryController:
    """Install, drive, diagnose, and restore surface-memory processors."""

    def __init__(
        self,
        unet: Any,
        config: Optional[
            Union[SurfaceMemoryConfig, Mapping[str, Any]]
        ] = None,
    ) -> None:
        self.unet = unet
        self.config = _coerce_config(config)
        self._state = _SurfaceMemoryState()
        self._original_processors: Optional[Dict[str, Any]] = None
        self._wrappers: Dict[str, SurfaceMemoryAttnProcessor2_0] = {}

    @property
    def installed(self) -> bool:
        return self._original_processors is not None

    @property
    def context_active(self) -> bool:
        return self._state.context is not None

    @property
    def wrapped_processor_names(self) -> Tuple[str, ...]:
        return tuple(self._wrappers)

    def _matches(self, name: str) -> bool:
        if not name.endswith(".attn1.processor"):
            return False
        return any(
            pattern in name or fnmatch.fnmatchcase(name, pattern)
            for pattern in self.config.processor_patterns
        )

    def install(self) -> "SurfaceMemoryController":
        if self.installed:
            return self
        if not hasattr(self.unet, "attn_processors") or not hasattr(
            self.unet, "set_attn_processor"
        ):
            raise TypeError(
                "unet must expose attn_processors and set_attn_processor"
            )

        original = dict(self.unet.attn_processors)
        updated: Dict[str, Any] = {}
        wrappers: Dict[str, SurfaceMemoryAttnProcessor2_0] = {}
        for name, processor in original.items():
            processor_name = str(name)
            if self._matches(processor_name):
                wrapped = SurfaceMemoryAttnProcessor2_0(
                    processor,
                    self._state,
                    self.config,
                    processor_name=processor_name,
                )
                updated[processor_name] = wrapped
                wrappers[processor_name] = wrapped
            else:
                updated[processor_name] = processor
        if not wrappers:
            raise ValueError(
                "no '.attn1.processor' matched processor_patterns=%r"
                % (self.config.processor_patterns,)
            )

        # diffusers 0.34 consumes processor dictionaries with pop().
        self.unet.set_attn_processor(dict(updated))
        self._original_processors = original
        self._wrappers = wrappers
        return self

    def uninstall(self) -> None:
        self.clear_context()
        if self._original_processors is None:
            return
        self.unet.set_attn_processor(dict(self._original_processors))
        self._original_processors = None
        self._wrappers = {}

    def set_context(
        self,
        uv: torch.Tensor,
        visibility: torch.Tensor,
        *,
        layer_ids: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        denoise_progress: float = 0.0,
        cfg_branches: int = 2,
        cfg_layout: str = "chunked",
    ) -> None:
        """Activate surface correspondence for subsequent U-Net calls.

        Maps have ``B x C x H x W`` layout.  ``B`` may contain one map per
        logical view (the usual CFG case) or one map per hidden batch item.
        Depth samples are valid only when finite and strictly positive.
        Omitting depth supplies a finite unit-depth map for callers that do
        not have an explicit z-buffer.
        """

        if not isinstance(uv, torch.Tensor):
            raise TypeError("uv must be a torch.Tensor")
        if uv.ndim != 4 or uv.shape[1] != 2:
            raise ValueError("uv must have shape B x 2 x H x W")
        if not isinstance(visibility, torch.Tensor):
            raise TypeError("visibility must be a torch.Tensor")
        expected = (uv.shape[0], 1, uv.shape[2], uv.shape[3])
        if tuple(visibility.shape) != expected:
            raise ValueError(
                "visibility must have shape B x 1 x H x W matching uv"
            )

        if layer_ids is None:
            layer_ids = torch.zeros(
                expected,
                dtype=torch.long,
                device=uv.device,
            )
        if not isinstance(layer_ids, torch.Tensor):
            raise TypeError("layer_ids must be a torch.Tensor")
        if tuple(layer_ids.shape) != expected:
            raise ValueError(
                "layer_ids must have shape B x 1 x H x W matching uv"
            )
        if layer_ids.dtype == torch.bool or torch.is_floating_point(
            layer_ids
        ):
            raise TypeError("layer_ids must use an integer dtype")

        if depth is None:
            depth = torch.ones(
                expected, dtype=torch.float32, device=uv.device
            )
        if not isinstance(depth, torch.Tensor):
            raise TypeError("depth must be a torch.Tensor")
        if tuple(depth.shape) != expected:
            raise ValueError(
                "depth must have shape B x 1 x H x W matching uv"
            )

        branches = int(cfg_branches)
        if branches <= 0:
            raise ValueError("cfg_branches must be positive")
        layout = str(cfg_layout).lower()
        if layout not in {"chunked", "interleaved"}:
            raise ValueError(
                "cfg_layout must be 'chunked' or 'interleaved'"
            )
        progress = float(denoise_progress)
        if not math.isfinite(progress):
            raise ValueError("denoise_progress must be finite")

        detached_uv = uv.detach()
        detached_visibility = visibility.detach()
        detached_layers = layer_ids.detach().to(dtype=torch.long)
        detached_depth = depth.detach()
        self._state.context = _SurfaceMemoryContext(
            uv=detached_uv,
            visibility=detached_visibility,
            layer_ids=detached_layers,
            depth=detached_depth,
            denoise_progress=min(max(progress, 0.0), 1.0),
            cfg_branches=branches,
            cfg_layout=layout,
        )
        self._state.contexts_set += 1
        self._state.maximum_views = max(
            self._state.maximum_views, int(uv.shape[0])
        )
        finite_depth = torch.isfinite(detached_depth) & (
            detached_depth > 0.0
        )
        visible = torch.isfinite(detached_visibility) & (
            detached_visibility > 0.0
        )
        self._state.visible_tokens += int(
            (finite_depth & visible).sum().item()
        )
        self._state.invalid_depth_tokens += int(
            (~finite_depth).sum().item()
        )

    def set_denoise_progress(self, progress: float) -> None:
        if self._state.context is None:
            raise RuntimeError(
                "set_context must be called before set_denoise_progress"
            )
        value = float(progress)
        if not math.isfinite(value):
            raise ValueError("denoise progress must be finite")
        self._state.context.denoise_progress = min(max(value, 0.0), 1.0)
        self._state.progress_updates += 1

    def clear_context(self) -> None:
        self._state.context = None

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "installed": bool(self.installed),
            "context_active": bool(self.context_active),
            "wrapped_processors": int(len(self._wrappers)),
            "contexts_set": int(self._state.contexts_set),
            "denoise_progress_updates": int(
                self._state.progress_updates
            ),
            "self_attention_calls": int(
                self._state.self_attention_calls
            ),
            "surface_attention_calls": int(
                self._state.surface_attention_calls
            ),
            "memory_queries": int(self._state.memory_queries),
            "memory_slots": int(self._state.memory_slots),
            "maximum_joint_views": int(self._state.maximum_views),
            "visible_surface_tokens": int(self._state.visible_tokens),
            "invalid_depth_tokens": int(
                self._state.invalid_depth_tokens
            ),
        }

    def load_diagnostics(self, values: Mapping[str, Any]) -> None:
        for name, key in (
            ("contexts_set", "contexts_set"),
            ("progress_updates", "denoise_progress_updates"),
            ("self_attention_calls", "self_attention_calls"),
            ("surface_attention_calls", "surface_attention_calls"),
            ("memory_queries", "memory_queries"),
            ("memory_slots", "memory_slots"),
            ("maximum_views", "maximum_joint_views"),
            ("visible_tokens", "visible_surface_tokens"),
            ("invalid_depth_tokens", "invalid_depth_tokens"),
        ):
            value = int(values.get(key, 0))
            if value < 0:
                raise ValueError(
                    f"surface-memory diagnostic {key} cannot be negative"
                )
            setattr(self._state, name, value)

    @contextmanager
    def context(
        self,
        uv: torch.Tensor,
        visibility: torch.Tensor,
        *,
        layer_ids: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        denoise_progress: float = 0.0,
        cfg_branches: int = 2,
        cfg_layout: str = "chunked",
    ) -> Iterator["SurfaceMemoryController"]:
        previous = self._state.context
        self.set_context(
            uv,
            visibility,
            layer_ids=layer_ids,
            depth=depth,
            denoise_progress=denoise_progress,
            cfg_branches=cfg_branches,
            cfg_layout=cfg_layout,
        )
        try:
            yield self
        finally:
            self._state.context = previous

    def __enter__(self) -> "SurfaceMemoryController":
        return self.install()

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.uninstall()


def install_surface_memory_attention(
    unet: Any,
    config: Optional[
        Union[SurfaceMemoryConfig, Mapping[str, Any]]
    ] = None,
) -> SurfaceMemoryController:
    """Create and install a :class:`SurfaceMemoryController`."""

    return SurfaceMemoryController(unet, config).install()


__all__ = [
    "SurfaceMemoryConfig",
    "SurfaceMemoryController",
    "SurfaceMemoryAttnProcessor2_0",
    "install_surface_memory_attention",
]
