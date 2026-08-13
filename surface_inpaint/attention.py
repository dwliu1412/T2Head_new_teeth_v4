"""Surface-aware feature propagation for diffusers attention processors.

The implementation deliberately wraps the processor already installed on a
diffusers ``Attention`` module.  The wrapped processor always runs first and
is returned directly when no surface context is active.  Consequently,
installing this module is a bit-for-bit no-op until :meth:`set_context` is
called on the controller.

UV maps use the canonical ``[0, 1]`` convention.  A context contains one UV
map and one confidence/visibility map per logical view.  It may additionally
contain a semantic layer ID per pixel.  Layer IDs become part of the canonical
token key, so equal UV coordinates on, for example, upper and lower teeth do
not exchange features.  During classifier-free guidance, diffusers
conventionally concatenates all unconditional views before all conditional
views; pass ``cfg_branches=2`` to keep the two feature memories strictly
separate.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SurfaceAttentionConfig:
    """Configuration for canonical surface feature propagation.

    Attributes:
        atlas_resolution: Width and height of the square canonical UV atlas.
        max_tokens: Maximum visible token contributions used to construct one
            CFG branch's atlas at one attention layer.
        min_views: Minimum number of distinct views that must observe a texel
            before its fused feature may be propagated.
        strength: Maximum linear blend strength at the end of denoising.
    """

    atlas_resolution: int = 256
    max_tokens: int = 65536
    min_views: int = 2
    strength: float = 1.0

    def __post_init__(self) -> None:
        if int(self.atlas_resolution) <= 0:
            raise ValueError("atlas_resolution must be positive")
        if int(self.max_tokens) <= 0:
            raise ValueError("max_tokens must be positive")
        if int(self.min_views) <= 0:
            raise ValueError("min_views must be positive")
        if not math.isfinite(float(self.strength)) or self.strength < 0.0:
            raise ValueError("strength must be finite and non-negative")


@dataclass
class _SurfaceContext:
    uv: torch.Tensor
    visibility: torch.Tensor
    layer_ids: Optional[torch.Tensor]
    denoise_progress: float
    cfg_branches: int
    cfg_layout: str


class _SurfaceContextState:
    def __init__(self) -> None:
        self.context: Optional[_SurfaceContext] = None
        self.contexts_set = 0
        self.self_attention_calls = 0
        self.progress_updates = 0
        self.maximum_views = 0
        self.visible_tokens = 0


def _coerce_config(
    config: Optional[
        Union[SurfaceAttentionConfig, Mapping[str, Any]]
    ] = None,
) -> SurfaceAttentionConfig:
    if config is None:
        return SurfaceAttentionConfig()
    if isinstance(config, SurfaceAttentionConfig):
        return config
    if isinstance(config, Mapping):
        return SurfaceAttentionConfig(**dict(config))
    raise TypeError(
        "config must be SurfaceAttentionConfig, a mapping, or None"
    )


def _infer_spatial_shape(
    sequence_length: int,
    reference_height: int,
    reference_width: int,
) -> Tuple[int, int]:
    """Infer a non-square token grid using the UV map's aspect ratio."""

    tokens = int(sequence_length)
    if tokens <= 0:
        raise ValueError("sequence_length must be positive")
    target_ratio = float(reference_width) / max(float(reference_height), 1.0)
    best_shape = (1, tokens)
    best_score = float("inf")
    best_size_error = float("inf")
    for divisor in range(1, math.isqrt(tokens) + 1):
        if tokens % divisor:
            continue
        quotient = tokens // divisor
        for height, width in ((divisor, quotient), (quotient, divisor)):
            ratio = float(width) / max(float(height), 1.0)
            score = abs(math.log(max(ratio, 1.0e-12) / target_ratio))
            size_error = abs(height - reference_height) + abs(
                width - reference_width
            )
            if (score, size_error) < (best_score, best_size_error):
                best_shape = (height, width)
                best_score = score
                best_size_error = size_error
    return best_shape


def _flatten_hidden(
    hidden_states: torch.Tensor,
    reference_height: int,
    reference_width: int,
) -> Tuple[torch.Tensor, Tuple[int, int], Any]:
    if hidden_states.ndim == 4:
        batch, channels, height, width = hidden_states.shape
        tokens = (
            hidden_states.permute(0, 2, 3, 1)
            .reshape(batch, height * width, channels)
        )

        def restore(value: torch.Tensor) -> torch.Tensor:
            return (
                value.reshape(batch, height, width, channels)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        return tokens, (height, width), restore
    if hidden_states.ndim == 3:
        batch, sequence_length, _ = hidden_states.shape
        height, width = _infer_spatial_shape(
            sequence_length, reference_height, reference_width
        )

        def restore(value: torch.Tensor) -> torch.Tensor:
            return value

        return hidden_states, (height, width), restore
    raise ValueError(
        "surface propagation expects 3D or 4D hidden states, got shape %s"
        % (tuple(hidden_states.shape),)
    )


def _branch_ids(
    batch_size: int,
    cfg_branches: int,
    cfg_layout: str,
    device: torch.device,
) -> torch.Tensor:
    if batch_size % cfg_branches:
        raise ValueError(
            "hidden batch %d is not divisible by cfg_branches=%d"
            % (batch_size, cfg_branches)
        )
    views = batch_size // cfg_branches
    if cfg_layout == "chunked":
        return torch.arange(
            cfg_branches, device=device, dtype=torch.long
        ).repeat_interleave(views)
    return torch.arange(
        cfg_branches, device=device, dtype=torch.long
    ).repeat(views)


def _expand_context_maps(
    context: _SurfaceContext,
    batch_size: int,
    device: torch.device,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
]:
    uv = context.uv.to(device=device, dtype=torch.float32)
    visibility = context.visibility.to(device=device, dtype=torch.float32)
    layer_ids = (
        context.layer_ids.to(device=device, dtype=torch.long)
        if context.layer_ids is not None
        else None
    )
    map_batch = int(uv.shape[0])
    branches = int(context.cfg_branches)

    if map_batch == batch_size:
        expanded_uv = uv
        expanded_visibility = visibility
        expanded_layer_ids = layer_ids
    elif map_batch * branches == batch_size:
        if context.cfg_layout == "chunked":
            expanded_uv = torch.cat([uv] * branches, dim=0)
            expanded_visibility = torch.cat(
                [visibility] * branches, dim=0
            )
            expanded_layer_ids = (
                torch.cat([layer_ids] * branches, dim=0)
                if layer_ids is not None
                else None
            )
        else:
            expanded_uv = (
                uv[:, None]
                .expand(-1, branches, -1, -1, -1)
                .reshape(batch_size, *uv.shape[1:])
            )
            expanded_visibility = (
                visibility[:, None]
                .expand(-1, branches, -1, -1, -1)
                .reshape(batch_size, *visibility.shape[1:])
            )
            expanded_layer_ids = (
                layer_ids[:, None]
                .expand(-1, branches, -1, -1, -1)
                .reshape(batch_size, *layer_ids.shape[1:])
                if layer_ids is not None
                else None
            )
    else:
        raise ValueError(
            "surface context has %d UV maps, but hidden batch is %d with "
            "cfg_branches=%d"
            % (map_batch, batch_size, branches)
        )

    ids = _branch_ids(
        batch_size, branches, context.cfg_layout, device
    )
    return expanded_uv, expanded_visibility, expanded_layer_ids, ids


def _resize_surface_maps(
    uv: torch.Tensor,
    visibility: torch.Tensor,
    layer_ids: Optional[torch.Tensor],
    height: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    size = (int(height), int(width))
    if tuple(uv.shape[-2:]) != size:
        # Nearest sampling does not interpolate across UV seams.
        uv = F.interpolate(uv, size=size, mode="nearest")
    if tuple(visibility.shape[-2:]) != size:
        source_height, source_width = visibility.shape[-2:]
        if height <= source_height and width <= source_width:
            visibility = F.interpolate(
                visibility, size=size, mode="area"
            )
        else:
            visibility = F.interpolate(
                visibility,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
    if (
        layer_ids is not None
        and tuple(layer_ids.shape[-2:]) != size
    ):
        # Semantic IDs are categorical and must never be interpolated.
        layer_ids = F.interpolate(
            layer_ids.to(dtype=torch.float32),
            size=size,
            mode="nearest",
        ).to(dtype=torch.long)
    return uv, visibility, layer_ids


def _surface_texels(
    uv: torch.Tensor,
    visibility: torch.Tensor,
    atlas_resolution: int,
    layer_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    confidence = torch.nan_to_num(
        visibility[:, 0], nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(0.0, 1.0)
    finite = torch.isfinite(uv).all(dim=1)
    inside = (
        (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= 1.0)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= 1.0)
    )
    valid = finite & inside & (confidence > 0.0)
    safe_uv = torch.nan_to_num(
        uv, nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    maximum = int(atlas_resolution) - 1
    u = (safe_uv[:, 0] * maximum).round().long()
    v = (safe_uv[:, 1] * maximum).round().long()
    texels = v * int(atlas_resolution) + u
    if layer_ids is not None:
        layers = layer_ids[:, 0].long()
        valid = valid & (layers >= 0)
        texels = (
            layers.clamp_min(0) * int(atlas_resolution) ** 2
            + texels
        )
    return texels.flatten(1), confidence.flatten(1), valid.flatten(1)


def _fuse_surface_branch(
    features: torch.Tensor,
    texels: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
    config: SurfaceAttentionConfig,
    gate: float,
) -> torch.Tensor:
    """Fuse one CFG branch without allowing information from another."""

    views, tokens, channels = features.shape
    if views < config.min_views or gate <= 0.0:
        return features

    flat_valid = valid.reshape(-1)
    selected = torch.nonzero(
        flat_valid, as_tuple=False
    ).squeeze(1)
    if selected.numel() == 0:
        return features

    flat_confidence = confidence.reshape(-1)
    if selected.numel() > config.max_tokens:
        scores = flat_confidence[selected]
        selected = selected[
            torch.topk(
                scores,
                k=int(config.max_tokens),
                largest=True,
                sorted=False,
            ).indices
        ]

    selected_weights = flat_confidence[selected].float()
    view_ids = torch.div(selected, tokens, rounding_mode="floor")
    selected_texels = texels.reshape(-1)[selected]
    selected_features = features.reshape(-1, channels)[selected].float()

    atlas_size = int(config.atlas_resolution) ** 2
    # ``selected_texels`` may include a semantic-layer offset.  Preserve the
    # historical atlas span exactly when no offsets are present, and otherwise
    # choose the smallest collision-free span for (view, surface-key) pairs.
    surface_key_span = max(
        atlas_size,
        int(selected_texels.max().item()) + 1,
    )
    pair_keys = view_ids * surface_key_span + selected_texels
    unique_pairs, pair_inverse = torch.unique(
        pair_keys, sorted=True, return_inverse=True
    )
    pair_count = int(unique_pairs.numel())
    pair_feature_sum = torch.zeros(
        (pair_count, channels),
        dtype=torch.float32,
        device=features.device,
    )
    pair_feature_sum.index_add_(
        0,
        pair_inverse,
        selected_features * selected_weights[:, None],
    )
    pair_weight_sum = torch.zeros(
        pair_count, dtype=torch.float32, device=features.device
    )
    pair_weight_sum.index_add_(0, pair_inverse, selected_weights)
    pair_sample_count = torch.zeros_like(pair_weight_sum)
    pair_sample_count.index_add_(
        0, pair_inverse, torch.ones_like(selected_weights)
    )
    pair_features = pair_feature_sum / pair_weight_sum.clamp_min(
        1.0e-8
    )[:, None]
    # One view should not receive extra weight merely because several latent
    # tokens quantize to the same atlas texel.
    pair_confidence = pair_weight_sum / pair_sample_count.clamp_min(1.0)
    pair_texels = unique_pairs.remainder(surface_key_span)

    atlas_texels, atlas_inverse = torch.unique(
        pair_texels, sorted=True, return_inverse=True
    )
    memory_count = int(atlas_texels.numel())
    memory_feature_sum = torch.zeros(
        (memory_count, channels),
        dtype=torch.float32,
        device=features.device,
    )
    memory_feature_sum.index_add_(
        0,
        atlas_inverse,
        pair_features * pair_confidence[:, None],
    )
    memory_weight_sum = torch.zeros(
        memory_count, dtype=torch.float32, device=features.device
    )
    memory_weight_sum.index_add_(
        0, atlas_inverse, pair_confidence
    )
    memory_view_count = torch.zeros(
        memory_count, dtype=torch.long, device=features.device
    )
    memory_view_count.index_add_(
        0,
        atlas_inverse,
        torch.ones_like(atlas_inverse, dtype=torch.long),
    )
    memory_features = memory_feature_sum / memory_weight_sum.clamp_min(
        1.0e-8
    )[:, None]

    target_texels = texels.reshape(-1)
    positions = torch.searchsorted(atlas_texels, target_texels)
    bounded = positions.clamp(max=max(memory_count - 1, 0))
    found = (
        (positions < memory_count)
        & (atlas_texels[bounded] == target_texels)
        & valid.reshape(-1)
        & (
            memory_view_count[bounded]
            >= int(config.min_views)
        )
    )
    if not bool(found.any()):
        return features

    flat_features = features.reshape(-1, channels)
    output = flat_features.clone()
    source = flat_features[found]
    memory = memory_features[bounded[found]].to(dtype=features.dtype)
    output[found] = source + float(gate) * (memory - source)
    return output.reshape(views, tokens, channels)


def _propagate_surface_features(
    hidden_states: torch.Tensor,
    context: _SurfaceContext,
    config: SurfaceAttentionConfig,
) -> torch.Tensor:
    gate = min(
        max(float(config.strength) * context.denoise_progress, 0.0),
        1.0,
    )
    if gate <= 0.0:
        return hidden_states

    reference_height, reference_width = context.uv.shape[-2:]
    try:
        tokens, (height, width), restore = _flatten_hidden(
            hidden_states, reference_height, reference_width
        )
    except ValueError:
        return hidden_states

    uv, visibility, layer_ids, ids = _expand_context_maps(
        context, int(tokens.shape[0]), tokens.device
    )
    uv, visibility, layer_ids = _resize_surface_maps(
        uv, visibility, layer_ids, height, width
    )
    texels, confidence, valid = _surface_texels(
        uv,
        visibility,
        config.atlas_resolution,
        layer_ids=layer_ids,
    )

    output = tokens.clone()
    for branch in range(context.cfg_branches):
        batch_indices = torch.nonzero(
            ids == branch, as_tuple=False
        ).squeeze(1)
        if batch_indices.numel() < config.min_views:
            continue
        branch_output = _fuse_surface_branch(
            tokens.index_select(0, batch_indices),
            texels.index_select(0, batch_indices),
            confidence.index_select(0, batch_indices),
            valid.index_select(0, batch_indices),
            config,
            gate,
        )
        output[batch_indices] = branch_output
    return restore(output)


class SurfaceCorrespondenceAttnProcessor(nn.Module):
    """Wrap one existing diffusers processor with surface propagation."""

    def __init__(
        self,
        base_processor: Any,
        state: _SurfaceContextState,
        config: SurfaceAttentionConfig,
        processor_name: str = "",
    ) -> None:
        super().__init__()
        self.base_processor = base_processor
        self._surface_state = state
        self.config = config
        self.processor_name = str(processor_name)

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
        result = self.base_processor(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            temb,
            *args,
            **kwargs,
        )
        context = self._surface_state.context
        # The direct return is intentional: without context, or when a caller
        # uses this layer as cross-attention, the wrapper adds zero operations.
        if context is None or encoder_hidden_states is not None:
            return result
        self._surface_state.self_attention_calls += 1
        return _propagate_surface_features(result, context, self.config)


class SurfaceAttentionController:
    """Install, configure, and restore surface attention on a diffusers U-Net."""

    def __init__(
        self,
        unet: Any,
        config: Optional[
            Union[SurfaceAttentionConfig, Mapping[str, Any]]
        ] = None,
    ) -> None:
        self.unet = unet
        self.config = _coerce_config(config)
        self._state = _SurfaceContextState()
        self._original_processors: Optional[Dict[str, Any]] = None
        self._wrappers: Dict[
            str, SurfaceCorrespondenceAttnProcessor
        ] = {}

    @property
    def installed(self) -> bool:
        return self._original_processors is not None

    @property
    def context_active(self) -> bool:
        return self._state.context is not None

    @property
    def wrapped_processor_names(self) -> Tuple[str, ...]:
        return tuple(self._wrappers)

    def install(self) -> "SurfaceAttentionController":
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
        wrappers: Dict[str, SurfaceCorrespondenceAttnProcessor] = {}
        for name, processor in original.items():
            if str(name).endswith(".attn1.processor"):
                wrapped = SurfaceCorrespondenceAttnProcessor(
                    processor,
                    self._state,
                    self.config,
                    processor_name=str(name),
                )
                updated[str(name)] = wrapped
                wrappers[str(name)] = wrapped
            else:
                updated[str(name)] = processor
        if not wrappers:
            raise ValueError(
                "the U-Net exposes no processors ending in '.attn1.processor'"
            )

        # diffusers 0.34 mutates processor dictionaries with pop().
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
        visibility: Optional[torch.Tensor] = None,
        *,
        layer_ids: Optional[torch.Tensor] = None,
        denoise_progress: float = 0.0,
        cfg_branches: int = 1,
        cfg_layout: str = "chunked",
    ) -> None:
        """Activate canonical UV propagation for subsequent U-Net calls.

        Args:
            uv: Canonical coordinates with shape ``B x 2 x H x W`` and values
                in ``[0, 1]``. Invalid coordinates may be NaN or outside the
                interval and will be ignored.
            visibility: Optional confidence map ``B x 1 x H x W``. Values are
                clamped to ``[0, 1]``. If omitted, every pixel is visible.
            layer_ids: Optional integer semantic map ``B x 1 x H x W``.
                Non-negative IDs distinguish surfaces that share UV
                coordinates. Negative IDs are treated as invalid pixels. If
                omitted, behavior is identical to the historical UV-only
                correspondence.
            denoise_progress: ``0`` at the noisy start and ``1`` at the final
                clean denoising step.
            cfg_branches: Usually ``2`` for classifier-free guidance and ``1``
                otherwise. Each branch receives an independent atlas.
            cfg_layout: ``"chunked"`` for diffusers' standard
                ``[uncond, cond]`` layout, or ``"interleaved"``.
        """

        if not isinstance(uv, torch.Tensor):
            raise TypeError("uv must be a torch.Tensor")
        if uv.ndim != 4 or uv.shape[1] != 2:
            raise ValueError("uv must have shape B x 2 x H x W")
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

        if visibility is None:
            visibility = torch.ones(
                (uv.shape[0], 1, uv.shape[2], uv.shape[3]),
                dtype=torch.float32,
                device=uv.device,
            )
        if not isinstance(visibility, torch.Tensor):
            raise TypeError("visibility must be a torch.Tensor")
        if (
            visibility.ndim != 4
            or visibility.shape[1] != 1
            or visibility.shape[0] != uv.shape[0]
            or visibility.shape[-2:] != uv.shape[-2:]
        ):
            raise ValueError(
                "visibility must have shape B x 1 x H x W matching uv"
            )
        if layer_ids is not None:
            if not isinstance(layer_ids, torch.Tensor):
                raise TypeError("layer_ids must be a torch.Tensor")
            if (
                layer_ids.ndim != 4
                or layer_ids.shape[1] != 1
                or layer_ids.shape[0] != uv.shape[0]
                or layer_ids.shape[-2:] != uv.shape[-2:]
            ):
                raise ValueError(
                    "layer_ids must have shape B x 1 x H x W matching uv"
                )
            if layer_ids.dtype == torch.bool or torch.is_floating_point(
                layer_ids
            ):
                raise TypeError("layer_ids must use an integer dtype")

        self._state.context = _SurfaceContext(
            uv=uv.detach(),
            visibility=visibility.detach(),
            layer_ids=(
                layer_ids.detach().to(dtype=torch.long)
                if layer_ids is not None
                else None
            ),
            denoise_progress=min(max(progress, 0.0), 1.0),
            cfg_branches=branches,
            cfg_layout=layout,
        )
        self._state.contexts_set += 1
        self._state.maximum_views = max(
            self._state.maximum_views, int(uv.shape[0])
        )
        self._state.visible_tokens += int(
            (visibility.detach() > 0.0).sum().item()
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
        """Return JSON-safe evidence that correspondence attention executed."""

        return {
            "installed": bool(self.installed),
            "wrapped_processors": int(len(self._wrappers)),
            "contexts_set": int(self._state.contexts_set),
            "self_attention_calls": int(
                self._state.self_attention_calls
            ),
            "denoise_progress_updates": int(
                self._state.progress_updates
            ),
            "maximum_joint_views": int(self._state.maximum_views),
            "visible_surface_tokens": int(self._state.visible_tokens),
        }

    def load_diagnostics(self, values: Mapping[str, Any]) -> None:
        """Restore cumulative counters from a training checkpoint."""

        for name, key in (
            ("contexts_set", "contexts_set"),
            ("self_attention_calls", "self_attention_calls"),
            ("progress_updates", "denoise_progress_updates"),
            ("maximum_views", "maximum_joint_views"),
            ("visible_tokens", "visible_surface_tokens"),
        ):
            value = int(values.get(key, 0))
            if value < 0:
                raise ValueError(
                    f"surface-attention diagnostic {key} cannot be negative"
                )
            setattr(self._state, name, value)

    @contextmanager
    def context(
        self,
        uv: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
        *,
        layer_ids: Optional[torch.Tensor] = None,
        denoise_progress: float = 0.0,
        cfg_branches: int = 1,
        cfg_layout: str = "chunked",
    ) -> Iterator["SurfaceAttentionController"]:
        previous = self._state.context
        self.set_context(
            uv,
            visibility,
            layer_ids=layer_ids,
            denoise_progress=denoise_progress,
            cfg_branches=cfg_branches,
            cfg_layout=cfg_layout,
        )
        try:
            yield self
        finally:
            self._state.context = previous

    def __enter__(self) -> "SurfaceAttentionController":
        return self.install()

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.uninstall()


def install_surface_attention(
    unet: Any,
    config: Optional[
        Union[SurfaceAttentionConfig, Mapping[str, Any]]
    ] = None,
) -> SurfaceAttentionController:
    """Create and install a :class:`SurfaceAttentionController`."""

    return SurfaceAttentionController(unet, config).install()
