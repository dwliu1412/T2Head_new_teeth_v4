"""Canonical UVD/CFD noise transport for first-phase UVD-SFD.

The animated avatar does not merely live on one FLAME UV sheet.  Every
Gaussian is attached by ``(surface_layer, u, v, d)`` where ``d`` is its signed
normal offset, and the generated upper/lower teeth deliberately reuse numeric
UV coordinates.  A single 2-D atlas therefore aliases distinct pieces of the
avatar.  This module draws a fresh Gaussian noise *volume* in that canonical
UVD reference space for every optimizer step and transports it to latent
pixels.  All views and regional crops evaluated in the same step reuse the
same draw; the next step is resampled so ISM remains a Monte-Carlo objective.

The implementation is renderer independent and intentionally contains no
diffusion-model dependency.  The caller supplies one visible UVD coordinate,
semantic layer id, and confidence per screen pixel.  For every latent pixel we
enumerate the canonical cells observed by its complete renderer-aligned image
tile.  Duplicate cells are removed and the remaining iid Gaussian cells are
summed with ``1 / sqrt(N)`` normalization.  This discrete transport rule
preserves unit marginal variance while correlating views that observe the same
animated surface.  Importantly, every high-resolution sample keeps its own
semantic layer: a mouth-boundary footprint may contain lips, upper teeth,
lower teeth, and oral cavity without aliasing them onto the centre sample.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch


def _as_bchw(
    value: torch.Tensor,
    channels: int,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device)
    if tensor.ndim == 3 and channels == 1:
        tensor = tensor[:, None]
    elif tensor.ndim == 4 and tensor.shape[1] == channels:
        pass
    elif tensor.ndim == 4 and tensor.shape[-1] == channels:
        tensor = tensor.permute(0, 3, 1, 2)
    if tensor.ndim != 4 or tensor.shape[1] != channels:
        raise ValueError(
            f"{name} must have shape Bx{channels}xHxW or BxHxWx{channels}, "
            f"got {tuple(tensor.shape)}"
        )
    return tensor.to(dtype=dtype)


class UVDNoiseVolume:
    """Per-step iid Gaussian cells indexed by semantic FLAME UVD.

    ``noise_volume`` has shape ``[L, 4, D, R, R]``.  Keeping the semantic
    layer and normal-offset dimensions explicit prevents face/hair, lips,
    upper teeth, lower teeth, and oral-cavity points from sharing noise merely
    because their numeric UV happens to coincide.
    """

    def __init__(
        self,
        *,
        uv_resolution: int,
        depth_resolution: int,
        layer_count: int,
        seed: int,
        device: torch.device,
        state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.uv_resolution = int(uv_resolution)
        self.depth_resolution = int(depth_resolution)
        self.layer_count = int(layer_count)
        self.seed = int(seed)
        self.device = torch.device(device)
        if self.uv_resolution < 64:
            raise ValueError("UVD flow uv_resolution must be at least 64")
        if self.depth_resolution < 1:
            raise ValueError("UVD flow depth_resolution must be positive")
        if self.layer_count < 1:
            raise ValueError("UVD flow layer_count must be positive")
        if self.seed < 0:
            raise ValueError("UVD flow seed must be non-negative")

        self.volume_generator = torch.Generator(device=self.device)
        self.volume_generator.manual_seed(self.seed)
        self.background_generator = torch.Generator(device=self.device)
        background_seed = (self.seed ^ 0x5DEECE66D) & ((1 << 63) - 1)
        self.background_generator.manual_seed(background_seed)
        expected = (
            self.layer_count,
            4,
            self.depth_resolution,
            self.uv_resolution,
            self.uv_resolution,
        )
        if state is None:
            self.noise_volume = torch.randn(
                expected,
                generator=self.volume_generator,
                device=self.device,
                dtype=torch.float32,
            )
            self.last_resample_step = -1
        else:
            schema_version = int(state.get("schema_version", -1))
            if schema_version != 5:
                raise ValueError(
                    "UVD flow checkpoint schema is incompatible with the "
                    "per-step ISM noise implementation: "
                    f"{schema_version}"
                )
            saved_backend = str(state.get("generator_device_type", ""))
            if saved_backend != self.device.type:
                raise ValueError(
                    "UVD flow RNG checkpoint backend differs from the current "
                    f"device ({saved_backend!r} vs {self.device.type!r})"
                )
            for key, current in (
                ("uv_resolution", self.uv_resolution),
                ("depth_resolution", self.depth_resolution),
                ("layer_count", self.layer_count),
                ("seed", self.seed),
            ):
                if int(state.get(key, -1)) != current:
                    raise ValueError(
                        f"UVD flow checkpoint {key} differs from the current "
                        f"configuration ({state.get(key)!r} vs {current})"
                    )
            self.noise_volume = torch.as_tensor(
                state["noise_volume"], dtype=torch.float32, device=self.device
            ).clone()
            if tuple(self.noise_volume.shape) != expected:
                raise ValueError(
                    "UVD flow checkpoint volume shape differs from the current "
                    f"configuration: {tuple(self.noise_volume.shape)} vs {expected}"
                )
            self.last_resample_step = int(
                state.get("last_resample_step", -1)
            )
            volume_generator_state = state.get("volume_generator_state")
            background_generator_state = state.get(
                "background_generator_state"
            )
            if (
                volume_generator_state is None
                or background_generator_state is None
            ):
                raise ValueError(
                    "UVD flow checkpoint has incomplete private RNG state"
                )
            self.volume_generator.set_state(
                torch.as_tensor(
                    volume_generator_state, dtype=torch.uint8, device="cpu"
                )
            )
            self.background_generator.set_state(
                torch.as_tensor(
                    background_generator_state,
                    dtype=torch.uint8,
                    device="cpu",
                )
            )

    def state_dict(self) -> dict[str, Any]:
        """Return the current per-step draw and private RNGs for exact resume."""

        return {
            "schema_version": 5,
            "noise_volume": self.noise_volume.detach().cpu().clone(),
            "volume_generator_state": (
                self.volume_generator.get_state().detach().cpu().clone()
            ),
            "background_generator_state": (
                self.background_generator.get_state().detach().cpu().clone()
            ),
            "generator_device_type": self.device.type,
            "last_resample_step": int(self.last_resample_step),
            "uv_resolution": int(self.uv_resolution),
            "depth_resolution": int(self.depth_resolution),
            "layer_count": int(self.layer_count),
            "seed": int(self.seed),
        }

    @torch.no_grad()
    def resample_for_step(self, step: int) -> None:
        """Draw once for ``step`` and reuse that draw across all its calls."""

        step = int(step)
        if step < 0:
            raise ValueError("UVD-SFD optimizer step must be non-negative")
        if step == self.last_resample_step:
            return
        if self.last_resample_step >= 0 and step < self.last_resample_step:
            raise ValueError(
                "UVD-SFD optimizer step moved backwards without restoring "
                "its checkpoint"
            )
        # The constructor already supplied the first iid draw.  Avoid drawing
        # and discarding another 5x4xDxRxR tensor at the first observed step.
        if self.last_resample_step >= 0:
            self.noise_volume.normal_(
                mean=0.0,
                std=1.0,
                generator=self.volume_generator,
            )
        self.last_resample_step = step

    @torch.no_grad()
    def sample(
        self,
        surface_uvd: torch.Tensor,
        layer_ids: torch.Tensor,
        confidence: torch.Tensor,
        *,
        latent_size: tuple[int, int],
        minimum_distinct_cells: int = 1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Transport canonical UVD noise into one latent noise tensor.

        The correspondence image must divide exactly into latent tiles.  The
        Stage-2 path supplies 512x512 correspondences for SD-1.5's 64x64
        latent, hence each footprint contains all 8x8 screen samples.  This is
        deliberately exhaustive rather than a fixed-count point estimate.
        """

        uvd = _as_bchw(
            surface_uvd,
            3,
            name="uvd_flow_surface_uvd",
            device=self.device,
            dtype=torch.float32,
        ).detach()
        layers = _as_bchw(
            layer_ids,
            1,
            name="uvd_flow_surface_layer",
            device=self.device,
            dtype=torch.float32,
        ).detach()
        confidence = _as_bchw(
            confidence,
            1,
            name="uvd_flow_surface_confidence",
            device=self.device,
            dtype=torch.float32,
        ).detach()
        if uvd.shape[0] != layers.shape[0] or uvd.shape[0] != confidence.shape[0]:
            raise ValueError("UVD flow correspondence batch sizes differ")
        if uvd.shape[-2:] != layers.shape[-2:] or uvd.shape[-2:] != confidence.shape[-2:]:
            raise ValueError("UVD flow correspondence spatial shapes differ")

        latent_height, latent_width = (int(value) for value in latent_size)
        if latent_height < 1 or latent_width < 1:
            raise ValueError("UVD flow latent size must be positive")
        minimum_distinct_cells = int(minimum_distinct_cells)
        if minimum_distinct_cells < 1:
            raise ValueError(
                "UVD flow minimum_distinct_cells must be positive"
            )
        source_height, source_width = uvd.shape[-2:]
        if (
            source_height < latent_height
            or source_width < latent_width
            or source_height % latent_height
            or source_width % latent_width
        ):
            raise ValueError(
                "UVD flow correspondence resolution must be an integer "
                "multiple of the latent resolution, got "
                f"{source_height}x{source_width} for "
                f"{latent_height}x{latent_width}"
            )
        tile_height = source_height // latent_height
        tile_width = source_width // latent_width

        def tile(value: torch.Tensor) -> torch.Tensor:
            batch, channels = value.shape[:2]
            return (
                value.reshape(
                    batch,
                    channels,
                    latent_height,
                    tile_height,
                    latent_width,
                    tile_width,
                )
                .permute(0, 2, 4, 3, 5, 1)
                .reshape(
                    batch,
                    latent_height,
                    latent_width,
                    tile_height * tile_width,
                    channels,
                )
            )

        uvd_samples = tile(uvd)
        layer_samples = tile(layers).round().long()[..., 0]
        confidence_samples = tile(
            torch.nan_to_num(
                confidence, nan=0.0, posinf=1.0, neginf=0.0
            ).clamp(0.0, 1.0)
        )[..., 0]
        finite = torch.isfinite(uvd_samples).all(dim=-1)
        in_bounds = ((uvd_samples >= 0.0) & (uvd_samples <= 1.0)).all(
            dim=-1
        )
        valid_layer = (
            (layer_samples >= 0) & (layer_samples < self.layer_count)
        )
        valid_samples = (
            finite
            & in_bounds
            & valid_layer
            & (confidence_samples > 0.0)
        )
        safe_uvd = torch.nan_to_num(
            uvd_samples, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        safe_layer = layer_samples.clamp(0, self.layer_count - 1)

        resolution = self.uv_resolution
        depth_resolution = self.depth_resolution
        u_index = torch.floor(safe_uvd[..., 0] * resolution).long().clamp(
            0, resolution - 1
        )
        v_index = torch.floor(safe_uvd[..., 1] * resolution).long().clamp(
            0, resolution - 1
        )
        d_index = torch.floor(
            safe_uvd[..., 2] * depth_resolution
        ).long().clamp(
            0, depth_resolution - 1
        )

        # Advanced indexing returns [B,H,W,N,4].  Cells are sorted by a unique
        # integer key so repeated renderer samples count exactly once.
        cells = self.noise_volume.permute(0, 2, 3, 4, 1)[
            safe_layer, d_index, v_index, u_index
        ]
        keys = (
            ((safe_layer * depth_resolution + d_index) * resolution + v_index)
            * resolution
            + u_index
        )
        invalid_key = self.layer_count * depth_resolution * resolution * resolution
        keys = torch.where(
            valid_samples, keys, torch.full_like(keys, invalid_key)
        )
        order = keys.argsort(dim=-1)
        sorted_keys = torch.gather(keys, -1, order)
        sorted_valid = torch.gather(valid_samples, -1, order)
        sorted_cells = torch.gather(
            cells,
            3,
            order[..., None].expand(-1, -1, -1, -1, 4),
        )
        sample_count = tile_height * tile_width
        unique = sorted_valid.clone()
        if sample_count > 1:
            unique[..., 1:] = sorted_keys[..., 1:] != sorted_keys[..., :-1]
            unique[..., 1:] &= sorted_valid[..., 1:]
        unique_count = unique.sum(dim=-1).clamp_min(1)
        surface_noise = (
            sorted_cells * unique[..., None].to(sorted_cells.dtype)
        ).sum(dim=3) / unique_count[..., None].sqrt()
        surface_noise = surface_noise.permute(0, 3, 1, 2).contiguous()

        valid_latent = valid_samples.any(dim=-1)
        surface_confidence = (
            confidence_samples * valid_samples.to(confidence_samples.dtype)
        ).mean(dim=-1)
        # A Gaussian-centre correspondence can collapse a complete image tile
        # onto very few UVD cells.  Its marginal variance is still one, but it
        # no longer approximates the per-pixel-independent noise assumed by
        # the diffusion teacher.  Treat cell support as correspondence
        # reliability instead of letting a one-cell footprint dominate the
        # spatial covariance of an ISM draw.
        transport_reliability = (
            unique_count.float() / float(minimum_distinct_cells)
        ).clamp(0.0, 1.0)
        transport_reliability = (
            transport_reliability * valid_latent.float()
        )[:, None]
        # Screen background and the unresolved part of a surface footprint
        # have no reliable canonical identity.  Use independent fallback
        # noise per view/call, and blend it with the canonical component using
        # square-root weights.  The marginal remains N(0, I), while reliable
        # correspondences retain strong cross-view covariance.  Crucially,
        # reliability changes the noise coupling, never the ISM gradient
        # magnitude.
        background = torch.randn(
            (uvd.shape[0], 4, latent_height, latent_width),
            generator=self.background_generator,
            device=self.device,
            dtype=torch.float32,
        )
        canonical_weight = transport_reliability.sqrt()
        fallback_weight = (1.0 - transport_reliability).sqrt()
        noise = (
            canonical_weight * surface_noise
            + fallback_weight * background
        ).detach()

        metrics = {
            "uvd_flow_noise_mean": noise.mean(),
            "uvd_flow_noise_std": noise.std(unbiased=False),
            "uvd_flow_surface_fraction": valid_latent.float().mean(),
            "uvd_flow_surface_confidence": surface_confidence.mean(),
            "uvd_flow_distinct_cells": unique_count[valid_latent].float().mean()
            if bool(valid_latent.any())
            else torch.ones((), device=self.device),
            "uvd_flow_cell_coverage": (
                unique_count[valid_latent].float().mean()
                / float(sample_count)
                if bool(valid_latent.any())
                else torch.zeros((), device=self.device)
            ),
            "uvd_flow_transport_reliability": (
                transport_reliability.mean()
            ),
        }
        return noise, metrics
