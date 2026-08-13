"""Prompt/view conventions copied from AnimPortrait3D's GSAvatar scripts."""

from __future__ import annotations

import torch


def view_direction_indices(azimuth: torch.Tensor) -> torch.Tensor:
    """Return GSAvatar direction indices (0=side, 1=front, 2=back).

    AnimPortrait3D samples yaw in [0, 360] with portrait front at 90 degrees.
    Its strict conditions label (60, 120) as front, (0, 60) U (120, 195)
    as side, and every remaining yaw as back.
    """

    angle = torch.remainder(azimuth, 360.0)
    front = (angle > 60.0) & (angle < 120.0)
    side = ((angle > 120.0) & (angle < 195.0)) | (
        (angle < 60.0) & (angle >= 0.0)
    )
    direction = torch.full_like(angle, 2, dtype=torch.long)
    direction[side] = 0
    direction[front] = 1
    return direction


def view_prompt(direction: str, prompt: str) -> str:
    """Match helper.py's ``f'{view_direction} view {original_prompt}'``."""

    direction = str(direction).strip().lower()
    if direction not in {"side", "front", "back"}:
        raise ValueError(f"Unsupported AnimPortrait3D view direction: {direction!r}")
    return f"{direction} view {prompt}"
