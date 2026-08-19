"""Shared command-line plumbing for reconstruction mouth/full refinement."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
GUIDANCE_MODES = ("ism", "uvd-sfd")
SDEDIT_MODES = ("independent", "flame-surface")
DEFAULT_SURFACE_SDEDIT_VIEWS = 4
FACE_LOCAL_SCALE_ROTATION_COMMENT = (
    "comment scale_rotation_space=flame_face_local_v1"
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def omega_override(name: str, value: object) -> str:
    """Encode a scalar so OmegaConf cannot reinterpret prompts or Windows paths."""

    if isinstance(value, bool):
        encoded = "true" if value else "false"
    elif isinstance(value, (int, float)):
        encoded = str(value)
    else:
        encoded = json.dumps(str(value), ensure_ascii=False)
    return f"{name}={encoded}"


def require_stage1(reconstruction_dir: Path) -> None:
    required = (
        reconstruction_dir / "resolved_config.yaml",
        reconstruction_dir / "model" / "uvd.ply",
        reconstruction_dir / "model" / "reconstruction_params.npz",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Incomplete Stage-1 reconstruction; missing: "
            + ", ".join(str(path) for path in missing)
        )
    uvd_path = reconstruction_dir / "model" / "uvd.ply"
    with uvd_path.open("rb") as stream:
        header_lines = []
        for _ in range(512):
            line = stream.readline()
            if not line:
                break
            decoded = line.decode("ascii", errors="replace").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
    if FACE_LOCAL_SCALE_ROTATION_COMMENT not in header_lines:
        raise ValueError(
            "Stage-1 uvd.ply uses the legacy UVD covariance representation. "
            "Rerun reconstruction, or convert it with "
            "tools/convert_legacy_uvd_ply.py before train_mouth."
        )


def output_overrides(output_dir: Path) -> list[str]:
    if not output_dir.name:
        raise ValueError(f"Output directory must have a final component: {output_dir}")
    return [
        omega_override("exp_root_dir", output_dir.parent),
        omega_override("name", output_dir.name),
        omega_override("tag", "run"),
        omega_override("use_timestamp", False),
    ]


def guidance_mode_overrides(
    mode: str,
    *,
    uvd_flow_seed: int = 0,
) -> list[str]:
    """Select raw ISM or CFD-consistent UVD-SFD."""

    mode = str(mode).strip().lower()
    if mode not in GUIDANCE_MODES:
        raise ValueError(
            f"guidance mode must be one of {GUIDANCE_MODES}, got {mode!r}"
        )
    seed = int(uvd_flow_seed)
    if seed < 0:
        raise ValueError("UVD-SFD noise seed must be non-negative")
    return [
        omega_override("system.guidance.use_ism", mode == "ism"),
        omega_override(
            "system.guidance.use_uvd_surface_flow", mode == "uvd-sfd"
        ),
        omega_override("system.guidance.uvd_flow_noise_seed", seed),
    ]


def sdedit_mode_overrides(
    mode: str,
    *,
    surface_views: int = DEFAULT_SURFACE_SDEDIT_VIEWS,
) -> list[str]:
    """Select the independent or joint FLAME-surface SDEdit branch.

    The switch is deliberately orthogonal to :func:`guidance_mode_overrides`:
    ISM/UVD-SFD controls the first optimization phase, while this
    function
    changes only the subsequent img2img teacher.  The independent branch does
    not override ``data.batch_size`` and therefore preserves legacy configs.
    """

    mode = str(mode).strip().lower()
    if mode not in SDEDIT_MODES:
        raise ValueError(
            f"SDEdit mode must be one of {SDEDIT_MODES}, got {mode!r}"
        )
    views = int(surface_views)
    enabled = mode == "flame-surface"
    if enabled and views < 2:
        raise ValueError("FLAME surface SDEdit requires at least two views")
    overrides = [
        omega_override("system.sdedit.mode", mode),
        omega_override(
            "system.sdedit.surface_memory.enabled", enabled
        ),
        # Row zero follows the ordinary B=1 camera distribution only for the
        # joint branch; the data module fills the remaining surface views.
        omega_override("data.surface_consistent_batch", enabled),
    ]
    if enabled:
        overrides.extend(
            [
                omega_override("data.batch_size", views),
                omega_override(
                    "system.sdedit.surface_memory.views", views
                ),
                # The shipped configs bound source slots at four.  Clamp the
                # bound for smaller joint batches so --surface-views 2/3 is
                # valid without turning a larger batch into an unbounded
                # memory expansion.
                omega_override(
                    "system.sdedit.surface_memory.max_memory_views",
                    min(views, DEFAULT_SURFACE_SDEDIT_VIEWS),
                ),
            ]
        )
    return overrides


def refinement_run_name(
    stage: str,
    guidance_mode: str,
    sdedit_mode: str,
) -> str:
    """Return a collision-free output name for the two independent switches."""

    stage = str(stage).strip().lower()
    if stage not in {"mouth", "full"}:
        raise ValueError("stage must be 'mouth' or 'full'")
    guidance = str(guidance_mode).strip().lower()
    sdedit = str(sdedit_mode).strip().lower()
    if guidance not in GUIDANCE_MODES:
        raise ValueError(
            f"guidance mode must be one of {GUIDANCE_MODES}, got {guidance!r}"
        )
    if sdedit not in SDEDIT_MODES:
        raise ValueError(
            f"SDEdit mode must be one of {SDEDIT_MODES}, got {sdedit!r}"
        )
    parts = [stage]
    if guidance == "uvd-sfd":
        parts.append("uvd_sfd")
    if sdedit == "flame-surface":
        parts.append("surface_sdedit")
    return "_".join(parts)


def run_launch(
    config: Path,
    gpu: str,
    overrides: Iterable[str],
    dry_run: bool,
) -> int:
    config = resolve_path(config)
    if not config.is_file():
        raise FileNotFoundError(f"Refinement config does not exist: {config}")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "launch.py"),
        "--config",
        str(config),
        "--train",
        "--gpu",
        str(gpu),
        *list(overrides),
    ]
    print(subprocess.list2cmdline(command), flush=True)
    if dry_run:
        return 0
    environment = os.environ.copy()
    environment.setdefault("THREESTUDIO_LAZY_IMPORT", "1")
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    )
    return int(completed.returncode)
