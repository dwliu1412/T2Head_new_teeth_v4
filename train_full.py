"""Run full dynamic refinement from the mouth-optimized UVD avatar.

This is the current UVD-FLAME counterpart of AnimPortrait3D/GSAvatar's
``train_all.py`` (the reference repository has no separate ``train_full.py``).
It uses full/face guidance followed by masked SDEdit reconstruction.  Eye and
mouth point parameters are preserved from the incoming mouth PLY; they remain
visible as context but do not participate in full-stage topology changes or
optimizer updates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from refinement_cli import (
    DEFAULT_SURFACE_SDEDIT_VIEWS,
    GUIDANCE_MODES,
    SDEDIT_MODES,
    guidance_mode_overrides,
    omega_override,
    output_overrides,
    refinement_run_name,
    require_stage1,
    resolve_path,
    run_launch,
    sdedit_mode_overrides,
)

ISM_STEPS = 1000
MOUTH_REQUIRED_STEPS = 1000
DEFAULT_ANIMPORTRAIT3D_ROOT = Path(r"H:\work2\others\AnimPortrait3D")
DEFAULT_CONTROLNET = (
    DEFAULT_ANIMPORTRAIT3D_ROOT
    / "pretrained_model"
    / "AnimPortrait3D_controlnet"
)
DEFAULT_DIFFUSION = (
    DEFAULT_ANIMPORTRAIT3D_ROOT
    / "pretrained_model"
    / "Realistic_Vision_V5.1_noVAE"
)
DEFAULT_VAE = (
    DEFAULT_ANIMPORTRAIT3D_ROOT
    / "pretrained_model"
    / "sd-vae-ft-ema"
)
DEFAULT_CONDITION_ASSETS = (
    Path("ckpts") / "animportrait3d_normal_seg_flame_teeth"
)
GENERIC_FULL_PROMPTS = {
    "a man",
    "a woman",
    "a person",
    "a human",
    "a portrait",
    "a head",
    "man",
    "woman",
    "person",
}


def validate_full_prompt(prompt: str) -> str:
    """Reject the generic prompt that silently broke the full pass."""

    prompt = " ".join(prompt.strip().split())
    if not prompt:
        raise ValueError("--prompt must not be empty")
    normalized = prompt.casefold().strip(" .,;:!?\t\r\n")
    if normalized in GENERIC_FULL_PROMPTS:
        raise ValueError(
            "Full refinement requires the per-identity appearance prompt "
            "used by AnimPortrait3D (hair, skin and clothing), not the "
            f"generic prompt {prompt!r}. A generic prompt gives SDEdit no "
            "target for repairing ears or clothing."
        )
    return prompt


def validate_abstract_prompt(
    abstract_prompt: str | None, full_prompt: str
) -> str:
    """Keep regional guidance tied to the identity-specific full prompt."""

    if abstract_prompt is None or not abstract_prompt.strip():
        return full_prompt
    prompt = " ".join(abstract_prompt.strip().split())
    normalized = prompt.casefold().strip(" .,;:!?\t\r\n")
    if normalized in GENERIC_FULL_PROMPTS:
        # Commands produced for the later train_all.py ablation explicitly
        # passed values such as ``--abstract-prompt 'a man'``.  Accept those
        # commands after restoring full_region_protected, but route them to
        # the detailed identity prompt used by that version.
        return full_prompt
    return prompt


def require_verified_mouth_sidecar(path: Path) -> int:
    """Return verified cumulative steps and reject no-op/smoke exports."""

    with np.load(path, allow_pickle=False) as archive:
        if "optimization_stage" not in archive.files:
            raise ValueError(
                f"Mouth sidecar has no optimization_stage provenance: {path}"
            )
        stage = str(np.asarray(archive["optimization_stage"]).reshape(-1)[0])
        if stage != "mouth":
            raise ValueError(
                f"Expected a mouth-stage sidecar, got {stage!r}: {path}"
            )
        if "optimizer_executed_steps" not in archive.files:
            raise ValueError(
                "Mouth sidecar predates optimizer-progress verification and "
                "may be one of the fp16 no-op results. Rerun train_mouth.py "
                f"with the repaired FP32 config before full refinement: {path}"
            )
        executed = int(
            np.asarray(archive["optimizer_executed_steps"]).reshape(-1)[0]
        )
        accounting_version = int(
            np.asarray(
                archive["optimizer_step_accounting_version"]
                if "optimizer_step_accounting_version" in archive.files
                else 1
            ).reshape(-1)[0]
        )
        if accounting_version >= 2:
            required_fields = {
                "optimizer_local_executed_steps",
                "optimizer_executed_step_offset",
            }
            missing = required_fields.difference(archive.files)
            if missing:
                raise ValueError(
                    "Mouth sidecar has incomplete optimizer-step accounting "
                    f"metadata ({', '.join(sorted(missing))}): {path}"
                )
            local_steps = int(
                np.asarray(
                    archive["optimizer_local_executed_steps"]
                ).reshape(-1)[0]
            )
            step_offset = int(
                np.asarray(
                    archive["optimizer_executed_step_offset"]
                ).reshape(-1)[0]
            )
            if (
                local_steps < 0
                or step_offset < 0
                or executed != local_steps + step_offset
            ):
                raise ValueError(
                    "Mouth sidecar has inconsistent cumulative optimizer-step "
                    f"accounting: total={executed}, local={local_steps}, "
                    f"offset={step_offset}: {path}"
                )

        if executed < MOUTH_REQUIRED_STEPS:
            raise ValueError(
                "Mouth refinement is incomplete: expected at least "
                f"{MOUTH_REQUIRED_STEPS} executed optimizer steps, found "
                f"{executed}. Do not use a first-phase smoke-test export for full "
                f"refinement: {path}"
            )
        return executed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reconstruction",
        type=Path,
        default=Path("outputs/reconstruction/00000001"),
        help="First-stage directory used for cameras and identity replay",
    )
    parser.add_argument(
        "--mouth-ply",
        type=Path,
        default=None,
        help=(
            "Mouth-pass UVD PLY (default: matching guidance/SDEdit output); "
            "pass one fixed PLY explicitly for a full-stage-only ablation"
        ),
    )
    parser.add_argument("--mouth-params", type=Path, default=None)
    parser.add_argument(
        "--prompt",
        required=True,
        help=(
            "Detailed per-identity full-avatar prompt; generic prompts such "
            "as 'A man' are rejected"
        ),
    )
    parser.add_argument(
        "--abstract-prompt",
        default=None,
        help="Short face description used by eye/mouth regional prompts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: method-specific full output)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reconstruction_full.yaml"),
    )
    parser.add_argument(
        "--diffusion-path", type=Path, default=DEFAULT_DIFFUSION
    )
    parser.add_argument(
        "--vae-path", type=Path, default=DEFAULT_VAE
    )
    parser.add_argument(
        "--controlnet-path", type=Path, default=DEFAULT_CONTROLNET
    )
    parser.add_argument(
        "--condition-assets", type=Path, default=DEFAULT_CONDITION_ASSETS
    )
    parser.add_argument(
        "--guidance-mode",
        choices=GUIDANCE_MODES,
        default="ism",
        help=(
            "First phase: original AnimPortrait3D ISM or CFD-consistent "
            "UVD-SFD"
        ),
    )
    parser.add_argument(
        "--sdedit-mode",
        choices=SDEDIT_MODES,
        default="independent",
        help=(
            "Second-phase teacher; 'flame-surface' jointly denoises matched "
            "FLAME surface tokens across views"
        ),
    )
    parser.add_argument(
        "--surface-views",
        type=int,
        default=DEFAULT_SURFACE_SDEDIT_VIEWS,
        help="Joint views per FLAME surface-consistent SDEdit step",
    )
    parser.add_argument(
        "--uvd-ism-seed",
        "--uvd-flow-seed",
        dest="uvd_flow_seed",
        type=int,
        default=0,
        help="Canonical UVD noise seed; the old option name remains an alias",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Override the total 1000-guidance + 750-SDEdit steps; values "
            "<=1000 run a first-phase-only smoke test"
        ),
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--disable-regional-guidance", action="store_true")
    parser.add_argument(
        "--densification-steps",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Enable UVD densification and set the first-phase optimizer "
            "steps that trigger it"
        ),
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=None,
        help="Override the full-stage densification point budget",
    )
    parser.add_argument(
        "--sdedit-only-smoke",
        action="store_true",
        help=(
            "Diagnostic: start directly in SDEdit; requires --max-steps and "
            "is not intended for a production refinement"
        ),
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reconstruction = resolve_path(args.reconstruction)
    require_stage1(reconstruction)
    mouth_run_name = refinement_run_name(
        "mouth", args.guidance_mode, args.sdedit_mode
    )
    default_mouth_dir = resolve_path(
        Path("outputs/refinement")
        / reconstruction.name
        / mouth_run_name
        / "save"
    )
    mouth_ply = resolve_path(args.mouth_ply or (default_mouth_dir / "mouth.ply"))
    mouth_params = resolve_path(
        args.mouth_params or mouth_ply.with_name(f"{mouth_ply.stem}_params.npz")
    )
    if not mouth_ply.is_file() or not mouth_params.is_file():
        raise FileNotFoundError(
            "Full refinement requires the mouth-pass PLY and sidecar: "
            f"{mouth_ply} / {mouth_params}"
        )
    require_verified_mouth_sidecar(mouth_params)
    output = (
        resolve_path(args.output)
        if args.output is not None
        else resolve_path(
            Path("outputs/refinement")
            / reconstruction.name
            / refinement_run_name(
                "full", args.guidance_mode, args.sdedit_mode
            )
        )
    )
    if (
        output.exists()
        and any(output.iterdir())
        and args.resume is None
        and not args.dry_run
    ):
        raise FileExistsError(
            f"Output directory is not empty: {output}. Run the repaired full "
            "stage in a fresh directory, or pass --resume only for a "
            "checkpoint produced by this implementation."
        )
    prompt = validate_full_prompt(args.prompt)
    abstract_prompt = validate_abstract_prompt(args.abstract_prompt, prompt)

    controlnet = resolve_path(args.controlnet_path)
    controlnet_config = controlnet / "config.json"
    if not controlnet_config.is_file():
        raise FileNotFoundError(
            f"AnimPortrait3D ControlNet config does not exist: {controlnet_config}"
        )
    with controlnet_config.open("r", encoding="utf-8") as file:
        controlnet_metadata = json.load(file)
    conditioning_channels = int(
        controlnet_metadata.get("conditioning_channels", -1)
    )
    if conditioning_channels != 4:
        raise ValueError(
            "Full refinement requires AnimPortrait3D's 4-channel normal+seg "
            f"ControlNet, but {controlnet_config} declares "
            f"conditioning_channels={conditioning_channels}"
        )

    condition_assets = resolve_path(args.condition_assets)
    required_condition_assets = (
        "face_region_faces.npy",
        "face_region_verts_mask.npy",
        "verts_seg.npy",
        "verts_seg_idxs.json",
    )
    missing_condition_assets = [
        condition_assets / name
        for name in required_condition_assets
        if not (condition_assets / name).is_file()
    ]
    if missing_condition_assets:
        raise FileNotFoundError(
            "AnimPortrait3D condition assets are incomplete; missing: "
            + ", ".join(str(path) for path in missing_condition_assets)
        )

    diffusion = resolve_path(args.diffusion_path)
    required_diffusion_files = (
        diffusion / "model_index.json",
        diffusion / "unet" / "config.json",
        diffusion / "scheduler" / "scheduler_config.json",
    )
    missing_diffusion_files = [
        path for path in required_diffusion_files if not path.is_file()
    ]
    if missing_diffusion_files:
        raise FileNotFoundError(
            "AnimPortrait3D diffusion checkpoint is incomplete; missing: "
            + ", ".join(str(path) for path in missing_diffusion_files)
        )
    vae = resolve_path(args.vae_path)
    if not (vae / "config.json").is_file():
        raise FileNotFoundError(
            f"AnimPortrait3D VAE config does not exist: {vae / 'config.json'}"
        )

    overrides = output_overrides(output)
    overrides.extend(
        guidance_mode_overrides(
            args.guidance_mode,
            uvd_flow_seed=args.uvd_flow_seed,
        )
    )
    overrides.extend(
        sdedit_mode_overrides(
            args.sdedit_mode,
            surface_views=args.surface_views,
        )
    )
    overrides.extend(
        [
            omega_override("data.reconstruction_dir", reconstruction),
            omega_override("system.initialization_dir", reconstruction),
            omega_override("system.initialization_ply", mouth_ply),
            omega_override("system.initialization_params", mouth_params),
            omega_override("system.optimization_stage", "full"),
            omega_override("system.prompt", prompt),
            omega_override("data.condition_type", "animportrait3d_normal_seg"),
            omega_override("data.use_condition", True),
            omega_override("data.animportrait3d_assets_dir", condition_assets),
            omega_override("system.guidance.control_type", "animportrait3d"),
            omega_override(
                "system.guidance.pretrained_controlnet_name_or_path",
                controlnet,
            ),
            omega_override(
                "system.guidance.controlnet_conditioning_channels", 4
            ),
            omega_override(
                "system.guidance.pretrained_model_name_or_path", diffusion
            ),
            omega_override(
                "system.guidance.pretrained_vae_name_or_path", vae
            ),
            omega_override(
                "system.guidance.ddim_scheduler_name_or_path", diffusion
            ),
            omega_override(
                "system.prompt_processor.pretrained_model_name_or_path",
                diffusion,
            ),
            omega_override(
                "system.regional_guidance.abstract_prompt", abstract_prompt
            ),
            omega_override(
                "system.regional_guidance.enabled",
                not args.disable_regional_guidance,
            ),
        ]
    )
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        if (
            args.sdedit_mode == "flame-surface"
            and args.max_steps <= ISM_STEPS
            and not args.sdedit_only_smoke
        ):
            raise ValueError(
                "--sdedit-mode flame-surface requires --max-steps to enter "
                f"the SDEdit phase (greater than {ISM_STEPS}), or use "
                "--sdedit-only-smoke"
            )
        ism_steps = min(args.max_steps, ISM_STEPS)
        schedule_end = max(ism_steps - 1, 1)
        overrides.extend(
            [
                omega_override("trainer.max_steps", args.max_steps),
                omega_override("system.optimization.max_steps", args.max_steps),
                "system.guidance.max_step_percent="
                f"[0,0.30,0.015,{schedule_end}]",
            ]
        )
        if args.max_steps <= ISM_STEPS:
            overrides.append(omega_override("system.sdedit.enabled", False))
    if args.densification_steps is not None:
        if any(step <= 0 for step in args.densification_steps):
            raise ValueError("--densification-steps must be positive")
        overrides.extend(
            [
                omega_override("system.densification.enabled", True),
                "system.densification.steps=["
                + ",".join(str(step) for step in args.densification_steps)
                + "]",
            ]
        )
    if args.max_gaussians is not None:
        if args.max_gaussians <= 0:
            raise ValueError("--max-gaussians must be positive")
        overrides.append(
            omega_override(
                "system.densification.max_gaussians", args.max_gaussians
            )
        )
    if args.sdedit_only_smoke:
        if args.max_steps is None:
            raise ValueError("--sdedit-only-smoke requires --max-steps")
        overrides.extend(
            [
                omega_override("system.sdedit.enabled", True),
                omega_override("system.sdedit.start_step", 0),
                omega_override("system.densification.enabled", False),
            ]
        )
    if args.resume is not None:
        resume = resolve_path(args.resume)
        if not resume.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {resume}")
        overrides.append(omega_override("resume", resume))

    status = run_launch(args.config, args.gpu, overrides, args.dry_run)
    if status != 0:
        raise SystemExit(status)
    if not args.dry_run:
        print(f"Full refinement: {output / 'save' / 'full.ply'}")


if __name__ == "__main__":
    main()
