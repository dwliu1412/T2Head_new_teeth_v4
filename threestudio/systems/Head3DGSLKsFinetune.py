"""Reconstruction-aware diffusion finetuning for FLAME-UVD Gaussians.

The Stage-1 reconstruction is treated as the centre of a constrained repair
problem, not as a disposable initialization:

    min_theta  L_diffusion(theta)
    s.t.       L_reference(theta) <= L_reference(theta_0) + epsilon
               theta stays in a small parameter-space trust region.

Only appearance is trainable at first.  Topology, FLAME shape, neck pose,
global pose, and translation never change.  Small face-bounded UV corrections,
normal offsets, scales, and rotations can be released later.  This keeps
animation binding intact while diffusion repairs high-frequency artifacts
under diverse expression poses.
"""

from __future__ import annotations

import math
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

import threestudio
from gaussiansplatting.arguments import PipelineParams
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene.gaussian_flame_face import GaussianFlameUVModel
from gaussiansplatting.utils.sh_utils import RGB2SH, SH2RGB
from surface_inpaint.layered_surface import (
    LayerSurfaceBuffers,
    SURFACE_LAYER_NAMES,
    compose_layered_surface,
    normalize_alpha_weighted,
)
from surface_inpaint.stability import stabilize_face_local_covariances
from threestudio.systems.base import BaseLift3DSystem
from train_reconstruction import (
    OpenCVCamera,
    apply_similarity_to_gaussians,
    aligned_scaling_rotation,
    save_world_ply,
)


@threestudio.register("head-3dgs-reconstruction-finetune-system")
class Head3DGSLKsReconstructionFinetune(BaseLift3DSystem):
    """Fine-tune a reconstructed avatar without destroying its rig."""

    @dataclass
    class Config(BaseLift3DSystem.Config):
        initialization_dir: str = ""
        # Optional chained initialization.  The mouth pass points these at the
        # Stage-1 files implicitly; the full pass can load the mouth export
        # while continuing to use initialization_dir for calibrated cameras
        # and static identity replay.
        initialization_ply: str = ""
        initialization_params: str = ""
        optimization_stage: str = "full"  # "mouth" or "full"
        prompt: str = ""

        diffusion_weight: float = 0.2
        diffusion_background_weight: float = 0.02
        guidance_warmup_steps: int = 300

        # Low-frequency replay defines the identity/structure constraint.
        reference_resolution: int = 128
        reference_kernel: int = 7
        reference_alpha_weight: float = 0.25
        reference_weight: float = 1.0
        reference_tolerance: float = 0.002
        reference_dual_lr: float = 10.0
        reference_max_weight: float = 10.0
        # Keep only the Stage-1 silhouette when RGB replay is disabled.  This
        # prevents diffusion from repairing texture by growing Gaussians into
        # the white background.
        silhouette_weight: float = 0.0
        silhouette_spill_weight: float = 0.0
        guidance_reference_dilation: int = 0

        # The chroma barrier complements conservative CFG and negative prompts.
        chroma_weight: float = 0.005
        chroma_threshold: float = 0.85

        # A loose global barrier catches new scale explosions without shrinking
        # legitimate hair/clothing Gaussians in the reconstruction.
        max_world_scale: float = 0.16
        world_scale_weight: float = 0.01
        # Full-stage stability is evaluated on aligned world-space axes. This
        # block configures the hard render guard plus per-point growth and
        # anisotropy envelopes; face-local scales have no shared threshold.
        scale_stability: dict = field(default_factory=dict)

        optimization: dict = field(default_factory=dict)
        proximal: dict = field(default_factory=dict)
        mouth: dict = field(default_factory=dict)
        # One-time pose-envelope repair plus a small in-training hard cap for
        # face-local ellipsoid outliers under the animated face scale.
        geometry_stability: dict = field(default_factory=dict)
        mouth_guidance_regions: list[str] = field(
            default_factory=lambda: ["lips", "teeth", "oral_cavity"]
        )
        # GSAvatar's mouth pass masks every optimizer gradient to its teeth
        # rows.  Keep the rendered lips/cavity in the guidance crop, but do
        # not let them drift during this dedicated pass.
        mouth_trainable_regions: list[str] = field(
            default_factory=lambda: ["teeth"]
        )
        mouth_guidance_dilation: int = 24
        freeze_dental_when_closed: bool = True
        # AnimPortrait3D feeds its mouth ControlNet a mouth bbox resized to
        # 512x512 in both ISM and SDEdit, rather than the complete portrait.
        guidance_crop_main: bool = False
        guidance_crop_padding: int = 8
        guidance_crop_square: bool = False
        # The reference mouth loss supervises the complete square crop.  A
        # rendered-alpha mask prevents new teeth from growing into currently
        # empty pixels, so mouth can explicitly opt out of that masking.
        guidance_full_crop_loss: bool = False
        mouth_crop_from_landmarks: bool = False
        mouth_crop_reference_size: int = 200
        mouth_crop_reference_resolution: int = 896
        # Optional masked face/eye/mouth guidance used by the AnimPortrait3D
        # full pass.  One diffusion model is shared; only prompt embeddings and
        # semantic masks differ between calls.
        regional_guidance: dict = field(default_factory=dict)
        # Renderer-side correspondence controls for UVD Surface-Flow
        # Distillation.  The diffusion/noise-volume controls live under
        # ``guidance``; these values determine which visible UVD surface is
        # considered unambiguous enough to supervise.
        uvd_surface_flow: dict = field(default_factory=dict)
        # The incoming mouth PLY is the source of truth for these fragile
        # regions.  Full refinement may render them as context, but should not
        # overwrite their point parameters or duplicate/prune their topology.
        # This is especially important for the UVD port: a zoomed CFG=100 eye
        # crop otherwise drives thousands of tiny eye splats towards opaque
        # blobs before SDEdit takes its fixed source snapshot.
        full_protection: dict = field(default_factory=dict)
        # Optional second phase matching AnimPortrait3D's SDEdit refinement.
        # The snapshot at start_step is held fixed as the img2img source.
        sdedit: dict = field(default_factory=dict)
        # Persist and evaluate the exact first-phase state before SDEdit can
        # update it. Mouth/full configs enable both the Gaussian export and a
        # complete driving-test render; old/custom configs remain opt-in.
        first_phase_artifacts: dict = field(default_factory=dict)
        # UVD-safe screen-gradient densification used by the full ISM phase.
        # GSAvatar densifies at optimizer iterations 50 and 100; keeping this
        # optional preserves fixed-topology mouth refinement and old configs.
        densification: dict = field(default_factory=dict)
        export_name: str = "stage2_last"

    cfg: Config

    # ------------------------------------------------------------------
    # Initialization: Stage-1 state is the immutable centre of the repair.
    # ------------------------------------------------------------------

    @staticmethod
    def _parameter(
        value: Any, width: int, device: torch.device
    ) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=device).reshape(1, -1)[:, :width]

    @staticmethod
    def _resolve_path(value: str, default: Path) -> Path:
        if not str(value).strip():
            return default
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else Path.cwd() / path

    def configure(self) -> None:
        root = Path(self.cfg.initialization_dir).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        self.optimization_stage = str(self.cfg.optimization_stage).lower()
        if self.optimization_stage not in {"mouth", "full"}:
            raise ValueError(
                "optimization_stage must be 'mouth' or 'full', got "
                f"{self.cfg.optimization_stage!r}"
            )
        if (
            float(self.cfg.silhouette_weight) < 0.0
            or float(self.cfg.silhouette_spill_weight) < 0.0
        ):
            raise ValueError(
                "silhouette weights must be non-negative"
            )
        if int(self.cfg.guidance_reference_dilation) < 0:
            raise ValueError(
                "guidance_reference_dilation must be non-negative"
            )
        if float(self.cfg.max_world_scale) <= 0.0:
            raise ValueError("max_world_scale must be positive")
        if float(self.cfg.world_scale_weight) < 0.0:
            raise ValueError("world_scale_weight must be non-negative")
        scale_stability = self.cfg.scale_stability
        if bool(scale_stability.get("enabled", False)):
            hard_world = float(
                scale_stability.get(
                    "hard_max_world_scale", self.cfg.max_world_scale
                )
            )
            if hard_world <= 0.0:
                raise ValueError(
                    "scale_stability.hard_max_world_scale must be positive"
                )
            for weight_name in (
                "world_anisotropy_weight",
                "reference_growth_weight",
                "reference_anisotropy_weight",
            ):
                if float(scale_stability.get(weight_name, 0.0)) < 0.0:
                    raise ValueError(
                        f"scale_stability.{weight_name} must be non-negative"
                    )
            for limit_name, weight_name in (
                ("reference_growth_limit", "reference_growth_weight"),
                (
                    "reference_anisotropy_growth_limit",
                    "reference_anisotropy_weight",
                ),
            ):
                if (
                    float(scale_stability.get(weight_name, 0.0)) > 0.0
                    and float(scale_stability.get(limit_name, 1.0)) < 1.0
                ):
                    raise ValueError(
                        f"scale_stability.{limit_name} must be at least 1"
                    )
            absolute_anisotropy = scale_stability.get(
                "max_world_anisotropy"
            )
            if absolute_anisotropy is not None and (
                not math.isfinite(float(absolute_anisotropy))
                or float(absolute_anisotropy) < 1.0
            ):
                raise ValueError(
                    "scale_stability.max_world_anisotropy must be finite and "
                    "at least 1"
                )
        if not isinstance(self.cfg.full_protection, Mapping):
            raise ValueError("full_protection must be a mapping")
        mouth_preservation = self.cfg.full_protection.get(
            "mouth_screen_preservation", {}
        )
        if not isinstance(mouth_preservation, Mapping):
            raise ValueError(
                "full_protection.mouth_screen_preservation must be a mapping"
            )
        if bool(mouth_preservation.get("enabled", False)):
            if not bool(self.cfg.full_protection.get("enabled", False)):
                raise ValueError(
                    "mouth screen preservation requires full_protection.enabled"
                )
            if not bool(self.cfg.full_protection.get("freeze_mouth", False)):
                raise ValueError(
                    "mouth screen preservation requires freeze_mouth=true"
                )
            if int(mouth_preservation.get("dilation", 0)) < 0:
                raise ValueError(
                    "mouth_screen_preservation.dilation must be non-negative"
                )
            for weight_name in (
                "rgb_weight",
                "alpha_weight",
                "depth_weight",
            ):
                if float(mouth_preservation.get(weight_name, 0.0)) < 0.0:
                    raise ValueError(
                        "mouth_screen_preservation."
                        f"{weight_name} must be non-negative"
                    )
            alpha_threshold = float(
                mouth_preservation.get("alpha_threshold", 0.05)
            )
            if not 0.0 < alpha_threshold < 1.0:
                raise ValueError(
                    "mouth_screen_preservation.alpha_threshold must be in "
                    "(0, 1)"
                )
        self.uvd_flow_enabled = bool(
            self.cfg.guidance.get("use_uvd_surface_flow", False)
        )
        if self.uvd_flow_enabled:
            configured_layers = int(
                self.cfg.guidance.get(
                    "uvd_flow_surface_layers", len(SURFACE_LAYER_NAMES)
                )
            )
            if configured_layers != len(SURFACE_LAYER_NAMES):
                raise ValueError(
                    "UVD-SFD requires exactly the five semantic layers "
                    f"{SURFACE_LAYER_NAMES}, got {configured_layers}"
                )
            surface_flow = self.cfg.uvd_surface_flow
            alpha_threshold = float(
                surface_flow.get("alpha_threshold", 0.05)
            )
            contribution_threshold = float(
                surface_flow.get("contribution_threshold", 0.01)
            )
            dominance_ratio = float(
                surface_flow.get("dominance_ratio", 1.10)
            )
            max_uvd_variance = float(
                surface_flow.get("max_uvd_variance", 0.0025)
            )
            d_padding = float(surface_flow.get("d_padding_ratio", 0.05))
            opacity_floor = float(surface_flow.get("opacity_floor", 0.0))
            if not 0.0 <= alpha_threshold <= 1.0:
                raise ValueError(
                    "uvd_surface_flow.alpha_threshold must be in [0, 1]"
                )
            if not 0.0 <= contribution_threshold <= 1.0:
                raise ValueError(
                    "uvd_surface_flow.contribution_threshold must be in [0, 1]"
                )
            if not math.isfinite(dominance_ratio) or dominance_ratio < 1.0:
                raise ValueError(
                    "uvd_surface_flow.dominance_ratio must be at least 1"
                )
            if (
                not math.isfinite(max_uvd_variance)
                or max_uvd_variance < 0.0
            ):
                raise ValueError(
                    "uvd_surface_flow.max_uvd_variance must be finite and "
                    "non-negative"
                )
            if not math.isfinite(d_padding) or d_padding < 0.0:
                raise ValueError(
                    "uvd_surface_flow.d_padding_ratio must be non-negative"
                )
            if not 0.0 <= opacity_floor <= 1.0:
                raise ValueError(
                    "uvd_surface_flow.opacity_floor must be in [0, 1]"
                )
        self.ism_accumulate_grad_batches = int(
            self.cfg.optimization.get("ism_accumulate_grad_batches", 1)
        )
        if self.ism_accumulate_grad_batches < 1:
            raise ValueError(
                "optimization.ism_accumulate_grad_batches must be positive"
            )
        # GSAvatar accumulates three samples only during ISM; SDEdit steps the
        # optimizer after every target. Lightning's trainer-wide accumulation
        # cannot express that phase change, so mouth uses manual optimization.
        self.automatic_optimization = self.ism_accumulate_grad_batches == 1
        self._manual_micro_step = 0
        self._manual_phase: Optional[str] = None
        self._consecutive_skipped_optimizer_steps = 0
        # Adam's per-parameter ``step`` is local to the current optimizer
        # state.  A configured SDEdit reset therefore keeps already executed
        # steps separately instead of silently restarting provenance at zero.
        self._optimizer_executed_step_offset = 0
        self._optimizer_stepped_this_batch = bool(
            self.automatic_optimization
        )
        self.sdedit_enabled = bool(self.cfg.sdedit.get("enabled", False))
        self.sdedit_start_step = int(self.cfg.sdedit.get("start_step", 0))
        self.sdedit_mode = str(
            self.cfg.sdedit.get("mode", "independent")
        ).strip().lower().replace("_", "-")
        if self.sdedit_mode not in {"independent", "flame-surface"}:
            raise ValueError(
                "sdedit.mode must be 'independent' or 'flame-surface'"
            )
        surface_memory = self.cfg.sdedit.get("surface_memory", {})
        if not isinstance(surface_memory, Mapping):
            raise ValueError("sdedit.surface_memory must be a mapping")
        configured_surface_memory = bool(
            surface_memory.get(
                "enabled", self.sdedit_mode == "flame-surface"
            )
        )
        if configured_surface_memory != (
            self.sdedit_mode == "flame-surface"
        ):
            raise ValueError(
                "sdedit.mode and sdedit.surface_memory.enabled disagree"
            )
        self.surface_sdedit_enabled = configured_surface_memory
        self.surface_memory_config = dict(surface_memory)
        if self.surface_sdedit_enabled:
            if not self.sdedit_enabled:
                raise ValueError(
                    "FLAME surface SDEdit requires sdedit.enabled=true"
                )
            views = int(surface_memory.get("views", 4))
            minimum_views = int(surface_memory.get("min_views", 2))
            maximum_memory_views = int(
                surface_memory.get("max_memory_views", views)
            )
            if views < 2:
                raise ValueError(
                    "sdedit.surface_memory.views must be at least two"
                )
            if not 2 <= minimum_views <= views:
                raise ValueError(
                    "sdedit.surface_memory.min_views must be in [2, views]"
                )
            if not minimum_views <= maximum_memory_views <= views:
                raise ValueError(
                    "sdedit.surface_memory.max_memory_views must be in "
                    "[min_views, views]"
                )
            if int(surface_memory.get("atlas_resolution", 64)) <= 0:
                raise ValueError(
                    "sdedit.surface_memory.atlas_resolution must be positive"
                )
            if int(surface_memory.get("max_tokens", 65536)) <= 0:
                raise ValueError(
                    "sdedit.surface_memory.max_tokens must be positive"
                )
            strength = float(surface_memory.get("strength", 0.65))
            start_progress = float(
                surface_memory.get("start_progress", 0.45)
            )
            end_progress = float(
                surface_memory.get("end_progress", 1.0)
            )
            if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
                raise ValueError(
                    "sdedit.surface_memory.strength must be in [0, 1]"
                )
            if not (
                math.isfinite(start_progress)
                and math.isfinite(end_progress)
                and 0.0 <= start_progress < end_progress <= 1.0
            ):
                raise ValueError(
                    "surface-memory progress must satisfy "
                    "0 <= start_progress < end_progress <= 1"
                )
            for key in (
                "alpha_threshold",
                "contribution_threshold",
                "variance_threshold",
                "opacity_floor",
            ):
                value = float(surface_memory.get(key, 0.0))
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"sdedit.surface_memory.{key} must be finite and "
                        "non-negative"
                    )
            for key in (
                "alpha_threshold",
                "contribution_threshold",
                "opacity_floor",
            ):
                if float(surface_memory.get(key, 0.0)) > 1.0:
                    raise ValueError(
                        f"sdedit.surface_memory.{key} must not exceed 1"
                    )
            dominance = float(
                surface_memory.get("dominance_ratio", 1.10)
            )
            if not math.isfinite(dominance) or dominance < 1.0:
                raise ValueError(
                    "sdedit.surface_memory.dominance_ratio must be at least 1"
                )
            depth_tolerance = surface_memory.get("depth_tolerance", None)
            if depth_tolerance is not None and (
                not math.isfinite(float(depth_tolerance))
                or float(depth_tolerance) < 0.0
            ):
                raise ValueError(
                    "sdedit.surface_memory.depth_tolerance must be null or "
                    "finite and non-negative"
                )
            patterns = surface_memory.get(
                "processor_patterns", ["up_blocks.2", "up_blocks.3"]
            )
            if (
                isinstance(patterns, (str, bytes))
                or not patterns
                or any(not str(value).strip() for value in patterns)
            ):
                raise ValueError(
                    "sdedit.surface_memory.processor_patterns must contain "
                    "one or more non-empty patterns"
                )
        if not isinstance(self.cfg.first_phase_artifacts, Mapping):
            raise ValueError("first_phase_artifacts must be a mapping")
        first_phase_fps = int(
            self.cfg.first_phase_artifacts.get("driving_fps", 30)
        )
        if first_phase_fps <= 0:
            raise ValueError(
                "first_phase_artifacts.driving_fps must be positive"
            )
        if bool(self.cfg.first_phase_artifacts.get("enabled", False)) and not (
            bool(
                self.cfg.first_phase_artifacts.get(
                    "save_gaussian", True
                )
            )
            or bool(
                self.cfg.first_phase_artifacts.get(
                    "render_driving_test", True
                )
            )
        ):
            raise ValueError(
                "Enabled first_phase_artifacts must save a Gaussian, render "
                "the driving test, or both"
            )
        densification_configured = bool(
            self.optimization_stage == "full"
            and self.cfg.densification.get("enabled", False)
        )
        # UVD noise is indexed by canonical coordinates, not Gaussian ids, so
        # the original full-stage densification is valid in both ablations.
        self.densification_enabled = densification_configured
        self._completed_densification_steps: set[int] = set()
        if self.densification_enabled:
            if self.automatic_optimization:
                raise ValueError(
                    "Full densification currently requires manual ISM "
                    "accumulation (ism_accumulate_grad_batches > 1)"
                )
            densification_steps = [
                int(value)
                for value in self.cfg.densification.get(
                    "steps", [50, 100]
                )
            ]
            if not densification_steps or any(
                value <= 0 for value in densification_steps
            ):
                raise ValueError(
                    "densification.steps must contain positive optimizer steps"
                )
            if len(set(densification_steps)) != len(densification_steps):
                raise ValueError("densification.steps must not contain duplicates")
            if self.sdedit_enabled and any(
                value >= self.sdedit_start_step
                for value in densification_steps
            ):
                raise ValueError(
                    "densification.steps must occur before sdedit.start_step"
                )
            self._densification_steps = frozenset(densification_steps)
        else:
            self._densification_steps = frozenset()
        if self.sdedit_enabled:
            maximum = int(self.cfg.optimization["max_steps"])
            strength = float(self.cfg.sdedit.get("strength", 0.3))
            inference_steps = int(
                self.cfg.sdedit.get("num_inference_steps", 20)
            )
            if not 0 <= self.sdedit_start_step < maximum:
                raise ValueError(
                    "sdedit.start_step must be in [0, optimization.max_steps)"
                )
            if not 0.0 < strength <= 1.0:
                raise ValueError("sdedit.strength must be in (0, 1]")
            if inference_steps <= 0:
                raise ValueError("sdedit.num_inference_steps must be positive")
            for key in ("weight", "l1_weight", "lpips_weight"):
                if float(self.cfg.sdedit.get(key, 0.0)) < 0.0:
                    raise ValueError(f"sdedit.{key} must be non-negative")
            region_weights = self.cfg.sdedit.get("region_weights", {})
            if not isinstance(region_weights, Mapping):
                raise ValueError("sdedit.region_weights must be a mapping")
            for name, value in region_weights.items():
                if float(value) < 0.0:
                    raise ValueError(
                        f"sdedit.region_weights.{name} must be non-negative"
                    )
            if int(self.cfg.sdedit.get("crop_padding", 0)) < 0:
                raise ValueError("sdedit.crop_padding must be non-negative")

        ply_path = self._resolve_path(
            self.cfg.initialization_ply, root / "model/uvd.ply"
        )
        if str(self.cfg.initialization_params).strip():
            sidecar_path = self._resolve_path(
                self.cfg.initialization_params,
                root / "model/reconstruction_params.npz",
            )
        elif str(self.cfg.initialization_ply).strip():
            chained_sidecar = ply_path.with_name(f"{ply_path.stem}_params.npz")
            sidecar_path = (
                chained_sidecar
                if chained_sidecar.is_file()
                else root / "model/reconstruction_params.npz"
            )
        else:
            sidecar_path = root / "model/reconstruction_params.npz"
        if not ply_path.is_file() or not sidecar_path.is_file():
            raise FileNotFoundError(
                "Missing reconstruction refinement initialization: "
                f"{ply_path} / {sidecar_path}"
            )

        self.initialization_ply = ply_path.resolve()
        self.initialization_params = sidecar_path.resolve()

        with np.load(sidecar_path, allow_pickle=False) as archive:
            sidecar = {key: np.asarray(archive[key]) for key in archive.files}

        device = torch.device("cuda")
        spatial_lr_scale = float(np.asarray(sidecar.get("spatial_lr_scale", 4.0)).reshape(-1)[0])
        flame_scale = float(np.asarray(sidecar.get("flame_scale", -10.0)).reshape(-1)[0])
        self.gaussian = GaussianFlameUVModel(sh_degree=0, device=str(device))
        self.gaussian.initialize_flame_state(spatial_lr_scale, flame_scale)
        self.gaussian.load_ply(str(ply_path))

        shape = self._parameter(sidecar["shape"], 300, device)
        expression = self._parameter(sidecar["expression"], 100, device)
        eyes = self._parameter(sidecar["eyes"], 6, device)
        if "jaw_pose" in sidecar:
            jaw = self._parameter(sidecar["jaw_pose"], 3, device)
        else:
            jaw = self._parameter(sidecar["pose"], 6, device)[:, 3:6]
        alignment = torch.as_tensor(sidecar["facelift_from_training"], dtype=torch.float32, device=device).reshape(4, 4)

        self.gaussian._shape.data.copy_(shape)
        self.gaussian._shape.requires_grad_(False)
        self.register_buffer("reference_shape", shape.clone())
        self.register_buffer("reference_expression", expression.clone())
        self.register_buffer("reference_jaw", jaw.clone())
        self.register_buffer("reference_leye", eyes[:, :3].clone())
        self.register_buffer("reference_reye", eyes[:, 3:6].clone())
        self.register_buffer("alignment", alignment.clone())

        self._set_reference_pose()
        self.mouth_slice: Optional[slice] = None
        self.teeth_slice: Optional[slice] = None
        mouth_points = int(self.cfg.mouth.get("points", 0))
        if mouth_points > 0:
            # Deterministic seeding makes checkpoint resume independent of the
            # global RNG state.
            with torch.random.fork_rng():
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                stats = self.gaussian.seed_flame_region(
                    region="oral_cavity",
                    num_points=mouth_points,
                    rgb=self.cfg.mouth["rgb"],
                    opacity=float(self.cfg.mouth["opacity"]),
                    min_world_scale=float(self.cfg.mouth["initial_scale_min"]),
                    max_world_scale=float(self.cfg.mouth["initial_scale_max"]),
                )
            self.mouth_slice = slice(stats["start"], stats["end"])
            threestudio.info(f"Seeded {stats['added']} articulated oral-cavity Gaussians.")

        # GSAvatar starts its mouth pass with roughly 152k dental points and
        # densifies them once.  A fixed-topology UVD pass cannot use its XYZ
        # densifier safely, so add the configured crown samples up front.
        teeth_points = int(self.cfg.mouth.get("teeth_points", 0))
        if teeth_points > 0:
            with torch.random.fork_rng():
                torch.manual_seed(1)
                torch.cuda.manual_seed_all(1)
                stats = self.gaussian.seed_flame_region(
                    region="teeth_crowns",
                    num_points=teeth_points,
                    rgb=self.cfg.mouth.get("teeth_rgb", [0.72, 0.70, 0.66]),
                    opacity=float(self.cfg.mouth.get("teeth_opacity", 0.55)),
                    min_world_scale=float(
                        self.cfg.mouth.get("teeth_scale_min", 2.5e-4)
                    ),
                    max_world_scale=float(
                        self.cfg.mouth.get("teeth_scale_max", 3.0e-3)
                    ),
                )
            self.teeth_slice = slice(stats["start"], stats["end"])
            threestudio.info(
                f"Seeded {stats['added']} fixed-topology tooth-crown Gaussians."
            )

        self._refresh_region_masks()
        if self._full_region_protection_enabled():
            protection = self.cfg.full_protection
            frozen_eye_points = (
                int(self.eye_point_mask.sum().item())
                if bool(protection.get("freeze_eyes", True))
                else 0
            )
            frozen_dental_points = (
                int(self.dental_point_mask.sum().item())
                if bool(protection.get("freeze_dental", True))
                else 0
            )
            frozen_mouth_points = (
                int(self.mouth_guidance_point_mask.sum().item())
                if bool(protection.get("freeze_mouth", False))
                else 0
            )
            threestudio.info(
                "Full-region protection: frozen eye points="
                f"{frozen_eye_points}, frozen mouth points="
                f"{frozen_mouth_points}, frozen dental points="
                f"{frozen_dental_points}; exclude protected topology from "
                "densification="
                f"{bool(protection.get('protect_from_densification', True))}."
            )
        self._bootstrap_dental_appearance()
        self.geometry_stability_report = self._apply_geometry_stability()
        self._active_open_mouth = True
        if self.optimization_stage == "mouth" and not bool(
            self.dental_point_mask.any().item()
        ):
            raise RuntimeError(
                "Mouth optimization requires teeth/oral-cavity Gaussians in "
                "the Stage-1 reconstruction"
            )
        dilation = int(self.cfg.mouth_guidance_dilation)
        if dilation < 0:
            raise ValueError("mouth_guidance_dilation must be non-negative")
        if int(self.cfg.guidance_crop_padding) < 0:
            raise ValueError("guidance_crop_padding must be non-negative")

        # These tensors are the proximal centre theta_0. Mouth UV can move at
        # a small learning rate but is projected back to its bound triangle.
        self.initial = {
            "uv": self.gaussian._uv.detach().clone(),
            "feature": self.gaussian._features_dc.detach().clone(),
            "opacity": self.gaussian._opacity.detach().clone(),
            "d": self.gaussian._d.detach().clone(),
            "scale": self.gaussian._scaling.detach().clone(),
            "rotation": F.normalize(self.gaussian._rotation.detach(), dim=-1),
        }
        self._configure_uvd_flow_d_range()
        # Continuous train_all.py cameras have no captured Stage-1 image/mask.
        # Keep the incoming mouth-pass state as an immutable, renderable target
        # so all silhouette/reference terms remain in the exact sampled view.
        self._initial_reference_state = {
            name: value.detach().clone() for name, value in self.initial.items()
        }
        self._initial_reference_state["face_idx"] = (
            self.gaussian._face_idx.detach().clone()
        )
        self._initial_reference_state["scale_trainable_mask"] = (
            self._active_trainable_point_mask().detach().clone()
        )
        self.register_buffer("reference_baseline", torch.tensor(float("nan"), device=device))
        self.register_buffer("reference_baseline_count", torch.tensor(0, dtype=torch.long, device=device))
        self.register_buffer("reference_dual", torch.tensor(float(self.cfg.reference_weight), device=device))
        self._reference_violation = torch.zeros((), device=device)

        self.background = torch.ones(3, device=device)
        parser = ArgumentParser(add_help=False)
        self.pipe = PipelineParams(parser)
        self.pipe.compute_cov3D_python = True
        self.pipe.convert_SHs_python = False
        self.pipe.debug = False

        self.radius = 4.0
        self.cameras_extent = spatial_lr_scale
        self.guidance = None
        self.prompt_processor = None
        self.sdedit_prompt_processor = None
        self.regional_prompt_processors = {}
        self.sdedit_regional_prompt_processors = {}
        self._pending_vsd_state = None
        self._pending_uvd_flow_state = None
        self._sdedit_reference_state: Optional[dict[str, torch.Tensor]] = None
        self._sdedit_optimizer_state_reset = False
        self._first_phase_artifacts_written = False
        self._first_phase_optimizer_steps = -1
        self.sdedit_perceptual: Optional[nn.Module] = None
        self._sdedit_lpips_fallback = False
        self._sdedit_control_released = False
        self._last_geometry_projection_step = -1

        threestudio.info(
            f"Loaded {self.gaussian.num_gs} reconstructed UVD Gaussians from "
            f"{self.initialization_ply}; refinement stage={self.optimization_stage}, "
            "identity shape/rig/neck/global pose are fixed; full-stage UVD "
            "topology may densify when configured."
        )

    # ------------------------------------------------------------------
    # FLAME state, aligned geometry, and calibrated rendering.
    # ------------------------------------------------------------------

    def _safe_point_region_mask(
        self,
        regions: Any,
        *,
        face_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Union available FLAME point regions without hiding bad configs."""

        if isinstance(regions, str):
            regions = [regions]
        point_count = (
            self.gaussian.num_gs
            if face_idx is None
            else int(torch.as_tensor(face_idx).numel())
        )
        mask = torch.zeros(
            point_count,
            dtype=torch.bool,
            device=self.gaussian.device,
        )
        missing = []
        for region in regions:
            try:
                if face_idx is None:
                    region_mask = self.gaussian.point_region_mask(str(region))
                else:
                    # Evaluate frozen snapshots directly through their saved
                    # face binding. Do not require an extended Gaussian-model
                    # method signature: this keeps the surface-SDEdit system
                    # compatible with existing GaussianFlameUVModel builds.
                    region_mask = self._point_region_mask_for_binding(
                        str(region), face_idx
                    )
                mask |= region_mask
            except (AttributeError, KeyError):
                missing.append(str(region))
        if missing:
            threestudio.warn(
                "FLAME refinement mask does not define region(s): "
                + ", ".join(missing)
            )
        return mask

    def _point_region_mask_for_binding(
        self,
        region: str,
        face_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Map one FLAME face region through an immutable point binding."""

        binding = torch.as_tensor(
            face_idx,
            dtype=torch.long,
            device=self.gaussian.device,
        )
        if binding.ndim != 1:
            raise ValueError("Frozen face_idx must be one-dimensional")
        region_faces = self.gaussian.model.mask.get_fid_by_region(
            [str(region)]
        ).to(device=self.gaussian.device, dtype=torch.long)
        face_count = int(self.gaussian._faces.shape[0])
        if bool(((binding < 0) | (binding >= face_count)).any().item()):
            raise ValueError("Frozen face_idx contains an out-of-range FLAME face")
        if region_faces.numel() == 0 or binding.numel() == 0:
            return torch.zeros_like(binding, dtype=torch.bool)
        region_faces = region_faces[
            (region_faces >= 0) & (region_faces < face_count)
        ]
        face_lookup = torch.zeros(
            face_count,
            dtype=torch.bool,
            device=self.gaussian.device,
        )
        face_lookup[region_faces] = True
        return face_lookup[binding]

    def _configure_uvd_flow_d_range(self) -> None:
        """Fix one canonical normal-offset range for the complete trajectory."""

        if not self.uvd_flow_enabled:
            self.uvd_flow_d_range = None
            return
        configured = self.cfg.uvd_surface_flow.get("d_range")
        if configured is not None:
            values = torch.as_tensor(
                configured,
                dtype=torch.float32,
                device=self.gaussian.device,
            ).reshape(-1)
            if values.numel() != 2:
                raise ValueError(
                    "uvd_surface_flow.d_range must contain [minimum, maximum]"
                )
            lower, upper = float(values[0]), float(values[1])
        else:
            initial_d = self.initial["d"].detach().float()
            if not bool(torch.isfinite(initial_d).all().item()):
                raise ValueError(
                    "Initial UVD normal offsets contain non-finite values"
                )
            lower = float(initial_d.min().item())
            upper = float(initial_d.max().item())
            span = max(upper - lower, 1.0e-4)
            padding = span * float(
                self.cfg.uvd_surface_flow.get("d_padding_ratio", 0.05)
            )
            lower -= padding
            upper += padding
        if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
            raise ValueError(
                "UVD-SFD canonical d range must contain two increasing finite values"
            )
        self.uvd_flow_d_range = torch.tensor(
            (lower, upper),
            dtype=torch.float32,
            device=self.gaussian.device,
        )
        threestudio.info(
            f"[UVD-SFD] Fixed canonical d range [{lower:.6g}, {upper:.6g}]"
        )

    def _canonical_surface_masks(
        self,
        *,
        face_idx: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Return the five-way FLAME partition shared by surface methods."""

        upper = self._safe_point_region_mask(
            "teeth_upper", face_idx=face_idx
        )
        lower = self._safe_point_region_mask(
            "teeth_lower", face_idx=face_idx
        )
        if bool((upper & lower).any().item()):
            raise RuntimeError(
                "FLAME upper/lower tooth layers overlap and have no unique "
                "surface identity"
            )
        # FLAME region masks intentionally share boundary vertices/faces in
        # some assets. Resolve them with the same semantic priority as the
        # repository's layered UVD renderer: teeth > cavity > lips > face.
        cavity = self._safe_point_region_mask(
            "oral_cavity", face_idx=face_idx
        ) & ~(
            upper | lower
        )
        lips = self._safe_point_region_mask("lips", face_idx=face_idx) & ~(
            upper | lower | cavity
        )
        oral = {
            "lips": lips,
            "teeth_upper": upper,
            "teeth_lower": lower,
            "oral_cavity": cavity,
        }
        for name, mask in oral.items():
            if not bool(mask.any().item()):
                raise RuntimeError(
                    f"Canonical memory requires a non-empty FLAME surface "
                    f"layer {name!r}"
                )
        oral_union = torch.zeros(
            upper.numel(),
            dtype=torch.bool,
            device=self.gaussian.device,
        )
        for mask in oral.values():
            oral_union |= mask
        masks = {"face": ~oral_union, **oral}
        assigned = torch.stack(
            [masks[name].to(torch.int32) for name in SURFACE_LAYER_NAMES]
        ).sum(dim=0)
        if not torch.equal(assigned, torch.ones_like(assigned)):
            raise RuntimeError(
                "FLAME semantic layers must partition every Gaussian exactly once"
            )
        return masks

    def _uvd_flow_surface_masks(self) -> dict[str, torch.Tensor]:
        """Return the UVD-SFD semantic surface partition."""

        if not self.uvd_flow_enabled:
            raise RuntimeError(
                "UVD surface layers requested while UVD-SFD is disabled"
            )
        return self._canonical_surface_masks()

    def _refresh_region_masks(self) -> None:
        self.dental_point_mask = self._safe_point_region_mask(
            ["teeth", "oral_cavity"]
        )
        self.mouth_trainable_point_mask = self._safe_point_region_mask(
            self.cfg.mouth_trainable_regions
        )
        if not bool(self.mouth_trainable_point_mask.any().item()):
            raise RuntimeError(
                "Mouth refinement has no Gaussians in "
                f"mouth_trainable_regions={self.cfg.mouth_trainable_regions!r}"
            )
        self.tooth_crown_point_mask = self._safe_point_region_mask(
            "teeth_crowns"
        )
        if not bool(self.tooth_crown_point_mask.any().item()):
            # Older FLAME masks expose only the coarser teeth region.
            self.tooth_crown_point_mask = self._safe_point_region_mask(
                "teeth"
            )
        self.mouth_guidance_point_mask = self._safe_point_region_mask(
            self.cfg.mouth_guidance_regions
        )
        if not bool(self.mouth_guidance_point_mask.any().item()):
            self.mouth_guidance_point_mask = self.dental_point_mask.clone()
        self.regional_point_masks = {
            "face": self._safe_point_region_mask("face"),
            "left_eye": self._safe_point_region_mask("left_eye"),
            "right_eye": self._safe_point_region_mask("right_eye"),
            "mouth": self.mouth_guidance_point_mask,
        }
        self.eye_point_mask = (
            self.regional_point_masks["left_eye"]
            | self.regional_point_masks["right_eye"]
        )
        if (
            self._full_region_protection_enabled()
            and bool(self.cfg.full_protection.get("freeze_eyes", True))
            and not bool(self.eye_point_mask.any().item())
        ):
            raise RuntimeError(
                "Full-region protection requested freeze_eyes, but the "
                "FLAME mask contains no eye Gaussians"
            )

    @torch.no_grad()
    def _bootstrap_dental_appearance(self) -> None:
        """Give unsupervised Stage-1 tooth crowns a visible learning prior."""

        bootstrap = self.cfg.mouth.get("bootstrap", {})
        if not bool(bootstrap.get("enabled", False)):
            return
        mask = self.tooth_crown_point_mask
        if not bool(mask.any().item()):
            return
        rgb = torch.as_tensor(
            bootstrap.get("rgb", [0.72, 0.70, 0.66]),
            dtype=self.gaussian._features_dc.dtype,
            device=self.gaussian.device,
        ).reshape(1, 3).clamp(0.0, 1.0)
        opacity = float(bootstrap.get("opacity", 0.55))
        if not 0.0 < opacity < 1.0:
            raise ValueError("mouth.bootstrap.opacity must be in (0, 1)")
        self.gaussian._features_dc.data[mask, 0, :] = RGB2SH(rgb)
        opacity_tensor = torch.full(
            (int(mask.sum().item()), 1),
            opacity,
            dtype=self.gaussian._opacity.dtype,
            device=self.gaussian.device,
        )
        self.gaussian._opacity.data[mask] = torch.logit(opacity_tensor)
        if bool(bootstrap.get("reset_d", True)):
            self.gaussian._d.data[mask] = 0.0
        threestudio.info(
            "Bootstrapped appearance/opacity for "
            f"{int(mask.sum().item())} tooth-crown Gaussians."
        )

    def _stability_pose_envelope(
        self,
    ) -> tuple[list[tuple[str, tuple[torch.Tensor, ...]]], tuple[torch.Tensor, ...]]:
        cfg = self.cfg.geometry_stability
        expression_path = self._resolve_path(
            str(cfg.get("expression_path", "")),
            Path.cwd() / "assets" / "open_mouth_exp.npy",
        )
        pose_path = self._resolve_path(
            str(cfg.get("pose_path", "")),
            Path.cwd() / "assets" / "open_mouth_pose.npy",
        )
        if not expression_path.is_file() or not pose_path.is_file():
            raise FileNotFoundError(
                "geometry_stability requires paired expression/pose files: "
                f"{expression_path} / {pose_path}"
            )
        expression = np.asarray(
            np.load(expression_path, allow_pickle=False), dtype=np.float32
        )
        pose = np.asarray(
            np.load(pose_path, allow_pickle=False), dtype=np.float32
        )
        if expression.ndim != 2 or expression.shape[1] < 100:
            raise ValueError(
                f"{expression_path}: expected shape (T, >=100), got "
                f"{expression.shape}"
            )
        if pose.ndim != 2 or pose.shape[1] < 15:
            raise ValueError(
                f"{pose_path}: expected shape (T, >=15), got {pose.shape}"
            )
        frame_count = min(expression.shape[0], pose.shape[0])
        if frame_count == 0:
            raise ValueError("geometry_stability pose envelope is empty")
        expression = expression[:frame_count, :100]
        pose = pose[:frame_count, :15]
        if not np.isfinite(expression).all() or not np.isfinite(pose).all():
            raise ValueError("geometry_stability pose envelope is non-finite")

        reference = (
            self.reference_expression,
            self.reference_jaw,
            self.reference_leye,
            self.reference_reye,
        )
        named: list[tuple[str, tuple[torch.Tensor, ...]]] = [
            ("reference", reference)
        ]
        jaw_x = pose[:, 6]
        quantiles = cfg.get("jaw_quantiles", [0.5, 0.9, 1.0])
        seen: set[int] = set()
        for value in quantiles:
            quantile = float(value)
            if not 0.0 <= quantile <= 1.0:
                raise ValueError(
                    "geometry_stability.jaw_quantiles must lie in [0, 1]"
                )
            target = float(np.quantile(jaw_x, quantile))
            index = int(np.abs(jaw_x - target).argmin())
            if index in seen:
                continue
            seen.add(index)
            tensors = (
                torch.from_numpy(expression[index:index + 1]).to(
                    self.gaussian.device
                ),
                torch.from_numpy(pose[index:index + 1, 6:9]).to(
                    self.gaussian.device
                ),
                torch.from_numpy(pose[index:index + 1, 9:12]).to(
                    self.gaussian.device
                ),
                torch.from_numpy(pose[index:index + 1, 12:15]).to(
                    self.gaussian.device
                ),
            )
            named.append((f"jaw_q{quantile:.2f}_frame_{index}", tensors))
        return named, reference

    @torch.no_grad()
    def _cap_region_over_pose_envelope(
        self,
        point_mask: torch.Tensor,
        named_poses: list[tuple[str, tuple[torch.Tensor, ...]]],
        reference_pose: tuple[torch.Tensor, ...],
        maximum: float,
        passes: int,
    ) -> dict[str, Any]:
        if maximum <= 0.0 or passes < 1:
            raise ValueError("Dental stability cap and passes must be positive")
        mask = point_mask.to(self.gaussian.device, dtype=torch.bool)
        updated_union = torch.zeros_like(mask)
        updates = 0
        before_max = 0.0
        try:
            for pass_index in range(passes):
                pass_updates = 0
                for _, pose in named_poses:
                    self._set_pose(*pose)
                    world_scale = (
                        self.gaussian.get_world_scale()[:, 0]
                        * self._scene_similarity_scale()
                    )
                    if pass_index == 0 and bool(mask.any().item()):
                        before_max = max(
                            before_max,
                            float(world_scale[mask].max().item()),
                        )
                    ratio = (maximum / world_scale.clamp_min(1.0e-12)).clamp(
                        max=1.0
                    )
                    selected = mask & (ratio < 1.0)
                    count = int(selected.sum().item())
                    if count:
                        indices = torch.nonzero(
                            selected, as_tuple=False
                        ).squeeze(1)
                        updated_scaling = self.gaussian._scaling.data.index_select(
                            0, indices
                        ) + ratio.index_select(0, indices)[:, None].log()
                        self.gaussian._scaling.data.index_copy_(
                            0, indices, updated_scaling
                        )
                        updated_union |= selected
                        updates += count
                        pass_updates += count
                if pass_updates == 0:
                    break

            after_max = 0.0
            for _, pose in named_poses:
                self._set_pose(*pose)
                scale = (
                    self.gaussian.get_world_scale()[:, 0]
                    * self._scene_similarity_scale()
                )
                if bool(mask.any().item()):
                    after_max = max(after_max, float(scale[mask].max().item()))
        finally:
            self._set_pose(*reference_pose)
        # A hard projection must also erase Adam moments for the projected
        # rows; otherwise the next optimizer step immediately recreates the
        # over-sized covariance that caused the streak.
        self._clear_parameter_optimizer_rows(
            self.gaussian._scaling, updated_union
        )
        return {
            "maximum": float(maximum),
            "before_max": float(before_max),
            "after_max": float(after_max),
            "updates": int(updates),
            "unique_updated": int(updated_union.sum().item()),
        }

    @torch.no_grad()
    def _apply_geometry_stability(self) -> dict[str, Any]:
        cfg = self.cfg.geometry_stability
        if not bool(cfg.get("enabled", False)):
            return {"enabled": False}
        named_poses, reference = self._stability_pose_envelope()
        global_cfg = cfg.get("global", {})
        global_report = stabilize_face_local_covariances(
            self.gaussian,
            named_poses,
            lambda pose: self._set_pose(*pose),
            global_cfg,
            reference_pose=reference,
            world_scale_multiplier=self._scene_similarity_scale(),
        )
        dental_report = self._cap_region_over_pose_envelope(
            self.dental_point_mask,
            named_poses,
            reference,
            maximum=float(cfg.get("dental_max_world_scale", 0.01)),
            passes=int(cfg.get("dental_passes", 3)),
        )
        before = global_report.get("before") or {}
        after = global_report.get("after") or {}
        threestudio.info(
            "Pose-envelope covariance stabilization: global max "
            f"{float(before.get('max_s0', 0.0)):.6f} -> "
            f"{float(after.get('max_s0', 0.0)):.6f}, "
            f"global unique repairs={global_report.get('unique_updated', 0)}; "
            "dental max "
            f"{dental_report['before_max']:.6f} -> "
            f"{dental_report['after_max']:.6f}, "
            f"dental unique caps={dental_report['unique_updated']}."
        )
        return {
            "enabled": True,
            "global": global_report,
            "dental": dental_report,
        }

    def _set_pose(
        self,
        expression: torch.Tensor,
        jaw: torch.Tensor,
        leye: torch.Tensor,
        reye: torch.Tensor,
    ) -> None:
        zeros = torch.zeros((1, 3), dtype=torch.float32, device=self.gaussian.device)
        self.gaussian._expression = expression.float()
        self.gaussian._jaw_pose = jaw.float()
        self.gaussian._leye_pose = leye.float()
        self.gaussian._reye_pose = reye.float()
        self.gaussian._global_orient = zeros
        self.gaussian._neck_pose = zeros
        self.gaussian._translation = zeros

    def _scene_similarity_scale(self) -> float:
        """Uniform FLAME-to-camera-scene scale from the saved alignment."""

        linear = self.alignment[:3, :3]
        return float(linear.square().sum().div(3.0).sqrt().item())

    def _set_reference_pose(self) -> None:
        self._set_pose(self.reference_expression, self.reference_jaw, self.reference_leye, self.reference_reye)

    def _batch_pose(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, ...]:
        device = self.gaussian.device
        return (
            self._parameter(batch["expression"], 100, device),
            self._parameter(batch["jaw_pose"], 3, device),
            self._parameter(batch["leye_pose"], 3, device),
            self._parameter(batch["reye_pose"], 3, device),
        )

    def _aligned_render_geometry(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pose FLAME, then produce final-scene scale/rotation for rendering."""

        with torch.cuda.amp.autocast(enabled=False):
            return aligned_scaling_rotation(self.gaussian, self.alignment)

    @torch.no_grad()
    def _reference_world_scales_current_pose(
        self, current_world_scales: torch.Tensor
    ) -> torch.Tensor:
        """Recover incoming-avatar world axes under the current FLAME pose.

        Face deformation and scene alignment multiply current and initial
        local axes by the same factors. Replacing the detached current local
        scale with the immutable incoming scale therefore gives the exact
        same-pose world reference without evaluating FLAME a second time.
        """

        if self.initial["scale"].shape != self.gaussian._scaling.shape:
            raise RuntimeError(
                "World-scale reference topology differs from the Gaussian model"
            )
        current_local = self.gaussian.scaling_activation(
            self.gaussian._scaling.detach()
        ).clamp_min(1.0e-12)
        reference_local = self.gaussian.scaling_activation(
            self.initial["scale"]
        ).clamp_min(1.0e-12)
        return current_world_scales.detach() * (
            reference_local / current_local
        )

    def _full_world_scale_axis_ratio(
        self,
        scales: torch.Tensor,
        trainable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-axis shrink ratios for absolute world-space guards."""

        maximum = float(
            self.cfg.scale_stability.get(
                "hard_max_world_scale", self.cfg.max_world_scale
            )
        )
        finite_scales = torch.nan_to_num(
            scales,
            nan=maximum * 2.0,
            posinf=maximum * 2.0,
            neginf=0.0,
        )
        ratio = (
            maximum / finite_scales.clamp_min(1.0e-12)
        ).clamp(max=1.0)
        maximum_anisotropy = self.cfg.scale_stability.get(
            "max_world_anisotropy"
        )
        if maximum_anisotropy is not None:
            # Shrink long axes instead of inflating thin axes. This removes
            # needle support without making a splat thicker or more opaque.
            shortest = finite_scales.detach().amin(dim=-1, keepdim=True)
            anisotropy_ratio = (
                shortest
                * float(maximum_anisotropy)
                / finite_scales.clamp_min(1.0e-12)
            ).clamp(max=1.0)
            ratio = torch.minimum(ratio, anisotropy_ratio)
        if self._full_region_protection_enabled():
            if trainable_mask is None:
                trainable_mask = self._active_trainable_point_mask()
            trainable = torch.as_tensor(
                trainable_mask,
                device=scales.device, dtype=torch.bool
            ).reshape(-1)
            if trainable.shape[0] != scales.shape[0]:
                raise ValueError(
                    "Full scale-guard mask topology does not match the "
                    f"rendered Gaussian topology: mask={trainable.shape[0]}, "
                    f"scales={scales.shape[0]}"
                )
            ratio = torch.where(
                trainable[:, None], ratio, torch.ones_like(ratio)
            )
        return ratio

    def _cap_full_render_world_scales(
        self, scales: torch.Tensor
    ) -> torch.Tensor:
        """Apply non-mutating axis and anisotropy guards in every FLAME pose."""

        if not self._full_scale_stability_enabled():
            return scales
        return scales * self._full_world_scale_axis_ratio(scales)

    def _cameras(
        self, batch: Mapping[str, Any]
    ) -> list[OpenCVCamera]:
        w2c = torch.as_tensor(batch["w2c"], dtype=torch.float32, device=self.gaussian.device)
        K = torch.as_tensor(batch["K"], dtype=torch.float32, device=self.gaussian.device)
        if w2c.ndim == 2:
            w2c, K = w2c[None], K[None]
        width = int(torch.as_tensor(batch["width"]).reshape(-1)[0])
        height = int(torch.as_tensor(batch["height"]).reshape(-1)[0])
        return [
            OpenCVCamera(
                {
                    "w": width,
                    "h": height,
                    "fx": float(K[index, 0, 0]),
                    "fy": float(K[index, 1, 1]),
                    "w2c": w2c[index],
                },
                self.gaussian.device,
            )
            for index in range(w2c.shape[0])
        ]

    @torch.no_grad()
    def _mouth_landmark_crop_boxes(
        self, batch: Mapping[str, Any]
    ) -> Optional[torch.Tensor]:
        """Project FLAME mouth corners and reproduce GSAvatar's fixed crop."""

        if not bool(self.cfg.mouth_crop_from_landmarks):
            return None
        result = self.gaussian.model(
            shape=self.gaussian._shape,
            expr=self.gaussian._expression,
            rotation=self.gaussian._global_orient,
            neck=self.gaussian._neck_pose,
            jaw=self.gaussian._jaw_pose,
            eyes=torch.cat(
                (self.gaussian._leye_pose, self.gaussian._reye_pose), dim=-1
            ),
            translation=self.gaussian._translation,
            zero_centered_at_root_node=False,
            return_landmarks=True,
        )
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise RuntimeError("FLAME did not return landmarks for mouth crop")
        landmarks = result[-1][0]
        if landmarks.shape[0] <= 54:
            raise RuntimeError(
                "FLAME mouth crop requires at least 55 facial landmarks"
            )
        landmarks = self.gaussian._normalize_flame_vertices(landmarks)
        linear = self.alignment[:3, :3]
        translation = self.alignment[:3, 3]
        landmarks = landmarks @ linear.T + translation

        device = self.gaussian.device
        w2c = torch.as_tensor(
            batch["w2c"], dtype=torch.float32, device=device
        )
        intrinsics = torch.as_tensor(
            batch["K"], dtype=torch.float32, device=device
        )
        if w2c.ndim == 2:
            w2c, intrinsics = w2c[None], intrinsics[None]
        camera_points = torch.einsum(
            "bij,nj->bni", w2c[:, :3, :3], landmarks
        ) + w2c[:, None, :3, 3]
        projected_h = torch.einsum(
            "bij,bnj->bni", intrinsics, camera_points
        )
        depth = projected_h[..., 2]
        if bool((depth[:, [48, 54]] <= 1.0e-6).any().item()):
            raise RuntimeError("Projected FLAME mouth landmarks are behind camera")
        projected = projected_h[..., :2] / projected_h[..., 2:3]
        centers = projected[:, [48, 54]].mean(dim=1)

        width = int(torch.as_tensor(batch["width"]).reshape(-1)[0])
        height = int(torch.as_tensor(batch["height"]).reshape(-1)[0])
        reference_size = int(self.cfg.mouth_crop_reference_size)
        reference_resolution = int(self.cfg.mouth_crop_reference_resolution)
        if reference_size <= 0 or reference_resolution <= 0:
            raise ValueError(
                "mouth crop reference size/resolution must be positive"
            )
        side = max(
            int(round(min(width, height) * reference_size / reference_resolution)),
            1,
        )
        side = min(side, width, height)
        x0 = (centers[:, 0] - side / 2.0).floor().long()
        y0 = (centers[:, 1] - side / 2.0).floor().long()
        x0 = x0.clamp(0, width - side)
        y0 = y0.clamp(0, height - side)
        return torch.stack((x0, y0, x0 + side, y0 + side), dim=-1)

    @torch.no_grad()
    def _full_regional_crop_boxes(
        self, batch: Mapping[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Reproduce train_all.py's fixed landmark-centred regional crops.

        The reference implementation renders at 896 px and uses square crops
        of 512 px for the face, 116 px for either eye and 200 px for the
        mouth.  The local renderer may use another resolution, so the side
        lengths are scaled before the crops are resized to the diffusion
        model's 512 px input.
        """

        regional = self.cfg.regional_guidance
        if not bool(regional.get("crop_from_landmarks", False)):
            return {}
        result = self.gaussian.model(
            shape=self.gaussian._shape,
            expr=self.gaussian._expression,
            rotation=self.gaussian._global_orient,
            neck=self.gaussian._neck_pose,
            jaw=self.gaussian._jaw_pose,
            eyes=torch.cat(
                (self.gaussian._leye_pose, self.gaussian._reye_pose), dim=-1
            ),
            translation=self.gaussian._translation,
            zero_centered_at_root_node=False,
            return_landmarks=True,
        )
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise RuntimeError("FLAME did not return landmarks for regional crops")
        landmarks = result[-1][0]
        if landmarks.shape[0] <= 54:
            raise RuntimeError(
                "Regional crops require at least 55 FLAME facial landmarks"
            )
        landmarks = self.gaussian._normalize_flame_vertices(landmarks)
        landmarks = (
            landmarks @ self.alignment[:3, :3].T
            + self.alignment[:3, 3]
        )

        device = self.gaussian.device
        w2c = torch.as_tensor(
            batch["w2c"], dtype=torch.float32, device=device
        )
        intrinsics = torch.as_tensor(
            batch["K"], dtype=torch.float32, device=device
        )
        if w2c.ndim == 2:
            w2c, intrinsics = w2c[None], intrinsics[None]
        camera_points = torch.einsum(
            "bij,nj->bni", w2c[:, :3, :3], landmarks
        ) + w2c[:, None, :3, 3]
        projected_h = torch.einsum(
            "bij,bnj->bni", intrinsics, camera_points
        )
        depth = projected_h[..., 2]
        if bool((depth <= 1.0e-6).all(dim=1).any().item()):
            raise RuntimeError("Projected FLAME landmarks are behind camera")
        projected = projected_h[..., :2] / projected_h[..., 2:3].clamp_min(
            1.0e-6
        )

        # GSAvatar appends two eye centres after the standard 68 landmarks.
        # Keep a contour-centre fallback for FLAME variants exposing only 68.
        if projected.shape[1] >= 70:
            left_eye = projected[:, 68]
            right_eye = projected[:, 69]
        elif projected.shape[1] >= 48:
            left_eye = projected[:, 36:42].mean(dim=1)
            right_eye = projected[:, 42:48].mean(dim=1)
        else:
            raise RuntimeError(
                "Regional eye crops require eye landmarks or eye centres"
            )
        centers = {
            "face": 0.5 * (left_eye + right_eye),
            "left_eye": left_eye,
            "right_eye": right_eye,
            "mouth": projected[:, [48, 54]].mean(dim=1),
        }
        default_sizes = {
            "face": 512,
            "left_eye": 116,
            "right_eye": 116,
            "mouth": 200,
        }
        configured_sizes = regional.get("crop_reference_sizes", {})
        reference_resolution = int(
            regional.get("crop_reference_resolution", 896)
        )
        if reference_resolution <= 0:
            raise ValueError(
                "regional_guidance.crop_reference_resolution must be positive"
            )
        width = int(torch.as_tensor(batch["width"]).reshape(-1)[0])
        height = int(torch.as_tensor(batch["height"]).reshape(-1)[0])
        boxes: dict[str, torch.Tensor] = {}
        for name, center in centers.items():
            reference_size = int(
                configured_sizes.get(name, default_sizes[name])
            )
            if reference_size <= 0:
                raise ValueError(
                    f"regional crop size for {name!r} must be positive"
                )
            side = max(
                int(
                    round(
                        min(width, height)
                        * reference_size
                        / reference_resolution
                    )
                ),
                1,
            )
            side = min(side, width, height)
            x0 = (center[:, 0] - side / 2.0).floor().long()
            y0 = (center[:, 1] - side / 2.0).floor().long()
            x0 = x0.clamp(0, width - side)
            y0 = y0.clamp(0, height - side)
            boxes[name] = torch.stack(
                (x0, y0, x0 + side, y0 + side), dim=-1
            )
        return boxes

    def forward(
        self,
        batch: Mapping[str, Any],
        background: Optional[torch.Tensor] = None,
        reference_pose: bool = False,
    ) -> dict[str, Any]:
        if "neck_pose" in batch:
            raise ValueError("Stage-2 does not accept neck_pose")
        if reference_pose:
            self._set_reference_pose()
        else:
            self._set_pose(*self._batch_pose(batch))

        background = (self.background if background is None else background.to(self.gaussian.device).float())
        with torch.cuda.amp.autocast(enabled=False):
            cameras = self._cameras(batch)
            means, scales, rotations = self._aligned_render_geometry()
            scales = self._cap_full_render_world_scales(scales)
            packed = (means, scales, rotations)

        images, raw_images, alphas, depths = [], [], [], []
        viewspace_points, visibility_filters, radii = [], [], []
        for camera in cameras:
            with torch.cuda.amp.autocast(enabled=False):
                package = render(camera, self.gaussian, self.pipe, background, precomputed_geometry=packed)
            raw = package["render"].permute(1, 2, 0)
            raw_images.append(raw)
            images.append(raw.clamp(0.0, 1.0))
            alphas.append(package["alpha_3dgs"].permute(1, 2, 0).clamp(0.0, 1.0))
            depths.append(package["depth_3dgs"].permute(1, 2, 0))
            viewspace_points.append(package["viewspace_points"])
            visibility_filters.append(package["visibility_filter"])
            radii.append(package["radii"])

        return {
            "comp_rgb": torch.stack(images),
            "comp_rgb_raw": torch.stack(raw_images),
            "alpha": torch.stack(alphas),
            "depth": torch.stack(depths),
            "means": means,
            "scales": scales,
            "rotations": rotations,
            "viewspace_points": viewspace_points,
            "visibility_filter": visibility_filters,
            "radii": radii,
        }

    @torch.no_grad()
    def _render_point_mask(
        self,
        batch: Mapping[str, Any],
        output: Mapping[str, torch.Tensor],
        point_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Rasterize a semantic Gaussian subset as a detached image mask."""

        point_mask = point_mask.to(
            device=self.gaussian.device, dtype=torch.bool
        )
        if point_mask.numel() != self.gaussian.num_gs:
            raise ValueError("Semantic point mask has the wrong topology")
        packed = (
            output["means"].detach(),
            output["scales"].detach(),
            output["rotations"].detach(),
        )
        opacity = self.gaussian.get_opacity.detach() * point_mask[:, None]
        masks = []
        with torch.cuda.amp.autocast(enabled=False):
            for camera in self._cameras(batch):
                package = render(
                    camera,
                    self.gaussian,
                    self.pipe,
                    torch.zeros(3, device=self.gaussian.device),
                    override_opacity=opacity,
                    precomputed_geometry=packed,
                )
                masks.append(
                    package["alpha_3dgs"]
                    .permute(1, 2, 0)
                    .clamp(0.0, 1.0)
                )
        return torch.stack(masks)

    def _diffusion_guidance_mask(
        self,
        batch: Mapping[str, Any],
        output: Mapping[str, torch.Tensor],
        face_crop_boxes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.optimization_stage == "mouth":
            mask = self._render_point_mask(
                batch, output, self.mouth_guidance_point_mask
            )
            return self._dilate_mask(
                mask, int(self.cfg.mouth_guidance_dilation)
            ).detach()

        # Do not derive the full-stage foreground from the trainable render.
        # With output["alpha"], one oversized Gaussian makes its new support
        # part of the guidance mask, so diffusion keeps rewarding the growth.
        # The calibrated Stage-1 alpha is fixed and breaks that feedback loop.
        mask = self._reference_silhouette_mask(batch, output)
        regional = self.cfg.regional_guidance
        if bool(regional.get("enabled", False)) and bool(
            regional.get("exclude_face_from_full", True)
        ):
            if (
                face_crop_boxes is not None
                and bool(regional.get("exclude_face_box_from_full", False))
            ):
                mask = mask.clone()
                azimuth = torch.as_tensor(
                    batch["azimuth"],
                    dtype=torch.float32,
                    device=mask.device,
                ).reshape(-1) % 360.0
                front = (azimuth >= 0.0) & (azimuth <= 180.0)
                height, width = mask.shape[1:3]
                for index in torch.nonzero(
                    front, as_tuple=False
                ).flatten().tolist():
                    x0, y0, x1, y1 = [
                        int(value)
                        for value in face_crop_boxes[index].tolist()
                    ]
                    x0, x1 = max(x0, 0), min(x1, width)
                    y0, y1 = max(y0, 0), min(y1, height)
                    mask[index, y0:y1, x0:x1] = 0.0
                return mask.clamp(0.0, 1.0)
            face = self._regional_guidance_mask("face", batch, output)
            return (mask * (1.0 - face)).clamp(0.0, 1.0)
        return mask

    def _reference_silhouette_mask(
        self,
        batch: Mapping[str, Any],
        output: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return the immutable calibrated foreground in render resolution."""

        if self._uses_continuous_camera(batch):
            _, continuous_alpha, _ = self._render_initial_continuous_reference(
                batch, reference_pose=False
            )
            alpha = self._bchw(
                continuous_alpha, self.gaussian.device
            )[:, :1]
        else:
            if "reference_alpha" not in batch:
                raise KeyError(
                    "Full refinement requires batch['reference_alpha'] for a "
                    "stable diffusion foreground"
                )
            alpha = self._bchw(
                batch["reference_alpha"], self.gaussian.device
            )[:, :1]
        batch_size, height, width = output["alpha"].shape[:3]
        if alpha.shape[0] == 1 and batch_size > 1:
            alpha = alpha.expand(batch_size, -1, -1, -1)
        if alpha.shape[0] != batch_size:
            raise ValueError(
                "reference_alpha batch size does not match the rendered batch"
            )
        if alpha.shape[-2:] != (height, width):
            alpha = F.interpolate(
                alpha,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        mask = alpha.permute(0, 2, 3, 1).clamp(0.0, 1.0)
        mask = self._dilate_mask(
            mask, int(self.cfg.guidance_reference_dilation)
        )
        return mask.detach()

    @staticmethod
    def _dilate_mask(mask: torch.Tensor, dilation: int) -> torch.Tensor:
        if dilation > 0:
            mask_bchw = mask.permute(0, 3, 1, 2)
            mask_bchw = F.max_pool2d(
                mask_bchw,
                kernel_size=2 * dilation + 1,
                stride=1,
                padding=dilation,
            )
            mask = mask_bchw.permute(0, 2, 3, 1)
        return mask

    def _regional_guidance_mask(
        self,
        name: str,
        batch: Mapping[str, Any],
        output: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        point_mask = self.regional_point_masks[name]
        if bool(point_mask.any().item()):
            mask = self._render_point_mask(batch, output, point_mask)
        else:
            mask = output["alpha"].detach()
        dilation_cfg = self.cfg.regional_guidance.get("dilation", 16)
        if isinstance(dilation_cfg, Mapping):
            dilation = int(dilation_cfg.get(name, 16))
        else:
            dilation = int(dilation_cfg)
        if dilation < 0:
            raise ValueError("regional_guidance.dilation must be non-negative")
        mask = self._dilate_mask(mask, dilation).detach()
        if (
            self.optimization_stage == "full"
            and bool(
                self.cfg.scale_stability.get(
                    "mask_regions_to_reference", True
                )
            )
        ):
            mask = mask * self._reference_silhouette_mask(batch, output)
        # train_all.py enables face/mouth only on its 0..180-degree face
        # hemisphere, and gates each eye at 60/120 degrees.  Prompt azimuth in
        # this calibrated dataset has the same front-at-90 convention.
        azimuth = torch.as_tensor(
            batch["azimuth"], dtype=torch.float32, device=mask.device
        ).reshape(-1) % 360.0
        face_gate = (azimuth >= 0.0) & (azimuth <= 180.0)
        if name == "left_eye":
            gate = face_gate & (azimuth < 120.0)
        elif name == "right_eye":
            gate = face_gate & (azimuth > 60.0)
        elif name in {"face", "mouth"}:
            gate = face_gate
        else:
            gate = torch.ones_like(face_gate)
        return mask * gate[:, None, None, None].to(mask.dtype)

    def _mask_protected_pixels_from_face_guidance(
        self,
        face_mask: torch.Tensor,
        batch: Mapping[str, Any],
        output: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, bool]:
        """Keep face guidance from modifying or occluding frozen regions."""

        protection = self.cfg.full_protection
        if not (
            self._full_region_protection_enabled()
            and bool(protection.get("mask_face_loss", True))
        ):
            return face_mask, False
        protected_points = torch.zeros_like(self.dental_point_mask)
        protected_regions = []
        if bool(protection.get("freeze_eyes", True)):
            protected_points |= self.eye_point_mask
            protected_regions.extend(("left_eye", "right_eye"))
        if bool(protection.get("freeze_mouth", False)):
            protected_points |= self.mouth_guidance_point_mask
            protected_regions.append("mouth")
        elif bool(protection.get("freeze_dental", True)):
            protected_points |= self.dental_point_mask
            protected_regions.append("mouth")
        if bool(protected_points.any().item()):
            protected = self._render_point_mask(
                batch, output, protected_points
            )
            dilation_cfg = self.cfg.regional_guidance.get("dilation", 16)
            if isinstance(dilation_cfg, Mapping):
                dilation = max(
                    int(dilation_cfg.get(name, 16))
                    for name in protected_regions
                )
            else:
                dilation = int(dilation_cfg)
            if dilation < 0:
                raise ValueError(
                    "regional_guidance.dilation must be non-negative"
                )
            protected = self._dilate_mask(protected, dilation).detach()
            if bool(
                self.cfg.scale_stability.get(
                    "mask_regions_to_reference", True
                )
            ):
                protected = (
                    protected
                    * self._reference_silhouette_mask(batch, output)
                )
        else:
            protected = torch.zeros_like(face_mask)
        # Preserve train_all.py's view gate while replacing its all-ones face
        # crop loss with all pixels except the protected, dilated subregions.
        active = (
            face_mask.flatten(1).amax(dim=1) > 1.0e-4
        ).to(face_mask.dtype)
        mask = (1.0 - protected.clamp(0.0, 1.0))
        mask = mask * active[:, None, None, None]
        return mask, True

    # ------------------------------------------------------------------
    # Diffusion and reconstruction-aware constrained objective.
    # ------------------------------------------------------------------

    def _prompt_config(
        self,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
    ):
        config = self.cfg.prompt_processor
        if OmegaConf.is_config(config):
            config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        else:
            config = OmegaConf.create(dict(config))
        config.prompt = self.cfg.prompt if prompt is None else prompt
        if negative_prompt is not None:
            config.negative_prompt = negative_prompt
        return config

    def on_fit_start(self) -> None:
        BaseLift3DSystem.on_fit_start(self)
        if int(self.trainer.world_size) != 1:
            raise RuntimeError("Stage-2 currently supports one GPU only")
        precision_plugin = getattr(self.trainer, "precision_plugin", None)
        scaler = getattr(precision_plugin, "scaler", None)
        if self.optimization_stage == "mouth" and scaler is not None:
            raise RuntimeError(
                "Mouth refinement must not use fp16 mixed precision. Its "
                "Gaussian gradients overflow the GradScaler and make every "
                "Adam step a silent no-op. Set trainer.precision='32-true' "
                "(recommended) or use bf16 mixed precision without a scaler."
            )
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(self._prompt_config())
        sdedit_negative_prompt = self.cfg.sdedit.get("negative_prompt")
        if (
            self.sdedit_enabled
            and sdedit_negative_prompt is not None
            and str(sdedit_negative_prompt).strip()
            and str(sdedit_negative_prompt)
            != str(self.cfg.prompt_processor.get("negative_prompt", ""))
        ):
            self.sdedit_prompt_processor = threestudio.find(
                self.cfg.prompt_processor_type
            )(
                self._prompt_config(
                    negative_prompt=str(sdedit_negative_prompt)
                )
            )
        else:
            self.sdedit_prompt_processor = self.prompt_processor
        self.regional_prompt_processors = {}
        self.sdedit_regional_prompt_processors = {}
        regional = self.cfg.regional_guidance
        if (
            self.optimization_stage == "full"
            and bool(regional.get("enabled", False))
        ):
            abstract = str(regional.get("abstract_prompt", self.cfg.prompt))
            configured_prompts = regional.get("prompts", {})
            prompts = {
                "face": str(configured_prompts.get("face", self.cfg.prompt)),
                "left_eye": str(
                    configured_prompts.get(
                        "left_eye", f"left eye region, {abstract}"
                    )
                ),
                "right_eye": str(
                    configured_prompts.get(
                        "right_eye", f"right eye region, {abstract}"
                    )
                ),
                "mouth": str(
                    configured_prompts.get(
                        "mouth", f"mouth region, {abstract}"
                    )
                ),
            }
            sdedit_region_weights = self.cfg.sdedit.get("region_weights", {})
            for name, prompt in prompts.items():
                ism_region_weight = float(
                    regional.get(f"{name}_weight", 1.0)
                )
                sdedit_region_weight = float(
                    sdedit_region_weights.get(name, ism_region_weight)
                )
                if ism_region_weight > 0.0:
                    if prompt == str(self.cfg.prompt):
                        processor = self.prompt_processor
                    else:
                        processor = threestudio.find(
                            self.cfg.prompt_processor_type
                        )(self._prompt_config(prompt))
                    self.regional_prompt_processors[name] = processor
                if self.sdedit_enabled and sdedit_region_weight > 0.0:
                    # train_all.py passes the same SDEdit negative prompt to
                    # each face/eye/mouth img2img call.  Keep these processors
                    # separate from their ISM counterparts so their
                    # unconditional embeddings cannot inherit the much longer
                    # ISM-only negative prompt.
                    self.sdedit_regional_prompt_processors[name] = (
                        threestudio.find(self.cfg.prompt_processor_type)(
                            self._prompt_config(
                                prompt,
                                negative_prompt=(
                                    str(sdedit_negative_prompt)
                                    if sdedit_negative_prompt is not None
                                    else None
                                ),
                            )
                        )
                    )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        if self.surface_sdedit_enabled:
            attention_keys = (
                "atlas_resolution",
                "max_tokens",
                "min_views",
                "max_memory_views",
                "strength",
                "start_progress",
                "end_progress",
                "exclude_self",
                "processor_patterns",
            )
            attention_config = {
                key: self.surface_memory_config[key]
                for key in attention_keys
                if key in self.surface_memory_config
            }
            if "processor_patterns" in attention_config:
                attention_config["processor_patterns"] = tuple(
                    attention_config["processor_patterns"]
                )
            self.guidance.configure_surface_memory_attention(
                attention_config
            )
        if self.sdedit_enabled and float(
            self.cfg.sdedit.get("lpips_weight", 0.0)
        ) > 0.0:
            try:
                from gaussiansplatting.lpipsPyTorch.modules.lpips import LPIPS

                perceptual = LPIPS(
                    net_type=str(self.cfg.sdedit.get("lpips_net", "alex"))
                ).to(self.gaussian.device).eval()
                for parameter in perceptual.parameters():
                    parameter.requires_grad_(False)
                # The metric is a frozen runtime dependency, not part of the
                # avatar checkpoint (AlexNet alone would add ~230 MB).
                object.__setattr__(self, "sdedit_perceptual", perceptual)
            except Exception as error:
                if bool(self.cfg.sdedit.get("require_lpips", False)):
                    raise RuntimeError("Unable to initialize SDEdit LPIPS") from error
                self._sdedit_lpips_fallback = True
                threestudio.warn(
                    "LPIPS could not be loaded; SDEdit will use a deterministic "
                    f"multi-scale feature loss instead ({error})."
                )
        if self._pending_vsd_state is not None:
            self.guidance.load_vsd_checkpoint_state(self._pending_vsd_state)
            self._pending_vsd_state = None
        if self._pending_uvd_flow_state is not None:
            self.guidance.load_uvd_flow_checkpoint_state(
                self._pending_uvd_flow_state
            )
            self._pending_uvd_flow_state = None

    @staticmethod
    def _bchw(value: Any, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if tensor.ndim == 3:
            tensor = tensor[None]
        if tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(0, 3, 1, 2)
        if tensor.max().detach() > 1.5:
            tensor = tensor / 255.0
        return tensor

    @staticmethod
    def _select_batch_views(
        batch: Mapping[str, Any], indices: torch.Tensor
    ) -> dict[str, Any]:
        """Select camera rows without slicing shared one-row FLAME pose data."""

        if "c2w" not in batch:
            raise KeyError("A refinement batch must contain c2w")
        batch_size = int(torch.as_tensor(batch["c2w"]).shape[0])
        selected = torch.as_tensor(indices, dtype=torch.long).reshape(-1)
        if selected.numel() == 0 or bool(
            ((selected < 0) | (selected >= batch_size)).any().item()
        ):
            raise ValueError("Batch-view indices are empty or out of range")
        result: dict[str, Any] = {}
        for name, value in batch.items():
            if (
                torch.is_tensor(value)
                and value.ndim > 0
                and int(value.shape[0]) == batch_size
            ):
                result[name] = value.index_select(
                    0, selected.to(value.device)
                )
            else:
                # expression/jaw/shape are intentionally shared across all
                # camera rows and normally have leading dimension one.
                result[name] = value
        return result

    def _sdedit_active(self) -> bool:
        return bool(
            self.sdedit_enabled
            and int(self.true_global_step) >= self.sdedit_start_step
        )

    def _sdedit_method_signature(self) -> dict[str, Any]:
        """Return the method-defining SDEdit settings saved in checkpoints."""

        signature: dict[str, Any] = {"mode": self.sdedit_mode}
        if not self.surface_sdedit_enabled:
            return signature
        config = self.surface_memory_config
        views = int(config.get("views", 4))
        integer_defaults = {
            "views": views,
            "atlas_resolution": 64,
            "max_tokens": 65536,
            "min_views": 2,
            "max_memory_views": views,
        }
        float_defaults = {
            "strength": 0.65,
            "start_progress": 0.45,
            "end_progress": 1.0,
            "alpha_threshold": 0.005,
            "contribution_threshold": 0.005,
            "variance_threshold": 0.0025,
            "dominance_ratio": 1.10,
            "opacity_floor": 0.0,
        }
        surface: dict[str, Any] = {
            key: int(config.get(key, default))
            for key, default in integer_defaults.items()
        }
        surface.update(
            {
                key: float(config.get(key, default))
                for key, default in float_defaults.items()
            }
        )
        depth_tolerance = config.get("depth_tolerance", 0.01)
        surface["depth_tolerance"] = (
            None
            if depth_tolerance is None
            else float(depth_tolerance)
        )
        surface["exclude_self"] = bool(
            config.get("exclude_self", False)
        )
        surface["processor_patterns"] = [
            str(value)
            for value in config.get(
                "processor_patterns", ("up_blocks.2", "up_blocks.3")
            )
        ]
        signature["surface_memory"] = surface
        return signature

    def _guidance_method_signature(self) -> dict[str, Any]:
        """Return the first-phase guidance identity saved in checkpoints."""

        if not self.uvd_flow_enabled:
            return {"mode": "ism"}
        guidance = self.cfg.guidance
        surface = self.cfg.uvd_surface_flow
        configured_d_range = surface.get("d_range")
        d_range = (
            None
            if configured_d_range is None
            else [float(value) for value in configured_d_range]
        )
        return {
            "mode": "uvd-sfd",
            "objective": {
                "name": "cfd-consistent-score-difference",
                "version": 1,
                "t_score": "negative-prompt-cfg",
                "s_score": "null-prompt",
                "weighting": "none",
                "interval": "animportrait3d-annealed-100-to-50",
            },
            "noise": {
                "seed": int(guidance.get("uvd_flow_noise_seed", 0)),
                "uv_resolution": int(
                    guidance.get("uvd_flow_uv_resolution", 256)
                ),
                "depth_resolution": int(
                    guidance.get("uvd_flow_depth_resolution", 8)
                ),
                "surface_layers": int(
                    guidance.get(
                        "uvd_flow_surface_layers",
                        len(SURFACE_LAYER_NAMES),
                    )
                ),
                "min_distinct_cells": int(
                    guidance.get("uvd_flow_min_distinct_cells", 1)
                ),
            },
            "correspondence": {
                "alpha_threshold": float(
                    surface.get("alpha_threshold", 0.05)
                ),
                "contribution_threshold": float(
                    surface.get("contribution_threshold", 0.01)
                ),
                "dominance_ratio": float(
                    surface.get("dominance_ratio", 1.10)
                ),
                "max_uvd_variance": float(
                    surface.get("max_uvd_variance", 0.0025)
                ),
                "opacity_floor": float(
                    surface.get("opacity_floor", 0.0)
                ),
                "d_padding_ratio": float(
                    surface.get("d_padding_ratio", 0.05)
                ),
                "d_range": d_range,
            },
        }

    def _validate_sdedit_checkpoint_method(
        self, checkpoint: Mapping[str, Any]
    ) -> None:
        """Lock the SDEdit ablation only after its first optimizer update."""

        saved_method = checkpoint.get("stage2_sdedit_method")
        current_method = self._sdedit_method_signature()
        explicit_started = checkpoint.get("stage2_sdedit_updates_started")
        saved_global_step = checkpoint.get("global_step")
        started = (
            bool(explicit_started)
            if explicit_started is not None
            else (
                int(saved_global_step) > self.sdedit_start_step
                if saved_global_step is not None
                else False
            )
        )
        if saved_method is None:
            if started or (
                int(checkpoint.get("stage2_schema_version", 0)) >= 7
                and saved_global_step is None
            ):
                raise ValueError(
                    "Cannot resume an active SDEdit checkpoint that does not "
                    "record its method configuration"
                )
            return
        if dict(saved_method) != current_method and started:
            raise ValueError(
                "Checkpoint SDEdit method differs from the current "
                f"configuration after SDEdit updates began: "
                f"saved={dict(saved_method)!r}, current={current_method!r}"
            )
        if dict(saved_method) != current_method:
            threestudio.info(
                "Switching SDEdit ablation at the untouched phase boundary: "
                f"saved={dict(saved_method)!r}, current={current_method!r}."
            )

    def _validate_guidance_checkpoint_method(
        self, checkpoint: Mapping[str, Any]
    ) -> None:
        """Reject a resume that mixes first-phase guidance objectives."""

        saved_method = checkpoint.get("stage2_guidance_method")
        current_method = self._guidance_method_signature()
        if saved_method is None:
            contains_uvd_state = (
                "stage2_uvd_ism_noise" in checkpoint
                or "stage2_uvd_surface_flow" in checkpoint
            )
            if (
                self.uvd_flow_enabled
                or contains_uvd_state
                or int(checkpoint.get("stage2_schema_version", 0)) >= 9
            ):
                raise ValueError(
                    "Cannot resume a checkpoint whose ISM guidance method "
                    "is not recorded"
                )
            return
        if dict(saved_method) != current_method:
            raise ValueError(
                "Checkpoint ISM guidance method differs from the current "
                f"configuration: saved={dict(saved_method)!r}, "
                f"current={current_method!r}"
            )
        if (
            str(dict(saved_method).get("mode", "")) == "uvd-sfd"
            and "stage2_uvd_ism_noise" not in checkpoint
        ):
            saved_global_step = checkpoint.get("global_step")
            if saved_global_step is None or int(saved_global_step) > 0:
                raise ValueError(
                    "UVD-SFD checkpoint is missing its canonical "
                    "noise/RNG state and cannot be resumed exactly"
                )

    @torch.no_grad()
    def _reset_adam_state_at_sdedit_boundary(self) -> None:
        """Optionally start SDEdit with fresh Adam moments."""

        reset_requested = bool(
            self.cfg.sdedit.get("reset_optimizer_at_start", False)
        )
        if (
            not reset_requested
            or not self._sdedit_active()
            or self._sdedit_optimizer_state_reset
        ):
            return
        if self._manual_micro_step != 0:
            raise RuntimeError(
                "Cannot reset Adam at the SDEdit boundary with unfinished "
                "first-phase gradient accumulation"
            )
        optimizer = getattr(self.gaussian, "optimizer", None)
        if optimizer is None:
            raise RuntimeError("Gaussian optimizer is unavailable at SDEdit start")
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        if not isinstance(raw_optimizer, torch.optim.Adam):
            raise TypeError(
                "Full refinement expects torch.optim.Adam at the SDEdit boundary, "
                f"got {type(raw_optimizer).__name__}"
            )

        # Capture the first-phase result before the first SDEdit update. Clearing state is
        # mathematically equivalent to constructing a new Adam over the same
        # parameter groups: PyTorch lazily rebuilds step/exp_avg/exp_avg_sq on
        # the next optimizer.step().  Keeping the optimizer object itself also
        # keeps Lightning's optimizer wrapper and checkpoint plumbing valid.
        if self._sdedit_reference_state is None:
            self._capture_sdedit_reference()
        local_steps = int(round(self._optimizer_progress(raw_optimizer)))
        self._optimizer_executed_step_offset += local_steps
        cleared_entries = len(raw_optimizer.state)
        raw_optimizer.zero_grad(set_to_none=True)
        raw_optimizer.state.clear()
        self._consecutive_skipped_optimizer_steps = 0
        self._sdedit_optimizer_state_reset = True
        threestudio.info(
            "Reset Adam state at the SDEdit boundary "
            f"(optimizer step {int(self.true_global_step)}, "
            f"accounted {local_steps} executed first-phase steps, "
            f"cumulative offset {self._optimizer_executed_step_offset}, "
            f"cleared {cleared_entries} parameter states)."
        )

    @torch.no_grad()
    def _release_unused_sdedit_controlnet(self) -> None:
        """Move the full-pass ControlNet off GPU once SDEdit no longer uses it."""

        if self._sdedit_control_released:
            return
        uses_control = bool(self.cfg.sdedit.get("main_use_control", True)) or bool(
            self.cfg.sdedit.get("regional_use_control", True)
        )
        if uses_control or self.guidance is None:
            return
        controlnet = getattr(self.guidance, "controlnet", None)
        if controlnet is not None:
            controlnet.to("cpu")
        pipe_controlnet = getattr(getattr(self.guidance, "pipe", None), "controlnet", None)
        if pipe_controlnet is not None and pipe_controlnet is not controlnet:
            pipe_controlnet.to("cpu")
        self._sdedit_control_released = True
        torch.cuda.empty_cache()
        threestudio.info(
            "Moved the unused full-pass ControlNet to CPU for SDEdit."
        )

    @torch.no_grad()
    def _capture_sdedit_reference(self) -> None:
        """Freeze the first-phase result used as every later img2img source."""

        # Never freeze an out-of-bounds first-phase state as the teacher for
        # all 750 SDEdit steps.  The optional repair is world-space only.
        scale_report = self._stabilize_current_full_scale()
        if scale_report["world_capped"]:
            threestudio.info(
                "Sanitized the SDEdit boundary geometry before capture: "
                f"world={scale_report['world_capped']}."
            )
        self._sdedit_reference_state = {
            "uv": self.gaussian._uv.detach().clone(),
            "d": self.gaussian._d.detach().clone(),
            "face_idx": self.gaussian._face_idx.detach().clone(),
            "feature": self.gaussian._features_dc.detach().clone(),
            "opacity": self.gaussian._opacity.detach().clone(),
            "scale": self.gaussian._scaling.detach().clone(),
            "rotation": self.gaussian._rotation.detach().clone(),
            "scale_trainable_mask": (
                self._active_trainable_point_mask().detach().clone()
            ),
        }
        threestudio.info(
            "Captured fixed SDEdit source at optimization step "
            f"{int(self.true_global_step)}."
        )

    @torch.no_grad()
    def _frozen_reference_geometry(
        self,
        state: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
        reference_pose: bool = False,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Map one immutable Gaussian state into the sampled live pose."""

        if reference_pose:
            self._set_reference_pose()
        else:
            self._set_pose(*self._batch_pose(batch))
        with torch.cuda.amp.autocast(enabled=False):
            vertices, normals = self.gaussian._flame_verts_and_normals()
            uv = state["uv"]
            distance = state["d"]
            face_idx = state["face_idx"]
            means = self.gaussian._map_uvd_to_xyz(
                torch.cat((uv, distance), dim=-1),
                vertices,
                normals,
                face_idx=face_idx,
            )
            scales, rotations = self.gaussian._deformed_scaling_rotation(
                vertices,
                face_idx=face_idx,
                local_scaling=self.gaussian.scaling_activation(
                    state["scale"]
                ),
                local_rotation=state["rotation"],
            )
            means, scales, rotations = apply_similarity_to_gaussians(
                means, scales, rotations, self.alignment
            )
            if self._full_scale_stability_enabled():
                # The frozen first-phase source is rendered under many later poses.
                # Bound both explicit axes and absolute anisotropy per pose
                # without mutating the snapshot.  A frozen source can retain
                # the pre-densification topology, so its region-protection
                # mask must come from the same snapshot rather than the live
                # Gaussian model.
                scales = scales * self._full_world_scale_axis_ratio(
                    scales, state.get("scale_trainable_mask")
                )
            packed = (means, scales, rotations)
            opacity = self.gaussian.opacity_activation(state["opacity"])
        return packed, opacity

    @torch.no_grad()
    def _render_frozen_reference_state(
        self,
        state: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
        background: torch.Tensor,
        reference_pose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render an immutable Gaussian state in a sampled camera/pose."""

        packed, opacity = self._frozen_reference_geometry(
            state, batch, reference_pose=reference_pose
        )
        color = SH2RGB(state["feature"][:, 0, :]).clamp(0.0, 1.0)
        cameras = self._cameras(batch)

        images, alphas, depths = [], [], []
        for camera in cameras:
            with torch.cuda.amp.autocast(enabled=False):
                package = render(
                    camera,
                    self.gaussian,
                    self.pipe,
                    background.float(),
                    override_color=color.float(),
                    override_opacity=opacity.float(),
                    precomputed_geometry=packed,
                )
            images.append(package["render"].permute(1, 2, 0).clamp(0.0, 1.0))
            alphas.append(
                package["alpha_3dgs"]
                .permute(1, 2, 0)
                .clamp(0.0, 1.0)
            )
            depths.append(package["depth_3dgs"].permute(1, 2, 0))
        return (
            torch.stack(images).detach(),
            torch.stack(alphas).detach(),
            torch.stack(depths).detach(),
        )

    @staticmethod
    def _uses_continuous_camera(batch: Mapping[str, Any]) -> bool:
        value = batch.get("continuous_camera")
        if value is None:
            return False
        return bool(torch.as_tensor(value, dtype=torch.bool).all().item())

    @torch.no_grad()
    def _render_initial_continuous_reference(
        self,
        batch: Mapping[str, Any],
        reference_pose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return same-camera targets for continuous training views."""

        if not self._uses_continuous_camera(batch):
            raise ValueError("Initial continuous reference requested for a calibrated batch")
        cache_key = (
            "_initial_reference_pose_cache"
            if reference_pose
            else "_initial_dynamic_pose_cache"
        )
        cached = batch.get(cache_key)
        if cached is not None:
            return cached
        rendered = self._render_frozen_reference_state(
            self._initial_reference_state,
            batch,
            torch.ones(3, device=self.gaussian.device),
            reference_pose=reference_pose,
        )
        if isinstance(batch, dict):
            batch[cache_key] = rendered
        return rendered

    @torch.no_grad()
    def _render_initial_dynamic_reference(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render the incoming full-stage avatar in the sampled live pose."""

        cache_key = "_initial_dynamic_pose_cache"
        cached = batch.get(cache_key)
        if cached is not None:
            return cached
        rendered = self._render_frozen_reference_state(
            self._initial_reference_state,
            batch,
            torch.ones(3, device=self.gaussian.device),
            reference_pose=False,
        )
        if isinstance(batch, dict):
            batch[cache_key] = rendered
        return rendered

    @torch.no_grad()
    def _render_sdedit_reference(
        self,
        batch: Mapping[str, Any],
        background: torch.Tensor,
    ) -> torch.Tensor:
        """Render the frozen phase-boundary state in the sampled live pose."""

        if self._sdedit_reference_state is None:
            self._capture_sdedit_reference()
        state = self._sdedit_reference_state
        assert state is not None
        image, _, _ = self._render_frozen_reference_state(
            state, batch, background, reference_pose=False
        )
        return image

    @staticmethod
    def _mask_bchw(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        mask = torch.as_tensor(mask, dtype=torch.float32, device=device)
        if mask.ndim == 3:
            mask = mask[:, None]
        elif mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask.permute(0, 3, 1, 2)
        if mask.ndim != 4:
            raise ValueError("SDEdit mask must be Bx1xHxW or BxHxWx1")
        if mask.shape[1] != 1:
            mask = mask.mean(dim=1, keepdim=True)
        return mask.clamp(0.0, 1.0)

    def _prepare_sdedit_images(
        self,
        prediction: torch.Tensor,
        source: torch.Tensor,
        control: torch.Tensor,
        mask: torch.Tensor,
        crop: bool,
        padding_override: Optional[int] = None,
        square: bool = False,
        crop_boxes: Optional[torch.Tensor] = None,
        return_crop_plan: bool = False,
    ) -> Optional[tuple[torch.Tensor, ...]]:
        """Select visible rows and optionally zoom each semantic region."""

        mask = self._mask_bchw(mask, prediction.device)
        active = mask.flatten(1).amax(dim=1) > 1.0e-4
        if crop_boxes is not None:
            crop_boxes = torch.as_tensor(
                crop_boxes, dtype=torch.long, device=prediction.device
            )
            if crop_boxes.shape != (prediction.shape[0], 4):
                raise ValueError(
                    "crop_boxes must have shape [B, 4] in x0,y0,x1,y1 order"
                )
        indices = torch.nonzero(active, as_tuple=False).flatten()
        if indices.numel() == 0:
            return None

        resolution = 512
        if not crop:
            height, width = prediction.shape[-2:]
            selected_boxes = torch.tensor(
                (0, 0, width, height),
                dtype=torch.long,
                device=prediction.device,
            )[None].repeat(indices.numel(), 1)
            prepared = (
                F.interpolate(
                    prediction[indices],
                    (resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                ),
                F.interpolate(
                    source[indices],
                    (resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                ),
                F.interpolate(
                    control[indices],
                    (resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                ),
                F.interpolate(
                    mask[indices],
                    (resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0.0, 1.0),
                indices,
            )
            return prepared + (selected_boxes,) if return_crop_plan else prepared

        padding = (
            int(self.cfg.sdedit.get("crop_padding", 8))
            if padding_override is None
            else int(padding_override)
        )
        predictions, sources, controls, masks, selected_boxes = [], [], [], [], []
        height, width = prediction.shape[-2:]
        for index in indices.tolist():
            if crop_boxes is not None:
                x0, y0, x1, y1 = [
                    int(value) for value in crop_boxes[index].tolist()
                ]
                x0, x1 = max(x0, 0), min(x1, width)
                y0, y1 = max(y0, 0), min(y1, height)
                if x1 <= x0 or y1 <= y0:
                    raise ValueError("crop_boxes contains an empty crop")
            else:
                coordinates = torch.nonzero(
                    mask[index, 0] > 1.0e-4, as_tuple=False
                )
                y_min = int(coordinates[:, 0].min().item())
                y_max = int(coordinates[:, 0].max().item()) + 1
                x_min = int(coordinates[:, 1].min().item())
                x_max = int(coordinates[:, 1].max().item()) + 1
                if square:
                    side = min(
                        max(y_max - y_min, x_max - x_min) + 2 * padding,
                        height,
                        width,
                    )
                    center_y = 0.5 * (y_min + y_max)
                    center_x = 0.5 * (x_min + x_max)
                    y0 = min(
                        max(int(round(center_y - side / 2.0)), 0),
                        height - side,
                    )
                    x0 = min(
                        max(int(round(center_x - side / 2.0)), 0),
                        width - side,
                    )
                    y1, x1 = y0 + side, x0 + side
                else:
                    y0 = max(y_min - padding, 0)
                    y1 = min(y_max + padding, height)
                    x0 = max(x_min - padding, 0)
                    x1 = min(x_max + padding, width)

            def resize(value: torch.Tensor) -> torch.Tensor:
                return F.interpolate(
                    value[index:index + 1, :, y0:y1, x0:x1],
                    (resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                )

            predictions.append(resize(prediction))
            sources.append(resize(source))
            controls.append(resize(control))
            masks.append(resize(mask).clamp(0.0, 1.0))
            selected_boxes.append((x0, y0, x1, y1))
        prepared = (
            torch.cat(predictions),
            torch.cat(sources),
            torch.cat(controls),
            torch.cat(masks),
            indices,
        )
        if not return_crop_plan:
            return prepared
        return prepared + (
            torch.as_tensor(
                selected_boxes,
                dtype=torch.long,
                device=prediction.device,
            ),
        )

    def _prepare_uvd_flow_correspondence(
        self,
        surface: Mapping[str, torch.Tensor],
        indices: torch.Tensor,
        crop_boxes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply the exact RGB/control crop plan to detached UVD buffers."""

        required = ("surface_uvd", "surface_layer", "surface_confidence")
        if any(name not in surface for name in required):
            raise ValueError("UVD-SFD surface correspondence is incomplete")
        uvd = surface["surface_uvd"]
        layer = surface["surface_layer"]
        confidence = surface["surface_confidence"]
        if (
            uvd.ndim != 4
            or uvd.shape[1] != 3
            or layer.shape != (uvd.shape[0], 1, *uvd.shape[-2:])
            or confidence.shape != layer.shape
        ):
            raise ValueError(
                "UVD-SFD correspondence must be UVD Bx3xHxW plus layer/"
                "confidence Bx1xHxW"
            )
        indices = torch.as_tensor(
            indices, dtype=torch.long, device=uvd.device
        ).reshape(-1)
        crop_boxes = torch.as_tensor(
            crop_boxes, dtype=torch.long, device=uvd.device
        )
        if crop_boxes.shape != (indices.numel(), 4):
            raise ValueError(
                "UVD-SFD crop plan must contain one box per selected view"
            )
        resolution = 512
        uvd_crops, layer_crops, confidence_crops = [], [], []
        height, width = uvd.shape[-2:]
        for index, box in zip(indices.tolist(), crop_boxes.tolist()):
            x0, y0, x1, y1 = (int(value) for value in box)
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError("UVD-SFD crop plan is outside the render")
            slices = (slice(index, index + 1), slice(None), slice(y0, y1), slice(x0, x1))
            # Coordinates and semantic ids are categorical surface samples;
            # nearest resize never interpolates across a UV seam or between
            # upper/lower teeth.  Confidence remains a smooth visibility mask.
            resized_uvd = F.interpolate(
                uvd[slices], (resolution, resolution), mode="nearest"
            )
            resized_layer = F.interpolate(
                layer[slices].float(),
                (resolution, resolution),
                mode="nearest",
            ).round().long()
            resized_confidence = F.interpolate(
                confidence[slices],
                (resolution, resolution),
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            # Bilinear confidence can bleed across a crop/resampling boundary.
            # Revalidate the categorical nearest-neighbour correspondence so
            # layer=-1, NaN, or out-of-volume coordinates can never receive a
            # diffusion gradient through background noise.
            valid = (
                (resized_layer >= 0)
                & torch.isfinite(resized_uvd).all(dim=1, keepdim=True)
                & ((resized_uvd >= 0.0) & (resized_uvd <= 1.0)).all(
                    dim=1, keepdim=True
                )
            )
            uvd_crops.append(resized_uvd)
            layer_crops.append(resized_layer)
            confidence_crops.append(
                resized_confidence * valid.to(resized_confidence.dtype)
            )
        return {
            "uvd_flow_surface_uvd": torch.cat(uvd_crops),
            "uvd_flow_surface_layer": torch.cat(layer_crops),
            "uvd_flow_surface_confidence": torch.cat(confidence_crops),
        }

    def _prepare_surface_memory_context(
        self,
        surface: Mapping[str, torch.Tensor],
        indices: torch.Tensor,
        crop_boxes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply the RGB crop plan to UV/layer/depth/visibility buffers."""

        required = (
            "surface_uv",
            "surface_layer",
            "surface_depth",
            "surface_visibility",
        )
        if any(name not in surface for name in required):
            raise ValueError(
                "FLAME surface-memory correspondence is incomplete"
            )
        uv = surface["surface_uv"]
        layer = surface["surface_layer"]
        depth = surface["surface_depth"]
        visibility = surface["surface_visibility"]
        scalar_shape = (uv.shape[0], 1, *uv.shape[-2:])
        if (
            uv.ndim != 4
            or uv.shape[1] != 2
            or layer.shape != scalar_shape
            or depth.shape != scalar_shape
            or visibility.shape != scalar_shape
        ):
            raise ValueError(
                "Surface memory expects UV Bx2xHxW and layer/depth/"
                "visibility Bx1xHxW"
            )
        indices = torch.as_tensor(
            indices, dtype=torch.long, device=uv.device
        ).reshape(-1)
        crop_boxes = torch.as_tensor(
            crop_boxes, dtype=torch.long, device=uv.device
        )
        if crop_boxes.shape != (indices.numel(), 4):
            raise ValueError(
                "Surface-memory crop plan must contain one box per view"
            )
        resolution = 512
        uv_crops: list[torch.Tensor] = []
        layer_crops: list[torch.Tensor] = []
        depth_crops: list[torch.Tensor] = []
        visibility_crops: list[torch.Tensor] = []
        height, width = uv.shape[-2:]
        for index, box in zip(indices.tolist(), crop_boxes.tolist()):
            x0, y0, x1, y1 = (int(value) for value in box)
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError(
                    "Surface-memory crop plan lies outside the render"
                )
            slices = (
                slice(index, index + 1),
                slice(None),
                slice(y0, y1),
                slice(x0, x1),
            )
            # UV, semantic ID, and depth must all originate from the same
            # categorical front-surface sample.  Interpolating any one of
            # them independently would create a false correspondence.
            resized_uv = F.interpolate(
                uv[slices], (resolution, resolution), mode="nearest"
            )
            resized_layer = F.interpolate(
                layer[slices].float(),
                (resolution, resolution),
                mode="nearest",
            ).round().long()
            resized_depth = F.interpolate(
                depth[slices], (resolution, resolution), mode="nearest"
            )
            resized_visibility = F.interpolate(
                visibility[slices],
                (resolution, resolution),
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            valid = (
                (resized_layer >= 0)
                & torch.isfinite(resized_uv).all(dim=1, keepdim=True)
                & ((resized_uv >= 0.0) & (resized_uv <= 1.0)).all(
                    dim=1, keepdim=True
                )
                & torch.isfinite(resized_depth)
                & (resized_depth > 0.0)
            )
            uv_crops.append(resized_uv)
            layer_crops.append(resized_layer)
            depth_crops.append(resized_depth)
            visibility_crops.append(
                resized_visibility * valid.to(resized_visibility.dtype)
            )
        return {
            "uv": torch.cat(uv_crops),
            "layer_ids": torch.cat(layer_crops),
            "depth": torch.cat(depth_crops),
            "visibility": torch.cat(visibility_crops),
        }

    @staticmethod
    def _masked_l1(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        numerator = ((prediction - target).abs() * mask).sum()
        denominator = mask.sum().clamp_min(1.0) * prediction.shape[1]
        return numerator / denominator

    @staticmethod
    def _multiscale_feature_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for scale in (1, 2, 4, 8):
            if scale == 1:
                pred_level, target_level = prediction, target
            else:
                pred_level = F.avg_pool2d(prediction, scale, scale)
                target_level = F.avg_pool2d(target, scale, scale)
            losses.append(F.l1_loss(pred_level, target_level))
        return torch.stack(losses).mean()

    def _sdedit_region_loss(
        self,
        prediction: torch.Tensor,
        source: torch.Tensor,
        control: torch.Tensor,
        mask: torch.Tensor,
        processor: Any,
        metadata: Mapping[str, torch.Tensor],
        *,
        crop: bool,
        use_control: bool,
        square_crop: bool = False,
        full_crop_loss: bool = False,
        crop_boxes: Optional[torch.Tensor] = None,
        native_crop_loss: bool = False,
        surface_memory: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prepared = self._prepare_sdedit_images(
            prediction,
            source,
            control,
            mask,
            crop,
            square=square_crop,
            crop_boxes=crop_boxes,
            return_crop_plan=surface_memory is not None,
        )
        zero = prediction.sum() * 0.0
        if prepared is None:
            return zero, {
                "active": torch.zeros((), device=prediction.device),
                "l1": zero.detach(),
                "lpips": zero.detach(),
            }
        if surface_memory is None:
            (
                pred_region,
                source_region,
                control_region,
                region_mask,
                indices,
            ) = prepared
            surface_context = None
        else:
            (
                pred_region,
                source_region,
                control_region,
                region_mask,
                indices,
                selected_crop_boxes,
            ) = prepared
            surface_context = self._prepare_surface_memory_context(
                surface_memory, indices, selected_crop_boxes
            )
        selected_metadata = {
            name: value[indices] for name, value in metadata.items()
        }
        with torch.no_grad():
            result = self.guidance(
                self.true_global_step,
                source_region,
                control_region,
                processor(),
                **selected_metadata,
                rgb_as_latents=False,
                edit_image=True,
                edit_strength=float(self.cfg.sdedit.get("strength", 0.3)),
                edit_guidance_scale=float(
                    self.cfg.sdedit.get("guidance_scale", 7.5)
                ),
                edit_num_inference_steps=int(
                    self.cfg.sdedit.get("num_inference_steps", 20)
                ),
                edit_use_control=use_control,
                view_dependent_prompting=bool(
                    self.cfg.sdedit.get("view_dependent_prompting", True)
                ),
                surface_memory_context=surface_context,
            )
        target = result["edit_images"].permute(0, 3, 1, 2).detach()
        target = target.to(dtype=pred_region.dtype).clamp(0.0, 1.0)
        region_mask = region_mask.to(dtype=pred_region.dtype)
        if crop and full_crop_loss:
            region_mask = torch.ones_like(region_mask)

        # AnimPortrait3D generates the SDEdit target at 512x512, resizes it
        # back to the native 200x200 mouth crop, and only then evaluates L1
        # and VGG-LPIPS.  Computing the loss on the enlarged teacher image
        # overweights interpolated high frequencies and gave almost no useful
        # gradient to the small teeth in the portrait render.
        if native_crop_loss:
            if not crop or crop_boxes is None:
                raise ValueError(
                    "native_crop_loss requires crop=True and fixed crop_boxes"
                )
            native_mask = self._mask_bchw(mask, prediction.device)
            native_l1, native_lpips, native_mask_means = [], [], []
            height, width = prediction.shape[-2:]
            for local_index, batch_index in enumerate(indices.tolist()):
                x0, y0, x1, y1 = [
                    int(value) for value in crop_boxes[batch_index].tolist()
                ]
                x0, x1 = max(x0, 0), min(x1, width)
                y0, y1 = max(y0, 0), min(y1, height)
                native_prediction = prediction[
                    batch_index:batch_index + 1, :, y0:y1, x0:x1
                ]
                native_target = F.interpolate(
                    target[local_index:local_index + 1],
                    size=native_prediction.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                ).clamp(0.0, 1.0)
                native_region_mask = native_mask[
                    batch_index:batch_index + 1, :, y0:y1, x0:x1
                ].to(dtype=native_prediction.dtype)
                if full_crop_loss:
                    native_region_mask = torch.ones_like(native_region_mask)
                row_l1 = self._masked_l1(
                    native_prediction, native_target, native_region_mask
                )
                composite = (
                    native_prediction * native_region_mask
                    + native_target * (1.0 - native_region_mask)
                )
                if self.sdedit_perceptual is not None:
                    row_lpips = self.sdedit_perceptual(
                        composite * 2.0 - 1.0,
                        native_target * 2.0 - 1.0,
                    ).mean()
                else:
                    row_lpips = self._multiscale_feature_loss(
                        composite, native_target
                    )
                native_l1.append(row_l1)
                native_lpips.append(row_lpips)
                native_mask_means.append(native_region_mask.mean())
            l1 = torch.stack(native_l1).mean()
            lpips = torch.stack(native_lpips).mean()
            mask_mean = torch.stack(native_mask_means).mean()
        else:
            l1 = self._masked_l1(pred_region, target, region_mask)
            composite = (
                pred_region * region_mask + target * (1.0 - region_mask)
            )
            if self.sdedit_perceptual is not None:
                lpips = self.sdedit_perceptual(
                    composite * 2.0 - 1.0,
                    target * 2.0 - 1.0,
                ).mean()
            else:
                lpips = self._multiscale_feature_loss(composite, target)
            mask_mean = region_mask.detach().mean()
        loss = (
            float(self.cfg.sdedit.get("l1_weight", 0.8)) * l1
            + float(self.cfg.sdedit.get("lpips_weight", 0.2)) * lpips
        )
        metrics = {
            "active": torch.as_tensor(
                float(indices.numel()), device=prediction.device
            ),
            "l1": l1.detach(),
            "lpips": lpips.detach(),
            "mask_mean": mask_mean.detach(),
            "timestep": result["timestep"].detach(),
        }
        for name, value in result.items():
            if (
                name.startswith("surface_memory_")
                and torch.is_tensor(value)
                and value.numel() == 1
            ):
                metrics[name] = value.detach()
        return loss, metrics

    def _sdedit_loss(
        self,
        batch: Mapping[str, Any],
        output: Mapping[str, torch.Tensor],
        source_rgb: torch.Tensor,
    ) -> tuple[torch.Tensor, Mapping[str, Any], float]:
        self._release_unused_sdedit_controlnet()
        device = self.gaussian.device
        prediction = output["comp_rgb"].permute(0, 3, 1, 2)
        source = source_rgb.permute(0, 3, 1, 2)
        control = self._bchw(batch["flame_conds"], device)
        metadata = {
            name: torch.as_tensor(
                batch[name], dtype=torch.float32, device=device
            ).reshape(-1)
            for name in ("elevation", "azimuth", "camera_distances")
        }
        surface_memory = (
            self._render_surface_memory_correspondence(batch)
            if self.surface_sdedit_enabled
            else None
        )
        regional = self.cfg.regional_guidance
        enabled = bool(regional.get("enabled", False))
        full_weight = float(regional.get("full_weight", 1.0)) if enabled else 1.0
        regional_crop_boxes = (
            self._full_regional_crop_boxes(batch)
            if self.optimization_stage == "full" and enabled
            else {}
        )
        mouth_crop_boxes = (
            self._mouth_landmark_crop_boxes(batch)
            if self.optimization_stage == "mouth"
            else None
        )
        main_loss, main_metrics = self._sdedit_region_loss(
            prediction,
            source,
            control,
            self._diffusion_guidance_mask(
                batch, output, regional_crop_boxes.get("face")
            ),
            self.sdedit_prompt_processor or self.prompt_processor,
            metadata,
            crop=bool(
                self.cfg.sdedit.get(
                    "crop_main", self.optimization_stage == "mouth"
                )
            ),
            use_control=bool(self.cfg.sdedit.get("main_use_control", True)),
            square_crop=bool(
                self.cfg.sdedit.get(
                    "square_crop", self.optimization_stage == "mouth"
                )
            ),
            full_crop_loss=bool(
                self.cfg.sdedit.get(
                    "full_crop_loss", self.optimization_stage == "mouth"
                )
            ),
            crop_boxes=mouth_crop_boxes,
            native_crop_loss=bool(
                self.cfg.sdedit.get(
                    "native_crop_loss", self.optimization_stage == "mouth"
                )
            ),
            surface_memory=surface_memory,
        )
        if self.surface_sdedit_enabled and bool(
            self.surface_memory_config.get(
                "require_runtime_activity", True
            )
        ):
            attention_calls = main_metrics.get(
                "surface_memory_surface_attention_calls"
            )
            memory_queries = main_metrics.get(
                "surface_memory_memory_queries"
            )
            active_attention = (
                attention_calls is not None
                and float(torch.as_tensor(attention_calls).item()) > 0.0
            )
            active_queries = (
                memory_queries is not None
                and float(torch.as_tensor(memory_queries).item()) > 0.0
            )
            if not active_attention or not active_queries:
                raise RuntimeError(
                    "FLAME surface SDEdit was enabled, but its main denoising "
                    "call produced no matched surface-memory queries. Check "
                    "the rendered visibility/depth buffers, atlas resolution, "
                    "min_views, and processor_patterns; set "
                    "sdedit.surface_memory.require_runtime_activity=false "
                    "only for diagnostics."
                )
        combined = full_weight * main_loss
        metrics: dict[str, Any] = {
            "loss_sdedit": combined,
            "sdedit_enabled": torch.ones((), device=device),
            "sdedit_lpips_fallback": torch.as_tensor(
                float(self._sdedit_lpips_fallback), device=device
            ),
            "full_weight": torch.as_tensor(full_weight, device=device),
            "surface_memory_enabled": torch.as_tensor(
                float(self.surface_sdedit_enabled), device=device
            ),
        }
        if surface_memory is not None:
            metrics["surface_memory_ambiguous_fraction"] = (
                surface_memory["ambiguous_fraction"].detach()
            )
            metrics["surface_memory_visible_fraction"] = (
                (surface_memory["surface_visibility"] > 0.0)
                .float()
                .mean()
            )
        for key, value in main_metrics.items():
            metrics[f"sdedit_full_{key}"] = value

        sdedit_region_weights = self.cfg.sdedit.get("region_weights", {})
        for name, processor in self.sdedit_regional_prompt_processors.items():
            region_weight = float(
                sdedit_region_weights.get(
                    name, regional.get(f"{name}_weight", 1.0)
                )
            )
            if region_weight <= 0.0:
                continue
            region_mask = self._regional_guidance_mask(
                name, batch, output
            )
            protected_face_loss = False
            if name == "face":
                region_mask, protected_face_loss = (
                    self._mask_protected_pixels_from_face_guidance(
                        region_mask, batch, output
                    )
                )
            region_loss, region_metrics = self._sdedit_region_loss(
                prediction,
                source,
                control,
                region_mask,
                processor,
                metadata,
                crop=bool(self.cfg.sdedit.get("crop_regions", True)),
                use_control=bool(
                    self.cfg.sdedit.get("regional_use_control", True)
                ),
                square_crop=bool(
                    self.cfg.sdedit.get("square_region_crops", True)
                ),
                full_crop_loss=bool(
                    self.cfg.sdedit.get("regional_full_crop_loss", False)
                ) and not protected_face_loss,
                crop_boxes=regional_crop_boxes.get(name),
                native_crop_loss=bool(
                    self.cfg.sdedit.get("regional_native_crop_loss", False)
                ),
                surface_memory=surface_memory,
            )
            combined = combined + region_weight * region_loss
            metrics[f"regional_{name}_weight"] = torch.as_tensor(
                region_weight, device=device
            )
            for key, value in region_metrics.items():
                metrics[f"sdedit_{name}_{key}"] = value
        weight = float(self.cfg.sdedit.get("weight", 1.0))
        metrics["loss_sdedit"] = combined
        return combined * weight, metrics, weight

    def _diffusion_loss(
        self,
        batch: Mapping[str, Any],
        output: Mapping[str, torch.Tensor],
        sdedit_source: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Mapping[str, Any], float]:
        if self.guidance is None or self.prompt_processor is None:
            raise RuntimeError("Guidance is initialized in on_fit_start")
        if self._sdedit_active():
            if sdedit_source is None:
                raise RuntimeError(
                    "SDEdit requires the frozen first-phase source render"
                )
            return self._sdedit_loss(batch, output, sdedit_source)
        device = self.gaussian.device
        rgb_full = output["comp_rgb"].permute(0, 3, 1, 2)
        control_full = self._bchw(batch["flame_conds"], device)
        metadata_full = {
            name: torch.as_tensor(batch[name], dtype=torch.float32, device=device).reshape(-1)
            for name in ("elevation", "azimuth", "camera_distances")
        }
        extra_guidance: dict[str, torch.Tensor] = {}
        use_uvd_flow = bool(self.uvd_flow_enabled)
        uvd_flow_surface = (
            self._render_uvd_flow_surface(batch, output)
            if use_uvd_flow
            else None
        )
        regional = self.cfg.regional_guidance
        regional_enabled = bool(regional.get("enabled", False))
        regional_crop_boxes = (
            self._full_regional_crop_boxes(batch)
            if self.optimization_stage == "full" and regional_enabled
            else {}
        )
        guidance_mask_full = self._diffusion_guidance_mask(
            batch, output, regional_crop_boxes.get("face")
        )
        rgb = rgb_full
        control = control_full
        metadata = metadata_full
        guidance_mask = guidance_mask_full
        crop_indices: Optional[torch.Tensor] = None
        main_surface_indices = torch.arange(
            rgb_full.shape[0], device=device, dtype=torch.long
        )
        height, width = rgb_full.shape[-2:]
        main_crop_plan = torch.tensor(
            (0, 0, width, height), dtype=torch.long, device=device
        )[None].repeat(rgb_full.shape[0], 1)
        if bool(self.cfg.guidance_crop_main):
            mouth_crop_boxes = (
                self._mouth_landmark_crop_boxes(batch)
                if self.optimization_stage == "mouth"
                else None
            )
            prepared = self._prepare_sdedit_images(
                rgb_full,
                rgb_full,
                control_full,
                guidance_mask_full,
                crop=True,
                padding_override=int(self.cfg.guidance_crop_padding),
                square=bool(self.cfg.guidance_crop_square),
                crop_boxes=mouth_crop_boxes,
                return_crop_plan=use_uvd_flow,
            )
            if prepared is not None:
                if use_uvd_flow:
                    (
                        rgb,
                        _,
                        control,
                        guidance_mask,
                        crop_indices,
                        main_crop_plan,
                    ) = prepared
                else:
                    rgb, _, control, guidance_mask, crop_indices = prepared
                if bool(self.cfg.guidance_full_crop_loss):
                    guidance_mask = torch.ones_like(guidance_mask)
                metadata = {
                    name: value[crop_indices]
                    for name, value in metadata_full.items()
                }
                main_surface_indices = crop_indices
        if use_uvd_flow:
            assert uvd_flow_surface is not None
            extra_guidance.update(
                self._prepare_uvd_flow_correspondence(
                    uvd_flow_surface,
                    main_surface_indices,
                    main_crop_plan,
                )
            )
        result = self.guidance(
            self.true_global_step,
            rgb,
            control,
            self.prompt_processor(),
            **metadata,
            rgb_as_latents=False,
            guidance_mask=guidance_mask,
            guidance_mask_background_weight=float(
                self.cfg.diffusion_background_weight
            ),
            use_control=bool(regional.get("full_use_control", True)),
            **extra_guidance,
        )
        full_weight = (
            float(regional.get("full_weight", 1.0))
            if regional_enabled
            else 1.0
        )
        combined_loss = full_weight * result["loss_sds"]
        combined_result = dict(result)
        if use_uvd_flow:
            assert uvd_flow_surface is not None
            combined_result["uvd_flow_d_out_of_range_fraction"] = (
                uvd_flow_surface["d_out_of_range_fraction"]
            )
            combined_result["uvd_flow_variance_rejected_fraction"] = (
                uvd_flow_surface["variance_rejected_fraction"]
            )
        combined_result["full_weight"] = torch.as_tensor(
            full_weight, device=device
        )
        combined_result["guidance_crop_active"] = torch.as_tensor(
            float(crop_indices is not None), device=device
        )
        shared_timestep = result.get("sampled_timestep")
        for name, processor in self.regional_prompt_processors.items():
            configured_region_weight = float(
                regional.get(f"{name}_weight", 1.0)
            )
            region_weight = configured_region_weight
            if configured_region_weight <= 0.0:
                continue
            region_mask = self._regional_guidance_mask(name, batch, output)
            protected_face_loss = False
            if name == "face":
                region_mask, protected_face_loss = (
                    self._mask_protected_pixels_from_face_guidance(
                        region_mask, batch, output
                    )
                )
            region_prepared = self._prepare_sdedit_images(
                rgb_full,
                rgb_full,
                control_full,
                region_mask,
                crop=bool(regional.get("crop_regions", True)),
                padding_override=int(regional.get("crop_padding", 8)),
                square=bool(regional.get("square_crops", True)),
                crop_boxes=regional_crop_boxes.get(name),
                return_crop_plan=use_uvd_flow,
            )
            if region_prepared is None:
                combined_result[f"regional_{name}_active"] = torch.zeros(
                    (), device=device
                )
                continue
            if use_uvd_flow:
                (
                    region_rgb,
                    _,
                    region_control,
                    region_guidance_mask,
                    region_indices,
                    region_crop_plan,
                ) = region_prepared
            else:
                (
                    region_rgb,
                    _,
                    region_control,
                    region_guidance_mask,
                    region_indices,
                ) = region_prepared
            if (
                bool(regional.get("full_crop_loss", True))
                and not protected_face_loss
            ):
                region_guidance_mask = torch.ones_like(
                    region_guidance_mask
                )
            region_metadata = {
                key: value[region_indices]
                for key, value in metadata_full.items()
            }
            region_timestep = shared_timestep
            if use_uvd_flow and torch.is_tensor(region_timestep):
                region_timestep = region_timestep.reshape(-1)[0]
            elif (
                torch.is_tensor(region_timestep)
                and region_timestep.numel() == rgb_full.shape[0]
            ):
                region_timestep = region_timestep[region_indices]
            region_extra_guidance: dict[str, torch.Tensor] = {}
            if use_uvd_flow:
                assert uvd_flow_surface is not None
                region_extra_guidance.update(
                    self._prepare_uvd_flow_correspondence(
                        uvd_flow_surface,
                        region_indices,
                        region_crop_plan,
                    )
                )
            region_result = self.guidance(
                self.true_global_step,
                region_rgb,
                region_control,
                processor(),
                **region_metadata,
                rgb_as_latents=False,
                guidance_mask=region_guidance_mask,
                guidance_mask_background_weight=0.0,
                guidance_timestep=region_timestep,
                use_control=bool(
                    regional.get("regional_use_control", True)
                ),
                **region_extra_guidance,
            )
            combined_loss = (
                combined_loss + region_weight * region_result["loss_sds"]
            )
            combined_result[f"regional_{name}_weight"] = torch.as_tensor(
                region_weight, device=device
            )
            combined_result[
                f"regional_{name}_configured_weight"
            ] = torch.as_tensor(configured_region_weight, device=device)
            combined_result[f"regional_{name}_active"] = torch.as_tensor(
                float(region_indices.numel()), device=device
            )
            for key, value in region_result.items():
                combined_result[f"regional_{name}_{key}"] = value
        weight = (
            0.0
            if self.true_global_step < int(self.cfg.guidance_warmup_steps)
            else float(self.cfg.diffusion_weight)
        )
        return combined_loss * weight, combined_result, weight

    @staticmethod
    def _select_uvd_flow_layer(
        contribution: torch.Tensor,
        alpha: torch.Tensor,
        variance: torch.Tensor,
        *,
        alpha_threshold: float,
        contribution_threshold: float,
        dominance_ratio: float,
        max_uvd_variance: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose the visible semantic winner without rear-layer fallback."""

        if (
            contribution.shape != alpha.shape
            or contribution.shape != variance.shape
            or contribution.ndim != 3
            or contribution.shape[0] < 2
        ):
            raise ValueError(
                "UVD-SFD layer statistics must share shape LxHxW with L>=2"
            )
        visible_candidate = (
            (alpha >= alpha_threshold)
            & (contribution >= contribution_threshold)
        )
        score = torch.where(
            visible_candidate,
            contribution,
            torch.zeros_like(contribution),
        )
        top_values, top_indices = score.topk(2, dim=0)
        best, second = top_values[0], top_values[1]
        winner = top_indices[0]
        dominant = (best > 0.0) & (best >= dominance_ratio * second)
        variance_ok = variance <= max_uvd_variance
        winner_variance_ok = torch.gather(
            variance_ok, 0, winner[None]
        )[0]
        # Never fall through to a weak rear layer merely because the truly
        # visible layer spans a seam or multiple surfaces. In that case the
        # pixel has no trustworthy single UVD identity.
        variance_rejected = dominant & ~winner_variance_ok
        return winner, dominant & winner_variance_ok, variance_rejected

    @torch.no_grad()
    def _render_surface_memory_correspondence(
        self,
        batch: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Rasterize frozen-source FLAME layer/UV keys for joint SDEdit.

        Every layer is rendered in isolation for its alpha-normalized UV,
        variance, and expected depth.  A second full-scene one-hot render
        measures the layer's actual front-to-back contribution.  The shared
        ``compose_layered_surface`` selector then rejects weak, ambiguous, or
        rear-depth candidates before any diffusion token can enter memory.
        Geometry and opacity come from the same phase-boundary snapshot as
        the image encoded by SDEdit, so correspondence cannot drift while the
        trainable avatar changes.
        """

        if not self.surface_sdedit_enabled:
            raise RuntimeError(
                "Surface-memory correspondence requested while FLAME "
                "surface SDEdit is disabled"
            )
        if self._sdedit_reference_state is None:
            raise RuntimeError(
                "Surface-memory correspondence requires the frozen SDEdit "
                "reference state"
            )
        state = self._sdedit_reference_state
        masks = self._canonical_surface_masks(face_idx=state["face_idx"])
        gaussian_uv = state["uv"].detach().float()
        uv_moment_color = torch.cat(
            (
                gaussian_uv,
                gaussian_uv.square().sum(dim=1, keepdim=True),
            ),
            dim=1,
        )
        packed, correspondence_opacity = self._frozen_reference_geometry(
            state,
            batch,
            reference_pose=False,
        )
        config = self.surface_memory_config
        correspondence_opacity = correspondence_opacity.detach().float()
        opacity_floor = float(config.get("opacity_floor", 0.0))
        if opacity_floor > 0.0:
            correspondence_opacity = correspondence_opacity.clone()
            oral = ~masks["face"]
            correspondence_opacity[oral] = correspondence_opacity[
                oral
            ].clamp_min(opacity_floor)
        correspondence_opacity = correspondence_opacity.clamp(0.0, 1.0)
        background = torch.zeros(3, device=self.gaussian.device)
        alpha_threshold = float(config.get("alpha_threshold", 0.005))
        contribution_threshold = float(
            config.get("contribution_threshold", 0.005)
        )
        variance_threshold = float(
            config.get("variance_threshold", 0.0025)
        )
        dominance_ratio = float(config.get("dominance_ratio", 1.10))
        configured_depth_tolerance = config.get("depth_tolerance", 0.01)
        depth_tolerance = (
            None
            if configured_depth_tolerance is None
            else float(configured_depth_tolerance)
        )

        surface_uv: list[torch.Tensor] = []
        surface_layer: list[torch.Tensor] = []
        surface_depth: list[torch.Tensor] = []
        surface_visibility: list[torch.Tensor] = []
        ambiguous_fractions: list[torch.Tensor] = []
        with torch.cuda.amp.autocast(enabled=False):
            for camera in self._cameras(batch):
                contributions: dict[str, torch.Tensor] = {}
                composite_alpha: Optional[torch.Tensor] = None
                for first_layer in range(0, len(SURFACE_LAYER_NAMES), 3):
                    group = SURFACE_LAYER_NAMES[
                        first_layer : first_layer + 3
                    ]
                    colors = torch.zeros(
                        self.gaussian.num_gs,
                        3,
                        device=self.gaussian.device,
                        dtype=torch.float32,
                    )
                    for channel, name in enumerate(group):
                        colors[masks[name], channel] = 1.0
                    package = render(
                        camera,
                        self.gaussian,
                        self.pipe,
                        background,
                        override_color=colors,
                        override_opacity=correspondence_opacity,
                        precomputed_geometry=packed,
                    )
                    if composite_alpha is None:
                        composite_alpha = package["alpha_3dgs"].clamp(
                            0.0, 1.0
                        )
                    for channel, name in enumerate(group):
                        contributions[name] = package["render"][
                            channel : channel + 1
                        ].clamp(0.0, 1.0)

                layers: dict[str, LayerSurfaceBuffers] = {}
                for name in SURFACE_LAYER_NAMES:
                    layer_opacity = correspondence_opacity * masks[
                        name
                    ][:, None].to(correspondence_opacity.dtype)
                    package = render(
                        camera,
                        self.gaussian,
                        self.pipe,
                        background,
                        override_color=uv_moment_color,
                        override_opacity=layer_opacity,
                        precomputed_geometry=packed,
                    )
                    alpha = package["alpha_3dgs"].clamp(0.0, 1.0)
                    moments = normalize_alpha_weighted(
                        package["render"].unsqueeze(0),
                        alpha.unsqueeze(0),
                    )[0]
                    uv = moments[:2].clamp(0.0, 1.0)
                    variance = (
                        moments[2:3]
                        - uv.square().sum(dim=0, keepdim=True)
                    ).clamp_min(0.0)
                    depth = normalize_alpha_weighted(
                        package["depth_3dgs"].unsqueeze(0),
                        alpha.unsqueeze(0),
                    )[0]
                    layers[name] = LayerSurfaceBuffers(
                        uv=uv.unsqueeze(0),
                        variance=variance.unsqueeze(0),
                        depth=depth.unsqueeze(0),
                        alpha=alpha.unsqueeze(0),
                        contribution=contributions[name].unsqueeze(0),
                    )

                composed = compose_layered_surface(
                    layers,
                    alpha_threshold=alpha_threshold,
                    contribution_threshold=contribution_threshold,
                    variance_threshold=variance_threshold,
                    depth_tolerance=depth_tolerance,
                    dominance_ratio=dominance_ratio,
                )
                if composite_alpha is None:
                    raise RuntimeError(
                        "Surface-memory correspondence rendered no layers"
                    )
                visibility = torch.minimum(
                    composed.surface_contribution[0], composite_alpha
                ) * composed.validity[0]
                surface_uv.append(composed.surface_uv[0])
                surface_layer.append(composed.layer_id[0])
                surface_depth.append(composed.surface_depth[0])
                surface_visibility.append(visibility)
                ambiguous_fractions.append(
                    composed.ambiguous[0].float().mean()
                )

        return {
            "surface_uv": torch.stack(surface_uv),
            "surface_layer": torch.stack(surface_layer),
            "surface_depth": torch.stack(surface_depth),
            "surface_visibility": torch.stack(surface_visibility),
            "ambiguous_fraction": torch.stack(
                ambiguous_fractions
            ).mean(),
        }

    @torch.no_grad()
    def _render_uvd_flow_surface(
        self,
        batch: Mapping[str, Any],
        output: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Render one ambiguity-rejected canonical UVD surface per pixel.

        Numeric UV alone is not a unique surface identity in this avatar: the
        generated upper/lower teeth intentionally overlap in UV, and normal
        offsets distinguish multiple splats on a surface.  We therefore use
        ``(semantic layer, u, v, normalized d)`` and rank layers by their
        occlusion-aware contribution in the complete scene.
        """

        if self.uvd_flow_d_range is None:
            raise RuntimeError("UVD-SFD has no fixed canonical d range")
        masks = self._uvd_flow_surface_masks()
        lower, upper = self.uvd_flow_d_range.unbind()
        canonical_d = self.gaussian._d.detach().float().reshape(
            self.gaussian.num_gs, -1
        )[:, :1]
        d_in_range = (canonical_d >= lower) & (canonical_d <= upper)
        d_normalized = (
            (canonical_d - lower) / (upper - lower)
        ).clamp(0.0, 1.0)
        uvd_color = torch.cat(
            (self.gaussian.get_uv.detach().float(), d_normalized), dim=-1
        )
        packed = (
            output["means"].detach(),
            output["scales"].detach(),
            output["rotations"].detach(),
        )
        scene_opacity = self.gaussian.get_opacity.detach().float()
        opacity_floor = float(
            self.cfg.uvd_surface_flow.get("opacity_floor", 0.0)
        )
        if not 0.0 <= opacity_floor <= 1.0:
            raise ValueError("uvd_surface_flow.opacity_floor must be in [0, 1]")
        if opacity_floor:
            scene_opacity = scene_opacity.clamp_min(opacity_floor)
        # Clamping an escaped offset into the edge bin would alias unrelated
        # surfaces. Exclude it from correspondence while retaining it as an
        # occluder in the full-scene contribution pass; existing UVD
        # proximal/barrier terms remain responsible for pulling it back.
        correspondence_opacity = scene_opacity * d_in_range.to(
            scene_opacity.dtype
        )
        background = torch.zeros(3, device=self.gaussian.device)
        alpha_threshold = float(
            self.cfg.uvd_surface_flow.get("alpha_threshold", 0.05)
        )
        contribution_threshold = float(
            self.cfg.uvd_surface_flow.get(
                "contribution_threshold", 0.01
            )
        )
        dominance_ratio = float(
            self.cfg.uvd_surface_flow.get("dominance_ratio", 1.10)
        )
        max_uvd_variance = float(
            self.cfg.uvd_surface_flow.get("max_uvd_variance", 0.0025)
        )
        surface_uvd, surface_layer, surface_confidence = [], [], []
        variance_rejected = []
        with torch.cuda.amp.autocast(enabled=False):
            for camera in self._cameras(batch):
                # Rendering one-hot layer colors while all Gaussians remain
                # present gives the front-to-back, occlusion-aware semantic
                # contribution—not the misleading alpha of an isolated layer.
                contributions = []
                for first_layer in range(0, len(SURFACE_LAYER_NAMES), 3):
                    semantic_color = torch.zeros(
                        (self.gaussian.num_gs, 3),
                        dtype=torch.float32,
                        device=self.gaussian.device,
                    )
                    for channel, layer_index in enumerate(
                        range(
                            first_layer,
                            min(first_layer + 3, len(SURFACE_LAYER_NAMES)),
                        )
                    ):
                        valid_points = (
                            masks[SURFACE_LAYER_NAMES[layer_index]]
                            & d_in_range[:, 0]
                        )
                        semantic_color[valid_points, channel] = 1.0
                    package = render(
                        camera,
                        self.gaussian,
                        self.pipe,
                        background,
                        override_color=semantic_color,
                        override_opacity=scene_opacity,
                        precomputed_geometry=packed,
                    )
                    channels = min(
                        3, len(SURFACE_LAYER_NAMES) - first_layer
                    )
                    contributions.extend(
                        package["render"][channel]
                        for channel in range(channels)
                    )

                layer_uvd, layer_alpha, layer_variance = [], [], []
                for name in SURFACE_LAYER_NAMES:
                    layer_opacity = (
                        correspondence_opacity
                        * masks[name][:, None].to(
                            correspondence_opacity.dtype
                        )
                    )
                    package = render(
                        camera,
                        self.gaussian,
                        self.pipe,
                        background,
                        override_color=uvd_color,
                        override_opacity=layer_opacity,
                        precomputed_geometry=packed,
                    )
                    alpha = package["alpha_3dgs"].clamp(0.0, 1.0)
                    normalized = torch.where(
                        alpha > 1.0e-6,
                        package["render"] / alpha.clamp_min(1.0e-6),
                        torch.zeros_like(package["render"]),
                    ).clamp(0.0, 1.0)
                    second_package = render(
                        camera,
                        self.gaussian,
                        self.pipe,
                        background,
                        override_color=uvd_color.square(),
                        override_opacity=layer_opacity,
                        precomputed_geometry=packed,
                    )
                    second_moment = torch.where(
                        alpha > 1.0e-6,
                        second_package["render"]
                        / alpha.clamp_min(1.0e-6),
                        torch.zeros_like(second_package["render"]),
                    ).clamp(0.0, 1.0)
                    layer_uvd.append(normalized)
                    layer_alpha.append(alpha[0])
                    layer_variance.append(
                        (second_moment - normalized.square())
                        .clamp_min(0.0)
                        .amax(dim=0)
                    )

                contribution = torch.stack(contributions).clamp(0.0, 1.0)
                alpha = torch.stack(layer_alpha).clamp(0.0, 1.0)
                variance = torch.stack(layer_variance)
                winner, valid, rejected = self._select_uvd_flow_layer(
                    contribution,
                    alpha,
                    variance,
                    alpha_threshold=alpha_threshold,
                    contribution_threshold=contribution_threshold,
                    dominance_ratio=dominance_ratio,
                    max_uvd_variance=max_uvd_variance,
                )
                variance_rejected.append(rejected.float().mean())
                winner_contribution = torch.gather(
                    contribution, 0, winner[None]
                )[0]
                stacked_uvd = torch.stack(layer_uvd)
                selected_uvd = torch.gather(
                    stacked_uvd,
                    0,
                    winner[None, None].expand(
                        1, stacked_uvd.shape[1], -1, -1
                    ),
                )[0]
                selected_layer = torch.where(
                    valid,
                    winner,
                    torch.full_like(winner, -1),
                )[None]
                selected_confidence = torch.where(
                    valid,
                    winner_contribution,
                    torch.zeros_like(winner_contribution),
                )[None]
                surface_uvd.append(
                    torch.where(
                        valid[None], selected_uvd, torch.zeros_like(selected_uvd)
                    )
                )
                surface_layer.append(selected_layer)
                surface_confidence.append(selected_confidence)
        return {
            "surface_uvd": torch.stack(surface_uvd),
            "surface_layer": torch.stack(surface_layer),
            "surface_confidence": torch.stack(surface_confidence),
            "d_out_of_range_fraction": (~d_in_range).float().mean(),
            "variance_rejected_fraction": torch.stack(
                variance_rejected
            ).mean(),
        }

    def _lowpass(self, image: torch.Tensor) -> torch.Tensor:
        kernel = max(int(self.cfg.reference_kernel), 1)
        if kernel % 2 == 0:
            kernel += 1
        if kernel > 1:
            image = F.avg_pool2d(image, kernel, stride=1, padding=kernel // 2)
        size = int(self.cfg.reference_resolution)
        return F.interpolate(image, (size, size), mode="bilinear", align_corners=False)

    def _reference_loss(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dynamic_pose = self._batch_pose(batch)
        reference = self.forward(
            batch,
            background=torch.ones(3, device=self.gaussian.device),
            reference_pose=True,
        )
        if self._uses_continuous_camera(batch):
            target_rgb_hwc, target_alpha_hwc, _ = (
                self._render_initial_continuous_reference(
                    batch, reference_pose=True
                )
            )
            target_rgb = self._bchw(
                target_rgb_hwc, self.gaussian.device
            )[:, :3]
            target_alpha = self._bchw(
                target_alpha_hwc, self.gaussian.device
            )[:, :1]
        else:
            target_rgb = self._bchw(
                batch["reference_rgb"], self.gaussian.device
            )[:, :3]
            target_alpha = self._bchw(
                batch["reference_alpha"], self.gaussian.device
            )[:, :1]
        self._set_pose(*dynamic_pose)

        prediction = self._lowpass(reference["comp_rgb_raw"].permute(0, 3, 1, 2))
        target = self._lowpass(target_rgb)
        alpha_prediction = self._lowpass(reference["alpha"].permute(0, 3, 1, 2))
        alpha_target = self._lowpass(target_alpha)

        # Core foreground is reliable identity evidence.  Background and the
        # antialiased silhouette retain a weak weight instead of a hard mask.
        confidence = 0.1 + 0.9 * alpha_target.detach()
        rgb_error = F.smooth_l1_loss(prediction, target, reduction="none")
        rgb_loss = (rgb_error * confidence).sum() / (confidence.sum() * 3.0).clamp_min(1.0)
        alpha_loss = F.smooth_l1_loss(alpha_prediction, alpha_target)
        total = rgb_loss + float(self.cfg.reference_alpha_weight) * alpha_loss
        return total, rgb_loss, alpha_loss

    @staticmethod
    def _masked_smooth_l1(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError("Masked loss prediction and target shapes differ")
        if mask.shape[:-1] != prediction.shape[:-1] or mask.shape[-1] != 1:
            raise ValueError("Masked loss expects a matching BxHxWx1 mask")
        error = F.smooth_l1_loss(prediction, target, reduction="none")
        weight = mask.to(device=error.device, dtype=error.dtype).expand_as(error)
        return (error * weight).sum() / weight.sum().clamp_min(1.0)

    def _mouth_screen_preservation_loss(
        self,
        batch: Mapping[str, Any],
        output: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Keep trainable splats from changing or occluding the frozen mouth."""

        zero = output["alpha"].sum() * 0.0
        config = self.cfg.full_protection.get(
            "mouth_screen_preservation", {}
        )
        if not (
            self._full_region_protection_enabled()
            and bool(config.get("enabled", False))
        ):
            return zero, {}

        target_rgb, target_alpha, target_depth = (
            self._render_initial_dynamic_reference(batch)
        )
        mouth_mask = self._render_point_mask(
            batch, output, self.mouth_guidance_point_mask
        )
        mouth_mask = self._dilate_mask(
            mouth_mask, int(config.get("dilation", 0))
        ).detach()
        alpha_threshold = float(config.get("alpha_threshold", 0.05))
        support = mouth_mask * (target_alpha > alpha_threshold).to(
            mouth_mask.dtype
        )

        rgb = self._masked_smooth_l1(
            output.get("comp_rgb_raw", output["comp_rgb"]),
            target_rgb,
            support,
        )
        alpha = self._masked_smooth_l1(
            output["alpha"], target_alpha, support
        )

        current_alpha = output["alpha"]
        current_expected_depth = torch.where(
            current_alpha > alpha_threshold,
            output["depth"] / current_alpha.clamp_min(alpha_threshold),
            torch.zeros_like(output["depth"]),
        )
        target_expected_depth = torch.where(
            target_alpha > alpha_threshold,
            target_depth / target_alpha.clamp_min(alpha_threshold),
            torch.zeros_like(target_depth),
        )
        # Compare relative depth so the weight is independent of the global
        # scene scale. A new splat in front of the lips/teeth changes this
        # expected depth even when its RGB happens to look plausible.
        depth_scale = target_expected_depth.abs().clamp_min(1.0e-3)
        depth = self._masked_smooth_l1(
            current_expected_depth / depth_scale,
            target_expected_depth / depth_scale,
            support,
        )
        parts = {
            "mouth_screen_rgb": float(config.get("rgb_weight", 0.0)) * rgb,
            "mouth_screen_alpha": float(config.get("alpha_weight", 0.0))
            * alpha,
            "mouth_screen_depth": float(config.get("depth_weight", 0.0))
            * depth,
        }
        return sum(parts.values()), parts

    def _silhouette_spill_loss(
        self,
        batch: Mapping[str, Any],
        output: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Penalize dynamic-pose opacity outside the dilated fixed silhouette."""

        target = self._reference_silhouette_mask(batch, output)
        outside = (1.0 - target).detach()
        alpha = output["alpha"].clamp(0.0, 1.0)
        return (alpha.square() * outside).sum() / outside.sum().clamp_min(1.0)

    def _point_weights(self) -> torch.Tensor:
        weights = torch.ones(self.gaussian.num_gs, device=self.gaussian.device)
        if self.optimization_stage == "mouth":
            # The closed-mouth Stage-1 views provide weak evidence for dental
            # appearance.  Keep a trust region, but let those points move more
            # freely during their dedicated open-mouth pass.
            weights[self.dental_point_mask] = 0.1
        if self.mouth_slice is not None:
            # Static views contain little evidence for the cavity; its seeded
            # support must be freer than visible skin/hair Gaussians.
            weights[self.mouth_slice] = 0.1
        return weights

    @staticmethod
    def _weighted_parameter_loss(
        current: torch.Tensor,
        initial: torch.Tensor,
        point_weights: torch.Tensor,
    ) -> torch.Tensor:
        error = F.smooth_l1_loss(current, initial, reduction="none").reshape(current.shape[0], -1).mean(dim=1)
        return (error * point_weights).sum() / point_weights.sum()

    def _proximal_loss(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        weights = self._point_weights()
        cfg = self.cfg.proximal
        parts = {
            "prox_uv": float(cfg.get("uv", 0.0))
            * self._weighted_parameter_loss(
                self.gaussian._uv,
                self.initial["uv"],
                weights,
            ),
            "prox_feature": float(cfg["feature"])
            * self._weighted_parameter_loss(
                self.gaussian._features_dc,
                self.initial["feature"],
                weights,
            ),
            "prox_opacity": float(cfg["opacity"])
            * self._weighted_parameter_loss(
                self.gaussian._opacity,
                self.initial["opacity"],
                weights,
            ),
            "prox_d": float(cfg["d"])
            * self._weighted_parameter_loss(
                self.gaussian._d, self.initial["d"], weights
            ),
            "prox_scale": float(cfg["scale"])
            * self._weighted_parameter_loss(
                self.gaussian._scaling,
                self.initial["scale"],
                weights,
            ),
        }
        rotation = F.normalize(self.gaussian._rotation, dim=-1)
        rotation_error = 1.0 - (rotation * self.initial["rotation"]).sum(dim=-1).abs()
        parts["prox_rotation"] = float(cfg["rotation"]) * (rotation_error * weights).sum() / weights.sum()
        return sum(parts.values()), parts

    def _geometry_barrier(
        self, output: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        def violation_mean_square(excess: torch.Tensor) -> torch.Tensor:
            active = excess > 0.0
            return excess.square().sum() / active.sum().clamp_min(1)

        world_scales = output["scales"]
        trainable_mask = None
        if self._full_scale_stability_enabled():
            trainable_mask = self._active_trainable_point_mask().to(
                device=world_scales.device, dtype=torch.bool
            )
        barrier_scales = (
            world_scales[trainable_mask]
            if trainable_mask is not None
            else world_scales
        )
        if barrier_scales.numel():
            world_scale = barrier_scales.amax(dim=-1)
            global_barrier = violation_mean_square(
                F.relu(world_scale - float(self.cfg.max_world_scale))
            )
        else:
            global_barrier = world_scales.sum() * 0.0
        parts = {"world_scale": float(self.cfg.world_scale_weight) * global_barrier}
        if trainable_mask is not None and bool(trainable_mask.any().item()):
            stability = self.cfg.scale_stability
            current = world_scales[trainable_mask].clamp_min(1.0e-12)
            reference = self._reference_world_scales_current_pose(
                world_scales
            )[trainable_mask].clamp_min(1.0e-12)
            growth_weight = float(
                stability.get("reference_growth_weight", 0.0)
            )
            if growth_weight > 0.0:
                growth_limit = float(
                    stability.get("reference_growth_limit", 1.0)
                )
                growth_excess = F.relu(
                    torch.log(current / reference) - math.log(growth_limit)
                )
                parts["world_scale_growth"] = (
                    growth_weight * violation_mean_square(growth_excess)
                )
            current_anisotropy = (
                current.amax(dim=-1) / current.amin(dim=-1)
            )
            absolute_anisotropy_weight = float(
                stability.get("world_anisotropy_weight", 0.0)
            )
            maximum_anisotropy = stability.get(
                "max_world_anisotropy"
            )
            if (
                absolute_anisotropy_weight > 0.0
                and maximum_anisotropy is not None
            ):
                absolute_excess = F.relu(
                    torch.log(current_anisotropy)
                    - math.log(float(maximum_anisotropy))
                )
                parts["world_anisotropy"] = (
                    absolute_anisotropy_weight
                    * violation_mean_square(absolute_excess)
                )
            anisotropy_weight = float(
                stability.get("reference_anisotropy_weight", 0.0)
            )
            if anisotropy_weight > 0.0:
                anisotropy_limit = float(
                    stability.get(
                        "reference_anisotropy_growth_limit", 1.0
                    )
                )
                reference_anisotropy = (
                    reference.amax(dim=-1) / reference.amin(dim=-1)
                )
                anisotropy_excess = F.relu(
                    torch.log(current_anisotropy / reference_anisotropy)
                    - math.log(anisotropy_limit)
                )
                parts["world_anisotropy_growth"] = (
                    anisotropy_weight
                    * violation_mean_square(anisotropy_excess)
                )
        world_scale = world_scales.amax(dim=-1)
        if bool(self.dental_point_mask.any().item()):
            mouth_scale = world_scale[self.dental_point_mask]
            mouth_d = self.gaussian._d[self.dental_point_mask, 0]
            mouth_barrier = (
                violation_mean_square(
                    F.relu(mouth_scale - float(self.cfg.mouth["max_scale"]))
                )
                + violation_mean_square(
                    F.relu(
                        mouth_d.abs() - float(self.cfg.mouth["max_abs_d"])
                    )
                )
            )
            parts["mouth"] = float(self.cfg.mouth["barrier_weight"]) * mouth_barrier
        return sum(parts.values()), parts

    def _chroma_loss(
        self, output: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = output["comp_rgb_raw"].clamp(0.0, 1.0)
        maximum = image.amax(dim=-1)
        minimum = image.amin(dim=-1)
        saturation = (maximum - minimum) / maximum.clamp_min(1.0e-4)
        alpha = output["alpha"][..., 0].detach()
        excess = F.relu(saturation - float(self.cfg.chroma_threshold)).square()
        penalty = (excess * alpha).sum() / alpha.sum().clamp_min(1.0)
        mean = (saturation * alpha).sum() / alpha.sum().clamp_min(1.0)
        return float(self.cfg.chroma_weight) * penalty, mean

    def training_step(self, batch, batch_idx):
        self._optimizer_stepped_this_batch = bool(
            self.automatic_optimization
        )
        step = int(self.true_global_step)
        surface_joint_batch = bool(
            self.surface_sdedit_enabled and self._sdedit_active()
        )
        if self.surface_sdedit_enabled and not surface_joint_batch:
            # The data loader returns K cameras so the SDEdit phase can build
            # one joint memory.  Keep the first ISM/UVD-SFD phase exactly
            # single-view; the random cyclic camera sampler makes this row an
            # unbiased draw from the selected elevation ring.
            batch = self._select_batch_views(
                batch, torch.zeros(1, dtype=torch.long)
            )
        elif surface_joint_batch:
            actual_views = int(torch.as_tensor(batch["c2w"]).shape[0])
            expected_views = int(
                self.surface_memory_config.get("views", 4)
            )
            if actual_views != expected_views:
                raise RuntimeError(
                    "FLAME surface SDEdit requires one joint batch with "
                    f"{expected_views} views, got {actual_views}. Set "
                    "data.batch_size to sdedit.surface_memory.views."
                )
        self._update_learning_rates()
        open_flags = torch.as_tensor(
            batch.get("is_open_mouth", True), dtype=torch.bool
        ).reshape(-1)
        if open_flags.numel() > 1 and not bool(
            torch.all(open_flags == open_flags[0]).item()
        ):
            raise ValueError(
                "Every calibrated view in a refinement batch must share one pose"
            )
        self._active_open_mouth = bool(open_flags[0].item())
        # Check every sampled expression/jaw pose before rendering it.  Any
        # optional hard guard is evaluated in aligned world space; face-local
        # scales are normalized by parent-face size and have no global limit.
        if (
            self.optimization_stage == "mouth"
            and bool(self.cfg.geometry_stability.get("enabled", False))
        ):
            self._set_pose(*self._batch_pose(batch))
            capped, scale_before = self._cap_current_dental_world_scale()
            self.log("train/pre_render_scale_caps", float(capped))
            self.log(
                "train/pre_render_dental_world_scale", float(scale_before)
            )
        elif self._full_scale_stability_enabled():
            # Cap in the newly sampled pose before it is ever rasterized.  A
            # cap only after optimizer.step() still exposes an optimizer
            # outlier to the next batch and writes it into previews.
            self._set_pose(*self._batch_pose(batch))
            scale_report = self._stabilize_current_full_scale()
            self.log(
                "train/pre_render_world_scale_caps",
                float(scale_report["world_capped"]),
            )
            self.log(
                "train/pre_render_world_scale_max",
                float(scale_report["world_before"]),
            )
        # The phase-boundary snapshot must be taken only after the current
        # pose has passed the geometry guards above.
        self._maybe_write_first_phase_artifacts()
        self._reset_adam_state_at_sdedit_boundary()
        # Use one deterministic white context for ISM, SDEdit source renders,
        # and the trainable avatar render.  Random RGB backgrounds change the
        # VAE/UNet input even when the explicit background loss is disabled.
        background = self.background
        sdedit_source = None
        if self._sdedit_active():
            if (
                self._sdedit_reference_state is None
                and step > self.sdedit_start_step
            ):
                threestudio.warn(
                    "SDEdit snapshot was absent while resuming after its phase "
                    "boundary; the restored Gaussian state is used as the fixed source."
                )
            sdedit_source = self._render_sdedit_reference(batch, background)
        output = self.forward(batch, background=background)

        diffusion, guidance_out, diffusion_weight = self._diffusion_loss(
            batch, output, sdedit_source
        )
        reference, reference_rgb, reference_alpha = self._reference_loss(batch)
        warmup = int(self.cfg.guidance_warmup_steps)
        if step < warmup or bool(torch.isnan(self.reference_baseline).item()):
            # During warmup every Gaussian lr is exactly zero.  This running
            # mean therefore estimates E_view[L_ref(theta_0)] instead of using
            # one arbitrary camera batch as the constraint baseline.
            with torch.no_grad():
                count = int(self.reference_baseline_count.item())
                value = reference.detach()
                if count == 0:
                    self.reference_baseline.copy_(value)
                else:
                    self.reference_baseline.add_((value - self.reference_baseline) / float(count + 1))
                self.reference_baseline_count.add_(1)
        budget = self.reference_baseline + float(self.cfg.reference_tolerance)
        if step < warmup:
            self._reference_violation = torch.zeros_like(reference.detach())
        else:
            self._reference_violation = reference.detach() - budget
        proximal, proximal_parts = self._proximal_loss()
        barrier, barrier_parts = self._geometry_barrier(output)
        mouth_screen, mouth_screen_parts = (
            self._mouth_screen_preservation_loss(batch, output)
        )
        chroma, mean_chroma = self._chroma_loss(output)
        spill_weight = float(self.cfg.silhouette_spill_weight)
        silhouette_spill = (
            self._silhouette_spill_loss(batch, output)
            if spill_weight > 0.0
            else output["alpha"].sum() * 0.0
        )
        silhouette = (
            float(self.cfg.silhouette_weight) * reference_alpha
            + spill_weight * silhouette_spill
        )
        total = (
            diffusion
            + self.reference_dual.detach() * reference
            + silhouette
            + proximal
            + barrier
            + mouth_screen
            + chroma
        )

        logs = {
            "loss": total,
            "diffusion": diffusion,
            "diffusion_weight": diffusion_weight,
            "reference": reference,
            "reference_rgb": reference_rgb,
            "reference_alpha": reference_alpha,
            "silhouette": silhouette,
            "silhouette_spill": silhouette_spill,
            "reference_budget": budget,
            "reference_dual": self.reference_dual,
            "proximal": proximal,
            "barrier": barrier,
            "mouth_screen": mouth_screen,
            "chroma": chroma,
            "mean_chroma": mean_chroma,
            "open_mouth": float(self._active_open_mouth),
            "sdedit_phase": float(self._sdedit_active()),
            "surface_sdedit": float(surface_joint_batch),
        }
        logs.update(proximal_parts)
        logs.update(barrier_parts)
        logs.update(mouth_screen_parts)
        for name, value in logs.items():
            self.log(f"train/{name}", value.detach() if torch.is_tensor(value) else value)
        for name, value in guidance_out.items():
            if torch.is_tensor(value) and value.numel() == 1:
                self.log(f"train/guidance_{name}", value.detach())
        if not self.automatic_optimization:
            optimizer = self.optimizers()
            phase = "sdedit" if self._sdedit_active() else "ism"
            accumulation = (
                1
                if phase == "sdedit"
                else self.ism_accumulate_grad_batches
            )
            if self._manual_phase != phase:
                if self._manual_micro_step != 0:
                    raise RuntimeError(
                        "Refinement phase changed with unfinished gradient "
                        "accumulation"
                    )
                optimizer.zero_grad(set_to_none=True)
                self._manual_phase = phase
            self.manual_backward(total / float(accumulation))
            self._manual_micro_step += 1
            if self._manual_micro_step >= accumulation:
                # train_all.py records densification statistics from the last
                # of its three ISM samples only; accumulating all three here
                # made cloning substantially more aggressive than the source.
                if phase == "ism":
                    self._accumulate_full_densification_stats(output)
                progress_before = self._optimizer_progress(optimizer)
                optimizer.step()
                progress_after = self._optimizer_progress(optimizer)
                if progress_after <= progress_before:
                    self._consecutive_skipped_optimizer_steps += 1
                    limit = int(
                        self.cfg.optimization.get(
                            "max_consecutive_skipped_steps", 8
                        )
                    )
                    if self._consecutive_skipped_optimizer_steps >= limit:
                        precision_plugin = getattr(
                            self.trainer, "precision_plugin", None
                        )
                        scaler = getattr(precision_plugin, "scaler", None)
                        scale = (
                            float(scaler.get_scale())
                            if scaler is not None
                            else None
                        )
                        raise RuntimeError(
                            "The Gaussian optimizer did not execute for "
                            f"{self._consecutive_skipped_optimizer_steps} "
                            "consecutive steps. This usually means non-finite "
                            "gradients or fp16 GradScaler overflow"
                            + (
                                f" (current scale={scale})."
                                if scale is not None
                                else "."
                            )
                        )
                else:
                    self._consecutive_skipped_optimizer_steps = 0
                    self._maybe_densify_full(int(progress_after))
                optimizer.zero_grad(set_to_none=True)
                self._manual_micro_step = 0
                self._optimizer_stepped_this_batch = True
            return {"loss": total.detach()}
        return {"loss": total}

    # ------------------------------------------------------------------
    # Small residual optimizer and adaptive reference constraint.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _reset_proximal_center_after_topology_change(self) -> None:
        """Make every proximal tensor match a newly densified topology."""

        self.initial = {
            "uv": self.gaussian._uv.detach().clone(),
            "feature": self.gaussian._features_dc.detach().clone(),
            "opacity": self.gaussian._opacity.detach().clone(),
            "d": self.gaussian._d.detach().clone(),
            "scale": self.gaussian._scaling.detach().clone(),
            "rotation": F.normalize(
                self.gaussian._rotation.detach(), dim=-1
            ),
        }

    @torch.no_grad()
    def _accumulate_full_densification_stats(
        self, output: Mapping[str, Any]
    ) -> None:
        if not self.densification_enabled or self._sdedit_active():
            return
        points = output.get("viewspace_points", ())
        visibility = output.get("visibility_filter", ())
        radii = output.get("radii", ())
        for screen_points, visible, screen_radii in zip(
            points, visibility, radii
        ):
            gradient = screen_points.grad
            if gradient is None:
                raise RuntimeError(
                    "Densification requested but the renderer produced no "
                    "screen-space gradient"
                )
            visible = visible.detach()
            self.gaussian.max_radii2D[visible] = torch.maximum(
                self.gaussian.max_radii2D[visible],
                screen_radii.detach()[visible],
            )
            # The UVD model's method accepts the already-extracted screen
            # gradient (Stage 1 uses the same API).
            self.gaussian.add_densification_stats(
                gradient.detach(), visible
            )

    @torch.no_grad()
    def _maybe_densify_full(self, optimizer_step: int) -> None:
        if (
            not self.densification_enabled
            or self._sdedit_active()
            or optimizer_step not in self._densification_steps
            or optimizer_step in self._completed_densification_steps
        ):
            return
        cfg = self.cfg.densification
        self.gaussian.percent_dense = float(
            cfg.get("percent_dense", 0.01)
        )
        if self.gaussian.percent_dense <= 0.0:
            raise ValueError("densification.percent_dense must be positive")
        dental = self.dental_point_mask.detach().clone()
        protected_mask = torch.zeros_like(dental)
        excluded_mask = torch.zeros_like(dental)
        if bool(cfg.get("protect_dental", True)):
            protected_mask |= dental
        if bool(cfg.get("exclude_dental", True)):
            excluded_mask |= dental
        protection = self.cfg.full_protection
        if (
            self._full_region_protection_enabled()
            and bool(protection.get("protect_from_densification", True))
        ):
            if bool(protection.get("freeze_eyes", True)):
                protected_mask |= self.eye_point_mask
                excluded_mask |= self.eye_point_mask
            if bool(protection.get("freeze_mouth", False)):
                protected_mask |= self.mouth_guidance_point_mask
                excluded_mask |= self.mouth_guidance_point_mask
            if bool(protection.get("freeze_dental", True)):
                protected_mask |= dental
                excluded_mask |= dental
        protected = (
            protected_mask if bool(protected_mask.any().item()) else None
        )
        densify_mask = (
            ~excluded_mask if bool(excluded_mask.any().item()) else None
        )
        stats = self.gaussian.densify_and_prune(
            max_grad=float(cfg.get("grad_threshold", 2.0e-4)),
            min_opacity=float(cfg.get("min_opacity", 0.005)),
            extent=float(self.cameras_extent),
            max_screen_size=(
                None
                if cfg.get("max_screen_size", 20.0) is None
                else float(cfg.get("max_screen_size", 20.0))
            ),
            protected_mask=protected,
            densify_mask=densify_mask,
            max_gaussians=int(cfg.get("max_gaussians", 2_000_000)),
        )
        self._refresh_region_masks()
        self._reset_proximal_center_after_topology_change()
        self._completed_densification_steps.add(optimizer_step)
        self.log("train/densified_cloned", float(stats["cloned"]))
        self.log("train/densified_split", float(stats["split"]))
        self.log("train/densified_pruned", float(stats["pruned"]))
        self.log("train/gaussian_count", float(stats["after"]))
        threestudio.info(
            "Full refinement densification at optimizer step "
            f"{optimizer_step}: pruned={stats['pruned']}, "
            f"cloned={stats['cloned']}, split={stats['split']}, "
            f"points={stats['after']}."
        )

    @staticmethod
    def _optimizer_progress(optimizer: Any) -> float:
        """Return Adam's largest executed step, excluding skipped AMP calls."""

        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        maximum = 0.0
        for state in raw_optimizer.state.values():
            value = state.get("step")
            if value is None:
                continue
            if torch.is_tensor(value):
                value = value.detach().item()
            maximum = max(maximum, float(value))
        return maximum

    def _cumulative_optimizer_progress(self, optimizer: Any) -> float:
        """Return executed steps across intentional Adam-state resets."""

        return float(self._optimizer_executed_step_offset) + float(
            self._optimizer_progress(optimizer)
        )

    def configure_optimizers(self):
        # Immutable rig coordinates and identity.
        for parameter in (
            self.gaussian._features_rest,
            self.gaussian._shape,
        ):
            parameter.requires_grad_(False)

        cfg = self.cfg.optimization
        optimize_uv = (
            float(cfg.get("uv_lr", 0.0)) > 0.0
        )
        if self.densification_enabled and not optimize_uv:
            raise ValueError(
                "Full UVD densification requires optimization.uv_lr > 0 so "
                "new points can be attached and repaired safely"
            )
        self.gaussian._uv.requires_grad_(optimize_uv)
        appearance_start = int(self.cfg.guidance_warmup_steps)
        geometry_start = max(int(cfg["geometry_start"]), appearance_start)
        groups = [
            {
                "params": [self.gaussian._features_dc],
                "name": "f_dc",
                "base_lr": float(cfg["feature_lr"]),
                "start": appearance_start,
            },
            {
                # The UVD densifier must extend every topology-shaped tensor.
                # SH-rest remains frozen, but keeping it in the optimizer at
                # lr=0 lets parameter/state replacement stay topology-safe.
                "params": [self.gaussian._features_rest],
                "name": "f_rest",
                "base_lr": 0.0,
                "start": appearance_start,
            },
            {
                "params": [self.gaussian._opacity],
                "name": "opacity",
                "base_lr": float(cfg["opacity_lr"]),
                "start": appearance_start,
            },
            {
                "params": [self.gaussian._d],
                "name": "d",
                "base_lr": float(cfg["d_lr"]),
                "start": geometry_start,
            },
            {
                "params": [self.gaussian._scaling],
                "name": "scaling",
                "base_lr": float(cfg["scale_lr"]),
                "start": geometry_start,
            },
            {
                "params": [self.gaussian._rotation],
                "name": "rotation",
                "base_lr": float(cfg["rotation_lr"]),
                "start": geometry_start,
            },
        ]
        if optimize_uv:
            groups.append(
                {
                    "params": [self.gaussian._uv],
                    "name": "uv",
                    "base_lr": float(cfg["uv_lr"]),
                    "start": geometry_start,
                }
            )
        self.gaussian.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1.0e-15)
        # Lightning forbids self.log() while configure_optimizers is running.
        self._update_learning_rates(log_values=False)
        return {"optimizer": self.gaussian.optimizer}

    def _update_learning_rates(self, log_values: bool = True) -> None:
        optimizer = getattr(self.gaussian, "optimizer", None)
        if optimizer is None:
            return
        step = int(self.true_global_step)
        maximum = int(self.cfg.optimization["max_steps"])
        final_ratio = float(self.cfg.optimization["final_lr_ratio"])
        for group in optimizer.param_groups:
            start = int(group["start"])
            if step < start:
                group["lr"] = 0.0
                continue
            progress = min(max((step - start) / max(maximum - start, 1), 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            multiplier = final_ratio + (1.0 - final_ratio) * cosine
            group["lr"] = float(group["base_lr"]) * multiplier
            if log_values:
                self.log(f"train/lr_{group['name']}", group["lr"])

    def on_before_optimizer_step(self, optimizer) -> None:
        max_grad_norm = float(
            self.cfg.optimization.get("max_grad_norm", 0.0)
        )
        # torch.nn.utils.clip_grad_norm_(..., 0) zeros every gradient.  Treat
        # non-positive values as an explicit switch that disables clipping.
        if max_grad_norm <= 0.0:
            return
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if parameters:
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)

    def _active_trainable_point_mask(self) -> torch.Tensor:
        if self.optimization_stage == "mouth":
            return self.mouth_trainable_point_mask
        if self._full_region_protection_enabled():
            trainable = torch.ones(
                self.gaussian.num_gs,
                dtype=torch.bool,
                device=self.gaussian.device,
            )
            protection = self.cfg.full_protection
            if bool(protection.get("freeze_eyes", True)):
                trainable &= ~self.eye_point_mask
            if bool(protection.get("freeze_mouth", False)):
                trainable &= ~self.mouth_guidance_point_mask
            if bool(protection.get("freeze_dental", True)):
                trainable &= ~self.dental_point_mask
            return trainable
        if (
            bool(self.cfg.freeze_dental_when_closed)
            and self.optimization_stage == "full"
            and not self._sdedit_active()
            and not self._active_open_mouth
        ):
            return ~self.dental_point_mask
        return torch.ones(
            self.gaussian.num_gs,
            dtype=torch.bool,
            device=self.gaussian.device,
        )

    @torch.no_grad()
    def _clear_frozen_optimizer_rows(
        self, trainable_mask: torch.Tensor
    ) -> None:
        """Prevent Adam momentum from moving rows whose gradient is masked."""

        frozen = ~trainable_mask.to(
            device=self.gaussian.device, dtype=torch.bool
        )
        if not bool(frozen.any().item()):
            return
        for group in self.gaussian.optimizer.param_groups:
            parameter = group["params"][0]
            state = self.gaussian.optimizer.state.get(parameter, {})
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = state.get(name)
                if (
                    torch.is_tensor(value)
                    and value.ndim > 0
                    and value.shape[0] == self.gaussian.num_gs
                ):
                    value[frozen] = 0

    def on_after_backward(self) -> None:
        # lr=0 must be a true freeze; otherwise Adam accumulates stale moments.
        for group in self.gaussian.optimizer.param_groups:
            if group["lr"] == 0.0:
                for parameter in group["params"]:
                    parameter.grad = None
        if any(
            parameter.grad is not None
            for group in self.gaussian.optimizer.param_groups
            for parameter in group["params"]
        ):
            # Mouth pass updates configured teeth points only.  The repaired
            # full pass treats completed eye/dental regions as immutable; old
            # configs retain the original closed-mouth-only dental freeze.
            trainable_mask = self._active_trainable_point_mask()
            self.gaussian.mask_out_gradient(
                trainable_mask, multiplier=0.0
            )
            self._clear_frozen_optimizer_rows(trainable_mask)
        if not any(
            parameter.grad is not None
            for group in self.gaussian.optimizer.param_groups
            for parameter in group["params"]
        ):
            # AMP requires one checked gradient even when the complete
            # Gaussian optimizer is intentionally frozen during VSD warmup.
            parameter = self.gaussian.optimizer.param_groups[0]["params"][0]
            parameter.grad = torch.zeros_like(parameter)
        self.gaussian._shape.grad = None

    @torch.no_grad()
    def _clear_parameter_optimizer_rows(
        self, parameter: torch.Tensor, rows: torch.Tensor
    ) -> None:
        optimizer = getattr(self.gaussian, "optimizer", None)
        if optimizer is None or not bool(rows.any().item()):
            return
        state = optimizer.state.get(parameter, {})
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(name)
            if (
                torch.is_tensor(value)
                and value.ndim > 0
                and value.shape[0] == self.gaussian.num_gs
            ):
                value[rows] = 0

    @torch.no_grad()
    def _project_dental_uv_to_bound_faces(self) -> int:
        if not self.gaussian._uv.requires_grad:
            return 0
        indices = torch.nonzero(
            self.dental_point_mask, as_tuple=False
        ).squeeze(1)
        if indices.numel() == 0:
            return 0
        uv = self.gaussian._uv[indices]
        face_idx = self.gaussian._face_idx[indices]
        inside, finite = self.gaussian._uv_inside_faces(
            uv, face_idx, 1.0e-6
        )
        repair = ~inside
        if not bool(repair.any().item()):
            return 0
        repair_indices = indices[repair]
        finite_repair = finite[repair]
        if bool(finite_repair.any().item()):
            selected_indices = repair_indices[finite_repair]
            selected_faces = self.gaussian._face_idx[selected_indices]
            triangles = self.gaussian.vt[
                self.gaussian.ft[selected_faces]
            ][:, None]
            projected, _ = self.gaussian._project_uv_to_triangles(
                self.gaussian._uv[selected_indices], triangles
            )
            self.gaussian._uv.data[selected_indices] = projected[:, 0].clamp(
                self.gaussian.UV_EPS, 1.0 - self.gaussian.UV_EPS
            )
        invalid_indices = repair_indices[~finite_repair]
        if invalid_indices.numel():
            self.gaussian._uv.data[invalid_indices] = self.initial["uv"][
                invalid_indices
            ]
        row_mask = torch.zeros_like(self.dental_point_mask)
        row_mask[repair_indices] = True
        self._clear_parameter_optimizer_rows(self.gaussian._uv, row_mask)
        return int(repair_indices.numel())

    def _full_region_protection_enabled(self) -> bool:
        return bool(
            self.optimization_stage == "full"
            and self.cfg.full_protection.get("enabled", False)
        )

    def _full_scale_stability_enabled(self) -> bool:
        return bool(
            self.optimization_stage == "full"
            and self.cfg.scale_stability.get("enabled", False)
        )

    @torch.no_grad()
    def _cap_current_full_world_scale(self) -> tuple[int, float]:
        """Shrink trainable rows exceeding the aligned world-space limit."""

        if not self._full_scale_stability_enabled():
            return 0, 0.0
        maximum = float(
            self.cfg.scale_stability.get(
                "hard_max_world_scale", self.cfg.max_world_scale
            )
        )
        _, scales, _ = self._aligned_render_geometry()
        world_scales = torch.nan_to_num(
            scales,
            nan=maximum * 2.0,
            posinf=maximum * 2.0,
            neginf=0.0,
        )
        world_scale = world_scales.amax(dim=-1)
        eligible = torch.ones_like(world_scale, dtype=torch.bool)
        if self._full_region_protection_enabled():
            eligible &= self._active_trainable_point_mask().to(
                device=world_scale.device, dtype=torch.bool
            )
        maximum_before = (
            float(world_scale[eligible].max().item())
            if bool(eligible.any().item())
            else 0.0
        )
        axis_ratio = self._full_world_scale_axis_ratio(scales)
        selected = eligible & (axis_ratio < 1.0).any(dim=-1)
        count = int(selected.sum().item())
        if count:
            indices = torch.nonzero(
                selected, as_tuple=False
            ).squeeze(1)
            updated = self.gaussian._scaling.data.index_select(0, indices)
            updated = updated + axis_ratio.index_select(0, indices).log()
            self.gaussian._scaling.data.index_copy_(0, indices, updated)
            self._clear_parameter_optimizer_rows(
                self.gaussian._scaling, selected
            )
        return count, maximum_before

    @torch.no_grad()
    def _stabilize_current_full_scale(self) -> dict[str, float | int]:
        """Apply the optional aligned world-space hard guard."""

        world_capped, world_before = self._cap_current_full_world_scale()
        return {
            "world_capped": world_capped,
            "world_before": world_before,
        }

    @torch.no_grad()
    def _cap_current_dental_world_scale(self) -> tuple[int, float]:
        maximum = float(
            self.cfg.geometry_stability.get(
                "dental_max_world_scale",
                self.cfg.mouth.get("max_scale", 0.01),
            )
        )
        world_scale = (
            self.gaussian.get_world_scale()[:, 0]
            * self._scene_similarity_scale()
        )
        ratio = (maximum / world_scale.clamp_min(1.0e-12)).clamp(max=1.0)
        selected = self.dental_point_mask & (ratio < 1.0)
        count = int(selected.sum().item())
        maximum_before = (
            float(world_scale[self.dental_point_mask].max().item())
            if bool(self.dental_point_mask.any().item())
            else 0.0
        )
        if count:
            indices = torch.nonzero(selected, as_tuple=False).squeeze(1)
            updated_scaling = self.gaussian._scaling.data.index_select(
                0, indices
            ) + ratio.index_select(0, indices)[:, None].log()
            self.gaussian._scaling.data.index_copy_(
                0, indices, updated_scaling
            )
            self._clear_parameter_optimizer_rows(
                self.gaussian._scaling, selected
            )
        return count, maximum_before

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if not self._optimizer_stepped_this_batch:
            return
        step = int(self.true_global_step)
        if (
            self.optimization_stage == "mouth"
            and step != self._last_geometry_projection_step
        ):
            projected_uv = self._project_dental_uv_to_bound_faces()
            interval = int(
                self.cfg.geometry_stability.get("project_interval", 0)
            )
            projected_scale, maximum_before = 0, 0.0
            if interval > 0 and step % interval == 0:
                projected_scale, maximum_before = (
                    self._cap_current_dental_world_scale()
                )
            envelope_interval = int(
                self.cfg.geometry_stability.get(
                    "envelope_project_interval", 0
                )
            )
            envelope_updates = 0
            envelope_maximum = 0.0
            if envelope_interval > 0 and step % envelope_interval == 0:
                named_poses, reference_pose = self._stability_pose_envelope()
                envelope_report = self._cap_region_over_pose_envelope(
                    self.dental_point_mask,
                    named_poses,
                    reference_pose,
                    maximum=float(
                        self.cfg.geometry_stability.get(
                            "dental_max_world_scale", 0.01
                        )
                    ),
                    passes=int(
                        self.cfg.geometry_stability.get("dental_passes", 3)
                    ),
                )
                envelope_updates = int(envelope_report["unique_updated"])
                envelope_maximum = float(envelope_report["after_max"])
            self.log("train/projected_uv", float(projected_uv))
            self.log("train/projected_scale", float(projected_scale))
            self.log("train/dental_world_scale_before_cap", maximum_before)
            self.log(
                "train/envelope_scale_caps", float(envelope_updates)
            )
            self.log(
                "train/envelope_dental_world_scale", envelope_maximum
            )
            self._last_geometry_projection_step = step
        elif (
            self.optimization_stage == "full"
            and step != self._last_geometry_projection_step
        ):
            repair = {"updated": 0, "projected": 0}
            if self.gaussian._uv.requires_grad:
                repair = self.gaussian.update_face_idx_from_uv(
                    mask=self._active_trainable_point_mask(),
                    return_stats=True
                )
            self.log("train/rebound_uv_faces", float(repair["updated"]))
            self.log("train/projected_uv", float(repair["projected"]))
            scale_report = self._stabilize_current_full_scale()
            self.log(
                "train/post_step_world_scale_caps",
                float(scale_report["world_capped"]),
            )
            self.log(
                "train/post_step_world_scale_max",
                float(scale_report["world_before"]),
            )
            self._last_geometry_projection_step = step
        dual_lr = float(self.cfg.reference_dual_lr)
        if dual_lr > 0.0:
            with torch.no_grad():
                self.reference_dual.add_(dual_lr * self._reference_violation).clamp_(min=float(self.cfg.reference_weight), max=float(self.cfg.reference_max_weight))
        # Manual accumulation marks only real optimizer updates. Triggering
        # here captures the state immediately after the final ISM update,
        # before the next batch can enter SDEdit.
        self._maybe_write_first_phase_artifacts()

    # ------------------------------------------------------------------
    # Exact mouth-topology / densified-full resume and compact exports.
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian_fields() -> dict[str, str]:
        return {
            "uv": "_uv",
            "d": "_d",
            "feature": "_features_dc",
            "feature_rest": "_features_rest",
            "opacity": "_opacity",
            "scale": "_scaling",
            "rotation": "_rotation",
            "face_idx": "_face_idx",
        }

    def _gaussian_state(self) -> dict[str, torch.Tensor]:
        return {
            key: getattr(self.gaussian, attribute).detach().cpu().clone()
            for key, attribute in self._gaussian_fields().items()
        }

    def _first_phase_artifacts_enabled(self) -> bool:
        return bool(
            self.cfg.first_phase_artifacts.get("enabled", False)
            and self.sdedit_start_step > 0
        )

    def _first_phase_artifact_name(self) -> str:
        guidance = "uvd_sfd" if self.uvd_flow_enabled else "ism"
        return f"{self.cfg.export_name}_{guidance}"

    @torch.no_grad()
    def _copy_phase_snapshot_to_gaussian(
        self, state: Mapping[str, torch.Tensor]
    ) -> None:
        fields = self._gaussian_fields()
        for key, value in state.items():
            if key not in fields:
                raise ValueError(
                    f"Unsupported first-phase Gaussian snapshot field {key!r}"
                )
            target = getattr(self.gaussian, fields[key])
            restored = torch.as_tensor(
                value, dtype=target.dtype, device=target.device
            )
            if restored.shape != target.shape:
                raise ValueError(
                    "First-phase Gaussian snapshot topology differs for "
                    f"field {key!r}: {tuple(restored.shape)} vs "
                    f"{tuple(target.shape)}"
                )
            target.data.copy_(restored)

    @torch.no_grad()
    def _render_first_phase_driving_test(
        self, prefix: str, fps: int
    ) -> None:
        """Render the configured test expression/pose sequence during fit."""

        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            raise RuntimeError(
                "First-phase driving test requires the active refinement "
                "datamodule"
            )
        if not hasattr(datamodule, "test_dataset"):
            datamodule.setup("test")
        loader = datamodule.test_dataloader()
        frame_count = 0
        try:
            for batch_idx, batch in enumerate(loader):
                batch = self.transfer_batch_to_device(
                    batch, self.gaussian.device, 0
                )
                self._save_evaluation_image(batch, batch_idx, prefix)
                frame_count += 1
            if frame_count <= 0:
                raise RuntimeError(
                    "First-phase driving test dataset contains no frames"
                )
            self.save_img_sequence(
                prefix,
                prefix,
                r"(\d+)\.png",
                save_format="mp4",
                fps=int(fps),
                name=prefix,
                step=int(self.sdedit_start_step),
            )
        finally:
            self._set_reference_pose()

    @torch.no_grad()
    def _maybe_write_first_phase_artifacts(self) -> None:
        """Persist the exact pre-SDEdit avatar once on every training path."""

        if (
            not self._first_phase_artifacts_enabled()
            or self._first_phase_artifacts_written
            or self._save_dir is None
        ):
            return
        optimizer_steps = int(
            self._optimizer_progress(self.gaussian.optimizer)
        )
        current_step = int(self.true_global_step)
        if (
            optimizer_steps < self.sdedit_start_step
            and current_step < self.sdedit_start_step
        ):
            return

        captured_now = self._sdedit_reference_state is None
        if captured_now:
            if current_step > self.sdedit_start_step:
                threestudio.warn(
                    "Recovering missing first-phase artifacts after the "
                    "SDEdit boundary from the current checkpoint state; an "
                    "older checkpoint did not preserve an exact boundary "
                    "snapshot."
                )
            self._capture_sdedit_reference()
        state = self._sdedit_reference_state
        assert state is not None
        boundary_steps = (
            optimizer_steps
            if optimizer_steps >= self.sdedit_start_step
            else self.sdedit_start_step
        )
        self._first_phase_optimizer_steps = int(boundary_steps)
        strategy = getattr(self.trainer, "strategy", None)
        if strategy is not None:
            strategy.barrier("first_phase_artifacts_start")

        if self.global_rank == 0:
            artifact_name = self._first_phase_artifact_name()
            prefix = f"{artifact_name}_driving_test"
            backup = None
            # A resumed SDEdit run may contain later trainable parameters even
            # though its frozen source is the exact phase-boundary state.
            if not captured_now:
                backup = {
                    key: getattr(
                        self.gaussian, self._gaussian_fields()[key]
                    ).detach().cpu().clone()
                    for key in state
                }
                self._copy_phase_snapshot_to_gaussian(state)
            try:
                if bool(
                    self.cfg.first_phase_artifacts.get(
                        "save_gaussian", True
                    )
                ):
                    self._export(
                        artifact_name,
                        refinement_phase="first_phase",
                        optimizer_executed_steps=boundary_steps,
                    )
                if bool(
                    self.cfg.first_phase_artifacts.get(
                        "render_driving_test", True
                    )
                ):
                    self._render_first_phase_driving_test(
                        prefix,
                        int(
                            self.cfg.first_phase_artifacts.get(
                                "driving_fps", 30
                            )
                        ),
                    )
            finally:
                if backup is not None:
                    self._copy_phase_snapshot_to_gaussian(backup)
                    self._set_reference_pose()

        if strategy is not None:
            strategy.barrier("first_phase_artifacts_end")
        self._first_phase_artifacts_written = True
        threestudio.info(
            "Saved first-phase Gaussian/driving-test artifacts as "
            f"{self._first_phase_artifact_name()} at optimizer step "
            f"{self._first_phase_optimizer_steps}."
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["stage2_schema_version"] = 9
        checkpoint["stage2_guidance_method"] = (
            self._guidance_method_signature()
        )
        checkpoint["stage2_sdedit_method"] = (
            self._sdedit_method_signature()
        )
        checkpoint["stage2_sdedit_updates_started"] = bool(
            int(self.true_global_step) > self.sdedit_start_step
        )
        checkpoint["stage2_full_scale_stability_version"] = 4
        checkpoint["stage2_full_region_protection_version"] = (
            2 if self._full_region_protection_enabled() else 0
        )
        checkpoint["stage2_gaussian"] = self._gaussian_state()
        checkpoint["stage2_completed_densification_steps"] = sorted(
            self._completed_densification_steps
        )
        checkpoint["stage2_sdedit_optimizer_state_reset"] = bool(
            self._sdedit_optimizer_state_reset
        )
        checkpoint["stage2_optimizer_executed_step_offset"] = int(
            self._optimizer_executed_step_offset
        )
        checkpoint["stage2_first_phase_artifacts_written"] = bool(
            self._first_phase_artifacts_written
        )
        checkpoint["stage2_first_phase_optimizer_steps"] = int(
            self._first_phase_optimizer_steps
        )
        if self._sdedit_reference_state is not None:
            checkpoint["stage2_sdedit_reference"] = {
                key: value.detach().cpu().clone()
                for key, value in self._sdedit_reference_state.items()
            }
        if self.uvd_flow_d_range is not None:
            checkpoint["stage2_uvd_flow_d_range"] = (
                self.uvd_flow_d_range.detach().cpu().clone()
            )
        if self.guidance is not None:
            state = self.guidance.vsd_checkpoint_state()
            if state is not None:
                checkpoint["stage2_vsd"] = state
            state = self.guidance.uvd_flow_checkpoint_state()
            if state is not None:
                checkpoint["stage2_uvd_ism_noise"] = state

    def on_load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        self._validate_sdedit_checkpoint_method(checkpoint)
        self._validate_guidance_checkpoint_method(checkpoint)
        if (
            self.uvd_flow_enabled
            and "stage2_uvd_surface_flow" in checkpoint
        ):
            raise ValueError(
                "This checkpoint contains an obsolete UVD-SFD state. Start "
                "the CFD-consistent score-difference objective from its verified "
                "PLY export instead of resuming the old trajectory."
            )
        saved_d_range = checkpoint.get("stage2_uvd_flow_d_range")
        if saved_d_range is not None:
            if self.uvd_flow_d_range is None:
                raise ValueError(
                    "Checkpoint contains UVD-SFD state but the current "
                    "configuration disables it"
                )
            saved_d_range = torch.as_tensor(
                saved_d_range,
                dtype=self.uvd_flow_d_range.dtype,
                device=self.uvd_flow_d_range.device,
            ).reshape(-1)
            if (
                saved_d_range.shape != self.uvd_flow_d_range.shape
                or not torch.allclose(
                    saved_d_range, self.uvd_flow_d_range, atol=1.0e-7, rtol=0.0
                )
            ):
                raise ValueError(
                    "UVD-SFD checkpoint canonical d range differs from the "
                    "current initialization/configuration"
                )
        if (
            self._full_region_protection_enabled()
            and int(
                checkpoint.get(
                    "stage2_full_region_protection_version", 0
                )
            )
            < 2
        ):
            raise ValueError(
                "This full-pass checkpoint predates protected face-binding "
                "and screen-space mouth preservation and may already contain "
                "a corrupted mouth. Start a fresh full run from the verified "
                "mouth PLY."
            )
        if (
            self._full_scale_stability_enabled()
            and int(
                checkpoint.get(
                    "stage2_full_scale_stability_version", 0
                )
            )
            < 4
        ):
            raise ValueError(
                "This full-pass checkpoint predates the absolute world-space "
                "anisotropy guard and may already contain needle-like scales. "
                "Start a fresh full run from the verified mouth PLY."
            )
        self._sdedit_optimizer_state_reset = bool(
            checkpoint.get("stage2_sdedit_optimizer_state_reset", False)
        )
        self._first_phase_artifacts_written = bool(
            checkpoint.get(
                "stage2_first_phase_artifacts_written", False
            )
        )
        self._first_phase_optimizer_steps = int(
            checkpoint.get("stage2_first_phase_optimizer_steps", -1)
        )
        saved_step_offset = checkpoint.get(
            "stage2_optimizer_executed_step_offset"
        )
        if saved_step_offset is None:
            # Schema-6 checkpoints created after the SDEdit reset have only
            # Adam's second-phase local counter.  The reset can only happen at
            # this configured boundary, and the exact first-phase count is
            # available in newer schema-6 checkpoints when artifacts were
            # enabled.  Recover it so resumed exports do not under-report.
            self._optimizer_executed_step_offset = (
                max(
                    int(self.sdedit_start_step),
                    int(self._first_phase_optimizer_steps),
                    0,
                )
                if self._sdedit_optimizer_state_reset
                else 0
            )
        else:
            self._optimizer_executed_step_offset = int(saved_step_offset)
            if self._optimizer_executed_step_offset < 0:
                raise ValueError(
                    "Checkpoint optimizer executed-step offset must be "
                    "non-negative"
                )
        self._completed_densification_steps = {
            int(value)
            for value in checkpoint.get(
                "stage2_completed_densification_steps", ()
            )
        }
        state = checkpoint.get("stage2_gaussian")
        if state is not None:
            topology_changed = False
            for key, attribute in self._gaussian_fields().items():
                current = getattr(self.gaussian, attribute)
                if key not in state:
                    raise ValueError(
                        "Stage-2 checkpoint predates the repaired mouth "
                        f"schema and has no {key!r} field. Start a fresh "
                        "mouth run from the Stage-1 reconstruction."
                    )
                restored = torch.as_tensor(state[key], dtype=current.dtype, device=current.device)
                if restored.shape != current.shape:
                    if self.optimization_stage != "full":
                        raise ValueError(
                            "Stage-2 checkpoint topology differs from the "
                            "configured reconstruction/mouth seed. Mouth "
                            "refinement must resume with its exact topology."
                        )
                    topology_changed = True
                    if key == "face_idx":
                        setattr(self.gaussian, attribute, restored.long())
                    else:
                        setattr(
                            self.gaussian,
                            attribute,
                            nn.Parameter(
                                restored,
                                requires_grad=(key != "feature_rest"),
                            ),
                        )
                else:
                    current.data.copy_(restored)
            if topology_changed:
                self.gaussian.num_gs = int(self.gaussian._uv.shape[0])
                self.gaussian._reset_densification_buffers()
            self._refresh_region_masks()
            self._reset_proximal_center_after_topology_change()
        sdedit_state = checkpoint.get("stage2_sdedit_reference")
        if sdedit_state is not None:
            expected = {
                "uv": self.gaussian._uv,
                "d": self.gaussian._d,
                "face_idx": self.gaussian._face_idx,
                "feature": self.gaussian._features_dc,
                "opacity": self.gaussian._opacity,
                "scale": self.gaussian._scaling,
                "rotation": self.gaussian._rotation,
            }
            restored_state = {}
            for key, current in expected.items():
                if key not in sdedit_state:
                    raise ValueError(
                        "SDEdit checkpoint predates the fixed-source UV "
                        f"snapshot and has no {key!r} field. Start a fresh "
                        "mouth run from Stage 1."
                    )
                restored = torch.as_tensor(
                    sdedit_state[key],
                    dtype=current.dtype,
                    device=current.device,
                )
                if restored.shape != current.shape:
                    raise ValueError(
                        "SDEdit checkpoint topology differs from the configured "
                        f"reconstruction for field {key}."
                    )
                restored_state[key] = restored.detach().clone()
            saved_scale_mask = sdedit_state.get("scale_trainable_mask")
            if saved_scale_mask is None:
                # Older checkpoints captured the SDEdit source only after the
                # final ISM topology was installed, so the refreshed live mask
                # is the topology-compatible fallback.
                saved_scale_mask = self._active_trainable_point_mask()
            restored_scale_mask = torch.as_tensor(
                saved_scale_mask,
                dtype=torch.bool,
                device=self.gaussian.device,
            ).reshape(-1)
            if restored_scale_mask.shape[0] != self.gaussian.num_gs:
                raise ValueError(
                    "SDEdit checkpoint scale-protection mask topology differs "
                    "from the restored Gaussian topology."
                )
            restored_state["scale_trainable_mask"] = (
                restored_scale_mask.detach().clone()
            )
            self._sdedit_reference_state = restored_state
        vsd_state = checkpoint.get("stage2_vsd")
        if vsd_state is not None:
            if self.guidance is None:
                self._pending_vsd_state = vsd_state
            else:
                self.guidance.load_vsd_checkpoint_state(vsd_state)
        uvd_flow_state = checkpoint.get("stage2_uvd_ism_noise")
        if uvd_flow_state is not None:
            if self.guidance is None:
                self._pending_uvd_flow_state = uvd_flow_state
            else:
                self.guidance.load_uvd_flow_checkpoint_state(uvd_flow_state)

    def _export(
        self,
        export_name: Optional[str] = None,
        *,
        refinement_phase: str = "final",
        optimizer_executed_steps: Optional[int] = None,
    ) -> None:
        if self._save_dir is None or self.global_rank != 0:
            return
        name = str(export_name or self.cfg.export_name).strip()
        if not name:
            raise ValueError("export_name must not be empty")
        path = Path(self.get_save_path(f"{name}.ply"))
        self._set_reference_pose()
        self.gaussian.save_ply(str(path))
        save_world_ply(
            self.gaussian,
            self.alignment,
            path.with_name(f"{path.stem}_world.ply"),
        )
        pose = np.zeros((1, 6), dtype=np.float32)
        pose[:, 3:6] = self.reference_jaw.detach().cpu().numpy()
        local_optimizer_steps = int(
            self._optimizer_progress(self.gaussian.optimizer)
        )
        optimizer_step_offset = int(
            self._optimizer_executed_step_offset
        )
        if optimizer_executed_steps is None:
            optimizer_executed_steps = int(
                self._cumulative_optimizer_progress(
                    self.gaussian.optimizer
                )
            )
        else:
            optimizer_executed_steps = int(optimizer_executed_steps)
            # An explicit count exports a logical phase-boundary snapshot,
            # which precedes any SDEdit optimizer reset even when recovered
            # later from a checkpoint.
            local_optimizer_steps = optimizer_executed_steps
            optimizer_step_offset = 0
        if optimizer_executed_steps < 0:
            raise ValueError("optimizer_executed_steps must be non-negative")
        np.savez(
            path.with_name(f"{path.stem}_params.npz"),
            shape=self.reference_shape.detach().cpu().numpy(),
            expression=self.reference_expression.detach().cpu().numpy(),
            pose=pose,
            jaw_pose=self.reference_jaw.detach().cpu().numpy(),
            eyes=torch.cat((self.reference_leye, self.reference_reye), dim=-1).detach().cpu().numpy(),
            facelift_from_training=self.alignment.detach().cpu().numpy(),
            flame_scale=np.float32(self.gaussian.flame_scale),
            spatial_lr_scale=np.float32(self.gaussian.spatial_lr_scale),
            scale_rotation_space=np.asarray("flame_face_local_v1"),
            representation_schema_version=np.int64(2),
            optimization_stage=np.asarray(self.optimization_stage),
            initialization_ply=np.asarray(str(self.initialization_ply)),
            optimizer_executed_steps=np.int64(optimizer_executed_steps),
            optimizer_local_executed_steps=np.int64(
                local_optimizer_steps
            ),
            optimizer_executed_step_offset=np.int64(
                optimizer_step_offset
            ),
            optimizer_step_accounting_version=np.int64(2),
            refinement_schema_version=np.int64(7),
            refinement_phase=np.asarray(str(refinement_phase)),
            phase_boundary_step=np.int64(self.sdedit_start_step),
            guidance_mode=np.asarray(
                "uvd-sfd" if self.uvd_flow_enabled else "ism"
            ),
            sdedit_mode=np.asarray(self.sdedit_mode),
            uvd_flow_d_range=(
                self.uvd_flow_d_range.detach().cpu().numpy()
                if self.uvd_flow_d_range is not None
                else np.asarray((np.nan, np.nan), dtype=np.float32)
            ),
        )

    def on_train_end(self) -> None:
        optimizer_executed_steps = int(
            self._cumulative_optimizer_progress(
                self.gaussian.optimizer
            )
        )
        if optimizer_executed_steps <= 0:
            raise RuntimeError(
                "Refinement reached on_train_end without one executed "
                "Gaussian optimizer step; refusing to export a no-op avatar."
            )
        if (
            self.optimization_stage == "mouth"
            and bool(self.cfg.geometry_stability.get("enabled", False))
        ):
            named_poses, reference_pose = self._stability_pose_envelope()
            report = self._cap_region_over_pose_envelope(
                self.dental_point_mask,
                named_poses,
                reference_pose,
                maximum=float(
                    self.cfg.geometry_stability.get(
                        "dental_max_world_scale", 0.01
                    )
                ),
                passes=int(
                    self.cfg.geometry_stability.get("dental_passes", 3)
                ),
            )
            threestudio.info(
                "Final mouth pose-envelope cap: "
                f"{report['before_max']:.6f} -> "
                f"{report['after_max']:.6f}; "
                f"updated={report['unique_updated']}."
            )
        elif self._full_scale_stability_enabled():
            self._set_reference_pose()
            report = self._stabilize_current_full_scale()
            threestudio.info(
                "Final full-pass scale cap in the reference pose: "
                f"world={report['world_capped']}, "
                f"world_max_before={report['world_before']:.6f}."
            )
        self._maybe_write_first_phase_artifacts()
        self._export()

    def _save_evaluation_image(
        self, batch: Mapping[str, Any], batch_idx: int, prefix: str
    ) -> None:
        with torch.no_grad():
            output = self.forward(batch)
        is_sequence = "sequence_index" in batch
        index = int(torch.as_tensor(batch.get("sequence_index", batch.get("index", batch_idx))).reshape(-1)[0])
        filename = f"{index:06d}" if is_sequence else str(index)
        self.save_image_grid(
            f"{prefix}/{filename}.png",
            [
                {
                    "type": "rgb",
                    "img": output["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                }
            ],
            name=prefix,
            step=self.true_global_step,
        )

    def validation_step(self, batch, batch_idx):
        self._save_evaluation_image(batch, batch_idx, f"validation-{self.true_global_step}")

    def on_validation_epoch_end(self) -> None:
        return None

    def test_step(self, batch, batch_idx):
        self._save_evaluation_image(batch, batch_idx, f"dynamic-test-{self.true_global_step}")

    def on_test_epoch_end(self) -> None:
        prefix = f"dynamic-test-{self.true_global_step}"
        self.save_img_sequence(
            prefix,
            prefix,
            r"(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        self._export()
