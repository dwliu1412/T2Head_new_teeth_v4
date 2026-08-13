import copy
import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from controlnet_aux import CannyDetector, NormalBaeDetector
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
)
from diffusers.utils.import_utils import is_xformers_available

import threestudio
from surface_inpaint.surface_memory_attention import (
    SurfaceMemoryController,
    install_surface_memory_attention,
)
from surface_inpaint.uvd_surface_flow import UVDNoiseVolume
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *
import matplotlib.cm as cm


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = float(alpha) / float(max(rank, 1))
        self.lora_scale = 0.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base_layer.out_features, bias=False)

        nn.init.normal_(self.lora_down.weight, std=1.0 / max(rank, 1))
        nn.init.zeros_(self.lora_up.weight)

        for p in self.base_layer.parameters():
            p.requires_grad_(False)

    def set_lora_scale(self, scale: float) -> None:
        self.lora_scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        if self.lora_scale == 0.0:
            return result
        lora_x = x.to(self.lora_down.weight.dtype)
        delta = self.lora_up(self.dropout(self.lora_down(lora_x)))
        delta = delta * self.scaling * self.lora_scale
        return result + delta.to(result.dtype)


@threestudio.register("controlnet-depth-guidance")
class ControlNetGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        pretrained_model_name_or_path: str = "stablediffusionapi/realistic-vision-51"
        # AnimPortrait3D loads sd-vae-ft-ema independently from its no-VAE
        # Realistic Vision checkpoint.  Leaving this unset preserves the
        # pipeline-embedded VAE used by older configurations.
        pretrained_vae_name_or_path: Optional[str] = None
        ddim_scheduler_name_or_path: str = "../HeadStudio_lib/stable-diffusion-v1-5"
        # GSAvatar uses an explicitly constructed DDIM scheduler for ISM but
        # lets its SDEdit pipeline retain the scheduler declared by the model
        # checkpoint (DEIS for Realistic Vision V5.1).
        edit_use_pipeline_scheduler: bool = False
        control_type: str = "normal"  # normal/canny
        # When set, this takes precedence over the built-in ControlNet aliases.
        # AnimPortrait3D's model has conditioning_channels=4 (normal RGB + seg).
        pretrained_controlnet_name_or_path: Optional[str] = None
        controlnet_subfolder: Optional[str] = None
        controlnet_conditioning_channels: Optional[int] = None

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        condition_scale: float = 1.5
        # PromptProcessorOutput contains (positive, negative, null) embeddings.
        # Keep "null" as the legacy behavior; strong-initialization fine-tuning
        # may opt into standard negative-prompt CFG with "negative".
        cfg_unconditional_source: str = "null"
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True
        # Deterministic VAE means reduce an otherwise unrelated source of
        # gradient variance during strong-initialization fine-tuning.
        vae_encode_mode: bool = False

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        diffusion_steps: int = 20

        use_nfsd: bool = False
        use_dsd: bool = False
        edit_image: bool = False

        # K-view Coupled SDS
        coupled_batch: bool = True
        coupled_share_t: bool = True
        coupled_share_noise: bool = True
        coupled_mean_grad: bool = False
        # Preserve the legacy from-scratch behavior by default. Stage-2 may
        # explicitly enable coupled timesteps/noise for its SDS ablation.
        coupled_apply_to_sds: bool = False

        # UVD-consistent ISM (kept behind the historical ``uvd-sfd`` CLI
        # label for experiment compatibility).  The caller supplies one
        # visible canonical (semantic layer, u, v, d) coordinate per pixel.
        # The ordinary AnimPortrait3D ISM objective is unchanged; only its
        # per-step Gaussian noise is coupled through canonical UVD cells.
        use_uvd_surface_flow: bool = False
        uvd_flow_noise_seed: int = 0
        uvd_flow_uv_resolution: int = 256
        uvd_flow_depth_resolution: int = 8
        uvd_flow_surface_layers: int = 5
        uvd_flow_min_distinct_cells: int = 1

        # ISM
        use_ism: bool = False
        ism_delta_t: int = 50
        # "interval" keeps the historical two-noise-level approximation.
        # "animportrait3d" reproduces the null-prompt DDIM inversion target
        # used by GSAvatar/helper.py::ism_step.
        ism_variant: str = "interval"
        # AnimPortrait3D uses one explicitly annealed all_t[iteration] rather
        # than drawing uniformly below the current maximum.
        ism_sample_at_max_step: bool = False

        # Online VSD. A small LoRA student is trained on current detached renders,
        # then guidance uses pretrained score minus LoRA score.
        use_vsd: bool = False
        vsd_start_step: int = 600
        vsd_warmup_sds_steps: int = 600
        vsd_lambda: float = 1.0
        vsd_lambda_ramp_steps: int = 600
        vsd_lora_train_start_step: int = 0
        vsd_lora_rank: int = 4
        vsd_lora_alpha: float = 4.0
        vsd_lora_dropout: float = 0.0
        vsd_lora_lr: float = 1.0e-4
        vsd_lora_weight_decay: float = 0.0
        vsd_lora_steps_per_iter: int = 1
        vsd_lora_grad_clip: float = 1.0
        vsd_lora_target_modules: Any = ("to_q", "to_k", "to_v", "to_out.0")
        vsd_lora_use_cfg: bool = False
        vsd_student_use_cfg: bool = False
        vsd_student_guidance_scale: float = 1.0
        vsd_debug: bool = True
        vsd_debug_interval: int = 200

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading ControlNet ...")

        explicit_controlnet = self.cfg.pretrained_controlnet_name_or_path
        controlnet_name_or_path: str
        if explicit_controlnet is not None and str(explicit_controlnet).strip():
            controlnet_name_or_path = str(explicit_controlnet)
        elif self.cfg.control_type == "normal":
            controlnet_name_or_path = "lllyasviel/control_v11p_sd15_normalbae"
        elif self.cfg.control_type == "canny":
            controlnet_name_or_path = "lllyasviel/control_v11p_sd15_canny"
        elif self.cfg.control_type == "depth":
            controlnet_name_or_path = "lllyasviel/control_v11f1p_sd15_depth"
        elif self.cfg.control_type == "openpose":
            controlnet_name_or_path = "lllyasviel/control_v11p_sd15_openpose"
        elif self.cfg.control_type == "mediapipe":
            # controlnet_name_or_path = "CrucibleAI/ControlNetMediaPipeFace"
            controlnet_name_or_path = "../HeadStudio_lib/ControlNetMediaPipeFace"
        else:
            raise ValueError(
                f"Unknown control_type {self.cfg.control_type!r}; provide "
                "pretrained_controlnet_name_or_path for a custom ControlNet"
            )

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
        }

        controlnet_kwargs = {
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
        }

        explicit_vae = self.cfg.pretrained_vae_name_or_path
        if explicit_vae is not None and str(explicit_vae).strip():
            pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(
                str(explicit_vae),
                torch_dtype=self.weights_dtype,
                cache_dir=self.cfg.cache_dir,
            )
            threestudio.info(
                f"Loaded explicit diffusion VAE {explicit_vae}."
            )
        if self.cfg.controlnet_subfolder is not None:
            controlnet_kwargs["subfolder"] = str(self.cfg.controlnet_subfolder)

        if explicit_controlnet is not None and str(explicit_controlnet).strip():
            controlnet = ControlNetModel.from_pretrained(
                controlnet_name_or_path,
                **controlnet_kwargs,
            )
        elif self.cfg.control_type == "mediapipe":
            if self.cfg.pretrained_model_name_or_path in ["stablediffusionapi/realistic-vision-51", "../HeadStudio_lib/realistic-vision-51",
                                                          "runwayml/stable-diffusion-v1-5", "../HeadStudio_lib/stable-diffusion-v1-5",
                                                          "../HeadStudio_lib/Realistic_Vision_V5.1_noVAE"]:
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_name_or_path,
                    subfolder="diffusion_sd15",
                    torch_dtype=self.weights_dtype,
                    cache_dir=self.cfg.cache_dir,
                    use_safetensors=False
                )
            else:
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_name_or_path,
                    torch_dtype=self.weights_dtype,
                    cache_dir=self.cfg.cache_dir,
                )

        else:
            controlnet = ControlNetModel.from_pretrained(
                controlnet_name_or_path,
                torch_dtype=self.weights_dtype,
                cache_dir=self.cfg.cache_dir,
            )
        self.conditioning_channels = int(
            getattr(controlnet.config, "conditioning_channels", 3)
        )
        expected_channels = self.cfg.controlnet_conditioning_channels
        if (
            expected_channels is not None
            and int(expected_channels) != self.conditioning_channels
        ):
            raise ValueError(
                f"ControlNet {controlnet_name_or_path!r} expects "
                f"{self.conditioning_channels} condition channels, but config "
                f"requires {int(expected_channels)}"
            )
        threestudio.info(
            f"Loaded ControlNet {controlnet_name_or_path} with "
            f"{self.conditioning_channels} conditioning channels."
        )
        # import pdb; pdb.set_trace()
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, controlnet=controlnet, **pipe_kwargs
        ).to(self.device)
        # import pdb; pdb.set_trace()
        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.ddim_scheduler_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        self.edit_scheduler = (
            copy.deepcopy(self.pipe.scheduler)
            if bool(self.cfg.edit_use_pipeline_scheduler)
            else self.scheduler
        )

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        self.controlnet = self.pipe.controlnet.eval()

        if self.cfg.control_type == "normal":
            self.preprocessor = NormalBaeDetector.from_pretrained(
                "lllyasviel/Annotators"
            )
            self.preprocessor.model.to(self.device)
        elif self.cfg.control_type == "canny":
            self.preprocessor = CannyDetector()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.controlnet.parameters():
            p.requires_grad_(False)

        # Installed only by the opt-in FLAME surface-consistent SDEdit branch.
        # Keeping this unset leaves every legacy U-Net processor untouched.
        self.surface_memory: Optional[SurfaceMemoryController] = None

        self.vsd_lora_layers = []
        self.vsd_lora_optimizer = None
        self.vsd_lora_num_layers = 0
        if self.cfg.use_vsd:
            self._setup_vsd_lora()

        enabled_distillation = {
            "UVD-consistent ISM": bool(self.cfg.use_uvd_surface_flow),
            "VSD": bool(self.cfg.use_vsd),
            "ISM": bool(self.cfg.use_ism),
            "NFSD/DSD": bool(self.cfg.use_nfsd or self.cfg.use_dsd),
        }
        selected = [name for name, enabled in enabled_distillation.items() if enabled]
        if len(selected) > 1:
            raise ValueError(
                "UVD-consistent ISM, VSD, ISM, and NFSD/DSD are mutually "
                "exclusive guidance "
                f"branches, got {selected}."
            )

        self.uvd_flow_noise: Optional[UVDNoiseVolume] = None
        if self.cfg.use_uvd_surface_flow:
            prediction_type = str(
                getattr(self.scheduler.config, "prediction_type", "epsilon")
            ).lower()
            if prediction_type != "epsilon":
                raise ValueError(
                    "UVD-consistent ISM requires an epsilon-prediction "
                    "diffusion teacher; "
                    f"scheduler prediction_type={prediction_type!r}"
                )
            self._setup_uvd_surface_flow()

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded ControlNet!")

    def configure_surface_memory_attention(
        self, config: Mapping[str, Any]
    ) -> SurfaceMemoryController:
        """Install the sparse canonical-surface K/V processor on self-attn."""

        if self.surface_memory is not None:
            self.surface_memory.uninstall()
        self.surface_memory = install_surface_memory_attention(
            self.unet, dict(config)
        )
        threestudio.info(
            "[FLAME surface SDEdit] Installed canonical K/V memory on "
            f"{len(self.surface_memory.wrapped_processor_names)} U-Net "
            "self-attention processors."
        )
        return self.surface_memory

    def _setup_uvd_surface_flow(self) -> None:
        if str(self.cfg.ism_variant).strip().lower() != "animportrait3d":
            raise ValueError(
                "UVD-consistent ISM requires "
                "ism_variant='animportrait3d' so only the noise coupling "
                "differs from the reference ISM ablation"
            )
        minimum_distinct_cells = int(self.cfg.uvd_flow_min_distinct_cells)
        if minimum_distinct_cells < 1:
            raise ValueError("uvd_flow_min_distinct_cells must be positive")
        if bool(self.cfg.coupled_mean_grad):
            raise ValueError(
                "UVD-consistent ISM cannot average gradients at matching "
                "screen pixels; "
                "cross-view coupling is defined only by canonical UVD cells"
            )
        self.uvd_flow_noise = UVDNoiseVolume(
            uv_resolution=int(self.cfg.uvd_flow_uv_resolution),
            depth_resolution=int(self.cfg.uvd_flow_depth_resolution),
            layer_count=int(self.cfg.uvd_flow_surface_layers),
            seed=int(self.cfg.uvd_flow_noise_seed),
            device=self.device,
        )
        threestudio.info(
            "[UVD-consistent ISM] Initialized canonical semantic UVD noise "
            "volume "
            f"L={int(self.cfg.uvd_flow_surface_layers)}, "
            f"D={int(self.cfg.uvd_flow_depth_resolution)}, "
            f"UV={int(self.cfg.uvd_flow_uv_resolution)}^2 "
            f"(seed={int(self.cfg.uvd_flow_noise_seed)}, "
            "fresh per optimizer step, shared within the step, "
            "variance-preserving reliability blend)."
        )

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @staticmethod
    def _get_submodule(root: nn.Module, path: str) -> nn.Module:
        module = root
        for name in path.split("."):
            module = getattr(module, name)
        return module

    @staticmethod
    def _target_matches(name: str, targets: Any) -> bool:
        if isinstance(targets, str):
            targets = [t.strip() for t in targets.split(",") if t.strip()]
        return any(name == target or name.endswith(f".{target}") for target in targets)

    def _setup_vsd_lora(self) -> None:
        rank = max(int(self.cfg.vsd_lora_rank), 1)
        targets = self.cfg.vsd_lora_target_modules
        replacements = []
        for name, module in self.unet.named_modules():
            if isinstance(module, nn.Linear) and self._target_matches(name, targets):
                replacements.append((name, module))

        for name, module in replacements:
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent = self._get_submodule(self.unet, parent_name)
            else:
                parent, child_name = self.unet, name
            lora_layer = LoRALinear(
                module,
                rank=rank,
                alpha=float(self.cfg.vsd_lora_alpha),
                dropout=float(self.cfg.vsd_lora_dropout),
            ).to(device=self.device)
            setattr(parent, child_name, lora_layer)
            self.vsd_lora_layers.append(lora_layer)

        self.vsd_lora_num_layers = len(self.vsd_lora_layers)
        if self.vsd_lora_num_layers == 0:
            threestudio.warn(
                "[VSD] No UNet Linear layers matched vsd_lora_target_modules; VSD will fall back to SDS."
            )
            return

        params = [p for layer in self.vsd_lora_layers for p in layer.parameters() if p.requires_grad]
        self.vsd_lora_optimizer = torch.optim.AdamW(
            params,
            lr=float(self.cfg.vsd_lora_lr),
            weight_decay=float(self.cfg.vsd_lora_weight_decay),
        )
        self._set_lora_scale(0.0)
        threestudio.info(
            f"[VSD] Installed online LoRA on {self.vsd_lora_num_layers} UNet attention linear layers "
            f"(rank={rank}, lr={float(self.cfg.vsd_lora_lr):.2e})."
        )

    def _set_lora_scale(self, scale: float) -> None:
        for layer in getattr(self, "vsd_lora_layers", []):
            layer.set_lora_scale(scale)

    def vsd_checkpoint_state(self) -> Optional[Dict[str, Any]]:
        """Return only trainable VSD state, without duplicating the frozen UNet."""

        layers = getattr(self, "vsd_lora_layers", [])
        optimizer = getattr(self, "vsd_lora_optimizer", None)
        if not layers or optimizer is None:
            return None
        return {
            "layers": [
                {
                    # ``Tensor.cpu()`` aliases storage when the tensor already
                    # lives on CPU.  Clone so a checkpoint snapshot cannot be
                    # mutated by subsequent online-student optimizer steps.
                    "lora_down": layer.lora_down.weight.detach().cpu().clone(),
                    "lora_up": layer.lora_up.weight.detach().cpu().clone(),
                }
                for layer in layers
            ],
            # Optimizer state_dict() is shallow with respect to moment tensors.
            # Deep-copy it for the same snapshot semantics as the LoRA weights.
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "rank": int(self.cfg.vsd_lora_rank),
            "targets": list(self.cfg.vsd_lora_target_modules),
        }

    def load_vsd_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        """Restore the online VSD student and AdamW moments for exact resume."""

        layers = getattr(self, "vsd_lora_layers", [])
        optimizer = getattr(self, "vsd_lora_optimizer", None)
        saved_layers = list(state.get("layers", []))
        if not layers or optimizer is None:
            raise RuntimeError(
                "Cannot restore VSD state because the configured guidance has "
                "no online LoRA optimizer."
            )
        if len(saved_layers) != len(layers):
            raise ValueError(
                "VSD checkpoint layer count does not match the current "
                f"configuration ({len(saved_layers)} vs {len(layers)})."
            )
        with torch.no_grad():
            for index, (layer, saved) in enumerate(zip(layers, saved_layers)):
                down = torch.as_tensor(
                    saved["lora_down"],
                    dtype=layer.lora_down.weight.dtype,
                    device=layer.lora_down.weight.device,
                )
                up = torch.as_tensor(
                    saved["lora_up"],
                    dtype=layer.lora_up.weight.dtype,
                    device=layer.lora_up.weight.device,
                )
                if down.shape != layer.lora_down.weight.shape:
                    raise ValueError(
                        f"VSD layer {index} down shape mismatch: "
                        f"{tuple(down.shape)} vs "
                        f"{tuple(layer.lora_down.weight.shape)}"
                    )
                if up.shape != layer.lora_up.weight.shape:
                    raise ValueError(
                        f"VSD layer {index} up shape mismatch: "
                        f"{tuple(up.shape)} vs "
                        f"{tuple(layer.lora_up.weight.shape)}"
                    )
                layer.lora_down.weight.copy_(down)
                layer.lora_up.weight.copy_(up)
        optimizer_state = state.get("optimizer")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        self._set_lora_scale(0.0)

    def uvd_flow_checkpoint_state(self) -> Optional[Dict[str, Any]]:
        """Return the per-step canonical UVD noise state for exact resume."""

        if not self.cfg.use_uvd_surface_flow or self.uvd_flow_noise is None:
            return None
        return self.uvd_flow_noise.state_dict()

    def load_uvd_flow_checkpoint_state(
        self, state: Mapping[str, Any]
    ) -> None:
        """Restore the UVD-ISM per-step volume and private RNGs."""

        if not self.cfg.use_uvd_surface_flow:
            raise RuntimeError(
                "Cannot restore UVD-consistent ISM state because "
                "use_uvd_surface_flow is disabled"
            )
        self.uvd_flow_noise = UVDNoiseVolume(
            uv_resolution=int(self.cfg.uvd_flow_uv_resolution),
            depth_resolution=int(self.cfg.uvd_flow_depth_resolution),
            layer_count=int(self.cfg.uvd_flow_surface_layers),
            seed=int(self.cfg.uvd_flow_noise_seed),
            device=self.device,
            state=state,
        )

    def _uvd_flow_noise_from_surface(
        self,
        surface_uvd: torch.Tensor,
        surface_layer: torch.Tensor,
        surface_confidence: torch.Tensor,
        latent_shape: tuple[int, int],
        step: int,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.uvd_flow_noise is None:
            raise RuntimeError(
                "UVD-consistent ISM noise state has not been initialized"
            )
        self.uvd_flow_noise.resample_for_step(step)
        return self.uvd_flow_noise.sample(
            surface_uvd,
            surface_layer,
            surface_confidence,
            latent_size=latent_shape,
            minimum_distinct_cells=int(
                self.cfg.uvd_flow_min_distinct_cells
            ),
        )

    def _vsd_active(self, step: int) -> bool:
        start_step = max(int(self.cfg.vsd_start_step), int(self.cfg.vsd_warmup_sds_steps))
        return bool(self.cfg.use_vsd and self.vsd_lora_optimizer is not None and step >= start_step)

    def _vsd_lora_train_active(self, step: int) -> bool:
        return bool(
            self.cfg.use_vsd
            and self.vsd_lora_optimizer is not None
            and step >= int(self.cfg.vsd_lora_train_start_step)
            and int(self.cfg.vsd_lora_steps_per_iter) > 0
        )

    def _current_vsd_lambda(self, step: int) -> float:
        start_step = max(int(self.cfg.vsd_start_step), int(self.cfg.vsd_warmup_sds_steps))
        if step < start_step:
            return 0.0
        ramp_steps = max(int(self.cfg.vsd_lambda_ramp_steps), 1)
        ramp = min(max((step - start_step) / ramp_steps, 0.0), 1.0)
        return float(self.cfg.vsd_lambda) * ramp

    @staticmethod
    def _safe_norm(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).norm()

    @staticmethod
    def _finite_fraction(x: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(x).float().mean()

    @torch.cuda.amp.autocast(enabled=False)
    def forward_controlnet(
            self,
            latents: Float[Tensor, "..."],
            t: Float[Tensor, "..."],
            image_cond: Float[Tensor, "..."],
            condition_scale: float,
            encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        return self.controlnet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            controlnet_cond=image_cond.to(self.weights_dtype),
            conditioning_scale=condition_scale,
            return_dict=False,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward_control_unet(
            self,
            latents: Float[Tensor, "..."],
            t: Float[Tensor, "..."],
            encoder_hidden_states: Float[Tensor, "..."],
            cross_attention_kwargs,
            down_block_additional_residuals,
            mid_block_additional_residual,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
            self,
            imgs: Float[Tensor, "B 3 512 512"],
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        if bool(self.cfg.vae_encode_mode):
            latents = posterior.mode()
        else:
            latents = posterior.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_cond_images(
            self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.mode()
        uncond_image_latents = torch.zeros_like(latents)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
            self,
            latents: Float[Tensor, "B 4 H W"],
            latent_height: int = 64,
            latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def edit_latents(
            self,
            text_embeddings: Float[Tensor, "BB 77 768"],
            latents: Float[Tensor, "B 4 64 64"],
            image_cond: Float[Tensor, "B 3 512 512"],
            t: Optional[Int[Tensor, "B"]] = None,
            strength: Optional[float] = None,
            guidance_scale: Optional[float] = None,
            use_control: bool = True,
            num_inference_steps: Optional[int] = None,
    ) -> tuple[Float[Tensor, "B 4 64 64"], Int[Tensor, "B"]]:
        """Run an img2img scheduler suffix from the requested noise strength.

        ``PromptProcessorOutput`` supplies three embedding batches ordered as
        (positive, negative, null).  Image editing uses ordinary two-branch
        classifier-free guidance, so the caller passes only the positive and
        negative batches here.  The old implementation denoised from the
        largest scheduler timestep even after adding noise at a smaller one;
        selecting the matching scheduler suffix is the essential SDEdit
        operation.
        """

        self._set_lora_scale(0.0)
        diffusion_steps = (
            int(self.cfg.diffusion_steps)
            if num_inference_steps is None
            else int(num_inference_steps)
        )
        if diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive for SDEdit")
        scheduler = self.edit_scheduler
        scheduler.set_timesteps(diffusion_steps)
        scheduler_timesteps = scheduler.timesteps.to(latents.device)
        if scheduler_timesteps.numel() == 0:
            raise ValueError("diffusion_steps must be positive for SDEdit")

        batch_size = latents.shape[0]
        if strength is not None:
            strength = float(strength)
            if not 0.0 < strength <= 1.0:
                raise ValueError("SDEdit strength must be in (0, 1]")
            # This is the img2img convention used by diffusers: strength is
            # the fraction of the inference trajectory retained for denoising.
            init_steps = min(max(int(diffusion_steps * strength), 1), diffusion_steps)
            start_index = max(diffusion_steps - init_steps, 0)
        else:
            if t is None:
                raise ValueError("SDEdit requires either strength or a timestep")
            requested = torch.as_tensor(t, device=latents.device, dtype=torch.long).reshape(-1)
            if requested.numel() == 1:
                requested = requested.repeat(batch_size)
            if requested.numel() != batch_size:
                raise ValueError(
                    "SDEdit timestep must be scalar or contain one value per batch item"
                )
            if not bool(torch.all(requested == requested[0]).item()):
                raise ValueError("A coupled SDEdit batch must share one timestep")
            start_index = int(
                torch.argmin(
                    (scheduler_timesteps.float() - requested[0].float()).abs()
                ).item()
            )

        denoise_timesteps = scheduler_timesteps[start_index:]
        if denoise_timesteps.numel() == 0:
            denoise_timesteps = scheduler_timesteps[-1:]
        noise_timestep = denoise_timesteps[:1].long().repeat(batch_size)
        cfg_scale = (
            float(self.cfg.guidance_scale)
            if guidance_scale is None
            else float(guidance_scale)
        )
        with torch.no_grad():
            noise = torch.randn_like(latents)
            if (
                bool(self.cfg.coupled_batch)
                and bool(self.cfg.coupled_share_noise)
                and batch_size > 1
            ):
                noise = noise[:1].repeat(batch_size, 1, 1, 1)
            latents = scheduler.add_noise(
                latents, noise, noise_timestep
            )  # type: ignore
            image_cond_input = torch.cat([image_cond] * 2)
            for denoise_index, timestep in enumerate(denoise_timesteps):
                if (
                    self.surface_memory is not None
                    and self.surface_memory.context_active
                ):
                    self.surface_memory.set_denoise_progress(
                        float(denoise_index + 1)
                        / float(denoise_timesteps.numel())
                    )
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(
                    latent_model_input, timestep
                )
                if use_control:
                    (
                        down_block_res_samples,
                        mid_block_res_sample,
                    ) = self.forward_controlnet(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=text_embeddings,
                        image_cond=image_cond_input,
                        condition_scale=self.cfg.condition_scale,
                    )
                else:
                    down_block_res_samples = None
                    mid_block_res_sample = None

                noise_pred = self.forward_control_unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (
                        noise_pred_text - noise_pred_uncond
                )
                latents = scheduler.step(
                    noise_pred, timestep, latents
                ).prev_sample
        return latents, noise_timestep

    def _apply_cfg(
        self,
        noise_pred_text: torch.Tensor,
        noise_pred_neg: torch.Tensor,
        noise_pred_null: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        source = str(self.cfg.cfg_unconditional_source).lower()
        if source == "null":
            noise_pred_uncond = noise_pred_null
        elif source == "negative":
            noise_pred_uncond = noise_pred_neg
        else:
            raise ValueError(
                "cfg_unconditional_source must be either 'null' or 'negative', "
                f"got {self.cfg.cfg_unconditional_source!r}"
            )
        return noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

    def _predict_noise(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents_noisy: Float[Tensor, "B 4 64 64"],
        image_cond: Float[Tensor, "B 3 512 512"],
        t: Int[Tensor, "B"],
        use_cfg: bool,
        guidance_scale: Optional[float] = None,
        lora_scale: float = 0.0,
        requires_grad: bool = False,
        use_control: bool = True,
    ) -> Float[Tensor, "B 4 64 64"]:
        batch_size = latents_noisy.shape[0]
        self._set_lora_scale(lora_scale)

        if use_cfg:
            latent_model_input = torch.cat([latents_noisy] * 3)
            image_cond_input = torch.cat([image_cond] * 3)
            t_input = torch.cat([t] * 3)
            encoder_hidden_states = text_embeddings
        else:
            latent_model_input = latents_noisy
            image_cond_input = image_cond
            t_input = t
            encoder_hidden_states = text_embeddings[:batch_size]

        if use_control:
            with torch.no_grad():
                (
                    down_block_res_samples,
                    mid_block_res_sample,
                ) = self.forward_controlnet(
                    latent_model_input,
                    t_input,
                    encoder_hidden_states=encoder_hidden_states,
                    image_cond=image_cond_input,
                    condition_scale=self.cfg.condition_scale,
                )
        else:
            down_block_res_samples = None
            mid_block_res_sample = None

        if requires_grad:
            noise_pred = self.forward_control_unet(
                latent_model_input,
                t_input,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )
        else:
            with torch.no_grad():
                noise_pred = self.forward_control_unet(
                    latent_model_input,
                    t_input,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )

        if not use_cfg:
            return noise_pred

        noise_pred_text, noise_pred_neg, noise_pred_null = noise_pred.chunk(3)
        scale = self.cfg.guidance_scale if guidance_scale is None else guidance_scale
        return self._apply_cfg(
            noise_pred_text, noise_pred_neg, noise_pred_null, float(scale)
        )

    def compute_grad_sds(
            self,
            text_embeddings: Float[Tensor, "BB 77 768"],
            latents: Float[Tensor, "B 4 64 64"],
            image_cond: Float[Tensor, "B 3 512 512"],
            t: Int[Tensor, "B"],
            noise=None
    ):
        self._set_lora_scale(0.0)
        with torch.no_grad():
            # add noise
            if noise is None:
                noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 3)
            image_cond_input = torch.cat([image_cond] * 3)
            down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                latent_model_input,
                torch.cat([t] * 3),
                encoder_hidden_states=text_embeddings,
                image_cond=image_cond_input,
                condition_scale=self.cfg.condition_scale,
            )

            noise_pred = self.forward_control_unet(
                latent_model_input,
                torch.cat([t] * 3),
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

        # perform classifier-free guidance
        noise_pred_text, noise_pred_neg, noise_pred_null = noise_pred.chunk(3)
        noise_pred = self._apply_cfg(
            noise_pred_text,
            noise_pred_neg,
            noise_pred_null,
            float(self.cfg.guidance_scale),
        )
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        return grad

    def train_vsd_lora(
            self,
            text_embeddings: Float[Tensor, "BB 77 768"],
            latents: Float[Tensor, "B 4 64 64"],
            image_cond: Float[Tensor, "B 3 512 512"],
            t: Int[Tensor, "B"],
            noise=None,
    ) -> Dict[str, torch.Tensor]:
        if self.vsd_lora_optimizer is None:
            return {
                "vsd_lora_loss": torch.zeros((), device=latents.device),
                "vsd_lora_grad_norm": torch.zeros((), device=latents.device),
            }

        latents_detached = latents.detach()
        image_cond_detached = image_cond.detach()
        text_embeddings_detached = text_embeddings.detach()
        batch_size = latents.shape[0]
        losses = []
        grad_norm = torch.zeros((), device=latents.device)

        for i in range(max(int(self.cfg.vsd_lora_steps_per_iter), 0)):
            if i == 0:
                train_t = t.detach()
                train_noise = noise.detach() if noise is not None else torch.randn_like(latents_detached)
            else:
                train_t = torch.randint(
                    self.min_step,
                    self.max_step + 1,
                    [batch_size],
                    dtype=torch.long,
                    device=self.device,
                )
                train_noise = torch.randn_like(latents_detached)

            latents_noisy = self.scheduler.add_noise(latents_detached, train_noise, train_t)
            self.vsd_lora_optimizer.zero_grad(set_to_none=True)
            self._set_lora_scale(1.0)

            with torch.enable_grad():
                noise_pred = self._predict_noise(
                    text_embeddings_detached,
                    latents_noisy,
                    image_cond_detached,
                    train_t,
                    use_cfg=bool(self.cfg.vsd_lora_use_cfg),
                    guidance_scale=float(self.cfg.vsd_student_guidance_scale),
                    lora_scale=1.0,
                    requires_grad=True,
                )
                loss = F.mse_loss(noise_pred.float(), train_noise.float(), reduction="mean")

            loss.backward()
            if self.cfg.vsd_lora_grad_clip and self.cfg.vsd_lora_grad_clip > 0:
                params = [
                    p
                    for layer in self.vsd_lora_layers
                    for p in layer.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                if params:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        params, max_norm=float(self.cfg.vsd_lora_grad_clip)
                    ).detach()
            self.vsd_lora_optimizer.step()
            losses.append(loss.detach())

        self.vsd_lora_optimizer.zero_grad(set_to_none=True)
        self._set_lora_scale(0.0)
        if not losses:
            mean_loss = torch.zeros((), device=latents.device)
        else:
            mean_loss = torch.stack(losses).mean()
        return {
            "vsd_lora_loss": mean_loss,
            "vsd_lora_grad_norm": grad_norm,
        }

    def compute_grad_vsd(
            self,
            text_embeddings: Float[Tensor, "BB 77 768"],
            latents: Float[Tensor, "B 4 64 64"],
            image_cond: Float[Tensor, "B 3 512 512"],
            t: Int[Tensor, "B"],
            step: int,
            noise=None,
    ):
        if noise is None:
            noise = torch.randn_like(latents)

        lora_metrics = self.train_vsd_lora(
            text_embeddings, latents, image_cond, t, noise=noise
        )
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        noise_pred_pretrained = self._predict_noise(
            text_embeddings,
            latents_noisy,
            image_cond,
            t,
            use_cfg=True,
            guidance_scale=float(self.cfg.guidance_scale),
            lora_scale=0.0,
            requires_grad=False,
        ).detach()
        noise_pred_lora = self._predict_noise(
            text_embeddings,
            latents_noisy,
            image_cond,
            t,
            use_cfg=bool(self.cfg.vsd_student_use_cfg),
            guidance_scale=float(self.cfg.vsd_student_guidance_scale),
            lora_scale=1.0,
            requires_grad=False,
        ).detach()
        self._set_lora_scale(0.0)

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad_sds = torch.nan_to_num(w * (noise_pred_pretrained - noise), nan=0.0, posinf=0.0, neginf=0.0)
        grad_vsd = torch.nan_to_num(w * (noise_pred_pretrained - noise_pred_lora), nan=0.0, posinf=0.0, neginf=0.0)

        lambda_vsd = self._current_vsd_lambda(step)
        grad = grad_sds + lambda_vsd * (grad_vsd - grad_sds)
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        metrics = {
            "vsd_enabled": torch.ones((), device=latents.device),
            "vsd_lambda": torch.as_tensor(lambda_vsd, device=latents.device),
            "vsd_lora_layers": torch.as_tensor(float(self.vsd_lora_num_layers), device=latents.device),
            "vsd_sds_grad_norm": self._safe_norm(grad_sds),
            "vsd_grad_norm": self._safe_norm(grad_vsd),
            "vsd_final_grad_norm": self._safe_norm(grad),
            "vsd_pretrained_eps_norm": self._safe_norm(noise_pred_pretrained),
            "vsd_lora_eps_norm": self._safe_norm(noise_pred_lora),
            "vsd_latents_finite": self._finite_fraction(latents),
            "vsd_final_grad_finite": self._finite_fraction(grad),
        }
        metrics.update(lora_metrics)
        return grad, metrics

    def compute_grad_nfsd(
            self,
            text_embeddings: Float[Tensor, "BB 77 768"],
            latents: Float[Tensor, "B 4 64 64"],
            image_cond: Float[Tensor, "B 3 512 512"],
            t: Int[Tensor, "B"],
            noise=None
    ):
        batch_size = latents.shape[0]
        self._set_lora_scale(0.0)
        with torch.no_grad():
            # add noise
            if noise is None:
                noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 3)
            image_cond_input = torch.cat([image_cond] * 3)
            down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                latent_model_input,
                torch.cat([t] * 3),
                encoder_hidden_states=text_embeddings,
                image_cond=image_cond_input,
                condition_scale=self.cfg.condition_scale,
            )

            noise_pred = self.forward_control_unet(
                latent_model_input,
                torch.cat([t] * 3),
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

        # perform classifier-free guidance
        noise_pred_text, noise_pred_neg, noise_pred_null = noise_pred.chunk(3)
        # Eq.6 in Noise-free Score Distillation, Katzir et al., arXiv preprint arXiv:2310.17590, 2023.
        delta_c = self.cfg.guidance_scale * (noise_pred_text - noise_pred_null)
        mask = (t < 200).int().view(batch_size, 1, 1, 1)
        if self.cfg.use_dsd:
            delta_d = mask * noise_pred_null + (1 - mask) * (noise_pred_null + (noise_pred_null - noise_pred_neg))
        else:
            delta_d = mask * noise_pred_null + (1 - mask) * (noise_pred_null - noise_pred_neg)

        # noise_pred = noise_pred_text + self.cfg.guidance_scale * (
        #     noise_pred_text - noise_pred_uncond
        # )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (delta_c + delta_d)
        return grad
    
    def _predict_ism_noise(
            self,
            latents: Float[Tensor, "B 4 64 64"],
            timesteps: Int[Tensor, "B"],
            embeddings: Float[Tensor, "B 77 768"],
            image_cond: Float[Tensor, "B C 512 512"],
            use_control: bool = True,
    ) -> Float[Tensor, "B 4 64 64"]:
        """Predict epsilon for one ISM branch, optionally without ControlNet.

        GSAvatar's full pass deliberately uses the base UNet for the global
        avatar loss and reserves AnimPortrait3D ControlNet for the zoomed
        face/eye/mouth losses.  Keeping this switch inside the ISM primitive
        makes that distinction exact instead of faking an empty condition.
        """

        latent_input = self.scheduler.scale_model_input(latents, timesteps)
        if use_control:
            down_samples, mid_sample = self.forward_controlnet(
                latent_input,
                timesteps,
                encoder_hidden_states=embeddings,
                image_cond=image_cond,
                condition_scale=self.cfg.condition_scale,
            )
        else:
            down_samples = None
            mid_sample = None
        return self.forward_control_unet(
            latent_input,
            timesteps,
            encoder_hidden_states=embeddings,
            cross_attention_kwargs=None,
            down_block_additional_residuals=down_samples,
            mid_block_additional_residual=mid_sample,
        )

    def _ddim_inverse_jump(
            self,
            sample: Float[Tensor, "B 4 64 64"],
            epsilon: Float[Tensor, "B 4 64 64"],
            current_t: Int[Tensor, "B"],
            next_t: Int[Tensor, "B"],
    ) -> Float[Tensor, "B 4 64 64"]:
        """Deterministically move x_current to the noisier x_next.

        This is the eta=0 DDIM equation used by AnimPortrait3D's custom
        ``ddim_step(..., delta_timestep=-delta)`` call.  It deliberately uses
        training timesteps directly instead of the scheduler inference grid.
        """

        prediction_type = str(
            getattr(self.scheduler.config, "prediction_type", "epsilon")
        )
        if prediction_type != "epsilon":
            raise ValueError(
                "AnimPortrait3D ISM requires an epsilon-prediction scheduler, "
                f"got {prediction_type!r}"
            )
        alpha_current = self.alphas[current_t].view(-1, 1, 1, 1)
        alpha_next = self.alphas[next_t].view(-1, 1, 1, 1)
        pred_x0 = (
            sample - (1.0 - alpha_current).sqrt() * epsilon
        ) / alpha_current.sqrt().clamp_min(1.0e-8)
        if bool(getattr(self.scheduler.config, "thresholding", False)):
            pred_x0 = self.scheduler._threshold_sample(pred_x0)
        elif bool(getattr(self.scheduler.config, "clip_sample", False)):
            clip_range = float(
                getattr(self.scheduler.config, "clip_sample_range", 1.0)
            )
            pred_x0 = pred_x0.clamp(-clip_range, clip_range)
        return (
            alpha_next.sqrt() * pred_x0
            + (1.0 - alpha_next).sqrt() * epsilon
        )

    def _compute_grad_animportrait3d_ism(
            self,
            text_embeddings: Float[Tensor, "BB 77 768"],
            latents: Float[Tensor, "B 4 64 64"],
            image_cond: Float[Tensor, "B C 512 512"],
            t: Int[Tensor, "B"],
            noise=None,
            step: int = -1,
            use_control: bool = True,
    ) -> Float[Tensor, "B 4 64 64"]:
        """GSAvatar's null-prompt inversion score-matching gradient."""

        batch_size = latents.shape[0]
        if text_embeddings.shape[0] != batch_size * 3:
            raise ValueError(
                "AnimPortrait3D ISM expects positive, negative and null text "
                f"embeddings (3B), got {text_embeddings.shape[0]} for B={batch_size}"
            )
        positive = text_embeddings[:batch_size]
        negative = text_embeddings[batch_size:2 * batch_size]
        null = text_embeddings[2 * batch_size:3 * batch_size]

        # helper.py anneals the inversion interval from 100 to 50 during the
        # first 400 optimizer steps, then keeps it at 50.
        effective_step = max(int(step), 0)
        warmup_rate = 1.0 - min(effective_step / 400.0, 1.0)
        current_delta = int(
            50 + math.ceil(warmup_rate * (100 - 50))
        )
        previous_t = torch.clamp(t - current_delta, min=0)
        inversion_t = torch.clamp(previous_t - 50 * 3, min=0)

        with torch.no_grad():
            if noise is None:
                noise = torch.randn_like(latents)
            inverted = self.scheduler.add_noise(latents, noise, inversion_t)

            # Three 50-timestep null-prompt inversion jumps produce x_s.
            current_t = inversion_t
            for _ in range(3):
                if bool(torch.all(current_t == previous_t).item()):
                    break
                target_epsilon = self._predict_ism_noise(
                    inverted, current_t, null, image_cond, use_control
                )
                next_t = torch.minimum(
                    current_t + torch.full_like(current_t, 50),
                    previous_t,
                )
                inverted = self._ddim_inverse_jump(
                    inverted, target_epsilon, current_t, next_t
                )
                current_t = next_t

            # The inversion score at s is the ISM target.  Use that same
            # score for the final deterministic s -> t jump, exactly as the
            # reference implementation does.
            target_epsilon = self._predict_ism_noise(
                inverted, previous_t, null, image_cond, use_control
            )
            latents_t = self._ddim_inverse_jump(
                inverted, target_epsilon, previous_t, t
            )

            latent_input = torch.cat([latents_t, latents_t], dim=0)
            timestep_input = torch.cat([t, t], dim=0)
            embedding_input = torch.cat([negative, positive], dim=0)
            condition_input = torch.cat([image_cond, image_cond], dim=0)
            guided_raw = self._predict_ism_noise(
                latent_input,
                timestep_input,
                embedding_input,
                condition_input,
                use_control,
            )
            noise_uncond, noise_text = guided_raw.chunk(2)
            noise_pred = noise_uncond + float(self.cfg.guidance_scale) * (
                noise_text - noise_uncond
            )

        alpha_prod_t = self.alphas[t].view(-1, 1, 1, 1)
        weight = ((1.0 - alpha_prod_t) / alpha_prod_t).sqrt()
        return weight * (noise_pred - target_epsilon)

    def compute_grad_ism(
            self,
            text_embeddings: Float[Tensor, "BB 77 768"],
            latents: Float[Tensor, "B 4 64 64"],
            image_cond: Float[Tensor, "B 3 512 512"],
            t: Int[Tensor, "B"],
            noise=None,
            step: int = -1,
            use_control: bool = True,
    ):
        """
        Implementation of Interval Score Matching (ISM) Loss based on the provided helper.py.
        """
        self._set_lora_scale(0.0)
        variant = str(self.cfg.ism_variant).strip().lower()
        if variant == "animportrait3d":
            return self._compute_grad_animportrait3d_ism(
                text_embeddings,
                latents,
                image_cond,
                t,
                noise=noise,
                step=step,
                use_control=use_control,
            )
        if variant != "interval":
            raise ValueError(
                "ism_variant must be 'interval' or 'animportrait3d', got "
                f"{self.cfg.ism_variant!r}"
            )
        delta_t = self.cfg.ism_delta_t
        s = t - delta_t
        s = torch.clamp(s, min=0)

        with torch.no_grad():
            # 1. Shared Noise Generation
            if noise is None:
                noise = torch.randn_like(latents)

            # 2. Get latents at step t and step s using the SAME noise
            # Note: scheduler.add_noise implements the forward process: z_t = alpha * z_0 + sigma * noise
            latents_t = self.scheduler.add_noise(latents, noise, t)
            latents_s = self.scheduler.add_noise(latents, noise, s)

            # 3. Predict epsilon at step t
            # Prepare inputs for CFG (text, neg, null) -> 3 chunks if using "perp-neg" style or standard 2 chunks?
            # Your SDS implementation uses 3 chunks (text, neg, null), assuming standard CFG here based on text_embeddings shape
            # Checking compute_grad_sds, it handles 3 chunks. We follow that pattern.

            latent_model_input_t = torch.cat([latents_t] * 3)
            image_cond_input = torch.cat([image_cond] * 3)
            t_input = torch.cat([t] * 3)

            # Forward ControlNet only for branches that request it.  The
            # global AnimPortrait3D full-avatar branch is intentionally plain
            # Stable Diffusion, matching GSAvatar/train_all.py.
            if use_control:
                down_block_res_samples_t, mid_block_res_sample_t = self.forward_controlnet(
                    latent_model_input_t,
                    t_input,
                    encoder_hidden_states=text_embeddings,
                    image_cond=image_cond_input,
                    condition_scale=self.cfg.condition_scale,
                )
            else:
                down_block_res_samples_t = None
                mid_block_res_sample_t = None

            # Forward UNet
            noise_pred_t_raw = self.forward_control_unet(
                latent_model_input_t,
                t_input,
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples_t,
                mid_block_additional_residual=mid_block_res_sample_t,
            )

            # CFG
            pred_text_t, pred_neg_t, pred_null_t = noise_pred_t_raw.chunk(3)
            noise_pred_t = self._apply_cfg(
                pred_text_t,
                pred_neg_t,
                pred_null_t,
                float(self.cfg.guidance_scale),
            )

            # 4. Predict epsilon at step s (Target)
            latent_model_input_s = torch.cat([latents_s] * 3)
            s_input = torch.cat([s] * 3)

            if use_control:
                down_block_res_samples_s, mid_block_res_sample_s = self.forward_controlnet(
                    latent_model_input_s,
                    s_input,
                    encoder_hidden_states=text_embeddings,
                    image_cond=image_cond_input,
                    condition_scale=self.cfg.condition_scale,
                )
            else:
                down_block_res_samples_s = None
                mid_block_res_sample_s = None

            noise_pred_s_raw = self.forward_control_unet(
                latent_model_input_s,
                s_input,
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples_s,
                mid_block_additional_residual=mid_block_res_sample_s,
            )

            pred_text_s, pred_neg_s, pred_null_s = noise_pred_s_raw.chunk(3)
            noise_pred_s = self._apply_cfg(
                pred_text_s,
                pred_neg_s,
                pred_null_s,
                float(self.cfg.guidance_scale),
            )

        # 5. Compute Gradient
        # Weighting from helper.py: w = sqrt((1 - alpha) / alpha)
        # This is 1/sqrt(SNR).
        # Note: self.alphas is typically alphas_cumprod in Diffusers schedulers
        alpha_prod_t = self.alphas[t].view(-1, 1, 1, 1)
        w = ((1 - alpha_prod_t) / alpha_prod_t) ** 0.5

        # Gradient direction: noise_pred_t - noise_pred_s(Target)
        grad = w * (noise_pred_t - noise_pred_s)

        return grad

    def save_t_grad(self, rgb, control_image, prompt_utils, t, elevation, azimuth, camera_distances):

        rgb_BCHW = rgb
        rgb_BCHW_512 = F.interpolate(
            rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
        )
        # encode image into latents with vae
        latents = self.encode_images(rgb_BCHW_512)
        image_cond = F.interpolate(
            control_image, (512, 512), mode="bilinear", align_corners=False
        )
        if image_cond.ndim != 4 or image_cond.shape[1] != self.conditioning_channels:
            raise ValueError(
                "ControlNet condition must be BCHW with "
                f"{self.conditioning_channels} channels, got "
                f"{tuple(image_cond.shape)}"
            )
        if not torch.isfinite(image_cond).all():
            raise FloatingPointError("ControlNet condition contains non-finite values")
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, True
        )
        if self.cfg.use_nfsd or self.cfg.use_dsd:
            grad = self.compute_grad_nfsd(text_embeddings, latents, image_cond, t)
        else:
            grad = self.compute_grad_sds(text_embeddings, latents, image_cond, t)
        grad = torch.nan_to_num(grad)

        # target = (latents - grad).detach()
        # diff = torch.abs(latents - target)
        # with torch.no_grad():
        #     grad_pixel = self.decode_latents(grad)

        heatmap = self.grad_to_heatmap(grad, upsample_size=(1024, 1024))
        # heatmap = self.grad_to_heatmap(grad_pixel, upsample_size=None)

        return heatmap

    def grad_to_heatmap(self, grad, upsample_size=None):
        """
        grad: (B, 4, 64, 64) 的梯度张量（对 latent 的 grad）
        upsample_size: 如果希望放大到原图大小，比如 (512,512)，就传这个；否则就用 64x64
        return: heatmap: (H, W, 3)，numpy，[0,1]
        """
        # 先取一张（第 0 个样本）
        g = grad[0]  # (4, 64, 64)
        #
        grad_norm = torch.norm(g, dim=0, keepdim=True)

        if upsample_size is not None:
            # 如果需要上采样
            g_processed = F.interpolate(
                grad_norm[None, ...],
                size=upsample_size,
                mode="bilinear",
                align_corners=False
            )[0, 0]  # 结果 (H, W)
        else:
            # 如果不需要上采样，直接去掉 channel 维
            g_processed = grad_norm[0]  # 结果 (64, 64)

        # 3. 归一化 (Min-Max Normalization)
        g_min = g_processed.min()
        g_max = g_processed.max()
        g_normalized = (g_processed - g_min) / (g_max - g_min + 1e-8)

        g_np = g_normalized.detach().cpu().numpy()  # (H, W)

        # 用 colormap 映射到 RGB
        heatmap = cm.jet(g_np)[..., :3]  # (H, W, 3)，[0,1]
        return heatmap

    def __call__(
            self,
            step,
            rgb: Float[Tensor, "B H W C"],
            control_image: Float[Tensor, "B H W C"],
            prompt_utils: PromptProcessorOutput,
            elevation: Float[Tensor, "B"],
            azimuth: Float[Tensor, "B"],
            camera_distances: Float[Tensor, "B"],
            rgb_as_latents=False,
            **kwargs,
    ):
        batch_size = rgb.shape[0]
        global_step = int(step) if step is not None else -1
        edit_image = bool(kwargs.get("edit_image", self.cfg.edit_image))
        vsd_active = self._vsd_active(global_step)
        uvd_flow_active = bool(
            self.cfg.use_uvd_surface_flow and not edit_image
        )
        coupling_active = bool(
            self.cfg.coupled_batch
            and (
                uvd_flow_active
                or vsd_active
                or self.cfg.coupled_apply_to_sds
            )
        )
        # assert batch_size == 1

        rgb_BCHW = rgb
        latents: Float[Tensor, "B 4 64 64"]
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            # Keep AnimPortrait3D's stochastic VAE encoding in both ISM
            # ablations.  UVD consistency changes only the diffusion noise,
            # so the comparison does not silently change a second variable.
            latents = self.encode_images(rgb_BCHW_512)

        # image_cond = control_image
        image_cond = F.interpolate(
            control_image, (512, 512), mode="bilinear", align_corners=False
        )

        view_dependent_prompting = bool(
            kwargs.get("view_dependent_prompting", True)
        )
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting,
        )
        view_metrics = {
            "view_dependent_prompting": torch.as_tensor(
                float(view_dependent_prompting), device=self.device
            )
        }
        if view_dependent_prompting:
            direction_indices = prompt_utils.get_view_direction_indices(
                elevation, azimuth, camera_distances
            )
            for direction in prompt_utils.directions:
                direction_index = prompt_utils.direction2idx[direction.name]
                view_metrics[f"view_{direction.name}_fraction"] = (
                    direction_indices == direction_index
                ).float().mean()
        # noise 共享
        shared_noise = None
        # if self.cfg.coupled_batch and self.cfg.coupled_share_noise and batch_size > 1:
        #     shared_noise = torch.randn_like(latents[:1]).repeat(batch_size, 1, 1, 1)
        forced_timestep = kwargs.get("guidance_timestep")
        if forced_timestep is not None:
            t = torch.as_tensor(
                forced_timestep,
                dtype=torch.long,
                device=self.device,
            ).reshape(-1)
            if t.numel() == 1:
                t = t.repeat(batch_size)
            if t.numel() != batch_size:
                raise ValueError(
                    "guidance_timestep must be scalar or contain one value "
                    f"per batch item ({batch_size}), got {t.numel()}"
                )
            lower = 0 if edit_image else self.min_step
            upper = self.num_train_timesteps - 1 if edit_image else self.max_step
            if bool(((t < lower) | (t > upper)).any().item()):
                raise ValueError(
                    "guidance_timestep is outside the active diffusion range "
                    f"[{lower}, {upper}]"
                )
        elif (
            (self.cfg.use_ism or uvd_flow_active)
            and self.cfg.ism_sample_at_max_step
        ):
            t = torch.full(
                (batch_size,),
                int(self.max_step),
                dtype=torch.long,
                device=self.device,
            )
        elif coupling_active and self.cfg.coupled_share_t and batch_size > 1:
            t0 = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
            t = t0.repeat(batch_size)
        else:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )
        if uvd_flow_active:
            if global_step < 0 or self.uvd_flow_noise is None:
                raise RuntimeError(
                    "UVD-consistent ISM requires a non-negative optimizer "
                    "step and an initialized UVD noise volume"
                )
            if not bool(torch.all(t == t[0]).item()):
                raise ValueError(
                    "Every view and region in a UVD-consistent ISM step must "
                    "share one timestep"
                )
        if edit_image:
            # Prompt embeddings are ordered (positive, negative, null).  The
            # SDEdit DDIM loop uses standard two-branch CFG.
            edit_embeddings = torch.cat(
                [
                    text_embeddings[:batch_size],
                    text_embeddings[batch_size:2 * batch_size],
                ],
                dim=0,
            )
            edit_strength = kwargs.get("edit_strength")
            surface_context = kwargs.get("surface_memory_context")
            controller = self.surface_memory
            surface_diagnostics_before: Optional[Mapping[str, Any]] = None
            if surface_context is not None:
                if not isinstance(surface_context, Mapping):
                    raise TypeError(
                        "surface_memory_context must be a mapping"
                    )
                if controller is None:
                    raise RuntimeError(
                        "FLAME surface-memory context was supplied before "
                        "installing its U-Net attention processors"
                    )
                surface_diagnostics_before = controller.diagnostics()
                controller.set_context(
                    surface_context["uv"],
                    surface_context["visibility"],
                    layer_ids=surface_context["layer_ids"],
                    depth=surface_context["depth"],
                    denoise_progress=0.0,
                    cfg_branches=2,
                    cfg_layout="chunked",
                )
            try:
                edit_latents, edit_timestep = self.edit_latents(
                    edit_embeddings,
                    latents,
                    image_cond,
                    None if edit_strength is not None else t,
                    strength=edit_strength,
                    guidance_scale=kwargs.get("edit_guidance_scale"),
                    use_control=bool(kwargs.get("edit_use_control", True)),
                    num_inference_steps=kwargs.get(
                        "edit_num_inference_steps"
                    ),
                )
            finally:
                if surface_context is not None and controller is not None:
                    controller.clear_context()
            edit_images = self.decode_latents(edit_latents)
            edit_images = F.interpolate(
                edit_images,
                (512, 512),
                mode="bilinear",
                align_corners=False,
            )
            edit_result = {
                "edit_images": edit_images.permute(0, 2, 3, 1),
                "sampled_timestep": edit_timestep.detach(),
                "timestep": edit_timestep.float().mean(),
            }
            if surface_context is not None and controller is not None:
                cumulative_counters = {
                    "contexts_set",
                    "denoise_progress_updates",
                    "self_attention_calls",
                    "surface_attention_calls",
                    "memory_queries",
                    "memory_slots",
                    "visible_surface_tokens",
                    "invalid_depth_tokens",
                }
                diagnostics_after = controller.diagnostics()
                diagnostics_before = surface_diagnostics_before or {}
                for name, value in diagnostics_after.items():
                    if isinstance(value, (bool, int, float)):
                        # A reconstruction step can invoke SDEdit for several
                        # semantic crops.  Export per-call counter deltas so a
                        # later call cannot appear active merely because an
                        # earlier call populated the cumulative controller.
                        if name in cumulative_counters:
                            value = float(value) - float(
                                diagnostics_before.get(name, 0)
                            )
                        edit_result[f"surface_memory_{name}"] = (
                            torch.as_tensor(
                                float(value), device=self.device
                            )
                        )
            edit_result.update(view_metrics)
            return edit_result

        vsd_metrics = {
            "vsd_enabled": torch.zeros((), device=self.device),
            "vsd_lambda": torch.zeros((), device=self.device),
            "vsd_lora_loss": torch.zeros((), device=self.device),
            "vsd_lora_grad_norm": torch.zeros((), device=self.device),
        }
        uvd_flow_metrics = {
            "uvd_ism_enabled": torch.zeros((), device=self.device),
            "uvd_ism_timestep": torch.zeros((), device=self.device),
            "uvd_flow_noise_mean": torch.zeros((), device=self.device),
            "uvd_flow_noise_std": torch.zeros((), device=self.device),
            "uvd_flow_surface_fraction": torch.zeros((), device=self.device),
            "uvd_flow_surface_confidence": torch.zeros((), device=self.device),
            "uvd_flow_distinct_cells": torch.zeros((), device=self.device),
            "uvd_flow_cell_coverage": torch.zeros((), device=self.device),
            "uvd_flow_transport_reliability": torch.zeros(
                (), device=self.device
            ),
            "uvd_ism_grad_norm": torch.zeros((), device=self.device),
        }
        if (
            not uvd_flow_active
            and coupling_active
            and self.cfg.coupled_share_noise
            and batch_size > 1
        ):
            shared_noise = torch.randn_like(latents[:1]).repeat(batch_size, 1, 1, 1)

        if uvd_flow_active:
            surface_uvd = kwargs.get("uvd_flow_surface_uvd")
            surface_layer = kwargs.get("uvd_flow_surface_layer")
            surface_confidence = kwargs.get("uvd_flow_surface_confidence")
            if (
                surface_uvd is None
                or surface_layer is None
                or surface_confidence is None
            ):
                raise ValueError(
                    "UVD-consistent ISM requires surface UVD, semantic "
                    "layer, and "
                    "correspondence confidence from the differentiable renderer"
                )
            uvd_noise, sampled_metrics = self._uvd_flow_noise_from_surface(
                surface_uvd,
                surface_layer,
                surface_confidence,
                latent_shape=latents.shape[-2:],
                step=global_step,
            )
            if uvd_noise.shape != latents.shape:
                raise ValueError(
                    "Canonical UVD noise shape must exactly match the ISM "
                    f"latents, got {tuple(uvd_noise.shape)} and "
                    f"{tuple(latents.shape)}"
                )
            if uvd_noise.device != latents.device:
                raise ValueError(
                    "Canonical UVD noise and ISM latents must be on the same "
                    f"device, got {uvd_noise.device} and {latents.device}"
                )
            if not uvd_noise.is_floating_point():
                raise TypeError("Canonical UVD noise must be floating point")
            uvd_noise = uvd_noise.to(dtype=latents.dtype)
            if not bool(torch.isfinite(uvd_noise).all().item()):
                raise ValueError("Canonical UVD noise contains non-finite values")
            uvd_flow_metrics.update(sampled_metrics)
            uvd_flow_metrics["uvd_ism_enabled"] = torch.ones(
                (), device=self.device
            )
            uvd_flow_metrics["uvd_ism_timestep"] = t.float().mean()
            # This is the exact same AnimPortrait3D null-prompt inversion
            # objective as the raw ISM branch below.  Canonically transported
            # noise is the only changed input.
            grad = self.compute_grad_ism(
                text_embeddings,
                latents,
                image_cond,
                t,
                noise=uvd_noise,
                step=global_step,
                use_control=bool(kwargs.get("use_control", True)),
            )
            uvd_flow_metrics["uvd_ism_grad_norm"] = self._safe_norm(grad)
        elif vsd_active:
            grad, vsd_metrics = self.compute_grad_vsd(
                text_embeddings,
                latents,
                image_cond,
                t,
                step=global_step,
                noise=shared_noise,
            )
        elif self.cfg.use_ism:
            if self._vsd_lora_train_active(global_step):
                vsd_metrics.update(
                    self.train_vsd_lora(
                        text_embeddings,
                        latents,
                        image_cond,
                        t,
                        noise=shared_noise,
                    )
                )
            grad = self.compute_grad_ism(
                text_embeddings,
                latents,
                image_cond,
                t,
                noise=shared_noise,
                step=global_step,
                use_control=bool(kwargs.get("use_control", True)),
            )
        elif self.cfg.use_nfsd or self.cfg.use_dsd:
            if self._vsd_lora_train_active(global_step):
                vsd_metrics.update(
                    self.train_vsd_lora(
                        text_embeddings,
                        latents,
                        image_cond,
                        t,
                        noise=shared_noise,
                    )
                )
            grad = self.compute_grad_nfsd(text_embeddings, latents, image_cond, t, noise=shared_noise)
        else:
            if self._vsd_lora_train_active(global_step):
                vsd_metrics.update(
                    self.train_vsd_lora(
                        text_embeddings,
                        latents,
                        image_cond,
                        t,
                        noise=shared_noise,
                    )
                )
            grad = self.compute_grad_sds(text_embeddings, latents, image_cond, t, noise=shared_noise)

        grad = torch.nan_to_num(grad)
        guidance_mask = kwargs.get("guidance_mask", None)
        guidance_mask_mean = torch.ones((), device=self.device)
        gradient_weight: Optional[torch.Tensor] = None
        if guidance_mask is not None:
            mask = torch.as_tensor(
                guidance_mask, dtype=grad.dtype, device=grad.device
            )
            if mask.ndim == 3:
                mask = mask[:, None]
            elif mask.ndim == 4 and mask.shape[-1] == 1:
                mask = mask.permute(0, 3, 1, 2)
            if mask.ndim != 4 or mask.shape[0] != batch_size:
                raise ValueError(
                    "guidance_mask must have shape Bx1xHxW, BxHxW, or BxHxWx1"
                )
            if mask.shape[1] != 1:
                mask = mask.mean(dim=1, keepdim=True)
            mask = F.interpolate(
                mask.detach(),
                size=grad.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            background_weight = float(
                kwargs.get("guidance_mask_background_weight", 0.0)
            )
            if not 0.0 <= background_weight <= 1.0:
                raise ValueError(
                    "guidance_mask_background_weight must be in [0, 1]"
                )
            gradient_weight = (
                background_weight + (1.0 - background_weight) * mask
            )
            guidance_mask_mean = mask.mean()
        if gradient_weight is not None:
            grad = grad * gradient_weight
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        if self.cfg.coupled_batch and self.cfg.coupled_mean_grad and batch_size > 1:
            g = grad.mean(dim=0, keepdim=True)
            grad = g.repeat(batch_size, 1, 1, 1)

        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        guidance_out = {
            "loss_sds": loss_sds,
            "loss_vsd": loss_sds if vsd_active else torch.zeros((), device=self.device),
            "loss_uvd_consistent_ism": loss_sds
            if uvd_flow_active
            else torch.zeros((), device=self.device),
            "grad_norm": grad.norm(),
            "guidance_mask_mean": guidance_mask_mean,
            # A full-refinement iteration calls the same guidance model for
            # full/face/eyes/mouth.  Returning the exact sampled tensor lets
            # the caller reuse one noise level across all semantic regions,
            # matching AnimPortrait3D's all_t[it] behavior.
            "sampled_timestep": t.detach(),
            "timestep": t.float().mean(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }
        guidance_out.update(vsd_metrics)
        guidance_out.update(uvd_flow_metrics)
        guidance_out.update(view_metrics)
        return guidance_out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(
                self.cfg.max_step_percent, epoch, global_step
            ),
        )


if __name__ == "__main__":
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.typing import Optional

    cfg = load_config("configs/experimental/controlnet-normal.yaml")
    guidance = threestudio.find(cfg.system.guidance_type)(cfg.system.guidance)
    prompt_processor = threestudio.find(cfg.system.prompt_processor_type)(
        cfg.system.prompt_processor
    )

    rgb_image = cv2.imread("assets/face.jpg")[:, :, ::-1].copy() / 255
    rgb_image = cv2.resize(rgb_image, (512, 512))
    rgb_image = torch.FloatTensor(rgb_image).unsqueeze(0).to(guidance.device)
    prompt_utils = prompt_processor()
    guidance_out = guidance(rgb_image, rgb_image, prompt_utils)
    edit_image = (
        (guidance_out["edit_images"][0].detach().cpu().clip(0, 1).numpy() * 255)
        .astype(np.uint8)[:, :, ::-1]
        .copy()
    )
    os.makedirs(".threestudio_cache", exist_ok=True)
    cv2.imwrite(".threestudio_cache/edit_image.jpg", edit_image)
