
from __future__ import annotations

import math
import os
import random
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm.auto import trange

os.environ.setdefault("THREESTUDIO_LAZY_IMPORT", "1")

from animation import get_c2w
from gaussiansplatting.arguments import OptimizationParams, PipelineParams
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene.cameras import Camera
from gaussiansplatting.scene.gaussian_flame_face import GaussianFlameUVModel
from threestudio.utils.animportrait3d_prompt import (
    view_direction_indices,
    view_prompt,
)
from threestudio.utils.head_v2 import FlamePointswRandomExp


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ANIMPORTRAIT3D_ROOT = Path("../../others/AnimPortrait3D")
DEFAULT_DIFFUSION_PATH = (DEFAULT_ANIMPORTRAIT3D_ROOT/ "pretrained_model" / "Realistic_Vision_V5.1_noVAE")
DEFAULT_CONTROLNET_PATH = (DEFAULT_ANIMPORTRAIT3D_ROOT / "pretrained_model" / "AnimPortrait3D_controlnet")

MOUTH_PROMPT_PREFIX = "mouth region, "
DEFAULT_NEGATIVE_PROMPT = ("tattoo, highlight, blur, no-shadow, lowres, bad anatomy, cropped, worst quality")
ANIMPORTRAIT3D_MOUTH_CROP_SIZE = 100


@dataclass(frozen=True)
class InitialGeometry:

    uv: torch.Tensor
    d: torch.Tensor
    point_count: int


@dataclass(frozen=True)
class MouthPromptEmbeddings:

    base: torch.Tensor
    directional: torch.Tensor
    negative: torch.Tensor
    null: torch.Tensor

    def get_text_embeddings(self, elevation: torch.Tensor, azimuth: torch.Tensor, view_dependent_prompting: bool = True) -> torch.Tensor:
        batch_size = elevation.shape[0]
        if view_dependent_prompting:
            positive = self.directional[view_direction_indices(azimuth)]
        else:
            positive = self.base[None].expand(batch_size, -1, -1)
        negative = self.negative[None].expand(batch_size, -1, -1)
        null = self.null[None].expand(batch_size, -1, -1)
        return torch.cat((positive, negative, null), dim=0)

def print_color(message: str, color: int = 36) -> None:
    print(f"\033[{color}m{message}\033[0m")

def resolve_path(path: Path | str) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def mouth_region_prompt(prompt: str) -> str:
    prompt = " ".join(prompt.strip().split())
    if not prompt:
        raise ValueError("--prompt must not be empty")
    if prompt.casefold().startswith(MOUTH_PROMPT_PREFIX.casefold()):
        return prompt
    return MOUTH_PROMPT_PREFIX + prompt


def build_parser() -> tuple[ArgumentParser, OptimizationParams, PipelineParams]:
    parser = ArgumentParser(description=__doc__)
    optimization_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)

    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/refinement/00000001/mouth_ism"))
    parser.add_argument( "--gaussian_train_iter", type=int, default=500)
    parser.add_argument( "--ism_begin_step", type=int, default=750)
    parser.add_argument("--ism_min_step", type=int, default=15)
    parser.add_argument("--points_cloud", type=Path, default=Path("outputs/reconstruction/00000001/model/uvd.ply"))
    parser.add_argument( "--sample_expr_path", type=Path, default=Path("assets/open_mouth_exp.npy"))
    parser.add_argument( "--sample_poses_path", type=Path, default=Path("assets/open_mouth_pose.npy"))

    parser.add_argument("--diffusion_path", type=Path, default=DEFAULT_DIFFUSION_PATH)
    parser.add_argument("--controlnet_path", type=Path, default=DEFAULT_CONTROLNET_PATH)
    parser.add_argument( "--vae_path", type=Path, default="../HeadStudio_lib/sd-vae-ft-mse")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--condition_scale", type=float, default=1.0)
    parser.add_argument("--grad_scale", type=float, default=1.0)

    parser.add_argument("--resolution", type=int, default=512)

    parser.add_argument("--gradient_accumulation", type=int, default=3)
    parser.add_argument("--lambda_xyz", type=float, default=0.0)
    parser.add_argument("--threshold_xyz", type=float, default=1.0)

    parser.add_argument("--lambda_scale", type=float, default=5.0e7)
    parser.add_argument("--threshold_scale", type=float, default=0.2)
    parser.add_argument("--densify_steps", type=int, default=[100])
    parser.add_argument("--max_gaussians", type=int, default=2_000_000)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--video_interval", type=int, default=50)
    parser.add_argument("--video_frames", type=int, default=240)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
   

    parser.set_defaults(
        position_lr_init=1.0e-5,
        position_lr_final=1.0e-5,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=600_000,
        feature_lr=0.0025 * 1.5,
        opacity_lr=0.05 * 1.5,
        scaling_lr=0.017 * 1.5,
        rotation_lr=0.001 * 1.5,
        shape_lr=0.0,
        compute_cov3D_python=True,
    )
    return parser, optimization_params, pipeline_params

def build_guidance(args: Namespace):
    from threestudio.models.guidance.controlnet_guidance import ControlNetGuidance

    config = {
        "pretrained_model_name_or_path": str(args.diffusion_path),
        "pretrained_vae_name_or_path": str(args.vae_path),
        "ddim_scheduler_name_or_path": str(args.diffusion_path),
        "control_type": "animportrait3d",
        "pretrained_controlnet_name_or_path": str(args.controlnet_path),
        "controlnet_conditioning_channels": 4,
        "guidance_scale": float(args.guidance_scale),
        "condition_scale": float(args.condition_scale),
        "cfg_unconditional_source": "negative",
        "half_precision_weights": True,
        "vae_encode_mode": False,
        "min_step_percent": float(args.ism_min_step) / 1000.0,
        "max_step_percent": float(args.ism_begin_step) / 1000.0,
        "diffusion_steps": 32,
        "coupled_batch": False,
        "coupled_share_t": False,
        "coupled_share_noise": False,
        "coupled_mean_grad": False,
        "use_ism": True,
        "ism_variant": "animportrait3d",
        "ism_delta_t": 50,
        "ism_sample_at_max_step": False,
    }
    return ControlNetGuidance(config)


@torch.no_grad()
def encode_prompts(guidance: ControlNetGuidance, prompts: list[str]) -> torch.Tensor:
    tokenizer = guidance.pipe.tokenizer
    text_encoder = guidance.pipe.text_encoder
    tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    attention_mask = None
    if bool(getattr(text_encoder.config, "use_attention_mask", False)):
        attention_mask = tokens.attention_mask.to(guidance.device)
    return text_encoder(tokens.input_ids.to(guidance.device), attention_mask=attention_mask)[0]


def build_prompt_utils(guidance, prompt, negative_prompt) -> MouthPromptEmbeddings:
    direction_names = ("side", "front", "back")
    encoded = encode_prompts(
        guidance,
        [prompt]
        + [view_prompt(name, prompt) for name in direction_names]
        + [negative_prompt, ""],
    )
    base_embedding = encoded[0]
    directional_embeddings = encoded[1:4]
    negative_embedding = encoded[4]
    null_embedding = encoded[5]

    return MouthPromptEmbeddings(
        base=base_embedding,
        directional=directional_embeddings,
        negative=negative_embedding,
        null=null_embedding,
    )


def set_pose(gaussian, expression, jaw_pose, leye_pose=None, reye_pose=None, neck_pose=None):
    gaussian._expression = expression.detach()
    gaussian._jaw_pose = jaw_pose.detach()
    if leye_pose is not None: gaussian._leye_pose = leye_pose.detach()
    if reye_pose is not None: gaussian._reye_pose = reye_pose.detach()
    if neck_pose is not None: gaussian._neck_pose = neck_pose.detach()

def get_camera(dist, elev, azim, fovy_deg=70.0):
    c2w = get_c2w(dist=dist, elev=elev, azim=azim)
    fovy_deg = torch.full_like(elev, fovy_deg)
    fovy = fovy_deg * math.pi / 180
    height, width = 512, 512
    viewpoint_cam = Camera(c2w=c2w[0], FoVy=fovy[0], height=height, width=width)
    return viewpoint_cam

def sample_camera(device):
    azimuth = torch.rand((1,)).to(device) * 100. + 40.
    elevation = torch.rand((1,)).to(device) * 40. - 10.
    distance = torch.full_like(azimuth, 2.0).to(device)
    camera = get_camera(distance, elevation, azimuth)

    return camera, distance, elevation, azimuth


@torch.no_grad()
def mouth_crop_box(
    skel: FlamePointswRandomExp,
    gaussian: GaussianFlameUVModel,
    distance: torch.Tensor,
    elevation: torch.Tensor,
    azimuth: torch.Tensor,
    fovy_degrees: float,
    resolution: int,
) -> tuple[int, int, int, int]:

    result = skel._flame_forward(
        betas=gaussian.get_shape.detach(),
        expression=gaussian.get_expression,
        jaw_pose=gaussian.get_jaw_pose,
        leye_pose=gaussian.get_leye_pose,
        reye_pose=gaussian.get_reye_pose,
        neck_pose=gaussian.get_neck_pose,
        global_orient=gaussian.get_global_orient,
        translation=gaussian.get_translation,
        return_landmarks=True,
    )

    landmarks = result[-1][0]
    landmarks = (landmarks - skel.center) * skel.scale
    landmarks = landmarks * (1.1 ** (-skel.flame_scale))

    at = torch.zeros((1, 3), dtype=torch.float32, device=skel.device)
    up = torch.tensor(((0.0, 0.0, 1.0),), device=skel.device)
    cameras = skel.get_camera(
        distance,
        elevation,
        90.0 - azimuth,
        skel.camera_conversion(at),
        skel.camera_conversion(up),
        torch.full_like(elevation, 70.),
    )
    projected = skel._project_landmarks_to_image(landmarks, cameras, gaussian_camera_convention=True)[0]
    center = projected[[48, 54]].mean(dim=0)

    side = max(int(round(resolution * ANIMPORTRAIT3D_MOUTH_CROP_SIZE / 512)), 1)
    side = min(side, resolution)
    x0 = int(math.floor(float(center[0].item()) - side / 2.0))
    y0 = int(math.floor(float(center[1].item()) - side / 2.0))
    x0 = min(max(x0, 0), resolution - side)
    y0 = min(max(y0, 0), resolution - side)
    return x0, y0, x0 + side, y0 + side


@torch.no_grad()
def render_control_condition(
    skel: FlamePointswRandomExp,
    gaussian: GaussianFlameUVModel,
    distance: torch.Tensor,
    elevation: torch.Tensor,
    azimuth: torch.Tensor,
    args: Namespace,
) -> torch.Tensor:
    at = torch.zeros((1, 3), dtype=torch.float32, device=skel.device)
    up = torch.tensor(((0.0, 0.0, 1.0),), dtype=torch.float32, device=skel.device)
    condition, _ = skel.get_cond_normal_semantic(
        dist=distance,
        elev=elevation,
        azim=azimuth,
        at=at,
        up=up,
        fov=torch.full_like(elevation, 70.),
        betas=gaussian.get_shape.detach(),
        expression=gaussian.get_expression,
        jaw_pose=gaussian.get_jaw_pose,
        leye_pose=gaussian.get_leye_pose,
        reye_pose=gaussian.get_reye_pose,
        neck_pose=gaussian.get_neck_pose,
        gaussian_camera_convention=True,
    )

    return condition.permute(0, 3, 1, 2).contiguous().float()


def render_training_sample(
    gaussian: GaussianFlameUVModel,
    skel: FlamePointswRandomExp,
    pipeline: Any,
    background: torch.Tensor,
    args: Namespace,
) -> dict[str, Any]:
    camera, distance, elevation, azimuth = sample_camera(gaussian.device)
    package = render(camera, gaussian, pipeline, background)
    rendered = package["render"][:3]
    condition = render_control_condition(skel, gaussian, distance, elevation, azimuth, args)
    box = mouth_crop_box(skel, gaussian, distance, elevation, azimuth, 70., int(args.resolution))
    x0, y0, x1, y1 = box
    rgb_crop = rendered[:, y0:y1, x0:x1][None]
    condition_crop = condition[:, :, y0:y1, x0:x1]

    package.update(
        {
            "rgb_crop": rgb_crop,
            "condition_crop": condition_crop,
            "crop_box": box,
            "distance": distance,
            "elevation": elevation,
            "azimuth": azimuth,
        }
    )
    return package

@torch.no_grad()
def make_ism_visualization(
    guidance: ControlNetGuidance,
    rgb_512: torch.Tensor,
    condition_512: torch.Tensor,
    target: torch.Tensor,
) -> Image.Image:
    decoded_target = guidance.decode_latents(torch.nan_to_num(target.float(), nan=0.0, posinf=0.0, neginf=0.0))
    panels = [
        rgb_512.detach().clamp(0.0, 1.0),
        condition_512[:, :3].detach().clamp(0.0, 1.0),
        condition_512[:, 3:4].detach().repeat(1, 3, 1, 1).clamp(0.0, 1.0),
        decoded_target.detach().clamp(0.0, 1.0),
    ]
    canvas = torch.cat(panels, dim=-1)[0]
    array = ((canvas * 255.0).round().byte().permute(1, 2, 0).contiguous().cpu().numpy())
    return Image.fromarray(array)

def ism_step(
    guidance: ControlNetGuidance,
    prompt_utils: MouthPromptEmbeddings,
    sample: dict[str, Any],
    timestep: torch.Tensor,
    iteration: int,
    grad_scale: float,
    return_vis: bool,
) -> dict[str, Any]:
    rgb_512 = F.interpolate(
        sample["rgb_crop"],
        size=(512, 512),
        mode="bilinear",
        align_corners=False,
    )
    condition_512 = F.interpolate(
        sample["condition_crop"],
        size=(512, 512),
        mode="bilinear",
        align_corners=False,
    )
    latents = guidance.encode_images(rgb_512)
    text_embeddings = prompt_utils.get_text_embeddings(
        sample["elevation"],
        sample["azimuth"],
        view_dependent_prompting=True,
    )
    t = timestep.to(device=latents.device, dtype=torch.long).reshape(1)
    grad = guidance.compute_grad_ism(
        text_embeddings,
        latents,
        condition_512,
        t,
        step=iteration,
        use_control=True,
    )
    grad = torch.nan_to_num(grad * float(grad_scale))
    target = (latents - grad).detach()
    loss = (0.5 * F.mse_loss(latents.float(), target.float(), reduction="sum") / latents.shape[0])
    result: dict[str, Any] = {
        "losses": {"ism": loss},
        "grad_norm": grad.detach().float().norm(),
        "timestep": t.detach(),
        "view_direction": int(
            view_direction_indices(sample["azimuth"])[0].item()
        ),
    }
    if return_vis:
        result["vis_img"] = make_ism_visualization(guidance, rgb_512, condition_512, target)
    return result


def gaussian_regularization(
    gaussian: GaussianFlameUVModel,
    initial: InitialGeometry,
    trainable_mask: torch.Tensor,
    visibility_filter: torch.Tensor,
    args: Namespace,
) -> dict[str, torch.Tensor]:

    zero = gaussian._uv.sum() * 0.0
    active = trainable_mask.bool() & visibility_filter.bool()
    losses: dict[str, torch.Tensor] = {}

    if args.lambda_xyz != 0.0:
        if bool(active.any().item()):
            current = torch.cat((gaussian._uv[active], gaussian._d[active]), dim=1)
            reference = torch.cat((initial.uv[active], initial.d[active]), dim=1)
            radial_change = (current.norm(dim=1) - reference.norm(dim=1)).abs()
            losses["xyz"] = (F.relu(radial_change - float(args.threshold_xyz)).mean() * float(args.lambda_xyz))
        else:
            losses["xyz"] = zero

    if args.lambda_scale != 0.0:
        if bool(active.any().item()):
            overflow = F.relu(gaussian.get_scaling[active] - float(args.threshold_scale))
            losses["scale"] = (overflow.norm(dim=1).sum() * float(args.lambda_scale))
        else:
            losses["scale"] = zero
    return losses


def assert_finite_gradients(gaussian: GaussianFlameUVModel) -> None:
    for group in gaussian.optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None and not bool(
                torch.isfinite(parameter.grad).all().item()
            ):
                raise FloatingPointError(
                    f"Non-finite gradient in optimizer group {group['name']!r}"
                )

@torch.no_grad()
def record_densification_statistics(gaussian: GaussianFlameUVModel, sample: dict[str, Any]) -> None:
    visibility = sample["visibility_filter"].bool()
    radii = sample["radii"]
    gaussian.max_radii2D[visibility] = torch.maximum(gaussian.max_radii2D[visibility], radii[visibility])
    viewspace_gradient = sample["viewspace_points"].grad
    if viewspace_gradient is not None:
        gaussian.add_densification_stats(viewspace_gradient.detach(), visibility)


@torch.no_grad()
def densify_teeth(
    gaussian: GaussianFlameUVModel,
    trainable_mask: torch.Tensor,
    initial: InitialGeometry,
    optimization: Any,
    args: Namespace,
    iteration: int,
) -> tuple[torch.Tensor, InitialGeometry]:
    if gaussian.num_gs >= int(args.max_gaussians):
        print_color(
            f"Skipping densification at {iteration}: point budget reached", 33
        )
        return trainable_mask, initial

    gradients = gaussian._densification_gradients().squeeze(-1)
    world_scale = gaussian.get_world_scale_max_approx()
    observed = gaussian.denom.squeeze(-1) > 0
    clone_mask = (
        (gradients >= float(optimization.densify_grad_threshold))
        & (world_scale <= gaussian.percent_dense * 4.0)
        & observed
        & trainable_mask
    )
    split_mask = (
        (gradients >= float(optimization.densify_grad_threshold))
        & (world_scale > gaussian.percent_dense * 4.0)
        & observed
        & trainable_mask
    )
    budget = max(int(args.max_gaussians) - gaussian.num_gs, 0)
    clone_mask = gaussian._limit_to_budget(clone_mask, gradients, budget)
    clone_count = int(clone_mask.sum().item())
    split_mask = gaussian._limit_to_budget(
        split_mask, gradients, budget - clone_count
    )
    split_count = int(split_mask.sum().item())

    if initial.point_count != gaussian.num_gs:
        raise RuntimeError("Cannot densify with stale geometry-reference rows")
    kept_reference_uv = initial.uv[~split_mask]
    kept_reference_d = initial.d[~split_mask]

    stats = gaussian.densify_and_prune(
        max_grad=float(optimization.densify_grad_threshold),
        min_opacity=0.0,
        extent=4.0,
        protected_mask=~trainable_mask,
        densify_mask=trainable_mask,
        max_gaussians=int(args.max_gaussians),
    )

    extension_start = kept_reference_uv.shape[0]
    initial = InitialGeometry(
        uv=torch.cat(
            (kept_reference_uv, gaussian._uv.detach()[extension_start:].clone()),
            dim=0,
        ),
        d=torch.cat(
            (kept_reference_d, gaussian._d.detach()[extension_start:].clone()),
            dim=0,
        ),
        point_count=gaussian.num_gs,
    )
    print_color(
        "Densification at iteration "
        f"{iteration}: cloned={stats['cloned']}, split={stats['split']}, "
        f"points={stats['after']}",
        36,
    )
    return gaussian.point_region_mask("teeth"), initial


@torch.no_grad()
def render_turntable(
    path: Path,
    gaussian: GaussianFlameUVModel,
    pipeline: Any,
    background: torch.Tensor,
    args: Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        path, mode="I", fps=int(args.video_fps), codec="libx264"
    )
    try:
        for index in range(int(args.video_frames)):
            azimuth = torch.tensor(
                [90.0 + 360.0 * index / int(args.video_frames)],
                dtype=torch.float32,
                device=gaussian.device,
            )
            elevation = torch.zeros_like(azimuth)
            distance = torch.full_like(azimuth, 2.0)
            c2w = get_c2w(distance, elevation, azimuth, device=gaussian.device)
            camera = Camera(
                c2w=c2w[0],
                FoVy=torch.tensor(
                    math.radians(70.),
                    dtype=torch.float32,
                    device=gaussian.device,
                ),
                height=int(args.resolution),
                width=int(args.resolution),
                data_device=str(gaussian.device),
            )
            image = render(camera, gaussian, pipeline, background)["render"][:3]
            frame = (image.clamp(0.0, 1.0).mul(255.0).round().byte().permute(1, 2, 0).contiguous().cpu().numpy())
            writer.append_data(frame)
    finally:
        writer.close()


def serializable_arguments(args: Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in vars(args).items():
        if isinstance(value, Path):
            result[name] = str(value)
        else:
            result[name] = value
    return result


def train(args: Namespace, optimization_params: OptimizationParams, pipeline_params: PipelineParams) -> Path:
    device = torch.device("cuda")
    
    expressions = torch.from_numpy(np.load(args.sample_expr_path)).to(device=device, dtype=torch.float32)
    poses = torch.from_numpy(np.load(args.sample_poses_path)).to(device=device, dtype=torch.float32) 

    mouth_prompt = mouth_region_prompt(args.prompt)
    print_color(f"Mouth prompt: {mouth_prompt}", 36)

    gaussian = GaussianFlameUVModel(sh_degree=0, device=str(device))
    gaussian.initialize_flame_state(spatial_lr_scale=4.0, flame_scale=-10.0)
    gaussian.load_ply(str(args.points_cloud))

    optimization = optimization_params.extract(args)

    optimization.shape_lr = 0.0
    gaussian.training_setup(optimization)
    pipeline = pipeline_params.extract(args)

    pipeline = SimpleNamespace(compute_cov3D_python=bool(pipeline.compute_cov3D_python), convert_SHs_python=bool(pipeline.convert_SHs_python), debug=bool(pipeline.debug))

    skel = FlamePointswRandomExp(
        device=str(device),
        batch_size=1,
        image_size=int(args.resolution),
        flame_scale=-10.0,
    )
    skel.betas = gaussian.get_shape.detach().clone()
    background = torch.ones(3, dtype=torch.float32, device=device)

    trainable_mask = gaussian.point_region_mask("teeth")

    initial = InitialGeometry(uv=gaussian._uv.detach().clone(), d=gaussian._d.detach().clone(), point_count=gaussian.num_gs)

    guidance = build_guidance(args)
    prompt_utils = build_prompt_utils(guidance, mouth_prompt, args.negative_prompt)

    all_t = torch.linspace(int(args.ism_begin_step), int(args.ism_min_step), int(args.gaussian_train_iter), device=device).long()
    log_dir = args.output_dir / "log" / "ism"
    result_dir = args.output_dir / "res" / "ism"
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "point_cloud.ply"

    gaussian.optimizer.zero_grad(set_to_none=True)
    print_color("Stage 1: ISM optimization", 36)
    progress = trange(int(args.gaussian_train_iter), desc="Mouth ISM", dynamic_ncols=True)
    for iteration in progress:
        ism_total = 0.0
        regularization_totals = {}
        last_sample = None
        last_ism_result = None

        for accumulation_index in range(int(args.gradient_accumulation)):
            exp_sample_idx =  torch.randint(0, len(expressions), (1,)).item()
            exp = expressions[exp_sample_idx:exp_sample_idx+1]
            pose = poses[exp_sample_idx:exp_sample_idx+1]
            jaw_pose, leye_pose, reye_pose = pose[:, 6:9], pose[:, 9:12], pose[:, 12:15]
            set_pose(gaussian, exp, jaw_pose, leye_pose, reye_pose)

            sample = render_training_sample(gaussian, skel, pipeline, background, args)
            return_vis = bool(iteration % int(args.log_interval) == 0 and accumulation_index == int(args.gradient_accumulation) - 1)
            
            ism_result = ism_step(
                guidance,
                prompt_utils,
                sample,
                all_t[iteration],
                iteration,
                float(args.grad_scale),
                return_vis,
            )
            ism_loss = sum(ism_result["losses"].values())
            regularization = gaussian_regularization(
                gaussian,
                initial,
                trainable_mask,
                sample["visibility_filter"],
                args,
            )
            regularization_loss = sum(regularization.values(), start=ism_loss.new_zeros(()))
            total_loss = (ism_loss + regularization_loss) / float(args.gradient_accumulation)

            total_loss.backward()

            gaussian.mask_out_gradient(trainable_mask, multiplier=0.0)
            gaussian._shape.grad = None

            ism_total += float(ism_loss.detach().item())
            for name, value in regularization.items():
                regularization_totals[name] = (regularization_totals.get(name, 0.0) + float(value.detach().item()))
            last_sample = sample
            last_ism_result = ism_result

        assert last_sample is not None and last_ism_result is not None
        assert_finite_gradients(gaussian)
        gaussian.optimizer.step()
        gaussian.optimizer.zero_grad(set_to_none=True)

        record_densification_statistics(gaussian, last_sample)
        if iteration in set(args.densify_steps):
            trainable_mask, initial = densify_teeth(
                gaussian,
                trainable_mask,
                initial,
                optimization,
                args,
                iteration,
            )

        accumulation = float(args.gradient_accumulation)
        mean_ism = ism_total / accumulation
        mean_reg = {name: value / accumulation for name, value in regularization_totals.items()}
        progress.set_postfix(
            ism=f"{mean_ism:.4f}",
            reg=f"{sum(mean_reg.values()):.4f}",
            t=int(all_t[iteration].item()),
            points=gaussian.num_gs,
        )

        if args.log_interval > 0 and iteration % int(args.log_interval) == 0:
            print(
                f"Iteration {iteration}: ISM={mean_ism:.6f}, "
                + ", ".join(
                    f"{name}={value:.6f}" for name, value in mean_reg.items()
                )
            )
            vis_image = last_ism_result.get("vis_img")
            if vis_image is not None:
                vis_image.save(log_dir / f"{iteration:04d}.png")

        if args.video_interval > 0 and iteration % int(args.video_interval) == 0:
            video_path = log_dir / f"video_step_{iteration:04d}.mp4"
            render_turntable(video_path, gaussian, pipeline, background, args)

    gaussian.save_ply(str(output_path))
    print_color(f"Point cloud saved at {output_path}", 32)
    print_color("ISM optimization finished!", 32)
    return output_path


def main() -> None:
    parser, optimization_params, pipeline_params = build_parser()
    args = parser.parse_args()

    # import debugpy
    # debugpy.listen(6666)
    # print("Waiting for debugger attach (rank 0)...")
    # debugpy.wait_for_client()

    for name in ("output_dir", "points_cloud", "sample_expr_path", "sample_poses_path", "diffusion_path", "controlnet_path", "vae_path"):
        setattr(args, name, resolve_path(getattr(args, name)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(serializable_arguments(args), sort_keys=False, allow_unicode=True), encoding="utf-8")

    torch.backends.cudnn.enabled = False
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    print_color(f"Random seed: {args.seed}", 36)
    train(args, optimization_params, pipeline_params)


if __name__ == "__main__":
    main()
