import io
import math

import cv2
import numpy as np
from dataclasses import dataclass, field
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import threestudio
# from threestudio.utils.poser import Skeleton
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.typing import *
from threestudio.utils.clip_eval import CLIPTextImageEvaluator

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.arguments import PipelineParams, OptimizationParams
from gaussiansplatting.scene.cameras import Camera
from gaussiansplatting.scene.gaussian_flame_face import GaussianFlameUVModel

@threestudio.register("head-3dgs-lks-rig-system")
class Head3DGSLKsRig(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        radius: float = 4
        texture_structure_joint: bool = False
        controlnet: bool = False
        pts_num: int = 100000
        initialize_teeth: bool = False

        disable_hand_densification: bool = False
        hand_radius: float = 0.05
        densify_prune_start_step: int = 300
        densify_prune_end_step: int = 2100
        densify_prune_interval: int = 300
        size_threshold: int = 20
        size_threshold_fix_step: int = 1500
        half_scheduler_max_step: int = 1500
        late_guidance_max_step_percent: float = 0.55
        max_grad: float = 0.0002
        prune_only_start_step: int = 2400
        prune_only_end_step: int = 3300
        prune_only_interval: int = 300
        prune_size_threshold: float = 0.008

        apose: bool = True
        bg_white: bool = False
        bg_random: bool = True
        bg_random_gray_prob: float = 0.7  #
        lower_face_prompt: str = ""
        lower_face_prompt_prefix: str = (
            "a realistic front-facing DSLR close-up crop from nose to chin, "
            "including the nose, lips, mouth, realistic teeth, chin and jaw, "
            "detailed facial skin, "
        )
        lower_face_negative_prompt: str = (
            "top of head, scalp-only crop, hair-only crop, forehead-only crop, "
            "side view, profile view, back view, full body, full figure"
        )
        open_mouth_prompt: str = ""
        open_mouth_prompt_prefix: str = (
            "wide open mouth expression, separated upper and lower lips, "
            "visible upper and lower teeth, dark oral cavity, open jaw, "
            "clear mouth interior, "
        )
        open_mouth_negative_prompt: str = (
            "closed mouth, sealed lips, pursed lips, skin-colored mouth interior, "
            "chin filling the mouth, lower face surface covering the mouth, "
            "fake painted mouth"
        )
        shape_update_end_step: int = 12000
        training_w_animation: bool = True

        # Text-image CLIP evaluation
        clip_eval: bool = True
        # Load three CLIP variants by default
        clip_model_names: tuple = ("ViT-B/16", "ViT-B/32", "ViT-L/14")
        clip_model_root: str = "../HeadStudio_lib/clip"

    cfg: Config

    def configure(self) -> None:
        self.radius = self.cfg.radius
        # self.gaussian = GaussianModel(sh_degree=0)
        self.gaussian = GaussianFlameUVModel(sh_degree=0)
        
        self.background_tensor = torch.tensor([1, 1, 1], dtype=torch.float32,
                                              device="cuda") if self.cfg.bg_white else torch.tensor([0, 0, 0],
                                                                                                    dtype=torch.float32,
                                                                                                    device="cuda")

        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)
        self.pipe.compute_cov3D_python = True

        self.texture_structure_joint = self.cfg.texture_structure_joint
        self.controlnet = self.cfg.controlnet

        self.cameras_extent = 4.0

        # Initialize CLIP evaluator
        self.clip_evaluator = None
        if getattr(self.cfg, "clip_eval", False):
            try:
                threestudio.info(f"Loading CLIP evaluation ...")
                self.clip_evaluator = CLIPTextImageEvaluator(
                    device=self.device,
                    model_name=self.cfg.clip_model_names,
                    model_root=self.cfg.clip_model_root,
                )
            except Exception as e:
                threestudio.info(f"[CLIP Eval] Failed to initialize CLIP models: {e}. CLIP evaluation will be disabled")
                self.clip_evaluator = None

    def save_gif_to_file(self, images, output_file):
        with io.BytesIO() as writer:
            images[0].save(
                writer, format="GIF", save_all=True, append_images=images[1:], duration=100, loop=0
            )
            writer.seek(0)
            with open(output_file, 'wb') as file:
                file.write(writer.read())

    def get_c2w(self, dist, elev, azim):
        elev = elev * math.pi / 180
        azim = azim * math.pi / 180
        batch_size = dist.shape[0]
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                dist * torch.cos(elev) * torch.cos(azim),
                dist * torch.cos(elev) * torch.sin(azim),
                dist * torch.sin(elev),
            ],
            dim=-1,
        )
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions, device=self.device)
        up: Float[Tensor, "B 3"] = torch.as_tensor(
            [0, 0, 1], dtype=torch.float32, device=self.device)[None, :].repeat(batch_size, 1)
        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1], device=self.device)], dim=1
        )
        c2w[:, 3, 3] = 1.0
        return c2w

    def set_pose(self, expression, jaw_pose, leye_pose, reye_pose, neck_pose=None):
        self.gaussian._expression = expression.detach()
        self.gaussian._jaw_pose = jaw_pose.detach()
        self.gaussian._leye_pose = leye_pose.detach()
        self.gaussian._reye_pose = reye_pose.detach()
        if neck_pose is not None:
            self.gaussian._neck_pose = neck_pose.detach()

    def _guidance_cfg(self, name, default=None):
        cfg = self.cfg.guidance
        if isinstance(cfg, dict):
            return cfg.get(name, default)
        return getattr(cfg, name, default)

    def _vsd_active(self) -> bool:
        if not bool(self._guidance_cfg("use_vsd", False)):
            return False
        start_step = max(
            int(self._guidance_cfg("vsd_start_step", 600)),
            int(self._guidance_cfg("vsd_warmup_sds_steps", 600)),
        )
        return self.true_global_step >= start_step

    def forward(self, batch: Dict[str, Any], renderbackground=None) -> Dict[str, Any]:

        if renderbackground is None:
            renderbackground = self.background_tensor

        images = []
        depths = []
        alphas = []
        uv_images = []
        self.viewspace_point_list = []
        self.radii_list = []
        uv = self.gaussian.get_uv  # (N,2)
        uv_color = torch.cat([uv, -torch.ones_like(uv[:, :1])], dim=1)  # (N,3)

        if self.cfg.training_w_animation:
            self.set_pose(
                batch['expression'],
                batch['jaw_pose'],
                batch['leye_pose'],
                batch['reye_pose'],
                # batch.get('neck_pose', None),
            )

        for id in range(batch['c2w'].shape[0]):
            viewpoint_cam = Camera(c2w=batch['c2w'][id], FoVy=batch['fovy'][id].item(), height=batch['height'], width=batch['width'])

            with torch.cuda.amp.autocast(False):
                render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground)
                # pkg_uv = render(viewpoint_cam, self.gaussian, self.pipe, bg_color=self.background_tensor, override_color=uv_color)
            image, viewspace_point_tensor, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)
            self.radii_list.append(radii)
            # uv_img = pkg_uv["render"]  # (3,H,W)
            # uv_images.append(uv_img)

            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii, self.radii)

            depth = render_pkg["depth_3dgs"]
            alpha = render_pkg["alpha_3dgs"]

            depth = depth.permute(1, 2, 0)
            image = image.permute(1, 2, 0).clamp(0, 1)
            alpha = alpha.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)
            alphas.append(alpha)

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        alphas = torch.stack(alphas, 0)
        # uv_images = torch.stack(uv_images, 0)
        # depth_min = torch.amin(depths, dim=[1, 2, 3], keepdim=True)
        # depth_max = torch.amax(depths, dim=[1, 2, 3], keepdim=True)
        # depths = (depths - depth_min) / (depth_max - depth_min + 1e-10)
        # depths = depths.repeat(1, 1, 1, 3)

        self.visibility_filter = self.radii > 0.0

        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        # render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        render_pkg["alpha"] = alphas
        render_pkg["opacity"] = alphas
        # render_pkg["uv_img"] = uv_images
        return {
            **render_pkg,
        }

    def _clone_prompt_config(self):
        if OmegaConf.is_config(self.cfg.prompt_processor):
            return OmegaConf.create(OmegaConf.to_container(self.cfg.prompt_processor, resolve=True))
        return OmegaConf.create(dict(self.cfg.prompt_processor))

    def _make_prompt_processor(self, prompt_text, negative_prompt_extra=""):
        prompt_cfg = self._clone_prompt_config()
        prompt_cfg.prompt = prompt_text
        negative_prompt_extra = negative_prompt_extra.strip()
        if negative_prompt_extra:
            base_negative = prompt_cfg.get("negative_prompt", "")
            prompt_cfg.negative_prompt = (
                f"{base_negative}, {negative_prompt_extra}" if base_negative else negative_prompt_extra
            )
        return threestudio.find(self.cfg.prompt_processor_type)(prompt_cfg), prompt_text

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        base_prompt = self._clone_prompt_config().get("prompt", "")
        self.lower_face_prompt_processor = None
        self.lower_face_prompt_text = None
        self.open_mouth_prompt_processor = None
        self.open_mouth_prompt_text = None
        self.lower_face_open_mouth_prompt_processor = None
        self.lower_face_open_mouth_prompt_text = None
        if self.cfg.lower_face_prompt or self.cfg.lower_face_prompt_prefix:
            lower_face_prompt = self.cfg.lower_face_prompt.strip()
            if not lower_face_prompt:
                lower_face_prompt = f"{self.cfg.lower_face_prompt_prefix}{base_prompt}"
            self.lower_face_prompt_processor, self.lower_face_prompt_text = self._make_prompt_processor(
                lower_face_prompt,
                self.cfg.lower_face_negative_prompt,
            )
        if self.cfg.open_mouth_prompt or self.cfg.open_mouth_prompt_prefix:
            open_mouth_prompt = self.cfg.open_mouth_prompt.strip()
            if not open_mouth_prompt:
                open_mouth_prompt = f"{self.cfg.open_mouth_prompt_prefix}{base_prompt}"
            self.open_mouth_prompt_processor, self.open_mouth_prompt_text = self._make_prompt_processor(
                open_mouth_prompt,
                self.cfg.open_mouth_negative_prompt,
            )
            lower_open_prompt = f"{self.cfg.lower_face_prompt_prefix}{self.cfg.open_mouth_prompt_prefix}{base_prompt}"
            lower_open_negative = ", ".join(
                part.strip()
                for part in (self.cfg.lower_face_negative_prompt, self.cfg.open_mouth_negative_prompt)
                if part.strip()
            )
            self.lower_face_open_mouth_prompt_processor, self.lower_face_open_mouth_prompt_text = self._make_prompt_processor(
                lower_open_prompt,
                lower_open_negative,
            )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)


    def training_step(self, batch, batch_idx):

        self.gaussian.update_learning_rate(self.true_global_step)

        if self.true_global_step > self.cfg.half_scheduler_max_step:
            late_max_step_percent = float(self.cfg.late_guidance_max_step_percent)
            self.guidance.set_min_max_steps(min_step_percent=0.02, max_step_percent=late_max_step_percent)

        bg = None
        r = torch.rand((), device=self.device)
        if r < 0.5:
            bg = torch.ones(3, device=self.device)
        elif r < 0.8:
            bg = torch.zeros(3, device=self.device)
        else:
            bg = torch.rand(3, device=self.device)

        vsd_active = self._vsd_active()

        control_images = batch["flame_conds"]
        if self.true_global_step < 0 and "neutral_flame_conds" in batch:
            self.cfg.training_w_animation = False
            control_images = batch["neutral_flame_conds"]
        else:
            self.cfg.training_w_animation = True

        out = self(batch, renderbackground=bg)
        # out = self(batch)

        is_lower_face = bool(
            "is_lower_face" in batch
            and torch.as_tensor(batch["is_lower_face"], device=self.device).any().item()
        )
        is_open_mouth = bool(
            "is_open_mouth" in batch
            and torch.as_tensor(batch["is_open_mouth"], device=self.device).any().item()
        )
        prompt_utils = self.prompt_processor()
        images = out["comp_rgb"]
        # control_images = out["depth"]

        guidance_eval = False

        prompt_cfg = self.cfg.prompt_processor
        prompt_text = prompt_cfg.get("prompt", None) if isinstance(prompt_cfg, dict) else getattr(prompt_cfg, "prompt", None)
        if (
            is_lower_face
            and is_open_mouth
            and self.lower_face_open_mouth_prompt_processor is not None
        ):
            prompt_utils = self.lower_face_open_mouth_prompt_processor()
            prompt_text = self.lower_face_open_mouth_prompt_text
        elif is_open_mouth and self.open_mouth_prompt_processor is not None:
            prompt_utils = self.open_mouth_prompt_processor()
            prompt_text = self.open_mouth_prompt_text
        elif is_lower_face and self.lower_face_prompt_processor is not None:
            prompt_utils = self.lower_face_prompt_processor()
            prompt_text = self.lower_face_prompt_text
        self.log("train/is_lower_face", float(is_lower_face))
        self.log("train/is_open_mouth", float(is_open_mouth))
        self.log(
            "train/uses_open_mouth_prompt",
            float(is_open_mouth and self.open_mouth_prompt_processor is not None),
        )
        guidance_out = self.guidance(self.true_global_step,
            images.permute(0, 3, 1, 2), control_images.permute(0, 3, 1, 2), prompt_utils,
            **batch, rgb_as_latents=False, rendered_rgb=images, prompt=prompt_text,
        )

        loss = 0.0

        self.log("train/loss_sds", guidance_out['loss_sds'])
        guidance_weight = self.C(self.cfg.loss['lambda_sds'])
        if bool(guidance_out.get("vsd_enabled", torch.zeros((), device=self.device)).detach().item() > 0.5):
            guidance_weight *= self.C(self.cfg.loss.get("lambda_vsd", 1.0))
            self.log("train/loss_vsd", guidance_out["loss_sds"])
        loss = loss + guidance_out['loss_sds'] * guidance_weight

        for key, value in guidance_out.items():
            if not key.startswith("vsd"):
                continue
            if torch.is_tensor(value) and value.numel() == 1:
                log_key = key.replace("vsd_", "vsd/").replace("vsd/", "vsd/")
                self.log(log_key, value.detach())

        if vsd_active and bool(self._guidance_cfg("vsd_debug", True)):
            interval = max(int(self._guidance_cfg("vsd_debug_interval", 200)), 1)
            if self.true_global_step % interval == 0:
                self.save_image_grid(
                    f"vsd/step_{self.true_global_step}_batch.png",
                    [
                        {
                            "type": "rgb",
                            "img": images[i].detach(),
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                        for i in range(images.shape[0])
                    ],
                    name="vsd_batch",
                    step=self.true_global_step,
                )

        scaling = self.gaussian.get_world_scale_max_approx()
        loss_scaling = F.relu(scaling - 0.006).mean()

        self.log("train/loss_scaling", loss_scaling)
        loss += loss_scaling * self.C(self.cfg.loss.lambda_scaling)

        if self.true_global_step >= self.cfg.prune_only_start_step:
            p = self.gaussian.get_opacity.clamp(1e-4, 1 - 1e-4)  # (N,1)
            eps = 0.1
            loss_opaque = (torch.log(p + eps) + torch.log((1.0 + eps) - p) - math.log(eps) - math.log(1.0 + eps)).mean()
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        loss_shape = torch.norm(self.gaussian._shape)
        self.log("train/loss_shape", loss_shape)
        loss += loss_shape * self.C(self.cfg.loss.lambda_shape)

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                guidance_out["eval"],
            )
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        return {"loss": loss}

    def _accumulate_grad_norm_and_radii(self):
        N = self.gaussian.num_gs
        grad_sum = torch.zeros((N, 1), device=self.device)
        vis_cnt = torch.zeros((N, 1), device=self.device)
        for idx in range(len(self.viewspace_point_list)):
            g = self.viewspace_point_list[idx].grad
            if g is None:
                continue
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            # 用当前视角自己的 radii 来算 vis_i
            radii_i = self.radii_list[idx] if hasattr(self, "radii_list") else None
            if radii_i is None:
                vis_i = self.visibility_filter
            else:
                vis_i = radii_i > 0.0

            gn = torch.norm(g[:, :2], dim=-1, keepdim=True)  # (N,1) 先 norm 再加
            grad_sum[vis_i] += gn[vis_i]
            vis_cnt[vis_i] += 1.0
        # 用 mean（每点按可见视角数平均），避免视角数影响阈值
        grad_mean = grad_sum / vis_cnt.clamp_min(1.0)
        return grad_mean

    def on_before_optimizer_step(self, optimizer):
        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], max_norm=1.0)

        with torch.no_grad():

            if self.true_global_step < self.cfg.densify_prune_end_step:
                viewspace_point_tensor_grad = self._accumulate_grad_norm_and_radii()
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)
                # densify_and_prune
                if self.true_global_step > self.cfg.densify_prune_start_step and self.true_global_step % self.cfg.densify_prune_interval == 0:  # 500 100
                    size_threshold = self.cfg.size_threshold if self.true_global_step > self.cfg.size_threshold_fix_step else None  # 3000
                    self.gaussian.densify_and_prune(
                        self.cfg.max_grad,
                        0.05,
                        self.cameras_extent,
                        size_threshold,
                    )

            # prune-only phase according to Gaussian size, rather than the stochastic gradient to eliminate floating artifacts.
            if self.true_global_step > self.cfg.prune_only_start_step and self.true_global_step < self.cfg.prune_only_end_step:
                viewspace_point_tensor_grad = self._accumulate_grad_norm_and_radii()
                # viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                # for idx in range(len(self.viewspace_point_list)):
                #     viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.true_global_step % self.cfg.prune_only_interval == 0:
                    self.gaussian.prune_only(min_opacity=0.05, extent=self.cameras_extent)

            if self.true_global_step > self.cfg.shape_update_end_step:
                for param_group in self.gaussian.optimizer.param_groups:
                    if param_group['name'] == 'flame_shape':
                        param_group['lr'] = 1e-10

    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():
            cnt = self.gaussian.update_face_idx_from_uv(return_stats=True)
            if cnt["updated"] > 0 or cnt.get("projected", 0) > 0:
                threestudio.info(
                    f"[Step {self.true_global_step}] repaired UVD topology: "
                    f"rebound {cnt['updated']}, projected {cnt.get('projected', 0)}, "
                    "using the nearest UV face."
                )

    def on_after_backward(self):
        self.dataset.skel.betas = self.gaussian.get_shape.detach()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            out = self(batch)
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ),
                name="validation_step",
                step=self.true_global_step,
            )
            # save_path = self.get_save_path(f"it{self.true_global_step}-val.ply")
            # self.gaussian.save_ply(save_path)
            # load_ply(save_path,self.get_save_path(f"it{self.true_global_step}-val-color.ply"))
            if self.true_global_step % 500 == 0:
                # save_path = self.get_save_path(f"last.ply")
                save_path = self.get_save_path(f"step_{self.true_global_step}.ply")
                self.gaussian.save_ply(save_path)
                weigth_path = self.get_save_path(f"ckpts/step_{self.true_global_step}.pt")
                # self.gaussian.save_ckpt(weigth_path, self.gaussian.optimizer, step=self.true_global_step)

            # 在验证阶段计算文本-图像 CLIP 相似度（多个模型）
            if self.clip_evaluator is not None:
                # 渲染图像转为 NCHW
                clip_imgs = out["comp_rgb"].permute(0, 3, 1, 2)
                prompts = [self.cfg.prompt_processor.prompt]

                clip_sims_dict = self.clip_evaluator.compute_similarity(clip_imgs, prompts)
                for model_name, sims in clip_sims_dict.items():
                    tag = model_name.replace("/", "").replace("ViT-", "vit_").lower()
                    self.log(f"val/clip_sim_{tag}", sims.mean().item())

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        only_rgb = True
        bg_color = [1, 1, 1] if self.cfg.bg_white else [0, 0, 0]

        testbackground_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        out = self(batch, testbackground_tensor)
        if only_rgb:
            self.save_image_grid(
                f"it{self.true_global_step}-dynamic-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )
        else:
            self.save_image_grid(
                f"it{self.true_global_step}-dynamic-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["depth"][0],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )

    def on_test_epoch_end(self):
        prefix = f"it{self.true_global_step}-dynamic-test"
        self.save_img_sequence(
            prefix,
            prefix,
            r"(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        save_path = self.get_save_path(f"last.ply")
        self.gaussian.save_ply(save_path)

    def configure_optimizers(self):
        opt = OptimizationParams(self.parser)

        self.gaussian.create_from_flame(
            self.cameras_extent,
            -10,
            num_points=self.cfg.pts_num,
            include_teeth=self.cfg.initialize_teeth,
        )
        self.gaussian.training_setup(opt)

        ret = {
            "optimizer": self.gaussian.optimizer,
        }

        return ret

    def guidance_evaluation_save(self, comp_rgb, guidance_eval_out):
        B, size = comp_rgb.shape[:2]
        resize = lambda x: F.interpolate(
            x.permute(0, 3, 1, 2), (size, size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        filename = f"it{self.true_global_step}-train.png"

        def merge12(x):
            return x.reshape(-1, *x.shape[2:])

        self.save_image_grid(
            filename,
            [
                {
                    "type": "rgb",
                    "img": merge12(comp_rgb),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            ),
            name="train_step",
            step=self.true_global_step,
            texts=guidance_eval_out["texts"],
        )
