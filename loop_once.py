import os
import os.path as osp
import math
import pickle
from pathlib import Path
import cv2
import imageio
import random
from argparse import ArgumentParser, Namespace

from PIL import Image
from tqdm.auto import tqdm, trange
import numpy as np
import yaml
from animation import batch_gs_render, batch_mesh_render, save_mp4_w_audio, get_c2w
from gaussiansplatting.arguments import ModelParams, PipelineParams, OptimizationParams
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from threestudio.utils.typing_ import *
from threestudio.utils.head_v2 import FlamePointswRandomExp
from threestudio.utils.perceptual.vgg_feature import VGGPerceptualLoss

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.arguments import PipelineParams
from gaussiansplatting.scene.cameras import Camera
from gaussiansplatting.scene.gaussian_flame_face import GaussianFlameUVModel
from torch.utils.data import DataLoader, Dataset
from threestudio.utils.loss_utils import ssim, l2_loss

# Loop op specific
from stablediff_finetune_control import StableDiffusion
from threestudio.utils.config import load_config
import threestudio

# Train Mouth specific
from sdedit_pipeline import SDeditPipeline
from diffusers import ControlNetModel, AutoencoderKL

import torchvision.transforms as transforms
from model.modnet import MODNet

device = torch.device('cuda')


def save_mp4(images, save_path, step):
    mp4_path = os.path.join(save_path, f"{step}.mp4")
    imageio.mimwrite(mp4_path, images, fps=5)


def load_yaml_config(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


def to_cuda(batch: Dict[str, Any], device: torch.device, non_blocking: bool = True) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out


def masked_l1(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if mask.ndim == 3: mask = mask[:, None]
    mask = mask.to(dtype=pred.dtype)
    num = (mask.sum() * pred.shape[1]).clamp_min(eps)
    return ((pred - tgt).abs() * mask).sum() / num


class Avatar:
    def __init__(self, cfg, gender="generic"):
        self.ply_path = cfg['ply_path']
        self.gaussian = GaussianFlameUVModel(sh_degree=0)
        skel = FlamePointswRandomExp( device="cuda", batch_size=1, flame_scale=-10)
        cameras_extent = 4.0
        flame_scale = -10.0
        self.gaussian.initialize_flame_state(cameras_extent, flame_scale)
        self.gaussian.load_ply(self.ply_path)

        self.black_background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.white_background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        self.renderbackground = self.white_background
        parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(parser)

        self.skel = skel
        self.skel.betas = self.gaussian.get_shape.detach()

        self.parser = ArgumentParser(description="Training script parameters")
        opt = OptimizationParams(self.parser)
        self.gaussian.training_setup(opt)

    def get_cond(self, dist, elev_deg, azim_deg, fovy_deg, expression, jaw_pose, neck_pose,
                 at=torch.tensor(((0, 0, 0),)), up=torch.tensor(((0, 0, 1),))):
        at, up = at.to(torch.float), up.to(torch.float)
        flame_depths = self.skel.get_cond(dist, elev_deg, azim_deg, at, up, fovy_deg, expression=expression,
                                          jaw_pose=jaw_pose, neck_pose=neck_pose, mesh_vis=True)
        return flame_depths

    def get_flame_cond(self, dist, elev_deg, azim_deg, fovy_deg, expression, jaw_pose, neck_pose, leye_pose, reye_pose,
                       at=torch.tensor(((0, 0, 0),)), up=torch.tensor(((0, 0, 1),))):
        at, up = at.to(torch.float), up.to(torch.float)
        flame_depths = self.skel.get_cond(dist, elev_deg, azim_deg, at, up, fovy_deg, expression=expression,leye_pose=leye_pose, reye_pose=reye_pose,
                                          jaw_pose=jaw_pose, neck_pose=neck_pose, lmk=True, mediapipe=True, mesh_vis=True, gaussian_camera_convention=True)
        return flame_depths

    def get_camera(self, dist, elev, azim, fovy_deg=70.0):
        c2w = get_c2w(dist=dist, elev=elev, azim=azim)
        fovy_deg = torch.full_like(elev, fovy_deg)
        fovy = fovy_deg * math.pi / 180
        height, width = 512, 512
        viewpoint_cam = Camera(c2w=c2w[0], FoVy=fovy[0], height=height, width=width)
        return viewpoint_cam

    def set_pose(self, expression, jaw_pose, leye_pose=None, reye_pose=None, neck_pose=None):
        self.gaussian._expression = expression.detach()
        self.gaussian._jaw_pose = jaw_pose.detach()
        if leye_pose is not None: self.gaussian._leye_pose = leye_pose.detach()
        if reye_pose is not None: self.gaussian._reye_pose = reye_pose.detach()
        if neck_pose is not None: self.gaussian._neck_pose = neck_pose.detach()

    def render_mesh(self, dist, elev, azim, expression, jaw_pose, neck_pose, fovy_deg=70.0):
        fovy_deg = torch.full_like(elev, fovy_deg)
        mesh = self.get_cond(dist, elev, azim, fovy_deg, expression, jaw_pose, neck_pose)
        return mesh

    def render(self, dist, elev, azim, bg=None):
        if bg is None: bg = self.renderbackground
        viewpoint_cam = self.get_camera(dist, elev, azim)
        render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, bg)
        image, viewspace_point_tensor, radii, visibility_filter = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["radii"], render_pkg["visibility_filter"]
        alpha = render_pkg.get("alpha_3dgs", None)  # Fallback for train_mouth which might not use alpha_3dgs
        image = image.permute(1, 2, 0)
        return image, viewspace_point_tensor, radii, alpha, visibility_filter

    def render_uv(self, dist, elev, azim):
        if not hasattr(self.gaussian, "get_uv"):
            raise RuntimeError("Gaussian has no get_uv. Use your UV-parameterized Gaussian model first.")
        viewpoint_cam = self.get_camera(dist, elev, azim)
        uv = self.gaussian.get_uv
        uv_color = torch.cat([uv, -torch.ones_like(uv[:, :1])], dim=1)
        pkg_uv = render(viewpoint_cam, self.gaussian, self.pipe, bg_color=self.white_background,
                        override_color=uv_color)
        uv_img = pkg_uv["render"]
        return uv_img

class Trainer:
    def __init__(self, avatar, device=torch.device('cuda'), log_dir='logs', cfg=None):
        self.avatar = avatar
        self.device = device
        self.cfg = cfg

        self._vgg_loss = VGGPerceptualLoss().to(self.device)
        self.cameras_extent = 4.0

        self.densify_prune_start_step = cfg['avatar']['densify_prune_start_step']
        self.densify_prune_end_step = cfg['avatar']['densify_prune_end_step']
        self.densify_prune_interval = cfg['avatar']['densify_prune_interval']
        self.prune_only_start_step = cfg['avatar']['prune_only_start_step']
        self.prune_only_end_step = cfg['avatar']['prune_only_end_step']
        self.prune_only_interval = cfg['avatar']['prune_only_interval']
        self.prune_size_threshold = cfg['avatar']['prune_size_threshold']
        self.size_threshold = cfg['avatar']['size_threshold']
        self.size_threshold_fix_step = cfg['avatar']['size_threshold_fix_step']
        self.max_grad = cfg['avatar']['max_grad']

        # diffusion_path = '../HeadStudio_lib/realistic-vision-51'
        diffusion_path = '../../others/AnimPortrait3D/pretrained_model/Realistic_Vision_V5.1_noVAE'
        vae_path = '../HeadStudio_lib/sd-vae-ft-mse'
        controlnet_path = '../HeadStudio_lib/ControlNetMediaPipeFace'

        vae = AutoencoderKL.from_pretrained(vae_path,torch_dtype=torch.float16).to(device)
        self.sdeditpipeline = SDeditPipeline.from_pretrained(diffusion_path, torch_dtype=torch.float16, vae=vae, safety_checker=None).to(device)
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path, subfolder="diffusion_sd15", torch_dtype=torch.float16, use_safetensors=False).to(device)

        # Init MODNet
        # self.modnet = MODNet(backbone_pretrained=False)
        # self.modnet = torch.nn.DataParallel(self.modnet)
        # modnet_ckpt = cfg['paths'].get('modnet_ckpt_path', 'pretrained/modnet_photographic_portrait_matting.ckpt')

        # if torch.cuda.is_available():
        #     self.modnet = self.modnet.cuda()
        #     weights = torch.load(modnet_ckpt)
        # else:
        #     weights = torch.load(modnet_ckpt, map_location=torch.device('cpu'))

        # self.modnet.load_state_dict(weights)
        # self.modnet.eval()

        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # ===============================================

        self.writer = SummaryWriter(log_dir=log_dir)

    def _apply_modnet_mask(self, image_np: np.ndarray) -> np.ndarray:
        """
        使用 MODNet 提取输入图像的掩码并返回去除背景的图像。
        参数 image_np 应该是一个形状为 (H, W, 3) 且取值在 [0, 255] 的 uint8 numpy 数组。
        """
        im_pil = Image.fromarray(image_np)
        im_tensor = self.im_transform(im_pil).unsqueeze(0).to(self.device)

        im_b, im_c, im_h, im_w = im_tensor.shape
        ref_size = 512

        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im_tensor_resized = F.interpolate(im_tensor, size=(im_rh, im_rw), mode='area')

        with torch.no_grad():
            _, _, matte = self.modnet(im_tensor_resized, True)

        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        matte = np.clip(matte, 0.0, 1.0)

        mask3 = matte[..., None]
        foreground = image_np.astype(np.float32) * mask3
        background = 255.0 * (1.0 - mask3)  # 白背景
        # background = 0.0 #黑背景
        result = np.clip(foreground + background, 0, 255)

        return result.astype(np.uint8)

    def train(self, iterations, save_every_iterations=10, ckpt_dir=None, img_dir=None, refine_log_dir=None):

        all_expr = np.load(osp.join(cfg["paths"]["train_data"], 'open_mouth_exp.npy'))
        all_poses = np.load(osp.join(cfg["paths"]["train_data"], 'open_mouth_pose.npy'))
        all_expr_norm = np.linalg.norm(all_expr, axis=1) 
        min_expr_norm = all_expr_norm.min()

        dist = torch.full((1,), 2.0, device=self.device) 
            
        sampling_idx = [] 
        for i in range(len(all_expr)):
            norm = all_expr_norm[i]
            sample_num = int(norm / min_expr_norm)**2
            sampling_idx.extend([i] * sample_num ) 

            jaw_pose = all_poses[i, 6:9]

            if jaw_pose[0] > 0.2:
                sampling_idx.extend([i] * 10)
        
        progress_bar = tqdm(range(0, iterations), desc="Training progress", leave=False, dynamic_ncols=True)

        for iteration in progress_bar:
            exp_sample_idx =  random.choice(sampling_idx)
            exp = all_expr[exp_sample_idx:exp_sample_idx+1]
            pose = all_poses[exp_sample_idx:exp_sample_idx+1]

            self._optim_zero_grad()

            expression = torch.from_numpy(exp).to(self.device)
            pose = torch.from_numpy(pose).to(self.device)
            jaw_pose, neck_pose, leye_pose, reye_pose = pose[:, 6:9], torch.zeros_like(pose[:, 6:9]), pose[:, 9:12], pose[:, 12:15]

            if random.random() < 0.3:
                azim = torch.rand(1).to(self.device) * 180 + 180
            else:
                azim = torch.rand(1).to(self.device) * 180
            
            elev = torch.rand(1).to(self.device) * 40 - 10

            self.avatar.set_pose(expression=expression, jaw_pose=jaw_pose, neck_pose=neck_pose, leye_pose=leye_pose, reye_pose=reye_pose)
            pred_img, viewspace_point_tensor, radii, alpha, visibility_filter = self.avatar.render(dist, elev, azim)

            mesh = self.avatar.get_flame_cond(dist, elev, azim, 70.0, expression, jaw_pose, neck_pose, leye_pose, reye_pose)[0]
            original_image = (torch.clamp(pred_img, min=0, max=1.0) * 255).byte().contiguous().cpu().numpy()
            original_image = Image.fromarray(original_image).resize((512, 512))
            controlnet_conditioning_image = (torch.clamp(mesh, min=0, max=1.0) * 255).byte().contiguous().cpu().numpy()
            controlnet_conditioning_image = Image.fromarray(controlnet_conditioning_image)

            refined_image = self.sdeditpipeline(
                controlnet = self.controlnet,
                prompt= 'a DSLR portrait of a handsome young brown lightly wavy Pompadour with fade, muscled sportsman in navy satin large pinstripe double-breasted suit, pompadour haircut',
                negative_prompt='oversaturation, low-resolution, unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull',
                image=original_image,
                controlnet_conditioning_image=controlnet_conditioning_image,
                width=original_image.size[0],
                height=original_image.size[1],
                strength=0.3, 
                num_inference_steps=50, 
                guidance_scale =6.,
                controlnet_conditioning_scale=0.65,
                controlnet_guidance_start=0.0,
                controlnet_guidance_end=0.55,
            ).images[0].resize((512, 512))

            refined_image = torch.from_numpy(np.array(refined_image)).to(self.device).float() / 255.0

            loss_l1 = (pred_img - refined_image).abs().mean() * self.cfg['loss']['l1_weight']

            loss_ssim = (1 - ssim(pred_img.permute(2, 0, 1), refined_image.permute(2, 0, 1))) * self.cfg['loss']['ssim_weight']

            total_loss = loss_l1 + loss_ssim

            total_loss.backward()
            self._lr_step(iteration)
            self.on_before_optimizer_step(iteration, visibility_filter, radii, viewspace_point_tensor)
            self.avatar.gaussian.optimizer.step()
            self.avatar.gaussian.update_face_idx_from_uv()

            self.writer.add_scalar("train/loss_l1", loss_l1.item(), iteration)
            self.writer.add_scalar("train/loss_ssim", loss_ssim.item(), iteration)
            self.writer.add_scalar("train/loss_total", total_loss.item(), iteration)

            progress_bar.set_postfix(
                loss=f"{total_loss.item():.6f}", l1=f"{loss_l1.item():.6f}", l_ssim=f"{loss_ssim.item():.6f}",
                num=f"{self.avatar.gaussian._xyz.shape[0]}",
            )

            if save_every_iterations and ((iteration % save_every_iterations == 0)):
                frames = self.test(save_frames=True, save_dir=img_dir, return_numpy=True, step=iteration)
                save_mp4(frames, img_dir, iteration)
                if not ckpt_dir:
                    raise ValueError("There is no folder ckpt_dir")
                ply_path = os.path.join(ckpt_dir, f"step_{iteration:04d}.ply")
                self.avatar.gaussian.save_ply(ply_path)
                print(f"[ckpt] saved {ply_path}")

            if iteration % 10 == 0:
                vis = torch.cat([pred_img, refined_image], dim=1)
                vis = (vis.detach().cpu().numpy() * 255.0).astype(np.uint8)
                out_path = os.path.join(refine_log_dir, f"{iteration:06d}.png")
                imageio.imwrite(out_path, vis)

    @torch.no_grad()
    def test(self, save_frames=False, save_dir=None, return_numpy=True, step="0"):
        frames = []
        os.makedirs(save_dir, exist_ok=True) if (save_frames and save_dir) else None

        all_expr = np.load(osp.join(cfg["paths"]["test_data"], 'exp.npy'))
        all_poses = np.load(osp.join(cfg["paths"]["test_data"], 'pose.npy'))
        num = all_expr.shape[0]

        for i in range(num):
                
            exp = all_expr[i:i+1]
            pose = all_poses[i:i+1]
            expression = torch.from_numpy(exp).to(self.device)
            pose = torch.from_numpy(pose).to(self.device)

            jaw_pose, neck_pose, leye_pose, reye_pose = pose[:, 6:9], torch.zeros_like(pose[:, 6:9]), pose[:, 9:12], pose[:, 12:15]
            dist, elev, azim = torch.full((1,), 2.0, device=self.device), torch.full((1,), 20.0, device=self.device), torch.full((1,), 90.0, device=self.device)
            self.avatar.set_pose(expression=expression, jaw_pose=jaw_pose, neck_pose=neck_pose, leye_pose=leye_pose, reye_pose=reye_pose)
            pred, _, _, _, _ = self.avatar.render(dist, elev, azim)

            if return_numpy:
                img = (pred.clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
                frames.append(img)
                if save_frames and save_dir:
                    out_path = os.path.join(save_dir, f"{step:04d}_{i:06d}.png")
                    imageio.imwrite(out_path, img)
            else:
                frames.append(pred.detach().cpu())

        return frames

    def on_before_optimizer_step(self, iteration, visibility_filter, radii, viewspace_point_tensor):
        with torch.no_grad():
            if iteration < self.densify_prune_end_step:

                self.avatar.gaussian.max_radii2D[visibility_filter] = torch.max(
                    self.avatar.gaussian.max_radii2D[visibility_filter], radii[visibility_filter])
                self.avatar.gaussian.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > self.densify_prune_start_step and iteration % self.densify_prune_interval == 0:
                    size_threshold = self.size_threshold if iteration > self.size_threshold_fix_step else None
                    self.avatar.gaussian.densify_and_prune(0.0002, 0.05, self.cameras_extent, size_threshold)

            if iteration > self.prune_only_start_step and iteration < self.prune_only_end_step:

                self.avatar.gaussian.max_radii2D[visibility_filter] = torch.max(
                    self.avatar.gaussian.max_radii2D[visibility_filter], radii[visibility_filter])
                self.avatar.gaussian.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration % self.prune_only_interval == 0:
                    self.avatar.gaussian.prune_only(min_opacity=0.01, extent=self.cameras_extent)

    def _optim_zero_grad(self):
        self.avatar.gaussian.optimizer.zero_grad(set_to_none=True)

    def _lr_step(self, global_step: int):
        gaussian = self.avatar.gaussian
        if hasattr(gaussian, "xyz_scheduler_args"):
            new_lr = gaussian.xyz_scheduler_args(global_step)
            for g in gaussian.optimizer.param_groups:
                if g.get("name") == "xyz":
                    g["lr"] = new_lr


if __name__ == '__main__':
    parser = ArgumentParser(description="Gaussian Head avatar training & testing")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")

    # import debugpy
    # debugpy.listen(6666)
    # print("Waiting for debugger attach (rank 0)...")
    # debugpy.wait_for_client()

    args = parser.parse_args()
    cfg = load_yaml_config(args.config)

    avatar = Avatar(cfg["paths"])

    cur_dir = Path(cfg["paths"]["output_root"]).resolve()

    optim_dir = cur_dir / "optim"
    os.makedirs(optim_dir, exist_ok=True)
    log_dir = optim_dir / "logs"
    img_dir = optim_dir / "imgs"
    ckpt_dir = optim_dir / "ckpt"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    refine_log_dir = cfg["paths"]["img_path"]
    os.makedirs(refine_log_dir, exist_ok=True)

    trainer = Trainer(
        avatar=avatar,
        device=device,
        log_dir=log_dir,
        cfg=cfg,
    )

    trainer.train(
        iterations=cfg['train']['iterations'],
        save_every_iterations=cfg['train']['save_every_iterations'],
        ckpt_dir=ckpt_dir,
        img_dir=img_dir,
        refine_log_dir=refine_log_dir
    )

    trainer.writer.flush()
    trainer.writer.close()