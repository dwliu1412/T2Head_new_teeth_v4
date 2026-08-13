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


class TVLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])

        # 计算水平和垂直方向的梯度
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()

        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def save_mp4(images, save_path, epoch):
    mp4_path = os.path.join(save_path, f"{epoch}.mp4")
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


def get_lip_weight_mask(avatar, dist, elev, azim, height, width, mouth_weight=5.0):
    lip_verts = avatar.gaussian.get_lip_vertices_world()
    camera = avatar.get_camera(dist, elev, azim)
    full_proj = camera.full_proj_transform

    ones = torch.ones(lip_verts.shape[0], 1, device=lip_verts.device)
    pts_homo = torch.cat([lip_verts, ones], dim=1)
    pts_clip = pts_homo @ full_proj
    pts_ndc = pts_clip[:, :3] / pts_clip[:, 3:4]

    pts_screen = pts_ndc[:, :2]
    pts_x = (pts_screen[:, 0] + 1) * 0.5 * width
    pts_y = (pts_screen[:, 1] + 1) * 0.5 * height

    pts_np = torch.stack([pts_x, pts_y], dim=1).detach().cpu().numpy().astype(np.int32)
    weight_mask = np.ones((height, width), dtype=np.float32)

    if pts_np.shape[0] > 0:
        hull = cv2.convexHull(pts_np)
        cv2.fillPoly(weight_mask, [hull], float(mouth_weight))
        weight_mask = cv2.GaussianBlur(weight_mask, (21, 21), 0)

    return torch.from_numpy(weight_mask).to(lip_verts.device).unsqueeze(0)


def get_mouth_mask(avatar, dist, elev, azim, H, W):
    lip_verts = avatar.gaussian.get_lip_vertices_world()
    cam = avatar.get_camera(dist, elev, azim)
    full_proj = cam.full_proj_transform

    ones = torch.ones(lip_verts.shape[0], 1, device=lip_verts.device)
    pts_homo = torch.cat([lip_verts, ones], dim=1)
    pts_clip = pts_homo @ full_proj
    pts_ndc = pts_clip[:, :3] / pts_clip[:, 3:4]

    pts_screen = pts_ndc[:, :2]
    pts_x = (pts_screen[:, 0] + 1) * 0.5 * W
    pts_y = (pts_screen[:, 1] + 1) * 0.5 * H

    pts_np = torch.stack([pts_x, pts_y], dim=1).detach().cpu().numpy().astype(np.int32)
    mask_np = np.zeros((H, W), dtype=np.uint8)

    if pts_np.shape[0] > 0:
        hull = cv2.convexHull(pts_np)
        cv2.fillPoly(mask_np, [hull], 255)
        kernel = np.ones((15, 15), np.uint8)  # slightly larger kernel as in train_mouth
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)

    mask_t = torch.from_numpy(mask_np).float().to(lip_verts.device) / 255.0
    return mask_t.unsqueeze(0)


def get_mouth_bbox(avatar, dist, elev, azim, H, W, crop_size=200):
    lip_verts = avatar.gaussian.get_lip_vertices_world()
    cam = avatar.get_camera(dist, elev, azim)
    full_proj = cam.full_proj_transform

    ones = torch.ones(lip_verts.shape[0], 1, device=lip_verts.device)
    pts_homo = torch.cat([lip_verts, ones], dim=1)
    pts_clip = pts_homo @ full_proj
    pts_ndc = pts_clip[:, :3] / pts_clip[:, 3:4]

    pts_screen_x = (pts_ndc[:, 0] + 1) * 0.5 * W
    pts_screen_y = (pts_ndc[:, 1] + 1) * 0.5 * H

    lips_min_x = torch.min(pts_screen_x).item()
    lips_max_x = torch.max(pts_screen_x).item()
    lips_min_y = torch.min(pts_screen_y).item()
    lips_max_y = torch.max(pts_screen_y).item()

    center_x = (lips_min_x + lips_max_x) / 2
    center_y = (lips_min_y + lips_max_y) / 2

    half_size = crop_size // 2
    min_x = int(center_x - half_size)
    max_x = min_x + crop_size
    min_y = int(center_y - half_size)
    max_y = min_y + crop_size

    if W >= crop_size:
        if min_x < 0:
            min_x = 0
            max_x = crop_size
        elif max_x > W:
            max_x = W
            min_x = W - crop_size
    else:
        min_x = 0
        max_x = W

    if H >= crop_size:
        if min_y < 0:
            min_y = 0
            max_y = crop_size
        elif max_y > H:
            max_y = H
            min_y = H - crop_size
    else:
        min_y = 0
        max_y = H

    min_x, max_x = int(min_x), int(max_x)
    min_y, max_y = int(min_y), int(max_y)

    if min_x >= max_x or min_y >= max_y:
        return None

    return [min_x, min_y, max_x, max_y]


def _uv_unblend_from_uv_render(uv_render: torch.Tensor, eps: float = 1e-6):
    ch2 = uv_render[:, 2:3]
    alpha = 1.0 - (ch2 + 1.0) / 2.0
    alpha = alpha.clamp(0.0, 1.0)
    bg_mask = alpha <= eps
    uv_nb = (uv_render - (1.0 - alpha)) / alpha.clamp_min(eps)
    return uv_nb, alpha, bg_mask


def tv_uv_loss_seam_aware(uv_render: torch.Tensor, alpha_eps: float = 1e-6,
                          uv_jump_thresh: float = 0.25) -> torch.Tensor:
    uv_nb, alpha, bg_mask = _uv_unblend_from_uv_render(uv_render, eps=alpha_eps)
    uv_nb = torch.nan_to_num(uv_nb, nan=0.0, posinf=0.0, neginf=0.0)

    bg_y = (bg_mask[:, :, 1:] | bg_mask[:, :, :-1])
    bg_x = (bg_mask[:, :, :, 1:] | bg_mask[:, :, :, :-1])

    duv_y = uv_nb[:, :2, 1:] - uv_nb[:, :2, :-1]
    duv_x = uv_nb[:, :2, :, 1:] - uv_nb[:, :2, :, :-1]
    jump_y = (duv_y.pow(2).sum(dim=1, keepdim=True).sqrt() > uv_jump_thresh)
    jump_x = (duv_x.pow(2).sum(dim=1, keepdim=True).sqrt() > uv_jump_thresh)

    mask_y = (bg_y | jump_y).repeat(1, 3, 1, 1)
    mask_x = (bg_x | jump_x).repeat(1, 3, 1, 1)

    diff_y = uv_nb[:, :, 1:] - uv_nb[:, :, :-1]
    diff_x = uv_nb[:, :, :, 1:] - uv_nb[:, :, :, :-1]

    tv_y = diff_y[~mask_y].abs().mean() if (~mask_y).any() else diff_y.abs().mean() * 0.0
    tv_x = diff_x[~mask_x].abs().mean() if (~mask_x).any() else diff_x.abs().mean() * 0.0
    return tv_x + tv_y


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
        height, width = 1024, 1024
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
        image, viewspace_point_tensor, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["radii"]
        alpha = render_pkg.get("alpha_3dgs", None)  # Fallback for train_mouth which might not use alpha_3dgs
        image = image.permute(1, 2, 0)
        return image, viewspace_point_tensor, radii, alpha

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


class NerSemble(Dataset):
    def __init__(self, cfg, is_train=True, return_image=None, img_path_override=None, mode='talk'):
        self.is_train = is_train
        self.return_image = return_image
        self.mode = mode

        if is_train:
            if mode == 'talk':
                self.exp_path = osp.join(cfg["train_data"], 'open_mouth_exp.npy')
                self.pose_path = osp.join(cfg["train_data"], 'open_mouth_pose.npy')
                self.azimuth_range = [-100.0, 140.0]
        else:
            self.exp_path = osp.join(cfg["test_data"], 'exp.npy')
            self.pose_path = osp.join(cfg["test_data"], 'pose.npy')
            self.azimuth_range = [40.0, 140.0]

        self.image_path = img_path_override if img_path_override is not None else cfg["img_path"]
        self.expression = torch.from_numpy(np.load(self.exp_path))

        self.jaw_pose = torch.from_numpy(np.load(self.pose_path))[:, 6:9]
        self.leye_pose = torch.from_numpy(np.load(self.pose_path))[:, 9:12]
        self.reye_pose = torch.from_numpy(np.load(self.pose_path))[:, 12:15]
        self.neck_pose = torch.zeros_like(self.reye_pose)

        self.n_frames = self.expression.shape[0]
        self.elevation_range = [-10.0, 30.0]

        if mode == 'talk':
            face_ratio = 0.5
            n_face = int(self.n_frames * face_ratio)
            n_uniform = self.n_frames - n_face
            center_angle = 90.0
            std_dev = 25.0
            face_samples = torch.normal(mean=center_angle, std=std_dev, size=(n_face,))
            face_samples = torch.clamp(face_samples, self.azimuth_range[0], self.azimuth_range[1])
            uniform_samples = torch.rand(n_uniform) * (self.azimuth_range[1] - self.azimuth_range[0]) + \
                              self.azimuth_range[0]
            azimuth_deg = torch.cat([face_samples, uniform_samples])
            azimuth_deg = azimuth_deg[torch.randperm(self.n_frames)]
        else:
            azimuth_deg = (torch.rand(self.n_frames) + torch.arange(self.n_frames)) / self.n_frames * (
                    self.azimuth_range[1] - self.azimuth_range[0]) + self.azimuth_range[0]

        mask = torch.rand(self.n_frames) < 0.5
        elev_linear = torch.rand(self.n_frames) * (self.elevation_range[1] - self.elevation_range[0]) + \
                      self.elevation_range[0]
        elevation_range_percent = [(self.elevation_range[0] + 90.0) / 180.0, (self.elevation_range[1] + 90.0) / 180.0]
        elev_spherical = torch.asin(2 * (
                torch.rand(self.n_frames) * (elevation_range_percent[1] - elevation_range_percent[0]) +
                elevation_range_percent[0]) - 1.0) * (180.0 / math.pi)
        elevation_deg = torch.where(mask, elev_linear, elev_spherical)
        camera_distances = torch.full_like(elevation_deg, 2.0)

        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_distances
        self.image_list = []

    def switch_image_path(self, new_path):
        self.image_path = new_path
        self._build_image_index()

    def _build_image_index(self):
        names = [f for f in os.listdir(self.image_path) if f.lower().endswith('.png')]
        self.image_list = sorted(names, key=lambda x: int(Path(x).stem))
        self.n_img = len(self.image_list)

    def _read_image(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None: raise FileNotFoundError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx) -> Dict[str, Any]:
        item = {
            'expression': self.expression[idx: idx + 1],
            'jaw_pose': self.jaw_pose[idx: idx + 1],
            'leye_pose': self.leye_pose[idx: idx + 1],
            'reye_pose': self.reye_pose[idx: idx + 1],
            'neck_pose': self.neck_pose[idx: idx + 1],
            'elev': self.elevation_deg[idx: idx + 1],
            'dist': self.camera_distances[idx: idx + 1],
            'idx': idx,
        }
        if self.mode == 'talk':
            item['azim'] = self.azimuth_deg[idx: idx + 1]
        else:
            cur = random.randrange(0, self.n_frames)
            item['azim'] = self.azimuth_deg[cur: cur + 1]

        if self.is_train:
            if self.image_list == []: return item
            img_name = self.image_list[idx]
            img_fp = osp.join(self.image_path, img_name)
            img_t = torch.from_numpy(self._read_image(img_fp)).permute(2, 0, 1)
            item['img'] = img_t
            conf_fp = osp.join(self.image_path, f"{Path(img_name).stem}_conf.npy")
            if osp.exists(conf_fp):
                conf_t = torch.from_numpy(np.load(conf_fp).astype(np.float32)).unsqueeze(0)
                item['conf'] = conf_t

        return item


class Trainer:
    def __init__(self, avatar, train_loader, test_loader=None, device=torch.device('cuda'), log_every=20,
                 log_dir='logs', cfg=None):
        self.avatar = avatar
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.log_every = log_every
        self.cfg = cfg

        self._vgg_loss = VGGPerceptualLoss().to(self.device)
        self._tv_loss = TVLoss(weight=0.1).to(self.device)
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

        diffusion_path = '../HeadStudio_lib/realistic-vision-51'
        diffusion_path = '../../others/AnimPortrait3D/pretrained_model/Realistic_Vision_V5.1_noVAE'
        vae_path = '../HeadStudio_lib/sd-vae-ft-mse'
        controlnet_path = '../HeadStudio_lib/ControlNetMediaPipeFace'

        vae = AutoencoderKL.from_pretrained(vae_path,torch_dtype=torch.float16).to(device)
        self.sdeditpipeline = SDeditPipeline.from_pretrained(diffusion_path, torch_dtype=torch.float16, vae=vae, safety_checker=None).to(device)
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path, subfolder="diffusion_sd15", torch_dtype=torch.float16, use_safetensors=False).to(device)

        # Init MODNet
        self.modnet = MODNet(backbone_pretrained=False)
        self.modnet = torch.nn.DataParallel(self.modnet)
        modnet_ckpt = cfg['paths'].get('modnet_ckpt_path', 'pretrained/modnet_photographic_portrait_matting.ckpt')

        if torch.cuda.is_available():
            self.modnet = self.modnet.cuda()
            weights = torch.load(modnet_ckpt)
        else:
            weights = torch.load(modnet_ckpt, map_location=torch.device('cpu'))

        self.modnet.load_state_dict(weights)
        self.modnet.eval()

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

        # 进行掩码推理
        with torch.no_grad():
            _, _, matte = self.modnet(im_tensor_resized, True)

        # 还原尺寸并裁剪到 [0, 1]
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        matte = np.clip(matte, 0.0, 1.0)

        # 应用掩码到生成的图像，抠除背景
        mask3 = matte[..., None]
        foreground = image_np.astype(np.float32) * mask3
        background = 255.0 * (1.0 - mask3)  # 白背景
        # background = 0.0 #黑背景
        result = np.clip(foreground + background, 0, 255)

        return result.astype(np.uint8)

    @torch.no_grad()
    def update_data(self, path, stage=False):
        os.makedirs(path, exist_ok=True)

        for idx, batch in enumerate(self.train_loader):
            batch = to_cuda(batch, self.device)
            B = batch['expression'].shape[0]
            for b in range(B):
                expression, jaw_pose, neck_pose, leye_pose, reye_pose = batch['expression'][b], batch['jaw_pose'][b], \
                    batch['neck_pose'][b], batch['leye_pose'][b], batch['reye_pose'][b]
                dist, elev, azim = batch['dist'][b], batch['elev'][b], batch['azim'][b]
                idx_t = batch['idx'][b]

                self.avatar.set_pose(expression=expression, jaw_pose=jaw_pose, neck_pose=neck_pose, leye_pose=leye_pose, reye_pose=reye_pose)
                pred_img, _, _, _ = self.avatar.render(dist, elev, azim)
                # cv2.imwrite(r'cur.png', pred_img.detach().cpu().numpy()[..., [2, 1, 0]] * 255)
                mesh = self.avatar.get_flame_cond(dist, elev, azim, 70.0, expression, jaw_pose, neck_pose, leye_pose, reye_pose)[0]
                # mouth_mask = get_mouth_mask(self.avatar, dist, elev, azim, 1024, 1024)
                img = pred_img.permute(2, 0, 1).unsqueeze(0)
                flame_cond = mesh.permute(2, 0, 1).unsqueeze(0).to(device)
                imgs = []

                original_image = (torch.clamp(pred_img, min=0, max=1.0) * 255).byte().contiguous().cpu().numpy()
                original_image = Image.fromarray(original_image).resize((512, 512))
                controlnet_conditioning_image = (torch.clamp(mesh, min=0, max=1.0) * 255).byte().contiguous().cpu().numpy()
                controlnet_conditioning_image = Image.fromarray(controlnet_conditioning_image)

                refined_image_nocontrol = self.sdeditpipeline(
                    controlnet = None,
                    prompt= 'a DSLR portrait of a handsome young brown lightly wavy Pompadour with fade, muscled sportsman in navy satin large pinstripe double-breasted suit, pompadour haircut',
                    negative_prompt='oversaturation, low-resolution, unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull',
                    image=original_image,
                    controlnet_conditioning_image=None,
                    width=original_image.size[0],
                    height=original_image.size[1],
                    strength=0.3, 
                    num_inference_steps=50, 
                    guidance_scale =7.5,
                ).images[0].resize((512, 512))

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
                    guidance_scale =7.5,
                    controlnet_conditioning_scale=0.65,
                    controlnet_guidance_start=0.0,
                    controlnet_guidance_end=0.55,
                ).images[0].resize((512, 512))

                original_image.save('cur.png')
                controlnet_conditioning_image.save('controlnet_conditioning_image.png')
                refined_image_nocontrol.save('refined_nocontrol.png')
                refined_image.save('refined.png')

                g = torch.Generator(device=self.device)
                idx_int = int(idx_t.detach().cpu().item())
                seed = idx_int
                g.manual_seed(seed)
                noise = torch.randn(1, 4, 128, 128, generator=g, device=self.device)
                fine_img = self.sd.denoise_img(self.guidance, img, flame_cond, self.cfg['paths']['config_path'], noise, elev, azim, dist)
                # cv2.imwrite('fine_img.png', fine_img[..., [2, 1, 0]])
                # import pdb;
                # pdb.set_trace()
                imgs.append(fine_img.astype(np.float32) / 255.0)

                stack = np.stack(imgs, axis=0)
                mean, var = stack.mean(axis=0), stack.var(axis=0)
                # unc = var.mean(axis=-1)
                # conf = np.exp(-unc / max(tau, 1e-8)).clip(0.0, 1.0).astype(np.float16)

                out_path = os.path.join(path, f"{idx_int:06d}.png")
                mean_u8 = (mean.clip(0, 1) * 255.0).astype(np.uint8)
                azim_val = azim.detach().cpu().item()
                # 计算与正脸（90度）的最小夹角
                angle_diff = abs((azim_val - 90.0 + 180.0) % 360.0 - 180.0)
                if angle_diff > 90.0:
                    foreground_u8 = mean_u8
                else:
                    foreground_u8 = self._apply_modnet_mask(mean_u8)

                imageio.imwrite(out_path, foreground_u8)

        self.train_loader.dataset.switch_image_path(path)

    def train_global_epoch(self, epoch, epochs, ckpt_dir, img_dir):
        if epoch == 0 or epoch == 3:
            self.update_data(self.cfg['paths']['img_path'], stage=True)

        batch_iter = tqdm(self.train_loader, desc=f"Train LoopOp {epoch + 1}/{epochs}", leave=False, dynamic_ncols=True)
        for batch in batch_iter:
            B = batch['expression'].shape[0]
            self._optim_zero_grad()
            batch = to_cuda(batch, self.device)
            preds, preds_alpha, uv_renders, weight_masks, mouth_masks = [], [], [], [], []
            self.viewspace_point_list = []

            for b in range(B):
                expression, jaw_pose, neck_pose, leye_pose, reye_pose = batch['expression'][b], batch['jaw_pose'][b], \
                    batch['neck_pose'][b], batch['leye_pose'][b], batch['reye_pose'][b]
                dist, elev, azim = batch['dist'][b], batch['elev'][b], batch['azim'][b]

                self.avatar.set_pose(expression=expression, jaw_pose=jaw_pose, neck_pose=neck_pose, leye_pose=leye_pose, reye_pose=reye_pose)
                pred_img, viewspace_point_tensor, radii, alpha = self.avatar.render(dist, elev, azim)
                # uv_img = self.avatar.render_uv(dist, elev, azim)
                # w_mask = get_lip_weight_mask(self.avatar, dist, elev, azim, pred_img.shape[0], pred_img.shape[1],
                #                              mouth_weight=2.0)
                # mouth_mask = get_mouth_mask(self.avatar, dist, elev, azim, 1024, 1024)

                # mouth_masks.append(mouth_mask)
                # weight_masks.append(w_mask)
                # uv_renders.append(uv_img)
                preds_alpha.append(alpha)
                preds.append(pred_img.permute(2, 0, 1))
                self.viewspace_point_list.append(viewspace_point_tensor)

                if b == 0:
                    self.radii = radii
                else:
                    self.radii = torch.max(radii, self.radii)

            self.visibility_filter = self.radii > 0.0
            preds, preds_alpha = torch.stack(preds), torch.stack(preds_alpha)
            # weight_masks, mouth_masks = torch.stack(weight_masks), torch.stack(mouth_masks)
            # uv_renders = torch.stack(uv_renders, dim=0)

            # jaw_open = batch['jaw_pose'][:, 0, 0].abs()
            # big_open_f = (jaw_open >= 0.25).float().view(-1, 1, 1, 1)
            # loss_mouth_alpha = (preds_alpha * mouth_masks * big_open_f).mean()

            diff = (preds - batch['img']).abs()
            # weighted_diff = diff * weight_masks
            loss_l1 = diff.mean() * self.cfg['loss']['l1_weight']
            # loss_vgg = self._vgg_loss(preds, batch['img']) * self.cfg['loss']['vgg_weight']
            loss_ssim = (1 - ssim(preds, batch['img'])) * self.cfg['loss']['ssim_weight']
            # loss_tv = self._tv_loss(preds * mouth_masks)

            # total_loss = loss_l1 + loss_ssim + loss_vgg + loss_tv + loss_mouth_alpha
            total_loss = loss_l1 + loss_ssim

            total_loss.backward()
            self._lr_step(self.global_step)
            self.on_before_optimizer_step()
            self.avatar.gaussian.optimizer.step()

            self.global_step += 1
            self.writer.add_scalar("train/loss_l1", loss_l1.item(), self.global_step)
            self.writer.add_scalar("train/loss_ssim", loss_ssim.item(), self.global_step)
            # self.writer.add_scalar("train/loss_vgg", loss_vgg.item(), self.global_step)
            # self.writer.add_scalar("train/loss_tv", loss_tv.item(), self.global_step)
            self.writer.add_scalar("train/loss_total", total_loss.item(), self.global_step)

            batch_iter.set_postfix(
                loss=f"{total_loss.item():.6f}", l1=f"{loss_l1.item():.6f}", l_ssim=f"{loss_ssim.item():.6f}",
                num=f"{self.avatar.gaussian._xyz.shape[0]}",
            )

    def train(self, epochs=1, save_every_n_epochs=10, ckpt_dir=None, img_dir=None):
        self.global_step = 0
        ep_iter = trange(epochs, desc="Epochs", dynamic_ncols=True)
        for ep in ep_iter:
            self.train_global_epoch(ep, epochs, ckpt_dir, img_dir)

            if save_every_n_epochs and ((ep % save_every_n_epochs == 0) or (ep == epochs - 1)):
                frames = self.test(save_frames=True, save_dir=img_dir, return_numpy=True, epoch=ep)
                save_mp4(frames, img_dir, ep)
                if not ckpt_dir:
                    raise ValueError("There is no folder ckpt_dir")
                ply_path = os.path.join(ckpt_dir, f"epoch_{ep:04d}.ply")
                self.avatar.gaussian.save_ply(ply_path)
                print(f"[ckpt] saved {ply_path}")

    @torch.no_grad()
    def test(self, save_frames=False, save_dir=None, return_numpy=True, epoch="0"):
        frames = []
        os.makedirs(save_dir, exist_ok=True) if (save_frames and save_dir) else None

        frame_idx = 0
        for batch in self.test_loader:
            B = batch['expression'].shape[0]
            batch = to_cuda(batch, self.device)
            for b in range(B):
                expression, jaw_pose, neck_pose, leye_pose, reye_pose = batch['expression'][b], batch['jaw_pose'][b], \
                    batch['neck_pose'][b], batch['leye_pose'][b], batch['reye_pose'][b]
                dist, elev, azim = batch['dist'][b], batch['elev'][b], batch['azim'][b]
                self.avatar.set_pose(expression=expression, jaw_pose=jaw_pose, neck_pose=neck_pose, leye_pose=leye_pose,
                                     reye_pose=reye_pose)
                pred, _, _, _ = self.avatar.render(dist, elev, azim)

                if return_numpy:
                    img = (pred.clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
                    frames.append(img)
                    if save_frames and save_dir:
                        out_path = os.path.join(save_dir, f"{epoch:04d}_{frame_idx:06d}.png")
                        imageio.imwrite(out_path, img)
                else:
                    frames.append(pred.detach().cpu())

                frame_idx += 1
        return frames

    def on_before_optimizer_step(self):
        with torch.no_grad():
            if self.global_step < self.densify_prune_end_step:
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                self.avatar.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                    self.avatar.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                self.avatar.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.global_step > self.densify_prune_start_step and self.global_step % self.densify_prune_interval == 0:
                    size_threshold = self.size_threshold if self.global_step > self.size_threshold_fix_step else None
                    self.avatar.gaussian.densify_and_prune(0.0002, 0.05, self.cameras_extent, size_threshold)

            if self.global_step > self.prune_only_start_step and self.global_step < self.prune_only_end_step:
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                self.avatar.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                    self.avatar.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                self.avatar.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.global_step % self.prune_only_interval == 0:
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

    import debugpy
    debugpy.listen(6666)
    print("Waiting for debugger attach (rank 0)...")
    debugpy.wait_for_client()

    args = parser.parse_args()
    cfg = load_yaml_config(args.config)

    train_dataset_global = NerSemble(cfg["paths"], is_train=True, mode='talk')
    # train_dataset_mouth = NerSemble(cfg["paths"], is_train=True, mode='mouth')
    test_dataset = NerSemble(cfg["paths"], is_train=False, mode='talk')

    train_dataloader_global = DataLoader(train_dataset_global, batch_size=cfg["train"]["train_bs"], shuffle=True)
    # train_dataloader_mouth = DataLoader(train_dataset_mouth, batch_size=cfg["train"]["train_bs"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg["train"]["test_bs"], shuffle=False)

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

    trainer = Trainer(
        avatar=avatar,
        train_loader=train_dataloader_global,
        test_loader=test_dataloader,
        device=device,
        log_every=20,
        log_dir=log_dir,
        cfg=cfg,
    )

    trainer.train(
        epochs=cfg['train']['epochs'],
        save_every_n_epochs=cfg['train']['save_every_n_epochs'],
        ckpt_dir=ckpt_dir,
        img_dir=img_dir
    )

    trainer.writer.flush()
    trainer.writer.close()