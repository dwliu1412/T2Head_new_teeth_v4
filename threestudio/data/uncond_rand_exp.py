import bisect
import math
import random
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_device
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *
from threestudio.utils.head_v2 import FlamePointswRandomExp

import os
import numpy as np
import pickle


@dataclass
class RandomCameraDataModuleConfig:
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 512
    width: Any = 512
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    n_val_views: int = 4
    elevation_range: Tuple[float, float] = (-30, 60)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (4., 6.)
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )  # in degrees, in vertical direction (along height)
    camera_perturb: float = 0.
    center_perturb: float = 0.
    up_perturb: float = 0.0
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 6.
    eval_fovy_deg: float = 70.0
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    progressive_until: int = 0  # progressive ranges for elevation, azimuth, r, fovy

    # near head pose
    enable_near_head_poses: bool = False
    enable_near_back_poses: bool = False
    head_offset: float = 0.65
    back_offset: float = 0.65
    head_camera_distance_range: Tuple[float, float] = (0.4, 0.6)
    back_camera_distance_range: Tuple[float, float] = (0.6, 0.8)
    head_fovy_range: Tuple[float, float] = (55, 70)
    back_fovy_range: Tuple[float, float] = (50, 70)
    head_elevation_range: Tuple[float, float] = (-6, 6)
    back_elevation_range: Tuple[float, float] = (-20, 20)
    head_prob: float = 0.25
    head_start_step: int = 1200
    head_end_step: int = 3600
    head_azimuth_range: Tuple[float, float] = (0, 180)
    back_prob: float = 0.20
    back_start_step: int = 1200
    back_end_step: int = 3600
    back_azimuth_range: Tuple[float, float] = (-180, 0)
    frontal_prob: float = 0.0
    frontal_azimuth_range: Tuple[float, float] = (45, 135)

    num_workers: int = 0
    talkshow_train_path: str = "path/to/talkshow_train"
    train_path: str = "path/to/talkshow_train"
    open_mouth_exp_path: str = "./assets/open_mouth_exp.npy"
    open_mouth_pose_path: str = "./assets/open_mouth_pose.npy"
    open_mouth_prob: float = 0.4
    open_mouth_lower_face_prob: float = 0.8
    talkshow_val_path: str = "path/to/talkshow_val"
    test_expression_path: str = "./assets/test/exp.npy"
    test_pose_path: str = "./assets/test/pose.npy"
    test_azimuth_deg: float = 90.0
    test_elevation_deg: float = 0.0
    eval_pose_mode: str = "talkshow"  # "talkshow" or "open_mouth"
    eval_open_mouth_index: int = -1  # < 0 selects the largest jaw-opening axis

    is_lmk: bool = True
    is_mediapipe: bool = True

    training_w_animation: bool = True


class RandomCameraIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]
        if (
                len(self.heights) == 1
                and len(self.widths) == 1
                and len(self.batch_sizes) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.elevation_range = self.cfg.elevation_range
        self.azimuth_range = self.cfg.azimuth_range
        self.camera_distance_range = self.cfg.camera_distance_range
        self.head_camera_distance_range = self.cfg.head_camera_distance_range
        self.back_camera_distance_range = self.cfg.back_camera_distance_range
        self.fovy_range = self.cfg.fovy_range
        self.cur_step = 0

        self.skel = FlamePointswRandomExp(
            device='cuda',
            batch_size=self.cfg.batch_size,
            flame_scale=-10
        )

        self.pose_train_list = np.load(self.cfg.talkshow_train_path, allow_pickle=True)

        self.open_mouth_exp = None
        self.open_mouth_pose = None
        if self.cfg.open_mouth_prob > 0:
            if os.path.exists(self.cfg.open_mouth_exp_path) and os.path.exists(self.cfg.open_mouth_pose_path):
                self.open_mouth_exp = np.load(self.cfg.open_mouth_exp_path, allow_pickle=True)
                self.open_mouth_pose = np.load(self.cfg.open_mouth_pose_path, allow_pickle=True)
            else:
                threestudio.warn(
                    "open_mouth_prob is set, but open-mouth exp/pose files were not found. "
                    "Falling back to talkshow expression sampling only."
                )

    def _to_cuda_tensor(self, array):
        return torch.as_tensor(array, dtype=torch.float32, device="cuda")

    def _sample_talkshow_pose(self):
        idx = random.randint(0, len(self.pose_train_list) - 1)
        pose = self.pose_train_list[idx]
        idx2 = random.randint(0, pose['expression'].shape[0] - 1)
        return (
            self._to_cuda_tensor(pose['expression'][idx2: idx2 + 1]),
            self._to_cuda_tensor(pose['jaw_pose'][idx2: idx2 + 1]),
            self._to_cuda_tensor(pose['leye_pose'][idx2: idx2 + 1]),
            self._to_cuda_tensor(pose['reye_pose'][idx2: idx2 + 1]),
            self._to_cuda_tensor(pose['neck_pose'][idx2: idx2 + 1]),
        )

    def _sample_open_mouth_pose(self):
        idx = random.randint(0, self.open_mouth_exp.shape[0] - 1)
        pose = self.open_mouth_pose[idx: idx + 1]
        return (
            self._to_cuda_tensor(self.open_mouth_exp[idx: idx + 1]),
            self._to_cuda_tensor(pose[:, 6:9]),
            self._to_cuda_tensor(pose[:, 9:12]),
            self._to_cuda_tensor(pose[:, 12:15]),
            self._to_cuda_tensor(pose[:, 3:6]),
        )

    def _sample_animation_pose(self, force_open_mouth=False):
        use_open_mouth = (
            self.open_mouth_exp is not None
            and self.open_mouth_pose is not None
            and (force_open_mouth or random.random() < self.cfg.open_mouth_prob)
        )
        if use_open_mouth:
            return (*self._sample_open_mouth_pose(), True)
        return (*self._sample_talkshow_pose(), False)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        self.width = self.widths[size_ind]
        self.batch_size = self.batch_sizes[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        # threestudio.debug(
        #     f"Training height: {self.height}, width: {self.width}, batch_size: {self.batch_size}"
        # )
        self.cur_step = global_step
        # progressive view
        self.progressive_view(global_step)

    def __iter__(self):
        while True:
            yield {}

    def progressive_view(self, global_step):
        pass

        # r = min(1.0, global_step / (self.cfg.progressive_until + 1))
        # self.elevation_range = [
        #     (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[0],
        #     (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[1],
        # ]
        # self.azimuth_range = [
        #     (1 - r) * 0.0 + r * self.cfg.azimuth_range[0],
        #     (1 - r) * 0.0 + r * self.cfg.azimuth_range[1],
        # ]

        # self.camera_distance_range = [
        #     (1 - r) * self.cfg.eval_camera_distance
        #     + r * self.cfg.camera_distance_range[0],
        #     (1 - r) * self.cfg.eval_camera_distance
        #     + r * self.cfg.camera_distance_range[1],
        # ]
        # self.fovy_range = [
        #     (1 - r) * self.cfg.eval_fovy_deg + r * self.cfg.fovy_range[0],
        #     (1 - r) * self.cfg.eval_fovy_deg + r * self.cfg.fovy_range[1],
        # ]

    def collate(self, batch) -> Dict[str, Any]:

        zoom_in_lower_face = False
        zoom_in_back = False
        camera_distance_range = self.camera_distance_range
        azimuth_range = self.azimuth_range
        fovy_range = self.fovy_range
        elevation_range = self.elevation_range

        if (
            self.cfg.enable_near_head_poses
            and self.cfg.head_start_step <= self.cur_step <= self.cfg.head_end_step
            and random.random() < self.cfg.head_prob
        ):
            zoom_in_lower_face = True
            camera_distance_range = self.head_camera_distance_range
            azimuth_range = self.cfg.head_azimuth_range
            fovy_range = self.cfg.head_fovy_range
            elevation_range = self.cfg.head_elevation_range
        elif (
            self.cfg.enable_near_back_poses
            and self.cfg.back_start_step <= self.cur_step <= self.cfg.back_end_step
            and random.random() < self.cfg.back_prob
        ):
            zoom_in_back = True
            camera_distance_range = self.back_camera_distance_range
            azimuth_range = self.cfg.back_azimuth_range
            fovy_range = self.cfg.back_fovy_range
            elevation_range = self.cfg.back_elevation_range
        elif random.random() < self.cfg.frontal_prob:
            azimuth_range = self.cfg.frontal_azimuth_range

        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                    torch.rand(self.batch_size)
                    * (elevation_range[1] - elevation_range[0])
                    + elevation_range[0]
            )
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (elevation_range[0] + 90.0) / 180.0,
                (elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                        torch.rand(self.batch_size)
                        * (elevation_range_percent[1] - elevation_range_percent[0])
                        + elevation_range_percent[0]
                )
                - 1.0
            )
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        if self.cfg.batch_uniform_azimuth:
            # ensures sampled azimuth angles in a batch cover the whole range
            azimuth_deg = (
                                  torch.rand(self.batch_size) + torch.arange(self.batch_size)
                          ) / self.batch_size * (
                                  azimuth_range[1] - azimuth_range[0]
                          ) + azimuth_range[
                              0
                          ]
        else:
            # simple random sampling
            azimuth_deg = (
                    torch.rand(self.batch_size)
                    * (azimuth_range[1] - azimuth_range[0])
                    + azimuth_range[0]
            )
        azimuth = azimuth_deg * math.pi / 180

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
                torch.rand(self.batch_size)
                * (camera_distance_range[1] - camera_distance_range[0])
                + camera_distance_range[0]
        )

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)

        if zoom_in_lower_face:
            center[:, 2] += self.cfg.head_offset
            camera_positions[:, 2] += self.cfg.head_offset
        elif zoom_in_back:
            center[:, 2] += self.cfg.back_offset
            camera_positions[:, 2] += self.cfg.back_offset

        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(self.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
                torch.rand(self.batch_size, 3) * 2 * self.cfg.camera_perturb
                - self.cfg.camera_perturb
        )
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
                torch.randn(self.batch_size, 3) * self.cfg.center_perturb
        )
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
                torch.randn(self.batch_size, 3) * self.cfg.up_perturb
        )
        up = up + up_perturb

        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
                torch.rand(self.batch_size) * (fovy_range[1] - fovy_range[0])
                + fovy_range[0]
        )
        fovy = fovy_deg * math.pi / 180

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
                torch.rand(self.batch_size)
                * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
                + self.cfg.light_distance_range[0]
        )

        if self.cfg.light_sample_strategy == "dreamfusion" or self.cfg.light_sample_strategy == "dreamfusion3dgs":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(self.batch_size, 3) * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                    light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                    torch.rand(self.batch_size) * math.pi * 2 - math.pi
            )  # [-pi, pi]
            light_elevation = (
                    torch.rand(self.batch_size) * math.pi / 3 + math.pi / 6
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
                                               None, :, :, :
                                               ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
                directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
                                   None, :
                                   ].repeat(self.batch_size, 1)

        if self.cfg.training_w_animation:
            force_open_mouth = (
                zoom_in_lower_face
                and random.random() < self.cfg.open_mouth_lower_face_prob
            )
            expression, jaw_pose, leye_pose, reye_pose, neck_pose, is_open_mouth = self._sample_animation_pose(force_open_mouth=force_open_mouth)
        else:
            expression = None
            jaw_pose = None
            leye_pose = None
            reye_pose = None
            neck_pose = None
            is_open_mouth = False

        flame_depths = self.skel.get_cond(
            dist=camera_distances,               # 8
            elev=elevation_deg,                  # 8
            azim=azimuth_deg,                    # 8
            at=center,                           # 8,3
            up=up,                               # 8,3
            fov=fovy_deg,                        # 8
            expression=expression,               # 1,100
            jaw_pose=jaw_pose,                   # 1,3
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            neck_pose=neck_pose,
            lmk=self.cfg.is_lmk,
            mediapipe=self.cfg.is_mediapipe,
        )
        return {
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
            "fovy": fovy,
            "is_lower_face": torch.full((self.batch_size,), zoom_in_lower_face, dtype=torch.bool, device=camera_positions.device),
            "is_open_mouth": torch.full((self.batch_size,), is_open_mouth, dtype=torch.bool, device=camera_positions.device),
            "flame_conds": flame_depths,
            'expression': expression,
            'jaw_pose': jaw_pose,
            'leye_pose': leye_pose,
            'reye_pose': reye_pose,
            'neck_pose': neck_pose,
        }


class RandomCameraDataset(Dataset):
    @staticmethod
    def _load_animation_sequence(
        expression_path: str,
        pose_path: str,
        label: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        expression = np.asarray(
            np.load(expression_path, allow_pickle=False), dtype=np.float32
        )
        pose = np.asarray(
            np.load(pose_path, allow_pickle=False), dtype=np.float32
        )
        if expression.ndim != 2 or expression.shape[1] < 100:
            raise ValueError(
                f"{expression_path}: expected (T, >=100), "
                f"got {expression.shape}"
            )
        if pose.ndim != 2 or pose.shape[1] < 15:
            raise ValueError(
                f"{pose_path}: expected (T, >=15), got {pose.shape}"
            )
        frame_count = min(expression.shape[0], pose.shape[0])
        if frame_count == 0:
            raise ValueError(
                f"{label.capitalize()} expression/pose sequences must not be empty"
            )
        if expression.shape[0] != pose.shape[0]:
            threestudio.warn(
                f"{label.capitalize()} expression/pose lengths differ "
                f"({expression.shape[0]} vs {pose.shape[0]}); "
                f"using the first {frame_count} paired frames."
            )
        expression = expression[:frame_count, :100].copy()
        pose = pose[:frame_count, :15].copy()
        if not (
            np.isfinite(expression).all()
            and np.isfinite(pose[:, 6:15]).all()
        ):
            raise ValueError(
                f"{label.capitalize()} expression/pose contains NaN or infinity"
            )
        return expression, pose

    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.split = split

        if split == "test":
            expression, pose = self._load_animation_sequence(
                str(self.cfg.test_expression_path),
                str(self.cfg.test_pose_path),
                "test",
            )
            self.n_views = expression.shape[0]
            self.expression = torch.from_numpy(expression).to("cuda")
            # pose layout: global, neck, jaw, left eye, right eye.  Global and
            # neck are intentionally ignored for fixed-head animation.
            self.jaw_pose = torch.from_numpy(pose[:, 6:9]).to("cuda")
            self.leye_pose = torch.from_numpy(pose[:, 9:12]).to("cuda")
            self.reye_pose = torch.from_numpy(pose[:, 12:15]).to("cuda")
            self.neck_pose = None
        else:
            self.n_views = self.cfg.n_val_views
            mode = str(self.cfg.eval_pose_mode).lower()
            if mode == "open_mouth":
                expression, pose = self._load_animation_sequence(
                    str(self.cfg.open_mouth_exp_path),
                    str(self.cfg.open_mouth_pose_path),
                    "validation",
                )
                index = int(self.cfg.eval_open_mouth_index)
                if index < 0:
                    index = int(np.argmax(pose[:, 6]))
                if index >= expression.shape[0]:
                    raise IndexError(
                        f"eval_open_mouth_index={index} is outside validation "
                        f"sequence length {expression.shape[0]}"
                    )
                self.expression = torch.from_numpy(
                    np.repeat(expression[index : index + 1], self.n_views, axis=0)
                ).to("cuda")
                self.jaw_pose = torch.from_numpy(
                    np.repeat(pose[index : index + 1, 6:9], self.n_views, axis=0)
                ).to("cuda")
                self.leye_pose = torch.from_numpy(
                    np.repeat(pose[index : index + 1, 9:12], self.n_views, axis=0)
                ).to("cuda")
                self.reye_pose = torch.from_numpy(
                    np.repeat(pose[index : index + 1, 12:15], self.n_views, axis=0)
                ).to("cuda")
                self.neck_pose = None
                threestudio.info(
                    f"Validation uses open-mouth frame {index} "
                    f"(jaw-x={pose[index, 6]:.5f})."
                )
            elif mode == "talkshow":
                with open(self.cfg.talkshow_val_path, "rb") as f:
                    data = pickle.load(f)
                self.expression = torch.from_numpy(data["expression"]).to("cuda")
                self.jaw_pose = torch.from_numpy(data["jaw_pose"]).to("cuda")
                self.leye_pose = torch.from_numpy(data["leye_pose"]).to("cuda")
                self.reye_pose = torch.from_numpy(data["reye_pose"]).to("cuda")
                self.neck_pose = (
                    torch.from_numpy(data["body_pose_axis"])
                    .reshape(-1, 21, 3)[:, 12]
                    .to("cuda")
                )
                exp_num = self.expression.shape[0]
                self.n_views = min(self.n_views, exp_num)
                if self.n_views < exp_num:
                    idx = torch.linspace(0, exp_num - 1, self.n_views).long()
                    self.expression = self.expression[idx]
                    self.jaw_pose = self.jaw_pose[idx]
                    self.leye_pose = self.leye_pose[idx]
                    self.reye_pose = self.reye_pose[idx]
                    self.neck_pose = self.neck_pose[idx]
            else:
                raise ValueError(
                    "eval_pose_mode must be 'talkshow' or 'open_mouth', "
                    f"got '{self.cfg.eval_pose_mode}'"
                )

        azimuth_deg: Float[Tensor, "B"]
        if self.split == "val":
            # make sure the first and last view are not the same
            azimuth_deg = torch.linspace(60, 120, self.n_views + 1)[: self.n_views]
            elevation_deg: Float[Tensor, "B"] = torch.full_like(
                azimuth_deg, self.cfg.eval_elevation_deg
            )
        else:
            azimuth_deg = torch.full(
                (self.n_views,),
                float(self.cfg.test_azimuth_deg),
                dtype=torch.float32,
            )
            elevation_deg = torch.full_like(
                azimuth_deg, float(self.cfg.test_elevation_deg)
            )
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_camera_distance
        )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
                                   None, :
                                   ].repeat(self.n_views, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180
        light_positions: Float[Tensor, "B 3"] = camera_positions

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        self.mvp_mtx = mvp_mtx
        self.c2w = c2w

        self.camera_positions = camera_positions
        self.light_positions = light_positions
        self.elevation, self.azimuth = elevation, azimuth
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_distances
        self.fovy = fovy

        self.fovy_deg = fovy_deg
        self.up = up
        self.center = center

        self.skel = FlamePointswRandomExp(
            device='cuda',
            batch_size=1,
            flame_scale=-10
        )

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):

        flame_depths = self.skel.get_cond(
            dist=self.camera_distances[index],
            elev=self.elevation_deg[index],
            azim=self.azimuth_deg[index],
            at=self.center[index: index + 1],
            up=self.up[index: index + 1],
            fov=self.fovy_deg[index],
            expression=self.expression[index: index + 1],
            jaw_pose=self.jaw_pose[index: index + 1],
            leye_pose=self.leye_pose[index: index + 1],
            reye_pose=self.reye_pose[index: index + 1],
            neck_pose=(
                self.neck_pose[index: index + 1]
                if self.neck_pose is not None
                else None
            ),
            lmk=self.cfg.is_lmk,
            mediapipe=self.cfg.is_mediapipe,
        )
        result = {
            "index": index,
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.c2w[index],
            "camera_positions": self.camera_positions[index],
            "light_positions": self.light_positions[index],
            "elevation": self.elevation_deg[index],
            "azimuth": self.azimuth_deg[index],
            "camera_distances": self.camera_distances[index],
            "height": self.cfg.eval_height,
            "width": self.cfg.eval_width,
            "fovy": self.fovy[index],
            'expression': self.expression[index],
            'jaw_pose': self.jaw_pose[index],
            'leye_pose': self.leye_pose[index],
            'reye_pose': self.reye_pose[index],
            "flame_conds": flame_depths[0],
        }
        if self.neck_pose is not None:
            result["neck_pose"] = self.neck_pose[index]
        return result

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch


@register("random-camera-exp-datamodule")
class RandomCameraDataModule(pl.LightningDataModule):
    cfg: RandomCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomCameraDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = RandomCameraIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=self.cfg.num_workers,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
