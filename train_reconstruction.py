"""Multi-view UVD Gaussian reconstruction for FaceLift outputs."""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from pytorch3d.transforms import matrix_to_quaternion
from tqdm import tqdm

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene.gaussian_flame_face import GaussianFlameUVModel
from gaussiansplatting.scene.gaussian_model import GaussianModel
from gaussiansplatting.utils.general_utils import build_rotation
from gaussiansplatting.utils.graphics_utils import getProjectionMatrix
from gaussiansplatting.utils.loss_utils import l1_loss, ssim


PROJECT_ROOT = Path(__file__).resolve().parent


class OpenCVCamera:
    """Camera wrapper expected by the Gaussian rasterizer."""

    def __init__(self, frame: dict, device: torch.device):
        self.image_width = int(frame["w"])
        self.image_height = int(frame["h"])
        self.FoVx = 2.0 * math.atan(self.image_width / (2.0 * float(frame["fx"])))
        self.FoVy = 2.0 * math.atan(self.image_height / (2.0 * float(frame["fy"])))
        self.znear, self.zfar = 0.01, 100.0

        w2c = torch.as_tensor(frame["w2c"], dtype=torch.float32, device=device)
        self.world_view_transform = w2c.T.contiguous()
        self.projection_matrix = getProjectionMatrix(
            self.znear, self.zfar, self.FoVx, self.FoVy
        ).to(device).T.contiguous()
        self.full_proj_transform = (
            self.world_view_transform @ self.projection_matrix
        )
        self.camera_center = torch.linalg.inv(self.world_view_transform)[3, :3]


@dataclass
class View:
    frame_index: int
    name: str
    azimuth: float
    elevation: float
    image_path: Path
    gt: torch.Tensor
    gt_u8: np.ndarray
    camera: OpenCVCamera


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def setup_logger(output_dir: Path, append: bool = False) -> logging.Logger:
    logger = logging.getLogger("uvd_reconstruction")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_mode = "a" if append else "w"
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(
            output_dir / "train.log", mode=file_mode, encoding="utf-8"
        ),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_views(
    input_dir: Path,
    device: torch.device,
) -> list[View]:
    with (input_dir / "cameras.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    frames = metadata["frames"]

    views, inverse_errors, rotation_determinants = [], [], []
    seen_indices: set[int] = set()
    for frame in frames:
        frame_index = int(frame["frame_index"])

        seen_indices.add(frame_index)

        c2w = np.asarray(frame["c2w"], dtype=np.float64)
        w2c = np.asarray(frame["w2c"], dtype=np.float64)
        inverse_errors.append(float(np.abs(c2w @ w2c - np.eye(4)).max()))
        rotation_determinants.append(float(np.linalg.det(w2c[:3, :3])))

        relative = Path(frame["file_path"])
        image_path = input_dir / relative
        if not image_path.is_file():
            image_path = input_dir / "rgb" / relative.name
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)

        gt = (
            torch.from_numpy(image.copy())
            .to(device=device, dtype=torch.float32)
            .permute(2, 0, 1)
            / 255.0
        )
        views.append(
            View(
                frame_index=frame_index,
                name=image_path.stem,
                azimuth=float(frame["azimuth_degrees"]),
                elevation=float(frame["elevation_degrees"]),
                image_path=image_path,
                gt=gt,
                gt_u8=image,
                camera=OpenCVCamera(frame, device),
            )
        )
    return views


def load_training_parameters(input_dir: Path, device: torch.device):
    with (input_dir / "optim.pkl").open("rb") as file:
        entries = pickle.load(file)
    entry = next(iter(entries.values()))
    shape = torch.as_tensor(np.asarray(entry["shapecode"], dtype=np.float32)[:300], device=device)[None]
    expression = torch.as_tensor(np.asarray(entry["expcode"], dtype=np.float32)[:100], device=device)[None]
    pose = torch.as_tensor(np.asarray(entry["posecode"], dtype=np.float32)[:6], device=device)[None]
    eyes = torch.as_tensor(np.asarray(entry["eyecode"], dtype=np.float32)[:6], device=device)[None]
    alignment = torch.as_tensor(
        np.load(input_dir / "flame_alignment" / "alignment.npz")[
            "facelift_from_training"
        ],
        dtype=torch.float32,
        device=device,
    )
    return shape, expression, pose, eyes, alignment


def set_training_pose(
    model: GaussianFlameUVModel,
    shape: torch.Tensor,
    expression: torch.Tensor,
    pose: torch.Tensor,
    eyes: torch.Tensor,
) -> None:
    zeros = torch.zeros((1, 3), dtype=torch.float32, device=model.device)
    model._shape.data.copy_(shape)
    model._expression = expression
    # model._global_orient = pose[:, :3]
    model._neck_pose = zeros
    model._jaw_pose = pose[:, 3:6]
    model._leye_pose = eyes[:, :3]
    model._reye_pose = eyes[:, 3:6]
    model._translation = zeros


def apply_similarity_to_gaussians(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    alignment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a possibly reflected similarity without forming covariance."""

    linear, translation = alignment[:3, :3], alignment[:3, 3]
    similarity_scale = linear.square().sum().div(3.0).sqrt()
    if not bool(torch.isfinite(similarity_scale).item()) or float(
        similarity_scale.item()
    ) <= 0.0:
        raise ValueError("Alignment has an invalid similarity scale")
    orthogonal = linear / similarity_scale
    identity = torch.eye(
        3, dtype=orthogonal.dtype, device=orthogonal.device
    )
    if not torch.allclose(
        orthogonal.T @ orthogonal, identity, rtol=1e-4, atol=1e-5
    ):
        raise ValueError("Alignment linear block is not a similarity transform")

    means = means @ linear.T + translation
    rotation_matrix = torch.matmul(
        orthogonal[None], build_rotation(rotations)
    )
    # The FaceLift alignment intentionally contains a handedness reflection.
    # A sign flip of one ellipsoid axis makes the matrix proper while leaving
    # R diag(s^2) R^T unchanged, so it can still be stored as a quaternion.
    improper = torch.det(rotation_matrix) < 0.0
    if improper.any():
        rotation_matrix = rotation_matrix.clone()
        rotation_matrix[improper, :, 2] *= -1.0
    rotations = F.normalize(matrix_to_quaternion(rotation_matrix), dim=-1)
    scales = scales * similarity_scale.abs()
    return means, scales, rotations


def aligned_scaling_rotation(
    model: GaussianFlameUVModel,
    alignment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return final-scene means, scales and rotations after posing FLAME."""

    vertices, normals = model._flame_verts_and_normals()
    means = model._map_uvd_to_xyz(
        torch.cat([model._uv, model._d], dim=1), vertices, normals
    )
    scales, rotations = model._deformed_scaling_rotation(vertices)
    return apply_similarity_to_gaussians(
        means, scales, rotations, alignment
    )


def aligned_geometry(
    model: GaussianFlameUVModel,
    alignment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility helper returning covariance built from explicit axes."""

    means, scales, rotations = aligned_scaling_rotation(model, alignment)
    covariance = model._covariance_matrix_from_scaling_rotation(
        scales, rotations
    )
    return means, covariance


def packed_geometry(
    model: GaussianFlameUVModel,
    alignment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return aligned_scaling_rotation(model, alignment)


def world_scale_regularization(
    world_scales: torch.Tensor,
    trainable_mask: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AnimPortrait3D-style hinge loss on aligned world-space scales."""
    trainable_scales = world_scales[trainable_mask]
    if trainable_scales.numel() == 0:
        return world_scales.sum() * 0.0, trainable_scales
    regularizer = F.relu(trainable_scales - float(threshold)).norm(dim=1).mean()
    return regularizer, trainable_scales


def render_view(
    view: View,
    model: GaussianFlameUVModel,
    alignment: torch.Tensor,
    pipeline: SimpleNamespace,
    background: torch.Tensor,
    geometry: tuple[torch.Tensor, ...] | None = None,
    override_color: torch.Tensor | None = None,
    override_opacity: torch.Tensor | None = None,
):
    return render(
        view.camera,
        model,
        pipeline,
        background,
        override_color=override_color,
        override_opacity=override_opacity,
        precomputed_geometry=geometry or packed_geometry(model, alignment),
    )


def rgb_u8(tensor: torch.Tensor) -> np.ndarray:
    return (tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).mul(255.0).byte().cpu().numpy())


def alpha_u8(tensor: torch.Tensor) -> np.ndarray:
    return (tensor.detach().squeeze().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy())


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(output, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def save_contact_sheet(
    paths: list[Path],
    output_path: Path,
    cell_size: tuple[int, int],
    columns: int,
) -> None:
    if not paths:
        return
    cell_width, cell_height = cell_size
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(image, (x, y))
    sheet.save(output_path)


@torch.inference_mode()
def save_training_preview(
    iteration: int,
    views: list[View],
    indices: list[int],
    model: GaussianFlameUVModel,
    alignment: torch.Tensor,
    pipeline: SimpleNamespace,
    background: torch.Tensor,
    output_dir: Path,
) -> None:
    geometry = packed_geometry(model, alignment)
    view_lookup = {view.frame_index: view for view in views}
    rows = []
    for frame_index in indices:
        view = view_lookup[int(frame_index)]
        package = render_view(view, model, alignment, pipeline, background, geometry)
        rendered = rgb_u8(package["render"])
        mask = np.repeat(alpha_u8(package["alpha_3dgs"])[..., None], 3, axis=2)
        rows.append(
            np.concatenate(
                [
                    add_label(view.gt_u8, f"GT | frame {view.frame_index}"),
                    add_label(rendered, f"render | iteration {iteration}"),
                    add_label(mask, "alpha mask"),
                ],
                axis=1,
            )
        )
    Image.fromarray(np.concatenate(rows, axis=0)).save(
        output_dir / f"iteration_{iteration:06d}.jpg", quality=92
    )


def train(
    views: list[View],
    model: GaussianFlameUVModel,
    alignment: torch.Tensor,
    pipeline: SimpleNamespace,
    background: torch.Tensor,
    cfg: dict,
    output_dir: Path,
    logger: logging.Logger,
    start_iteration: int = 0,
) -> None:
    optimization = SimpleNamespace(**cfg["optimization"], shape_lr=0.0)
    model._shape.requires_grad_(False)
    model.training_setup(optimization)
    training_cfg = cfg["training"]
    total_iterations = int(training_cfg["iterations"])
    if not 0 <= start_iteration <= total_iterations:
        raise ValueError(
            f"start_iteration must be between 0 and {total_iterations}, "
            f"got {start_iteration}"
        )
    densify_cfg = cfg["densify"]
    regularization_cfg = cfg.get("regularization", {})
    lambda_scale = float(regularization_cfg.get("lambda_scale", 0.0))
    threshold_scale = float(regularization_cfg.get("threshold_scale", 0.05))
    if lambda_scale < 0.0:
        raise ValueError("regularization.lambda_scale must be non-negative")
    if threshold_scale <= 0.0:
        raise ValueError("regularization.threshold_scale must be positive")
    prune_cfg = cfg.get("prune_only", {})
    prune_enabled = bool(prune_cfg.get("enabled", True))
    frozen_regions = list(
        training_cfg.get("frozen_regions", ["teeth", "oral_cavity"])
    )
    initial_frozen = (
        model.point_region_mask(frozen_regions)
        if frozen_regions
        else torch.zeros(model.num_gs, dtype=torch.bool, device=model.device)
    )
    logger.info(
        "Static reconstruction freezes %d/%d Gaussians bound to %s",
        int(initial_frozen.sum().item()),
        model.num_gs,
        frozen_regions,
    )
    logger.info(
        "World-scale regularization: lambda %.4g | threshold %.4g",
        lambda_scale,
        threshold_scale,
    )
    preview_dir = output_dir / "training_renders"
    intermediate_dir = output_dir / "intermediate_models"
    preview_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train.jsonl"
    generator = random.Random(int(cfg["seed"]))
    # Each completed iteration consumes exactly one view sample. Advancing the
    # local RNG preserves the original view order after a PLY-based resume.
    for _ in range(start_iteration):
        generator.randrange(len(views))
    progress = tqdm(
        range(start_iteration + 1, total_iterations + 1),
        total=total_iterations,
        initial=start_iteration,
        desc="UVD reconstruction",
        dynamic_ncols=True,
    )

    metrics_mode = "a" if start_iteration > 0 else "w"
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for iteration in progress:
            model.update_learning_rate(iteration)
            view = views[generator.randrange(len(views))]
            frozen_mask = (
                model.point_region_mask(frozen_regions)
                if frozen_regions
                else torch.zeros(
                    model.num_gs, dtype=torch.bool, device=model.device
                )
            )
            trainable_mask = ~frozen_mask

            # FLAME is posed first; the learned face-local ellipsoid is then
            # composed into final-scene scale/rotation and passed directly to
            # the rasterizer.  No UVD covariance Jacobian is involved.
            means, world_scales, world_rotations = aligned_scaling_rotation(
                model, alignment
            )
            geometry = (means, world_scales, world_rotations)
            package = render_view(
                view,
                model,
                alignment,
                pipeline,
                background,
                geometry=geometry,
            )
            image = package["render"]
            l1 = l1_loss(image, view.gt)
            dssim = 1.0 - ssim(image, view.gt)
            reconstruction_loss = (
                float(training_cfg["l1_weight"]) * l1
                + float(training_cfg["dssim_weight"]) * dssim
            )
            scale_regularizer, trainable_world_scales = world_scale_regularization(
                world_scales,
                trainable_mask,
                threshold_scale,
            )
            scale_loss = lambda_scale * scale_regularizer
            loss = reconstruction_loss + scale_loss
            loss.backward()
            screen_gradient = package["viewspace_points"].grad.detach()

            with torch.no_grad():
                if iteration < int(densify_cfg["end"]):
                    # Hidden dental/oral points have no reliable supervision in
                    # the static closed-mouth views.  Excluding them here also
                    # prevents densification from changing their point set.
                    visible = package["visibility_filter"] & trainable_mask
                    model.max_radii2D[visible] = torch.maximum(
                        model.max_radii2D[visible], package["radii"][visible]
                    )
                    model.add_densification_stats(screen_gradient, visible)

                # AnimPortrait3D freezes teeth before its optimizer step.  Use
                # our face_idx binding to freeze the full dental/oral property
                # rows (UVD, colour, opacity, scale, and rotation).
                model.mask_out_gradient(trainable_mask, multiplier=0.0)
                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none=True)
                model.update_face_idx_from_uv(mask=trainable_mask)
                protected_mask = (
                    model.point_region_mask(frozen_regions)
                    if frozen_regions
                    else None
                )

                densify_stats = None
                if (
                    int(densify_cfg["start"])
                    <= iteration
                    < int(densify_cfg["end"])
                    and iteration % int(densify_cfg["interval"]) == 0
                ):
                    max_screen_size = densify_cfg.get("max_screen_size")
                    densify_stats = model.densify_and_prune(
                        float(densify_cfg["grad_threshold"]),
                        float(densify_cfg["min_opacity"]),
                        float(cfg["model"]["spatial_lr_scale"]),
                        (
                            None
                            if max_screen_size is None
                            else float(max_screen_size)
                        ),
                        protected_mask=protected_mask,
                    )

                pruned = 0
                if (
                    prune_enabled
                    and int(prune_cfg["start"]) <= iteration < int(prune_cfg["end"])
                    and iteration % int(prune_cfg["interval"]) == 0
                ):
                    protected_mask = (
                        model.point_region_mask(frozen_regions)
                        if frozen_regions
                        else None
                    )
                    pruned = model.prune_only(
                        float(prune_cfg["min_opacity"]),
                        float(cfg["model"]["spatial_lr_scale"]),
                        protected_mask=protected_mask,
                    )

                mse = torch.mean((image.clamp(0.0, 1.0) - view.gt) ** 2)
                current_psnr = float((-10.0 * torch.log10(mse.clamp_min(1e-10))).item())
                if iteration % int(training_cfg["log_interval"]) == 0:
                    record = {
                        "iteration": iteration,
                        "view": view.frame_index,
                        "loss": float(loss.item()),
                        "reconstruction_loss": float(reconstruction_loss.item()),
                        "l1": float(l1.item()),
                        "dssim": float(dssim.item()),
                        "scale_loss": float(scale_loss.item()),
                        "scale_regularizer": float(scale_regularizer.item()),
                        "max_world_scale": (
                            float(trainable_world_scales.max().item())
                            if trainable_world_scales.numel()
                            else 0.0
                        ),
                        "psnr": current_psnr,
                        "gaussians": model.num_gs,
                        "frozen_gaussians": (
                            int(
                                model.point_region_mask(frozen_regions)
                                .sum()
                                .item()
                            )
                            if frozen_regions
                            else 0
                        ),
                        "densify": densify_stats,
                        "pruned": pruned,
                    }
                    metrics_file.write(json.dumps(record) + "\n")
                    metrics_file.flush()
                    progress.set_postfix(
                        loss=f"{record['loss']:.5f}",
                        scale=f"{record['scale_loss']:.5f}",
                        psnr=f"{current_psnr:.2f}",
                        points=model.num_gs,
                    )

                if iteration % int(training_cfg["output_interval"]) == 0:
                    save_training_preview(
                        iteration,
                        views,
                        training_cfg["preview_view_indices"],
                        model,
                        alignment,
                        pipeline,
                        background,
                        preview_dir,
                    )
                    iteration_dir = intermediate_dir / f"iteration_{iteration:06d}"
                    iteration_dir.mkdir(parents=True, exist_ok=True)
                    model.save_ply(str(iteration_dir / "uvd.ply"))
                    save_world_ply(model, alignment, iteration_dir / "world.ply")
                    logger.info(
                        "Iteration %d | loss %.6f (scale %.6f, max %.5f) | "
                        "PSNR %.2f | %d Gaussians | saved %s",
                        iteration,
                        float(loss.item()),
                        float(scale_loss.item()),
                        (
                            float(trainable_world_scales.max().item())
                            if trainable_world_scales.numel()
                            else 0.0
                        ),
                        current_psnr,
                        model.num_gs,
                        iteration_dir,
                    )


@torch.inference_mode()
def save_world_ply(
    model: GaussianFlameUVModel,
    alignment: torch.Tensor,
    path: Path,
) -> None:
    means, scales, rotations = aligned_scaling_rotation(model, alignment)
    world = GaussianModel(model.max_sh_degree)
    world.active_sh_degree = model.active_sh_degree
    world._xyz = means.detach()
    world._features_dc = model._features_dc.detach()
    world._features_rest = model._features_rest.detach()
    world._opacity = model._opacity.detach()
    world._scaling = scales.log().detach()
    world._rotation = rotations.detach()
    world.save_ply(str(path))


@torch.inference_mode()
def save_final_views(
    views: list[View],
    model: GaussianFlameUVModel,
    alignment: torch.Tensor,
    pipeline: SimpleNamespace,
    background: torch.Tensor,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    render_dir = output_dir / "render"
    mask_dir = output_dir / "mask"
    comparison_dir = output_dir / "comparison"
    for directory in (render_dir, mask_dir, comparison_dir):
        directory.mkdir(parents=True, exist_ok=True)

    geometry = packed_geometry(model, alignment)
    results, comparison_paths = [], []
    for view in tqdm(views, desc="Final views", dynamic_ncols=True):
        package = render_view(view, model, alignment, pipeline, background, geometry)
        image = package["render"].clamp(0.0, 1.0)
        rendered = rgb_u8(image)
        mask = alpha_u8(package["alpha_3dgs"])
        error = np.clip(
            np.abs(rendered.astype(np.float32) - view.gt_u8.astype(np.float32))
            * 3.0,
            0,
            255,
        ).astype(np.uint8)
        mask_rgb = np.repeat(mask[..., None], 3, axis=2)
        mse = torch.mean((image - view.gt) ** 2)
        l1 = torch.mean(torch.abs(image - view.gt))
        dssim = 1.0 - ssim(image, view.gt)
        psnr_value = -10.0 * torch.log10(mse.clamp_min(1e-10))
        results.append(
            {
                "frame_index": view.frame_index,
                "file": view.image_path.name,
                "l1": float(l1.item()),
                "dssim": float(dssim.item()),
                "psnr": float(psnr_value.item()),
            }
        )

        Image.fromarray(rendered).save(render_dir / f"{view.name}.png")
        Image.fromarray(mask).save(mask_dir / f"{view.name}.png")
        comparison = np.concatenate(
            [
                add_label(view.gt_u8, "GT"),
                add_label(rendered, f"render | PSNR {psnr_value.item():.2f}"),
                add_label(error, "absolute error x3"),
                add_label(mask_rgb, "alpha mask"),
            ],
            axis=1,
        )
        comparison_path = comparison_dir / f"{view.name}.jpg"
        Image.fromarray(comparison).save(comparison_path, quality=92)
        comparison_paths.append(comparison_path)

    summary = {
        "num_views": len(results),
        "num_gaussians": model.num_gs,
        "mean_l1": float(np.mean([item["l1"] for item in results])),
        "mean_dssim": float(np.mean([item["dssim"] for item in results])),
        "mean_psnr": float(np.mean([item["psnr"] for item in results])),
        "per_view": results,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_contact_sheet(
        comparison_paths,
        output_dir / "contact_sheet.jpg",
        cell_size=(512, 128),
        columns=3,
    )
    logger.info(
        "Final metrics: L1 %.6f | DSSIM %.6f | PSNR %.2f",
        summary["mean_l1"],
        summary["mean_dssim"],
        summary["mean_psnr"],
    )


@torch.inference_mode()
def render_driven_sequence(
    view: View,
    model: GaussianFlameUVModel,
    alignment: torch.Tensor,
    pipeline: SimpleNamespace,
    background: torch.Tensor,
    expression_path: Path,
    pose_path: Path,
    fps: int,
    output_dir: Path,
) -> None:
    expressions = np.load(expression_path).astype(np.float32)
    poses = np.load(pose_path).astype(np.float32)
    frame_count = min(expressions.shape[0], poses.shape[0])
    frame_dir, mask_dir = output_dir / "frames", output_dir / "masks"
    frame_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    video_frames, frame_paths = [], []
    for index in tqdm(range(frame_count), desc="Driven sequence", dynamic_ncols=True):
        expression = torch.as_tensor(
            expressions[index : index + 1], device=model.device
        )
        pose = torch.as_tensor(poses[index : index + 1], device=model.device)
        model._expression = expression
        # model._global_orient = pose[:, 0:3]
        model._neck_pose = pose[:, 3:6]
        model._jaw_pose = pose[:, 6:9]
        model._leye_pose = pose[:, 9:12]
        model._reye_pose = pose[:, 12:15]
        package = render_view(view, model, alignment, pipeline, background)
        image = rgb_u8(package["render"])
        mask = alpha_u8(package["alpha_3dgs"])
        frame_path = frame_dir / f"{index:06d}.png"
        Image.fromarray(image).save(frame_path)
        Image.fromarray(mask).save(mask_dir / f"{index:06d}.png")
        video_frames.append(image)
        frame_paths.append(frame_path)

    imageio.mimsave(output_dir / "driven.mp4", video_frames, fps=int(fps))
    sampled = frame_paths[:: max(frame_count // 12, 1)]
    save_contact_sheet(
        sampled[:12],
        output_dir / "contact_sheet.jpg",
        cell_size=(256, 256),
        columns=4,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/reconstruction.yaml"))
    parser.add_argument("--input", type=Path, default=None, help="Override input_dir, e.g. outputs/facelift_multiview/00000001")
    parser.add_argument("--output", type=Path, default='outputs/reconstruction/00000001')
    parser.add_argument(
        "--resume-ply",
        type=Path,
        default=None,
        help="Resume model parameters from an intermediate UVD PLY",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=0,
        help="Last completed global iteration stored in --resume-ply",
    )
    parser.add_argument(
        "--render-driven-only",
        action="store_true",
        help="Skip training/final-view evaluation and rerender driven output from --resume-ply",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    with config_path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if args.resume_ply is None and args.start_iteration != 0:
        raise ValueError("--start-iteration requires --resume-ply")
    if args.resume_ply is not None and args.start_iteration <= 0:
        raise ValueError("--resume-ply requires --start-iteration greater than 0")
    if args.render_driven_only and args.resume_ply is None:
        raise ValueError("--render-driven-only requires --resume-ply")
    total_iterations = int(cfg["training"]["iterations"])
    if args.start_iteration > total_iterations:
        raise ValueError(
            f"--start-iteration ({args.start_iteration}) exceeds configured "
            f"training.iterations ({total_iterations})"
        )
    resume_ply = resolve_path(args.resume_ply) if args.resume_ply else None
    if resume_ply is not None and not resume_ply.is_file():
        raise FileNotFoundError(f"Resume PLY does not exist: {resume_ply}")

    input_dir = resolve_path(args.input or cfg["input_dir"])
    output_root = resolve_path(cfg["output_root"])
    output_dir = resolve_path(args.output) if args.output else output_root / input_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["input_dir"] = str(input_dir)
    cfg["resolved_output_dir"] = str(output_dir)
    cfg["resume_ply"] = str(resume_ply) if resume_ply is not None else None
    cfg["start_iteration"] = int(args.start_iteration)
    cfg["render_driven_only"] = bool(args.render_driven_only)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    logger = setup_logger(output_dir, append=resume_ply is not None)
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(cfg["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The Gaussian rasterizer requires CUDA")

    views = load_views(input_dir, device)
    shape, expression, pose, eyes, alignment = load_training_parameters(input_dir, device)
    model_cfg = cfg["model"]
    model = GaussianFlameUVModel(int(model_cfg["sh_degree"]), device=str(device))
    model.create_from_flame(
        float(model_cfg["spatial_lr_scale"]),
        float(model_cfg["flame_scale"]),
        num_points=int(model_cfg["initial_points"]),
        include_teeth=bool(model_cfg["include_teeth"]),
        teeth_points=int(model_cfg.get("teeth_points", 0)),
        oral_cavity_points=int(model_cfg.get("oral_cavity_points", 0)),
        teeth_rgb=model_cfg.get("teeth_color"),
        oral_cavity_rgb=model_cfg.get("oral_cavity_color"),
    )
    if resume_ply is not None:
        model.load_ply(str(resume_ply))
        logger.info(
            "Resuming from iteration %d using %s (%d Gaussians); "
            "optimizer and densification statistics are reinitialized",
            args.start_iteration,
            resume_ply,
            model.num_gs,
        )
    set_training_pose(model, shape, expression, pose, eyes)

    pipeline = SimpleNamespace(
        compute_cov3D_python=True,
        convert_SHs_python=False,
        debug=False,
    )
    background = torch.as_tensor(cfg["training"]["background"], dtype=torch.float32, device=device)

    if not args.render_driven_only:
        logger.info(
            "Training %d views for %d iterations from %s",
            len(views),
            int(cfg["training"]["iterations"]),
            input_dir,
        )
        train(
            views,
            model,
            alignment,
            pipeline,
            background,
            cfg,
            output_dir,
            logger,
            start_iteration=args.start_iteration,
        )

        set_training_pose(model, shape, expression, pose, eyes)
        model_dir = output_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_ply(str(model_dir / "uvd.ply"))
        save_world_ply(model, alignment, model_dir / "world.ply")
        np.savez(
            model_dir / "reconstruction_params.npz",
            shape=shape.detach().cpu().numpy(),
            expression=expression.detach().cpu().numpy(),
            pose=pose.detach().cpu().numpy(),
            eyes=eyes.detach().cpu().numpy(),
            facelift_from_training=alignment.detach().cpu().numpy(),
            flame_scale=np.float32(model_cfg["flame_scale"]),
            spatial_lr_scale=np.float32(model_cfg["spatial_lr_scale"]),
            scale_rotation_space=np.asarray("flame_face_local_v1"),
            representation_schema_version=np.int64(2),
        )
        save_final_views(
            views,
            model,
            alignment,
            pipeline,
            background,
            output_dir / "final_views",
            logger,
        )
    else:
        logger.info("Skipping training and final views; rendering driven output only")

    drive_cfg = cfg["drive"]
    view_lookup = {view.frame_index: view for view in views}
    driven_view = view_lookup[int(drive_cfg["camera_index"])]
    logger.info(
        "Driven camera: frame %d | %s | azimuth %.1f | elevation %.1f",
        driven_view.frame_index,
        driven_view.image_path.name,
        driven_view.azimuth,
        driven_view.elevation,
    )
    render_driven_sequence(
        driven_view,
        model,
        alignment,
        pipeline,
        background,
        resolve_path(drive_cfg["exp_path"]),
        resolve_path(drive_cfg["pose_path"]),
        int(drive_cfg["fps"]),
        output_dir / "driven",
    )
    set_training_pose(model, shape, expression, pose, eyes)
    logger.info("Reconstruction complete: %s", output_dir)


if __name__ == "__main__":
    main()
