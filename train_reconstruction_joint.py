"""Joint FaceLift multi-view + LivePortrait FLAME-UVD reconstruction.

Unlike ``train_reconstruction.py``, every observation carries an explicit
``flame_index``.  The training loop switches expression, jaw, and eye state
before deforming the UVD-bound Gaussians for that observation.  Identity shape
and the FaceLift alignment remain shared across the complete dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_reconstruction as base  # noqa: E402
from gaussiansplatting.scene.gaussian_flame_face import (  # noqa: E402
    GaussianFlameUVModel,
)
from gaussiansplatting.utils.loss_utils import ssim  # noqa: E402
from tools.facelift_joint_dataset import (  # noqa: E402
    FlameStateTable,
    load_flame_state_table,
    load_joint_camera_metadata,
    resolve_joint_parameter_path,
)


@dataclass
class JointView(base.View):
    flame_index: int
    source: str


@dataclass(frozen=True)
class TorchFlameStates:
    shape: torch.Tensor
    expression: torch.Tensor
    global_orient: torch.Tensor
    neck_pose: torch.Tensor
    jaw_pose: torch.Tensor
    eyes: torch.Tensor
    state_names: tuple[str, ...]
    state_sources: tuple[str, ...]

    @classmethod
    def from_numpy(
        cls,
        table: FlameStateTable,
        device: torch.device,
    ) -> "TorchFlameStates":
        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(value, dtype=torch.float32, device=device)

        return cls(
            shape=tensor(table.shape),
            expression=tensor(table.expression),
            global_orient=tensor(table.global_orient),
            neck_pose=tensor(table.neck_pose),
            jaw_pose=tensor(table.jaw_pose),
            eyes=tensor(table.eyes),
            state_names=tuple(table.state_names.tolist()),
            state_sources=tuple(table.state_sources.tolist()),
        )

    @property
    def num_states(self) -> int:
        return int(self.expression.shape[0])

    def apply(self, model: GaussianFlameUVModel, index: int) -> None:
        index = int(index)
        if not 0 <= index < self.num_states:
            raise IndexError(f"Invalid FLAME state {index}")
        model._shape.data.copy_(self.shape)
        model._expression = self.expression[index : index + 1]
        model._global_orient = self.global_orient[index : index + 1]
        model._neck_pose = self.neck_pose[index : index + 1]
        model._jaw_pose = self.jaw_pose[index : index + 1]
        model._leye_pose = self.eyes[index : index + 1, :3]
        model._reye_pose = self.eyes[index : index + 1, 3:6]
        model._translation = torch.zeros(
            (1, 3), dtype=torch.float32, device=model.device
        )


class PoseAwareTrainingViews(Sequence[JointView]):
    """Sequence that applies a view's FLAME state when sampled by base.train."""

    def __init__(
        self,
        views: list[JointView],
        model: GaussianFlameUVModel,
        states: TorchFlameStates,
        source_repeats: dict[str, int] | None = None,
    ) -> None:
        self.views = views
        self.model = model
        self.states = states
        source_repeats = source_repeats or {}
        self.sample_indices: list[int] = []
        for index, view in enumerate(views):
            repeat = int(source_repeats.get(view.source, 1))
            if repeat <= 0:
                raise ValueError(
                    f"sampling.source_repeats[{view.source!r}] must be positive"
                )
            self.sample_indices.extend([index] * repeat)
        if not self.sample_indices:
            raise ValueError("No joint training observations")

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: int | slice) -> JointView | list[JointView]:
        if isinstance(index, slice):
            return [self.views[item] for item in self.sample_indices[index]]
        view = self.views[self.sample_indices[index]]
        self.states.apply(self.model, view.flame_index)
        return view

    def __iter__(self) -> Iterator[JointView]:
        # Iteration is used only to build preview lookups. Pose application is
        # explicit in the dynamic preview/final-render functions below.
        return iter(self.views)


def resolve_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def resolve_input_path(input_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else input_dir / path


def load_joint_views(
    input_dir: Path,
    camera_path: Path,
    table: FlameStateTable,
    device: torch.device,
) -> tuple[dict, list[JointView]]:
    metadata = load_joint_camera_metadata(
        camera_path,
        input_dir=input_dir,
        flame_states=table,
        require_images=True,
    )
    views: list[JointView] = []
    for fallback_index, frame in enumerate(metadata["frames"]):
        frame_index = int(frame.get("frame_index", fallback_index))
        image_path = Path(frame["file_path"])
        if not image_path.is_absolute():
            image_path = input_dir / image_path
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        expected_shape = (int(frame["h"]), int(frame["w"]), 3)
        if image.shape != expected_shape:
            raise ValueError(
                f"Frame {frame_index} image shape {image.shape} does not match "
                f"camera {expected_shape}"
            )
        gt = (
            torch.from_numpy(image.copy())
            .to(device=device, dtype=torch.float32)
            .permute(2, 0, 1)
            / 255.0
        )
        views.append(
            JointView(
                frame_index=frame_index,
                name=image_path.stem,
                azimuth=float(
                    frame.get("azimuth_degrees", frame.get("relative_azimuth_deg", 0.0))
                ),
                elevation=float(
                    frame.get("elevation_degrees", frame.get("elevation_deg", 0.0))
                ),
                image_path=image_path,
                gt=gt,
                gt_u8=image,
                camera=base.OpenCVCamera(frame, device),
                flame_index=int(frame["flame_index"]),
                source=str(frame.get("source", "unknown")),
            )
        )
    return metadata, views


@torch.inference_mode()
def save_joint_training_preview(
    iteration: int,
    views: Sequence[JointView],
    indices: list[int],
    model: GaussianFlameUVModel,
    alignment: torch.Tensor,
    pipeline: SimpleNamespace,
    background: torch.Tensor,
    output_dir: Path,
) -> None:
    states: TorchFlameStates = model._joint_flame_states
    view_lookup = {view.frame_index: view for view in views}
    missing = [int(index) for index in indices if int(index) not in view_lookup]
    if missing:
        raise KeyError(f"preview_view_indices are absent from cameras_joint.json: {missing}")
    rows = []
    for frame_index in indices:
        view = view_lookup[int(frame_index)]
        states.apply(model, view.flame_index)
        geometry = base.packed_geometry(model, alignment)
        package = base.render_view(
            view,
            model,
            alignment,
            pipeline,
            background,
            geometry,
        )
        rendered = base.rgb_u8(package["render"])
        mask = np.repeat(
            base.alpha_u8(package["alpha_3dgs"])[..., None], 3, axis=2
        )
        rows.append(
            np.concatenate(
                [
                    base.add_label(
                        view.gt_u8,
                        f"GT | {view.source} | frame {view.frame_index} | flame {view.flame_index}",
                    ),
                    base.add_label(rendered, f"render | iteration {iteration}"),
                    base.add_label(mask, "alpha mask"),
                ],
                axis=1,
            )
        )
    Image.fromarray(np.concatenate(rows, axis=0)).save(
        output_dir / f"iteration_{iteration:06d}.jpg",
        quality=92,
    )


@torch.inference_mode()
def save_joint_final_views(
    views: list[JointView],
    states: TorchFlameStates,
    model: GaussianFlameUVModel,
    alignment: torch.Tensor,
    pipeline: SimpleNamespace,
    background: torch.Tensor,
    output_dir: Path,
    logger,
) -> None:
    render_dir = output_dir / "render"
    mask_dir = output_dir / "mask"
    comparison_dir = output_dir / "comparison"
    for directory in (render_dir, mask_dir, comparison_dir):
        directory.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    comparison_paths: list[Path] = []
    for view in tqdm(views, desc="Final joint views", dynamic_ncols=True):
        states.apply(model, view.flame_index)
        geometry = base.packed_geometry(model, alignment)
        package = base.render_view(
            view,
            model,
            alignment,
            pipeline,
            background,
            geometry,
        )
        image = package["render"].clamp(0.0, 1.0)
        rendered = base.rgb_u8(image)
        mask = base.alpha_u8(package["alpha_3dgs"])
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
        record = {
            "frame_index": view.frame_index,
            "file": view.image_path.name,
            "source": view.source,
            "flame_index": view.flame_index,
            "flame_state": states.state_names[view.flame_index],
            "l1": float(l1.item()),
            "dssim": float(dssim.item()),
            "psnr": float(psnr_value.item()),
        }
        results.append(record)

        Image.fromarray(rendered).save(render_dir / f"{view.name}.png")
        Image.fromarray(mask).save(mask_dir / f"{view.name}.png")
        comparison = np.concatenate(
            [
                base.add_label(
                    view.gt_u8,
                    f"GT | {view.source} | flame {view.flame_index}",
                ),
                base.add_label(rendered, f"render | PSNR {psnr_value.item():.2f}"),
                base.add_label(error, "absolute error x3"),
                base.add_label(mask_rgb, "alpha mask"),
            ],
            axis=1,
        )
        comparison_path = comparison_dir / f"{view.name}.jpg"
        Image.fromarray(comparison).save(comparison_path, quality=92)
        comparison_paths.append(comparison_path)

    def aggregate(items: list[dict]) -> dict:
        return {
            "num_views": len(items),
            "mean_l1": float(np.mean([item["l1"] for item in items])),
            "mean_dssim": float(np.mean([item["dssim"] for item in items])),
            "mean_psnr": float(np.mean([item["psnr"] for item in items])),
        }

    sources = sorted({item["source"] for item in results})
    summary = {
        **aggregate(results),
        "num_gaussians": model.num_gs,
        "by_source": {
            source: aggregate([item for item in results if item["source"] == source])
            for source in sources
        },
        "per_view": results,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    base.save_contact_sheet(
        comparison_paths,
        output_dir / "contact_sheet.jpg",
        cell_size=(512, 128),
        columns=3,
    )
    logger.info(
        "Final joint metrics: L1 %.6f | DSSIM %.6f | PSNR %.2f",
        summary["mean_l1"],
        summary["mean_dssim"],
        summary["mean_psnr"],
    )
    for source, values in summary["by_source"].items():
        logger.info(
            "Final %s metrics (%d views): L1 %.6f | DSSIM %.6f | PSNR %.2f",
            source,
            values["num_views"],
            values["mean_l1"],
            values["mean_dssim"],
            values["mean_psnr"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reconstruction_joint.yaml"),
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--camera-json", default=None)
    parser.add_argument("--flame-params", default=None)
    parser.add_argument("--alignment", default=None)
    parser.add_argument("--resume-ply", type=Path, default=None)
    parser.add_argument("--start-iteration", type=int, default=0)
    parser.add_argument("--render-driven-only", action="store_true")
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
            f"--start-iteration ({args.start_iteration}) exceeds training.iterations "
            f"({total_iterations})"
        )

    input_dir = resolve_path(args.input or cfg["input_dir"]).resolve()
    camera_value = args.camera_json or cfg.get("camera_file", "cameras_joint.json")
    camera_path = resolve_input_path(input_dir, camera_value).resolve()
    metadata = load_joint_camera_metadata(
        camera_path,
        input_dir=input_dir,
        require_images=True,
    )
    flame_path = resolve_joint_parameter_path(
        input_dir,
        metadata,
        args.flame_params or cfg.get("flame_parameter_file"),
    ).resolve()
    flame_table = load_flame_state_table(flame_path)
    # Validate camera-to-state mappings once both files are known.
    metadata = load_joint_camera_metadata(
        camera_path,
        input_dir=input_dir,
        flame_states=flame_table,
        require_images=True,
    )
    alignment_value = args.alignment or cfg.get(
        "alignment_file", "flame_alignment_joint/alignment.npz"
    )
    alignment_path = resolve_input_path(input_dir, alignment_value).resolve()
    if not alignment_path.is_file():
        raise FileNotFoundError(
            f"Joint alignment does not exist: {alignment_path}. Run "
            "tools/align_facelift_flame_joint.py first."
        )

    resume_ply = resolve_path(args.resume_ply).resolve() if args.resume_ply else None
    if resume_ply is not None and not resume_ply.is_file():
        raise FileNotFoundError(f"Resume PLY does not exist: {resume_ply}")

    output_root = resolve_path(cfg["output_root"])
    output_dir = (
        resolve_path(args.output).resolve()
        if args.output is not None
        else (output_root / input_dir.name).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["input_dir"] = str(input_dir)
    cfg["camera_file"] = str(camera_path)
    cfg["flame_parameter_file"] = str(flame_path)
    cfg["alignment_file"] = str(alignment_path)
    cfg["resolved_output_dir"] = str(output_dir)
    cfg["resume_ply"] = str(resume_ply) if resume_ply is not None else None
    cfg["start_iteration"] = int(args.start_iteration)
    cfg["render_driven_only"] = bool(args.render_driven_only)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    logger = base.setup_logger(output_dir, append=resume_ply is not None)
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(cfg["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The Gaussian rasterizer requires CUDA")

    _, views = load_joint_views(input_dir, camera_path, flame_table, device)
    states = TorchFlameStates.from_numpy(flame_table, device)
    with np.load(alignment_path, allow_pickle=False) as archive:
        alignment_array = np.asarray(
            archive["facelift_from_training"], dtype=np.float32
        ).copy()
        aligned_state = int(
            np.asarray(archive.get("flame_state_index", 0)).reshape(-1)[0]
        )
        if "shape" not in archive:
            raise ValueError(
                f"{alignment_path} has no shared shape fingerprint; rerun "
                "tools/align_facelift_flame_joint.py"
            )
        aligned_shape = np.asarray(archive["shape"], dtype=np.float32).copy()
        aligned_shape_source = str(
            np.asarray(archive.get("shape_source", "")).reshape(-1)[0]
        )
        aligned_flame_scale = float(
            np.asarray(archive.get("flame_scale", np.nan)).reshape(-1)[0]
        )
    if alignment_array.shape != (4, 4) or not np.isfinite(alignment_array).all():
        raise ValueError(f"Invalid 4x4 alignment matrix in {alignment_path}")
    alignment = torch.as_tensor(
        alignment_array,
        dtype=torch.float32,
        device=device,
    )
    reference_state = int(metadata.get("alignment_flame_index", 0))
    if aligned_state != reference_state:
        raise ValueError(
            f"Alignment was fitted with FLAME state {aligned_state}, but camera "
            f"metadata declares {reference_state}"
        )
    if aligned_shape.shape != flame_table.shape.shape or not np.allclose(
        aligned_shape, flame_table.shape, rtol=0.0, atol=1e-7
    ):
        raise ValueError(
            "Alignment was fitted with a different shared shape; rerun "
            "tools/align_facelift_flame_joint.py"
        )
    if aligned_shape_source != flame_table.shape_source:
        raise ValueError(
            f"Alignment shape source '{aligned_shape_source}' does not match "
            f"FLAME parameters '{flame_table.shape_source}'"
        )

    model_cfg = cfg["model"]
    if not np.isfinite(aligned_flame_scale) or not math.isclose(
        aligned_flame_scale,
        float(model_cfg["flame_scale"]),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"Alignment flame_scale {aligned_flame_scale} does not match model "
            f"flame_scale {model_cfg['flame_scale']}"
        )
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
            "Resuming from iteration %d using %s (%d Gaussians); optimizer "
            "and densification statistics are reinitialized",
            args.start_iteration,
            resume_ply,
            model.num_gs,
        )
    states.apply(model, reference_state)
    model._joint_flame_states = states

    repeats = {
        str(key): int(value)
        for key, value in cfg.get("sampling", {}).get("source_repeats", {}).items()
    }
    training_views = PoseAwareTrainingViews(views, model, states, repeats)
    pipeline = SimpleNamespace(
        compute_cov3D_python=True,
        convert_SHs_python=False,
        debug=False,
    )
    background = torch.as_tensor(
        cfg["training"]["background"],
        dtype=torch.float32,
        device=device,
    )

    source_counts = Counter(view.source for view in views)
    logger.info(
        "Loaded %d observations and %d FLAME states from %s",
        len(views),
        states.num_states,
        input_dir,
    )
    logger.info("Observation sources: %s", dict(source_counts))
    logger.info("Sampling source repeats: %s", repeats or {"default": 1})
    logger.info(
        "Shared shape source: %s | LivePortrait pose layout: %s",
        flame_table.shape_source,
        flame_table.live_pose_layout,
    )

    # The static trainer is reused for its optimizer/densification behavior.
    # Its one static-only preview hook is replaced with the pose-aware version.
    base.save_training_preview = save_joint_training_preview
    if not args.render_driven_only:
        logger.info(
            "Joint training for %d iterations (%d effective sampling slots)",
            total_iterations,
            len(training_views),
        )
        base.train(
            training_views,
            model,
            alignment,
            pipeline,
            background,
            cfg,
            output_dir,
            logger,
            start_iteration=args.start_iteration,
        )

        states.apply(model, reference_state)
        model_dir = output_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_ply(str(model_dir / "uvd.ply"))
        base.save_world_ply(model, alignment, model_dir / "world.ply")
        reference_pose = torch.cat(
            [states.global_orient[reference_state : reference_state + 1],
             states.jaw_pose[reference_state : reference_state + 1]],
            dim=1,
        )
        np.savez_compressed(
            model_dir / "reconstruction_params.npz",
            shape=states.shape.detach().cpu().numpy(),
            expression=states.expression[
                reference_state : reference_state + 1
            ].detach().cpu().numpy(),
            pose=reference_pose.detach().cpu().numpy(),
            global_orient=states.global_orient[
                reference_state : reference_state + 1
            ].detach().cpu().numpy(),
            neck_pose=states.neck_pose[
                reference_state : reference_state + 1
            ].detach().cpu().numpy(),
            jaw_pose=states.jaw_pose[
                reference_state : reference_state + 1
            ].detach().cpu().numpy(),
            eyes=states.eyes[
                reference_state : reference_state + 1
            ].detach().cpu().numpy(),
            reference_flame_index=np.asarray(reference_state, dtype=np.int32),
            facelift_from_training=alignment.detach().cpu().numpy(),
            flame_scale=np.float32(model_cfg["flame_scale"]),
            spatial_lr_scale=np.float32(model_cfg["spatial_lr_scale"]),
            scale_rotation_space=np.asarray("flame_face_local_v1"),
            representation_schema_version=np.int64(2),
            joint_camera_file=np.asarray(str(camera_path)),
            joint_flame_parameter_file=np.asarray(str(flame_path)),
        )
        save_joint_final_views(
            views,
            states,
            model,
            alignment,
            pipeline,
            background,
            output_dir / "final_views",
            logger,
        )
    else:
        logger.info("Skipping training and final views; rendering driven output only")

    states.apply(model, reference_state)
    drive_cfg = cfg["drive"]
    view_lookup = {view.frame_index: view for view in views}
    driven_index = int(drive_cfg["camera_index"])
    if driven_index not in view_lookup:
        raise KeyError(f"drive.camera_index {driven_index} is not in joint cameras")
    driven_view = view_lookup[driven_index]
    logger.info(
        "Driven camera: frame %d | %s | azimuth %.1f | elevation %.1f",
        driven_view.frame_index,
        driven_view.image_path.name,
        driven_view.azimuth,
        driven_view.elevation,
    )
    base.render_driven_sequence(
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
    states.apply(model, reference_state)
    logger.info("Joint reconstruction complete: %s", output_dir)


if __name__ == "__main__":
    main()
