"""Convert a legacy UVD-covariance checkpoint to face-local ellipsoids.

The legacy ``scale_*`` and ``rot_*`` fields lived in the nonlinear UVD
parameterization and cannot be reinterpreted safely.  Reconstruction also
writes ``world.ply`` at its reference FLAME pose.  This tool pulls those
world-space ellipsoids through the saved scene similarity and the reference
face frames, preserving the reference-pose covariance exactly (up to floating
point round-off) while adopting the new GaussianAvatars-style deformation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData
from pytorch3d.transforms import matrix_to_quaternion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaussiansplatting.scene.gaussian_flame_face import GaussianFlameUVModel
from gaussiansplatting.utils.general_utils import build_rotation
from train_reconstruction import (
    aligned_scaling_rotation,
    apply_similarity_to_gaussians,
)


def _archive_parameter(
    archive: dict[str, np.ndarray],
    name: str,
    width: int,
    device: torch.device,
    *,
    default_zero: bool = False,
) -> torch.Tensor:
    if name not in archive:
        if default_zero:
            return torch.zeros((1, width), dtype=torch.float32, device=device)
        raise KeyError(f"Missing {name!r} in reconstruction parameter sidecar")
    value = torch.as_tensor(archive[name], dtype=torch.float32, device=device)
    value = value.reshape(1, -1)
    if value.shape[1] < width:
        raise ValueError(
            f"Sidecar field {name!r} has {value.shape[1]} values; expected {width}"
        )
    return value[:, :width]


def _world_geometry(
    path: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or ())
    scale_names = [f"scale_{index}" for index in range(3)]
    rotation_names = [f"rot_{index}" for index in range(4)]
    position_names = ["x", "y", "z"]
    missing = [
        name
        for name in (*position_names, *scale_names, *rotation_names)
        if name not in names
    ]
    if missing:
        raise ValueError(
            f"World PLY {path} is missing: {', '.join(missing)}"
        )
    raw_scale = np.stack(
        [np.asarray(vertex[name], dtype=np.float32) for name in scale_names],
        axis=1,
    )
    rotation = np.stack(
        [np.asarray(vertex[name], dtype=np.float32) for name in rotation_names],
        axis=1,
    )
    return (
        torch.as_tensor(
            np.stack(
                [
                    np.asarray(vertex[name], dtype=np.float32)
                    for name in position_names
                ],
                axis=1,
            ),
            device=device,
        ),
        torch.as_tensor(raw_scale, device=device).exp(),
        F.normalize(torch.as_tensor(rotation, device=device), dim=-1),
    )


def _set_reference_pose(
    model: GaussianFlameUVModel, archive: dict[str, np.ndarray]
) -> torch.Tensor:
    device = model.device
    shape = _archive_parameter(archive, "shape", 300, device)
    expression = _archive_parameter(archive, "expression", 100, device)
    eyes = _archive_parameter(archive, "eyes", 6, device)
    if "jaw_pose" in archive:
        jaw = _archive_parameter(archive, "jaw_pose", 3, device)
    else:
        pose = _archive_parameter(archive, "pose", 6, device)
        jaw = pose[:, 3:6]
    global_orient = _archive_parameter(
        archive, "global_orient", 3, device, default_zero=True
    )
    neck = _archive_parameter(
        archive, "neck_pose", 3, device, default_zero=True
    )
    zeros = torch.zeros((1, 3), dtype=torch.float32, device=device)

    model._shape.data.copy_(shape)
    model._expression = expression
    model._global_orient = global_orient
    model._neck_pose = neck
    model._jaw_pose = jaw
    model._leye_pose = eyes[:, :3]
    model._reye_pose = eyes[:, 3:6]
    model._translation = zeros
    return torch.as_tensor(
        archive["facelift_from_training"],
        dtype=torch.float32,
        device=device,
    ).reshape(4, 4)


@torch.inference_mode()
def convert(
    input_path: Path,
    output_path: Path,
    world_path: Path,
    params_path: Path,
    device: torch.device,
    spatial_lr_scale: float | None,
    flame_scale: float | None,
) -> dict[str, float | int]:
    with np.load(params_path, allow_pickle=False) as opened:
        archive = {name: np.asarray(opened[name]) for name in opened.files}
    resolved_spatial_lr_scale = (
        float(np.asarray(archive.get("spatial_lr_scale", 4.0)).reshape(-1)[0])
        if spatial_lr_scale is None
        else float(spatial_lr_scale)
    )
    resolved_flame_scale = (
        float(np.asarray(archive.get("flame_scale", -10.0)).reshape(-1)[0])
        if flame_scale is None
        else float(flame_scale)
    )

    model = GaussianFlameUVModel(0, device=str(device))
    model.initialize_flame_state(
        resolved_spatial_lr_scale, resolved_flame_scale
    )
    model.load_ply(str(input_path), allow_legacy_scale_rotation=True)
    alignment = _set_reference_pose(model, archive)

    final_means, final_scales, final_rotations = _world_geometry(
        world_path, device
    )
    if final_scales.shape[0] != model.num_gs:
        raise ValueError(
            "Legacy UVD and world PLY point counts differ: "
            f"{model.num_gs} vs {final_scales.shape[0]}"
        )

    vertices, normals = model._flame_verts_and_normals()
    training_means = model._map_uvd_to_xyz(
        torch.cat((model._uv, model._d), dim=1), vertices, normals
    )
    expected_final_means = (
        training_means @ alignment[:3, :3].T + alignment[:3, 3]
    )
    mean_error = (expected_final_means - final_means).abs().amax()
    if not torch.isfinite(final_means).all() or float(mean_error) > 2.0e-4:
        raise RuntimeError(
            "world.ply does not match the UVD PLY/sidecar reference pose: "
            f"maximum mean error={float(mean_error):.3e}"
        )

    inverse_alignment = torch.linalg.inv(alignment)
    dummy_means = torch.zeros_like(final_scales)
    _, training_scales, training_rotations = apply_similarity_to_gaussians(
        dummy_means, final_scales, final_rotations, inverse_alignment
    )
    face_orientation, face_scale = model._face_properties(vertices)
    local_scales = training_scales / face_scale
    local_rotation_matrix = torch.bmm(
        face_orientation.transpose(1, 2), build_rotation(training_rotations)
    )
    local_rotations = F.normalize(
        matrix_to_quaternion(local_rotation_matrix), dim=-1
    )
    model._scaling.data.copy_(local_scales.clamp_min(1.0e-12).log())
    model._rotation.data.copy_(local_rotations)

    # Compare covariance, rather than raw quaternion/axis order, because both
    # have several equivalent representations for the same ellipsoid.
    _, converted_scales, converted_rotations = aligned_scaling_rotation(
        model, alignment
    )
    source_covariance = model._covariance_matrix_from_scaling_rotation(
        final_scales, final_rotations
    )
    converted_covariance = model._covariance_matrix_from_scaling_rotation(
        converted_scales, converted_rotations
    )
    covariance_error = (converted_covariance - source_covariance).abs()
    relative_error = covariance_error.amax() / source_covariance.abs().amax().clamp_min(
        1.0e-12
    )
    if not torch.isfinite(converted_covariance).all() or float(relative_error) > 2.0e-4:
        raise RuntimeError(
            "Converted reference covariance does not match world.ply: "
            f"relative max error={float(relative_error):.3e}"
        )
    inside, finite = model._uv_inside_faces(
        model.get_uv, model._face_idx, 2.0e-4
    )
    if not bool((inside & finite).all()):
        raise RuntimeError("PLY contains invalid UV/face bindings")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_ply(str(output_path))
    return {
        "points": int(model.num_gs),
        "maximum_mean_error": float(mean_error),
        "relative_max_covariance_error": float(relative_error),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--world-ply", type=Path)
    parser.add_argument("--params", type=Path)
    parser.add_argument("--spatial-lr-scale", type=float)
    parser.add_argument("--flame-scale", type=float)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        raise ValueError("Input and output must differ")
    chained_world = input_path.with_name(f"{input_path.stem}_world.ply")
    default_world = (
        chained_world if chained_world.is_file() else input_path.with_name("world.ply")
    )
    world_path = (
        args.world_ply.resolve()
        if args.world_ply is not None
        else default_world
    )
    chained_params = input_path.with_name(f"{input_path.stem}_params.npz")
    default_params = (
        chained_params
        if chained_params.is_file()
        else input_path.with_name("reconstruction_params.npz")
    )
    params_path = (
        args.params.resolve()
        if args.params is not None
        else default_params
    )
    for description, path in (
        ("input PLY", input_path),
        ("reference world PLY", world_path),
        ("reconstruction sidecar", params_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")

    stats = convert(
        input_path,
        output_path,
        world_path,
        params_path,
        torch.device(args.device),
        args.spatial_lr_scale,
        args.flame_scale,
    )
    print(
        "converted {points} Gaussians to face-local scale/rotation at {path}; "
        "reference mean max error={mean_error:.3e}, covariance relative "
        "error={error:.3e}".format(
            points=stats["points"],
            path=output_path,
            mean_error=stats["maximum_mean_error"],
            error=stats["relative_max_covariance_error"],
        )
    )


if __name__ == "__main__":
    main()
