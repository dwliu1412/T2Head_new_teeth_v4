"""Align the shared joint-dataset FLAME identity to FaceLift cameras.

Only observations marked ``use_for_alignment=true`` are used.  LivePortrait
frames intentionally do not participate because their expression/articulation
changes between frames; after fitting, the one global similarity transform is
shared by every static and dynamic FLAME state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import align_facelift_flame as base  # noqa: E402
from tools.facelift_joint_dataset import (  # noqa: E402
    FlameStateTable,
    load_flame_state_table,
    load_joint_camera_metadata,
    resolve_joint_parameter_path,
)


DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "facelift_multiview" / "00000002"


def resolve_path(path: Path, root: Path = PROJECT_ROOT) -> Path:
    return path if path.is_absolute() else root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--camera-json", default="cameras_joint.json")
    parser.add_argument(
        "--flame-params",
        default=None,
        help="Override flame_parameter_file recorded in cameras_joint.json.",
    )
    parser.add_argument("--alignment-flame-index", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--flame-scale", type=float, default=-10.0)
    parser.add_argument("--front-azimuth", type=float, default=270.0)
    parser.add_argument("--core-half-range", type=float, default=30.1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.7)
    parser.add_argument("--triangulation-max-median-px", type=float, default=8.0)
    parser.add_argument("--triangulation-max-observation-px", type=float, default=14.0)
    parser.add_argument("--huber-px", type=float, default=4.0)
    parser.add_argument("--max-nfev", type=int, default=500)
    parser.add_argument(
        "--parity", choices=("auto", "proper", "reflected"), default="auto"
    )
    parser.add_argument(
        "--refine-profiles",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--profile-max-view-median-px", type=float, default=20.0)
    parser.add_argument("--profile-max-point-px", type=float, default=16.0)
    parser.add_argument("--profile-min-points", type=int, default=35)
    parser.add_argument(
        "--render", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--render-device", default="cuda:0")
    parser.add_argument("--contact-sheet-cell", type=int, default=224)
    return parser.parse_args()


@torch.inference_mode()
def build_flame_for_state(
    table: FlameStateTable,
    state_index: int,
    flame_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the exact FLAME state referenced by alignment observations."""

    state = table.state(state_index)
    model = base.FlameHead(
        shape_params=300,
        expr_params=100,
        include_mask=True,
        add_teeth=True,
    ).cpu().eval()

    zeros_shape = torch.zeros(1, 300, dtype=torch.float32)
    zeros_expression = torch.zeros(1, 100, dtype=torch.float32)
    zeros3 = torch.zeros(1, 3, dtype=torch.float32)
    zeros6 = torch.zeros(1, 6, dtype=torch.float32)
    neutral = model(
        shape=zeros_shape,
        expr=zeros_expression,
        rotation=zeros3,
        neck=zeros3,
        jaw=zeros3,
        eyes=zeros6,
        translation=zeros3,
        zero_centered_at_root_node=False,
        return_landmarks=False,
    ).squeeze(0)
    vmin, vmax = neutral.amin(dim=0), neutral.amax(dim=0)
    center = (vmin + vmax) / 2.0
    normalization_scale = 0.6 / (vmax - vmin).amax()

    vertices = model(
        shape=torch.from_numpy(state["shape"]),
        expr=torch.from_numpy(state["expression"]),
        rotation=torch.from_numpy(state["global_orient"]),
        neck=torch.from_numpy(state["neck_pose"]),
        jaw=torch.from_numpy(state["jaw_pose"]),
        eyes=torch.from_numpy(state["eyes"]),
        translation=zeros3,
        zero_centered_at_root_node=False,
        return_landmarks=False,
    ).squeeze(0)
    vertices = (vertices - center) * normalization_scale
    vertices = vertices.clone()
    vertices[:, [1, 2]] = vertices[:, [2, 1]]
    vertices *= 1.1 ** (-float(flame_scale))

    faces = model.faces.detach().cpu().numpy().astype(np.int64)
    embedding = np.load(base.FLAME_MEDIAPIPE_LMK_PATH, allow_pickle=True)
    face_indices = np.asarray(embedding["lmk_face_idx"], dtype=np.int64)
    barycentric = np.asarray(embedding["lmk_b_coords"], dtype=np.float64)
    landmark_indices = np.asarray(embedding["landmark_indices"], dtype=np.int64)
    vertices_np = vertices.detach().cpu().numpy().astype(np.float64)
    landmark_triangles = vertices_np[faces[face_indices]]
    landmarks = (landmark_triangles * barycentric[:, :, None]).sum(axis=1)
    return vertices_np, faces, landmarks, landmark_indices


def alignment_frame_indices(metadata: dict, state_index: int) -> set[int]:
    indices: set[int] = set()
    mismatches: list[tuple[int, int]] = []
    for fallback_index, frame in enumerate(metadata["frames"]):
        if not bool(frame.get("use_for_alignment", False)):
            continue
        frame_index = int(frame.get("frame_index", fallback_index))
        flame_index = int(frame["flame_index"])
        if flame_index != state_index:
            mismatches.append((frame_index, flame_index))
        indices.add(frame_index)
    if mismatches:
        raise ValueError(
            "All alignment observations must share one FLAME state; "
            f"expected {state_index}, mismatches={mismatches[:10]}"
        )
    if not indices:
        raise ValueError("No observations are marked use_for_alignment=true")
    return indices


def main() -> None:
    args = parse_args()
    input_dir = resolve_path(args.input_dir).resolve()
    camera_path = resolve_path(Path(args.camera_json), input_dir).resolve()
    metadata = load_joint_camera_metadata(
        camera_path,
        input_dir=input_dir,
        require_images=True,
    )
    flame_path = resolve_joint_parameter_path(
        input_dir, metadata, args.flame_params
    ).resolve()
    table = load_flame_state_table(flame_path)
    # Validate flame_index ranges now that the parameter table is available.
    metadata = load_joint_camera_metadata(
        camera_path,
        input_dir=input_dir,
        flame_states=table,
        require_images=True,
    )
    state_index = (
        int(args.alignment_flame_index)
        if args.alignment_flame_index is not None
        else int(metadata.get("alignment_flame_index", 0))
    )
    if not 0 <= state_index < table.num_states:
        raise ValueError(f"Invalid alignment FLAME state {state_index}")

    output_dir = (
        resolve_path(args.output_dir).resolve()
        if args.output_dir is not None
        else input_dir / "flame_alignment_joint"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (
        input_dir,
        camera_path,
        flame_path,
        base.FLAME_MODEL_PATH,
        base.FLAME_LMK_PATH,
        base.FLAME_MESH_PATH,
        base.FLAME_MEDIAPIPE_LMK_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    _, all_frames = base.load_frames(input_dir, camera_path)
    selected_indices = alignment_frame_indices(metadata, state_index)
    alignment_frames = [
        frame for frame in all_frames if frame.index in selected_indices
    ]
    if len(alignment_frames) != len(selected_indices):
        loaded = {frame.index for frame in alignment_frames}
        raise ValueError(
            f"Could not load alignment frame indices {sorted(selected_indices - loaded)}"
        )

    vertices_training, faces, landmarks_training, landmark_indices = (
        build_flame_for_state(table, state_index, args.flame_scale)
    )
    detections, detection_failures = base.detect_landmarks(
        alignment_frames,
        landmark_indices,
        args.min_detection_confidence,
    )
    detected_frames = [
        frame for frame in alignment_frames if frame.index in detections
    ]
    core_frames = [
        frame
        for frame in detected_frames
        if abs(
            base.angular_difference_degrees(
                frame.azimuth_deg, args.front_azimuth
            )
        )
        <= args.core_half_range
    ]
    if len(core_frames) < 4:
        raise RuntimeError(
            f"Only {len(core_frames)} reliable core views detected; need at least 4"
        )

    triangulated, triangulation_errors, positive_depth = base.triangulate_landmarks(
        core_frames,
        detections,
        landmarks_training.shape[0],
    )
    triangulation_median = np.median(triangulation_errors, axis=0)
    reliable = (
        np.isfinite(triangulated).all(axis=1)
        & positive_depth
        & (triangulation_median <= args.triangulation_max_median_px)
    )
    if int(reliable.sum()) < 20:
        raise RuntimeError(
            f"Only {int(reliable.sum())} triangulated landmarks passed checks"
        )

    core_masks = base.build_observation_masks(
        core_frames,
        landmarks_training.shape[0],
        reliable,
        triangulation_errors,
        args.triangulation_max_observation_px,
    )
    initial, parity_report = base.choose_initial_parity(
        args.parity,
        landmarks_training,
        triangulated,
        reliable,
        core_frames,
        detections,
        core_masks,
    )
    stage1, stage1_result = base.optimize_similarity(
        initial,
        landmarks_training,
        core_frames,
        detections,
        core_masks,
        args.huber_px,
        args.max_nfev,
    )

    optimization_frames = list(core_frames)
    optimization_masks = dict(core_masks)
    final = stage1
    stage2_result = None
    if args.refine_profiles:
        core_indices = {frame.index for frame in core_frames}
        candidates = [
            frame for frame in detected_frames if frame.index not in core_indices
        ]
        profile_masks, _ = base.select_profile_observations(
            stage1,
            landmarks_training,
            candidates,
            detections,
            args.profile_max_view_median_px,
            args.profile_max_point_px,
            args.profile_min_points,
        )
        accepted_indices = set(profile_masks)
        optimization_frames.extend(
            frame for frame in candidates if frame.index in accepted_indices
        )
        optimization_masks.update(profile_masks)
        if profile_masks:
            candidate_final, stage2_result = base.optimize_similarity(
                stage1,
                landmarks_training,
                optimization_frames,
                detections,
                optimization_masks,
                args.huber_px,
                args.max_nfev,
            )
            stage1_core = base.aggregate_masked_errors(
                stage1, landmarks_training, core_frames, detections, core_masks
            )
            stage2_core = base.aggregate_masked_errors(
                candidate_final,
                landmarks_training,
                core_frames,
                detections,
                core_masks,
            )
            if np.median(stage2_core) <= np.median(stage1_core) + 0.75:
                final = candidate_final
            else:
                optimization_frames = list(core_frames)
                optimization_masks = dict(core_masks)

    final_core_errors = base.aggregate_masked_errors(
        final,
        landmarks_training,
        core_frames,
        detections,
        core_masks,
    )
    vertices_world = final.transform(vertices_training)
    landmarks_world = final.transform(landmarks_training)
    metrics = base.build_metrics(
        final,
        landmarks_training,
        alignment_frames,
        detections,
        optimization_masks,
    )

    np.savez_compressed(
        output_dir / "alignment.npz",
        facelift_from_training=final.matrix,
        shape=table.shape,
        flame_state_index=np.asarray(state_index, dtype=np.int32),
        flame_state_name=np.asarray(table.state_names[state_index]),
        shape_source=np.asarray(table.shape_source),
        flame_scale=np.asarray(args.flame_scale, dtype=np.float32),
    )
    report = {
        "camera_file": str(camera_path),
        "flame_parameter_file": str(flame_path),
        "alignment_flame_index": state_index,
        "alignment_flame_state": str(table.state_names[state_index]),
        "shape_source": table.shape_source,
        "alignment_observations": len(alignment_frames),
        "detected_observations": len(detected_frames),
        "core_observations": len(core_frames),
        "reliable_landmarks": int(reliable.sum()),
        "selected_parity": parity_report["selected"],
        "scale": final.scale,
        "rotation": final.rotation.tolist(),
        "handedness": final.handedness.tolist(),
        "translation": final.translation.tolist(),
        "matrix": final.matrix.tolist(),
        "core_reprojection_px": base.distribution(final_core_errors),
        "detection_failures": {
            str(index): reason for index, reason in detection_failures.items()
        },
        "metrics": metrics,
    }
    (output_dir / "alignment_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.render:
        base.render_comparisons(
            output_dir,
            vertices_world,
            faces,
            landmarks_world,
            alignment_frames,
            detections,
            metrics,
            args.render_device,
            args.contact_sheet_cell,
        )

    print(f"Alignment observations: {len(alignment_frames)}")
    print(f"Detected alignment views: {len(detected_frames)}/{len(alignment_frames)}")
    print(f"Core fit views: {len(core_frames)}")
    print(f"Reliable triangulated landmarks: {int(reliable.sum())}/{len(reliable)}")
    print(f"Shared shape source: {table.shape_source}")
    print(f"Alignment FLAME state: {state_index} ({table.state_names[state_index]})")
    print(f"Selected parity: {parity_report['selected']}")
    print(f"Scale: {final.scale:.9f}")
    print(f"Translation: {final.translation.tolist()}")
    print(
        "Core reprojection mean/median/p90: "
        f"{np.mean(final_core_errors):.3f}/"
        f"{np.median(final_core_errors):.3f}/"
        f"{np.percentile(final_core_errors, 90):.3f} px"
    )
    print(f"Stage 1 function evaluations: {stage1_result.nfev}")
    if stage2_result is not None:
        print(f"Stage 2 function evaluations: {stage2_result.nfev}")
    print(f"Saved reconstruction transform to: {output_dir / 'alignment.npz'}")


if __name__ == "__main__":
    main()
