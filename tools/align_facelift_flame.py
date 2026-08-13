"""Align the training FLAME mesh to calibrated FaceLift multi-view images.

The FaceLift camera rig is kept fixed.  Only one global similarity transform is
estimated for the shared FLAME mesh:

    X_facelift = scale * rotation @ handedness @ X_training + translation

The fixed ``handedness`` matrix accounts for the reflection introduced by the
training FLAME Y/Z swap.  Shape, expression, pose, eyes, intrinsics, and the
relative camera poses are never optimized.

Default invocation (from the repository root):

    F:\\Anaconda3\\envs\\headstudio\\python.exe tools\\align_facelift_flame.py

The script detects only the views where MediaPipe finds a face, initializes the
global transform from multi-view triangulation + Umeyama, refines it with a
robust reprojection loss, rejects false/profile detections, and writes one
compact reconstruction parameter file containing the FLAME-to-FaceLift matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flame_model.flame import (  # noqa: E402
    FLAME_LMK_PATH,
    FLAME_MEDIAPIPE_LMK_PATH,
    FLAME_MESH_PATH,
    FLAME_MODEL_PATH,
    FlameHead,
)


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "facelift_multiview"
    / "00000001"
)


@dataclass(frozen=True)
class Frame:
    index: int
    image_type: str
    name: str
    image_path: Path
    width: int
    height: int
    K: np.ndarray
    c2w: np.ndarray
    w2c: np.ndarray
    azimuth_deg: float
    elevation_deg: float


@dataclass
class Similarity:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    handedness: np.ndarray

    @property
    def orthogonal(self) -> np.ndarray:
        return self.rotation @ self.handedness

    @property
    def linear(self) -> np.ndarray:
        return self.scale * self.orthogonal

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.linear
        matrix[:3, 3] = self.translation
        return matrix

    def transform(self, points: np.ndarray) -> np.ndarray:
        return points @ self.linear.T + self.translation[None]


def resolve_path(path: Path, base: Path = PROJECT_ROOT) -> Path:
    return path if path.is_absolute() else base / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one global FLAME similarity transform to a fixed calibrated "
            "FaceLift camera rig and render alignment comparisons."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--camera-json", default="cameras.json")
    parser.add_argument("--optim-pkl", default="optim.pkl")
    parser.add_argument("--optim-key", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--flame-scale", type=float, default=-10.0)

    parser.add_argument("--front-azimuth", type=float, default=270.0)
    parser.add_argument(
        "--core-half-range",
        type=float,
        default=30.1,
        help="Use detected render views within this many degrees of the front for initialization.",
    )
    parser.add_argument("--min-detection-confidence", type=float, default=0.7)
    parser.add_argument("--triangulation-max-median-px", type=float, default=8.0)
    parser.add_argument("--triangulation-max-observation-px", type=float, default=14.0)
    parser.add_argument("--huber-px", type=float, default=4.0)
    parser.add_argument("--max-nfev", type=int, default=500)
    parser.add_argument(
        "--parity",
        choices=("auto", "proper", "reflected"),
        default="auto",
        help="Auto compares a proper Sim(3) with the required fixed Y reflection.",
    )
    parser.add_argument(
        "--refine-profiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refine with only residual-consistent landmarks from accepted side views.",
    )
    parser.add_argument("--profile-max-view-median-px", type=float, default=20.0)
    parser.add_argument("--profile-max-point-px", type=float, default=16.0)
    parser.add_argument("--profile-min-points", type=int, default=35)

    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-device", default="cuda:0")
    parser.add_argument("--contact-sheet-cell", type=int, default=224)
    return parser.parse_args()


def resolve_image_path(input_dir: Path, file_path: str) -> Path:
    relative = Path(file_path)
    candidates = [
        relative if relative.is_absolute() else input_dir / relative,
        input_dir / "rgb" / relative.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot resolve image '{file_path}'. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def load_frames(input_dir: Path, camera_path: Path) -> tuple[dict[str, Any], list[Frame]]:
    with camera_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    raw_frames = metadata.get("frames", [])
    if not raw_frames:
        raise ValueError(f"No frames found in {camera_path}")

    frames: list[Frame] = []
    for fallback_index, raw in enumerate(raw_frames):
        width, height = int(raw["w"]), int(raw["h"])
        K = np.array(
            [
                [float(raw["fx"]), 0.0, float(raw["cx"])],
                [0.0, float(raw["fy"]), float(raw["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        c2w = np.asarray(raw["c2w"], dtype=np.float64)
        w2c = np.asarray(raw.get("w2c", np.linalg.inv(c2w)), dtype=np.float64)
        if c2w.shape != (4, 4) or w2c.shape != (4, 4):
            raise ValueError(f"Frame {fallback_index} has non-4x4 camera matrices")
        inverse_error = float(np.abs(c2w @ w2c - np.eye(4)).max())
        if inverse_error > 1e-5:
            raise ValueError(
                f"Frame {fallback_index} c2w/w2c mismatch: max error {inverse_error:.3e}"
            )

        image_path = resolve_image_path(input_dir, raw["file_path"])
        frames.append(
            Frame(
                index=int(raw.get("frame_index", fallback_index)),
                image_type=str(raw.get("image_type", "render")),
                name=image_path.name,
                image_path=image_path,
                width=width,
                height=height,
                K=K,
                c2w=c2w,
                w2c=w2c,
                azimuth_deg=float(
                    raw.get("azimuth_degrees", raw.get("relative_azimuth_deg", 0.0))
                ),
                elevation_deg=float(
                    raw.get("elevation_degrees", raw.get("elevation_deg", 0.0))
                ),
            )
        )
    return metadata, frames


def load_optim_parameters(path: Path, requested_key: str | None) -> tuple[str, dict[str, np.ndarray]]:
    with path.open("rb") as file:
        data = pickle.load(file)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Expected a non-empty dictionary in {path}")
    if requested_key is None:
        if len(data) != 1:
            raise ValueError(
                f"{path} has {len(data)} keys; pass --optim-key explicitly: {list(data)}"
            )
        key = next(iter(data))
    else:
        key = requested_key
    if key not in data or not isinstance(data[key], dict):
        raise KeyError(f"Missing FLAME parameter dictionary for key '{key}' in {path}")

    entry = data[key]
    required = {
        "shapecode": 300,
        "expcode": 100,
        "posecode": 6,
        "eyecode": 6,
    }
    result: dict[str, np.ndarray] = {}
    for name, expected_size in required.items():
        value = np.asarray(entry[name], dtype=np.float32).reshape(-1)
        if value.size < expected_size:
            raise ValueError(
                f"{name} contains {value.size} values, expected at least {expected_size}"
            )
        result[name] = value[:expected_size].copy()
    return key, result


@torch.inference_mode()
def build_training_flame(
    params: dict[str, np.ndarray],
    flame_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = FlameHead(
        shape_params=300,
        expr_params=100,
        include_mask=True,
        add_teeth=True,
    ).cpu().eval()

    zeros_shape = torch.zeros(1, 300, dtype=torch.float32)
    zeros_expr = torch.zeros(1, 100, dtype=torch.float32)
    zeros3 = torch.zeros(1, 3, dtype=torch.float32)
    zeros6 = torch.zeros(1, 6, dtype=torch.float32)
    neutral = model(
        shape=zeros_shape,
        expr=zeros_expr,
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

    shape = torch.from_numpy(params["shapecode"])[None]
    expression = torch.from_numpy(params["expcode"])[None]
    pose = torch.from_numpy(params["posecode"])[None]
    eyes = torch.from_numpy(params["eyecode"])[None]
    vertices = model(
        shape=shape,
        expr=expression,
        rotation=pose[:, :3],
        neck=zeros3,
        jaw=pose[:, 3:6],
        eyes=eyes,
        translation=zeros3,
        zero_centered_at_root_node=False,
        return_landmarks=False,
    ).squeeze(0)

    vertices = (vertices - center) * normalization_scale
    vertices = vertices.clone()
    vertices[:, [1, 2]] = vertices[:, [2, 1]]
    scale_multiplier = 1.1 ** (-float(flame_scale))
    vertices *= scale_multiplier

    faces = model.faces.detach().cpu().numpy().astype(np.int64)
    embedding = np.load(FLAME_MEDIAPIPE_LMK_PATH, allow_pickle=True)
    face_indices = np.asarray(embedding["lmk_face_idx"], dtype=np.int64)
    barycentric = np.asarray(embedding["lmk_b_coords"], dtype=np.float64)
    landmark_indices = np.asarray(embedding["landmark_indices"], dtype=np.int64)
    vertices_np = vertices.detach().cpu().numpy().astype(np.float64)
    landmark_triangles = vertices_np[faces[face_indices]]
    landmarks = (landmark_triangles * barycentric[:, :, None]).sum(axis=1)

    return vertices_np, faces, landmarks, landmark_indices


def detect_landmarks(
    frames: Iterable[Frame],
    landmark_indices: np.ndarray,
    min_detection_confidence: float,
) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    detections: dict[int, np.ndarray] = {}
    failures: dict[int, str] = {}
    max_index = int(landmark_indices.max())
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=float(min_detection_confidence),
    ) as detector:
        for frame in frames:
            image_bgr = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                failures[frame.index] = "image_read_failed"
                continue
            result = detector.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            if not result.multi_face_landmarks:
                failures[frame.index] = "face_not_detected"
                continue
            raw = result.multi_face_landmarks[0].landmark
            if len(raw) <= max_index:
                failures[frame.index] = f"only_{len(raw)}_landmarks"
                continue
            points = np.array(
                [[raw[index].x * frame.width, raw[index].y * frame.height] for index in landmark_indices],
                dtype=np.float64,
            )
            if not np.isfinite(points).all():
                failures[frame.index] = "non_finite_landmarks"
                continue
            detections[frame.index] = points
    return detections, failures


def angular_difference_degrees(value: float, reference: float) -> float:
    return (value - reference + 180.0) % 360.0 - 180.0


def projection_matrix(frame: Frame) -> np.ndarray:
    return frame.K @ frame.w2c[:3, :]


def project_points(points_world: np.ndarray, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    camera = points_world @ frame.w2c[:3, :3].T + frame.w2c[:3, 3]
    z = camera[:, 2]
    projected_h = camera @ frame.K.T
    projected = projected_h[:, :2] / np.clip(projected_h[:, 2:3], 1e-10, None)
    return projected, z


def triangulate_landmarks(
    frames: list[Frame],
    detections: dict[int, np.ndarray],
    landmark_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.full((landmark_count, 3), np.nan, dtype=np.float64)
    errors = np.full((len(frames), landmark_count), np.inf, dtype=np.float64)
    positive_depth = np.zeros(landmark_count, dtype=bool)
    matrices = [projection_matrix(frame) for frame in frames]

    for landmark_index in range(landmark_count):
        rows = []
        for frame, matrix in zip(frames, matrices):
            u, v = detections[frame.index][landmark_index]
            rows.extend([u * matrix[2] - matrix[0], v * matrix[2] - matrix[1]])
        _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64), full_matrices=False)
        homogeneous = vh[-1]
        if abs(homogeneous[3]) < 1e-10:
            continue
        point = homogeneous[:3] / homogeneous[3]
        points[landmark_index] = point
        depths = []
        for view_index, frame in enumerate(frames):
            projected, z = project_points(point[None], frame)
            errors[view_index, landmark_index] = np.linalg.norm(
                projected[0] - detections[frame.index][landmark_index]
            )
            depths.append(float(z[0]))
        positive_depth[landmark_index] = all(depth > 0.0 for depth in depths)
    return points, errors, positive_depth


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 arrays, got {src.shape} and {dst.shape}")
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_centered, dst_centered = src - src_mean, dst - dst_mean
    covariance = dst_centered.T @ src_centered / src.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    variance = float((src_centered * src_centered).sum() / src.shape[0])
    scale = float((singular_values * sign).sum() / max(variance, 1e-12))
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def robust_umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    handedness: np.ndarray,
) -> tuple[Similarity, np.ndarray]:
    adapted = src @ handedness.T
    keep = np.ones(src.shape[0], dtype=bool)
    similarity: Similarity | None = None
    for _ in range(5):
        scale, rotation, translation = umeyama_similarity(adapted[keep], dst[keep])
        similarity = Similarity(scale, rotation, translation, handedness)
        residual = np.linalg.norm(similarity.transform(src) - dst, axis=1)
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median)))
        threshold = max(0.04, median + 3.5 * 1.4826 * max(mad, 1e-6))
        updated = residual <= threshold
        if updated.sum() < 20 or np.array_equal(updated, keep):
            break
        keep = updated
    assert similarity is not None
    return similarity, keep


def pack_similarity(similarity: Similarity) -> np.ndarray:
    return np.concatenate(
        [
            Rotation.from_matrix(similarity.rotation).as_rotvec(),
            np.array([math.log(similarity.scale)], dtype=np.float64),
            similarity.translation,
        ]
    )


def unpack_similarity(parameters: np.ndarray, handedness: np.ndarray) -> Similarity:
    return Similarity(
        scale=float(np.exp(parameters[3])),
        rotation=Rotation.from_rotvec(parameters[:3]).as_matrix(),
        translation=np.asarray(parameters[4:7], dtype=np.float64),
        handedness=handedness,
    )


def build_observation_masks(
    frames: list[Frame],
    landmark_count: int,
    reliable_landmarks: np.ndarray,
    triangulation_errors: np.ndarray,
    max_observation_error: float,
) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    for view_index, frame in enumerate(frames):
        mask = reliable_landmarks.copy()
        mask &= triangulation_errors[view_index] <= max_observation_error
        if mask.shape != (landmark_count,):
            raise AssertionError("Unexpected observation mask shape")
        masks[frame.index] = mask
    return masks


def optimize_similarity(
    initial: Similarity,
    landmarks_training: np.ndarray,
    frames: list[Frame],
    detections: dict[int, np.ndarray],
    masks: dict[int, np.ndarray],
    huber_px: float,
    max_nfev: int,
) -> tuple[Similarity, Any]:
    active = [(frame, masks[frame.index]) for frame in frames if masks[frame.index].any()]
    if not active:
        raise ValueError("No valid landmark observations remain for optimization")

    # Divide every view by sqrt(N_i): each camera contributes comparable weight.
    def residual(parameters: np.ndarray) -> np.ndarray:
        similarity = unpack_similarity(parameters, initial.handedness)
        points_world = similarity.transform(landmarks_training)
        chunks = []
        for frame, mask in active:
            projected, depth = project_points(points_world, frame)
            delta = projected[mask] - detections[frame.index][mask]
            delta = delta / math.sqrt(max(int(mask.sum()), 1))
            invalid_depth = depth[mask] <= 1e-5
            if invalid_depth.any():
                delta[invalid_depth] += 1e3
            chunks.append(delta.reshape(-1))
        return np.concatenate(chunks)

    result = least_squares(
        residual,
        pack_similarity(initial),
        method="trf",
        loss="huber",
        f_scale=float(huber_px) / math.sqrt(105.0),
        x_scale="jac",
        max_nfev=int(max_nfev),
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    return unpack_similarity(result.x, initial.handedness), result


def pixel_errors(
    similarity: Similarity,
    landmarks_training: np.ndarray,
    frame: Frame,
    detected: np.ndarray,
) -> np.ndarray:
    projected, depth = project_points(similarity.transform(landmarks_training), frame)
    errors = np.linalg.norm(projected - detected, axis=1)
    errors[depth <= 0.0] = np.inf
    return errors


def aggregate_masked_errors(
    similarity: Similarity,
    landmarks_training: np.ndarray,
    frames: list[Frame],
    detections: dict[int, np.ndarray],
    masks: dict[int, np.ndarray],
) -> np.ndarray:
    values = []
    for frame in frames:
        if frame.index not in masks:
            continue
        mask = masks[frame.index]
        values.append(pixel_errors(similarity, landmarks_training, frame, detections[frame.index])[mask])
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def choose_initial_parity(
    requested: str,
    landmarks_training: np.ndarray,
    triangulated: np.ndarray,
    reliable: np.ndarray,
    core_frames: list[Frame],
    detections: dict[int, np.ndarray],
    core_masks: dict[int, np.ndarray],
) -> tuple[Similarity, dict[str, Any]]:
    candidates: list[tuple[str, np.ndarray]] = []
    if requested in ("auto", "proper"):
        candidates.append(("proper", np.eye(3, dtype=np.float64)))
    if requested in ("auto", "reflected"):
        candidates.append(("reflected_y", np.diag([1.0, -1.0, 1.0])))

    reports: dict[str, Any] = {}
    solutions: dict[str, Similarity] = {}
    for name, handedness in candidates:
        similarity, keep = robust_umeyama(
            landmarks_training[reliable], triangulated[reliable], handedness
        )
        errors = aggregate_masked_errors(
            similarity, landmarks_training, core_frames, detections, core_masks
        )
        reports[name] = {
            "scale": similarity.scale,
            "orthogonal_determinant": float(np.linalg.det(similarity.orthogonal)),
            "robust_3d_inliers": int(keep.sum()),
            "mean_px": float(np.mean(errors)),
            "median_px": float(np.median(errors)),
            "p90_px": float(np.percentile(errors, 90)),
        }
        solutions[name] = similarity
    selected_name = min(reports, key=lambda name: reports[name]["median_px"])
    reports["selected"] = selected_name
    return solutions[selected_name], reports


def select_profile_observations(
    similarity: Similarity,
    landmarks_training: np.ndarray,
    candidate_frames: list[Frame],
    detections: dict[int, np.ndarray],
    max_view_median: float,
    max_point_error: float,
    min_points: int,
) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    masks: dict[int, np.ndarray] = {}
    reasons: dict[int, str] = {}
    for frame in candidate_frames:
        errors = pixel_errors(
            similarity, landmarks_training, frame, detections[frame.index]
        )
        finite = np.isfinite(errors)
        view_median = float(np.median(errors[finite])) if finite.any() else float("inf")
        if view_median > max_view_median:
            reasons[frame.index] = f"view_median_{view_median:.2f}px"
            continue
        median = view_median
        mad = float(np.median(np.abs(errors[finite] - median))) if finite.any() else 0.0
        adaptive = max(8.0, min(max_point_error, median + 2.5 * 1.4826 * max(mad, 1e-6)))
        mask = finite & (errors <= adaptive)
        if int(mask.sum()) < min_points:
            reasons[frame.index] = f"only_{int(mask.sum())}_consistent_points"
            continue
        masks[frame.index] = mask
        reasons[frame.index] = f"accepted_{int(mask.sum())}_points_threshold_{adaptive:.2f}px"
    return masks, reasons


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
    }


def build_metrics(
    similarity: Similarity,
    landmarks_training: np.ndarray,
    render_frames: list[Frame],
    detections: dict[int, np.ndarray],
    optimization_masks: dict[int, np.ndarray],
) -> dict[str, Any]:
    per_view = []
    all_full = []
    all_selected = []
    for frame in render_frames:
        if frame.index not in detections:
            continue
        errors = pixel_errors(
            similarity, landmarks_training, frame, detections[frame.index]
        )
        all_full.append(errors)
        mask = optimization_masks.get(frame.index)
        selected = errors[mask] if mask is not None else np.empty(0)
        if selected.size:
            all_selected.append(selected)
        per_view.append(
            {
                "frame_index": frame.index,
                "file": frame.name,
                "azimuth_deg": frame.azimuth_deg,
                "elevation_deg": frame.elevation_deg,
                "used_for_optimization": mask is not None,
                "selected_points": int(mask.sum()) if mask is not None else 0,
                "all_105": distribution(errors),
                "selected": distribution(selected),
            }
        )
    return {
        "all_detected_views": distribution(np.concatenate(all_full) if all_full else np.empty(0)),
        "optimization_observations": distribution(
            np.concatenate(all_selected) if all_selected else np.empty(0)
        ),
        "per_view": per_view,
    }


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 27), (20, 20, 20), -1)
    cv2.putText(
        output,
        text,
        (7, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def render_comparisons(
    output_dir: Path,
    vertices_world: np.ndarray,
    faces: np.ndarray,
    landmarks_world: np.ndarray,
    render_frames: list[Frame],
    detections: dict[int, np.ndarray],
    metrics: dict[str, Any],
    device_name: str,
    contact_sheet_cell: int,
) -> None:
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    from pytorch3d.structures import Meshes
    from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_name}, but CUDA is unavailable")
    device = torch.device(device_name)
    mesh = Meshes(
        verts=[torch.from_numpy(vertices_world).float().to(device)],
        faces=[torch.from_numpy(faces).long().to(device)],
    )
    rendered_dir = output_dir / "rendered_flame"
    overlay_dir = output_dir / "overlays"
    comparison_dir = output_dir / "comparisons"
    for directory in (rendered_dir, overlay_dir, comparison_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metric_lookup = {item["frame_index"]: item for item in metrics["per_view"]}
    overlay_paths: list[tuple[Frame, Path]] = []
    with torch.inference_mode():
        for frame in render_frames:
            r = torch.from_numpy(frame.w2c[:3, :3]).float()[None].to(device)
            t = torch.from_numpy(frame.w2c[:3, 3]).float()[None].to(device)
            k = torch.from_numpy(frame.K).float()[None].to(device)
            image_size = torch.tensor([[frame.height, frame.width]], dtype=torch.float32, device=device)
            cameras = cameras_from_opencv_projection(r, t, k, image_size)
            rasterizer = MeshRasterizer(
                cameras=cameras,
                raster_settings=RasterizationSettings(
                    image_size=(frame.height, frame.width),
                    blur_radius=0.0,
                    faces_per_pixel=1,
                    cull_backfaces=False,
                ),
            )
            fragments = rasterizer(mesh)
            face_map = fragments.pix_to_face[0, ..., 0].detach().cpu().numpy()
            depth = fragments.zbuf[0, ..., 0].detach().cpu().numpy()
            mask = face_map >= 0

            flame = np.full((frame.height, frame.width, 3), 255, dtype=np.uint8)
            if mask.any():
                valid_depth = depth[mask]
                low, high = np.percentile(valid_depth, [2.0, 98.0])
                normalized = np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0)
                base = np.array([55.0, 180.0, 225.0], dtype=np.float64)
                shading = 0.72 + 0.28 * (1.0 - normalized)
                flame[mask] = np.clip(base[None] * shading[mask, None], 0, 255).astype(np.uint8)

            target_bgr = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
            if target_bgr is None:
                raise FileNotFoundError(frame.image_path)
            target = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
            overlay = target.copy()
            overlay[mask] = (
                0.58 * target[mask].astype(np.float32)
                + 0.42 * flame[mask].astype(np.float32)
            ).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, (0, 210, 255), 1, cv2.LINE_AA)

            projected, _ = project_points(landmarks_world, frame)
            view_metrics = metric_lookup.get(frame.index)
            accepted_detection = bool(
                view_metrics is not None and view_metrics["used_for_optimization"]
            )
            if frame.index in detections and accepted_detection:
                observed = detections[frame.index]
                for target_point, model_point in zip(observed, projected):
                    if not np.isfinite(model_point).all():
                        continue
                    a = tuple(np.rint(target_point).astype(int))
                    b = tuple(np.rint(model_point).astype(int))
                    cv2.line(overlay, a, b, (255, 210, 0), 1, cv2.LINE_AA)
                    cv2.circle(overlay, a, 1, (40, 220, 40), -1, cv2.LINE_AA)
                    cv2.circle(overlay, b, 1, (235, 55, 55), -1, cv2.LINE_AA)

            error_text = "no landmarks"
            if view_metrics is not None:
                median = view_metrics["all_105"]["median"]
                error_text = f"median {median:.2f}px" if median is not None else "no finite landmarks"
                if not accepted_detection:
                    error_text = "REJECTED detector result | " + error_text
            descriptor = f"az {frame.azimuth_deg:g} el {frame.elevation_deg:g} | {error_text}"
            target_labeled = add_label(target, "GT | " + descriptor)
            flame_labeled = add_label(flame, "aligned training FLAME")
            overlay_labeled = add_label(
                overlay, "overlay: green=detected red=projected cyan=silhouette"
            )
            comparison = np.concatenate([target_labeled, flame_labeled, overlay_labeled], axis=1)

            stem = frame.image_path.stem
            Image.fromarray(flame).save(rendered_dir / f"{stem}.png")
            overlay_path = overlay_dir / f"{stem}.png"
            Image.fromarray(overlay).save(overlay_path)
            Image.fromarray(comparison).save(comparison_dir / f"{stem}.png")
            overlay_paths.append((frame, overlay_path))

    save_contact_sheet(overlay_paths, output_dir / "contact_sheet.png", contact_sheet_cell)
    key_azimuths = {90.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0}
    key_paths = [
        item
        for item in overlay_paths
        if abs(item[0].elevation_deg) < 1e-4
        and any(abs(angular_difference_degrees(item[0].azimuth_deg, az)) < 1e-4 for az in key_azimuths)
    ]
    save_contact_sheet(key_paths, output_dir / "key_views.png", max(contact_sheet_cell, 256), columns=len(key_paths))


def save_contact_sheet(
    items: list[tuple[Frame, Path]],
    output_path: Path,
    cell_size: int,
    columns: int = 6,
) -> None:
    if not items or cell_size <= 0:
        return
    columns = max(1, min(columns, len(items)))
    rows = math.ceil(len(items) / columns)
    label_height = 24
    sheet = Image.new("RGB", (columns * cell_size, rows * (cell_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (frame, path) in enumerate(items):
        with Image.open(path) as image:
            tile = image.convert("RGB").resize((cell_size, cell_size), Image.Resampling.LANCZOS)
        x = (index % columns) * cell_size
        y = (index // columns) * (cell_size + label_height)
        sheet.paste(tile, (x, y))
        draw.rectangle((x, y + cell_size, x + cell_size, y + cell_size + label_height), fill=(20, 20, 20))
        draw.text(
            (x + 5, y + cell_size + 5),
            f"az={frame.azimuth_deg:g} el={frame.elevation_deg:g}",
            fill=(255, 255, 255),
        )
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    input_dir = resolve_path(args.input_dir).resolve()
    camera_path = resolve_path(Path(args.camera_json), input_dir).resolve()
    optim_path = resolve_path(Path(args.optim_pkl), input_dir).resolve()
    output_dir = (
        resolve_path(args.output_dir).resolve()
        if args.output_dir is not None
        else input_dir / "flame_alignment"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    required_paths = [
        input_dir,
        camera_path,
        optim_path,
        FLAME_MODEL_PATH,
        FLAME_LMK_PATH,
        FLAME_MESH_PATH,
        FLAME_MEDIAPIPE_LMK_PATH,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    _, frames = load_frames(input_dir, camera_path)
    render_frames = [frame for frame in frames if frame.image_type == "render"]
    if not render_frames:
        raise ValueError("No frames marked image_type='render' were found")
    _, flame_params = load_optim_parameters(optim_path, args.optim_key)
    vertices_training, faces, landmarks_training, landmark_indices = build_training_flame(
        flame_params,
        args.flame_scale,
    )

    detections, _ = detect_landmarks(
        frames, landmark_indices, args.min_detection_confidence
    )
    detected_render_frames = [frame for frame in render_frames if frame.index in detections]
    core_frames = [
        frame
        for frame in detected_render_frames
        if abs(angular_difference_degrees(frame.azimuth_deg, args.front_azimuth))
        <= args.core_half_range
    ]
    if len(core_frames) < 4:
        raise RuntimeError(
            f"Only {len(core_frames)} reliable core views detected; need at least 4. "
            "Check --front-azimuth, --core-half-range, or the input cameras."
        )

    triangulated, triangulation_errors, positive_depth = triangulate_landmarks(
        core_frames, detections, landmarks_training.shape[0]
    )
    triangulation_median = np.median(triangulation_errors, axis=0)
    reliable = (
        np.isfinite(triangulated).all(axis=1)
        & positive_depth
        & (triangulation_median <= args.triangulation_max_median_px)
    )
    if int(reliable.sum()) < 20:
        raise RuntimeError(
            f"Only {int(reliable.sum())} triangulated landmarks passed consistency checks"
        )
    core_masks = build_observation_masks(
        core_frames,
        landmarks_training.shape[0],
        reliable,
        triangulation_errors,
        args.triangulation_max_observation_px,
    )
    initial, parity_report = choose_initial_parity(
        args.parity,
        landmarks_training,
        triangulated,
        reliable,
        core_frames,
        detections,
        core_masks,
    )
    stage1, stage1_result = optimize_similarity(
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
            frame for frame in detected_render_frames if frame.index not in core_indices
        ]
        profile_masks, _ = select_profile_observations(
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
            candidate_final, stage2_result = optimize_similarity(
                stage1,
                landmarks_training,
                optimization_frames,
                detections,
                optimization_masks,
                args.huber_px,
                args.max_nfev,
            )
            stage1_core_errors = aggregate_masked_errors(
                stage1, landmarks_training, core_frames, detections, core_masks
            )
            stage2_core_errors = aggregate_masked_errors(
                candidate_final, landmarks_training, core_frames, detections, core_masks
            )
            if np.median(stage2_core_errors) <= np.median(stage1_core_errors) + 0.75:
                final = candidate_final
            else:
                optimization_frames = list(core_frames)
                optimization_masks = dict(core_masks)

    final_core_errors = aggregate_masked_errors(
        final, landmarks_training, core_frames, detections, core_masks
    )
    vertices_world = final.transform(vertices_training)
    landmarks_world = final.transform(landmarks_training)
    metrics = build_metrics(
        final, landmarks_training, render_frames, detections, optimization_masks
    )

    np.savez_compressed(
        output_dir / "alignment.npz",
        facelift_from_training=final.matrix,
    )

    if args.render:
        render_comparisons(
            output_dir,
            vertices_world,
            faces,
            landmarks_world,
            render_frames,
            detections,
            metrics,
            args.render_device,
            args.contact_sheet_cell,
        )

    print(f"Detected render views: {len(detected_render_frames)}/{len(render_frames)}")
    print(f"Core fit views: {len(core_frames)}")
    print(f"Reliable triangulated landmarks: {int(reliable.sum())}/{len(reliable)}")
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
