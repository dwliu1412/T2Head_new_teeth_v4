"""Render ``assets/test`` pose/expression as standalone FLAME videos.

Outputs a natural-colour frontal video and a four-panel diagnostic video with
mouth and isolated-dentition views.  The renderer imports only the standalone
``flame_model.flame_teeth`` model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flame_model.flame_teeth import FlameHead  # noqa: E402
from pytorch3d.renderer import (  # noqa: E402
    BlendParams,
    FoVPerspectiveCameras,
    HardPhongShader,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pose",
        type=Path,
        default=PROJECT_ROOT / "assets" / "test" / "pose.npy",
    )
    parser.add_argument(
        "--expression",
        type=Path,
        default=PROJECT_ROOT / "assets" / "test" / "exp.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "flame_teeth_test_video",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--stabilize-head",
        action="store_true",
        help="Zero global and neck rotations while retaining jaw/eye pose.",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sequence(
    pose_path: Path,
    expression_path: Path,
    max_frames: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    pose = np.load(pose_path)
    expression = np.load(expression_path)
    if pose.ndim != 2 or pose.shape[1] != 15:
        raise ValueError(f"Expected pose shape (frames, 15), got {pose.shape}")
    if expression.ndim != 2 or expression.shape[1] != 100:
        raise ValueError(
            f"Expected expression shape (frames, 100), got {expression.shape}"
        )
    if pose.shape[0] != expression.shape[0] or pose.shape[0] == 0:
        raise ValueError(
            "Pose and expression must contain the same non-zero number of frames"
        )
    if not np.isfinite(pose).all() or not np.isfinite(expression).all():
        raise ValueError("Pose and expression must contain only finite values")
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("--max-frames must be positive")
        pose = pose[:max_frames]
        expression = expression[:max_frames]
    return pose.astype(np.float32, copy=False), expression.astype(np.float32, copy=False)


def _vertex_colours(
    model: FlameHead,
    device: torch.device,
    *,
    diagnostic: bool,
) -> torch.Tensor:
    colours = torch.full(
        (model.v_template.shape[0], 3),
        0.66,
        dtype=torch.float32,
        device=device,
    )
    upper_colour = (0.98, 0.76, 0.20) if diagnostic else (0.93, 0.90, 0.78)
    lower_colour = (0.18, 0.76, 0.96) if diagnostic else (0.93, 0.90, 0.78)
    colours[model.mask.v.teeth_upper] = torch.tensor(upper_colour, device=device)
    colours[model.mask.v.teeth_lower] = torch.tensor(lower_colour, device=device)
    if "gums" in model.mask.v.keys():
        gum_colour = (0.68, 0.16, 0.22) if diagnostic else (0.58, 0.20, 0.25)
        colours[model.mask.v.gums] = torch.tensor(gum_colour, device=device)
    if "oral_cavity" in model.mask.v.keys():
        colours[model.mask.v.oral_cavity] = torch.tensor(
            (0.13, 0.015, 0.025), device=device
        )
    if "lips" in model.mask.v.keys():
        colours[model.mask.v.lips] = torch.tensor((0.55, 0.24, 0.28), device=device)
    return colours


def _renderer(
    *,
    batch_size: int,
    device: torch.device,
    image_size: int,
    azimuth: float,
    target: Sequence[float],
    distance: float,
    field_of_view: float,
) -> MeshRenderer:
    at = torch.tensor((target,), dtype=torch.float32, device=device).expand(
        batch_size, -1
    )
    up = torch.tensor(((0.0, 1.0, 0.0),), device=device).expand(batch_size, -1)
    rotation, translation = look_at_view_transform(
        dist=torch.full((batch_size,), distance, device=device),
        elev=torch.zeros(batch_size, device=device),
        azim=torch.full((batch_size,), azimuth, device=device),
        at=at,
        up=up,
        device=device,
    )
    cameras = FoVPerspectiveCameras(
        device=device,
        R=rotation,
        T=translation,
        fov=field_of_view,
        znear=0.01,
        zfar=10.0,
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=False,
            perspective_correct=True,
        ),
    )
    lights = PointLights(
        device=device,
        location=cameras.get_camera_center(),
        ambient_color=((0.55, 0.55, 0.55),) * batch_size,
        diffuse_color=((0.65, 0.65, 0.65),) * batch_size,
        specular_color=((0.16, 0.16, 0.16),) * batch_size,
    )
    materials = Materials(
        device=device,
        ambient_color=((0.80, 0.80, 0.80),),
        diffuse_color=((0.75, 0.75, 0.75),),
        specular_color=((0.10, 0.10, 0.10),),
        shininess=28.0,
    )
    return MeshRenderer(
        rasterizer=rasterizer,
        shader=HardPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials,
            blend_params=BlendParams(background_color=(0.965, 0.965, 0.97)),
        ),
    )


def _render_batch(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    colours: torch.Tensor,
    *,
    image_size: int,
    azimuth: float,
    target: Sequence[float],
    distance: float,
    field_of_view: float,
) -> np.ndarray:
    batch_size = vertices.shape[0]
    renderer = _renderer(
        batch_size=batch_size,
        device=vertices.device,
        image_size=image_size,
        azimuth=azimuth,
        target=target,
        distance=distance,
        field_of_view=field_of_view,
    )
    meshes = Meshes(
        verts=vertices,
        faces=faces.unsqueeze(0).expand(batch_size, -1, -1),
        textures=TexturesVertex(
            verts_features=colours.unsqueeze(0).expand(batch_size, -1, -1)
        ),
    )
    images = renderer(meshes)[..., :3].clamp(0.0, 1.0)
    return (images.detach().cpu().numpy() * 255.0).round().astype(np.uint8)


def _label(image: np.ndarray, text: str) -> Image.Image:
    result = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.rectangle((8, 8, 18 + width, 18 + height), fill=(242, 243, 245))
    draw.text((13, 13), text, font=font, fill=(24, 27, 31))
    return result


def _diagnostic_frame(
    full: np.ndarray,
    mouth: np.ndarray,
    left_teeth: np.ndarray,
    right_teeth: np.ndarray,
    *,
    frame_index: int,
    jaw_x: float,
) -> np.ndarray:
    size = full.shape[0]
    sheet = Image.new("RGB", (2 * size, 2 * size), (242, 243, 245))
    tiles = (
        _label(full, f"full front | frame {frame_index:03d}"),
        _label(mouth, f"mouth crop | jaw.x {jaw_x:.3f}"),
        _label(left_teeth, "dentition left 3/4"),
        _label(right_teeth, "dentition right 3/4"),
    )
    sheet.paste(tiles[0], (0, 0))
    sheet.paste(tiles[1], (size, 0))
    sheet.paste(tiles[2], (0, size))
    sheet.paste(tiles[3], (size, size))
    return np.asarray(sheet)


def _write_contact_sheet(
    frames: Dict[int, np.ndarray],
    output_path: Path,
) -> None:
    ordered = sorted(frames.items())
    if not ordered:
        return
    columns = 4
    thumb = 256
    rows = math.ceil(len(ordered) / columns)
    sheet = Image.new("RGB", (columns * thumb, rows * thumb), (242, 243, 245))
    for item, (frame_index, pixels) in enumerate(ordered):
        image = Image.fromarray(pixels, mode="RGB")
        image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        x = (item % columns) * thumb
        y = (item // columns) * thumb
        sheet.paste(image, (x, y))
        draw = ImageDraw.Draw(sheet)
        draw.text((x + 6, y + 6), f"frame {frame_index}", fill=(20, 22, 25))
    sheet.save(output_path)


def _writer(path: Path, fps: int):
    return imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=16,
        ffmpeg_params=["-movflags", "+faststart"],
    )


def main() -> None:
    args = _parse_args()
    if args.fps < 1 or args.image_size < 128 or args.batch_size < 1:
        raise ValueError("fps, image-size, and batch-size must be positive")
    pose_np, expression_np = _load_sequence(
        args.pose, args.expression, args.max_frames
    )
    device = _device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    keyframe_dir = args.output_dir / "keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    model = FlameHead(shape_params=300, expr_params=100, add_teeth=True).to(device)
    model.eval()
    faces = model.faces.to(device)
    dental_faces = faces[model.mask.f.teeth]
    natural_colours = _vertex_colours(model, device, diagnostic=False)
    diagnostic_colours = _vertex_colours(model, device, diagnostic=True)

    with torch.inference_mode():
        zeros_shape = torch.zeros((1, 300), device=device)
        zeros_expression = torch.zeros((1, 100), device=device)
        zeros_pose = torch.zeros((1, 3), device=device)
        neutral = model(
            zeros_shape,
            zeros_expression,
            zeros_pose,
            zeros_pose,
            zeros_pose,
            torch.zeros((1, 6), device=device),
            zeros_pose,
            return_landmarks=False,
        )[0]
    dental_target = neutral[model.mask.v.teeth].mean(dim=0).tolist()
    full_target = (0.0, -0.030, -0.030)
    mouth_target = (0.0, -0.048, 0.028)

    frame_count = pose_np.shape[0]
    selected = set(np.linspace(0, frame_count - 1, min(12, frame_count)).round().astype(int))
    selected.update(index for index in (14, 16, 21, 25) if index < frame_count)
    keyframes: Dict[int, np.ndarray] = {}
    front_path = args.output_dir / "front.mp4"
    diagnostic_path = args.output_dir / "diagnostic.mp4"
    front_writer = _writer(front_path, args.fps)
    diagnostic_writer = _writer(diagnostic_path, args.fps)

    try:
        for start in range(0, frame_count, args.batch_size):
            end = min(start + args.batch_size, frame_count)
            pose = torch.from_numpy(pose_np[start:end]).to(device)
            expression = torch.from_numpy(expression_np[start:end]).to(device)
            if args.stabilize_head:
                pose = pose.clone()
                pose[:, 0:6] = 0.0
            batch = end - start
            with torch.inference_mode():
                vertices = model(
                    shape=torch.zeros((batch, 300), device=device),
                    expr=expression,
                    rotation=pose[:, 0:3],
                    neck=pose[:, 3:6],
                    jaw=pose[:, 6:9],
                    eyes=pose[:, 9:15],
                    translation=torch.zeros((batch, 3), device=device),
                    return_landmarks=False,
                )
                full = _render_batch(
                    vertices,
                    faces,
                    natural_colours,
                    image_size=args.image_size,
                    azimuth=0.0,
                    target=full_target,
                    distance=0.70,
                    field_of_view=28.0,
                )
                mouth = _render_batch(
                    vertices,
                    faces,
                    diagnostic_colours,
                    image_size=args.image_size,
                    azimuth=0.0,
                    target=mouth_target,
                    distance=0.27,
                    field_of_view=27.0,
                )
                left_teeth = _render_batch(
                    vertices,
                    dental_faces,
                    diagnostic_colours,
                    image_size=args.image_size,
                    azimuth=-45.0,
                    target=dental_target,
                    distance=0.30,
                    field_of_view=26.0,
                )
                right_teeth = _render_batch(
                    vertices,
                    dental_faces,
                    diagnostic_colours,
                    image_size=args.image_size,
                    azimuth=45.0,
                    target=dental_target,
                    distance=0.30,
                    field_of_view=26.0,
                )

            for local_index in range(batch):
                frame_index = start + local_index
                diagnostic = _diagnostic_frame(
                    full[local_index],
                    mouth[local_index],
                    left_teeth[local_index],
                    right_teeth[local_index],
                    frame_index=frame_index,
                    jaw_x=float(pose_np[frame_index, 6]),
                )
                front_writer.append_data(full[local_index])
                diagnostic_writer.append_data(diagnostic)
                if frame_index in selected:
                    keyframes[frame_index] = diagnostic
                    Image.fromarray(diagnostic, mode="RGB").save(
                        keyframe_dir / f"{frame_index:06d}.png"
                    )
            print(f"rendered frames {start:03d}-{end - 1:03d} / {frame_count - 1:03d}")
    finally:
        front_writer.close()
        diagnostic_writer.close()

    contact_sheet = args.output_dir / "contact_sheet.png"
    _write_contact_sheet(keyframes, contact_sheet)
    metadata = {
        "pose": str(args.pose.resolve()),
        "expression": str(args.expression.resolve()),
        "pose_sha256": _sha256(args.pose),
        "expression_sha256": _sha256(args.expression),
        "pose_shape": list(pose_np.shape),
        "expression_shape": list(expression_np.shape),
        "pose_layout": {
            "global": [0, 3],
            "neck": [3, 6],
            "jaw": [6, 9],
            "left_eye": [9, 12],
            "right_eye": [12, 15],
        },
        "frames": frame_count,
        "fps": args.fps,
        "duration_seconds": frame_count / args.fps,
        "stabilize_head": bool(args.stabilize_head),
        "mesh": {
            "vertices": int(model.v_template.shape[0]),
            "faces": int(model.faces.shape[0]),
            "dental_vertices": int(model.mask.v.teeth.numel()),
            "dental_faces": int(model.mask.f.teeth.numel()),
            "crown_vertices": int(model.mask.v.teeth_crowns.numel()),
            "crown_faces": int(model.mask.f.teeth_crowns.numel()),
            "gum_vertices": int(model.mask.v.gums.numel()),
            "gum_faces": int(model.mask.f.gums.numel()),
            "oral_cavity_vertices": (
                int(model.mask.v.oral_cavity.numel())
                if "oral_cavity" in model.mask.v.keys()
                else 0
            ),
            "oral_cavity_faces": (
                int(model.mask.f.oral_cavity.numel())
                if "oral_cavity" in model.mask.f.keys()
                else 0
            ),
        },
        "outputs": {
            "front": str(front_path.resolve()),
            "diagnostic": str(diagnostic_path.resolve()),
            "contact_sheet": str(contact_sheet.resolve()),
        },
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"saved: {front_path}")
    print(f"saved: {diagnostic_path}")
    print(f"saved: {contact_sheet}")
    print(f"saved: {metadata_path}")


if __name__ == "__main__":
    main()
