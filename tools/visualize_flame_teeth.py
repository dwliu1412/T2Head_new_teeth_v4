"""Render a multi-view visual check for ``flame_model.flame_teeth``.

The diagnostic sheet uses deliberately different upper/lower tooth colours so
that rigid jaw binding, occlusion, gaps, and accidental intersections are easy
to inspect.  It renders the isolated neutral dentition plus closed- and
open-mouth crops of the full FLAME mesh.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flame_model.flame_teeth import FlameHead  # noqa: E402
from pytorch3d.io import save_obj  # noqa: E402
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


VIEW_AZIMUTHS: Tuple[Tuple[str, float], ...] = (
    ("front", 0.0),
    ("three-quarter", -45.0),
    ("side", -82.0),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "flame_teeth_visual_check.png",
    )
    parser.add_argument(
        "--export-obj",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "flame_teeth_neutral.obj",
    )
    parser.add_argument("--image-size", type=int, default=420)
    parser.add_argument("--expression-index", type=int, default=398)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _pose_inputs(
    device: torch.device,
    expression_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    expression_path = PROJECT_ROOT / "assets" / "open_mouth_exp.npy"
    pose_path = PROJECT_ROOT / "assets" / "open_mouth_pose.npy"
    if expression_path.exists() and pose_path.exists():
        expressions = np.load(expression_path)
        poses = np.load(pose_path)
        count = min(len(expressions), len(poses))
        if count < 1:
            raise RuntimeError("Open-mouth expression assets are empty")
        index = min(max(expression_index, 0), count - 1)
        expression = torch.as_tensor(
            expressions[index], dtype=torch.float32, device=device
        ).reshape(1, -1)
        pose = torch.as_tensor(poses[index], dtype=torch.float32, device=device).reshape(
            1, -1
        )
        return expression, pose

    expression = torch.zeros((1, 100), dtype=torch.float32, device=device)
    pose = torch.zeros((1, 15), dtype=torch.float32, device=device)
    pose[:, 6] = 0.42
    return expression, pose


def _forward(
    model: FlameHead,
    expression: torch.Tensor,
    pose: torch.Tensor,
) -> torch.Tensor:
    batch = expression.shape[0]
    zeros = lambda width: torch.zeros(  # noqa: E731
        (batch, width), dtype=expression.dtype, device=expression.device
    )
    return model(
        shape=zeros(model.n_shape_params),
        expr=expression,
        rotation=pose[:, 0:3],
        neck=pose[:, 3:6],
        jaw=pose[:, 6:9],
        eyes=pose[:, 9:15],
        translation=zeros(3),
        return_landmarks=False,
    )


def _vertex_colours(model: FlameHead, device: torch.device) -> torch.Tensor:
    colours = torch.full(
        (model.v_template.shape[0], 3),
        0.64,
        dtype=torch.float32,
        device=device,
    )
    colours[model.mask.v.teeth_upper] = torch.tensor(
        (0.98, 0.78, 0.24), device=device
    )
    colours[model.mask.v.teeth_lower] = torch.tensor(
        (0.20, 0.76, 0.94), device=device
    )
    if "gums" in model.mask.v.keys():
        colours[model.mask.v.gums] = torch.tensor(
            (0.58, 0.20, 0.25), device=device
        )
    if "oral_cavity" in model.mask.v.keys():
        colours[model.mask.v.oral_cavity] = torch.tensor(
            (0.18, 0.025, 0.035), device=device
        )
    if "lips" in model.mask.v.keys():
        colours[model.mask.v.lips] = torch.tensor((0.56, 0.25, 0.28), device=device)
    return colours


def _render(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    colours: torch.Tensor,
    *,
    azimuth: float,
    target: Sequence[float],
    distance: float,
    field_of_view: float,
    image_size: int,
) -> Image.Image:
    device = vertices.device
    at = torch.tensor((target,), dtype=vertices.dtype, device=device)
    up = torch.tensor(((0.0, 1.0, 0.0),), dtype=vertices.dtype, device=device)
    rotation, translation = look_at_view_transform(
        dist=distance,
        elev=0.0,
        azim=azimuth,
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
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        cull_backfaces=False,
        perspective_correct=True,
    )
    lights = PointLights(
        device=device,
        location=cameras.get_camera_center(),
        ambient_color=((0.55, 0.55, 0.55),),
        diffuse_color=((0.65, 0.65, 0.65),),
        specular_color=((0.18, 0.18, 0.18),),
    )
    materials = Materials(
        device=device,
        ambient_color=((0.80, 0.80, 0.80),),
        diffuse_color=((0.75, 0.75, 0.75),),
        specular_color=((0.12, 0.12, 0.12),),
        shininess=32.0,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials,
            blend_params=BlendParams(background_color=(0.965, 0.965, 0.97)),
        ),
    )
    mesh = Meshes(
        verts=[vertices],
        faces=[faces],
        textures=TexturesVertex(verts_features=[colours]),
    )
    image = renderer(mesh)[0, ..., :3].clamp(0.0, 1.0)
    pixels = (image.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _label_sheet(
    rows: Iterable[Tuple[str, Sequence[Image.Image]]],
    column_labels: Sequence[str],
    tile_size: int,
) -> Image.Image:
    rows = list(rows)
    header = 42
    label_width = 150
    sheet = Image.new(
        "RGB",
        (label_width + tile_size * len(column_labels), header + tile_size * len(rows)),
        (242, 243, 245),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column, label in enumerate(column_labels):
        x = label_width + column * tile_size + tile_size // 2
        draw.text((x, header // 2), label, fill=(30, 33, 38), font=font, anchor="mm")
    for row, (row_label, images) in enumerate(rows):
        y = header + row * tile_size
        draw.text(
            (label_width // 2, y + tile_size // 2),
            row_label,
            fill=(30, 33, 38),
            font=font,
            anchor="mm",
        )
        for column, image in enumerate(images):
            sheet.paste(image, (label_width + column * tile_size, y))
    return sheet


def main() -> None:
    args = _parse_args()
    device = _device(args.device)
    model = FlameHead(shape_params=300, expr_params=100, add_teeth=True).to(device)
    model.eval()

    open_expression, open_pose = _pose_inputs(device, args.expression_index)
    neutral_expression = torch.zeros_like(open_expression)
    neutral_pose = torch.zeros_like(open_pose)
    with torch.inference_mode():
        neutral_vertices = _forward(model, neutral_expression, neutral_pose)[0]
        open_vertices = _forward(model, open_expression, open_pose)[0]

    faces = model.faces.to(device)
    dental_faces = model.mask.f.teeth.to(device)
    colours = _vertex_colours(model, device)
    dental_center = neutral_vertices[model.mask.v.teeth].mean(dim=0)
    neutral_mouth = torch.tensor(
        (0.0, -0.045, 0.030), dtype=torch.float32, device=device
    )
    open_mouth = 0.5 * (
        open_vertices[model.mask.v.teeth_upper].mean(dim=0)
        + open_vertices[model.mask.v.teeth_lower].mean(dim=0)
    )

    isolated = []
    closed = []
    opened = []
    for _, azimuth in VIEW_AZIMUTHS:
        isolated.append(
            _render(
                neutral_vertices,
                faces[dental_faces],
                colours,
                azimuth=azimuth,
                target=dental_center.tolist(),
                distance=0.25,
                field_of_view=22.0,
                image_size=args.image_size,
            )
        )
        closed.append(
            _render(
                neutral_vertices,
                faces,
                colours,
                azimuth=azimuth,
                target=neutral_mouth.tolist(),
                distance=0.25,
                field_of_view=24.0,
                image_size=args.image_size,
            )
        )
        opened.append(
            _render(
                open_vertices,
                faces,
                colours,
                azimuth=azimuth,
                target=open_mouth.tolist(),
                distance=0.27,
                field_of_view=27.0,
                image_size=args.image_size,
            )
        )

    sheet = _label_sheet(
        (
            ("isolated bite", isolated),
            ("closed mouth", closed),
            ("open mouth", opened),
        ),
        tuple(label for label, _ in VIEW_AZIMUTHS),
        args.image_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)

    args.export_obj.parent.mkdir(parents=True, exist_ok=True)
    save_obj(args.export_obj, neutral_vertices.detach().cpu(), faces.detach().cpu())

    print(f"saved visual check: {args.output}")
    print(f"saved neutral mesh: {args.export_obj}")
    print(
        "mesh stats: "
        f"{model.v_template.shape[0]} vertices, {model.faces.shape[0]} faces, "
        f"{model.mask.v.teeth.shape[0]} dental vertices, "
        f"{model.mask.f.teeth.shape[0]} dental faces"
    )


if __name__ == "__main__":
    main()
