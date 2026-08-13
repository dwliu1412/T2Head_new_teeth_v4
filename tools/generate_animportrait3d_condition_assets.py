"""Generate normal+seg ControlNet assets for this repository's FLAME teeth.

The reference face-normal region is exactly the union of this model's ``face``
and ``nose`` masks.  Consequently every face index, eye index, dental index and
segmentation value can be rebuilt solely from the current
``flame_model.flame_teeth.FlameHead`` topology, without reading AnimPortrait3D
at generation or training time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flame_model.flame_teeth import FLAME_VERSION, FlameHead  # noqa: E402


DEFAULT_OUTPUT = (
    PROJECT_ROOT / "ckpts" / "animportrait3d_normal_seg_flame_teeth"
)
REQUIRED_FILES = (
    "face_region_faces.npy",
    "face_region_verts_mask.npy",
    "verts_seg.npy",
    "verts_seg_idxs.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate an existing destination without modifying it",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing complete asset package",
    )
    return parser.parse_args()


def _region(model: FlameHead, name: str) -> np.ndarray:
    return (
        torch.unique(model.mask.get_vid_by_region(name).long())
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )


def _face_region(model: FlameHead) -> tuple[np.ndarray, np.ndarray]:
    vertex_ids = np.union1d(_region(model, "face"), _region(model, "nose"))
    model_faces = (
        torch.as_tensor(model.faces, dtype=torch.long).detach().cpu().numpy()
    )
    vertex_count = int(model.v_template.shape[0])
    dental_count = int(model.num_verts_teeth)
    base_vertex_count = vertex_count - dental_count

    if (
        vertex_ids.size == 0
        or np.unique(vertex_ids).size != vertex_ids.size
        or vertex_ids.min() < 0
        or vertex_ids.max() >= base_vertex_count
    ):
        raise ValueError(
            "The current face/nose masks are not a unique selection of the "
            "base FLAME topology"
        )

    current_global_faces = model_faces[
        np.all(np.isin(model_faces, vertex_ids), axis=1)
    ]
    global_to_local = np.full(base_vertex_count, -1, dtype=np.int64)
    global_to_local[vertex_ids] = np.arange(vertex_ids.size, dtype=np.int64)
    current_local_faces = global_to_local[current_global_faces]
    if np.any(current_local_faces < 0):
        raise AssertionError("Failed to localize current face-region triangles")
    return vertex_ids, current_local_faces


def _expected_current_assets(
    model: FlameHead,
    face_vertex_ids: np.ndarray,
    face_faces: np.ndarray,
) -> tuple[np.ndarray, dict[str, list[int]], dict[str, Any]]:
    vertex_count = int(model.v_template.shape[0])
    base_vertex_count = vertex_count - int(model.num_verts_teeth)
    model_faces = (
        torch.as_tensor(model.faces, dtype=torch.long).detach().cpu().numpy()
    )

    left_eye = _region(model, "left_eyeball")
    right_eye = _region(model, "right_eyeball")
    left_iris = _region(model, "left_iris")
    right_iris = _region(model, "right_iris")
    crowns = _region(model, "teeth_crowns")
    gums = _region(model, "gums")
    teeth = _region(model, "teeth")
    oral = _region(model, "oral_cavity")
    expected_dental = np.arange(base_vertex_count, vertex_count, dtype=np.int64)

    if not np.all(np.isin(left_iris, left_eye)):
        raise ValueError("Current left iris is not a subset of the left eyeball")
    if not np.all(np.isin(right_iris, right_eye)):
        raise ValueError("Current right iris is not a subset of the right eyeball")
    if np.intersect1d(crowns, gums).size:
        raise ValueError("Current teeth_crowns and gums regions overlap")
    if not np.array_equal(np.union1d(crowns, gums), expected_dental):
        raise ValueError("Current crowns/gums do not partition the dentition")
    if not np.array_equal(teeth, expected_dental):
        raise ValueError("Current teeth mask does not equal the dentition")
    if not np.all(np.isin(oral, gums)):
        raise ValueError("Current oral_cavity must be a subset of gums")

    # Match AnimPortrait3D's semantic palette exactly.  Sclera=255, iris=85,
    # enamel=125, while skin, gingiva and the inner oral wall stay zero.
    vertex_seg = np.zeros((1, vertex_count, 3), dtype=np.float32)
    vertex_seg[0, left_eye] = 255.0
    vertex_seg[0, right_eye] = 255.0
    vertex_seg[0, left_iris] = 85.0
    vertex_seg[0, right_iris] = 85.0
    vertex_seg[0, crowns] = 125.0
    seg_indices = {
        "left_eye": left_eye.tolist(),
        "right_eye": right_eye.tolist(),
        "teeth": crowns.tolist(),
    }

    topology_hash = hashlib.sha256()
    for array in (
        np.asarray(model.v_template.detach().cpu(), dtype=np.float32),
        np.asarray(model_faces, dtype=np.int64),
    ):
        topology_hash.update(str(array.shape).encode("ascii"))
        topology_hash.update(array.dtype.str.encode("ascii"))
        topology_hash.update(array.tobytes(order="C"))
    manifest = {
        "schema": "t2head.animportrait3d-normal-seg-assets.v1",
        "flame_version": FLAME_VERSION,
        "topology_sha256": topology_hash.hexdigest(),
        "vertex_count": vertex_count,
        "face_count": int(model_faces.shape[0]),
        "base_vertex_count": base_vertex_count,
        "dental_vertex_count": int(vertex_count - base_vertex_count),
        "face_region_vertex_count": int(face_vertex_ids.size),
        "face_region_face_count": int(face_faces.shape[0]),
        "regions": {
            "left_eye": int(left_eye.size),
            "right_eye": int(right_eye.size),
            "left_iris": int(left_iris.size),
            "right_iris": int(right_iris.size),
            "teeth_crowns": int(crowns.size),
            "gums": int(gums.size),
            "oral_cavity": int(oral.size),
        },
        "semantic_rgb_255": {
            "background_skin_gums_oral": [0, 0, 0],
            "iris": [85, 85, 85],
            "teeth_crowns": [125, 125, 125],
            "sclera": [255, 255, 255],
        },
    }
    return vertex_seg, seg_indices, manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_package(output: Path, model: FlameHead) -> dict[str, Any]:
    missing = [output / name for name in REQUIRED_FILES if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Condition asset package is incomplete; missing: "
            + ", ".join(str(path) for path in missing)
        )
    face_vertex_ids = np.asarray(
        np.load(output / "face_region_verts_mask.npy", allow_pickle=False),
        dtype=np.int64,
    ).reshape(-1)
    face_faces = np.asarray(
        np.load(output / "face_region_faces.npy", allow_pickle=False),
        dtype=np.int64,
    )
    expected_face_vertex_ids, expected_face_faces = _face_region(model)
    if not np.array_equal(face_vertex_ids, expected_face_vertex_ids):
        raise ValueError(
            "face_region_verts_mask.npy is not the current face + nose region"
        )
    if not np.array_equal(face_faces, expected_face_faces):
        raise ValueError(
            "face_region_faces.npy is not the current localized face region"
        )
    expected_seg, expected_indices, expected_manifest = _expected_current_assets(
        model, face_vertex_ids, face_faces
    )
    model_faces = (
        torch.as_tensor(model.faces, dtype=torch.long).detach().cpu().numpy()
    )
    if (
        face_faces.ndim != 2
        or face_faces.shape[1] != 3
        or face_faces.size == 0
        or face_faces.min() < 0
        or face_faces.max() >= face_vertex_ids.size
    ):
        raise ValueError("Generated face_region_faces.npy has invalid indices")
    global_faces = face_vertex_ids[face_faces]
    expected_global_faces = model_faces[
        np.all(np.isin(model_faces, face_vertex_ids), axis=1)
    ]
    if not np.array_equal(global_faces, expected_global_faces):
        raise ValueError("Generated face-region faces do not match current FLAME")

    actual_seg = np.asarray(
        np.load(output / "verts_seg.npy", allow_pickle=False), dtype=np.float32
    )
    if not np.array_equal(actual_seg, expected_seg):
        raise ValueError("verts_seg.npy does not exactly match current regions")
    with (output / "verts_seg_idxs.json").open("r", encoding="utf-8") as file:
        actual_indices = json.load(file)
    if actual_indices != expected_indices:
        raise ValueError(
            "verts_seg_idxs.json does not exactly match current eye/crown regions"
        )

    manifest_path = output / "asset_manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as file:
            actual_manifest = json.load(file)
        for key, value in expected_manifest.items():
            if actual_manifest.get(key) != value:
                raise ValueError(f"asset_manifest.json mismatch for {key!r}")
        expected_hashes = actual_manifest.get("files_sha256")
        actual_hashes = {name: _sha256(output / name) for name in REQUIRED_FILES}
        if expected_hashes != actual_hashes:
            raise ValueError("asset_manifest.json file hashes do not match")
    return expected_manifest


def _write_package(
    output: Path,
    face_vertex_ids: np.ndarray,
    face_faces: np.ndarray,
    vertex_seg: np.ndarray,
    seg_indices: dict[str, list[int]],
    manifest: dict[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "face_region_verts_mask.npy", face_vertex_ids)
    np.save(output / "face_region_faces.npy", face_faces)
    np.save(output / "verts_seg.npy", vertex_seg)
    with (output / "verts_seg_idxs.json").open("w", encoding="utf-8") as file:
        json.dump(seg_indices, file, indent=2)
        file.write("\n")
    manifest = dict(manifest)
    manifest["files_sha256"] = {
        name: _sha256(output / name) for name in REQUIRED_FILES
    }
    with (output / "asset_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    model = FlameHead(shape_params=300, expr_params=100, add_teeth=True)
    model.eval()

    if args.check:
        manifest = _validate_package(output, model)
        print(f"Validated current-topology condition assets: {output}")
        print(json.dumps(manifest, indent=2))
        return

    existing = [output / name for name in REQUIRED_FILES if (output / name).exists()]
    if existing and not args.force:
        if len(existing) == len(REQUIRED_FILES):
            manifest = _validate_package(output, model)
            print(f"Existing condition assets are already valid: {output}")
            print(json.dumps(manifest, indent=2))
            return
        raise FileExistsError(
            "Destination contains a partial asset package; pass --force after "
            f"reviewing it: {output}"
        )

    face_vertex_ids, face_faces = _face_region(model)
    vertex_seg, seg_indices, manifest = _expected_current_assets(
        model, face_vertex_ids, face_faces
    )
    _write_package(
        output,
        face_vertex_ids,
        face_faces,
        vertex_seg,
        seg_indices,
        manifest,
    )
    validated = _validate_package(output, model)
    print(f"Generated current-topology condition assets: {output}")
    print(json.dumps(validated, indent=2))


if __name__ == "__main__":
    main()
