"""Build explicit joint FaceLift multi-view + LivePortrait training metadata.

Example:

    F:\\Anaconda3\\envs\\headstudio\\python.exe \
        tools\\prepare_facelift_joint_dataset.py \
        --input-dir outputs\\facelift_multiview\\00000002

The command does not copy images.  It points every observation at ``all/`` and
writes two new files next to the original inputs:

* ``cameras_joint.json``
* ``flame_params_joint.npz``
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.facelift_joint_dataset import (  # noqa: E402
    SCHEMA_NAME,
    SCHEMA_VERSION,
    load_flame_state_table,
    load_joint_camera_metadata,
)


DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "facelift_multiview" / "00000002"
FRAME_PATTERN = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)


def resolve_path(path: Path, base: Path = PROJECT_ROOT) -> Path:
    return path if path.is_absolute() else base / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--camera-input", default="cameras.json")
    parser.add_argument("--optim-pkl", default="optim.pkl")
    parser.add_argument("--optim-key", default=None)
    parser.add_argument("--live-flame", default="flame_results.npy")
    parser.add_argument("--image-dir", default="all")
    parser.add_argument("--camera-output", default="cameras_joint.json")
    parser.add_argument("--flame-output", default="flame_params_joint.npz")
    parser.add_argument("--front-image", default="elev_0_azim_270.png")
    parser.add_argument(
        "--shape-source",
        choices=("optim", "liveportrait-first", "liveportrait-mean"),
        default="optim",
        help=(
            "One identity shape is shared by every observation. 'optim' is the "
            "default because it is fitted to the image that generated the "
            "FaceLift multi-view set."
        ),
    )
    parser.add_argument(
        "--live-pose-layout",
        choices=("jaw-eyes", "global-neck-jaw"),
        default="jaw-eyes",
        help=(
            "Meaning of the 9-D flame_results pose array. The tracker output in "
            "00000002 is jaw(3)+left/right eyes(6)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing joint metadata files.",
    )
    return parser.parse_args()


def _resolve_child(input_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else input_dir / path


def _vector(entry: dict[str, Any], name: str, size: int) -> np.ndarray:
    if name not in entry:
        raise KeyError(f"optim.pkl entry is missing '{name}'")
    value = np.asarray(entry[name], dtype=np.float32).reshape(-1)
    if value.size < size:
        raise ValueError(f"{name} has {value.size} values; expected at least {size}")
    value = value[:size]
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return value.copy()


def load_optim(path: Path, requested_key: str | None) -> tuple[str, dict[str, np.ndarray]]:
    with path.open("rb") as file:
        entries = pickle.load(file)
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"Expected a non-empty dictionary in {path}")
    if requested_key is None:
        if len(entries) != 1:
            raise ValueError(
                f"{path} contains {len(entries)} entries; pass --optim-key from "
                f"{list(entries)}"
            )
        key = next(iter(entries))
    else:
        key = requested_key
    if key not in entries or not isinstance(entries[key], dict):
        raise KeyError(f"No FLAME parameter dictionary for optim key '{key}'")
    entry = entries[key]
    return key, {
        "shape": _vector(entry, "shapecode", 300),
        "expression": _vector(entry, "expcode", 100),
        "pose": _vector(entry, "posecode", 6),
        "eyes": _vector(entry, "eyecode", 6),
    }


def _matrix(value: Any, name: str, rows: int, columns: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != rows or array.shape[1] < columns:
        raise ValueError(
            f"flame_results['{name}'] must have shape ({rows}, >= {columns}), "
            f"got {array.shape}"
        )
    array = np.ascontiguousarray(array[:, :columns])
    if not np.isfinite(array).all():
        raise ValueError(f"flame_results['{name}'] contains NaN or infinity")
    return array


def load_live_flame(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    if not isinstance(raw, np.ndarray) or raw.shape != ():
        raise ValueError(f"Expected a scalar object array containing a dictionary in {path}")
    data = raw.item()
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dictionary in {path}, got {type(data).__name__}")
    for key in ("shape", "exp", "pose"):
        if key not in data:
            raise KeyError(f"{path} is missing '{key}'")
    shape_raw = np.asarray(data["shape"])
    if shape_raw.ndim != 2:
        raise ValueError(f"flame_results['shape'] must be 2-D, got {shape_raw.shape}")
    count = int(shape_raw.shape[0])
    if count == 0:
        raise ValueError("flame_results contains no frames")
    return {
        "shape": _matrix(data["shape"], "shape", count, 300),
        "expression": _matrix(data["exp"], "exp", count, 100),
        "pose": _matrix(data["pose"], "pose", count, 9),
    }


def choose_shape(
    optim: dict[str, np.ndarray],
    live: dict[str, np.ndarray],
    source: str,
) -> np.ndarray:
    if source == "optim":
        value = optim["shape"]
    elif source == "liveportrait-first":
        value = live["shape"][0]
    elif source == "liveportrait-mean":
        value = live["shape"].mean(axis=0)
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(source)
    return np.asarray(value, dtype=np.float32).reshape(1, 300)


def build_flame_arrays(
    optim: dict[str, np.ndarray],
    live: dict[str, np.ndarray],
    live_names: list[str],
    shape_source: str,
    live_pose_layout: str,
) -> dict[str, np.ndarray]:
    count = int(live["expression"].shape[0])
    if len(live_names) != count:
        raise ValueError(f"Found {len(live_names)} images for {count} LivePortrait states")

    expression = np.zeros((count + 1, 100), dtype=np.float32)
    global_orient = np.zeros((count + 1, 3), dtype=np.float32)
    neck_pose = np.zeros((count + 1, 3), dtype=np.float32)
    jaw_pose = np.zeros((count + 1, 3), dtype=np.float32)
    eyes = np.zeros((count + 1, 6), dtype=np.float32)

    expression[0] = optim["expression"]
    global_orient[0] = optim["pose"][:3]
    jaw_pose[0] = optim["pose"][3:6]
    eyes[0] = optim["eyes"]
    expression[1:] = live["expression"]

    if live_pose_layout == "jaw-eyes":
        jaw_pose[1:] = live["pose"][:, :3]
        eyes[1:] = live["pose"][:, 3:9]
    elif live_pose_layout == "global-neck-jaw":
        global_orient[1:] = live["pose"][:, :3]
        neck_pose[1:] = live["pose"][:, 3:6]
        jaw_pose[1:] = live["pose"][:, 6:9]
        # This layout has no eye estimates. Reuse the static fit explicitly.
        eyes[1:] = optim["eyes"][None]
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(live_pose_layout)

    return {
        "schema_name": np.asarray(SCHEMA_NAME),
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
        "shape": choose_shape(optim, live, shape_source),
        "expression": expression,
        "global_orient": global_orient,
        "neck_pose": neck_pose,
        "jaw_pose": jaw_pose,
        "eyes": eyes,
        "state_names": np.asarray(["multiview_shared", *live_names]),
        "state_sources": np.asarray(["multiview", *(["liveportrait"] * count)]),
        "shape_source": np.asarray(shape_source),
        "live_pose_layout": np.asarray(live_pose_layout),
    }


def find_live_images(image_dir: Path, expected_count: int) -> list[Path]:
    indexed: dict[int, Path] = {}
    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        match = FRAME_PATTERN.match(path.name)
        if match is None:
            continue
        index = int(match.group(1))
        if index in indexed:
            raise ValueError(f"Duplicate LivePortrait frame index {index} in {image_dir}")
        indexed[index] = path
    expected = set(range(expected_count))
    actual = set(indexed)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "LivePortrait filenames must map exactly to flame_results rows; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return [indexed[index] for index in range(expected_count)]


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"Image must be inside input directory: {path}") from error


def _validate_image(path: Path, frame: dict[str, Any]) -> None:
    with Image.open(path) as image:
        size = image.size
    expected = (int(frame["w"]), int(frame["h"]))
    if size != expected:
        raise ValueError(f"Image {path} has size {size}; camera declares {expected}")


def build_camera_metadata(
    input_dir: Path,
    camera_input: Path,
    image_dir: Path,
    live_images: list[Path],
    front_image: str,
    flame_output: Path,
    optim_key: str,
    shape_source: str,
    live_pose_layout: str,
) -> dict[str, Any]:
    with camera_input.open("r", encoding="utf-8") as file:
        original = json.load(file)
    original_frames = original.get("frames")
    if not isinstance(original_frames, list) or not original_frames:
        raise ValueError(f"No frames found in {camera_input}")

    static_frames: list[dict[str, Any]] = []
    front: dict[str, Any] | None = None
    seen_indices: set[int] = set()
    for fallback_index, source in enumerate(original_frames):
        frame = copy.deepcopy(source)
        frame_index = int(frame.get("frame_index", fallback_index))
        if frame_index in seen_indices:
            raise ValueError(f"Duplicate original frame_index {frame_index}")
        seen_indices.add(frame_index)
        image_name = Path(str(frame["file_path"])).name
        target_image = image_dir / image_name
        if not target_image.is_file():
            raise FileNotFoundError(
                f"Static image '{image_name}' is missing from joint image directory {image_dir}"
            )
        _validate_image(target_image, frame)
        frame.update(
            {
                "frame_index": frame_index,
                "file_path": _relative_posix(target_image, input_dir),
                "source": "multiview",
                "flame_index": 0,
                "use_for_alignment": True,
                "source_frame_index": frame_index,
            }
        )
        static_frames.append(frame)
        if image_name == front_image:
            front = frame

    if front is None:
        raise ValueError(
            f"Could not find front camera '{front_image}' in {camera_input}"
        )

    next_index = max(seen_indices) + 1
    live_frames: list[dict[str, Any]] = []
    for live_index, image_path in enumerate(live_images):
        frame = copy.deepcopy(front)
        _validate_image(image_path, frame)
        frame.update(
            {
                "frame_index": next_index + live_index,
                "image_type": "liveportrait",
                "file_path": _relative_posix(image_path, input_dir),
                "source": "liveportrait",
                "flame_index": live_index + 1,
                "use_for_alignment": False,
                "sequence_frame_index": live_index,
                "camera_reference_file": front_image,
            }
        )
        frame.pop("source_frame_index", None)
        live_frames.append(frame)

    result = copy.deepcopy(original)
    result.update(
        {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "flame_parameter_file": _relative_posix(flame_output, input_dir),
            "alignment_flame_index": 0,
            "shape_source": shape_source,
            "live_pose_layout": live_pose_layout,
            "optim_key": optim_key,
            "front_camera_reference": front_image,
            "num_images": len(static_frames) + len(live_frames),
            "num_flame_states": len(live_frames) + 1,
            "source_counts": {
                "multiview": len(static_frames),
                "liveportrait": len(live_frames),
            },
            "frames": [*static_frames, *live_frames],
        }
    )
    return result


def main() -> None:
    args = parse_args()
    input_dir = resolve_path(args.input_dir).resolve()
    camera_input = _resolve_child(input_dir, args.camera_input).resolve()
    optim_path = _resolve_child(input_dir, args.optim_pkl).resolve()
    live_path = _resolve_child(input_dir, args.live_flame).resolve()
    image_dir = _resolve_child(input_dir, args.image_dir).resolve()
    camera_output = _resolve_child(input_dir, args.camera_output).resolve()
    flame_output = _resolve_child(input_dir, args.flame_output).resolve()

    for path in (input_dir, camera_input, optim_path, live_path, image_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (camera_output, flame_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")

    optim_key, optim = load_optim(optim_path, args.optim_key)
    live = load_live_flame(live_path)
    live_images = find_live_images(image_dir, int(live["expression"].shape[0]))
    live_names = [path.stem for path in live_images]
    flame_arrays = build_flame_arrays(
        optim,
        live,
        live_names,
        args.shape_source,
        args.live_pose_layout,
    )
    camera_metadata = build_camera_metadata(
        input_dir,
        camera_input,
        image_dir,
        live_images,
        args.front_image,
        flame_output,
        optim_key,
        args.shape_source,
        args.live_pose_layout,
    )

    flame_output.parent.mkdir(parents=True, exist_ok=True)
    camera_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(flame_output, **flame_arrays)
    camera_output.write_text(
        json.dumps(camera_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Re-read what was written so truncated arrays, broken paths, or accidental
    # schema drift fail in the preparation command rather than during training.
    states = load_flame_state_table(flame_output)
    load_joint_camera_metadata(
        camera_output,
        input_dir=input_dir,
        flame_states=states,
        require_images=True,
    )

    live_shape_std = live["shape"].std(axis=0)
    print(f"Wrote {camera_output}")
    print(f"Wrote {flame_output}")
    print(
        f"Observations: {camera_metadata['num_images']} "
        f"({camera_metadata['source_counts']['multiview']} multiview + "
        f"{camera_metadata['source_counts']['liveportrait']} LivePortrait)"
    )
    print(f"FLAME states: {states.num_states}; shared shape source: {states.shape_source}")
    print(
        "LivePortrait per-frame shape variation (diagnostic only): "
        f"mean std={float(live_shape_std.mean()):.6f}, "
        f"max std={float(live_shape_std.max()):.6f}"
    )
    print(f"LivePortrait pose layout: {states.live_pose_layout}")


if __name__ == "__main__":
    main()
