"""Shared schema and validation for FaceLift + LivePortrait reconstruction data.

The joint dataset deliberately separates observations from FLAME states:

* ``cameras_joint.json`` contains one record per RGB observation and maps it
  to a FLAME state through ``flame_index``.
* ``flame_params_joint.npz`` contains one identity shape and one row of
  expression/articulation parameters per FLAME state.

Keeping this mapping explicit prevents camera/image order from silently being
used as a pose association.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_NAME = "facelift-liveportrait-joint"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FlameStateTable:
    """Validated, array-backed FLAME state table."""

    shape: np.ndarray
    expression: np.ndarray
    global_orient: np.ndarray
    neck_pose: np.ndarray
    jaw_pose: np.ndarray
    eyes: np.ndarray
    state_names: np.ndarray
    state_sources: np.ndarray
    shape_source: str
    live_pose_layout: str

    @property
    def num_states(self) -> int:
        return int(self.expression.shape[0])

    def state(self, index: int) -> dict[str, np.ndarray]:
        if not 0 <= int(index) < self.num_states:
            raise IndexError(
                f"FLAME state index {index} is outside [0, {self.num_states})"
            )
        index = int(index)
        return {
            "shape": self.shape,
            "expression": self.expression[index : index + 1],
            "global_orient": self.global_orient[index : index + 1],
            "neck_pose": self.neck_pose[index : index + 1],
            "jaw_pose": self.jaw_pose[index : index + 1],
            "eyes": self.eyes[index : index + 1],
        }


def _scalar_string(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain one string, got shape {array.shape}")
    return str(array.reshape(-1)[0])


def _float_array(
    archive: Any,
    name: str,
    expected_shape: tuple[int | None, ...],
) -> np.ndarray:
    if name not in archive:
        raise KeyError(f"Missing array '{name}' in joint FLAME parameter file")
    value = np.asarray(archive[name], dtype=np.float32)
    if value.ndim != len(expected_shape):
        raise ValueError(
            f"{name} must have {len(expected_shape)} dimensions, got {value.shape}"
        )
    for axis, expected in enumerate(expected_shape):
        if expected is not None and value.shape[axis] != expected:
            raise ValueError(
                f"{name} axis {axis} must have size {expected}, got {value.shape}"
            )
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(value)


def load_flame_state_table(path: str | Path) -> FlameStateTable:
    """Load and strictly validate ``flame_params_joint.npz``."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        schema_name = _scalar_string(archive["schema_name"], "schema_name")
        schema_version = int(np.asarray(archive["schema_version"]).reshape(-1)[0])
        if schema_name != SCHEMA_NAME:
            raise ValueError(
                f"Unsupported FLAME schema '{schema_name}' in {path}; "
                f"expected '{SCHEMA_NAME}'"
            )
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported FLAME schema version {schema_version} in {path}; "
                f"expected {SCHEMA_VERSION}"
            )

        shape = _float_array(archive, "shape", (1, 300))
        expression = _float_array(archive, "expression", (None, 100))
        count = int(expression.shape[0])
        if count == 0:
            raise ValueError("Joint FLAME parameter file contains no states")
        global_orient = _float_array(archive, "global_orient", (count, 3))
        neck_pose = _float_array(archive, "neck_pose", (count, 3))
        jaw_pose = _float_array(archive, "jaw_pose", (count, 3))
        eyes = _float_array(archive, "eyes", (count, 6))

        state_names = np.asarray(archive["state_names"]).astype(str)
        state_sources = np.asarray(archive["state_sources"]).astype(str)
        if state_names.shape != (count,):
            raise ValueError(
                f"state_names must have shape ({count},), got {state_names.shape}"
            )
        if state_sources.shape != (count,):
            raise ValueError(
                f"state_sources must have shape ({count},), got {state_sources.shape}"
            )
        if len(set(state_names.tolist())) != count:
            raise ValueError("state_names must be unique")

        shape_source = _scalar_string(archive["shape_source"], "shape_source")
        live_pose_layout = _scalar_string(
            archive["live_pose_layout"], "live_pose_layout"
        )

    return FlameStateTable(
        shape=shape,
        expression=expression,
        global_orient=global_orient,
        neck_pose=neck_pose,
        jaw_pose=jaw_pose,
        eyes=eyes,
        state_names=state_names,
        state_sources=state_sources,
        shape_source=shape_source,
        live_pose_layout=live_pose_layout,
    )


def load_joint_camera_metadata(
    path: str | Path,
    *,
    input_dir: str | Path | None = None,
    flame_states: FlameStateTable | None = None,
    require_images: bool = True,
) -> dict[str, Any]:
    """Load joint camera metadata and validate all observation mappings."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if metadata.get("schema_name") != SCHEMA_NAME:
        raise ValueError(
            f"Unsupported camera schema '{metadata.get('schema_name')}' in {path}"
        )
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported camera schema version {metadata.get('schema_version')} in {path}"
        )
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames found in {path}")

    if flame_states is not None:
        declared_states = int(metadata.get("num_flame_states", flame_states.num_states))
        if declared_states != flame_states.num_states:
            raise ValueError(
                f"num_flame_states says {declared_states}, but FLAME file has "
                f"{flame_states.num_states} states"
            )
        declared_shape_source = metadata.get("shape_source")
        if (
            declared_shape_source is not None
            and str(declared_shape_source) != flame_states.shape_source
        ):
            raise ValueError(
                f"Camera shape_source '{declared_shape_source}' does not match "
                f"FLAME file '{flame_states.shape_source}'"
            )
        declared_pose_layout = metadata.get("live_pose_layout")
        if (
            declared_pose_layout is not None
            and str(declared_pose_layout) != flame_states.live_pose_layout
        ):
            raise ValueError(
                f"Camera live_pose_layout '{declared_pose_layout}' does not match "
                f"FLAME file '{flame_states.live_pose_layout}'"
            )

    root = Path(input_dir) if input_dir is not None else path.parent
    seen_indices: set[int] = set()
    for fallback_index, frame in enumerate(frames):
        frame_index = int(frame.get("frame_index", fallback_index))
        if frame_index in seen_indices:
            raise ValueError(f"Duplicate frame_index {frame_index} in {path}")
        seen_indices.add(frame_index)

        for key in ("w", "h", "fx", "fy", "cx", "cy", "c2w", "w2c"):
            if key not in frame:
                raise KeyError(f"Frame {frame_index} is missing camera field '{key}'")
        if int(frame["w"]) <= 0 or int(frame["h"]) <= 0:
            raise ValueError(f"Frame {frame_index} has invalid image dimensions")
        if float(frame["fx"]) <= 0.0 or float(frame["fy"]) <= 0.0:
            raise ValueError(f"Frame {frame_index} has invalid focal length")

        c2w = np.asarray(frame["c2w"], dtype=np.float64)
        w2c = np.asarray(frame["w2c"], dtype=np.float64)
        if c2w.shape != (4, 4) or w2c.shape != (4, 4):
            raise ValueError(f"Frame {frame_index} camera matrices must be 4x4")
        inverse_error = float(np.abs(c2w @ w2c - np.eye(4)).max())
        if inverse_error > 1e-5:
            raise ValueError(
                f"Frame {frame_index} c2w/w2c mismatch: {inverse_error:.3e}"
            )

        if "flame_index" not in frame:
            raise KeyError(f"Frame {frame_index} is missing flame_index")
        flame_index = int(frame["flame_index"])
        if flame_states is not None and not 0 <= flame_index < flame_states.num_states:
            raise ValueError(
                f"Frame {frame_index} refers to invalid FLAME state {flame_index}"
            )
        if flame_states is not None and "source" in frame:
            expected_source = str(flame_states.state_sources[flame_index])
            if str(frame["source"]) != expected_source:
                raise ValueError(
                    f"Frame {frame_index} source '{frame['source']}' does not match "
                    f"FLAME state {flame_index} source '{expected_source}'"
                )

        if require_images:
            image_path = Path(frame["file_path"])
            if not image_path.is_absolute():
                image_path = root / image_path
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Frame {frame_index} image does not exist: {image_path}"
                )

    declared_count = int(metadata.get("num_images", len(frames)))
    if declared_count != len(frames):
        raise ValueError(
            f"num_images says {declared_count}, but cameras file has {len(frames)} frames"
        )
    return metadata


def resolve_joint_parameter_path(
    input_dir: str | Path,
    camera_metadata: dict[str, Any],
    override: str | Path | None = None,
) -> Path:
    """Resolve the FLAME parameter file recorded by the camera metadata."""

    input_dir = Path(input_dir)
    value = override or camera_metadata.get("flame_parameter_file")
    if value is None:
        raise KeyError(
            "Camera metadata has no flame_parameter_file; pass an explicit override"
        )
    path = Path(value)
    return path if path.is_absolute() else input_dir / path
