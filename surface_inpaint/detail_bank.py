"""Lossless, resumable screen-space targets for detail refinement.

The coherent-base stage can supervise arbitrary renders through a canonical
UV atlas.  A direct detail target is different: its RGB pixels are valid only
for the exact FLAME pose and calibrated camera that produced the diffusion
teacher image.  This module stores those observations as uint8 PNG files and
samples cameras from one pose group at a time.

The implementation deliberately has no renderer or CUDA dependency.  Teacher
generation can write observations incrementally, while training and resume can
load the completed manifest on CPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image


DETAIL_BANK_SCHEMA_VERSION = 1
DETAIL_SUPERVISION_MODE = "direct_teacher"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_1d(value: Any, name: str, expected: Optional[int]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if expected is not None and array.size != expected:
        raise ValueError(
            f"Pose field {name!r} has {array.size} values; expected {expected}"
        )
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"Pose field {name!r} must be non-empty and finite")
    return array


def pose_to_manifest(pose: Any) -> dict[str, Any]:
    """Serialize the dynamic FLAME components needed by ``builder.build``."""

    expression = _array_1d(pose.expression, "expression", None)
    jaw = _array_1d(pose.jaw_pose, "jaw_pose", 3)
    left_eye = _array_1d(pose.leye_pose, "leye_pose", 3)
    right_eye = _array_1d(pose.reye_pose, "reye_pose", 3)
    return {
        "expression": expression.tolist(),
        "jaw_pose": jaw.tolist(),
        "leye_pose": left_eye.tolist(),
        "reye_pose": right_eye.tolist(),
        "source_index": int(pose.source_index),
        "is_open_mouth": bool(pose.is_open_mouth),
        "is_reference": bool(pose.is_reference),
    }


def pose_from_manifest(value: Mapping[str, Any]) -> SimpleNamespace:
    """Reconstruct the pose object shape expected by the batch builder."""

    if not isinstance(value, Mapping):
        raise ValueError("A detail-bank pose must be a mapping")
    expression = _array_1d(value.get("expression"), "expression", None)
    jaw = _array_1d(value.get("jaw_pose"), "jaw_pose", 3)
    left_eye = _array_1d(value.get("leye_pose"), "leye_pose", 3)
    right_eye = _array_1d(value.get("reye_pose"), "reye_pose", 3)
    try:
        source_index = int(value["source_index"])
        is_open_mouth = bool(value["is_open_mouth"])
        is_reference = bool(value["is_reference"])
    except KeyError as error:
        raise ValueError(
            f"Detail-bank pose is missing field {error.args[0]!r}"
        ) from error
    return SimpleNamespace(
        expression=expression[None].copy(),
        jaw_pose=jaw[None].copy(),
        leye_pose=left_eye[None].copy(),
        reye_pose=right_eye[None].copy(),
        source_index=source_index,
        is_open_mouth=is_open_mouth,
        is_reference=is_reference,
    )


def _json_copy(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        encoded = json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON values") from error
    loaded = json.loads(encoded)
    if not isinstance(loaded, dict):
        raise AssertionError("JSON mapping round-trip unexpectedly changed type")
    return loaded


def _rgb_u8(value: Any) -> np.ndarray:
    is_tensor = torch.is_tensor(value)
    if is_tensor:
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(
            f"Direct RGB target must be a 3D CHW/HWC value, got {array.shape}"
        )
    if is_tensor:
        if array.shape[0] != 3:
            raise ValueError(
                "Tensor RGB target must have channel-first shape [3,H,W]"
            )
        array = np.moveaxis(array, 0, -1)
    elif array.shape[-1] == 3:
        pass
    elif array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    else:
        raise ValueError("RGB target must have exactly three channels")
    if np.issubdtype(array.dtype, np.integer):
        if array.min() < 0 or array.max() > 255:
            raise ValueError("Integer RGB target values must be in [0,255]")
        return np.ascontiguousarray(array, dtype=np.uint8)
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("RGB target contains NaN or infinity")
    tolerance = 1.0e-6
    if float(array.min()) < -tolerance or float(array.max()) > 1.0 + tolerance:
        raise ValueError("Floating RGB target values must be in [0,1]")
    return np.ascontiguousarray(
        np.rint(np.clip(array, 0.0, 1.0) * 255.0), dtype=np.uint8
    )


def _mask_u8(value: Any) -> np.ndarray:
    is_tensor = torch.is_tensor(value)
    if is_tensor:
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 3:
        if array.shape[0] == 1:
            array = array[0]
        elif array.shape[-1] == 1:
            array = array[..., 0]
        else:
            raise ValueError("Edit mask must have one channel")
    if array.ndim != 2:
        raise ValueError(
            f"Direct edit mask must have shape [H,W] or [1,H,W], got {array.shape}"
        )
    if np.issubdtype(array.dtype, np.integer):
        if array.min() < 0 or array.max() > 255:
            raise ValueError("Integer edit-mask values must be in [0,255]")
        return np.ascontiguousarray(array, dtype=np.uint8)
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Edit mask contains NaN or infinity")
    tolerance = 1.0e-6
    if float(array.min()) < -tolerance or float(array.max()) > 1.0 + tolerance:
        raise ValueError("Floating edit-mask values must be in [0,1]")
    return np.ascontiguousarray(
        np.rint(np.clip(array, 0.0, 1.0) * 255.0), dtype=np.uint8
    )


def _atomic_save_png(array: np.ndarray, path: Path, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    Image.fromarray(array, mode=mode).save(temporary, format="PNG")
    os.replace(temporary, path)


def _atomic_write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            dict(value),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe_member(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path must be a non-empty relative string")
    member = Path(relative)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"{label} path escapes the detail-bank directory")
    root_resolved = root.resolve()
    resolved = (root / member).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(
            f"{label} path escapes the detail-bank directory"
        ) from error
    return resolved


@dataclass(frozen=True)
class DetailTargetObservation:
    key: str
    pose_id: str
    pose: SimpleNamespace
    camera_index: int
    frame_index: int
    target_u8: torch.Tensor
    edit_mask_u8: torch.Tensor
    metadata: Mapping[str, Any]

    @property
    def height(self) -> int:
        return int(self.target_u8.shape[-2])

    @property
    def width(self) -> int:
        return int(self.target_u8.shape[-1])

    def target_float(self) -> torch.Tensor:
        return self.target_u8.to(dtype=torch.float32).div(255.0)

    def edit_mask_float(self) -> torch.Tensor:
        return self.edit_mask_u8.to(dtype=torch.float32).div(255.0)


@dataclass(frozen=True)
class DetailTargetBatch:
    """A training batch whose cameras all share exactly one FLAME pose."""

    pose_id: str
    pose: SimpleNamespace
    camera_indices: tuple[int, ...]
    frame_indices: tuple[int, ...]
    targets: torch.Tensor
    edit_masks: torch.Tensor
    observations: tuple[DetailTargetObservation, ...]


class DetailTargetBank:
    """An immutable, validated collection of direct screen-space targets."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        expected_config_sha256: Optional[str] = None,
        expected_implementation_sha256: Optional[str] = None,
        expected_manifest_sha256: Optional[str] = None,
        verify_files: bool = True,
        require_success: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing detail-target manifest: {self.manifest_path}"
            )
        self.directory = self.manifest_path.parent
        raw = self.manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if (
            expected_manifest_sha256 is not None
            and self.manifest_sha256 != str(expected_manifest_sha256)
        ):
            raise ValueError("Detail-target manifest SHA256 differs")
        try:
            manifest = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Detail-target manifest is not valid UTF-8 JSON") from error
        if not isinstance(manifest, Mapping):
            raise ValueError("Detail-target manifest must be a mapping")
        self._validate_header(
            manifest,
            expected_config_sha256,
            expected_implementation_sha256,
        )
        if require_success:
            self._validate_success_marker()

        self.refresh_step = int(manifest["refresh_step"])
        self.teacher_timestep = int(manifest["teacher_timestep"])
        self.config_sha256 = str(manifest["config_sha256"])
        self.implementation_sha256 = str(
            manifest["implementation_sha256"]
        )
        self.metadata = _json_copy(
            manifest.get("metadata", {}), "manifest metadata"
        )
        image_size = manifest.get("image_size")
        if (
            not isinstance(image_size, Sequence)
            or isinstance(image_size, (str, bytes))
            or len(image_size) != 2
        ):
            raise ValueError("Detail-target image_size must be [height,width]")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError("Detail-target image dimensions must be positive")

        pose_values = manifest.get("poses")
        if not isinstance(pose_values, list) or not pose_values:
            raise ValueError("Detail-target manifest must contain poses")
        poses: dict[str, SimpleNamespace] = {}
        pose_serialized: dict[str, Mapping[str, Any]] = {}
        for record in pose_values:
            if not isinstance(record, Mapping):
                raise ValueError("Each detail-target pose record must be a mapping")
            pose_id = str(record.get("pose_id", "")).strip()
            if not pose_id or pose_id in poses:
                raise ValueError("Detail-target pose IDs must be non-empty and unique")
            serialized = record.get("pose")
            if not isinstance(serialized, Mapping):
                raise ValueError(f"Pose {pose_id!r} is missing its FLAME values")
            poses[pose_id] = pose_from_manifest(serialized)
            pose_serialized[pose_id] = serialized
        self.poses = poses
        self.pose_serialized = pose_serialized

        observation_values = manifest.get("observations")
        if not isinstance(observation_values, list) or not observation_values:
            raise ValueError("Detail-target manifest must contain observations")
        declared_count = int(manifest.get("observation_count", -1))
        if declared_count != len(observation_values):
            raise ValueError(
                "Detail-target observation_count does not match observations"
            )

        observations: list[DetailTargetObservation] = []
        keys: set[str] = set()
        pose_cameras: set[tuple[str, int]] = set()
        by_pose: dict[str, list[DetailTargetObservation]] = {
            pose_id: [] for pose_id in poses
        }
        for record in observation_values:
            observation = self._load_observation(
                record, poses, verify_files=verify_files
            )
            if observation.key in keys:
                raise ValueError(
                    f"Duplicate detail-target observation key {observation.key!r}"
                )
            pair = (observation.pose_id, observation.camera_index)
            if pair in pose_cameras:
                raise ValueError(
                    "A detail-target pose contains a duplicate calibrated camera"
                )
            keys.add(observation.key)
            pose_cameras.add(pair)
            observations.append(observation)
            by_pose[observation.pose_id].append(observation)
        empty = [pose_id for pose_id, values in by_pose.items() if not values]
        if empty:
            raise ValueError(
                "Detail-target poses without observations: " + ", ".join(empty)
            )
        self.observations = tuple(observations)
        self.by_pose = {
            pose_id: tuple(values) for pose_id, values in by_pose.items()
        }
        self.pose_ids = tuple(by_pose)
        self.open_pose_ids = tuple(
            pose_id
            for pose_id, pose in poses.items()
            if bool(pose.is_open_mouth)
        )
        self.closed_pose_ids = tuple(
            pose_id
            for pose_id, pose in poses.items()
            if not bool(pose.is_open_mouth)
        )

    @staticmethod
    def _validate_header(
        manifest: Mapping[str, Any],
        expected_config_sha256: Optional[str],
        expected_implementation_sha256: Optional[str],
    ) -> None:
        if int(manifest.get("schema_version", -1)) != DETAIL_BANK_SCHEMA_VERSION:
            raise ValueError("Unsupported detail-target manifest schema")
        if manifest.get("supervision_mode") != DETAIL_SUPERVISION_MODE:
            raise ValueError("Manifest is not a direct-teacher target bank")
        for field in (
            "refresh_step",
            "teacher_timestep",
            "config_sha256",
            "implementation_sha256",
        ):
            if field not in manifest:
                raise ValueError(
                    f"Detail-target manifest is missing field {field!r}"
                )
        if int(manifest["refresh_step"]) < 0:
            raise ValueError("Detail-target refresh_step must be non-negative")
        if int(manifest["teacher_timestep"]) < 0:
            raise ValueError("Detail-target teacher_timestep must be non-negative")
        if (
            expected_config_sha256 is not None
            and manifest["config_sha256"] != expected_config_sha256
        ):
            raise ValueError("Detail-target cache configuration differs")
        if (
            expected_implementation_sha256 is not None
            and manifest["implementation_sha256"]
            != expected_implementation_sha256
        ):
            raise ValueError("Detail-target cache implementation differs")

    def _validate_success_marker(self) -> None:
        marker_path = self.directory / "_SUCCESS.json"
        if not marker_path.is_file():
            raise FileNotFoundError(
                f"Detail-target bank is incomplete: {marker_path} is missing"
            )
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("Detail-target success marker is invalid") from error
        if not isinstance(marker, Mapping):
            raise ValueError("Detail-target success marker must be a mapping")
        if int(marker.get("schema_version", -1)) != DETAIL_BANK_SCHEMA_VERSION:
            raise ValueError("Detail-target success marker schema differs")
        if marker.get("manifest_sha256") != self.manifest_sha256:
            raise ValueError("Detail-target success marker manifest SHA256 differs")

    def _load_observation(
        self,
        record: Any,
        poses: Mapping[str, SimpleNamespace],
        *,
        verify_files: bool,
    ) -> DetailTargetObservation:
        if not isinstance(record, Mapping):
            raise ValueError(
                "Each detail-target observation record must be a mapping"
            )
        key = str(record.get("key", "")).strip()
        pose_id = str(record.get("pose_id", "")).strip()
        if not key:
            raise ValueError("Detail-target observation key must be non-empty")
        if pose_id not in poses:
            raise ValueError(
                f"Observation {key!r} references unknown pose {pose_id!r}"
            )
        camera_index = int(record.get("camera_index", -1))
        frame_index = int(record.get("frame_index", -1))
        if camera_index < 0 or frame_index < 0:
            raise ValueError(
                "Detail-target camera_index and frame_index must be non-negative"
            )
        target_path = _safe_member(
            self.directory, record.get("target"), "target"
        )
        mask_path = _safe_member(
            self.directory, record.get("edit_mask"), "edit mask"
        )
        for path in (target_path, mask_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing detail-target image: {path}")
        if verify_files:
            if file_sha256(target_path) != record.get("target_sha256"):
                raise ValueError(
                    f"Detail-target RGB SHA256 differs for {key!r}"
                )
            if file_sha256(mask_path) != record.get("edit_mask_sha256"):
                raise ValueError(
                    f"Detail-target mask SHA256 differs for {key!r}"
                )

        with Image.open(target_path) as image:
            if image.mode != "RGB":
                raise ValueError(
                    f"Detail-target RGB image for {key!r} must use RGB mode"
                )
            target = np.asarray(image, dtype=np.uint8).copy()
        with Image.open(mask_path) as image:
            if image.mode != "L":
                raise ValueError(
                    f"Detail-target edit mask for {key!r} must use L mode"
                )
            edit_mask = np.asarray(image, dtype=np.uint8).copy()
        expected = self.image_size
        if target.shape[:2] != expected or edit_mask.shape != expected:
            raise ValueError(
                f"Detail-target observation {key!r} has inconsistent image size"
            )
        target_tensor = torch.from_numpy(target).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(edit_mask)[None].contiguous()
        return DetailTargetObservation(
            key=key,
            pose_id=pose_id,
            pose=poses[pose_id],
            camera_index=camera_index,
            frame_index=frame_index,
            target_u8=target_tensor,
            edit_mask_u8=mask_tensor,
            metadata=_json_copy(
                record.get("metadata", {}),
                f"observation {key!r} metadata",
            ),
        )

    def sample(
        self,
        batch_size: int,
        *,
        rng: Optional[random.Random] = None,
        open_mouth_probability: Optional[float] = None,
    ) -> DetailTargetBatch:
        """Sample distinct calibrated cameras from one exact pose group."""

        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("Detail-target batch_size must be positive")
        generator: Any = random if rng is None else rng
        pose_ids = self.pose_ids
        if open_mouth_probability is not None:
            probability = float(open_mouth_probability)
            if not 0.0 <= probability <= 1.0:
                raise ValueError("open_mouth_probability must be in [0,1]")
            if self.open_pose_ids and self.closed_pose_ids:
                pose_ids = (
                    self.open_pose_ids
                    if generator.random() < probability
                    else self.closed_pose_ids
                )
            elif self.open_pose_ids:
                pose_ids = self.open_pose_ids
            else:
                pose_ids = self.closed_pose_ids
        eligible = [
            pose_id
            for pose_id in pose_ids
            if len(self.by_pose[pose_id]) >= batch_size
        ]
        if not eligible:
            raise ValueError(
                f"No detail-target pose has {batch_size} distinct cameras"
            )
        pose_id = generator.choice(eligible)
        selected = tuple(generator.sample(self.by_pose[pose_id], batch_size))
        return DetailTargetBatch(
            pose_id=pose_id,
            pose=self.poses[pose_id],
            camera_indices=tuple(item.camera_index for item in selected),
            frame_indices=tuple(item.frame_index for item in selected),
            targets=torch.stack(
                [item.target_float() for item in selected], dim=0
            ),
            edit_masks=torch.stack(
                [item.edit_mask_float() for item in selected], dim=0
            ),
            observations=selected,
        )

    def checkpoint_descriptor(
        self, relative_to: Optional[Path] = None
    ) -> dict[str, Any]:
        manifest = self.manifest_path
        if relative_to is not None:
            try:
                manifest_value = manifest.relative_to(
                    Path(relative_to).resolve()
                ).as_posix()
            except ValueError as error:
                raise ValueError(
                    "Detail-target manifest is outside the checkpoint root"
                ) from error
        else:
            manifest_value = str(manifest)
        return {
            "schema_version": DETAIL_BANK_SCHEMA_VERSION,
            "refresh_step": int(self.refresh_step),
            "teacher_timestep": int(self.teacher_timestep),
            "manifest": manifest_value,
            "manifest_sha256": self.manifest_sha256,
        }

    @classmethod
    def from_checkpoint_descriptor(
        cls,
        value: Mapping[str, Any],
        *,
        root: Optional[Path] = None,
        expected_config_sha256: Optional[str] = None,
        expected_implementation_sha256: Optional[str] = None,
        verify_files: bool = True,
    ) -> "DetailTargetBank":
        if not isinstance(value, Mapping):
            raise ValueError("Detail-target checkpoint descriptor must be a mapping")
        if int(value.get("schema_version", -1)) != DETAIL_BANK_SCHEMA_VERSION:
            raise ValueError("Unsupported detail-target checkpoint descriptor")
        raw_path = Path(str(value.get("manifest", "")))
        if not str(raw_path):
            raise ValueError("Detail-target checkpoint descriptor has no manifest")
        if root is not None:
            if raw_path.is_absolute() or ".." in raw_path.parts:
                raise ValueError(
                    "Checkpoint detail-target manifest must be relative to root"
                )
            manifest = _safe_member(
                Path(root), raw_path.as_posix(), "checkpoint manifest"
            )
        else:
            manifest = raw_path
        bank = cls(
            manifest,
            expected_config_sha256=expected_config_sha256,
            expected_implementation_sha256=expected_implementation_sha256,
            expected_manifest_sha256=str(value.get("manifest_sha256", "")),
            verify_files=verify_files,
        )
        if int(value.get("refresh_step", -1)) != bank.refresh_step:
            raise ValueError("Checkpoint detail-target refresh step differs")
        if int(value.get("teacher_timestep", -1)) != bank.teacher_timestep:
            raise ValueError("Checkpoint detail-target timestep differs")
        return bank


class DetailTargetBankWriter:
    """Incrementally write one teacher refresh and publish it atomically."""

    def __init__(
        self,
        directory: Path,
        *,
        refresh_step: int,
        teacher_timestep: int,
        config_sha256: str,
        implementation_sha256: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        success = self.directory / "_SUCCESS.json"
        if success.exists():
            raise FileExistsError(
                f"Refusing to overwrite a completed detail-target bank: {success}"
            )
        self.refresh_step = int(refresh_step)
        self.teacher_timestep = int(teacher_timestep)
        if self.refresh_step < 0 or self.teacher_timestep < 0:
            raise ValueError("Refresh step and teacher timestep must be non-negative")
        self.config_sha256 = str(config_sha256)
        self.implementation_sha256 = str(implementation_sha256)
        if not self.config_sha256 or not self.implementation_sha256:
            raise ValueError("Detail-target provenance digests must be non-empty")
        self.metadata = _json_copy(metadata or {}, "bank metadata")
        self._poses: dict[str, dict[str, Any]] = {}
        self._observations: list[dict[str, Any]] = []
        self._pose_cameras: set[tuple[str, int]] = set()
        self._image_size: Optional[tuple[int, int]] = None
        self._finalized = False

    def add(
        self,
        *,
        pose_id: str,
        pose: Any,
        camera_index: int,
        frame_index: int,
        target: Any,
        edit_mask: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        if self._finalized:
            raise RuntimeError("Cannot add to a finalized detail-target bank")
        pose_id = str(pose_id).strip()
        if not pose_id:
            raise ValueError("pose_id must be non-empty")
        serialized_pose = pose_to_manifest(pose)
        previous = self._poses.get(pose_id)
        if previous is not None and previous != serialized_pose:
            raise ValueError(
                f"Pose ID {pose_id!r} was reused with different FLAME values"
            )
        self._poses.setdefault(pose_id, serialized_pose)

        camera_index = int(camera_index)
        frame_index = int(frame_index)
        if camera_index < 0 or frame_index < 0:
            raise ValueError("camera_index and frame_index must be non-negative")
        pair = (pose_id, camera_index)
        if pair in self._pose_cameras:
            raise ValueError(
                f"Pose {pose_id!r} already contains camera {camera_index}"
            )

        target_u8 = _rgb_u8(target)
        mask_u8 = _mask_u8(edit_mask)
        image_size = (int(target_u8.shape[0]), int(target_u8.shape[1]))
        if mask_u8.shape != image_size:
            raise ValueError("Direct target and edit mask have different sizes")
        if self._image_size is None:
            self._image_size = image_size
        elif image_size != self._image_size:
            raise ValueError("All direct targets in a bank must share one size")

        key = f"observation_{len(self._observations):06d}"
        target_relative = Path("targets") / f"{key}.png"
        mask_relative = Path("edit_masks") / f"{key}.png"
        target_path = self.directory / target_relative
        mask_path = self.directory / mask_relative
        _atomic_save_png(target_u8, target_path, "RGB")
        _atomic_save_png(mask_u8, mask_path, "L")
        self._observations.append(
            {
                "key": key,
                "pose_id": pose_id,
                "camera_index": camera_index,
                "frame_index": frame_index,
                "target": target_relative.as_posix(),
                "edit_mask": mask_relative.as_posix(),
                "target_sha256": file_sha256(target_path),
                "edit_mask_sha256": file_sha256(mask_path),
                "metadata": _json_copy(
                    metadata or {}, f"observation {key!r} metadata"
                ),
            }
        )
        self._pose_cameras.add(pair)
        return key

    def finalize(self, *, verify_files: bool = True) -> DetailTargetBank:
        if self._finalized:
            raise RuntimeError("Detail-target bank has already been finalized")
        if not self._observations or self._image_size is None:
            raise ValueError("Cannot finalize an empty detail-target bank")
        manifest = {
            "schema_version": DETAIL_BANK_SCHEMA_VERSION,
            "supervision_mode": DETAIL_SUPERVISION_MODE,
            "refresh_step": self.refresh_step,
            "teacher_timestep": self.teacher_timestep,
            "config_sha256": self.config_sha256,
            "implementation_sha256": self.implementation_sha256,
            "image_encoding": {
                "target": "RGB uint8 PNG",
                "edit_mask": "L uint8 PNG",
                "float_decode": "value / 255",
            },
            "image_size": list(self._image_size),
            "pose_count": len(self._poses),
            "observation_count": len(self._observations),
            "poses": [
                {"pose_id": pose_id, "pose": pose}
                for pose_id, pose in self._poses.items()
            ],
            "observations": self._observations,
            "metadata": self.metadata,
        }
        manifest_path = self.directory / "manifest.json"
        _atomic_write_json(manifest, manifest_path)
        manifest_sha256 = file_sha256(manifest_path)
        _atomic_write_json(
            {
                "schema_version": DETAIL_BANK_SCHEMA_VERSION,
                "supervision_mode": DETAIL_SUPERVISION_MODE,
                "refresh_step": self.refresh_step,
                "teacher_timestep": self.teacher_timestep,
                "pose_count": len(self._poses),
                "observation_count": len(self._observations),
                "manifest_sha256": manifest_sha256,
            },
            self.directory / "_SUCCESS.json",
        )
        self._finalized = True
        return DetailTargetBank(
            manifest_path,
            expected_config_sha256=self.config_sha256,
            expected_implementation_sha256=self.implementation_sha256,
            expected_manifest_sha256=manifest_sha256,
            verify_files=verify_files,
        )

