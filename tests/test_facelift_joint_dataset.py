import json

import numpy as np
from PIL import Image

from tools.facelift_joint_dataset import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    load_flame_state_table,
    load_joint_camera_metadata,
    resolve_joint_parameter_path,
)
from tools.prepare_facelift_joint_dataset import build_flame_arrays


def _source_parameters():
    optim = {
        "shape": np.arange(300, dtype=np.float32),
        "expression": np.arange(100, dtype=np.float32),
        "pose": np.arange(6, dtype=np.float32),
        "eyes": np.arange(6, dtype=np.float32) + 10.0,
    }
    live = {
        "shape": np.stack(
            [
                np.full(300, 2.0, dtype=np.float32),
                np.full(300, 4.0, dtype=np.float32),
            ]
        ),
        "expression": np.stack(
            [
                np.full(100, 20.0, dtype=np.float32),
                np.full(100, 40.0, dtype=np.float32),
            ]
        ),
        "pose": np.stack(
            [
                np.arange(9, dtype=np.float32) + 100.0,
                np.arange(9, dtype=np.float32) + 200.0,
            ]
        ),
    }
    return optim, live


def test_jaw_eye_pose_layout_is_explicit():
    optim, live = _source_parameters()
    arrays = build_flame_arrays(
        optim,
        live,
        ["frame_00000", "frame_00001"],
        "optim",
        "jaw-eyes",
    )

    np.testing.assert_array_equal(arrays["shape"], optim["shape"][None])
    np.testing.assert_array_equal(arrays["global_orient"][0], optim["pose"][:3])
    np.testing.assert_array_equal(arrays["jaw_pose"][0], optim["pose"][3:6])
    np.testing.assert_array_equal(arrays["eyes"][0], optim["eyes"])
    np.testing.assert_array_equal(arrays["jaw_pose"][1:], live["pose"][:, :3])
    np.testing.assert_array_equal(arrays["eyes"][1:], live["pose"][:, 3:9])
    assert not arrays["global_orient"][1:].any()
    assert not arrays["neck_pose"][1:].any()


def test_liveportrait_mean_shape_is_shared():
    optim, live = _source_parameters()
    arrays = build_flame_arrays(
        optim,
        live,
        ["frame_00000", "frame_00001"],
        "liveportrait-mean",
        "jaw-eyes",
    )
    assert arrays["shape"].shape == (1, 300)
    np.testing.assert_array_equal(arrays["shape"], np.full((1, 300), 3.0))


def test_joint_files_round_trip_and_validate_mapping(tmp_path):
    optim, live = _source_parameters()
    arrays = build_flame_arrays(
        optim,
        live,
        ["frame_00000", "frame_00001"],
        "optim",
        "jaw-eyes",
    )
    flame_path = tmp_path / "flame_params_joint.npz"
    np.savez_compressed(flame_path, **arrays)
    table = load_flame_state_table(flame_path)
    assert table.num_states == 3
    assert table.shape_source == "optim"
    assert table.live_pose_layout == "jaw-eyes"

    Image.new("RGB", (2, 2), "white").save(tmp_path / "frame.png")
    identity = np.eye(4).tolist()
    metadata = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "num_images": 1,
        "flame_parameter_file": flame_path.name,
        "frames": [
            {
                "frame_index": 7,
                "file_path": "frame.png",
                "w": 2,
                "h": 2,
                "fx": 2.0,
                "fy": 2.0,
                "cx": 1.0,
                "cy": 1.0,
                "c2w": identity,
                "w2c": identity,
                "flame_index": 2,
            }
        ],
    }
    camera_path = tmp_path / "cameras_joint.json"
    camera_path.write_text(json.dumps(metadata), encoding="utf-8")

    loaded = load_joint_camera_metadata(
        camera_path,
        input_dir=tmp_path,
        flame_states=table,
        require_images=True,
    )
    assert loaded["frames"][0]["flame_index"] == 2
    assert resolve_joint_parameter_path(tmp_path, loaded) == flame_path
