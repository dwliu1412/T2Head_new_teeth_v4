import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from surface_inpaint.detail_bank import (
    DetailTargetBank,
    DetailTargetBankWriter,
    pose_from_manifest,
    pose_to_manifest,
)
from surface_inpaint.pipeline import LoopInpaintTrainer


def make_pose(index: int, *, open_mouth: bool) -> SimpleNamespace:
    expression = np.arange(100, dtype=np.float32)[None] + index / 10.0
    jaw = np.asarray([[float(index), 0.2, -0.1]], dtype=np.float32)
    left = np.asarray([[0.01, 0.02, 0.03]], dtype=np.float32)
    right = np.asarray([[-0.01, -0.02, -0.03]], dtype=np.float32)
    return SimpleNamespace(
        expression=expression,
        jaw_pose=jaw,
        leye_pose=left,
        reye_pose=right,
        source_index=index,
        is_open_mouth=open_mouth,
        is_reference=index < 0,
    )


class PoseManifestTests(unittest.TestCase):
    def test_pose_round_trip_preserves_builder_shape_and_float32_values(self):
        original = make_pose(7, open_mouth=True)

        restored = pose_from_manifest(pose_to_manifest(original))

        self.assertEqual(restored.expression.shape, (1, 100))
        self.assertEqual(restored.jaw_pose.shape, (1, 3))
        self.assertEqual(restored.expression.dtype, np.float32)
        np.testing.assert_array_equal(restored.expression, original.expression)
        np.testing.assert_array_equal(restored.jaw_pose, original.jaw_pose)
        np.testing.assert_array_equal(restored.leye_pose, original.leye_pose)
        np.testing.assert_array_equal(restored.reye_pose, original.reye_pose)
        self.assertEqual(restored.source_index, original.source_index)
        self.assertTrue(restored.is_open_mouth)


class DetailTargetBankTests(unittest.TestCase):
    CONFIG_DIGEST = "config-digest"
    IMPLEMENTATION_DIGEST = "implementation-digest"

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def build_bank(self) -> DetailTargetBank:
        writer = DetailTargetBankWriter(
            self.root / "teacher" / "step_000100" / "direct_targets",
            refresh_step=100,
            teacher_timestep=300,
            config_sha256=self.CONFIG_DIGEST,
            implementation_sha256=self.IMPLEMENTATION_DIGEST,
            metadata={"stage": "detail"},
        )
        for pose_index, open_mouth in ((0, False), (1, True)):
            pose = make_pose(pose_index, open_mouth=open_mouth)
            for camera_index in range(3):
                value = (pose_index * 60 + camera_index * 10 + 5) / 255.0
                target = torch.full((3, 8, 10), value, dtype=torch.float32)
                mask = torch.full(
                    (1, 8, 10),
                    (camera_index + 1) / 4.0,
                    dtype=torch.float32,
                )
                writer.add(
                    pose_id=f"pose_{pose_index}",
                    pose=pose,
                    camera_index=camera_index,
                    frame_index=20 + camera_index,
                    target=target,
                    edit_mask=mask,
                    metadata={"view_order": camera_index},
                )
        return writer.finalize()

    def test_png_bank_loads_quantized_targets_and_samples_one_pose(self):
        bank = self.build_bank()

        batch = bank.sample(2, rng=random.Random(11))

        self.assertEqual(len(set(item.pose_id for item in batch.observations)), 1)
        self.assertEqual(batch.pose_id, batch.observations[0].pose_id)
        self.assertEqual(batch.camera_indices, tuple(
            item.camera_index for item in batch.observations
        ))
        self.assertEqual(batch.targets.shape, (2, 3, 8, 10))
        self.assertEqual(batch.edit_masks.shape, (2, 1, 8, 10))
        self.assertEqual(batch.targets.dtype, torch.float32)
        for index, observation in enumerate(batch.observations):
            torch.testing.assert_close(
                batch.targets[index],
                observation.target_u8.float() / 255.0,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                batch.edit_masks[index],
                observation.edit_mask_u8.float() / 255.0,
                rtol=0.0,
                atol=0.0,
            )

    def test_open_mouth_sampling_can_be_forced(self):
        bank = self.build_bank()

        batch = bank.sample(
            2,
            rng=random.Random(3),
            open_mouth_probability=1.0,
        )

        self.assertTrue(batch.pose.is_open_mouth)
        self.assertEqual(batch.pose_id, "pose_1")

    def test_checkpoint_descriptor_reloads_relative_manifest(self):
        bank = self.build_bank()
        descriptor = bank.checkpoint_descriptor(relative_to=self.root)

        restored = DetailTargetBank.from_checkpoint_descriptor(
            descriptor,
            root=self.root,
            expected_config_sha256=self.CONFIG_DIGEST,
            expected_implementation_sha256=self.IMPLEMENTATION_DIGEST,
        )

        self.assertEqual(restored.manifest_sha256, bank.manifest_sha256)
        self.assertEqual(restored.refresh_step, 100)
        self.assertEqual(len(restored.observations), 6)

    def test_corrupt_target_is_rejected_by_sha256_validation(self):
        bank = self.build_bank()
        manifest = json.loads(bank.manifest_path.read_text(encoding="utf-8"))
        target_path = bank.directory / manifest["observations"][0]["target"]
        target_path.write_bytes(b"not a png")

        with self.assertRaisesRegex(ValueError, "RGB SHA256 differs"):
            DetailTargetBank(bank.manifest_path)

    def test_same_pose_id_cannot_change_flame_values(self):
        writer = DetailTargetBankWriter(
            self.root / "bank",
            refresh_step=0,
            teacher_timestep=20,
            config_sha256=self.CONFIG_DIGEST,
            implementation_sha256=self.IMPLEMENTATION_DIGEST,
        )
        target = np.zeros((8, 10, 3), dtype=np.uint8)
        mask = np.zeros((8, 10), dtype=np.uint8)
        writer.add(
            pose_id="pose",
            pose=make_pose(0, open_mouth=False),
            camera_index=0,
            frame_index=0,
            target=target,
            edit_mask=mask,
        )

        with self.assertRaisesRegex(ValueError, "different FLAME values"):
            writer.add(
                pose_id="pose",
                pose=make_pose(1, open_mouth=True),
                camera_index=1,
                frame_index=1,
                target=target,
                edit_mask=mask,
            )


class DeterministicDirectPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def build_preview_bank(self) -> DetailTargetBank:
        writer = DetailTargetBankWriter(
            self.root / "direct_bank",
            refresh_step=100,
            teacher_timestep=200,
            config_sha256="config",
            implementation_sha256="implementation",
        )
        poses = (
            ("closed_large_jaw", make_pose(9, open_mouth=False), [0, 1]),
            ("open_small_jaw", make_pose(2, open_mouth=True), [0, 1]),
            (
                "open_largest_jaw",
                make_pose(5, open_mouth=True),
                [5, 1, 7, 3, 0, 6, 2, 4],
            ),
        )
        for pose_id, pose, view_orders in poses:
            for view_order in view_orders:
                target_value = view_order * 20 + 3
                mask_value = view_order * 10 + 1
                writer.add(
                    pose_id=pose_id,
                    pose=pose,
                    camera_index=100 + view_order,
                    frame_index=200 + view_order,
                    target=np.full(
                        (6, 8, 3), target_value, dtype=np.uint8
                    ),
                    edit_mask=np.full(
                        (6, 8), mask_value, dtype=np.uint8
                    ),
                    metadata={"view_order": view_order},
                )
        return writer.finalize()

    @staticmethod
    def numpy_random_state_equal(left, right):
        return (
            left[0] == right[0]
            and np.array_equal(left[1], right[1])
            and left[2:] == right[2:]
        )

    def test_preview_uses_largest_open_jaw_and_even_view_order_without_rng(self):
        trainer = LoopInpaintTrainer.__new__(LoopInpaintTrainer)
        trainer.direct_bank = self.build_preview_bank()
        random.seed(9173)
        np.random.seed(421)
        python_before = random.getstate()
        numpy_before = np.random.get_state()

        first = trainer._deterministic_direct_preview(4)
        python_between = random.getstate()
        numpy_between = np.random.get_state()
        second = trainer._deterministic_direct_preview(4)
        python_after = random.getstate()
        numpy_after = np.random.get_state()

        self.assertEqual(first[0], "open_largest_jaw")
        self.assertTrue(first[1].is_open_mouth)
        self.assertEqual(float(first[1].jaw_pose[0, 0]), 5.0)
        # Eight sorted view orders sampled at four equal intervals: 0,2,4,6.
        self.assertEqual(first[2], (100, 102, 104, 106))
        self.assertEqual(first[3], (200, 202, 204, 206))
        expected_targets = torch.tensor(
            [3, 43, 83, 123], dtype=torch.float32
        ).div(255.0)
        expected_masks = torch.tensor(
            [1, 21, 41, 61], dtype=torch.float32
        ).div(255.0)
        torch.testing.assert_close(
            first[4][:, 0, 0, 0],
            expected_targets,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            first[5][:, 0, 0, 0],
            expected_masks,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(first[:4], second[:4])
        torch.testing.assert_close(first[4], second[4], rtol=0.0, atol=0.0)
        torch.testing.assert_close(first[5], second[5], rtol=0.0, atol=0.0)
        self.assertEqual(python_before, python_between)
        self.assertEqual(python_before, python_after)
        self.assertTrue(
            self.numpy_random_state_equal(numpy_before, numpy_between)
        )
        self.assertTrue(
            self.numpy_random_state_equal(numpy_before, numpy_after)
        )


if __name__ == "__main__":
    unittest.main()
