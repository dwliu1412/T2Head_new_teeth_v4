import copy
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from surface_inpaint.pipeline import (
    LoopInpaintTrainer,
    canonical_camera_indices,
    expected_active_refresh_step,
    expected_base_atlas_refresh_step,
    load_config,
    stratified_refresh_pose_envelope,
    teacher_observation_schedule,
    teacher_preview_view_orders,
    validate_config,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "loop_inpaint.yaml"
)


class FakeAssets:
    def __init__(self):
        self.chemistry_jaw = np.zeros((12, 3), dtype=np.float32)
        self.chemistry_jaw[:, 0] = np.arange(12, dtype=np.float32)
        self.chemistry_expression = np.zeros((12, 100), dtype=np.float32)
        self.chemistry_expression[:, 0] = np.arange(12, dtype=np.float32)
        self.chemistry_open_indices = np.asarray([9, 10, 11])
        self.reference_pose = self._pose(-1, reference=True)
        self.validation_pose = self._pose(20, validation=True)

    def _pose(self, index, reference=False, validation=False):
        jaw_x = 20.0 if validation else max(float(index), 0.0)
        expression = np.zeros((1, 100), dtype=np.float32)
        expression[0, 0] = float(index)
        jaw = np.zeros((1, 3), dtype=np.float32)
        jaw[0, 0] = jaw_x
        eye = np.zeros((1, 3), dtype=np.float32)
        return SimpleNamespace(
            expression=expression,
            jaw_pose=jaw,
            leye_pose=eye.copy(),
            reye_pose=eye.copy(),
            source_index=int(index),
            is_open_mouth=bool(validation or index >= 9),
            is_reference=bool(reference),
        )

    def chemistry_pose(self, local_index):
        return self._pose(int(local_index))

    def sample_pose(self):
        return self.chemistry_pose(0)


class TeacherPreviewAllocationTests(unittest.TestCase):
    def test_save_all_returns_every_view_for_each_pose(self):
        output = {
            "save_all_teacher_observations": True,
            "teacher_previews_per_pose": 1,
        }
        self.assertEqual(
            teacher_preview_view_orders(12, output),
            list(range(12)),
        )

    def test_limited_previews_are_evenly_spaced_per_pose(self):
        output = {
            "save_all_teacher_observations": False,
            "teacher_previews_per_pose": 4,
        }
        self.assertEqual(
            teacher_preview_view_orders(12, output),
            [0, 3, 6, 9],
        )


class TeacherCameraSamplingTests(unittest.TestCase):
    def test_stratified_camera_sampling_covers_every_elevation_ring(self):
        frames = []
        groups = []
        elevations = (-30.0, -15.0, 0.0, 15.0, 30.0)
        for elevation in elevations:
            group = []
            for azimuth_index in range(12):
                group.append(len(frames))
                frames.append(
                    SimpleNamespace(
                        source_azimuth_deg=float(azimuth_index * 30),
                        source_elevation_deg=elevation,
                    )
                )
            groups.append(group)
        assets = SimpleNamespace(frames=frames, elevation_groups=groups)

        selected = canonical_camera_indices(
            assets, 24, mode="stratified_all_rings"
        )

        selected_elevations = [
            frames[index].source_elevation_deg for index in selected
        ]
        counts = [
            selected_elevations.count(elevation) for elevation in elevations
        ]
        self.assertEqual(len(selected), 24)
        self.assertEqual(len(set(selected)), 24)
        self.assertTrue(all(count in {4, 5} for count in counts))
        self.assertEqual(sum(counts), 24)


class StratifiedTeacherPoseTests(unittest.TestCase):
    def setUp(self):
        self.assets = FakeAssets()
        random.seed(7)

    def test_reference_plus_three_jaw_strata_contains_open_mouth(self):
        named = stratified_refresh_pose_envelope(
            self.assets,
            4,
            include_reference=True,
            require_open_mouth=True,
        )
        self.assertEqual(len(named), 4)
        self.assertEqual(named[0][0], "reference")
        dynamic = [pose for _, pose in named[1:]]
        jaw_x = [float(pose.jaw_pose[0, 0]) for pose in dynamic]
        self.assertTrue(0.0 <= jaw_x[0] < 4.0)
        self.assertTrue(4.0 <= jaw_x[1] < 8.0)
        self.assertTrue(8.0 <= jaw_x[2] < 12.0)
        self.assertTrue(any(pose.is_open_mouth for pose in dynamic))
        self.assertEqual(len({pose.source_index for pose in dynamic}), 3)

    def test_coarse_boundary_switches_to_detail_pose_policy(self):
        trainer = LoopInpaintTrainer.__new__(LoopInpaintTrainer)
        trainer.assets = self.assets
        trainer.config = {
            "teacher": {
                "coarse_iterations": 10,
                "coarse_poses_per_refresh": 5,
                "coarse_pose_quantiles": [0.50, 0.80, 0.95],
                "include_validation_coarse_pose": True,
                "poses_per_refresh": 4,
                "include_reference_pose": True,
                "stratify_dynamic_poses_by_jaw": True,
                "require_open_mouth_pose": True,
            },
            "data": {"use_dynamic_expression": True},
        }

        anchors = trainer._refresh_poses(0)
        coarse_dynamic = trainer._refresh_poses(9)
        detail = trainer._refresh_poses(10)

        self.assertEqual(
            [label for label, _ in anchors],
            [
                "reference",
                "jaw_q500",
                "jaw_q800",
                "jaw_q950",
                "validation_open_mouth",
            ],
        )
        self.assertEqual(len(coarse_dynamic), 5)
        self.assertTrue(
            any(pose.is_open_mouth for _, pose in coarse_dynamic)
        )
        self.assertEqual(len(detail), 4)
        self.assertTrue(
            all(label != "validation_open_mouth" for label, _ in detail)
        )
        self.assertTrue(any(pose.is_open_mouth for _, pose in detail))


class TeacherEvidenceConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid_config = load_config(CONFIG_PATH, ())
        validate_config(cls.valid_config, check_files=False)

    def test_removed_global_preview_limit_is_rejected(self):
        config = copy.deepcopy(self.valid_config)
        config["output"]["teacher_previews"] = 6
        with self.assertRaisesRegex(ValueError, "removed global preview limit"):
            validate_config(config, check_files=False)

    def test_default_observation_budget_matches_documented_schedule(self):
        schedule = teacher_observation_schedule(self.valid_config)
        self.assertEqual(schedule["base_refresh_count"], 10)
        self.assertEqual(schedule["detail_refresh_count"], 6)
        self.assertEqual(schedule["base_observation_budget"], 1200)
        self.assertEqual(schedule["detail_observation_budget"], 576)

    def test_default_refresh_resume_boundaries(self):
        # A refresh becomes active only after its training step has completed.
        # The Base atlas stays frozen at the final Base refresh after Detail
        # begins, while the direct target bank continues to advance.
        expected = {
            1: (0, 0),
            10000: (9000, 9000),
            10001: (10000, 9000),
            10500: (10000, 9000),
            13000: (12500, 9000),
        }
        for completed_steps, (
            active_refresh,
            base_refresh,
        ) in expected.items():
            with self.subTest(completed_steps=completed_steps):
                self.assertEqual(
                    expected_active_refresh_step(
                        self.valid_config, completed_steps
                    ),
                    active_refresh,
                )
                self.assertEqual(
                    expected_base_atlas_refresh_step(
                        self.valid_config, completed_steps
                    ),
                    base_refresh,
                )

    def test_both_base_and_detail_stages_are_required(self):
        total_steps = int(
            self.valid_config["optimization"]["iterations"]
        )
        for coarse_steps in (0, total_steps):
            config = copy.deepcopy(self.valid_config)
            config["teacher"]["coarse_iterations"] = coarse_steps
            with self.subTest(coarse_iterations=coarse_steps):
                with self.assertRaisesRegex(
                    ValueError,
                    r"coarse_iterations must be in .*both Base and Detail",
                ):
                    validate_config(config, check_files=False)

    def test_direct_batch_cannot_exceed_views_per_pose(self):
        config = copy.deepcopy(self.valid_config)
        config["data"]["batch_size"] = (
            int(config["teacher"]["views_per_pose"]) + 1
        )
        with self.assertRaisesRegex(
            ValueError,
            r"data\.batch_size must be positive and no greater than "
            r"teacher\.views_per_pose",
        ):
            validate_config(config, check_files=False)

    def test_v3_layered_direct_supervision_schema_is_explicit(self):
        self.assertEqual(
            self.valid_config["output"]["name"],
            "00000001_surface_coherent_v3",
        )
        self.assertEqual(
            self.valid_config["detail_supervision"],
            {
                "mode": "direct_teacher",
                "base_direct_probability": 0.5,
                "open_mouth_probability": 0.5,
            },
        )
        self.assertEqual(
            self.valid_config["fusion"]["layered_surface"],
            {
                "enabled": True,
                "layers": [
                    "lips",
                    "teeth_upper",
                    "teeth_lower",
                    "oral_cavity",
                ],
                "opacity_floor": 0.03,
                "contribution_threshold": 0.005,
                "residual_decomposition_floor": 0.01,
                "required_effective_layers": [
                    "teeth_upper",
                    "teeth_lower",
                ],
                "minimum_effective_gaussians": 1,
                "dominance_ratio": 1.20,
            },
        )
        self.assertEqual(
            self.valid_config["loss"]["layered_oral_weight"],
            2.0,
        )
        self.assertLessEqual(
            self.valid_config["surface_attention"]["alpha_threshold"],
            self.valid_config["fusion"]["layered_surface"][
                "contribution_threshold"
            ],
        )

    def test_attention_cannot_filter_accepted_oral_correspondence(self):
        config = copy.deepcopy(self.valid_config)
        config["surface_attention"]["alpha_threshold"] = 0.02
        with self.assertRaisesRegex(
            ValueError,
            r"surface_attention\.alpha_threshold must be no greater",
        ):
            validate_config(config, check_files=False)

    def test_residual_decomposition_floor_bounds_weak_layers(self):
        config = copy.deepcopy(self.valid_config)
        config["fusion"]["layered_surface"][
            "residual_decomposition_floor"
        ] = 0.001
        with self.assertRaisesRegex(
            ValueError,
            r"residual_decomposition_floor must be no smaller",
        ):
            validate_config(config, check_files=False)

    def test_required_effective_layers_must_be_known_and_unique(self):
        invalid = (
            ["teeth_upper", "teeth_upper"],
            ["teeth_upper", "tongue"],
        )
        for names in invalid:
            with self.subTest(names=names):
                config = copy.deepcopy(self.valid_config)
                config["fusion"]["layered_surface"][
                    "required_effective_layers"
                ] = names
                with self.assertRaisesRegex(
                    ValueError,
                    r"required_effective_layers",
                ):
                    validate_config(config, check_files=False)

    def test_per_pose_preview_count_must_fit_when_not_saving_all(self):
        config = copy.deepcopy(self.valid_config)
        config["output"]["save_all_teacher_observations"] = False
        config["output"]["teacher_previews_per_pose"] = (
            int(config["teacher"]["views_per_pose"]) + 1
        )
        with self.assertRaisesRegex(
            ValueError, "cannot exceed teacher.views_per_pose"
        ):
            validate_config(config, check_files=False)


class TeacherRefreshRetryTests(unittest.TestCase):
    def test_completed_outer_and_inner_markers_create_lossless_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            refresh_dir = Path(temporary) / "step_010000"
            bank_dir = refresh_dir / "direct_bank"
            target_dir = bank_dir / "targets"
            target_dir.mkdir(parents=True)
            outer_marker = refresh_dir / "_SUCCESS.json"
            inner_marker = bank_dir / "_SUCCESS.json"
            manifest = bank_dir / "manifest.json"
            target = target_dir / "teacher.png"
            outer_bytes = b'{"refresh_step": 10000}'
            inner_bytes = b'{"observation_count": 96}'
            manifest_bytes = b'{"schema_version": 1}'
            target_bytes = b"existing-target-must-survive"
            outer_marker.write_bytes(outer_bytes)
            inner_marker.write_bytes(inner_bytes)
            manifest.write_bytes(manifest_bytes)
            target.write_bytes(target_bytes)

            retry_dir = LoopInpaintTrainer._prepare_refresh_attempt(
                refresh_dir
            )

            self.assertEqual(
                retry_dir,
                refresh_dir / "direct_bank_retry_01",
            )
            self.assertFalse(outer_marker.exists())
            self.assertEqual(
                (
                    refresh_dir / "_SUCCESS.superseded_01.json"
                ).read_bytes(),
                outer_bytes,
            )
            # A retry must never mutate or delete the completed target bank.
            self.assertEqual(inner_marker.read_bytes(), inner_bytes)
            self.assertEqual(manifest.read_bytes(), manifest_bytes)
            self.assertEqual(target.read_bytes(), target_bytes)
            self.assertTrue(bank_dir.is_dir())
            self.assertFalse(retry_dir.exists())


if __name__ == "__main__":
    unittest.main()
