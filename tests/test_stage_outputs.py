import copy
import json
import tempfile
import unittest
from pathlib import Path

from surface_inpaint.pipeline import (
    LoopInpaintTrainer,
    load_config,
    validate_config,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "loop_inpaint.yaml"
)


class StageCompletionMarkerTests(unittest.TestCase):
    STAGE_ID = "stage_01_coarse"
    COMPLETED_STEPS = 120
    CONFIG_DIGEST = "test-config-sha256"

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)

        self.trainer = LoopInpaintTrainer.__new__(LoopInpaintTrainer)
        self.trainer.stages_dir = Path(self.temporary_directory.name)
        self.trainer.config_digest = self.CONFIG_DIGEST
        self.trainer.implementation_digest = "test-implementation-sha256"

    @property
    def marker_path(self):
        return self.trainer.stages_dir / self.STAGE_ID / "_SUCCESS.json"

    def write_marker(self, **overrides):
        marker = {
            "stage_id": self.STAGE_ID,
            "completed_steps": self.COMPLETED_STEPS,
            "config_sha256": self.CONFIG_DIGEST,
            "implementation_sha256": self.trainer.implementation_digest,
        }
        marker.update(overrides)
        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        self.marker_path.write_text(
            json.dumps(marker),
            encoding="utf-8",
        )

    def test_matching_marker_is_complete(self):
        self.write_marker()

        self.assertTrue(
            self.trainer._stage_is_complete(
                self.STAGE_ID,
                self.COMPLETED_STEPS,
            )
        )

    def test_stage_step_or_config_mismatch_is_not_complete(self):
        mismatches = {
            "stage_id": {"stage_id": "stage_02_refined"},
            "completed_steps": {
                "completed_steps": self.COMPLETED_STEPS + 1
            },
            "config_sha256": {"config_sha256": "different-config"},
            "implementation_sha256": {
                "implementation_sha256": "different-implementation"
            },
        }
        for label, override in mismatches.items():
            with self.subTest(mismatch=label):
                self.write_marker(**override)
                self.assertFalse(
                    self.trainer._stage_is_complete(
                        self.STAGE_ID,
                        self.COMPLETED_STEPS,
                    )
                )

    def test_corrupt_json_is_not_complete(self):
        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        self.marker_path.write_text("{not valid JSON", encoding="utf-8")

        self.assertFalse(
            self.trainer._stage_is_complete(
                self.STAGE_ID,
                self.COMPLETED_STEPS,
            )
        )


class StageOutputConfigValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid_config = load_config(CONFIG_PATH, ())
        validate_config(cls.valid_config, check_files=False)

    def test_stage_output_counts_must_be_positive(self):
        for key in ("diagnostic_views", "comparison_frames"):
            with self.subTest(key=key):
                config = copy.deepcopy(self.valid_config)
                config["stage_outputs"][key] = 0

                with self.assertRaisesRegex(
                    ValueError,
                    rf"stage_outputs\.{key} must be positive",
                ):
                    validate_config(config, check_files=False)

    def test_stage_output_pose_quantiles_must_be_nonempty_and_bounded(self):
        invalid_quantiles = (
            [],
            [-0.01],
            [1.01],
        )
        for quantiles in invalid_quantiles:
            with self.subTest(quantiles=quantiles):
                config = copy.deepcopy(self.valid_config)
                config["stage_outputs"][
                    "diagnostic_pose_quantiles"
                ] = quantiles

                with self.assertRaisesRegex(
                    ValueError,
                    "stage_outputs.diagnostic_pose_quantiles",
                ):
                    validate_config(config, check_files=False)


class StageIndexLayoutTests(unittest.TestCase):
    @staticmethod
    def make_trainer(root):
        trainer = LoopInpaintTrainer.__new__(LoopInpaintTrainer)
        trainer.directory = root
        trainer.stages_dir = root
        trainer.teacher_dir = root / "teacher"
        trainer.config_digest = "config"
        trainer.implementation_digest = "implementation"
        trainer.config = {
            "teacher": {
                "coarse_iterations": 10,
                "coarse_refresh_interval": 10,
                "refresh_interval": 10,
                "views_per_pose": 1,
                "coarse_poses_per_refresh": 1,
                "poses_per_refresh": 1,
            },
            "optimization": {"iterations": 20},
        }
        return trainer

    @staticmethod
    def write_stage_markers(trainer):
        expected_steps = {
            "00_stage1_input": 0,
            "01_geometry_stabilized": 0,
            "02_coherent_base": 10,
            "03_detail_refinement": 20,
        }
        for stage_id, completed_steps in expected_steps.items():
            directory = trainer.directory / stage_id
            directory.mkdir()
            (directory / "_SUCCESS.json").write_text(
                json.dumps(
                    {
                        "stage_id": stage_id,
                        "completed_steps": completed_steps,
                        "config_sha256": trainer.config_digest,
                        "implementation_sha256": (
                            trainer.implementation_digest
                        ),
                    }
                ),
                encoding="utf-8",
            )
        return expected_steps

    def test_numbered_stages_are_indexed_directly_under_run_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trainer = self.make_trainer(root)
            expected_steps = self.write_stage_markers(trainer)

            trainer._write_stage_index()

            index = json.loads(
                (root / "stage_index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(index["schema_version"], 3)
            self.assertEqual(
                index["stage_order"], list(expected_steps.keys())
            )
            self.assertTrue(
                all(stage["status"] == "complete" for stage in index["stages"])
            )
            self.assertTrue(
                all(
                    not stage["path"].startswith("stages/")
                    for stage in index["stages"]
                )
            )
            for stage in index["stages"]:
                artifacts = stage["artifacts"]
                self.assertTrue(
                    artifacts["oral_correspondence"].endswith(
                        "diagnostics/oral_correspondence_grid.jpg"
                    )
                )
                self.assertTrue(
                    artifacts[
                        "oral_appearance_contribution"
                    ].endswith(
                        "diagnostics/"
                        "oral_appearance_contribution_grid.jpg"
                    )
                )

    def test_run_success_requires_and_records_all_stage_and_teacher_markers(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trainer = self.make_trainer(root)
            self.write_stage_markers(trainer)
            for refresh_step in (0, 10):
                refresh = (
                    trainer.teacher_dir / f"step_{refresh_step:06d}"
                )
                refresh.mkdir(parents=True)
                (refresh / "_SUCCESS.json").write_text(
                    json.dumps(
                        {
                            "refresh_step": refresh_step,
                            "processed_observations": 1,
                            "config_sha256": trainer.config_digest,
                            "implementation_sha256": (
                                trainer.implementation_digest
                            ),
                        }
                    ),
                    encoding="utf-8",
                )
            for name in (
                "geometry_stability_comparison.jpg",
                "driving_comparison.jpg",
                "diagnostic_comparison.jpg",
                "metrics_summary.json",
            ):
                (root / name).write_bytes(name.encode("utf-8"))

            trainer._write_run_success(20)

            success = json.loads(
                (root / "_RUN_SUCCESS.json").read_text(encoding="utf-8")
            )
            self.assertEqual(success["teacher_refresh_count"], 2)
            self.assertEqual(set(success["stages"]), {
                "00_stage1_input",
                "01_geometry_stabilized",
                "02_coherent_base",
                "03_detail_refinement",
            })


if __name__ == "__main__":
    unittest.main()
