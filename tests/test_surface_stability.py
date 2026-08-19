import json
import unittest

import torch

from surface_inpaint.stability import (
    StabilityConfig,
    cap_world_covariance,
    covariance_from_scaling_rotation,
    pullback_covariance,
    pushforward_covariance,
    stabilize_uvd_covariances,
    world_principal_scales,
)


DTYPE = torch.float64


def diagonal_covariance(scales):
    values = torch.tensor(scales, dtype=DTYPE)
    return torch.diag_embed(values.square())


class FakeFaceLocalModel:
    def __init__(self, canonical_scales, transforms):
        scales = torch.tensor(canonical_scales, dtype=DTYPE)
        identity_quaternion = torch.zeros(
            (scales.shape[0], 4), dtype=DTYPE
        )
        identity_quaternion[:, 0] = 1.0
        self._scaling = torch.nn.Parameter(scales.log())
        self._rotation = torch.nn.Parameter(identity_quaternion)
        self.scaling_inverse_activation = torch.log
        self.transforms = {
            key: value.to(dtype=DTYPE) for key, value in transforms.items()
        }
        self.current_pose = None

    @property
    def get_scaling(self):
        return self._scaling.exp()

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation, dim=-1)

    @property
    def get_local_scaling(self):
        return self.get_scaling

    @property
    def get_local_rotation(self):
        return self.get_rotation

    def current_covariance_transform(self):
        return self.transforms[self.current_pose]


class SurfaceStabilityMathTests(unittest.TestCase):
    def test_thin_disk_is_not_misclassified_as_planar_streak(self):
        # s0/s2 is 400, but the in-plane ratio s0/s1 is exactly one.
        canonical = diagonal_covariance([[0.040, 0.040, 0.0001]])
        jacobian = torch.eye(3, dtype=DTYPE).unsqueeze(0)
        result = cap_world_covariance(
            canonical,
            jacobian,
            StabilityConfig(
                absolute_max_scale=0.100,
                min_streak_scale=0.020,
                max_planar_aspect=10.0,
            ),
        )

        self.assertFalse(bool(result.flagged.item()))
        self.assertTrue(torch.equal(result.canonical_covariance, canonical))
        torch.testing.assert_close(
            result.before_world_scales,
            torch.tensor([[0.040, 0.040, 0.0001]], dtype=DTYPE),
        )

    def test_long_in_plane_gaussian_is_capped_without_changing_s1_s2(self):
        canonical = diagonal_covariance([[0.040, 0.001, 0.0005]])
        jacobian = torch.eye(3, dtype=DTYPE).unsqueeze(0)
        result = cap_world_covariance(
            canonical,
            jacobian,
            StabilityConfig(
                absolute_max_scale=0.100,
                min_streak_scale=0.020,
                max_planar_aspect=10.0,
            ),
        )

        self.assertTrue(bool(result.flagged.item()))
        self.assertFalse(bool(result.absolute_violations.item()))
        self.assertTrue(bool(result.streak_violations.item()))
        expected = torch.tensor(
            [[0.010, 0.001, 0.0005]], dtype=DTYPE
        )
        torch.testing.assert_close(result.after_world_scales, expected)
        actual, _ = world_principal_scales(
            result.canonical_covariance, jacobian
        )
        torch.testing.assert_close(actual, expected)

    def test_absolute_limit_also_caps_a_large_round_disk(self):
        canonical = diagonal_covariance([[0.200, 0.180, 0.0005]])
        jacobian = torch.eye(3, dtype=DTYPE).unsqueeze(0)
        result = cap_world_covariance(
            canonical,
            jacobian,
            StabilityConfig(
                absolute_max_scale=0.190,
                min_streak_scale=0.020,
                max_planar_aspect=10.0,
            ),
        )

        self.assertTrue(bool(result.flagged.item()))
        self.assertTrue(bool(result.absolute_violations.item()))
        self.assertFalse(bool(result.streak_violations.item()))
        torch.testing.assert_close(
            result.after_world_scales,
            torch.tensor([[0.190, 0.180, 0.0005]], dtype=DTYPE),
        )

    def test_aggressive_absolute_cap_limits_every_giant_axis(self):
        canonical = diagonal_covariance([[0.200, 0.180, 0.0005]])
        jacobian = torch.eye(3, dtype=DTYPE).unsqueeze(0)
        result = cap_world_covariance(
            canonical,
            jacobian,
            StabilityConfig(
                absolute_max_scale=0.100,
                absolute_cap=0.006,
                min_streak_scale=0.020,
                max_planar_aspect=10.0,
            ),
        )

        self.assertTrue(bool(result.absolute_violations.item()))
        torch.testing.assert_close(
            result.after_world_scales,
            torch.tensor([[0.006, 0.006, 0.0005]], dtype=DTYPE),
        )

    def test_world_pullback_repair_pushforward_preserves_axes_and_minor_scales(
        self,
    ):
        angle = torch.tensor(torch.pi / 4.0, dtype=DTYPE)
        cosine, sine = torch.cos(angle), torch.sin(angle)
        rotation = torch.tensor(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=DTYPE,
        ).unsqueeze(0)
        before = torch.tensor([[0.050, 0.004, 0.001]], dtype=DTYPE)
        world = torch.bmm(
            rotation * before.square()[:, None, :],
            rotation.transpose(1, 2),
        )
        jacobian = torch.tensor(
            [[[2.0, 0.1, 0.0], [0.0, 0.5, 0.2], [0.1, 0.0, 1.5]]],
            dtype=DTYPE,
        )
        canonical = pullback_covariance(world, jacobian)
        torch.testing.assert_close(
            pushforward_covariance(canonical, jacobian),
            world,
            rtol=1.0e-10,
            atol=1.0e-12,
        )

        result = cap_world_covariance(
            canonical,
            jacobian,
            StabilityConfig(
                absolute_max_scale=0.030,
                min_streak_scale=0.010,
                max_planar_aspect=5.0,
            ),
        )
        expected_scales = torch.tensor(
            [[0.020, 0.004, 0.001]], dtype=DTYPE
        )
        expected_world = torch.bmm(
            rotation * expected_scales.square()[:, None, :],
            rotation.transpose(1, 2),
        )
        repaired_world = pushforward_covariance(
            result.canonical_covariance, jacobian
        )
        torch.testing.assert_close(
            repaired_world,
            expected_world,
            rtol=1.0e-9,
            atol=1.0e-12,
        )


class SurfaceStabilityModelTests(unittest.TestCase):
    def test_pose_envelope_repairs_dynamic_streak_and_restores_reference(self):
        identity = torch.eye(3, dtype=DTYPE).unsqueeze(0)
        stretched = torch.diag(
            torch.tensor([5.0, 1.0, 1.0], dtype=DTYPE)
        ).unsqueeze(0)
        model = FakeFaceLocalModel(
            [[0.010, 0.009, 0.001]],
            {"reference": identity, "stretched": stretched},
        )
        calls = []

        def set_pose(pose):
            calls.append(pose)
            model.current_pose = pose

        report = stabilize_uvd_covariances(
            model=model,
            named_poses=(
                ("reference", "reference"),
                ("jaw_open", "stretched"),
            ),
            set_pose=set_pose,
            reference_pose="reference",
            config={
                "passes": 3,
                "absolute_max_scale": 0.100,
                "min_streak_scale": 0.020,
                "max_planar_aspect": 4.0,
            },
        )

        self.assertEqual(calls[-1], "reference")
        self.assertEqual(model.current_pose, "reference")
        self.assertEqual(report["before"]["flagged_unique"], 1)
        self.assertEqual(
            report["before"]["poses"][0]["flagged"], 0
        )
        self.assertEqual(
            report["before"]["poses"][1]["flagged"], 1
        )
        self.assertEqual(report["after"]["flagged_unique"], 0)
        self.assertEqual(report["unique_updated"], 1)
        self.assertGreaterEqual(report["passes_completed"], 1)
        json.dumps(report, allow_nan=False)

        canonical = covariance_from_scaling_rotation(
            model.get_scaling, model.get_rotation
        )
        stretched_scales, _ = world_principal_scales(
            canonical, stretched
        )
        self.assertLessEqual(
            float(
                stretched_scales[0, 0]
                / stretched_scales[0, 1]
            ),
            4.0 + 1.0e-10,
        )


if __name__ == "__main__":
    unittest.main()
