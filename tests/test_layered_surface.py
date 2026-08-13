import unittest

import torch

from surface_inpaint.layered_surface import (
    INVALID_SURFACE_LAYER_ID,
    LayerSurfaceBuffers,
    SURFACE_LAYER_IDS,
    SURFACE_LAYER_NAMES,
    SurfaceLayerId,
    compose_layered_surface,
    decode_surface_rgb_residual,
    encode_surface_rgb_residual,
    normalize_alpha_weighted,
    validate_layer_names,
)


def layer_buffers(
    *,
    uv,
    alpha,
    contribution,
    depth=None,
    variance=None,
):
    uv_tensor = torch.as_tensor(uv, dtype=torch.float32)
    if uv_tensor.ndim == 3:
        uv_tensor = uv_tensor.unsqueeze(0)
    batch, _, height, width = uv_tensor.shape

    def scalar(value, default):
        if value is None:
            value = default
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.numel() == 1:
            tensor = tensor.expand(batch, 1, height, width).clone()
        elif tensor.ndim == 1 and tensor.numel() == width:
            tensor = tensor.reshape(1, 1, 1, width).expand(
                batch, 1, height, width
            ).clone()
        return tensor

    return LayerSurfaceBuffers(
        uv=uv_tensor,
        variance=scalar(variance, 0.0),
        depth=scalar(depth, 1.0),
        alpha=scalar(alpha, 1.0),
        contribution=scalar(contribution, 1.0),
    )


class LayerDefinitionTests(unittest.TestCase):
    def test_fixed_layer_names_and_ids(self):
        self.assertEqual(
            SURFACE_LAYER_NAMES,
            (
                "face",
                "lips",
                "teeth_upper",
                "teeth_lower",
                "oral_cavity",
            ),
        )
        self.assertEqual(
            SURFACE_LAYER_IDS["teeth_upper"],
            int(SurfaceLayerId.TEETH_UPPER),
        )
        self.assertEqual(
            validate_layer_names(
                ["oral_cavity", "face", "teeth_lower"]
            ),
            ("face", "teeth_lower", "oral_cavity"),
        )
        with self.assertRaisesRegex(ValueError, "Unknown surface layer"):
            validate_layer_names(["tongue"])
        with self.assertRaisesRegex(ValueError, "must be unique"):
            validate_layer_names(["lips", "lips"])


class LayeredSurfaceCompositionTests(unittest.TestCase):
    def test_upper_and_lower_teeth_with_equal_uv_remain_separate(self):
        # Both dental layers deliberately use exactly the same UV in both
        # pixels. Only the semantic ID can keep their correspondences apart.
        shared_uv = torch.tensor(
            [[[[0.5, 0.5]], [[0.98, 0.98]]]],
            dtype=torch.float32,
        )
        result = compose_layered_surface(
            {
                "teeth_upper": layer_buffers(
                    uv=shared_uv,
                    alpha=[1.0, 1.0],
                    contribution=[0.9, 0.1],
                ),
                "teeth_lower": layer_buffers(
                    uv=shared_uv,
                    alpha=[1.0, 1.0],
                    contribution=[0.1, 0.9],
                ),
            },
            dominance_ratio=1.25,
        )

        torch.testing.assert_close(result.surface_uv, shared_uv)
        self.assertEqual(
            result.layer_id.flatten().tolist(),
            [
                int(SurfaceLayerId.TEETH_UPPER),
                int(SurfaceLayerId.TEETH_LOWER),
            ],
        )
        self.assertEqual(result.validity.flatten().tolist(), [1.0, 1.0])

    def test_equal_contributions_are_rejected_as_ambiguous(self):
        shared_uv = torch.tensor(
            [[[[0.5]], [[0.98]]]],
            dtype=torch.float32,
        )
        result = compose_layered_surface(
            {
                "teeth_upper": layer_buffers(
                    uv=shared_uv,
                    alpha=1.0,
                    contribution=0.5,
                ),
                "teeth_lower": layer_buffers(
                    uv=shared_uv,
                    alpha=1.0,
                    contribution=0.5,
                ),
            },
            dominance_ratio=1.25,
        )

        self.assertTrue(result.ambiguous.item())
        self.assertEqual(result.validity.item(), 0.0)
        self.assertEqual(
            result.layer_id.item(), INVALID_SURFACE_LAYER_ID
        )
        self.assertTrue(
            torch.equal(
                result.surface_uv, torch.zeros_like(result.surface_uv)
            )
        )

    def test_alpha_threshold_boundary_is_explicit(self):
        uv = torch.tensor(
            [[[[0.25, 0.75]], [[0.25, 0.75]]]],
            dtype=torch.float32,
        )
        result = compose_layered_surface(
            {
                "lips": layer_buffers(
                    uv=uv,
                    alpha=[0.2, 0.2001],
                    contribution=[0.2, 0.2001],
                )
            },
            alpha_threshold=0.2,
        )

        self.assertEqual(result.validity.flatten().tolist(), [0.0, 1.0])
        self.assertEqual(
            result.layer_id.flatten().tolist(),
            [INVALID_SURFACE_LAYER_ID, int(SurfaceLayerId.LIPS)],
        )


class AlphaNormalizationTests(unittest.TestCase):
    def test_zero_and_epsilon_alpha_do_not_create_large_values(self):
        epsilon = 1.0e-4
        alpha = torch.tensor(
            [[[[0.0, epsilon, 2.0 * epsilon, 1.0]]]],
            dtype=torch.float32,
        )
        expected_value = torch.tensor(
            [[[[2.0, 2.0, 2.0, 2.0]]]],
            dtype=torch.float32,
        )
        premultiplied = expected_value * alpha

        normalized = normalize_alpha_weighted(
            premultiplied,
            alpha,
            alpha_epsilon=epsilon,
        )

        torch.testing.assert_close(
            normalized,
            torch.tensor(
                [[[[0.0, 0.0, 2.0, 2.0]]]],
                dtype=torch.float32,
            ),
        )
        self.assertTrue(torch.isfinite(normalized).all())


class SurfaceResidualTests(unittest.TestCase):
    def test_round_trip_recovers_teacher_when_not_clipped(self):
        current = torch.tensor(
            [[[[0.2]], [[0.4]], [[0.6]]]],
            dtype=torch.float32,
        )
        teacher = torch.tensor(
            [[[[0.3]], [[0.1]], [[0.8]]]],
            dtype=torch.float32,
        )

        encoded = encode_surface_rgb_residual(
            teacher,
            current,
            torch.ones(1, 1, 1, 1),
            contribution_floor=0.01,
        )
        decoded = decode_surface_rgb_residual(encoded, current)

        torch.testing.assert_close(decoded, teacher)

    def test_unchanged_composite_decodes_to_layer_reference(self):
        screen_rgb = torch.tensor(
            [[[[0.8]], [[0.1]], [[0.2]]]],
            dtype=torch.float32,
        )
        tooth_reference = torch.tensor(
            [[[[0.9]], [[0.9]], [[0.85]]]],
            dtype=torch.float32,
        )

        encoded = encode_surface_rgb_residual(
            screen_rgb,
            screen_rgb,
            torch.full((1, 1, 1, 1), 0.01),
            contribution_floor=0.01,
        )
        decoded = decode_surface_rgb_residual(
            encoded,
            tooth_reference,
        )

        torch.testing.assert_close(decoded, tooth_reference)
        torch.testing.assert_close(
            encoded,
            torch.full_like(encoded, 0.5),
        )

    def test_per_gaussian_rgb_layout_is_supported(self):
        encoded = torch.full((4, 3), 0.5, dtype=torch.float32)
        reference = torch.rand(4, 3)

        decoded = decode_surface_rgb_residual(encoded, reference)

        torch.testing.assert_close(decoded, reference)

    def test_weak_visible_layer_is_decomposited_before_encoding(self):
        current = torch.full((1, 3, 1, 1), 0.4)
        teacher = torch.full((1, 3, 1, 1), 0.405)
        contribution = torch.full((1, 1, 1, 1), 0.01)

        encoded = encode_surface_rgb_residual(
            teacher,
            current,
            contribution,
            contribution_floor=0.01,
        )
        intrinsic_target = decode_surface_rgb_residual(encoded, current)

        torch.testing.assert_close(
            encoded,
            torch.full_like(encoded, 0.75),
        )
        torch.testing.assert_close(
            intrinsic_target,
            torch.full_like(intrinsic_target, 0.9),
        )

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            encode_surface_rgb_residual(
                torch.zeros(1, 3, 2, 2),
                torch.zeros(1, 3, 1, 2),
                torch.zeros(1, 1, 2, 2),
                contribution_floor=0.01,
            )


if __name__ == "__main__":
    unittest.main()
