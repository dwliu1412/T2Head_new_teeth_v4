import unittest

import torch

from surface_inpaint.pipeline import SurfaceNoiseAtlas


class LayeredSurfaceNoiseAtlasTests(unittest.TestCase):
    DEVICE = torch.device("cpu")
    LAYER_COUNT = 5

    def test_equal_uv_samples_distinct_canonical_noise_per_layer(self):
        noise = SurfaceNoiseAtlas(
            resolution=4,
            seed=17,
            device=self.DEVICE,
            layer_count=self.LAYER_COUNT,
        )
        with torch.no_grad():
            for layer_id in range(self.LAYER_COUNT):
                noise.atlas[layer_id].fill_(float(layer_id + 1))

        surface_uv = torch.zeros(2, 2, 1, 1)
        alpha = torch.ones(2, 1, 1, 1)
        layer_ids = torch.tensor(
            [
                [[[2]]],
                [[[3]]],
            ],
            dtype=torch.long,
        )

        sampled = noise.sample(
            surface_uv,
            alpha,
            latent_size=(1, 1),
            layer_ids=layer_ids,
        )

        torch.testing.assert_close(
            sampled[0],
            torch.full((4, 1, 1), 3.0),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            sampled[1],
            torch.full((4, 1, 1), 4.0),
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(torch.equal(sampled[0], sampled[1]))

    def test_negative_layer_id_uses_finite_background_noise(self):
        seed = 29
        noise = SurfaceNoiseAtlas(
            resolution=4,
            seed=seed,
            device=self.DEVICE,
            layer_count=self.LAYER_COUNT,
        )
        with torch.no_grad():
            # An invalid layer must not leak any canonical atlas value.
            noise.atlas.fill_(1234.0)

        sampled = noise.sample(
            torch.full((1, 2, 1, 1), 0.5),
            torch.ones(1, 1, 1, 1),
            latent_size=(1, 1),
            layer_ids=-torch.ones(1, 1, 1, 1, dtype=torch.long),
        )
        generator = torch.Generator(device=self.DEVICE)
        generator.manual_seed(seed + 1_000_003)
        expected_background = torch.randn(
            sampled.shape,
            generator=generator,
            device=self.DEVICE,
            dtype=sampled.dtype,
        )

        torch.testing.assert_close(
            sampled,
            expected_background,
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(torch.isfinite(sampled).all())
        self.assertEqual(noise.background_counter, 1)

    def test_state_dict_round_trip_requires_five_layer_atlas_shape(self):
        noise = SurfaceNoiseAtlas(
            resolution=8,
            seed=41,
            device=self.DEVICE,
            layer_count=self.LAYER_COUNT,
        )

        state = noise.state_dict()

        self.assertEqual(tuple(state["atlas"].shape), (5, 4, 8, 8))
        self.assertEqual(state["layer_count"], 5)
        restored = SurfaceNoiseAtlas(
            resolution=8,
            seed=41,
            device=self.DEVICE,
            layer_count=self.LAYER_COUNT,
            state=state,
        )
        self.assertTrue(torch.equal(restored.atlas, noise.atlas))

        invalid_state = dict(state)
        invalid_state["atlas"] = state["atlas"][:4]
        with self.assertRaisesRegex(ValueError, "checkpoint shape differs"):
            SurfaceNoiseAtlas(
                resolution=8,
                seed=41,
                device=self.DEVICE,
                layer_count=self.LAYER_COUNT,
                state=invalid_state,
            )


if __name__ == "__main__":
    unittest.main()
