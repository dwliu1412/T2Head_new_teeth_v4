from __future__ import annotations

import unittest
import torch

from surface_inpaint.uvd_surface_flow import UVDNoiseVolume


def make_volume(seed: int = 7) -> UVDNoiseVolume:
    return UVDNoiseVolume(
        uv_resolution=64,
        depth_resolution=2,
        layer_count=5,
        seed=seed,
        device=torch.device("cpu"),
    )


def test_same_semantic_uvd_cell_is_shared_across_views() -> None:
    volume = make_volume()
    uvd = torch.full((2, 3, 8, 8), 0.25)
    layers = torch.full((2, 1, 8, 8), 2)
    confidence = torch.ones(2, 1, 8, 8)

    noise, metrics = volume.sample(
        uvd, layers, confidence, latent_size=(1, 1)
    )

    torch.testing.assert_close(noise[0], noise[1], rtol=0.0, atol=0.0)
    assert metrics["uvd_flow_distinct_cells"].item() == 1.0
    assert metrics["uvd_flow_surface_fraction"].item() == 1.0


def test_underresolved_latent_footprint_is_downweighted() -> None:
    volume = make_volume()
    # All 64 renderer samples collapse to one canonical centre.  The sampled
    # tensor still has unit-scale marginal noise, but it is not the eight-cell
    # support required by the configured transport reliability gate.
    uvd = torch.full((1, 3, 8, 8), 0.25)
    layers = torch.ones(1, 1, 8, 8)
    confidence = torch.ones(1, 1, 8, 8)

    noise, metrics = volume.sample(
        uvd,
        layers,
        confidence,
        latent_size=(1, 1),
        minimum_distinct_cells=8,
    )

    torch.testing.assert_close(
        metrics["uvd_flow_transport_reliability"],
        torch.tensor(1.0 / 8.0),
    )
    torch.testing.assert_close(
        metrics["uvd_flow_cell_coverage"],
        torch.tensor(1.0 / 64.0),
    )
    assert torch.isfinite(noise).all()
    assert "_uvd_flow_transport_reliability" not in metrics


def test_complete_tile_keeps_layers_and_deduplicates_cells() -> None:
    volume = make_volume()
    volume.noise_volume.zero_()
    uvd = torch.tensor(
        [
            [
                [[0.10, 0.10], [0.10, 0.10]],
                [[0.20, 0.20], [0.20, 0.20]],
                [[0.25, 0.25], [0.25, 0.75]],
            ]
        ],
        dtype=torch.float32,
    )
    # The first two samples are duplicate layer-1 cells.  Equal numeric UV on
    # layers 2 and 3 must remain two distinct canonical surface identities.
    layers = torch.tensor([[[[1, 1], [2, 3]]]], dtype=torch.long)
    confidence = torch.ones(1, 1, 2, 2)
    cell_a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    cell_b = torch.tensor([5.0, 6.0, 7.0, 8.0])
    cell_c = torch.tensor([9.0, 10.0, 11.0, 12.0])
    u_index, v_index = 6, 12
    volume.noise_volume[1, :, 0, v_index, u_index] = cell_a
    volume.noise_volume[2, :, 0, v_index, u_index] = cell_b
    volume.noise_volume[3, :, 1, v_index, u_index] = cell_c

    noise, metrics = volume.sample(
        uvd, layers, confidence, latent_size=(1, 1)
    )

    expected = (cell_a + cell_b + cell_c) / torch.sqrt(torch.tensor(3.0))
    torch.testing.assert_close(noise[0, :, 0, 0], expected)
    assert metrics["uvd_flow_distinct_cells"].item() == 3.0


def test_normal_offset_is_part_of_the_canonical_noise_key() -> None:
    volume = make_volume()
    volume.noise_volume.zero_()
    uvd = torch.tensor(
        [[[[0.10, 0.10]], [[0.20, 0.20]], [[0.25, 0.75]]]],
        dtype=torch.float32,
    )
    layers = torch.ones(1, 1, 1, 2, dtype=torch.long)
    confidence = torch.ones(1, 1, 1, 2)
    first = torch.tensor([1.0, 2.0, 3.0, 4.0])
    second = torch.tensor([5.0, 6.0, 7.0, 8.0])
    volume.noise_volume[1, :, 0, 12, 6] = first
    volume.noise_volume[1, :, 1, 12, 6] = second

    noise, metrics = volume.sample(
        uvd, layers, confidence, latent_size=(1, 1)
    )

    expected = (first + second) / torch.sqrt(torch.tensor(2.0))
    torch.testing.assert_close(noise[0, :, 0, 0], expected)
    assert metrics["uvd_flow_distinct_cells"].item() == 2.0


def test_invalid_correspondence_uses_independent_iid_fallback() -> None:
    volume = make_volume()
    uvd = torch.full((2, 3, 8, 8), float("nan"))
    layers = torch.full((2, 1, 8, 8), -1)
    confidence = torch.full((2, 1, 8, 8), float("nan"))

    noise, metrics = volume.sample(
        uvd, layers, confidence, latent_size=(1, 1)
    )

    assert torch.isfinite(noise).all()
    assert not torch.equal(noise[0], noise[1])
    repeated, _ = volume.sample(
        uvd, layers, confidence, latent_size=(1, 1)
    )
    assert not torch.equal(repeated, noise)
    assert metrics["uvd_flow_surface_fraction"].item() == 0.0
    assert metrics["uvd_flow_surface_confidence"].item() == 0.0


def test_bchw_layout_with_unit_width_is_not_misread_as_bhwc() -> None:
    volume = make_volume()
    uvd = torch.full((1, 3, 2, 1), 0.25)
    layers = torch.ones(1, 1, 2, 1)
    confidence = torch.ones(1, 1, 2, 1)

    noise, _ = volume.sample(
        uvd, layers, confidence, latent_size=(1, 1)
    )

    assert noise.shape == (1, 4, 1, 1)


def test_checkpoint_restores_next_step_volume_and_rng() -> None:
    first = make_volume(seed=11)
    invalid_uvd = torch.zeros(1, 3, 8, 8)
    invalid_layer = torch.full((1, 1, 8, 8), -1)
    confidence = torch.zeros(1, 1, 8, 8)
    first.resample_for_step(0)
    first.sample(
        invalid_uvd, invalid_layer, confidence, latent_size=(1, 1)
    )
    state = first.state_dict()

    first.resample_for_step(1)
    expected_noise, _ = first.sample(
        invalid_uvd, invalid_layer, confidence, latent_size=(1, 1)
    )

    restored = UVDNoiseVolume(
        uv_resolution=64,
        depth_resolution=2,
        layer_count=5,
        seed=11,
        device=torch.device("cpu"),
        state=state,
    )
    restored.resample_for_step(1)
    actual_noise, _ = restored.sample(
        invalid_uvd, invalid_layer, confidence, latent_size=(1, 1)
    )
    torch.testing.assert_close(
        restored.noise_volume, first.noise_volume, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(actual_noise, expected_noise, rtol=0.0, atol=0.0)

    before = restored.noise_volume.clone()
    restored.resample_for_step(1)
    torch.testing.assert_close(
        restored.noise_volume, before, rtol=0.0, atol=0.0
    )
    restored.resample_for_step(2)
    assert not torch.equal(restored.noise_volume, before)
    with unittest.TestCase().assertRaisesRegex(ValueError, "backwards"):
        restored.resample_for_step(1)


def test_checkpoint_rejects_changed_volume_configuration() -> None:
    state = make_volume(seed=3).state_dict()
    with unittest.TestCase().assertRaisesRegex(ValueError, "seed"):
        UVDNoiseVolume(
            uv_resolution=64,
            depth_resolution=2,
            layer_count=5,
            seed=4,
            device=torch.device("cpu"),
            state=state,
        )


def test_background_call_count_cannot_change_next_canonical_draw() -> None:
    initial = make_volume(seed=19).state_dict()
    first = UVDNoiseVolume(
        uv_resolution=64,
        depth_resolution=2,
        layer_count=5,
        seed=19,
        device=torch.device("cpu"),
        state=initial,
    )
    second = UVDNoiseVolume(
        uv_resolution=64,
        depth_resolution=2,
        layer_count=5,
        seed=19,
        device=torch.device("cpu"),
        state=initial,
    )
    invalid_uvd = torch.zeros(1, 3, 8, 8)
    invalid_layer = torch.full((1, 1, 8, 8), -1)
    confidence = torch.zeros(1, 1, 8, 8)
    for _ in range(3):
        first.sample(
            invalid_uvd, invalid_layer, confidence, latent_size=(1, 1)
        )
    first.resample_for_step(0)
    second.resample_for_step(0)
    first.resample_for_step(1)
    second.resample_for_step(1)
    torch.testing.assert_close(
        first.noise_volume, second.noise_volume, rtol=0.0, atol=0.0
    )


def load_tests(
    _loader: unittest.TestLoader,
    _tests: unittest.TestSuite,
    _pattern: str | None,
) -> unittest.TestSuite:
    suite = unittest.TestSuite()
    for test in (
        test_same_semantic_uvd_cell_is_shared_across_views,
        test_underresolved_latent_footprint_is_downweighted,
        test_complete_tile_keeps_layers_and_deduplicates_cells,
        test_normal_offset_is_part_of_the_canonical_noise_key,
        test_invalid_correspondence_uses_independent_iid_fallback,
        test_bchw_layout_with_unit_width_is_not_misread_as_bhwc,
        test_checkpoint_restores_next_step_volume_and_rng,
        test_checkpoint_rejects_changed_volume_configuration,
        test_background_call_count_cannot_change_next_canonical_draw,
    ):
        suite.addTest(unittest.FunctionTestCase(test))
    return suite
