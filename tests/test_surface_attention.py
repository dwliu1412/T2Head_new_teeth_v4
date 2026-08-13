import unittest

import torch

from surface_inpaint.attention import (
    SurfaceAttentionConfig,
    SurfaceCorrespondenceAttnProcessor,
    install_surface_attention,
)


class IdentityProcessor:
    def __init__(self):
        self.calls = 0
        self.last_kwargs = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        self.calls += 1
        self.last_kwargs = kwargs
        return hidden_states


class AffineProcessor:
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        return hidden_states.mul(1.25).add(0.125)


class FakeUNet:
    def __init__(self):
        self._processors = {
            "down.0.transformer.attn1.processor": IdentityProcessor(),
            "down.0.transformer.attn2.processor": IdentityProcessor(),
            "mid.transformer.attn1.processor": IdentityProcessor(),
        }

    @property
    def attn_processors(self):
        return self._processors

    def set_attn_processor(self, processors):
        self._processors = dict(processors)


def make_uv(batch, height, width):
    u = torch.linspace(0.0, 1.0, width)
    v = torch.linspace(0.0, 1.0, height)
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
    uv = torch.stack((grid_u, grid_v), dim=0)
    return uv.unsqueeze(0).repeat(batch, 1, 1, 1)


class SurfaceAttentionTests(unittest.TestCase):
    def test_install_wraps_only_attn1_and_uninstall_restores(self):
        unet = FakeUNet()
        original = dict(unet.attn_processors)
        controller = install_surface_attention(
            unet,
            SurfaceAttentionConfig(
                atlas_resolution=8,
                max_tokens=64,
                min_views=2,
                strength=1.0,
            ),
        )

        self.assertTrue(controller.installed)
        self.assertIsInstance(
            unet.attn_processors[
                "down.0.transformer.attn1.processor"
            ],
            SurfaceCorrespondenceAttnProcessor,
        )
        self.assertIs(
            unet.attn_processors[
                "down.0.transformer.attn2.processor"
            ],
            original["down.0.transformer.attn2.processor"],
        )
        self.assertEqual(len(controller.wrapped_processor_names), 2)

        controller.uninstall()
        self.assertFalse(controller.installed)
        for name, processor in original.items():
            self.assertIs(unet.attn_processors[name], processor)

    def test_no_context_is_bitwise_identical_to_base_processor(self):
        unet = FakeUNet()
        base = AffineProcessor()
        unet._processors[
            "down.0.transformer.attn1.processor"
        ] = base
        controller = install_surface_attention(unet)
        wrapped = unet.attn_processors[
            "down.0.transformer.attn1.processor"
        ]
        hidden = torch.randn(2, 3, 4, 5)

        expected = base(None, hidden)
        actual = wrapped(None, hidden)
        self.assertTrue(torch.equal(actual, expected))

        controller.set_context(
            make_uv(2, 4, 5),
            torch.ones(2, 1, 4, 5),
            denoise_progress=1.0,
        )
        controller.clear_context()
        cleared = wrapped(None, hidden)
        self.assertTrue(torch.equal(cleared, expected))

    def test_confidence_weighted_fusion_for_4d_states(self):
        unet = FakeUNet()
        controller = install_surface_attention(
            unet,
            {
                "atlas_resolution": 4,
                "max_tokens": 32,
                "min_views": 2,
                "strength": 1.0,
            },
        )
        hidden = torch.tensor(
            [
                [[[1.0, 10.0]]],
                [[[3.0, 30.0]]],
            ]
        )
        visibility = torch.tensor(
            [
                [[[0.25, 0.25]]],
                [[[0.75, 0.75]]],
            ]
        )
        controller.set_context(
            make_uv(2, 1, 2),
            visibility,
            denoise_progress=1.0,
        )
        processor = unet.attn_processors[
            "down.0.transformer.attn1.processor"
        ]
        output = processor(None, hidden)
        expected = torch.tensor(
            [
                [[[2.5, 25.0]]],
                [[[2.5, 25.0]]],
            ]
        )
        torch.testing.assert_close(output, expected)
        self.assertEqual(output.dtype, hidden.dtype)
        self.assertEqual(output.device, hidden.device)

    def test_layer_ids_keep_equal_uv_coordinates_semantically_isolated(self):
        unet = FakeUNet()
        controller = install_surface_attention(
            unet,
            SurfaceAttentionConfig(
                atlas_resolution=4,
                max_tokens=32,
                min_views=2,
                strength=1.0,
            ),
        )
        hidden = torch.tensor(
            [
                [[[1.0, 10.0]]],
                [[[3.0, 30.0]]],
            ]
        )
        # Every token deliberately shares one UV coordinate. Without semantic
        # keys all four values collapse into one memory; with layer IDs the
        # two oral surfaces retain independent cross-view memories.
        uv = torch.full((2, 2, 1, 2), 0.5)
        layer_ids = torch.tensor(
            [
                [[[0, 1]]],
                [[[0, 1]]],
            ],
            dtype=torch.long,
        )
        controller.set_context(
            uv,
            torch.ones(2, 1, 1, 2),
            layer_ids=layer_ids,
            denoise_progress=1.0,
        )
        processor = unet.attn_processors[
            "down.0.transformer.attn1.processor"
        ]

        output = processor(None, hidden)

        torch.testing.assert_close(
            output,
            torch.tensor(
                [
                    [[[2.0, 20.0]]],
                    [[[2.0, 20.0]]],
                ]
            ),
        )

    def test_negative_layer_id_is_not_propagated(self):
        unet = FakeUNet()
        controller = install_surface_attention(
            unet,
            SurfaceAttentionConfig(
                atlas_resolution=4,
                max_tokens=32,
                min_views=2,
                strength=1.0,
            ),
        )
        hidden = torch.tensor([[[[1.0]]], [[[3.0]]]])
        controller.set_context(
            make_uv(2, 1, 1),
            torch.ones(2, 1, 1, 1),
            layer_ids=-torch.ones(2, 1, 1, 1, dtype=torch.long),
            denoise_progress=1.0,
        )
        processor = unet.attn_processors[
            "down.0.transformer.attn1.processor"
        ]

        output = processor(None, hidden)

        self.assertTrue(torch.equal(output, hidden))

    def test_layer_ids_must_be_integer_and_match_uv_shape(self):
        controller = install_surface_attention(FakeUNet())
        uv = make_uv(2, 2, 2)
        visibility = torch.ones(2, 1, 2, 2)

        with self.assertRaisesRegex(TypeError, "integer dtype"):
            controller.set_context(
                uv,
                visibility,
                layer_ids=torch.zeros(2, 1, 2, 2),
            )
        with self.assertRaisesRegex(ValueError, "matching uv"):
            controller.set_context(
                uv,
                visibility,
                layer_ids=torch.zeros(2, 1, 1, 2, dtype=torch.long),
            )

    def test_gate_increases_with_denoise_progress(self):
        unet = FakeUNet()
        controller = install_surface_attention(
            unet,
            SurfaceAttentionConfig(
                atlas_resolution=4,
                max_tokens=16,
                min_views=2,
                strength=1.0,
            ),
        )
        hidden = torch.tensor([[[[0.0]]], [[[4.0]]]])
        uv = make_uv(2, 1, 1)
        visibility = torch.ones(2, 1, 1, 1)
        controller.set_context(
            uv,
            visibility,
            denoise_progress=0.25,
        )
        processor = unet.attn_processors[
            "down.0.transformer.attn1.processor"
        ]
        early = processor(None, hidden)
        torch.testing.assert_close(
            early, torch.tensor([[[[0.5]]], [[[3.5]]]])
        )

        controller.set_denoise_progress(1.0)
        late = processor(None, hidden)
        torch.testing.assert_close(
            late, torch.tensor([[[[2.0]]], [[[2.0]]]])
        )

    def test_cfg_unconditional_and_conditional_memories_are_isolated(self):
        unet = FakeUNet()
        controller = install_surface_attention(
            unet,
            SurfaceAttentionConfig(
                atlas_resolution=4,
                max_tokens=32,
                min_views=2,
                strength=1.0,
            ),
        )
        # Standard diffusers CFG layout:
        # [uncond_view_0, uncond_view_1, cond_view_0, cond_view_1].
        hidden = torch.tensor(
            [[[0.0]], [[2.0]], [[100.0]], [[104.0]]]
        )
        logical_uv = make_uv(2, 1, 1)
        controller.set_context(
            logical_uv,
            torch.ones(2, 1, 1, 1),
            denoise_progress=1.0,
            cfg_branches=2,
            cfg_layout="chunked",
        )
        processor = unet.attn_processors[
            "down.0.transformer.attn1.processor"
        ]
        output = processor(None, hidden)
        expected = torch.tensor(
            [[[1.0]], [[1.0]], [[102.0]], [[102.0]]]
        )
        torch.testing.assert_close(output, expected)

    def test_3d_non_square_tokens_and_minimum_view_gate(self):
        unet = FakeUNet()
        controller = install_surface_attention(
            unet,
            SurfaceAttentionConfig(
                atlas_resolution=16,
                max_tokens=64,
                min_views=2,
                strength=1.0,
            ),
        )
        hidden = torch.stack(
            (
                torch.arange(6, dtype=torch.float32),
                torch.arange(10, 16, dtype=torch.float32),
            ),
            dim=0,
        ).unsqueeze(-1)
        controller.set_context(
            make_uv(2, 2, 3),
            torch.ones(2, 1, 2, 3),
            denoise_progress=1.0,
        )
        processor = unet.attn_processors[
            "mid.transformer.attn1.processor"
        ]
        output = processor(None, hidden)
        expected = (
            torch.arange(5, 11, dtype=torch.float32)
            .reshape(1, 6, 1)
            .repeat(2, 1, 1)
        )
        torch.testing.assert_close(output, expected)
        self.assertEqual(tuple(output.shape), (2, 6, 1))

        controller.uninstall()
        controller = install_surface_attention(
            unet,
            SurfaceAttentionConfig(
                atlas_resolution=16,
                max_tokens=64,
                min_views=3,
                strength=1.0,
            ),
        )
        controller.set_context(
            make_uv(2, 2, 3),
            torch.ones(2, 1, 2, 3),
            denoise_progress=1.0,
        )
        unchanged = unet.attn_processors[
            "mid.transformer.attn1.processor"
        ](None, hidden)
        self.assertTrue(torch.equal(unchanged, hidden))

    def test_cross_attention_call_is_not_modified(self):
        unet = FakeUNet()
        controller = install_surface_attention(unet)
        controller.set_context(
            make_uv(2, 1, 2),
            torch.ones(2, 1, 1, 2),
            denoise_progress=1.0,
        )
        hidden = torch.randn(2, 2, 3)
        encoder = torch.randn(2, 4, 3)
        processor = unet.attn_processors[
            "down.0.transformer.attn1.processor"
        ]
        output = processor(
            None, hidden, encoder_hidden_states=encoder
        )
        self.assertTrue(torch.equal(output, hidden))

    def test_runtime_diagnostics_record_surface_attention_execution(self):
        unet = FakeUNet()
        controller = install_surface_attention(unet)
        controller.set_context(
            make_uv(2, 1, 2),
            torch.ones(2, 1, 1, 2),
            denoise_progress=0.0,
        )
        controller.set_denoise_progress(0.5)
        hidden = torch.randn(2, 2, 3)
        unet.attn_processors[
            "down.0.transformer.attn1.processor"
        ](None, hidden)

        diagnostics = controller.diagnostics()
        self.assertTrue(diagnostics["installed"])
        self.assertEqual(diagnostics["wrapped_processors"], 2)
        self.assertEqual(diagnostics["contexts_set"], 1)
        self.assertEqual(diagnostics["self_attention_calls"], 1)
        self.assertEqual(diagnostics["denoise_progress_updates"], 1)
        self.assertEqual(diagnostics["maximum_joint_views"], 2)
        self.assertEqual(diagnostics["visible_surface_tokens"], 4)

        restored = install_surface_attention(FakeUNet())
        restored.load_diagnostics(diagnostics)
        restored_diagnostics = restored.diagnostics()
        for key in (
            "contexts_set",
            "self_attention_calls",
            "denoise_progress_updates",
            "maximum_joint_views",
            "visible_surface_tokens",
        ):
            self.assertEqual(restored_diagnostics[key], diagnostics[key])


if __name__ == "__main__":
    unittest.main()
