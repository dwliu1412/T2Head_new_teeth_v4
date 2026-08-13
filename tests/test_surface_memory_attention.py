import unittest

import torch
from torch import nn
from torch.nn import functional as F

from surface_inpaint.surface_memory_attention import (
    SurfaceMemoryAttnProcessor2_0,
    SurfaceMemoryConfig,
    install_surface_memory_attention,
)


class TinyAttnProcessor2_0:
    """Small CPU equivalent of the relevant diffusers 0.34 processor."""

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
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch, channels, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch, channels, height * width
            ).transpose(1, 2)
        batch = hidden_states.shape[0]
        source = (
            hidden_states
            if encoder_hidden_states is None
            else encoder_hidden_states
        )
        query = attn.to_q(hidden_states)
        key = attn.to_k(source)
        value = attn.to_v(source)
        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch, -1, attn.heads, head_dim).transpose(
            1, 2
        )
        key = key.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch, -1, attn.heads, head_dim).transpose(
            1, 2
        )
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch, -1, attn.heads * head_dim
        )
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch, channels, height, width
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


class TinyAttention(nn.Module):
    def __init__(self, channels=1, heads=1):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(channels, channels, bias=False)
        self.to_v = nn.Linear(channels, channels, bias=False)
        self.to_out = nn.ModuleList(
            (nn.Linear(channels, channels, bias=False), nn.Identity())
        )
        self.spatial_norm = None
        self.group_norm = None
        self.norm_q = None
        self.norm_k = None
        self.residual_connection = False
        self.rescale_output_factor = 1.0
        with torch.no_grad():
            for projection in (
                self.to_q,
                self.to_k,
                self.to_v,
                self.to_out[0],
            ):
                projection.weight.copy_(torch.eye(channels))


class AffineProcessor:
    def __init__(self):
        self.calls = 0

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
        return hidden_states.mul(1.25).add(0.125)


class FakeUNet:
    def __init__(self, processor_factory=TinyAttnProcessor2_0):
        self._processors = {
            "down_blocks.0.block.attn1.processor": processor_factory(),
            "up_blocks.2.block.attn1.processor": processor_factory(),
            "up_blocks.2.block.attn2.processor": processor_factory(),
            "up_blocks.3.block.attn1.processor": processor_factory(),
        }

    @property
    def attn_processors(self):
        return self._processors

    def set_attn_processor(self, processors):
        self._processors = dict(processors)


def memory_config(**overrides):
    values = {
        "atlas_resolution": 8,
        "max_tokens": 128,
        "min_views": 2,
        "max_memory_views": 4,
        "strength": 1.0,
        "start_progress": 0.0,
        "end_progress": 1.0,
        "exclude_self": True,
        "processor_patterns": ("up_blocks.2",),
    }
    values.update(overrides)
    return SurfaceMemoryConfig(**values)


def constant_uv(batch, height=1, width=1, value=0.5):
    return torch.full((batch, 2, height, width), float(value))


def set_valid_context(
    controller,
    batch,
    height=1,
    width=1,
    *,
    uv=None,
    layer_ids=None,
    visibility=None,
    depth=None,
    cfg_branches=1,
):
    if uv is None:
        uv = constant_uv(batch, height, width)
    if layer_ids is None:
        layer_ids = torch.zeros(
            batch, 1, height, width, dtype=torch.long
        )
    if visibility is None:
        visibility = torch.ones(batch, 1, height, width)
    if depth is None:
        depth = torch.ones(batch, 1, height, width)
    controller.set_context(
        uv,
        visibility,
        layer_ids=layer_ids,
        depth=depth,
        denoise_progress=1.0,
        cfg_branches=cfg_branches,
        cfg_layout="chunked",
    )


class SurfaceMemoryAttentionTests(unittest.TestCase):
    def test_no_context_directly_delegates_bitwise(self):
        unet = FakeUNet(processor_factory=AffineProcessor)
        original = unet.attn_processors[
            "up_blocks.2.block.attn1.processor"
        ]
        controller = install_surface_memory_attention(
            unet, memory_config()
        )
        wrapped = unet.attn_processors[
            "up_blocks.2.block.attn1.processor"
        ]
        hidden = torch.randn(2, 3, 5)

        expected = hidden.mul(1.25).add(0.125)
        actual = wrapped(None, hidden)

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(original.calls, 1)
        self.assertEqual(
            controller.diagnostics()["self_attention_calls"], 0
        )

    def test_active_context_never_changes_cross_attention(self):
        unet = FakeUNet()
        controller = install_surface_memory_attention(
            unet, memory_config(max_memory_views=2)
        )
        set_valid_context(controller, 2)
        processor = unet.attn_processors[
            "up_blocks.2.block.attn1.processor"
        ]
        attention = TinyAttention()
        hidden = torch.tensor([[[1.0]], [[3.0]]])
        encoder = torch.tensor(
            [[[2.0], [4.0]], [[20.0], [40.0]]]
        )
        expected = TinyAttnProcessor2_0()(
            attention, hidden, encoder_hidden_states=encoder
        )

        output = processor(
            attention, hidden, encoder_hidden_states=encoder
        )

        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(
            controller.diagnostics()["self_attention_calls"], 0
        )

    def test_masked_self_attention_delegates_bitwise(self):
        unet = FakeUNet(processor_factory=AffineProcessor)
        controller = install_surface_memory_attention(
            unet, memory_config()
        )
        set_valid_context(controller, 2)
        processor = unet.attn_processors[
            "up_blocks.2.block.attn1.processor"
        ]
        hidden = torch.randn(2, 3, 5)
        attention_mask = torch.ones(2, 1, 3, 3)

        output = processor(
            None, hidden, attention_mask=attention_mask
        )

        torch.testing.assert_close(
            output, hidden.mul(1.25).add(0.125)
        )
        self.assertEqual(
            controller.diagnostics()["surface_attention_calls"], 0
        )

    def test_matching_surface_reads_other_view_kv(self):
        unet = FakeUNet()
        controller = install_surface_memory_attention(
            unet, memory_config(max_memory_views=2)
        )
        set_valid_context(controller, 2)
        processor = unet.attn_processors[
            "up_blocks.2.block.attn1.processor"
        ]
        hidden = torch.tensor([[[1.0]], [[3.0]]])

        output = processor(TinyAttention(), hidden)

        torch.testing.assert_close(
            output, torch.tensor([[[3.0]], [[1.0]]])
        )
        diagnostics = controller.diagnostics()
        self.assertEqual(diagnostics["surface_attention_calls"], 1)
        self.assertEqual(diagnostics["memory_queries"], 2)
        self.assertEqual(diagnostics["memory_slots"], 2)

    def test_query_dot_key_attention_is_not_value_averaging(self):
        unet = FakeUNet()
        controller = install_surface_memory_attention(
            unet,
            memory_config(
                min_views=3,
                max_memory_views=3,
                exclude_self=True,
            ),
        )
        set_valid_context(controller, 3)
        processor = unet.attn_processors[
            "up_blocks.2.block.attn1.processor"
        ]
        hidden = torch.tensor([[[0.0]], [[1.0]], [[3.0]]])

        output = processor(TinyAttention(), hidden)

        # q=0 sees an even distribution over K={1,3}. q=1 favors K=3;
        # these differ despite both targets reading the same two view slots.
        self.assertAlmostEqual(float(output[0, 0, 0]), 2.0, places=5)
        self.assertGreater(float(output[1, 0, 0]), 2.8)
        self.assertLess(float(output[1, 0, 0]), 3.0)

    def test_source_slot_visibility_biases_cross_view_attention(self):
        unet = FakeUNet()
        controller = install_surface_memory_attention(
            unet,
            memory_config(
                min_views=3,
                max_memory_views=3,
                exclude_self=True,
            ),
        )
        visibility = torch.tensor([[[[1.0]]], [[[0.9]]], [[[0.1]]]])
        set_valid_context(
            controller, 3, visibility=visibility
        )
        processor = unet.attn_processors[
            "up_blocks.2.block.attn1.processor"
        ]
        # q=0 gives equal q.K logits for view zero.  Visibility priors 0.9
        # and 0.1 must therefore mix source values 10 and 0 into 9, not 5.
        hidden = torch.tensor([[[0.0]], [[10.0]], [[0.0]]])

        output = processor(TinyAttention(), hidden)

        self.assertAlmostEqual(float(output[0, 0, 0]), 9.0, places=5)

    def test_equal_uv_on_different_semantic_layers_never_mix(self):
        unet = FakeUNet()
        controller = install_surface_memory_attention(
            unet, memory_config(max_memory_views=2)
        )
        layers = torch.tensor([[[[0]]], [[[1]]]], dtype=torch.long)
        set_valid_context(controller, 2, layer_ids=layers)
        processor = unet.attn_processors[
            "up_blocks.2.block.attn1.processor"
        ]
        attention = TinyAttention()
        hidden = torch.tensor([[[1.0]], [[100.0]]])
        expected = TinyAttnProcessor2_0()(attention, hidden)

        output = processor(attention, hidden)

        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(controller.diagnostics()["memory_queries"], 0)

    def test_visibility_and_invalid_depth_mask_sources_and_targets(self):
        unet = FakeUNet()
        controller = install_surface_memory_attention(
            unet, memory_config(max_memory_views=3)
        )
        visibility = torch.tensor(
            [[[[1.0, 1.0, 1.0]]], [[[1.0, 0.0, 1.0]]]]
        )
        depth = torch.tensor(
            [[[[1.0, 1.0, 1.0]]], [[[2.0, 2.0, float("nan")]]]]
        )
        uv = torch.tensor(
            [
                [[[0.1, 0.5, 0.9]], [[0.5, 0.5, 0.5]]],
                [[[0.1, 0.5, 0.9]], [[0.5, 0.5, 0.5]]],
            ]
        )
        set_valid_context(
            controller,
            2,
            width=3,
            uv=uv,
            visibility=visibility,
            depth=depth,
        )
        processor = unet.attn_processors[
            "up_blocks.2.block.attn1.processor"
        ]
        attention = TinyAttention()
        hidden = torch.tensor(
            [[[1.0], [2.0], [3.0]], [[10.0], [20.0], [30.0]]]
        )
        base = TinyAttnProcessor2_0()(attention, hidden)

        output = processor(attention, hidden)

        # Only texel zero has two finite, visible observations.
        self.assertEqual(float(output[0, 0, 0]), 10.0)
        self.assertEqual(float(output[1, 0, 0]), 1.0)
        self.assertTrue(torch.equal(output[:, 1:], base[:, 1:]))
        self.assertEqual(controller.diagnostics()["memory_queries"], 2)

    def test_cfg_branches_have_independent_memories(self):
        unet = FakeUNet()
        controller = install_surface_memory_attention(
            unet, memory_config(max_memory_views=2)
        )
        # Context has two logical views and is expanded over standard
        # [uncond views, cond views] CFG layout.
        set_valid_context(controller, 2, cfg_branches=2)
        processor = unet.attn_processors[
            "up_blocks.2.block.attn1.processor"
        ]
        hidden = torch.tensor(
            [[[1.0]], [[3.0]], [[100.0]], [[300.0]]]
        )

        output = processor(TinyAttention(), hidden)

        torch.testing.assert_close(
            output,
            torch.tensor(
                [[[3.0]], [[1.0]], [[300.0]], [[100.0]]]
            ),
        )

    def test_pattern_filter_wraps_only_selected_attn1(self):
        unet = FakeUNet()
        original = dict(unet.attn_processors)
        controller = install_surface_memory_attention(
            unet, memory_config(processor_patterns=("up_blocks.2",))
        )

        selected = "up_blocks.2.block.attn1.processor"
        self.assertIsInstance(
            unet.attn_processors[selected],
            SurfaceMemoryAttnProcessor2_0,
        )
        for name in (
            "down_blocks.0.block.attn1.processor",
            "up_blocks.2.block.attn2.processor",
            "up_blocks.3.block.attn1.processor",
        ):
            self.assertIs(unet.attn_processors[name], original[name])
        self.assertEqual(controller.wrapped_processor_names, (selected,))

        controller.uninstall()
        for name, processor in original.items():
            self.assertIs(unet.attn_processors[name], processor)

    def test_minimum_view_gate_falls_back_exactly_to_base(self):
        unet = FakeUNet()
        controller = install_surface_memory_attention(
            unet,
            memory_config(min_views=3, max_memory_views=3),
        )
        set_valid_context(controller, 2)
        processor = unet.attn_processors[
            "up_blocks.2.block.attn1.processor"
        ]
        attention = TinyAttention()
        hidden = torch.tensor([[[1.0]], [[3.0]]])
        expected = TinyAttnProcessor2_0()(attention, hidden)

        output = processor(attention, hidden)

        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(controller.diagnostics()["memory_queries"], 0)

    def test_per_view_duplicate_texels_are_visibility_aggregated(self):
        unet = FakeUNet()
        controller = install_surface_memory_attention(
            unet, memory_config(max_memory_views=2)
        )
        visibility = torch.tensor(
            [[[[0.25, 0.75]]], [[[0.5, 0.5]]]]
        )
        set_valid_context(
            controller,
            2,
            width=2,
            uv=constant_uv(2, 1, 2),
            visibility=visibility,
        )
        processor = unet.attn_processors[
            "up_blocks.2.block.attn1.processor"
        ]
        hidden = torch.tensor(
            [[[0.0], [2.0]], [[10.0], [20.0]]]
        )

        output = processor(TinyAttention(), hidden)

        # Source slots are 0*.25 + 2*.75 = 1.5 and
        # 10*.5 + 20*.5 = 15.  Target visibility also gates the blend.
        torch.testing.assert_close(
            output,
            torch.tensor(
                [[[4.5], [11.740986]], [[10.75], [10.75]]]
            ),
        )

    def test_late_denoise_ramp_preserves_exact_base_before_start(self):
        unet = FakeUNet()
        controller = install_surface_memory_attention(
            unet,
            memory_config(start_progress=0.5, end_progress=1.0),
        )
        set_valid_context(controller, 2)
        controller.set_denoise_progress(0.25)
        processor = unet.attn_processors[
            "up_blocks.2.block.attn1.processor"
        ]
        attention = TinyAttention()
        hidden = torch.tensor([[[1.0]], [[3.0]]])
        expected = TinyAttnProcessor2_0()(attention, hidden)

        early = processor(attention, hidden)

        self.assertTrue(torch.equal(early, expected))
        controller.set_denoise_progress(0.75)
        halfway = processor(attention, hidden)
        torch.testing.assert_close(
            halfway, expected + 0.5 * (hidden.flip(0) - expected)
        )


if __name__ == "__main__":
    unittest.main()
