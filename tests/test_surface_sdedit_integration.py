from types import SimpleNamespace
import inspect
import unittest

import torch
import yaml

from refinement_cli import (
    refinement_run_name,
    sdedit_mode_overrides,
)
from threestudio.data.reconstruction_finetune import _ReconstructionAssets
from threestudio.systems.Head3DGSLKsFinetune import (
    Head3DGSLKsReconstructionFinetune,
)


def test_independent_mode_does_not_change_the_data_batch_size():
    overrides = sdedit_mode_overrides("independent", surface_views=4)

    assert "system.sdedit.mode=\"independent\"" in overrides
    assert "system.sdedit.surface_memory.enabled=false" in overrides
    assert "data.surface_consistent_batch=false" in overrides
    assert not any(value.startswith("data.batch_size=") for value in overrides)


def test_flame_surface_mode_enables_a_joint_view_batch():
    overrides = sdedit_mode_overrides("flame-surface", surface_views=6)

    assert "system.sdedit.mode=\"flame-surface\"" in overrides
    assert "system.sdedit.surface_memory.enabled=true" in overrides
    assert "data.surface_consistent_batch=true" in overrides
    assert "data.batch_size=6" in overrides
    assert "system.sdedit.surface_memory.views=6" in overrides
    assert "system.sdedit.surface_memory.max_memory_views=4" in overrides


def test_small_surface_batch_clamps_the_memory_view_bound():
    overrides = sdedit_mode_overrides("flame-surface", surface_views=2)

    assert "system.sdedit.surface_memory.views=2" in overrides
    assert "system.sdedit.surface_memory.max_memory_views=2" in overrides


def test_surface_mode_requires_multiple_views():
    try:
        sdedit_mode_overrides("flame-surface", surface_views=1)
    except ValueError as error:
        assert "at least two views" in str(error)
    else:
        raise AssertionError("single-view surface SDEdit was accepted")


def test_method_names_keep_every_ablation_in_a_distinct_directory():
    assert refinement_run_name("mouth", "ism", "independent") == "mouth"
    assert (
        refinement_run_name("mouth", "uvd-sfd", "independent")
        == "mouth_uvd_sfd"
    )
    assert (
        refinement_run_name("full", "ism", "flame-surface")
        == "full_surface_sdedit"
    )
    assert (
        refinement_run_name("full", "uvd-sfd", "flame-surface")
        == "full_uvd_sfd_surface_sdedit"
    )


def test_pre_sdedit_view_selection_preserves_shared_flame_pose():
    batch = {
        "c2w": torch.arange(4 * 16).reshape(4, 4, 4),
        "K": torch.arange(4 * 9).reshape(4, 3, 3),
        "azimuth": torch.arange(4),
        "flame_conds": torch.zeros(4, 8, 8, 4),
        "expression": torch.arange(100).reshape(1, 100),
        "jaw_pose": torch.arange(3).reshape(1, 3),
        "height": 8,
    }

    selected = Head3DGSLKsReconstructionFinetune._select_batch_views(
        batch, torch.tensor([2])
    )

    assert selected["c2w"].shape == (1, 4, 4)
    assert selected["azimuth"].tolist() == [2]
    assert selected["flame_conds"].shape == (1, 8, 8, 4)
    assert selected["expression"] is batch["expression"]
    assert selected["jaw_pose"] is batch["jaw_pose"]
    assert selected["height"] == 8


def test_surface_camera_group_keeps_the_legacy_anchor_in_row_zero():
    assets = SimpleNamespace(
        sample_camera_indices=lambda batch_size: [3],
        _training_camera_groups=lambda batch_size: [[2, 3]],
        elevation_groups=[list(range(8))],
    )

    selected = _ReconstructionAssets.sample_surface_camera_indices(
        assets, 4
    )

    assert selected == [3, 5, 7, 1]
    assert len(set(selected)) == 4


def test_surface_context_uses_one_crop_for_uv_layer_depth_and_visibility():
    uv = torch.tensor(
        [
            [
                [[0.0, 0.5], [0.0, 0.5]],
                [[0.0, 0.0], [1.0, 1.0]],
            ],
            [
                [[0.25, 0.75], [0.25, 0.75]],
                [[0.25, 0.25], [0.75, 0.75]],
            ],
        ],
        dtype=torch.float32,
    )
    layer = torch.tensor(
        [[[[0, 1], [2, 3]]], [[[4, 4], [3, 2]]]], dtype=torch.long
    )
    depth = torch.ones(2, 1, 2, 2)
    depth[1, 0, 0, 0] = 0.0
    visibility = torch.ones(2, 1, 2, 2)
    surface = {
        "surface_uv": uv,
        "surface_layer": layer,
        "surface_depth": depth,
        "surface_visibility": visibility,
    }

    context = (
        Head3DGSLKsReconstructionFinetune
        ._prepare_surface_memory_context(
            SimpleNamespace(),
            surface,
            torch.tensor([1]),
            torch.tensor([[0, 0, 2, 2]]),
        )
    )

    assert context["uv"].shape == (1, 2, 512, 512)
    assert context["layer_ids"].shape == (1, 1, 512, 512)
    assert context["depth"].shape == (1, 1, 512, 512)
    assert context["visibility"].shape == (1, 1, 512, 512)
    # The invalid top-left depth suppresses the exact same nearest-sampled
    # surface tokens after visibility interpolation.
    assert context["visibility"][0, 0, 0, 0].item() == 0.0
    assert context["layer_ids"][0, 0, -1, -1].item() == 2


def test_reconstruction_configs_leave_surface_memory_disabled_by_default():
    for name in ("reconstruction_mouth.yaml", "reconstruction_full.yaml"):
        with open(f"configs/{name}", "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        sdedit = config["system"]["sdedit"]
        assert sdedit["mode"] == "independent"
        assert sdedit["surface_memory"]["enabled"] is False
        assert sdedit["surface_memory"]["views"] == 4
        assert config["data"]["surface_consistent_batch"] is False
        assert sdedit["surface_memory"]["processor_patterns"] == [
            "up_blocks.2",
            "up_blocks.3",
        ]


def test_sdedit_checkpoint_signature_separates_the_two_methods():
    with open(
        "configs/reconstruction_mouth.yaml", "r", encoding="utf-8"
    ) as file:
        surface_config = yaml.safe_load(file)["system"]["sdedit"][
            "surface_memory"
        ]
    independent = SimpleNamespace(
        sdedit_mode="independent",
        surface_sdedit_enabled=False,
        surface_memory_config=surface_config,
    )
    surface = SimpleNamespace(
        sdedit_mode="flame-surface",
        surface_sdedit_enabled=True,
        surface_memory_config=surface_config,
    )
    sparse_surface = SimpleNamespace(
        sdedit_mode="flame-surface",
        surface_sdedit_enabled=True,
        surface_memory_config={"views": 6},
    )

    independent_signature = (
        Head3DGSLKsReconstructionFinetune._sdedit_method_signature(
            independent
        )
    )
    surface_signature = (
        Head3DGSLKsReconstructionFinetune._sdedit_method_signature(surface)
    )
    sparse_signature = (
        Head3DGSLKsReconstructionFinetune._sdedit_method_signature(
            sparse_surface
        )
    )

    assert independent_signature == {"mode": "independent"}
    assert surface_signature["mode"] == "flame-surface"
    assert surface_signature["surface_memory"]["views"] == 4
    assert surface_signature["surface_memory"]["processor_patterns"] == [
        "up_blocks.2",
        "up_blocks.3",
    ]
    assert sparse_signature["surface_memory"]["views"] == 6
    assert sparse_signature["surface_memory"]["max_memory_views"] == 6
    assert sparse_signature["surface_memory"]["atlas_resolution"] == 64
    assert sparse_signature["surface_memory"]["alpha_threshold"] == 0.005
    assert sparse_signature["surface_memory"]["depth_tolerance"] == 0.01


def test_resume_rejects_a_silent_sdedit_mode_switch():
    system = SimpleNamespace(
        surface_sdedit_enabled=False,
        uvd_flow_enabled=False,
        sdedit_start_step=500,
        _sdedit_method_signature=lambda: {"mode": "independent"},
        _guidance_method_signature=lambda: {"mode": "ism"},
    )

    try:
        Head3DGSLKsReconstructionFinetune._validate_sdedit_checkpoint_method(
            system,
            {
                "stage2_sdedit_method": {"mode": "flame-surface"},
                "stage2_sdedit_updates_started": True,
            },
        )
    except ValueError as error:
        assert "SDEdit method differs" in str(error)
    else:
        raise AssertionError("Checkpoint silently changed SDEdit mode")


def test_sdedit_mode_can_branch_at_the_untouched_phase_boundary():
    system = SimpleNamespace(
        sdedit_start_step=500,
        _sdedit_method_signature=lambda: {"mode": "flame-surface"},
    )

    Head3DGSLKsReconstructionFinetune._validate_sdedit_checkpoint_method(
        system,
        {
            "stage2_sdedit_method": {"mode": "independent"},
            "stage2_sdedit_updates_started": False,
            "global_step": 500,
        },
    )


def test_resume_rejects_switching_the_ism_ablation_in_both_directions():
    for current, saved in (
        ({"mode": "ism"}, {"mode": "uvd-sfd"}),
        ({"mode": "uvd-sfd"}, {"mode": "ism"}),
    ):
        system = SimpleNamespace(
            _guidance_method_signature=lambda current=current: current
        )
        try:
            Head3DGSLKsReconstructionFinetune._validate_guidance_checkpoint_method(
                system, {"stage2_guidance_method": saved}
            )
        except ValueError as error:
            assert "ISM guidance method differs" in str(error)
        else:
            raise AssertionError("Checkpoint silently changed ISM mode")


def test_resume_rejects_uvd_checkpoint_without_exact_noise_state():
    system = SimpleNamespace(
        _guidance_method_signature=lambda: {"mode": "uvd-sfd"}
    )

    try:
        Head3DGSLKsReconstructionFinetune._validate_guidance_checkpoint_method(
            system,
            {
                "stage2_guidance_method": {"mode": "uvd-sfd"},
                "global_step": 12,
            },
        )
    except ValueError as error:
        assert "noise/RNG state" in str(error)
    else:
        raise AssertionError("Incomplete UVD checkpoint was accepted")


def test_surface_correspondence_requires_the_frozen_sdedit_source():
    system = SimpleNamespace(
        surface_sdedit_enabled=True,
        _sdedit_reference_state=None,
    )

    try:
        Head3DGSLKsReconstructionFinetune._render_surface_memory_correspondence(
            system, {}
        )
    except RuntimeError as error:
        assert "frozen SDEdit reference" in str(error)
    else:
        raise AssertionError("Live geometry was accepted as SDEdit memory")


def test_sdedit_capture_freezes_the_discrete_face_binding():
    gaussian = SimpleNamespace(
        _uv=torch.zeros(2, 2),
        _d=torch.zeros(2, 1),
        _face_idx=torch.tensor([3, 7], dtype=torch.long),
        _features_dc=torch.zeros(2, 1, 3),
        _opacity=torch.zeros(2, 1),
        _scaling=torch.zeros(2, 3),
        _rotation=torch.zeros(2, 4),
    )
    system = SimpleNamespace(
        gaussian=gaussian,
        true_global_step=5,
        _stabilize_current_full_scale=lambda: {
            "world_capped": 0,
            "world_before": 0.0,
        },
    )

    Head3DGSLKsReconstructionFinetune._capture_sdedit_reference(system)
    captured = system._sdedit_reference_state["face_idx"]
    gaussian._face_idx.fill_(1)

    assert captured.dtype == torch.long
    assert captured.tolist() == [3, 7]


def test_full_geometry_barrier_uses_only_aligned_world_scale():
    system = SimpleNamespace(
        cfg=SimpleNamespace(
            max_world_scale=0.08,
            world_scale_weight=10_000.0,
            mouth={
                "max_scale": 0.015,
                "max_abs_d": 0.002,
                "barrier_weight": 0.02,
            },
        ),
        dental_point_mask=torch.zeros(2, dtype=torch.bool),
        gaussian=SimpleNamespace(
            # A very large face-local scale is valid when the corresponding
            # FLAME triangle maps it to a small world-space ellipsoid.
            _scaling=torch.tensor([[13.0, 13.0, 13.0], [0.0, 0.0, 0.0]]),
            _d=torch.zeros(2, 1),
        ),
        _full_scale_stability_enabled=lambda: False,
    )
    output = {
        "scales": torch.tensor(
            [[0.03, 0.02, 0.01], [0.09, 0.02, 0.01]],
            dtype=torch.float32,
        )
    }

    loss, parts = Head3DGSLKsReconstructionFinetune._geometry_barrier(
        system, output
    )

    assert set(parts) == {"world_scale"}
    assert torch.isclose(loss, torch.tensor(1.0), atol=1.0e-5)


def test_reference_world_scale_replaces_only_the_learned_local_axes():
    current_raw = torch.tensor(
        [[2.0, 1.0, 0.5]], dtype=torch.float32
    ).log()
    initial_raw = torch.tensor(
        [[1.0, 2.0, 0.25]], dtype=torch.float32
    ).log()
    system = SimpleNamespace(
        gaussian=SimpleNamespace(
            _scaling=current_raw,
            scaling_activation=torch.exp,
        ),
        initial={"scale": initial_raw},
    )
    current_world = torch.tensor(
        [[0.04, 0.02, 0.01]], requires_grad=True
    )

    reference = (
        Head3DGSLKsReconstructionFinetune._reference_world_scales_current_pose(
            system, current_world
        )
    )

    assert torch.allclose(
        reference, torch.tensor([[0.02, 0.04, 0.005]])
    )
    assert not reference.requires_grad


def test_full_geometry_barrier_limits_reference_world_growth_and_anisotropy():
    reference = torch.tensor(
        [[0.02, 0.02, 0.01], [0.02, 0.01, 0.01], [0.01, 0.01, 0.01]]
    )
    current = torch.tensor(
        [[0.03, 0.02, 0.01], [0.04, 0.01, 0.01], [1.0, 0.001, 0.001]]
    )
    system = SimpleNamespace(
        cfg=SimpleNamespace(
            max_world_scale=2.0,
            world_scale_weight=1.0,
            scale_stability={
                "enabled": True,
                "reference_growth_limit": 1.5,
                "reference_growth_weight": 1.0,
                "reference_anisotropy_growth_limit": 1.5,
                "reference_anisotropy_weight": 1.0,
            },
            mouth={
                "max_scale": 0.015,
                "max_abs_d": 0.002,
                "barrier_weight": 0.02,
            },
        ),
        dental_point_mask=torch.zeros(3, dtype=torch.bool),
        gaussian=SimpleNamespace(_d=torch.zeros(3, 1)),
        _full_scale_stability_enabled=lambda: True,
        _active_trainable_point_mask=lambda: torch.tensor(
            [True, True, False]
        ),
        _reference_world_scales_current_pose=lambda scales: reference,
    )

    loss, parts = Head3DGSLKsReconstructionFinetune._geometry_barrier(
        system, {"scales": current}
    )

    expected_excess = torch.log(torch.tensor(2.0 / 1.5))
    assert set(parts) == {
        "world_scale",
        "world_scale_growth",
        "world_anisotropy_growth",
    }
    assert torch.isclose(
        loss, 2.0 * expected_excess.square(), atol=1.0e-6
    )


def test_full_hard_cap_uses_world_scale_and_skips_protected_rows():
    scaling = torch.nn.Parameter(
        torch.tensor(
            [[13.0, 13.0, 13.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        )
    )
    world_scales = torch.tensor(
        [[0.04, 0.03, 0.02], [0.16, 0.08, 0.04], [0.16, 0.08, 0.04]]
    )
    cleared = []
    system = SimpleNamespace(
        cfg=SimpleNamespace(
            max_world_scale=0.08,
            scale_stability={"enabled": True, "hard_max_world_scale": 0.08},
        ),
        gaussian=SimpleNamespace(_scaling=scaling),
        _full_scale_stability_enabled=lambda: True,
        _aligned_render_geometry=lambda: (None, world_scales, None),
        _full_region_protection_enabled=lambda: True,
        _active_trainable_point_mask=lambda: torch.tensor(
            [True, True, False], dtype=torch.bool
        ),
        _clear_parameter_optimizer_rows=lambda parameter, mask: cleared.append(
            (parameter, mask.clone())
        ),
    )
    system._full_world_scale_axis_ratio = lambda scales: (
        Head3DGSLKsReconstructionFinetune._full_world_scale_axis_ratio(
            system, scales
        )
    )

    count, maximum_before = (
        Head3DGSLKsReconstructionFinetune._cap_current_full_world_scale(system)
    )

    assert count == 1
    assert abs(maximum_before - 0.16) < 1.0e-6
    assert torch.equal(scaling[0], torch.tensor([13.0, 13.0, 13.0]))
    assert torch.allclose(
        scaling[1], torch.tensor([0.5, 1.0, 1.0]).log()
    )
    assert torch.equal(scaling[2], torch.tensor([1.0, 1.0, 1.0]))
    assert len(cleared) == 1
    assert torch.equal(cleared[0][1], torch.tensor([False, True, False]))


def test_full_render_cap_is_per_axis_and_preserves_protected_rows():
    system = SimpleNamespace(
        cfg=SimpleNamespace(
            max_world_scale=0.05,
            scale_stability={"enabled": True, "hard_max_world_scale": 0.05},
        ),
        _full_scale_stability_enabled=lambda: True,
        _full_region_protection_enabled=lambda: True,
        _active_trainable_point_mask=lambda: torch.tensor([True, False]),
    )
    system._full_world_scale_axis_ratio = lambda current: (
        Head3DGSLKsReconstructionFinetune._full_world_scale_axis_ratio(
            system, current
        )
    )
    scales = torch.tensor(
        [[0.10, 0.04, 0.02], [0.10, 0.04, 0.02]], requires_grad=True
    )

    capped = Head3DGSLKsReconstructionFinetune._cap_full_render_world_scales(
        system, scales
    )

    assert torch.allclose(capped[0], torch.tensor([0.05, 0.04, 0.02]))
    assert torch.equal(capped[1], scales[1])


def test_absolute_world_anisotropy_barrier_uses_only_trainable_rows():
    current = torch.tensor(
        [[0.03, 0.001, 0.001], [0.03, 0.001, 0.001]],
        dtype=torch.float32,
    )
    system = SimpleNamespace(
        cfg=SimpleNamespace(
            max_world_scale=0.05,
            world_scale_weight=0.0,
            scale_stability={
                "enabled": True,
                "max_world_anisotropy": 10.0,
                "world_anisotropy_weight": 2.0,
                "reference_growth_weight": 0.0,
                "reference_anisotropy_weight": 0.0,
            },
            mouth={
                "max_scale": 0.015,
                "max_abs_d": 0.002,
                "barrier_weight": 0.02,
            },
        ),
        dental_point_mask=torch.zeros(2, dtype=torch.bool),
        gaussian=SimpleNamespace(_d=torch.zeros(2, 1)),
        _full_scale_stability_enabled=lambda: True,
        _active_trainable_point_mask=lambda: torch.tensor([True, False]),
        _reference_world_scales_current_pose=lambda scales: scales.detach(),
    )

    loss, parts = Head3DGSLKsReconstructionFinetune._geometry_barrier(
        system, {"scales": current}
    )

    expected = 2.0 * torch.log(torch.tensor(3.0)).square()
    assert set(parts) == {"world_scale", "world_anisotropy"}
    assert torch.isclose(loss, expected, atol=1.0e-6)


def test_absolute_world_anisotropy_hard_cap_shrinks_only_long_axes():
    scaling = torch.nn.Parameter(torch.zeros(2, 3))
    world_scales = torch.tensor(
        [[0.04, 0.001, 0.001], [0.04, 0.001, 0.001]]
    )
    system = SimpleNamespace(
        cfg=SimpleNamespace(
            max_world_scale=0.05,
            scale_stability={
                "enabled": True,
                "hard_max_world_scale": 0.05,
                "max_world_anisotropy": 10.0,
            },
        ),
        gaussian=SimpleNamespace(_scaling=scaling),
        _full_scale_stability_enabled=lambda: True,
        _aligned_render_geometry=lambda: (None, world_scales, None),
        _full_region_protection_enabled=lambda: True,
        _active_trainable_point_mask=lambda: torch.tensor([True, False]),
        _clear_parameter_optimizer_rows=lambda *_args: None,
    )
    system._full_world_scale_axis_ratio = lambda current: (
        Head3DGSLKsReconstructionFinetune._full_world_scale_axis_ratio(
            system, current
        )
    )

    count, _ = (
        Head3DGSLKsReconstructionFinetune._cap_current_full_world_scale(system)
    )

    assert count == 1
    assert torch.allclose(
        scaling[0], torch.tensor([0.25, 1.0, 1.0]).log()
    )
    assert torch.equal(scaling[1], torch.zeros(3))


def test_full_face_rebinding_receives_only_the_trainable_point_mask():
    captured = {}
    trainable = torch.tensor([True, False, False, True])
    gaussian = SimpleNamespace(
        _uv=torch.nn.Parameter(torch.zeros(4, 2)),
        update_face_idx_from_uv=lambda **kwargs: (
            captured.update(kwargs) or {"updated": 0, "projected": 0}
        ),
    )
    system = SimpleNamespace(
        gaussian=gaussian,
        optimization_stage="full",
        _optimizer_stepped_this_batch=True,
        true_global_step=1,
        _last_geometry_projection_step=0,
        _active_trainable_point_mask=lambda: trainable,
        _stabilize_current_full_scale=lambda: {
            "world_capped": 0,
            "world_before": 0.0,
        },
        log=lambda *_args, **_kwargs: None,
        cfg=SimpleNamespace(reference_dual_lr=0.0),
        _maybe_write_first_phase_artifacts=lambda: None,
    )

    Head3DGSLKsReconstructionFinetune.on_train_batch_end(
        system, None, {}, 0
    )

    assert captured["return_stats"] is True
    assert captured["mask"] is trainable


def test_mouth_screen_depth_loss_detects_an_occluder_inside_the_mouth():
    target_rgb = torch.zeros(1, 2, 2, 3)
    target_alpha = torch.ones(1, 2, 2, 1)
    target_depth = torch.full((1, 2, 2, 1), 2.0)
    current_depth = target_depth.clone().requires_grad_(True)
    current_depth.data[0, 0, 0, 0] = 1.0
    mouth_mask = torch.zeros(1, 2, 2, 1)
    mouth_mask[0, 0, 0, 0] = 1.0
    system = SimpleNamespace(
        cfg=SimpleNamespace(
            full_protection={
                "mouth_screen_preservation": {
                    "enabled": True,
                    "dilation": 0,
                    "rgb_weight": 100.0,
                    "alpha_weight": 50.0,
                    "depth_weight": 10.0,
                    "alpha_threshold": 0.05,
                }
            }
        ),
        _full_region_protection_enabled=lambda: True,
        _render_initial_dynamic_reference=lambda _batch: (
            target_rgb,
            target_alpha,
            target_depth,
        ),
        _render_point_mask=lambda *_args: mouth_mask,
        _dilate_mask=lambda mask, _dilation: mask,
        mouth_guidance_point_mask=torch.ones(1, dtype=torch.bool),
    )
    system._masked_smooth_l1 = lambda prediction, target, mask: (
        Head3DGSLKsReconstructionFinetune._masked_smooth_l1(
            prediction, target, mask
        )
    )
    output = {
        "comp_rgb": target_rgb.clone(),
        "alpha": target_alpha.clone(),
        "depth": current_depth,
    }

    loss, parts = (
        Head3DGSLKsReconstructionFinetune._mouth_screen_preservation_loss(
            system, {}, output
        )
    )
    loss.backward()

    assert parts["mouth_screen_rgb"].item() == 0.0
    assert parts["mouth_screen_alpha"].item() == 0.0
    assert torch.isclose(
        parts["mouth_screen_depth"], torch.tensor(1.25), atol=1.0e-6
    )
    assert current_depth.grad[0, 0, 0, 0].item() < 0.0
    assert current_depth.grad[0, 1, 1, 0].item() == 0.0


def test_phase_snapshot_copy_restores_the_discrete_face_binding():
    gaussian = SimpleNamespace(
        _face_idx=torch.tensor([8, 9], dtype=torch.long)
    )
    system = SimpleNamespace(gaussian=gaussian)
    system._gaussian_fields = lambda: {"face_idx": "_face_idx"}

    Head3DGSLKsReconstructionFinetune._copy_phase_snapshot_to_gaussian(
        system, {"face_idx": torch.tensor([3, 7], dtype=torch.long)}
    )

    assert gaussian._face_idx.tolist() == [3, 7]


def test_frozen_geometry_uses_snapshot_face_binding_not_live_binding():
    class FakeGaussian:
        device = torch.device("cpu")

        def __init__(self):
            self._face_idx = torch.tensor([91, 92], dtype=torch.long)
            self.map_binding = None
            self.frame_binding = None

        def _flame_verts_and_normals(self):
            return torch.zeros(3, 3), torch.zeros(3, 3)

        def _map_uvd_to_xyz(self, uvd, vertices, normals, *, face_idx):
            self.map_binding = face_idx.detach().clone()
            return torch.zeros(uvd.shape[0], 3)

        def _deformed_scaling_rotation(
            self, vertices, *, face_idx, local_scaling, local_rotation
        ):
            self.frame_binding = face_idx.detach().clone()
            rotation = torch.zeros(local_scaling.shape[0], 4)
            rotation[:, 0] = 1.0
            return local_scaling, rotation

        @staticmethod
        def scaling_activation(scale):
            return scale

        @staticmethod
        def opacity_activation(opacity):
            return opacity

    gaussian = FakeGaussian()
    system = SimpleNamespace(
        gaussian=gaussian,
        alignment=torch.eye(4),
        _set_pose=lambda *args: None,
        _batch_pose=lambda batch: (),
        _set_reference_pose=lambda: None,
        _full_scale_stability_enabled=lambda: False,
    )
    snapshot_binding = torch.tensor([3, 7], dtype=torch.long)
    state = {
        "uv": torch.zeros(2, 2),
        "d": torch.zeros(2, 1),
        "face_idx": snapshot_binding,
        "scale": torch.ones(2, 3),
        "rotation": torch.zeros(2, 4),
        "opacity": torch.ones(2, 1),
    }

    Head3DGSLKsReconstructionFinetune._frozen_reference_geometry(
        system, state, {}
    )

    assert torch.equal(gaussian.map_binding, snapshot_binding)
    assert torch.equal(gaussian.frame_binding, snapshot_binding)


def test_frozen_geometry_uses_snapshot_scale_mask_after_live_densification():
    class FakeGaussian:
        device = torch.device("cpu")
        num_gs = 3

        def _flame_verts_and_normals(self):
            return torch.zeros(3, 3), torch.zeros(3, 3)

        def _map_uvd_to_xyz(self, uvd, vertices, normals, *, face_idx):
            return torch.zeros(uvd.shape[0], 3)

        def _deformed_scaling_rotation(
            self, vertices, *, face_idx, local_scaling, local_rotation
        ):
            rotation = torch.zeros(local_scaling.shape[0], 4)
            rotation[:, 0] = 1.0
            return local_scaling, rotation

        @staticmethod
        def scaling_activation(scale):
            return scale

        @staticmethod
        def opacity_activation(opacity):
            return opacity

    gaussian = FakeGaussian()
    system = SimpleNamespace(
        gaussian=gaussian,
        alignment=torch.eye(4),
        cfg=SimpleNamespace(
            scale_stability={
                "hard_max_world_scale": 0.05,
                "max_world_anisotropy": None,
            },
            max_world_scale=0.05,
        ),
        _set_pose=lambda *args: None,
        _batch_pose=lambda batch: (),
        _set_reference_pose=lambda: None,
        _full_scale_stability_enabled=lambda: True,
        _full_region_protection_enabled=lambda: True,
        _active_trainable_point_mask=lambda: (_ for _ in ()).throw(
            AssertionError("live densified mask must not be used")
        ),
    )
    system._full_world_scale_axis_ratio = (
        lambda scales, trainable_mask=None: (
            Head3DGSLKsReconstructionFinetune._full_world_scale_axis_ratio(
                system, scales, trainable_mask
            )
        )
    )
    state = {
        "uv": torch.zeros(2, 2),
        "d": torch.zeros(2, 1),
        "face_idx": torch.tensor([3, 7], dtype=torch.long),
        "scale": torch.full((2, 3), 0.10),
        "rotation": torch.zeros(2, 4),
        "opacity": torch.ones(2, 1),
        "scale_trainable_mask": torch.tensor([True, False]),
    }

    packed, _ = (
        Head3DGSLKsReconstructionFinetune._frozen_reference_geometry(
            system, state, {}
        )
    )
    scales = packed[1]

    assert torch.allclose(scales[0], torch.full((3,), 0.05))
    assert torch.allclose(scales[1], torch.full((3,), 0.10))


def test_snapshot_surface_masks_ignore_live_face_rebinding():
    class FakeGaussian:
        num_gs = 5
        device = torch.device("cpu")

        def __init__(self):
            self._face_idx = torch.zeros(5, dtype=torch.long)
            self._faces = torch.zeros(14, 3, dtype=torch.long)
            face_by_region = {
                "teeth_upper": 10,
                "teeth_lower": 11,
                "oral_cavity": 12,
                "lips": 13,
            }
            self.model = SimpleNamespace(
                mask=SimpleNamespace(
                    get_fid_by_region=lambda regions: torch.tensor(
                        [face_by_region[regions[0]]], dtype=torch.long
                    )
                )
            )

        # Deliberately retain the legacy one-argument API. Snapshot masks must
        # not pass a face_idx keyword into this method.
        def point_region_mask(self, region):
            region_faces = self.model.mask.get_fid_by_region([region])
            return self._face_idx == region_faces[0]

    gaussian = FakeGaussian()
    system = SimpleNamespace(gaussian=gaussian)
    system._point_region_mask_for_binding = lambda region, face_idx: (
        Head3DGSLKsReconstructionFinetune._point_region_mask_for_binding(
            system, region, face_idx
        )
    )
    system._safe_point_region_mask = lambda regions, face_idx=None: (
        Head3DGSLKsReconstructionFinetune._safe_point_region_mask(
            system, regions, face_idx=face_idx
        )
    )
    snapshot_binding = torch.tensor([0, 10, 11, 12, 13])

    masks = Head3DGSLKsReconstructionFinetune._canonical_surface_masks(
        system, face_idx=snapshot_binding
    )

    assert masks["face"].tolist() == [True, False, False, False, False]
    assert masks["teeth_upper"].tolist() == [False, True, False, False, False]
    assert masks["teeth_lower"].tolist() == [False, False, True, False, False]
    assert masks["oral_cavity"].tolist() == [False, False, False, True, False]
    assert masks["lips"].tolist() == [False, False, False, False, True]


def test_uvd_regional_guidance_has_no_removed_crop_area_metric():
    source = inspect.getsource(
        Head3DGSLKsReconstructionFinetune._diffusion_loss
    )

    assert "crop_area_fraction" not in source


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value))
    return suite
