from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import yaml

from threestudio.utils.animportrait3d_prompt import (
    view_direction_indices,
    view_prompt,
)
from threestudio.data.reconstruction_finetune import (
    _calibrated_orbit_camera,
    _fit_calibrated_orbit,
)
from threestudio.systems.Head3DGSLKsFinetune import (
    Head3DGSLKsReconstructionFinetune,
)
from threestudio.models.guidance.controlnet_guidance import ControlNetGuidance
from refinement_cli import guidance_mode_overrides
from train_mouth import mouth_region_prompt
from train_full import (
    require_verified_mouth_sidecar,
    validate_abstract_prompt,
    validate_full_prompt,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_animportrait3d_direction_boundaries_match_reference():
    azimuth = torch.tensor(
        [0.0, 40.0, 59.9, 60.0, 60.1, 90.0, 119.9, 120.0,
         120.1, 180.0, 194.9, 195.0, 270.0, 359.0]
    )
    # 0=side, 1=front, 2=back. Strict boundary behavior intentionally
    # matches train_mouth.py/train_all.py rather than widening the bins.
    expected = torch.tensor([0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 2, 2, 2])
    assert torch.equal(view_direction_indices(azimuth), expected)


def test_animportrait3d_prompt_prefixes_match_helper():
    prompt = "mouth region, a portrait"
    assert [view_prompt(direction, prompt) for direction in ("side", "front", "back")] == [
        "side view mouth region, a portrait",
        "front view mouth region, a portrait",
        "back view mouth region, a portrait",
    ]


def test_mouth_entrypoint_adds_reference_region_prefix_once():
    assert mouth_region_prompt("A man") == "mouth region, A man"
    assert (
        mouth_region_prompt("mouth region, A man")
        == "mouth region, A man"
    )


def test_refinement_configs_keep_the_repaired_precision_and_camera_ranges():
    with (PROJECT_ROOT / "configs" / "reconstruction_mouth.yaml").open(
        "r", encoding="utf-8"
    ) as file:
        mouth = yaml.safe_load(file)
    with (PROJECT_ROOT / "configs" / "reconstruction_full.yaml").open(
        "r", encoding="utf-8"
    ) as file:
        full = yaml.safe_load(file)

    assert mouth["trainer"]["precision"] == "32-true"
    assert full["trainer"]["precision"] == "32-true"
    assert mouth["data"]["train_elevation_range"] == [-20.0, 20.0]
    assert full["data"]["train_elevation_range"] == [-20.0, 20.0]
    assert full["data"]["batch_size"] == 1
    assert "train_camera_sampling" not in full["data"]
    assert "animportrait3d_camera_radius" not in full["data"]
    assert "animportrait3d_camera_pivot" not in full["data"]
    assert "animportrait3d_camera_fov_degrees" not in full["data"]
    assert full["data"]["train_front_hemisphere_probability"] == 0.70
    assert full["data"]["condition_type"] == "animportrait3d_normal_seg"
    assert (
        full["system"]["optimization"]["ism_accumulate_grad_batches"]
        == 3
    )
    assert full["system"]["optimization"]["feature_lr"] == 3.75e-3
    assert full["system"]["optimization"]["opacity_lr"] == 7.5e-2
    assert full["system"]["optimization"]["uv_lr"] == 2.985e-5
    assert full["system"]["optimization"]["scale_lr"] == 2.55e-2
    assert full["system"]["optimization"]["max_grad_norm"] == 0.0
    assert full["system"]["guidance"]["guidance_scale"] == 100.0
    assert full["system"]["guidance"]["control_type"] == "animportrait3d"
    assert full["system"]["guidance"]["ism_variant"] == "animportrait3d"
    assert mouth["system"]["guidance"]["use_ism"] is True
    assert full["system"]["guidance"]["use_ism"] is True
    assert mouth["system"]["guidance"]["use_uvd_surface_flow"] is False
    assert full["system"]["guidance"]["use_uvd_surface_flow"] is False
    expected_phase_artifacts = {
        "enabled": True,
        "save_gaussian": True,
        "render_driving_test": True,
        "driving_fps": 30,
    }
    assert (
        mouth["system"]["first_phase_artifacts"]
        == expected_phase_artifacts
    )
    assert (
        full["system"]["first_phase_artifacts"]
        == expected_phase_artifacts
    )
    assert full["system"]["guidance"]["uvd_flow_surface_layers"] == 5
    assert mouth["system"]["guidance"]["uvd_flow_uv_resolution"] == 512
    assert full["system"]["guidance"]["uvd_flow_uv_resolution"] == 512
    removed_flow_fields = {
        "uvd_flow_noise_injection_rate",
        "uvd_flow_gradient_scale",
        "uvd_flow_guidance_scale",
        "uvd_flow_cfg_rescale",
        "uvd_flow_max_grad_rms",
        "uvd_flow_max_step_percent",
    }
    for config in (mouth, full):
        assert not removed_flow_fields.intersection(
            config["system"]["guidance"]
        )
        assert config["system"]["guidance"][
            "uvd_flow_min_distinct_cells"
        ] == 8
        assert not {
            "optimizer_lr_scale",
            "max_chroma_drift",
            "reference_excess_weight",
            "reset_optimizer_at_sdedit",
            "regional_area_weight_power",
            "minimum_regional_area_weight",
            "negative_prompt",
        }.intersection(config["system"]["uvd_surface_flow"])
    assert full["system"]["uvd_surface_flow"]["d_range"] is None
    assert full["system"]["uvd_surface_flow"]["max_uvd_variance"] == 0.0025
    assert full["system"]["guidance"]["vae_encode_mode"] is False
    assert full["system"]["regional_guidance"]["full_use_control"] is False
    assert full["system"]["regional_guidance"]["crop_from_landmarks"] is True
    assert full["system"]["sdedit"]["regional_full_crop_loss"] is True
    assert full["system"]["sdedit"]["lpips_net"] == "vgg"
    assert full["system"]["reference_dual_lr"] == 0.0
    assert full["system"]["densification"]["steps"] == [50, 100]
    assert full["system"]["full_protection"] == {
        "enabled": True,
        "freeze_eyes": True,
        "freeze_mouth": True,
        "freeze_dental": True,
        "protect_from_densification": True,
        "mask_face_loss": True,
    }
    assert full["system"]["regional_guidance"]["abstract_prompt"] == (
        "a photorealistic human face with natural skin, eyes, lips and teeth"
    )
    assert full["system"]["regional_guidance"]["full_weight"] == 1.0
    assert full["system"]["regional_guidance"]["face_weight"] == 1.0
    for name in ("left_eye", "right_eye", "mouth"):
        assert full["system"]["regional_guidance"][f"{name}_weight"] == 0.0
    assert full["system"]["sdedit"]["region_weights"]["face"] == 1.0
    for name in ("left_eye", "right_eye", "mouth"):
        assert full["system"]["sdedit"]["region_weights"][name] == 0.0
    assert full["system"]["densification"]["protect_dental"] is True
    assert full["system"]["densification"]["exclude_dental"] is True
    assert full["system"]["prompt_processor"]["negative_prompt"] == (
        "tattoo, blur, lowres, bad anatomy, bad hands, cropped, worst quality, "
        "low quality, blurry, jpeg artifacts, watermark, text, duplicate face, "
        "extra eyes, extra mouth, deformed eyes, asymmetry, overexposed, HDR, "
        "neon, glow, chromatic aberration, oversaturated, waxy skin, cgi, "
        "cartoon, fused lips, duplicated teeth, floating teeth, malformed teeth"
    )


def test_guidance_mode_switch_preserves_ism_as_an_explicit_ablation():
    ism = guidance_mode_overrides("ism")
    flow = guidance_mode_overrides(
        "uvd-sfd",
        uvd_flow_seed=13,
    )
    assert "system.guidance.use_ism=true" in ism
    assert "system.guidance.use_uvd_surface_flow=false" in ism
    assert "system.guidance.use_ism=false" in flow
    assert "system.guidance.use_uvd_surface_flow=true" in flow
    assert "system.guidance.uvd_flow_noise_seed=13" in flow
    assert not any("injection" in value for value in flow)


def test_guidance_checkpoint_signature_records_uvd_coupling_only():
    raw = SimpleNamespace(uvd_flow_enabled=False)
    uvd = SimpleNamespace(
        uvd_flow_enabled=True,
        cfg=SimpleNamespace(
            guidance={
                "uvd_flow_noise_seed": 13,
                "uvd_flow_uv_resolution": 512,
                "uvd_flow_depth_resolution": 8,
                "uvd_flow_surface_layers": 5,
                "uvd_flow_min_distinct_cells": 8,
            },
            uvd_surface_flow={
                "alpha_threshold": 0.05,
                "contribution_threshold": 0.01,
                "dominance_ratio": 1.10,
                "max_uvd_variance": 0.0025,
                "opacity_floor": 0.0,
                "d_range": None,
                "d_padding_ratio": 0.05,
            },
        ),
    )

    raw_signature = (
        Head3DGSLKsReconstructionFinetune._guidance_method_signature(raw)
    )
    uvd_signature = (
        Head3DGSLKsReconstructionFinetune._guidance_method_signature(uvd)
    )

    assert raw_signature == {"mode": "ism"}
    assert uvd_signature["mode"] == "uvd-sfd"
    assert uvd_signature["noise"]["seed"] == 13
    assert uvd_signature["noise"]["uv_resolution"] == 512
    assert uvd_signature["correspondence"]["d_range"] is None


def test_uvd_ism_uses_the_same_negative_prompt_as_raw_ism():
    system = SimpleNamespace(
        cfg=SimpleNamespace(
            prompt="a portrait",
            prompt_processor={"negative_prompt": "base negative"},
        ),
        uvd_flow_enabled=True,
    )

    flow_config = Head3DGSLKsReconstructionFinetune._prompt_config(system)
    explicit_config = Head3DGSLKsReconstructionFinetune._prompt_config(
        system, negative_prompt="sdedit negative"
    )

    assert flow_config.negative_prompt == "base negative"
    assert explicit_config.negative_prompt == "sdedit negative"


def test_uvd_correspondence_reuses_the_exact_rgb_crop_plan():
    system = SimpleNamespace(
        cfg=SimpleNamespace(sdedit={}),
        _mask_bchw=Head3DGSLKsReconstructionFinetune._mask_bchw,
    )
    prediction = torch.zeros(1, 3, 8, 8)
    control = torch.zeros(1, 4, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, 2:6, 1:5] = 1.0
    prepared = Head3DGSLKsReconstructionFinetune._prepare_sdedit_images(
        system,
        prediction,
        prediction,
        control,
        mask,
        crop=True,
        padding_override=0,
        return_crop_plan=True,
    )
    assert prepared is not None
    _, _, _, _, indices, crop_plan = prepared
    assert crop_plan.tolist() == [[1, 2, 5, 6]]

    y, x = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    surface = {
        "surface_uvd": torch.stack(
            (
                x.float() / 7.0,
                y.float() / 7.0,
                torch.full_like(x, 0.25).float(),
            )
        )[None],
        "surface_layer": torch.where(x < 3, 2, 3)[None, None],
        "surface_confidence": torch.ones(1, 1, 8, 8),
    }
    # The crop's last source pixel is deliberately invalid while its
    # confidence is one. Resize-time validity must gate the confidence back
    # to zero instead of optimizing a persistent background-noise cell.
    surface["surface_layer"][0, 0, 5, 4] = -1
    surface["surface_uvd"][0, :, 5, 4] = 1.5
    correspondence = (
        Head3DGSLKsReconstructionFinetune._prepare_uvd_flow_correspondence(
            system, surface, indices, crop_plan
        )
    )
    assert correspondence["uvd_flow_surface_uvd"].shape == (1, 3, 512, 512)
    assert correspondence["uvd_flow_surface_layer"].shape == (1, 1, 512, 512)
    assert set(
        correspondence["uvd_flow_surface_layer"].unique().tolist()
    ) == {-1, 2, 3}
    torch.testing.assert_close(
        correspondence["uvd_flow_surface_uvd"][0, :, 0, 0],
        surface["surface_uvd"][0, :, 2, 1],
    )
    assert correspondence["uvd_flow_surface_confidence"][0, 0, -1, -1] == 0


def test_uvd_variance_rejection_never_falls_through_to_a_rear_layer():
    contribution = torch.tensor([0.80, 0.10, 0.0, 0.0, 0.0])[:, None, None]
    alpha = torch.ones_like(contribution)
    variance = torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0])[:, None, None]
    winner, valid, rejected = (
        Head3DGSLKsReconstructionFinetune._select_uvd_flow_layer(
            contribution,
            alpha,
            variance,
            alpha_threshold=0.05,
            contribution_threshold=0.01,
            dominance_ratio=1.10,
            max_uvd_variance=0.0025,
        )
    )
    assert winner.item() == 0
    assert not valid.item()
    assert rejected.item()


def test_animportrait3d_ism_uses_the_explicit_noise_in_its_inversion():
    explicit_noise = torch.randn(1, 4, 2, 2)
    observed_noise = []
    prediction_calls = []

    def add_noise(latents, noise, timestep):
        observed_noise.append(noise.clone())
        return latents + noise + timestep[:, None, None, None] * 0.0

    def predict(sample, timestep, embeddings, image_cond, use_control):
        prediction_calls.append(
            (sample.shape[0], timestep.clone(), use_control)
        )
        if sample.shape[0] == 2:
            return torch.cat(
                (torch.full_like(sample[:1], 2.0),
                 torch.full_like(sample[:1], 5.0))
            )
        return torch.ones_like(sample)

    guidance = SimpleNamespace(
        cfg=SimpleNamespace(guidance_scale=4.0),
        scheduler=SimpleNamespace(add_noise=add_noise),
        alphas=torch.full((1000,), 0.25),
        _predict_ism_noise=predict,
        _ddim_inverse_jump=(
            lambda sample, _epsilon, _current_t, _next_t: sample
        ),
    )
    gradient = ControlNetGuidance._compute_grad_animportrait3d_ism(
        guidance,
        torch.zeros(3, 1, 1),
        torch.zeros_like(explicit_noise),
        torch.zeros(1, 4, 4, 4),
        torch.tensor([300]),
        noise=explicit_noise,
        step=400,
        use_control=False,
    )

    assert len(observed_noise) == 1
    torch.testing.assert_close(observed_noise[0], explicit_noise)
    assert len(prediction_calls) == 5
    assert prediction_calls[-1][0] == 2
    assert all(not call[2] for call in prediction_calls)
    # target epsilon=1; CFG epsilon=2 + 4*(5-2)=14; ISM weight=sqrt(3).
    torch.testing.assert_close(
        gradient,
        torch.full_like(gradient, 13.0 * (3.0 ** 0.5)),
    )


def test_uvd_mode_dispatches_to_the_same_ism_with_canonical_noise():
    canonical_noise = torch.randn(1, 4, 64, 64)
    calls = {}

    class Prompt:
        directions = ()

        @staticmethod
        def get_text_embeddings(*_args, **_kwargs):
            return torch.zeros(3, 1, 1)

    def compute_ism(
        text_embeddings,
        latents,
        image_cond,
        timestep,
        *,
        noise,
        step,
        use_control,
    ):
        calls.update(
            noise=noise,
            step=step,
            timestep=timestep.clone(),
            use_control=use_control,
        )
        return torch.zeros_like(latents)

    guidance = SimpleNamespace(
        cfg=SimpleNamespace(
            edit_image=False,
            use_uvd_surface_flow=True,
            coupled_batch=False,
            coupled_apply_to_sds=False,
            coupled_share_t=True,
            coupled_share_noise=False,
            coupled_mean_grad=False,
            ism_sample_at_max_step=True,
            use_ism=False,
            use_nfsd=False,
            use_dsd=False,
        ),
        device=torch.device("cpu"),
        min_step=10,
        max_step=300,
        num_train_timesteps=1000,
        uvd_flow_noise=object(),
        grad_clip_val=None,
        _vsd_active=lambda _step: False,
        encode_images=lambda _images: torch.zeros(1, 4, 64, 64),
        _uvd_flow_noise_from_surface=(
            lambda *_args, **_kwargs: (
                canonical_noise,
                {"uvd_flow_transport_reliability": torch.tensor(1.0)},
            )
        ),
        compute_grad_ism=compute_ism,
        _safe_norm=lambda value: value.norm(),
    )
    result = ControlNetGuidance.__call__(
        guidance,
        7,
        torch.zeros(1, 3, 64, 64),
        torch.zeros(1, 4, 64, 64),
        Prompt(),
        torch.zeros(1),
        torch.zeros(1),
        torch.ones(1),
        uvd_flow_surface_uvd=torch.zeros(1, 3, 512, 512),
        uvd_flow_surface_layer=torch.zeros(1, 1, 512, 512),
        uvd_flow_surface_confidence=torch.ones(1, 1, 512, 512),
        use_control=False,
        view_dependent_prompting=False,
    )

    assert calls["noise"] is canonical_noise
    assert calls["step"] == 7
    assert calls["timestep"].tolist() == [300]
    assert calls["use_control"] is False
    assert result["loss_uvd_consistent_ism"].item() == 0.0

    guidance._uvd_flow_noise_from_surface = (
        lambda *_args, **_kwargs: (
            canonical_noise[..., :32, :],
            {},
        )
    )
    try:
        ControlNetGuidance.__call__(
            guidance,
            8,
            torch.zeros(1, 3, 64, 64),
            torch.zeros(1, 4, 64, 64),
            Prompt(),
            torch.zeros(1),
            torch.zeros(1),
            torch.ones(1),
            uvd_flow_surface_uvd=torch.zeros(1, 3, 512, 512),
            uvd_flow_surface_layer=torch.zeros(1, 1, 512, 512),
            uvd_flow_surface_confidence=torch.ones(1, 1, 512, 512),
            view_dependent_prompting=False,
        )
    except ValueError as error:
        assert "noise shape must exactly match" in str(error)
    else:
        raise AssertionError("Mismatched UVD noise shape was accepted")


def test_uvd_mode_requires_the_animportrait3d_ism_variant():
    guidance = SimpleNamespace(
        cfg=SimpleNamespace(ism_variant="interval")
    )

    try:
        ControlNetGuidance._setup_uvd_surface_flow(guidance)
    except ValueError as error:
        assert "ism_variant='animportrait3d'" in str(error)
    else:
        raise AssertionError("UVD mode accepted the interval ISM variant")


def test_uvd_and_raw_ism_use_the_same_adam_learning_rate():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam(
        [
            {
                "params": [parameter],
                "name": "feature",
                "base_lr": 0.02,
                "start": 0,
            }
        ],
        lr=0.0,
    )
    system = SimpleNamespace(
        gaussian=SimpleNamespace(optimizer=optimizer),
        true_global_step=0,
        cfg=SimpleNamespace(
            optimization={"max_steps": 100, "final_lr_ratio": 1.0},
        ),
        uvd_flow_enabled=True,
        _sdedit_active=lambda: False,
        log=lambda *_args, **_kwargs: None,
    )

    Head3DGSLKsReconstructionFinetune._update_learning_rates(system)
    assert optimizer.param_groups[0]["lr"] == 0.02

    system.uvd_flow_enabled = False
    Head3DGSLKsReconstructionFinetune._update_learning_rates(system)
    assert optimizer.param_groups[0]["lr"] == 0.02


def test_first_phase_artifacts_are_named_and_written_exactly_once():
    events = []

    class Strategy:
        def barrier(self, name):
            events.append(("barrier", name))

    class Harness:
        _gaussian_fields = staticmethod(
            Head3DGSLKsReconstructionFinetune._gaussian_fields
        )
        _optimizer_progress = staticmethod(
            Head3DGSLKsReconstructionFinetune._optimizer_progress
        )
        _first_phase_artifacts_enabled = (
            Head3DGSLKsReconstructionFinetune._first_phase_artifacts_enabled
        )
        _first_phase_artifact_name = (
            Head3DGSLKsReconstructionFinetune._first_phase_artifact_name
        )
        _copy_phase_snapshot_to_gaussian = (
            Head3DGSLKsReconstructionFinetune._copy_phase_snapshot_to_gaussian
        )
        _maybe_write_first_phase_artifacts = (
            Head3DGSLKsReconstructionFinetune._maybe_write_first_phase_artifacts
        )

        def _export(self, name, **kwargs):
            events.append(("export", name, kwargs))

        def _render_first_phase_driving_test(self, prefix, fps):
            events.append(("driving", prefix, fps))

        def _set_reference_pose(self):
            return None

    harness = Harness()
    parameter_names = {
        "uv": "_uv",
        "d": "_d",
        "feature": "_features_dc",
        "opacity": "_opacity",
        "scale": "_scaling",
        "rotation": "_rotation",
    }
    gaussian = SimpleNamespace()
    for attribute in parameter_names.values():
        setattr(gaussian, attribute, torch.nn.Parameter(torch.zeros(2, 1)))
    optimizer = torch.optim.Adam(
        [getattr(gaussian, attribute) for attribute in parameter_names.values()]
    )
    for parameter in optimizer.param_groups[0]["params"]:
        optimizer.state[parameter]["step"] = torch.tensor(500.0)
    gaussian.optimizer = optimizer
    harness.gaussian = gaussian
    harness.cfg = SimpleNamespace(
        export_name="mouth",
        first_phase_artifacts={
            "enabled": True,
            "save_gaussian": True,
            "render_driving_test": True,
            "driving_fps": 24,
        },
    )
    harness.sdedit_start_step = 500
    harness.uvd_flow_enabled = False
    harness._first_phase_artifacts_written = False
    harness._first_phase_optimizer_steps = -1
    harness._save_dir = Path(".")
    harness.true_global_step = 500
    harness.global_rank = 0
    harness.trainer = SimpleNamespace(strategy=Strategy())
    harness._sdedit_reference_state = {
        key: getattr(gaussian, attribute).detach().clone()
        for key, attribute in parameter_names.items()
    }

    harness._maybe_write_first_phase_artifacts()
    harness._maybe_write_first_phase_artifacts()

    assert harness._first_phase_artifacts_written is True
    assert harness._first_phase_optimizer_steps == 500
    assert [event for event in events if event[0] == "export"] == [
        (
            "export",
            "mouth_ism",
            {
                "refinement_phase": "first_phase",
                "optimizer_executed_steps": 500,
            },
        )
    ]
    assert [event for event in events if event[0] == "driving"] == [
        ("driving", "mouth_ism_driving_test", 24)
    ]
    assert [event for event in events if event[0] == "barrier"] == [
        ("barrier", "first_phase_artifacts_start"),
        ("barrier", "first_phase_artifacts_end"),
    ]


def test_optimizer_executed_steps_survive_configured_sdedit_adam_reset():
    class Harness:
        _optimizer_progress = staticmethod(
            Head3DGSLKsReconstructionFinetune._optimizer_progress
        )
        _cumulative_optimizer_progress = (
            Head3DGSLKsReconstructionFinetune._cumulative_optimizer_progress
        )
        _reset_adam_state_at_sdedit_boundary = (
            Head3DGSLKsReconstructionFinetune
            ._reset_adam_state_at_sdedit_boundary
        )

        def _sdedit_active(self):
            return True

    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter])
    optimizer.state[parameter]["step"] = torch.tensor(500.0)
    harness = Harness()
    harness.gaussian = SimpleNamespace(optimizer=optimizer)
    harness.cfg = SimpleNamespace(
        sdedit={"reset_optimizer_at_start": True},
    )
    harness._sdedit_optimizer_state_reset = False
    harness._optimizer_executed_step_offset = 0
    harness._manual_micro_step = 0
    harness._consecutive_skipped_optimizer_steps = 0
    harness._sdedit_reference_state = {}
    harness.true_global_step = 500

    harness._reset_adam_state_at_sdedit_boundary()

    assert harness._sdedit_optimizer_state_reset is True
    assert harness._optimizer_executed_step_offset == 500
    assert len(optimizer.state) == 0
    assert harness._cumulative_optimizer_progress(optimizer) == 500.0

    # Adam lazily recreates state in the new SDEdit phase.  Its local 500
    # must be reported as 500 first-phase + 500 second-phase, not as 500.
    optimizer.state[parameter]["step"] = torch.tensor(500.0)
    assert harness._optimizer_progress(optimizer) == 500.0
    assert harness._cumulative_optimizer_progress(optimizer) == 1000.0
    harness._reset_adam_state_at_sdedit_boundary()
    assert harness._optimizer_executed_step_offset == 500
    assert harness._cumulative_optimizer_progress(optimizer) == 1000.0


def test_full_stage_rejects_the_generic_prompt_that_broke_region_repair():
    try:
        validate_full_prompt("A man")
    except ValueError as error:
        assert "per-identity appearance prompt" in str(error)
    else:
        raise AssertionError("Generic full-stage prompt was accepted")

    detailed = (
        "a photorealistic young man with dark hair, natural ears, "
        "a white shirt and a navy suit"
    )
    assert validate_full_prompt(detailed) == detailed


def test_full_stage_maps_generic_regional_prompt_to_identity_prompt():
    detailed = (
        "a photorealistic young man with dark hair, natural ears, "
        "a white shirt and a navy suit"
    )
    assert validate_abstract_prompt(None, detailed) == detailed
    assert validate_abstract_prompt(detailed, detailed) == detailed
    assert validate_abstract_prompt("A man", detailed) == detailed
    assert validate_abstract_prompt("a woman", detailed) == detailed


def test_continuous_camera_preserves_the_calibrated_facelift_orbit():
    K = np.array(
        [[548.99377, 0.0, 256.0], [0.0, 548.99377, 256.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    pivot = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    radius = 2.7
    poses = []
    for azimuth, elevation in ((0.0, 0.0), (90.0, -15.0), (270.0, 15.0)):
        c2w, w2c = _calibrated_orbit_camera(
            azimuth_deg=azimuth,
            elevation_deg=elevation,
            radius=radius,
            pivot=pivot,
            intrinsics=K,
        )
        poses.append(c2w)
        assert np.allclose(c2w @ w2c, np.eye(4), atol=1.0e-5)
        assert np.isclose(np.linalg.det(c2w[:3, :3]), 1.0, atol=1.0e-5)
        assert np.isclose(np.linalg.norm(c2w[:3, 3] - pivot), radius)

    fitted_pivot, fitted_radius = _fit_calibrated_orbit(poses)
    assert np.allclose(fitted_pivot, pivot, atol=1.0e-5)
    assert np.isclose(fitted_radius, radius, atol=1.0e-5)

    c2w = poses[0]
    expected_rotation = np.array(
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    assert np.allclose(c2w[:3, :3], expected_rotation, atol=1.0e-6)


def test_full_teeth_freeze_applies_only_to_closed_mouth_ism():
    system = SimpleNamespace(
        optimization_stage="full",
        cfg=SimpleNamespace(
            freeze_dental_when_closed=True,
            full_protection={},
        ),
        _active_open_mouth=False,
        dental_point_mask=torch.tensor([False, True, True]),
        gaussian=SimpleNamespace(num_gs=3, device=torch.device("cpu")),
        _full_region_protection_enabled=lambda: False,
        _sdedit_active=lambda: False,
    )
    active_mask = (
        Head3DGSLKsReconstructionFinetune._active_trainable_point_mask
    )
    assert active_mask(system).tolist() == [True, False, False]

    system._active_open_mouth = True
    assert active_mask(system).tolist() == [True, True, True]

    system._active_open_mouth = False
    system._sdedit_active = lambda: True
    assert active_mask(system).tolist() == [True, True, True]


def test_full_stage_rejects_unverified_or_incomplete_mouth_exports():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "mouth_params.npz"
        np.savez(path, optimization_stage=np.asarray("mouth"))
        try:
            require_verified_mouth_sidecar(path)
        except ValueError as error:
            assert "predates optimizer-progress verification" in str(error)
        else:
            raise AssertionError("Unverified mouth sidecar was accepted")

        np.savez(
            path,
            optimization_stage=np.asarray("mouth"),
            optimizer_executed_steps=np.int64(1),
        )
        try:
            require_verified_mouth_sidecar(path)
        except ValueError as error:
            assert "incomplete" in str(error)
        else:
            raise AssertionError("One-step mouth smoke test was accepted")

        np.savez(
            path,
            optimization_stage=np.asarray("mouth"),
            optimizer_executed_steps=np.int64(1000),
        )
        assert require_verified_mouth_sidecar(path) == 1000


def test_full_stage_validates_new_cumulative_step_accounting():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "mouth_params.npz"
        np.savez(
            path,
            optimization_stage=np.asarray("mouth"),
            optimizer_executed_steps=np.int64(1000),
            optimizer_local_executed_steps=np.int64(500),
            optimizer_executed_step_offset=np.int64(500),
            optimizer_step_accounting_version=np.int64(2),
        )
        assert require_verified_mouth_sidecar(path) == 1000

        np.savez(
            path,
            optimization_stage=np.asarray("mouth"),
            optimizer_executed_steps=np.int64(500),
            optimizer_local_executed_steps=np.int64(500),
            optimizer_executed_step_offset=np.int64(500),
            optimizer_step_accounting_version=np.int64(2),
        )
        try:
            require_verified_mouth_sidecar(path)
        except ValueError as error:
            assert "inconsistent cumulative" in str(error)
        else:
            raise AssertionError("Inconsistent cumulative metadata was accepted")


def load_tests(
    _loader: unittest.TestLoader,
    _tests: unittest.TestSuite,
    _pattern: str | None,
) -> unittest.TestSuite:
    """Keep this regression file runnable when pytest is unavailable."""

    suite = unittest.TestSuite()
    for test in (
        test_animportrait3d_direction_boundaries_match_reference,
        test_animportrait3d_prompt_prefixes_match_helper,
        test_mouth_entrypoint_adds_reference_region_prefix_once,
        test_refinement_configs_keep_the_repaired_precision_and_camera_ranges,
        test_guidance_mode_switch_preserves_ism_as_an_explicit_ablation,
        test_guidance_checkpoint_signature_records_uvd_coupling_only,
        test_uvd_ism_uses_the_same_negative_prompt_as_raw_ism,
        test_uvd_correspondence_reuses_the_exact_rgb_crop_plan,
        test_uvd_variance_rejection_never_falls_through_to_a_rear_layer,
        test_animportrait3d_ism_uses_the_explicit_noise_in_its_inversion,
        test_uvd_mode_dispatches_to_the_same_ism_with_canonical_noise,
        test_uvd_mode_requires_the_animportrait3d_ism_variant,
        test_uvd_and_raw_ism_use_the_same_adam_learning_rate,
        test_first_phase_artifacts_are_named_and_written_exactly_once,
        test_optimizer_executed_steps_survive_configured_sdedit_adam_reset,
        test_full_stage_rejects_the_generic_prompt_that_broke_region_repair,
        test_full_stage_maps_generic_regional_prompt_to_identity_prompt,
        test_continuous_camera_preserves_the_calibrated_facelift_orbit,
        test_full_teeth_freeze_applies_only_to_closed_mouth_ism,
        test_full_stage_rejects_unverified_or_incomplete_mouth_exports,
        test_full_stage_validates_new_cumulative_step_accounting,
    ):
        suite.addTest(unittest.FunctionTestCase(test))
    return suite
