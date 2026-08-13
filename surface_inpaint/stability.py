"""Pose-envelope covariance stabilization for FLAME-bound UVD Gaussians.

The canonical covariance of a UVD Gaussian is pushed to a posed world frame
by the local surface Jacobian ``J``::

    C_world = J @ C_uvd @ J.T

This module detects streaks from the *two largest* world-space standard
deviations, ``s0 >= s1 >= s2``.  In particular, it deliberately uses
``s0 / s1`` rather than ``s0 / s2``: a healthy, thin surface splat may have a
very small normal-axis scale ``s2`` without being a long in-plane streak.

The tensor helpers are CPU-safe and do not import the FLAME or CUDA extension
stack.  :func:`stabilize_uvd_covariances` is the small adapter used with
``GaussianFlameUVModel``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch


Tensor = torch.Tensor
NamedPose = Tuple[str, Any]


@dataclass(frozen=True)
class StabilityConfig:
    """Configuration for deterministic Stage-0 covariance stabilization.

    ``absolute_max_scale`` is the hard outlier threshold.  ``absolute_cap`` is
    the value used when repairing a detected point and defaults to that same
    threshold.  Keeping the two fields separate also permits migration from
    the older ``absolute_scale_threshold`` / ``streak_max_world_scale``
    configuration.
    """

    enabled: bool = True
    passes: int = 2
    absolute_max_scale: float = 0.100
    absolute_cap: Optional[float] = None
    min_streak_scale: float = 0.020
    max_planar_aspect: float = 100.0
    repair_margin: float = 1.0
    min_canonical_scale: float = 1.0e-8
    epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.passes < 1:
            raise ValueError("stability passes must be at least one")
        if self.absolute_max_scale <= 0.0:
            raise ValueError("absolute_max_scale must be positive")
        if self.resolved_absolute_cap <= 0.0:
            raise ValueError("absolute_cap must be positive")
        if self.resolved_absolute_cap > self.absolute_max_scale:
            raise ValueError(
                "absolute_cap cannot exceed absolute_max_scale"
            )
        if self.min_streak_scale < 0.0:
            raise ValueError("min_streak_scale cannot be negative")
        if self.max_planar_aspect < 1.0:
            raise ValueError("max_planar_aspect must be at least one")
        if not 0.0 < self.repair_margin <= 1.0:
            raise ValueError("repair_margin must be in (0, 1]")
        if self.min_canonical_scale <= 0.0:
            raise ValueError("min_canonical_scale must be positive")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

    @property
    def resolved_absolute_cap(self) -> float:
        if self.absolute_cap is None:
            return float(self.absolute_max_scale)
        return float(self.absolute_cap)

    def to_dict(self) -> Dict[str, Any]:
        """Return resolved, JSON-serializable configuration values."""

        return {
            "enabled": bool(self.enabled),
            "passes": int(self.passes),
            "absolute_max_scale": float(self.absolute_max_scale),
            "absolute_cap": float(self.resolved_absolute_cap),
            "min_streak_scale": float(self.min_streak_scale),
            "max_planar_aspect": float(self.max_planar_aspect),
            "repair_margin": float(self.repair_margin),
            "min_canonical_scale": float(self.min_canonical_scale),
            "epsilon": float(self.epsilon),
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "StabilityConfig":
        """Build a config while accepting current and legacy field names.

        A full pipeline mapping may be passed; the first nested mapping named
        ``stability``, ``covariance_stability``, or ``stage0_stability`` is
        selected automatically.
        """

        source = values
        for section in (
            "stability",
            "covariance_stability",
            "stage0_stability",
        ):
            nested = source.get(section)
            if isinstance(nested, Mapping):
                source = nested
                break

        def first(keys: Sequence[str], default: Any) -> Any:
            for key in keys:
                if key in source:
                    return source[key]
            return default

        absolute_max = float(
            first(
                ("absolute_max_scale", "absolute_scale_threshold"),
                cls.absolute_max_scale,
            )
        )
        absolute_cap_value = first(
            (
                "absolute_cap",
                "absolute_scale_cap",
                "streak_max_world_scale",
            ),
            None,
        )
        return cls(
            enabled=bool(
                first(("enabled", "isotropize_streaks"), cls.enabled)
            ),
            passes=int(first(("passes", "max_passes"), cls.passes)),
            absolute_max_scale=absolute_max,
            absolute_cap=(
                None
                if absolute_cap_value is None
                else float(absolute_cap_value)
            ),
            min_streak_scale=float(
                first(
                    ("min_streak_scale", "streak_scale_threshold"),
                    cls.min_streak_scale,
                )
            ),
            max_planar_aspect=float(
                first(
                    ("max_planar_aspect", "streak_aspect_threshold"),
                    cls.max_planar_aspect,
                )
            ),
            repair_margin=float(
                first(("repair_margin",), cls.repair_margin)
            ),
            min_canonical_scale=float(
                first(
                    ("min_canonical_scale",),
                    cls.min_canonical_scale,
                )
            ),
            epsilon=float(first(("epsilon", "eps"), cls.epsilon)),
        )


@dataclass
class CovarianceRepairResult:
    """Tensor diagnostics from one pose-local repair operation."""

    canonical_covariance: Tensor
    flagged: Tensor
    absolute_violations: Tensor
    streak_violations: Tensor
    before_world_scales: Tensor
    after_world_scales: Tensor
    target_s0: Tensor


def _symmetrize(matrix: Tensor) -> Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _check_matrix_batch(matrix: Tensor, name: str) -> None:
    if matrix.ndim != 3 or matrix.shape[-2:] != (3, 3):
        raise ValueError(
            "{} must have shape [N, 3, 3], got {}".format(
                name, tuple(matrix.shape)
            )
        )
    if not bool(torch.isfinite(matrix).all()):
        invalid = int((~torch.isfinite(matrix)).sum().item())
        raise ValueError(
            "{} contains {} non-finite values".format(name, invalid)
        )


def quaternion_to_matrix(quaternion: Tensor) -> Tensor:
    """Convert real-first ``[w, x, y, z]`` quaternions to matrices."""

    if quaternion.ndim != 2 or quaternion.shape[-1] != 4:
        raise ValueError(
            "quaternion must have shape [N, 4], got {}".format(
                tuple(quaternion.shape)
            )
        )
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    tiny = torch.finfo(quaternion.dtype).tiny
    quaternion = quaternion / norm.clamp_min(tiny)
    w, x, y, z = quaternion.unbind(dim=-1)
    two = quaternion.new_tensor(2.0)
    matrix = torch.stack(
        (
            1.0 - two * (y * y + z * z),
            two * (x * y - w * z),
            two * (x * z + w * y),
            two * (x * y + w * z),
            1.0 - two * (x * x + z * z),
            two * (y * z - w * x),
            two * (x * z - w * y),
            two * (y * z + w * x),
            1.0 - two * (x * x + y * y),
        ),
        dim=-1,
    )
    return matrix.reshape(-1, 3, 3)


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Convert proper rotation matrices to real-first quaternions."""

    _check_matrix_batch(matrix, "matrix")
    result = torch.empty(
        (matrix.shape[0], 4), dtype=matrix.dtype, device=matrix.device
    )
    diagonal = torch.diagonal(matrix, dim1=-2, dim2=-1)
    trace = diagonal.sum(dim=-1)
    eps = torch.finfo(matrix.dtype).eps

    trace_case = trace > 0.0
    if bool(trace_case.any()):
        selected = matrix[trace_case]
        scale = (
            torch.sqrt((trace[trace_case] + 1.0).clamp_min(eps)) * 2.0
        )
        result[trace_case, 0] = 0.25 * scale
        result[trace_case, 1] = (
            selected[:, 2, 1] - selected[:, 1, 2]
        ) / scale
        result[trace_case, 2] = (
            selected[:, 0, 2] - selected[:, 2, 0]
        ) / scale
        result[trace_case, 3] = (
            selected[:, 1, 0] - selected[:, 0, 1]
        ) / scale

    remaining = ~trace_case
    x_case = (
        remaining
        & (diagonal[:, 0] >= diagonal[:, 1])
        & (diagonal[:, 0] >= diagonal[:, 2])
    )
    if bool(x_case.any()):
        selected = matrix[x_case]
        scale = torch.sqrt(
            (
                1.0
                + diagonal[x_case, 0]
                - diagonal[x_case, 1]
                - diagonal[x_case, 2]
            ).clamp_min(eps)
        ) * 2.0
        result[x_case, 0] = (
            selected[:, 2, 1] - selected[:, 1, 2]
        ) / scale
        result[x_case, 1] = 0.25 * scale
        result[x_case, 2] = (
            selected[:, 0, 1] + selected[:, 1, 0]
        ) / scale
        result[x_case, 3] = (
            selected[:, 0, 2] + selected[:, 2, 0]
        ) / scale

    y_case = (
        remaining
        & (~x_case)
        & (diagonal[:, 1] >= diagonal[:, 2])
    )
    if bool(y_case.any()):
        selected = matrix[y_case]
        scale = torch.sqrt(
            (
                1.0
                + diagonal[y_case, 1]
                - diagonal[y_case, 0]
                - diagonal[y_case, 2]
            ).clamp_min(eps)
        ) * 2.0
        result[y_case, 0] = (
            selected[:, 0, 2] - selected[:, 2, 0]
        ) / scale
        result[y_case, 1] = (
            selected[:, 0, 1] + selected[:, 1, 0]
        ) / scale
        result[y_case, 2] = 0.25 * scale
        result[y_case, 3] = (
            selected[:, 1, 2] + selected[:, 2, 1]
        ) / scale

    z_case = remaining & (~x_case) & (~y_case)
    if bool(z_case.any()):
        selected = matrix[z_case]
        scale = torch.sqrt(
            (
                1.0
                + diagonal[z_case, 2]
                - diagonal[z_case, 0]
                - diagonal[z_case, 1]
            ).clamp_min(eps)
        ) * 2.0
        result[z_case, 0] = (
            selected[:, 1, 0] - selected[:, 0, 1]
        ) / scale
        result[z_case, 1] = (
            selected[:, 0, 2] + selected[:, 2, 0]
        ) / scale
        result[z_case, 2] = (
            selected[:, 1, 2] + selected[:, 2, 1]
        ) / scale
        result[z_case, 3] = 0.25 * scale

    return result / torch.linalg.vector_norm(
        result, dim=-1, keepdim=True
    ).clamp_min(eps)


def covariance_from_scaling_rotation(
    scaling: Tensor, quaternion: Tensor
) -> Tensor:
    """Construct canonical covariance from scale and real-first rotation."""

    if scaling.ndim != 2 or scaling.shape[-1] != 3:
        raise ValueError(
            "scaling must have shape [N, 3], got {}".format(
                tuple(scaling.shape)
            )
        )
    if scaling.shape[0] != quaternion.shape[0]:
        raise ValueError("scaling and quaternion batch sizes differ")
    rotation = quaternion_to_matrix(quaternion)
    factor = rotation * scaling[:, None, :]
    return torch.bmm(factor, factor.transpose(1, 2))


def scaling_rotation_from_covariance(
    covariance: Tensor, min_scale: float = 1.0e-8
) -> Tuple[Tensor, Tensor]:
    """Factor positive-semidefinite covariance into scale and quaternion."""

    _check_matrix_batch(covariance, "covariance")
    if min_scale <= 0.0:
        raise ValueError("min_scale must be positive")
    values, vectors = torch.linalg.eigh(_symmetrize(covariance))
    values = values.clamp_min(float(min_scale) ** 2)
    improper = torch.linalg.det(vectors) < 0.0
    if bool(improper.any()):
        vectors = vectors.clone()
        vectors[improper, :, 2] *= -1.0
    return values.sqrt(), matrix_to_quaternion(vectors)


def pushforward_covariance(
    canonical_covariance: Tensor, jacobian: Tensor
) -> Tensor:
    """Push a UVD covariance through a posed surface Jacobian."""

    _check_matrix_batch(canonical_covariance, "canonical_covariance")
    _check_matrix_batch(jacobian, "jacobian")
    if canonical_covariance.shape[0] != jacobian.shape[0]:
        raise ValueError("covariance and jacobian batch sizes differ")
    return _symmetrize(
        torch.bmm(
            torch.bmm(jacobian, canonical_covariance),
            jacobian.transpose(1, 2),
        )
    )


def pullback_covariance(world_covariance: Tensor, jacobian: Tensor) -> Tensor:
    """Pull a world covariance back with ``J^{-1} C J^{-T}``."""

    _check_matrix_batch(world_covariance, "world_covariance")
    _check_matrix_batch(jacobian, "jacobian")
    if world_covariance.shape[0] != jacobian.shape[0]:
        raise ValueError("covariance and jacobian batch sizes differ")
    identity = torch.eye(
        3, dtype=jacobian.dtype, device=jacobian.device
    ).expand_as(jacobian)
    inverse = torch.linalg.solve(jacobian, identity)
    return _symmetrize(
        torch.bmm(
            torch.bmm(inverse, world_covariance),
            inverse.transpose(1, 2),
        )
    )


def world_principal_scales(
    canonical_covariance: Tensor, jacobian: Tensor
) -> Tuple[Tensor, Tensor]:
    """Return ``s0 >= s1 >= s2`` and corresponding world principal axes."""

    world_covariance = pushforward_covariance(
        canonical_covariance, jacobian
    )
    values, vectors = torch.linalg.eigh(world_covariance)
    values = values.clamp_min(0.0)
    order = torch.tensor(
        (2, 1, 0), dtype=torch.long, device=values.device
    )
    scales = values.sqrt().index_select(1, order)
    axes = vectors.index_select(2, order)
    return scales, axes


def classify_world_scales(
    world_scales: Tensor,
    config: Union[StabilityConfig, Mapping[str, Any]],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Classify hard-size and in-plane streak violations.

    The streak ratio is exactly ``s0 / s1``.  ``s2`` is intentionally absent
    from this decision.
    """

    resolved = _resolve_config(config)
    if world_scales.ndim != 2 or world_scales.shape[-1] != 3:
        raise ValueError(
            "world_scales must have shape [N, 3], got {}".format(
                tuple(world_scales.shape)
            )
        )
    denominator = world_scales[:, 1].clamp_min(resolved.epsilon)
    planar_aspect = world_scales[:, 0] / denominator
    # Eigendecomposition followed by J^{-1} pullback can round a value capped
    # exactly at the boundary a few ulps above it.  Ignore only that numerical
    # rebound so repeated passes converge instead of rewriting the same point.
    comparison_slack = max(
        resolved.epsilon,
        float(torch.finfo(world_scales.dtype).eps) * 16.0,
    )
    absolute = world_scales[:, 0] > (
        resolved.absolute_max_scale * (1.0 + comparison_slack)
    )
    streak = (
        (world_scales[:, 0] > resolved.min_streak_scale)
        & (
            planar_aspect
            > resolved.max_planar_aspect * (1.0 + comparison_slack)
        )
    )
    return absolute, streak, absolute | streak


def cap_world_covariance(
    canonical_covariance: Tensor,
    jacobian: Tensor,
    config: Union[StabilityConfig, Mapping[str, Any]],
) -> CovarianceRepairResult:
    """Cap pose-local covariance outliers while preserving world axes.

    A streak-only violation changes just its largest world principal scale:

    ``s0' = min(s0, absolute_cap, max_planar_aspect * s1)``.

    An absolute-size violation is different: a Gaussian can have two or three
    giant axes.  Capping only ``s0`` merely promotes the old ``s1`` to the new
    maximum (the source of the residual long splats in open-mouth views), so
    every principal scale of an absolute outlier is clamped to
    ``absolute_cap`` first.

    The repaired world covariance is then pulled back to canonical UVD space.
    """

    resolved = _resolve_config(config)
    before_scales, world_axes = world_principal_scales(
        canonical_covariance, jacobian
    )
    absolute, streak, flagged = classify_world_scales(
        before_scales, resolved
    )
    target_s0 = torch.minimum(
        before_scales[:, 0],
        torch.minimum(
            torch.full_like(
                before_scales[:, 0],
                resolved.resolved_absolute_cap * resolved.repair_margin,
            ),
            resolved.max_planar_aspect
            * before_scales[:, 1]
            * resolved.repair_margin,
        ),
    )
    repaired_canonical = canonical_covariance
    after_scales = before_scales
    if bool(flagged.any()):
        # Work only on selected points: a Stage-1 avatar can contain hundreds
        # of thousands of healthy Gaussians, for which a batched J solve would
        # be wasted.  ``axis_scales`` remains paired with the original world
        # axes while constructing the covariance.
        selected_axes = world_axes[flagged]
        axis_scales = before_scales[flagged].clone()
        selected_absolute = absolute[flagged]
        if bool(selected_absolute.any()):
            absolute_cap = (
                resolved.resolved_absolute_cap * resolved.repair_margin
            )
            axis_scales[selected_absolute] = torch.minimum(
                axis_scales[selected_absolute],
                torch.full_like(axis_scales[selected_absolute], absolute_cap),
            )
        # Re-evaluate the planar cap after an absolute clamp because s1 may
        # also have changed.  Healthy surface splats keep s1/s2 untouched.
        axis_scales[:, 0] = torch.minimum(
            axis_scales[:, 0],
            resolved.max_planar_aspect
            * axis_scales[:, 1]
            * resolved.repair_margin,
        )
        repaired_world = torch.bmm(
            selected_axes * axis_scales.square()[:, None, :],
            selected_axes.transpose(1, 2),
        )
        pulled = pullback_covariance(
            repaired_world, jacobian[flagged]
        )
        repaired_canonical = canonical_covariance.clone()
        indices = torch.nonzero(flagged, as_tuple=False).squeeze(1)
        repaired_canonical.index_copy_(0, indices, pulled)
        # Re-sort after reconstruction.  If an aggressive absolute cap falls
        # below the old s1, the old s1 correctly becomes the new s0.
        after_scales, _ = world_principal_scales(
            repaired_canonical, jacobian
        )
    return CovarianceRepairResult(
        canonical_covariance=repaired_canonical,
        flagged=flagged,
        absolute_violations=absolute,
        streak_violations=streak,
        before_world_scales=before_scales,
        after_world_scales=after_scales,
        target_s0=target_s0,
    )


def _resolve_config(
    config: Union[StabilityConfig, Mapping[str, Any]]
) -> StabilityConfig:
    if isinstance(config, StabilityConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("config must be StabilityConfig or a mapping")
    return StabilityConfig.from_mapping(config)


def _model_value(model: Any, name: str) -> Tensor:
    if not hasattr(model, name):
        raise AttributeError("model has no {!r} attribute".format(name))
    value = getattr(model, name)
    if callable(value):
        value = value()
    if not isinstance(value, torch.Tensor):
        raise TypeError("model.{} must resolve to a tensor".format(name))
    return value


def _model_canonical_covariance(model: Any) -> Tensor:
    scaling = _model_value(model, "get_scaling")
    rotation = _model_value(model, "get_rotation")
    return covariance_from_scaling_rotation(scaling, rotation)


def _model_current_jacobian(model: Any) -> Tensor:
    for name in ("current_uvd_jacobian", "get_uvd_jacobian"):
        method = getattr(model, name, None)
        if callable(method):
            jacobian = method()
            if not isinstance(jacobian, torch.Tensor):
                raise TypeError(
                    "model.{}() must return a tensor".format(name)
                )
            return jacobian

    flame_geometry = getattr(model, "_flame_verts_and_normals", None)
    uvd_jacobian = getattr(model, "_uvd_jacobian", None)
    if not callable(flame_geometry) or not callable(uvd_jacobian):
        raise AttributeError(
            "model must expose current_uvd_jacobian(), "
            "get_uvd_jacobian(), or GaussianFlameUVModel's private "
            "FLAME/Jacobian methods"
        )
    vertices, normals = flame_geometry()
    return uvd_jacobian(vertices, normals)


def _write_model_covariance(
    model: Any,
    covariance: Tensor,
    mask: Tensor,
    min_scale: float,
) -> int:
    count = int(mask.sum().item())
    if count == 0:
        return 0
    if not hasattr(model, "_scaling") or not hasattr(model, "_rotation"):
        raise AttributeError(
            "model must expose writable _scaling and _rotation tensors"
        )

    selected = covariance[mask]
    converter = getattr(model, "_covariance_to_scaling_rotation", None)
    if callable(converter):
        scaling, rotation = converter(
            selected, min_scale=float(min_scale)
        )
    else:
        scaling, rotation = scaling_rotation_from_covariance(
            selected, min_scale=float(min_scale)
        )

    inverse_activation = getattr(
        model, "scaling_inverse_activation", None
    )
    if callable(inverse_activation):
        raw_scaling = inverse_activation(scaling)
    else:
        raw_scaling = scaling.log()

    target_scaling = getattr(model, "_scaling")
    target_rotation = getattr(model, "_rotation")
    indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
    target_scaling.index_copy_(
        0,
        indices,
        raw_scaling.to(
            dtype=target_scaling.dtype, device=target_scaling.device
        ),
    )
    target_rotation.index_copy_(
        0,
        indices,
        rotation.to(
            dtype=target_rotation.dtype, device=target_rotation.device
        ),
    )
    return count


def _safe_max(values: Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.max().item())


def _scale_statistics(
    label: str,
    scales: Tensor,
    config: StabilityConfig,
) -> Tuple[Dict[str, Any], Tensor]:
    absolute, streak, flagged = classify_world_scales(scales, config)
    aspect = scales[:, 0] / scales[:, 1].clamp_min(config.epsilon)
    aspect_active = aspect[
        scales[:, 0] > float(config.min_streak_scale)
    ]
    stats = {
        "label": str(label),
        "points": int(scales.shape[0]),
        "flagged": int(flagged.sum().item()),
        "absolute_violations": int(absolute.sum().item()),
        "streak_violations": int(streak.sum().item()),
        "max_s0": _safe_max(scales[:, 0]),
        "mean_s0": (
            float(scales[:, 0].mean().item())
            if scales.shape[0]
            else 0.0
        ),
        "max_planar_aspect": _safe_max(aspect),
        "max_planar_aspect_above_min_streak": _safe_max(
            aspect_active
        ),
    }
    return stats, flagged


def _scan_pose_envelope(
    model: Any,
    named_poses: Sequence[NamedPose],
    set_pose: Callable[[Any], None],
    config: StabilityConfig,
) -> Dict[str, Any]:
    point_count = int(_model_value(model, "get_scaling").shape[0])
    first_tensor = _model_value(model, "get_scaling")
    flagged_union = torch.zeros(
        point_count, dtype=torch.bool, device=first_tensor.device
    )
    absolute_union = torch.zeros_like(flagged_union)
    streak_union = torch.zeros_like(flagged_union)
    pose_stats = []
    flagged_observations = 0
    max_s0 = 0.0
    max_aspect = 0.0
    max_active_aspect = 0.0

    for label, pose in named_poses:
        set_pose(pose)
        canonical = _model_canonical_covariance(model)
        jacobian = _model_current_jacobian(model).to(
            dtype=canonical.dtype, device=canonical.device
        )
        scales, _ = world_principal_scales(canonical, jacobian)
        stats, flagged = _scale_statistics(str(label), scales, config)
        absolute, streak, _ = classify_world_scales(scales, config)
        flagged_union |= flagged
        absolute_union |= absolute
        streak_union |= streak
        flagged_observations += int(flagged.sum().item())
        max_s0 = max(max_s0, float(stats["max_s0"]))
        max_aspect = max(
            max_aspect, float(stats["max_planar_aspect"])
        )
        max_active_aspect = max(
            max_active_aspect,
            float(stats["max_planar_aspect_above_min_streak"]),
        )
        pose_stats.append(stats)

    return {
        "points": point_count,
        "poses": pose_stats,
        "flagged_unique": int(flagged_union.sum().item()),
        "flagged_observations": int(flagged_observations),
        "absolute_unique": int(absolute_union.sum().item()),
        "streak_unique": int(streak_union.sum().item()),
        "max_s0": float(max_s0),
        "max_planar_aspect": float(max_aspect),
        "max_planar_aspect_above_min_streak": float(
            max_active_aspect
        ),
    }


@torch.no_grad()
def stabilize_uvd_covariances(
    model: Any,
    named_poses: Sequence[NamedPose],
    set_pose: Callable[[Any], None],
    config: Union[StabilityConfig, Mapping[str, Any]],
    reference_pose: Optional[Any] = None,
) -> Dict[str, Any]:
    """Stabilize a Gaussian model over a named FLAME-pose envelope.

    Args:
        model: A ``GaussianFlameUVModel``-compatible object.
        named_poses: Sequence of ``(label, pose)`` pairs.
        set_pose: Callback invoked as ``set_pose(pose)``.
        config: :class:`StabilityConfig` or a compatible mapping.
        reference_pose: Pose restored in ``finally`` after every exit path.

    Returns:
        A JSON-serializable report containing envelope-wide before/after
        summaries and per-pose statistics for every repair pass.

    The model's canonical covariance is transactionally restored if scanning
    or repair raises.  A successful call only changes ``_scaling`` and
    ``_rotation``.
    """

    resolved = _resolve_config(config)
    poses = list(named_poses)
    if not poses:
        raise ValueError("named_poses must contain at least one pose")
    for item in poses:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(
                "each named pose must be a (label, pose) pair"
            )
    if not callable(set_pose):
        raise TypeError("set_pose must be callable")

    report: Dict[str, Any] = {
        "enabled": bool(resolved.enabled),
        "config": resolved.to_dict(),
        "pose_labels": [str(label) for label, _ in poses],
        "passes_requested": int(resolved.passes),
        "passes_completed": 0,
        "total_updates": 0,
        "unique_updated": 0,
        "converged": False,
        "before": None,
        "passes": [],
        "after": None,
    }
    if not resolved.enabled:
        if reference_pose is not None:
            set_pose(reference_pose)
        report["converged"] = True
        return report

    if not hasattr(model, "_scaling") or not hasattr(model, "_rotation"):
        raise AttributeError(
            "model must expose _scaling and _rotation tensors"
        )
    saved_scaling = model._scaling.detach().clone()
    saved_rotation = model._rotation.detach().clone()
    point_count = int(saved_scaling.shape[0])
    updated_union = torch.zeros(
        point_count, dtype=torch.bool, device=saved_scaling.device
    )

    try:
        report["before"] = _scan_pose_envelope(
            model, poses, set_pose, resolved
        )

        for pass_index in range(resolved.passes):
            pass_union = torch.zeros_like(updated_union)
            pose_reports = []
            pass_updates = 0

            for label, pose in poses:
                set_pose(pose)
                canonical = _model_canonical_covariance(model)
                jacobian = _model_current_jacobian(model).to(
                    dtype=canonical.dtype, device=canonical.device
                )
                repair = cap_world_covariance(
                    canonical, jacobian, resolved
                )
                before_stats, _ = _scale_statistics(
                    str(label), repair.before_world_scales, resolved
                )
                after_stats, _ = _scale_statistics(
                    str(label), repair.after_world_scales, resolved
                )
                updated = _write_model_covariance(
                    model,
                    repair.canonical_covariance,
                    repair.flagged,
                    resolved.min_canonical_scale,
                )
                pass_updates += updated
                pass_union |= repair.flagged
                updated_union |= repair.flagged
                selected_targets = repair.target_s0[repair.flagged]
                pose_reports.append(
                    {
                        "label": str(label),
                        "updated": int(updated),
                        "before": before_stats,
                        "after_local_cap": after_stats,
                        "target_s0_min": (
                            float(selected_targets.min().item())
                            if selected_targets.numel()
                            else None
                        ),
                        "target_s0_max": (
                            float(selected_targets.max().item())
                            if selected_targets.numel()
                            else None
                        ),
                    }
                )

            report["passes"].append(
                {
                    "index": int(pass_index + 1),
                    "updates": int(pass_updates),
                    "unique_updated": int(pass_union.sum().item()),
                    "poses": pose_reports,
                }
            )
            report["passes_completed"] = int(pass_index + 1)
            report["total_updates"] += int(pass_updates)
            if pass_updates == 0:
                break

        report["unique_updated"] = int(updated_union.sum().item())
        report["after"] = _scan_pose_envelope(
            model, poses, set_pose, resolved
        )
        report["converged"] = (
            int(report["after"]["flagged_unique"]) == 0
        )
        return report
    except Exception:
        model._scaling.copy_(saved_scaling)
        model._rotation.copy_(saved_rotation)
        raise
    finally:
        if reference_pose is not None:
            set_pose(reference_pose)


# A migration-friendly verb for callers that still refer to Stage-0 as
# "sanitation".  Both names intentionally share one implementation.
sanitize_uvd_covariances = stabilize_uvd_covariances


__all__ = [
    "CovarianceRepairResult",
    "StabilityConfig",
    "cap_world_covariance",
    "classify_world_scales",
    "covariance_from_scaling_rotation",
    "matrix_to_quaternion",
    "pullback_covariance",
    "pushforward_covariance",
    "quaternion_to_matrix",
    "sanitize_uvd_covariances",
    "scaling_rotation_from_covariance",
    "stabilize_uvd_covariances",
    "world_principal_scales",
]
