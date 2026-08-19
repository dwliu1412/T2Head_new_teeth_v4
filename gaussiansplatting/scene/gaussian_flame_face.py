import math
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from flame_model.flame_teeth import FlameHead
from plyfile import PlyData, PlyElement
from pytorch3d.transforms import matrix_to_quaternion
from simple_knn._C import distCUDA2
from torch import nn

from gaussiansplatting.scene.gaussian_model import GaussianModel
from gaussiansplatting.utils.general_utils import (
    build_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussiansplatting.utils.sh_utils import RGB2SH
from gaussiansplatting.utils.system_utils import mkdir_p


class GaussianFlameUVModel(GaussianModel):
    """FLAME-bound Gaussians with UVD centres and face-local ellipsoids.

    Each Gaussian is attached to one FLAME face through ``(face_idx, u, v, d)``.
    The centre keeps the existing movable UV plus normal-offset representation.
    Scale and rotation are learned in the orthonormal frame of the parent face,
    following GaussianAvatars, and are explicitly composed with the posed face
    frame before rasterization.  No UVD Jacobian is used for the ellipsoid.
    """

    EPS = 1e-6
    UV_EPS = 1e-6
    SIGMA_FLOOR = 1e-4
    DEFAULT_MAX_GAUSSIANS = 500_000
    SCALE_ROTATION_COMMENT = "scale_rotation_space=flame_face_local_v1"
    # Effective normalized colours loaded by AnimPortrait3D after its rigged
    # point-cloud initializer stores the RGB fields as uint8.
    ANIM_PORTRAIT3D_TEETH_RGB = (
        141.0 / 255.0,
        133.0 / 255.0,
        122.0 / 255.0,
    )
    ANIM_PORTRAIT3D_ORAL_RGB = (
        64.0 / 255.0,
        30.0 / 255.0,
        29.0 / 255.0,
    )
    covariance_space = "face_local"
    requires_precomputed_covariance = False

    def setup_functions(self):
        def covariance_from_scaling_rotation(
            scaling, scaling_modifier, rotation
        ):
            rotation_matrix = build_rotation(F.normalize(rotation, dim=-1))
            factor = rotation_matrix * (
                float(scaling_modifier) * scaling
            )[:, None, :]
            covariance = torch.bmm(factor, factor.transpose(1, 2))
            return strip_symmetric(covariance)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, device: str = "cuda"):
        super().__init__(sh_degree)
        self.device = torch.device(device)
        self.num_betas = 300
        self.num_expression = 100
        self.model = FlameHead(
            shape_params=self.num_betas,
            expr_params=self.num_expression,
            add_teeth=True,
        ).to(self.device)

        self._faces = self.model.faces.long().to(self.device)
        self.vt = self.model.verts_uvs.float().to(self.device)
        self.ft = self.model.textures_idx.long().to(self.device)
        self._precompute_uv_differentials()
        self._build_uv_grid()
        self._build_face_masks()

        self.flame_scale = 0.0
        self.center = torch.zeros(3, dtype=torch.float32, device=self.device)
        self.scale = torch.ones((), dtype=torch.float32, device=self.device)
        self.num_gs = 0
        self._uv = torch.empty((0, 2), dtype=torch.float32, device=self.device)
        self._d = torch.empty((0, 1), dtype=torch.float32, device=self.device)
        self._face_idx = torch.empty((0,), dtype=torch.long, device=self.device)
        self._shape = torch.empty((0,), dtype=torch.float32, device=self.device)
        self._expression = torch.empty((0,), dtype=torch.float32, device=self.device)
        self._global_orient = torch.empty((0,), dtype=torch.float32, device=self.device)
        self._neck_pose = torch.empty((0,), dtype=torch.float32, device=self.device)
        self._jaw_pose = torch.empty((0,), dtype=torch.float32, device=self.device)
        self._leye_pose = torch.empty((0,), dtype=torch.float32, device=self.device)
        self._reye_pose = torch.empty((0,), dtype=torch.float32, device=self.device)
        self._translation = torch.empty((0,), dtype=torch.float32, device=self.device)

    def _precompute_uv_differentials(self) -> None:
        tri = self.vt[self.ft]
        uv_matrix = torch.stack(
            [tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]], dim=-1
        )
        determinant = (
            uv_matrix[:, 0, 0] * uv_matrix[:, 1, 1]
            - uv_matrix[:, 0, 1] * uv_matrix[:, 1, 0]
        )
        valid = determinant.abs() > 1e-12
        inverse = torch.empty_like(uv_matrix)
        safe_determinant = torch.where(valid, determinant, torch.ones_like(determinant))
        inverse[:, 0, 0] = uv_matrix[:, 1, 1] / safe_determinant
        inverse[:, 0, 1] = -uv_matrix[:, 0, 1] / safe_determinant
        inverse[:, 1, 0] = -uv_matrix[:, 1, 0] / safe_determinant
        inverse[:, 1, 1] = uv_matrix[:, 0, 0] / safe_determinant
        if (~valid).any():
            inverse[~valid] = torch.linalg.pinv(uv_matrix[~valid])
        self.face_uv_inv = inverse
        self.face_uv_valid = valid

    def _build_face_masks(self) -> None:
        def region_mask(region: str) -> torch.Tensor:
            mask = torch.zeros(self.ft.shape[0], dtype=torch.bool, device=self.device)
            try:
                face_ids = self.model.mask.get_fid_by_region(region).to(self.device)
            except (AttributeError, KeyError):
                return mask
            face_ids = face_ids[(face_ids >= 0) & (face_ids < mask.numel())]
            mask[face_ids] = True
            return mask

        self.teeth_face_mask = region_mask("teeth")
        self.oral_cavity_face_mask = region_mask("oral_cavity")
        self.upper_teeth_face_mask = region_mask("teeth_upper")
        self.lower_teeth_face_mask = region_mask("teeth_lower")
        # UV islands for the generated upper/lower teeth intentionally reuse
        # numeric coordinates.  Keep face repair and densification inside the
        # source semantic layer so a point can never jump between coincident
        # face, upper-teeth, lower-teeth, and oral-cavity surfaces.
        self.face_surface_layer = torch.zeros(
            self.ft.shape[0], dtype=torch.int8, device=self.device
        )
        self.face_surface_layer[self.upper_teeth_face_mask] = 1
        self.face_surface_layer[self.lower_teeth_face_mask] = 2
        self.face_surface_layer[self.oral_cavity_face_mask] = 3

    @property
    def get_shape(self):
        return self._shape

    @property
    def get_expression(self):
        return self._expression

    @property
    def get_faces(self):
        return self._faces

    @property
    def get_jaw_pose(self):
        return self._jaw_pose

    @property
    def get_leye_pose(self):
        return self._leye_pose

    @property
    def get_reye_pose(self):
        return self._reye_pose

    @property
    def get_neck_pose(self):
        return self._neck_pose

    @property
    def get_global_orient(self):
        return self._global_orient

    @property
    def get_translation(self):
        return self._translation

    @property
    def get_uv(self):
        return self._uv

    @property
    def get_local_scaling(self) -> torch.Tensor:
        """Learned Gaussian scales in the parent-face coordinate system."""

        return self.scaling_activation(self._scaling)

    @property
    def get_local_rotation(self) -> torch.Tensor:
        """Learned Gaussian rotations in the parent-face coordinate system."""

        return self.rotation_activation(self._rotation)

    @property
    def get_scaling(self) -> torch.Tensor:
        """Current posed world-space scales (before any scene alignment)."""

        vertices, _ = self._flame_verts_and_normals()
        scaling, _ = self._deformed_scaling_rotation(vertices)
        return scaling

    @property
    def get_rotation(self) -> torch.Tensor:
        """Current posed world-space WXYZ quaternions."""

        vertices, _ = self._flame_verts_and_normals()
        _, rotation = self._deformed_scaling_rotation(vertices)
        return rotation

    @property
    def get_xyz(self):
        verts, normals = self._flame_verts_and_normals()
        return self._map_uvd_to_xyz(
            torch.cat([self._uv, self._d], dim=1), verts, normals
        )

    def _normalize_flame_vertices(self, verts: torch.Tensor) -> torch.Tensor:
        vertices = (verts - self.center) * self.scale
        vertices = vertices.clone()
        vertices[:, [1, 2]] = vertices[:, [2, 1]]
        return vertices * (1.1 ** (-self.flame_scale))

    def _flame_forward_vertices(
        self,
        shape: torch.Tensor,
        expression: torch.Tensor,
        global_orient: torch.Tensor,
        neck_pose: torch.Tensor,
        jaw_pose: torch.Tensor,
        leye_pose: torch.Tensor,
        reye_pose: torch.Tensor,
        translation: torch.Tensor,
    ) -> torch.Tensor:
        vertices = self.model(
            shape=shape,
            expr=expression,
            rotation=global_orient,
            neck=neck_pose,
            jaw=jaw_pose,
            eyes=torch.cat([leye_pose, reye_pose], dim=-1),
            translation=translation,
            zero_centered_at_root_node=False,
            return_landmarks=False,
        )
        return vertices.squeeze(0)

    def _flame_verts_and_normals(self) -> tuple[torch.Tensor, torch.Tensor]:
        vertices = self._flame_forward_vertices(
            self._shape,
            self._expression,
            self._global_orient,
            self._neck_pose,
            self._jaw_pose,
            self._leye_pose,
            self._reye_pose,
            self._translation,
        )
        vertices = self._normalize_flame_vertices(vertices)
        normals = self.compute_vertex_normals(vertices, self._faces[:, [0, 2, 1]])
        return vertices, normals

    @staticmethod
    def compute_vertex_normals(
        vertices: torch.Tensor, faces: torch.Tensor
    ) -> torch.Tensor:
        v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        normals = torch.zeros_like(vertices)
        normals.index_add_(0, faces[:, 0], face_normals)
        normals.index_add_(0, faces[:, 1], face_normals)
        normals.index_add_(0, faces[:, 2], face_normals)
        return F.normalize(normals, dim=-1)

    @staticmethod
    def _normalize_safe(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return values / values.norm(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _barycentric_3d(
        point: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ab, ac, ap = b - a, c - a, point - a
        d00, d01, d11 = (ab * ab).sum(-1), (ab * ac).sum(-1), (ac * ac).sum(-1)
        d20, d21 = (ap * ab).sum(-1), (ap * ac).sum(-1)
        denominator = (d00 * d11 - d01.square()).clamp_min(1e-12)
        w1 = (d11 * d20 - d01 * d21) / denominator
        w2 = (d00 * d21 - d01 * d20) / denominator
        return 1.0 - w1 - w2, w1, w2

    def _face_barycentric_uv(
        self, uv: torch.Tensor, face_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        face_idx = face_idx.long().clamp(0, self.ft.shape[0] - 1)
        tri_uv = self.vt[self.ft[face_idx]]
        weights12 = torch.bmm(
            self.face_uv_inv[face_idx], (uv - tri_uv[:, 0]).unsqueeze(-1)
        ).squeeze(-1)
        w1, w2 = weights12[:, 0], weights12[:, 1]
        return 1.0 - w1 - w2, w1, w2, tri_uv

    def _map_uvd_to_xyz(
        self,
        uvd: torch.Tensor,
        vertices: torch.Tensor,
        normals: torch.Tensor,
        face_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        face_idx = self._face_idx if face_idx is None else face_idx
        face_idx = face_idx.long().clamp(0, self._faces.shape[0] - 1)
        w0, w1, w2, _ = self._face_barycentric_uv(uvd[:, :2], face_idx)
        triangle = self._faces[face_idx]
        p0, p1, p2 = vertices[triangle[:, 0]], vertices[triangle[:, 1]], vertices[triangle[:, 2]]
        n0, n1, n2 = normals[triangle[:, 0]], normals[triangle[:, 1]], normals[triangle[:, 2]]
        surface = w0[:, None] * p0 + w1[:, None] * p1 + w2[:, None] * p2
        surface_normal = self._normalize_safe(w0[:, None] * n0 + w1[:, None] * n1 + w2[:, None] * n2)
        return surface + uvd[:, 2:3] * surface_normal

    def _offset_uvd_by_world_delta(
        self,
        uv: torch.Tensor,
        d: torch.Tensor,
        face_idx: torch.Tensor,
        world_delta: torch.Tensor,
        vertices: torch.Tensor,
        normals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a physical split offset while retaining the UVD centre form."""

        face_idx = face_idx.long().clamp(0, self._faces.shape[0] - 1)
        w0, w1, w2, _ = self._face_barycentric_uv(uv, face_idx)
        triangle = self._faces[face_idx]
        p0 = vertices[triangle[:, 0]]
        p1 = vertices[triangle[:, 1]]
        p2 = vertices[triangle[:, 2]]
        n0 = normals[triangle[:, 0]]
        n1 = normals[triangle[:, 1]]
        n2 = normals[triangle[:, 2]]
        surface = w0[:, None] * p0 + w1[:, None] * p1 + w2[:, None] * p2
        surface_normal = self._normalize_safe(
            w0[:, None] * n0 + w1[:, None] * n1 + w2[:, None] * n2
        )
        normal_delta = (world_delta * surface_normal).sum(
            dim=-1, keepdim=True
        )
        target_surface = surface + world_delta - normal_delta * surface_normal
        new_w0, new_w1, new_w2 = self._barycentric_3d(
            target_surface, p0, p1, p2
        )
        uv_triangle = self.vt[self.ft[face_idx]]
        new_uv = (
            new_w0[:, None] * uv_triangle[:, 0]
            + new_w1[:, None] * uv_triangle[:, 1]
            + new_w2[:, None] * uv_triangle[:, 2]
        )
        return new_uv, d + normal_delta

    @classmethod
    def _face_orientation_and_scale(
        cls, vertices: torch.Tensor, faces: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return GaussianAvatars-style orthonormal frames and face scales.

        Frame columns are the first edge, face normal, and the remaining
        in-plane axis.  The scalar is the average of the first-edge length and
        the triangle altitude, matching the official GaussianAvatars code.
        Degenerate faces receive an identity frame and a small positive scale.
        """

        triangle = vertices[faces.long()]
        edge0 = triangle[:, 1] - triangle[:, 0]
        edge1 = triangle[:, 2] - triangle[:, 0]
        edge0_length = edge0.norm(dim=-1, keepdim=True)
        axis0 = cls._normalize_safe(edge0)
        normal_raw = torch.cross(axis0, edge1, dim=-1)
        normal_length = normal_raw.norm(dim=-1, keepdim=True)
        axis1 = cls._normalize_safe(normal_raw)
        axis2 = -cls._normalize_safe(torch.cross(axis1, axis0, dim=-1))
        orientation = torch.stack((axis0, axis1, axis2), dim=-1)
        altitude = (axis2 * edge1).sum(dim=-1, keepdim=True).abs()
        face_scale = 0.5 * (edge0_length + altitude)

        finite = (
            torch.isfinite(orientation).flatten(1).all(dim=1)
            & torch.isfinite(face_scale[:, 0])
        )
        valid = finite & (edge0_length[:, 0] > cls.EPS) & (
            normal_length[:, 0] > cls.EPS
        )
        identity = torch.eye(
            3, dtype=vertices.dtype, device=vertices.device
        ).expand(orientation.shape[0], -1, -1)
        orientation = torch.where(valid[:, None, None], orientation, identity)
        face_scale = torch.nan_to_num(
            face_scale, nan=cls.EPS, posinf=cls.EPS, neginf=cls.EPS
        ).clamp_min(cls.EPS)
        return orientation, face_scale

    def _face_properties(
        self,
        vertices: torch.Tensor,
        face_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        face_idx = self._face_idx if face_idx is None else face_idx
        face_idx = face_idx.long().clamp(0, self._faces.shape[0] - 1)
        return self._face_orientation_and_scale(
            vertices, self._faces[face_idx]
        )

    def current_covariance_transform(self) -> torch.Tensor:
        """Map a face-local covariance factor into the current FLAME pose."""

        vertices, _ = self._flame_verts_and_normals()
        orientation, face_scale = self._face_properties(vertices)
        return orientation * face_scale[:, None, :]

    def _deformed_scaling_rotation(
        self,
        vertices: torch.Tensor,
        face_idx: Optional[torch.Tensor] = None,
        local_scaling: Optional[torch.Tensor] = None,
        local_rotation: Optional[torch.Tensor] = None,
        scaling_modifier: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose learned face-local ellipsoids into the posed world frame."""

        local_scaling = (
            self.get_local_scaling
            if local_scaling is None
            else local_scaling
        )
        local_rotation = (
            self.get_local_rotation
            if local_rotation is None
            else self.rotation_activation(local_rotation)
        )
        face_orientation, face_scale = self._face_properties(
            vertices, face_idx
        )
        world_scaling = (
            float(scaling_modifier) * face_scale * local_scaling
        ).clamp_min(self.SIGMA_FLOOR)
        world_rotation_matrix = torch.bmm(
            face_orientation, build_rotation(local_rotation)
        )
        world_rotation = F.normalize(
            matrix_to_quaternion(world_rotation_matrix), dim=-1
        )
        return world_scaling, world_rotation

    @staticmethod
    def _covariance_matrix_from_scaling_rotation(
        scaling: torch.Tensor, rotation: torch.Tensor
    ) -> torch.Tensor:
        rotation_matrix = build_rotation(F.normalize(rotation, dim=-1))
        factor = rotation_matrix * scaling[:, None, :]
        return torch.bmm(factor, factor.transpose(1, 2))

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        vertices, _ = self._flame_verts_and_normals()
        scaling, rotation = self._deformed_scaling_rotation(
            vertices, scaling_modifier=scaling_modifier
        )
        return strip_symmetric(
            self._covariance_matrix_from_scaling_rotation(scaling, rotation)
        )

    def get_deformed_gaussians(
        self, scaling_modifier: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vertices, normals = self._flame_verts_and_normals()
        xyz = self._map_uvd_to_xyz(
            torch.cat([self._uv, self._d], dim=1), vertices, normals
        )
        # ``GaussianRasterizationSettings.scale_modifier`` is applied by the
        # CUDA rasterizer to explicit scales.  Keep these unmodified to avoid
        # applying a non-unit preview modifier twice.
        scaling, rotation = self._deformed_scaling_rotation(vertices)
        return xyz, scaling, rotation

    def get_world_scale(self) -> torch.Tensor:
        vertices, _ = self._flame_verts_and_normals()
        scaling, _ = self._deformed_scaling_rotation(vertices)
        return torch.sort(scaling, dim=-1, descending=True).values

    def get_world_scale_max_approx(self) -> torch.Tensor:
        vertices, _ = self._flame_verts_and_normals()
        scaling, _ = self._deformed_scaling_rotation(vertices)
        return scaling.amax(dim=-1)

    def _build_uv_grid(self, resolution: int = 256) -> None:
        vt = self.vt.detach().cpu().numpy()
        face_uvs = vt[self.ft.detach().cpu().numpy()]

        def cell_index(values: np.ndarray) -> np.ndarray:
            return np.clip((values * resolution).astype(np.int32), 0, resolution - 1)

        min_u, max_u = cell_index(face_uvs[:, :, 0].min(1)), cell_index(face_uvs[:, :, 0].max(1))
        min_v, max_v = cell_index(face_uvs[:, :, 1].min(1)), cell_index(face_uvs[:, :, 1].max(1))
        cells = [[] for _ in range(resolution * resolution)]
        for face_index in range(self.ft.shape[0]):
            for v in range(min_v[face_index], max_v[face_index] + 1):
                base = v * resolution
                for u in range(min_u[face_index], max_u[face_index] + 1):
                    cells[base + u].append(face_index)

        max_candidates = max((len(item) for item in cells), default=1)
        grid = -np.ones((resolution * resolution, max(max_candidates, 1)), dtype=np.int32)
        for cell, candidates in enumerate(cells):
            if candidates:
                grid[cell, : len(candidates)] = candidates
        self.uv_grid_resolution = resolution
        self.uv_grid = torch.from_numpy(grid).to(self.device)

    def _uv_inside_faces(
        self, uv: torch.Tensor, face_idx: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        face_idx = face_idx.long().clamp(0, self.ft.shape[0] - 1)
        w0, w1, w2, _ = self._face_barycentric_uv(uv, face_idx)
        finite = (
            torch.isfinite(w0)
            & torch.isfinite(w1)
            & torch.isfinite(w2)
            & self.face_uv_valid[face_idx]
        )
        return finite & (w0 >= -eps) & (w1 >= -eps) & (w2 >= -eps), finite

    @staticmethod
    def _project_uv_to_triangles(
        uv: torch.Tensor, triangles: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        point = uv[:, None, :]
        projections, distances = [], []
        for start, end in ((0, 1), (1, 2), (2, 0)):
            a = triangles[:, :, start]
            edge = triangles[:, :, end] - a
            t = ((point - a) * edge).sum(-1) / edge.square().sum(-1).clamp_min(1e-20)
            projection = a + t.clamp(0.0, 1.0)[..., None] * edge
            projections.append(projection)
            distances.append((projection - point).square().sum(dim=-1))
        projections = torch.stack(projections, dim=2)
        distances = torch.stack(distances, dim=2)
        edge_index = distances.argmin(dim=2)
        gather_index = edge_index[..., None, None].expand(-1, -1, 1, 2)
        return (
            projections.gather(2, gather_index).squeeze(2),
            distances.gather(2, edge_index[..., None]).squeeze(2),
        )

    def _find_uv_faces_from_grid(
        self, uv: torch.Tensor, fallback_face_idx: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        resolution = self.uv_grid_resolution
        ij = (uv * resolution).long().clamp(0, resolution - 1)
        candidates = self.uv_grid[ij[:, 1] * resolution + ij[:, 0]].long()
        safe_candidates = candidates.clamp_min(0)
        valid = (candidates >= 0) & self.face_uv_valid[safe_candidates]
        source_layer = self.face_surface_layer[
            fallback_face_idx.long().clamp(0, self.ft.shape[0] - 1)
        ]
        valid &= self.face_surface_layer[safe_candidates] == source_layer[:, None]
        triangle_uv = self.vt[self.ft[safe_candidates]]
        delta = uv[:, None, :] - triangle_uv[:, :, 0]
        weights12 = torch.matmul(
            self.face_uv_inv[safe_candidates], delta.unsqueeze(-1)
        ).squeeze(-1)
        w1, w2 = weights12[..., 0], weights12[..., 1]
        w0 = 1.0 - w1 - w2
        inside = valid & torch.isfinite(w0) & torch.isfinite(w1) & torch.isfinite(w2)
        inside &= (w0 >= -eps) & (w1 >= -eps) & (w2 >= -eps)
        scores = torch.where(
            inside,
            torch.minimum(torch.minimum(w0, w1), w2),
            torch.full_like(w0, -1e9),
        )
        rows = torch.arange(uv.shape[0], device=self.device)
        best_inside = scores.argmax(dim=1)
        has_inside = inside.any(dim=1)
        face_idx = candidates[rows, best_inside].clamp_min(0)
        repaired_uv = uv.clone()

        projected, distance = self._project_uv_to_triangles(uv, triangle_uv)
        distance = torch.where(valid, distance, torch.full_like(distance, float("inf")))
        best_projected = distance.argmin(dim=1)
        has_candidate = valid.any(dim=1)
        outside = ~has_inside
        use_projection = outside & has_candidate
        face_idx[use_projection] = candidates[rows, best_projected][use_projection].clamp_min(0)
        repaired_uv[use_projection] = projected[rows, best_projected][use_projection]

        no_candidate = outside & (~has_candidate)
        if no_candidate.any():
            fallback = fallback_face_idx[no_candidate].clamp(0, self.ft.shape[0] - 1)
            fallback_triangles = self.vt[self.ft[fallback]][:, None]
            fallback_uv, _ = self._project_uv_to_triangles(uv[no_candidate], fallback_triangles)
            face_idx[no_candidate] = fallback
            repaired_uv[no_candidate] = fallback_uv[:, 0]
        return face_idx, repaired_uv, outside

    @torch.no_grad()
    def update_face_idx_from_uv(
        self,
        eps: float = 1e-6,
        mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ):
        count = self.num_gs
        selected = (
            torch.ones(count, dtype=torch.bool, device=self.device)
            if mask is None
            else mask.to(self.device, dtype=torch.bool)
        )
        if selected.numel() != count:
            raise ValueError("UVD update mask has the wrong length")
        if not selected.any():
            return {"updated": 0, "projected": 0} if return_stats else None

        indices = torch.nonzero(selected, as_tuple=False).squeeze(1)
        uv = self._uv[indices]
        old_faces = self._face_idx[indices]
        inside, finite = self._uv_inside_faces(uv, old_faces, eps)
        repair = (~inside) | (~finite)
        if not repair.any():
            return {"updated": 0, "projected": 0} if return_stats else None

        repaired_faces, repaired_uv, projected = self._find_uv_faces_from_grid(
            uv[repair], old_faces[repair], eps
        )
        repair_indices = indices[repair]
        changed_face = repaired_faces != old_faces[repair]
        updated = int(changed_face.sum().item())
        changed_indices = repair_indices[changed_face]
        if updated:
            # UV is intentionally allowed to cross triangle boundaries.  The
            # learned ellipsoid is expressed in the parent-face frame, so a
            # binding change must be a pure reparameterization: preserve its
            # current world scale/rotation and express them in the new frame.
            vertices, _ = self._flame_verts_and_normals()
            old_orientation, old_face_scale = self._face_properties(
                vertices, old_faces[repair][changed_face]
            )
            new_orientation, new_face_scale = self._face_properties(
                vertices, repaired_faces[changed_face]
            )
            local_scaling = self.get_local_scaling[changed_indices]
            preserved_world_scaling = old_face_scale * local_scaling
            self._scaling.data[changed_indices] = (
                preserved_world_scaling / new_face_scale
            ).clamp_min(torch.finfo(local_scaling.dtype).tiny).log()

            local_rotation = build_rotation(
                self.get_local_rotation[changed_indices]
            )
            preserved_world_rotation = torch.bmm(
                old_orientation, local_rotation
            )
            rebound_local_rotation = torch.bmm(
                new_orientation.transpose(1, 2), preserved_world_rotation
            )
            self._rotation.data[changed_indices] = F.normalize(
                matrix_to_quaternion(rebound_local_rotation), dim=-1
            )
        self._face_idx[repair_indices] = repaired_faces
        self._uv.data[repair_indices] = repaired_uv.clamp(self.UV_EPS, 1.0 - self.UV_EPS)
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                if group["name"] == "uv":
                    reset_indices = repair_indices
                elif group["name"] in {"scaling", "rotation"}:
                    reset_indices = changed_indices
                else:
                    continue
                state = self.optimizer.state.get(group["params"][0], {})
                for name in ("exp_avg", "exp_avg_sq"):
                    if name in state and reset_indices.numel():
                        state[name][reset_indices] = 0
        if return_stats:
            return {"updated": updated, "projected": int(projected.sum().item())}
        return None

    @staticmethod
    def _covariance_to_scaling_rotation(
        covariance: torch.Tensor, min_scale: float = 1e-8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        covariance = 0.5 * (covariance + covariance.transpose(1, 2))
        values, vectors = torch.linalg.eigh(covariance)
        values = values.clamp_min(min_scale ** 2)
        improper = torch.det(vectors) < 0
        if improper.any():
            vectors = vectors.clone()
            vectors[improper, :, 2] *= -1.0
        return values.sqrt(), F.normalize(matrix_to_quaternion(vectors), dim=-1)

    def _initialize_flame_template(self, flame_scale: float):
        shape = torch.zeros((1, self.num_betas), dtype=torch.float32, device=self.device)
        expression = torch.zeros((1, self.num_expression), dtype=torch.float32, device=self.device)
        zeros3 = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        vertices = self._flame_forward_vertices(
            shape, expression, zeros3, zeros3, zeros3, zeros3, zeros3, zeros3
        )
        self.center = ((vertices.amin(0) + vertices.amax(0)) / 2).detach()
        self.scale = (0.6 / (vertices.amax(0) - vertices.amin(0)).amax()).detach()
        self.flame_scale = float(flame_scale)
        self._shape = nn.Parameter(shape.requires_grad_(True))
        self._expression = expression.detach()
        self._global_orient = zeros3.detach()
        self._neck_pose = zeros3.detach()
        self._jaw_pose = zeros3.detach()
        self._leye_pose = zeros3.detach()
        self._reye_pose = zeros3.detach()
        self._translation = zeros3.detach()
        return self._normalize_flame_vertices(vertices), self._faces

    def initialize_flame_state(self, spatial_lr_scale: float, flame_scale: float):
        self.spatial_lr_scale = float(spatial_lr_scale)
        return self._initialize_flame_template(flame_scale)

    def create_from_flame(
        self,
        spatial_lr_scale: float,
        flame_scale: float,
        num_points: int,
        include_teeth: bool,
        teeth_points: int = 0,
        oral_cavity_points: int = 0,
        teeth_rgb=None,
        oral_cavity_rgb=None,
    ) -> None:
        vertices, faces = self.initialize_flame_state(spatial_lr_scale, flame_scale)
        num_points = int(num_points)
        teeth_points = int(teeth_points)
        oral_cavity_points = int(oral_cavity_points)
        if num_points <= 0:
            raise ValueError("num_points must be positive")
        if teeth_points < 0 or oral_cavity_points < 0:
            raise ValueError("Dental Gaussian counts must be non-negative")
        if not include_teeth and (teeth_points or oral_cavity_points):
            raise ValueError(
                "Dental Gaussian counts require include_teeth=True"
            )

        face_ids = torch.arange(faces.shape[0], device=self.device)
        face_ids = face_ids[~self.oral_cavity_face_mask]
        # When a dedicated dental count is configured, keep those points out
        # of the generic grey surface sample and initialize them separately
        # with the AnimPortrait3D priors below.
        if not include_teeth or teeth_points > 0:
            face_ids = face_ids[~self.teeth_face_mask[face_ids]]
        if face_ids.numel() == 0:
            raise RuntimeError("No FLAME faces available for Gaussian initialization")

        mesh = trimesh.Trimesh(
            vertices.detach().cpu().numpy(),
            faces[face_ids].detach().cpu().numpy(),
            process=False,
        )
        samples, sampled_faces = trimesh.sample.sample_surface(mesh, num_points)
        face_idx = face_ids[
            torch.from_numpy(sampled_faces.astype(np.int64)).to(self.device)
        ]
        samples = torch.from_numpy(np.asarray(samples)).float().to(self.device)
        triangle = faces[face_idx]
        w0, w1, w2 = self._barycentric_3d(
            samples,
            vertices[triangle[:, 0]],
            vertices[triangle[:, 1]],
            vertices[triangle[:, 2]],
        )
        uv_triangle = self.ft[face_idx]
        uv = (
            w0[:, None] * self.vt[uv_triangle[:, 0]]
            + w1[:, None] * self.vt[uv_triangle[:, 1]]
            + w2[:, None] * self.vt[uv_triangle[:, 2]]
        ).clamp(self.UV_EPS, 1.0 - self.UV_EPS)

        self.num_gs = num_points
        self._face_idx = face_idx
        self._uv = nn.Parameter(uv.contiguous().requires_grad_(True))
        self._d = nn.Parameter(torch.zeros((num_points, 1), device=self.device).requires_grad_(True))

        features = torch.zeros(
            (num_points, 3, (self.max_sh_degree + 1) ** 2),
            dtype=torch.float32,
            device=self.device,
        )
        features[:, :3, 0] = RGB2SH(
            torch.full((num_points, 3), 0.5, dtype=torch.float32, device=self.device)
        )
        distance2 = torch.clamp_min(distCUDA2(self.get_xyz), 1e-7)
        _, face_scale = self._face_properties(vertices, face_idx)
        scales = distance2.sqrt()[:, None].repeat(1, 3) / face_scale
        rotations = torch.zeros(
            (num_points, 4), dtype=torch.float32, device=self.device
        )
        rotations[:, 0] = 1.0
        self._features_dc = nn.Parameter(
            features[:, :, :1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            inverse_sigmoid(0.1 * torch.ones((num_points, 1), device=self.device)).requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.log().requires_grad_(True))
        self._rotation = nn.Parameter(rotations.requires_grad_(True))
        self._reset_densification_buffers()
        if include_teeth and teeth_points > 0:
            self.seed_flame_region(
                "teeth",
                teeth_points,
                rgb=(
                    self.ANIM_PORTRAIT3D_TEETH_RGB
                    if teeth_rgb is None
                    else teeth_rgb
                ),
                opacity=0.1,
                exclude_regions=("oral_cavity",),
            )
        if include_teeth and oral_cavity_points > 0:
            self.seed_flame_region(
                "oral_cavity",
                oral_cavity_points,
                rgb=(
                    self.ANIM_PORTRAIT3D_ORAL_RGB
                    if oral_cavity_rgb is None
                    else oral_cavity_rgb
                ),
                opacity=0.1,
            )
        print(
            "Number of points at initialization: "
            f"{self.num_gs} (surface={num_points}, teeth={teeth_points}, "
            f"oral_cavity={oral_cavity_points})"
        )

    def point_region_mask(self, regions) -> torch.Tensor:
        """Return the Gaussian mask induced by one or more FLAME face regions.

        The mask is evaluated through ``face_idx`` rather than UV ranges.  This
        keeps regional regularizers valid across UV seams and makes the mouth,
        teeth, and oral-cavity ablations independent of atlas layout details.
        """

        if isinstance(regions, str):
            regions = [regions]
        region_faces = self.model.mask.get_fid_by_region(list(regions)).to(
            device=self.device, dtype=torch.long
        )
        if region_faces.numel() == 0 or self._face_idx.numel() == 0:
            return torch.zeros(
                self.num_gs, dtype=torch.bool, device=self.device
            )
        face_lookup = torch.zeros(
            self._faces.shape[0], dtype=torch.bool, device=self.device
        )
        region_faces = region_faces[
            (region_faces >= 0) & (region_faces < face_lookup.numel())
        ]
        face_lookup[region_faces] = True
        return face_lookup[self._face_idx]

    @torch.no_grad()
    def seed_flame_region(
        self,
        region: str,
        num_points: int,
        rgb=(0.025, 0.012, 0.012),
        opacity: float = 0.35,
        min_world_scale: float = 5.0e-4,
        max_world_scale: float = 1.0e-2,
        exclude_regions=(),
    ) -> Dict[str, int]:
        """Append surface-bound Gaussians on a named FLAME region.

        This helper adds *real* UVD points on articulated FLAME faces rather
        than synthesizing a detached shell.  ``exclude_regions`` resolves the
        intentional overlap between the ``teeth`` and ``oral_cavity`` vertex
        masks, so each surface receives its own colour prior.

        The new ellipsoid starts approximately isotropic in posed world space.
        Its learned scale is divided by the parent-face scale and its local
        rotation starts at identity, so later FLAME poses can compose it using
        only the explicit face frame.
        """

        count = int(num_points)
        start = int(self.num_gs)
        if count <= 0:
            return {"start": start, "end": start, "added": 0}
        if self._shape.numel() == 0:
            raise RuntimeError(
                "FLAME state must be initialized before seeding a face region"
            )

        face_ids = self.model.mask.get_fid_by_region(region).to(
            device=self.device, dtype=torch.long
        )
        face_ids = face_ids[
            (face_ids >= 0) & (face_ids < self._faces.shape[0])
        ]
        if exclude_regions:
            excluded_ids = self.model.mask.get_fid_by_region(
                list(exclude_regions)
            ).to(device=self.device, dtype=torch.long)
            excluded_ids = excluded_ids[
                (excluded_ids >= 0) & (excluded_ids < self._faces.shape[0])
            ]
            excluded_lookup = torch.zeros(
                self._faces.shape[0], dtype=torch.bool, device=self.device
            )
            excluded_lookup[excluded_ids] = True
            face_ids = face_ids[~excluded_lookup[face_ids]]
        if face_ids.numel() == 0:
            raise ValueError(f"FLAME region {region!r} contains no faces")

        vertices, normals = self._flame_verts_and_normals()
        triangles = vertices[self._faces[face_ids]]
        areas = 0.5 * torch.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=-1,
        ).norm(dim=-1)
        probabilities = areas.clamp_min(1.0e-12)
        probabilities = probabilities / probabilities.sum()
        sampled_local = torch.multinomial(
            probabilities, count, replacement=True
        )
        sampled_faces = face_ids[sampled_local]

        # Square-root sampling gives uniform barycentric surface samples.
        random_values = torch.rand(
            (count, 2), dtype=vertices.dtype, device=self.device
        )
        root = random_values[:, 0].sqrt()
        weights = torch.stack(
            (
                1.0 - root,
                root * (1.0 - random_values[:, 1]),
                root * random_values[:, 1],
            ),
            dim=-1,
        )
        texture_triangles = self.vt[self.ft[sampled_faces]]
        uv = (weights[..., None] * texture_triangles).sum(dim=1)
        uv = uv.clamp(self.UV_EPS, 1.0 - self.UV_EPS)
        d = torch.zeros((count, 1), dtype=uv.dtype, device=self.device)

        xyz = self._map_uvd_to_xyz(
            torch.cat((uv, d), dim=1),
            vertices,
            normals,
            face_idx=sampled_faces,
        )
        if count > 1:
            distance2 = distCUDA2(xyz).clamp_min(min_world_scale ** 2)
        else:
            distance2 = torch.full(
                (1,),
                min_world_scale ** 2,
                dtype=xyz.dtype,
                device=self.device,
            )
        distance2 = distance2.clamp_max(max_world_scale ** 2)
        _, face_scale = self._face_properties(vertices, sampled_faces)
        scales = distance2.sqrt()[:, None].repeat(1, 3) / face_scale
        rotations = torch.zeros(
            (count, 4), dtype=torch.float32, device=self.device
        )
        rotations[:, 0] = 1.0

        color = torch.as_tensor(
            rgb, dtype=torch.float32, device=self.device
        ).reshape(1, 3)
        if color.shape != (1, 3):
            raise ValueError("rgb must contain exactly three values")
        features = torch.zeros(
            (count, 3, (self.max_sh_degree + 1) ** 2),
            dtype=torch.float32,
            device=self.device,
        )
        features[:, :, 0] = RGB2SH(color.expand(count, -1))
        features_dc = features[:, :, :1].transpose(1, 2).contiguous()
        features_rest = features[:, :, 1:].transpose(1, 2).contiguous()
        opacity_value = float(min(max(opacity, self.EPS), 1.0 - self.EPS))
        opacity_logits = inverse_sigmoid(
            torch.full(
                (count, 1),
                opacity_value,
                dtype=torch.float32,
                device=self.device,
            )
        )

        if self.optimizer is None:
            self._uv = nn.Parameter(
                torch.cat((self._uv.detach(), uv), dim=0).requires_grad_(True)
            )
            self._d = nn.Parameter(
                torch.cat((self._d.detach(), d), dim=0).requires_grad_(True)
            )
            self._features_dc = nn.Parameter(
                torch.cat(
                    (self._features_dc.detach(), features_dc), dim=0
                ).requires_grad_(True)
            )
            self._features_rest = nn.Parameter(
                torch.cat(
                    (self._features_rest.detach(), features_rest), dim=0
                ).requires_grad_(True)
            )
            self._opacity = nn.Parameter(
                torch.cat(
                    (self._opacity.detach(), opacity_logits), dim=0
                ).requires_grad_(True)
            )
            self._scaling = nn.Parameter(
                torch.cat(
                    (self._scaling.detach(), scales.log()), dim=0
                ).requires_grad_(True)
            )
            self._rotation = nn.Parameter(
                torch.cat(
                    (self._rotation.detach(), rotations), dim=0
                ).requires_grad_(True)
            )
            self._face_idx = torch.cat(
                (self._face_idx, sampled_faces), dim=0
            )
            self.num_gs = int(self._uv.shape[0])
            self._reset_densification_buffers()
        else:
            self.densification_postfix(
                uv,
                d,
                features_dc,
                features_rest,
                opacity_logits,
                scales.log(),
                rotations,
                sampled_faces,
            )

        return {
            "start": start,
            "end": int(self.num_gs),
            "added": int(self.num_gs) - start,
        }

    def _vertex_dtype(self):
        dtype = [
            ("u", "f4"), ("v", "f4"), ("d", "f4"),
            ("face_idx", "i4"), ("opacity", "f4"),
        ]
        dtype += [(f"f_dc_{index}", "f4") for index in range(3)]
        dtype += [
            (f"f_rest_{index}", "f4")
            for index in range(3 * (self.max_sh_degree + 1) ** 2 - 3)
        ]
        dtype += [(f"scale_{index}", "f4") for index in range(3)]
        dtype += [(f"rot_{index}", "f4") for index in range(4)]
        return dtype

    @staticmethod
    def _flatten_features(features: torch.Tensor) -> np.ndarray:
        return (
            features.detach().transpose(1, 2).contiguous().cpu().numpy().astype(np.float32)
            .reshape(features.shape[0], -1)
        )

    def save_ply(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            mkdir_p(directory)
        vertex = np.empty(self.num_gs, dtype=self._vertex_dtype())
        uv = self._uv.detach().cpu().numpy().astype(np.float32)
        vertex["u"], vertex["v"] = uv[:, 0], uv[:, 1]
        vertex["d"] = self._d.detach().cpu().numpy().astype(np.float32)[:, 0]
        vertex["face_idx"] = self._face_idx.detach().cpu().numpy().astype(np.int32)
        vertex["opacity"] = self._opacity.detach().cpu().numpy().astype(np.float32)[:, 0]
        for index, values in enumerate(self._flatten_features(self._features_dc).T):
            vertex[f"f_dc_{index}"] = values
        for index, values in enumerate(self._flatten_features(self._features_rest).T):
            vertex[f"f_rest_{index}"] = values
        for index in range(3):
            vertex[f"scale_{index}"] = self._scaling[:, index].detach().cpu().numpy().astype(np.float32)
        for index in range(4):
            vertex[f"rot_{index}"] = self._rotation[:, index].detach().cpu().numpy().astype(np.float32)

        shape = self._shape.detach().cpu().numpy().astype(np.float32)
        shape_dtype = [(f"shape_{index}", "f4") for index in range(shape.shape[1])]
        shape_vertex = np.empty(shape.shape[0], dtype=shape_dtype)
        for index in range(shape.shape[1]):
            shape_vertex[f"shape_{index}"] = shape[:, index]
        PlyData(
            [PlyElement.describe(vertex, "vertex"), PlyElement.describe(shape_vertex, "shape")],
            comments=[
                "coordinate_space=flame_uvd",
                self.SCALE_ROTATION_COMMENT,
            ],
        ).write(path)

    @staticmethod
    def _property_matrix(vertex, prefix: str, expected: int) -> np.ndarray:
        if expected == 0:
            return np.empty((len(vertex.data), 0), dtype=np.float32)
        names = sorted(
            [item.name for item in vertex.properties if item.name.startswith(prefix)],
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        if len(names) != expected:
            raise ValueError(f"Expected {expected} {prefix} fields, found {len(names)}")
        return np.stack([np.asarray(vertex[name], dtype=np.float32) for name in names], axis=1)

    def load_ply(
        self, path: str, *, allow_legacy_scale_rotation: bool = False
    ) -> None:
        ply = PlyData.read(path)
        comments = {str(comment).strip() for comment in ply.comments}
        if (
            self.SCALE_ROTATION_COMMENT not in comments
            and not allow_legacy_scale_rotation
        ):
            raise ValueError(
                "The PLY stores legacy UVD-space scale/rotation. This model "
                "requires face-local scale/rotation; rerun reconstruction or "
                "convert the legacy model with tools/convert_legacy_uvd_ply.py."
            )
        vertex = ply.elements[0]
        names = set(vertex.data.dtype.names)
        required = {"u", "v", "d", "face_idx", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"}
        missing = sorted(required - names)
        if missing:
            raise ValueError(
                "Only direct UVD PLY files are supported; missing " + ", ".join(missing)
            )
        uv = np.stack([vertex["u"], vertex["v"]], axis=1).astype(np.float32)
        features_dc = np.stack(
            [vertex[f"f_dc_{index}"] for index in range(3)], axis=1
        ).astype(np.float32)[:, :, None]
        rest_count = 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_rest = self._property_matrix(vertex, "f_rest_", rest_count)
        features_rest = features_rest.reshape(
            uv.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1
        )
        self._uv = nn.Parameter(torch.as_tensor(uv, dtype=torch.float32, device=self.device).requires_grad_(True))
        self._d = nn.Parameter(
            torch.as_tensor(np.asarray(vertex["d"], dtype=np.float32)[:, None], device=self.device).requires_grad_(True)
        )
        face_idx = np.asarray(vertex["face_idx"], dtype=np.int64)
        if face_idx.size and (
            face_idx.min() < 0 or face_idx.max() >= self._faces.shape[0]
        ):
            raise ValueError("UVD PLY contains an out-of-range face_idx")
        self._face_idx = torch.as_tensor(
            face_idx, dtype=torch.long, device=self.device
        )
        self._features_dc = nn.Parameter(
            torch.as_tensor(features_dc, dtype=torch.float32, device=self.device).transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.as_tensor(features_rest, dtype=torch.float32, device=self.device).transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.as_tensor(np.asarray(vertex["opacity"], dtype=np.float32)[:, None], device=self.device).requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.as_tensor(self._property_matrix(vertex, "scale_", 3), dtype=torch.float32, device=self.device).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.as_tensor(self._property_matrix(vertex, "rot_", 4), dtype=torch.float32, device=self.device).requires_grad_(True)
        )
        shape = self._property_matrix(ply["shape"], "shape_", self.num_betas)
        self._shape = nn.Parameter(
            torch.as_tensor(shape, dtype=torch.float32, device=self.device).requires_grad_(True)
        )
        self.num_gs = self._uv.shape[0]
        self.active_sh_degree = self.max_sh_degree
        self._reset_densification_buffers()

    def training_setup(self, training_args) -> None:
        self.percent_dense = float(training_args.percent_dense)
        self._reset_densification_buffers()
        position_lr = training_args.position_lr_init * self.spatial_lr_scale
        groups = [
            {"params": [self._uv], "lr": position_lr, "name": "uv"},
            {"params": [self._d], "lr": position_lr, "name": "d"},
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
        ]
        if training_args.shape_lr > 0:
            groups.append(
                {
                    "params": [self._shape],
                    "lr": training_args.shape_lr,
                    "name": "flame_shape",
                }
            )
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=position_lr,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    @torch.no_grad()
    def mask_out_gradient(
        self, trainable_mask: torch.Tensor, multiplier: float = 0.0
    ) -> None:
        """Scale gradients outside ``trainable_mask`` for every point property.

        This mirrors AnimPortrait3D's static reconstruction guard, but derives
        membership from this model's own UVD face binding.  Geometry, colour,
        opacity, scale, and rotation are therefore frozen together for hidden
        dental/oral points.
        """

        trainable_mask = trainable_mask.to(
            device=self.device, dtype=torch.bool
        ).detach()
        if trainable_mask.numel() != self.num_gs:
            raise ValueError("Gradient mask has the wrong number of Gaussians")
        frozen_mask = ~trainable_mask
        if not frozen_mask.any():
            return
        for group in self.optimizer.param_groups:
            if group["name"] == "flame_shape":
                continue
            parameter = group["params"][0]
            if parameter.grad is not None:
                row_multiplier = torch.where(
                    trainable_mask,
                    torch.ones((), dtype=parameter.grad.dtype, device=self.device),
                    torch.as_tensor(
                        multiplier, dtype=parameter.grad.dtype, device=self.device
                    ),
                ).reshape(
                    (self.num_gs,) + (1,) * (parameter.grad.ndim - 1)
                )
                parameter.grad.mul_(row_multiplier)

    def update_learning_rate(self, iteration: int) -> float:
        lr = self.xyz_scheduler_args(iteration)
        for group in self.optimizer.param_groups:
            if group["name"] in {"uv", "d"}:
                group["lr"] = lr
        return lr

    def _reset_densification_buffers(self) -> None:
        self.xyz_gradient_accum = torch.zeros((self.num_gs, 1), device=self.device)
        self.denom = torch.zeros((self.num_gs, 1), device=self.device)
        self.max_radii2D = torch.zeros((self.num_gs,), device=self.device)

    def _replace_optimizer_parameter(
        self, group: Dict, value: torch.Tensor, transform_state=None
    ) -> nn.Parameter:
        old_parameter = group["params"][0]
        requires_grad = bool(old_parameter.requires_grad)
        state = self.optimizer.state.get(old_parameter)
        if state is not None and transform_state is not None:
            state = transform_state(state)
            del self.optimizer.state[old_parameter]
        parameter = nn.Parameter(value, requires_grad=requires_grad)
        group["params"][0] = parameter
        if state is not None:
            self.optimizer.state[parameter] = state
        return parameter

    def _append_to_optimizer(self, extensions: Dict[str, torch.Tensor]) -> Dict[str, nn.Parameter]:
        output = {}
        for group in self.optimizer.param_groups:
            name = group["name"]
            if name == "flame_shape":
                continue
            extension = extensions[name]

            def append_state(state, ext=extension):
                state["exp_avg"] = torch.cat([state["exp_avg"], torch.zeros_like(ext)], dim=0)
                state["exp_avg_sq"] = torch.cat([state["exp_avg_sq"], torch.zeros_like(ext)], dim=0)
                return state

            output[name] = self._replace_optimizer_parameter(
                group, torch.cat([group["params"][0], extension], dim=0), append_state
            )
        return output

    def _prune_optimizer(self, keep: torch.Tensor) -> Dict[str, nn.Parameter]:
        output = {}
        for group in self.optimizer.param_groups:
            name = group["name"]
            if name == "flame_shape":
                continue

            def prune_state(state, mask=keep):
                state["exp_avg"] = state["exp_avg"][mask]
                state["exp_avg_sq"] = state["exp_avg_sq"][mask]
                return state

            output[name] = self._replace_optimizer_parameter(
                group, group["params"][0][keep], prune_state
            )
        return output

    def _assign_point_tensors(self, tensors: Dict[str, nn.Parameter]) -> None:
        self._uv = tensors["uv"]
        self._d = tensors["d"]
        self._features_dc = tensors["f_dc"]
        self._features_rest = tensors["f_rest"]
        self._opacity = tensors["opacity"]
        self._scaling = tensors["scaling"]
        self._rotation = tensors["rotation"]
        self.num_gs = self._uv.shape[0]

    def densification_postfix(
        self,
        uv: torch.Tensor,
        d: torch.Tensor,
        features_dc: torch.Tensor,
        features_rest: torch.Tensor,
        opacity: torch.Tensor,
        scaling: torch.Tensor,
        rotation: torch.Tensor,
        face_idx: torch.Tensor,
    ) -> None:
        previous_count = self.num_gs
        tensors = self._append_to_optimizer(
            {
                "uv": uv,
                "d": d,
                "f_dc": features_dc,
                "f_rest": features_rest,
                "opacity": opacity,
                "scaling": scaling,
                "rotation": rotation,
            }
        )
        self._assign_point_tensors(tensors)
        self._face_idx = torch.cat([self._face_idx, face_idx], dim=0)
        self._reset_densification_buffers()
        new_points = torch.zeros(self.num_gs, dtype=torch.bool, device=self.device)
        new_points[previous_count:] = True
        self.update_face_idx_from_uv(mask=new_points)

    def prune_points(self, mask: torch.Tensor) -> None:
        keep = ~mask
        tensors = self._prune_optimizer(keep)
        self._assign_point_tensors(tensors)
        self._face_idx = self._face_idx[keep]
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep]
        self.denom = self.denom[keep]
        self.max_radii2D = self.max_radii2D[keep]

    def add_densification_stats(
        self, viewspace_points: torch.Tensor, visible: torch.Tensor
    ) -> None:
        gradients = torch.norm(viewspace_points[visible, :2], dim=-1, keepdim=True)
        gradients = torch.nan_to_num(gradients, nan=0.0, posinf=0.0, neginf=0.0)
        self.xyz_gradient_accum[visible] += gradients
        self.denom[visible] += 1

    def _densification_gradients(self) -> torch.Tensor:
        gradients = self.xyz_gradient_accum / self.denom.clamp_min(1.0)
        gradients = torch.nan_to_num(gradients, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.where(self.denom > 0.0, gradients, torch.zeros_like(gradients))

    def _limit_to_budget(
        self, selected: torch.Tensor, score: torch.Tensor, count: int
    ) -> torch.Tensor:
        count = max(int(count), 0)
        indices = torch.nonzero(selected, as_tuple=False).squeeze(1)
        if count == 0:
            return torch.zeros_like(selected)
        if indices.numel() <= count:
            return selected
        chosen = indices[torch.topk(score[indices], k=count, sorted=False).indices]
        limited = torch.zeros_like(selected)
        limited[chosen] = True
        return limited

    def densify_and_prune(
        self,
        max_grad: float,
        min_opacity: float,
        extent: float,
        max_screen_size: Optional[float] = None,
        protected_mask: Optional[torch.Tensor] = None,
        densify_mask: Optional[torch.Tensor] = None,
        max_gaussians: Optional[int] = None,
    ) -> dict[str, int]:
        if protected_mask is not None:
            protected_mask = protected_mask.to(
                device=self.device, dtype=torch.bool
            )
            if protected_mask.numel() != self.num_gs:
                raise ValueError("Protected mask has the wrong number of Gaussians")
        if densify_mask is not None:
            densify_mask = densify_mask.to(
                device=self.device, dtype=torch.bool
            )
            if densify_mask.numel() != self.num_gs:
                raise ValueError("Densification mask has the wrong number of Gaussians")

        budget_limit = (
            self.DEFAULT_MAX_GAUSSIANS
            if max_gaussians is None
            else int(max_gaussians)
        )
        if budget_limit <= 0:
            raise ValueError("max_gaussians must be positive")
        budget = max(budget_limit - self.num_gs, 0)
        gradients = self._densification_gradients()
        world_scale = self.get_world_scale_max_approx()
        observed = self.denom.squeeze(-1) > 0
        clone_mask = (
            (gradients.squeeze(-1) >= max_grad)
            & (world_scale <= self.percent_dense * extent)
            & observed
        )
        split_mask = (
            (gradients.squeeze(-1) >= max_grad)
            & (world_scale > self.percent_dense * extent)
            & observed
        )
        if densify_mask is not None:
            clone_mask &= densify_mask
            split_mask &= densify_mask
        clone_mask = self._limit_to_budget(
            clone_mask, gradients.squeeze(-1), budget
        )
        cloned = int(clone_mask.sum().item())
        split_mask = self._limit_to_budget(
            split_mask, gradients.squeeze(-1), budget - cloned
        )
        split = int(split_mask.sum().item())

        if cloned or split:
            post_protected_mask = None
            if protected_mask is not None:
                protection_parts = [protected_mask[clone_mask]]
                if split:
                    protection_parts.append(
                        protected_mask[split_mask].repeat(2)
                    )
                post_protected_mask = torch.cat(
                    [protected_mask, *protection_parts], dim=0
                )

            uv_parts = [self._uv[clone_mask]]
            d_parts = [self._d[clone_mask]]
            dc_parts = [self._features_dc[clone_mask]]
            rest_parts = [self._features_rest[clone_mask]]
            opacity_parts = [self._opacity[clone_mask]]
            scaling_parts = [self._scaling[clone_mask]]
            rotation_parts = [self._rotation[clone_mask]]
            face_parts = [self._face_idx[clone_mask]]

            if split:
                vertices, normals = self._flame_verts_and_normals()
                split_faces = self._face_idx[split_mask]
                split_local_scaling = self.get_local_scaling[split_mask]
                split_local_rotation = self.get_local_rotation[split_mask]
                split_world_scaling, split_world_rotation = (
                    self._deformed_scaling_rotation(
                        vertices,
                        face_idx=split_faces,
                        local_scaling=split_local_scaling,
                        local_rotation=split_local_rotation,
                    )
                )
                samples = torch.randn(
                    (split * 2, 3), dtype=self._uv.dtype, device=self.device
                )
                samples *= split_world_scaling.repeat(2, 1)
                delta = torch.bmm(
                    build_rotation(split_world_rotation).repeat(2, 1, 1),
                    samples.unsqueeze(-1),
                ).squeeze(-1)
                split_uv, split_d = self._offset_uvd_by_world_delta(
                    self._uv[split_mask].repeat(2, 1),
                    self._d[split_mask].repeat(2, 1),
                    split_faces.repeat(2),
                    delta,
                    vertices,
                    normals,
                )
                uv_parts.append(split_uv)
                d_parts.append(split_d)
                dc_parts.append(self._features_dc[split_mask].repeat(2, 1, 1))
                rest_parts.append(self._features_rest[split_mask].repeat(2, 1, 1))
                opacity_parts.append(self._opacity[split_mask].repeat(2, 1))
                scaling_parts.append(
                    self._scaling[split_mask].repeat(2, 1) - math.log(1.6)
                )
                rotation_parts.append(self._rotation[split_mask].repeat(2, 1))
                face_parts.append(self._face_idx[split_mask].repeat(2))

            original_count = self.num_gs
            extension_count = cloned + split * 2
            self.densification_postfix(
                torch.cat(uv_parts),
                torch.cat(d_parts),
                torch.cat(dc_parts),
                torch.cat(rest_parts),
                torch.cat(opacity_parts),
                torch.cat(scaling_parts),
                torch.cat(rotation_parts),
                torch.cat(face_parts),
            )
            split_parents = torch.cat(
                [
                    split_mask,
                    torch.zeros(
                        extension_count, dtype=torch.bool, device=self.device
                    ),
                ]
            )
            if split:
                self.prune_points(split_parents)
                if post_protected_mask is not None:
                    post_protected_mask = post_protected_mask[
                        ~split_parents
                    ]
            assert self.num_gs == original_count + cloned + split
            protected_mask = post_protected_mask

        # Match AnimPortrait3D's full-stage ordering: densification resets
        # max_radii2D before the immediately following prune.  Pruning first
        # would apply the historical maximum screen radius and remove the
        # large splats that still provide essential silhouette coverage.
        opacity = self.get_opacity.squeeze(-1)
        prune = opacity < min_opacity
        if max_screen_size is not None:
            prune |= self.max_radii2D > float(max_screen_size)
        if protected_mask is not None:
            if protected_mask.numel() != self.num_gs:
                raise ValueError(
                    "Post-densification protected mask has the wrong number "
                    "of Gaussians"
                )
            prune &= ~protected_mask
        pruned = int(prune.sum().item())
        if prune.any():
            self.prune_points(prune)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"pruned": pruned, "cloned": cloned, "split": split, "after": self.num_gs}

    def prune_only(
        self,
        min_opacity: float,
        extent: float,
        protected_mask: Optional[torch.Tensor] = None,
    ) -> int:
        world_scale = self.get_world_scale_max_approx()
        prune = (
            (self.get_opacity.squeeze(-1) < min_opacity)
            | (world_scale > 0.03 * extent)
            | (world_scale < 0.001)
        )
        if protected_mask is not None:
            protected_mask = protected_mask.to(
                device=self.device, dtype=torch.bool
            )
            if protected_mask.numel() != self.num_gs:
                raise ValueError("Protected mask has the wrong number of Gaussians")
            prune &= ~protected_mask
        count = int(prune.sum().item())
        if prune.any():
            self.prune_points(prune)
        return count
