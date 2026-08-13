# Code heavily inspired by https://github.com/HavenFeng/photometric_optimization/blob/master/models/FLAME.py.
# Please consider citing their work if you find this code useful. The code is subject to the license available via
# https://github.com/vchoutas/smplx/edit/master/LICENSE

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de


"""Standalone FLAME model with detailed teeth.

This module intentionally contains its own copy of the canonical implementation.
It has no dependency on the older head-model module; only the dental
construction differs from that implementation.
"""
from .lbs import lbs, vertices2landmarks, blend_shapes, vertices2joints

import math as _math
from typing import Dict as _Dict, List as _List, Tuple as _Tuple

import torch
import torch.nn as nn
import numpy as np
import pickle
from collections import defaultdict
from pathlib import Path
from pytorch3d.io import load_obj

FLAME_VERSION = "FLAME_2023"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLAME_ASSET_DIR = PROJECT_ROOT / "ckpts" / FLAME_VERSION
FLAME_MESH_PATH = FLAME_ASSET_DIR / "head_template_mesh.obj"
FLAME_LMK_PATH = FLAME_ASSET_DIR / "landmark_embedding_with_eyes.npy"
FLAME_MEDIAPIPE_468_PATH = FLAME_ASSET_DIR / "flame2facemesh.npy"
FLAME_MEDIAPIPE_LMK_PATH = FLAME_ASSET_DIR / "mediapipe_landmark_embedding.npz"
FLAME_MODEL_PATH = FLAME_ASSET_DIR / "flame2023.pkl"
FLAME_PARTS_PATH = FLAME_ASSET_DIR / "FLAME_masks.pkl"


def to_tensor(array, dtype=torch.float32):
    if "torch.tensor" not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


class FlameHead(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(
            self,
            shape_params,
            expr_params,
            include_mask=True,
            add_teeth=True,
    ):
        super().__init__()

        self.n_shape_params = shape_params
        self.n_expr_params = expr_params

        with open(FLAME_MODEL_PATH, "rb") as f:
            ss = pickle.load(f, encoding="latin1")
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        # The vertices of the template model
        self.register_buffer(
            "v_template", to_tensor(to_np(flame_model.v_template), dtype=self.dtype)
        )

        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat(
            [shapedirs[:, :, :shape_params], shapedirs[:, :, 300: 300 + expr_params]],
            2,
        )
        self.register_buffer("shapedirs", shapedirs)

        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype))
        #
        self.register_buffer(
            "J_regressor", to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype)
        )
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer(
            "lbs_weights", to_tensor(to_np(flame_model.weights), dtype=self.dtype)
        )

        # Landmark embeddings for FLAME
        lmk_embeddings = np.load(
            FLAME_LMK_PATH, allow_pickle=True, encoding="latin1"
        )
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer(
            "full_lmk_faces_idx",
            torch.tensor(lmk_embeddings["full_lmk_faces_idx"], dtype=torch.long),
        )
        self.register_buffer(
            "full_lmk_bary_coords",
            torch.tensor(lmk_embeddings["full_lmk_bary_coords"], dtype=self.dtype),
        )

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer("neck_kin_chain", torch.stack(neck_kin_chain))

        # add faces and uvs
        verts, faces, aux = load_obj(FLAME_MESH_PATH, load_textures=False)

        vertex_uvs = aux.verts_uvs
        face_uvs_idx = faces.textures_idx  # index into verts_uvs

        # create uvcoords per face --> this is what you can use for uv map rendering
        # range from -1 to 1 (-1, -1) = left top; (+1, +1) = right bottom
        # pad 1 to the end
        pad = torch.ones(vertex_uvs.shape[0], 1)
        vertex_uvs = torch.cat([vertex_uvs, pad], dim=-1)
        vertex_uvs = vertex_uvs * 2 - 1
        vertex_uvs[..., 1] = -vertex_uvs[..., 1]

        face_uv_coords = face_vertices(vertex_uvs[None], face_uvs_idx[None])[0]
        self.register_buffer("face_uvcoords", face_uv_coords, persistent=False)
        self.register_buffer("faces", faces.verts_idx, persistent=False)

        self.register_buffer("verts_uvs", aux.verts_uvs, persistent=False)
        self.register_buffer("textures_idx", faces.textures_idx, persistent=False)

        if include_mask:
            self.mask = FlameMask(
                faces=self.faces,
                faces_t=self.textures_idx,
                num_verts=self.v_template.shape[0],
                num_faces=self.faces.shape[0],
            )

        if add_teeth:
            self.add_teeth()


    _TEETH_PER_ARCH = 14
    _RING_SEGMENTS = 16
    _GUM_SUBDIVISIONS = 4
    _GUM_SECTION_VERTICES = 6

    # Fourteen-tooth subset of the measured GSAvatar arch (M3 omitted), in
    # subject-right to subject-left order. Direct angular placement preserves
    # the tight anterior spacing and the much deeper posterior ellipse.
    _TOOTH_THETA_CENTERS: _Tuple[float, ...] = (
        -1.338,
        -1.116,
        -0.957,
        -0.788,
        -0.653,
        -0.403,
        -0.137,
        0.137,
        0.403,
        0.653,
        0.788,
        0.957,
        1.116,
        1.338,
    )

    # Posterior right -> anterior -> posterior left. The second value is kept
    # as descriptive relative width metadata; measured physical width ratios
    # in ``_TYPE_PROFILE`` drive the actual geometry.
    _TOOTH_SPECS: _Tuple[_Tuple[str, float], ...] = (
        ("molar2", 1.35),
        ("molar1", 1.28),
        ("premolar2", 0.94),
        ("premolar1", 0.92),
        ("canine", 0.88),
        ("lateral", 0.76),
        ("central", 1.05),
        ("central", 1.05),
        ("lateral", 0.76),
        ("canine", 0.88),
        ("premolar1", 0.92),
        ("premolar2", 0.94),
        ("molar1", 1.28),
        ("molar2", 1.35),
    )

    _TYPE_PROFILE: _Dict[str, _Dict[str, float]] = {
        # Ratios are relative to mouth width. ``depth`` is a half-depth and
        # crown is a full axial length. Like the GSAvatar assets, the crowns
        # stop at an open cervical ring and have no separately modelled roots.
        "central": {
            "depth": 0.076,
            "upper_width": 0.160,
            "lower_width": 0.120,
            "upper_crown": 0.120,
            "lower_crown": 0.132,
            "exponent": 4.2,
            "cusp": 0.0,
            "cusps": 0.0,
        },
        "lateral": {
            "depth": 0.072,
            "upper_width": 0.122,
            "lower_width": 0.122,
            "upper_crown": 0.122,
            "lower_crown": 0.136,
            "exponent": 3.8,
            "cusp": 0.0,
            "cusps": 0.0,
        },
        "canine": {
            "depth": 0.081,
            "upper_width": 0.132,
            "lower_width": 0.138,
            "upper_crown": 0.132,
            "lower_crown": 0.150,
            "exponent": 2.4,
            "cusp": 0.080,
            "cusps": 1.0,
        },
        "premolar1": {
            "depth": 0.088,
            "upper_width": 0.140,
            "lower_width": 0.150,
            "upper_crown": 0.118,
            "lower_crown": 0.128,
            "exponent": 3.0,
            "cusp": 0.042,
            "cusps": 2.0,
        },
        "premolar2": {
            "depth": 0.094,
            "upper_width": 0.150,
            "lower_width": 0.152,
            "upper_crown": 0.114,
            "lower_crown": 0.118,
            "exponent": 3.2,
            "cusp": 0.038,
            "cusps": 2.0,
        },
        "molar1": {
            "depth": 0.100,
            "upper_width": 0.205,
            "lower_width": 0.215,
            "upper_crown": 0.119,
            "lower_crown": 0.115,
            "exponent": 4.4,
            "cusp": 0.034,
            "cusps": 4.0,
        },
        "molar2": {
            "depth": 0.101,
            "upper_width": 0.205,
            "lower_width": 0.208,
            "upper_crown": 0.112,
            "lower_crown": 0.128,
            "exponent": 4.2,
            "cusp": 0.032,
            "cusps": 3.0,
        },
    }

    @staticmethod
    def _signed_power(value: torch.Tensor, power: float) -> torch.Tensor:
        return value.sign() * value.abs().pow(power)

    @staticmethod
    def _orient_faces(
        vertices: torch.Tensor,
        faces: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Flip triangles whose normals point away from their expected side."""

        triangles = vertices[faces]
        normals = torch.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=-1,
        )
        flip = (normals * targets).sum(dim=-1) < 0
        if bool(flip.any()):
            faces = faces.clone()
            faces[flip] = faces[flip][:, [0, 2, 1]]
        return faces

    def _build_dental_arch(
        self,
        *,
        upper: bool,
        mouth_width: torch.Tensor,
        occlusion_y: torch.Tensor,
        front_z: torch.Tensor,
    ) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build fourteen anatomically varied open-neck crowns.

        Returns vertices, local faces, UV vertices, and a per-vertex tooth id.
        Upper and lower arches deliberately receive the same numeric UV layout;
        the repository's layered-surface representation keeps them semantically
        separate even where those coordinates overlap.
        """

        device = self.v_template.device
        dtype = self.v_template.dtype
        scale = mouth_width

        arch_half_width = 0.51 * scale
        arch_depth = 1.02 * scale
        overjet = (0.018 if upper else -0.008) * scale
        vertical_overlap = (0.017 if upper else -0.014) * scale
        root_axis = torch.tensor(
            [0.0, 1.0 if upper else -1.0, 0.0],
            dtype=dtype,
            device=device,
        )

        theta_centers = torch.tensor(
            self._TOOTH_THETA_CENTERS, dtype=dtype, device=device
        )
        centerline_x = arch_half_width * torch.sin(theta_centers)
        centerline_z = -arch_depth * (1.0 - torch.cos(theta_centers))
        centerline = torch.stack((centerline_x, centerline_z), dim=-1)
        center_spacings = (centerline[1:] - centerline[:-1]).norm(dim=-1)

        vertices: _List[torch.Tensor] = []
        uvs: _List[torch.Tensor] = []
        faces: _List[torch.Tensor] = []
        tooth_ids: _List[torch.Tensor] = []
        vertex_cursor = 0

        ring_axial_fractions = (0.025, 0.18, 0.56, 1.0)
        width_profile = (0.84, 0.97, 1.00, 0.84)
        depth_profile = (0.74, 0.97, 1.00, 0.82)

        for tooth_idx, (tooth_type, _) in enumerate(self._TOOTH_SPECS):
            profile = self._TYPE_PROFILE[tooth_type]
            theta = theta_centers[tooth_idx]

            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            posterior = (
                theta.abs() / abs(self._TOOTH_THETA_CENTERS[-1])
            ).clamp(max=1.0)
            posterior_drop = 0.055 * posterior.pow(1.7) * scale
            center = torch.stack(
                (
                    arch_half_width * sin_theta,
                    occlusion_y + vertical_overlap - posterior_drop,
                    front_z - arch_depth * (1.0 - cos_theta) + overjet,
                )
            )

            tangent = torch.stack(
                (
                    arch_half_width * cos_theta,
                    theta.new_zeros(()),
                    -arch_depth * sin_theta,
                )
            )
            tangent = tangent / tangent.norm().clamp_min(torch.finfo(dtype).eps)
            outward = torch.stack((-tangent[2], tangent.new_zeros(()), tangent[0]))

            target_width = profile[
                "upper_width" if upper else "lower_width"
            ] * scale
            if tooth_idx == 0:
                local_spacing = center_spacings[0]
            elif tooth_idx == self._TEETH_PER_ARCH - 1:
                local_spacing = center_spacings[-1]
            else:
                local_spacing = torch.minimum(
                    center_spacings[tooth_idx - 1], center_spacings[tooth_idx]
                )
            # Simple superelliptic crowns do not have the asymmetric contact
            # facets of the scan-derived reference. Reserve a small interdental
            # clearance so their broader middle rings never self-intersect.
            half_width = 0.5 * torch.minimum(target_width, 0.86 * local_spacing)
            half_depth = profile["depth"] * scale
            crown = profile["upper_crown" if upper else "lower_crown"] * scale

            axial_values = [fraction * crown for fraction in ring_axial_fractions]

            local_vertices: _List[torch.Tensor] = []
            local_coordinates: _List[torch.Tensor] = []
            local_uvs: _List[torch.Tensor] = []

            tile_width = 0.24 / self._TEETH_PER_ARCH
            # Geometry is generated from x-negative (subject right) to
            # x-positive (subject left).  The canonical FLAME lip UV runs in
            # the opposite index direction, so increase U here to preserve the
            # original left/right texture orientation.
            tile_center = 0.38 + (tooth_idx + 0.5) * tile_width
            for ring_idx, axial in enumerate(axial_values):
                for segment in range(self._RING_SEGMENTS):
                    phi = 2.0 * _math.pi * segment / self._RING_SEGMENTS
                    cos_phi = torch.cos(theta.new_tensor(phi))
                    sin_phi = torch.sin(theta.new_tensor(phi))
                    exponent = profile["exponent"]
                    local_tangent = self._signed_power(cos_phi, 2.0 / exponent)
                    local_depth = self._signed_power(sin_phi, 2.0 / exponent)
                    tangent_offset = local_tangent * half_width * width_profile[ring_idx]
                    depth_offset = local_depth * half_depth * depth_profile[ring_idx]

                    ring_axial = axial
                    if ring_idx == 0 and profile["cusps"] > 1.0:
                        cusp_wave = 0.5 + 0.5 * torch.cos(
                            theta.new_tensor(profile["cusps"] * phi)
                        )
                        ring_axial = axial - profile["cusp"] * crown * cusp_wave

                    point = (
                        center
                        + tangent * tangent_offset
                        + outward * depth_offset
                        + root_axis * ring_axial
                    )
                    local_vertices.append(point)
                    local_coordinates.append(
                        torch.stack((tangent_offset, depth_offset, ring_axial))
                    )

                    u = tile_center + (segment / self._RING_SEGMENTS - 0.5) * (0.90 * tile_width)
                    # Keep cusp tips slightly outside the crown's nominal UV
                    # interval.  Clamping them to zero would collapse several
                    # cap triangles onto one UV line even though their 3D area
                    # is valid.
                    normalized_axial = (ring_axial / crown).clamp(-0.08, 1.0)
                    v = 0.992 - 0.038 * normalized_axial
                    local_uvs.append(torch.stack((theta.new_tensor(u), v)))

            edge_center_axial = -profile["cusp"] * crown if profile["cusps"] == 1.0 else 0.0 * crown
            edge_center = center + root_axis * edge_center_axial
            local_vertices.append(edge_center)
            local_coordinates.append(
                torch.stack(
                    (
                        theta.new_zeros(()),
                        theta.new_zeros(()),
                        edge_center_axial,
                    )
                )
            )
            local_uvs.append(
                torch.stack(
                    (theta.new_tensor(tile_center), theta.new_tensor(0.992))
                )
            )

            tooth_vertices = torch.stack(local_vertices)
            tooth_local = torch.stack(local_coordinates)
            edge_center_idx = 4 * self._RING_SEGMENTS
            tooth_faces: _List[_Tuple[int, int, int]] = []
            tooth_targets: _List[torch.Tensor] = []

            for ring_idx in range(3):
                row = ring_idx * self._RING_SEGMENTS
                next_row = (ring_idx + 1) * self._RING_SEGMENTS
                for segment in range(self._RING_SEGMENTS):
                    next_segment = (segment + 1) % self._RING_SEGMENTS
                    a = row + segment
                    b = row + next_segment
                    c = next_row + segment
                    d = next_row + next_segment
                    tooth_faces.extend(((a, b, c), (b, d, c)))
                    for tri in ((a, b, c), (b, d, c)):
                        radial = tooth_local[list(tri), :2].mean(dim=0)
                        tooth_targets.append(tangent * radial[0] + outward * radial[1])

            for segment in range(self._RING_SEGMENTS):
                next_segment = (segment + 1) % self._RING_SEGMENTS
                tooth_faces.append((edge_center_idx, segment, next_segment))
                tooth_targets.append(-root_axis)

            tooth_faces_tensor = torch.tensor(
                tooth_faces, dtype=torch.long, device=device
            )
            tooth_targets_tensor = torch.stack(tooth_targets)
            tooth_faces_tensor = self._orient_faces(
                tooth_vertices, tooth_faces_tensor, tooth_targets_tensor
            )

            vertices.append(tooth_vertices)
            uvs.append(torch.stack(local_uvs).to(dtype=self.verts_uvs.dtype))
            faces.append(tooth_faces_tensor + vertex_cursor)
            tooth_ids.append(
                torch.full(
                    (tooth_vertices.shape[0],),
                    tooth_idx,
                    dtype=torch.long,
                    device=device,
                )
            )
            vertex_cursor += tooth_vertices.shape[0]

        return (
            torch.cat(vertices, dim=0),
            torch.cat(faces, dim=0),
            torch.cat(uvs, dim=0),
            torch.cat(tooth_ids, dim=0),
        )

    def _build_gum_arch(
        self,
        *,
        upper: bool,
        mouth_width: torch.Tensor,
        occlusion_y: torch.Tensor,
        front_z: torch.Tensor,
    ) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a continuous scalloped gingival base around one tooth arch.

        The open posterior ends follow the reference GSAvatar OBJ layout. The
        six-sided section is closed at the occlusal and root-facing surfaces;
        open crown necks pass into its scalloped cervical collar. Geometry and
        UV faces are returned separately because the UV section duplicates its
        seam.
        """

        device = self.v_template.device
        dtype = self.v_template.dtype
        scale = mouth_width
        arch_half_width = 0.51 * scale
        arch_depth = 1.02 * scale
        overjet = (0.018 if upper else -0.008) * scale
        vertical_overlap = (0.017 if upper else -0.014) * scale
        root_axis = torch.tensor(
            [0.0, 1.0 if upper else -1.0, 0.0],
            dtype=dtype,
            device=device,
        )

        theta_centers = torch.tensor(
            self._TOOTH_THETA_CENTERS, dtype=dtype, device=device
        )
        theta_midpoints = 0.5 * (theta_centers[:-1] + theta_centers[1:])
        first_edge = theta_centers[0] - 0.5 * (
            theta_centers[1] - theta_centers[0]
        )
        last_edge = theta_centers[-1] + 0.5 * (
            theta_centers[-1] - theta_centers[-2]
        )
        theta_edges = torch.cat(
            (first_edge.reshape(1), theta_midpoints, last_edge.reshape(1))
        )
        theta_limit = theta_edges[-1]

        column_specs: _List[_Tuple[int, float, torch.Tensor]] = []
        for tooth_idx in range(self._TEETH_PER_ARCH):
            theta_lo = theta_edges[tooth_idx]
            theta_hi = theta_edges[tooth_idx + 1]
            for subdivision in range(self._GUM_SUBDIVISIONS):
                phase = subdivision / self._GUM_SUBDIVISIONS
                theta = torch.lerp(theta_lo, theta_hi, phase)
                column_specs.append((tooth_idx, phase, theta))
        column_specs.append((self._TEETH_PER_ARCH - 1, 1.0, theta_edges[-1]))

        vertices: _List[torch.Tensor] = []
        uvs: _List[torch.Tensor] = []
        outwards: _List[torch.Tensor] = []
        column_count = len(column_specs)
        for column_idx, (tooth_idx, phase, theta) in enumerate(column_specs):
            tooth_type = self._TOOTH_SPECS[tooth_idx][0]
            profile = self._TYPE_PROFILE[tooth_type]
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            posterior = (
                theta.abs() / abs(self._TOOTH_THETA_CENTERS[-1])
            ).clamp(max=1.0)
            posterior_drop = 0.055 * posterior.pow(1.7) * scale
            center = torch.stack(
                (
                    arch_half_width * sin_theta,
                    occlusion_y + vertical_overlap - posterior_drop,
                    front_z - arch_depth * (1.0 - cos_theta) + overjet,
                )
            )
            tangent = torch.stack(
                (
                    arch_half_width * cos_theta,
                    theta.new_zeros(()),
                    -arch_depth * sin_theta,
                )
            )
            tangent = tangent / tangent.norm().clamp_min(torch.finfo(dtype).eps)
            outward = torch.stack((-tangent[2], tangent.new_zeros(()), tangent[0]))
            outwards.append(outward)

            crown = profile["upper_crown" if upper else "lower_crown"] * scale
            tooth_depth = profile["depth"] * scale
            phase_tensor = theta.new_tensor(phase)
            scallop = -torch.cos(2.0 * _math.pi * phase_tensor)
            margin_axial = (0.94 + 0.045 * scallop) * crown
            posterior = theta.abs() / theta_limit
            base_axial = (
                (0.420 if upper else 0.435)
                + (0.035 if upper else 0.100) * posterior
            ) * scale
            mid_axial = torch.lerp(margin_axial, base_axial, 0.43)

            neck_outer = 0.86 * tooth_depth + 0.008 * scale
            neck_inner = 0.78 * tooth_depth + 0.006 * scale
            base_outer = (0.100 + 0.025 * posterior) * scale
            base_inner = (0.085 + 0.020 * posterior) * scale
            base_outer = torch.maximum(base_outer, neck_outer + 0.018 * scale)
            base_inner = torch.maximum(base_inner, neck_inner + 0.016 * scale)
            mid_outer = torch.lerp(neck_outer, base_outer, 0.55)
            mid_inner = torch.lerp(neck_inner, base_inner, 0.55)

            section = torch.stack(
                (
                    center + outward * neck_outer + root_axis * margin_axial,
                    center + outward * mid_outer + root_axis * mid_axial,
                    center + outward * base_outer + root_axis * base_axial,
                    center - outward * base_inner + root_axis * base_axial,
                    center - outward * mid_inner + root_axis * mid_axial,
                    center - outward * neck_inner + root_axis * margin_axial,
                )
            )
            vertices.append(section)

            u = 0.355 + 0.290 * column_idx / (column_count - 1)
            perimeter_v = (0.950, 0.938, 0.920, 0.904, 0.920, 0.938, 0.950)
            uvs.append(
                torch.stack(
                    tuple(
                        torch.stack((theta.new_tensor(u), theta.new_tensor(v)))
                        for v in perimeter_v
                    )
                ).to(dtype=self.verts_uvs.dtype)
            )

        gum_vertices = torch.cat(vertices, dim=0)
        raw_faces: _List[_Tuple[int, int, int]] = []
        texture_faces: _List[_Tuple[int, int, int]] = []
        targets: _List[torch.Tensor] = []
        section_vertices = self._GUM_SECTION_VERTICES
        uv_section_vertices = section_vertices + 1
        for column_idx in range(column_count - 1):
            average_outward = outwards[column_idx] + outwards[column_idx + 1]
            average_outward = average_outward / average_outward.norm().clamp_min(
                torch.finfo(dtype).eps
            )
            for section_idx in range(section_vertices):
                next_section = (section_idx + 1) % section_vertices
                a = column_idx * section_vertices + section_idx
                b = (column_idx + 1) * section_vertices + section_idx
                c = column_idx * section_vertices + next_section
                d = (column_idx + 1) * section_vertices + next_section
                raw_faces.extend(((a, b, c), (b, d, c)))

                ta = column_idx * uv_section_vertices + section_idx
                tb = (column_idx + 1) * uv_section_vertices + section_idx
                tc = column_idx * uv_section_vertices + section_idx + 1
                td = (column_idx + 1) * uv_section_vertices + section_idx + 1
                texture_faces.extend(((ta, tb, tc), (tb, td, tc)))

                if section_idx <= 1:
                    target = average_outward
                elif section_idx == 2:
                    target = root_axis
                elif section_idx <= 4:
                    target = -average_outward
                else:
                    target = -root_axis
                targets.extend((target, target))

        raw_faces_tensor = torch.tensor(raw_faces, dtype=torch.long, device=device)
        gum_faces = self._orient_faces(
            gum_vertices,
            raw_faces_tensor,
            torch.stack(targets),
        )
        gum_texture_faces = torch.tensor(
            texture_faces, dtype=torch.long, device=device
        )
        flipped = gum_faces[:, 1] != raw_faces_tensor[:, 1]
        if bool(flipped.any()):
            gum_texture_faces[flipped] = gum_texture_faces[flipped][:, [0, 2, 1]]

        return (
            gum_vertices,
            gum_faces,
            torch.cat(uvs, dim=0),
            gum_texture_faces,
        )

    def _build_oral_cavity_wall(
        self,
        *,
        dental_vertices: torch.Tensor,
        num_upper_tooth: int,
        num_upper: int,
        num_lower_tooth: int,
        gum_columns: int,
    ) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bridge the lingual upper/lower gum boundaries into an oral wall.

        AnimPortrait3D closes the otherwise empty mouth by connecting matching
        boundary vertices of its upper- and lower-teeth meshes.  The generated
        dentition here has a regular six-vertex gingival section, so section 3
        (the inner/root boundary) provides the equivalent ordered curves.  The
        faces reuse those articulated vertices: upper vertices retain head LBS
        weights and lower vertices retain jaw LBS weights, while Gaussians bind
        to the new faces through this project's UVD/face-index representation.
        """

        device = dental_vertices.device
        dtype = dental_vertices.dtype
        section_vertices = self._GUM_SECTION_VERTICES
        inner_section = 3
        columns = torch.arange(gum_columns, dtype=torch.long, device=device)
        upper_boundary = (
            num_upper_tooth + columns * section_vertices + inner_section
        )
        lower_boundary = (
            num_upper
            + num_lower_tooth
            + columns * section_vertices
            + inner_section
        )

        upper_left, upper_right = upper_boundary[:-1], upper_boundary[1:]
        lower_left, lower_right = lower_boundary[:-1], lower_boundary[1:]
        raw_faces = torch.stack(
            (
                torch.stack((upper_left, upper_right, lower_left), dim=-1),
                torch.stack((lower_left, upper_right, lower_right), dim=-1),
            ),
            dim=1,
        ).reshape(-1, 3)
        # FLAME uses +Z towards the mouth opening.  Keep the cavity-facing side
        # consistently oriented so normals and UVD depth point predictably.
        targets = torch.tensor(
            (0.0, 0.0, 1.0), dtype=dtype, device=device
        ).expand(raw_faces.shape[0], -1)
        oral_faces = self._orient_faces(dental_vertices, raw_faces, targets)

        u = torch.linspace(0.355, 0.645, gum_columns, dtype=dtype, device=device)
        oral_uvs = torch.cat(
            (
                torch.stack((u, torch.full_like(u, 0.895)), dim=-1),
                torch.stack((u, torch.full_like(u, 0.850)), dim=-1),
            ),
            dim=0,
        ).to(dtype=self.verts_uvs.dtype)
        upper_uv = columns
        lower_uv = columns + gum_columns
        raw_texture_faces = torch.stack(
            (
                torch.stack(
                    (upper_uv[:-1], upper_uv[1:], lower_uv[:-1]), dim=-1
                ),
                torch.stack(
                    (lower_uv[:-1], upper_uv[1:], lower_uv[1:]), dim=-1
                ),
            ),
            dim=1,
        ).reshape(-1, 3)
        oral_texture_faces = raw_texture_faces.clone()
        flipped = oral_faces[:, 1] != raw_faces[:, 1]
        if bool(flipped.any()):
            oral_texture_faces[flipped] = oral_texture_faces[flipped][
                :, [0, 2, 1]
            ]

        return (
            oral_faces,
            oral_uvs,
            oral_texture_faces,
            torch.cat((upper_boundary, lower_boundary), dim=0),
        )

    @staticmethod
    def _interpolated_lip_shapedirs(
        shapedirs: torch.Tensor,
        lip_ids: torch.Tensor,
        tooth_count: int,
        vertices_per_tooth: int,
        n_shape_params: int,
    ) -> torch.Tensor:
        """Give each rigid tooth the interpolated motion of its lip anchor."""

        chunks: _List[torch.Tensor] = []
        max_position = lip_ids.numel() - 1
        for tooth_idx in range(tooth_count):
            position = (tooth_idx + 0.5) * max_position / tooth_count
            left = int(_math.floor(position))
            right = min(left + 1, max_position)
            alpha = position - left
            source = torch.lerp(
                shapedirs[lip_ids[left], :, :n_shape_params],
                shapedirs[lip_ids[right], :, :n_shape_params],
                alpha,
            )
            chunks.append(source.unsqueeze(0).expand(vertices_per_tooth, -1, -1))
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _interpolated_curve_shapedirs(
        shapedirs: torch.Tensor,
        lip_ids: torch.Tensor,
        column_count: int,
        vertices_per_column: int,
        n_shape_params: int,
    ) -> torch.Tensor:
        """Interpolate lip shape motion along every gingival arch column."""

        chunks: _List[torch.Tensor] = []
        max_position = lip_ids.numel() - 1
        for column_idx in range(column_count):
            position = column_idx * max_position / (column_count - 1)
            left = int(_math.floor(position))
            right = min(left + 1, max_position)
            alpha = position - left
            source = torch.lerp(
                shapedirs[lip_ids[left], :, :n_shape_params],
                shapedirs[lip_ids[right], :, :n_shape_params],
                alpha,
            )
            chunks.append(source.unsqueeze(0).expand(vertices_per_column, -1, -1))
        return torch.cat(chunks, dim=0)

    def add_teeth(self) -> None:
        """Append dentition, gingival bases, and an articulated oral wall."""

        upper_lip_ids = self.mask.get_vid_by_region(
            "lip_outside_ring_upper", keep_order=True
        )
        lower_lip_ids = self.mask.get_vid_by_region(
            "lip_outside_ring_lower", keep_order=True
        )
        if upper_lip_ids.numel() != lower_lip_ids.numel() or upper_lip_ids.numel() < 2:
            raise RuntimeError("FLAME lip rings cannot define a dental arch")

        upper_lip = self.v_template[upper_lip_ids]
        lower_lip = self.v_template[lower_lip_ids]
        mouth_width = torch.maximum(
            upper_lip[:, 0].max() - upper_lip[:, 0].min(),
            lower_lip[:, 0].max() - lower_lip[:, 0].min(),
        )
        occlusion_y = 0.5 * (upper_lip[:, 1].mean() + lower_lip[:, 1].mean())
        center_column = upper_lip_ids.numel() // 2
        lip_front_z = 0.5 * (
            upper_lip[center_column, 2] + lower_lip[center_column, 2]
        )
        # GSAvatar aligns the labial crown surface about 0.35--0.38 mouth
        # widths behind the lip. Central crown half-depth is about 0.076 W.
        front_z = lip_front_z - 0.46 * mouth_width

        (
            upper_tooth_vertices,
            upper_tooth_faces,
            upper_tooth_uvs,
            upper_tooth_ids,
        ) = self._build_dental_arch(
            upper=True,
            mouth_width=mouth_width,
            occlusion_y=occlusion_y,
            front_z=front_z,
        )
        (
            lower_tooth_vertices,
            lower_tooth_faces,
            lower_tooth_uvs,
            lower_tooth_ids,
        ) = self._build_dental_arch(
            upper=False,
            mouth_width=mouth_width,
            occlusion_y=occlusion_y,
            front_z=front_z,
        )
        (
            upper_gum_vertices,
            upper_gum_faces,
            upper_gum_uvs,
            upper_gum_texture_faces,
        ) = self._build_gum_arch(
            upper=True,
            mouth_width=mouth_width,
            occlusion_y=occlusion_y,
            front_z=front_z,
        )
        (
            lower_gum_vertices,
            lower_gum_faces,
            lower_gum_uvs,
            lower_gum_texture_faces,
        ) = self._build_gum_arch(
            upper=False,
            mouth_width=mouth_width,
            occlusion_y=occlusion_y,
            front_z=front_z,
        )

        num_verts_orig = int(self.v_template.shape[0])
        num_upper_tooth = int(upper_tooth_vertices.shape[0])
        num_lower_tooth = int(lower_tooth_vertices.shape[0])
        num_upper_gum = int(upper_gum_vertices.shape[0])
        num_lower_gum = int(lower_gum_vertices.shape[0])
        num_upper = num_upper_tooth + num_upper_gum
        num_lower = num_lower_tooth + num_lower_gum
        num_dental_vertices = num_upper + num_lower
        vertices_per_tooth = num_upper_tooth // self._TEETH_PER_ARCH
        gum_columns = num_upper_gum // self._GUM_SECTION_VERTICES

        upper_vertices = torch.cat((upper_tooth_vertices, upper_gum_vertices), dim=0)
        lower_vertices = torch.cat((lower_tooth_vertices, lower_gum_vertices), dim=0)
        dental_vertices = torch.cat((upper_vertices, lower_vertices), dim=0)
        (
            oral_cavity_faces,
            oral_cavity_uvs,
            oral_cavity_texture_faces,
            oral_cavity_local_ids,
        ) = self._build_oral_cavity_wall(
            dental_vertices=dental_vertices,
            num_upper_tooth=num_upper_tooth,
            num_upper=num_upper,
            num_lower_tooth=num_lower_tooth,
            gum_columns=gum_columns,
        )
        self.v_template = torch.cat((self.v_template, dental_vertices), dim=0)
        upper_ids = torch.arange(
            num_verts_orig,
            num_verts_orig + num_upper,
            dtype=torch.long,
            device=self.v_template.device,
        )
        lower_ids = torch.arange(
            num_verts_orig + num_upper,
            num_verts_orig + num_dental_vertices,
            dtype=torch.long,
            device=self.v_template.device,
        )
        teeth_ids = torch.cat((upper_ids, lower_ids), dim=0)
        upper_crown_ids = upper_ids[:num_upper_tooth]
        upper_gum_ids = upper_ids[num_upper_tooth:]
        lower_crown_ids = lower_ids[:num_lower_tooth]
        lower_gum_ids = lower_ids[num_lower_tooth:]
        crown_ids = torch.cat((upper_crown_ids, lower_crown_ids), dim=0)
        gum_ids = torch.cat((upper_gum_ids, lower_gum_ids), dim=0)
        oral_cavity_ids = oral_cavity_local_ids + num_verts_orig

        # ``teeth`` intentionally includes the base so existing include-teeth
        # paths never leave floating gingiva behind. Finer masks split enamel
        # crowns from gingiva for colouring and supervision.
        self.num_verts_teeth = num_dental_vertices
        self.num_upper_teeth = num_upper
        self.num_lower_teeth = num_lower
        self.num_verts_tooth_crowns = num_upper_tooth + num_lower_tooth
        self.num_verts_gums = num_upper_gum + num_lower_gum
        self.num_faces_oral_cavity = int(oral_cavity_faces.shape[0])
        self.mask.v.register_buffer("teeth_upper", upper_ids)
        self.mask.v.register_buffer("teeth_lower", lower_ids)
        self.mask.v.register_buffer("teeth", teeth_ids)
        self.mask.v.register_buffer("teeth_crowns_upper", upper_crown_ids)
        self.mask.v.register_buffer("teeth_crowns_lower", lower_crown_ids)
        self.mask.v.register_buffer("teeth_crowns", crown_ids)
        self.mask.v.register_buffer("gums_upper", upper_gum_ids)
        self.mask.v.register_buffer("gums_lower", lower_gum_ids)
        self.mask.v.register_buffer("gums", gum_ids)
        self.mask.v.register_buffer("oral_cavity", oral_cavity_ids)

        half_tolerance = float(mouth_width.detach()) * 1.0e-4
        dental_left = teeth_ids[dental_vertices[:, 0] >= -half_tolerance]
        dental_right = teeth_ids[dental_vertices[:, 0] <= half_tolerance]
        self.mask.v.left_half = torch.cat((self.mask.v.left_half, dental_left), dim=0)
        self.mask.v.right_half = torch.cat((self.mask.v.right_half, dental_right), dim=0)

        num_uv_orig = int(self.verts_uvs.shape[0])
        upper_uvs = torch.cat((upper_tooth_uvs, upper_gum_uvs), dim=0)
        lower_uvs = torch.cat((lower_tooth_uvs, lower_gum_uvs), dim=0)
        dental_uvs = torch.cat((upper_uvs, lower_uvs), dim=0)
        self.verts_uvs = torch.cat(
            (self.verts_uvs, dental_uvs, oral_cavity_uvs), dim=0
        )

        old_shapedirs = self.shapedirs
        dental_shapedirs = torch.zeros(
            (num_dental_vertices, old_shapedirs.shape[1], old_shapedirs.shape[2]),
            dtype=old_shapedirs.dtype,
            device=old_shapedirs.device,
        )
        dental_shapedirs[:num_upper_tooth, :, : self.n_shape_params] = (
            self._interpolated_lip_shapedirs(
                old_shapedirs,
                upper_lip_ids.flip(0),
                self._TEETH_PER_ARCH,
                vertices_per_tooth,
                self.n_shape_params,
            )
        )
        dental_shapedirs[
            num_upper_tooth:num_upper, :, : self.n_shape_params
        ] = self._interpolated_curve_shapedirs(
            old_shapedirs,
            upper_lip_ids.flip(0),
            gum_columns,
            self._GUM_SECTION_VERTICES,
            self.n_shape_params,
        )
        lower_tooth_start = num_upper
        lower_gum_start = lower_tooth_start + num_lower_tooth
        dental_shapedirs[
            lower_tooth_start:lower_gum_start, :, : self.n_shape_params
        ] = (
            self._interpolated_lip_shapedirs(
                old_shapedirs,
                lower_lip_ids.flip(0),
                self._TEETH_PER_ARCH,
                vertices_per_tooth,
                self.n_shape_params,
            )
        )
        dental_shapedirs[
            lower_gum_start:, :, : self.n_shape_params
        ] = self._interpolated_curve_shapedirs(
            old_shapedirs,
            lower_lip_ids.flip(0),
            gum_columns,
            self._GUM_SECTION_VERTICES,
            self.n_shape_params,
        )
        self.shapedirs = torch.cat((old_shapedirs, dental_shapedirs), dim=0)

        pose_basis = len(self.parents) - 1
        posedirs = self.posedirs.reshape(pose_basis, 9, num_verts_orig, 3)
        posedirs = torch.cat(
            (
                posedirs,
                torch.zeros(
                    (pose_basis, 9, num_dental_vertices, 3),
                    dtype=posedirs.dtype,
                    device=posedirs.device,
                ),
            ),
            dim=2,
        )
        self.posedirs = posedirs.reshape(pose_basis * 9, -1)
        self.J_regressor = torch.cat(
            (
                self.J_regressor,
                torch.zeros(
                    (self.J_regressor.shape[0], num_dental_vertices),
                    dtype=self.J_regressor.dtype,
                    device=self.J_regressor.device,
                ),
            ),
            dim=1,
        )

        dental_weights = torch.zeros(
            (num_dental_vertices, self.lbs_weights.shape[1]),
            dtype=self.lbs_weights.dtype,
            device=self.lbs_weights.device,
        )
        dental_weights[:num_upper, 1] = 1.0
        dental_weights[num_upper:, 2] = 1.0
        self.lbs_weights = torch.cat((self.lbs_weights, dental_weights), dim=0)

        upper_faces = torch.cat(
            (upper_tooth_faces, upper_gum_faces + num_upper_tooth), dim=0
        )
        lower_faces = torch.cat(
            (lower_tooth_faces, lower_gum_faces + num_lower_tooth), dim=0
        )
        upper_texture_faces = torch.cat(
            (
                upper_tooth_faces,
                upper_gum_texture_faces + int(upper_tooth_uvs.shape[0]),
            ),
            dim=0,
        )
        lower_texture_faces = torch.cat(
            (
                lower_tooth_faces,
                lower_gum_texture_faces + int(lower_tooth_uvs.shape[0]),
            ),
            dim=0,
        )
        dental_faces = torch.cat((upper_faces, lower_faces + num_upper), dim=0)
        dental_texture_faces = torch.cat(
            (
                upper_texture_faces,
                lower_texture_faces + int(upper_uvs.shape[0]),
            ),
            dim=0,
        )
        appended_faces = torch.cat(
            (dental_faces, oral_cavity_faces), dim=0
        )
        appended_texture_faces = torch.cat(
            (
                dental_texture_faces,
                oral_cavity_texture_faces + int(dental_uvs.shape[0]),
            ),
            dim=0,
        )
        self.faces = torch.cat(
            (self.faces, appended_faces + num_verts_orig), dim=0
        )
        self.textures_idx = torch.cat(
            (self.textures_idx, appended_texture_faces + num_uv_orig), dim=0
        )

        self.register_buffer(
            "teeth_vertex_to_tooth",
            torch.cat((upper_tooth_ids, lower_tooth_ids + self._TEETH_PER_ARCH)),
            persistent=False,
        )
        self.register_buffer(
            "dental_vertex_to_tooth",
            torch.cat(
                (
                    upper_tooth_ids,
                    torch.full(
                        (num_upper_gum,),
                        -1,
                        dtype=torch.long,
                        device=self.v_template.device,
                    ),
                    lower_tooth_ids + self._TEETH_PER_ARCH,
                    torch.full(
                        (num_lower_gum,),
                        -1,
                        dtype=torch.long,
                        device=self.v_template.device,
                    ),
                )
            ),
            persistent=False,
        )
        self.mask.update(self.faces, self.textures_idx)

    def forward(
            self,
            shape,
            expr,
            rotation,
            neck,
            jaw,
            eyes,
            translation,
            zero_centered_at_root_node=False,  # otherwise, zero centered at the face
            return_landmarks=True,
            return_verts_cano=False,
            static_offset=None,
            dynamic_offset=None,
    ):
        """
        Input:
            shape_params: N X number of shape parameters
            expression_params: N X number of expression parameters
            pose_params: N X number of pose parameters (6)
        return:d
            vertices: N X V X 3
            landmarks: N X number of landmarks X 3
        """
        batch_size = shape.shape[0]

        betas = torch.cat([shape, expr], dim=1)
        full_pose = torch.cat([rotation, neck, jaw, eyes], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        # Add shape contribution
        v_shaped = template_vertices + blend_shapes(betas, self.shapedirs)

        # Add personal offsets
        if static_offset is not None:
            v_shaped += static_offset

        vertices, J, mat_rot = lbs(
            full_pose,
            v_shaped,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            dtype=self.dtype,
        )

        if zero_centered_at_root_node:
            vertices = vertices - J[:, [0]]
            J = J - J[:, [0]]

        vertices = vertices + translation[:, None, :]
        J = J + translation[:, None, :]

        ret_vals = [vertices]

        if return_verts_cano:
            ret_vals.append(v_shaped)

        # compute landmarks if desired
        if return_landmarks:
            bz = vertices.shape[0]
            landmarks = vertices2landmarks(
                vertices,
                self.faces,
                self.full_lmk_faces_idx.repeat(bz, 1),
                self.full_lmk_bary_coords.repeat(bz, 1, 1),
            )
            ret_vals.append(landmarks)

        if len(ret_vals) > 1:
            return ret_vals
        else:
            return ret_vals[0]


class BufferContainer(nn.Module):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        main_str = super().__repr__() + '\n'
        for name, buf in self.named_buffers():
            main_str += f'    {name:20}\t{buf.shape}\t{buf.dtype}\n'
        return main_str

    def __iter__(self):
        for name, buf in self.named_buffers():
            yield name, buf

    def keys(self):
        return [name for name, buf in self.named_buffers()]

    def items(self):
        return [(name, buf) for name, buf in self.named_buffers()]


class FlameMask(nn.Module):
    def __init__(
            self,
            faces=None,
            faces_t=None,
            num_verts=5023,
            num_faces=9976,
            face_clusters=[],
    ):
        super().__init__()
        self.faces = faces
        self.faces_t = faces_t
        self.face_clusters = face_clusters
        self.num_verts = num_verts
        if faces is not None:
            self.num_faces = faces.shape[0]
        else:
            self.num_faces = num_faces

        self.process_vertex_mask(FLAME_PARTS_PATH)

        if self.faces is not None:
            self.construct_vid_table()
            self.process_face_mask(self.faces)
            self.process_face_clusters(self.face_clusters)
            if self.faces_t is not None:
                self.process_vt_mask(self.faces, self.faces_t)

    def update(self, faces=None, faces_t=None, face_clusters=None):
        """Update the faces properties when vertex masks are changed"""
        if faces is not None:
            self.faces = faces
            self.num_faces = faces.shape[0]
        if faces_t is not None:
            self.faces_t = faces_t
        if face_clusters is not None:
            self.face_clusters = face_clusters

        self.construct_vid_table()
        self.process_face_mask(self.faces)
        self.process_face_clusters(self.face_clusters)
        if self.faces_t is not None:
            self.process_vt_mask(self.faces, self.faces_t)

    def process_vertex_mask(self, flame_parts_path):
        """Load the vertex masks from the FLAME model and add custom masks"""

        part_masks = np.load(flame_parts_path, allow_pickle=True, encoding="latin1")
        """ Available part masks from the FLAME model: 
                face, neck, scalp, boundary, right_eyeball, left_eyeball, 
                right_ear, left_ear, forehead, eye_region, nose, lips,
                right_eye_region, left_eye_region.
        """

        self.v = BufferContainer()
        for k, v_mask in part_masks.items():
            self.v.register_buffer(k, torch.tensor(v_mask, dtype=torch.long))

        self.create_custom_mask()

    def create_custom_mask(self):
        """Add some cutom masks based on the original FLAME masks"""

        self.v.register_buffer("neck_left_point", torch.tensor([3193]))
        self.v.register_buffer("neck_right_point", torch.tensor([3296]))
        self.v.register_buffer("front_middle_bottom_point_boundary", torch.tensor([3285]))
        self.v.register_buffer("back_middle_bottom_point_boundary", torch.tensor([3248]))

        self.v.register_buffer(
            "neck_top",
            torch.tensor([
                10, 11, 111, 112, 784, 795, 1325, 1901, 2115, 2162, 2251, 2254, 2483, 2979, 3142, 3174, 3441, 3442,
                3443, 3444, 3445, 3446, 3447, 3448, 3449, 3562, 3673, 3676, 3677, 3678, 3679, 3680, 3681, 3685,
            ])
        )

        self.v.register_buffer(
            "lip_inside_ring_upper",
            torch.tensor([
                1595, 1746, 1747, 1742, 1739, 1665, 1666, 3514, 2783, 2782, 2854, 2857, 2862, 2861, 2731
            ])
        )

        self.v.register_buffer(
            "lip_inside_ring_lower",
            torch.tensor([
                1572, 1573, 1860, 1862, 1830, 1835, 1852, 3497, 2941, 2933, 2930, 2945, 2943, 2709, 2708
            ])
        )

        self.v.register_buffer(
            "lip_outside_ring_upper",
            torch.tensor([
                1713, 1715, 1716, 1735, 1696, 1694, 1657, 3543, 2774, 2811, 2813, 2850, 2833, 2832, 2830
            ])
        )

        self.v.register_buffer(
            "lip_outside_ring_lower",
            torch.tensor([
                1576, 1577, 1773, 1774, 1795, 1802, 1865, 3503, 2948, 2905, 2898, 2881, 2880, 2713, 2712
            ])
        )

        self.v.register_buffer(
            "lip_inside_upper",
            torch.tensor([
                1588, 1589, 1590, 1591, 1594, 1595, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1724, 1725, 1739,
                1741, 1742, 1743, 1744, 1745, 1746, 1747, 2724, 2725, 2726, 2727, 2730, 2731, 2776, 2777, 2778, 2779,
                2780, 2781, 2782, 2783, 2841, 2842, 2854, 2856, 2857, 2858, 2859, 2860, 2861, 2862, 3514, 3547, 3549,
            ])
        )

        self.v.register_buffer(
            "lip_inside_lower",
            torch.tensor([
                1572, 1573, 1592, 1593, 1764, 1765, 1779, 1780, 1781, 1830, 1831, 1832, 1835, 1846, 1847, 1851, 1852,
                1854, 1860, 1861, 1862, 2708, 2709, 2728, 2729, 2872, 2873, 2886, 2887, 2888, 2930, 2931, 2932, 2933,
                2935, 2936, 2940, 2941, 2942, 2943, 2944, 2945, 3497, 3500, 3512,
            ])
        )

        self.v.register_buffer(
            "lip_inside",
            torch.tensor([
                1572, 1573, 1580, 1581, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1659, 1660, 1661, 1662, 1663,
                1664, 1665, 1666, 1667, 1668, 1718, 1719, 1722, 1724, 1725, 1728, 1739, 1740, 1741, 1742, 1743, 1744,
                1745, 1746, 1747, 1748, 1764, 1765, 1777, 1778, 1779, 1780, 1781, 1782, 1827, 1830, 1831, 1832, 1835,
                1836, 1846, 1847, 1851, 1852, 1854, 1860, 1861, 1862, 2708, 2709, 2716, 2717, 2724, 2725, 2726, 2727,
                2728, 2729, 2730, 2731, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2835, 2836, 2839,
                2841, 2842, 2843, 2854, 2855, 2856, 2857, 2858, 2859, 2860, 2861, 2862, 2863, 2872, 2873, 2884, 2885,
                2886, 2887, 2888, 2889, 2929, 2930, 2931, 2932, 2933, 2934, 2935, 2936, 2940, 2941, 2942, 2943, 2944,
                2945, 3497, 3500, 3512, 3513, 3514, 3533, 3547, 3549,
            ])
        )

        self.v.register_buffer(
            "neck_upper",
            torch.tensor([
                10, 11, 12, 13, 14, 15, 111, 112, 219, 220, 221, 222, 372, 373, 374, 375, 462, 463, 496, 497, 552, 553,
                558, 559, 563, 564, 649, 650, 736, 737, 784, 795, 1210, 1211, 1212, 1213, 1325, 1326, 1359, 1360, 1386,
                1726, 1727, 1759, 1790, 1886, 1898, 1901, 1931, 1932, 1933, 1934, 1940, 1941, 1948, 1949, 2036, 2115,
                2149, 2150, 2151, 2162, 2218, 2219, 2251, 2254, 2483, 2484, 2531, 2870, 2893, 2964, 2976, 2979, 3012,
                3013, 3142, 3174, 3184, 3185, 3186, 3187, 3188, 3189, 3193, 3194, 3196, 3199, 3200, 3202, 3203, 3206,
                3209, 3281, 3282, 3286, 3291, 3292, 3296, 3297, 3299, 3302, 3303, 3305, 3306, 3309, 3312, 3376, 3441,
                3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3452, 3453, 3454, 3455, 3456, 3457, 3458, 3459, 3460,
                3461, 3462, 3463, 3494, 3496, 3544, 3562, 3673, 3676, 3677, 3678, 3679, 3680, 3681, 3685, 3695, 3697,
                3698, 3701, 3703, 3707, 3709, 3713,
            ])
        )

        self.v.register_buffer(
            "neck_lower",
            torch.tensor([
                3188, 3189, 3190, 3191, 3192, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204,
                3205, 3206, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3215, 3220, 3222, 3223, 3231, 3232, 3233,
                3234, 3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3250, 3251, 3253,
                3254, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3275, 3276, 3277, 3278, 3281, 3282, 3283, 3286,
                3288, 3290, 3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3305,
                3306, 3307, 3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3323, 3332, 3333, 3334,
                3335, 3336, 3337, 3338, 3339, 3340, 3341, 3342, 3343, 3344, 3345, 3346, 3347, 3348, 3349, 3350, 3352,
                3353, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3369, 3376, 3378,
            ])
        )

        # the bottomline of "neck"
        self.v.register_buffer(
            "neck_base",
            torch.tensor([
                3231, 3232, 3237, 3238, 3240, 3242, 3243, 3251, 3263, 3290, 3332, 3333, 3338, 3339, 3341, 3343, 3344,
                3350, 3362,  # 4-th ring from bottom (drop 7 front verts)
            ])
        )

        # As a subset of "boundary", "bottomline" only contains vertices on the edge
        self.v.register_buffer(
            "bottomline",
            torch.tensor([
                3218, 3219, 3226, 3272, 3273, 3229, 3228, 3261, 3260, 3248, 3359, 3360, 3329, 3330, 3372, 3371, 3327,
                3322, 3321, 3355, 3354, 3356, 3357, 3379, 3285, 3289, 3258, 3257, 3255, 3256
            ])
        )

        self.v.register_buffer(
            "left_iris",
            torch.tensor([
                3931, 3932, 3933, 3935, 3936, 3937, 3939, 3940, 3941, 3943, 3944, 3945, 3947, 3948, 3949, 3951, 3952,
                3953, 3955, 3956, 3957, 3959, 3960, 3961, 3963, 3964, 3965, 3967, 3968, 3969, 3971, 3972, 3973, 3975,
                3976, 3977, 3979, 3980, 3981, 3983, 3984, 3985, 3987, 3988, 3989, 3991, 3992, 3993, 3995, 3996, 3997,
                3999, 4000, 4001, 4003, 4004, 4005, 4007, 4008, 4009, 4011, 4012, 4013, 4015, 4016, 4017, 4019, 4020,
                4021, 4023, 4024, 4025, 4027, 4028, 4029, 4031, 4032, 4033, 4035, 4036, 4037, 4039, 4040, 4041, 4043,
                4044, 4045, 4047, 4048, 4049, 4051, 4052, 4053, 4054, 4056, 4057, 4058,
            ])
        )

        self.v.register_buffer(
            "right_iris",
            torch.tensor([
                4477, 4478, 4479, 4481, 4482, 4483, 4485, 4486, 4487, 4489, 4490, 4491, 4493, 4494, 4495, 4497, 4498,
                4499, 4501, 4502, 4503, 4505, 4506, 4507, 4509, 4510, 4511, 4513, 4514, 4515, 4517, 4518, 4519, 4521,
                4522, 4523, 4525, 4526, 4527, 4529, 4530, 4531, 4533, 4534, 4535, 4537, 4538, 4539, 4541, 4542, 4543,
                4545, 4546, 4547, 4549, 4550, 4551, 4553, 4554, 4555, 4557, 4558, 4559, 4561, 4562, 4563, 4565, 4566,
                4567, 4569, 4570, 4571, 4573, 4574, 4575, 4577, 4578, 4579, 4581, 4582, 4583, 4585, 4586, 4587, 4589,
                4590, 4591, 4593, 4594, 4595, 4597, 4598, 4599, 4600, 4602, 4603, 4604,
            ])
        )

        self.v.register_buffer(
            "left_eyelid",  # 30 vertices
            torch.tensor([
                807, 808, 809, 814, 815, 816, 821, 822, 823, 824, 825, 826, 827, 828, 829, 841, 842, 848, 864, 865, 877,
                878, 879, 880, 881, 882, 883, 884, 885, 896, 897, 903, 904, 905, 922, 923, 924, 926, 945, 946, 947, 948,
                949, 950, 951, 952, 953, 954, 955, 958, 959, 991, 992, 993, 994, 995, 999, 1000, 1003, 1006, 1008, 1011,
                1023, 1033, 1034, 1045, 1046, 1059, 1060, 1061, 1062, 1093, 1096, 1101, 1108, 1113, 1114, 1115, 1125,
                1126, 1132, 1134, 1135, 1142, 1143, 1144, 1146, 1147, 1150, 1151, 1152, 1153, 1154, 1170, 1175, 1182,
                1183, 1194, 1195, 1200, 1201, 1202, 1216, 1217, 1218, 1224, 1227, 1230, 1232, 1233, 1243, 1244, 1283,
                1289, 1292, 1293, 1294, 1320, 1329, 1331, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345,
                1352, 1353, 1354, 1355, 1356, 1357, 1358, 1361, 3827, 3832, 3833, 3835, 3853, 3855, 3856, 3861,
            ])
        )

        self.v.register_buffer(
            "right_eyelid",  # 30 vertices
            torch.tensor([
                2264, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2282, 2283,
                2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2297, 2298, 2299, 2303, 2304, 2305,
                2312, 2313, 2314, 2315, 2323, 2324, 2325, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335,
                2355, 2356, 2357, 2358, 2359, 2360, 2361, 2364, 2365, 2367, 2369, 2381, 2382, 2383, 2386, 2387, 2388,
                2389, 2390, 2391, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2411, 2412, 2416, 2417, 2418, 2419, 2420,
                2421, 2422, 2423, 2424, 2425, 2426, 2427, 2428, 2436, 2437, 2440, 2441, 2446, 2447, 2448, 2449, 2450,
                2451, 2452, 2453, 2454, 2457, 2460, 2461, 2462, 2465, 2466, 2467, 2470, 2471, 2472, 2473, 2478, 2485,
                2486, 2487, 2488, 2489, 2490, 2491, 2492, 2493, 2494, 2495, 2496, 2503, 2504, 2505, 2506, 2507, 2508,
                2509, 2510, 3619, 3631, 3632, 3638, 3687, 3689, 3690, 3700,
            ])
        )

        self.v.register_buffer(
            "lips_tight",  # 30 vertices
            torch.tensor([
                1572, 1573, 1578, 1580, 1581, 1582, 1583, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1659, 1660,
                1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1718, 1719, 1720, 1721, 1722, 1723, 1724,
                1725, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744,
                1745, 1746, 1747, 1748, 1750, 1751, 1758, 1764, 1765, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780,
                1781, 1782, 1787, 1788, 1789, 1791, 1792, 1793, 1794, 1795, 1802, 1803, 1804, 1826, 1827, 1830, 1831,
                1832, 1835, 1836, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1854, 1860, 1861, 1862, 1865, 2708, 2709,
                2714, 2716, 2717, 2718, 2719, 2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2776, 2777, 2778, 2779,
                2780, 2781, 2782, 2783, 2784, 2785, 2786, 2787, 2835, 2836, 2837, 2838, 2839, 2840, 2841, 2842, 2843,
                2844, 2845, 2846, 2847, 2848, 2849, 2851, 2852, 2853, 2854, 2855, 2856, 2857, 2858, 2859, 2860, 2861,
                2862, 2863, 2865, 2866, 2869, 2872, 2873, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888, 2889,
                2890, 2891, 2892, 2894, 2895, 2896, 2897, 2898, 2905, 2906, 2907, 2928, 2929, 2930, 2931, 2932, 2933,
                2934, 2935, 2936, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2948, 3497, 3500, 3503, 3504,
                3506, 3509, 3512, 3513, 3514, 3531, 3533, 3546, 3547, 3549,
            ])
        )

        self.v.register_buffer(
            "left_half",
            torch.tensor([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
                88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 113, 114,
                115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
                136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156,
                157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177,
                178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198,
                199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 223,
                224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244,
                245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265,
                266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286,
                287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307,
                308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328,
                329, 330, 331, 332, 333, 334, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353,
                354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374,
                375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395,
                396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416,
                417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437,
                438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458,
                459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,
                480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500,
                501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521,
                530, 531, 532, 533, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 558,
                559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579,
                580, 581, 582, 583, 588, 589, 590, 591, 592, 593, 594, 603, 604, 605, 622, 623, 624, 625, 626, 627, 628,
                629, 630, 631, 632, 633, 638, 639, 644, 645, 646, 647, 648, 649, 650, 667, 668, 669, 670, 671, 672, 673,
                674, 679, 680, 681, 682, 683, 688, 691, 692, 693, 694, 695, 696, 697, 702, 703, 704, 705, 706, 707, 708,
                709, 712, 713, 714, 715, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738,
                739, 740, 745, 746, 747, 748, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767,
                768, 769, 770, 771, 772, 773, 774, 775, 783, 784, 785, 786, 795, 796, 797, 798, 799, 802, 803, 804, 805,
                806, 807, 808, 809, 814, 815, 816, 821, 822, 823, 824, 825, 826, 827, 828, 829, 837, 838, 840, 841, 842,
                846, 847, 848, 864, 865, 877, 878, 879, 880, 881, 882, 883, 884, 885, 896, 897, 898, 899, 902, 903, 904,
                905, 906, 907, 908, 909, 918, 919, 922, 923, 924, 926, 927, 928, 929, 939, 942, 943, 944, 945, 946, 947,
                948, 949, 950, 951, 952, 953, 954, 955, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970,
                971, 972, 977, 978, 979, 980, 985, 986, 991, 992, 993, 994, 995, 999, 1000, 1001, 1002, 1003, 1006,
                1007, 1008, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1033,
                1034, 1043, 1044, 1045, 1046, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1068, 1075, 1085, 1086, 1087,
                1088, 1092, 1093, 1096, 1101, 1108, 1113, 1114, 1115, 1116, 1117, 1125, 1126, 1127, 1128, 1129, 1132,
                1134, 1135, 1142, 1143, 1144, 1146, 1147, 1150, 1151, 1152, 1153, 1154, 1155, 1161, 1162, 1163, 1164,
                1168, 1169, 1170, 1175, 1176, 1181, 1182, 1183, 1184, 1189, 1190, 1193, 1194, 1195, 1200, 1201, 1202,
                1216, 1217, 1218, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1232, 1233, 1241, 1242, 1243, 1244, 1283,
                1284, 1287, 1289, 1292, 1293, 1294, 1298, 1299, 1308, 1309, 1320, 1321, 1322, 1323, 1324, 1325, 1326,
                1329, 1331, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350,
                1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369,
                1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390,
                1391, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1410, 1411, 1412, 1413, 1414, 1415,
                1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432,
                1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449,
                1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466,
                1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483,
                1484, 1485, 1486, 1487, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501,
                1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518,
                1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535,
                1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552,
                1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569,
                1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586,
                1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603,
                1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1617, 1618, 1623, 1624, 1625, 1626, 1638, 1639,
                1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656,
                1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673,
                1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690,
                1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707,
                1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724,
                1725, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743,
                1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1756, 1757, 1758, 1759, 1763, 1764, 1765, 1766, 1767,
                1768, 1769, 1770, 1771, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1787, 1788, 1789,
                1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806,
                1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1823, 1824,
                1825, 1826, 1827, 1830, 1831, 1832, 1835, 1836, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1854, 1860,
                1861, 1862, 1863, 1864, 1865, 1866, 1867, 1868, 1869, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878,
                1879, 1880, 1881, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899,
                1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1914, 1915, 1917, 1918, 1919,
                1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1938, 1939, 1942, 1943, 1944, 1945, 1946, 1947,
                1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1964, 1965, 1966, 1967, 1968,
                1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1986, 1987, 1988, 1989,
                1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2004, 2009, 2010, 2011, 2012, 2021, 2022,
                2023, 2024, 2025, 2026, 2029, 2030, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043,
                2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060,
                2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077,
                2078, 2079, 2080, 2081, 2082, 2083, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 2102,
                2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119,
                2120, 2121, 2122, 2125, 2126, 2127, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2148,
                2151, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 2161, 2162, 2163, 2164, 2169, 2170, 2171,
                2172, 2173, 2174, 2175, 3186, 3187, 3188, 3189, 3190, 3191, 3192, 3193, 3194, 3195, 3196, 3197, 3198,
                3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3215,
                3216, 3217, 3218, 3219, 3220, 3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232,
                3233, 3234, 3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3248, 3249,
                3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266,
                3267, 3268, 3269, 3270, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278, 3279, 3280, 3281, 3282, 3283,
                3284, 3285, 3286, 3287, 3288, 3289, 3290, 3399, 3400, 3401, 3404, 3414, 3442, 3457, 3459, 3461, 3463,
                3487, 3494, 3495, 3496, 3497, 3498, 3499, 3500, 3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508, 3509,
                3510, 3511, 3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3525, 3526,
                3527, 3528, 3529, 3530, 3531, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3539, 3540, 3541, 3542, 3543,
                3544, 3545, 3546, 3547, 3548, 3549, 3550, 3551, 3552, 3553, 3554, 3555, 3556, 3557, 3558, 3559, 3560,
                3561, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3569, 3570, 3571, 3572, 3573, 3574, 3575, 3576, 3577,
                3578, 3579, 3580, 3581, 3582, 3583, 3584, 3587, 3588, 3593, 3594, 3595, 3596, 3598, 3599, 3600, 3601,
                3604, 3605, 3611, 3614, 3623, 3624, 3625, 3626, 3628, 3629, 3630, 3634, 3635, 3636, 3637, 3643, 3644,
                3646, 3649, 3650, 3652, 3653, 3654, 3655, 3656, 3658, 3659, 3660, 3662, 3663, 3664, 3665, 3666, 3667,
                3668, 3670, 3671, 3672, 3673, 3676, 3677, 3678, 3679, 3680, 3681, 3685, 3691, 3693, 3695, 3697, 3698,
                3701, 3703, 3704, 3707, 3709, 3713, 3714, 3715, 3716, 3717, 3722, 3724, 3725, 3726, 3727, 3728, 3730,
                3734, 3737, 3738, 3739, 3740, 3742, 3745, 3752, 3753, 3754, 3756, 3757, 3760, 3761, 3762, 3769, 3771,
                3772, 3785, 3786, 3790, 3801, 3807, 3808, 3809, 3810, 3811, 3812, 3813, 3814, 3815, 3816, 3817, 3818,
                3819, 3820, 3821, 3822, 3823, 3824, 3825, 3826, 3827, 3828, 3829, 3830, 3831, 3832, 3833, 3834, 3835,
                3836, 3837, 3838, 3839, 3840, 3841, 3842, 3843, 3844, 3845, 3846, 3847, 3848, 3849, 3850, 3851, 3852,
                3853, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3861, 3862, 3863, 3864, 3865, 3866, 3867, 3868, 3869,
                3870, 3871, 3872, 3873, 3874, 3875, 3876, 3877, 3878, 3879, 3880, 3881, 3882, 3883, 3884, 3885, 3886,
                3887, 3888, 3889, 3890, 3891, 3892, 3893, 3894, 3895, 3896, 3897, 3898, 3899, 3900, 3901, 3902, 3903,
                3904, 3905, 3906, 3907, 3908, 3909, 3910, 3911, 3912, 3913, 3914, 3915, 3916, 3917, 3918, 3919, 3920,
                3921, 3922, 3923, 3924, 3925, 3926, 3927, 3928, 3929, 3931, 3932, 3933, 3934, 3935, 3936, 3937, 3938,
                3939, 3940, 3941, 3942, 3943, 3944, 3945, 3946, 3947, 3948, 3949, 3950, 3951, 3952, 3953, 3954, 3955,
                3956, 3957, 3958, 3959, 3960, 3961, 3962, 3963, 3964, 3965, 3966, 3967, 3968, 3969, 3970, 3971, 3972,
                3973, 3974, 3975, 3976, 3977, 3978, 3979, 3980, 3981, 3982, 3983, 3984, 3985, 3986, 3987, 3988, 3989,
                3990, 3991, 3992, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004, 4005, 4006,
                4007, 4008, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018, 4019, 4020, 4021, 4022, 4023,
                4024, 4025, 4026, 4027, 4028, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4036, 4037, 4038, 4039, 4040,
                4041, 4042, 4043, 4044, 4045, 4046, 4047, 4048, 4049, 4050, 4051, 4052, 4053, 4054, 4055, 4056, 4057,
                4058, 4059, 4060, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4068, 4069, 4070, 4071, 4072, 4073, 4074,
                4075, 4076, 4077, 4078, 4079, 4080, 4081, 4082, 4083, 4084, 4085, 4086, 4087, 4088, 4089, 4090, 4091,
                4092, 4093, 4094, 4095, 4096, 4097, 4098, 4099, 4100, 4101, 4102, 4103, 4104, 4105, 4106, 4107, 4108,
                4109, 4110, 4111, 4112, 4113, 4114, 4115, 4116, 4117, 4118, 4119, 4120, 4121, 4122, 4123, 4124, 4125,
                4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137, 4138, 4139, 4140, 4141, 4142,
                4143, 4144, 4145, 4146, 4147, 4148, 4149, 4150, 4151, 4152, 4153, 4154, 4155, 4156, 4157, 4158, 4159,
                4160, 4161, 4162, 4163, 4164, 4165, 4166, 4167, 4168, 4169, 4170, 4171, 4172, 4173, 4174, 4175, 4176,
                4177, 4178, 4179, 4180, 4181, 4182, 4183, 4184, 4185, 4186, 4187, 4188, 4189, 4190, 4191, 4192, 4193,
                4194, 4195, 4196, 4197, 4198, 4199, 4200, 4201, 4202, 4203, 4204, 4205, 4206, 4207, 4208, 4209, 4210,
                4211, 4212, 4213, 4214, 4215, 4216, 4217, 4218, 4219, 4220, 4221, 4222, 4223, 4224, 4225, 4226, 4227,
                4228, 4229, 4230, 4231, 4232, 4233, 4234, 4235, 4236, 4237, 4238, 4239, 4240, 4241, 4242, 4243, 4244,
                4245, 4246, 4247, 4248, 4249, 4250, 4251, 4252, 4253, 4254, 4255, 4256, 4257, 4258, 4259, 4260, 4261,
                4262, 4263, 4264, 4265, 4266, 4267, 4268, 4269, 4270, 4271, 4272, 4273, 4274, 4275, 4276, 4277, 4278,
                4279, 4280, 4281, 4282, 4283, 4284, 4285, 4286, 4287, 4288, 4289, 4290, 4291, 4292, 4293, 4294, 4295,
                4296, 4297, 4298, 4299, 4300, 4301, 4302, 4303, 4304, 4305, 4306, 4307, 4308, 4309, 4310, 4311, 4312,
                4313, 4314, 4315, 4316, 4317, 4318, 4319, 4320, 4321, 4322, 4323, 4324, 4325, 4326, 4327, 4328, 4329,
                4330, 4331, 4332, 4333, 4334, 4335, 4336, 4337, 4338, 4339, 4340, 4341, 4342, 4343, 4344, 4345, 4346,
                4347, 4348, 4349, 4350, 4351, 4352, 4353, 4354, 4355, 4356, 4357, 4358, 4359, 4360, 4361, 4362, 4363,
                4364, 4365, 4366, 4367, 4368, 4369, 4370, 4371, 4372, 4373, 4374, 4375, 4376, 4377, 4378, 4379, 4380,
                4381, 4382, 4383, 4384, 4385, 4386, 4387, 4388, 4389, 4390, 4391, 4392, 4393, 4394, 4395, 4396, 4397,
                4398, 4399, 4400, 4401, 4402, 4403, 4404, 4405, 4406, 4407, 4408, 4409, 4410, 4411, 4412, 4413, 4414,
                4415, 4416, 4417, 4418, 4419, 4420, 4421, 4422, 4423, 4424, 4425, 4426, 4427, 4428, 4429, 4430, 4431,
                4432, 4433, 4434, 4435, 4436, 4437, 4438, 4439, 4440, 4441, 4442, 4443, 4444, 4445, 4446, 4447, 4448,
                4449, 4450, 4451, 4452, 4453, 4454, 4455, 4456, 4457, 4458, 4459, 4460, 4461, 4462, 4463, 4464, 4465,
                4466, 4467, 4468, 4469, 4470, 4471, 4472, 4473, 4474, 4475, 4476,
            ])
        )

        self.v.register_buffer(
            "right_half",
            torch.tensor([
                19, 20, 21, 22, 23, 24, 25, 26, 109, 110, 111, 112, 219, 220, 221, 222, 335, 336, 337, 338, 522, 523,
                524, 525, 526, 527, 528, 529, 534, 535, 536, 537, 554, 555, 556, 557, 584, 585, 586, 587, 595, 596, 597,
                598, 599, 600, 601, 602, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621,
                634, 635, 636, 637, 640, 641, 642, 643, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663,
                664, 665, 666, 675, 676, 677, 678, 684, 685, 686, 687, 689, 690, 698, 699, 700, 701, 710, 711, 716, 717,
                718, 719, 720, 721, 722, 741, 742, 743, 744, 749, 750, 751, 752, 776, 777, 778, 779, 780, 781, 782, 787,
                788, 789, 790, 791, 792, 793, 794, 800, 801, 810, 811, 812, 813, 817, 818, 819, 820, 830, 831, 832, 833,
                834, 835, 836, 839, 843, 844, 845, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862,
                863, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 886, 887, 888, 889, 890, 891, 892, 893, 894,
                895, 900, 901, 910, 911, 912, 913, 914, 915, 916, 917, 920, 921, 925, 930, 931, 932, 933, 934, 935, 936,
                937, 938, 940, 941, 956, 957, 973, 974, 975, 976, 981, 982, 983, 984, 987, 988, 989, 990, 996, 997, 998,
                1004, 1005, 1009, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1035, 1036, 1037, 1038, 1039,
                1040, 1041, 1042, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1066, 1067,
                1069, 1070, 1071, 1072, 1073, 1074, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1089, 1090,
                1091, 1094, 1095, 1097, 1098, 1099, 1100, 1102, 1103, 1104, 1105, 1106, 1107, 1109, 1110, 1111, 1112,
                1118, 1119, 1120, 1121, 1122, 1123, 1124, 1130, 1131, 1133, 1136, 1137, 1138, 1139, 1140, 1141, 1145,
                1148, 1149, 1156, 1157, 1158, 1159, 1160, 1165, 1166, 1167, 1171, 1172, 1173, 1174, 1177, 1178, 1179,
                1180, 1185, 1186, 1187, 1188, 1191, 1192, 1196, 1197, 1198, 1199, 1203, 1204, 1205, 1206, 1207, 1208,
                1209, 1210, 1211, 1212, 1213, 1214, 1215, 1219, 1220, 1221, 1222, 1223, 1231, 1234, 1235, 1236, 1237,
                1238, 1239, 1240, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258,
                1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275,
                1276, 1277, 1278, 1279, 1280, 1281, 1282, 1285, 1286, 1288, 1290, 1291, 1295, 1296, 1297, 1300, 1301,
                1302, 1303, 1304, 1305, 1306, 1307, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1327,
                1328, 1330, 1332, 1333, 1334, 1335, 1359, 1360, 1379, 1380, 1381, 1382, 1392, 1393, 1394, 1395, 1406,
                1407, 1408, 1409, 1488, 1613, 1614, 1615, 1616, 1619, 1620, 1621, 1622, 1627, 1628, 1629, 1630, 1631,
                1632, 1633, 1634, 1635, 1636, 1637, 1726, 1727, 1752, 1753, 1754, 1755, 1760, 1761, 1762, 1772, 1783,
                1784, 1785, 1786, 1822, 1828, 1829, 1833, 1834, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1845,
                1853, 1855, 1856, 1857, 1858, 1859, 1870, 1882, 1883, 1884, 1885, 1912, 1913, 1916, 1929, 1930, 1931,
                1932, 1933, 1934, 1935, 1936, 1937, 1940, 1941, 1960, 1961, 1962, 1963, 1982, 1983, 1984, 1985, 2000,
                2001, 2002, 2003, 2005, 2006, 2007, 2008, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2027, 2028,
                2031, 2032, 2036, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2123, 2124, 2128, 2129, 2130, 2131,
                2132, 2133, 2144, 2145, 2146, 2147, 2149, 2150, 2151, 2165, 2166, 2167, 2168, 2176, 2177, 2178, 2179,
                2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196,
                2197, 2198, 2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213,
                2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2226, 2227, 2228, 2229, 2230,
                2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247,
                2248, 2249, 2250, 2251, 2252, 2253, 2254, 2255, 2256, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2264,
                2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2279, 2280, 2281,
                2282, 2283, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2297, 2298,
                2299, 2300, 2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315,
                2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2324, 2325, 2326, 2327, 2328, 2329, 2330, 2331, 2332,
                2333, 2334, 2335, 2336, 2337, 2338, 2339, 2340, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348, 2349,
                2350, 2351, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2365, 2366,
                2367, 2368, 2369, 2370, 2371, 2372, 2373, 2374, 2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 2383,
                2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2392, 2393, 2394, 2395, 2396, 2397, 2398, 2399, 2400,
                2401, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2409, 2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417,
                2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 2426, 2427, 2428, 2429, 2430, 2431, 2432, 2433, 2434,
                2435, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2446, 2447, 2448, 2449, 2450, 2451,
                2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2465, 2466, 2467, 2468,
                2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480, 2481, 2482, 2483, 2484, 2485,
                2486, 2487, 2488, 2489, 2490, 2491, 2492, 2493, 2494, 2495, 2496, 2497, 2498, 2499, 2500, 2501, 2502,
                2503, 2504, 2505, 2506, 2507, 2508, 2509, 2510, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2518, 2519,
                2520, 2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2531, 2532, 2533, 2534, 2535, 2536,
                2537, 2538, 2539, 2540, 2541, 2542, 2543, 2544, 2545, 2546, 2547, 2548, 2549, 2550, 2551, 2552, 2553,
                2554, 2555, 2556, 2557, 2558, 2559, 2560, 2561, 2562, 2563, 2564, 2565, 2566, 2567, 2568, 2569, 2570,
                2571, 2572, 2573, 2574, 2575, 2576, 2577, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2585, 2586, 2587,
                2588, 2589, 2590, 2591, 2592, 2593, 2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 2602, 2603, 2604,
                2605, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621,
                2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638,
                2639, 2640, 2641, 2642, 2643, 2644, 2645, 2646, 2647, 2648, 2649, 2650, 2651, 2652, 2653, 2654, 2655,
                2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672,
                2673, 2674, 2675, 2676, 2677, 2678, 2679, 2680, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 2688, 2689,
                2690, 2691, 2692, 2693, 2694, 2695, 2696, 2697, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2706,
                2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2720, 2721, 2722, 2723,
                2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739, 2740,
                2741, 2742, 2743, 2744, 2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757,
                2758, 2759, 2760, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769, 2770, 2771, 2772, 2773, 2774,
                2775, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786, 2787, 2788, 2789, 2790, 2791,
                2792, 2793, 2794, 2795, 2796, 2797, 2798, 2799, 2800, 2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808,
                2809, 2810, 2811, 2812, 2813, 2814, 2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2825,
                2826, 2827, 2828, 2829, 2830, 2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2841, 2842,
                2843, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2856, 2857, 2858, 2859,
                2860, 2861, 2862, 2863, 2864, 2865, 2866, 2867, 2868, 2869, 2870, 2871, 2872, 2873, 2874, 2875, 2876,
                2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888, 2889, 2890, 2891, 2892, 2893,
                2894, 2895, 2896, 2897, 2898, 2899, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909, 2910,
                2911, 2912, 2913, 2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924, 2925, 2926, 2927,
                2928, 2929, 2930, 2931, 2932, 2933, 2934, 2935, 2936, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944,
                2945, 2946, 2947, 2948, 2949, 2950, 2951, 2952, 2953, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961,
                2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978,
                2979, 2980, 2981, 2982, 2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995,
                2996, 2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012,
                3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029,
                3030, 3031, 3032, 3033, 3034, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046,
                3047, 3048, 3049, 3050, 3051, 3052, 3053, 3054, 3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063,
                3064, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 3077, 3078, 3079, 3080,
                3081, 3082, 3083, 3084, 3085, 3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097,
                3098, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112, 3113, 3114,
                3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125, 3126, 3127, 3128, 3129, 3130, 3131,
                3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3142, 3143, 3144, 3145, 3146, 3147, 3148,
                3149, 3150, 3151, 3152, 3153, 3154, 3155, 3156, 3157, 3158, 3159, 3160, 3161, 3162, 3163, 3164, 3165,
                3166, 3167, 3168, 3169, 3170, 3171, 3172, 3173, 3174, 3175, 3176, 3177, 3178, 3179, 3180, 3181, 3182,
                3183, 3184, 3185, 3222, 3223, 3248, 3249, 3275, 3276, 3277, 3278, 3281, 3282, 3283, 3284, 3285, 3290,
                3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3305, 3306, 3307,
                3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320, 3321, 3322, 3323, 3324,
                3325, 3326, 3327, 3328, 3329, 3330, 3331, 3332, 3333, 3334, 3335, 3336, 3337, 3338, 3339, 3340, 3341,
                3342, 3343, 3344, 3345, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3354, 3355, 3356, 3357, 3358,
                3359, 3360, 3361, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3369, 3370, 3371, 3372, 3373, 3374, 3375,
                3376, 3377, 3378, 3379, 3380, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3392,
                3393, 3394, 3395, 3396, 3397, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408, 3409,
                3410, 3411, 3412, 3413, 3414, 3415, 3416, 3417, 3418, 3419, 3420, 3421, 3422, 3423, 3424, 3425, 3426,
                3427, 3428, 3429, 3430, 3431, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3443,
                3444, 3445, 3446, 3447, 3448, 3449, 3450, 3451, 3452, 3453, 3454, 3455, 3456, 3457, 3458, 3459, 3460,
                3461, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3469, 3470, 3471, 3472, 3473, 3474, 3475, 3476, 3477,
                3478, 3479, 3480, 3481, 3482, 3483, 3484, 3485, 3486, 3487, 3488, 3489, 3490, 3491, 3492, 3493, 3494,
                3495, 3496, 3497, 3498, 3499, 3500, 3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508, 3509, 3510, 3511,
                3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528,
                3529, 3530, 3531, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3539, 3540, 3541, 3542, 3543, 3544, 3545,
                3546, 3547, 3548, 3549, 3550, 3551, 3552, 3553, 3554, 3555, 3556, 3557, 3558, 3559, 3560, 3561, 3562,
                3563, 3564, 3565, 3566, 3567, 3568, 3569, 3570, 3571, 3572, 3573, 3574, 3575, 3585, 3586, 3589, 3590,
                3591, 3592, 3597, 3602, 3603, 3606, 3607, 3608, 3609, 3610, 3612, 3613, 3615, 3616, 3617, 3618, 3619,
                3620, 3621, 3622, 3627, 3631, 3632, 3633, 3638, 3639, 3640, 3641, 3642, 3645, 3647, 3648, 3651, 3657,
                3661, 3668, 3669, 3674, 3675, 3682, 3683, 3684, 3686, 3687, 3688, 3689, 3690, 3692, 3694, 3696, 3699,
                3700, 3702, 3704, 3705, 3706, 3708, 3710, 3711, 3712, 3718, 3719, 3720, 3721, 3723, 3729, 3731, 3732,
                3733, 3735, 3736, 3741, 3743, 3744, 3746, 3747, 3748, 3749, 3750, 3751, 3755, 3758, 3759, 3763, 3764,
                3765, 3766, 3767, 3768, 3770, 3773, 3774, 3775, 3776, 3777, 3778, 3779, 3780, 3781, 3782, 3783, 3784,
                3785, 3786, 3787, 3788, 3789, 3790, 3791, 3792, 3793, 3794, 3795, 3796, 3797, 3798, 3799, 3800, 3801,
                3802, 3803, 3804, 3805, 3806, 3930, 4477, 4478, 4479, 4480, 4481, 4482, 4483, 4484, 4485, 4486, 4487,
                4488, 4489, 4490, 4491, 4492, 4493, 4494, 4495, 4496, 4497, 4498, 4499, 4500, 4501, 4502, 4503, 4504,
                4505, 4506, 4507, 4508, 4509, 4510, 4511, 4512, 4513, 4514, 4515, 4516, 4517, 4518, 4519, 4520, 4521,
                4522, 4523, 4524, 4525, 4526, 4527, 4528, 4529, 4530, 4531, 4532, 4533, 4534, 4535, 4536, 4537, 4538,
                4539, 4540, 4541, 4542, 4543, 4544, 4545, 4546, 4547, 4548, 4549, 4550, 4551, 4552, 4553, 4554, 4555,
                4556, 4557, 4558, 4559, 4560, 4561, 4562, 4563, 4564, 4565, 4566, 4567, 4568, 4569, 4570, 4571, 4572,
                4573, 4574, 4575, 4576, 4577, 4578, 4579, 4580, 4581, 4582, 4583, 4584, 4585, 4586, 4587, 4588, 4589,
                4590, 4591, 4592, 4593, 4594, 4595, 4596, 4597, 4598, 4599, 4600, 4601, 4602, 4603, 4604, 4605, 4606,
                4607, 4608, 4609, 4610, 4611, 4612, 4613, 4614, 4615, 4616, 4617, 4618, 4619, 4620, 4621, 4622, 4623,
                4624, 4625, 4626, 4627, 4628, 4629, 4630, 4631, 4632, 4633, 4634, 4635, 4636, 4637, 4638, 4639, 4640,
                4641, 4642, 4643, 4644, 4645, 4646, 4647, 4648, 4649, 4650, 4651, 4652, 4653, 4654, 4655, 4656, 4657,
                4658, 4659, 4660, 4661, 4662, 4663, 4664, 4665, 4666, 4667, 4668, 4669, 4670, 4671, 4672, 4673, 4674,
                4675, 4676, 4677, 4678, 4679, 4680, 4681, 4682, 4683, 4684, 4685, 4686, 4687, 4688, 4689, 4690, 4691,
                4692, 4693, 4694, 4695, 4696, 4697, 4698, 4699, 4700, 4701, 4702, 4703, 4704, 4705, 4706, 4707, 4708,
                4709, 4710, 4711, 4712, 4713, 4714, 4715, 4716, 4717, 4718, 4719, 4720, 4721, 4722, 4723, 4724, 4725,
                4726, 4727, 4728, 4729, 4730, 4731, 4732, 4733, 4734, 4735, 4736, 4737, 4738, 4739, 4740, 4741, 4742,
                4743, 4744, 4745, 4746, 4747, 4748, 4749, 4750, 4751, 4752, 4753, 4754, 4755, 4756, 4757, 4758, 4759,
                4760, 4761, 4762, 4763, 4764, 4765, 4766, 4767, 4768, 4769, 4770, 4771, 4772, 4773, 4774, 4775, 4776,
                4777, 4778, 4779, 4780, 4781, 4782, 4783, 4784, 4785, 4786, 4787, 4788, 4789, 4790, 4791, 4792, 4793,
                4794, 4795, 4796, 4797, 4798, 4799, 4800, 4801, 4802, 4803, 4804, 4805, 4806, 4807, 4808, 4809, 4810,
                4811, 4812, 4813, 4814, 4815, 4816, 4817, 4818, 4819, 4820, 4821, 4822, 4823, 4824, 4825, 4826, 4827,
                4828, 4829, 4830, 4831, 4832, 4833, 4834, 4835, 4836, 4837, 4838, 4839, 4840, 4841, 4842, 4843, 4844,
                4845, 4846, 4847, 4848, 4849, 4850, 4851, 4852, 4853, 4854, 4855, 4856, 4857, 4858, 4859, 4860, 4861,
                4862, 4863, 4864, 4865, 4866, 4867, 4868, 4869, 4870, 4871, 4872, 4873, 4874, 4875, 4876, 4877, 4878,
                4879, 4880, 4881, 4882, 4883, 4884, 4885, 4886, 4887, 4888, 4889, 4890, 4891, 4892, 4893, 4894, 4895,
                4896, 4897, 4898, 4899, 4900, 4901, 4902, 4903, 4904, 4905, 4906, 4907, 4908, 4909, 4910, 4911, 4912,
                4913, 4914, 4915, 4916, 4917, 4918, 4919, 4920, 4921, 4922, 4923, 4924, 4925, 4926, 4927, 4928, 4929,
                4930, 4931, 4932, 4933, 4934, 4935, 4936, 4937, 4938, 4939, 4940, 4941, 4942, 4943, 4944, 4945, 4946,
                4947, 4948, 4949, 4950, 4951, 4952, 4953, 4954, 4955, 4956, 4957, 4958, 4959, 4960, 4961, 4962, 4963,
                4964, 4965, 4966, 4967, 4968, 4969, 4970, 4971, 4972, 4973, 4974, 4975, 4976, 4977, 4978, 4979, 4980,
                4981, 4982, 4983, 4984, 4985, 4986, 4987, 4988, 4989, 4990, 4991, 4992, 4993, 4994, 4995, 4996, 4997,
                4998, 4999, 5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010, 5011, 5012, 5013, 5014,
                5015, 5016, 5017, 5018, 5019, 5020, 5021, 5022
            ])
        )

        # remove the intersection with neck from scalp and get the region for hair
        face_and_neck = torch.cat([self.v.face, self.v.neck]).unique()
        # get the intersection between scalp and face_and_neck
        uniques, counts = torch.cat([self.v.scalp, face_and_neck]).unique(return_counts=True)
        intersection = uniques[counts == 2]
        uniques, counts = torch.cat([self.v.scalp, intersection]).unique(return_counts=True)
        hair = uniques[counts == 1]
        self.v.register_buffer("hair", hair)

        # unions
        self.v.register_buffer("ears", torch.cat([self.v.right_ear, self.v.left_ear]))
        self.v.register_buffer("eyeballs", torch.cat([self.v.right_eyeball, self.v.left_eyeball]))
        self.v.register_buffer("irises", torch.cat([self.v.right_iris, self.v.left_iris]))
        self.v.register_buffer("left_eye", torch.cat([self.v.left_eye_region, self.v.left_eyeball]))
        self.v.register_buffer("right_eye", torch.cat([self.v.right_eye_region, self.v.right_eyeball]))
        self.v.register_buffer("eyelids", torch.cat([self.v.left_eyelid, self.v.right_eyelid]))
        self.v.register_buffer("lip_inside_ring", torch.cat(
            [self.v.lip_inside_ring_upper, self.v.lip_inside_ring_lower, torch.tensor([1594, 2730])]))

        # remove the intersection with irises from eyeballs and get the region for scleras
        uniques, counts = torch.cat([self.v.eyeballs, self.v.irises]).unique(return_counts=True)
        intersection = uniques[counts == 2]
        uniques, counts = torch.cat([self.v.eyeballs, intersection]).unique(return_counts=True)
        sclerae = uniques[counts == 1]
        self.v.register_buffer("sclerae", sclerae)

        # skin
        skin_except = ["eyeballs", "hair", "lips_tight", "boundary"]
        if self.num_verts == 5083:
            skin_except.append("teeth")
        skin = self.get_vid_except_region(skin_except)
        self.v.register_buffer("skin", skin)

    def construct_vid_table(self):
        self.vid_to_region = defaultdict(list)  # vertex id -> region name
        for region_name, v_mask in self.v:
            for v_id in v_mask:
                self.vid_to_region[v_id.item()].append(region_name)

    def process_face_mask(self, faces):

        face_masks = defaultdict(list)  # region name -> face id
        for f_id, f in enumerate(faces):
            counters = defaultdict(int)
            for v_id in f:
                for region_name in self.vid_to_region[v_id.item()]:
                    counters[region_name] += 1

            for region_name, count in counters.items():
                if count >= 3:  # create straight boundaries, with seams
                    # if count > 1:  # create zigzag boundaries, no seams
                    face_masks[region_name].append(f_id)

        self.f = BufferContainer()
        for region_name, f_mask in face_masks.items():
            self.f.register_buffer(region_name, torch.tensor(f_mask, dtype=torch.long))

    def process_face_clusters(self, face_clusters):
        """ Construct a lookup table from face id to cluster id.

            cluster #0: background
            cluster #1: foreground
            cluster #2: faces in face_clusters[0]
            cluster #3: faces in face_clusters[1]
            ...
        """
        fid2cid = torch.ones(self.num_faces + 1, dtype=torch.long)  # faces are always treated as foreground
        for cid, cluster in enumerate(face_clusters):
            try:
                fids = self.get_fid_by_region([cluster])
            except Exception as e:
                continue
            fid2cid[
                fids] = cid + 2  # reserve cluster #0 for the background and #1 for faces that do not belong to any cluster
        self.register_buffer("fid2cid", fid2cid)

    def process_vt_mask(self, faces, faces_t):
        vt_masks = defaultdict(list)  # region name -> vt id
        for f_id, (face, face_t) in enumerate(zip(faces, faces_t)):
            for v_id, vt_id in zip(face, face_t):
                for region_name in self.vid_to_region[v_id.item()]:
                    vt_masks[region_name].append(vt_id.item())

        self.vt = BufferContainer()
        for region_name, vt_mask in vt_masks.items():
            self.vt.register_buffer(region_name, torch.tensor(vt_mask, dtype=torch.long))

    def get_vid_by_region(self, regions, keep_order=False):
        """Get vertex indicies by regions"""
        if isinstance(regions, str):
            regions = [regions]
        if len(regions) > 0:
            vid = torch.cat([self.v.get_buffer(k) for k in regions])
            if keep_order:
                return vid
            else:
                return vid.unique()
        else:
            return torch.tensor([], dtype=torch.long)

    def get_vid_except_region(self, regions):
        if isinstance(regions, str):
            regions = [regions]
        if len(regions) > 0:
            indices = torch.cat([self.v.get_buffer(k) for k in regions]).unique()
        else:
            indices = torch.tensor([], dtype=torch.long)

        # get the vertex indicies that are not included by regions
        vert_idx = torch.arange(0, self.num_verts, device=indices.device)
        combined = torch.cat((indices, vert_idx))
        uniques, counts = combined.unique(return_counts=True)
        return uniques[counts == 1]

    def get_fid_by_region(self, regions):
        """Get face indicies by regions"""
        if isinstance(regions, str):
            regions = [regions]
        if len(regions) > 0:
            return torch.cat([self.f.get_buffer(k) for k in regions]).unique()
        else:
            return torch.tensor([], dtype=torch.long)

    def get_fid_except_region(self, regions):
        if isinstance(regions, str):
            regions = [regions]
        if len(regions) > 0:
            indices = torch.cat([self.f.get_buffer(k) for k in regions]).unique()
        else:
            indices = torch.tensor([], dtype=torch.long)

        # get the face indicies that are not included by regions
        face_idx = torch.arange(0, self.num_faces, device=indices.device)
        combined = torch.cat((indices, face_idx))
        uniques, counts = combined.unique(return_counts=True)
        return uniques[counts == 1]

    def get_fid_except_fids(self, fids):
        # get the face indicies that are not included
        face_idx = torch.arange(0, self.num_faces, device=fids.device)
        combined = torch.cat((fids, face_idx))
        uniques, counts = combined.unique(return_counts=True)
        return uniques[counts == 1]


if __name__ == '__main__':
    flame_model = FlameHead(shape_params=300, expr_params=100)
    print(1)
