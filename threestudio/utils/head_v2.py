import cv2
import numpy as np

import torch
import torch.nn as nn

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    TexturesAtlas,
)

from threestudio.utils.mediapipe_utils import draw_landmarks_468
from threestudio.utils.mediapipe_utils_v2 import draw_landmarks_105
from flame_model.flame import (
    FLAME_MEDIAPIPE_468_PATH,
    FLAME_MEDIAPIPE_LMK_PATH,
    FlameHead,
)


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    ''' Calculates landmarks by barycentric interpolation

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks

        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    '''
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        batch_size, -1, 3)

    lmk_faces += torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(
        batch_size, -1, 3, 3)

    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks


def draw_openpose(all_lmks, H, W, eps=0.01):
    bs = all_lmks.shape[0]
    canvas = np.zeros(shape=(bs, H, W, 3), dtype=np.uint8)
    for i, lmks in enumerate(all_lmks):
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            # x = int(x * W)
            # y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas[i], (int(x), int(y)), 3, (255, 255, 255), thickness=-1)
    return canvas


class MeshRendererWithDepth(nn.Module):
    def __init__(
            self,
            rasterizer,
            shader=None,
    ):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world):
        fragments = self.rasterizer(meshes_world)
        output = fragments.zbuf
        if self.shader is not None:
            images = self.shader(fragments, meshes_world)
            output = (output, images)

        return output


class FlamePointswRandomExp:
    def __init__(
            self,
            device='cuda',
            batch_size=1,
            image_size=512,
            flame_scale=-10,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.batch_size = batch_size

        self.num_betas = 300
        self.num_expression = 100
        self.flame_scale = flame_scale
        self.image_size = image_size

        self.model = FlameHead(
            shape_params=self.num_betas,
            expr_params=self.num_expression,
            include_mask=True,
            add_teeth=True,
        ).to(self.device)

        self.center = 0
        self.scale = 0

        self.init_mesh()

        # initizalize raster
        self.raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, 3.0]])

        self.betas = torch.zeros([1, self.num_betas], dtype=torch.float32, device=self.device)

        # facial landmarks
        # 68-DECA
        # flame_lmk_embedding_path = "/home/zhenglin/Documents/DECA/data/landmark_embedding.npy"
        # lmk_embeddings = np.load(flame_lmk_embedding_path, allow_pickle=True, encoding="latin1")
        # lmk_embeddings = lmk_embeddings[()]
        # self.full_lmk_faces_idx = torch.from_numpy(lmk_embeddings['full_lmk_faces_idx']).to(torch.long).to(self.device)
        # self.full_lmk_bary_coords = torch.from_numpy(lmk_embeddings['full_lmk_bary_coords']).to(torch.float32).to(self.device)

        # 468-HeadSculpt
        self.lmk_faces_idx_mediapipe_468 = np.load(FLAME_MEDIAPIPE_468_PATH).astype(int)
        # Add: Left Eye 4597 and Right Eye 4051
        left_eye_index = 4597  # 4597
        right_eye_index = 4051  # 4051
        self.lmk_faces_idx_mediapipe_468 = np.append(self.lmk_faces_idx_mediapipe_468,
                                                     [left_eye_index, right_eye_index])

        # 105-EMOCA
        lmk_embeddings_mediapipe = np.load(FLAME_MEDIAPIPE_LMK_PATH, allow_pickle=True, encoding='latin1')
        self.lmk_faces_idx_mediapipe_105 = torch.tensor(lmk_embeddings_mediapipe['lmk_face_idx'].astype(np.int64),
                                                        dtype=torch.long).to(self.device)
        self.lmk_bary_coords_mediapipe_105 = torch.tensor(lmk_embeddings_mediapipe['lmk_b_coords'],
                                                          dtype=torch.float32).to(self.device)

    def _flame_forward(
            self,
            betas=None,
            expression=None,
            jaw_pose=None,
            leye_pose=None,
            reye_pose=None,
            neck_pose=None,
            global_orient=None,
            translation=None,
            return_landmarks=False,
    ):
        if betas is None:
            betas = self.betas

        B = betas.shape[0]

        if expression is None:
            expression = torch.zeros([B, self.num_expression], dtype=torch.float32, device=self.device)

        if global_orient is None:
            global_orient = torch.zeros([B, 3], dtype=torch.float32, device=self.device)

        if neck_pose is None:
            neck_pose = torch.zeros([B, 3], dtype=torch.float32, device=self.device)

        if jaw_pose is None:
            jaw_pose = torch.zeros([B, 3], dtype=torch.float32, device=self.device)

        if leye_pose is None:
            leye_pose = torch.zeros([B, 3], dtype=torch.float32, device=self.device)

        if reye_pose is None:
            reye_pose = torch.zeros([B, 3], dtype=torch.float32, device=self.device)

        if translation is None:
            translation = torch.zeros([B, 3], dtype=torch.float32, device=self.device)

        eyes = torch.cat([leye_pose, reye_pose], dim=-1)

        out = self.model(
            shape=betas,
            expr=expression,
            rotation=global_orient,
            neck=neck_pose,
            jaw=jaw_pose,
            eyes=eyes,
            translation=translation,
            zero_centered_at_root_node=False,
            return_landmarks=return_landmarks,
        )

        if return_landmarks:
            vertices, landmarks = out
            return vertices, landmarks

        return out

    def init_mesh(self):
        betas = torch.zeros([1, self.num_betas], dtype=torch.float32, device=self.device)
        expression = torch.zeros([1, self.num_expression], dtype=torch.float32, device=self.device)

        # output = self.model(betas=betas, expression=expression, return_verts=True)
        # vertices = output.vertices.squeeze()
        vertices = self._flame_forward(
            betas=betas,
            expression=expression,
            return_landmarks=False,
        ).squeeze(0)

        # rescale and recenter
        vmin = vertices.min(0)[0]
        vmax = vertices.max(0)[0]
        ori_center = (vmin + vmax) / 2
        ori_scale = 0.6 / (vmax - vmin).max()
        vertices = (vertices - ori_center) * ori_scale
        # coordinate system: opengl --> blender (switch y/z)
        vertices[:, [1, 2]] = vertices[:, [2, 1]]
        vertices *= 1.1 ** (-self.flame_scale)

        self.center = ori_center
        self.scale = ori_scale

    def get_camera(self, dist=0.6, elev=0, azim=0, at=((0, 0, 0),), up=((0, 1, 0),), fov=40):
        R, T = look_at_view_transform(dist, elev, azim, degrees=True, at=at, up=up)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=fov)
        return cameras

    def depth_postprocess(self, fragments):
        depthmap = fragments[..., 0]
        # depthmap: (bs, 512, 512)
        depth_min = torch.amin(depthmap, dim=[1, 2], keepdim=True)
        depth_max = torch.amax(depthmap, dim=[1, 2], keepdim=True)
        depth_images = (depthmap - depth_min) / (depth_max - depth_min + 1e-10)
        depth_images = depth_images.clip(0, 1).to(torch.float32)
        depth_images = depth_images[..., None].expand(-1, -1, -1, 3)
        return depth_images

    def camera_conversion(self, v):
        # v: [bs, 3]
        # threestudio -> FLAME
        # x, y, z -> -y, z, -x
        v_x, v_y, v_z = v[:, 0], v[:, 1], v[:, 2]
        temp_v = torch.stack([-v_y, v_z, -v_x])
        return temp_v.permute(1, 0)

    def render(self, cameras, mesh):
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self.raster_settings
        )
        render = MeshRendererWithDepth(rasterizer=rasterizer)

        fragments = render(mesh)
        depths = self.depth_postprocess(fragments)
        return depths

    def render_mesh(self, cameras, mesh, colors):
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self.raster_settings
        )
        shader = SoftPhongShader(
            device=self.device,
            cameras=cameras,
            lights=self.lights,

        )
        render = MeshRendererWithDepth(rasterizer=rasterizer, shader=shader)
        fragments, images = render(mesh)

        # bug
        # render = MeshRenderer(rasterizer, shader=shader)
        # images = render(mesh, materials={"diffuse_color": colors.unsqueeze(0)})
        # from pytorch3d.renderer.materials import Materials
        # materials = Materials(diffuse_color=colors)
        # images = render(mesh, materials=materials)
        images = self.image_postprocee(images)
        return images

    def image_postprocee(self, images):
        return images[..., :3]

    def _project_landmarks_to_image(
        self, lmk3d, cameras, gaussian_camera_convention=False
    ):
        """Project FLAME landmarks with an explicit screen-space convention.

        PyTorch3D and the Gaussian renderer use opposite horizontal NDC signs
        for this project's camera conversion, while their vertical signs agree
        only after the historical y flip.  Keeping this conversion explicit is
        important for asymmetric eyes/mouths: a centered frontal silhouette can
        look correct even when its anatomical left and right sides are swapped.
        """
        if len(lmk3d.shape) == 2:
            lmk3d = lmk3d.repeat(self.batch_size, 1, 1)
        proj_lmk = cameras.transform_points(lmk3d)[:, :, :2]
        if gaussian_camera_convention:
            # Match diff-gaussian-rasterization's ndc2Pix exactly:
            # ((ndc + 1) * size - 1) / 2.
            size = float(self.image_size)
            image_x = 0.5 * ((proj_lmk[..., 0] + 1.0) * size - 1.0)
            image_y = 0.5 * ((-proj_lmk[..., 1] + 1.0) * size - 1.0)
            return torch.stack((image_x, image_y), dim=-1)
        return 0.5 * (-proj_lmk + 1.0) * self.image_size

    def get_cond_lmk_openpose(
        self, joints, cameras, gaussian_camera_convention=False
    ):
        # lmk3d = vertices2landmarks(
        #     vertices.repeat(self.batch_size, 1, 1),
        #     faces,
        #     self.full_lmk_faces_idx.repeat(self.batch_size, 1),
        #     self.full_lmk_bary_coords.repeat(self.batch_size, 1, 1)
        # )
        lmk3d = joints[-68:]
        lmk3d = (lmk3d - self.center) * self.scale
        lmk3d *= 1.1 ** (-self.flame_scale)
        img_lmk = self._project_landmarks_to_image(
            lmk3d, cameras, gaussian_camera_convention
        )

        # Draw Pose
        imgs = draw_openpose(img_lmk.detach().cpu(), self.image_size, self.image_size)

        return imgs

    def get_cond_lmk_mediapipe(
        self, vertices, faces, cameras, gaussian_camera_convention=False
    ):
        lmk3d_105 = vertices2landmarks(
            vertices.repeat(self.batch_size, 1, 1),
            faces,
            self.lmk_faces_idx_mediapipe_105.repeat(self.batch_size, 1),
            self.lmk_bary_coords_mediapipe_105.repeat(self.batch_size, 1, 1)
        )

        lmk3d_468 = vertices[self.lmk_faces_idx_mediapipe_468]

        def proj_lmk3d(lmk3d):
            return self._project_landmarks_to_image(
                lmk3d, cameras, gaussian_camera_convention
            )

        img_lmk_105 = proj_lmk3d(lmk3d_105)
        img_lmk_468 = proj_lmk3d(lmk3d_468)

        # Draw 105 Mediapipe
        canvas = np.ones(
            (self.batch_size, self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )
        imgs = draw_landmarks_105(canvas, img_lmk_105.detach().cpu().numpy())

        # Draw face oval in 468 Mediapipe
        for idx, img in enumerate(imgs):
            imgs[idx] = draw_landmarks_468(img, img_lmk_468[idx].detach().cpu().numpy())

        # Draw 468 Mediapipe
        # imgs = draw_landmarks(canvas, img_lmk_468[0].long().detach().cpu().numpy())
        # imgs = np.array(imgs, dtype=np.int32)
        # imgs = draw_openpose(img_lmk.detach().cpu(), self.image_size, self.image_size)
        return imgs / 255.0

    def get_cond_depth(self, vertices, faces, cameras, mesh_vis=False, mesh_rgb=False):
        # verts_rgb = torch.ones(vertices.shape, device=self.device)
        # textures = TexturesVertex([verts_rgb for _ in range(self.batch_size)])

        # 渲染成灰色
        gray = 0.9  # 0~1 之间，1 更白，0.7 更灰
        colors = torch.full(
            (1, faces.shape[0], 1, 1, 3),
            fill_value=gray,
            device=self.device,
            dtype=torch.float32
        )
        textures = TexturesAtlas(atlas=colors)
        mesh = Meshes(
            verts=[vertices for _ in range(self.batch_size)],
            faces=[faces for _ in range(self.batch_size)],
            textures=textures
        )
        if mesh_vis:
            results = self.render_mesh(cameras, mesh, colors)
        else:
            results = self.render(cameras, mesh)
        return results

    def get_cond(self, dist=0.6, elev=0, azim=0, at=((0, 0, 0),), up=((0, 1, 0),), fov=40,
                 betas=None, expression=None, jaw_pose=None, leye_pose=None, reye_pose=None, neck_pose=None,
                 lmk=False, mediapipe=True, mesh_vis=False,
                 gaussian_camera_convention=False):
        if betas is None:
            betas = self.betas
        if expression is None:
            expression = torch.zeros([1, self.num_expression], device=self.device)
        # output = self.model(
        #     betas=betas,
        #     expression=expression,
        #     jaw_pose=jaw_pose,
        #     leye_pose=leye_pose,
        #     reye_pose=reye_pose,
        #     neck_pose=neck_pose,
        #     return_verts=True
        # )
        vertices, landmarks = self._flame_forward(
            betas=betas,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            neck_pose=neck_pose,
            return_landmarks=True,
        )
        vertices = vertices.squeeze(0)
        vertices = (vertices - self.center) * self.scale
        vertices *= 1.1 ** (-self.flame_scale)
        # joints = output.joints.detach().squeeze()
        joints = landmarks.detach().squeeze(0)
        # faces = torch.tensor(self.model.faces.astype(np.int32), dtype=torch.int32, device=self.device)
        faces = self.model.faces.to(device=self.device, dtype=torch.int64)
        # threestudio -> FLAME
        # R, T = look_at_view_transform(
        #     dist, elev, (azim - 90),
        #     degrees=True,
        #     at=self.camera_conversion(at),
        #     up=self.camera_conversion(up))
        # cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=fov)
        camera_azimuth = (90 - azim) if gaussian_camera_convention else (azim - 90)
        cameras = self.get_camera(
            dist, elev, camera_azimuth,
            self.camera_conversion(at),
            self.camera_conversion(up),
            fov
        )

        if lmk:
            if mediapipe:
                cond_lmks = self.get_cond_lmk_mediapipe(
                    vertices,
                    faces,
                    cameras,
                    gaussian_camera_convention=gaussian_camera_convention,
                )
            else:
                cond_lmks = self.get_cond_lmk_openpose(
                    joints,
                    cameras,
                    gaussian_camera_convention=gaussian_camera_convention,
                )
            # result = {
            #     'depths': cond_depths,
            #     'lmks': torch.from_numpy(cond_lmks)
            # }
            result = torch.from_numpy(cond_lmks)
        else:
            cond_depths = self.get_cond_depth(vertices, faces, cameras, mesh_vis)
            result = cond_depths

        return result

    def get_cond_normal_semantic(
            self, dist=0.6, elev=0, azim=0, at=((0, 0, 0),),
            up=((0, 1, 0),), fov=40, betas=None, expression=None,
            jaw_pose=None, leye_pose=None, reye_pose=None, neck_pose=None,
            gaussian_camera_convention=False):
        """Render AnimPortrait3D's four-channel ControlNet condition.

        The first three channels are FLAME vertex normals with a white
        background.  The last channel follows the checkpoint's training
        convention: sclera/eyeballs=1/3, teeth=1/2 and irises=1.  Keeping the
        semantic values continuous (rather than one-hot) is important because
        the released ControlNet has exactly four conditioning channels.

        Returns:
            condition: ``(B,H,W,4)`` normal + scalar semantic condition.
            mask: ``(B,H,W,1)`` rasterized FLAME foreground.
        """
        if betas is None:
            betas = self.betas
        if expression is None:
            expression = torch.zeros([1, self.num_expression], device=self.device)
        # output = self.model(
        #     betas=betas,
        #     expression=expression,
        #     jaw_pose=jaw_pose,
        #     leye_pose=leye_pose,
        #     reye_pose=reye_pose,
        #     neck_pose=neck_pose,
        #     return_verts=True
        # )
        vertices = self._flame_forward(
            betas=betas,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            neck_pose=neck_pose,
            return_landmarks=False,
        )
        vertices = vertices.squeeze(0)
        vertices = (vertices - self.center) * self.scale
        vertices *= 1.1 ** (-self.flame_scale)

        # joints = output.joints.detach().squeeze()

        # faces = torch.tensor(self.model.faces.astype(np.int32), dtype=torch.int32, device=self.device)
        faces = self.model.faces.to(device=self.device, dtype=torch.int64)

        camera_azimuth = (90 - azim) if gaussian_camera_convention else (azim - 90)
        cameras = self.get_camera(
            dist, elev, camera_azimuth,
            self.camera_conversion(at),
            self.camera_conversion(up),
            fov
        )
        verts_list = [vertices for _ in range(self.batch_size)]
        faces_list = [faces for _ in range(self.batch_size)]

        temp_mesh = Meshes(verts=verts_list, faces=faces_list)
        # verts_normals_packed 返回形状为 (Total_V, 3)
        verts_normals = temp_mesh.verts_normals_packed()

        # 法线范围 [-1, 1] -> 颜色范围 [0, 1]
        normal_colors = (verts_normals + 1.0) * 0.5
        num_verts_per_mesh = [v.shape[0] for v in verts_list]
        features_list = list(torch.split(normal_colors, num_verts_per_mesh, dim=0))

        textures = TexturesVertex(verts_features=features_list)

        #
        mesh = Meshes(
            verts=verts_list,
            faces=faces_list,
            textures=textures
        )
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )

        fragments = rasterizer(mesh)

        # Output shape: (B,H,W,3).
        normal_images = mesh.sample_textures(fragments).squeeze(-2)

        # fragments.pix_to_face 形状为 (B, H, W, K)
        pix_to_face = fragments.pix_to_face[..., 0]
        mask = (pix_to_face > -1).float().unsqueeze(-1)  # (B, H, W, 1)

        # The released AnimPortrait3D examples use white normal-map
        # backgrounds, while the scalar semantic background is zero.
        normal_images = normal_images * mask + (1.0 - mask)

        semantic_values = torch.zeros(
            vertices.shape[0], 1, dtype=vertices.dtype, device=vertices.device
        )
        vertex_regions = self.model.mask.v

        def assign_region(name, value):
            indices = getattr(vertex_regions, name, None)
            if indices is None:
                return
            indices = indices.to(device=vertices.device, dtype=torch.long)
            indices = indices[(indices >= 0) & (indices < vertices.shape[0])]
            if indices.numel():
                semantic_values[indices] = float(value)

        assign_region("eyeballs", 1.0 / 3.0)
        assign_region("teeth", 0.5)
        # Irises are a subset of eyeballs and intentionally override them.
        assign_region("irises", 1.0)
        semantic_mesh = Meshes(
            verts=verts_list,
            faces=faces_list,
            textures=TexturesVertex(
                verts_features=[semantic_values for _ in range(self.batch_size)]
            ),
        )
        semantic = semantic_mesh.sample_textures(fragments).squeeze(-2) * mask

        # PyTorch3D's raster X convention is opposite to the Gaussian
        # renderer's ndc2Pix convention.  This is the same explicit horizontal
        # conversion used by the landmark path above; the subject-specific
        # Stage-1 reflection is applied later by Stage2Avatar.
        normal_images = torch.flip(normal_images, [2])
        semantic = torch.flip(semantic, [2])
        mask = torch.flip(mask, [2])
        return torch.cat((normal_images, semantic), dim=-1), mask

    def get_cond_normal(self, *args, **kwargs):
        """Backward-compatible three-channel normal-map renderer."""

        condition, mask = self.get_cond_normal_semantic(*args, **kwargs)
        return condition[..., :3], mask
