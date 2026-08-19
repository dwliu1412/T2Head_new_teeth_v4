import torch

from gaussiansplatting.scene.gaussian_flame_face import GaussianFlameUVModel


class FakeCloneDensifier:
    DEFAULT_MAX_GAUSSIANS = 500_000

    def __init__(self):
        self.device = torch.device("cpu")
        self.percent_dense = 0.01
        self.num_gs = 2
        self._uv = torch.zeros(2, 2)
        self._d = torch.zeros(2, 1)
        self._features_dc = torch.zeros(2, 1, 3)
        self._features_rest = torch.zeros(2, 0, 3)
        self._opacity = torch.zeros(2, 1)
        self._scaling = torch.zeros(2, 3)
        self._rotation = torch.zeros(2, 4)
        self._rotation[:, 0] = 1.0
        self._face_idx = torch.zeros(2, dtype=torch.long)
        self.xyz_gradient_accum = torch.tensor([[1.0], [0.0]])
        self.denom = torch.ones(2, 1)
        # The first source splat would have been removed by the old
        # prune-before-densify order.
        self.max_radii2D = torch.tensor([100.0, 0.0])

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def get_world_scale_max_approx(self):
        return torch.full((self.num_gs,), 0.001)

    _densification_gradients = GaussianFlameUVModel._densification_gradients
    _limit_to_budget = GaussianFlameUVModel._limit_to_budget

    def densification_postfix(
        self,
        uv,
        d,
        features_dc,
        features_rest,
        opacity,
        scaling,
        rotation,
        face_idx,
    ):
        self._uv = torch.cat((self._uv, uv))
        self._d = torch.cat((self._d, d))
        self._features_dc = torch.cat((self._features_dc, features_dc))
        self._features_rest = torch.cat(
            (self._features_rest, features_rest)
        )
        self._opacity = torch.cat((self._opacity, opacity))
        self._scaling = torch.cat((self._scaling, scaling))
        self._rotation = torch.cat((self._rotation, rotation))
        self._face_idx = torch.cat((self._face_idx, face_idx))
        self.num_gs = self._uv.shape[0]
        self.xyz_gradient_accum = torch.zeros(self.num_gs, 1)
        self.denom = torch.zeros(self.num_gs, 1)
        self.max_radii2D = torch.zeros(self.num_gs)

    def prune_points(self, mask):
        keep = ~mask
        for name in (
            "_uv",
            "_d",
            "_features_dc",
            "_features_rest",
            "_opacity",
            "_scaling",
            "_rotation",
            "_face_idx",
            "xyz_gradient_accum",
            "denom",
            "max_radii2D",
        ):
            setattr(self, name, getattr(self, name)[keep])
        self.num_gs = int(keep.sum().item())


def test_densification_resets_historical_screen_radii_before_pruning():
    model = FakeCloneDensifier()

    stats = GaussianFlameUVModel.densify_and_prune(
        model,
        max_grad=0.5,
        min_opacity=0.005,
        extent=1.0,
        max_screen_size=20.0,
        max_gaussians=3,
    )

    assert stats == {"pruned": 0, "cloned": 1, "split": 0, "after": 3}
    assert model.num_gs == 3
    assert torch.equal(model.max_radii2D, torch.zeros(3))
