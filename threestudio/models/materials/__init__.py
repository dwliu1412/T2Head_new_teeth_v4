import os

from . import base

if os.environ.get("THREESTUDIO_LAZY_IMPORT", "0") != "1":
    from . import (
        diffuse_with_point_light_material,
        hybrid_rgb_latent_material,
        neural_radiance_material,
        no_material,
        pbr_material,
        sd_latent_adapter_material,
    )
