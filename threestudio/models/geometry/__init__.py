import os

from . import base

if os.environ.get("THREESTUDIO_LAZY_IMPORT", "0") != "1":
    from . import (
        custom_mesh,
        implicit_sdf,
        implicit_volume,
        tetrahedra_sdf_grid,
        volume_grid,
    )
