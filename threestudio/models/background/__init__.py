import os

from . import base

if os.environ.get("THREESTUDIO_LAZY_IMPORT", "0") != "1":
    from . import (
        neural_environment_map_background,
        solid_color_background,
        textured_background,
    )
