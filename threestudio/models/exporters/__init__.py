import os

from . import base

if os.environ.get("THREESTUDIO_LAZY_IMPORT", "0") != "1":
    from . import mesh_exporter
