import os


if os.environ.get("THREESTUDIO_LAZY_IMPORT", "0") != "1":
    from . import (
        background,
        exporters,
        geometry,
        guidance,
        materials,
        prompt_processors,
        renderers,
    )
