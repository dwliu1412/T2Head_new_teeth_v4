import os


if os.environ.get("THREESTUDIO_LAZY_IMPORT", "0") != "1":
    from . import (
        Head3DGSLKs,
        Head3DGSLKsFinetune,
    )
