import os


__modules__ = {}


def register(name):
    def decorator(cls):
        __modules__[name] = cls
        return cls

    return decorator


def find(name):
    return __modules__[name]


###  grammar sugar for logging utilities  ###
import logging

logger = logging.getLogger("pytorch_lightning")

if os.environ.get("THREESTUDIO_LAZY_IMPORT", "0") == "1":
    # Focused utilities do not need Lightning's distributed rank helpers, and
    # importing Lightning pulls in the entire training stack on Windows.
    debug = logger.debug
    info = logger.info

    def rank_zero_only(function):
        return function
else:
    from pytorch_lightning.utilities.rank_zero import (
        rank_zero_debug,
        rank_zero_info,
        rank_zero_only,
    )

    debug = rank_zero_debug
    info = rank_zero_info


@rank_zero_only
def warn(*args, **kwargs):
    logger.warn(*args, **kwargs)


# Legacy entrypoints rely on eager registry population.  Focused tools may
# opt into selective imports to avoid loading every system/CUDA extension.
if os.environ.get("THREESTUDIO_LAZY_IMPORT", "0") != "1":
    from . import data, models, systems
