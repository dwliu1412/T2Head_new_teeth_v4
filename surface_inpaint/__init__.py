"""Surface-correspondence feature propagation for diffusion U-Nets."""

from .attention import (
    SurfaceAttentionConfig,
    SurfaceAttentionController,
    SurfaceCorrespondenceAttnProcessor,
    install_surface_attention,
)
from .surface_memory_attention import (
    SurfaceMemoryAttnProcessor2_0,
    SurfaceMemoryConfig,
    SurfaceMemoryController,
    install_surface_memory_attention,
)
from .stability import (
    StabilityConfig,
    sanitize_uvd_covariances,
    stabilize_face_local_covariances,
    stabilize_uvd_covariances,
)

__all__ = [
    "SurfaceAttentionConfig",
    "SurfaceAttentionController",
    "SurfaceCorrespondenceAttnProcessor",
    "install_surface_attention",
    "SurfaceMemoryAttnProcessor2_0",
    "SurfaceMemoryConfig",
    "SurfaceMemoryController",
    "install_surface_memory_attention",
    "StabilityConfig",
    "sanitize_uvd_covariances",
    "stabilize_face_local_covariances",
    "stabilize_uvd_covariances",
]
