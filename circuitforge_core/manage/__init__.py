"""circuitforge_core.manage — cross-platform product process manager."""
from .config import ManageConfig, NativeService
from .docker_mode import DockerManager, docker_available
from .native_mode import NativeManager

__all__ = [
    "ManageConfig",
    "NativeService",
    "DockerManager",
    "docker_available",
    "NativeManager",
]
