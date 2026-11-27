"""Faust engine API.

CPU-side scene and asset types stay importable without loading a GPU runtime.
The Vulkan backend is imported lazily when Renderer is requested.
"""

from importlib import import_module
from pathlib import Path

from .camera import Camera, look_at, perspective, perspective_vulkan, rotation_y


_LAZY_EXPORTS = {
    "Material": (".material", "Material"),
    "MeshData": (".mesh", "MeshData"),
    "load_meshes": (".importer", "load_meshes"),
    "Scene": (".scene", "Scene"),
    "RenderObject": (".scene", "RenderObject"),
    "DirectionalLight": (".scene", "DirectionalLight"),
    "Renderer": (".renderer", "Renderer"),
    "VulkanRenderer": (".vulkan", "VulkanRenderer"),
    "GPUInfo": (".gpu", "GPUInfo"),
    "inspect_device_properties": (".gpu", "inspect_device_properties"),
}


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


def find_project_asset(*parts):
    """Find an asset relative to the project root or the current module."""
    here = Path(__file__).resolve().parent
    candidates = (here.joinpath(*parts), here.parent.joinpath(*parts))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


__all__ = [
    "Camera", "perspective", "perspective_vulkan", "look_at", "rotation_y",
    "find_project_asset", *_LAZY_EXPORTS.keys(),
]
