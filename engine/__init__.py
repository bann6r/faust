"""Faust rendering engine public API.

The package root intentionally stays lightweight. GPU, asset-import, and EXR
modules are imported only when their exported objects are first requested.
"""

from importlib import import_module
from pathlib import Path

from .camera import Camera, look_at, perspective, rotation_y


_LAZY_EXPORTS = {
    "Material": (".material", "Material"),
    "PBRRenderer": (".rasterizer", "PBRRenderer"),
    "Mesh": (".mesh", "Mesh"),
    "MeshData": (".mesh", "MeshData"),
    "load_meshes": (".importer", "load_meshes"),
    "load_color_texture": (".texture", "load_color_texture"),
    "Skybox": (".skybox", "Skybox"),
    "load_exr_rgb": (".skybox", "load_exr_rgb"),
    "create_skybox_cube": (".skybox", "create_skybox_cube"),
    "Scene": (".scene", "Scene"),
    "RenderObject": (".scene", "RenderObject"),
    "DirectionalLight": (".scene", "DirectionalLight"),
    "Renderer": (".renderer", "Renderer"),
    "GPUInfo": (".gpu", "GPUInfo"),
    "inspect_context": (".gpu", "inspect_context"),
    "validate_context": (".gpu", "validate_context"),
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
    "Camera", "perspective", "look_at", "rotation_y", "find_project_asset",
    *_LAZY_EXPORTS.keys(),
]
