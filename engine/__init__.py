"""Faust engine API.

The established ModernGL scene renderer remains the default until the Vulkan
backend reaches feature parity. VulkanRenderer is available explicitly.
"""
from importlib import import_module
from pathlib import Path
from .camera import Camera, look_at, perspective, perspective_vulkan, rotation_y

_LAZY_EXPORTS = {
    "Material": (".material", "Material"),
    "PBRRenderer": (".rasterizer", "PBRRenderer"),
    "HDRRenderer": (".hdr", "HDRRenderer"),
    "HDREngine": (".hdr", "HDREngine"),
    "PBRScene": (".hdr", "Scene"),
    "PBRMaterial": (".hdr", "Material"),
    "PBRMesh": (".hdr", "Mesh"),
    "PBRMeshData": (".hdr", "MeshData"),
    "PBRDirectionalLight": (".hdr", "DirectionalLight"),
    "PBRPointLight": (".hdr", "PointLight"),
    "PBREnvironment": (".hdr", "Environment"),
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
    "VulkanRenderer": (".vulkan", "VulkanRenderer"),
    "GPUInfo": (".gpu", "GPUInfo"),
    "inspect_device_properties": (".gpu", "inspect_device_properties"),
    "validate_context": (".gpu", "validate_context"),
    "inspect_context": (".gpu", "inspect_context"),
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
