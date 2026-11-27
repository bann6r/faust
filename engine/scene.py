"""CPU-side scene descriptions, independent of Vulkan resources."""

from dataclasses import dataclass, field
import math

import numpy as np

from .material import Material


def _color(name, value):
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain three numbers") from exc
    if len(result) != 3 or not all(math.isfinite(v) and v >= 0.0 for v in result):
        raise ValueError(f"{name} must be three finite, non-negative values")
    return result


def _matrix4(value):
    try:
        matrix = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError("model must be a numeric 4x4 matrix") from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("model must be a finite 4x4 matrix")
    return np.ascontiguousarray(matrix)


@dataclass(frozen=True, slots=True)
class DirectionalLight:
    """Single directional light; direction points from the surface to the light."""

    direction: tuple = (0.45, 0.8, 0.35)
    radiance: tuple = (4.0, 3.8, 3.5)
    ambient_radiance: tuple = (0.10, 0.12, 0.16)

    def __post_init__(self):
        original = np.asarray(self.direction, dtype=np.float64)
        if original.shape != (3,) or not np.isfinite(original).all():
            raise ValueError("direction must be a finite 3D vector")
        length = float(np.linalg.norm(original))
        if length <= 1e-12:
            raise ValueError("direction must not be the zero vector")
        object.__setattr__(self, "direction", tuple(float(v) for v in original / length))
        object.__setattr__(self, "radiance", _color("radiance", self.radiance))
        object.__setattr__(
            self, "ambient_radiance", _color("ambient_radiance", self.ambient_radiance)
        )


@dataclass(slots=True)
class RenderObject:
    """One CPU mesh instance and its material/model state."""

    mesh: object
    material: Material
    model: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    base_color_texture: object = None

    def __post_init__(self):
        # Accept CPU MeshData (vertices) or a future backend-owned mesh object.
        if not (hasattr(self.mesh, "vertices") or callable(getattr(self.mesh, "render", None))):
            raise TypeError("mesh must be Faust mesh data or a backend mesh resource")
        if not isinstance(self.material, Material):
            raise TypeError("material must be a Faust Material")
        self.model = _matrix4(self.model)


class Scene:
    """A CPU scene description with one directional key light."""

    def __init__(self, light=None):
        self.light = light if light is not None else DirectionalLight()
        if not isinstance(self.light, DirectionalLight):
            raise TypeError("light must be a DirectionalLight")
        self.objects = []

    def add(self, mesh, material, model=None, base_color_texture=None):
        """Add a mesh instance and return its mutable RenderObject handle."""
        transform = np.eye(4, dtype=np.float32) if model is None else model
        item = RenderObject(mesh, material, transform, base_color_texture)
        self.objects.append(item)
        return item

    def remove(self, item):
        self.objects.remove(item)

    def clear(self):
        self.objects.clear()

    def __len__(self):
        return len(self.objects)
