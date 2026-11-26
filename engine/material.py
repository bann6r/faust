"""Validated metallic-roughness material data for Faust."""

from dataclasses import dataclass
from math import isfinite
from typing import Optional, Tuple

Color3 = Tuple[float, float, float]


def _color3(name: str, value, *, maximum=None) -> Color3:
    try:
        color = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an iterable of three numbers") from exc
    if len(color) != 3:
        raise ValueError(f"{name} must contain exactly three components")
    for component in color:
        if not isfinite(component) or component < 0.0:
            raise ValueError(f"{name} components must be finite and non-negative")
        if maximum is not None and component > maximum:
            raise ValueError(f"{name} components must not exceed {maximum}")
    return color


def _unit_interval(name: str, value: float) -> float:
    number = float(value)
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return number


@dataclass(frozen=True, slots=True)
class Material:
    """Material parameters; colors are linear-light RGB."""

    name: str = "Default"
    base_color: Color3 = (0.8, 0.8, 0.8)
    metallic: float = 0.0
    roughness: float = 0.5
    emissive: Color3 = (0.0, 0.0, 0.0)
    base_color_texture: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("material name must be a non-empty string")
        object.__setattr__(self, "base_color", _color3("base_color", self.base_color, maximum=1.0))
        object.__setattr__(self, "emissive", _color3("emissive", self.emissive))
        object.__setattr__(self, "metallic", _unit_interval("metallic", self.metallic))
        object.__setattr__(self, "roughness", _unit_interval("roughness", self.roughness))
        if self.base_color_texture is not None:
            if not isinstance(self.base_color_texture, str) or not self.base_color_texture.strip():
                raise ValueError("base_color_texture must be a non-empty path or None")

    def shader_parameters(self) -> dict:
        return {
            "base_color": self.base_color,
            "metallic": self.metallic,
            "roughness": self.roughness,
            "emissive": self.emissive,
            "base_color_texture": self.base_color_texture,
        }
