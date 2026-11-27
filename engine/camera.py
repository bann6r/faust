"""Camera math and validated perspective-camera state for Faust.

Matrices are row-major NumPy arrays and use a right-handed camera looking down
-Z. OpenGL and Vulkan use different clip-space depth ranges, so both projection
conventions are exposed explicitly instead of silently mixing them.
"""

from dataclasses import dataclass
import math

import numpy as np


def _vec3(name, value):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain three numeric values") from exc
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3D vector")
    return result.copy()


def _unit(name, value):
    vector = _vec3(name, value)
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError(f"{name} must not be the zero vector")
    return vector / length


def _projection_parameters(fov_y, aspect, near, far):
    values = tuple(float(value) for value in (fov_y, aspect, near, far))
    fov_y, aspect, near, far = values
    if not all(math.isfinite(value) for value in values):
        raise ValueError("projection parameters must be finite")
    if not 0.0 < fov_y < math.pi:
        raise ValueError("fov_y must be between 0 and pi radians")
    if aspect <= 0.0:
        raise ValueError("aspect must be greater than zero")
    if near <= 0.0 or far <= near:
        raise ValueError("clip planes must satisfy 0 < near < far")
    return fov_y, aspect, near, far


def perspective(fov_y, aspect, near, far):
    """Return a right-handed OpenGL projection (NDC depth -1..1).

    fov_y is in radians. The camera looks down -Z.
    """
    fov_y, aspect, near, far = _projection_parameters(fov_y, aspect, near, far)
    focal = 1.0 / math.tan(fov_y * 0.5)
    return np.asarray(
        (
            (focal / aspect, 0.0, 0.0, 0.0),
            (0.0, focal, 0.0, 0.0),
            (0.0, 0.0, (far + near) / (near - far),
             (2.0 * far * near) / (near - far)),
            (0.0, 0.0, -1.0, 0.0),
        ),
        dtype=np.float32,
    )


def perspective_vulkan(fov_y, aspect, near, far):
    """Return a right-handed Vulkan projection (NDC depth 0..1).

    The Y scale is negated to account for Vulkan's framebuffer Y direction
    when using a positive-height viewport. fov_y is in radians.
    """
    fov_y, aspect, near, far = _projection_parameters(fov_y, aspect, near, far)
    focal = 1.0 / math.tan(fov_y * 0.5)
    return np.asarray(
        (
            (focal / aspect, 0.0, 0.0, 0.0),
            (0.0, -focal, 0.0, 0.0),
            (0.0, 0.0, far / (near - far),
             (far * near) / (near - far)),
            (0.0, 0.0, -1.0, 0.0),
        ),
        dtype=np.float32,
    )


def look_at(eye, target, up):
    """Build a right-handed view matrix, including parallel-up fallback."""
    eye = _vec3("eye", eye)
    target = _vec3("target", target)
    up = _unit("up", up)
    forward = _unit("target - eye", target - eye)

    right = np.cross(forward, up)
    if np.linalg.norm(right) <= 1e-8:
        fallback = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(forward)))]
        right = np.cross(forward, fallback)
    right = _unit("camera right", right)
    camera_up = _unit("camera up", np.cross(right, forward))

    matrix = np.eye(4, dtype=np.float32)
    matrix[0, :3] = right
    matrix[1, :3] = camera_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -np.dot(right, eye)
    matrix[1, 3] = -np.dot(camera_up, eye)
    matrix[2, 3] = np.dot(forward, eye)
    return matrix


def rotation_y(angle):
    """Return a Y-axis rotation matrix; angle is in radians."""
    angle = float(angle)
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        (
            (cosine, 0.0, sine, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (-sine, 0.0, cosine, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )


@dataclass(slots=True)
class Camera:
    """Perspective camera; public field of view is expressed in degrees."""

    position: tuple = (0.0, 0.0, 5.0)
    target: tuple = (0.0, 0.0, 0.0)
    up: tuple = (0.0, 1.0, 0.0)
    fov_y_degrees: float = 70.0
    near: float = 0.1
    far: float = 100.0
    aspect: float = 16.0 / 9.0

    def __post_init__(self):
        self.position = _vec3("position", self.position).astype(np.float32)
        self.target = _vec3("target", self.target).astype(np.float32)
        self.up = _unit("up", self.up).astype(np.float32)
        self.fov_y_degrees = float(self.fov_y_degrees)
        self.near = float(self.near)
        self.far = float(self.far)
        self.aspect = float(self.aspect)
        if not math.isfinite(self.fov_y_degrees) or not 1.0 < self.fov_y_degrees < 179.0:
            raise ValueError("fov_y_degrees must be between 1 and 179")
        if not math.isfinite(self.aspect) or self.aspect <= 0.0:
            raise ValueError("aspect must be greater than zero")
        perspective(math.radians(self.fov_y_degrees), self.aspect, self.near, self.far)
        look_at(self.position, self.target, self.up)

    @property
    def forward(self):
        return _unit("target - position", self.target - self.position).astype(np.float32)

    @property
    def view_matrix(self):
        return look_at(self.position, self.target, self.up)

    @property
    def projection_matrix(self):
        """OpenGL projection matrix (NDC depth -1..1)."""
        return perspective(
            math.radians(self.fov_y_degrees), self.aspect, self.near, self.far
        )

    @property
    def vulkan_projection_matrix(self):
        """Vulkan projection matrix (NDC depth 0..1, framebuffer Y flipped)."""
        return perspective_vulkan(
            math.radians(self.fov_y_degrees), self.aspect, self.near, self.far
        )

    def resize(self, width, height):
        """Update aspect ratio; zero-sized minimized windows are ignored."""
        width, height = int(width), int(height)
        if width < 0 or height < 0:
            raise ValueError("framebuffer dimensions cannot be negative")
        if width == 0 or height == 0:
            return False
        self.aspect = width / height
        return True

    def look_at(self, target):
        """Retarget the camera while preserving its current position."""
        target = _vec3("target", target)
        look_at(self.position, target, self.up)
        self.target = target.astype(np.float32)
