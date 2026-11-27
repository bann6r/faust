"""Scene-level forward renderer built on Faust's ModernGL PBR backend."""

import math

import moderngl
import numpy as np

from .camera import Camera
from .rasterizer import PBRRenderer
from .scene import Scene


class Renderer:
    """Owns the PBR backend and draws all objects in a Scene.

    The window and OpenGL context are supplied by the application. This keeps
    Faust usable from GLFW, SDL, Qt, or an existing ModernGL context.
    """

    def __init__(self, context, clear_color=(0.025, 0.03, 0.045, 1.0)):
        self.context = context
        self.backend = PBRRenderer(context)
        color = tuple(float(v) for v in clear_color)
        if len(color) != 4 or not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in color):
            self.backend.release()
            raise ValueError("clear_color must be four finite values in [0, 1]")
        self.clear_color = color
        self._released = False

    def render(self, scene, camera, *, viewport=None, clear=True):
        """Draw a scene; viewport is an optional (x, y, width, height) tuple."""
        if self._released:
            raise RuntimeError("renderer has been released")
        if not isinstance(scene, Scene):
            raise TypeError("scene must be a Faust Scene")
        if not isinstance(camera, Camera):
            raise TypeError("camera must be a Faust Camera")

        if viewport is not None:
            if len(viewport) != 4:
                raise ValueError("viewport must be (x, y, width, height)")
            x, y, width, height = (int(v) for v in viewport)
            if width <= 0 or height <= 0:
                raise ValueError("viewport width and height must be positive")
            self.context.viewport = (x, y, width, height)
        if clear:
            self.context.clear(*self.clear_color, depth=1.0)

        self.context.enable(moderngl.DEPTH_TEST)
        self.context.depth_func = "<"

        view = camera.view_matrix
        projection = camera.projection_matrix
        light = scene.light
        for item in scene.objects:
            model = np.asarray(item.model, dtype=np.float32)
            if model.shape != (4, 4) or not np.isfinite(model).all():
                raise ValueError("render object contains an invalid model matrix")
            self.backend.render_mesh(
                item.mesh,
                model,
                view,
                projection,
                item.material,
                camera.position,
                base_color_texture=item.base_color_texture,
                light_direction=light.direction,
                light_radiance=light.radiance,
                ambient_radiance=light.ambient_radiance,
            )

    def release(self):
        if not self._released:
            self.backend.release()
            self._released = True
