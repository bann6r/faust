"""Equirectangular HDR environment background rendering."""

from pathlib import Path

import moderngl
import numpy as np


SKYBOX_VERTEX_SHADER = """
#version 330
in vec3 in_position;
uniform mat4 view;
uniform mat4 projection;
out vec3 sample_direction;
void main() {
    sample_direction = in_position;
    mat4 rotation_only_view = mat4(mat3(view));
    vec4 clip_position = projection * rotation_only_view * vec4(in_position, 1.0);
    gl_Position = clip_position.xyww;
}
"""

SKYBOX_FRAGMENT_SHADER = """
#version 330
in vec3 sample_direction;
uniform sampler2D environment_map;
out vec4 frag_color;
const float PI = 3.14159265358979323846;
void main() {
    vec3 direction = normalize(sample_direction);
    float u = atan(direction.z, direction.x) / (2.0 * PI) + 0.5;
    float v = asin(clamp(direction.y, -1.0, 1.0)) / PI + 0.5;
    vec3 hdr = max(texture(environment_map, vec2(u, v)).rgb, vec3(0.0));
    vec3 mapped = hdr / (hdr + vec3(1.0));
    mapped = pow(mapped, vec3(1.0 / 2.2));
    frag_color = vec4(mapped, 1.0);
}
"""


def create_skybox_cube():
    """Create the 36 position-only vertices used by the background cube."""
    vertices = [
        -1,  1, -1,  -1, -1, -1,   1, -1, -1,
         1, -1, -1,   1,  1, -1,  -1,  1, -1,
        -1, -1,  1,  -1, -1, -1,  -1,  1, -1,
        -1,  1, -1,  -1,  1,  1,  -1, -1,  1,
         1, -1, -1,   1, -1,  1,   1,  1,  1,
         1,  1,  1,   1,  1, -1,   1, -1, -1,
        -1, -1,  1,  -1,  1,  1,   1,  1,  1,
         1,  1,  1,   1, -1,  1,  -1, -1,  1,
        -1,  1, -1,   1,  1, -1,   1,  1,  1,
         1,  1,  1,  -1,  1,  1,  -1,  1, -1,
        -1, -1, -1,  -1, -1,  1,   1, -1, -1,
         1, -1, -1,  -1, -1,  1,   1, -1,  1,
    ]
    return np.asarray(vertices, dtype=np.float32)


def _channel_array(channel):
    return np.asarray(getattr(channel, "pixels", channel))


def load_exr_rgb(path):
    """Load RGB float pixels from an equirectangular EXR image."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"EXR environment map not found: {source}")
    try:
        import OpenEXR
    except ImportError as exc:
        raise ImportError(
            "EXR support requires OpenEXR; install the OpenEXR Python package"
        ) from exc

    exr_file = OpenEXR.File(str(source), separate_channels=True)
    channels = exr_file.channels()

    def find_channel(short_name):
        if short_name in channels:
            return channels[short_name]
        matches = [
            value for name, value in channels.items()
            if name.split(".")[-1] == short_name
        ]
        return matches[0] if len(matches) == 1 else None

    red, green, blue = (find_channel(name) for name in ("R", "G", "B"))
    if red is not None and green is not None and blue is not None:
        rgb = np.stack(
            [_channel_array(red), _channel_array(green), _channel_array(blue)],
            axis=-1,
        )
    else:
        packed = channels.get("RGB")
        if packed is None:
            packed = channels.get("rgba")
        if packed is None:
            raise ValueError(
                "EXR has no supported RGB channels; found: "
                + ", ".join(channels.keys())
            )
        rgb = _channel_array(packed)[..., :3]

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 EXR image, got {rgb.shape}")
    rgb = np.asarray(rgb, dtype=np.float32)
    if not np.isfinite(rgb).all():
        raise ValueError("EXR contains NaN or infinite pixel values")
    return np.ascontiguousarray(np.flipud(rgb))


class Skybox:
    """GPU-backed equirectangular EXR background.

    OpenEXR is imported only when an EXR is actually loaded, so importing
    Faust or running non-skybox scenes does not require the EXR dependency.
    """

    def __init__(self, context, path):
        self.context = context
        self.path = Path(path)
        self.texture = None
        self.program = None
        self.vbo = None
        self.vao = None
        try:
            pixels = load_exr_rgb(self.path)
            height, width, _ = pixels.shape
            self.texture = context.texture(
                (width, height), 3, pixels.tobytes(), dtype="f4", alignment=1
            )
            self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.texture.repeat_x = True
            self.texture.repeat_y = False
            self.program = context.program(
                vertex_shader=SKYBOX_VERTEX_SHADER,
                fragment_shader=SKYBOX_FRAGMENT_SHADER,
            )
            self.vbo = context.buffer(create_skybox_cube().tobytes())
            self.vao = context.vertex_array(
                self.program, [(self.vbo, "3f", "in_position")]
            )
            self.program["environment_map"].value = 0
        except Exception:
            self.release()
            raise

    def render(self, view, projection):
        if self.vao is None:
            raise RuntimeError("skybox has been released")
        self.program["view"].write(np.asarray(view, dtype=np.float32).T.tobytes())
        self.program["projection"].write(
            np.asarray(projection, dtype=np.float32).T.tobytes()
        )
        self.context.disable(moderngl.DEPTH_TEST)
        try:
            self.texture.use(location=0)
            self.vao.render(mode=moderngl.TRIANGLES)
        finally:
            self.context.enable(moderngl.DEPTH_TEST)

    def release(self):
        for resource_name in ("vao", "vbo", "program", "texture"):
            resource = getattr(self, resource_name, None)
            if resource is not None:
                try:
                    resource.release()
                except Exception:
                    pass
                setattr(self, resource_name, None)
