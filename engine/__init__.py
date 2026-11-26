"""
Faust prototype: EXR equirectangular skybox + rotating color cube.

Dependencies:
    pip install glfw moderngl numpy OpenEXR

Place an equirectangular .exr environment map at assets/sky.exr.
Run this file directly to test the skybox.
"""

from pathlib import Path
import math

import glfw
import moderngl
import numpy as np
import OpenEXR

if __package__:
    from .material import Material
    from .rasterizer import PBRRenderer
    from .mesh import Mesh
    from .importer import load_meshes
    from .texture import load_color_texture
else:
    from material import Material
    from rasterizer import PBRRenderer
    from mesh import Mesh
    from importer import load_meshes
    from texture import load_color_texture


SKYBOX_VERTEX_SHADER = """
#version 330

in vec3 in_position;

uniform mat4 view;
uniform mat4 projection;

out vec3 sample_direction;

void main() {
    sample_direction = in_position;

    // Remove camera translation: the skybox should rotate with the camera,
    // but it must always remain infinitely far away.
    mat4 rotation_only_view = mat4(mat3(view));
    vec4 clip_position =
        projection * rotation_only_view * vec4(in_position, 1.0);

    // Force skybox depth to the far plane.
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

    // Equirectangular (longitude/latitude) mapping.
    float u = atan(direction.z, direction.x) / (2.0 * PI) + 0.5;
    float v = asin(clamp(direction.y, -1.0, 1.0)) / PI + 0.5;

    vec3 hdr_color = texture(environment_map, vec2(u, v)).rgb;

    // Simple display mapping for this prototype. A future HDR renderer
    // should tone-map the whole scene in a shared post-processing pass.
    hdr_color = max(hdr_color, vec3(0.0));
    vec3 mapped = hdr_color / (hdr_color + vec3(1.0));
    mapped = pow(mapped, vec3(1.0 / 2.2));

    frag_color = vec4(mapped, 1.0);
}
"""


def perspective(fov, aspect, near, far):
    f = 1.0 / math.tan(fov / 2.0)
    return np.array(
        [
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far),
             (2.0 * far * near) / (near - far)],
            [0, 0, -1, 0],
        ],
        dtype="f4",
    )


def look_at(eye, target, up):
    forward = target - eye
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    camera_up = np.cross(right, forward)

    return np.array(
        [
            [right[0], right[1], right[2], -np.dot(right, eye)],
            [camera_up[0], camera_up[1], camera_up[2],
             -np.dot(camera_up, eye)],
            [-forward[0], -forward[1], -forward[2], np.dot(forward, eye)],
            [0, 0, 0, 1],
        ],
        dtype="f4",
    )


def rotation_y(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1],
        ],
        dtype="f4",
    )


def create_skybox_cube():
    # 36 triangle vertices. Culling is disabled while drawing the skybox,
    # so the camera can see the inside faces.
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
    return np.asarray(vertices, dtype="f4")


def _channel_array(channel):
    """Return the pixel array from an OpenEXR Channel object."""
    return np.asarray(getattr(channel, "pixels", channel))


def load_exr_rgb(path):
    """
    Load the RGB channels from an OpenEXR image as float32.

    This expects an equirectangular environment map. It supports separate
    R/G/B channels and a packed RGB channel. Alpha and extra channels are
    ignored. The returned array is top-to-bottom image order.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"EXR environment map not found: {path}")

    exr_file = OpenEXR.File(str(path), separate_channels=True)
    channels = exr_file.channels()

    def find_channel(short_name):
        if short_name in channels:
            return channels[short_name]

        # Multi-layer EXRs often name channels like "beauty.R".
        matches = [
            value for name, value in channels.items()
            if name.split(".")[-1] == short_name
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    red = find_channel("R")
    green = find_channel("G")
    blue = find_channel("B")

    if red is not None and green is not None and blue is not None:
        rgb = np.stack(
            [
                _channel_array(red),
                _channel_array(green),
                _channel_array(blue),
            ],
            axis=-1,
        )
    else:
        # Some EXRs expose a packed RGB channel instead.
        packed = channels.get("RGB")
        if packed is None:
            packed = channels.get("rgba")
        if packed is None:
            available = ", ".join(channels.keys())
            raise ValueError(
                "Could not find RGB channels in EXR. "
                f"Available channels: {available}"
            )

        rgb = _channel_array(packed)[..., :3]

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"Expected an HxWx3 EXR image, got shape {rgb.shape}"
        )

    rgb = np.asarray(rgb, dtype=np.float32)
    if not np.isfinite(rgb).all():
        raise ValueError("EXR contains NaN or infinite pixel values")

    # OpenEXR arrays are top-to-bottom. Flip so the uploaded texture's
    # OpenGL v=1 edge corresponds to the top of the source image.
    return np.ascontiguousarray(np.flipud(rgb))


class Skybox:
    """GPU-backed equirectangular EXR skybox."""

    def __init__(self, ctx, path):
        self.ctx = ctx
        self.path = Path(path)

        pixels = load_exr_rgb(self.path)
        height, width, _ = pixels.shape

        self.texture = ctx.texture(
            (width, height),
            components=3,
            data=pixels.tobytes(),
            dtype="f4",
            alignment=1,
        )
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texture.repeat_x = True
        self.texture.repeat_y = False

        self.program = ctx.program(
            vertex_shader=SKYBOX_VERTEX_SHADER,
            fragment_shader=SKYBOX_FRAGMENT_SHADER,
        )

        self.vbo = ctx.buffer(create_skybox_cube().tobytes())
        self.vao = ctx.vertex_array(
            self.program,
            [(self.vbo, "3f", "in_position")],
        )

        self.program["environment_map"].value = 0

    def render(self, view, projection):
        self.program["view"].write(view.T.tobytes())
        self.program["projection"].write(projection.T.tobytes())

        # The skybox is the background, so draw it with depth testing off.
        # OpenGL does not update the depth buffer while depth testing is
        # disabled. Re-enable it before the regular scene is rendered.
        self.ctx.disable(moderngl.DEPTH_TEST)
        try:
            self.texture.use(location=0)
            self.vao.render(mode=moderngl.TRIANGLES)
        finally:
            self.ctx.enable(moderngl.DEPTH_TEST)

    def release(self):
        self.vao.release()
        self.vbo.release()
        self.program.release()
        self.texture.release()


def find_project_asset(*parts):
    """
    Resolve assets whether this file lives in the project root or in engine/.
    The shown project layout keeps assets/ beside engine/.
    """
    here = Path(__file__).resolve().parent
    candidates = (here.joinpath(*parts), here.parent.joinpath(*parts))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Return the project-root candidate so the eventual error names the
    # expected location clearly.
    return candidates[-1]



__all__ = [
    "Material",
    "PBRRenderer",
    "Mesh",
    "load_meshes",
    "load_color_texture",
    "Skybox",
    "perspective",
    "look_at",
    "rotation_y",
    "find_project_asset",
    "load_exr_rgb",
]
