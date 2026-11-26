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
else:
    from material import Material
    from rasterizer import PBRRenderer


VERTEX_SHADER = """
#version 330

in vec3 in_position;
in vec3 in_color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 vertex_color;

void main() {
    vertex_color = in_color;
    gl_Position = projection * view * model * vec4(in_position, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330

in vec3 vertex_color;
out vec4 frag_color;

void main() {
    frag_color = vec4(vertex_color, 1.0);
}
"""

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


def create_cube():
    faces = [
        ([-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1], (1, 0, 0)),
        ([1, -1, -1], [-1, -1, -1], [-1, 1, -1], [1, 1, -1], (0, 1, 0)),
        ([1, -1, 1], [1, -1, -1], [1, 1, -1], [1, 1, 1], (0, 0, 1)),
        ([-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1], (1, 1, 0)),
        ([-1, 1, 1], [1, 1, 1], [1, 1, -1], [-1, 1, -1], (1, 0, 1)),
        ([-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1], (0, 1, 1)),
    ]

    vertices = []
    for a, b, c, d, color in faces:
        for position in (a, b, c, a, c, d):
            vertices.extend(position)
            vertices.extend(color)

    return np.asarray(vertices, dtype="f4")


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


def main():
    if not glfw.init():
        raise RuntimeError("could not initialize GLFW")

    window = None
    ctx = None
    resources = []

    try:
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(
            glfw.OPENGL_PROFILE,
            glfw.OPENGL_CORE_PROFILE,
        )

        window = glfw.create_window(1280, 720, "Faust", None, None)
        if not window:
            raise RuntimeError("could not create GLFW window")

        glfw.make_context_current(window)
        glfw.swap_interval(1)

        ctx = moderngl.create_context()

        material = Material(
            name="PrototypePaint",
            base_color=(0.80, 0.12, 0.04),
            metallic=0.15,
            roughness=0.32,
        )
        pbr_renderer = PBRRenderer(ctx)
        resources.append(pbr_renderer)

        skybox_path = find_project_asset("assets", "sky.exr")
        skybox = Skybox(ctx, skybox_path)
        resources.append(skybox)

        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<"

        eye = np.array([0, 0, 5], dtype="f4")
        target = np.array([0, 0, 0], dtype="f4")
        up = np.array([0, 1, 0], dtype="f4")
        view = look_at(eye, target, up)

        start_time = glfw.get_time()

        print("Faust initialized")
        print("GPU:", ctx.info.get("GL_RENDERER", "Unknown"))
        print("OpenGL:", ctx.info.get("GL_VERSION", "Unknown"))
        print("Skybox:", skybox_path)

        while not glfw.window_should_close(window):
            glfw.poll_events()

            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break

            width, height = glfw.get_framebuffer_size(window)
            if width == 0 or height == 0:
                continue

            ctx.viewport = (0, 0, width, height)
            ctx.clear(0.025, 0.03, 0.045, 1.0, depth=1.0)

            projection = perspective(
                math.radians(70.0),
                width / height,
                0.1,
                100.0,
            )

            # Draw the environment first, then ordinary scene geometry.
            skybox.render(view, projection)

            elapsed = glfw.get_time() - start_time
            model = rotation_y(elapsed)

            pbr_renderer.render_cube(
                model,
                view,
                projection,
                material,
                camera_position=eye,
            )
            glfw.swap_buffers(window)

    finally:
        for resource in reversed(resources):
            try:
                resource.release()
            except Exception:
                pass

        if ctx is not None:
            ctx.release()

        if window is not None:
            glfw.destroy_window(window)

        glfw.terminate()


if __name__ == "__main__":
    main()
