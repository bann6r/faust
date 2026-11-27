"""Forward PBR renderer for Faust's current OpenGL prototype.

This module owns the geometry/shader/GPU resources for this test renderer.
It does not own windows, scenes, asset loading, or the game loop.
"""

import moderngl
import numpy as np

try:
    from .material import Material
except ImportError:  # Supports running engine/__init__.py directly.
    from material import Material


PBR_VERTEX_SHADER = """
#version 330

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normal_matrix;

out vec3 world_position;
out vec3 world_normal;
out vec2 texcoord;

void main() {
    vec4 world = model * vec4(in_position, 1.0);
    world_position = world.xyz;
    world_normal = normalize(normal_matrix * in_normal);
    texcoord = in_uv;
    gl_Position = projection * view * world;
}
"""


PBR_FRAGMENT_SHADER = """
#version 330

in vec3 world_position;
in vec3 world_normal;
in vec2 texcoord;

uniform sampler2D base_color_map;
uniform bool use_base_color_map;
uniform vec3 camera_position;

uniform vec3 material_base_color;
uniform float material_metallic;
uniform float material_roughness;
uniform vec3 material_emissive;

uniform vec3 light_direction;
uniform vec3 light_radiance;
uniform vec3 ambient_radiance;

out vec4 frag_color;

const float PI = 3.14159265358979323846;
const float MIN_ROUGHNESS = 0.045;
const float EPSILON = 0.000001;

float distribution_ggx(float n_dot_h, float roughness) {
    float alpha = roughness * roughness;
    float alpha_squared = alpha * alpha;
    float n_dot_h_squared = n_dot_h * n_dot_h;
    float denominator = n_dot_h_squared * (alpha_squared - 1.0) + 1.0;
    return alpha_squared /
        max(PI * denominator * denominator, EPSILON);
}

float geometry_schlick_ggx(float n_dot_x, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return n_dot_x / max(n_dot_x * (1.0 - k) + k, EPSILON);
}

float geometry_smith(float n_dot_v, float n_dot_l, float roughness) {
    float view_visibility = geometry_schlick_ggx(n_dot_v, roughness);
    float light_visibility = geometry_schlick_ggx(n_dot_l, roughness);
    return view_visibility * light_visibility;
}

vec3 fresnel_schlick(float v_dot_h, vec3 f0) {
    float factor = pow(clamp(1.0 - v_dot_h, 0.0, 1.0), 5.0);
    return f0 + (1.0 - f0) * factor;
}

void main() {
    vec3 sampled_base_color = material_base_color;
    if (use_base_color_map) {
        sampled_base_color *= texture(base_color_map, texcoord).rgb;
    }
    vec3 n = normalize(world_normal);
    vec3 v = normalize(camera_position - world_position);
    vec3 l = normalize(light_direction);
    vec3 half_vector = v + l;
    vec3 h = length(half_vector) > EPSILON ? normalize(half_vector) : n;

    float n_dot_v = max(dot(n, v), 0.001);
    float n_dot_l = max(dot(n, l), 0.0);
    float n_dot_h = max(dot(n, h), 0.0);
    float v_dot_h = max(dot(v, h), 0.0);

    float roughness = clamp(material_roughness, MIN_ROUGHNESS, 1.0);
    float metallic = clamp(material_metallic, 0.0, 1.0);

    // Dielectrics start near 4% reflectance; metals use their base color.
    vec3 f0 = mix(vec3(0.04), sampled_base_color, metallic);
    vec3 fresnel = fresnel_schlick(v_dot_h, f0);

    float distribution = distribution_ggx(n_dot_h, roughness);
    float geometry = geometry_smith(n_dot_v, n_dot_l, roughness);
    vec3 specular = (distribution * geometry * fresnel) /
        max(4.0 * n_dot_v * n_dot_l, EPSILON);

    vec3 diffuse_weight = (vec3(1.0) - fresnel) * (1.0 - metallic);
    vec3 diffuse = diffuse_weight * sampled_base_color / PI;

    // One directional key light.
    vec3 direct_light = (diffuse + specular) * light_radiance * n_dot_l;

    // Temporary ambient fill only; this is not environment-map IBL.
    vec3 ambient_light =
        sampled_base_color * ambient_radiance * (1.0 - metallic);

    vec3 linear_color = direct_light + ambient_light + material_emissive;

    // Temporary per-object display mapping. Faust should later move this
    // into one shared HDR tone-mapping/post-processing pass.
    vec3 mapped_color = linear_color / (linear_color + vec3(1.0));
    mapped_color = pow(max(mapped_color, vec3(0.0)), vec3(1.0 / 2.2));

    frag_color = vec4(mapped_color, 1.0);
}
"""


def create_cube_geometry() -> np.ndarray:
    """Create 36 triangle vertices containing position and face normal."""
    faces = (
        ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1), (0, 0, 1)),
        ((1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1), (0, 0, -1)),
        ((1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1), (1, 0, 0)),
        ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1), (-1, 0, 0)),
        ((-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1), (0, 1, 0)),
        ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1), (0, -1, 0)),
    )

    vertices = []
    for a, b, c, d, normal in faces:
        for position in (a, b, c, a, c, d):
            vertices.extend(position)
            vertices.extend(normal)
            vertices.extend(((position[0] + 1.0) * 0.5, (position[1] + 1.0) * 0.5))

    return np.asarray(vertices, dtype="f4")


class PBRRenderer:
    """GPU resources for rendering the prototype cube using a PBR material."""

    def __init__(self, context):
        self.context = context
        self.program = context.program(
            vertex_shader=PBR_VERTEX_SHADER,
            fragment_shader=PBR_FRAGMENT_SHADER,
        )
        self.vertex_buffer = None
        self.vertex_array = None
        try:
            self.vertex_buffer = context.buffer(create_cube_geometry().tobytes())
            self.vertex_array = context.vertex_array(
                self.program,
                [(self.vertex_buffer, "3f 3f 2f", "in_position", "in_normal", "in_uv")],
            )
        except Exception:
            self.release()
            raise

    @staticmethod
    def _finite_vec3(name, value, *, nonzero=False):
        try:
            vector = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain three numeric values") from exc
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"{name} must be a finite 3D vector")
        if nonzero and float(np.linalg.norm(vector)) <= 1e-8:
            raise ValueError(f"{name} must not be the zero vector")
        return vector

    @staticmethod
    def _matrix4(name, value):
        try:
            matrix = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a numeric 4x4 matrix") from exc
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"{name} must be a finite 4x4 matrix")
        return matrix

    def _set_scene_uniforms(
        self, model, view, projection, material, camera_position,
        light_direction, light_radiance, ambient_radiance,
    ):
        model_matrix = self._matrix4("model", model)
        view_matrix = self._matrix4("view", view)
        projection_matrix = self._matrix4("projection", projection)
        linear_transform = model_matrix[:3, :3]
        determinant = float(np.linalg.det(linear_transform))
        if not np.isfinite(determinant) or abs(determinant) <= 1e-10:
            raise ValueError("model transform must have a non-singular 3x3 basis")
        normal_matrix = np.linalg.inv(linear_transform).T.astype("f4")

        camera = self._finite_vec3("camera_position", camera_position)
        direction = self._finite_vec3("light_direction", light_direction, nonzero=True)
        direction /= np.linalg.norm(direction)
        radiance = self._finite_vec3("light_radiance", light_radiance)
        ambient = self._finite_vec3("ambient_radiance", ambient_radiance)
        if np.any(radiance < 0.0) or np.any(ambient < 0.0):
            raise ValueError("light radiance values must be non-negative")

        self.program["model"].write(model_matrix.T.tobytes())
        self.program["view"].write(view_matrix.T.tobytes())
        self.program["projection"].write(projection_matrix.T.tobytes())
        self.program["normal_matrix"].write(normal_matrix.T.tobytes())

        parameters = material.shader_parameters()
        self.program["material_base_color"].value = parameters["base_color"]
        self.program["material_metallic"].value = parameters["metallic"]
        self.program["material_roughness"].value = parameters["roughness"]
        self.program["material_emissive"].value = parameters["emissive"]
        self.program["camera_position"].value = tuple(float(v) for v in camera)
        self.program["light_direction"].value = tuple(float(v) for v in direction)
        self.program["light_radiance"].value = tuple(float(v) for v in radiance)
        self.program["ambient_radiance"].value = tuple(float(v) for v in ambient)

    def render_mesh(self, mesh, model, view, projection, material, camera_position, base_color_texture=None, *, light_direction=(0.45, 0.8, 0.35), light_radiance=(4.0, 3.8, 3.5), ambient_radiance=(0.10, 0.12, 0.16)):
        """Render a Faust Mesh using its UV0 channel and optional sRGB albedo."""
        if not isinstance(material, Material):
            raise TypeError("material must be a Faust Material")
        self._set_scene_uniforms(model, view, projection, material, camera_position, light_direction, light_radiance, ambient_radiance)
        self.program["base_color_map"].value = 0
        self.program["use_base_color_map"].value = base_color_texture is not None
        if base_color_texture is not None:
            base_color_texture.use(location=0)
        mesh.render(self.program)

    def render_cube(
        self,
        model: np.ndarray,
        view: np.ndarray,
        projection: np.ndarray,
        material: Material,
        camera_position,
        *,
        light_direction=(0.45, 0.8, 0.35),
        light_radiance=(4.0, 3.8, 3.5),
        ambient_radiance=(0.10, 0.12, 0.16),
    ) -> None:
        if not isinstance(material, Material):
            raise TypeError("material must be a Faust Material")

        self._set_scene_uniforms(model, view, projection, material, camera_position, light_direction, light_radiance, ambient_radiance)
        self.program["use_base_color_map"].value = False
        self.vertex_array.render(mode=moderngl.TRIANGLES)

    def release(self) -> None:
        self.vertex_array.release()
        self.vertex_buffer.release()
        self.program.release()
