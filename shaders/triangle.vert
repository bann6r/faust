#version 450

// Vertex data is generated from gl_VertexIndex, so this first pipeline
// deliberately needs no vertex buffer or descriptor set.
const vec2 positions[3] = vec2[](
    vec2(0.0, -0.62),
    vec2(0.62, 0.52),
    vec2(-0.62, 0.52)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
