# Faust renderer roadmap

Faust is a renderer project, not a drop-in Unreal Engine replacement today. The
current implementation is a Vulkan 1.0 presentation milestone: it creates a
GLFW surface, selects a graphics/present device, builds a swapchain and a
triangle-only graphics pipeline, and presents frames. CPU scene, camera,
material, and imported mesh data exist, but the Vulkan backend does not consume
them yet.

This roadmap separates working functionality from planned work. A milestone
is complete only when its implementation, automated tests, and a runnable
sample agree.

## Current state

| Area | State | Remaining work |
| --- | --- | --- |
| Vulkan instance/device/surface | Implemented | Validation layers and diagnostics |
| Swapchain and frame synchronization | Implemented | Stress-test resize, minimize, and out-of-date paths |
| Procedural triangle pipeline | Implemented | Replace fixed pipeline with scene draw path |
| CPU camera and scene types | Implemented | Vulkan projection is explicit; connect camera uniforms |
| CPU mesh/material data | Implemented | GPU resources and material descriptors |
| FBX import | Optional CPU-side path | Preserve material slots, transforms, and texture references |
| Depth, indexed drawing | Not implemented | Depth image, depth attachment, vertex/index buffers |
| Textures | Not implemented in Vulkan | Staging uploads, image layouts, samplers, mip generation |
| PBR | Not implemented | Metallic-roughness BRDF, normal/roughness/metallic/AO maps |
| Image-based lighting | Not implemented | HDR decode, environment convolution, irradiance and prefiltered maps |
| Shadows | Not implemented | Shadow maps and bias/filtering |
| Profiling and scalability | Not implemented | GPU timestamps, memory budgets, batching and instancing |

## Ordered implementation plan

### 1. Scene submission and geometry

- Define one stable GPU vertex layout matching MeshData (position, normal,
  UV0) and use indexed geometry.
- Add device-local vertex/index buffers with staging uploads and explicit
  transfer-to-vertex-input barriers.
- Add a depth format query, depth image/view, and depth attachment.
- Create one descriptor-set layout and per-frame camera uniform buffer.
- Record one indexed draw per RenderObject; verify multiple overlapping
  objects render with correct depth.

**Exit criteria:** the existing FBX sample renders as geometry with a movable
camera, correct depth, and no per-frame GPU allocation.

### 2. Material and texture pipeline

- Add image decoding behind a small CPU-side asset boundary.
- Upload sRGB base-color textures and linear data textures with correct formats.
- Add samplers, mip levels, descriptor pools/sets, and material binding.
- Define explicit ownership and destruction order for every GPU resource.

**Exit criteria:** a textured mesh renders with base color, normal, metallic,
roughness, and ambient-occlusion inputs; missing maps use documented defaults.

### 3. Physically based lighting

- Implement a tested metallic-roughness BRDF (GGX distribution, Smith
  visibility, and Schlick Fresnel) in GLSL.
- Add directional-light uniforms and a stable tone-mapping/output transform.
- Decode HDR environment maps and precompute diffuse irradiance plus a
  roughness-prefiltered specular environment and BRDF integration lookup.

**Exit criteria:** the sample uses the included environment asset and produces
consistent results across a small material test grid.

### 4. Quality and performance

- Add shadow mapping, GPU timestamp queries, and a repeatable benchmark scene.
- Track CPU frame time, GPU frame time, draw count, upload bytes, and memory.
- Add frustum culling, instancing, and render-state/material sorting only after
  profiling demonstrates a bottleneck.
- Keep quality scalable through resolution, shadow, texture, and lighting
  settings rather than assuming high-end hardware.

**Exit criteria:** benchmark results and visual regression captures are
repeatable on at least one integrated and one discrete Vulkan GPU.

## Engineering rules

- Keep CPU scene/asset types importable without initializing Vulkan.
- Keep Vulkan handles inside backend resource objects; make ownership explicit.
- Prefer device-local resources and batched staging uploads for static assets.
- Do not claim a renderer feature is complete until it is used by the sample.
- Keep the CPU-only test suite independent of a display server and GPU driver.

## Non-goals for the first renderer

An editor, blueprint system, physics engine, animation suite, and general
game-engine tooling are not prerequisites for the renderer. Finish the
rendering path first; build engine/editor features only when the renderer
provides a stable foundation for them.
