# Faust

Faust is a Python-first renderer/engine project moving from ModernGL to
**Vulkan**. The current branch is the first backend milestone: it creates a
Vulkan instance and GLFW surface, selects a graphics/present device, creates a
swapchain and graphics pipeline, records command buffers, synchronizes frames,
and presents a triangle.

This is an engine foundation, not a full Unreal replacement.

## Current status

Implemented in the Vulkan migration:

- GLFW window with no OpenGL context
- Vulkan instance, surface, physical device, logical device, and queues
- Swapchain, image views, render pass, and graphics pipeline
- GLSL shaders compiled to SPIR-V with `glslc`
- Command buffers, acquire/submit/present, and a frame fence
- Swapchain recreation when the framebuffer changes size
- CPU-side camera, material, scene, and FBX mesh-data types remain separate
  from the graphics API

Not migrated yet:

- Vulkan vertex/index buffer allocation and mesh upload
- Depth buffering and multiple render objects
- Descriptor sets, uniform buffers, and material binding
- Texture upload, sRGB sampling, and mipmaps
- PBR shaders, directional-light data, and EXR skyboxes
- The existing FBX sample scene
- Vulkan-specific camera projection conventions

The triangle is intentionally procedural: its three vertices are generated
from `gl_VertexIndex`, so this first pipeline does not hide buffer allocation
behind an abstraction. Camera math is still CPU-side and is not used by this sample yet.

## Requirements

Python 3.10+ and these Python dependencies:

```sh
python -m pip install -r requirements.txt
```

Faust also needs a system Vulkan loader and a Vulkan-capable graphics driver.
Install the Vulkan SDK (or equivalent development tools) and make
`glslc` available on `PATH`; the sample compiles the checked-in GLSL shaders
to temporary SPIR-V files at startup.

For optional FBX/texture asset-import work:

```sh
python -m pip install -r requirements-assets.txt
```

PyAssimp additionally requires the native Assimp library. The Python package
alone does not provide that parser.

## Run the Vulkan sample

```sh
python game.py
```

The sample opens a 1280×720 window and draws an orange triangle. Press Escape
to close it. The application owns GLFW and the window; `VulkanRenderer` owns
the Vulkan instance, surface, device, swapchain, pipeline, command buffers,
and synchronization objects.

## Public API

The CPU-side types remain importable without creating a Vulkan instance:

```python
from faust import engine

camera = engine.Camera(position=(0, 0, 5), target=(0, 0, 0))
scene = engine.Scene()
material = engine.Material()
```

The current `Renderer` is the Vulkan renderer and takes a GLFW window:

```python
import glfw
from faust import engine

if not glfw.init():
    raise RuntimeError("could not initialize GLFW")

glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
window = glfw.create_window(1280, 720, "Faust", None, None)
if window is None:
    glfw.terminate()
    raise RuntimeError("could not create window")

try:
    with engine.Renderer(window) as renderer:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            renderer.draw_frame()
finally:
    glfw.destroy_window(window)
    glfw.terminate()
```

## Tests

The test suite is CPU-only; it does not create a window or require a Vulkan
driver:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
