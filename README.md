# faust

faust is a ~~python-first~~ python GPU accelerated forward renderer ~~built on moderngl~~ PLANNING TO GO FULL VULKAN SHIT

this rubbish application owns the window and OpenGL context for the currently working sample. the Vulkan backend is being built alongside it, because apparently one graphics API wasn't enough.

## capabilities

- OpenGL 3.3 forward metallic/roughness renderer (GGX + Smith)
- mesh instances, materials, directional lighting, perspective cameras
- FBX import through pyassimp
- sRGB base-color texture sampling with mipmaps
- optional equirectangular EXR background
- headless CPU-side camera, material, scene, and capability tests
- Vulkan instance, device, swapchain, render pass and graphics pipeline
- Vulkan command buffers, synchronization and presentation
- ~~one (1) procedural triangle. groundbreaking.~~ currently the Vulkan sample

even though faust is inspired by Unreal's renderer, faust is an engine foundation, not a full Unreal replacement.

**Important:** `python game.py` runs the existing scene renderer and shows the FBX bottle with its texture and optional EXR skybox. The Vulkan backend is still an early triangle-only prototype; it has not replaced the scene renderer yet.

the plan is to move the existing scene features to Vulkan one by one, without deleting the working demo before the replacement can do the same job. revolutionary concept, apparently.

see [docs/renderer-roadmap.md](docs/renderer-roadmap.md) for the ~~boring technical details~~ sacred scrolls.

## Install

~~python 3.10+ is recommended~~ GO FULL PYTHON 3.12+

    python -m pip install -r requirements.txt

FBX import needs PyAssimp and the native Assimp library. See [docs/asset_importing.md](docs/asset_importing.md). EXR skyboxes need OpenEXR.

Vulkan development additionally needs the Vulkan SDK and `glslc` on PATH.

## run the sample

    python game.py

press esc to close. SHOUTOUT TO POLY HAVEN

this restores the pre-migration sample: FBX bottle, base-color texture, directional light, camera, rotation, and optional EXR skybox.

to run the tiny Vulkan bring-up sample instead:

    python game_vulkan.py

## test

    python -m pytest -q

CPU-side tests do not create a window or require a graphics driver.

---

faust IS IN WIP btw.


## OpenGL 3.3 PBR renderer

A separate, lightweight renderer lives in `faust/`. It does not replace or
import the experimental Vulkan backend, and it leaves window/context creation
to the application.

```bash
python demo.py
```

The material lab uses left-drag orbit, scroll zoom, and lighting presets 1–4.
It demonstrates dielectric roughness sweeps, gold, copper, chrome, brushed and
matte metals, an emissive highlight, HDR rendering, bloom, ACES tone mapping,
and a PCF-filtered directional shadow map.

The API is available from `faust`:

```python
import moderngl
from faust import Camera, DirectionalLight, Material, Mesh, Renderer, Scene, sphere_data

ctx = moderngl.create_context()  # create a current OpenGL 3.3 context first
mesh = Mesh(ctx, sphere_data())
scene = Scene(DirectionalLight())
scene.add(mesh, Material("ceramic", (0.7, 0.12, 0.06), metallic=0.0, roughness=0.3))
camera = Camera()
renderer = Renderer(ctx, (1280, 720))
try:
    renderer.render(scene, camera)  # renders to the current default framebuffer
finally:
    renderer.release()
    mesh.release()
```

For image-based lighting, pass linearly encoded irradiance and prefiltered
cubemaps plus a BRDF integration LUT with `Renderer.set_environment(Environment(...))`.
Faust binds these resources but deliberately does not hide expensive HDR
convolution or claim that a constant fallback is a physically captured studio.
An offline asset baker can generate the maps, while external texture/model
loaders can supply their own ModernGL textures and `MeshData`.

Run the offscreen GPU smoke test with `python -m pytest tests/test_render.py -q`.
It uses `moderngl.create_standalone_context()`; systems without a suitable
headless OpenGL driver skip the test. The existing CPU-side tests and Vulkan
prototype remain separate.
