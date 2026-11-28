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
