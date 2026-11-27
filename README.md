# faust

faust is a ~~python-first~~ python GPU accelerated forward renderer ~~built on moderngl~~ PLANNING TO GO FULL VULKAN SHIT

this rubbish application owns the window and Vulkan context, and the renderer owns scene data, mesh upload and akldsjfa;ldskjfa;sdljfa;sdlkfj

## capabilities

- ~~opengl 3.3 core renderer using moderngl~~ why
- ~~forward metallic/roughness brdf with ggx distribution and smith visibility~~ judgment day
- ~~mesh instances, materials, directional lighting, perspective cameras~~ judgment day
- ~~fbx import through pyassimp~~ also judgment day x 2
- ~~sRGB base-color texture sampling with mipmaps~~ no way
- ~~optional equirectangular exr background~~ no way
- ~~headless cpu-side camera, material, scene, and capability tests~~ end of the world
- ~~Vulkan instance, device, swapchain, render pass and graphics pipeline~~ send the addy of the creator's house
- ~~command buffers, synchronization and presentation~~ come on man
- ~~one (1) procedural triangle. groundbreaking.~~ no

even though faust is inspired by Unreal's renderer, faust is an engine foundation, not a full unreal replacement

the Vulkan backend currently knows how to present a triangle. the CPU-side camera, scene, material, and mesh systems exist, but the renderer has not yet figured out how to use them.

still missing:
- GPU mesh and index buffers
- depth buffering and multiple objects
- descriptor sets and uniform buffers
- texture upload and material binding
- PBR and image-based lighting
- shadow maps
- actually rendering the FBX sample

we have a roadmap. unfortunately, the roadmap does not render anything either.

see [docs/renderer-roadmap.md](docs/renderer-roadmap.md) for the ~~boring technical details~~ sacred scrolls.

## Install

~~python 3.10+ is recommended~~ GO FULL PYTHON 3.12+

    python -m pip install -r requirements.txt

~~FBX import needs both PyAssimp and the native Assimp library. See
[docs/asset_importing.md](docs/asset_importing.md) for setup notes. EXR skyboxes
need the OpenEXR Python package; the rest of the engine can be imported without
loading OpenEXR.~~ SETUPTOOLS

you will also need Vulkan support and `glslc` available on your PATH. good luck.

## run the sample

    python game.py

press esc to close. ~~the sample uses the FBX and textures in assets/.
When the EXR skybox cannot be loaded, Faust reports the issue and continues
with the scene clear color.~~ SHOUTOUT TO POLY HAVEN

if you see a triangle, everything is working. probably.

## architecture

python handles engine orchestration, scene management and asset handling. Vulkan and GLSL handle the actual rendering.

the goal is realistic, lightweight and powerful. ~~three words that will surely not cause scope creep~~ what the fuck

## test

run CPU-side tests with:

    python -m pytest -q

these tests do not create a window or require a graphics driver.

---

faust IS IN WIP btw.
