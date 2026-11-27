# faust

faust is a ~~python-first~~ python gpu accelerated forward renderer ~~built on moderngl~~ PLANNING TO GO FULL VULKAN SHIT
this rubbish application owns the window and opengl context and the renderer owns scene data, mesh upload and akldsjfa;ldskjfa;sdljfa;sdlkfj

## capabilities

- ~~opengl 3.3 core renderer using moderngl~~ why
- ~~forward metallic/roughness brdf with ggx distribution and smith visibility~~ judgment day
- ~~mesh instances, materials, directional lighting, perspective cameras~~ judgment day
- ~~fbx import through pyassimp~~ also judgment day x 2
- ~~sRGB base-color texture sampling with mipmaps~~ no way
- ~~optional equirectangular exr background~~ no way
- ~~headless cpu-side camera, material, scene, and capability tests~~ end of the world

even though faust is inspired by unreal's renderer, faust is an engine foundation, not a full unreal replacement.
it currently uses one directional key light and a simple ambient fill; it does not yet provide shadow maps, clustered lights, skeletal animation, deferred rendering, or physically based image-based lighting.

## Install

~~python 3.10+ is recommended~~ GO FULL PYTHON 3.12+

    python -m pip install -r requirements.txt

~~FBX import needs both PyAssimp and the native Assimp library. See
[docs/asset_importing.md](docs/asset_importing.md) for setup notes. EXR skyboxes
need the OpenEXR Python package; the rest of the engine can be imported without
loading OpenEXR.~~ SETUPTOOLS

## run the sample

    python game.py

press esc to close. ~~the sample uses the FBX and textures in assets/.
When the EXR skybox cannot be loaded, Faust reports the issue and continues
with the scene clear color.~~ SHOUTOUT TO POLY HAVEN

## use the renderer

    import moderngl
    from faust import engine

    # Create a ModernGL context after creating/making current your own window.
    camera = engine.Camera(position=(0, 0, 5), target=(0, 0, 0))
    scene = engine.Scene()
    renderer = engine.Renderer(context)

    # Import and upload a mesh, then add it to the scene.
    for mesh_data in engine.load_meshes("model.fbx"):
        mesh = engine.Mesh(context, mesh_data)
        scene.add(mesh, engine.Material())

    camera.resize(framebuffer_width, framebuffer_height)
    renderer.render(scene, camera, viewport=(0, 0, framebuffer_width, framebuffer_height))
    renderer.release()

~~the caller owns and must release uploaded meshes and textures, plus its window
and ModernGL context. The Renderer releases only its own GPU resources.~~ nvm

## test

run CPU-side tests with:

    python -m pytest -q

these tests do not create a window or require a graphics driver.
