# Faust

Faust is a Python-first, GPU-accelerated forward renderer built on ModernGL.
The application owns the window and OpenGL context; Faust owns scene data,
mesh upload, material shading, and draw submission.

## Current capabilities

- OpenGL 3.3 core renderer using ModernGL
- Forward metallic/roughness BRDF with GGX distribution and Smith visibility
- Mesh instances, materials, directional lighting, perspective cameras
- FBX import through Assimp/PyAssimp
- sRGB base-color texture sampling with mipmaps
- Optional equirectangular EXR background
- Headless CPU-side camera, material, scene, and capability tests

Faust is an engine foundation, not a full Unreal replacement. It currently
uses one directional key light and a simple ambient fill; it does not yet
provide shadow maps, clustered lights, skeletal animation, deferred rendering,
or physically based image-based lighting.

## Install

Python 3.10+ is recommended.

    python -m pip install -r requirements.txt

FBX import needs both PyAssimp and the native Assimp library. See
[docs/asset_importing.md](docs/asset_importing.md) for setup notes. EXR skyboxes
need the OpenEXR Python package; the rest of the engine can be imported without
loading OpenEXR.

## Run the sample

    python game.py

Press Escape to close. The sample uses the FBX and textures in assets/.
When the EXR skybox cannot be loaded, Faust reports the issue and continues
with the scene clear color.

## Use the renderer

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

The caller owns and must release uploaded meshes and textures, plus its window
and ModernGL context. The Renderer releases only its own GPU resources.

## Tests

Run CPU-side tests with:

    python -m pytest -q

These tests do not create a window or require a graphics driver.
