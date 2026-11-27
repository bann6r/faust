# Importing 3D assets

Faust keeps asset parsing separate from graphics API resources. Assimp decodes
interchange files such as FBX; the importer copies the resulting vertex data
into Faust-owned `MeshData` objects.

## Optional dependencies

Install the Python packages used by the importer:

```powershell
python -m pip install -r requirements-assets.txt
```

PyAssimp also needs the native Assimp shared library. Install a build of
Assimp compatible with your Python architecture and make its library
discoverable by Windows (for example, on `PATH`) before importing FBX files.
The Python package alone is not the native parser.

## Current mesh import path

`engine.load_meshes(path)` returns one `MeshData` per imported mesh.
Positions, normals, and UV channel 0 are expanded into a non-indexed triangle
stream. Assimp triangulates polygons and applies node transforms during import.

The demo assets are:

- `assets/multi_cleaner_bottle_4k.fbx`
- `assets/textures/multi_cleaner_bottle_diff_4k.jpg`

## Vulkan migration status

The current Vulkan sample draws a procedural triangle and does not yet upload
`MeshData` into Vulkan vertex/index buffers. The PBR material, texture,
normal-map, and EXR skybox paths have not yet been ported to Vulkan.

The importer currently targets triangle meshes with normals and UV0. It does
not yet expose tangents, skinning, animation, multiple UV sets, or per-material
texture binding.
