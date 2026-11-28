# Importing 3D assets

Faust keeps asset parsing separate from graphics API resources. Assimp decodes
interchange files such as FBX; the importer copies the resulting vertex data
into Faust-owned `MeshData` objects.

## Optional dependencies

Install the Python packages used by the importer:

```powershell
python -m pip install -r requirements-assets.txt
```

PyAssimp also needs the native Assimp shared library; the Python package alone
is not the native parser. On Windows, Faust automatically looks for the
`assimp-vc145-mt.dll` build in the current user's
`Downloads/windows-x64-v6.0.6/Release` folder. It registers that directory
for DLL dependency lookup and directs PyAssimp's native-library lookup to the
DLL, so a global Windows PATH edit is not required.

To use a different Assimp build, set `ASSIMP_LIBRARY_DIR` to either the full
DLL path or the directory containing an `assimp*.dll` file. The DLL must
match the Python interpreter architecture and have its runtime dependencies
available.

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
