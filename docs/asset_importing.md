# Importing 3D assets

Faust uses Assimp only to decode interchange files such as FBX. The imported
vertex data is copied into Faust-owned `MeshData` objects; Faust creates the
GPU buffers, textures, shaders, and draw calls.

## Dependencies

Install the Python packages used by the prototype:

```powershell
python -m pip install -r requirements.txt
```

PyAssimp also needs the native Assimp shared library. Install a build of
Assimp compatible with your Python architecture and make its library
discoverable by Windows (for example, on `PATH`) before importing FBX files.
The Python package alone is not the native parser.

## Current mesh import path

`engine.load_meshes(path)` returns one `MeshData` per imported mesh.
Positions, normals, and UV channel 0 are expanded into a non-indexed triangle
stream. Assimp triangulates polygons and applies node transforms during import.

The current scene uses:

- `assets/multi_cleaner_bottle_4k.fbx`
- `assets/textures/multi_cleaner_bottle_diff_4k.jpg`

The base-color image is uploaded as an sRGB texture, so sampling converts it
to linear light before Faust's PBR shader uses it.

## Current limitations

This first integration renders the base-color map only. The roughness EXR,
ARM packed map, normal EXR, and AO image are present in the asset folder but
are not yet connected to the shader. Their channel packing and normal-map
orientation should be verified from the source material before wiring them
in. Imported Assimp material assignments are retained as mesh indices, but
the current demo applies one configured material and one base-color texture
to all imported submeshes.

The importer currently targets triangle meshes with normals and UV0. It does
not yet expose tangents, skinning, animation, multiple UV sets, or per-material
texture binding.
