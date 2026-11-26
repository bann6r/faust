"""FBX/mesh import boundary. Assimp parses files; Faust owns the resulting data."""

from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .mesh import MeshData


@contextmanager
def _open_scene(path):
    try:
        import pyassimp
        from pyassimp import postprocess
    except ImportError as exc:
        raise ImportError(
            "FBX import requires PyAssimp and the native Assimp library. "
            "See docs/asset_importing.md."
        ) from exc

    flags = (
        postprocess.aiProcess_Triangulate
        | postprocess.aiProcess_GenSmoothNormals
        | postprocess.aiProcess_CalcTangentSpace
        | postprocess.aiProcess_JoinIdenticalVertices
        | postprocess.aiProcess_PreTransformVertices
    )
    with pyassimp.load(str(path), processing=flags) as scene:
        yield scene


def load_meshes(path) -> list[MeshData]:
    """Import triangle meshes into Faust-owned position/normal/UV arrays.

    Each expanded vertex has eight float32 values: position(3), normal(3),
    and UV0(2). Returned data outlives the Assimp scene.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Model file not found: {source}")

    output = []
    with _open_scene(source) as scene:
        if not scene.meshes:
            raise ValueError(f"No meshes found in model: {source}")

        for index, imported in enumerate(scene.meshes):
            positions = np.asarray(imported.vertices, dtype=np.float32)
            if positions.ndim != 2 or positions.shape[1] < 3 or not len(positions):
                continue
            positions = positions[:, :3]

            normals = np.asarray(imported.normals, dtype=np.float32)
            if normals.shape != positions.shape or not np.isfinite(normals).all():
                raise ValueError(f"Mesh {index} has missing or invalid normals")

            channels = getattr(imported, "texturecoords", None)
            if channels is not None and len(channels):
                channel = np.asarray(channels[0], dtype=np.float32)
                uvs = channel[:, :2] if channel.ndim == 2 and len(channel) == len(positions) else np.zeros((len(positions), 2), dtype=np.float32)
            else:
                uvs = np.zeros((len(positions), 2), dtype=np.float32)

            faces = np.asarray(imported.faces, dtype=np.int64)
            if not faces.size:
                continue
            faces = faces.reshape(-1, 3)
            if np.any(faces < 0) or np.any(faces >= len(positions)):
                raise ValueError(f"Mesh {index} contains an invalid face index")

            packed = np.concatenate((positions, normals, uvs), axis=1)
            expanded = packed[faces.reshape(-1)]
            if not np.isfinite(expanded).all():
                raise ValueError(f"Mesh {index} contains non-finite vertex data")

            output.append(MeshData(
                vertices=np.ascontiguousarray(expanded, dtype=np.float32),
                name=str(getattr(imported, "name", "") or f"mesh_{index}"),
                material_index=int(getattr(imported, "materialindex", 0)),
            ))

    if not output:
        raise ValueError(f"No renderable triangle meshes found in: {source}")
    return output
