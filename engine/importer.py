"""FBX/mesh import boundary. Assimp parses files; Faust owns the resulting data."""

from contextlib import contextmanager
import ctypes
import ctypes.util
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    from .mesh import MeshData
except ImportError:  # Supports direct execution of engine/__init__.py.
    from mesh import MeshData


# Keep the Windows DLL search-directory handle alive for the process lifetime.
_ASSIMP_DLL_DIRECTORY_HANDLE = None
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ASSIMP_DLL = _PROJECT_ROOT / "assimp-vc145-mt.dll"
_DOWNLOADS_ASSIMP_DLL = (
    Path.home()
    / "Downloads"
    / "windows-x64-v6.0.6"
    / "Release"
    / "assimp-vc145-mt.dll"
)


def _find_assimp_dll():
    """Resolve the native Assimp DLL without requiring a global PATH edit.

    ASSIMP_LIBRARY_DIR may point either to the DLL itself or to its containing
    directory. The user's local Windows installation is the default fallback.
    """
    configured = os.environ.get("ASSIMP_LIBRARY_DIR")
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_file():
            candidate = configured_path
        elif configured_path.is_dir():
            candidates = sorted(configured_path.glob("assimp*.dll"))
            candidate = candidates[0] if candidates else None
        else:
            candidate = None
        if candidate is None or not candidate.is_file():
            raise FileNotFoundError(
                "ASSIMP_LIBRARY_DIR does not contain an Assimp DLL: "
                f"{configured_path}"
            )
        return candidate.resolve()

    # Prefer a DLL committed/copied into the Faust repository root.
    if _PROJECT_ASSIMP_DLL.is_file():
        return _PROJECT_ASSIMP_DLL.resolve()

    if _DOWNLOADS_ASSIMP_DLL.is_file():
        return _DOWNLOADS_ASSIMP_DLL.resolve()

    candidates = sorted(_DOWNLOADS_ASSIMP_DLL.parent.glob("assimp*.dll"))
    if candidates:
        return candidates[0].resolve()
    return None


def _load_pyassimp():
    """Load PyAssimp while directing its native-library lookup to our DLL."""
    global _ASSIMP_DLL_DIRECTORY_HANDLE

    if os.name != "nt":
        try:
            import pyassimp
            from pyassimp import postprocess
            return pyassimp, postprocess
        except ImportError as exc:
            raise ImportError(
                "FBX import requires PyAssimp and the native Assimp library. "
                "See docs/asset_importing.md."
            ) from exc

    try:
        dll_path = _find_assimp_dll()
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc

    if dll_path is None:
        raise RuntimeError(
            "Faust could not find the Assimp DLL. Expected "
            f"{_PROJECT_ASSIMP_DLL} (or {_DOWNLOADS_ASSIMP_DLL}), "
            "or set ASSIMP_LIBRARY_DIR to the DLL "
            "file or its containing directory."
        )

    if _ASSIMP_DLL_DIRECTORY_HANDLE is None:
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None:
            _ASSIMP_DLL_DIRECTORY_HANDLE = add_dll_directory(str(dll_path.parent))

    # PyAssimp versions differ: some call ctypes.CDLL directly, while others
    # use ctypes.util.find_library. Redirect either lookup only while PyAssimp
    # initializes, and restore the standard ctypes functions immediately.
    original_cdll = ctypes.CDLL
    original_find_library = ctypes.util.find_library

    def assimp_cdll(name, *args, **kwargs):
        try:
            requested = os.fsdecode(name)
        except (TypeError, ValueError):
            requested = ""
        if "assimp" in Path(requested).name.lower():
            name = str(dll_path)
        return original_cdll(name, *args, **kwargs)

    def assimp_find_library(name):
        if "assimp" in str(name).lower():
            return str(dll_path)
        return original_find_library(name)

    try:
        with patch.object(ctypes, "CDLL", assimp_cdll), patch.object(
            ctypes.util, "find_library", assimp_find_library
        ):
            import pyassimp
            from pyassimp import postprocess
    except ImportError as exc:
        raise ImportError(
            "FBX import requires PyAssimp. Install the Python package with "
            "python -m pip install pyassimp."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Faust found {dll_path}, but Windows could not load it. "
            "Check that the DLL matches this Python interpreter's "
            "architecture and that its Visual C++ runtime dependencies "
            f"are available. Original error: {exc}"
        ) from exc

    return pyassimp, postprocess


@contextmanager
def _open_scene(path):
    pyassimp, postprocess = _load_pyassimp()

    flags = (
        postprocess.aiProcess_Triangulate
        | postprocess.aiProcess_GenSmoothNormals
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
