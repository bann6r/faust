"""CPU-side mesh data used by the importer and future Vulkan buffer uploads."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MeshData:
    """Triangle data independent of a graphics API.

    Each vertex contains position(3), normal(3), and UV0(2) float32 values.
    GPU buffers are intentionally not created here; the Vulkan backend will
    own their allocation and lifetime.
    """

    vertices: np.ndarray
    name: str = "mesh"
    material_index: int = 0

    def __post_init__(self):
        data = np.asarray(self.vertices, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != 8:
            raise ValueError("vertices must have shape (N, 8): position, normal, UV")
        if len(data) == 0 or len(data) % 3:
            raise ValueError("mesh must contain a non-empty triangle vertex list")
        if not np.isfinite(data).all():
            raise ValueError("mesh vertices must be finite")
        object.__setattr__(self, "vertices", np.ascontiguousarray(data))
