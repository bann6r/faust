"""Faust-owned CPU and GPU mesh representations."""

from dataclasses import dataclass

import moderngl
import numpy as np


@dataclass(frozen=True, slots=True)
class MeshData:
    """CPU-side triangle data independent of the source file importer."""

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


class Mesh:
    """GPU buffers for a Faust MeshData object."""

    def __init__(self, context, data: MeshData):
        if not isinstance(data, MeshData):
            raise TypeError("data must be a Faust MeshData")
        self.context = context
        self.name = data.name
        self.vertex_count = len(data.vertices)
        self.vertex_buffer = context.buffer(data.vertices.tobytes())
        self._vertex_arrays = {}

    def vertex_array(self, program):
        key = id(program)
        if key not in self._vertex_arrays:
            self._vertex_arrays[key] = self.context.vertex_array(
                program,
                [(self.vertex_buffer, "3f 3f 2f", "in_position", "in_normal", "in_uv")],
            )
        return self._vertex_arrays[key]

    def render(self, program):
        self.vertex_array(program).render(mode=moderngl.TRIANGLES)

    def release(self):
        for vertex_array in self._vertex_arrays.values():
            vertex_array.release()
        self._vertex_arrays.clear()
        self.vertex_buffer.release()
