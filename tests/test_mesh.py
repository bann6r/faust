import numpy as np
import pytest

from engine.mesh import MeshData


def test_mesh_data_accepts_triangle_vertex_stream():
    vertices = np.zeros((3, 8), dtype=np.float32)
    mesh = MeshData(vertices=vertices, name="triangle", material_index=2)

    assert mesh.vertices.shape == (3, 8)
    assert mesh.name == "triangle"
    assert mesh.material_index == 2
    assert mesh.vertices.flags.c_contiguous


@pytest.mark.parametrize(
    "vertices",
    [
        np.zeros((3, 7), dtype=np.float32),
        np.zeros((4, 8), dtype=np.float32),
        np.zeros((0, 8), dtype=np.float32),
    ],
)
def test_mesh_data_rejects_invalid_vertex_stream(vertices):
    with pytest.raises(ValueError):
        MeshData(vertices=vertices)


def test_mesh_data_rejects_non_finite_values():
    vertices = np.zeros((3, 8), dtype=np.float32)
    vertices[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        MeshData(vertices=vertices)
