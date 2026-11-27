import math

import numpy as np
import pytest

from engine.camera import Camera, look_at, perspective, rotation_y


def test_perspective_matrix_has_expected_shape_and_values():
    matrix = perspective(math.radians(90), 2.0, 0.1, 100.0)
    assert matrix.shape == (4, 4)
    assert matrix.dtype == np.float32
    assert matrix[0, 0] == pytest.approx(0.5)
    assert matrix[1, 1] == pytest.approx(1.0)
    assert matrix[3, 2] == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "args",
    [
        (0.0, 1.0, 0.1, 10.0),
        (math.pi, 1.0, 0.1, 10.0),
        (1.0, 0.0, 0.1, 10.0),
        (1.0, 1.0, 0.0, 10.0),
        (1.0, 1.0, 10.0, 1.0),
    ],
)
def test_perspective_rejects_invalid_parameters(args):
    with pytest.raises(ValueError):
        perspective(*args)


def test_look_at_handles_up_vector_parallel_to_view():
    matrix = look_at((0, 0, 0), (0, 1, 0), (0, 1, 0))
    assert np.isfinite(matrix).all()
    assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1.0)


def test_camera_resize_and_projection():
    camera = Camera()
    assert camera.resize(1920, 1080)
    assert camera.aspect == pytest.approx(16 / 9)
    assert not camera.resize(0, 0)
    assert camera.projection_matrix.shape == (4, 4)


def test_rotation_y_is_orthonormal():
    matrix = rotation_y(0.7)
    rotation = matrix[:3, :3]
    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-6)
