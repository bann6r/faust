import numpy as np
import pytest

from engine.material import Material
from engine.scene import DirectionalLight, RenderObject, Scene


class FakeMesh:
    def render(self, program):
        pass


def test_scene_adds_and_removes_render_objects():
    scene = Scene()
    item = scene.add(FakeMesh(), Material())
    assert len(scene) == 1
    assert item.model == pytest.approx(np.eye(4))
    scene.remove(item)
    assert len(scene) == 0


def test_render_object_rejects_invalid_model():
    with pytest.raises(ValueError):
        RenderObject(FakeMesh(), Material(), np.zeros((3, 3)))


def test_directional_light_normalizes_direction():
    light = DirectionalLight(direction=(0, 4, 0))
    assert light.direction == pytest.approx((0, 1, 0))


def test_directional_light_rejects_zero_direction():
    with pytest.raises(ValueError):
        DirectionalLight(direction=(0, 0, 0))
