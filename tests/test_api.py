def test_package_root_exposes_camera_and_scene_api():
    from engine import Camera, DirectionalLight, Material, Scene

    assert Camera is not None
    assert DirectionalLight is not None
    assert Material is not None
    assert len(Scene()) == 0
