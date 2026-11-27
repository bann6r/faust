def test_package_root_exposes_cpu_scene_api_without_gpu_initialization():
    from engine import Camera, DirectionalLight, Material, MeshData, Scene

    assert Camera is not None
    assert DirectionalLight is not None
    assert Material is not None
    assert MeshData is not None
    assert len(Scene()) == 0
