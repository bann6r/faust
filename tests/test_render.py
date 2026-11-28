"""GPU smoke test for Faust's OpenGL renderer.

Run with: python -m pytest tests/test_render.py -q
A machine without a headless OpenGL 3.3 driver skips this test cleanly.
"""
import numpy as np
import pytest


def test_standalone_pbr_render_produces_pixels():
    moderngl=pytest.importorskip("moderngl")
    from engine.hdr import Camera, DirectionalLight, Material, Mesh, HDRRenderer as Renderer, Scene, sphere_data

    try:
        ctx=moderngl.create_standalone_context()
    except Exception as exc:
        pytest.skip(f"standalone OpenGL context unavailable: {exc}")
    renderer=mesh=target=None
    try:
        mesh=Mesh(ctx,sphere_data(24,16))
        scene=Scene(DirectionalLight(direction=(-0.4,-1.0,-0.3),radiance=(4,4,4)))
        scene.add(mesh,Material("test",(0.8,0.12,0.06),0.0,0.32))
        camera=Camera(position=(0,0,4),target=(0,0,0),aspect=1.0)
        color=ctx.texture((96,96),4,dtype="f1")
        depth=ctx.depth_renderbuffer((96,96))
        target=ctx.framebuffer([color],depth)
        renderer=Renderer(ctx,(96,96),shadow_size=256,bloom=False)
        renderer.render(scene,camera,target_fbo=target)
        assert ctx.viewport == (0, 0, 96, 96)
        assert renderer.bloom_a is None and renderer.bloom_b is None
        pixels=np.frombuffer(color.read(alignment=1),dtype=np.uint8).reshape(96,96,4)
        assert pixels[:,:,:3].max()>0
        assert np.count_nonzero(pixels[:,:,:3])>96*96
    finally:
        if renderer: renderer.release()
        if mesh: mesh.release()
        if target: target.release()
        ctx.release()
