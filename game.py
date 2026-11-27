"""Faust sample application.

Run from the repository root with: python game.py
The sample owns GLFW/window lifecycle; Faust owns scene and render submission.
"""

from pathlib import Path
import time

import glfw
import moderngl

from faust import engine


GAME_CONFIG = {
    "title": "Faust",
    "window_size": (1280, 720),
}

GPU_CONFIG = {
    "vsync": True,
    "max_fps": 144,
    "msaa_samples": 4,
    "clear_color": (0.025, 0.03, 0.045, 1.0),
}

SKYBOX_CONFIG = {
    "enabled": True,
    "path": "assets/sky.exr",
}

SCENE_CONFIG = {
    "camera_position": (0.0, 0.0, 5.0),
    "camera_target": (0.0, 0.0, 0.0),
    "camera_up": (0.0, 1.0, 0.0),
    "field_of_view": 70.0,
    "near_clip": 0.1,
    "far_clip": 100.0,
    "mesh": {
        "path": "assets/multi_cleaner_bottle_4k.fbx",
        "base_color_texture": "assets/textures/multi_cleaner_bottle_diff_4k.jpg",
        "material": {
            "name": "CleanerBottle",
            "base_color": (1.0, 1.0, 1.0),
            "metallic": 0.0,
            "roughness": 0.5,
            "emissive": (0.0, 0.0, 0.0),
        },
        "rotation_speed": 0.35,
    },
    "lighting": {
        "direction": (0.45, 0.8, 0.35),
        "radiance": (4.0, 3.8, 3.5),
        "ambient_radiance": (0.10, 0.12, 0.16),
    },
}


def create_window():
    if not glfw.init():
        raise RuntimeError("could not initialize GLFW")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.SAMPLES, max(0, int(GPU_CONFIG["msaa_samples"])))

    width, height = GAME_CONFIG["window_size"]
    window = glfw.create_window(width, height, GAME_CONFIG["title"], None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("could not create an OpenGL 3.3 window")

    glfw.make_context_current(window)
    glfw.swap_interval(1 if GPU_CONFIG["vsync"] else 0)
    return window


def create_scene(context):
    """Create the demo's CPU scene and GPU resources with rollback on failure."""
    allocated = []
    try:
        mesh_config = SCENE_CONFIG["mesh"]
        material = engine.Material(**mesh_config["material"])
        model_path = engine.find_project_asset(*Path(mesh_config["path"]).parts)
        texture_path = engine.find_project_asset(
            *Path(mesh_config["base_color_texture"]).parts
        )

        mesh_data = engine.load_meshes(model_path)
        meshes = [engine.Mesh(context, item) for item in mesh_data]
        allocated.extend(meshes)

        base_color_texture = engine.load_color_texture(context, texture_path)
        allocated.append(base_color_texture)

        light = engine.DirectionalLight(**SCENE_CONFIG["lighting"])
        scene = engine.Scene(light=light)
        for mesh in meshes:
            scene.add(mesh, material, base_color_texture=base_color_texture)

        renderer = engine.Renderer(context, clear_color=GPU_CONFIG["clear_color"])
        allocated.append(renderer)

        skybox = None
        skybox_path = None
        if SKYBOX_CONFIG["enabled"]:
            skybox_path = engine.find_project_asset(
                *Path(SKYBOX_CONFIG["path"]).parts
            )
            try:
                skybox = engine.Skybox(context, skybox_path)
                allocated.append(skybox)
            except (FileNotFoundError, ImportError, ValueError, RuntimeError) as exc:
                print(f"Faust: skybox disabled ({exc})")

        return scene, renderer, skybox, allocated, model_path, skybox_path
    except Exception:
        for resource in reversed(allocated):
            try:
                resource.release()
            except Exception:
                pass
        raise


def run():
    window = None
    context = None
    resources = []
    try:
        window = create_window()
        context = moderngl.create_context()
        gpu = engine.validate_context(context, minimum=(3, 3))

        scene, renderer, skybox, resources, model_path, skybox_path = create_scene(context)
        camera = engine.Camera(
            position=SCENE_CONFIG["camera_position"],
            target=SCENE_CONFIG["camera_target"],
            up=SCENE_CONFIG["camera_up"],
            fov_y_degrees=SCENE_CONFIG["field_of_view"],
            near=SCENE_CONFIG["near_clip"],
            far=SCENE_CONFIG["far_clip"],
        )

        context.enable(moderngl.DEPTH_TEST)
        context.depth_func = "<"
        start_time = glfw.get_time()
        frame_interval = 1.0 / max(1, int(GPU_CONFIG["max_fps"]))

        print("Faust initialized")
        print("GPU:", gpu.renderer)
        print("OpenGL:", gpu.version)
        print("Model:", model_path)
        print("Imported mesh instances:", len(scene))
        print("Skybox:", skybox_path if skybox else "disabled")
        print("VSync:", GPU_CONFIG["vsync"])

        while not glfw.window_should_close(window):
            frame_start = time.perf_counter()
            glfw.poll_events()
            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break

            width, height = glfw.get_framebuffer_size(window)
            if width <= 0 or height <= 0:
                continue

            camera.resize(width, height)
            context.viewport = (0, 0, width, height)
            context.clear(*GPU_CONFIG["clear_color"], depth=1.0)

            if skybox is not None:
                skybox.render(camera.view_matrix, camera.projection_matrix)

            elapsed = glfw.get_time() - start_time
            model = engine.rotation_y(
                elapsed * SCENE_CONFIG["mesh"]["rotation_speed"]
            )
            for item in scene.objects:
                item.model = model

            # The frame was cleared and the optional background was drawn above.
            renderer.render(scene, camera, clear=False)
            glfw.swap_buffers(window)

            if not GPU_CONFIG["vsync"]:
                remaining = frame_interval - (time.perf_counter() - frame_start)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        for resource in reversed(resources):
            try:
                resource.release()
            except Exception:
                pass
        if context is not None:
            context.release()
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    run()
