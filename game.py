"""Faust prototype game entry point.

Run from the repository root:
    python game.py

This file owns the prototype's window/GPU configuration, scene contents,
and frame loop. Reusable rendering implementation lives in engine/.
"""

from pathlib import Path
import math
import time

import glfw
import moderngl
import numpy as np

from faust import engine


# ---------------------------------------------------------------------------
# Game and GPU configuration
# ---------------------------------------------------------------------------

GAME_CONFIG = {
    "title": "Faust",
    "window_size": (1280, 720),
}

GPU_CONFIG = {
    "vsync": True,
    "max_fps": 144,       # Used when VSync is disabled.
    "msaa_samples": 4,    # GLFW default framebuffer multisampling.
    "clear_color": (0.025, 0.03, 0.045, 1.0),
}

# Environment configuration belongs to the game, not the renderer.
SKYBOX_CONFIG = {
    "path": "assets/sky.exr",
}

# Prototype scene contents. The current PBRRenderer exposes a test cube;
# mesh-file importing and arbitrary mesh instances are future engine work.
SCENE_CONFIG = {
    "camera_position": (0.0, 0.0, 5.0),
    "camera_target": (0.0, 0.0, 0.0),
    "camera_up": (0.0, 1.0, 0.0),
    "field_of_view": 70.0,
    "near_clip": 0.1,
    "far_clip": 100.0,
    "mesh": {
        "type": "prototype_cube",
        "material": {
            "name": "PrototypePaint",
            "base_color": (0.80, 0.12, 0.04),
            "metallic": 0.15,
            "roughness": 0.32,
            "emissive": (0.0, 0.0, 0.0),
        },
        "rotation_speed": 1.0,
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
    glfw.window_hint(
        glfw.SAMPLES,
        max(0, int(GPU_CONFIG["msaa_samples"])),
    )

    width, height = GAME_CONFIG["window_size"]
    window = glfw.create_window(
        width,
        height,
        GAME_CONFIG["title"],
        None,
        None,
    )

    if not window:
        glfw.terminate()
        raise RuntimeError("could not create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1 if GPU_CONFIG["vsync"] else 0)
    return window


def create_scene(context):
    material_config = SCENE_CONFIG["mesh"]["material"]
    material = engine.Material(**material_config)

    pbr_renderer = engine.PBRRenderer(context)
    skybox_path = engine.find_project_asset(*Path(SKYBOX_CONFIG["path"]).parts)
    skybox = engine.Skybox(context, skybox_path)

    return material, pbr_renderer, skybox, skybox_path


def run():
    window = None
    context = None
    resources = []

    try:
        window = create_window()
        context = moderngl.create_context()

        material, pbr_renderer, skybox, skybox_path = create_scene(context)
        resources.extend((pbr_renderer, skybox))

        context.enable(moderngl.DEPTH_TEST)
        context.depth_func = "<"

        eye = np.asarray(SCENE_CONFIG["camera_position"], dtype="f4")
        target = np.asarray(SCENE_CONFIG["camera_target"], dtype="f4")
        up = np.asarray(SCENE_CONFIG["camera_up"], dtype="f4")
        view = engine.look_at(eye, target, up)

        start_time = glfw.get_time()
        frame_interval = 1.0 / max(1, int(GPU_CONFIG["max_fps"]))

        print("Faust initialized")
        print("GPU:", context.info.get("GL_RENDERER", "Unknown"))
        print("OpenGL:", context.info.get("GL_VERSION", "Unknown"))
        print("Skybox:", skybox_path)
        print("VSync:", GPU_CONFIG["vsync"])
        print("MSAA samples requested:", GPU_CONFIG["msaa_samples"])

        while not glfw.window_should_close(window):
            frame_start = time.perf_counter()
            glfw.poll_events()

            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break

            width, height = glfw.get_framebuffer_size(window)
            if width == 0 or height == 0:
                continue

            context.viewport = (0, 0, width, height)
            context.clear(*GPU_CONFIG["clear_color"], depth=1.0)

            projection = engine.perspective(
                math.radians(SCENE_CONFIG["field_of_view"]),
                width / height,
                SCENE_CONFIG["near_clip"],
                SCENE_CONFIG["far_clip"],
            )

            # Draw the environment first, then the scene mesh.
            skybox.render(view, projection)

            mesh_config = SCENE_CONFIG["mesh"]
            elapsed = glfw.get_time() - start_time
            model = engine.rotation_y(
                elapsed * mesh_config["rotation_speed"]
            )

            lighting = SCENE_CONFIG["lighting"]
            pbr_renderer.render_cube(
                model,
                view,
                projection,
                material,
                camera_position=eye,
                light_direction=lighting["direction"],
                light_radiance=lighting["radiance"],
                ambient_radiance=lighting["ambient_radiance"],
            )

            glfw.swap_buffers(window)

            # VSync is the primary limiter when enabled. Otherwise honour
            # the game's explicit max_fps setting.
            if not GPU_CONFIG["vsync"]:
                remaining = frame_interval - (time.perf_counter() - frame_start)
                if remaining > 0:
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
