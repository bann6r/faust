"""Faust's first Vulkan sample: open a window and draw a triangle."""

import glfw

from faust import engine


GAME_CONFIG = {
    "title": "Faust | Vulkan",
    "window_size": (1280, 720),
}
CLEAR_COLOR = (0.025, 0.03, 0.045, 1.0)


def create_window():
    if not glfw.init():
        raise RuntimeError("could not initialize GLFW")
    if not glfw.vulkan_supported():
        glfw.terminate()
        raise RuntimeError(
            "GLFW could not find a Vulkan loader/driver. Install a Vulkan "
            "runtime and a graphics driver that supports Vulkan."
        )

    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
    width, height = GAME_CONFIG["window_size"]
    window = glfw.create_window(width, height, GAME_CONFIG["title"], None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("could not create the GLFW Vulkan window")
    return window


def run():
    window = create_window()
    renderer = None
    try:
        renderer = engine.VulkanRenderer(window, clear_color=CLEAR_COLOR)
        print("Faust initialized")
        print("Vulkan device:", renderer.device_name)
        print("Renderer: Vulkan 1.0 swapchain + graphics pipeline")
        print("Sample: indexed-free triangle (gl_VertexIndex)")

        while not glfw.window_should_close(window):
            glfw.poll_events()
            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break
            renderer.draw_frame()
    finally:
        if renderer is not None:
            renderer.release()
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    run()
