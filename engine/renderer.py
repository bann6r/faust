"""Public renderer entry point.

The OpenGL renderer has been retired. Renderer now names the Vulkan backend.
"""

from .vulkan import VulkanRenderer

Renderer = VulkanRenderer

__all__ = ["Renderer", "VulkanRenderer"]
