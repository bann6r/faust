"""A deliberately small, explicit Vulkan renderer.

The first Vulkan milestone draws a triangle to a GLFW swapchain.  Scene and
asset data remain CPU-side; mesh buffers, depth, descriptors, and PBR are
subsequent backend work.  This module never creates an OpenGL context.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import glfw
import vulkan as vk
from vulkan import ffi


_SHADER_DIR = Path(__file__).resolve().parent.parent / "shaders"
_MAX_TIMEOUT = 0xFFFFFFFFFFFFFFFF


def _decode_name(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class VulkanRenderer:
    """Owns Vulkan objects used to present a single triangle in a GLFW window.

    The application owns the GLFW window and its event loop.  The renderer
    owns the Vulkan instance, surface, device, swapchain, pipeline, command
    buffers, and synchronization objects.
    """

    def __init__(self, window, clear_color=(0.025, 0.03, 0.045, 1.0)):
        color = tuple(float(component) for component in clear_color)
        if len(color) != 4 or any(not 0.0 <= c <= 1.0 for c in color):
            raise ValueError("clear_color must contain four values in [0, 1]")

        self.window = window
        self.clear_color = color
        self.instance = None
        self.surface = None
        self.physical_device = None
        self.device = None
        self.graphics_queue = None
        self.present_queue = None
        self.graphics_family = None
        self.present_family = None
        self.swapchain = None
        self.swapchain_images = []
        self.swapchain_views = []
        self.swapchain_format = None
        self.swapchain_extent = None
        self.render_pass = None
        self.pipeline_layout = None
        self.pipeline = None
        self.framebuffers = []
        self.command_pool = None
        self.command_buffers = []
        self.image_available = None
        self.render_finished = None
        self.frame_fence = None
        self.device_name = "Unknown Vulkan device"
        self._released = False
        self._create_instance()
        try:
            self._create_surface()
            self._select_physical_device()
            self._create_device()
            self._create_command_pool()
            self._create_swapchain_resources()
            self._create_sync_objects()
        except Exception:
            self.release()
            raise

    @staticmethod
    def _instance_function(instance, name):
        function = vk.vkGetInstanceProcAddr(instance, name)
        if function is None:
            raise RuntimeError(f"Vulkan instance function is unavailable: {name}")
        return function

    def _device_function(self, name):
        function = vk.vkGetDeviceProcAddr(self.device, name)
        if function is None:
            raise RuntimeError(f"Vulkan device function is unavailable: {name}")
        return function

    def _create_instance(self):
        extensions = glfw.get_required_instance_extensions()
        if not extensions:
            raise RuntimeError(
                "GLFW returned no Vulkan surface extensions. Check that the "
                "Vulkan loader and a window-system Vulkan driver are installed."
            )
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="Faust",
            applicationVersion=vk.VK_MAKE_VERSION(0, 1, 0),
            pEngineName="Faust",
            engineVersion=vk.VK_MAKE_VERSION(0, 1, 0),
            apiVersion=vk.VK_API_VERSION_1_0,
        )
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions,
            enabledLayerCount=0,
        )
        self.instance = vk.vkCreateInstance(create_info, None)

    def _create_surface(self):
        surface_out = ffi.new("VkSurfaceKHR *")
        result = glfw.create_window_surface(self.instance, self.window, None, surface_out)
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"GLFW could not create a Vulkan surface (VkResult {result})")
        self.surface = surface_out[0]

    def _find_queue_families(self, physical_device):
        surface_support = self._instance_function(
            self.instance, "vkGetPhysicalDeviceSurfaceSupportKHR"
        )
        graphics = None
        present = None
        for index, family in enumerate(
            vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
        ):
            if family.queueCount == 0:
                continue
            if family.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT:
                graphics = index
            if surface_support(physical_device, index, self.surface):
                present = index
            if graphics is not None and present is not None:
                break
        return graphics, present

    def _supports_swapchain(self, physical_device):
        extensions = vk.vkEnumerateDeviceExtensionProperties(physical_device, None)
        available = {_decode_name(item.extensionName) for item in extensions}
        return "VK_KHR_swapchain" in available

    def _has_swapchain_support(self, physical_device):
        get_formats = self._instance_function(
            self.instance, "vkGetPhysicalDeviceSurfaceFormatsKHR"
        )
        get_modes = self._instance_function(
            self.instance, "vkGetPhysicalDeviceSurfacePresentModesKHR"
        )
        return bool(
            get_formats(physical_device, self.surface)
            and get_modes(physical_device, self.surface)
        )

    def _select_physical_device(self):
        devices = vk.vkEnumeratePhysicalDevices(self.instance)
        if not devices:
            raise RuntimeError("No Vulkan physical devices were found")

        candidates = []
        for physical_device in devices:
            graphics, present = self._find_queue_families(physical_device)
            if (
                graphics is not None
                and present is not None
                and self._supports_swapchain(physical_device)
                and self._has_swapchain_support(physical_device)
            ):
                properties = vk.vkGetPhysicalDeviceProperties(physical_device)
                # Prefer a discrete GPU, but accept integrated and software Vulkan devices.
                score = 1 if properties.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU else 0
                candidates.append((score, physical_device, graphics, present, properties))

        if not candidates:
            raise RuntimeError(
                "No Vulkan device supports graphics, presentation, and VK_KHR_swapchain"
            )
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, self.physical_device, self.graphics_family, self.present_family, properties = candidates[0]
        self.device_name = _decode_name(properties.deviceName)

    def _create_device(self):
        unique_families = sorted({self.graphics_family, self.present_family})
        queue_infos = [
            vk.VkDeviceQueueCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex=index,
                queueCount=1,
                pQueuePriorities=[1.0],
            )
            for index in unique_families
        ]
        create_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=len(queue_infos),
            pQueueCreateInfos=queue_infos,
            pEnabledFeatures=[vk.VkPhysicalDeviceFeatures()],
            enabledExtensionCount=1,
            ppEnabledExtensionNames=[vk.VK_KHR_SWAPCHAIN_EXTENSION_NAME],
            enabledLayerCount=0,
        )
        self.device = vk.vkCreateDevice(self.physical_device, create_info, None)
        self.graphics_queue = vk.vkGetDeviceQueue(self.device, self.graphics_family, 0)
        self.present_queue = vk.vkGetDeviceQueue(self.device, self.present_family, 0)

    def _create_command_pool(self):
        create_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=self.graphics_family,
        )
        self.command_pool = vk.vkCreateCommandPool(self.device, create_info, None)

    def _surface_details(self):
        get_capabilities = self._instance_function(
            self.instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"
        )
        get_formats = self._instance_function(
            self.instance, "vkGetPhysicalDeviceSurfaceFormatsKHR"
        )
        capabilities = get_capabilities(self.physical_device, self.surface)
        formats = get_formats(self.physical_device, self.surface)
        if not formats:
            raise RuntimeError("The Vulkan surface exposes no image formats")
        return capabilities, formats

    def _choose_extent(self, capabilities):
        current = capabilities.currentExtent
        if current.width != 0xFFFFFFFF:
            return current
        width, height = glfw.get_framebuffer_size(self.window)
        width = max(capabilities.minImageExtent.width, min(width, capabilities.maxImageExtent.width))
        height = max(capabilities.minImageExtent.height, min(height, capabilities.maxImageExtent.height))
        return vk.VkExtent2D(width=width, height=height)

    def _create_swapchain_resources(self):
        capabilities, formats = self._surface_details()
        surface_format = formats[0]
        preferred = (
            vk.VK_FORMAT_B8G8R8A8_SRGB,
            vk.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        )
        for candidate in formats:
            if (candidate.format, candidate.colorSpace) == preferred:
                surface_format = candidate
                break

        self.swapchain_extent = self._choose_extent(capabilities)
        if self.swapchain_extent.width == 0 or self.swapchain_extent.height == 0:
            raise RuntimeError("Cannot create a Vulkan swapchain for a zero-sized framebuffer")
        if not (capabilities.supportedUsageFlags & vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT):
            raise RuntimeError("The Vulkan surface cannot be used as a color attachment")

        image_count = capabilities.minImageCount + 1
        if capabilities.maxImageCount and image_count > capabilities.maxImageCount:
            image_count = capabilities.maxImageCount

        queue_families = sorted({self.graphics_family, self.present_family})
        sharing_mode = (
            vk.VK_SHARING_MODE_CONCURRENT
            if len(queue_families) > 1
            else vk.VK_SHARING_MODE_EXCLUSIVE
        )
        composite_alpha = next(
            flag
            for flag in (
                vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                vk.VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                vk.VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
                vk.VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
            )
            if capabilities.supportedCompositeAlpha & flag
        )
        create_info = vk.VkSwapchainCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            surface=self.surface,
            minImageCount=image_count,
            imageFormat=surface_format.format,
            imageColorSpace=surface_format.colorSpace,
            imageExtent=self.swapchain_extent,
            imageArrayLayers=1,
            imageUsage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            imageSharingMode=sharing_mode,
            queueFamilyIndexCount=len(queue_families) if len(queue_families) > 1 else 0,
            pQueueFamilyIndices=queue_families if len(queue_families) > 1 else None,
            preTransform=capabilities.currentTransform,
            compositeAlpha=composite_alpha,
            presentMode=vk.VK_PRESENT_MODE_FIFO_KHR,
            clipped=vk.VK_TRUE,
            oldSwapchain=vk.VK_NULL_HANDLE,
        )
        create_swapchain = self._device_function("vkCreateSwapchainKHR")
        self.swapchain = create_swapchain(self.device, create_info, None)
        get_images = self._device_function("vkGetSwapchainImagesKHR")
        self.swapchain_images = list(get_images(self.device, self.swapchain))
        self.swapchain_format = surface_format.format

        self._create_image_views()
        self._create_render_pass()
        self._create_graphics_pipeline()
        self._create_framebuffers()
        self._record_command_buffers()

    def _create_image_views(self):
        components = vk.VkComponentMapping(
            r=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            b=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            a=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
        )
        subresource = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1,
        )
        for image in self.swapchain_images:
            create_info = vk.VkImageViewCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                image=image,
                viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                format=self.swapchain_format,
                components=components,
                subresourceRange=subresource,
            )
            self.swapchain_views.append(vk.vkCreateImageView(self.device, create_info, None))

    def _create_render_pass(self):
        color_attachment = vk.VkAttachmentDescription(
            format=self.swapchain_format,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        )
        color_reference = vk.VkAttachmentReference(
            attachment=0,
            layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        )
        subpass = vk.VkSubpassDescription(
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=[color_reference],
        )
        create_info = vk.VkRenderPassCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            attachmentCount=1,
            pAttachments=[color_attachment],
            subpassCount=1,
            pSubpasses=[subpass],
        )
        self.render_pass = vk.vkCreateRenderPass(self.device, create_info, None)

    @staticmethod
    def _compile_shader(source: Path) -> bytes:
        compiler = shutil.which("glslc")
        if compiler is None:
            raise RuntimeError(
                "Faust needs glslc to compile its GLSL shaders. Install the "
                "Vulkan SDK (or shaderc tools) and make glslc available on PATH."
            )
        with tempfile.TemporaryDirectory(prefix="faust-shaders-") as directory:
            output = Path(directory) / (source.name + ".spv")
            try:
                subprocess.run(
                    [compiler, str(source), "-o", str(output)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                details = (exc.stderr or exc.stdout or str(exc)).strip()
                raise RuntimeError(f"Could not compile {source.name}: {details}") from exc
            return output.read_bytes()

    def _create_shader_module(self, source: Path):
        code = self._compile_shader(source)
        create_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(code),
            pCode=code,
        )
        return vk.vkCreateShaderModule(self.device, create_info, None)

    def _create_graphics_pipeline(self):
        vertex_module = self._create_shader_module(_SHADER_DIR / "triangle.vert")
        fragment_module = None
        try:
            fragment_module = self._create_shader_module(_SHADER_DIR / "triangle.frag")
            stages = [
                vk.VkPipelineShaderStageCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                    module=vertex_module,
                    pName="main",
                ),
                vk.VkPipelineShaderStageCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                    module=fragment_module,
                    pName="main",
                ),
            ]
            vertex_input = vk.VkPipelineVertexInputStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                vertexBindingDescriptionCount=0,
                vertexAttributeDescriptionCount=0,
            )
            assembly = vk.VkPipelineInputAssemblyStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                primitiveRestartEnable=vk.VK_FALSE,
            )
            viewport = vk.VkViewport(
                x=0.0,
                y=0.0,
                width=float(self.swapchain_extent.width),
                height=float(self.swapchain_extent.height),
                minDepth=0.0,
                maxDepth=1.0,
            )
            scissor = vk.VkRect2D(
                offset=vk.VkOffset2D(x=0, y=0),
                extent=self.swapchain_extent,
            )
            viewport_state = vk.VkPipelineViewportStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                viewportCount=1,
                pViewports=[viewport],
                scissorCount=1,
                pScissors=[scissor],
            )
            rasterizer = vk.VkPipelineRasterizationStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                depthClampEnable=vk.VK_FALSE,
                rasterizerDiscardEnable=vk.VK_FALSE,
                polygonMode=vk.VK_POLYGON_MODE_FILL,
                cullMode=vk.VK_CULL_MODE_NONE,
                frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                depthBiasEnable=vk.VK_FALSE,
                lineWidth=1.0,
            )
            multisampling = vk.VkPipelineMultisampleStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                sampleShadingEnable=vk.VK_FALSE,
                rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
            )
            color_blend_attachment = vk.VkPipelineColorBlendAttachmentState(
                blendEnable=vk.VK_FALSE,
                colorWriteMask=(
                    vk.VK_COLOR_COMPONENT_R_BIT
                    | vk.VK_COLOR_COMPONENT_G_BIT
                    | vk.VK_COLOR_COMPONENT_B_BIT
                    | vk.VK_COLOR_COMPONENT_A_BIT
                ),
            )
            color_blending = vk.VkPipelineColorBlendStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                logicOpEnable=vk.VK_FALSE,
                attachmentCount=1,
                pAttachments=[color_blend_attachment],
                blendConstants=[0.0, 0.0, 0.0, 0.0],
            )
            layout_info = vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=0,
                pushConstantRangeCount=0,
            )
            self.pipeline_layout = vk.vkCreatePipelineLayout(self.device, layout_info, None)
            pipeline_info = vk.VkGraphicsPipelineCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                stageCount=len(stages),
                pStages=stages,
                pVertexInputState=vertex_input,
                pInputAssemblyState=assembly,
                pViewportState=viewport_state,
                pRasterizationState=rasterizer,
                pMultisampleState=multisampling,
                pColorBlendState=color_blending,
                layout=self.pipeline_layout,
                renderPass=self.render_pass,
                subpass=0,
                basePipelineHandle=vk.VK_NULL_HANDLE,
                basePipelineIndex=-1,
            )
            self.pipeline = vk.vkCreateGraphicsPipelines(
                self.device, vk.VK_NULL_HANDLE, 1, [pipeline_info], None
            )[0]
        finally:
            if fragment_module is not None:
                vk.vkDestroyShaderModule(self.device, fragment_module, None)
            vk.vkDestroyShaderModule(self.device, vertex_module, None)

    def _create_framebuffers(self):
        for image_view in self.swapchain_views:
            create_info = vk.VkFramebufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                renderPass=self.render_pass,
                attachmentCount=1,
                pAttachments=[image_view],
                width=self.swapchain_extent.width,
                height=self.swapchain_extent.height,
                layers=1,
            )
            self.framebuffers.append(vk.vkCreateFramebuffer(self.device, create_info, None))

    def _record_command_buffers(self):
        allocate_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=len(self.framebuffers),
        )
        self.command_buffers = list(
            vk.vkAllocateCommandBuffers(self.device, allocate_info)
        )
        clear_value = vk.VkClearValue(
            color=vk.VkClearColorValue(float32=list(self.clear_color))
        )
        for index, command_buffer in enumerate(self.command_buffers):
            vk.vkBeginCommandBuffer(
                command_buffer,
                vk.VkCommandBufferBeginInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    flags=0,
                ),
            )
            render_begin = vk.VkRenderPassBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                renderPass=self.render_pass,
                framebuffer=self.framebuffers[index],
                renderArea=vk.VkRect2D(
                    offset=vk.VkOffset2D(x=0, y=0),
                    extent=self.swapchain_extent,
                ),
                clearValueCount=1,
                pClearValues=[clear_value],
            )
            vk.vkCmdBeginRenderPass(
                command_buffer, render_begin, vk.VK_SUBPASS_CONTENTS_INLINE
            )
            vk.vkCmdBindPipeline(
                command_buffer,
                vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                self.pipeline,
            )
            vk.vkCmdDraw(command_buffer, 3, 1, 0, 0)
            vk.vkCmdEndRenderPass(command_buffer)
            vk.vkEndCommandBuffer(command_buffer)

    def _create_sync_objects(self):
        self.image_available = vk.vkCreateSemaphore(
            self.device,
            vk.VkSemaphoreCreateInfo(sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO),
            None,
        )
        self.render_finished = vk.vkCreateSemaphore(
            self.device,
            vk.VkSemaphoreCreateInfo(sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO),
            None,
        )
        self.frame_fence = vk.vkCreateFence(
            self.device,
            vk.VkFenceCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                flags=vk.VK_FENCE_CREATE_SIGNALED_BIT,
            ),
            None,
        )

    @staticmethod
    def _is_swapchain_exception(error):
        return error.__class__.__name__ in {
            "VkErrorOutOfDateKhr",
            "VkSuboptimalKhr",
        }

    def draw_frame(self):
        """Acquire, submit, and present one frame. Safe to call from the GLFW loop."""
        if self._released:
            raise RuntimeError("Vulkan renderer has been released")
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            return False
        if (
            self.swapchain_extent.width != width
            or self.swapchain_extent.height != height
        ):
            self._recreate_swapchain()

        vk.vkWaitForFences(
            self.device, 1, [self.frame_fence], vk.VK_TRUE, _MAX_TIMEOUT
        )
        acquire = self._device_function("vkAcquireNextImageKHR")
        try:
            image_index = acquire(
                self.device,
                self.swapchain,
                _MAX_TIMEOUT,
                self.image_available,
                vk.VK_NULL_HANDLE,
            )
        except Exception as error:
            if self._is_swapchain_exception(error):
                self._recreate_swapchain()
                return False
            raise

        vk.vkResetFences(self.device, 1, [self.frame_fence])
        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            waitSemaphoreCount=1,
            pWaitSemaphores=[self.image_available],
            pWaitDstStageMask=[vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT],
            commandBufferCount=1,
            pCommandBuffers=[self.command_buffers[image_index]],
            signalSemaphoreCount=1,
            pSignalSemaphores=[self.render_finished],
        )
        vk.vkQueueSubmit(
            self.graphics_queue, 1, [submit_info], self.frame_fence
        )
        present_info = vk.VkPresentInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            waitSemaphoreCount=1,
            pWaitSemaphores=[self.render_finished],
            swapchainCount=1,
            pSwapchains=[self.swapchain],
            pImageIndices=[image_index],
        )
        present = self._device_function("vkQueuePresentKHR")
        try:
            present(self.present_queue, present_info)
            # One frame in flight: wait for the presentation wait to consume
            # render_finished before re-signaling this binary semaphore.
            vk.vkQueueWaitIdle(self.present_queue)
        except Exception as error:
            if self._is_swapchain_exception(error):
                self._recreate_swapchain()
                return False
            raise
        return True

    def _destroy_swapchain_resources(self):
        if self.device is None:
            return
        if self.command_buffers and self.command_pool is not None:
            vk.vkFreeCommandBuffers(
                self.device,
                self.command_pool,
                len(self.command_buffers),
                self.command_buffers,
            )
        self.command_buffers = []
        for framebuffer in reversed(self.framebuffers):
            vk.vkDestroyFramebuffer(self.device, framebuffer, None)
        self.framebuffers = []
        if self.pipeline is not None:
            vk.vkDestroyPipeline(self.device, self.pipeline, None)
            self.pipeline = None
        if self.pipeline_layout is not None:
            vk.vkDestroyPipelineLayout(self.device, self.pipeline_layout, None)
            self.pipeline_layout = None
        if self.render_pass is not None:
            vk.vkDestroyRenderPass(self.device, self.render_pass, None)
            self.render_pass = None
        for image_view in reversed(self.swapchain_views):
            vk.vkDestroyImageView(self.device, image_view, None)
        self.swapchain_views = []
        self.swapchain_images = []

    def _destroy_swapchain(self):
        if self.device is not None and self.swapchain is not None:
            self._device_function("vkDestroySwapchainKHR")(
                self.device, self.swapchain, None
            )
        self.swapchain = None
        self.swapchain_extent = None

    def _recreate_swapchain(self):
        while True:
            width, height = glfw.get_framebuffer_size(self.window)
            if width > 0 and height > 0:
                break
            glfw.wait_events()
        vk.vkDeviceWaitIdle(self.device)
        self._destroy_swapchain_resources()
        self._destroy_swapchain()
        self._create_swapchain_resources()

    def release(self):
        """Release all owned Vulkan resources. Safe after partial initialization."""
        if self._released:
            return
        if self.device is not None:
            vk.vkDeviceWaitIdle(self.device)
            self._destroy_swapchain_resources()
            self._destroy_swapchain()
            for resource_name in ("frame_fence", "render_finished", "image_available"):
                resource = getattr(self, resource_name)
                if resource is not None:
                    if resource_name == "frame_fence":
                        vk.vkDestroyFence(self.device, resource, None)
                    else:
                        vk.vkDestroySemaphore(self.device, resource, None)
                    setattr(self, resource_name, None)
            if self.command_pool is not None:
                vk.vkDestroyCommandPool(self.device, self.command_pool, None)
                self.command_pool = None
            vk.vkDestroyDevice(self.device, None)
            self.device = None
        if self.instance is not None and self.surface is not None:
            destroy_surface = self._instance_function(self.instance, "vkDestroySurfaceKHR")
            destroy_surface(self.instance, self.surface, None)
            self.surface = None
        if self.instance is not None:
            vk.vkDestroyInstance(self.instance, None)
            self.instance = None
        self._released = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False
