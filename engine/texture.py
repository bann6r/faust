"""Image loading helpers for Faust-owned GPU textures."""

from pathlib import Path

import moderngl
import numpy as np
from PIL import Image


def load_color_texture(context, path):
    """Load an 8-bit color image as an sRGB GPU texture.

    OpenGL's sRGB internal format performs the color-space decode during
    sampling, keeping the PBR lighting calculations in linear space.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Texture file not found: {source}")
    with Image.open(source) as image:
        pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    height, width, _ = pixels.shape
    texture = context.texture(
        (width, height), 4, pixels.tobytes(),
        alignment=1, internal_format=0x8C43,  # GL_SRGB8_ALPHA8
    )
    texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    texture.build_mipmaps()
    texture.repeat_x = True
    texture.repeat_y = True
    return texture
