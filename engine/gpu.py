"""OpenGL capability inspection and minimum-version validation."""

from dataclasses import dataclass
import re


@dataclass(frozen=True, slots=True)
class GPUInfo:
    vendor: str
    renderer: str
    version: str
    version_code: int
    shading_language: str
    max_texture_size: int
    max_samples: int


def inspect_context(context):
    """Return stable, printable GPU information from a ModernGL context."""
    info = getattr(context, "info", {}) or {}

    def read(key, default="Unknown"):
        value = info.get(key, default)
        return str(value) if value is not None else default

    version_code = int(getattr(context, "version_code", 0) or 0)
    version = read("GL_VERSION")
    if version_code <= 0:
        match = re.search(r"(\d+)\.(\d+)", version)
        if match:
            version_code = int(match.group(1)) * 100 + int(match.group(2)) * 10

    def integer(key):
        try:
            return int(info.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0

    return GPUInfo(
        vendor=read("GL_VENDOR"),
        renderer=read("GL_RENDERER"),
        version=version,
        version_code=version_code,
        shading_language=read("GL_SHADING_LANGUAGE_VERSION"),
        max_texture_size=integer("GL_MAX_TEXTURE_SIZE"),
        max_samples=integer("GL_MAX_SAMPLES"),
    )


def validate_context(context, minimum=(3, 3)):
    """Require the requested OpenGL version and return GPUInfo."""
    if len(minimum) != 2 or any(int(part) < 0 for part in minimum):
        raise ValueError("minimum must be a (major, minor) version pair")
    required = int(minimum[0]) * 100 + int(minimum[1]) * 10
    details = inspect_context(context)
    if details.version_code < required:
        major, minor = minimum
        raise RuntimeError(
            f"Faust requires OpenGL {major}.{minor}+; "
            f"the active context reports {details.version!r}"
        )
    return details
