import pytest

from engine.gpu import inspect_context, validate_context


class FakeContext:
    version_code = 330
    info = {
        "GL_VENDOR": "Test Vendor",
        "GL_RENDERER": "Test Renderer",
        "GL_VERSION": "3.3 Test",
        "GL_SHADING_LANGUAGE_VERSION": "3.30",
        "GL_MAX_TEXTURE_SIZE": 8192,
        "GL_MAX_SAMPLES": 8,
    }


def test_inspect_context_reads_capabilities():
    info = inspect_context(FakeContext())
    assert info.version_code == 330
    assert info.max_texture_size == 8192
    assert info.renderer == "Test Renderer"


def test_validate_context_accepts_minimum_and_rejects_old_version():
    assert validate_context(FakeContext()).version_code == 330

    class OldContext:
        version_code = 320
        info = {"GL_VERSION": "3.2"}

    with pytest.raises(RuntimeError, match="requires OpenGL"):
        validate_context(OldContext())
