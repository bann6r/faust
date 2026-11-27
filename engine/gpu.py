"""Vulkan physical-device metadata helpers."""

from dataclasses import dataclass


def _decode_name(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _api_version_tuple(version):
    return (
        (int(version) >> 22) & 0x7F,
        (int(version) >> 12) & 0x3FF,
        int(version) & 0xFFF,
    )


@dataclass(frozen=True, slots=True)
class GPUInfo:
    device_name: str
    vendor_id: int
    device_id: int
    device_type: int
    api_version: tuple[int, int, int]
    driver_version: int


def inspect_device_properties(properties):
    """Convert VkPhysicalDeviceProperties into stable Python values."""
    return GPUInfo(
        device_name=_decode_name(properties.deviceName),
        vendor_id=int(properties.vendorID),
        device_id=int(properties.deviceID),
        device_type=int(properties.deviceType),
        api_version=_api_version_tuple(properties.apiVersion),
        driver_version=int(properties.driverVersion),
    )
