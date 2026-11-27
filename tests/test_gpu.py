from types import SimpleNamespace

from engine.gpu import inspect_device_properties


def test_inspect_device_properties_reads_vulkan_metadata():
    properties = SimpleNamespace(
        deviceName=b"Test Vulkan Device",
        vendorID=0x1234,
        deviceID=0x5678,
        deviceType=2,
        apiVersion=(1 << 22) | (3 << 12) | 280,
        driverVersion=99,
    )

    info = inspect_device_properties(properties)

    assert info.device_name == "Test Vulkan Device"
    assert info.vendor_id == 0x1234
    assert info.device_id == 0x5678
    assert info.device_type == 2
    assert info.api_version == (1, 3, 280)
    assert info.driver_version == 99
