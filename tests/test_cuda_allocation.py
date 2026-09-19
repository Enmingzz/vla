"""Driver failures must be detected before any expensive model startup."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location("cuda_probe", Path(__file__).resolve().parents[1] / "scripts/check_cuda_allocation.py")
cuda_probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cuda_probe)


def driver(count=1, name=b"NVIDIA H100 80GB HBM3", gib=80):
    def output(pointer, value):
        pointer._obj.value = value
        return 0
    def device_name(buffer, size, device):
        buffer.value = name
        return 0
    return Mock(cuInit=Mock(return_value=0), cuDeviceGetCount=Mock(side_effect=lambda p: output(p, count)),
        cuDeviceGet=Mock(side_effect=lambda p, ordinal: output(p, ordinal)),
        cuDeviceGetName=Mock(side_effect=device_name),
        cuDeviceTotalMem_v2=Mock(side_effect=lambda p, device: output(p, gib * 2**30)))


def test_valid_single_full_h100_and_reported_capacity():
    result = cuda_probe.probe(driver())
    assert result["visible_device_count"] == 1
    assert result["total_memory_bytes"] == 80 * 2**30


def test_no_device_error_stops_before_device_queries():
    allocated = driver()
    allocated.cuInit.return_value = 100
    with pytest.raises(RuntimeError, match="cuInit failed with CUDA driver status 100"):
        cuda_probe.probe(allocated)
    allocated.cuDeviceGetCount.assert_not_called()


@pytest.mark.parametrize("arguments", [dict(count=0), dict(count=2), dict(gib=40), dict(name=b"NVIDIA A100")])
def test_unexpected_device_allocation_is_rejected(arguments):
    with pytest.raises(RuntimeError):
        cuda_probe.probe(driver(**arguments))
