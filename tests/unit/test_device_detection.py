"""Tests for GPU/device detection utilities in stg.gpu_utils.

Unit tests only -- no model training. GPU-specific tests are skipped
when the corresponding hardware is unavailable.
"""

import pytest
import torch

from stg.gpu_utils import detect_best_device, get_device_info, is_gpu_available

HAS_CUDA = torch.cuda.is_available()
HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def test_detect_best_device_returns_valid_type():
    device = detect_best_device()
    assert device.type in ("cuda", "mps", "cpu")


def test_get_device_info_has_type():
    info = get_device_info()
    assert "type" in info
    assert info["type"] in ("cuda", "mps", "cpu")


def test_is_gpu_available_returns_bool():
    result = is_gpu_available()
    assert isinstance(result, bool)


def test_device_detection_consistency():
    """All three detection functions must agree on GPU availability."""
    device = detect_best_device()
    info = get_device_info()
    gpu_available = is_gpu_available()

    if gpu_available:
        assert device.type in ("cuda", "mps")
        assert info["type"] in ("cuda", "mps")
    else:
        assert device.type == "cpu"
        assert info["type"] == "cpu"


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
@pytest.mark.cuda
class TestCudaDeviceInfo:
    def test_cuda_device_info_fields(self):
        info = get_device_info()
        assert info["type"] == "cuda"
        for key in ("name", "memory_gb", "compute_capability", "device_count", "cuda_version"):
            assert key in info, f"Missing key: {key}"

    def test_detect_best_device_is_cuda(self):
        assert detect_best_device().type == "cuda"


@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
class TestMpsDeviceInfo:
    def test_mps_device_info(self):
        info = get_device_info()
        assert info["type"] == "mps"
        assert info["name"] == "Apple Silicon GPU"
