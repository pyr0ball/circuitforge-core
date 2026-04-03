# tests/test_hardware/test_detect.py
"""Tests for hardware auto-detection (subprocess mocked throughout)."""
import json
from unittest.mock import MagicMock, patch

import pytest

from circuitforge_core.hardware.detect import detect_hardware, detect_hardware_json
from circuitforge_core.hardware.models import HardwareSpec


_NVIDIA_SMI_OUTPUT = "NVIDIA GeForce RTX 4080, 16376\n"
_ROCM_SMI_JSON = json.dumps({
    "card0": {
        "Card series": "Radeon RX 7900 XTX",
        "VRAM Total Memory (B)": str(24 * 1024 * 1024 * 1024),
    }
})
_SYSTEM_PROFILER_JSON = json.dumps({
    "SPHardwareDataType": [{
        "chip_type": "Apple M2 Pro",
        "physical_memory": "32 GB",
    }]
})


def _mock_run(outputs: dict[str, str]):
    """Return a _run replacement that maps first arg → output."""
    def fake_run(*args, **kwargs):
        cmd = args[0] if args else ""
        return outputs.get(cmd, "")
    return fake_run


class TestDetectNvidia:
    def test_returns_nvidia_spec(self):
        with patch("circuitforge_core.hardware.detect._run",
                   side_effect=lambda *a, **kw: _NVIDIA_SMI_OUTPUT if "query-gpu" in " ".join(a) else ""), \
             patch("circuitforge_core.hardware.detect._ram_mb", return_value=32768):
            spec = detect_hardware()
        assert spec.gpu_vendor == "nvidia"
        assert spec.vram_mb == 16376
        assert spec.gpu_name == "NVIDIA GeForce RTX 4080"
        assert spec.ram_mb == 32768

    def test_gpu_count_from_line_count(self):
        dual_gpu = "RTX 4080, 16376\nRTX 3090, 24576\n"
        with patch("circuitforge_core.hardware.detect._run",
                   side_effect=lambda *a, **kw: dual_gpu if "query-gpu" in " ".join(a) else ""), \
             patch("circuitforge_core.hardware.detect._ram_mb", return_value=65536):
            spec = detect_hardware()
        assert spec.gpu_count == 2
        assert spec.vram_mb == 16376  # first GPU


class TestDetectAmd:
    def test_returns_amd_spec_when_nvidia_absent(self):
        with patch("circuitforge_core.hardware.detect._run",
                   side_effect=lambda *a, **kw:
                       "" if "nvidia" in a[0] else
                       _ROCM_SMI_JSON if "rocm-smi" in a[0] else ""), \
             patch("circuitforge_core.hardware.detect._ram_mb", return_value=65536):
            spec = detect_hardware()
        assert spec.gpu_vendor == "amd"
        assert spec.vram_mb == 24576
        assert "7900" in spec.gpu_name


class TestDetectApple:
    def test_returns_apple_spec_on_macos(self):
        with patch("platform.system", return_value="Darwin"), \
             patch("circuitforge_core.hardware.detect._run",
                   side_effect=lambda *a, **kw:
                       "1" if "arm64" in " ".join(a) else
                       _SYSTEM_PROFILER_JSON if "SPHardware" in " ".join(a) else ""), \
             patch("circuitforge_core.hardware.detect._ram_mb", return_value=32768):
            spec = detect_hardware()
        assert spec.gpu_vendor == "apple"
        assert spec.vram_mb == 32768  # 32 GB unified
        assert "M2" in spec.gpu_name


class TestCpuFallback:
    def test_cpu_fallback_when_no_gpu_detected(self):
        with patch("circuitforge_core.hardware.detect._run", return_value=""), \
             patch("platform.system", return_value="Linux"), \
             patch("circuitforge_core.hardware.detect._ram_mb", return_value=16384):
            spec = detect_hardware()
        assert spec.gpu_vendor == "cpu"
        assert spec.vram_mb == 0
        assert spec.gpu_count == 0

    def test_never_raises(self):
        with patch("circuitforge_core.hardware.detect._run", side_effect=RuntimeError("boom")), \
             patch("circuitforge_core.hardware.detect._ram_mb", return_value=0):
            spec = detect_hardware()
        assert isinstance(spec, HardwareSpec)


class TestDetectHardwareJson:
    def test_returns_valid_json(self):
        with patch("circuitforge_core.hardware.detect._run", return_value=""), \
             patch("circuitforge_core.hardware.detect._ram_mb", return_value=8192):
            out = detect_hardware_json()
        data = json.loads(out)
        assert "vram_mb" in data
        assert "gpu_vendor" in data
