# circuitforge_core/hardware/detect.py
"""
Cross-platform hardware auto-detection.

Reads GPU info from:
  - nvidia-smi     (NVIDIA, Linux/Windows)
  - rocm-smi       (AMD, Linux)
  - system_profiler (Apple Silicon, macOS)
  - /proc/meminfo  (Linux RAM)
  - psutil         (cross-platform RAM fallback)

Returns a HardwareSpec. On detection failure, returns a conservative
CPU-only spec so callers always get a usable result.
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from pathlib import Path

from .models import HardwareSpec


def _run(*args: str, timeout: int = 5) -> str:
    """Run a subprocess and return stdout, or empty string on any error."""
    try:
        result = subprocess.run(
            list(args), capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _ram_mb() -> int:
    """Return total system RAM in MB."""
    # psutil is optional but preferred
    try:
        import psutil  # type: ignore[import-untyped]
        return psutil.virtual_memory().total // (1024 * 1024)
    except ImportError:
        pass

    # Linux /proc/meminfo fallback
    mem_info = Path("/proc/meminfo")
    if mem_info.exists():
        for line in mem_info.read_text().splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return kb // 1024

    return 0


def _detect_nvidia() -> tuple[int, int, str, str, str] | None:
    """
    Returns (vram_mb, gpu_count, gpu_name, cuda_version, vendor) or None.
    Uses nvidia-smi --query-gpu for reliable machine-parseable output.
    """
    out = _run(
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    )
    if not out:
        return None

    lines = [l.strip() for l in out.splitlines() if l.strip()]
    if not lines:
        return None

    gpu_count = len(lines)
    # Use the first GPU's VRAM as the representative value
    parts = lines[0].split(",")
    if len(parts) < 2:
        return None

    gpu_name = parts[0].strip()
    try:
        vram_mb = int(parts[1].strip())
    except ValueError:
        return None

    # CUDA version from nvidia-smi header
    header = _run("nvidia-smi", "--query", "--display=COMPUTE")
    cuda_match = re.search(r"CUDA Version\s*:\s*([\d.]+)", header)
    cuda_version = cuda_match.group(1) if cuda_match else ""

    return vram_mb, gpu_count, gpu_name, cuda_version, "nvidia"


def _detect_amd() -> tuple[int, int, str, str, str] | None:
    """Returns (vram_mb, gpu_count, gpu_name, rocm_version, vendor) or None."""
    out = _run("rocm-smi", "--showmeminfo", "vram", "--json")
    if not out:
        return None

    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return None

    cards = [k for k in data if k.startswith("card")]
    if not cards:
        return None

    gpu_count = len(cards)
    first = data[cards[0]]
    try:
        vram_mb = int(first.get("VRAM Total Memory (B)", 0)) // (1024 * 1024)
    except (ValueError, TypeError):
        return None

    gpu_name = first.get("Card series", "AMD GPU")
    rocm_out = _run("rocminfo")
    rocm_match = re.search(r"ROCm Runtime Version\s*:\s*([\d.]+)", rocm_out)
    rocm_version = rocm_match.group(1) if rocm_match else ""

    return vram_mb, gpu_count, gpu_name, rocm_version, "amd"


def _detect_apple() -> tuple[int, int, str, str, str] | None:
    """
    Returns (unified_ram_mb, 1, gpu_name, '', 'apple') or None.
    Apple Silicon shares RAM between CPU and GPU; we report total RAM as VRAM.
    """
    if platform.system() != "Darwin":
        return None

    # Check for Apple Silicon
    arm_check = _run("sysctl", "-n", "hw.optional.arm64")
    if arm_check.strip() != "1":
        return None

    out = _run("system_profiler", "SPHardwareDataType", "-json")
    try:
        data = json.loads(out)
        hw = data["SPHardwareDataType"][0]
        chip = hw.get("chip_type", "Apple Silicon")
        ram_str = hw.get("physical_memory", "0 GB")
        ram_gb = int(re.search(r"(\d+)", ram_str).group(1))  # type: ignore[union-attr]
        vram_mb = ram_gb * 1024  # unified memory
    except Exception:
        return None

    return vram_mb, 1, chip, "", "apple"


def detect_hardware() -> HardwareSpec:
    """
    Auto-detect hardware and return a HardwareSpec.

    Detection order: NVIDIA → AMD → Apple → CPU fallback.
    Never raises — returns a CPU-only spec on any detection failure.
    """
    ram_mb = _ram_mb()

    for detector in (_detect_nvidia, _detect_amd, _detect_apple):
        try:
            result = detector()
        except Exception:
            result = None
        if result is not None:
            vram_mb, gpu_count, gpu_name, version, vendor = result
            return HardwareSpec(
                vram_mb=vram_mb,
                ram_mb=ram_mb,
                gpu_count=gpu_count,
                gpu_vendor=vendor,
                gpu_name=gpu_name,
                cuda_version=version if vendor == "nvidia" else "",
                rocm_version=version if vendor == "amd" else "",
            )

    # CPU-only fallback
    return HardwareSpec(
        vram_mb=0,
        ram_mb=ram_mb,
        gpu_count=0,
        gpu_vendor="cpu",
        gpu_name="",
    )


def detect_hardware_json() -> str:
    """Return detect_hardware() result as a JSON string (for CLI / one-liner use)."""
    import dataclasses
    spec = detect_hardware()
    return json.dumps(dataclasses.asdict(spec), indent=2)


if __name__ == "__main__":
    print(detect_hardware_json())
