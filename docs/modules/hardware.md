# hardware

GPU enumeration and VRAM-tier profile generation. Used by `manage.sh` at startup to recommend a Docker Compose profile and by the cf-orch coordinator for resource allocation.

```python
from circuitforge_core.hardware import get_gpus, recommend_profile, HardwareProfile
```

## GPU detection

`get_gpus()` returns a list of detected GPUs with their VRAM capacity. Detection strategy:

1. Try `nvidia-smi` (Linux/Windows NVIDIA)
2. Fall back to `system_profiler SPDisplaysDataType` on Darwin when `hw.optional.arm64=1` (Apple Silicon)
3. Return CPU-only profile if neither succeeds

```python
gpus = get_gpus()
# [{"name": "RTX 4090", "vram_gb": 24.0, "type": "nvidia"},
#  {"name": "Apple M2 Max", "vram_gb": 32.0, "type": "apple_silicon"}]
```

## Compose profile recommendation

```python
profile = recommend_profile(gpus)
# "single-gpu" | "dual-gpu" | "cpu" | "remote"
```

Profile selection rules:
- `single-gpu`: one NVIDIA GPU with >= 8GB VRAM
- `dual-gpu`: two or more NVIDIA GPUs
- `cpu`: no NVIDIA GPU (Apple Silicon uses `cpu` since Docker on Mac has no Metal passthrough)
- `remote`: explicitly requested or when local inference would exceed available VRAM

!!! note "Apple Silicon"
    Apple Silicon Macs should run Ollama natively (outside Docker) for Metal-accelerated inference. Docker on macOS runs in a Linux VM with no Metal passthrough. `preflight.py` in each product detects native Ollama on :11434 and adopts it automatically.

## VRAM tiers

| VRAM | Models that fit |
|------|----------------|
| < 4 GB | Quantized 1B–3B models (Phi-3 mini, Llama 3.2 3B Q4) |
| 4–8 GB | 7B–8B models Q4 (Llama 3.1 8B, Mistral 7B) |
| 8–16 GB | 13B–14B models Q4, 7B models in full precision |
| 16–24 GB | 30B models Q4, 13B full precision |
| 24 GB+ | 70B models Q4 |

## HardwareProfile

The `HardwareProfile` dataclass is written to `compose.override.yml` by `preflight.py` at product startup, making GPU capabilities available to Docker Compose without hardcoding.
