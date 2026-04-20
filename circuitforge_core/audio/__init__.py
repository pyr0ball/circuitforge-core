"""
circuitforge_core.audio — shared PCM and audio signal utilities.

MIT licensed. No model weights. No HuggingFace. Dependency: numpy only
(scipy optional for high-quality resampling).

Consumers:
  cf-voice   — replaces hand-rolled PCM conversion in stt.py / context.py
  Sparrow    — torchaudio stitching, export, acoustic analysis
  Avocet     — audio preprocessing for classifier training corpus
  Linnet     — chunk accumulation for real-time tone annotation
"""
from circuitforge_core.audio.convert import (
    bytes_to_float32,
    float32_to_pcm,
    pcm_to_float32,
)
from circuitforge_core.audio.gate import is_silent
from circuitforge_core.audio.resample import resample
from circuitforge_core.audio.buffer import ChunkAccumulator

__all__ = [
    "bytes_to_float32",
    "float32_to_pcm",
    "pcm_to_float32",
    "is_silent",
    "resample",
    "ChunkAccumulator",
]
