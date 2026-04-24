"""
TTSBackend Protocol — backend-agnostic TTS interface.

All backends return TTSResult with audio bytes in the requested format.
Supported formats: ogg (default, smallest), wav (uncompressed, always works), mp3.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

AudioFormat = Literal["ogg", "wav", "mp3"]


@dataclass(frozen=True)
class TTSResult:
    audio_bytes: bytes
    sample_rate: int
    duration_s: float
    format: AudioFormat = "ogg"
    model: str = ""


@runtime_checkable
class TTSBackend(Protocol):
    def synthesize(
        self,
        text: str,
        *,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        audio_prompt: bytes | None = None,
        format: AudioFormat = "ogg",
    ) -> TTSResult: ...

    @property
    def model_name(self) -> str: ...

    @property
    def vram_mb(self) -> int: ...


def _encode_audio(
    wav_tensor,        # torch.Tensor shape [1, T] or [T]
    sample_rate: int,
    format: AudioFormat,
) -> bytes:
    """Convert a torch tensor to audio bytes in the requested format."""
    import torch
    import torchaudio

    wav = wav_tensor
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav = wav.to(torch.float32).cpu()

    buf = io.BytesIO()
    if format == "wav":
        torchaudio.save(buf, wav, sample_rate, format="wav")
    elif format == "ogg":
        # libvorbis may not be available on all torchaudio builds; fall back to wav
        try:
            torchaudio.save(buf, wav, sample_rate, format="ogg", encoding="vorbis")
        except Exception:
            buf = io.BytesIO()
            torchaudio.save(buf, wav, sample_rate, format="wav")
    elif format == "mp3":
        # torchaudio MP3 encode requires ffmpeg backend; fall back to wav on failure
        try:
            torchaudio.save(buf, wav, sample_rate, format="mp3")
        except Exception:
            buf = io.BytesIO()
            torchaudio.save(buf, wav, sample_rate, format="wav")
    return buf.getvalue()


def make_tts_backend(
    model_path: str,
    *,
    mock: bool = False,
    device: str = "cuda",
) -> TTSBackend:
    if mock:
        from circuitforge_core.tts.backends.mock import MockTTSBackend
        return MockTTSBackend()
    from circuitforge_core.tts.backends.chatterbox import ChatterboxTurboBackend
    return ChatterboxTurboBackend(model_path=model_path, device=device)
