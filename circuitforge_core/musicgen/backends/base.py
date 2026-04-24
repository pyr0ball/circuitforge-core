"""
MusicGenBackend Protocol — backend-agnostic music continuation interface.

All backends accept an audio prompt (raw bytes, any ffmpeg-readable format) and
return MusicContinueResult with the generated continuation as audio bytes.

The continuation is the *new* audio only (not prompt + continuation). Callers
that want a seamless joined file can concatenate the original + result themselves.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

AudioFormat = Literal["wav", "mp3"]

MODEL_SMALL = "facebook/musicgen-small"
MODEL_MELODY = "facebook/musicgen-melody"


@dataclass(frozen=True)
class MusicContinueResult:
    audio_bytes: bytes
    sample_rate: int
    duration_s: float
    format: AudioFormat
    model: str
    prompt_duration_s: float


@runtime_checkable
class MusicGenBackend(Protocol):
    def continue_audio(
        self,
        audio_bytes: bytes,
        *,
        description: str | None = None,
        duration_s: float = 15.0,
        prompt_duration_s: float = 10.0,
        format: AudioFormat = "wav",
    ) -> MusicContinueResult: ...

    @property
    def model_name(self) -> str: ...

    @property
    def vram_mb(self) -> int: ...


def encode_audio(wav_tensor, sample_rate: int, format: AudioFormat) -> bytes:
    """Encode a [C, T] or [1, C, T] torch tensor to audio bytes."""
    import io
    import torch
    import torchaudio

    wav = wav_tensor
    if wav.dim() == 3:
        wav = wav.squeeze(0)          # [1, C, T] -> [C, T]
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)        # [T] -> [1, T]
    wav = wav.to(torch.float32).cpu()

    buf = io.BytesIO()
    if format == "wav":
        torchaudio.save(buf, wav, sample_rate, format="wav")
    elif format == "mp3":
        try:
            torchaudio.save(buf, wav, sample_rate, format="mp3")
        except Exception:
            # ffmpeg backend not available; fall back to wav
            buf = io.BytesIO()
            torchaudio.save(buf, wav, sample_rate, format="wav")
    return buf.getvalue()


def decode_audio(audio_bytes: bytes) -> tuple:
    """Decode arbitrary audio bytes to (waveform [C, T], sample_rate)."""
    import io
    import torchaudio

    buf = io.BytesIO(audio_bytes)
    wav, sr = torchaudio.load(buf)
    return wav, sr


def make_musicgen_backend(
    model_name: str = MODEL_MELODY,
    *,
    mock: bool = False,
    device: str = "cuda",
) -> MusicGenBackend:
    if mock:
        from circuitforge_core.musicgen.backends.mock import MockMusicGenBackend
        return MockMusicGenBackend()
    from circuitforge_core.musicgen.backends.audiocraft import AudioCraftBackend
    return AudioCraftBackend(model_name=model_name, device=device)
