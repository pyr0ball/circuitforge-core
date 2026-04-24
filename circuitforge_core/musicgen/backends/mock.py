"""
Mock MusicGenBackend — returns silent WAV audio; no GPU required.

Used in unit tests and CI where GPU is unavailable.
"""
from __future__ import annotations

import io
import struct
import wave

from circuitforge_core.musicgen.backends.base import AudioFormat, MusicContinueResult


class MockMusicGenBackend:
    """Returns a silent WAV file of the requested duration."""

    @property
    def model_name(self) -> str:
        return "mock"

    @property
    def vram_mb(self) -> int:
        return 0

    def continue_audio(
        self,
        audio_bytes: bytes,
        *,
        description: str | None = None,
        duration_s: float = 15.0,
        prompt_duration_s: float = 10.0,
        format: AudioFormat = "wav",
    ) -> MusicContinueResult:
        sample_rate = 32000
        n_samples = int(duration_s * sample_rate)
        silent_samples = b"\x00\x00" * n_samples  # 16-bit PCM silence

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(silent_samples)

        return MusicContinueResult(
            audio_bytes=buf.getvalue(),
            sample_rate=sample_rate,
            duration_s=duration_s,
            format="wav",
            model="mock",
            prompt_duration_s=prompt_duration_s,
        )
