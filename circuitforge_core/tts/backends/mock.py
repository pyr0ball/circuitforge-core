"""MockTTSBackend — no GPU, no model required. Returns a silent WAV clip."""
from __future__ import annotations

import io
import struct
import wave

from circuitforge_core.tts.backends.base import AudioFormat, TTSBackend, TTSResult

_SAMPLE_RATE = 24000


def _silent_wav(duration_s: float = 0.5, sample_rate: int = _SAMPLE_RATE) -> bytes:
    num_samples = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack(f"<{num_samples}h", *([0] * num_samples)))
    return buf.getvalue()


class MockTTSBackend:
    """Minimal TTSBackend implementation for tests and CI."""

    @property
    def model_name(self) -> str:
        return "mock-tts"

    @property
    def vram_mb(self) -> int:
        return 0

    def synthesize(
        self,
        text: str,
        *,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        audio_prompt: bytes | None = None,
        format: AudioFormat = "ogg",
    ) -> TTSResult:
        duration_s = max(0.1, len(text.split()) * 0.3)
        audio = _silent_wav(duration_s)
        return TTSResult(
            audio_bytes=audio,
            sample_rate=_SAMPLE_RATE,
            duration_s=duration_s,
            format="wav",
            model=self.model_name,
        )


assert isinstance(MockTTSBackend(), TTSBackend), "MockTTSBackend must satisfy TTSBackend Protocol"
