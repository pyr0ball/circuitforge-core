"""ChatterboxTurboBackend — ResembleAI chatterbox-turbo TTS via chatterbox-tts package."""
from __future__ import annotations

import io
import os
import tempfile

from circuitforge_core.tts.backends.base import (
    AudioFormat,
    TTSBackend,
    TTSResult,
    _encode_audio,
)

_VRAM_MB = 768  # conservative estimate for chatterbox-turbo weights


class ChatterboxTurboBackend:
    def __init__(self, model_path: str, device: str = "cuda") -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        from chatterbox.models.s3gen import S3GEN_SR
        from chatterbox.tts import ChatterboxTTS

        self._sr = S3GEN_SR
        self._device = device
        self._model = ChatterboxTTS.from_local(model_path, device=device)
        self._model_path = model_path

    @property
    def model_name(self) -> str:
        return f"chatterbox-turbo@{os.path.basename(self._model_path)}"

    @property
    def vram_mb(self) -> int:
        return _VRAM_MB

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
        audio_prompt_path: str | None = None
        _tmp = None

        if audio_prompt is not None:
            _tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            _tmp.write(audio_prompt)
            _tmp.flush()
            audio_prompt_path = _tmp.name

        try:
            wav = self._model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                audio_prompt_path=audio_prompt_path,
            )
        finally:
            if _tmp is not None:
                _tmp.close()
                os.unlink(_tmp.name)

        duration_s = wav.shape[-1] / self._sr
        audio_bytes = _encode_audio(wav, self._sr, format)
        return TTSResult(
            audio_bytes=audio_bytes,
            sample_rate=self._sr,
            duration_s=duration_s,
            format=format,
            model=self.model_name,
        )


assert isinstance(
    ChatterboxTurboBackend.__new__(ChatterboxTurboBackend), TTSBackend
), "ChatterboxTurboBackend must satisfy TTSBackend Protocol"
