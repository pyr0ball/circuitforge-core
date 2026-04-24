"""
AudioCraft MusicGen backend — music continuation via Meta's MusicGen.

Models are downloaded to /Library/Assets/LLM/musicgen/ (HF hub cache).
The melody model (~8 GB VRAM) is the default; small (~1.5 GB) is available
for lower-VRAM nodes.

Continuation workflow:
  1. Decode input audio with torchaudio (any format ffmpeg understands)
  2. Trim to the last `prompt_duration_s` seconds — this anchors the generation
  3. Call model.generate_continuation(prompt_waveform, prompt_sample_rate, ...)
  4. Output tensor is the NEW audio only (not prompt + continuation)
  5. Encode to the requested format and return
"""
from __future__ import annotations

import logging
import os

from circuitforge_core.musicgen.backends.base import (
    AudioFormat,
    MusicContinueResult,
    decode_audio,
    encode_audio,
)

# All MusicGen/AudioCraft weights land here — consistent with other CF model dirs.
_MUSICGEN_CACHE = "/Library/Assets/LLM/musicgen"

# VRAM estimates (MB) per model variant
_VRAM_MB: dict[str, int] = {
    "facebook/musicgen-small": 1500,
    "facebook/musicgen-medium": 4500,
    "facebook/musicgen-melody": 8000,
    "facebook/musicgen-large": 8500,
}

logger = logging.getLogger(__name__)


class AudioCraftBackend:
    """MusicGen backend using Meta's AudioCraft library."""

    def __init__(self, model_name: str = "facebook/musicgen-melody", device: str = "cuda") -> None:
        # Redirect HF hub cache before the first import so weights go to /Library/Assets
        os.environ.setdefault("HF_HOME", _MUSICGEN_CACHE)
        os.makedirs(_MUSICGEN_CACHE, exist_ok=True)

        from audiocraft.models import MusicGen  # noqa: PLC0415

        logger.info("Loading MusicGen model: %s on %s", model_name, device)
        self._model = MusicGen.get_pretrained(model_name, device=device)
        self._model_name = model_name
        self._device = device
        logger.info("MusicGen ready: %s", model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def vram_mb(self) -> int:
        return _VRAM_MB.get(self._model_name, 8000)

    def continue_audio(
        self,
        audio_bytes: bytes,
        *,
        description: str | None = None,
        duration_s: float = 15.0,
        prompt_duration_s: float = 10.0,
        format: AudioFormat = "wav",
    ) -> MusicContinueResult:
        import torch

        # Decode input audio -> [C, T] tensor
        wav, sr = decode_audio(audio_bytes)

        # Trim to the last `prompt_duration_s` seconds to form the conditioning prompt.
        # Using the end of the track (not the beginning) gives the model the musical
        # context closest to where we want to continue.
        max_prompt_samples = int(prompt_duration_s * sr)
        if wav.shape[-1] > max_prompt_samples:
            wav = wav[..., -max_prompt_samples:]

        # MusicGen expects [batch, channels, time]
        prompt_tensor = wav.unsqueeze(0).to(self._device)

        # Build descriptions list — one entry per batch item (batch=1 here)
        descriptions = [description] if description else [None]

        self._model.set_generation_params(
            duration=duration_s,
            top_k=250,
            temperature=1.0,
            cfg_coef=3.0,
        )

        logger.info(
            "Generating %.1fs continuation (prompt=%.1fs) model=%s",
            duration_s,
            prompt_duration_s,
            self._model_name,
        )

        with torch.no_grad():
            output = self._model.generate_continuation(
                prompt=prompt_tensor,
                prompt_sample_rate=sr,
                descriptions=descriptions,
                progress=True,
            )

        # output: [batch, channels, time] at model sample rate (32 kHz)
        output_wav = output[0]  # [C, T]
        model_sr = self._model.sample_rate

        actual_duration_s = output_wav.shape[-1] / model_sr
        audio_bytes_out = encode_audio(output_wav, model_sr, format)

        return MusicContinueResult(
            audio_bytes=audio_bytes_out,
            sample_rate=model_sr,
            duration_s=actual_duration_s,
            format=format,
            model=self._model_name,
            prompt_duration_s=prompt_duration_s,
        )
