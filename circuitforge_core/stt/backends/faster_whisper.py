# circuitforge_core/stt/backends/faster_whisper.py — FasterWhisperBackend
#
# MIT licensed. Requires: pip install -e "circuitforge-core[stt-faster-whisper]"
#
# Model path can be:
#   - A size name:  "base", "small", "medium", "large-v3"
#     (faster-whisper downloads and caches it on first use)
#   - A local path: "/Library/Assets/LLM/whisper/models/Whisper/faster-whisper/..."
#     (preferred for air-gapped nodes — no download needed)
from __future__ import annotations

import io
import logging
import os
import tempfile

from circuitforge_core.stt.backends.base import STTResult, STTSegment

logger = logging.getLogger(__name__)

# VRAM estimates by model size. Used by cf-orch for VRAM budgeting.
_VRAM_MB_BY_SIZE: dict[str, int] = {
    "tiny":       200,
    "base":       350,
    "small":      600,
    "medium":    1024,
    "large":     2048,
    "large-v2":  2048,
    "large-v3":  2048,
    "distil-large-v3": 1500,
}

# Aggregate confidence from per-segment no_speech_prob values.
# faster-whisper doesn't expose a direct confidence score, so we invert the
# mean no_speech_prob as a proxy. This is conservative but directionally correct.
def _aggregate_confidence(segments: list) -> float:
    if not segments:
        return 0.0
    probs = [max(0.0, 1.0 - getattr(s, "no_speech_prob", 0.0)) for s in segments]
    return sum(probs) / len(probs)


class FasterWhisperBackend:
    """
    faster-whisper STT backend.

    Thread-safe after construction: WhisperModel internally manages its own
    CUDA context and is safe to call from multiple threads.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is not installed. "
                "Run: pip install -e 'circuitforge-core[stt-faster-whisper]'"
            ) from exc

        logger.info("Loading faster-whisper model from %r (device=%s)", model_path, device)
        self._model_path = model_path
        self._device = device
        self._compute_type = compute_type
        self._model = WhisperModel(model_path, device=device, compute_type=compute_type)
        logger.info("faster-whisper model ready")

        # Determine VRAM footprint from model name/path stem.
        stem = os.path.basename(model_path.rstrip("/")).lower()
        self._vram_mb = next(
            (v for k, v in _VRAM_MB_BY_SIZE.items() if k in stem),
            1024,   # conservative default if size can't be inferred
        )

    def transcribe(
        self,
        audio: bytes,
        *,
        language: str | None = None,
        confidence_threshold: float = STTResult.CONFIDENCE_DEFAULT_THRESHOLD,
    ) -> STTResult:
        """
        Transcribe raw audio bytes.

        audio can be any format ffmpeg understands (WAV, MP3, OGG, FLAC, etc.).
        faster-whisper writes audio to a temp file internally; we follow the
        same pattern to avoid holding the bytes in memory longer than needed.
        """
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
            tmp.write(audio)
            tmp_path = tmp.name

        try:
            segments_gen, info = self._model.transcribe(
                tmp_path,
                language=language,
                word_timestamps=True,
                vad_filter=True,
            )
            segments = list(segments_gen)
        finally:
            os.unlink(tmp_path)

        text = " ".join(s.text.strip() for s in segments).strip()
        confidence = _aggregate_confidence(segments)
        duration_s = info.duration if hasattr(info, "duration") else None
        detected_language = getattr(info, "language", language)

        stt_segments = [
            STTSegment(
                start_s=s.start,
                end_s=s.end,
                text=s.text.strip(),
                confidence=max(0.0, 1.0 - getattr(s, "no_speech_prob", 0.0)),
            )
            for s in segments
        ]

        return STTResult(
            text=text,
            confidence=confidence,
            below_threshold=confidence < confidence_threshold,
            language=detected_language,
            duration_s=duration_s,
            segments=stt_segments,
            model=self._model_path,
        )

    @property
    def model_name(self) -> str:
        return self._model_path

    @property
    def vram_mb(self) -> int:
        return self._vram_mb
