# tts

Text-to-speech router. **Stub — not yet implemented.**

```python
from circuitforge_core.tts import TTSRouter  # planned
```

## Planned design

The TTS module will mirror the [LLM router](llm.md) pattern: a configurable fallback chain over local and cloud TTS backends.

**Planned backends:**
- `piper` — local, fast, offline-capable; excellent quality for a neural TTS
- `espeak` — local, minimal resource use, robotic but reliable fallback
- `openai_tts` — cloud, `tts-1` and `tts-1-hd`; requires `OPENAI_API_KEY`

**Planned use cases:**
- Osprey: reading back IVR menus aloud; accessibility for users who can't monitor hold music
- Linnet: speaking annotated tone labels alongside the original audio
- Any product: accessible audio output for users with print disabilities

## Current status

Stub only. Planned to ship alongside or shortly after the STT module, as most use cases need both.

**Piper** is the recommended path when this lands: it runs locally at 10–20x real-time on CPU, supports 40+ language/speaker models, and has no API key requirement. See [rhasspy/piper](https://github.com/rhasspy/piper) for model downloads.
