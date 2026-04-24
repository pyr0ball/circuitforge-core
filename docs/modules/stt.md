# stt

Speech-to-text router. **Stub — not yet implemented.**

```python
from circuitforge_core.stt import STTRouter  # planned
```

## Planned design

The STT module will provide a unified interface over local speech-to-text backends, following the same fallback-chain pattern as the [LLM router](llm.md).

**Planned backends:**
- `whisper_cpp` — local, CPU/GPU, various model sizes
- `faster_whisper` — local, GPU-accelerated, CTranslate2 backend
- `whisper_openai` — cloud, requires `OPENAI_API_KEY`

**Planned use cases across the menagerie:**
- Osprey: transcribe hold music + IVR menu audio for navigation
- Linnet: real-time speech annotation (tone classification requires transcript)
- Peregrine: interview practice sessions

## Current status

The `circuitforge_core.stt` directory exists in-tree with a stub `__init__.py`. No working implementation yet. Planned for the milestone after Osprey reaches beta.

If you need STT before this module ships, use `faster-whisper` directly in the product and plan to migrate to this interface once it stabilizes.
