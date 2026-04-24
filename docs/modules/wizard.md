# wizard

First-run wizard base class. **Stub.**

```python
from circuitforge_core.wizard import BaseWizard  # planned
```

## Purpose

`BaseWizard` provides a standard scaffold for first-run product setup. Every CircuitForge product has a first-run wizard that:

1. Validates prerequisites (Docker, required ports, disk space)
2. Configures the LLM backend (local Ollama / vLLM / BYOK cloud)
3. Sets user preferences and accessibility options
4. Issues or validates a license key
5. Runs a smoke test and confirms everything is working

## Existing implementations

Each product currently implements its own wizard:

- **Peregrine**: `app/pages/0_Setup.py` (Streamlit) — gates app until `config/user.yaml` exists
- **Kiwi**: Vue 3 wizard component with step-by-step hardware detection, LLM config, dietary preferences

These will be refactored to share the `BaseWizard` scaffold once the interface stabilizes.

## Planned `BaseWizard` API

```python
class BaseWizard:
    steps: list[WizardStep]         # ordered list of setup steps

    def run(self) -> WizardResult:
        """Execute all steps in order. Returns result with completion status."""
        ...

    def resume(self, from_step: int) -> WizardResult:
        """Resume from a specific step (e.g., after fixing a failed prereq)."""
        ...
```

## Accessibility in the wizard

The wizard is the first thing new users see. It must meet CF's accessibility standards:

- All steps must be completable with keyboard only
- No time limits on any step
- Plain-language instructions throughout (no jargon)
- Accessibility preferences collected early (step 2 or 3) so the rest of the wizard can immediately adapt
- Progress saved after each step so users can pause and return
