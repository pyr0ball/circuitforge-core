# BSL vs MIT — Licensing Boundaries

circuitforge-core contains both MIT and BSL 1.1 licensed code. Understanding the boundary matters for contributors and for deciding where new modules belong.

## The rule

| Code category | License |
|---------------|---------|
| Discovery, ingestion, data pipeline | **MIT** |
| LLM inference, AI generation, fine-tuned model access | **BSL 1.1** |
| UI scaffolding, process management | **MIT** |
| Tier gates, license validation | **BSL 1.1** |
| Database, storage, configuration | **MIT** |

**Heuristic:** If a competitor could use the module to build a commercial AI product without building the hard parts themselves, it's BSL 1.1. If it's plumbing that any software project might need, it's MIT.

## BSL 1.1 in practice

BSL 1.1 means:
- Free for personal non-commercial self-hosting
- Free for internal business use (using the software, not selling it)
- Commercial SaaS re-hosting requires a paid license from Circuit Forge LLC
- Converts to MIT after 4 years

"Commercial SaaS re-hosting" means: taking cf-core's AI features and building a competing product that charges users for them without a license. It does NOT restrict:
- Running cf-core on your own server for your own use
- Modifying cf-core for personal use
- Contributing back to cf-core

## What this means for contributors

If you're adding a module:
- Add MIT code to the `MIT` section of `pyproject.toml`
- Add BSL 1.1 code to the `BSL` section
- Don't mix MIT and BSL code in the same module
- If uncertain, ask before submitting — wrong license on a module causes legal headaches

## The `Co-Authored-By` policy

Do NOT add `Co-Authored-By: Claude` (or any AI attribution trailer) to commits in CircuitForge repos. This is required for BSL 1.1 commercial viability — AI-assisted code with attribution claims can complicate licensing in ways that affect the ability to enforce BSL terms.

This is not about hiding AI use. It's a legal precaution for a company that depends on BSL enforcement to fund its mission.

## BSL conversion timeline

| Module | BSL since | MIT date |
|--------|-----------|----------|
| `tiers` | 2025-01-01 | 2029-01-01 |
| `llm` | 2025-01-01 | 2029-01-01 |

The conversion dates are tracked in `LICENSE` and will be updated as modules are added.
