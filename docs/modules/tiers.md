# tiers

Tier system implementation. Provides the `@require_tier()` decorator used on FastAPI endpoints and the BYOK (bring your own key) unlock logic.

```python
from circuitforge_core.tiers import require_tier, TierLevel
```

## Tier levels

| Tier | Constant | What it unlocks |
|------|----------|----------------|
| Free | `TierLevel.FREE` | Core pipeline, basic AI assist, local LLM only |
| Paid | `TierLevel.PAID` | Cloud LLM, integrations, full AI generation suite |
| Premium | `TierLevel.PREMIUM` | Fine-tuned models, multi-user, advanced analytics |
| Ultra | `TierLevel.ULTRA` | Human-in-the-loop operator execution |

## BYOK unlocks

Users who configure their own LLM backend (via `config/llm.yaml`) can unlock features that would otherwise require a paid tier. The `tiers` module checks for configured BYOK backends before enforcing tier gates.

This is intentional: privacy-preserving self-hosting is rewarded, not penalized. A user running their own Ollama instance gets AI features without a subscription.

## `@require_tier(tier: str)`

Decorator for FastAPI route handlers. Resolves the calling user's tier from the request context (Heimdall JWT, validated by Caddy) and raises HTTP 403 if insufficient.

```python
from circuitforge_core.tiers import require_tier

@router.post("/recipes/suggest")
@require_tier("paid")
async def suggest_recipes(request: Request, body: SuggestRequest):
    ...
```

In local (non-cloud) mode with no license configured, all users default to `free`. BYOK detection runs first — if a local LLM backend is configured, relevant paid features unlock regardless of license tier.

## Per-product overrides

Products define which specific features are gated at which tier in their own `app/tiers.py`, using the cf-core decorators as building blocks. The cf-core `tiers` module provides the mechanism; the product owns the policy.

```python
# kiwi/app/tiers.py
from circuitforge_core.tiers import require_tier

# Re-export with product-specific names if desired
require_paid = require_tier("paid")
require_premium = require_tier("premium")

# BYOK unlockable features — defined per product
BYOK_UNLOCKABLE = [
    "recipe_suggestion_l3",
    "receipt_ocr",
    "expiry_llm_fallback",
]
```

## Checking tier in non-endpoint code

```python
from circuitforge_core.tiers import get_user_tier, TierLevel

tier = get_user_tier(user_id)
if tier >= TierLevel.PAID:
    # run AI feature
```
