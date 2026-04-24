# affiliates

Affiliate URL wrapping with user opt-out and BYOK user IDs. Shared across all CircuitForge products that surface external purchase or listing links.

```python
from circuitforge_core.affiliates import wrap_url
```

## Design principle

Affiliate links are disclosed to users and opt-out is always one click away. CF earns a small commission when users buy through wrapped links; this is the primary monetization path for free-tier products. The implementation is transparent: no dark patterns, no hidden redirects.

## `wrap_url(url, user_id=None, product=None) -> str`

Wraps a URL with the configured affiliate parameters. Returns the original URL unchanged if:
- Affiliate links are disabled globally (`CF_AFFILIATES_ENABLED=false`)
- The user has opted out (`preferences.get("affiliates.opted_out")`)
- The domain is not in the supported affiliate network list

```python
from circuitforge_core.affiliates import wrap_url

wrapped = wrap_url(
    "https://www.ebay.com/itm/123456",
    user_id="user_abc123",
    product="snipe",
)
# → "https://www.ebay.com/itm/123456?mkrid=711-53200-19255-0&campid=CF_SNIPE_abc123&..."
```

## User opt-out

```python
from circuitforge_core.preferences import get_prefs

prefs = get_prefs(user_id)
prefs.set("affiliates.opted_out", True)
```

When `opted_out` is `True`, `wrap_url()` returns the bare URL. The UI should surface this setting prominently — never bury it.

## BYOK user IDs

BYOK users (those with their own license key or API key) get a unique affiliate sub-ID so their contributions are tracked separately. This is handled automatically when a `user_id` is passed.

## Supported networks

| Product | Network | Notes |
|---------|---------|-------|
| Snipe | eBay Partner Network | `campid` encodes product + user |
| Kiwi | Amazon Associates (planned) | For pantry staples / equipment |
| Waxwing | Various garden suppliers (planned) | |

## Environment variables

```bash
CF_AFFILIATES_ENABLED=true          # global kill switch
CF_EBAY_CAMPAIGN_ID=your_campaign   # eBay Partner Network campaign ID
CF_AMAZON_ASSOCIATE_TAG=your_tag    # Amazon Associates tag
```
