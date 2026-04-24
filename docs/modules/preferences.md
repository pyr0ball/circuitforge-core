# preferences

Per-user preference store. Provides a dot-path `get()`/`set()` API over a local YAML file, with a pluggable backend for cloud deployments.

```python
from circuitforge_core.preferences import get_prefs, UserPreferences
```

## API

### `get_prefs(user_id: str | None = None) -> UserPreferences`

Returns the preference store for the given user. In local mode, `user_id` is ignored and a shared local file is used. In cloud mode, each user gets an isolated preference file under `CLOUD_DATA_ROOT`.

### `prefs.get(key: str, default=None) -> Any`

Dot-path key access. Returns `default` if the key doesn't exist.

```python
prefs = get_prefs()
theme = prefs.get("ui.theme", "light")
opted_out = prefs.get("affiliates.opted_out", False)
```

### `prefs.set(key: str, value: Any)`

Sets a value at the dot path. Creates intermediate keys as needed. Persists immediately.

```python
prefs.set("ui.theme", "dark")
prefs.set("dietary.restrictions", ["vegan", "gluten-free"])
```

### `prefs.delete(key: str)`

Removes a key. No-ops silently if the key doesn't exist.

## Accessibility preferences

The `preferences` module includes first-class support for accessibility needs under the `accessibility.*` namespace. These are surfaced in product settings UIs and respected throughout the UI layer.

```yaml
# Stored in user preferences
accessibility:
  reduce_motion: true           # No animations or transitions
  high_contrast: false
  font_size: large              # small | medium | large | x-large
  screen_reader_hints: true     # Extra ARIA labels and descriptions
  plain_language: true          # Simplified text throughout UI
  extra_confirmation_steps: true # Additional "are you sure?" prompts
```

Products should read these at render time and pass them to UI components. See the design philosophy for why ND/adaptive needs users are a primary audience.

## Pluggable backend

The default backend is a local YAML file. Products can substitute a database backend for cloud deployments:

```python
from circuitforge_core.preferences import get_prefs, SQLitePreferenceBackend

prefs = get_prefs(user_id, backend=SQLitePreferenceBackend(db_path))
```

## Storage format

```yaml
# ~/.local/share/circuitforge/myproduct/prefs.yaml (or per-user cloud path)
ui:
  theme: dark
affiliates:
  opted_out: false
dietary:
  restrictions:
    - vegan
```
