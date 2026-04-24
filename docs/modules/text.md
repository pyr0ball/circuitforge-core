# text

Text processing utilities. Normalization, truncation, chunking, and token estimation — shared across all products that manipulate text before or after LLM inference.

```python
from circuitforge_core.text import normalize, chunk, truncate, estimate_tokens
```

## `normalize(text: str) -> str`

Strips excess whitespace, normalizes unicode (NFC), and removes null bytes and control characters that can cause downstream issues with SQLite FTS5 or LLM tokenizers.

```python
from circuitforge_core.text import normalize

clean = normalize("  Hello\u00a0world\x00  ")
# → "Hello world"
```

## `truncate(text: str, max_tokens: int, model: str = "default") -> str`

Truncates text to approximately `max_tokens` tokens, breaking at sentence or paragraph boundaries where possible. Uses a simple byte-based heuristic (1 token ≈ 4 bytes) unless a specific model tokenizer is requested.

```python
excerpt = truncate(long_doc, max_tokens=2048)
```

## `chunk(text: str, chunk_size: int, overlap: int = 0) -> list[str]`

Splits text into overlapping chunks for RAG (retrieval-augmented generation) pipelines. Respects paragraph boundaries.

```python
chunks = chunk(article_text, chunk_size=512, overlap=64)
```

## `estimate_tokens(text: str, model: str = "default") -> int`

Estimates token count without loading a full tokenizer. Accurate enough for context window budget planning (within ~10%).

## FTS5 helpers

SQLite FTS5 has quirks with special characters in MATCH expressions. The `text` module provides helpers used by the recipe engine and other FTS5 consumers:

```python
from circuitforge_core.text import fts_quote, strip_apostrophes

# Always double-quote FTS5 terms — bare tokens break on brand names
query = " ".join(fts_quote(term) for term in tokens)
# → '"chicken" "breast" "lemon"'

# Strip apostrophes before FTS5 queries
clean = strip_apostrophes("O'Doul's")
# → "ODoulS"
```

!!! warning "FTS5 gotcha"
    Always quote ALL terms in MATCH expressions. Bare tokens break on brand names (e.g., `O'Doul's`), plant-based ingredient names, and anything with punctuation.
