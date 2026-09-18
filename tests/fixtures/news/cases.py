"""Loader for the WP02 news corpora (DESIGN.md 15.2 `tests/fixtures/news/`).

Four JSON-lines files, one object per line, all owned by WP02:

* `benign_items.jsonl`     - 12 ordinary items (`id`, `created_at`, `headline`, `summary`, `source`, `symbols`); the
                             pipeline must keep them, so they are also the news of the `entry_text` / `manage_text` goldens.
* `hostile_headlines.jsonl` - prompt injections and leak attempts (`id`, `headline`, `why`, optional `summary`,
                             `expect` = `"dropped"` when `is_suspicious` must fire, `"neutralised"` when the item may
                             survive but must carry no leak and no instruction).
* `masking_cases.jsonl`    - `id`, `text`, `symbols`, `masked`: the exact 5.8 step-4 output (a golden per line).
* `stale_relative_cases.jsonl` - `id`, `age_hours`, `headline`, `hostile`: benign and hostile items whose text says
                             "tomorrow" / "later today" at ages from 1 h to 70 h (the code-side recency split, 5.6 / 5.8).

`created_at` is an RFC 3339 UTC instant; every other time is derived from `age_hours` by the test.
"""

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from jevbot.types import NewsItem

NEWS_DIR: Final = Path(__file__).resolve().parent
BENIGN: Final = "benign_items.jsonl"
HOSTILE: Final = "hostile_headlines.jsonl"
MASKING: Final = "masking_cases.jsonl"
STALE: Final = "stale_relative_cases.jsonl"


def load(name: str) -> list[dict[str, Any]]:
    """Every object of one corpus file, in file order."""
    path = NEWS_DIR / name
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("//"):
            continue
        row = json.loads(text)
        if not isinstance(row, dict):
            raise ValueError(f"{path}: every line must be a JSON object, got {type(row).__name__}")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: the corpus is empty")
    return rows


def _at(row: dict[str, Any], as_of: datetime, lag_s: int) -> datetime:
    if "created_at" in row:
        return datetime.fromisoformat(str(row["created_at"])).astimezone(UTC)
    return as_of - timedelta(hours=float(row["age_hours"])) - timedelta(seconds=lag_s)


def as_items(rows: list[dict[str, Any]], *, as_of: datetime, lag_s: int = 60, symbols: tuple[str, ...] = ("SPY",)) -> list[NewsItem]:
    """Corpus rows as `NewsItem`s. `created_at` is taken from the row, else derived from `age_hours` so that
    `knowable_at = created_at + lag_s` lands exactly `age_hours` before `as_of` (the 5.1 archive rule)."""
    items: list[NewsItem] = []
    for row in rows:
        created = _at(row, as_of, lag_s)
        items.append(
            NewsItem(
                id=str(row["id"]),
                created_at=created,
                updated_at=created,
                received_at=None,
                knowable_at=created + timedelta(seconds=lag_s),
                headline=str(row["headline"]),
                summary=row.get("summary"),
                source=str(row.get("source", "benzinga")),
                symbols=tuple(row.get("symbols", symbols)),
            )
        )
    return items


def iter_files() -> Iterator[Path]:
    """The four corpus files (a test asserts that all of them exist and parse)."""
    for name in (BENIGN, HOSTILE, MASKING, STALE):
        yield NEWS_DIR / name
