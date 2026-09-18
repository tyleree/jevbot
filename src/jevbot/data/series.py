"""`PitTable` and the table-backed news / event sources (DESIGN.md 3.2, 5.1, 5.4; WP01).

Every non-chain dataset (`bars:<UND>`, `daily:<UND>`, `volidx:<NAME>`, `rates`, `events`) is one `PitTable`: a frame plus
its point-in-time rules (D24, INV-14).

* the ROW gate is the tz-aware `knowable_at` column - the earliest moment ANY part of the row may be seen;
* `column_knowable` adds a SECOND, per-column gate for values that become visible later than their row. There are exactly
  two in the project (3.2): a bar row's `high` / `low` / `close` / `volume` behind `hlcv_knowable_at` (the next session's
  open - today's close is never readable at today's decision) and a `daily` row's `close_c` behind `close_knowable_at`;
* **range reads FILTER, then ASSERT**: `asof()` drops rows the caller may not see, nulls every gated value whose own
  timestamp is missing or in the future, and then re-asserts that nothing left is in the future (a defensive `PitViolation`
  that fires if a frame is ever mutated behind our back);
* **keyed reads RAISE**: `row()` / `value()` raise `PitViolation` for a record the caller may not see yet and
  `DataUnavailable` when there is no such row or the value has not been recorded. `MarketView.close()` IS `value()`.

`TableEventSource` serves ONLY scheduled, non-cancelled, knowable rows (5.1: an unscheduled 2008-10-08 meeting and a
cancelled meeting are stored for audit and `data verify`, never served). `TableNewsSource` applies the B6.3 rules (newest
first, summary dropped when the item was revised after `as_of`) and distinguishes "no archive" from "no news" through its
coverage ranges. `NullNewsSource` is the news-is-off source: no items, never covered.

Pure: no clock, no network, no filesystem. Frames come from `data/store.py`.
"""

import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any, Final

import msgspec
import numpy as np
import pandas as pd

from jevbot.errors import DataError, DataUnavailable, PitViolation
from jevbot.types import NewsItem, ScheduledEvent

__all__ = [
    "EVENT_KINDS",
    "KNOWABLE_AT",
    "NullNewsSource",
    "PitTable",
    "TableEventSource",
    "TableNewsSource",
]

KNOWABLE_AT: Final = "knowable_at"
# canonical order of the event kinds in `coverage()` (2.7 `ScheduledEvent.kind`)
EVENT_KINDS: Final[tuple[str, ...]] = ("fomc_decision", "cpi", "nfp", "ex_dividend")

# nullable replacements used when a gated column has to be able to hold NA (so `asof()` returns one stable dtype whether or
# not anything is gated on this particular read)
_NULLABLE_DTYPE: Final[Mapping[str, str]] = {
    "int8": "Int8",
    "int16": "Int16",
    "int32": "Int32",
    "int64": "Int64",
    "uint8": "UInt8",
    "uint16": "UInt16",
    "uint32": "UInt32",
    "uint64": "UInt64",
    "bool": "boolean",
}


def _as_utc(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be tz-aware (all datetimes are tz-aware UTC), got a naive datetime")
    return value.astimezone(UTC)


def _norm_key_part(value: object) -> object:
    """One component of a table key, in the form the index uses: dates / timestamps as `pd.Timestamp`, enums as their
    string value, numpy scalars as builtins. `date(2024, 5, 17)`, `pd.Timestamp("2024-05-17")` and `np.datetime64` of the
    same day are therefore ONE key."""
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value)
    if isinstance(value, datetime | date):
        return pd.Timestamp(value)
    if isinstance(value, str):  # StrEnum members compare equal to their value but must not keep the enum identity
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _python_value(value: Any) -> Any:
    """A pandas / numpy cell as a plain Python object (`Int64` -> int, `Timestamp` -> tz-aware datetime, NA -> None)."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime(warn=False)
    if isinstance(value, np.generic):
        return value.item()
    return value


class PitTable:
    """One point-in-time dataset: a frame, a unique key and its row / column `knowable_at` gates (3.2).

    `df` must carry a tz-aware `knowable_at` column with no missing values and a unique `key`. The frame is copied and
    re-indexed 0..n-1; gated value columns are converted to their nullable dtype once, so `asof()` returns the same dtypes
    whether or not this particular read gates anything. `paths` records the partitions the frame was read from, for the
    13.1 "every DataView logs the partitions it opens" check.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        name: str,
        key: str | Sequence[str] = "session",
        column_knowable: Mapping[str, str] | None = None,
        paths: Sequence[str] = (),
    ) -> None:
        self.name = name
        self.key: tuple[str, ...] = (key,) if isinstance(key, str) else tuple(key)
        self.column_knowable: Mapping[str, str] = dict(column_knowable or {})
        self.paths: tuple[str, ...] = tuple(paths)
        if not self.key:
            raise DataError(f"table {name}: the key must name at least one column")

        frame = df.copy().reset_index(drop=True)
        missing = [c for c in (*self.key, KNOWABLE_AT) if c not in frame.columns]
        if missing:
            raise DataError(f"table {name}: missing required columns {missing}")
        for value_column, ts_column in self.column_knowable.items():
            if value_column not in frame.columns:
                raise DataError(f"table {name}: column_knowable names an unknown value column {value_column!r}")
            if ts_column not in frame.columns:
                raise DataError(f"table {name}: column_knowable names an unknown timestamp column {ts_column!r}")
            self._require_utc(frame, ts_column)
            nullable = _NULLABLE_DTYPE.get(str(frame[value_column].dtype))
            if nullable is not None:
                frame[value_column] = frame[value_column].astype(nullable)

        self._require_utc(frame, KNOWABLE_AT)
        if bool(frame[KNOWABLE_AT].isna().any()):
            raise DataError(f"table {name}: `knowable_at` is the row gate and may never be null")

        self._df = frame
        self._index: dict[tuple[object, ...], int] = {}
        for position, row_key in enumerate(self._key_tuples(frame)):
            if row_key in self._index:
                raise DataError(f"table {name}: duplicate key {row_key} (the key {list(self.key)} must be unique)")
            self._index[row_key] = position

    def _require_utc(self, frame: pd.DataFrame, column: str) -> None:
        dtype = frame[column].dtype
        if not isinstance(dtype, pd.DatetimeTZDtype):
            raise DataError(f"table {self.name}: `{column}` must be a tz-aware datetime column, got dtype {dtype}")

    def _key_tuples(self, frame: pd.DataFrame) -> list[tuple[object, ...]]:
        columns = [[_norm_key_part(v) for v in frame[c].tolist()] for c in self.key]
        return [tuple(parts) for parts in zip(*columns, strict=True)]

    # --- introspection --------------------------------------------------------------------------------------------------

    def __len__(self) -> int:
        return int(len(self._df))

    def __repr__(self) -> str:
        return f"PitTable({self.name!r}, rows={len(self._df)}, key={list(self.key)})"

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self._df.columns)

    def frame(self) -> pd.DataFrame:
        """A copy of the whole table - NO point-in-time gate. Only `data verify` and the writers may use it."""
        return self._df.copy()

    def sha256(self) -> str:
        """Content digest of the normalised frame (the provenance `payload_sha256` of a whole-table read)."""
        return hashlib.sha256(self._df.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()

    # --- reads ----------------------------------------------------------------------------------------------------------

    def asof(self, as_of: datetime) -> pd.DataFrame:
        """Rows with `knowable_at <= as_of`, with every not-yet-knowable gated value nulled (3.2).

        The index labels are the row's position in the table: `DataView.daily()` picks rows by label and the caller can
        map a returned row back to the table. Range reads FILTER then ASSERT (D24).
        """
        when = pd.Timestamp(_as_utc("as_of", as_of))
        frame = self._df[self._df[KNOWABLE_AT] <= when].copy()
        if not frame.empty and frame[KNOWABLE_AT].max() > when:
            raise PitViolation(f"table {self.name}: a row knowable after {when.isoformat()} survived the as-of filter")
        for value_column, ts_column in self.column_knowable.items():
            gated = frame[ts_column].isna() | (frame[ts_column] > when)
            if bool(gated.any()):
                frame.loc[gated, value_column] = pd.NA
        return frame

    def row(self, key: object, as_of: datetime) -> pd.Series:
        """One row by key. `PitViolation` when the ROW is not knowable yet, `DataUnavailable` when it does not exist;
        gated columns that are not knowable yet come back NULL."""
        when = pd.Timestamp(_as_utc("as_of", as_of))
        position = self._position(key)
        row_knowable = self._df.at[position, KNOWABLE_AT]
        if row_knowable > when:
            raise PitViolation(
                f"table {self.name}: row {self._key_text(key)} is knowable at {row_knowable.isoformat()} > as_of {when.isoformat()}"
            )
        out = self._df.iloc[position].copy()
        for value_column, ts_column in self.column_knowable.items():
            stamp = out[ts_column]
            if pd.isna(stamp) or stamp > when:
                out[value_column] = pd.NA
        return out

    def value(self, key: object, column: str, as_of: datetime) -> object:
        """One column of one row, gated by ITS OWN knowable-at when it has one, else by the row's (3.2).

        `PitViolation` when that timestamp is after `as_of`; `DataUnavailable` when the row is absent, the gating timestamp
        has not been recorded yet, or the value itself is null. `MarketView.close()` is exactly this call.
        """
        when = pd.Timestamp(_as_utc("as_of", as_of))
        if column not in self._df.columns:
            raise DataError(f"table {self.name}: no column {column!r}")
        position = self._position(key)
        ts_column = self.column_knowable.get(column)
        if ts_column is None:
            gate = self._df.at[position, KNOWABLE_AT]
        else:
            gate = self._df.at[position, ts_column]
            if pd.isna(gate):
                raise DataUnavailable(f"table {self.name}: {column} of row {self._key_text(key)} has not been recorded yet")
        if gate > when:
            raise PitViolation(
                f"table {self.name}: {column} of row {self._key_text(key)} is knowable at {gate.isoformat()} > as_of {when.isoformat()}"
            )
        value = self._df.at[position, column]
        if pd.isna(value):
            raise DataUnavailable(f"table {self.name}: {column} of row {self._key_text(key)} is null")
        return _python_value(value)

    def knowable_at(self, key: object, column: str | None = None) -> datetime | None:
        """The effective gate of a row (or of one of its columns) - metadata, NOT a gated read: it never raises
        `PitViolation`. `None` when a column gate has not been recorded yet. Used for provenance (`ProvenanceInput`)."""
        position = self._position(key)
        ts_column = self.column_knowable.get(column) if column is not None else None
        stamp = self._df.at[position, ts_column if ts_column is not None else KNOWABLE_AT]
        if pd.isna(stamp):
            return None
        return pd.Timestamp(stamp).to_pydatetime(warn=False)

    def has(self, key: object) -> bool:
        return self._normalise(key) in self._index

    def _position(self, key: object) -> int:
        position = self._index.get(self._normalise(key))
        if position is None:
            raise DataUnavailable(f"table {self.name}: no row for key {self._key_text(key)}")
        return position

    def _normalise(self, key: object) -> tuple[object, ...]:
        parts = tuple(key) if isinstance(key, tuple) else (key,)
        if len(parts) != len(self.key):
            raise DataError(f"table {self.name}: key {key!r} has {len(parts)} parts, the key {list(self.key)} needs {len(self.key)}")
        return tuple(_norm_key_part(part) for part in parts)

    def _key_text(self, key: object) -> str:
        parts = tuple(key) if isinstance(key, tuple) else (key,)
        return "(" + ", ".join(str(part) for part in parts) + ")"


# ======================================================================================================================
# News
# ======================================================================================================================


class TableNewsSource:
    """`NewsSource` (3.2) over the `pq/news` archive: the item frame plus the coverage ranges.

    `items` columns: the `NewsItem` fields (`id, created_at, updated_at, received_at, knowable_at, headline, summary,
    source, symbols`); `coverage` columns: `underlying, start, end, fetched_at`. Coverage is what makes "the archive does
    not cover this session" distinguishable from "there was no news" (B6.3 / 5.8: an absent archive must never look like a
    quiet day).
    """

    def __init__(
        self,
        items: "pd.DataFrame | Sequence[NewsItem]",
        coverage: "pd.DataFrame | Sequence[tuple[str, date, date]] | None" = None,
    ) -> None:
        self._items: tuple[NewsItem, ...] = tuple(items if isinstance(items, Sequence) else _news_items(items))
        ranges: Sequence[tuple[str, date, date]]
        if coverage is None:
            ranges = ()
        elif isinstance(coverage, Sequence):
            ranges = coverage
        else:
            ranges = _coverage_ranges(coverage)
        self._coverage: tuple[tuple[str, date, date], ...] = tuple((str(u), s, e) for u, s, e in ranges)

    @classmethod
    def from_items(cls, items: Sequence[NewsItem], coverage: Sequence[tuple[str, date, date]] = ()) -> "TableNewsSource":
        """Build from structs (the recorder's JSONL archive and tests) instead of frames."""
        return cls(items, coverage)

    def items(self, underlying: str, as_of: datetime, lookback_hours: int) -> tuple[NewsItem, ...]:
        """Items naming `underlying`, knowable at `as_of`, created within the look-back; newest first; the summary is
        dropped when the item was revised after `as_of` (B6.3 rule 4: we may not see the revised text)."""
        when = _as_utc("as_of", as_of)
        if lookback_hours < 0:
            raise ValueError("lookback_hours must be >= 0")
        since = when - timedelta(hours=lookback_hours)
        chosen = [i for i in self._items if underlying in i.symbols and i.knowable_at <= when and i.created_at >= since]
        chosen.sort(key=lambda i: (i.knowable_at, i.id), reverse=True)
        return tuple(_drop_revised_summary(i, when) for i in chosen)

    def covered(self, underlying: str, session: date) -> bool:
        """True iff a fetched range of this underlying contains `session`."""
        return any(name == underlying and start <= session <= end for name, start, end in self._coverage)

    def coverage(self) -> tuple[tuple[str, date, date], ...]:
        return self._coverage

    def __len__(self) -> int:
        return len(self._items)


class NullNewsSource:
    """`NewsSource` for a run with news off (V12 / 5.8): no items, and `covered()` is False so the state builder records
    `no_archive` instead of pretending the day was quiet."""

    def items(self, underlying: str, as_of: datetime, lookback_hours: int) -> tuple[NewsItem, ...]:
        _as_utc("as_of", as_of)
        if lookback_hours < 0:
            raise ValueError("lookback_hours must be >= 0")
        return ()

    def covered(self, underlying: str, session: date) -> bool:
        return False


def _drop_revised_summary(item: NewsItem, as_of: datetime) -> NewsItem:
    if item.updated_at <= as_of:
        return item
    revised: NewsItem = msgspec.structs.replace(item, summary=None)
    return revised


def _news_items(frame: pd.DataFrame) -> list[NewsItem]:
    missing = [c for c in ("id", "created_at", "updated_at", "knowable_at", "headline", "source", "symbols") if c not in frame.columns]
    if missing:
        raise DataError(f"news frame is missing columns {missing}")
    out: list[NewsItem] = []
    for row in frame.to_dict("records"):
        symbols = row["symbols"]
        if isinstance(symbols, str):
            symbols = [s for s in symbols.split(",") if s]
        summary = row.get("summary")
        received = row.get("received_at")
        out.append(
            NewsItem(
                id=str(row["id"]),
                created_at=_utc_cell("created_at", row["created_at"]),
                updated_at=_utc_cell("updated_at", row["updated_at"]),
                received_at=None if received is None or pd.isna(received) else _utc_cell("received_at", received),
                knowable_at=_utc_cell("knowable_at", row["knowable_at"]),
                headline=str(row["headline"]),
                summary=None if summary is None or pd.isna(summary) else str(summary),
                source=str(row["source"]),
                symbols=tuple(str(s) for s in symbols),
            )
        )
    return out


def _coverage_ranges(frame: pd.DataFrame | None) -> list[tuple[str, date, date]]:
    if frame is None or len(frame) == 0:
        return []
    missing = [c for c in ("underlying", "start", "end") if c not in frame.columns]
    if missing:
        raise DataError(f"news coverage frame is missing columns {missing}")
    out: list[tuple[str, date, date]] = []
    for row in frame.to_dict("records"):
        out.append((str(row["underlying"]), _date_cell("start", row["start"]), _date_cell("end", row["end"])))
    return out


# ======================================================================================================================
# Scheduled events
# ======================================================================================================================


class TableEventSource:
    """`EventSource` (3.2) over `pq/events/events.csv`.

    `events()` serves ONLY rows that are `scheduled`, not `cancelled` and knowable at `as_of` (5.1): an unscheduled or
    emergency meeting (2008-10-08, 2020-03-15) and a meeting the Fed marks cancelled are stored for audit and listed by
    `data verify` through `all_events()`, and can never reach a decision.
    """

    def __init__(self, frame: "pd.DataFrame | Sequence[ScheduledEvent]") -> None:
        self._events: tuple[ScheduledEvent, ...] = tuple(frame if isinstance(frame, Sequence) else _scheduled_events(frame))

    @classmethod
    def from_events(cls, events: Sequence[ScheduledEvent]) -> "TableEventSource":
        """Build from structs (the fetcher's own rows and tests) instead of a frame."""
        return cls(events)

    def events(self, as_of: datetime, start: date, end: date, underlying: str | None = None) -> tuple[ScheduledEvent, ...]:
        """Knowable, scheduled, non-cancelled rows with `start <= event_date <= end`.

        `underlying=None` serves every such row; `underlying=U` serves the market-wide rows (`underlying is None`:
        fomc_decision / cpi / nfp) PLUS U's own rows (ex_dividend) - never another underlying's.
        """
        when = _as_utc("as_of", as_of)
        chosen = [
            e
            for e in self._events
            if e.scheduled
            and not e.cancelled
            and e.knowable_at <= when
            and start <= e.event_date <= end
            and (underlying is None or e.underlying in (None, underlying))
        ]
        chosen.sort(key=lambda e: (e.event_date, e.kind, e.underlying or ""))
        return tuple(chosen)

    def coverage(self) -> tuple[str, ...]:
        """Event kinds with at least one verified (scheduled, non-cancelled) row, in the canonical kind order."""
        kinds = {e.kind for e in self._events if e.scheduled and not e.cancelled}
        return (*[k for k in EVENT_KINDS if k in kinds], *sorted(kinds - set(EVENT_KINDS)))

    def all_events(self) -> tuple[ScheduledEvent, ...]:
        """EVERY stored row, unscheduled and cancelled ones included - `data verify` lists them; decisions never see them."""
        return self._events

    def __len__(self) -> int:
        return len(self._events)


def _scheduled_events(frame: pd.DataFrame) -> list[ScheduledEvent]:
    missing = [c for c in ("kind", "event_date", "scheduled", "knowable_at", "knowable_rule", "source_url", "fetched_at") if c not in frame.columns]
    if missing:
        raise DataError(f"events frame is missing columns {missing}")
    out: list[ScheduledEvent] = []
    for row in frame.to_dict("records"):
        underlying = row.get("underlying")
        amount = row.get("amount_cents")
        cancelled = row.get("cancelled", False)
        out.append(
            ScheduledEvent(
                kind=str(row["kind"]),
                event_date=_date_cell("event_date", row["event_date"]),
                underlying=None if underlying is None or pd.isna(underlying) else str(underlying),
                amount_cents=None if amount is None or pd.isna(amount) else int(amount),
                scheduled=_bool_cell("scheduled", row["scheduled"]),
                cancelled=False if cancelled is None or pd.isna(cancelled) else _bool_cell("cancelled", cancelled),
                knowable_at=_utc_cell("knowable_at", row["knowable_at"]),
                knowable_rule=str(row["knowable_rule"]),
                source_url=str(row["source_url"]),
                fetched_at=_utc_cell("fetched_at", row["fetched_at"]),
            )
        )
    return out


# ======================================================================================================================
# Cell parsers (a CSV / parquet cell -> the struct field it feeds)
# ======================================================================================================================


def _utc_cell(name: str, value: object) -> datetime:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise DataError(f"{name} must be a timestamp, got {value!r}")
    stamp = stamp.tz_localize(UTC) if stamp.tzinfo is None else stamp.tz_convert(UTC)
    out: datetime = stamp.to_pydatetime(warn=False)
    return out


def _date_cell(name: str, value: object) -> date:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise DataError(f"{name} must be a date, got {value!r}")
    parsed: date = stamp.date()
    return parsed


def _bool_cell(name: str, value: object) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "yes"):
            return True
        if text in ("false", "0", "no"):
            return False
    if isinstance(value, int | float | np.integer | np.floating):
        return bool(value)
    raise DataError(f"{name} must be a boolean, got {value!r}")
