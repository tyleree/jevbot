"""`DataView` - THE single point-in-time gate (DESIGN.md 3.2, 5.1, 5.4; INV-14; WP01).

Every data read of the system goes through this class: `StateBuilder`, `RiskEngine`, `SimBroker`, `CandidateGenerator`,
`FillModel` and the outcome resolver receive only a `MarketView`. `DataView` is the one concrete implementation over a
`ChainProvider` (enriched chains) plus the `PitTable`s of `data/series.py` (`daily:<UND>`, `bars:<UND>`, `volidx:<NAME>`,
`rates`), a `NewsSource` and an `EventSource`.

The rules it enforces, once, for every mode (mirror, synthetic, recorded, live):

* **Chains.** `chain()` re-checks `knowable_at <= as_of` even though the provider should never hand out a future snapshot -
  a buggy provider raises `PitViolation` here rather than leaking (INV-14).
* **One value per session.** `closes()` and `close()` read `eod` rows only, because `close_c` lives once per session on the
  `eod` row (5.4). `daily()` returns exactly ONE ROW PER SESSION: for each past session the row of the DESIGNATED SLOT (the
  slot of this view's own key) when it exists, else that session's `eod` row, then this snapshot's own row last. A `dec`
  decision is therefore compared with past `dec` rows wherever they exist, and a 3-slot recorder archive and the same data
  collapsed to one slot produce the identical frame - which is what makes `iv_rank`, `iv_chg_1w`, `skew_pctile` and
  `rv20_pctile` index by SESSION in every mode.
* **Completed sessions only.** `closes()` and `bars()` stop before this view's session; today's `high` / `low` / `close` /
  `volume` are gated by `hlcv_knowable_at` (column gating, 3.2), so the current session's close can never be used at the
  decision, in either mode (0.1 item 12). `today_open_ratio()` is the one thing today's bar row may contribute.
* **Filter vs raise.** Range reads filter (and assert); keyed reads (`close()`, the `daily()` own row) raise `PitViolation`.
  A missing table / snapshot is `DataUnavailable`; an existing table with no knowable rows gives an empty frame or Series.
* **Provenance.** Every read is recorded as a `ProvenanceInput` for the sidecar (2.7), with the knowable-at that actually
  gated it - and the partition paths the view opened are available for the 13.1 subset check.

Pure and deterministic: no clock (the `as_of` is injected), no RNG, no network.
"""

import hashlib
from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import msgspec
import numpy as np
import pandas as pd

from jevbot.data.series import PitTable
from jevbot.errors import DataError, DataUnavailable, PitViolation
from jevbot.types import Cents, ChainSnapshot, Fidelity, NewsItem, ProvenanceInput, ScheduledEvent, Slot, SnapshotKey

if TYPE_CHECKING:
    from jevbot.protocols import Calendar, ChainProvider, EventSource, MarketView, NewsSource

__all__ = ["DataView", "bars_table_name", "daily_table_name", "volidx_table_name"]

_RATES_TABLE = "rates"


def daily_table_name(underlying: str) -> str:
    """The `tables` key of an underlying's derived daily series (13.2)."""
    return f"daily:{underlying}"


def bars_table_name(underlying: str) -> str:
    """The `tables` key of an underlying's raw daily bars (13.1; ratios only, 5.2)."""
    return f"bars:{underlying}"


def volidx_table_name(name: str) -> str:
    """The `tables` key of a Cboe volatility index."""
    return f"volidx:{name}"


@runtime_checkable
class _OpensPartitions(Protocol):
    """A provider that can name the partition files it has opened (13.1: a run asserts the opened set is a subset of the
    selection). Optional: a provider without it simply contributes nothing."""

    def opened_partitions(self) -> tuple[str, ...]: ...


class DataView:
    """Implements `protocols.MarketView` (3.2) over a `ChainProvider` and the point-in-time tables."""

    def __init__(
        self,
        *,
        key: SnapshotKey,
        as_of: datetime,
        calendar: "Calendar",
        chains: "ChainProvider",
        tables: Mapping[str, PitTable],
        news: "NewsSource",
        events: "EventSource",
    ) -> None:
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must be tz-aware (all datetimes are tz-aware UTC)")
        self._key = key
        self._as_of = as_of.astimezone(UTC)
        self._calendar = calendar
        self._chains = chains
        self._tables = dict(tables)
        self._news = news
        self._events = events
        self._touched: list[ProvenanceInput] = []
        self._opened: list[str] = []

    def __repr__(self) -> str:
        return f"DataView({self._key.session.isoformat()} {self._key.slot.value} as_of={self._as_of.isoformat()})"

    # --- identity -------------------------------------------------------------------------------------------------------

    @property
    def as_of(self) -> datetime:
        return self._as_of

    @property
    def key(self) -> SnapshotKey:
        return self._key

    @property
    def session(self) -> date:
        return self._key.session

    @property
    def calendar(self) -> "Calendar":
        return self._calendar

    @property
    def fidelity(self) -> Fidelity:
        return self._chains.fidelity

    # --- provenance -----------------------------------------------------------------------------------------------------

    def touched(self) -> tuple[ProvenanceInput, ...]:
        """Every distinct read of this view, in the order it happened (the `Provenance.inputs` sidecar, 2.7)."""
        return tuple(self._touched)

    def opened_partitions(self) -> tuple[str, ...]:
        """The partition paths this view has opened (13.1: the engine asserts this set is a subset of the run's selection)."""
        return tuple(dict.fromkeys(self._opened))

    def _touch(self, field: str, source: str, payload: object, knowable_at: datetime | None, event_time: datetime | None = None) -> None:
        stamp = self._as_of if knowable_at is None else pd.Timestamp(knowable_at).to_pydatetime(warn=False).astimezone(UTC)
        item = ProvenanceInput(field=field, source=source, event_time=event_time, knowable_at=stamp, payload_sha256=_digest(payload))
        if item not in self._touched:
            self._touched.append(item)

    def _table(self, name: str) -> PitTable:
        table = self._tables.get(name)
        if table is None:
            raise DataUnavailable(f"no `{name}` table in this view")
        self._opened.extend(table.paths)
        return table

    # --- chains ---------------------------------------------------------------------------------------------------------

    def chain(self, underlying: str) -> ChainSnapshot:
        """The enriched chain snapshot of this (underlying, key). `DataUnavailable` when there is none; `PitViolation` when
        the provider hands out a snapshot that is not knowable yet (INV-14: a provider bug never leaks)."""
        chain = self._chains.get_chain(underlying, self._key)
        if chain is None:
            raise DataUnavailable(f"no chain snapshot for {underlying} at {self._key.session} {self._key.slot.value}")
        if chain.knowable_at > self._as_of:  # INV-14
            raise PitViolation(
                f"chain {underlying} {self._key.session} {self._key.slot.value} is knowable at "
                f"{chain.knowable_at.isoformat()} > as_of {self._as_of.isoformat()}"
            )
        if isinstance(self._chains, _OpensPartitions):
            self._opened.extend(self._chains.opened_partitions())
        self._touch(f"chain:{underlying}", f"{chain.source}.chain", chain.content_hash, chain.knowable_at, chain.ts)
        return chain

    def spot(self, underlying: str) -> Cents:
        """The decision-time reference price `ref` under the run's price measure (5.2)."""
        return self.chain(underlying).spot

    # --- the derived daily series (5.4) ---------------------------------------------------------------------------------

    def closes(self, underlying: str, n: int) -> pd.Series:
        """`close_c` of the last `n` COMPLETED sessions, oldest first: one value per session, from `eod` rows only."""
        if n < 0:
            raise ValueError("n must be >= 0")
        table = self._table(daily_table_name(underlying))
        rows = table.asof(self._as_of)
        rows = rows[(rows["slot"] == Slot.EOD.value) & (rows["session"] < pd.Timestamp(self.session)) & rows["close_c"].notna()]
        rows = rows.sort_values("session", kind="mergesort").tail(n)
        out = pd.Series(
            rows["close_c"].astype("int64").to_numpy(),
            index=pd.DatetimeIndex(rows["session"], name="session"),
            name="close_c",
            dtype="int64",
        )
        if not rows.empty:
            self._touch(f"closes:{underlying}", "derived.daily", out, rows["close_knowable_at"].max())
        return out

    def close(self, underlying: str, session: date) -> Cents:
        """That session's `close_c`. `PitViolation` before its `close_knowable_at`; `DataUnavailable` when the `eod` row is
        missing or carries no recorded close yet (5.4). This IS `PitTable.value()`."""
        table = self._table(daily_table_name(underlying))
        key = (session, Slot.EOD.value)
        raw = table.value(key, "close_c", self._as_of)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise DataError(f"close_c of {underlying} on {session.isoformat()} is not an integer number of cents: {raw!r}")
        self._touch(f"close:{underlying}:{session.isoformat()}", "derived.daily", raw, table.knowable_at(key, "close_c"))
        return raw

    def daily(self, underlying: str, n: int) -> pd.DataFrame:
        """The derived daily rows (13.2): ONE ROW PER SESSION - the last `n - 1` past sessions by the designated-slot rule,
        then this snapshot's own row last. `close_c` is nulled wherever it is not knowable yet (column gating)."""
        if n < 1:
            raise ValueError("n must be >= 1")
        table = self._table(daily_table_name(underlying))
        today = pd.Timestamp(self.session)
        slot = self._key.slot.value
        table.row((self.session, slot), self._as_of)  # PitViolation / DataUnavailable for the snapshot's OWN row
        rows = table.asof(self._as_of)
        own_labels = rows.index[(rows["session"] == today) & (rows["slot"] == slot)]
        if len(own_labels) != 1:
            raise DataUnavailable(f"daily row for {underlying} at {self.session} {slot} is not readable")
        chosen: list[int] = []
        past = rows[rows["session"] < today]
        for _, group in past.groupby("session", sort=True):
            designated = group.index[group["slot"] == slot]
            fallback = group.index[group["slot"] == Slot.EOD.value]
            if len(designated):
                chosen.append(int(designated[0]))
            elif len(fallback):
                chosen.append(int(fallback[0]))
        window = chosen[-(n - 1) :] if n > 1 else []
        out = rows.loc[[*window, int(own_labels[0])]].reset_index(drop=True)
        self._touch(f"daily:{underlying}", "derived.daily", out, out["knowable_at"].max())
        return out

    # --- raw bars (within-bar and adjacent-bar ratios ONLY, 5.2) --------------------------------------------------------

    def bars(self, underlying: str, n: int) -> pd.DataFrame:
        """The last `n` COMPLETED bars (`session, open, high, low, close`), oldest first. A bar counts as completed when its
        `hlcv_knowable_at` has passed, so today's row is never among them."""
        if n < 0:
            raise ValueError("n must be >= 0")
        table = self._table(bars_table_name(underlying))
        rows = table.asof(self._as_of)
        done = rows[(rows["hlcv_knowable_at"] <= pd.Timestamp(self._as_of)) & (rows["session"] < pd.Timestamp(self.session))]
        done = done.sort_values("session", kind="mergesort").tail(n)
        out = (
            done[["session", "open", "high", "low", "close"]]
            .astype({"open": "float64", "high": "float64", "low": "float64", "close": "float64"})
            .reset_index(drop=True)
        )
        if not done.empty:
            self._touch(f"bars:{underlying}", "raw.bars", out, done["hlcv_knowable_at"].max())
        return out

    def today_open_ratio(self, underlying: str) -> float | None:
        """`open(today) / close(previous session)` - the only use of today's bar row (its `open` is knowable at open + 60 s,
        its high / low / close / volume stay null until the next open). `None` when either side is missing."""
        table = self._table(bars_table_name(underlying))
        rows = table.asof(self._as_of)
        today = rows[rows["session"] == pd.Timestamp(self.session)]
        prev = rows[(rows["session"] < pd.Timestamp(self.session)) & rows["close"].notna()].sort_values("session", kind="mergesort")
        if today.empty or prev.empty or pd.isna(today.iloc[0]["open"]):
            return None
        previous_close = float(prev.iloc[-1]["close"])
        if previous_close <= 0.0:
            return None
        ratio = float(today.iloc[0]["open"]) / previous_close
        self._touch(f"bars_open:{underlying}", "raw.bars", repr(ratio), today.iloc[0]["open_knowable_at"])
        return ratio

    # --- side tables ----------------------------------------------------------------------------------------------------

    def vol_index(self, name: str, n: int) -> pd.Series:
        """The last `n` knowable closes of a Cboe index (the newest is normally the previous session's, D22)."""
        if n < 0:
            raise ValueError("n must be >= 0")
        table = self._table(volidx_table_name(name))
        rows = table.asof(self._as_of).sort_values("session", kind="mergesort").tail(n)
        out = pd.Series(
            rows["close"].to_numpy(dtype=np.float64),
            index=pd.DatetimeIndex(rows["session"], name="session"),
            name=name,
            dtype="float64",
        )
        if not rows.empty:
            self._touch(volidx_table_name(name), f"cboe.{name}", out, rows["knowable_at"].max())
        return out

    def rate(self) -> float:
        """The last knowable 13-week bill coupon-equivalent, decimal (D22: normally the previous session's value)."""
        table = self._table(_RATES_TABLE)
        rows = table.asof(self._as_of).sort_values("session", kind="mergesort")
        if rows.empty:
            raise DataUnavailable(f"no bill rate knowable at {self._as_of.isoformat()}")
        value = float(rows.iloc[-1]["rate_bp"]) / 1e4
        self._touch(_RATES_TABLE, "treasury.tbill_13w", repr(value), rows.iloc[-1]["knowable_at"])
        return value

    # --- events and news ------------------------------------------------------------------------------------------------

    def events(self, start: date, end: date, underlying: str | None = None) -> tuple[ScheduledEvent, ...]:
        """Knowable, scheduled, non-cancelled events in [start, end]. `underlying=U` serves the market-wide rows plus U's
        own ex-dividends, never another underlying's (3.2)."""
        out = self._events.events(self._as_of, start, end, underlying)
        if out:
            self._touch(
                f"events:{start.isoformat()}:{end.isoformat()}:{underlying or '*'}",
                "events",
                out,
                max(e.knowable_at for e in out),
            )
        return out

    def event_coverage(self) -> tuple[str, ...]:
        return self._events.coverage()

    def news(self, underlying: str, lookback_hours: int) -> tuple[NewsItem, ...]:
        out = self._news.items(underlying, self._as_of, lookback_hours)
        if out:
            self._touch(f"news:{underlying}", "news", out, max(i.knowable_at for i in out))
        return out

    def news_covered(self, underlying: str) -> bool:
        """True iff the news archive covers this session for this underlying ("no archive" is not "no news")."""
        return self._news.covered(underlying, self.session)


def _digest(obj: object) -> str:
    """The provenance `payload_sha256`: CSV bytes for frames / series, the text itself for strings, JSON otherwise."""
    if isinstance(obj, pd.DataFrame | pd.Series):
        text = obj.to_csv(lineterminator="\n")
    elif isinstance(obj, str):
        text = obj
    else:
        text = msgspec.json.encode(msgspec.to_builtins(obj)).decode("utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if TYPE_CHECKING:
    # Static proof, checked by the mypy gate: DataView implements the MarketView Protocol of 3.2.
    def _data_view_is_a_market_view(view: DataView) -> "MarketView":
        return view
