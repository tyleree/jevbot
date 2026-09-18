"""`build_market_data` - THE one function that turns any `ChainProvider` into everything `DataView` needs besides
chains (`bars:<U>`, `daily:<U>`, `volidx:<NAME>`, `rates`; a `NewsSource`; an `EventSource`).

Why this module exists: `cycle.py` and `backtest.py` must never assume the provider is `SyntheticProvider`. A second
`ChainProvider` (a future `jevbot.data.mirror.MirrorParquetProvider`, backed by the historical mirror parquet tree)
only has to supply chains PLUS one small extra method - `daily_closes(underlying: str) -> pd.Series` (index = session
date, values = int cents) - the same shape `SyntheticProvider.daily_closes` already carries. `build_market_data`
builds `bars:<U>` from that close series (open = high = low = close: a documented simplification for a provider that
carries no intraday range; `SyntheticProvider`'s own bars are built the same way, on its own official close, so both
sources reach `DataView.bars()` / `today_open_ratio()` identically) and `daily:<U>` from the provider's OWN ENRICHED
CHAINS via `data.surface.chain_nodes` / `const_maturity_iv` - so `iv30_bp`, `iv90_bp`, `atm_term_json` and hence
`iv_rank`, `em_1` / `em_5` / `em_hold` (the REQUIRED features of 5.3) are always derived from that provider's real
surface, never hard-coded to synthetic behaviour.

`volidx:<NAME>` tables are OPTIONAL (`vol_index=` - Cboe archives for a mirror provider, `SyntheticProvider.vol_index()`
for the synthetic one): `features.py` treats a missing vol-index table as `DataUnavailable` -> the feature reads
`"unavailable"` (5.3: `vix_pctile` / `vvix_pctile` / `skewidx_pctile` are NOT in `FeatureSet.required_ok`), so an
engine can run with zero vol-index coverage. `rates` is a flat `rate_bp` unless the caller passes a real series.

News / events are `NullNewsSource()` / an empty `TableEventSource` unless the caller supplies its own (this minimal
engine never builds `entry_text` / `manage_text` requests, so a real news archive would be inert here anyway - see
`cycle.py`'s skip list).
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from jevbot.config import Config
from jevbot.data.series import NullNewsSource, PitTable, TableEventSource
from jevbot.data.surface import chain_nodes, const_maturity_iv, term_of
from jevbot.data.view import bars_table_name, daily_table_name, volidx_table_name
from jevbot.errors import ConfigError, DataUnavailable
from jevbot.types import ChainSnapshot

if TYPE_CHECKING:
    from jevbot.protocols import Calendar, ChainProvider, EventSource, NewsSource

__all__ = ["MarketData", "build_market_data"]

_EARLIEST: Final = date(1990, 1, 1)  # the practical lower bound (XnysCalendar's own default floor)
_LATEST: Final = date(2050, 12, 31)
_MIN_DTE_FOR_IV: Final = 7  # data.surface.const_maturity_iv's own floor


@runtime_checkable
class _DailyClosesSource(Protocol):
    """The one extra method a `ChainProvider` needs beyond the frozen 3.2 protocol (duck-typed, checked with
    `isinstance`: `@runtime_checkable` Protocols support it for methods with no data members)."""

    def daily_closes(self, underlying: str) -> pd.Series: ...


@dataclass(frozen=True)
class MarketData:
    """What `run_backtest` consumes: a `ChainProvider` plus the `PitTable`s / news / events `DataView` needs."""

    provider: "ChainProvider"
    tables: dict[str, PitTable]
    news: "NewsSource"
    events: "EventSource"


def build_market_data(
    cfg: Config,
    calendar: "Calendar",
    provider: "ChainProvider",
    *,
    vol_index: Mapping[str, pd.Series] | None = None,
    rate_bp: int = 0,
    news: "NewsSource | None" = None,
    events: "EventSource | None" = None,
    iv_proxy: Mapping[str, pd.Series] | None = None,
) -> MarketData:
    """Every table `DataView` needs besides chains, from ANY `ChainProvider` (see the module docstring).

    `iv_proxy` (V10): per underlying, a scaled vol-index series (index = session date, values = iv30 in bp) that
    back-fills the `daily:<U>` IV history for sessions BEFORE the provider's first chain. Those rows carry
    `source = "proxy"` so `features.iv_hist_proxy_pct` reports them; they are never mixed with a chain row.
    """
    if not isinstance(provider, _DailyClosesSource):
        raise ConfigError(
            f"{type(provider).__name__} does not implement daily_closes(underlying) -> pd.Series; "
            "build_market_data needs it to build the bars:<U> / daily:<U> tables"
        )
    tables: dict[str, PitTable] = {}
    for underlying in provider.underlyings():
        closes = provider.daily_closes(underlying)
        tables[bars_table_name(underlying)] = _bars_table(underlying, closes, calendar)
        proxy = (iv_proxy or {}).get(underlying)
        tables[daily_table_name(underlying)] = _daily_table(underlying, provider, closes, calendar, proxy)
    for name, series in (vol_index or {}).items():
        tables[volidx_table_name(name)] = _volidx_table(name, series, calendar)
    tables["rates"] = _rates_table(_session_union(provider), rate_bp, calendar)
    return MarketData(
        provider=provider,
        tables=tables,
        news=news if news is not None else NullNewsSource(),
        events=events if events is not None else TableEventSource.from_events(()),
    )


def _session_union(provider: "ChainProvider") -> list[date]:
    sessions: set[date] = set()
    for underlying in provider.underlyings():
        sessions.update(k.session for k in provider.keys(underlying, _EARLIEST, _LATEST))
    return sorted(sessions)


# ======================================================================================================================
# bars:<U>  (raw daily bars; open == high == low == close on a close-only source, 5.2 WITHIN-BAR-RATIO convention)
# ======================================================================================================================


def _bars_table(underlying: str, closes: pd.Series, calendar: "Calendar") -> PitTable:
    sessions = [pd.Timestamp(s).date() for s in closes.index]
    if not sessions:
        raise ConfigError(f"build_market_data: daily_closes({underlying!r}) returned an empty series")
    open_knowable: list[pd.Timestamp] = []
    hlcv_knowable: list[pd.Timestamp] = []
    for session in sessions:
        # With close-only data, the synthetic "open" is also the final close.
        # It must not become visible before that close was knowable.
        open_knowable.append(pd.Timestamp(calendar.next_open_after(calendar.open_close(session)[1])))
        hlcv_knowable.append(pd.Timestamp(calendar.next_open_after(calendar.open_close(session)[1])))
    values = closes.to_numpy(dtype="float64")
    frame = pd.DataFrame(
        {
            "session": pd.to_datetime(sessions),
            "open": values,
            "high": values,
            "low": values,
            "close": values,
            "knowable_at": open_knowable,
            "open_knowable_at": open_knowable,
            "hlcv_knowable_at": hlcv_knowable,
        }
    )
    for column in ("knowable_at", "open_knowable_at", "hlcv_knowable_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return PitTable(
        frame,
        name=bars_table_name(underlying),
        key="session",
        column_knowable={"high": "hlcv_knowable_at", "low": "hlcv_knowable_at", "close": "hlcv_knowable_at"},
    )


# ======================================================================================================================
# daily:<U>  (13.2 derived columns, from the provider's OWN enriched chains)
# ======================================================================================================================


def _rv20_bp(closes: pd.Series) -> pd.Series:
    log_close = np.log(closes.astype("float64").clip(lower=1.0))
    returns = log_close.diff()
    rv = returns.rolling(20).apply(lambda x: math.sqrt(252.0 * float(np.mean(x * x))), raw=True)
    return (rv * 1e4).round()


def _chain_daily_row(chain: ChainSnapshot, calendar: "Calendar") -> dict[str, Any]:
    """The 13.2 surface-derived cells of one snapshot's `daily` row, from its OWN enriched chain."""
    try:
        nodes = chain_nodes(chain, calendar)
        term = term_of(nodes)
    except DataUnavailable:
        term = []
    row: dict[str, Any] = {"atm_term_json": None, "iv30_bp": None, "iv90_bp": None, "skew25_bp": None}
    if term:
        row["atm_term_json"] = term
        try:
            row["iv30_bp"] = round(const_maturity_iv(term, 30, min_dte=_MIN_DTE_FOR_IV)[0] * 1e4)
        except DataUnavailable:
            pass
        try:
            row["iv90_bp"] = round(const_maturity_iv(term, 90, min_dte=_MIN_DTE_FOR_IV)[0] * 1e4)
        except DataUnavailable:
            pass
    return row


def _proxy_rows(proxy: pd.Series, closes: pd.Series, rv_series: pd.Series, before: date, calendar: "Calendar") -> list[dict[str, Any]]:
    """V10 back-fill rows: one `eod` row per proxy session strictly before `before`, knowable at the next open."""
    rows: list[dict[str, Any]] = []
    for raw_session, iv_bp in proxy.items():
        session = pd.Timestamp(raw_session).date()
        if session >= before or pd.isna(iv_bp) or not calendar.is_session(session):
            continue
        close = closes.get(pd.Timestamp(session))
        if close is None or pd.isna(close):
            continue
        knowable = pd.Timestamp(calendar.next_open_after(calendar.open_close(session)[1]))
        rv = rv_series.get(pd.Timestamp(session))
        rows.append(
            {
                "atm_term_json": None,
                "iv30_bp": int(iv_bp),
                "iv90_bp": None,
                "skew25_bp": None,
                "session": pd.Timestamp(session),
                "slot": "eod",
                "px_c": int(close),
                "close_c": int(close),
                "close_knowable_at": knowable,
                "rv20_bp": None if rv is None or pd.isna(rv) else int(rv),
                "spot_measure": "file_close",
                "div_unmodelled": True,
                "basis_suspect": False,
                "source": "proxy",
                "knowable_at": knowable,
            }
        )
    return rows


def _daily_table(
    underlying: str, provider: "ChainProvider", closes: pd.Series, calendar: "Calendar", proxy: pd.Series | None = None
) -> PitTable:
    rv_series = _rv20_bp(closes)
    rows: list[dict[str, Any]] = []
    keys = provider.keys(underlying, _EARLIEST, _LATEST)
    if proxy is not None and keys:
        rows.extend(_proxy_rows(proxy, closes, rv_series, min(k.session for k in keys), calendar))
    for key in keys:
        chain = provider.get_chain(underlying, key)
        if chain is None:
            continue
        is_eod = key.slot.value == "eod"
        row = _chain_daily_row(chain, calendar)
        rv = rv_series.get(pd.Timestamp(key.session))
        row.update(
            session=pd.Timestamp(key.session),
            slot=key.slot.value,
            px_c=int(chain.spot),
            close_c=int(chain.spot) if is_eod else None,
            close_knowable_at=pd.Timestamp(chain.knowable_at) if is_eod else pd.NaT,
            rv20_bp=None if rv is None or pd.isna(rv) else int(rv),
            spot_measure=chain.spot_measure,
            div_unmodelled=bool(chain.div_unmodelled),
            basis_suspect=False,
            source=chain.source,
            knowable_at=pd.Timestamp(chain.knowable_at),
            atm_term_json=None if row["atm_term_json"] is None else _json_dumps(row["atm_term_json"]),
        )
        rows.append(row)
    if not rows:
        raise ConfigError(f"build_market_data: provider has no chains for {underlying!r}")
    frame = pd.DataFrame(rows)
    for column in ("session", "close_knowable_at", "knowable_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True) if column != "session" else pd.to_datetime(frame[column])
    return PitTable(frame, name=daily_table_name(underlying), key=("session", "slot"), column_knowable={"close_c": "close_knowable_at"})


def _json_dumps(term: list[tuple[float, float, int, int, int]]) -> str:
    import json

    return json.dumps([list(node) for node in term])


# ======================================================================================================================
# volidx:<NAME>, rates  (D22 lag: knowable at the next session's open)
# ======================================================================================================================


def _volidx_table(name: str, series: pd.Series, calendar: "Calendar") -> PitTable:
    sessions = [pd.Timestamp(s).date() for s in series.index]
    knowable = [pd.Timestamp(calendar.next_open_after(calendar.open_close(s)[1])) for s in sessions]
    frame = pd.DataFrame({"session": pd.to_datetime(sessions), "close": series.to_numpy(dtype="float64"), "knowable_at": knowable})
    frame["knowable_at"] = pd.to_datetime(frame["knowable_at"], utc=True)
    return PitTable(frame, name=volidx_table_name(name), key="session")


def _rates_table(sessions: list[date], rate_bp: int, calendar: "Calendar") -> PitTable:
    if not sessions:
        raise ConfigError("build_market_data: no sessions to build the rates table from")
    knowable = [pd.Timestamp(calendar.next_open_after(calendar.open_close(s)[1])) for s in sessions]
    frame = pd.DataFrame({"session": pd.to_datetime(sessions), "rate_bp": [int(rate_bp)] * len(sessions), "knowable_at": knowable})
    frame["knowable_at"] = pd.to_datetime(frame["knowable_at"], utc=True)
    return PitTable(frame, name="rates", key="session")
