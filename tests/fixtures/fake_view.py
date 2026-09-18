"""In-memory `MarketView` double - WP00's shared test double of `data.view.DataView` (DESIGN.md 3.2, 15.2).

`FakeView` holds whole tables (chains keyed by `(underlying, key)`, the `daily` / `bars` / `volidx` / `rates` frames in their
13.1 / 13.2 parquet layouts with their `knowable_at` columns, events, news) and answers every `MarketView` read with the
point-in-time rules of 3.2 / 5.1 / 5.4 (INV-14): range reads FILTER on `knowable_at <= as_of`, keyed reads RAISE
`PitViolation`, a missing snapshot / table / row is `DataUnavailable`, `closes()` / `bars()` return COMPLETED sessions only,
today's bar row exposes `open` but never `high` / `low` / `close` / `volume`, `daily()` returns one row per session by
the designated-slot rule with the snapshot's own row last, `close_c` is gated by its own `close_knowable_at`, news is
newest first with the summary dropped when `updated_at > as_of`, events serve only scheduled, non-cancelled, knowable rows.
`at(key=..., as_of=...)` gives another view over the same tables (truncation-invariance and multi-session tests).

`make_view()` builds a complete, coherent world on the chain factory: chains for the underlyings, ~300 sessions of
generated history (closes, bars, a `daily` series whose own row carries the chain's two-strike ATM term, constant-maturity
IVs, skew and realised vol; past rows carry anchored pseudo-random `iv30_bp` / `rv20_bp` / `skew25_bp`), the seven Cboe
indices, the bill rate and one scheduled FOMC decision inside the holding window - all deterministic per seed, nothing
from a clock or the environment. Frame conventions (also what consumers should expect from `DataView`): `session` columns
are `datetime64[ns]` like the chain's date columns; timestamps are `datetime64[ns, UTC]`; money is integer cents.

The self-tests at the bottom are collected by `tests/conftest.py` (`pytest_collect_file`).
"""

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

import msgspec
import numpy as np
import pandas as pd

from jevbot.cal import trading_time, year_fraction
from jevbot.config import DataConfig, DataSyntheticConfig
from jevbot.errors import DataUnavailable, PitViolation
from jevbot.types import (
    Cents,
    ChainSnapshot,
    Fidelity,
    NewsItem,
    ProvenanceInput,
    Right,
    ScheduledEvent,
    Slot,
    SnapshotKey,
)
from tests.fixtures.chain_factory import DEFAULT_SESSION, make_chain, snapshot_ts, xnys

if TYPE_CHECKING:
    from jevbot.protocols import Calendar

__all__ = [
    "BARS_COLUMNS",
    "DAILY_COLUMNS",
    "EVENT_KINDS",
    "FakeView",
    "History",
    "ex_dividend_event",
    "fomc_event",
    "make_history",
    "make_rates",
    "make_view",
    "make_vol_indices",
    "news_item",
]

DAILY_COLUMNS: Final[tuple[str, ...]] = (  # 13.2
    "session",
    "slot",
    "px_c",
    "close_c",
    "close_knowable_at",
    "iv30_bp",
    "iv30_2s_bp",
    "iv90_bp",
    "skew25_bp",
    "atm_term_json",
    "rv20_bp",
    "spot_measure",
    "div_unmodelled",
    "basis_suspect",
    "source",
    "knowable_at",
)
BARS_COLUMNS: Final[tuple[str, ...]] = (
    "session",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "knowable_at",
    "open_knowable_at",
    "hlcv_knowable_at",
)
EVENT_KINDS: Final[tuple[str, ...]] = ("fomc_decision", "cpi", "nfp", "ex_dividend")
_FOMC_URL: Final = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
_BAR_ROW_LAG_S: Final = 60  # open(D) + 60 s = the bar row's knowable_at (5.1)
_SESSIONS_PER_YEAR: Final = 252
_START_PRICE_CENTS: Final[Mapping[str, Cents]] = {u: round(p * 100) for u, p in DataSyntheticConfig().start_price_usd.items()}
_MILLI_PER_CENT: Final = 10


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be tz-aware UTC")
    return value.astimezone(UTC)


def _as_calendar(calendar: "Calendar | None") -> "Calendar":
    return xnys() if calendar is None else calendar


def _rng(seed: int, purpose: str) -> np.random.Generator:
    """The 10.9 seeding recipe: draws never depend on call order."""
    digest = hashlib.sha256(f"{seed}|{purpose}".encode()).hexdigest()
    return np.random.Generator(np.random.PCG64(int(digest[:16], 16)))


def _sha256_of(obj: object) -> str:
    """The provenance `payload_sha256`: CSV bytes for frames / series, the text itself for strings, JSON otherwise."""
    if isinstance(obj, pd.DataFrame | pd.Series):
        text = obj.to_csv(lineterminator="\n")
    elif isinstance(obj, str):
        text = obj
    else:
        text = msgspec.json.encode(msgspec.to_builtins(obj)).decode("utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sessions_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "session" not in out.columns:
        raise ValueError("a FakeView table needs a `session` column")
    out["session"] = pd.to_datetime(out["session"])
    for column in out.columns:
        if column.endswith("knowable_at"):
            out[column] = pd.to_datetime(out[column], utc=True)
    return out.sort_values("session", kind="mergesort").reset_index(drop=True)


# ======================================================================================================================
# Event / news constructors
# ======================================================================================================================


def fomc_event(event_date: date, *, knowable_at: datetime | None = None, scheduled: bool = True, cancelled: bool = False) -> ScheduledEvent:
    """A (historical, scheduled) FOMC decision: knowable 45 days ahead at 00:00 UTC by the stated assumption of 5.1."""
    known = (
        _utc(knowable_at, "knowable_at")
        if knowable_at is not None
        else datetime.combine(event_date - timedelta(days=DataConfig().fomc_knowable_days), datetime.min.time(), tzinfo=UTC)
    )
    return ScheduledEvent(
        kind="fomc_decision",
        event_date=event_date,
        underlying=None,
        amount_cents=None,
        scheduled=scheduled,
        cancelled=cancelled,
        knowable_at=known,
        knowable_rule="fetched_at" if knowable_at is not None else "scheduled_minus_45d_assumption",
        source_url=_FOMC_URL,
        fetched_at=known,
    )


def ex_dividend_event(underlying: str, ex_date: date, amount_cents: Cents, *, knowable_at: datetime | None = None) -> ScheduledEvent:
    """A verified ex-dividend date: knowable 14 days ahead at 00:00 UTC by the stated assumption of 5.1 (G8)."""
    known = (
        _utc(knowable_at, "knowable_at")
        if knowable_at is not None
        else datetime.combine(ex_date - timedelta(days=DataConfig().exdiv_knowable_days), datetime.min.time(), tzinfo=UTC)
    )
    return ScheduledEvent(
        kind="ex_dividend",
        event_date=ex_date,
        underlying=underlying,
        amount_cents=amount_cents,
        scheduled=True,
        cancelled=False,
        knowable_at=known,
        knowable_rule="fetched_at" if knowable_at is not None else "exdiv_minus_14d_assumption",
        source_url="https://data.alpaca.markets/v1/corporate-actions",
        fetched_at=known,
    )


def news_item(
    id: str,
    headline: str,
    *,
    created_at: datetime,
    symbols: Sequence[str] = ("SPY",),
    summary: str | None = None,
    updated_at: datetime | None = None,
    received_at: datetime | None = None,
    source: str = "benzinga",
    lag_s: int = 60,
) -> NewsItem:
    """A NewsItem with the 5.1 knowability rule: `received_at` when recorded, else `created_at + lag_s`."""
    created = _utc(created_at, "created_at")
    received = _utc(received_at, "received_at") if received_at is not None else None
    return NewsItem(
        id=id,
        created_at=created,
        updated_at=_utc(updated_at, "updated_at") if updated_at is not None else created,
        received_at=received,
        knowable_at=received if received is not None else created + timedelta(seconds=lag_s),
        headline=headline,
        summary=summary,
        source=source,
        symbols=tuple(symbols),
    )


# ======================================================================================================================
# History generator (anchored at a chain snapshot)
# ======================================================================================================================


@dataclass(frozen=True)
class History:
    """Generated per-underlying history ending at a chain snapshot: the `daily` frame (13.2), the raw `bars` frame (13.1)
    and the `iv30_bp` series (session -> bp) that the vol indices are derived from."""

    underlying: str
    daily: pd.DataFrame
    bars: pd.DataFrame
    iv30_bp: pd.Series


def _two_strike_atm(chain: ChainSnapshot, expiry: date) -> float | None:
    """The two-strike ATM IV of 5.3: strikes K1 <= F < K2 bracketing the forward, per strike the OTM side's IV (put below
    F, call above; the other side if missing), linear interpolation in strike to F."""
    fwd = chain.forward(expiry)
    calls = chain.side(expiry, Right.CALL).set_index("strike_milli")["iv"]
    puts = chain.side(expiry, Right.PUT).set_index("strike_milli")["iv"]
    below = [s for s in puts.index.union(calls.index) if s / _MILLI_PER_CENT <= fwd]
    above = [s for s in puts.index.union(calls.index) if s / _MILLI_PER_CENT > fwd]
    if not below or not above:
        return None
    k1, k2 = max(below), min(above)

    def iv_at(strike: int, prefer: pd.Series, other: pd.Series) -> float | None:
        for series in (prefer, other):
            if strike in series.index and not pd.isna(series[strike]):
                return float(series[strike])
        return None

    iv1, iv2 = iv_at(k1, puts, calls), iv_at(k2, calls, puts)
    if iv1 is None or iv2 is None:
        return None
    w = (fwd - k1 / _MILLI_PER_CENT) / ((k2 - k1) / _MILLI_PER_CENT)
    return float(iv1 + w * (iv2 - iv1))


def atm_term(chain: ChainSnapshot, *, calendar: "Calendar | None" = None) -> list[tuple[float, float, int, int, int]]:
    """`[(tau_years, tt_sessions, atm_iv_bp, atm_iv_2s_bp, fwd_c), ...]` for every listed expiry with dte >= 1 that has a
    two-strike ATM IV, ordered by last_session (3.7). The primary and the two-strike value coincide here (no fit)."""
    cal = _as_calendar(calendar)
    out: list[tuple[float, float, int, int, int]] = []
    for expiry in chain.expiries():
        atm = _two_strike_atm(chain, expiry)
        if atm is None:
            continue
        close = cal.open_close(chain.last_session(expiry))[1]
        bp = round(atm * 10_000)
        out.append((year_fraction(chain.ts, close), trading_time(cal, chain.ts, close), bp, bp, chain.forward(expiry)))
    return out


def _const_maturity_iv_bp(term: Sequence[tuple[float, float, int, int, int]], days: int, *, min_dte_days: int = 7) -> int | None:
    """Total-variance interpolation in CALENDAR time over nodes with dte >= 7; one-sided => the nearest node (3.7)."""
    nodes = [(tau, bp) for tau, _, bp, _, _ in term if tau * 365.0 >= min_dte_days - 0.5]
    if not nodes:
        return None
    target = days / 365.0
    below = [n for n in nodes if n[0] <= target]
    above = [n for n in nodes if n[0] >= target]
    if below and above:
        (t1, v1), (t2, v2) = max(below), min(above)
        if t2 == t1:
            return v1
        w1, w2 = (v1 / 1e4) ** 2 * t1, (v2 / 1e4) ** 2 * t2
        w = w1 + (w2 - w1) * (target - t1) / (t2 - t1)
        return round(math.sqrt(w / target) * 1e4)
    nearest = min(nodes, key=lambda n: abs(n[0] - target))
    return nearest[1]


def _skew25_bp(chain: ChainSnapshot, atm_bp: int | None) -> int | None:
    """`(iv(put, |delta| nearest 0.25) - iv(call, |delta| nearest 0.25)) / iv_atm` at the expiry nearest 30 DTE with dte >= 7."""
    if atm_bp is None or atm_bp <= 0:
        return None
    table = chain.table[chain.table["dte"] >= 7]
    if table.empty:
        return None
    nearest = table.iloc[(table["dte"] - 30).abs().argmin()]
    rows = table[table["expiry"] == nearest["expiry"]]
    rows = rows[rows["delta"].notna()]
    puts, calls = rows[rows["right"] == "P"], rows[rows["right"] == "C"]
    if puts.empty or calls.empty:
        return None
    put_iv = float(puts.iloc[(puts["delta"].abs() - 0.25).abs().argmin()]["iv"])
    call_iv = float(calls.iloc[(calls["delta"].abs() - 0.25).abs().argmin()]["iv"])
    return round((put_iv - call_iv) / (atm_bp / 1e4) * 1e4)


def _rv20_bp(log_returns: np.ndarray[Any, np.dtype[np.float64]]) -> list[int | None]:
    """rv20 = sqrt(252 * mean(r[-20:]^2)) per row, None until 20 returns exist (5.3)."""
    out: list[int | None] = []
    for i in range(log_returns.size):
        if i + 1 < 20:
            out.append(None)
        else:
            window = log_returns[i - 19 : i + 1]
            out.append(round(math.sqrt(_SESSIONS_PER_YEAR * float(np.mean(window * window))) * 1e4))
    return out


def make_history(
    chain: ChainSnapshot,
    *,
    n_sessions: int = 300,
    seed: int = 0,
    daily_vol: float = 0.14,
    drift_bp: float = 4.0,
    slots: Sequence[Slot] = (Slot.EOD,),
    calendar: "Calendar | None" = None,
) -> History:
    """Deterministic history for `chain.underlying` over the `n_sessions` sessions ending at the chain's session.

    Closes: log returns `N(drift_bp / 1e4, daily_vol / sqrt(252))` generated backwards from the chain's spot, so the last
    completed close is one plausible return away from `ref`. Bars follow the 5.10 recipe (open = a quarter gap, high / low
    around open / close). `daily` rows: `eod` rows carry `close_c` (mirror style: `close_knowable_at` = the close); extra
    `slots` (dec / exec) carry perturbed `px_c` and no `close_c` (a 3-slot recorder archive). The snapshot's own row holds
    the chain's two-strike ATM term, const-maturity `iv30_bp` / `iv90_bp`, `skew25_bp` and `rv20_bp`; past rows carry an
    AR(1) `iv30_bp` anchored to it and `atm_term_json = null`.
    """
    cal = _as_calendar(calendar)
    if n_sessions < 2:
        raise ValueError("n_sessions must be >= 2")
    if chain.key.slot not in slots:
        raise ValueError(f"slots must include the chain's own slot {chain.key.slot.value}")
    session = chain.key.session
    sessions = cal.sessions(cal.prev_session(session, n_sessions - 1), session)
    if len(sessions) != n_sessions or sessions[-1] != session:
        raise ValueError("the calendar does not give n_sessions sessions ending at the chain's session")
    u = chain.underlying
    rng = _rng(seed, f"history|{u}")
    sigma_d = daily_vol / math.sqrt(_SESSIONS_PER_YEAR)
    draws = drift_bp / 1e4 + sigma_d * rng.standard_normal(n_sessions)  # r[i] = ln(c[i] / c[i-1]); r[-1] is today's move
    closes = np.empty(n_sessions, dtype=np.int64)
    closes[-1] = chain.spot
    level = float(chain.spot)
    for i in range(n_sessions - 2, -1, -1):
        level = level / math.exp(draws[i + 1])
        closes[i] = round(level)
    prev_closes = np.concatenate(([round(closes[0] / math.exp(draws[0]))], closes[:-1]))
    # the returns features see: log differences of the ROUNDED closes (the last one is ref versus the last completed close)
    returns = np.diff(np.log(np.concatenate((prev_closes[:1], closes)).astype(np.float64)))
    z_open, z_hl = rng.standard_normal(n_sessions), rng.standard_normal(n_sessions)
    opens = prev_closes * np.exp(0.25 * np.abs(returns) * np.sign(z_open + 1e-12))
    highs = np.maximum(opens, closes) * np.exp(0.3 * sigma_d * np.abs(z_hl))
    lows = np.minimum(opens, closes) * np.exp(-0.3 * sigma_d * np.abs(z_hl))
    volumes = (50_000_000 + rng.integers(0, 20_000_000, n_sessions)).astype(np.int64)

    opens_at = [cal.open_close(s)[0] for s in sessions]
    closes_at = [cal.open_close(s)[1] for s in sessions]
    next_opens = [cal.next_open_after(c) for c in closes_at]
    bars = pd.DataFrame(
        {
            "session": pd.to_datetime(sessions),
            "open": np.round(opens / 100.0, 2),
            "high": np.round(highs / 100.0, 2),
            "low": np.round(lows / 100.0, 2),
            "close": closes / 100.0,
            "volume": volumes,
            "knowable_at": [o + timedelta(seconds=_BAR_ROW_LAG_S) for o in opens_at],
            "open_knowable_at": [o + timedelta(seconds=_BAR_ROW_LAG_S) for o in opens_at],
            "hlcv_knowable_at": next_opens,
        }
    )
    bars["knowable_at"] = pd.to_datetime(bars["knowable_at"], utc=True)
    bars["open_knowable_at"] = pd.to_datetime(bars["open_knowable_at"], utc=True)
    bars["hlcv_knowable_at"] = pd.to_datetime(bars["hlcv_knowable_at"], utc=True)

    # the own row's surface numbers from the chain itself
    term = atm_term(chain, calendar=cal)
    iv30_today = _const_maturity_iv_bp(term, 30)
    iv90_today = _const_maturity_iv_bp(term, 90)
    skew_today = _skew25_bp(chain, iv30_today)
    if iv30_today is None:
        iv30_today = 1600
    # anchored AR(1) log-IV history: x[-1] = 0 => iv30[-1] = today's value
    x = np.zeros(n_sessions)
    shocks = 0.04 * rng.standard_normal(n_sessions)
    for i in range(1, n_sessions):
        x[i] = 0.97 * x[i - 1] + shocks[i]
    x -= x[-1]
    iv30_hist = np.round(iv30_today * np.exp(x)).astype(np.int64)
    iv30_hist[-1] = iv30_today
    iv90_hist = np.round(iv30_hist * (1.0 + 0.05 * math.log(3.0))).astype(np.int64)
    skew_hist = np.round((skew_today if skew_today is not None else 2000) + 150.0 * rng.standard_normal(n_sessions)).astype(np.int64)
    rv20 = _rv20_bp(returns)

    rows: list[dict[str, Any]] = []
    perturb = rng.standard_normal((n_sessions, len(slots)))
    for i, s in enumerate(sessions):
        own_session = s == session
        for j, slot in enumerate(slots):
            own = own_session and slot is chain.key.slot
            if own:
                px = chain.spot
                known = chain.knowable_at
            else:
                px = int(closes[i]) if slot is Slot.EOD else round(closes[i] * math.exp(0.3 * sigma_d * perturb[i, j]))
                known = (
                    closes_at[i]
                    if chain.fidelity in (Fidelity.EOD_QUOTES, Fidelity.SYNTHETIC)
                    else snapshot_ts(s, slot, chain.fidelity, calendar=cal)
                )
            eod = slot is Slot.EOD
            rows.append(
                {
                    "session": pd.Timestamp(s),
                    "slot": slot.value,
                    "px_c": px,
                    "close_c": (chain.spot if own else int(closes[i])) if eod else None,
                    "close_knowable_at": (chain.ts if own else closes_at[i]) if eod else None,
                    "iv30_bp": int(iv30_hist[i]) if not own else iv30_today,
                    "iv30_2s_bp": int(iv30_hist[i]) if not own else iv30_today,
                    "iv90_bp": (int(iv90_hist[i]) if not own else iv90_today),
                    "skew25_bp": int(skew_hist[i]) if not own else skew_today,
                    "atm_term_json": json.dumps([list(node) for node in term]) if own else None,
                    "rv20_bp": rv20[i],
                    "spot_measure": chain.spot_measure,
                    "div_unmodelled": chain.div_unmodelled,
                    "basis_suspect": False,
                    "source": chain.source,
                    "knowable_at": known,
                }
            )
    daily = pd.DataFrame(rows, columns=list(DAILY_COLUMNS))
    daily["session"] = pd.to_datetime(daily["session"])
    daily["close_c"] = daily["close_c"].astype("Int64")
    daily["close_knowable_at"] = pd.to_datetime(daily["close_knowable_at"], utc=True)
    daily["knowable_at"] = pd.to_datetime(daily["knowable_at"], utc=True)
    for column in ("iv30_bp", "iv30_2s_bp", "iv90_bp", "skew25_bp", "rv20_bp"):
        daily[column] = daily[column].astype("Int32")
    daily["atm_term_json"] = daily["atm_term_json"].astype("object")
    iv30_series = pd.Series(iv30_hist, index=pd.DatetimeIndex(pd.to_datetime(sessions), name="session"), name="iv30_bp")
    return History(underlying=u, daily=daily, bars=bars, iv30_bp=iv30_series)


def make_vol_indices(
    iv30_bp: pd.Series, *, seed: int = 0, names: Sequence[str] | None = None, calendar: "Calendar | None" = None
) -> dict[str, pd.DataFrame]:
    """`volidx:<NAME>` frames (`session, close, knowable_at` = next session open, D22) derived from an `iv30_bp` series:
    VIX = iv30 in percent points x 1.05, VIX9D = 0.97 VIX, VIX3M = 1.06 VIX, VXN = 1.2 VIX, RVX = 1.25 VIX, VVIX = 90 + 10 z,
    SKEW = 125 + 3 z (the 5.10 shapes). `names` defaults to `data.cboe_indices`."""
    cal = _as_calendar(calendar)
    if names is None:
        names = DataConfig().cboe_indices
    sessions = [pd.Timestamp(s).date() for s in iv30_bp.index]
    knowable = pd.to_datetime([cal.next_open_after(cal.open_close(s)[1]) for s in sessions], utc=True)
    vix = iv30_bp.to_numpy(dtype=np.float64) / 100.0 * 1.05
    rng = _rng(seed, "volidx")
    z = rng.standard_normal((2, len(sessions)))
    shapes: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {
        "VIX": vix,
        "VIX9D": 0.97 * vix,
        "VIX3M": 1.06 * vix,
        "VVIX": 90.0 + 10.0 * z[0],
        "SKEW": 125.0 + 3.0 * z[1],
        "VXN": 1.2 * vix,
        "RVX": 1.25 * vix,
    }
    out: dict[str, pd.DataFrame] = {}
    for name in names:
        if name not in shapes:
            raise ValueError(f"no synthetic shape for vol index {name}")
        out[name] = pd.DataFrame({"session": pd.to_datetime(sessions), "close": np.round(shapes[name], 2), "knowable_at": knowable})
    return out


def make_rates(sessions: Sequence[date], rate: float, *, calendar: "Calendar | None" = None) -> pd.DataFrame:
    """The `rates` frame (`session, rate_bp, knowable_at` = next session open) at a constant rate."""
    cal = _as_calendar(calendar)
    knowable = pd.to_datetime([cal.next_open_after(cal.open_close(s)[1]) for s in sessions], utc=True)
    return pd.DataFrame({"session": pd.to_datetime(list(sessions)), "rate_bp": round(rate * 1e4), "knowable_at": knowable})


# ======================================================================================================================
# The view
# ======================================================================================================================


@dataclass
class _Tables:
    chains: dict[tuple[str, SnapshotKey], ChainSnapshot] = field(default_factory=dict)
    daily: dict[str, pd.DataFrame] = field(default_factory=dict)
    bars: dict[str, pd.DataFrame] = field(default_factory=dict)
    vol_indices: dict[str, pd.DataFrame] = field(default_factory=dict)
    rates: pd.DataFrame | float | None = None
    events: tuple[ScheduledEvent, ...] = ()
    event_coverage: tuple[str, ...] | None = None
    news: tuple[NewsItem, ...] = ()
    news_covered: bool | Mapping[str, bool] | None = None


class FakeView:
    """Implements `protocols.MarketView` over in-memory tables (module docstring). Attributes `as_of`, `key`, `session`,
    `calendar`, `fidelity` are plain attributes, as the Protocol declares them."""

    as_of: datetime
    key: SnapshotKey
    session: date
    calendar: "Calendar"
    fidelity: Fidelity

    def __init__(
        self,
        *,
        key: SnapshotKey,
        as_of: datetime,
        calendar: "Calendar | None" = None,
        fidelity: Fidelity = Fidelity.EOD_QUOTES,
        chains: Iterable[ChainSnapshot] | Mapping[str, ChainSnapshot] = (),
        daily: Mapping[str, pd.DataFrame] | None = None,
        bars: Mapping[str, pd.DataFrame] | None = None,
        vol_indices: Mapping[str, pd.DataFrame | pd.Series] | None = None,
        rates: pd.DataFrame | float | None = None,
        events: Sequence[ScheduledEvent] = (),
        event_coverage: Sequence[str] | None = None,
        news: Sequence[NewsItem] = (),
        news_covered: bool | Mapping[str, bool] | None = None,
        _tables: _Tables | None = None,
    ) -> None:
        self.key = key
        self.session = key.session
        self.as_of = _utc(as_of, "as_of")
        self.calendar = _as_calendar(calendar)
        self.fidelity = fidelity
        self._touched: list[ProvenanceInput] = []
        if _tables is not None:
            self._t = _tables
            return
        t = _Tables()
        for chain in chains.values() if isinstance(chains, Mapping) else chains:
            t.chains[(chain.underlying, chain.key)] = chain
        t.daily = {u: _sessions_col(df) for u, df in (daily or {}).items()}
        t.bars = {u: _sessions_col(df) for u, df in (bars or {}).items()}
        for name, obj in (vol_indices or {}).items():
            frame = obj.rename("close").rename_axis("session").reset_index() if isinstance(obj, pd.Series) else obj
            t.vol_indices[name] = _sessions_col(frame)
        t.rates = _sessions_col(rates) if isinstance(rates, pd.DataFrame) else rates
        t.events = tuple(events)
        t.event_coverage = tuple(event_coverage) if event_coverage is not None else None
        t.news = tuple(news)
        t.news_covered = news_covered
        self._t = t

    # --- construction helpers ---------------------------------------------------------------------------------------

    def at(self, *, key: SnapshotKey | None = None, as_of: datetime | None = None, fidelity: Fidelity | None = None) -> "FakeView":
        """Another view over the SAME tables at another (key, as_of): PIT and multi-session tests."""
        k = self.key if key is None else key
        when = self.as_of if as_of is None else as_of
        return FakeView(
            key=k, as_of=when, calendar=self.calendar, fidelity=self.fidelity if fidelity is None else fidelity, _tables=self._t
        )

    def add_chain(self, chain: ChainSnapshot) -> None:
        """Register (or replace) the snapshot for `(chain.underlying, chain.key)` in the shared tables."""
        self._t.chains[(chain.underlying, chain.key)] = chain

    @property
    def tables(self) -> _Tables:
        """The underlying store (tests may poison future rows here to prove they are never read)."""
        return self._t

    # --- provenance --------------------------------------------------------------------------------------------------

    def _touch(self, name: str, payload: object, knowable_at: datetime | pd.Timestamp, event_time: datetime | None = None) -> None:
        stamp = pd.Timestamp(knowable_at).to_pydatetime(warn=False).astimezone(UTC)
        item = ProvenanceInput(
            field=name, source=f"fake.{name.split(':', 1)[0]}", event_time=event_time, knowable_at=stamp, payload_sha256=_sha256_of(payload)
        )
        if item not in self._touched:
            self._touched.append(item)

    def touched(self) -> tuple[ProvenanceInput, ...]:
        return tuple(self._touched)

    # --- chains ------------------------------------------------------------------------------------------------------

    def chain(self, underlying: str) -> ChainSnapshot:
        chain = self._t.chains.get((underlying, self.key))
        if chain is None:
            raise DataUnavailable(f"no chain snapshot for {underlying} at {self.key.session} {self.key.slot.value}")
        if chain.knowable_at > self.as_of:  # INV-14
            raise PitViolation(
                f"chain {underlying} {self.key.session} {self.key.slot.value} is knowable at {chain.knowable_at.isoformat()} > as_of {self.as_of.isoformat()}"
            )
        self._touch(f"chain:{underlying}", chain.content_hash, chain.knowable_at, chain.ts)
        return chain

    def spot(self, underlying: str) -> Cents:
        return self.chain(underlying).spot

    # --- daily series ------------------------------------------------------------------------------------------------

    def _daily_table(self, underlying: str) -> pd.DataFrame:
        df = self._t.daily.get(underlying)
        if df is None:
            raise DataUnavailable(f"no daily series for {underlying}")
        return df

    def _eod_rows(self, underlying: str) -> pd.DataFrame:
        df = self._daily_table(underlying)
        return df[(df["slot"] == Slot.EOD.value) & (df["knowable_at"] <= self.as_of)]

    def closes(self, underlying: str, n: int) -> pd.Series:
        if n < 0:
            raise ValueError("n must be >= 0")
        rows = self._eod_rows(underlying)
        rows = rows[(rows["session"] < pd.Timestamp(self.session)) & rows["close_c"].notna() & rows["close_knowable_at"].notna()]
        rows = rows[rows["close_knowable_at"] <= self.as_of].sort_values("session", kind="mergesort").tail(n)
        out = pd.Series(
            rows["close_c"].astype("int64").to_numpy(),
            index=pd.DatetimeIndex(rows["session"], name="session"),
            name="close_c",
            dtype="int64",
        )
        if not rows.empty:
            self._touch(f"closes:{underlying}", out, rows["close_knowable_at"].max())
        return out

    def close(self, underlying: str, session: date) -> Cents:
        df = self._daily_table(underlying)
        rows = df[(df["slot"] == Slot.EOD.value) & (df["session"] == pd.Timestamp(session))]
        if rows.empty:
            raise DataUnavailable(f"no eod daily row for {underlying} on {session}")
        row = rows.iloc[0]
        if pd.isna(row["close_c"]) or pd.isna(row["close_knowable_at"]):
            raise DataUnavailable(f"no close recorded for {underlying} on {session}")
        if row["close_knowable_at"] > self.as_of:
            raise PitViolation(
                f"close of {underlying} on {session} is knowable at {row['close_knowable_at']} > as_of {self.as_of.isoformat()}"
            )
        value = int(row["close_c"])
        self._touch(f"close:{underlying}:{session.isoformat()}", value, row["close_knowable_at"])
        return value

    def daily(self, underlying: str, n: int) -> pd.DataFrame:
        if n < 1:
            raise ValueError("n must be >= 1")
        df = self._daily_table(underlying)
        own = df[(df["session"] == pd.Timestamp(self.session)) & (df["slot"] == self.key.slot.value)]
        if own.empty:
            raise DataUnavailable(f"no daily row for {underlying} at {self.session} {self.key.slot.value}")
        if own.iloc[0]["knowable_at"] > self.as_of:
            raise PitViolation(
                f"daily row {underlying} {self.session} {self.key.slot.value} is knowable at {own.iloc[0]['knowable_at']} > as_of"
            )
        past = df[(df["session"] < pd.Timestamp(self.session)) & (df["knowable_at"] <= self.as_of)]
        labels: list[Any] = []
        for _, group in past.groupby("session", sort=True):
            designated = group.index[group["slot"] == self.key.slot.value]
            fallback = group.index[group["slot"] == Slot.EOD.value]
            if len(designated):
                labels.append(designated[0])
            elif len(fallback):
                labels.append(fallback[0])
        chosen = labels[-(n - 1) :] if n > 1 else []
        out = df.loc[[*chosen, own.index[0]]].reset_index(drop=True)
        gated = out["close_knowable_at"].isna() | (out["close_knowable_at"] > self.as_of)
        out.loc[gated, "close_c"] = pd.NA
        self._touch(f"daily:{underlying}", out, out["knowable_at"].max())
        return out

    # --- bars --------------------------------------------------------------------------------------------------------

    def _bars_table(self, underlying: str) -> pd.DataFrame:
        df = self._t.bars.get(underlying)
        if df is None:
            raise DataUnavailable(f"no bars for {underlying}")
        return df

    def bars(self, underlying: str, n: int) -> pd.DataFrame:
        if n < 0:
            raise ValueError("n must be >= 0")
        df = self._bars_table(underlying)
        done = (
            df[(df["hlcv_knowable_at"] <= self.as_of) & (df["session"] < pd.Timestamp(self.session))]
            .sort_values("session", kind="mergesort")
            .tail(n)
        )
        out = (
            done[["session", "open", "high", "low", "close"]]
            .astype({"open": "float64", "high": "float64", "low": "float64", "close": "float64"})
            .reset_index(drop=True)
        )
        if not done.empty:
            self._touch(f"bars:{underlying}", out, done["hlcv_knowable_at"].max())
        return out

    def today_open_ratio(self, underlying: str) -> float | None:
        df = self._bars_table(underlying)
        today = df[(df["session"] == pd.Timestamp(self.session)) & (df["open_knowable_at"] <= self.as_of)]
        prev = df[(df["session"] < pd.Timestamp(self.session)) & (df["hlcv_knowable_at"] <= self.as_of)].sort_values(
            "session", kind="mergesort"
        )
        if (
            today.empty
            or prev.empty
            or pd.isna(today.iloc[0]["open"])
            or pd.isna(prev.iloc[-1]["close"])
            or float(prev.iloc[-1]["close"]) <= 0.0
        ):
            return None
        ratio = float(today.iloc[0]["open"]) / float(prev.iloc[-1]["close"])
        self._touch(f"bars_open:{underlying}", repr(ratio), today.iloc[0]["open_knowable_at"])
        return ratio

    # --- side tables -------------------------------------------------------------------------------------------------

    def vol_index(self, name: str, n: int) -> pd.Series:
        if n < 0:
            raise ValueError("n must be >= 0")
        df = self._t.vol_indices.get(name)
        if df is None:
            raise DataUnavailable(f"no vol index {name}")
        rows = df[df["knowable_at"] <= self.as_of].sort_values("session", kind="mergesort").tail(n)
        out = pd.Series(
            rows["close"].to_numpy(dtype=np.float64), index=pd.DatetimeIndex(rows["session"], name="session"), name=name, dtype="float64"
        )
        if not rows.empty:
            self._touch(f"volidx:{name}", out, rows["knowable_at"].max())
        return out

    def rate(self) -> float:
        r = self._t.rates
        if r is None:
            raise DataUnavailable("no rates table")
        if isinstance(r, float | int):
            return float(r)
        rows = r[r["knowable_at"] <= self.as_of].sort_values("session", kind="mergesort")
        if rows.empty:
            raise DataUnavailable(f"no bill rate knowable at {self.as_of.isoformat()}")
        value = float(rows.iloc[-1]["rate_bp"]) / 1e4
        self._touch("rates", repr(value), rows.iloc[-1]["knowable_at"])
        return value

    def events(self, start: date, end: date, underlying: str | None = None) -> tuple[ScheduledEvent, ...]:
        """Scheduled, non-cancelled rows with `knowable_at <= as_of` and `start <= event_date <= end`. `underlying=None`
        serves every row; `underlying=U` serves the market-wide rows (underlying None) plus U's own (ex-dividend) rows."""
        out = [
            e
            for e in self._t.events
            if e.scheduled
            and not e.cancelled
            and e.knowable_at <= self.as_of
            and start <= e.event_date <= end
            and (underlying is None or e.underlying in (None, underlying))
        ]
        out.sort(key=lambda e: (e.event_date, e.kind, e.underlying or ""))
        if out:
            self._touch(f"events:{start.isoformat()}:{end.isoformat()}:{underlying or '*'}", tuple(out), max(e.knowable_at for e in out))
        return tuple(out)

    def event_coverage(self) -> tuple[str, ...]:
        """Event kinds with at least one verified (scheduled, non-cancelled) row, in the canonical kind order."""
        if self._t.event_coverage is not None:
            return self._t.event_coverage
        kinds = {e.kind for e in self._t.events if e.scheduled and not e.cancelled}
        return (*[k for k in EVENT_KINDS if k in kinds], *sorted(kinds - set(EVENT_KINDS)))

    def news(self, underlying: str, lookback_hours: int) -> tuple[NewsItem, ...]:
        """Items naming `underlying` with `knowable_at <= as_of` and `created_at >= as_of - lookback_hours`, newest first
        (by knowable_at); `summary = None` when `updated_at > as_of` (B6.3 rule 4)."""
        if lookback_hours < 0:
            raise ValueError("lookback_hours must be >= 0")
        since = self.as_of - timedelta(hours=lookback_hours)
        items = [i for i in self._t.news if underlying in i.symbols and i.knowable_at <= self.as_of and i.created_at >= since]
        items.sort(key=lambda i: (i.knowable_at, i.id), reverse=True)
        out = tuple(msgspec.structs.replace(i, summary=None) if i.updated_at > self.as_of else i for i in items)
        if out:
            self._touch(f"news:{underlying}", out, max(i.knowable_at for i in out))
        return out

    def news_covered(self, underlying: str) -> bool:
        covered = self._t.news_covered
        if covered is None:
            return any(underlying in i.symbols for i in self._t.news)
        if isinstance(covered, bool):
            return covered
        return bool(covered.get(underlying, False))


# ======================================================================================================================
# A complete world in one call
# ======================================================================================================================


def make_view(
    underlyings: Sequence[str] = ("SPY",),
    *,
    session: date = DEFAULT_SESSION,
    slot: Slot = Slot.EOD,
    fidelity: Fidelity = Fidelity.EOD_QUOTES,
    n_sessions: int = 300,
    seed: int = 0,
    spots: Mapping[str, Cents] | None = None,
    chains: Mapping[str, ChainSnapshot] | None = None,
    slots: Sequence[Slot] | None = None,
    events: Sequence[ScheduledEvent] | None = None,
    news: Sequence[NewsItem] = (),
    news_covered: bool | Mapping[str, bool] | None = None,
    as_of: datetime | None = None,
    calendar: "Calendar | None" = None,
    **chain_kwargs: Any,
) -> FakeView:
    """A coherent `FakeView` for `underlyings` at `(session, slot)`: factory chains (default spots $450 / $380 / $200 for
    SPY / QQQ / IWM, else $450; `chains` overrides per underlying; `chain_kwargs` go to `make_chain`), generated history
    (`make_history`, `n_sessions` sessions, `slots` = the recorder slots to generate, default the view's slot only), the
    Cboe indices, the bill rate, `events` (default: one scheduled FOMC decision 12 sessions ahead), `news`. `as_of`
    defaults to the chain's `ts`."""
    cal = _as_calendar(calendar)
    key = SnapshotKey(session=session, slot=slot)
    built: dict[str, ChainSnapshot] = {}
    for u in underlyings:
        if chains is not None and u in chains:
            built[u] = chains[u]
        else:
            spot = (spots or {}).get(u, _START_PRICE_CENTS.get(u, 45_000))
            built[u] = make_chain(u, session=session, slot=slot, spot=spot, fidelity=fidelity, calendar=calendar, **chain_kwargs)
    history_slots: tuple[Slot, ...] = tuple(slots) if slots is not None else ((Slot.EOD,) if slot is Slot.EOD else (slot, Slot.EOD))
    histories = {u: make_history(built[u], n_sessions=n_sessions, seed=seed, slots=history_slots, calendar=cal) for u in underlyings}
    first = built[underlyings[0]]
    vol = make_vol_indices(histories[underlyings[0]].iv30_bp, seed=seed, calendar=cal)
    rates = make_rates([pd.Timestamp(s).date() for s in histories[underlyings[0]].iv30_bp.index], first.rate, calendar=cal)
    default_events = (fomc_event(cal.next_session(session, 12)),) if events is None else tuple(events)
    return FakeView(
        key=key,
        as_of=first.ts if as_of is None else as_of,
        calendar=cal,
        fidelity=fidelity,
        chains=built.values(),
        daily={u: h.daily for u, h in histories.items()},
        bars={u: h.bars for u, h in histories.items()},
        vol_indices=vol,
        rates=rates,
        events=default_events,
        news=news,
        news_covered=news_covered,
    )


if TYPE_CHECKING:
    # Static proof, checked by mypy: FakeView implements the MarketView Protocol of 3.2.
    from jevbot.protocols import MarketView

    _FAKE_VIEW_IS_A_MARKET_VIEW: MarketView = FakeView(
        key=SnapshotKey(session=DEFAULT_SESSION, slot=Slot.EOD), as_of=datetime(2024, 5, 17, tzinfo=UTC)
    )


# ======================================================================================================================
# Self-tests (collected by tests/conftest.py::pytest_collect_file)
# ======================================================================================================================


def _hand_world() -> tuple[FakeView, ChainSnapshot]:
    """A tiny hand-written world: sessions 2024-05-14 .. 05-17 for SPY with explicit knowable_at stamps."""
    cal = xnys()
    chain = make_chain("SPY")
    sessions = [date(2024, 5, 14), date(2024, 5, 15), date(2024, 5, 16), date(2024, 5, 17)]
    closes_at = [cal.open_close(s)[1] for s in sessions]
    opens_at = [cal.open_close(s)[0] for s in sessions]
    daily = pd.DataFrame(
        {
            "session": pd.to_datetime(sessions),
            "slot": "eod",
            "px_c": [44_000, 44_500, 44_800, 45_000],
            "close_c": pd.array([44_000, 44_500, 44_800, 45_000], dtype="Int64"),
            "close_knowable_at": closes_at,
            "iv30_bp": pd.array([1500, 1550, 1580, 1600], dtype="Int32"),
            "iv30_2s_bp": pd.array([1500, 1550, 1580, 1600], dtype="Int32"),
            "iv90_bp": pd.array([1600, 1650, 1680, 1700], dtype="Int32"),
            "skew25_bp": pd.array([2000, 2100, 2050, 2000], dtype="Int32"),
            "atm_term_json": [None, None, None, "[[0.1, 25.0, 1600, 1600, 45173]]"],
            "rv20_bp": pd.array([1200, 1210, 1190, 1205], dtype="Int32"),
            "spot_measure": "parity",
            "div_unmodelled": False,
            "basis_suspect": False,
            "source": "mirror",
            "knowable_at": closes_at,
        }
    )
    bars = pd.DataFrame(
        {
            "session": pd.to_datetime(sessions),
            "open": [439.0, 441.0, 446.0, 448.5],
            "high": [441.0, 446.0, 449.0, 451.0],
            "low": [438.0, 440.0, 445.0, 447.0],
            "close": [440.0, 445.0, 448.0, 450.0],
            "volume": [1, 2, 3, 4],
            "knowable_at": [o + timedelta(seconds=60) for o in opens_at],
            "open_knowable_at": [o + timedelta(seconds=60) for o in opens_at],
            "hlcv_knowable_at": [cal.next_open_after(c) for c in closes_at],
        }
    )
    vix = pd.DataFrame(
        {"session": pd.to_datetime(sessions), "close": [15.0, 15.5, 15.8, 16.0], "knowable_at": [cal.next_open_after(c) for c in closes_at]}
    )
    rates = pd.DataFrame(
        {"session": pd.to_datetime(sessions), "rate_bp": [500, 510, 520, 530], "knowable_at": [cal.next_open_after(c) for c in closes_at]}
    )
    view = FakeView(
        key=chain.key,
        as_of=chain.ts,
        chains=[chain],
        daily={"SPY": daily},
        bars={"SPY": bars},
        vol_indices={"VIX": vix},
        rates=rates,
        events=(
            fomc_event(date(2024, 6, 12)),
            fomc_event(date(2024, 7, 31)),  # knowable 06-16: invisible on 05-17
            fomc_event(date(2024, 5, 29), scheduled=False),  # unscheduled: never served
            fomc_event(date(2024, 6, 5), cancelled=True),  # cancelled: never served
            ex_dividend_event("SPY", date(2024, 6, 21), 165, knowable_at=datetime(2024, 5, 1, tzinfo=UTC)),  # fetched forward
            ex_dividend_event("QQQ", date(2024, 6, 24), 50, knowable_at=datetime(2024, 5, 1, tzinfo=UTC)),
            ex_dividend_event("SPY", date(2024, 6, 28), 10),  # historical back-fill: knowable 06-14, invisible on 05-17
        ),
        news=(
            news_item("n1", "old one", created_at=datetime(2024, 5, 14, 12, tzinfo=UTC), symbols=("SPY",)),
            news_item("n2", "fresh", created_at=datetime(2024, 5, 17, 18, tzinfo=UTC), symbols=("SPY", "QQQ"), summary="s2"),
            news_item(
                "n3",
                "revised later",
                created_at=datetime(2024, 5, 17, 15, tzinfo=UTC),
                symbols=("SPY",),
                summary="s3",
                updated_at=datetime(2024, 5, 18, tzinfo=UTC),
            ),
            news_item("n4", "not yet knowable", created_at=datetime(2024, 5, 17, 19, 59, 30, tzinfo=UTC), symbols=("SPY",)),
            news_item("n5", "other symbol", created_at=datetime(2024, 5, 17, 17, tzinfo=UTC), symbols=("QQQ",)),
        ),
    )
    return view, chain


def test_fake_view_implements_the_market_view_protocol() -> None:
    import inspect

    from jevbot.protocols import MarketView

    view, _ = _hand_world()
    for name, member in vars(MarketView).items():
        if name.startswith("_") or not callable(member):
            continue
        impl = getattr(FakeView, name)
        want = [(p.name, p.kind, p.default) for p in inspect.signature(member).parameters.values()][1:]
        got = [(p.name, p.kind, p.default) for p in inspect.signature(impl).parameters.values()][1:]
        assert got == want, name
    for attr in ("as_of", "key", "session", "calendar", "fidelity"):
        assert isinstance(vars(MarketView)[attr], property) and hasattr(view, attr)  # read-only data members (section 3)
    assert view.session == view.key.session == DEFAULT_SESSION and view.fidelity is Fidelity.EOD_QUOTES
    assert view.calendar.is_session(DEFAULT_SESSION)


def test_chain_reads_are_pit_gated_and_spot_is_the_chain_spot() -> None:
    import pytest

    view, chain = _hand_world()
    assert view.chain("SPY") is chain and view.spot("SPY") == 45_000
    assert [p.field for p in view.touched()] == ["chain:SPY"]  # two identical reads, logged once
    with pytest.raises(DataUnavailable):
        view.chain("QQQ")
    early = view.at(as_of=chain.ts - timedelta(minutes=1))
    with pytest.raises(PitViolation):
        early.chain("SPY")
    late = make_chain("SPY", knowable_at=chain.ts + timedelta(seconds=30))
    view.add_chain(late)
    with pytest.raises(PitViolation):
        view.chain("SPY")
    assert len(view.touched()) == 1 and early.touched() == ()  # the refused reads logged nothing
    later = view.at(as_of=chain.ts + timedelta(seconds=30))
    assert later.spot("SPY") == 45_000 and later.chain("SPY") is late
    logged = later.touched()
    assert [p.field for p in logged] == ["chain:SPY"] and logged[0].knowable_at == late.knowable_at and logged[0].event_time == late.ts
    assert logged[0].source == "fake.chain" and logged[0].payload_sha256 == hashlib.sha256(late.content_hash.encode()).hexdigest()


def test_closes_close_and_daily_follow_the_pit_rules() -> None:
    import pytest

    view, chain = _hand_world()
    c = view.closes("SPY", 260)
    assert list(c) == [44_000, 44_500, 44_800] and str(c.dtype) == "int64" and c.name == "close_c"  # completed sessions only
    assert [d.date() for d in c.index] == [date(2024, 5, 14), date(2024, 5, 15), date(2024, 5, 16)] and c.index.name == "session"
    assert list(view.closes("SPY", 2)) == [44_500, 44_800] and view.closes("SPY", 0).empty
    assert view.close("SPY", date(2024, 5, 16)) == 44_800
    assert view.close("SPY", date(2024, 5, 17)) == 45_000  # its close_knowable_at equals as_of (the eod snapshot IS the close)
    with pytest.raises(PitViolation):
        view.at(as_of=chain.ts - timedelta(seconds=1)).close("SPY", date(2024, 5, 17))
    with pytest.raises(DataUnavailable):
        view.close("SPY", date(2024, 5, 13))
    with pytest.raises(DataUnavailable):
        view.close("QQQ", date(2024, 5, 16))
    # an eod row whose close is not recorded yet (paper: record_close fills it next morning)
    pending = view.tables.daily["SPY"].copy()
    pending.loc[pending["session"] == pd.Timestamp(date(2024, 5, 17)), ["close_c", "close_knowable_at"]] = [pd.NA, pd.NaT]
    view.tables.daily["SPY"] = pending
    with pytest.raises(DataUnavailable):
        view.close("SPY", date(2024, 5, 17))
    assert list(view.closes("SPY", 10)) == [44_000, 44_500, 44_800]
    d = view.daily("SPY", 3)
    assert list(d.columns) == list(DAILY_COLUMNS) and len(d) == 3
    assert [s.date() for s in d["session"]] == [date(2024, 5, 15), date(2024, 5, 16), date(2024, 5, 17)]  # own row last
    assert pd.isna(d["close_c"].iloc[-1]) and list(d["close_c"].iloc[:2]) == [44_500, 44_800]
    assert d["atm_term_json"].iloc[-1] == "[[0.1, 25.0, 1600, 1600, 45173]]" and d["iv30_bp"].iloc[-1] == 1600
    with pytest.raises(PitViolation):
        view.at(as_of=chain.ts - timedelta(seconds=1)).daily("SPY", 3)
    with pytest.raises(DataUnavailable):
        view.at(key=SnapshotKey(session=date(2024, 5, 20), slot=Slot.EOD), as_of=chain.ts + timedelta(days=3)).daily("SPY", 3)
    with pytest.raises(DataUnavailable):
        view.daily("QQQ", 3)


def test_daily_designated_slot_rule_three_slot_archive_equals_collapsed() -> None:
    view = make_view(("SPY",), slot=Slot.DEC, fidelity=Fidelity.RECORDED_INDICATIVE, n_sessions=30, slots=(Slot.DEC, Slot.EXEC, Slot.EOD))
    full = view.tables.daily["SPY"]
    assert set(full["slot"]) == {"dec", "exec", "eod"} and len(full) == 90
    three = view.daily("SPY", 30)
    assert len(three) == 30 and (three["slot"] == "dec").all()  # the designated slot everywhere, own row last
    assert three["close_c"].isna().all()  # dec rows never carry a close
    collapsed = full[full["slot"] == "dec"]
    flat = FakeView(key=view.key, as_of=view.as_of, fidelity=view.fidelity, chains=[view.chain("SPY")], daily={"SPY": collapsed})
    assert flat.daily("SPY", 30).drop(columns=["close_c"]).equals(three.drop(columns=["close_c"]))
    # a session without a dec row falls back to its eod row (mirror-era history seen from a dec view)
    mixed = full[~((full["slot"] == "dec") & (full["session"] == full["session"].iloc[0]))]
    fallback = FakeView(key=view.key, as_of=view.as_of, fidelity=view.fidelity, chains=[view.chain("SPY")], daily={"SPY": mixed}).daily(
        "SPY", 30
    )
    assert list(fallback["slot"]) == ["eod", *(["dec"] * 29)]
    # a session with neither is skipped; an eod-only archive read from an eod view is itself
    eod_view = make_view(("SPY",), n_sessions=12)
    d = eod_view.daily("SPY", 5)
    assert len(d) == 5 and (d["slot"] == "eod").all() and int(d["close_c"].iloc[-1]) == 45_000 and int(d["px_c"].iloc[-1]) == 45_000


def test_bars_today_open_ratio_vol_index_rate_events_and_news() -> None:
    import pytest

    view, chain = _hand_world()
    b = view.bars("SPY", 15)
    assert list(b.columns) == ["session", "open", "high", "low", "close"] and len(b) == 3  # today's H / L / C are not knowable
    assert list(b["close"]) == [440.0, 445.0, 448.0] and b["session"].iloc[-1].date() == date(2024, 5, 16)
    assert len(view.bars("SPY", 2)) == 2 and view.bars("SPY", 0).empty
    assert view.today_open_ratio("SPY") == 448.5 / 448.0  # open(D) / file close(D-1)
    before_open = view.at(as_of=xnys().open_close(date(2024, 5, 17))[0] + timedelta(seconds=59))
    assert before_open.today_open_ratio("SPY") is None and len(before_open.bars("SPY", 15)) == 3
    assert view.at(as_of=xnys().open_close(date(2024, 5, 17))[0] + timedelta(seconds=60)).today_open_ratio("SPY") == 448.5 / 448.0
    first = view.at(key=SnapshotKey(session=date(2024, 5, 14), slot=Slot.EOD), as_of=xnys().open_close(date(2024, 5, 14))[1])
    assert first.today_open_ratio("SPY") is None and first.bars("SPY", 5).empty
    with pytest.raises(DataUnavailable):
        view.bars("QQQ", 5)
    v = view.vol_index("VIX", 10)
    assert list(v) == [15.0, 15.5, 15.8] and v.name == "VIX" and v.index[-1].date() == date(2024, 5, 16)  # D-1 is the newest (D22)
    assert list(view.vol_index("VIX", 1)) == [15.8]
    next_day = view.at(key=SnapshotKey(session=date(2024, 5, 20), slot=Slot.EOD), as_of=xnys().open_close(date(2024, 5, 20))[1])
    assert list(next_day.vol_index("VIX", 10)) == [15.0, 15.5, 15.8, 16.0]
    with pytest.raises(DataUnavailable):
        view.vol_index("VXN", 10)
    assert view.rate() == 0.052 and next_day.rate() == 0.053 and view.at(fidelity=Fidelity.SYNTHETIC).rate() == 0.052
    assert FakeView(key=chain.key, as_of=chain.ts, rates=0.0123).rate() == 0.0123
    with pytest.raises(DataUnavailable):
        FakeView(key=chain.key, as_of=chain.ts).rate()
    # events: scheduled + knowable + window; unscheduled / cancelled / future-knowable never served; underlying scoping
    got = view.events(date(2024, 5, 17), date(2024, 8, 31))
    assert [(e.kind, e.event_date, e.underlying) for e in got] == [
        ("fomc_decision", date(2024, 6, 12), None),
        ("ex_dividend", date(2024, 6, 21), "SPY"),
        ("ex_dividend", date(2024, 6, 24), "QQQ"),
    ]
    assert [e.event_date for e in view.events(date(2024, 5, 17), date(2024, 8, 31), "SPY")] == [date(2024, 6, 12), date(2024, 6, 21)]
    assert view.events(date(2024, 6, 13), date(2024, 6, 20)) == ()
    assert [e.event_date for e in view.events(date(2024, 6, 12), date(2024, 6, 12))] == [date(2024, 6, 12)]  # inclusive bounds
    july = view.at(as_of=datetime(2024, 6, 16, 12, tzinfo=UTC))
    assert [e.event_date for e in july.events(date(2024, 7, 1), date(2024, 8, 31))] == [date(2024, 7, 31)]
    assert view.event_coverage() == ("fomc_decision", "ex_dividend")
    assert FakeView(key=chain.key, as_of=chain.ts, event_coverage=("cpi",)).event_coverage() == ("cpi",)
    assert FakeView(key=chain.key, as_of=chain.ts).event_coverage() == ()
    # news: newest first, knowable only, symbol-scoped, lookback window, summary dropped when revised after as_of
    items = view.news("SPY", 72)
    assert [i.id for i in items] == ["n2", "n3"] and items[0].summary == "s2" and items[1].summary is None
    assert [i.id for i in view.news("SPY", 24 * 5)] == ["n2", "n3", "n1"]
    assert [i.id for i in view.news("QQQ", 72)] == ["n2", "n5"] and view.news("IWM", 72) == ()
    assert [i.id for i in view.at(as_of=chain.ts + timedelta(seconds=60)).news("SPY", 72)] == ["n4", "n2", "n3"]
    assert view.news_covered("SPY") and view.news_covered("QQQ") and not view.news_covered("IWM")
    assert not FakeView(key=chain.key, as_of=chain.ts, news=view.tables.news, news_covered=False).news_covered("SPY")
    assert FakeView(key=chain.key, as_of=chain.ts, news_covered={"SPY": True}).news_covered("SPY")
    assert not FakeView(key=chain.key, as_of=chain.ts, news_covered={"SPY": True}).news_covered("QQQ")
    assert view.tables.news[2].summary == "s3"  # the stored item is never mutated
    with pytest.raises(ValueError):
        FakeView(key=chain.key, as_of=datetime(2024, 5, 17, 20))  # noqa: DTZ001 - the naive datetime IS the case under test
    fields = list(dict.fromkeys(p.field for p in view.touched()))
    assert fields[:4] == ["bars:SPY", "bars_open:SPY", "volidx:VIX", "rates"] and all(p.knowable_at <= view.as_of for p in view.touched())
    assert {p.source for p in view.touched()} == {"fake.bars", "fake.bars_open", "fake.volidx", "fake.rates", "fake.events", "fake.news"}
    assert len(view.touched()) == len(set(view.touched()))  # identical reads are logged once


def test_make_view_is_coherent_and_deterministic() -> None:
    view = make_view(("SPY", "QQQ"), n_sessions=300, seed=3)
    again = make_view(("SPY", "QQQ"), n_sessions=300, seed=3)
    for u, spot in (("SPY", 45_000), ("QQQ", 38_000)):
        assert view.spot(u) == spot and view.chain(u).content_hash == again.chain(u).content_hash
        closes = view.closes(u, 260)
        assert len(closes) == 260 and closes.index[-1].date() == xnys().prev_session(DEFAULT_SESSION) and (closes > 0).all()
        assert list(closes) == list(again.closes(u, 260))
        d = view.daily(u, 260)
        assert (
            len(d) == 260
            and d["session"].iloc[-1].date() == DEFAULT_SESSION
            and int(d["close_c"].iloc[-1]) == spot
            and int(d["px_c"].iloc[-1]) == spot
        )
        assert d["session"].is_monotonic_increasing and d["session"].is_unique and (d["slot"] == "eod").all()
        assert d["iv30_bp"].notna().all() and d["rv20_bp"].notna().all() and (d["iv30_bp"] > 0).all() and (d["rv20_bp"] > 0).all()
        assert d["atm_term_json"].iloc[:-1].isna().all() and isinstance(d["atm_term_json"].iloc[-1], str)
        term = json.loads(d["atm_term_json"].iloc[-1])
        assert len(term) == 10 and all(len(node) == 5 for node in term) and term[4][2] == term[4][3]
        assert [node[0] for node in term] == sorted(node[0] for node in term) and abs(term[4][0] - 35 / 365) < 1e-9
        assert abs(term[4][1] - xnys().sessions_between(DEFAULT_SESSION, date(2024, 6, 21))) < 1e-9  # whole sessions of trading time
        assert 1550 <= int(d["iv30_bp"].iloc[-1]) <= 1650 and int(d["iv90_bp"].iloc[-1]) > int(d["iv30_bp"].iloc[-1])  # 16% ATM, contango
        assert 1000 <= int(d["skew25_bp"].iloc[-1]) <= 3000  # index-like smirk
        # rv20 of the own row equals the hand formula over the last 20 generated returns
        c = np.concatenate((closes.to_numpy(dtype=np.float64), [spot]))
        r = np.diff(np.log(c))[-20:]
        assert int(d["rv20_bp"].iloc[-1]) == round(math.sqrt(252 * float(np.mean(r * r))) * 1e4)
        b = view.bars(u, 15)
        assert (
            len(b) == 15 and (b["high"] >= b[["open", "close"]].max(axis=1)).all() and (b["low"] <= b[["open", "close"]].min(axis=1)).all()
        )
        assert abs(float(b["close"].iloc[-1]) * 100 - closes.iloc[-1]) < 1e-6
        ratio = view.today_open_ratio(u)
        assert ratio is not None and 0.9 < ratio < 1.1
    for name in DataConfig().cboe_indices:
        series = view.vol_index(name, 252)
        assert len(series) == 252 and series.index[-1].date() == xnys().prev_session(DEFAULT_SESSION)
    assert abs(view.vol_index("VIX", 1).iloc[0] - 16.8) < 3.0 and view.rate() == 0.04
    assert [e.kind for e in view.events(DEFAULT_SESSION, xnys().next_session(DEFAULT_SESSION, 20))] == ["fomc_decision"]
    assert view.events(DEFAULT_SESSION, xnys().next_session(DEFAULT_SESSION, 20))[0].event_date == xnys().next_session(DEFAULT_SESSION, 12)
    assert view.event_coverage() == ("fomc_decision",) and not view.news_covered("SPY") and view.news("SPY", 72) == ()
    assert make_view(("SPY",), seed=4).closes("SPY", 5).tolist() != view.closes("SPY", 5).tolist()
    assert make_view(("SPY",), events=()).events(DEFAULT_SESSION, date(2024, 12, 31)) == ()
    # a poisoned future row (knowable after as_of, absurd values) changes no read: the view never looks ahead
    full = view.tables.daily["SPY"]
    future = full.iloc[[-1]].copy()
    row = future.index[0]
    future.loc[row, "session"] = pd.Timestamp(xnys().next_session(DEFAULT_SESSION))
    future.loc[row, "knowable_at"] = pd.Timestamp(view.as_of + timedelta(days=1))
    future.loc[row, ["px_c", "close_c", "iv30_bp"]] = [1, 1, 1]
    assert future.dtypes.equals(full.dtypes)
    poisoned = FakeView(
        key=view.key, as_of=view.as_of, chains=[view.chain("SPY")], daily={"SPY": pd.concat([full, future], ignore_index=True)}
    )
    assert poisoned.daily("SPY", 260).equals(view.daily("SPY", 260)) and poisoned.closes("SPY", 260).equals(view.closes("SPY", 260))
    same_day = FakeView(
        key=view.key, as_of=view.as_of, chains=[view.chain("SPY")], daily={"SPY": full[full["session"] <= pd.Timestamp(DEFAULT_SESSION)]}
    )
    assert same_day.daily("SPY", 260).equals(view.daily("SPY", 260))
