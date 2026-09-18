"""Deterministic small enriched-chain builder - WP00's shared test double of `ChainSnapshot` (DESIGN.md 2.2, 15.2).

Every later package builds and tests against this factory (section 16), so it is faithful to the chain contract and stable:

* the table has exactly `types.CHAIN_COLUMNS` with the 2.2 dtypes, sorted `(expiry, right, strike_milli)`, filtered to
  `1 <= dte <= max_dte` and `|ln(K / fwd)| <= moneyness_window`;
* `expiry` is the LISTED OCC date (identity only); `last_session = calendar.prev_or_same_session(expiry)`; `dte` and every
  time-to-expiry are measured to `close(last_session)` (Conventions, INV-11), never to `expiry`;
* one parity forward per expiry, `fwd = round(spot * exp(rate * T_E))` (no dividends), and every quote is Black-76 priced
  ON THAT FORWARD from a planted smile - `w(k) = atm_E^2 * T_E * (1 + skew * k + curvature * k^2)`, `k = ln(K / fwd)`, a
  quadratic in TOTAL VARIANCE, so a smile fit can recover it exactly (6.4) - and rounded to whole cents;
* fixed spreads of 2 / 4 / 6 cents by price level (mid below $1 / below $5 / from $5), symmetric around the rounded mid, so
  the mid of every two-sided quote is an integer and put-call parity recovers `fwd` within a cent (5.2);
* `iv`, `delta`, `vega` are OUR Black-76 values re-solved from the rounded mid on the forward, on `Quote.valid()` rows only
  and accepted iff `0.02 < iv < 5.0` - exactly what `data/surface.enrich` does - so far-OTM rows carry a zero bid and NaN
  greeks, like real data; `iv_vendor` holds the planted (unrounded) smile IV for QC comparisons;
* constant displayed sizes and open interest.

Variants (15.2) are pure functions that return a NEW snapshot (the input is never mutated): crossed / locked / zero-bid /
missing quotes, a widened spread, a stale quote, thin or missing open interest, a dropped contract ("missing leg"), the
$600-priced chain for the budget fit (section 8), Saturday-dated monthlies and a Good-Friday week (2014), and the
"zero-bid wing on a winning condor" scenario of 10.4 / 10.5. Structure helpers (`make_structure`, `open_legs`, `close_legs`)
pick legs by delta so that same-wave packages need neither the CandidateGenerator nor each other.

Nothing here reads a clock, the environment or a file; two calls with equal arguments return equal bytes. The self-tests
at the bottom are collected by `tests/conftest.py` (`pytest_collect_file`).
"""

import hashlib
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from jevbot import bs
from jevbot.cal import XnysCalendar, year_fraction
from jevbot.config import CadenceConfig, CandidatesConfig
from jevbot.errors import DataUnavailable
from jevbot.types import (
    CHAIN_COLUMNS,
    Cents,
    ChainSnapshot,
    Fidelity,
    Leg,
    OptionContract,
    OrderLeg,
    PositionIntent,
    Quote,
    Right,
    Side,
    Slot,
    SnapshotKey,
    Structure,
    StructureKind,
)

if TYPE_CHECKING:
    from jevbot.protocols import Calendar

__all__ = [
    "DEFAULT_RATE",
    "DEFAULT_SESSION",
    "DEFAULT_SPOT",
    "GOOD_FRIDAY_SESSION",
    "SATURDAY_MONTHLY_SESSION",
    "CondorScenario",
    "Smile",
    "close_legs",
    "contract_at",
    "crossed_quote",
    "drop_contract",
    "good_friday_chain",
    "hundred_dollar_chain",
    "listed_expiries",
    "locked_quote",
    "make_chain",
    "make_structure",
    "missing_oi",
    "nearest_delta",
    "no_quote",
    "open_legs",
    "quote_of",
    "saturday_monthly_chain",
    "set_quote",
    "six_hundred_dollar_chain",
    "snapshot_ts",
    "stale_quote",
    "strikes",
    "target_expiry",
    "thin_oi",
    "widened_spread",
    "winning_condor",
    "xnys",
    "zero_bid",
]

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.int64]]
BoolArray = np.ndarray[Any, np.dtype[np.bool_]]

# ======================================================================================================================
# Defaults
# ======================================================================================================================

DEFAULT_SESSION: Final = date(2024, 5, 17)  # a Friday: the weekly Fridays land on dte 7, 14, ..., the June monthly on dte 35
DEFAULT_SPOT: Final[Cents] = 45_000  # $450.00
DEFAULT_RATE: Final = 0.04  # 13-week bill, decimal
SATURDAY_MONTHLY_SESSION: Final = date(2014, 5, 16)  # the Saturday-dated June 2014 monthly (06-21) sits at dte 35
GOOD_FRIDAY_SESSION: Final = date(2014, 3, 13)  # the Saturday-dated April 2014 monthly (04-19) sits at dte 35; Good Friday 04-18

_IV_LO: Final = 0.02  # enrichment accepts a solved IV iff _IV_LO < iv < _IV_HI (5.3)
_IV_HI: Final = 5.0
_DEFAULT_STRIKE_STEP: Final[Cents] = 100  # $1 strikes (SPY / QQQ / IWM)
_DEFAULT_MONEYNESS_WINDOW: Final = 0.35  # the enriched-chain window of 2.2
_DEFAULT_MAX_DTE: Final = 120  # data.max_dte
_WEEKLY_HORIZON_DAYS: Final = 56  # weeklies are listed eight weeks out; monthlies out to max_dte
_QUOTE_LAG_S: Final = 5  # recorded / live quote_ts = ts - 5 s
_MILLI_PER_CENT: Final = 10

_SOURCE_BY_FIDELITY: Final[Mapping[Fidelity, str]] = {
    Fidelity.EOD_QUOTES: "mirror",
    Fidelity.SYNTHETIC: "synthetic",
    Fidelity.RECORDED_INDICATIVE: "alpaca_recorded",
    Fidelity.LIVE_INDICATIVE: "alpaca_live",
}
_SPOT_MEASURE_BY_FIDELITY: Final[Mapping[Fidelity, str]] = {
    Fidelity.EOD_QUOTES: "parity",
    Fidelity.SYNTHETIC: "synthetic",
    Fidelity.RECORDED_INDICATIVE: "live_mid",
    Fidelity.LIVE_INDICATIVE: "live_mid",
}
_EOD_ONLY: Final = frozenset({Fidelity.EOD_QUOTES, Fidelity.SYNTHETIC})

_DTYPES: Final[Mapping[str, str]] = {
    "occ": "object",
    "expiry": "datetime64[ns]",
    "last_session": "datetime64[ns]",
    "right": "object",
    "strike_milli": "int64",
    "dte": "int64",
    "bid": "int64",
    "ask": "int64",
    "bid_size": "Int64",
    "ask_size": "Int64",
    "oi_prev": "Int64",
    "iv": "float64",
    "delta": "float64",
    "vega": "float64",
    "fwd": "int64",
    "iv_vendor": "float64",
    "quote_ts": "datetime64[ns, UTC]",
}

_calendar: XnysCalendar | None = None


def xnys() -> XnysCalendar:
    """The shared default calendar (its session tables are cached inside `cal.py`)."""
    global _calendar
    if _calendar is None:
        _calendar = XnysCalendar()
    return _calendar


def _as_calendar(calendar: "Calendar | None") -> "Calendar":
    return xnys() if calendar is None else calendar


# ======================================================================================================================
# The planted smile
# ======================================================================================================================


@dataclass(frozen=True)
class Smile:
    """Total variance `w(k) = atm(dte)^2 * tau * (1 + skew * k + curvature * k^2)` with `k = ln(K / fwd)`.

    `atm(dte) = atm_iv * clip(1 + term_slope * ln(dte / 30), 0.5, 1.5)` (the 5.10 term rule: contango by default). The
    defaults give an index-like smirk: about +2 vol points at the 25-delta put and -1.5 at the 25-delta call one month out.
    `skew^2 < 4 * curvature` keeps `w` positive for every `k`.
    """

    atm_iv: float = 0.16
    term_slope: float = 0.05
    skew: float = -6.0
    curvature: float = 20.0

    def __post_init__(self) -> None:
        if not (0.0 < self.atm_iv < 5.0):
            raise ValueError(f"atm_iv must be in (0, 5), got {self.atm_iv}")
        flat = self.skew == 0.0 and self.curvature == 0.0
        if not flat and (self.curvature <= 0.0 or self.skew * self.skew >= 4.0 * self.curvature):
            raise ValueError("the smile needs skew^2 < 4 * curvature (or skew = curvature = 0) so that total variance stays positive")

    def atm(self, dte: int) -> float:
        return self.atm_iv * min(max(1.0 + self.term_slope * math.log(dte / 30.0), 0.5), 1.5)

    def coefficients(self, dte: int, tau: float) -> tuple[float, float, float]:
        """(a, b, c) of `w(k) = a + b k + c k^2` for one expiry - what `surface.fit_smile` should recover."""
        a = self.atm(dte) ** 2 * tau
        return (a, a * self.skew, a * self.curvature)

    def iv(self, k: FloatArray, dte: int, tau: float) -> FloatArray:
        a, b, c = self.coefficients(dte, tau)
        w = a + b * k + c * k * k
        out: FloatArray = np.sqrt(np.maximum(w, 0.0) / tau)
        return out


# ======================================================================================================================
# Expiries and snapshot time
# ======================================================================================================================


def _is_third_friday(d: date) -> bool:
    return d.weekday() == 4 and 15 <= d.day <= 21


def listed_expiries(
    session: date,
    *,
    saturday_monthlies: bool = False,
    max_dte: int = _DEFAULT_MAX_DTE,
    weekly_horizon_days: int = _WEEKLY_HORIZON_DAYS,
    calendar: "Calendar | None" = None,
) -> tuple[date, ...]:
    """The LISTED expiry dates a chain carries on `session`: every Friday within `weekly_horizon_days` plus every third
    Friday within `max_dte`, listed as the previous session when the Friday is a holiday (a Good-Friday week lists the
    Thursday, 5.10) - or, with `saturday_monthlies`, third Fridays listed on the SATURDAY (monthlies before February 2015).
    Only expiries whose `last_session` gives `1 <= dte <= max_dte` are kept."""
    cal = _as_calendar(calendar)
    out: set[date] = set()
    friday = session + timedelta(days=(4 - session.weekday()) % 7 or 7)
    while (friday - session).days <= max_dte + 1:
        monthly = _is_third_friday(friday)
        if monthly or (friday - session).days <= weekly_horizon_days:
            listed = friday + timedelta(days=1) if (monthly and saturday_monthlies) else cal.prev_or_same_session(friday)
            dte = (cal.prev_or_same_session(listed) - session).days
            if 1 <= dte <= max_dte:
                out.add(listed)
        friday += timedelta(days=7)
    return tuple(sorted(out))


def snapshot_ts(session: date, slot: Slot, fidelity: Fidelity, *, calendar: "Calendar | None" = None) -> datetime:
    """The snapshot time of a (session, slot): the calendar close for EOD data; the [cadence] offsets from the close for
    recorded / live slots (`dec` = close - 25 min, `exec` = close - 20 min, `eod` = close + 2 min). INV-13: no clock literals."""
    cal = _as_calendar(calendar)
    if fidelity in _EOD_ONLY:
        if slot is not Slot.EOD:
            raise ValueError(f"{fidelity.value} data has only the eod slot (Conventions), got {slot.value}")
        return cal.open_close(session)[1]
    cad = CadenceConfig()
    minutes = {Slot.DEC: cad.decide_offset_min, Slot.EXEC: cad.exec_offset_min, Slot.EOD: cad.eod_offset_min}[slot]
    return cal.offset_from_close(session, minutes)


# ======================================================================================================================
# The chain
# ======================================================================================================================


def _occ(underlying: str, expiry: date, right: str, strike_milli: int) -> str:
    return f"{underlying}{expiry:%y%m%d}{right}{strike_milli:08d}"


def _content_hash(table: pd.DataFrame) -> str:
    return hashlib.sha256(table.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def _spread_cents(mid: IntArray) -> IntArray:
    out: IntArray = np.where(mid < 100, 2, np.where(mid < 500, 4, 6)).astype(np.int64)
    return out


def _greeks(
    bid: IntArray, ask: IntArray, fwd: float, strike_c: FloatArray, tau: float, df: float, is_call: BoolArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """OUR iv / delta / vega from the quote mid on the forward: valid() rows only, accepted iff 0.02 < iv < 5.0 (5.3)."""
    valid = (bid > 0) & (ask > bid)
    mid = (bid + ask) / 2.0
    iv = bs.b76_implied_vol(mid, fwd, strike_c, tau, df, is_call)
    iv = np.where(valid & (iv > _IV_LO) & (iv < _IV_HI), iv, np.nan)
    delta = bs.b76_delta(fwd, strike_c, tau, iv, df, is_call)
    vega = bs.b76_vega(fwd, strike_c, tau, iv, df)
    return iv, delta, vega


def _empty_table() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in _DTYPES.items()})[CHAIN_COLUMNS]


def make_chain(
    underlying: str = "SPY",
    *,
    session: date = DEFAULT_SESSION,
    slot: Slot = Slot.EOD,
    spot: Cents = DEFAULT_SPOT,
    rate: float = DEFAULT_RATE,
    smile: Smile | None = None,
    expiries: Sequence[date] | None = None,
    saturday_monthlies: bool = False,
    strike_step: Cents = _DEFAULT_STRIKE_STEP,
    moneyness_window: float = _DEFAULT_MONEYNESS_WINDOW,
    max_dte: int = _DEFAULT_MAX_DTE,
    size: int = 100,
    oi_prev: int = 2000,
    fidelity: Fidelity = Fidelity.EOD_QUOTES,
    source: str | None = None,
    spot_measure: str | None = None,
    div_unmodelled: bool = False,
    ts: datetime | None = None,
    knowable_at: datetime | None = None,
    calendar: "Calendar | None" = None,
) -> ChainSnapshot:
    """Build one enriched `ChainSnapshot` for `(underlying, session, slot)`.

    `expiries` are LISTED dates (default: `listed_expiries(session, saturday_monthlies=...)`); `strike_step` is in cents;
    `ts` defaults to `snapshot_ts(...)` and `knowable_at` to `ts` (EOD data) - pass `knowable_at` explicitly to model a
    recorder's later `received_at`. `source` / `spot_measure` default per fidelity ("mirror" / "parity" for EOD_QUOTES).
    """
    cal = _as_calendar(calendar)
    if spot <= 0 or strike_step <= 0:
        raise ValueError("spot and strike_step must be positive cents")
    if moneyness_window <= 0.0 or max_dte < 1:
        raise ValueError("moneyness_window must be positive and max_dte >= 1")
    sm = smile if smile is not None else Smile()
    when = snapshot_ts(session, slot, fidelity, calendar=cal) if ts is None else ts
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError("ts must be tz-aware UTC")
    listed = (
        listed_expiries(session, saturday_monthlies=saturday_monthlies, max_dte=max_dte, calendar=cal)
        if expiries is None
        else tuple(expiries)
    )
    args = (
        underlying,
        session,
        when,
        spot,
        rate,
        sm,
        tuple(sorted(set(listed))),
        strike_step,
        moneyness_window,
        max_dte,
        size,
        oi_prev,
        fidelity in _EOD_ONLY,
    )
    if calendar is None:
        table, digest = _cached_table(*args)  # the shared default calendar: memoised; every caller gets its own copy
        table = table.copy()
    else:
        table, digest = _build_table(cal, *args)
    return ChainSnapshot(
        underlying=underlying,
        key=SnapshotKey(session=session, slot=slot),
        ts=when,
        knowable_at=when if knowable_at is None else knowable_at,
        spot=spot,
        spot_measure=_SPOT_MEASURE_BY_FIDELITY[fidelity] if spot_measure is None else spot_measure,
        div_unmodelled=div_unmodelled,
        rate=rate,
        table=table,
        fidelity=fidelity,
        source=_SOURCE_BY_FIDELITY[fidelity] if source is None else source,
        content_hash=digest,
    )


@lru_cache(maxsize=64)
def _cached_table(
    underlying: str,
    session: date,
    when: datetime,
    spot: Cents,
    rate: float,
    sm: Smile,
    listed: tuple[date, ...],
    strike_step: Cents,
    moneyness_window: float,
    max_dte: int,
    size: int,
    oi_prev: int,
    eod_data: bool,
) -> tuple[pd.DataFrame, str]:
    return _build_table(
        xnys(), underlying, session, when, spot, rate, sm, listed, strike_step, moneyness_window, max_dte, size, oi_prev, eod_data
    )


def _build_table(
    cal: "Calendar",
    underlying: str,
    session: date,
    when: datetime,
    spot: Cents,
    rate: float,
    sm: Smile,
    listed: tuple[date, ...],
    strike_step: Cents,
    moneyness_window: float,
    max_dte: int,
    size: int,
    oi_prev: int,
    eod_data: bool,
) -> tuple[pd.DataFrame, str]:
    quote_ts = pd.NaT if eod_data else pd.Timestamp(when - timedelta(seconds=_QUOTE_LAG_S))
    blocks: list[pd.DataFrame] = []
    for expiry in listed:
        last = cal.prev_or_same_session(expiry)
        dte = (last - session).days
        if not (1 <= dte <= max_dte):
            continue
        tau = year_fraction(when, cal.open_close(last)[1])  # T_E to close(last_session), never close(expiry)
        if tau <= 0.0:
            continue
        df = math.exp(-rate * tau)
        fwd = math.floor(spot * math.exp(rate * tau) + 0.5)
        lo = math.ceil(fwd * math.exp(-moneyness_window) / strike_step)
        hi = math.floor(fwd * math.exp(moneyness_window) / strike_step)
        if hi < lo:
            continue
        strike_c: IntArray = np.arange(lo, hi + 1, dtype=np.int64) * strike_step
        k = np.log(strike_c / fwd)
        planted = sm.iv(k, dte, tau)
        n = strike_c.size
        for right, is_call in ((Right.CALL, True), (Right.PUT, False)):
            call_mask: BoolArray = np.full(n, is_call, dtype=np.bool_)
            theo = bs.b76_price(fwd, strike_c, tau, planted, df, call_mask)
            mid: IntArray = np.floor(theo + 0.5).astype(np.int64)
            half = _spread_cents(mid) // 2
            bid: IntArray = np.maximum(mid - half, 0)
            ask: IntArray = mid + half
            iv, delta, vega = _greeks(bid, ask, fwd, strike_c.astype(np.float64), tau, df, call_mask)
            strike_milli = strike_c * _MILLI_PER_CENT
            blocks.append(
                pd.DataFrame(
                    {
                        "occ": [_occ(underlying, expiry, right.value, int(m)) for m in strike_milli],
                        "expiry": np.full(n, np.datetime64(expiry, "ns")),
                        "last_session": np.full(n, np.datetime64(last, "ns")),
                        "right": right.value,
                        "strike_milli": strike_milli,
                        "dte": np.int64(dte),
                        "bid": bid,
                        "ask": ask,
                        "bid_size": pd.array(np.full(n, size), dtype="Int64"),
                        "ask_size": pd.array(np.full(n, size), dtype="Int64"),
                        "oi_prev": pd.array(np.full(n, oi_prev), dtype="Int64"),
                        "iv": iv,
                        "delta": delta,
                        "vega": vega,
                        "fwd": np.int64(fwd),
                        "iv_vendor": planted,
                        "quote_ts": pd.Series(quote_ts, index=range(n), dtype="datetime64[ns, UTC]"),
                    }
                )
            )
    table = pd.concat(blocks, ignore_index=True) if blocks else _empty_table()
    table = table.astype(dict(_DTYPES))[CHAIN_COLUMNS]
    table = table.sort_values(["expiry", "right", "strike_milli"], kind="mergesort").reset_index(drop=True)
    return table, _content_hash(table)


# ======================================================================================================================
# Named chains (15.2)
# ======================================================================================================================


def six_hundred_dollar_chain(underlying: str = "SPY", **kwargs: Any) -> ChainSnapshot:
    """The $600-priced chain of the budget-fit tests (section 8): at ~35 DTE a 0.25 / 0.12-delta wing is far wider than the
    $500 per-trade budget allows, so every credit vertical and condor needs the budget fit."""
    return make_chain(underlying, spot=60_000, **kwargs)


def hundred_dollar_chain(underlying: str = "IWM", **kwargs: Any) -> ChainSnapshot:
    """A $100-priced chain (cheap options: the 2-cent spread class and zero-bid wings are common)."""
    return make_chain(underlying, spot=10_000, **kwargs)


def _assert_calendar_facts(cal: "Calendar", session: date, listed: date, last: date) -> None:
    # the generator asserts the calendar facts it relies on instead of trusting the sentence in DESIGN 15.2
    if cal.is_session(listed) or cal.prev_or_same_session(listed) != last or (last - session).days != 35:
        raise AssertionError(
            f"calendar facts changed: {listed} should be a non-session listed date whose last session is {last}, 35 days out"
        )


def saturday_monthly_chain(underlying: str = "SPY", **kwargs: Any) -> ChainSnapshot:
    """Session Friday 2014-05-16 with pre-2015 listings: the Saturday-dated June monthly 2014-06-21 (last session Friday
    06-20) sits exactly at dte 35, inside the 28-45 entry window, beside Friday-dated weeklies."""
    cal = _as_calendar(kwargs.get("calendar"))
    _assert_calendar_facts(cal, SATURDAY_MONTHLY_SESSION, date(2014, 6, 21), date(2014, 6, 20))
    return make_chain(underlying, session=SATURDAY_MONTHLY_SESSION, saturday_monthlies=True, **kwargs)


def good_friday_chain(underlying: str = "SPY", **kwargs: Any) -> ChainSnapshot:
    """Session Thursday 2014-03-13: the Saturday-dated April monthly 2014-04-19 sits at dte 35 and its last trading day is
    THURSDAY 2014-04-17 (Good Friday 04-18 is a holiday), so `sessions_to_expiry` and every exit count to the Thursday."""
    cal = _as_calendar(kwargs.get("calendar"))
    _assert_calendar_facts(cal, GOOD_FRIDAY_SESSION, date(2014, 4, 19), date(2014, 4, 17))
    if cal.is_session(date(2014, 4, 18)):
        raise AssertionError("2014-04-18 (Good Friday) is expected to be a holiday")
    return make_chain(underlying, session=GOOD_FRIDAY_SESSION, saturday_monthlies=True, **kwargs)


# ======================================================================================================================
# Lookups
# ======================================================================================================================


def _rows_of(chain: ChainSnapshot, expiry: date, right: Right) -> pd.DataFrame:
    rows = chain.side(expiry, right)
    if rows.empty:
        raise DataUnavailable(f"{chain.underlying} {chain.key.session}: no {right.value} rows for expiry {expiry}")
    return rows


def _contract(chain: ChainSnapshot, expiry: date, right: Right, strike_milli: int) -> OptionContract:
    return OptionContract(underlying=chain.underlying, expiry=expiry, right=right, strike_milli=strike_milli)


def strikes(chain: ChainSnapshot, expiry: date, right: Right) -> tuple[int, ...]:
    """Listed `strike_milli` values of one (expiry, right), ascending."""
    return tuple(int(s) for s in _rows_of(chain, expiry, right)["strike_milli"])


def contract_at(chain: ChainSnapshot, expiry: date, right: Right, strike_milli: int) -> OptionContract:
    """The listed contract at that strike (DataUnavailable when it is not in the chain)."""
    c = _contract(chain, expiry, right, strike_milli)
    if chain.quote(c) is None:
        raise DataUnavailable(f"{c.occ} is not listed in the {chain.key.session} chain")
    return c


def quote_of(chain: ChainSnapshot, contract: OptionContract) -> Quote:
    """`chain.quote(contract)`, raising DataUnavailable instead of returning None."""
    q = chain.quote(contract)
    if q is None:
        raise DataUnavailable(f"{contract.occ} is not listed in the {chain.key.session} chain")
    return q


def target_expiry(chain: ChainSnapshot, target_dte: int = 35) -> date:
    """The listed expiry with `argmin |dte - target_dte|` (dte to `last_session`); ties => the later `last_session`."""
    table = chain.table[["expiry", "last_session", "dte"]].drop_duplicates()
    if table.empty:
        raise DataUnavailable("the chain lists no expiry")
    ranked: list[tuple[int, int, date]] = sorted(
        (abs(int(row.dte) - target_dte), -pd.Timestamp(row.last_session).toordinal(), pd.Timestamp(row.expiry).date())
        for row in table.itertuples()
    )
    return ranked[0][2]


def nearest_delta(
    chain: ChainSnapshot,
    expiry: date,
    right: Right,
    target: float,
    *,
    min_bid: int = 1,
    exclude: Sequence[int] = (),
) -> OptionContract:
    """The listed contract whose `|delta|` is nearest `target` among two-sided rows with `bid >= min_bid` (section 8:
    `argmin |abs(delta) - target|`, ties => the strike further OTM). `exclude` skips strike_milli values."""
    rows = _rows_of(chain, expiry, right)
    usable = rows[
        (rows["bid"] >= min_bid) & (rows["ask"] > rows["bid"]) & rows["delta"].notna() & ~rows["strike_milli"].isin(list(exclude))
    ]
    if usable.empty:
        raise DataUnavailable(f"{chain.underlying} {expiry} {right.value}: no usable row for delta {target}")
    distance = (usable["delta"].abs() - target).abs()
    best = usable[distance == distance.min()]
    # ties => further OTM: the highest strike for calls, the lowest for puts
    strike = int(best["strike_milli"].max() if right is Right.CALL else best["strike_milli"].min())
    return _contract(chain, expiry, right, strike)


# ======================================================================================================================
# Variants: every function returns a NEW snapshot with a re-hashed table; the input is never mutated
# ======================================================================================================================


class _Keep:
    """Sentinel type: `set_quote` leaves the field as it is."""


_KEEP: Final = _Keep()


def _with_table(chain: ChainSnapshot, table: pd.DataFrame) -> ChainSnapshot:
    table = table.reset_index(drop=True)
    return replace(chain, table=table, content_hash=_content_hash(table))


def _row_index(chain: ChainSnapshot, contract: OptionContract) -> int:
    t = chain.table
    hit = t.index[
        (t["strike_milli"] == contract.strike_milli) & (t["right"] == contract.right.value) & (t["expiry"] == pd.Timestamp(contract.expiry))
    ]
    if len(hit) != 1 or contract.underlying != chain.underlying:
        raise DataUnavailable(f"{contract.occ} is not listed in the {chain.key.session} chain")
    return int(hit[0])


def set_quote(
    chain: ChainSnapshot,
    contract: OptionContract,
    *,
    bid: int | _Keep = _KEEP,
    ask: int | _Keep = _KEEP,
    bid_size: int | _Keep | None = _KEEP,
    ask_size: int | _Keep | None = _KEEP,
    oi_prev: int | _Keep | None = _KEEP,
    quote_ts: datetime | _Keep | None = _KEEP,
) -> ChainSnapshot:
    """A copy of `chain` with one row's quote fields replaced (`None` = null for the nullable columns). `iv` / `delta` /
    `vega` are re-derived from the new quote exactly as enrichment would (NaN unless the quote is two-sided)."""
    i = _row_index(chain, contract)
    table = chain.table.copy()
    if not isinstance(bid, _Keep):
        if bid < 0:
            raise ValueError("bid must be >= 0")
        table.loc[i, "bid"] = bid
    if not isinstance(ask, _Keep):
        if ask < 0:
            raise ValueError("ask must be >= 0")
        table.loc[i, "ask"] = ask
    for column, value in (("bid_size", bid_size), ("ask_size", ask_size), ("oi_prev", oi_prev)):
        if not isinstance(value, _Keep):
            table.loc[i, column] = pd.NA if value is None else value
    if not isinstance(quote_ts, _Keep):
        table.loc[i, "quote_ts"] = pd.NaT if quote_ts is None else pd.Timestamp(quote_ts)
    if not (isinstance(bid, _Keep) and isinstance(ask, _Keep)):
        row = table.loc[i]
        last = pd.Timestamp(row["last_session"]).date()
        tau = year_fraction(chain.ts, xnys().open_close(last)[1])
        df = math.exp(-chain.rate * tau)
        iv, delta, vega = _greeks(
            np.array([int(row["bid"])], dtype=np.int64),
            np.array([int(row["ask"])], dtype=np.int64),
            float(row["fwd"]),
            np.array([int(row["strike_milli"]) / _MILLI_PER_CENT], dtype=np.float64),
            tau,
            df,
            np.array([row["right"] == Right.CALL.value], dtype=np.bool_),
        )
        table.loc[i, ["iv", "delta", "vega"]] = [float(iv[0]), float(delta[0]), float(vega[0])]
    return _with_table(chain, table)


def crossed_quote(chain: ChainSnapshot, contract: OptionContract) -> ChainSnapshot:
    """bid > ask with a positive bid (the sides swapped): `crossed_or_locked` / `liq:crossed`."""
    q = quote_of(chain, contract)
    if q.bid <= 0 or q.ask <= q.bid:
        raise ValueError(f"{contract.occ} needs a two-sided quote to be crossed")
    return set_quote(chain, contract, bid=q.ask, ask=q.bid)


def locked_quote(chain: ChainSnapshot, contract: OptionContract) -> ChainSnapshot:
    """ask == bid > 0 (locked market)."""
    q = quote_of(chain, contract)
    if q.bid <= 0:
        raise ValueError(f"{contract.occ} needs a positive bid to be locked")
    return set_quote(chain, contract, ask=q.bid)


def zero_bid(chain: ChainSnapshot, contract: OptionContract) -> ChainSnapshot:
    """bid = 0 with the ask kept (at least 1 cent): not `valid()`, still `usable_buy()` / `usable_sell_close()` (10.4)."""
    q = quote_of(chain, contract)
    return set_quote(chain, contract, bid=0, ask=max(q.ask, 1))


def no_quote(chain: ChainSnapshot, contract: OptionContract) -> ChainSnapshot:
    """bid = ask = 0: no usable quote on either side (`no_quote`)."""
    return set_quote(chain, contract, bid=0, ask=0)


def widened_spread(chain: ChainSnapshot, contract: OptionContract, spread_cents: int) -> ChainSnapshot:
    """The same mid with the spread widened to `spread_cents` (bid floored at 0): `liq:spread` / `wide_spread` cases."""
    q = quote_of(chain, contract)
    if spread_cents < 1:
        raise ValueError("spread_cents must be >= 1")
    half_lo, half_hi = spread_cents // 2, spread_cents - spread_cents // 2
    mid = q.mid2 // 2
    return set_quote(chain, contract, bid=max(mid - half_lo, 0), ask=mid + half_hi)


def stale_quote(chain: ChainSnapshot, contract: OptionContract, age_s: int = 3600) -> ChainSnapshot:
    """quote_ts = ts - age_s (older than `health.max_quote_age_s` by default): the `stale_quote` case of recorded / live data."""
    if age_s < 0:
        raise ValueError("age_s must be >= 0")
    return set_quote(chain, contract, quote_ts=chain.ts - timedelta(seconds=age_s))


def thin_oi(chain: ChainSnapshot, contract: OptionContract, oi_prev: int = 10) -> ChainSnapshot:
    """`oi_prev` below `liquidity.min_open_interest` (`liq:oi`)."""
    return set_quote(chain, contract, oi_prev=oi_prev)


def missing_oi(chain: ChainSnapshot, contract: OptionContract) -> ChainSnapshot:
    """`oi_prev` null (passes the filter only with `allow_missing_open_interest`)."""
    return set_quote(chain, contract, oi_prev=None)


def drop_contract(chain: ChainSnapshot, contract: OptionContract) -> ChainSnapshot:
    """The contract removed from the snapshot ("missing leg" / `missing_contract`)."""
    i = _row_index(chain, contract)
    return _with_table(chain, chain.table.drop(index=i))


# ======================================================================================================================
# Structures picked by delta (NOT the CandidateGenerator: no budget fit, width clamp or liquidity economics)
# ======================================================================================================================


def _legs(kind: StructureKind, chain: ChainSnapshot, expiry: date, cfg: CandidatesConfig, min_bid_sold: int) -> list[Leg]:
    def leg(right: Right, side: Side, target: float, exclude: Sequence[int] = ()) -> Leg:
        c = nearest_delta(chain, expiry, right, target, min_bid=min_bid_sold if side is Side.SELL else 1, exclude=exclude)
        return Leg(contract=c, side=side)

    if kind is StructureKind.LONG_CALL:
        return [leg(Right.CALL, Side.BUY, cfg.long_delta)]
    if kind is StructureKind.LONG_PUT:
        return [leg(Right.PUT, Side.BUY, cfg.long_delta)]
    if kind is StructureKind.CALL_DEBIT:
        long = leg(Right.CALL, Side.BUY, cfg.debit_long_delta)
        return [long, leg(Right.CALL, Side.SELL, cfg.debit_short_delta, (long.contract.strike_milli,))]
    if kind is StructureKind.PUT_DEBIT:
        long = leg(Right.PUT, Side.BUY, cfg.debit_long_delta)
        return [long, leg(Right.PUT, Side.SELL, cfg.debit_short_delta, (long.contract.strike_milli,))]
    if kind is StructureKind.CALL_CREDIT:
        short = leg(Right.CALL, Side.SELL, cfg.credit_short_delta)
        return [short, leg(Right.CALL, Side.BUY, cfg.credit_long_delta, (short.contract.strike_milli,))]
    if kind is StructureKind.PUT_CREDIT:
        short = leg(Right.PUT, Side.SELL, cfg.credit_short_delta)
        return [short, leg(Right.PUT, Side.BUY, cfg.credit_long_delta, (short.contract.strike_milli,))]
    short_put = leg(Right.PUT, Side.SELL, cfg.condor_short_delta)
    short_call = leg(Right.CALL, Side.SELL, cfg.condor_short_delta)
    return [
        short_put,
        leg(Right.PUT, Side.BUY, cfg.condor_long_delta, (short_put.contract.strike_milli,)),
        short_call,
        leg(Right.CALL, Side.BUY, cfg.condor_long_delta, (short_call.contract.strike_milli,)),
    ]


def make_structure(
    chain: ChainSnapshot,
    kind: StructureKind,
    *,
    expiry: date | None = None,
    target_dte: int = 35,
    cfg: CandidatesConfig | None = None,
    min_bid_sold: int = 10,
) -> Structure:
    """A `Structure` of `kind` on `chain`, legs picked by the [candidates] delta targets (8: long 0.35; debit 0.50 / 0.25;
    credit 0.25 / 0.12; condor 0.16 / 0.07), sold legs from rows with `bid >= min_bid_sold`, canonical leg order (puts
    before calls, ascending strike), `last_session` copied from the chain row. Raises ValueError when the picked legs do
    not form the kind's defined-risk shape (a very coarse grid)."""
    from jevbot.structmath import defined_risk_ok  # pure, WP00; imported lazily so the factory stays importable alone

    e = target_expiry(chain, target_dte) if expiry is None else expiry
    legs = _legs(StructureKind(kind), chain, e, cfg if cfg is not None else CandidatesConfig(), min_bid_sold)
    ordered = tuple(sorted(legs, key=lambda leg: (leg.contract.right is Right.CALL, leg.contract.strike_milli)))
    if not defined_risk_ok(kind, ordered):
        raise ValueError(f"the delta-picked legs of {kind.value} on {chain.key.session} do not form a defined-risk {kind.value}")
    return Structure(kind=kind, underlying=chain.underlying, expiry=e, last_session=chain.last_session(e), legs=ordered)


def open_legs(structure: Structure) -> tuple[OrderLeg, ...]:
    """OPEN order legs: BUY -> buy_to_open, SELL -> sell_to_open (2.4)."""
    return tuple(
        OrderLeg(
            contract=leg.contract,
            side=leg.side,
            position_intent=PositionIntent.BTO if leg.side is Side.BUY else PositionIntent.STO,
            ratio=leg.ratio,
        )
        for leg in structure.legs
    )


def close_legs(structure: Structure) -> tuple[OrderLeg, ...]:
    """CLOSE / KILL order legs: sides flipped, long leg -> (SELL, sell_to_close), short leg -> (BUY, buy_to_close) (2.4)."""
    return tuple(
        OrderLeg(
            contract=leg.contract,
            side=Side.SELL if leg.side is Side.BUY else Side.BUY,
            position_intent=PositionIntent.STC if leg.side is Side.BUY else PositionIntent.BTC,
            ratio=leg.ratio,
        )
        for leg in structure.legs
    )


# ======================================================================================================================
# Scenario: zero-bid wing on a winning condor (10.4 / 10.5)
# ======================================================================================================================


@dataclass(frozen=True)
class CondorScenario:
    """An iron condor opened on `open_chain` and re-quoted on `later_chain` (same spot, `later_dte` calendar days before its
    last session): both long wings are quoted `0 x ask` there - not `valid()`, still `usable_sell_close()` - the short
    legs are two-sided, and the conservative liquidation cost is below half of the opening credit on every band, so the
    profit-target close fires, the time-exit close fills with the wings sold at 0, and the mark at 0 keeps `stale_marks`
    at 0. `open_credit_worst` / `open_credit_mid` are cents/share received at open (worst band = bid sold / ask bought;
    mid band per 10.3); `liquidation_cost` = ask of the shorts minus bid of the longs on `later_chain` (10.5)."""

    structure: Structure
    open_chain: ChainSnapshot
    later_chain: ChainSnapshot
    wings: tuple[OptionContract, OptionContract]
    open_credit_worst: Cents
    open_credit_mid: Cents
    liquidation_cost: Cents


def _band_credit(chain: ChainSnapshot, structure: Structure) -> tuple[Cents, Cents]:
    """(worst, mid) credit received for OPENING a credit structure, cents/share: the 10.3 band arithmetic, integers only
    (worst: sell at bid, buy at ask; mid: sell at (bid + ask) // 2, buy at cdiv(bid + ask, 2))."""
    worst = mid = 0
    for leg in structure.legs:
        q = quote_of(chain, leg.contract)
        if leg.side is Side.SELL:
            worst += q.bid
            mid += (q.bid + q.ask) // 2
        else:
            worst -= q.ask
            mid -= -(-(q.bid + q.ask) // 2)
    return worst, mid


def winning_condor(
    underlying: str = "SPY",
    *,
    session: date = DEFAULT_SESSION,
    later_dte: int = 10,
    spot: Cents = DEFAULT_SPOT,
    calendar: "Calendar | None" = None,
    **kwargs: Any,
) -> CondorScenario:
    """Build the 10.4 / 10.5 fixture case. `kwargs` go to `make_chain` for both snapshots (not `session`, `spot`,
    `expiries` or `calendar`)."""
    cal = _as_calendar(calendar)
    open_chain = make_chain(underlying, session=session, spot=spot, calendar=cal, **kwargs)
    structure = make_structure(open_chain, StructureKind.IRON_CONDOR)
    later = cal.prev_or_same_session(structure.last_session - timedelta(days=later_dte))
    if not session < later < structure.last_session:
        raise ValueError(f"later_dte={later_dte} does not land strictly between {session} and {structure.last_session}")
    alive = tuple(e for e in open_chain.expiries() if cal.prev_or_same_session(e) > later)
    later_chain = make_chain(underlying, session=later, spot=spot, expiries=alive, calendar=cal, **kwargs)
    wings = tuple(leg.contract for leg in structure.legs if leg.side is Side.BUY)
    if len(wings) != 2:
        raise AssertionError("an iron condor has two long wings")
    for wing in wings:
        later_chain = zero_bid(later_chain, wing)
    worst, mid = _band_credit(open_chain, structure)
    cost = 0
    for leg in structure.legs:
        q = quote_of(later_chain, leg.contract)
        if leg.side is Side.SELL:
            if not q.valid():
                raise AssertionError(f"short leg {leg.contract.occ} must stay two-sided on the later chain")
            cost += q.ask
        else:
            cost -= q.bid
    if worst - cost < -(-mid // 2):
        raise AssertionError(f"the condor is not winning by half its credit: credit worst={worst} mid={mid} liquidation={cost}")
    return CondorScenario(
        structure=structure,
        open_chain=open_chain,
        later_chain=later_chain,
        wings=(wings[0], wings[1]),
        open_credit_worst=worst,
        open_credit_mid=mid,
        liquidation_cost=cost,
    )


# ======================================================================================================================
# Self-tests (collected by tests/conftest.py::pytest_collect_file)
# ======================================================================================================================


def _iter_contracts(chain: ChainSnapshot) -> Iterator[OptionContract]:
    for row in chain.table.itertuples():
        yield OptionContract(
            underlying=chain.underlying, expiry=pd.Timestamp(row.expiry).date(), right=Right(row.right), strike_milli=int(row.strike_milli)
        )


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _black76_cents(fwd: float, strike: float, tau: float, sigma: float, rate: float, *, call: bool) -> float:
    """An independent scalar Black-76 (erf-based), used only to check the factory's prices."""
    v = sigma * math.sqrt(tau)
    d1 = (math.log(fwd / strike) + 0.5 * v * v) / v
    d2 = d1 - v
    df = math.exp(-rate * tau)
    if call:
        return df * (fwd * _norm_cdf(d1) - strike * _norm_cdf(d2))
    return df * (strike * _norm_cdf(-d2) - fwd * _norm_cdf(-d1))


def test_default_chain_schema_and_sorting() -> None:
    from jevbot.occ import parse_occ

    chain = make_chain()
    t = chain.table
    assert list(t.columns) == CHAIN_COLUMNS
    assert {c: str(t[c].dtype) for c in CHAIN_COLUMNS} == dict(_DTYPES)
    assert chain.key == SnapshotKey(session=DEFAULT_SESSION, slot=Slot.EOD)
    assert chain.ts == xnys().open_close(DEFAULT_SESSION)[1] == chain.knowable_at
    assert chain.ts.isoformat() == "2024-05-17T20:00:00+00:00"  # a regular 16:00 ET close in EDT
    assert (chain.spot, chain.spot_measure, chain.source, chain.fidelity, chain.rate) == (
        45_000,
        "parity",
        "mirror",
        Fidelity.EOD_QUOTES,
        0.04,
    )
    assert chain.div_unmodelled is False
    keys = list(zip(t["expiry"], t["right"], t["strike_milli"], strict=True))
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    assert t["occ"].is_unique
    # calendar arithmetic: dte to last_session, the window, the forward, quote_ts null for EOD data
    for row in t.itertuples():
        last = pd.Timestamp(row.last_session).date()
        assert last == xnys().prev_or_same_session(pd.Timestamp(row.expiry).date())
        assert int(row.dte) == (last - DEFAULT_SESSION).days and 1 <= int(row.dte) <= _DEFAULT_MAX_DTE
        assert abs(math.log(int(row.strike_milli) / _MILLI_PER_CENT / int(row.fwd))) <= _DEFAULT_MONEYNESS_WINDOW + 1e-12
    assert t["quote_ts"].isna().all()
    assert (t["bid_size"] == 100).all() and (t["ask_size"] == 100).all() and (t["oi_prev"] == 2000).all()
    # every occ symbol parses back to its row's contract (and to the OCC form the mirror / Alpaca use)
    for contract, sym in zip(_iter_contracts(chain), t["occ"], strict=True):
        assert parse_occ(sym) == contract and contract.occ == sym
    # the listed expiries: eight weekly Fridays plus the monthlies out to 120 days; the June monthly at dte 35
    expiries = chain.expiries()
    assert expiries[:4] == (date(2024, 5, 24), date(2024, 5, 31), date(2024, 6, 7), date(2024, 6, 14))
    assert date(2024, 6, 21) in expiries and date(2024, 8, 16) in expiries and date(2024, 9, 20) not in expiries
    assert len(expiries) == 10
    assert chain.last_session(date(2024, 6, 21)) == date(2024, 6, 21) and target_expiry(chain) == date(2024, 6, 21)
    assert len(t) == 2 * sum(len(chain.side(e, Right.CALL)) for e in expiries)
    assert len(strikes(chain, date(2024, 6, 21), Right.PUT)) == 323  # floor(451.73 e^0.35) - ceil(451.73 e^-0.35) + 1 = 641 - 319 + 1


def test_prices_agree_with_an_independent_black76_and_the_spread_rule() -> None:
    chain = make_chain()
    expiry = date(2024, 6, 21)
    last = xnys().prev_or_same_session(expiry)
    tau = (xnys().open_close(last)[1] - chain.ts).total_seconds() / (365 * 86400)
    assert abs(tau - 35 / 365) < 1e-12
    fwd = chain.forward(expiry)
    assert fwd == round(45_000 * math.exp(0.04 * 35 / 365)) == 45_173
    smile = Smile()
    for strike_c in (40_000, 43_000, 45_000, 47_000, 50_000):
        k = math.log(strike_c / fwd)
        a, b, c = smile.coefficients(35, tau)
        sigma = math.sqrt((a + b * k + c * k * k) / tau)
        for right, call in ((Right.CALL, True), (Right.PUT, False)):
            q = quote_of(chain, contract_at(chain, expiry, right, strike_c * _MILLI_PER_CENT))
            theo = _black76_cents(fwd, strike_c, tau, sigma, 0.04, call=call)
            mid = math.floor(theo + 0.5)
            spread = 2 if mid < 100 else 4 if mid < 500 else 6
            assert (q.bid, q.ask) == (max(mid - spread // 2, 0), mid + spread // 2), (strike_c, right)
            assert q.mid2 == 2 * mid
    # hand-checked literals (35 DTE, F = 451.73, smile IV 16.31% at K = 450): Black-76 gives 993.66c for the call and
    # 821.32c for the put; rounded mids 994 / 821 with the 6-cent class => 991 x 997 and 818 x 824
    atm_call = quote_of(chain, contract_at(chain, expiry, Right.CALL, 450_000))
    atm_put = quote_of(chain, contract_at(chain, expiry, Right.PUT, 450_000))
    assert (atm_call.bid, atm_call.ask) == (991, 997) and (atm_put.bid, atm_put.ask) == (818, 824)
    assert (
        atm_call.iv is not None and abs(atm_call.iv - 0.1631) < 5e-4 and atm_call.delta is not None and abs(atm_call.delta - 0.538) < 2e-3
    )
    # the spread classes 2 / 4 / 6 by mid level, symmetric around the integer mid, bid floored at 0, ask >= 1
    t = chain.table
    mid_all = (t["bid"] + t["ask"]) // 2
    expected = np.where(mid_all < 100, 2, np.where(mid_all < 500, 4, 6))
    two_sided = t["bid"] > 0
    assert ((t["ask"] - t["bid"])[two_sided] == expected[two_sided]).all()
    assert (t["ask"] >= 1).all() and (t["bid"] >= 0).all()
    assert (t["bid"] == 0).any(), "far-OTM rows carry a zero bid like real data"


def test_parity_recovers_the_forward_and_greeks_match_the_planted_smile() -> None:
    chain = make_chain()
    for expiry in chain.expiries():
        calls = chain.side(expiry, Right.CALL).set_index("strike_milli")
        puts = chain.side(expiry, Right.PUT).set_index("strike_milli")
        fwd = chain.forward(expiry)
        last = pd.Timestamp(calls["last_session"].iloc[0]).date()
        tau = year_fraction(chain.ts, xnys().open_close(last)[1])
        rows: list[tuple[float, float]] = []
        for strike in calls.index.intersection(puts.index):
            c, p = calls.loc[strike], puts.loc[strike]
            if c["bid"] > 0 and c["ask"] > c["bid"] and p["bid"] > 0 and p["ask"] > p["bid"]:
                c_mid, p_mid = (c["bid"] + c["ask"]) / 2, (p["bid"] + p["ask"]) / 2
                rows.append((abs(c_mid - p_mid), strike / _MILLI_PER_CENT + math.exp(0.04 * tau) * (c_mid - p_mid)))
        rows.sort()
        assert len(rows) >= 3
        recovered = round(float(np.median([f for _, f in rows[:3]])))
        assert abs(recovered - fwd) <= 1, (expiry, recovered, fwd)  # the 5.2 parity forward within one cent
    # near-the-money solved IVs agree with the planted smile; deltas are signed and bounded; invalid rows carry NaN
    t = chain.table
    k = np.log(t["strike_milli"] / _MILLI_PER_CENT / t["fwd"])
    near = (k.abs() < 0.05) & (t["dte"] >= 14)
    assert near.sum() > 100
    assert (t.loc[near, "iv"] - t.loc[near, "iv_vendor"]).abs().max() < 0.004
    calls_mask, puts_mask = t["right"] == "C", t["right"] == "P"
    assert t.loc[near & calls_mask, "delta"].between(0.0, 1.0).all() and t.loc[near & puts_mask, "delta"].between(-1.0, 0.0).all()
    assert (t.loc[near, "vega"] > 0).all()
    invalid = ~((t["bid"] > 0) & (t["ask"] > t["bid"]))
    assert invalid.sum() > 0 and t.loc[invalid, ["iv", "delta", "vega"]].isna().all().all()
    assert t.loc[~invalid & t["iv"].notna(), "iv"].between(_IV_LO, _IV_HI, inclusive="neither").all()
    assert t["iv_vendor"].notna().all()


def test_delta_targets_are_reachable_and_structures_are_defined_risk() -> None:
    from jevbot.structmath import defined_risk_ok

    chain = make_chain()
    expiry = target_expiry(chain)
    cfg = CandidatesConfig()
    targets = (
        cfg.long_delta,
        cfg.debit_long_delta,
        cfg.debit_short_delta,
        cfg.credit_short_delta,
        cfg.credit_long_delta,
        cfg.condor_short_delta,
        cfg.condor_long_delta,
    )
    for target in targets:
        for right in Right:
            c = nearest_delta(chain, expiry, right, target, min_bid=10)
            q = quote_of(chain, c)
            assert q.delta is not None and abs(abs(q.delta) - target) <= cfg.delta_tolerance and q.bid >= 10
    for kind in StructureKind:
        s = make_structure(chain, kind)
        assert s.kind is kind and s.expiry == expiry and s.last_session == date(2024, 6, 21) and s.underlying == "SPY"
        assert defined_risk_ok(kind, s.legs)
        assert list(s.legs) == sorted(s.legs, key=lambda leg: (leg.contract.right is Right.CALL, leg.contract.strike_milli))
        assert all(leg.contract.expiry == expiry and leg.ratio == 1 for leg in s.legs)
        opened, closed = open_legs(s), close_legs(s)
        assert [o.side for o in opened] == [leg.side for leg in s.legs]
        assert [o.position_intent for o in opened] == [PositionIntent.BTO if leg.side is Side.BUY else PositionIntent.STO for leg in s.legs]
        assert [c.side for c in closed] == [Side.SELL if leg.side is Side.BUY else Side.BUY for leg in s.legs]
        assert [c.position_intent for c in closed] == [PositionIntent.STC if leg.side is Side.BUY else PositionIntent.BTC for leg in s.legs]
    condor = make_structure(chain, StructureKind.IRON_CONDOR)
    short_put = next(leg for leg in condor.legs if leg.contract.right is Right.PUT and leg.side is Side.SELL)
    short_call = next(leg for leg in condor.legs if leg.contract.right is Right.CALL and leg.side is Side.SELL)
    assert short_put.contract.strike_milli < 450_000 < short_call.contract.strike_milli
    assert len(condor.legs) == 4 and condor.width == max(condor.wing_widths) > 0
    # ties => further OTM: two rows with an identical delta resolve to the higher call strike / the lower put strike
    tied = chain.table.copy()
    for strike in (470_000, 471_000):
        tied.loc[(tied["expiry"] == pd.Timestamp(expiry)) & (tied["right"] == "C") & (tied["strike_milli"] == strike), "delta"] = 0.31
    for strike in (430_000, 431_000):
        tied.loc[(tied["expiry"] == pd.Timestamp(expiry)) & (tied["right"] == "P") & (tied["strike_milli"] == strike), "delta"] = -0.31
    tie = _with_table(chain, tied)
    assert nearest_delta(tie, expiry, Right.CALL, 0.31).strike_milli == 471_000
    assert nearest_delta(tie, expiry, Right.PUT, 0.31).strike_milli == 430_000


def test_named_chains_saturday_monthly_good_friday_and_price_levels() -> None:
    sat = saturday_monthly_chain()
    assert sat.key.session == date(2014, 5, 16) and date(2014, 6, 21) in sat.expiries()
    assert not xnys().is_session(date(2014, 6, 21)) and sat.last_session(date(2014, 6, 21)) == date(2014, 6, 20)
    assert target_expiry(sat) == date(2014, 6, 21)
    june = sat.side(date(2014, 6, 21), Right.CALL)
    assert (june["dte"] == 35).all()  # 35 calendar days to FRIDAY 06-20, not 36 to the Saturday
    close_last = xnys().open_close(date(2014, 6, 20))[1]
    assert abs(year_fraction(sat.ts, close_last) - 35 / 365) < 1e-12
    assert sat.forward(date(2014, 6, 21)) == round(45_000 * math.exp(0.04 * 35 / 365))
    s = make_structure(sat, StructureKind.PUT_CREDIT)
    assert s.expiry == date(2014, 6, 21) and s.last_session == date(2014, 6, 20)
    assert all(leg.contract.expiry == date(2014, 6, 21) for leg in s.legs) and s.legs[0].contract.occ.startswith("SPY140621P")
    assert date(2014, 5, 23) in sat.expiries() and date(2014, 7, 19) in sat.expiries()  # Friday weeklies, Saturday monthlies

    gf = good_friday_chain()
    assert gf.key.session == date(2014, 3, 13) and date(2014, 4, 19) in gf.expiries()
    assert gf.last_session(date(2014, 4, 19)) == date(2014, 4, 17)  # Thursday: Good Friday 04-18 is a holiday
    assert (gf.side(date(2014, 4, 19), Right.PUT)["dte"] == 35).all()
    assert xnys().sessions_between(date(2014, 3, 13), date(2014, 4, 17)) == 25  # counted by hand: 12 in March, 13 in April
    assert date(2014, 4, 18) not in gf.expiries() and date(2014, 4, 17) not in gf.expiries()
    # the 2012 case of the calendar tests, through an explicit listing
    old = make_chain(session=date(2012, 2, 10), expiries=[date(2012, 3, 17)])
    assert old.expiries() == (date(2012, 3, 17),) and old.last_session(date(2012, 3, 17)) == date(2012, 3, 16)
    assert int(old.table["dte"].iloc[0]) == (date(2012, 3, 16) - date(2012, 2, 10)).days == 35
    # a modern Good-Friday week lists the Thursday (5.10): 2024-03-29 is Good Friday
    modern = make_chain(session=date(2024, 3, 8))
    assert date(2024, 3, 28) in modern.expiries() and date(2024, 3, 29) not in modern.expiries()
    assert modern.last_session(date(2024, 3, 28)) == date(2024, 3, 28)

    six = six_hundred_dollar_chain()
    assert six.spot == 60_000 and six.forward(date(2024, 6, 21)) == round(60_000 * math.exp(0.04 * 35 / 365))
    spread = make_structure(six, StructureKind.PUT_CREDIT)
    worst_credit = sum(
        (quote_of(six, leg.contract).bid if leg.side is Side.SELL else -quote_of(six, leg.contract).ask) for leg in spread.legs
    )
    assert spread.width >= 1_000 and (spread.width - worst_credit) * 100 > 50_000, "the delta rule alone exceeds the $500 budget floor"
    small = hundred_dollar_chain()
    assert small.underlying == "IWM" and small.spot == 10_000 and (small.table["ask"] - small.table["bid"]).max() <= 6


def test_variants_return_new_snapshots_and_leave_the_input_untouched() -> None:
    import pytest

    chain = make_chain()
    before = chain.content_hash
    expiry = target_expiry(chain)
    c = contract_at(chain, expiry, Right.PUT, 440_000)
    base = quote_of(chain, c)
    assert base.valid() and base.iv is not None

    crossed = quote_of(crossed_quote(chain, c), c)
    assert crossed.bid == base.ask and crossed.ask == base.bid and crossed.bid > crossed.ask > 0
    assert crossed.iv is None and crossed.delta is None and crossed.vega is None and not crossed.valid()
    locked = quote_of(locked_quote(chain, c), c)
    assert locked.bid == locked.ask == base.bid and not locked.valid() and not locked.usable_buy()
    zb = quote_of(zero_bid(chain, c), c)
    assert (zb.bid, zb.ask) == (0, base.ask) and not zb.valid() and zb.usable_sell_close() and zb.usable_buy() and zb.iv is None
    nq = quote_of(no_quote(chain, c), c)
    assert (nq.bid, nq.ask) == (0, 0) and not nq.usable_buy() and not nq.usable_sell_close()
    wide = quote_of(widened_spread(chain, c, 40), c)
    assert wide.ask - wide.bid == 40 and wide.mid2 // 2 == base.mid2 // 2 and wide.iv is not None
    assert abs(wide.iv - base.iv) < 1e-9  # same mid => same solved IV
    thin = quote_of(thin_oi(chain, c), c)
    assert thin.oi_prev == 10 and (thin.bid, thin.ask, thin.iv) == (base.bid, base.ask, base.iv)
    assert quote_of(missing_oi(chain, c), c).oi_prev is None
    sized = quote_of(set_quote(chain, c, bid_size=3, ask_size=None), c)
    assert (sized.bid_size, sized.ask_size) == (3, None)
    dropped = drop_contract(chain, c)
    assert (
        dropped.quote(c) is None
        and len(dropped.table) == len(chain.table) - 1
        and dropped.quote(contract_at(chain, expiry, Right.PUT, 441_000)) is not None
    )
    with_ts = set_quote(chain, c, quote_ts=chain.ts - timedelta(seconds=7))
    assert quote_of(with_ts, c).quote_ts == chain.ts - timedelta(seconds=7)

    assert chain.content_hash == before and quote_of(chain, c) == base  # the input is untouched
    assert len({before, crossed_quote(chain, c).content_hash, zero_bid(chain, c).content_hash, dropped.content_hash}) == 4
    same = quote_of(set_quote(chain, c, bid=base.bid), c)  # an unchanged quote re-solves the same greeks
    assert same.iv is not None and abs(same.iv - base.iv) < 1e-12 and same.delta == base.delta
    with pytest.raises(DataUnavailable):
        set_quote(chain, OptionContract(underlying="SPY", expiry=expiry, right=Right.PUT, strike_milli=1), bid=1)
    with pytest.raises(DataUnavailable):
        contract_at(chain, expiry, Right.CALL, 999_000)
    with pytest.raises(DataUnavailable):
        quote_of(chain, OptionContract(underlying="QQQ", expiry=expiry, right=Right.PUT, strike_milli=440_000))
    with pytest.raises(ValueError, match="skew"):
        Smile(skew=-8.0, curvature=10.0)
    with pytest.raises(ValueError, match="two-sided"):
        crossed_quote(chain, next(c2 for c2 in _iter_contracts(chain) if quote_of(chain, c2).bid == 0))


def test_recorded_fidelity_slots_and_stale_quotes() -> None:
    import pytest

    dec = make_chain(slot=Slot.DEC, fidelity=Fidelity.RECORDED_INDICATIVE)
    cal = xnys()
    assert dec.ts == cal.offset_from_close(DEFAULT_SESSION, 25) and dec.key.slot is Slot.DEC
    assert (dec.source, dec.spot_measure) == ("alpaca_recorded", "live_mid")
    assert dec.table["quote_ts"].notna().all() and str(dec.table["quote_ts"].dtype) == "datetime64[ns, UTC]"
    c = contract_at(dec, target_expiry(dec), Right.CALL, 460_000)
    fresh = quote_of(dec, c)
    assert fresh.quote_ts == dec.ts - timedelta(seconds=_QUOTE_LAG_S)
    stale = quote_of(stale_quote(dec, c, age_s=3600), c)
    assert stale.quote_ts == dec.ts - timedelta(seconds=3600) and (stale.bid, stale.ask) == (fresh.bid, fresh.ask)
    eod = make_chain(slot=Slot.EOD, fidelity=Fidelity.RECORDED_INDICATIVE)
    assert eod.ts == cal.offset_from_close(DEFAULT_SESSION, -2)  # two minutes AFTER the close
    assert make_chain(slot=Slot.EXEC, fidelity=Fidelity.LIVE_INDICATIVE).ts == cal.offset_from_close(DEFAULT_SESSION, 20)
    received = dec.ts + timedelta(seconds=3)
    assert make_chain(slot=Slot.DEC, fidelity=Fidelity.RECORDED_INDICATIVE, knowable_at=received).knowable_at == received
    with pytest.raises(ValueError, match="only the eod slot"):
        make_chain(slot=Slot.DEC)
    syn = make_chain(fidelity=Fidelity.SYNTHETIC)
    assert (syn.source, syn.spot_measure) == ("synthetic", "synthetic") and syn.content_hash == make_chain(
        fidelity=Fidelity.SYNTHETIC
    ).content_hash
    assert make_chain(source="custom", spot_measure="file_close", div_unmodelled=True).source == "custom"


def test_winning_condor_scenario() -> None:
    sc = winning_condor()
    assert sc.structure.kind is StructureKind.IRON_CONDOR and sc.open_chain.key.session == DEFAULT_SESSION
    assert sc.later_chain.key.session == date(2024, 6, 11) and sc.later_chain.spot == sc.open_chain.spot
    assert sc.structure.last_session == date(2024, 6, 21)
    assert (sc.structure.last_session - sc.later_chain.key.session).days == 10
    assert xnys().sessions_between(sc.later_chain.key.session, sc.structure.last_session) == 7  # Juneteenth 06-19 is a holiday
    assert set(sc.wings) == {leg.contract for leg in sc.structure.legs if leg.side is Side.BUY}
    for wing in sc.wings:
        q = quote_of(sc.later_chain, wing)
        assert q.bid == 0 < q.ask and not q.valid() and q.usable_sell_close() and q.iv is None
        assert quote_of(sc.open_chain, wing).valid()
    for leg in sc.structure.legs:
        if leg.side is Side.SELL:
            assert quote_of(sc.later_chain, leg.contract).valid()
    assert sc.liquidation_cost == sum(
        quote_of(sc.later_chain, leg.contract).ask if leg.side is Side.SELL else -quote_of(sc.later_chain, leg.contract).bid
        for leg in sc.structure.legs
    )
    assert 0 < sc.open_credit_worst <= sc.open_credit_mid
    assert sc.open_credit_worst - sc.liquidation_cost >= math.ceil(sc.open_credit_mid / 2)  # >= 50% of the max profit on every band
    assert sc.later_chain.expiries() == tuple(
        e for e in sc.open_chain.expiries() if xnys().prev_or_same_session(e) > sc.later_chain.key.session
    )


def test_determinism_and_argument_validation() -> None:
    import pytest

    a, b = make_chain(), make_chain()
    assert a.content_hash == b.content_hash and a.table.equals(b.table)
    assert make_chain(spot=45_001).content_hash != a.content_hash
    assert make_chain(smile=Smile(skew=-3.0, curvature=6.0)).content_hash != a.content_hash
    assert make_chain(underlying="QQQ").table["occ"].str.startswith("QQQ").all()
    sized = make_chain(size=7, oi_prev=55).table
    assert sized["bid_size"].eq(7).all() and sized["ask_size"].eq(7).all() and sized["oi_prev"].eq(55).all()
    with pytest.raises(ValueError):
        make_chain(spot=0)
    with pytest.raises(ValueError):
        make_chain(strike_step=0)
    with pytest.raises(ValueError):
        make_chain(ts=datetime(2024, 5, 17, 20, 0, 0))  # noqa: DTZ001 - the naive datetime IS the case under test
    empty = make_chain(expiries=[date(2024, 5, 17)])  # dte 0 => no rows
    assert empty.table.empty and list(empty.table.columns) == CHAIN_COLUMNS and empty.expiries() == ()
    with pytest.raises(DataUnavailable):
        target_expiry(empty)
    assert listed_expiries(date(2024, 5, 17))[:2] == (date(2024, 5, 24), date(2024, 5, 31))
    # a Thursday session in March 2014 with pre-2015 listings: tomorrow's Friday weekly (dte 1), then the Saturday monthly
    assert listed_expiries(date(2014, 3, 13), saturday_monthlies=True)[:3] == (date(2014, 3, 14), date(2014, 3, 22), date(2014, 3, 28))
    assert listed_expiries(date(2024, 5, 17), max_dte=30) == (date(2024, 5, 24), date(2024, 5, 31), date(2024, 6, 7), date(2024, 6, 14))
