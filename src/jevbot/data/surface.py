"""The option-surface maths: parity forwards and spot, chain enrichment, the ATM term, the smile fit and the implied
digital (DESIGN.md 3.7, 5.2, 5.3, 6.4; WP01).

Everything here is pure, vectorised and calendar-aware. Two rules run through the whole module:

* **`T_E` is always `year_fraction(ts, close(last_session(E)))`, never `close(E)`.** `expiry` is contract identity only: a
  standard monthly listed before February 2015 carries a SATURDAY date and a Good-Friday week's Friday is a holiday, so
  `close(E)` does not exist and a date-based `T_E` would be one to two days too long (about 3% of IV at 7-10 DTE).
  `last_session(E) = calendar.prev_or_same_session(E)` (Conventions, INV-11).
* **Two clocks.** `year_fraction` (calendar / 365) discounts, builds forwards and annualises IV levels; `cal.trading_time`
  (sessions) allocates TOTAL VARIANCE to a horizon between or below listed expiries (V13). `total_variance_at` and
  `implied_prob_above` share one node set - every expiry with `dte >= 1` that has a forward and an ATM IV, no `dte >= 7`
  filter - so the expected move shown to Jev and the option-implied reference it is scored against can never disagree
  (5.3, 6.4).

Own IV everywhere (D21): `enrich` re-solves Black-76 on the row's own parity forward from the mid of two-sided quotes and
accepts `0.02 < iv < 5.0`; the vendor's IV is carried as `iv_vendor` for QC only. ATM IV per expiry is the FITTED smile at
`k = 0` when a fit exists (many strikes, vega / spread weighted) and the two-strike interpolation otherwise; the two-strike
value is always computed too and stored beside it (`atm_iv_2s`, `daily.iv30_2s_bp`) so that `data derive` can raise the
`atm_iv_divergence` anomaly when the two disagree by more than `data.atm_iv_tolerance` (5.3).

No IO, no clock, no RNG.
"""

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from jevbot import bs
from jevbot.cal import trading_time, year_fraction
from jevbot.errors import DataError, DataUnavailable
from jevbot.types import Cents, ChainSnapshot, ScheduledEvent, SmileFit

if TYPE_CHECKING:
    from jevbot.protocols import Calendar

__all__ = [
    "IV_MAX",
    "IV_MIN",
    "MIN_FIT_POINTS",
    "MIN_PARITY_STRIKES",
    "SurfaceNode",
    "atm_term",
    "chain_nodes",
    "const_maturity_iv",
    "enrich",
    "fit_smile",
    "fit_smiles",
    "implied_prob_above",
    "parity_forwards",
    "parity_spot",
    "surface_nodes",
    "term_of",
    "total_variance_at",
    "two_strike_atm_iv",
]

BoolArray = npt.NDArray[np.bool_]

IV_MIN: Final = 0.02  # an enriched row keeps its solved IV iff IV_MIN < iv < IV_MAX (5.3)
IV_MAX: Final = 5.0
MIN_PARITY_STRIKES: Final = 3  # an expiry needs 3 two-sided call/put pairs to get a forward (5.2)
MIN_FIT_POINTS: Final = 5  # a smile fit needs 5 OTM points inside the window (6.4 step 1)
FIT_K_HALF_WIDTH_MULT: Final = 3.0  # |k| <= 3 * atm_iv * sqrt(T_E)
FRONT_EXPIRY_MIN_DTE: Final = 2  # the parity spot's front expiry needs dte >= 2 (5.2)
_MILLI_PER_CENT: Final = 10.0
_MIN_SPREAD_CENTS: Final = 1.0  # weights are vega / max(spread, 1 cent)
_ADDED_COLUMNS: Final[tuple[str, ...]] = ("last_session", "dte", "iv", "delta", "vega", "fwd")
_INTERPOLATED: Final = "interpolated"
_EXTRAPOLATED: Final = "extrapolated"
_SMILE_DIGITAL: Final = "smile_digital"
_ND2_PLAIN: Final = "nd2_plain"

# `atm_term` node: (tau_years, tt_sessions, atm_iv_bp, atm_iv_2s_bp, fwd_c)  - the 3.7 tuple
TermNode = tuple[float, float, int, int, int]


# ======================================================================================================================
# Small shared helpers
# ======================================================================================================================


def _as_utc(name: str, value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be tz-aware (all datetimes are tz-aware UTC)")
    return value


def _session_of(ts: datetime, session: date | None, calendar: "Calendar") -> date:
    """The snapshot's session: the caller's value, else the UTC date of `ts` (every XNYS close is 18:00-21:00 UTC, so the
    two agree for the `dec` / `exec` / `eod` slots alike)."""
    chosen = session if session is not None else ts.date()
    if not calendar.is_session(chosen):
        raise ValueError(f"{chosen.isoformat()} is not an XNYS session: pass the snapshot's session explicitly")
    return chosen


def _expiry_dates(table: pd.DataFrame) -> tuple[date, ...]:
    """The distinct listed expiries of a chain table, ascending."""
    if "expiry" not in table.columns:
        raise DataError("a chain table needs an `expiry` column")
    values = pd.to_datetime(pd.Series(table["expiry"])).dropna().unique()
    return tuple(sorted(pd.Timestamp(v).date() for v in values))


def _blocks(table: pd.DataFrame) -> Iterator[tuple[date, pd.DataFrame]]:
    stamps = pd.to_datetime(table["expiry"])
    for expiry in _expiry_dates(table):
        yield expiry, table[stamps == pd.Timestamp(expiry)]


def _expiry_times(expiry: date, ts: datetime, calendar: "Calendar") -> tuple[date, float, float] | None:
    """`(last_session, T_E, tt_E)` of an expiry - the ONE place `close(last_session(E))` is spelt. `None` once the expiry's
    last session has closed (`T_E <= 0`)."""
    last = calendar.prev_or_same_session(expiry)
    close = calendar.open_close(last)[1]
    tau = year_fraction(ts, close)
    if tau <= 0.0:
        return None
    return last, tau, trading_time(calendar, ts, close)


def _round_cents(value: float) -> int:
    """Round half up to the cent - the project's one money rounding for derived price levels."""
    return int(math.floor(value + 0.5))


def _single_fwd(block: pd.DataFrame) -> int | None:
    if "fwd" not in block.columns:
        return None
    values = {int(v) for v in pd.Series(block["fwd"]).dropna().unique()}
    if len(values) != 1:
        return None
    fwd = values.pop()
    return fwd if fwd > 0 else None


def _two_sided(block: pd.DataFrame) -> FloatArray:
    bid = pd.Series(block["bid"]).to_numpy(dtype=np.float64)
    ask = pd.Series(block["ask"]).to_numpy(dtype=np.float64)
    mask: FloatArray = np.where((bid > 0.0) & (ask > bid), 1.0, 0.0)
    return mask


# ======================================================================================================================
# 5.2 Parity forwards and the parity spot
# ======================================================================================================================


def parity_forwards(table: pd.DataFrame, rate: float, ts: datetime, calendar: "Calendar") -> dict[date, int]:
    """Put-call parity forward per listed expiry, in cents (5.2).

    Among strikes where the call AND the put are `valid()` (two-sided), take the `MIN_PARITY_STRIKES` strikes with the
    smallest `|C_mid - P_mid|` - the strikes closest to the money, where the American early-exercise premium is smallest -
    and set `F_k = K + exp(r * T_E) * (C_mid - P_mid)`; `F_E = median(F_k)`, rounded to the cent. An expiry with fewer than
    three such pairs gets NO forward (its rows keep `iv = NaN` and `fwd = 0` in `enrich`).

    `T_E = year_fraction(ts, close(last_session(E)))`: a Saturday-dated monthly discounts to the FRIDAY close and a
    Good-Friday week's monthly to the THURSDAY close. American early exercise is ignored (stated approximation, 5.2).
    """
    _as_utc("ts", ts)
    for column in ("expiry", "right", "strike_milli", "bid", "ask"):
        if column not in table.columns:
            raise DataError(f"parity_forwards needs a `{column}` column")
    out: dict[date, int] = {}
    for expiry, block in _blocks(table):
        times = _expiry_times(expiry, ts, calendar)
        if times is None:
            continue
        _, tau, _ = times
        two_sided = _two_sided(block) > 0.0
        right = pd.Series(block["right"]).astype(str).to_numpy()
        strike = pd.Series(block["strike_milli"]).to_numpy(dtype=np.int64)
        mid2 = pd.Series(block["bid"]).to_numpy(dtype=np.float64) + pd.Series(block["ask"]).to_numpy(dtype=np.float64)
        calls = {int(s): float(m) for s, m, r, ok in zip(strike, mid2, right, two_sided, strict=True) if ok and r == "C"}
        puts = {int(s): float(m) for s, m, r, ok in zip(strike, mid2, right, two_sided, strict=True) if ok and r == "P"}
        pairs = sorted(
            ((abs(calls[s] - puts[s]), s) for s in calls.keys() & puts.keys()),
            key=lambda item: (item[0], item[1]),
        )
        if len(pairs) < MIN_PARITY_STRIKES:
            continue
        carry = math.exp(rate * tau)
        implied = [
            s / _MILLI_PER_CENT + carry * (calls[s] - puts[s]) / 2.0 for _, s in pairs[:MIN_PARITY_STRIKES]
        ]  # (C_mid - P_mid) = (mid2_c - mid2_p) / 2
        fwd = _round_cents(float(np.median(np.asarray(implied, dtype=np.float64))))
        if fwd > 0:
            out[expiry] = fwd
    return out


def parity_spot(
    forwards: Mapping[date, int],
    rate: float,
    ts: datetime,
    dividends: Sequence[ScheduledEvent],
    *,
    calendar: "Calendar",
    session: date | None = None,
    covered: bool = False,
) -> tuple[int, bool, date]:
    """The parity-implied spot of a snapshot: `(spot_cents, div_unmodelled, front_expiry)` (5.2).

    `S = F_E1 * exp(-r * T_E1) + sum(div_j * exp(-r * t_j))` over VERIFIED `ex_dividend` events with
    `session < ex_date <= last_session(E1)`, where `E1` is the nearest expiry with `dte >= 2` that has a forward.
    `div_unmodelled` is `not covered`: it is True whenever the verified ex-dividend table does not cover the window, i.e.
    whenever the spot could not include a verified dividend PV - never a claim that there were no dividends.

    `DataUnavailable` when no expiry qualifies. The 5.2 cross-check `S'` is the caller's job (`data derive` calls this again
    without `front_expiry` in `forwards` and raises `parity_basis_suspect` when `|S / S' - 1| > 15 bp`), which is exactly why
    the front expiry is returned.

    DESIGN 3.7 prints this signature without `calendar` / `session`; both are needed to reach `close(last_session(E))` and
    to place the dividend window, and are therefore keyword-only additions (`session` defaults to the UTC date of `ts`).
    """
    _as_utc("ts", ts)
    today = _session_of(ts, session, calendar)
    ranked: list[tuple[date, date]] = []
    for expiry, fwd in forwards.items():
        if fwd <= 0:
            continue
        last = calendar.prev_or_same_session(expiry)
        if (last - today).days >= FRONT_EXPIRY_MIN_DTE:
            ranked.append((last, expiry))
    if not ranked:
        raise DataUnavailable(f"no expiry with a forward and dte >= {FRONT_EXPIRY_MIN_DTE} at {today.isoformat()}")
    ranked.sort()
    last_session, front = ranked[0]
    tau = year_fraction(ts, calendar.open_close(last_session)[1])
    value = forwards[front] * math.exp(-rate * tau)
    for event in dividends:
        if event.kind != "ex_dividend" or event.cancelled or event.amount_cents is None:
            continue
        if event.knowable_at > ts or not (today < event.event_date <= last_session):
            continue
        t_j = year_fraction(ts, calendar.open_close(calendar.prev_or_same_session(event.event_date))[1])
        if t_j < 0.0:
            continue
        value += event.amount_cents * math.exp(-rate * t_j)
    return _round_cents(value), not covered, front


# ======================================================================================================================
# 5.3 Enrichment: own IV, delta, vega, the forward, last_session and dte
# ======================================================================================================================


def enrich(
    table: pd.DataFrame,
    forwards: Mapping[date, int],
    rate: float,
    ts: datetime,
    calendar: "Calendar",
    *,
    session: date | None = None,
) -> pd.DataFrame:
    """Add `last_session`, `dte`, `fwd`, `iv`, `delta`, `vega` to a raw chain table (5.3, D21).

    Own IV only: the vectorised Black-76 bisection on the row's own parity forward, from the MID of two-sided quotes,
    accepted iff `IV_MIN < iv < IV_MAX`; `delta` and `vega` follow from that IV (NaN wherever the IV is NaN). One solve per
    snapshot, whole-array - no per-row Python solver ever runs inside a backtest (1.1 performance budget).

    Rows of an expiry with no parity forward keep `fwd = 0` and NaN greeks (5.2); the `|ln(K / fwd)| <= data.moneyness_window`
    filter of 2.2 - applied by `data derive`, not here - drops them by construction. `enrich` never filters rows.
    """
    _as_utc("ts", ts)
    for column in ("expiry", "right", "strike_milli", "bid", "ask"):
        if column not in table.columns:
            raise DataError(f"enrich needs a `{column}` column")
    today = _session_of(ts, session, calendar)
    out = table.copy().reset_index(drop=True)
    out["expiry"] = pd.to_datetime(out["expiry"])
    n = len(out)
    last_session = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    dte = np.zeros(n, dtype=np.int64)
    fwd_c = np.zeros(n, dtype=np.int64)
    iv = np.full(n, np.nan, dtype=np.float64)
    delta = np.full(n, np.nan, dtype=np.float64)
    vega = np.full(n, np.nan, dtype=np.float64)

    if n:
        expiry_all = out["expiry"].to_numpy().astype("datetime64[ns]")
        strike_all = out["strike_milli"].to_numpy(dtype=np.float64) / _MILLI_PER_CENT
        bid_all = out["bid"].to_numpy(dtype=np.float64)
        ask_all = out["ask"].to_numpy(dtype=np.float64)
        call_all = out["right"].astype(str).to_numpy() == "C"
        for expiry in _expiry_dates(out):
            rows = expiry_all == np.datetime64(expiry, "ns")
            last = calendar.prev_or_same_session(expiry)
            last_session[rows] = np.datetime64(last, "ns")
            dte[rows] = (last - today).days
            forward = forwards.get(expiry, 0)
            tau = year_fraction(ts, calendar.open_close(last)[1])
            if forward <= 0 or tau <= 0.0:
                continue
            fwd_c[rows] = forward
            discount = math.exp(-rate * tau)
            bid, ask, strike, is_call = bid_all[rows], ask_all[rows], strike_all[rows], call_all[rows]
            two_sided = (bid > 0.0) & (ask > bid)
            solved = bs.b76_implied_vol((bid + ask) / 2.0, float(forward), strike, tau, discount, is_call)
            solved = np.where(two_sided & (solved > IV_MIN) & (solved < IV_MAX), solved, np.nan)
            iv[rows] = solved
            delta[rows] = bs.b76_delta(float(forward), strike, tau, solved, discount, is_call)
            vega[rows] = bs.b76_vega(float(forward), strike, tau, solved, discount)

    out["last_session"] = pd.Series(last_session, index=out.index)
    out["dte"] = pd.Series(dte, index=out.index, dtype="int64")
    out["fwd"] = pd.Series(fwd_c, index=out.index, dtype="int64")
    out["iv"] = pd.Series(iv, index=out.index, dtype="float64")
    out["delta"] = pd.Series(delta, index=out.index, dtype="float64")
    out["vega"] = pd.Series(vega, index=out.index, dtype="float64")
    return out


# ======================================================================================================================
# 6.4 step 1: the smile fit
# ======================================================================================================================


def two_strike_atm_iv(table: pd.DataFrame, expiry: date, fwd_c: int) -> float | None:
    """The two-strike ATM IV of 5.3: the strikes `K1 <= F < K2` bracketing the forward, per strike the OTM side's IV (put
    below `F`, call above; the other side if missing), linear in STRIKE to `F`. `None` when either side is unusable.

    A level that rests on two mid quotes is fragile, which is why it is only the fallback for `atm_iv` - but it is always
    computed and stored as the QC twin of the fitted value (5.3 `atm_iv_divergence`).
    """
    if fwd_c <= 0:
        return None
    block = table[pd.to_datetime(table["expiry"]) == pd.Timestamp(expiry)]
    if block.empty or "iv" not in block.columns:
        return None
    right = pd.Series(block["right"]).astype(str).to_numpy()
    strike = pd.Series(block["strike_milli"]).to_numpy(dtype=np.int64)
    values = pd.Series(block["iv"]).to_numpy(dtype=np.float64)
    calls: dict[int, float] = {}
    puts: dict[int, float] = {}
    for s, r, v in zip(strike, right, values, strict=True):
        if math.isnan(v):
            continue
        (calls if r == "C" else puts)[int(s)] = float(v)
    strikes = sorted(calls.keys() | puts.keys())
    below = [s for s in strikes if s / _MILLI_PER_CENT <= fwd_c]
    above = [s for s in strikes if s / _MILLI_PER_CENT > fwd_c]
    if not below or not above:
        return None
    k1, k2 = max(below), min(above)
    iv1 = puts.get(k1, calls.get(k1))  # below the forward the put is the OTM side
    iv2 = calls.get(k2, puts.get(k2))  # above it the call is
    if iv1 is None or iv2 is None:
        return None
    span = (k2 - k1) / _MILLI_PER_CENT
    if span <= 0.0:
        return None
    weight = (fwd_c - k1 / _MILLI_PER_CENT) / span
    return float(iv1 + weight * (iv2 - iv1))


def fit_smile(table: pd.DataFrame, expiry: date, ts: datetime, calendar: "Calendar") -> SmileFit | None:
    """Weighted least squares of TOTAL VARIANCE on `(1, k, k^2)`, `k = ln(K / F_E)` (6.4 step 1).

    Points: OTM two-sided quotes with a solved IV and a vega (puts `K < F`, calls `K >= F`) inside
    `|k| <= 3 * atm_iv * sqrt(T_E)`; weights `vega / max(spread, 1 cent)`. The fit needs `MIN_FIT_POINTS` points, full rank
    and `w(k) > 0` across the whole fitted range - otherwise there is NO fit and the caller falls back to a flat smile at the
    node's ATM total variance, so the node set never changes.

    Fitting `w = iv^2 * T_E` rather than `iv` is what makes the interpolation of 6.4 step 2 linear in the right quantity,
    and a fit over many strikes is what makes the implied reference robust to per-quote noise (G2).
    """
    _as_utc("ts", ts)
    block = table[pd.to_datetime(table["expiry"]) == pd.Timestamp(expiry)]
    if block.empty:
        return None
    for column in ("right", "strike_milli", "bid", "ask", "iv", "vega"):
        if column not in block.columns:
            raise DataError(f"fit_smile needs a `{column}` column")
    forward = _single_fwd(block)
    times = _expiry_times(expiry, ts, calendar)
    if forward is None or times is None:
        return None
    last, tau, tt = times

    strike = pd.Series(block["strike_milli"]).to_numpy(dtype=np.float64) / _MILLI_PER_CENT
    bid = pd.Series(block["bid"]).to_numpy(dtype=np.float64)
    ask = pd.Series(block["ask"]).to_numpy(dtype=np.float64)
    iv = pd.Series(block["iv"]).to_numpy(dtype=np.float64)
    vega = pd.Series(block["vega"]).to_numpy(dtype=np.float64)
    is_call = pd.Series(block["right"]).astype(str).to_numpy() == "C"

    k = np.log(strike / float(forward))
    usable = (bid > 0.0) & (ask > bid) & np.isfinite(iv) & np.isfinite(vega) & (vega > 0.0) & (iv > 0.0)
    otm = np.where(is_call, k >= 0.0, k < 0.0)
    atm_guess = two_strike_atm_iv(block, expiry, forward)
    if atm_guess is None or atm_guess <= 0.0:
        candidates = iv[usable]
        if candidates.size == 0:
            return None
        atm_guess = float(np.median(candidates))
    half_width = FIT_K_HALF_WIDTH_MULT * atm_guess * math.sqrt(tau)
    chosen = usable & otm & (np.abs(k) <= half_width)
    if int(chosen.sum()) < MIN_FIT_POINTS:
        return None

    kk = k[chosen]
    w = iv[chosen] ** 2 * tau
    weight = vega[chosen] / np.maximum(ask[chosen] - bid[chosen], _MIN_SPREAD_CENTS)
    root = np.sqrt(weight)
    design = np.column_stack((np.ones_like(kk), kk, kk * kk)) * root[:, None]
    coefficients, _residuals, rank, _sv = np.linalg.lstsq(design, w * root, rcond=None)
    if int(rank) < 3:
        return None
    a, b, c = (float(coefficients[0]), float(coefficients[1]), float(coefficients[2]))
    k_lo, k_hi = float(kk.min()), float(kk.max())
    if not _positive_on(a, b, c, k_lo, k_hi):
        return None
    return SmileFit(
        expiry=expiry,
        last_session=last,
        tau_years=tau,
        tt_sessions=tt,
        fwd=forward,
        a=a,
        b=b,
        c=c,
        k_lo=k_lo,
        k_hi=k_hi,
        n_points=int(chosen.sum()),
    )


def fit_smiles(
    table: pd.DataFrame, ts: datetime, calendar: "Calendar", expiries: Sequence[date] | None = None
) -> dict[date, SmileFit]:
    """`fit_smile` for every listed expiry that admits one (expiries without a fit are simply absent from the mapping)."""
    wanted = _expiry_dates(table) if expiries is None else tuple(expiries)
    out: dict[date, SmileFit] = {}
    for expiry in wanted:
        fit = fit_smile(table, expiry, ts, calendar)
        if fit is not None:
            out[expiry] = fit
    return out


def _positive_on(a: float, b: float, c: float, k_lo: float, k_hi: float) -> bool:
    """Is `w(k) = a + b k + c k^2` strictly positive across `[k_lo, k_hi]`? (A fitted total variance that goes negative
    anywhere on its own range is not a surface; 6.4 rejects the fit.)"""
    if not all(math.isfinite(v) for v in (a, b, c, k_lo, k_hi)):
        return False
    lowest = min(a + b * k_lo + c * k_lo * k_lo, a + b * k_hi + c * k_hi * k_hi)
    if c > 0.0:
        vertex = -b / (2.0 * c)
        if k_lo <= vertex <= k_hi:
            lowest = min(lowest, a + b * vertex + c * vertex * vertex)
    return lowest > 0.0


# ======================================================================================================================
# 5.3 The ATM term structure - THE node set shared by expected moves and implied probabilities
# ======================================================================================================================


@dataclass(frozen=True)
class SurfaceNode:
    """One expiry of the term structure: its times, its ATM IV (fitted and two-strike) and its smile fit, if any.

    `atm_term()` renders these as the 3.7 tuples; `implied_prob_above` needs the fits as well, so both read the same nodes -
    that is the mechanical guarantee behind "the threshold and its reference share one variance rule" (6.4).
    """

    expiry: date
    last_session: date
    dte: int
    tau_years: float
    tt_sessions: float
    atm_iv: float  # the fitted smile at k = 0 when a fit exists, else the two-strike value
    atm_iv_2s: float  # always the two-strike value (QC twin); equal to atm_iv when it is unavailable
    fwd: Cents
    fit: SmileFit | None

    @property
    def w_atm(self) -> float:
        """The node's ATM TOTAL variance `iv^2 * tau` at the quantised (basis-point) IV that `atm_term` publishes."""
        return (round(self.atm_iv * 1e4) / 1e4) ** 2 * self.tau_years


def surface_nodes(
    table: pd.DataFrame,
    fits: Mapping[date, SmileFit],
    ts: datetime,
    calendar: "Calendar",
    *,
    session: date | None = None,
) -> list[SurfaceNode]:
    """Every expiry with `dte >= 1` that has a forward AND an ATM IV, ordered by `(last_session, expiry)` (3.7, 5.3).

    No `dte >= 7` filter here: that filter belongs to `const_maturity_iv` (iv30 / iv90) and `skew25` alone. The expected
    moves of 5.3 and the implied probabilities of 6.4 use ALL of these nodes.
    """
    _as_utc("ts", ts)
    today = _session_of(ts, session, calendar)
    nodes: list[SurfaceNode] = []
    for expiry, block in _blocks(table):
        forward = _single_fwd(block)
        times = _expiry_times(expiry, ts, calendar)
        if forward is None or times is None:
            continue
        last, tau, tt = times
        dte = (last - today).days
        if dte < 1:
            continue
        fit = fits.get(expiry)
        fitted = math.sqrt(fit.w(0.0) / tau) if fit is not None and fit.w(0.0) > 0.0 else None
        two_strike = two_strike_atm_iv(block, expiry, forward)
        atm = fitted if fitted is not None else two_strike
        if atm is None or atm <= 0.0:
            continue
        twin = two_strike if two_strike is not None and two_strike > 0.0 else atm
        nodes.append(
            SurfaceNode(
                expiry=expiry,
                last_session=last,
                dte=dte,
                tau_years=tau,
                tt_sessions=tt,
                atm_iv=atm,
                atm_iv_2s=twin,
                fwd=forward,
                fit=fit,
            )
        )
    nodes.sort(key=lambda node: (node.last_session, node.expiry))
    return nodes


def term_of(nodes: Sequence[SurfaceNode]) -> list[TermNode]:
    """`SurfaceNode`s as the 3.7 `atm_term` tuples `(tau_years, tt_sessions, atm_iv_bp, atm_iv_2s_bp, fwd_c)`."""
    return [(n.tau_years, n.tt_sessions, round(n.atm_iv * 1e4), round(n.atm_iv_2s * 1e4), int(n.fwd)) for n in nodes]


def atm_term(
    table: pd.DataFrame,
    fits: Mapping[date, SmileFit],
    ts: datetime,
    calendar: "Calendar",
    *,
    session: date | None = None,
) -> list[TermNode]:
    """`[(tau_years, tt_sessions, atm_iv_bp, atm_iv_2s_bp, fwd_c)]` for every node of `surface_nodes` (3.7).

    `atm_iv_bp` is the fitted smile at `k = 0` when a fit exists, else the two-strike interpolation; `atm_iv_2s_bp` is
    always the two-strike value (5.3's QC twin, stored as `daily.iv30_2s_bp`).
    """
    return term_of(surface_nodes(table, fits, ts, calendar, session=session))


def chain_nodes(chain: ChainSnapshot, calendar: "Calendar") -> list[SurfaceNode]:
    """The nodes of an enriched `ChainSnapshot`: fit every expiry, then build the term. THE entry point for consumers that
    need both the term and the smiles (`data derive`, `implied_prob_above`)."""
    fits = fit_smiles(chain.table, chain.ts, calendar)
    return surface_nodes(chain.table, fits, chain.ts, calendar, session=chain.key.session)


# ======================================================================================================================
# Total variance in trading time (V13) and constant-maturity IV in calendar time
# ======================================================================================================================


def _sorted_term(term: Sequence[TermNode]) -> list[TermNode]:
    """Nodes by trading time, one per distinct `tt` (a Saturday-dated monthly and a Friday weekly of the same week share a
    last session; the first in `(last_session, expiry)` order wins)."""
    out: list[TermNode] = []
    seen: set[float] = set()
    for node in sorted(term, key=lambda n: n[1]):
        tt = float(node[1])
        if tt <= 0.0 or tt in seen:
            continue
        seen.add(tt)
        out.append(node)
    return out


def total_variance_at(term: Sequence[TermNode], tt: float) -> tuple[float, str]:
    """Total variance `w` at TRADING time `tt` (sessions) and its quality (3.7, V13).

    Node values are `w_i = iv_i^2 * tau_i`; `w` is LINEAR IN `tt` between the bracketing nodes ("interpolated"), and
    `w_1 * tt / tt_1` below the first node / `w_n * tt / tt_n` beyond the last ("extrapolated"). Every node with `dte >= 1`
    participates - no `dte >= 7` filter - so where an expiry's last session IS the horizon, `w` is exactly the market's
    total variance to that close. Allocating in trading time is what stops a Friday's one-session expected move from being
    inflated by sqrt(3) for the weekend's calendar days.
    """
    nodes = _sorted_term(term)
    if not nodes:
        raise DataUnavailable("no ATM term nodes: total variance is undefined")
    if not (tt > 0.0 and math.isfinite(tt)):
        raise ValueError(f"tt must be a positive number of sessions, got {tt!r}")
    variance = [(node[0] > 0.0) and (node[2] / 1e4) ** 2 * node[0] or 0.0 for node in nodes]
    times = [float(node[1]) for node in nodes]
    if tt <= times[0]:
        if tt == times[0]:
            return variance[0], _INTERPOLATED
        return variance[0] * tt / times[0], _EXTRAPOLATED
    if tt >= times[-1]:
        if tt == times[-1]:
            return variance[-1], _INTERPOLATED
        return variance[-1] * tt / times[-1], _EXTRAPOLATED
    upper = next(i for i, value in enumerate(times) if value >= tt)
    lower = upper - 1
    span = times[upper] - times[lower]
    theta = (tt - times[lower]) / span
    return variance[lower] + theta * (variance[upper] - variance[lower]), _INTERPOLATED


def const_maturity_iv(term: Sequence[TermNode], days: int, *, min_dte: int = 7) -> tuple[float, str]:
    """Constant-maturity ATM IV (iv30 / iv90 only): total-variance interpolation in CALENDAR time over the nodes with
    `dte >= min_dte`; one-sided gives the nearest node, flagged extrapolated (3.7, 5.3).

    Calendar time, not trading time: iv30 / iv90 are annualised IV LEVELS, and every level in this project is annualised on
    calendar / 365 (Conventions).
    """
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days}")
    # tau * 365 is the node's dte to the half-day (a `dec` snapshot sits 25 minutes before the close), hence the 0.5 slack
    nodes = [(float(tau), int(bp)) for tau, _tt, bp, _twin, _fwd in term if tau > 0.0 and bp > 0 and tau * 365.0 >= min_dte - 0.5]
    if not nodes:
        raise DataUnavailable(f"no ATM term node with dte >= {min_dte}: constant-maturity IV is undefined")
    target = days / 365.0
    below = [n for n in nodes if n[0] <= target]
    above = [n for n in nodes if n[0] >= target]
    if below and above:
        tau_lo, bp_lo = max(below)
        tau_hi, bp_hi = min(above)
        if tau_hi == tau_lo:
            return bp_lo / 1e4, _INTERPOLATED
        w_lo = (bp_lo / 1e4) ** 2 * tau_lo
        w_hi = (bp_hi / 1e4) ** 2 * tau_hi
        w = w_lo + (w_hi - w_lo) * (target - tau_lo) / (tau_hi - tau_lo)
        return math.sqrt(max(w, 0.0) / target), _INTERPOLATED
    nearest = min(nodes, key=lambda n: abs(n[0] - target))
    return nearest[1] / 1e4, _EXTRAPOLATED


# ======================================================================================================================
# 6.4 The skew-consistent implied digital
# ======================================================================================================================


def _node_smile(node: SurfaceNode, k: float) -> tuple[float, float]:
    """`(w(k), dw/dk(k))` of one node: its fitted smile, or a FLAT smile at its ATM total variance when it has no fit -
    so a node without a fit still contributes and the node set never differs from `atm_term`'s (6.4 step 1)."""
    if node.fit is None:
        return node.w_atm, 0.0
    return node.fit.w(k), node.fit.dw_dk(k)


def _scaled_smile(node: SurfaceNode, k: float, rho: float) -> tuple[float, float]:
    """The standardised-moneyness rescaling of 6.4 step 2: evaluate a node's smile at `k' = k / sqrt(rho)` and scale,
    `w = rho * w_i(k')`, `dw/dk = sqrt(rho) * w_i'(k')`, with `rho` the TRADING-time ratio. Used below the first node (every
    1-session question before daily expiries existed) and beyond the last."""
    root = math.sqrt(rho)
    w, dw = _node_smile(node, k / root)
    return rho * w, root * dw


def _blend_smile(nodes: Sequence[SurfaceNode], tt: float, k: float) -> tuple[float, float, str]:
    """`(w, dw/dk, quality)` at trading time `tt` and log-moneyness `k`, interpolating LINEARLY IN TRADING TIME between the
    bracketing nodes - the same allocation rule `total_variance_at` uses for the expected move (V13)."""
    times = [node.tt_sessions for node in nodes]
    if tt <= times[0]:
        if tt == times[0]:
            w, dw = _node_smile(nodes[0], k)
            return w, dw, _INTERPOLATED
        w, dw = _scaled_smile(nodes[0], k, tt / times[0])
        return w, dw, _EXTRAPOLATED
    if tt >= times[-1]:
        if tt == times[-1]:
            w, dw = _node_smile(nodes[-1], k)
            return w, dw, _INTERPOLATED
        w, dw = _scaled_smile(nodes[-1], k, tt / times[-1])
        return w, dw, _EXTRAPOLATED
    upper = next(i for i, value in enumerate(times) if value >= tt)
    lower = upper - 1
    if times[upper] == tt:  # the node's last session IS the resolve session: the market's own variance to that close
        w, dw = _node_smile(nodes[upper], k)
        return w, dw, _INTERPOLATED
    theta = (tt - times[lower]) / (times[upper] - times[lower])
    w_lo, dw_lo = _node_smile(nodes[lower], k)
    w_hi, dw_hi = _node_smile(nodes[upper], k)
    return w_lo + theta * (w_hi - w_lo), dw_lo + theta * (dw_hi - dw_lo), _INTERPOLATED


def _unique_nodes(nodes: Sequence[SurfaceNode]) -> list[SurfaceNode]:
    out: list[SurfaceNode] = []
    seen: set[float] = set()
    for node in sorted(nodes, key=lambda n: n.tt_sessions):
        if node.tt_sessions <= 0.0 or node.tt_sessions in seen:
            continue
        seen.add(node.tt_sessions)
        out.append(node)
    return out


def implied_prob_above(
    chain: ChainSnapshot,
    strike_c: int,
    resolve_ts: datetime,
    calendar: "Calendar",
    *,
    force_plain: bool = False,
    nodes: Sequence[SurfaceNode] | None = None,
) -> tuple[float, str, str] | None:
    """`P(S_T > K)` implied by the option surface, with its method and quality (6.4). `None` when no usable expiry exists.

    1. Nodes = `chain_nodes`: every expiry with `dte >= 1` that has a forward and an ATM IV - the SAME node set the expected
       move of 5.3 uses. Each node contributes its fitted smile, or a flat smile at its ATM total variance.
    2. Total variance and its strike slope are allocated to the horizon in TRADING time (`k' = k / sqrt(rho)` below the
       first node and beyond the last; linear in `tt` in between; the node itself when its last session IS the resolve
       session). Calendar `tau` is used only for the forward and for annualising - the digital depends on `w` and `dw/dk`.
    3. `sigma = sqrt(w / tau)`, `dsigma/dK = (dw/dk) / (2 sigma tau K)`, `PA = bs.prob_above(F_tau, K, tau, sigma, dsigma/dK)`:
       the SKEW-CONSISTENT digital. The plain `N(d2)` omits the smile-slope term and is wrong by several points under index
       skew, so it is only the labelled fallback `nd2_plain`, used when no node has a fit - or when the caller detects that
       the three `PA` values of a horizon are not monotone in `K` and asks for it with `force_plain` (that check needs all
       three thresholds and therefore lives in `outcomes.py`).

    `nodes` lets a caller that already built the term (the enricher, the outcome builder pricing three strikes) pass it in;
    the result is identical, it just skips re-fitting the smiles.
    """
    _as_utc("resolve_ts", resolve_ts)
    if strike_c <= 0:
        raise ValueError(f"strike_c must be positive cents, got {strike_c}")
    built = chain_nodes(chain, calendar) if nodes is None else nodes
    usable = _unique_nodes(built)
    if not usable:
        return None
    tau = year_fraction(chain.ts, resolve_ts)
    tt = trading_time(calendar, chain.ts, resolve_ts)
    if tau <= 0.0 or tt <= 0.0 or chain.spot <= 0:
        return None
    forward = chain.spot * math.exp(chain.rate * tau)
    if forward <= 0.0:
        return None
    k = math.log(strike_c / forward)
    if force_plain or all(node.fit is None for node in usable):
        w, quality = total_variance_at(term_of(usable), tt)
        slope = 0.0
        method = _ND2_PLAIN
    else:
        w, slope, quality = _blend_smile(usable, tt, k)
        method = _SMILE_DIGITAL
    if not (w > 0.0 and math.isfinite(w)) or not math.isfinite(slope):
        return None
    sigma = math.sqrt(w / tau)
    dsigma_dk = slope / (2.0 * sigma * tau * strike_c)
    return bs.prob_above(forward, float(strike_c), tau, sigma, dsigma_dk), method, quality
