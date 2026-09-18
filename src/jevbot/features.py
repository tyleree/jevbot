"""Numeric features of one underlying from a `MarketView` (DESIGN.md 5.3). Pure: no clock, no RNG, no IO, no network.

`compute_features(view, underlying, hold_sessions)` returns a `FeatureSet` whose every member is `None` when the inputs
for it are missing; `StateBuilder.entry` refuses to build a request when a REQUIRED feature is None (5.3, `required_ok`).

Two facts shape the whole module:

* **Every look-back window is indexed by SESSION.** Closes come from `view.closes()` (one per session, `eod` rows only)
  and the surface series from `view.daily()` (one row per session by the designated-slot rule of 5.4), so "252 sessions
  ago" means the same thing whether the archive holds one slot per session (mirror) or three (recorder / paper).
* **Total variance is allocated in TRADING time** (V13, 5.3): an expected move over `h` sessions is
  `sqrt(total_variance_at(atm_term, trading_time(as_of, close(next_session(D, h)))))`, so a Friday's one-session move is
  not inflated by the weekend's calendar days. The same node set and rule serve the implied reference of 6.4.

`total_variance_at` is the 3.7 formula. `data/surface.py` (WP01) owns the same formula for the data layer; wave-1
packages may not import each other (section 16), so it is spelt here too - node-exact, linear in trading time between
the bracketing nodes, proportional below the first and beyond the last node. The two are pinned against the same
hand-computed numbers in their own tests.

Today's high, low, close and volume are never read: `view.bars()` returns completed sessions only and
`view.today_open_ratio()` is the single use of today's bar row (its `open`, knowable at open + 60 s).
"""

import itertools
import json
import math
from collections.abc import Sequence
from typing import Any, Final

import msgspec
import numpy as np
import pandas as pd

from jevbot.cal import trading_time
from jevbot.config import DataConfig
from jevbot.errors import DataUnavailable, InvariantError
from jevbot.protocols import MarketView
from jevbot.types import Bp, Cents

__all__ = [
    "ATM_NODE_FIELDS",
    "AtmNode",
    "FeatureSet",
    "compute_features",
    "expected_move",
    "parse_atm_term",
    "pctile",
    "total_variance_at",
]

# `[(tau_years, tt_sessions, atm_iv_bp, atm_iv_2s_bp, fwd_c), ...]` - the 13.2 `atm_term_json` payload (3.7 `atm_term`)
AtmNode = tuple[float, float, int, int, int]
ATM_NODE_FIELDS: Final = ("tau_years", "tt_sessions", "atm_iv_bp", "atm_iv_2s_bp", "fwd_c")

SESSIONS_PER_YEAR: Final = 252  # ONLY to annualise close-to-close realised vol for bucketing ratios of like quantities (5.3)
YEAR_SESSIONS: Final = 252  # percentile / rank windows
CLOSES_LOOKBACK: Final = 260
DAILY_LOOKBACK: Final = 260
BARS_LOOKBACK: Final = 15  # 15 completed bars give 14 true ranges (each needs the previous close)
ATR_BARS: Final = 14
MIN_CLOSES: Final = 60  # 5.3: the underlying block needs >= 60 closes
BP_PER_UNIT: Final = 10_000.0
TENTHS_PER_UNIT: Final = 1000.0

_INTERPOLATED: Final = "interpolated"
_EXTRAPOLATED: Final = "extrapolated"


class FeatureSet(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """The 5.3 features of one underlying at one snapshot. `None` = unavailable (the bucket renders `"unavailable"`).

    Integer members are integers by nature: `ref` and `iv30_bp` are the raw material of `EntryFacts` / `EntryContext`
    (cents, basis points), `streak` is a signed session count and the three `em_*_tenths` are the integers that reach
    Jev AND the outcome resolver (6.4) - they are never re-derived from the floats.
    """

    underlying: str
    hold_sessions: int
    n_closes: int  # completed closes plus `ref`
    ref: Cents | None  # the decision-time reference price (`ChainSnapshot.spot`, 5.2)
    # trend / momentum / range (closes and completed bars only)
    ma20: float | None = None
    ma50: float | None = None
    ma20_prev: float | None = None
    rv20: float | None = None
    rv5: float | None = None
    sigma_d: float | None = None
    atr_pct: float | None = None
    atr: float | None = None
    trend_z: float | None = None
    dist_ma20_atr: float | None = None
    streak: int | None = None
    rv20_pctile: int | None = None
    rv_change: float | None = None
    gap_sigma: float | None = None
    move_sigma: float | None = None
    dd_52w: float | None = None
    # Cboe vol indices (newest knowable close = the prior session, D22); raw levels never enter the state (B6.1)
    vix_pctile: int | None = None
    vix_chg_1w: float | None = None
    vix_term: float | None = None
    vix_near: float | None = None
    vvix_pctile: int | None = None
    skewidx_pctile: int | None = None
    # own surface (5.3 / 5.4: one `daily` row per session)
    iv30: float | None = None
    iv30_bp: Bp | None = None
    iv90: float | None = None
    iv_rank: int | None = None
    iv_rv: float | None = None
    iv_chg_1w: float | None = None
    iv_term: float | None = None
    skew25: float | None = None
    skew_pctile: int | None = None
    # expected moves, total variance allocated in TRADING time (V13)
    em_1: float | None = None
    em_5: float | None = None
    em_hold: float | None = None
    em_1_tenths: int | None = None
    em_5_tenths: int | None = None
    em_hold_tenths: int | None = None
    iv_var_5: float | None = None  # implied TOTAL variance to the 5-session close, for `rv_gt_iv` (6.4)
    # the snapshot's own ATM term (13.2 `atm_term_json`), parsed once: the expected moves above and the manage state's
    # position distances (5.5 SHORT_DIST / BREAKEVEN) read the SAME nodes
    atm_term: tuple[AtmNode, ...] = ()
    # provenance-only facts
    iv_hist_proxy_pct: int = 0  # share of proxy-filled rows in the trailing 252-session IV history (5.4, V10)

    @property
    def trend_inputs_ok(self) -> bool:
        """The TREND_DIR rule needs `ref` and the three moving averages (5.5)."""
        return None not in (self.ref, self.ma20, self.ma50, self.ma20_prev)

    @property
    def required_ok(self) -> bool:
        """5.3 REQUIRED features: trend direction, `rv20`, `iv30`, `iv_rank`, `iv_rv`, `em_1`, `em_5`, `em_H`."""
        return self.trend_inputs_ok and None not in (
            self.rv20,
            self.iv30,
            self.iv30_bp,
            self.iv_rank,
            self.iv_rv,
            self.em_1_tenths,
            self.em_5_tenths,
            self.em_hold_tenths,
        )

    def missing_required(self) -> tuple[str, ...]:
        """The names of the required features that are unavailable (diagnostics for the `dq:insufficient` reason)."""
        missing = [name for name in ("ref", "ma20", "ma50", "ma20_prev") if getattr(self, name) is None]
        missing += [
            name
            for name in ("rv20", "iv30", "iv30_bp", "iv_rank", "iv_rv", "em_1_tenths", "em_5_tenths", "em_hold_tenths")
            if getattr(self, name) is None
        ]
        return tuple(missing)

    def raw(self) -> dict[str, str]:
        """Every feature rendered as a string, for `Provenance.raw_features` (audit only, never sent to Jev)."""
        return {name: repr(getattr(self, name)) for name in self.__struct_fields__ if name not in ("underlying", "atm_term")}


# ======================================================================================================================
# Pure helpers (no view)
# ======================================================================================================================


def pctile(x: float, hist: Sequence[float], *, min_len: int) -> int | None:
    """5.3: `round(100 * (count(hist <= x) - 1) / (len(hist) - 1))`; `hist` INCLUDES `x`. None below `min_len`."""
    n = len(hist)
    if n < max(2, min_len):
        return None
    at_or_below = sum(1 for value in hist if value <= x)
    return round(100.0 * (at_or_below - 1) / (n - 1))


def parse_atm_term(payload: object) -> tuple[AtmNode, ...]:
    """Decode the `atm_term_json` column (13.2) into 3.7 nodes, ordered by trading time. `()` when the column is null.

    A malformed payload is an `InvariantError`: the state must never be built on a half-read surface.
    """
    if payload is None or (isinstance(payload, float) and math.isnan(payload)):
        return ()
    if isinstance(payload, str):
        if not payload.strip():
            return ()
        try:
            decoded: Any = json.loads(payload)
        except ValueError as exc:
            raise InvariantError(f"features: atm_term_json is not JSON ({exc})") from None
    else:
        decoded = payload
    if not isinstance(decoded, list):
        raise InvariantError(f"features: atm_term_json must be a list of {len(ATM_NODE_FIELDS)}-element nodes")
    nodes: list[AtmNode] = []
    for raw in decoded:
        if not isinstance(raw, list | tuple) or len(raw) != len(ATM_NODE_FIELDS):
            raise InvariantError(f"features: atm_term node must have the fields {ATM_NODE_FIELDS}, got {raw!r}")
        tau, tt, iv_bp, iv2s_bp, fwd_c = raw
        nodes.append((float(tau), float(tt), int(iv_bp), int(iv2s_bp), int(fwd_c)))
    nodes.sort(key=lambda node: (node[1], node[0]))
    return tuple(nodes)


def total_variance_at(term: Sequence[AtmNode], tt: float) -> tuple[float, str] | None:
    """THE 3.7 rule: total variance `w` at TRADING time `tt` (sessions), with `"interpolated"` / `"extrapolated"`.

    Node values are `w_i = (atm_iv_bp / 1e4)^2 * tau_years_i`, i.e. the market's total variance to that expiry's close.
    Between two nodes `w` is LINEAR IN `tt`; below the first node it is `w_1 * tt / tt_1`, beyond the last `w_n * tt / tt_n`.
    Every node with `dte >= 1` is used - there is no `dte >= 7` filter here (that one belongs to `iv30` / `iv90` / `skew25`).
    `None` when no usable node exists; `tt <= 0` gives zero variance.
    """
    nodes: list[tuple[float, float]] = []
    seen: set[float] = set()
    for tau, tt_i, iv_bp, _iv2s_bp, _fwd in term:
        if tt_i <= 0.0 or tau <= 0.0 or iv_bp <= 0 or tt_i in seen:
            continue
        seen.add(tt_i)
        nodes.append((tt_i, (iv_bp / BP_PER_UNIT) ** 2 * tau))
    if not nodes:
        return None
    nodes.sort()
    if tt <= 0.0:
        return 0.0, _INTERPOLATED
    first_tt, first_w = nodes[0]
    if tt < first_tt:
        return first_w * tt / first_tt, _EXTRAPOLATED
    last_tt, last_w = nodes[-1]
    if tt > last_tt:
        return last_w * tt / last_tt, _EXTRAPOLATED
    for (tt_lo, w_lo), (tt_hi, w_hi) in itertools.pairwise(nodes):
        if tt_lo <= tt <= tt_hi:
            if tt_hi == tt_lo:
                return w_lo, _INTERPOLATED
            return w_lo + (w_hi - w_lo) * (tt - tt_lo) / (tt_hi - tt_lo), _INTERPOLATED
    return last_w, _INTERPOLATED  # tt == last_tt (a single node, or float bookkeeping at the top end)


def expected_move(term: Sequence[AtmNode], tt: float) -> tuple[float, int, str] | None:
    """`(em, em_tenths, quality)` at trading time `tt`: `em = sqrt(w)`, `em_tenths = max(1, round(1000 * em))` (5.3).

    The integer is what Jev sees AND what the outcome resolver turns into a price threshold (6.4): one rounding, once.
    """
    found = total_variance_at(term, tt)
    if found is None:
        return None
    w, quality = found
    if w < 0.0 or not math.isfinite(w):
        raise InvariantError(f"features: total variance must be finite and >= 0, got {w!r}")
    em = math.sqrt(w)
    return em, max(1, round(TENTHS_PER_UNIT * em)), quality


# ======================================================================================================================
# View readers (DataUnavailable => the feature is None; PitViolation is a BUG and propagates, INV-14)
# ======================================================================================================================


def _closes(view: MarketView, underlying: str) -> list[int]:
    try:
        series = view.closes(underlying, CLOSES_LOOKBACK)
    except DataUnavailable:
        return []
    return [int(value) for value in series.to_numpy()]


def _bars(view: MarketView, underlying: str) -> pd.DataFrame:
    try:
        return view.bars(underlying, BARS_LOOKBACK)
    except DataUnavailable:
        return pd.DataFrame(columns=["session", "open", "high", "low", "close"])


def _daily(view: MarketView, underlying: str) -> pd.DataFrame:
    try:
        return view.daily(underlying, DAILY_LOOKBACK)
    except DataUnavailable:
        return pd.DataFrame()


def _index(view: MarketView, name: str) -> list[float]:
    try:
        series = view.vol_index(name, YEAR_SESSIONS)
    except DataUnavailable:
        return []
    return [float(value) for value in series.to_numpy() if not pd.isna(value)]


def _column(frame: pd.DataFrame, name: str) -> list[float | None]:
    """One `daily` column as `float | None` per session row, oldest first."""
    if frame.empty or name not in frame.columns:
        return []
    return [None if pd.isna(value) else float(value) for value in frame[name].tolist()]


def _last(values: Sequence[float | None]) -> float | None:
    return values[-1] if values else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _window_pctile(values: Sequence[float | None], *, min_len: int) -> int | None:
    """Percentile rank of the LAST non-null value within the last `YEAR_SESSIONS` non-null values (5.3)."""
    x = _last(values)
    if x is None:
        return None
    hist = [value for value in values[-YEAR_SESSIONS:] if value is not None]
    return pctile(x, hist, min_len=min_len)


def _mean(values: Sequence[int]) -> float:
    return float(sum(values)) / len(values)


def _realised_vol(returns: "np.ndarray[Any, np.dtype[np.float64]]", n: int) -> float | None:
    if returns.size < n:
        return None
    window = returns[-n:]
    return float(math.sqrt(SESSIONS_PER_YEAR * float(np.mean(window * window))))


def _streak(closes: Sequence[int]) -> int | None:
    """Signed count of consecutive up / down close-to-close changes ending at `ref` (0 when the last change is zero)."""
    if len(closes) < 2:
        return None
    last = closes[-1] - closes[-2]
    if last == 0:
        return 0
    sign = 1 if last > 0 else -1
    count = 0
    for index in range(len(closes) - 1, 0, -1):
        step = closes[index] - closes[index - 1]
        if step == 0 or (step > 0) != (sign > 0):
            break
        count += 1
    return sign * count


def _atr_pct(bars: pd.DataFrame) -> float | None:
    """Mean true range over the last `ATR_BARS` COMPLETED bars, in ratio form (5.2 rule b: within-bar ratios only)."""
    if len(bars) < ATR_BARS + 1:
        return None
    highs = bars["high"].to_numpy(dtype=np.float64)
    lows = bars["low"].to_numpy(dtype=np.float64)
    closes = bars["close"].to_numpy(dtype=np.float64)
    ranges: list[float] = []
    for index in range(len(bars) - ATR_BARS, len(bars)):
        high, low, close, prev_close = highs[index], lows[index], closes[index], closes[index - 1]
        if not all(math.isfinite(value) for value in (high, low, close, prev_close)) or close <= 0.0:
            return None
        ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)) / close)
    return float(sum(ranges) / len(ranges))


def _resolve_tt(view: MarketView, horizon: int) -> float | None:
    """Trading time from `as_of` to the close of the session `horizon` sessions after this one (5.3, V13)."""
    if horizon < 1:
        return None
    resolve_session = view.calendar.next_session(view.session, horizon)
    resolve_ts = view.calendar.open_close(resolve_session)[1]
    return trading_time(view.calendar, view.as_of, resolve_ts)


def _proxy_pct(frame: pd.DataFrame) -> int:
    """Share (percent) of proxy-filled rows in the trailing 252-session IV history (5.4 / V10, `Provenance`)."""
    if frame.empty or "source" not in frame.columns:
        return 0
    window = frame["source"].tolist()[-YEAR_SESSIONS:]
    if not window:
        return 0
    return round(100.0 * sum(1 for source in window if source == "proxy") / len(window))


# ======================================================================================================================
# compute_features
# ======================================================================================================================


def compute_features(view: MarketView, underlying: str, hold_sessions: int, *, data: DataConfig | None = None) -> FeatureSet:
    """The 5.3 feature set of `underlying` at `view`'s snapshot. `hold_sessions` = `dte.hold_horizon_sessions`.

    `data` carries the three section-4 knobs the formulas name (`min_history_sessions`, `iv_rank_lo_pct`,
    `iv_rank_hi_pct`); it defaults to the shipped values so that the 5.3 signature stays callable as printed.
    """
    cfg = DataConfig() if data is None else data
    if hold_sessions < 1:
        raise InvariantError(f"features: hold_sessions must be >= 1, got {hold_sessions}")
    min_history = cfg.min_history_sessions

    try:
        ref: Cents | None = view.spot(underlying)
    except DataUnavailable:
        ref = None

    closes = _closes(view, underlying)
    series: list[int] = [*closes, ref] if ref is not None else []
    out: dict[str, Any] = {"underlying": underlying, "hold_sessions": hold_sessions, "ref": ref, "n_closes": len(series)}

    # --- underlying block (needs >= 60 closes) ------------------------------------------------------------------------
    if len(series) >= MIN_CLOSES:
        levels = np.asarray(series, dtype=np.float64)
        returns = np.diff(np.log(levels))
        ma20 = _mean(series[-20:])
        ma50 = _mean(series[-50:])
        ma20_prev = _mean(series[-25:-5])
        rv20 = _realised_vol(returns, 20)
        rv5 = _realised_vol(returns, 5)
        sigma_d = None if rv20 is None else rv20 / math.sqrt(SESSIONS_PER_YEAR)
        atr_pct = _atr_pct(_bars(view, underlying))
        atr = None if atr_pct is None or ref is None else atr_pct * ref
        trend_z = None
        if sigma_d is not None and sigma_d > 0.0 and series[-21] > 0:
            trend_z = abs(math.log(series[-1] / series[-21])) / (sigma_d * math.sqrt(20.0))
        open_ratio = view.today_open_ratio(underlying)
        high_52w = max(series[-YEAR_SESSIONS:])
        out.update(
            ma20=ma20,
            ma50=ma50,
            ma20_prev=ma20_prev,
            rv20=rv20,
            rv5=rv5,
            sigma_d=sigma_d,
            atr_pct=atr_pct,
            atr=atr,
            trend_z=trend_z,
            dist_ma20_atr=None if atr is None or atr <= 0.0 or ref is None else (ref - ma20) / atr,
            streak=_streak(series),
            rv_change=_ratio(rv5, rv20),
            gap_sigma=None if open_ratio is None or not sigma_d else (open_ratio - 1.0) / sigma_d,
            move_sigma=None if not sigma_d else (series[-1] / series[-2] - 1.0) / sigma_d,
            dd_52w=None if high_52w <= 0 else series[-1] / high_52w - 1.0,
        )

    # --- Cboe vol indices ---------------------------------------------------------------------------------------------
    vix, vix3m, vix9d = _index(view, "VIX"), _index(view, "VIX3M"), _index(view, "VIX9D")
    vvix, skew_index = _index(view, "VVIX"), _index(view, "SKEW")
    out.update(
        vix_pctile=None if not vix else pctile(vix[-1], vix[-YEAR_SESSIONS:], min_len=min_history),
        vix_chg_1w=None if len(vix) < 6 or vix[-6] == 0.0 else vix[-1] / vix[-6] - 1.0,
        vix_term=None if not vix or not vix3m else _ratio(vix[-1], vix3m[-1]),
        vix_near=None if not vix or not vix9d else _ratio(vix9d[-1], vix[-1]),
        vvix_pctile=None if not vvix else pctile(vvix[-1], vvix[-YEAR_SESSIONS:], min_len=min_history),
        skewidx_pctile=None if not skew_index else pctile(skew_index[-1], skew_index[-YEAR_SESSIONS:], min_len=min_history),
    )

    # --- own surface (one `daily` row per session, 5.4) ---------------------------------------------------------------
    daily = _daily(view, underlying)
    iv30_bp_series = _column(daily, "iv30_bp")
    iv30_bp = _last(iv30_bp_series)
    iv90_bp = _last(_column(daily, "iv90_bp"))
    skew25_bp_series = _column(daily, "skew25_bp")
    skew25_bp = _last(skew25_bp_series)
    iv30 = None if iv30_bp is None else iv30_bp / BP_PER_UNIT
    iv30_5_ago = iv30_bp_series[-6] if len(iv30_bp_series) >= 6 else None
    out.update(
        iv30=iv30,
        iv30_bp=None if iv30_bp is None else int(iv30_bp),
        iv90=None if iv90_bp is None else iv90_bp / BP_PER_UNIT,
        iv_rank=_iv_rank(iv30_bp, iv30_bp_series, cfg),
        iv_rv=_ratio(iv30, out.get("rv20")),
        iv_chg_1w=None if iv30_bp is None or not iv30_5_ago else iv30_bp / iv30_5_ago - 1.0,
        iv_term=_ratio(iv30, None if iv90_bp is None else iv90_bp / BP_PER_UNIT),
        skew25=None if skew25_bp is None else skew25_bp / BP_PER_UNIT,
        skew_pctile=_window_pctile(skew25_bp_series, min_len=min_history),
        rv20_pctile=_window_pctile(_column(daily, "rv20_bp"), min_len=min_history),
        iv_hist_proxy_pct=_proxy_pct(daily),
    )

    # --- expected moves, in trading time (V13) ------------------------------------------------------------------------
    term = parse_atm_term(daily["atm_term_json"].iloc[-1] if not daily.empty and "atm_term_json" in daily.columns else None)
    out["atm_term"] = term
    for horizon, name in ((1, "1"), (5, "5"), (hold_sessions, "hold")):
        tt = _resolve_tt(view, horizon)
        found = None if tt is None else expected_move(term, tt)
        if found is not None:
            em, tenths, _quality = found
            out[f"em_{name}"] = em
            out[f"em_{name}_tenths"] = tenths
            if name == "5":
                out["iv_var_5"] = em * em
    return FeatureSet(**out)


def _iv_rank(iv30_bp: float | None, history: Sequence[float | None], cfg: DataConfig) -> int | None:
    """5.3 robust IV rank: the 2nd / 98th percentiles of the last 252 sessions' `iv30_bp` (today included) as the range.

    One absurd snapshot can therefore not pin the year's low or high for 252 sessions. `hi == lo` gives 50.
    """
    if iv30_bp is None:
        return None
    window = [value for value in history[-YEAR_SESSIONS:] if value is not None]
    if len(window) < max(2, cfg.min_history_sessions):
        return None
    values = np.asarray(window, dtype=np.float64)
    lo = float(np.percentile(values, cfg.iv_rank_lo_pct))
    hi = float(np.percentile(values, cfg.iv_rank_hi_pct))
    if hi == lo:
        return 50
    return int(min(100, max(0, round(100.0 * (iv30_bp - lo) / (hi - lo)))))
