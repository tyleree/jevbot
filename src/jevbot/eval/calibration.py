"""Calibration analysis - the primary endpoint's machinery (DESIGN.md 12.3; WP07).

Everything here is a **pure function** of the frames it is handed: no IO, no clock, no config object, no other `eval`
module.  The statistics are numpy + `scipy.special.ndtr` / `scipy.stats.norm` only (D15, 1.1).

Frame contracts
---------------
`events` - one row per *scored forecast* (`eval.load.calibration_frame`: FORECAST joined to OUTCOME by `event_key`, 12.3).
Columns read here:

==================  =========================================================================================
`session`           `datetime.date` - the decision session D
`question_id`       str
`horizon`           int - the question's own horizon h in sessions (`OutcomeSpec.horizon_sessions`)
`p`                 float in [0, 1] - the evaluated forecaster's P(yes); NaN = MISSING (NULL `p_ppm`, 6.4)
`p_implied`         float in [0, 1] - the frozen option-implied comparison probability
`y`                 1 / 0 / NaN - the realised outcome; NaN = void (excluded and listed by the caller)
`resolved_on`       `datetime.date`
`event_key`         str - one event, every decider and reference; used to de-duplicate training pairs
==================  =========================================================================================

`p_ppm` / `p_implied_ppm` (the raw `v_forecasts` integer columns) are accepted as a fallback and divided by 1e6.

`history` - `eval.load.reference_history(run_store)`: columns `question_id, horizon, session, p_implied, y, resolved_on`
from the run named in `[prereg.reference_history]`.  It is passed **only** to `base_rate_expanding` and
`recalibrate_walkforward`; no other function in this module accepts it, so history rows can never reach a scored table
(12.1 exemption 1).

The purge (12.1 `[prereg.reference_history].purge`)
---------------------------------------------------
A training pair is usable for a forecast made at session D only if `resolved_on <= prev_session(D, h)`, h = the
question's own horizon.  `prev_session` is evaluated on the **session grid**: the sorted union of every `session` and
`resolved_on` in `events` and `history` (both frames are dense in trading sessions), or the explicit `sessions`
argument when the caller has the real `Calendar` at hand.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final, Literal, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import norm

from jevbot.errors import EvalError, PreregError

__all__ = [
    "CoherenceStats",
    "IsotonicMap",
    "MurphyDecomposition",
    "ReliabilityBin",
    "ReliabilityTable",
    "base_rate_expanding",
    "base_rate_in_sample",
    "brier",
    "brier_skill",
    "build_references",
    "coherence",
    "ece",
    "eligible_sessions",
    "impute_missing",
    "ineligible_sessions",
    "log_loss",
    "loss_differential",
    "mce",
    "murphy",
    "pav_isotonic",
    "question_kind",
    "recalibrate_walkforward",
    "reliability_table",
    "require_evaluable_look",
    "session_grid",
    "sharpness",
    "unique_sessions",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: `pav_isotonic` returns this: a monotone non-decreasing map, clipped to the fitted range at both ends.
IsotonicMap = Callable[[npt.ArrayLike], FloatArray]

BinStrategy = Literal["quantile", "fixed"]

LOG_LOSS_EPS: Final = 0.01  # Nouls look clipped to [0.01, 0.99] (12.3)
SHARPNESS_BINS: Final = 20  # histogram of p (12.3)
#: default tolerance of `coherence`: three 0.01-quantised, [0.01, 0.99]-clipped probabilities can sum to 1 +/- 0.015
#: from rounding alone, and the clip adds up to another 0.01 per component.
COHERENCE_TOL: Final = 0.03

_HORIZON_SUFFIXES: Final[tuple[str, ...]] = ("_1s", "_5s", "_hold")
_EPS: Final = 1e-12


# ======================================================================================================================
# small array helpers
# ======================================================================================================================


def _as_float(values: npt.ArrayLike) -> FloatArray:
    return np.asarray(values, dtype=np.float64).ravel()


def _as_prob(values: npt.ArrayLike, name: str, *, allow_nan: bool = False) -> FloatArray:
    arr = _as_float(values)
    finite = np.isfinite(arr)
    if not allow_nan and not bool(finite.all()):
        raise EvalError(f"{name} contains non-finite values; exclude MISSING / void rows first")
    if arr.size and finite.any():
        lo = float(arr[finite].min())
        hi = float(arr[finite].max())
        if lo < -_EPS or hi > 1.0 + _EPS:
            raise EvalError(f"{name} outside [0, 1]: min={lo!r} max={hi!r}")
    return arr


def _as_binary(values: npt.ArrayLike, name: str) -> FloatArray:
    arr = _as_float(values)
    if not bool(np.isfinite(arr).all()):
        raise EvalError(f"{name} contains non-finite values; void outcomes are excluded, never imputed (12.1)")
    if arr.size and not bool(np.isin(arr, (0.0, 1.0)).all()):
        raise EvalError(f"{name} must be 0 / 1")
    return arr


def _same_length(**arrays: FloatArray) -> int:
    lengths = {name: int(arr.size) for name, arr in arrays.items()}
    if len(set(lengths.values())) > 1:
        raise EvalError(f"length mismatch: {lengths}")
    return next(iter(lengths.values()), 0)


# ======================================================================================================================
# scores
# ======================================================================================================================


def brier(p: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """`mean((p - y)^2)` (12.3)."""
    pa = _as_prob(p, "p")
    ya = _as_binary(y, "y")
    n = _same_length(p=pa, y=ya)
    if n == 0:
        return float("nan")
    return float(np.mean((pa - ya) ** 2))


def brier_skill(p: npt.ArrayLike, y: npt.ArrayLike, p_ref: npt.ArrayLike) -> float:
    """`1 - BS(p) / BS(p_ref)` (12.3).  NaN when the reference is perfect (`BS(p_ref) == 0`)."""
    bs = brier(p, y)
    bs_ref = brier(p_ref, y)
    if not np.isfinite(bs_ref) or abs(bs_ref) < _EPS:
        return float("nan")
    return float(1.0 - bs / bs_ref)


def log_loss(p: npt.ArrayLike, y: npt.ArrayLike, eps: float = LOG_LOSS_EPS) -> float:
    """Mean negative log-likelihood with `p` clipped to `[eps, 1 - eps]` (Nouls look clipped to [0.01, 0.99])."""
    if not 0.0 < eps < 0.5:
        raise EvalError(f"eps must be in (0, 0.5): {eps!r}")
    pa = _as_prob(p, "p")
    ya = _as_binary(y, "y")
    n = _same_length(p=pa, y=ya)
    if n == 0:
        return float("nan")
    clipped = np.clip(pa, eps, 1.0 - eps)
    return float(-np.mean(ya * np.log(clipped) + (1.0 - ya) * np.log1p(-clipped)))


def sharpness(p: npt.ArrayLike) -> IntArray:
    """Histogram of `p` over `[0, 1]` in `SHARPNESS_BINS` equal bins (12.3)."""
    pa = _as_prob(p, "p", allow_nan=True)
    counts, _ = np.histogram(pa[np.isfinite(pa)], bins=SHARPNESS_BINS, range=(0.0, 1.0))
    return np.asarray(counts, dtype=np.int64)


def impute_missing(p: npt.ArrayLike, p_ref: npt.ArrayLike) -> FloatArray:
    """The pre-registered `missing` rule (12.1): a MISSING forecast (NaN `p`) takes the reference's value.

    Worst case for skill - the imputed row contributes exactly zero loss differential against that reference.
    """
    pa = _as_prob(p, "p", allow_nan=True)
    ra = _as_prob(p_ref, "p_ref", allow_nan=True)
    _same_length(p=pa, p_ref=ra)
    return np.where(np.isfinite(pa), pa, ra)


# ======================================================================================================================
# reliability / ECE / Murphy
# ======================================================================================================================


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One reliability bin: its forecast range, count, mean forecast, observed frequency and Wilson interval."""

    lo: float
    hi: float
    n: int
    mean_p: float
    freq: float
    ci_lo: float
    ci_hi: float


@dataclass(frozen=True, slots=True)
class ReliabilityTable:
    """The binned reliability curve of 12.3 plus the bin assignment `murphy` needs.

    `p` and `bin_of` are in the ORIGINAL row order of the forecasts the table was built from, so `murphy(table, y)`
    must be given `y` in that same order.
    """

    bins: tuple[ReliabilityBin, ...]
    strategy: BinStrategy
    n: int
    base_rate: float
    level: float
    p: FloatArray
    bin_of: IntArray
    fell_back: bool  # a quantile table that collapsed to fewer than 3 bins and used fixed edges instead

    def rows(self) -> list[dict[str, float]]:
        """Bin rows as plain dicts (the `tables/reliability_<qid>.csv` shape of 12.8)."""
        return [
            {
                "bin": float(i),
                "lo": b.lo,
                "hi": b.hi,
                "n": float(b.n),
                "mean_p": b.mean_p,
                "freq": b.freq,
                "ci_lo": b.ci_lo,
                "ci_hi": b.ci_hi,
            }
            for i, b in enumerate(self.bins)
        ]


class MurphyDecomposition(NamedTuple):
    """`BS = reliability - resolution + uncertainty + residual` (12.3); a plain 4-tuple by construction."""

    reliability: float
    resolution: float
    uncertainty: float
    residual: float


def _wilson(k: float, n: int, z: float) -> tuple[float, float]:
    """Wilson score interval for `k` successes out of `n`."""
    if n <= 0:
        return (float("nan"), float("nan"))
    zz = z * z
    denom = n + zz
    centre = (k + zz / 2.0) / denom
    half = (z / denom) * float(np.sqrt(k * (n - k) / n + zz / 4.0))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _quantile_groups(ps_sorted: FloatArray, n_bins: int, min_per_bin: int) -> list[tuple[int, int]]:
    """Contiguous `[start, end)` index groups of the SORTED forecasts under the 12.3 tie rule.

    A bin edge is moved **forward to the next distinct value**, so equal `p` never straddle two bins; bins under
    `min_per_bin` then merge with their smaller neighbour.
    """
    n = int(ps_sorted.size)
    if n == 0:
        return []
    # every position at which the sorted value changes: the only legal cut points
    changes = np.flatnonzero(np.diff(ps_sorted) != 0.0) + 1
    cuts: list[int] = []
    for i in range(1, n_bins):
        target = (i * n) // n_bins
        if target <= 0:
            continue
        pos = int(np.searchsorted(changes, target, side="left"))
        if pos >= changes.size:
            continue  # all remaining values are equal: no further edge exists
        cuts.append(int(changes[pos]))
    bounds = [0, *sorted(set(cuts)), n]
    groups = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    while len(groups) > 1:
        small = next((i for i, (s, e) in enumerate(groups) if e - s < min_per_bin), None)
        if small is None:
            break
        if small == 0:
            other = 1
        elif small == len(groups) - 1:
            other = small - 1
        else:
            left = groups[small - 1][1] - groups[small - 1][0]
            right = groups[small + 1][1] - groups[small + 1][0]
            other = small + 1 if right <= left else small - 1
        lo, hi = min(small, other), max(small, other)
        groups[lo : hi + 1] = [(groups[lo][0], groups[hi][1])]
    return groups


def reliability_table(
    p: npt.ArrayLike,
    y: npt.ArrayLike,
    n_bins: int = 10,
    strategy: BinStrategy = "quantile",
    min_per_bin: int = 20,
    *,
    level: float = 0.95,
) -> ReliabilityTable:
    """Bin lo / hi, n, mean p, observed frequency and Wilson CI per bin (12.3).

    `strategy="quantile"` applies the tie rule of 12.3 (edges moved forward to the next distinct value, under-sized
    bins merged); fewer than 3 surviving bins fall back to the fixed edges `0, 1/n_bins, ..., 1` and set `fell_back`.
    Both the quantile and the fixed-width table are reported by the caller, hence the explicit `strategy` argument.
    """
    if n_bins < 1:
        raise EvalError(f"n_bins must be >= 1: {n_bins!r}")
    if min_per_bin < 1:
        raise EvalError(f"min_per_bin must be >= 1: {min_per_bin!r}")
    if not 0.0 < level < 1.0:
        raise EvalError(f"level must be in (0, 1): {level!r}")
    pa = _as_prob(p, "p")
    ya = _as_binary(y, "y")
    n = _same_length(p=pa, y=ya)
    z = float(norm.ppf(0.5 + level / 2.0))

    bin_of = np.full(n, -1, dtype=np.int64)
    edges: list[tuple[float, float]] = []
    fell_back = False

    groups: list[tuple[int, int]] = []
    if strategy == "quantile" and n > 0:
        order = np.argsort(pa, kind="stable")
        ps_sorted = pa[order]
        groups = _quantile_groups(ps_sorted, n_bins, min_per_bin)
        if len(groups) >= 3:
            for k, (s, e) in enumerate(groups):
                bin_of[order[s:e]] = k
                edges.append((float(ps_sorted[s]), float(ps_sorted[e - 1])))
        else:
            fell_back = True
    elif strategy != "quantile" and strategy != "fixed":
        raise EvalError(f"unknown strategy {strategy!r}")

    if strategy == "fixed" or fell_back:
        edges = []
        bin_of = np.full(n, -1, dtype=np.int64)
        fixed = np.linspace(0.0, 1.0, n_bins + 1)
        raw = np.clip(np.searchsorted(fixed, pa, side="right") - 1, 0, n_bins - 1)
        keep = 0
        for k in range(n_bins):
            members = raw == k
            if not bool(members.any()):
                continue
            bin_of[members] = keep
            edges.append((float(fixed[k]), float(fixed[k + 1])))
            keep += 1

    bins: list[ReliabilityBin] = []
    for k, (lo, hi) in enumerate(edges):
        members = bin_of == k
        count = int(members.sum())
        successes = float(ya[members].sum())
        ci_lo, ci_hi = _wilson(successes, count, z)
        bins.append(
            ReliabilityBin(
                lo=lo,
                hi=hi,
                n=count,
                mean_p=float(pa[members].mean()),
                freq=successes / count,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
            )
        )

    base = float(ya.mean()) if n else float("nan")
    return ReliabilityTable(
        bins=tuple(bins),
        strategy=strategy,
        n=n,
        base_rate=base,
        level=level,
        p=pa,
        bin_of=bin_of,
        fell_back=fell_back,
    )


def ece(table: ReliabilityTable) -> float:
    """Expected calibration error: `sum_k n_k/N * |mean_p_k - freq_k|` (12.3)."""
    if table.n == 0 or not table.bins:
        return float("nan")
    total = sum(b.n for b in table.bins)
    if total == 0:
        return float("nan")
    return float(sum(b.n * abs(b.mean_p - b.freq) for b in table.bins) / total)


def mce(table: ReliabilityTable) -> float:
    """Maximum calibration error: `max_k |mean_p_k - freq_k|` (12.3)."""
    if not table.bins:
        return float("nan")
    return float(max(abs(b.mean_p - b.freq) for b in table.bins))


def murphy(table: ReliabilityTable, y: npt.ArrayLike) -> MurphyDecomposition:
    """Murphy decomposition with the exact within-bin residual: `BS = REL - RES + UNC + residual` (12.3).

    `REL = 1/N sum n_k (p_k - o_k)^2`, `RES = 1/N sum n_k (o_k - o)^2`, `UNC = o (1 - o)` and
    `residual = 1/N sum_k [ sum_i (p_i - p_k)^2 - 2 sum_i (p_i - p_k) y_i ]` over the members of bin k - the term that
    makes the identity exact when a bin holds more than one distinct forecast.
    """
    ya = _as_binary(y, "y")
    if int(ya.size) != table.n:
        raise EvalError(f"y has {ya.size} rows, the table was built from {table.n}")
    if table.n == 0 or not table.bins:
        return MurphyDecomposition(float("nan"), float("nan"), float("nan"), float("nan"))

    n = float(table.n)
    base = float(ya.mean())
    rel = 0.0
    res = 0.0
    residual = 0.0
    for k in range(len(table.bins)):
        members = table.bin_of == k
        count = float(members.sum())
        if count == 0.0:
            continue
        p_bin = table.p[members]
        y_bin = ya[members]
        p_mean = float(p_bin.mean())
        o_k = float(y_bin.mean())
        rel += count * (p_mean - o_k) ** 2
        res += count * (o_k - base) ** 2
        dev = p_bin - p_mean
        residual += float(np.sum(dev * dev) - 2.0 * np.sum(dev * y_bin))
    return MurphyDecomposition(
        reliability=rel / n,
        resolution=res / n,
        uncertainty=base * (1.0 - base),
        residual=residual / n,
    )


# ======================================================================================================================
# coherence
# ======================================================================================================================


@dataclass(frozen=True, slots=True)
class CoherenceStats:
    """`|p_down + p_up + p_inside - 1|` per event (12.3); each triplet is exhaustive by construction (6.4)."""

    n: int
    mean_abs_error: float
    max_abs_error: float
    q95_abs_error: float
    frac_within_tol: float
    tol: float
    errors: FloatArray


def coherence(
    p_down: npt.ArrayLike,
    p_up: npt.ArrayLike,
    p_inside: npt.ArrayLike,
    *,
    tol: float = COHERENCE_TOL,
) -> CoherenceStats:
    """Per-event coherence of an exhaustive down / up / inside triplet (12.3).

    The caller groups by horizon: the three arrays must be the three forecasts of the SAME events, in one order.
    """
    down = _as_prob(p_down, "p_down", allow_nan=True)
    up = _as_prob(p_up, "p_up", allow_nan=True)
    inside = _as_prob(p_inside, "p_inside", allow_nan=True)
    _same_length(p_down=down, p_up=up, p_inside=inside)
    err = np.abs(down + up + inside - 1.0)
    finite = np.isfinite(err)
    usable = err[finite]
    if usable.size == 0:
        return CoherenceStats(0, float("nan"), float("nan"), float("nan"), float("nan"), tol, err)
    return CoherenceStats(
        n=int(usable.size),
        mean_abs_error=float(usable.mean()),
        max_abs_error=float(usable.max()),
        q95_abs_error=float(np.quantile(usable, 0.95)),
        frac_within_tol=float(np.mean(usable <= tol + _EPS)),
        tol=tol,
        errors=err,
    )


# ======================================================================================================================
# isotonic regression (pool adjacent violators)
# ======================================================================================================================


def pav_isotonic(x: npt.ArrayLike, y: npt.ArrayLike) -> IsotonicMap:
    """Pool-adjacent-violators isotonic regression in numpy; returns the fitted monotone map (12.3).

    Duplicate `x` are aggregated first (weighted by their count), so the result is a genuine function of `x`.  The map
    interpolates linearly between the fitted knots and is clipped to the end values outside the fitted range.
    """
    xa = _as_float(x)
    ya = _as_float(y)
    n = _same_length(x=xa, y=ya)
    if n == 0:
        raise EvalError("pav_isotonic needs at least one point")
    if not bool(np.isfinite(xa).all()) or not bool(np.isfinite(ya).all()):
        raise EvalError("pav_isotonic needs finite x and y")

    knots_x, inverse, counts = np.unique(xa, return_inverse=True, return_counts=True)
    sums = np.zeros(knots_x.size, dtype=np.float64)
    np.add.at(sums, inverse, ya)
    values = sums / counts
    weights = counts.astype(np.float64)

    # pool adjacent violators over the (already x-sorted) unique knots
    stack_val: list[float] = []
    stack_w: list[float] = []
    for value, weight in zip(values.tolist(), weights.tolist(), strict=True):
        v, w = float(value), float(weight)
        while stack_val and stack_val[-1] > v:
            pv, pw = stack_val.pop(), stack_w.pop()
            v = (pv * pw + v * w) / (pw + w)
            w = pw + w
        stack_val.append(v)
        stack_w.append(w)

    fitted = np.empty(knots_x.size, dtype=np.float64)
    pos = 0
    for v, w in zip(stack_val, stack_w, strict=True):
        # the block spans as many knots as its pooled weight accounts for
        span = 0
        acc = 0.0
        while pos + span < knots_x.size and acc < w - _EPS:
            acc += float(weights[pos + span])
            span += 1
        fitted[pos : pos + span] = v
        pos += span
    if pos != knots_x.size:  # pragma: no cover - defensive: the block weights sum to the total weight
        raise EvalError("isotonic block weights do not cover every knot")

    xs = knots_x.copy()
    ys = fitted.copy()

    def _map(query: npt.ArrayLike) -> FloatArray:
        q = np.asarray(query, dtype=np.float64)
        out = np.interp(q, xs, ys, left=float(ys[0]), right=float(ys[-1]))
        return np.asarray(out, dtype=np.float64)

    return _map


# ======================================================================================================================
# session grid and the purge
# ======================================================================================================================


def question_kind(question_id: str) -> str:
    """The question's *kind* for recalibration grouping: the id without its horizon suffix (`_1s` / `_5s` / `_hold`).

    `eval.down_1em_5s -> eval.down_1em`; `under.direction#bullish` keeps its `#suffix` (the three derived Nouls of 6.4
    are different outcomes of the same base question and must never share a recalibration map).
    """
    head, sep, tail = question_id.partition("#")
    for suffix in _HORIZON_SUFFIXES:
        if head.endswith(suffix):
            head = head[: -len(suffix)]
            break
    return head + sep + tail


def _dates(values: Iterable[Any]) -> list[date]:
    out: list[date] = []
    for value in values:
        if isinstance(value, date) and not isinstance(value, pd.Timestamp):
            out.append(value)
        elif value is None or (isinstance(value, float) and not np.isfinite(value)):
            continue
        else:
            out.append(pd.Timestamp(value).date())
    return out


def session_grid(events: pd.DataFrame, history: pd.DataFrame | None = None) -> list[date]:
    """The sorted trading-session grid the purge is evaluated on: every `session` and `resolved_on` of both frames."""
    values: list[date] = []
    for frame in (events, history):
        if frame is None or len(frame) == 0:
            continue
        for column in ("session", "resolved_on"):
            if column in frame.columns:
                values.extend(_dates(frame[column].tolist()))
    return sorted(set(values))


def _prev_session(grid: Sequence[date], day: date, back: int) -> date | None:
    """`prev_session(day, back)` on the grid; None when the grid does not reach that far back."""
    pos = int(np.searchsorted(np.asarray(grid, dtype="datetime64[D]"), np.datetime64(day, "D"), side="left"))
    target = pos - back
    if target < 0:
        return None
    return grid[target]


def _need(frame: pd.DataFrame, name: str, *, ppm: str | None = None) -> FloatArray:
    """A float column, accepting the raw ppm integer column of `v_forecasts` (13.4) as a fallback."""
    if name in frame.columns:
        return np.asarray(pd.to_numeric(frame[name], errors="coerce"), dtype=np.float64)
    if ppm is not None and ppm in frame.columns:
        return np.asarray(pd.to_numeric(frame[ppm], errors="coerce"), dtype=np.float64) / 1_000_000.0
    raise EvalError(f"frame is missing column {name!r}" + (f" (or {ppm!r})" if ppm else ""))


def _kinds(frame: pd.DataFrame) -> list[str]:
    if "question_kind" in frame.columns:
        return [str(v) for v in frame["question_kind"].tolist()]
    return [question_kind(str(v)) for v in frame["question_id"].tolist()]


def _training_pairs(
    events: pd.DataFrame,
    history: pd.DataFrame | None,
    *,
    key: Callable[[pd.DataFrame], list[str]],
    with_implied: bool,
) -> dict[str, tuple[npt.NDArray[np.datetime64], FloatArray, FloatArray]]:
    """Per group: the resolved training pairs, sorted by `resolved_on`.

    History rows plus the run's OWN resolved events (12.1 `own_events`), de-duplicated by `event_key` so the two
    with-text / without-text forecasts of one event contribute one training pair.
    """
    frames: list[pd.DataFrame] = []
    if history is not None and len(history) > 0:
        frames.append(history)
    if len(events) > 0:
        own = events.drop_duplicates(subset=["event_key"]) if "event_key" in events.columns else events
        frames.append(own)

    out: dict[str, list[tuple[date, float, float]]] = {}
    for frame in frames:
        groups = key(frame)
        resolved = _dates_or_none(frame["resolved_on"].tolist()) if "resolved_on" in frame.columns else [None] * len(frame)
        y = _need(frame, "y")
        implied = _need(frame, "p_implied", ppm="p_implied_ppm") if with_implied else np.zeros(len(frame), dtype=np.float64)
        for i, group in enumerate(groups):
            day = resolved[i]
            if day is None or not np.isfinite(y[i]):
                continue  # open or void: never a training pair
            if with_implied and not np.isfinite(implied[i]):
                continue
            out.setdefault(group, []).append((day, float(y[i]), float(implied[i])))

    packed: dict[str, tuple[npt.NDArray[np.datetime64], FloatArray, FloatArray]] = {}
    for group, rows in out.items():
        rows.sort(key=lambda row: row[0])
        days = np.asarray([row[0] for row in rows], dtype="datetime64[D]")
        ys = np.asarray([row[1] for row in rows], dtype=np.float64)
        imp = np.asarray([row[2] for row in rows], dtype=np.float64)
        packed[group] = (days, ys, imp)
    return packed


def _dates_or_none(values: Iterable[Any]) -> list[date | None]:
    out: list[date | None] = []
    for value in values:
        if value is None:
            out.append(None)
        elif isinstance(value, float) and not np.isfinite(value):
            out.append(None)
        elif isinstance(value, date) and not isinstance(value, pd.Timestamp):
            out.append(value)
        else:
            try:
                out.append(pd.Timestamp(value).date())
            except (ValueError, TypeError):  # pragma: no cover - defensive
                out.append(None)
    return out


# ======================================================================================================================
# the two reference forecasters (the ONLY functions that accept `history`)
# ======================================================================================================================


def base_rate_expanding(
    events: pd.DataFrame,
    history: pd.DataFrame | None,
    *,
    min_events: int,
    sessions: Sequence[date] | None = None,
) -> FloatArray:
    """The expanding base rate reference, warm-started from the reference history (12.3).

    Per question: the frequency of `y` among the training events (history rows + the run's own earlier events)
    **resolved on or before `prev_session(D, h)`** - the purge.  Fewer than `min_events` usable pairs leaves the
    reference **unavailable** (`NaN`): never a noisy early estimate, and there is no fallback (12.1).
    """
    if min_events < 1:
        raise EvalError(f"min_events must be >= 1: {min_events!r}")
    n = len(events)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    grid = list(sessions) if sessions is not None else session_grid(events, history)
    training = _training_pairs(events, history, key=lambda f: [str(v) for v in f["question_id"].tolist()], with_implied=False)

    question_ids = [str(v) for v in events["question_id"].tolist()]
    event_sessions = _dates_or_none(events["session"].tolist())
    horizons = np.asarray(pd.to_numeric(events["horizon"], errors="coerce"), dtype=np.float64)

    for i in range(n):
        pack = training.get(question_ids[i])
        day = event_sessions[i]
        if pack is None or day is None or not np.isfinite(horizons[i]):
            continue
        cutoff = _prev_session(grid, day, int(horizons[i]))
        if cutoff is None:
            continue
        days, ys, _ = pack
        k = int(np.searchsorted(days, np.datetime64(cutoff, "D"), side="right"))
        if k < min_events:
            continue
        out[i] = float(ys[:k].mean())
    return out


def base_rate_in_sample(events: pd.DataFrame) -> FloatArray:
    """The in-sample base rate per question - **display only** (12.3); it peeks and can never be a verdict reference."""
    n = len(events)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    y = _need(events, "y")
    question_ids = np.asarray([str(v) for v in events["question_id"].tolist()])
    for qid in np.unique(question_ids):
        members = question_ids == qid
        usable = members & np.isfinite(y)
        if not bool(usable.any()):
            continue
        out[members] = float(y[usable].mean())
    return out


def recalibrate_walkforward(
    events: pd.DataFrame,
    history: pd.DataFrame | None,
    *,
    min_events: int,
    refit_sessions: int,
    sessions: Sequence[date] | None = None,
) -> FloatArray:
    """The walk-forward recalibrated option-implied reference (12.3).

    An isotonic (PAV) map `p_implied -> frequency`, fitted ONLY on `(p_implied, market outcome)` pairs usable under the
    purge rule, **per question kind and horizon**, refitted every `refit_sessions` sessions.  It uses no Jev output, so
    pre-release market history (the reference history) is legitimate TRAINING data.  Fewer than `min_events` pairs
    leaves the reference **unavailable** (`NaN`): there is NO fallback to raw implied here (`never_sufficient`, 12.1).
    """
    if min_events < 1:
        raise EvalError(f"min_events must be >= 1: {min_events!r}")
    if refit_sessions < 1:
        raise EvalError(f"refit_sessions must be >= 1: {refit_sessions!r}")
    n = len(events)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    grid = list(sessions) if sessions is not None else session_grid(events, history)

    def _group(frame: pd.DataFrame) -> list[str]:
        horizons = pd.to_numeric(frame["horizon"], errors="coerce").tolist()
        return [f"{kind}|{int(h) if h == h else -1}" for kind, h in zip(_kinds(frame), horizons, strict=True)]

    training = _training_pairs(events, history, key=_group, with_implied=True)

    implied = _need(events, "p_implied", ppm="p_implied_ppm")
    event_sessions = _dates_or_none(events["session"].tolist())
    horizons = np.asarray(pd.to_numeric(events["horizon"], errors="coerce"), dtype=np.float64)
    groups = np.asarray(_group(events))

    # one global refit clock, so every group refits on the same schedule
    ordered = sorted({day for day in event_sessions if day is not None})
    epoch_of: dict[date, int] = {day: i // refit_sessions for i, day in enumerate(ordered)}
    epoch_start: dict[int, date] = {}
    for day in ordered:
        epoch_start.setdefault(epoch_of[day], day)

    for group in np.unique(groups):
        pack = training.get(str(group))
        members = np.flatnonzero(groups == group)
        if pack is None or members.size == 0:
            continue
        days, ys, imp = pack
        by_epoch: dict[int, list[int]] = {}
        for i in members.tolist():
            event_day = event_sessions[i]
            if event_day is None:
                continue
            by_epoch.setdefault(epoch_of[event_day], []).append(i)
        for epoch, rows in by_epoch.items():
            fit_day = epoch_start[epoch]
            horizon = horizons[rows[0]]
            if not np.isfinite(horizon):
                continue
            cutoff = _prev_session(grid, fit_day, int(horizon))
            if cutoff is None:
                continue
            k = int(np.searchsorted(days, np.datetime64(cutoff, "D"), side="right"))
            if k < min_events:
                continue
            mapping = pav_isotonic(imp[:k], ys[:k])
            idx = np.asarray(rows, dtype=np.int64)
            out[idx] = np.clip(mapping(implied[idx]), 0.0, 1.0)
    return out


def build_references(
    events: pd.DataFrame,
    history: pd.DataFrame | None,
    *,
    recal_min_events: int,
    base_rate_min_events: int,
    refit_sessions: int,
    sessions: Sequence[date] | None = None,
) -> dict[str, FloatArray]:
    """Every reference forecaster of 12.3 on the same events, aligned row for row with `events`.

    `implied_recalibrated` and `base_rate_expanding` are the two **verdict** references of the pre-registration;
    `raw_implied`, `base_rate_in_sample` and `coin` are reported, never a verdict (`never_sufficient`, 12.1).
    MockJev's constants are supplied by the caller (they need the mock's profile, not this module).
    """
    n = len(events)
    raw = _need(events, "p_implied", ppm="p_implied_ppm") if n else np.zeros(0, dtype=np.float64)
    return {
        "raw_implied": raw,
        "implied_recalibrated": recalibrate_walkforward(
            events, history, min_events=recal_min_events, refit_sessions=refit_sessions, sessions=sessions
        ),
        "base_rate_expanding": base_rate_expanding(events, history, min_events=base_rate_min_events, sessions=sessions),
        "base_rate_in_sample": base_rate_in_sample(events),
        "coin": np.full(n, 0.5, dtype=np.float64),
    }


# ======================================================================================================================
# eligibility and the d_t series
# ======================================================================================================================


def unique_sessions(sessions: Iterable[Any]) -> pd.Index:
    """The sorted unique sessions of a column - the index `loss_differential` returns its values in."""
    return pd.Index(sorted(set(_dates(sessions))), name="session")


def eligible_sessions(events: pd.DataFrame, refs: Mapping[str, FloatArray], primary: Sequence[str]) -> pd.Index:
    """Sessions on which EVERY primary question has BOTH references available (12.1 `eligible_session`).

    `refs` maps reference name -> an array aligned with `events`.  A session on which any primary question is absent,
    or on which any reference is `NaN` for it, is **excluded** from `d_t` and from every look's `n_sessions`;
    `ineligible_sessions` lists the rest for the report.
    """
    if not primary:
        raise EvalError("the primary family is empty")
    if not refs:
        raise EvalError("eligible_sessions needs at least one reference")
    n = len(events)
    for name, values in refs.items():
        if int(np.asarray(values).size) != n:
            raise EvalError(f"reference {name!r} has {np.asarray(values).size} rows, events has {n}")
    if n == 0:
        return pd.Index([], name="session")

    available = np.ones(n, dtype=bool)
    for values in refs.values():
        available &= np.isfinite(np.asarray(values, dtype=np.float64))

    question_ids = np.asarray([str(v) for v in events["question_id"].tolist()])
    event_sessions = np.asarray(_dates(events["session"].tolist()), dtype=object)
    wanted = set(primary)

    eligible: list[date] = []
    for day in sorted(set(event_sessions.tolist())):
        on_day = event_sessions == day
        ok = True
        for qid in wanted:
            rows = on_day & (question_ids == qid)
            if not bool(rows.any()) or not bool(available[rows].all()):
                ok = False
                break
        if ok:
            eligible.append(day)
    return pd.Index(eligible, name="session")


def ineligible_sessions(events: pd.DataFrame, eligible: pd.Index) -> pd.Index:
    """The sessions excluded by `eligible_sessions` - listed in every report (12.1)."""
    everything = unique_sessions(events["session"].tolist()) if len(events) else pd.Index([], name="session")
    keep = set(eligible.tolist())
    return pd.Index([day for day in everything.tolist() if day not in keep], name="session")


def require_evaluable_look(eligible: pd.Index, *, n_sessions: int, look: str = "look") -> pd.Index:
    """The first `n_sessions` eligible sessions of a pre-registered look, or `PreregError` (12.1(b)).

    "a look with fewer than `n_sessions` eligible sessions **cannot be evaluated**" - there is no fallback reference
    and no small-sample verdict; the look is simply not evaluable yet.
    """
    if n_sessions < 1:
        raise EvalError(f"n_sessions must be >= 1: {n_sessions!r}")
    if len(eligible) < n_sessions:
        raise PreregError(f"look not evaluable: {look} needs {n_sessions} eligible sessions, {len(eligible)} available")
    return pd.Index(eligible.tolist()[:n_sessions], name="session")


def loss_differential(
    p_a: npt.ArrayLike,
    p_b: npt.ArrayLike,
    y: npt.ArrayLike,
    sessions: Iterable[Any],
    question_ids: Iterable[Any],
) -> FloatArray:
    """The pre-registered `d_t` series (12.1 `unit` / `weighting`).

    `d_t = mean over PRIMARY QUESTIONS of [ mean over underlyings of ((p_a - y)^2 - (p_b - y)^2) ]`: each question's
    loss differential is averaged first, then the questions are averaged (`weighting = "equal per question"`).
    `p_a` is the REFERENCE and `p_b` the evaluated forecaster, so `d_t > 0` means the forecaster beat the reference.
    Rows with a non-finite `y` (void outcomes) are excluded; a session on which a question has no usable row yields
    `NaN` for that session.  The result is ordered by `unique_sessions(sessions)`.
    """
    pa = _as_prob(p_a, "p_a", allow_nan=True)
    pb = _as_prob(p_b, "p_b", allow_nan=True)
    ya = _as_float(y)
    session_list = _dates(sessions)
    qid_list = [str(v) for v in question_ids]
    n = _same_length(p_a=pa, p_b=pb, y=ya)
    if len(session_list) != n or len(qid_list) != n:
        raise EvalError(f"sessions ({len(session_list)}) / question_ids ({len(qid_list)}) do not match {n} rows")

    diff = (pa - ya) ** 2 - (pb - ya) ** 2
    usable = np.isfinite(diff) & np.isfinite(ya)

    index = unique_sessions(session_list)
    days = np.asarray(session_list, dtype=object)
    qids = np.asarray(qid_list)
    questions = sorted(set(qid_list))

    out = np.full(len(index), np.nan, dtype=np.float64)
    for pos, day in enumerate(index.tolist()):
        on_day = days == day
        per_question: list[float] = []
        complete = True
        for qid in questions:
            rows = on_day & (qids == qid) & usable
            if not bool(rows.any()):
                complete = False
                break
            per_question.append(float(diff[rows].mean()))
        if complete and per_question:
            out[pos] = float(np.mean(per_question))
    return out
