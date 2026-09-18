"""Block bootstrap: stationary (Politis-Romano) and cluster resampling (DESIGN.md 12.5; WP07).

numpy only (D15, 1.1) - `scipy.stats.norm` is not needed here.  Every function is pure: the caller owns the
`numpy.random.Generator`, and `bootstrap_seed` derives a reproducible one from `sha256(trial | "bootstrap" | metric)`
so a report re-run produces byte-identical intervals (12.5).

Overlapping evaluation horizons (5-session outcomes) mean inference uses **block methods only**; the i.i.d. bootstrap
(`block <= 1`) is deliberately reachable so that `eval power` can demonstrate it is over-sized (12.3).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from jevbot.errors import EvalError, PreregError

__all__ = [
    "IntervalMethod",
    "block_length_sample",
    "block_se",
    "bootstrap_ci",
    "bootstrap_seed",
    "cluster_bootstrap_ci",
    "default_block",
    "lower_bound",
    "paired_bootstrap_ci",
    "rng_for",
    "stationary_bootstrap_indices",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

CiMethod = Literal["percentile", "studentised"]
IntervalMethod = Literal["percentile", "studentised", "null_calibrated"]

MIN_BLOCK: Final = 5.0  # the 12.5 default floor
_EPS: Final = 1e-15


# ======================================================================================================================
# block length, seeds
# ======================================================================================================================


def default_block(n: int, *, longest_horizon: int = 0) -> float:
    """`max(5, 2 * longest horizon in the statistic, ceil(n ** (1/3)))` (12.5).

    The pre-registration additionally pins `min_block = 10` for the primary test: block >= 2 x the longest primary
    horizon (overlapping 5-session outcomes, 12.1).
    """
    if n < 1:
        raise EvalError(f"n must be >= 1: {n!r}")
    if longest_horizon < 0:
        raise EvalError(f"longest_horizon must be >= 0: {longest_horizon!r}")
    return float(max(MIN_BLOCK, 2.0 * longest_horizon, math.ceil(n ** (1.0 / 3.0))))


def bootstrap_seed(trial: str, metric: str) -> int:
    """`sha256(trial | "bootstrap" | metric)` -> a 64-bit seed (12.5): no run id, no wall clock, so it is reproducible."""
    digest = hashlib.sha256("|".join((trial, "bootstrap", metric)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def rng_for(trial: str, metric: str) -> np.random.Generator:
    """The seeded generator of `bootstrap_seed` (12.5)."""
    return np.random.default_rng(bootstrap_seed(trial, metric))


# ======================================================================================================================
# resampling indices
# ======================================================================================================================


def stationary_bootstrap_indices(n: int, block: float, reps: int, rng: np.random.Generator) -> IntArray:
    """Politis-Romano stationary bootstrap: geometric block lengths with `p = 1 / block`, wrapped circularly (12.5).

    Returns an `(reps, n)` integer array of positions into a length-`n` series.  `block <= 1` gives `p = 1`, i.e. the
    plain i.i.d. bootstrap - the deliberately mis-sized comparator of the size check (12.3).
    """
    if n < 1:
        raise EvalError(f"n must be >= 1: {n!r}")
    if reps < 1:
        raise EvalError(f"reps must be >= 1: {reps!r}")
    if not math.isfinite(block) or block <= 0.0:
        raise EvalError(f"block must be a positive finite number: {block!r}")
    p = min(1.0, 1.0 / block)

    starts = rng.integers(0, n, size=(reps, n), dtype=np.int64)
    new_block = rng.random((reps, n)) < p
    new_block[:, 0] = True

    positions = np.arange(n, dtype=np.int64)
    # the position at which the current block started (new_block[:, 0] is True, so the running maximum is well defined)
    block_start = np.maximum.accumulate(np.where(new_block, positions, 0), axis=1)
    offset = positions - block_start
    chosen = np.take_along_axis(starts, block_start, axis=1)
    return np.asarray((chosen + offset) % n, dtype=np.int64)


# ======================================================================================================================
# variance of a dependent mean
# ======================================================================================================================


def block_se(x: npt.ArrayLike, block: float) -> float:
    """Bartlett-kernel (Newey-West) standard error of the mean of a serially dependent series.

    Lag truncation `L = ceil(block) - 1`, weights `1 - l / (L + 1)`.  A non-positive long-run variance estimate falls
    back to the i.i.d. variance, so the studentised bootstrap never divides by a negative number.
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    n = int(arr.size)
    if n < 2:
        return float("nan")
    centred = arr - float(arr.mean())
    gamma0 = float(centred @ centred) / n
    if gamma0 <= _EPS:
        return 0.0
    lag_max = int(min(n - 1, max(1, math.ceil(block) - 1)))
    total = gamma0
    for lag in range(1, lag_max + 1):
        gamma = float(centred[lag:] @ centred[:-lag]) / n
        total += 2.0 * (1.0 - lag / (lag_max + 1.0)) * gamma
    if total <= 0.0:
        total = gamma0
    return float(math.sqrt(total / n))


def _mean(values: FloatArray) -> float:
    return float(np.mean(values))


# ======================================================================================================================
# confidence intervals
# ======================================================================================================================


def _check_level(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise EvalError(f"level must be in (0, 1): {level!r}")
    return 1.0 - level


def _resampled(x: FloatArray, indices: IntArray) -> FloatArray:
    return np.asarray(x[indices], dtype=np.float64)


def bootstrap_ci(
    x: npt.ArrayLike,
    stat: Callable[[FloatArray], float] = _mean,
    *,
    block: float | None,
    reps: int,
    level: float,
    rng: np.random.Generator,
    one_sided: bool = False,
    method: CiMethod = "percentile",
    se_fn: Callable[[FloatArray], float] | None = None,
) -> tuple[float, float, float]:
    """`(point, ci_lo, ci_hi)` of `stat` under the stationary bootstrap (12.5).

    `block=None` uses `default_block(n)`.  `one_sided=True` returns a one-sided LOWER bound (`ci_hi = +inf`).
    `method="studentised"` is the bootstrap-t: it needs a standard-error function for the statistic and defaults to the
    block-based `block_se` of the mean, which is what the pre-registered `d_t` test uses (12.3).
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    n = int(arr.size)
    if n < 2:
        raise EvalError(f"the bootstrap needs at least 2 observations, got {n}")
    alpha = _check_level(level)
    width = float(block) if block is not None else default_block(n)
    indices = stationary_bootstrap_indices(n, width, reps, rng)
    point = float(stat(arr))

    if method == "percentile":
        if stat is _mean:  # the pre-registered statistic: one vectorised pass instead of `reps` python calls
            draws = np.asarray(arr[indices].mean(axis=1), dtype=np.float64)
        else:
            draws = np.asarray([stat(_resampled(arr, indices[b])) for b in range(reps)], dtype=np.float64)
        usable = draws[np.isfinite(draws)]
        if usable.size == 0:
            raise EvalError("every bootstrap replicate was non-finite")
        if one_sided:
            return (point, float(np.quantile(usable, alpha)), float("inf"))
        return (point, float(np.quantile(usable, alpha / 2.0)), float(np.quantile(usable, 1.0 - alpha / 2.0)))

    if method != "studentised":
        raise EvalError(f"unknown interval method {method!r}")

    se_of = se_fn if se_fn is not None else (lambda series: block_se(series, width))
    se = se_of(arr)
    if not math.isfinite(se) or se <= 0.0:
        raise EvalError("the studentised bootstrap needs a positive standard-error estimate")
    tstats: list[float] = []
    for b in range(reps):
        sample = _resampled(arr, indices[b])
        se_b = se_of(sample)
        theta_b = float(stat(sample))
        if math.isfinite(se_b) and se_b > 0.0 and math.isfinite(theta_b):
            tstats.append((theta_b - point) / se_b)
    if not tstats:
        raise EvalError("every studentised bootstrap replicate was degenerate")
    t = np.asarray(tstats, dtype=np.float64)
    if one_sided:
        return (point, point - float(np.quantile(t, 1.0 - alpha)) * se, float("inf"))
    return (
        point,
        point - float(np.quantile(t, 1.0 - alpha / 2.0)) * se,
        point - float(np.quantile(t, alpha / 2.0)) * se,
    )


def lower_bound(
    x: npt.ArrayLike,
    *,
    alpha: float,
    block: float,
    reps: int,
    rng: np.random.Generator,
    interval: IntervalMethod,
    null_critical: float | None = None,
) -> float:
    """The pre-registered one-sided `(1 - alpha)` LOWER bound of `mean(d_t)` (12.1 `success`, 12.5).

    `interval` comes from the prereg file (`bootstrap.interval`, which must equal the size-validated `chosen_interval`
    of `prereg/power.v1.json`, 12.3):

    * `percentile`      - the `alpha` quantile of the stationary-bootstrap distribution of the mean;
    * `studentised`     - `mean - q_{1-alpha}(t*) * se`, `se` the block-based standard error (bootstrap-t);
    * `null_calibrated` - `mean - null_critical * se`, `null_critical` = the empirical `1 - alpha` quantile of the
      studentised statistic under the N1 nulls, taken from `prereg/power.v1.json`.  No bootstrap draw is needed.

    In every case the pre-registered rejection rule is the same: the look succeeds against a reference when this bound
    is `> 0`.
    """
    if not 0.0 < alpha < 1.0:
        raise EvalError(f"alpha must be in (0, 1): {alpha!r}")
    arr = np.asarray(x, dtype=np.float64).ravel()
    if int(arr.size) < 2:
        raise EvalError(f"the bootstrap needs at least 2 observations, got {arr.size}")

    if interval == "null_calibrated":
        if null_critical is None or not math.isfinite(null_critical):
            raise PreregError("interval 'null_calibrated' needs the critical value from prereg/power.v1.json")
        se = block_se(arr, block)
        if not math.isfinite(se) or se <= 0.0:
            raise EvalError("null-calibrated bound needs a positive standard-error estimate")
        return float(float(arr.mean()) - null_critical * se)

    if interval not in ("percentile", "studentised"):
        raise EvalError(f"unknown interval method {interval!r}")

    _, lo, _ = bootstrap_ci(
        arr,
        _mean,
        block=block,
        reps=reps,
        level=1.0 - alpha,
        rng=rng,
        one_sided=True,
        method=interval,
    )
    return lo


def paired_bootstrap_ci(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    stat: Callable[[FloatArray, FloatArray], float],
    *,
    block: float | None,
    reps: int,
    level: float,
    rng: np.random.Generator,
    one_sided: bool = False,
) -> tuple[float, float, float]:
    """Paired percentile interval: **the same resampling indices** are applied to both series (12.5).

    This is how the headline paired differences are formed (Jev versus baseline 3, B7.6): resampling the two series
    independently would destroy the pairing and widen the interval.
    """
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    if aa.size != bb.size:
        raise EvalError(f"paired series differ in length: {aa.size} vs {bb.size}")
    n = int(aa.size)
    if n < 2:
        raise EvalError(f"the bootstrap needs at least 2 observations, got {n}")
    alpha = _check_level(level)
    width = float(block) if block is not None else default_block(n)
    indices = stationary_bootstrap_indices(n, width, reps, rng)
    point = float(stat(aa, bb))
    draws = np.asarray([stat(_resampled(aa, indices[i]), _resampled(bb, indices[i])) for i in range(reps)], dtype=np.float64)
    usable = draws[np.isfinite(draws)]
    if usable.size == 0:
        raise EvalError("every paired bootstrap replicate was non-finite")
    if one_sided:
        return (point, float(np.quantile(usable, alpha)), float("inf"))
    return (point, float(np.quantile(usable, alpha / 2.0)), float(np.quantile(usable, 1.0 - alpha / 2.0)))


def cluster_bootstrap_ci(
    frame: pd.DataFrame,
    cluster_col: str,
    stat: Callable[[pd.DataFrame], float],
    *,
    reps: int,
    level: float,
    rng: np.random.Generator,
    one_sided: bool = False,
) -> tuple[float, float, float]:
    """Cluster bootstrap for trade-level statistics: whole clusters (entry-date cohorts) are resampled (12.5).

    Trades opened on one session share their entry snapshot, their candidate and their risk decision, so the cluster -
    not the trade - is the independent unit.
    """
    if cluster_col not in frame.columns:
        raise EvalError(f"frame has no cluster column {cluster_col!r}")
    if reps < 1:
        raise EvalError(f"reps must be >= 1: {reps!r}")
    alpha = _check_level(level)
    labels: list[Any] = sorted(set(frame[cluster_col].tolist()), key=repr)
    if len(labels) < 2:
        raise EvalError(f"the cluster bootstrap needs at least 2 clusters, got {len(labels)}")
    blocks: list[pd.DataFrame] = [frame[frame[cluster_col] == label] for label in labels]
    point = float(stat(frame))

    draws: list[float] = []
    for _ in range(reps):
        picks = rng.integers(0, len(blocks), size=len(blocks))
        sample = pd.concat([blocks[int(i)] for i in picks.tolist()], ignore_index=True)
        value = float(stat(sample))
        if math.isfinite(value):
            draws.append(value)
    if not draws:
        raise EvalError("every cluster bootstrap replicate was non-finite")
    values = np.asarray(draws, dtype=np.float64)
    if one_sided:
        return (point, float(np.quantile(values, alpha)), float("inf"))
    return (point, float(np.quantile(values, alpha / 2.0)), float(np.quantile(values, 1.0 - alpha / 2.0)))


def block_length_sample(indices: IntArray) -> FloatArray:
    """The realised block lengths of a resampling index matrix - the QC of the geometric block distribution (15.1).

    A fresh block that happens to start at `previous + 1` is indistinguishable from a continuation, so the measured
    mean is biased upward by O(1/n); the test allows for that.
    """
    lengths: list[int] = []
    for row in np.asarray(indices, dtype=np.int64):
        n = int(row.size)
        run = 1
        for j in range(1, n):
            if int(row[j]) == (int(row[j - 1]) + 1) % n:
                run += 1
            else:
                lengths.append(run)
                run = 1
        lengths.append(run)
    return np.asarray(lengths, dtype=np.float64)
