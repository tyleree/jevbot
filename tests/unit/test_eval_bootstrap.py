"""Unit tests for `jevbot.eval.bootstrap` (DESIGN.md 12.5, 12.3 size rule, 15.1 `eval/*` row; WP07).

The two acceptance tests of 15.1 / 12.3 are here:

* **CI coverage on AR(1)** - the stationary bootstrap holds its nominal coverage on a serially dependent series, the
  i.i.d. bootstrap does not;
* **the size check** - on a zero-mean `d_t` with the pre-registered 5-session overlap and cross-sectional correlation,
  the one-sided lower bound of the chosen interval method rejects at about `alpha`, while a deliberately mis-sized
  method (an i.i.d. bootstrap, `block = 1`) is detected as **over-sized** by the same check.

Everything is seeded; the Monte-Carlo tolerances are stated as multiples of the simulation standard error.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
import pytest

from jevbot.errors import EvalError, PreregError
from jevbot.eval.bootstrap import (
    block_length_sample,
    block_se,
    bootstrap_ci,
    bootstrap_seed,
    cluster_bootstrap_ci,
    default_block,
    lower_bound,
    paired_bootstrap_ci,
    rng_for,
    stationary_bootstrap_indices,
)

FloatArray = np.ndarray


# ======================================================================================================================
# resampling indices
# ======================================================================================================================


def test_indices_have_the_documented_shape_and_range() -> None:
    idx = stationary_bootstrap_indices(37, 8.0, 25, np.random.default_rng(1))
    assert idx.shape == (25, 37)
    assert idx.dtype == np.int64
    assert int(idx.min()) >= 0
    assert int(idx.max()) <= 36


def test_indices_are_deterministic_for_a_given_generator() -> None:
    a = stationary_bootstrap_indices(50, 6.0, 10, np.random.default_rng(99))
    b = stationary_bootstrap_indices(50, 6.0, 10, np.random.default_rng(99))
    assert np.array_equal(a, b)
    c = stationary_bootstrap_indices(50, 6.0, 10, np.random.default_rng(100))
    assert not np.array_equal(a, c)


def test_blocks_are_contiguous_and_wrap_circularly() -> None:
    """Politis-Romano: within a block the index advances by one, modulo n (12.5)."""
    n = 40
    idx = stationary_bootstrap_indices(n, 10.0, 200, np.random.default_rng(5))
    steps = (idx[:, 1:] - idx[:, :-1]) % n
    # every step is either a continuation (+1) or the start of a fresh block (any value)
    assert int((steps == 1).sum()) > int(0.7 * steps.size)
    assert bool(((idx >= 0) & (idx < n)).all())


@pytest.mark.parametrize("block", [2.0, 5.0, 12.0])
def test_block_lengths_are_geometric_with_mean_block(block: float) -> None:
    """The geometric block-length distribution of 12.5: `p = 1 / block`, so `E[L] = block`."""
    idx = stationary_bootstrap_indices(400, block, 400, np.random.default_rng(int(block) + 3))
    lengths = block_length_sample(idx)
    # a fresh block that happens to start at previous + 1 looks like a continuation: a small upward bias
    assert float(lengths.mean()) == pytest.approx(block, rel=0.12)


def test_block_one_is_the_iid_bootstrap() -> None:
    idx = stationary_bootstrap_indices(200, 1.0, 200, np.random.default_rng(8))
    lengths = block_length_sample(idx)
    assert float(lengths.mean()) == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize(("n", "block", "reps"), [(0, 5.0, 10), (10, 0.0, 10), (10, 5.0, 0), (10, float("nan"), 10)])
def test_indices_reject_impossible_arguments(n: int, block: float, reps: int) -> None:
    with pytest.raises(EvalError):
        stationary_bootstrap_indices(n, block, reps, np.random.default_rng(0))


# ======================================================================================================================
# block length default and seeds
# ======================================================================================================================


def test_default_block_is_the_documented_maximum() -> None:
    assert default_block(8) == 5.0  # the floor
    assert default_block(1000) == 10.0  # ceil(1000 ** (1/3))
    assert default_block(100, longest_horizon=5) == 10.0  # 2 x the longest horizon
    assert default_block(100, longest_horizon=20) == 40.0


def test_default_block_rejects_impossible_arguments() -> None:
    with pytest.raises(EvalError):
        default_block(0)
    with pytest.raises(EvalError):
        default_block(10, longest_horizon=-1)


def test_bootstrap_seed_is_the_documented_hash() -> None:
    expected = int.from_bytes(hashlib.sha256(b"trial-7|bootstrap|sharpe_orats").digest()[:8], "big")
    assert bootstrap_seed("trial-7", "sharpe_orats") == expected
    assert bootstrap_seed("trial-7", "sharpe_worst") != expected
    assert bootstrap_seed("trial-8", "sharpe_orats") != expected


def test_rng_for_is_reproducible() -> None:
    a = rng_for("trial-7", "brier").standard_normal(5)
    b = rng_for("trial-7", "brier").standard_normal(5)
    assert np.array_equal(a, b)


# ======================================================================================================================
# the long-run standard error
# ======================================================================================================================


def test_block_se_matches_the_iid_standard_error_on_white_noise() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal(4000)
    naive = float(x.std()) / math.sqrt(x.size)
    assert block_se(x, 5.0) == pytest.approx(naive, rel=0.15)


def test_block_se_grows_with_positive_autocorrelation() -> None:
    """An AR(1) with phi = 0.6 has a long-run variance 4x the marginal one, so the SE is about 2x the naive one."""
    rng = np.random.default_rng(3)
    x = _ar1(rng, n=8000, phi=0.6)
    naive = float(x.std()) / math.sqrt(x.size)
    assert block_se(x, 25.0) > 1.5 * naive


def test_block_se_of_a_constant_series_is_zero() -> None:
    assert block_se(np.zeros(100), 5.0) == 0.0
    assert math.isnan(block_se([1.0], 5.0))


# ======================================================================================================================
# confidence intervals
# ======================================================================================================================


def _ar1(rng: np.random.Generator, *, n: int, phi: float, mu: float = 0.0) -> FloatArray:
    eps = rng.standard_normal(n)
    out = np.empty(n, dtype=np.float64)
    value = 0.0
    for i in range(n):
        value = phi * value + eps[i]
        out[i] = value
    return np.asarray(out * math.sqrt(1.0 - phi * phi) + mu, dtype=np.float64)


def test_percentile_ci_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(0.5, 1.0, 300)
    point, lo, hi = bootstrap_ci(x, block=10.0, reps=800, level=0.95, rng=rng)
    assert point == pytest.approx(float(x.mean()))
    assert lo < point < hi


def test_one_sided_ci_has_an_infinite_upper_end() -> None:
    rng = np.random.default_rng(6)
    x = rng.normal(0.5, 1.0, 300)
    point, lo, hi = bootstrap_ci(x, block=10.0, reps=800, level=0.95, rng=rng, one_sided=True)
    assert math.isinf(hi)
    assert lo < point


def test_studentised_ci_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(10)
    x = _ar1(rng, n=400, phi=0.5, mu=0.3)
    point, lo, hi = bootstrap_ci(x, block=12.0, reps=600, level=0.95, rng=rng, method="studentised")
    assert lo < point < hi


def test_unknown_method_is_refused() -> None:
    rng = np.random.default_rng(12)
    with pytest.raises(EvalError):
        bootstrap_ci(rng.standard_normal(50), block=5.0, reps=10, level=0.95, rng=rng, method="bca")  # type: ignore[arg-type]


def test_ci_refuses_a_degenerate_sample_or_level() -> None:
    rng = np.random.default_rng(14)
    with pytest.raises(EvalError):
        bootstrap_ci([1.0], block=5.0, reps=10, level=0.95, rng=rng)
    with pytest.raises(EvalError):
        bootstrap_ci(rng.standard_normal(30), block=5.0, reps=10, level=1.0, rng=rng)


def test_ci_supports_an_arbitrary_statistic() -> None:
    rng = np.random.default_rng(16)
    x = rng.normal(0.0, 1.0, 400)
    point, lo, hi = bootstrap_ci(x, lambda v: float(np.median(v)), block=8.0, reps=400, level=0.9, rng=rng)
    assert point == pytest.approx(float(np.median(x)))
    assert lo < point < hi


# ======================================================================================================================
# coverage on AR(1) - 15.1 "bootstrap block-length distribution and CI coverage on AR(1)"
# ======================================================================================================================


def test_ci_coverage_on_ar1_is_nominal_for_the_block_bootstrap_and_broken_for_iid() -> None:
    n_sims, n, phi, level = 300, 400, 0.6, 0.90
    rng = np.random.default_rng(2026)
    covered_block = 0
    covered_iid = 0
    for _ in range(n_sims):
        x = _ar1(rng, n=n, phi=phi)  # true mean 0
        _, lo, hi = bootstrap_ci(x, block=12.0, reps=400, level=level, rng=rng)
        covered_block += int(lo <= 0.0 <= hi)
        _, lo_i, hi_i = bootstrap_ci(x, block=1.0, reps=400, level=level, rng=rng)
        covered_iid += int(lo_i <= 0.0 <= hi_i)

    se = math.sqrt(level * (1.0 - level) / n_sims)  # about 0.017
    block_rate = covered_block / n_sims
    iid_rate = covered_iid / n_sims
    assert block_rate == pytest.approx(level, abs=4.0 * se), f"block coverage {block_rate}"
    assert iid_rate < level - 0.08, f"the i.i.d. bootstrap should under-cover, got {iid_rate}"


# ======================================================================================================================
# the SIZE check of 12.3 / the prereg `size_rule`
# ======================================================================================================================


def _overlapping_correlated_d(rng: np.random.Generator, *, n: int, horizon: int = 5, n_underlyings: int = 3, rho: float = 0.85) -> FloatArray:
    """A zero-mean `d_t` with the pre-registered structure: 5-session overlap and ~0.85 cross-ETF correlation (12.3).

    `d_t` is the cross-sectional mean of per-underlying `horizon`-session moving averages of correlated shocks, so it
    carries exactly the positive autocorrelation that an i.i.d. bootstrap ignores.
    """
    total = n + horizon - 1
    common = rng.standard_normal(total)
    idio = rng.standard_normal((n_underlyings, total))
    shocks = math.sqrt(rho) * common[None, :] + math.sqrt(1.0 - rho) * idio
    kernel = np.full(horizon, 1.0 / horizon)
    smoothed = np.vstack([np.convolve(shocks[j], kernel, mode="valid") for j in range(n_underlyings)])
    return np.asarray(smoothed.mean(axis=0), dtype=np.float64)


def test_size_of_the_one_sided_bound_is_nominal_for_the_block_method_and_over_sized_for_iid() -> None:
    """12.3 / 15.1: nominal size for the chosen interval, `over-sized` detected for an i.i.d. bootstrap (block = 1)."""
    n_sims, n, alpha = 250, 320, 0.05
    rng = np.random.default_rng(4242)
    rejected_block = 0
    rejected_iid = 0
    for _ in range(n_sims):
        d = _overlapping_correlated_d(rng, n=n)  # true mean 0: every rejection is a false positive
        rejected_block += int(lower_bound(d, alpha=alpha, block=12.0, reps=400, rng=rng, interval="percentile") > 0.0)
        rejected_iid += int(lower_bound(d, alpha=alpha, block=1.0, reps=400, rng=rng, interval="percentile") > 0.0)

    # the prereg `size_rule`: empirical rejection rate <= alpha + 2 * sqrt(alpha (1 - alpha) / reps)
    tolerance = alpha + 3.0 * math.sqrt(alpha * (1.0 - alpha) / n_sims)
    block_rate = rejected_block / n_sims
    iid_rate = rejected_iid / n_sims
    assert block_rate <= tolerance + 0.02, f"block method size {block_rate}, budget {tolerance}"
    assert iid_rate > 0.15, f"the i.i.d. bootstrap must be flagged over-sized, got {iid_rate}"
    assert iid_rate > block_rate + 0.05


def test_the_bound_has_power_against_a_genuinely_positive_mean() -> None:
    """The size check must not be passed by a bound that never rejects anything."""
    n_sims, n, alpha = 120, 320, 0.05
    rng = np.random.default_rng(777)
    rejected = 0
    for _ in range(n_sims):
        d = _overlapping_correlated_d(rng, n=n) + 0.35
        rejected += int(lower_bound(d, alpha=alpha, block=12.0, reps=400, rng=rng, interval="percentile") > 0.0)
    assert rejected / n_sims > 0.5


# ======================================================================================================================
# lower_bound and its three interval methods
# ======================================================================================================================


def test_percentile_lower_bound_equals_the_alpha_quantile_of_the_bootstrap_means() -> None:
    rng_a = np.random.default_rng(31)
    rng_b = np.random.default_rng(31)
    x = np.random.default_rng(0).normal(0.2, 1.0, 200)
    got = lower_bound(x, alpha=0.05, block=10.0, reps=500, rng=rng_a, interval="percentile")
    indices = stationary_bootstrap_indices(x.size, 10.0, 500, rng_b)
    expected = float(np.quantile(x[indices].mean(axis=1), 0.05))
    assert got == pytest.approx(expected)


def test_studentised_lower_bound_is_below_the_point_estimate() -> None:
    rng = np.random.default_rng(33)
    x = _ar1(rng, n=300, phi=0.5, mu=0.4)
    bound = lower_bound(x, alpha=0.05, block=12.0, reps=500, rng=rng, interval="studentised")
    assert bound < float(x.mean())


def test_null_calibrated_lower_bound_is_mean_minus_critical_times_se() -> None:
    """`null_calibrated`: the critical value comes from `prereg/power.v1.json`, so `bound > 0` iff `t > c` (12.3)."""
    x = np.random.default_rng(35).normal(0.3, 1.0, 400)
    critical = 1.9
    got = lower_bound(x, alpha=0.05, block=10.0, reps=100, rng=np.random.default_rng(1), interval="null_calibrated", null_critical=critical)
    assert got == pytest.approx(float(x.mean()) - critical * block_se(x, 10.0))


def test_null_calibrated_without_a_critical_value_is_a_prereg_error() -> None:
    x = np.random.default_rng(37).normal(0.3, 1.0, 100)
    with pytest.raises(PreregError):
        lower_bound(x, alpha=0.05, block=10.0, reps=100, rng=np.random.default_rng(1), interval="null_calibrated")


def test_lower_bound_refuses_an_unknown_interval_or_alpha() -> None:
    x = np.random.default_rng(39).normal(0.3, 1.0, 100)
    with pytest.raises(EvalError):
        lower_bound(x, alpha=0.05, block=10.0, reps=50, rng=np.random.default_rng(1), interval="bca")  # type: ignore[arg-type]
    with pytest.raises(EvalError):
        lower_bound(x, alpha=0.0, block=10.0, reps=50, rng=np.random.default_rng(1), interval="percentile")


# ======================================================================================================================
# paired and cluster resampling
# ======================================================================================================================


def test_paired_bootstrap_uses_the_same_indices_for_both_series() -> None:
    """Identical series => the paired difference is exactly 0 in every replicate (12.5)."""
    rng = np.random.default_rng(41)
    a = rng.normal(0.0, 1.0, 200)
    point, lo, hi = paired_bootstrap_ci(
        a, a.copy(), lambda u, v: float(np.mean(u) - np.mean(v)), block=8.0, reps=300, level=0.95, rng=rng
    )
    assert (point, lo, hi) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_ci_covers_a_known_shift() -> None:
    rng = np.random.default_rng(43)
    a = rng.normal(0.0, 1.0, 400)
    b = a + 0.5
    point, lo, hi = paired_bootstrap_ci(a, b, lambda u, v: float(np.mean(v) - np.mean(u)), block=8.0, reps=400, level=0.95, rng=rng)
    assert point == pytest.approx(0.5)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_paired_bootstrap_refuses_mismatched_series() -> None:
    rng = np.random.default_rng(45)
    with pytest.raises(EvalError):
        paired_bootstrap_ci([1.0, 2.0, 3.0], [1.0, 2.0], lambda u, v: 0.0, block=5.0, reps=10, level=0.95, rng=rng)


def _trade_frame(rng: np.random.Generator, *, clusters: int, per_cluster: int) -> pd.DataFrame:
    rows = []
    for c in range(clusters):
        level = float(rng.normal(0.0, 1.0))  # the whole entry-date cohort shares its shock
        for _ in range(per_cluster):
            rows.append({"entry_session": f"2024-01-{c + 1:02d}", "pnl": level + float(rng.normal(0.0, 0.1))})
    return pd.DataFrame(rows)


def test_cluster_bootstrap_resamples_whole_entry_date_cohorts() -> None:
    rng = np.random.default_rng(47)
    frame = _trade_frame(rng, clusters=25, per_cluster=8)
    point, lo, hi = cluster_bootstrap_ci(
        frame, "entry_session", lambda f: float(f["pnl"].mean()), reps=400, level=0.95, rng=rng
    )
    assert point == pytest.approx(float(frame["pnl"].mean()))
    assert lo < point < hi
    # the cluster interval must be far wider than a naive i.i.d. one: 200 trades, but only 25 independent cohorts
    naive_half = 1.96 * float(frame["pnl"].std(ddof=1)) / math.sqrt(len(frame))
    assert (hi - lo) / 2.0 > 1.5 * naive_half


def test_cluster_bootstrap_refuses_a_missing_column_or_a_single_cluster() -> None:
    frame = pd.DataFrame({"entry_session": ["2024-01-01"] * 4, "pnl": [1.0, 2.0, 3.0, 4.0]})
    rng = np.random.default_rng(49)
    with pytest.raises(EvalError):
        cluster_bootstrap_ci(frame, "missing", lambda f: 0.0, reps=10, level=0.95, rng=rng)
    with pytest.raises(EvalError):
        cluster_bootstrap_ci(frame, "entry_session", lambda f: 0.0, reps=10, level=0.95, rng=rng)


def test_cluster_bootstrap_one_sided() -> None:
    rng = np.random.default_rng(51)
    frame = _trade_frame(rng, clusters=20, per_cluster=5)
    point, lo, hi = cluster_bootstrap_ci(
        frame, "entry_session", lambda f: float(f["pnl"].mean()), reps=200, level=0.95, rng=rng, one_sided=True
    )
    assert math.isinf(hi)
    assert lo < point
