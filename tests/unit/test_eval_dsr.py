"""Unit tests for `jevbot.eval.dsr` (DESIGN.md 12.6, 15.1 `eval/*` row; WP07).

Every identity is checked against a number computed **independently** of the module under test: the normal CDF /
quantile come from `statistics.NormalDist` (stdlib) rather than from scipy, the moments from explicit sums, and the
MinTRL worked example from Bailey & Lopez de Prado is pinned to its published value (2.73 years).
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

from jevbot.errors import EvalError
from jevbot.eval.dsr import (
    DERIVED_FAMILY_SUFFIXES,
    EULER_MASCHERONI,
    TRADING_DAYS_PER_YEAR,
    counts_as_selection_trial,
    deflated_sharpe,
    min_trl,
    moments,
    psr,
    psr_radicand,
    sample_kurtosis,
    sample_skew,
    sharpe,
    sr0,
    windows_overlap,
)

NORMAL = NormalDist()


# ======================================================================================================================
# moments: Pearson (non-excess) kurtosis
# ======================================================================================================================


def test_moments_match_explicit_sums() -> None:
    """`m3 / m2^1.5` and `m4 / m2^2` against sums written out by hand."""
    x = [0.01, -0.02, 0.03, 0.00, -0.01, 0.05]
    n = len(x)
    mean = sum(x) / n
    m2 = sum((v - mean) ** 2 for v in x) / n
    m3 = sum((v - mean) ** 3 for v in x) / n
    m4 = sum((v - mean) ** 4 for v in x) / n

    stats = moments(x)
    assert stats.n == n
    assert stats.mean == pytest.approx(mean)
    assert stats.sd == pytest.approx(math.sqrt(m2))
    assert stats.skew == pytest.approx(m3 / m2**1.5)
    assert stats.kurt == pytest.approx(m4 / m2**2)
    assert sharpe(x) == pytest.approx(mean / math.sqrt(m2))


def test_normal_sample_kurtosis_is_three_not_zero() -> None:
    """15.1: `kurt` of a large Normal sample is 3 +/- 0.1 - it is Pearson, never excess (12.6)."""
    rng = np.random.default_rng(20260917)
    sample = rng.standard_normal(500_000)
    assert sample_kurtosis(sample) == pytest.approx(3.0, abs=0.1)
    assert sample_skew(sample) == pytest.approx(0.0, abs=0.02)


def test_radicand_collapses_to_one_plus_half_sr_squared_for_a_normal_sample() -> None:
    """`1 - skew*SR + (kurt - 1)/4 * SR^2` with skew 0 / kurt 3 is `1 + SR^2/2` (12.6).

    With an *excess* kurtosis of 0 the same expression would be `1 - SR^2/4`: the mis-scaling the spec warns about.
    """
    sr = 0.13
    assert psr_radicand(sr, skew=0.0, kurt=3.0) == pytest.approx(1.0 + sr * sr / 2.0)
    assert psr_radicand(sr, skew=0.0, kurt=0.0) == pytest.approx(1.0 - sr * sr / 4.0)


def test_moments_of_a_constant_series_are_undefined_but_do_not_raise() -> None:
    stats = moments([0.01] * 20)
    assert stats.sd == 0.0
    assert math.isnan(stats.skew)
    assert math.isnan(stats.kurt)
    assert math.isnan(sharpe([0.01] * 20))


def test_moments_reject_non_finite_returns() -> None:
    with pytest.raises(EvalError):
        moments([0.01, float("nan")])


# ======================================================================================================================
# PSR
# ======================================================================================================================


def test_psr_matches_the_formula_evaluated_by_hand() -> None:
    sr, sr_star, t, skew, kurt = 0.09, 0.03, 500, -0.4, 4.2
    radicand = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    expected = NORMAL.cdf((sr - sr_star) * math.sqrt(t - 1) / math.sqrt(radicand))
    got = psr(sr, sr_star, t=t, skew=skew, kurt=kurt)
    assert got is not None
    assert got == pytest.approx(expected, abs=1e-12)


def test_psr_is_one_half_when_sr_equals_the_benchmark() -> None:
    got = psr(0.07, 0.07, t=300, skew=0.0, kurt=3.0)
    assert got == pytest.approx(0.5)


def test_psr_is_na_for_a_short_track_record_or_a_negative_radicand() -> None:
    assert psr(0.1, 0.0, t=1, skew=0.0, kurt=3.0) is None
    # skew * SR large enough to drive `1 - skew*SR + (kurt-1)/4 SR^2` below zero
    assert psr(0.5, 0.0, t=500, skew=4.0, kurt=3.0) is None


# ======================================================================================================================
# SR0 (the deflation term)
# ======================================================================================================================


def test_sr0_matches_the_formula_evaluated_by_hand() -> None:
    var_sr, n_trials = 0.0025, 12
    expected = math.sqrt(var_sr) * (
        (1.0 - EULER_MASCHERONI) * NORMAL.inv_cdf(1.0 - 1.0 / n_trials) + EULER_MASCHERONI * NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    )
    assert sr0(var_sr, n_trials) == pytest.approx(expected, abs=1e-9)


def test_sr0_grows_with_the_number_of_trials() -> None:
    assert sr0(0.0025, 2) < sr0(0.0025, 20) < sr0(0.0025, 200)


@pytest.mark.parametrize(("var_sr", "n_trials"), [(0.0025, 1), (0.0025, 0), (0.0, 10), (float("nan"), 10), (-1.0, 10)])
def test_sr0_guards_never_produce_minus_infinity(var_sr: float, n_trials: int) -> None:
    """12.6: `N < 2` or an undefined / zero `Var(SR_n)` => `SR0 = 0`, never `Z(0) = -inf`."""
    assert sr0(var_sr, n_trials) == 0.0


# ======================================================================================================================
# MinTRL - the worked example of Bailey & Lopez de Prado (12.6, G1)
# ======================================================================================================================


def test_min_trl_worked_example_is_about_2_73_years() -> None:
    """Annual Sharpe 2 against a benchmark of 1, 95% confidence, Normal daily returns => about 2.73 years (15.1)."""
    sr = 2.0 / math.sqrt(TRADING_DAYS_PER_YEAR)
    sr_ref = 1.0 / math.sqrt(TRADING_DAYS_PER_YEAR)

    observations = min_trl(sr, sr_ref, skew=0.0, kurt=3.0, conf=0.95)
    assert observations is not None

    # independent evaluation of the published formula
    z = NORMAL.inv_cdf(0.95)
    expected = 1.0 + (1.0 + 0.5 * sr * sr) * (z / (sr - sr_ref)) ** 2
    assert observations == pytest.approx(expected, rel=1e-12)

    # and the published number itself
    assert observations == pytest.approx(688.2, abs=0.5)
    assert observations / TRADING_DAYS_PER_YEAR == pytest.approx(2.73, abs=0.01)


def test_min_trl_worked_example_against_zero_needs_about_2_7_years() -> None:
    """G1's companion figure: Sharpe 1 versus 0 also needs about 2.7 years."""
    sr = 1.0 / math.sqrt(TRADING_DAYS_PER_YEAR)
    observations = min_trl(sr, 0.0, skew=0.0, kurt=3.0, conf=0.95)
    assert observations is not None
    assert observations / TRADING_DAYS_PER_YEAR == pytest.approx(2.7, abs=0.05)


def test_min_trl_uses_pearson_kurtosis() -> None:
    """Feeding excess kurtosis (0 for a Normal) shortens the requirement - the silent mis-scaling of 12.6."""
    sr = 2.0 / math.sqrt(TRADING_DAYS_PER_YEAR)
    sr_ref = 1.0 / math.sqrt(TRADING_DAYS_PER_YEAR)
    pearson = min_trl(sr, sr_ref, skew=0.0, kurt=3.0, conf=0.95)
    excess = min_trl(sr, sr_ref, skew=0.0, kurt=0.0, conf=0.95)
    assert pearson is not None and excess is not None
    assert pearson > excess


def test_min_trl_is_na_when_sr_does_not_beat_the_reference() -> None:
    assert min_trl(0.05, 0.05, skew=0.0, kurt=3.0, conf=0.95) is None
    assert min_trl(0.04, 0.05, skew=0.0, kurt=3.0, conf=0.95) is None


def test_min_trl_is_na_on_a_negative_radicand() -> None:
    assert min_trl(0.5, 0.0, skew=4.0, kurt=3.0, conf=0.95) is None


def test_min_trl_rejects_an_impossible_confidence() -> None:
    for conf in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(EvalError):
            min_trl(0.1, 0.0, skew=0.0, kurt=3.0, conf=conf)


# ======================================================================================================================
# deflated_sharpe: the assembled result and its guards
# ======================================================================================================================


def _series(rng: np.random.Generator, n: int = 750, mu: float = 0.0015, sd: float = 0.01) -> np.ndarray:
    return np.asarray(rng.normal(mu, sd, n), dtype=np.float64)


def test_deflated_sharpe_assembles_the_documented_quantities() -> None:
    rng = np.random.default_rng(7)
    returns = _series(rng)
    completed = [0.02, 0.05, 0.03, 0.07]
    result = deflated_sharpe(returns, n_trials=9, completed_sharpes=completed, n_failed=3, n_abandoned=2)

    stats = moments(returns)
    var_sr = float(np.var(np.asarray(completed), ddof=1))
    expected_sr0 = sr0(var_sr, 9)
    assert result.t == stats.n
    assert result.sr == pytest.approx(sharpe(returns))
    assert result.var_sr == pytest.approx(var_sr)
    assert result.sr0 == pytest.approx(expected_sr0)
    assert result.dsr == pytest.approx(psr(result.sr, expected_sr0, t=stats.n, skew=stats.skew, kurt=stats.kurt))
    assert result.psr_zero == pytest.approx(psr(result.sr, 0.0, t=stats.n, skew=stats.skew, kurt=stats.kurt))
    assert result.n_trials == 9
    assert result.n_completed == 4
    assert result.n_failed == 3
    assert result.n_abandoned == 2
    assert result.flags == ()


def test_deflation_lowers_the_probability_as_trials_accumulate() -> None:
    rng = np.random.default_rng(11)
    returns = _series(rng)
    completed = [0.02, 0.05, 0.03, 0.07]
    few = deflated_sharpe(returns, n_trials=2, completed_sharpes=completed)
    many = deflated_sharpe(returns, n_trials=200, completed_sharpes=completed)
    assert few.sr0 < many.sr0
    assert few.dsr is not None and many.dsr is not None
    assert few.dsr > many.dsr


def test_dsr_guard_for_fewer_than_two_trials() -> None:
    """12.6: `N < 2` => `SR0 = 0`, `DSR = PSR(0)`, flagged `dsr_trials<2`."""
    rng = np.random.default_rng(13)
    returns = _series(rng)
    result = deflated_sharpe(returns, n_trials=1, completed_sharpes=[0.03])
    assert result.sr0 == 0.0
    assert "dsr_trials<2" in result.flags
    assert result.dsr == pytest.approx(result.psr_zero)


def test_dsr_guard_for_an_undefined_variance_of_the_trial_sharpes() -> None:
    rng = np.random.default_rng(17)
    returns = _series(rng)
    result = deflated_sharpe(returns, n_trials=6, completed_sharpes=[0.03])  # one completed trial => Var undefined
    assert math.isnan(result.var_sr)
    assert result.sr0 == 0.0
    assert "dsr_var_sr_undefined" in result.flags
    assert result.dsr == pytest.approx(result.psr_zero)


def test_dsr_and_min_trl_report_na_on_a_one_observation_series() -> None:
    result = deflated_sharpe([0.01], n_trials=5, completed_sharpes=[0.01, 0.02])
    assert result.dsr is None
    assert result.min_trl is None
    assert "dsr_na" in result.flags
    assert "min_trl_na" in result.flags


def test_min_trl_years_uses_the_252_day_convention() -> None:
    rng = np.random.default_rng(19)
    returns = _series(rng, n=1200, mu=0.0012, sd=0.01)
    result = deflated_sharpe(returns, n_trials=4, completed_sharpes=[0.02, 0.04, 0.05, 0.03])
    assert result.min_trl is not None and result.min_trl_years is not None
    assert result.min_trl_years == pytest.approx(result.min_trl / TRADING_DAYS_PER_YEAR)


def test_rule_of_thumb_t_greater_than_three() -> None:
    rng = np.random.default_rng(23)
    strong = deflated_sharpe(_series(rng, n=1000, mu=0.0015, sd=0.01), n_trials=3, completed_sharpes=[0.1, 0.12, 0.09])
    weak = deflated_sharpe(_series(rng, n=1000, mu=0.00002, sd=0.01), n_trials=3, completed_sharpes=[0.1, 0.12, 0.09])
    assert strong.significant_rule_of_thumb
    assert not weak.significant_rule_of_thumb


def test_deflated_sharpe_rejects_a_negative_trial_count() -> None:
    with pytest.raises(EvalError):
        deflated_sharpe([0.01, 0.02], n_trials=-1, completed_sharpes=[])


# ======================================================================================================================
# the scope of N (12.6)
# ======================================================================================================================

_SCOPE = {"family": "exp1", "namespace": "ns", "reported_family": "exp1", "reported_namespace": "ns"}


@pytest.mark.parametrize("purpose", ["tune", "validate", "final"])
def test_selection_purposes_count(purpose: str) -> None:
    assert counts_as_selection_trial(purpose=purpose, flags=(), **_SCOPE)


@pytest.mark.parametrize("purpose", ["diagnostic", "reference", "", "smoke"])
def test_other_purposes_do_not_count(purpose: str) -> None:
    assert not counts_as_selection_trial(purpose=purpose, flags=(), **_SCOPE)


@pytest.mark.parametrize(
    "flag",
    [
        "baseline:4",
        "baseline:7",
        "placebo",
        "unmasked",
        "diagnostic",
        "shadow",
        "reference_history",
        "model_overlap",
        "ablation:buckets_only",
    ],
)
def test_excluded_flags_never_inflate_n(flag: str) -> None:
    """12.6: the ~1000 random-entry seeds would otherwise make `N` about 1000 and `SR0` meaningless."""
    assert not counts_as_selection_trial(purpose="tune", flags=(flag,), **_SCOPE)


@pytest.mark.parametrize("suffix", DERIVED_FAMILY_SUFFIXES)
def test_derived_families_never_count(suffix: str) -> None:
    assert not counts_as_selection_trial(
        purpose="final",
        flags=(),
        family=f"exp1{suffix}",
        namespace="ns",
        reported_family=f"exp1{suffix}",
        reported_namespace="ns",
    )


def test_another_family_or_namespace_does_not_count() -> None:
    assert not counts_as_selection_trial(
        purpose="tune", flags=(), family="exp2", namespace="ns", reported_family="exp1", reported_namespace="ns"
    )
    assert not counts_as_selection_trial(
        purpose="tune", flags=(), family="exp1", namespace="other", reported_family="exp1", reported_namespace="ns"
    )


def test_n_counts_failed_and_abandoned_trials() -> None:
    """12.6: a crashed or spend-stopped attempt was still a look at the data; `Var(SR_n)` uses the completed subset.

    The registry supplies the counts; this asserts the arithmetic the report prints and that `Var` is NOT taken over
    the failed rows (they have no Sharpe at all).
    """
    rows = [
        {"purpose": "tune", "status": "completed", "flags": (), "sharpe": 0.04},
        {"purpose": "tune", "status": "failed", "flags": (), "sharpe": None},
        {"purpose": "validate", "status": "abandoned", "flags": (), "sharpe": None},
        {"purpose": "final", "status": "completed", "flags": (), "sharpe": 0.06},
        {"purpose": "tune", "status": "completed", "flags": ("baseline:4",), "sharpe": 0.30},
        {"purpose": "diagnostic", "status": "completed", "flags": (), "sharpe": 0.90},
    ]
    scoped = [r for r in rows if counts_as_selection_trial(purpose=str(r["purpose"]), flags=tuple(r["flags"]), **_SCOPE)]  # type: ignore[arg-type]
    assert len(scoped) == 4  # 2 completed + 1 failed + 1 abandoned; the baseline and the diagnostic are out

    completed = [float(r["sharpe"]) for r in scoped if r["status"] == "completed"]  # type: ignore[arg-type]
    assert completed == [0.04, 0.06]

    rng = np.random.default_rng(29)
    result = deflated_sharpe(
        _series(rng),
        n_trials=len(scoped),
        completed_sharpes=completed,
        n_completed=2,
        n_failed=1,
        n_abandoned=1,
    )
    assert result.n_trials == 4
    assert (result.n_completed, result.n_failed, result.n_abandoned) == (2, 1, 1)
    assert result.var_sr == pytest.approx(float(np.var(np.asarray(completed), ddof=1)))
    assert result.flags == ()


def test_windows_overlap() -> None:
    from datetime import date

    assert windows_overlap(date(2024, 1, 1), date(2024, 6, 30), date(2024, 6, 30), date(2024, 12, 31))
    assert windows_overlap(date(2024, 1, 1), date(2024, 6, 30), date(2024, 2, 1), date(2024, 3, 1))
    assert not windows_overlap(date(2024, 1, 1), date(2024, 6, 29), date(2024, 6, 30), date(2024, 12, 31))
    assert windows_overlap("2024-01-01", "2024-06-30", "2024-05-01", "2024-08-01")
