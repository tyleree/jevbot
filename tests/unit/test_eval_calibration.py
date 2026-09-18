"""Unit tests for `jevbot.eval.calibration` (DESIGN.md 12.3, 12.1, 15.1 `eval/*` row; WP07).

The acceptance tests named in 12.1(c) / 15.1 live here:

* Brier / ECE / Murphy **identities** against hand-computed numbers (perfect and constant forecasters);
* the must-fix **regression test**: a constant climatological forecaster earns a positive BSS against the RAW
  option-implied probability on synthetic risk-premium data, yet **fails the pre-registered joint test** against the
  recalibrated implied probability and the expanding base rate;
* the **reference-history rules**: both references warm-start from the history under the purge rule, a reference below
  `reference_min_events` is `NaN` (never a raw-implied fallback), such sessions are ineligible and listed, and a look
  with too few eligible sessions raises `PreregError`;
* recalibration is **point in time** (future outcomes cannot change a past reference);
* the quantile-bin **tie rule** on 0.01-quantised data;
* history rows can never reach a scored table (only the two reference builders accept `history`).
"""

from __future__ import annotations

import inspect
import math
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from jevbot.errors import EvalError, PreregError
from jevbot.eval import calibration as cal
from jevbot.eval.bootstrap import lower_bound

EPOCH = date(2023, 1, 2)


def _day(i: int) -> date:
    return EPOCH + timedelta(days=i)


# ======================================================================================================================
# scores and identities
# ======================================================================================================================


def test_brier_of_a_perfect_forecaster_is_zero() -> None:
    assert cal.brier([1.0, 0.0, 1.0, 0.0], [1, 0, 1, 0]) == 0.0


def test_brier_of_a_constant_forecaster_matches_the_closed_form() -> None:
    """`BS(c) = base*(1-c)^2 + (1-base)*c^2` - written out by hand."""
    y = [1] * 11 + [0] * 89
    c = 0.11
    expected = 0.11 * (1 - c) ** 2 + 0.89 * c**2
    assert cal.brier([c] * 100, y) == pytest.approx(expected)


def test_brier_skill_is_one_minus_the_ratio() -> None:
    y = [1, 0, 1, 0, 1, 0]
    p = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3]
    ref = [0.5] * 6
    assert cal.brier_skill(p, y, ref) == pytest.approx(1.0 - cal.brier(p, y) / cal.brier(ref, y))


def test_brier_skill_is_nan_against_a_perfect_reference() -> None:
    assert math.isnan(cal.brier_skill([0.5, 0.5], [1, 0], [1.0, 0.0]))


def test_log_loss_clips_to_the_noul_resolution() -> None:
    """Nouls look clipped to [0.01, 0.99] (12.3): a 0 forecast on a realised event costs `-log(0.01)`."""
    assert cal.log_loss([0.0], [1]) == pytest.approx(-math.log(0.01))
    assert cal.log_loss([1.0], [0]) == pytest.approx(-math.log(0.01))
    assert cal.log_loss([0.5, 0.5], [1, 0]) == pytest.approx(-math.log(0.5))


def test_log_loss_rejects_an_impossible_eps() -> None:
    with pytest.raises(EvalError):
        cal.log_loss([0.5], [1], eps=0.0)


def test_scores_refuse_missing_or_void_rows() -> None:
    with pytest.raises(EvalError):
        cal.brier([0.5, float("nan")], [1, 0])
    with pytest.raises(EvalError):
        cal.brier([0.5, 0.5], [1, float("nan")])
    with pytest.raises(EvalError):
        cal.brier([0.5, 1.5], [1, 0])
    with pytest.raises(EvalError):
        cal.brier([0.5, 0.5], [1, 2])
    with pytest.raises(EvalError):
        cal.brier([0.5, 0.5, 0.5], [1, 0])


def test_sharpness_is_a_twenty_bin_histogram() -> None:
    counts = cal.sharpness([0.0, 0.049, 0.5, 0.99, 1.0])
    assert counts.shape == (20,)
    assert int(counts.sum()) == 5
    assert int(counts[0]) == 2  # 0.0 and 0.049
    assert int(counts[19]) == 2  # 0.99 and 1.0


def test_impute_missing_takes_the_reference_value() -> None:
    out = cal.impute_missing([0.3, float("nan"), 0.7], [0.5, 0.42, 0.5])
    assert out.tolist() == pytest.approx([0.3, 0.42, 0.7])


# ======================================================================================================================
# reliability, ECE, Murphy
# ======================================================================================================================


def test_reliability_table_bins_and_wilson_interval_by_hand() -> None:
    p = [0.2, 0.2, 0.8, 0.8]
    y = [0, 1, 1, 1]
    table = cal.reliability_table(p, y, strategy="fixed", min_per_bin=1)
    assert len(table.bins) == 2
    low, high = table.bins
    assert (low.lo, low.hi) == pytest.approx((0.2, 0.3))
    assert (low.n, low.mean_p, low.freq) == (2, pytest.approx(0.2), pytest.approx(0.5))
    assert (high.n, high.mean_p, high.freq) == (2, pytest.approx(0.8), pytest.approx(1.0))

    # Wilson score interval for 1 success out of 2 at 95%, written out
    z = 1.959963984540054
    denom = 2 + z * z
    centre = (1 + z * z / 2) / denom
    half = (z / denom) * math.sqrt(1 * 1 / 2 + z * z / 4)
    assert low.ci_lo == pytest.approx(centre - half)
    assert low.ci_hi == pytest.approx(centre + half)


def test_ece_and_mce_by_hand() -> None:
    table = cal.reliability_table([0.2, 0.2, 0.8, 0.8], [0, 1, 1, 1], strategy="fixed", min_per_bin=1)
    # |0.2 - 0.5| = 0.3 on 2 rows, |0.8 - 1.0| = 0.2 on 2 rows
    assert cal.ece(table) == pytest.approx((2 * 0.3 + 2 * 0.2) / 4)
    assert cal.mce(table) == pytest.approx(0.3)


def test_murphy_decomposition_by_hand_without_a_within_bin_residual() -> None:
    p = [0.2, 0.2, 0.8, 0.8]
    y = [0, 1, 1, 1]
    table = cal.reliability_table(p, y, strategy="fixed", min_per_bin=1)
    rel, res, unc, residual = cal.murphy(table, y)
    assert rel == pytest.approx(0.065)
    assert res == pytest.approx(0.0625)
    assert unc == pytest.approx(0.1875)
    assert residual == pytest.approx(0.0, abs=1e-15)
    assert rel - res + unc + residual == pytest.approx(cal.brier(p, y))
    assert cal.brier(p, y) == pytest.approx(0.19)


def test_murphy_residual_is_exact_when_a_bin_holds_distinct_forecasts() -> None:
    """One bin, two different forecasts: the residual is the term that keeps `BS = REL - RES + UNC + residual` exact."""
    p = [0.2, 0.8]
    y = [0, 1]
    table = cal.reliability_table(p, y, n_bins=1, strategy="fixed", min_per_bin=1)
    assert len(table.bins) == 1
    rel, res, unc, residual = cal.murphy(table, y)
    assert rel == pytest.approx(0.0)
    assert res == pytest.approx(0.0)
    assert unc == pytest.approx(0.25)
    assert residual == pytest.approx(-0.21)
    assert rel - res + unc + residual == pytest.approx(cal.brier(p, y)) == pytest.approx(0.04)


@pytest.mark.parametrize("strategy", ["quantile", "fixed"])
def test_murphy_identity_holds_on_random_data(strategy: str) -> None:
    rng = np.random.default_rng(11)
    p = np.round(rng.uniform(0.01, 0.99, 900), 2)
    y = (rng.random(900) < p).astype(int)
    table = cal.reliability_table(p, y, strategy=strategy)  # type: ignore[arg-type]
    rel, res, unc, residual = cal.murphy(table, y)
    assert rel - res + unc + residual == pytest.approx(cal.brier(p, y), abs=1e-12)


def test_murphy_of_a_perfect_forecaster() -> None:
    p = [1.0, 0.0, 1.0, 0.0]
    y = [1, 0, 1, 0]
    table = cal.reliability_table(p, y, strategy="fixed", min_per_bin=1)
    rel, res, unc, residual = cal.murphy(table, y)
    assert rel == pytest.approx(0.0)
    assert unc == pytest.approx(0.25)
    assert res == pytest.approx(0.25)  # the forecaster fully resolves the uncertainty
    assert rel - res + unc + residual == pytest.approx(0.0)


def test_murphy_refuses_a_mismatched_outcome_vector() -> None:
    table = cal.reliability_table([0.2, 0.8], [0, 1], n_bins=1, strategy="fixed", min_per_bin=1)
    with pytest.raises(EvalError):
        cal.murphy(table, [0, 1, 1])


def test_reliability_table_refuses_bad_arguments() -> None:
    with pytest.raises(EvalError):
        cal.reliability_table([0.5], [1], n_bins=0)
    with pytest.raises(EvalError):
        cal.reliability_table([0.5], [1], min_per_bin=0)
    with pytest.raises(EvalError):
        cal.reliability_table([0.5], [1], level=1.0)
    with pytest.raises(EvalError):
        cal.reliability_table([0.5], [1], strategy="deciles")  # type: ignore[arg-type]


def test_reliability_rows_are_serialisable() -> None:
    table = cal.reliability_table([0.2, 0.2, 0.8, 0.8], [0, 1, 1, 1], strategy="fixed", min_per_bin=1)
    rows = table.rows()
    assert [r["bin"] for r in rows] == [0.0, 1.0]
    assert set(rows[0]) == {"bin", "lo", "hi", "n", "mean_p", "freq", "ci_lo", "ci_hi"}


# ----------------------------------------------------------------------------------------------------------------------
# the TIE RULE of 12.3
# ----------------------------------------------------------------------------------------------------------------------


def test_tie_rule_never_splits_equal_forecasts_across_two_bins() -> None:
    """0.01-quantised, [0.01, 0.99]-clipped outputs: equal `p` must always land in ONE bin (12.3)."""
    rng = np.random.default_rng(3)
    p = np.clip(np.round(rng.normal(0.2, 0.12, 2000), 2), 0.01, 0.99)
    y = (rng.random(2000) < p).astype(int)
    table = cal.reliability_table(p, y, n_bins=10, strategy="quantile", min_per_bin=20)

    assert not table.fell_back
    assert len(table.bins) >= 3
    frame = pd.DataFrame({"p": p, "bin": table.bin_of})
    per_value = frame.groupby("p")["bin"].nunique()
    assert int(per_value.max()) == 1, "a quantised forecast value straddled two bins"
    assert all(b.n >= 20 for b in table.bins)
    # bins are ordered and do not overlap
    assert [b.lo for b in table.bins] == sorted(b.lo for b in table.bins)
    for left, right in zip(table.bins, table.bins[1:], strict=False):
        assert left.hi < right.lo


def test_tie_rule_moves_an_edge_forward_to_the_next_distinct_value() -> None:
    p = [0.1] * 40 + [0.2] * 40 + [0.3] * 40
    y = [0] * 40 + [1] * 20 + [0] * 20 + [1] * 40
    table = cal.reliability_table(p, y, n_bins=3, strategy="quantile", min_per_bin=10)
    assert not table.fell_back
    assert [b.n for b in table.bins] == [40, 40, 40]
    assert [(b.lo, b.hi) for b in table.bins] == [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)]
    assert [b.freq for b in table.bins] == pytest.approx([0.0, 0.5, 1.0])


def test_fewer_than_three_bins_falls_back_to_fixed_edges() -> None:
    """12.3: fewer than 3 bins => fall back to fixed edges 0, 0.1, ..., 1."""
    p = [0.15] * 60 + [0.85] * 60
    y = [0] * 60 + [1] * 60
    table = cal.reliability_table(p, y, n_bins=10, strategy="quantile", min_per_bin=20)
    assert table.fell_back
    assert len(table.bins) == 2
    assert [(b.lo, b.hi) for b in table.bins] == [(pytest.approx(0.1), pytest.approx(0.2)), (pytest.approx(0.8), pytest.approx(0.9))]


def _quantile_bin_sizes(p: list[float], *, n_bins: int, min_per_bin: int) -> list[int]:
    table = cal.reliability_table(p, [0] * len(p), n_bins=n_bins, strategy="quantile", min_per_bin=min_per_bin)
    assert not table.fell_back
    return [b.n for b in table.bins]


def test_an_under_sized_first_bin_merges_forward() -> None:
    p = [0.1] * 10 + [0.2] * 3 + [0.3] * 40 + [0.4] * 40 + [0.5] * 40
    assert _quantile_bin_sizes(p, n_bins=44, min_per_bin=20) == [53, 40, 40]


def test_an_under_sized_middle_bin_merges_with_its_smaller_neighbour() -> None:
    """Right neighbour smaller => merge right; left neighbour smaller => merge left."""
    merge_right = [0.1] * 40 + [0.2] * 3 + [0.3] * 3 + [0.4] * 40 + [0.5] * 40
    assert _quantile_bin_sizes(merge_right, n_bins=42, min_per_bin=20) == [40, 46, 40]

    merge_left = [0.1] * 30 + [0.2] * 3 + [0.3] * 40 + [0.4] * 40
    assert _quantile_bin_sizes(merge_left, n_bins=37, min_per_bin=20) == [33, 40, 40]


def test_an_under_sized_last_bin_merges_backward() -> None:
    p = [0.1] * 40 + [0.2] * 40 + [0.3] * 40 + [0.4] * 2
    assert _quantile_bin_sizes(p, n_bins=41, min_per_bin=20) == [40, 40, 42]


def test_both_the_quantile_and_the_fixed_width_ece_are_available() -> None:
    rng = np.random.default_rng(5)
    p = np.round(rng.uniform(0.02, 0.98, 1200), 2)
    y = (rng.random(1200) < p).astype(int)
    quantile = cal.ece(cal.reliability_table(p, y, strategy="quantile"))
    fixed = cal.ece(cal.reliability_table(p, y, strategy="fixed"))
    assert math.isfinite(quantile)
    assert math.isfinite(fixed)
    assert quantile < 0.1 and fixed < 0.1  # a well-calibrated generator


# ======================================================================================================================
# coherence
# ======================================================================================================================


def test_coherence_of_an_exhaustive_triplet_is_zero() -> None:
    down = [0.10, 0.20, 0.05]
    up = [0.12, 0.10, 0.07]
    inside = [0.78, 0.70, 0.88]
    stats = cal.coherence(down, up, inside)
    assert stats.n == 3
    assert stats.mean_abs_error == pytest.approx(0.0, abs=1e-15)
    assert stats.max_abs_error == pytest.approx(0.0, abs=1e-15)
    assert stats.frac_within_tol == 1.0


def test_coherence_reports_the_per_event_error_by_hand() -> None:
    stats = cal.coherence([0.10, 0.20], [0.12, 0.10], [0.75, 0.75])
    assert stats.errors.tolist() == pytest.approx([0.03, 0.05])
    assert stats.mean_abs_error == pytest.approx(0.04)
    assert stats.max_abs_error == pytest.approx(0.05)
    assert stats.frac_within_tol == pytest.approx(0.5)  # tol 0.03: only the first triplet is inside


def test_coherence_ignores_missing_components() -> None:
    stats = cal.coherence([0.1, float("nan")], [0.1, 0.1], [0.8, 0.8])
    assert stats.n == 1
    assert stats.mean_abs_error == pytest.approx(0.0, abs=1e-15)


# ======================================================================================================================
# isotonic regression
# ======================================================================================================================


def test_pav_leaves_a_monotone_sequence_untouched() -> None:
    fit = cal.pav_isotonic([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    assert fit([0.0, 1.0, 2.0]).tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_pav_pools_adjacent_violators_by_hand() -> None:
    assert cal.pav_isotonic([0.0, 1.0, 2.0], [3.0, 1.0, 2.0])([0.0, 1.0, 2.0]).tolist() == pytest.approx([2.0, 2.0, 2.0])
    assert cal.pav_isotonic([0.0, 1.0, 2.0], [1.0, 3.0, 2.0])([0.0, 1.0, 2.0]).tolist() == pytest.approx([1.0, 2.5, 2.5])


def test_pav_aggregates_duplicate_x_with_their_weights() -> None:
    """Duplicate `x` are averaged first, so the fit is a genuine function of `x`."""
    fit = cal.pav_isotonic([0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0])
    assert float(fit([0.0])[0]) == pytest.approx(2.0 / 3.0)
    assert float(fit([1.0])[0]) == pytest.approx(1.0)


def test_pav_is_monotone_and_clipped_outside_the_fitted_range() -> None:
    rng = np.random.default_rng(7)
    x = np.sort(rng.uniform(0.0, 1.0, 400))
    y = (rng.random(400) < x).astype(float)
    fit = cal.pav_isotonic(x, y)
    grid = np.linspace(-1.0, 2.0, 200)
    values = fit(grid)
    assert bool((np.diff(values) >= -1e-12).all())
    assert float(values[0]) == pytest.approx(float(fit([float(x[0])])[0]))
    assert float(values[-1]) == pytest.approx(float(fit([float(x[-1])])[0]))


def test_pav_refuses_empty_or_non_finite_input() -> None:
    with pytest.raises(EvalError):
        cal.pav_isotonic([], [])
    with pytest.raises(EvalError):
        cal.pav_isotonic([0.0, 1.0], [0.0, float("nan")])


# ======================================================================================================================
# question kinds and the session grid
# ======================================================================================================================


@pytest.mark.parametrize(
    ("question_id", "expected"),
    [
        ("eval.down_1em_1s", "eval.down_1em"),
        ("eval.down_1em_5s", "eval.down_1em"),
        ("eval.down_1em_hold", "eval.down_1em"),
        ("eval.up_1s", "eval.up"),
        ("eval.rv_gt_iv_5s", "eval.rv_gt_iv"),
        ("under.direction#bullish", "under.direction#bullish"),
    ],
)
def test_question_kind_strips_only_the_horizon_suffix(question_id: str, expected: str) -> None:
    assert cal.question_kind(question_id) == expected


def test_session_grid_is_the_union_of_sessions_and_resolutions() -> None:
    events = pd.DataFrame({"session": [_day(3), _day(4)], "resolved_on": [_day(4), _day(5)]})
    history = pd.DataFrame({"session": [_day(0)], "resolved_on": [_day(1)]})
    assert cal.session_grid(events, history) == [_day(0), _day(1), _day(3), _day(4), _day(5)]


# ======================================================================================================================
# frames for the reference tests
# ======================================================================================================================

QUESTIONS: tuple[str, ...] = ("eval.down_1em_1s", "eval.up_1em_1s")
UNDERLYINGS: tuple[str, ...] = ("SPY", "QQQ", "IWM")
HORIZON = 1
# the variance risk premium: the risk-neutral implied probability sits well above the physical one (12.1, V11)
IMPLIED_LEVELS: tuple[float, ...] = (0.16, 0.20, 0.24)
TRUE_P: dict[float, float] = {0.16: 0.09, 0.20: 0.11, 0.24: 0.13}


def _history_frame(rng: np.random.Generator, *, n_sessions: int, start: int = 0) -> pd.DataFrame:
    """The reference history: `question_id, horizon, session, p_implied, y, resolved_on` (12.3) - no Jev output."""
    rows: list[dict[str, Any]] = []
    for i in range(n_sessions):
        for question_id in QUESTIONS:
            implied = float(rng.choice(IMPLIED_LEVELS))
            rows.append(
                {
                    "question_id": question_id,
                    "horizon": HORIZON,
                    "session": _day(start + i),
                    "p_implied": implied,
                    "y": int(rng.random() < TRUE_P[implied]),
                    "resolved_on": _day(start + i + HORIZON),
                }
            )
    return pd.DataFrame(rows)


def _events_frame(
    rng: np.random.Generator,
    *,
    n_sessions: int,
    start: int,
    p_forecast: float,
    questions: tuple[str, ...] = QUESTIONS,
    underlyings: tuple[str, ...] = UNDERLYINGS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i in range(n_sessions):
        for question_id in questions:
            for underlying in underlyings:
                implied = float(rng.choice(IMPLIED_LEVELS))
                rows.append(
                    {
                        "session": _day(start + i),
                        "question_id": question_id,
                        "horizon": HORIZON,
                        "underlying": underlying,
                        "with_text": True,
                        "p": p_forecast,
                        "p_implied": implied,
                        "y": int(rng.random() < TRUE_P[implied]),
                        "resolved_on": _day(start + i + HORIZON),
                        "event_key": f"e{start + i}-{question_id}-{underlying}",
                    }
                )
    return pd.DataFrame(rows)


# ======================================================================================================================
# base_rate_expanding: the purge and the min-events rule
# ======================================================================================================================


def test_base_rate_is_nan_below_min_events_and_available_exactly_when_the_purge_allows() -> None:
    """12.1: below `reference_min_events` a reference is UNAVAILABLE (NaN), never a noisy early estimate.

    With a Tier-A-only history (3 underlyings per session, horizon 1) the purge makes `3 * (i - 1)` training events
    available at grid position `i`, so 250 events are first reached at `i = 85` - the "about 84 sessions" of 12.1.
    """
    rng = np.random.default_rng(101)
    events = _events_frame(rng, n_sessions=100, start=0, p_forecast=0.12, questions=(QUESTIONS[0],))
    values = cal.base_rate_expanding(events, None, min_events=250)

    positions = {day: i for i, day in enumerate(sorted(set(events["session"].tolist())))}
    finite = [positions[row.session] for row, ok in zip(events.itertuples(), np.isfinite(values), strict=True) if ok]
    assert min(finite) == 85
    assert max(finite) == 99
    assert all(math.isnan(v) for v, day in zip(values, events["session"], strict=True) if positions[day] < 85)


def test_base_rate_counts_history_rows_and_own_events_under_one_purge() -> None:
    """The expanding base rate = history rows + the run's own earlier events, all purged (12.1 `own_events`)."""
    rng = np.random.default_rng(103)
    history = _history_frame(rng, n_sessions=300, start=0)
    events = _events_frame(rng, n_sessions=20, start=301, p_forecast=0.12)
    values = cal.base_rate_expanding(events, history, min_events=250)
    assert bool(np.isfinite(values).all())

    # the first event session sees exactly the history rows of its own question, nothing of the run itself
    first_session = _day(301)
    first_row = int(np.flatnonzero((events["session"] == first_session) & (events["question_id"] == QUESTIONS[0]))[0])
    training = history[history["question_id"] == QUESTIONS[0]]
    assert values[first_row] == pytest.approx(float(training["y"].mean()))


def test_base_rate_purge_excludes_outcomes_resolved_after_prev_session() -> None:
    """A pair is usable only if `resolved_on <= prev_session(D, h)` - a later resolution must not move the reference."""
    rng = np.random.default_rng(105)
    history = _history_frame(rng, n_sessions=300, start=0)
    events = _events_frame(rng, n_sessions=20, start=301, p_forecast=0.12)
    base = cal.base_rate_expanding(events, history, min_events=250)

    poisoned = events.copy()
    last_session = _day(320)
    poisoned.loc[poisoned["session"] >= last_session, "y"] = 1  # rewrite the FUTURE outcomes
    after = cal.base_rate_expanding(poisoned, history, min_events=250)

    early = np.asarray(events["session"] < last_session)
    assert np.allclose(base[early], after[early], equal_nan=True)


def test_base_rate_is_nan_when_the_grid_does_not_reach_back() -> None:
    rng = np.random.default_rng(107)
    events = _events_frame(rng, n_sessions=3, start=0, p_forecast=0.12)
    assert bool(np.isnan(cal.base_rate_expanding(events, None, min_events=1)[:6]).all())


def test_base_rate_in_sample_is_the_whole_window_frequency() -> None:
    rng = np.random.default_rng(109)
    events = _events_frame(rng, n_sessions=30, start=0, p_forecast=0.12)
    values = cal.base_rate_in_sample(events)
    for question_id in QUESTIONS:
        rows = np.asarray(events["question_id"] == question_id)
        assert values[rows].tolist() == pytest.approx([float(events.loc[rows, "y"].mean())] * int(rows.sum()))


def test_void_and_open_outcomes_are_never_training_pairs() -> None:
    """A void outcome (`y = None`, 2.7) and a still-open forecast (no `resolved_on`) cannot train a reference."""
    rng = np.random.default_rng(106)
    history = _history_frame(rng, n_sessions=260, start=0)
    events = _events_frame(rng, n_sessions=5, start=261, p_forecast=0.12)

    clean = cal.base_rate_expanding(events, history, min_events=250)
    assert bool(np.isfinite(clean).all())

    voided = history.copy()
    voided.loc[voided.index[:20], "y"] = None  # 20 void outcomes disappear from the training set
    fewer = cal.base_rate_expanding(events, voided, min_events=250)
    assert not np.allclose(clean, fewer)

    opened = history.copy()
    opened.loc[opened.index[:20], "resolved_on"] = None  # still open: also excluded
    assert np.allclose(cal.base_rate_expanding(events, opened, min_events=250), fewer, equal_nan=True)


def test_timestamp_typed_session_columns_are_accepted() -> None:
    """`eval.load` may hand over pandas datetime64 columns instead of `datetime.date` objects."""
    rng = np.random.default_rng(108)
    history = _history_frame(rng, n_sessions=260, start=0)
    events = _events_frame(rng, n_sessions=5, start=261, p_forecast=0.12)
    as_dates = cal.base_rate_expanding(events, history, min_events=250)

    stamped_events = events.assign(session=pd.to_datetime(events["session"]), resolved_on=pd.to_datetime(events["resolved_on"]))
    stamped_history = history.assign(session=pd.to_datetime(history["session"]), resolved_on=pd.to_datetime(history["resolved_on"]))
    assert np.allclose(cal.base_rate_expanding(stamped_events, stamped_history, min_events=250), as_dates, equal_nan=True)


def _stamped(frame: pd.DataFrame) -> pd.DataFrame:
    """The same frame with datetime64 `session` / `resolved_on` columns, as `eval.load` hands them over."""
    return frame.assign(
        session=pd.to_datetime(frame["session"]),
        resolved_on=pd.to_datetime(frame["resolved_on"]),
        y=frame["y"].astype(float),
    )


def test_still_open_rows_are_missing_dates_not_dates_on_a_datetime64_frame() -> None:
    """A still-open forecast is `NaT` on a datetime64 `resolved_on` column - and `NaT` is NOT a date.

    `NaTType` subclasses `datetime.datetime`, so a plain `isinstance(value, date)` test accepts it, the sorted session
    grid then dies with a raw `TypeError: Cannot compare NaT with datetime.date object`, and the caller gets no
    diagnostic at all.  12.3 has the report state "N resolved / open / void / missing", so open rows are a normal part
    of the calibration frame and every date reader of the module must treat `NaT` exactly like `None`.
    """
    rng = np.random.default_rng(112)
    history = _stamped(_history_frame(rng, n_sessions=260, start=0))
    events = _stamped(_events_frame(rng, n_sessions=5, start=261, p_forecast=0.12))

    open_rows = np.asarray(events["session"] == events["session"].max())
    events.loc[open_rows, "resolved_on"] = pd.NaT  # the last session has not resolved yet
    events.loc[open_rows, "y"] = np.nan

    grid = cal.session_grid(events, history)
    assert grid == sorted(set(grid))
    assert all(isinstance(day, date) and not pd.isna(day) for day in grid)

    # the same frame with the open rows spelled `None` on an object column must give the identical answer
    as_objects = events.assign(
        session=[v.date() for v in events["session"]],
        resolved_on=[None if pd.isna(v) else v.date() for v in events["resolved_on"]],
    )
    history_objects = history.assign(
        session=[v.date() for v in history["session"]], resolved_on=[v.date() for v in history["resolved_on"]]
    )
    for min_events in (250,):
        assert np.allclose(
            cal.base_rate_expanding(events, history, min_events=min_events),
            cal.base_rate_expanding(as_objects, history_objects, min_events=min_events),
            equal_nan=True,
        )
        assert np.allclose(
            cal.recalibrate_walkforward(events, history, min_events=min_events, refit_sessions=21),
            cal.recalibrate_walkforward(as_objects, history_objects, min_events=min_events, refit_sessions=21),
            equal_nan=True,
        )
    refs = cal.build_references(events, history, recal_min_events=250, base_rate_min_events=250, refit_sessions=21)
    eligible = cal.eligible_sessions(events, {k: refs[k] for k in ("implied_recalibrated", "base_rate_expanding")}, QUESTIONS)
    assert list(eligible) == sorted(set(v.date() for v in events["session"]))


def test_a_missing_session_never_becomes_an_eligible_day() -> None:
    """The counterpart on the `session` column: a row without a session is aligned positionally and simply never on a day."""
    rng = np.random.default_rng(114)
    history = _stamped(_history_frame(rng, n_sessions=260, start=0))
    events = _stamped(_events_frame(rng, n_sessions=4, start=261, p_forecast=0.12))
    events.loc[events.index[0], "session"] = pd.NaT

    assert len(cal.unique_sessions(events["session"].tolist())) == 3
    refs = cal.build_references(events, history, recal_min_events=250, base_rate_min_events=250, refit_sessions=21)
    for name in ("implied_recalibrated", "base_rate_expanding"):
        assert len(refs[name]) == len(events)  # positional alignment survives the missing date
    eligible = cal.eligible_sessions(events, {k: refs[k] for k in ("implied_recalibrated", "base_rate_expanding")}, QUESTIONS)
    assert len(eligible) == 3
    assert not any(pd.isna(day) for day in eligible)

    d = cal.loss_differential(refs["base_rate_expanding"], events["p"], events["y"], events["session"], events["question_id"])
    assert d.size == 3


def test_an_explicit_session_grid_overrides_the_derived_one() -> None:
    """A caller holding the real `Calendar` can pass the trading-session grid instead of the data-derived one."""
    rng = np.random.default_rng(110)
    events = _events_frame(rng, n_sessions=100, start=0, p_forecast=0.12, questions=(QUESTIONS[0],))
    derived = cal.base_rate_expanding(events, None, min_events=250)
    explicit = cal.base_rate_expanding(events, None, min_events=250, sessions=cal.session_grid(events))
    assert np.allclose(derived, explicit, equal_nan=True)

    # a coarser grid moves `prev_session(D, h)` further back, so a different training set is in scope
    sparse = cal.session_grid(events)[::2]
    on_sparse = cal.base_rate_expanding(events, None, min_events=250, sessions=sparse)
    assert not np.allclose(derived, on_sparse, equal_nan=True)


def test_base_rate_rejects_an_impossible_min_events() -> None:
    rng = np.random.default_rng(111)
    with pytest.raises(EvalError):
        cal.base_rate_expanding(_events_frame(rng, n_sessions=2, start=0, p_forecast=0.1), None, min_events=0)


# ======================================================================================================================
# recalibrate_walkforward: PIT, min events, no fallback
# ======================================================================================================================


def test_recalibration_maps_implied_to_the_physical_frequency() -> None:
    rng = np.random.default_rng(113)
    history = _history_frame(rng, n_sessions=1200, start=0)
    events = _events_frame(rng, n_sessions=20, start=1201, p_forecast=0.12)
    values = cal.recalibrate_walkforward(events, history, min_events=250, refit_sessions=21)
    assert bool(np.isfinite(values).all())

    implied = np.asarray(events["p_implied"], dtype=float)
    for level in IMPLIED_LEVELS:
        rows = np.isclose(implied, level)
        assert float(values[rows].mean()) == pytest.approx(TRUE_P[level], abs=0.03)
    # monotone in the implied probability, as an isotonic map must be
    means = [float(values[np.isclose(implied, level)].mean()) for level in IMPLIED_LEVELS]
    assert means == sorted(means)


def test_recalibration_is_nan_below_min_events_and_never_falls_back_to_raw_implied() -> None:
    """12.1 / 12.3: there is NO fallback to raw implied - below `min_events` the reference is simply unavailable."""
    rng = np.random.default_rng(115)
    history = _history_frame(rng, n_sessions=40, start=0)
    events = _events_frame(rng, n_sessions=20, start=41, p_forecast=0.12)
    values = cal.recalibrate_walkforward(events, history, min_events=250, refit_sessions=21)
    assert bool(np.isnan(values).all())
    assert not np.allclose(np.nan_to_num(values), np.asarray(events["p_implied"], dtype=float))


def test_recalibration_uses_only_pre_forecast_outcomes() -> None:
    """Recalibration PIT: rewriting the outcomes of later sessions cannot change an earlier reference (12.3)."""
    rng = np.random.default_rng(117)
    history = _history_frame(rng, n_sessions=400, start=0)
    events = _events_frame(rng, n_sessions=60, start=401, p_forecast=0.12)
    base = cal.recalibrate_walkforward(events, history, min_events=250, refit_sessions=7)

    poisoned = events.copy()
    cut = _day(431)
    poisoned.loc[poisoned["session"] >= cut, "y"] = 1
    after = cal.recalibrate_walkforward(poisoned, history, min_events=250, refit_sessions=7)

    early = np.asarray(events["session"] < cut)
    assert np.allclose(base[early], after[early], equal_nan=True)


def test_recalibration_groups_by_question_kind_and_horizon() -> None:
    """A 1-session and a 5-session question of the same kind never share one map (12.3)."""
    rng = np.random.default_rng(119)
    history = _history_frame(rng, n_sessions=600, start=0)
    long_history = history.copy()
    long_history["question_id"] = "eval.down_1em_5s"
    long_history["horizon"] = 5
    long_history["y"] = 1 - long_history["y"]  # a deliberately different regime
    combined = pd.concat([history, long_history], ignore_index=True)

    events = _events_frame(rng, n_sessions=10, start=601, p_forecast=0.12, questions=("eval.down_1em_1s",))
    short_only = cal.recalibrate_walkforward(events, history, min_events=250, refit_sessions=21)
    with_other_kind = cal.recalibrate_walkforward(events, combined, min_events=250, refit_sessions=21)
    assert np.allclose(short_only, with_other_kind, equal_nan=True)


def test_recalibration_rejects_impossible_arguments() -> None:
    rng = np.random.default_rng(121)
    events = _events_frame(rng, n_sessions=2, start=0, p_forecast=0.1)
    with pytest.raises(EvalError):
        cal.recalibrate_walkforward(events, None, min_events=0, refit_sessions=5)
    with pytest.raises(EvalError):
        cal.recalibrate_walkforward(events, None, min_events=10, refit_sessions=0)


# ======================================================================================================================
# eligibility, look evaluability, d_t
# ======================================================================================================================


def test_eligible_sessions_need_every_primary_question_and_every_reference() -> None:
    rng = np.random.default_rng(123)
    events = _events_frame(rng, n_sessions=4, start=0, p_forecast=0.12)
    n = len(events)
    ref_a = np.full(n, 0.11)
    ref_b = np.full(n, 0.12)
    assert cal.eligible_sessions(events, {"a": ref_a, "b": ref_b}, QUESTIONS).tolist() == [_day(i) for i in range(4)]

    # one reference unavailable on one (session, question) makes the whole session ineligible
    ref_b = ref_b.copy()
    ref_b[np.asarray((events["session"] == _day(2)) & (events["question_id"] == QUESTIONS[1]))] = np.nan
    eligible = cal.eligible_sessions(events, {"a": ref_a, "b": ref_b}, QUESTIONS)
    assert eligible.tolist() == [_day(0), _day(1), _day(3)]
    assert cal.ineligible_sessions(events, eligible).tolist() == [_day(2)]


def test_a_session_missing_a_primary_question_is_ineligible() -> None:
    rng = np.random.default_rng(125)
    events = _events_frame(rng, n_sessions=3, start=0, p_forecast=0.12)
    events = events[~((events["session"] == _day(1)) & (events["question_id"] == QUESTIONS[0]))].reset_index(drop=True)
    refs = {"a": np.full(len(events), 0.11)}
    assert cal.eligible_sessions(events, refs, QUESTIONS).tolist() == [_day(0), _day(2)]


def test_eligible_sessions_refuse_a_misaligned_reference_or_empty_family() -> None:
    rng = np.random.default_rng(127)
    events = _events_frame(rng, n_sessions=2, start=0, p_forecast=0.12)
    with pytest.raises(EvalError):
        cal.eligible_sessions(events, {"a": np.zeros(3)}, QUESTIONS)
    with pytest.raises(EvalError):
        cal.eligible_sessions(events, {"a": np.zeros(len(events))}, ())
    with pytest.raises(EvalError):
        cal.eligible_sessions(events, {}, QUESTIONS)


def test_require_evaluable_look_raises_when_too_few_sessions_are_eligible() -> None:
    """12.1(b): a look with fewer than `n_sessions` eligible sessions CANNOT be evaluated."""
    eligible = pd.Index([_day(i) for i in range(10)], name="session")
    with pytest.raises(PreregError, match="look not evaluable"):
        cal.require_evaluable_look(eligible, n_sessions=120, look="look 1")
    taken = cal.require_evaluable_look(eligible, n_sessions=10, look="look 1")
    assert taken.tolist() == eligible.tolist()
    assert cal.require_evaluable_look(eligible, n_sessions=4).tolist() == [_day(i) for i in range(4)]
    with pytest.raises(EvalError):
        cal.require_evaluable_look(eligible, n_sessions=0)


def test_loss_differential_averages_per_question_first() -> None:
    """`weighting = "equal per question"` (12.1): a question with one row weighs as much as one with three."""
    sessions = [_day(0)] * 4
    questions = ["q1", "q1", "q1", "q2"]
    p_ref = [0.5, 0.5, 0.5, 0.5]
    p_jev = [0.0, 0.0, 0.0, 1.0]
    y = [0, 0, 0, 0]
    # q1: each row (0.25 - 0) = 0.25 -> 0.25 ; q2: (0.25 - 1.0) = -0.75
    got = cal.loss_differential(p_ref, p_jev, y, sessions, questions)
    assert got.tolist() == pytest.approx([(0.25 + -0.75) / 2])


def test_loss_differential_is_ordered_by_session_and_positive_when_the_forecaster_wins() -> None:
    sessions = [_day(1), _day(1), _day(0), _day(0)]
    questions = ["q", "q", "q", "q"]
    p_ref = [0.5, 0.5, 0.5, 0.5]
    p_jev = [0.9, 0.9, 0.1, 0.1]
    y = [1, 1, 0, 0]
    got = cal.loss_differential(p_ref, p_jev, y, sessions, questions)
    assert cal.unique_sessions(sessions).tolist() == [_day(0), _day(1)]
    assert got.tolist() == pytest.approx([0.25 - 0.01, 0.25 - 0.01])


def test_loss_differential_excludes_void_outcomes_and_reports_nan_for_an_empty_question() -> None:
    sessions = [_day(0), _day(0), _day(1), _day(1)]
    questions = ["q1", "q2", "q1", "q2"]
    got = cal.loss_differential(
        [0.5, 0.5, 0.5, 0.5],
        [0.1, 0.1, 0.1, 0.1],
        [0, 0, 0, float("nan")],
        sessions,
        questions,
    )
    assert got[0] == pytest.approx(0.25 - 0.01)
    assert math.isnan(got[1])  # q2 has no usable row on the second session


def test_loss_differential_refuses_misaligned_inputs() -> None:
    with pytest.raises(EvalError):
        cal.loss_differential([0.5, 0.5], [0.1, 0.1], [0, 0], [_day(0)], ["q", "q"])


def test_an_unscorable_session_is_excluded_and_listed_never_averaged_away() -> None:
    """12.1 `missing`: void outcomes are EXCLUDED AND LISTED - the bound may not absorb them replicate by replicate.

    `eligible_sessions` calls a session eligible on reference availability alone, so a session whose every primary
    question is void is eligible and carries a `NaN` `d_t`.  The caller splits the series here and reports the dropped
    sessions; `lower_bound` refuses a series that still carries one.
    """
    sessions = [_day(i) for i in range(3) for _ in range(2)]
    questions = ["q1", "q2"] * 3
    y = [0, 0, 0, float("nan"), 0, 0]  # the middle session's q2 is void
    d = cal.loss_differential([0.5] * 6, [0.1] * 6, y, sessions, questions)
    index = cal.unique_sessions(sessions)

    # the void session IS eligible: eligibility is defined on reference availability only (12.1 `eligible_session`)
    events = pd.DataFrame({"session": sessions, "question_id": questions})
    refs = {"a": np.zeros(6), "b": np.zeros(6)}
    assert cal.eligible_sessions(events, refs, ("q1", "q2")).tolist() == list(index)

    kept, kept_sessions, dropped = cal.exclude_unscorable_sessions(d, index)
    assert kept.tolist() == pytest.approx([0.24, 0.24])
    assert list(kept_sessions) == [_day(0), _day(2)]
    assert list(dropped) == [_day(1)]

    with pytest.raises(EvalError, match="non-finite at positions"):
        lower_bound(d, alpha=0.05, block=2.0, reps=50, rng=np.random.default_rng(0), interval="percentile")
    assert math.isfinite(lower_bound(kept, alpha=0.05, block=2.0, reps=50, rng=np.random.default_rng(0), interval="percentile"))


def test_exclude_unscorable_sessions_refuses_a_misaligned_index() -> None:
    with pytest.raises(EvalError, match="d_t has 2 values"):
        cal.exclude_unscorable_sessions([0.1, 0.2], cal.unique_sessions([_day(0), _day(1), _day(2)]))


# ======================================================================================================================
# history can never reach a scored table (12.1 exemption 1)
# ======================================================================================================================


def test_only_the_two_reference_builders_accept_a_history_frame() -> None:
    """12.3: `history` is passed ONLY to the two reference builders; nothing else in the module accepts it."""
    accepting = {
        name
        for name, obj in vars(cal).items()
        if not name.startswith("_") and inspect.isfunction(obj) and "history" in inspect.signature(obj).parameters
    }
    assert accepting == {"base_rate_expanding", "recalibrate_walkforward", "build_references", "session_grid"}


def test_references_are_aligned_with_the_events_and_never_add_history_rows() -> None:
    rng = np.random.default_rng(131)
    history = _history_frame(rng, n_sessions=400, start=0)
    events = _events_frame(rng, n_sessions=15, start=401, p_forecast=0.12)
    refs = cal.build_references(events, history, recal_min_events=250, base_rate_min_events=250, refit_sessions=21)
    assert set(refs) == {"raw_implied", "implied_recalibrated", "base_rate_expanding", "base_rate_in_sample", "coin"}
    for name, values in refs.items():
        assert len(values) == len(events), name
    assert refs["coin"].tolist() == [0.5] * len(events)
    assert refs["raw_implied"].tolist() == pytest.approx(list(events["p_implied"]))


def test_references_are_read_from_the_ppm_columns_when_the_float_columns_are_absent() -> None:
    rng = np.random.default_rng(133)
    events = _events_frame(rng, n_sessions=3, start=0, p_forecast=0.12)
    ppm = events.drop(columns=["p_implied"])
    ppm["p_implied_ppm"] = [round(float(v) * 1_000_000) for v in events["p_implied"]]
    assert cal.base_rate_in_sample(ppm).tolist() == pytest.approx(cal.base_rate_in_sample(events).tolist())
    refs = cal.build_references(ppm, None, recal_min_events=250, base_rate_min_events=250, refit_sessions=21)
    assert refs["raw_implied"].tolist() == pytest.approx(list(events["p_implied"]))


def test_a_missing_column_is_reported_precisely() -> None:
    rng = np.random.default_rng(135)
    events = _events_frame(rng, n_sessions=2, start=0, p_forecast=0.12).drop(columns=["p_implied"])
    with pytest.raises(EvalError, match="p_implied"):
        cal.recalibrate_walkforward(events, None, min_events=1, refit_sessions=5)


# ======================================================================================================================
# THE REGRESSION TEST OF 12.1(c) / V11 (the judges' must-fix)
# ======================================================================================================================


def _joint_test(events: pd.DataFrame, refs: dict[str, np.ndarray], look: pd.Index, *, alpha: float) -> dict[str, float]:
    """The pre-registered intersection-union test: the one-sided lower bound of `mean d_t` against BOTH references."""
    keep = np.asarray(events["session"].isin(list(look)))
    sessions = list(events.loc[keep, "session"])
    questions = list(events.loc[keep, "question_id"])
    y = np.asarray(events.loc[keep, "y"], dtype=float)
    p = np.asarray(events.loc[keep, "p"], dtype=float)
    bounds: dict[str, float] = {}
    for name in ("implied_recalibrated", "base_rate_expanding"):
        d = cal.loss_differential(refs[name][keep], p, y, sessions, questions)
        bounds[name] = lower_bound(d, alpha=alpha, block=10.0, reps=2000, rng=np.random.default_rng(2026), interval="percentile")
    return bounds


def test_constant_climatological_forecaster_beats_raw_implied_but_fails_the_joint_test() -> None:
    """The must-fix regression test of 12.1(c) / V11, on synthetic risk-premium data.

    A constant forecaster with a sensible prior (0.12) earns a clearly POSITIVE Brier skill score against the raw,
    risk-neutral option-implied probability - D12's literal statistic - while the pre-registered joint test against the
    recalibrated implied probability AND the expanding base rate rejects it.
    """
    rng = np.random.default_rng(20260917)
    history = _history_frame(rng, n_sessions=1200, start=0)
    # the climatological prior: the constant that a forecaster with no skill at all would quote
    events = _events_frame(rng, n_sessions=250, start=1201, p_forecast=0.11)

    y = np.asarray(events["y"], dtype=float)
    p_const = np.asarray(events["p"], dtype=float)
    raw_implied = np.asarray(events["p_implied"], dtype=float)

    # 1. D12 literal: clearly positive BSS against the RAW implied probability - "not a verdict" (V11)
    bss_raw = cal.brier_skill(p_const, y, raw_implied)
    assert bss_raw > 0.05, f"the synthetic risk premium is too small: BSS {bss_raw}"

    refs = cal.build_references(events, history, recal_min_events=250, base_rate_min_events=250, refit_sessions=21)
    assert bool(np.isfinite(refs["implied_recalibrated"]).all())
    assert bool(np.isfinite(refs["base_rate_expanding"]).all())

    # 2. the two references remove exactly what the constant forecaster was exploiting: the risk premium and the
    #    climatology.  Against them the same forecaster shows no skill worth speaking of.
    assert float(np.mean(refs["base_rate_expanding"])) == pytest.approx(float(y.mean()), abs=0.02)
    implied = np.asarray(events["p_implied"], dtype=float)
    for level in IMPLIED_LEVELS:
        rows = np.isclose(implied, level)
        assert float(refs["implied_recalibrated"][rows].mean()) == pytest.approx(TRUE_P[level], abs=0.03)
    assert abs(cal.brier_skill(p_const, y, refs["implied_recalibrated"])) < 0.015
    assert abs(cal.brier_skill(p_const, y, refs["base_rate_expanding"])) < 0.015

    # 3. the pre-registered verdict: the look is evaluable and the joint test FAILS
    eligible = cal.eligible_sessions(
        events,
        {k: refs[k] for k in ("implied_recalibrated", "base_rate_expanding")},
        QUESTIONS,
    )
    assert len(eligible) == 250
    look = cal.require_evaluable_look(eligible, n_sessions=120, look="look 1")
    bounds = _joint_test(events, refs, look, alpha=0.01)
    # A climatological constant IS the expanding base rate, so against that reference it ties.  The bound against it
    # lands a sliver ABOVE zero - a false rejection of the over-sized percentile interval (test_eval_bootstrap measures
    # its size at about 0.085 against a nominal 0.05 on this `d_t` structure, which is why 12.3 has `eval power` choose
    # the interval by measured size), not skill: it is four orders of magnitude below the Brier scale of the question.
    assert abs(bounds["base_rate_expanding"]) < 1e-3, bounds
    # That is exactly why the pre-registration requires BOTH references: against the recalibrated implied probability
    # the same forecaster shows nothing at all.
    assert bounds["implied_recalibrated"] <= 0.0, bounds
    assert not all(bound > 0.0 for bound in bounds.values()), "the zero-skill forecaster passed the joint test"


def test_a_tier_a_only_history_leaves_the_first_look_not_evaluable() -> None:
    """12.1(b) / (c): without the reference-history warm start the references are unavailable and look 1 is refused."""
    rng = np.random.default_rng(20260918)
    events = _events_frame(rng, n_sessions=100, start=0, p_forecast=0.12)
    refs = cal.build_references(events, None, recal_min_events=250, base_rate_min_events=250, refit_sessions=21)

    assert bool(np.isnan(refs["implied_recalibrated"]).all())  # the refit epochs never reach 250 pairs
    assert bool(np.isnan(refs["base_rate_expanding"][: 6 * 85]).any())

    eligible = cal.eligible_sessions(
        events,
        {k: refs[k] for k in ("implied_recalibrated", "base_rate_expanding")},
        QUESTIONS,
    )
    assert len(eligible) == 0
    assert len(cal.ineligible_sessions(events, eligible)) == 100
    with pytest.raises(PreregError, match="look not evaluable"):
        cal.require_evaluable_look(eligible, n_sessions=120, look="look 1")


def test_a_genuinely_skilled_forecaster_passes_the_joint_test() -> None:
    """The counterpart: the test must be able to detect real skill, or the regression test above proves nothing."""
    rng = np.random.default_rng(20260919)
    history = _history_frame(rng, n_sessions=400, start=0)
    events = _events_frame(rng, n_sessions=250, start=401, p_forecast=0.12)
    # a forecaster that half-knows the truth: a mixture of the physical probability and the outcome
    truth = np.asarray([TRUE_P[float(v)] for v in events["p_implied"]], dtype=float)
    y = np.asarray(events["y"], dtype=float)
    events = events.assign(p=np.clip(0.45 * truth + 0.55 * y, 0.01, 0.99))

    refs = cal.build_references(events, history, recal_min_events=250, base_rate_min_events=250, refit_sessions=21)
    eligible = cal.eligible_sessions(
        events,
        {k: refs[k] for k in ("implied_recalibrated", "base_rate_expanding")},
        QUESTIONS,
    )
    look = cal.require_evaluable_look(eligible, n_sessions=120, look="look 1")
    bounds = _joint_test(events, refs, look, alpha=0.01)
    assert all(bound > 0.0 for bound in bounds.values()), bounds
