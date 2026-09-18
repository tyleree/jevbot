"""P&L, exposure, cost and decision-process metrics (DESIGN.md 12.2; test plan 15.1 `eval/*`).

Every formula is checked against a number worked out by hand from the definition in the specification - never by running
the implementation twice. The two structural claims of 12.2 get their own tests: the abstention funnel is the **join of
DECISION and RISK_VERDICT on `decision_id`** (an entry that died in the risk engine is invisible in either frame alone),
and a configuration that is profitable only at mid is `REJECTED_MID_ONLY` (10.3).
"""

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jevbot.eval import load, metrics
from jevbot.types import FillRule
from tests.fixtures.make_run_fixture import RunFixture, RunFixtureSpec, make_run_store

SQRT_252 = math.sqrt(252)


def _daily(equities: dict[str, list[int]], start: date = date(2014, 1, 6)) -> pd.DataFrame:
    n = len(next(iter(equities.values())))
    sessions = [start + timedelta(days=index) for index in range(n)]
    data: dict[str, object] = {"session": sessions}
    for band, values in equities.items():
        data[f"equity_{band}"] = values
    return pd.DataFrame(data)


# ======================================================================================================================
# Equity-series formulas, each against a hand-computed value
# ======================================================================================================================


def test_sharpe_matches_the_hand_computed_value() -> None:
    returns = np.array([0.010, -0.005, 0.020, 0.000, 0.010])
    # by hand: mean = 0.035 / 5 = 0.007; deviations 0.003, -0.012, 0.013, -0.007, 0.003;
    # sum of squares = 9 + 144 + 169 + 49 + 9 = 380 (units of 1e-6); sample variance = 380e-6 / 4 = 95e-6
    hand = 0.007 / math.sqrt(0.000095) * SQRT_252
    assert metrics.sharpe(returns) == pytest.approx(hand, rel=1e-12)
    assert metrics.sharpe(returns) == pytest.approx(11.4009, rel=1e-4)


def test_sharpe_is_excess_of_zero_not_of_a_risk_free_rate() -> None:
    # V8: cash earns no interest in any run or baseline, so a flat 0.01 % a day is NOT a zero Sharpe
    returns = np.full(30, 0.0001)
    assert math.isnan(metrics.sharpe(returns))  # zero variance has no Sharpe...
    varied = np.array([0.0001, 0.0002] * 15)
    assert metrics.sharpe(varied) > 0  # ...but a positive mean with any variance does, with rf = 0


def test_sortino_uses_downside_deviation_over_the_whole_sample() -> None:
    returns = np.array([0.010, -0.005, 0.020, 0.000, 0.010])
    # by hand: the only negative return is -0.005; mean square of min(r, 0) = 0.000025 / 5 = 5e-6
    hand = 0.007 / math.sqrt(0.000005) * SQRT_252
    assert metrics.sortino(returns) == pytest.approx(hand, rel=1e-12)
    assert metrics.sortino(returns) == pytest.approx(49.6952, rel=1e-4)


def test_max_drawdown_is_the_deepest_fall_from_the_running_peak() -> None:
    equity = np.array([100.0, 120.0, 90.0, 110.0, 80.0])
    # peaks 100, 120, 120, 120, 120; falls 0, 0, 25 %, 8.33 %, 33.33 % -> the deepest is (120 - 80) / 120 = 1/3
    assert metrics.max_drawdown(equity) == pytest.approx(1.0 / 3.0)


def test_cvar_is_the_mean_of_the_worst_five_percent() -> None:
    returns = np.array([value / 1000.0 for value in range(-10, 11)])  # 21 values, -0.010 .. 0.010
    # the 5 % quantile of 21 evenly spaced values sits exactly on the second smallest, -0.009;
    # the values at or below it are -0.010 and -0.009, whose mean is -0.0095
    assert metrics.cvar(returns) == pytest.approx(-0.0095)


def test_cagr_compounds_over_trading_years() -> None:
    # 504 sessions is exactly two 252-session years; 1.21 over two years is 10 % a year
    assert metrics.cagr(100.0, 121.0, 504) == pytest.approx(0.10)
    assert math.isnan(metrics.cagr(0.0, 121.0, 504))
    assert math.isnan(metrics.cagr(100.0, 121.0, 0))


def test_equity_metrics_over_a_hand_built_series() -> None:
    daily = _daily({"orats": [1000, 1100, 990, 1089]})
    result = metrics.equity_metrics(daily, "orats")
    assert result.n_sessions == 4
    assert result.start_equity == 1000
    assert result.end_equity == 1089
    assert result.total_pnl == 89
    assert result.total_return == pytest.approx(0.089)
    # returns: +10 %, -10 %, +10 %; peaks 1000, 1100, 1100, 1100 -> deepest fall (1100 - 990) / 1100 = 10 %
    assert result.max_drawdown == pytest.approx(0.1)
    assert result.worst_day == pytest.approx(-0.1)
    assert result.best_day == pytest.approx(0.1)


def test_daily_returns_are_session_over_session() -> None:
    daily = _daily({"mid": [1000, 1100, 990]})
    assert metrics.daily_returns(daily, "mid") == pytest.approx(np.array([0.1, -0.1]))
    assert metrics.daily_returns(_daily({"mid": [1000]}), "mid").size == 0


def test_headline_band_follows_the_fill_rule() -> None:
    assert metrics.headline_band(FillRule.NEXT_SNAPSHOT) == "orats"
    assert metrics.headline_band(FillRule.SAME_SNAPSHOT_WORST) == "worst"
    with pytest.raises(ValueError, match="unknown fill rule"):
        metrics.headline_band("some_other_rule")


# ======================================================================================================================
# Trades
# ======================================================================================================================


def _trades(pnl: list[float], **extra: object) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "position_id": [f"p{index}" for index in range(len(pnl))],
            "closed": [True] * len(pnl),
            "pnl_mid": pnl,
            "holding_sessions": [4] * len(pnl),
            "forced": [False] * len(pnl),
            "exit_reason": ["profit_target"] * len(pnl),
        }
    )
    for key, value in extra.items():
        frame[key] = value
    return frame


def test_trade_metrics_match_the_hand_computed_values() -> None:
    result = metrics.trade_metrics(_trades([100.0, -50.0, 200.0, -25.0]), "mid")
    assert result.n_trades == 4
    assert result.hit_rate == pytest.approx(0.5)  # two of four are positive
    assert result.avg_win == pytest.approx(150.0)  # (100 + 200) / 2
    assert result.avg_loss == pytest.approx(-37.5)  # (-50 - 25) / 2
    assert result.profit_factor == pytest.approx(4.0)  # 300 won / 75 lost
    assert result.expectancy == pytest.approx(56.25)  # 225 / 4
    assert result.total_pnl == pytest.approx(225.0)
    assert result.worst_trade == pytest.approx(-50.0)
    assert result.best_trade == pytest.approx(200.0)
    assert result.avg_holding_sessions == pytest.approx(4.0)
    assert result.exit_reason_mix == {"profit_target": 4}


def test_trade_metrics_on_a_run_without_losses_and_without_trades() -> None:
    winners = metrics.trade_metrics(_trades([10.0, 20.0]), "mid")
    assert winners.hit_rate == 1.0
    assert math.isinf(winners.profit_factor)  # nothing was lost
    assert math.isnan(winners.avg_loss)

    nothing = metrics.trade_metrics(pd.DataFrame(), "mid")
    assert nothing.n_trades == 0
    assert math.isnan(nothing.hit_rate)


def test_trade_metrics_by_slice_partitions_the_trades() -> None:
    frame = _trades([100.0, -50.0, 200.0], underlying=["SPY", "SPY", "QQQ"])
    sliced = metrics.trade_metrics_by_slice(frame, "underlying", "mid")
    assert set(sliced) == {"SPY", "QQQ"}
    assert sliced["SPY"].n_trades == 2
    assert sliced["QQQ"].total_pnl == pytest.approx(200.0)


def test_max_adverse_excursion_is_the_worst_unrealised_mark() -> None:
    trades = pd.DataFrame(
        {
            "position_id": ["p1"],
            "qty": [1],
            "entry_session": [date(2014, 1, 6)],
            "exit_session": [date(2014, 1, 9)],
            "open_net_mid": [100],
        }
    )
    marks = pd.DataFrame(
        {
            "position_id": ["p1", "p1", "p1"],
            "session": [date(2014, 1, 6), date(2014, 1, 7), date(2014, 1, 8)],
            "liq_value": [-100, -85, -120],
            "mid_value": [-100, -85, -120],
            "stale": [False, False, False],
        }
    )
    result = metrics.max_adverse_excursion(trades, marks, "mid").iloc[0]
    # unrealised = (-liq - open_net) * 100 * qty: (100 - 100), (85 - 100), (120 - 100) -> 0, -1500, +2000 cents
    assert result["mae_cents"] == pytest.approx(-1500.0)
    assert result["mfe_cents"] == pytest.approx(2000.0)
    assert result["n_marks"] == 3


# ======================================================================================================================
# Exposure, risk normalisation and costs
# ======================================================================================================================


def test_exposure_metrics_from_hand_built_marks() -> None:
    marks = pd.DataFrame(
        {
            "bp_utilisation_ppm": [100_000, 300_000],
            "equity_orats": [1000, 2000],
            "open_max_loss": [100, 400],
            "net_delta_milli": [1000, 3000],
            "net_vega_milli": [-500, -1500],
        }
    )
    result = metrics.exposure_metrics(marks)
    assert result.avg_bp_utilisation == pytest.approx(0.2)  # (0.1 + 0.3) / 2
    assert result.max_bp_utilisation == pytest.approx(0.3)
    assert result.avg_max_loss_utilisation == pytest.approx(0.15)  # (100/1000 + 400/2000) / 2
    assert result.max_max_loss_utilisation == pytest.approx(0.2)
    assert result.avg_net_delta == pytest.approx(2.0)  # milli units
    assert result.avg_net_vega == pytest.approx(-1.0)
    assert result.max_loss_days_at_risk == 500  # 100 + 400 "max-loss-days at risk"


def test_risk_normalised_pnl_is_pnl_per_max_loss_day() -> None:
    marks = pd.DataFrame({"open_max_loss": [100, 200, 300]})
    assert metrics.risk_normalised_pnl(1000.0, marks) == pytest.approx(1000.0 / 600.0)
    assert math.isnan(metrics.risk_normalised_pnl(1000.0, pd.DataFrame({"open_max_loss": [0, 0]})))


def test_cost_metrics_charge_the_headline_minus_mid_difference() -> None:
    fills = pd.DataFrame(
        {
            "net_orats": [105, -115],  # a buy at 105 and a sell-to-close at -115 (a credit)
            "net_mid": [100, -120],
            "qty": [1, 1],
            "fees_micro": [130_000, 130_000],
        }
    )
    trades = _trades([2000.0])
    result = metrics.cost_metrics(fills, trades)
    # (105 - 100) * 100 + (-115 + 120) * 100 = 500 + 500 = 1000 cents paid away against the mid
    assert result.slippage_cents == pytest.approx(1000.0)
    assert result.fees_cents == 26  # 260,000 micro-dollars = 26 cents
    assert result.gross_pnl_mid == pytest.approx(2000.0)
    assert result.slippage_frac_of_gross == pytest.approx(0.5)


def test_attribution_recovers_an_exact_linear_relation() -> None:
    rng = np.random.default_rng(11)
    underlying = rng.standard_normal(200)
    iv_change = rng.standard_normal(200)
    pnl = 3.0 + 2.0 * underlying - 5.0 * iv_change
    result = metrics.attribution(pnl, {"underlying_return": underlying, "iv30_change": iv_change})
    assert result.alpha == pytest.approx(3.0, abs=1e-9)
    assert result.coefficients["underlying_return"] == pytest.approx(2.0, abs=1e-9)
    assert result.coefficients["iv30_change"] == pytest.approx(-5.0, abs=1e-9)
    assert result.r_squared == pytest.approx(1.0, abs=1e-12)
    assert result.n == 200


def test_attribution_refuses_mismatched_series() -> None:
    with pytest.raises(ValueError, match="same non-zero length"):
        metrics.attribution(np.zeros(5), {"x": np.zeros(4)})


# ======================================================================================================================
# The DECISION x RISK_VERDICT funnel
# ======================================================================================================================


def _decision(decision_id: str, action: str, reasons: tuple[str, ...] = ()) -> dict[str, object]:
    return {
        "decision_id": decision_id,
        "kind": "entry",
        "action": action,
        "reasons": reasons,
        "variant_agreement": {},
        "underlying": "SPY",
        "structure_kind": "iron_condor" if action == "enter" else None,
        "n_requests": 2,
    }


def test_abstention_funnel_needs_both_sides_of_the_join() -> None:
    decisions = pd.DataFrame(
        [
            _decision("d1", "no_trade", ("score:below_min", "hold")),
            _decision("d2", "enter"),
            _decision("d3", "enter"),
            _decision("d4", "enter"),
            _decision("d5", "no_trade"),
        ]
    )
    verdicts = pd.DataFrame(
        [
            {"seq": 1, "decision_id": "d2", "reject_codes": (), "approved": True},
            {"seq": 2, "decision_id": "d3", "reject_codes": ("candidate:exceeds_risk_budget",), "approved": False},
            {"seq": 3, "decision_id": "d4", "reject_codes": ("risk:buying_power", "risk:qty_cap"), "approved": False},
        ]
    )
    funnel = metrics.abstention_funnel(decisions, verdicts)
    rows = {(row["stage"], row["code"]): row["n"] for _, row in funnel.iterrows()}
    assert rows[("rules", "score:below_min")] == 1  # the FIRST rules reason, not the whole trail
    assert rows[("rules", "no_trade")] == 1  # an abstention with no reason is still an abstention
    assert rows[("post_decision", "candidate:exceeds_risk_budget")] == 1
    assert rows[("post_decision", "risk:buying_power")] == 1  # the FIRST reject code
    assert rows[("emitted", metrics.FUNNEL_EMITTED)] == 1
    assert sum(rows.values()) == len(decisions)
    # neither frame alone can produce this: the DECISIONs of d3 and d4 carry no code at all
    assert decisions[decisions["decision_id"].isin(["d3", "d4"])]["reasons"].map(len).sum() == 0


def test_abstention_funnel_orders_rules_before_post_decision_before_emitted() -> None:
    decisions = pd.DataFrame([_decision("d1", "no_trade", ("dq:insufficient",)), _decision("d2", "enter")])
    verdicts = pd.DataFrame([{"seq": 1, "decision_id": "d2", "reject_codes": (), "approved": True}])
    funnel = metrics.abstention_funnel(decisions, verdicts)
    assert list(funnel["stage"]) == ["rules", "emitted"]
    assert metrics.abstention_funnel(pd.DataFrame(), pd.DataFrame()).empty


def test_jev_usage_from_hand_built_frames() -> None:
    decisions = pd.DataFrame({"n_requests": [2, 2, 2]})
    sidecar = pd.DataFrame(
        {
            "input_tokens": pd.array([1000, 2000, 3000], dtype="Int64"),
            "latency_ms": pd.array([100, 200, 300], dtype="Int64"),
            "cache_hit": [True, True, False],
        }
    )
    usage = metrics.jev_usage(decisions, sidecar)
    assert usage.n_requests == 6
    assert usage.input_tokens == 6000
    assert usage.cache_hit_rate == pytest.approx(2 / 3)
    assert usage.cost_usd == pytest.approx(6000 / 1_000_000 * 0.042)
    assert usage.latency_p50_ms == pytest.approx(200.0)


# ======================================================================================================================
# Against the fixture run
# ======================================================================================================================


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory: pytest.TempPathFactory) -> RunFixture:
    return make_run_store(tmp_path_factory.mktemp("metrics") / "run.sqlite")


def test_rejected_mid_only_fires_only_for_the_mid_only_configuration(fixture_run: RunFixture, tmp_path: Path) -> None:
    def per_band(path: Path) -> dict[str, metrics.EquityMetrics]:
        with load.RunStore(path) as store:
            daily = load.daily_frame(store)
            return {band: metrics.equity_metrics(daily, band) for band in metrics.BANDS}

    ordinary = per_band(fixture_run.path)
    assert ordinary["orats"].total_pnl > 0
    assert not metrics.rejected_mid_only(ordinary)

    mid_only = make_run_store(tmp_path / "mid_only.sqlite", RunFixtureSpec(mid_only=True))
    optimistic = per_band(mid_only.path)
    assert optimistic["mid"].total_pnl > 0
    assert optimistic["orats"].total_pnl <= 0
    assert optimistic["worst"].total_pnl <= 0
    assert metrics.rejected_mid_only(optimistic)


def test_all_three_bands_are_computed_and_ordered_as_the_fill_model_implies(fixture_run: RunFixture) -> None:
    with load.RunStore(fixture_run.path) as store:
        daily = load.daily_frame(store)
        results = {band: metrics.equity_metrics(daily, band) for band in metrics.BANDS}
    # 10.3: mid is the best case and worst is the worst, on the same trade list
    assert results["mid"].total_pnl > results["orats"].total_pnl > results["worst"].total_pnl


def test_decision_process_metrics_over_the_fixture(fixture_run: RunFixture) -> None:
    with load.RunStore(fixture_run.path) as store:
        result = metrics.decision_process_metrics(
            decisions=load.decisions_frame(store),
            verdicts=load.risk_verdicts_frame(store),
            fills=load.fills_frame(store),
            order_status=load.order_status_frame(store),
            anomalies=load.anomalies_frame(store),
            risk_events=load.risk_events_frame(store),
            kills=load.kill_frame(store),
        )
    assert result.n_entry_decisions == 3 * len(fixture_run.sessions)
    assert result.n_manage_decisions == 0
    assert 0.0 < result.veto_rates["vol.explained_by_event"] < 1.0
    assert 0.0 < result.crosscheck_reject_rates["trend_vs_direction"] < 1.0
    assert result.risk_reject_rates["size_zero"] > 0.0
    assert result.candidate_reject_rates["exceeds_risk_budget"] > 0.0
    assert any(pair.startswith("SPY/") for pair in result.size_zero_by_pair)
    # the key_perm variant is the one the fixture makes disagree on `score:below_min` decisions (7.8)
    assert result.perturbation_flip_rates["key_perm"] > 0.0
    assert result.perturbation_flip_rates["opt_perm"] == 0.0
    assert result.zero_bid_close_legs == len(fixture_run.position_ids)
    assert result.forced_fills > 0
    assert result.degraded_fills > 0
    assert result.anomaly_counts == {"stale_mark": 1, "assignment_sim": 1}
    assert result.assignment_sims == 1
    assert result.kill_events == 4  # tripped, cancelled, close_submitted, flat_verified
    assert result.halt_events == 1


def test_kill_affected_slice_separates_the_windows(fixture_run: RunFixture) -> None:
    with load.RunStore(fixture_run.path) as store:
        daily = load.daily_frame(store)
        affected = load.kill_affected_sessions(store)
    sliced = metrics.equity_metrics_by_kill_window(daily, affected, "orats")
    assert sliced["kill_affected"].n_sessions == len(affected)
    assert sliced["clean"].n_sessions == len(daily) - len(affected)
    # the slice exists so a window in which the bot could not trade is never averaged into the headline number
    assert sliced["kill_affected"].n_sessions + sliced["clean"].n_sessions == len(daily)
