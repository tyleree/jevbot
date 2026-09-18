"""P&L, exposure, cost and decision-process metrics (DESIGN.md 12.2; numpy + scipy only, D15).

Everything here is computed on **all three fill bands** (ORATS / worst / mid, D13) from the frames of `eval/load.py`.
Three conventions from the design shape the formulas and are stated once, here:

* **`rf = 0` everywhere** (V8): cash earns no interest in any run or baseline, so Sharpe and Sortino are excess-of-zero.
  Buy-and-hold is the only thing reported in excess of the 13-week bill, and that happens in `baselines.py`.
* **Mid is the best case only.** A configuration that is profitable only at mid is labelled `REJECTED_MID_ONLY` in the
  report (10.3, B7.3); `rejected_mid_only()` is the test the report renders.
* **Headline numbers are paired differences versus baseline 3 and the percentile versus baseline 4, never absolute P&L**
  (B7.6). This module computes the absolute series that those comparisons are built from; it never claims one.

The abstention funnel is the **join of DECISION and RISK_VERDICT on `decision_id`** (7.9, 12.2): the rules-side trail
lives in `EntryDecision.reasons` and everything that happened after the decision - gate, candidate, risk - lives in the
RISK_VERDICT's `reject_codes`. Neither frame alone can tell you where an entry died.
"""

from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from typing import Any, Final

import msgspec
import numpy as np
import pandas as pd

from jevbot import vocab
from jevbot.types import FillRule

__all__ = [
    "BANDS",
    "JEV_COST_USD_PER_MTOK",
    "TRADING_DAYS",
    "AttributionResult",
    "CostMetrics",
    "DecisionProcessMetrics",
    "EquityMetrics",
    "ExposureMetrics",
    "JevUsage",
    "TradeMetrics",
    "abstention_funnel",
    "attribution",
    "cagr",
    "cost_metrics",
    "cvar",
    "daily_returns",
    "decision_process_metrics",
    "equity_metrics",
    "equity_metrics_by_kill_window",
    "exposure_metrics",
    "headline_band",
    "jev_usage",
    "max_adverse_excursion",
    "max_drawdown",
    "metrics_tree",
    "rejected_mid_only",
    "risk_normalised_pnl",
    "sharpe",
    "sortino",
    "trade_metrics",
    "trade_metrics_by_slice",
]

BANDS: Final[tuple[str, ...]] = ("orats", "worst", "mid")
TRADING_DAYS: Final[int] = 252  # the 252-day annualisation of 12.2
JEV_COST_USD_PER_MTOK: Final[float] = 0.042  # 12.2 / 6.7: cost at $0.042 per million input tokens
CONTRACT_MULTIPLIER: Final[int] = 100  # one contract at price p is worth p * 100 cents (Conventions)
CVAR_TAIL: Final[float] = 0.05  # CVaR 5% of daily returns


def headline_band(fill_rule: FillRule | str) -> str:
    """ "Headline band" = `orats` under `next_snapshot`, `worst` under `same_snapshot_worst` (Conventions)."""
    value = fill_rule.value if isinstance(fill_rule, FillRule) else str(fill_rule)
    if value == FillRule.SAME_SNAPSHOT_WORST.value:
        return "worst"
    if value == FillRule.NEXT_SNAPSHOT.value:
        return "orats"
    raise ValueError(f"unknown fill rule {fill_rule!r}")


# ======================================================================================================================
# Equity series (12.2 "From the daily equity series per band")
# ======================================================================================================================


class EquityMetrics(msgspec.Struct, frozen=True, kw_only=True):
    band: str
    n_sessions: int
    start_equity: int
    end_equity: int
    total_pnl: int  # cents
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float  # a positive fraction of the running peak
    calmar: float
    cvar_5: float
    worst_day: float
    best_day: float


def daily_returns(daily: pd.DataFrame, band: str) -> np.ndarray:
    """Session-over-session simple returns of the equity series of one band.

    The series comes from the SESSION_END entries, whose headline-band value is the next session's `day_start_equity`
    (9.5), so these are exactly the returns the daily-loss halt and the drawdown kill were evaluated against.
    """
    equity = _equity_series(daily, band)
    if equity.size < 2:
        return np.zeros(0, dtype=float)
    previous = equity[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.where(previous != 0.0, equity[1:] / previous - 1.0, 0.0)
    return np.asarray(returns, dtype=float)


def _is_degenerate(spread: float, values: np.ndarray) -> bool:
    """True when a dispersion measure is zero to within floating-point noise.

    A constant equity series is a real case, not a pathology: baseline 1 (cash) never trades, and a run that was halted
    throughout is flat. `np.std` of a constant array is ~1e-21 rather than exactly 0, so an equality test would turn
    "no risk taken" into a Sharpe of 1e16. The ratio has no meaning there; `nan` is the honest answer.
    """
    if not np.isfinite(spread):
        return True
    scale = max(1.0, float(np.max(np.abs(values))) if values.size else 1.0)
    return spread <= 1e-12 * scale


def sharpe(returns: np.ndarray, *, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe with `rf = 0` (V8). Fewer than two returns, or a flat series, give `nan`."""
    values = np.asarray(returns, dtype=float)
    if values.size < 2:
        return float("nan")
    sd = float(np.std(values, ddof=1))
    if _is_degenerate(sd, values):
        return float("nan")
    return float(np.mean(values)) / sd * float(np.sqrt(periods))


def sortino(returns: np.ndarray, *, periods: int = TRADING_DAYS) -> float:
    """Annualised Sortino with `rf = 0`: downside deviation over the full sample (zeros for non-negative returns)."""
    values = np.asarray(returns, dtype=float)
    if values.size < 2:
        return float("nan")
    downside = np.minimum(values, 0.0)
    dd = float(np.sqrt(np.mean(np.square(downside))))
    if _is_degenerate(dd, values):
        return float("nan")
    return float(np.mean(values)) / dd * float(np.sqrt(periods))


def max_drawdown(equity: np.ndarray) -> float:
    """The deepest peak-to-trough fall of the equity series, as a positive fraction of the running peak."""
    values = np.asarray(equity, dtype=float)
    if values.size == 0:
        return float("nan")
    peaks = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(peaks > 0.0, (peaks - values) / peaks, 0.0)
    return float(np.max(drawdowns))


def cagr(start_equity: float, end_equity: float, n_sessions: int, *, periods: int = TRADING_DAYS) -> float:
    """Compound annual growth rate over `n_sessions` trading sessions."""
    if n_sessions <= 0 or start_equity <= 0.0:
        return float("nan")
    years = n_sessions / periods
    if years <= 0.0:
        return float("nan")
    ratio = end_equity / start_equity
    if ratio <= 0.0:
        return float("nan")
    return float(ratio ** (1.0 / years) - 1.0)


def cvar(returns: np.ndarray, *, tail: float = CVAR_TAIL) -> float:
    """Conditional value at risk: the mean of the worst `tail` share of daily returns (a negative number)."""
    values = np.asarray(returns, dtype=float)
    if values.size == 0:
        return float("nan")
    threshold = float(np.quantile(values, tail))
    worst = values[values <= threshold]
    if worst.size == 0:  # pragma: no cover - the quantile is always attained by at least one sample
        return threshold
    return float(np.mean(worst))


def equity_metrics(daily: pd.DataFrame, band: str) -> EquityMetrics:
    """Every headline number of 12.2's first sentence, for one fill band."""
    equity = _equity_series(daily, band)
    returns = daily_returns(daily, band)
    n = int(equity.size)
    start = int(equity[0]) if n else 0
    end = int(equity[-1]) if n else 0
    drawdown = max_drawdown(equity)
    growth = cagr(float(start), float(end), max(n - 1, 0))
    return EquityMetrics(
        band=band,
        n_sessions=n,
        start_equity=start,
        end_equity=end,
        total_pnl=end - start,
        total_return=(end / start - 1.0) if start else float("nan"),
        cagr=growth,
        ann_vol=float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS)) if returns.size > 1 else float("nan"),
        sharpe=sharpe(returns),
        sortino=sortino(returns),
        max_drawdown=drawdown,
        calmar=(growth / drawdown) if drawdown > 0.0 else float("nan"),
        cvar_5=cvar(returns),
        worst_day=float(np.min(returns)) if returns.size else float("nan"),
        best_day=float(np.max(returns)) if returns.size else float("nan"),
    )


def equity_metrics_by_kill_window(daily: pd.DataFrame, kill_sessions: Iterable[date], band: str) -> dict[str, EquityMetrics]:
    """The `kill-affected windows` slice of 12.2: the same metrics on the affected and the clean sessions.

    A window in which the kill switch or an entry halt was active is not a fair sample of the strategy, so it is sliced
    out and reported beside the rest rather than averaged into one number.
    """
    affected = set(kill_sessions)
    mask = daily["session"].isin(affected)
    return {
        "kill_affected": equity_metrics(daily[mask], band),
        "clean": equity_metrics(daily[~mask], band),
    }


def rejected_mid_only(per_band: Mapping[str, EquityMetrics], *, headline: str = "orats") -> bool:
    """10.3 / B7.3: profitable at MID but not at the headline band and not at the worst band => `REJECTED_MID_ONLY`.

    Mid is the best case only; a configuration that survives only there has been paid for by an optimistic fill model,
    not by the strategy.
    """
    mid = per_band.get("mid")
    head = per_band.get(headline)
    worst = per_band.get("worst")
    if mid is None or head is None or worst is None:
        return False
    return mid.total_pnl > 0 and head.total_pnl <= 0 and worst.total_pnl <= 0


def _equity_series(daily: pd.DataFrame, band: str) -> np.ndarray:
    column = f"equity_{band}"
    if daily is None or daily.empty or column not in daily.columns:
        return np.zeros(0, dtype=float)
    series = daily.sort_values("session")[column].dropna()
    return np.asarray(series.to_numpy(), dtype=float)


# ======================================================================================================================
# Closed trades (12.2 "From closed trades")
# ======================================================================================================================


class TradeMetrics(msgspec.Struct, frozen=True, kw_only=True):
    band: str
    n_trades: int
    hit_rate: float
    avg_win: float  # cents
    avg_loss: float  # cents, negative
    profit_factor: float
    expectancy: float  # cents per trade
    total_pnl: float  # cents
    worst_trade: float
    best_trade: float
    avg_holding_sessions: float
    forced_exits: int
    exit_reason_mix: dict[str, int]


def trade_metrics(trades: pd.DataFrame, band: str) -> TradeMetrics:
    """Closed-trade statistics for one band. Positions still open at the end of the run are not trades (12.2)."""
    column = f"pnl_{band}"
    if trades is None or trades.empty or column not in trades.columns:
        return TradeMetrics(
            band=band,
            n_trades=0,
            hit_rate=float("nan"),
            avg_win=float("nan"),
            avg_loss=float("nan"),
            profit_factor=float("nan"),
            expectancy=float("nan"),
            total_pnl=0.0,
            worst_trade=float("nan"),
            best_trade=float("nan"),
            avg_holding_sessions=float("nan"),
            forced_exits=0,
            exit_reason_mix={},
        )
    closed = trades[trades["closed"].astype(bool)] if "closed" in trades.columns else trades
    pnl = np.asarray(closed[column].dropna().to_numpy(), dtype=float)
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    loss_sum = float(np.sum(np.abs(losses)))
    holding = (
        np.asarray(closed["holding_sessions"].dropna().to_numpy(), dtype=float) if "holding_sessions" in closed.columns else np.zeros(0)
    )
    mix: dict[str, int] = {}
    if "exit_reason" in closed.columns:
        counts = closed["exit_reason"].dropna().value_counts()
        mix = {str(key): int(value) for key, value in counts.items()}
    return TradeMetrics(
        band=band,
        n_trades=int(pnl.size),
        hit_rate=float(wins.size / pnl.size) if pnl.size else float("nan"),
        avg_win=float(np.mean(wins)) if wins.size else float("nan"),
        avg_loss=float(np.mean(losses)) if losses.size else float("nan"),
        profit_factor=(float(np.sum(wins)) / loss_sum) if loss_sum > 0.0 else float("inf") if wins.size else float("nan"),
        expectancy=float(np.mean(pnl)) if pnl.size else float("nan"),
        total_pnl=float(np.sum(pnl)),
        worst_trade=float(np.min(pnl)) if pnl.size else float("nan"),
        best_trade=float(np.max(pnl)) if pnl.size else float("nan"),
        avg_holding_sessions=float(np.mean(holding)) if holding.size else float("nan"),
        forced_exits=int(closed["forced"].astype(bool).sum()) if "forced" in closed.columns else 0,
        exit_reason_mix=mix,
    )


def trade_metrics_by_slice(trades: pd.DataFrame, by: str, band: str) -> dict[str, TradeMetrics]:
    """The 12.2 slices over closed trades (underlying, structure, exit reason, ...) - one `TradeMetrics` per value."""
    if trades is None or trades.empty or by not in trades.columns:
        return {}
    return {str(value): trade_metrics(group, band) for value, group in trades.groupby(by, dropna=False)}


def max_adverse_excursion(trades: pd.DataFrame, position_marks: pd.DataFrame, band: str) -> pd.DataFrame:
    """Per-trade maximum adverse excursion in cents (12.2), from the per-position MARK values.

    The unrealised P&L of an open structure at a mark is `(-liq_value - open_net) * 100 * qty`: `liq_value` is what
    closing would COST (negative when it would pay us, 10.5) and `open_net` is the signed net actually paid at entry.
    The MAE of a trade is the most negative such value over the sessions it was held - the drawdown a live operator
    would have had to sit through, which an end-to-end P&L number hides.
    """
    columns = ("position_id", "mae_cents", "mfe_cents", "n_marks")
    if trades is None or trades.empty or position_marks is None or position_marks.empty:
        return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})
    open_net_column = f"open_net_{band}"
    records: list[dict[str, Any]] = []
    marks_by_position = dict(iter(position_marks.groupby("position_id")))
    for _, trade in trades.iterrows():
        marks = marks_by_position.get(trade["position_id"])
        if marks is None or marks.empty:
            continue
        entry_session = trade["entry_session"]
        exit_session = trade["exit_session"]
        window = marks[marks["session"] >= entry_session]
        if exit_session is not None and not pd.isna(exit_session):
            window = window[window["session"] <= exit_session]
        if window.empty:
            continue
        liq = np.asarray(window["liq_value"].astype(float).to_numpy(), dtype=float)
        open_net = float(trade[open_net_column] or 0.0)
        qty = float(trade["qty"] or 0)
        unrealised = (-liq - open_net) * CONTRACT_MULTIPLIER * qty
        records.append(
            {
                "position_id": trade["position_id"],
                "mae_cents": float(np.min(unrealised)),
                "mfe_cents": float(np.max(unrealised)),
                "n_marks": int(unrealised.size),
            }
        )
    return pd.DataFrame.from_records(records) if records else pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


# ======================================================================================================================
# Exposure and costs (12.2)
# ======================================================================================================================


class ExposureMetrics(msgspec.Struct, frozen=True, kw_only=True):
    avg_bp_utilisation: float
    max_bp_utilisation: float
    avg_max_loss_utilisation: float  # aggregate open max loss over equity, headline band
    max_max_loss_utilisation: float
    avg_net_delta: float  # contracts-equivalent delta (the ledger's milli units / 1000)
    avg_net_vega: float
    max_loss_days_at_risk: int  # sum over sessions of the open max loss, in cents ("max-loss-days at risk")
    turnover: float  # traded notional over mean equity


def exposure_metrics(marks: pd.DataFrame, *, band: str = "orats", fills: pd.DataFrame | None = None) -> ExposureMetrics:
    """Average and maximum buying-power and max-loss utilisation, mean greeks and turnover (12.2)."""
    if marks is None or marks.empty:
        return ExposureMetrics(
            avg_bp_utilisation=float("nan"),
            max_bp_utilisation=float("nan"),
            avg_max_loss_utilisation=float("nan"),
            max_max_loss_utilisation=float("nan"),
            avg_net_delta=float("nan"),
            avg_net_vega=float("nan"),
            max_loss_days_at_risk=0,
            turnover=float("nan"),
        )
    bp = np.asarray(marks["bp_utilisation_ppm"].dropna().to_numpy(), dtype=float) / 1_000_000.0
    equity = np.asarray(marks[f"equity_{band}"].astype(float).to_numpy(), dtype=float)
    open_max_loss = np.asarray(marks["open_max_loss"].fillna(0).astype(float).to_numpy(), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        utilisation = np.where(equity > 0.0, open_max_loss / equity, np.nan)
    delta = np.asarray(marks["net_delta_milli"].fillna(0).astype(float).to_numpy(), dtype=float) / 1000.0
    vega = np.asarray(marks["net_vega_milli"].fillna(0).astype(float).to_numpy(), dtype=float) / 1000.0
    mean_equity = float(np.mean(equity)) if equity.size else 0.0
    traded = 0.0
    if fills is not None and not fills.empty:
        net = np.asarray(fills[f"net_{band}"].fillna(0).astype(float).to_numpy(), dtype=float)
        qty = np.asarray(fills["qty"].fillna(0).astype(float).to_numpy(), dtype=float)
        traded = float(np.sum(np.abs(net) * CONTRACT_MULTIPLIER * qty))
    return ExposureMetrics(
        avg_bp_utilisation=float(np.mean(bp)) if bp.size else float("nan"),
        max_bp_utilisation=float(np.max(bp)) if bp.size else float("nan"),
        avg_max_loss_utilisation=float(np.nanmean(utilisation)) if utilisation.size else float("nan"),
        max_max_loss_utilisation=float(np.nanmax(utilisation)) if utilisation.size else float("nan"),
        avg_net_delta=float(np.mean(delta)) if delta.size else float("nan"),
        avg_net_vega=float(np.mean(vega)) if vega.size else float("nan"),
        max_loss_days_at_risk=int(np.sum(open_max_loss)),
        turnover=(traded / mean_equity) if mean_equity > 0.0 else float("nan"),
    )


def risk_normalised_pnl(total_pnl: float, marks: pd.DataFrame) -> float:
    """**Risk-normalised P&L** of 12.2: `pnl / sum over sessions of open max_loss` - P&L per unit of max-loss-days at
    risk. This is the number that survives a change of account size, and the one every baseline is compared on."""
    if marks is None or marks.empty or "open_max_loss" not in marks.columns:
        return float("nan")
    at_risk = float(marks["open_max_loss"].fillna(0).astype(float).sum())
    if at_risk <= 0.0:
        return float("nan")
    return float(total_pnl) / at_risk


class CostMetrics(msgspec.Struct, frozen=True, kw_only=True):
    fees_cents: int
    slippage_cents: float  # sum((net[headline] - net[mid]) * 100 * qty): what the fill model says we paid away
    gross_pnl_mid: float
    slippage_frac_of_gross: float


def cost_metrics(fills: pd.DataFrame, trades: pd.DataFrame, *, headline: str = "orats") -> CostMetrics:
    """Fees and slippage paid (12.2): `sum(net[headline] - net[mid])` relative to gross P&L at mid.

    A buy's headline price is above its mid and a sell-to-close's is below, so the signed difference is positive
    whenever the fill model charged us - there is no sign convention to remember at the call site.
    """
    if fills is None or fills.empty:
        return CostMetrics(fees_cents=0, slippage_cents=0.0, gross_pnl_mid=0.0, slippage_frac_of_gross=float("nan"))
    fees_micro = float(fills["fees_micro"].fillna(0).astype(float).sum())
    head = np.asarray(fills[f"net_{headline}"].fillna(0).astype(float).to_numpy(), dtype=float)
    mid = np.asarray(fills["net_mid"].fillna(0).astype(float).to_numpy(), dtype=float)
    qty = np.asarray(fills["qty"].fillna(0).astype(float).to_numpy(), dtype=float)
    slippage = float(np.sum((head - mid) * CONTRACT_MULTIPLIER * qty))
    gross_mid = float(trade_metrics(trades, "mid").total_pnl) if trades is not None else 0.0
    return CostMetrics(
        fees_cents=round(fees_micro / 10_000.0),  # micro-dollars (1e-6 USD) -> cents (1e-2 USD)
        slippage_cents=slippage,
        gross_pnl_mid=gross_mid,
        slippage_frac_of_gross=(slippage / abs(gross_mid)) if gross_mid else float("nan"),
    )


# ======================================================================================================================
# Attribution (12.2: "so beta is not mistaken for skill")
# ======================================================================================================================


class AttributionResult(msgspec.Struct, frozen=True, kw_only=True):
    n: int
    names: tuple[str, ...]
    alpha: float
    coefficients: dict[str, float]
    r_squared: float
    residual_std: float


def attribution(pnl: np.ndarray, factors: Mapping[str, np.ndarray]) -> AttributionResult:
    """OLS of daily P&L on its factors (numpy, 12.2) - typically the underlying return and the IV30 change.

    The factor series come from the `daily` tables, not from the run store, so the caller passes them in; the run store
    holds no market data of its own.
    """
    y = np.asarray(pnl, dtype=float)
    names = tuple(factors)
    n = int(y.size)
    if n == 0 or any(np.asarray(factors[name]).size != n for name in names):
        raise ValueError("attribution: the P&L series and every factor must have the same non-zero length")
    design = np.column_stack([np.ones(n), *[np.asarray(factors[name], dtype=float) for name in names]])
    if n <= design.shape[1]:
        return AttributionResult(
            n=n,
            names=names,
            alpha=float("nan"),
            coefficients=dict.fromkeys(names, float("nan")),
            r_squared=float("nan"),
            residual_std=float("nan"),
        )
    beta, _residuals, _rank, _sv = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    resid = y - fitted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return AttributionResult(
        n=n,
        names=names,
        alpha=float(beta[0]),
        coefficients={name: float(beta[index + 1]) for index, name in enumerate(names)},
        r_squared=(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
        residual_std=float(np.sqrt(ss_res / (n - design.shape[1]))),
    )


# ======================================================================================================================
# The decision process (12.2) - the DECISION x RISK_VERDICT funnel and the rates around it
# ======================================================================================================================

FUNNEL_EMITTED: Final[str] = "order emitted"
FUNNEL_STAGES: Final[tuple[str, ...]] = ("rules", "post_decision", "emitted")


def abstention_funnel(decisions: pd.DataFrame, verdicts: pd.DataFrame) -> pd.DataFrame:
    """The abstention funnel by first failing step - **the join of DECISION and RISK_VERDICT on `decision_id`** (12.2).

    For each entry DECISION: the first `EntryDecision.reasons` code when the rules abstained (`action = "no_trade"`),
    else the first `reject_codes` entry of its RISK_VERDICT (`gate:*`, `candidate:*`, `risk:*`), else "order emitted".
    `EntryDecision.reasons` never contains a post-decision code by construction (7.9), which is precisely why the join
    is needed: without the verdicts the funnel would end at "no rules reason" and hide every gate, candidate and risk
    rejection.

    Returns `stage, code, n`, ordered by stage then by descending count.
    """
    columns = ("stage", "code", "n")
    if decisions is None or decisions.empty:
        return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})
    entries = decisions[decisions["kind"] == "entry"] if "kind" in decisions.columns else decisions
    if entries.empty:
        return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})

    first_reject: dict[str, str] = {}
    if verdicts is not None and not verdicts.empty:
        for _, verdict in verdicts.sort_values("seq").iterrows():
            codes = tuple(verdict["reject_codes"] or ())
            decision_id = verdict["decision_id"]
            if codes and decision_id not in first_reject:
                first_reject[str(decision_id)] = str(codes[0])

    counts: dict[tuple[str, str], int] = {}
    for _, decision in entries.iterrows():
        reasons = tuple(decision["reasons"] or ())
        if str(decision.get("action")) == "no_trade":
            code = str(reasons[0]) if reasons else "no_trade"
            key = ("rules", code)
        else:
            reject = first_reject.get(str(decision["decision_id"]))
            key = ("post_decision", reject) if reject else ("emitted", FUNNEL_EMITTED)
        counts[key] = counts.get(key, 0) + 1
    records = [{"stage": stage, "code": code, "n": n} for (stage, code), n in counts.items()]
    frame = pd.DataFrame.from_records(records)
    frame["_order"] = frame["stage"].map({stage: index for index, stage in enumerate(FUNNEL_STAGES)})
    return frame.sort_values(["_order", "n", "code"], ascending=[True, False, True]).drop(columns="_order").reset_index(drop=True)


class DecisionProcessMetrics(msgspec.Struct, frozen=True, kw_only=True):
    n_decisions: int
    n_entry_decisions: int
    n_manage_decisions: int
    n_entries_emitted: int
    veto_rates: dict[str, float]  # question id -> share of entry decisions vetoed by it (7.4 bands pooled)
    crosscheck_reject_rates: dict[str, float]
    perturbation_flip_rates: dict[str, float]  # variant -> share of decisions where the variant disagreed (7.8)
    risk_reject_rates: dict[str, float]  # risk check code -> share of risk verdicts rejecting on it
    candidate_reject_rates: dict[str, float]
    exceeds_risk_budget_by_pair: dict[str, float]  # "<underlying>/<kind>" -> rate
    size_zero_by_pair: dict[str, float]
    fill_reject_counts: dict[str, int]
    forced_fills: int
    degraded_fills: int
    zero_bid_close_legs: int
    anomaly_counts: dict[str, int]
    assignment_sims: int
    kill_events: int
    halt_events: int


def decision_process_metrics(
    *,
    decisions: pd.DataFrame,
    verdicts: pd.DataFrame,
    fills: pd.DataFrame | None = None,
    order_status: pd.DataFrame | None = None,
    anomalies: pd.DataFrame | None = None,
    risk_events: pd.DataFrame | None = None,
    kills: pd.DataFrame | None = None,
) -> DecisionProcessMetrics:
    """Every "Decision process" rate of 12.2, computed from the ledger frames alone."""
    entries = decisions[decisions["kind"] == "entry"] if decisions is not None and not decisions.empty else _empty_like()
    manages = decisions[decisions["kind"] == "manage"] if decisions is not None and not decisions.empty else _empty_like()
    n_entries = len(entries)

    veto_counts: dict[str, int] = {}
    crosscheck_counts: dict[str, int] = {}
    variant_runs: dict[str, int] = {}
    variant_flips: dict[str, int] = {}
    for _, decision in entries.iterrows():
        for reason in tuple(decision["reasons"] or ()):
            text = str(reason)
            if text.startswith("veto:") and not text.startswith("veto:regime:"):
                qid = text.split(":")[1]
                veto_counts[qid] = veto_counts.get(qid, 0) + 1
            elif text.startswith("crosscheck:"):
                name = text.split(":", 1)[1]
                crosscheck_counts[name] = crosscheck_counts.get(name, 0) + 1
        for variant, agreed in dict(decision["variant_agreement"] or {}).items():
            variant_runs[variant] = variant_runs.get(variant, 0) + 1
            if not agreed:
                variant_flips[variant] = variant_flips.get(variant, 0) + 1

    risk_counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}
    budget_by_pair: dict[str, int] = {}
    size_zero_by_pair: dict[str, int] = {}
    pair_totals: dict[str, int] = {}
    n_emitted = 0
    pair_of_decision = _pair_index(decisions)
    if verdicts is not None and not verdicts.empty:
        for _, verdict in verdicts.iterrows():
            pair = pair_of_decision.get(str(verdict["decision_id"]))
            if pair is not None:
                pair_totals[pair] = pair_totals.get(pair, 0) + 1
            if verdict["approved"]:
                n_emitted += 1
            for code in tuple(verdict["reject_codes"] or ()):
                text = str(code)
                if text.startswith("risk:"):
                    risk_counts[text[len("risk:") :]] = risk_counts.get(text[len("risk:") :], 0) + 1
                    if text == f"risk:{vocab.SIZE_ZERO}" and pair is not None:
                        size_zero_by_pair[pair] = size_zero_by_pair.get(pair, 0) + 1
                elif text.startswith("candidate:"):
                    name = text[len("candidate:") :]
                    candidate_counts[name] = candidate_counts.get(name, 0) + 1
                    if name == "exceeds_risk_budget" and pair is not None:
                        budget_by_pair[pair] = budget_by_pair.get(pair, 0) + 1
    n_verdicts = len(verdicts) if verdicts is not None else 0

    fill_rejects: dict[str, int] = {}
    forced = degraded = zero_bid = 0
    if fills is not None and not fills.empty:
        forced = int(fills["forced"].astype(bool).sum())
        degraded = int((fills["quality"] == "degraded").sum())
        zero_bid = int(fills["zero_bid_close_legs"].fillna(0).astype(int).sum())
        for codes in fills["model_reject"]:
            for code in tuple(codes or ()):
                fill_rejects[str(code)] = fill_rejects.get(str(code), 0) + 1
    if order_status is not None and not order_status.empty and "tag" in order_status.columns:
        for tag in order_status["tag"].dropna():
            if str(tag) in vocab.FILL_REJECTS:
                fill_rejects[str(tag)] = fill_rejects.get(str(tag), 0) + 1

    anomaly_counts: dict[str, int] = {}
    if anomalies is not None and not anomalies.empty:
        anomaly_counts = {str(key): int(value) for key, value in anomalies["type"].dropna().value_counts().items()}

    halt_events = 0
    if risk_events is not None and not risk_events.empty:
        halting = {"halt_set", "daily_loss_halt"}
        halt_events = int(risk_events["type"].isin(halting).sum())

    return DecisionProcessMetrics(
        n_decisions=len(decisions) if decisions is not None else 0,
        n_entry_decisions=n_entries,
        n_manage_decisions=len(manages),
        n_entries_emitted=n_emitted,
        veto_rates=_rates(veto_counts, n_entries),
        crosscheck_reject_rates=_rates(crosscheck_counts, n_entries),
        perturbation_flip_rates={variant: (variant_flips.get(variant, 0) / runs) for variant, runs in variant_runs.items() if runs},
        risk_reject_rates=_rates(risk_counts, n_verdicts),
        candidate_reject_rates=_rates(candidate_counts, n_verdicts),
        exceeds_risk_budget_by_pair={pair: budget_by_pair.get(pair, 0) / total for pair, total in pair_totals.items() if total},
        size_zero_by_pair={pair: size_zero_by_pair.get(pair, 0) / total for pair, total in pair_totals.items() if total},
        fill_reject_counts=fill_rejects,
        forced_fills=forced,
        degraded_fills=degraded,
        zero_bid_close_legs=zero_bid,
        anomaly_counts=anomaly_counts,
        assignment_sims=anomaly_counts.get("assignment_sim", 0),
        kill_events=len(kills) if kills is not None else 0,
        halt_events=halt_events,
    )


class JevUsage(msgspec.Struct, frozen=True, kw_only=True):
    n_requests: int
    cache_hit_rate: float
    input_tokens: int
    cost_usd: float
    latency_p50_ms: float
    latency_p95_ms: float


def jev_usage(decisions: pd.DataFrame, sidecar: pd.DataFrame) -> JevUsage:
    """Requests, cache hit rate, input tokens, cost and latency (12.2).

    Token counts, request ids and latencies live in the unhashed sidecar, never in the ledger (2.7, INV-24), so this is
    the one metric that reads the sidecar table.
    """
    n_requests = 0
    if decisions is not None and not decisions.empty and "n_requests" in decisions.columns:
        n_requests = int(decisions["n_requests"].fillna(0).astype(int).sum())
    tokens = 0
    hit_rate = float("nan")
    p50 = p95 = float("nan")
    if sidecar is not None and not sidecar.empty:
        tokens = int(sidecar["input_tokens"].dropna().astype(int).sum())
        hits = sidecar["cache_hit"].dropna()
        if len(hits):
            hit_rate = float(hits.astype(bool).mean())
        latency = np.asarray(sidecar["latency_ms"].dropna().astype(float).to_numpy(), dtype=float)
        if latency.size:
            p50 = float(np.percentile(latency, 50))
            p95 = float(np.percentile(latency, 95))
    return JevUsage(
        n_requests=n_requests,
        cache_hit_rate=hit_rate,
        input_tokens=tokens,
        cost_usd=tokens / 1_000_000.0 * JEV_COST_USD_PER_MTOK,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
    )


# ======================================================================================================================
# helpers
# ======================================================================================================================


def _rates(counts: Mapping[str, int], total: int) -> dict[str, float]:
    if total <= 0:
        return dict.fromkeys(counts, float("nan"))
    return {key: value / total for key, value in counts.items()}


def _pair_index(decisions: pd.DataFrame) -> dict[str, str]:
    """`decision_id -> "<underlying>/<structure kind>"` for the per-(underlying, kind) rates of 12.2."""
    if decisions is None or decisions.empty:
        return {}
    out: dict[str, str] = {}
    for _, row in decisions.iterrows():
        underlying = row.get("underlying")
        kind = row.get("structure_kind")
        if underlying is None or kind is None or pd.isna(underlying) or pd.isna(kind):
            continue
        out[str(row["decision_id"])] = f"{underlying}/{kind}"
    return out


def _empty_like() -> pd.DataFrame:
    return pd.DataFrame({"reasons": pd.Series(dtype="object"), "variant_agreement": pd.Series(dtype="object")})


def metrics_tree(values: Sequence[tuple[str, float]]) -> dict[str, float]:
    """Flatten `(name, value)` pairs into the `report.json` metrics tree shape (12.8: floats, 6 dp, never hashed)."""
    return {name: round(float(value), 6) for name, value in values}
