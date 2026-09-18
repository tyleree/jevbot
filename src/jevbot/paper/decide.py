"""One diagnostic entry cycle on live data. No broker, ledger writes or submission path.

Sizing uses an explicitly hypothetical empty portfolio, not the Alpaca account.
The lean cycle omits perturbations, news, reconciliation and service health gates;
its output must never be used as authorization to submit an order.
"""

from datetime import datetime
from typing import Any

from jevbot.candidates import CandidateGenerator
from jevbot.config import Config, headline_band, load_mask_terms, resolve_path
from jevbot.cycle import decide_entries, namespace_for
from jevbot.data.market import build_market_data
from jevbot.data.view import DataView
from jevbot.features import compute_features
from jevbot.fills import BandFillModel
from jevbot.paper.live_data import AlpacaLiveProvider
from jevbot.portfolio import Book
from jevbot.protocols import Calendar, Decider
from jevbot.risk import DefaultRiskEngine
from jevbot.rules import DecisionRules
from jevbot.state import StateBuilder


def decide_snapshot(cfg: Config, provider: AlpacaLiveProvider, calendar: Calendar, decider: Decider, as_of: datetime) -> dict[str, Any]:
    """Run the real state -> decider -> rules -> candidates -> risk pipeline without executing its intents."""
    market = build_market_data(
        cfg,
        calendar,
        provider,
        vol_index=provider.vol_index(),
        iv_proxy=provider.iv_proxy(),
        rate_bp=round(provider.rate * 10000),
    )
    view = DataView(
        key=provider.key, as_of=as_of, calendar=calendar, chains=provider, tables=market.tables, news=market.news, events=market.events
    )
    builder = StateBuilder(cfg, load_mask_terms(resolve_path(cfg.news.mask_terms_file)), news_resolved=False)
    book = Book(initial_cash=cfg.run.initial_equity_usd * 100, headline=headline_band(cfg), cfg=cfg, calendar=calendar)
    writes, orders = decide_entries(
        cfg=cfg,
        namespace=namespace_for(cfg, decider),
        session=provider.key.session,
        as_of=as_of,
        view=view,
        decider=decider,
        rules=DecisionRules(cfg.rules, cfg.structures.enabled),
        state_builder=builder,
        candidates=CandidateGenerator(cfg, BandFillModel(cfg)),
        risk=DefaultRiskEngine(cfg),
        pf_state=book.state,
    )
    diagnostics = {}
    for underlying in cfg.universe.underlyings:
        features = compute_features(view, underlying, cfg.dte.hold_horizon_sessions, data=cfg.data)
        diagnostics[underlying] = {
            "missing_required": list(features.missing_required()),
            "iv_history_proxy_pct": features.iv_hist_proxy_pct,
        }
    return {
        "dry_run": True,
        "orders_submitted": 0,
        "session": provider.key.session.isoformat(),
        "slot": provider.key.slot.value,
        "as_of": as_of.isoformat(),
        "model": decider.model,
        "fidelity": provider.fidelity.value,
        "hypothetical_equity_usd": cfg.run.initial_equity_usd,
        "portfolio_assumption": "empty; not reconciled with the broker",
        "limitations": [
            "indicative quotes",
            "proxy IV history",
            "no news or event feed",
            "no perturbation confirmation",
            "no reconciliation or service health gates",
        ],
        "diagnostics": diagnostics,
        "records": [{"kind": kind.value, "payload": payload} for kind, payload in writes],
        "hypothetical_orders": len(orders),
    }
