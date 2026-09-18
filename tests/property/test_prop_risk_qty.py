"""Property: `RiskEngine.approve()` may only REDUCE a quantity, and every 9.1 limit holds after it (DESIGN.md 15.1 `risk`).

For random portfolios and random attempt limits:

* `0 <= verdict.qty_approved <= intent.qty` - the engine can never increase a quantity, loosen a price or change legs;
* an approved order satisfies checks 15-18 at **1.0x**, recomputed here from the 9.2 formulas of a credit vertical
  (`max_loss_pc = (width + net) * 100 + fee_rt`, `bp_required_pc = (width + net) * 100`) rather than by calling `structmath`;
* the approved quantity is MAXIMAL: one more contract would breach at least one of those limits;
* a rejected verdict carries `qty_approved == 0`, no `ApprovedOrder` and at least one reject code.

`fee_rt` is pinned by handing `approve()` a `Candidate` whose `max_loss_per_contract` encodes it exactly (9.2), so the test
knows the number without asking the code for it. Seeded numpy generators, no hypothesis (1.1).
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Any

import numpy as np

from jevbot.cal import XnysCalendar
from jevbot.config import Config
from jevbot.risk import DefaultRiskEngine
from jevbot.types import (
    BandPrices,
    Candidate,
    EntryContext,
    KillState,
    Leg,
    OrderIntent,
    OrderLeg,
    OrderPurpose,
    PortfolioState,
    Position,
    PositionIntent,
    Right,
    Side,
    Slot,
    SnapshotKey,
    Structure,
    StructureKind,
)
from tests.fixtures.chain_factory import contract_at, make_chain, make_structure, quote_of, set_quote, target_expiry

CAL = XnysCalendar()
CFG = Config()
SESSION = date(2024, 5, 17)
KEY = SnapshotKey(session=SESSION, slot=Slot.EOD)
MULTIPLIER = 100
WIDTH = 500  # the $5-wide put credit spread below
PER_TRADE_PCT = (1, 100)  # risk.max_loss_per_trade_pct = 0.01
AGGREGATE_PCT = (10, 100)  # risk.max_aggregate_open_loss_pct = 0.10
BP_PCT = (50, 100)  # risk.max_bp_utilisation = 0.50
MAX_CONTRACTS = 10
MAX_NOTIONAL_CENTS = 15_000 * MULTIPLIER

ENTRY_CTX = EntryContext(
    entry_thesis="t",
    entry_codes={"trend": "up", "iv_vs_realized": "rich", "iv_rank": "p60_p80"},
    entry_spot=45_000,
    entry_iv30_bp=1800,
    entry_em_hold_tenths=40,
    open_mid_at_decision=-250,
)


def _narrow() -> tuple[Any, Structure]:
    chain = make_chain()
    expiry = target_expiry(chain, 35)
    short = contract_at(chain, expiry, Right.PUT, 436_000)
    long = contract_at(chain, expiry, Right.PUT, 431_000)
    chain = set_quote(chain, short, bid=400, ask=404)
    chain = set_quote(chain, long, bid=150, ask=154)
    structure = Structure(
        kind=StructureKind.PUT_CREDIT,
        underlying="SPY",
        expiry=expiry,
        last_session=chain.last_session(expiry),
        legs=(Leg(contract=long, side=Side.BUY), Leg(contract=short, side=Side.SELL)),
    )
    return chain, structure


CHAIN, NARROW = _narrow()
# bearish structures on other underlyings: they fill the aggregate and the buying power without touching check 12
OTHERS = tuple(
    make_structure(make_chain(underlying, spot=spot), StructureKind.CALL_CREDIT) for underlying, spot in (("QQQ", 38_000), ("IWM", 20_000))
)


def view() -> Any:
    from tests.fixtures.fake_view import FakeView

    return FakeView(key=KEY, as_of=CAL.open_close(SESSION)[1], calendar=CAL, chains=(CHAIN,))


VIEW = view()


def rng(seed: int, purpose: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}|{purpose}".encode()).hexdigest()
    return np.random.Generator(np.random.PCG64(int(digest[:16], 16)))


def floor_pct(pct: tuple[int, int], value: int) -> int:
    return value * pct[0] // pct[1]


def max_loss_pc(limit: int, fee_rt: int) -> int:
    """9.2, credit vertical: `(w - credit) * 100 + fee_rt` with `credit = -net`."""
    return (WIDTH + limit) * MULTIPLIER + fee_rt


def bp_required_pc(limit: int) -> int:
    """9.2, credit vertical at `bp_haircut_mult = 1.0`: `(w - credit) * 100`, never negative."""
    return max((WIDTH + limit) * MULTIPLIER, 0)


def candidate(fee_rt: int, net_worst: int) -> Candidate:
    return Candidate(
        structure=NARROW,
        key=KEY,
        dte=35,
        sessions_to_expiry=CAL.sessions_between(SESSION, NARROW.last_session),
        quotes=tuple(quote_of(CHAIN, leg.contract) for leg in NARROW.legs),
        net=BandPrices(orats=net_worst - 4, worst=net_worst, mid=net_worst - 6),
        budget_floor=50_000,
        max_loss_per_contract=max_loss_pc(net_worst, fee_rt),
        max_profit_per_contract=-net_worst * MULTIPLIER,
        bp_required_per_contract=bp_required_pc(net_worst),
        breakevens=(43_600 + net_worst,),
        short_distance_em=1.0,
        net_delta=0.1,
        net_vega=-0.3,
    )


def intent(qty: int, limit_natural: int) -> OrderIntent:
    legs = tuple(
        OrderLeg(
            contract=leg.contract,
            side=leg.side,
            position_intent=PositionIntent.BTO if leg.side is Side.BUY else PositionIntent.STO,
        )
        for leg in NARROW.legs
    )
    return OrderIntent(
        intent_id="jb1-aaaaaaaa-240517-dddddddddddd-open-00",
        decision_id="d" * 32,
        position_id="p" * 16,
        purpose=OrderPurpose.OPEN,
        part=0,
        underlying="SPY",
        legs=legs,
        qty=qty,
        limit_start=limit_natural,
        limit_natural=limit_natural,
        reason="entry",
        mandatory=False,
        session=SESSION,
        key=KEY,
        tier_ppm=1_000_000,
        structure=NARROW,
        entry_ctx=ENTRY_CTX,
    )


def position(structure: Structure, *, index: int, max_loss: int, bp_reserved: int) -> Position:
    return Position(
        position_id=f"other{index}",
        structure=structure,
        qty=1,
        open_key=KEY,
        open_decision_id="d" * 32,
        open_net=BandPrices(orats=-200, worst=-190, mid=-205),
        max_loss=max_loss,
        max_profit=20_000,
        bp_reserved=bp_reserved,
        entry=ENTRY_CTX,
        liq_value=100,
        mid_value=100,
    )


def portfolio(**overrides: Any) -> PortfolioState:
    base: dict[str, Any] = {
        "key": KEY,
        "cash": BandPrices(orats=10_000_000, worst=10_000_000, mid=10_000_000),
        "equity": BandPrices(orats=10_000_000, worst=10_000_000, mid=10_000_000),
        "positions": (),
        "working": (),
        "peak_equity": 10_000_000,
        "day_start_equity": 10_000_000,
        "opened_today": 0,
        "fees_accrued_micro": 0,
        "halt_entries": False,
        "halt_reasons": (),
        "kill_state": KillState.ARMED,
        "kill_event_id": None,
        "cooldowns": (),
        "jev_fail_sessions": 0,
        "stale_sessions": 0,
        "broker_fail_streak": 0,
        "orders_last_minute": 0,
    }
    base.update(overrides)
    return PortfolioState(**base)


def fits(qty: int, *, limit: int, fee_rt: int, equity: int, held_loss: int, reserved: int, broker_bp: int | None) -> bool:
    """Every 9.1 quantity limit at 1.0x, recomputed here from the 9.2 formulas."""
    loss = max_loss_pc(limit, fee_rt)
    bp = bp_required_pc(limit)
    internal = floor_pct(BP_PCT, equity) - reserved
    available = internal if broker_bp is None else min(internal, broker_bp)
    return (
        qty >= 1
        and qty * loss <= floor_pct(PER_TRADE_PCT, equity)
        and held_loss + qty * loss <= floor_pct(AGGREGATE_PCT, equity)
        and (bp <= 0 or qty * bp <= available)
        and qty <= MAX_CONTRACTS
        and qty * abs(limit) * MULTIPLIER <= MAX_NOTIONAL_CENTS
    )


def test_approve_only_reduces_and_every_quantity_limit_holds_afterwards() -> None:
    engine = DefaultRiskEngine(CFG)
    approvals = 0
    reductions = 0
    for seed in range(120):
        draw = rng(seed, "risk-qty")
        equity = int(draw.integers(2_000_000, 60_000_000))
        fee_rt = int(draw.integers(0, 60))
        natural = -int(draw.integers(20, 60))  # the least credit on the tape: any limit below it is not "beyond natural"
        limit = -int(draw.integers(60, WIDTH - 20))
        requested = int(draw.integers(1, 12))
        held = []
        held_loss = 0
        reserved = 0
        for index in range(int(draw.integers(0, 3))):
            loss = int(draw.integers(0, 800_000))
            bp = int(draw.integers(0, 3_000_000))
            held.append(position(OTHERS[index % len(OTHERS)], index=index, max_loss=loss, bp_reserved=bp))
            held_loss += loss
            reserved += bp
        broker_bp = int(draw.integers(0, 6_000_000)) if draw.random() < 0.4 else None
        pf = portfolio(
            equity=BandPrices(orats=equity, worst=equity, mid=equity),
            cash=BandPrices(orats=equity, worst=equity, mid=equity),
            positions=tuple(held),
            opened_today=int(draw.integers(0, 2)),
            broker_options_bp=broker_bp,
        )
        order_intent = intent(requested, natural)
        verdict, order = engine.approve(
            order_intent,
            pf,
            VIEW,
            now=VIEW.as_of,
            limit=limit,
            cand=candidate(fee_rt, natural),
        )
        assert 0 <= verdict.qty_approved <= requested, (seed, verdict.reject_codes)
        if order is None:
            assert verdict.qty_approved == 0 and verdict.reject_codes, seed
            assert not fits(
                1, limit=limit, fee_rt=fee_rt, equity=equity, held_loss=held_loss, reserved=reserved, broker_bp=broker_bp
            ) or any(
                not code.startswith(("risk:max_loss_per_trade", "risk:agg_max_loss", "risk:buying_power", "risk:size_zero"))
                for code in verdict.reject_codes
            ), (seed, verdict.reject_codes)
            continue
        approvals += 1
        qty = order.qty
        assert qty == verdict.qty_approved and order.limit == limit and order.intent is order_intent
        assert fits(qty, limit=limit, fee_rt=fee_rt, equity=equity, held_loss=held_loss, reserved=reserved, broker_bp=broker_bp), (
            seed,
            qty,
        )
        if qty < requested:
            reductions += 1
            # maximal: one more contract breaches at least one 9.1 limit
            assert not fits(
                qty + 1, limit=limit, fee_rt=fee_rt, equity=equity, held_loss=held_loss, reserved=reserved, broker_bp=broker_bp
            ), (seed, qty)
        assert verdict.max_loss == qty * max_loss_pc(limit, fee_rt)
        assert verdict.bp_required == qty * bp_required_pc(limit)
    assert approvals >= 40 and reductions >= 10, (approvals, reductions)


def test_a_rejected_attempt_never_yields_an_order_and_always_names_a_code() -> None:
    engine = DefaultRiskEngine(CFG)
    for seed in range(40):
        draw = rng(seed, "risk-reject")
        pf = portfolio(
            halt_entries=bool(draw.integers(0, 2)),
            kill_state=KillState.ARMED if draw.random() < 0.5 else KillState.TRIPPED,
            opened_today=int(draw.integers(0, 4)),
            orders_last_minute=int(draw.integers(0, 20)),
        )
        verdict, order = engine.approve(intent(3, -40), pf, VIEW, now=VIEW.as_of, limit=-240, cand=candidate(18, -40))
        assert (order is None) == (not verdict.approved)
        assert (order is None) == bool(verdict.reject_codes)
        assert verdict.qty_approved == (0 if order is None else order.qty)
        assert all(code.startswith("risk:") for code in verdict.reject_codes)
        from jevbot.vocab import is_reject_code

        assert all(is_reject_code(code) for code in verdict.reject_codes)
