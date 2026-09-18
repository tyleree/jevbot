"""`risk.DefaultRiskEngine` - the 21 ordered checks of DESIGN.md 9.1, the 9.3 sizing, the 9.4 hard exits and the 9.5 triggers.

Every numeric expectation is hand-computed from the 9.2 / 10.7 formulas and written as a literal; check 5's max-loss claim is
verified against an INDEPENDENT expiry-payoff evaluation (`payoff_max_loss` below), not by calling `structmath` twice.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Any

import msgspec
import pytest

from jevbot.cal import XnysCalendar
from jevbot.config import Config, RiskConfig
from jevbot.errors import InvariantError
from jevbot.risk import DefaultRiskEngine, headline_band
from jevbot.types import (
    ApprovedOrder,
    Band,
    BandPrices,
    Candidate,
    ClockReading,
    EntryContext,
    ExitReason,
    FillRule,
    HealthSnapshot,
    KillState,
    KillTrigger,
    Leg,
    OptionContract,
    OrderIntent,
    OrderLeg,
    OrderPurpose,
    PortfolioState,
    Position,
    PositionIntent,
    Right,
    RiskCheck,
    RiskVerdict,
    Side,
    Slot,
    SnapshotKey,
    Structure,
    StructureKind,
    TriggerAction,
)
from tests.fixtures.chain_factory import (
    contract_at,
    make_chain,
    make_structure,
    quote_of,
    set_quote,
    target_expiry,
)
from tests.fixtures.fake_view import FakeView, ex_dividend_event, fomc_event

CAL = XnysCalendar()
SESSION = date(2024, 5, 17)
CLOSE = CAL.open_close(SESSION)[1]
CHAIN = make_chain()
EXPIRY = target_expiry(CHAIN, 35)
KEY = SnapshotKey(session=SESSION, slot=Slot.EOD)
INITIAL_EQUITY = 10_000_000  # $100,000 in cents
NS_DECISION = "d" * 32
MULTIPLIER = 100

ENTRY_CTX = EntryContext(
    entry_thesis="t",
    entry_codes={"trend": "up", "iv_vs_realized": "rich", "iv_rank": "p60_p80"},
    entry_spot=45_000,
    entry_iv30_bp=1800,
    entry_em_hold_tenths=40,
    open_mid_at_decision=-250,
)


# ======================================================================================================================
# world builders
# ======================================================================================================================


def view_of(
    chain: Any = CHAIN,
    *,
    session: date = SESSION,
    slot: Slot = Slot.EOD,
    as_of: datetime | None = None,
    events: Sequence[Any] = (),
) -> FakeView:
    return FakeView(
        key=SnapshotKey(session=session, slot=slot),
        as_of=as_of if as_of is not None else CAL.open_close(session)[1],
        calendar=CAL,
        chains=() if chain is None else (chain,),
        events=events,
    )


def narrow_put_credit(
    chain: Any = CHAIN,
    *,
    short_milli: int = 436_000,
    long_milli: int = 431_000,
    short_quote: tuple[int, int] = (400, 404),
    long_quote: tuple[int, int] = (150, 154),
) -> tuple[Any, Structure]:
    """A $5-wide put credit spread with hand-chosen quotes, so every 9.2 number below is a literal.

    short 436 at 400 x 404, long 431 at 150 x 154: width 500c, worst-band natural = 154 - 400 = -246c.
    """
    expiry = target_expiry(chain, 35)
    short = contract_at(chain, expiry, Right.PUT, short_milli)
    long = contract_at(chain, expiry, Right.PUT, long_milli)
    chain = set_quote(chain, short, bid=short_quote[0], ask=short_quote[1])
    chain = set_quote(chain, long, bid=long_quote[0], ask=long_quote[1])
    structure = Structure(
        kind=StructureKind.PUT_CREDIT,
        underlying=chain.underlying,
        expiry=expiry,
        last_session=chain.last_session(expiry),
        legs=(Leg(contract=long, side=Side.BUY), Leg(contract=short, side=Side.SELL)),
    )
    return chain, structure


NARROW_CHAIN, NARROW = narrow_put_credit()
OTHER = make_structure(make_chain("QQQ", spot=38_000), StructureKind.CALL_CREDIT)  # bearish QQQ: unrelated to NARROW
NARROW_VIEW = view_of(NARROW_CHAIN)
# hand-computed for NARROW: fee_rt = ceil((167,780 + 3,132 + 8,282) micro-dollars / 10,000) = 18c (10.7 round trip, mids 152 / 402)
FEE_RT = 18
WIDTH = 500


def open_legs_of(structure: Structure) -> tuple[OrderLeg, ...]:
    return tuple(
        OrderLeg(
            contract=leg.contract,
            side=leg.side,
            position_intent=PositionIntent.BTO if leg.side is Side.BUY else PositionIntent.STO,
        )
        for leg in structure.legs
    )


def close_legs_of(structure: Structure) -> tuple[OrderLeg, ...]:
    return tuple(
        OrderLeg(
            contract=leg.contract,
            side=Side.SELL if leg.side is Side.BUY else Side.BUY,
            position_intent=PositionIntent.STC if leg.side is Side.BUY else PositionIntent.BTC,
        )
        for leg in structure.legs
    )


def intent_of(
    structure: Structure | None,
    *,
    purpose: OrderPurpose = OrderPurpose.OPEN,
    qty: int = 1,
    limit_start: int = -250,
    limit_natural: int = -246,
    session: date = SESSION,
    key: SnapshotKey | None = None,
    mandatory: bool = False,
    part: int = 0,
    legs: tuple[OrderLeg, ...] | None = None,
    underlying: str = "SPY",
    equity: tuple[str, Side, int] | None = None,
) -> OrderIntent:
    if legs is None:
        legs = () if structure is None else (open_legs_of(structure) if purpose is OrderPurpose.OPEN else close_legs_of(structure))
    return OrderIntent(
        intent_id=f"jb1-aaaaaaaa-{session:%y%m%d}-{NS_DECISION[:12]}-{purpose.value}-{part:02d}",
        decision_id=NS_DECISION,
        position_id="p" * 16,
        purpose=purpose,
        part=part,
        underlying=equity[0] if equity is not None else underlying,
        legs=() if equity is not None else legs,
        qty=0 if equity is not None else qty,
        limit_start=limit_start,
        limit_natural=limit_natural,
        reason="entry" if purpose is OrderPurpose.OPEN else ExitReason.KILL.value,
        mandatory=mandatory,
        session=session,
        key=key if key is not None else SnapshotKey(session=session, slot=Slot.EOD),
        tier_ppm=1_000_000 if purpose is OrderPurpose.OPEN else 0,
        structure=None if equity is not None else structure,
        entry_ctx=ENTRY_CTX if purpose is OrderPurpose.OPEN else None,
        equity_symbol=None if equity is None else equity[0],
        equity_side=None if equity is None else equity[1],
        equity_qty=None if equity is None else equity[2],
    )


def position_of(
    structure: Structure,
    *,
    qty: int = 1,
    open_net: BandPrices | None = None,
    max_loss: int = 30_000,
    max_profit: int | None = 20_000,
    bp_reserved: int = 30_000,
    liq_value: int = 100,
    position_id: str = "p" * 16,
) -> Position:
    return Position(
        position_id=position_id,
        structure=structure,
        qty=qty,
        open_key=KEY,
        open_decision_id=NS_DECISION,
        open_net=open_net if open_net is not None else BandPrices(orats=-200, worst=-190, mid=-205),
        max_loss=max_loss,
        max_profit=max_profit,
        bp_reserved=bp_reserved,
        entry=ENTRY_CTX,
        liq_value=liq_value,
        mid_value=liq_value,
    )


def portfolio(**overrides: Any) -> PortfolioState:
    base: dict[str, Any] = {
        "key": KEY,
        "cash": BandPrices(orats=INITIAL_EQUITY, worst=INITIAL_EQUITY, mid=INITIAL_EQUITY),
        "equity": BandPrices(orats=INITIAL_EQUITY, worst=INITIAL_EQUITY, mid=INITIAL_EQUITY),
        "positions": (),
        "working": (),
        "peak_equity": INITIAL_EQUITY,
        "day_start_equity": INITIAL_EQUITY,
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


def engine(cfg: Config | None = None, **meta_overrides: Any) -> DefaultRiskEngine:
    return DefaultRiskEngine(cfg if cfg is not None else Config(), _meta(**meta_overrides) if meta_overrides else None)


def _meta(**overrides: Any) -> Any:
    from jevbot.types import Fidelity, RunMeta, RunMode

    base: dict[str, Any] = {
        "run_id": "r",
        "trial_id": None,
        "experiment": "test",
        "family": "test",
        "namespace": "test:jev-1.13.0:g0",
        "mode": RunMode.BACKTEST,
        "decider": "mock_jev",
        "model": "jev-1.13.0",
        "model_release_date": date(2026, 9, 15),
        "fidelity": Fidelity.EOD_QUOTES,
        "fill_rule": FillRule.NEXT_SNAPSHOT,
        "spot_measure": "parity",
        "news_resolved": False,
        "news_reason": "auto_no_keys",
        "config_hash": "c",
        "state_config_hash": "s",
        "rules_hash": "r",
        "risk_config_hash": "rk",
        "entry_qset_hash": "e",
        "entry_text_qset_hash": "et",
        "manage_qset_hash": "m",
        "manage_text_qset_hash": "mt",
        "git_commit": None,
        "data_manifest_hash": "d",
        "cache_manifest_hash": None,
        "start": SESSION,
        "end": None,
        "purpose": "validate",
        "flags": (),
    }
    base.update(overrides)
    return RunMeta(**base)


def check(verdict: RiskVerdict, code: str) -> RiskCheck:
    found = [c for c in verdict.checks if c.code == code]
    assert len(found) == 1, f"{code} appears {len(found)} times"
    return found[0]


def approve(
    intent: OrderIntent,
    *,
    pf: PortfolioState | None = None,
    view: FakeView | None = None,
    now: datetime | None = None,
    cfg: Config | None = None,
    eng: DefaultRiskEngine | None = None,
    **kwargs: Any,
) -> tuple[RiskVerdict, ApprovedOrder | None]:
    used = eng if eng is not None else engine(cfg)
    used_view = view if view is not None else NARROW_VIEW
    return used.approve(
        intent,
        pf if pf is not None else portfolio(),
        used_view,
        now=now if now is not None else used_view.as_of,
        **kwargs,
    )


# ======================================================================================================================
# the shape of a verdict
# ======================================================================================================================


def test_every_check_of_9_1_is_recorded_once_in_the_fixed_order() -> None:
    from jevbot.vocab import RISK_CHECKS

    verdict, order = approve(intent_of(NARROW, qty=2))
    assert order is not None and verdict.approved and verdict.qty_approved == 2
    assert [c.code for c in verdict.checks] == [code for group in RISK_CHECKS for code in group]
    assert order.client_order_id.endswith("-00") and order.attempt == 0 and order.limit == -250
    assert order.approved_at == NARROW_VIEW.as_of and order.verdict_id == verdict.verdict_id
    # max_loss at THIS attempt's limit: (500 - 250) * 100 + 18 = 25,018 per contract
    assert verdict.max_loss == 25_018 * 2 and verdict.bp_required == 25_000 * 2


def test_approve_never_reads_a_clock_and_the_verdict_id_binds_the_attempt_and_the_limit() -> None:
    intent = intent_of(NARROW)
    eng = engine()
    first, _ = approve(intent, eng=eng)
    same, _ = approve(intent, eng=eng)
    other_attempt, _ = approve(intent, eng=eng, attempt=1)
    other_limit, _ = approve(intent, eng=eng, limit=-240)
    assert first.verdict_id == same.verdict_id
    assert len({first.verdict_id, other_attempt.verdict_id, other_limit.verdict_id}) == 3
    # `now` is the ONLY time input: a different `now` on the same view changes nothing but approved_at
    _, order = approve(intent, eng=eng, now=NARROW_VIEW.as_of - timedelta(hours=3))
    assert order is not None and order.approved_at == NARROW_VIEW.as_of - timedelta(hours=3)
    with pytest.raises(InvariantError, match="tz-aware"):
        approve(intent, eng=eng, now=datetime(2024, 5, 17, 20))  # noqa: DTZ001 - naive IS the case under test


# ======================================================================================================================
# check 1 / 2 - kill state, halts, diagnostic runs (the applies-to column, INV-21)
# ======================================================================================================================


@pytest.mark.parametrize("state", [KillState.TRIPPED, KillState.FLATTENING, KillState.NOT_FLAT, KillState.LOCKED])
def test_check_1_an_open_is_refused_in_every_non_armed_kill_state(state: KillState) -> None:
    verdict, order = approve(intent_of(NARROW), pf=portfolio(kill_state=state))
    assert order is None and "risk:kill_active" in verdict.reject_codes
    assert not check(verdict, "kill_active").passed


def test_check_1_a_close_is_approved_under_a_kill_a_halt_and_a_stale_book() -> None:
    """INV-21: no entry-side condition may block a risk-reducing close."""
    pf = portfolio(
        positions=(position_of(NARROW),),
        kill_state=KillState.NOT_FLAT,
        halt_entries=True,
        halt_reasons=("daily_loss_halt",),
        stale_sessions=5,
        jev_fail_sessions=9,
    )
    closing = intent_of(NARROW, purpose=OrderPurpose.CLOSE, limit_start=240, limit_natural=246)
    verdict, order = approve(closing, pf=pf)
    assert order is not None and verdict.approved
    assert check(verdict, "kill_active").detail.startswith("n/a") and check(verdict, "halt_entries").detail.startswith("n/a")
    for code in ("structure_not_allowed", "dte_window", "event_blackout", "dup_underlying_direction", "max_new_per_day"):
        assert check(verdict, code).detail.startswith("n/a"), code
    # the order-rate cap of check 21 DOES apply to a discretionary close, and only a mandatory one is exempt (9.1)
    busy = msgspec.structs.replace(pf, orders_last_minute=99)
    verdict, order = approve(closing, pf=busy)
    assert order is None and "risk:order_rate" in verdict.reject_codes
    verdict, order = approve(msgspec.structs.replace(closing, mandatory=True), pf=busy)
    assert order is not None and check(verdict, "order_rate").detail.startswith("n/a")


def test_check_1_a_kill_order_is_refused_only_when_the_account_is_locked() -> None:
    pf_tripped = portfolio(positions=(position_of(NARROW),), kill_state=KillState.TRIPPED, halt_entries=True)
    intent = intent_of(NARROW, purpose=OrderPurpose.KILL, mandatory=True, limit_start=250, limit_natural=246)
    eng = engine()
    eng.set_kill_holdings({NARROW.legs[0].contract.occ: 1, NARROW.legs[1].contract.occ: -1})
    verdict, order = approve(intent, pf=pf_tripped, eng=eng)
    assert order is not None and check(verdict, "kill_active").passed
    locked, order = approve(intent, pf=portfolio(positions=(position_of(NARROW),), kill_state=KillState.LOCKED), eng=eng)
    assert order is None and "risk:kill_active" in locked.reject_codes


def test_check_1_and_2_an_entry_halt_and_a_diagnostic_run_stop_an_open_but_not_a_close() -> None:
    halted, order = approve(intent_of(NARROW), pf=portfolio(halt_entries=True, halt_reasons=("daily_loss_halt",)))
    assert order is None and "risk:halt_entries" in halted.reject_codes
    assert check(halted, "halt_entries").detail == "daily_loss_halt"

    diagnostic = engine(flags=("diagnostic",))
    pf = portfolio(positions=(position_of(NARROW),))
    for purpose, limits in ((OrderPurpose.OPEN, (-250, -246)), (OrderPurpose.CLOSE, (240, 246))):
        verdict, order = approve(intent_of(NARROW, purpose=purpose, limit_start=limits[0], limit_natural=limits[1]), pf=pf, eng=diagnostic)
        assert order is None and "risk:diagnostic_run" in verdict.reject_codes
        assert check(verdict, "diagnostic_run").detail == "diagnostic"


# ======================================================================================================================
# check 3 / 4 - universe and structure templates
# ======================================================================================================================


def test_check_3_refuses_an_underlying_outside_the_whitelist_and_a_leg_of_another_root() -> None:
    cfg = msgspec.structs.replace(Config(), universe=msgspec.structs.replace(Config().universe, underlyings=("QQQ",)))
    verdict, order = approve(intent_of(NARROW), cfg=cfg)
    assert order is None and not check(verdict, "underlying_not_allowed").passed

    foreign = OptionContract(underlying="AAPL", expiry=EXPIRY, right=Right.PUT, strike_milli=431_000)
    legs = (
        OrderLeg(contract=foreign, side=Side.BUY, position_intent=PositionIntent.BTO),
        OrderLeg(contract=NARROW.legs[1].contract, side=Side.SELL, position_intent=PositionIntent.STO),
    )
    verdict, order = approve(intent_of(NARROW, legs=legs))
    assert order is None and not check(verdict, "underlying_not_allowed").passed


def test_check_4_refuses_a_disabled_kind_and_legs_that_are_not_the_kinds_template() -> None:
    cfg = msgspec.structs.replace(Config(), structures=msgspec.structs.replace(Config().structures, enabled=(StructureKind.LONG_CALL,)))
    verdict, order = approve(intent_of(NARROW), cfg=cfg)
    assert order is None and not check(verdict, "structure_not_allowed").passed

    mislabelled = msgspec.structs.replace(NARROW, underlying="QQQ")  # SPY legs under a QQQ structure
    verdict, order = approve(intent_of(mislabelled))
    assert order is None and not check(verdict, "structure_not_allowed").passed
    assert check(verdict, "underlying_not_allowed").passed  # the LEGS still match the intent's underlying (check 3)

    # every other template violation - a wrong leg count, a second expiry, a ratio other than 1 - is also a defined-risk
    # violation, and check 5 turns those into an InvariantError rather than a soft reject (9.1)
    second_expiry = contract_at(CHAIN, CHAIN.expiries()[0], Right.PUT, 431_000)
    calendarised = msgspec.structs.replace(NARROW, legs=(Leg(contract=second_expiry, side=Side.BUY), NARROW.legs[1]))
    with pytest.raises(InvariantError, match="not defined risk"):
        approve(intent_of(calendarised))
    with pytest.raises(InvariantError, match="not defined risk"):
        approve(intent_of(msgspec.structs.replace(NARROW, kind=StructureKind.LONG_PUT)))


def test_check_4_and_5_are_not_applicable_to_closes() -> None:
    pf = portfolio(positions=(position_of(NARROW),))
    verdict, _ = approve(intent_of(NARROW, purpose=OrderPurpose.CLOSE, limit_start=240, limit_natural=246), pf=pf)
    assert check(verdict, "structure_not_allowed").detail.startswith("n/a")
    assert check(verdict, "not_defined_risk").detail.startswith("n/a")


# ======================================================================================================================
# check 5 - defined risk (credit AND debit kinds), cross-checked against an independent payoff evaluation
# ======================================================================================================================


def payoff_max_loss(structure: Structure, net: int, fee_rt: int) -> int:
    """The expiry payoff's worst case, computed from the LEGS alone: `max_loss = (net - min value) * 100 + fee_rt`.

    `value(S) = sum over legs of (+1 buy / -1 sell) * intrinsic(S)`; a piecewise-linear payoff takes its extremes at the
    kinks, so evaluating at 0, every strike and twice the highest strike is exact. Independent of `structmath` (9.2).
    """
    strikes = sorted({leg.contract.strike_milli for leg in structure.legs})
    grid = [0, *strikes, max(strikes) * 2]
    values = []
    for spot_milli in grid:
        value = 0
        for leg in structure.legs:
            strike = leg.contract.strike_milli
            intrinsic = max(spot_milli - strike, 0) if leg.contract.right is Right.CALL else max(strike - spot_milli, 0)
            value += intrinsic * (1 if leg.side is Side.BUY else -1)
        values.append(value)
    worst_milli = min(values)
    assert worst_milli % 10 == 0, "the fixture's strikes are whole cents"
    return (net - worst_milli // 10) * MULTIPLIER + fee_rt


@pytest.mark.parametrize("kind", list(StructureKind))
def test_check_5_passes_every_shipped_kind_and_its_max_loss_equals_the_payoff_worst_case(kind: StructureKind) -> None:
    structure = make_structure(CHAIN, kind)
    natural = 0
    for leg in structure.legs:
        quote = quote_of(CHAIN, leg.contract)
        natural += quote.ask if leg.side is Side.BUY else -quote.bid
    intent = intent_of(structure, limit_start=natural, limit_natural=natural)
    verdict, _ = approve(intent, view=view_of(CHAIN))
    recorded = check(verdict, "not_defined_risk")
    assert recorded.passed and recorded.observed is not None and recorded.observed > 0
    # the same number, derived from the legs' expiry payoff instead of from the 9.2 table
    fee_rt = recorded.observed - payoff_max_loss(structure, natural, 0)
    assert 0 <= fee_rt < 100
    assert recorded.observed == payoff_max_loss(structure, natural, fee_rt)


def test_check_5_passes_both_debit_verticals_where_the_long_leg_is_closer_to_the_money() -> None:
    """9.1 check 5: demanding "further OTM" on a debit vertical would raise on 2 of the 7 D3 structures."""
    for kind in (StructureKind.CALL_DEBIT, StructureKind.PUT_DEBIT):
        structure = make_structure(CHAIN, kind)
        long = next(leg for leg in structure.legs if leg.side is Side.BUY)
        short = next(leg for leg in structure.legs if leg.side is Side.SELL)
        if kind is StructureKind.CALL_DEBIT:
            assert long.contract.strike_milli < short.contract.strike_milli
        else:
            assert long.contract.strike_milli > short.contract.strike_milli
        verdict, _ = approve(intent_of(structure, limit_start=600, limit_natural=600), view=view_of(CHAIN))
        assert check(verdict, "not_defined_risk").passed


def test_check_5_raises_an_invariant_error_on_a_mis_ordered_or_uncovered_short() -> None:
    long, short = NARROW.legs[0].contract, NARROW.legs[1].contract
    mis_ordered = msgspec.structs.replace(  # the long put sits ABOVE the short put: nothing caps the loss
        NARROW, legs=(Leg(contract=short, side=Side.BUY), Leg(contract=contract_at(CHAIN, EXPIRY, Right.PUT, 426_000), side=Side.SELL))
    )
    with pytest.raises(InvariantError, match="not defined risk"):
        approve(intent_of(mis_ordered))
    uncovered = msgspec.structs.replace(NARROW, legs=(Leg(contract=long, side=Side.SELL), Leg(contract=short, side=Side.SELL)))
    with pytest.raises(InvariantError, match="not defined risk"):
        approve(intent_of(uncovered))


def test_check_5_raises_when_the_economics_are_invalid() -> None:
    """A credit at or above the width prices to `max_loss_pc <= 0`: a CandidateGenerator bug, never a soft reject."""
    with pytest.raises(InvariantError, match="max_loss_pc"):
        approve(intent_of(NARROW, limit_start=-WIDTH - 20, limit_natural=-WIDTH - 20))


# ======================================================================================================================
# check 6 - closes and kills only reduce
# ======================================================================================================================


def test_check_6_requires_close_intents_opposite_sides_and_a_quantity_that_fits() -> None:
    pf = portfolio(positions=(position_of(NARROW, qty=2),))
    good = intent_of(NARROW, purpose=OrderPurpose.CLOSE, qty=2, limit_start=240, limit_natural=246)
    verdict, order = approve(good, pf=pf)
    assert order is not None and check(verdict, "close_only_reduces").passed

    too_big = msgspec.structs.replace(good, qty=3)
    verdict, order = approve(too_big, pf=pf)
    assert order is None and not check(verdict, "close_only_reduces").passed

    opening_legs = open_legs_of(NARROW)  # *_to_open on a CLOSE order
    verdict, _ = approve(msgspec.structs.replace(good, legs=opening_legs), pf=pf)
    assert not check(verdict, "close_only_reduces").passed

    verdict, _ = approve(good, pf=portfolio())  # nothing held
    assert not check(verdict, "close_only_reduces").passed


def test_check_6_measures_a_kill_against_the_brokers_holdings_not_the_books() -> None:
    """9.1: the flatten closes what the BROKER reports; `set_kill_holdings` is how the pure engine learns it."""
    intent = intent_of(NARROW, purpose=OrderPurpose.KILL, qty=3, mandatory=True, limit_start=250, limit_natural=246)
    eng = engine()
    book_only = portfolio(positions=(position_of(NARROW, qty=1),))
    verdict, order = approve(intent, pf=book_only, eng=eng)
    assert order is None and not check(verdict, "close_only_reduces").passed  # the book holds 1, the order closes 3
    eng.set_kill_holdings({NARROW.legs[0].contract.occ: 3, NARROW.legs[1].contract.occ: -3})
    verdict, order = approve(intent, pf=book_only, eng=eng)
    assert order is not None and check(verdict, "close_only_reduces").passed
    eng.set_kill_holdings(None)
    verdict, order = approve(intent, pf=book_only, eng=eng)
    assert order is None


def test_the_equity_flatten_passes_checks_3_and_6_and_is_sent_as_a_market_order() -> None:
    """9.5 K4 / 9.1: `equity_symbol` is whitelisted, `equity_side` is opposite the share position, `equity_qty` fits."""
    eng = engine()
    eng.set_kill_holdings({"SPY": 100})
    intent = intent_of(None, purpose=OrderPurpose.KILL, mandatory=True, equity=("SPY", Side.SELL, 100), limit_start=0, limit_natural=0)
    verdict, order = approve(intent, eng=eng, market=True)
    assert order is not None and order.limit is None and order.qty == 0
    assert check(verdict, "underlying_not_allowed").passed and check(verdict, "close_only_reduces").passed
    assert check(verdict, "past_order_cutoff").detail.startswith("n/a")

    wrong_side = msgspec.structs.replace(intent, equity_side=Side.BUY)
    verdict, order = approve(wrong_side, eng=eng, market=True)
    assert order is None and not check(verdict, "close_only_reduces").passed

    too_many = msgspec.structs.replace(intent, equity_qty=101)
    verdict, order = approve(too_many, eng=eng, market=True)
    assert order is None and not check(verdict, "close_only_reduces").passed

    eng.set_kill_holdings({"NVDA": -100})
    foreign = msgspec.structs.replace(intent, underlying="NVDA", equity_symbol="NVDA", equity_side=Side.BUY)
    verdict, order = approve(foreign, eng=eng, market=True)
    assert order is None and not check(verdict, "underlying_not_allowed").passed

    eng.set_kill_holdings({"SPY": -100})  # covering a SHORT share position buys
    covering = msgspec.structs.replace(intent, equity_side=Side.BUY)
    _, order = approve(covering, eng=eng, market=True)
    assert order is not None


# ======================================================================================================================
# check 7 - the order cutoff (WP05's named acceptance case)
# ======================================================================================================================


@pytest.mark.parametrize("session", [SESSION, date(2024, 7, 3)])
@pytest.mark.parametrize("fill_rule", [FillRule.NEXT_SNAPSHOT, FillRule.SAME_SNAPSHOT_WORST])
def test_check_7_is_not_applicable_on_an_eod_decision_snapshot_at_the_close(session: date, fill_rule: FillRule) -> None:
    """An end-of-day simulation has no intraday cutoff to enforce: `as_of` IS the close (9.1 check 7)."""
    if session != SESSION:
        assert CAL.is_early_close(session), "the second case must be an early close"
    chain, structure = narrow_put_credit(make_chain(session=session))
    view = view_of(chain, session=session)
    assert view.as_of == CAL.open_close(session)[1]
    cfg = msgspec.structs.replace(Config(), cadence=msgspec.structs.replace(Config().cadence, fill_rule=fill_rule))
    key = SnapshotKey(session=session, slot=Slot.EOD)
    for purpose, limits in ((OrderPurpose.OPEN, (-250, -246)), (OrderPurpose.CLOSE, (240, 246))):
        held = () if purpose is OrderPurpose.OPEN else (position_of(structure),)
        pf = portfolio(key=key, positions=held)
        intent = intent_of(structure, purpose=purpose, session=session, limit_start=limits[0], limit_natural=limits[1])
        verdict, order = approve(intent, pf=pf, view=view, cfg=cfg, now=view.as_of)
        recorded = check(verdict, "past_order_cutoff")
        assert recorded.passed and recorded.detail == "n/a: eod decision snapshot"
        assert order is not None, verdict.reject_codes


def test_check_7_rejects_a_dec_snapshot_past_the_cutoff_and_approves_before_it() -> None:
    chain, structure = narrow_put_credit(make_chain(slot=Slot.DEC, fidelity=_recorded()))
    view = view_of(chain, slot=Slot.DEC, as_of=CAL.offset_from_close(SESSION, 25))
    dec_key = SnapshotKey(session=SESSION, slot=Slot.DEC)
    pf = portfolio(key=dec_key, positions=(position_of(structure),))
    for purpose, limits in ((OrderPurpose.OPEN, (-250, -246)), (OrderPurpose.CLOSE, (240, 246))):
        pf = portfolio(key=dec_key, positions=() if purpose is OrderPurpose.OPEN else (position_of(structure),))
        intent = intent_of(structure, purpose=purpose, limit_start=limits[0], limit_natural=limits[1])
        late, order = approve(intent, pf=pf, view=view, now=CAL.offset_from_close(SESSION, 4))
        assert order is None and "risk:past_order_cutoff" in late.reject_codes
        early, order = approve(intent, pf=pf, view=view, now=CAL.offset_from_close(SESSION, 6))
        assert order is not None and check(early, "past_order_cutoff").passed
    # outside the session entirely (after the close) the check fails too
    intent = intent_of(structure)
    after, order = approve(intent, pf=pf, view=view, now=CAL.open_close(SESSION)[1] + timedelta(minutes=30))
    assert order is None and not check(after, "past_order_cutoff").passed
    # a KILL order is exempt
    eng = engine()
    eng.set_kill_holdings({leg.contract.occ: (1 if leg.side is Side.BUY else -1) for leg in structure.legs})
    kill = intent_of(structure, purpose=OrderPurpose.KILL, mandatory=True, limit_start=250, limit_natural=246)
    verdict, order = approve(kill, pf=pf, view=view, eng=eng, now=CAL.offset_from_close(SESSION, 1))
    assert order is not None and check(verdict, "past_order_cutoff").detail.startswith("n/a")


def _recorded() -> Any:
    from jevbot.types import Fidelity

    return Fidelity.RECORDED_INDICATIVE


# ======================================================================================================================
# check 8 / 9 - clock skew and quotes
# ======================================================================================================================


def test_check_8_blocks_an_open_on_clock_skew_but_never_a_close() -> None:
    reading = ClockReading(
        broker_ts=CLOSE,
        local_ts=CLOSE,
        rtt_ms=10,
        skew_ms=6_000,
        is_open=True,
        next_open=CLOSE,
        next_close=CLOSE,
    )
    verdict, order = approve(intent_of(NARROW), clock=reading)
    assert order is None and "risk:clock_skew" in verdict.reject_codes
    pf = portfolio(positions=(position_of(NARROW),))
    verdict, order = approve(intent_of(NARROW, purpose=OrderPurpose.CLOSE, limit_start=240, limit_natural=246), pf=pf, clock=reading)
    assert order is not None and check(verdict, "clock_skew").detail.startswith("n/a")
    ok = msgspec.structs.replace(reading, skew_ms=5_000)
    _, order = approve(intent_of(NARROW), clock=ok)
    assert order is not None


def test_check_9_stops_an_open_on_a_crossed_thin_or_stale_quote() -> None:
    crossed_chain, crossed = narrow_put_credit(short_quote=(400, 399))
    verdict, order = approve(intent_of(crossed, limit_start=-250, limit_natural=-249), view=view_of(crossed_chain))
    assert order is None and not check(verdict, "crossed_quote").passed

    thin_chain, thin = narrow_put_credit(short_quote=(5, 9))
    verdict, order = approve(intent_of(thin, limit_start=-5, limit_natural=-1), view=view_of(thin_chain))
    assert order is None and not check(verdict, "stale_quote").passed
    assert "liq:bid" in check(verdict, "stale_quote").detail

    stale_view = view_of(NARROW_CHAIN, as_of=CLOSE + timedelta(seconds=600))
    verdict, order = approve(intent_of(NARROW), view=stale_view, now=stale_view.as_of)
    assert order is None and "chain_age=600" in check(verdict, "stale_quote").detail


def test_check_9_only_crossed_quotes_delay_a_discretionary_close_and_never_a_mandatory_one() -> None:
    crossed_chain, crossed = narrow_put_credit(short_quote=(400, 399))
    pf = portfolio(positions=(position_of(crossed),))
    view = view_of(crossed_chain)
    discretionary = intent_of(crossed, purpose=OrderPurpose.CLOSE, limit_start=240, limit_natural=246)
    verdict, order = approve(discretionary, pf=pf, view=view)
    assert order is None and not check(verdict, "crossed_quote").passed
    assert check(verdict, "stale_quote").detail.startswith("n/a")

    mandatory = msgspec.structs.replace(discretionary, mandatory=True, reason=ExitReason.FORCE_EXPIRY.value)
    verdict, order = approve(mandatory, pf=pf, view=view)
    assert order is not None and check(verdict, "crossed_quote").detail.startswith("n/a")


def test_check_9_a_zero_bid_long_leg_never_blocks_a_close() -> None:
    """10.4: a worthless 0.07-delta wing is the normal state of a winning short-premium trade."""
    zero_chain, zero = narrow_put_credit(long_quote=(0, 4))
    pf = portfolio(positions=(position_of(zero),))
    verdict, order = approve(
        intent_of(zero, purpose=OrderPurpose.CLOSE, limit_start=240, limit_natural=246), pf=pf, view=view_of(zero_chain)
    )
    assert order is not None and check(verdict, "crossed_quote").passed
    # on an OPEN the same leg is a liquidity reject (we never open by buying a leg with no bid at all is fine, but the
    # sold leg's floor and the spread rule still apply): the entry-side filter runs in full
    opening, _ = approve(intent_of(zero), view=view_of(zero_chain))
    assert not check(opening, "stale_quote").passed


# ======================================================================================================================
# check 10 / 11 - time rules and events
# ======================================================================================================================


def test_check_10_enforces_the_dte_window_and_the_expiry_policy_to_last_session() -> None:
    verdict, _ = approve(intent_of(NARROW))
    assert check(verdict, "dte_window").observed == 35 and check(verdict, "dte_window").passed
    assert check(verdict, "expiry_policy").observed == CAL.sessions_between(SESSION, NARROW.last_session)

    cfg = msgspec.structs.replace(Config(), dte=msgspec.structs.replace(Config().dte, min_entry=36))
    verdict, order = approve(intent_of(NARROW), cfg=cfg)
    assert order is None and not check(verdict, "dte_window").passed

    cfg = msgspec.structs.replace(Config(), dte=msgspec.structs.replace(Config().dte, min_sessions_beyond_hard_exit=40))
    verdict, order = approve(intent_of(NARROW), cfg=cfg)
    assert order is None and not check(verdict, "expiry_policy").passed


def test_check_11_blocks_a_short_premium_entry_inside_the_fomc_blackout_only() -> None:
    inside = CAL.next_session(SESSION, 1)
    outside = CAL.next_session(SESSION, 2)
    blocked, order = approve(intent_of(NARROW), view=view_of(NARROW_CHAIN, events=(fomc_event(inside),)))
    assert order is None and not check(blocked, "event_blackout").passed
    clear, order = approve(intent_of(NARROW), view=view_of(NARROW_CHAIN, events=(fomc_event(outside),)))
    assert order is not None and check(clear, "event_blackout").passed

    debit = make_structure(CHAIN, StructureKind.CALL_DEBIT)  # long premium: no blackout
    verdict, _ = approve(intent_of(debit, limit_start=581, limit_natural=581), view=view_of(CHAIN, events=(fomc_event(inside),)))
    assert check(verdict, "event_blackout").detail == "not short premium"


def test_check_11_blocks_a_short_call_across_a_verified_ex_dividend_date() -> None:
    condor = make_structure(CHAIN, StructureKind.IRON_CONDOR)
    short_call = next(leg for leg in condor.legs if leg.side is Side.SELL and leg.contract.right is Right.CALL)
    # spot 45,000c; the short call sits at 472.00: an ex-date whose dividend lifts spot + div above the strike blocks it
    dividend = ex_dividend_event("SPY", CAL.next_session(SESSION, 3), amount_cents=2_300)
    blocked, _ = approve(intent_of(condor, limit_start=-245, limit_natural=-245), view=view_of(CHAIN, events=(dividend,)))
    assert not check(blocked, "exdiv_short_call").passed
    assert short_call.contract.occ in check(blocked, "exdiv_short_call").detail

    small = ex_dividend_event("SPY", CAL.next_session(SESSION, 3), amount_cents=50)
    clear, _ = approve(intent_of(condor, limit_start=-245, limit_natural=-245), view=view_of(CHAIN, events=(small,)))
    assert check(clear, "exdiv_short_call").passed
    # no short call at all
    verdict, _ = approve(intent_of(NARROW), view=view_of(NARROW_CHAIN, events=(dividend,)))
    assert check(verdict, "exdiv_short_call").detail == "no short call"


# ======================================================================================================================
# check 12 / 13 / 14 - exposure
# ======================================================================================================================


def test_check_12_one_position_per_underlying_and_direction_a_cooldown_and_the_direction_cap() -> None:
    held = portfolio(positions=(position_of(NARROW),))
    verdict, order = approve(intent_of(NARROW), pf=held)
    assert order is None and not check(verdict, "dup_underlying_direction").passed

    working = portfolio(working=(intent_of(NARROW),))
    verdict, order = approve(intent_of(NARROW), pf=working)
    assert order is None and not check(verdict, "dup_underlying_direction").passed

    _, approved_open = approve(intent_of(NARROW))
    assert approved_open is not None
    verdict, order = approve(intent_of(NARROW), approved_so_far=(approved_open,))
    assert order is None and not check(verdict, "dup_underlying_direction").passed

    cooling = portfolio(cooldowns=(("SPY", "bullish", CAL.next_session(SESSION, 2)),))
    verdict, order = approve(intent_of(NARROW), pf=cooling)
    assert order is None and not check(verdict, "reentry_cooldown").passed
    expired = portfolio(cooldowns=(("SPY", "bullish", SESSION),))
    _, order = approve(intent_of(NARROW), pf=expired)
    assert order is not None  # "first session entries are allowed again" is inclusive

    # three bullish structures across the universe already: the cap bites before the dup rule can be evaded
    others = tuple(
        position_of(make_structure(make_chain(u, spot=20_000), StructureKind.PUT_CREDIT), position_id=f"other{i}")
        for i, u in enumerate(("QQQ", "IWM", "QQQ"))
    )
    verdict, order = approve(intent_of(NARROW), pf=portfolio(positions=others))
    assert order is None and not check(verdict, "same_direction_cap").passed
    assert check(verdict, "same_direction_cap").observed == 3


def test_checks_13_and_14_count_open_working_and_already_approved_structures() -> None:
    six = tuple(
        position_of(make_structure(make_chain(u, spot=20_000), kind), position_id=f"p{i}")
        for i, (u, kind) in enumerate(
            (
                ("QQQ", StructureKind.PUT_CREDIT),
                ("IWM", StructureKind.CALL_CREDIT),
                ("QQQ", StructureKind.CALL_CREDIT),
                ("IWM", StructureKind.PUT_DEBIT),
                ("QQQ", StructureKind.PUT_DEBIT),
                ("IWM", StructureKind.LONG_CALL),
            )
        )
    )
    verdict, order = approve(intent_of(NARROW), pf=portfolio(positions=six))
    assert order is None and not check(verdict, "max_open_structures").passed
    assert check(verdict, "max_open_structures").observed == 6 and check(verdict, "max_open_structures").limit == 6

    verdict, order = approve(intent_of(NARROW), pf=portfolio(opened_today=2))
    assert order is None and not check(verdict, "max_new_per_day").passed
    verdict, order = approve(intent_of(NARROW), pf=portfolio(opened_today=1))
    assert order is not None and check(verdict, "max_new_per_day").observed == 1


# ======================================================================================================================
# checks 15-18 - sizing (9.3)
# ======================================================================================================================


def test_check_15_reduces_the_quantity_to_the_per_trade_budget_at_1_0x() -> None:
    # max_loss_pc at the limit -250 = (500 - 250) * 100 + 18 = 25,018c; budget = floor(1% of $100,000) = 100,000c
    verdict, order = approve(intent_of(NARROW, qty=9))
    assert order is not None and order.qty == 3  # 100,000 // 25,018 = 3
    assert check(verdict, "max_loss_per_trade").observed == 3 * 25_018
    assert check(verdict, "max_loss_per_trade").limit == 100_000
    assert verdict.qty_approved == 3 and verdict.max_loss == 75_054

    poor = portfolio(equity=BandPrices(orats=2_000_000, worst=2_000_000, mid=2_000_000))
    verdict, order = approve(intent_of(NARROW, qty=9), pf=poor)  # budget 20,000c < 25,018c
    assert order is None and "risk:max_loss_per_trade" in verdict.reject_codes and "risk:size_zero" in verdict.reject_codes


def test_check_16_leaves_room_for_the_open_working_and_already_approved_loss() -> None:
    held = tuple(position_of(OTHER, max_loss=490_000, position_id=f"p{i}") for i in range(2))
    pf = portfolio(positions=held)
    verdict, order = approve(intent_of(NARROW, qty=5), pf=pf)
    # aggregate cap 10% of $100,000 = 1,000,000c; 980,000 held leaves 20,000 < 25,018
    assert order is None and not check(verdict, "agg_max_loss").passed
    smaller = (position_of(OTHER, max_loss=400_000, position_id="p0"),)
    verdict, order = approve(intent_of(NARROW, qty=5), pf=portfolio(positions=smaller))
    assert order is not None and order.qty == 3 and check(verdict, "agg_max_loss").limit == 1_000_000


def test_check_17_fits_the_quantity_into_the_internal_and_broker_buying_power() -> None:
    from jevbot.types import RunMode

    # bp_required_pc at -250 = (500 - 250) * 100 = 25,000c; internal cap = 50% of $100,000 = 5,000,000c
    verdict, order = approve(intent_of(NARROW, qty=3))
    assert order is not None and check(verdict, "buying_power").detail == "bp_pc=25000"

    tight = portfolio(positions=(position_of(OTHER, bp_reserved=4_960_000, position_id="p0"),), broker_options_bp=None)
    verdict, order = approve(intent_of(NARROW, qty=3), pf=tight)
    assert order is not None and order.qty == 1  # 40,000 // 25,000 = 1

    paper = DefaultRiskEngine(Config(), _meta(mode=RunMode.PAPER))
    broker_tight = portfolio(broker_options_bp=25_000, broker_equity=INITIAL_EQUITY)
    _, order = approve(intent_of(NARROW, qty=3), pf=broker_tight, eng=paper)
    assert order is not None and order.qty == 1


def test_check_18_caps_the_contract_count_and_the_order_notional() -> None:
    cfg = msgspec.structs.replace(Config(), risk=msgspec.structs.replace(Config().risk, max_contracts_per_trade=2))
    _, order = approve(intent_of(NARROW, qty=9), cfg=cfg)
    assert order is not None and order.qty == 2

    cfg = msgspec.structs.replace(Config(), risk=msgspec.structs.replace(Config().risk, max_order_notional_usd=500))
    verdict, order = approve(intent_of(NARROW, qty=9), cfg=cfg)
    # per contract notional = 250c * 100 = 25,000c; cap 500 USD = 50,000c -> 2 contracts
    assert order is not None and order.qty == 2 and check(verdict, "notional_cap").limit == 50_000


def test_the_paper_equity_basis_is_the_worse_of_the_broker_and_the_book() -> None:
    from jevbot.types import RunMode

    paper = DefaultRiskEngine(Config(), _meta(mode=RunMode.PAPER))
    pf = portfolio(broker_equity=2_000_000)
    assert paper.equity_basis(pf) == 2_000_000
    assert engine().equity_basis(pf) == INITIAL_EQUITY  # a backtest reads the headline book equity only
    verdict, order = approve(intent_of(NARROW, qty=9), pf=pf, eng=paper)
    assert order is None and not check(verdict, "max_loss_per_trade").passed


def test_budget_floor_and_size_entry_follow_the_worked_example_of_9_3() -> None:
    eng = engine()
    pf = portfolio()
    assert eng.lowest_tier() == 1 / 2
    assert eng.budget_floor(pf) == 50_000  # floor(1% * $100,000 * 0.5) = $500
    candidate = _candidate(max_loss_per_contract=47_520)
    assert eng.size_entry(candidate, 1_000_000, pf, ()) == 2  # $1,000 budget
    assert eng.size_entry(candidate, 750_000, pf, ()) == 1
    assert eng.size_entry(candidate, 500_000, pf, ()) == 1
    assert eng.size_entry(candidate, 0, pf, ()) == 0
    assert eng.size_entry(_candidate(max_loss_per_contract=150_000), 1_000_000, pf, ()) == 0
    assert eng.size_entry(_candidate(max_loss_per_contract=1_000), 1_000_000, pf, ()) == 10  # max_contracts_per_trade


def _candidate(*, max_loss_per_contract: int) -> Candidate:
    return Candidate(
        structure=NARROW,
        key=KEY,
        dte=35,
        sessions_to_expiry=24,
        quotes=tuple(quote_of(NARROW_CHAIN, leg.contract) for leg in NARROW.legs),
        net=BandPrices(orats=-250, worst=-246, mid=-252),
        budget_floor=50_000,
        max_loss_per_contract=max_loss_per_contract,
        max_profit_per_contract=25_000,
        bp_required_per_contract=25_000,
        breakevens=(43_354,),
        short_distance_em=1.1,
        net_delta=0.13,
        net_vega=-0.4,
    )


def test_the_candidate_recovers_fee_rt_exactly_for_the_sizing_formulas() -> None:
    """9.2: `fee_rt` is `max_loss_per_contract - max_loss_pc(..., fee_rt = 0)` of the candidate that was sized."""
    candidate = _candidate(max_loss_per_contract=(WIDTH - 246) * 100 + 37)  # a fee_rt of 37c, whatever the quotes say
    verdict, _ = approve(intent_of(NARROW, qty=1), cand=candidate)
    assert check(verdict, "max_loss_per_trade").detail == f"max_loss_pc={(WIDTH - 250) * 100 + 37}"


# ======================================================================================================================
# check 19 / 20 / 21 - price, drift, rate
# ======================================================================================================================


def test_check_19_rejects_an_off_tick_limit_and_a_price_beyond_natural() -> None:
    cfg = msgspec.structs.replace(Config(), universe=msgspec.structs.replace(Config().universe, penny_all=()))
    verdict, order = approve(intent_of(NARROW, limit_start=-247), cfg=cfg)  # SPY on the generic 5c grid
    assert order is None and not check(verdict, "price_increment").passed
    assert check(verdict, "price_increment").limit == 5

    verdict, order = approve(intent_of(NARROW, limit_start=-200, limit_natural=-246))
    assert order is None and not check(verdict, "beyond_natural").passed
    assert check(verdict, "beyond_natural").observed == 46

    pf = portfolio(positions=(position_of(NARROW),))
    closing = intent_of(NARROW, purpose=OrderPurpose.CLOSE, limit_start=260, limit_natural=246)
    verdict, order = approve(closing, pf=pf)
    assert order is None and not check(verdict, "beyond_natural").passed
    # a mandatory close may go past natural by the cushion: ceil(0.15 * 500) = 75c
    mandatory = msgspec.structs.replace(closing, mandatory=True, limit_start=246 + 76)
    verdict, order = approve(mandatory, pf=pf)
    assert order is None and check(verdict, "beyond_natural").limit == 75
    verdict, order = approve(msgspec.structs.replace(mandatory, limit_start=246 + 75), pf=pf)
    assert order is not None and check(verdict, "beyond_natural").observed == 75


def test_check_19_raises_on_the_condor_sign_trap() -> None:
    condor = make_structure(CHAIN, StructureKind.IRON_CONDOR)
    with pytest.raises(InvariantError, match="credit structure"):
        approve(intent_of(condor, limit_start=245, limit_natural=245), view=view_of(CHAIN))


def test_check_20_stops_an_entry_whose_natural_drifted_against_us() -> None:
    verdict, _ = approve(intent_of(NARROW))
    assert check(verdict, "adverse_drift").observed == 0 and check(verdict, "adverse_drift").limit == 62  # ceil(0.25 * 246)

    drifted, structure = narrow_put_credit(short_quote=(280, 284))  # the credit collapsed: natural is now 154 - 280 = -126
    verdict, order = approve(intent_of(structure, limit_start=-250, limit_natural=-246), view=view_of(drifted))
    assert order is None and not check(verdict, "adverse_drift").passed
    assert check(verdict, "adverse_drift").observed == 120


def test_check_21_throttles_entries_and_discretionary_closes_only() -> None:
    busy = portfolio(orders_last_minute=10, positions=(position_of(NARROW),))
    verdict, order = approve(intent_of(NARROW), pf=busy)
    assert order is None and not check(verdict, "order_rate").passed

    discretionary = intent_of(NARROW, purpose=OrderPurpose.CLOSE, limit_start=240, limit_natural=246)
    verdict, order = approve(discretionary, pf=busy)
    assert order is None and not check(verdict, "order_rate").passed

    mandatory = msgspec.structs.replace(discretionary, mandatory=True)
    verdict, order = approve(mandatory, pf=busy)
    assert order is not None and check(verdict, "order_rate").detail.startswith("n/a")

    verdict, order = approve(intent_of(NARROW), attempt=6)
    assert order is None and not check(verdict, "attempt_cap").passed
    verdict, order = approve(mandatory, pf=busy, attempt=99)
    assert order is not None and check(verdict, "attempt_cap").detail.startswith("n/a")


# ======================================================================================================================
# 9.3 recheck_fill
# ======================================================================================================================


def test_recheck_fill_resizes_at_the_actual_fill_price_at_1_0x() -> None:
    eng = engine()
    intent = intent_of(NARROW, qty=3)
    pf = portfolio()
    qty, code = eng.recheck_fill(intent, BandPrices(orats=-250, worst=-240, mid=-255), pf, NARROW_VIEW)
    assert (qty, code) == (3, None)  # (500 - 240) * 100 + 18 = 26,018; 100,000 // 26,018 = 3
    qty, code = eng.recheck_fill(intent, BandPrices(orats=-60, worst=-50, mid=-65), pf, NARROW_VIEW)
    assert (qty, code) == (2, None)  # (500 - 50) * 100 + 18 = 45,018; 100,000 // 45,018 = 2
    qty, code = eng.recheck_fill(intent, BandPrices(orats=-10, worst=0, mid=-15), pf, NARROW_VIEW)
    assert (qty, code) == (1, None)  # 50,018 -> 1
    poor = portfolio(equity=BandPrices(orats=2_000_000, worst=2_000_000, mid=2_000_000))
    qty, code = eng.recheck_fill(intent, BandPrices(orats=-250, worst=-240, mid=-255), poor, NARROW_VIEW)
    assert (qty, code) == (0, "max_loss_per_trade")
    crowded = portfolio(positions=(position_of(NARROW, max_loss=990_000),))
    qty, code = eng.recheck_fill(intent, BandPrices(orats=-250, worst=-240, mid=-255), crowded, NARROW_VIEW)
    assert (qty, code) == (0, "agg_max_loss")
    bp_bound = portfolio(broker_options_bp=0)
    qty, code = eng.recheck_fill(intent, BandPrices(orats=-250, worst=-240, mid=-255), bp_bound, NARROW_VIEW)
    assert (qty, code) == (0, "buying_power")


def test_recheck_fill_re_evaluates_the_blackout_at_the_fill_snapshot() -> None:
    """9.3 / WP05 + WP09: an FOMC two sessions after the decision passes at D and fails at the D+1 fill."""
    fomc = CAL.next_session(SESSION, 2)
    assert CAL.sessions_between(SESSION, fomc) == 2
    decision_view = view_of(NARROW_CHAIN, events=(fomc_event(fomc),))
    eng = engine()
    intent = intent_of(NARROW, qty=1)
    _, order = approve(intent, view=decision_view, eng=eng)
    assert order is not None  # approved at D: the blackout is one session wide

    fill_session = CAL.next_session(SESSION, 1)
    fill_chain, fill_structure = narrow_put_credit(make_chain(session=fill_session))
    fill_view = view_of(fill_chain, session=fill_session, events=(fomc_event(fomc),))
    qty, code = eng.recheck_fill(intent, BandPrices(orats=-250, worst=-240, mid=-255), portfolio(), fill_view)
    assert (qty, code) == (0, "event_blackout")
    assert f"risk:recheck_failed:{code}" == "risk:recheck_failed:event_blackout"
    del fill_structure


def test_recheck_fill_re_evaluates_the_dte_window_and_the_ex_dividend_block() -> None:
    eng = engine()
    intent = intent_of(NARROW, qty=1)
    cfg_tight = msgspec.structs.replace(Config(), dte=msgspec.structs.replace(Config().dte, max_entry=34))
    tight = DefaultRiskEngine(cfg_tight)
    qty, code = tight.recheck_fill(intent, BandPrices(orats=-250, worst=-240, mid=-255), portfolio(), NARROW_VIEW)
    assert (qty, code) == (0, "dte_window")

    cfg_policy = msgspec.structs.replace(Config(), dte=msgspec.structs.replace(Config().dte, min_sessions_beyond_hard_exit=40))
    policy = DefaultRiskEngine(cfg_policy)
    qty, code = policy.recheck_fill(intent, BandPrices(orats=-250, worst=-240, mid=-255), portfolio(), NARROW_VIEW)
    assert (qty, code) == (0, "expiry_policy")

    condor = make_structure(CHAIN, StructureKind.IRON_CONDOR)
    exdiv_view = view_of(CHAIN, events=(ex_dividend_event("SPY", CAL.next_session(SESSION, 3), amount_cents=2_300),))
    qty, code = eng.recheck_fill(
        intent_of(condor, qty=1, limit_start=-245, limit_natural=-245),
        BandPrices(orats=-245, worst=-240, mid=-250),
        portfolio(),
        exdiv_view,
    )
    assert (qty, code) == (0, "exdiv_short_call")


def test_recheck_fill_passes_a_close_through_untouched() -> None:
    qty, code = engine().recheck_fill(
        intent_of(NARROW, purpose=OrderPurpose.CLOSE, qty=4, limit_start=240, limit_natural=246),
        BandPrices(orats=250, worst=260, mid=245),
        portfolio(),
        NARROW_VIEW,
    )
    assert (qty, code) == (4, None)


# ======================================================================================================================
# 9.4 hard exits
# ======================================================================================================================


def structure_expiring(expiry: date) -> Structure:
    last = CAL.prev_or_same_session(expiry)
    short = OptionContract(underlying="SPY", expiry=expiry, right=Right.PUT, strike_milli=436_000)
    long = OptionContract(underlying="SPY", expiry=expiry, right=Right.PUT, strike_milli=431_000)
    return Structure(
        kind=StructureKind.PUT_CREDIT,
        underlying="SPY",
        expiry=expiry,
        last_session=last,
        legs=(Leg(contract=long, side=Side.BUY), Leg(contract=short, side=Side.SELL)),
    )


@pytest.mark.parametrize(
    ("expiry", "last_session"),
    [
        (date(2014, 4, 19), date(2014, 4, 17)),  # a Saturday-dated monthly in a Good-Friday week: L is the THURSDAY
        (date(2014, 6, 21), date(2014, 6, 20)),  # a Saturday-dated monthly: L is the FRIDAY
    ],
)
def test_hard_exit_forces_the_expiry_close_counted_to_last_session(expiry: date, last_session: date) -> None:
    assert CAL.prev_or_same_session(expiry) == last_session
    structure = structure_expiring(expiry)
    eng = engine()
    position = position_of(structure, liq_value=100)
    at_four = CAL.prev_session(last_session, 4)
    assert CAL.sessions_between(at_four, last_session) == 4
    # four sessions out the expiry rule has not fired yet; the calendar-day time exit already has (dte <= 7)
    assert eng.hard_exit(position, view_of(None, session=at_four)) is ExitReason.TIME_EXIT
    for distance in (3, 2):
        session = CAL.prev_session(last_session, distance)
        assert CAL.sessions_between(session, last_session) == distance
        assert eng.hard_exit(position, view_of(None, session=session)) is ExitReason.FORCE_EXPIRY


def test_hard_exit_time_exit_uses_calendar_days_to_last_session_per_premium_side() -> None:
    eng = engine()
    far = structure_expiring(date(2024, 6, 21))
    quiet = position_of(far, liq_value=150)  # pnl = (200 - 150) * 100 = 5,000: neither the stop nor the target
    session = date(2024, 6, 13)  # 8 calendar days to 06-21
    assert eng.hard_exit(quiet, view_of(None, session=session)) is None
    session = date(2024, 6, 14)  # 7 days: the short-premium time exit fires
    assert eng.hard_exit(quiet, view_of(None, session=session)) is ExitReason.TIME_EXIT
    cfg = msgspec.structs.replace(Config(), dte=msgspec.structs.replace(Config().dte, time_exit_short_premium=3))
    assert DefaultRiskEngine(cfg).hard_exit(quiet, view_of(None, session=session)) is None


def test_hard_exit_stop_loss_and_profit_target_use_the_headline_conservative_mark() -> None:
    eng = engine()
    structure = structure_expiring(date(2024, 6, 21))
    view = view_of(None)
    # pnl = (-open_net.orats - liq_value) * 100 * qty
    losing = position_of(structure, open_net=BandPrices(orats=-200, worst=-190, mid=-205), liq_value=400, max_loss=30_000)
    assert eng.hard_exit(losing, view) is ExitReason.STOP_LOSS  # -pnl = 20,000 >= 0.5 * 30,000
    just_short = msgspec.structs.replace(losing, liq_value=349)
    assert eng.hard_exit(just_short, view) is None  # -pnl = 14,900 < 15,000

    winning = position_of(structure, open_net=BandPrices(orats=-200, worst=-190, mid=-205), liq_value=100, max_profit=20_000)
    assert eng.hard_exit(winning, view) is ExitReason.PROFIT_TARGET  # pnl = 10,000 >= 0.5 * 20,000
    assert eng.hard_exit(msgspec.structs.replace(winning, liq_value=101), view) is None

    long_option = position_of(
        msgspec.structs.replace(structure, kind=StructureKind.LONG_PUT, legs=(structure.legs[0],)),
        open_net=BandPrices(orats=500, worst=510, mid=495),
        liq_value=-800,
        max_profit=None,
        max_loss=50_000,
    )
    assert eng.hard_exit(long_option, view) is ExitReason.PROFIT_TARGET  # pnl 30,000 >= 0.5 * debit paid 50,000
    assert eng.hard_exit(msgspec.structs.replace(long_option, liq_value=-749), view) is None


def test_hard_exit_ex_dividend_fires_two_sessions_out_and_not_three() -> None:
    condor = make_structure(CHAIN, StructureKind.IRON_CONDOR)
    position = position_of(condor, open_net=BandPrices(orats=-245, worst=-240, mid=-250), liq_value=100, max_profit=24_500)
    eng = engine()
    # the short call sits at 472.00 with spot 450.00: a $23 dividend puts it "ITM or near ITM across the ex-date"
    two = ex_dividend_event("SPY", CAL.next_session(SESSION, 2), amount_cents=2_300)
    three = ex_dividend_event("SPY", CAL.next_session(SESSION, 3), amount_cents=2_300)
    assert eng.hard_exit(position, view_of(CHAIN, events=(three,))) is not ExitReason.EX_DIVIDEND
    itm_call = contract_at(CHAIN, EXPIRY, Right.CALL, 440_000)  # spot 450: ITM by $10
    itm_chain = set_quote(CHAIN, itm_call, bid=1_040, ask=1_045)
    itm = Structure(
        kind=StructureKind.CALL_CREDIT,
        underlying="SPY",
        expiry=EXPIRY,
        last_session=CHAIN.last_session(EXPIRY),
        legs=(Leg(contract=itm_call, side=Side.SELL), Leg(contract=contract_at(CHAIN, EXPIRY, Right.CALL, 450_000), side=Side.BUY)),
    )
    held = position_of(itm, open_net=BandPrices(orats=-400, worst=-390, mid=-405), liq_value=100, max_profit=40_000)
    # extrinsic = ask 1,045 - intrinsic 1,000 = 45c < the $23 dividend
    assert eng.hard_exit(held, view_of(itm_chain, events=(two,))) is ExitReason.EX_DIVIDEND
    assert eng.hard_exit(held, view_of(itm_chain, events=(three,))) is not ExitReason.EX_DIVIDEND
    off = msgspec.structs.replace(Config(), exits=msgspec.structs.replace(Config().exits, ex_dividend_guard=False))
    assert DefaultRiskEngine(off).hard_exit(held, view_of(itm_chain, events=(two,))) is not ExitReason.EX_DIVIDEND


def test_hard_exit_assignment_risk_fires_on_a_short_leg_with_almost_no_extrinsic() -> None:
    eng = engine()
    itm_call = contract_at(CHAIN, EXPIRY, Right.CALL, 440_000)
    thin = set_quote(CHAIN, itm_call, bid=1_000, ask=1_005)  # extrinsic 5c < the 10c floor
    fat = set_quote(CHAIN, itm_call, bid=1_040, ask=1_060)  # extrinsic 60c
    structure = Structure(
        kind=StructureKind.CALL_CREDIT,
        underlying="SPY",
        expiry=EXPIRY,
        last_session=CHAIN.last_session(EXPIRY),
        legs=(Leg(contract=itm_call, side=Side.SELL), Leg(contract=contract_at(CHAIN, EXPIRY, Right.CALL, 450_000), side=Side.BUY)),
    )
    position = position_of(structure, open_net=BandPrices(orats=-400, worst=-390, mid=-405), liq_value=100, max_profit=40_000)
    assert eng.hard_exit(position, view_of(thin)) is ExitReason.ASSIGNMENT_RISK
    assert eng.hard_exit(position, view_of(fat)) is not ExitReason.ASSIGNMENT_RISK


# ======================================================================================================================
# 9.5 triggers: the daily-loss halt, the drawdown kill, proportionate actions
# ======================================================================================================================


def test_the_daily_loss_halt_measures_against_the_previous_session_end() -> None:
    eng = engine()
    halting = portfolio(
        day_start_equity=INITIAL_EQUITY,
        equity=BandPrices(orats=9_790_000, worst=9_700_000, mid=9_900_000),
    )
    detail = eng.daily_loss(halting)
    assert detail is not None and detail.startswith("book loss_ppm=21000")

    just_short = portfolio(day_start_equity=INITIAL_EQUITY, equity=BandPrices(orats=9_810_000, worst=9_700_000, mid=9_900_000))
    assert eng.daily_loss(just_short) is None  # 1.9%

    exactly = portfolio(day_start_equity=INITIAL_EQUITY, equity=BandPrices(orats=9_800_000, worst=0, mid=0))
    assert eng.daily_loss(exactly) is not None  # 2.0% is a breach: ">= risk.daily_loss_halt_pct"
    assert eng.daily_loss(portfolio(day_start_equity=0)) is None


def test_the_paper_daily_loss_halt_takes_the_worse_of_the_book_and_the_broker() -> None:
    eng = engine()
    pf = portfolio(
        day_start_equity=INITIAL_EQUITY,
        equity=BandPrices(orats=9_900_000, worst=9_900_000, mid=9_900_000),  # book: 1.0%
        broker_prev_equity=INITIAL_EQUITY,
        broker_equity=9_780_000,  # broker: 2.2%
    )
    detail = eng.daily_loss(pf)
    assert detail is not None and detail.startswith("broker loss_ppm=22000")
    # the other way round: the book is worse and the broker is calm
    pf = msgspec.structs.replace(pf, equity=BandPrices(orats=9_700_000, worst=0, mid=0), broker_equity=9_990_000)
    detail = eng.daily_loss(pf)
    assert detail is not None and detail.startswith("book loss_ppm=30000")


def test_the_drawdown_kill_fires_at_the_decision_mark_on_the_worse_of_book_and_broker() -> None:
    eng = engine()
    assert eng.on_mark(portfolio()) == ()
    breached = portfolio(peak_equity=INITIAL_EQUITY, equity=BandPrices(orats=9_200_000, worst=0, mid=0))
    ((trigger, action, detail),) = eng.on_mark(breached)
    assert trigger is KillTrigger.DRAWDOWN and action is TriggerAction.KILL and "drawdown_ppm=80000" in detail
    assert eng.on_mark(portfolio(peak_equity=INITIAL_EQUITY, equity=BandPrices(orats=9_210_000, worst=0, mid=0))) == ()
    broker = portfolio(peak_equity=INITIAL_EQUITY, equity=BandPrices(orats=9_900_000, worst=0, mid=0), broker_equity=9_100_000)
    assert eng.on_mark(broker)[0][0] is KillTrigger.DRAWDOWN


def health(**overrides: Any) -> HealthSnapshot:
    base: dict[str, Any] = {
        "clock_skew_ms": None,
        "chain_age_s": {"SPY": 0},
        "two_sided_frac_ppm": {"SPY": 900_000},
        "stale_quote_frac_ppm": {"SPY": 0},
        "reconcile_ok": True,
        "model_ok": True,
        "ledger_ok": True,
        "spend_blocked": False,
        "expiry_violation": False,
        "assignment_seen": False,
    }
    base.update(overrides)
    return HealthSnapshot(**base)


def test_pre_cycle_kills_on_every_hard_trigger() -> None:
    eng = engine()
    for field, trigger in (
        ("reconcile_ok", KillTrigger.RECONCILE_MISMATCH),
        ("model_ok", KillTrigger.MODEL_MISMATCH),
        ("ledger_ok", KillTrigger.LEDGER_CORRUPT),
    ):
        gate, triggers = eng.pre_cycle(portfolio(), health(**{field: False}))
        assert triggers[0][0] is trigger and triggers[0][1] is TriggerAction.KILL
        assert gate.kill and not gate.allow_entries
    for field, trigger in (("expiry_violation", KillTrigger.EXPIRY_VIOLATION), ("assignment_seen", KillTrigger.ASSIGNMENT)):
        _, triggers = eng.pre_cycle(portfolio(), health(**{field: True}))
        assert triggers[0][0] is trigger and triggers[0][1] is TriggerAction.KILL


def test_pre_cycle_stale_quotes_halt_at_once_and_kill_only_after_three_stale_sessions_with_a_position() -> None:
    eng = engine()
    stale = health(chain_age_s={"SPY": 600})
    held = portfolio(positions=(position_of(NARROW),))
    _, triggers = eng.pre_cycle(held, stale)
    assert triggers[0] == (KillTrigger.STALE_QUOTES, TriggerAction.HALT, "SPY")
    _, triggers = eng.pre_cycle(msgspec.structs.replace(held, stale_sessions=1), stale)
    assert triggers[0][1] is TriggerAction.HALT
    _, triggers = eng.pre_cycle(msgspec.structs.replace(held, stale_sessions=2), stale)
    assert triggers[0][1] is TriggerAction.KILL  # the third consecutive stale session
    # with a FLAT book there is nothing to flatten: it never escalates
    _, triggers = eng.pre_cycle(portfolio(stale_sessions=9), stale)
    assert triggers[0][1] is TriggerAction.HALT
    # the other two stale conditions of 9.5
    _, triggers = eng.pre_cycle(held, health(two_sided_frac_ppm={"SPY": 500_000}))
    assert triggers[0][0] is KillTrigger.STALE_QUOTES
    _, triggers = eng.pre_cycle(held, health(stale_quote_frac_ppm={"SPY": 250_000}))
    assert triggers[0][0] is KillTrigger.STALE_QUOTES
    # an operator who downgrades the trigger to a plain halt never gets the escalation (deviation V2)
    downgraded = msgspec.structs.replace(
        Config(),
        kill=msgspec.structs.replace(
            Config().kill, actions=msgspec.structs.replace(Config().kill.actions, stale_quotes=TriggerAction.HALT)
        ),
    )
    _, triggers = DefaultRiskEngine(downgraded).pre_cycle(msgspec.structs.replace(held, stale_sessions=5), stale)
    assert triggers[0][1] is TriggerAction.HALT


def test_pre_cycle_escalates_decider_and_broker_failures_by_their_persistence_thresholds() -> None:
    eng = engine()
    _, triggers = eng.pre_cycle(portfolio(jev_fail_sessions=1), health())
    assert triggers[0][:2] == (KillTrigger.JEV_ERRORS, TriggerAction.HALT)
    _, triggers = eng.pre_cycle(portfolio(jev_fail_sessions=3), health())
    assert triggers[0][:2] == (KillTrigger.JEV_ERRORS, TriggerAction.KILL)
    _, triggers = eng.pre_cycle(portfolio(broker_fail_streak=1), health())
    assert triggers == ()
    _, triggers = eng.pre_cycle(portfolio(broker_fail_streak=2), health())
    assert triggers[0][:2] == (KillTrigger.BROKER_ERRORS, TriggerAction.HALT)
    _, triggers = eng.pre_cycle(portfolio(broker_fail_streak=6), health())
    assert triggers[0][:2] == (KillTrigger.BROKER_ERRORS, TriggerAction.KILL)
    _, triggers = eng.pre_cycle(portfolio(orders_last_minute=10), health())
    assert triggers[0][:2] == (KillTrigger.ORDER_RATE, TriggerAction.KILL)
    _, triggers = eng.pre_cycle(portfolio(), health(clock_skew_ms=6_000))
    assert triggers[0][:2] == (KillTrigger.CLOCK_SKEW, TriggerAction.HALT)


def test_a_spend_stop_halts_entries_and_is_never_a_kill_input() -> None:
    """INV-17: a spend stop halts entries in paper and never feeds the kill switch."""
    gate, triggers = engine().pre_cycle(portfolio(), health(spend_blocked=True))
    assert triggers == () and not gate.kill
    assert not gate.allow_entries and not gate.allow_manage_jev and "spend_blocked" in gate.reasons


def test_the_cycle_gate_reports_the_book_halt_and_the_kill_state() -> None:
    eng = engine()
    gate, triggers = eng.pre_cycle(portfolio(), health())
    assert gate.allow_entries and gate.allow_manage_jev and not gate.kill and triggers == ()
    gate, _ = eng.pre_cycle(portfolio(halt_entries=True, halt_reasons=("daily_loss_halt",)), health())
    assert not gate.allow_entries and "daily_loss_halt" in gate.reasons
    gate, _ = eng.pre_cycle(portfolio(kill_state=KillState.LOCKED), health())
    assert not gate.allow_entries
    gate, _ = eng.pre_cycle(portfolio(jev_fail_sessions=1), health())
    assert not gate.allow_manage_jev


def test_the_headline_band_follows_the_fill_rule() -> None:
    assert headline_band(Config()) is Band.ORATS
    worst = msgspec.structs.replace(Config(), cadence=msgspec.structs.replace(Config().cadence, fill_rule=FillRule.SAME_SNAPSHOT_WORST))
    assert headline_band(worst) is Band.WORST
    assert DefaultRiskEngine(worst).headline is Band.WORST


def test_the_lowest_tier_comes_from_the_configured_tier_tables() -> None:
    cfg = msgspec.structs.replace(
        Config(),
        rules=msgspec.structs.replace(
            Config().rules,
            tiers=msgspec.structs.replace(
                Config().rules.tiers, score=((0.8, 1.0),), peakedness=((0.8, 1.0),), environment=(1.0, 1.0, 1.0, 0.0)
            ),
        ),
    )
    eng = DefaultRiskEngine(cfg)
    assert eng.lowest_tier() == 1
    assert eng.budget_floor(portfolio()) == 100_000
    assert isinstance(RiskConfig(), RiskConfig)
