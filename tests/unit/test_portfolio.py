"""`portfolio.Book` - the ledger fold of DESIGN.md 10.8 (2.4, 2.11, 3.6 `BookP`).

Every expected number below is hand-computed from the 9.2 / 10.7 formulas and written as a literal, never re-derived by
calling the same function the code calls.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import msgspec
import pytest

from jevbot.cal import XnysCalendar
from jevbot.config import Config
from jevbot.errors import InvariantError
from jevbot.portfolio import Book
from jevbot.types import (
    Band,
    BandPrices,
    EntryContext,
    Fill,
    KillState,
    LedgerEntry,
    LedgerKind,
    Leg,
    LegFill,
    OptionContract,
    OrderIntent,
    OrderLeg,
    OrderPurpose,
    OrderStatus,
    PositionIntent,
    Right,
    Side,
    Slot,
    SnapshotKey,
    Structure,
    StructureKind,
)
from tests.fixtures.memory_ledger import MemoryLedger

CAL = XnysCalendar()
CFG = Config()
SESSION = date(2024, 5, 17)
NEXT_SESSION = date(2024, 5, 20)
EXPIRY = date(2024, 6, 21)
LAST_SESSION = date(2024, 6, 21)
AS_OF = CAL.open_close(SESSION)[1]
KEY = SnapshotKey(session=SESSION, slot=Slot.EOD)
INITIAL_CASH = 10_000_000  # $100,000

# hand-computed constants of the fixture spread (see the module docstring and the assertions below)
FEE_RT = 18  # cents, 10.7 round trip for 2 legs at 200 / 80 cents: ceil((167,780 + 5,768) micro-dollars / 10,000)
WIDTH_C = 500  # $5 wide put credit spread


def contract(strike_milli: int, right: Right = Right.PUT, expiry: date = EXPIRY) -> OptionContract:
    return OptionContract(underlying="SPY", expiry=expiry, right=right, strike_milli=strike_milli)


SHORT_PUT = contract(440_000)
LONG_PUT = contract(435_000)
SPREAD = Structure(
    kind=StructureKind.PUT_CREDIT,
    underlying="SPY",
    expiry=EXPIRY,
    last_session=LAST_SESSION,
    legs=(Leg(contract=LONG_PUT, side=Side.BUY), Leg(contract=SHORT_PUT, side=Side.SELL)),
)
ENTRY_CTX = EntryContext(
    entry_thesis="trend up, iv rich",
    entry_codes={"trend": "up", "iv_vs_realized": "rich", "iv_rank": "p60_p80"},
    entry_spot=45_000,
    entry_iv30_bp=1830,
    entry_em_hold_tenths=42,
    open_mid_at_decision=-125,
)
POSITION_ID = "p" * 16
DECISION_ID = "d" * 32


def open_intent(*, qty: int = 2, session: date = SESSION, key: SnapshotKey = KEY) -> OrderIntent:
    return OrderIntent(
        intent_id=f"jb1-aaaaaaaa-{session:%y%m%d}-{DECISION_ID[:12]}-open-00",
        decision_id=DECISION_ID,
        position_id=POSITION_ID,
        purpose=OrderPurpose.OPEN,
        part=0,
        underlying="SPY",
        legs=(
            OrderLeg(contract=LONG_PUT, side=Side.BUY, position_intent=PositionIntent.BTO),
            OrderLeg(contract=SHORT_PUT, side=Side.SELL, position_intent=PositionIntent.STO),
        ),
        qty=qty,
        limit_start=-120,
        limit_natural=-110,
        reason="entry",
        mandatory=False,
        session=session,
        key=key,
        tier_ppm=1_000_000,
        structure=SPREAD,
        entry_ctx=ENTRY_CTX,
    )


def close_intent(*, qty: int = 2, session: date = NEXT_SESSION) -> OrderIntent:
    return OrderIntent(
        intent_id=f"jb1-aaaaaaaa-{session:%y%m%d}-{DECISION_ID[:12]}-close-00",
        decision_id=DECISION_ID,
        position_id=POSITION_ID,
        purpose=OrderPurpose.CLOSE,
        part=0,
        underlying="SPY",
        legs=(
            OrderLeg(contract=LONG_PUT, side=Side.SELL, position_intent=PositionIntent.STC),
            OrderLeg(contract=SHORT_PUT, side=Side.BUY, position_intent=PositionIntent.BTC),
        ),
        qty=qty,
        limit_start=60,
        limit_natural=70,
        reason="profit_target",
        mandatory=False,
        session=session,
        key=SnapshotKey(session=session, slot=Slot.EOD),
        structure=SPREAD,
    )


def leg_fills(long_price: int = 80, short_price: int = 200) -> tuple[LegFill, ...]:
    return (
        LegFill(
            occ=LONG_PUT.occ, side=Side.BUY, bid=long_price - 2, ask=long_price + 2, orats=long_price, worst=long_price + 2, mid=long_price
        ),
        LegFill(
            occ=SHORT_PUT.occ,
            side=Side.SELL,
            bid=short_price - 2,
            ask=short_price + 2,
            orats=short_price,
            worst=short_price - 2,
            mid=short_price,
        ),
    )


def fill_payload(
    intent: OrderIntent,
    *,
    qty: int,
    net: BandPrices,
    cum_qty: int | None = None,
    session: date = SESSION,
    realised: BandPrices | None = None,
    legs: tuple[LegFill, ...] | None = None,
    fees_micro: int = 0,
) -> dict[str, Any]:
    cum = cum_qty if cum_qty is not None else qty
    fill = Fill(
        fill_id=f"fill-{intent.intent_id[-8:]}-{cum}",
        client_order_id=f"{intent.intent_id}-00",
        intent_id=intent.intent_id,
        decision_id=intent.decision_id,
        position_id=intent.position_id,
        purpose=intent.purpose,
        structure_id=SPREAD.structure_id,
        qty=qty,
        key=SnapshotKey(session=session, slot=Slot.EOD),
        ts=CAL.open_close(session)[1],
        net=net,
        legs=legs if legs is not None else leg_fills(),
        fees_micro=fees_micro,
        forced=False,
        model_reject=(),
        quality="ok",
        source="sim",
        broker_order_id="b-1",
        broker_net=None,
    )
    payload: dict[str, Any] = msgspec.to_builtins(fill)
    if realised is not None:
        payload["realised_pnl"] = msgspec.to_builtins(realised)
    return payload


class World:
    """A ledger + a Book folding every entry the ledger receives (what the live cycle does)."""

    def __init__(self, *, strict: bool = True, cfg: Config = CFG) -> None:
        self.ledger = MemoryLedger(strict=strict)
        self.book = Book(initial_cash=INITIAL_CASH, headline=Band.ORATS, cfg=cfg, calendar=CAL)

    def add(self, kind: LedgerKind, payload: dict[str, Any], *, session: date = SESSION, as_of: datetime | None = None) -> LedgerEntry:
        entry = self.ledger.append(kind, session, as_of if as_of is not None else CAL.open_close(session)[1], payload)
        self.book.apply(entry)
        return entry

    def session_start(self, session: date, slot: Slot = Slot.EOD) -> None:
        self.add(LedgerKind.SESSION_START, {"session": session.isoformat(), "slot": slot.value, "phase": "full"}, session=session)

    def session_end(self, session: date, equity: BandPrices) -> None:
        self.add(
            LedgerKind.SESSION_END,
            {
                "equity": msgspec.to_builtins(equity),
                "positions_digest": "",
                "n_decisions": 0,
                "n_forecasts": 0,
                "n_intents": 0,
                "invariant_no_expiry_risk": True,
            },
            session=session,
        )

    def order_status(
        self,
        intent: OrderIntent,
        status: OrderStatus,
        *,
        attempt: int = 0,
        filled_qty: int = 0,
        session: date = SESSION,
        as_of: datetime | None = None,
    ) -> None:
        self.add(
            LedgerKind.ORDER_STATUS,
            {
                "client_order_id": f"{intent.intent_id}-{attempt:02d}",
                "intent_id": intent.intent_id,
                "attempt": attempt,
                "status": status.value,
                "qty": intent.qty,
                "filled_qty": filled_qty,
                "limit": intent.limit_start,
                "broker_order_id": None,
                "reject_code": None,
                "tag": "",
            },
            session=session,
            as_of=as_of,
        )

    def replayed(self, cfg: Config = CFG) -> Book:
        return Book.replay(self.ledger, initial_cash=INITIAL_CASH, headline=Band.ORATS, cfg=cfg, calendar=CAL)


def open_position(world: World, *, qty: int = 2, net: BandPrices | None = None) -> OrderIntent:
    intent = open_intent(qty=qty)
    world.session_start(SESSION)
    world.add(LedgerKind.ORDER_INTENT, msgspec.to_builtins(intent))
    world.order_status(intent, OrderStatus.SUBMITTING)
    world.order_status(intent, OrderStatus.FILLED, filled_qty=qty)
    world.add(LedgerKind.FILL, fill_payload(intent, qty=qty, net=net or BandPrices(orats=-120, worst=-110, mid=-125)))
    return intent


# ======================================================================================================================
# open / close accounting (10.8, 9.2)
# ======================================================================================================================


def test_open_fill_moves_cash_per_band_and_builds_the_position_from_the_intent() -> None:
    world = World()
    open_position(world)
    state = world.book.state()
    assert len(state.positions) == 1
    position = state.positions[0]

    # cash: -net[b] * 100 * qty on every band (10.8) - a credit of 120 / 110 / 125 cents per share on 2 contracts
    assert state.cash == BandPrices(orats=INITIAL_CASH + 24_000, worst=INITIAL_CASH + 22_000, mid=INITIAL_CASH + 25_000)
    # max_loss at the ACTUAL worst-band fill: (width 500 - credit 110) * 100 + fee_rt 18 = 39,018 per contract
    assert position.max_loss == (WIDTH_C - 110) * 100 * 2 + FEE_RT * 2 == 78_036
    # max_profit at the headline band: the credit kept, 120 * 100 per contract
    assert position.max_profit == 24_000
    # bp: (width - credit_worst) * 100 * bp_haircut_mult = 39,000 per contract
    assert position.bp_reserved == 78_000
    assert position.qty == 2 and position.open_net == BandPrices(orats=-120, worst=-110, mid=-125)
    assert position.open_key == KEY and position.open_decision_id == DECISION_ID
    assert position.structure.last_session == LAST_SESSION  # ledger-sourced, never recomputed
    assert position.entry == ENTRY_CTX  # every EntryContext field is copied from the OPEN intent (10.8)
    assert state.opened_today == 1
    assert world.book.leg_positions() == {LONG_PUT.occ: 2, SHORT_PUT.occ: -2}


def test_equity_is_cash_minus_the_liquidation_value_of_every_position() -> None:
    world = World()
    open_position(world)
    world.add(
        LedgerKind.MARK,
        {
            "equity": msgspec.to_builtins(BandPrices(orats=0, worst=0, mid=0)),
            "cash": msgspec.to_builtins(BandPrices(orats=0, worst=0, mid=0)),
            "open_max_loss": 0,
            "bp_used": 0,
            "bp_utilisation_ppm": 0,
            "positions": {POSITION_ID: {"liq_value": 60, "mid_value": 55, "stale": False}},
            "net_delta_milli": 0,
            "net_vega_milli": 0,
        },
    )
    state = world.book.state()
    # liq_value 60 cents/share to close 2 contracts costs 12,000 cents
    assert state.equity.orats == state.cash.orats - 12_000 == INITIAL_CASH + 24_000 - 12_000
    assert state.positions[0].liq_value == 60 and state.positions[0].mid_value == 55 and state.positions[0].stale_marks == 0


def test_a_partial_fill_grows_the_position_and_blends_the_open_price_against_us() -> None:
    world = World()
    intent = open_intent(qty=3)
    world.session_start(SESSION)
    world.add(LedgerKind.ORDER_INTENT, msgspec.to_builtins(intent))
    world.order_status(intent, OrderStatus.SUBMITTING)
    world.order_status(intent, OrderStatus.PARTIAL, filled_qty=1)
    world.add(LedgerKind.FILL, fill_payload(intent, qty=1, cum_qty=1, net=BandPrices(orats=-130, worst=-120, mid=-135)))
    world.order_status(intent, OrderStatus.FILLED, filled_qty=3)
    world.add(LedgerKind.FILL, fill_payload(intent, qty=2, cum_qty=3, net=BandPrices(orats=-121, worst=-111, mid=-126)))
    position = world.book.state().positions[0]
    assert position.qty == 3
    # weighted average, signed value CEILED (against us on both signs): (-130 * 1 + -121 * 2) / 3 = -124.0 -> -124
    #                                                                   (-120 * 1 + -111 * 2) / 3 = -114.0 -> -114
    #                                                                   (-135 * 1 + -126 * 2) / 3 = -129.0 -> -129
    assert position.open_net == BandPrices(orats=-124, worst=-114, mid=-129)
    assert position.max_loss == ((WIDTH_C - 114) * 100 + FEE_RT) * 3 == 115_854
    assert world.book.state().opened_today == 1  # one STRUCTURE was opened, not two
    assert world.book.filled_qty(f"{intent.intent_id}-00") == 3


def test_blending_rounds_a_debit_up_and_a_credit_toward_zero() -> None:
    """Both directions of "against us" on a price that is not a whole cent per contract."""
    world = World()
    intent = open_intent(qty=3)
    world.session_start(SESSION)
    world.add(LedgerKind.ORDER_INTENT, msgspec.to_builtins(intent))
    world.order_status(intent, OrderStatus.SUBMITTING)
    world.order_status(intent, OrderStatus.PARTIAL, filled_qty=1)
    world.add(LedgerKind.FILL, fill_payload(intent, qty=1, cum_qty=1, net=BandPrices(orats=-100, worst=100, mid=-100)))
    world.order_status(intent, OrderStatus.FILLED, filled_qty=3)
    world.add(LedgerKind.FILL, fill_payload(intent, qty=2, cum_qty=3, net=BandPrices(orats=-101, worst=101, mid=-101)))
    # credits: (-100 - 202) / 3 = -100.667 -> ceil -100 (LESS credit than reality); debits: 302 / 3 = 100.667 -> 101 (MORE)
    assert world.book.state().positions[0].open_net == BandPrices(orats=-100, worst=101, mid=-100)


def test_close_fill_removes_the_position_and_sets_a_re_entry_cooldown() -> None:
    world = World()
    open_position(world)
    world.session_end(SESSION, BandPrices(orats=INITIAL_CASH, worst=INITIAL_CASH, mid=INITIAL_CASH))
    world.session_start(NEXT_SESSION)
    closing = close_intent()
    world.add(LedgerKind.ORDER_INTENT, msgspec.to_builtins(closing), session=NEXT_SESSION)
    world.order_status(closing, OrderStatus.FILLED, filled_qty=2, session=NEXT_SESSION)
    world.add(
        LedgerKind.FILL,
        fill_payload(
            closing,
            qty=2,
            net=BandPrices(orats=60, worst=70, mid=55),
            session=NEXT_SESSION,
            realised=BandPrices(orats=12_000, worst=8_000, mid=14_000),
        ),
        session=NEXT_SESSION,
    )
    state = world.book.state()
    assert state.positions == ()
    # cash: +24,000 on the open, -12,000 on the close (orats); the realised P&L in the payload is (120 - 60) * 100 * 2
    assert state.cash.orats == INITIAL_CASH + 24_000 - 12_000
    assert state.cash.orats - INITIAL_CASH == 12_000
    # the cooldown runs to the 3rd session after the close (rules.reentry_cooldown_sessions = 3)
    assert state.cooldowns == (("SPY", "bullish", CAL.next_session(NEXT_SESSION, 3)),)
    assert world.book.leg_positions() == {}


def test_a_partial_close_scales_the_position_limits_down() -> None:
    world = World()
    open_position(world, qty=4)
    closing = close_intent(qty=2)
    world.add(LedgerKind.ORDER_INTENT, msgspec.to_builtins(closing), session=SESSION)
    world.add(LedgerKind.FILL, fill_payload(closing, qty=2, net=BandPrices(orats=60, worst=70, mid=55)))
    position = world.book.state().positions[0]
    assert position.qty == 2
    assert position.max_loss == (39_018 * 4) * 2 // 4 == 78_036
    assert position.bp_reserved == 78_000 and position.max_profit == 24_000
    assert world.book.state().cooldowns == ()  # a cooldown is set when the position is GONE


def test_an_open_fill_without_its_order_intent_is_an_invariant_error() -> None:
    world = World(strict=False)
    world.session_start(SESSION)
    with pytest.raises(InvariantError, match="ORDER_INTENT"):
        world.add(LedgerKind.FILL, fill_payload(open_intent(), qty=1, net=BandPrices(orats=-120, worst=-110, mid=-125)))


def test_a_kill_close_of_a_leg_the_ledger_never_knew_only_moves_cash() -> None:
    """9.5 K3 flattens what the BROKER reports; an unmatched leg has no Book position to reduce."""
    world = World()
    world.session_start(SESSION)
    stray = msgspec.structs.replace(close_intent(session=SESSION), position_id="z" * 16, purpose=OrderPurpose.KILL)
    world.add(LedgerKind.ORDER_INTENT, msgspec.to_builtins(stray))
    world.add(LedgerKind.FILL, fill_payload(stray, qty=1, net=BandPrices(orats=40, worst=45, mid=38)))
    assert world.book.state().positions == ()
    assert world.book.state().cash.orats == INITIAL_CASH - 4_000


# ======================================================================================================================
# day_start_equity, counters, halts (9.5, 10.8)
# ======================================================================================================================


def test_day_start_equity_is_the_previous_session_end_and_initial_cash_on_the_first_session() -> None:
    world = World()
    assert world.book.state().day_start_equity == INITIAL_CASH  # nothing ended yet
    world.session_start(SESSION)
    assert world.book.state().day_start_equity == INITIAL_CASH
    world.session_end(SESSION, BandPrices(orats=10_210_000, worst=10_100_000, mid=10_300_000))
    world.session_start(NEXT_SESSION)
    # the HEADLINE value of the previous SESSION_END, not the worst or mid band and not today's first mark
    assert world.book.state().day_start_equity == 10_210_000
    assert world.book.state().peak_equity == 10_210_000


def test_session_start_resets_opened_today_and_the_session_halt_only_on_a_new_session() -> None:
    world = World()
    open_position(world)
    world.add(LedgerKind.RISK_EVENT, {"type": "daily_loss_halt", "trigger": "", "detail": "book loss_ppm=21000", "counters": {}})
    state = world.book.state()
    assert state.halt_entries and state.halt_reasons == ("book loss_ppm=21000",) and state.opened_today == 1
    world.session_start(SESSION, Slot.EXEC)  # a later slot of the SAME session must not clear the halt (9.5)
    assert world.book.state().halt_entries and world.book.state().opened_today == 1
    assert world.book.last_key == SnapshotKey(session=SESSION, slot=Slot.EXEC)
    world.session_start(NEXT_SESSION)
    state = world.book.state()
    assert not state.halt_entries and state.halt_reasons == () and state.opened_today == 0


def test_risk_events_clear_a_halt_and_carry_the_session_counters() -> None:
    world = World()
    world.session_start(SESSION)
    world.add(LedgerKind.RISK_EVENT, {"type": "halt_set", "trigger": "stale_quotes", "detail": "SPY", "counters": {}})
    assert world.book.state().halt_reasons == ("SPY",)
    world.add(
        LedgerKind.RISK_EVENT,
        {
            "type": "trigger_seen",
            "trigger": "stale_quotes",
            "detail": "",
            "counters": {"stale_sessions": 2, "jev_fail_sessions": 1, "nope": 9},
        },
    )
    state = world.book.state()
    assert state.stale_sessions == 2 and state.jev_fail_sessions == 1 and state.broker_fail_streak == 0
    world.add(LedgerKind.RISK_EVENT, {"type": "halt_cleared", "trigger": "", "detail": "", "counters": {}})
    assert not world.book.state().halt_entries and world.book.state().halt_reasons == ()


def test_fee_charges_every_band_and_resets_the_accumulator() -> None:
    world = World()
    open_position(world)
    world.add(LedgerKind.FILL, fill_payload(open_intent(), qty=0, cum_qty=99, net=BandPrices(orats=0, worst=0, mid=0), fees_micro=173_548))
    assert world.book.state().fees_accrued_micro == 173_548
    world.add(LedgerKind.FEE, {"fee_cents": 18, "accrued_micro_before": 173_548})  # cdiv(173,548, 10,000) = 18
    state = world.book.state()
    assert state.fees_accrued_micro == 0
    assert state.cash == BandPrices(orats=INITIAL_CASH + 24_000 - 18, worst=INITIAL_CASH + 22_000 - 18, mid=INITIAL_CASH + 25_000 - 18)


def test_marks_track_stale_counters_broker_fields_and_the_peak() -> None:
    world = World()
    open_position(world)

    def mark(liq: int, *, stale: bool, broker_equity: int | None = None) -> None:
        payload: dict[str, Any] = {
            "equity": msgspec.to_builtins(BandPrices(orats=0, worst=0, mid=0)),
            "cash": msgspec.to_builtins(BandPrices(orats=0, worst=0, mid=0)),
            "open_max_loss": 0,
            "bp_used": 0,
            "bp_utilisation_ppm": 0,
            "positions": {POSITION_ID: {"liq_value": liq, "mid_value": liq, "stale": stale}},
            "net_delta_milli": 0,
            "net_vega_milli": 0,
        }
        if broker_equity is not None:
            payload |= {"broker_equity": broker_equity, "broker_prev_equity": broker_equity + 1, "broker_options_bp": 7}
        world.add(LedgerKind.MARK, payload)

    mark(100, stale=True)
    assert world.book.state().positions[0].stale_marks == 1
    mark(100, stale=True)
    assert world.book.state().positions[0].stale_marks == 2
    mark(20, stale=False)
    assert world.book.state().positions[0].stale_marks == 0
    peak = world.book.state().equity.orats
    assert world.book.state().peak_equity == peak
    mark(400, stale=False, broker_equity=9_000_000)  # a loss: the peak stays where it was
    state = world.book.state()
    assert state.peak_equity == peak and state.equity.orats < peak
    assert (state.broker_equity, state.broker_prev_equity, state.broker_options_bp) == (9_000_000, 9_000_001, 7)


def test_orders_last_minute_counts_only_submissions_inside_the_trailing_window() -> None:
    world = World()
    intent = open_intent()
    world.session_start(SESSION)
    world.add(LedgerKind.ORDER_INTENT, msgspec.to_builtins(intent))
    world.order_status(intent, OrderStatus.SUBMITTING, as_of=AS_OF - timedelta(seconds=90))
    world.order_status(intent, OrderStatus.SUBMITTING, attempt=1, as_of=AS_OF - timedelta(seconds=30))
    world.order_status(intent, OrderStatus.SUBMITTED, attempt=1, as_of=AS_OF)
    assert world.book.state().orders_last_minute == 1


def test_a_manage_decision_carries_the_latch_and_the_text_watch_counter() -> None:
    world = World()
    open_position(world)
    world.add(
        LedgerKind.DECISION,
        {
            "decision_id": DECISION_ID,
            "kind": "manage",
            "subject_alias": "UNDERLYING_A",
            "text": "on",
            "requests": [],
            "rules": {"position_id": POSITION_ID, "action": "hold", "exit_latch": True, "watch_text": 2},
            "facts": {},
            "tier": "B",
        },
    )
    position = world.book.state().positions[0]
    assert position.exit_latch is True and position.watch_text == 2
    world.add(
        LedgerKind.DECISION,
        {
            "decision_id": DECISION_ID,
            "kind": "entry",  # an ENTRY decision never touches a position
            "subject_alias": "UNDERLYING_A",
            "text": "on",
            "requests": [],
            "rules": {"position_id": POSITION_ID, "exit_latch": False, "watch_text": 0},
            "facts": {},
            "tier": "B",
        },
    )
    assert world.book.state().positions[0].exit_latch is True


def test_kill_steps_and_a_rearm_move_the_state_and_the_peak() -> None:
    world = World()
    open_position(world)
    assert world.book.state().kill_state is KillState.ARMED
    for step, expected in (
        ("tripped", KillState.TRIPPED),
        ("cancelled", KillState.TRIPPED),
        ("close_submitted", KillState.FLATTENING),
        ("not_flat", KillState.NOT_FLAT),
        ("flat_verified", KillState.FLATTENING),
        ("locked", KillState.LOCKED),
    ):
        world.add(LedgerKind.KILL, {"event_id": "ev-1", "step": step, "trigger": "drawdown", "detail": ""})
        assert world.book.state().kill_state is expected
    assert world.book.state().kill_event_id == "ev-1"
    peak_before = world.book.state().peak_equity
    world.add(LedgerKind.REARM, {"event_id": "ev-1", "reset_peak": False, "operator_note": "n", "ledger_head_at_rearm": "h"})
    assert world.book.state().kill_state is KillState.ARMED and world.book.state().kill_event_id is None
    assert world.book.state().peak_equity == peak_before
    world.add(LedgerKind.KILL, {"event_id": "ev-2", "step": "tripped", "trigger": "drawdown", "detail": ""})
    world.add(LedgerKind.REARM, {"event_id": "ev-2", "reset_peak": True, "operator_note": "n", "ledger_head_at_rearm": "h"})
    assert world.book.state().peak_equity == world.book.state().equity.orats


def test_run_start_sets_the_opening_cash_from_the_ledger() -> None:
    world = World(strict=False)
    world.add(LedgerKind.RUN_START, {"initial_cash": 5_000_000, "mode": "backtest"})
    state = world.book.state()
    assert state.cash.orats == state.day_start_equity == state.peak_equity == 5_000_000


# ======================================================================================================================
# working orders (3.6 BookP)
# ======================================================================================================================


def test_working_orders_intent_lookup_and_the_synthetic_intent_state() -> None:
    world = World()
    intent = open_intent()
    world.session_start(SESSION)
    world.add(LedgerKind.ORDER_INTENT, msgspec.to_builtins(intent))
    assert world.book.has_intent(intent.intent_id) and world.book.intent(intent.intent_id) == intent
    with pytest.raises(InvariantError):
        world.book.intent("nope")
    # ledgered but never submitted: open_orders() reports a synthetic INTENT state under the attempt-0 id (9.6)
    (pair,) = world.book.open_orders()
    assert pair[0] == intent and pair[1].status is OrderStatus.INTENT and pair[1].client_order_id == f"{intent.intent_id}-00"
    assert world.book.state().working == (intent,)

    world.order_status(intent, OrderStatus.SUBMITTING)
    (pair,) = world.book.open_orders()
    assert pair[1].status is OrderStatus.SUBMITTING
    world.order_status(intent, OrderStatus.CANCELLED)
    assert world.book.open_orders() == () and world.book.state().working == ()

    # a repricing: attempt 0 terminal, attempt 1 working
    world.order_status(intent, OrderStatus.SUBMITTING, attempt=1)
    (pair,) = world.book.open_orders()
    assert pair[1].client_order_id == f"{intent.intent_id}-01"
    assert world.book.state().working == (intent,)


# ======================================================================================================================
# replay == live (10.8, INV-24)
# ======================================================================================================================


def test_replay_equals_the_live_book_field_for_field() -> None:
    world = World()
    intent = open_position(world)
    world.add(
        LedgerKind.DECISION,
        {
            "decision_id": DECISION_ID,
            "kind": "manage",
            "subject_alias": "UNDERLYING_A",
            "text": "on",
            "requests": [],
            "rules": {"position_id": POSITION_ID, "action": "hold", "exit_latch": True, "watch_text": 3},
            "facts": {},
            "tier": "B",
        },
    )
    world.add(
        LedgerKind.MARK,
        {
            "equity": msgspec.to_builtins(BandPrices(orats=0, worst=0, mid=0)),
            "cash": msgspec.to_builtins(BandPrices(orats=0, worst=0, mid=0)),
            "open_max_loss": 0,
            "bp_used": 0,
            "bp_utilisation_ppm": 0,
            "positions": {POSITION_ID: {"liq_value": 70, "mid_value": 66, "stale": True}},
            "net_delta_milli": 0,
            "net_vega_milli": 0,
            "broker_equity": 9_999_000,
            "broker_prev_equity": 10_050_000,
            "broker_options_bp": 4_000_000,
        },
    )
    world.add(LedgerKind.FEE, {"fee_cents": 7, "accrued_micro_before": 65_000})
    world.session_end(SESSION, BandPrices(orats=10_012_000, worst=10_000_000, mid=10_020_000))
    world.session_start(NEXT_SESSION)
    world.add(
        LedgerKind.RISK_EVENT,
        {"type": "halt_set", "trigger": "spend", "detail": "spend_blocked", "counters": {"stale_sessions": 1}},
        session=NEXT_SESSION,
    )
    world.add(LedgerKind.KILL, {"event_id": "ev-9", "step": "tripped", "trigger": "drawdown", "detail": "dd"}, session=NEXT_SESSION)

    live = world.book.state()
    replayed = world.replayed()
    assert replayed.state() == live
    assert replayed.last_key == world.book.last_key
    assert replayed.leg_positions() == world.book.leg_positions()
    assert replayed.open_orders() == world.book.open_orders()
    assert replayed.intent(intent.intent_id) == intent
    # the fields 10.8 names explicitly
    position = replayed.state().positions[0]
    assert position.entry == ENTRY_CTX
    for name in EntryContext.__struct_fields__:
        assert getattr(position.entry, name) == getattr(ENTRY_CTX, name)
    assert position.structure.last_session == LAST_SESSION
    assert (position.exit_latch, position.watch_text, position.stale_marks) == (True, 3, 1)
    assert replayed.state().day_start_equity == 10_012_000
    assert replayed.state().kill_state is KillState.TRIPPED


def test_the_positions_entry_codes_are_not_shared_with_the_folded_intent() -> None:
    """`EntryContext.entry_codes` is a mutable dict on a frozen struct: the Position gets its own copy (10.8)."""
    world = World()
    intent_id = open_position(world).intent_id
    folded = world.book.intent(intent_id)
    assert folded.entry_ctx is not None
    folded.entry_ctx.entry_codes["trend"] = "tampered"
    assert world.book.state().positions[0].entry.entry_codes["trend"] == "up"


def test_apply_refuses_anything_that_is_not_a_ledger_entry() -> None:
    book = Book(initial_cash=INITIAL_CASH, headline=Band.ORATS, cfg=CFG, calendar=CAL)
    with pytest.raises(InvariantError):
        book.apply({"kind": "fill"})  # type: ignore[arg-type]
    with pytest.raises(InvariantError):
        Book(initial_cash=True, headline=Band.ORATS)  # type: ignore[arg-type]


def test_the_book_ignores_entry_kinds_it_does_not_fold() -> None:
    world = World(strict=False)
    open_position(world)
    before = world.book.state()
    world.add(LedgerKind.ANOMALY, {"type": "stale_mark", "detail": "x"})
    world.add(LedgerKind.RECONCILE, {"ok": True, "order_actions": [], "diff": {}, "foreign_orders": [], "activities": []})
    world.add(LedgerKind.BROKER_FILL, {"client_order_id": "c", "cum_qty": 1, "broker_net": None, "ts": "2024-05-17T20:00:00Z"})
    assert world.book.state() == before


def test_the_headline_band_decides_which_equity_the_book_reports() -> None:
    world = World()
    open_position(world)
    worst_book = Book.replay(world.ledger, initial_cash=INITIAL_CASH, headline=Band.WORST, cfg=CFG, calendar=CAL)
    assert worst_book.headline is Band.WORST
    world.session_end(SESSION, BandPrices(orats=1, worst=2, mid=3))
    worst_book = Book.replay(world.ledger, initial_cash=INITIAL_CASH, headline=Band.WORST, cfg=CFG, calendar=CAL)
    assert worst_book.state().day_start_equity == 2
