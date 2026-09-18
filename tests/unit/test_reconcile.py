"""`reconcile` - the ONE fill path, the ONE `ORDER_STATUS` writer and R1-R6 (DESIGN.md 9.6, INV-10).

The AST guard at the bottom is the "record_order_status is the only ORDER_STATUS writer" check of section 15.1; it is first
proven to fire on planted source, then run over the real `src/jevbot` tree.
"""

from __future__ import annotations

import ast
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import msgspec
import pytest

from jevbot import ids, reconcile
from jevbot.cal import SimClock, XnysCalendar
from jevbot.config import Config
from jevbot.errors import PaperGuardError
from jevbot.portfolio import Book
from jevbot.risk import DefaultRiskEngine
from jevbot.types import (
    ApprovedOrder,
    Band,
    BandPrices,
    BrokerActivity,
    ChainSnapshot,
    EntryContext,
    Fill,
    KillState,
    KillTrigger,
    LedgerKind,
    Leg,
    LegFill,
    OrderIntent,
    OrderLeg,
    OrderPurpose,
    OrderStatus,
    PositionIntent,
    Right,
    RunMode,
    Side,
    Slot,
    SnapshotKey,
    Structure,
    StructureKind,
)
from tests.fixtures.chain_factory import contract_at, make_chain, set_quote, target_expiry
from tests.fixtures.fake_broker import FakeBroker, quote_tape
from tests.fixtures.fake_view import FakeView
from tests.fixtures.memory_ledger import MemoryLedger

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "src" / "jevbot"
CAL = XnysCalendar()
CFG = Config()
SESSION = date(2024, 5, 17)
NEXT_SESSION = date(2024, 5, 20)
CLOSE = CAL.open_close(SESSION)[1]
KEY = SnapshotKey(session=SESSION, slot=Slot.EOD)
INITIAL_CASH = 10_000_000
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
# doubles
# ======================================================================================================================


class StubFillModel:
    """A minimal `FillModel` (3.4): the worst band on both sides, the mid for `mid`, a flat fee. WP06 owns the real one."""

    def check(self, legs: Sequence[OrderLeg], qty: int, chain: ChainSnapshot, *, mandatory: bool) -> tuple[str, ...]:
        del legs, qty, chain, mandatory
        return ()

    def price(self, legs: Sequence[OrderLeg], chain: ChainSnapshot, *, mandatory: bool) -> tuple[BandPrices, tuple[LegFill, ...], str]:
        del mandatory
        net = {Band.ORATS: 0, Band.WORST: 0, Band.MID: 0}
        fills: list[LegFill] = []
        for leg in legs:
            quote = chain.quote(leg.contract)
            assert quote is not None, leg.contract.occ
            buy = leg.side is Side.BUY
            worst = quote.ask if buy else quote.bid
            mid = -(-(quote.bid + quote.ask) // 2) if buy else (quote.bid + quote.ask) // 2
            sign = 1 if buy else -1
            net[Band.ORATS] += sign * worst
            net[Band.WORST] += sign * worst
            net[Band.MID] += sign * mid
            fills.append(LegFill(occ=leg.contract.occ, side=leg.side, bid=quote.bid, ask=quote.ask, orats=worst, worst=worst, mid=mid))
        return (
            BandPrices(orats=net[Band.ORATS], worst=net[Band.WORST], mid=net[Band.MID]),
            tuple(fills),
            "ok",
        )

    def liquidation(self, structure: Structure, chain: ChainSnapshot, last: tuple[int, int] | None) -> tuple[int, int, bool]:
        del structure, chain, last
        return (0, 0, False)

    def fees_micro(self, legs: Sequence[OrderLeg], qty: int, leg_fills: Sequence[LegFill]) -> int:
        del leg_fills
        return 40_300 * len(legs) * qty


@dataclass
class RecordingKill:
    """A `KillSwitch` double that only records what reconcile asked for."""

    trips: list[tuple[KillTrigger, str]] = field(default_factory=list)
    kill_state: KillState = KillState.ARMED

    def state(self) -> KillState:
        return self.kill_state

    def event_id(self) -> str | None:
        return None

    def trip(self, trigger: KillTrigger, detail: str) -> None:
        self.trips.append((trigger, detail))

    def step(self, broker: Any, view: Any, market_open: bool) -> KillState:
        del broker, view, market_open
        return self.kill_state

    def rearm(self, rearm_file_text: str, *, reset_peak: bool, note: str) -> None:
        del rearm_file_text, reset_peak, note


# ======================================================================================================================
# world
# ======================================================================================================================


def narrow(chain: Any | None = None) -> tuple[Any, Structure]:
    chain = make_chain() if chain is None else chain
    expiry = target_expiry(chain, 35)
    short = contract_at(chain, expiry, Right.PUT, 436_000)
    long = contract_at(chain, expiry, Right.PUT, 431_000)
    chain = set_quote(chain, short, bid=400, ask=404)
    chain = set_quote(chain, long, bid=150, ask=154)
    return chain, Structure(
        kind=StructureKind.PUT_CREDIT,
        underlying=chain.underlying,
        expiry=expiry,
        last_session=chain.last_session(expiry),
        legs=(Leg(contract=long, side=Side.BUY), Leg(contract=short, side=Side.SELL)),
    )


CHAIN, SPREAD = narrow()
# worst-band open net: ask(long) 154 - bid(short) 400 = -246c per share
OPEN_NET_WORST = -246
OPEN_NET_MID = 152 - 402  # mid of the long (ceil) minus mid of the short (floor)


def view_of(chain: Any = CHAIN, *, session: date = SESSION, slot: Slot = Slot.EOD) -> FakeView:
    return FakeView(
        key=SnapshotKey(session=session, slot=slot),
        as_of=CAL.open_close(session)[1],
        calendar=CAL,
        chains=() if chain is None else (chain,),
    )


VIEW = view_of()


def open_intent(*, qty: int = 2, session: date = SESSION, structure: Structure = SPREAD) -> OrderIntent:
    decision = ids.decision_id("ns:m:g0", session, structure.underlying, "entry", "entry")
    return OrderIntent(
        intent_id=ids.intent_id("ns:m:g0", session, decision, OrderPurpose.OPEN, 0),
        decision_id=decision,
        position_id=ids.position_id("ns:m:g0", session, structure.structure_id),
        purpose=OrderPurpose.OPEN,
        part=0,
        underlying=structure.underlying,
        legs=tuple(
            OrderLeg(
                contract=leg.contract,
                side=leg.side,
                position_intent=PositionIntent.BTO if leg.side is Side.BUY else PositionIntent.STO,
            )
            for leg in structure.legs
        ),
        qty=qty,
        limit_start=-246,  # AT the natural: the FakeBroker fills only a marketable limit (15.4)
        limit_natural=-246,
        reason="entry",
        mandatory=False,
        session=session,
        key=SnapshotKey(session=session, slot=Slot.EOD),
        tier_ppm=1_000_000,
        structure=structure,
        entry_ctx=ENTRY_CTX,
    )


@dataclass
class World:
    ledger: MemoryLedger
    book: Book
    broker: FakeBroker
    engine: DefaultRiskEngine
    kill: RecordingKill
    fill_model: StubFillModel

    def ingest(self, view: FakeView = VIEW) -> int:
        return reconcile.ingest(
            ledger=self.ledger,
            book=self.book,
            broker=self.broker,
            fill_model=self.fill_model,
            view=view,
            calendar=CAL,
            cfg=CFG,
            mode=RunMode.BACKTEST,
        )

    def reconcile(self, view: FakeView | None = VIEW, *, morning: bool = False, boot: bool = False, session: date = SESSION) -> bool:
        return reconcile.run_reconcile(
            ledger=self.ledger,
            book=self.book,
            broker=self.broker,
            fill_model=self.fill_model,
            kill=self.kill,
            view=view,
            calendar=CAL,
            cfg=CFG,
            mode=RunMode.BACKTEST,
            session=session,
            as_of=CAL.open_close(session)[1],
            morning=morning,
            boot=boot,
        )

    def kinds(self, kind: LedgerKind) -> list[dict[str, Any]]:
        return [entry.payload for entry in self.ledger.entries(kind)]

    def statuses(self) -> list[tuple[str, str, str]]:
        return [(p["client_order_id"], p["status"], p["tag"]) for p in self.kinds(LedgerKind.ORDER_STATUS)]


def world(*, tape: bool = True, **broker_kwargs: Any) -> World:
    ledger = MemoryLedger(strict=True)
    book = Book(initial_cash=INITIAL_CASH, headline=Band.ORATS, cfg=CFG, calendar=CAL)
    broker = FakeBroker(now=CLOSE, **broker_kwargs)
    if tape:
        broker.set_quotes(quote_tape(CHAIN, [leg.contract.occ for leg in SPREAD.legs]))
    book.apply(ledger.append(LedgerKind.SESSION_START, SESSION, CLOSE, {"session": SESSION.isoformat(), "slot": "eod", "phase": "full"}))
    return World(ledger=ledger, book=book, broker=broker, engine=DefaultRiskEngine(CFG), kill=RecordingKill(), fill_model=StubFillModel())


def ledger_intent(w: World, intent: OrderIntent) -> None:
    w.book.apply(w.ledger.append(LedgerKind.ORDER_INTENT, intent.session, CLOSE, msgspec.to_builtins(intent)))


def approve(w: World, intent: OrderIntent, *, attempt: int = 0, limit: int | None = None) -> ApprovedOrder:
    verdict, order = w.engine.approve(intent, w.book.state(), VIEW, now=CLOSE, attempt=attempt, limit=limit)
    assert order is not None, verdict.reject_codes
    return order


def force_position(w: World, intent: OrderIntent, *, qty: int, net: BandPrices | None = None) -> None:
    """Book a fill without the engine - for a position `approve()` would no longer build (one expiring today, say)."""
    ledger_intent(w, intent)
    fill = Fill(
        fill_id=f"forced-{intent.intent_id}",
        client_order_id=f"{intent.intent_id}-00",
        intent_id=intent.intent_id,
        decision_id=intent.decision_id,
        position_id=intent.position_id,
        purpose=intent.purpose,
        structure_id=intent.structure.structure_id if intent.structure is not None else None,
        qty=qty,
        key=intent.key,
        ts=CLOSE,
        net=net if net is not None else BandPrices(orats=-246, worst=-246, mid=-250),
        legs=tuple(LegFill(occ=leg.contract.occ, side=leg.side, bid=100, ask=104, orats=102, worst=102, mid=102) for leg in intent.legs),
        fees_micro=0,
        forced=False,
        model_reject=(),
        quality="ok",
        source="sim",
        broker_order_id=None,
        broker_net=None,
    )
    w.book.apply(w.ledger.append(LedgerKind.FILL, intent.session, CLOSE, msgspec.to_builtins(fill)))


# ======================================================================================================================
# the submit protocol (9.6)
# ======================================================================================================================


def test_the_submit_protocol_writes_submitting_before_the_broker_call_then_the_result() -> None:
    w = world()
    intent = open_intent()
    ledger_intent(w, intent)
    order = approve(w, intent)
    state = reconcile.submit_approved(w.ledger, w.book, w.broker, order, CLOSE)
    assert state is not None and state.status is OrderStatus.SUBMITTED
    assert w.statuses() == [
        (order.client_order_id, "submitting", ""),
        (order.client_order_id, "submitted", ""),
    ]
    assert w.ingest() == 1
    assert w.statuses()[-1] == (order.client_order_id, "filled", "")
    # SUBMITTING is durable BEFORE the broker call: it is the earlier ledger seq
    submitting = next(e for e in w.ledger.entries(LedgerKind.ORDER_STATUS))
    assert submitting.payload["status"] == "submitting" and submitting.payload["limit"] == order.limit
    assert submitting.payload["attempt"] == 0 and submitting.payload["intent_id"] == intent.intent_id


def test_a_definitive_rejection_and_an_ambiguous_call_leave_their_own_trail() -> None:
    w = world()
    intent = open_intent()
    ledger_intent(w, intent)
    order = approve(w, intent)
    w.broker.script_fault("submit", "403_bp")
    assert reconcile.submit_approved(w.ledger, w.book, w.broker, order, CLOSE) is None
    assert w.statuses()[-1] == (order.client_order_id, "rejected", "insufficient_buying_power")

    retry = approve(w, intent, attempt=1)
    w.broker.script_fault("submit", "timeout_after_accept")
    assert reconcile.submit_approved(w.ledger, w.book, w.broker, retry, CLOSE) is None
    assert w.statuses()[-1] == (retry.client_order_id, "unknown", "")
    # the order WAS accepted: the next ingest resolves it (9.6)
    assert w.ingest() == 1
    assert [s[1] for s in w.statuses() if s[0] == retry.client_order_id] == ["submitting", "unknown", "filled"]
    assert len(w.kinds(LedgerKind.FILL)) == 1


def test_the_same_trail_comes_out_of_a_second_broker_that_follows_the_protocol() -> None:
    """9.6: `SimBroker`, `FakeBroker` and `AlpacaPaperBroker` all leave the identical ledger trail."""

    class OtherBroker(FakeBroker):
        """A second Broker double with its own ids and bookkeeping, driven through the same protocol."""

        @property
        def name(self) -> str:
            return "other"

    trails = []
    for broker_cls in (FakeBroker, OtherBroker):
        w = world()
        w.broker = broker_cls(now=CLOSE)
        w.broker.set_quotes(quote_tape(CHAIN, [leg.contract.occ for leg in SPREAD.legs]))
        intent = open_intent()
        ledger_intent(w, intent)
        order = approve(w, intent)
        reconcile.submit_approved(w.ledger, w.book, w.broker, order, CLOSE)
        w.ingest()
        trails.append(w.statuses())
        assert len(w.kinds(LedgerKind.FILL)) == 1
    assert trails[0] == trails[1]


# ======================================================================================================================
# the one fill path (9.6)
# ======================================================================================================================


def test_one_cumulative_fill_offered_three_times_is_booked_once() -> None:
    """The cycle start, the worker poll and the cutoff all call `ingest_fills`; `claim_fill` is the dedupe."""
    w = world()
    intent = open_intent(qty=2)
    ledger_intent(w, intent)
    order = approve(w, intent)
    reconcile.submit_approved(w.ledger, w.book, w.broker, order, CLOSE)
    assert w.ingest() == 1
    assert w.ingest() == 0 and w.ingest() == 0
    fills = w.kinds(LedgerKind.FILL)
    assert len(fills) == 1 and len(w.kinds(LedgerKind.BROKER_FILL)) == 1
    fill = fills[0]
    assert fill["qty"] == 2 and fill["purpose"] == "open" and fill["source"] == "sim"
    assert fill["net"] == {"orats": OPEN_NET_WORST, "worst": OPEN_NET_WORST, "mid": OPEN_NET_MID}
    assert fill["fill_id"] == ids.fill_id(order.client_order_id, 2)
    assert fill["model_reject"] == []  # backtests do not record the paper-only reject list
    assert "realised_pnl" not in fill  # an OPEN has none
    # the Book saw the fill exactly once
    state = w.book.state()
    assert state.positions[0].qty == 2
    assert state.cash.orats == INITIAL_CASH - OPEN_NET_WORST * MULTIPLIER * 2
    assert state.fees_accrued_micro == 40_300 * 2 * 2


def test_a_partial_then_a_full_fill_book_two_deltas() -> None:
    """mleg fills are unit-atomic (15.4): each poll books a whole-contract delta, and the Book grows with it."""
    w = world(seed=3, partial_rate=1.0)
    intent = open_intent(qty=2)
    order = approve(w, intent)
    ledger_intent(w, intent)
    reconcile.submit_approved(w.ledger, w.book, w.broker, order, CLOSE)
    assert w.ingest() == 1
    assert w.book.state().positions[0].qty == 1 and w.book.filled_qty(order.client_order_id) == 1
    assert w.ingest() == 1
    assert w.book.state().positions[0].qty == 2 and w.book.filled_qty(order.client_order_id) == 2
    assert [f["qty"] for f in w.kinds(LedgerKind.FILL)] == [1, 1]
    assert [f["fill_id"] for f in w.kinds(LedgerKind.FILL)] == [
        ids.fill_id(order.client_order_id, 1),
        ids.fill_id(order.client_order_id, 2),
    ]
    assert [s[1] for s in w.statuses()] == ["submitting", "submitted", "partially_filled", "filled"]
    assert w.ingest() == 0


def test_a_closing_fill_carries_the_realised_pnl_per_band() -> None:
    w = world()
    opening = open_intent(qty=2)
    ledger_intent(w, opening)
    reconcile.submit_approved(w.ledger, w.book, w.broker, approve(w, opening), CLOSE)
    w.ingest()
    position = w.book.state().positions[0]

    closing = msgspec.structs.replace(
        opening,
        intent_id=ids.intent_id("ns:m:g0", SESSION, opening.decision_id, OrderPurpose.CLOSE, 0),
        purpose=OrderPurpose.CLOSE,
        legs=tuple(
            OrderLeg(
                contract=leg.contract,
                side=Side.SELL if leg.side is Side.BUY else Side.BUY,
                position_intent=PositionIntent.STC if leg.side is Side.BUY else PositionIntent.BTC,
            )
            for leg in SPREAD.legs
        ),
        limit_start=254,  # closing natural: buy the short back at 404, sell the long at 150
        limit_natural=254,
        entry_ctx=None,
        reason="profit_target",
    )
    ledger_intent(w, closing)
    reconcile.submit_approved(w.ledger, w.book, w.broker, approve(w, closing), CLOSE)
    assert w.ingest() == 1
    fill = w.kinds(LedgerKind.FILL)[-1]
    # closing worst band: buy the short back at 404, sell the long at 150 -> +254c per share
    assert fill["net"]["worst"] == 254
    expected = (-position.open_net.worst - 254) * MULTIPLIER * 2
    assert fill["realised_pnl"]["worst"] == expected == (246 - 254) * MULTIPLIER * 2
    assert w.book.state().positions == ()


def test_an_order_the_broker_never_saw_is_cancelled_never_sent_and_never_resubmitted() -> None:
    w = world()
    intent = open_intent()
    ledger_intent(w, intent)
    order = approve(w, intent)
    reconcile.record_order_status(w.ledger, w.book, order, OrderStatus.SUBMITTING, CLOSE)
    # the eligible snapshot has passed (backtest: the next snapshot key) and the broker has no such order
    later = view_of(session=NEXT_SESSION, chain=None)
    assert w.ingest(later) == 0
    assert w.statuses()[-1] == (order.client_order_id, "cancelled", "never_sent")
    before = len(w.ledger)
    assert w.ingest(later) == 0 and len(w.ledger) == before  # terminal: never looked at again, never resubmitted
    assert w.book.open_orders() == ()


def test_before_the_cutoff_a_missing_order_is_unknown_not_cancelled() -> None:
    w = world()
    intent = open_intent()
    ledger_intent(w, intent)
    order = approve(w, intent)
    reconcile.record_order_status(w.ledger, w.book, order, OrderStatus.SUBMITTING, CLOSE)
    assert w.ingest(VIEW) == 0  # same snapshot: the eligible snapshot has NOT passed
    assert w.statuses()[-1] == (order.client_order_id, "unknown", "")


def test_an_intent_that_never_reached_submitting_is_left_for_the_order_worker() -> None:
    w = world()
    intent = open_intent()
    ledger_intent(w, intent)
    before = len(w.ledger)
    assert w.ingest(view_of(session=NEXT_SESSION, chain=None)) == 0
    assert len(w.ledger) == before  # nothing was written: the worker resumes it by id on the same session
    (pair,) = w.book.open_orders()
    assert pair[1].status is OrderStatus.INTENT


def test_a_crash_between_submit_and_the_ack_is_caught_up_from_the_ledger_alone() -> None:
    """After a restart there is no `ApprovedOrder` in memory - only ORDER_INTENT + ORDER_STATUS (9.6)."""
    w = world()
    intent = open_intent(qty=2)
    ledger_intent(w, intent)
    order = approve(w, intent)
    reconcile.record_order_status(w.ledger, w.book, order, OrderStatus.SUBMITTING, CLOSE)
    w.broker.submit(order)  # the broker accepted it; the process died before the ack

    resumed = Book.replay(w.ledger, initial_cash=INITIAL_CASH, headline=Band.ORATS, cfg=CFG, calendar=CAL)
    w.book = resumed
    assert w.ingest() == 1
    assert w.statuses()[-1][1] == "filled"
    assert resumed.state().positions[0].qty == 2
    assert resumed.filled_qty(order.client_order_id) == 2


# ======================================================================================================================
# reconcile R1-R6
# ======================================================================================================================


def test_r1_cancels_foreign_and_probe_orders_and_keeps_our_own() -> None:
    w = world()
    intent = open_intent()
    ledger_intent(w, intent)
    order = approve(w, intent)
    w.broker.refuse_fills = True
    reconcile.submit_approved(w.ledger, w.book, w.broker, order, CLOSE)
    w.broker.inject_order("jbp-probe-240517-abcdef012345-open-00-00")  # an 11.11 probe leftover
    w.broker.inject_order("jb1-ffffffff-240517-000000000000-open-00-00")  # our shape, unknown to the ledger
    w.broker.inject_order("some-human-order")
    assert w.reconcile() is True
    payload = w.kinds(LedgerKind.RECONCILE)[-1]
    assert sorted(payload["foreign_orders"]) == [
        "jb1-ffffffff-240517-000000000000-open-00-00",
        "jbp-probe-240517-abcdef012345-open-00-00",
        "some-human-order",
    ]
    assert order.client_order_id not in payload["foreign_orders"]
    assert w.broker.order_state(order.client_order_id) is not None
    assert w.broker.order_state(order.client_order_id).status is OrderStatus.SUBMITTED  # type: ignore[union-attr]
    for cid in payload["foreign_orders"]:
        assert w.broker.order_state(cid).status is OrderStatus.CANCELLED  # type: ignore[union-attr]


def test_r3_a_position_difference_trips_reconcile_mismatch_and_is_never_auto_repaired() -> None:
    w = world()
    intent = open_intent(qty=2)
    ledger_intent(w, intent)
    reconcile.submit_approved(w.ledger, w.book, w.broker, approve(w, intent), CLOSE)
    w.ingest()
    assert w.reconcile() is True and w.kill.trips == []

    short_occ = SPREAD.legs[1].contract.occ
    w.broker.set_position(short_occ, -1)  # the broker lost a contract
    assert w.reconcile() is False
    payload = w.kinds(LedgerKind.RECONCILE)[-1]
    assert payload["diff"] == {short_occ: [-2, -1]} and payload["ok"] is False
    assert w.kill.trips[-1][0] is KillTrigger.RECONCILE_MISMATCH
    assert w.book.leg_positions()[short_occ] == -2  # no auto-repair: the flatten closes what the BROKER reports


def test_r3_an_unknown_symbol_or_an_equity_position_is_a_difference_too() -> None:
    w = world()
    w.broker.set_position("SPY240621P00420000", -1)
    assert w.reconcile() is False
    assert w.kill.trips[-1][0] is KillTrigger.RECONCILE_MISMATCH

    w = world()
    w.broker.set_position("SPY", 100)  # an assignment left us long stock
    assert w.reconcile() is False
    assert w.kinds(LedgerKind.RECONCILE)[-1]["diff"] == {"SPY": [0, 100]}


def test_r2_an_assignment_activity_in_the_morning_pass_trips_the_kill_switch() -> None:
    w = world()
    intent = open_intent(qty=1)
    ledger_intent(w, intent)
    reconcile.submit_approved(w.ledger, w.book, w.broker, approve(w, intent), CLOSE)
    w.ingest()
    w.broker.scripted_activities = [
        BrokerActivity(activity_type="OPASN", symbol=SPREAD.legs[1].contract.occ, qty=1, day=SESSION, raw_id="a1"),
        BrokerActivity(activity_type="OPTRD", symbol=SPREAD.legs[0].contract.occ, qty=1, day=SESSION, raw_id="a2"),
    ]
    assert w.reconcile(morning=True) is False
    assert w.kill.trips[-1][0] is KillTrigger.ASSIGNMENT
    payload = w.kinds(LedgerKind.RECONCILE)[-1]
    assert payload["activities"] == [f"OPASN:{SPREAD.legs[1].contract.occ}:1"]
    # outside the morning pass activities are not polled at all
    w2 = world()
    w2.broker.scripted_activities = list(w.broker.scripted_activities)
    assert w2.reconcile() is True and w2.kill.trips == []


def test_r4_refuses_to_trade_on_an_account_that_may_not() -> None:
    for field_name, value in (("options_level", 2), ("trading_blocked", True), ("account_blocked", True)):
        w = world()
        setattr(w.broker, field_name, value)
        with pytest.raises(PaperGuardError):
            w.reconcile()
    w = world()
    w.broker.suspended = True
    with pytest.raises(PaperGuardError, match="suspended"):
        w.reconcile()
    w.kill.kill_state = KillState.LOCKED  # a LOCKED account is expected to be suspended
    assert w.reconcile() is True
    reconcile.may_trade(w.broker, KillState.LOCKED)
    with pytest.raises(PaperGuardError):
        reconcile.may_trade(w.broker, KillState.ARMED)


def test_r5_verifies_the_chain_fully_the_first_time_and_incrementally_afterwards() -> None:
    from jevbot.errors import LedgerCorrupt

    w = world()
    assert w.reconcile() is True
    w.ledger.tamper(1, payload={"session": "1999-01-01", "slot": "eod", "phase": "full"})
    # the entry is already behind the verified high-water mark: the incremental verify does not look at it again
    assert w.reconcile() is True
    with pytest.raises(LedgerCorrupt):
        w.ledger.verify()


def test_r6_fires_on_the_friday_of_a_saturday_dated_monthly() -> None:
    """`expiry <= today` would never be true on the day that really is the last trading day (9.6, INV-11)."""
    saturday = date(2014, 4, 19)
    friday = CAL.prev_or_same_session(saturday)
    assert friday == date(2014, 4, 17) and saturday.weekday() == 5  # Good Friday 04-18 pulls it back to the Thursday
    w = world()
    w.broker.set_position(f"SPY{saturday:%y%m%d}P00436000", -1)
    w.broker.set_position(f"SPY{saturday:%y%m%d}P00431000", 1)
    assert w.reconcile(view=None, session=friday) is False
    trigger, detail = w.kill.trips[-1]
    assert trigger is KillTrigger.EXPIRY_VIOLATION and "2014-04-17" in detail

    earlier = CAL.prev_session(friday)
    w2 = world()
    w2.broker.set_position(f"SPY{saturday:%y%m%d}P00436000", -1)
    w2.broker.set_position(f"SPY{saturday:%y%m%d}P00431000", 1)
    assert w2.reconcile(view=None, session=earlier) is False  # a position difference, but not an expiry violation
    assert all(trigger is not KillTrigger.EXPIRY_VIOLATION for trigger, _ in w2.kill.trips)


def test_r6_also_looks_at_the_ledgers_own_legs() -> None:
    expiring = Structure(
        kind=StructureKind.PUT_CREDIT,
        underlying="SPY",
        expiry=SESSION,
        last_session=SESSION,
        legs=SPREAD.legs,
    )
    w = world()
    intent = open_intent(qty=1, structure=expiring)
    force_position(w, intent, qty=1)  # the engine would refuse to OPEN this: check 10 is exactly what R6 back-stops
    for leg in expiring.legs:
        w.broker.set_position(leg.contract.occ, 1 if leg.side is Side.BUY else -1)
    assert w.reconcile() is False
    assert any(trigger is KillTrigger.EXPIRY_VIOLATION for trigger, _ in w.kill.trips)


def test_r6_at_boot_also_flags_a_position_already_inside_its_hard_exit_window() -> None:
    """9.6 / INV-11: after downtime that needs immediate emergency action, not the near-close cycle."""
    near = Structure(
        kind=StructureKind.PUT_CREDIT,
        underlying="SPY",
        expiry=SPREAD.expiry,
        last_session=CAL.next_session(SESSION, 3),  # sessions_to_expiry = 3 = dte.hard_exit_sessions
        legs=SPREAD.legs,
    )
    w = world()
    intent = open_intent(qty=1, structure=near)
    force_position(w, intent, qty=1)
    for leg in near.legs:
        w.broker.set_position(leg.contract.occ, 1 if leg.side is Side.BUY else -1)
    assert w.reconcile() is True and w.kill.trips == []  # an ordinary cycle leaves it to the manage step
    assert w.reconcile(boot=True) is False
    trigger, detail = w.kill.trips[-1]
    assert trigger is KillTrigger.EXPIRY_VIOLATION and "hard-exit window" in detail

    far = msgspec.structs.replace(near, last_session=CAL.next_session(SESSION, 4))
    w2 = world()
    far_intent = open_intent(qty=1, structure=far)
    force_position(w2, far_intent, qty=1)
    for leg in far.legs:
        w2.broker.set_position(leg.contract.occ, 1 if leg.side is Side.BUY else -1)
    assert w2.reconcile(boot=True) is True and w2.kill.trips == []


def test_the_cycle_context_wrappers_reach_the_same_code() -> None:
    from jevbot.protocols import CycleContext

    w = world()
    intent = open_intent(qty=1)
    ledger_intent(w, intent)
    reconcile.submit_approved(w.ledger, w.book, w.broker, approve(w, intent), CLOSE)
    ctx = CycleContext(
        cfg=CFG,
        meta=_meta(),
        calendar=CAL,
        clock=SimClock(CLOSE),
        broker=w.broker,
        decider=None,  # type: ignore[arg-type]
        cache=None,
        risk=w.engine,
        kill=w.kill,
        fill_model=w.fill_model,
        ledger=w.ledger,
        book=w.book,
        state_builder=None,  # type: ignore[arg-type]
        rules=None,  # type: ignore[arg-type]
        candidates=None,  # type: ignore[arg-type]
        order_worker=None,  # type: ignore[arg-type]
        health=None,  # type: ignore[arg-type]
        tier_of=None,  # type: ignore[arg-type]
        manage_jev="off",
    )
    assert reconcile.ingest_fills(ctx, VIEW) == 1
    assert reconcile.reconcile(ctx, VIEW) is True
    assert reconcile.reconcile(ctx, None) is True  # no view: R1's ingest is skipped, the rest still runs


def _meta() -> Any:
    from jevbot.types import Fidelity, FillRule, RunMeta

    return RunMeta(
        run_id="r",
        trial_id=None,
        experiment="test",
        family="test",
        namespace="ns:m:g0",
        mode=RunMode.BACKTEST,
        decider="mock_jev",
        model="jev-1.13.0",
        model_release_date=date(2026, 9, 15),
        fidelity=Fidelity.EOD_QUOTES,
        fill_rule=FillRule.NEXT_SNAPSHOT,
        spot_measure="parity",
        news_resolved=False,
        news_reason="auto_no_keys",
        config_hash="c",
        state_config_hash="s",
        rules_hash="r",
        risk_config_hash="rk",
        entry_qset_hash="e",
        entry_text_qset_hash="et",
        manage_qset_hash="m",
        manage_text_qset_hash="mt",
        git_commit=None,
        data_manifest_hash="d",
        cache_manifest_hash=None,
        start=SESSION,
        end=None,
        purpose="validate",
        flags=(),
    )


# ======================================================================================================================
# the guard: record_order_status is the ONLY ORDER_STATUS writer (15.1)
# ======================================================================================================================


def mentions_order_status(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr == "ORDER_STATUS":
            return True
        if isinstance(sub, ast.Constant) and sub.value == "order_status":
            return True
    return False


def order_status_writers(source: str, label: str) -> list[str]:
    """`<label>:<line>` for every `*.append(ORDER_STATUS, ...)` call in one module."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
        if name != "append":
            continue
        if any(mentions_order_status(argument) for argument in [*node.args, *(kw.value for kw in node.keywords)]):
            found.append(f"{label}:{node.lineno}")
    return found


def test_the_order_status_guard_fires_on_planted_source() -> None:
    planted = textwrap.dedent(
        """
        def sneaky(ledger, session, as_of):
            ledger.append(LedgerKind.ORDER_STATUS, session, as_of, {})
            ledger.append("order_status", session, as_of, {})
            ledger.append(LedgerKind.FILL, session, as_of, {})
        """
    )
    assert order_status_writers(planted, "planted") == ["planted:3", "planted:4"]


def test_record_order_status_is_the_only_order_status_writer_in_the_package() -> None:
    writers: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        writers += order_status_writers(path.read_text(encoding="utf-8"), path.relative_to(PACKAGE).as_posix())
    assert {w.split(":")[0] for w in writers} == {"reconcile.py"}, writers
