"""`killswitch.DefaultKillSwitch` - the state machine and the flatten driver of DESIGN.md 9.5 (INV-21, G6 / D17).

The drills of section 15.1: persistence before action (with a crash injected between the file write and the first broker
call), a flatten driven from the BROKER's positions including an assigned equity position, shorts first in the per-leg
fallback, kill orders exempt from the rate and attempt caps, suspend only after a verified flat, NOT_FLAT retries that never
suspend and never exit, the market-closed branch with its urgent and normal delays, re-arm gating and the peak reset, and
the backtest cooldown.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import msgspec
import pytest

from jevbot import ids
from jevbot.cal import SimClock, XnysCalendar
from jevbot.config import Config
from jevbot.errors import InvariantError
from jevbot.killswitch import KILL_FILE, REARM_FILE, DefaultKillSwitch
from jevbot.portfolio import Book
from jevbot.risk import DefaultRiskEngine
from jevbot.types import (
    Band,
    BandPrices,
    ChainSnapshot,
    EntryContext,
    Fidelity,
    Fill,
    FillRule,
    KillState,
    KillTrigger,
    LedgerKind,
    Leg,
    LegFill,
    OrderIntent,
    OrderLeg,
    OrderPurpose,
    PositionIntent,
    Right,
    RunMeta,
    RunMode,
    Side,
    Slot,
    SnapshotKey,
    Structure,
    StructureKind,
)
from tests.fixtures.chain_factory import contract_at, make_chain, make_structure, target_expiry
from tests.fixtures.fake_broker import FakeBroker, quote_tape
from tests.fixtures.fake_view import FakeView
from tests.fixtures.memory_ledger import MemoryLedger

CAL = XnysCalendar()
CFG = Config()
SESSION = date(2024, 5, 17)
CLOSE = CAL.open_close(SESSION)[1]
KEY = SnapshotKey(session=SESSION, slot=Slot.EOD)
CHAIN = make_chain()
EXPIRY = target_expiry(CHAIN, 35)
INITIAL_CASH = 10_000_000
NAMESPACE = "test:jev-1.13.0:g0"

ENTRY_CTX = EntryContext(
    entry_thesis="t",
    entry_codes={"trend": "up", "iv_vs_realized": "rich", "iv_rank": "p60_p80"},
    entry_spot=45_000,
    entry_iv30_bp=1800,
    entry_em_hold_tenths=40,
    open_mid_at_decision=-250,
)


class StubFillModel:
    """A minimal `FillModel` (3.4): the worst band on both sides, the mid for `mid`, no fees. WP06 owns the real one."""

    def check(self, legs: Sequence[OrderLeg], qty: int, chain: ChainSnapshot, *, mandatory: bool) -> tuple[str, ...]:
        del legs, qty, chain, mandatory
        return ()

    def price(self, legs: Sequence[OrderLeg], chain: ChainSnapshot, *, mandatory: bool) -> tuple[BandPrices, tuple[LegFill, ...], str]:
        del mandatory
        worst = mid = 0
        fills: list[LegFill] = []
        for leg in legs:
            quote = chain.quote(leg.contract)
            assert quote is not None, leg.contract.occ
            buy = leg.side is Side.BUY
            price = quote.ask if buy else quote.bid
            half = -(-(quote.bid + quote.ask) // 2) if buy else (quote.bid + quote.ask) // 2
            worst += price if buy else -price
            mid += half if buy else -half
            fills.append(LegFill(occ=leg.contract.occ, side=leg.side, bid=quote.bid, ask=quote.ask, orats=price, worst=price, mid=half))
        return (BandPrices(orats=worst, worst=worst, mid=mid), tuple(fills), "ok")

    def liquidation(self, structure: Structure, chain: ChainSnapshot, last: tuple[int, int] | None) -> tuple[int, int, bool]:
        del structure, chain, last
        return (0, 0, False)

    def fees_micro(self, legs: Sequence[OrderLeg], qty: int, leg_fills: Sequence[LegFill]) -> int:
        del legs, qty, leg_fills
        return 0


def meta(*, mode: RunMode = RunMode.PAPER) -> RunMeta:
    return RunMeta(
        run_id="r",
        trial_id=None,
        experiment="test",
        family="test",
        namespace=NAMESPACE,
        mode=mode,
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
        purpose="paper",
        flags=(),
    )


def spread(short_milli: int, long_milli: int) -> Structure:
    short = contract_at(CHAIN, EXPIRY, Right.PUT, short_milli)
    long = contract_at(CHAIN, EXPIRY, Right.PUT, long_milli)
    return Structure(
        kind=StructureKind.PUT_CREDIT,
        underlying="SPY",
        expiry=EXPIRY,
        last_session=CHAIN.last_session(EXPIRY),
        legs=(Leg(contract=long, side=Side.BUY), Leg(contract=short, side=Side.SELL)),
    )


SPREADS = tuple(spread(436_000 - 2_000 * i, 431_000 - 2_000 * i) for i in range(6))
CONDOR = make_structure(CHAIN, StructureKind.IRON_CONDOR)
ALL_LEGS = tuple({leg.contract.occ for s in (*SPREADS, CONDOR) for leg in s.legs})


@dataclass
class World:
    ledger: MemoryLedger
    book: Book
    broker: FakeBroker
    clock: SimClock
    risk: DefaultRiskEngine
    kill: DefaultKillSwitch
    state_dir: Path
    alerts: list[str]
    view: FakeView

    def rebuild_kill(self, *, mode: RunMode = RunMode.PAPER) -> DefaultKillSwitch:
        """What a restart does: replay the book, construct the switch again over the same ledger and state directory."""
        self.book = Book.replay(self.ledger, initial_cash=INITIAL_CASH, headline=Band.ORATS, cfg=CFG, calendar=CAL)
        self.kill = make_switch(self, mode=mode)
        return self.kill


def make_switch(w: World, *, mode: RunMode = RunMode.PAPER, cfg: Config = CFG) -> DefaultKillSwitch:
    return DefaultKillSwitch(
        cfg=cfg,
        meta=meta(mode=mode),
        ledger=w.ledger,
        book=w.book,
        risk=w.risk,
        clock=w.clock,
        fill_model=StubFillModel(),
        calendar=CAL,
        state_dir=w.state_dir,
        broker=w.broker,
        sleep=lambda _seconds: None,
        alert=w.alerts.append,
    )


def world(tmp_path: Path, *, mode: RunMode = RunMode.PAPER, cfg: Config = CFG, **broker_kwargs: Any) -> World:
    ledger = MemoryLedger(strict=True)
    book = Book(initial_cash=INITIAL_CASH, headline=Band.ORATS, cfg=cfg, calendar=CAL)
    broker = FakeBroker(now=CLOSE, **broker_kwargs)
    broker.set_quotes(quote_tape(CHAIN, ALL_LEGS))
    clock = SimClock(CLOSE)
    book.apply(ledger.append(LedgerKind.SESSION_START, SESSION, CLOSE, {"session": SESSION.isoformat(), "slot": "eod", "phase": "full"}))
    w = World(
        ledger=ledger,
        book=book,
        broker=broker,
        clock=clock,
        risk=DefaultRiskEngine(cfg, meta(mode=mode)),
        kill=None,  # type: ignore[arg-type]
        state_dir=tmp_path / "state",
        alerts=[],
        view=FakeView(key=KEY, as_of=CLOSE, calendar=CAL, chains=(CHAIN,)),
    )
    w.kill = make_switch(w, mode=mode, cfg=cfg)
    return w


def hold(w: World, structure: Structure, *, index: int, qty: int = 1) -> None:
    """Give the Book a position and the BROKER the matching legs (the flatten reads the broker, 9.5 K3)."""
    decision = ids.decision_id(NAMESPACE, SESSION, structure.underlying, "entry", f"entry{index}")
    intent = OrderIntent(
        intent_id=ids.intent_id(NAMESPACE, SESSION, decision, OrderPurpose.OPEN, 0),
        decision_id=decision,
        position_id=ids.position_id(NAMESPACE, SESSION, structure.structure_id),
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
        limit_start=-200,
        limit_natural=-200,
        reason="entry",
        mandatory=False,
        session=SESSION,
        key=KEY,
        tier_ppm=1_000_000,
        structure=structure,
        entry_ctx=ENTRY_CTX,
    )
    w.book.apply(w.ledger.append(LedgerKind.ORDER_INTENT, SESSION, CLOSE, msgspec.to_builtins(intent)))
    fill = Fill(
        fill_id=f"open-{index}",
        client_order_id=f"{intent.intent_id}-00",
        intent_id=intent.intent_id,
        decision_id=decision,
        position_id=intent.position_id,
        purpose=OrderPurpose.OPEN,
        structure_id=structure.structure_id,
        qty=qty,
        key=KEY,
        ts=CLOSE,
        net=BandPrices(orats=-200, worst=-200, mid=-205),
        legs=tuple(
            LegFill(occ=leg.contract.occ, side=leg.side, bid=100, ask=104, orats=102, worst=102, mid=102) for leg in intent.legs
        ),
        fees_micro=0,
        forced=False,
        model_reject=(),
        quality="ok",
        source="paper",
        broker_order_id=None,
        broker_net=None,
    )
    w.book.apply(w.ledger.append(LedgerKind.FILL, SESSION, CLOSE, msgspec.to_builtins(fill)))
    for leg in structure.legs:
        occ = leg.contract.occ
        signed = qty * (1 if leg.side is Side.BUY else -1)
        w.broker.set_position(occ, w.broker.positions_by_symbol.get(occ, 0) + signed)


def kill_steps(w: World) -> list[str]:
    return [entry.payload["step"] for entry in w.ledger.entries(LedgerKind.KILL)]


def kill_intents(w: World) -> list[OrderIntent]:
    return [
        msgspec.convert(entry.payload, OrderIntent)
        for entry in w.ledger.entries(LedgerKind.ORDER_INTENT)
        if entry.payload["purpose"] == "kill"
    ]


# ======================================================================================================================
# persistence before any action (9.5)
# ======================================================================================================================


def test_trip_is_durable_before_a_single_broker_call_and_is_idempotent(tmp_path: Path) -> None:
    w = world(tmp_path)
    hold(w, SPREADS[0], index=0)
    w.broker.calls.clear()
    assert w.kill.state() is KillState.ARMED and w.kill.event_id() is None

    w.kill.trip(KillTrigger.DRAWDOWN, "peak-to-trough 8.1%")
    assert w.broker.calls == []  # the file and the ledger entry come BEFORE any action
    assert w.kill.state() is KillState.TRIPPED and w.kill.event_id() is not None
    stored = json.loads((w.state_dir / KILL_FILE).read_text(encoding="utf-8"))
    assert stored["event_id"] == w.kill.event_id() and stored["trigger"] == "drawdown"
    assert stored["detail"] == "peak-to-trough 8.1%" and stored["ledger_head"]
    assert kill_steps(w) == ["tripped"]
    assert w.book.state().kill_state is KillState.TRIPPED and w.book.state().kill_event_id == w.kill.event_id()

    first = w.kill.event_id()
    w.kill.trip(KillTrigger.OPERATOR, "second thoughts")  # sticky and idempotent
    assert w.kill.event_id() == first and kill_steps(w) == ["tripped"]


def test_the_event_id_is_deterministic_and_carries_no_wall_clock(tmp_path: Path) -> None:
    ids_seen = set()
    for _ in range(2):
        w = world(tmp_path / str(len(ids_seen)))
        hold(w, SPREADS[0], index=0)
        w.kill.trip(KillTrigger.DRAWDOWN, "d")
        ids_seen.add(w.kill.event_id())
        payload = next(w.ledger.entries(LedgerKind.KILL)).payload
        assert set(payload) == {"event_id", "step", "trigger", "detail"}  # no `ts` in the hashed payload (INV-24)
    assert len(ids_seen) == 1


def test_a_crash_between_the_file_write_and_the_first_broker_call_resumes_in_kill_mode(tmp_path: Path) -> None:
    w = world(tmp_path)
    hold(w, SPREADS[0], index=0)
    w.kill.trip(KillTrigger.OPERATOR, "state/KILL")
    event = w.kill.event_id()

    resumed = w.rebuild_kill()  # the process died here; a restart replays the ledger and re-reads the file
    assert resumed.state() is KillState.TRIPPED and resumed.event_id() == event
    assert resumed.step(w.broker, w.view, market_open=True) is KillState.LOCKED
    assert w.broker.positions() == ()


def test_a_kill_file_without_a_ledger_entry_still_starts_the_process_in_kill_mode(tmp_path: Path) -> None:
    """9.5 startup rule: if EITHER the file or the ledger replay says tripped, the process starts in kill mode."""
    w = world(tmp_path)
    w.state_dir.mkdir(parents=True, exist_ok=True)
    (w.state_dir / KILL_FILE).write_text(
        json.dumps({"event_id": "ev-manual", "trigger": "operator", "detail": "by hand", "ledger_head": "x"}), encoding="utf-8"
    )
    resumed = w.rebuild_kill()
    assert resumed.state() is KillState.TRIPPED and resumed.event_id() == "ev-manual"


# ======================================================================================================================
# the flatten sequence (9.5 K1-K7)
# ======================================================================================================================


def test_the_flatten_cancels_closes_verifies_flat_and_suspends_strictly_last(tmp_path: Path) -> None:
    w = world(tmp_path)
    hold(w, SPREADS[0], index=0)
    hold(w, SPREADS[1], index=1)
    resting = w.broker  # an order left resting before the trip must be cancelled by K2
    w.kill.trip(KillTrigger.RECONCILE_MISMATCH, "positions differ")
    assert w.kill.step(resting, w.view, market_open=True) is KillState.LOCKED
    assert kill_steps(w) == [
        "tripped",
        "cancelled",
        "close_submitted",
        "close_submitted",
        "flat_verified",
        "suspended",
        "locked",
    ]
    assert w.broker.positions() == () and w.broker.open_orders() == ()
    assert w.broker.suspended is True
    # set_suspended comes AFTER the flat verification (K6: suspend_trade blocks closing orders too)
    assert w.broker.calls.index("set_suspended:True") > max(i for i, c in enumerate(w.broker.calls) if c.startswith("submit:"))
    assert w.book.state().kill_state is KillState.LOCKED and w.book.state().positions == ()
    # every closer went through the 9.6 submit protocol and the ONE fill path
    assert len(list(w.ledger.entries(LedgerKind.FILL))) == 4  # two opens + two kill closes
    for intent in kill_intents(w):
        assert intent.purpose is OrderPurpose.KILL and intent.mandatory and intent.part == 0
        assert intent.decision_id == ids.decision_id(NAMESPACE, SESSION, "SPY", "manage", ids.kill_subject(intent.position_id))


def test_structures_with_short_legs_go_first_and_nearest_expiry_first(tmp_path: Path) -> None:
    w = world(tmp_path)
    near = Structure(
        kind=StructureKind.LONG_CALL,
        underlying="SPY",
        expiry=EXPIRY,
        last_session=CHAIN.last_session(EXPIRY),
        legs=(Leg(contract=contract_at(CHAIN, EXPIRY, Right.CALL, 460_000), side=Side.BUY),),
    )
    hold(w, near, index=0)  # no short leg
    hold(w, SPREADS[0], index=1)  # has a short leg
    w.kill.trip(KillTrigger.OPERATOR, "order")
    w.kill.step(w.broker, w.view, market_open=True)
    submitted = [c for c in w.broker.calls if c.startswith("submit:")]
    order_of = {intent.position_id: index for index, intent in enumerate(kill_intents(w))}
    short_first = ids.position_id(NAMESPACE, SESSION, SPREADS[0].structure_id)
    long_only = ids.position_id(NAMESPACE, SESSION, near.structure_id)
    assert order_of[short_first] < order_of[long_only]
    assert len(submitted) == 2


def test_an_unmatched_broker_leg_forms_its_own_single_leg_group(tmp_path: Path) -> None:
    w = world(tmp_path)
    stray = contract_at(CHAIN, EXPIRY, Right.CALL, 470_000)
    w.broker.set_position(stray.occ, -2)  # the ledger never knew about it
    w.kill.trip(KillTrigger.RECONCILE_MISMATCH, "unknown leg")
    assert w.kill.step(w.broker, w.view, market_open=True) is KillState.LOCKED
    (intent,) = kill_intents(w)
    assert intent.structure is None and intent.qty == 2
    assert intent.legs[0].side is Side.BUY and intent.legs[0].position_intent is PositionIntent.BTC
    assert intent.limit_start > 0  # buying a short leg back is a debit
    assert w.broker.positions() == ()


def test_the_per_leg_fallback_closes_short_legs_first_in_parts_1_to_4(tmp_path: Path) -> None:
    """9.5 K4: legs separately, SHORT legs first (buy_to_close), then the longs, parts 1..4."""
    w = world(tmp_path, max_mleg_legs=2)  # the broker will not trade a four-leg structure in one order
    hold(w, CONDOR, index=0)
    w.kill.trip(KillTrigger.OPERATOR, "fallback")
    assert w.kill.step(w.broker, w.view, market_open=True) is KillState.LOCKED
    assert "fallback_legs" in kill_steps(w)
    fallback = [i for i in kill_intents(w) if i.part > 0]
    assert [i.part for i in fallback] == [1, 2, 3, 4]
    assert [i.legs[0].position_intent.value for i in fallback] == [
        "buy_to_close",
        "buy_to_close",
        "sell_to_close",
        "sell_to_close",
    ]
    assert all(i.structure is None and i.qty == 1 for i in fallback)
    assert w.broker.positions() == ()


def test_kill_orders_are_exempt_from_the_rate_and_attempt_caps(tmp_path: Path) -> None:
    """9.5 / 9.1 check 21: six structures plus a four-leg fallback complete although the rate cap is long past."""
    w = world(tmp_path, max_mleg_legs=2)
    for index, structure in enumerate(SPREADS):
        hold(w, structure, index=index)
    hold(w, CONDOR, index=len(SPREADS))
    w.kill.trip(KillTrigger.DRAWDOWN, "cap")
    assert w.kill.step(w.broker, w.view, market_open=True) is KillState.LOCKED
    assert w.broker.positions() == ()
    submissions = w.book.state().orders_last_minute
    assert submissions > CFG.risk.max_orders_per_minute, submissions
    verdicts = [entry.payload for entry in w.ledger.entries(LedgerKind.RISK_VERDICT)]
    assert verdicts and all(v["approved"] for v in verdicts)


def test_an_assigned_equity_position_is_flattened_with_a_market_order(tmp_path: Path) -> None:
    w = world(tmp_path)
    hold(w, SPREADS[0], index=0)
    w.broker.set_position("SPY", 100)  # an assignment left us long stock
    w.kill.trip(KillTrigger.ASSIGNMENT, "OPASN")
    assert w.kill.step(w.broker, w.view, market_open=True) is KillState.LOCKED
    (equity_intent,) = [i for i in kill_intents(w) if i.equity_symbol is not None]
    assert equity_intent.equity_symbol == "SPY" and equity_intent.equity_side is Side.SELL
    assert equity_intent.equity_qty == 100 and equity_intent.qty == 0 and equity_intent.legs == ()
    assert equity_intent.position_id == ids.equity_position_id("SPY")
    assert equity_intent.decision_id == ids.decision_id(NAMESPACE, SESSION, "SPY", "manage", ids.equity_kill_subject("SPY"))
    submitted = [e.payload for e in w.ledger.entries(LedgerKind.ORDER_STATUS) if e.payload["intent_id"] == equity_intent.intent_id]
    assert submitted[0]["limit"] is None  # a market order (9.5 K4)
    assert w.broker.positions_by_symbol.get("SPY", 0) == 0

    short = world(tmp_path / "short")
    short.broker.set_position("SPY", -50)  # covering a SHORT share position buys
    short.kill.trip(KillTrigger.ASSIGNMENT, "OPASN")
    assert short.kill.step(short.broker, short.view, market_open=True) is KillState.LOCKED
    (covering,) = [i for i in kill_intents(short) if i.equity_symbol is not None]
    assert covering.equity_side is Side.BUY and covering.equity_qty == 50


def test_a_flatten_that_cannot_fill_ends_not_flat_never_suspends_and_retries(tmp_path: Path) -> None:
    w = world(tmp_path, refuse_fills=True)
    hold(w, SPREADS[0], index=0)
    w.kill.trip(KillTrigger.OPERATOR, "stuck")
    assert w.kill.step(w.broker, w.view, market_open=True) is KillState.NOT_FLAT
    assert w.broker.suspended is False and "set_suspended:True" not in w.broker.calls
    assert kill_steps(w)[-1] == "not_flat"
    assert w.alerts and "NOT FLAT" in w.alerts[-1]
    assert w.book.state().kill_state is KillState.NOT_FLAT

    calls = len(w.broker.calls)
    assert w.kill.step(w.broker, w.view, market_open=True) is KillState.NOT_FLAT  # inside kill.not_flat_retry_s: no retry
    assert len(w.broker.calls) == calls

    w.clock.set(CLOSE + timedelta(seconds=CFG.kill.not_flat_retry_s))
    w.broker.refuse_fills = False
    assert w.kill.step(w.broker, w.view, market_open=True) is KillState.LOCKED
    assert w.broker.positions() == () and w.broker.suspended is True


def test_a_crash_mid_flatten_resumes_where_it_stopped(tmp_path: Path) -> None:
    w = world(tmp_path, refuse_fills=True)
    hold(w, SPREADS[0], index=0)
    hold(w, SPREADS[1], index=1)
    w.kill.trip(KillTrigger.DRAWDOWN, "mid-flatten")
    assert w.kill.step(w.broker, w.view, market_open=True) is KillState.NOT_FLAT

    resumed = w.rebuild_kill()  # restart: the book is replayed, the switch reads the ledger and the file
    assert resumed.state() is KillState.NOT_FLAT and resumed.event_id() == w.kill.event_id()
    w.broker.refuse_fills = False
    w.clock.set(CLOSE + timedelta(seconds=CFG.kill.not_flat_retry_s))
    assert resumed.step(w.broker, w.view, market_open=True) is KillState.LOCKED
    assert w.broker.positions() == () and w.book.state().positions == ()


def test_a_trip_while_the_market_is_closed_cancels_now_and_defers_the_flatten(tmp_path: Path) -> None:
    w = world(tmp_path)
    hold(w, SPREADS[0], index=0)
    w.kill.trip(KillTrigger.DRAWDOWN, "after hours")
    assert w.kill.step(w.broker, w.view, market_open=False) is KillState.TRIPPED
    assert kill_steps(w) == ["tripped", "cancelled"]  # K1-K2 now, K3 later
    assert w.broker.positions() != () and not w.broker.suspended

    next_open = CAL.next_open_after(CLOSE)
    w.clock.set(next_open + timedelta(minutes=CFG.kill.post_open_delay_min - 1))
    assert w.kill.step(w.broker, w.view, market_open=True) is KillState.TRIPPED  # the first minutes have the widest quotes
    w.clock.set(next_open + timedelta(minutes=CFG.kill.post_open_delay_min))
    assert w.kill.step(w.broker, w.view, market_open=True) is KillState.LOCKED


@pytest.mark.parametrize("trigger", [KillTrigger.EXPIRY_VIOLATION, KillTrigger.ASSIGNMENT])
def test_an_urgent_trigger_uses_the_short_post_open_delay(tmp_path: Path, trigger: KillTrigger) -> None:
    w = world(tmp_path)
    hold(w, SPREADS[0], index=0)
    w.kill.trip(trigger, "urgent")
    assert w.kill.step(w.broker, w.view, market_open=False) is KillState.TRIPPED
    next_open = CAL.next_open_after(CLOSE)
    w.clock.set(next_open + timedelta(minutes=CFG.kill.post_open_delay_urgent_min))
    assert w.kill.step(w.broker, w.view, market_open=True) is KillState.LOCKED


def test_a_position_inside_its_hard_exit_window_is_urgent_too(tmp_path: Path) -> None:
    w = world(tmp_path)
    expiring = Structure(
        kind=StructureKind.PUT_CREDIT,
        underlying="SPY",
        expiry=EXPIRY,
        last_session=CAL.next_session(SESSION, 2),  # sessions_to_expiry = 2 <= dte.hard_exit_sessions
        legs=SPREADS[0].legs,
    )
    hold(w, expiring, index=0)
    w.kill.trip(KillTrigger.DRAWDOWN, "urgent by position")
    assert w.kill.step(w.broker, w.view, market_open=False) is KillState.TRIPPED
    next_open = CAL.next_open_after(CLOSE)
    w.clock.set(next_open + timedelta(minutes=CFG.kill.post_open_delay_urgent_min))
    assert w.kill.step(w.broker, w.view, market_open=True) is KillState.LOCKED


def test_without_a_view_only_k1_and_k2_can_run(tmp_path: Path) -> None:
    """K3 prices from a fresh snapshot; without one the sequence waits instead of guessing a limit."""
    w = world(tmp_path)
    hold(w, SPREADS[0], index=0)
    w.kill.trip(KillTrigger.OPERATOR, "no snapshot")
    assert w.kill.step(w.broker, None, market_open=True) is KillState.TRIPPED
    assert kill_steps(w) == ["tripped", "cancelled"]
    assert w.kill.step(w.broker, w.view, market_open=True) is KillState.LOCKED


# ======================================================================================================================
# re-arm (9.5)
# ======================================================================================================================


def flatten(w: World) -> None:
    w.kill.trip(KillTrigger.DRAWDOWN, "peak-to-trough")
    assert w.kill.step(w.broker, w.view, market_open=True) is KillState.LOCKED


def test_rearm_needs_the_hand_made_file_with_the_event_id_and_a_flat_broker(tmp_path: Path) -> None:
    w = world(tmp_path)
    hold(w, SPREADS[0], index=0)
    flatten(w)
    event = w.kill.event_id()
    assert event is not None
    with pytest.raises(InvariantError, match=REARM_FILE):
        w.kill.rearm("not-the-event-id", reset_peak=False, note="n")
    assert w.kill.state() is KillState.LOCKED

    w.broker.set_position(SPREADS[1].legs[1].contract.occ, -1)  # not flat any more
    with pytest.raises(InvariantError, match="not flat"):
        w.kill.rearm(event, reset_peak=False, note="n")
    w.broker.set_position(SPREADS[1].legs[1].contract.occ, 0)

    (w.state_dir / REARM_FILE).write_text(f"{event}\n", encoding="utf-8")
    w.kill.rearm(f"{event}\n", reset_peak=False, note="checked by hand")
    assert w.kill.state() is KillState.ARMED and w.kill.event_id() is None
    assert w.broker.suspended is False
    assert not (w.state_dir / KILL_FILE).exists() and not (w.state_dir / REARM_FILE).exists()
    payload = next(w.ledger.entries(LedgerKind.REARM)).payload
    assert payload == {
        "event_id": event,
        "reset_peak": False,
        "operator_note": "checked by hand",
        "ledger_head_at_rearm": payload["ledger_head_at_rearm"],
    }
    assert w.book.state().kill_state is KillState.ARMED


def test_without_reset_peak_a_drawdown_kill_re_trips_at_once(tmp_path: Path) -> None:
    w = world(tmp_path)
    hold(w, SPREADS[0], index=0)
    # mark the book down past the 8% drawdown limit, then flatten
    w.book.apply(
        w.ledger.append(
            LedgerKind.MARK,
            SESSION,
            CLOSE,
            {
                "equity": msgspec.to_builtins(BandPrices(orats=0, worst=0, mid=0)),
                "cash": msgspec.to_builtins(BandPrices(orats=0, worst=0, mid=0)),
                "open_max_loss": 0,
                "bp_used": 0,
                "bp_utilisation_ppm": 0,
                "positions": {ids.position_id(NAMESPACE, SESSION, SPREADS[0].structure_id): {"liq_value": 9_000, "mid_value": 9_000, "stale": False}},
                "net_delta_milli": 0,
                "net_vega_milli": 0,
            },
        )
    )
    peak = w.book.state().peak_equity
    assert w.risk.on_mark(w.book.state())[0][0] is KillTrigger.DRAWDOWN
    flatten(w)
    event = w.kill.event_id()
    assert event is not None

    w.kill.rearm(event, reset_peak=False, note="no reset")
    assert w.book.state().peak_equity == peak
    assert w.risk.on_mark(w.book.state()) and w.risk.on_mark(w.book.state())[0][0] is KillTrigger.DRAWDOWN

    w.kill.trip(KillTrigger.DRAWDOWN, "again")
    again = w.kill.event_id()
    assert again is not None and again != event
    w.kill.step(w.broker, w.view, market_open=True)
    w.kill.rearm(again, reset_peak=True, note="reset")
    assert w.book.state().peak_equity == w.book.state().equity.orats
    assert w.risk.on_mark(w.book.state()) == ()


# ======================================================================================================================
# backtest semantics (9.5)
# ======================================================================================================================


def test_a_backtest_re_arms_itself_after_the_cooldown_with_the_peak_reset(tmp_path: Path) -> None:
    w = world(tmp_path, mode=RunMode.BACKTEST)
    hold(w, SPREADS[0], index=0)
    flatten(w)
    cooldown_end = CAL.next_session(SESSION, CFG.kill.backtest_cooldown_sessions)

    early = FakeView(
        key=SnapshotKey(session=CAL.next_session(SESSION, 5), slot=Slot.EOD),
        as_of=CAL.open_close(CAL.next_session(SESSION, 5))[1],
        calendar=CAL,
        chains=(CHAIN,),
    )
    assert w.kill.step(w.broker, early, market_open=True) is KillState.LOCKED

    late = FakeView(key=SnapshotKey(session=cooldown_end, slot=Slot.EOD), as_of=CAL.open_close(cooldown_end)[1], calendar=CAL, chains=(CHAIN,))
    assert w.kill.step(w.broker, late, market_open=True) is KillState.ARMED
    assert w.broker.suspended is False
    payload = next(w.ledger.entries(LedgerKind.REARM)).payload
    assert payload["reset_peak"] is True and "cooldown" in payload["operator_note"]
    assert w.book.state().kill_state is KillState.ARMED


def test_stop_run_ends_the_run_instead_of_cooling_down(tmp_path: Path) -> None:
    cfg = msgspec.structs.replace(CFG, kill=msgspec.structs.replace(CFG.kill, backtest_behaviour="stop_run"))
    w = world(tmp_path, mode=RunMode.BACKTEST, cfg=cfg)
    hold(w, SPREADS[0], index=0)
    flatten(w)
    assert w.kill.run_stopped is True
    late = CAL.next_session(SESSION, 30)
    view = FakeView(key=SnapshotKey(session=late, slot=Slot.EOD), as_of=CAL.open_close(late)[1], calendar=CAL, chains=(CHAIN,))
    assert w.kill.step(w.broker, view, market_open=True) is KillState.LOCKED  # never re-arms itself


def test_a_paper_service_never_re_arms_itself(tmp_path: Path) -> None:
    w = world(tmp_path)
    hold(w, SPREADS[0], index=0)
    flatten(w)
    late = CAL.next_session(SESSION, 60)
    view = FakeView(key=SnapshotKey(session=late, slot=Slot.EOD), as_of=CAL.open_close(late)[1], calendar=CAL, chains=(CHAIN,))
    for _ in range(3):
        assert w.kill.step(w.broker, view, market_open=True) is KillState.LOCKED
    assert list(w.ledger.entries(LedgerKind.REARM)) == []


def test_an_armed_switch_does_nothing_and_a_missing_broker_is_an_invariant_error(tmp_path: Path) -> None:
    w = world(tmp_path)
    assert w.kill.step(w.broker, w.view, market_open=True) is KillState.ARMED
    assert list(w.ledger.entries(LedgerKind.KILL)) == []
    lonely = DefaultKillSwitch(
        cfg=CFG,
        meta=meta(),
        ledger=w.ledger,
        book=w.book,
        risk=w.risk,
        clock=w.clock,
        fill_model=StubFillModel(),
        calendar=CAL,
        state_dir=w.state_dir,
        sleep=lambda _s: None,
    )
    with pytest.raises(InvariantError, match="no broker"):
        lonely.rearm("x", reset_peak=False, note="n")


def test_rearm_is_refused_while_the_switch_is_armed(tmp_path: Path) -> None:
    w = world(tmp_path)
    with pytest.raises(InvariantError):
        w.kill.rearm("anything", reset_peak=False, note="n")
