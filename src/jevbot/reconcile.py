"""The ONE fill-ingestion path, the ONE `ORDER_STATUS` writer and reconcile R1-R6 (DESIGN.md 9.6, INV-10).

`record_order_status()` is the only function in the package that appends an `ORDER_STATUS` entry. Every caller that submits -
`backtest.sim_order_worker`, `paper.runner.paper_order_worker` and the kill-switch flatten driver - follows the same protocol
(`submit_approved()` below implements it once), so `SimBroker`, `FakeBroker` and `AlpacaPaperBroker` leave the identical
ledger trail and the `never_sent` branch does not depend on which broker is plugged in:

    record_order_status(..., SUBMITTING)          # durable BEFORE any broker call
    try:    st = broker.submit(order)             # idempotent on client_order_id; adopts an existing order
    except BrokerRejected as e:   record_order_status(..., REJECTED, tag=e.tag)     # definitive; never retried
    except BrokerAmbiguous:       record_order_status(..., UNKNOWN)                 # resolved later by ingest_fills / R1
    else:                         record_order_status(..., st.status, state=st)

`ingest_fills()` is the only function that turns broker order states into FILL entries, in backtest and paper alike; the
`Ledger.claim_fill` unique key makes a second route (cycle start, worker poll, cutoff) a no-op instead of a double booking.

A `Broker` is ledger-free (3.4): it has no Ledger and no Book and never writes an entry. Nothing here commits - the Ledger's
own commit mode does (paper: per append and fsynced; backtest: once per session).

`ingest_fills(ctx, view)` and `reconcile(ctx, view)` are the `CycleContext`-facing wrappers of 9.6; `ingest()` and
`run_reconcile()` take the same collaborators explicitly, because the kill switch drives them without a CycleContext.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Final
from weakref import WeakKeyDictionary

import msgspec

from jevbot import ids, occ
from jevbot.config import Config
from jevbot.errors import BrokerAmbiguous, BrokerError, BrokerRejected, InvariantError, PaperGuardError
from jevbot.structmath import MULTIPLIER
from jevbot.types import (
    ApprovedOrder,
    BandPrices,
    BrokerActivity,
    Fill,
    KillState,
    KillTrigger,
    LedgerEntry,
    LedgerKind,
    OrderIntent,
    OrderPurpose,
    OrderState,
    OrderStatus,
    Position,
    RunMode,
    Slot,
    SnapshotKey,
)

if TYPE_CHECKING:
    from jevbot.protocols import BookP, Broker, Calendar, CycleContext, FillModel, KillSwitch, Ledger, MarketView

__all__ = [
    "ingest",
    "ingest_fills",
    "may_trade",
    "reconcile",
    "record_order_status",
    "run_reconcile",
    "submit_approved",
]

# R2: broker activity types that mean an option position changed outside our order path (9.6)
_ASSIGNMENT_ACTIVITIES: Final[frozenset[str]] = frozenset({"OPASN", "OPEXC", "OPEXP"})
_MIN_OPTIONS_LEVEL: Final = 3
_SLOT_RANK: Final[Mapping[Slot, int]] = {Slot.DEC: 0, Slot.EXEC: 1, Slot.EOD: 2}

# `Ledger.verify()` is full at startup and incremental since the last verified seq every cycle (R5). The high-water mark lives
# beside the ledger object, never inside it: the ledger is a Protocol and the mark is a per-process optimisation, not state.
_VERIFIED: Final[WeakKeyDictionary[Any, int]] = WeakKeyDictionary()


def _rank(key: SnapshotKey) -> tuple[date, int]:
    """Chronological order of a snapshot key: `dec` -> `exec` -> `eod` (the enum's own string order is not chronological)."""
    return (key.session, _SLOT_RANK[key.slot])


# ======================================================================================================================
# THE one ORDER_STATUS writer (9.6)
# ======================================================================================================================


def _write_status(
    ledger: Ledger,
    book: BookP,
    *,
    client_order_id: str,
    intent_id: str,
    attempt: int,
    status: OrderStatus,
    qty: int,
    filled_qty: int,
    limit: int | None,
    broker_order_id: str | None,
    reject_code: int | None,
    tag: str,
    session: date,
    as_of: datetime,
) -> LedgerEntry:
    """The single append + fold. `record_order_status` is its public face; `ingest` uses it for orders it resumes from the
    ledger alone (after a crash there is no `ApprovedOrder` in memory, only ORDER_INTENT + ORDER_STATUS)."""
    payload: dict[str, Any] = {
        "client_order_id": client_order_id,
        "intent_id": intent_id,
        "attempt": attempt,
        "status": OrderStatus(status).value,
        "qty": qty,
        "filled_qty": filled_qty,
        "limit": limit,
        "broker_order_id": broker_order_id,
        "reject_code": reject_code,
        "tag": tag,
    }
    entry = ledger.append(LedgerKind.ORDER_STATUS, session, as_of, payload)
    book.apply(entry)
    return entry


def record_order_status(
    ledger: Ledger,
    book: BookP,
    order: ApprovedOrder,
    status: OrderStatus,
    as_of: datetime,
    *,
    state: OrderState | None = None,
    tag: str = "",
) -> LedgerEntry:
    """THE one writer of ORDER_STATUS entries (9.6): `ledger.append(ORDER_STATUS{...})` + `book.apply(entry)`.

    Nothing else in `src/` may append that kind (`tests/unit/test_reconcile.py` asserts it over the whole tree). Durability is
    the Ledger's commit mode: paper commits and fsyncs on every append, a backtest once per session.
    """
    return _write_status(
        ledger,
        book,
        client_order_id=order.client_order_id,
        intent_id=order.intent.intent_id,
        attempt=order.attempt,
        status=status,
        qty=order.qty,
        filled_qty=state.filled_qty if state is not None else book.filled_qty(order.client_order_id),
        limit=order.limit,
        broker_order_id=state.broker_order_id if state is not None else None,
        reject_code=state.reject_code if state is not None else None,
        tag=tag,
        session=order.intent.session,
        as_of=as_of,
    )


def submit_approved(
    ledger: Ledger,
    book: BookP,
    broker: Broker,
    order: ApprovedOrder,
    as_of: datetime,
) -> OrderState | None:
    """THE submit protocol of 9.6, written once for every caller. Returns the broker's state, or None when it was rejected or
    the outcome is unknown (both are already ledgered)."""
    record_order_status(ledger, book, order, OrderStatus.SUBMITTING, as_of)
    try:
        state = broker.submit(order)
    except BrokerRejected as rejected:
        record_order_status(ledger, book, order, OrderStatus.REJECTED, as_of, tag=rejected.tag or "rejected")
        return None
    except BrokerAmbiguous:
        record_order_status(ledger, book, order, OrderStatus.UNKNOWN, as_of)
        return None
    record_order_status(ledger, book, order, state.status, as_of, state=state)
    return state


# ======================================================================================================================
# THE one fill path (9.6)
# ======================================================================================================================


def ingest_fills(ctx: CycleContext, view: MarketView) -> int:
    """Book every new broker fill; returns how many FILL entries were appended (9.6)."""
    return ingest(
        ledger=ctx.ledger,
        book=ctx.book,
        broker=ctx.broker,
        fill_model=ctx.fill_model,
        view=view,
        calendar=ctx.calendar,
        cfg=ctx.cfg,
        mode=ctx.meta.mode,
    )


def ingest(
    *,
    ledger: Ledger,
    book: BookP,
    broker: Broker,
    fill_model: FillModel,
    view: MarketView,
    calendar: Calendar,
    cfg: Config,
    mode: RunMode,
) -> int:
    """The 9.6 loop over every Book order whose latest status is not terminal, sorted by client_order_id."""
    booked = 0
    positions = {p.position_id: p for p in book.state().positions}
    for intent, ledger_state in book.open_orders():
        cid = ledger_state.client_order_id
        state = broker.get_order(cid)
        if state is None:
            _missing_order(ledger, book, intent, ledger_state, view, calendar, cfg, mode)
            continue
        previous = book.filled_qty(cid)
        if state.status is not ledger_state.status or state.filled_qty != ledger_state.filled_qty:
            _write_status(
                ledger,
                book,
                client_order_id=cid,
                intent_id=intent.intent_id,
                attempt=_attempt_of(cid),
                status=state.status,
                qty=state.qty,
                filled_qty=state.filled_qty,
                limit=None,  # a broker status report carries no limit; the attempt's limit is on its SUBMITTING entry
                broker_order_id=state.broker_order_id,
                reject_code=state.reject_code,
                tag="",
                session=intent.session,
                as_of=view.as_of,
            )
        delta = state.filled_qty - previous
        if delta <= 0:
            continue
        if not ledger.claim_fill(ids.fill_id(cid, state.filled_qty)):
            continue  # a second route already booked this cumulative quantity (the dedupe of the ONE fill path)
        _book_fill(
            ledger=ledger,
            book=book,
            fill_model=fill_model,
            view=view,
            mode=mode,
            intent=intent,
            state=state,
            delta=delta,
            open_net=_open_net(positions, intent),
        )
        booked += 1
    return booked


def _attempt_of(client_order_id: str) -> int:
    tail = client_order_id.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def _open_net(positions: Mapping[str, Position], intent: OrderIntent) -> BandPrices | None:
    if intent.purpose is OrderPurpose.OPEN:
        return None
    position = positions.get(intent.position_id)
    return None if position is None else position.open_net


def _missing_order(
    ledger: Ledger,
    book: BookP,
    intent: OrderIntent,
    ledger_state: OrderState,
    view: MarketView,
    calendar: Calendar,
    cfg: Config,
    mode: RunMode,
) -> None:
    """`broker.get_order` returned None (9.6): never sent, still only an intent, or genuinely unknown."""
    if ledger_state.status is OrderStatus.INTENT:
        return  # ledgered but never reached SUBMITTING: the order worker resumes it by id on the same session
    status = OrderStatus.UNKNOWN
    tag = ""
    if ledger_state.status is OrderStatus.SUBMITTING and _cutoff_passed(intent, view, calendar, cfg, mode):
        status, tag = OrderStatus.CANCELLED, "never_sent"
    _write_status(
        ledger,
        book,
        client_order_id=ledger_state.client_order_id,
        intent_id=intent.intent_id,
        attempt=_attempt_of(ledger_state.client_order_id),
        status=status,
        qty=ledger_state.qty,
        filled_qty=ledger_state.filled_qty,
        limit=None,
        broker_order_id=None,
        reject_code=None,
        tag=tag,
        session=intent.session,
        as_of=view.as_of,
    )


def _cutoff_passed(intent: OrderIntent, view: MarketView, calendar: Calendar, cfg: Config, mode: RunMode) -> bool:
    if mode is RunMode.BACKTEST:
        return _rank(view.key) > _rank(intent.key)  # the order's eligible snapshot has passed
    return view.as_of >= calendar.offset_from_close(view.key.session, cfg.cadence.order_cutoff_offset_min)


def _book_fill(
    *,
    ledger: Ledger,
    book: BookP,
    fill_model: FillModel,
    view: MarketView,
    mode: RunMode,
    intent: OrderIntent,
    state: OrderState,
    delta: int,
    open_net: BandPrices | None,
) -> None:
    chain = view.chain(intent.underlying)
    net, leg_fills, quality = fill_model.price(intent.legs, chain, mandatory=intent.mandatory)
    model_reject: tuple[str, ...] = ()
    if mode is not RunMode.BACKTEST:
        # paper records what our fill model WOULD have rejected; the broker's fill is still booked (one operational book)
        model_reject = fill_model.check(intent.legs, delta, chain, mandatory=intent.mandatory)
    fill = Fill(
        fill_id=ids.fill_id(state.client_order_id, state.filled_qty),
        client_order_id=state.client_order_id,
        intent_id=intent.intent_id,
        decision_id=intent.decision_id,
        position_id=intent.position_id,
        purpose=intent.purpose,
        structure_id=intent.structure.structure_id if intent.structure is not None else None,
        qty=delta,
        key=view.key,
        ts=view.as_of,
        net=net,
        legs=tuple(leg_fills),
        fees_micro=fill_model.fees_micro(intent.legs, delta, leg_fills),
        forced=intent.mandatory,
        model_reject=model_reject,
        quality=quality,
        source="sim" if mode is RunMode.BACKTEST else "paper",
        broker_order_id=state.broker_order_id,
        broker_net=state.filled_net,
    )
    book.apply(
        ledger.append(
            LedgerKind.BROKER_FILL,
            intent.session,
            view.as_of,
            {
                "client_order_id": state.client_order_id,
                "cum_qty": state.filled_qty,
                "broker_net": state.filled_net,
                "ts": msgspec.to_builtins(view.as_of),
            },
        )
    )
    payload: dict[str, Any] = msgspec.to_builtins(fill)
    if open_net is not None:
        realised = BandPrices(
            orats=(-open_net.orats - net.orats) * MULTIPLIER * delta,
            worst=(-open_net.worst - net.worst) * MULTIPLIER * delta,
            mid=(-open_net.mid - net.mid) * MULTIPLIER * delta,
        )
        payload["realised_pnl"] = msgspec.to_builtins(realised)
    book.apply(ledger.append(LedgerKind.FILL, intent.session, view.as_of, payload))


# ======================================================================================================================
# reconcile R1-R6 (9.6, INV-10)
# ======================================================================================================================


def reconcile(ctx: CycleContext, view: MarketView | None, *, morning: bool = False, boot: bool = False) -> bool:
    """R1-R6 at startup, at every cycle start, after each terminal order state, in the morning pass and post-close.

    `boot=True` adds the second half of R6: after downtime, a position that is already inside its hard-exit window needs
    immediate emergency action, not the near-close cycle (9.6, INV-11).
    """
    return run_reconcile(
        ledger=ctx.ledger,
        book=ctx.book,
        broker=ctx.broker,
        fill_model=ctx.fill_model,
        kill=ctx.kill,
        view=view,
        calendar=ctx.calendar,
        cfg=ctx.cfg,
        mode=ctx.meta.mode,
        session=view.key.session if view is not None else _book_session(ctx.book),
        as_of=view.as_of if view is not None else ctx.clock.now(),
        morning=morning,
        boot=boot,
    )


def run_reconcile(
    *,
    ledger: Ledger,
    book: BookP,
    broker: Broker,
    fill_model: FillModel,
    kill: KillSwitch,
    view: MarketView | None,
    calendar: Calendar,
    cfg: Config,
    mode: RunMode,
    session: date,
    as_of: datetime,
    morning: bool = False,
    boot: bool = False,
) -> bool:
    ok = True
    order_actions: list[str] = []
    foreign: list[str] = []
    activities: list[str] = []
    diff: dict[str, list[int]] = {}

    # R1 ORDERS -------------------------------------------------------------------------------------------------------
    if view is not None:
        booked = ingest(ledger=ledger, book=book, broker=broker, fill_model=fill_model, view=view, calendar=calendar, cfg=cfg, mode=mode)
        if booked:
            order_actions.append(f"ingested:{booked}")
    for state in broker.open_orders():
        cid = state.client_order_id
        # ours iff the id has the shape ids.py emits AND its intent is in the ledger; "jbp-" probe leftovers are foreign
        if ids.is_bot_order_id(cid) and book.has_intent(cid.rsplit("-", 1)[0]):
            continue
        foreign.append(cid)
        try:
            broker.cancel(cid)
            order_actions.append(f"cancelled_foreign:{cid}")
        except BrokerError:
            order_actions.append(f"cancel_failed:{cid}")

    # R2 ACTIVITIES ---------------------------------------------------------------------------------------------------
    if morning:
        since = _prev_session(calendar, session)
        ours = set(book.leg_positions())
        for activity in _activities(broker, since):
            if activity.activity_type not in _ASSIGNMENT_ACTIVITIES:
                continue
            activities.append(f"{activity.activity_type}:{activity.symbol}:{activity.qty}")
            if activity.symbol in ours or _is_our_root(activity.symbol, cfg):
                ok = False
                kill.trip(KillTrigger.ASSIGNMENT, f"{activity.activity_type} {activity.symbol} on {activity.day.isoformat()}")

    # R3 POSITIONS ----------------------------------------------------------------------------------------------------
    expected = book.leg_positions()
    actual = {p.symbol: p.qty for p in broker.positions() if p.qty != 0}
    if expected != actual:
        ok = False
        for symbol in sorted(set(expected) | set(actual)):
            if expected.get(symbol, 0) != actual.get(symbol, 0):
                diff[symbol] = [expected.get(symbol, 0), actual.get(symbol, 0)]

    # R4 ACCOUNT ------------------------------------------------------------------------------------------------------
    account = broker.account()
    problems = []
    if account.options_level < _MIN_OPTIONS_LEVEL:
        problems.append(f"options_level={account.options_level}")
    if account.trading_blocked:
        problems.append("trading_blocked")
    if account.account_blocked:
        problems.append("account_blocked")
    if account.suspended and kill.state() is not KillState.LOCKED:
        problems.append("suspended")
    if problems:
        raise PaperGuardError(f"reconcile R4 refuses to trade: {', '.join(problems)}")

    # R5 CHAIN --------------------------------------------------------------------------------------------------------
    from_seq = _VERIFIED.get(ledger, 0) + 1
    ledger.verify(from_seq=from_seq)
    _VERIFIED[ledger] = ledger.head()[0]

    # R6 EXPIRY -------------------------------------------------------------------------------------------------------
    expiring = sorted(_expiring_legs(book, broker, calendar, session))
    inside_window = sorted(_inside_hard_exit_window(book, calendar, cfg, session)) if boot else []
    if expiring or inside_window:
        ok = False

    book.apply(
        ledger.append(
            LedgerKind.RECONCILE,
            session,
            as_of,
            {
                "ok": ok,
                "order_actions": order_actions,
                "diff": diff,
                "foreign_orders": foreign,
                "activities": activities,
            },
        )
    )
    if diff:
        kill.trip(KillTrigger.RECONCILE_MISMATCH, f"ledger/broker positions differ: {sorted(diff)}")
    if expiring:
        kill.trip(KillTrigger.EXPIRY_VIOLATION, f"last_session <= {session.isoformat()}: {','.join(expiring)}")
    elif inside_window:
        kill.trip(KillTrigger.EXPIRY_VIOLATION, f"inside the hard-exit window at boot: {','.join(inside_window)}")
    return ok


def may_trade(broker: Broker, kill_state: KillState) -> None:
    """R4 on its own: the account-level refusal, for callers (the paper boot) that need it before a full reconcile."""
    account = broker.account()
    if account.options_level < _MIN_OPTIONS_LEVEL or account.trading_blocked or account.account_blocked:
        raise PaperGuardError("the account may not trade options (R4)")
    if account.suspended and kill_state is not KillState.LOCKED:
        raise PaperGuardError("the account is suspended while the kill switch is not LOCKED (R4)")


def _book_session(book: BookP) -> date:
    key = book.last_key
    if key is None:
        raise InvariantError("reconcile without a view needs a book that has folded at least one SESSION_START")
    return key.session


def _prev_session(calendar: Calendar, session: date) -> date:
    try:
        return calendar.prev_session(session)
    except ValueError:
        return session


def _activities(broker: Broker, since: date) -> Sequence[BrokerActivity]:
    try:
        return broker.activities(since)
    except BrokerError:
        return ()


def _is_our_root(symbol: str, cfg: Config) -> bool:
    if symbol in cfg.universe.underlyings:
        return True
    try:
        return occ.parse_occ(symbol).underlying in cfg.universe.underlyings
    except ValueError:
        return False


def _inside_hard_exit_window(book: BookP, calendar: Calendar, cfg: Config, session: date) -> list[str]:
    """Positions already inside their mandatory-exit window - the BOOT half of R6 (9.6, INV-11)."""
    return [
        position.position_id
        for position in book.state().positions
        if calendar.sessions_between(session, position.structure.last_session) <= cfg.dte.hard_exit_sessions
    ]


def _expiring_legs(book: BookP, broker: Broker, calendar: Calendar, session: date) -> list[str]:
    """Every ledger or broker leg whose LAST TRADING DAY is today or past (R6, INV-11).

    `expiry <= today` would never be true on the day that really is the last trading day of a Saturday-dated monthly, so the
    test is on `last_session = calendar.prev_or_same_session(expiry)` - the Friday of a Saturday-dated monthly, the Thursday
    of a Good-Friday week.
    """
    found: list[str] = []
    for position in book.state().positions:
        if position.structure.last_session <= session:
            found.append(position.structure.legs[0].contract.occ)
    for broker_position in broker.positions():
        if not broker_position.is_option or broker_position.qty == 0:
            continue
        try:
            contract = occ.parse_occ(broker_position.symbol)
        except ValueError:
            continue
        if calendar.prev_or_same_session(contract.expiry) <= session:
            found.append(broker_position.symbol)
    return sorted(set(found))
