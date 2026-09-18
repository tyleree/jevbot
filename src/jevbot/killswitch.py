"""`DefaultKillSwitch`: the sticky, persisted kill-switch state machine and the flatten driver of DESIGN.md 9.5 (INV-21).

States: `ARMED -> TRIPPED -> FLATTENING -> LOCKED`, or `FLATTENING -> NOT_FLAT -> FLATTENING ...`.

**Persistence before any action** (`trip()`): `state/KILL` (JSON `event_id, trigger, detail, ts, ledger_head`; written to a
temporary file, fsynced, renamed, directory fsynced) *and* a ledger `KILL{step:"tripped"}` entry are durable before `trip()`
returns and before a single broker call is made. A process that crashes between the file write and the first broker call
starts again in kill mode: the constructor resumes from the book's folded KILL / REARM entries and from the file, and
`step()` picks the sequence up where it stopped.

The flatten sequence is exactly G6 / D17, idempotent and resumable, every step ledgered as `KILL{step}`:

    K1  stop trading decisions            (the cycle's own manage step is suspended while the kill switch owns the book)
    K2  broker.cancel_all(), poll open_orders() until empty (kill.cancel_wait_s), then continue and re-cancel in K3
    K3  flatten what the BROKER reports, grouped into structures with the Book where it matches; structures with short legs
        first, nearest expiry first; ONE mleg order per structure, RiskEngine.approve(now=clock.now()), the 9.6 submit
        protocol, wait kill.flatten_wait_s, then cancel and resubmit at natural + attempt * kill.cushion_ticks
    K4  per-leg fallback, SHORT legs first, parts 1..4; last resort a market order inside market hours; any EQUITY position
        (assignment) is closed with a market order
    K5  verify flat on two consecutive polls 5 s apart
    K6  ONLY when flat: broker.set_suspended(True) -> LOCKED
    K7  otherwise NOT_FLAT: alert and retry K2-K5 every kill.not_flat_retry_s while the market is open. The process never
        exits and never suspends while not flat.

Tripping while the market is closed does K1-K2 at once and defers K3 to `kill.post_open_delay_min` after the next open -
`kill.post_open_delay_urgent_min` when the trigger is urgent or any position sits inside its hard-exit window.

Kill orders are exempt from the order-rate and attempt caps (9.1 check 21): the kill path must never throttle itself.
Re-arm is manual (9.5): the operator writes `state/REARM` by hand with the kill `event_id`. In a backtest
`kill.backtest_behaviour` decides instead: `flatten_and_cooldown` re-arms automatically after
`kill.backtest_cooldown_sessions` **with** the peak reset, `stop_run` sets `run_stopped`.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import msgspec

from jevbot import canon, ids, money, occ, reconcile
from jevbot.config import Config
from jevbot.errors import BrokerError, DataError, InvariantError
from jevbot.types import (
    TERMINAL_STATUSES,
    ChainSnapshot,
    ExitReason,
    KillState,
    KillTrigger,
    LedgerKind,
    OptionContract,
    OrderIntent,
    OrderLeg,
    OrderPurpose,
    PositionIntent,
    RiskVerdict,
    RunMeta,
    RunMode,
    Side,
    SnapshotKey,
    Structure,
)

if TYPE_CHECKING:
    from jevbot.protocols import BookP, Broker, Calendar, Clock, FillModel, Ledger, MarketView, RiskEngine

__all__ = ["DefaultKillSwitch"]

KILL_FILE: Final = "KILL"
REARM_FILE: Final = "REARM"

# 9.5: these trip states must be flattened as soon as the market opens, not after the ordinary post-open delay
URGENT_TRIGGERS: Final[frozenset[KillTrigger]] = frozenset({KillTrigger.EXPIRY_VIOLATION, KillTrigger.ASSIGNMENT})

_FLAT_POLL_S: Final = 5  # K5: "two consecutive polls 5 s apart" (a duration, never a time of day - INV-13)
_MAX_FALLBACK_PARTS: Final = 4  # ids.intent_id: parts 1..4 enumerate the legs of the per-leg fallback


@dataclass(frozen=True)
class _Group:
    """One flatten unit: a Book structure the broker still holds, or a single unmatched broker leg (9.5 K3)."""

    position_id: str
    underlying: str
    structure: Structure | None
    legs: tuple[OrderLeg, ...]  # CLOSE legs: sides already flipped against what the broker holds
    qty: int
    expiry: date
    has_short: bool

    @property
    def sort_key(self) -> tuple[int, date, str]:
        return (0 if self.has_short else 1, self.expiry, self.position_id)


def _close_leg(contract: OptionContract, held: int) -> OrderLeg:
    """The order leg that reduces a held broker leg: long -> (SELL, sell_to_close), short -> (BUY, buy_to_close) (2.4)."""
    if held > 0:
        return OrderLeg(contract=contract, side=Side.SELL, position_intent=PositionIntent.STC)
    return OrderLeg(contract=contract, side=Side.BUY, position_intent=PositionIntent.BTC)


def _frac(value: float) -> Fraction:
    return Fraction(repr(float(value)))


def _ceil(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


class DefaultKillSwitch:
    """Implements `protocols.KillSwitch` (3.4) - see the module docstring."""

    def __init__(
        self,
        *,
        cfg: Config,
        meta: RunMeta,
        ledger: Ledger,
        book: BookP,
        risk: RiskEngine,
        clock: Clock,
        fill_model: FillModel,
        calendar: Calendar,
        state_dir: Path,
        broker: Broker | None = None,
        sleep: Callable[[float], None] = time.sleep,
        alert: Callable[[str], None] | None = None,
    ) -> None:
        self._cfg: Final = cfg
        self._meta: Final = meta
        self._ledger: Final = ledger
        self._book: Final = book
        self._risk: Final = risk
        self._clock: Final = clock
        self._fill_model: Final = fill_model
        self._calendar: Final = calendar
        self._state_dir: Final = Path(state_dir)
        self._broker = broker
        self._sleep: Final = sleep
        self._alert_hook: Final = alert
        self._trigger: KillTrigger | None = None
        self._detail = ""
        self._cancelled = False
        self._k3_not_before: datetime | None = None
        self._retry_at: datetime | None = None
        self._cooldown_until: date | None = None
        self.run_stopped = False
        self._state, self._event_id = self._resume()

    # ------------------------------------------------------------------------------------------------------------------
    # KillSwitch protocol
    # ------------------------------------------------------------------------------------------------------------------

    def state(self) -> KillState:
        return self._state

    def event_id(self) -> str | None:
        return self._event_id

    def trip(self, trigger: KillTrigger, detail: str) -> None:
        """Idempotent; the KILL file and the `KILL{tripped}` entry are durable BEFORE this returns (9.5)."""
        if self._state is not KillState.ARMED:
            return
        head_seq, head_hash = self._ledger.head()
        session = self._session()
        event = canon.sha256_hex("|".join([self._meta.namespace, session.isoformat(), trigger.value, str(head_seq)]))[:16]
        self._event_id = event
        self._trigger = KillTrigger(trigger)
        self._detail = detail
        self._write_kill_file(head_hash)
        self._state = KillState.TRIPPED
        self._ledger_step("tripped", detail)

    def step(self, broker: Broker, view: MarketView | None, market_open: bool) -> KillState:
        """Advance the flatten sequence one round (9.5). Never exits the process and never suspends while not flat."""
        self._broker = broker
        if self._state is KillState.ARMED:
            return self._state
        if self._state is KillState.LOCKED:
            self._maybe_backtest_rearm(broker, view)
            return self._state
        now = self._clock.now()
        if not self._cancelled:
            self._k2_cancel(broker)
        if not market_open:
            self._defer_k3(now)
            return self._state
        if self._k3_not_before is not None and now < self._k3_not_before:
            return self._state
        if self._state is KillState.NOT_FLAT:
            if self._retry_at is not None and now < self._retry_at:
                return self._state
            self._cancelled = False
            self._k2_cancel(broker)
        if view is None:
            return self._state  # K3 prices from a fresh snapshot; without one only K1-K2 can run
        self._state = KillState.FLATTENING
        self._flatten(broker, view)
        if self._verify_flat(broker):
            self._ledger_step("flat_verified", "")
            broker.set_suspended(True)  # K6: suspend_trade blocks closing orders too, so it is strictly last
            self._ledger_step("suspended", "")
            self._state = KillState.LOCKED
            self._ledger_step("locked", "")
            self._on_locked(view)
        else:
            self._state = KillState.NOT_FLAT
            self._ledger_step("not_flat", self._open_summary(broker))
            self._retry_at = now + timedelta(seconds=self._cfg.kill.not_flat_retry_s)
            self._alert(f"kill {self._event_id}: NOT FLAT, retrying")
        return self._state

    def rearm(self, rearm_file_text: str, *, reset_peak: bool, note: str) -> None:
        """Manual re-arm (9.5): the hand-written `state/REARM` must carry the kill `event_id`, the state must be LOCKED or
        TRIPPED and the broker must be flat with no open orders. Without `--reset-peak` a DRAWDOWN kill re-trips at once."""
        if self._event_id is None or rearm_file_text.strip() != self._event_id:
            raise InvariantError(f"rearm: {REARM_FILE} must contain the kill event id {self._event_id!r}")
        if self._state not in (KillState.LOCKED, KillState.TRIPPED):
            raise InvariantError(f"rearm: the kill state is {self._state.value}, not locked or tripped")
        broker = self._require_broker()
        if broker.positions() or broker.open_orders():
            raise InvariantError("rearm: the broker is not flat / still has open orders")
        broker.set_suspended(False)
        self._append_rearm(reset_peak=reset_peak, note=note)
        for name in (KILL_FILE, REARM_FILE):
            (self._state_dir / name).unlink(missing_ok=True)
        self._reset_to_armed()

    # ------------------------------------------------------------------------------------------------------------------
    # K2 / K5 / K6 / K7
    # ------------------------------------------------------------------------------------------------------------------

    def _k2_cancel(self, broker: Broker) -> None:
        broker.cancel_all()
        polls = max(1, self._cfg.kill.cancel_wait_s // max(self._cfg.orders.poll_s, 1))
        remaining = broker.open_orders()
        for _ in range(polls):
            if not remaining:
                break
            self._sleep(self._cfg.orders.poll_s)
            remaining = broker.open_orders()
        self._cancelled = True
        self._ledger_step("cancelled", f"open_orders={len(remaining)}")

    def _verify_flat(self, broker: Broker) -> bool:
        """K5: flat on two consecutive polls `_FLAT_POLL_S` apart."""
        for index in range(2):
            if broker.positions() or broker.open_orders():
                return False
            if index == 0:
                self._sleep(_FLAT_POLL_S)
        return True

    def _open_summary(self, broker: Broker) -> str:
        positions = ",".join(sorted(f"{p.symbol}:{p.qty}" for p in broker.positions()))
        return f"positions={positions} open_orders={len(broker.open_orders())}"

    def _defer_k3(self, now: datetime) -> None:
        if self._k3_not_before is not None:
            return
        urgent = self._trigger in URGENT_TRIGGERS or self._inside_hard_exit_window()
        minutes = self._cfg.kill.post_open_delay_urgent_min if urgent else self._cfg.kill.post_open_delay_min
        self._k3_not_before = self._calendar.next_open_after(now) + timedelta(minutes=minutes)

    def _inside_hard_exit_window(self) -> bool:
        session = self._session()
        floor = self._cfg.dte.hard_exit_sessions
        return any(
            self._calendar.sessions_between(session, position.structure.last_session) <= floor for position in self._book.state().positions
        )

    # ------------------------------------------------------------------------------------------------------------------
    # K3 / K4
    # ------------------------------------------------------------------------------------------------------------------

    def _flatten(self, broker: Broker, view: MarketView) -> None:
        holdings = {p.symbol: p.qty for p in broker.positions() if p.qty != 0}
        self._push_holdings(holdings)
        try:
            for group in sorted(self._groups(holdings), key=lambda g: g.sort_key):
                if self._close_group(broker, view, group):
                    continue
                self._fallback_legs(broker, view, group)
            self._flatten_equity(broker, view, holdings)
        finally:
            self._push_holdings(None)

    def _push_holdings(self, holdings: Mapping[str, int] | None) -> None:
        """Hand the BROKER's holdings to the (pure) risk engine for check 6 (9.1); it never calls a broker itself."""
        setter = getattr(self._risk, "set_kill_holdings", None)
        if callable(setter):
            setter(holdings)

    def _groups(self, holdings: Mapping[str, int]) -> list[_Group]:
        """Group the broker's option legs into structures using the Book where it matches; the rest are single-leg groups."""
        remaining = {symbol: qty for symbol, qty in holdings.items() if occ.is_occ(symbol) and qty != 0}
        groups: list[_Group] = []
        for position in sorted(self._book.state().positions, key=lambda p: p.position_id):
            structure = position.structure
            matched: list[tuple[OptionContract, int, int]] = []  # (contract, broker qty, ratio)
            for leg in structure.legs:
                held = remaining.get(leg.contract.occ, 0)
                long_leg = leg.side is Side.BUY
                if held == 0 or (held > 0) is not long_leg:
                    matched = []
                    break
                matched.append((leg.contract, held, leg.ratio))
            if not matched:
                continue
            qty = min(abs(held) // ratio for _contract, held, ratio in matched)
            if qty <= 0:
                continue
            for leg in structure.legs:
                signed = qty * leg.ratio * (1 if leg.side is Side.BUY else -1)
                remaining[leg.contract.occ] -= signed
                if remaining[leg.contract.occ] == 0:
                    del remaining[leg.contract.occ]
            groups.append(
                _Group(
                    position_id=position.position_id,
                    underlying=structure.underlying,
                    structure=structure,
                    legs=tuple(_close_leg(contract, held) for contract, held, _ratio in matched),
                    qty=qty,
                    expiry=structure.last_session,
                    has_short=any(leg.side is Side.SELL for leg in structure.legs),
                )
            )
        for symbol in sorted(remaining):
            held = remaining[symbol]
            contract = occ.parse_occ(symbol)
            groups.append(
                _Group(
                    position_id=canon.sha256_hex(f"LEG:{symbol}")[:16],
                    underlying=contract.underlying,
                    structure=None,
                    legs=(_close_leg(contract, held),),
                    qty=abs(held),
                    expiry=self._calendar.prev_or_same_session(contract.expiry),
                    has_short=held < 0,
                )
            )
        return groups

    def _close_group(self, broker: Broker, view: MarketView, group: _Group) -> bool:
        """K3: one mleg (or single-leg) order per structure, up to `kill.flatten_attempts` rungs. True when it is flat."""
        intent = self._intent(view, group, group.legs, group.qty, part=0)
        if intent is None:
            return False
        self._append_intent(intent, view)
        for attempt in range(self._cfg.kill.flatten_attempts):
            if self._flat(broker, group.legs):
                return True
            limit = self._limit(view, intent, group.legs, attempt)
            if not self._send(broker, view, intent, attempt, limit):
                continue
            self._await_fill(broker, view, intent, attempt)
        return self._flat(broker, group.legs)

    def _fallback_legs(self, broker: Broker, view: MarketView, group: _Group) -> None:
        """K4: legs separately, SHORT legs first, parts 1..4; last resort a market order inside market hours."""
        self._ledger_step("fallback_legs", f"{group.position_id}:{len(group.legs)} legs")
        ordered = sorted(group.legs, key=lambda leg: 0 if leg.side is Side.BUY else 1)  # buy_to_close = the SHORT legs first
        for index, leg in enumerate(ordered[:_MAX_FALLBACK_PARTS], start=1):
            held = self._held(broker, leg.contract.occ)
            if held == 0:
                continue
            intent = self._intent(view, group, (leg,), abs(held), part=index)
            if intent is None:
                continue
            self._append_intent(intent, view)
            for attempt in range(self._cfg.kill.flatten_attempts):
                if self._flat(broker, (leg,)):
                    break
                limit = self._limit(view, intent, (leg,), attempt)
                if self._send(broker, view, intent, attempt, limit):
                    self._await_fill(broker, view, intent, attempt)
            if not self._flat(broker, (leg,)):
                # last resort: a market order (limit None), market hours only - K3 / K4 run only while the market is open
                last = self._cfg.kill.flatten_attempts
                if self._send(broker, view, intent, last, None, market=True):
                    self._await_fill(broker, view, intent, last)

    def _flatten_equity(self, broker: Broker, view: MarketView, holdings: Mapping[str, int]) -> None:
        """K4: any EQUITY position (an assignment) is closed with a market order (9.5); checks 3 and 6 still run (9.1)."""
        for symbol in sorted(s for s in holdings if not occ.is_occ(s)):
            shares = self._held(broker, symbol)
            if shares == 0:
                continue
            session = self._session(view)
            decision = ids.decision_id(self._meta.namespace, session, symbol, "manage", ids.equity_kill_subject(symbol))
            intent = OrderIntent(
                intent_id=ids.intent_id(self._meta.namespace, session, decision, OrderPurpose.KILL, 0),
                decision_id=decision,
                position_id=ids.equity_position_id(symbol),
                purpose=OrderPurpose.KILL,
                part=0,
                underlying=symbol,
                legs=(),
                qty=0,
                limit_start=0,
                limit_natural=0,
                reason=ExitReason.KILL.value,
                mandatory=True,
                session=session,
                key=view.key,
                structure=None,
                equity_symbol=symbol,
                equity_side=Side.SELL if shares > 0 else Side.BUY,
                equity_qty=abs(shares),
            )
            self._append_intent(intent, view)
            if self._send(broker, view, intent, 0, None, market=True):
                self._await_fill(broker, view, intent, 0)

    # --- order plumbing ------------------------------------------------------------------------------------------------

    def _intent(self, view: MarketView, group: _Group, legs: Sequence[OrderLeg], qty: int, *, part: int) -> OrderIntent | None:
        session = self._session(view)
        natural = self._natural(view, legs)
        if natural is None:
            return None
        decision = ids.decision_id(self._meta.namespace, session, group.underlying, "manage", ids.kill_subject(group.position_id))
        structure = group.structure if part == 0 else None
        limit_natural = self._round_natural(natural, self._tick(view, legs), legs, structure)
        return OrderIntent(
            intent_id=ids.intent_id(self._meta.namespace, session, decision, OrderPurpose.KILL, part),
            decision_id=decision,
            position_id=group.position_id,
            purpose=OrderPurpose.KILL,
            part=part,
            underlying=group.underlying,
            legs=tuple(legs),
            qty=qty,
            limit_start=limit_natural,
            limit_natural=limit_natural,
            reason=ExitReason.KILL.value,
            mandatory=True,
            session=session,
            key=view.key,
            structure=structure,
        )

    def _append_intent(self, intent: OrderIntent, view: MarketView) -> None:
        if self._book.has_intent(intent.intent_id):
            return  # resumed after a crash: the ledgered intent is never rebuilt (10.1)
        payload: dict[str, Any] = msgspec.to_builtins(intent)
        self._book.apply(self._ledger.append(LedgerKind.ORDER_INTENT, intent.session, view.as_of, payload))

    def _send(
        self,
        broker: Broker,
        view: MarketView,
        intent: OrderIntent,
        attempt: int,
        limit: int | None,
        *,
        market: bool = False,
    ) -> bool:
        verdict, order = self._risk.approve(
            intent,
            self._book.state(),
            view,
            now=self._clock.now(),
            attempt=attempt,
            limit=limit,
            market=market,
        )
        self._book.apply(self._ledger.append(LedgerKind.RISK_VERDICT, intent.session, view.as_of, _verdict_payload(verdict)))
        if order is None:
            return False
        state = reconcile.submit_approved(self._ledger, self._book, broker, order, view.as_of)
        self._ledger_step("close_submitted", f"{order.client_order_id} limit={order.limit}")
        return state is not None

    def _await_fill(self, broker: Broker, view: MarketView, intent: OrderIntent, attempt: int) -> None:
        cid = ids.client_order_id(intent, attempt)
        polls = max(1, self._cfg.kill.flatten_wait_s // max(self._cfg.orders.poll_s, 1))
        for index in range(polls):
            state = broker.get_order(cid)
            if state is not None and state.filled_qty > 0:
                self._ingest(view)
            if state is not None and state.status in TERMINAL_STATUSES:
                return
            if index < polls - 1:
                self._sleep(self._cfg.orders.poll_s)
        try:
            broker.cancel(cid)
        except BrokerError:
            pass
        self._ingest(view)

    def _ingest(self, view: MarketView) -> None:
        broker = self._require_broker()
        reconcile.ingest(
            ledger=self._ledger,
            book=self._book,
            broker=broker,
            fill_model=self._fill_model,
            view=view,
            calendar=self._calendar,
            cfg=self._cfg,
            mode=self._meta.mode,
        )

    # --- pricing -------------------------------------------------------------------------------------------------------

    def _chain(self, view: MarketView, legs: Sequence[OrderLeg]) -> ChainSnapshot | None:
        if not legs:
            return None
        try:
            return view.chain(legs[0].contract.underlying)
        except DataError:  # no fresh snapshot for this underlying; K4's market order is the fallback
            return None

    def _natural(self, view: MarketView, legs: Sequence[OrderLeg]) -> int | None:
        """The natural (worst-band) close price from a fresh snapshot, through THE fill model - never a second copy of 10.3."""
        chain = self._chain(view, legs)
        if chain is None:
            return None
        net, _legs, _quality = self._fill_model.price(legs, chain, mandatory=False)
        return net.worst

    def _tick(self, view: MarketView, legs: Sequence[OrderLeg]) -> int:
        chain = self._chain(view, legs)
        penny = self._cfg.universe.penny_all
        prices = []
        if chain is not None:
            for leg in legs:
                quote = chain.quote(leg.contract)
                if quote is not None:
                    prices.append(quote.mid2 // 2)
        if not prices:
            prices = [0]
        return min(money.tick_cents(legs[0].contract.underlying, price, penny) for price in prices)

    def _round_natural(self, natural: int, tick: int, legs: Sequence[OrderLeg], structure: Structure | None) -> int:
        rounded = money.round_net(natural, tick, aggressive=True)
        if structure is not None and len(legs) > 1:
            return rounded
        # a single-leg order is sent as abs(limit) with the leg's side, so its limit must carry that side's sign: selling a
        # long leg to close is negative even when the bid is 0 (money.round_net leaves that clamp to the caller)
        if legs[0].side is Side.SELL:
            return min(rounded, -tick)
        return max(rounded, tick)

    def _limit(self, view: MarketView, intent: OrderIntent, legs: Sequence[OrderLeg], attempt: int) -> int:
        """`natural + attempt * kill.cushion_ticks`, capped at `kill.cushion_max_frac_width` of the width (9.5)."""
        tick = self._tick(view, legs)
        kill = self._cfg.kill
        width = intent.structure.width if intent.structure is not None else 0
        pad = _ceil(_frac(kill.cushion_max_frac_width) * width) if width > 0 else kill.cushion_ticks * kill.flatten_attempts * tick
        cushion = min(attempt * kill.cushion_ticks * tick, pad)
        limit = intent.limit_natural + cushion
        if intent.structure is not None and len(legs) > 1:
            return limit
        return min(limit, -tick) if legs[0].side is Side.SELL else max(limit, tick)

    # --- broker helpers ------------------------------------------------------------------------------------------------

    def _held(self, broker: Broker, symbol: str) -> int:
        return next((p.qty for p in broker.positions() if p.symbol == symbol), 0)

    def _flat(self, broker: Broker, legs: Sequence[OrderLeg]) -> bool:
        wanted = {leg.contract.occ for leg in legs}
        return not any(p.symbol in wanted and p.qty != 0 for p in broker.positions())

    def _require_broker(self) -> Broker:
        if self._broker is None:
            raise InvariantError("the kill switch has no broker: pass one to the constructor or call step(broker, ...)")
        return self._broker

    # ------------------------------------------------------------------------------------------------------------------
    # persistence and resume
    # ------------------------------------------------------------------------------------------------------------------

    def _resume(self) -> tuple[KillState, str | None]:
        """The 9.5 startup rule: if EITHER the file or the ledger replay says tripped, the process starts in kill mode."""
        state = self._book.state()
        found = state.kill_state
        event_id = state.kill_event_id
        stored = self._read_kill_file()
        if stored is not None:
            event_id = event_id or str(stored.get("event_id") or "") or None
            self._detail = str(stored.get("detail", ""))
            trigger = stored.get("trigger")
            if isinstance(trigger, str):
                try:
                    self._trigger = KillTrigger(trigger)
                except ValueError:
                    self._trigger = None
            if found is KillState.ARMED:
                found = KillState.TRIPPED
        if found is not KillState.ARMED and self._trigger is None:
            self._trigger = KillTrigger.OPERATOR
        return (found, event_id)

    def _read_kill_file(self) -> Mapping[str, Any] | None:
        path = self._state_dir / KILL_FILE
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            loaded = json.loads(text)
        except ValueError:
            return None
        return loaded if isinstance(loaded, dict) else None

    def _write_kill_file(self, ledger_head: str) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "event_id": self._event_id,
            "trigger": self._trigger.value if self._trigger is not None else "",
            "detail": self._detail,
            "ts": self._clock.now().isoformat(),
            "ledger_head": ledger_head,
        }
        path = self._state_dir / KILL_FILE
        tmp = path.with_name(f"{KILL_FILE}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
        directory = os.open(str(self._state_dir), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _ledger_step(self, step: str, detail: str) -> None:
        payload = {
            "event_id": self._event_id,
            "step": step,
            "trigger": self._trigger.value if self._trigger is not None else "",
            "detail": detail,
        }
        self._book.apply(self._ledger.append(LedgerKind.KILL, self._session(), self._clock.now(), payload))

    def _append_rearm(self, *, reset_peak: bool, note: str) -> None:
        _, head = self._ledger.head()
        payload = {
            "event_id": self._event_id,
            "reset_peak": bool(reset_peak),
            "operator_note": note,
            "ledger_head_at_rearm": head,
        }
        self._book.apply(self._ledger.append(LedgerKind.REARM, self._session(), self._clock.now(), payload))

    def _reset_to_armed(self) -> None:
        self._state = KillState.ARMED
        self._event_id = None
        self._trigger = None
        self._detail = ""
        self._cancelled = False
        self._k3_not_before = None
        self._retry_at = None
        self._cooldown_until = None

    def _session(self, view: MarketView | None = None) -> date:
        if view is not None:
            return view.key.session
        key: SnapshotKey | None = self._book.last_key
        if key is not None:
            return key.session
        return self._calendar.prev_or_same_session(self._clock.now().date())

    def _alert(self, message: str) -> None:
        if self._alert_hook is not None:
            self._alert_hook(message)

    # ------------------------------------------------------------------------------------------------------------------
    # backtest semantics (9.5)
    # ------------------------------------------------------------------------------------------------------------------

    def _on_locked(self, view: MarketView) -> None:
        if self._meta.mode is not RunMode.BACKTEST:
            return
        if self._cfg.kill.backtest_behaviour == "stop_run":
            self.run_stopped = True
            return
        self._cooldown_until = self._calendar.next_session(view.key.session, self._cfg.kill.backtest_cooldown_sessions)

    def _maybe_backtest_rearm(self, broker: Broker, view: MarketView | None) -> None:
        """`flatten_and_cooldown`: no entries for `kill.backtest_cooldown_sessions`, then an automatic re-arm WITH the peak
        reset (the explicit peak rule of 9.5). Manual re-arm in paper is different and every report says so."""
        if self._meta.mode is not RunMode.BACKTEST or self._cooldown_until is None or view is None:
            return
        if view.key.session < self._cooldown_until:
            return
        broker.set_suspended(False)
        self._append_rearm(reset_peak=True, note="backtest cooldown elapsed")
        self._reset_to_armed()


def _verdict_payload(verdict: RiskVerdict) -> dict[str, Any]:
    payload: dict[str, Any] = msgspec.to_builtins(verdict)
    payload["candidate"] = None  # a kill close builds no candidate (2.11)
    return payload
