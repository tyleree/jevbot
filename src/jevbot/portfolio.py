"""`Book`: the operational portfolio as a pure fold of the ledger (DESIGN.md 10.8, 2.4, 2.11; `protocols.BookP`, 3.6).

Resume = replay. There are no checkpoints and no second source of truth: every writer does `book.apply(ledger.append(...))`
(the cycle, `reconcile.ingest_fills`, `reconcile.record_order_status`, the kill switch), so `Book.replay(ledger)` reproduces
the live book exactly - including every `Position.entry` field, `structure.last_session`, the hysteresis latch, the text-watch
counter and `day_start_equity`.

Accounting (10.8, 9.2):

* open fill:  `cash[b] -= net[b] * 100 * qty` for each band; a `Position` is created (or grown, for a partial) whose
  `max_loss` comes from `structmath.max_loss_pc` at the ACTUAL worst-band fill and whose `entry` is copied from the OPEN
  intent's `entry_ctx` (folded from the ORDER_INTENT entry: every manage-state field is therefore ledger-sourced);
* close fill: `cash[b] -= close_net[b] * 100 * qty`; the position is reduced / removed and a re-entry cooldown is set;
* `FEE` charges `fee_cents` to all three balances and resets the accumulator; `MARK` carries the per-position liquidation
  values (and, in paper, the broker's equity / prior-session equity / options buying power - the Book's only source for them);
* `SESSION_END`'s headline equity becomes the NEXT session's `day_start_equity` (9.5) - never "the first mark of today", which
  is the mark the daily-loss halt is evaluated on;
* `equity[b] = cash[b] - sum(liq_value * 100 * qty)`; sizing, exits, drawdown and the daily-loss halt read the headline band.
  Cash earns no interest (V8).

The 10.8 signature carries only `initial_cash` and `headline`; `max_loss` / `bp_reserved` need `[fees]` and `[risk]` and the
re-entry cooldown needs a calendar, so `cfg` and `calendar` are additional keyword arguments (both defaulted, so the 10.8 call
site keeps working). Nothing here reads a clock, the environment or a file (import rule 3 of section 1).
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

import msgspec

from jevbot import structmath
from jevbot.cal import XnysCalendar
from jevbot.config import Config
from jevbot.errors import InvariantError
from jevbot.structmath import MULTIPLIER
from jevbot.types import (
    TERMINAL_STATUSES,
    Band,
    BandPrices,
    Cents,
    EntryContext,
    Fill,
    KillState,
    LedgerEntry,
    LedgerKind,
    LegFill,
    Micros,
    OrderIntent,
    OrderPurpose,
    OrderState,
    OrderStatus,
    PortfolioState,
    Position,
    Side,
    Slot,
    SnapshotKey,
    Structure,
)

if TYPE_CHECKING:
    from jevbot.protocols import Calendar, Ledger

__all__ = ["Book"]

# Before the first SESSION_START there is no snapshot key at all (`BookP.last_key` is None there); `PortfolioState.key` is not
# optional, so a state taken that early carries this placeholder. It never reaches a calendar computation.
_NO_KEY: Final = SnapshotKey(session=date.min, slot=Slot.EOD)

_RATE_WINDOW: Final = timedelta(seconds=60)  # check 21 counts submissions in the trailing 60 s (9.1)
_MICROS_PER_CENT: Final = 10_000

# `RISK_EVENT.counters` keys the Book folds into `PortfolioState` (2.11 leaves the counter map open; anything else is ignored)
_COUNTER_FIELDS: Final[frozenset[str]] = frozenset({"opened_today", "jev_fail_sessions", "stale_sessions", "broker_fail_streak"})
_HALT_TYPES: Final[frozenset[str]] = frozenset({"daily_loss_halt", "halt_set"})

# KILL.step (vocab.KILL_STEPS) -> the state the book is in once that step is durable (9.5)
_KILL_STEP_STATE: Final[Mapping[str, str]] = {
    "tripped": "tripped",
    "cancelled": "tripped",
    "close_submitted": "flattening",
    "fallback_legs": "flattening",
    "flat_verified": "flattening",
    "not_flat": "not_flat",
    "suspended": "locked",
    "locked": "locked",
}


def _leg_band(leg: LegFill, band: Band) -> Cents:
    if band is Band.ORATS:
        return leg.orats
    if band is Band.WORST:
        return leg.worst
    return leg.mid


def _blend(old: BandPrices, old_qty: int, new: BandPrices, new_qty: int) -> BandPrices:
    """Quantity-weighted average open price of a position grown by a partial fill, rounded AGAINST us on every band.

    The signed value is ceiled, so a debit is never understated and a credit never overstated (Conventions: + debit / - credit).
    """
    total = old_qty + new_qty
    if total <= 0:
        raise InvariantError(f"portfolio: cannot blend open prices for a total quantity of {total}")

    def avg(a: int, b: int) -> int:
        numerator = a * old_qty + b * new_qty
        return -((-numerator) // total)

    return BandPrices(orats=avg(old.orats, new.orats), worst=avg(old.worst, new.worst), mid=avg(old.mid, new.mid))


class Book:
    """Implements `protocols.BookP` (3.6) - see the module docstring."""

    def __init__(
        self,
        *,
        initial_cash: Cents,
        headline: Band,
        cfg: Config | None = None,
        calendar: Calendar | None = None,
    ) -> None:
        if isinstance(initial_cash, bool) or not isinstance(initial_cash, int):
            raise InvariantError(f"initial_cash must be an int (cents), got {type(initial_cash).__name__}")
        self._cfg: Final = cfg if cfg is not None else Config()
        self._headline: Final[Band] = Band(headline)
        self._calendar: Final[Calendar] = calendar if calendar is not None else XnysCalendar()
        self._initial_cash = int(initial_cash)
        self._cash: dict[Band, int] = dict.fromkeys(Band, self._initial_cash)
        self._positions: dict[str, Position] = {}
        self._intents: dict[str, OrderIntent] = {}
        self._orders: dict[str, OrderState] = {}  # client_order_id -> latest state
        self._order_intent: dict[str, str] = {}  # client_order_id -> intent_id
        self._filled: dict[str, int] = {}  # client_order_id -> quantity BOOKED by FILL entries (9.6)
        self._key: SnapshotKey | None = None
        self._session: date | None = None
        self._last_as_of: datetime | None = None
        self._submits: list[datetime] = []
        self._peak_equity: Cents = self._initial_cash
        self._day_start_equity: Cents = self._initial_cash
        self._opened_today = 0
        self._fees_accrued_micro: Micros = 0
        self._halt_entries = False
        self._halt_reasons: tuple[str, ...] = ()
        self._kill_state = "armed"
        self._kill_event_id: str | None = None
        self._cooldowns: dict[tuple[str, str], date] = {}
        self._jev_fail_sessions = 0
        self._stale_sessions = 0
        self._broker_fail_streak = 0
        self._broker_equity: Cents | None = None
        self._broker_prev_equity: Cents | None = None
        self._broker_options_bp: Cents | None = None

    # ------------------------------------------------------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def replay(
        cls,
        ledger: Ledger,
        *,
        initial_cash: Cents,
        headline: Band,
        cfg: Config | None = None,
        calendar: Calendar | None = None,
    ) -> Book:
        """Fold the whole ledger into a fresh Book: resume = replay (10.8, INV-24)."""
        book = cls(initial_cash=initial_cash, headline=headline, cfg=cfg, calendar=calendar)
        for entry in ledger.entries():
            book.apply(entry)
        return book

    # ------------------------------------------------------------------------------------------------------------------
    # BookP
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def last_key(self) -> SnapshotKey | None:
        """The last snapshot key folded in (None before the first SESSION_START); a resume starts after it (10.1)."""
        return self._key

    @property
    def headline(self) -> Band:
        """The band sizing, exits, drawdown and the daily-loss halt read (10.8)."""
        return self._headline

    def apply(self, entry: LedgerEntry) -> None:
        """THE only mutator. Every entry kind the book folds is listed in 10.8; anything else is ignored by design."""
        if not isinstance(entry, LedgerEntry):
            raise InvariantError(f"Book.apply takes a LedgerEntry, got {type(entry).__name__}")
        self._last_as_of = entry.as_of
        handler = _HANDLERS.get(entry.kind)
        if handler is not None:
            handler(self, entry)

    def state(self) -> PortfolioState:
        cash = BandPrices(orats=self._cash[Band.ORATS], worst=self._cash[Band.WORST], mid=self._cash[Band.MID])
        positions = tuple(self._positions[pid] for pid in sorted(self._positions))
        equity = BandPrices(
            orats=self._equity(Band.ORATS),
            worst=self._equity(Band.WORST),
            mid=self._equity(Band.MID),
        )
        working = tuple(self._intents[iid] for iid in sorted(self._working_intent_ids()))
        cooldowns = tuple((u, d, when) for (u, d), when in sorted(self._cooldowns.items()))
        return PortfolioState(
            key=self._key if self._key is not None else _NO_KEY,
            cash=cash,
            equity=equity,
            positions=positions,
            working=working,
            peak_equity=self._peak_equity,
            day_start_equity=self._day_start_equity,
            opened_today=self._opened_today,
            fees_accrued_micro=self._fees_accrued_micro,
            halt_entries=self._halt_entries,
            halt_reasons=self._halt_reasons,
            kill_state=KillState(self._kill_state),
            kill_event_id=self._kill_event_id,
            cooldowns=cooldowns,
            jev_fail_sessions=self._jev_fail_sessions,
            stale_sessions=self._stale_sessions,
            broker_fail_streak=self._broker_fail_streak,
            orders_last_minute=self._orders_last_minute(),
            broker_equity=self._broker_equity,
            broker_prev_equity=self._broker_prev_equity,
            broker_options_bp=self._broker_options_bp,
        )

    def intent(self, intent_id: str) -> OrderIntent:
        found = self._intents.get(intent_id)
        if found is None:
            raise InvariantError(f"no ORDER_INTENT with intent_id {intent_id!r} was folded into the book")
        return found

    def has_intent(self, intent_id: str) -> bool:
        return intent_id in self._intents

    def filled_qty(self, client_order_id: str) -> int:
        """The cumulative quantity BOOKED by FILL entries for that order (9.6).

        It is deliberately not the last ORDER_STATUS's `filled_qty`: the submit protocol records the broker's cumulative
        quantity as soon as it is known, and `ingest_fills` measures its delta against what has actually been booked -
        `delta = st.filled_qty - book.filled_qty(cid)`. The two differ exactly while a fill is known but not yet booked.
        """
        return self._filled.get(client_order_id, 0)

    def open_orders(self) -> tuple[tuple[OrderIntent, OrderState], ...]:
        """(intent, state) for every order whose latest status is not terminal, sorted by client_order_id.

        An intent that reached the ledger but never reached SUBMITTING (a crash in between, 9.6) has no ORDER_STATUS row; it
        is reported with a synthetic `INTENT` state under the attempt-0 client id, which is what the order worker resumes.
        """
        out: list[tuple[OrderIntent, OrderState]] = []
        for cid, state in self._orders.items():
            if state.status not in TERMINAL_STATUSES:
                out.append((self._intents[self._order_intent[cid]], state))
        for intent_id, intent in self._intents.items():
            if not any(self._order_intent[cid] == intent_id for cid in self._orders):
                out.append((intent, self._intent_state(intent)))
        return tuple(sorted(out, key=lambda pair: pair[1].client_order_id))

    def leg_positions(self) -> dict[str, int]:
        """occ -> signed contracts (+ long / - short) over every open position."""
        legs: dict[str, int] = {}
        for pos in self._positions.values():
            for leg in pos.structure.legs:
                signed = pos.qty * leg.ratio * (1 if leg.side is Side.BUY else -1)
                legs[leg.contract.occ] = legs.get(leg.contract.occ, 0) + signed
        return {occ: qty for occ, qty in sorted(legs.items()) if qty != 0}

    # ------------------------------------------------------------------------------------------------------------------
    # read-only conveniences (not part of BookP)
    # ------------------------------------------------------------------------------------------------------------------

    def position(self, position_id: str) -> Position | None:
        return self._positions.get(position_id)

    def order_state(self, client_order_id: str) -> OrderState | None:
        return self._orders.get(client_order_id)

    def intent_of_order(self, client_order_id: str) -> OrderIntent | None:
        intent_id = self._order_intent.get(client_order_id)
        return None if intent_id is None else self._intents[intent_id]

    # ------------------------------------------------------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------------------------------------------------------

    def _equity(self, band: Band) -> Cents:
        held = sum(pos.liq_value * MULTIPLIER * pos.qty for pos in self._positions.values())
        return self._cash[band] - held

    def _orders_last_minute(self) -> int:
        if self._last_as_of is None:
            return 0
        floor = self._last_as_of - _RATE_WINDOW
        return sum(1 for when in self._submits if when > floor)

    def _working_intent_ids(self) -> set[str]:
        working = {self._order_intent[cid] for cid, st in self._orders.items() if st.status not in TERMINAL_STATUSES}
        seen = {self._order_intent[cid] for cid in self._orders}
        working |= {iid for iid in self._intents if iid not in seen}
        return working

    def _intent_state(self, intent: OrderIntent) -> OrderState:
        return OrderState(
            client_order_id=f"{intent.intent_id}-00",
            broker_order_id=None,
            status=OrderStatus.INTENT,
            qty=intent.qty,
            filled_qty=0,
            filled_net=None,
            reject_code=None,
            message=None,
            updated_at=self._last_as_of if self._last_as_of is not None else datetime.min.replace(tzinfo=_dt.UTC),
        )

    def _fee_round_trip(self, structure: Structure, legs: Iterable[LegFill]) -> Cents:
        prices = [abs(_leg_band(leg, self._headline)) for leg in legs]
        n_legs = len(structure.legs)
        if len(prices) != n_legs:
            raise InvariantError(f"portfolio: {len(prices)} leg fills for a {n_legs}-leg {structure.kind.value}")
        sells = sum(1 for leg in structure.legs if leg.side is Side.SELL)
        return structmath.fee_round_trip(n_legs, sells, prices, self._cfg.fees)

    # --- entry handlers -------------------------------------------------------------------------------------------------

    def _on_run_start(self, entry: LedgerEntry) -> None:
        cash = entry.payload.get("initial_cash")
        if isinstance(cash, int) and not isinstance(cash, bool):
            self._initial_cash = cash
            self._cash = dict.fromkeys(Band, cash)
            self._peak_equity = cash
            self._day_start_equity = cash

    def _on_session_start(self, entry: LedgerEntry) -> None:
        session = _as_date(entry.payload.get("session"), entry.session)
        slot = Slot(str(entry.payload.get("slot", Slot.EOD.value)))
        if session != self._session:
            self._session = session
            self._opened_today = 0
            self._halt_entries = False
            self._halt_reasons = ()
        self._key = SnapshotKey(session=session, slot=slot)

    def _on_decision(self, entry: LedgerEntry) -> None:
        if entry.payload.get("kind") != "manage":
            return
        rules = entry.payload.get("rules")
        if not isinstance(rules, Mapping):
            return
        position_id = rules.get("position_id")
        pos = self._positions.get(position_id) if isinstance(position_id, str) else None
        if pos is None:
            return
        latch = rules.get("exit_latch", pos.exit_latch)
        watch = rules.get("watch_text", pos.watch_text)
        self._positions[pos.position_id] = msgspec.structs.replace(
            pos,
            exit_latch=bool(latch),
            watch_text=int(watch) if isinstance(watch, int) and not isinstance(watch, bool) else pos.watch_text,
        )

    def _on_order_intent(self, entry: LedgerEntry) -> None:
        intent = msgspec.convert(entry.payload, OrderIntent)
        self._intents[intent.intent_id] = intent

    def _on_order_status(self, entry: LedgerEntry) -> None:
        payload = entry.payload
        cid = str(payload["client_order_id"])
        intent_id = str(payload["intent_id"])
        status = OrderStatus(str(payload["status"]))
        self._order_intent[cid] = intent_id
        self._orders[cid] = OrderState(
            client_order_id=cid,
            broker_order_id=_opt_str(payload.get("broker_order_id")),
            status=status,
            qty=int(payload["qty"]),
            filled_qty=int(payload["filled_qty"]),
            filled_net=None,
            reject_code=_opt_int(payload.get("reject_code")),
            message=_opt_str(payload.get("tag")) or None,
            updated_at=entry.as_of,
        )
        if status is OrderStatus.SUBMITTING:
            self._submits.append(entry.as_of)
            floor = entry.as_of - _RATE_WINDOW
            self._submits = [when for when in self._submits if when > floor]

    def _on_fill(self, entry: LedgerEntry) -> None:
        payload = {k: v for k, v in entry.payload.items() if k != "realised_pnl"}
        fill = msgspec.convert(payload, Fill)
        for band in Band:
            self._cash[band] -= fill.net.get(band) * MULTIPLIER * fill.qty
        self._fees_accrued_micro += fill.fees_micro
        self._filled[fill.client_order_id] = self._filled.get(fill.client_order_id, 0) + fill.qty
        if fill.purpose is OrderPurpose.OPEN:
            self._open_fill(fill)
        else:
            self._close_fill(fill, entry.session)

    def _open_fill(self, fill: Fill) -> None:
        intent = self._intents.get(fill.intent_id)
        if intent is None:
            raise InvariantError(f"FILL {fill.fill_id}: no ORDER_INTENT {fill.intent_id!r} was folded before it (10.8)")
        structure, entry_ctx = intent.structure, intent.entry_ctx
        if structure is None or entry_ctx is None:
            raise InvariantError(f"OPEN intent {intent.intent_id!r} carries no structure / entry_ctx (2.4)")
        existing = self._positions.get(fill.position_id)
        if existing is None:
            qty, open_net = fill.qty, fill.net
            self._opened_today += 1
        else:
            qty = existing.qty + fill.qty
            open_net = _blend(existing.open_net, existing.qty, fill.net, fill.qty)
        widths = structure.wing_widths
        fee_rt = self._fee_round_trip(structure, fill.legs)
        max_profit_pc = structmath.max_profit_pc(structure.kind, widths, open_net.get(self._headline))
        self._positions[fill.position_id] = Position(
            position_id=fill.position_id,
            structure=structure,
            qty=qty,
            open_key=intent.key,
            open_decision_id=fill.decision_id,
            open_net=open_net,
            max_loss=structmath.max_loss_pc(structure.kind, widths, open_net.worst, fee_rt) * qty,
            max_profit=None if max_profit_pc is None else max_profit_pc * qty,
            bp_reserved=structmath.bp_required_pc(structure.kind, widths, open_net.worst, fee_rt, self._cfg.risk) * qty,
            entry=_entry_context(entry_ctx),
            exit_latch=existing.exit_latch if existing is not None else False,
            watch_text=existing.watch_text if existing is not None else 0,
            liq_value=existing.liq_value if existing is not None else 0,
            mid_value=existing.mid_value if existing is not None else 0,
            stale_marks=existing.stale_marks if existing is not None else 0,
        )

    def _close_fill(self, fill: Fill, session: date) -> None:
        pos = self._positions.get(fill.position_id)
        if pos is None:
            return  # a kill close of a broker leg the ledger never knew (9.5 K3): cash is booked, there is nothing to reduce
        remaining = pos.qty - fill.qty
        if remaining > 0:
            scale_num, scale_den = remaining, pos.qty
            self._positions[fill.position_id] = msgspec.structs.replace(
                pos,
                qty=remaining,
                max_loss=pos.max_loss * scale_num // scale_den,
                max_profit=None if pos.max_profit is None else pos.max_profit * scale_num // scale_den,
                bp_reserved=pos.bp_reserved * scale_num // scale_den,
            )
            return
        del self._positions[fill.position_id]
        self._set_cooldown(pos.structure, session)

    def _set_cooldown(self, structure: Structure, session: date) -> None:
        sessions = self._cfg.rules.reentry_cooldown_sessions
        if sessions <= 0:
            return
        first_allowed = self._calendar.next_session(session, sessions)
        key = (structure.underlying, structure.direction.value)
        current = self._cooldowns.get(key)
        if current is None or current < first_allowed:
            self._cooldowns[key] = first_allowed

    def _on_mark(self, entry: LedgerEntry) -> None:
        payload = entry.payload
        marks = payload.get("positions")
        if isinstance(marks, Mapping):
            for position_id, values in marks.items():
                pos = self._positions.get(position_id)
                if pos is None or not isinstance(values, Mapping):
                    continue
                stale = bool(values.get("stale", False))
                self._positions[position_id] = msgspec.structs.replace(
                    pos,
                    liq_value=int(values.get("liq_value", pos.liq_value)),
                    mid_value=int(values.get("mid_value", pos.mid_value)),
                    stale_marks=pos.stale_marks + 1 if stale else 0,
                )
        self._broker_equity = _opt_int(payload.get("broker_equity"))
        self._broker_prev_equity = _opt_int(payload.get("broker_prev_equity"))
        self._broker_options_bp = _opt_int(payload.get("broker_options_bp"))
        self._peak_equity = max(self._peak_equity, self._equity(self._headline))

    def _on_fee(self, entry: LedgerEntry) -> None:
        fee_cents = int(entry.payload["fee_cents"])
        for band in Band:
            self._cash[band] -= fee_cents
        self._fees_accrued_micro = 0

    def _on_risk_event(self, entry: LedgerEntry) -> None:
        payload = entry.payload
        kind = str(payload.get("type", ""))
        if kind in _HALT_TYPES:
            reason = str(payload.get("detail") or payload.get("trigger") or kind)
            self._halt_entries = True
            if reason not in self._halt_reasons:
                self._halt_reasons = (*self._halt_reasons, reason)
        elif kind == "halt_cleared":
            self._halt_entries = False
            self._halt_reasons = ()
        counters = payload.get("counters")
        if isinstance(counters, Mapping):
            for name, value in counters.items():
                if name in _COUNTER_FIELDS and isinstance(value, int) and not isinstance(value, bool):
                    setattr(self, f"_{name}", value)

    def _on_kill(self, entry: LedgerEntry) -> None:
        step = str(entry.payload.get("step", ""))
        state = _KILL_STEP_STATE.get(step)
        if state is None:
            return
        self._kill_state = state
        event_id = entry.payload.get("event_id")
        if isinstance(event_id, str) and event_id:
            self._kill_event_id = event_id

    def _on_rearm(self, entry: LedgerEntry) -> None:
        self._kill_state = "armed"
        self._kill_event_id = None
        if bool(entry.payload.get("reset_peak", False)):
            self._peak_equity = self._equity(self._headline)

    def _on_session_end(self, entry: LedgerEntry) -> None:
        equity = entry.payload.get("equity")
        if isinstance(equity, Mapping):
            value = equity.get(self._headline.value)
            if isinstance(value, int) and not isinstance(value, bool):
                self._day_start_equity = value
                self._peak_equity = max(self._peak_equity, value)


def _entry_context(ctx: EntryContext) -> EntryContext:
    """A defensive copy: `EntryContext.entry_codes` is a mutable dict on a frozen struct."""
    return msgspec.structs.replace(ctx, entry_codes=dict(ctx.entry_codes))


def _as_date(value: Any, fallback: date) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    return fallback


def _opt_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


_HANDLERS: Final[Mapping[LedgerKind, Any]] = {
    LedgerKind.RUN_START: Book._on_run_start,
    LedgerKind.SESSION_START: Book._on_session_start,
    LedgerKind.DECISION: Book._on_decision,
    LedgerKind.ORDER_INTENT: Book._on_order_intent,
    LedgerKind.ORDER_STATUS: Book._on_order_status,
    LedgerKind.FILL: Book._on_fill,
    LedgerKind.MARK: Book._on_mark,
    LedgerKind.FEE: Book._on_fee,
    LedgerKind.RISK_EVENT: Book._on_risk_event,
    LedgerKind.KILL: Book._on_kill,
    LedgerKind.REARM: Book._on_rearm,
    LedgerKind.SESSION_END: Book._on_session_end,
}


if TYPE_CHECKING:
    from jevbot.protocols import BookP

    _BOOK_IS_A_BOOKP: BookP = Book(initial_cash=0, headline=Band.ORATS)
