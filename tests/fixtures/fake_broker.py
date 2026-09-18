"""`FakeBroker` - the protocol-level `Broker` double of DESIGN.md 15.4 (owned by WP05, consumed read-only by later waves).

It implements `protocols.Broker` (3.4) over plain dictionaries and a scripted quote tape. It is **ledger-free**, like every
real broker: it holds no Ledger and no Book and never writes an entry - `reconcile.record_order_status` does that around
`submit()`, which is what makes the ledger trail identical for `FakeBroker`, `SimBroker` and `AlpacaPaperBroker` (9.6).

What it models (15.4):

* orders keyed by `client_order_id`; a duplicate id returns the EXISTING order (idempotent submit, D18), unless
  `duplicate_accepted_twice=True` - the switch that lets a test exercise deviation V4's assumption;
* a fill happens only when the order's limit is **marketable** against the quote tape (G2): the natural (worst-band) price of
  the order's legs is `sum(ask of the legs we buy) - sum(bid of the legs we sell)` and a limit is marketable when it is at
  least that (a market order always is). Nothing fills on a quote the tape does not carry;
* mleg fills are unit-atomic (whole contracts across every leg); a seeded `partial_rate` fills part of the quantity first;
* positions are signed per OCC symbol and may include an injected EQUITY position (an assignment);
* `activities()` reports OPASN / OPEXP for the NEXT day, as the real one does;
* scripted faults per call: `timeout_after_accept`, `504_after_accept` (accepted server-side, `BrokerAmbiguous` to us),
  `connection_reset` (not accepted), `403_bp` / `422_validation` (definitive `BrokerRejected`), `hang` (deadline),
  `late_post` (an abandoned call that lands on the next lookup);
* `suspended` blocks closing orders too, which is why the kill sequence suspends strictly last (9.5 K6).

Nothing here reads a clock, the network or a file; `now` is whatever the caller sets with `set_now()`.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Final

import msgspec

from jevbot import occ
from jevbot.errors import BrokerAmbiguous, BrokerRejected
from jevbot.types import (
    TERMINAL_STATUSES,
    AccountSnapshot,
    ApprovedOrder,
    BrokerActivity,
    BrokerPosition,
    OrderState,
    OrderStatus,
    Side,
)

if TYPE_CHECKING:
    from jevbot.protocols import MarketView

__all__ = ["FAULTS", "FakeBroker", "quote_tape"]

FAULTS: Final[tuple[str, ...]] = (
    "timeout_after_accept",
    "504_after_accept",
    "connection_reset",
    "403_bp",
    "422_validation",
    "hang",
    "late_post",
)

_REJECT_CODES: Final[Mapping[str, tuple[int, int, str]]] = {  # fault -> (http status, broker code, coarse tag)
    "403_bp": (403, 40310000, "insufficient_buying_power"),
    "422_validation": (422, 42210000, "validation"),
}


def quote_tape(chain: object, symbols: Iterable[str]) -> dict[str, tuple[int, int]]:
    """`{occ: (bid, ask)}` read off a `ChainSnapshot`-like object for the given symbols (a convenience for tests)."""
    table = getattr(chain, "table", None)
    if table is None:
        raise TypeError("quote_tape needs a ChainSnapshot")
    wanted = set(symbols)
    rows = table[table["occ"].isin(list(wanted))]
    return {str(row.occ): (int(row.bid), int(row.ask)) for row in rows.itertuples()}


@dataclass
class _Order:
    state: OrderState
    order: ApprovedOrder
    visible: bool = True  # False for a `late_post`: accepted, but the next lookup is the first that sees it


@dataclass
class FakeBroker:
    """Implements `protocols.Broker` (3.4) - see the module docstring."""

    quotes: dict[str, tuple[int, int]] = field(default_factory=dict)
    positions_by_symbol: dict[str, int] = field(default_factory=dict)
    equity: int = 10_000_000
    cash: int = 10_000_000
    options_buying_power: int = 5_000_000
    last_equity: int | None = None
    options_level: int = 3
    trading_blocked: bool = False
    account_blocked: bool = False
    suspended: bool = False
    seed: int = 0
    partial_rate: float = 0.0
    duplicate_accepted_twice: bool = False
    refuse_fills: bool = False
    faults: dict[str, list[str]] = field(default_factory=dict)
    scripted_activities: list[BrokerActivity] = field(default_factory=list)
    now: datetime = datetime(2024, 5, 17, tzinfo=UTC)
    calls: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._orders: dict[str, _Order] = {}
        self._rng = random.Random(self.seed)

    # ------------------------------------------------------------------------------------------------------------------
    # test controls
    # ------------------------------------------------------------------------------------------------------------------

    def set_now(self, when: datetime) -> None:
        self.now = when

    def set_quotes(self, quotes: Mapping[str, tuple[int, int]]) -> None:
        self.quotes.update(dict(quotes))

    def set_position(self, symbol: str, qty: int) -> None:
        if qty == 0:
            self.positions_by_symbol.pop(symbol, None)
        else:
            self.positions_by_symbol[symbol] = qty

    def script_fault(self, call: str, fault: str) -> None:
        if fault not in FAULTS:
            raise ValueError(f"unknown fault {fault!r}; known: {FAULTS}")
        self.faults.setdefault(call, []).append(fault)

    def force_fill(self, client_order_ids: Sequence[str] = ()) -> None:
        """Fill every (or the named) working order whatever the tape says - the escape hatch beside `refuse_fills`."""
        wanted = set(client_order_ids)
        for cid, record in self._orders.items():
            if (wanted and cid not in wanted) or record.state.status in TERMINAL_STATUSES:
                continue
            self._book(record, record.order.qty - record.state.filled_qty)

    def order_state(self, client_order_id: str) -> OrderState | None:
        found = self._orders.get(client_order_id)
        return None if found is None else found.state

    def _pop_fault(self, call: str) -> str | None:
        queue = self.faults.get(call)
        return queue.pop(0) if queue else None

    # ------------------------------------------------------------------------------------------------------------------
    # Broker protocol
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "fake"

    def account(self) -> AccountSnapshot:
        self.calls.append("account")
        return AccountSnapshot(
            equity=self.equity,
            cash=self.cash,
            options_buying_power=self.options_buying_power,
            last_equity=self.last_equity,
            options_level=self.options_level,
            trading_blocked=self.trading_blocked,
            account_blocked=self.account_blocked,
            suspended=self.suspended,
            ts=self.now,
        )

    def positions(self) -> tuple[BrokerPosition, ...]:
        self.calls.append("positions")
        return tuple(
            BrokerPosition(symbol=symbol, qty=qty, is_option=_is_option(symbol))
            for symbol, qty in sorted(self.positions_by_symbol.items())
            if qty != 0
        )

    def open_orders(self) -> tuple[OrderState, ...]:
        self.calls.append("open_orders")
        return tuple(
            record.state for _, record in sorted(self._orders.items()) if record.visible and record.state.status not in TERMINAL_STATUSES
        )

    def submit(self, order: ApprovedOrder) -> OrderState:
        self.calls.append(f"submit:{order.client_order_id}")
        if self.suspended:
            raise BrokerRejected("account is suspended", status=403, reject_code=40310000, tag="suspended")
        existing = self._orders.get(order.client_order_id)
        if existing is not None and not self.duplicate_accepted_twice:
            return existing.state  # idempotent on client_order_id (D18): the same order is adopted, never duplicated
        fault = self._pop_fault("submit") or ""
        if fault in _REJECT_CODES:
            status, code, tag = _REJECT_CODES[fault]
            raise BrokerRejected(f"fake broker {fault}", status=status, reject_code=code, tag=tag)
        if fault == "connection_reset":
            raise BrokerAmbiguous("connection reset before the order was accepted")
        accepted = self._accept(order, visible=fault != "late_post")
        if fault in ("timeout_after_accept", "504_after_accept", "hang"):
            raise BrokerAmbiguous(f"fake broker {fault}: the order was accepted, the response was not received")
        return accepted

    def get_order(self, client_order_id: str) -> OrderState | None:
        self.calls.append(f"get_order:{client_order_id}")
        record = self._orders.get(client_order_id)
        if record is None:
            return None
        if not record.visible:
            record.visible = True  # a late POST: this lookup is the first that sees it
            return None
        return record.state

    def cancel(self, client_order_id: str) -> OrderState:
        self.calls.append(f"cancel:{client_order_id}")
        record = self._orders.get(client_order_id)
        if record is None:
            raise BrokerRejected(f"no order {client_order_id}", status=404, tag="not_found")
        if record.state.status not in TERMINAL_STATUSES:
            record.state = _replace(record.state, status=OrderStatus.CANCELLED, updated_at=self.now)
        return record.state

    def cancel_all(self) -> int:
        self.calls.append("cancel_all")
        cancelled = 0
        for record in self._orders.values():
            if record.state.status not in TERMINAL_STATUSES:
                record.state = _replace(record.state, status=OrderStatus.CANCELLED, updated_at=self.now)
                cancelled += 1
        return cancelled

    def activities(self, since: date) -> tuple[BrokerActivity, ...]:
        self.calls.append("activities")
        return tuple(a for a in self.scripted_activities if a.day >= since)

    def set_suspended(self, suspended: bool) -> None:
        self.calls.append(f"set_suspended:{suspended}")
        self.suspended = bool(suspended)

    def on_snapshot(self, view: MarketView) -> None:
        del view  # a real broker prices nothing for us; only SimBroker does

    # ------------------------------------------------------------------------------------------------------------------
    # fills
    # ------------------------------------------------------------------------------------------------------------------

    def _accept(self, order: ApprovedOrder, *, visible: bool) -> OrderState:
        state = OrderState(
            client_order_id=order.client_order_id,
            broker_order_id=f"fake-{len(self._orders) + 1:04d}",
            status=OrderStatus.SUBMITTED,
            qty=order.qty,
            filled_qty=0,
            filled_net=None,
            reject_code=None,
            message=None,
            updated_at=self.now,
        )
        self._orders[order.client_order_id] = _Order(state=state, order=order, visible=visible)
        if visible:
            self._maybe_fill(order.client_order_id)
        return self._orders[order.client_order_id].state

    def _maybe_fill(self, client_order_id: str) -> None:
        record = self._orders[client_order_id]
        order = record.order
        if self.refuse_fills or not self._marketable(order):
            return
        if order.intent.equity_symbol is not None:
            self._fill_equity(record)
            return
        qty = order.qty
        if self.partial_rate > 0 and qty > 1 and self._rng.random() < self.partial_rate:
            qty = max(1, qty // 2)
        self._book(record, qty)

    def _marketable(self, order: ApprovedOrder) -> bool:
        if order.limit is None:
            return True  # a market order
        natural = self._natural(order)
        return natural is not None and order.limit >= natural

    def _natural(self, order: ApprovedOrder) -> int | None:
        """The worst-band natural of the order's legs: buy at the ask, sell at the bid (10.3)."""
        net = 0
        for leg in order.intent.legs:
            quote = self.quotes.get(leg.contract.occ)
            if quote is None:
                return None
            bid, ask = quote
            net += ask * leg.ratio if leg.side is Side.BUY else -bid * leg.ratio
        return net

    def _book(self, record: _Order, qty: int) -> None:
        order = record.order
        for leg in order.intent.legs:
            signed = qty * leg.ratio * (1 if leg.side is Side.BUY else -1)
            self.set_position(leg.contract.occ, self.positions_by_symbol.get(leg.contract.occ, 0) + signed)
        filled = record.state.filled_qty + qty
        record.state = _replace(
            record.state,
            status=OrderStatus.FILLED if filled >= order.qty else OrderStatus.PARTIAL,
            filled_qty=filled,
            filled_net=order.limit,
            updated_at=self.now,
        )

    def _fill_equity(self, record: _Order) -> None:
        intent = record.order.intent
        symbol = intent.equity_symbol or ""
        shares = intent.equity_qty or 0
        signed = shares if intent.equity_side is Side.BUY else -shares
        self.set_position(symbol, self.positions_by_symbol.get(symbol, 0) + signed)
        record.state = _replace(record.state, status=OrderStatus.FILLED, filled_qty=0, updated_at=self.now)


def _is_option(symbol: str) -> bool:
    return occ.is_occ(symbol)


def _replace(state: OrderState, **changes: object) -> OrderState:
    return msgspec.structs.replace(state, **changes)


if TYPE_CHECKING:
    # Static proof, checked by mypy: FakeBroker implements the Broker Protocol of 3.4.
    from jevbot.protocols import Broker

    _FAKE_BROKER_IS_A_BROKER: Broker = FakeBroker()
