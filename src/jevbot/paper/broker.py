"""`AlpacaPaperBroker`: the `Broker` (3.4) adapter over the paper trading client (DESIGN.md 11.5; D18, G5; INV-03, INV-08, INV-09).

Three properties define this module.

**Ledger-free.** The adapter holds no `Ledger` and no `Book` and writes no entry. The durable `ORDER_STATUS{SUBMITTING}`
before a call and the `SUBMITTED` / `REJECTED` / `UNKNOWN` entry after it are written by the CALLER through
`reconcile.record_order_status` (the submit protocol of 9.6), identically for `SimBroker`, `FakeBroker` and this class. That
is what makes the ledger trail independent of which broker is plugged in - and it is asserted by
`tests/unit/test_paper_broker.py::test_the_adapter_is_ledger_free`.

**Every call is bounded twice** (INV-08): the `(connect, read)` timeout installed on the session by `alpaca_client`, AND a
hard wall-clock deadline (`orders.call_deadline_s`) enforced by running the call in a worker thread that the adapter simply
stops waiting for. A read may be retried under OUR policy; a write never is. A token bucket keeps the REST rate under
`orders.rest_calls_per_minute`.

**Submission is idempotent on the client order id** (D18, INV-07). `submit()` looks the id up BEFORE it posts, so a retry
after a crash adopts the existing order instead of opening the position twice. When the outcome of a POST is unknown
(timeout, 5xx, 429, deadline) the adapter looks the id up a few times over ~10 s; if the order is still not there it raises
`BrokerAmbiguous` and - by default (V4) - does NOT post again: the caller ledgers `UNKNOWN`, entries are abandoned for the
session and exits are re-issued by the next cycle under a fresh attempt-stamped id. A late POST from an abandoned deadline
worker is harmless by construction: it carries our client id, so reconcile R1 adopts it, CANCEL_ALL cancels it if it is still
resting, and any fill is booked through the one fill path (9.6).

`replace_order_by_id` is never called: repricing is cancel + confirm terminal + a fresh `approve()` + submit with
`attempt + 1` (INV-09).
"""

import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Final, TypeVar

from alpaca.common.exceptions import APIError
from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest, OptionLegRequest, OrderRequest

from jevbot.config import OrdersConfig
from jevbot.errors import BrokerAmbiguous, BrokerError, BrokerRejected, InvariantError
from jevbot.occ import format_occ, is_occ
from jevbot.paper.alpaca_client import AlpacaClients, account_snapshot
from jevbot.paper.clock import RawClock, to_utc
from jevbot.types import (
    AccountSnapshot,
    ApprovedOrder,
    BrokerActivity,
    BrokerPosition,
    OrderIntent,
    OrderState,
    OrderStatus,
    Side,
)

__all__ = [
    "ACTIVITY_TYPES",
    "AMBIGUOUS_LOOKUP_WINDOW_S",
    "CANCEL_TERMINAL_DEADLINE_S",
    "TERMINAL_STATUSES",
    "AlpacaPaperBroker",
    "build_order_request",
    "to_broker_position",
    "to_order_state",
]

BROKER_NAME: Final = "alpaca_paper"

TERMINAL_STATUSES: Final[frozenset[OrderStatus]] = frozenset(
    {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
)

AMBIGUOUS_LOOKUP_WINDOW_S: Final = 10.0
"""11.5 step 4: the lookups after an ambiguous write are spread over about ten seconds."""

CANCEL_TERMINAL_DEADLINE_S: Final = 10.0
"""11.5: `cancel` polls for a terminal status for at most this long; a cancel that races a fill is resolved by reading."""

ACTIVITY_TYPES: Final[tuple[str, ...]] = ("OPASN", "OPEXC", "OPEXP", "OPTRD")
"""The activity types reconcile R2 acts on (assignment / exercise / expiry / option trade)."""

READ_BACKOFF_S: Final[tuple[float, ...]] = (0.2, 0.8)
"""Our own read backoff; the SDK's blind retry loop is disabled at construction (G5)."""

# 11.5 step 4: the outcome of a call that ended this way is UNKNOWN, never "rejected".
AMBIGUOUS_STATUSES: Final[frozenset[int]] = frozenset({408, 429})

_ACTIVITY_PAGE_SIZE: Final = 100
_ACTIVITY_MAX_PAGES: Final = 50
_CONTRACTS_PER_CALL: Final = 1
_T = TypeVar("_T")

# Coarse DIAGNOSTIC tags (critique correction 4): logged and ledgered, never branched on beyond the tag itself.
_REJECT_TAGS: Final[tuple[tuple[str, str], ...]] = (
    ("buying power", "insufficient_buying_power"),
    ("insufficient", "insufficient"),
    ("not tradable", "not_tradable"),
    ("market is closed", "market_closed"),
    ("wash trade", "wash_trade"),
    ("duplicate", "duplicate_client_order_id"),
    ("option level", "options_level"),
    ("trading level", "options_level"),
    ("limit price", "limit_price"),
    ("not found", "not_found"),
    ("forbidden", "forbidden"),
)

_STATUS_MAP: Final[dict[str, OrderStatus]] = {
    "new": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "accepted_for_bidding": OrderStatus.SUBMITTED,
    "pending_review": OrderStatus.SUBMITTED,
    "calculated": OrderStatus.SUBMITTED,
    "held": OrderStatus.SUBMITTED,
    "stopped": OrderStatus.SUBMITTED,
    "suspended": OrderStatus.SUBMITTED,
    "pending_cancel": OrderStatus.SUBMITTED,
    "pending_replace": OrderStatus.SUBMITTED,
    "done_for_day": OrderStatus.SUBMITTED,  # not terminal for us: CANCEL_ALL still cancels it and confirms the final state
    "partially_filled": OrderStatus.PARTIAL,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "replaced": OrderStatus.CANCELLED,  # we never replace (INV-09); if one ever appears the order no longer works
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
}


# ======================================================================================================================
# Vendor -> our types
# ======================================================================================================================


def _decimal(value: object, what: str) -> Decimal:
    try:
        return Decimal(str(value))
    except InvalidOperation:
        raise BrokerError(f"the broker reported {what}={value!r}, which is not a number") from None


def _whole(value: object, what: str) -> int:
    """A vendor quantity -> whole units, truncated toward zero.

    Option and assigned-share quantities are integral by construction (no fractional order is ever sent, and an assignment
    moves whole lots), so this only ever drops a zero fraction; truncating toward zero keeps a flatten from over-ordering.
    """
    return int(_decimal(value, what))


def _price_cents(value: object, what: str) -> int:
    """A vendor price in dollars -> signed cents per share. Plumbing only: no P&L is ever computed from it (2.4)."""
    return int((_decimal(value, what) * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _model(value: Any, what: str) -> Any:
    """Unwrap an SDK response that must be a model object. The clients are never built with the raw-data flag (11.1), so a
    bare `dict` here means the SDK surface changed under us: fail closed rather than silently attribute-miss."""
    if isinstance(value, dict):
        raise BrokerError(f"the broker returned raw {what} data instead of a model object: the pinned alpaca-py surface changed")
    return value


def to_order_state(raw: Any, *, fallback_client_order_id: str | None = None) -> OrderState:
    """`alpaca.trading.models.Order` -> `OrderState` (2.4).

    `filled_net` is the signed net price per share as the broker reports it: for an mleg parent Alpaca already signs
    `filled_avg_price` the way we do (positive = debit, negative = credit, Conventions), while a single-leg order reports a
    positive price and the sign comes from the side. When the parent of a multi-leg order carries no filled quantity the
    legs decide it: a leg fills `ratio_qty` contracts per structure, so the structure count is the minimum over the legs.
    """
    cid = getattr(raw, "client_order_id", None) or fallback_client_order_id
    if not cid:
        raise BrokerError("the broker returned an order without a client_order_id")
    status_text = _enum_value(getattr(raw, "status", None))
    status = _STATUS_MAP.get(status_text, OrderStatus.UNKNOWN)
    qty = _whole(getattr(raw, "qty", None) or 0, "order qty")
    filled_qty = _filled_qty(raw)
    order_id = getattr(raw, "id", None)
    return OrderState(
        client_order_id=str(cid),
        broker_order_id=None if order_id is None else str(order_id),
        status=status,
        qty=qty,
        filled_qty=filled_qty,
        filled_net=_filled_net(raw, filled_qty),
        reject_code=None,  # the numeric code only ever arrives on the error of a call, never on an order row
        message=None,
        updated_at=to_utc(getattr(raw, "updated_at", None) or getattr(raw, "submitted_at", None), "order updated_at"),
    )


def _filled_qty(raw: Any) -> int:
    reported = getattr(raw, "filled_qty", None)
    if reported is not None:
        return _whole(reported, "order filled_qty")
    legs = getattr(raw, "legs", None) or ()
    if not legs:
        return 0
    per_leg: list[int] = []
    for leg in legs:
        ratio = _whole(getattr(leg, "ratio_qty", None) or _CONTRACTS_PER_CALL, "leg ratio_qty") or _CONTRACTS_PER_CALL
        per_leg.append(_whole(getattr(leg, "filled_qty", None) or 0, "leg filled_qty") // ratio)
    return min(per_leg)


def _filled_net(raw: Any, filled_qty: int) -> int | None:
    price = getattr(raw, "filled_avg_price", None)
    if price is None or price == "" or filled_qty <= 0:
        return None
    magnitude = _price_cents(price, "filled_avg_price")
    if _enum_value(getattr(raw, "order_class", None)) == OrderClass.MLEG.value:
        return magnitude  # already signed by Alpaca: + debit / - credit
    side = _enum_value(getattr(raw, "side", None))
    return abs(magnitude) if side == OrderSide.BUY.value else -abs(magnitude)


def to_broker_position(raw: Any) -> BrokerPosition:
    """`alpaca.trading.models.Position` -> `BrokerPosition` (2.4): the symbol, a SIGNED quantity and the option flag.

    Alpaca reports `qty` as a string and the direction separately in `side`; the sign is taken from `side` so the adapter is
    correct whichever convention a response uses.
    """
    symbol = str(getattr(raw, "symbol", "") or "")
    if not symbol:
        raise BrokerError("the broker returned a position without a symbol")
    magnitude = abs(_whole(getattr(raw, "qty", None) or 0, f"position qty of {symbol}"))
    short = _enum_value(getattr(raw, "side", None)) == "short"
    asset_class = _enum_value(getattr(raw, "asset_class", None))
    return BrokerPosition(
        symbol=symbol,
        qty=-magnitude if short else magnitude,
        is_option=asset_class == "us_option" or (asset_class in ("", "None") and is_occ(symbol)),
    )


def _activity_day(row: dict[str, Any]) -> date:
    stamp = row.get("date") or row.get("transaction_time") or row.get("system_date")
    if stamp is None:
        raise BrokerError(f"the broker activity {row.get('id')!r} carries no date")
    if isinstance(stamp, datetime):
        return stamp.astimezone(UTC).date()
    if isinstance(stamp, date):
        return stamp
    text = str(stamp)
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        raise BrokerError(f"the broker activity {row.get('id')!r} has an unparsable date {text!r}") from None


def _to_broker_activity(row: dict[str, Any]) -> BrokerActivity:
    qty = row.get("qty")
    return BrokerActivity(
        activity_type=str(row.get("activity_type", "")),
        symbol=str(row.get("symbol") or ""),
        qty=0 if qty in (None, "") else _whole(qty, "activity qty"),
        day=_activity_day(row),
        raw_id=str(row.get("id", "")),
    )


# ======================================================================================================================
# Order payloads (11.5 step 2)
# ======================================================================================================================


def _dollars(cents: int) -> float:
    """Signed cents per share -> the dollar number the API takes. Asserted to round-trip, so no cent is ever lost here."""
    value = cents / 100
    if int(Decimal(str(value)) * 100) != cents:
        raise InvariantError(f"the limit {cents} cents does not round-trip through dollars ({value!r})")
    return value


def _leg_symbol(intent: OrderIntent, index: int) -> str:
    return format_occ(intent.legs[index].contract)


def build_order_request(order: ApprovedOrder) -> OrderRequest:
    """The exact payload of 11.5 step 2 for one approved order. Pure: it is built before any worker thread starts.

    * mleg (2-4 option legs, limit): `LimitOrderRequest(qty, order_class=MLEG, limit_price=SIGNED net dollars,
      legs=[OptionLegRequest(symbol, ratio_qty, side, position_intent), ...])` - `position_intent` on EVERY leg;
    * single option leg, limit: `LimitOrderRequest(symbol, qty, side, limit_price=ABS(net) dollars, position_intent)`;
    * single option leg, market (the kill switch's last resort inside market hours, 9.5 K4): `MarketOrderRequest(...)`;
    * the equity flatten of an assignment (9.5 K4; `equity_symbol` set, `legs = ()`): a plain stock `MarketOrderRequest`
      with NO `position_intent` - it is not an option order.

    Signs follow Alpaca's mleg convention, which is ours (Conventions): positive = net debit, negative = net credit.
    """
    intent = order.intent
    tif = TimeInForce.DAY
    cid = order.client_order_id

    if intent.equity_symbol is not None:
        if intent.legs or intent.equity_side is None or intent.equity_qty is None:
            raise InvariantError(f"the equity flatten intent {intent.intent_id} must carry no option leg and a side and a share count")
        if order.limit is not None:
            raise InvariantError(f"the equity flatten intent {intent.intent_id} is a market order (9.5 K4), but a limit was approved")
        return MarketOrderRequest(
            symbol=intent.equity_symbol,
            qty=int(intent.equity_qty),
            side=_side(intent.equity_side),
            time_in_force=tif,
            client_order_id=cid,
        )

    legs = intent.legs
    if not legs:
        raise InvariantError(f"the intent {intent.intent_id} has no legs and is not an equity flatten")
    if order.qty <= 0:
        raise InvariantError(f"the approved order {cid} has qty {order.qty}: an option order is at least one contract")

    if order.limit is None:
        if len(legs) != 1:
            raise InvariantError(
                f"the approved order {cid} is a market order over {len(legs)} legs: the kill switch's last resort is one "
                "market order per leg (9.5 K4)"
            )
        return MarketOrderRequest(
            symbol=_leg_symbol(intent, 0),
            qty=order.qty,
            side=_side(legs[0].side),
            time_in_force=tif,
            position_intent=_position_intent(legs[0].position_intent),
            client_order_id=cid,
        )

    if len(legs) == 1:
        return LimitOrderRequest(
            symbol=_leg_symbol(intent, 0),
            qty=order.qty,
            side=_side(legs[0].side),
            limit_price=_dollars(abs(order.limit)),
            time_in_force=tif,
            position_intent=_position_intent(legs[0].position_intent),
            client_order_id=cid,
        )

    return LimitOrderRequest(
        qty=order.qty,
        order_class=OrderClass.MLEG,
        limit_price=_dollars(order.limit),
        time_in_force=tif,
        legs=[
            OptionLegRequest(
                symbol=_leg_symbol(intent, i),
                ratio_qty=leg.ratio,
                side=_side(leg.side),
                position_intent=_position_intent(leg.position_intent),
            )
            for i, leg in enumerate(legs)
        ],
        client_order_id=cid,
    )


def _side(side: Side) -> OrderSide:
    return OrderSide.BUY if Side(side) is Side.BUY else OrderSide.SELL


def _position_intent(intent: Any) -> PositionIntent:
    return PositionIntent(str(intent))


# ======================================================================================================================
# Transport layers
# ======================================================================================================================


class _TokenBucket:
    """A plain token bucket over `rest_calls_per_minute` (11.5). Starts full; refills continuously."""

    def __init__(self, calls_per_minute: int, *, monotonic: Callable[[], float], sleep: Callable[[float], None]) -> None:
        if calls_per_minute <= 0:
            raise ValueError(f"rest_calls_per_minute must be positive, got {calls_per_minute}")
        self._capacity = float(calls_per_minute)
        self._per_second = float(calls_per_minute) / 60.0
        self._tokens = float(calls_per_minute)
        self._monotonic = monotonic
        self._sleep = sleep
        self._last = monotonic()
        self._lock = threading.Lock()

    def take(self) -> float:
        """Consume one token, waiting if the bucket is empty. Returns the seconds waited (0.0 when it did not throttle)."""
        with self._lock:
            now = self._monotonic()
            self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._per_second)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0
            wait = (1.0 - self._tokens) / self._per_second
            self._tokens = 0.0
            self._last = now + wait
        self._sleep(wait)
        return wait


class AlpacaPaperBroker:
    """`Broker` (3.4) over the paper trading client. Ledger-free, deadline-bounded, idempotent on the client order id."""

    def __init__(
        self,
        clients: AlpacaClients,
        orders: OrdersConfig,
        *,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clients = clients
        self._trading = clients.trading
        self._orders = orders
        self._now = now if now is not None else (lambda: datetime.now(UTC))
        self._sleep = sleep
        self._monotonic = monotonic
        self._deadline_s = float(orders.call_deadline_s)
        self._bucket = _TokenBucket(orders.rest_calls_per_minute, monotonic=monotonic, sleep=sleep)
        self._calls = 0
        self._posts = 0

    # --- identity ------------------------------------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return BROKER_NAME

    @property
    def calls(self) -> int:
        """Every HTTP attempt this adapter has made (diagnostics and the drills of 15.4)."""
        return self._calls

    @property
    def posts(self) -> int:
        """Every order POST this adapter has made: the drill "exactly one order" reads this (15.4)."""
        return self._posts

    # --- the two bounds of INV-08 --------------------------------------------------------------------------------------

    def _run_once(self, fn: Callable[[], _T]) -> _T:
        """One attempt, in a worker thread with a hard wall-clock deadline on top of the session timeout (INV-08).

        The thread is a daemon and is simply abandoned when the deadline passes: its POST may still land, which is the
        documented late-POST case of 11.5 and is adopted by the next lookup or by reconcile R1.
        """
        self._calls += 1
        box: list[_T] = []
        failure: list[BaseException] = []

        def target() -> None:
            try:
                box.append(fn())
            except BaseException as exc:
                failure.append(exc)

        worker = threading.Thread(target=target, name="alpaca-call", daemon=True)
        worker.start()
        worker.join(self._deadline_s)
        if worker.is_alive():
            raise BrokerAmbiguous(f"the broker call passed its {self._deadline_s:g}s deadline; the outcome is unknown (INV-08)")
        if failure:
            raise _classify(failure[0])
        if not box:
            raise BrokerAmbiguous("the broker call returned nothing and raised nothing")
        return box[0]

    def _call(self, fn: Callable[[], _T], *, write: bool) -> _T:
        """`_call(fn, write=)` of 11.5: reads retry under OUR policy, writes never do."""
        attempts = 1 if write else self._orders.read_retries + 1
        last: BrokerError | None = None
        for attempt in range(attempts):
            if attempt:
                self._sleep(READ_BACKOFF_S[min(attempt - 1, len(READ_BACKOFF_S) - 1)])
            self._bucket.take()
            try:
                return self._run_once(fn)
            except BrokerRejected:
                raise  # definitive: a retry would only repeat it
            except BrokerAmbiguous as exc:
                last = exc
        raise last if last is not None else BrokerAmbiguous("no broker attempt was made")

    # --- reads ---------------------------------------------------------------------------------------------------------

    def account(self) -> AccountSnapshot:
        """The broker's account, including `last_equity` - the prior-session closing equity of the daily-loss halt (9.5)."""
        raw = self._call(self._trading.get_account, write=False)
        return account_snapshot(raw, now=self._now())

    def positions(self) -> tuple[BrokerPosition, ...]:
        """Leg level, exactly as the broker reports it, INCLUDING any equity left by an assignment (3.4, 9.5 K3)."""
        rows = self._call(self._trading.get_all_positions, write=False)
        return tuple(sorted((to_broker_position(row) for row in rows), key=lambda p: p.symbol))

    def open_orders(self) -> tuple[OrderState, ...]:
        """Every working order the broker holds - ours and foreign: reconcile R1 is the one that tells them apart."""
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)
        rows = self._call(lambda: self._trading.get_orders(filter=request), write=False)
        return tuple(sorted((to_order_state(row) for row in rows), key=lambda s: s.client_order_id))

    def get_order(self, client_order_id: str) -> OrderState | None:
        """By CLIENT id (D18). `None` means the broker has no such order - the only reading of a 404 here."""
        try:
            raw = self._call(lambda: self._trading.get_order_by_client_id(client_order_id), write=False)
        except BrokerRejected as exc:
            if exc.status == 404:
                return None
            raise
        if raw is None:
            return None
        return to_order_state(raw, fallback_client_order_id=client_order_id)

    def raw_clock(self) -> RawClock:
        """The vendor clock reading, through this adapter's deadline / timeout / rate limit; `BrokerClock` calls it (11.4)."""
        clock: RawClock = _model(self._call(self._trading.get_clock, write=False), "clock")
        return clock

    def raw_calendar(self, start: date, end: date) -> Sequence[Any]:
        """The vendor calendar rows for [start, end]; `AlpacaCalendar` is built from them (11.4)."""
        from jevbot.paper.clock import calendar_rows

        return self._call(lambda: calendar_rows(self._trading, start, end), write=False)

    def activities(self, since: date) -> tuple[BrokerActivity, ...]:
        """OPASN / OPEXC / OPEXP / OPTRD since `since`, inclusive (reconcile R2).

        `TradingClient` wraps no activities endpoint in the pinned wheel (only `BrokerClient` does), so this goes through the
        client's own authenticated `get()` against `/v2/account/activities` and maps the raw rows itself.
        """
        collected: list[BrokerActivity] = []
        page_token: str | None = None
        for _ in range(_ACTIVITY_MAX_PAGES):
            params: dict[str, Any] = {
                "activity_types": ",".join(ACTIVITY_TYPES),
                "after": since.isoformat(),
                "page_size": _ACTIVITY_PAGE_SIZE,
            }
            if page_token is not None:
                params["page_token"] = page_token
            rows = self._call(lambda p=params: self._trading.get("/account/activities", p), write=False)  # type: ignore[misc]
            page = list(rows or ())
            collected += [_to_broker_activity(row) for row in page]
            if len(page) < _ACTIVITY_PAGE_SIZE:
                break
            page_token = str(page[-1].get("id", ""))
            if not page_token:
                break
        wanted = set(ACTIVITY_TYPES)
        return tuple(
            sorted(
                (a for a in collected if a.activity_type in wanted and a.day >= since),
                key=lambda a: (a.day, a.activity_type, a.symbol, a.raw_id),
            )
        )

    # --- writes --------------------------------------------------------------------------------------------------------

    def submit(self, order: ApprovedOrder) -> OrderState:
        """11.5: look the client id up, then POST at most once; adopt on ambiguity, never post twice by default (V4).

        Ledger-free (3.4): the caller has already made `ORDER_STATUS{SUBMITTING}` durable and writes the result entry with
        `reconcile.record_order_status` (9.6). This method only returns an `OrderState` or raises.
        """
        request = build_order_request(order)  # built first: a payload bug must never look like an ambiguous call
        cid = order.client_order_id

        existing = self.get_order(cid)  # step 1 - crash / retry safety: never POST twice for one client id (INV-07)
        if existing is not None:
            return existing

        try:
            return self._post(request, cid)
        except BrokerAmbiguous as ambiguous:
            adopted = self._lookup_after_ambiguity(cid)
            if adopted is not None:
                return adopted
            if self._orders.repost_same_id:  # V4: off until probe P-ALP-2 confirms that a duplicate id is rejected
                try:
                    return self._post(request, cid)
                except BrokerAmbiguous:
                    adopted = self._lookup_after_ambiguity(cid)
                    if adopted is not None:
                        return adopted
            raise BrokerAmbiguous(
                f"the outcome of order {cid} is unknown after {self._orders.ambiguous_lookups} lookups: {ambiguous}"
            ) from ambiguous

    def _post(self, request: OrderRequest, cid: str) -> OrderState:
        self._posts += 1
        raw = self._call(lambda: self._trading.submit_order(request), write=True)
        return to_order_state(raw, fallback_client_order_id=cid)

    def _lookup_after_ambiguity(self, cid: str) -> OrderState | None:
        """Step 4: up to `orders.ambiguous_lookups` lookups by client id, spread over about ten seconds.

        The wait comes FIRST: the point is to give a POST that is still in flight - or an abandoned deadline worker's late
        POST - time to land before we decide the order does not exist.
        """
        lookups = max(1, self._orders.ambiguous_lookups)
        pause = AMBIGUOUS_LOOKUP_WINDOW_S / lookups
        for _ in range(lookups):
            self._sleep(pause)
            try:
                state = self.get_order(cid)
            except BrokerError:
                continue
            if state is not None:
                return state
        return None

    def cancel(self, client_order_id: str) -> OrderState:
        """Cancel and confirm: the FINAL state decides, so a cancel that races a fill is never assumed away (11.5)."""
        state = self.get_order(client_order_id)
        if state is None:
            raise BrokerAmbiguous(f"cannot cancel {client_order_id}: the broker has no order with that client id")
        if state.status in TERMINAL_STATUSES:
            return state
        if state.broker_order_id is None:
            raise BrokerAmbiguous(f"cannot cancel {client_order_id}: the broker reported no order id for it")
        order_id = state.broker_order_id
        try:
            self._call(lambda: self._trading.cancel_order_by_id(order_id), write=True)
        except BrokerError:
            pass  # gone, already filled, or unknown: the poll below reads what really happened
        deadline = self._monotonic() + CANCEL_TERMINAL_DEADLINE_S
        while True:
            latest = self.get_order(client_order_id)
            if latest is not None:
                state = latest
                if state.status in TERMINAL_STATUSES:
                    return state
            if self._monotonic() >= deadline:
                return state  # not terminal yet: the caller re-reads it (CANCEL_ALL and reconcile R1 both do)
            self._sleep(float(self._orders.poll_s))

    def cancel_all(self) -> int:
        """`DELETE /v2/orders`: the number of orders the broker reports it cancelled (9.5 K2, the CANCEL_ALL phase)."""
        responses = self._call(self._trading.cancel_orders, write=True)
        return sum(1 for r in responses or () if _cancel_accepted(r))

    def set_suspended(self, suspended: bool) -> None:
        """Alpaca `suspend_trade`. The setter takes the FULL configuration object, so it is read first (11.1, 11.5).

        The kill switch flips this to True only once the account is verified flat (9.5 K6): a suspended account cannot place
        closing orders either.
        """
        current = _model(self._call(self._trading.get_account_configurations, write=False), "account configuration")
        updated = current.model_copy(update={"suspend_trade": bool(suspended)})
        self._call(lambda: self._trading.set_account_configurations(updated), write=True)

    def on_snapshot(self, view: Any) -> None:
        """No-op: only `SimBroker` prices queued orders on a snapshot (3.4)."""


def _cancel_accepted(response: Any) -> bool:
    status = getattr(response, "status", None)
    return status is None or int(status) < 300


# ======================================================================================================================
# Error classification (11.5 step 3 / 4)
# ======================================================================================================================


def reject_tag(message: str) -> str:
    """A coarse DIAGNOSTIC tag for a rejection message. Never a control decision beyond the tag (critique correction 4)."""
    lowered = message.lower()
    for needle, tag in _REJECT_TAGS:
        if needle in lowered:
            return tag
    return "other"


def _api_error_parts(exc: APIError) -> tuple[int | None, int | None, str]:
    try:
        status = exc.status_code
    except Exception:
        status = None
    try:
        code = exc.code
    except Exception:
        code = None
    try:
        message = exc.message
    except Exception:
        message = str(exc)
    return (None if status is None else int(status), None if code is None else int(code), str(message))


def _classify(exc: BaseException) -> BaseException:
    """Definitive vs unknown (11.5). 403 / 422 - and any other 4xx that is not a rate limit or a request timeout - mean the
    order was not created: `BrokerRejected`, never retried. A timeout, a connection error, a 5xx, a 429 or our own deadline
    mean the outcome is UNKNOWN: `BrokerAmbiguous`, resolved by a lookup on the client id.
    """
    if isinstance(exc, BrokerError | InvariantError):
        return exc
    if isinstance(exc, APIError):
        status, code, message = _api_error_parts(exc)
        if status is not None and 400 <= status < 500 and status not in AMBIGUOUS_STATUSES:
            return BrokerRejected(message, status=status, reject_code=code, tag=reject_tag(message))
        return BrokerAmbiguous(f"broker call failed with status {status}: {message}")
    return BrokerAmbiguous(f"{type(exc).__name__}: {exc}")
