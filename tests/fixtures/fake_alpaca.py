"""`FakeTradingClient`: a double of the alpaca-py surface `paper/*` uses, shaped from the REAL model classes (DESIGN 15.4, 11.1).

The point of this double is that it lies as little as possible. Every response is an actual
`alpaca.trading.models.{Order,Position,TradeAccount,AccountConfiguration,Clock,Calendar,CancelOrderResponse}` built from a raw
payload shaped like Alpaca's JSON, so the adapter meets the vendor's real parsing (`Order.__init__` turning an mleg's empty
`symbol` / `side` strings into `None`, `Calendar.__init__` composing a NAIVE Eastern datetime out of the date and a "%H:%M"
string, `Position.qty` arriving as a string with the direction in `side`). Every request is the real request object, so the
vendor's own validators run on the payloads `broker.build_order_request` produces - a two-leg minimum for `mleg`, `side`
required off it, `limit_price` required for a limit order.

The private transport surface (`_base_url`, `_session`, `_retry`, `_retry_wait`, `_retry_codes`) mirrors
`alpaca.common.rest.RESTClient`, with the wheel's own default constants, so `alpaca_client.make_clients` can install its
retry and timeout layers on this object exactly as it does on a real client.

Scripted faults (15.4) are queued per operation with `fail_next(fault, on=...)`:

| fault | what the fake does |
|---|---|
| `timeout_after_accept` | stores the order, then raises `TimeoutError`: the order EXISTS but our call failed |
| `504_after_accept` | stores the order, then raises `APIError` with status 504 |
| `connection_reset` | raises `ConnectionResetError` without storing anything |
| `403_bp` | `APIError` 403 / 40310000 "insufficient buying power" - definitive, never retried |
| `422_validation` | `APIError` 422 / 42210000 - definitive, never retried |
| `hang` | blocks until `release_hang()` or `hang_timeout_s`, so the caller's wall-clock deadline fires |
| `late_post` | blocks past the caller's deadline, THEN stores the order and returns: the abandoned worker's late POST |
"""

import threading
import uuid
from collections import deque
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, Final

from alpaca.common.constants import DEFAULT_RETRY_ATTEMPTS, DEFAULT_RETRY_EXCEPTION_CODES, DEFAULT_RETRY_WAIT_SECONDS
from alpaca.common.enums import BaseURL
from alpaca.common.exceptions import APIError
from alpaca.trading.enums import OrderClass, QueryOrderStatus
from alpaca.trading.models import (
    AccountConfiguration,
    Calendar,
    Clock,
    ClosePositionResponse,
    Order,
    Position,
    TradeAccount,
)
from alpaca.trading.requests import CancelOrderResponse, GetCalendarRequest, GetOrdersRequest, OrderRequest

from jevbot.occ import is_occ

__all__ = [
    "FAULTS",
    "FakeDataClient",
    "FakeRestClient",
    "FakeSession",
    "FakeTradingClient",
    "api_error",
    "calendar_models",
    "fake_clients",
]

FAULTS: Final[tuple[str, ...]] = (
    "timeout_after_accept",
    "504_after_accept",
    "connection_reset",
    "403_bp",
    "422_validation",
    "hang",
    "late_post",
)

TERMINAL_STATUSES: Final[frozenset[str]] = frozenset({"filled", "canceled", "expired", "rejected", "replaced"})

_DEFAULT_CLOCK_TS: Final = datetime(2026, 9, 17, 19, 45, tzinfo=UTC)


class _HttpError:
    """The `requests.exceptions.HTTPError` shape `APIError.status_code` reads: `.response.status_code`."""

    def __init__(self, status: int) -> None:
        self.response = _Response(status)
        self.request = None


class _Response:
    def __init__(self, status: int) -> None:
        self.status_code = status


def api_error(status: int, code: int, message: str) -> APIError:
    """A real `alpaca.common.exceptions.APIError` carrying a json body and an HTTP status, exactly as the SDK raises it."""
    body = '{"code": %d, "message": %s}' % (code, _json_string(message))  # noqa: UP031 - literal body text, not formatting logic
    return APIError(body, _HttpError(status))


def _json_string(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


class FakeSession:
    """A `requests.Session` stand-in. It records every call so the forced-timeout layer of 11.1 can be asserted."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> None:
        self.calls.append((method, url, dict(kwargs)))


class FakeRestClient:
    """The `alpaca.common.rest.RESTClient` private surface `make_clients` installs its layers on (11.1 steps 4-6)."""

    def __init__(
        self,
        *,
        base_url: BaseURL | str = BaseURL.TRADING_PAPER,
        api_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._base_url: BaseURL | str = base_url
        self._session = FakeSession()
        self._retry: int = DEFAULT_RETRY_ATTEMPTS
        self._retry_wait: int = DEFAULT_RETRY_WAIT_SECONDS
        self._retry_codes: list[int] = list(DEFAULT_RETRY_EXCEPTION_CODES)
        self._api_version = "v2"
        self._use_raw_data = False


class FakeDataClient(FakeRestClient):
    """The four historical-data clients: the same private surface, pointed at the market-data host."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("base_url", BaseURL.DATA)
        super().__init__(**kwargs)


class FakeTradingClient(FakeRestClient):
    """The `TradingClient` surface `paper/broker.py` and `paper/clock.py` call, with scripted faults (15.4)."""

    def __init__(
        self,
        *,
        equity: str = "100000.00",
        cash: str = "100000.00",
        options_buying_power: str = "100000.00",
        last_equity: str | None = "102000.00",
        options_trading_level: int | None = 3,
        trading_blocked: bool = False,
        account_blocked: bool = False,
        suspended: bool = False,
        clock_timestamp: datetime = _DEFAULT_CLOCK_TS,
        is_open: bool = True,
        calendar: Sequence[tuple[str, str, str]] = (("2026-09-17", "09:30", "16:00"),),
        duplicate_client_id_accepted: bool = False,
        hang_timeout_s: float = 5.0,
        late_post_delay_s: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.account_raw: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "account_number": "PA3PAPER001",
            "status": "ACTIVE",
            "currency": "USD",
            "cash": cash,
            "equity": equity,
            "last_equity": last_equity,
            "buying_power": equity,
            "options_buying_power": options_buying_power,
            "options_approved_level": options_trading_level,
            "options_trading_level": options_trading_level,
            "trading_blocked": trading_blocked,
            "account_blocked": account_blocked,
            "trade_suspended_by_user": suspended,
            "pattern_day_trader": False,
            "daytrade_count": 0,
        }
        self.configuration = AccountConfiguration(
            dtbp_check="entry",
            fractional_trading=False,
            max_margin_multiplier="1",
            no_shorting=False,
            pdt_check="entry",
            suspend_trade=suspended,
            trade_confirm_email="all",
            ptp_no_exception_entry=False,
            max_options_trading_level=options_trading_level,
        )
        self.clock_timestamp = clock_timestamp
        self.is_open = is_open
        self.calendar_rows: list[tuple[str, str, str]] = list(calendar)

        self.orders: dict[str, dict[str, Any]] = {}
        self.orders_by_id: dict[str, str] = {}
        self.submitted: list[OrderRequest] = []
        self.accepted: list[str] = []
        self.duplicate_accepts: int = 0
        self.duplicate_client_id_accepted = duplicate_client_id_accepted
        self.positions_raw: list[dict[str, Any]] = []
        self.activities_rows: list[dict[str, Any]] = []
        self.calls: list[str] = []

        self._faults: dict[str, deque[str]] = {}
        self.hang_released = threading.Event()
        self.hang_timeout_s = hang_timeout_s
        self.late_post_delay_s = late_post_delay_s

    # ------------------------------------------------------------------------------------------------------------------
    # Scripting
    # ------------------------------------------------------------------------------------------------------------------

    def fail_next(self, fault: str, *, on: str = "submit_order", count: int = 1) -> None:
        """Queue `fault` for the next `count` calls of the operation `on` (a method name of this class)."""
        if fault not in FAULTS:
            raise ValueError(f"unknown fault {fault!r}: one of {FAULTS}")
        self._faults.setdefault(on, deque()).extend([fault] * count)

    def release_hang(self) -> None:
        """Let every `hang` fault return at once (a test's `finally`, so no worker thread outlives the test)."""
        self.hang_released.set()

    def set_positions(self, *positions: tuple[str, int]) -> None:
        """Replace the broker's positions with `(symbol, signed_qty)` pairs; an OCC symbol becomes an option position."""
        self.positions_raw = [self._position_raw(symbol, qty) for symbol, qty in positions]

    def add_activity(self, activity_type: str, symbol: str, qty: int, day: date, *, raw_id: str | None = None) -> None:
        self.activities_rows.append(
            {
                "id": raw_id or f"{day:%Y%m%d}000000000::{uuid.uuid4()}",
                "account_id": self.account_raw["id"],
                "activity_type": activity_type,
                "date": day.isoformat(),
                "symbol": symbol,
                "qty": str(qty),
                "net_amount": "0",
                "description": activity_type,
            }
        )

    def fill(self, client_order_id: str, *, qty: int | None = None, avg_price: str | None = None) -> None:
        """Fill (or partially fill) a resting order, as the broker would report it."""
        payload = self.orders[client_order_id]
        total = int(Decimal(payload["qty"]))
        filled = total if qty is None else int(qty)
        payload["filled_qty"] = str(filled)
        payload["status"] = "filled" if filled >= total else "partially_filled"
        payload["updated_at"] = _stamp(self.clock_timestamp)
        if avg_price is not None:
            payload["filled_avg_price"] = avg_price
        for leg in payload.get("legs") or ():
            ratio = int(Decimal(str(leg.get("ratio_qty", 1))))
            leg["filled_qty"] = str(filled * ratio)
            leg["status"] = payload["status"]

    # ------------------------------------------------------------------------------------------------------------------
    # The TradingClient surface
    # ------------------------------------------------------------------------------------------------------------------

    def get_account(self) -> TradeAccount:
        self._enter("get_account")
        return TradeAccount(**self.account_raw)

    def get_account_configurations(self) -> AccountConfiguration:
        self._enter("get_account_configurations")
        return self.configuration

    def set_account_configurations(self, account_configurations: AccountConfiguration) -> AccountConfiguration:
        self._enter("set_account_configurations")
        self.configuration = account_configurations
        self.account_raw["trade_suspended_by_user"] = account_configurations.suspend_trade
        return self.configuration

    def get_clock(self) -> Clock:
        self._enter("get_clock")
        session_open = self.clock_timestamp.replace(microsecond=0)
        return Clock(
            timestamp=self.clock_timestamp,
            is_open=self.is_open,
            next_open=session_open + timedelta(days=1),
            next_close=session_open + timedelta(days=1, hours=6),
        )

    def get_calendar(self, filters: GetCalendarRequest | None = None) -> list[Calendar]:
        self._enter("get_calendar")
        start = getattr(filters, "start", None)
        end = getattr(filters, "end", None)
        rows = []
        for day, open_hhmm, close_hhmm in self.calendar_rows:
            when = date.fromisoformat(day)
            if (start is not None and when < start) or (end is not None and when > end):
                continue
            rows.append(Calendar(date=day, open=open_hhmm, close=close_hhmm))
        return rows

    def submit_order(self, order_data: OrderRequest) -> Order:
        fault = self._enter("submit_order", defer=True)
        self.submitted.append(order_data)
        if fault == "connection_reset":
            raise ConnectionResetError("connection reset by peer")
        if fault == "403_bp":
            raise api_error(403, 40310000, "insufficient buying power")
        if fault == "422_validation":
            raise api_error(422, 42210000, "invalid limit price for this contract")
        if fault == "hang":
            self._hang()
            raise TimeoutError("read timed out")
        if fault == "late_post":
            self._hang()
            return self._accept(order_data)
        accepted = self._accept(order_data)
        if fault == "timeout_after_accept":
            raise TimeoutError("read timed out after the order was accepted")
        if fault == "504_after_accept":
            raise api_error(504, 50410000, "gateway timeout")
        return accepted

    def get_order_by_client_id(self, client_id: str) -> Order:
        self._enter("get_order_by_client_id")
        payload = self.orders.get(client_id)
        if payload is None:
            raise api_error(404, 40410000, "order not found")
        return Order(**payload)

    def get_orders(self, filter: GetOrdersRequest | None = None) -> list[Order]:
        self._enter("get_orders")
        wanted = getattr(filter, "status", None) or QueryOrderStatus.OPEN
        rows = []
        for payload in self.orders.values():
            terminal = payload["status"] in TERMINAL_STATUSES
            if wanted == QueryOrderStatus.OPEN and terminal:
                continue
            if wanted == QueryOrderStatus.CLOSED and not terminal:
                continue
            rows.append(Order(**payload))
        return rows

    def cancel_order_by_id(self, order_id: Any) -> None:
        self._enter("cancel_order_by_id")
        cid = self.orders_by_id.get(str(order_id))
        if cid is None:
            raise api_error(404, 40410000, "order not found")
        payload = self.orders[cid]
        if payload["status"] in TERMINAL_STATUSES:
            raise api_error(422, 42210000, "order is not cancelable")
        payload["status"] = "canceled"
        payload["updated_at"] = _stamp(self.clock_timestamp)

    def cancel_orders(self) -> list[CancelOrderResponse]:
        self._enter("cancel_orders")
        responses = []
        for payload in self.orders.values():
            if payload["status"] in TERMINAL_STATUSES:
                continue
            payload["status"] = "canceled"
            payload["updated_at"] = _stamp(self.clock_timestamp)
            responses.append(CancelOrderResponse(id=payload["id"], status=200, body=None))
        return responses

    def get_all_positions(self) -> list[Position]:
        self._enter("get_all_positions")
        return [Position(**raw) for raw in self.positions_raw]

    def close_all_positions(self, cancel_orders: bool | None = None) -> list[ClosePositionResponse]:
        self._enter("close_all_positions")
        if cancel_orders:
            self.cancel_orders()
        responses = [
            ClosePositionResponse(order_id=uuid.uuid4(), status=200, symbol=raw["symbol"], body=Order(**self._flatten_order(raw)))
            for raw in self.positions_raw
        ]
        self.positions_raw = []
        return responses

    def get(self, path: str, data: dict[str, Any] | None = None, **_: Any) -> list[dict[str, Any]]:
        """The client's own authenticated GET; the adapter uses it for `/account/activities` (no wrapper exists)."""
        self._enter("get")
        if path != "/account/activities":
            raise api_error(404, 40410000, f"no such path {path}")
        params = data or {}
        wanted = {t for t in str(params.get("activity_types", "")).split(",") if t}
        after = params.get("after")
        rows = [
            row
            for row in self.activities_rows
            if (not wanted or row["activity_type"] in wanted) and (after is None or row["date"] >= str(after))
        ]
        token = params.get("page_token")
        if token is not None:
            ids = [row["id"] for row in rows]
            rows = rows[ids.index(token) + 1 :] if token in ids else []
        size = int(params.get("page_size", 100))
        return rows[:size]

    # ------------------------------------------------------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------------------------------------------------------

    def _enter(self, operation: str, *, defer: bool = False) -> str | None:
        """Record the call and apply (or, for `submit_order`, return) the next scripted fault for this operation."""
        self.calls.append(operation)
        queue = self._faults.get(operation)
        fault = queue.popleft() if queue else None
        if fault is None or defer:
            return fault
        if fault == "connection_reset":
            raise ConnectionResetError("connection reset by peer")
        if fault == "403_bp":
            raise api_error(403, 40310000, "insufficient buying power")
        if fault == "422_validation":
            raise api_error(422, 42210000, "validation failed")
        if fault == "504_after_accept":
            raise api_error(504, 50410000, "gateway timeout")
        if fault in ("hang", "late_post"):
            self._hang()
            raise TimeoutError("read timed out")
        raise TimeoutError("read timed out")  # timeout_after_accept on a read is simply a timeout

    def _hang(self) -> None:
        self.hang_released.wait(self.hang_timeout_s)

    def _accept(self, request: OrderRequest) -> Order:
        cid = str(request.client_order_id)
        if cid in self.orders:
            if not self.duplicate_client_id_accepted:
                raise api_error(422, 42210000, f"duplicate client_order_id {cid}")
            self.duplicate_accepts += 1
        payload = self._order_payload(request, cid)
        self.orders[cid] = payload
        self.orders_by_id[payload["id"]] = cid
        self.accepted.append(cid)
        return Order(**payload)

    def _order_payload(self, request: OrderRequest, cid: str) -> dict[str, Any]:
        stamp = _stamp(self.clock_timestamp)
        order_id = str(uuid.uuid4())
        mleg = getattr(request, "order_class", None) == OrderClass.MLEG
        limit = getattr(request, "limit_price", None)
        payload: dict[str, Any] = {
            "id": order_id,
            "client_order_id": cid,
            "created_at": stamp,
            "updated_at": stamp,
            "submitted_at": stamp,
            "asset_id": "" if mleg else str(uuid.uuid4()),
            "symbol": "" if mleg else str(request.symbol),
            "asset_class": "" if mleg else _asset_class(str(request.symbol)),
            "qty": str(int(request.qty or 0)),
            "filled_qty": "0",
            "filled_avg_price": None,
            "order_class": "mleg" if mleg else "simple",
            "order_type": "limit" if limit is not None else "market",
            "type": "limit" if limit is not None else "market",
            "side": "" if mleg else _enum_text(request.side),
            "position_intent": "" if mleg else _enum_text(getattr(request, "position_intent", None)),
            "time_in_force": _enum_text(request.time_in_force),
            "limit_price": None if limit is None else str(limit),
            "status": "accepted",
            "extended_hours": False,
            "legs": None,
        }
        legs = getattr(request, "legs", None)
        if legs:
            payload["legs"] = [self._leg_payload(leg, cid, i, stamp, int(request.qty or 0)) for i, leg in enumerate(legs)]
        return payload

    @staticmethod
    def _leg_payload(leg: Any, cid: str, index: int, stamp: str, parent_qty: int) -> dict[str, Any]:
        ratio = int(leg.ratio_qty)
        return {
            "id": str(uuid.uuid4()),
            "client_order_id": f"{cid}-leg{index}",
            "created_at": stamp,
            "updated_at": stamp,
            "submitted_at": stamp,
            "asset_id": str(uuid.uuid4()),
            "symbol": leg.symbol,
            "asset_class": "us_option",
            "qty": str(parent_qty * ratio),
            "filled_qty": "0",
            "filled_avg_price": None,
            "ratio_qty": str(ratio),
            "order_class": "mleg",
            "order_type": "limit",
            "type": "limit",
            "side": _enum_text(leg.side),
            "position_intent": _enum_text(leg.position_intent),
            "time_in_force": "day",
            "status": "accepted",
            "extended_hours": False,
        }

    def _flatten_order(self, position_raw: dict[str, Any]) -> dict[str, Any]:
        stamp = _stamp(self.clock_timestamp)
        qty = abs(int(Decimal(position_raw["qty"])))
        return {
            "id": str(uuid.uuid4()),
            "client_order_id": str(uuid.uuid4()),
            "created_at": stamp,
            "updated_at": stamp,
            "submitted_at": stamp,
            "symbol": position_raw["symbol"],
            "qty": str(qty),
            "filled_qty": "0",
            "order_class": "simple",
            "order_type": "market",
            "type": "market",
            "side": "sell" if position_raw["side"] == "long" else "buy",
            "time_in_force": "day",
            "status": "accepted",
            "extended_hours": False,
        }

    @staticmethod
    def _position_raw(symbol: str, qty: int) -> dict[str, Any]:
        return {
            "asset_id": str(uuid.uuid4()),
            "symbol": symbol,
            "exchange": "NASDAQ" if is_occ(symbol) else "NYSE",
            "asset_class": "us_option" if is_occ(symbol) else "us_equity",
            "avg_entry_price": "1.00",
            "qty": str(abs(qty)),
            "side": "short" if qty < 0 else "long",
            "cost_basis": str(qty * 100),
        }


def _stamp(when: datetime) -> str:
    return when.astimezone(UTC).isoformat()


def _enum_text(value: Any) -> str:
    return "" if value is None else str(getattr(value, "value", value))


def _asset_class(symbol: str) -> str:
    return "us_option" if is_occ(symbol) else "us_equity"


def fake_clients(trading: FakeTradingClient | None = None) -> Any:
    """An `AlpacaClients` tuple whose trading client is the fake and whose data clients are bare REST doubles."""
    from jevbot.paper.alpaca_client import AlpacaClients

    return AlpacaClients(
        trading=trading if trading is not None else FakeTradingClient(),
        options=FakeDataClient(),
        stocks=FakeDataClient(),
        news=FakeDataClient(),
        corporate_actions=FakeDataClient(),
    )


def calendar_models(rows: Iterable[tuple[str, str, str]]) -> list[Calendar]:
    """Build real `Calendar` models from `(date, "HH:MM", "HH:MM")` rows, the way `get_calendar` parses them."""
    return [Calendar(date=day, open=open_hhmm, close=close_hhmm) for day, open_hhmm, close_hhmm in rows]
