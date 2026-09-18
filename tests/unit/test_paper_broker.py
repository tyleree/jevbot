"""`paper/broker.py`: the Alpaca paper adapter (DESIGN 11.5, 3.4; D18, G5; INV-03, INV-08, INV-09).

Every drill of 15.4 that concerns the adapter lives here: the idempotent-submit cases, the wall-clock deadline, the "never
re-POST by default" rule (V4), the late-POST adoption, the four payload shapes, and the proof that the adapter is ledger-free.
Payloads are asserted on `to_request_fields()` - the dict that actually goes on the wire - after the vendor's own validators
have run on the request object.
"""

import ast
import re
import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from jevbot import ids
from jevbot.config import OrdersConfig
from jevbot.errors import BrokerAmbiguous, BrokerError, BrokerRejected, InvariantError
from jevbot.paper.broker import (
    ACTIVITY_TYPES,
    AlpacaPaperBroker,
    _TokenBucket,
    build_order_request,
    reject_tag,
    to_broker_position,
    to_order_state,
)
from jevbot.types import (
    ApprovedOrder,
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
)
from tests.fixtures.fake_alpaca import FakeTradingClient, fake_clients

NAMESPACE = ids.namespace("exp001", "jev-1.13.0", 0)
SESSION = date(2026, 9, 17)
EXPIRY = date(2026, 10, 16)
KEY = SnapshotKey(session=SESSION, slot=Slot.EXEC)
NOW = datetime(2026, 9, 17, 19, 35, tzinfo=UTC)

SHORT_CALL = OptionContract(underlying="SPY", expiry=EXPIRY, right=Right.CALL, strike_milli=600_000)
LONG_CALL = OptionContract(underlying="SPY", expiry=EXPIRY, right=Right.PUT, strike_milli=610_000)


# ======================================================================================================================
# Builders
# ======================================================================================================================


def decision(subject: str = "entry", kind: str = "entry", underlying: str = "SPY") -> str:
    return ids.decision_id(NAMESPACE, SESSION, underlying, kind, subject)  # type: ignore[arg-type]


def intent(
    *,
    legs: Sequence[OrderLeg] = (),
    purpose: OrderPurpose = OrderPurpose.OPEN,
    qty: int = 2,
    limit_start: int = -125,
    part: int = 0,
    subject: str = "entry",
    kind: str = "entry",
    equity: tuple[str, Side, int] | None = None,
) -> OrderIntent:
    """An `OrderIntent` with a REAL deterministic id (INV-07). `structure` stays None: the adapter never reads it."""
    did = decision(subject, kind)
    return OrderIntent(
        intent_id=ids.intent_id(NAMESPACE, SESSION, did, purpose, part),
        decision_id=did,
        position_id=ids.position_id(NAMESPACE, SESSION, "structure-id-for-tests"),
        purpose=purpose,
        part=part,
        underlying="SPY",
        legs=tuple(legs),
        qty=qty,
        limit_start=limit_start,
        limit_natural=limit_start,
        reason="entry" if purpose is OrderPurpose.OPEN else "kill_switch",
        mandatory=purpose is not OrderPurpose.OPEN,
        session=SESSION,
        key=KEY,
        equity_symbol=None if equity is None else equity[0],
        equity_side=None if equity is None else equity[1],
        equity_qty=None if equity is None else equity[2],
    )


def approve(order_intent: OrderIntent, *, attempt: int = 0, qty: int | None = None, limit: int | None = -125) -> ApprovedOrder:
    return ApprovedOrder(
        intent=order_intent,
        verdict_id="v" * 24,
        client_order_id=ids.client_order_id(order_intent, attempt),
        attempt=attempt,
        qty=order_intent.qty if qty is None else qty,
        limit=limit,
        approved_at=NOW,
    )


def credit_spread_legs() -> tuple[OrderLeg, ...]:
    return (
        OrderLeg(contract=SHORT_CALL, side=Side.SELL, position_intent=PositionIntent.STO, ratio=1),
        OrderLeg(contract=LONG_CALL, side=Side.BUY, position_intent=PositionIntent.BTO, ratio=1),
    )


def single_leg() -> tuple[OrderLeg, ...]:
    return (OrderLeg(contract=SHORT_CALL, side=Side.BUY, position_intent=PositionIntent.BTO, ratio=1),)


def orders_config(**overrides: object) -> OrdersConfig:
    base: dict[str, object] = {"read_retries": 2, "ambiguous_lookups": 3, "call_deadline_s": 15, "rest_calls_per_minute": 100_000}
    return OrdersConfig(**{**base, **overrides})  # type: ignore[arg-type]


def make_broker(
    client: FakeTradingClient | None = None,
    *,
    sleep: Callable[[float], None] = lambda _: None,
    **config: object,
) -> tuple[AlpacaPaperBroker, FakeTradingClient]:
    trading = client if client is not None else FakeTradingClient()
    broker = AlpacaPaperBroker(fake_clients(trading), orders_config(**config), now=lambda: NOW, sleep=sleep)
    return broker, trading


# ======================================================================================================================
# Payload shapes (11.5 step 2)
# ======================================================================================================================


def test_the_mleg_credit_payload_is_signed_and_carries_a_position_intent_on_every_leg() -> None:
    order = approve(intent(legs=credit_spread_legs(), qty=2), limit=-125)

    wire = build_order_request(order).to_request_fields()

    assert wire["order_class"].value == "mleg"
    assert wire["type"].value == "limit"
    assert wire["time_in_force"].value == "day"
    assert wire["qty"] == 2
    assert wire["limit_price"] == -1.25  # NEGATIVE = net credit: Alpaca's mleg convention is ours (Conventions)
    assert "symbol" not in wire and "side" not in wire and "position_intent" not in wire
    assert wire["client_order_id"] == order.client_order_id
    assert [leg["symbol"] for leg in wire["legs"]] == ["SPY261016C00600000", "SPY261016P00610000"]
    assert [leg["ratio_qty"] for leg in wire["legs"]] == [1, 1]
    assert [leg["side"].value for leg in wire["legs"]] == ["sell", "buy"]
    assert [leg["position_intent"].value for leg in wire["legs"]] == ["sell_to_open", "buy_to_open"]


def test_the_mleg_debit_payload_is_positive() -> None:
    order = approve(intent(legs=credit_spread_legs()), limit=237)

    assert build_order_request(order).to_request_fields()["limit_price"] == 2.37  # POSITIVE = net debit (we pay)


def test_the_single_leg_payload_uses_the_absolute_limit_and_its_own_side() -> None:
    order = approve(intent(legs=single_leg(), qty=3), limit=-415)

    wire = build_order_request(order).to_request_fields()

    assert wire["symbol"] == "SPY261016C00600000"
    assert wire["qty"] == 3
    assert wire["side"].value == "buy"
    assert wire["position_intent"].value == "buy_to_open"
    assert wire["limit_price"] == 4.15  # a single-leg limit is always positive; the side carries the direction
    assert "legs" not in wire and "order_class" not in wire


def test_the_option_market_payload_is_the_kill_switch_last_resort() -> None:
    legs = (OrderLeg(contract=SHORT_CALL, side=Side.BUY, position_intent=PositionIntent.BTC, ratio=1),)
    order = approve(intent(legs=legs, purpose=OrderPurpose.KILL, part=1, subject="p|kill", kind="manage", qty=2), limit=None)

    wire = build_order_request(order).to_request_fields()

    assert wire["type"].value == "market"
    assert wire["symbol"] == "SPY261016C00600000"
    assert wire["qty"] == 2
    assert wire["side"].value == "buy"
    assert wire["position_intent"].value == "buy_to_close"
    assert "limit_price" not in wire


def test_the_equity_flatten_payload_is_a_plain_stock_market_order() -> None:
    order = approve(
        intent(legs=(), qty=0, purpose=OrderPurpose.KILL, subject="EQ:SPY|kill", kind="manage", equity=("SPY", Side.SELL, 100)),
        limit=None,
    )

    wire = build_order_request(order).to_request_fields()

    assert wire["symbol"] == "SPY"
    assert wire["qty"] == 100  # shares, not contracts
    assert wire["side"].value == "sell"
    assert wire["type"].value == "market"
    assert "position_intent" not in wire  # a stock order carries none (9.5 K4)
    assert "legs" not in wire


def test_the_limit_round_trips_through_dollars_to_the_cent() -> None:
    for cents in (-1, 1, -3, 7, -125, 237, -9_999, 12_345):
        order = approve(intent(legs=credit_spread_legs()), limit=cents)
        wire = build_order_request(order).to_request_fields()
        assert round(wire["limit_price"] * 100) == cents
        assert f"{wire['limit_price']:.2f}" == f"{cents / 100:.2f}"


@pytest.mark.parametrize(
    ("order", "needle"),
    [
        (
            lambda: approve(intent(legs=credit_spread_legs(), purpose=OrderPurpose.KILL, subject="p|kill", kind="manage"), limit=None),
            "one market order per leg",
        ),
        (lambda: approve(intent(legs=(), qty=1), limit=-100), "no legs and is not an equity flatten"),
        (lambda: approve(intent(legs=single_leg(), qty=0), limit=-100), "at least one contract"),
        (lambda: approve(intent(legs=single_leg(), equity=("SPY", Side.SELL, 100)), limit=None), "no option leg"),
        (lambda: approve(intent(legs=(), qty=0, equity=("SPY", Side.SELL, 100)), limit=-5), "is a market order"),
    ],
)
def test_an_impossible_payload_is_a_bug_not_an_order(order: Callable[[], ApprovedOrder], needle: str) -> None:
    with pytest.raises(InvariantError, match=needle):
        build_order_request(order())


def test_the_vendor_validators_really_run_on_our_payloads() -> None:
    # a 2-4 leg minimum for mleg and a required limit price are the SDK's own rules; our shapes satisfy them
    request = build_order_request(approve(intent(legs=credit_spread_legs())))
    assert len(request.legs or ()) == 2
    assert request.limit_price is not None


# ======================================================================================================================
# Idempotent submission (11.5 steps 1-4; the drills of 15.4)
# ======================================================================================================================


def submit_credit_spread(broker: AlpacaPaperBroker, *, attempt: int = 0) -> ApprovedOrder:
    return approve(intent(legs=credit_spread_legs()), attempt=attempt)


def test_a_timeout_after_the_broker_accepted_is_adopted_by_the_lookup_and_posts_once() -> None:
    broker, client = make_broker()
    client.fail_next("timeout_after_accept")
    order = submit_credit_spread(broker)

    state = broker.submit(order)

    assert state.client_order_id == order.client_order_id
    assert state.status is OrderStatus.SUBMITTED
    assert client.accepted == [order.client_order_id]  # EXACTLY one order exists at the broker
    assert broker.posts == 1  # and we POSTed exactly once
    assert client.calls.count("submit_order") == 1


def test_a_504_after_the_broker_accepted_behaves_the_same_way() -> None:
    broker, client = make_broker()
    client.fail_next("504_after_accept")

    state = broker.submit(submit_credit_spread(broker))

    assert state.status is OrderStatus.SUBMITTED
    assert len(client.accepted) == 1
    assert broker.posts == 1


def test_a_lost_post_is_ambiguous_and_is_never_re_posted_by_default() -> None:
    broker, client = make_broker()
    client.fail_next("connection_reset")  # nothing reached the broker
    order = submit_credit_spread(broker)

    with pytest.raises(BrokerAmbiguous, match="unknown after 3 lookups"):
        broker.submit(order)

    assert client.accepted == []
    assert broker.posts == 1  # V4: lookup-only. The caller ledgers UNKNOWN and abandons the intent this session
    assert client.calls.count("get_order_by_client_id") == 4  # the pre-lookup plus three after the ambiguity


def test_one_re_post_of_the_same_id_is_allowed_only_when_the_operator_enabled_it() -> None:
    broker, client = make_broker(repost_same_id=True)
    client.fail_next("connection_reset")
    order = submit_credit_spread(broker)

    state = broker.submit(order)

    assert state.status is OrderStatus.SUBMITTED
    assert broker.posts == 2  # exactly ONE re-POST, not a loop
    assert client.accepted == [order.client_order_id]


def test_a_second_re_post_is_not_attempted_even_when_enabled() -> None:
    broker, client = make_broker(repost_same_id=True)
    client.fail_next("connection_reset", count=2)

    with pytest.raises(BrokerAmbiguous):
        broker.submit(submit_credit_spread(broker))

    assert broker.posts == 2


def test_an_order_that_already_exists_is_adopted_without_posting_at_all() -> None:
    broker, client = make_broker()
    order = submit_credit_spread(broker)
    broker.submit(order)
    client.fill(order.client_order_id, avg_price="-1.25")

    again = broker.submit(order)  # the crash-restart case (INV-07)

    assert broker.posts == 1
    assert again.status is OrderStatus.FILLED
    assert again.filled_qty == 2
    assert again.filled_net == -125


@pytest.mark.parametrize(
    ("fault", "status", "code", "tag"),
    [
        ("403_bp", 403, 40310000, "insufficient_buying_power"),
        ("422_validation", 422, 42210000, "limit_price"),
    ],
)
def test_a_definitive_rejection_is_never_retried(fault: str, status: int, code: int, tag: str) -> None:
    broker, client = make_broker()
    client.fail_next(fault)

    with pytest.raises(BrokerRejected) as caught:
        broker.submit(submit_credit_spread(broker))

    assert caught.value.status == status
    assert caught.value.reject_code == code
    assert caught.value.tag == tag
    assert client.calls.count("submit_order") == 1  # exactly one attempt: a retry would only repeat it


def test_the_wall_clock_deadline_fires_on_a_hang() -> None:
    broker, client = make_broker(call_deadline_s=1, ambiguous_lookups=1)
    client.fail_next("hang")
    started = time.monotonic()
    try:
        with pytest.raises(BrokerAmbiguous, match="deadline"):
            broker.submit(submit_credit_spread(broker))
        elapsed = time.monotonic() - started
    finally:
        client.release_hang()

    assert 1.0 <= elapsed < 4.0  # the deadline, not the hang's own timeout, ended the call (INV-08)


def test_a_late_post_from_an_abandoned_worker_is_adopted_by_the_next_lookup() -> None:
    client = FakeTradingClient(hang_timeout_s=10.0)
    broker, _ = make_broker(client, call_deadline_s=1, ambiguous_lookups=1)
    client.fail_next("late_post")
    order = submit_credit_spread(broker)

    try:
        with pytest.raises(BrokerAmbiguous):
            broker.submit(order)  # the deadline fires; the worker thread is abandoned, still in flight
        assert client.accepted == []
        client.release_hang()  # the abandoned POST lands, carrying OUR client id
        state = wait_for(lambda: broker.get_order(order.client_order_id))
    finally:
        client.release_hang()

    assert state.client_order_id == order.client_order_id  # reconcile R1 / ingest_fills adopt it (11.5)
    assert client.accepted == [order.client_order_id]
    assert broker.posts == 1


def wait_for(read: Callable[[], object], timeout_s: float = 5.0) -> object:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        value = read()
        if value is not None:
            return value
        time.sleep(0.01)
    raise AssertionError("the late POST never landed")


def test_a_payload_bug_never_looks_like_an_ambiguous_call() -> None:
    broker, client = make_broker()
    bad = approve(intent(legs=(), qty=1), limit=-100)

    with pytest.raises(InvariantError):
        broker.submit(bad)

    assert client.calls == []  # nothing was sent, so nothing has to be reconciled


# ======================================================================================================================
# Reads, retries and the rate limit
# ======================================================================================================================


def test_a_read_is_retried_under_our_own_policy() -> None:
    broker, client = make_broker(read_retries=2)
    client.fail_next("connection_reset", on="get_account", count=2)

    snapshot = broker.account()

    assert snapshot.equity == 10_000_000
    assert client.calls.count("get_account") == 3


def test_a_read_that_keeps_failing_ends_ambiguous() -> None:
    broker, client = make_broker(read_retries=1)
    client.fail_next("connection_reset", on="get_account", count=5)

    with pytest.raises(BrokerAmbiguous):
        broker.account()

    assert client.calls.count("get_account") == 2


def test_a_write_is_never_retried() -> None:
    broker, client = make_broker(read_retries=2)
    client.fail_next("504_after_accept", on="cancel_orders", count=2)

    with pytest.raises(BrokerAmbiguous):
        broker.cancel_all()

    assert client.calls.count("cancel_orders") == 1


def test_the_token_bucket_holds_the_rest_rate() -> None:
    clock = [0.0]
    waits: list[float] = []
    bucket = _TokenBucket(60, monotonic=lambda: clock[0], sleep=waits.append)  # one call per second

    for _ in range(60):
        assert bucket.take() == 0.0  # the bucket starts full
    assert bucket.take() == pytest.approx(1.0)  # the 61st waits a second
    assert waits == [pytest.approx(1.0)]

    clock[0] += 10.0
    assert bucket.take() == 0.0  # ten seconds later there are tokens again


def test_a_zero_rate_is_refused() -> None:
    with pytest.raises(ValueError, match="rest_calls_per_minute"):
        _TokenBucket(0, monotonic=time.monotonic, sleep=time.sleep)


# ======================================================================================================================
# Order state, positions, activities
# ======================================================================================================================


def test_get_order_returns_none_for_an_id_the_broker_does_not_have() -> None:
    broker, _ = make_broker()
    assert broker.get_order("jb1-deadbeef-260917-0123456789ab-open-00-00") is None


def test_open_orders_reports_ours_and_foreign_alike() -> None:
    broker, client = make_broker()
    mine = submit_credit_spread(broker)
    broker.submit(mine)
    # something placed by hand in the Alpaca web console: the adapter reports it; reconcile R1 is what calls it foreign
    client.orders["hand-placed-order"] = dict(client.orders[mine.client_order_id], client_order_id="hand-placed-order")

    states = broker.open_orders()

    assert {s.client_order_id for s in states} == {mine.client_order_id, "hand-placed-order"}
    assert [s.client_order_id for s in states] == sorted(s.client_order_id for s in states)


def test_an_open_order_id_does_not_change_when_the_candidate_does() -> None:
    # INV-07: the id is a function of (decision_id, purpose, part, attempt) only, so a rebuilt candidate maps to the SAME
    # broker-side order - which is exactly why the lookup-before-POST of 11.5 step 1 can adopt it after a crash
    assert approve(intent(legs=credit_spread_legs())).client_order_id == approve(intent(legs=single_leg())).client_order_id


def test_a_filled_single_leg_order_gets_its_sign_from_the_side() -> None:
    broker, client = make_broker()
    buy = approve(intent(legs=single_leg(), qty=1), limit=415)
    broker.submit(buy)
    client.fill(buy.client_order_id, avg_price="4.15")

    state = broker.get_order(buy.client_order_id)

    assert state is not None
    assert state.filled_net == 415  # a BUY is a debit: positive


def test_a_sell_to_open_single_leg_reports_a_credit() -> None:
    legs = (OrderLeg(contract=SHORT_CALL, side=Side.SELL, position_intent=PositionIntent.STO, ratio=1),)
    broker, client = make_broker()
    sell = approve(intent(legs=legs, qty=1), limit=-415)
    broker.submit(sell)
    client.fill(sell.client_order_id, avg_price="4.15")

    state = broker.get_order(sell.client_order_id)

    assert state is not None and state.filled_net == -415


def test_a_partial_fill_is_reported_as_such() -> None:
    broker, client = make_broker()
    order = submit_credit_spread(broker)
    broker.submit(order)
    client.fill(order.client_order_id, qty=1, avg_price="-1.25")

    state = broker.get_order(order.client_order_id)

    assert state is not None
    assert state.status is OrderStatus.PARTIAL
    assert (state.qty, state.filled_qty) == (2, 1)


def test_an_mleg_parent_without_a_filled_quantity_falls_back_to_its_legs() -> None:
    payload = {
        "id": "b1",
        "client_order_id": "jb1-deadbeef-260917-0123456789ab-open-00-00",
        "created_at": datetime(2026, 9, 17, 19, 35, tzinfo=UTC),
        "updated_at": datetime(2026, 9, 17, 19, 36, tzinfo=UTC),
        "submitted_at": datetime(2026, 9, 17, 19, 35, tzinfo=UTC),
        "order_class": "mleg",
        "time_in_force": "day",
        "status": "partially_filled",
        "extended_hours": False,
        "qty": "4",
        "filled_qty": None,
        "legs": [
            {"symbol": "A", "ratio_qty": "1", "filled_qty": "3"},
            {"symbol": "B", "ratio_qty": "2", "filled_qty": "4"},  # 4 / 2 = 2 structures
        ],
    }
    state = to_order_state(_Row(payload))

    assert state.filled_qty == 2  # the minimum over the legs, per structure
    assert state.updated_at == datetime(2026, 9, 17, 19, 36, tzinfo=UTC)


class _Row:
    """A plain attribute bag: `to_order_state` reads attributes, so a dict-shaped stand-in is enough here."""

    def __init__(self, payload: dict[str, object]) -> None:
        for key, value in payload.items():
            setattr(self, key, [_Row(v) if isinstance(v, dict) else v for v in value] if isinstance(value, list) else value)


@pytest.mark.parametrize(
    ("vendor", "ours"),
    [
        ("new", OrderStatus.SUBMITTED),
        ("accepted", OrderStatus.SUBMITTED),
        ("pending_cancel", OrderStatus.SUBMITTED),
        ("done_for_day", OrderStatus.SUBMITTED),
        ("partially_filled", OrderStatus.PARTIAL),
        ("filled", OrderStatus.FILLED),
        ("canceled", OrderStatus.CANCELLED),
        ("replaced", OrderStatus.CANCELLED),
        ("expired", OrderStatus.EXPIRED),
        ("rejected", OrderStatus.REJECTED),
        ("something_new_in_a_later_api", OrderStatus.UNKNOWN),
    ],
)
def test_every_vendor_status_maps_or_falls_back_to_unknown(vendor: str, ours: OrderStatus) -> None:
    state = to_order_state(
        _Row(
            {
                "client_order_id": "c",
                "status": vendor,
                "qty": "1",
                "filled_qty": "0",
                "order_class": "simple",
                "updated_at": datetime(2026, 9, 17, tzinfo=UTC),
            }
        )
    )
    assert state.status is ours


def test_positions_are_signed_and_tagged() -> None:
    broker, client = make_broker()
    client.set_positions(("SPY261016C00600000", -2), ("SPY261016P00610000", 2), ("SPY", 100))

    positions = broker.positions()

    assert [(p.symbol, p.qty, p.is_option) for p in positions] == [
        ("SPY", 100, False),
        ("SPY261016C00600000", -2, True),
        ("SPY261016P00610000", 2, True),
    ]


def test_a_position_without_a_symbol_is_refused() -> None:
    with pytest.raises(BrokerError, match="without a symbol"):
        to_broker_position(_Row({"symbol": "", "qty": "1", "side": "long"}))


def test_activities_are_filtered_by_type_and_day_and_paginate() -> None:
    broker, client = make_broker()
    client.add_activity("OPASN", "SPY261016C00600000", -2, date(2026, 9, 16))
    client.add_activity("OPEXP", "SPY261016P00610000", 2, date(2026, 9, 17))
    client.add_activity("DIV", "SPY", 0, date(2026, 9, 17))  # not one of the four we act on

    activities = broker.activities(date(2026, 9, 16))

    assert [(a.activity_type, a.symbol, a.qty, a.day) for a in activities] == [
        ("OPASN", "SPY261016C00600000", -2, date(2026, 9, 16)),
        ("OPEXP", "SPY261016P00610000", 2, date(2026, 9, 17)),
    ]
    assert set(ACTIVITY_TYPES) == {"OPASN", "OPEXC", "OPEXP", "OPTRD"}


def test_activities_before_the_cut_off_are_dropped() -> None:
    broker, client = make_broker()
    client.add_activity("OPASN", "SPY261016C00600000", -2, date(2026, 9, 10))

    assert broker.activities(date(2026, 9, 16)) == ()


def test_a_full_page_of_activities_is_followed_by_the_next_one() -> None:
    broker, client = make_broker()
    for i in range(150):
        client.add_activity("OPTRD", "SPY261016C00600000", 1, date(2026, 9, 17), raw_id=f"id-{i:04d}")

    activities = broker.activities(date(2026, 9, 17))

    assert len(activities) == 150
    assert client.calls.count("get") == 2


# ======================================================================================================================
# Cancel and suspend
# ======================================================================================================================


def test_cancel_confirms_a_terminal_state() -> None:
    broker, client = make_broker()
    order = submit_credit_spread(broker)
    broker.submit(order)

    state = broker.cancel(order.client_order_id)

    assert state.status is OrderStatus.CANCELLED
    assert client.calls.count("cancel_order_by_id") == 1


def test_a_cancel_that_races_a_fill_reports_the_fill() -> None:
    broker, client = make_broker()
    order = submit_credit_spread(broker)
    broker.submit(order)
    client.fill(order.client_order_id, avg_price="-1.25")  # it filled while we were deciding to cancel

    state = broker.cancel(order.client_order_id)

    assert state.status is OrderStatus.FILLED  # the FINAL state decides, never the cancel's own outcome
    assert state.filled_qty == 2


def test_a_cancel_whose_delete_is_refused_still_reads_the_final_state() -> None:
    broker, client = make_broker()
    order = submit_credit_spread(broker)
    broker.submit(order)
    client.fail_next("422_validation", on="cancel_order_by_id")

    state = broker.cancel(order.client_order_id)

    assert state.status is OrderStatus.SUBMITTED  # not terminal: the caller re-reads it at CANCEL_ALL / reconcile R1
    assert client.calls.count("cancel_order_by_id") == 1


def test_cancelling_an_unknown_id_is_ambiguous_not_silent() -> None:
    broker, _ = make_broker()
    with pytest.raises(BrokerAmbiguous, match="no order with that client id"):
        broker.cancel("jb1-deadbeef-260917-0123456789ab-open-00-00")


def test_cancel_all_counts_what_the_broker_cancelled() -> None:
    broker, client = make_broker()
    for attempt in range(3):
        broker.submit(approve(intent(legs=credit_spread_legs()), attempt=attempt))
    client.fill(ids.client_order_id(intent(legs=credit_spread_legs()), 0), avg_price="-1.25")

    assert broker.cancel_all() == 2  # the filled one is already terminal
    assert broker.open_orders() == ()


def test_set_suspended_reads_the_full_configuration_and_writes_it_back() -> None:
    broker, client = make_broker()
    before = client.get_account_configurations()

    broker.set_suspended(True)

    after = client.configuration
    assert after.suspend_trade is True
    assert after.max_margin_multiplier == before.max_margin_multiplier  # every other field survives the round trip
    assert after.trade_confirm_email == before.trade_confirm_email
    assert broker.account().suspended is True

    broker.set_suspended(False)
    assert client.configuration.suspend_trade is False


def test_on_snapshot_is_a_no_op_for_a_live_broker() -> None:
    broker, client = make_broker()
    calls_before = list(client.calls)
    broker.on_snapshot(object())
    assert client.calls == calls_before


def test_the_adapter_names_itself() -> None:
    broker, _ = make_broker()
    assert broker.name == "alpaca_paper"


# ======================================================================================================================
# The classification table and the diagnostic tags
# ======================================================================================================================


@pytest.mark.parametrize(
    ("message", "tag"),
    [
        ("insufficient buying power for this order", "insufficient_buying_power"),
        ("account is not authorized to trade at this option level", "options_level"),
        ("duplicate client_order_id", "duplicate_client_order_id"),
        ("the market is closed", "market_closed"),
        ("potential wash trade detected", "wash_trade"),
        ("something entirely new", "other"),
    ],
)
def test_rejection_messages_get_a_coarse_tag_only(message: str, tag: str) -> None:
    assert reject_tag(message) == tag


# ======================================================================================================================
# The adapter is ledger-free (3.4, 11.5) and never replaces an order (INV-09)
# ======================================================================================================================

PAPER_SRC = Path(__file__).resolve().parents[2] / "src" / "jevbot" / "paper"
BROKER_SRC = PAPER_SRC / "broker.py"


def test_the_adapter_holds_no_ledger_and_no_book() -> None:
    broker, _ = make_broker()

    held = {name.lower() for name in vars(broker)}
    assert not any("ledger" in name or "book" in name for name in held), held
    assert not any(hasattr(broker, name) for name in ("ledger", "book", "append", "record_order_status"))


def test_the_module_imports_no_ledger_and_no_book() -> None:
    tree = ast.parse(BROKER_SRC.read_text(encoding="utf-8"))
    imported = {
        alias.name if isinstance(node, ast.Import) else f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    forbidden = {"jevbot.ledger", "jevbot.portfolio", "jevbot.reconcile", "jevbot.risk", "jevbot.cycle"}
    assert not any(any(name.startswith(f) for f in forbidden) for name in imported), sorted(imported)


def test_order_replace_is_never_called_anywhere_in_the_paper_package() -> None:
    pattern = re.compile(r"\breplace_order\w*\s*\(")  # INV-09: repricing is cancel + confirm + approve + submit
    hits = [
        f"{path.name}:{number}"
        for path in sorted(PAPER_SRC.glob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if pattern.search(line)
    ]
    assert hits == []


def test_no_daemon_worker_outlives_a_completed_call() -> None:
    broker, _ = make_broker()
    before = threading.active_count()

    broker.account()
    broker.positions()

    assert threading.active_count() == before
