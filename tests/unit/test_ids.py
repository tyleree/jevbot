"""ids.py (DESIGN.md 2.10, INV-07): deterministic, state-independent, collision-free identifiers.

Golden digests were produced with coreutils (`printf '%s' '<joined fields>' | sha256sum`), never with the code under test.
"""

import hashlib
import inspect
import re
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from itertools import product

import pytest

from jevbot import ids
from jevbot.types import (
    Leg,
    OptionContract,
    OrderIntent,
    OrderLeg,
    OrderPurpose,
    OutcomeSpec,
    PositionIntent,
    RequestKind,
    Right,
    Side,
    Slot,
    SnapshotKey,
    Structure,
    StructureKind,
)

NS = "exp001:jev-1.13.0:g0"
D = date(2026, 9, 17)

# full sha256 digests (coreutils) of the joined strings named in the comments
SHA_NS = "a072c982a49cd93b6c97c46c14743678fe2a8db2fec87311a150325f5b2cecf5"  # exp001:jev-1.13.0:g0
SHA_ENTRY = "9d9337b3a17fc89d11057ee4ee789b563f725ff249973fa38db87e18aaebf025"  # NS|2026-09-17|SPY|entry|entry
SHA_MANAGE = "7356ff158e99ffa432411ac9282d558ec3aed0e6f5d1b6bd5acd4cb05f54fe9e"  # NS|2026-09-17|SPY|manage|0123456789abcdef
SHA_KILL = "ceddfb2e4b24074b349d85c85cbac532f14fadfbac9aea6c186a2b60c74e6df2"  # NS|2026-09-17|SPY|manage|0123456789abcdef|kill
SHA_EQ_KILL = "72d21024f341ba9b30fffb0ace9e0d98e0589c8280b35aae625b3be38d9be249"  # NS|2026-09-17|SPY|manage|EQ:SPY|kill
SHA_POSITION = "ef6c34e40534f42b9272b9582d9ad0e3c206b47a2f86eb6bb4b0a7837524d0cc"  # NS|2026-09-17|feedfacefeedface
SHA_EQ_POSITION = "c5d2351807270be3162d5ab1bd79de30d1fe80ecadc9e9e10e7fe6eb6a68de7e"  # EQ:SPY
# {"session":"2026-09-17","slot":"eod","spec":{"hi":66012,"horizon_sessions":5,"iv_var_ppm":null,"kind":"close_gt","lo":null,
#  "ref":66012,"resolve_on":"2026-09-24"},"u":"SPY"}
SHA_EVENT = "935c0ab931f12c96fe2d50fee12f6e0d8ce24404a1ca037d0203d628c5d9bc81"

POS = "0123456789abcdef"
ID_CHARSET = re.compile(r"[a-z0-9-]+")


def _contract(strike_milli: int, right: Right = Right.PUT) -> OptionContract:
    return OptionContract(underlying="SPY", expiry=date(2026, 10, 16), right=right, strike_milli=strike_milli)


def _put_credit(short_strike: int, long_strike: int) -> Structure:
    return Structure(
        kind=StructureKind.PUT_CREDIT,
        underlying="SPY",
        expiry=date(2026, 10, 16),
        last_session=date(2026, 10, 16),
        legs=(Leg(contract=_contract(long_strike), side=Side.BUY), Leg(contract=_contract(short_strike), side=Side.SELL)),
    )


def _intent(intent_id: str, decision_id: str, purpose: OrderPurpose, *, part: int = 0, structure: Structure | None = None) -> OrderIntent:
    legs: tuple[OrderLeg, ...] = ()
    if structure is not None:
        legs = tuple(
            OrderLeg(
                contract=leg.contract, side=leg.side, position_intent=PositionIntent.BTO if leg.side is Side.BUY else PositionIntent.STO
            )
            for leg in structure.legs
        )
    return OrderIntent(
        intent_id=intent_id,
        decision_id=decision_id,
        position_id=ids.position_id(NS, D, structure.structure_id) if structure is not None else POS,
        purpose=purpose,
        part=part,
        underlying="SPY",
        legs=legs,
        qty=1,
        limit_start=-100,
        limit_natural=-90,
        reason="entry" if purpose is OrderPurpose.OPEN else "kill_switch",
        mandatory=purpose is not OrderPurpose.OPEN,
        session=D,
        key=SnapshotKey(session=D, slot=Slot.EOD),
        structure=structure,
    )


# ======================================================================================================================
# formulas and goldens
# ======================================================================================================================


def test_namespace_and_ns8() -> None:
    assert ids.namespace("exp001", "jev-1.13.0", 0) == NS
    assert ids.namespace("exp001", "jev-1.13.0", 3) == "exp001:jev-1.13.0:g3"  # a refresh => a new namespace (D8)
    assert ids.namespace("exp001", "jev-1.14.0", 0) != NS  # a forced model change => a new namespace (D12)
    assert ids.ns8(NS) == SHA_NS[:8] == "a072c982"


def test_decision_id_goldens() -> None:
    assert ids.decision_id(NS, D, "SPY", "entry", "entry") == SHA_ENTRY[:32]
    assert ids.decision_id(NS, D, "SPY", "manage", POS) == SHA_MANAGE[:32]
    assert ids.decision_id(NS, D, "SPY", "manage", ids.kill_subject(POS)) == SHA_KILL[:32]
    assert ids.decision_id(NS, D, "SPY", "manage", ids.equity_kill_subject("SPY")) == SHA_EQ_KILL[:32]
    assert ids.kill_subject(POS) == POS + "|kill"
    assert ids.equity_kill_subject("SPY") == "EQ:SPY|kill"
    assert ids.ENTRY_SUBJECT == "entry"


def test_position_ids() -> None:
    assert ids.position_id(NS, D, "feedfacefeedface") == SHA_POSITION[:16]
    assert ids.equity_position_id("SPY") == SHA_EQ_POSITION[:16]
    structure = _put_credit(640000, 635000)
    assert len(ids.position_id(NS, D, structure.structure_id)) == 16
    assert ids.position_id(NS, D, structure.structure_id) != ids.position_id(NS, D + timedelta(days=1), structure.structure_id)


def test_intent_and_client_order_id_format() -> None:
    did = ids.decision_id(NS, D, "SPY", "entry", "entry")
    iid = ids.intent_id(NS, D, did, OrderPurpose.OPEN, 0)
    assert iid == "jb1-a072c982-260917-9d9337b3a17f-open-00"
    intent = _intent(iid, did, OrderPurpose.OPEN, structure=_put_credit(640000, 635000))
    assert ids.client_order_id(intent, 0) == "jb1-a072c982-260917-9d9337b3a17f-open-00-00"
    assert ids.client_order_id(intent, 1) == "jb1-a072c982-260917-9d9337b3a17f-open-00-01"
    assert ids.client_order_id(intent, 12) == "jb1-a072c982-260917-9d9337b3a17f-open-00-12"
    kill = ids.decision_id(NS, D, "SPY", "manage", ids.kill_subject(POS))
    assert ids.intent_id(NS, D, kill, OrderPurpose.KILL, 3) == "jb1-a072c982-260917-ceddfb2e4b24-kill-03"
    close = ids.decision_id(NS, D, "SPY", "manage", POS)
    assert ids.intent_id(NS, D, close, OrderPurpose.CLOSE, 0) == "jb1-a072c982-260917-7356ff158e99-close-00"
    # the client order id is the intent id plus the attempt suffix (2.4)
    assert ids.client_order_id(intent, 7).rsplit("-", 1) == [iid, "07"]


def test_fill_and_forecast_id_formulas() -> None:
    cid = "jb1-a072c982-260917-9d9337b3a17f-open-00-00"
    assert ids.fill_id(cid, 3) == hashlib.sha256(b"jb1-a072c982-260917-9d9337b3a17f-open-00-00|3").hexdigest()[:24]
    assert ids.fill_id(cid, 3) != ids.fill_id(cid, 4)  # keyed by the CUMULATIVE quantity: each partial fill has its own id
    did = SHA_ENTRY[:32]
    assert ids.forecast_id(did, "eval.up_1s", True) == hashlib.sha256(f"{did}|eval.up_1s|1".encode()).hexdigest()[:24]
    assert ids.forecast_id(did, "eval.up_1s", False) == hashlib.sha256(f"{did}|eval.up_1s|0".encode()).hexdigest()[:24]


def test_event_key_golden_and_sensitivity() -> None:
    key = SnapshotKey(session=D, slot=Slot.EOD)
    spec = OutcomeSpec(kind="close_gt", horizon_sessions=5, resolve_on=date(2026, 9, 24), ref=66012, hi=66012)
    base = ids.event_key("SPY", key, spec)
    assert base == SHA_EVENT[:24]
    others = {
        ids.event_key("QQQ", key, spec),
        ids.event_key("SPY", SnapshotKey(session=D + timedelta(days=1), slot=Slot.EOD), spec),
        ids.event_key("SPY", SnapshotKey(session=D, slot=Slot.DEC), spec),
        ids.event_key("SPY", key, OutcomeSpec(kind="close_gt", horizon_sessions=5, resolve_on=date(2026, 9, 24), ref=66012, hi=66013)),
        ids.event_key("SPY", key, OutcomeSpec(kind="close_lt", horizon_sessions=5, resolve_on=date(2026, 9, 24), ref=66012, lo=66012)),
        ids.event_key("SPY", key, OutcomeSpec(kind="close_gt", horizon_sessions=1, resolve_on=date(2026, 9, 18), ref=66012, hi=66012)),
    }
    assert base not in others and len(others) == 6


def test_event_key_is_independent_of_the_decider() -> None:
    # the key is a function of (underlying, snapshot, outcome spec) ONLY: no namespace, model, decision id, question id or text flag
    assert list(inspect.signature(ids.event_key).parameters) == ["underlying", "key", "spec"]
    key = SnapshotKey(session=D, slot=Slot.EOD)
    spec = OutcomeSpec(kind="close_inside", horizon_sessions=5, resolve_on=date(2026, 9, 24), ref=66012, lo=65000, hi=67000)
    # two deciders (namespaces) forecasting the same event get different decision / forecast ids but ONE event key
    ns_live, ns_baseline = ids.namespace("exp001", "jev-1.13.0", 0), ids.namespace("exp001-baseline", "baseline-4", 0)
    did_live, did_base = ids.decision_id(ns_live, D, "SPY", "entry", "entry"), ids.decision_id(ns_baseline, D, "SPY", "entry", "entry")
    assert did_live != did_base
    assert ids.forecast_id(did_live, "eval.inside_1em_5s", True) != ids.forecast_id(did_base, "eval.inside_1em_5s", True)
    assert ids.event_key("SPY", key, spec) == ids.event_key("SPY", SnapshotKey(session=D, slot=Slot.EOD), spec)


# ======================================================================================================================
# decision-level cases (2.10, CON-2)
# ======================================================================================================================


def test_entry_and_entry_text_requests_share_one_decision_id() -> None:
    # the id takes the DECISION-level kind: there is no way to hash a RequestKind into it
    assert list(inspect.signature(ids.decision_id).parameters) == ["namespace", "session", "underlying", "kind", "subject"]
    subject_of = {RequestKind.ENTRY: ("entry", "entry"), RequestKind.ENTRY_TEXT: ("entry", "entry")}
    got = {rk: ids.decision_id(NS, D, "SPY", kind, subject) for rk, (kind, subject) in subject_of.items()}  # type: ignore[arg-type]
    assert got[RequestKind.ENTRY] == got[RequestKind.ENTRY_TEXT] == SHA_ENTRY[:32]
    manage = {rk: ids.decision_id(NS, D, "SPY", "manage", POS) for rk in (RequestKind.MANAGE, RequestKind.MANAGE_TEXT)}
    assert manage[RequestKind.MANAGE] == manage[RequestKind.MANAGE_TEXT] == SHA_MANAGE[:32]
    # RequestKind.ENTRY happens to equal "entry" and is accepted as such; a text / probe request kind is refused outright
    assert ids.decision_id(NS, D, "SPY", RequestKind.ENTRY, "entry") == SHA_ENTRY[:32]  # type: ignore[arg-type]
    for bad in (RequestKind.ENTRY_TEXT, RequestKind.MANAGE_TEXT, RequestKind.PROBE, "Entry", "", "close"):
        with pytest.raises(ValueError, match="DECISION-level"):
            ids.decision_id(NS, D, "SPY", bad, "entry")  # type: ignore[arg-type]


def test_entry_and_manage_decisions_never_collide() -> None:
    sessions = [D + timedelta(days=i) for i in range(5)]
    positions = [ids.position_id(NS, D, f"{i:016x}") for i in range(20)]
    seen: set[str] = set()
    n = 0
    for session, underlying in product(sessions, ("SPY", "QQQ", "IWM")):
        seen.add(ids.decision_id(NS, session, underlying, "entry", "entry"))
        n += 1
        # even a manage decision whose subject is the literal "entry" differs from the entry decision (the kind is hashed)
        seen.add(ids.decision_id(NS, session, underlying, "manage", "entry"))
        n += 1
        for pos in positions:
            seen.add(ids.decision_id(NS, session, underlying, "manage", pos))
            seen.add(ids.decision_id(NS, session, underlying, "manage", ids.kill_subject(pos)))
            n += 2
        seen.add(ids.decision_id(NS, session, underlying, "manage", ids.equity_kill_subject(underlying)))
        n += 1
    assert len(seen) == n
    assert all(re.fullmatch(r"[0-9a-f]{32}", d) for d in seen)


def test_with_text_and_without_text_forecast_ids_differ() -> None:
    did = ids.decision_id(NS, D, "SPY", "entry", "entry")
    questions = [f"eval.q{i}" for i in range(12)] + ["under.direction#bullish"]
    with_text = {ids.forecast_id(did, q, True) for q in questions}
    without_text = {ids.forecast_id(did, q, False) for q in questions}
    assert len(with_text) == len(without_text) == len(questions)
    assert with_text.isdisjoint(without_text)  # the 12 eval questions ride in both batches under ONE decision id (2.7)
    assert all(re.fullmatch(r"[0-9a-f]{24}", f) for f in with_text | without_text)
    with pytest.raises(TypeError):
        ids.forecast_id(did, "eval.up_1s", 1)  # type: ignore[arg-type]


# ======================================================================================================================
# the collision test over a generated day (15.1)
# ======================================================================================================================


def _generated_day(namespace: str) -> dict[str, tuple[str, ...]]:
    """client_order_id -> the (purpose, session, underlying, subject, part, attempt) tuple that produced it."""
    sessions = [date(2026, 9, 14) + timedelta(days=i) for i in range(5)]  # Mon..Fri
    underlyings = ("SPY", "QQQ", "IWM")
    attempts = range(4)  # 0 plus three reprices (orders.max_order_attempts / kill.flatten_attempts)
    out: dict[str, tuple[str, ...]] = {}

    def emit(session: date, underlying: str, kind: str, subject: str, purpose: OrderPurpose, part: int) -> None:
        did = ids.decision_id(namespace, session, underlying, kind, subject)  # type: ignore[arg-type]
        intent = _intent(ids.intent_id(namespace, session, did, purpose, part), did, purpose, part=part)
        for attempt in attempts:
            cid = ids.client_order_id(intent, attempt)
            origin = (purpose.value, session.isoformat(), underlying, subject, str(part), str(attempt))
            assert out.setdefault(cid, origin) == origin, f"collision: {cid} <- {out[cid]} and {origin}"

    for underlying in underlyings:
        # two structures per underlying, opened on the first two sessions
        positions = [
            ids.position_id(namespace, sessions[i], _put_credit(640000 - 5000 * i, 630000 - 5000 * i).structure_id) for i in range(2)
        ]
        for session in sessions:
            emit(session, underlying, "entry", "entry", OrderPurpose.OPEN, 0)  # the one OPEN of the entry decision, repriced
            for pos in positions:
                emit(session, underlying, "manage", pos, OrderPurpose.CLOSE, 0)  # a close re-issued on every later session
                emit(session, underlying, "manage", ids.kill_subject(pos), OrderPurpose.KILL, 0)  # kill mleg
                for part in (1, 2, 3, 4):  # per-leg kill fallback orders
                    emit(session, underlying, "manage", ids.kill_subject(pos), OrderPurpose.KILL, part)
            emit(session, underlying, "manage", ids.equity_kill_subject(underlying), OrderPurpose.KILL, 0)  # assigned-stock flatten
    return out


def test_no_collisions_over_a_generated_day() -> None:
    day = _generated_day(NS)
    # 3 underlyings x 5 sessions x (1 open + 2 x (1 close + 1 kill + 4 legs) + 1 equity flatten) x 4 attempts
    assert len(day) == 3 * 5 * (1 + 2 * 6 + 1) * 4 == 840
    assert len(set(day.values())) == len(day)
    for cid in day:
        assert len(cid) <= 48
        assert ID_CHARSET.fullmatch(cid)
        assert cid.startswith(ids.ORDER_ID_PREFIX)
        assert ids.is_bot_order_id(cid)


def test_a_restart_reproduces_the_same_ids() -> None:
    assert _generated_day(NS) == _generated_day(NS)
    assert list(_generated_day(NS)) == list(_generated_day(NS))


def test_other_namespaces_get_other_ids() -> None:
    a, b = _generated_day(NS), _generated_day(ids.namespace("exp001", "jev-1.13.0", 1))
    assert set(a).isdisjoint(b)


def test_an_open_id_does_not_change_when_the_candidates_strikes_change() -> None:
    did = ids.decision_id(NS, D, "SPY", "entry", "entry")
    iid = ids.intent_id(NS, D, did, OrderPurpose.OPEN, 0)
    before = _intent(iid, did, OrderPurpose.OPEN, structure=_put_credit(640000, 635000))
    after = _intent(iid, did, OrderPurpose.OPEN, structure=_put_credit(638000, 633000))  # a restarted cycle rebuilt another candidate
    assert before.structure is not None and after.structure is not None
    assert before.structure.structure_id != after.structure.structure_id
    assert before.position_id != after.position_id
    assert [ids.client_order_id(before, a) for a in range(3)] == [ids.client_order_id(after, a) for a in range(3)]
    # nothing about the candidate can even be passed in
    assert list(inspect.signature(ids.intent_id).parameters) == ["namespace", "session", "decision_id", "purpose", "part"]


def test_probe_and_foreign_ids_are_not_bot_ids() -> None:
    assert ids.ORDER_ID_PREFIX == "jb1-" and ids.PROBE_ID_PREFIX == "jbp-"
    assert ids.is_bot_order_id("jb1-a072c982-260917-9d9337b3a17f-open-00-00")
    for foreign in (
        "jbp-a072c982-260917-9d9337b3a17f-open-00-00",
        "jb1-a072c982-260917-9d9337b3a17f-open-00",  # an intent id is not a client order id
        "jb1-a072c982-260917-9d9337b3a17f-roll-00-00",
        "jb1-A072C982-260917-9d9337b3a17f-open-00-00",
        "jb1-a072c982-260917-9d9337b3a17f-open-00-00\n",
        "3f6c1a1e-manual-order",
        "",
    ):
        assert not ids.is_bot_order_id(foreign)
    assert not ids.is_bot_order_id(None)  # type: ignore[arg-type]


# ======================================================================================================================
# validation: inputs that would break determinism or the collision argument are refused
# ======================================================================================================================


def test_separator_injection_is_refused() -> None:
    # "a|b" + "c" and "a" + "b|c" would hash the same joined string
    with pytest.raises(ValueError, match=r"\|"):
        ids.decision_id(NS, D, "SPY|entry", "entry", "entry")
    with pytest.raises(ValueError, match=r"\|"):
        ids.decision_id(NS + "|x", D, "SPY", "entry", "entry")
    with pytest.raises(ValueError, match=r"\|"):
        ids.position_id(NS + "|x", D, "feedfacefeedface")
    with pytest.raises(ValueError, match=r"\|"):
        ids.kill_subject(POS + "|kill")
    with pytest.raises(ValueError, match=r"\|"):
        ids.forecast_id(SHA_ENTRY[:32], "eval.up_1s|1", False)
    with pytest.raises(ValueError, match=r"\|"):
        ids.fill_id("cid|1", 1)
    with pytest.raises(ValueError, match=":"):
        ids.namespace("exp:001", "jev-1.13.0", 0)


def test_empty_fields_are_refused() -> None:
    calls: tuple[Callable[[], str], ...] = (
        lambda: ids.namespace("", "m", 0),
        lambda: ids.namespace("e", "", 0),
        lambda: ids.ns8(""),
        lambda: ids.decision_id("", D, "SPY", "entry", "entry"),
        lambda: ids.decision_id(NS, D, "", "entry", "entry"),
        lambda: ids.decision_id(NS, D, "SPY", "manage", ""),
        lambda: ids.position_id(NS, D, ""),
        lambda: ids.equity_position_id(""),
        lambda: ids.fill_id("", 1),
        lambda: ids.forecast_id("", "q", True),
        lambda: ids.forecast_id(SHA_ENTRY[:32], "", True),
    )
    for call in calls:
        with pytest.raises(ValueError):
            call()


def test_a_datetime_is_not_a_session() -> None:
    stamp = datetime(2026, 9, 17, 20, 0, tzinfo=UTC)
    with pytest.raises(TypeError):
        ids.decision_id(NS, stamp, "SPY", "entry", "entry")
    with pytest.raises(TypeError):
        ids.position_id(NS, stamp, "feedfacefeedface")
    with pytest.raises(TypeError):
        ids.intent_id(NS, stamp, SHA_ENTRY[:32], OrderPurpose.OPEN, 0)
    with pytest.raises(TypeError):
        ids.decision_id(NS, "2026-09-17", "SPY", "entry", "entry")  # type: ignore[arg-type]


def test_intent_id_validation() -> None:
    did = SHA_ENTRY[:32]
    for bad_decision in (did[:12], did.upper(), "x" * 32, POS, ""):
        with pytest.raises(ValueError, match="decision_id"):
            ids.intent_id(NS, D, bad_decision, OrderPurpose.OPEN, 0)
    for purpose in (OrderPurpose.OPEN, OrderPurpose.CLOSE):
        with pytest.raises(ValueError, match="part"):
            ids.intent_id(NS, D, did, purpose, 1)  # per-leg parts exist for kill fallback orders only
    for part in (-1, 5, 100):
        with pytest.raises(ValueError):
            ids.intent_id(NS, D, did, OrderPurpose.KILL, part)
    with pytest.raises(TypeError):
        ids.intent_id(NS, D, did, OrderPurpose.KILL, True)
    with pytest.raises(ValueError):
        ids.intent_id(NS, D, did, "roll", 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="yymmdd"):
        ids.intent_id(NS, date(2100, 1, 4), did, OrderPurpose.OPEN, 0)
    assert ids.intent_id(NS, D, did, "open", 0) == ids.intent_id(NS, D, did, OrderPurpose.OPEN, 0)  # type: ignore[arg-type]


def test_client_order_id_validation() -> None:
    did = SHA_ENTRY[:32]
    good = _intent(ids.intent_id(NS, D, did, OrderPurpose.OPEN, 0), did, OrderPurpose.OPEN)
    for attempt in (-1, 100):
        with pytest.raises(ValueError):
            ids.client_order_id(good, attempt)
    with pytest.raises(TypeError):
        ids.client_order_id(good, 1.0)  # type: ignore[arg-type]
    assert len(ids.client_order_id(good, 99)) == len(ids.client_order_id(good, 0)) <= 48
    for bad_intent_id in ("", "jbp-a072c982-260917-9d9337b3a17f-open-00", "jb1-a072c982-260917-9d9337b3a17f-open-00-00", "manual"):
        with pytest.raises(ValueError, match="intent_id"):
            ids.client_order_id(_intent(bad_intent_id, did, OrderPurpose.OPEN), 0)


def test_numeric_argument_validation() -> None:
    with pytest.raises(ValueError):
        ids.namespace("exp001", "jev-1.13.0", -1)
    with pytest.raises(TypeError):
        ids.namespace("exp001", "jev-1.13.0", "0")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ids.fill_id("jb1-a072c982-260917-9d9337b3a17f-open-00-00", 0)
    with pytest.raises(TypeError):
        ids.fill_id("jb1-a072c982-260917-9d9337b3a17f-open-00-00", 1.0)  # type: ignore[arg-type]
