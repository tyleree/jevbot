"""canon.py (DESIGN.md 3.7, 5.9): canonical JSON, sha256, ensure_state_safe, cache key.

Golden digests in this file were produced with coreutils (`printf '%s' '<hand-written JSON literal>' | sha256sum`), never with the
code under test. The `canon.cache_key` golden is the INLINE literal required by section 1 / 15.1 (WP03's `golden/cache_key.txt`
is a different artefact: the key set of one full real request).
"""

import copy
import enum
import hashlib
import inspect
import json
import re
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pandas as pd
import pytest

from jevbot import canon
from jevbot.errors import StateError, StateLeak, StateTooLarge, StateTypeError
from jevbot.types import OrderPurpose, Side

REPO = Path(__file__).resolve().parents[2]
DESIGN_MD = REPO / "docs" / "design" / "DESIGN.md"
UNIVERSE = ("SPY", "QQQ", "IWM")


# ======================================================================================================================
# dumps_ordered / dumps_sorted
# ======================================================================================================================


def test_golden_bytes() -> None:
    obj = {"b": 1, "a": {"z": [True, None, -5, 'é "q" \\ \n'], "k": "v"}, "c": []}
    assert canon.dumps_ordered(obj) == '{"b":1,"a":{"z":[true,null,-5,"é \\"q\\" \\\\ \\n"],"k":"v"},"c":[]}'
    assert canon.dumps_sorted(obj) == '{"a":{"k":"v","z":[true,null,-5,"é \\"q\\" \\\\ \\n"]},"b":1,"c":[]}'
    # compact separators, no ASCII escaping of non-ASCII text, no trailing newline
    assert canon.dumps_ordered({"k": "naïve — ✓"}) == '{"k":"naïve — ✓"}'
    assert canon.dumps_ordered([]) == "[]"
    assert canon.dumps_ordered("x") == '"x"'
    assert canon.dumps_ordered(None) == "null"
    assert canon.dumps_sorted(10**30) == "1000000000000000000000000000000"


def test_ordered_keeps_insertion_order_sorted_does_not() -> None:
    forward = {"b": 1, "a": 2, "c": {"y": 1, "x": 2}}
    backward = {"c": {"x": 2, "y": 1}, "a": 2, "b": 1}
    assert forward == backward  # same content for Python ...
    assert canon.dumps_ordered(forward) == '{"b":1,"a":2,"c":{"y":1,"x":2}}'
    assert canon.dumps_ordered(backward) == '{"c":{"x":2,"y":1},"a":2,"b":1}'
    assert canon.dumps_ordered(forward) != canon.dumps_ordered(backward)  # ... different content on the wire (V1)
    assert canon.dumps_sorted(forward) == canon.dumps_sorted(backward) == '{"a":2,"b":1,"c":{"x":2,"y":1}}'
    # list order is content in both encodings
    assert canon.dumps_sorted([2, 1]) == "[2,1]"


def test_matches_the_literal_json_dumps_formula_of_the_contract() -> None:
    obj = {"z": {"b": [1, 2, {"q": None}], "a": "é"}, "a": True}
    assert canon.dumps_ordered(obj) == json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    assert canon.dumps_sorted(obj) == json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def test_ordered_equals_the_sdk_wire_encoding() -> None:
    # 5.9 wire-order guarantee: the SDK serialises with msgspec.json.encode; what we hash must be what is sent
    assert msgspec.json.encode({"b": 1, "a": 2}) == b'{"b":1,"a":2}'
    tricky = {"text": 'quote " backslash \\ slash / tab \t newline \n nul \x00 del \x7f ls   emoji 😀 é', "n": [0, -1, 1000]}
    assert msgspec.json.encode(tricky) == canon.dumps_ordered(tricky).encode("utf-8")


def test_tuples_are_arrays_and_enums_are_their_values() -> None:
    # msgspec.to_builtins keeps tuples, so ledger payloads and ids legitimately contain them
    assert canon.dumps_sorted({"legs": ("a", "b"), "n": ()}) == '{"legs":["a","b"],"n":[]}'

    class Level(enum.IntEnum):
        THREE = 3

    assert (
        canon.dumps_sorted({"side": Side.BUY, "purpose": OrderPurpose.KILL, "lvl": Level.THREE})
        == '{"lvl":3,"purpose":"kill","side":"buy"}'
    )
    assert canon.dumps_sorted({Side.SELL: 1, "a": 2}) == '{"a":2,"sell":1}'


@pytest.mark.parametrize(
    "bad",
    [
        1.5,
        0.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        Decimal("1.10"),
        datetime(2026, 9, 17, 20, 0, tzinfo=UTC),
        date(2026, 9, 17),
        b"bytes",
        bytearray(b"ba"),
        np.float64(1.5),
        np.float32(1.5),
        np.int64(3),
        np.int32(3),
        np.bool_(True),
        np.str_("s"),
        np.array([1, 2]),
        np.datetime64("2026-09-17"),
        pd.Timestamp("2026-09-17", tz="UTC"),
        pd.NA,
        pd.NaT,
        {1, 2},
        frozenset({1}),
        object(),
        1 + 2j,
    ],
    ids=lambda v: type(v).__module__ + "." + type(v).__name__ + ":" + repr(v)[:24],
)
def test_non_canonical_values_are_rejected_everywhere(bad: Any) -> None:
    for wrapped in (bad, [bad], (bad,), {"k": bad}, {"outer": [{"inner": (1, bad)}]}):
        with pytest.raises(TypeError, match="canonical JSON"):
            canon.dumps_ordered(wrapped)
        with pytest.raises(TypeError, match="canonical JSON"):
            canon.dumps_sorted(wrapped)


def test_rejection_names_the_path() -> None:
    with pytest.raises(TypeError, match=re.escape("$.a[1].b")):
        canon.dumps_sorted({"a": [0, {"b": 1.0}]})


@pytest.mark.parametrize("key", [1, 1.5, True, None, (1, 2), b"k", np.str_("k")], ids=repr)
def test_non_string_dict_keys_are_rejected(key: Any) -> None:
    # json.dumps would silently coerce 1 -> "1", True -> "true", None -> "null": {1: x} and {"1": x} would collide
    with pytest.raises(TypeError, match="dict key"):
        canon.dumps_ordered({key: "v"})
    with pytest.raises(TypeError, match="dict key"):
        canon.dumps_sorted({"outer": {key: "v"}})


def test_circular_references_are_a_value_error_not_a_recursion_error() -> None:
    loop: dict[str, Any] = {"a": []}
    loop["a"].append(loop)
    with pytest.raises(ValueError, match="circular"):
        canon.dumps_ordered(loop)
    ring: list[Any] = [1]
    ring.append((ring,))
    with pytest.raises(ValueError, match="circular"):
        canon.dumps_sorted(ring)
    shared = [1, 2]
    assert canon.dumps_ordered({"x": shared, "y": shared}) == '{"x":[1,2],"y":[1,2]}'  # sharing is not a cycle


# ======================================================================================================================
# sha256_hex
# ======================================================================================================================


def test_sha256_hex() -> None:
    # FIPS 180-2 test vectors
    assert canon.sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert canon.sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert canon.sha256_hex(b"abc") == canon.sha256_hex("abc")
    assert canon.sha256_hex("é") == hashlib.sha256(b"\xc3\xa9").hexdigest()  # a str is hashed as UTF-8
    with pytest.raises(TypeError):
        canon.sha256_hex(123)  # type: ignore[arg-type]
    with pytest.raises(UnicodeEncodeError):
        canon.sha256_hex("\ud800")


def test_structure_id_is_sha256_of_dumps_sorted_kind_and_legs() -> None:
    """types.py spells `Structure.structure_id` inline (it cannot import canon); the two spellings must stay byte-identical (2.3).

    Golden `303f7820fb8e2482` = `printf '%s' '["put_credit_spread",["SPY261016P00550000:buy","SPY261016P00560000:sell"]]' | sha256sum`.
    """
    from jevbot.types import Leg, OptionContract, Right, Structure, StructureKind

    def leg(right: Right, strike_milli: int, side: Side) -> Leg:
        return Leg(
            contract=OptionContract(underlying="SPY", expiry=date(2026, 10, 16), right=right, strike_milli=strike_milli), side=side, ratio=1
        )

    legs = (leg(Right.PUT, 550_000, Side.BUY), leg(Right.PUT, 560_000, Side.SELL))
    structure = Structure(
        kind=StructureKind.PUT_CREDIT, underlying="SPY", expiry=date(2026, 10, 16), last_session=date(2026, 10, 16), legs=legs
    )
    # StrEnum members encode by value in canon (a str-valued Enum is a plain scalar), so the f-string spelling of the side is its value
    material = [structure.kind, [f"{lg.contract.occ}:{lg.side.value}" for lg in structure.legs]]
    assert canon.dumps_sorted(material) == '["put_credit_spread",["SPY261016P00550000:buy","SPY261016P00560000:sell"]]'
    assert canon.sha256_hex(canon.dumps_sorted(material))[:16] == structure.structure_id == "303f7820fb8e2482"
    # the leg tuple itself (msgspec keeps tuples) encodes identically to the list spelling
    as_tuples = (structure.kind, tuple(f"{lg.contract.occ}:{lg.side.value}" for lg in structure.legs))
    assert canon.dumps_sorted(as_tuples) == canon.dumps_sorted(material)


# ======================================================================================================================
# The ledger hash chain (2.7): ONE function, ONE spelling of the time columns (INV-19, INV-24)
# ======================================================================================================================

SESSION = date(2024, 5, 17)
AS_OF = datetime(2024, 5, 17, 20, 0, tzinfo=UTC)
# the seq-1 material, written by hand: sorted keys, compact separators, ISO date, RFC 3339 "Z" instant, the kind's value
MATERIAL_1 = (
    '{"as_of":"2024-05-17T20:00:00Z","kind":"mark","payload":{"a":{"y":2,"z":1},"b":[1,"x",null,true]},"seq":1,"session":"2024-05-17"}'
)
# printf '%s\n%s' "$(printf '0%.0s' {1..64})" '<MATERIAL_1>' | sha256sum   (coreutils, not the code under test)
HASH_1 = "86bf32a97b8a585d4deeea62af4ec178fa21421fd73bffdab0701536d441f0ad"


def test_genesis_hash() -> None:
    assert canon.GENESIS_HASH == "0" * 64 and re.fullmatch(r"[0-9a-f]{64}", canon.GENESIS_HASH)


def test_ledger_entry_hash_golden_and_the_literal_formula() -> None:
    payload = {"b": (1, "x", None, True), "a": {"z": 1, "y": 2}}
    got = canon.ledger_entry_hash(canon.GENESIS_HASH, 1, "mark", SESSION, AS_OF, payload)
    assert got == HASH_1
    assert got == hashlib.sha256((canon.GENESIS_HASH + "\n" + MATERIAL_1).encode("utf-8")).hexdigest()
    # the formula of 2.7, literally, with the persisted texts in place of the date / datetime that dumps_sorted refuses
    literal = canon.sha256_hex(
        canon.GENESIS_HASH
        + "\n"
        + canon.dumps_sorted({"seq": 1, "kind": "mark", "session": "2024-05-17", "as_of": "2024-05-17T20:00:00Z", "payload": payload})
    )
    assert got == literal
    # the chain: seq 2 links through seq 1's hash, and any change of any hashed field changes the hash
    second = canon.ledger_entry_hash(
        got, 2, "fee", date(2024, 5, 20), datetime(2024, 5, 20, 20, 5, 0, 250_000, tzinfo=UTC), {"fee_cents": 7}
    )
    material_2 = '{"as_of":"2024-05-20T20:05:00.250000Z","kind":"fee","payload":{"fee_cents":7},"seq":2,"session":"2024-05-20"}'
    assert second == hashlib.sha256((got + "\n" + material_2).encode()).hexdigest()
    variants = [
        canon.ledger_entry_hash("1" + "0" * 63, 1, "mark", SESSION, AS_OF, payload),
        canon.ledger_entry_hash(canon.GENESIS_HASH, 2, "mark", SESSION, AS_OF, payload),
        canon.ledger_entry_hash(canon.GENESIS_HASH, 1, "fee", SESSION, AS_OF, payload),
        canon.ledger_entry_hash(canon.GENESIS_HASH, 1, "mark", date(2024, 5, 16), AS_OF, payload),
        canon.ledger_entry_hash(canon.GENESIS_HASH, 1, "mark", SESSION, AS_OF.replace(second=1), payload),
        canon.ledger_entry_hash(canon.GENESIS_HASH, 1, "mark", SESSION, AS_OF, {**payload, "c": 0}),
    ]
    assert len({got, *variants}) == 7


def test_ledger_entry_hash_over_objects_equals_over_the_persisted_texts() -> None:
    """A store appends from objects and verifies from its TEXT columns (13.4): both routes must give the same hash."""
    from jevbot.types import LedgerKind, Slot

    payload = {"slot": Slot.EOD, "flags": ("a", "b"), "n": {"x": (1, 2)}}
    text = canon.dumps_sorted(payload)
    assert text == '{"flags":["a","b"],"n":{"x":[1,2]},"slot":"eod"}'
    from_objects = canon.ledger_entry_hash(canon.GENESIS_HASH, 1, LedgerKind.SESSION_START, SESSION, AS_OF, payload)
    from_texts = canon.ledger_entry_hash(canon.GENESIS_HASH, 1, "session_start", "2024-05-17", "2024-05-17T20:00:00Z", text)
    from_loaded = canon.ledger_entry_hash(canon.GENESIS_HASH, 1, "session_start", "2024-05-17", "2024-05-17T20:00:00Z", json.loads(text))
    assert from_objects == from_texts == from_loaded
    # a payload text that is not the canonical encoding cannot have been written by a conforming store
    for bad_text in (
        '{"slot": "eod"}',
        '{"slot":"eod","flags":["a","b"],"n":{"x":[1,2]}}',
        "[1]",
        "not json",
        '{"flags":["a","b"],"n":{"x":[1,2]},"slot":"eod"} ',
    ):
        with pytest.raises(ValueError):
            canon.ledger_entry_hash(canon.GENESIS_HASH, 1, "session_start", "2024-05-17", "2024-05-17T20:00:00Z", bad_text)


def test_ledger_entry_hash_refusals() -> None:
    ok = {"i": 1}
    with pytest.raises(TypeError):
        canon.ledger_entry_hash(canon.GENESIS_HASH, 1, "mark", SESSION, AS_OF, {"p": 0.5})  # floats never reach the chain
    with pytest.raises(TypeError):
        canon.ledger_entry_hash(canon.GENESIS_HASH, 1, "mark", SESSION, AS_OF, {"ts": AS_OF})
    with pytest.raises(TypeError):
        canon.ledger_entry_hash(canon.GENESIS_HASH, 1, "mark", SESSION, AS_OF, ["not", "a", "mapping"])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        canon.ledger_entry_hash(canon.GENESIS_HASH, 1, "mark", AS_OF, AS_OF, ok)  # a datetime is not a session
    with pytest.raises(ValueError):
        canon.ledger_entry_hash(canon.GENESIS_HASH, 1, "mark", SESSION, datetime(2024, 5, 17, 20), ok)  # noqa: DTZ001 - naive IS the case
    with pytest.raises(TypeError):
        canon.ledger_entry_hash(canon.GENESIS_HASH, 1, "", SESSION, AS_OF, ok)
    with pytest.raises(TypeError):
        canon.ledger_entry_hash(canon.GENESIS_HASH, 1, 3, SESSION, AS_OF, ok)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        canon.ledger_entry_hash(canon.GENESIS_HASH, True, "mark", SESSION, AS_OF, ok)
    with pytest.raises(TypeError):
        canon.ledger_entry_hash(b"0" * 64, 1, "mark", SESSION, AS_OF, ok)  # type: ignore[arg-type]


def test_render_session_and_render_as_of() -> None:
    from datetime import timedelta, timezone

    assert canon.render_session(SESSION) == "2024-05-17"
    assert canon.render_session(date(2012, 1, 3)) == "2012-01-03"
    with pytest.raises(TypeError):
        canon.render_session(AS_OF)  # a datetime's UTC date and the exchange-local session date can differ
    with pytest.raises(TypeError):
        canon.render_session("2024-05-17")  # type: ignore[arg-type]
    assert canon.render_as_of(AS_OF) == "2024-05-17T20:00:00Z"
    assert canon.render_as_of(datetime(2024, 5, 20, 20, 5, 0, 250_000, tzinfo=UTC)) == "2024-05-20T20:05:00.250000Z"
    assert canon.render_as_of(datetime(2024, 5, 20, 20, 5, 0, 1, tzinfo=UTC)) == "2024-05-20T20:05:00.000001Z"
    assert canon.render_as_of(datetime(2024, 5, 17, 16, 0, tzinfo=timezone(timedelta(hours=-4)))) == "2024-05-17T20:00:00Z"
    assert canon.render_as_of(datetime(2024, 5, 18, 1, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))) == "2024-05-17T20:00:00Z"
    with pytest.raises(ValueError):
        canon.render_as_of(datetime(2024, 5, 17, 20))  # noqa: DTZ001 - naive IS the case
    with pytest.raises(TypeError):
        canon.render_as_of(SESSION)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        canon.render_as_of("2024-05-17T20:00:00Z")  # type: ignore[arg-type]
    # the spelling is msgspec's rendering of a payload datetime (a store that decodes with msgspec reads the same instant back)
    for instant in (
        AS_OF,
        datetime(2024, 5, 20, 20, 5, 0, 250_000, tzinfo=UTC),
        datetime(2024, 5, 17, 16, 0, tzinfo=timezone(timedelta(hours=-4))),
    ):
        utc = instant.astimezone(UTC)
        assert canon.render_as_of(instant) == msgspec.to_builtins(utc) == msgspec.json.decode(msgspec.json.encode(utc))
        assert msgspec.json.decode(msgspec.json.encode(canon.render_as_of(instant)), type=datetime) == instant
        assert datetime.fromisoformat(canon.render_as_of(instant)) == instant
    # the persisted texts are what dumps_sorted needs: it refuses the objects themselves
    with pytest.raises(TypeError):
        canon.dumps_sorted({"as_of": AS_OF})
    with pytest.raises(TypeError):
        canon.dumps_sorted({"session": SESSION})


def test_the_chain_contract_names_the_single_source() -> None:
    assert list(inspect.signature(canon.ledger_entry_hash).parameters) == ["prev_hash", "seq", "kind", "session", "as_of", "payload"]
    for name in ("GENESIS_HASH", "ledger_entry_hash", "render_as_of", "render_session"):
        assert name in canon.__all__


# ======================================================================================================================
# ensure_state_safe
# ======================================================================================================================


def _design_entry_state() -> dict[str, Any]:
    text = DESIGN_MD.read_text(encoding="utf-8")
    section = text[text.index("### 5.6 `state.v1.entry`") :]
    block = re.search(r"```json\n(.*?)```", section, flags=re.DOTALL)
    assert block is not None
    state: dict[str, Any] = json.loads(block.group(1))
    return state


def test_the_design_entry_state_is_state_safe() -> None:
    state = _design_entry_state()
    assert state["schema"] == "state.v1.entry"
    canon.ensure_state_safe(state, underlyings=UNIVERSE)
    canon.ensure_state_safe(state, underlyings=UNIVERSE, max_chars=16000)
    # ... and with the entry_text news block appended
    state["news_status"] = "present"
    state["news"] = {
        "since_previous_session": [
            {
                "age": "6h",
                "source_type": "newswire",
                "headline": "the central bank holds rates, [number] officials dissent",
                "summary": None,
            }
        ],
        "earlier": [
            {"age": "2d", "source_type": "newswire", "headline": "a large company beats estimates in [period] [year]", "summary": "x"}
        ],
    }
    canon.ensure_state_safe(state, underlyings=UNIVERSE)


def test_accepts_every_allowed_type_and_boundary_ints() -> None:
    ok = {"s": "text", "i": 1000, "neg": -1000, "zero": 0, "t": True, "f": False, "n": None, "l": [1, "a", None, [], {}], "d": {"k_1": {}}}
    canon.ensure_state_safe(ok)
    canon.ensure_state_safe("a bare string")
    canon.ensure_state_safe([])
    canon.ensure_state_safe(None)


@pytest.mark.parametrize(
    "bad",
    [
        1.5,
        float("nan"),
        (1, 2),
        (),
        Decimal("1"),
        datetime(2026, 9, 17, tzinfo=UTC),
        date(2026, 9, 17),
        b"x",
        np.int64(5),
        np.float64(0.5),
        np.bool_(True),
        np.str_("s"),
        pd.Timestamp("2026-09-17"),
        {1, 2},
        Side.BUY,  # an enum is not a plain str: states hold builtins only
    ],
    ids=lambda v: type(v).__module__ + "." + type(v).__name__,
)
def test_rejects_non_builtin_types(bad: Any) -> None:
    for wrapped in (bad, [bad], {"k": bad}, {"a": {"b": [0, bad]}}):
        with pytest.raises(StateTypeError):
            canon.ensure_state_safe(wrapped)
        with pytest.raises(StateTypeError):  # the type rules hold in unmasked mode too
            canon.ensure_state_safe(wrapped, masked=False)


def test_rejects_container_subclasses() -> None:
    class MyDict(dict):  # type: ignore[type-arg]
        pass

    class MyList(list):  # type: ignore[type-arg]
        pass

    with pytest.raises(StateTypeError):
        canon.ensure_state_safe(MyDict(a=1))
    with pytest.raises(StateTypeError):
        canon.ensure_state_safe({"a": MyList([1])})


@pytest.mark.parametrize("value", [1001, -1001, 45000, 10**12])
def test_rejects_big_ints(value: int) -> None:
    with pytest.raises(StateTypeError, match="1000"):
        canon.ensure_state_safe({"value": value})
    with pytest.raises(StateTypeError):
        canon.ensure_state_safe({"value": value}, masked=False)
    canon.ensure_state_safe({"flag": True})  # a bool is not a big int (and True is not 1001)


@pytest.mark.parametrize("key", ["Upper", "1abc", "_lead", "has-dash", "has space", "dotted.key", "", "ünï", "..."])
def test_rejects_malformed_keys(key: str) -> None:
    with pytest.raises(StateTypeError, match="key"):
        canon.ensure_state_safe({key: 1})
    with pytest.raises(StateTypeError, match="key"):
        canon.ensure_state_safe({key: 1}, masked=False)


def test_rejects_non_string_keys() -> None:
    for key in (1, None, True, ("a",), Side.BUY):
        with pytest.raises(StateTypeError):
            canon.ensure_state_safe({key: 1})


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("SPY looks strong", "underlying"),
        ("rotation into QQQ.", "underlying"),
        ("(IWM)", "underlying"),
        ("as of 2024-03-15 the", "iso_date"),
        ("since 2019", "year"),
        ("in 1987 markets", "year"),
        ("FY2024 guidance", None),  # no word boundary inside FY2024 for the year rule ... (the long-number rule has none either)
        ("costs $5 more", "dollar_amount"),
        ("costs $ 5 more", "dollar_amount"),
        ("price near 450", "long_number"),
        ("price near 450.25", "long_number"),
        ("a 1,000 point move", "long_number"),
        ("about 12345 contracts", "long_number"),
    ],
)
def test_leak_patterns(text: str, code: str | None) -> None:
    if code is None:
        canon.ensure_state_safe({"headline": text}, underlyings=UNIVERSE)
        return
    for wrapped in (text, [text], {"headline": text}, {"news": {"earlier": [{"headline": text}]}}):
        with pytest.raises(StateLeak, match=code):
            canon.ensure_state_safe(wrapped, underlyings=UNIVERSE)
        canon.ensure_state_safe(wrapped, masked=False, underlyings=UNIVERSE)  # the flagged diagnostic mode skips leak rules only


@pytest.mark.parametrize(
    "text",
    [
        "low: 20th to 40th percentile of the past year",
        "2_to_4_percent: one expected move over this horizon is about 2 to 4 percent",
        "major central-bank rate decision in 2 sessions",
        "price above rising 20-day and 50-day averages",
        "UNDERLYING_A",
        "spy agencies, a spyglass and ESPYS",  # tickers are matched case-sensitively and word-bounded
        "up 99 percent in 12 sessions",
        "[number] basis points, [year], [month] [day]",
        "state.v1.entry",
        "cash",  # no `$` amount
        "US$ terms",
    ],
)
def test_legitimate_strings_pass(text: str) -> None:
    canon.ensure_state_safe({"field": text}, underlyings=UNIVERSE)


def test_underlyings_are_only_checked_when_given_and_are_regex_escaped() -> None:
    canon.ensure_state_safe({"headline": "SPY looks strong"})  # no underlyings given: nothing to match
    with pytest.raises(StateLeak):
        canon.ensure_state_safe({"headline": "BRK.B rallies"}, underlyings=("BRK.B",))
    canon.ensure_state_safe({"headline": "BRKXB rallies"}, underlyings=("BRK.B",))  # "." is literal, not a wildcard
    with pytest.raises(ValueError, match="non-empty"):
        canon.ensure_state_safe({"a": "b"}, underlyings=("SPY", ""))  # an empty alternative would match everything
    with pytest.raises(TypeError):
        canon.ensure_state_safe({"a": "b"}, underlyings="SPY")


@pytest.mark.parametrize("key", ["uid", "uuid", "timestamp", "ts", "date", "datetime", "time", "created_at"])
def test_forbidden_keys(key: str) -> None:
    with pytest.raises(StateLeak, match=key):
        canon.ensure_state_safe({"context": {key: "x"}})
    # the documented unmasked probe state {"schema": ..., "ticker": "SPY", "date": "2024-03-15"} (6.6) must pass with masked=False
    canon.ensure_state_safe({"context": {key: "x"}}, masked=False)


def test_forbidden_keys_match_the_whole_name_only() -> None:
    canon.ensure_state_safe({"time_to_expiry": {"value": 27}, "as_of": "prior session close", "update_time_bucket": "x", "dates": 1})
    canon.ensure_state_safe({"schema": "state.v1.probe_recall", "ticker": "SPY", "date": "2024-03-15"}, masked=False)
    with pytest.raises(StateLeak):
        canon.ensure_state_safe({"schema": "state.v1.probe_recall", "ticker": "SPY", "date": "2024-03-15"}, underlyings=UNIVERSE)


def test_dict_keys_are_screened_like_any_other_string() -> None:
    # "any string": a key is a string too. Real tickers are upper case and keys are lower case, so this needs a lower-case symbol
    with pytest.raises(StateLeak, match="underlying"):
        canon.ensure_state_safe({"context": {"spy": "x"}}, underlyings=("spy",))
    canon.ensure_state_safe({"context": {"spy_like": "x", "y2024": 1, "a_450": 2}}, underlyings=("spy",))  # no word boundary inside a key
    canon.ensure_state_safe({"context": {"spy": "x"}}, masked=False, underlyings=("spy",))


def test_leak_messages_never_echo_third_party_text() -> None:
    with pytest.raises(StateLeak) as info:
        canon.ensure_state_safe({"news": [{"headline": "secret phrase about SPY 2024-03-15"}]}, underlyings=UNIVERSE)
    assert "secret phrase" not in str(info.value)
    assert "$.news[0].headline" in str(info.value)


def test_state_too_large() -> None:
    state = {"text": "x" * 100}
    size = len(canon.dumps_ordered(state))
    canon.ensure_state_safe(state, max_chars=size)
    with pytest.raises(StateTooLarge):
        canon.ensure_state_safe(state, max_chars=size - 1)
    canon.ensure_state_safe(state)  # no cap unless asked


def test_state_errors_share_one_base_and_cycles_are_type_errors() -> None:
    assert issubclass(StateTypeError, StateError) and issubclass(StateLeak, StateError) and issubclass(StateTooLarge, StateError)
    loop: dict[str, Any] = {}
    loop["self"] = loop
    with pytest.raises(StateTypeError, match="circular"):
        canon.ensure_state_safe(loop)
    ring: list[Any] = []
    ring.append({"inner": ring})
    with pytest.raises(StateTypeError, match="circular"):
        canon.ensure_state_safe(ring)
    shared = {"value": 1}
    canon.ensure_state_safe({"a": shared, "b": [shared, shared]})  # sharing is not a cycle


def test_the_documented_signature() -> None:
    params = inspect.signature(canon.ensure_state_safe).parameters
    assert list(params)[:3] == ["obj", "masked", "underlyings"]
    assert params["masked"].default is True and params["masked"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["underlyings"].default == () and params["underlyings"].kind is inspect.Parameter.KEYWORD_ONLY


# ======================================================================================================================
# cache key (D8)
# ======================================================================================================================

MODEL = "jev-1.13.0"
QSET_HASH = "ab" * 32
STATE: dict[str, Any] = {"schema": "state.v1.entry", "b": {"value": 3, "bucket": "low: é"}, "a": [1, True, None]}
QUESTION: dict[str, Any] = {"type": "noul", "instructions": "Is `b` low?", "criteria": {"true": 'yes "quoted"', "false": "no"}}
CHOICE: dict[str, Any] = {
    "type": "choice",
    "instructions": "Which stance?",
    "criteria": {"sell_premium": "rich", "buy_premium": "cheap", "unclear": "neither"},
}

# printf '%s' '{"v":1,"model":"jev-1.13.0","state":{"schema":"state.v1.entry","b":{"value":3,"bucket":"low: é"},"a":[1,true,null]},
#   "question_set_hash":"abab...ab","question":{"type":"noul","instructions":"Is `b` low?","criteria":{"true":"yes \"quoted\"","false":"no"}}}'
#   | sha256sum        (one line, 310 bytes, UTF-8)
CACHE_KEY_GOLDEN = "662b03707de939e27ebbdb41df38424d244e80e6a0f085c65be3a04797966cc9"
# the same request with every state dict in REVERSED key order (the KEY_PERM variant of 5.9)
CACHE_KEY_KEY_PERM_GOLDEN = "96f22978e0bec9175efb5ad4529182c87a96bea91d666d72072425ccc70a3c35"


def _key_perm(obj: Any) -> Any:
    """5.9 KEY_PERM: every dict rebuilt with reversed key order at every level (lists untouched)."""
    if isinstance(obj, dict):
        return {k: _key_perm(obj[k]) for k in reversed(list(obj))}
    if isinstance(obj, list):
        return [_key_perm(v) for v in obj]
    return obj


def _opt_perm(question: dict[str, Any]) -> dict[str, Any]:
    """5.9 OPT_PERM: a Choice question's `criteria` rebuilt in reversed option order."""
    out = dict(question)
    out["criteria"] = {k: question["criteria"][k] for k in reversed(list(question["criteria"]))}
    return out


def test_cache_key_golden() -> None:
    assert canon.cache_key(MODEL, STATE, QSET_HASH, QUESTION) == CACHE_KEY_GOLDEN
    assert canon.cache_key(MODEL, _key_perm(STATE), QSET_HASH, QUESTION) == CACHE_KEY_KEY_PERM_GOLDEN


def test_cache_key_is_the_d8_formula_literally() -> None:
    material = {"v": 1, "model": MODEL, "state": STATE, "question_set_hash": QSET_HASH, "question": CHOICE}
    assert canon.cache_key(MODEL, STATE, QSET_HASH, CHOICE) == canon.sha256_hex(canon.dumps_ordered(material))
    assert list(material) == ["v", "model", "state", "question_set_hash", "question"]


def test_base_key_perm_and_opt_perm_keys_all_differ() -> None:
    base = canon.cache_key(MODEL, STATE, QSET_HASH, CHOICE)
    key_perm = canon.cache_key(MODEL, _key_perm(STATE), QSET_HASH, CHOICE)
    opt_perm = canon.cache_key(MODEL, STATE, QSET_HASH, _opt_perm(CHOICE))
    both = canon.cache_key(MODEL, _key_perm(STATE), QSET_HASH, _opt_perm(CHOICE))
    assert len({base, key_perm, opt_perm, both}) == 4
    # the variants are the same content for a sorted hash - which is exactly why the Jev-facing encoding is ordered (V1)
    assert canon.dumps_sorted(_key_perm(STATE)) == canon.dumps_sorted(STATE)
    assert canon.dumps_sorted(_opt_perm(CHOICE)) == canon.dumps_sorted(CHOICE)


def test_every_member_of_the_key_matters() -> None:
    base = canon.cache_key(MODEL, STATE, QSET_HASH, QUESTION)
    other_state = copy.deepcopy(STATE)
    other_state["b"]["value"] = 4
    other_question = copy.deepcopy(QUESTION)
    other_question["instructions"] = "Is `b` low ?"
    keys = {
        base,
        canon.cache_key("jev-1.14.0", STATE, QSET_HASH, QUESTION),
        canon.cache_key(MODEL, other_state, QSET_HASH, QUESTION),
        canon.cache_key(MODEL, STATE, "cd" * 32, QUESTION),
        canon.cache_key(MODEL, STATE, QSET_HASH, other_question),
    }
    assert len(keys) == 5
    assert re.fullmatch(r"[0-9a-f]{64}", base)


def test_question_id_namespace_and_sample_index_are_not_in_the_key() -> None:
    # the key is pure content: the same question dict asked under two ids (or namespaces) is ONE cache entry
    assert list(inspect.signature(canon.cache_key).parameters) == ["model", "state", "question_set_hash", "question"]
    batch = {"vol.stance": CHOICE, "some.other_id": copy.deepcopy(CHOICE)}
    keys = {qid: canon.cache_key(MODEL, STATE, QSET_HASH, q) for qid, q in batch.items()}
    assert keys["vol.stance"] == keys["some.other_id"]
    assert canon.cache_key(MODEL, STATE, QSET_HASH, CHOICE) == canon.cache_key(MODEL, copy.deepcopy(STATE), QSET_HASH, CHOICE)


def test_cache_key_refuses_floats_and_bad_arguments() -> None:
    with pytest.raises(TypeError, match="canonical JSON"):
        canon.cache_key(MODEL, {"x": 0.5}, QSET_HASH, QUESTION)
    with pytest.raises(TypeError, match="canonical JSON"):
        canon.cache_key(MODEL, STATE, QSET_HASH, {"type": "score", "weight": 1.0})
    with pytest.raises(TypeError):
        canon.cache_key("", STATE, QSET_HASH, QUESTION)
    with pytest.raises(TypeError):
        canon.cache_key(MODEL, STATE, "", QUESTION)
    with pytest.raises(TypeError):
        canon.cache_key(MODEL, [STATE], QSET_HASH, QUESTION)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        canon.cache_key(MODEL, STATE, QSET_HASH, "question")  # type: ignore[arg-type]


def test_hash_formulas_of_section_5_9() -> None:
    questions = {"q.one": QUESTION, "q.two": CHOICE}
    state_hash = canon.sha256_hex(canon.dumps_ordered(STATE))
    question_set_hash = canon.sha256_hex(canon.dumps_ordered(list(questions.values())))
    assert state_hash == hashlib.sha256(json.dumps(STATE, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    # batch order is bound, ids are not (they are not sent)
    reordered = {"q.two": CHOICE, "q.one": QUESTION}
    renamed = {"x.one": QUESTION, "x.two": CHOICE}
    assert canon.sha256_hex(canon.dumps_ordered(list(reordered.values()))) != question_set_hash
    assert canon.sha256_hex(canon.dumps_ordered(list(renamed.values()))) == question_set_hash
