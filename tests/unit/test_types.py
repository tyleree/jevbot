"""Contract tests for src/jevbot/types.py, src/jevbot/errors.py and the project setup files (WP00).

The "verbatim" tests parse the fenced contract blocks of DESIGN.md sections 2.1-2.9 and 2.11 at test time and compare them with
the importable code, so the spec and the code cannot drift apart silently. Behavioural tests use hand-computed numbers
(structure-id goldens were produced with `printf ... | sha256sum`, not with the code under test).
"""

import dataclasses
import enum
import pickle
import re
import subprocess
import sys
import tomllib
import types as pytypes
import typing
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import msgspec
import pandas as pd
import pytest

import jevbot
import jevbot.errors as errors
import jevbot.types as T
from jevbot.errors import DataError, DataUnavailable, ExitCode, exit_code_for
from jevbot.types import (
    CHAIN_COLUMNS,
    Band,
    BandPrices,
    ChainSnapshot,
    Direction,
    Fidelity,
    Leg,
    OptionContract,
    Quote,
    Right,
    Side,
    Slot,
    SmileFit,
    SnapshotKey,
    Structure,
    StructureKind,
    VolStance,
)

REPO = Path(__file__).resolve().parents[2]
DESIGN_MD = REPO / "docs" / "design" / "DESIGN.md"


# ======================================================================================================================
# DESIGN.md parsing helpers
# ======================================================================================================================


@pytest.fixture(scope="module")
def design() -> str:
    return DESIGN_MD.read_text(encoding="utf-8")


def section(text: str, number: str) -> str:
    """The section whose heading is `## <number>` / `### <number>`, up to the next heading of level 2 or 3."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if re.match(rf"^#{{2,3}} {re.escape(number)}[ .]", line))
    end = next((j for j in range(start + 1, len(lines)) if re.match(r"^#{2,3} ", lines[j])), len(lines))
    return "\n".join(lines[start:end])


def fenced(section_text: str, lang: str) -> list[str]:
    return re.findall(rf"```{lang}\n(.*?)```", section_text, re.S)


def strip_comment(line: str) -> str:
    return line.split("#", 1)[0].rstrip()


def parse_enums(block: str) -> dict[str, dict[str, str]]:
    """`class Name(StrEnum):  A="a"; B="b"` with indented continuation lines -> {Name: {A: a, B: b}}."""
    out: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for raw in block.splitlines():
        code = strip_comment(raw)
        if not code.strip():
            continue
        head = re.match(r"^class (\w+)\(StrEnum\):(.*)$", code)
        if head:
            current = out.setdefault(head[1], {})
            code = head[2]
        elif not raw.startswith(" "):
            current = None
            continue
        if current is not None:
            for name, value in re.findall(r'(\w+)\s*=\s*"([^"]*)"', code):
                assert name not in current, f"duplicate member {name}"
                current[name] = value
    return out


@dataclass
class SpecClass:
    name: str
    header: str
    fields: list[tuple[str, str, str | None]] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)


def parse_classes(block: str) -> list[SpecClass]:
    """Struct / dataclass contract blocks: field lines (`a: T = d`, several per line with `;`) and method names."""
    out: list[SpecClass] = []
    current: SpecClass | None = None
    for raw in block.splitlines():
        code = strip_comment(raw)
        if not code.strip():
            continue
        head = re.match(r"^class (\w+)(?:\((.*)\))?:", code)
        if head:
            current = SpecClass(name=head[1], header=head[2] or "")
            out.append(current)
            continue
        if not raw.startswith(" "):  # any other top-level statement ends the class body
            current = None
            continue
        if current is None or not re.match(r"^ {4}\S", code):  # deeper indentation = continuation of a comment / signature
            continue
        stmt = code.strip()
        if stmt.startswith("@"):
            continue
        if stmt.startswith("def "):
            name = re.match(r"def (\w+)", stmt)
            assert name is not None
            current.methods.append(name[1])
            continue
        for part in stmt.split(";"):
            parsed = re.match(r"^(\w+): (.+?)(?: = (.+))?$", part.strip())
            assert parsed is not None, (current.name, part)
            current.fields.append((parsed[1], parsed[2].strip(), parsed[3]))
    return out


@pytest.fixture(scope="module")
def spec_classes(design: str) -> dict[str, SpecClass]:
    found: dict[str, SpecClass] = {}
    for number in ("2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8"):
        for block in fenced(section(design, number), "python"):
            for cls in parse_classes(block):
                assert cls.name not in found, f"{cls.name} defined twice in DESIGN.md"
                found[cls.name] = cls
    return found


def module_structs() -> dict[str, type[msgspec.Struct]]:
    return {
        name: obj
        for name, obj in vars(T).items()
        if isinstance(obj, type) and issubclass(obj, msgspec.Struct) and obj.__module__ == T.__name__
    }


# ======================================================================================================================
# 2.1 enums and static tables - verbatim against DESIGN.md
# ======================================================================================================================


def test_enums_equal_design_section_2_1(design: str) -> None:
    spec = parse_enums(fenced(section(design, "2.1"), "python")[0])
    assert len(spec) == 26, sorted(spec)  # every `class X(StrEnum)` of the contract block
    actual = {
        name: {member.name: member.value for member in obj}
        for name, obj in vars(T).items()
        if isinstance(obj, type) and issubclass(obj, enum.StrEnum) and obj.__module__ == T.__name__
    }
    assert actual == spec
    for name, members in spec.items():  # declaration ORDER is part of the contract too (e.g. `list(Band)`)
        assert [m.name for m in getattr(T, name)] == list(members), name


def test_enum_values_spot_checks() -> None:
    # independent transcription of values other modules persist or send to the broker
    assert T.Right.CALL == "C" and T.Right.PUT == "P"
    assert [i.value for i in T.PositionIntent] == ["buy_to_open", "sell_to_open", "buy_to_close", "sell_to_close"]
    assert T.OrderStatus.PARTIAL.value == "partially_filled"
    assert T.ExitReason.FORCE_EXPIRY.value == "force_exit_expiry"
    assert T.ExitReason.JEV.value == "jev_discretionary"
    assert T.ExitReason.KILL.value == "kill_switch"
    assert T.Direction.NEUTRAL.value == "neutral_range"
    assert T.StructureKind.CALL_DEBIT.value == "call_debit_spread"
    assert T.EvidenceTier.NONE.value == "NONE"
    assert str(T.LedgerKind.RISK_VERDICT) == "risk_verdict"  # StrEnum: the value is what is persisted
    assert typing.get_args(T.DecisionKind) == ("entry", "manage")
    assert T.Cents is int and T.Micros is int and T.Ppm is int and T.Bp is int


def test_terminal_statuses_and_mandatory_exits_equal_design(design: str) -> None:
    block = fenced(section(design, "2.1"), "python")[0]
    terminal = re.search(r"TERMINAL_STATUSES = \{(.*?)\}", block)
    mandatory = re.search(r"MANDATORY_EXITS = \{(.*?)\}", block)
    assert terminal is not None and mandatory is not None
    assert frozenset(T.OrderStatus[n.strip()] for n in terminal[1].split(",")) == T.TERMINAL_STATUSES
    assert frozenset(T.ExitReason[n.strip()] for n in mandatory[1].split(",")) == T.MANDATORY_EXITS
    assert isinstance(T.TERMINAL_STATUSES, frozenset) and isinstance(T.MANDATORY_EXITS, frozenset)
    assert {s.value for s in T.TERMINAL_STATUSES} == {"filled", "cancelled", "rejected", "expired"}
    assert {r.value for r in T.MANDATORY_EXITS} == {"force_exit_expiry", "ex_dividend", "assignment_risk", "kill_switch"}
    # there is NO text-only exit reason (INV-16)
    assert not any("text" in r.value and r is not T.ExitReason.TEXT_CONFIRMED for r in T.ExitReason)


def _braces(block: str, name: str) -> str:
    start = block.index(name)
    open_ = block.index("{", start)
    return block[open_ + 1 : block.index("}", open_)]


def test_static_tables_equal_design(design: str) -> None:
    block = fenced(section(design, "2.1"), "python")[1]
    direction = {StructureKind[k]: Direction[v] for k, v in re.findall(r"([A-Z_]+): ([A-Z_]+)", _braces(block, "STRUCTURE_DIRECTION:"))}
    stance = {StructureKind[k]: VolStance[v] for k, v in re.findall(r"([A-Z_]+): ([A-Z_]+)", _braces(block, "STRUCTURE_STANCE:"))}
    mapping = {
        (Direction[d], VolStance[s]): (None if k == "None" else StructureKind[k])
        for d, s, k in re.findall(r"\(([A-Z_]+), ([A-Z_]+)\): ([A-Za-z_]+)", _braces(block, "MAPPING:"))
    }
    short = re.search(r"SHORT_PREMIUM = frozenset\(\{(.*?)\}\)", block)
    assert short is not None
    assert len(direction) == 7 and len(stance) == 7 and len(mapping) == 9
    assert dict(T.STRUCTURE_DIRECTION) == direction
    assert dict(T.STRUCTURE_STANCE) == stance
    assert dict(T.MAPPING) == mapping
    assert frozenset(StructureKind[n.strip()] for n in short[1].split(",")) == T.SHORT_PREMIUM


def test_mapping_equals_the_7_3_table(design: str) -> None:
    rows = [line for line in section(design, "7.3").splitlines() if line.startswith("|")]
    header = [c.strip() for c in rows[0].strip("|").split("|")]
    assert header[1:] == ["sell_premium", "buy_premium", "limit_vol_exposure"]
    parsed: dict[tuple[str, str], str | None] = {}
    for row in rows[2:5]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        for stance, kind in zip(header[1:], cells[1:], strict=True):
            parsed[(cells[0], stance)] = None if kind == "no trade" else kind
    actual = {(d.value, s.value): (k.value if k is not None else None) for (d, s), k in T.MAPPING.items()}
    assert actual == parsed
    assert parsed[("neutral_range", "sell_premium")] == "iron_condor" and parsed[("neutral_range", "buy_premium")] is None


def test_static_tables_are_consistent_and_immutable() -> None:
    kinds = [k for k in T.MAPPING.values() if k is not None]
    assert sorted(kinds) == sorted(StructureKind)  # every structure is reachable from exactly one (direction, stance) cell
    for (direction, stance), kind in T.MAPPING.items():
        if kind is not None:
            assert T.STRUCTURE_DIRECTION[kind] is direction
            assert T.STRUCTURE_STANCE[kind] is stance
    assert {k for k, s in T.STRUCTURE_STANCE.items() if s is VolStance.SELL} == T.SHORT_PREMIUM
    for table in (T.STRUCTURE_DIRECTION, T.STRUCTURE_STANCE, T.MAPPING, T.LEDGER_PAYLOAD_FIELDS, T.LEDGER_PAYLOAD_OPTIONAL_FIELDS):
        with pytest.raises(TypeError):
            table[StructureKind.LONG_CALL] = Direction.BEARISH  # type: ignore[index]


# ======================================================================================================================
# 2.2-2.8 structs - verbatim against DESIGN.md
# ======================================================================================================================


def test_every_contract_class_exists_and_nothing_else(spec_classes: dict[str, SpecClass]) -> None:
    assert len(spec_classes) == 51  # 50 Structs + the ChainSnapshot dataclass
    assert set(module_structs()) | {"ChainSnapshot"} == set(spec_classes)


def test_struct_fields_types_defaults_equal_design(spec_classes: dict[str, SpecClass]) -> None:
    namespace = dict(vars(T))
    for name, cls in module_structs().items():
        spec = spec_classes[name]
        infos = msgspec.structs.fields(cls)  # also proves that every annotation resolves
        assert [f.name for f in infos] == [f[0] for f in spec.fields], name
        assert cls.__struct_fields__ == tuple(f[0] for f in spec.fields), name
        for info, (fname, annotation, default) in zip(infos, spec.fields, strict=True):
            assert info.type == eval(annotation, namespace), f"{name}.{fname}"
            if default is None:
                assert info.required, f"{name}.{fname} must be required"
            else:
                assert not info.required and info.default == eval(default, namespace), f"{name}.{fname}"


def test_struct_config_equals_design(spec_classes: dict[str, SpecClass]) -> None:
    # "All structs are msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True unless noted"
    for name, cls in module_structs().items():
        config = cls.__struct_config__
        header = spec_classes[name].header
        assert header.startswith("Struct"), name
        assert config.frozen and config.forbid_unknown_fields and config.eq, name
        assert config.order == ("order=True" in header), name
        tag = re.search(r'tag="(\w+)"', header)
        assert config.tag == (tag[1] if tag else None), name
        assert not config.array_like and not config.omit_defaults, name  # payload shape = field names, defaults included
    assert [n for n, c in module_structs().items() if c.__struct_config__.order] == ["OptionContract", "SnapshotKey"]
    assert {c.__struct_config__.tag for c in (T.NoulAns, T.ChoiceAns, T.ScoreAns)} == {"noul", "choice", "score"}


def test_contract_methods_exist(spec_classes: dict[str, SpecClass]) -> None:
    for name, spec in spec_classes.items():
        for method in spec.methods:
            assert hasattr(getattr(T, name), method), f"{name}.{method}"
    for cls, prop in [(OptionContract, "occ"), (Quote, "mid2"), (Structure, "structure_id"), (Structure, "width")]:
        assert isinstance(vars(cls)[prop], property), f"{cls.__name__}.{prop} must be a property"
    for prop in ("wing_widths", "direction", "short_legs"):
        assert isinstance(vars(Structure)[prop], property)
    assert not isinstance(vars(Quote)["valid"], property)  # valid() / usable_*() are methods in the contract


def test_chain_snapshot_dataclass_equals_design(spec_classes: dict[str, SpecClass]) -> None:
    spec = spec_classes["ChainSnapshot"]
    namespace = dict(vars(T))
    fields = dataclasses.fields(ChainSnapshot)
    assert [f.name for f in fields] == [f[0] for f in spec.fields]
    for f, (fname, annotation, default) in zip(fields, spec.fields, strict=True):
        assert f.type == eval(annotation, namespace), fname
        assert default is None and f.default is dataclasses.MISSING, fname
    params = ChainSnapshot.__dataclass_params__  # type: ignore[attr-defined]
    assert params.frozen and not params.eq


def test_chain_columns_equal_design(design: str) -> None:
    block = fenced(section(design, "2.2"), "python")[0]
    listed = re.search(r"CHAIN_COLUMNS = \[(.*?)\]", block, re.S)
    assert listed is not None
    assert re.findall(r'"(\w+)"', listed[1]) == CHAIN_COLUMNS
    assert len(CHAIN_COLUMNS) == 17 and "volume" not in CHAIN_COLUMNS  # deliberately NO same-day volume column (0.1 item 12)


# ----------------------------------------------------------------------------------------------------------------------
# generic struct behaviour: frozen, keyword-only, unknown fields refused, lossless JSON / builtins round trips
# ----------------------------------------------------------------------------------------------------------------------

SAMPLE_CONTRACT = OptionContract(underlying="SPY", expiry=date(2026, 10, 16), right=Right.CALL, strike_milli=600000)


def sample(tp: Any, *, none: bool) -> Any:
    """A deterministic value of type `tp`; `none=True` picks None for every Optional."""
    origin = typing.get_origin(tp)
    if tp is Any:
        return {"k": [1, "x", None, True]}
    if tp is bool:
        return True
    if tp is int:
        return 7
    if tp is str:
        return "s"
    if tp is float:
        return 0.25
    if tp is date:
        return date(2026, 10, 16)
    if tp is datetime:
        return datetime(2026, 9, 17, 19, 35, 0, tzinfo=UTC)
    if tp is OptionContract:
        return SAMPLE_CONTRACT
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return next(iter(tp))
    if isinstance(tp, type) and issubclass(tp, msgspec.Struct):
        return tp(**{f.name: sample(f.type, none=none) for f in msgspec.structs.fields(tp)})
    if origin is tuple:
        args = typing.get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return (sample(args[0], none=none),)
        return tuple(sample(a, none=none) for a in args)
    if origin is dict:
        key, value = typing.get_args(tp)
        return {sample(key, none=none): sample(value, none=none)}
    if origin in (pytypes.UnionType, typing.Union):
        arms = typing.get_args(tp)
        if none and type(None) in arms:
            return None
        return sample(next(a for a in arms if a is not type(None)), none=none)
    raise AssertionError(f"no sample for {tp!r}")


DECODABLE = sorted(set(module_structs()) - {"BuiltState"})


@pytest.mark.parametrize("name", DECODABLE)
@pytest.mark.parametrize("none", [False, True], ids=["populated", "optionals_none"])
def test_struct_round_trips_losslessly(name: str, none: bool) -> None:
    cls = module_structs()[name]
    value = sample(cls, none=none)
    assert msgspec.json.decode(msgspec.json.encode(value), type=cls) == value
    assert msgspec.convert(msgspec.to_builtins(value), cls) == value
    assert set(msgspec.to_builtins(value)) - {"type"} == set(cls.__struct_fields__)  # payload keys = field names (+ the union tag)


@pytest.mark.parametrize("name", sorted(module_structs()))
def test_struct_is_frozen_keyword_only_and_refuses_unknown_fields(name: str) -> None:
    cls = module_structs()[name]
    value = sample(cls, none=False)
    first = cls.__struct_fields__[0]
    with pytest.raises(AttributeError):
        setattr(value, first, getattr(value, first))
    with pytest.raises(TypeError):
        cls(*[getattr(value, f) for f in cls.__struct_fields__])  # positional construction is not part of the contract
    if name != "BuiltState":
        payload = msgspec.to_builtins(value)
        payload["not_a_field"] = 1
        with pytest.raises(msgspec.ValidationError):
            msgspec.convert(payload, cls)


def test_required_fields_are_enforced() -> None:
    with pytest.raises(TypeError):
        Leg(side=Side.BUY)  # type: ignore[call-arg]
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert({"orats": 1, "worst": 2}, BandPrices)
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert({"orats": 1, "worst": 2, "mid": 1.5}, BandPrices)  # integer money: a float price is refused on decode


def test_built_state_encodes_but_is_in_memory_only() -> None:
    built = sample(T.BuiltState, none=False)
    assert isinstance(built.facts, T.EntryFacts)
    payload = msgspec.to_builtins(built)
    assert payload["facts"] == msgspec.to_builtins(built.facts)
    assert "type" not in payload["facts"]  # EntryFacts / ManageFacts stay untagged: their builtins are hashed in DECISION payloads
    manage = T.BuiltState(state={}, state_hash="h", provenance=built.provenance, facts=sample(T.ManageFacts, none=True))
    assert msgspec.to_builtins(manage)["facts"]["short_dist_code"] is None


def test_answer_is_a_tagged_union() -> None:
    noul = msgspec.json.decode(b'{"type":"noul","p":0.25}', type=T.Answer)
    assert noul == T.NoulAns(p=0.25)
    score = T.ScoreAns(
        probs=(0.5, 0.25, 0.25, 0.0),
        mean=0.75,
        norm=0.25,
        top=0,
        p_top=0.5,
        margin=0.25,
        entropy=0.75,
        raw_sum=1.0,
        server_score=1.0,
        server_confidence=0.5,
    )
    result = sample(T.DecisionResult, none=True)
    result = msgspec.structs.replace(result, answers={"under.stretched": noul, "risk.environment": score})
    decoded = msgspec.json.decode(msgspec.json.encode(result), type=T.DecisionResult)
    assert decoded == result
    assert isinstance(decoded.answers["risk.environment"], T.ScoreAns)
    assert msgspec.to_builtins(noul) == {"type": "noul", "p": 0.25}
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(b'{"type":"quantity","qty":3}', type=T.Answer)  # INV-04: no other answer shape exists


def _floats(obj: Any, path: str = "") -> list[str]:
    if isinstance(obj, float):
        return [path]
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in _floats(v, f"{path}.{k}")]
    if isinstance(obj, list | tuple):
        return [p for i, v in enumerate(obj) for p in _floats(v, f"{path}[{i}]")]
    return []


@pytest.mark.parametrize(
    "name",
    ["Forecast", "Outcome", "RiskVerdict", "OrderIntent", "Fill", "EntryDecision", "ManageDecision", "EntryFacts", "ManageFacts"],
)
def test_hashed_payload_structs_hold_no_floats(name: str) -> None:
    # 2.11: ledger payloads are ints / strs / bools / None / lists / dicts - probabilities as ppm, prices as cents (INV-24)
    cls = module_structs()[name]
    assert _floats(msgspec.to_builtins(sample(cls, none=False))) == []
    assert _floats(msgspec.to_builtins(sample(T.Candidate, none=False))) != []  # which is why RISK_VERDICT carries a SUMMARY


# ======================================================================================================================
# OptionContract, SnapshotKey, Quote, SmileFit
# ======================================================================================================================


def contract(strike_milli: int, right: Right = Right.PUT, expiry: date = date(2026, 10, 16), underlying: str = "SPY") -> OptionContract:
    return OptionContract(underlying=underlying, expiry=expiry, right=right, strike_milli=strike_milli)


def test_occ_symbol_format() -> None:
    assert SAMPLE_CONTRACT.occ == "SPY261016C00600000"  # the example of DESIGN 2.2
    assert contract(450500, Right.PUT).occ == "SPY261016P00450500"  # 450.5 -> 450500
    assert contract(130000, Right.PUT, date(2012, 3, 17)).occ == "SPY120317P00130000"  # Saturday-dated monthly: the LISTED date
    assert contract(1, Right.CALL, date(2030, 1, 4), "ABCDEF").occ == "ABCDEF300104C00000001"
    assert contract(99_999_999, Right.CALL).occ.endswith("C99999999")
    assert re.fullmatch(r"^([A-Z]{1,6})\s*(\d{6})([CP])(\d{8})$", SAMPLE_CONTRACT.occ)  # parse_occ's regex


@pytest.mark.parametrize("root", ["", "spy", "SPY1", "SP Y", "TOOLONGX", "BRK.B"])
def test_option_contract_refuses_bad_roots(root: str) -> None:
    with pytest.raises(ValueError, match="OCC root"):
        contract(600000, underlying=root)


@pytest.mark.parametrize("strike", [0, -1, 10**8, 10**9])
def test_option_contract_refuses_out_of_range_strikes(strike: int) -> None:
    with pytest.raises(ValueError, match="strike_milli"):
        contract(strike)
    with pytest.raises(msgspec.ValidationError):  # the same invariant holds on decode
        msgspec.convert({"underlying": "SPY", "expiry": "2026-10-16", "right": "C", "strike_milli": strike}, OptionContract)


def test_option_contract_is_hashable_and_ordered() -> None:
    a = contract(600000, Right.CALL)
    assert a == SAMPLE_CONTRACT and hash(a) == hash(SAMPLE_CONTRACT) and len({a, SAMPLE_CONTRACT}) == 1
    unordered = [
        contract(610000, Right.PUT),
        contract(600000, Right.PUT),
        contract(600000, Right.CALL, date(2026, 11, 20)),
        contract(600000, Right.CALL),
        contract(600000, Right.CALL, underlying="QQQ"),
    ]
    # field order: underlying, expiry, right ("C" < "P"), strike
    assert [c.occ for c in sorted(unordered)] == [
        "QQQ261016C00600000",
        "SPY261016C00600000",
        "SPY261016P00600000",
        "SPY261016P00610000",
        "SPY261120C00600000",
    ]


def test_snapshot_key_orders_by_session_then_slot() -> None:
    d1, d2 = date(2026, 9, 16), date(2026, 9, 17)
    keys = [SnapshotKey(session=d2, slot=Slot.DEC), SnapshotKey(session=d1, slot=Slot.EOD), SnapshotKey(session=d1, slot=Slot.DEC)]
    assert sorted(keys) == [keys[2], keys[1], keys[0]]
    assert len({SnapshotKey(session=d1, slot=Slot.EOD), SnapshotKey(session=d1, slot=Slot.EOD)}) == 1
    assert msgspec.to_builtins(keys[0]) == {"session": "2026-09-17", "slot": "dec"}


def quote(bid: int, ask: int) -> Quote:
    return Quote(
        contract=SAMPLE_CONTRACT,
        bid=bid,
        ask=ask,
        bid_size=None,
        ask_size=None,
        oi_prev=None,
        iv=None,
        delta=None,
        vega=None,
        quote_ts=None,
    )


@pytest.mark.parametrize(
    ("bid", "ask", "valid", "usable_buy", "usable_sell_close"),
    [
        (100, 105, True, True, True),  # two-sided
        (0, 5, False, True, True),  # zero bid: not two-sided, but a legitimate price to buy at / to sell-to-close at 0 (10.4, 10.5)
        (0, 0, False, False, False),  # no quote at all
        (5, 0, False, False, False),  # no ask
        (105, 105, False, False, False),  # locked
        (110, 105, False, False, False),  # crossed
    ],
)
def test_quote_usability_is_per_side(bid: int, ask: int, valid: bool, usable_buy: bool, usable_sell_close: bool) -> None:
    q = quote(bid, ask)
    assert (q.valid(), q.usable_buy(), q.usable_sell_close()) == (valid, usable_buy, usable_sell_close)
    assert q.mid2 == bid + ask and isinstance(q.mid2, int)


def test_smile_fit_total_variance_and_slope() -> None:
    fit = SmileFit(
        expiry=date(2026, 10, 16),
        last_session=date(2026, 10, 16),
        tau_years=0.08,
        tt_sessions=21.0,
        fwd=60_000,
        a=0.04,
        b=-0.1,
        c=0.5,
        k_lo=-0.2,
        k_hi=0.2,
        n_points=9,
    )
    # w(k) = a + b*k + c*k^2 ; dw/dk = b + 2*c*k     (hand-computed)
    assert fit.w(0.0) == pytest.approx(0.04)
    assert fit.w(0.2) == pytest.approx(0.04 - 0.02 + 0.02)
    assert fit.w(-0.2) == pytest.approx(0.04 + 0.02 + 0.02)
    assert fit.dw_dk(0.0) == pytest.approx(-0.1)
    assert fit.dw_dk(0.2) == pytest.approx(0.1)
    assert fit.dw_dk(-0.1) == pytest.approx(-0.2)


# ======================================================================================================================
# Structure, BandPrices
# ======================================================================================================================

EXP = date(2026, 10, 16)


def structure(kind: StructureKind, legs: list[tuple[Right, int, Side]], expiry: date = EXP, last_session: date | None = None) -> Structure:
    return Structure(
        kind=kind,
        underlying="SPY",
        expiry=expiry,
        last_session=last_session or expiry,
        legs=tuple(Leg(contract=contract(k, right, expiry), side=side) for right, k, side in legs),
    )


SEVEN = {
    StructureKind.LONG_CALL: structure(StructureKind.LONG_CALL, [(Right.CALL, 600000, Side.BUY)]),
    StructureKind.LONG_PUT: structure(StructureKind.LONG_PUT, [(Right.PUT, 580000, Side.BUY)]),
    StructureKind.CALL_DEBIT: structure(StructureKind.CALL_DEBIT, [(Right.CALL, 600000, Side.BUY), (Right.CALL, 610000, Side.SELL)]),
    StructureKind.PUT_DEBIT: structure(StructureKind.PUT_DEBIT, [(Right.PUT, 590000, Side.SELL), (Right.PUT, 600000, Side.BUY)]),
    StructureKind.CALL_CREDIT: structure(StructureKind.CALL_CREDIT, [(Right.CALL, 640000, Side.SELL), (Right.CALL, 650500, Side.BUY)]),
    StructureKind.PUT_CREDIT: structure(StructureKind.PUT_CREDIT, [(Right.PUT, 550000, Side.BUY), (Right.PUT, 560000, Side.SELL)]),
    StructureKind.IRON_CONDOR: structure(
        StructureKind.IRON_CONDOR,
        [(Right.PUT, 130000, Side.BUY), (Right.PUT, 135000, Side.SELL), (Right.CALL, 145000, Side.SELL), (Right.CALL, 150500, Side.BUY)],
        expiry=date(2012, 3, 17),
        last_session=date(2012, 3, 16),
    ),
}


@pytest.mark.parametrize(
    ("kind", "wings", "width", "direction", "n_short"),
    [
        (StructureKind.LONG_CALL, (0, 0), 0, "bullish", 0),
        (StructureKind.LONG_PUT, (0, 0), 0, "bearish", 0),
        (StructureKind.CALL_DEBIT, (0, 1000), 1000, "bullish", 1),  # 600 / 610 calls: $10 = 1000 cents/share
        (StructureKind.PUT_DEBIT, (1000, 0), 1000, "bearish", 1),
        (StructureKind.CALL_CREDIT, (0, 1050), 1050, "bearish", 1),  # 640 / 650.5 calls: $10.50
        (StructureKind.PUT_CREDIT, (1000, 0), 1000, "bullish", 1),
        (StructureKind.IRON_CONDOR, (500, 550), 550, "neutral_range", 2),  # put wing $5, call wing $5.50: width = max wing
    ],
)
def test_structure_geometry_for_all_seven_kinds(
    kind: StructureKind, wings: tuple[int, int], width: int, direction: str, n_short: int
) -> None:
    s = SEVEN[kind]
    assert s.wing_widths == wings
    assert s.width == width
    assert s.direction.value == direction
    assert len(s.short_legs) == n_short and all(leg.side is Side.SELL for leg in s.short_legs)
    assert all(isinstance(w, int) for w in s.wing_widths)


def test_structure_short_legs_keep_canonical_order() -> None:
    condor = SEVEN[StructureKind.IRON_CONDOR]
    assert [leg.contract.occ for leg in condor.short_legs] == ["SPY120317P00135000", "SPY120317C00145000"]  # puts before calls


def test_width_is_never_understated_for_sub_cent_strike_distances() -> None:
    # 22.375 / 25.000 strikes: 2625 milli-dollars = 262.5 cents -> 263 (a width bounds the max loss, so it rounds UP)
    s = structure(StructureKind.PUT_CREDIT, [(Right.PUT, 22375, Side.BUY), (Right.PUT, 25000, Side.SELL)])
    assert s.wing_widths == (263, 0) and s.width == 263


def test_structure_id_goldens() -> None:
    # goldens from `printf '%s' '<canonical json>' | sha256sum`, canonical json = canon.dumps_sorted([kind, ["<occ>:<side>", ...]])
    #   ["put_credit_spread",["SPY261016P00550000:buy","SPY261016P00560000:sell"]]
    assert SEVEN[StructureKind.PUT_CREDIT].structure_id == "303f7820fb8e2482"
    #   ["long_call",["SPY261016C00600000:buy"]]
    assert SEVEN[StructureKind.LONG_CALL].structure_id == "6dfb93f6670f87d8"
    #   ["iron_condor",["SPY120317P00130000:buy","SPY120317P00135000:sell","SPY120317C00145000:sell","SPY120317C00150500:buy"]]
    assert SEVEN[StructureKind.IRON_CONDOR].structure_id == "f956ce7062893559"
    ids = {s.structure_id for s in SEVEN.values()}
    assert len(ids) == 7 and all(re.fullmatch(r"[0-9a-f]{16}", i) for i in ids)


def test_structure_id_is_identity_only() -> None:
    base = SEVEN[StructureKind.PUT_CREDIT]
    assert msgspec.structs.replace(base, last_session=date(2026, 10, 15)).structure_id == base.structure_id  # not identity
    flipped = structure(StructureKind.PUT_CREDIT, [(Right.PUT, 550000, Side.SELL), (Right.PUT, 560000, Side.BUY)])
    other_kind = msgspec.structs.replace(base, kind=StructureKind.PUT_DEBIT)
    other_strike = structure(StructureKind.PUT_CREDIT, [(Right.PUT, 549000, Side.BUY), (Right.PUT, 560000, Side.SELL)])
    other_expiry = structure(
        StructureKind.PUT_CREDIT, [(Right.PUT, 550000, Side.BUY), (Right.PUT, 560000, Side.SELL)], expiry=date(2026, 11, 20)
    )
    assert (
        len({base.structure_id, flipped.structure_id, other_kind.structure_id, other_strike.structure_id, other_expiry.structure_id}) == 5
    )


def test_band_prices_get() -> None:
    net = BandPrices(orats=-118, worst=-110, mid=-125)  # a credit: negative = we receive
    assert [net.get(b) for b in Band] == [-118, -110, -125]
    assert {b.value: net.get(b) for b in Band} == msgspec.to_builtins(net)
    with pytest.raises(ValueError, match="band"):
        net.get("headline")  # type: ignore[arg-type]


# ======================================================================================================================
# ChainSnapshot
# ======================================================================================================================

SESSION = date(2012, 2, 10)
WEEKLY = date(2012, 3, 9)  # a Friday: last_session == expiry
MONTHLY = date(2012, 3, 17)  # SATURDAY-dated monthly: last_session is Friday 2012-03-16
QUOTE_TS = datetime(2012, 2, 10, 20, 35, 1, tzinfo=UTC)


def chain_table(rows: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    def row(expiry: date, last: date, right: str, strike: int, bid: int, ask: int, fwd: int, **extra: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "occ": f"SPY{expiry:%y%m%d}{right}{strike:08d}",
            "expiry": pd.Timestamp(expiry),
            "last_session": pd.Timestamp(last),
            "right": right,
            "strike_milli": strike,
            "dte": (last - SESSION).days,
            "bid": bid,
            "ask": ask,
            "bid_size": 12,
            "ask_size": 30,
            "oi_prev": 1500,
            "iv": 0.2,
            "delta": -0.25 if right == "P" else 0.25,
            "vega": 0.11,
            "fwd": fwd,
            "iv_vendor": 0.21,
            "quote_ts": pd.NaT,
        }
        base.update(extra)
        return base

    if rows is None:
        friday = date(2012, 3, 16)
        rows = [  # deliberately NOT sorted: the accessors must not rely on the caller's order
            row(MONTHLY, friday, "P", 135000, 210, 214, 13490),
            row(MONTHLY, friday, "C", 145000, 95, 99, 13490, quote_ts=pd.Timestamp(QUOTE_TS)),
            row(
                WEEKLY,
                WEEKLY,
                "P",
                130000,
                0,
                3,
                13480,
                bid_size=pd.NA,
                oi_prev=pd.NA,
                iv=float("nan"),
                delta=float("nan"),
                vega=float("nan"),
            ),
            row(MONTHLY, friday, "P", 130000, 101, 104, 13490),
            row(WEEKLY, WEEKLY, "P", 135000, 150, 153, 13480),
            row(MONTHLY, friday, "C", 150500, 20, 23, 13490),
        ]
    df = pd.DataFrame(rows, columns=CHAIN_COLUMNS)
    df = df.astype(
        {
            "occ": "object",
            "right": "object",
            "strike_milli": "int64",
            "dte": "int64",
            "bid": "int64",
            "ask": "int64",
            "bid_size": "Int64",
            "ask_size": "Int64",
            "oi_prev": "Int64",
            "iv": "float64",
            "delta": "float64",
            "vega": "float64",
            "fwd": "int64",
            "iv_vendor": "float64",
        }
    )
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["last_session"] = pd.to_datetime(df["last_session"])
    df["quote_ts"] = pd.to_datetime(df["quote_ts"], utc=True)
    return df


def snapshot(table: pd.DataFrame | None = None) -> ChainSnapshot:
    close = datetime(2012, 2, 10, 21, 0, tzinfo=UTC)
    return ChainSnapshot(
        underlying="SPY",
        key=SnapshotKey(session=SESSION, slot=Slot.EOD),
        ts=close,
        knowable_at=close,
        spot=13470,
        spot_measure="parity",
        div_unmodelled=False,
        rate=0.0009,
        table=chain_table() if table is None else table,
        fidelity=Fidelity.EOD_QUOTES,
        source="mirror",
        content_hash="0" * 64,
    )


def test_chain_table_fixture_has_the_contract_dtypes() -> None:
    dtypes = chain_table().dtypes.astype(str).to_dict()
    assert dtypes["expiry"] == "datetime64[ns]" and dtypes["last_session"] == "datetime64[ns]"
    assert dtypes["quote_ts"] == "datetime64[ns, UTC]" and dtypes["oi_prev"] == "Int64" and dtypes["fwd"] == "int64"


def test_chain_quote_materialises_python_values() -> None:
    snap = snapshot()
    c = contract(145000, Right.CALL, MONTHLY)
    q = snap.quote(c)
    assert q == Quote(contract=c, bid=95, ask=99, bid_size=12, ask_size=30, oi_prev=1500, iv=0.2, delta=0.25, vega=0.11, quote_ts=QUOTE_TS)
    assert q is not None
    assert [type(v) for v in (q.bid, q.ask, q.bid_size, q.oi_prev)] == [int] * 4  # never numpy scalars: they reach hashed payloads
    assert type(q.iv) is float and type(q.quote_ts) is datetime
    assert q.quote_ts is not None and q.quote_ts.utcoffset() == timedelta(0)
    assert msgspec.json.decode(msgspec.json.encode(q), type=Quote) == q


def test_chain_quote_maps_missing_values_to_none() -> None:
    q = snapshot().quote(contract(130000, Right.PUT, WEEKLY))
    assert q is not None
    assert (q.bid, q.ask, q.bid_size, q.ask_size, q.oi_prev) == (0, 3, None, 30, None)
    assert (q.iv, q.delta, q.vega, q.quote_ts) == (None, None, None, None)  # EOD data has no feed timestamp
    assert not q.valid() and q.usable_sell_close()


def test_chain_quote_returns_none_for_unlisted_contracts() -> None:
    snap = snapshot()
    assert snap.quote(contract(131000, Right.PUT, MONTHLY)) is None  # strike not listed
    assert snap.quote(contract(130000, Right.CALL, MONTHLY)) is None  # right not listed at that strike
    assert snap.quote(contract(130000, Right.PUT, date(2012, 3, 16))) is None  # identity is the LISTED date, not last_session
    assert snap.quote(contract(130000, Right.PUT, MONTHLY, underlying="QQQ")) is None  # another underlying's chain


def test_chain_quote_converts_other_zones_to_utc_and_refuses_naive_timestamps() -> None:
    table = chain_table()
    eastern = table.copy()
    eastern["quote_ts"] = eastern["quote_ts"].dt.tz_convert("America/New_York")
    q = snapshot(eastern).quote(contract(145000, Right.CALL, MONTHLY))
    assert q is not None and q.quote_ts == QUOTE_TS and q.quote_ts.utcoffset() == timedelta(0)
    naive = table.copy()
    naive["quote_ts"] = naive["quote_ts"].dt.tz_localize(None)
    with pytest.raises(DataError, match="tz-naive"):
        snapshot(naive).quote(contract(145000, Right.CALL, MONTHLY))


def test_chain_quote_refuses_duplicate_rows() -> None:
    table = chain_table()
    doubled = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataError, match="2 chain rows"):
        snapshot(doubled).quote(contract(135000, Right.PUT, MONTHLY))


def test_chain_expiries_and_last_session() -> None:
    snap = snapshot()
    assert snap.expiries() == (WEEKLY, MONTHLY)  # ordered by (last_session, expiry), whatever the row order
    assert all(type(e) is date for e in snap.expiries())
    assert snap.last_session(MONTHLY) == date(2012, 3, 16)  # Saturday-dated monthly -> the Friday (INV-11)
    assert snap.last_session(WEEKLY) == WEEKLY
    assert type(snap.last_session(MONTHLY)) is date
    with pytest.raises(DataUnavailable):
        snap.last_session(date(2012, 3, 16))  # the last trading day is not a listed expiry
    assert snapshot(chain_table().iloc[0:0]).expiries() == ()


def test_chain_side_is_one_expiry_one_right_strike_sorted() -> None:
    snap = snapshot()
    puts = snap.side(MONTHLY, Right.PUT)
    assert list(puts.columns) == CHAIN_COLUMNS
    assert puts["strike_milli"].tolist() == [130000, 135000] and puts["occ"].tolist() == ["SPY120317P00130000", "SPY120317P00135000"]
    assert list(puts.index) == [0, 1]
    assert snap.side(MONTHLY, Right.CALL)["strike_milli"].tolist() == [145000, 150500]
    assert snap.side(WEEKLY, Right.CALL).empty and snap.side(date(2012, 4, 21), Right.PUT).empty
    puts.loc[0, "bid"] = 1  # a copy: mutating it cannot corrupt the snapshot
    assert snap.side(MONTHLY, Right.PUT)["bid"].tolist() == [101, 210]


def test_chain_forward() -> None:
    snap = snapshot()
    assert snap.forward(MONTHLY) == 13490 and snap.forward(WEEKLY) == 13480
    assert type(snap.forward(MONTHLY)) is int
    with pytest.raises(DataUnavailable, match="not listed"):
        snap.forward(date(2012, 4, 21))
    inconsistent = chain_table()
    inconsistent.loc[0, "fwd"] = 13491  # one row of the monthly disagrees about ITS expiry's forward
    with pytest.raises(DataError, match="distinct `fwd`"):
        snapshot(inconsistent).forward(MONTHLY)


def test_chain_snapshot_is_frozen_with_identity_equality() -> None:
    snap = snapshot()
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.spot = 1  # type: ignore[misc]
    assert snap != dataclasses.replace(snap) and snap == snap  # eq=False: a DataFrame has no scalar equality


# ======================================================================================================================
# 2.11 ledger payload fields
# ======================================================================================================================

# descriptive cells of the 2.11 table that types.py freezes as literal keys: key -> the phrase it stands for
PAYLOAD_PHRASES = {
    "equity": "equity per band",
    "cash": "cash per band",
    "positions": "per-position {liq_value, mid_value, stale}",
    "positions_digest": "positions digest",
    "entry_qset_hash": "qset hashes (4)",
    "entry_text_qset_hash": "qset hashes (4)",
    "manage_qset_hash": "qset hashes (4)",
    "manage_text_qset_hash": "qset hashes (4)",
    "candidate": "candidate summary",
    "realised_pnl": "realised P&L per band",
}
STRUCT_BACKED = {
    T.LedgerKind.FORECAST: T.Forecast,
    T.LedgerKind.OUTCOME: T.Outcome,
    T.LedgerKind.RISK_VERDICT: T.RiskVerdict,
    T.LedgerKind.ORDER_INTENT: T.OrderIntent,
    T.LedgerKind.FILL: T.Fill,
}


@pytest.fixture(scope="module")
def payload_rows(design: str) -> dict[str, str]:
    rows = {}
    for line in section(design, "2.11").splitlines():
        cells = [c.strip() for c in line.strip("|").split("|", 1)]
        if line.startswith("| ") and len(cells) == 2 and re.fullmatch(r"[A-Z_]+", cells[0]):
            rows[cells[0]] = cells[1]
    return rows


def test_ledger_payload_table_covers_every_kind(payload_rows: dict[str, str]) -> None:
    assert set(payload_rows) == {k.name for k in T.LedgerKind} == {k.name for k in T.LEDGER_PAYLOAD_FIELDS}
    assert [k.name for k in T.LEDGER_PAYLOAD_FIELDS] == list(payload_rows)  # same order as the 2.11 table
    assert set(T.LEDGER_PAYLOAD_OPTIONAL_FIELDS) <= set(T.LEDGER_PAYLOAD_FIELDS)


def test_ledger_payload_fields_are_named_in_design(payload_rows: dict[str, str]) -> None:
    for kind, fields in T.LEDGER_PAYLOAD_FIELDS.items():
        row = payload_rows[kind.name]
        optional = T.LEDGER_PAYLOAD_OPTIONAL_FIELDS.get(kind, ())
        assert len(set(fields) | set(optional)) == len(fields) + len(optional), kind  # no duplicates
        struct = STRUCT_BACKED.get(kind)
        if struct is not None:
            assert f"{struct.__name__} builtins" in row, kind
            assert fields[: len(struct.__struct_fields__)] == struct.__struct_fields__, kind
        for name in (*fields, *optional):
            if struct is not None and name in struct.__struct_fields__:
                continue
            phrase = PAYLOAD_PHRASES.get(name, name)
            assert re.search(rf"(?<![\w]){re.escape(phrase)}(?![\w])", row), f"{kind.name}: `{name}` is not in the 2.11 row"


def test_ledger_payload_literal_rows_are_complete(payload_rows: dict[str, str]) -> None:
    # rows that are a plain comma list of identifiers must be reproduced exactly, in order
    for kind in (T.LedgerKind.ORDER_STATUS, T.LedgerKind.REARM, T.LedgerKind.KILL, T.LedgerKind.BROKER_FILL, T.LedgerKind.RECONCILE):
        text = re.sub(r"\([^)]*\)|\{[^}]*\}|\[\]", "", payload_rows[kind.name])
        assert tuple(re.findall(r"[a-z_]+", text)) == T.LEDGER_PAYLOAD_FIELDS[kind], kind
    run_start = [w for w in re.findall(r"[a-z_]+", re.sub(r"\([^)]*\)", "", payload_rows["RUN_START"])) if w not in ("qset", "hashes")]
    expected = [f for f in T.LEDGER_PAYLOAD_FIELDS[T.LedgerKind.RUN_START] if not f.endswith("_qset_hash")]
    assert run_start == expected
    # the four question-set hashes carry RunMeta's names, and every RUN_START key except initial_cash is a RunMeta field
    run_meta = set(T.RunMeta.__struct_fields__)
    assert set(T.LEDGER_PAYLOAD_FIELDS[T.LedgerKind.RUN_START]) - run_meta == {"initial_cash"}
    # run_id / trial_id / wall-clock material is NEVER hashed (INV-24)
    assert not {"run_id", "trial_id", "experiment", "family", "cache_manifest_hash"} & set(T.LEDGER_PAYLOAD_FIELDS[T.LedgerKind.RUN_START])


def test_ledger_payload_nested_shapes(payload_rows: dict[str, str]) -> None:
    decision = payload_rows["DECISION"]
    requests = re.search(r"requests: \[\{(.*?)\}\]", decision)
    assert requests is not None
    names = tuple(re.sub(r"\{[^}]*\}|\([^)]*\)", "", part).split(":")[0].strip() for part in re.split(r",\s*(?![^{(]*[})])", requests[1]))
    assert names == T.DECISION_REQUEST_FIELDS
    summary = re.search(r"candidate summary \{(.*?)\}", payload_rows["RISK_VERDICT"])
    assert summary is not None
    flat = re.sub(r"\[[^\]]*\]| per band", "", summary[1])  # `legs[occ, side]` -> `legs`, `net per band` -> `net`
    assert tuple(part.strip() for part in flat.split(",")) == T.CANDIDATE_SUMMARY_FIELDS
    assert T.MARK_POSITION_FIELDS == ("liq_value", "mid_value", "stale")
    # the OPEN intent carries the EntryContext: the ledger is the only source of Position.entry (10.8)
    assert "entry_ctx" in T.LEDGER_PAYLOAD_FIELDS[T.LedgerKind.ORDER_INTENT]
    assert set(T.EntryContext.__struct_fields__) == {
        "entry_thesis",
        "entry_codes",
        "entry_spot",
        "entry_iv30_bp",
        "entry_em_hold_tenths",
        "open_mid_at_decision",
    }


# ======================================================================================================================
# 2.9 errors and exit codes
# ======================================================================================================================


def parse_error_tree(design: str) -> dict[str, str | None]:
    tree = fenced(section(design, "2.9"), "")[0]
    parents: dict[str, str | None] = {}
    stack: dict[int, str] = {}
    for line in tree.splitlines():
        if re.match(r"^\w+$", line.strip()) and "+-" not in line:
            parents[line.strip()] = None
            stack = {0: line.strip()}
            continue
        node = re.search(r"\+- (\w+)", line)
        if node is None:
            continue
        depth = 1 if line.index("+-") < 4 else 2
        parents[node[1]] = stack[depth - 1]
        stack[depth] = node[1]
    return parents


def test_error_hierarchy_equals_design(design: str) -> None:
    parents = parse_error_tree(design)
    assert len(parents) == 28 and parents["JevbotError"] is None
    actual = {
        name: obj
        for name, obj in vars(errors).items()
        if isinstance(obj, type) and issubclass(obj, BaseException) and obj.__module__ == errors.__name__
    }
    assert set(actual) == set(parents)
    for name, parent in parents.items():
        expected_base = Exception if parent is None else actual[parent]
        assert actual[name].__bases__ == (expected_base,), name
    assert set(errors.__all__) == set(parents) | {"ExitCode", "exit_code_for"}


def test_fail_closed_classification() -> None:
    # CacheMissError and ModelMismatchError are deliberately NOT DeciderErrors: they are always RAISED (D7, INV-06)
    assert not issubclass(errors.CacheMissError, errors.DeciderError)
    assert not issubclass(errors.ModelMismatchError, errors.DeciderError)
    assert issubclass(errors.SpendLimitError, errors.DeciderError)
    assert issubclass(errors.PaperGuardError, errors.BrokerError)
    assert issubclass(errors.PitViolation, errors.DataError)
    with pytest.raises(errors.JevbotError):
        raise errors.StateLeak("ticker in masked state")


def test_broker_rejected_carries_its_diagnostics() -> None:
    exc = errors.BrokerRejected("insufficient options buying power", status=403, reject_code=40310000, tag="buying_power")
    assert (exc.status, exc.reject_code, exc.message, exc.tag) == (403, 40310000, "insufficient options buying power", "buying_power")
    assert str(exc) == "insufficient options buying power status=403 reject_code=40310000 tag=buying_power"
    clone = pickle.loads(pickle.dumps(exc))  # exceptions cross process pools (baseline seeds)
    assert type(clone) is errors.BrokerRejected
    assert (clone.status, clone.reject_code, clone.message, clone.tag, clone.args) == (403, 40310000, exc.message, "buying_power", exc.args)
    bare = errors.BrokerRejected()
    assert (bare.status, bare.reject_code, bare.message, bare.tag) == (None, None, "", None) and str(bare) == "order rejected"


def test_exit_codes_equal_design(design: str) -> None:
    text = section(design, "2.9")
    sentence = text[text.index("Process exit codes:") :]
    listed = {int(n): what for n, what in re.findall(r"(?:^|[\s,])(\d) ([a-z][a-z /]+?)(?=[,(.]| \()", sentence)}
    assert listed == {
        0: "ok",
        1: "generic error",
        2: "config/usage",
        3: "safety guard refused",
        4: "paper guard failure",
        5: "replay cache miss",
        6: "model mismatch",
        7: "spend limit",
        8: "data / manifest error",
        9: "ledger corrupt",
    }
    assert {c.value for c in ExitCode} == set(listed)
    assert {c.name: c.value for c in ExitCode} == {
        "OK": 0,
        "ERROR": 1,
        "CONFIG": 2,
        "SAFETY_REFUSED": 3,
        "PAPER_GUARD": 4,
        "CACHE_MISS": 5,
        "MODEL_MISMATCH": 6,
        "SPEND_LIMIT": 7,
        "DATA": 8,
        "LEDGER_CORRUPT": 9,
    }


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (errors.ConfigError("x"), 2),
        (errors.PaperGuardError("x"), 4),
        (errors.CacheMissError("x"), 5),
        (errors.ModelMismatchError("x"), 6),
        (errors.SpendLimitError("x"), 7),
        (errors.DataError("x"), 8),
        (errors.DataUnavailable("x"), 8),
        (errors.PitViolation("x"), 8),
        (errors.ManifestMismatch("x"), 8),
        (errors.LedgerCorrupt("x"), 9),
        (errors.DeciderTransportError("x"), 1),
        (errors.DeciderConfigError("x"), 1),  # a decider auth error is NOT a config/usage exit
        (errors.BrokerRejected("x"), 1),
        (errors.InvariantError("x"), 1),
        (errors.TierViolation("x"), 1),
        (ValueError("x"), 1),
        (KeyboardInterrupt(), 1),
    ],
)
def test_exit_code_for(exc: BaseException, code: int) -> None:
    assert exit_code_for(exc) == code and isinstance(exit_code_for(exc), ExitCode)


# ======================================================================================================================
# import hygiene and project setup (pyproject.toml, uv.lock, .gitignore, .env.example, __init__.py)
# ======================================================================================================================


def _modules_after_import(module: str) -> set[str]:
    out = subprocess.run(
        [sys.executable, "-c", f"import {module}, sys; print('\\n'.join(sorted(sys.modules)))"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    return set(out.stdout.split())


def test_types_imports_only_errors_and_no_network_library() -> None:
    loaded = _modules_after_import("jevbot.types")
    assert {m for m in loaded if m.split(".")[0] == "jevbot"} == {"jevbot", "jevbot.errors", "jevbot.types"}
    assert not {m.split(".")[0] for m in loaded} & {"typesafe_sdk", "alpaca", "httpx2", "httpcore2", "requests", "urllib3", "websockets"}


def test_errors_imports_nothing_from_the_package() -> None:
    loaded = _modules_after_import("jevbot.errors")
    assert {m for m in loaded if m.split(".")[0] == "jevbot"} == {"jevbot", "jevbot.errors"}
    assert not {m.split(".")[0] for m in loaded} & {"pandas", "numpy", "msgspec"}


@pytest.fixture(scope="module")
def pyproject() -> dict[str, Any]:
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


def test_version_is_single_sourced(pyproject: dict[str, Any]) -> None:
    assert jevbot.__version__ == pyproject["project"]["version"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", jevbot.__version__)


def test_pyproject_follows_design_1_1(pyproject: dict[str, Any]) -> None:
    project = pyproject["project"]
    deps = project["dependencies"]
    assert "typesafe-sdk==0.6.0" in deps and "alpaca-py==0.44.0" in deps  # the two EXACT pins (D26)
    names = sorted(re.split(r"[<>=!~ ]", d, maxsplit=1)[0] for d in deps)
    assert names == sorted(
        ["numpy", "pandas", "pyarrow", "scipy", "msgspec", "exchange-calendars", "typer", "matplotlib", "typesafe-sdk", "alpaca-py"]
    )  # the runtime list is closed: "Nothing else"
    assert project["scripts"] == {"jevbot": "jevbot.cli.main:app"}
    assert project["requires-python"].startswith(">=3.12")
    dev = sorted(re.split(r"[<>=!~ ]", d, maxsplit=1)[0] for d in pyproject["dependency-groups"]["dev"])
    assert dev == ["mypy", "pytest", "pytest-cov", "ruff"]
    assert pyproject["tool"]["mypy"]["strict"] is True
    ignored = {m for o in pyproject["tool"]["mypy"]["overrides"] if o.get("ignore_missing_imports") for m in o["module"]}
    assert {"pandas", "pyarrow", "exchange_calendars", "alpaca"} <= ignored
    assert "." in pyproject["tool"]["pytest"]["ini_options"]["pythonpath"]  # `tests.fixtures.<double>` imports
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["src/jevbot"]  # src layout


def test_pypi_is_the_only_index(pyproject: dict[str, Any]) -> None:
    uv_cfg = pyproject["tool"].get("uv", {})
    assert not {"index", "index-url", "extra-index-url", "find-links", "sources"} & set(uv_cfg)
    assert "pip" not in pyproject["tool"]
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    packages = {p["name"]: p for p in lock["package"]}
    sources = {name: p["source"] for name, p in packages.items()}
    assert sources.pop("jevbot") == {"editable": "."}
    assert set(map(str, sources.values())) == {str({"registry": "https://pypi.org/simple"})}
    assert packages["typesafe-sdk"]["version"] == "0.6.0" and packages["alpaca-py"]["version"] == "0.44.0"
    assert "websockets" in packages and "httpx2" in packages  # everything pinned in uv.lock, including websockets
    assert not {"typesafe-client", "cooksafe"} & set(packages)
    assert packages["pandas"]["version"].startswith("2.")


def test_env_example_lists_names_only_and_env_is_git_ignored() -> None:
    lines = [ln for ln in (REPO / ".env.example").read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    assert all(re.fullmatch(r"[A-Z][A-Z0-9_]*=", ln) for ln in lines), "names only: no value may ever be committed"
    assert [ln[:-1] for ln in lines] == [
        "JEVBOT_DATA",
        "TYPESAFE_API_KEY",
        "ALPACA_PAPER_KEY",
        "ALPACA_PAPER_SECRET",
        "JEVBOT_ALERT_WEBHOOK",
    ]
    everything = (REPO / ".env.example").read_text(encoding="utf-8")
    # forbidden names are assembled from fragments so that this file itself stays clean for the repo-wide guard greps
    forbidden_names = ("TYPESAFE_" + "LOG_LEVEL", "AP" + "CA_", "ALPACA_" + "API_KEY", "ALPACA_" + "SECRET_KEY")
    for forbidden in forbidden_names:
        assert forbidden not in everything
    ignore = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignore and "!.env.example" in ignore and ".venv/" in ignore
    assert "uv.lock" not in ignore  # the lock file is committed
