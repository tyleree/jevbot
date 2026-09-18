"""`state.py` (DESIGN.md 5, 5.6-5.9): the four states, their variants, their facts and their provenance.

The goldens in `tests/fixtures/golden/` are the reviewed bytes of each state; every test that is not a golden pins a
property the spec names: key order, path set, purity, text isolation, path independence of the manage state, the
`bucket_only` render and the leakage-diagnostic identity block.
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final

import msgspec
import pandas as pd
import pytest

from jevbot import buckets, structmath, vocab
from jevbot.canon import dumps_ordered, ensure_state_safe, sha256_hex
from jevbot.config import Config, StateConfig, load_mask_terms
from jevbot.errors import StateLeak
from jevbot.state import (
    DIRECTIONAL_EXPOSURE,
    EARNINGS_TEXT,
    MARKET_AS_OF_TEXT,
    VOL_EXPOSURE,
    StateBuilder,
)
from jevbot.types import (
    BandPrices,
    EntryContext,
    EntryFacts,
    Fidelity,
    Leg,
    ManageFacts,
    MaskTerms,
    NewsItem,
    OptionContract,
    Position,
    Right,
    Side,
    Slot,
    SnapshotKey,
    Structure,
    StructureKind,
    Variant,
)
from tests.fixtures.chain_factory import DEFAULT_SESSION, make_chain
from tests.fixtures.fake_view import FakeView, fomc_event, make_view, news_item
from tests.fixtures.news import cases

REPO: Final = Path(__file__).resolve().parents[2]
MASK_TERMS_FILE: Final = REPO / "config" / "mask_terms.toml"
HOLD: Final = 20
OPEN_MID: Final = -120  # signed cents/share at the DECISION snapshot: a 120c credit
ENTRY_SPOT: Final = 44_500
ENTRY_IV30_BP: Final = 1_800
ENTRY_EM_TENTHS: Final = 42
ENTRY_THESIS: Final = (
    "opened with trend up, implied volatility iv_rich versus realized, iv rank upper_middle, no tracked event inside the holding window"
)


# ======================================================================================================================
# Fixtures
# ======================================================================================================================


@pytest.fixture(scope="module")
def terms() -> MaskTerms:
    return load_mask_terms(MASK_TERMS_FILE)


@pytest.fixture
def cfg() -> Config:
    return Config()


def _news_items(as_of: datetime) -> list[NewsItem]:
    return cases.as_items(cases.load(cases.BENIGN), as_of=as_of)


@pytest.fixture
def world() -> FakeView:
    """The shared world of the goldens: SPY on 2024-05-17 with the benign corpus and one FOMC inside the window."""
    session = DEFAULT_SESSION
    as_of = make_chain("SPY", session=session).ts
    return make_view(("SPY",), news=_news_items(as_of), news_covered=True)


@pytest.fixture
def builder(cfg: Config, terms: MaskTerms) -> StateBuilder:
    return StateBuilder(cfg, terms, news_resolved=True)


def _structure(view: FakeView, kind: StructureKind = StructureKind.PUT_CREDIT) -> Structure:
    """A deterministic structure on the factory chain: the 0.25-delta short leg and a long leg two strikes further OTM."""
    chain = view.chain("SPY")
    expiry = next(e for e in chain.expiries() if (chain.last_session(e) - view.session).days >= 28)
    right = Right.PUT if kind in (StructureKind.PUT_CREDIT, StructureKind.PUT_DEBIT, StructureKind.LONG_PUT) else Right.CALL
    side = chain.side(expiry, right)
    short = side.iloc[(side["delta"].abs() - 0.25).abs().argmin()]

    def contract(strike_milli: int) -> OptionContract:
        return OptionContract(underlying="SPY", expiry=expiry, right=right, strike_milli=int(strike_milli))

    step = 1_000  # the factory's strike step in milli-dollars
    last_session = chain.last_session(expiry)
    if kind is StructureKind.LONG_CALL:
        legs = (Leg(contract=contract(short["strike_milli"]), side=Side.BUY),)
    elif kind is StructureKind.PUT_CREDIT:
        legs = (
            Leg(contract=contract(int(short["strike_milli"]) - 2 * step), side=Side.BUY),
            Leg(contract=contract(short["strike_milli"]), side=Side.SELL),
        )
    elif kind is StructureKind.IRON_CONDOR:
        put_side = chain.side(expiry, Right.PUT)
        call_side = chain.side(expiry, Right.CALL)
        short_put = int(put_side.iloc[(put_side["delta"].abs() - 0.16).abs().argmin()]["strike_milli"])
        short_call = int(call_side.iloc[(call_side["delta"].abs() - 0.16).abs().argmin()]["strike_milli"])
        legs = (
            Leg(
                contract=OptionContract(underlying="SPY", expiry=expiry, right=Right.PUT, strike_milli=short_put - 2 * step), side=Side.BUY
            ),
            Leg(contract=OptionContract(underlying="SPY", expiry=expiry, right=Right.PUT, strike_milli=short_put), side=Side.SELL),
            Leg(contract=OptionContract(underlying="SPY", expiry=expiry, right=Right.CALL, strike_milli=short_call), side=Side.SELL),
            Leg(
                contract=OptionContract(underlying="SPY", expiry=expiry, right=Right.CALL, strike_milli=short_call + 2 * step),
                side=Side.BUY,
            ),
        )
    else:  # pragma: no cover - the helper serves the kinds the tests use
        raise AssertionError(kind)
    return Structure(kind=kind, underlying="SPY", expiry=expiry, last_session=last_session, legs=legs)


def _entry_context(open_mid: int = OPEN_MID) -> EntryContext:
    return EntryContext(
        entry_thesis=ENTRY_THESIS,
        entry_codes={"trend": "up", "iv_vs_realized": "iv_rich", "iv_rank": "upper_middle"},
        entry_spot=ENTRY_SPOT,
        entry_iv30_bp=ENTRY_IV30_BP,
        entry_em_hold_tenths=ENTRY_EM_TENTHS,
        open_mid_at_decision=open_mid,
    )


def _position(
    view: FakeView,
    *,
    kind: StructureKind = StructureKind.PUT_CREDIT,
    sessions_held: int = 6,
    mid_value: int = 95,
    liq_value: int = 100,
    open_net: BandPrices | None = None,
    entry: EntryContext | None = None,
    qty: int = 2,
) -> Position:
    structure = _structure(view, kind)
    return Position(
        position_id="pos-1",
        structure=structure,
        qty=qty,
        open_key=SnapshotKey(session=view.calendar.prev_session(view.session, sessions_held), slot=Slot.EOD),
        open_decision_id="dec-1",
        open_net=open_net or BandPrices(orats=-118, worst=-115, mid=OPEN_MID),
        max_loss=38_000,
        max_profit=24_000,
        bp_reserved=38_000,
        entry=entry or _entry_context(),
        liq_value=liq_value,
        mid_value=mid_value,
    )


def _golden_text(state: dict[str, Any]) -> str:
    return json.dumps(state, indent=2, ensure_ascii=False) + "\n"


def _check_golden(goldens: Any, name: str, state: dict[str, Any]) -> None:
    """Compare the reviewed bytes AND prove that the file's KEY ORDER is the one that reaches the wire (V1)."""
    goldens.check_text(name, _golden_text(state))
    path = goldens.path(name)
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))  # json.loads keeps the file's key order
        assert dumps_ordered(loaded) == dumps_ordered(state)


# ======================================================================================================================
# Golden states (15.2: all four kinds and all variants)
# ======================================================================================================================


def test_entry_state_golden(builder: StateBuilder, world: FakeView, goldens: Any) -> None:
    built = builder.entry(world, "SPY")
    assert built is not None
    _check_golden(goldens, "entry_state.json", built.state)
    _check_golden(goldens, "entry_state_key_perm.json", builder.variant(built.state, Variant.KEY_PERM))
    _check_golden(goldens, "entry_state_bucket_only.json", builder.variant(built.state, Variant.BUCKET_ONLY))
    assert built.state_hash == sha256_hex(dumps_ordered(built.state))


def test_entry_text_state_golden(builder: StateBuilder, world: FakeView, goldens: Any) -> None:
    base = builder.entry(world, "SPY")
    assert base is not None
    built = builder.entry_text(world, "SPY", base)
    assert built is not None
    _check_golden(goldens, "entry_text_state.json", built.state)
    _check_golden(goldens, "entry_text_state_key_perm.json", builder.variant(built.state, Variant.KEY_PERM))
    _check_golden(goldens, "entry_text_state_bucket_only.json", builder.variant(built.state, Variant.BUCKET_ONLY))
    assert built.state_hash == sha256_hex(dumps_ordered(built.state))


def test_manage_state_golden(builder: StateBuilder, world: FakeView, goldens: Any) -> None:
    built = builder.manage(world, _position(world))
    _check_golden(goldens, "manage_state.json", built.state)
    _check_golden(goldens, "manage_state_key_perm.json", builder.variant(built.state, Variant.KEY_PERM))
    _check_golden(goldens, "manage_state_bucket_only.json", builder.variant(built.state, Variant.BUCKET_ONLY))
    assert built.state_hash == sha256_hex(dumps_ordered(built.state))


def test_manage_text_state_golden(builder: StateBuilder, world: FakeView, goldens: Any) -> None:
    position = _position(world)
    base = builder.manage(world, position)
    built = builder.manage_text(world, position, base)
    assert built is not None
    _check_golden(goldens, "manage_text_state.json", built.state)
    _check_golden(goldens, "manage_text_state_key_perm.json", builder.variant(built.state, Variant.KEY_PERM))
    _check_golden(goldens, "manage_text_state_bucket_only.json", builder.variant(built.state, Variant.BUCKET_ONLY))
    assert built.state_hash == sha256_hex(dumps_ordered(built.state))


def _three_slot_worlds() -> tuple[FakeView, FakeView]:
    """A three-slot recorder archive and the same data collapsed to one row per session (5.4)."""
    session = DEFAULT_SESSION
    three = make_view(
        ("SPY",),
        slot=Slot.DEC,
        slots=(Slot.DEC, Slot.EXEC, Slot.EOD),
        fidelity=Fidelity.RECORDED_INDICATIVE,
    )
    daily = three.tables.daily["SPY"]
    today = pd.Timestamp(session)
    collapsed_daily = pd.concat(
        [
            daily[(daily["session"] < today) & (daily["slot"] == Slot.EOD.value)],
            daily[(daily["session"] == today) & (daily["slot"] == Slot.DEC.value)],
        ]
    ).reset_index(drop=True)
    collapsed = FakeView(
        key=three.key,
        as_of=three.as_of,
        calendar=three.calendar,
        fidelity=three.fidelity,
        chains=[three.chain("SPY")],
        daily={"SPY": collapsed_daily},
        bars={"SPY": three.tables.bars["SPY"]},
        vol_indices=three.tables.vol_indices,
        rates=three.tables.rates,
        events=three.tables.events,
    )
    return three, collapsed


def test_entry_state_3slot_golden_equals_the_collapsed_fixture(builder: StateBuilder, goldens: Any) -> None:
    """5.4: the same 30 sessions with three slots and collapsed to the designated slot give the same state."""
    three, collapsed = _three_slot_worlds()
    # the two archives really take different paths through the 5.4 read rule
    assert set(three.daily("SPY", 30)["slot"][:-1]) == {Slot.DEC.value}
    assert set(collapsed.daily("SPY", 30)["slot"][:-1]) == {Slot.EOD.value}  # designated slot missing -> `eod` fallback
    assert three.daily("SPY", 30)["slot"].iloc[-1] == collapsed.daily("SPY", 30)["slot"].iloc[-1] == Slot.DEC.value
    built = builder.entry(three, "SPY")
    same = builder.entry(collapsed, "SPY")
    assert built is not None and same is not None
    assert built.state == same.state and built.state_hash == same.state_hash
    assert built.facts == same.facts
    _check_golden(goldens, "entry_state_3slot.json", built.state)
    assert built.state_hash == sha256_hex(dumps_ordered(built.state))


# ======================================================================================================================
# Shape: key order and paths
# ======================================================================================================================


def test_entry_key_order_is_exactly_5_6(builder: StateBuilder, world: FakeView) -> None:
    state = builder.entry(world, "SPY")
    assert state is not None
    assert list(state.state) == ["schema", "context", "market", "underlying", "vol_surface", "events"]
    assert list(state.state["context"]) == ["underlying_alias", "underlying_kind", "decision_session", "holding_window_sessions"]
    assert list(state.state["market"]) == [
        "as_of",
        "vol_index_pctile_1y",
        "vol_index_change_1w",
        "vol_term_structure",
        "near_term_stress",
        "vol_of_vol",
        "tail_skew_index",
    ]
    assert list(state.state["underlying"]) == ["trend", "momentum", "range", "levels"]
    assert list(state.state["vol_surface"]) == [
        "iv_rank_1y",
        "iv_vs_realized",
        "iv_change_1w",
        "term_structure",
        "skew",
        "expected_move_1_session",
        "expected_move_5_sessions",
        "expected_move_holding_window",
    ]
    assert list(state.state["events"]) == ["coverage", "inside_holding_window", "next_session", "earnings"]
    assert state.state["schema"] == "state.v1.entry" and state.state["market"]["as_of"] == MARKET_AS_OF_TEXT
    assert state.state["events"]["earnings"] == EARNINGS_TEXT


def test_entry_text_appends_exactly_two_keys_after_events(builder: StateBuilder, world: FakeView) -> None:
    base = builder.entry(world, "SPY")
    assert base is not None
    text = builder.entry_text(world, "SPY", base)
    assert text is not None
    assert list(text.state) == [*list(base.state), "news_status", "news"]
    assert text.state["schema"] == "state.v1.entry_text"
    assert list(text.state["news"]) == ["since_previous_session", "earlier"]
    for key, value in base.state.items():
        if key != "schema":
            assert text.state[key] == value  # the SAME object, plus news


def test_manage_key_order_is_exactly_5_7(builder: StateBuilder, world: FakeView) -> None:
    state = builder.manage(world, _position(world)).state
    assert list(state) == ["schema", "context", "position", "changes_since_entry", "market", "underlying", "vol_surface", "events"]
    assert list(state["position"]) == [
        "structure",
        "directional_exposure",
        "vol_exposure",
        "entry_thesis",
        "sessions_held",
        "time_to_expiry",
        "pnl",
        "short_strike_distance",
        "price_vs_breakeven",
    ]
    assert list(state["changes_since_entry"]) == [
        "trend_at_entry",
        "trend_now",
        "iv_vs_realized_at_entry",
        "iv_vs_realized_now",
        "iv_change_since_entry",
        "underlying_move_since_entry",
    ]
    assert "expected_move_1_session" not in state["vol_surface"]  # 5.7: the manage surface has no expected moves
    assert list(state["context"]) == ["underlying_alias", "underlying_kind", "decision_session"]


def test_every_built_state_has_exactly_the_vocab_paths(builder: StateBuilder, world: FakeView) -> None:
    position = _position(world)
    entry = builder.entry(world, "SPY")
    assert entry is not None
    entry_text = builder.entry_text(world, "SPY", entry)
    manage = builder.manage(world, position)
    manage_text = builder.manage_text(world, position, manage)
    assert entry_text is not None and manage_text is not None
    for kind, built in (("entry", entry), ("entry_text", entry_text), ("manage", manage), ("manage_text", manage_text)):
        assert vocab.state_paths(built.state) == vocab.STATE_PATHS[kind], kind
        assert vocab.state_paths(builder.variant(built.state, Variant.BUCKET_ONLY)) == vocab.STATE_PATHS_BUCKET_ONLY[kind], kind


def test_the_structure_strings_are_the_5_7_table(builder: StateBuilder, world: FakeView) -> None:
    for kind in (StructureKind.PUT_CREDIT, StructureKind.LONG_CALL, StructureKind.IRON_CONDOR):
        state = builder.manage(world, _position(world, kind=kind)).state
        assert state["position"]["structure"] == kind.value
        assert state["position"]["directional_exposure"] == DIRECTIONAL_EXPOSURE[kind]
        assert state["position"]["vol_exposure"] == VOL_EXPOSURE[kind]
    assert VOL_EXPOSURE[StructureKind.LONG_PUT] == VOL_EXPOSURE[StructureKind.LONG_CALL]
    assert VOL_EXPOSURE[StructureKind.IRON_CONDOR] == VOL_EXPOSURE[StructureKind.PUT_CREDIT]


# ======================================================================================================================
# Facts and provenance
# ======================================================================================================================


def test_entry_facts_carry_the_entry_context_raw_material(builder: StateBuilder, world: FakeView) -> None:
    built = builder.entry(world, "SPY")
    assert built is not None
    facts = built.facts
    assert isinstance(facts, EntryFacts)
    assert facts.spot == world.spot("SPY")
    surface = built.state["vol_surface"]
    assert facts.em_hold_tenths == surface["expected_move_holding_window"]["value"]
    assert facts.iv30_bp == round(world.daily("SPY", 1)["iv30_bp"].iloc[-1])
    assert facts.events_in_window == len(built.state["events"]["inside_holding_window"])
    assert facts.trend_code == vocab.bucket_code(built.state["underlying"]["trend"]["direction"])
    assert facts.iv_rank_code == vocab.bucket_code(surface["iv_rank_1y"]["bucket"])
    assert facts.iv_rv_code == vocab.bucket_code(surface["iv_vs_realized"])
    assert facts.dist_code == vocab.bucket_code(built.state["underlying"]["momentum"]["distance_from_20d_avg_in_atr"]["bucket"])
    assert facts.news_count == 0 and facts.news_recent_count == 0 and facts.news_enabled is True


def test_the_thesis_is_written_from_bucket_codes_only(builder: StateBuilder, world: FakeView) -> None:
    built = builder.entry(world, "SPY")
    assert built is not None and isinstance(built.facts, EntryFacts)
    facts = built.facts
    expected = (
        f"opened with trend {facts.trend_code}, implied volatility {facts.iv_rv_code} versus realized, "
        f"iv rank {facts.iv_rank_code}, {facts.events_in_window} tracked event(s) inside the holding window"
    )
    assert facts.events_in_window >= 1 and facts.thesis == expected
    ensure_state_safe({"entry_thesis": facts.thesis}, masked=True, underlyings=("SPY", "QQQ", "IWM"))


def test_the_thesis_says_no_event_when_the_window_is_empty(builder: StateBuilder, terms: MaskTerms, cfg: Config) -> None:
    quiet = make_view(("SPY",), events=())
    built = builder.entry(quiet, "SPY")
    assert built is not None and isinstance(built.facts, EntryFacts)
    assert built.facts.events_in_window == 0
    assert built.facts.thesis.endswith("no tracked event inside the holding window")
    assert built.state["events"]["inside_holding_window"] == [] and built.state["events"]["coverage"].endswith("none")


def test_the_events_block_is_code_generated(builder: StateBuilder) -> None:
    session = DEFAULT_SESSION
    calendar = make_view(("SPY",)).calendar
    world = make_view(
        ("SPY",),
        events=(fomc_event(calendar.next_session(session, 2)), fomc_event(calendar.next_session(session, 1))),
    )
    built = builder.entry(world, "SPY")
    assert built is not None
    events = built.state["events"]
    assert events["inside_holding_window"] == [
        "major central-bank rate decision in 1 session",
        "major central-bank rate decision in 2 sessions",
    ]
    assert events["next_session"] == ["major central-bank rate decision in 1 session"]
    assert events["coverage"] == "scheduled events tracked: central-bank rate decisions only"


def test_provenance_carries_the_mask_version_and_the_audit_fields(builder: StateBuilder, world: FakeView, terms: MaskTerms) -> None:
    built = builder.entry(world, "SPY")
    assert built is not None
    provenance = built.provenance
    assert provenance.mask_version == terms.version and provenance.state_sha256 == built.state_hash
    assert provenance.real_symbol == "SPY" and provenance.spot_measure == world.chain("SPY").spot_measure
    assert provenance.as_of == world.as_of and provenance.data_fidelity is world.fidelity
    assert provenance.iv_hist_proxy_pct == 0 and provenance.news_ids == () and provenance.news_dropped == 0
    assert provenance.inputs and all(item.knowable_at <= world.as_of for item in provenance.inputs)
    assert provenance.raw_features["iv30_bp"] and provenance.request_id is None
    assert provenance.decision_id == ""  # the cycle stamps it (2.10)


def test_manage_facts_are_hand_computable(builder: StateBuilder, world: FakeView, cfg: Config) -> None:
    position = _position(world, liq_value=100, qty=2)
    facts = builder.manage(world, position).facts
    assert isinstance(facts, ManageFacts)
    # headline band is ORATS by default: (-(-118) - 100) * 100 * 2 contracts
    assert facts.pnl_headline == (118 - 100) * 100 * 2
    assert facts.pnl_frac_loss_ppm == 0
    losing = _position(world, liq_value=300, qty=2)
    losing_facts = builder.manage(world, losing).facts
    assert isinstance(losing_facts, ManageFacts)
    assert losing_facts.pnl_headline == (118 - 300) * 100 * 2
    assert losing_facts.pnl_frac_loss_ppm == round(1_000_000 * 36_400 / 38_000)
    assert losing_facts.move_code in vocab.MOVE_SINCE_ENTRY and losing_facts.short_dist_code in vocab.SHORT_DIST


# ======================================================================================================================
# The position blocks of 5.5 / 5.7
# ======================================================================================================================


def test_the_pnl_bucket_is_mid_to_mid_and_uses_the_structmath_formulas(builder: StateBuilder, world: FakeView) -> None:
    position = _position(world, mid_value=95)
    state = builder.manage(world, position).state
    widths = position.structure.wing_widths
    loss_base = structmath.max_loss_pc(position.structure.kind, widths, OPEN_MID, 0)
    gain_base = structmath.max_profit_pc(position.structure.kind, widths, OPEN_MID)
    pnl_mid = (-OPEN_MID - 95) * 100
    assert gain_base == -OPEN_MID * 100 and loss_base == (widths[0] + OPEN_MID) * 100
    assert state["position"]["pnl"] == buckets.pnl_bucket(pnl_mid=pnl_mid, gain_base=gain_base, loss_base=loss_base, long_premium=False)
    assert vocab.bucket_code(state["position"]["pnl"]) == "small_gain"  # 2500 of 12000 cents


def test_the_pnl_bucket_of_a_long_option_is_relative_to_the_premium_paid(builder: StateBuilder, world: FakeView) -> None:
    entry = _entry_context(open_mid=400)  # a 400c debit
    position = _position(world, kind=StructureKind.LONG_CALL, entry=entry, mid_value=-600)
    state = builder.manage(world, position).state
    assert "of the premium paid" in state["position"]["pnl"]
    assert vocab.bucket_code(state["position"]["pnl"]) == "large_gain"  # (600 - 400) / 400 = 0.5
    assert state["position"]["short_strike_distance"] is None  # no short leg
    assert state["position"]["price_vs_breakeven"] is not None  # a debit structure has one


def test_short_distance_and_breakeven_are_expressed_in_expected_moves(builder: StateBuilder, world: FakeView) -> None:
    credit = builder.manage(world, _position(world)).state["position"]
    assert credit["price_vs_breakeven"] is None  # 5.5: null for credit structures
    assert vocab.bucket_code(credit["short_strike_distance"]) in vocab.SHORT_DIST
    condor = builder.manage(world, _position(world, kind=StructureKind.IRON_CONDOR)).state["position"]
    assert vocab.bucket_code(condor["short_strike_distance"]) in vocab.SHORT_DIST
    assert condor["price_vs_breakeven"] is None


def test_the_move_since_entry_is_signed_towards_the_position(builder: StateBuilder, world: FakeView) -> None:
    ref = world.spot("SPY")
    x = math.log(ref / ENTRY_SPOT) / (ENTRY_EM_TENTHS / 1000.0)
    assert x > 0  # the fixture's spot is above the entry spot
    bullish = builder.manage(world, _position(world)).state["changes_since_entry"]["underlying_move_since_entry"]
    assert bullish["value"] == round(x) and bullish["bucket"] == buckets.bucketize(x, buckets.MOVE_SINCE_ENTRY)
    condor = builder.manage(world, _position(world, kind=StructureKind.IRON_CONDOR)).state["changes_since_entry"]
    move = condor["underlying_move_since_entry"]
    assert move["value"] == round(-abs(x))  # 5.5: the condor uses -abs(x)
    assert move["bucket"] == buckets.bucketize(-abs(x), buckets.MOVE_SINCE_ENTRY)


def test_sessions_held_and_time_to_expiry_count_to_the_last_trading_day(builder: StateBuilder, world: FakeView) -> None:
    position = _position(world, sessions_held=6)
    state = builder.manage(world, position).state
    assert state["position"]["sessions_held"]["value"] == 6
    assert state["position"]["time_to_expiry"]["value"] == (position.structure.last_session - world.session).days
    assert position.structure.last_session == world.calendar.prev_or_same_session(position.structure.expiry)


def test_changes_since_entry_carry_bare_codes(builder: StateBuilder, world: FakeView) -> None:
    changes = builder.manage(world, _position(world)).state["changes_since_entry"]
    for key in ("trend_at_entry", "trend_now", "iv_vs_realized_at_entry", "iv_vs_realized_now"):
        assert ":" not in changes[key] and changes[key] in vocab.ALL_BUCKET_CODES
    assert changes["trend_at_entry"] == "up" and changes["iv_vs_realized_at_entry"] == "iv_rich"
    iv30_bp = round(world.daily("SPY", 1)["iv30_bp"].iloc[-1])
    assert changes["iv_change_since_entry"] == buckets.bucketize(iv30_bp / ENTRY_IV30_BP - 1.0, buckets.CHANGE5)


# ======================================================================================================================
# Purity, text isolation and path independence
# ======================================================================================================================


def test_two_builds_of_the_same_view_give_the_same_bytes(builder: StateBuilder, world: FakeView) -> None:
    position = _position(world)
    first_entry = builder.entry(world, "SPY")
    second_entry = builder.entry(world, "SPY")
    assert first_entry is not None and second_entry is not None
    assert dumps_ordered(first_entry.state) == dumps_ordered(second_entry.state)
    assert first_entry.state_hash == second_entry.state_hash and first_entry.facts == second_entry.facts
    assert dumps_ordered(builder.manage(world, position).state) == dumps_ordered(builder.manage(world, position).state)
    text_first = builder.entry_text(world, "SPY", first_entry)
    text_second = builder.entry_text(world, "SPY", second_entry)
    assert text_first is not None and text_second is not None
    assert text_first.state_hash == text_second.state_hash


def test_the_text_free_states_are_invariant_to_arbitrary_news(builder: StateBuilder, terms: MaskTerms, cfg: Config) -> None:
    """D11 / V6 / INV-16: no news byte can move a gate, a size or a rank, because it never enters these states."""
    session = DEFAULT_SESSION
    as_of = make_chain("SPY", session=session).ts
    hostile = cases.as_items(cases.load(cases.HOSTILE), as_of=as_of)
    benign = cases.as_items(cases.load(cases.BENIGN), as_of=as_of)
    quiet = make_view(("SPY",), news=(), news_covered=False)
    hashes = set()
    manage_hashes = set()
    corpora: tuple[tuple[list[NewsItem], bool], ...] = (
        ([], False),
        (hostile, True),
        (benign, True),
        ([*hostile, *benign], True),
    )
    for news, covered in corpora:
        world = make_view(("SPY",), news=news, news_covered=covered)
        built = builder.entry(world, "SPY")
        assert built is not None
        hashes.add(built.state_hash)
        manage_hashes.add(builder.manage(world, _position(world)).state_hash)
    assert len(hashes) == 1 and len(manage_hashes) == 1
    assert builder.entry(quiet, "SPY") is not None


def test_the_manage_state_is_invariant_to_our_fill_prices(builder: StateBuilder, world: FakeView) -> None:
    """5.7 / 10.8: everything position-specific comes from `pos.structure` and `pos.entry`, never from the fill."""
    base = builder.manage(world, _position(world)).state_hash
    for open_net in (BandPrices(orats=-100, worst=-90, mid=-95), BandPrices(orats=-140, worst=-130, mid=-135)):
        other = _position(world, open_net=open_net, liq_value=250, qty=7)
        assert builder.manage(world, other).state_hash == base


def test_the_manage_state_is_rebuilt_byte_identically_from_a_replayed_entry_context(builder: StateBuilder, world: FakeView) -> None:
    """The `EntryContext` reaches a resumed `Position` through the ledger (ORDER_INTENT -> Book.apply(FILL))."""
    live = _position(world)
    replayed_entry = msgspec.json.decode(msgspec.json.encode(live.entry), type=EntryContext)
    replayed = _position(world, entry=replayed_entry)
    assert replayed.entry == live.entry
    assert builder.manage(world, replayed).state_hash == builder.manage(world, live).state_hash


def test_the_state_is_gated_by_ensure_state_safe(builder: StateBuilder, world: FakeView) -> None:
    built = builder.entry(world, "SPY")
    assert built is not None
    ensure_state_safe(built.state, masked=True, underlyings=("SPY", "QQQ", "IWM"), max_chars=16_000)
    with pytest.raises(StateLeak):  # the gate really is the one that would catch a leak
        ensure_state_safe({**built.state, "leak": "SPY at $450 on 2024-05-17"}, masked=True, underlyings=("SPY",))


# ======================================================================================================================
# News in the state (5.6 / 5.8)
# ======================================================================================================================


def test_the_news_block_is_split_by_the_code_side_cutoff(builder: StateBuilder, world: FakeView) -> None:
    base = builder.entry(world, "SPY")
    assert base is not None
    built = builder.entry_text(world, "SPY", base)
    assert built is not None and isinstance(built.facts, EntryFacts)
    news = built.state["news"]
    assert built.state["news_status"] == "present"
    assert len(news["since_previous_session"]) == 5 and len(news["earlier"]) == 3
    assert built.facts.news_count == 8 and built.facts.news_recent_count == 5
    assert built.provenance.news_ids == ("b05", "b04", "b03", "b02", "b01", "b06", "b07", "b08")
    ages = [item["age"] for item in news["since_previous_session"]]
    assert ages == sorted(ages, key=lambda text: int(text.rstrip("hd")) if text != "under 1h" else 0)


def test_entry_text_is_none_when_news_is_off_or_uncovered(cfg: Config, terms: MaskTerms, world: FakeView) -> None:
    off = StateBuilder(cfg, terms, news_resolved=False)
    base = off.entry(world, "SPY")
    assert base is not None
    assert off.entry_text(world, "SPY", base) is None  # `text: "off"` in the DECISION payload
    uncovered = make_view(("SPY",), news=(), news_covered=False)
    builder = StateBuilder(cfg, terms, news_resolved=True)
    uncovered_base = builder.entry(uncovered, "SPY")
    assert uncovered_base is not None
    assert builder.entry_text(uncovered, "SPY", uncovered_base) is None  # `text: "no_archive"`


def test_news_status_is_none_in_window_when_nothing_survives(builder: StateBuilder, cfg: Config) -> None:
    world = make_view(("SPY",), news=(), news_covered=True)
    base = builder.entry(world, "SPY")
    assert base is not None
    built = builder.entry_text(world, "SPY", base)
    assert built is not None and isinstance(built.facts, EntryFacts)
    assert built.state["news_status"] == "none_in_window"
    assert built.state["news"] == {"since_previous_session": [], "earlier": []}
    assert built.facts.news_count == 0 and built.facts.news_recent_count == 0


def test_the_whole_state_is_trimmed_to_state_max_chars_oldest_first(terms: MaskTerms, world: FakeView) -> None:
    plain = StateBuilder(Config(), terms, news_resolved=True)
    untrimmed_base = plain.entry(world, "SPY")
    assert untrimmed_base is not None
    untrimmed = plain.entry_text(world, "SPY", untrimmed_base)
    assert untrimmed is not None
    budget = len(dumps_ordered(untrimmed.state)) - 400  # room for about two fewer items
    cfg = msgspec.structs.replace(Config(), state=StateConfig(max_chars=budget))
    builder = StateBuilder(cfg, terms, news_resolved=True)
    base = builder.entry(world, "SPY")
    assert base is not None
    built = builder.entry_text(world, "SPY", base)
    assert built is not None and isinstance(built.facts, EntryFacts)
    assert len(dumps_ordered(built.state)) <= budget
    assert 0 < built.facts.news_count < 8
    kept = built.facts.news_count
    assert built.provenance.news_ids == ("b05", "b04", "b03", "b02", "b01", "b06", "b07", "b08")[:kept]  # the oldest went first


def test_manage_text_is_none_without_news_since_entry(builder: StateBuilder, cfg: Config, terms: MaskTerms) -> None:
    world = make_view(("SPY",), news=(), news_covered=True)
    position = _position(world)
    base = builder.manage(world, position)
    assert builder.manage_text(world, position, base) is None


def test_manage_text_reads_only_items_created_since_the_opening_session(builder: StateBuilder) -> None:
    session = DEFAULT_SESSION
    calendar = make_view(("SPY",)).calendar
    as_of = make_chain("SPY", session=session).ts
    open_session = calendar.prev_session(session, 6)
    before = calendar.open_close(open_session)[0] - timedelta(hours=2)
    items = [
        news_item("old", "Filed before the position existed", created_at=before, symbols=("SPY",)),
        news_item("new", "Filed after the position was opened", created_at=as_of - timedelta(hours=3), symbols=("SPY",)),
    ]
    world = make_view(("SPY",), news=items, news_covered=True)
    position = _position(world, sessions_held=6)
    base = builder.manage(world, position)
    built = builder.manage_text(world, position, base)
    assert built is not None and isinstance(built.facts, ManageFacts)
    assert list(built.state)[-1] == "news_since_entry"
    headlines = [item["headline"] for key in ("since_previous_session", "earlier") for item in built.state["news_since_entry"][key]]
    assert headlines == ["Filed after the position was opened"]
    assert built.facts.news_count == 1 and built.facts.news_recent_count == 1
    assert built.state["schema"] == "state.v1.manage_text"


# ======================================================================================================================
# Variants and the bucket_only render (5.9)
# ======================================================================================================================


def test_key_perm_reverses_every_dict_and_keeps_list_order(builder: StateBuilder, world: FakeView) -> None:
    base = builder.entry(world, "SPY")
    assert base is not None
    text = builder.entry_text(world, "SPY", base)
    assert text is not None
    permuted = builder.variant(text.state, Variant.KEY_PERM)
    assert list(permuted) == list(reversed(list(text.state)))
    assert list(permuted["market"]) == list(reversed(list(text.state["market"])))
    assert list(permuted["underlying"]["trend"]) == list(reversed(list(text.state["underlying"]["trend"])))
    assert permuted["news"]["since_previous_session"] == [
        {key: item[key] for key in reversed(list(item))} for item in text.state["news"]["since_previous_session"]
    ]
    assert [item["headline"] for item in permuted["news"]["since_previous_session"]] == [
        item["headline"] for item in text.state["news"]["since_previous_session"]
    ]
    assert dumps_ordered(permuted) != dumps_ordered(text.state)  # different content, hence a different cache key (D8)


def test_bucket_only_collapses_value_dicts_except_the_expected_moves(builder: StateBuilder, world: FakeView) -> None:
    base = builder.entry(world, "SPY")
    assert base is not None
    only = builder.variant(base.state, Variant.BUCKET_ONLY)
    assert only["market"]["vol_index_pctile_1y"] == base.state["market"]["vol_index_pctile_1y"]["bucket"]
    assert only["context"]["holding_window_sessions"] == base.state["context"]["holding_window_sessions"]["bucket"]
    assert only["vol_surface"]["iv_rank_1y"] == base.state["vol_surface"]["iv_rank_1y"]["bucket"]
    for name in ("expected_move_1_session", "expected_move_5_sessions", "expected_move_holding_window"):
        assert only["vol_surface"][name] == base.state["vol_surface"][name]  # their integers are the eval thresholds
    assert builder.variant(only, Variant.BUCKET_ONLY) == only  # idempotent


def test_opt_perm_and_base_are_copies_of_the_state(builder: StateBuilder, world: FakeView) -> None:
    base = builder.entry(world, "SPY")
    assert base is not None
    for variant in (Variant.BASE, Variant.OPT_PERM):
        rendered = builder.variant(base.state, variant)
        assert rendered == base.state and rendered is not base.state
        rendered["market"]["as_of"] = "mutated"
        assert base.state["market"]["as_of"] == MARKET_AS_OF_TEXT  # a deep copy: the source is untouched


def test_state_render_bucket_only_equals_the_variant_byte_for_byte(world: FakeView, terms: MaskTerms) -> None:
    """5.1: a recorded run with `perturbation_scope = "all"` already holds every answer the D11 ablation arm needs."""
    plain = StateBuilder(Config(), terms, news_resolved=True)
    rendered = StateBuilder(msgspec.structs.replace(Config(), state=StateConfig(render="bucket_only")), terms, news_resolved=True)
    base = plain.entry(world, "SPY")
    only = rendered.entry(world, "SPY")
    assert base is not None and only is not None
    assert dumps_ordered(only.state) == dumps_ordered(plain.variant(base.state, Variant.BUCKET_ONLY))
    assert only.state_hash == sha256_hex(dumps_ordered(plain.variant(base.state, Variant.BUCKET_ONLY)))
    position = _position(world)
    manage_base = plain.manage(world, position)
    manage_only = rendered.manage(world, position)
    assert dumps_ordered(manage_only.state) == dumps_ordered(plain.variant(manage_base.state, Variant.BUCKET_ONLY))
    # the text state inherits the rendering of its base
    text = rendered.entry_text(world, "SPY", only)
    assert text is not None and text.state["market"]["vol_index_pctile_1y"] == only.state["market"]["vol_index_pctile_1y"]


# ======================================================================================================================
# Unavailable features, the diagnostic identity block and refusals
# ======================================================================================================================


def test_entry_returns_none_when_a_required_feature_is_missing(builder: StateBuilder, world: FakeView) -> None:
    short = world.at(as_of=world.as_of)
    short.tables.daily["SPY"] = short.tables.daily["SPY"].tail(30).reset_index(drop=True)
    assert builder.entry(short, "SPY") is None


def test_manage_never_returns_none_even_without_a_surface(builder: StateBuilder, world: FakeView) -> None:
    position = _position(world)
    stripped = world.at(as_of=world.as_of)
    stripped.tables.daily["SPY"] = stripped.tables.daily["SPY"].tail(3).reset_index(drop=True)
    built = builder.manage(stripped, position)
    assert built.state["vol_surface"]["iv_rank_1y"] == vocab.UNAVAILABLE
    assert built.state["position"]["entry_thesis"] == ENTRY_THESIS  # the ledgered context still carries the thesis
    assert vocab.state_paths(built.state) <= vocab.STATE_PATHS["manage"]


def test_the_identity_block_appears_only_with_the_leakage_flag(world: FakeView, terms: MaskTerms) -> None:
    cfg = msgspec.structs.replace(Config(), state=StateConfig(unmasked=True))
    builder = StateBuilder(cfg, terms, news_resolved=True)
    built = builder.entry(world, "SPY")
    assert built is not None
    identity = built.state["identity"]
    assert list(built.state)[-1] == "identity"
    assert identity["ticker"] == "SPY" and identity["date"] == world.session.isoformat()
    assert identity["spot"] == "450.00 dollars"
    with pytest.raises(StateLeak):  # it would never pass the masked gate
        ensure_state_safe(built.state, masked=True, underlyings=("SPY",))
    text = builder.entry_text(world, "SPY", built)
    assert text is not None and list(text.state)[-1] == "identity"  # the identity block stays last
    assert list(text.state)[-3:] == ["news_status", "news", "identity"]


def test_a_missing_alias_is_refused_rather_than_leaking_the_ticker(world: FakeView, terms: MaskTerms) -> None:
    from jevbot.config import UniverseConfig
    from jevbot.errors import InvariantError

    cfg = msgspec.structs.replace(Config(), universe=UniverseConfig(alias={}, kind={"SPY": "an index ETF"}))
    builder = StateBuilder(cfg, terms, news_resolved=True)
    with pytest.raises(InvariantError, match="alias"):
        builder.entry(world, "SPY")
    no_kind = msgspec.structs.replace(Config(), universe=UniverseConfig(kind={}))
    with pytest.raises(InvariantError, match="kind"):
        StateBuilder(no_kind, terms, news_resolved=True).entry(world, "SPY")
    assert builder.entry(world, "ZZZ") is None  # an underlying without data never reaches the alias table


def test_the_builder_implements_the_protocol() -> None:
    from jevbot.protocols import StateBuilderP

    builder: StateBuilderP = StateBuilder(Config(), load_mask_terms(MASK_TERMS_FILE), news_resolved=False)
    assert callable(builder.entry) and callable(builder.manage) and callable(builder.variant)


def test_no_state_is_larger_than_the_documented_budget(builder: StateBuilder, world: FakeView) -> None:
    base = builder.entry(world, "SPY")
    assert base is not None
    text = builder.entry_text(world, "SPY", base)
    assert text is not None
    assert len(dumps_ordered(base.state)) <= 3_500  # 5.6: ~2.5k characters text-free
    assert len(dumps_ordered(text.state)) <= 6_500  # 5.6: <= 6.5k with news
