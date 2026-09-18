"""Text-isolation / monotonicity properties (DESIGN.md 7.4, 7.5, 7.7; INV-16), on the entry AND the manage path.

INV-16: text-derived answers can veto or re-rank only. They never relax a gate, raise a tier, raise a limit, or alone
produce `enter` **or** `close`. The properties below hold that line for random answer sets:

entry (7.4 / 7.5)
  E1  changing **any** text answer arbitrarily never changes `score_core_ppm` or `tier_ppm` (the composite is text-free);
  E2  making any text veto worse (higher probability) never flips `no_trade` to `enter` and never raises `tier_ppm`;
  E3  moving the rank inputs in the unfavourable direction does the same - only `score_rank_ppm` may move, and only down;
  E4  no text answer can turn a `no_trade` into an `enter`, whatever its value.

manage (7.7)
  M1  without market-data confirmation in the facts, **no** value of the two text answers changes `action` or `reason`;
  M2  with confirmation, raising a text answer can only move `hold` to `close`, never the reverse;
  M3  text answers never change `exit_latch` or `pressure_ppm`.

Seeded `numpy` generators only - the project takes no hypothesis dependency (DESIGN 1.1).
"""

import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, Final

import numpy as np
import pytest

from jevbot import vocab
from jevbot.config import RulesConfig
from jevbot.rules import DecisionRules
from jevbot.types import (
    Answer,
    BandPrices,
    ChoiceAns,
    DecisionResult,
    EntryContext,
    EntryFacts,
    ManageFacts,
    NoulAns,
    Position,
    RequestKind,
    ScoreAns,
    Slot,
    SnapshotKey,
    StructureKind,
    Variant,
)
from tests.fixtures.chain_factory import make_chain, make_structure

SEED: Final = 20260918
DRAWS: Final = 250
UNDERLYING: Final = "SPY"
DECISION_ID: Final = "d" * 24
POSITION_ID: Final = "p" * 24

TEXT_VETOES: Final[tuple[str, ...]] = ("text.pending_binary", "text.market_stress", "text.clearly_negative", "text.clearly_positive")
TEXT_IDS: Final[tuple[str, ...]] = ("text.material_present", *TEXT_VETOES)
MANAGE_TEXT_IDS: Final[tuple[str, ...]] = ("pos.adverse_news_since_entry", "pos.pending_binary_since_entry")
TREND_CODES: Final[tuple[str, ...]] = vocab.TREND_DIR
CONFIRMING_MOVES: Final[frozenset[str]] = frozenset({"adverse", "strongly_adverse"})
ALL_KINDS: Final[tuple[StructureKind, ...]] = tuple(StructureKind)


# ======================================================================================================================
# answer builders (7.1)
# ======================================================================================================================


def choice(qid: str, probs: Sequence[float]) -> ChoiceAns:
    labels = vocab.CHOICE_LABELS[qid]
    total = float(sum(probs))
    values = {label: float(p) / total for label, p in zip(labels, probs, strict=True)}
    ranked = sorted(values.items(), key=lambda item: (-item[1], labels.index(item[0])))
    entropy = -sum(p * math.log(p) for p in values.values() if p > 0.0) / math.log(len(labels))
    return ChoiceAns(
        probs=values,
        top=ranked[0][0],
        p_top=ranked[0][1],
        margin=ranked[0][1] - ranked[1][1],
        entropy=entropy,
        raw_sum=1.0,
        server_choice=ranked[0][0],
        server_confidence=ranked[0][1],
    )


def score(probs: Sequence[float]) -> ScoreAns:
    total = float(sum(probs))
    values = tuple(float(p) / total for p in probs)
    mean = sum(i * p for i, p in enumerate(values))
    ranked = sorted(range(len(values)), key=lambda i: (-values[i], i))
    entropy = -sum(p * math.log(p) for p in values if p > 0.0) / math.log(len(values))
    return ScoreAns(
        probs=values,
        mean=mean,
        norm=mean / (len(values) - 1),
        top=ranked[0],
        p_top=values[ranked[0]],
        margin=values[ranked[0]] - values[ranked[1]],
        entropy=entropy,
        raw_sum=1.0,
        server_score=mean,
        server_confidence=values[ranked[0]],
    )


def result(answers: Mapping[str, Answer], kind: RequestKind) -> DecisionResult:
    return DecisionResult(
        decision_id=DECISION_ID,
        kind=kind,
        variant=Variant.BASE,
        state_hash="0" * 64,
        question_set_hash="1" * 64,
        model="mock-1",
        answers=dict(answers),
        cache_keys=dict.fromkeys(answers, "2" * 64),
        source="mock",
        request_id=None,
        input_tokens=None,
        latency_ms=None,
    )


# ======================================================================================================================
# random draws
# ======================================================================================================================


def random_core(rng: np.random.Generator) -> dict[str, Answer]:
    """A random text-free entry answer set: Dirichlet-mixed Choices (so gates pass sometimes and fail sometimes)."""

    def dirichlet(qid: str, sharpness: float) -> ChoiceAns:
        labels = vocab.CHOICE_LABELS[qid]
        return choice(qid, list(rng.dirichlet(np.full(len(labels), sharpness))))

    sharpness = float(rng.choice([0.2, 0.5, 1.0, 3.0]))
    return {
        "regime.market": dirichlet("regime.market", sharpness),
        "under.direction": dirichlet("under.direction", sharpness),
        "under.stretched": NoulAns(p=float(rng.random())),
        "vol.stance": dirichlet("vol.stance", sharpness),
        "vol.explained_by_event": NoulAns(p=float(rng.random())),
        "fit.structure_family": dirichlet("fit.structure_family", sharpness),
        "risk.environment": score(list(rng.dirichlet(np.full(4, sharpness)))),
    }


def random_text(rng: np.random.Generator) -> dict[str, Answer]:
    return {qid: NoulAns(p=float(rng.random())) for qid in TEXT_IDS}


def random_entry_facts(rng: np.random.Generator) -> EntryFacts:
    return EntryFacts(
        trend_code=str(rng.choice(list(TREND_CODES))),
        iv_rank_code=str(rng.choice(list(vocab.PCTL5))),
        iv_rv_code=str(rng.choice(list(vocab.IV_RV))),
        dist_code=str(rng.choice(list(vocab.DIST_ATR))),
        news_enabled=True,
        news_count=int(rng.integers(1, 8)),
        news_recent_count=int(rng.integers(0, 4)),
        spot=45_000,
        iv30_bp=1600,
        em_hold_tenths=35,
        events_in_window=int(rng.integers(0, 3)),
        thesis="generated",
    )


def worsen_vetoes(text: Mapping[str, Answer], rng: np.random.Generator) -> dict[str, Answer]:
    """Every text veto probability moved UP (never down): strictly worse news, in the rules' own direction."""
    out = dict(text)
    for qid in TEXT_VETOES:
        current = out[qid]
        assert isinstance(current, NoulAns)
        out[qid] = NoulAns(p=current.p + (1.0 - current.p) * float(rng.random()))
    return out


def worsen_rank(text: Mapping[str, Answer], direction_is_bullish: bool) -> dict[str, Answer]:
    """The 7.5 rank inputs moved unfavourably: material news, tone pushed against the structure's direction."""
    out = dict(text)
    out["text.material_present"] = NoulAns(p=1.0)
    out["text.clearly_positive"] = NoulAns(p=0.0 if direction_is_bullish else 1.0)
    out["text.clearly_negative"] = NoulAns(p=1.0 if direction_is_bullish else 0.0)
    return out


# ======================================================================================================================
# entry path
# ======================================================================================================================


def _peaked(qid: str, top: str, weight: float) -> ChoiceAns:
    labels = vocab.CHOICE_LABELS[qid]
    rest = (1.0 - weight) / (len(labels) - 1)
    return choice(qid, [weight if label == top else rest for label in labels])


def favourable_core(kind: StructureKind, rng: np.random.Generator) -> dict[str, Answer]:
    """An answer set aimed at `kind`: peaked enough that the 7.2 gates and the score floor are often cleared."""
    from jevbot.types import STRUCTURE_DIRECTION, STRUCTURE_STANCE, Direction

    direction = STRUCTURE_DIRECTION[kind]
    regime_top = {Direction.BULLISH: "trending_up_calm", Direction.BEARISH: "orderly_downtrend", Direction.NEUTRAL: "range_bound_calm"}[
        direction
    ]
    return {
        "regime.market": _peaked("regime.market", regime_top, float(rng.uniform(0.50, 0.95))),
        "under.direction": _peaked("under.direction", direction.value, float(rng.uniform(0.60, 0.95))),
        "under.stretched": NoulAns(p=float(rng.uniform(0.0, 0.5))),
        "vol.stance": _peaked("vol.stance", STRUCTURE_STANCE[kind].value, float(rng.uniform(0.60, 0.95))),
        "vol.explained_by_event": NoulAns(p=float(rng.uniform(0.0, 0.4))),
        "fit.structure_family": _peaked("fit.structure_family", kind.value, float(rng.uniform(0.50, 0.95))),
        "risk.environment": score([3.0, 1.0, float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 0.5))]),
    }


def facts_for(kind: StructureKind, rng: np.random.Generator) -> EntryFacts:
    """Facts that satisfy the 7.3 cross-checks of `kind` (so the crosscheck step is not the one that always fires)."""
    from jevbot.types import SHORT_PREMIUM, STRUCTURE_DIRECTION, Direction

    direction = STRUCTURE_DIRECTION[kind]
    trend = {
        Direction.BULLISH: ("up", "flat", "mixed"),
        Direction.BEARISH: ("down", "flat", "mixed"),
        Direction.NEUTRAL: ("flat", "mixed"),
    }[direction]
    iv_rv = ("iv_rich", "iv_very_rich") if kind in SHORT_PREMIUM else vocab.IV_RV
    iv_rank = ("very_low", "low") if kind in (StructureKind.LONG_CALL, StructureKind.LONG_PUT) else vocab.PCTL5
    return EntryFacts(
        trend_code=str(rng.choice(list(trend))),
        iv_rank_code=str(rng.choice(list(iv_rank))),
        iv_rv_code=str(rng.choice(list(iv_rv))),
        dist_code=str(rng.choice(list(vocab.DIST_ATR))),
        news_enabled=True,
        news_count=int(rng.integers(1, 8)),
        news_recent_count=int(rng.integers(0, 4)),
        spot=45_000,
        iv30_bp=1600,
        em_hold_tenths=35,
        events_in_window=int(rng.integers(0, 3)),
        thesis="generated",
    )


def _entry_draws() -> list[tuple[dict[str, Answer], dict[str, Answer], EntryFacts]]:
    """Half the draws aim at a structure (so `enter` actually happens), half are pure Dirichlet noise."""
    rng = np.random.default_rng(SEED)
    draws: list[tuple[dict[str, Answer], dict[str, Answer], EntryFacts]] = []
    for i in range(DRAWS):
        if i % 2 == 0:
            kind = ALL_KINDS[int(rng.integers(len(ALL_KINDS)))]
            draws.append((favourable_core(kind, rng), random_text(rng), facts_for(kind, rng)))
        else:
            draws.append((random_core(rng), random_text(rng), random_entry_facts(rng)))
    return draws


ENTRY_DRAWS: Final = _entry_draws()


def _non_text(reasons: Sequence[str]) -> list[str]:
    return [code for code in reasons if not code.startswith("veto:text.")]


def _veto_qids(reasons: Sequence[str]) -> set[str]:
    return {code.split(":")[1] for code in reasons if code.startswith("veto:text.")}


def decide(engine: DecisionRules, core: Mapping[str, Answer], text: Mapping[str, Answer] | None, facts: EntryFacts) -> Any:
    text_result = None if text is None else result(text, RequestKind.ENTRY_TEXT)
    return engine.decide_entry(UNDERLYING, DECISION_ID, result(core, RequestKind.ENTRY), text_result, facts)


def test_the_entry_draws_cover_both_outcomes() -> None:
    engine = DecisionRules(RulesConfig(), ALL_KINDS)
    actions = {decide(engine, core, None, facts).action for core, _text, facts in ENTRY_DRAWS}
    assert actions == {"enter", "no_trade"}, "the generator must produce both outcomes or the properties are vacuous"


def test_e1_no_text_answer_can_move_the_composite_or_the_tier() -> None:
    engine = DecisionRules(RulesConfig(), ALL_KINDS)
    rng = np.random.default_rng(SEED + 1)
    for core, text, facts in ENTRY_DRAWS:
        text_free = decide(engine, core, None, facts)
        for _ in range(3):
            other = random_text(rng)
            with_text = decide(engine, core, other, facts)
            assert with_text.score_core_ppm == text_free.score_core_ppm
            assert with_text.tier_ppm == text_free.tier_ppm
            assert with_text.kind == text_free.kind
            for name in ("align", "volfit", "fit", "regimefit", "calm"):
                assert with_text.features_ppm.get(name) == text_free.features_ppm.get(name)
        # the text result only ever adds veto reasons on top of the text-free ones
        vetoed = decide(engine, core, text, facts)
        assert [code for code in vetoed.reasons if not code.startswith("veto:text")] == list(text_free.reasons)


def test_e2_worse_text_vetoes_never_open_a_trade_and_never_raise_the_tier() -> None:
    engine = DecisionRules(RulesConfig(), ALL_KINDS)
    rng = np.random.default_rng(SEED + 2)
    flips = 0
    for core, text, facts in ENTRY_DRAWS:
        base = decide(engine, core, text, facts)
        worse = decide(engine, core, worsen_vetoes(text, rng), facts)
        if base.action == "no_trade":
            assert worse.action == "no_trade"
        elif worse.action == "no_trade":
            flips += 1
        assert worse.tier_ppm <= base.tier_ppm
        assert worse.score_core_ppm == base.score_core_ppm
        # the text-free reasons are untouched, and no text veto ever RELAXES (a band may only move CLEAR -> UNCERTAIN -> VETO)
        assert _non_text(base.reasons) == _non_text(worse.reasons)
        assert _veto_qids(base.reasons) <= _veto_qids(worse.reasons)
        for qid in _veto_qids(base.reasons):
            if f"veto:{qid}:hard" in base.reasons:
                assert f"veto:{qid}:hard" in worse.reasons
    assert flips > 0, "worsening the vetoes must actually block some entries or the property is vacuous"


def test_e3_unfavourable_rank_inputs_only_move_the_rank_score_down() -> None:
    engine = DecisionRules(RulesConfig(), ALL_KINDS)
    moved = 0
    for core, text, facts in ENTRY_DRAWS:
        base = decide(engine, core, text, facts)
        if base.kind is None:
            continue
        from jevbot.types import STRUCTURE_DIRECTION, Direction

        bullish = STRUCTURE_DIRECTION[base.kind] is Direction.BULLISH
        if STRUCTURE_DIRECTION[base.kind] is Direction.NEUTRAL:
            continue  # news_align is 1 - |tone| there: both tone directions are unfavourable, covered by E1 / E4
        worse = decide(engine, core, worsen_rank(text, bullish), facts)
        assert worse.score_core_ppm == base.score_core_ppm
        assert worse.tier_ppm <= base.tier_ppm
        assert worse.features_ppm["news_align"] <= base.features_ppm["news_align"]
        assert worse.score_rank_ppm <= base.score_rank_ppm
        moved += int(worse.score_rank_ppm < base.score_rank_ppm)
    assert moved > 0


def test_e4_text_can_never_turn_a_no_trade_into_an_enter() -> None:
    engine = DecisionRules(RulesConfig(), ALL_KINDS)
    rng = np.random.default_rng(SEED + 3)
    for core, _text, facts in ENTRY_DRAWS:
        text_free = decide(engine, core, None, facts)
        if text_free.action != "no_trade":
            continue
        for _ in range(4):
            assert decide(engine, core, random_text(rng), facts).action == "no_trade"
        best_case = {qid: NoulAns(p=0.0) for qid in TEXT_IDS}
        assert decide(engine, core, best_case, facts).action == "no_trade"


# ======================================================================================================================
# manage path
# ======================================================================================================================


def _position(kind: StructureKind) -> Position:
    chain = make_chain()
    return Position(
        position_id=POSITION_ID,
        structure=make_structure(chain, kind),
        qty=1,
        open_key=SnapshotKey(session=date(2024, 5, 17), slot=Slot.EOD),
        open_decision_id=DECISION_ID,
        open_net=BandPrices(orats=-100, worst=-95, mid=-102),
        max_loss=40_000,
        max_profit=10_000,
        bp_reserved=40_000,
        entry=EntryContext(
            entry_thesis="generated",
            entry_codes={"trend": "up", "iv_vs_realized": "iv_rich", "iv_rank": "middle"},
            entry_spot=45_000,
            entry_iv30_bp=1600,
            entry_em_hold_tenths=35,
            open_mid_at_decision=-102,
        ),
    )


POSITIONS: Final[Mapping[StructureKind, Position]] = {kind: _position(kind) for kind in ALL_KINDS}


def random_manage_core(rng: np.random.Generator) -> dict[str, Answer]:
    return {
        "pos.thesis_invalidated": NoulAns(p=float(rng.random())),
        "pos.short_strike_threat": score(list(rng.dirichlet(np.full(4, 0.6)))),
        "pos.action": choice("pos.action", list(rng.dirichlet(np.full(4, 0.6)))),
    }


def random_manage_text(rng: np.random.Generator) -> dict[str, Answer]:
    return {qid: NoulAns(p=float(rng.random())) for qid in MANAGE_TEXT_IDS}


def random_manage_facts(rng: np.random.Generator, *, confirmed: bool) -> ManageFacts:
    moves = sorted(CONFIRMING_MOVES) if confirmed else [c for c in vocab.MOVE_SINCE_ENTRY if c not in CONFIRMING_MOVES]
    return ManageFacts(
        pnl_headline=int(rng.integers(-50_000, 50_000)),
        pnl_frac_loss_ppm=int(rng.integers(250_000, 1_000_000)) if confirmed else int(rng.integers(0, 250_000)),
        move_code=str(rng.choice(moves)),
        short_dist_code=str(rng.choice(list(vocab.SHORT_DIST))),
        news_count=int(rng.integers(1, 8)),
        news_recent_count=int(rng.integers(0, 4)),
    )


def manage(engine: DecisionRules, pos: Position, core: Mapping[str, Answer], text: Mapping[str, Answer] | None, facts: ManageFacts) -> Any:
    text_result = None if text is None else result(text, RequestKind.MANAGE_TEXT)
    return engine.decide_manage(pos, DECISION_ID, None, result(core, RequestKind.MANAGE), text_result, facts)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_m1_without_confirmation_text_changes_nothing_but_the_watch_counter(kind: StructureKind) -> None:
    engine = DecisionRules(RulesConfig(), ALL_KINDS)
    rng = np.random.default_rng(SEED + 10)
    pos = POSITIONS[kind]
    for _ in range(DRAWS):
        core = random_manage_core(rng)
        facts = random_manage_facts(rng, confirmed=False)
        baseline = manage(engine, pos, core, None, facts)
        for text in (
            {qid: NoulAns(p=0.0) for qid in MANAGE_TEXT_IDS},
            {qid: NoulAns(p=1.0) for qid in MANAGE_TEXT_IDS},
            random_manage_text(rng),
        ):
            decision = manage(engine, pos, core, text, facts)
            assert decision.action == baseline.action
            assert decision.reason == baseline.reason
            assert decision.exit_latch == baseline.exit_latch
            assert decision.pressure_ppm == baseline.pressure_ppm
            assert decision.watch_text in (0, pos.watch_text + 1)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_m2_with_confirmation_raising_a_text_answer_can_only_close(kind: StructureKind) -> None:
    engine = DecisionRules(RulesConfig(), ALL_KINDS)
    rng = np.random.default_rng(SEED + 20)
    pos = POSITIONS[kind]
    closes = 0
    for _ in range(DRAWS):
        core = random_manage_core(rng)
        facts = random_manage_facts(rng, confirmed=True)
        low = {qid: NoulAns(p=float(rng.random()) * 0.5) for qid in MANAGE_TEXT_IDS}
        high = {qid: NoulAns(p=0.5 + float(rng.random()) * 0.5) for qid in MANAGE_TEXT_IDS}
        lower = manage(engine, pos, core, low, facts)
        higher = manage(engine, pos, core, high, facts)
        if lower.action == "close":
            assert higher.action == "close"
        closes += int(higher.action == "close" and lower.action == "hold")
        assert higher.exit_latch == lower.exit_latch
        assert higher.pressure_ppm == lower.pressure_ppm
    assert closes > 0, "confirmed adverse text must close something or the property is vacuous"


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_m3_text_never_touches_the_latch_or_the_exit_pressure(kind: StructureKind) -> None:
    engine = DecisionRules(RulesConfig(), ALL_KINDS)
    rng = np.random.default_rng(SEED + 30)
    pos = POSITIONS[kind]
    latched = POSITIONS[kind].__class__(
        **{
            **{field: getattr(POSITIONS[kind], field) for field in POSITIONS[kind].__struct_fields__},
            "exit_latch": True,
        }
    )
    for holder in (pos, latched):
        for _ in range(DRAWS // 2):
            core = random_manage_core(rng)
            for confirmed in (False, True):
                facts = random_manage_facts(rng, confirmed=confirmed)
                text_free = manage(engine, holder, core, None, facts)
                with_text = manage(engine, holder, core, random_manage_text(rng), facts)
                assert with_text.exit_latch == text_free.exit_latch
                assert with_text.pressure_ppm == text_free.pressure_ppm


def test_a_text_only_close_is_impossible_over_a_long_unconfirmed_run() -> None:
    """7.7: whatever the text says, an unconfirmed reading only advances `watch_text` - it never places an order."""
    engine = DecisionRules(RulesConfig(), ALL_KINDS)
    pos = POSITIONS[StructureKind.PUT_CREDIT]
    facts = ManageFacts(
        pnl_headline=1_000,
        pnl_frac_loss_ppm=0,
        move_code="little_change",
        short_dist_code="far",
        news_count=5,
        news_recent_count=2,
    )
    calm_core = {
        "pos.thesis_invalidated": NoulAns(p=0.0),
        "pos.short_strike_threat": score([1.0, 0.0, 0.0, 0.0]),
        "pos.action": choice("pos.action", [0.9, 0.04, 0.03, 0.03]),
    }
    hostile = {qid: NoulAns(p=1.0) for qid in MANAGE_TEXT_IDS}
    for session in range(1, 41):
        decision = manage(engine, pos, calm_core, hostile, facts)
        assert decision.action == "hold" and decision.reason == "hold"
        assert decision.watch_text == session
        pos = pos.__class__(**{**{field: getattr(pos, field) for field in pos.__struct_fields__}, "watch_text": decision.watch_text})
