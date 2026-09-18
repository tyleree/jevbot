"""Guard: evaluation-only answers are invisible to `rules.py` (INV-23, DESIGN.md 0.2, 6.2, 15.5).

Two halves:

* **static** - the AST of `src/jevbot/rules.py` (and of `src/jevbot/candidates.py`, which sees no answers at all) never
  names an `eval.*` or `probe.*` question id, and every question id it does name lives in
  `vocab.RULES_READABLE_IDS = TRADING_IDS | TEXT_IDS | MANAGE_IDS | MANAGE_TEXT_IDS`;
* **behavioural** - a `DecisionResult` carrying arbitrary (including hostile) answers for every evaluation question
  produces byte-identical `EntryDecision` and `ManageDecision` values, and `rules.py` refuses to read one if asked.

The evaluation questions ride in both entry batches so that forecasts accumulate every session (0.1 item 5); nothing about
them may reach a gate, a composite, a tier or an exit.
"""

import ast
import math
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Final

import pytest

from jevbot import rules as rules_module
from jevbot import vocab
from jevbot.config import RulesConfig
from jevbot.errors import InvariantError
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

SRC: Final[Path] = Path(__file__).resolve().parents[2] / "src" / "jevbot"
GUARDED_MODULES: Final[tuple[str, ...]] = ("rules.py", "candidates.py")
FORBIDDEN_IDS: Final[frozenset[str]] = vocab.EVAL_IDS | vocab.PROBE_IDS | vocab.DERIVED_EVAL_IDS
DECISION_ID: Final = "d" * 24
POSITION_ID: Final = "p" * 24


def string_constants(source: str) -> list[str]:
    return [node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Constant) and isinstance(node.value, str)]


# ======================================================================================================================
# the vocabulary itself
# ======================================================================================================================


def test_the_evaluation_vocabulary_is_disjoint_from_what_the_rules_may_read() -> None:
    assert vocab.EVAL_IDS
    assert not (vocab.EVAL_IDS & vocab.RULES_READABLE_IDS)
    assert not (vocab.PROBE_IDS & vocab.RULES_READABLE_IDS)
    assert not (vocab.DERIVED_EVAL_IDS & vocab.RULES_READABLE_IDS)
    assert vocab.RULES_READABLE_IDS == (vocab.TRADING_IDS | vocab.TEXT_IDS | vocab.MANAGE_IDS | vocab.MANAGE_TEXT_IDS)
    assert vocab.EVAL_IDS <= vocab.ALL_QUESTION_IDS and vocab.RULES_READABLE_IDS <= vocab.ALL_QUESTION_IDS
    # the evaluation questions still ride in both entry batches (6.2): they exist, they are simply invisible here
    assert vocab.EVAL_IDS <= set(vocab.QUESTION_SETS["entry.v1"])
    assert vocab.EVAL_IDS <= set(vocab.QUESTION_SETS["entry_text.v1"])


# ======================================================================================================================
# static: the AST of the guarded modules
# ======================================================================================================================


@pytest.mark.parametrize("name", GUARDED_MODULES)
def test_no_evaluation_question_id_appears_in_the_source(name: str) -> None:
    path = SRC / name
    assert path.is_file(), path
    literals = string_constants(path.read_text(encoding="utf-8"))
    named = {text for text in literals if text in vocab.ALL_QUESTION_IDS}
    assert not (named & FORBIDDEN_IDS), sorted(named & FORBIDDEN_IDS)
    assert named <= vocab.RULES_READABLE_IDS, sorted(named - vocab.RULES_READABLE_IDS)
    # a prefix match catches a would-be f-string or a sliced id as well
    for text in literals:
        assert not text.startswith("eval."), f"{name} mentions {text!r}"
        assert not text.startswith("probe."), f"{name} mentions {text!r}"


def test_the_rules_module_names_every_trading_question_it_uses() -> None:
    """The counterpart of the guard: `rules.py` really does read the gate / veto / composite / manage questions."""
    named = {text for text in string_constants((SRC / "rules.py").read_text(encoding="utf-8")) if text in vocab.ALL_QUESTION_IDS}
    assert vocab.TRADING_IDS <= named
    assert vocab.TEXT_IDS <= named
    assert vocab.MANAGE_IDS <= named
    assert vocab.MANAGE_TEXT_IDS <= named
    assert named == vocab.RULES_READABLE_IDS
    assert not {text for text in string_constants((SRC / "candidates.py").read_text(encoding="utf-8")) if text in vocab.ALL_QUESTION_IDS}


def test_the_module_refuses_to_read_an_evaluation_answer_at_runtime() -> None:
    """`rules._answer` is the single read point; asking it for an evaluation id is an InvariantError, not a silent read."""
    answers: dict[str, Answer] = {qid: NoulAns(p=0.5) for qid in vocab.EVAL_ORDER}
    with pytest.raises(InvariantError, match="may not read"):
        rules_module._answer(result(answers), vocab.EVAL_ORDER[0])
    assert rules_module._READ_IDS <= vocab.RULES_READABLE_IDS
    assert rules_module._READ_IDS == vocab.RULES_READABLE_IDS


# ======================================================================================================================
# behavioural: garbage evaluation answers change nothing
# ======================================================================================================================


def choice(qid: str, top: str, weight: float) -> ChoiceAns:
    labels = vocab.CHOICE_LABELS[qid]
    rest = (1.0 - weight) / (len(labels) - 1)
    probs = {label: (weight if label == top else rest) for label in labels}
    ranked = sorted(probs.items(), key=lambda item: (-item[1], labels.index(item[0])))
    entropy = -sum(p * math.log(p) for p in probs.values() if p > 0.0) / math.log(len(labels))
    return ChoiceAns(
        probs=probs,
        top=ranked[0][0],
        p_top=ranked[0][1],
        margin=ranked[0][1] - ranked[1][1],
        entropy=entropy,
        raw_sum=1.0,
        server_choice=ranked[0][0],
        server_confidence=ranked[0][1],
    )


def score(probs: Sequence[float]) -> ScoreAns:
    values = tuple(float(p) / sum(probs) for p in probs)
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


def result(answers: Mapping[str, Answer], kind: RequestKind = RequestKind.ENTRY) -> DecisionResult:
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


def trading_answers() -> dict[str, Answer]:
    return {
        "regime.market": choice("regime.market", "trending_up_calm", 0.70),
        "under.direction": choice("under.direction", "bullish", 0.70),
        "under.stretched": NoulAns(p=0.10),
        "vol.stance": choice("vol.stance", "sell_premium", 0.70),
        "vol.explained_by_event": NoulAns(p=0.10),
        "fit.structure_family": choice("fit.structure_family", "put_credit_spread", 0.60),
        "risk.environment": score([0.8, 0.2, 0.0, 0.0]),
    }


def entry_facts() -> EntryFacts:
    return EntryFacts(
        trend_code="up",
        iv_rank_code="middle",
        iv_rv_code="iv_rich",
        dist_code="near_average",
        news_enabled=False,
        news_count=0,
        news_recent_count=0,
        spot=45_000,
        iv30_bp=1600,
        em_hold_tenths=35,
        events_in_window=1,
        thesis="trend up, iv_rich",
    )


EVAL_POISONS: Final[tuple[tuple[str, Any], ...]] = (
    ("all true", 1.0),
    ("all false", 0.0),
    ("all uncertain", 0.5),
)


@pytest.mark.parametrize(("label", "p"), EVAL_POISONS, ids=[case[0] for case in EVAL_POISONS])
def test_entry_decisions_ignore_every_evaluation_answer(label: str, p: float) -> None:
    engine = DecisionRules(RulesConfig(), tuple(StructureKind))
    clean = engine.decide_entry("SPY", DECISION_ID, result(trading_answers()), None, entry_facts())
    poisoned_answers = {**trading_answers(), **{qid: NoulAns(p=p) for qid in vocab.EVAL_ORDER}}
    poisoned = engine.decide_entry("SPY", DECISION_ID, result(poisoned_answers), None, entry_facts())
    assert poisoned == clean, label
    assert clean.action == "enter"  # the guard is meaningful only on a decision that could have been changed


def test_manage_decisions_ignore_every_evaluation_answer() -> None:
    engine = DecisionRules(RulesConfig(), tuple(StructureKind))
    chain = make_chain()
    position = Position(
        position_id=POSITION_ID,
        structure=make_structure(chain, StructureKind.PUT_CREDIT),
        qty=1,
        open_key=SnapshotKey(session=date(2024, 5, 17), slot=Slot.EOD),
        open_decision_id=DECISION_ID,
        open_net=BandPrices(orats=-100, worst=-95, mid=-102),
        max_loss=40_000,
        max_profit=10_000,
        bp_reserved=40_000,
        entry=EntryContext(
            entry_thesis="trend up, iv_rich",
            entry_codes={"trend": "up", "iv_vs_realized": "iv_rich", "iv_rank": "middle"},
            entry_spot=45_000,
            entry_iv30_bp=1600,
            entry_em_hold_tenths=35,
            open_mid_at_decision=-102,
        ),
    )
    facts = ManageFacts(
        pnl_headline=-2_000,
        pnl_frac_loss_ppm=100_000,
        move_code="little_change",
        short_dist_code="close",
        news_count=0,
        news_recent_count=0,
    )
    manage_answers: dict[str, Answer] = {
        "pos.thesis_invalidated": NoulAns(p=0.50),
        "pos.short_strike_threat": score([0.4, 0.3, 0.2, 0.1]),
        "pos.action": choice("pos.action", "hold", 0.70),
    }
    clean = engine.decide_manage(position, DECISION_ID, None, result(manage_answers, RequestKind.MANAGE), None, facts)
    poisoned_answers = {**manage_answers, **{qid: NoulAns(p=1.0) for qid in vocab.EVAL_ORDER}}
    poisoned = engine.decide_manage(position, DECISION_ID, None, result(poisoned_answers, RequestKind.MANAGE), None, facts)
    assert poisoned == clean
    assert clean.action in ("hold", "close") and clean.pressure_ppm == 500_000
