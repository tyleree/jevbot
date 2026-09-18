"""DecisionRules (DESIGN.md section 7, test plan 15.1 row `rules`).

Table-driven: every step of the 7.2 entry pipeline fails with its own code, the three-valued veto bands are pinned at
0.294 / 0.295 / 0.705 / 0.706, the composite and the tiers are checked against hand-computed numbers (never by re-running
the implementation's arithmetic), the 7.3 mapping is exhaustive, and the 7.7 management rules - hysteresis, the code
default, the discretionary fallback and the text rule that can never close alone - are driven through whole session
sequences.

The answer builders below construct `ChoiceAns` / `ScoreAns` / `NoulAns` exactly as `jev/stats.to_answers` (7.1) would:
renormalised probabilities in the question's AUTHORED option order, ties broken by that order, `raw_sum` carried
separately so that an out-of-band sum can be simulated without disturbing the probabilities.
"""

import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, Final

import pytest

from jevbot import vocab
from jevbot.config import Config, GateConfig, RulesConfig, RulesGates, RulesTiers, RulesWeights
from jevbot.errors import DeciderError, DeciderTransportError, InvariantError
from jevbot.rules import (
    ANOMALY_TEXT_VETO_ON_EMPTY_NEWS,
    DecisionRules,
    entry_anomalies,
    no_trade,
    text_watch_alert,
    tri_band,
)
from jevbot.types import (
    Answer,
    BandPrices,
    ChoiceAns,
    DecisionResult,
    Direction,
    EntryContext,
    EntryDecision,
    EntryFacts,
    ExitReason,
    ManageFacts,
    NoulAns,
    Position,
    RequestKind,
    ScoreAns,
    SnapshotKey,
    Slot,
    Structure,
    StructureKind,
    Tri,
    Variant,
    VolStance,
)
from tests.fixtures.chain_factory import make_chain, make_structure

UNDERLYING: Final = "SPY"
DECISION_ID: Final = "d" * 24
POSITION_ID: Final = "p" * 24

# 7.7 entry-side hysteresis: a close with one of these reasons blocks (underlying, direction) for `reentry_cooldown_sessions`
COOLDOWN_REASONS: Final[frozenset[str]] = frozenset(
    {
        ExitReason.STOP_LOSS.value,
        ExitReason.JEV.value,
        ExitReason.TEXT_CONFIRMED.value,
        ExitReason.CODE_DEFAULT.value,
        ExitReason.ASSIGNMENT_RISK.value,
        ExitReason.KILL.value,
    }
)
ALL_KINDS: Final[tuple[StructureKind, ...]] = tuple(StructureKind)


# ======================================================================================================================
# answer builders (7.1)
# ======================================================================================================================


def choice(qid: str, weights: Mapping[str, float], *, raw_sum: float = 1.0) -> ChoiceAns:
    labels = vocab.CHOICE_LABELS[qid]
    if set(weights) - set(labels):
        raise AssertionError(f"{qid}: unknown labels {sorted(set(weights) - set(labels))}")
    raw = {label: float(weights.get(label, 0.0)) for label in labels}
    total = sum(raw.values())
    probs = {label: value / total for label, value in raw.items()}
    ranked = sorted(probs.items(), key=lambda item: (-item[1], labels.index(item[0])))
    entropy = -sum(p * math.log(p) for p in probs.values() if p > 0.0) / math.log(len(labels))
    return ChoiceAns(
        probs=probs,
        top=ranked[0][0],
        p_top=ranked[0][1],
        margin=ranked[0][1] - ranked[1][1],
        entropy=entropy,
        raw_sum=raw_sum,
        server_choice=ranked[0][0],
        server_confidence=ranked[0][1],
    )


def score(weights: Sequence[float], *, raw_sum: float = 1.0) -> ScoreAns:
    total = sum(weights)
    probs = tuple(float(w) / total for w in weights)
    mean = sum(i * p for i, p in enumerate(probs))
    ranked = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
    entropy = -sum(p * math.log(p) for p in probs if p > 0.0) / math.log(len(probs))
    return ScoreAns(
        probs=probs,
        mean=mean,
        norm=mean / (len(probs) - 1),
        top=ranked[0],
        p_top=probs[ranked[0]],
        margin=probs[ranked[0]] - probs[ranked[1]],
        entropy=entropy,
        raw_sum=raw_sum,
        server_score=mean,
        server_confidence=probs[ranked[0]],
    )


def noul(p: float) -> NoulAns:
    return NoulAns(p=float(p))


def result(answers: Mapping[str, Answer], kind: RequestKind = RequestKind.ENTRY, variant: Variant = Variant.BASE) -> DecisionResult:
    return DecisionResult(
        decision_id=DECISION_ID,
        kind=kind,
        variant=variant,
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
# the PASSING baseline: a bullish / sell-premium set that maps to put_credit_spread and clears every step of 7.2
# ======================================================================================================================

REGIME_PASS: Final[Mapping[str, float]] = {
    "trending_up_calm": 0.60,
    "trending_up_volatile": 0.20,
    "range_bound_calm": 0.05,
    "range_bound_volatile": 0.05,
    "orderly_downtrend": 0.04,
    "disorderly_selloff": 0.03,
    "unclear_or_transition": 0.03,
}
DIRECTION_PASS: Final[Mapping[str, float]] = {"bullish": 0.70, "bearish": 0.05, "neutral_range": 0.15, "conflicting_signals": 0.10}
STANCE_PASS: Final[Mapping[str, float]] = {"sell_premium": 0.70, "buy_premium": 0.10, "limit_vol_exposure": 0.10, "unclear": 0.10}
FIT_PASS: Final[Mapping[str, float]] = {
    "put_credit_spread": 0.60,
    "call_debit_spread": 0.10,
    "long_call": 0.10,
    "iron_condor": 0.05,
    "call_credit_spread": 0.05,
    "put_debit_spread": 0.04,
    "long_put": 0.03,
    "no_trade": 0.03,
}


# a set that clears every gate but scores below `rules.min_score`: regimefit 0.45, fit 0.50, calm 0.0
WEAK_REGIME: Final[Mapping[str, float]] = {
    "trending_up_calm": 0.40,
    "trending_up_volatile": 0.05,
    "range_bound_calm": 0.35,
    "range_bound_volatile": 0.10,
    "orderly_downtrend": 0.05,
    "disorderly_selloff": 0.03,
    "unclear_or_transition": 0.02,
}
WEAK_FIT: Final[Mapping[str, float]] = {
    "put_credit_spread": 0.50,
    "iron_condor": 0.30,
    "call_debit_spread": 0.04,
    "long_call": 0.04,
    "call_credit_spread": 0.04,
    "put_debit_spread": 0.04,
    "long_put": 0.02,
    "no_trade": 0.02,
}


def core_answers(**overrides: Answer) -> dict[str, Answer]:
    """The passing text-free entry answers; any question can be replaced by keyword (dots written as underscores)."""
    answers: dict[str, Answer] = {
        "regime.market": choice("regime.market", REGIME_PASS),
        "under.direction": choice("under.direction", DIRECTION_PASS),
        "under.stretched": noul(0.10),
        "vol.stance": choice("vol.stance", STANCE_PASS),
        "vol.explained_by_event": noul(0.10),
        "fit.structure_family": choice("fit.structure_family", FIT_PASS),
        "risk.environment": score((0.80, 0.20, 0.0, 0.0)),
    }
    for key, value in overrides.items():
        answers[key.replace("__", ".")] = value
    return answers


def text_answers(**overrides: Answer) -> dict[str, Answer]:
    answers: dict[str, Answer] = {
        "text.material_present": noul(0.10),
        "text.clearly_negative": noul(0.10),
        "text.clearly_positive": noul(0.10),
        "text.pending_binary": noul(0.10),
        "text.market_stress": noul(0.10),
    }
    for key, value in overrides.items():
        answers[key.replace("__", ".")] = value
    return answers


def facts(**overrides: Any) -> EntryFacts:
    values: dict[str, Any] = {
        "trend_code": "up",
        "iv_rank_code": "middle",
        "iv_rv_code": "iv_rich",
        "dist_code": "near_average",
        "news_enabled": False,
        "news_count": 0,
        "news_recent_count": 0,
        "spot": 45_000,
        "iv30_bp": 1600,
        "em_hold_tenths": 35,
        "events_in_window": 1,
        "thesis": "trend up, iv_rich",
    }
    values.update(overrides)
    return EntryFacts(**values)


def rules(**overrides: Any) -> DecisionRules:
    enabled = overrides.pop("enabled", ALL_KINDS)
    return DecisionRules(RulesConfig(**overrides), enabled)


def decide(engine: DecisionRules, core: Mapping[str, Answer], f: EntryFacts, text: Mapping[str, Answer] | None = None) -> EntryDecision:
    text_result = None if text is None else result(text, kind=RequestKind.ENTRY_TEXT)
    return engine.decide_entry(UNDERLYING, DECISION_ID, result(core), text_result, f)


# ======================================================================================================================
# the baseline itself: the composite, the tier and the rank term against hand-computed numbers
# ======================================================================================================================


def test_the_passing_baseline_enters_with_hand_computed_score_and_tier() -> None:
    decision = decide(rules(), core_answers(), facts())
    assert decision.action == "enter" and decision.reasons == ()
    assert decision.kind is StructureKind.PUT_CREDIT
    # 7.5, by hand: align .70, volfit .70, fit .60, regimefit .60 + .20 = .80, calm 1 - .10 = .90
    #   S_core = 0.30*0.70 + 0.20*0.70 + 0.20*0.60 + 0.15*0.80 + 0.15*0.90 = 0.21 + 0.14 + 0.12 + 0.12 + 0.135 = 0.725
    assert decision.features_ppm["align"] == 700_000
    assert decision.features_ppm["volfit"] == 700_000
    assert decision.features_ppm["fit"] == 600_000
    assert decision.features_ppm["regimefit"] == 800_000
    assert decision.features_ppm["calm"] == 900_000
    assert decision.score_core_ppm == 725_000
    # no text => tone 0 => news_align 0.5; S_rank = 0.95 * 725000 + 0.05 * 500000 = 688750 + 25000
    assert decision.features_ppm["news_align"] == 500_000
    assert decision.score_rank_ppm == 713_750
    # 7.6, by hand: tier_S = 0.75 (0.725 >= 0.655, < 0.755); tier_peak = 0.5 (min(.70,.70) >= 0.595, < 0.705);
    #               env mean = 0.2 -> level max(0, 0) = 0 -> tier_env = 1.0; tier = min = 0.5
    assert decision.tier_ppm == 500_000
    assert decision.variant_agreement == {}
    assert decision.underlying == UNDERLYING and decision.decision_id == DECISION_ID


def test_decision_reasons_never_carry_a_post_decision_code() -> None:
    """7.9: `EntryDecision.reasons` is a pure function of answers + facts; candidate / risk / gate codes live in RISK_VERDICT."""
    cases = [
        decide(rules(), core_answers(), facts()),
        decide(rules(), core_answers(regime__market=choice("regime.market", {"disorderly_selloff": 0.9, "unclear_or_transition": 0.1})), facts()),
        decide(rules(), core_answers(), facts(trend_code="down")),
        decide(rules(), core_answers(risk__environment=score((0.0, 0.0, 0.0, 1.0))), facts()),
    ]
    for decision in cases:
        for code in decision.reasons:
            assert vocab.is_reason(code), code
            assert not code.startswith(("candidate:", "risk:"))
            assert code not in vocab.GATE_CODES
    assert all(not vocab.is_reason(code) for code in vocab.GATE_CODES)


# ======================================================================================================================
# 7.2 - each step fails with its own code
# ======================================================================================================================


def _no_trade_steps() -> list[tuple[str, dict[str, Any], dict[str, Any], str]]:
    """(name, core overrides, facts overrides, the FIRST expected reason code)."""
    return [
        (
            "step2 regime disorderly_selloff",
            {"regime__market": choice("regime.market", {"disorderly_selloff": 0.70, "orderly_downtrend": 0.20, "unclear_or_transition": 0.10})},
            {},
            "veto:regime:disorderly_selloff",
        ),
        (
            "step2 regime unclear",
            {"regime__market": choice("regime.market", {"unclear_or_transition": 0.70, "trending_up_calm": 0.20, "range_bound_calm": 0.10})},
            {},
            "veto:regime:unclear_or_transition",
        ),
        (
            "step2 regime raw_sum out of band reads as unclear",
            {"regime__market": choice("regime.market", REGIME_PASS, raw_sum=0.90)},
            {},
            "veto:regime:unclear_or_transition",
        ),
        (
            "step3 direction conflicting",
            {"under__direction": choice("under.direction", {"conflicting_signals": 0.70, "bullish": 0.20, "bearish": 0.05, "neutral_range": 0.05})},
            {},
            "gate:direction:conflicting",
        ),
        (
            "step3 direction p_top",
            {"under__direction": choice("under.direction", {"bullish": 0.59, "bearish": 0.01, "neutral_range": 0.20, "conflicting_signals": 0.20})},
            {},
            "gate:direction:p_top",
        ),
        (
            "step3 direction margin",
            {"under__direction": choice("under.direction", {"bullish": 0.60, "bearish": 0.40, "neutral_range": 0.0, "conflicting_signals": 0.0})},
            {},
            "gate:direction:margin",
        ),
        (
            "step4 vol_stance unclear",
            {"vol__stance": choice("vol.stance", {"unclear": 0.70, "sell_premium": 0.20, "buy_premium": 0.05, "limit_vol_exposure": 0.05})},
            {},
            "gate:vol_stance:unclear",
        ),
        (
            "step4 vol_stance p_top",
            {"vol__stance": choice("vol.stance", {"sell_premium": 0.59, "buy_premium": 0.01, "limit_vol_exposure": 0.20, "unclear": 0.20})},
            {},
            "gate:vol_stance:p_top",
        ),
        (
            "step4 vol_stance margin",
            {"vol__stance": choice("vol.stance", {"sell_premium": 0.60, "buy_premium": 0.40, "limit_vol_exposure": 0.0, "unclear": 0.0})},
            {},
            "gate:vol_stance:margin",
        ),
        (
            "step5 no structure (neutral_range + buy_premium)",
            {
                "under__direction": choice("under.direction", {"neutral_range": 0.70, "bullish": 0.10, "bearish": 0.10, "conflicting_signals": 0.10}),
                "vol__stance": choice("vol.stance", {"buy_premium": 0.70, "sell_premium": 0.10, "limit_vol_exposure": 0.10, "unclear": 0.10}),
            },
            {},
            "map:no_structure",
        ),
        ("step6 crosscheck trend_not_opposed", {}, {"trend_code": "down"}, "crosscheck:trend_not_opposed"),
        ("step6 crosscheck sell_needs_rich", {}, {"iv_rv_code": "iv_fair"}, "crosscheck:sell_needs_rich"),
        (
            "step7 fit no_trade",
            {"fit__structure_family": choice("fit.structure_family", {"no_trade": 0.60, "put_credit_spread": 0.30, "long_call": 0.10})},
            {},
            "fit:no_trade",
        ),
        (
            "step7 fit disagrees",
            {"fit__structure_family": choice("fit.structure_family", {"iron_condor": 0.60, "put_credit_spread": 0.30, "long_call": 0.10})},
            {},
            "fit:disagrees_with_mapping",
        ),
        (
            "step7 fit p_top",
            {"fit__structure_family": choice("fit.structure_family", {"put_credit_spread": 0.49, "long_call": 0.21, "iron_condor": 0.30})},
            {},
            "fit:p_top",
        ),
        (
            "step7 fit margin",
            {"fit__structure_family": choice("fit.structure_family", {"put_credit_spread": 0.51, "iron_condor": 0.35, "long_call": 0.14})},
            {},
            "fit:margin",
        ),
        ("step8 veto vol.explained_by_event hard", {"vol__explained_by_event": noul(0.706)}, {}, "veto:vol.explained_by_event:hard"),
        ("step8 veto vol.explained_by_event uncertain", {"vol__explained_by_event": noul(0.50)}, {}, "veto:vol.explained_by_event:uncertain"),
        (
            "step9 score below min",
            {
                "regime__market": choice("regime.market", WEAK_REGIME),
                "fit__structure_family": choice("fit.structure_family", WEAK_FIT),
                "under__stretched": noul(1.0),
            },
            {},
            "score:below_min",
        ),
        ("step10 tier zero env", {"risk__environment": score((0.0, 0.0, 0.0, 1.0))}, {}, "tier:zero:env"),
    ]


@pytest.mark.parametrize(("name", "core_kw", "facts_kw", "expected"), _no_trade_steps(), ids=[c[0] for c in _no_trade_steps()])
def test_each_step_of_the_entry_pipeline_fails_with_its_code(name: str, core_kw: dict[str, Any], facts_kw: dict[str, Any], expected: str) -> None:
    decision = decide(rules(), core_answers(**core_kw), facts(**facts_kw))
    assert decision.action == "no_trade", name
    assert decision.reasons[0] == expected, f"{name}: {decision.reasons}"
    assert all(vocab.is_reason(code) for code in decision.reasons), decision.reasons


def test_map_disabled_when_the_mapped_structure_is_not_enabled() -> None:
    engine = rules(enabled=(StructureKind.IRON_CONDOR,))
    decision = decide(engine, core_answers(), facts())
    assert decision.action == "no_trade" and "map:disabled" in decision.reasons
    assert "map:no_structure" not in decision.reasons and decision.kind is StructureKind.PUT_CREDIT


def test_later_steps_are_still_evaluated_and_logged_in_order() -> None:
    """7.2: the FIRST failing step is the funnel's reason, but every later check that can run is recorded too."""
    decision = decide(
        rules(),
        core_answers(
            regime__market=choice("regime.market", {"unclear_or_transition": 0.50, "trending_up_calm": 0.30, "trending_up_volatile": 0.20}),
            under__stretched=noul(0.99),
        ),
        facts(iv_rv_code="iv_cheap"),
    )
    assert decision.reasons[0] == "veto:regime:unclear_or_transition"
    assert list(decision.reasons) == [
        "veto:regime:unclear_or_transition",
        "crosscheck:sell_needs_rich",
        "score:below_min",
        "tier:zero:score",
    ]


def test_code_crosschecks_can_be_switched_off_for_the_always_enter_baseline() -> None:
    assert "crosscheck:trend_not_opposed" in decide(rules(), core_answers(), facts(trend_code="down")).reasons
    assert decide(rules(code_crosschecks=False), core_answers(), facts(trend_code="down")).action == "enter"


@pytest.mark.parametrize(
    ("kind", "bad_facts", "expected"),
    [
        (StructureKind.PUT_CREDIT, {"trend_code": "down"}, "crosscheck:trend_not_opposed"),
        (StructureKind.CALL_CREDIT, {"trend_code": "up"}, "crosscheck:trend_not_opposed"),
        (StructureKind.IRON_CONDOR, {"trend_code": "up"}, "crosscheck:range_needs_no_trend"),
        (StructureKind.IRON_CONDOR, {"trend_code": "flat", "iv_rv_code": "iv_cheap"}, "crosscheck:sell_needs_rich"),
        (StructureKind.LONG_CALL, {"iv_rank_code": "high"}, "crosscheck:long_single_needs_cheap"),
        (StructureKind.LONG_PUT, {"iv_rank_code": "middle"}, "crosscheck:long_single_needs_cheap"),
    ],
)
def test_the_four_code_side_crosschecks(kind: StructureKind, bad_facts: dict[str, Any], expected: str) -> None:
    core = core_answers(**_mapping_overrides(kind))
    decision = decide(rules(), core, facts(**bad_facts))
    assert decision.kind is kind
    assert expected in decision.reasons


def test_crosschecks_that_pass_leave_no_reason() -> None:
    good = {
        StructureKind.PUT_CREDIT: {"trend_code": "up", "iv_rv_code": "iv_rich"},
        StructureKind.CALL_CREDIT: {"trend_code": "down", "iv_rv_code": "iv_very_rich"},
        StructureKind.IRON_CONDOR: {"trend_code": "flat", "iv_rv_code": "iv_rich"},
        StructureKind.LONG_CALL: {"trend_code": "up", "iv_rank_code": "low"},
        StructureKind.LONG_PUT: {"trend_code": "down", "iv_rank_code": "very_low"},
        StructureKind.CALL_DEBIT: {"trend_code": "up"},
        StructureKind.PUT_DEBIT: {"trend_code": "down"},
    }
    for kind, facts_kw in good.items():
        decision = decide(rules(), core_answers(**_mapping_overrides(kind)), facts(**facts_kw))
        assert decision.kind is kind
        assert not [code for code in decision.reasons if code.startswith("crosscheck:")], (kind, decision.reasons)


# ======================================================================================================================
# 7.3 mapping, exhaustively
# ======================================================================================================================


def _mapping_overrides(kind: StructureKind) -> dict[str, Answer]:
    """Answers that map to `kind` and keep the fit question in agreement."""
    from jevbot.types import STRUCTURE_DIRECTION, STRUCTURE_STANCE

    direction = STRUCTURE_DIRECTION[kind]
    stance = STRUCTURE_STANCE[kind]
    return {
        "under__direction": choice("under.direction", _peaked(vocab.CHOICE_LABELS["under.direction"], direction.value, 0.70)),
        "vol__stance": choice("vol.stance", _peaked(vocab.CHOICE_LABELS["vol.stance"], stance.value, 0.70)),
        "fit__structure_family": choice("fit.structure_family", _peaked(vocab.CHOICE_LABELS["fit.structure_family"], kind.value, 0.60)),
    }


def _peaked(labels: Sequence[str], top: str, weight: float) -> dict[str, float]:
    rest = (1.0 - weight) / (len(labels) - 1)
    return {label: (weight if label == top else rest) for label in labels}


@pytest.mark.parametrize("direction", list(Direction))
@pytest.mark.parametrize("stance", list(VolStance))
def test_the_deterministic_mapping_is_exhaustive(direction: Direction, stance: VolStance) -> None:
    expected = {
        (Direction.BULLISH, VolStance.SELL): StructureKind.PUT_CREDIT,
        (Direction.BULLISH, VolStance.BUY): StructureKind.LONG_CALL,
        (Direction.BULLISH, VolStance.LIMIT): StructureKind.CALL_DEBIT,
        (Direction.BEARISH, VolStance.SELL): StructureKind.CALL_CREDIT,
        (Direction.BEARISH, VolStance.BUY): StructureKind.LONG_PUT,
        (Direction.BEARISH, VolStance.LIMIT): StructureKind.PUT_DEBIT,
        (Direction.NEUTRAL, VolStance.SELL): StructureKind.IRON_CONDOR,
        (Direction.NEUTRAL, VolStance.BUY): None,
        (Direction.NEUTRAL, VolStance.LIMIT): None,
    }[(direction, stance)]
    core = core_answers(
        under__direction=choice("under.direction", _peaked(vocab.CHOICE_LABELS["under.direction"], direction.value, 0.70)),
        vol__stance=choice("vol.stance", _peaked(vocab.CHOICE_LABELS["vol.stance"], stance.value, 0.70)),
        fit__structure_family=choice(
            "fit.structure_family",
            _peaked(vocab.CHOICE_LABELS["fit.structure_family"], "no_trade" if expected is None else expected.value, 0.60),
        ),
    )
    decision = decide(rules(), core, facts(trend_code="flat", iv_rank_code="very_low", iv_rv_code="iv_rich"))
    assert decision.kind is expected
    if expected is None:
        assert "map:no_structure" in decision.reasons


# ======================================================================================================================
# 7.4 - veto bands and text-veto skipping
# ======================================================================================================================


@pytest.mark.parametrize(
    ("p", "band"),
    [(0.0, Tri.CLEAR), (0.294, Tri.CLEAR), (0.295, Tri.UNCERTAIN), (0.5, Tri.UNCERTAIN), (0.705, Tri.UNCERTAIN), (0.706, Tri.VETO), (1.0, Tri.VETO)],
)
def test_veto_bands_at_their_edges(p: float, band: Tri) -> None:
    cfg = RulesConfig()
    assert tri_band(p, cfg) is band
    decision = decide(rules(), core_answers(vol__explained_by_event=noul(p)), facts())
    codes = [code for code in decision.reasons if code.startswith("veto:vol.explained_by_event")]
    assert codes == {Tri.CLEAR: [], Tri.UNCERTAIN: ["veto:vol.explained_by_event:uncertain"], Tri.VETO: ["veto:vol.explained_by_event:hard"]}[band]


def test_the_event_veto_applies_to_short_premium_only() -> None:
    long_call = core_answers(vol__explained_by_event=noul(0.99), **_mapping_overrides(StructureKind.LONG_CALL))
    decision = decide(rules(), long_call, facts(iv_rank_code="low"))
    assert decision.action == "enter" and decision.kind is StructureKind.LONG_CALL


def test_text_vetoes_are_skipped_when_the_text_result_is_absent_by_design() -> None:
    hostile = text_answers(text__market_stress=noul(0.99), text__clearly_negative=noul(0.99), text__pending_binary=noul(0.99))
    with_text = decide(rules(), core_answers(), facts(news_enabled=True, news_count=3, news_recent_count=2), hostile)
    assert with_text.action == "no_trade"
    assert "veto:text.market_stress:hard" in with_text.reasons and "veto:text.clearly_negative:hard" in with_text.reasons
    assert "veto:text.pending_binary:hard" in with_text.reasons
    without = decide(rules(), core_answers(), facts(), None)
    assert without.action == "enter"


def test_text_vetoes_are_skipped_on_an_empty_news_list_and_the_anomaly_is_counted() -> None:
    """7.4 / G9: `news_count == 0` is known exactly in code, so the vetoes are skipped; a non-CLEAR reading is an ANOMALY."""
    hostile = text_answers(text__market_stress=noul(0.99), text__clearly_negative=noul(0.80))
    empty = facts(news_enabled=True, news_count=0, news_recent_count=0)
    decision = decide(rules(), core_answers(), empty, hostile)
    assert decision.action == "enter" and not [c for c in decision.reasons if c.startswith("veto:text")]
    anomalies = entry_anomalies(result(hostile, kind=RequestKind.ENTRY_TEXT), empty)
    assert anomalies == (ANOMALY_TEXT_VETO_ON_EMPTY_NEWS, ANOMALY_TEXT_VETO_ON_EMPTY_NEWS)
    assert ANOMALY_TEXT_VETO_ON_EMPTY_NEWS in vocab.KNOWN_ANOMALY_TYPES
    assert rules().anomalies(result(hostile, kind=RequestKind.ENTRY_TEXT), empty) == anomalies
    assert entry_anomalies(None, empty) == ()
    assert entry_anomalies(result(hostile, kind=RequestKind.ENTRY_TEXT), facts(news_count=2)) == ()
    assert entry_anomalies(result(text_answers(), kind=RequestKind.ENTRY_TEXT), empty) == ()


def test_pending_binary_is_skipped_when_no_news_arrived_since_the_previous_session() -> None:
    hostile = text_answers(text__pending_binary=noul(0.99))
    stale_only = decide(rules(), core_answers(), facts(news_enabled=True, news_count=4, news_recent_count=0), hostile)
    assert stale_only.action == "enter"
    fresh = decide(rules(), core_answers(), facts(news_enabled=True, news_count=4, news_recent_count=1), hostile)
    assert fresh.action == "no_trade" and "veto:text.pending_binary:hard" in fresh.reasons


@pytest.mark.parametrize(
    ("kind", "qid", "applies"),
    [
        (StructureKind.PUT_CREDIT, "text.clearly_negative", True),
        (StructureKind.PUT_CREDIT, "text.clearly_positive", False),
        (StructureKind.CALL_CREDIT, "text.clearly_positive", True),
        (StructureKind.CALL_CREDIT, "text.clearly_negative", False),
        (StructureKind.IRON_CONDOR, "text.clearly_negative", True),
        (StructureKind.IRON_CONDOR, "text.clearly_positive", True),
    ],
)
def test_directional_text_vetoes_apply_to_their_own_side(kind: StructureKind, qid: str, applies: bool) -> None:
    text = text_answers(**{qid.replace(".", "__"): noul(0.99)})
    good_facts = facts(news_enabled=True, news_count=2, news_recent_count=0, trend_code="flat", iv_rv_code="iv_rich")
    decision = decide(rules(), core_answers(**_mapping_overrides(kind)), good_facts, text)
    assert (f"veto:{qid}:hard" in decision.reasons) is applies


# ======================================================================================================================
# 7.5 / 7.6 - the rank term and the tiers
# ======================================================================================================================


def test_the_text_rank_term_orders_but_never_scores() -> None:
    """7.5: `tone` enters `S_rank` only; `S_core`, the floor and every tier stay text-free (INV-16)."""
    good_news = text_answers(text__material_present=noul(0.90), text__clearly_positive=noul(0.80), text__clearly_negative=noul(0.20))
    f = facts(news_enabled=True, news_count=3, news_recent_count=0)
    decision = decide(rules(), core_answers(), f, good_news)
    # tone = 0.80 - 0.20 = 0.60; bullish => news_align = 0.5 + 0.5*0.60 = 0.80
    assert decision.features_ppm["news_align"] == 800_000
    assert decision.score_core_ppm == 725_000 and decision.tier_ppm == 500_000
    # S_rank = 0.95 * 725000 + 0.05 * 800000 = 688750 + 40000
    assert decision.score_rank_ppm == 728_750
    immaterial = text_answers(text__material_present=noul(0.49), text__clearly_positive=noul(0.80), text__clearly_negative=noul(0.20))
    assert decide(rules(), core_answers(), f, immaterial).features_ppm["news_align"] == 500_000


def test_news_align_for_bearish_and_neutral_structures() -> None:
    text = text_answers(text__material_present=noul(0.90), text__clearly_positive=noul(0.70), text__clearly_negative=noul(0.10))
    f = facts(news_enabled=True, news_count=3, news_recent_count=0, trend_code="down", iv_rv_code="iv_rich")
    bearish = decide(rules(), core_answers(**_mapping_overrides(StructureKind.CALL_CREDIT)), f, text)
    # tone = 0.60; bearish => 0.5 - 0.5*0.60 = 0.20
    assert bearish.features_ppm["news_align"] == 200_000
    neutral = decide(
        rules(),
        core_answers(**_mapping_overrides(StructureKind.IRON_CONDOR)),
        facts(news_enabled=True, news_count=3, news_recent_count=0, trend_code="flat", iv_rv_code="iv_rich"),
        text,
    )
    # neutral_range => 1 - abs(tone) = 0.40
    assert neutral.features_ppm["news_align"] == 400_000


@pytest.mark.parametrize(
    ("stretched", "expected_core", "expected_tier"),
    [
        (0.10, 725_000, 500_000),  # tier_S 0.75, tier_peak 0.5
        (0.30, 695_000, 500_000),
        (0.70, 635_000, 500_000),  # tier_S 0.5
        (0.95, 597_500, 500_000),
    ],
)
def test_score_tier_lookup(stretched: float, expected_core: int, expected_tier: int) -> None:
    decision = decide(rules(), core_answers(under__stretched=noul(stretched)), facts())
    assert decision.score_core_ppm == expected_core
    assert decision.tier_ppm == expected_tier


def sharp_core(**overrides: Answer) -> dict[str, Answer]:
    """A set whose score tier and peakedness tier are both 1.0, so the environment tier alone decides `tier_ppm`.

    S_core = .30*.85 + .20*.82 + .20*.90 + .15*(.90 + 0.1/6) + .15*1.0 = .255 + .164 + .18 + .1375 + .15 = 0.8865
    """
    base: dict[str, Answer] = {
        "under__direction": choice("under.direction", {"bullish": 0.85, "bearish": 0.05, "neutral_range": 0.05, "conflicting_signals": 0.05}),
        "vol__stance": choice("vol.stance", {"sell_premium": 0.82, "buy_premium": 0.06, "limit_vol_exposure": 0.06, "unclear": 0.06}),
        "under__stretched": noul(0.0),
        "fit__structure_family": choice("fit.structure_family", _peaked(vocab.CHOICE_LABELS["fit.structure_family"], "put_credit_spread", 0.90)),
        "regime__market": choice("regime.market", _peaked(vocab.CHOICE_LABELS["regime.market"], "trending_up_calm", 0.90)),
    }
    base.update(overrides)
    return core_answers(**base)


def test_peakedness_tier_is_the_minimum_of_the_two_gate_peaks() -> None:
    decision = decide(rules(), sharp_core(), facts())
    assert decision.score_core_ppm == 886_500
    assert decision.tier_ppm == 1_000_000  # tier_S 1.0, min(p_top) = 0.82 >= 0.805 -> tier_peak 1.0, env level 0 -> 1.0
    softer = sharp_core(
        vol__stance=choice("vol.stance", {"sell_premium": 0.80, "buy_premium": 0.08, "limit_vol_exposure": 0.06, "unclear": 0.06})
    )
    assert decide(rules(), softer, facts()).tier_ppm == 750_000  # min(p_top) = 0.80 -> tier_peak 0.75
    blunter = sharp_core(
        under__direction=choice("under.direction", {"bullish": 0.70, "bearish": 0.05, "neutral_range": 0.15, "conflicting_signals": 0.10})
    )
    assert decide(rules(), blunter, facts()).tier_ppm == 500_000  # min(p_top) = 0.70 -> tier_peak 0.5


@pytest.mark.parametrize(
    ("weights", "expected_level", "expected_tier_ppm"),
    [
        ((1.0, 0.0, 0.0, 0.0), 0, 1_000_000),  # mean 0, top 0 -> level 0 -> tier_env 1.0
        ((0.0, 1.0, 0.0, 0.0), 1, 750_000),
        ((0.0, 0.0, 1.0, 0.0), 2, 500_000),
        ((0.0, 0.0, 0.0, 1.0), 3, 0),
        ((0.45, 0.0, 0.0, 0.55), 3, 0),  # mean 1.65 -> round_half_up 2, top 3 -> max = 3: the TOP is the conservative one
        ((0.0, 0.34, 0.33, 0.33), 2, 500_000),  # mean 1.99 -> 2, top 1 -> max = 2: here the MEAN is
    ],
)
def test_environment_tier_is_conservative(weights: tuple[float, ...], expected_level: int, expected_tier_ppm: int) -> None:
    """7.6: `env_level = max(round_half_up(mean), top)`, so neither a hostile mean nor a hostile mode can be averaged away."""
    decision = decide(rules(), sharp_core(risk__environment=score(weights)), facts())
    assert decision.tier_ppm == expected_tier_ppm
    assert decision.tier_ppm == round(RulesTiers().environment[expected_level] * 1_000_000)
    if expected_tier_ppm == 0:
        assert "tier:zero:env" in decision.reasons and decision.action == "no_trade"
    else:
        assert decision.action == "enter"


def test_an_out_of_band_environment_score_reads_as_the_hostile_level() -> None:
    decision = decide(rules(), sharp_core(risk__environment=score((1.0, 0.0, 0.0, 0.0), raw_sum=0.90)), facts())
    assert decision.tier_ppm == 0 and "tier:zero:env" in decision.reasons


def test_rank_orders_by_rank_score_then_by_universe_order() -> None:
    def entry(underlying: str, rank_ppm: int) -> EntryDecision:
        return EntryDecision(
            underlying=underlying,
            decision_id=DECISION_ID,
            action="enter",
            kind=StructureKind.PUT_CREDIT,
            score_core_ppm=700_000,
            score_rank_ppm=rank_ppm,
            tier_ppm=500_000,
            reasons=(),
            features_ppm={},
            variant_agreement={},
        )

    entries = [entry("IWM", 700_000), entry("SPY", 700_000), entry("QQQ", 900_000), entry("XLE", 700_000)]
    ordered = rules().rank(entries, ("SPY", "QQQ", "IWM"))
    assert [e.underlying for e in ordered] == ["QQQ", "SPY", "IWM", "XLE"]


# ======================================================================================================================
# 7.8 - perturbation agreement
# ======================================================================================================================


def test_confirm_entry_requires_every_variant_to_agree_and_takes_the_minimum() -> None:
    engine = rules()
    base = decide(engine, core_answers(), facts())
    assert base.action == "enter"
    weaker = core_answers(under__stretched=noul(0.40))  # calm 0.60 -> S_core 0.725 - 0.15*0.30 = 0.68
    variants = {
        Variant.OPT_PERM: result(core_answers(), variant=Variant.OPT_PERM),
        Variant.KEY_PERM: result(weaker, variant=Variant.KEY_PERM),
        Variant.BUCKET_ONLY: result(core_answers(), variant=Variant.BUCKET_ONLY),
    }
    confirmed = engine.confirm_entry(base, variants, None, facts())
    assert confirmed.action == "enter" and confirmed.reasons == ()
    assert confirmed.variant_agreement == {"opt_perm": True, "key_perm": True, "bucket_only": True}
    assert confirmed.score_core_ppm == 680_000  # min over base + variants
    assert confirmed.tier_ppm == 500_000


def test_a_disagreeing_variant_blocks_the_entry() -> None:
    engine = rules()
    base = decide(engine, core_answers(), facts())
    flipped = core_answers(
        under__direction=choice("under.direction", {"bearish": 0.70, "bullish": 0.05, "neutral_range": 0.15, "conflicting_signals": 0.10})
    )
    confirmed = engine.confirm_entry(base, {Variant.OPT_PERM: result(flipped, variant=Variant.OPT_PERM)}, None, facts())
    assert confirmed.action == "no_trade"
    assert confirmed.variant_agreement == {"opt_perm": False}
    assert confirmed.reasons and confirmed.reasons[0].startswith("perturb:disagree:opt_perm:")
    assert all(vocab.is_reason(code) for code in confirmed.reasons)


def test_an_errored_variant_counts_as_disagreement() -> None:
    engine = rules()
    base = decide(engine, core_answers(), facts())
    confirmed = engine.confirm_entry(base, {Variant.KEY_PERM: DeciderTransportError("boom")}, None, facts())
    assert confirmed.action == "no_trade"
    assert confirmed.reasons == ("perturb:decider_failed",)
    assert confirmed.variant_agreement == {"key_perm": False}
    assert isinstance(DeciderTransportError("x"), DeciderError)


def test_confirm_entry_leaves_a_no_trade_base_untouched() -> None:
    engine = rules()
    base = decide(engine, core_answers(regime__market=choice("regime.market", {"disorderly_selloff": 0.8, "trending_up_calm": 0.2})), facts())
    assert base.action == "no_trade"
    assert engine.confirm_entry(base, {Variant.OPT_PERM: result(core_answers(), variant=Variant.OPT_PERM)}, None, facts()) is base


# ======================================================================================================================
# helper constructors and the guard rails of the module surface
# ======================================================================================================================


def test_no_trade_helper_validates_its_reason_codes() -> None:
    decision = no_trade(UNDERLYING, DECISION_ID, "decider_failed_cycle:DeciderError")
    assert decision.action == "no_trade" and decision.kind is None
    assert decision.reasons == ("decider_failed_cycle:DeciderError",)
    assert decision.score_core_ppm == 0 and decision.score_rank_ppm == 0 and decision.tier_ppm == 0
    assert no_trade(UNDERLYING, DECISION_ID, "dq:insufficient").reasons == ("dq:insufficient",)
    assert no_trade(UNDERLYING, DECISION_ID, "news_pipeline_error", "state_rejected").reasons == ("news_pipeline_error", "state_rejected")
    with pytest.raises(InvariantError):
        no_trade(UNDERLYING, DECISION_ID)
    with pytest.raises(InvariantError):
        no_trade(UNDERLYING, DECISION_ID, "risk:size_zero")


def test_a_missing_or_mistyped_answer_is_an_invariant_error() -> None:
    engine = rules()
    partial = core_answers()
    del partial["vol.stance"]
    with pytest.raises(InvariantError, match="vol.stance"):
        decide(engine, partial, facts())
    wrong_type = core_answers(vol__stance=noul(0.5))
    with pytest.raises(InvariantError, match="Choice"):
        decide(engine, wrong_type, facts())


def test_the_rules_hash_follows_the_rules_section() -> None:
    from jevbot import config as config_module

    default = rules()
    assert default.rules_hash == config_module.rules_hash(Config())
    other = rules(min_score=0.6)
    assert other.rules_hash != default.rules_hash


def test_evaluation_answers_are_invisible_to_the_rules() -> None:
    """INV-23: garbage in every `eval.*` answer changes nothing about the decision."""
    engine = rules()
    clean = decide(engine, core_answers(), facts())
    poisoned = core_answers()
    for qid in vocab.EVAL_ORDER:
        poisoned[qid] = noul(1.0)
    for qid in vocab.PROBE_ORDER:
        poisoned[qid] = noul(0.0)
    dirty = decide(engine, poisoned, facts())
    assert dirty == clean


# ======================================================================================================================
# 7.7 - management
# ======================================================================================================================


def position(kind: StructureKind = StructureKind.PUT_CREDIT, **overrides: Any) -> Position:
    chain = make_chain()
    structure: Structure = make_structure(chain, kind)
    values: dict[str, Any] = {
        "position_id": POSITION_ID,
        "structure": structure,
        "qty": 1,
        "open_key": SnapshotKey(session=date(2024, 5, 17), slot=Slot.EOD),
        "open_decision_id": DECISION_ID,
        "open_net": BandPrices(orats=-100, worst=-95, mid=-102),
        "max_loss": 40_000,
        "max_profit": 10_000,
        "bp_reserved": 40_000,
        "entry": EntryContext(
            entry_thesis="trend up, iv_rich",
            entry_codes={"trend": "up", "iv_vs_realized": "iv_rich", "iv_rank": "middle"},
            entry_spot=45_000,
            entry_iv30_bp=1600,
            entry_em_hold_tenths=35,
            open_mid_at_decision=-102,
        ),
    }
    values.update(overrides)
    return Position(**values)


def manage_answers(thesis: float = 0.10, threat: Sequence[float] = (1.0, 0.0, 0.0, 0.0), action: str = "hold") -> dict[str, Answer]:
    return {
        "pos.thesis_invalidated": noul(thesis),
        "pos.short_strike_threat": score(threat),
        "pos.action": choice("pos.action", _peaked(vocab.CHOICE_LABELS["pos.action"], action, 0.70)),
    }


def manage_text(adverse: float = 0.10, pending: float = 0.10) -> dict[str, Answer]:
    return {"pos.adverse_news_since_entry": noul(adverse), "pos.pending_binary_since_entry": noul(pending)}


def mfacts(**overrides: Any) -> ManageFacts:
    values: dict[str, Any] = {
        "pnl_headline": 5_000,
        "pnl_frac_loss_ppm": 0,
        "move_code": "little_change",
        "short_dist_code": "far",
        "news_count": 0,
        "news_recent_count": 0,
    }
    values.update(overrides)
    return ManageFacts(**values)


def manage(
    engine: DecisionRules,
    pos: Position,
    *,
    hard: ExitReason | None = None,
    core: Mapping[str, Answer] | None = None,
    text: Mapping[str, Answer] | None = None,
    f: ManageFacts | None = None,
) -> Any:
    core_result = None if core is None else result(core, kind=RequestKind.MANAGE)
    text_result = None if text is None else result(text, kind=RequestKind.MANAGE_TEXT)
    return engine.decide_manage(pos, DECISION_ID, hard, core_result, text_result, f if f is not None else mfacts())


def test_the_hard_exit_always_wins_and_never_consults_jev() -> None:
    engine = rules()
    pos = position(exit_latch=True, watch_text=1)
    decision = manage(engine, pos, hard=ExitReason.FORCE_EXPIRY, core=manage_answers(thesis=0.0), text=manage_text(adverse=0.99))
    assert decision.action == "close" and decision.reason == "force_exit_expiry" and decision.source == "hard_exit"
    assert decision.pressure_ppm is None and decision.exit_latch is True and decision.watch_text == 1
    assert decision.reasons == ()


@pytest.mark.parametrize(
    ("short_dist", "expect_close"),
    [("far", False), ("about_one_move", False), ("close", False), ("at_strike", True), ("breached", True)],
)
def test_the_code_default_when_the_decider_is_down(short_dist: str, expect_close: bool) -> None:
    """7.7 step 2 / D19: no decider result => close only on a breached or at-strike short, `reason = code_default`."""
    decision = manage(rules(), position(), core=None, f=mfacts(short_dist_code=short_dist))
    assert decision.source == "code_default" and decision.pressure_ppm is None
    if expect_close:
        assert decision.action == "close" and decision.reason == ExitReason.CODE_DEFAULT.value
        assert decision.reason in COOLDOWN_REASONS
    else:
        assert decision.action == "hold" and decision.reason == "hold" and decision.reasons == ("hold",)


def test_the_hysteresis_latch_over_a_session_sequence() -> None:
    """7.7: 0.71 sets the latch and closes; 0.60 keeps it (still closing); 0.44 releases it and holds."""
    engine = rules()
    pos = position()
    first = manage(engine, pos, core=manage_answers(thesis=0.71))
    assert first.action == "close" and first.exit_latch is True and first.reason == ExitReason.JEV.value
    assert first.pressure_ppm == 710_000 and first.source == "jev" and first.reasons == ()

    held = position(exit_latch=first.exit_latch)
    second = manage(engine, held, core=manage_answers(thesis=0.60))
    assert second.action == "close" and second.exit_latch is True and second.pressure_ppm == 600_000

    third = manage(engine, position(exit_latch=second.exit_latch), core=manage_answers(thesis=0.44))
    assert third.action == "hold" and third.exit_latch is False
    assert third.reasons == ("hysteresis:released",) and third.pressure_ppm == 440_000

    # without a latch the same 0.60 lands in the discretionary zone and the gated `hold` label holds
    fresh = manage(engine, position(), core=manage_answers(thesis=0.60, action="hold"))
    assert fresh.action == "hold" and fresh.exit_latch is False


def test_exit_pressure_is_the_max_of_thesis_and_short_strike_threat() -> None:
    engine = rules()
    # a Score of levels 0..3: norm = mean / 3; (0,0,0,1) -> 1.0, (0,1,0,0) -> 1/3
    hot = manage(engine, position(), core=manage_answers(thesis=0.10, threat=(0.0, 0.0, 0.0, 1.0)))
    assert hot.pressure_ppm == 1_000_000 and hot.action == "close" and hot.exit_latch is True
    long_call = position(StructureKind.LONG_CALL)
    assert not long_call.structure.short_legs
    ignored = manage(engine, long_call, core=manage_answers(thesis=0.10, threat=(0.0, 0.0, 0.0, 1.0)))
    assert ignored.pressure_ppm == 100_000 and ignored.action == "hold"


def test_an_out_of_band_threat_score_reads_as_the_discretionary_zone() -> None:
    engine = rules()
    decision = manage(
        engine,
        position(),
        core=manage_answers(thesis=0.0, threat=(1.0, 0.0, 0.0, 0.0), action="hold"),
        f=mfacts(pnl_headline=-1),
    )
    assert decision.pressure_ppm == 0 and decision.action == "hold"
    untrusted = {**manage_answers(thesis=0.0, action="hold"), "pos.short_strike_threat": score((1.0, 0.0, 0.0, 0.0), raw_sum=0.5)}
    decision = manage(engine, position(), core=untrusted, f=mfacts(pnl_headline=-1))
    assert decision.pressure_ppm == 500_000  # UNCERTAIN: the discretionary zone, not a free pass


@pytest.mark.parametrize(
    ("action", "pnl", "expect_close"),
    [
        ("take_profit", 5_000, True),
        ("take_profit", -5_000, True),  # label inconsistent with the sign => the risk-reducing default (losing => close)
        ("close_to_cut_loss", -5_000, True),
        ("close_to_cut_loss", 5_000, False),  # inconsistent => default: in profit => hold
        ("hold", 5_000, False),
        ("hold", -5_000, False),
        ("unclear", -5_000, True),
        ("unclear", 5_000, False),
    ],
)
def test_the_discretionary_zone_and_its_risk_reducing_default(action: str, pnl: int, expect_close: bool) -> None:
    decision = manage(rules(), position(), core=manage_answers(thesis=0.50, action=action), f=mfacts(pnl_headline=pnl))
    assert (decision.action == "close") is expect_close
    if expect_close:
        assert decision.reason == ExitReason.JEV.value and decision.source == "jev"
        assert decision.reason in COOLDOWN_REASONS
    assert decision.exit_latch is False  # the discretionary zone never latches


def test_a_failed_action_gate_falls_back_to_the_risk_reducing_default() -> None:
    engine = rules()
    flat = choice("pos.action", {"hold": 0.40, "take_profit": 0.30, "close_to_cut_loss": 0.20, "unclear": 0.10})
    answers = {**manage_answers(thesis=0.50), "pos.action": flat}
    assert flat.p_top < RulesGates().action.p_top
    losing = manage(engine, position(), core=answers, f=mfacts(pnl_headline=-1))
    winning = manage(engine, position(), core=answers, f=mfacts(pnl_headline=1))
    assert losing.action == "close" and losing.reason == ExitReason.JEV.value
    assert winning.action == "hold"


# --- the text rule (INV-16) -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("move_code", "loss_ppm", "confirmed"),
    [
        ("little_change", 0, False),
        ("favourable", 0, False),
        ("adverse", 0, True),
        ("strongly_adverse", 0, True),
        ("little_change", 249_999, False),
        ("little_change", 250_000, True),
    ],
)
def test_text_closes_only_with_code_side_market_data_confirmation(move_code: str, loss_ppm: int, confirmed: bool) -> None:
    decision = manage(
        rules(),
        position(),
        core=manage_answers(thesis=0.10, action="hold"),
        text=manage_text(adverse=0.99),
        f=mfacts(move_code=move_code, pnl_frac_loss_ppm=loss_ppm, pnl_headline=-1_000),
    )
    if confirmed:
        assert decision.action == "close" and decision.reason == ExitReason.TEXT_CONFIRMED.value
        assert decision.reason in COOLDOWN_REASONS and decision.watch_text == 0
    else:
        assert decision.action == "hold" and decision.reason == "hold" and decision.watch_text == 1


@pytest.mark.parametrize("sessions", [2, 5, 20])
def test_unconfirmed_text_never_closes_and_raises_text_watch_exactly_once(sessions: int) -> None:
    """7.7 / INV-16: a persistent (or hostile) headline must never deterministically close a position."""
    engine = rules()
    cfg = RulesConfig()
    pos = position()
    alerts = 0
    for session in range(1, sessions + 1):
        decision = manage(
            engine,
            pos,
            core=manage_answers(thesis=0.10, action="hold"),
            text=manage_text(adverse=1.0, pending=1.0),
            f=mfacts(move_code="little_change", pnl_frac_loss_ppm=0, pnl_headline=100),
        )
        assert decision.action == "hold" and decision.reason == "hold"
        assert decision.watch_text == session
        if text_watch_alert(decision, cfg):
            alerts += 1
        pos = position(exit_latch=decision.exit_latch, watch_text=decision.watch_text)
    assert alerts == 1
    assert "text_watch" in vocab.KNOWN_RISK_EVENT_TYPES
    assert not any(reason.value == "text_watch" for reason in ExitReason)  # there is no text-only exit reason


def test_the_watch_counter_resets_when_the_text_reading_clears() -> None:
    engine = rules()
    pos = position(watch_text=1)
    decision = manage(engine, pos, core=manage_answers(thesis=0.10, action="hold"), text=manage_text(adverse=0.10))
    assert decision.watch_text == 0 and decision.action == "hold"


def test_pending_binary_since_entry_needs_short_premium_and_recent_news() -> None:
    engine = rules()
    text = manage_text(adverse=0.0, pending=1.0)
    confirming = mfacts(move_code="strongly_adverse", pnl_headline=100, news_recent_count=1)
    short_premium = manage(engine, position(StructureKind.PUT_CREDIT), core=manage_answers(action="hold"), text=text, f=confirming)
    assert short_premium.action == "close" and short_premium.reason == ExitReason.TEXT_CONFIRMED.value
    stale = manage(
        engine,
        position(StructureKind.PUT_CREDIT),
        core=manage_answers(action="hold"),
        text=text,
        f=mfacts(move_code="strongly_adverse", pnl_headline=100, news_recent_count=0),
    )
    assert stale.action == "hold"
    long_premium = manage(engine, position(StructureKind.LONG_CALL), core=manage_answers(action="hold"), text=text, f=confirming)
    assert long_premium.action == "hold"


def test_a_confirmed_text_close_does_not_override_an_existing_jev_close_reason() -> None:
    decision = manage(
        rules(),
        position(),
        core=manage_answers(thesis=0.90, action="hold"),
        text=manage_text(adverse=0.99),
        f=mfacts(move_code="strongly_adverse", pnl_headline=-100),
    )
    assert decision.action == "close" and decision.reason == ExitReason.JEV.value and decision.exit_latch is True


def test_every_manage_reason_is_a_known_exit_reason_or_hold() -> None:
    engine = rules()
    seen = {
        manage(engine, position(), hard=ExitReason.STOP_LOSS).reason,
        manage(engine, position(), core=None, f=mfacts(short_dist_code="breached")).reason,
        manage(engine, position(), core=manage_answers(thesis=0.99)).reason,
        manage(
            engine,
            position(),
            core=manage_answers(thesis=0.10, action="hold"),
            text=manage_text(adverse=0.99),
            f=mfacts(move_code="adverse", pnl_headline=-1),
        ).reason,
        manage(engine, position(), core=manage_answers(thesis=0.10, action="hold")).reason,
    }
    known = {reason.value for reason in ExitReason} | {"hold"}
    assert seen <= known
    assert {ExitReason.JEV.value, ExitReason.CODE_DEFAULT.value, ExitReason.TEXT_CONFIRMED.value, ExitReason.STOP_LOSS.value} <= seen
    assert {ExitReason.JEV.value, ExitReason.CODE_DEFAULT.value, ExitReason.TEXT_CONFIRMED.value} <= COOLDOWN_REASONS


def test_custom_thresholds_are_honoured() -> None:
    engine = DecisionRules(
        RulesConfig(
            min_score=0.90,
            veto_hi=0.50,
            veto_lo=0.20,
            weights=RulesWeights(align=0.20, volfit=0.20, fit=0.20, regimefit=0.20, calm=0.20),
            gates=RulesGates(
                direction=GateConfig(p_top=0.80, margin=0.50),
                vol_stance=GateConfig(p_top=0.595, margin=0.245),
                structure=GateConfig(p_top=0.495, margin=0.195),
                action=GateConfig(p_top=0.545, margin=0.195),
            ),
            tiers=RulesTiers(score=((0.90, 1.0),), peakedness=((0.60, 1.0),), environment=(1.0, 1.0, 1.0, 0.0)),
        ),
        ALL_KINDS,
    )
    decision = decide(engine, core_answers(), facts())
    assert decision.action == "no_trade"
    assert "gate:direction:p_top" in decision.reasons and "score:below_min" in decision.reasons
    # S_core = 0.20 * (0.70 + 0.70 + 0.60 + 0.80 + 0.90) = 0.20 * 3.70 = 0.74
    assert decision.score_core_ppm == 740_000
    assert "veto:vol.explained_by_event:hard" not in decision.reasons  # 0.10 < veto_lo 0.20 is still CLEAR
