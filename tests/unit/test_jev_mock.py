"""`jev/mock.py`: the documented bucket-code mapping and the constant climatological forecaster (DESIGN.md 6.8, D7).

MockJev is the default decider without a key, the comparator every calibration table is read against, and the answer source
of the offline transport fixture. Three properties matter and are pinned here: the mapping is exactly the documented one,
it is **variant-invariant** (so a MockJev run never fires `perturb:disagree:*`), and the evaluation Nouls are fixed
constants - that is what makes MockJev the constant climatological forecaster of 12.1.
"""

import copy
import json
from typing import Any

import pytest

from jevbot import questions as questions_module, vocab
from jevbot.errors import DeciderResponseError
from jevbot.jev.mock import EVAL_CONSTANTS, MOCK_MODEL, MOCK_NAME, MockJev, mock_wire_answers
from jevbot.jev.probe import _bucket_only, _key_perm
from jevbot.jev.stats import to_answers
from jevbot.types import MAPPING, ChoiceAns, Direction, NoulAns, RequestKind, ScoreAns, StructureKind, VolStance
from tests.fixtures.jev_transport import make_request, sample_state

ENTRY = questions_module.ENTRY_V1


def state_with(**paths: str) -> dict[str, Any]:
    """The documented entry state with some bucket labels replaced (`"underlying.trend.direction"="down: ..."`)."""
    state = sample_state(RequestKind.ENTRY)
    for path, label in paths.items():
        node: Any = state
        parts = path.split(".")
        for part in parts[:-1]:
            node = node[part]
        leaf = parts[-1]
        if isinstance(node[leaf], dict) and "bucket" in node[leaf]:
            node[leaf]["bucket"] = label
        else:
            node[leaf] = label
    return state


def answer_for(qid: str, state: dict[str, Any], *, profile: str = "full") -> Any:
    wire = mock_wire_answers(state, {qid: ENTRY[qid]} if qid in ENTRY else {qid: questions_module._ALL_QUESTIONS[qid]}, profile=profile)  # type: ignore[arg-type]
    return to_answers({qid: questions_module._ALL_QUESTIONS[qid]}, wire)[qid]


# ======================================================================================================================
# The documented mapping
# ======================================================================================================================


@pytest.mark.parametrize(
    ("trend", "expected"),
    [("up", "bullish"), ("down", "bearish"), ("flat", "neutral_range"), ("mixed", "conflicting_signals")],
)
def test_direction_comes_from_the_trend_code(trend: str, expected: str) -> None:
    answer = answer_for("under.direction", state_with(**{"underlying.trend.direction": f"{trend}: whatever the label says"}))
    assert isinstance(answer, ChoiceAns)
    assert answer.top == expected and answer.p_top == pytest.approx(0.70)
    others = [p for label, p in answer.probs.items() if label != expected]
    assert others == pytest.approx([0.10, 0.10, 0.10]), "the remaining mass is spread evenly"


def test_a_stretched_far_code_moves_fifteen_points_to_neutral() -> None:
    stretched = state_with(
        **{"underlying.trend.direction": "up: x", "underlying.momentum.distance_from_20d_avg_in_atr": "stretched_far_above: x"}
    )
    answer = answer_for("under.direction", stretched)
    assert isinstance(answer, ChoiceAns)
    assert answer.probs["bullish"] == pytest.approx(0.55) and answer.probs["neutral_range"] == pytest.approx(0.25)
    assert answer.probs["bearish"] == pytest.approx(0.10) and answer.top == "bullish"
    # the `trend_ivrank` profile reads the trend code ONLY (baseline 5)
    plain = answer_for("under.direction", stretched, profile="trend_ivrank")
    assert isinstance(plain, ChoiceAns) and plain.probs["bullish"] == pytest.approx(0.70)


@pytest.mark.parametrize(
    ("rank", "rv", "expected"),
    [
        ("upper_middle", "iv_rich", "sell_premium"),
        ("high", "iv_very_rich", "sell_premium"),
        ("very_low", "iv_cheap", "buy_premium"),
        ("low", "iv_fair", "buy_premium"),
        ("middle", "iv_fair", "limit_vol_exposure"),
        ("middle", "iv_rich", "unclear"),
        ("high", "iv_cheap", "unclear"),
        ("unavailable", "unavailable", "unclear"),
    ],
)
def test_stance_comes_from_the_iv_rank_and_iv_vs_realized_codes(rank: str, rv: str, expected: str) -> None:
    state = state_with(**{"vol_surface.iv_rank_1y": f"{rank}: x", "vol_surface.iv_vs_realized": f"{rv}: x"})
    answer = answer_for("vol.stance", state)
    assert isinstance(answer, ChoiceAns) and answer.top == expected and answer.p_top == pytest.approx(0.70)


def test_structure_is_the_shared_direction_x_stance_mapping() -> None:
    for (direction, stance), structure in MAPPING.items():
        trend = {Direction.BULLISH: "up", Direction.BEARISH: "down", Direction.NEUTRAL: "flat"}[direction]
        rank, rv = {VolStance.SELL: ("high", "iv_rich"), VolStance.BUY: ("low", "iv_cheap"), VolStance.LIMIT: ("middle", "iv_fair")}[stance]
        state = state_with(
            **{
                "underlying.trend.direction": f"{trend}: x",
                "vol_surface.iv_rank_1y": f"{rank}: x",
                "vol_surface.iv_vs_realized": f"{rv}: x",
            }
        )
        answer = answer_for("fit.structure_family", state)
        assert isinstance(answer, ChoiceAns)
        expected = structure.value if isinstance(structure, StructureKind) else "no_trade"
        assert answer.top == expected, f"{direction}/{stance} must map to {expected}"
    # a conflicting direction or an unclear stance can never produce a structure
    conflicting = state_with(**{"underlying.trend.direction": "mixed: x"})
    assert getattr(answer_for("fit.structure_family", conflicting), "top") == "no_trade"


def test_regime_reads_the_trend_the_vol_percentile_and_the_term_structure() -> None:
    calm = {"market.vol_index_pctile_1y": "low: x", "market.vol_term_structure": "contango: x"}
    stressed = {"market.vol_index_pctile_1y": "high: x", "market.vol_term_structure": "backwardation: x"}
    cases = [
        ("up", calm, "trending_up_calm"),
        ("up", stressed, "trending_up_volatile"),
        ("flat", calm, "range_bound_calm"),
        ("flat", stressed, "range_bound_volatile"),
        ("down", calm, "orderly_downtrend"),
        ("down", stressed, "disorderly_selloff"),
        ("mixed", calm, "unclear_or_transition"),
    ]
    for trend, market, expected in cases:
        answer = answer_for("regime.market", state_with(**{"underlying.trend.direction": f"{trend}: x"}, **market))
        assert isinstance(answer, ChoiceAns) and answer.top == expected, f"{trend} + {market}"


def test_risk_environment_counts_the_stressed_codes() -> None:
    benign = state_with(
        **{
            "market.vol_index_pctile_1y": "low: x",
            "market.near_term_stress": "calm: x",
            "market.vol_term_structure": "contango: x",
            "underlying.range.realized_vol_change": "stable: x",
            "underlying.range.move_today": "quiet: x",
            "underlying.range.gap_today": "none: x",
        }
    )
    benign["events"]["inside_holding_window"] = []
    assert getattr(answer_for("risk.environment", benign), "top") == 0
    one = copy.deepcopy(benign)
    one["market"]["near_term_stress"] = "stressed: x"
    assert getattr(answer_for("risk.environment", one), "top") == 1
    two = copy.deepcopy(one)
    two["underlying"]["range"]["realized_vol_change"] = "expanding_sharply: x"
    assert getattr(answer_for("risk.environment", two), "top") == 2
    hostile = copy.deepcopy(two)
    hostile["market"]["vol_index_pctile_1y"]["bucket"] = "high: x"
    hostile["market"]["vol_term_structure"] = "backwardation: x"
    hostile["events"]["inside_holding_window"] = ["major central-bank rate decision in 2 sessions"]
    top = getattr(answer_for("risk.environment", hostile), "top")
    assert top == 3, "the level saturates at the last documented rubric level"


@pytest.mark.parametrize(
    ("code", "expected"),
    [("stretched_far_above", 0.80), ("stretched_far_below", 0.80), ("extended_above", 0.40), ("near_average", 0.10)],
)
def test_under_stretched_reads_the_distance_code(code: str, expected: float) -> None:
    answer = answer_for("under.stretched", state_with(**{"underlying.momentum.distance_from_20d_avg_in_atr": f"{code}: x"}))
    assert isinstance(answer, NoulAns) and answer.p == pytest.approx(expected)


def test_vol_explained_by_event_needs_both_a_listed_event_and_a_rich_iv_rank() -> None:
    rich = state_with(**{"vol_surface.iv_rank_1y": "high: x"})
    assert getattr(answer_for("vol.explained_by_event", rich), "p") == pytest.approx(0.80)
    rich_no_event = copy.deepcopy(rich)
    rich_no_event["events"]["inside_holding_window"] = []
    assert getattr(answer_for("vol.explained_by_event", rich_no_event), "p") == pytest.approx(0.10)
    cheap = state_with(**{"vol_surface.iv_rank_1y": "low: x"})
    assert getattr(answer_for("vol.explained_by_event", cheap), "p") == pytest.approx(0.10)


def test_every_text_noul_answers_no() -> None:
    text_state = sample_state(RequestKind.ENTRY_TEXT)
    wire = mock_wire_answers(text_state, questions_module.ENTRY_TEXT_V1)
    for qid in vocab.TEXT_ORDER:
        assert wire[qid] == {"type": "noul", "noul": 0.05}, "MockJev cannot read text (6.8)"
    manage_text = sample_state(RequestKind.MANAGE_TEXT)
    manage_wire = mock_wire_answers(manage_text, questions_module.MANAGE_TEXT_V1)
    assert all(answer["noul"] == 0.05 for answer in manage_wire.values())


def test_management_answers_come_from_the_pnl_short_distance_and_trend_change_codes() -> None:
    state = sample_state(RequestKind.MANAGE)
    answers = to_answers(questions_module.MANAGE_V1, mock_wire_answers(state, questions_module.MANAGE_V1))
    # the documented sample has trend `up` at entry and `mixed` now -> the thesis reads invalidated
    assert getattr(answers["pos.thesis_invalidated"], "p") == pytest.approx(0.80)
    assert getattr(answers["pos.short_strike_threat"], "top") == 1  # "about_one_move" -> Watch
    # "small_gain" is below the take-profit codes and the trend changed since entry -> the no-match label
    assert getattr(answers["pos.action"], "top") == "unclear"

    winning = copy.deepcopy(state)
    winning["position"]["pnl"] = "large_gain: x"
    assert getattr(to_answers(questions_module.MANAGE_V1, mock_wire_answers(winning, questions_module.MANAGE_V1))["pos.action"], "top") == (
        "take_profit"
    )

    breached = copy.deepcopy(state)
    breached["position"]["short_strike_distance"] = "breached: x"
    breached["position"]["pnl"] = "large_loss: x"
    breached_answers = to_answers(questions_module.MANAGE_V1, mock_wire_answers(breached, questions_module.MANAGE_V1))
    assert getattr(breached_answers["pos.short_strike_threat"], "top") == 3
    assert getattr(breached_answers["pos.action"], "top") == "close_to_cut_loss"

    steady = copy.deepcopy(state)
    steady["changes_since_entry"]["trend_now"] = steady["changes_since_entry"]["trend_at_entry"]
    steady["position"]["pnl"] = "flat: x"
    steady["position"]["short_strike_distance"] = "far: x"
    steady_answers = to_answers(questions_module.MANAGE_V1, mock_wire_answers(steady, questions_module.MANAGE_V1))
    assert getattr(steady_answers["pos.thesis_invalidated"], "p") == pytest.approx(0.10)
    assert getattr(steady_answers["pos.action"], "top") == "hold"


def test_the_evaluation_nouls_are_the_fixed_base_rate_constants() -> None:
    wire = mock_wire_answers(sample_state(), ENTRY)
    for qid in vocab.EVAL_ORDER:
        assert wire[qid]["noul"] == EVAL_CONSTANTS[qid]
    assert {EVAL_CONSTANTS[qid] for qid in ("eval.up_1s", "eval.up_5s")} == {0.53}
    assert {EVAL_CONSTANTS[qid] for qid in vocab.EVAL_ORDER if "_1em_" in qid and "inside" not in qid} == {0.16}
    assert {EVAL_CONSTANTS[qid] for qid in vocab.EVAL_ORDER if "inside" in qid} == {0.68}
    assert EVAL_CONSTANTS["eval.rv_gt_iv_5s"] == 0.30
    # a constant forecaster: the same numbers whatever the state says
    other = state_with(**{"underlying.trend.direction": "down: x", "vol_surface.iv_rank_1y": "high: x"})
    assert {qid: mock_wire_answers(other, ENTRY)[qid] for qid in vocab.EVAL_ORDER} == {qid: wire[qid] for qid in vocab.EVAL_ORDER}


# ======================================================================================================================
# Determinism and variant invariance
# ======================================================================================================================


def test_mock_answers_are_deterministic_and_every_distribution_sums_to_one() -> None:
    state = sample_state()
    first = mock_wire_answers(state, ENTRY)
    assert first == mock_wire_answers(state, ENTRY)
    answers = to_answers(ENTRY, first)
    for qid, answer in answers.items():
        if isinstance(answer, ChoiceAns):
            assert sum(answer.probs.values()) == pytest.approx(1.0), qid
            assert answer.raw_sum == pytest.approx(1.0), qid
        elif isinstance(answer, ScoreAns):
            assert sum(answer.probs) == pytest.approx(1.0), qid
            assert answer.raw_sum == pytest.approx(1.0), qid
        else:
            assert 0.0 <= answer.p <= 1.0, qid


def test_mock_answers_are_invariant_under_every_5_9_variant() -> None:
    state = sample_state()
    base = mock_wire_answers(state, ENTRY)
    assert mock_wire_answers(_key_perm(state), ENTRY) == base, "KEY_PERM: MockJev reads paths, not positions"
    assert mock_wire_answers(_bucket_only(state), ENTRY) == base, "BUCKET_ONLY: it reads codes, not values"
    permuted = questions_module.opt_perm(ENTRY)
    permuted_answers = mock_wire_answers(state, permuted)
    for qid in ENTRY:
        if ENTRY[qid]["type"] == "choice":
            assert permuted_answers[qid]["choice"] == base[qid]["choice"]
            assert permuted_answers[qid]["probabilities"] == base[qid]["probabilities"]
        else:
            assert permuted_answers[qid] == base[qid]
    assert mock_wire_answers({**state, "note": "reference 7"}, ENTRY) == base, "an irrelevant field changes nothing"


# ======================================================================================================================
# Failure modes and the Decider surface
# ======================================================================================================================


def test_an_unknown_schema_or_question_fails_closed() -> None:
    with pytest.raises(DeciderResponseError, match="unknown state schema"):
        mock_wire_answers({"schema": "state.v2.entry"}, ENTRY)
    with pytest.raises(DeciderResponseError, match="unknown state schema"):
        mock_wire_answers({}, ENTRY)
    with pytest.raises(DeciderResponseError, match="unknown question id"):
        mock_wire_answers(sample_state(), {"made.up": {"type": "noul", "criteria": {"true": "a", "false": "b"}}})
    with pytest.raises(DeciderResponseError, match="unknown profile"):
        mock_wire_answers(sample_state(), ENTRY, profile="guess")  # type: ignore[arg-type]
    with pytest.raises(DeciderResponseError, match="unknown profile"):
        MockJev("guess")  # type: ignore[arg-type]


def test_the_decider_surface(tmp_path: Any) -> None:
    jev = MockJev()
    assert (jev.name, jev.model) == (MOCK_NAME, MOCK_MODEL) == ("mock_jev", "mock-1")
    req = make_request()
    result = jev.decide(req)
    assert result.source == "mock" and result.model == "mock-1"
    assert set(result.answers) == set(req.questions) and set(result.cache_keys) == set(req.questions)
    assert result.request_id is None and result.input_tokens is None and result.latency_ms is None
    assert result.decision_id == req.decision_id and result.state_hash == req.state_hash
    assert jev.decide(req).answers == result.answers
    jev.close()
    # the probe set answers too (the transport fixture serves the Step 0 suites with the same function)
    probe_req = make_request(RequestKind.PROBE)
    probe_result = MockJev().decide(probe_req)
    assert getattr(probe_result.answers["probe.closed_higher_5s"], "p") == pytest.approx(0.5)


def test_the_wire_form_is_json_and_carries_the_score_legend() -> None:
    wire = mock_wire_answers(sample_state(), ENTRY)
    text = json.dumps(wire)  # every value is plain JSON
    assert '"legend"' in text and '"probabilities"' in text
    score = wire["risk.environment"]
    assert set(score["legend"]) == {"0", "1", "2", "3"} and set(score["probabilities"]) == {"0", "1", "2", "3"}
    assert score["legend"]["0"].startswith("Benign")
