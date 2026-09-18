"""`jev/stats.py`: wire answers -> `Answer` structs (DESIGN.md 7.1).

Every number below is hand-computed and written as a literal: the test must not re-derive the statistic with the same
expression the implementation uses.
"""

import math
from typing import Any

import pytest

from jevbot.errors import DeciderResponseError
from jevbot.jev.stats import RAW_SUM_MAX, RAW_SUM_MIN, entropy_of, raw_sum_ok, to_answers
from jevbot.types import ChoiceAns, NoulAns, ScoreAns

NOUL_Q: dict[str, Any] = {"type": "noul", "instructions": "q?", "criteria": {"true": "yes", "false": "no"}}
CHOICE_Q: dict[str, Any] = {
    "type": "choice",
    "instructions": "q?",
    "criteria": {"alpha": "a", "beta": "b", "gamma": "c", "delta": "d"},
}
SCORE_Q: dict[str, Any] = {"type": "score", "instructions": "q?", "criteria": ["lowest level", "second level", "third level", "top level"]}


def _choice(probabilities: dict[str, float], *, choice: str = "alpha", confidence: float = 0.5) -> dict[str, Any]:
    return {"type": "choice", "choice": choice, "confidence": confidence, "probabilities": probabilities}


def _score(probabilities: dict[str, float], *, score: float = 1.0, confidence: float = 0.5) -> dict[str, Any]:
    return {
        "type": "score",
        "score": score,
        "confidence": confidence,
        "legend": {"0": "lowest level", "1": "second level", "2": "third level", "3": "top level"},
        "probabilities": probabilities,
    }


# ======================================================================================================================
# Noul
# ======================================================================================================================


def test_noul_probability_is_clipped_to_the_unit_interval() -> None:
    answers = to_answers(
        {"a": NOUL_Q, "b": NOUL_Q, "c": NOUL_Q},
        {"a": {"type": "noul", "noul": 0.37}, "b": {"type": "noul", "noul": 1.4}, "c": {"type": "noul", "noul": -0.2}},
    )
    assert [a.p for a in answers.values() if isinstance(a, NoulAns)] == [0.37, 1.0, 0.0]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_noul_fails_closed(value: float) -> None:
    with pytest.raises(DeciderResponseError, match="not finite"):
        to_answers({"a": NOUL_Q}, {"a": {"type": "noul", "noul": value}})


@pytest.mark.parametrize("wire", [{"type": "noul"}, {"type": "noul", "noul": "0.4"}, {"type": "noul", "noul": True}, "0.4", None])
def test_a_malformed_noul_fails_closed(wire: Any) -> None:
    with pytest.raises(DeciderResponseError):
        to_answers({"a": NOUL_Q}, {"a": wire})


# ======================================================================================================================
# Choice
# ======================================================================================================================


def test_choice_statistics_are_the_hand_computed_ones() -> None:
    answers = to_answers({"q": CHOICE_Q}, {"q": _choice({"alpha": 0.4, "beta": 0.3, "gamma": 0.2, "delta": 0.1}, choice="alpha", confidence=0.61)})
    answer = answers["q"]
    assert isinstance(answer, ChoiceAns)
    assert answer.top == "alpha" and answer.p_top == pytest.approx(0.4)
    assert answer.margin == pytest.approx(0.1)  # 0.4 - 0.3
    # H = -(0.4 ln0.4 + 0.3 ln0.3 + 0.2 ln0.2 + 0.1 ln0.1) = 1.279854225833668; ln 4 = 1.386294361119891
    assert answer.entropy == pytest.approx(0.9232198, abs=1e-6)
    assert answer.raw_sum == pytest.approx(1.0)
    assert answer.server_choice == "alpha" and answer.server_confidence == pytest.approx(0.61)
    assert list(answer.probs) == ["alpha", "beta", "gamma", "delta"]  # AUTHORED option order, not the wire's


def test_choice_probabilities_are_renormalised_and_the_raw_sum_is_kept() -> None:
    answer = to_answers({"q": CHOICE_Q}, {"q": _choice({"alpha": 0.2, "beta": 0.2, "gamma": 0.2, "delta": 0.2})})["q"]
    assert isinstance(answer, ChoiceAns)
    assert answer.raw_sum == pytest.approx(0.8)
    assert all(value == pytest.approx(0.25) for value in answer.probs.values())
    assert math.fsum(answer.probs.values()) == pytest.approx(1.0)
    assert not raw_sum_ok(answer.raw_sum)  # 7.1: outside [0.98, 1.02] the question reads UNCERTAIN in rules.py
    assert answer.entropy == pytest.approx(1.0)  # a uniform distribution has maximal entropy


def test_choice_ties_are_broken_by_the_authored_option_order() -> None:
    answer = to_answers({"q": CHOICE_Q}, {"q": _choice({"delta": 0.5, "beta": 0.5, "alpha": 0.0, "gamma": 0.0}, choice="delta")})["q"]
    assert isinstance(answer, ChoiceAns)
    assert answer.top == "beta"  # `beta` is authored before `delta`; the server's own `choice` never decides
    assert answer.margin == pytest.approx(0.0) and answer.server_choice == "delta"


def test_choice_values_are_clipped_before_renormalisation() -> None:
    answer = to_answers({"q": CHOICE_Q}, {"q": _choice({"alpha": 1.5, "beta": -0.5, "gamma": 0.0, "delta": 0.0})})["q"]
    assert isinstance(answer, ChoiceAns)
    assert answer.probs == {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0}
    assert answer.raw_sum == pytest.approx(1.0) and answer.entropy == pytest.approx(0.0)


def test_choice_label_sets_must_match_exactly() -> None:
    with pytest.raises(DeciderResponseError, match="label set differs"):
        to_answers({"q": CHOICE_Q}, {"q": _choice({"alpha": 0.4, "beta": 0.3, "gamma": 0.3})})
    with pytest.raises(DeciderResponseError, match="label set differs"):
        to_answers({"q": CHOICE_Q}, {"q": _choice({"alpha": 0.3, "beta": 0.3, "gamma": 0.2, "delta": 0.1, "epsilon": 0.1})})
    with pytest.raises(DeciderResponseError, match="label set differs"):
        to_answers({"q": CHOICE_Q}, {"q": _choice({"alpha": 0.4, "beta": 0.3, "gamma": 0.2, "not_an_option": 0.1})})


def test_a_choice_whose_probabilities_sum_to_zero_fails_closed() -> None:
    with pytest.raises(DeciderResponseError, match="sum to zero"):
        to_answers({"q": CHOICE_Q}, {"q": _choice({"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "delta": -1.0})})


def test_a_choice_without_probabilities_or_choice_fails_closed() -> None:
    with pytest.raises(DeciderResponseError, match="probabilities"):
        to_answers({"q": CHOICE_Q}, {"q": {"type": "choice", "choice": "alpha", "confidence": 0.5}})
    wire = _choice({"alpha": 0.4, "beta": 0.3, "gamma": 0.2, "delta": 0.1})
    del wire["choice"]
    with pytest.raises(DeciderResponseError, match="'choice'"):
        to_answers({"q": CHOICE_Q}, {"q": wire})


# ======================================================================================================================
# Score
# ======================================================================================================================


def test_score_statistics_are_the_hand_computed_ones() -> None:
    answer = to_answers({"q": SCORE_Q}, {"q": _score({"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}, score=2.05, confidence=0.4)})["q"]
    assert isinstance(answer, ScoreAns)
    assert answer.probs == (0.1, 0.2, 0.3, 0.4)
    assert answer.mean == pytest.approx(2.0)  # 0*0.1 + 1*0.2 + 2*0.3 + 3*0.4
    assert answer.norm == pytest.approx(2.0 / 3.0)  # mean / (K - 1)
    assert answer.top == 3 and answer.p_top == pytest.approx(0.4) and answer.margin == pytest.approx(0.1)
    assert answer.server_score == pytest.approx(2.05) and answer.server_confidence == pytest.approx(0.4)
    # ours, not the server's: `score` 2.05 is carried but never used
    assert answer.mean != answer.server_score


def test_score_keys_are_the_strings_zero_to_k_minus_one() -> None:
    with pytest.raises(DeciderResponseError, match="score keys differ"):
        to_answers({"q": SCORE_Q}, {"q": _score({"0": 0.25, "1": 0.25, "2": 0.25, "4": 0.25})})
    with pytest.raises(DeciderResponseError, match="score keys differ"):
        to_answers({"q": SCORE_Q}, {"q": _score({"0": 0.5, "1": 0.5})})


def test_score_renormalises_and_ties_take_the_lowest_level() -> None:
    answer = to_answers({"q": SCORE_Q}, {"q": _score({"0": 0.0, "1": 0.3, "2": 0.3, "3": 0.0})})["q"]
    assert isinstance(answer, ScoreAns)
    assert answer.raw_sum == pytest.approx(0.6)
    assert answer.probs == pytest.approx((0.0, 0.5, 0.5, 0.0))
    assert answer.top == 1 and answer.margin == pytest.approx(0.0)
    assert answer.mean == pytest.approx(1.5) and answer.norm == pytest.approx(0.5)


# ======================================================================================================================
# The batch as a whole
# ======================================================================================================================


def test_every_question_must_be_answered_with_its_own_type() -> None:
    questions = {"n": NOUL_Q, "c": CHOICE_Q}
    with pytest.raises(DeciderResponseError, match="missing 1 answer"):
        to_answers(questions, {"n": {"type": "noul", "noul": 0.5}})
    with pytest.raises(DeciderResponseError, match="were not asked"):
        to_answers(
            questions,
            {
                "n": {"type": "noul", "noul": 0.5},
                "c": _choice({"alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25}),
                "extra": {"type": "noul", "noul": 0.5},
            },
        )
    with pytest.raises(DeciderResponseError, match="has type 'noul', expected 'choice'"):
        to_answers(questions, {"n": {"type": "noul", "noul": 0.5}, "c": {"type": "noul", "noul": 0.5}})


def test_an_unknown_answer_type_fails_closed() -> None:
    # the SDK DROPS an answer whose type it does not model; we see either a missing answer or a foreign `type`
    with pytest.raises(DeciderResponseError, match="expected 'noul'"):
        to_answers({"n": NOUL_Q}, {"n": {"type": "quantum", "value": 1}})
    with pytest.raises(DeciderResponseError, match="unknown type"):
        to_answers({"n": {"type": "quantum", "instructions": "?"}}, {"n": {"type": "quantum"}})


def test_the_result_keeps_the_batch_order() -> None:
    questions = {"c": CHOICE_Q, "n": NOUL_Q, "s": SCORE_Q}
    answers = to_answers(
        questions,
        {
            "n": {"type": "noul", "noul": 0.5},
            "s": _score({"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}),
            "c": _choice({"alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25}),
        },
    )
    assert list(answers) == ["c", "n", "s"]


def test_the_raw_sum_band_is_the_one_7_1_documents() -> None:
    assert (RAW_SUM_MIN, RAW_SUM_MAX) == (0.98, 1.02)
    assert raw_sum_ok(0.98) and raw_sum_ok(1.0) and raw_sum_ok(1.02)
    assert not raw_sum_ok(0.9799) and not raw_sum_ok(1.0201)


def test_entropy_helper_endpoints() -> None:
    assert entropy_of((1.0, 0.0, 0.0, 0.0)) == 0.0
    assert entropy_of((0.25, 0.25, 0.25, 0.25)) == pytest.approx(1.0)
    assert entropy_of((0.5, 0.5)) == pytest.approx(1.0)
    assert entropy_of((1.0,)) == 0.0  # a degenerate single-option question
