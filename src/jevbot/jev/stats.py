"""Wire answers -> `Answer` structs: validation, renormalisation and the derived statistics (DESIGN.md 7.1).

`to_answers(questions, wire_answers)` is the ONE place where a server answer becomes a typed `Answer`. It runs **before**
anything is cached (6.8 step 5), so an incomplete, mistyped or mislabelled batch never reaches the decision cache and never
reaches `rules.py`: every failure is a `DeciderResponseError` (fail closed, INV-05).

The wire form is the parsed JSON sub-object of `raw_http_response.json()["answers"][qid]` (string keys - the most stable shape
across SDK versions), not the SDK's decoded struct:

    noul    {"type": "noul",   "noul": 0.71}
    choice  {"type": "choice", "choice": "bullish", "confidence": 0.62, "probabilities": {"bullish": 0.55, ...}}
    score   {"type": "score",  "score": 1.4, "confidence": 0.5, "legend": {"0": "...", ...}, "probabilities": {"0": 0.1, ...}}

Rules of 7.1, in order:

* every question id of the batch must be answered, with the question's own `type` (an id we did not ask is refused too);
* Noul: `p = clip(noul, 0, 1)`; a non-finite value (NaN / inf) is a `DeciderResponseError`;
* Choice: the probability label set must EQUAL the question's `criteria` keys; each value is clipped to [0, 1] and divided by
  their sum (B1.4: "sum to approximately 1"); the pre-normalisation sum is kept as `raw_sum`; a sum <= 0 raises;
* Score: the wire keys are the strings "0".."K-1" for a K-level question; same renormalisation; `mean = sum(i * p_i)`,
  `norm = mean / (K - 1)`;
* `top` is the argmax with ties broken by the AUTHORED option order (for a Score: the lowest level), `margin = p1 - p2`,
  `entropy = -sum(p ln p) / ln K` in [0, 1];
* the server's own `choice` / `score` / `confidence` are carried for analysis only and never gate anything (B1.4).

A `raw_sum` outside [`RAW_SUM_MIN`, `RAW_SUM_MAX`] is NOT an error here: 7.1 makes that question read UNCERTAIN / fail its
gate in `rules.py` (and counts an anomaly). `raw_sum_ok` is that band, single-sourced.
"""

import math
from collections.abc import Mapping
from typing import Any, Final

from jevbot.errors import DeciderResponseError
from jevbot.types import Answer, ChoiceAns, NoulAns, ScoreAns

__all__ = ["RAW_SUM_MAX", "RAW_SUM_MIN", "entropy_of", "raw_sum_ok", "to_answers"]

# 7.1: outside this band the question is treated as UNCERTAIN / its gate fails (an anomaly is counted by rules.py)
RAW_SUM_MIN: Final = 0.98
RAW_SUM_MAX: Final = 1.02


def raw_sum_ok(raw_sum: float) -> bool:
    """True iff the pre-normalisation probability sum is inside the 7.1 band [0.98, 1.02]."""
    return RAW_SUM_MIN <= raw_sum <= RAW_SUM_MAX


def _fail(message: str) -> DeciderResponseError:
    # Messages name ids, types and label names - never a probability, a headline or any other response content.
    return DeciderResponseError(message)


def _number(value: object, qid: str, field: str) -> float:
    """A finite float from a JSON number (a bool is not a number; NaN / inf are refused, 2.5)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _fail(f"answer {qid!r}: field {field!r} is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise _fail(f"answer {qid!r}: field {field!r} is not finite")
    return number


def _clip01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def entropy_of(probs: tuple[float, ...]) -> float:
    """`-sum(p ln p) / ln K`, in [0, 1]; 0.0 for a degenerate single-option question."""
    k = len(probs)
    if k < 2:
        return 0.0
    total = 0.0
    for p in probs:
        if p > 0.0:
            total -= p * math.log(p)
    return min(1.0, max(0.0, total / math.log(k)))


def _top_and_margin(probs: tuple[float, ...]) -> tuple[int, float, float]:
    """(index of the argmax with ties broken by authored order, p_top, p1 - p2)."""
    best = 0
    for index in range(1, len(probs)):
        if probs[index] > probs[best]:  # strictly greater: the earlier (authored-first) option keeps a tie
            best = index
    if len(probs) == 1:
        return best, probs[best], probs[best]
    second = max(p for index, p in enumerate(probs) if index != best)
    return best, probs[best], probs[best] - second


def _renormalise(raw: tuple[float, ...], qid: str) -> tuple[tuple[float, ...], float]:
    """(probabilities divided by their sum, the pre-normalisation sum of the clipped values)."""
    clipped = tuple(_clip01(value) for value in raw)
    raw_sum = math.fsum(clipped)
    if raw_sum <= 0.0:
        raise _fail(f"answer {qid!r}: the probabilities sum to zero or less")
    return tuple(value / raw_sum for value in clipped), raw_sum


def _wire_object(wire: object, qid: str, expected_type: str) -> Mapping[str, Any]:
    if not isinstance(wire, Mapping):
        raise _fail(f"answer {qid!r} is not a JSON object")
    kind = wire.get("type")
    if kind != expected_type:
        got = kind if isinstance(kind, str) else type(kind).__name__
        raise _fail(f"answer {qid!r} has type {got!r}, expected {expected_type!r}")
    return wire


def _probability_map(wire: Mapping[str, Any], qid: str) -> Mapping[str, Any]:
    probabilities = wire.get("probabilities")
    if not isinstance(probabilities, Mapping):
        raise _fail(f"answer {qid!r}: field 'probabilities' is missing or not a JSON object")
    return probabilities


def _noul(qid: str, wire: object) -> NoulAns:
    answer = _wire_object(wire, qid, "noul")
    return NoulAns(p=_clip01(_number(answer.get("noul"), qid, "noul")))


def _choice(qid: str, question: Mapping[str, Any], wire: object) -> ChoiceAns:
    answer = _wire_object(wire, qid, "choice")
    criteria = question.get("criteria")
    if not isinstance(criteria, Mapping):
        raise _fail(f"question {qid!r} is a choice without a criteria mapping")
    labels = tuple(criteria)  # AUTHORED option order: it decides ties and the order of ChoiceAns.probs
    probabilities = _probability_map(answer, qid)
    if set(probabilities) != set(labels):
        missing = sorted(set(labels) - set(probabilities))
        unexpected = sorted(k for k in probabilities if k not in set(labels))
        raise _fail(f"answer {qid!r}: label set differs (missing {missing}, unexpected {unexpected})")
    probs, raw_sum = _renormalise(tuple(_number(probabilities[label], qid, f"probabilities.{label}") for label in labels), qid)
    best, p_top, margin = _top_and_margin(probs)
    server_choice = answer.get("choice")
    if not isinstance(server_choice, str):
        raise _fail(f"answer {qid!r}: field 'choice' is missing or not a string")
    return ChoiceAns(
        probs=dict(zip(labels, probs, strict=True)),
        top=labels[best],
        p_top=p_top,
        margin=margin,
        entropy=entropy_of(probs),
        raw_sum=raw_sum,
        server_choice=server_choice,
        server_confidence=_number(answer.get("confidence"), qid, "confidence"),
    )


def _score(qid: str, question: Mapping[str, Any], wire: object) -> ScoreAns:
    answer = _wire_object(wire, qid, "score")
    criteria = question.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise _fail(f"question {qid!r} is a score without a non-empty criteria list")
    levels = len(criteria)
    probabilities = _probability_map(answer, qid)
    keys = tuple(str(level) for level in range(levels))
    if set(probabilities) != set(keys):
        missing = sorted(set(keys) - set(probabilities))
        unexpected = sorted(k for k in probabilities if k not in set(keys))
        raise _fail(f"answer {qid!r}: score keys differ (missing {missing}, unexpected {unexpected})")
    probs, raw_sum = _renormalise(tuple(_number(probabilities[key], qid, f"probabilities.{key}") for key in keys), qid)
    best, p_top, margin = _top_and_margin(probs)
    mean = math.fsum(level * p for level, p in enumerate(probs))
    return ScoreAns(
        probs=probs,
        mean=mean,
        norm=mean / (levels - 1) if levels > 1 else 0.0,
        top=best,
        p_top=p_top,
        margin=margin,
        entropy=entropy_of(probs),
        raw_sum=raw_sum,
        server_score=_number(answer.get("score"), qid, "score"),
        server_confidence=_number(answer.get("confidence"), qid, "confidence"),
    )


def to_answers(questions: Mapping[str, Mapping[str, Any]], wire_answers: Mapping[str, Any]) -> dict[str, Answer]:
    """Validate and convert one batch of wire answers (7.1); `DeciderResponseError` on anything unexpected.

    The result has exactly the question ids of `questions`, in their batch order.
    """
    if not isinstance(wire_answers, Mapping):
        raise _fail("the response has no 'answers' object")
    missing = [qid for qid in questions if qid not in wire_answers]
    if missing:
        raise _fail(f"the response is missing {len(missing)} answer(s): {sorted(missing)}")
    unexpected = sorted(qid for qid in wire_answers if qid not in questions)
    if unexpected:
        raise _fail(f"the response carries {len(unexpected)} answer(s) that were not asked: {unexpected}")
    answers: dict[str, Answer] = {}
    for qid, question in questions.items():
        kind = question.get("type")
        wire = wire_answers[qid]
        if kind == "noul":
            answers[qid] = _noul(qid, wire)
        elif kind == "choice":
            answers[qid] = _choice(qid, question, wire)
        elif kind == "score":
            answers[qid] = _score(qid, question, wire)
        else:
            raise _fail(f"question {qid!r} has an unknown type {kind!r}")
    return answers
