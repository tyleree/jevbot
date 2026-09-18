"""What `LiveJev` and `ReplayJev` share: the cache keys of a request and the `DecisionResult` built from cached rows.

Keeping both here is what makes 6.8 step 6 true - "the result is then built from the just-stored rows (same code path as a
hit)" - and it is why a recorded run and its replay can never disagree: exactly one function turns rows into answers, and it
checks the model binding (INV-06) on every path, cache hits included.

`dumps_answer` is `CachedAnswer.answer_json`'s spelling (2.5). DESIGN writes it as `canon.dumps_sorted(...)`, but an answer
IS probabilities: `canon.dumps_sorted` refuses every float (INV-24: nothing HASHED INTO THE LEDGER may be a float), so the
two statements cannot both hold. Resolved here, in the narrowest possible place: the same canonical spelling as
`canon.dumps_sorted` - sorted keys, no spaces, `ensure_ascii=False`, `allow_nan=False` - with floats permitted. Nothing this
encodes reaches hashed ledger material: `answer_json` feeds `answers.answer_sha256` and the cache manifest hash, both
provenance digests of the run store's `meta` table, never the hash chain. A NaN / infinity probability is refused, so it can
never be cached.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from jevbot import canon
from jevbot.errors import DeciderResponseError, InvariantError, ModelMismatchError
from jevbot.jev.stats import to_answers
from jevbot.types import CachedAnswer, DecisionRequest, DecisionResult

__all__ = ["cache_keys_for", "check_request_hashes", "dumps_answer", "result_from_rows"]


def _answer_walk(value: object, path: str) -> None:
    """`canon`'s pre-walk with floats allowed: only JSON scalars, lists and str-keyed dicts may be cached."""
    if value is None or type(value) in (str, int, bool, float):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str:
                raise DeciderResponseError(f"answer JSON: dict key of type {type(key).__name__} at {path}")
            _answer_walk(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _answer_walk(item, f"{path}[{index}]")
        return
    raise DeciderResponseError(f"answer JSON: unsupported value of type {type(value).__name__} at {path}")


def dumps_answer(answer: object) -> str:
    """The canonical text of ONE wire answer (`CachedAnswer.answer_json`): sorted keys, floats allowed, NaN refused."""
    _answer_walk(answer, "$")
    try:
        return json.dumps(answer, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except ValueError:  # allow_nan=False: a NaN / infinity probability is never cached
        raise DeciderResponseError("answer JSON: a non-finite number cannot be cached") from None


def cache_keys_for(model: str, req: DecisionRequest) -> dict[str, str]:
    """`{question id: canon.cache_key(model, state, question_set_hash, question)}` for the FULL batch (6.8 step 1, D8)."""
    return {qid: canon.cache_key(model, req.state, req.question_set_hash, dict(question)) for qid, question in req.questions.items()}


def check_request_hashes(req: DecisionRequest) -> None:
    """The request's own hashes must match its bytes: a builder bug must not reach the wire (or a paid request)."""
    state_hash = canon.sha256_hex(canon.dumps_ordered(req.state))
    if state_hash != req.state_hash:
        raise InvariantError(f"DecisionRequest.state_hash does not match its state ({req.state_hash} vs {state_hash})")
    qset_hash = canon.sha256_hex(canon.dumps_ordered([dict(q) for q in req.questions.values()]))
    if qset_hash != req.question_set_hash:
        raise InvariantError(f"DecisionRequest.question_set_hash does not match its questions ({req.question_set_hash} vs {qset_hash})")


def result_from_rows(
    req: DecisionRequest, model: str, keys: Mapping[str, str], rows: Mapping[str, CachedAnswer], *, source: str
) -> DecisionResult:
    """Build the `DecisionResult` of a complete batch of cached rows.

    Every row's `response_model` is verified against `model` (INV-06 is checked on cache hits too), then the stored
    `answer_json` texts are parsed back into the wire form and validated by `stats.to_answers` - the same validation a fresh
    response walks through. A missing key here is a bug in the caller (both deciders check completeness first).
    """
    wire: dict[str, Any] = {}
    sidecar: Sequence[CachedAnswer] = [rows[key] for key in keys.values() if key in rows]
    for qid, key in keys.items():
        row = rows.get(key)
        if row is None:
            raise InvariantError(f"result_from_rows: no cached row for question {qid!r}")
        if row.response_model != model:
            raise ModelMismatchError(
                f"cached answer for {qid!r} was produced by model {row.response_model!r}, not {model!r}: fail closed (INV-06)"
            )
        try:
            wire[qid] = json.loads(row.answer_json)
        except ValueError:
            raise DeciderResponseError(f"cached answer for {qid!r} is not valid JSON") from None
    first = sidecar[0] if sidecar else None
    return DecisionResult(
        decision_id=req.decision_id,
        kind=req.kind,
        variant=req.variant,
        state_hash=req.state_hash,
        question_set_hash=req.question_set_hash,
        model=model,
        answers=to_answers(req.questions, wire),
        cache_keys=dict(keys),
        source=source,
        request_id=first.request_id if first is not None else None,
        input_tokens=first.input_tokens if first is not None else None,
        latency_ms=first.latency_ms if first is not None else None,
    )
