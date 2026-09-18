"""What `LiveJev` and `ReplayJev` share: the cache keys of a request and the `DecisionResult` built from cached rows.

Keeping both here is what makes 6.8 step 6 true - "the result is then built from the just-stored rows (same code path as a
hit)" - and it is why a recorded run and its replay can never disagree: exactly one function turns rows into answers, and it
checks the model binding (INV-06) on every path, cache hits included.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from jevbot import canon
from jevbot.errors import DeciderResponseError, InvariantError, ModelMismatchError
from jevbot.jev.stats import to_answers
from jevbot.types import CachedAnswer, DecisionRequest, DecisionResult

__all__ = ["cache_keys_for", "check_request_hashes", "result_from_rows"]


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
