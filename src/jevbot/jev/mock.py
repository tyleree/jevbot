"""MockJev: a fixed, documented mapping from bucket CODES to wire answers (DESIGN.md 6.8, D7).

`mock_wire_answers(state, questions)` returns exactly what the API would put in `answers`, so the offline transport fixture
(15.3) serves it through the real SDK and MockJev, LiveJev-over-the-fixture and the replayed cache all walk the same parsing
path. MockJev is the project's default decider without a key (D7) and, with the fixed evaluation constants below, the
**constant climatological forecaster** every calibration comparison needs (12.1).

It reads only bucket **codes** - the text before the first colon of a bucket label - and never a value, so:

* the `BUCKET_ONLY` variant (the `{"value", "bucket"}` dicts replaced by their bucket string) answers identically;
* `KEY_PERM` (reversed key order) and `OPT_PERM` (reversed Choice options) answer identically;

which is what "all perturbation variants agree by construction" means in 6.8: MockJev runs never fire `perturb:disagree:*`.

Profile `full` is the documented mapping below. Profile `trend_ivrank` is baseline 5 (12.4): direction from
`underlying.trend.direction` alone (no stretch adjustment) and stance from `vol_surface.iv_rank_1y` alone; everything else is
unchanged. An unknown state schema or an unknown question id is a `DeciderResponseError` (fail closed like any decider).
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final

from jevbot import canon, vocab
from jevbot.config import MockProfile
from jevbot.errors import DeciderResponseError
from jevbot.jev.stats import to_answers
from jevbot.types import MAPPING, DecisionRequest, DecisionResult, Direction, StructureKind, VolStance

__all__ = ["EVAL_CONSTANTS", "MOCK_MODEL", "MOCK_NAME", "MockJev", "mock_wire_answers"]

MOCK_NAME: Final = "mock_jev"
MOCK_MODEL: Final = "mock-1"

_TOP: Final = 0.70  # every Choice / Score answer puts this much on its mapped label, the rest spread evenly
_STRETCH_SHIFT: Final = 0.15  # a `stretched_far_*` code moves this much from the top direction to neutral_range

# 6.8: the evaluation Nouls are FIXED base-rate constants - MockJev is the constant climatological comparator
EVAL_CONSTANTS: Final[Mapping[str, float]] = {
    "eval.up_1s": 0.53,
    "eval.down_1em_1s": 0.16,
    "eval.up_1em_1s": 0.16,
    "eval.inside_1em_1s": 0.68,
    "eval.up_5s": 0.53,
    "eval.down_1em_5s": 0.16,
    "eval.up_1em_5s": 0.16,
    "eval.inside_1em_5s": 0.68,
    "eval.rv_gt_iv_5s": 0.30,
    "eval.down_1em_hold": 0.16,
    "eval.up_1em_hold": 0.16,
    "eval.inside_1em_hold": 0.68,
}
_TEXT_NOUL: Final = 0.05  # MockJev cannot read text: every text Noul answers "no"
_PROBE_NOUL: Final = 0.50  # the recall probe has no data to recall: an honest coin flip

# `under.stretched`: 0.80 / 0.40 / 0.10 by DIST_ATR code (an unavailable distance is not evidence of a stretch)
_STRETCHED_P: Final[Mapping[str, float]] = {
    "stretched_far_below": 0.80,
    "stretched_far_above": 0.80,
    "extended_below": 0.40,
    "extended_above": 0.40,
    "near_average": 0.10,
}
_DIRECTION_BY_TREND: Final[Mapping[str, str]] = {
    "up": "bullish",
    "down": "bearish",
    "flat": "neutral_range",
    "mixed": "conflicting_signals",
}
# `pos.short_strike_threat` level by SHORT_DIST code (a position without a short strike renders "unavailable" -> Safe)
_THREAT_LEVEL: Final[Mapping[str, int]] = {"far": 0, "about_one_move": 1, "close": 2, "at_strike": 3, "breached": 3}
_RICH_RANKS: Final[frozenset[str]] = frozenset({"upper_middle", "high"})
_CHEAP_RANKS: Final[frozenset[str]] = frozenset({"very_low", "low"})
_LOSS_CODES: Final[frozenset[str]] = frozenset({"large_loss", "loss", "moderate_loss"})
_GAIN_CODES: Final[frozenset[str]] = frozenset({"gain", "large_gain"})


# ======================================================================================================================
# Reading the state: codes only, so every 5.9 variant reads the same
# ======================================================================================================================


def _code(state: Mapping[str, Any], path: str) -> str:
    """The bucket CODE at a dotted state path; `vocab.UNAVAILABLE` when the path or its label is missing.

    Accepts both renderings of a bucketed field: `{"value": ..., "bucket": "<code>: <meaning>"}` and the bare string the
    `BUCKET_ONLY` variant leaves behind.
    """
    node: Any = state
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return vocab.UNAVAILABLE
        node = node[part]
    if isinstance(node, Mapping):
        node = node.get("bucket")
    if not isinstance(node, str):
        return vocab.UNAVAILABLE
    return vocab.bucket_code(node)


def _items(state: Mapping[str, Any], path: str) -> list[Any]:
    node: Any = state
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return []
        node = node[part]
    return list(node) if isinstance(node, list) else []


# ======================================================================================================================
# Wire answer builders
# ======================================================================================================================


def _noul(p: float) -> dict[str, Any]:
    return {"type": "noul", "noul": p}


def _spread(labels: Sequence[str], top: str) -> dict[str, float]:
    if top not in labels:  # pragma: no cover - every mapping below returns an authored label
        raise DeciderResponseError(f"mock: {top!r} is not an option of this question")
    rest = (1.0 - _TOP) / (len(labels) - 1) if len(labels) > 1 else 0.0
    return {label: (_TOP if label == top else rest) for label in labels}


def _choice(question: Mapping[str, Any], probs: Mapping[str, float]) -> dict[str, Any]:
    criteria = question.get("criteria")
    if not isinstance(criteria, Mapping):  # pragma: no cover - the question sets are frozen
        raise DeciderResponseError("mock: a choice question without criteria")
    ordered = {label: float(probs.get(label, 0.0)) for label in criteria}
    top = max(ordered, key=lambda label: (ordered[label], -list(ordered).index(label)))
    return {"type": "choice", "choice": top, "confidence": ordered[top], "probabilities": ordered}


def _score(question: Mapping[str, Any], level: int) -> dict[str, Any]:
    criteria = question.get("criteria")
    if not isinstance(criteria, list) or not criteria:  # pragma: no cover - the question sets are frozen
        raise DeciderResponseError("mock: a score question without criteria")
    levels = len(criteria)
    top = min(max(level, 0), levels - 1)
    rest = (1.0 - _TOP) / (levels - 1) if levels > 1 else 0.0
    probabilities = {str(index): (_TOP if index == top else rest) for index in range(levels)}
    mean = sum(index * probability for index, probability in enumerate(probabilities.values()))
    return {
        "type": "score",
        "score": mean,
        "confidence": _TOP,
        # the vendor quickstart omits `legend`; the real API sends it and the SDK decodes it, so the mock does too (B1.4)
        "legend": {str(index): str(text) for index, text in enumerate(criteria)},
        "probabilities": probabilities,
    }


# ======================================================================================================================
# The mapping (6.8)
# ======================================================================================================================


def _direction_probs(state: Mapping[str, Any], question: Mapping[str, Any], profile: MockProfile) -> dict[str, float]:
    labels = tuple(question.get("criteria", {}))
    top = _DIRECTION_BY_TREND.get(_code(state, "underlying.trend.direction"), "conflicting_signals")
    probs = _spread(labels, top)
    if (
        profile == "full"
        and top != "neutral_range"
        and _code(state, "underlying.momentum.distance_from_20d_avg_in_atr").startswith("stretched_far")
    ):
        probs[top] -= _STRETCH_SHIFT
        probs["neutral_range"] += _STRETCH_SHIFT
    return probs


def _stance_label(state: Mapping[str, Any], profile: MockProfile) -> str:
    rank = _code(state, "vol_surface.iv_rank_1y")
    if profile == "trend_ivrank":  # baseline 5: the iv rank alone
        if rank in _RICH_RANKS:
            return "sell_premium"
        if rank in _CHEAP_RANKS:
            return "buy_premium"
        return "limit_vol_exposure" if rank == "middle" else "unclear"
    rv = _code(state, "vol_surface.iv_vs_realized")
    if rank in _RICH_RANKS and rv in {"iv_rich", "iv_very_rich"}:
        return "sell_premium"
    if rank in _CHEAP_RANKS and rv in {"iv_cheap", "iv_fair"}:
        return "buy_premium"
    if rank == "middle" and rv == "iv_fair":
        return "limit_vol_exposure"
    return "unclear"


def _structure_label(direction: str, stance: str) -> str:
    """`types.MAPPING` at the mapped direction x stance; `no_trade` when either is a no-match label or the cell is None."""
    try:
        cell = MAPPING[(Direction(direction), VolStance(stance))]
    except ValueError:
        return "no_trade"
    return cell.value if isinstance(cell, StructureKind) else "no_trade"


def _calm(state: Mapping[str, Any]) -> bool:
    return _code(state, "market.vol_index_pctile_1y") not in _RICH_RANKS and _code(state, "market.vol_term_structure") != "backwardation"


def _regime_label(state: Mapping[str, Any]) -> str:
    trend = _code(state, "underlying.trend.direction")
    calm = _calm(state)
    if trend == "up":
        return "trending_up_calm" if calm else "trending_up_volatile"
    if trend == "flat":
        return "range_bound_calm" if calm else "range_bound_volatile"
    if trend == "down":
        return "orderly_downtrend" if calm else "disorderly_selloff"
    return "unclear_or_transition"  # "mixed" and an unavailable trend


def _stress_count(state: Mapping[str, Any]) -> int:
    """How many of the six documented stress indicators are on; `risk.environment` reports `min(count, 3)`."""
    return sum(
        (
            _code(state, "market.vol_index_pctile_1y") in _RICH_RANKS,
            _code(state, "market.near_term_stress") == "stressed",
            _code(state, "underlying.range.realized_vol_change") in {"expanding", "expanding_sharply"},
            _code(state, "underlying.range.move_today") in {"large_decline", "large_advance"}
            or _code(state, "underlying.range.gap_today") in {"large_gap_down", "large_gap_up"},
            _code(state, "market.vol_term_structure") == "backwardation",
            bool(_items(state, "events.inside_holding_window")),
        )
    )


def _action_label(state: Mapping[str, Any]) -> str:
    short_dist = _code(state, "position.short_strike_distance")
    pnl = _code(state, "position.pnl")
    if short_dist in {"at_strike", "breached"} or pnl in {"large_loss", "loss"}:
        return "close_to_cut_loss"
    if pnl in _GAIN_CODES:
        return "take_profit"
    if _code(state, "changes_since_entry.trend_now") != _code(state, "changes_since_entry.trend_at_entry"):
        return "unclear"
    return "hold"


def _answer(qid: str, question: Mapping[str, Any], state: Mapping[str, Any], profile: MockProfile) -> dict[str, Any]:
    if qid in EVAL_CONSTANTS:
        return _noul(EVAL_CONSTANTS[qid])
    if qid in vocab.TEXT_IDS or qid in vocab.MANAGE_TEXT_IDS:
        return _noul(_TEXT_NOUL)
    if qid == "regime.market":
        return _choice(question, _spread(tuple(question.get("criteria", {})), _regime_label(state)))
    if qid == "under.direction":
        return _choice(question, _direction_probs(state, question, profile))
    if qid == "under.stretched":
        return _noul(_STRETCHED_P.get(_code(state, "underlying.momentum.distance_from_20d_avg_in_atr"), 0.10))
    if qid == "vol.stance":
        return _choice(question, _spread(tuple(question.get("criteria", {})), _stance_label(state, profile)))
    if qid == "vol.explained_by_event":
        listed = bool(_items(state, "events.inside_holding_window"))
        rich = _code(state, "vol_surface.iv_rank_1y") in _RICH_RANKS
        return _noul(0.80 if listed and rich else 0.10)
    if qid == "fit.structure_family":
        direction = _DIRECTION_BY_TREND.get(_code(state, "underlying.trend.direction"), "conflicting_signals")
        label = _structure_label(direction, _stance_label(state, profile))
        return _choice(question, _spread(tuple(question.get("criteria", {})), label))
    if qid == "risk.environment":
        return _score(question, min(_stress_count(state), 3))
    if qid == "pos.thesis_invalidated":
        changed = _code(state, "changes_since_entry.trend_now") != _code(state, "changes_since_entry.trend_at_entry")
        return _noul(0.80 if changed else 0.10)
    if qid == "pos.short_strike_threat":
        return _score(question, _THREAT_LEVEL.get(_code(state, "position.short_strike_distance"), 0))
    if qid == "pos.action":
        return _choice(question, _spread(tuple(question.get("criteria", {})), _action_label(state)))
    if qid in vocab.PROBE_IDS:
        return _noul(_PROBE_NOUL)
    raise DeciderResponseError(f"mock: unknown question id {qid!r}")


def mock_wire_answers(
    state: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]], *, profile: MockProfile = "full"
) -> dict[str, Any]:
    """The wire `answers` object MockJev (and the offline transport fixture) produce for one batch.

    `state` must carry a known `schema` (5.6 / 5.7 / 6.6) and `questions` must be the raw-dict batch; anything else is a
    `DeciderResponseError`.
    """
    if not isinstance(state, Mapping):
        raise DeciderResponseError("mock: the state is not a JSON object")
    schema = state.get("schema")
    if not isinstance(schema, str) or schema not in set(vocab.STATE_SCHEMA.values()):
        raise DeciderResponseError(f"mock: unknown state schema {schema!r}")
    if profile not in ("full", "trend_ivrank"):
        raise DeciderResponseError(f"mock: unknown profile {profile!r}")
    return {qid: _answer(qid, question, state, profile) for qid, question in questions.items()}


class MockJev:
    """The `Decider` of D7: no key, no network, deterministic (`name = "mock_jev"`, `model = "mock-1"`)."""

    def __init__(self, profile: MockProfile = "full") -> None:
        if profile not in ("full", "trend_ivrank"):
            raise DeciderResponseError(f"mock: unknown profile {profile!r}")
        self.profile: MockProfile = profile

    @property
    def name(self) -> str:
        return MOCK_NAME

    @property
    def model(self) -> str:
        return MOCK_MODEL

    def decide(self, req: DecisionRequest) -> DecisionResult:
        """Answer the full batch from the state's bucket codes; the answers walk the same validation as a live response."""
        wire = mock_wire_answers(req.state, req.questions, profile=self.profile)
        answers = to_answers(req.questions, wire)
        return DecisionResult(
            decision_id=req.decision_id,
            kind=req.kind,
            variant=req.variant,
            state_hash=req.state_hash,
            question_set_hash=req.question_set_hash,
            model=MOCK_MODEL,
            answers=answers,
            cache_keys={
                qid: canon.cache_key(MOCK_MODEL, req.state, req.question_set_hash, dict(question))
                for qid, question in req.questions.items()
            },
            source="mock",
            request_id=None,
            input_tokens=None,
            latency_ms=None,
        )

    def close(self) -> None:
        """Nothing to release (MockJev holds no connection)."""
