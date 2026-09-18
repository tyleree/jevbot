"""Cross-namespace model-agreement statistics for a forced model change (DESIGN.md 12.9; WP07).

`jevbot eval model-agreement RUN_OLD RUN_NEW` prints exactly the three pre-registered statistics of
`[prereg.model_change].agreement_report`:

1. **gate-decision agreement rate** - the share of joined `(underlying, session)` pairs whose `EntryDecision.action`
   AND `kind` are equal;
2. **mean `|dp|` on the primary family** - over joined FORECAST pairs, per question and pooled, with-text and
   without-text separately;
3. **top-label agreement on the Choices** - `regime.market`, `under.direction`, `vol.stance`, `fit.structure_family`.

This module is **explicitly exempt from the pooling guard** (12.1, INV-22): it compares two namespaces *with each
other* and emits no skill number.  `eval/report.py` still refuses any report spanning both.  The pre-registration
states no pass / fail threshold: whatever the agreement, the new model is a new experiment with its own looks.

It reads run stores through their own `ledger` table (13.4) with stdlib sqlite3, read-only: no engine, no other `eval`
module, no writes of any kind.  Sessions present in only one store are **listed, not imputed**.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from jevbot import vocab
from jevbot.errors import EvalError

__all__ = [
    "AGREEMENT_CHOICES",
    "AgreementReport",
    "DpStats",
    "RunStoreRef",
    "model_agreement",
    "read_namespace",
]

#: the four Choice questions of the pre-registered agreement report (12.9c)
AGREEMENT_CHOICES: Final[tuple[str, ...]] = ("regime.market", "under.direction", "vol.stance", "fit.structure_family")

_PPM: Final = 1_000_000.0


@runtime_checkable
class _HasPath(Protocol):
    """Anything that exposes the run store's sqlite path - e.g. `eval.load`'s `RunStore`."""

    @property
    def path(self) -> Path: ...


#: A run store: its `run.sqlite` path, or any object exposing that path as `.path`.
RunStoreRef = str | Path | _HasPath


def _store_path(ref: RunStoreRef) -> Path:
    if isinstance(ref, str | Path):
        return Path(ref)
    return Path(ref.path)


def _connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise EvalError(f"run store not found: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


# ======================================================================================================================
# run store reads
# ======================================================================================================================


def read_namespace(store: RunStoreRef) -> str:
    """The run's namespace: the `meta` table's `namespace` key, else the RUN_START payload's (13.4, 2.11)."""
    path = _store_path(store)
    with closing(_connect(path)) as connection:
        try:
            row = connection.execute("SELECT value FROM meta WHERE key = 'namespace'").fetchone()
        except sqlite3.DatabaseError:  # pragma: no cover - a store without a meta table
            row = None
        if row is not None and str(row["value"]).strip():
            return str(row["value"])
        start = connection.execute("SELECT payload FROM ledger WHERE kind = 'run_start' ORDER BY seq LIMIT 1").fetchone()
    if start is None:
        raise EvalError(f"{path} has neither a meta namespace nor a RUN_START entry")
    namespace = json.loads(str(start["payload"])).get("namespace")
    if not isinstance(namespace, str) or not namespace:
        raise EvalError(f"{path}: RUN_START carries no namespace")
    return namespace


@dataclass(frozen=True, slots=True)
class _EntryDecision:
    session: date
    underlying: str
    action: str
    kind: str | None
    top_labels: Mapping[str, str]


def _top_label(question_id: str, answer: object) -> str | None:
    """The Choice question's top label from a ledgered answer (2.11 `answers{qid: ppm ints}`).

    The DECISION payload records a Choice answer as `{label: ppm int}`; an explicit `{"top": ..., "probs": {...}}`
    shape is also accepted.  Ties are broken by the AUTHORED option order of `vocab.CHOICE_LABELS` (2.5).
    """
    labels = vocab.CHOICE_LABELS.get(question_id)
    if labels is None:
        raise EvalError(f"{question_id!r} is not a Choice question")
    if not isinstance(answer, Mapping):
        raise EvalError(f"answer for {question_id!r} is {type(answer).__name__}, expected a label -> ppm mapping")
    top = answer.get("top")
    if isinstance(top, str):
        return top
    probs = answer.get("probs", answer)
    if not isinstance(probs, Mapping):
        raise EvalError(f"answer for {question_id!r} has no label -> ppm mapping")
    best_label: str | None = None
    best_value = float("-inf")
    for label in labels:  # authored order: the first maximum wins
        value = probs.get(label)
        if value is None:
            continue
        numeric = float(value)
        if numeric > best_value:
            best_value = numeric
            best_label = label
    return best_label


def _read_entry_decisions(path: Path) -> dict[tuple[date, str], _EntryDecision]:
    """Entry DECISION entries keyed by `(session, underlying)`; the base-variant `entry` request supplies the Choices."""
    out: dict[tuple[date, str], _EntryDecision] = {}
    with closing(_connect(path)) as connection:
        rows = connection.execute("SELECT seq, session, payload FROM ledger WHERE kind = 'decision' ORDER BY seq").fetchall()
    for row in rows:
        payload = json.loads(str(row["payload"]))
        if payload.get("kind") != "entry":
            continue  # manage decisions are path dependent and are not part of the agreement report
        rules = payload.get("rules")
        if not isinstance(rules, Mapping):
            raise EvalError(f"{path}: entry DECISION at seq {row['seq']} has no rules block")
        underlying = rules.get("underlying") or payload.get("subject_alias")
        if not isinstance(underlying, str) or not underlying:
            raise EvalError(f"{path}: entry DECISION at seq {row['seq']} names no underlying")
        # 12.9(a) is *about* `action`, and 2.6 makes it a non-optional field of `EntryDecision`: a payload without it is
        # a malformed store, not a missing value.  Defaulting it to "" would let two malformed stores "agree".
        action = rules.get("action")
        if not isinstance(action, str) or not action:
            raise EvalError(f"{path}: entry DECISION at seq {row['seq']} names no action")
        session = date.fromisoformat(str(row["session"]))
        kind = rules.get("kind")
        labels: dict[str, str] = {}
        for request in payload.get("requests") or ():
            if not isinstance(request, Mapping):
                continue
            if request.get("request_kind") != "entry" or request.get("variant") != "base":
                continue
            answers = request.get("answers")
            if not isinstance(answers, Mapping):
                continue
            for question_id in AGREEMENT_CHOICES:
                if question_id not in answers:
                    continue
                label = _top_label(question_id, answers[question_id])
                if label is not None:
                    labels[question_id] = label
        out[(session, underlying)] = _EntryDecision(
            session=session,
            underlying=underlying,
            action=action,
            kind=None if kind is None else str(kind),
            top_labels=labels,
        )
    return out


def _read_forecasts(path: Path, questions: frozenset[str]) -> dict[tuple[str, bool], tuple[str, float | None]]:
    """FORECAST entries of the wanted questions keyed by `(event_key, with_text)` -> `(question_id, p)`.

    A NULL `p_ppm` (a MISSING forecast, 6.4) becomes `None`: such a pair is counted and excluded, never imputed here -
    the pre-registered imputation rule belongs to the scored tables, not to a model-agreement report.
    """
    out: dict[tuple[str, bool], tuple[str, float | None]] = {}
    with closing(_connect(path)) as connection:
        rows = connection.execute("SELECT seq, payload FROM ledger WHERE kind = 'forecast' ORDER BY seq").fetchall()
    for row in rows:
        payload = json.loads(str(row["payload"]))
        question_id = payload.get("question_id")
        if not isinstance(question_id, str) or question_id not in questions:
            continue
        event_key = payload.get("event_key")
        if not isinstance(event_key, str) or not event_key:
            raise EvalError(f"{path}: FORECAST at seq {row['seq']} has no event_key")
        p_ppm = payload.get("p_ppm")
        value = None if p_ppm is None else float(p_ppm) / _PPM
        out[(event_key, bool(payload.get("with_text")))] = (question_id, value)
    return out


# ======================================================================================================================
# the report
# ======================================================================================================================


@dataclass(frozen=True, slots=True)
class DpStats:
    """Mean absolute probability difference over joined FORECAST pairs (12.9b)."""

    n: int
    mean_abs_dp: float
    by_question: Mapping[str, float]
    n_by_question: Mapping[str, int]
    n_missing: int  # pairs excluded because one side was a MISSING forecast (NULL p_ppm)


@dataclass(frozen=True, slots=True)
class AgreementReport:
    """The three pre-registered statistics of 12.9 plus the unmatched rows, which are listed and never imputed."""

    old_store: str
    new_store: str
    old_namespace: str
    new_namespace: str
    primary_family: tuple[str, ...]

    n_sessions_joined: int
    n_gate_pairs: int
    gate_agreement_rate: float
    action_agreement_rate: float
    kind_agreement_rate: float

    dp_pooled: DpStats
    dp_with_text: DpStats
    dp_without_text: DpStats

    top_label_agreement: Mapping[str, float]
    n_top_label_pairs: Mapping[str, int]
    top_label_agreement_pooled: float

    sessions_only_old: tuple[date, ...]
    sessions_only_new: tuple[date, ...]
    pairs_only_old: tuple[tuple[date, str], ...]
    pairs_only_new: tuple[tuple[date, str], ...]
    forecasts_only_old: int
    forecasts_only_new: int

    def to_dict(self) -> dict[str, Any]:
        """The `--json` shape of `jevbot eval model-agreement` (14)."""
        return {
            "old_store": self.old_store,
            "new_store": self.new_store,
            "old_namespace": self.old_namespace,
            "new_namespace": self.new_namespace,
            "primary_family": list(self.primary_family),
            "n_sessions_joined": self.n_sessions_joined,
            "gate_decision_agreement_rate": self.gate_agreement_rate,
            "action_agreement_rate": self.action_agreement_rate,
            "kind_agreement_rate": self.kind_agreement_rate,
            "n_gate_pairs": self.n_gate_pairs,
            "mean_abs_dp": {
                "pooled": _dp_dict(self.dp_pooled),
                "with_text": _dp_dict(self.dp_with_text),
                "without_text": _dp_dict(self.dp_without_text),
            },
            "top_label_agreement": dict(self.top_label_agreement),
            "top_label_agreement_pooled": self.top_label_agreement_pooled,
            "n_top_label_pairs": dict(self.n_top_label_pairs),
            "unmatched": {
                "sessions_only_old": [d.isoformat() for d in self.sessions_only_old],
                "sessions_only_new": [d.isoformat() for d in self.sessions_only_new],
                "pairs_only_old": [[d.isoformat(), u] for d, u in self.pairs_only_old],
                "pairs_only_new": [[d.isoformat(), u] for d, u in self.pairs_only_new],
                "forecasts_only_old": self.forecasts_only_old,
                "forecasts_only_new": self.forecasts_only_new,
            },
        }


def _dp_dict(stats: DpStats) -> dict[str, Any]:
    return {
        "n": stats.n,
        "mean_abs_dp": stats.mean_abs_dp,
        "by_question": dict(stats.by_question),
        "n_by_question": dict(stats.n_by_question),
        "n_missing": stats.n_missing,
    }


def _dp_stats(pairs: Sequence[tuple[str, float]], missing: int) -> DpStats:
    by_question: dict[str, list[float]] = {}
    for question_id, value in pairs:
        by_question.setdefault(question_id, []).append(value)
    means = {qid: sum(values) / len(values) for qid, values in by_question.items()}
    counts = {qid: len(values) for qid, values in by_question.items()}
    pooled = sum(value for _, value in pairs) / len(pairs) if pairs else float("nan")
    return DpStats(n=len(pairs), mean_abs_dp=pooled, by_question=means, n_by_question=counts, n_missing=missing)


def model_agreement(old: RunStoreRef, new: RunStoreRef, primary_family: Sequence[str]) -> AgreementReport:
    """The forced-model-change agreement report of 12.9.

    Asserts the two namespaces DIFFER (`EvalError` otherwise: two runs of one namespace are one experiment, and the
    report would be meaningless); joins on `(session, underlying)` for entry DECISION entries (base variant) and on
    `(event_key, with_text)` for FORECAST entries; sessions present in only one store are listed, not imputed.
    """
    family = tuple(str(q) for q in primary_family)
    if not family:
        raise EvalError("the primary family is empty")
    unknown = [q for q in family if q not in vocab.ALL_QUESTION_IDS and q not in vocab.DERIVED_EVAL_IDS]
    if unknown:
        raise EvalError(f"unknown question ids in the primary family: {unknown}")

    old_path = _store_path(old)
    new_path = _store_path(new)
    old_namespace = read_namespace(old_path)
    new_namespace = read_namespace(new_path)
    if old_namespace == new_namespace:
        raise EvalError(
            f"model-agreement compares two namespaces; both stores are {old_namespace!r} "
            "(a forced model change creates a NEW namespace, 12.9)"
        )

    old_decisions = _read_entry_decisions(old_path)
    new_decisions = _read_entry_decisions(new_path)
    joined_keys = sorted(set(old_decisions) & set(new_decisions))
    only_old = sorted(set(old_decisions) - set(new_decisions))
    only_new = sorted(set(new_decisions) - set(old_decisions))

    old_sessions = {key[0] for key in old_decisions}
    new_sessions = {key[0] for key in new_decisions}

    action_hits = 0
    kind_hits = 0
    gate_hits = 0
    label_hits: dict[str, int] = dict.fromkeys(AGREEMENT_CHOICES, 0)
    label_pairs: dict[str, int] = dict.fromkeys(AGREEMENT_CHOICES, 0)
    for key in joined_keys:
        a = old_decisions[key]
        b = new_decisions[key]
        same_action = a.action == b.action
        same_kind = a.kind == b.kind
        action_hits += int(same_action)
        kind_hits += int(same_kind)
        gate_hits += int(same_action and same_kind)
        for question_id in AGREEMENT_CHOICES:
            left = a.top_labels.get(question_id)
            right = b.top_labels.get(question_id)
            if left is None or right is None:
                continue
            label_pairs[question_id] += 1
            label_hits[question_id] += int(left == right)

    n_pairs = len(joined_keys)
    questions = frozenset(family)
    old_forecasts = _read_forecasts(old_path, questions)
    new_forecasts = _read_forecasts(new_path, questions)
    joined_forecasts = set(old_forecasts) & set(new_forecasts)

    pooled: list[tuple[str, float]] = []
    with_text: list[tuple[str, float]] = []
    without_text: list[tuple[str, float]] = []
    missing_pooled = 0
    missing_with = 0
    missing_without = 0
    for event_key, flag in sorted(joined_forecasts):
        question_id, p_old = old_forecasts[(event_key, flag)]
        other_question, p_new = new_forecasts[(event_key, flag)]
        if question_id != other_question:
            raise EvalError(f"event_key {event_key!r} names {question_id!r} in one store and {other_question!r} in the other")
        if p_old is None or p_new is None:
            missing_pooled += 1
            if flag:
                missing_with += 1
            else:
                missing_without += 1
            continue
        item = (question_id, abs(p_new - p_old))
        pooled.append(item)
        (with_text if flag else without_text).append(item)

    label_rates = {q: (label_hits[q] / label_pairs[q] if label_pairs[q] else float("nan")) for q in AGREEMENT_CHOICES}
    total_label_pairs = sum(label_pairs.values())
    pooled_label_rate = sum(label_hits.values()) / total_label_pairs if total_label_pairs else float("nan")

    return AgreementReport(
        old_store=str(old_path),
        new_store=str(new_path),
        old_namespace=old_namespace,
        new_namespace=new_namespace,
        primary_family=family,
        n_sessions_joined=len(old_sessions & new_sessions),
        n_gate_pairs=n_pairs,
        gate_agreement_rate=gate_hits / n_pairs if n_pairs else float("nan"),
        action_agreement_rate=action_hits / n_pairs if n_pairs else float("nan"),
        kind_agreement_rate=kind_hits / n_pairs if n_pairs else float("nan"),
        dp_pooled=_dp_stats(pooled, missing_pooled),
        dp_with_text=_dp_stats(with_text, missing_with),
        dp_without_text=_dp_stats(without_text, missing_without),
        top_label_agreement=label_rates,
        n_top_label_pairs=dict(label_pairs),
        top_label_agreement_pooled=pooled_label_rate,
        sessions_only_old=tuple(sorted(old_sessions - new_sessions)),
        sessions_only_new=tuple(sorted(new_sessions - old_sessions)),
        pairs_only_old=tuple(only_old),
        pairs_only_new=tuple(only_new),
        forecasts_only_old=len(set(old_forecasts) - set(new_forecasts)),
        forecasts_only_new=len(set(new_forecasts) - set(old_forecasts)),
    )
