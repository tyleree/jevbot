"""Unit tests for `jevbot.eval.agreement` (DESIGN.md 12.9, 15.1 `eval/*` row; WP07).

The acceptance cases of 12.9 / 15.1: identical stores give `1.0 / 0 / 1.0`, a store with shifted answers gives the
hand-computed values, two runs of the SAME namespace are refused, and unmatched sessions are listed.

The run stores are built here from **raw SQL** in the schema of 13.4 (ledger table, append-only triggers, meta) with
payloads shaped by 2.11 - the report reads run stores only, so it needs no engine.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from jevbot import vocab
from jevbot.errors import EvalError
from jevbot.eval.agreement import AGREEMENT_CHOICES, model_agreement, read_namespace

EPOCH = date(2026, 3, 2)
PRIMARY: tuple[str, ...] = ("eval.down_1em_1s", "eval.up_1em_1s")

_SCHEMA = """
CREATE TABLE ledger (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, session TEXT NOT NULL, as_of TEXT NOT NULL,
  payload TEXT NOT NULL, prev_hash TEXT NOT NULL, hash TEXT NOT NULL UNIQUE);
CREATE INDEX ledger_kind_session ON ledger(kind, session);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TRIGGER ledger_no_update BEFORE UPDATE ON ledger BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
CREATE TRIGGER ledger_no_delete BEFORE DELETE ON ledger BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
"""

GENESIS = "0" * 64


def _day(i: int) -> date:
    return EPOCH + timedelta(days=i)


def _as_of(session: date) -> str:
    return datetime(session.year, session.month, session.day, 20, 0, tzinfo=UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Decision:
    """One entry DECISION as 2.11 shapes it."""

    session: date
    underlying: str
    action: str
    kind: str | None
    labels: Mapping[str, str]
    decision_kind: str = "entry"
    variant: str = "base"
    request_kind: str = "entry"


@dataclass(frozen=True)
class Forecast:
    """One FORECAST entry (2.7 / 2.11); `p_ppm=None` is a MISSING forecast (6.4)."""

    session: date
    event_key: str
    question_id: str
    with_text: bool
    p_ppm: int | None


def _choice_answer(question_id: str, top: str) -> dict[str, int]:
    """A Choice answer as the DECISION payload records it: `{label: ppm}` summing to 1e6 (2.11)."""
    labels = vocab.CHOICE_LABELS[question_id]
    if top not in labels:
        raise AssertionError(f"{top!r} is not a label of {question_id!r}")
    rest = (1_000_000 - 700_000) // (len(labels) - 1)
    answer = dict.fromkeys(labels, rest)
    answer[top] = 1_000_000 - rest * (len(labels) - 1)
    return answer


def build_store(
    path: Path,
    *,
    namespace: str,
    decisions: Sequence[Decision] = (),
    forecasts: Sequence[Forecast] = (),
    write_meta: bool = True,
) -> Path:
    """A minimal `run.sqlite` in the 13.4 schema, filled from raw SQL."""
    connection = sqlite3.connect(path)
    try:
        connection.executescript(_SCHEMA)
        seq = 0
        prev = GENESIS

        def append(kind: str, session: date, payload: dict[str, Any]) -> None:
            nonlocal seq, prev
            seq += 1
            text = json.dumps(payload, sort_keys=True)
            digest = hashlib.sha256(f"{prev}\n{seq}\n{kind}\n{session}\n{text}".encode()).hexdigest()
            connection.execute(
                "INSERT INTO ledger (seq, kind, session, as_of, payload, prev_hash, hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (seq, kind, session.isoformat(), _as_of(session), text, prev, digest),
            )
            prev = digest

        first = min([d.session for d in decisions] + [f.session for f in forecasts], default=_day(0))
        append("run_start", first, {"mode": "backtest", "namespace": namespace, "model": "jev-1.13.0", "purpose": "validate"})
        for decision in decisions:
            append(
                "decision",
                decision.session,
                {
                    "decision_id": f"{decision.session.isoformat()}-{decision.underlying}",
                    "kind": decision.decision_kind,
                    "subject_alias": "the_underlying",
                    "text": "on",
                    "requests": [
                        {
                            "request_kind": decision.request_kind,
                            "variant": decision.variant,
                            "state_hash": "0" * 8,
                            "question_set_id": "entry.v1",
                            "answers": {qid: _choice_answer(qid, top) for qid, top in decision.labels.items()},
                            "error": None,
                        }
                    ],
                    "rules": {
                        "underlying": decision.underlying,
                        "decision_id": f"{decision.session.isoformat()}-{decision.underlying}",
                        "action": decision.action,
                        "kind": decision.kind,
                        "reasons": [],
                    },
                    "facts": {},
                    "tier": "B",
                },
            )
        for forecast in forecasts:
            append(
                "forecast",
                forecast.session,
                {
                    "forecast_id": f"{forecast.event_key}-{int(forecast.with_text)}",
                    "event_key": forecast.event_key,
                    "question_id": forecast.question_id,
                    "with_text": forecast.with_text,
                    "p_ppm": forecast.p_ppm,
                    "missing_reason": None if forecast.p_ppm is not None else "DeciderTransportError",
                    "tier": "B",
                },
            )
        if write_meta:
            connection.execute("INSERT INTO meta (key, value) VALUES ('namespace', ?)", (namespace,))
        connection.commit()
    finally:
        connection.close()
    return path


# ======================================================================================================================
# fixtures: an identical pair and a hand-computed shifted pair
# ======================================================================================================================

_LABELS_A: dict[str, str] = {
    "regime.market": "range_bound_calm",
    "under.direction": "bullish",
    "vol.stance": "sell_premium",
    "fit.structure_family": "put_credit_spread",
}
_LABELS_B: dict[str, str] = {
    "regime.market": "trending_up_calm",
    "under.direction": "bearish",
    "vol.stance": "sell_premium",
    "fit.structure_family": "iron_condor",
}


def _sample_decisions() -> list[Decision]:
    return [
        Decision(_day(0), "SPY", "enter", "iron_condor", _LABELS_A),
        Decision(_day(0), "QQQ", "no_trade", None, _LABELS_B),
        Decision(_day(1), "SPY", "enter", "put_credit_spread", _LABELS_A),
        Decision(_day(1), "QQQ", "enter", "long_call", _LABELS_B),
    ]


def _sample_forecasts() -> list[Forecast]:
    out: list[Forecast] = []
    for i, session in enumerate((_day(0), _day(1))):
        for underlying in ("SPY", "QQQ"):
            for question_id in PRIMARY:
                key = f"ek-{i}-{underlying}-{question_id}"
                for with_text in (True, False):
                    out.append(Forecast(session, key, question_id, with_text, 100_000 + 1_000 * i))
    return out


@pytest.fixture
def identical_pair(tmp_path: Path) -> tuple[Path, Path]:
    decisions = _sample_decisions()
    forecasts = _sample_forecasts()
    old = build_store(tmp_path / "old.sqlite", namespace="exp:jev-1.13.0:g0", decisions=decisions, forecasts=forecasts)
    new = build_store(tmp_path / "new.sqlite", namespace="exp2:jev-1.14.0:g0", decisions=decisions, forecasts=forecasts)
    return old, new


# ======================================================================================================================
# identical stores => 1.0 / 0 / 1.0
# ======================================================================================================================


def test_identical_stores_agree_completely(identical_pair: tuple[Path, Path]) -> None:
    """12.9 / 15.1: identical stores => gate 1.0, mean |dp| 0, top-label agreement 1.0."""
    report = model_agreement(*identical_pair, PRIMARY)
    assert report.gate_agreement_rate == 1.0
    assert report.action_agreement_rate == 1.0
    assert report.kind_agreement_rate == 1.0
    assert report.dp_pooled.mean_abs_dp == 0.0
    assert report.dp_with_text.mean_abs_dp == 0.0
    assert report.dp_without_text.mean_abs_dp == 0.0
    assert report.top_label_agreement_pooled == 1.0
    assert set(report.top_label_agreement) == set(AGREEMENT_CHOICES)
    assert all(rate == 1.0 for rate in report.top_label_agreement.values())
    assert report.n_gate_pairs == 4
    assert report.n_sessions_joined == 2
    assert report.sessions_only_old == () and report.sessions_only_new == ()
    assert report.dp_pooled.n == 16  # 2 sessions x 2 underlyings x 2 questions x with/without text
    assert report.dp_with_text.n == 8 and report.dp_without_text.n == 8


def test_identical_stores_report_the_two_namespaces(identical_pair: tuple[Path, Path]) -> None:
    report = model_agreement(*identical_pair, PRIMARY)
    assert report.old_namespace == "exp:jev-1.13.0:g0"
    assert report.new_namespace == "exp2:jev-1.14.0:g0"
    assert report.primary_family == PRIMARY


# ======================================================================================================================
# the hand-computed shifted case
# ======================================================================================================================


def test_shifted_store_gives_the_hand_computed_statistics(tmp_path: Path) -> None:
    old_decisions = [
        Decision(_day(0), "SPY", "enter", "iron_condor", _LABELS_A),
        Decision(_day(0), "QQQ", "no_trade", None, _LABELS_B),
        Decision(_day(1), "SPY", "enter", "put_credit_spread", _LABELS_A),
        Decision(_day(1), "QQQ", "enter", "long_call", _LABELS_B),
    ]
    # SPY on day 1 keeps its action but changes structure; QQQ on day 0 changes both.
    new_decisions = [
        Decision(_day(0), "SPY", "enter", "iron_condor", _LABELS_A),
        Decision(_day(0), "QQQ", "enter", "long_call", {**_LABELS_B, "regime.market": "disorderly_selloff"}),
        Decision(_day(1), "SPY", "enter", "call_credit_spread", {**_LABELS_A, "under.direction": "bearish"}),
        Decision(_day(1), "QQQ", "enter", "long_call", {**_LABELS_B, "fit.structure_family": "long_call"}),
    ]
    old_forecasts = [
        Forecast(_day(0), "ek1", "eval.down_1em_1s", True, 100_000),
        Forecast(_day(0), "ek1", "eval.down_1em_1s", False, 100_000),
        Forecast(_day(0), "ek2", "eval.up_1em_1s", True, 200_000),
        Forecast(_day(0), "ek2", "eval.up_1em_1s", False, 200_000),
    ]
    new_forecasts = [
        Forecast(_day(0), "ek1", "eval.down_1em_1s", True, 150_000),
        Forecast(_day(0), "ek1", "eval.down_1em_1s", False, 120_000),
        Forecast(_day(0), "ek2", "eval.up_1em_1s", True, 200_000),
        Forecast(_day(0), "ek2", "eval.up_1em_1s", False, 260_000),
    ]
    old = build_store(tmp_path / "old.sqlite", namespace="ns_old", decisions=old_decisions, forecasts=old_forecasts)
    new = build_store(tmp_path / "new.sqlite", namespace="ns_new", decisions=new_decisions, forecasts=new_forecasts)

    report = model_agreement(old, new, PRIMARY)

    # (a) gate decisions: only (d0, SPY) and (d1, QQQ) match on BOTH action and kind
    assert report.gate_agreement_rate == pytest.approx(2 / 4)
    assert report.action_agreement_rate == pytest.approx(3 / 4)  # only (d0, QQQ) changed its action
    assert report.kind_agreement_rate == pytest.approx(2 / 4)

    # (b) mean |dp|: 0.05, 0.02, 0.00, 0.06
    assert report.dp_pooled.n == 4
    assert report.dp_pooled.mean_abs_dp == pytest.approx((0.05 + 0.02 + 0.00 + 0.06) / 4)
    assert report.dp_with_text.mean_abs_dp == pytest.approx((0.05 + 0.00) / 2)
    assert report.dp_without_text.mean_abs_dp == pytest.approx((0.02 + 0.06) / 2)
    assert report.dp_pooled.by_question["eval.down_1em_1s"] == pytest.approx((0.05 + 0.02) / 2)
    assert report.dp_pooled.by_question["eval.up_1em_1s"] == pytest.approx((0.00 + 0.06) / 2)
    assert report.dp_pooled.n_by_question == {"eval.down_1em_1s": 2, "eval.up_1em_1s": 2}

    # (c) top labels: regime.market differs once, under.direction once, vol.stance never, fit.structure_family once
    assert report.top_label_agreement["regime.market"] == pytest.approx(3 / 4)
    assert report.top_label_agreement["under.direction"] == pytest.approx(3 / 4)
    assert report.top_label_agreement["vol.stance"] == pytest.approx(4 / 4)
    assert report.top_label_agreement["fit.structure_family"] == pytest.approx(3 / 4)
    assert report.n_top_label_pairs == dict.fromkeys(AGREEMENT_CHOICES, 4)
    assert report.top_label_agreement_pooled == pytest.approx((3 + 3 + 4 + 3) / 16)


# ======================================================================================================================
# refusals and unmatched rows
# ======================================================================================================================


def test_two_runs_of_the_same_namespace_are_refused(tmp_path: Path) -> None:
    """12.9 / 15.1: `model_agreement` asserts the two namespaces DIFFER."""
    decisions = _sample_decisions()
    old = build_store(tmp_path / "old.sqlite", namespace="same:ns:g0", decisions=decisions)
    new = build_store(tmp_path / "new.sqlite", namespace="same:ns:g0", decisions=decisions)
    with pytest.raises(EvalError, match="namespace"):
        model_agreement(old, new, PRIMARY)


def test_sessions_present_in_only_one_store_are_listed_not_imputed(tmp_path: Path) -> None:
    shared = _sample_decisions()
    old = build_store(
        tmp_path / "old.sqlite",
        namespace="ns_old",
        decisions=[*shared, Decision(_day(2), "SPY", "enter", "long_put", _LABELS_A)],
    )
    new = build_store(
        tmp_path / "new.sqlite",
        namespace="ns_new",
        decisions=[*shared, Decision(_day(3), "IWM", "no_trade", None, _LABELS_B)],
    )
    report = model_agreement(old, new, PRIMARY)
    assert report.sessions_only_old == (_day(2),)
    assert report.sessions_only_new == (_day(3),)
    assert report.pairs_only_old == ((_day(2), "SPY"),)
    assert report.pairs_only_new == ((_day(3), "IWM"),)
    assert report.n_gate_pairs == 4  # the unmatched pairs never enter the rate
    assert report.n_sessions_joined == 2


def test_missing_forecasts_are_counted_and_excluded(tmp_path: Path) -> None:
    """A NULL `p_ppm` (a MISSING forecast, 6.4) is never imputed into an agreement number."""
    old = build_store(
        tmp_path / "old.sqlite",
        namespace="ns_old",
        forecasts=[
            Forecast(_day(0), "ek1", "eval.down_1em_1s", True, 100_000),
            Forecast(_day(0), "ek2", "eval.up_1em_1s", True, 200_000),
        ],
    )
    new = build_store(
        tmp_path / "new.sqlite",
        namespace="ns_new",
        forecasts=[
            Forecast(_day(0), "ek1", "eval.down_1em_1s", True, None),
            Forecast(_day(0), "ek2", "eval.up_1em_1s", True, 250_000),
        ],
    )
    report = model_agreement(old, new, PRIMARY)
    assert report.dp_pooled.n == 1
    assert report.dp_pooled.n_missing == 1
    assert report.dp_pooled.mean_abs_dp == pytest.approx(0.05)


def test_forecasts_present_in_only_one_store_are_counted(tmp_path: Path) -> None:
    old = build_store(
        tmp_path / "old.sqlite",
        namespace="ns_old",
        forecasts=[
            Forecast(_day(0), "ek1", "eval.down_1em_1s", True, 100_000),
            Forecast(_day(0), "only_old", "eval.up_1em_1s", True, 300_000),
        ],
    )
    new = build_store(
        tmp_path / "new.sqlite",
        namespace="ns_new",
        forecasts=[
            Forecast(_day(0), "ek1", "eval.down_1em_1s", True, 100_000),
            Forecast(_day(0), "only_new_a", "eval.up_1em_1s", True, 300_000),
            Forecast(_day(0), "only_new_b", "eval.up_1em_1s", False, 300_000),
        ],
    )
    report = model_agreement(old, new, PRIMARY)
    assert report.forecasts_only_old == 1
    assert report.forecasts_only_new == 2
    assert report.dp_pooled.n == 1


def test_only_primary_family_forecasts_enter_the_statistic(tmp_path: Path) -> None:
    extra = Forecast(_day(0), "ek_other", "eval.inside_1em_5s", True, 900_000)
    old = build_store(
        tmp_path / "old.sqlite",
        namespace="ns_old",
        forecasts=[Forecast(_day(0), "ek1", "eval.down_1em_1s", True, 100_000), extra],
    )
    new = build_store(
        tmp_path / "new.sqlite",
        namespace="ns_new",
        forecasts=[
            Forecast(_day(0), "ek1", "eval.down_1em_1s", True, 140_000),
            Forecast(_day(0), "ek_other", "eval.inside_1em_5s", True, 100_000),
        ],
    )
    report = model_agreement(old, new, PRIMARY)
    assert report.dp_pooled.n == 1
    assert report.dp_pooled.mean_abs_dp == pytest.approx(0.04)


def test_manage_decisions_and_non_base_variants_are_ignored(tmp_path: Path) -> None:
    """The report joins ENTRY decisions of the BASE variant only (12.9)."""
    decisions = [
        Decision(_day(0), "SPY", "enter", "iron_condor", _LABELS_A),
        Decision(_day(0), "IWM", "close", None, {}, decision_kind="manage"),
    ]
    shifted = [
        Decision(_day(0), "SPY", "enter", "iron_condor", _LABELS_B, variant="key_perm"),
        Decision(_day(0), "IWM", "hold", None, {}, decision_kind="manage"),
    ]
    old = build_store(tmp_path / "old.sqlite", namespace="ns_old", decisions=decisions)
    new = build_store(tmp_path / "new.sqlite", namespace="ns_new", decisions=shifted)
    report = model_agreement(old, new, PRIMARY)
    assert report.n_gate_pairs == 1  # the manage decision is not joined
    assert report.gate_agreement_rate == 1.0
    # the new store's Choices sit on a non-base variant, so no top-label pair exists at all
    assert report.n_top_label_pairs == dict.fromkeys(AGREEMENT_CHOICES, 0)
    assert all(rate != rate for rate in report.top_label_agreement.values())  # NaN


def test_unknown_primary_question_is_refused(tmp_path: Path) -> None:
    old = build_store(tmp_path / "old.sqlite", namespace="ns_old", decisions=_sample_decisions())
    new = build_store(tmp_path / "new.sqlite", namespace="ns_new", decisions=_sample_decisions())
    with pytest.raises(EvalError, match="unknown question"):
        model_agreement(old, new, ("eval.not_a_question",))
    with pytest.raises(EvalError, match="primary family is empty"):
        model_agreement(old, new, ())


def test_a_missing_run_store_is_refused(tmp_path: Path) -> None:
    old = build_store(tmp_path / "old.sqlite", namespace="ns_old", decisions=_sample_decisions())
    with pytest.raises(EvalError, match="not found"):
        model_agreement(old, tmp_path / "absent.sqlite", PRIMARY)


# ======================================================================================================================
# store plumbing
# ======================================================================================================================


def test_namespace_falls_back_to_run_start_when_meta_is_empty(tmp_path: Path) -> None:
    path = build_store(tmp_path / "no_meta.sqlite", namespace="ns_from_run_start", decisions=_sample_decisions(), write_meta=False)
    assert read_namespace(path) == "ns_from_run_start"
    assert read_namespace(str(path)) == "ns_from_run_start"


def test_a_run_store_object_exposing_path_is_accepted(identical_pair: tuple[Path, Path]) -> None:
    """`eval.load`'s `RunStore` is accepted through its `.path` property, as are plain paths and strings."""

    class _Store:
        def __init__(self, path: Path) -> None:
            self._path = path

        @property
        def path(self) -> Path:
            return self._path

    old, new = identical_pair
    report = model_agreement(_Store(old), _Store(new), PRIMARY)
    assert report.gate_agreement_rate == 1.0
    assert model_agreement(str(old), new, PRIMARY).gate_agreement_rate == 1.0


def test_the_report_never_writes_to_either_store(identical_pair: tuple[Path, Path]) -> None:
    old, new = identical_pair
    before = [p.read_bytes() for p in (old, new)]
    model_agreement(old, new, PRIMARY)
    assert [p.read_bytes() for p in (old, new)] == before


def test_to_dict_is_json_serialisable_and_carries_every_statistic(identical_pair: tuple[Path, Path]) -> None:
    payload = model_agreement(*identical_pair, PRIMARY).to_dict()
    text = json.dumps(payload)  # the `--json` output of `jevbot eval model-agreement`
    assert "gate_decision_agreement_rate" in text
    assert payload["mean_abs_dp"]["with_text"]["mean_abs_dp"] == 0.0
    assert payload["mean_abs_dp"]["without_text"]["mean_abs_dp"] == 0.0
    assert payload["top_label_agreement_pooled"] == 1.0
    assert payload["unmatched"]["sessions_only_old"] == []


def test_top_label_ties_follow_the_authored_option_order(tmp_path: Path) -> None:
    """2.5: a Choice's `top` is the argmax with ties broken by the AUTHORED option order."""
    labels = vocab.CHOICE_LABELS["vol.stance"]
    tied = dict.fromkeys(labels, 250_000)

    def _store(path: Path, namespace: str, answer: dict[str, int]) -> Path:
        decision = Decision(_day(0), "SPY", "enter", "iron_condor", {})
        built = build_store(path, namespace=namespace, decisions=[decision])
        connection = sqlite3.connect(built)
        try:
            row = connection.execute("SELECT seq, payload FROM ledger WHERE kind = 'decision'").fetchone()
            payload = json.loads(row[1])
            payload["requests"][0]["answers"]["vol.stance"] = answer
            # the ledger is append-only, so rebuild the file rather than update it
            connection.close()
            built.unlink()
            connection = sqlite3.connect(built)
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT INTO ledger (seq, kind, session, as_of, payload, prev_hash, hash) VALUES (1, 'decision', ?, ?, ?, ?, ?)",
                (_day(0).isoformat(), _as_of(_day(0)), json.dumps(payload, sort_keys=True), GENESIS, namespace),
            )
            connection.execute("INSERT INTO meta (key, value) VALUES ('namespace', ?)", (namespace,))
            connection.commit()
        finally:
            connection.close()
        return built

    old = _store(tmp_path / "old.sqlite", "ns_old", tied)
    new = _store(tmp_path / "new.sqlite", "ns_new", {**tied, labels[0]: 250_000})
    report = model_agreement(old, new, PRIMARY)
    assert report.top_label_agreement["vol.stance"] == 1.0  # both resolve to the first authored label


def test_the_append_only_triggers_of_the_fixture_store_really_bite(tmp_path: Path) -> None:
    """The fixture reproduces 13.4 faithfully, so a test that mutated a store would fail loudly."""
    path = build_store(tmp_path / "old.sqlite", namespace="ns_old", decisions=_sample_decisions())
    connection = sqlite3.connect(path)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            connection.execute("UPDATE ledger SET session = '2030-01-01' WHERE seq = 1")
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            connection.execute("DELETE FROM ledger WHERE seq = 1")
    finally:
        connection.close()
