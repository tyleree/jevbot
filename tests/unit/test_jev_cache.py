"""`jev/cache.py`: the immutable, content-addressed decision cache (DESIGN.md 13.3, 3.3, D8).

The cache is the project's evidence store: every replay, sweep, baseline and ablation arm reads back exactly the bytes a
recorded run paid for. The tests below pin the properties that make that true - the key formula, `(namespace, key)`
separation, all-or-nothing hits and puts, first-write-wins with a `nondeterminism` row, the model binding, the immutability
triggers and a stable manifest hash - plus the `cache stats` / `cache verify` commands that read them.
"""

import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from jevbot import canon, ids
from jevbot import questions as questions_module
from jevbot.cli import jev_cmds
from jevbot.cli.main import GlobalOptions
from jevbot.errors import ConfigError, InvariantError, ModelMismatchError
from jevbot.jev.cache import SqliteDecisionCache
from jevbot.types import CachedAnswer, RequestKind
from tests.fixtures.jev_transport import NAMESPACE, SESSION, sample_state

MODEL = "jev-1.13.0"
RELEASE = date(2026, 9, 15)
OTHER_NAMESPACE = ids.namespace("other", MODEL, 0)


@pytest.fixture
def cache(tmp_path: Path) -> SqliteDecisionCache:
    store = SqliteDecisionCache(tmp_path / "cache" / "decisions.sqlite")
    store.ensure_namespace(NAMESPACE, MODEL, RELEASE, refresh=False)
    return store


def make_rows(
    namespace: str = NAMESPACE,
    *,
    model: str = MODEL,
    response_model: str | None = None,
    state: dict[str, Any] | None = None,
    answer: float = 0.5,
    kind: RequestKind = RequestKind.ENTRY,
    variant: str = "base",
) -> tuple[list[CachedAnswer], str, str]:
    """One whole `entry.v1` request: its rows, its state JSON and its questions JSON (as `put_request` takes them)."""
    body = state if state is not None else sample_state(RequestKind.ENTRY)
    questions = questions_module.question_set("entry.v1")
    qset_hash = questions_module.QUESTION_SET_HASHES["entry.v1"]
    state_hash = canon.sha256_hex(canon.dumps_ordered(body))
    rows = [
        CachedAnswer(
            key=canon.cache_key(model, body, qset_hash, question),
            namespace=namespace,
            requested_model=model,
            response_model=response_model or model,
            question_set_id="entry.v1",
            question_set_hash=qset_hash,
            state_hash=state_hash,
            question_hash=questions_module.QUESTION_HASHES[qid],
            question_id=qid,
            request_kind=kind.value,
            variant=variant,
            answer_json=json.dumps({"noul": answer, "type": "noul"}, separators=(",", ":"), sort_keys=True),
            request_id="req-1",
            input_tokens=1234,
            latency_ms=42,
            sdk_version="0.6.0",
            created_at=datetime(2026, 9, 17, 20, 0, tzinfo=UTC),
        )
        for qid, question in questions.items()
    ]
    return rows, canon.dumps_ordered(body), canon.dumps_ordered(questions)


def put(cache: SqliteDecisionCache, rows: Sequence[CachedAnswer], state_json: str, questions_json: str, *, underlying: str = "SPY") -> None:
    cache.put_request(rows[0].namespace, rows, state_json, questions_json, (SESSION, underlying, rows[0].request_kind, rows[0].variant))


# ======================================================================================================================
# Keys and namespaces
# ======================================================================================================================


def test_the_key_is_the_d8_content_key_and_the_golden_matches(cache: SqliteDecisionCache) -> None:
    rows, state_json, questions_json = make_rows()
    put(cache, rows, state_json, questions_json)
    golden = (Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "cache_key.txt").read_text(encoding="utf-8")
    stored = {row.question_id: row.key for row in rows}
    for line in golden.splitlines():
        if line.startswith("#"):
            continue
        qid, key = line.split()
        assert stored[qid] == key, f"the stored key of {qid} drifted from golden/cache_key.txt"
    assert len(set(stored.values())) == len(stored)


def test_the_same_key_in_two_namespaces_is_two_rows(cache: SqliteDecisionCache) -> None:
    rows, state_json, questions_json = make_rows()
    put(cache, rows, state_json, questions_json)
    cache.ensure_namespace(OTHER_NAMESPACE, MODEL, RELEASE, refresh=False)
    other = [CachedAnswer(**{**{f: getattr(row, f) for f in row.__struct_fields__}, "namespace": OTHER_NAMESPACE}) for row in rows]
    put(cache, other, state_json, questions_json)
    keys = [row.key for row in rows]
    assert len(cache.get_many(NAMESPACE, keys)) == len(rows)
    assert len(cache.get_many(OTHER_NAMESPACE, keys)) == len(rows)
    assert cache.stats()["answers"] == 2 * len(rows)
    assert cache.stats(NAMESPACE)["answers"] == len(rows)
    assert cache.manifest_hash(NAMESPACE) == cache.manifest_hash(OTHER_NAMESPACE)  # same content, different namespace


def test_ensure_namespace_binds_one_model_and_one_diagnostic_flag(cache: SqliteDecisionCache) -> None:
    cache.ensure_namespace(NAMESPACE, MODEL, RELEASE, refresh=False)  # idempotent
    with pytest.raises(ConfigError, match="bound to model"):
        cache.ensure_namespace(NAMESPACE, "jev-1.14.0", RELEASE, refresh=False)
    with pytest.raises(ConfigError, match="release date"):
        cache.ensure_namespace(NAMESPACE, MODEL, date(2026, 1, 1), refresh=False)
    with pytest.raises(ConfigError, match="diagnostic"):
        cache.ensure_namespace(NAMESPACE, MODEL, RELEASE, refresh=False, diagnostic=True)
    assert cache.is_diagnostic(NAMESPACE) is False and cache.namespace_model(NAMESPACE) == MODEL
    diagnostic = ids.namespace("probe", MODEL, 0)
    cache.ensure_namespace(diagnostic, MODEL, RELEASE, refresh=False, diagnostic=True)
    assert cache.is_diagnostic(diagnostic) is True
    assert cache.is_diagnostic("never-created") is False and cache.namespace_model("never-created") is None


def test_refresh_requires_an_empty_namespace(cache: SqliteDecisionCache) -> None:
    cache.ensure_namespace(NAMESPACE, MODEL, RELEASE, refresh=True)  # still empty: fine
    rows, state_json, questions_json = make_rows()
    put(cache, rows, state_json, questions_json)
    with pytest.raises(ConfigError, match="refresh"):
        cache.ensure_namespace(NAMESPACE, MODEL, RELEASE, refresh=True)


def test_put_request_refuses_an_unknown_namespace_or_a_foreign_model(cache: SqliteDecisionCache) -> None:
    rows, state_json, questions_json = make_rows(namespace="not:created:g0")
    with pytest.raises(ConfigError, match="does not exist"):
        put(cache, rows, state_json, questions_json)
    rows, state_json, questions_json = make_rows(response_model="jev-1.14.0")
    with pytest.raises(ModelMismatchError, match="INV-06"):
        put(cache, rows, state_json, questions_json)
    assert cache.stats()["answers"] == 0  # INV-06: a mismatch caches NOTHING


# ======================================================================================================================
# All-or-nothing
# ======================================================================================================================


def test_a_request_is_a_hit_only_when_every_key_is_present(cache: SqliteDecisionCache) -> None:
    rows, state_json, questions_json = make_rows()
    keys = [row.key for row in rows]
    assert cache.get_many(NAMESPACE, keys) == {} and cache.has_request(NAMESPACE, keys) is False
    put(cache, rows, state_json, questions_json)
    assert cache.has_request(NAMESPACE, keys) is True
    assert cache.has_request(NAMESPACE, [*keys, "0" * 64]) is False
    assert cache.has_request(NAMESPACE, []) is False
    assert cache.has_request(OTHER_NAMESPACE, keys) is False
    got = cache.get_many(NAMESPACE, keys)
    assert set(got) == set(keys)
    sample = got[rows[0].key]
    assert sample.question_set_id == "entry.v1" and sample.request_kind == "entry" and sample.variant == "base"
    assert sample.input_tokens == 1234 and sample.latency_ms == 42 and sample.request_id == "req-1"
    assert sample.created_at == datetime(2026, 9, 17, 20, 0, tzinfo=UTC) and sample.sdk_version == "0.6.0"


def test_put_request_is_one_transaction(cache: SqliteDecisionCache) -> None:
    rows, state_json, questions_json = make_rows()
    broken = [*rows[:3], CachedAnswer(**{**{f: getattr(rows[3], f) for f in rows[3].__struct_fields__}, "variant": "opt_perm"})]
    with pytest.raises(InvariantError, match="disagree about variant"):
        put(cache, broken, state_json, questions_json)
    assert cache.stats()["answers"] == 0
    with pytest.raises(InvariantError, match="at least one answer row"):
        cache.put_request(NAMESPACE, [], state_json, questions_json, (SESSION, "SPY", "entry", "base"))


def test_the_request_index_records_one_row_per_request(cache: SqliteDecisionCache) -> None:
    rows, state_json, questions_json = make_rows()
    put(cache, rows, state_json, questions_json)
    put(cache, rows, state_json, questions_json)  # the same request again: an idempotent no-op
    variant_rows, _, _ = make_rows(variant="key_perm", state={**sample_state(), "schema": "state.v1.entry"})
    assert cache.stats()["requests"] == 1
    other_state = sample_state()
    other_state["context"]["underlying_alias"] = "UNDERLYING_B"
    rows_b, state_b, questions_b = make_rows(state=other_state)
    put(cache, rows_b, state_b, questions_b, underlying="QQQ")
    assert cache.stats()["requests"] == 2
    assert cache.stats()["states"] == 2 and cache.stats()["question_sets"] == 1
    assert variant_rows[0].variant == "key_perm"


# ======================================================================================================================
# Immutability
# ======================================================================================================================


def test_first_write_wins_and_a_divergence_is_recorded(cache: SqliteDecisionCache) -> None:
    rows, state_json, questions_json = make_rows(answer=0.5)
    put(cache, rows, state_json, questions_json)
    again, _, _ = make_rows(answer=0.9)  # the same keys, a different answer
    assert [row.key for row in again] == [row.key for row in rows]
    put(cache, again, state_json, questions_json)
    stored = cache.get_many(NAMESPACE, [rows[0].key])[rows[0].key]
    assert json.loads(stored.answer_json)["noul"] == 0.5, "the FIRST answer is kept"
    assert cache.stats()["nondeterminism"] == len(rows)
    assert cache.stats()["answers"] == len(rows)


def test_update_and_delete_are_refused_by_the_triggers(cache: SqliteDecisionCache, tmp_path: Path) -> None:
    rows, state_json, questions_json = make_rows()
    put(cache, rows, state_json, questions_json)
    raw = sqlite3.connect(tmp_path / "cache" / "decisions.sqlite")
    try:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            raw.execute("UPDATE answers SET answer_json = '{}'")
        with pytest.raises(sqlite3.IntegrityError, match="never evicted"):
            raw.execute("DELETE FROM answers")
    finally:
        raw.close()
    assert cache.stats()["answers"] == len(rows)


def test_a_read_only_cache_never_writes(tmp_path: Path, cache: SqliteDecisionCache) -> None:
    rows, state_json, questions_json = make_rows()
    put(cache, rows, state_json, questions_json)
    replay = SqliteDecisionCache(tmp_path / "cache" / "decisions.sqlite", read_only=True)
    try:
        assert len(replay.get_many(NAMESPACE, [row.key for row in rows])) == len(rows)
        with pytest.raises(InvariantError, match="read-only"):
            replay.ensure_namespace("x:y:g0", MODEL, RELEASE, refresh=False)
        with pytest.raises(InvariantError, match="read-only"):
            put(replay, rows, state_json, questions_json)
    finally:
        replay.close()
    with pytest.raises(ConfigError, match="does not exist"):
        SqliteDecisionCache(tmp_path / "missing" / "decisions.sqlite", read_only=True)


# ======================================================================================================================
# Manifest and verification
# ======================================================================================================================


def test_manifest_hash_is_stable_and_content_addressed(cache: SqliteDecisionCache) -> None:
    assert cache.manifest_hash(NAMESPACE) == canon.sha256_hex("")  # an empty namespace hashes the empty material
    rows, state_json, questions_json = make_rows()
    put(cache, rows, state_json, questions_json)
    first = cache.manifest_hash(NAMESPACE)
    assert first == cache.manifest_hash(NAMESPACE)
    material = "".join(f"{row.key}:{canon.sha256_hex(row.answer_json)}\n" for row in sorted(rows, key=lambda r: r.key))
    assert first == canon.sha256_hex(material)  # 3.3, spelled out independently
    other_state = sample_state()
    other_state["market"]["as_of"] = "two sessions ago"
    rows_b, state_b, questions_b = make_rows(state=other_state)
    put(cache, rows_b, state_b, questions_b)
    assert cache.manifest_hash(NAMESPACE) != first


def test_verify_re_derives_every_key_and_finds_a_tampered_row(cache: SqliteDecisionCache, tmp_path: Path) -> None:
    rows, state_json, questions_json = make_rows()
    put(cache, rows, state_json, questions_json)
    report = cache.verify()
    assert report.ok and report.rows == len(rows) and report.namespaces == 1
    assert report.manifest_hashes[NAMESPACE] == cache.manifest_hash(NAMESPACE)
    # tamper with a stored state through a raw connection with the triggers bypassed (answers are untouched)
    raw = sqlite3.connect(tmp_path / "cache" / "decisions.sqlite")
    try:
        raw.execute("UPDATE states SET state_json = ?", (canon.dumps_ordered({"schema": "state.v1.entry"}),))
        raw.commit()
    finally:
        raw.close()
    broken = cache.verify()
    assert not broken.ok
    assert {problem.problem for problem in broken.problems} == {"key_mismatch"}
    assert all(problem.namespace == NAMESPACE for problem in broken.problems)


def test_verify_reports_a_model_binding_problem(cache: SqliteDecisionCache, tmp_path: Path) -> None:
    rows, state_json, questions_json = make_rows()
    put(cache, rows, state_json, questions_json)
    raw = sqlite3.connect(tmp_path / "cache" / "decisions.sqlite")
    try:
        raw.execute("UPDATE namespaces SET model = 'jev-1.14.0'")
        raw.commit()
    finally:
        raw.close()
    report = cache.verify(NAMESPACE)
    assert {problem.problem for problem in report.problems} == {"model_binding"}


# ======================================================================================================================
# The `cache` sub-app
# ======================================================================================================================


def _run(args: list[str], data_dir: Path, *, as_json: bool = False) -> Any:
    # a sub-app invoked without the root gets its global options through `obj` (the section-14 sub-app contract)
    options = GlobalOptions(overrides=(f"paths.data_dir={data_dir}",), json=as_json)
    return CliRunner().invoke(jev_cmds.cache_app, args, obj=options)


def test_cache_stats_and_verify_commands(data_dir: Path) -> None:
    store = SqliteDecisionCache(data_dir / "cache" / "decisions.sqlite")
    store.ensure_namespace(NAMESPACE, MODEL, RELEASE, refresh=False)
    rows, state_json, questions_json = make_rows()
    put(store, rows, state_json, questions_json)
    store.close()
    result = _run(["stats"], data_dir, as_json=True)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["counts"]["answers"] == len(rows)
    assert payload["breakdown"]["models"] == {MODEL: len(rows)}
    assert payload["breakdown"]["request_kinds"] == {"entry": len(rows)}
    assert set(payload["spend_by_day"]) == {"paper", "batch"}
    assert payload["manifest_hashes"][NAMESPACE]
    plain = _run(["stats"], data_dir)
    assert plain.exit_code == 0 and "answers: 19" in plain.output
    verified = _run(["verify"], data_dir, as_json=True)
    assert verified.exit_code == 0 and json.loads(verified.output)["ok"] is True
    assert _run(["verify"], data_dir).exit_code == 0
