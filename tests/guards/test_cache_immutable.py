"""Guard: the decision cache is append-only and is never evicted (DESIGN.md 13.3, 15.5; D8).

The cache is the evidence store every replay, sweep, baseline and ablation arm reads back, and `cache_manifest_hash` is
part of a trial's record. Two independent guards:

* a SOURCE guard - no `DELETE` or `UPDATE` statement against `answers` may appear anywhere in `src/` (AST + text);
* a RUNTIME guard - the shipped schema's triggers abort both statements even on a raw connection that bypasses our code.
"""

import ast
import re
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import pytest

from jevbot import canon, questions as questions_module
from jevbot.jev.cache import SCHEMA_SQL, SqliteDecisionCache
from jevbot.types import CachedAnswer

SRC: Final = Path(__file__).resolve().parents[2] / "src" / "jevbot"
# a statement is suspicious when it writes to `answers` in a way that is not an INSERT
_FORBIDDEN: Final = re.compile(r"\b(delete\s+from|update)\s+answers\b", re.IGNORECASE)
_ANY_DESTRUCTIVE: Final = re.compile(r"\b(delete\s+from|update|drop\s+table|drop\s+trigger)\b", re.IGNORECASE)


_SQL_START: Final = re.compile(r"^\s*(select|insert|update|delete|create|pragma|begin|commit|rollback|with)\b", re.IGNORECASE)


def _string_literals(path: Path) -> list[str]:
    """Every string constant in a module (SQL is always written as a literal in this project)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]


def _sql_literals(path: Path) -> list[str]:
    """The string constants that are SQL statements (a docstring that merely NAMES a statement is not one)."""
    return [literal for literal in _string_literals(path) if _SQL_START.match(literal)]


def test_no_source_file_writes_to_the_answers_table() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        for literal in _string_literals(path):
            if _FORBIDDEN.search(literal):
                offenders.append(f"{path.relative_to(SRC)}: {literal.strip()[:80]}")
    assert offenders == [], "the decision cache is immutable and never evicted (13.3)"


def test_the_cache_module_never_drops_a_trigger_or_a_table() -> None:
    # the only destructive-looking statements in jev/cache.py must be the trigger BODIES the schema installs
    literals = _string_literals(SRC / "jev" / "cache.py")
    for literal in literals:
        if literal is SCHEMA_SQL or "CREATE TRIGGER" in literal:
            continue
        assert not _ANY_DESTRUCTIVE.search(literal), f"jev/cache.py must not write destructive SQL: {literal[:80]}"


def test_the_shipped_schema_installs_both_triggers() -> None:
    assert "answers_no_update" in SCHEMA_SQL and "answers_no_delete" in SCHEMA_SQL
    assert "RAISE(ABORT, 'cache is immutable')" in SCHEMA_SQL
    assert "RAISE(ABORT, 'cache is never evicted')" in SCHEMA_SQL


def test_the_triggers_abort_even_on_a_raw_connection(tmp_path: Path) -> None:
    path = tmp_path / "decisions.sqlite"
    cache = SqliteDecisionCache(path)
    namespace = "guard:jev-1.13.0:g0"
    cache.ensure_namespace(namespace, "jev-1.13.0", date(2026, 9, 15), refresh=False)
    question = questions_module.ENTRY_V1["regime.market"]
    qset_hash = questions_module.QUESTION_SET_HASHES["entry.v1"]
    state = {"schema": "state.v1.entry"}
    row = CachedAnswer(
        key=canon.cache_key("jev-1.13.0", state, qset_hash, question),
        namespace=namespace,
        requested_model="jev-1.13.0",
        response_model="jev-1.13.0",
        question_set_id="entry.v1",
        question_set_hash=qset_hash,
        state_hash=canon.sha256_hex(canon.dumps_ordered(state)),
        question_hash=questions_module.QUESTION_HASHES["regime.market"],
        question_id="regime.market",
        request_kind="entry",
        variant="base",
        answer_json='{"noul":0.5,"type":"noul"}',
        request_id=None,
        input_tokens=None,
        latency_ms=None,
        sdk_version="0.6.0",
        created_at=datetime(2026, 9, 17, 20, 0, tzinfo=UTC),
    )
    cache.put_request(
        namespace, [row], canon.dumps_ordered(state), canon.dumps_ordered({"regime.market": question}), (date(2024, 5, 17), "SPY", "entry", "base")
    )
    before = cache.manifest_hash(namespace)
    raw = sqlite3.connect(path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="cache is immutable"):
            raw.execute("UPDATE answers SET answer_json = '{\"noul\":0.9,\"type\":\"noul\"}'")
        with pytest.raises(sqlite3.IntegrityError, match="cache is never evicted"):
            raw.execute("DELETE FROM answers WHERE 1=1")
        raw.rollback()
    finally:
        raw.close()
    assert cache.manifest_hash(namespace) == before
    assert cache.stats()["answers"] == 1
    cache.close()
