"""The decision cache (`$JEVBOT_DATA/cache/decisions.sqlite`, DESIGN.md 13.3, D8).

Content-addressed and **immutable**: a row's key is `canon.cache_key(model, state, question_set_hash, question)` - pure content,
no namespace, no sample index, no question id - and the schema carries triggers that abort every `UPDATE` and every `DELETE`
(`tests/guards/test_cache_immutable.py` also forbids the statements anywhere in `src/`). Nothing is ever evicted: a recorded
answer is the evidence a replay, a threshold sweep, a baseline and the ablation arm all read back for free.

Contract points that live here rather than in the deciders:

* a request is a **hit** only when EVERY key of the full batch is present in the run's namespace (`get_many` / `has_request`);
  otherwise the full batch is sent and all rows are written in ONE transaction (`put_request`, all-or-nothing);
* a namespace is bound to ONE model id: `put_request` refuses a row whose `response_model` differs from `namespaces.model`
  (`ModelMismatchError`, INV-06's backstop), and `ensure_namespace` refuses to rebind an existing namespace;
* first write wins (`INSERT OR IGNORE`). A re-answer that DIFFERS is kept out and the divergence is recorded in table
  `nondeterminism` - free evidence for unknown U1 (is the model deterministic at temperature 0?);
* `diagnostic=True` marks probe / leakage namespaces; `is_diagnostic` is read by the run-start guard and reaches risk check 2
  through the RUN_START flags, so a diagnostic namespace can never back a trading run.

Threading: `decide_batch` fans out over a small thread pool, so the connection is opened with `check_same_thread=False` and
every statement runs under one `threading.Lock` (3.3). Replay opens the file read-only (`mode=ro`).
"""

import json
import re
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import msgspec

from jevbot import canon
from jevbot.errors import ConfigError, DataError, InvariantError, ModelMismatchError
from jevbot.types import CachedAnswer

__all__ = ["SCHEMA_SQL", "CacheProblem", "CacheVerifyReport", "SqliteDecisionCache"]

# ======================================================================================================================
# Schema (13.3, verbatim)
# ======================================================================================================================

SCHEMA_SQL: Final = """
CREATE TABLE IF NOT EXISTS namespaces (namespace TEXT PRIMARY KEY, experiment TEXT NOT NULL, model TEXT NOT NULL, model_release_date TEXT NOT NULL,
  refresh_generation INTEGER NOT NULL, diagnostic INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, note TEXT);
CREATE TABLE IF NOT EXISTS answers (
  namespace TEXT NOT NULL REFERENCES namespaces, key TEXT NOT NULL,
  requested_model TEXT NOT NULL, response_model TEXT NOT NULL, question_set_hash TEXT NOT NULL, state_hash TEXT NOT NULL,
  question_hash TEXT NOT NULL, question_id TEXT NOT NULL, request_kind TEXT NOT NULL, variant TEXT NOT NULL,
  answer_json TEXT NOT NULL, answer_sha256 TEXT NOT NULL, request_id TEXT, input_tokens INTEGER, latency_ms INTEGER,
  sdk_version TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY (namespace, key)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS answers_by_request ON answers(namespace, state_hash, question_set_hash);
CREATE TABLE IF NOT EXISTS states        (state_hash TEXT PRIMARY KEY, state_json TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS question_sets (question_set_hash TEXT PRIMARY KEY, question_set_id TEXT NOT NULL, questions_json TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS request_index (namespace TEXT NOT NULL, session TEXT NOT NULL, underlying TEXT NOT NULL, request_kind TEXT NOT NULL, variant TEXT NOT NULL,
  state_hash TEXT NOT NULL, question_set_hash TEXT NOT NULL, PRIMARY KEY (namespace, session, underlying, request_kind, variant, state_hash));
CREATE TABLE IF NOT EXISTS nondeterminism (namespace TEXT, key TEXT, first_sha256 TEXT, other_sha256 TEXT, seen_at TEXT);
CREATE TRIGGER IF NOT EXISTS answers_no_update BEFORE UPDATE ON answers BEGIN SELECT RAISE(ABORT, 'cache is immutable'); END;
CREATE TRIGGER IF NOT EXISTS answers_no_delete BEFORE DELETE ON answers BEGIN SELECT RAISE(ABORT, 'cache is never evicted'); END;
"""

_ANSWER_COLUMNS: Final = (
    "namespace, key, requested_model, response_model, question_set_hash, state_hash, question_hash, question_id, "
    "request_kind, variant, answer_json, answer_sha256, request_id, input_tokens, latency_ms, sdk_version, created_at"
)
# `CachedAnswer.question_set_id` lives in `question_sets` (one row per set), so every read joins it back in
_SELECT_ANSWER: Final = (
    "SELECT a.namespace, a.key, a.requested_model, a.response_model, a.question_set_hash, a.state_hash, a.question_hash, "
    "a.question_id, a.request_kind, a.variant, a.answer_json, a.request_id, a.input_tokens, a.latency_ms, a.sdk_version, "
    "a.created_at, COALESCE(q.question_set_id, '') AS question_set_id FROM answers a "
    "LEFT JOIN question_sets q ON q.question_set_hash = a.question_set_hash"
)
_NAMESPACE_RE: Final = re.compile(r"^(?P<experiment>[^:]+):(?P<model>.+):g(?P<generation>\d+)$")
_CHUNK: Final = 400  # keys per IN (...) statement, well below SQLite's variable limit


class CacheProblem(msgspec.Struct, frozen=True, kw_only=True):
    """One finding of `verify()`: what is wrong with which row (never the answer content itself)."""

    namespace: str
    key: str
    problem: str  # "key_mismatch" | "answer_sha256" | "model_binding" | "state_missing" | "question_set_missing" | "question_missing"
    detail: str


class CacheVerifyReport(msgspec.Struct, frozen=True, kw_only=True):
    """What `cache verify` prints (section 14): counts plus every problem found."""

    namespaces: int
    rows: int
    states: int
    question_sets: int
    problems: tuple[CacheProblem, ...]
    manifest_hashes: dict[str, str]

    @property
    def ok(self) -> bool:
        return not self.problems


def _now_text() -> str:
    """The `created_at` / `seen_at` sidecar text. Audit columns only: never hashed, never in a cache key (INV-24)."""
    return canon.render_as_of(datetime.now(UTC))


def _parse_created_at(text: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise DataError(f"decision cache: created_at {text!r} is not an ISO timestamp") from None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _split_namespace(namespace: str) -> tuple[str, int]:
    """(experiment, refresh_generation) of `ids.namespace(...)`; an unparseable namespace keeps itself as the experiment."""
    match = _NAMESPACE_RE.match(namespace)
    if match is None:
        return namespace, 0
    return match.group("experiment"), int(match.group("generation"))


class SqliteDecisionCache:
    """`DecisionCache` (3.3) over `cache/decisions.sqlite`; `read_only=True` opens the file `mode=ro` for replay."""

    def __init__(self, path: Path | str, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = bool(read_only)
        self._lock = threading.RLock()  # re-entrant: verify() calls manifest_hash() while it holds the lock
        if read_only:
            if not self.path.exists():
                raise ConfigError(f"decision cache {self.path} does not exist: a replay run needs a recorded cache (D7)")
            self._db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False, timeout=5.0)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
        self._db.row_factory = sqlite3.Row
        self._db.isolation_level = None  # explicit BEGIN IMMEDIATE / COMMIT around put_request
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute("PRAGMA foreign_keys=ON")
        if not read_only:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            with self._lock:
                self._db.executescript(SCHEMA_SQL)

    # ------------------------------------------------------------------ namespaces

    def ensure_namespace(self, namespace: str, model: str, model_release_date: date, *, refresh: bool, diagnostic: bool = False) -> None:
        """Create the namespace or check that the existing one agrees (3.3).

        `refresh=True` requires an EMPTY namespace (a refresh generation must never mix bytes from two model versions);
        an existing namespace can never change its model, its release date or its `diagnostic` flag.
        """
        self._require_writable("ensure_namespace")
        experiment, generation = _split_namespace(namespace)
        release = canon.render_session(model_release_date)
        with self._lock:
            row = self._db.execute(
                "SELECT model, model_release_date, diagnostic FROM namespaces WHERE namespace = ?", (namespace,)
            ).fetchone()
            if row is None:
                self._db.execute(
                    "INSERT INTO namespaces (namespace, experiment, model, model_release_date, refresh_generation, diagnostic, created_at, note) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                    (namespace, experiment, model, release, generation, int(diagnostic), _now_text()),
                )
                return
            if row["model"] != model:
                raise ConfigError(
                    f"namespace {namespace!r} is bound to model {row['model']!r} and can never answer for {model!r} (13.3): use a new experiment"
                )
            if row["model_release_date"] != release:
                raise ConfigError(
                    f"namespace {namespace!r} was created with model release date {row['model_release_date']} , not {release}"
                )
            if bool(row["diagnostic"]) != bool(diagnostic):
                raise ConfigError(
                    f"namespace {namespace!r} already exists with diagnostic={bool(row['diagnostic'])}: the flag can never be changed"
                )
            if refresh:
                rows = self._count("SELECT COUNT(*) FROM answers WHERE namespace = ?", (namespace,))
                if rows:
                    raise ConfigError(
                        f'cache mode "refresh" needs an empty namespace: {namespace!r} already holds {rows} answers - '
                        "bump jev.refresh_generation (6.8)"
                    )

    def is_diagnostic(self, namespace: str) -> bool:
        """True iff the namespace was created with `diagnostic=True` (probe / leakage). An unknown namespace is False."""
        with self._lock:
            row = self._db.execute("SELECT diagnostic FROM namespaces WHERE namespace = ?", (namespace,)).fetchone()
        return bool(row["diagnostic"]) if row is not None else False

    def namespace_model(self, namespace: str) -> str | None:
        """The model a namespace is bound to, or None when it does not exist."""
        with self._lock:
            row = self._db.execute("SELECT model FROM namespaces WHERE namespace = ?", (namespace,)).fetchone()
        return str(row["model"]) if row is not None else None

    # ------------------------------------------------------------------ reads

    def get_many(self, namespace: str, keys: Sequence[str]) -> dict[str, CachedAnswer]:
        """The cached rows of `keys` present in `namespace`, keyed by cache key (a partial result is a MISS for the caller)."""
        wanted = list(dict.fromkeys(keys))
        out: dict[str, CachedAnswer] = {}
        with self._lock:
            for chunk in _chunks(wanted):
                placeholders = ",".join("?" * len(chunk))
                cursor = self._db.execute(
                    f"{_SELECT_ANSWER} WHERE a.namespace = ? AND a.key IN ({placeholders})",
                    (namespace, *chunk),
                )
                for row in cursor.fetchall():
                    out[str(row["key"])] = _row_to_answer(row)
        return out

    def has_request(self, namespace: str, keys: Sequence[str]) -> bool:
        """True iff EVERY key is present (the 11.8 shadow-replay probe and the hit rule of 13.3)."""
        wanted = list(dict.fromkeys(keys))
        if not wanted:
            return False
        with self._lock:
            found = 0
            for chunk in _chunks(wanted):
                placeholders = ",".join("?" * len(chunk))
                found += self._db.execute(
                    f"SELECT COUNT(*) FROM answers WHERE namespace = ? AND key IN ({placeholders})",
                    (namespace, *chunk),
                ).fetchone()[0]
        return found == len(wanted)

    # ------------------------------------------------------------------ the one write path

    def put_request(
        self, namespace: str, rows: Sequence[CachedAnswer], state_json: str, questions_json: str, index: tuple[date, str, str, str]
    ) -> None:
        """Write one whole request: ONE transaction, all-or-nothing, `INSERT OR IGNORE` (3.3, 13.3).

        Every row of a request shares its state, question set, kind and variant (asserted); `question_sets.question_set_id`
        comes from `rows[0]`. A key that is already present keeps its FIRST answer; a differing re-answer is recorded in
        `nondeterminism` instead.
        """
        self._require_writable("put_request")
        if not rows:
            raise InvariantError("put_request: a request has at least one answer row")
        first = rows[0]
        for row in rows:
            if row.namespace != namespace:
                raise InvariantError(f"put_request: row {row.key!r} carries namespace {row.namespace!r}, not {namespace!r}")
            for field in (
                "question_set_id",
                "question_set_hash",
                "state_hash",
                "request_kind",
                "variant",
                "requested_model",
                "response_model",
            ):
                if getattr(row, field) != getattr(first, field):
                    raise InvariantError(f"put_request: rows of one request disagree about {field}")
        session, underlying, request_kind, variant = index
        if (request_kind, variant) != (first.request_kind, first.variant):
            raise InvariantError("put_request: `index` disagrees with the rows about request kind / variant")
        session_text = canon.render_session(session)
        now = _now_text()
        with self._lock:
            bound = self._db.execute("SELECT model FROM namespaces WHERE namespace = ?", (namespace,)).fetchone()
            if bound is None:
                raise ConfigError(f"namespace {namespace!r} does not exist: call ensure_namespace() before caching answers")
            if first.response_model != bound["model"]:
                raise ModelMismatchError(
                    f"namespace {namespace!r} is bound to model {bound['model']!r} but the response came from "
                    f"{first.response_model!r}: nothing is cached (INV-06)"
                )
            self._db.execute("BEGIN IMMEDIATE")
            try:
                existing = {
                    str(row["key"]): str(row["answer_sha256"])
                    for chunk in _chunks([row.key for row in rows])
                    for row in self._db.execute(
                        f"SELECT key, answer_sha256 FROM answers WHERE namespace = ? AND key IN ({','.join('?' * len(chunk))})",
                        (namespace, *chunk),
                    ).fetchall()
                }
                self._db.execute("INSERT OR IGNORE INTO states (state_hash, state_json) VALUES (?, ?)", (first.state_hash, state_json))
                self._db.execute(
                    "INSERT OR IGNORE INTO question_sets (question_set_hash, question_set_id, questions_json) VALUES (?, ?, ?)",
                    (first.question_set_hash, first.question_set_id, questions_json),
                )
                for row in rows:
                    digest = canon.sha256_hex(row.answer_json)
                    seen = existing.get(row.key)
                    if seen is not None:
                        if seen != digest:  # first write wins; the divergence is the evidence (3.3)
                            self._db.execute(
                                "INSERT INTO nondeterminism (namespace, key, first_sha256, other_sha256, seen_at) VALUES (?, ?, ?, ?, ?)",
                                (namespace, row.key, seen, digest, now),
                            )
                        continue
                    self._db.execute(
                        f"INSERT OR IGNORE INTO answers ({_ANSWER_COLUMNS}) VALUES ({','.join('?' * 17)})",
                        (
                            namespace,
                            row.key,
                            row.requested_model,
                            row.response_model,
                            row.question_set_hash,
                            row.state_hash,
                            row.question_hash,
                            row.question_id,
                            row.request_kind,
                            row.variant,
                            row.answer_json,
                            digest,
                            row.request_id,
                            row.input_tokens,
                            row.latency_ms,
                            row.sdk_version,
                            canon.render_as_of(row.created_at),
                        ),
                    )
                self._db.execute(
                    "INSERT OR IGNORE INTO request_index (namespace, session, underlying, request_kind, variant, state_hash, question_set_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (namespace, session_text, underlying, request_kind, variant, first.state_hash, first.question_set_hash),
                )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------ statistics, manifest, verification

    def stats(self, namespace: str | None = None) -> dict[str, int]:
        """Row counts of the whole cache or of one namespace (3.3): everything an operator counts, all ints."""
        where, args = ("WHERE namespace = ?", (namespace,)) if namespace is not None else ("", ())
        with self._lock:
            state_rows = (
                self._count("SELECT COUNT(DISTINCT state_hash) FROM answers " + where, args)
                if namespace is not None
                else self._count("SELECT COUNT(*) FROM states", ())
            )
            qset_rows = (
                self._count("SELECT COUNT(DISTINCT question_set_hash) FROM answers " + where, args)
                if namespace is not None
                else self._count("SELECT COUNT(*) FROM question_sets", ())
            )
            return {
                "namespaces": self._count("SELECT COUNT(*) FROM namespaces " + (where or ""), args),
                "diagnostic_namespaces": self._count(
                    "SELECT COUNT(*) FROM namespaces WHERE diagnostic = 1" + (" AND namespace = ?" if namespace else ""), args
                ),
                "answers": self._count("SELECT COUNT(*) FROM answers " + where, args),
                "requests": self._count("SELECT COUNT(*) FROM request_index " + where, args),
                "states": state_rows,
                "question_sets": qset_rows,
                "nondeterminism": self._count("SELECT COUNT(*) FROM nondeterminism " + where, args),
                # input_tokens is repeated on every row of a request: count it once per (namespace, state, question set, variant)
                "input_tokens": self._count(
                    "SELECT COALESCE(SUM(input_tokens), 0) FROM (SELECT DISTINCT namespace, state_hash, question_set_hash, variant, "
                    f"input_tokens FROM answers {where})",
                    args,
                ),
            }

    def breakdown(self, namespace: str | None = None) -> dict[str, dict[str, int]]:
        """Answer counts per model / request kind / variant / question set - the tables `cache stats` prints."""
        where, args = ("WHERE namespace = ?", (namespace,)) if namespace is not None else ("", ())
        out: dict[str, dict[str, int]] = {}
        with self._lock:
            for label, column in (
                ("namespaces", "namespace"),
                ("models", "response_model"),
                ("request_kinds", "request_kind"),
                ("variants", "variant"),
            ):
                cursor = self._db.execute(
                    f"SELECT {column} AS bucket, COUNT(*) AS n FROM answers {where} GROUP BY {column} ORDER BY {column}",
                    args,
                )
                out[label] = {str(row["bucket"]): int(row["n"]) for row in cursor.fetchall()}
        return out

    def manifest_hash(self, namespace: str) -> str:
        """`sha256` over `"key:sha256(answer_json)\\n"` for all rows of the namespace, ORDER BY key (3.3)."""
        with self._lock:
            cursor = self._db.execute("SELECT key, answer_sha256 FROM answers WHERE namespace = ? ORDER BY key", (namespace,))
            material = "".join(f"{row['key']}:{row['answer_sha256']}\n" for row in cursor.fetchall())
        return canon.sha256_hex(material)

    def verify(self, namespace: str | None = None) -> CacheVerifyReport:
        """Re-derive every key from the stored state and question JSON, recompute `answer_sha256`, check the model binding."""
        where, args = ("WHERE a.namespace = ?", (namespace,)) if namespace is not None else ("", ())
        problems: list[CacheProblem] = []
        rows = 0
        question_cache: dict[str, dict[str, Any]] = {}
        state_cache: dict[str, dict[str, Any]] = {}
        with self._lock:
            names = [
                str(row["namespace"])
                for row in self._db.execute(
                    "SELECT namespace FROM namespaces" + (" WHERE namespace = ?" if namespace is not None else "") + " ORDER BY namespace",
                    args,
                ).fetchall()
            ]
            models = {
                str(row["namespace"]): str(row["model"]) for row in self._db.execute("SELECT namespace, model FROM namespaces").fetchall()
            }
            cursor = self._db.execute(
                "SELECT a.namespace, a.key, a.requested_model, a.response_model, a.question_set_hash, a.state_hash, a.question_hash, "
                f"a.question_id, a.answer_json, a.answer_sha256, s.state_json, q.questions_json FROM answers a "
                f"LEFT JOIN states s ON s.state_hash = a.state_hash "
                f"LEFT JOIN question_sets q ON q.question_set_hash = a.question_set_hash {where} ORDER BY a.namespace, a.key",
                args,
            )
            for row in cursor.fetchall():
                rows += 1
                problems.extend(_verify_row(row, models, state_cache, question_cache))
            totals = {
                "states": self._count("SELECT COUNT(*) FROM states", ()),
                "question_sets": self._count("SELECT COUNT(*) FROM question_sets", ()),
            }
        return CacheVerifyReport(
            namespaces=len(names),
            rows=rows,
            states=totals["states"],
            question_sets=totals["question_sets"],
            problems=tuple(problems),
            manifest_hashes={name: self.manifest_hash(name) for name in names},
        )

    # ------------------------------------------------------------------ plumbing

    def _count(self, sql: str, args: Sequence[Any]) -> int:
        return int(self._db.execute(sql, tuple(args)).fetchone()[0])

    def _require_writable(self, what: str) -> None:
        if self.read_only:
            raise InvariantError(f"{what} on a read-only decision cache: replay never writes (D7)")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> "SqliteDecisionCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _chunks(keys: Sequence[str]) -> Iterator[Sequence[str]]:
    for start in range(0, len(keys), _CHUNK):
        yield keys[start : start + _CHUNK]


def _row_to_answer(row: sqlite3.Row) -> CachedAnswer:
    return CachedAnswer(
        key=str(row["key"]),
        namespace=str(row["namespace"]),
        requested_model=str(row["requested_model"]),
        response_model=str(row["response_model"]),
        question_set_id=str(row["question_set_id"]),
        question_set_hash=str(row["question_set_hash"]),
        state_hash=str(row["state_hash"]),
        question_hash=str(row["question_hash"]),
        question_id=str(row["question_id"]),
        request_kind=str(row["request_kind"]),
        variant=str(row["variant"]),
        answer_json=str(row["answer_json"]),
        request_id=None if row["request_id"] is None else str(row["request_id"]),
        input_tokens=None if row["input_tokens"] is None else int(row["input_tokens"]),
        latency_ms=None if row["latency_ms"] is None else int(row["latency_ms"]),
        sdk_version=str(row["sdk_version"]),
        created_at=_parse_created_at(str(row["created_at"])),
    )


def _verify_row(
    row: sqlite3.Row, models: Mapping[str, str], state_cache: dict[str, dict[str, Any]], question_cache: dict[str, dict[str, Any]]
) -> list[CacheProblem]:
    namespace, key = str(row["namespace"]), str(row["key"])

    def problem(kind: str, detail: str) -> CacheProblem:
        return CacheProblem(namespace=namespace, key=key, problem=kind, detail=detail)

    found: list[CacheProblem] = []
    if canon.sha256_hex(str(row["answer_json"])) != str(row["answer_sha256"]):
        found.append(problem("answer_sha256", "the stored digest does not match the stored answer"))
    bound = models.get(namespace)
    if bound is not None and str(row["response_model"]) != bound:
        found.append(problem("model_binding", f"response_model {row['response_model']!r} is not the namespace's model {bound!r}"))
    if row["state_json"] is None:
        return [*found, problem("state_missing", f"no states row for state_hash {row['state_hash']}")]
    if row["questions_json"] is None:
        return [*found, problem("question_set_missing", f"no question_sets row for {row['question_set_hash']}")]
    state = state_cache.get(str(row["state_hash"]))
    if state is None:
        state = json.loads(str(row["state_json"]))
        state_cache[str(row["state_hash"])] = state
    questions = question_cache.get(str(row["question_set_hash"]))
    if questions is None:
        questions = json.loads(str(row["questions_json"]))
        question_cache[str(row["question_set_hash"])] = questions
    question = questions.get(str(row["question_id"]))
    if question is None:
        return [*found, problem("question_missing", f"question {row['question_id']!r} is not in the stored question set")]
    if canon.sha256_hex(canon.dumps_ordered(question)) != str(row["question_hash"]):
        found.append(problem("key_mismatch", "the stored question does not hash to the stored question_hash"))
    derived = canon.cache_key(str(row["requested_model"]), state, str(row["question_set_hash"]), question)
    if derived != key:
        found.append(problem("key_mismatch", "the key does not re-derive from the stored state and question"))
    return found
