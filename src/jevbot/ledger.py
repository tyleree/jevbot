"""`SqliteLedger` - the run store (`run.sqlite`): one append-only, hash-chained `ledger` table plus the unhashed sidecar,
fill claims, content-addressed states and meta (DESIGN.md 2.7, 2.11, 3.4, 13.4; implements `protocols.Ledger`).

* **The chain (INV-19, INV-24).** Every entry is hashed with `canon.ledger_entry_hash` - THE 2.7 formula - over exactly the
  texts this store persists: `prev_hash + "\\n" + dumps_sorted({"seq", "kind", "session", "as_of", "payload"})`, with
  `prev_hash = canon.GENESIS_HASH` for seq 1, `kind` the `LedgerKind` value, `session` the ISO date (`canon.render_session`),
  `as_of` the RFC 3339 UTC text (`canon.render_as_of`) and `payload` its canonical `dumps_sorted` JSON. `verify()` re-hashes
  the stored texts through the same function, so this store and the `MemoryLedger` double give the same head hash for the
  same entries. A payload holding a float, a datetime or a non-string key is refused BEFORE anything is written.
* **Append-only.** `ledger_no_update` / `ledger_no_delete` triggers abort every UPDATE and DELETE, so tampering needs a raw
  connection with the triggers dropped - which `verify()` then detects (the hash covers every column it re-reads).
* **Nothing wall-clock-shaped is hashed (INV-24).** Run ids, trial ids, request ids, token counts, latency and the wall
  clock live in `meta` and in the `sidecar` table, which is keyed by `seq`, written beside the row and never hashed.
* **Commit modes (10.1).** `per_session` (backtest): one SQLite transaction per session, so a crash never leaves a partial
  session and `rollback()` drops the uncommitted one - its entries AND the fill claims, states and meta written since the
  last commit. `per_append` (paper): every write is durable at once (`synchronous=FULL`) and `rollback()` is a no-op.
* **The other tables (13.4).** `fill_ids` is the UNIQUE dedupe of the ONE fill path (`claim_fill`); `states` /
  `state_index` hold every built state of the run, content-addressed and indexed by (session, underlying, request_kind,
  variant) - re-putting the SAME hash is an idempotent no-op, a DIFFERENT hash for an indexed key is a determinism bug
  (`InvariantError`); `meta` keys are write-once. Outcomes and calibration are VIEWS over the ledger (SQLite JSON1); a
  forecast with a NULL `p_ppm` is a MISSING forecast and is never filtered out.

`verify(from_seq)` is the incremental check every cycle runs; the highest verified seq is kept in the `last_verified_seq`
meta row by this module itself (that row is bookkeeping, not run data, so `set_meta` refuses it).
"""

import json
import sqlite3
import threading
from collections.abc import Iterator, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal

import msgspec

from jevbot.canon import GENESIS_HASH, dumps_sorted, ledger_entry_hash, render_as_of, render_session
from jevbot.errors import InvariantError, LedgerCorrupt
from jevbot.types import LedgerEntry, LedgerKind

__all__ = [
    "GENESIS_HASH",
    "LAST_VERIFIED_SEQ",
    "SCHEMA_SQL",
    "VIEW_COLUMNS",
    "CommitMode",
    "SqliteLedger",
]

CommitMode = Literal["per_session", "per_append"]
_MODES: Final[frozenset[str]] = frozenset({"per_session", "per_append"})

LAST_VERIFIED_SEQ: Final = "last_verified_seq"  # 13.4 meta row, maintained by verify() (write-once `set_meta` refuses it)
_PAGE: Final = 512  # rows per query in the streaming walks (entries / verify): a 10-year run never lands in memory at once
_SCHEMA_VERSION: Final = 1

_LEDGER_COLUMNS: Final[tuple[str, ...]] = ("seq", "kind", "session", "as_of", "payload", "prev_hash", "hash")
_ROW_SQL: Final = "SELECT seq, kind, session, as_of, payload, prev_hash, hash FROM ledger"

# The run-store schema of 13.4, verbatim (plus IF NOT EXISTS so reopening an existing store is a no-op).
SCHEMA_SQL: Final = """
CREATE TABLE IF NOT EXISTS ledger (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, session TEXT NOT NULL, as_of TEXT NOT NULL,
  payload TEXT NOT NULL,
  prev_hash TEXT NOT NULL, hash TEXT NOT NULL UNIQUE);
CREATE INDEX IF NOT EXISTS ledger_kind_session ON ledger(kind, session);
CREATE TABLE IF NOT EXISTS sidecar  (seq INTEGER PRIMARY KEY REFERENCES ledger(seq), wall_created_at TEXT NOT NULL, provenance TEXT, diagnostics TEXT);
CREATE TABLE IF NOT EXISTS fill_ids (fill_id TEXT PRIMARY KEY, seq INTEGER);
CREATE TABLE IF NOT EXISTS states      (state_hash TEXT PRIMARY KEY, state_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS state_index (session TEXT NOT NULL, underlying TEXT NOT NULL, request_kind TEXT NOT NULL, variant TEXT NOT NULL,
  state_hash TEXT NOT NULL REFERENCES states, PRIMARY KEY (session, underlying, request_kind, variant));
CREATE TABLE IF NOT EXISTS meta     (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS ledger_no_update BEFORE UPDATE ON ledger BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS ledger_no_delete BEFORE DELETE ON ledger BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
CREATE VIEW IF NOT EXISTS v_forecasts AS SELECT seq, session, json_extract(payload,'$.forecast_id') AS forecast_id, json_extract(payload,'$.event_key') AS event_key,
  json_extract(payload,'$.decision_id') AS decision_id, json_extract(payload,'$.question_id') AS question_id, json_extract(payload,'$.with_text') AS with_text,
  json_extract(payload,'$.underlying') AS underlying, json_extract(payload,'$.p_ppm') AS p_ppm,
  json_extract(payload,'$.missing_reason') AS missing_reason, json_extract(payload,'$.p_abstain_ppm') AS p_abstain_ppm,
  json_extract(payload,'$.p_implied_ppm') AS p_implied_ppm,
  json_extract(payload,'$.implied_method') AS implied_method, json_extract(payload,'$.implied_quality') AS implied_quality,
  json_extract(payload,'$.spec.horizon_sessions') AS horizon, json_extract(payload,'$.spec.resolve_on') AS resolve_on,
  json_extract(payload,'$.tier') AS tier, json_extract(payload,'$.fidelity') AS fidelity, json_extract(payload,'$.iv_history') AS iv_history,
  json_extract(payload,'$.prereg') AS prereg FROM ledger WHERE kind = 'forecast';
CREATE VIEW IF NOT EXISTS v_outcomes AS SELECT json_extract(payload,'$.event_key') AS event_key, json_extract(payload,'$.resolved_on') AS resolved_on,
  json_extract(payload,'$.y') AS y, json_extract(payload,'$.div_in_window') AS div_in_window FROM ledger WHERE kind = 'outcome';
CREATE VIEW IF NOT EXISTS v_calibration AS SELECT f.*, o.y, o.resolved_on, o.div_in_window FROM v_forecasts f JOIN v_outcomes o USING (event_key);
CREATE VIEW IF NOT EXISTS v_daily AS SELECT session, payload FROM ledger WHERE kind = 'session_end';
CREATE VIEW IF NOT EXISTS v_fills AS SELECT session, payload FROM ledger WHERE kind = 'fill';
CREATE VIEW IF NOT EXISTS v_decisions AS SELECT session, payload FROM ledger WHERE kind = 'decision';
"""

_FORECAST_COLUMNS: Final[tuple[str, ...]] = (
    "seq",
    "session",
    "forecast_id",
    "event_key",
    "decision_id",
    "question_id",
    "with_text",
    "underlying",
    "p_ppm",  # NULL = a MISSING forecast (never filtered out, 6.4 / 12.3)
    "missing_reason",
    "p_abstain_ppm",
    "p_implied_ppm",
    "implied_method",
    "implied_quality",
    "horizon",
    "resolve_on",
    "tier",
    "fidelity",
    "iv_history",
    "prereg",
)

# The columns every view of 13.4 returns, in order (asserted by the tests and read by `eval/load.py`).
VIEW_COLUMNS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "v_forecasts": _FORECAST_COLUMNS,
        "v_outcomes": ("event_key", "resolved_on", "y", "div_in_window"),
        "v_calibration": (*_FORECAST_COLUMNS, "y", "resolved_on", "div_in_window"),
        "v_daily": ("session", "payload"),
        "v_fills": ("session", "payload"),
        "v_decisions": ("session", "payload"),
    }
)

# sidecar (13.4): `wall_created_at` is the only wall clock this store keeps; `provenance` holds the Provenance of 2.7 and
# `diagnostics` everything else. None of it is hashed (INV-24).
_SIDECAR_WALL_KEYS: Final[tuple[str, ...]] = ("wall_created_at", "ledgered_wall")
_SIDECAR_COLUMN_KEYS: Final[frozenset[str]] = frozenset({*_SIDECAR_WALL_KEYS, "provenance", "diagnostics"})


def _json(value: object) -> str:
    """Sidecar JSON: `msgspec` encoding, so floats, datetimes and Structs are welcome (nothing here is hashed)."""
    return msgspec.json.encode(value).decode("utf-8")


def _sidecar_row(sidecar: Mapping[str, Any]) -> tuple[str, str | None, str | None]:
    """`(wall_created_at, provenance, diagnostics)` for the 13.4 sidecar columns.

    `wall_created_at` comes from the mapping (`wall_created_at` or `ledgered_wall`, a tz-aware datetime or its text) and
    otherwise from the wall clock; `provenance` is that key's value as JSON; every other key lands in `diagnostics`.
    """
    raw = sidecar.get(_SIDECAR_WALL_KEYS[0], sidecar.get(_SIDECAR_WALL_KEYS[1]))
    if raw is None:
        wall = render_as_of(datetime.now(UTC))
    elif isinstance(raw, datetime):
        wall = render_as_of(raw)
    elif isinstance(raw, str):
        wall = raw
    else:
        raise TypeError(f"sidecar wall_created_at must be a tz-aware datetime or its text, got {type(raw).__name__}")
    provenance = sidecar.get("provenance")
    extra = {key: value for key, value in sidecar.items() if key not in _SIDECAR_COLUMN_KEYS}
    stated = sidecar.get("diagnostics")
    diagnostics: Any
    if stated is None:
        diagnostics = extra or None
    elif not extra:
        diagnostics = stated
    elif isinstance(stated, Mapping):
        diagnostics = {**stated, **extra}
    else:
        diagnostics = {"diagnostics": stated, **extra}
    return (wall, None if provenance is None else _json(provenance), None if diagnostics is None else _json(diagnostics))


class SqliteLedger:
    """Implements `protocols.Ledger` over one `run.sqlite` file (module docstring).

    `commit_mode` is `per_session` for backtests (`commit()` once per session; `rollback()` drops the uncommitted one) and
    `per_append` for paper (every write durable at once). The connection is guarded by a re-entrant lock, so a worker
    thread may append; `close()` (or the context manager) drops an uncommitted session - the store holds whole sessions.
    """

    def __init__(self, path: Path | str, *, commit_mode: CommitMode = "per_append") -> None:
        if commit_mode not in _MODES:
            raise ValueError(f"commit_mode must be one of {sorted(_MODES)}, got {commit_mode!r}")
        if isinstance(path, str | Path):
            store = Path(path)
        else:
            raise TypeError(f"path must be a Path or str, got {type(path).__name__}")
        store.parent.mkdir(parents=True, exist_ok=True, mode=0o700)  # $JEVBOT_DATA is mode 700 (D1); a missing run dir is created
        self.path: Final = store
        self.commit_mode: CommitMode = commit_mode
        self.commits = 0  # how many commit() calls happened (the per-session tests assert one per session)
        self._lock = threading.RLock()
        self._closed = False
        self._uncommitted = 0
        self._conn = sqlite3.connect(store, isolation_level=None, check_same_thread=False)
        try:
            self._configure()
            self._seq, self._hash = self._read_head()
        except BaseException:
            self._conn.close()
            self._closed = True
            raise

    # --- construction -------------------------------------------------------------------------------------------------

    def _configure(self) -> None:
        """13.4: WAL + `synchronous=FULL`; foreign keys on; then the schema (idempotent) and a shape check.

        A file that is not a run store of this schema is refused before anything is written to it."""
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        if self._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'ledger'").fetchone():
            self._check_ledger_shape()
        try:
            self._conn.executescript(SCHEMA_SQL)
        except sqlite3.DatabaseError as exc:
            raise LedgerCorrupt(f"{self.path}: the run store schema of 13.4 could not be applied ({exc})") from exc
        self._conn.commit()
        self._check_ledger_shape()
        version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if version == 0:
            self._conn.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
        elif version != _SCHEMA_VERSION:
            raise LedgerCorrupt(f"{self.path}: run store schema version {version}, this build writes {_SCHEMA_VERSION}")

    def _check_ledger_shape(self) -> None:
        columns = tuple(str(row[1]) for row in self._conn.execute("PRAGMA table_info(ledger)"))
        if columns != _LEDGER_COLUMNS:
            raise LedgerCorrupt(f"{self.path}: `ledger` has columns {columns}, expected {_LEDGER_COLUMNS} (13.4)")

    def _read_head(self) -> tuple[int, str]:
        row = self._conn.execute("SELECT seq, hash FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
        return (0, GENESIS_HASH) if row is None else (int(row[0]), str(row[1]))

    # --- transaction bookkeeping --------------------------------------------------------------------------------------

    def _require_open(self) -> None:
        if self._closed:
            raise InvariantError(f"{self.path}: the run store is closed")

    def _begin(self) -> None:
        """Open the write transaction if none is open (`per_append` closes it again at once; `per_session` at `commit()`)."""
        if not self._conn.in_transaction:
            self._conn.execute("BEGIN IMMEDIATE")

    def _autocommit(self) -> None:
        if self.commit_mode == "per_append":
            self._conn.commit()
            self._uncommitted = 0

    @property
    def uncommitted(self) -> int:
        """Entries appended since the last commit (always 0 in `per_append` mode)."""
        with self._lock:
            return self._uncommitted

    def __len__(self) -> int:
        with self._lock:
            return self._seq

    def __enter__(self) -> "SqliteLedger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Roll back an uncommitted session (the store holds whole sessions only) and close the connection."""
        with self._lock:
            if self._closed:
                return
            if self._conn.in_transaction:
                self._conn.rollback()
            self._uncommitted = 0
            self._conn.close()
            self._closed = True

    # --- Ledger protocol ----------------------------------------------------------------------------------------------

    def append(
        self, kind: LedgerKind, session: date, as_of: datetime, payload: Mapping[str, Any], *, sidecar: Mapping[str, Any] | None = None
    ) -> LedgerEntry:
        """Append one entry and return it. The payload is canonicalised and the hash computed BEFORE anything is written,
        so a float / datetime / non-string key (TypeError), a naive `as_of` (ValueError) or an unknown kind (ValueError)
        leaves the chain untouched. `sidecar` is stored beside the row and never hashed (2.7, INV-24)."""
        k = LedgerKind(kind)
        session_text = render_session(session)
        as_of_text = render_as_of(as_of)
        if not isinstance(payload, Mapping):
            raise TypeError(f"payload must be a mapping, got {type(payload).__name__}")
        payload_json = dumps_sorted(dict(payload))
        loaded: dict[str, Any] = json.loads(payload_json)
        if sidecar is not None and not isinstance(sidecar, Mapping):
            raise TypeError(f"sidecar must be a mapping or None, got {type(sidecar).__name__}")
        columns = None if sidecar is None else _sidecar_row(sidecar)
        with self._lock:
            self._require_open()
            seq = self._seq + 1
            prev = self._hash
            digest = ledger_entry_hash(prev, seq, k.value, session_text, as_of_text, loaded)
            self._begin()
            self._conn.execute(
                "INSERT INTO ledger (seq, kind, session, as_of, payload, prev_hash, hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (seq, k.value, session_text, as_of_text, payload_json, prev, digest),
            )
            if columns is not None:
                self._conn.execute(
                    "INSERT INTO sidecar (seq, wall_created_at, provenance, diagnostics) VALUES (?, ?, ?, ?)", (seq, *columns)
                )
            self._seq, self._hash = seq, digest
            self._uncommitted += 1
            self._autocommit()
        return LedgerEntry(
            seq=seq,
            kind=k,
            session=date.fromisoformat(session_text),
            as_of=datetime.fromisoformat(as_of_text),
            payload=loaded,
            prev_hash=prev,
            hash=digest,
        )

    def commit(self) -> None:
        """Make everything written since the last commit durable (backtest: once per session; paper: after every append)."""
        with self._lock:
            self._require_open()
            self._conn.commit()
            self._uncommitted = 0
            self.commits += 1

    def rollback(self) -> None:
        """`per_session`: drop the uncommitted session - its entries, fill claims, states and meta (one SQLite
        transaction). `per_append`: a no-op (everything is already durable)."""
        with self._lock:
            self._require_open()
            if self.commit_mode == "per_append":
                return
            if self._conn.in_transaction:
                self._conn.rollback()
            self._uncommitted = 0
            self._seq, self._hash = self._read_head()

    def head(self) -> tuple[int, str]:
        """`(seq, hash)` of the last entry; `(0, "0" * 64)` when the store is empty."""
        with self._lock:
            self._require_open()
            return (self._seq, self._hash)

    def entries(self, kind: LedgerKind | None = None, since_seq: int = 0) -> Iterator[LedgerEntry]:
        """Entries with `seq > since_seq`, ascending, optionally one kind. The walk is bounded by the head at call time
        (appending meanwhile does not extend it) and streams in pages, so a long run never lands in memory at once."""
        if isinstance(since_seq, bool) or not isinstance(since_seq, int) or since_seq < 0:
            raise ValueError(f"since_seq must be an int >= 0, got {since_seq!r}")
        wanted = LedgerKind(kind).value if kind is not None else None
        with self._lock:
            self._require_open()
            stop = self._seq
        last = since_seq
        while last < stop:
            sql = f"{_ROW_SQL} WHERE seq > ? AND seq <= ?"
            params: list[object] = [last, stop]
            if wanted is not None:
                sql += " AND kind = ?"
                params.append(wanted)
            sql += " ORDER BY seq LIMIT ?"
            params.append(_PAGE)
            with self._lock:
                self._require_open()
                rows = self._conn.execute(sql, params).fetchall()
            if not rows:
                return
            for row in rows:
                yield self._entry(row)
            last = int(rows[-1][0])

    def verify(self, from_seq: int = 1) -> None:
        """Recompute the chain from `from_seq` over the PERSISTED texts - gapless seqs, prev-hash linkage, every hash
        through `canon.ledger_entry_hash` - and raise `LedgerCorrupt` on the first mismatch (INV-19).

        A successful walk that starts no later than one past the highest verified seq advances the `last_verified_seq`
        meta row, which is what the incremental per-cycle check reads.
        """
        with self._lock:
            self._require_open()
            count = int(self._conn.execute("SELECT COUNT(*) FROM ledger").fetchone()[0])
        if isinstance(from_seq, bool) or not isinstance(from_seq, int) or not 1 <= from_seq <= count + 1:
            raise ValueError(f"from_seq must be in 1..{count + 1}, got {from_seq!r}")
        prev = GENESIS_HASH
        if from_seq > 1:
            with self._lock:
                row = self._conn.execute("SELECT hash FROM ledger WHERE seq = ?", (from_seq - 1,)).fetchone()
            if row is None:
                raise LedgerCorrupt(f"ledger seq gap at position {from_seq - 1}: no such entry")
            prev = str(row[0])
        expected = from_seq
        while True:
            with self._lock:
                self._require_open()
                rows = self._conn.execute(f"{_ROW_SQL} WHERE seq >= ? ORDER BY seq LIMIT ?", (expected, _PAGE)).fetchall()
            if not rows:
                break
            for seq, kind, session, as_of, payload, prev_hash, stored in rows:
                prev = self._verify_row(
                    int(seq), str(kind), str(session), str(as_of), str(payload), str(prev_hash), str(stored), expected, prev
                )
                expected += 1
        self._advance_verified(from_seq, count)

    @staticmethod
    def _verify_row(
        seq: int, kind: str, session: str, as_of: str, payload: str, prev_hash: str, stored: str, expected: int, prev: str
    ) -> str:
        """Check one persisted row against the chain and return its hash as the next `prev_hash`."""
        if seq != expected:
            raise LedgerCorrupt(f"ledger seq gap at position {expected}: found seq {seq}")
        if prev_hash != prev:
            raise LedgerCorrupt(f"ledger seq {seq}: prev_hash does not match the hash of seq {seq - 1}")
        try:
            LedgerKind(kind)
            recomputed = ledger_entry_hash(prev, seq, kind, session, as_of, payload)
        except (TypeError, ValueError) as exc:
            raise LedgerCorrupt(f"ledger seq {seq}: unreadable row ({exc})") from exc
        if stored != recomputed:
            raise LedgerCorrupt(f"ledger seq {seq}: hash does not match its content")
        return stored

    def _advance_verified(self, from_seq: int, count: int) -> None:
        if from_seq <= self.last_verified_seq + 1 and count > self.last_verified_seq:
            with self._lock:
                self._begin()
                self._conn.execute(
                    "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (LAST_VERIFIED_SEQ, str(count)),
                )
                self._autocommit()

    @property
    def last_verified_seq(self) -> int:
        """The highest seq a `verify()` of this store has checked (0 when it has never been verified)."""
        stored = self.get_meta(LAST_VERIFIED_SEQ)
        try:
            return 0 if stored is None else int(stored)
        except ValueError as exc:
            raise LedgerCorrupt(f"{self.path}: meta {LAST_VERIFIED_SEQ} is not an integer ({stored!r})") from exc

    def claim_fill(self, fill_id: str) -> bool:
        """True the first time a fill id is claimed, False afterwards: the dedupe of the ONE fill path (9.6)."""
        if not isinstance(fill_id, str) or not fill_id:
            raise ValueError("fill_id must be a non-empty str")
        with self._lock:
            self._require_open()
            self._begin()
            try:
                cursor = self._conn.execute("INSERT OR IGNORE INTO fill_ids (fill_id, seq) VALUES (?, ?)", (fill_id, self._seq))
                return cursor.rowcount == 1
            finally:
                self._autocommit()

    def put_state(self, state_hash: str, state_json: str, *, session: date, underlying: str, request_kind: str, variant: str) -> None:
        """Record one built state (13.4): content-addressed in `states`, indexed by (session, underlying, request_kind,
        variant) in `state_index`. Re-putting the same hash for the same key is an idempotent no-op (the 10.1 restart
        case); a DIFFERENT hash for an indexed key is a determinism bug and raises `InvariantError`."""
        for name, value in (
            ("state_hash", state_hash),
            ("state_json", state_json),
            ("underlying", underlying),
            ("request_kind", request_kind),
            ("variant", variant),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty str")
        key = (render_session(session), underlying, request_kind, variant)
        with self._lock:
            self._require_open()
            row = self._conn.execute(
                "SELECT state_hash FROM state_index WHERE session = ? AND underlying = ? AND request_kind = ? AND variant = ?", key
            ).fetchone()
            if row is not None and str(row[0]) != state_hash:
                raise InvariantError(
                    f"put_state: {key} already indexes state {str(row[0])[:12]}, got {state_hash[:12]} "
                    "(a rebuilt state must be byte-identical, 10.1)"
                )
            self._begin()
            self._conn.execute("INSERT OR IGNORE INTO states (state_hash, state_json) VALUES (?, ?)", (state_hash, state_json))
            if row is None:
                self._conn.execute(
                    "INSERT INTO state_index (session, underlying, request_kind, variant, state_hash) VALUES (?, ?, ?, ?, ?)",
                    (*key, state_hash),
                )
            self._autocommit()

    def get_states(self, request_kind: str, variant: str = "base") -> Iterator[tuple[date, str, str]]:
        """`(session, underlying, state_json)` of every indexed state of that kind / variant, ordered by (session,
        underlying): THE source of baseline 6's recorded states (12.4) and of the probe suites' `run:RUN_ID` source."""
        with self._lock:
            self._require_open()
            rows = self._conn.execute(
                "SELECT i.session, i.underlying, s.state_json FROM state_index i JOIN states s ON s.state_hash = i.state_hash "
                "WHERE i.request_kind = ? AND i.variant = ? ORDER BY i.session, i.underlying",
                (request_kind, variant),
            ).fetchall()
        for session, underlying, state_json in rows:
            yield (date.fromisoformat(str(session)), str(underlying), str(state_json))

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            self._require_open()
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        """Write-once: the same value again is a no-op, a different value is an `InvariantError`. `last_verified_seq` is
        this module's own bookkeeping and is refused here - `verify()` maintains it."""
        if not isinstance(key, str) or not key or not isinstance(value, str):
            raise ValueError("meta key must be a non-empty str and the value a str")
        if key == LAST_VERIFIED_SEQ:
            raise InvariantError(f"meta key {LAST_VERIFIED_SEQ!r} is maintained by verify(), not by set_meta()")
        with self._lock:
            self._require_open()
            existing = self.get_meta(key)
            if existing is not None:
                if existing != value:
                    raise InvariantError(f"meta key {key!r} is write-once and already holds a different value")
                return
            self._begin()
            self._conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)", (key, value))
            self._autocommit()

    # --- materialisation ------------------------------------------------------------------------------------------------

    @staticmethod
    def _entry(row: tuple[Any, ...]) -> LedgerEntry:
        """One persisted row as a `LedgerEntry` (a fresh object every time: nothing a consumer mutates reaches the store)."""
        seq, kind, session, as_of, payload, prev_hash, digest = row
        try:
            loaded = json.loads(payload)
            if not isinstance(loaded, dict):
                raise TypeError("payload is not a JSON object")
            entry = LedgerEntry(
                seq=int(seq),
                kind=LedgerKind(kind),
                session=date.fromisoformat(str(session)),
                as_of=datetime.fromisoformat(str(as_of)),
                payload=loaded,
                prev_hash=str(prev_hash),
                hash=str(digest),
            )
        except (TypeError, ValueError) as exc:
            raise LedgerCorrupt(f"ledger seq {seq}: unreadable row ({exc})") from exc
        return entry

    # --- reads beside the Protocol --------------------------------------------------------------------------------------

    def sidecar(self, seq: int) -> dict[str, Any] | None:
        """The unhashed sidecar of entry `seq` as `{"wall_created_at", "provenance", "diagnostics"}` (13.4), or None."""
        with self._lock:
            self._require_open()
            row = self._conn.execute("SELECT wall_created_at, provenance, diagnostics FROM sidecar WHERE seq = ?", (seq,)).fetchone()
        if row is None:
            return None
        wall, provenance, diagnostics = row
        return {
            "wall_created_at": str(wall),
            "provenance": None if provenance is None else json.loads(provenance),
            "diagnostics": None if diagnostics is None else json.loads(diagnostics),
        }

    def fill_ids(self) -> tuple[str, ...]:
        """Every claimed fill id, ascending (the dedupe table of 9.6)."""
        with self._lock:
            self._require_open()
            rows = self._conn.execute("SELECT fill_id FROM fill_ids ORDER BY fill_id").fetchall()
        return tuple(str(row[0]) for row in rows)

    def view(self, name: str) -> tuple[tuple[str, ...], list[tuple[Any, ...]]]:
        """`(column names, rows)` of one documented view of 13.4 (`v_forecasts`, `v_outcomes`, `v_calibration`, `v_daily`,
        `v_fills`, `v_decisions`). Reports read these; a forecast with a NULL `p_ppm` is MISSING, never filtered out."""
        if name not in VIEW_COLUMNS:
            raise ValueError(f"unknown view {name!r}; the run store has {sorted(VIEW_COLUMNS)}")
        with self._lock:
            self._require_open()
            cursor = self._conn.execute(f"SELECT * FROM {name}")  # name is one of the frozen keys above
            columns = tuple(str(column[0]) for column in cursor.description)
            rows = cursor.fetchall()
        return (columns, [tuple(row) for row in rows])


if TYPE_CHECKING:
    # Static proof, checked by mypy: SqliteLedger implements the Ledger Protocol of 3.4.
    from jevbot.protocols import Ledger

    def _sqlite_ledger_is_a_ledger(path: Path) -> "Ledger":
        return SqliteLedger(path)
