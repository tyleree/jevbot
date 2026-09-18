"""In-memory `Ledger` double - WP00's shared test double of `ledger.SqliteLedger` (DESIGN.md 2.7, 2.11, 3.4, 13.4).

Same hash chain as the run store (INV-19, INV-24), through the SAME function - `canon.ledger_entry_hash`, the single
implementation of the 2.7 formula:

    hash = sha256(prev_hash + "\\n" + canon.dumps_sorted({"seq", "kind", "session", "as_of", "payload"}))

with `prev_hash = canon.GENESIS_HASH` ("0" * 64) for seq 1, `kind` = the `LedgerKind` value, `session` = `canon.render_session`
(the ISO date), `as_of` = `canon.render_as_of` (the RFC 3339 UTC text msgspec uses for every datetime in a payload:
`2024-05-17T20:00:00Z`; six sub-second digits only when the microseconds are non-zero; any tz-aware instant is normalised to UTC
first) and `payload` = the caller's mapping after the canonical round trip (tuples become lists, str-enums their values;
floats / datetimes / non-string keys are refused BEFORE anything is appended - probabilities are ppm ints, prices are cents).
`SqliteLedger` must append and verify through that same `canon` function over the texts it persists; this double proves the
function against a hand-written literal below. Rows are kept as the SQLite table would keep them (text columns); every
`LedgerEntry` handed out is freshly materialised, so nothing a consumer mutates reaches the chain. The sidecar is stored beside
the rows and is never hashed. `commit_mode` mirrors the store: `per_append` (paper: every write is durable at once,
`rollback()` is a no-op) or `per_session` (backtest: `commit()` once per session; `rollback()` drops every append, fill claim,
state and meta write since the last commit). `states` / `state_index` and `meta` follow 13.4: content-addressed states, one
index row per (session, underlying, request_kind, variant), write-once meta keys. `strict=True` additionally checks every
payload's top-level keys against `types.LEDGER_PAYLOAD_FIELDS` (2.11) and every `put_state` against
`sha256(state_json) == state_hash`.

The self-tests at the bottom are collected by `tests/conftest.py` (`pytest_collect_file`).
"""

import copy
import json
import threading
from collections.abc import Iterator, Mapping
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple

import msgspec

from jevbot.canon import GENESIS_HASH, dumps_sorted, ledger_entry_hash, render_as_of, render_session, sha256_hex
from jevbot.errors import InvariantError, LedgerCorrupt
from jevbot.types import LEDGER_PAYLOAD_FIELDS, LEDGER_PAYLOAD_OPTIONAL_FIELDS, LedgerEntry, LedgerKind

__all__ = ["GENESIS_HASH", "CommitMode", "MemoryLedger", "entry_hash", "render_as_of", "render_session"]

CommitMode = Literal["per_session", "per_append"]
_MODES: Final[frozenset[str]] = frozenset({"per_session", "per_append"})


class _Row(NamedTuple):
    """One `ledger` table row (13.4): text columns, exactly what the store persists and re-hashes."""

    seq: int
    kind: LedgerKind
    session: str
    as_of: str
    payload_json: str
    prev_hash: str
    hash: str


def entry_hash(prev_hash: str, seq: int, kind: LedgerKind, session: str, as_of: str, payload: Mapping[str, Any]) -> str:
    """The 2.7 chain formula over the persisted texts = `canon.ledger_entry_hash` (kept under its old name for the tests that
    check a store against this double)."""
    return ledger_entry_hash(prev_hash, seq, LedgerKind(kind).value, session, as_of, payload)


def _canonical_payload(payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """(payload_json, the payload after the JSON round trip). Raises TypeError / ValueError for non-canonical values (canon)."""
    if not isinstance(payload, Mapping):
        raise TypeError(f"payload must be a mapping, got {type(payload).__name__}")
    text = dumps_sorted(dict(payload))
    loaded: dict[str, Any] = json.loads(text)
    return text, loaded


class MemoryLedger:
    """Implements `protocols.Ledger` in memory (module docstring)."""

    def __init__(self, *, commit_mode: CommitMode = "per_append", strict: bool = False) -> None:
        if commit_mode not in _MODES:
            raise ValueError(f"commit_mode must be one of {sorted(_MODES)}, got {commit_mode!r}")
        self.commit_mode: CommitMode = commit_mode
        self.strict = strict
        self._lock = threading.RLock()
        self._rows: list[_Row] = []
        self._sidecars: dict[int, dict[str, Any]] = {}
        self._fill_ids: dict[str, int] = {}
        self._states: dict[str, str] = {}
        self._index: dict[tuple[str, str, str, str], str] = {}
        self._meta: dict[str, str] = {}
        self._committed = self._snapshot()
        self.commits = 0  # how many commit() calls happened (per-session tests assert one per session)

    # --- transaction bookkeeping -----------------------------------------------------------------------------------

    def _snapshot(
        self,
    ) -> tuple[list[_Row], dict[int, dict[str, Any]], dict[str, int], dict[str, str], dict[tuple[str, str, str, str], str], dict[str, str]]:
        return (
            list(self._rows),
            copy.deepcopy(self._sidecars),
            dict(self._fill_ids),
            dict(self._states),
            dict(self._index),
            dict(self._meta),
        )

    def _autocommit(self) -> None:
        if self.commit_mode == "per_append":
            self._committed = self._snapshot()

    @property
    def uncommitted(self) -> int:
        """Entries appended since the last commit (always 0 in per_append mode)."""
        with self._lock:
            return len(self._rows) - len(self._committed[0])

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)

    # --- Ledger protocol ---------------------------------------------------------------------------------------------

    def append(
        self, kind: LedgerKind, session: date, as_of: datetime, payload: Mapping[str, Any], *, sidecar: Mapping[str, Any] | None = None
    ) -> LedgerEntry:
        k = LedgerKind(kind)
        session_text = render_session(session)
        as_of_text = render_as_of(as_of)
        payload_json, loaded = _canonical_payload(payload)
        if sidecar is not None and not isinstance(sidecar, Mapping):
            raise TypeError(f"sidecar must be a mapping or None, got {type(sidecar).__name__}")
        if self.strict:
            required = set(LEDGER_PAYLOAD_FIELDS[k])
            allowed = required | set(LEDGER_PAYLOAD_OPTIONAL_FIELDS.get(k, ()))
            missing, unknown = sorted(required - set(loaded)), sorted(set(loaded) - allowed)
            if missing or unknown:
                raise InvariantError(f"{k.value} payload does not match 2.11: missing {missing}, unknown {unknown}")
        with self._lock:
            seq = len(self._rows) + 1
            prev = self._rows[-1].hash if self._rows else GENESIS_HASH
            row = _Row(seq, k, session_text, as_of_text, payload_json, prev, entry_hash(prev, seq, k, session_text, as_of_text, loaded))
            self._rows.append(row)
            if sidecar is not None:
                self._sidecars[seq] = copy.deepcopy(dict(sidecar))
            self._autocommit()
            return self._entry(row)

    def commit(self) -> None:
        with self._lock:
            self._committed = self._snapshot()
            self.commits += 1

    def rollback(self) -> None:
        """per_session: drop everything since the last commit (rows, sidecars, fill claims, states, meta); per_append: no-op."""
        with self._lock:
            if self.commit_mode == "per_append":
                return
            rows, sidecars, fills, states, index, meta = self._committed
            self._rows = list(rows)
            self._sidecars = copy.deepcopy(sidecars)
            self._fill_ids = dict(fills)
            self._states = dict(states)
            self._index = dict(index)
            self._meta = dict(meta)

    def head(self) -> tuple[int, str]:
        with self._lock:
            return (self._rows[-1].seq, self._rows[-1].hash) if self._rows else (0, GENESIS_HASH)

    def entries(self, kind: LedgerKind | None = None, since_seq: int = 0) -> Iterator[LedgerEntry]:
        """Entries with `seq > since_seq`, ascending, optionally one kind; iterates a snapshot (appending meanwhile is safe)."""
        if isinstance(since_seq, bool) or not isinstance(since_seq, int) or since_seq < 0:
            raise ValueError(f"since_seq must be an int >= 0, got {since_seq!r}")
        wanted = LedgerKind(kind) if kind is not None else None
        with self._lock:
            rows = self._rows[since_seq:]
        for row in rows:
            if wanted is None or row.kind is wanted:
                yield self._entry(row)

    def verify(self, from_seq: int = 1) -> None:
        """Recompute the chain from `from_seq` (gapless seqs, prev-hash linkage, every hash); raises LedgerCorrupt (INV-19)."""
        with self._lock:
            rows = list(self._rows)
        if isinstance(from_seq, bool) or not isinstance(from_seq, int) or not 1 <= from_seq <= len(rows) + 1:
            raise ValueError(f"from_seq must be in 1..{len(rows) + 1}, got {from_seq!r}")
        for i in range(from_seq - 1, len(rows)):
            row = rows[i]
            prev = rows[i - 1].hash if i > 0 else GENESIS_HASH
            if row.seq != i + 1:
                raise LedgerCorrupt(f"ledger seq gap at position {i + 1}: found seq {row.seq}")
            if row.prev_hash != prev:
                raise LedgerCorrupt(f"ledger seq {row.seq}: prev_hash does not match the hash of seq {row.seq - 1}")
            expected = entry_hash(prev, row.seq, row.kind, row.session, row.as_of, json.loads(row.payload_json))
            if row.hash != expected:
                raise LedgerCorrupt(f"ledger seq {row.seq}: hash does not match its content")

    def claim_fill(self, fill_id: str) -> bool:
        """True the first time a fill id is claimed, False afterwards (the dedupe of the ONE fill path, 9.6)."""
        if not isinstance(fill_id, str) or not fill_id:
            raise ValueError("fill_id must be a non-empty str")
        with self._lock:
            if fill_id in self._fill_ids:
                return False
            self._fill_ids[fill_id] = len(self._rows)
            self._autocommit()
            return True

    def put_state(self, state_hash: str, state_json: str, *, session: date, underlying: str, request_kind: str, variant: str) -> None:
        for name, value in (
            ("state_hash", state_hash),
            ("state_json", state_json),
            ("underlying", underlying),
            ("request_kind", request_kind),
            ("variant", variant),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty str")
        if self.strict and sha256_hex(state_json) != state_hash:
            raise InvariantError("put_state: state_hash is not sha256(state_json) (states are content-addressed, 5.9 / 13.4)")
        key = (render_session(session), underlying, request_kind, variant)
        with self._lock:
            self._states.setdefault(state_hash, state_json)  # INSERT OR IGNORE: first write wins
            existing = self._index.get(key)
            if existing is not None and existing != state_hash:
                raise InvariantError(
                    f"put_state: {key} already indexes state {existing[:12]}, got {state_hash[:12]} (a rebuilt state must be byte-identical, 10.1)"
                )
            self._index[key] = state_hash
            self._autocommit()

    def get_states(self, request_kind: str, variant: str = "base") -> Iterator[tuple[date, str, str]]:
        """(session, underlying, state_json) of every indexed state of that kind / variant, ordered by (session, underlying)."""
        with self._lock:
            found = sorted((k[0], k[1], self._states[h]) for k, h in self._index.items() if k[2] == request_kind and k[3] == variant)
        for session_text, underlying, state_json in found:
            yield (date.fromisoformat(session_text), underlying, state_json)

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            return self._meta.get(key)

    def set_meta(self, key: str, value: str) -> None:
        """Write-once: the same value again is a no-op, a different value is an InvariantError."""
        if not isinstance(key, str) or not key or not isinstance(value, str):
            raise ValueError("meta key must be a non-empty str and the value a str")
        with self._lock:
            existing = self._meta.get(key)
            if existing is not None and existing != value:
                raise InvariantError(f"meta key {key!r} is write-once and already holds a different value")
            self._meta[key] = value
            self._autocommit()

    # --- test conveniences (not part of the Protocol) ----------------------------------------------------------------

    def of_kind(self, kind: LedgerKind) -> list[LedgerEntry]:
        return list(self.entries(kind))

    def last(self, kind: LedgerKind | None = None) -> LedgerEntry | None:
        found = list(self.entries(kind))
        return found[-1] if found else None

    def sidecar(self, seq: int) -> dict[str, Any] | None:
        """The (unhashed) sidecar stored with entry `seq`, or None."""
        with self._lock:
            stored = self._sidecars.get(seq)
            return copy.deepcopy(stored) if stored is not None else None

    @property
    def fill_ids(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._fill_ids)

    def states(self) -> dict[str, str]:
        """state_hash -> state_json (the content-addressed `states` table)."""
        with self._lock:
            return dict(self._states)

    def tamper(self, seq: int, *, payload: Mapping[str, Any] | None = None, hash: str | None = None) -> None:
        """Corrupt a stored row WITHOUT re-hashing (what an attacker with a raw connection could do): `verify()` must catch it."""
        with self._lock:
            i = seq - 1
            if not 0 <= i < len(self._rows):
                raise ValueError(f"no entry with seq {seq}")
            row = self._rows[i]
            payload_json = row.payload_json if payload is None else dumps_sorted(dict(payload))
            self._rows[i] = row._replace(payload_json=payload_json, hash=row.hash if hash is None else hash)

    # --- materialisation --------------------------------------------------------------------------------------------

    @staticmethod
    def _entry(row: _Row) -> LedgerEntry:
        return LedgerEntry(
            seq=row.seq,
            kind=row.kind,
            session=date.fromisoformat(row.session),
            as_of=datetime.fromisoformat(row.as_of),
            payload=json.loads(row.payload_json),
            prev_hash=row.prev_hash,
            hash=row.hash,
        )


if TYPE_CHECKING:
    # Static proof, checked by mypy: MemoryLedger implements the Ledger Protocol of 3.4.
    from jevbot.protocols import Ledger

    _MEMORY_LEDGER_IS_A_LEDGER: Ledger = MemoryLedger()


# ======================================================================================================================
# Self-tests (collected by tests/conftest.py::pytest_collect_file)
# ======================================================================================================================

_SESSION: Final = date(2024, 5, 17)
_AS_OF: Final = datetime(2024, 5, 17, 20, 0, 0, tzinfo=UTC)


def test_memory_ledger_implements_the_ledger_protocol() -> None:
    import inspect

    from jevbot.protocols import Ledger

    for name, member in vars(Ledger).items():
        if name.startswith("_") or not callable(member):
            continue
        want = [(p.name, p.kind, p.default) for p in inspect.signature(member).parameters.values()][1:]
        got = [(p.name, p.kind, p.default) for p in inspect.signature(getattr(MemoryLedger, name)).parameters.values()][1:]
        assert got == want, name


def test_hash_chain_matches_the_formula_of_2_7_with_an_independent_literal() -> None:
    import hashlib

    ledger = MemoryLedger()
    assert ledger.head() == (0, GENESIS_HASH) and len(ledger) == 0
    first = ledger.append(LedgerKind.MARK, _SESSION, _AS_OF, {"b": [1, "x", None, True], "a": {"z": 1, "y": 2}})
    # the material, written by hand: sorted keys, no spaces, ISO date, RFC 3339 "Z" instant, the kind's value
    material = (
        '{"as_of":"2024-05-17T20:00:00Z","kind":"mark","payload":{"a":{"y":2,"z":1},"b":[1,"x",null,true]},"seq":1,"session":"2024-05-17"}'
    )
    expected = hashlib.sha256((GENESIS_HASH + "\n" + material).encode("utf-8")).hexdigest()
    assert first.hash == expected == "86bf32a97b8a585d4deeea62af4ec178fa21421fd73bffdab0701536d441f0ad"  # the literal pins the rendering
    assert first == LedgerEntry(
        seq=1,
        kind=LedgerKind.MARK,
        session=_SESSION,
        as_of=_AS_OF,
        payload={"a": {"y": 2, "z": 1}, "b": [1, "x", None, True]},
        prev_hash=GENESIS_HASH,
        hash=expected,
    )
    assert ledger.head() == (1, expected)
    second = ledger.append(
        LedgerKind.FEE,
        date(2024, 5, 20),
        datetime(2024, 5, 20, 20, 5, 0, 250_000, tzinfo=UTC),
        {"fee_cents": 7, "accrued_micro_before": 65_000},
    )
    material2 = '{"as_of":"2024-05-20T20:05:00.250000Z","kind":"fee","payload":{"accrued_micro_before":65000,"fee_cents":7},"seq":2,"session":"2024-05-20"}'
    assert second.prev_hash == expected and second.hash == hashlib.sha256((expected + "\n" + material2).encode()).hexdigest()
    assert second.as_of == datetime(2024, 5, 20, 20, 5, 0, 250_000, tzinfo=UTC)
    assert (
        entry_hash(
            GENESIS_HASH, 1, LedgerKind.MARK, "2024-05-17", "2024-05-17T20:00:00Z", {"b": (1, "x", None, True), "a": {"z": 1, "y": 2}}
        )
        == expected
    )
    ledger.verify()


def test_the_chain_is_hashed_through_canon_and_spelled_as_msgspec_spells_a_datetime() -> None:
    """INV-24: the double and `SqliteLedger` share ONE function (`canon.ledger_entry_hash`) and ONE spelling of the time columns."""
    from datetime import timedelta, timezone

    from jevbot import canon

    assert GENESIS_HASH == canon.GENESIS_HASH == "0" * 64
    assert render_as_of is canon.render_as_of and render_session is canon.render_session
    ledger = MemoryLedger()
    entry = ledger.append(LedgerKind.MARK, _SESSION, _AS_OF, {"i": 1})
    # the store's `verify()` re-hashes its TEXT columns (13.4): the same function over the texts gives the same hash ...
    assert entry.hash == canon.ledger_entry_hash(GENESIS_HASH, 1, "mark", "2024-05-17", "2024-05-17T20:00:00Z", '{"i":1}')
    # ... and over the objects a store holds at append time
    assert entry.hash == canon.ledger_entry_hash(GENESIS_HASH, 1, LedgerKind.MARK, _SESSION, _AS_OF, {"i": 1})
    # the as_of spelling IS msgspec's rendering of a payload datetime, for whole seconds, microseconds and non-UTC offsets alike
    instants = [
        _AS_OF,
        datetime(2024, 5, 20, 20, 5, 0, 250_000, tzinfo=UTC),
        datetime(2024, 5, 20, 20, 5, 0, 1, tzinfo=UTC),
        datetime(2012, 1, 3, 21, 0, tzinfo=UTC),
        datetime(2024, 5, 17, 16, 0, 0, tzinfo=timezone(timedelta(hours=-4))),
        datetime(2024, 5, 17, 1, 30, 0, 999_999, tzinfo=timezone(timedelta(hours=5, minutes=30))),
    ]
    for instant in instants:
        utc = instant.astimezone(UTC)
        assert render_as_of(instant) == msgspec.to_builtins(utc) == msgspec.json.decode(msgspec.json.encode(utc))
        assert datetime.fromisoformat(render_as_of(instant)) == instant  # the text reads back as the same instant


def test_payload_canonicalisation_and_refusals() -> None:
    import pytest

    from jevbot.types import Band, Slot

    ledger = MemoryLedger()
    entry = ledger.append(
        LedgerKind.SESSION_START,
        _SESSION,
        _AS_OF,
        {"session": "2024-05-17", "slot": Slot.EOD, "phase": "full", "flags": ("a", "b"), "n": {"x": (1, 2)}},
    )
    assert entry.payload == {
        "session": "2024-05-17",
        "slot": "eod",
        "phase": "full",
        "flags": ["a", "b"],
        "n": {"x": [1, 2]},
    }  # the JSON round trip
    assert type(entry.payload["slot"]) is str and type(entry.payload["flags"]) is list
    head = ledger.head()
    for bad in ({"p": 0.5}, {"ts": _AS_OF}, {"d": _SESSION}, {1: "x"}, {"b": b"x"}, {"band": Band.MID, "f": float("nan")}):
        with pytest.raises(TypeError):
            ledger.append(LedgerKind.ANOMALY, _SESSION, _AS_OF, bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ledger.append(LedgerKind.ANOMALY, _SESSION, _AS_OF, ["not", "a", "mapping"])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ledger.append(LedgerKind.ANOMALY, _SESSION, datetime(2024, 5, 17, 20), {"type": "x", "detail": "y"})  # noqa: DTZ001 - naive IS the case
    with pytest.raises(TypeError):
        ledger.append(LedgerKind.ANOMALY, _AS_OF, _AS_OF, {"type": "x", "detail": "y"})  # a datetime is not a session
    with pytest.raises(ValueError):
        ledger.append("no_such_kind", _SESSION, _AS_OF, {})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ledger.append(LedgerKind.ANOMALY, _SESSION, _AS_OF, {"type": "x"}, sidecar=["x"])  # type: ignore[arg-type]
    assert ledger.head() == head and len(ledger) == 1  # nothing was appended by a refused call
    ledger.verify()
    # a str kind value is accepted (the persisted form) and coerced
    assert ledger.append("anomaly", _SESSION, _AS_OF, {"type": "x", "detail": "y"}).kind is LedgerKind.ANOMALY  # type: ignore[arg-type]
    # a non-UTC instant is normalised: same hash as its UTC twin
    from datetime import timedelta, timezone

    a = MemoryLedger().append(LedgerKind.FEE, _SESSION, _AS_OF, {"fee_cents": 1, "accrued_micro_before": 0})
    b = MemoryLedger().append(
        LedgerKind.FEE, _SESSION, _AS_OF.astimezone(timezone(timedelta(hours=-4))), {"fee_cents": 1, "accrued_micro_before": 0}
    )
    assert a.hash == b.hash and b.as_of == _AS_OF and b.as_of.tzinfo == UTC
    assert render_as_of(datetime(2024, 5, 17, 16, 0, 0, tzinfo=timezone(timedelta(hours=-4)))) == "2024-05-17T20:00:00Z"
    assert render_session(_SESSION) == "2024-05-17"
    # what a consumer mutates never reaches the chain: entries are materialised fresh
    got = next(ledger.entries(LedgerKind.SESSION_START))
    got.payload["flags"].append("evil")
    got.payload["phase"] = "decide"
    fresh = next(ledger.entries(LedgerKind.SESSION_START))
    assert fresh.payload["flags"] == ["a", "b"] and fresh.payload["phase"] == "full"
    ledger.verify()
    # the input mapping is not retained either
    inner = {"k": [1]}
    ledger.append(LedgerKind.ANOMALY, _SESSION, _AS_OF, {"type": "x", "detail": inner})
    inner["k"].append(2)
    assert ledger.last(LedgerKind.ANOMALY) is not None and ledger.last(LedgerKind.ANOMALY).payload == {"type": "x", "detail": {"k": [1]}}  # type: ignore[union-attr]


def test_entries_head_since_seq_and_iteration_snapshot() -> None:
    import pytest

    ledger = MemoryLedger()
    for i in range(1, 6):
        ledger.append(LedgerKind.MARK if i % 2 else LedgerKind.FEE, _SESSION, _AS_OF, {"i": i})
    assert [e.seq for e in ledger.entries()] == [1, 2, 3, 4, 5]
    assert [e.seq for e in ledger.entries(LedgerKind.FEE)] == [2, 4]
    assert [e.seq for e in ledger.entries(since_seq=3)] == [4, 5] and [e.seq for e in ledger.entries(LedgerKind.MARK, since_seq=3)] == [5]
    assert [e.seq for e in ledger.entries(since_seq=5)] == [] and [e.seq for e in ledger.entries(since_seq=99)] == []
    assert [e.seq for e in ledger.entries("fee")] == [2, 4]  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        list(ledger.entries(since_seq=-1))
    seen = []
    for e in ledger.entries():
        seen.append(e.seq)
        if e.seq == 2:
            ledger.append(LedgerKind.FEE, _SESSION, _AS_OF, {"i": 6})  # appending during iteration does not extend the walk
    assert seen == [1, 2, 3, 4, 5] and len(ledger) == 6 and ledger.head()[0] == 6
    assert [e.seq for e in ledger.of_kind(LedgerKind.FEE)] == [2, 4, 6] and ledger.last().seq == 6 and ledger.last(LedgerKind.MARK).seq == 5  # type: ignore[union-attr]
    assert MemoryLedger().last() is None
    chain_prev = [e.prev_hash for e in ledger.entries()]
    chain_hash = [e.hash for e in ledger.entries()]
    assert chain_prev == [GENESIS_HASH, *chain_hash[:-1]] and len(set(chain_hash)) == 6


def test_commit_modes_rollback_and_the_transaction_scope() -> None:
    ledger = MemoryLedger(commit_mode="per_session")
    ledger.append(LedgerKind.RUN_START, _SESSION, _AS_OF, {"initial_cash": 1})
    ledger.set_meta("config_hash", "abc")
    ledger.commit()
    assert ledger.commits == 1 and ledger.uncommitted == 0
    ledger.append(LedgerKind.SESSION_START, _SESSION, _AS_OF, {"session": "2024-05-17"})
    ledger.append(LedgerKind.FILL, _SESSION, _AS_OF, {"fill_id": "f1"})
    assert ledger.claim_fill("f1") and ledger.uncommitted == 2
    ledger.put_state("h1", "{}", session=_SESSION, underlying="SPY", request_kind="entry", variant="base")
    ledger.set_meta("namespace", "ns")
    ledger.rollback()
    assert ledger.head()[0] == 1 and ledger.uncommitted == 0 and [e.kind for e in ledger.entries()] == [LedgerKind.RUN_START]
    assert ledger.claim_fill("f1") is True  # the claim was rolled back with the session
    assert list(ledger.get_states("entry")) == [] and ledger.get_meta("namespace") is None and ledger.get_meta("config_hash") == "abc"
    ledger.verify()
    ledger.append(LedgerKind.SESSION_END, _SESSION, _AS_OF, {"n": 1})
    ledger.commit()
    ledger.rollback()  # nothing uncommitted: a no-op
    assert ledger.head()[0] == 2 and ledger.commits == 2 and ledger.fill_ids == frozenset({"f1"})

    paper = MemoryLedger()  # per_append: every write is durable at once
    assert paper.commit_mode == "per_append"
    paper.append(LedgerKind.RUN_START, _SESSION, _AS_OF, {"initial_cash": 1})
    paper.claim_fill("x")
    paper.set_meta("k", "v")
    paper.put_state("h", "{}", session=_SESSION, underlying="SPY", request_kind="entry", variant="base")
    assert paper.uncommitted == 0
    paper.rollback()
    assert (
        paper.head()[0] == 1
        and paper.fill_ids == frozenset({"x"})
        and paper.get_meta("k") == "v"
        and len(list(paper.get_states("entry"))) == 1
    )
    paper.commit()
    assert paper.commits == 1
    import pytest

    with pytest.raises(ValueError):
        MemoryLedger(commit_mode="per_day")  # type: ignore[arg-type]


def test_verify_detects_tampering_and_incremental_verify_starts_where_told() -> None:
    import pytest

    ledger = MemoryLedger()
    for i in range(1, 5):
        ledger.append(LedgerKind.MARK, _SESSION, _AS_OF, {"i": i})
    ledger.verify()
    ledger.verify(from_seq=4)
    ledger.verify(from_seq=5)  # nothing after the head: fine
    with pytest.raises(ValueError):
        ledger.verify(from_seq=0)
    with pytest.raises(ValueError):
        ledger.verify(from_seq=6)
    tampered_payload = MemoryLedger()
    for i in range(1, 5):
        tampered_payload.append(LedgerKind.MARK, _SESSION, _AS_OF, {"i": i})
    tampered_payload.tamper(2, payload={"i": 99})
    with pytest.raises(LedgerCorrupt, match="seq 2"):
        tampered_payload.verify()
    with pytest.raises(LedgerCorrupt, match="seq 2"):
        tampered_payload.verify(from_seq=2)
    tampered_payload.verify(from_seq=3)  # incremental: the linkage from seq 3 on is intact; the full verify at startup catches seq 2
    tampered_hash = MemoryLedger()
    for i in range(1, 4):
        tampered_hash.append(LedgerKind.MARK, _SESSION, _AS_OF, {"i": i})
    tampered_hash.tamper(2, hash="f" * 64)
    with pytest.raises(LedgerCorrupt, match="seq 2"):
        tampered_hash.verify()
    with pytest.raises(LedgerCorrupt, match="seq 3"):
        tampered_hash.verify(from_seq=3)  # seq 3's prev_hash no longer matches seq 2
    with pytest.raises(ValueError):
        tampered_hash.tamper(9, hash="0" * 64)


def test_fill_claims_states_and_meta() -> None:
    import pytest

    ledger = MemoryLedger(strict=True)
    assert ledger.claim_fill("fill-a") is True and ledger.claim_fill("fill-a") is False and ledger.claim_fill("fill-b") is True
    assert ledger.fill_ids == frozenset({"fill-a", "fill-b"})
    with pytest.raises(ValueError):
        ledger.claim_fill("")
    s1, s2 = '{"schema":"state.v1.entry","x":1}', '{"schema":"state.v1.entry","x":2}'
    h1, h2 = sha256_hex(s1), sha256_hex(s2)
    ledger.put_state(h2, s2, session=date(2024, 5, 20), underlying="QQQ", request_kind="entry", variant="base")
    ledger.put_state(h1, s1, session=_SESSION, underlying="SPY", request_kind="entry", variant="base")
    ledger.put_state(h1, s1, session=_SESSION, underlying="SPY", request_kind="entry", variant="base")  # idempotent re-put (10.1 restart)
    ledger.put_state(h1, s1, session=date(2024, 5, 20), underlying="SPY", request_kind="entry", variant="base")  # two sessions, one hash
    ledger.put_state(h2, s2, session=_SESSION, underlying="SPY", request_kind="entry", variant="key_perm")
    ledger.put_state(h2, s2, session=_SESSION, underlying="SPY", request_kind="manage", variant="base")
    assert list(ledger.get_states("entry")) == [(_SESSION, "SPY", s1), (date(2024, 5, 20), "QQQ", s2), (date(2024, 5, 20), "SPY", s1)]
    assert list(ledger.get_states("entry", "key_perm")) == [(_SESSION, "SPY", s2)] and list(ledger.get_states("manage")) == [
        (_SESSION, "SPY", s2)
    ]
    assert list(ledger.get_states("entry_text")) == [] and ledger.states() == {h1: s1, h2: s2}
    with pytest.raises(InvariantError):  # a different state for an indexed key is a determinism bug
        ledger.put_state(h2, s2, session=_SESSION, underlying="SPY", request_kind="entry", variant="base")
    with pytest.raises(InvariantError):  # strict: content addressing
        ledger.put_state("not-the-hash", s1, session=_SESSION, underlying="IWM", request_kind="entry", variant="base")
    with pytest.raises(ValueError):
        ledger.put_state(h1, "", session=_SESSION, underlying="IWM", request_kind="entry", variant="base")
    lax = MemoryLedger()
    lax.put_state("h", "{}", session=_SESSION, underlying="SPY", request_kind="entry", variant="base")
    assert list(lax.get_states("entry")) == [(_SESSION, "SPY", "{}")]
    assert ledger.get_meta("missing") is None
    ledger.set_meta("config_hash", "abc")
    ledger.set_meta("config_hash", "abc")  # the same value again is fine
    with pytest.raises(InvariantError):
        ledger.set_meta("config_hash", "xyz")
    assert ledger.get_meta("config_hash") == "abc"
    with pytest.raises(ValueError):
        ledger.set_meta("", "x")
    with pytest.raises(ValueError):
        ledger.set_meta("k", 1)  # type: ignore[arg-type]


def test_sidecar_is_stored_beside_the_rows_and_never_hashed() -> None:
    plain = MemoryLedger()
    with_sidecar = MemoryLedger()
    payload = {"decision_id": "d1", "kind": "entry"}
    sidecar = {
        "request_id": "req-1",
        "latency_ms": 12,
        "raw_features": {"rv20": 0.1234},
        "ledgered_wall": datetime(2026, 9, 17, 12, tzinfo=UTC),
    }
    a = plain.append(LedgerKind.DECISION, _SESSION, _AS_OF, payload)
    b = with_sidecar.append(LedgerKind.DECISION, _SESSION, _AS_OF, payload, sidecar=sidecar)
    assert a.hash == b.hash and a == b  # floats and datetimes may live in the sidecar; they never touch the chain (INV-24)
    stored = with_sidecar.sidecar(1)
    assert stored == sidecar and stored is not sidecar
    sidecar["request_id"] = "changed"
    assert with_sidecar.sidecar(1) is not None and with_sidecar.sidecar(1)["request_id"] == "req-1"  # type: ignore[index]
    assert plain.sidecar(1) is None and with_sidecar.sidecar(2) is None
    with_sidecar.verify()


def test_strict_payload_keys_follow_2_11() -> None:
    import pytest

    strict = MemoryLedger(strict=True)
    strict.append(LedgerKind.FEE, _SESSION, _AS_OF, {"fee_cents": 3, "accrued_micro_before": 25_000})
    mark = {
        "equity": {"orats": 1, "worst": 1, "mid": 1},
        "cash": {"orats": 1, "worst": 1, "mid": 1},
        "open_max_loss": 0,
        "bp_used": 0,
        "bp_utilisation_ppm": 0,
        "positions": {},
        "net_delta_milli": 0,
        "net_vega_milli": 0,
    }
    strict.append(LedgerKind.MARK, _SESSION, _AS_OF, mark)
    strict.append(
        LedgerKind.MARK, _SESSION, _AS_OF, {**mark, "broker_equity": 1, "broker_prev_equity": 1, "broker_options_bp": 1}
    )  # paper extras
    with pytest.raises(InvariantError, match="missing"):
        strict.append(LedgerKind.FEE, _SESSION, _AS_OF, {"fee_cents": 3})
    with pytest.raises(InvariantError, match="unknown"):
        strict.append(LedgerKind.FEE, _SESSION, _AS_OF, {"fee_cents": 3, "accrued_micro_before": 0, "fee_usd": 0})
    assert strict.head()[0] == 3
    strict.verify()
    MemoryLedger().append(LedgerKind.FEE, _SESSION, _AS_OF, {"anything": "goes"})  # non-strict accepts any canonical payload
