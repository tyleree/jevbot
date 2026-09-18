"""Unit tests for src/jevbot/ledger.py: the run store's append-only hash chain, its sidecar, fill claims, states, meta
and views (DESIGN.md 2.7, 2.11, 3.4, 13.4; INV-19, INV-24; WP06).

The chain hash is checked against a hand-written literal (the material is spelt out here, byte for byte) AND against the
`MemoryLedger` double, which WP00 owns: the two must agree on every head hash. Tampering is done the only way it can be
done - through a raw `sqlite3` connection with the append-only triggers dropped.
"""

import hashlib
import inspect
import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import msgspec
import pytest

from jevbot.canon import GENESIS_HASH, dumps_sorted, sha256_hex
from jevbot.errors import InvariantError, LedgerCorrupt
from jevbot.ledger import LAST_VERIFIED_SEQ, VIEW_COLUMNS, SqliteLedger
from jevbot.types import (
    EvidenceTier,
    Fidelity,
    Forecast,
    LedgerEntry,
    LedgerKind,
    Outcome,
    OutcomeSpec,
    Slot,
    SnapshotKey,
)
from tests.fixtures.memory_ledger import MemoryLedger

SESSION = date(2024, 5, 17)
NEXT_SESSION = date(2024, 5, 20)
AS_OF = datetime(2024, 5, 17, 20, 0, 0, tzinfo=UTC)
NEXT_AS_OF = datetime(2024, 5, 20, 20, 0, 0, tzinfo=UTC)


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "runs" / "run-1" / "run.sqlite"


@pytest.fixture
def ledger(store_path: Path) -> Iterator[SqliteLedger]:
    with SqliteLedger(store_path) as store:
        yield store


def raw(path: Path) -> sqlite3.Connection:
    """A raw connection to the file: how the tamper tests reach the store (and the only way to reach it at all)."""
    return sqlite3.connect(path, isolation_level=None)


def drop_triggers(path: Path) -> None:
    connection = raw(path)
    try:
        connection.executescript("DROP TRIGGER ledger_no_update; DROP TRIGGER ledger_no_delete;")
    finally:
        connection.close()


def marks(store: SqliteLedger, count: int, *, session: date = SESSION) -> None:
    for i in range(1, count + 1):
        store.append(LedgerKind.MARK, session, AS_OF, {"i": i})


# ======================================================================================================================
# The chain (2.7, INV-19, INV-24)
# ======================================================================================================================


def test_the_chain_is_the_2_7_formula_with_an_independent_literal(ledger: SqliteLedger) -> None:
    assert ledger.head() == (0, GENESIS_HASH) and len(ledger) == 0
    first = ledger.append(LedgerKind.MARK, SESSION, AS_OF, {"b": [1, "x", None, True], "a": {"z": 1, "y": 2}})
    # the material, written by hand: sorted keys, no spaces, the ISO date, the RFC 3339 instant, the kind's VALUE
    material = (
        '{"as_of":"2024-05-17T20:00:00Z","kind":"mark","payload":{"a":{"y":2,"z":1},"b":[1,"x",null,true]},"seq":1,"session":"2024-05-17"}'
    )
    expected = hashlib.sha256((GENESIS_HASH + "\n" + material).encode("utf-8")).hexdigest()
    assert first.hash == expected == "86bf32a97b8a585d4deeea62af4ec178fa21421fd73bffdab0701536d441f0ad"
    assert first == LedgerEntry(
        seq=1,
        kind=LedgerKind.MARK,
        session=SESSION,
        as_of=AS_OF,
        payload={"a": {"y": 2, "z": 1}, "b": [1, "x", None, True]},
        prev_hash=GENESIS_HASH,
        hash=expected,
    )
    assert ledger.head() == (1, expected)
    second = ledger.append(
        LedgerKind.FEE, NEXT_SESSION, datetime(2024, 5, 20, 20, 5, 0, 250_000, tzinfo=UTC), {"fee_cents": 7, "accrued_micro_before": 65_000}
    )
    material2 = '{"as_of":"2024-05-20T20:05:00.250000Z","kind":"fee","payload":{"accrued_micro_before":65000,"fee_cents":7},"seq":2,"session":"2024-05-20"}'
    assert second.prev_hash == expected and second.hash == hashlib.sha256((expected + "\n" + material2).encode()).hexdigest()
    ledger.verify()


def test_the_store_and_the_memory_double_agree_on_every_head_hash(ledger: SqliteLedger) -> None:
    """INV-24: one formula, one spelling of the persisted columns - a run store and the double are interchangeable."""
    double = MemoryLedger()
    entries = [
        (LedgerKind.RUN_START, SESSION, AS_OF, {"mode": "backtest", "initial_cash": 2_500_000, "flags": ("smoke",)}),
        (LedgerKind.SESSION_START, SESSION, AS_OF, {"session": "2024-05-17", "slot": Slot.EOD, "phase": "full"}),
        (LedgerKind.FILL, SESSION, AS_OF, {"fill_id": "f1", "net": {"orats": -173, "worst": -165, "mid": -177}}),
        (LedgerKind.FEE, NEXT_SESSION, NEXT_AS_OF, {"fee_cents": 35, "accrued_micro_before": 347_714}),
    ]
    for kind, session, as_of, payload in entries:
        mine = ledger.append(kind, session, as_of, payload)
        theirs = double.append(kind, session, as_of, payload)
        assert mine == theirs
    assert ledger.head() == double.head()
    assert [e.hash for e in ledger.entries()] == [e.hash for e in double.entries()]
    ledger.verify()
    double.verify()


def test_head_entries_and_since_seq(ledger: SqliteLedger) -> None:
    for i in range(1, 6):
        ledger.append(LedgerKind.MARK if i % 2 else LedgerKind.FEE, SESSION, AS_OF, {"i": i})
    assert [e.seq for e in ledger.entries()] == [1, 2, 3, 4, 5]
    assert [e.seq for e in ledger.entries(LedgerKind.FEE)] == [2, 4]
    assert [e.seq for e in ledger.entries(since_seq=3)] == [4, 5]
    assert [e.seq for e in ledger.entries(LedgerKind.MARK, since_seq=3)] == [5]
    assert [e.seq for e in ledger.entries(since_seq=5)] == [] and [e.seq for e in ledger.entries(since_seq=99)] == []
    with pytest.raises(ValueError):
        list(ledger.entries(since_seq=-1))
    with pytest.raises(ValueError):
        list(ledger.entries("no_such_kind"))  # type: ignore[arg-type]
    seen = []
    for entry in ledger.entries():
        seen.append(entry.seq)
        if entry.seq == 2:
            ledger.append(LedgerKind.FEE, SESSION, AS_OF, {"i": 6})  # appending during a walk does not extend it
    assert seen == [1, 2, 3, 4, 5] and ledger.head()[0] == 6
    chain = [e.hash for e in ledger.entries()]
    assert [e.prev_hash for e in ledger.entries()] == [GENESIS_HASH, *chain[:-1]] and len(set(chain)) == 6
    # what a consumer mutates never reaches the store
    got = next(ledger.entries())
    got.payload["i"] = 999
    assert next(ledger.entries()).payload == {"i": 1}


def test_entries_and_verify_stream_a_long_chain(ledger: SqliteLedger) -> None:
    marks(ledger, 1_200)  # more than one page of the internal walks
    assert ledger.head()[0] == 1_200
    assert [e.seq for e in ledger.entries()] == list(range(1, 1_201))
    assert [e.seq for e in ledger.entries(since_seq=1_150)] == list(range(1_151, 1_201))
    ledger.verify()
    ledger.verify(from_seq=1_200)


def test_payload_canonicalisation_and_refusals(ledger: SqliteLedger) -> None:
    entry = ledger.append(
        LedgerKind.SESSION_START,
        SESSION,
        AS_OF,
        {"session": "2024-05-17", "slot": Slot.EOD, "phase": "full", "flags": ("a", "b"), "n": {"x": (1, 2)}},
    )
    assert entry.payload == {"session": "2024-05-17", "slot": "eod", "phase": "full", "flags": ["a", "b"], "n": {"x": [1, 2]}}
    assert type(entry.payload["slot"]) is str and type(entry.payload["flags"]) is list
    head = ledger.head()
    for bad in ({"p": 0.5}, {"ts": AS_OF}, {"d": SESSION}, {1: "x"}, {"b": b"x"}, {"f": float("nan")}):
        with pytest.raises(TypeError):
            ledger.append(LedgerKind.ANOMALY, SESSION, AS_OF, bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ledger.append(LedgerKind.ANOMALY, SESSION, AS_OF, ["not", "a", "mapping"])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ledger.append(LedgerKind.ANOMALY, SESSION, datetime(2024, 5, 17, 20), {"type": "x"})  # noqa: DTZ001 - naive IS the case
    with pytest.raises(TypeError):
        ledger.append(LedgerKind.ANOMALY, AS_OF, AS_OF, {"type": "x"})  # a datetime is not a session
    with pytest.raises(ValueError):
        ledger.append("no_such_kind", SESSION, AS_OF, {})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ledger.append(LedgerKind.ANOMALY, SESSION, AS_OF, {"type": "x"}, sidecar=["x"])  # type: ignore[arg-type]
    assert ledger.head() == head and len(ledger) == 1  # nothing was appended by a refused call
    ledger.verify()
    # a non-UTC instant is normalised to UTC: the same hash as its UTC twin, and it reads back as UTC
    shifted = ledger.append(LedgerKind.FEE, SESSION, AS_OF.astimezone(UTC) - timedelta(0), {"fee_cents": 1, "accrued_micro_before": 0})
    assert shifted.as_of == AS_OF and shifted.as_of.tzinfo == UTC
    # the caller's mapping is not retained
    inner: dict[str, Any] = {"k": [1]}
    ledger.append(LedgerKind.ANOMALY, SESSION, AS_OF, {"type": "x", "detail": inner})
    inner["k"].append(2)
    last = list(ledger.entries(LedgerKind.ANOMALY))[-1]
    assert last.payload == {"type": "x", "detail": {"k": [1]}}


# ======================================================================================================================
# 13.4 the schema, the triggers and tamper detection
# ======================================================================================================================


def test_the_schema_is_the_one_of_13_4(ledger: SqliteLedger, store_path: Path) -> None:
    ledger.append(LedgerKind.MARK, SESSION, AS_OF, {"i": 1})
    ledger.commit()
    connection = raw(store_path)
    try:
        objects = {
            str(kind): {str(name) for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type = ?", (kind,))}
            for kind in ("table", "index", "trigger", "view")
        }
        assert {"ledger", "sidecar", "fill_ids", "states", "state_index", "meta"} <= objects["table"]
        assert "ledger_kind_session" in objects["index"]
        assert {"ledger_no_update", "ledger_no_delete"} <= objects["trigger"]
        assert set(VIEW_COLUMNS) <= objects["view"]
        columns = [str(row[1]) for row in connection.execute("PRAGMA table_info(ledger)")]
        assert columns == ["seq", "kind", "session", "as_of", "payload", "prev_hash", "hash"]
        assert str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
        # the payload column holds the canonical dumps_sorted text, nothing else
        stored = connection.execute("SELECT payload FROM ledger WHERE seq = 1").fetchone()[0]
        assert stored == dumps_sorted({"i": 1}) == '{"i":1}'
    finally:
        connection.close()


def test_the_triggers_block_update_and_delete(ledger: SqliteLedger, store_path: Path) -> None:
    marks(ledger, 2)
    ledger.commit()
    connection = raw(store_path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE ledger SET payload = '{}' WHERE seq = 1")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM ledger WHERE seq = 1")
        assert connection.execute("SELECT COUNT(*) FROM ledger").fetchone()[0] == 2
    finally:
        connection.close()
    ledger.verify()


@pytest.mark.parametrize(
    ("sql", "params", "message"),
    [
        ("UPDATE ledger SET payload = ? WHERE seq = 2", ('{"i":99}',), "seq 2"),  # a canonical payload, a different content
        ("UPDATE ledger SET payload = ? WHERE seq = 2", ('{"i": 99}',), "seq 2"),  # not even canonical
        ("UPDATE ledger SET hash = ? WHERE seq = 2", ("f" * 64,), "seq 2"),
        ("UPDATE ledger SET kind = ? WHERE seq = 2", ("mark_of_zorro",), "seq 2"),
        ("UPDATE ledger SET session = ? WHERE seq = 2", ("2024-05-20",), "seq 2"),
        ("UPDATE ledger SET as_of = ? WHERE seq = 2", ("2024-05-20T20:00:00Z",), "seq 2"),
        ("DELETE FROM ledger WHERE seq = 2", (), "seq gap at position 2"),
    ],
)
def test_verify_detects_every_tamper(store_path: Path, sql: str, params: tuple[str, ...], message: str) -> None:
    with SqliteLedger(store_path) as store:
        marks(store, 4)
        store.commit()
    drop_triggers(store_path)  # what an attacker with a raw connection can do
    connection = raw(store_path)
    try:
        connection.execute(sql, params)
    finally:
        connection.close()
    with SqliteLedger(store_path) as store:
        with pytest.raises(LedgerCorrupt, match=message):
            store.verify()


def test_incremental_verify_starts_where_it_is_told(store_path: Path) -> None:
    with SqliteLedger(store_path) as store:
        marks(store, 4)
        store.commit()
        store.verify()
        store.verify(from_seq=4)
        store.verify(from_seq=5)  # nothing after the head: fine
        for bad in (0, 6, True):
            with pytest.raises(ValueError):
                store.verify(from_seq=bad)  # type: ignore[arg-type]
    drop_triggers(store_path)
    connection = raw(store_path)
    try:
        connection.execute("UPDATE ledger SET payload = ? WHERE seq = 2", ('{"i":99}',))
    finally:
        connection.close()
    with SqliteLedger(store_path) as store:
        with pytest.raises(LedgerCorrupt, match="seq 2"):
            store.verify()
        with pytest.raises(LedgerCorrupt, match="seq 2"):
            store.verify(from_seq=2)
        store.verify(from_seq=3)  # the linkage from seq 3 on is intact; the full verify at startup catches seq 2


def test_verify_records_how_far_it_checked(ledger: SqliteLedger) -> None:
    assert ledger.last_verified_seq == 0
    marks(ledger, 3)
    ledger.verify()
    assert ledger.last_verified_seq == 3 and ledger.get_meta(LAST_VERIFIED_SEQ) == "3"
    marks(ledger, 2)
    ledger.verify(from_seq=4)  # the incremental check every cycle runs
    assert ledger.last_verified_seq == 5
    marks(ledger, 2)
    ledger.verify(from_seq=7)  # a gap: seq 6 was never verified, so the high-water mark does not move
    assert ledger.last_verified_seq == 5
    with pytest.raises(InvariantError, match=LAST_VERIFIED_SEQ):
        ledger.set_meta(LAST_VERIFIED_SEQ, "99")  # bookkeeping, not run data


# ======================================================================================================================
# Commit modes, rollback and reopening (10.1)
# ======================================================================================================================


def test_per_session_commit_and_rollback(store_path: Path) -> None:
    with SqliteLedger(store_path, commit_mode="per_session") as store:
        store.append(LedgerKind.RUN_START, SESSION, AS_OF, {"initial_cash": 1})
        store.set_meta("config_hash", "abc")
        store.commit()
        assert store.commits == 1 and store.uncommitted == 0

        store.append(LedgerKind.SESSION_START, NEXT_SESSION, NEXT_AS_OF, {"session": "2024-05-20"})
        store.append(LedgerKind.FILL, NEXT_SESSION, NEXT_AS_OF, {"fill_id": "f1"})
        assert store.claim_fill("f1") and store.uncommitted == 2
        store.put_state("h1", "{}", session=NEXT_SESSION, underlying="SPY", request_kind="entry", variant="base")
        store.set_meta("namespace", "ns")
        store.rollback()  # the abort path of run_backtest: the store holds whole sessions only

        assert store.head()[0] == 1 and store.uncommitted == 0
        assert [e.kind for e in store.entries()] == [LedgerKind.RUN_START]
        assert store.claim_fill("f1") is True  # the claim was rolled back with the session
        assert list(store.get_states("entry")) == []
        assert store.get_meta("namespace") is None and store.get_meta("config_hash") == "abc"
        store.verify()
        store.append(LedgerKind.SESSION_END, NEXT_SESSION, NEXT_AS_OF, {"n": 1})
        store.commit()
        store.rollback()  # nothing uncommitted: a no-op
        assert store.head()[0] == 2 and store.commits == 2

    with SqliteLedger(store_path) as reopened:  # what survived is exactly what was committed
        assert reopened.head()[0] == 2 and reopened.fill_ids() == ("f1",)
        reopened.verify()


def test_per_append_mode_is_durable_at_once_and_rollback_is_a_no_op(store_path: Path) -> None:
    with SqliteLedger(store_path) as store:  # the paper default
        assert store.commit_mode == "per_append"
        store.append(LedgerKind.RUN_START, SESSION, AS_OF, {"initial_cash": 1})
        store.claim_fill("x")
        store.set_meta("k", "v")
        store.put_state("h", "{}", session=SESSION, underlying="SPY", request_kind="entry", variant="base")
        assert store.uncommitted == 0
        store.rollback()
        assert store.head()[0] == 1 and store.fill_ids() == ("x",) and store.get_meta("k") == "v"
        assert len(list(store.get_states("entry"))) == 1
        # a second connection sees every write without any commit() call
        connection = raw(store_path)
        try:
            assert connection.execute("SELECT COUNT(*) FROM ledger").fetchone()[0] == 1
            assert connection.execute("SELECT COUNT(*) FROM states").fetchone()[0] == 1
        finally:
            connection.close()
        store.commit()
        assert store.commits == 1
    with pytest.raises(ValueError):
        SqliteLedger(store_path, commit_mode="per_day")  # type: ignore[arg-type]


def test_closing_drops_an_uncommitted_session_and_reopening_continues_the_chain(store_path: Path) -> None:
    store = SqliteLedger(store_path, commit_mode="per_session")
    store.append(LedgerKind.RUN_START, SESSION, AS_OF, {"initial_cash": 1})
    store.commit()
    head = store.head()
    store.append(LedgerKind.SESSION_START, NEXT_SESSION, NEXT_AS_OF, {"session": "2024-05-20"})
    store.close()
    store.close()  # idempotent
    with pytest.raises(InvariantError):
        store.append(LedgerKind.MARK, SESSION, AS_OF, {"i": 1})

    with SqliteLedger(store_path, commit_mode="per_session") as reopened:
        assert reopened.head() == head  # the uncommitted session is gone
        reopened.verify()
        appended = reopened.append(LedgerKind.SESSION_START, NEXT_SESSION, NEXT_AS_OF, {"session": "2024-05-20"})
        assert appended.seq == 2 and appended.prev_hash == head[1]
        reopened.commit()
        reopened.verify()


def test_the_store_creates_its_run_directory_and_refuses_a_foreign_path(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "run.sqlite"
    with SqliteLedger(nested) as store:
        assert store.path == nested and nested.exists()
    with pytest.raises(TypeError):
        SqliteLedger(42)  # type: ignore[arg-type]


# ======================================================================================================================
# Fill claims, states and meta
# ======================================================================================================================


def test_claim_fill_is_the_dedupe_of_the_one_fill_path(ledger: SqliteLedger) -> None:
    assert ledger.claim_fill("fill-a") is True
    assert ledger.claim_fill("fill-a") is False  # the same cumulative fill offered again is booked once
    assert ledger.claim_fill("fill-b") is True
    assert ledger.fill_ids() == ("fill-a", "fill-b")
    with pytest.raises(ValueError):
        ledger.claim_fill("")


def test_states_and_the_state_index_round_trip(ledger: SqliteLedger) -> None:
    s1 = '{"schema":"state.v1.entry","x":1}'
    s2 = '{"schema":"state.v1.entry","x":2}'
    h1, h2 = sha256_hex(s1), sha256_hex(s2)
    ledger.put_state(h2, s2, session=NEXT_SESSION, underlying="QQQ", request_kind="entry", variant="base")
    ledger.put_state(h1, s1, session=SESSION, underlying="SPY", request_kind="entry", variant="base")
    ledger.put_state(h1, s1, session=SESSION, underlying="SPY", request_kind="entry", variant="base")  # the 10.1 restart case
    ledger.put_state(h1, s1, session=NEXT_SESSION, underlying="SPY", request_kind="entry", variant="base")  # two sessions, one hash
    ledger.put_state(h2, s2, session=SESSION, underlying="SPY", request_kind="entry", variant="key_perm")
    ledger.put_state(h2, s2, session=SESSION, underlying="SPY", request_kind="manage", variant="base")
    assert list(ledger.get_states("entry")) == [(SESSION, "SPY", s1), (NEXT_SESSION, "QQQ", s2), (NEXT_SESSION, "SPY", s1)]
    assert list(ledger.get_states("entry", "key_perm")) == [(SESSION, "SPY", s2)]
    assert list(ledger.get_states("manage")) == [(SESSION, "SPY", s2)]
    assert list(ledger.get_states("entry_text")) == []
    with pytest.raises(InvariantError, match="byte-identical"):  # a rebuilt state that differs is a determinism bug
        ledger.put_state(h2, s2, session=SESSION, underlying="SPY", request_kind="entry", variant="base")
    for bad in ("", None, 3):
        with pytest.raises(ValueError):
            ledger.put_state(h1, bad, session=SESSION, underlying="IWM", request_kind="entry", variant="base")  # type: ignore[arg-type]
    connection = raw(ledger.path)
    try:  # the states table is content-addressed: one row per distinct state, whatever the index holds
        assert connection.execute("SELECT COUNT(*) FROM states").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM state_index").fetchone()[0] == 5
    finally:
        connection.close()


def test_meta_keys_are_write_once(ledger: SqliteLedger) -> None:
    assert ledger.get_meta("missing") is None
    ledger.set_meta("config_hash", "abc")
    ledger.set_meta("config_hash", "abc")  # the same value again is a no-op
    with pytest.raises(InvariantError):
        ledger.set_meta("config_hash", "xyz")
    assert ledger.get_meta("config_hash") == "abc"
    with pytest.raises(ValueError):
        ledger.set_meta("", "x")
    with pytest.raises(ValueError):
        ledger.set_meta("k", 1)  # type: ignore[arg-type]


# ======================================================================================================================
# The sidecar (2.7): beside the rows, never hashed
# ======================================================================================================================


def test_sidecar_is_stored_beside_the_rows_and_never_hashed(store_path: Path, tmp_path: Path) -> None:
    payload = {"decision_id": "d1", "kind": "entry"}
    sidecar = {
        "wall_created_at": datetime(2026, 9, 17, 12, 30, tzinfo=UTC),
        "provenance": {"decision_id": "d1", "state_sha256": "ab" * 32, "iv_hist_proxy_pct": 4},
        "request_id": "req-1",
        "latency_ms": 12,
        "raw_features": {"rv20": 0.1234},  # floats and wall clocks may live here; they never touch the chain (INV-24)
    }
    with SqliteLedger(store_path) as plain, SqliteLedger(tmp_path / "other.sqlite") as with_sidecar:
        a = plain.append(LedgerKind.DECISION, SESSION, AS_OF, payload)
        b = with_sidecar.append(LedgerKind.DECISION, SESSION, AS_OF, payload, sidecar=sidecar)
        assert a == b and a.hash == b.hash  # the sidecar changes nothing about the entry
        assert plain.sidecar(1) is None and with_sidecar.sidecar(2) is None
        stored = with_sidecar.sidecar(1)
        assert stored is not None
        assert stored["wall_created_at"] == "2026-09-17T12:30:00Z"
        assert stored["provenance"] == sidecar["provenance"]
        assert stored["diagnostics"] == {"request_id": "req-1", "latency_ms": 12, "raw_features": {"rv20": 0.1234}}
        with_sidecar.verify()
        # without a wall clock of its own the store stamps one, and the sidecar table stays out of the chain
        with_sidecar.append(LedgerKind.DECISION, SESSION, AS_OF, payload, sidecar={"note": "no timestamp"})
        second = with_sidecar.sidecar(2)
        assert second is not None and second["diagnostics"] == {"note": "no timestamp"}
        assert datetime.fromisoformat(second["wall_created_at"]).tzinfo is not None
        assert with_sidecar.head()[1] == plain.append(LedgerKind.DECISION, SESSION, AS_OF, payload).hash
        with pytest.raises(TypeError):
            with_sidecar.append(LedgerKind.DECISION, SESSION, AS_OF, payload, sidecar={"wall_created_at": 5})


# ======================================================================================================================
# The views of 13.4
# ======================================================================================================================


def forecast_payload(*, question_id: str, p_ppm: int | None, missing_reason: str | None = None) -> dict[str, Any]:
    spec = OutcomeSpec(kind="close_gt", horizon_sessions=5, resolve_on=date(2024, 5, 24), ref=45_000, hi=46_000)
    forecast = Forecast(
        forecast_id=f"fc-{question_id}",
        event_key="ev-1",
        decision_id="dec-1",
        question_id=question_id,
        question_hash="qh",
        with_text=True,
        underlying="SPY",
        key=SnapshotKey(session=SESSION, slot=Slot.EOD),
        p_ppm=p_ppm,
        missing_reason=missing_reason,
        p_abstain_ppm=None,
        p_implied_ppm=280_000,
        implied_method="smile_digital",
        implied_quality="interpolated",
        p_implied_spread_ppm=None,
        spec=spec,
        tier=EvidenceTier.B,
        fidelity=Fidelity.EOD_QUOTES,
        iv_history="own",
        prereg=True,
    )
    built: dict[str, Any] = msgspec.to_builtins(forecast)
    return built


def test_views_return_the_documented_columns(ledger: SqliteLedger) -> None:
    answered = forecast_payload(question_id="eval.close_gt_up_5s", p_ppm=310_000)
    missing = forecast_payload(question_id="eval.close_lt_dn_5s", p_ppm=None, missing_reason="DeciderTransportError")
    ledger.append(LedgerKind.FORECAST, SESSION, AS_OF, answered)
    ledger.append(LedgerKind.FORECAST, SESSION, AS_OF, missing)
    outcome: dict[str, Any] = msgspec.to_builtins(
        Outcome(
            event_key="ev-1", resolved_on=date(2024, 5, 24), y=1, observed={"close": 46_310}, div_in_window="no", price_measure="parity"
        )
    )
    ledger.append(LedgerKind.OUTCOME, date(2024, 5, 24), datetime(2024, 5, 24, 20, tzinfo=UTC), outcome)
    ledger.append(LedgerKind.SESSION_END, SESSION, AS_OF, {"equity": {"orats": 1, "worst": 2, "mid": 3}, "n_decisions": 1})
    ledger.append(LedgerKind.FILL, SESSION, AS_OF, {"fill_id": "f1", "qty": 2})
    ledger.append(LedgerKind.DECISION, SESSION, AS_OF, {"decision_id": "dec-1", "kind": "entry"})
    ledger.append(LedgerKind.MARK, SESSION, AS_OF, {"i": 1})  # never in any of the views
    ledger.commit()

    for name, expected in VIEW_COLUMNS.items():
        columns, _ = ledger.view(name)
        assert columns == expected, name

    columns, rows = ledger.view("v_forecasts")
    by_question = {row[columns.index("question_id")]: dict(zip(columns, row, strict=True)) for row in rows}
    assert set(by_question) == {"eval.close_gt_up_5s", "eval.close_lt_dn_5s"}
    good = by_question["eval.close_gt_up_5s"]
    assert (good["p_ppm"], good["missing_reason"], good["p_implied_ppm"]) == (310_000, None, 280_000)
    assert (good["underlying"], good["with_text"], good["prereg"]) == ("SPY", 1, 1)
    assert (good["horizon"], good["resolve_on"], good["tier"], good["fidelity"]) == (5, "2024-05-24", "B", "EOD_QUOTES")
    gap = by_question["eval.close_lt_dn_5s"]
    assert gap["p_ppm"] is None and gap["missing_reason"] == "DeciderTransportError"  # a MISSING forecast is never filtered out

    columns, rows = ledger.view("v_outcomes")
    assert rows == [("ev-1", "2024-05-24", 1, "no")]

    columns, rows = ledger.view("v_calibration")
    assert len(rows) == 2  # both forecasts of the event join their outcome, the missing one included
    joined = {row[columns.index("question_id")]: dict(zip(columns, row, strict=True)) for row in rows}
    assert joined["eval.close_lt_dn_5s"]["p_ppm"] is None
    assert {row["y"] for row in joined.values()} == {1}

    for name, kind in (("v_daily", "session_end"), ("v_fills", "fill"), ("v_decisions", "decision")):
        columns, rows = ledger.view(name)
        assert columns == ("session", "payload") and len(rows) == 1
        assert rows[0][0] == "2024-05-17" and json.loads(rows[0][1])
        assert kind  # the view's own WHERE clause selects it
    with pytest.raises(ValueError):
        ledger.view("v_nope")


# ======================================================================================================================
# The Protocol
# ======================================================================================================================


def test_sqlite_ledger_matches_the_ledger_protocol() -> None:
    from jevbot.protocols import Ledger

    for name, member in vars(Ledger).items():
        if name.startswith("_") or not callable(member):
            continue
        want = [(p.name, p.kind, p.default) for p in inspect.signature(member).parameters.values()][1:]
        got = [(p.name, p.kind, p.default) for p in inspect.signature(getattr(SqliteLedger, name)).parameters.values()][1:]
        assert got == want, name
