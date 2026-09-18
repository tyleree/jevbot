"""Build a `run.sqlite` from raw SQL for the evaluation tests (DESIGN.md 13.4, section 16 WP07).

The report and every evaluation loader read run stores **only**, so WP07 needs no engine: this fixture writes the schema
of 13.4 with its own hands - the hash-chained `ledger`, the `sidecar`, `states` / `state_index`, `fill_ids` and `meta`
tables, the append-only triggers and the six views - and fills it with one small, fully consistent run.

What "consistent" means here is worth stating, because the eval tests lean on it:

* the chain is real: every entry is hashed with `canon.ledger_entry_hash`, the ONE formula of 2.7, so `RunStore.verify()`
  passes and a tampered copy fails;
* the book adds up: `cash[band]` moves by `-net[band] * 100 * qty` on every fill and `equity[band] = cash[band] -
  sum(liq_value * 100 * qty)` over the open positions (10.5, 10.8), so the equity series in SESSION_END really is the
  series the trades produced, in all three bands;
* the forecasts are a decision problem, not noise: each event has a latent probability, the option-implied probability
  carries a risk premium and the decider's probability is a shrunk version of the truth, so Brier, BSS and the
  reference machinery see a realistic signal;
* the awkward cases are present on purpose: MISSING forecasts (`p_ppm = null`), VOID outcomes (`y = null`), a kill
  window, a forced and a degraded fill, a zero-bid sell-to-close leg, rules-side abstentions, gate / candidate / risk
  rejections and emitted orders - the rows the funnel, the banner and the slice assertions are made of.

`make_reference_history_store()` writes the Jev-free MockJev run of `[prereg.reference_history]` (12.1): FORECAST entries
carrying `p_implied` and OUTCOME entries carrying `y`, nothing else - the training pairs both references warm-start from.
"""

import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import msgspec
import numpy as np

from jevbot import canon, ids
from jevbot.types import (
    EvidenceTier,
    Fidelity,
    FillRule,
    LedgerKind,
    OrderPurpose,
    OutcomeSpec,
    RunMeta,
    RunMode,
    SnapshotKey,
    Slot,
)

# ======================================================================================================================
# The schema of 13.4, written out in full
# ======================================================================================================================

RUN_STORE_SCHEMA = """
CREATE TABLE ledger (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, session TEXT NOT NULL, as_of TEXT NOT NULL,
  payload TEXT NOT NULL,
  prev_hash TEXT NOT NULL, hash TEXT NOT NULL UNIQUE);
CREATE INDEX ledger_kind_session ON ledger(kind, session);
CREATE TABLE sidecar  (seq INTEGER PRIMARY KEY REFERENCES ledger(seq), wall_created_at TEXT NOT NULL, provenance TEXT, diagnostics TEXT);
CREATE TABLE fill_ids (fill_id TEXT PRIMARY KEY, seq INTEGER);
CREATE TABLE states      (state_hash TEXT PRIMARY KEY, state_json TEXT NOT NULL);
CREATE TABLE state_index (session TEXT NOT NULL, underlying TEXT NOT NULL, request_kind TEXT NOT NULL, variant TEXT NOT NULL,
  state_hash TEXT NOT NULL REFERENCES states, PRIMARY KEY (session, underlying, request_kind, variant));
CREATE TABLE meta     (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TRIGGER ledger_no_update BEFORE UPDATE ON ledger BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
CREATE TRIGGER ledger_no_delete BEFORE DELETE ON ledger BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
CREATE VIEW v_forecasts AS SELECT seq, session, json_extract(payload,'$.forecast_id') AS forecast_id,
  json_extract(payload,'$.event_key') AS event_key,
  json_extract(payload,'$.decision_id') AS decision_id, json_extract(payload,'$.question_id') AS question_id,
  json_extract(payload,'$.with_text') AS with_text,
  json_extract(payload,'$.underlying') AS underlying, json_extract(payload,'$.p_ppm') AS p_ppm,
  json_extract(payload,'$.missing_reason') AS missing_reason, json_extract(payload,'$.p_abstain_ppm') AS p_abstain_ppm,
  json_extract(payload,'$.p_implied_ppm') AS p_implied_ppm,
  json_extract(payload,'$.implied_method') AS implied_method, json_extract(payload,'$.implied_quality') AS implied_quality,
  json_extract(payload,'$.spec.horizon_sessions') AS horizon, json_extract(payload,'$.spec.resolve_on') AS resolve_on,
  json_extract(payload,'$.tier') AS tier, json_extract(payload,'$.fidelity') AS fidelity,
  json_extract(payload,'$.iv_history') AS iv_history,
  json_extract(payload,'$.prereg') AS prereg FROM ledger WHERE kind = 'forecast';
CREATE VIEW v_outcomes AS SELECT json_extract(payload,'$.event_key') AS event_key,
  json_extract(payload,'$.resolved_on') AS resolved_on,
  json_extract(payload,'$.y') AS y, json_extract(payload,'$.div_in_window') AS div_in_window FROM ledger WHERE kind = 'outcome';
CREATE VIEW v_calibration AS SELECT f.*, o.y, o.resolved_on, o.div_in_window FROM v_forecasts f JOIN v_outcomes o USING (event_key);
CREATE VIEW v_daily AS SELECT session, payload FROM ledger WHERE kind = 'session_end';
CREATE VIEW v_fills AS SELECT session, payload FROM ledger WHERE kind = 'fill';
CREATE VIEW v_decisions AS SELECT session, payload FROM ledger WHERE kind = 'decision';
"""


# ======================================================================================================================
# Specification and result
# ======================================================================================================================

DEFAULT_UNDERLYINGS: tuple[str, ...] = ("SPY", "QQQ", "IWM")
ALIASES: dict[str, str] = {"SPY": "UNDERLYING_A", "QQQ": "UNDERLYING_B", "IWM": "UNDERLYING_C"}
# the pre-registered primary family (12.1) plus one non-primary evaluation question
PRIMARY_FAMILY: tuple[str, ...] = (
    "eval.down_1em_1s",
    "eval.up_1em_1s",
    "eval.inside_1em_1s",
    "eval.down_1em_5s",
    "eval.up_1em_5s",
    "eval.inside_1em_5s",
)
FIXTURE_QUESTIONS: tuple[str, ...] = (*PRIMARY_FAMILY, "eval.up_1s")

# Enough per-question structure that a reference or a calibration curve has something to find.
_QUESTION_KIND: dict[str, str] = {
    "eval.down_1em_1s": "close_lt",
    "eval.up_1em_1s": "close_gt",
    "eval.inside_1em_1s": "close_inside",
    "eval.down_1em_5s": "close_lt",
    "eval.up_1em_5s": "close_gt",
    "eval.inside_1em_5s": "close_inside",
    "eval.up_1s": "close_gt",
}
# latent "true" probability of each question, and the risk premium the option-implied probability carries on it
_TRUE_P: dict[str, float] = {
    "eval.down_1em_1s": 0.11,
    "eval.up_1em_1s": 0.12,
    "eval.inside_1em_1s": 0.77,
    "eval.down_1em_5s": 0.13,
    "eval.up_1em_5s": 0.14,
    "eval.inside_1em_5s": 0.73,
    "eval.up_1s": 0.52,
}
_PREMIUM: dict[str, float] = {
    "eval.down_1em_1s": 0.05,  # the risk-neutral probability of a down tail is inflated (variance risk premium)
    "eval.up_1em_1s": 0.01,
    "eval.inside_1em_1s": -0.06,
    "eval.down_1em_5s": 0.06,
    "eval.up_1em_5s": 0.01,
    "eval.inside_1em_5s": -0.07,
    "eval.up_1s": -0.02,
}


def question_horizon(question_id: str) -> int:
    """1 or 5 sessions, read off the question id (6.4)."""
    tail = question_id.rsplit("_", 1)[-1]
    return int(tail[:-1]) if tail.endswith("s") and tail[:-1].isdigit() else 1


class RunFixtureSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Everything the fixture varies. The defaults give a Tier B backtest of 40 sessions over three underlyings."""

    run_id: str = "20240115T120000-fixture"
    experiment: str = "exp_fixture"
    family: str = ""
    namespace: str = ""  # "" => ids.namespace(experiment, model, 0)
    model: str = "jev-fixture-1"
    model_release_date: date = date(2013, 1, 1)  # before the window => Tier B (12.1)
    mode: RunMode = RunMode.BACKTEST
    decider: str = "mock_jev"
    fidelity: Fidelity = Fidelity.EOD_QUOTES
    fill_rule: FillRule = FillRule.NEXT_SNAPSHOT
    spot_measure: str = "parity_forward"
    purpose: str = "validate"
    flags: tuple[str, ...] = ()
    start: date = date(2014, 1, 6)
    n_sessions: int = 40
    underlyings: tuple[str, ...] = DEFAULT_UNDERLYINGS
    questions: tuple[str, ...] = FIXTURE_QUESTIONS
    seed: int = 20260917
    news_resolved: bool = True
    news_reason: str = "auto_keys_present"
    initial_cash: int = 10_000_000  # cents ($100k)
    kill_session_index: int | None = 25
    missing_forecasts: bool = True
    void_outcomes: bool = True
    mid_only: bool = False  # True => profitable at MID only (the REJECTED_MID_ONLY case of 10.3)
    tier: EvidenceTier | None = None  # override the tier stamped on DECISION / FORECAST entries
    jev_skill: float = 0.55  # weight on the truth in the decider's forecast (0 = the implied probability, 1 = the truth)
    config_hash: str = "c" * 64
    state_config_hash: str = "s" * 64
    rules_hash: str = "r" * 64
    risk_config_hash: str = "k" * 64
    data_manifest_hash: str = "d" * 64
    git_commit: str | None = "0" * 40


class RunFixture(msgspec.Struct, frozen=True, kw_only=True):
    """What `make_run_store` wrote, so a test can assert against it without re-deriving anything."""

    path: Path
    meta: RunMeta
    sessions: tuple[date, ...]
    head_seq: int
    head_hash: str
    kill_sessions: tuple[date, ...]
    position_ids: tuple[str, ...]
    n_forecasts: int
    n_outcomes: int
    n_missing_forecasts: int
    n_void_outcomes: int
    initial_cash: int
    final_equity: dict[str, int]


# ======================================================================================================================
# The raw-SQL ledger writer
# ======================================================================================================================


class _Writer:
    """Appends ledger entries exactly as a conforming `Ledger` implementation would (2.7, 13.4)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.seq = 0
        self.prev = canon.GENESIS_HASH

    def append(
        self,
        kind: LedgerKind,
        session: date,
        as_of: datetime,
        payload: dict[str, Any],
        *,
        sidecar: dict[str, Any] | None = None,
    ) -> int:
        self.seq += 1
        session_text = canon.render_session(session)
        as_of_text = canon.render_as_of(as_of)
        payload_text = canon.dumps_sorted(payload)
        entry_hash = canon.ledger_entry_hash(self.prev, self.seq, kind.value, session_text, as_of_text, payload_text)
        self.conn.execute(
            "INSERT INTO ledger (seq, kind, session, as_of, payload, prev_hash, hash) VALUES (?,?,?,?,?,?,?)",
            (self.seq, kind.value, session_text, as_of_text, payload_text, self.prev, entry_hash),
        )
        if sidecar is not None:
            self.conn.execute(
                "INSERT INTO sidecar (seq, wall_created_at, provenance, diagnostics) VALUES (?,?,?,?)",
                (
                    self.seq,
                    as_of_text,
                    json.dumps(sidecar.get("provenance"), sort_keys=True) if sidecar.get("provenance") else None,
                    json.dumps(sidecar.get("diagnostics"), sort_keys=True) if sidecar.get("diagnostics") else None,
                ),
            )
        self.prev = entry_hash
        return self.seq


def _sessions(start: date, count: int) -> tuple[date, ...]:
    """`count` consecutive weekdays from `start` - a calendar is not needed to test a loader, only monotone sessions."""
    out: list[date] = []
    day = start
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return tuple(out)


def _close(session: date) -> datetime:
    """The session's `as_of`: a fixed instant after the exchange close, spelled in UTC."""
    return datetime(session.year, session.month, session.day, 21, 0, tzinfo=UTC)


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = FULL")
    conn.executescript(RUN_STORE_SCHEMA)
    return conn


def _put_meta(conn: sqlite3.Connection, meta: RunMeta, head_seq: int) -> None:
    rows = {
        "run_meta": msgspec.json.encode(meta).decode(),
        "config_hash": meta.config_hash,
        "state_config_hash": meta.state_config_hash,
        "data_manifest_hash": meta.data_manifest_hash,
        "namespace": meta.namespace,
        "family": meta.family,
        "last_verified_seq": str(head_seq),
    }
    conn.executemany("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", list(rows.items()))


def _put_state(conn: sqlite3.Connection, session: date, underlying: str, request_kind: str, state: dict[str, Any]) -> str:
    state_json = canon.dumps_ordered(state)
    state_hash = canon.sha256_hex(state_json)
    conn.execute("INSERT OR IGNORE INTO states (state_hash, state_json) VALUES (?,?)", (state_hash, state_json))
    conn.execute(
        "INSERT OR IGNORE INTO state_index (session, underlying, request_kind, variant, state_hash) VALUES (?,?,?,?,?)",
        (canon.render_session(session), underlying, request_kind, "base", state_hash),
    )
    return state_hash


# ======================================================================================================================
# The run store
# ======================================================================================================================


def make_run_store(path: Path | str, spec: RunFixtureSpec | None = None) -> RunFixture:
    """Write one complete `run.sqlite` and return what is in it."""
    spec = spec or RunFixtureSpec()
    path = Path(path)
    rng = np.random.default_rng(spec.seed)
    sessions = _sessions(spec.start, spec.n_sessions)
    namespace = spec.namespace or ids.namespace(spec.experiment, spec.model, 0)
    family = spec.family or spec.experiment
    tier = spec.tier or (EvidenceTier.C if sessions[0] < spec.model_release_date else EvidenceTier.B)
    meta = RunMeta(
        run_id=spec.run_id,
        trial_id=None,
        experiment=spec.experiment,
        family=family,
        namespace=namespace,
        mode=spec.mode,
        decider=spec.decider,
        model=spec.model,
        model_release_date=spec.model_release_date,
        fidelity=spec.fidelity,
        fill_rule=spec.fill_rule,
        spot_measure=spec.spot_measure,
        news_resolved=spec.news_resolved,
        news_reason=spec.news_reason,
        config_hash=spec.config_hash,
        state_config_hash=spec.state_config_hash,
        rules_hash=spec.rules_hash,
        risk_config_hash=spec.risk_config_hash,
        entry_qset_hash="e" * 64,
        entry_text_qset_hash="t" * 64,
        manage_qset_hash="m" * 64,
        manage_text_qset_hash="n" * 64,
        git_commit=spec.git_commit,
        data_manifest_hash=spec.data_manifest_hash,
        cache_manifest_hash=None,
        start=sessions[0],
        end=sessions[-1],
        purpose=spec.purpose,
        flags=spec.flags,
    )

    conn = _connect(path)
    writer = _Writer(conn)
    state = _RunState(spec=spec, sessions=sessions, tier=tier, namespace=namespace, rng=rng)

    writer.append(
        LedgerKind.RUN_START,
        sessions[0],
        _close(sessions[0]),
        {
            "mode": meta.mode.value,
            "namespace": namespace,
            "model": meta.model,
            "model_release_date": canon.render_session(meta.model_release_date),
            "decider": meta.decider,
            "fidelity": meta.fidelity.value,
            "fill_rule": meta.fill_rule.value,
            "spot_measure": meta.spot_measure,
            "news_resolved": meta.news_resolved,
            "news_reason": meta.news_reason,
            "config_hash": meta.config_hash,
            "state_config_hash": meta.state_config_hash,
            "rules_hash": meta.rules_hash,
            "risk_config_hash": meta.risk_config_hash,
            "entry_qset_hash": meta.entry_qset_hash,
            "entry_text_qset_hash": meta.entry_text_qset_hash,
            "manage_qset_hash": meta.manage_qset_hash,
            "manage_text_qset_hash": meta.manage_text_qset_hash,
            "data_manifest_hash": meta.data_manifest_hash,
            "git_commit": meta.git_commit,
            "purpose": meta.purpose,
            "flags": list(meta.flags),
            "initial_cash": spec.initial_cash,
        },
    )

    for index, session in enumerate(sessions):
        state.session_index = index
        _write_session(conn, writer, state, index, session)

    head_seq, head_hash = writer.seq, writer.prev
    _put_meta(conn, meta, head_seq)
    conn.commit()
    conn.close()

    return RunFixture(
        path=path,
        meta=meta,
        sessions=sessions,
        head_seq=head_seq,
        head_hash=head_hash,
        kill_sessions=tuple(state.kill_sessions),
        position_ids=tuple(state.closed_positions),
        n_forecasts=state.n_forecasts,
        n_outcomes=state.n_outcomes,
        n_missing_forecasts=state.n_missing,
        n_void_outcomes=state.n_void,
        initial_cash=spec.initial_cash,
        final_equity=dict(state.equity),
    )


class _OpenPosition:
    def __init__(self, position_id: str, underlying: str, kind: str, qty: int, open_index: int, open_net: dict[str, int]):
        self.position_id = position_id
        self.underlying = underlying
        self.kind = kind
        self.qty = qty
        self.open_index = open_index
        self.open_net = open_net
        self.liq_value = -open_net["mid"]


class _RunState:
    """The little book the fixture keeps so the equity series really is the series the fills produced (10.8)."""

    def __init__(self, *, spec: RunFixtureSpec, sessions: tuple[date, ...], tier: EvidenceTier, namespace: str, rng: Any):
        self.spec = spec
        self.sessions = sessions
        self.tier = tier
        self.namespace = namespace
        self.rng = rng
        self.cash: dict[str, int] = {band: spec.initial_cash for band in ("orats", "worst", "mid")}
        self.equity: dict[str, int] = dict(self.cash)
        self.open_positions: list[_OpenPosition] = []
        self.closed_positions: list[str] = []
        self.pending_outcomes: dict[int, list[dict[str, Any]]] = {}
        self.kill_sessions: list[date] = []
        self.session_index = 0
        self.n_forecasts = 0
        self.n_outcomes = 0
        self.n_missing = 0
        self.n_void = 0
        self.accrued_micro = 0

    def mark_equity(self) -> None:
        for band in ("orats", "worst", "mid"):
            open_value = sum(position.liq_value * 100 * position.qty for position in self.open_positions)
            self.equity[band] = self.cash[band] - open_value


def _write_session(conn: sqlite3.Connection, writer: _Writer, state: _RunState, index: int, session: date) -> None:
    spec = state.spec
    as_of = _close(session)
    killed = spec.kill_session_index is not None and index == spec.kill_session_index
    halted = killed or index == (spec.kill_session_index or -99) + 1

    writer.append(LedgerKind.SESSION_START, session, as_of, {"session": canon.render_session(session), "slot": Slot.EOD.value, "phase": "full"})

    n_decisions = 0
    n_forecasts_here = 0
    n_intents = 0

    for underlying_index, underlying in enumerate(spec.underlyings):
        decision_id = ids.decision_id(state.namespace, session, underlying, "entry", underlying)
        n_decisions += 1
        outcome = _decision_outcome(index, underlying_index, killed=killed, halted=halted)
        _write_decision(conn, writer, state, session, as_of, underlying, decision_id, outcome)
        n_forecasts_here += _write_forecasts(writer, state, session, as_of, underlying, decision_id)
        n_intents += _write_verdict_and_orders(writer, state, session, as_of, underlying, decision_id, outcome, index)

    _resolve_due_outcomes(writer, state, session, as_of, index)
    _close_due_positions(writer, state, session, as_of, index)

    if killed:
        state.kill_sessions.append(session)
        writer.append(
            LedgerKind.RISK_EVENT,
            session,
            as_of,
            {"type": "daily_loss_halt", "trigger": "daily_loss", "detail": "book loss 2.1% of day_start_equity", "counters": {"halts": 1}},
        )
        for step in ("tripped", "cancelled", "close_submitted", "flat_verified"):
            writer.append(
                LedgerKind.KILL,
                session,
                as_of,
                {"event_id": f"kill-{index}", "step": step, "trigger": "drawdown", "detail": "fixture kill drill"},
            )
    if index == 7:
        writer.append(LedgerKind.ANOMALY, session, as_of, {"type": "stale_mark", "detail": {"position_id": "fixture", "sessions": 1}})
    if index == 11:
        writer.append(LedgerKind.ANOMALY, session, as_of, {"type": "assignment_sim", "detail": {"underlying": spec.underlyings[0]}})

    _write_mark(writer, state, session, as_of)

    fee_cents = 12
    state.accrued_micro += 4000
    for band in ("orats", "worst", "mid"):
        state.cash[band] -= fee_cents
    writer.append(LedgerKind.FEE, session, as_of, {"fee_cents": fee_cents, "accrued_micro_before": state.accrued_micro})

    state.mark_equity()
    writer.append(
        LedgerKind.SESSION_END,
        session,
        as_of,
        {
            "session": canon.render_session(session),
            "equity": dict(state.equity),
            "cash": dict(state.cash),
            "positions_digest": canon.sha256_hex(",".join(sorted(p.position_id for p in state.open_positions)))[:16],
            "n_decisions": n_decisions,
            "n_forecasts": n_forecasts_here,
            "n_intents": n_intents,
            "invariant_no_expiry_risk": True,
        },
    )


def _decision_outcome(index: int, underlying_index: int, *, killed: bool, halted: bool) -> str:
    """Which funnel branch this decision takes - one of every kind appears in the fixture."""
    if killed and underlying_index == 0:
        return "gate_kill"
    if halted and underlying_index == 1:
        return "gate_halt"
    slot = (index * 3 + underlying_index) % 7
    return {
        0: "enter",
        1: "rules_veto",
        2: "rules_score",
        3: "candidate_reject",
        4: "risk_reject",
        5: "enter",
        6: "rules_crosscheck",
    }[slot]


def _write_decision(
    conn: sqlite3.Connection,
    writer: _Writer,
    state: _RunState,
    session: date,
    as_of: datetime,
    underlying: str,
    decision_id: str,
    outcome: str,
) -> None:
    spec = state.spec
    entered = outcome in ("enter", "candidate_reject", "risk_reject", "gate_kill", "gate_halt")
    reasons: tuple[str, ...] = ()
    if outcome == "rules_veto":
        reasons = ("veto:vol.explained_by_event:hard",)
    elif outcome == "rules_score":
        reasons = ("score:below_min",)
    elif outcome == "rules_crosscheck":
        reasons = ("crosscheck:trend_vs_direction",)

    state_payload = {
        "meta": {"schema": "state.v1.entry"},
        "regime": {"trend": "up", "iv_rank": "p60_p80"},
        "underlying": {"kind": "broad US large-cap equity index ETF"},
    }
    state_hash = _put_state(conn, session, underlying, "entry", state_payload)

    answers = {
        qid: 500_000 + 100_000 * ((int(canon.sha256_hex(f"{decision_id}|{qid}")[:6], 16) % 5) - 2)
        for qid in ("under.direction", "vol.stance")
    }
    requests = [
        {
            "request_kind": "entry",
            "variant": "base",
            "state_hash": state_hash,
            "question_set_id": "entry.v1",
            "question_set_hash": spec.config_hash,
            "question_hashes": {"under.direction": "q" * 16},
            "cache_keys": {"under.direction": "c" * 16},
            "model": spec.model,
            "source": "cache",
            "answers": answers,
            "error": None,
        },
        {
            "request_kind": "entry_text",
            "variant": "base",
            "state_hash": state_hash,
            "question_set_id": "entry_text.v1",
            "question_set_hash": spec.config_hash,
            "question_hashes": {"text.material_present": "q" * 16},
            "cache_keys": {"text.material_present": "c" * 16},
            "model": spec.model,
            "source": "cache",
            "answers": {"text.material_present": 300_000},
            "error": None,
        },
    ]
    writer.append(
        LedgerKind.DECISION,
        session,
        as_of,
        {
            "decision_id": decision_id,
            "kind": "entry",
            "subject_alias": ALIASES.get(underlying, "UNDERLYING_X"),
            "text": "on" if spec.news_resolved else "off",
            "requests": requests,
            "rules": {
                "underlying": underlying,
                "decision_id": decision_id,
                "action": "enter" if entered else "no_trade",
                "kind": "iron_condor" if entered else None,
                "score_core_ppm": 620_000 if entered else 310_000,
                "score_rank_ppm": 640_000 if entered else 300_000,
                "tier_ppm": 750_000 if entered else 0,
                "reasons": list(reasons),
                "features_ppm": {"trend": 600_000, "iv_rank": 700_000},
                "variant_agreement": {"opt_perm": True, "key_perm": outcome != "rules_score", "bucket_only": True},
            },
            "facts": {
                "trend_code": "up",
                "iv_rank_code": "p60_p80",
                "iv_rv_code": "rich",
                "dist_code": "near",
                "news_enabled": spec.news_resolved,
                "news_count": 3,
                "news_recent_count": 1,
                "spot": 45_000,
                "iv30_bp": 1830,
                "em_hold_tenths": 21,
                "events_in_window": 0,
                "thesis": "trend up, iv rich, no tracked event in the window",
            },
            "tier": state.tier.value,
        },
        sidecar={
            "provenance": {
                "decision_id": decision_id,
                "request_id": f"req-{writer.seq + 1}",
                "input_tokens": 3100 + (writer.seq % 7) * 25,
                "latency_ms": 700 + (writer.seq % 11) * 30,
                "evidence_tier": state.tier.value,
                "data_fidelity": spec.fidelity.value,
            },
            "diagnostics": {"cache_hit": writer.seq % 4 != 0},
        },
    )


def _write_forecasts(
    writer: _Writer, state: _RunState, session: date, as_of: datetime, underlying: str, decision_id: str
) -> int:
    spec = state.spec
    index = state.session_index
    key = SnapshotKey(session=session, slot=Slot.EOD)
    written = 0
    for question_id in spec.questions:
        horizon = question_horizon(question_id)
        resolve_index = index + horizon
        if resolve_index >= len(state.sessions):
            continue
        resolve_on = state.sessions[resolve_index]
        spec_obj = OutcomeSpec(
            kind=_QUESTION_KIND[question_id],
            horizon_sessions=horizon,
            resolve_on=resolve_on,
            ref=45_000,
            lo=44_100 if _QUESTION_KIND[question_id] in ("close_lt", "close_inside") else None,
            hi=45_900 if _QUESTION_KIND[question_id] in ("close_gt", "close_inside") else None,
        )
        event_key = ids.event_key(underlying, key, spec_obj)
        true_p = float(np.clip(_TRUE_P[question_id] + 0.03 * float(state.rng.standard_normal()), 0.02, 0.98))
        implied_p = float(np.clip(true_p + _PREMIUM[question_id], 0.01, 0.99))
        jev_p = float(np.clip(spec.jev_skill * true_p + (1.0 - spec.jev_skill) * implied_p, 0.01, 0.99))
        y = int(state.rng.random() < true_p)
        void = spec.void_outcomes and (index % 17 == 3) and question_id == spec.questions[0]
        missing = spec.missing_forecasts and (index % 13 == 5) and question_id == spec.questions[-1]

        spec_payload = {
            "kind": spec_obj.kind,
            "horizon_sessions": spec_obj.horizon_sessions,
            "resolve_on": canon.render_session(spec_obj.resolve_on),
            "ref": spec_obj.ref,
            "lo": spec_obj.lo,
            "hi": spec_obj.hi,
            "iv_var_ppm": spec_obj.iv_var_ppm,
        }
        for with_text in (True, False):
            forecast_id = ids.forecast_id(decision_id, question_id, with_text)
            p_ppm = None if missing else round((jev_p if with_text else jev_p * 0.98 + 0.01) * 1_000_000)
            writer.append(
                LedgerKind.FORECAST,
                session,
                as_of,
                {
                    "forecast_id": forecast_id,
                    "event_key": event_key,
                    "decision_id": decision_id,
                    "question_id": question_id,
                    "question_hash": "q" * 24,
                    "with_text": with_text,
                    "underlying": underlying,
                    "key": {"session": canon.render_session(session), "slot": Slot.EOD.value},
                    "p_ppm": p_ppm,
                    "missing_reason": "DeciderTransportError" if missing else None,
                    "p_abstain_ppm": None,
                    "p_implied_ppm": round(implied_p * 1_000_000),
                    "implied_method": "smile_digital",
                    "implied_quality": "interpolated",
                    "p_implied_spread_ppm": None,
                    "spec": spec_payload,
                    "tier": state.tier.value,
                    "fidelity": spec.fidelity.value,
                    "iv_history": "own",
                    "prereg": False,
                },
            )
            written += 1
            state.n_forecasts += 1
            if missing:
                state.n_missing += 1
        state.pending_outcomes.setdefault(resolve_index, []).append(
            {
                "event_key": event_key,
                "y": None if void else y,
                "observed": {"close": 45_310},
                "div_in_window": "no",
                "price_measure": spec.spot_measure,
            }
        )
    return written


def _resolve_due_outcomes(writer: _Writer, state: _RunState, session: date, as_of: datetime, index: int) -> None:
    for pending in state.pending_outcomes.pop(index, []):
        writer.append(
            LedgerKind.OUTCOME,
            session,
            as_of,
            {
                "event_key": pending["event_key"],
                "resolved_on": canon.render_session(session),
                "y": pending["y"],
                "observed": pending["observed"],
                "div_in_window": pending["div_in_window"],
                "price_measure": pending["price_measure"],
            },
        )
        state.n_outcomes += 1
        if pending["y"] is None:
            state.n_void += 1


def _write_verdict_and_orders(
    writer: _Writer,
    state: _RunState,
    session: date,
    as_of: datetime,
    underlying: str,
    decision_id: str,
    outcome: str,
    index: int,
) -> int:
    spec = state.spec
    reject_codes: tuple[str, ...]
    approved = False
    if outcome == "enter":
        reject_codes, approved = (), True
    elif outcome == "gate_kill":
        reject_codes = ("gate:kill_active",)
    elif outcome == "gate_halt":
        reject_codes = ("gate:halt_entries",)
    elif outcome == "candidate_reject":
        reject_codes = ("candidate:exceeds_risk_budget",)
    elif outcome == "risk_reject":
        reject_codes = ("risk:size_zero",)
    else:
        return 0  # the rules abstained: there is no post-DECISION verdict at all

    intent_id = ids.intent_id(state.namespace, session, decision_id, OrderPurpose.OPEN, 0) if approved else None
    candidate_summary = {
        "structure_id": f"{underlying}-cond-{index}",
        "legs": [{"occ": f"{underlying}240119C00450000", "side": "sell"}],
        "net": {"orats": 105, "worst": 110, "mid": 100},
        "max_loss_per_contract": 40_000,
        "budget_floor": 50_000,
        "rejects": [code.split(":", 1)[1] for code in reject_codes if code.startswith("candidate:")],
    }
    writer.append(
        LedgerKind.RISK_VERDICT,
        session,
        as_of,
        {
            "verdict_id": canon.sha256_hex(f"{decision_id}|{outcome}")[:24],
            "decision_id": decision_id,
            "intent_id": intent_id,
            "approved": approved,
            "qty_approved": 1 if approved else 0,
            "checks": [
                {"code": "max_loss_per_trade", "passed": True, "observed": 40_000, "limit": 100_000, "detail": ""},
                {"code": "buying_power", "passed": approved, "observed": 40_000, "limit": 900_000, "detail": ""},
            ],
            "reject_codes": list(reject_codes),
            "max_loss": 40_000 if approved else 0,
            "bp_required": 40_000 if approved else 0,
            "candidate": candidate_summary,
        },
    )
    if not approved or intent_id is None:
        return 0

    position_id = ids.position_id(state.namespace, session, candidate_summary["structure_id"])
    open_net = {"orats": 105, "worst": 110, "mid": 100}
    client_order_id = f"{intent_id}-0"
    writer.append(
        LedgerKind.ORDER_INTENT,
        session,
        as_of,
        {
            "intent_id": intent_id,
            "decision_id": decision_id,
            "position_id": position_id,
            "purpose": "open",
            "part": 0,
            "underlying": underlying,
            "legs": [{"occ": f"{underlying}240119C00450000", "side": "sell", "position_intent": "sell_to_open"}],
            "qty": 1,
            "limit_start": 104,
            "limit_natural": 110,
            "reason": "entry",
            "mandatory": False,
            "session": canon.render_session(session),
            "key": {"session": canon.render_session(session), "slot": Slot.EOD.value},
            "tier_ppm": 750_000,
            "structure": {"kind": "iron_condor", "structure_id": candidate_summary["structure_id"]},
            "entry_ctx": {
                "entry_thesis": "trend up, iv rich",
                "entry_codes": {"trend": "up", "iv_vs_realized": "rich", "iv_rank": "p60_p80"},
                "entry_spot": 45_000,
                "entry_iv30_bp": 1830,
                "entry_em_hold_tenths": 21,
                "open_mid_at_decision": 100,
            },
        },
    )
    writer.append(
        LedgerKind.ORDER_STATUS,
        session,
        as_of,
        {
            "client_order_id": client_order_id,
            "intent_id": intent_id,
            "attempt": 0,
            "status": "filled",
            "qty": 1,
            "filled_qty": 1,
            "limit": 104,
            "broker_order_id": None,
            "reject_code": None,
            "tag": None,
        },
    )
    fill_id = ids.fill_id(client_order_id, 1)
    writer.conn.execute("INSERT OR IGNORE INTO fill_ids (fill_id, seq) VALUES (?,?)", (fill_id, writer.seq + 1))
    writer.append(
        LedgerKind.FILL,
        session,
        as_of,
        _fill_payload(
            fill_id=fill_id,
            client_order_id=client_order_id,
            intent_id=intent_id,
            decision_id=decision_id,
            position_id=position_id,
            purpose="open",
            structure_id=candidate_summary["structure_id"],
            qty=1,
            session=session,
            as_of=as_of,
            net=open_net,
            legs=[{"occ": f"{underlying}240119C00450000", "side": "sell", "bid": 95, "ask": 115, "orats": 105, "worst": 110, "mid": 100}],
            forced=False,
            quality="ok",
        ),
    )
    for band, value in open_net.items():
        state.cash[band] -= value * 100 * 1
    state.open_positions.append(
        _OpenPosition(position_id, underlying, "iron_condor", 1, index, open_net)
    )
    return 1


def _close_due_positions(writer: _Writer, state: _RunState, session: date, as_of: datetime, index: int) -> None:
    """Close every position that has been held for the fixed holding window; the last ones stay open on purpose."""
    hold = 4
    still_open: list[_OpenPosition] = []
    for position in state.open_positions:
        if index - position.open_index < hold:
            # the mark path: one adverse session, then a recovery - so the maximum adverse excursion is not the exit
            held = index - position.open_index
            position.liq_value = -(85 if held == 1 else 100 + 6 * held)
            still_open.append(position)
            continue
        close_net = _close_net(state.spec.mid_only)
        forced = position.open_index % 9 == 0
        degraded = position.open_index % 11 == 0
        client_order_id = f"{position.position_id}-close-0"
        close_decision_id = ids.decision_id(state.namespace, session, position.underlying, "manage", position.position_id)
        intent_id = ids.intent_id(state.namespace, session, close_decision_id, OrderPurpose.CLOSE, 0)
        writer.append(
            LedgerKind.ORDER_INTENT,
            session,
            as_of,
            {
                "intent_id": intent_id,
                "decision_id": close_decision_id,
                "position_id": position.position_id,
                "purpose": "close",
                "part": 0,
                "underlying": position.underlying,
                "legs": [{"occ": f"{position.underlying}240119C00450000", "side": "buy", "position_intent": "buy_to_close"}],
                "qty": position.qty,
                "limit_start": close_net["orats"],
                "limit_natural": close_net["worst"],
                "reason": "force_exit_expiry" if forced else "profit_target",
                "mandatory": forced,
                "session": canon.render_session(session),
                "key": {"session": canon.render_session(session), "slot": Slot.EOD.value},
                "tier_ppm": 0,
                "structure": {"kind": position.kind, "structure_id": position.position_id},
                "entry_ctx": None,
            },
        )
        realised = {
            band: (-close_net[band] - position.open_net[band]) * 100 * position.qty for band in ("orats", "worst", "mid")
        }
        fill_id = ids.fill_id(client_order_id, position.qty)
        writer.conn.execute("INSERT OR IGNORE INTO fill_ids (fill_id, seq) VALUES (?,?)", (fill_id, writer.seq + 1))
        writer.append(
            LedgerKind.FILL,
            session,
            as_of,
            _fill_payload(
                fill_id=fill_id,
                client_order_id=client_order_id,
                intent_id=intent_id,
                decision_id=close_decision_id,
                position_id=position.position_id,
                purpose="close",
                structure_id=position.position_id,
                qty=position.qty,
                session=session,
                as_of=as_of,
                net=close_net,
                legs=[
                    # a zero-bid sell-to-close leg: the normal state of a winning wing (10.4), sold at 0 on all bands
                    {"occ": f"{position.underlying}240119C00470000", "side": "sell", "bid": 0, "ask": 4, "orats": 0, "worst": 0, "mid": 0},
                    {"occ": f"{position.underlying}240119C00450000", "side": "buy", "bid": 95, "ask": 115, "orats": 105, "worst": 110, "mid": 100},
                ],
                forced=forced,
                quality="degraded" if degraded else "ok",
                realised=realised,
            ),
        )
        for band, value in close_net.items():
            state.cash[band] -= value * 100 * position.qty
        state.closed_positions.append(position.position_id)
    state.open_positions = still_open


def _close_net(mid_only: bool) -> dict[str, int]:
    """Signed net of the closing fill (negative = credit received).

    The default run wins at every band; `mid_only` makes it win ONLY at mid - the configuration 10.3 labels
    `REJECTED_MID_ONLY`, which the report has to refuse to celebrate.
    """
    if mid_only:
        return {"orats": -100, "worst": -95, "mid": -120}
    return {"orats": -115, "worst": -112, "mid": -120}


def _fill_payload(
    *,
    fill_id: str,
    client_order_id: str,
    intent_id: str,
    decision_id: str,
    position_id: str,
    purpose: str,
    structure_id: str,
    qty: int,
    session: date,
    as_of: datetime,
    net: dict[str, int],
    legs: list[dict[str, Any]],
    forced: bool,
    quality: str,
    realised: dict[str, int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "fill_id": fill_id,
        "client_order_id": client_order_id,
        "intent_id": intent_id,
        "decision_id": decision_id,
        "position_id": position_id,
        "purpose": purpose,
        "structure_id": structure_id,
        "qty": qty,
        "key": {"session": canon.render_session(session), "slot": Slot.EOD.value},
        "ts": canon.render_as_of(as_of),
        "net": dict(net),
        "legs": legs,
        "fees_micro": 130_000,
        "forced": forced,
        "model_reject": [],
        "quality": quality,
        "source": "sim",
        "broker_order_id": None,
        "broker_net": None,
    }
    if realised is not None:
        payload["realised_pnl"] = dict(realised)
    return payload


def _write_mark(writer: _Writer, state: _RunState, session: date, as_of: datetime) -> None:
    state.mark_equity()
    open_max_loss = sum(40_000 * position.qty for position in state.open_positions)
    bp_used = sum(40_000 * position.qty for position in state.open_positions)
    equity_headline = max(state.equity["orats"], 1)
    writer.append(
        LedgerKind.MARK,
        session,
        as_of,
        {
            "equity": dict(state.equity),
            "cash": dict(state.cash),
            "open_max_loss": open_max_loss,
            "bp_used": bp_used,
            "bp_utilisation_ppm": round(bp_used / equity_headline * 1_000_000),
            "positions": {
                position.position_id: {"liq_value": position.liq_value, "mid_value": position.liq_value, "stale": False}
                for position in state.open_positions
            },
            "net_delta_milli": 120 * len(state.open_positions),
            "net_vega_milli": -45 * len(state.open_positions),
        },
    )


# ======================================================================================================================
# The reference-history store (12.1 [prereg.reference_history])
# ======================================================================================================================


def make_reference_history_store(
    path: Path | str,
    *,
    run_id: str = "20130102T000000-reference",
    experiment: str = "exp_fixture",
    model: str = "jev-fixture-1",
    model_release_date: date = date(2024, 1, 1),
    n_sessions: int = 300,
    start: date = date(2012, 1, 3),
    underlyings: tuple[str, ...] = DEFAULT_UNDERLYINGS,
    questions: tuple[str, ...] = PRIMARY_FAMILY,
    seed: int = 4242,
    data_manifest_hash: str = "d" * 64,
    spot_measure: str = "parity_forward",
) -> RunFixture:
    """The registered, Jev-free MockJev run over the mirror that BOTH references warm-start from (12.1, 12.3).

    Its FORECAST entries carry `p_implied`, its OUTCOME entries carry `y`, and those pairs - and only those - are the
    training data. Its tier (C) and price measure are a stated approximation of a *training* set, which is why the
    pooling guard exempts it for this purpose alone and why nothing here is ever scored.
    """
    path = Path(path)
    rng = np.random.default_rng(seed)
    sessions = _sessions(start, n_sessions)
    namespace = ids.namespace(experiment, model, 0)
    meta = RunMeta(
        run_id=run_id,
        trial_id=None,
        experiment=experiment,
        family=f"{experiment}#reference",
        namespace=namespace,
        mode=RunMode.BACKTEST,
        decider="mock_jev",
        model=model,
        model_release_date=model_release_date,
        fidelity=Fidelity.EOD_QUOTES,
        fill_rule=FillRule.NEXT_SNAPSHOT,
        spot_measure=spot_measure,
        news_resolved=False,
        news_reason="explicit_off",
        config_hash="c" * 64,
        state_config_hash="s" * 64,
        rules_hash="r" * 64,
        risk_config_hash="k" * 64,
        entry_qset_hash="e" * 64,
        entry_text_qset_hash="t" * 64,
        manage_qset_hash="m" * 64,
        manage_text_qset_hash="n" * 64,
        git_commit="0" * 40,
        data_manifest_hash=data_manifest_hash,
        cache_manifest_hash=None,
        start=sessions[0],
        end=sessions[-1],
        purpose="reference",
        flags=("reference_history",),
    )

    conn = _connect(path)
    writer = _Writer(conn)
    writer.append(
        LedgerKind.RUN_START,
        sessions[0],
        _close(sessions[0]),
        {
            "mode": meta.mode.value,
            "namespace": namespace,
            "model": meta.model,
            "model_release_date": canon.render_session(meta.model_release_date),
            "decider": meta.decider,
            "fidelity": meta.fidelity.value,
            "fill_rule": meta.fill_rule.value,
            "spot_measure": meta.spot_measure,
            "news_resolved": meta.news_resolved,
            "news_reason": meta.news_reason,
            "config_hash": meta.config_hash,
            "state_config_hash": meta.state_config_hash,
            "rules_hash": meta.rules_hash,
            "risk_config_hash": meta.risk_config_hash,
            "entry_qset_hash": meta.entry_qset_hash,
            "entry_text_qset_hash": meta.entry_text_qset_hash,
            "manage_qset_hash": meta.manage_qset_hash,
            "manage_text_qset_hash": meta.manage_text_qset_hash,
            "data_manifest_hash": meta.data_manifest_hash,
            "git_commit": meta.git_commit,
            "purpose": meta.purpose,
            "flags": list(meta.flags),
            "initial_cash": 10_000_000,
        },
    )

    pending: dict[int, list[dict[str, Any]]] = {}
    n_forecasts = 0
    n_outcomes = 0
    for index, session in enumerate(sessions):
        as_of = _close(session)
        for pending_row in pending.pop(index, []):
            writer.append(
                LedgerKind.OUTCOME,
                session,
                as_of,
                {
                    "event_key": pending_row["event_key"],
                    "resolved_on": canon.render_session(session),
                    "y": pending_row["y"],
                    "observed": {"close": 45_120},
                    "div_in_window": "no",
                    "price_measure": spot_measure,
                },
            )
            n_outcomes += 1
        for underlying in underlyings:
            decision_id = ids.decision_id(namespace, session, underlying, "entry", underlying)
            key = SnapshotKey(session=session, slot=Slot.EOD)
            for question_id in questions:
                horizon = question_horizon(question_id)
                if index + horizon >= len(sessions):
                    continue
                resolve_on = sessions[index + horizon]
                spec_obj = OutcomeSpec(
                    kind=_QUESTION_KIND[question_id],
                    horizon_sessions=horizon,
                    resolve_on=resolve_on,
                    ref=45_000,
                    lo=44_100 if _QUESTION_KIND[question_id] in ("close_lt", "close_inside") else None,
                    hi=45_900 if _QUESTION_KIND[question_id] in ("close_gt", "close_inside") else None,
                )
                event_key = ids.event_key(underlying, key, spec_obj)
                true_p = float(np.clip(_TRUE_P[question_id] + 0.03 * float(rng.standard_normal()), 0.02, 0.98))
                implied_p = float(np.clip(true_p + _PREMIUM[question_id], 0.01, 0.99))
                y = int(rng.random() < true_p)
                writer.append(
                    LedgerKind.FORECAST,
                    session,
                    as_of,
                    {
                        "forecast_id": ids.forecast_id(decision_id, question_id, False),
                        "event_key": event_key,
                        "decision_id": decision_id,
                        "question_id": question_id,
                        "question_hash": "q" * 24,
                        "with_text": False,
                        "underlying": underlying,
                        "key": {"session": canon.render_session(session), "slot": Slot.EOD.value},
                        "p_ppm": round(implied_p * 1_000_000),  # MockJev's constants: never used as a reference
                        "missing_reason": None,
                        "p_abstain_ppm": None,
                        "p_implied_ppm": round(implied_p * 1_000_000),
                        "implied_method": "smile_digital",
                        "implied_quality": "interpolated",
                        "p_implied_spread_ppm": None,
                        "spec": {
                            "kind": spec_obj.kind,
                            "horizon_sessions": spec_obj.horizon_sessions,
                            "resolve_on": canon.render_session(spec_obj.resolve_on),
                            "ref": spec_obj.ref,
                            "lo": spec_obj.lo,
                            "hi": spec_obj.hi,
                            "iv_var_ppm": None,
                        },
                        "tier": EvidenceTier.C.value,
                        "fidelity": Fidelity.EOD_QUOTES.value,
                        "iv_history": "own",
                        "prereg": False,
                    },
                )
                n_forecasts += 1
                pending.setdefault(index + horizon, []).append({"event_key": event_key, "y": y})
        writer.append(
            LedgerKind.SESSION_END,
            session,
            as_of,
            {
                "session": canon.render_session(session),
                "equity": {"orats": 10_000_000, "worst": 10_000_000, "mid": 10_000_000},
                "cash": {"orats": 10_000_000, "worst": 10_000_000, "mid": 10_000_000},
                "positions_digest": "",
                "n_decisions": len(underlyings),
                "n_forecasts": len(underlyings) * len(questions),
                "n_intents": 0,
                "invariant_no_expiry_risk": True,
            },
        )

    head_seq, head_hash = writer.seq, writer.prev
    _put_meta(conn, meta, head_seq)
    conn.commit()
    conn.close()
    return RunFixture(
        path=path,
        meta=meta,
        sessions=sessions,
        head_seq=head_seq,
        head_hash=head_hash,
        kill_sessions=(),
        position_ids=(),
        n_forecasts=n_forecasts,
        n_outcomes=n_outcomes,
        n_missing_forecasts=0,
        n_void_outcomes=0,
        initial_cash=10_000_000,
        final_equity={"orats": 10_000_000, "worst": 10_000_000, "mid": 10_000_000},
    )
