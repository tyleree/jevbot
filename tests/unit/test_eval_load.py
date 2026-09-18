"""Run store -> frames (DESIGN.md 12.2 / 12.3 / 13.4; test plan 15.1 `eval/*`).

The properties that matter for evidence, not for convenience:

* the hash chain is **re-verified from the persisted texts**, and a tampered payload is caught (INV-19);
* FORECAST joins OUTCOME **by `event_key`**, so the with-text and without-text forecasts of one event - and every
  baseline and reference asked about it - land on the same event;
* a **NULL `p_ppm` loads as MISSING**, with its reason, and is never dropped (12.1 `missing`);
* a **void outcome** (`y = null`) is kept and flagged rather than silently removed;
* `reference_history()` returns the six training columns and **no Jev output at all**, which is what makes the pooling
  exemption of 12.1 safe.
"""

import shutil
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from jevbot.errors import DataUnavailable, LedgerCorrupt
from jevbot.eval import load
from tests.fixtures.make_run_fixture import (
    RunFixture,
    RunFixtureSpec,
    make_reference_history_store,
    make_run_store,
)


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory: pytest.TempPathFactory) -> RunFixture:
    path = tmp_path_factory.mktemp("run_store") / "run.sqlite"
    return make_run_store(path)


@pytest.fixture
def store(fixture_run: RunFixture) -> load.RunStore:
    with load.RunStore(fixture_run.path) as handle:
        yield handle


# ======================================================================================================================
# The store handle
# ======================================================================================================================


def test_verify_recomputes_the_chain_and_reads_the_meta(store: load.RunStore, fixture_run: RunFixture) -> None:
    store.verify()  # the whole chain, from the persisted texts
    assert store.head() == (fixture_run.head_seq, fixture_run.head_hash)
    assert store.meta == fixture_run.meta
    assert store.run_id == fixture_run.meta.run_id
    assert store.sessions() == fixture_run.sessions


def test_verify_catches_an_edited_payload(fixture_run: RunFixture, tmp_path: Path) -> None:
    tampered = tmp_path / "tampered.sqlite"
    shutil.copy(fixture_run.path, tampered)
    # the store's own trigger forbids UPDATE; an attacker with the file would drop it first, so the test does too
    conn = sqlite3.connect(tampered)
    conn.execute("DROP TRIGGER ledger_no_update")
    original = conn.execute("SELECT payload FROM ledger WHERE kind = 'session_end' ORDER BY seq LIMIT 1").fetchone()[0]
    conn.execute(
        "UPDATE ledger SET payload = ? WHERE kind = 'session_end' AND payload = ?",
        (original.replace('"n_decisions":3', '"n_decisions":4'), original),
    )
    conn.commit()
    conn.close()
    with load.RunStore(tampered) as handle, pytest.raises(LedgerCorrupt, match="hash mismatch"):
        handle.verify()


def test_verify_catches_a_broken_chain(fixture_run: RunFixture, tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.sqlite"
    shutil.copy(fixture_run.path, truncated)
    conn = sqlite3.connect(truncated)
    conn.execute("DROP TRIGGER ledger_no_delete")
    conn.execute("DELETE FROM ledger WHERE seq = 5")
    conn.commit()
    conn.close()
    with load.RunStore(truncated) as handle, pytest.raises(LedgerCorrupt, match="gapless"):
        handle.verify()


def test_a_missing_store_is_data_unavailable(tmp_path: Path) -> None:
    with pytest.raises(DataUnavailable):
        load.RunStore(tmp_path / "absent.sqlite")


def test_the_store_is_opened_read_only(store: load.RunStore) -> None:
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        store.conn.execute("INSERT INTO meta (key, value) VALUES ('x', 'y')")


# ======================================================================================================================
# Forecasts, outcomes, calibration
# ======================================================================================================================


def test_calibration_joins_forecasts_to_outcomes_by_event_key(store: load.RunStore) -> None:
    forecasts = load.forecasts_frame(store)
    outcomes = load.outcomes_frame(store)
    frame = load.calibration_frame(store)

    resolved_keys = set(outcomes["event_key"])
    expected = int(forecasts["event_key"].isin(resolved_keys).sum())
    assert len(frame) == expected

    # every event carries BOTH forecast sets on ONE event_key: the join is by event, never by forecast id
    pairs = frame.groupby("event_key")["with_text"].nunique()
    assert set(pairs.unique()) == {2}
    assert frame["forecast_id"].nunique() == len(frame)


def test_null_p_ppm_loads_as_missing_and_is_never_dropped(store: load.RunStore, fixture_run: RunFixture) -> None:
    forecasts = load.forecasts_frame(store)
    missing = forecasts[forecasts["missing"]]
    assert len(missing) == fixture_run.n_missing_forecasts > 0
    assert missing["p"].isna().all()
    assert missing["p_ppm"].isna().all()
    assert (missing["missing_reason"] == "DeciderTransportError").all()
    # the row still carries a real spec, event key and implied probability (2.7, 12.1 `missing`)
    assert missing["event_key"].notna().all()
    assert missing["p_implied"].notna().all()
    assert missing["horizon"].notna().all()

    # and it survives the join into the scored frame, where the prereg imputes it with the reference
    scored = load.calibration_frame(store)
    assert int(scored["missing"].sum()) > 0


def test_void_outcomes_are_kept_and_flagged(store: load.RunStore, fixture_run: RunFixture) -> None:
    frame = load.calibration_frame(store)
    voids = frame[frame["void"]]
    assert len(voids) > 0
    assert voids["y"].isna().all()
    assert voids["resolved"].all()
    # one void OUTCOME entry covers both forecasts of its event
    assert voids["event_key"].nunique() == fixture_run.n_void_outcomes


def test_unresolved_forecasts_appear_only_when_asked_for(store: load.RunStore) -> None:
    resolved = load.calibration_frame(store)
    everything = load.calibration_frame(store, resolved_only=False)
    assert len(everything) > len(resolved)
    open_rows = everything[~everything["resolved"].astype(bool)]
    assert len(open_rows) == len(everything) - len(resolved)
    assert open_rows["y"].isna().all()
    assert not open_rows["void"].any()  # still open is not the same as void


def test_calibration_frame_over_several_stores_keeps_the_run_id(fixture_run: RunFixture, tmp_path: Path) -> None:
    second = make_run_store(tmp_path / "second.sqlite", RunFixtureSpec(run_id="20240115T120000-second", seed=7))
    frame = load.calibration_frame([fixture_run.path, second.path])
    assert set(frame["run_id"]) == {fixture_run.meta.run_id, second.meta.run_id}


def test_reference_history_has_six_training_columns_and_no_jev_output(tmp_path: Path) -> None:
    reference = make_reference_history_store(tmp_path / "reference.sqlite", n_sessions=40)
    history = load.reference_history(reference.path)

    assert tuple(history.columns) == load.REFERENCE_HISTORY_COLUMNS
    # the pooling exemption of 12.1 only holds because nothing here can be scored: no p, no tier, no measure
    for forbidden in ("p", "p_ppm", "tier", "fidelity", "price_measure", "with_text", "run_id"):
        assert forbidden not in history.columns
    assert history["p_implied"].between(0.0, 1.0).all()
    assert set(history["y"].unique()) <= {0, 1}
    assert set(history["horizon"].unique()) <= {1, 5}
    assert (history["resolved_on"] > history["session"]).all()


# ======================================================================================================================
# P&L frames
# ======================================================================================================================


def test_daily_frame_is_one_row_per_session_with_all_three_bands(store: load.RunStore, fixture_run: RunFixture) -> None:
    daily = load.daily_frame(store)
    assert len(daily) == len(fixture_run.sessions)
    assert list(daily["session"]) == list(fixture_run.sessions)
    for band in load.BANDS:
        assert daily[f"equity_{band}"].notna().all()
        assert daily[f"cash_{band}"].notna().all()
    assert {int(value) for value in daily["n_decisions"]} == {3}
    assert daily.iloc[-1][f"equity_{load.BANDS[0]}"] == fixture_run.final_equity["orats"]


def test_trades_are_assembled_from_fills_and_intents(store: load.RunStore) -> None:
    trades = load.trades_frame(store)
    closed = trades[trades["closed"]]
    assert len(closed) > 0
    # the fixture holds every position for exactly four SESSIONS - counted in sessions, not calendar days
    assert set(closed["holding_sessions"].unique()) == {4}
    assert set(closed["exit_reason"].unique()) <= {"profit_target", "force_exit_expiry"}

    # P&L identity, hand-computed: entry paid 105 / 110 / 100 per share, the winner closes at -115 / -112 / -120
    winner = closed[closed["pnl_mid"] == 2000.0].iloc[0]
    assert winner["open_net_orats"] == 105
    assert winner["close_net_orats"] == -115
    assert winner["pnl_orats"] == (115 - 105) * 100 * winner["qty"]
    assert winner["pnl_worst"] == (112 - 110) * 100 * winner["qty"]

    loser = closed[closed["pnl_mid"] < 0].iloc[0]
    assert loser["pnl_orats"] == (80 - 105) * 100 * loser["qty"]


def test_fills_count_zero_bid_sell_to_close_legs(store: load.RunStore) -> None:
    fills = load.fills_frame(store)
    closes = fills[fills["purpose"] == "close"]
    opens = fills[fills["purpose"] == "open"]
    # 10.4: the winning wing is sold at 0 on a CLOSE order; an OPEN order never sells into a zero bid
    assert (closes["zero_bid_close_legs"] == 1).all()
    assert (opens["zero_bid_close_legs"] == 0).all()
    assert fills["forced"].any()
    assert (fills["quality"] == "degraded").any()


def test_marks_and_position_marks(store: load.RunStore) -> None:
    marks = load.marks_frame(store)
    positions = load.position_marks_frame(store)
    assert len(marks) > 0
    assert marks["bp_utilisation_ppm"].notna().all()
    assert positions["position_id"].nunique() > 1
    assert set(positions.columns) == {"seq", "session", "position_id", "liq_value", "mid_value", "stale"}
    assert not positions["stale"].any()


def test_kill_affected_sessions_covers_kills_and_halts(store: load.RunStore, fixture_run: RunFixture) -> None:
    affected = load.kill_affected_sessions(store)
    assert affected == set(fixture_run.kill_sessions)
    assert affected  # the fixture contains one kill window on purpose


def test_decisions_and_verdicts_carry_the_funnel_material(store: load.RunStore) -> None:
    decisions = load.decisions_frame(store)
    verdicts = load.risk_verdicts_frame(store)
    assert (decisions["kind"] == "entry").all()
    assert set(decisions["action"].unique()) == {"enter", "no_trade"}

    # 7.9: a DECISION never carries a post-decision code, which is exactly why the funnel needs the join
    for reasons in decisions["reasons"]:
        for code in reasons:
            assert not code.startswith(("gate:", "candidate:", "risk:"))
    assert verdicts["decision_id"].isin(decisions["decision_id"]).all()
    # a no-intent verdict is the entry that died between the DECISION and an OrderIntent (2.6)
    no_intent = verdicts[verdicts["intent_id"].isna()]
    assert len(no_intent) > 0
    assert not no_intent["approved"].any()


def test_sidecar_carries_the_usage_numbers_the_ledger_must_not(store: load.RunStore) -> None:
    sidecar = load.sidecar_frame(store)
    assert len(sidecar) > 0
    assert sidecar["input_tokens"].notna().all()
    assert sidecar["latency_ms"].notna().all()
    assert sidecar["cache_hit"].notna().all()
    # the hashed ledger carries no request id, token count or wall clock (2.7, INV-24)
    decisions = load.decisions_frame(store)
    for payload_columns in ("request_id", "input_tokens", "latency_ms", "wall_created_at"):
        assert payload_columns not in decisions.columns


def test_risk_events_anomalies_and_kill_steps_load(store: load.RunStore) -> None:
    assert set(load.risk_events_frame(store)["type"]) == {"daily_loss_halt"}
    assert set(load.anomalies_frame(store)["type"]) == {"stale_mark", "assignment_sim"}
    assert list(load.kill_frame(store)["step"]) == ["tripped", "cancelled", "close_submitted", "flat_verified"]


def test_band_maps_accept_the_nested_and_the_flat_spelling() -> None:
    # 2.11 says "equity per band" without fixing the JSON shape; both spellings must load, neither as zero
    nested = load._band_map({"equity": {"orats": 11, "worst": 12, "mid": 13}}, "equity")
    flat = load._band_map({"equity_orats": 11, "equity_worst": 12, "equity_mid": 13}, "equity")
    assert nested == flat == {"orats": 11, "worst": 12, "mid": 13}
    assert load._band_map({}, "equity") == {}


def test_empty_frames_have_their_columns(tmp_path: Path) -> None:
    empty = make_run_store(
        tmp_path / "empty.sqlite",
        RunFixtureSpec(n_sessions=1, kill_session_index=None, missing_forecasts=False, void_outcomes=False),
    )
    with load.RunStore(empty.path) as handle:
        # a one-session run resolves nothing and trades nothing: every frame is empty but still typed
        assert load.calibration_frame(handle).empty
        columns = tuple(load.calibration_frame(handle).columns)
        assert columns[: len(load.CALIBRATION_COLUMNS)] == load.CALIBRATION_COLUMNS
        assert load.reference_history(handle).empty
        assert tuple(load.reference_history(handle).columns) == load.REFERENCE_HISTORY_COLUMNS
        # the one position opened on that session is still open: it is not a trade, and it has no P&L
        trades = load.trades_frame(handle)
        assert not trades["closed"].any()
        assert trades["pnl_mid"].isna().all()
        assert isinstance(load.kill_frame(handle), pd.DataFrame)
        assert load.kill_affected_sessions(handle) == set()


def test_sessions_are_ascending_and_distinct(store: load.RunStore) -> None:
    sessions = store.sessions()
    assert len(set(sessions)) == len(sessions)
    assert list(sessions) == sorted(sessions)
    assert all(isinstance(session, date) for session in sessions)
