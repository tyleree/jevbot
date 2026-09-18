"""`jev/probe.py`: the Step 0 suites end to end on the mock transport with `file:` states (DESIGN.md 6.8, 2.8; D10, G3).

Step 0 is a hard gate (the registry refuses `tune` / `final` Jev trials without matching records, and paper news stays off
until the `text` record exists), so what is tested here is the whole path: every suite runs against the real SDK over the
offline transport, the verdict carries the measurements 6.8 names, and the record lands in
`probes/step0/records/<suite>-<key12>.json` keyed by (model, SDK version, both entry question-set hashes).
"""

import json
from pathlib import Path
from typing import Any

import msgspec
import pytest
from typer.testing import CliRunner

from jevbot import canon, questions as questions_module, vocab
from jevbot.cli import jev_cmds
from jevbot.cli.main import GlobalOptions
from jevbot.config import Config, JevSpendConfig
from jevbot.errors import ConfigError, SpendLimitError
from jevbot.jev import probe as probe_module
from jevbot.jev.cache import SqliteDecisionCache
from jevbot.jev.spend import SCOPE_BATCH, SCOPE_PAPER, SpendGuard
from jevbot.types import ProbeRecord, RequestKind
from tests.fixtures.jev_transport import DEFAULT_MODEL, Fault, make_jev_transport, sample_state

TRENDS = ("up", "down", "flat", "mixed")


@pytest.fixture
def states_file(tmp_path: Path) -> Path:
    """Four `file:` states that differ in their regime-relevant codes (so stratification has something to do)."""
    path = tmp_path / "states.jsonl"
    lines = []
    for trend in TRENDS:
        state = sample_state(RequestKind.ENTRY)
        state["underlying"]["trend"]["direction"] = f"{trend}: price evidence"
        lines.append(json.dumps(state, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(
    suite: str,
    data_dir: Path,
    states_file: Path,
    *,
    n_states: int = 2,
    repeats: int = 3,
    max_tokens: int = 10_000_000,
    faults: tuple[Fault, ...] = (),
    known_models: frozenset[str] | None = None,
    cfg: Config | None = None,
    spend: SpendGuard | None = None,
    cache: SqliteDecisionCache | None = None,
) -> ProbeRecord:
    return probe_module.run_suite(
        suite,
        cfg or Config(),
        states=probe_module.file_state_source(states_file),
        api_key="dummy",
        out_dir=data_dir / "probes" / "step0" / f"20260917T120000Z-{suite}",
        max_tokens=max_tokens,
        transport=make_jev_transport(faults=faults, known_models=known_models),
        data_dir=data_dir,
        n_states=n_states,
        repeats=repeats,
        spend=spend,
        cache=cache,
    )


# ======================================================================================================================
# The five suites
# ======================================================================================================================


def test_meta_records_both_model_ids_the_bogus_error_and_the_chars_per_token(data_dir: Path, states_file: Path) -> None:
    record = run("meta", data_dir, states_file, known_models=frozenset({DEFAULT_MODEL, "jev-latest"}))
    verdict = record.verdict
    assert verdict["pinned_model_requested"] == DEFAULT_MODEL and verdict["pinned_model_reported"] == DEFAULT_MODEL
    assert verdict["pinned_matches"] is True and verdict["latest_is_pinned"] is True
    assert verdict["bogus_model_error"] == "TypeSafeNotFoundError", "an unknown model id must be recorded, not raised"
    assert verdict["bogus_model_status"] == 404
    assert verdict["chars_per_token_milli"] > 0 and verdict["request_chars"] > 1000
    assert verdict["requests"] == 3 and verdict["truncated"] is False


def test_determinism_measures_the_spread_and_the_flip_rates(data_dir: Path, states_file: Path) -> None:
    record = run("determinism", data_dir, states_file, n_states=3, repeats=4)
    verdict = record.verdict
    assert verdict["states"] == 3 and verdict["repeats"] == 4
    assert verdict["requests"] == 12
    # the mock answers byte-identically, so the measured spread is exactly zero and the verdict is `deterministic`
    assert verdict["deterministic"] is True
    assert verdict["max_noul_std_ppm"] == 0 and verdict["max_noul_range_ppm"] == 0
    assert verdict["choice_flip_rate_ppm"] == 0 and verdict["gate_flip_rate_ppm"] == 0
    assert set(verdict["noul_range_ppm"]) == {qid for qid in questions_module.ENTRY_V1 if vocab.QUESTION_TYPES[qid] == "noul"}


def test_batch_compares_the_full_batch_with_singles_and_halves(data_dir: Path, states_file: Path) -> None:
    record = run("batch", data_dir, states_file, n_states=1)
    verdict = record.verdict
    assert verdict["states"] == 1
    assert verdict["requests"] == 1 + len(questions_module.ENTRY_V1) + 2, "full batch, every question alone, two halves"
    assert set(verdict["per_question_max_ppm"]) == set(questions_module.ENTRY_V1)
    assert verdict["max_abs_diff_ppm"] == 0  # MockJev answers each question independently of the batch


def test_order_measures_every_variant_including_the_irrelevant_field(data_dir: Path, states_file: Path) -> None:
    record = run("order", data_dir, states_file, n_states=2)
    verdict = record.verdict
    assert set(verdict["variants"]) == {"opt_perm", "key_perm", "bucket_only", "irrelevant_field"}
    assert verdict["requests"] == 2 * 5  # base + four variants per state
    for name, measured in verdict["variants"].items():
        assert measured["trials"] == 2
        assert measured["max_abs_diff_ppm"] == 0, f"{name}: MockJev is variant-invariant by construction (6.8)"
        assert measured["gate_disagreement_rate_ppm"] == 0


def test_text_sends_empty_benign_and_hostile_news_with_the_sanitiser_off(data_dir: Path, states_file: Path) -> None:
    record = run("text", data_dir, states_file, n_states=1)
    verdict = record.verdict
    assert verdict["sanitiser"] == "off" and verdict["states"] == 1
    assert verdict["requests"] == 5  # empty, benign, hostile, plus the two trading controls
    assert verdict["empty_news_veto_max_ppm"] == 50_000, "the text vetoes read 0.05 on empty news (MockJev cannot read text)"
    assert verdict["hostile_shift_max_ppm"] == 0 and verdict["benign_shift_max_ppm"] == 0
    assert verdict["trading_questions_read_text"] is False and verdict["trading_shift_max_ppm"] == 0
    # the hostile corpus really is instruction-shaped, and it really reached the request
    lines = [json.loads(line) for line in (Path(record.run_dir) / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    hostile = next(line for line in lines if line["label"].endswith("-hostile"))
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" in json.dumps(hostile["state"])
    assert hostile["state"]["schema"] == "state.v1.entry_text"


# ======================================================================================================================
# Records, files and keys
# ======================================================================================================================


def test_the_record_is_keyed_by_model_sdk_and_both_question_set_hashes(data_dir: Path, states_file: Path) -> None:
    record = run("order", data_dir, states_file, n_states=1)
    assert record.model == DEFAULT_MODEL and record.sdk_version == "0.6.0"
    assert record.entry_qset_hash == questions_module.QUESTION_SET_HASHES["entry.v1"]
    assert record.entry_text_qset_hash == questions_module.QUESTION_SET_HASHES["entry_text.v1"]
    key12 = canon.sha256_hex(
        canon.dumps_sorted(
            {
                "model": record.model,
                "sdk_version": record.sdk_version,
                "entry_qset_hash": record.entry_qset_hash,
                "entry_text_qset_hash": record.entry_text_qset_hash,
            }
        )
    )[:12]
    path = data_dir / "probes" / "step0" / "records" / f"order-{key12}.json"
    assert path.is_file() and probe_module.record_path(data_dir, record) == path
    stored = msgspec.json.decode(path.read_bytes(), type=ProbeRecord)
    assert stored.verdict == record.verdict and stored.run_dir == record.run_dir
    assert stored.recorded_at.tzinfo is not None
    # a record never satisfies another wording: change one question and the key changes
    assert probe_module.probe_key12(record.model, record.sdk_version, "0" * 64, record.entry_text_qset_hash) != key12
    assert probe_module.probe_key12(record.model, "0.7.0", record.entry_qset_hash, record.entry_text_qset_hash) != key12


def test_config_probe_status_reads_the_records_we_write(data_dir: Path, states_file: Path) -> None:
    from jevbot import config as config_module

    cfg = Config()
    before = config_module.probe_status(
        data_dir,
        model=cfg.jev.model,
        sdk_version=cfg.jev.sdk_version,
        entry_qset_hash=questions_module.QUESTION_SET_HASHES["entry.v1"],
        entry_text_qset_hash=questions_module.QUESTION_SET_HASHES["entry_text.v1"],
    )
    assert not before.order
    run("order", data_dir, states_file, n_states=1)
    after = config_module.probe_status(
        data_dir,
        model=cfg.jev.model,
        sdk_version=cfg.jev.sdk_version,
        entry_qset_hash=questions_module.QUESTION_SET_HASHES["entry.v1"],
        entry_text_qset_hash=questions_module.QUESTION_SET_HASHES["entry_text.v1"],
    )
    assert after.order is True and after.text is False
    # a different model id never counts
    other = config_module.probe_status(
        data_dir,
        model="jev-1.14.0",
        sdk_version=cfg.jev.sdk_version,
        entry_qset_hash=questions_module.QUESTION_SET_HASHES["entry.v1"],
        entry_text_qset_hash=questions_module.QUESTION_SET_HASHES["entry_text.v1"],
    )
    assert not other.order


def test_every_suite_writes_its_four_output_files(data_dir: Path, states_file: Path) -> None:
    record = run("determinism", data_dir, states_file, n_states=1, repeats=2)
    out = Path(record.run_dir)
    assert sorted(p.name for p in out.iterdir()) == ["probe.sqlite", "requests.jsonl", "summary.json", "summary.md"]
    lines = [json.loads(line) for line in (out / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and {line["repeat"] for line in lines} == {0, 1}
    assert all(line["suite"] == "determinism" and line["model"] == DEFAULT_MODEL for line in lines)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == record.verdict and summary["suite"] == "determinism"
    assert "# Step 0 suite `determinism`" in (out / "summary.md").read_text(encoding="utf-8")


def test_repeats_are_stored_in_the_probe_table_and_never_in_the_decision_cache(data_dir: Path, states_file: Path) -> None:
    import sqlite3

    cache = SqliteDecisionCache(data_dir / "cache" / "decisions.sqlite")
    try:
        record = run("determinism", data_dir, states_file, n_states=1, repeats=3, cache=cache)
        assert cache.stats()["answers"] == 0, "6.8: probe repeats never enter the decision cache"
        namespace = record.verdict["namespace"]
        assert cache.is_diagnostic(namespace) is True, "the probe namespace can never back a trading run (risk check 2)"
        assert namespace.startswith("exp001-step0:")
    finally:
        cache.close()
    db = sqlite3.connect(Path(record.run_dir) / "probe.sqlite")
    try:
        rows = db.execute("SELECT DISTINCT repeat_index FROM probe_answers ORDER BY repeat_index").fetchall()
        assert [row[0] for row in rows] == [0, 1, 2], "the probe table is keyed by repeat index"
        answered = db.execute("SELECT COUNT(*) FROM probe_answers").fetchone()[0]
        assert answered == 3 * len(questions_module.ENTRY_V1)
    finally:
        db.close()


# ======================================================================================================================
# Budget, spend and refusals
# ======================================================================================================================


def test_the_token_budget_truncates_a_suite_instead_of_half_reporting_it(data_dir: Path, states_file: Path) -> None:
    record = run("determinism", data_dir, states_file, n_states=4, repeats=20, max_tokens=20_000)
    verdict = record.verdict
    assert verdict["truncated"] is True
    assert verdict["input_tokens"] <= 20_000 and verdict["requests"] >= 2
    assert verdict["states"] >= 1, "a truncated suite still reports what it measured"


def test_the_suites_run_under_the_batch_spend_scope(data_dir: Path, states_file: Path) -> None:
    spend = SpendGuard(data_dir / "state" / "spend.sqlite", scope=SCOPE_BATCH, cfg=JevSpendConfig())
    try:
        record = run("order", data_dir, states_file, n_states=1, spend=spend)
        assert spend.reservations() == record.verdict["requests"], "one reservation per HTTP attempt (INV-17)"
        assert spend.totals()[1] > 0
    finally:
        spend.close()
    paper = SpendGuard(data_dir / "state" / "spend.sqlite", scope=SCOPE_PAPER, cfg=JevSpendConfig())
    try:
        assert paper.totals()[1] == 0, "a probe run never touches the paper scope's counters"
    finally:
        paper.close()


def test_a_paper_scope_guard_is_refused(data_dir: Path, states_file: Path) -> None:
    spend = SpendGuard(data_dir / "state" / "spend.sqlite", scope=SCOPE_PAPER, cfg=JevSpendConfig())
    try:
        with pytest.raises(Exception, match="batch spend scope"):
            run("order", data_dir, states_file, n_states=1, spend=spend)
    finally:
        spend.close()


def test_a_spend_stop_propagates_out_of_a_suite(data_dir: Path, states_file: Path) -> None:
    tiny = JevSpendConfig(max_input_tokens_per_run=10, max_input_tokens_per_day=10, paper_max_input_tokens_per_day=10)
    spend = SpendGuard(data_dir / "state" / "spend.sqlite", scope=SCOPE_BATCH, cfg=tiny)
    try:
        with pytest.raises(SpendLimitError):
            run("order", data_dir, states_file, n_states=1, spend=spend)
    finally:
        spend.close()


def test_unknown_suites_and_empty_state_sources_are_refused(data_dir: Path, states_file: Path, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown Step 0 suite"):
        run("guessing", data_dir, states_file)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="holds no states"):
        run("order", data_dir, empty)
    missing = tmp_path / "nope.jsonl"
    with pytest.raises(ConfigError, match="does not exist"):
        run("order", data_dir, missing)
    broken = tmp_path / "broken.jsonl"
    broken.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="is not JSON"):
        run("order", data_dir, broken)


def test_the_file_state_source_checks_the_states_it_yields(tmp_path: Path) -> None:
    path = tmp_path / "states.jsonl"
    path.write_text(json.dumps({"schema": "state.v1.entry", "bad": 1.5}) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="float"):
        probe_module.file_state_source(path)(1)
    good = tmp_path / "good.jsonl"
    good.write_text("\n\n" + json.dumps(sample_state()) + "\n", encoding="utf-8")  # blank lines are skipped
    assert len(probe_module.file_state_source(good)(5)) == 1


# ======================================================================================================================
# The `jev` sub-app
# ======================================================================================================================


def _invoke(args: list[str], data_dir: Path, *, as_json: bool = False) -> Any:
    options = GlobalOptions(overrides=(f"paths.data_dir={data_dir}",), json=as_json)
    return CliRunner().invoke(jev_cmds.app, args, obj=options)


def test_jev_questions_prints_the_hashes(data_dir: Path) -> None:
    result = _invoke(["questions", "--set", "entry.v1", "--hashes"], data_dir, as_json=True)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["entry.v1"]["question_set_hash"] == questions_module.QUESTION_SET_HASHES["entry.v1"]
    assert payload["entry.v1"]["questions"]["regime.market"] == questions_module.QUESTION_HASHES["regime.market"]
    plain = _invoke(["questions", "--set", "manage.v1"], data_dir)
    assert plain.exit_code == 0 and "pos.thesis_invalidated" in plain.output and "instructions" in plain.output
    unknown = _invoke(["questions", "--set", "entry.v9"], data_dir)
    assert unknown.exit_code != 0


def test_jev_probe_step0_writes_records_through_the_cli(data_dir: Path, states_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe_module, "run_suite", _fake_run_suite)
    result = _invoke(["probe-step0", "--suite", "order", "--states-from", f"file:{states_file}", "--states", "1"], data_dir)
    assert result.exit_code == 0, result.output
    assert "order-" in result.output


def _fake_run_suite(suite: str, cfg: Config, **kwargs: Any) -> ProbeRecord:
    """The CLI wiring is tested without spending a probe run; `run_suite` itself is covered above."""
    from datetime import UTC, datetime

    record = ProbeRecord(
        suite=suite,
        model=cfg.jev.model,
        sdk_version=cfg.jev.sdk_version,
        entry_qset_hash=questions_module.QUESTION_SET_HASHES["entry.v1"],
        entry_text_qset_hash=questions_module.QUESTION_SET_HASHES["entry_text.v1"],
        verdict={"states": 1},
        run_dir=str(kwargs["out_dir"]),
        recorded_at=datetime.now(UTC),
    )
    probe_module.write_record(kwargs["data_dir"], record)
    return record


def test_jev_probe_step0_refuses_a_big_budget_without_yes_spend(data_dir: Path, states_file: Path) -> None:
    result = _invoke(["probe-step0", "--states-from", f"file:{states_file}", "--max-tokens", "9000000"], data_dir)
    assert result.exit_code != 0 and "yes-spend" in str(result.exception or result.output)


def test_jev_probe_step0_refuses_an_unknown_states_spec(data_dir: Path) -> None:
    result = _invoke(["probe-step0", "--states-from", "guess"], data_dir)
    assert result.exit_code != 0 and "--states-from" in str(result.exception or result.output)


def test_the_lazy_cross_wave_paths_print_not_built_yet(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # section 16: a cross-package convenience imports INSIDE the command and prints "not built yet" when the module is absent
    monkeypatch.setattr(jev_cmds, "_load_preview_request", lambda: None)
    result = _invoke(["show-request", "--underlying", "SPY", "--session", "2024-05-17"], data_dir)
    assert result.exit_code == 1 and "not built yet" in result.output and "jevbot.cycle" in result.output
    monkeypatch.setattr(jev_cmds, "_load_sample_entry_states", lambda: None)
    mirror = _invoke(["probe-step0", "--states-from", "mirror"], data_dir)
    assert mirror.exit_code == 1 and "not built yet" in mirror.output
    monkeypatch.setattr(jev_cmds, "_load_sqlite_ledger", lambda: None)
    from_run = _invoke(["probe-step0", "--states-from", "run:20260917T120000-abcdef12"], data_dir)
    assert from_run.exit_code == 1 and "not built yet" in from_run.output


def test_show_request_uses_the_injected_preview_builder(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.fixtures.jev_transport import make_request

    monkeypatch.setattr(jev_cmds, "_load_preview_request", lambda: lambda cfg, **kwargs: make_request(kwargs["kind"]))
    result = _invoke(["show-request", "--underlying", "SPY", "--session", "2024-05-17", "--kind", "entry_text"], data_dir)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["kind"] == "entry_text" and payload["question_set_id"] == "entry_text.v1"
    assert payload["question_set_hash"] == questions_module.QUESTION_SET_HASHES["entry_text.v1"]
    assert payload["state"]["schema"] == "state.v1.entry_text" and len(payload["questions"]) == 17


def test_show_request_refuses_unmasked_outside_a_diagnostic_run(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jev_cmds, "_load_preview_request", lambda: lambda cfg, **kwargs: None)
    options = GlobalOptions(overrides=(f"paths.data_dir={data_dir}", "state.unmasked=true"))
    result = CliRunner().invoke(
        jev_cmds.app, ["show-request", "--underlying", "SPY", "--session", "2024-05-17"], obj=options
    )
    assert result.exit_code != 0
    assert "unmasked" in str(result.exception or result.output)
