"""The trial registry: holdout guard, Step 0 gate, scan-facts gate, families and the DSR scope (DESIGN.md 12.6, 13.5).

The registry is what makes `N` honest, so the tests are mostly refusals:

* `purpose = "tune"` is refused on a window that reaches the model's release date (`HoldoutViolation`);
* `purpose = "tune"` / `"final"` with a Jev decider is refused without the `determinism`, `order` and `batch` probe
  records of **exactly** this (model, SDK version, question-set hashes) key - a record taken on another model id does
  not count (B3.5);
* `purpose = "final"` is refused without sizeable `data scan-candidates` facts for the current
  `candidate_config_hash` (section 8, 11.2 B3);
* `N` counts completed, failed **and** abandoned selection trials of the same family and namespace, and excludes every
  `#baseline` / `#shadow` / `#reference` / `#diagnostic` / `#ablation` run, while `Var(SR_n)` is taken over the
  completed subset only.
"""

import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import msgspec
import pytest

from jevbot.config import Config
from jevbot.errors import ConfigError, EvalError, HoldoutViolation, PreregError
from jevbot.eval import registry as reg
from jevbot.types import EvidenceTier, Fidelity, FillRule, ProbeRecord, RunMeta, RunMode

RELEASE = date(2026, 9, 15)
SDK = "0.6.0"
ENTRY_HASH = "e" * 64
ENTRY_TEXT_HASH = "t" * 64
CANDIDATE_HASH = "cand0001"


def run_meta(**overrides: object) -> RunMeta:
    """A `RunMeta` with sane defaults; every test varies only what it is about."""
    base: dict[str, object] = {
        "run_id": "20260101T000000-aaaa",
        "trial_id": None,
        "experiment": "exp001",
        "family": "exp001",
        "namespace": "exp001:jev-1:g0",
        "mode": RunMode.BACKTEST,
        "decider": "mock_jev",
        "model": "jev-1",
        "model_release_date": RELEASE,
        "fidelity": Fidelity.EOD_QUOTES,
        "fill_rule": FillRule.NEXT_SNAPSHOT,
        "spot_measure": "parity_forward",
        "news_resolved": True,
        "news_reason": "auto_keys_present",
        "config_hash": "c" * 64,
        "state_config_hash": "s" * 64,
        "rules_hash": "r" * 64,
        "risk_config_hash": "k" * 64,
        "entry_qset_hash": ENTRY_HASH,
        "entry_text_qset_hash": ENTRY_TEXT_HASH,
        "manage_qset_hash": "m" * 64,
        "manage_text_qset_hash": "n" * 64,
        "git_commit": "0" * 40,
        "data_manifest_hash": "d" * 64,
        "cache_manifest_hash": None,
        "start": date(2012, 1, 3),
        "end": date(2027, 1, 4),
        "purpose": "validate",
        "flags": (),
    }
    base.update(overrides)
    return RunMeta(**base)  # type: ignore[arg-type]


@pytest.fixture
def registry(tmp_path: Path) -> reg.Registry:
    with reg.Registry.open(tmp_path) as handle:
        yield handle


def write_probe_record(data_dir: Path, suite: str, **overrides: object) -> Path:
    record = ProbeRecord(
        suite=suite,
        model=str(overrides.get("model", "jev-1")),
        sdk_version=str(overrides.get("sdk_version", SDK)),
        entry_qset_hash=str(overrides.get("entry_qset_hash", ENTRY_HASH)),
        entry_text_qset_hash=str(overrides.get("entry_text_qset_hash", ENTRY_TEXT_HASH)),
        verdict=dict(overrides.get("verdict", {"ok": True})),  # type: ignore[arg-type]
        run_dir=str(overrides.get("run_dir", "probes/step0/20260101T000000")),
        recorded_at=overrides.get("recorded_at", datetime(2026, 1, 1, 12, 0, tzinfo=UTC)),  # type: ignore[arg-type]
    )
    directory = data_dir / "probes" / "step0" / "records"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{suite}-{record.model}-{record.sdk_version}.json"
    path.write_bytes(msgspec.json.encode(record))
    return path


def write_scan_facts(data_dir: Path, rows: list[dict[str, object]], *, candidate_config_hash: str = CANDIDATE_HASH) -> Path:
    directory = data_dir / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "scan_candidates.json"
    path.write_text(
        json.dumps(
            {
                candidate_config_hash: {
                    "candidate_config_hash": candidate_config_hash,
                    "created_at": "2026-01-01T00:00:00Z",
                    "initial_equity_usd": 100000,
                    "rows": rows,
                }
            }
        )
    )
    return path


def scan_row(underlying: str, kind: str, year: int, unsizeable: float) -> dict[str, object]:
    return {
        "underlying": underlying,
        "kind": kind,
        "year": year,
        "n_candidates": 200,
        "exceeds_risk_budget_rate": 0.02,
        "unsizeable_rate_by_tier": {"500000": unsizeable, "750000": 0.0, "1000000": 0.0},
        # a field this gate does not read, to prove unknown keys are tolerated
        "reject_codes": {"width": 3},
    }


# ======================================================================================================================
# Schema
# ======================================================================================================================


def test_the_registry_is_append_only(registry: reg.Registry, tmp_path: Path) -> None:
    meta = run_meta()
    registry.register_trial(meta, git_commit="abc", git_dirty=False)
    registry.record_result(meta.run_id, n_days=1, ledger_head="head")
    registry.put_prereg(prereg_id="prereg.v1", sha256="a" * 64, git_commit="c" * 40, body_toml="[prereg]\n")
    write_probe_record(tmp_path, "determinism")
    registry.sync_step0(tmp_path)
    for table in ("trials", "trial_results", "prereg", "step0_records"):
        assert registry.connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"] == 1
        with pytest.raises(sqlite3.IntegrityError):
            registry.connection.execute(f"DELETE FROM {table}")


def test_registering_stores_the_identity_columns(registry: reg.Registry) -> None:
    meta = run_meta(flags=("news_off",))
    trial_id = registry.register_trial(meta, git_commit="cafebabe", git_dirty=True, notes="first")
    row = registry.trial(meta.run_id)
    assert row is not None
    assert row.trial_id == trial_id
    assert row.family == "exp001"
    assert row.state_config_hash == meta.state_config_hash  # the sub-hash the report and the sweep gate read
    assert row.qset_hashes["entry"] == ENTRY_HASH
    assert row.git_commit == "cafebabe"
    assert row.git_dirty is True
    assert row.flags == ("news_off",)
    assert row.status == "registered"
    assert row.notes == "first"
    assert row.start == meta.start
    assert row.end == meta.end
    assert row.touches_holdout is True  # the window reaches past the release date


def test_an_empty_family_falls_back_to_the_experiment_label(registry: reg.Registry) -> None:
    registry.register_trial(run_meta(family=""), git_commit=None, git_dirty=False)
    row = registry.trial("20260101T000000-aaaa")
    assert row is not None and row.family == "exp001"


def test_a_run_is_registered_once_and_resumed_afterwards(registry: reg.Registry) -> None:
    meta = run_meta()
    trial_id = registry.register_trial(meta, git_commit=None, git_dirty=False)
    with pytest.raises(EvalError, match="already registered"):
        registry.register_trial(meta, git_commit=None, git_dirty=False)

    registry.set_status(meta.run_id, "running")
    registry.set_status(meta.run_id, "failed", notes="failed:spend_limit")
    resumed = registry.resume(meta.run_id)
    assert resumed.trial_id == trial_id  # a resumed run is ONE trial, counted once in N
    assert resumed.status == "running"

    registry.set_status(meta.run_id, "completed")
    with pytest.raises(EvalError, match="only a failed or running trial"):
        registry.resume(meta.run_id)
    with pytest.raises(EvalError, match="refusing the status move"):
        registry.set_status(meta.run_id, "running")


def test_results_need_a_registered_trial(registry: reg.Registry) -> None:
    with pytest.raises(EvalError, match="not registered"):
        registry.record_result("never-registered", n_days=10, ledger_head="h")
    meta = run_meta()
    registry.register_trial(meta, git_commit=None, git_dirty=False)
    registry.record_result(meta.run_id, n_days=10, ledger_head="head", sharpe_orats=1.5, skew=-0.2, kurt=3.4)
    result = registry.result(meta.run_id)
    assert result is not None
    assert result.n_days == 10
    assert result.sharpe_orats == pytest.approx(1.5)
    assert result.kurt == pytest.approx(3.4)  # the PEARSON kurtosis (Normal = 3), per 12.6


# ======================================================================================================================
# The holdout guard (12.1)
# ======================================================================================================================


def test_tune_is_refused_on_a_window_that_reaches_the_release_date(registry: reg.Registry) -> None:
    with pytest.raises(HoldoutViolation, match="tune"):
        registry.register_trial(run_meta(purpose="tune"), git_commit=None, git_dirty=False)


def test_tune_is_allowed_strictly_before_the_release_date(registry: reg.Registry) -> None:
    meta = run_meta(purpose="tune", end=RELEASE.replace(day=14))
    registry.register_trial(meta, git_commit=None, git_dirty=False)
    row = registry.trial(meta.run_id)
    assert row is not None and row.touches_holdout is False


def test_an_open_ended_window_touches_the_holdout(registry: reg.Registry) -> None:
    with pytest.raises(HoldoutViolation, match="open end"):
        registry.register_trial(run_meta(purpose="tune", end=None), git_commit=None, git_dirty=False)
    # other purposes may of course run there; they are simply flagged
    registry.register_trial(run_meta(purpose="paper", end=None, run_id="paper-exp001"), git_commit=None, git_dirty=False)
    row = registry.trial("paper-exp001")
    assert row is not None and row.touches_holdout is True


def test_holdout_looks_are_counted_per_namespace(registry: reg.Registry) -> None:
    registry.add_holdout_look(namespace="ns-a", tiers=[EvidenceTier.A, EvidenceTier.B], n_sessions=120, prereg_look=True)
    registry.add_holdout_look(namespace="ns-a", tiers=["B"], n_sessions=40)
    registry.add_holdout_look(namespace="ns-b", tiers=["A"])
    assert registry.count_holdout_looks() == 3
    assert registry.count_holdout_looks("ns-a") == 2
    assert registry.count_holdout_looks("ns-a", prereg_only=True) == 1
    looks = registry.holdout_looks("ns-a")
    assert looks[0].tiers == ("A", "B")
    assert looks[0].prereg_look is True


# ======================================================================================================================
# The Step 0 gate (G3, B3.5)
# ======================================================================================================================


def test_step0_gate_refuses_a_jev_tune_or_final_without_records(registry: reg.Registry) -> None:
    for purpose in ("tune", "final"):
        with pytest.raises(ConfigError) as excinfo:
            registry.register_trial(
                run_meta(purpose=purpose, decider="live_jev", end=RELEASE.replace(day=14), run_id=f"r-{purpose}"),
                git_commit=None,
                git_dirty=False,
                gates=reg.TrialGates(sdk_version=SDK),
            )
        message = str(excinfo.value)
        assert "determinism" in message and "order" in message and "batch" in message


def test_step0_gate_passes_once_the_three_suites_are_synced(registry: reg.Registry, tmp_path: Path) -> None:
    for suite in ("determinism", "order", "batch"):
        write_probe_record(tmp_path, suite)
    assert registry.sync_step0(tmp_path) == 3
    assert registry.sync_step0(tmp_path) == 0  # idempotent: the same records again change nothing

    registry.register_trial(
        run_meta(purpose="tune", decider="live_jev", end=RELEASE.replace(day=14)),
        git_commit=None,
        git_dirty=False,
        gates=reg.TrialGates(sdk_version=SDK),
    )


def test_a_record_for_another_model_or_sdk_does_not_count(registry: reg.Registry, tmp_path: Path) -> None:
    for suite in ("determinism", "order", "batch"):
        write_probe_record(tmp_path, suite, model="jev-2")  # the right suites, the wrong model
    registry.sync_step0(tmp_path)
    assert (
        registry.step0_suites(model="jev-1", sdk_version=SDK, entry_qset_hash=ENTRY_HASH, entry_text_qset_hash=ENTRY_TEXT_HASH)
        == frozenset()
    )
    with pytest.raises(ConfigError, match="Step 0"):
        registry.register_trial(
            run_meta(purpose="final", decider="live_jev"),
            git_commit=None,
            git_dirty=False,
            gates=reg.TrialGates(sdk_version=SDK),
        )
    # and neither does the right model on the wrong SDK version
    assert (
        registry.step0_suites(model="jev-2", sdk_version="0.5.0", entry_qset_hash=ENTRY_HASH, entry_text_qset_hash=ENTRY_TEXT_HASH)
        == frozenset()
    )


def test_a_partial_step0_key_is_still_refused(registry: reg.Registry, tmp_path: Path) -> None:
    write_probe_record(tmp_path, "determinism")
    write_probe_record(tmp_path, "order")
    registry.sync_step0(tmp_path)
    with pytest.raises(ConfigError, match="batch"):
        registry.register_trial(
            run_meta(purpose="tune", decider="live_jev", end=RELEASE.replace(day=14)),
            git_commit=None,
            git_dirty=False,
            gates=reg.TrialGates(sdk_version=SDK),
        )


def test_the_step0_gate_does_not_apply_to_a_non_jev_decider(registry: reg.Registry) -> None:
    # MockJev's constants are not the pinned model; Step 0 says nothing about them
    registry.register_trial(
        run_meta(purpose="tune", decider="mock_jev", end=RELEASE.replace(day=14)),
        git_commit=None,
        git_dirty=False,
        gates=reg.TrialGates(sdk_version=SDK),
    )


def test_sync_step0_skips_unreadable_records_and_prefers_the_newer_verdict(registry: reg.Registry, tmp_path: Path) -> None:
    records = tmp_path / "probes" / "step0" / "records"
    records.mkdir(parents=True, exist_ok=True)
    (records / "broken.json").write_text("{not json")
    write_probe_record(tmp_path, "determinism", verdict={"deterministic": False})
    assert registry.sync_step0(tmp_path) == 1

    write_probe_record(tmp_path, "determinism", verdict={"deterministic": True}, recorded_at=datetime(2026, 6, 1, tzinfo=UTC))
    assert registry.sync_step0(tmp_path) == 1
    row = registry.connection.execute("SELECT verdict_json FROM step0_records WHERE suite = 'determinism'").fetchone()
    assert json.loads(row["verdict_json"]) == {"deterministic": True}


def test_sync_step0_on_a_data_dir_without_records(registry: reg.Registry, tmp_path: Path) -> None:
    assert registry.sync_step0(tmp_path / "nothing-here") == 0


# ======================================================================================================================
# The scan-facts gate (section 8, 11.2 B3)
# ======================================================================================================================


def test_final_is_refused_without_scan_facts(registry: reg.Registry) -> None:
    with pytest.raises(ConfigError, match="scan-candidates"):
        registry.register_trial(
            run_meta(purpose="final"),
            git_commit=None,
            git_dirty=False,
            gates=reg.TrialGates(candidate_config_hash=CANDIDATE_HASH, enabled_pairs=(("SPY", "iron_condor"),)),
        )


def test_final_is_refused_when_a_pair_is_unsizeable_too_often(registry: reg.Registry, tmp_path: Path) -> None:
    write_scan_facts(
        tmp_path,
        [
            scan_row("SPY", "iron_condor", 2024, 0.05),
            scan_row("QQQ", "iron_condor", 2024, 0.40),  # above candidates.max_unsizeable_rate
            scan_row("QQQ", "iron_condor", 2023, 0.01),  # a good earlier year must not rescue it
        ],
    )
    facts = reg.load_scan_facts(tmp_path / "manifests" / "scan_candidates.json", CANDIDATE_HASH)
    assert facts is not None and facts.latest_year() == 2024
    with pytest.raises(ConfigError) as excinfo:
        registry.register_trial(
            run_meta(purpose="final"),
            git_commit=None,
            git_dirty=False,
            gates=reg.TrialGates(
                candidate_config_hash=CANDIDATE_HASH,
                enabled_pairs=(("SPY", "iron_condor"), ("QQQ", "iron_condor")),
                max_unsizeable_rate=0.25,
                scan_facts=facts,
            ),
        )
    assert "QQQ/iron_condor" in str(excinfo.value)
    assert "40.0%" in str(excinfo.value)


def test_final_is_refused_when_an_enabled_pair_was_never_scanned(registry: reg.Registry, tmp_path: Path) -> None:
    write_scan_facts(tmp_path, [scan_row("SPY", "iron_condor", 2024, 0.05)])
    facts = reg.load_scan_facts(tmp_path / "manifests" / "scan_candidates.json", CANDIDATE_HASH)
    problems = reg.scan_facts_problems(facts, enabled_pairs=(("SPY", "iron_condor"), ("IWM", "long_call")), max_unsizeable_rate=0.25)
    assert problems == ["IWM/long_call: no scan facts in the latest scanned year (2024)"]


def test_final_passes_on_sizeable_facts(registry: reg.Registry, tmp_path: Path) -> None:
    write_scan_facts(tmp_path, [scan_row("SPY", "iron_condor", 2024, 0.05), scan_row("QQQ", "long_call", 2024, 0.10)])
    facts = reg.load_scan_facts(tmp_path / "manifests" / "scan_candidates.json", CANDIDATE_HASH)
    registry.register_trial(
        run_meta(purpose="final"),
        git_commit=None,
        git_dirty=False,
        gates=reg.TrialGates(
            candidate_config_hash=CANDIDATE_HASH,
            enabled_pairs=(("SPY", "iron_condor"), ("QQQ", "long_call")),
            scan_facts=facts,
        ),
    )


def test_scan_facts_for_another_config_hash_are_not_facts_for_this_one(tmp_path: Path) -> None:
    write_scan_facts(tmp_path, [scan_row("SPY", "iron_condor", 2024, 0.05)], candidate_config_hash="other")
    assert reg.load_scan_facts(tmp_path / "manifests" / "scan_candidates.json", CANDIDATE_HASH) is None
    assert reg.load_scan_facts(tmp_path / "manifests" / "absent.json", CANDIDATE_HASH) is None


def test_a_malformed_scan_facts_file_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "scan.json"
    path.write_text('{"hash": {"rows": "not a list"}}')
    with pytest.raises(ConfigError, match="scan-candidates facts file"):
        reg.load_scan_facts(path, "hash")


# ======================================================================================================================
# Families and the DSR scope (12.6)
# ======================================================================================================================


def test_derived_families_are_idempotent_and_closed() -> None:
    assert reg.derived_family("exp001", "#baseline") == "exp001#baseline"
    assert reg.derived_family("exp001#baseline", "#baseline") == "exp001#baseline"
    assert reg.derived_family("exp001#shadow", "#reference") == "exp001#reference"
    with pytest.raises(ValueError, match="unknown derived family suffix"):
        reg.derived_family("exp001", "#something")


@pytest.mark.parametrize(
    ("purpose", "family", "flags", "counts"),
    [
        ("validate", "exp001", (), True),
        ("tune", "exp001", (), True),
        ("final", "exp001", (), True),
        ("diagnostic", "exp001", (), False),
        ("reference", "exp001", (), False),
        ("paper", "exp001", (), False),
        ("validate", "exp001#baseline", (), False),
        ("validate", "exp001#shadow", (), False),
        ("validate", "exp001#reference", (), False),
        ("validate", "exp001#diagnostic", (), False),
        ("validate", "exp001#ablation", (), False),
        ("validate", "exp001", ("placebo",), False),
        ("validate", "exp001", ("unmasked",), False),
        ("validate", "exp001", ("baseline:4:seed=17",), False),
        ("validate", "exp001", ("ablation:buckets_only",), False),
        ("validate", "exp001", ("model_overlap",), False),
        ("validate", "exp001", ("reference_history",), False),
    ],
)
def test_is_selection_trial_scope(purpose: str, family: str, flags: tuple[str, ...], counts: bool) -> None:
    assert reg.is_selection_trial(purpose=purpose, family=family, flags=flags) is counts


def test_n_counts_failed_and_abandoned_and_excludes_the_derived_families(registry: reg.Registry) -> None:
    def register(run_id: str, *, status: str, **overrides: object) -> None:
        meta = run_meta(run_id=run_id, **overrides)
        registry.register_trial(meta, git_commit=None, git_dirty=False)
        if status != "registered":
            registry.set_status(run_id, "running")
            if status != "running":
                registry.set_status(run_id, status)

    register("sel-completed-1", status="completed")
    register("sel-completed-2", status="completed")
    register("sel-failed", status="failed")  # a spend-stopped attempt was still a look at the data
    register("sel-abandoned", status="abandoned")
    register("sel-running", status="running")  # not yet a look: it has produced no result
    register("sel-registered", status="registered")
    register("baseline-4", status="completed", family="exp001#baseline", flags=("baseline:4:seed=17",))
    register("shadow", status="completed", family="exp001#shadow", flags=("shadow",))
    register("reference", status="completed", family="exp001#reference", purpose="reference")
    register("diag", status="completed", family="exp001#diagnostic", purpose="diagnostic")
    register("ablation", status="completed", family="exp001#ablation", flags=("ablation:buckets_only",))
    register("other-family", status="completed", family="exp002")
    register("other-namespace", status="completed", namespace="exp001:jev-2:g0")
    register("other-window", status="completed", start=date(2000, 1, 3), end=date(2001, 12, 31))

    registry.record_result("sel-completed-1", n_days=500, ledger_head="h1", sharpe_orats=1.2, sharpe_worst=0.8, sharpe_mid=1.6)
    registry.record_result("sel-completed-2", n_days=500, ledger_head="h2", sharpe_orats=0.4, sharpe_worst=0.1, sharpe_mid=0.9)
    registry.record_result("baseline-4", n_days=500, ledger_head="h3", sharpe_orats=9.9)

    scope = registry.selection_trials(family="exp001", namespace="exp001:jev-1:g0", start=date(2012, 1, 3), end=date(2016, 12, 30))
    assert scope.n == 4  # two completed, one failed, one abandoned
    assert scope.completed == 2
    assert scope.failed == 1
    assert scope.abandoned == 1
    assert set(scope.run_ids) == {"sel-completed-1", "sel-completed-2", "sel-failed", "sel-abandoned"}
    # Var(SR_n) is taken over the COMPLETED subset only, and the ~1000 baseline seeds never reach it
    assert scope.sharpes["orats"] == (1.2, 0.4)
    assert 9.9 not in scope.sharpes["orats"]
    assert scope.composition == "4 (completed 2 / failed 1 / abandoned 1)"


def test_window_overlap_decides_membership(registry: reg.Registry) -> None:
    for run_id, start, end in [
        ("inside", date(2013, 1, 2), date(2014, 12, 31)),
        ("touching", date(2016, 12, 30), date(2018, 1, 2)),
        ("before", date(2008, 1, 2), date(2012, 1, 2)),
        ("open-ended", date(2018, 1, 2), None),
    ]:
        registry.register_trial(run_meta(run_id=run_id, start=start, end=end), git_commit=None, git_dirty=False)
        registry.set_status(run_id, "running")
        registry.set_status(run_id, "completed")
    scope = registry.selection_trials(family="exp001", namespace="exp001:jev-1:g0", start=date(2012, 1, 3), end=date(2016, 12, 30))
    assert set(scope.run_ids) == {"inside", "touching"}


# ======================================================================================================================
# Pre-registration rows
# ======================================================================================================================


def test_prereg_rows_are_write_once_per_hash(registry: reg.Registry) -> None:
    stored = registry.put_prereg(prereg_id="prereg.v1", sha256="a" * 64, git_commit="c" * 40, body_toml="[prereg]\n")
    assert stored.sha256 == "a" * 64
    again = registry.put_prereg(prereg_id="prereg.v1", sha256="a" * 64, git_commit="c" * 40, body_toml="[prereg]\n")
    assert again.registered_at == stored.registered_at  # idempotent
    with pytest.raises(PreregError, match="NEW prereg id"):
        registry.put_prereg(prereg_id="prereg.v1", sha256="b" * 64, git_commit="c" * 40, body_toml="[prereg]\n")
    assert registry.get_prereg("absent") is None


# ======================================================================================================================
# git helpers
# ======================================================================================================================


def make_git_repo(root: Path, files: dict[str, str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    assert reg.run_git(["init", "-q"], root).returncode == 0
    for name, text in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    assert reg.run_git(["add", "-A"], root).returncode == 0
    commit = reg.run_git(
        [
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "user.name=Fixture",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        root,
    )
    assert commit.returncode == 0, commit.stderr


def test_git_head_and_path_state_on_a_real_repository(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    make_git_repo(root, {"prereg/prereg.v1.toml": "[prereg]\nid = 'prereg.v1'\n"})

    commit, dirty = reg.git_head(root)
    assert commit is not None and len(commit) == 40
    assert dirty is False
    assert reg.git_path_committed(root / "prereg" / "prereg.v1.toml", root) is True

    (root / "prereg" / "prereg.v1.toml").write_text("[prereg]\nid = 'prereg.v2'\n")
    assert reg.git_path_committed(root / "prereg" / "prereg.v1.toml", root) is False
    assert reg.git_head(root)[1] is True

    (root / "untracked.toml").write_text("x = 1\n")
    assert reg.git_path_committed(root / "untracked.toml", root) is False


def test_git_helpers_on_a_directory_that_is_not_a_repository(tmp_path: Path) -> None:
    commit, dirty = reg.git_head(tmp_path)
    assert commit is None
    assert dirty is True  # a tree git cannot describe is never reported as reproducible
    assert reg.git_path_committed(tmp_path / "file.toml", tmp_path) is False


def test_a_path_outside_the_repository_is_never_committed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    make_git_repo(root, {"a.toml": "x = 1\n"})
    outside = tmp_path / "outside.toml"
    outside.write_text("x = 1\n")
    assert reg.git_path_committed(outside, root) is False


def test_trial_gates_from_config_reads_the_enabled_pairs_and_the_recorded_facts(tmp_path: Path) -> None:
    cfg = Config()
    write_scan_facts(tmp_path, [scan_row("SPY", "iron_condor", 2024, 0.05)])
    gates = reg.TrialGates.from_config(cfg, candidate_config_hash=CANDIDATE_HASH, data_dir=tmp_path, sdk_version=SDK)

    assert len(gates.enabled_pairs) == len(cfg.universe.underlyings) * len(cfg.structures.enabled)
    assert ("SPY", "iron_condor") in gates.enabled_pairs
    assert gates.max_unsizeable_rate == pytest.approx(cfg.candidates.max_unsizeable_rate)
    assert gates.scan_facts is not None
    assert gates.scan_facts.latest_year() == 2024
    assert gates.sdk_version == SDK
    # without an explicit version the gate keys on the installed pin, never on "whatever was recorded"
    assert reg.TrialGates.from_config(cfg, candidate_config_hash="other", data_dir=tmp_path).sdk_version == "0.6.0"


def test_installed_sdk_version_is_the_pinned_one() -> None:
    assert reg.installed_sdk_version() == "0.6.0"  # the exact pin of 1.1
