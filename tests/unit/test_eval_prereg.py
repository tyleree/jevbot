"""The pre-registration: the shipped file, its validator, the test-defining hash and every `register` refusal.

DESIGN.md 12.1 and section 14. The point of these tests is that the endpoint cannot be moved after the fact:

* the committed `prereg/prereg.v1.toml` really is the file 12.1 specifies, and it validates;
* the validator refuses the edits that would quietly change what is being tested - a family that includes the near-0.5
  direction questions, a variant other than `base`, a fallback to raw implied, a bootstrap block shorter than twice the
  longest horizon, an interval that is not a size candidate, alphas beyond the Bonferroni budget;
* `register` refuses an uncommitted file, an empty or unverifiable reference history, a missing or stale power file, a
  size study in which no interval holds, an interval that differs from the validated one, and `EM_WEEKDAY_BIAS`.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import msgspec
import pytest

from jevbot import canon
from jevbot.errors import PreregError
from jevbot.eval import prereg as pre
from jevbot.eval import registry as reg
from tests.fixtures.make_run_fixture import RunFixture, make_reference_history_store
from tests.unit.test_eval_registry import make_git_repo

REPO = Path(__file__).resolve().parents[2]
SHIPPED = REPO / "prereg" / "prereg.v1.toml"


# ======================================================================================================================
# The shipped file
# ======================================================================================================================


def test_the_committed_prereg_file_loads_and_validates() -> None:
    loaded = pre.load_prereg(SHIPPED)
    spec = loaded.prereg

    assert loaded.sha256 == canon.sha256_hex(SHIPPED.read_bytes())
    assert spec.id == "prereg.v1"
    assert spec.primary_family == (
        "eval.down_1em_1s",
        "eval.up_1em_1s",
        "eval.inside_1em_1s",
        "eval.down_1em_5s",
        "eval.up_1em_5s",
        "eval.inside_1em_5s",
    )
    assert pre.NON_EM_DIRECTION_IDS.isdisjoint(spec.primary_family)
    assert spec.primary_forecasts == "with_text"  # V9
    assert spec.primary_variant == "base"
    assert spec.primary_tier == "A"
    assert spec.unit == "session"
    assert spec.references == ("implied_recalibrated", "base_rate_expanding")
    assert spec.reference_min_events == 250
    assert spec.reference_fallback == "none"  # there is NO fallback on the verdict path
    assert spec.bootstrap.kind == "stationary"
    assert spec.bootstrap.min_block == 10  # 2 x the longest primary horizon (5 sessions)
    assert spec.bootstrap.interval == "percentile"
    assert spec.interval_candidates == ("percentile", "studentised", "null_calibrated")
    assert [(look.n_sessions, look.alpha) for look in spec.looks] == [(120, 0.01), (250, 0.04)]
    assert pre.BSS_VS_RAW_IMPLIED in spec.never_sufficient
    assert "D12 literal - not a verdict" in spec.d12_literal


def test_the_shipped_reference_history_block_is_documented_but_not_yet_filled() -> None:
    spec = pre.load_prereg(SHIPPED).prereg
    history = spec.reference_history
    # the committed file names the rules; the run id / head / manifest are filled once the reference run exists
    assert history.use and history.purge and history.own_events
    assert not history.is_filled


def test_the_ablation_and_model_change_blocks_are_registered_with_their_expected_result() -> None:
    spec = pre.load_prereg(SHIPPED).prereg
    assert len(spec.ablation_buckets_only.arms) == 2
    assert "0.90" in spec.ablation_buckets_only.expected  # the pre-registered agreement threshold (G9)
    assert spec.ablation_buckets_only.command.startswith("jevbot baselines run")
    assert "never pooled" in spec.model_change.policy
    assert len(spec.model_change.agreement_report) == 3


def test_primary_horizons_are_read_off_the_question_ids() -> None:
    assert pre.primary_horizon("eval.down_1em_1s") == 1
    assert pre.primary_horizon("eval.inside_1em_5s") == 5
    assert pre.primary_horizon("eval.up_1em_hold") is None  # only the config knows the hold horizon


# ======================================================================================================================
# The validator
# ======================================================================================================================


def _spec(**overrides: object) -> pre.Prereg:
    return msgspec.structs.replace(pre.load_prereg(SHIPPED).prereg, **overrides)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"primary_family": ("eval.up_1s", "eval.down_1em_1s")}, "must exclude eval.up_1s"),
        ({"primary_family": ("under.direction",)}, "non-evaluation question ids"),
        ({"primary_family": ()}, "primary_family is empty"),
        ({"primary_family": ("eval.down_1em_1s", "eval.down_1em_1s")}, "repeats a question id"),
        ({"primary_variant": "bucket_only"}, "primary_variant must be 'base'"),
        ({"primary_tier": "C"}, "primary_tier must be 'A' or 'B'"),
        ({"primary_forecasts": "both"}, "primary_forecasts must be"),
        ({"unit": "trade"}, "unit must be 'session'"),
        ({"references": ("implied_recalibrated",)}, "JOINT"),
        ({"references": ("implied_recalibrated", "raw_implied")}, "unknown reference"),
        ({"reference_fallback": "raw_implied"}, "reference_fallback must be 'none'"),
        ({"reference_min_events": 0}, "reference_min_events must be positive"),
        ({"never_sufficient": ()}, "never_sufficient must contain"),
        ({"d12_literal": ""}, "d12_literal is empty"),
        ({"interval_candidates": ("percentile", "percentile")}, "repeats a method"),
        ({"interval_candidates": ("bayesian",)}, "unknown interval candidate"),
    ],
)
def test_validate_prereg_refuses_edits_that_change_the_test(overrides: dict[str, object], expected: str) -> None:
    with pytest.raises(PreregError, match=expected):
        pre.validate_prereg(_spec(**overrides))


def test_validate_prereg_refuses_a_block_shorter_than_twice_the_longest_horizon() -> None:
    bootstrap = msgspec.structs.replace(pre.load_prereg(SHIPPED).prereg.bootstrap, min_block=9)
    with pytest.raises(PreregError, match="at least 2 x the longest primary horizon \\(10\\)"):
        pre.validate_prereg(_spec(bootstrap=bootstrap))


def test_validate_prereg_refuses_an_interval_outside_the_size_candidates() -> None:
    bootstrap = msgspec.structs.replace(pre.load_prereg(SHIPPED).prereg.bootstrap, interval="null_calibrated")
    pre.validate_prereg(_spec(bootstrap=bootstrap))  # a candidate: fine at load time, checked against the size study later
    bootstrap = msgspec.structs.replace(pre.load_prereg(SHIPPED).prereg.bootstrap, interval="bca")
    with pytest.raises(PreregError, match="is not one of the interval_candidates"):
        pre.validate_prereg(_spec(bootstrap=bootstrap))


def test_validate_prereg_enforces_the_look_schedule_and_the_alpha_budget() -> None:
    with pytest.raises(PreregError, match="must increase"):
        pre.validate_prereg(_spec(looks=(pre.Look(n_sessions=250, alpha=0.01), pre.Look(n_sessions=120, alpha=0.04))))
    with pytest.raises(PreregError, match="Bonferroni budget"):
        pre.validate_prereg(_spec(looks=(pre.Look(n_sessions=120, alpha=0.04), pre.Look(n_sessions=250, alpha=0.04))))
    with pytest.raises(PreregError, match="alpha must be in"):
        pre.validate_prereg(_spec(looks=(pre.Look(n_sessions=120, alpha=0.0),)))
    with pytest.raises(PreregError, match="looks is empty"):
        pre.validate_prereg(_spec(looks=()))


def test_load_prereg_reports_a_bad_file_rather_than_guessing(tmp_path: Path) -> None:
    missing = tmp_path / "absent.toml"
    with pytest.raises(PreregError, match="cannot be read"):
        pre.load_prereg(missing)

    not_toml = tmp_path / "broken.toml"
    not_toml.write_text("this is not = = toml")
    with pytest.raises(PreregError, match="not valid TOML"):
        pre.load_prereg(not_toml)

    no_table = tmp_path / "empty.toml"
    no_table.write_text("[other]\nx = 1\n")
    with pytest.raises(PreregError, match="no \\[prereg\\] table"):
        pre.load_prereg(no_table)

    unknown_key = tmp_path / "typo.toml"
    unknown_key.write_text(SHIPPED.read_text() + '\n[prereg.extra]\nwhat = "is this"\n')
    with pytest.raises(PreregError):
        pre.load_prereg(unknown_key)


# ======================================================================================================================
# The test-defining hash
# ======================================================================================================================


def test_the_test_defining_hash_covers_the_test_and_not_the_prose() -> None:
    spec = pre.load_prereg(SHIPPED).prereg
    base = pre.test_defining_hash(spec)

    # re-wording a caveat does not invalidate a committed size study
    assert pre.test_defining_hash(_spec(sensitivity=("a different list",))) == base
    assert pre.test_defining_hash(_spec(secondary=("fewer things",))) == base
    assert pre.test_defining_hash(_spec(pnl_verdict="reworded")) == base

    # but anything that changes what `eval power` simulated does
    assert pre.test_defining_hash(_spec(looks=(pre.Look(n_sessions=120, alpha=0.02),))) != base
    assert pre.test_defining_hash(_spec(reference_min_events=200)) != base
    assert pre.test_defining_hash(_spec(primary_family=spec.primary_family[:3])) != base
    assert pre.test_defining_hash(_spec(references=("base_rate_expanding", "implied_recalibrated"))) != base


def test_alphas_are_hashed_as_integer_parts_per_million() -> None:
    # canon refuses floats in hashed material (Conventions, INV-24): the hash existing at all proves the ppm rendering
    spec = pre.load_prereg(SHIPPED).prereg
    assert len(pre.test_defining_hash(spec)) == 64
    # 0.01 and 0.010000000000000002 are the same pre-registration
    nudged = pre.Look(n_sessions=120, alpha=0.01 + 1e-18)
    assert pre.test_defining_hash(_spec(looks=(nudged, spec.looks[1]))) == pre.test_defining_hash(spec)


# ======================================================================================================================
# Registration
# ======================================================================================================================


def power_file(
    spec: pre.Prereg,
    *,
    chosen_interval: str | None = "percentile",
    flags: tuple[str, ...] = (),
    weekday_tail_ratio: float = 1.2,
    test_hash: str | None = None,
    prereg_id: str | None = None,
) -> pre.PowerFile:
    size = tuple(
        pre.SizeEntry(
            interval=interval,
            null=null,
            look=index,
            n_sessions=look.n_sessions,
            alpha=look.alpha,
            rejection_rate=look.alpha * 0.9,
            reps=2000,
            holds=interval == chosen_interval,
        )
        for interval in spec.interval_candidates
        for null in ("N1:implied_recalibrated+noise", "N2:constant_climatological")
        for index, look in enumerate(spec.looks)
    )
    power = tuple(
        pre.PowerEntry(interval=chosen_interval or "percentile", look=index, bss=bss, power=0.5, reps=2000)
        for index, _look in enumerate(spec.looks)
        for bss in (0.005, 0.01, 0.02, 0.05)
    )
    return pre.PowerFile(
        prereg_id=prereg_id or spec.id,
        test_hash=test_hash or pre.test_defining_hash(spec),
        generated_at="2026-09-17T00:00:00Z",
        null_sim_reps=2000,
        size=size,
        power=power,
        chosen_interval=chosen_interval,
        weekday=tuple(pre.WeekdayEntry(weekday=day, n_events=100, frequency=0.11) for day in range(5)),
        weekday_tail_ratio=weekday_tail_ratio,
        flags=flags,
    )


class Bench:
    """A committed repository, a verified reference-history store and a registry - the state `register` needs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.repo = root / "repo"
        self.data_dir = root / "data"
        self.reference: RunFixture = make_reference_history_store(
            self.data_dir / "runs" / "20130102T000000-reference" / "run.sqlite", n_sessions=30
        )
        self.registry = reg.Registry.open(root)

    def prereg_text(self, *, fill_history: bool = True, **replacements: str) -> str:
        text = SHIPPED.read_text()
        if fill_history:
            text = text.replace('run_id = ""', f'run_id = "{self.reference.meta.run_id}"', 1)
            text = text.replace('ledger_head = ""', f'ledger_head = "{self.reference.head_hash}"', 1)
            text = text.replace('data_manifest_hash = ""', f'data_manifest_hash = "{self.reference.meta.data_manifest_hash}"', 1)
        for old, new in replacements.items():
            text = text.replace(old.replace("__", " "), new, 1)
        return text

    def commit(self, text: str, *, power: pre.PowerFile | None = None, commit_prereg: bool = True) -> Path:
        files = {"README.md": "fixture\n"}
        if commit_prereg:
            files["prereg/prereg.v1.toml"] = text
        if power is not None:
            files["prereg/power.v1.json"] = msgspec.json.encode(power).decode()
        make_git_repo(self.repo, files)
        path = self.repo / "prereg" / "prereg.v1.toml"
        if not commit_prereg:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return path

    def register(self, path: Path, **kwargs: object) -> reg.PreregRow:
        return pre.register(
            path,
            registry=self.registry,
            repo_root=self.repo,
            data_dir=self.data_dir,
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.fixture
def bench(tmp_path: Path) -> Bench:
    made = Bench(tmp_path)
    yield made
    made.registry.close()


def test_register_stores_the_hash_and_the_commit(bench: Bench) -> None:
    text = bench.prereg_text()
    spec = msgspec.convert(msgspec.toml.decode(text.encode())["prereg"], type=pre.Prereg)
    path = bench.commit(text, power=power_file(spec))

    row = bench.register(path)
    assert row.prereg_id == "prereg.v1"
    assert row.sha256 == canon.sha256_hex(path.read_bytes())
    assert len(row.git_commit) == 40
    assert row.body_toml == text
    assert bench.registry.get_prereg("prereg.v1") == row


def test_register_refuses_an_uncommitted_file(bench: Bench) -> None:
    text = bench.prereg_text()
    spec = msgspec.convert(msgspec.toml.decode(text.encode())["prereg"], type=pre.Prereg)
    path = bench.commit(text, power=power_file(spec), commit_prereg=False)
    with pytest.raises(PreregError, match="not committed"):
        bench.register(path)


def test_register_refuses_an_empty_reference_history(bench: Bench) -> None:
    text = bench.prereg_text(fill_history=False)
    spec = msgspec.convert(msgspec.toml.decode(text.encode())["prereg"], type=pre.Prereg)
    path = bench.commit(text, power=power_file(spec))
    with pytest.raises(PreregError) as excinfo:
        bench.register(path)
    message = str(excinfo.value)
    assert "reference_history" in message
    assert "run_id" in message and "ledger_head" in message and "data_manifest_hash" in message


def test_register_refuses_a_reference_history_that_does_not_verify(bench: Bench) -> None:
    wrong_head = bench.prereg_text().replace(bench.reference.head_hash, "f" * 64, 1)
    spec = msgspec.convert(msgspec.toml.decode(wrong_head.encode())["prereg"], type=pre.Prereg)
    path = bench.commit(wrong_head, power=power_file(spec))
    with pytest.raises(PreregError, match="ledger head"):
        bench.register(path)


def test_register_refuses_a_reference_store_that_is_missing(bench: Bench) -> None:
    text = bench.prereg_text()
    spec = msgspec.convert(msgspec.toml.decode(text.encode())["prereg"], type=pre.Prereg)
    path = bench.commit(text, power=power_file(spec))
    with pytest.raises(PreregError, match="does not exist"):
        bench.register(path, reference_store=bench.root / "nowhere" / "run.sqlite")


def test_register_refuses_a_run_that_is_not_the_reference_run(bench: Bench, tmp_path: Path) -> None:
    other = make_reference_history_store(tmp_path / "other" / "run.sqlite", run_id="other-run", n_sessions=10)
    text = bench.prereg_text()
    spec = msgspec.convert(msgspec.toml.decode(text.encode())["prereg"], type=pre.Prereg)
    path = bench.commit(text, power=power_file(spec))
    with pytest.raises(PreregError, match="ledger head"):
        bench.register(path, reference_store=other.path)


def test_register_refuses_a_missing_or_stale_power_file(bench: Bench) -> None:
    text = bench.prereg_text()
    spec = msgspec.convert(msgspec.toml.decode(text.encode())["prereg"], type=pre.Prereg)

    path = bench.commit(text)  # no power file at all
    with pytest.raises(PreregError, match="jevbot eval power"):
        bench.register(path)

    stale = power_file(spec, test_hash="0" * 64)
    (bench.repo / "prereg" / "power.v1.json").write_text(msgspec.json.encode(stale).decode())
    with pytest.raises(PreregError, match="stale"):
        bench.register(path)

    wrong_id = power_file(spec, prereg_id="prereg.v0")
    (bench.repo / "prereg" / "power.v1.json").write_text(msgspec.json.encode(wrong_id).decode())
    with pytest.raises(PreregError, match="stale"):
        bench.register(path)

    broken = bench.repo / "prereg" / "power.v1.json"
    broken.write_text("{not json")
    with pytest.raises(PreregError, match="not a valid"):
        bench.register(path)


def test_register_refuses_when_no_interval_holds_its_size(bench: Bench) -> None:
    text = bench.prereg_text()
    spec = msgspec.convert(msgspec.toml.decode(text.encode())["prereg"], type=pre.Prereg)
    path = bench.commit(text, power=power_file(spec, chosen_interval=None))
    with pytest.raises(PreregError, match="no interval candidate holds its size"):
        bench.register(path)


def test_register_refuses_an_interval_that_is_not_the_size_validated_one(bench: Bench) -> None:
    text = bench.prereg_text()
    spec = msgspec.convert(msgspec.toml.decode(text.encode())["prereg"], type=pre.Prereg)
    path = bench.commit(text, power=power_file(spec, chosen_interval="studentised"))
    with pytest.raises(PreregError, match="MUST equal the first candidate"):
        bench.register(path)


def test_register_refuses_the_weekday_bias_flag(bench: Bench) -> None:
    text = bench.prereg_text()
    spec = msgspec.convert(msgspec.toml.decode(text.encode())["prereg"], type=pre.Prereg)
    path = bench.commit(text, power=power_file(spec, flags=(pre.EM_WEEKDAY_BIAS,)))
    with pytest.raises(PreregError, match=pre.EM_WEEKDAY_BIAS):
        bench.register(path)


def test_register_refuses_a_weekday_tail_ratio_above_the_limit(bench: Bench) -> None:
    text = bench.prereg_text()
    spec = msgspec.convert(msgspec.toml.decode(text.encode())["prereg"], type=pre.Prereg)
    path = bench.commit(text, power=power_file(spec, weekday_tail_ratio=1.9))
    with pytest.raises(PreregError, match="weekday tail ratio"):
        bench.register(path, weekday_tail_ratio_max=1.6)


def test_register_needs_somewhere_to_find_the_reference_store(bench: Bench) -> None:
    text = bench.prereg_text()
    spec = msgspec.convert(msgspec.toml.decode(text.encode())["prereg"], type=pre.Prereg)
    path = bench.commit(text, power=power_file(spec))
    with pytest.raises(PreregError, match="must be verified"):
        pre.register(path, registry=bench.registry, repo_root=bench.repo)


def test_a_power_file_round_trips_through_json(bench: Bench, tmp_path: Path) -> None:
    spec = pre.load_prereg(SHIPPED).prereg
    path = tmp_path / "power.v1.json"
    written = power_file(spec)
    path.write_text(msgspec.json.encode(written).decode())
    read_back = pre.load_power_file(path)
    assert read_back.chosen_interval == "percentile"
    assert read_back.test_hash == pre.test_defining_hash(spec)
    assert json.loads(path.read_text())["null_sim_reps"] == 2000


# ======================================================================================================================
# Prereg status and tagging
# ======================================================================================================================


def test_prereg_status_reports_a_changed_file(bench: Bench) -> None:
    loaded = pre.load_prereg(SHIPPED)
    assert pre.prereg_status(None, None) == "absent"
    assert pre.prereg_status(None, loaded) == "unregistered"

    row = bench.registry.put_prereg(prereg_id="prereg.v1", sha256=loaded.sha256, git_commit="c" * 40, body_toml=loaded.body_toml)
    assert pre.prereg_status(row, loaded) == "registered"

    other = bench.registry.put_prereg(prereg_id="prereg.v2", sha256="0" * 64, git_commit="c" * 40, body_toml=loaded.body_toml)
    assert pre.prereg_status(other, loaded) == "changed"


def test_forecasts_are_tagged_only_after_the_registration_and_with_an_unchanged_file() -> None:
    registered = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)
    after = datetime(2026, 9, 18, 19, 35, tzinfo=UTC)
    before = datetime(2026, 9, 16, 19, 35, tzinfo=UTC)

    assert pre.forecast_is_prereg(registered_at=registered, first_tier_a_at=after, status="registered")
    assert not pre.forecast_is_prereg(registered_at=registered, first_tier_a_at=before, status="registered")
    assert not pre.forecast_is_prereg(registered_at=registered, first_tier_a_at=after, status="changed")
    assert not pre.forecast_is_prereg(registered_at=None, first_tier_a_at=after, status="registered")
    # the stored text form compares exactly like the instants it spells
    assert pre.forecast_is_prereg(
        registered_at=canon.render_as_of(registered), first_tier_a_at=canon.render_as_of(after), status="registered"
    )
