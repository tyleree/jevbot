"""Tests for `jevbot.config` (DESIGN.md section 4; 15.1 row `config`, `logsetup`; INV-02, INV-18; V2, V4, V12)."""

import hashlib
import importlib.util
import inspect
import json
import logging
import math
import os
import re
import stat
import subprocess
import sys
import tomllib
from collections.abc import Iterator, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import msgspec
import pytest

from jevbot import config, logsetup
from jevbot.config import (
    PROTECTED_SECTIONS_PAPER,
    Config,
    ProbeStatus,
    Secrets,
    load_config,
    resolve,
    validate,
)
from jevbot.errors import ConfigError, PaperGuardError
from jevbot.types import (
    Band,
    CacheMode,
    FillRule,
    KillTrigger,
    MaskTerms,
    ProbeRecord,
    RunMode,
    Slot,
    StructureKind,
    TriggerAction,
    Variant,
)

REPO = Path(__file__).resolve().parents[2]
DESIGN = REPO / "docs" / "design" / "DESIGN.md"
DEFAULT_TOML = REPO / "config" / "default.toml"
PAPER_TOML = REPO / "config" / "paper.toml"

TS_KEY = "ts-key-7f3a9c2e51b8"
PK_KEY = "PKTESTKEY4Q9Z"
PK_SECRET = "alpaca-secret-Zx81mNq0"


@pytest.fixture(autouse=True)
def _restore_redaction_registry() -> Iterator[None]:
    """Secrets built in these tests register themselves with the process-wide redaction layer: put the registry back."""
    before = set(logsetup._secrets)
    yield
    logsetup.clear_secrets()
    for value in before:
        logsetup.register_secret(value)


def design_toml() -> dict[str, Any]:
    text = DESIGN.read_text(encoding="utf-8")
    section = text[text.index("## 4. Config schema") : text.index("## 5. StateBuilder spec")]
    blocks = re.findall(r"```toml\n(.*?)```", section, flags=re.S)
    assert len(blocks) == 1, "section 4 holds exactly one fenced toml block"
    return tomllib.loads(blocks[0])


def leaf_paths(tree: Mapping[str, Any], prefix: str = "") -> set[str]:
    out: set[str] = set()
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping) and value:
            out |= leaf_paths(value, path)
        else:
            out.add(path)
    return out


def normalise(obj: Any) -> Any:
    """tuples -> lists, so that struct builtins compare equal to decoded TOML."""
    if isinstance(obj, Mapping):
        return {k: normalise(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [normalise(v) for v in obj]
    return obj


def write(path: Path, text: str, mode: int = 0o600) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


def no_keys() -> Secrets:
    return Secrets()


def jev_key() -> Secrets:
    return Secrets(typesafe_api_key=TS_KEY)


def alpaca_keys() -> Secrets:
    return Secrets(alpaca_paper_key=PK_KEY, alpaca_paper_secret=PK_SECRET)


def all_keys() -> Secrets:
    return Secrets(typesafe_api_key=TS_KEY, alpaca_paper_key=PK_KEY, alpaca_paper_secret=PK_SECRET)


ALL_PROBES = ProbeStatus(meta=True, determinism=True, batch=True, order=True, text=True)

# the two non-config inputs of state_config_hash (section 4): a MaskTerms.version (sha256[:12], 5.8) and a buckets.bucket_spec_hash (5.5)
MASK_VERSION = "0123456789ab"
BUCKET_SPEC_HASH = "f" * 64


def resolve_(
    cfg: Config, secrets_: Secrets, probes: ProbeStatus, *, mask_version: str = MASK_VERSION, bucket_spec_hash: str = BUCKET_SPEC_HASH
) -> config.ResolvedConfig:
    """`config.resolve` with the two required state inputs filled in (they only enter `state_config_hash`, tested below)."""
    return resolve(cfg, secrets_, probes, mask_version=mask_version, bucket_spec_hash=bucket_spec_hash)


# ======================================================================================================================
# default.toml: loads, equals the DESIGN block, equals the struct defaults, round-trips
# ======================================================================================================================


def test_default_toml_loads_and_equals_struct_defaults() -> None:
    cfg = load_config(None)
    assert isinstance(cfg, Config)
    assert cfg == Config(), "the struct defaults and config/default.toml are the same values"
    assert cfg.run.mode is RunMode.BACKTEST
    assert cfg.run.start == date(2012, 1, 3) and cfg.run.end == date(2025, 12, 12)
    assert cfg.jev.model == "jev-1.13.0" and cfg.jev.model_release_date == date(2026, 9, 15)
    assert cfg.cadence.fill_rule is FillRule.NEXT_SNAPSHOT
    assert cfg.jev.cache.mode is CacheMode.REPLAY
    assert cfg.structures.enabled == tuple(StructureKind)
    assert cfg.rules.perturbation_variants == (Variant.OPT_PERM, Variant.KEY_PERM, Variant.BUCKET_ONLY)
    assert cfg.recorder.slots == (Slot.DEC, Slot.EXEC, Slot.EOD)
    assert cfg.orders.repost_same_id is False  # V4
    assert cfg.orders.http_timeout_s == (3.05, 10.0)
    assert cfg.rules.tiers.score == ((0.755, 1.0), (0.655, 0.75), (0.555, 0.5))
    assert cfg.data.fomc.exceptions == ()  # shipped EMPTY (D23)
    assert cfg.jev.spend.max_input_tokens_per_day == 300_000_000 > cfg.jev.spend.max_input_tokens_per_run == 250_000_000
    assert cfg.paper.wind_down is False


def test_default_toml_is_the_design_block_key_for_key() -> None:
    documented = design_toml()
    shipped = tomllib.loads(DEFAULT_TOML.read_text(encoding="utf-8"))
    assert shipped == documented, "config/default.toml must equal the fenced toml block of DESIGN.md section 4"


def test_every_documented_key_exists_in_the_schema_and_nothing_else() -> None:
    documented = design_toml()
    builtins = normalise(msgspec.to_builtins(load_config(None)))
    assert leaf_paths(builtins) == leaf_paths(documented)
    assert builtins == normalise(documented), "every documented default is the loaded value (dates as ISO strings)"
    # spot checks of keys that revision 2 of the design added
    paths = leaf_paths(builtins)
    for key in (
        "run.family",
        "run.ledger_forecasts",
        "candidates.long_min_delta",
        "candidates.max_unsizeable_rate",
        "data.fomc.expected_per_year",
        "data.fomc.exceptions",
        "data.synthetic.planted_drift_bp",
        "eval.null_sim_reps",
        "eval.weekday_tail_ratio_max",
        "paper.wind_down",
        "health.stale_kill_after_sessions",
        "kill.actions.stale_quotes",
    ):
        assert key in paths


def test_kill_actions_cover_every_trigger_and_default_to_escalation() -> None:
    cfg = load_config(None)
    fields = {f.name for f in msgspec.structs.fields(config.KillActions)}
    assert fields == {t.value for t in KillTrigger}, "every D17 trigger is present"
    halt_then_kill = {KillTrigger.CLOCK_SKEW, KillTrigger.STALE_QUOTES, KillTrigger.JEV_ERRORS, KillTrigger.BROKER_ERRORS}
    for trigger in KillTrigger:
        expected = TriggerAction.HALT_THEN_KILL if trigger in halt_then_kill else TriggerAction.KILL
        assert cfg.kill.actions.for_trigger(trigger) is expected
    assert config.kill_disabled(cfg) == ()  # V2: with the shipped defaults every trigger escalates


def test_toml_round_trip_default_and_modified(tmp_path: Path) -> None:
    cfg = load_config(None)
    text = config.dumps_toml(cfg)
    assert msgspec.convert(tomllib.loads(text), type=Config) == cfg
    assert load_config(write(tmp_path / "resolved.toml", text)) == cfg

    exception_file = write(
        tmp_path / "fomc.toml",
        '[data.fomc]\nexceptions = [{year = 2020, expected = 7, reason = "a cancelled meeting", source_url = "https://www.federalreserve.gov/x"}]\n',
    )
    modified = load_config(
        exception_file,
        [
            "run.experiment=exp-rt.2",
            'universe.kind.SPY="a \\"quoted\\" fund ' + chr(0xE9) + '"',
            "rules.min_score=0.6",
            "orders.http_timeout_s=[2.5, 9]",
            "risk.one_per_underlying_direction=false",
            "cadence.fill_rule=same_snapshot_worst",
            "run.start=2015-06-01",
        ],
    )
    assert modified.data.fomc.exceptions[0].year == 2020
    assert modified.universe.kind["SPY"] == 'a "quoted" fund ' + chr(0xE9)
    dumped = config.dumps_toml(modified)
    assert load_config(write(tmp_path / "modified.toml", dumped)) == modified
    assert msgspec.convert(tomllib.loads(dumped), type=Config) == modified


def test_paper_toml_loads_as_paper_profile() -> None:
    cfg = load_config(PAPER_TOML)
    assert cfg.run.mode is RunMode.PAPER
    assert cfg.run.purpose == "paper"
    assert cfg.jev.cache.mode is CacheMode.RECORD
    # the expected initial state (waitlisted, no TypeSafe key): the default paper profile must resolve without any hand-edited flag (V12)
    rc = resolve_(cfg, alpaca_keys(), ProbeStatus())
    assert rc.cfg.decider.kind == "mock"
    assert (rc.news_resolved, rc.news_reason) == (True, "auto_keys_present")


# ======================================================================================================================
# unknown keys, merge order, overrides
# ======================================================================================================================


@pytest.mark.parametrize(
    "text",
    [
        "[risk]\nmax_loss_per_trade_pct_typo = 0.01\n",
        "[riskk]\nmax_open_structures = 3\n",
        "[rules.weights]\nallign = 0.3\n",
        '[kill.actions]\nmeteor_strike = "kill"\n',
        "top_level_key = 1\n",
    ],
)
def test_unknown_key_in_a_file_is_a_startup_error(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError, match="unknown field"):
        load_config(write(tmp_path / "typo.toml", text))


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ('[run]\nmode = "live"\n', "run.mode"),
        ('[run]\nseed = "abc"\n', "run.seed"),
        ("[risk]\nmax_open_structures = 2.5\n", "risk.max_open_structures"),
        ('[kill.actions]\ndrawdown = "ignore"\n', "kill.actions.drawdown"),
        ('[structures]\nenabled = ["naked_call"]\n', "structures.enabled"),
        ("[orders]\nhttp_timeout_s = [1.0, 2.0, 3.0]\n", "orders.http_timeout_s"),
        ('[run]\nstart = "2012-13-45"\n', "run.start"),
    ],
)
def test_wrong_types_and_enum_values_are_refused(tmp_path: Path, text: str, fragment: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(f"$.{fragment}")):
        load_config(write(tmp_path / "bad.toml", text))


def test_invalid_toml_and_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(write(tmp_path / "broken.toml", "[run\nmode = "))
    with pytest.raises(ConfigError, match="cannot read config file"):
        load_config(tmp_path / "absent.toml")
    with pytest.raises(ConfigError, match="cannot read config file"):
        load_config(None, default_path=tmp_path / "no-default.toml")


def test_merge_order_default_then_files_then_overrides(tmp_path: Path) -> None:
    first = write(tmp_path / "a.toml", '[risk]\nmax_open_structures = 4\nmax_new_per_day = 1\n[universe.kind]\nSPY = "fund one"\n')
    second = write(tmp_path / "b.toml", "[risk]\nmax_open_structures = 5\n")
    cfg = load_config([first, second], ["risk.max_open_structures=3"])
    assert cfg.risk.max_open_structures == 3  # override beats files
    assert cfg.risk.max_new_per_day == 1  # first file survives where the second is silent
    assert cfg.risk.max_same_direction_structures == 3  # default survives
    assert cfg.universe.kind["SPY"] == "fund one" and cfg.universe.kind["QQQ"].startswith("US large-cap")  # tables merge key by key
    assert load_config([first, second]).risk.max_open_structures == 5  # later file beats earlier file
    assert load_config(str(first)).risk.max_open_structures == 4  # a str path is accepted too


def test_override_value_typing() -> None:
    cfg = load_config(
        None,
        [
            "run.seed=7",
            "rules.min_score=0.6",
            "risk.one_per_underlying_direction=false",
            "run.experiment=007",  # a str key keeps the raw text: never an int, never a TOML error
            'run.family="fam-1"',  # one pair of quotes is stripped
            "news.enabled=off",
            "cadence.fill_rule=same_snapshot_worst",
            "jev.cache.mode=record",
            'universe.underlyings=["SPY", "QQQ"]',
            "run.start=2015-01-02",
            "rules.gates.direction.p_top=0.61",
            "universe.alias.SPY=UNDERLYING_Z",
            "data.synthetic.start_price_usd.SPY=512.5",
            "rules.tiers.score=[[0.8, 1.0], [0.6, 0.5]]",
            "kill.actions.stale_quotes=halt",
            " eval.ci_level = 0.9 ",
        ],
    )
    assert cfg.run.seed == 7
    assert cfg.rules.min_score == 0.6
    assert cfg.risk.one_per_underlying_direction is False
    assert cfg.run.experiment == "007"
    assert cfg.run.family == "fam-1"
    assert cfg.news.enabled == "off"
    assert cfg.cadence.fill_rule is FillRule.SAME_SNAPSHOT_WORST
    assert cfg.jev.cache.mode is CacheMode.RECORD
    assert cfg.universe.underlyings == ("SPY", "QQQ")
    assert cfg.run.start == date(2015, 1, 2)
    assert cfg.rules.gates.direction.p_top == 0.61 and cfg.rules.gates.direction.margin == 0.245
    assert cfg.universe.alias == {"SPY": "UNDERLYING_Z", "QQQ": "UNDERLYING_B", "IWM": "UNDERLYING_C"}
    assert cfg.data.synthetic.start_price_usd["SPY"] == 512.5
    assert cfg.rules.tiers.score == ((0.8, 1.0), (0.6, 0.5))
    assert cfg.kill.actions.stale_quotes is TriggerAction.HALT
    assert config.kill_disabled(cfg) == ("stale_quotes",)  # V2: KILL_DISABLED:stale_quotes
    assert cfg.eval.ci_level == 0.9


def test_override_edge_cases(tmp_path: Path) -> None:
    # a quoted value that is not a valid TOML string (a Windows path: "\q" is no TOML escape) still loses exactly its quotes
    windows = load_config(None, [r'paper.alert_cmd="C:\qtools\alert.exe"'])
    assert windows.paper.alert_cmd == r"C:\qtools\alert.exe"
    # a quoted value with valid TOML escapes is read as a TOML string
    assert load_config(None, [r'paper.alert_cmd="say \"hi\""']).paper.alert_cmd == 'say "hi"'
    # an inline table merges into the existing table
    merged = load_config(None, ["rules.weights={align = 0.25, calm = 0.20}"])
    assert (merged.rules.weights.align, merged.rules.weights.calm, merged.rules.weights.fit) == (0.25, 0.20, 0.20)
    # an override may create tables that no file mentions: the struct defaults fill the rest
    sparse = write(tmp_path / "sparse.toml", "[run]\nseed = 3\n")
    built = load_config(None, ["jev.cache.mode=record", "kill.actions.stale_quotes=halt"], default_path=sparse)
    assert built.run.seed == 3
    assert built.jev.cache.mode is CacheMode.RECORD and built.jev.model == "jev-1.13.0"
    assert built.kill.actions.stale_quotes is TriggerAction.HALT and built.kill.actions.drawdown is TriggerAction.KILL
    expected = msgspec.structs.replace(
        Config(),
        run=msgspec.structs.replace(Config().run, seed=3),
        jev=msgspec.structs.replace(Config().jev, cache=config.JevCacheConfig(mode=CacheMode.RECORD)),
        kill=msgspec.structs.replace(
            Config().kill, actions=msgspec.structs.replace(Config().kill.actions, stale_quotes=TriggerAction.HALT)
        ),
    )
    assert built == expected


def test_unwritable_and_unhashable_values_are_refused() -> None:
    poisoned = msgspec.structs.replace(Config(), rules=msgspec.structs.replace(Config().rules, min_score=float("nan")))
    with pytest.raises(ConfigError, match="non-finite"):
        config.dumps_toml(poisoned)
    with pytest.raises(ConfigError, match="non-finite"):
        config.rules_hash(poisoned)
    with pytest.raises(ConfigError, match="non-finite"):
        validate(poisoned)
    with pytest.raises(ConfigError, match="cannot be hashed"):
        config._hash_material({"when": date(2026, 9, 17)})
    with pytest.raises(ConfigError, match="cannot write a value of type"):
        config._toml_value(date(2026, 9, 17))


def test_last_override_wins() -> None:
    assert load_config(None, ["run.seed=1", "run.seed=2"]).run.seed == 2


@pytest.mark.parametrize(
    ("item", "message"),
    [
        ("run.seed", "expected section.key=value"),
        ("seed=3", "expected section.key=value"),
        ("run..seed=3", "expected section.key=value"),
        ("=3", "expected section.key=value"),
        ("run.sed=3", "unknown config key"),
        ("rnu.seed=3", "unknown config key"),
        ("run.seed.deeper=3", "goes below a scalar"),
        ("rules.weights.allign=0.3", "unknown config key"),
        ("run.seed=abc", "cannot parse the value as TOML"),
        ("universe.underlyings=SPY,QQQ", "cannot parse the value as TOML"),
        ("run.seed=2.5", r"\$\.run\.seed"),
        ("run.mode=live", r"\$\.run\.mode"),
    ],
)
def test_malformed_or_unknown_override_is_refused(item: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(None, [item])


# ======================================================================================================================
# PROTECTED_SECTIONS_PAPER: one parametrised test over the constant
# ======================================================================================================================

_PROTECTED_EXAMPLES = {
    "risk": "risk.max_open_structures=5",
    "kill": "kill.actions.stale_quotes=halt",
    "health": "health.max_chain_age_s=60",
    "orders": "orders.repost_same_id=true",
    "dte": "dte.target=36",
    "exits": "exits.stop_loss_frac=0.4",
}


def test_protected_sections_constant_is_the_single_definition() -> None:
    assert PROTECTED_SECTIONS_PAPER == ("risk", "kill", "health", "orders", "dte", "exits")
    assert set(_PROTECTED_EXAMPLES) == set(PROTECTED_SECTIONS_PAPER), "this test file must cover every protected section"
    assert set(PROTECTED_SECTIONS_PAPER) <= {f.name for f in msgspec.structs.fields(Config)}


@pytest.mark.parametrize("section", PROTECTED_SECTIONS_PAPER)
def test_every_protected_section_refuses_override_in_paper_mode(section: str) -> None:
    item = _PROTECTED_EXAMPLES[section]
    assert item.startswith(section + ".")
    with pytest.raises(ConfigError, match="refused in paper mode") as err:
        load_config(PAPER_TOML, [item])
    assert item.split("=")[0] in str(err.value)
    # paper mode reached through -o itself is paper mode too, whatever the order of the options
    with pytest.raises(ConfigError, match="refused in paper mode"):
        load_config(None, [item, "run.mode=paper"])
    with pytest.raises(ConfigError, match="refused in paper mode"):
        load_config(None, ["run.mode=paper", item])
    # the very same override is fine in a backtest
    assert load_config(None, [item]) != load_config(None)


def test_unprotected_sections_accept_override_in_paper_mode() -> None:
    cfg = load_config(PAPER_TOML, ["news.enabled=off", "paper.heartbeat_s=10", "run.experiment=paper002"])
    assert (cfg.news.enabled, cfg.paper.heartbeat_s, cfg.run.experiment) == ("off", 10, "paper002")


def test_protected_limits_still_come_from_files_in_paper_mode(tmp_path: Path) -> None:
    limits = write(tmp_path / "limits.toml", "[risk]\nmax_open_structures = 4\n")
    assert load_config([PAPER_TOML, limits]).risk.max_open_structures == 4


# ======================================================================================================================
# Load-time validation rules (section 4)
# ======================================================================================================================


def bad(overrides: list[str], message: str, *, flags: tuple[str, ...] = ()) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(None, overrides, flags=flags)


def test_weights_must_sum_to_one() -> None:
    bad(["rules.weights.align=0.31"], "rules.weights must sum to 1.0")
    bad(["rules.weights.align=1.3", "rules.weights.calm=-0.15"], "every weight must be in")
    ok = load_config(None, ["rules.weights.align=0.25", "rules.weights.calm=0.20"])
    assert math.isclose(sum(msgspec.structs.astuple(ok.rules.weights)), 1.0)


def test_veto_and_hysteresis_ordering() -> None:
    message = "veto_lo < exit_pressure_release < exit_pressure_enter <= veto_hi"
    bad(["rules.exit_pressure_release=0.295"], message)  # release must be strictly above veto_lo
    bad(["rules.exit_pressure_release=0.705"], message)  # release must be strictly below enter
    bad(["rules.exit_pressure_enter=0.706"], message)  # enter must not exceed veto_hi
    bad(["rules.veto_lo=0.5"], message)
    assert load_config(None, ["rules.exit_pressure_enter=0.705"]).rules.exit_pressure_enter == 0.705  # == veto_hi is allowed
    assert load_config(None, ["rules.exit_pressure_enter=0.65"]).rules.exit_pressure_enter == 0.65


def test_cadence_offsets_ordering() -> None:
    message = "order_cutoff_offset_min < exec_offset_min < decide_offset_min"
    bad(["cadence.exec_offset_min=25"], message)
    bad(["cadence.exec_offset_min=5", "cadence.decision_deadline_offset_min=6"], message)
    bad(["cadence.decide_offset_min=20"], message)
    bad(["cadence.decision_deadline_offset_min=30"], "decision_deadline_offset_min")
    bad(["cadence.cancel_all_offset_min=6"], "cancel_all_offset_min")
    bad(["cadence.eod_offset_min=2"], "eod_offset_min")
    assert load_config(None, ["cadence.decide_offset_min=30", "cadence.exec_offset_min=24"]).cadence.exec_offset_min == 24


def test_hard_exit_sessions_at_least_two() -> None:
    bad(["dte.hard_exit_sessions=1"], "dte.hard_exit_sessions must be >= 2")
    assert load_config(None, ["dte.hard_exit_sessions=2"]).dte.hard_exit_sessions == 2
    bad(["dte.min_entry=40"], "min_entry <= target <= max_entry")
    bad(["dte.max_entry=121"], "data.max_dte")


@pytest.mark.parametrize("ladder", ["entry_ladder", "exit_ladder"])
def test_ladders_increase_and_end_at_one(ladder: str) -> None:
    message = f"orders.{ladder}: rungs"
    bad([f"orders.{ladder}=[0.5, 0.9]"], message)  # does not end at 1.0
    bad([f"orders.{ladder}=[0.75, 0.5, 1.0]"], message)  # not increasing
    bad([f"orders.{ladder}=[0.5, 0.5, 1.0]"], message)  # not STRICTLY increasing
    bad([f"orders.{ladder}=[0.5, 1.0, 1.1]"], message)  # past natural
    bad([f"orders.{ladder}=[0.0, 1.0]"], message)
    bad([f"orders.{ladder}=[]"], message)
    assert getattr(load_config(None, [f"orders.{ladder}=[1.0]"]).orders, ladder) == (1.0,)
    assert getattr(load_config(None, [f"orders.{ladder}=[0.25, 0.5, 0.75, 1.0]"]).orders, ladder) == (0.25, 0.5, 0.75, 1.0)


@pytest.mark.parametrize(
    "key",
    [
        "max_loss_per_trade_pct",
        "max_aggregate_open_loss_pct",
        "max_open_structures",
        "max_new_per_day",
        "max_same_direction_structures",
        "daily_loss_halt_pct",
        "drawdown_kill_pct",
        "max_bp_utilisation",
        "bp_drift_halt_pct",
        "max_adverse_drift",
        "max_contracts_per_trade",
        "max_order_notional_usd",
        "max_orders_per_minute",
        "max_order_attempts",
    ],
)
def test_every_risk_limit_must_be_positive(key: str) -> None:
    is_float = isinstance(getattr(Config().risk, key), float)
    bad([f"risk.{key}={'0.0' if is_float else '0'}"], f"risk.{key} must be")
    bad([f"risk.{key}={'-0.5' if is_float else '-1'}"], f"risk.{key} must be")


def test_risk_limit_bounds() -> None:
    bad(["risk.max_contracts_per_trade=11"], "max_contracts_per_trade must be <= 10")
    assert load_config(None, ["risk.max_contracts_per_trade=10"]).risk.max_contracts_per_trade == 10
    bad(["risk.bp_haircut_mult=0.9"], "bp_haircut_mult must be >= 1.0")
    bad(["risk.drawdown_kill_pct=1.5"], "risk.drawdown_kill_pct must be in")
    assert load_config(None, ["risk.drawdown_kill_pct=1.0"]).risk.drawdown_kill_pct == 1.0
    assert load_config(None, ["risk.event_blackout_sessions=0"]).risk.event_blackout_sessions == 0


def test_candidate_feasibility_verticals() -> None:
    # hand-computed cap: 0.75 * (0.25 + 0.12) / 2 = 0.13875
    bad(["candidates.min_credit_to_width=0.139"], r"min_credit_to_width = 0\.139 is infeasible.*0\.138750")
    assert load_config(None, ["candidates.min_credit_to_width=0.13875"]).candidates.min_credit_to_width == 0.13875
    # narrower deltas shrink the cap below the shipped floor: 0.75 * (0.20 + 0.10) / 2 = 0.1125 < 0.12
    bad(
        ["candidates.credit_short_delta=0.20", "candidates.credit_long_delta=0.10"], r"min_credit_to_width = 0\.12 is infeasible.*0\.112500"
    )
    assert load_config(
        None, ["candidates.credit_short_delta=0.20", "candidates.credit_long_delta=0.10", "candidates.min_credit_to_width=0.11"]
    )


def test_candidate_feasibility_condor() -> None:
    # hand-computed cap: 0.75 * (0.16 + 0.07) = 0.1725
    bad(["candidates.condor_min_credit_to_width=0.173"], r"condor_min_credit_to_width = 0\.173 is infeasible.*0\.172500")
    assert load_config(None, ["candidates.condor_min_credit_to_width=0.1725"]).candidates.condor_min_credit_to_width == 0.1725
    # 0.75 * (0.14 + 0.07) = 0.1575 < 0.17
    bad(["candidates.condor_short_delta=0.14"], r"condor_min_credit_to_width = 0\.17 is infeasible.*0\.157500")


def test_candidate_delta_sanity() -> None:
    bad(["candidates.long_min_delta=0.40"], "long_min_delta must be <= long_delta")
    bad(["candidates.credit_long_delta=0.30"], "credit_long_delta must be < credit_short_delta")
    bad(["candidates.debit_short_delta=0.55"], "debit_short_delta must be < debit_long_delta")
    bad(["candidates.max_credit_to_width=0.10"], "min_credit_to_width < max_credit_to_width")


def test_unmasked_state_forces_backtest_diagnostic() -> None:
    message = "state.unmasked = true forces"
    bad(["state.unmasked=true"], message)  # default purpose is "validate"
    bad(["state.unmasked=true", "run.purpose=final"], message)
    bad(["state.unmasked=true", "run.purpose=diagnostic", "run.mode=paper"], message)
    cfg = load_config(None, ["state.unmasked=true", "run.purpose=diagnostic"])
    assert cfg.state.unmasked and cfg.run.purpose == "diagnostic" and cfg.run.mode is RunMode.BACKTEST


def test_ledger_forecasts_false_only_for_baseline_4_or_shadow() -> None:
    message = "ledger_forecasts = false is accepted only"
    bad(["run.ledger_forecasts=false"], message)
    bad(["run.ledger_forecasts=false"], message, flags=("placebo",))
    bad(["run.ledger_forecasts=false"], message, flags=("baseline:3",))
    bad(["run.ledger_forecasts=false"], message, flags=("baseline:44:seed=1",))
    bad(["run.ledger_forecasts=false"], message, flags=("shadowy",))
    assert load_config(None, ["run.ledger_forecasts=false"], flags=("baseline:4:seed=17",)).run.ledger_forecasts is False
    assert load_config(None, ["run.ledger_forecasts=false"], flags=("shadow",)).run.ledger_forecasts is False
    # a programmatically built config is validated the same way, and flags=None skips exactly this rule (resolve_() uses it)
    built = msgspec.structs.replace(Config(), run=msgspec.structs.replace(Config().run, ledger_forecasts=False))
    with pytest.raises(ConfigError, match=message):
        validate(built)
    validate(built, flags=("baseline:4:seed=3",))
    validate(built, flags=None)
    assert resolve_(built, no_keys(), ProbeStatus()).cfg.run.ledger_forecasts is False


def test_stale_kill_after_sessions_at_least_two() -> None:
    bad(["health.stale_kill_after_sessions=1"], "stale_kill_after_sessions must be >= 2")
    assert load_config(None, ["health.stale_kill_after_sessions=2"]).health.stale_kill_after_sessions == 2


def test_kill_actions_downgrade_allowed_but_meaningless_escalation_refused() -> None:
    cfg = load_config(None, ["kill.actions.drawdown=halt", "kill.actions.jev_errors=halt"])
    assert config.kill_disabled(cfg) == ("drawdown", "jev_errors")
    bad(["kill.actions.drawdown=halt_then_kill"], "kill.actions.drawdown.*no persistence threshold")
    assert load_config(None, ["kill.actions.stale_quotes=kill"]).kill.actions.stale_quotes is TriggerAction.KILL


def test_other_consistency_rules() -> None:
    bad(["rules.min_score=nan"], "non-finite float")
    bad(["fees.orf=inf"], "non-finite float")
    bad(["jev.model=jev-latest"], "PINNED model id")
    bad(["jev.model=jev:1"], "jev.model")
    bad(["run.experiment=exp:1"], "run.experiment")
    bad(["run.experiment=exp#baseline"], "run.experiment")
    bad(["run.start=2026-01-01"], "run.start must not be after run.end")
    bad(["run.purpose=paper"], 'run.purpose = "paper" requires run.mode = "paper"')
    bad(["run.purpose=reference", "decider.kind=live"], "reference")
    bad(["universe.iv_proxy.SPY=VXX"], "not listed in data.cboe_indices")
    bad(['universe.underlyings=["SPY", "DIA"]'], "universe.alias: no entry for underlying DIA")
    bad(['universe.underlyings=["SPY", "SPY"]'], "underlyings must be unique")
    bad(['universe.underlyings=["spy"]'], "not an OCC root")
    bad(["universe.alias.QQQ=UNDERLYING_A"], "two underlyings share one alias")
    bad(["structures.enabled=[]"], "structures.enabled must not be empty")
    bad(['rules.perturbation_variants=["base"]'], 'must not contain "base"')
    bad(["fills.orats_p=[0.75, 0.66, 0.56, 0.4]"], "fills.orats_p")
    bad(["state.max_chars=20000"], "max_chars <= hard_max_chars")
    bad(["data.iv_rank_lo_pct=98"], "iv_rank_lo_pct < iv_rank_hi_pct")
    bad(["jev.retry_max=3"], "retry_backoff_s")
    bad(["rules.code_crosschecks=false", "run.mode=paper"], "code_crosschecks")
    bad(
        [
            "data.provider=synthetic",
            'universe.underlyings=["SPY", "DIA"]',
            "universe.alias.DIA=UNDERLYING_D",
            'universe.kind.DIA="fund"',
            "universe.iv_proxy.DIA=VIX",
        ],
        "start_price_usd: no entry for underlying DIA",
    )
    bad(["rules.tiers.score=[[0.5, 0.5], [0.7, 1.0]]"], "rules.tiers.score")
    bad(["rules.tiers.environment=[0.5, 0.75, 1.0, 0.0]"], "rules.tiers.environment")
    assert load_config(None, ["rules.code_crosschecks=false"]).rules.code_crosschecks is False  # baseline 3 (backtest)


def test_fomc_exceptions_are_never_assumed(tmp_path: Path) -> None:
    def exception(expected: int, reason: str, url: str) -> Path:
        return write(
            tmp_path / "fomc.toml",
            f'[[data.fomc.exceptions]]\nyear = 2020\nexpected = {expected}\nreason = "{reason}"\nsource_url = "{url}"\n',
        )

    good = load_config(
        exception(
            7,
            "a scheduled meeting is marked cancelled on the source page",
            "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        )
    )
    assert [(e.year, e.expected) for e in good.data.fomc.exceptions] == [(2020, 7)]
    with pytest.raises(ConfigError, match="7 is the only accepted value"):
        load_config(exception(6, "reason", "https://example.org/x"))
    with pytest.raises(ConfigError, match="reason must not be blank"):
        load_config(exception(7, " ", "https://example.org/x"))
    with pytest.raises(ConfigError, match="https:// source"):
        load_config(exception(7, "reason", "ftp://example.org/x"))


def test_all_violations_are_reported_together() -> None:
    with pytest.raises(ConfigError) as err:
        load_config(None, ["dte.hard_exit_sessions=1", "health.stale_kill_after_sessions=1", "rules.weights.align=0.5"])
    text = str(err.value)
    assert "dte.hard_exit_sessions" in text and "stale_kill_after_sessions" in text and "rules.weights must sum" in text


# ======================================================================================================================
# secrets(): blank = absent, forbidden env, SDK log level, .env handling, redaction, repr
# ======================================================================================================================


@pytest.mark.parametrize("blank", ["", " ", "\t", " \n "])
def test_blank_secrets_are_absent(blank: str) -> None:
    got = config.secrets({"TYPESAFE_API_KEY": blank, "ALPACA_PAPER_KEY": blank, "ALPACA_PAPER_SECRET": blank, "JEVBOT_DATA": blank}, None)
    assert got == Secrets()
    assert (got.typesafe_api_key, got.alpaca_paper_key, got.alpaca_paper_secret, got.data_dir) == (None, None, None, None)
    assert not got.has_typesafe_key and not got.has_alpaca_key and got.alpaca_key_paper_hint is None
    assert Secrets(typesafe_api_key=blank).typesafe_api_key is None, "a directly constructed Secrets never holds a blank key either"
    assert config.secrets({}, None) == Secrets()


def test_secret_values_are_stripped_and_read_only_from_the_constant_names() -> None:
    env = {
        "TYPESAFE_API_KEY": f"  {TS_KEY}\n",
        "ALPACA_PAPER_KEY": PK_KEY,
        "ALPACA_PAPER_SECRET": PK_SECRET,
        "JEVBOT_DATA": "/srv/jevbot-data",
        "TYPESAFE_KEY": "ignored",
        "ALPACA_KEY": "ignored",
    }
    got = config.secrets(env, None)
    assert got.typesafe_api_key == TS_KEY
    assert (got.alpaca_paper_key, got.alpaca_paper_secret, got.data_dir) == (PK_KEY, PK_SECRET, "/srv/jevbot-data")
    assert got.has_typesafe_key and got.has_alpaca_key and got.alpaca_key_paper_hint is True
    assert Secrets(alpaca_paper_key="AKLIVEKEY123").alpaca_key_paper_hint is False
    assert (config.ENV_JEVBOT_DATA, config.ENV_TYPESAFE_API_KEY, config.ENV_ALPACA_PAPER_KEY, config.ENV_ALPACA_PAPER_SECRET) == (
        "JEVBOT_DATA",
        "TYPESAFE_API_KEY",
        "ALPACA_PAPER_KEY",
        "ALPACA_PAPER_SECRET",
    )


@pytest.mark.parametrize(
    "name", ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "APCA_API_BASE_URL", "APCA_RETRY_MAX"]
)
@pytest.mark.parametrize("value", ["something", ""])
def test_forbidden_alpaca_env_is_a_paper_guard_error(name: str, value: str) -> None:  # INV-02
    with pytest.raises(PaperGuardError, match=name):
        config.secrets({"ALPACA_PAPER_KEY": PK_KEY, "ALPACA_PAPER_SECRET": PK_SECRET, name: value}, None)


def test_forbidden_names_inside_the_env_file_are_refused_too(tmp_path: Path) -> None:
    with pytest.raises(PaperGuardError, match="APCA_API_BASE_URL"):
        config.secrets({}, write(tmp_path / ".env", "APCA_API_BASE_URL=https://example.invalid\n"))
    with pytest.raises(ConfigError, match="TYPESAFE_BASE_URL"):
        config.secrets({}, write(tmp_path / ".env", "TYPESAFE_BASE_URL=https://example.invalid\n"))


@pytest.mark.parametrize("name", ["TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL"])
def test_sdk_environment_knobs_are_a_config_error(name: str) -> None:
    with pytest.raises(ConfigError, match=name) as err:
        config.secrets({name: "jev-latest"}, None)
    assert not isinstance(err.value, PaperGuardError)


@pytest.mark.parametrize("value", ["debug", " debug", "DEBUG ", "info", "INFO", "  Info\t", "verbose", "warnings", "trace", "0", "wa rn"])
def test_sdk_log_level_below_warning_or_unknown_is_refused(value: str) -> None:  # INV-18
    with pytest.raises(ConfigError, match="TYPESAFE_LOG_LEVEL"):
        config.check_sdk_log_level({"TYPESAFE_LOG_LEVEL": value})
    with pytest.raises(ConfigError, match="TYPESAFE_LOG_LEVEL"):
        config.secrets({"TYPESAFE_API_KEY": TS_KEY, "TYPESAFE_LOG_LEVEL": value}, None)


@pytest.mark.parametrize("value", ["", "   ", "warn", "WARNING ", " warning", "Warn", "error", "ERROR", "off", " OFF\n"])
def test_sdk_log_level_warning_or_stricter_passes(value: str) -> None:
    config.check_sdk_log_level({"TYPESAFE_LOG_LEVEL": value})
    assert config.secrets({"TYPESAFE_API_KEY": TS_KEY, "TYPESAFE_LOG_LEVEL": value}, None).typesafe_api_key == TS_KEY
    config.check_sdk_log_level({})
    assert config.ALLOWED_SDK_LOG_LEVELS == {"warn", "warning", "error", "off"}


def test_sdk_log_level_in_env_file_is_checked_too(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="TYPESAFE_LOG_LEVEL"):
        config.secrets({}, write(tmp_path / ".env", "TYPESAFE_LOG_LEVEL= debug\n"))
    with pytest.raises(ConfigError, match="TYPESAFE_LOG_LEVEL"):  # a blank variable in the environment does not hide the file's value
        config.secrets({"TYPESAFE_LOG_LEVEL": ""}, write(tmp_path / ".env", "TYPESAFE_LOG_LEVEL=debug\n"))


def test_our_normalisation_is_the_sdks_normalisation() -> None:
    """The guard mirrors `typesafe_sdk/_core/logging.py::setup_logging`; if the pinned SDK ever changes how it reads the level,
    this test fails. The SDK source is READ, never imported."""
    spec = importlib.util.find_spec("typesafe_sdk")
    assert spec is not None and spec.origin is not None
    source = (Path(spec.origin).parent / "_core" / "logging.py").read_text(encoding="utf-8")
    assert '(os.environ.get(LOG_LEVEL_ENV) or "").strip().lower()' in source
    levels = set(re.findall(r'^\s+"([a-z]+)": logging\.', source, flags=re.M))
    assert levels == {"debug", "info", "warn", "warning", "error", "off"}
    assert levels - config.ALLOWED_SDK_LOG_LEVELS == {"debug", "info"}
    constants = (Path(spec.origin).parent / "constants.py").read_text(encoding="utf-8")
    assert '"TYPESAFE_LOG_LEVEL"' in constants and config.ENV_TYPESAFE_LOG_LEVEL == "TYPESAFE_LOG_LEVEL"


def test_config_and_logsetup_never_import_the_vendor_sdks() -> None:
    code = (
        "import sys; import jevbot.config, jevbot.logsetup; "
        "bad = [m for m in ('typesafe_sdk', 'alpaca', 'httpx2', 'requests') if m in sys.modules]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    env = {k: v for k, v in os.environ.items() if k != "TYPESAFE_LOG_LEVEL"}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, env=env, check=False, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr


def test_env_file_is_read_but_never_overrides_set_variables(tmp_path: Path) -> None:
    env_file = write(
        tmp_path / ".env",
        "# jevbot secrets\n\n"
        f"TYPESAFE_API_KEY={TS_KEY}\n"
        f'export ALPACA_PAPER_KEY="{PK_KEY}"\n'
        f"ALPACA_PAPER_SECRET='{PK_SECRET}'\n"
        "JEVBOT_DATA = /srv/from-file \n",
    )
    from_file = config.secrets({}, env_file)
    assert from_file == Secrets(typesafe_api_key=TS_KEY, alpaca_paper_key=PK_KEY, alpaca_paper_secret=PK_SECRET, data_dir="/srv/from-file")
    # a variable that is already set wins; a blank one counts as absent, so the file fills it
    mixed = config.secrets({"TYPESAFE_API_KEY": "ts-from-environment", "ALPACA_PAPER_KEY": "  "}, env_file)
    assert mixed.typesafe_api_key == "ts-from-environment"
    assert mixed.alpaca_paper_key == PK_KEY
    assert config.secrets({"TYPESAFE_API_KEY": TS_KEY}, tmp_path / "absent.env").typesafe_api_key == TS_KEY  # a missing file is fine
    assert config.read_env_file(tmp_path / "absent.env") == {}
    assert config.read_env_file(write(tmp_path / "eq.env", "K=a=b#c $HOME\n")) == {
        "K": "a=b#c $HOME"
    }  # verbatim: no comments, no interpolation


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o660, 0o666])
def test_env_file_mode_600_is_enforced(tmp_path: Path, mode: int) -> None:
    env_file = write(tmp_path / ".env", f"TYPESAFE_API_KEY={TS_KEY}\n", mode)
    with pytest.raises(ConfigError, match="must be 600") as err:
        config.secrets({}, env_file)
    assert TS_KEY not in str(err.value)
    env_file.chmod(0o400)
    assert config.secrets({}, env_file).typesafe_api_key == TS_KEY  # owner-only read is as tight as 600


def test_env_file_errors_never_echo_the_line(tmp_path: Path) -> None:
    env_file = write(tmp_path / ".env", f"GOOD=1\n{TS_KEY}\n")
    with pytest.raises(ConfigError, match="line 2 is not KEY=VALUE") as err:
        config.secrets({}, env_file)
    assert TS_KEY not in str(err.value)
    with pytest.raises(ConfigError, match="line 1 is not KEY=VALUE"):
        config.read_env_file(write(tmp_path / "bad.env", "1BAD=x\n"))
    with pytest.raises(ConfigError, match="not a regular file"):
        config.read_env_file(tmp_path)


def test_env_file_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_bytes(b"TYPESAFE_API_KEY=\xff\xfe\n")
    env_file.chmod(0o600)
    with pytest.raises(ConfigError, match="cannot read env file"):
        config.read_env_file(env_file)


def test_load_secrets_uses_the_configs_env_file(tmp_path: Path) -> None:
    env_file = write(tmp_path / "paper.env", f"ALPACA_PAPER_KEY={PK_KEY}\nALPACA_PAPER_SECRET={PK_SECRET}\n")
    cfg = load_config(None, [f"paths.env_file={env_file}"])
    got = config.load_secrets(cfg, {"TYPESAFE_API_KEY": TS_KEY})
    assert got == Secrets(typesafe_api_key=TS_KEY, alpaca_paper_key=PK_KEY, alpaca_paper_secret=PK_SECRET)
    # the default ".env" is taken from the repository root, never from the working directory
    assert config.resolve_path(load_config(None).paths.env_file) == REPO / ".env"


def test_secrets_never_show_in_repr_str_or_format() -> None:
    got = all_keys()
    for text in (repr(got), str(got), f"{got}", f"{got!r}", "%s" % (got,), repr([got]), repr({"s": got})):  # noqa: UP031
        assert TS_KEY not in text and PK_KEY not in text and PK_SECRET not in text
        assert "<set>" in text
    assert "<absent>" in repr(Secrets())
    rich_pairs = dict(got.__rich_repr__())
    assert rich_pairs["typesafe_api_key"] == "<set>" and TS_KEY not in str(rich_pairs)


def test_every_loaded_secret_is_registered_for_redaction() -> None:
    logsetup.clear_secrets()
    webhook = "https://hooks.example.invalid/T000/B000/s3cr3tpath"
    config.secrets(
        {
            "TYPESAFE_API_KEY": f" {TS_KEY} ",
            "ALPACA_PAPER_KEY": PK_KEY,
            "ALPACA_PAPER_SECRET": PK_SECRET,
            "JEVBOT_ALERT_WEBHOOK": webhook,
            "JEVBOT_DATA": "/srv/data",
        },
        None,
    )
    line = logsetup.redact(f"Authorization: Bearer {TS_KEY} key={PK_KEY} secret={PK_SECRET} hook={webhook} dir=/srv/data")
    assert line == "Authorization: Bearer [REDACTED] key=[REDACTED] secret=[REDACTED] hook=[REDACTED] dir=/srv/data"
    logsetup.clear_secrets()
    Secrets(typesafe_api_key="constructed-directly-9981")
    assert logsetup.redact("x constructed-directly-9981 y") == "x [REDACTED] y"


# ======================================================================================================================
# probe_status(): only the exact (model, SDK, question-set) key counts
# ======================================================================================================================

KEY = {"model": "jev-1.13.0", "sdk_version": "0.6.0", "entry_qset_hash": "e" * 64, "entry_text_qset_hash": "t" * 64}


def put_record(data_dir: Path, suite: str, **changes: str) -> Path:
    fields = {**KEY, **changes}
    record = ProbeRecord(
        suite=suite,
        verdict={"deterministic": True},
        run_dir="probes/step0/20260917T150000",
        recorded_at=datetime(2026, 9, 17, 15, 0, tzinfo=UTC),
        **fields,
    )
    records = data_dir / "probes" / "step0" / "records"
    records.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:12]
    path = records / f"{suite}-{tag}.json"
    path.write_bytes(msgspec.json.encode(record))
    return path


def test_probe_status_without_records(tmp_path: Path) -> None:
    assert config.probe_status(tmp_path, **KEY) == ProbeStatus(meta=False, determinism=False, batch=False, order=False, text=False)
    assert ProbeStatus() == ProbeStatus(False, False, False, False, False)


def test_probe_status_counts_each_recorded_suite(tmp_path: Path) -> None:
    put_record(tmp_path, "determinism")
    put_record(tmp_path, "order")
    assert config.probe_status(tmp_path, **KEY) == ProbeStatus(determinism=True, order=True)
    for suite in ("meta", "batch", "text"):
        put_record(tmp_path, suite)
    assert config.probe_status(tmp_path, **KEY) == ALL_PROBES
    assert config.PROBE_SUITES == ("meta", "determinism", "batch", "order", "text")


@pytest.mark.parametrize(
    ("field", "other"),
    [
        ("model", "jev-1.14.0"),
        ("model", "jev-1.13.0 "),
        ("sdk_version", "0.6.1"),
        ("entry_qset_hash", "f" * 64),
        ("entry_text_qset_hash", "u" * 64),
    ],
)
def test_probe_record_for_another_key_never_counts(tmp_path: Path, field: str, other: str) -> None:
    for suite in config.PROBE_SUITES:
        put_record(tmp_path, suite, **{field: other})
    assert config.probe_status(tmp_path, **KEY) == ProbeStatus()
    assert config.probe_status(tmp_path, **{**KEY, field: other}) == ALL_PROBES
    put_record(tmp_path, "text")  # the exact key next to the foreign ones
    assert config.probe_status(tmp_path, **KEY) == ProbeStatus(text=True)


def test_malformed_probe_records_never_count(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    good = put_record(tmp_path, "batch")
    records = good.parent
    (records / "text-000000000000.json").write_text("{not json", encoding="utf-8")
    (records / "order-111111111111.json").write_text(json.dumps({"suite": "order", **KEY}), encoding="utf-8")  # fields missing
    (records / "notes.txt").write_text("ignored", encoding="utf-8")
    unknown = msgspec.json.decode(good.read_bytes())
    unknown["suite"] = "telepathy"
    (records / "telepathy-222222222222.json").write_text(json.dumps(unknown), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="jevbot.config"):
        status = config.probe_status(tmp_path, **KEY)
    assert status == ProbeStatus(batch=True)
    warned = " ".join(r.getMessage() for r in caplog.records)
    assert "text-000000000000.json" in warned and "order-111111111111.json" in warned and "telepathy-222222222222.json" in warned


# ======================================================================================================================
# resolve_(): decider, the news truth table (V12), the perturbation gate, forced paper values
# ======================================================================================================================


def cfg_for(mode: str, *overrides: str) -> Config:
    return load_config(None, [f"run.mode={mode}", *overrides])


@pytest.mark.parametrize(
    ("mode", "overrides", "secrets_", "expected"),
    [
        ("backtest", (), no_keys, "mock"),  # D7: no key => MockJev
        ("paper", (), no_keys, "mock"),
        ("paper", (), alpaca_keys, "mock"),
        ("paper", (), jev_key, "live"),
        ("backtest", (), jev_key, "replay"),  # jev.cache.mode defaults to replay (G12)
        ("backtest", ("jev.cache.mode=record",), jev_key, "live"),
        ("backtest", ("jev.cache.mode=refresh",), jev_key, "live"),
        ("backtest", ("decider.kind=mock",), jev_key, "mock"),
        ("backtest", ("decider.kind=replay",), no_keys, "replay"),  # replay needs no key: no network client is constructed
        ("backtest", ("decider.kind=live",), jev_key, "live"),
        ("paper", ("decider.kind=live",), jev_key, "live"),
        ("paper", ("decider.kind=mock",), jev_key, "mock"),
    ],
)
def test_decider_resolution(mode: str, overrides: tuple[str, ...], secrets_: Any, expected: str) -> None:
    rc = resolve_(cfg_for(mode, *overrides), secrets_(), ALL_PROBES)
    assert rc.cfg.decider.kind == expected
    assert rc.cfg.decider.mock_profile == "full"


def test_decider_refused_combinations() -> None:
    with pytest.raises(ConfigError, match="needs TYPESAFE_API_KEY"):
        resolve_(cfg_for("backtest", "decider.kind=live"), no_keys(), ALL_PROBES)
    with pytest.raises(ConfigError, match="needs TYPESAFE_API_KEY"):
        resolve_(cfg_for("paper", "decider.kind=live"), Secrets(typesafe_api_key="   "), ALL_PROBES)
    with pytest.raises(ConfigError, match="refused in paper mode"):
        resolve_(cfg_for("paper", "decider.kind=replay"), jev_key(), ALL_PROBES)
    with pytest.raises(ConfigError, match="reference"):  # the Jev-free reference run must not silently become a Jev run
        resolve_(cfg_for("backtest", "run.purpose=reference"), jev_key(), ALL_PROBES)
    assert resolve_(cfg_for("backtest", "run.purpose=reference"), no_keys(), ProbeStatus()).cfg.decider.kind == "mock"
    assert resolve_(cfg_for("backtest", "run.purpose=reference", "decider.kind=mock"), jev_key(), ProbeStatus()).cfg.decider.kind == "mock"


NO_TEXT = ProbeStatus(meta=True, determinism=True, batch=True, order=True, text=False)
TEXT_ONLY = ProbeStatus(text=True)

# (news.enabled, mode, secrets, probes) -> (news_resolved, news_reason); "live Jev" = a TypeSafe key in paper mode
NEWS_TRUTH_TABLE = [
    # explicit off: always off
    ("off", "backtest", no_keys, ProbeStatus(), False, "explicit_off"),
    ("off", "paper", all_keys, ALL_PROBES, False, "explicit_off"),
    ("off", "paper", all_keys, NO_TEXT, False, "explicit_off"),
    # explicit on: on, except paper + live Jev + no text record (a ConfigError, tested below)
    ("on", "backtest", no_keys, ProbeStatus(), True, "explicit_on"),
    ("on", "backtest", all_keys, NO_TEXT, True, "explicit_on"),
    ("on", "paper", alpaca_keys, ProbeStatus(), True, "explicit_on"),  # MockJev cannot read text
    ("on", "paper", all_keys, TEXT_ONLY, True, "explicit_on"),
    # auto without Alpaca keys: off
    ("auto", "backtest", no_keys, ALL_PROBES, False, "auto_no_keys"),
    ("auto", "backtest", jev_key, ALL_PROBES, False, "auto_no_keys"),
    ("auto", "paper", jev_key, ALL_PROBES, False, "auto_no_keys"),
    # auto with keys: on in backtests, on with MockJev, on with live Jev once the text probe is recorded
    ("auto", "backtest", alpaca_keys, ProbeStatus(), True, "auto_keys_present"),
    ("auto", "backtest", all_keys, NO_TEXT, True, "auto_keys_present"),
    ("auto", "paper", alpaca_keys, ProbeStatus(), True, "auto_keys_present"),
    ("auto", "paper", all_keys, TEXT_ONLY, True, "auto_keys_present"),
    # V12: paper + live Jev + no text-probe record => OFF, text_probe_pending
    ("auto", "paper", all_keys, NO_TEXT, False, "text_probe_pending"),
    ("auto", "paper", all_keys, ProbeStatus(), False, "text_probe_pending"),
]


@pytest.mark.parametrize(("enabled", "mode", "secrets_", "probes", "resolved", "reason"), NEWS_TRUTH_TABLE)
def test_news_truth_table(enabled: str, mode: str, secrets_: Any, probes: ProbeStatus, resolved: bool, reason: str) -> None:
    rc = resolve_(cfg_for(mode, f"news.enabled={enabled}"), secrets_(), probes)
    assert (rc.news_resolved, rc.news_reason) == (resolved, reason)
    assert rc.cfg.news.enabled == ("on" if resolved else "off"), "the resolved config carries the concrete value"
    assert reason in config.NEWS_REASONS


def test_explicit_news_on_is_refused_for_paper_live_jev_without_text_probe() -> None:
    with pytest.raises(ConfigError, match=r"jevbot jev probe-step0 --suite text"):
        resolve_(cfg_for("paper", "news.enabled=on"), all_keys(), NO_TEXT)
    # a MockJev paper service that has a key available but is pinned to mock is not "live Jev"
    rc = resolve_(cfg_for("paper", "news.enabled=on", "decider.kind=mock"), all_keys(), NO_TEXT)
    assert (rc.news_resolved, rc.news_reason) == (True, "explicit_on")
    rc = resolve_(cfg_for("paper", "news.enabled=auto", "decider.kind=mock"), all_keys(), NO_TEXT)
    assert (rc.news_resolved, rc.news_reason) == (True, "auto_keys_present")


@pytest.mark.parametrize(
    ("probes", "allowed"),
    [
        (ProbeStatus(), False),
        (ProbeStatus(determinism=True), False),
        (ProbeStatus(order=True), False),
        (ProbeStatus(meta=True, batch=True, text=True), False),
        (ProbeStatus(determinism=True, order=True), True),
        (ALL_PROBES, True),
    ],
)
def test_perturbation_scope_off_gate_in_paper_with_live_jev(probes: ProbeStatus, allowed: bool) -> None:
    cfg = cfg_for("paper", "rules.perturbation_scope_paper=off", "news.enabled=off")
    if allowed:
        assert resolve_(cfg, all_keys(), probes).cfg.rules.perturbation_scope_paper == "off"
    else:
        with pytest.raises(ConfigError, match="perturbation_scope_paper"):
            resolve_(cfg, all_keys(), probes)
    # the gate concerns paper + LIVE Jev only
    assert resolve_(cfg, alpaca_keys(), probes).cfg.decider.kind == "mock"
    assert (
        resolve_(cfg_for("backtest", "rules.perturbation_scope_paper=off", "jev.cache.mode=record"), all_keys(), probes).cfg.decider.kind
        == "live"
    )


def test_paper_forces_purpose_and_record_mode() -> None:
    rc = resolve_(cfg_for("paper", "run.purpose=validate", "jev.cache.mode=replay"), all_keys(), ALL_PROBES)
    assert rc.cfg.run.purpose == "paper"
    assert rc.cfg.jev.cache.mode is CacheMode.RECORD
    backtest = resolve_(cfg_for("backtest", "run.purpose=tune"), all_keys(), ALL_PROBES)
    assert backtest.cfg.run.purpose == "tune" and backtest.cfg.jev.cache.mode is CacheMode.REPLAY


def test_resolve_revalidates_a_programmatically_built_config() -> None:
    broken = msgspec.structs.replace(Config(), dte=msgspec.structs.replace(Config().dte, hard_exit_sessions=1))
    with pytest.raises(ConfigError, match="hard_exit_sessions"):
        resolve_(broken, no_keys(), ProbeStatus())


# ======================================================================================================================
# hashes
# ======================================================================================================================

HASHES = ("config_hash", "state_config_hash", "rules_hash", "risk_config_hash", "candidate_config_hash")


def hashes(*overrides: str, secrets_: Secrets | None = None, **kwargs: str) -> dict[str, str]:
    rc = resolve_(load_config(None, list(overrides)), secrets_ if secrets_ is not None else no_keys(), ProbeStatus(), **kwargs)
    return {name: getattr(rc, name) for name in HASHES}


def test_config_hash_is_stable_and_hex() -> None:
    first, second = hashes(), hashes()
    assert first == second
    for value in first.values():
        assert re.fullmatch(r"[0-9a-f]{64}", value)
    assert len(set(first.values())) == len(HASHES)
    assert resolve_(Config(), no_keys(), ProbeStatus()).config_hash == first["config_hash"], "struct defaults and default.toml hash alike"


def test_config_hash_is_resolved_never_auto() -> None:
    """`news auto` and `decider auto` are replaced by concrete values BEFORE hashing: the hash never depends silently on the
    environment, and two environments that resolve alike hash alike."""
    auto_no_keys = hashes()
    explicit = hashes("news.enabled=off", "decider.kind=mock")
    assert auto_no_keys == explicit
    auto_with_alpaca = hashes(secrets_=alpaca_keys())  # news resolves ON
    assert auto_with_alpaca == hashes("news.enabled=on", "decider.kind=mock")
    assert auto_with_alpaca["config_hash"] != auto_no_keys["config_hash"]
    assert auto_with_alpaca["state_config_hash"] != auto_no_keys["state_config_hash"], "news is part of the state scope"
    assert auto_with_alpaca["rules_hash"] == auto_no_keys["rules_hash"]
    with_jev_key = hashes(secrets_=jev_key())  # decider resolves to replay
    assert with_jev_key["config_hash"] != auto_no_keys["config_hash"]
    assert with_jev_key == hashes("decider.kind=replay")


def test_config_hash_excludes_paths_only() -> None:
    base = hashes()
    assert hashes("paths.data_dir=/somewhere/else", "paths.env_file=/etc/jevbot.env") == base
    assert hashes("run.seed=1")["config_hash"] != base["config_hash"]
    direct = config.config_hash(resolve_(Config(), no_keys(), ProbeStatus()).cfg)
    assert direct == base["config_hash"]


# override -> the sub-hashes it must change (config_hash always changes)
SCOPES = [
    ("rules.min_score=0.6", {"rules_hash"}),
    ("rules.weights.align=0.25", {"rules_hash"}),  # + calm below keeps the sum at 1.0
    ("rules.tiers.score=[[0.8, 1.0], [0.6, 0.5]]", {"rules_hash", "candidate_config_hash"}),
    ("risk.max_open_structures=5", {"risk_config_hash"}),
    ("risk.max_loss_per_trade_pct=0.02", {"risk_config_hash", "candidate_config_hash"}),
    ("kill.flatten_attempts=5", {"risk_config_hash"}),
    ("kill.actions.stale_quotes=halt", {"risk_config_hash"}),
    ("health.max_chain_age_s=60", {"risk_config_hash"}),
    ("exits.stop_loss_frac=0.4", {"risk_config_hash"}),
    ("dte.target=36", {"risk_config_hash", "candidate_config_hash"}),
    ("dte.hold_horizon_sessions=15", {"state_config_hash", "risk_config_hash", "candidate_config_hash"}),
    ("cadence.decide_offset_min=26", {"state_config_hash"}),  # the news recency cutoff (5.6)
    ("cadence.exec_offset_min=21", set()),
    ("data.min_history_sessions=100", {"state_config_hash"}),
    ("data.iv_rank_lo_pct=5", {"state_config_hash"}),
    ("data.iv_rank_hi_pct=95", {"state_config_hash"}),
    ("data.max_dte=150", set()),
    ('universe.kind.SPY="a fund"', {"state_config_hash"}),
    ("state.max_chars=11000", {"state_config_hash"}),
    ("state.render=bucket_only", {"state_config_hash"}),
    ("news.max_items=6", {"state_config_hash"}),
    ("candidates.long_delta=0.30", {"candidate_config_hash"}),
    ("liquidity.min_open_interest=50", {"candidate_config_hash"}),
    ("run.initial_equity_usd=50000", {"candidate_config_hash"}),
    ("run.seed=5", set()),
    ("fees.orf=0.02", set()),
    ("eval.bootstrap_reps=100", set()),
]


@pytest.mark.parametrize(("override", "changed"), SCOPES)
def test_sub_hash_scopes(override: str, changed: set[str]) -> None:
    extra = ["rules.weights.calm=0.20"] if override.startswith("rules.weights.align") else []
    base, got = hashes(), hashes(override, *extra)
    actually_changed = {name for name in HASHES if got[name] != base[name]}
    assert actually_changed == changed | {"config_hash"}


def test_state_config_hash_takes_mask_version_and_bucket_spec() -> None:
    """Section 4: the bucket tables (5.5) and the mask version (5.8) are in the state scope and in NO other sub-hash."""
    base = hashes()
    masked = hashes(mask_version="ba9876543210")  # another mask dictionary
    bucketed = hashes(bucket_spec_hash="e" * 64)  # another bucket table
    for other in (masked, bucketed):
        assert {name for name in HASHES if other[name] != base[name]} == {"state_config_hash"}, (
            "mask_version / bucket_spec_hash change state_config_hash only: config_hash, rules_hash, risk_config_hash and "
            "candidate_config_hash are functions of the config alone"
        )
    assert masked["state_config_hash"] != bucketed["state_config_hash"]
    cfg = resolve_(Config(), no_keys(), ProbeStatus()).cfg
    assert config.state_config_hash(cfg, mask_version="ba9876543210", bucket_spec_hash=BUCKET_SPEC_HASH) == masked["state_config_hash"]
    assert config.state_config_hash(cfg, mask_version=MASK_VERSION, bucket_spec_hash="e" * 64) == bucketed["state_config_hash"]
    assert config.state_config_hash(cfg, mask_version=MASK_VERSION, bucket_spec_hash=BUCKET_SPEC_HASH) == base["state_config_hash"]
    # both values are hashed material: an independent re-computation of the state scope with them in place
    material = {
        "universe": msgspec.to_builtins(cfg.universe),
        "state": msgspec.to_builtins(cfg.state),
        "news": msgspec.to_builtins(cfg.news),
        "dte.hold_horizon_sessions": cfg.dte.hold_horizon_sessions,
        "cadence.decide_offset_min": cfg.cadence.decide_offset_min,
        "data.min_history_sessions": cfg.data.min_history_sessions,
        "data.iv_rank_lo_pct": cfg.data.iv_rank_lo_pct,
        "data.iv_rank_hi_pct": cfg.data.iv_rank_hi_pct,
        "bucket_spec_hash": BUCKET_SPEC_HASH,
        "mask_version": MASK_VERSION,
    }
    text = json.dumps(config._hash_material(material), ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == base["state_config_hash"]


def test_resolve_requires_mask_version_and_bucket_spec_hash() -> None:
    """The section-4 sketch `resolve(cfg, secrets, probes)` is refused: a caller can never get a state_config_hash that
    silently ignores the bucket tables or the mask dictionary (they would not change when a threshold or a term does)."""
    cfg = Config()
    with pytest.raises(TypeError):
        resolve(cfg, no_keys(), ProbeStatus())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        resolve(cfg, no_keys(), ProbeStatus(), mask_version=MASK_VERSION)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        resolve(cfg, no_keys(), ProbeStatus(), bucket_spec_hash=BUCKET_SPEC_HASH)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        resolve(cfg, no_keys(), ProbeStatus(), MASK_VERSION, BUCKET_SPEC_HASH)  # type: ignore[call-arg]  # keyword-only
    for name in ("mask_version", "bucket_spec_hash"):
        param = inspect.signature(resolve).parameters[name]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is inspect.Parameter.empty
        param = inspect.signature(config.state_config_hash).parameters[name]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is inspect.Parameter.empty
    # blank / padded / non-string values are a ConfigError in both functions (before any other resolution work)
    for bad in ("", "   ", " 0123456789ab", "0123456789ab\n", None, 12, b"0123456789ab"):
        with pytest.raises(ConfigError, match="mask_version"):
            resolve(cfg, no_keys(), ProbeStatus(), mask_version=bad, bucket_spec_hash=BUCKET_SPEC_HASH)  # type: ignore[arg-type]
        with pytest.raises(ConfigError, match="bucket_spec_hash"):
            resolve(cfg, no_keys(), ProbeStatus(), mask_version=MASK_VERSION, bucket_spec_hash=bad)  # type: ignore[arg-type]
        with pytest.raises(ConfigError, match="mask_version"):
            config.state_config_hash(cfg, mask_version=bad, bucket_spec_hash=BUCKET_SPEC_HASH)  # type: ignore[arg-type]
        with pytest.raises(ConfigError, match="bucket_spec_hash"):
            config.state_config_hash(cfg, mask_version=MASK_VERSION, bucket_spec_hash=bad)  # type: ignore[arg-type]
    # two resolves that differ ONLY in one of the two inputs: identical everywhere except state_config_hash
    a = resolve(cfg, no_keys(), ProbeStatus(), mask_version="aaaaaaaaaaaa", bucket_spec_hash=BUCKET_SPEC_HASH)
    b = resolve(cfg, no_keys(), ProbeStatus(), mask_version="bbbbbbbbbbbb", bucket_spec_hash=BUCKET_SPEC_HASH)
    c = resolve(cfg, no_keys(), ProbeStatus(), mask_version="aaaaaaaaaaaa", bucket_spec_hash="c" * 64)
    assert a.cfg == b.cfg == c.cfg and (a.news_resolved, a.news_reason) == (b.news_resolved, b.news_reason) == (
        c.news_resolved,
        c.news_reason,
    )
    assert a.state_config_hash != b.state_config_hash != c.state_config_hash != a.state_config_hash
    for name in ("config_hash", "rules_hash", "risk_config_hash", "candidate_config_hash"):
        assert getattr(a, name) == getattr(b, name) == getattr(c, name)
    # and the same inputs always give the same hash (a restart / another directory reproduces RUN_START's state_config_hash)
    assert resolve(cfg, no_keys(), ProbeStatus(), mask_version="aaaaaaaaaaaa", bucket_spec_hash=BUCKET_SPEC_HASH) == a


def test_resolve_takes_the_version_of_a_loaded_mask_terms_file(tmp_path: Path) -> None:
    terms = config.load_mask_terms(write(tmp_path / "mask_terms.toml", MASK_TABLE_FORM))
    rc = resolve(Config(), no_keys(), ProbeStatus(), mask_version=terms.version, bucket_spec_hash=BUCKET_SPEC_HASH)
    assert rc.state_config_hash == config.state_config_hash(rc.cfg, mask_version=terms.version, bucket_spec_hash=BUCKET_SPEC_HASH)
    other = config.load_mask_terms(write(tmp_path / "other.toml", MASK_TABLE_FORM.replace("ECB", "BoE")))
    assert other.version != terms.version
    assert (
        resolve(Config(), no_keys(), ProbeStatus(), mask_version=other.version, bucket_spec_hash=BUCKET_SPEC_HASH).state_config_hash
        != rc.state_config_hash
    )


def test_hash_formula_is_sorted_canonical_json_without_floats() -> None:
    """Independent re-computation: sha256 over sorted-key compact JSON, floats as their repr strings."""
    cfg = resolve_(Config(), no_keys(), ProbeStatus()).cfg

    def strings_for_floats(obj: Any) -> Any:
        if isinstance(obj, float):
            return repr(obj)
        if isinstance(obj, dict):
            return {k: strings_for_floats(v) for k, v in obj.items()}
        if isinstance(obj, list | tuple):
            return [strings_for_floats(v) for v in obj]
        return obj

    def contains_float(obj: Any) -> bool:
        if isinstance(obj, float):
            return True
        if isinstance(obj, dict):
            return any(contains_float(v) for v in obj.values())
        return isinstance(obj, list | tuple) and any(contains_float(v) for v in obj)

    material = {"rules": strings_for_floats(msgspec.to_builtins(cfg.rules))}
    assert material["rules"]["min_score"] == "0.555" and material["rules"]["tiers"]["score"][0] == ["0.755", "1.0"]
    text = json.dumps(material, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == config.rules_hash(cfg)

    candidate = {
        "candidates": strings_for_floats(msgspec.to_builtins(cfg.candidates)),
        "liquidity": strings_for_floats(msgspec.to_builtins(cfg.liquidity)),
        "dte": msgspec.to_builtins(cfg.dte),
        "risk.max_loss_per_trade_pct": "0.01",
        "rules.tiers": strings_for_floats(msgspec.to_builtins(cfg.rules.tiers)),
        "run.initial_equity_usd": 100000,
    }
    text = json.dumps(candidate, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == config.candidate_config_hash(cfg)

    whole = config._hash_material(msgspec.to_builtins(cfg))
    assert not contains_float(whole), "nothing hashed may be a float (Conventions)"
    assert whole["run"]["start"] == "2012-01-03" and whole["jev"]["timeout_s"] == "8.0"


def test_hash_formula_is_canon_dumps_sorted() -> None:
    """Section 4: `config_hash = sha256(dumps_sorted(...))`. `config.py` repeats the literal 3.7 formula instead of importing
    `canon`; this pins the literal, and - as soon as `jevbot.canon` exists in the tree - proves the two agree byte for byte."""
    cfg = resolve_(Config(), no_keys(), ProbeStatus()).cfg
    builtins = msgspec.to_builtins(cfg)
    builtins.pop("paths")
    material = config._hash_material(builtins)
    literal = json.dumps(material, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    assert hashlib.sha256(literal.encode("utf-8")).hexdigest() == config.config_hash(cfg)
    if importlib.util.find_spec("jevbot.canon") is not None:
        canon = importlib.import_module("jevbot.canon")
        assert canon.dumps_sorted(material) == literal
        assert canon.sha256_hex(canon.dumps_sorted(material)) == config.config_hash(cfg)


# ======================================================================================================================
# load_mask_terms (the real config/mask_terms.toml belongs to WP02: a temporary file here)
# ======================================================================================================================

MASK_TABLE_FORM = """\
[funds]
terms = ["SPDR S&P 500 ETF Trust", "SPY", "Invesco QQQ"]

[central_banks]
replacement = "the monetary authority"
terms = ["Federal Reserve", "FOMC", "ECB"]
"""

MASK_ARRAY_FORM = """\
indices = ["S&P 500", "Nasdaq 100", "Russell 2000"]
agencies = ["BLS", "BEA", " Treasury "]
"""


def test_load_mask_terms_table_form(tmp_path: Path) -> None:
    path = write(tmp_path / "mask_terms.toml", MASK_TABLE_FORM)
    terms = config.load_mask_terms(path)
    assert isinstance(terms, MaskTerms)
    assert terms.groups["funds"] == ("SPDR S&P 500 ETF Trust", "SPY", "Invesco QQQ")
    assert terms.groups["central_banks"] == ("Federal Reserve", "FOMC", "ECB")
    assert set(terms.groups) == set(config.MASK_GROUPS) == set(terms.replacements)
    assert terms.groups["people"] == () and terms.groups["geo_events"] == ()
    assert terms.replacements["central_banks"] == "the monetary authority"  # the file may override a phrase
    assert terms.replacements["funds"] == "the fund"
    assert terms.replacements["people"] == "a senior official"  # generic patterns need the phrase even without a dictionary
    # mask_version = sha256(RULES_VERSION + file bytes)[:12]
    expected = hashlib.sha256(config.MASK_RULES_VERSION.encode("utf-8") + MASK_TABLE_FORM.encode("utf-8")).hexdigest()[:12]
    assert terms.version == expected and re.fullmatch(r"[0-9a-f]{12}", terms.version)


def test_load_mask_terms_array_form_and_version_changes_with_bytes(tmp_path: Path) -> None:
    path = write(tmp_path / "mask_terms.toml", MASK_ARRAY_FORM)
    terms = config.load_mask_terms(path)
    assert terms.groups["indices"] == ("S&P 500", "Nasdaq 100", "Russell 2000")
    assert terms.groups["agencies"] == ("BLS", "BEA", "Treasury")
    assert terms.replacements["indices"] == "a major equity index"
    again = config.load_mask_terms(path)
    assert again == terms
    changed = config.load_mask_terms(write(tmp_path / "other.toml", MASK_ARRAY_FORM + "# a comment changes the bytes\n"))
    assert changed.groups == terms.groups and changed.version != terms.version


def test_mask_replacements_are_the_design_phrases() -> None:
    assert dict(config.MASK_REPLACEMENTS) == {
        "funds": "the fund",
        "indices": "a major equity index",
        "central_banks": "the central bank",
        "agencies": "a government agency",
        "releases": "a major economic data release",
        "people": "a senior official",
        "companies": "a large company",
        "geo_events": "a major event",
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('[planets]\nterms = ["Mars"]\n', "unknown mask group"),
        ('moons = ["Io"]\n', "unknown mask group"),
        ("[funds]\nterms = []\n", "empty terms"),
        ("funds = []\n", "empty terms"),
        ('[funds]\nreplacement = "the fund"\n', "empty terms"),
        ('[funds]\nterms = ["SPY", ""]\n', "empty or non-string term"),
        ('[funds]\nterms = ["SPY", "  "]\n', "empty or non-string term"),
        ('[funds]\nterms = ["SPY", 7]\n', "empty or non-string term"),
        ('[funds]\nterms = "SPY"\n', "empty terms"),
        ('[funds]\nterms = ["SPY"]\nreplacment = "x"\n', "unknown key"),
        ('[funds]\nterms = ["SPY"]\nreplacement = " "\n', "replacement must be a non-blank string"),
        ('[funds]\nterms = ["SPY", "spy"]\n', "listed twice"),
        ('[funds]\nterms = ["Fed"]\n[central_banks]\nterms = ["FED"]\n', "listed twice"),
        ("", "no mask group found"),
        ("# only a comment\n", "no mask group found"),
        ("[funds\nterms = ", "invalid TOML"),
    ],
)
def test_load_mask_terms_refuses(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        config.load_mask_terms(write(tmp_path / "mask_terms.toml", text))


def test_load_mask_terms_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read mask terms file"):
        config.load_mask_terms(tmp_path / "absent.toml")


# ======================================================================================================================
# paths and small derived facts
# ======================================================================================================================


def test_data_dir_resolution_order() -> None:
    cfg = load_config(None)
    assert config.data_dir(cfg, Secrets()) == Path("/home/tyler/jevbot-data")  # D1 default
    assert config.data_dir(cfg, Secrets(data_dir="/srv/env-data")) == Path("/srv/env-data")
    explicit = load_config(None, ["paths.data_dir=/srv/cli-data"])
    assert config.data_dir(explicit, Secrets(data_dir="/srv/env-data")) == Path("/srv/cli-data")


def test_ensure_data_dir(tmp_path: Path) -> None:
    created = config.ensure_data_dir(tmp_path / "data" / "jevbot")
    assert created.is_dir() and stat.S_IMODE(created.stat().st_mode) == 0o700
    assert config.ensure_data_dir(created) == created
    created.chmod(0o750)
    with pytest.raises(ConfigError, match="must be 700"):
        config.ensure_data_dir(created)
    with pytest.raises(ConfigError, match="inside the git repository"):
        config.ensure_data_dir(REPO / "data")
    with pytest.raises(ConfigError, match="inside the git repository"):
        config.ensure_data_dir(REPO)
    with pytest.raises(ConfigError, match="does not exist"):
        config.ensure_data_dir(tmp_path / "missing", create=False)
    a_file = write(tmp_path / "file", "x")
    with pytest.raises(ConfigError, match="not a directory"):
        config.ensure_data_dir(a_file)
    assert not (REPO / "data").exists()


def test_resolve_path_is_repo_relative() -> None:
    assert config.REPO_ROOT == REPO
    assert config.DEFAULT_CONFIG_PATH == DEFAULT_TOML
    assert config.resolve_path(".env") == REPO / ".env"
    assert config.resolve_path(load_config(None).news.mask_terms_file) == REPO / "config" / "mask_terms.toml"
    assert config.resolve_path("/etc/jevbot/.env") == Path("/etc/jevbot/.env")


def test_headline_band_and_family() -> None:
    assert config.headline_band(load_config(None)) is Band.ORATS
    assert config.headline_band(load_config(None, ["cadence.fill_rule=same_snapshot_worst"])) is Band.WORST
    assert load_config(None).run.effective_family == "exp001"
    assert load_config(None, ["run.family=sweep-a"]).run.effective_family == "sweep-a"


def test_config_is_frozen() -> None:
    cfg = load_config(None)
    with pytest.raises(AttributeError):
        cfg.risk.max_open_structures = 99  # type: ignore[misc]
    with pytest.raises(AttributeError):
        cfg.run = cfg.run  # type: ignore[misc]
    assert Config().universe.alias is not Config().universe.alias, "mutable defaults are never shared between instances"
