"""Config structs, `load_config()`, `secrets()`, run-start resolution and resolved-config hashing (DESIGN.md section 4).

- `config/default.toml` holds every key with its default; the struct defaults below are the same values (a test keeps the two
  equal), so `Config()`, `RiskConfig()` ... are usable without any file.
- Merge order: `default.toml` <- `--config FILE` (repeatable) <- CLI `-o section.key=value`. Tables merge key by key, every other
  value replaces. Unknown keys are a `ConfigError` (`forbid_unknown_fields`). There are NO environment-variable config overrides.
- In paper mode `-o` is refused for every section of `PROTECTED_SECTIONS_PAPER`: the service's limits come from files under
  version control only.
- Secrets are never in TOML. `secrets()` reads the four constant environment names (plus a git-ignored, mode-600 `.env`), refuses
  ambiguous credentials (INV-02) and an SDK log level below WARNING (INV-18) and registers every value with the log redaction layer.
- `resolve(cfg, secrets, probes, *, mask_version, bucket_spec_hash)` fixes `decider.kind` and `news.enabled` ONCE at run start
  (V12) and computes `config_hash` plus the four sub-hashes. The two keyword arguments are REQUIRED (section 4 puts the bucket
  tables and the mask version in `state_config_hash`; this module does no IO, so the caller passes `MaskTerms.version` and
  `jevbot.buckets.bucket_spec_hash`); the section-4 sketch `resolve(cfg, secrets, probes)` is therefore a `TypeError` here on
  purpose - a hash that silently ignored those two inputs would defeat the sub-hash.

Hashing: `sha256(dumps_sorted(material))` with the literal `canon.dumps_sorted` formula of section 3.7
(`json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)`). Nothing hashed may be a float
(Conventions), so every config float enters the material as its shortest round-trip `repr()` string. The formula is repeated here
(as `types.py` does for its identity strings) so that this module depends on nothing but `errors`, `types` and `logsetup`.

This module never imports `typesafe_sdk` or `alpaca`: `check_sdk_log_level` has to run BEFORE the SDK is imported anywhere.
"""

import hashlib
import json
import logging
import math
import re
import stat
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from datetime import date
from enum import Enum
from itertools import pairwise
from pathlib import Path
from typing import Any, Final, Literal, get_args, get_origin, get_type_hints

import msgspec

from jevbot import logsetup
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

__all__ = [
    "ALLOWED_SDK_LOG_LEVELS",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_DATA_DIR",
    "ENV_ALPACA_PAPER_KEY",
    "ENV_ALPACA_PAPER_SECRET",
    "ENV_JEVBOT_DATA",
    "ENV_TYPESAFE_API_KEY",
    "ENV_TYPESAFE_LOG_LEVEL",
    "FORBIDDEN_ALPACA_ENV",
    "FORBIDDEN_TYPESAFE_ENV",
    "MASK_GROUPS",
    "MASK_REPLACEMENTS",
    "MASK_RULES_VERSION",
    "NEWS_REASONS",
    "PROBE_SUITES",
    "PROTECTED_SECTIONS_PAPER",
    "REPO_ROOT",
    "CadenceConfig",
    "CandidatesConfig",
    "Config",
    "DataConfig",
    "DataFomcConfig",
    "DataSyntheticConfig",
    "DeciderConfig",
    "DteConfig",
    "EvalConfig",
    "EvidenceConfig",
    "ExitsConfig",
    "FeesConfig",
    "FillsConfig",
    "FomcException",
    "GateConfig",
    "HealthConfig",
    "JevCacheConfig",
    "JevConfig",
    "JevSpendConfig",
    "KillActions",
    "KillConfig",
    "LiquidityConfig",
    "NewsConfig",
    "OrdersConfig",
    "PaperConfig",
    "PathsConfig",
    "ProbeStatus",
    "RecorderConfig",
    "ResolvedConfig",
    "RiskConfig",
    "RulesConfig",
    "RulesGates",
    "RulesTiers",
    "RulesWeights",
    "RunConfig",
    "Secrets",
    "StateConfig",
    "StructuresConfig",
    "UniverseConfig",
    "candidate_config_hash",
    "check_sdk_log_level",
    "config_hash",
    "data_dir",
    "dumps_toml",
    "ensure_data_dir",
    "headline_band",
    "kill_disabled",
    "load_config",
    "load_mask_terms",
    "load_secrets",
    "probe_status",
    "read_env_file",
    "resolve",
    "resolve_path",
    "risk_config_hash",
    "rules_hash",
    "secrets",
    "state_config_hash",
    "validate",
]

_log = logging.getLogger(__name__)

# ======================================================================================================================
# Constants
# ======================================================================================================================

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Final[Path] = REPO_ROOT / "config" / "default.toml"
DEFAULT_DATA_DIR: Final[str] = "/home/tyler/jevbot-data"  # D1

# The single definition that 0.4 and the CLI (section 14) refer to.
PROTECTED_SECTIONS_PAPER: Final[tuple[str, ...]] = ("risk", "kill", "health", "orders", "dte", "exits")

# Environment names are constants, not configurable (D2, D29). Never in TOML, never logged.
ENV_JEVBOT_DATA: Final = "JEVBOT_DATA"
ENV_TYPESAFE_API_KEY: Final = "TYPESAFE_API_KEY"
ENV_ALPACA_PAPER_KEY: Final = "ALPACA_PAPER_KEY"
ENV_ALPACA_PAPER_SECRET: Final = "ALPACA_PAPER_SECRET"
ENV_TYPESAFE_LOG_LEVEL: Final = "TYPESAFE_LOG_LEVEL"

# INV-02: ambiguous credentials. Any of these names - and any other APCA_* name (11.1) - is a PaperGuardError.
FORBIDDEN_ALPACA_ENV: Final[tuple[str, ...]] = (
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "APCA_API_BASE_URL",
)
_FORBIDDEN_ALPACA_PREFIX: Final = "APCA_"
# the pinned model and the default URL are passed explicitly: these SDK environment knobs are a ConfigError
FORBIDDEN_TYPESAFE_ENV: Final[tuple[str, ...]] = ("TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL")
# INV-18: the only NON-EMPTY normalised values of TYPESAFE_LOG_LEVEL that pass
ALLOWED_SDK_LOG_LEVELS: Final[frozenset[str]] = frozenset({"warn", "warning", "error", "off"})
# values that are redacted from logs when present but are not carried in `Secrets` (the Windows watchdog's webhook, 11.9)
_REDACT_ONLY_ENV: Final[tuple[str, ...]] = ("JEVBOT_ALERT_WEBHOOK",)

PROBE_SUITES: Final[tuple[str, ...]] = ("meta", "determinism", "batch", "order", "text")  # 6.8
NEWS_REASONS: Final[tuple[str, ...]] = ("explicit_on", "explicit_off", "auto_keys_present", "auto_no_keys", "text_probe_pending")

# 5.8: `mask_version = sha256(RULES_VERSION + mask_terms file bytes)[:12]`. This IS that RULES_VERSION: `load_mask_terms` lives
# here (WP00), so the constant does too; `textmask.py` imports it and bumps it whenever a sanitiser / masking rule changes.
MASK_RULES_VERSION: Final = "textmask.v1"
MASK_GROUPS: Final[tuple[str, ...]] = ("funds", "indices", "central_banks", "agencies", "releases", "people", "companies", "geo_events")
MASK_REPLACEMENTS: Final[Mapping[str, str]] = {
    "funds": "the fund",
    "indices": "a major equity index",
    "central_banks": "the central bank",
    "agencies": "a government agency",
    "releases": "a major economic data release",
    "people": "a senior official",
    "companies": "a large company",
    "geo_events": "a major event",
}

Purpose = Literal["validate", "tune", "diagnostic", "final", "reference", "paper", "shadow"]
DeciderKind = Literal["auto", "live", "replay", "mock"]
MockProfile = Literal["full", "trend_ivrank"]
NewsEnabled = Literal["auto", "on", "off"]
RecordedMode = Literal["dec_exec", "eod_eod"]
PerturbationScope = Literal["all", "passing", "off"]
ManageUseJev = Literal["on", "off", "cached_only"]
CondorBpMode = Literal["sum_wings", "max_wing"]
EquityBasis = Literal["min_broker_book"]
BacktestKillBehaviour = Literal["flatten_and_cooldown", "stop_run"]
StateRender = Literal["value_and_bucket", "bucket_only"]
DataProvider = Literal["mirror", "synthetic", "recorded"]
SpotMeasure = Literal["parity", "file_close", "auto"]
SyntheticMode = Literal["offline", "d20"]

# triggers that own a persistence threshold in [health]; `halt_then_kill` is meaningful for these only (V2, 9.5)
_ESCALATING_TRIGGERS: Final[frozenset[KillTrigger]] = frozenset(
    {KillTrigger.CLOCK_SKEW, KillTrigger.STALE_QUOTES, KillTrigger.JEV_ERRORS, KillTrigger.BROKER_ERRORS}
)

# ======================================================================================================================
# Config structs - one per TOML table, field order and defaults exactly as in section 4
# ======================================================================================================================


class RunConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    experiment: str = "exp001"  # experiment label; part of the namespace (D12)
    family: str = ""  # trial family for the Deflated Sharpe count (12.6); "" => run.experiment; CLI --family overrides it
    mode: RunMode = RunMode.BACKTEST  # "backtest" | "paper" (there is no other value; D2)
    purpose: Purpose = "validate"  # paper forces "paper" (resolve()); "shadow" is set by the shadow replay only
    seed: int = 20260917  # seeds every RNG
    ledger_forecasts: bool = True  # false ONLY for baseline 4 seeds and shadow stores
    initial_equity_usd: int = 100000  # backtest starting equity; paper reads it from the broker on first start
    start: date = date(2012, 1, 3)  # backtest window (inclusive)
    end: date = date(2025, 12, 12)

    @property
    def effective_family(self) -> str:
        """`run.family`, or the experiment label when it is empty (12.6)."""
        return self.family or self.experiment


class PathsConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    data_dir: str = ""  # "" => $JEVBOT_DATA => DEFAULT_DATA_DIR (D1). Outside the git repo (checked), mode 700
    env_file: str = ".env"  # git-ignored; KEY=VALUE lines; never overrides variables already set; mode 600 enforced


class UniverseConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):  # D3 (user-tunable)
    underlyings: tuple[str, ...] = ("SPY", "QQQ", "IWM")
    penny_all: tuple[str, ...] = ("SPY", "QQQ", "IWM")  # classes quoted in $0.01 at any price (B4.2)
    alias: dict[str, str] = msgspec.field(default_factory=lambda: {"SPY": "UNDERLYING_A", "QQQ": "UNDERLYING_B", "IWM": "UNDERLYING_C"})
    kind: dict[str, str] = msgspec.field(
        default_factory=lambda: {
            "SPY": "broad US large-cap equity index ETF",
            "QQQ": "US large-cap growth and technology equity index ETF",
            "IWM": "US small-cap equity index ETF",
        }
    )
    # Cboe index used ONLY to back-fill holes in the own-IV history (5.4)
    iv_proxy: dict[str, str] = msgspec.field(default_factory=lambda: {"SPY": "VIX", "QQQ": "VXN", "IWM": "RVX"})


class StructuresConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    enabled: tuple[StructureKind, ...] = (
        StructureKind.LONG_CALL,
        StructureKind.LONG_PUT,
        StructureKind.CALL_DEBIT,
        StructureKind.PUT_DEBIT,
        StructureKind.CALL_CREDIT,
        StructureKind.PUT_CREDIT,
        StructureKind.IRON_CONDOR,
    )


class CadenceConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # D4: offsets from that day's calendar close, never clock times (INV-13)
    decide_offset_min: int = 25  # paper: take the `dec` snapshot and start the cycle this many minutes before the close
    exec_offset_min: int = 20  # paper: take the `exec` snapshot and open the order ladder
    order_cutoff_offset_min: int = 5  # no submissions after close - 5 min (G4)
    cancel_all_offset_min: int = 4  # every resting order is cancelled at close - 4 min
    decision_deadline_offset_min: int = 17  # entries not decided by close - 17 min are dropped (fail closed)
    eod_offset_min: int = -2  # negative = minutes AFTER the close: `eod` snapshot for marks
    post_close_offset_min: int = -10  # final reconcile, SESSION_END, shadow replay
    morning_reconcile_after_open_min: int = 10
    min_session_minutes: int = 120  # skip entries on a session shorter than this (defensive)
    fill_rule: FillRule = FillRule.NEXT_SNAPSHOT  # backtest headline (decide D, fill D+1) | same_snapshot_worst (sensitivity)
    recorded_mode: RecordedMode = "dec_exec"  # replay of recorded days


class DteConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):  # D5
    target: int = 35  # calendar days
    min_entry: int = 28
    max_entry: int = 45
    time_exit_short_premium: int = 7  # close when calendar DTE <= this (credit verticals, condor)
    time_exit_long_premium: int = 7  # close when calendar DTE <= this (long options, debit verticals)
    hard_exit_sessions: int = 3  # mandatory close when sessions_to_expiry <= this (INV-11), counted to L = last_session
    min_sessions_beyond_hard_exit: int = 5  # never open an expiry with sessions_to_expiry <= hard_exit_sessions + this
    hold_horizon_sessions: int = 20  # holding window shown to Jev; horizon of the *_hold eval questions


class CandidatesConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):  # section 8
    long_delta: float = 0.35
    long_min_delta: float = 0.15
    debit_long_delta: float = 0.50
    debit_short_delta: float = 0.25
    credit_short_delta: float = 0.25
    credit_long_delta: float = 0.12
    condor_short_delta: float = 0.16
    condor_long_delta: float = 0.07
    delta_tolerance: float = 0.08
    max_unsizeable_rate: float = 0.25
    credit_short_min_em: float = 0.80
    min_width_strikes: int = 1
    max_width_pct_spot: float = 0.03
    min_credit_to_width: float = 0.12
    max_credit_to_width: float = 0.50
    condor_min_credit_to_width: float = 0.17
    condor_max_credit_to_width: float = 0.60
    max_debit_to_width: float = 0.60
    min_two_sided_frac: float = 0.60


class LiquidityConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # B7.4; applied per leg at candidate time, at approve() time, and by the fill model. NO same-day volume.
    min_bid_cents_sold: int = 10
    min_bid_cents_bought: int = 1
    max_rel_spread: float = 0.15  # (ask-bid)/mid, legs with mid >= 50 cents
    max_abs_spread_cents: int = 10  # legs with mid < 50 cents
    min_open_interest: int = 100  # on oi_prev (PIT-safe)
    allow_missing_open_interest: bool = True
    max_pct_displayed_size: float = 0.50
    max_pct_open_interest: float = 0.05


class ExitsConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):  # hard exits, code only (9.4)
    profit_target_frac: float = 0.50
    stop_loss_frac: float = 0.50
    ex_dividend_guard: bool = True
    ex_div_exit_sessions: int = 2
    assignment_extrinsic_floor_cents: int = 10
    assignment_sim_itm: float = 0.05
    assignment_sim_dte: int = 4


class RulesWeights(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # composite S_core (text-free; sum = 1.0)
    align: float = 0.30
    volfit: float = 0.20
    fit: float = 0.20
    regimefit: float = 0.15
    calm: float = 0.15


class GateConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    p_top: float
    margin: float


class RulesGates(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    direction: GateConfig = GateConfig(p_top=0.595, margin=0.245)  # under.direction (4 options)
    vol_stance: GateConfig = GateConfig(p_top=0.595, margin=0.245)  # vol.stance (4 options)
    structure: GateConfig = GateConfig(p_top=0.495, margin=0.195)  # fit.structure_family (8 options => lower bar)
    action: GateConfig = GateConfig(p_top=0.545, margin=0.195)  # pos.action (4 options) - the loosest bar


class RulesTiers(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    score: tuple[tuple[float, float], ...] = ((0.755, 1.0), (0.655, 0.75), (0.555, 0.5))  # S_core >= a => tier b; else 0
    peakedness: tuple[tuple[float, float], ...] = ((0.805, 1.0), (0.705, 0.75), (0.595, 0.5))  # min(p_top direction, p_top vol_stance)
    environment: tuple[float, float, float, float] = (1.0, 0.75, 0.5, 0.0)  # by risk.environment level 0..3


class RulesConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):  # section 7
    min_score: float = 0.555
    veto_hi: float = 0.705
    veto_lo: float = 0.295
    perturbation_variants: tuple[Variant, ...] = (Variant.OPT_PERM, Variant.KEY_PERM, Variant.BUCKET_ONLY)
    perturbation_scope_backtest: PerturbationScope = "all"
    perturbation_scope_paper: PerturbationScope = "passing"  # "off" is refused in paper until a Step 0 result is recorded
    reentry_cooldown_sessions: int = 3
    manage_use_jev: ManageUseJev = "on"
    exit_pressure_enter: float = 0.705  # hysteresis: latch exit at X >= this
    exit_pressure_release: float = 0.445  # hysteresis: release latch only at X < this
    text_watch_alert_sessions: int = 2  # raises an ALERT; it NEVER closes (INV-16)
    text_rank_weight: float = 0.05  # S_rank = (1 - w) * S_core + w * news_align   (ranking ONLY)
    code_crosschecks: bool = True  # 7.3; false ONLY for baseline 3 (always-enter), which flags the run
    weights: RulesWeights = msgspec.field(default_factory=RulesWeights)
    gates: RulesGates = msgspec.field(default_factory=RulesGates)
    tiers: RulesTiers = msgspec.field(default_factory=RulesTiers)


class RiskConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):  # D17 (all user-tunable)
    max_loss_per_trade_pct: float = 0.01
    max_aggregate_open_loss_pct: float = 0.10
    max_open_structures: int = 6
    max_new_per_day: int = 2
    one_per_underlying_direction: bool = True
    max_same_direction_structures: int = 3
    daily_loss_halt_pct: float = 0.02  # => no new entries for the rest of the day
    drawdown_kill_pct: float = 0.08  # peak-to-trough => kill switch
    max_bp_utilisation: float = 0.50  # reserved BP <= this * equity
    bp_haircut_mult: float = 1.00  # multiplier on the Cboe-minimum requirement
    condor_bp_mode: CondorBpMode = "sum_wings"  # conservative until probe P-ALP-5
    bp_drift_halt_pct: float = 0.10
    event_blackout_sessions: int = 1
    max_adverse_drift: float = 0.25
    max_contracts_per_trade: int = 10
    max_order_notional_usd: int = 15000
    max_orders_per_minute: int = 10
    max_order_attempts: int = 6
    equity_basis: EquityBasis = "min_broker_book"


class KillActions(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # deviation V2; every D17 trigger is present (one field per `KillTrigger` value)
    operator: TriggerAction = TriggerAction.KILL
    drawdown: TriggerAction = TriggerAction.KILL
    reconcile_mismatch: TriggerAction = TriggerAction.KILL
    model_mismatch: TriggerAction = TriggerAction.KILL
    expiry_violation: TriggerAction = TriggerAction.KILL
    assignment: TriggerAction = TriggerAction.KILL
    ledger_corrupt: TriggerAction = TriggerAction.KILL
    order_rate: TriggerAction = TriggerAction.KILL
    clock_skew: TriggerAction = TriggerAction.HALT_THEN_KILL
    stale_quotes: TriggerAction = TriggerAction.HALT_THEN_KILL
    jev_errors: TriggerAction = TriggerAction.HALT_THEN_KILL
    broker_errors: TriggerAction = TriggerAction.HALT_THEN_KILL

    def for_trigger(self, trigger: KillTrigger) -> TriggerAction:
        action: TriggerAction = getattr(self, trigger.value)
        return action


class KillConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):  # kill-sequence execution (9.5)
    cushion_ticks: int = 3
    cushion_max_frac_width: float = 0.15
    flatten_attempts: int = 4
    flatten_wait_s: int = 30
    not_flat_retry_s: int = 300
    cancel_wait_s: int = 20
    post_open_delay_min: int = 15
    post_open_delay_urgent_min: int = 2
    backtest_behaviour: BacktestKillBehaviour = "flatten_and_cooldown"
    backtest_cooldown_sessions: int = 20
    actions: KillActions = msgspec.field(default_factory=KillActions)


class HealthConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    max_clock_skew_ms: int = 5000  # D25
    clock_skew_kill_after_s: int = 900
    max_chain_age_s: int = 120
    max_quote_age_s: int = 1200  # PROVISIONAL until probe P-ALP-8
    max_stale_quote_frac: float = 0.20
    min_two_sided_frac: float = 0.60
    stale_kill_after_sessions: int = 3
    jev_error_kill_after_sessions: int = 3
    broker_error_halt_after: int = 2
    broker_error_kill_after: int = 6
    max_stale_mark_sessions: int = 3


class FeesConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # D14 / G11, USD per contract per side unless noted
    orf: float = 0.015
    occ: float = 0.025
    taf_sell: float = 0.00329
    cat: float = 0.0003
    sec_sell_rate: float = 0.0000206  # x sell notional
    commission: float = 0.0


class FillsConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):  # D13
    orats_p: tuple[float, float, float, float] = (0.75, 0.66, 0.56, 0.53)  # by number of legs 1,2,3,4+
    max_rel_spread: float = 0.25  # fill-time rejection (looser than the entry liquidity filter)
    max_abs_spread_cents: int = 15
    forced_penalty_frac_spread: float = 0.10
    forced_no_quote_pad_cents: int = 5


class OrdersConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):  # paper
    entry_ladder: tuple[float, ...] = (0.50, 0.75, 1.00)  # fraction of the way from mid to natural; never past natural
    exit_ladder: tuple[float, ...] = (0.50, 1.00)
    ladder_step_s: int = 45
    poll_s: int = 2
    http_timeout_s: tuple[float, float] = (3.05, 10.0)  # (connect, read) on every Alpaca call (D18)
    call_deadline_s: int = 15  # hard wall-clock deadline per broker call (worker thread)
    read_retries: int = 2
    ambiguous_lookups: int = 3
    repost_same_id: bool = False  # V4: enable ONLY after probe P-ALP-2 recorded that duplicates are rejected
    rest_calls_per_minute: int = 180


class JevCacheConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    mode: CacheMode = CacheMode.REPLAY  # backtests default to replay (G12); paper forces "record"


class JevSpendConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    max_input_tokens_per_run: int = 250000000  # batch scope only
    max_input_tokens_per_day: int = 300000000  # BATCH scope, per UTC day
    paper_max_input_tokens_per_day: int = 1000000  # PAPER scope ceiling; counted separately (INV-17)
    estimate_chars_per_token: float = 3.0
    estimate_overhead_tokens: int = 300


class JevConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):  # D6-D9
    model: str = "jev-1.13.0"
    model_release_date: date = date(2026, 9, 15)  # property of the namespace; drives evidence tiers
    sdk_version: str = "0.6.0"  # asserted against typesafe_sdk.__version__
    refresh_generation: int = 0  # bump => new namespace ("refresh")
    timeout_s: float = 8.0  # per HTTP operation
    retry_max: int = 2  # OUR retry loop; the SDK is always called with RetryPolicy(max_retries=0)
    retry_backoff_s: tuple[float, ...] = (0.5, 2.0)
    max_concurrency: int = 4
    max_rps: int = 10
    max_tokens_per_s: int = 100000
    cache: JevCacheConfig = msgspec.field(default_factory=JevCacheConfig)
    spend: JevSpendConfig = msgspec.field(default_factory=JevSpendConfig)


class DeciderConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    kind: DeciderKind = "auto"  # resolved ONCE by resolve(); recorded in RUN_START
    mock_profile: MockProfile = "full"  # "full" (baseline 7 / D11 ablation comparator) | "trend_ivrank" (baseline 5)


class StateConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    render: StateRender = "value_and_bucket"
    max_chars: int = 12000  # soft cap: news is trimmed to fit
    hard_max_chars: int = 16000  # StateTooLarge
    unmasked: bool = False  # true ONLY for the leakage diagnostic; can never trade


class NewsConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):  # D11
    enabled: NewsEnabled = "auto"  # resolved ONCE at run start by resolve() (V12), recorded with its reason, hashed
    lookback_hours: int = 72
    lag_s: int = 60  # knowable_at = created_at + lag_s (historical archive)
    max_items: int = 8
    max_headline_chars: int = 200
    max_summary_chars: int = 300
    max_total_chars: int = 4000
    max_symbols_per_item: int = 3
    mask: bool = True
    mask_terms_file: str = "config/mask_terms.toml"


class FomcException(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    year: int
    expected: int  # 7 is the only other accepted value
    reason: str
    source_url: str


class DataFomcConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    expected_per_year: int = 8
    exceptions: tuple[FomcException, ...] = ()  # shipped EMPTY (D23: nothing is assumed)


class DataSyntheticConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # SyntheticProvider (5.10); SYNTHETIC fidelity, tests and smoke runs ONLY
    mode: SyntheticMode = "offline"
    base_vol: float = 0.16
    vol_mean_revert: float = 0.03
    vol_of_vol: float = 0.06
    cross_corr: float = 0.85
    implied_premium: float = 0.10
    planted_drift_bp: int = 0  # 0 = NO skill. Tests set 12
    term_slope: float = 0.05
    spread_pct: float = 0.04
    strike_step_pct: float = 0.005
    quote_size: int = 100
    open_interest: int = 1000
    rate_bp: int = 400
    start_price_usd: dict[str, float] = msgspec.field(default_factory=lambda: {"SPY": 450.0, "QQQ": 380.0, "IWM": 200.0})


class DataConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    provider: DataProvider = "mirror"  # paper mode always uses alpaca live + recorder
    spot_measure: SpotMeasure = "auto"
    max_dte: int = 120
    moneyness_window: float = 0.35  # |ln(K / fwd)| kept in enriched chains
    cboe_indices: tuple[str, ...] = ("VIX", "VIX9D", "VIX3M", "VVIX", "SKEW", "VXN", "RVX")
    mirror_repo: str = "https://github.com/anahatsingh-ui/options-dataset-hist"
    min_history_sessions: int = 126
    iv_rank_lo_pct: int = 2
    iv_rank_hi_pct: int = 98
    atm_iv_tolerance: float = 0.10
    fomc_knowable_days: int = 45
    exdiv_knowable_days: int = 14
    fomc: DataFomcConfig = msgspec.field(default_factory=DataFomcConfig)
    synthetic: DataSyntheticConfig = msgspec.field(default_factory=DataSyntheticConfig)


class RecorderConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    strike_window_pct: float = 0.20  # record strikes within +-20% of spot
    max_dte: int = 75
    slots: tuple[Slot, ...] = (Slot.DEC, Slot.EXEC, Slot.EOD)


class EvidenceConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):  # D12
    tier_a_max_log_delay_s: int = 600
    prereg_file: str = "prereg/prereg.v1.toml"


class EvalConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    bootstrap_reps: int = 5000
    bootstrap_block: int = 0  # 0 => max(5, 2 * longest horizon in the statistic, ceil(n ** (1/3)))
    ece_bins: int = 10
    ece_min_per_bin: int = 20
    base_rate_min_events: int = 250
    recal_min_events: int = 250
    recal_refit_sessions: int = 21
    random_baseline_seeds: int = 1000
    placebo_min_distance_sessions: int = 60
    ci_level: float = 0.95
    null_sim_reps: int = 2000
    weekday_tail_ratio_max: float = 1.6


class PaperConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    heartbeat_s: int = 15
    deadman_stale_s: int = 600
    alert_cmd: str = ""  # optional command; receives a fixed short non-secret message as argv[1]
    shadow_replay: bool = True
    wind_down: bool = False  # model-change / shutdown mode (12.9)


class Config(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    run: RunConfig = msgspec.field(default_factory=RunConfig)
    paths: PathsConfig = msgspec.field(default_factory=PathsConfig)
    universe: UniverseConfig = msgspec.field(default_factory=UniverseConfig)
    structures: StructuresConfig = msgspec.field(default_factory=StructuresConfig)
    cadence: CadenceConfig = msgspec.field(default_factory=CadenceConfig)
    dte: DteConfig = msgspec.field(default_factory=DteConfig)
    candidates: CandidatesConfig = msgspec.field(default_factory=CandidatesConfig)
    liquidity: LiquidityConfig = msgspec.field(default_factory=LiquidityConfig)
    exits: ExitsConfig = msgspec.field(default_factory=ExitsConfig)
    rules: RulesConfig = msgspec.field(default_factory=RulesConfig)
    risk: RiskConfig = msgspec.field(default_factory=RiskConfig)
    kill: KillConfig = msgspec.field(default_factory=KillConfig)
    health: HealthConfig = msgspec.field(default_factory=HealthConfig)
    fees: FeesConfig = msgspec.field(default_factory=FeesConfig)
    fills: FillsConfig = msgspec.field(default_factory=FillsConfig)
    orders: OrdersConfig = msgspec.field(default_factory=OrdersConfig)
    jev: JevConfig = msgspec.field(default_factory=JevConfig)
    decider: DeciderConfig = msgspec.field(default_factory=DeciderConfig)
    state: StateConfig = msgspec.field(default_factory=StateConfig)
    news: NewsConfig = msgspec.field(default_factory=NewsConfig)
    data: DataConfig = msgspec.field(default_factory=DataConfig)
    recorder: RecorderConfig = msgspec.field(default_factory=RecorderConfig)
    evidence: EvidenceConfig = msgspec.field(default_factory=EvidenceConfig)
    eval: EvalConfig = msgspec.field(default_factory=EvalConfig)
    paper: PaperConfig = msgspec.field(default_factory=PaperConfig)


# ======================================================================================================================
# Validation (load-time rules of section 4; every violation is collected, one ConfigError names them all)
# ======================================================================================================================

_LABEL_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")  # path-safe; no ":" (namespace separator), no "#" (family suffix)
_TICKER_RE: Final = re.compile(r"[A-Z]{1,6}")  # OCC root
_EPS: Final = 1e-9


class _Problems:
    def __init__(self) -> None:
        self.items: list[str] = []

    def need(self, ok: bool, message: str) -> None:
        if not ok:
            self.items.append(message)


def _non_finite_floats(obj: Any, path: str, out: list[str]) -> None:
    if isinstance(obj, float):
        if not math.isfinite(obj):
            out.append(path)
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            _non_finite_floats(value, f"{path}.{key}" if path else str(key), out)
    elif isinstance(obj, list | tuple):
        for i, value in enumerate(obj):
            _non_finite_floats(value, f"{path}[{i}]", out)


def _strictly_increasing(values: Sequence[float]) -> bool:
    return all(a < b for a, b in pairwise(values))


def _check_ladder(p: _Problems, name: str, ladder: Sequence[float]) -> None:
    ok = len(ladder) >= 1 and all(0.0 < x <= 1.0 for x in ladder) and _strictly_increasing(ladder) and ladder[-1] == 1.0
    p.need(ok, f"orders.{name}: rungs must be in (0, 1], strictly increasing and end at 1.0 (never past natural)")


def _check_tier_table(p: _Problems, name: str, table: Sequence[tuple[float, float]]) -> None:
    thresholds = [row[0] for row in table]
    tiers = [row[1] for row in table]
    ok = (
        len(table) >= 1
        and all(0.0 < a < 1.0 for a in thresholds)
        and all(0.0 < b <= 1.0 for b in tiers)
        and _strictly_increasing(thresholds[::-1])
        and all(x >= y for x, y in pairwise(tiers))
    )
    p.need(
        ok, f"rules.tiers.{name}: rows [threshold, tier] need thresholds in (0, 1) strictly decreasing and tiers in (0, 1] non-increasing"
    )


def _has_flag(flags: Sequence[str], stem: str) -> bool:
    return any(f == stem or f.startswith(stem + ":") for f in flags)


def validate(cfg: Config, *, flags: Sequence[str] | None = ()) -> None:
    """Load-time validation (section 4). Raises one `ConfigError` listing every violated rule.

    `flags` are the run's flags (`RunMeta.flags`, CLI `--flag`): `run.ledger_forecasts = false` is accepted only for runs flagged
    `baseline:4:*` or `shadow`. `flags=None` means "flags unknown here" and skips exactly the flag-dependent rules
    (`resolve()` re-validates a programmatically built config this way).
    """
    p = _Problems()
    run, cad, dte, cand, rules, risk = cfg.run, cfg.cadence, cfg.dte, cfg.candidates, cfg.rules, cfg.risk
    paper_mode = run.mode is RunMode.PAPER

    bad: list[str] = []
    _non_finite_floats(msgspec.to_builtins(cfg), "", bad)
    p.need(not bad, f"non-finite float(s) at: {', '.join(bad)}")

    # --- [run] ---------------------------------------------------------------------------------------------------------
    p.need(
        _LABEL_RE.fullmatch(run.experiment) is not None,
        "run.experiment: use letters, digits, '_', '.', '-' (it is part of the namespace and of paths)",
    )
    p.need(
        run.family == "" or _LABEL_RE.fullmatch(run.family) is not None,
        "run.family: use letters, digits, '_', '.', '-' ('#' suffixes are derived by code)",
    )
    p.need(run.seed >= 0, "run.seed must be >= 0")
    p.need(run.initial_equity_usd > 0, "run.initial_equity_usd must be positive")
    p.need(run.start <= run.end, "run.start must not be after run.end")
    p.need(run.purpose != "paper" or paper_mode, 'run.purpose = "paper" requires run.mode = "paper"')
    p.need(run.purpose != "shadow" or not paper_mode, 'run.purpose = "shadow" is a backtest-mode replay: run.mode must be "backtest"')
    p.need(
        run.purpose != "reference" or cfg.decider.kind in ("auto", "mock"),
        'run.purpose = "reference" is the Jev-free reference-history run: decider.kind must be "mock" (or "auto" without a key)',
    )
    if flags is not None:
        p.need(
            run.ledger_forecasts or _has_flag(flags, "baseline:4") or _has_flag(flags, "shadow"),
            "run.ledger_forecasts = false is accepted only for runs flagged baseline:4:* or shadow",
        )
        p.need(run.purpose != "shadow" or _has_flag(flags, "shadow"), 'run.purpose = "shadow" requires the run flag "shadow"')

    # --- [universe] / [structures] ----------------------------------------------------------------------------------------
    uni = cfg.universe
    p.need(len(uni.underlyings) >= 1, "universe.underlyings must not be empty")
    p.need(len(set(uni.underlyings)) == len(uni.underlyings), "universe.underlyings must be unique")
    for u in (*uni.underlyings, *uni.penny_all):
        p.need(_TICKER_RE.fullmatch(u) is not None, f"universe: {u!r} is not an OCC root ([A-Z]{{1,6}})")
    for table_name, table in (("alias", uni.alias), ("kind", uni.kind), ("iv_proxy", uni.iv_proxy)):
        for u in uni.underlyings:
            p.need(bool(table.get(u, "").strip()), f"universe.{table_name}: no entry for underlying {u}")
    used_aliases = [uni.alias[u] for u in uni.underlyings if u in uni.alias]
    p.need(len(set(used_aliases)) == len(used_aliases), "universe.alias: two underlyings share one alias")
    for u in uni.underlyings:
        proxy = uni.iv_proxy.get(u)
        p.need(proxy is None or proxy in cfg.data.cboe_indices, f"universe.iv_proxy.{u} = {proxy!r} is not listed in data.cboe_indices")
    p.need(len(cfg.structures.enabled) >= 1, "structures.enabled must not be empty")
    p.need(len(set(cfg.structures.enabled)) == len(cfg.structures.enabled), "structures.enabled must be unique")

    # --- [cadence] (INV-13: offsets from the close) -------------------------------------------------------------------------
    p.need(
        cad.order_cutoff_offset_min < cad.exec_offset_min < cad.decide_offset_min,
        "cadence: order_cutoff_offset_min < exec_offset_min < decide_offset_min must hold",
    )
    p.need(
        0 < cad.cancel_all_offset_min <= cad.order_cutoff_offset_min,
        "cadence: 0 < cancel_all_offset_min <= order_cutoff_offset_min must hold",
    )
    p.need(
        cad.order_cutoff_offset_min < cad.decision_deadline_offset_min < cad.decide_offset_min,
        "cadence: order_cutoff_offset_min < decision_deadline_offset_min < decide_offset_min must hold",
    )
    p.need(
        cad.post_close_offset_min <= cad.eod_offset_min < 0,
        "cadence: post_close_offset_min <= eod_offset_min < 0 must hold (minutes AFTER the close)",
    )
    p.need(cad.morning_reconcile_after_open_min >= 0, "cadence.morning_reconcile_after_open_min must be >= 0")
    p.need(cad.min_session_minutes > cad.decide_offset_min, "cadence.min_session_minutes must exceed decide_offset_min")

    # --- [dte] ------------------------------------------------------------------------------------------------------------
    p.need(dte.hard_exit_sessions >= 2, "dte.hard_exit_sessions must be >= 2 (INV-11)")
    p.need(0 < dte.min_entry <= dte.target <= dte.max_entry, "dte: 0 < min_entry <= target <= max_entry must hold")
    p.need(dte.max_entry <= cfg.data.max_dte, "dte.max_entry must be <= data.max_dte (enriched chains keep no longer expiries)")
    p.need(not paper_mode or dte.max_entry <= cfg.recorder.max_dte, "dte.max_entry must be <= recorder.max_dte in paper mode")
    p.need(0 <= dte.time_exit_short_premium < dte.min_entry, "dte.time_exit_short_premium must be in [0, min_entry)")
    p.need(0 <= dte.time_exit_long_premium < dte.min_entry, "dte.time_exit_long_premium must be in [0, min_entry)")
    p.need(dte.min_sessions_beyond_hard_exit >= 0, "dte.min_sessions_beyond_hard_exit must be >= 0")
    p.need(dte.hold_horizon_sessions >= 1, "dte.hold_horizon_sessions must be >= 1")

    # --- [candidates] -----------------------------------------------------------------------------------------------------
    for name in (
        "long_delta",
        "long_min_delta",
        "debit_long_delta",
        "debit_short_delta",
        "credit_short_delta",
        "credit_long_delta",
        "condor_short_delta",
        "condor_long_delta",
    ):
        p.need(0.0 < getattr(cand, name) < 1.0, f"candidates.{name} must be in (0, 1)")
    p.need(cand.long_min_delta <= cand.long_delta, "candidates.long_min_delta must be <= long_delta")
    p.need(cand.debit_short_delta < cand.debit_long_delta, "candidates.debit_short_delta must be < debit_long_delta")
    p.need(cand.credit_long_delta < cand.credit_short_delta, "candidates.credit_long_delta must be < credit_short_delta")
    p.need(cand.condor_long_delta < cand.condor_short_delta, "candidates.condor_long_delta must be < condor_short_delta")
    p.need(cand.delta_tolerance > 0.0, "candidates.delta_tolerance must be positive")
    p.need(0.0 <= cand.max_unsizeable_rate <= 1.0, "candidates.max_unsizeable_rate must be in [0, 1]")
    p.need(cand.credit_short_min_em >= 0.0, "candidates.credit_short_min_em must be >= 0")
    p.need(cand.min_width_strikes >= 1, "candidates.min_width_strikes must be >= 1")
    p.need(cand.max_width_pct_spot > 0.0, "candidates.max_width_pct_spot must be positive")
    p.need(
        0.0 < cand.min_credit_to_width < cand.max_credit_to_width < 1.0,
        "candidates: 0 < min_credit_to_width < max_credit_to_width < 1 must hold",
    )
    p.need(
        0.0 < cand.condor_min_credit_to_width < cand.condor_max_credit_to_width < 1.0,
        "candidates: 0 < condor_min_credit_to_width < condor_max_credit_to_width < 1 must hold",
    )
    p.need(0.0 < cand.max_debit_to_width < 1.0, "candidates.max_debit_to_width must be in (0, 1)")
    p.need(0.0 <= cand.min_two_sided_frac <= 1.0, "candidates.min_two_sided_frac must be in [0, 1]")
    # candidate feasibility: credit / width is approximately the risk-neutral probability mass at the strikes
    vertical_cap = 0.75 * (cand.credit_short_delta + cand.credit_long_delta) / 2.0
    p.need(
        cand.min_credit_to_width <= vertical_cap + _EPS,
        f"candidates.min_credit_to_width = {cand.min_credit_to_width} is infeasible: it must be <= 0.75 * (credit_short_delta + credit_long_delta) / 2 = {vertical_cap:.6f}",
    )
    condor_cap = 0.75 * (cand.condor_short_delta + cand.condor_long_delta)
    p.need(
        cand.condor_min_credit_to_width <= condor_cap + _EPS,
        f"candidates.condor_min_credit_to_width = {cand.condor_min_credit_to_width} is infeasible: it must be <= 0.75 * (condor_short_delta + condor_long_delta) = {condor_cap:.6f}",
    )

    # --- [liquidity] / [exits] ----------------------------------------------------------------------------------------------
    liq = cfg.liquidity
    p.need(liq.min_bid_cents_sold >= 0 and liq.min_bid_cents_bought >= 0, "liquidity.min_bid_cents_* must be >= 0")
    p.need(liq.max_rel_spread > 0.0 and liq.max_abs_spread_cents > 0, "liquidity spread limits must be positive")
    p.need(liq.min_open_interest >= 0, "liquidity.min_open_interest must be >= 0")
    p.need(0.0 < liq.max_pct_displayed_size <= 1.0, "liquidity.max_pct_displayed_size must be in (0, 1]")
    p.need(0.0 < liq.max_pct_open_interest <= 1.0, "liquidity.max_pct_open_interest must be in (0, 1]")
    ex = cfg.exits
    p.need(0.0 < ex.profit_target_frac <= 1.0, "exits.profit_target_frac must be in (0, 1]")
    p.need(0.0 < ex.stop_loss_frac <= 1.0, "exits.stop_loss_frac must be in (0, 1]")
    p.need(ex.ex_div_exit_sessions >= 1, "exits.ex_div_exit_sessions must be >= 1")
    p.need(ex.assignment_extrinsic_floor_cents >= 0, "exits.assignment_extrinsic_floor_cents must be >= 0")
    p.need(ex.assignment_sim_itm >= 0.0 and ex.assignment_sim_dte >= 0, "exits.assignment_sim_* must be >= 0")

    # --- [rules] ------------------------------------------------------------------------------------------------------------
    w = rules.weights
    weights = (w.align, w.volfit, w.fit, w.regimefit, w.calm)
    p.need(all(0.0 <= x <= 1.0 for x in weights), "rules.weights: every weight must be in [0, 1]")
    p.need(abs(math.fsum(weights) - 1.0) <= _EPS, f"rules.weights must sum to 1.0 (got {math.fsum(weights)!r})")
    p.need(
        0.0 < rules.veto_lo < rules.exit_pressure_release < rules.exit_pressure_enter <= rules.veto_hi < 1.0,
        "rules: 0 < veto_lo < exit_pressure_release < exit_pressure_enter <= veto_hi < 1 must hold",
    )
    p.need(0.0 < rules.min_score < 1.0, "rules.min_score must be in (0, 1)")
    p.need(Variant.BASE not in rules.perturbation_variants, 'rules.perturbation_variants must not contain "base"')
    p.need(len(set(rules.perturbation_variants)) == len(rules.perturbation_variants), "rules.perturbation_variants must be unique")
    p.need(rules.reentry_cooldown_sessions >= 0, "rules.reentry_cooldown_sessions must be >= 0")
    p.need(rules.text_watch_alert_sessions >= 1, "rules.text_watch_alert_sessions must be >= 1")
    p.need(0.0 <= rules.text_rank_weight <= 1.0, "rules.text_rank_weight must be in [0, 1]")
    p.need(rules.code_crosschecks or not paper_mode, "rules.code_crosschecks = false is for baseline 3 only: refused in paper mode")
    for gate_name in ("direction", "vol_stance", "structure", "action"):
        gate: GateConfig = getattr(rules.gates, gate_name)
        p.need(
            0.0 < gate.p_top < 1.0 and 0.0 <= gate.margin < 1.0, f"rules.gates.{gate_name}: p_top must be in (0, 1) and margin in [0, 1)"
        )
    _check_tier_table(p, "score", rules.tiers.score)
    _check_tier_table(p, "peakedness", rules.tiers.peakedness)
    env = rules.tiers.environment
    p.need(
        all(0.0 <= x <= 1.0 for x in env) and all(a >= b for a, b in pairwise(env)),
        "rules.tiers.environment: four tiers in [0, 1], non-increasing with the risk.environment level",
    )

    # --- [risk]: every risk limit positive ------------------------------------------------------------------------------------
    for name in (
        "max_loss_per_trade_pct",
        "max_aggregate_open_loss_pct",
        "daily_loss_halt_pct",
        "drawdown_kill_pct",
        "max_bp_utilisation",
        "bp_drift_halt_pct",
    ):
        p.need(0.0 < getattr(risk, name) <= 1.0, f"risk.{name} must be in (0, 1]")
    for name in (
        "max_open_structures",
        "max_new_per_day",
        "max_same_direction_structures",
        "max_contracts_per_trade",
        "max_order_notional_usd",
        "max_orders_per_minute",
        "max_order_attempts",
    ):
        p.need(getattr(risk, name) > 0, f"risk.{name} must be positive")
    p.need(risk.max_adverse_drift > 0.0, "risk.max_adverse_drift must be positive")
    p.need(risk.bp_haircut_mult >= 1.0, "risk.bp_haircut_mult must be >= 1.0 (never below the Cboe-minimum requirement)")
    p.need(risk.event_blackout_sessions >= 0, "risk.event_blackout_sessions must be >= 0")
    p.need(risk.max_contracts_per_trade <= 10, "risk.max_contracts_per_trade must be <= 10 (one order per structure; 2.10)")
    p.need(
        risk.max_loss_per_trade_pct <= risk.max_aggregate_open_loss_pct,
        "risk.max_loss_per_trade_pct must be <= max_aggregate_open_loss_pct",
    )

    # --- [kill] / [health] ----------------------------------------------------------------------------------------------------
    kill = cfg.kill
    p.need(kill.cushion_ticks >= 0, "kill.cushion_ticks must be >= 0")
    p.need(0.0 < kill.cushion_max_frac_width <= 1.0, "kill.cushion_max_frac_width must be in (0, 1]")
    for name in ("flatten_attempts", "flatten_wait_s", "not_flat_retry_s", "cancel_wait_s"):
        p.need(getattr(kill, name) > 0, f"kill.{name} must be positive")
    p.need(
        0 <= kill.post_open_delay_urgent_min <= kill.post_open_delay_min,
        "kill: 0 <= post_open_delay_urgent_min <= post_open_delay_min must hold",
    )
    p.need(kill.backtest_cooldown_sessions >= 0, "kill.backtest_cooldown_sessions must be >= 0")
    for trigger in KillTrigger:
        action = kill.actions.for_trigger(trigger)
        p.need(
            action is not TriggerAction.HALT_THEN_KILL or trigger in _ESCALATING_TRIGGERS,
            f'kill.actions.{trigger.value} = "halt_then_kill" has no persistence threshold: use "kill" (or "halt", flagged KILL_DISABLED)',
        )
    health = cfg.health
    for name in (
        "max_clock_skew_ms",
        "clock_skew_kill_after_s",
        "max_chain_age_s",
        "max_quote_age_s",
        "jev_error_kill_after_sessions",
        "broker_error_halt_after",
        "broker_error_kill_after",
        "max_stale_mark_sessions",
    ):
        p.need(getattr(health, name) > 0, f"health.{name} must be positive")
    p.need(health.stale_kill_after_sessions >= 2, "health.stale_kill_after_sessions must be >= 2")
    p.need(0.0 < health.max_stale_quote_frac <= 1.0, "health.max_stale_quote_frac must be in (0, 1]")
    p.need(0.0 <= health.min_two_sided_frac <= 1.0, "health.min_two_sided_frac must be in [0, 1]")
    p.need(
        health.broker_error_halt_after <= health.broker_error_kill_after,
        "health.broker_error_halt_after must be <= broker_error_kill_after",
    )

    # --- [fees] / [fills] / [orders] ------------------------------------------------------------------------------------------
    fees = cfg.fees
    for name in ("orf", "occ", "taf_sell", "cat", "sec_sell_rate", "commission"):
        p.need(getattr(fees, name) >= 0.0, f"fees.{name} must be >= 0")
    fills = cfg.fills
    p.need(all(0.5 <= x <= 1.0 for x in fills.orats_p), "fills.orats_p: every share must be in [0.5, 1.0] (between mid and natural)")
    p.need(fills.max_rel_spread > 0.0 and fills.max_abs_spread_cents > 0, "fills spread limits must be positive")
    p.need(fills.forced_penalty_frac_spread >= 0.0 and fills.forced_no_quote_pad_cents >= 0, "fills.forced_* must be >= 0")
    orders = cfg.orders
    _check_ladder(p, "entry_ladder", orders.entry_ladder)
    _check_ladder(p, "exit_ladder", orders.exit_ladder)
    for name in ("ladder_step_s", "poll_s", "call_deadline_s", "ambiguous_lookups", "rest_calls_per_minute"):
        p.need(getattr(orders, name) > 0, f"orders.{name} must be positive")
    p.need(all(x > 0.0 for x in orders.http_timeout_s), "orders.http_timeout_s: (connect, read) must both be positive")
    p.need(orders.read_retries >= 0, "orders.read_retries must be >= 0")

    # --- [jev] / [state] / [news] -------------------------------------------------------------------------------------------------
    jev = cfg.jev
    p.need(
        bool(jev.model.strip()) and ":" not in jev.model and jev.model == jev.model.strip(), "jev.model must be a non-blank id without ':'"
    )
    p.need(not jev.model.strip().endswith("latest"), "jev.model must be a PINNED model id, never a '-latest' alias (D6)")
    p.need(bool(jev.sdk_version.strip()), "jev.sdk_version must not be blank")
    p.need(jev.refresh_generation >= 0, "jev.refresh_generation must be >= 0")
    p.need(jev.timeout_s > 0.0, "jev.timeout_s must be positive")
    p.need(jev.retry_max >= 0, "jev.retry_max must be >= 0")
    p.need(
        len(jev.retry_backoff_s) >= jev.retry_max and all(x >= 0.0 for x in jev.retry_backoff_s),
        "jev.retry_backoff_s needs one non-negative delay per retry (len >= retry_max)",
    )
    for name in ("max_concurrency", "max_rps", "max_tokens_per_s"):
        p.need(getattr(jev, name) >= 1, f"jev.{name} must be >= 1")
    spend = jev.spend
    for name in ("max_input_tokens_per_run", "max_input_tokens_per_day", "paper_max_input_tokens_per_day"):
        p.need(getattr(spend, name) > 0, f"jev.spend.{name} must be positive")
    p.need(spend.estimate_chars_per_token > 0.0, "jev.spend.estimate_chars_per_token must be positive")
    p.need(spend.estimate_overhead_tokens >= 0, "jev.spend.estimate_overhead_tokens must be >= 0")
    state = cfg.state
    p.need(0 < state.max_chars <= state.hard_max_chars, "state: 0 < max_chars <= hard_max_chars must hold")
    if state.unmasked:
        p.need(
            run.mode is RunMode.BACKTEST and run.purpose == "diagnostic",
            'state.unmasked = true forces run.mode = "backtest" and run.purpose = "diagnostic" (it can never trade)',
        )
    news = cfg.news
    for name in ("lookback_hours", "max_items", "max_headline_chars", "max_summary_chars", "max_total_chars", "max_symbols_per_item"):
        p.need(getattr(news, name) > 0, f"news.{name} must be positive")
    p.need(news.lag_s >= 0, "news.lag_s must be >= 0")
    p.need(bool(news.mask_terms_file.strip()), "news.mask_terms_file must not be blank")

    # --- [data] / [recorder] / [evidence] / [eval] / [paper] ------------------------------------------------------------------------
    data = cfg.data
    p.need(data.max_dte > 0 and data.moneyness_window > 0.0, "data.max_dte and data.moneyness_window must be positive")
    p.need(
        len(data.cboe_indices) >= 1 and len(set(data.cboe_indices)) == len(data.cboe_indices),
        "data.cboe_indices must be non-empty and unique",
    )
    p.need(bool(data.mirror_repo.strip()), "data.mirror_repo must not be blank")
    p.need(data.min_history_sessions >= 1, "data.min_history_sessions must be >= 1")
    p.need(0 <= data.iv_rank_lo_pct < data.iv_rank_hi_pct <= 100, "data: 0 <= iv_rank_lo_pct < iv_rank_hi_pct <= 100 must hold")
    p.need(data.atm_iv_tolerance > 0.0, "data.atm_iv_tolerance must be positive")
    p.need(data.fomc_knowable_days >= 0 and data.exdiv_knowable_days >= 0, "data.*_knowable_days must be >= 0")
    p.need(data.fomc.expected_per_year > 0, "data.fomc.expected_per_year must be positive")
    years = [e.year for e in data.fomc.exceptions]
    p.need(len(set(years)) == len(years), "data.fomc.exceptions: one entry per year")
    for e in data.fomc.exceptions:
        p.need(e.expected == 7, f"data.fomc.exceptions[{e.year}].expected: 7 is the only accepted value")
        p.need(bool(e.reason.strip()), f"data.fomc.exceptions[{e.year}].reason must not be blank (exceptions are never assumed)")
        p.need(e.source_url.startswith("https://"), f"data.fomc.exceptions[{e.year}].source_url must be an https:// source")
    syn = data.synthetic
    p.need(syn.base_vol > 0.0, "data.synthetic.base_vol must be positive")
    p.need(0.0 < syn.vol_mean_revert <= 1.0, "data.synthetic.vol_mean_revert must be in (0, 1]")
    p.need(syn.vol_of_vol >= 0.0, "data.synthetic.vol_of_vol must be >= 0")
    p.need(0.0 <= syn.cross_corr <= 1.0, "data.synthetic.cross_corr must be in [0, 1] (one common factor)")
    p.need(syn.implied_premium > -1.0, "data.synthetic.implied_premium must be > -1")
    p.need(syn.spread_pct >= 0.0 and syn.strike_step_pct > 0.0, "data.synthetic: spread_pct >= 0 and strike_step_pct > 0 must hold")
    p.need(syn.quote_size > 0 and syn.open_interest >= 0, "data.synthetic: quote_size > 0 and open_interest >= 0 must hold")
    p.need(all(v > 0.0 for v in syn.start_price_usd.values()), "data.synthetic.start_price_usd: prices must be positive")
    if data.provider == "synthetic":
        for u in uni.underlyings:
            p.need(u in syn.start_price_usd, f"data.synthetic.start_price_usd: no entry for underlying {u}")
    rec = cfg.recorder
    p.need(0.0 < rec.strike_window_pct < 1.0, "recorder.strike_window_pct must be in (0, 1)")
    p.need(rec.max_dte > 0, "recorder.max_dte must be positive")
    p.need(len(rec.slots) >= 1 and len(set(rec.slots)) == len(rec.slots), "recorder.slots must be non-empty and unique")
    p.need(not paper_mode or set(rec.slots) == set(Slot), "recorder.slots: paper mode needs all of dec, exec, eod")
    p.need(cfg.evidence.tier_a_max_log_delay_s > 0, "evidence.tier_a_max_log_delay_s must be positive")
    p.need(bool(cfg.evidence.prereg_file.strip()), "evidence.prereg_file must not be blank")
    ev = cfg.eval
    for name in (
        "bootstrap_reps",
        "ece_min_per_bin",
        "base_rate_min_events",
        "recal_min_events",
        "recal_refit_sessions",
        "random_baseline_seeds",
        "placebo_min_distance_sessions",
        "null_sim_reps",
    ):
        p.need(getattr(ev, name) >= 1, f"eval.{name} must be >= 1")
    p.need(ev.bootstrap_block >= 0, "eval.bootstrap_block must be >= 0 (0 = automatic)")
    p.need(ev.ece_bins >= 2, "eval.ece_bins must be >= 2")
    p.need(0.0 < ev.ci_level < 1.0, "eval.ci_level must be in (0, 1)")
    p.need(ev.weekday_tail_ratio_max > 1.0, "eval.weekday_tail_ratio_max must be > 1")
    p.need(cfg.paper.heartbeat_s > 0, "paper.heartbeat_s must be positive")
    p.need(cfg.paper.deadman_stale_s > cfg.paper.heartbeat_s, "paper.deadman_stale_s must exceed paper.heartbeat_s")

    if p.items:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(p.items))


# ======================================================================================================================
# Loading: default.toml <- files <- CLI overrides
# ======================================================================================================================


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc.strerror or type(exc).__name__}") from None
    try:
        decoded = msgspec.toml.decode(raw)
    except msgspec.DecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from None
    if not isinstance(decoded, dict):
        raise ConfigError(f"{path}: the top level of a config file must be a table")
    return decoded


def _deep_merge(base: dict[str, Any], top: Mapping[str, Any]) -> dict[str, Any]:
    """Tables merge key by key; every other value (scalars, arrays) replaces."""
    out = dict(base)
    for key, value in top.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        elif isinstance(value, Mapping):
            out[key] = _deep_merge({}, value)
        else:
            out[key] = value
    return out


def _is_struct(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, msgspec.Struct)


def _leaf_type(parts: Sequence[str], key: str) -> Any:
    """The annotation of the config key `section.key[.sub...]`; ConfigError when the path does not exist in the schema."""
    tp: Any = Config
    for part in parts:
        if _is_struct(tp):
            hints = get_type_hints(tp)
            if part not in hints:
                raise ConfigError(f"-o {key}: unknown config key (no {part!r} in [{tp.__name__}])")
            tp = hints[part]
        elif get_origin(tp) is dict:
            tp = get_args(tp)[1]
        else:
            raise ConfigError(f"-o {key}: {part!r} goes below a scalar value")
    return tp


def _is_stringy(tp: Any) -> bool:
    if tp is str:
        return True
    if isinstance(tp, type) and issubclass(tp, Enum) and issubclass(tp, str):
        return True
    return get_origin(tp) is Literal and all(isinstance(arg, str) for arg in get_args(tp))


def _parse_override_value(raw: str, tp: Any, key: str) -> Any:
    """String-typed keys take the raw text (a quoted value is read as a TOML string); everything else is a TOML value."""
    if _is_stringy(tp):
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
            try:
                quoted = tomllib.loads(f"value = {raw}")["value"]
            except tomllib.TOMLDecodeError:
                return raw[1:-1]
            return quoted if isinstance(quoted, str) else raw[1:-1]
        return raw
    try:
        return tomllib.loads(f"value = {raw}")["value"]
    except tomllib.TOMLDecodeError:
        raise ConfigError(f"-o {key}: cannot parse the value as TOML (arrays look like [1, 2], booleans are true / false)") from None


def _parse_override(item: str) -> tuple[list[str], Any]:
    key, sep, raw = item.partition("=")
    key = key.strip()
    parts = key.split(".")
    if not sep or len(parts) < 2 or not all(parts):
        raise ConfigError(f"-o {item.partition('=')[0]!s}: expected section.key=value")
    return parts, _parse_override_value(raw.strip(), _leaf_type(parts, key), key)


def _set_path(tree: dict[str, Any], parts: Sequence[str], value: Any) -> None:
    node = tree
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    last = parts[-1]
    if isinstance(value, Mapping) and isinstance(node.get(last), dict):
        node[last] = _deep_merge(node[last], value)
    else:
        node[last] = value


def load_config(
    path: Path | str | Sequence[Path | str] | None = None,
    overrides: Sequence[str] = (),
    *,
    flags: Sequence[str] = (),
    default_path: Path | None = None,
) -> Config:
    """`default.toml` <- `path` (one file or several, merged in order) <- `overrides` (`section.key=value`), then `validate()`.

    A typo (unknown key) is a startup error. In paper mode (the MERGED `run.mode`) an override that touches a section of
    `PROTECTED_SECTIONS_PAPER` is refused. `flags` are the run's flags (see `validate`).
    """
    merged = _read_toml(default_path if default_path is not None else DEFAULT_CONFIG_PATH)
    files: Sequence[Path | str] = () if path is None else (path,) if isinstance(path, Path | str) else tuple(path)
    for file in files:
        merged = _deep_merge(merged, _read_toml(Path(file)))

    touched: list[str] = []
    for item in overrides:
        parts, value = _parse_override(item)
        _set_path(merged, parts, value)
        touched.append(".".join(parts))

    try:
        cfg = msgspec.convert(merged, type=Config)
    except msgspec.ValidationError as exc:
        raise ConfigError(f"invalid configuration: {exc}") from None

    if cfg.run.mode is RunMode.PAPER:
        refused = [key for key in touched if key.split(".", 1)[0] in PROTECTED_SECTIONS_PAPER]
        if refused:
            raise ConfigError(
                f"-o is refused in paper mode for the sections {', '.join(PROTECTED_SECTIONS_PAPER)} "
                f"(got: {', '.join(refused)}): the service's limits come from files under version control only"
            )
    validate(cfg, flags=flags)
    return cfg


# ----------------------------------------------------------------------------------------------------------------------
# TOML writer (config.resolved.toml, 13.1). msgspec's TOML encoder needs `tomli_w`, which is not a dependency (1.1).
# ----------------------------------------------------------------------------------------------------------------------

_BARE_KEY_RE: Final = re.compile(r"[A-Za-z0-9_-]+")


def _toml_key(key: str) -> str:
    return key if _BARE_KEY_RE.fullmatch(key) else _toml_str(key)


def _toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError("cannot write a non-finite float to TOML")
        return repr(value)
    if isinstance(value, str):
        return _toml_str(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{_toml_key(str(k))} = {_toml_value(v)}" for k, v in value.items()) + "}"
    raise ConfigError(f"cannot write a value of type {type(value).__name__} to TOML")


def _toml_table(name: str, table: Mapping[str, Any], out: list[str]) -> None:
    scalars = {k: v for k, v in table.items() if not isinstance(v, Mapping)}
    tables = {k: v for k, v in table.items() if isinstance(v, Mapping)}
    if name and (scalars or not tables):
        out.append(f"[{name}]")
    out.extend(f"{_toml_key(str(k))} = {_toml_value(v)}" for k, v in scalars.items())
    if name and (scalars or not tables):
        out.append("")
    for k, v in tables.items():
        _toml_table(f"{name}.{_toml_key(str(k))}" if name else _toml_key(str(k)), v, out)


def dumps_toml(cfg: Config) -> str:
    """The config as TOML text; `load_config` reads it back to an equal `Config` (used for `config.resolved.toml`)."""
    out: list[str] = []
    _toml_table("", msgspec.to_builtins(cfg), out)
    return "\n".join(out).rstrip("\n") + "\n"


# ======================================================================================================================
# Paths
# ======================================================================================================================


def resolve_path(value: str | Path) -> Path:
    """A path written in the config (`paths.env_file`, `news.mask_terms_file`, `evidence.prereg_file`): absolute stays,
    relative is taken from the repository root (never from the process's working directory)."""
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def data_dir(cfg: Config, secrets: "Secrets") -> Path:
    """`paths.data_dir`, else `$JEVBOT_DATA`, else `/home/tyler/jevbot-data` (D1)."""
    return Path(cfg.paths.data_dir.strip() or secrets.data_dir or DEFAULT_DATA_DIR).expanduser()


def ensure_data_dir(path: Path, *, create: bool = True) -> Path:
    """The data directory must be outside the git repository and mode 700 (no group / other access); created when missing."""
    resolved = path.expanduser().resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ConfigError(f"the data directory {resolved} is inside the git repository {REPO_ROOT}: move it outside (D1)")
    if not resolved.exists():
        if not create:
            raise ConfigError(f"the data directory {resolved} does not exist")
        resolved.mkdir(mode=0o700, parents=True)
        resolved.chmod(0o700)
    if not resolved.is_dir():
        raise ConfigError(f"the data directory {resolved} is not a directory")
    mode = stat.S_IMODE(resolved.stat().st_mode)
    if mode & 0o077 or (mode & 0o700) != 0o700:
        raise ConfigError(f"the data directory {resolved} has mode {mode:03o}: it must be 700 (chmod 700 {resolved})")
    return resolved


# ======================================================================================================================
# Secrets (D2, D29; INV-02, INV-18)
# ======================================================================================================================


def _present(value: str | None) -> bool:
    return value is not None and bool(value.strip())


class Secrets(msgspec.Struct, frozen=True, kw_only=True):
    """The secret values of one process. Blank means absent (None). `repr()` never shows a value, and every value is registered
    with the log redaction layer on construction - however the instance was built."""

    typesafe_api_key: str | None = None
    alpaca_paper_key: str | None = None
    alpaca_paper_secret: str | None = None
    data_dir: str | None = None  # $JEVBOT_DATA: not a secret, carried here because it arrives through the same env / .env source

    def __post_init__(self) -> None:
        for name in ("typesafe_api_key", "alpaca_paper_key", "alpaca_paper_secret", "data_dir"):
            value = getattr(self, name)
            if value is not None:
                stripped = value.strip()
                # a missing, empty or whitespace value is ABSENT: an empty key is never passed to an SDK (D29)
                msgspec.structs.force_setattr(self, name, stripped or None)
        for secret in (self.typesafe_api_key, self.alpaca_paper_key, self.alpaca_paper_secret):
            logsetup.register_secret(secret)

    @property
    def has_typesafe_key(self) -> bool:
        return self.typesafe_api_key is not None

    @property
    def has_alpaca_key(self) -> bool:
        return self.alpaca_paper_key is not None

    @property
    def alpaca_key_paper_hint(self) -> bool | None:
        """Does the Alpaca key id start with "PK"? None when absent. A hint only: `make_clients` is the guard (11.1)."""
        return None if self.alpaca_paper_key is None else self.alpaca_paper_key.startswith("PK")

    def _summary(self) -> str:
        def show(value: str | None) -> str:
            return "<set>" if value is not None else "<absent>"

        return (
            f"Secrets(typesafe_api_key={show(self.typesafe_api_key)}, alpaca_paper_key={show(self.alpaca_paper_key)}, "
            f"alpaca_paper_secret={show(self.alpaca_paper_secret)}, data_dir={self.data_dir!r})"
        )

    def __repr__(self) -> str:
        return self._summary()

    def __str__(self) -> str:
        return self._summary()

    def __rich_repr__(self) -> Iterator[tuple[str, str | None]]:
        yield "typesafe_api_key", "<set>" if self.typesafe_api_key is not None else "<absent>"
        yield "alpaca_paper_key", "<set>" if self.alpaca_paper_key is not None else "<absent>"
        yield "alpaca_paper_secret", "<set>" if self.alpaca_paper_secret is not None else "<absent>"
        yield "data_dir", self.data_dir


def check_sdk_log_level(env: Mapping[str, str]) -> None:
    """INV-18. `TYPESAFE_LOG_LEVEL`, normalised EXACTLY as the SDK does it (`(value or "").strip().lower()`,
    `typesafe_sdk/_core/logging.py::setup_logging`), must be empty or one of warn / warning / error / off.

    `debug`, `" debug"`, `"DEBUG "`, `info` and typos alike are a ConfigError: at DEBUG the SDK logs full request and response
    bodies. Called by `secrets()` and again by `LiveJev.__init__`, always BEFORE `typesafe_sdk` is imported (the SDK applies
    the level once, at import).
    """
    raw = env.get(ENV_TYPESAFE_LOG_LEVEL)
    level = (raw or "").strip().lower()
    if level and level not in ALLOWED_SDK_LOG_LEVELS:
        raise ConfigError(
            f"{ENV_TYPESAFE_LOG_LEVEL}={raw!r} is refused: only {', '.join(sorted(ALLOWED_SDK_LOG_LEVELS))} (or unset) are allowed - "
            "at debug the SDK logs full request and response bodies (INV-18)"
        )


_ENV_KEY_RE: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def read_env_file(path: Path) -> dict[str, str]:
    """Parse a `.env` file: `KEY=VALUE` lines, `#` comment lines, an optional `export ` prefix, one pair of matching quotes
    around the value stripped. No inline comments, no interpolation (a secret may contain '#' or '$').

    A missing file is an empty mapping. The file must not be readable by group or others (mode 600). Error messages name the
    line NUMBER only - never its content.
    """
    try:
        st = path.stat()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigError(f"cannot stat env file {path}: {exc.strerror or type(exc).__name__}") from None
    if not stat.S_ISREG(st.st_mode):
        raise ConfigError(f"env file {path} is not a regular file")
    mode = stat.S_IMODE(st.st_mode)
    if mode & 0o077:
        raise ConfigError(f"env file {path} has mode {mode:03o}: it must be 600 (chmod 600 {path})")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read env file {path}: {type(exc).__name__}") from None
    out: dict[str, str] = {}
    for number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or _ENV_KEY_RE.fullmatch(key) is None:
            raise ConfigError(f"env file {path}: line {number} is not KEY=VALUE")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def secrets(env: Mapping[str, str], env_file: Path | None) -> Secrets:
    """Read the secrets from `env` (normally `os.environ`) and the git-ignored `.env` file (section 4 rules).

    - A `.env` value never overrides a variable that is already set; a missing, empty or whitespace value is ABSENT.
    - INV-02: any of `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` or an `APCA_*` name (in the environment or in the file) is a
      `PaperGuardError` - ambiguous credentials.
    - `TYPESAFE_BASE_URL` / `TYPESAFE_DEFAULT_MODEL` are a `ConfigError`: the pinned model and default URL are passed explicitly.
    - INV-18: `check_sdk_log_level` on the merged view, before `typesafe_sdk` is imported anywhere.
    - Every loaded secret value is registered with the log redaction filter.
    """
    file_vars = read_env_file(env_file) if env_file is not None else {}
    names = set(env) | set(file_vars)

    ambiguous = sorted(n for n in names if n in FORBIDDEN_ALPACA_ENV or n.startswith(_FORBIDDEN_ALPACA_PREFIX))
    if ambiguous:
        raise PaperGuardError(
            f"ambiguous Alpaca credentials: unset {', '.join(ambiguous)} - only {ENV_ALPACA_PAPER_KEY} / {ENV_ALPACA_PAPER_SECRET} are read (INV-02)"
        )
    pinned = sorted(n for n in names if n in FORBIDDEN_TYPESAFE_ENV)
    if pinned:
        raise ConfigError(
            f"unset {', '.join(pinned)}: the pinned model and the default URL are passed explicitly, never taken from the environment"
        )

    merged: dict[str, str] = dict(file_vars)
    for name, value in env.items():
        if _present(value) or name not in merged:  # a set variable always wins; a blank one never hides the .env value
            merged[name] = value
    check_sdk_log_level(merged)

    for name in _REDACT_ONLY_ENV:
        logsetup.register_secret(merged.get(name))
    return Secrets(
        typesafe_api_key=merged.get(ENV_TYPESAFE_API_KEY),
        alpaca_paper_key=merged.get(ENV_ALPACA_PAPER_KEY),
        alpaca_paper_secret=merged.get(ENV_ALPACA_PAPER_SECRET),
        data_dir=merged.get(ENV_JEVBOT_DATA),
    )


def load_secrets(cfg: Config, env: Mapping[str, str]) -> Secrets:
    """`secrets()` with the `.env` file of this config: `paths.env_file`, taken from the repository root when relative."""
    return secrets(env, resolve_path(cfg.paths.env_file))


# ======================================================================================================================
# Hashes (no floats, no paths, no wall clock, no run id in hashed material)
# ======================================================================================================================


def _hash_material(obj: Any) -> Any:
    """Builtins with every float replaced by its shortest round-trip `repr()` string (nothing hashed may be a float)."""
    if obj is None or isinstance(obj, bool | int | str):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ConfigError("a non-finite float cannot be hashed")
        return repr(obj)
    if isinstance(obj, Mapping):
        return {str(k): _hash_material(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_hash_material(v) for v in obj]
    raise ConfigError(f"config value of type {type(obj).__name__} cannot be hashed")


def _digest(material: Mapping[str, Any]) -> str:
    # canon.dumps_sorted + canon.sha256_hex, literally (3.7)
    text = json.dumps(_hash_material(material), ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _section(cfg: Config, name: str) -> Any:
    return msgspec.to_builtins(getattr(cfg, name))


def config_hash(cfg: Config) -> str:
    """sha256 over the whole config minus `[paths]`. Call it on the RESOLVED config (`resolve()` does)."""
    builtins: dict[str, Any] = msgspec.to_builtins(cfg)
    builtins.pop("paths", None)
    return _digest(builtins)


def _require_state_input(name: str, value: object) -> str:
    """The two non-config inputs of `state_config_hash` are REQUIRED and can never be blank: a caller that had nothing to pass
    would silently get a hash that ignores the bucket tables or the mask dictionary."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConfigError(
            f"state_config_hash needs a non-blank {name} (a str with no surrounding whitespace), got {value!r}: pass "
            "MaskTerms.version (5.8) and buckets.bucket_spec_hash (5.5) - a run cannot be resolved without them"
        )
    return value


def state_config_hash(cfg: Config, *, mask_version: str, bucket_spec_hash: str) -> str:
    """Everything that shapes state bytes: `universe`, `state`, `news`, `dte.hold_horizon_sessions`, `cadence.decide_offset_min`
    (the news recency cutoff, 5.6), `data.min_history_sessions`, `data.iv_rank_lo_pct` / `iv_rank_hi_pct`, the bucket tables
    (`bucket_spec_hash`, 5.5) and the mask version (`MaskTerms.version`, 5.8).

    `mask_version` and `bucket_spec_hash` are REQUIRED keyword arguments (no default, `ConfigError` when blank): the section-4
    scope names both, and a hash computed without them would not change when a bucket threshold or the mask dictionary does."""
    _require_state_input("mask_version", mask_version)
    _require_state_input("bucket_spec_hash", bucket_spec_hash)
    return _digest(
        {
            "universe": _section(cfg, "universe"),
            "state": _section(cfg, "state"),
            "news": _section(cfg, "news"),
            "dte.hold_horizon_sessions": cfg.dte.hold_horizon_sessions,
            "cadence.decide_offset_min": cfg.cadence.decide_offset_min,
            "data.min_history_sessions": cfg.data.min_history_sessions,
            "data.iv_rank_lo_pct": cfg.data.iv_rank_lo_pct,
            "data.iv_rank_hi_pct": cfg.data.iv_rank_hi_pct,
            "bucket_spec_hash": bucket_spec_hash,
            "mask_version": mask_version,
        }
    )


def rules_hash(cfg: Config) -> str:
    return _digest({"rules": _section(cfg, "rules")})


def risk_config_hash(cfg: Config) -> str:
    return _digest({name: _section(cfg, name) for name in ("risk", "kill", "health", "exits", "dte")})


def candidate_config_hash(cfg: Config) -> str:
    """Keys the recorded `data scan-candidates` facts (section 8)."""
    return _digest(
        {
            "candidates": _section(cfg, "candidates"),
            "liquidity": _section(cfg, "liquidity"),
            "dte": _section(cfg, "dte"),
            "risk.max_loss_per_trade_pct": cfg.risk.max_loss_per_trade_pct,
            "rules.tiers": msgspec.to_builtins(cfg.rules.tiers),
            "run.initial_equity_usd": cfg.run.initial_equity_usd,
        }
    )


# ======================================================================================================================
# Run-start resolution (one function, called by every entry point before RUN_START)
# ======================================================================================================================


class ProbeStatus(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Which Step 0 suites are recorded for EXACTLY this (model, SDK version, entry question sets) key (6.8)."""

    meta: bool = False
    determinism: bool = False
    batch: bool = False
    order: bool = False
    text: bool = False


def probe_status(data_dir: Path, *, model: str, sdk_version: str, entry_qset_hash: str, entry_text_qset_hash: str) -> ProbeStatus:
    """Read `$JEVBOT_DATA/probes/step0/records/*.json`. A record counts ONLY if all four key fields match exactly: a Step 0 taken
    on another model id, SDK version or question wording never satisfies this namespace (B3.5: re-run on any model change).
    An unreadable or malformed record never counts (fail closed) and is logged by file name."""
    records_dir = Path(data_dir) / "probes" / "step0" / "records"
    found: set[str] = set()
    if records_dir.is_dir():
        for path in sorted(records_dir.glob("*.json")):
            try:
                record = msgspec.json.decode(path.read_bytes(), type=ProbeRecord)
            except (OSError, msgspec.MsgspecError):
                _log.warning("ignoring unreadable Step 0 probe record %s", path.name)
                continue
            if record.suite not in PROBE_SUITES:
                _log.warning("ignoring Step 0 probe record %s: unknown suite", path.name)
                continue
            if (record.model, record.sdk_version, record.entry_qset_hash, record.entry_text_qset_hash) == (
                model,
                sdk_version,
                entry_qset_hash,
                entry_text_qset_hash,
            ):
                found.add(record.suite)
    return ProbeStatus(**{suite: suite in found for suite in PROBE_SUITES})


class ResolvedConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    cfg: Config  # decider.kind and news.enabled replaced by their concrete values (paper also forces purpose and cache mode)
    news_resolved: bool
    news_reason: str  # one of NEWS_REASONS
    config_hash: str
    state_config_hash: str
    rules_hash: str
    risk_config_hash: str
    candidate_config_hash: str


def _resolve_decider(cfg: Config, secrets: Secrets) -> Literal["live", "replay", "mock"]:
    paper = cfg.run.mode is RunMode.PAPER
    kind = cfg.decider.kind
    if kind == "auto":
        if not secrets.has_typesafe_key:
            return "mock"  # D7
        if paper:
            return "live"
        return "replay" if cfg.jev.cache.mode is CacheMode.REPLAY else "live"
    if kind == "live":
        if not secrets.has_typesafe_key:
            raise ConfigError(f'decider.kind = "live" needs {ENV_TYPESAFE_API_KEY} (absent or blank); use "auto" or "mock" without a key')
        return "live"
    if kind == "replay":
        if paper:
            raise ConfigError('decider.kind = "replay" is refused in paper mode: the paper service records (live Jev) or runs MockJev')
        return "replay"
    return "mock"


def _resolve_news(cfg: Config, secrets: Secrets, probes: ProbeStatus, decider: str) -> tuple[bool, str]:
    paper_live = cfg.run.mode is RunMode.PAPER and decider == "live"
    enabled = cfg.news.enabled
    if enabled == "off":
        return False, "explicit_off"
    if enabled == "on":
        if paper_live and not probes.text:
            raise ConfigError(
                'news.enabled = "on" is refused in paper mode with live Jev until the hostile-text probe is recorded for the pinned '
                "model: run `jevbot jev probe-step0 --suite text` first (V12)"
            )
        return True, "explicit_on"
    if not secrets.has_alpaca_key:
        return False, "auto_no_keys"
    if not paper_live or probes.text:
        return True, "auto_keys_present"
    return False, "text_probe_pending"  # V12


def resolve(cfg: Config, secrets: Secrets, probes: ProbeStatus, *, mask_version: str, bucket_spec_hash: str) -> ResolvedConfig:
    """Resolve `decider.kind` and `news.enabled` ONCE at run start and hash the resolved config. Pure: no IO, no environment.

    - `decider.kind = "auto"`: `mock` when `TYPESAFE_API_KEY` is absent (D7); else `live` in paper and per `jev.cache.mode` in
      backtests (`replay` -> replay, `record` / `refresh` -> live).
    - `news.enabled`: the truth table of section 4 (V12); the reason goes to RUN_START, the report header and the heartbeat.
    - Paper mode with the live decider and `rules.perturbation_scope_paper = "off"` requires the Step 0 `determinism` and
      `order` records.
    - Paper forces `run.purpose = "paper"` and `jev.cache.mode = "record"`.
    - `mask_version` (`MaskTerms.version`, from `load_mask_terms(...)`, 5.8) and `bucket_spec_hash` (`jevbot.buckets`, 5.5) are
      REQUIRED keyword arguments: they are inputs of `state_config_hash` only, but section 4 puts the bucket tables and the mask
      version in that hash's scope, so every entry point (`run_backtest`, the paper runner, the shadow replay, the baselines)
      passes them - `resolve(cfg, secrets, probes)` alone is a `TypeError`, a blank value a `ConfigError`. This function does
      no IO, so it cannot read them itself. They change `state_config_hash` only: `config_hash`, `rules_hash`,
      `risk_config_hash` and `candidate_config_hash` are functions of the config alone.
    """
    _require_state_input("mask_version", mask_version)
    _require_state_input("bucket_spec_hash", bucket_spec_hash)
    validate(cfg, flags=None)
    paper = cfg.run.mode is RunMode.PAPER
    decider = _resolve_decider(cfg, secrets)
    if cfg.run.purpose == "reference" and decider != "mock":
        raise ConfigError(
            'run.purpose = "reference" is the Jev-free reference-history run (12.1): it needs MockJev (decider.kind = "mock")'
        )
    news_resolved, news_reason = _resolve_news(cfg, secrets, probes, decider)
    if paper and decider == "live" and cfg.rules.perturbation_scope_paper == "off" and not (probes.determinism and probes.order):
        raise ConfigError(
            'rules.perturbation_scope_paper = "off" is refused in paper mode with live Jev until the Step 0 determinism and order '
            "records exist for the pinned model: run `jevbot jev probe-step0 --suite determinism` and `--suite order` first"
        )

    replace = msgspec.structs.replace
    run, jev = cfg.run, cfg.jev
    if paper:
        run = replace(run, purpose="paper")
        jev = replace(jev, cache=replace(jev.cache, mode=CacheMode.RECORD))
    resolved = replace(
        cfg,
        run=run,
        jev=jev,
        decider=replace(cfg.decider, kind=decider),
        news=replace(cfg.news, enabled="on" if news_resolved else "off"),
    )
    return ResolvedConfig(
        cfg=resolved,
        news_resolved=news_resolved,
        news_reason=news_reason,
        config_hash=config_hash(resolved),
        state_config_hash=state_config_hash(resolved, mask_version=mask_version, bucket_spec_hash=bucket_spec_hash),
        rules_hash=rules_hash(resolved),
        risk_config_hash=risk_config_hash(resolved),
        candidate_config_hash=candidate_config_hash(resolved),
    )


# ======================================================================================================================
# Small config-derived facts shared by several packages
# ======================================================================================================================


def headline_band(cfg: Config) -> Band:
    """Conventions: `orats` when `cadence.fill_rule = "next_snapshot"`; `worst` when `"same_snapshot_worst"`."""
    return Band.ORATS if cfg.cadence.fill_rule is FillRule.NEXT_SNAPSHOT else Band.WORST


def kill_disabled(cfg: Config) -> tuple[str, ...]:
    """Triggers downgraded to plain `halt` (V2): the report header flag `KILL_DISABLED:<triggers>` and a `doctor` warning."""
    return tuple(t.value for t in KillTrigger if cfg.kill.actions.for_trigger(t) is TriggerAction.HALT)


# ======================================================================================================================
# Mask terms (5.8)
# ======================================================================================================================


def _mask_group(path: Path, group: str, value: Any) -> tuple[tuple[str, ...], str | None]:
    """One group of the file: either `group = ["term", ...]` or a table `[group]` with `terms = [...]` and an optional
    `replacement = "..."`."""
    replacement: str | None = None
    terms: Any = value
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - {"terms", "replacement"})
        if unknown:
            raise ConfigError(f"{path}: [{group}] has unknown key(s) {', '.join(unknown)} (allowed: terms, replacement)")
        terms = value.get("terms")
        raw_replacement = value.get("replacement")
        if raw_replacement is not None:
            if not isinstance(raw_replacement, str) or not raw_replacement.strip():
                raise ConfigError(f"{path}: [{group}].replacement must be a non-blank string")
            replacement = raw_replacement.strip()
    if not isinstance(terms, list) or not terms:
        raise ConfigError(f"{path}: group {group!r} has empty terms (give a non-empty array of strings, or drop the group)")
    cleaned: list[str] = []
    for term in terms:
        if not isinstance(term, str) or not term.strip():
            raise ConfigError(f"{path}: group {group!r} holds an empty or non-string term")
        cleaned.append(term.strip())
    return tuple(cleaned), replacement


def load_mask_terms(path: Path) -> MaskTerms:
    """Parse `config/mask_terms.toml` (5.8): `msgspec.toml` decode + validation.

    File format: one entry per group of `MASK_GROUPS`, in either of two shapes -

        [funds]                                   # a table: `terms` (required) and an optional `replacement` phrase
        terms = ["SPDR S&P 500 ETF Trust", "SPY"]
        replacement = "the fund"

        indices = ["S&P 500", "Nasdaq 100"]       # or a plain array of terms (top of the file, before the first table)

    `ConfigError` on an unknown group, empty terms, a term listed twice (case-insensitively, within or across groups - the
    replacement would be ambiguous) or a file without any group. `groups` and `replacements` always carry all of `MASK_GROUPS`
    (a group the file omits has no terms; its replacement phrase - used by the generic patterns too - is the 5.8 default).
    `version = sha256(MASK_RULES_VERSION + file bytes)[:12]` is the `mask_version` of provenance and `state_config_hash`.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read mask terms file {path}: {exc.strerror or type(exc).__name__}") from None
    try:
        decoded = msgspec.toml.decode(raw)
    except msgspec.DecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from None
    if not isinstance(decoded, dict) or not decoded:
        raise ConfigError(f"{path}: no mask group found (known groups: {', '.join(MASK_GROUPS)})")
    unknown = sorted(set(decoded) - set(MASK_GROUPS))
    if unknown:
        raise ConfigError(f"{path}: unknown mask group(s) {', '.join(unknown)} (known groups: {', '.join(MASK_GROUPS)})")

    groups: dict[str, tuple[str, ...]] = {}
    replacements: dict[str, str] = dict(MASK_REPLACEMENTS)
    seen: dict[str, str] = {}
    for group in MASK_GROUPS:
        if group not in decoded:
            groups[group] = ()
            continue
        terms, replacement = _mask_group(Path(path), group, decoded[group])
        for term in terms:
            folded = term.casefold()
            if folded in seen:
                raise ConfigError(
                    f"{path}: term {term!r} is listed twice (groups {seen[folded]!r} and {group!r}): the replacement would be ambiguous"
                )
            seen[folded] = group
        groups[group] = terms
        if replacement is not None:
            replacements[group] = replacement
    version = hashlib.sha256(MASK_RULES_VERSION.encode("utf-8") + raw).hexdigest()[:12]
    return MaskTerms(version=version, groups=groups, replacements=replacements)
