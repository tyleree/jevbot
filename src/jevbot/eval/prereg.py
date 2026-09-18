"""The pre-registration: loader, validator and registration (DESIGN.md 12.1, D12, G1).

`prereg/prereg.v1.toml` is committed to git and its sha256 is stored in the registry. A forecast is tagged
`prereg = true` only if the first Tier A forecast postdates the registration and the file hash is unchanged - so the
endpoint, the references, the eligibility rule, the interval method and the two looks are all fixed *before* any forward
evidence exists, and a later edit is visible rather than silent.

`register()` is where the discipline lives. It refuses:

1. an **uncommitted** file - a registered hash must point at something a reader can check out;
2. an **empty or unverifiable** `[prereg.reference_history]` - both references warm-start from that Jev-free MockJev run,
   and without it the test can be won with zero skill (12.1);
3. a **missing or stale** `prereg/power.v1.json` - the size of the test must have been simulated for exactly these
   test-defining keys (12.3);
4. a power file in which **no interval candidate holds its size**, or whose `chosen_interval` differs from
   `bootstrap.interval`: a percentile bound over ~12 effective blocks of heavy-tailed, cross-correlated, overlapping
   `d_t` under-covers, and the intersection-union logic does not repair an over-sized component test;
5. the **`EM_WEEKDAY_BIAS`** flag - weekday-dependent base rates of the `*_1em_1s` events mean the trading-time
   thresholds of 5.3 are miscalibrated, which no monotone recalibration can repair (V13).

The validator is deliberately strict about the things a typo could quietly change: the primary family must be evaluation
questions (and must exclude the near-0.5 `eval.up_1s` / `eval.up_5s` pair), the variant must be `base`, the references
must be the two named ones, there must be no fallback on the verdict path, the bootstrap block must cover twice the
longest primary horizon, and the looks' alphas must respect the Bonferroni budget.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Final

import msgspec

from jevbot import canon, vocab
from jevbot.errors import PreregError
from jevbot.eval import load
from jevbot.eval.registry import PreregRow, Registry, Runner, git_head, git_path_committed, run_git

__all__ = [
    "BSS_VS_RAW_IMPLIED",
    "EM_WEEKDAY_BIAS",
    "INTERVAL_METHODS",
    "KNOWN_REFERENCES",
    "NON_EM_DIRECTION_IDS",
    "POWER_FILENAME",
    "TEST_DEFINING_KEYS",
    "AblationSpec",
    "BootstrapSpec",
    "LoadedPrereg",
    "Look",
    "ModelChangeSpec",
    "PowerFile",
    "PowerEntry",
    "Prereg",
    "ReferenceHistorySpec",
    "SizeEntry",
    "WeekdayEntry",
    "forecast_is_prereg",
    "load_power_file",
    "load_prereg",
    "prereg_status",
    "primary_horizon",
    "register",
    "test_defining_hash",
    "validate_prereg",
]

KNOWN_REFERENCES: Final[tuple[str, ...]] = ("implied_recalibrated", "base_rate_expanding")
INTERVAL_METHODS: Final[tuple[str, ...]] = ("percentile", "studentised", "null_calibrated")
BSS_VS_RAW_IMPLIED: Final[str] = "BSS_vs_raw_implied"
EM_WEEKDAY_BIAS: Final[str] = "EM_WEEKDAY_BIAS"
POWER_FILENAME: Final[str] = "power.v1.json"
# 12.1: "deliberately EXCLUDES eval.up_*: against an implied reference near 0.5 they are almost uninformative".
NON_EM_DIRECTION_IDS: Final[frozenset[str]] = frozenset({"eval.up_1s", "eval.up_5s"})
# the Bonferroni budget the two looks of 12.1 split
ALPHA_BUDGET: Final[float] = 0.05


# ======================================================================================================================
# The file
# ======================================================================================================================


class Look(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    n_sessions: int  # counts ELIGIBLE sessions (12.1 `eligible_session`)
    alpha: float


class BootstrapSpec(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    kind: str
    reps: int
    min_block: int  # >= 2 x the longest primary horizon (overlapping 5-session outcomes)
    interval: str  # MUST equal the size-validated `chosen_interval` of prereg/power.v1.json


class ReferenceHistorySpec(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    run_id: str
    ledger_head: str
    data_manifest_hash: str
    use: str
    purge: str
    own_events: str

    @property
    def is_filled(self) -> bool:
        return bool(self.run_id and self.ledger_head and self.data_manifest_hash)


class AblationSpec(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    arms: tuple[str, ...]
    statistics: tuple[str, ...]
    expected: str
    interpretation: str
    command: str


class ModelChangeSpec(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    policy: str
    shadow_overlap: str
    agreement_report: tuple[str, ...]
    procedure: str


class Prereg(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """The `[prereg]` table of `prereg/prereg.v1.toml` (12.1), field for field."""

    id: str
    primary_family: tuple[str, ...]
    primary_forecasts: str  # "with_text" (V9) | "without_text"
    primary_variant: str
    primary_tier: str
    unit: str
    weighting: str
    references: tuple[str, ...]
    reference_min_events: int
    reference_fallback: str
    eligible_session: str
    bootstrap: BootstrapSpec
    interval_candidates: tuple[str, ...]
    size_rule: str
    looks: tuple[Look, ...]
    success: str
    futility: str
    never_sufficient: tuple[str, ...]
    d12_literal: str
    missing: str
    sensitivity: tuple[str, ...]
    secondary: tuple[str, ...]
    pnl_verdict: str
    power: str
    reference_history: ReferenceHistorySpec
    ablation_buckets_only: AblationSpec
    model_change: ModelChangeSpec

    @property
    def alpha_by_look(self) -> tuple[float, ...]:
        return tuple(look.alpha for look in self.looks)


class LoadedPrereg(msgspec.Struct, frozen=True, kw_only=True):
    """One parsed, validated pre-registration together with the bytes it was parsed from."""

    prereg: Prereg
    sha256: str
    body_toml: str
    path: Path


def load_prereg(path: Path | str) -> LoadedPrereg:
    """Read, decode and validate `prereg/prereg.v1.toml`. Raises `PreregError` on anything malformed."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PreregError(f"pre-registration file {path} cannot be read: {exc}") from exc
    try:
        decoded = msgspec.toml.decode(raw)
    except msgspec.MsgspecError as exc:
        raise PreregError(f"pre-registration file {path} is not valid TOML: {exc}") from exc
    if not isinstance(decoded, dict) or "prereg" not in decoded:
        raise PreregError(f"pre-registration file {path} has no [prereg] table")
    try:
        prereg = msgspec.convert(decoded["prereg"], type=Prereg)
    except msgspec.ValidationError as exc:
        raise PreregError(f"pre-registration file {path}: {exc}") from exc
    validate_prereg(prereg)
    return LoadedPrereg(
        prereg=prereg,
        sha256=canon.sha256_hex(raw),
        body_toml=raw.decode("utf-8"),
        path=path,
    )


def primary_horizon(question_id: str) -> int | None:
    """The horizon in sessions of an evaluation question id; `None` for the `hold` horizon, which only the config knows
    (`dte.hold_horizon_sessions`, 6.4)."""
    if question_id.endswith("_hold"):
        return None
    tail = question_id.rsplit("_", 1)[-1]
    if tail.endswith("s") and tail[:-1].isdigit():
        return int(tail[:-1])
    return None


def validate_prereg(prereg: Prereg) -> None:
    """Every structural rule of 12.1, collected into one `PreregError`."""
    problems: list[str] = []

    if not prereg.id:
        problems.append("id is empty")

    if not prereg.primary_family:
        problems.append("primary_family is empty")
    if len(set(prereg.primary_family)) != len(prereg.primary_family):
        problems.append("primary_family repeats a question id")
    unknown = [qid for qid in prereg.primary_family if qid not in vocab.EVAL_IDS]
    if unknown:
        problems.append(f"primary_family contains non-evaluation question ids: {', '.join(sorted(unknown))}")
    included = NON_EM_DIRECTION_IDS.intersection(prereg.primary_family)
    if included:
        problems.append(
            f"primary_family must exclude {', '.join(sorted(included))}: against an implied reference near 0.5 they are "
            "almost uninformative and would dominate a pooled Brier score (12.1)"
        )

    if prereg.primary_forecasts not in ("with_text", "without_text"):
        problems.append(f"primary_forecasts must be 'with_text' or 'without_text', got {prereg.primary_forecasts!r}")
    if prereg.primary_variant != "base":
        problems.append(
            f"primary_variant must be 'base': evaluation forecasts always come from the base variant (7.8, 12.3), "
            f"got {prereg.primary_variant!r}"
        )
    if prereg.primary_tier not in ("A", "B"):
        problems.append(f"primary_tier must be 'A' or 'B' (only those enter a go / no-go table), got {prereg.primary_tier!r}")
    if prereg.unit != "session":
        problems.append(f"unit must be 'session' (3 correlated ETFs => one cluster per session), got {prereg.unit!r}")
    if not prereg.weighting:
        problems.append("weighting is empty")

    if len(set(prereg.references)) != len(prereg.references):
        problems.append("references repeats a reference name")
    bad_refs = [name for name in prereg.references if name not in KNOWN_REFERENCES]
    if bad_refs:
        problems.append(f"unknown reference(s): {', '.join(sorted(bad_refs))}; known: {', '.join(KNOWN_REFERENCES)}")
    if len(prereg.references) < 2:
        problems.append(f"the verdict is a JOINT (intersection-union) test against BOTH references (12.1); got {len(prereg.references)}")
    if prereg.reference_min_events < 1:
        problems.append("reference_min_events must be positive")
    if prereg.reference_fallback != "none":
        problems.append(
            f"reference_fallback must be 'none': there is NO fallback to raw implied on the verdict path (12.1), "
            f"got {prereg.reference_fallback!r}"
        )
    if not prereg.eligible_session:
        problems.append("eligible_session is empty")

    if prereg.bootstrap.kind != "stationary":
        problems.append(f"bootstrap.kind must be 'stationary' (overlapping horizons need block methods), got {prereg.bootstrap.kind!r}")
    if prereg.bootstrap.reps < 1:
        problems.append("bootstrap.reps must be positive")
    horizons = [horizon for horizon in (primary_horizon(qid) for qid in prereg.primary_family) if horizon is not None]
    longest = max(horizons, default=0)
    if longest and prereg.bootstrap.min_block < 2 * longest:
        problems.append(
            f"bootstrap.min_block must be at least 2 x the longest primary horizon ({2 * longest}), got {prereg.bootstrap.min_block}"
        )
    if not prereg.interval_candidates:
        problems.append("interval_candidates is empty")
    if len(set(prereg.interval_candidates)) != len(prereg.interval_candidates):
        problems.append("interval_candidates repeats a method")
    bad_methods = [method for method in prereg.interval_candidates if method not in INTERVAL_METHODS]
    if bad_methods:
        problems.append(f"unknown interval candidate(s): {', '.join(sorted(bad_methods))}")
    if prereg.bootstrap.interval not in prereg.interval_candidates:
        problems.append(
            f"bootstrap.interval {prereg.bootstrap.interval!r} is not one of the interval_candidates {list(prereg.interval_candidates)}"
        )
    if not prereg.size_rule:
        problems.append("size_rule is empty")

    if not prereg.looks:
        problems.append("looks is empty")
    previous = 0
    for index, look in enumerate(prereg.looks):
        if look.n_sessions <= previous:
            problems.append(f"looks[{index}].n_sessions must increase (got {look.n_sessions} after {previous})")
        previous = max(previous, look.n_sessions)
        if not 0.0 < look.alpha < 1.0:
            problems.append(f"looks[{index}].alpha must be in (0, 1), got {look.alpha}")
    total_alpha = sum(look.alpha for look in prereg.looks)
    if total_alpha > ALPHA_BUDGET + 1e-12:
        problems.append(f"the looks' alphas sum to {total_alpha:g}, above the Bonferroni budget {ALPHA_BUDGET:g}")

    if BSS_VS_RAW_IMPLIED not in prereg.never_sufficient:
        problems.append(
            f"never_sufficient must contain {BSS_VS_RAW_IMPLIED!r}: drift and the variance risk premium let a constant "
            "forecaster beat the raw implied probability (V11)"
        )
    for name in ("success", "futility", "d12_literal", "missing", "pnl_verdict", "power"):
        if not getattr(prereg, name):
            problems.append(f"{name} is empty")

    history = prereg.reference_history
    for name in ("use", "purge", "own_events"):
        if not getattr(history, name):
            problems.append(f"reference_history.{name} is empty")

    ablation = prereg.ablation_buckets_only
    if len(ablation.arms) != 2:
        problems.append(f"ablation_buckets_only.arms must name exactly two arms, got {len(ablation.arms)}")
    if not ablation.statistics:
        problems.append("ablation_buckets_only.statistics is empty")
    for name in ("expected", "interpretation", "command"):
        if not getattr(ablation, name):
            problems.append(f"ablation_buckets_only.{name} is empty")

    change = prereg.model_change
    for name in ("policy", "shadow_overlap", "procedure"):
        if not getattr(change, name):
            problems.append(f"model_change.{name} is empty")
    if not change.agreement_report:
        problems.append("model_change.agreement_report is empty")

    if problems:
        raise PreregError("invalid pre-registration: " + "; ".join(problems))


# ======================================================================================================================
# The test-defining hash (12.3: `eval power` records it, `register` re-checks it)
# ======================================================================================================================

TEST_DEFINING_KEYS: Final[tuple[str, ...]] = (
    "id",
    "primary_family",
    "primary_forecasts",
    "primary_variant",
    "primary_tier",
    "unit",
    "weighting",
    "references",
    "reference_min_events",
    "reference_fallback",
    "eligible_session",
    "bootstrap",
    "interval_candidates",
    "size_rule",
    "looks",
    "success",
    "futility",
    "missing",
)


def test_defining_hash(prereg: Prereg) -> str:
    """sha256 over the keys that DEFINE the test (12.3).

    Everything that changes what `eval power` simulated is in here; prose that does not (`sensitivity`, `secondary`,
    the ablation and model-change blocks) is not, so re-wording a caveat does not invalidate a committed size study.
    Alphas are hashed as integer parts per million: no float ever enters hashed material (Conventions, INV-24).
    """
    material: dict[str, Any] = {
        "id": prereg.id,
        "primary_family": list(prereg.primary_family),
        "primary_forecasts": prereg.primary_forecasts,
        "primary_variant": prereg.primary_variant,
        "primary_tier": prereg.primary_tier,
        "unit": prereg.unit,
        "weighting": prereg.weighting,
        "references": list(prereg.references),
        "reference_min_events": prereg.reference_min_events,
        "reference_fallback": prereg.reference_fallback,
        "eligible_session": prereg.eligible_session,
        "bootstrap": {
            "kind": prereg.bootstrap.kind,
            "reps": prereg.bootstrap.reps,
            "min_block": prereg.bootstrap.min_block,
            "interval": prereg.bootstrap.interval,
        },
        "interval_candidates": list(prereg.interval_candidates),
        "size_rule": prereg.size_rule,
        "looks": [{"n_sessions": look.n_sessions, "alpha_ppm": round(look.alpha * 1_000_000)} for look in prereg.looks],
        "success": prereg.success,
        "futility": prereg.futility,
        "missing": prereg.missing,
    }
    return canon.sha256_hex(canon.dumps_sorted(material))


# ======================================================================================================================
# prereg/power.v1.json (written by `eval power`, 12.3; required by `register`)
# ======================================================================================================================


class SizeEntry(msgspec.Struct, frozen=True, kw_only=True):
    """One empirical rejection rate: interval method x null forecaster x look."""

    interval: str
    null: str  # "N1:<reference>+noise" | "N2:constant_climatological" | "N3:mockjev_constants"
    look: int  # index into prereg.looks
    n_sessions: int
    alpha: float
    rejection_rate: float
    reps: int
    holds: bool  # rejection_rate <= alpha + 2 * sqrt(alpha * (1 - alpha) / reps)  (the `size_rule`)


class PowerEntry(msgspec.Struct, frozen=True, kw_only=True):
    interval: str
    look: int
    bss: float
    power: float
    reps: int


class WeekdayEntry(msgspec.Struct, frozen=True, kw_only=True):
    weekday: int  # 0 = Monday
    n_events: int
    frequency: float


class PowerFile(msgspec.Struct, frozen=True, kw_only=True):
    """`prereg/power.v1.json` (12.3): size table, power table, chosen interval, weekday QC and the test hash."""

    prereg_id: str
    test_hash: str  # test_defining_hash(prereg) at the time `eval power` ran; a mismatch means the file is STALE
    generated_at: str
    null_sim_reps: int
    size: tuple[SizeEntry, ...]
    power: tuple[PowerEntry, ...]
    chosen_interval: str | None  # the first candidate holding its size under ALL nulls at BOTH looks; None = none did
    weekday: tuple[WeekdayEntry, ...] = ()
    weekday_tail_ratio: float = 0.0
    flags: tuple[str, ...] = ()


def load_power_file(path: Path | str) -> PowerFile:
    """Read `prereg/power.v1.json`. A missing or malformed file is a `PreregError`: the size study cannot be assumed."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PreregError(f"{path} is missing: run `jevbot eval power` and commit its output before registering (12.1 `power`)") from exc
    try:
        return msgspec.json.decode(raw, type=PowerFile)
    except msgspec.MsgspecError as exc:
        raise PreregError(f"{path} is not a valid `eval power` output: {exc}") from exc


# ======================================================================================================================
# Registration (section 14 `jevbot eval prereg register`)
# ======================================================================================================================


def register(
    prereg_path: Path | str,
    *,
    registry: Registry,
    repo_root: Path | str,
    power_path: Path | str | None = None,
    reference_store: Path | str | None = None,
    data_dir: Path | str | None = None,
    weekday_tail_ratio_max: float = 1.6,
    now: datetime | None = None,
    git_runner: Runner = run_git,
) -> PreregRow:
    """Validate everything 12.1 demands and store `(id, sha256, git_commit, registered_at)` in the registry.

    Raises `PreregError` for every refusal of section 14: an uncommitted file, an empty or unverifiable
    `[prereg.reference_history]`, a missing or stale power file, `bootstrap.interval` different from the size-validated
    `chosen_interval`, no interval method holding its size, or the `EM_WEEKDAY_BIAS` flag.
    """
    prereg_path = Path(prereg_path)
    repo_root = Path(repo_root)
    loaded = load_prereg(prereg_path)
    spec = loaded.prereg

    # (1) the file must be committed -------------------------------------------------------------------------------
    if not git_path_committed(prereg_path, repo_root, runner=git_runner):
        raise PreregError(
            f"{prereg_path} is not committed: a registered pre-registration hash must point at a file a reader can "
            "check out. Commit it first, then register"
        )
    commit, _dirty = git_head(repo_root, runner=git_runner)
    if not commit:
        raise PreregError(f"{repo_root} is not a git repository with a HEAD commit; the registration would be unverifiable")

    # (2) the reference history must exist and verify ---------------------------------------------------------------
    history = spec.reference_history
    if not history.is_filled:
        missing = [name for name in ("run_id", "ledger_head", "data_manifest_hash") if not getattr(history, name)]
        raise PreregError(
            "[prereg.reference_history] is incomplete (" + ", ".join(missing) + "): both references warm-start from "
            "the registered Jev-free MockJev reference run; run "
            "`jevbot backtest run --purpose reference --decider mock` and fill the block before registering"
        )
    store_path = _resolve_reference_store(history.run_id, reference_store=reference_store, data_dir=data_dir)
    _verify_reference_store(store_path, history)

    # (3) - (5) the size study --------------------------------------------------------------------------------------
    power_file_path = Path(power_path) if power_path is not None else prereg_path.with_name(POWER_FILENAME)
    power = load_power_file(power_file_path)
    expected_hash = test_defining_hash(spec)
    if power.prereg_id != spec.id or power.test_hash != expected_hash:
        raise PreregError(
            f"{power_file_path} is stale: it was produced for prereg {power.prereg_id!r} / test hash "
            f"{power.test_hash[:12]}..., this file is {spec.id!r} / {expected_hash[:12]}.... Re-run `jevbot eval power`"
        )
    if power.chosen_interval is None:
        raise PreregError(
            f"{power_file_path}: no interval candidate holds its size under every null forecaster at both looks "
            f"({', '.join(spec.interval_candidates)}); the pre-registered test would be over-sized (12.3)"
        )
    if spec.bootstrap.interval != power.chosen_interval:
        raise PreregError(
            f"bootstrap.interval is {spec.bootstrap.interval!r} but the size-validated chosen_interval is "
            f"{power.chosen_interval!r}; `interval` MUST equal the first candidate that holds its size (12.1)"
        )
    if EM_WEEKDAY_BIAS in power.flags or power.weekday_tail_ratio > weekday_tail_ratio_max:
        raise PreregError(
            f"{power_file_path} flags {EM_WEEKDAY_BIAS} (weekday tail ratio {power.weekday_tail_ratio:g} > "
            f"{weekday_tail_ratio_max:g}): the 1-session expected-move thresholds of 5.3 would be miscalibrated"
        )

    return registry.put_prereg(
        prereg_id=spec.id,
        sha256=loaded.sha256,
        git_commit=commit,
        body_toml=loaded.body_toml,
        now=now,
    )


def _resolve_reference_store(run_id: str, *, reference_store: Path | str | None, data_dir: Path | str | None) -> Path:
    if reference_store is not None:
        return Path(reference_store)
    if data_dir is not None:
        return Path(data_dir) / "runs" / run_id / "run.sqlite"
    raise PreregError(
        "the reference-history run store must be verified before registering: pass its path (or the data directory "
        f"holding runs/{run_id}/run.sqlite)"
    )


def _verify_reference_store(store_path: Path, history: ReferenceHistorySpec) -> None:
    """The reference run store must exist, verify, and be exactly the run the pre-registration names (12.1)."""
    if not store_path.exists():
        raise PreregError(f"the reference-history run store {store_path} does not exist")
    with load.RunStore(store_path) as store:
        store.verify()
        head_seq, head_hash = store.head()
        if head_seq == 0:
            raise PreregError(f"the reference-history run store {store_path} is empty")
        if head_hash != history.ledger_head:
            raise PreregError(
                f"the reference-history run store {store_path} has ledger head {head_hash[:12]}..., the "
                f"pre-registration names {history.ledger_head[:12]}...: the training history has changed"
            )
        meta = store.meta
        if meta.run_id != history.run_id:
            raise PreregError(f"{store_path} is run {meta.run_id!r}, the pre-registration names {history.run_id!r}")
        if meta.purpose != "reference":
            raise PreregError(
                f"the reference history must be a run with purpose 'reference' (a Jev-free MockJev run over the "
                f"mirror); {meta.run_id} has purpose {meta.purpose!r}"
            )
        if meta.data_manifest_hash != history.data_manifest_hash:
            raise PreregError(
                f"{store_path} was produced from data manifest {meta.data_manifest_hash[:12]}..., the "
                f"pre-registration names {history.data_manifest_hash[:12]}..."
            )


# ======================================================================================================================
# Status of a registration (the report's `prereg {id, sha256, status}` block, 12.8)
# ======================================================================================================================


def prereg_status(row: PreregRow | None, loaded: LoadedPrereg | None) -> str:
    """`"registered"` | `"changed"` | `"unregistered"` | `"absent"` - what a report prints beside the prereg id."""
    if loaded is None:
        return "absent"
    if row is None:
        return "unregistered"
    return "registered" if row.sha256 == loaded.sha256 else "changed"


def forecast_is_prereg(*, registered_at: datetime | str | None, first_tier_a_at: datetime | str | None, status: str) -> bool:
    """12.1: forecasts are tagged `prereg = true` only if the first Tier A forecast POSTDATES the registration and the
    file hash is unchanged."""
    if status != "registered" or registered_at is None or first_tier_a_at is None:
        return False
    return _as_text(first_tier_a_at) >= _as_text(registered_at)


def _as_text(value: datetime | str) -> str:
    """RFC 3339 UTC text compares lexicographically in the same order as the instants it spells."""
    return canon.render_as_of(value) if isinstance(value, datetime) else value
