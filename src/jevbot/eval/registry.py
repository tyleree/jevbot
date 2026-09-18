"""The trial registry (`registry.sqlite`): DESIGN.md 12.6 and the schema of 13.5.

Every run, sweep point, threshold or wording change is registered **before** it starts; nothing is registered by hand; a
run missing from the registry cannot be reported. The registry is therefore not bookkeeping - it is the thing that makes
the Deflated Sharpe count `N` honest, and it is where four refusals live:

* **holdout guard** - `purpose = "tune"` is refused for any run whose window includes a session at or after the model's
  release date (`HoldoutViolation`); every look at Tier A / B outcomes inserts a `holdout_looks` row.
* **Step 0 gate** - `purpose = "tune"` / `"final"` is refused for a Jev decider whose (model, SDK version, entry
  question-set hashes) key has no `determinism`, `order` and `batch` probe records (G3, B3.5). `sync_step0()` imports
  those records from `$JEVBOT_DATA/probes/step0/records` (6.8).
* **scan-facts gate** - `purpose = "final"` is refused without recorded `data scan-candidates` facts for the current
  `candidate_config_hash`, or when any enabled (underlying, structure) is unsizeable at the lowest non-zero tier more
  often than `candidates.max_unsizeable_rate` in the latest scanned year (section 8). The paper boot applies the same
  rule at B3 and exits 2, so both gates raise `ConfigError`.
* **family scoping** - runs that are not strategy selection register under derived families (`#baseline`, `#shadow`,
  `#reference`, `#diagnostic`, `#ablation`) and carry excluding flags, so they can never inflate `N` (12.6).

`selection_trials()` implements the scope of `N` and of `Var(SR_n)` exactly as 12.6 words it; `eval/dsr.py` turns that
scope into SR0 / DSR / MinTRL and never re-derives the scope itself.
"""

import json
import sqlite3
import subprocess
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import msgspec

from jevbot import canon
from jevbot.config import Config
from jevbot.errors import ConfigError, EvalError, HoldoutViolation, PreregError
from jevbot.types import EvidenceTier, ProbeRecord, RunMeta

__all__ = [
    "DERIVED_FAMILY_SUFFIXES",
    "EXCLUDED_FLAGS",
    "EXCLUDED_FLAG_PREFIXES",
    "JEV_DECIDERS",
    "REGISTRY_FILENAME",
    "SCHEMA",
    "SELECTION_PURPOSES",
    "SELECTION_STATUSES",
    "STEP0_REQUIRED_SUITES",
    "HoldoutLookRow",
    "PreregRow",
    "Registry",
    "ScanFacts",
    "ScanRow",
    "SelectionScope",
    "TrialGates",
    "TrialResultRow",
    "TrialRow",
    "derived_family",
    "git_head",
    "git_path_committed",
    "installed_sdk_version",
    "is_selection_trial",
    "load_scan_facts",
    "scan_facts_problems",
]

REGISTRY_FILENAME: Final[str] = "registry.sqlite"

# 12.6: derived families, so a baseline sweep or a shadow store can never inflate N or dominate Var(SR_n).
DERIVED_FAMILY_SUFFIXES: Final[tuple[str, ...]] = ("#baseline", "#shadow", "#reference", "#diagnostic", "#ablation")

# 12.6: flags that take a run out of the strategy-selection count, whatever family it registered under.
EXCLUDED_FLAGS: Final[frozenset[str]] = frozenset({"placebo", "unmasked", "diagnostic", "shadow", "reference_history", "model_overlap"})
EXCLUDED_FLAG_PREFIXES: Final[tuple[str, ...]] = ("baseline:", "ablation:")

SELECTION_PURPOSES: Final[frozenset[str]] = frozenset({"tune", "validate", "final"})
# "a crashed or spend-stopped attempt was still a look at the data; a resumed run is one trial" (12.6)
SELECTION_STATUSES: Final[frozenset[str]] = frozenset({"completed", "failed", "abandoned"})

# deciders whose answers come from the pinned Jev model: Step 0 is about that model, not about MockJev's constants.
JEV_DECIDERS: Final[frozenset[str]] = frozenset({"live_jev", "replay_jev"})
STEP0_REQUIRED_SUITES: Final[tuple[str, ...]] = ("determinism", "order", "batch")

TRIAL_STATUSES: Final[tuple[str, ...]] = ("registered", "running", "completed", "failed", "abandoned")
# status moves registered -> running -> completed | failed | abandoned (13.5)
_ALLOWED_TRANSITIONS: Final[dict[str, frozenset[str]]] = {
    "registered": frozenset({"running", "failed", "abandoned"}),
    "running": frozenset({"running", "completed", "failed", "abandoned"}),
    "failed": frozenset({"running"}),  # --resume RUN_ID, same trial row
    "abandoned": frozenset(),
    "completed": frozenset(),
}

BANDS: Final[tuple[str, ...]] = ("orats", "worst", "mid")
# 7.6 / 12.6: tier_ppm is 0 | 500000 | 750000 | 1000000; the scan-facts gate reads the LOWEST NON-ZERO tier.
LOWEST_NONZERO_TIER_PPM: Final[int] = 500_000

SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS trials (trial_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL UNIQUE,
  registered_at TEXT NOT NULL, family TEXT NOT NULL,
  purpose TEXT NOT NULL, experiment TEXT NOT NULL, namespace TEXT NOT NULL, mode TEXT NOT NULL, decider TEXT NOT NULL, model TEXT NOT NULL,
  config_hash TEXT NOT NULL, state_config_hash TEXT NOT NULL, rules_hash TEXT NOT NULL, risk_config_hash TEXT NOT NULL,
  qset_hashes TEXT NOT NULL, data_manifest_hash TEXT NOT NULL,
  fidelity TEXT NOT NULL, fill_rule TEXT NOT NULL, spot_measure TEXT NOT NULL, git_commit TEXT, git_dirty INTEGER NOT NULL,
  start TEXT NOT NULL, "end" TEXT NOT NULL, touches_holdout INTEGER NOT NULL, flags TEXT NOT NULL, status TEXT NOT NULL, notes TEXT);
CREATE TABLE IF NOT EXISTS trial_results (run_id TEXT PRIMARY KEY REFERENCES trials(run_id), finished_at TEXT NOT NULL,
  n_days INTEGER NOT NULL,
  sharpe_orats REAL, sharpe_worst REAL, sharpe_mid REAL, skew REAL, kurt REAL, ledger_head TEXT NOT NULL, cache_manifest_hash TEXT);
CREATE TABLE IF NOT EXISTS prereg (prereg_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, git_commit TEXT NOT NULL,
  registered_at TEXT NOT NULL, body_toml TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS holdout_looks (look_id INTEGER PRIMARY KEY, at TEXT NOT NULL, namespace TEXT NOT NULL, tiers TEXT NOT NULL,
  n_sessions INTEGER, report_path TEXT, prereg_look INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS step0_records (suite TEXT NOT NULL, model TEXT NOT NULL, sdk_version TEXT NOT NULL,
  entry_qset_hash TEXT NOT NULL, entry_text_qset_hash TEXT NOT NULL,
  verdict_json TEXT NOT NULL, run_dir TEXT NOT NULL, recorded_at TEXT NOT NULL,
  PRIMARY KEY (suite, model, sdk_version, entry_qset_hash, entry_text_qset_hash));
CREATE INDEX IF NOT EXISTS trials_by_family ON trials(family, namespace);
CREATE TRIGGER IF NOT EXISTS trials_no_delete BEFORE DELETE ON trials
  BEGIN SELECT RAISE(ABORT, 'the trial registry is append-only'); END;
CREATE TRIGGER IF NOT EXISTS trial_results_no_delete BEFORE DELETE ON trial_results
  BEGIN SELECT RAISE(ABORT, 'the trial registry is append-only'); END;
CREATE TRIGGER IF NOT EXISTS prereg_no_delete BEFORE DELETE ON prereg
  BEGIN SELECT RAISE(ABORT, 'a pre-registration is never deleted'); END;
CREATE TRIGGER IF NOT EXISTS step0_records_no_delete BEFORE DELETE ON step0_records
  BEGIN SELECT RAISE(ABORT, 'Step 0 records are never deleted'); END;
"""


# ======================================================================================================================
# Rows
# ======================================================================================================================


class TrialRow(msgspec.Struct, frozen=True, kw_only=True):
    trial_id: int
    run_id: str
    registered_at: str
    family: str
    purpose: str
    experiment: str
    namespace: str
    mode: str
    decider: str
    model: str
    config_hash: str
    state_config_hash: str
    rules_hash: str
    risk_config_hash: str
    qset_hashes: dict[str, str]
    data_manifest_hash: str
    fidelity: str
    fill_rule: str
    spot_measure: str
    git_commit: str | None
    git_dirty: bool
    start: date
    end: date | None  # None = an open-ended (paper) window; stored as "" (the column is NOT NULL)
    touches_holdout: bool
    flags: tuple[str, ...]
    status: str
    notes: str


class TrialResultRow(msgspec.Struct, frozen=True, kw_only=True):
    run_id: str
    finished_at: str
    n_days: int
    sharpe_orats: float | None
    sharpe_worst: float | None
    sharpe_mid: float | None
    skew: float | None
    kurt: float | None  # the PEARSON (non-excess) kurtosis, Normal = 3 (12.6)
    ledger_head: str
    cache_manifest_hash: str | None


class PreregRow(msgspec.Struct, frozen=True, kw_only=True):
    prereg_id: str
    sha256: str
    git_commit: str
    registered_at: str
    body_toml: str


class HoldoutLookRow(msgspec.Struct, frozen=True, kw_only=True):
    look_id: int
    at: str
    namespace: str
    tiers: tuple[str, ...]
    n_sessions: int | None
    report_path: str | None
    prereg_look: bool


class SelectionScope(msgspec.Struct, frozen=True, kw_only=True):
    """The scope of `N` and `Var(SR_n)` for one reported run (12.6) - nothing else.

    `n` counts registered selection trials of the same family and namespace whose window overlaps the reported run's,
    `completed` / `failed` / `abandoned` alike. `sharpes` holds the daily Sharpe of the **completed** subset only, per
    band; `eval/dsr.py` takes its variance.
    """

    family: str
    namespace: str
    start: date
    end: date | None
    n: int
    completed: int
    failed: int
    abandoned: int
    run_ids: tuple[str, ...]
    completed_run_ids: tuple[str, ...]
    sharpes: dict[str, tuple[float, ...]]

    @property
    def composition(self) -> str:
        """`N` with its composition, as the report prints it (12.6)."""
        return f"{self.n} (completed {self.completed} / failed {self.failed} / abandoned {self.abandoned})"


# ======================================================================================================================
# `data scan-candidates` facts (section 8, `manifests/scan_candidates.json`) - the `purpose = "final"` gate
# ======================================================================================================================


class ScanRow(msgspec.Struct, frozen=True, kw_only=True):
    """One (underlying, structure kind, year) row of the scan facts. Unknown keys are ignored on purpose: the writer
    (`data scan-candidates`, WP01 / WP04) reports more per row than this gate needs."""

    underlying: str
    kind: str
    year: int
    n_candidates: int = 0
    exceeds_risk_budget_rate: float = 0.0
    # tier_ppm rendered as a string -> share of (session, underlying) pairs that could not be sized at that tier
    unsizeable_rate_by_tier: dict[str, float] = msgspec.field(default_factory=dict)


class ScanFacts(msgspec.Struct, frozen=True, kw_only=True):
    """The facts recorded for ONE `candidate_config_hash`."""

    candidate_config_hash: str = ""
    created_at: str = ""
    initial_equity_usd: int = 0
    rows: tuple[ScanRow, ...] = ()

    def latest_year(self) -> int | None:
        return max((row.year for row in self.rows), default=None)


def load_scan_facts(path: Path, candidate_config_hash: str) -> ScanFacts | None:
    """Read `manifests/scan_candidates.json` (a mapping candidate_config_hash -> facts) and return one entry.

    Returns None when the file, or that key, is absent - which the `final` gate treats as "no facts recorded".
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        by_hash = msgspec.json.decode(raw, type=dict[str, ScanFacts])
    except msgspec.MsgspecError as exc:
        raise ConfigError(f"{path} is not a scan-candidates facts file: {exc}") from exc
    facts = by_hash.get(candidate_config_hash)
    if facts is None:
        return None
    if not facts.candidate_config_hash:
        facts = msgspec.structs.replace(facts, candidate_config_hash=candidate_config_hash)
    return facts


def scan_facts_problems(
    facts: ScanFacts | None,
    *,
    enabled_pairs: Iterable[tuple[str, str]],
    max_unsizeable_rate: float,
    tier_ppm: int = LOWEST_NONZERO_TIER_PPM,
) -> list[str]:
    """Section 8 / 11.2 B3: every reason the recorded facts do not license a `final` run (empty list = they do).

    Checked in the **latest scanned year**: facts exist at all, every enabled (underlying, kind) has a row there, and
    none of them is unsizeable at the lowest non-zero tier more often than `candidates.max_unsizeable_rate`.
    """
    pairs = sorted(set(enabled_pairs))
    if facts is None or not facts.rows:
        return ["no `data scan-candidates` facts recorded for the current candidate_config_hash"]
    year = facts.latest_year()
    by_pair = {(row.underlying, row.kind): row for row in facts.rows if row.year == year}
    problems: list[str] = []
    for pair in pairs:
        row = by_pair.get(pair)
        if row is None:
            problems.append(f"{pair[0]}/{pair[1]}: no scan facts in the latest scanned year ({year})")
            continue
        rate = row.unsizeable_rate_by_tier.get(str(tier_ppm))
        if rate is None:
            problems.append(f"{pair[0]}/{pair[1]}: no unsizeable rate recorded for tier {tier_ppm} ppm in {year}")
        elif rate > max_unsizeable_rate:
            problems.append(
                f"{pair[0]}/{pair[1]}: unsizeable at tier {tier_ppm} ppm in {rate:.1%} of {year} (limit {max_unsizeable_rate:.1%})"
            )
    return problems


# ======================================================================================================================
# Gate inputs
# ======================================================================================================================


class TrialGates(msgspec.Struct, frozen=True, kw_only=True):
    """Everything `register_trial()` needs beyond `RunMeta` to run the Step 0 and scan-facts gates.

    The defaults are deliberately inert: an unregistered `validate` run needs none of this, and a `tune` / `final` run
    that passes nothing is refused rather than silently waved through (the gates fail closed).
    """

    sdk_version: str = ""  # part of the Step 0 key: a record taken on another SDK version never satisfies it
    candidate_config_hash: str = ""
    enabled_pairs: tuple[tuple[str, str], ...] = ()  # (underlying, StructureKind value)
    max_unsizeable_rate: float = 0.25
    scan_facts: ScanFacts | None = None

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        *,
        candidate_config_hash: str,
        data_dir: Path,
        sdk_version: str | None = None,
    ) -> "TrialGates":
        """Build the gate inputs from a resolved config: the enabled (underlying, kind) pairs, the unsizeable-rate limit
        and the recorded scan facts of `manifests/scan_candidates.json`."""
        pairs = tuple((underlying, kind.value) for underlying in cfg.universe.underlyings for kind in cfg.structures.enabled)
        facts = load_scan_facts(Path(data_dir) / "manifests" / "scan_candidates.json", candidate_config_hash)
        return cls(
            sdk_version=sdk_version if sdk_version is not None else installed_sdk_version(),
            candidate_config_hash=candidate_config_hash,
            enabled_pairs=pairs,
            max_unsizeable_rate=cfg.candidates.max_unsizeable_rate,
            scan_facts=facts,
        )


def installed_sdk_version() -> str:
    """The installed `typesafe-sdk` distribution version, read from the package metadata.

    The import rules of section 1 allow only `jev/live.py` and `jev/probe.py` to import `typesafe_sdk`; reading the
    distribution's metadata imports nothing.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("typesafe-sdk")
    except PackageNotFoundError:  # pragma: no cover - the pin is a hard dependency of this project
        return ""


# ======================================================================================================================
# git helpers (the `git_dirty` column of 13.5 and the prereg's "uncommitted file" refusal of section 14)
# ======================================================================================================================

Runner = Callable[[Sequence[str], Path], "subprocess.CompletedProcess[str]"]


def _run_git(args: Sequence[str], cwd: Path) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(  # fixed argv, no shell
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def git_head(repo_root: Path, *, runner: Runner = _run_git) -> tuple[str | None, bool]:
    """`(commit, dirty)` of the working tree at `repo_root`; `(None, True)` when it is not a git repository.

    A tree that git cannot describe is treated as dirty: the report's `UNREPRODUCIBLE (dirty git tree)` flag must never
    be missing because a command failed.
    """
    head = runner(["rev-parse", "HEAD"], repo_root)
    if head.returncode != 0:
        return None, True
    status = runner(["status", "--porcelain"], repo_root)
    if status.returncode != 0:  # pragma: no cover - rev-parse succeeded, so status does too
        return head.stdout.strip() or None, True
    return head.stdout.strip() or None, bool(status.stdout.strip())


def git_path_committed(path: Path, repo_root: Path, *, runner: Runner = _run_git) -> bool:
    """True iff `path` is tracked by git AND has no staged or unstaged modification (section 14: `eval prereg register`
    refuses an uncommitted pre-registration file - a registered hash must point at something a reader can check out)."""
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    tracked = runner(["ls-files", "--error-unmatch", "--", relative.as_posix()], repo_root)
    if tracked.returncode != 0:
        return False
    status = runner(["status", "--porcelain", "--", relative.as_posix()], repo_root)
    if status.returncode != 0:  # pragma: no cover - ls-files succeeded, so status does too
        return False
    return not status.stdout.strip()


# ======================================================================================================================
# Families
# ======================================================================================================================


def derived_family(family: str, suffix: str) -> str:
    """`<family>#baseline` and friends (12.6). Appending a suffix twice is a no-op, so a caller that registers a shadow
    store of an already-derived family cannot build `exp#shadow#shadow`."""
    if suffix not in DERIVED_FAMILY_SUFFIXES:
        raise ValueError(f"unknown derived family suffix {suffix!r}; expected one of {DERIVED_FAMILY_SUFFIXES}")
    base = family.split("#", 1)[0]
    return f"{base}{suffix}"


def is_selection_trial(*, purpose: str, family: str, flags: Iterable[str]) -> bool:
    """True iff this run counts toward the strategy-selection trial count `N` (12.6)."""
    if purpose not in SELECTION_PURPOSES:
        return False
    if any(family.endswith(suffix) for suffix in DERIVED_FAMILY_SUFFIXES):
        return False
    for flag in flags:
        if flag in EXCLUDED_FLAGS or flag.startswith(EXCLUDED_FLAG_PREFIXES):
            return False
    return True


# ======================================================================================================================
# The registry
# ======================================================================================================================


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _render_date(value: date | None) -> str:
    return canon.render_session(value) if value is not None else ""


def _parse_date(value: str) -> date | None:
    return date.fromisoformat(value) if value else None


class Registry:
    """`registry.sqlite` (13.5). Open it with `Registry.open(data_dir)`; it is a context manager."""

    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        if read_only:
            self._conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if not read_only:
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = FULL")
            self._conn.execute("PRAGMA busy_timeout = 5000")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    @classmethod
    def open(cls, data_dir: Path, *, read_only: bool = False) -> "Registry":
        return cls(Path(data_dir) / REGISTRY_FILENAME, read_only=read_only)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Registry":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- registration ---------------------------------------------------------------------------------------------

    def register_trial(
        self,
        meta: RunMeta,
        *,
        git_commit: str | None,
        git_dirty: bool,
        gates: TrialGates | None = None,
        notes: str = "",
        now: datetime | None = None,
    ) -> int:
        """Register a run BEFORE it starts and return its `trial_id` (12.6).

        Raises `HoldoutViolation` for a `tune` run that touches post-release sessions, and `ConfigError` when the Step 0
        or scan-facts gate of 12.1 / section 8 is not satisfied.
        """
        gates = gates or TrialGates()
        family = meta.family or meta.experiment
        if not family:
            raise EvalError(f"trial {meta.run_id}: a trial family is required (RunMeta.family or run.experiment)")
        touches_holdout = meta.end is None or meta.end >= meta.model_release_date

        # --- holdout guard (12.1) ---------------------------------------------------------------------------------
        if meta.purpose == "tune" and touches_holdout:
            raise HoldoutViolation(
                f"trial {meta.run_id}: purpose 'tune' is refused on a window that reaches "
                f"{_render_date(meta.end) or 'the open end'} - the model was released on "
                f"{_render_date(meta.model_release_date)}; tune strictly before that date"
            )

        # --- Step 0 gate (G3, B3.5) -------------------------------------------------------------------------------
        if meta.purpose in ("tune", "final") and meta.decider in JEV_DECIDERS:
            have = self.step0_suites(
                model=meta.model,
                sdk_version=gates.sdk_version,
                entry_qset_hash=meta.entry_qset_hash,
                entry_text_qset_hash=meta.entry_text_qset_hash,
            )
            missing = [suite for suite in STEP0_REQUIRED_SUITES if suite not in have]
            if missing:
                raise ConfigError(
                    f"trial {meta.run_id}: purpose {meta.purpose!r} with decider {meta.decider!r} needs the Step 0 "
                    f"records of THIS model and question sets (model={meta.model!r}, sdk={gates.sdk_version!r}); "
                    f"missing suites: {', '.join(missing)}. Run `jevbot jev probe-step0` and "
                    f"`jevbot eval registry sync-step0` first"
                )

        # --- scan-facts gate (section 8, 11.2 B3) -------------------------------------------------------------------
        if meta.purpose == "final":
            problems = scan_facts_problems(
                gates.scan_facts,
                enabled_pairs=gates.enabled_pairs,
                max_unsizeable_rate=gates.max_unsizeable_rate,
            )
            if problems:
                raise ConfigError(
                    f"trial {meta.run_id}: purpose 'final' needs sizeable `data scan-candidates` facts for "
                    f"candidate_config_hash {gates.candidate_config_hash!r}: " + "; ".join(problems)
                )

        qset_hashes = {
            "entry": meta.entry_qset_hash,
            "entry_text": meta.entry_text_qset_hash,
            "manage": meta.manage_qset_hash,
            "manage_text": meta.manage_text_qset_hash,
        }
        try:
            cursor = self._conn.execute(
                "INSERT INTO trials (run_id, registered_at, family, purpose, experiment, namespace, mode, decider, model,"
                " config_hash, state_config_hash, rules_hash, risk_config_hash, qset_hashes, data_manifest_hash,"
                ' fidelity, fill_rule, spot_measure, git_commit, git_dirty, start, "end", touches_holdout, flags, status, notes)'
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    meta.run_id,
                    canon.render_as_of(now or _utcnow()),
                    family,
                    meta.purpose,
                    meta.experiment,
                    meta.namespace,
                    meta.mode.value,
                    meta.decider,
                    meta.model,
                    meta.config_hash,
                    meta.state_config_hash,
                    meta.rules_hash,
                    meta.risk_config_hash,
                    json.dumps(qset_hashes, sort_keys=True),
                    meta.data_manifest_hash,
                    meta.fidelity.value,
                    meta.fill_rule.value,
                    meta.spot_measure,
                    git_commit,
                    int(git_dirty),
                    _render_date(meta.start),
                    _render_date(meta.end),
                    int(touches_holdout),
                    json.dumps(list(meta.flags)),
                    "registered",
                    notes,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise EvalError(
                f"trial {meta.run_id} is already registered: a resumed run keeps its trial row "
                f"(`--resume {meta.run_id}`), it is never registered twice"
            ) from exc
        self._conn.commit()
        trial_id = int(cursor.lastrowid or 0)
        return trial_id

    def set_status(self, run_id: str, status: str, *, notes: str | None = None) -> None:
        """Move a trial along `registered -> running -> completed | failed | abandoned` (13.5).

        `failed` carries its reason in `notes` ("failed:spend_limit", "failed:cache_miss", ...).
        """
        if status not in TRIAL_STATUSES:
            raise ValueError(f"unknown trial status {status!r}")
        row = self._conn.execute("SELECT status FROM trials WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise EvalError(f"trial {run_id} is not registered; a run missing from the registry cannot be reported")
        current = str(row["status"])
        if status != current and status not in _ALLOWED_TRANSITIONS[current]:
            raise EvalError(f"trial {run_id}: refusing the status move {current!r} -> {status!r} (13.5)")
        if notes is None:
            self._conn.execute("UPDATE trials SET status = ? WHERE run_id = ?", (status, run_id))
        else:
            self._conn.execute("UPDATE trials SET status = ?, notes = ? WHERE run_id = ?", (status, notes, run_id))
        self._conn.commit()

    def resume(self, run_id: str, *, notes: str | None = None) -> TrialRow:
        """`--resume RUN_ID`: move `failed` | `running` back to `running` in the SAME trial row - a resumed run is one
        trial, counted once in `N` (12.6, 13.5)."""
        row = self.trial(run_id)
        if row is None:
            raise EvalError(f"trial {run_id} is not registered; it cannot be resumed")
        if row.status not in ("failed", "running"):
            raise EvalError(f"trial {run_id}: only a failed or running trial can be resumed (status {row.status!r})")
        self.set_status(run_id, "running", notes=notes)
        resumed = self.trial(run_id)
        if resumed is None:  # pragma: no cover - the row was read one statement earlier
            raise EvalError(f"trial {run_id} disappeared while it was resumed")
        return resumed

    def record_result(
        self,
        run_id: str,
        *,
        n_days: int,
        ledger_head: str,
        sharpe_orats: float | None = None,
        sharpe_worst: float | None = None,
        sharpe_mid: float | None = None,
        skew: float | None = None,
        kurt: float | None = None,
        cache_manifest_hash: str | None = None,
        finished_at: datetime | None = None,
    ) -> None:
        """Store the finished run's headline statistics. `kurt` is the PEARSON (non-excess) kurtosis (12.6)."""
        if self.trial(run_id) is None:
            raise EvalError(f"trial {run_id} is not registered; its results cannot be stored")
        self._conn.execute(
            "INSERT INTO trial_results (run_id, finished_at, n_days, sharpe_orats, sharpe_worst, sharpe_mid, skew, kurt,"
            " ledger_head, cache_manifest_hash) VALUES (?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(run_id) DO UPDATE SET finished_at = excluded.finished_at, n_days = excluded.n_days,"
            " sharpe_orats = excluded.sharpe_orats, sharpe_worst = excluded.sharpe_worst, sharpe_mid = excluded.sharpe_mid,"
            " skew = excluded.skew, kurt = excluded.kurt, ledger_head = excluded.ledger_head,"
            " cache_manifest_hash = excluded.cache_manifest_hash",
            (
                run_id,
                canon.render_as_of(finished_at or _utcnow()),
                n_days,
                sharpe_orats,
                sharpe_worst,
                sharpe_mid,
                skew,
                kurt,
                ledger_head,
                cache_manifest_hash,
            ),
        )
        self._conn.commit()

    # --- reads ----------------------------------------------------------------------------------------------------

    def trial(self, run_id: str) -> TrialRow | None:
        row = self._conn.execute("SELECT * FROM trials WHERE run_id = ?", (run_id,)).fetchone()
        return _trial_row(row) if row is not None else None

    def trials(self, *, family: str | None = None, namespace: str | None = None, status: str | None = None) -> list[TrialRow]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (("family", family), ("namespace", namespace), ("status", status)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(f"SELECT * FROM trials{where} ORDER BY trial_id", params).fetchall()
        return [_trial_row(row) for row in rows]

    def result(self, run_id: str) -> TrialResultRow | None:
        row = self._conn.execute("SELECT * FROM trial_results WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return TrialResultRow(
            run_id=str(row["run_id"]),
            finished_at=str(row["finished_at"]),
            n_days=int(row["n_days"]),
            sharpe_orats=_opt_float(row["sharpe_orats"]),
            sharpe_worst=_opt_float(row["sharpe_worst"]),
            sharpe_mid=_opt_float(row["sharpe_mid"]),
            skew=_opt_float(row["skew"]),
            kurt=_opt_float(row["kurt"]),
            ledger_head=str(row["ledger_head"]),
            cache_manifest_hash=None if row["cache_manifest_hash"] is None else str(row["cache_manifest_hash"]),
        )

    # --- the DSR scope (12.6) -------------------------------------------------------------------------------------

    def selection_trials(self, *, family: str, namespace: str, start: date, end: date | None) -> SelectionScope:
        """`N`, its composition and the completed subset's Sharpes for one reported run's family / namespace / window.

        A trial counts when its purpose is one of `tune` / `validate` / `final`, its status is `completed`, `failed` or
        `abandoned`, it carries no excluding flag, its family is not a derived one, and its window overlaps the reported
        run's. `Var(SR_n)` is then taken over the completed subset by `eval/dsr.py`.
        """
        rows = [_trial_row(row) for row in self._conn.execute("SELECT * FROM trials ORDER BY trial_id").fetchall()]
        scoped: list[TrialRow] = []
        for row in rows:
            if row.family != family or row.namespace != namespace:
                continue
            if row.status not in SELECTION_STATUSES:
                continue
            if not is_selection_trial(purpose=row.purpose, family=row.family, flags=row.flags):
                continue
            if not _windows_overlap(row.start, row.end, start, end):
                continue
            scoped.append(row)
        completed = [row for row in scoped if row.status == "completed"]
        sharpes: dict[str, tuple[float, ...]] = {}
        for band in BANDS:
            values: list[float] = []
            for row in completed:
                result = self.result(row.run_id)
                if result is None:
                    continue
                value = getattr(result, f"sharpe_{band}")
                if value is not None:
                    values.append(float(value))
            sharpes[band] = tuple(values)
        return SelectionScope(
            family=family,
            namespace=namespace,
            start=start,
            end=end,
            n=len(scoped),
            completed=len(completed),
            failed=sum(1 for row in scoped if row.status == "failed"),
            abandoned=sum(1 for row in scoped if row.status == "abandoned"),
            run_ids=tuple(row.run_id for row in scoped),
            completed_run_ids=tuple(row.run_id for row in completed),
            sharpes=sharpes,
        )

    # --- Step 0 records (6.8) -------------------------------------------------------------------------------------

    def sync_step0(self, data_dir: Path) -> int:
        """Import `$JEVBOT_DATA/probes/step0/records/*.json` into `step0_records` and return the number of rows written.

        A record is keyed by (suite, model, SDK version, entry question-set hashes): a Step 0 taken on another model id,
        SDK version or question wording never satisfies this namespace (B3.5). An unreadable record is skipped, never
        guessed at - the gate then simply stays closed.
        """
        records_dir = Path(data_dir) / "probes" / "step0" / "records"
        if not records_dir.is_dir():
            return 0
        written = 0
        for path in sorted(records_dir.glob("*.json")):
            try:
                record = msgspec.json.decode(path.read_bytes(), type=ProbeRecord)
            except (OSError, msgspec.MsgspecError):
                continue
            written += self.put_step0(record)
        self._conn.commit()
        return written

    def put_step0(self, record: ProbeRecord) -> int:
        """Insert one probe record; a newer record for the same key replaces the older verdict. Returns 1 when the table
        changed, 0 when an equally old or newer record was already there."""
        cursor = self._conn.execute(
            "INSERT INTO step0_records (suite, model, sdk_version, entry_qset_hash, entry_text_qset_hash, verdict_json,"
            " run_dir, recorded_at) VALUES (?,?,?,?,?,?,?,?)"
            " ON CONFLICT(suite, model, sdk_version, entry_qset_hash, entry_text_qset_hash) DO UPDATE SET"
            " verdict_json = excluded.verdict_json, run_dir = excluded.run_dir, recorded_at = excluded.recorded_at"
            " WHERE excluded.recorded_at > step0_records.recorded_at",
            (
                record.suite,
                record.model,
                record.sdk_version,
                record.entry_qset_hash,
                record.entry_text_qset_hash,
                json.dumps(record.verdict, sort_keys=True),
                record.run_dir,
                canon.render_as_of(record.recorded_at),
            ),
        )
        self._conn.commit()
        return int(cursor.rowcount or 0)

    def step0_suites(self, *, model: str, sdk_version: str, entry_qset_hash: str, entry_text_qset_hash: str) -> frozenset[str]:
        """The suites recorded for EXACTLY this key (6.8). All four fields must match: no partial credit."""
        rows = self._conn.execute(
            "SELECT suite FROM step0_records WHERE model = ? AND sdk_version = ? AND entry_qset_hash = ? AND entry_text_qset_hash = ?",
            (model, sdk_version, entry_qset_hash, entry_text_qset_hash),
        ).fetchall()
        return frozenset(str(row["suite"]) for row in rows)

    # --- pre-registration (12.1) ----------------------------------------------------------------------------------

    def put_prereg(self, *, prereg_id: str, sha256: str, git_commit: str, body_toml: str, now: datetime | None = None) -> PreregRow:
        """Store `(id, sha256, git_commit, registered_at)` plus the body (12.1).

        Registering the same id again with the SAME hash is an idempotent no-op; with a different hash it is refused -
        a changed pre-registration is a new pre-registration, never a silent amendment of the old one.
        """
        existing = self.get_prereg(prereg_id)
        if existing is not None:
            if existing.sha256 != sha256:
                raise PreregError(
                    f"pre-registration {prereg_id!r} is already registered with sha256 {existing.sha256} "
                    f"(registered_at {existing.registered_at}); a changed file needs a NEW prereg id"
                )
            return existing
        self._conn.execute(
            "INSERT INTO prereg (prereg_id, sha256, git_commit, registered_at, body_toml) VALUES (?,?,?,?,?)",
            (prereg_id, sha256, git_commit, canon.render_as_of(now or _utcnow()), body_toml),
        )
        self._conn.commit()
        stored = self.get_prereg(prereg_id)
        if stored is None:  # pragma: no cover - the row was inserted one statement earlier
            raise PreregError(f"pre-registration {prereg_id!r} could not be stored")
        return stored

    def get_prereg(self, prereg_id: str) -> PreregRow | None:
        row = self._conn.execute("SELECT * FROM prereg WHERE prereg_id = ?", (prereg_id,)).fetchone()
        if row is None:
            return None
        return PreregRow(
            prereg_id=str(row["prereg_id"]),
            sha256=str(row["sha256"]),
            git_commit=str(row["git_commit"]),
            registered_at=str(row["registered_at"]),
            body_toml=str(row["body_toml"]),
        )

    # --- holdout looks (12.1) -------------------------------------------------------------------------------------

    def add_holdout_look(
        self,
        *,
        namespace: str,
        tiers: Iterable[EvidenceTier | str],
        n_sessions: int | None = None,
        report_path: str | None = None,
        prereg_look: bool = False,
        at: datetime | None = None,
    ) -> int:
        """Every report that includes Tier A / B outcomes inserts one row here and prints the count in its header."""
        labels = sorted({tier.value if isinstance(tier, EvidenceTier) else str(tier) for tier in tiers})
        cursor = self._conn.execute(
            "INSERT INTO holdout_looks (at, namespace, tiers, n_sessions, report_path, prereg_look) VALUES (?,?,?,?,?,?)",
            (canon.render_as_of(at or _utcnow()), namespace, ",".join(labels), n_sessions, report_path, int(prereg_look)),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def count_holdout_looks(self, namespace: str | None = None, *, prereg_only: bool = False) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if namespace is not None:
            clauses.append("namespace = ?")
            params.append(namespace)
        if prereg_only:
            clauses.append("prereg_look = 1")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        row = self._conn.execute(f"SELECT COUNT(*) AS n FROM holdout_looks{where}", params).fetchone()
        return int(row["n"])

    def holdout_looks(self, namespace: str | None = None) -> list[HoldoutLookRow]:
        where = " WHERE namespace = ?" if namespace is not None else ""
        params: tuple[Any, ...] = (namespace,) if namespace is not None else ()
        rows = self._conn.execute(f"SELECT * FROM holdout_looks{where} ORDER BY look_id", params).fetchall()
        return [
            HoldoutLookRow(
                look_id=int(row["look_id"]),
                at=str(row["at"]),
                namespace=str(row["namespace"]),
                tiers=tuple(part for part in str(row["tiers"]).split(",") if part),
                n_sessions=None if row["n_sessions"] is None else int(row["n_sessions"]),
                report_path=None if row["report_path"] is None else str(row["report_path"]),
                prereg_look=bool(row["prereg_look"]),
            )
            for row in rows
        ]


# ======================================================================================================================
# helpers
# ======================================================================================================================


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _trial_row(row: sqlite3.Row) -> TrialRow:
    return TrialRow(
        trial_id=int(row["trial_id"]),
        run_id=str(row["run_id"]),
        registered_at=str(row["registered_at"]),
        family=str(row["family"]),
        purpose=str(row["purpose"]),
        experiment=str(row["experiment"]),
        namespace=str(row["namespace"]),
        mode=str(row["mode"]),
        decider=str(row["decider"]),
        model=str(row["model"]),
        config_hash=str(row["config_hash"]),
        state_config_hash=str(row["state_config_hash"]),
        rules_hash=str(row["rules_hash"]),
        risk_config_hash=str(row["risk_config_hash"]),
        qset_hashes=dict(json.loads(str(row["qset_hashes"]))),
        data_manifest_hash=str(row["data_manifest_hash"]),
        fidelity=str(row["fidelity"]),
        fill_rule=str(row["fill_rule"]),
        spot_measure=str(row["spot_measure"]),
        git_commit=None if row["git_commit"] is None else str(row["git_commit"]),
        git_dirty=bool(row["git_dirty"]),
        start=date.fromisoformat(str(row["start"])),
        end=_parse_date(str(row["end"])),
        touches_holdout=bool(row["touches_holdout"]),
        flags=tuple(json.loads(str(row["flags"]))),
        status=str(row["status"]),
        notes="" if row["notes"] is None else str(row["notes"]),
    )


def _windows_overlap(a_start: date, a_end: date | None, b_start: date, b_end: date | None) -> bool:
    """Closed intervals, an open end (`None`) reaching forever - the 12.6 "window that overlaps the reported run's"."""
    if a_end is not None and a_end < b_start:
        return False
    if b_end is not None and b_end < a_start:
        return False
    return True
