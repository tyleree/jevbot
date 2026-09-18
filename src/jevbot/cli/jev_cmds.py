"""The `jev` and `cache` sub-apps (DESIGN.md section 14; WP03).

`jev probe-step0` runs the Step 0 suites of 6.8 and writes the keyed probe records that gate `tune` / `final` trials and the
paper news rule. `jev show-request` prints the exact bytes a request would carry, offline. `jev questions` prints the wire
JSON and the hashes of a question set - the review aid a wording change is diffed with. `cache stats` / `cache verify` read
the decision cache of 13.3.

Two commands reach outside WP03 and both do it the documented way (section 16): the dependency is imported **inside the
command function** through `_lazy_attribute`, and a missing module prints "not built yet" instead of breaking the CLI.
`jev show-request` needs `jevbot.cycle.preview_request` (WP09) and `jev probe-step0 --states-from mirror` needs
`jevbot.cycle.sample_entry_states`; `--states-from run:RUN_ID` needs `jevbot.ledger.SqliteLedger` (WP06). `file:` states need
nothing beyond WP00 and WP03, which is what this package's own tests use.
"""

import importlib
import json as json_module
import logging
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Final, NoReturn

import typer

from jevbot import questions as questions_module
from jevbot.cli.main import NOT_BUILT, get_globals
from jevbot.errors import ConfigError, ExitCode
from jevbot.jev import probe as probe_module
from jevbot.jev.cache import SqliteDecisionCache
from jevbot.jev.spend import SCOPES, SpendGuard
from jevbot.types import RequestKind, Variant

__all__ = ["YES_SPEND_TOKENS", "app", "cache_app"]

_log = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True, help="Jev decider: Step 0 probes, show-request, questions.")
cache_app = typer.Typer(no_args_is_help=True, help="Decision cache: stats, verify.")

YES_SPEND_TOKENS: Final = 5_000_000  # above this a probe run needs --yes-spend (the `backtest run` rule of 14 / 6.7)
DEFAULT_MAX_TOKENS: Final = 7_000_000  # the documented Step 0 budget: about 2k requests, about $0.30


# ======================================================================================================================
# Lazy cross-package conveniences (section 16)
# ======================================================================================================================


def _lazy_attribute(module: str, attribute: str) -> Any | None:
    """The attribute of a module that another work package owns, or None when that module is not in this checkout."""
    try:
        loaded = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name is not None and (exc.name == module or module.startswith(exc.name + ".")):
            return None
        raise
    return getattr(loaded, attribute, None)


def _not_built(what: str, missing: str) -> NoReturn:
    typer.echo(f"jevbot {what}: {NOT_BUILT} (module {missing} is not part of this checkout)")
    raise typer.Exit(int(ExitCode.ERROR))


def _load_preview_request() -> Callable[..., Any] | None:
    return _lazy_attribute("jevbot.cycle", "preview_request")  # WP09


def _load_sample_entry_states() -> Callable[..., Any] | None:
    return _lazy_attribute("jevbot.cycle", "sample_entry_states")  # WP09


def _load_sqlite_ledger() -> Callable[..., Any] | None:
    return _lazy_attribute("jevbot.ledger", "SqliteLedger")  # WP06


# ======================================================================================================================
# jev questions
# ======================================================================================================================


@app.command("questions")
def questions_cmd(
    ctx: typer.Context,
    question_set: Annotated[
        str | None, typer.Option("--set", metavar="NAME", help="entry.v1 | entry_text.v1 | manage.v1 | manage_text.v1 | probe.recall.v1")
    ] = None,
    hashes: Annotated[bool, typer.Option("--hashes", help="Print the hashes only.")] = False,
) -> None:
    """Print the wire JSON and the hashes of the question sets (review aid; a wording change is diffed here)."""
    names = list(questions_module.QUESTION_SETS) if question_set is None else [question_set]
    for name in names:
        if name not in questions_module.QUESTION_SETS:
            raise ConfigError(f"unknown question set {name!r}: one of {', '.join(questions_module.QUESTION_SETS)}")
    as_json = get_globals(ctx).json
    payload = {
        name: {
            "question_set_hash": questions_module.QUESTION_SET_HASHES[name],
            "questions": {qid: questions_module.QUESTION_HASHES[qid] for qid in questions_module.QUESTION_SETS[name]}
            if hashes
            else questions_module.QUESTION_SETS[name],
        }
        for name in names
    }
    if as_json:
        typer.echo(json_module.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    for name in names:
        typer.echo(f"{name}  question_set_hash={questions_module.QUESTION_SET_HASHES[name]}")
        for qid in questions_module.QUESTION_SETS[name]:
            typer.echo(f"  {qid}  {questions_module.QUESTION_HASHES[qid]}")
            if not hashes:
                typer.echo("    " + json_module.dumps(questions_module.QUESTION_SETS[name][qid], ensure_ascii=False, sort_keys=True))


# ======================================================================================================================
# jev show-request
# ======================================================================================================================


@app.command("show-request")
def show_request_cmd(
    ctx: typer.Context,
    underlying: Annotated[str, typer.Option("--underlying", metavar="U", help="The underlying symbol.")],
    session: Annotated[datetime, typer.Option("--session", formats=["%Y-%m-%d"], help="The decision session (YYYY-MM-DD).")],
    kind: Annotated[RequestKind, typer.Option("--kind", help="entry | entry_text | manage | manage_text")] = RequestKind.ENTRY,
    variant: Annotated[Variant, typer.Option("--variant", help="base | opt_perm | key_perm | bucket_only")] = Variant.BASE,
) -> None:
    """Print the exact state and questions that would be sent, with their hashes (offline, no network, no spend)."""
    globals_ = get_globals(ctx)
    loaded = globals_.load()
    if loaded.cfg.state.unmasked and loaded.cfg.run.purpose != "diagnostic":
        raise ConfigError('state.unmasked is a leakage-diagnostic flag: `jev show-request` refuses it outside run.purpose = "diagnostic"')
    preview = _load_preview_request()
    if preview is None:
        _not_built("jev show-request", "jevbot.cycle")
    request = preview(loaded.cfg, underlying=underlying, session=session.date(), kind=kind, variant=variant)
    payload = {
        "kind": request.kind.value,
        "variant": request.variant.value,
        "question_set_id": request.question_set_id,
        "state_hash": request.state_hash,
        "question_set_hash": request.question_set_hash,
        "decision_id": request.decision_id,
        "namespace": request.namespace,
        "state": request.state,
        "questions": request.questions,
    }
    typer.echo(json_module.dumps(payload, ensure_ascii=False, indent=2))


# ======================================================================================================================
# jev probe-step0
# ======================================================================================================================


def _states_from(spec: str, data_dir: Path, *, seed: int, cfg: Any) -> probe_module.StateSource:
    """Resolve `--states-from`: `file:PATH` (ours), `run:RUN_ID` (WP06's ledger) or `mirror` (WP09), imported lazily."""
    if spec.startswith("file:"):
        return probe_module.file_state_source(spec[len("file:") :])
    if spec.startswith("run:"):
        run_id = spec[len("run:") :]
        ledger_class = _load_sqlite_ledger()
        if ledger_class is None:
            _not_built("jev probe-step0 --states-from run:", "jevbot.ledger")
        store = data_dir / "runs" / run_id / "run.sqlite"
        if not store.exists():
            raise ConfigError(f"run store {store} does not exist")

        def from_run(n: int) -> list[dict[str, Any]]:
            ledger = ledger_class(store, read_only=True)
            out: list[dict[str, Any]] = []
            for _session, _underlying, state_json in ledger.get_states("entry"):
                out.append(json_module.loads(state_json))
                if len(out) >= n:
                    break
            return out

        return from_run
    if spec == "mirror":
        sample = _load_sample_entry_states()
        if sample is None:
            _not_built("jev probe-step0 --states-from mirror", "jevbot.cycle")

        def from_mirror(n: int) -> Iterable[dict[str, Any]]:
            states: Iterable[dict[str, Any]] = sample(cfg, n, seed)
            return states

        return from_mirror
    raise ConfigError(f"--states-from must be file:PATH, run:RUN_ID or mirror, got {spec!r}")


@app.command("probe-step0")
def probe_step0_cmd(
    ctx: typer.Context,
    suite: Annotated[str, typer.Option("--suite", help="meta | determinism | batch | order | text | all")] = "all",
    states_from: Annotated[str, typer.Option("--states-from", metavar="SPEC", help="run:RUN_ID | file:PATH | mirror")] = "mirror",
    states: Annotated[int, typer.Option("--states", metavar="N", help="States per suite.")] = probe_module.DEFAULT_STATES,
    repeats: Annotated[
        int, typer.Option("--repeats", metavar="R", help="Repeats of the determinism suite.")
    ] = probe_module.DEFAULT_REPEATS,
    max_tokens: Annotated[int, typer.Option("--max-tokens", metavar="T", help="Input-token budget for this run.")] = DEFAULT_MAX_TOKENS,
    yes_spend: Annotated[bool, typer.Option("--yes-spend", help="Confirm a budget above 5M input tokens.")] = False,
) -> None:
    """Run the Step 0 live probes of 6.8 and write their keyed probe records (needs `TYPESAFE_API_KEY`)."""
    loaded = get_globals(ctx).load()
    wanted = list(probe_module.SUITES) if suite == "all" else [suite]
    for name in wanted:
        if name not in probe_module.SUITES:
            raise ConfigError(f"unknown suite {name!r}: one of {', '.join(probe_module.SUITES)}, or all")
    api_key = loaded.secrets.typesafe_api_key
    if api_key is None:
        raise ConfigError("the Step 0 probes need TYPESAFE_API_KEY (they ask the real model; there is nothing to probe without it)")
    if max_tokens > YES_SPEND_TOKENS and not yes_spend:
        raise ConfigError(f"--max-tokens {max_tokens} is above {YES_SPEND_TOKENS}: confirm the spend with --yes-spend")
    source = _states_from(states_from, loaded.data_dir, seed=loaded.cfg.run.seed, cfg=loaded.cfg)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    cache = SqliteDecisionCache(loaded.data_dir / "cache" / "decisions.sqlite")
    written: list[dict[str, Any]] = []
    try:
        for name in wanted:
            out_dir = loaded.data_dir / "probes" / "step0" / f"{stamp}-{name}"
            record = probe_module.run_suite(
                name,
                loaded.cfg,
                states=source,
                api_key=api_key,
                out_dir=out_dir,
                max_tokens=max_tokens,
                cache=cache,
                data_dir=loaded.data_dir,
                n_states=states,
                repeats=repeats,
            )
            path = probe_module.record_path(loaded.data_dir, record)
            written.append({"suite": name, "record": str(path), "run_dir": str(out_dir), "verdict": record.verdict})
            typer.echo(f"{name}: {path}")
    finally:
        cache.close()
    if get_globals(ctx).json:
        typer.echo(json_module.dumps(written, ensure_ascii=False, indent=2, sort_keys=True))


# ======================================================================================================================
# cache stats / verify
# ======================================================================================================================


def _spend_summary(data_dir: Path) -> dict[str, dict[str, int]]:
    """Per-scope UTC-day spend (INV-17: the two scopes are counted separately and never share a block)."""
    out: dict[str, dict[str, int]] = {}
    for scope in SCOPES:
        guard = SpendGuard(data_dir / "state" / "spend.sqlite", scope=scope, cfg=_spend_config())
        try:
            out[scope] = guard.by_day()
        finally:
            guard.close()
    return out


def _spend_config() -> Any:
    from jevbot.config import JevSpendConfig

    return JevSpendConfig()


@cache_app.command("stats")
def cache_stats_cmd(
    ctx: typer.Context,
    namespace: Annotated[str | None, typer.Option("--namespace", metavar="NS", help="Restrict to one namespace.")] = None,
) -> None:
    """Rows, namespaces, models, kinds, variants, tokens, per-day spend, nondeterminism rows and the manifest hash."""
    loaded = get_globals(ctx).load()
    cache = SqliteDecisionCache(loaded.data_dir / "cache" / "decisions.sqlite")
    try:
        counts = cache.stats(namespace)
        breakdown = cache.breakdown(namespace)
        manifests = {name: cache.manifest_hash(name) for name in breakdown.get("namespaces", {})}
    finally:
        cache.close()
    payload: dict[str, Any] = {
        "namespace": namespace,
        "counts": counts,
        "breakdown": breakdown,
        "manifest_hashes": manifests,
        "spend_by_day": _spend_summary(loaded.data_dir),
    }
    if get_globals(ctx).json:
        typer.echo(json_module.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    typer.echo(f"decision cache: {loaded.data_dir / 'cache' / 'decisions.sqlite'}")
    for key, value in counts.items():
        typer.echo(f"  {key}: {value}")
    for label, table in breakdown.items():
        typer.echo(f"  {label}: " + (", ".join(f"{k}={v}" for k, v in table.items()) or "-"))
    for name, digest in manifests.items():
        typer.echo(f"  manifest {name}: {digest}")
    for scope, days in _spend_summary(loaded.data_dir).items():
        typer.echo(f"  spend[{scope}]: " + (", ".join(f"{day}={tokens}" for day, tokens in days.items()) or "-"))


@cache_app.command("verify")
def cache_verify_cmd(
    ctx: typer.Context,
    namespace: Annotated[str | None, typer.Option("--namespace", metavar="NS", help="Restrict to one namespace.")] = None,
) -> None:
    """Re-derive every key from the stored state and question JSON, recompute the digests, check the model binding."""
    loaded = get_globals(ctx).load()
    cache = SqliteDecisionCache(loaded.data_dir / "cache" / "decisions.sqlite", read_only=True)
    try:
        report = cache.verify(namespace)
    finally:
        cache.close()
    payload: dict[str, Any] = {
        "ok": report.ok,
        "namespaces": report.namespaces,
        "rows": report.rows,
        "states": report.states,
        "question_sets": report.question_sets,
        "manifest_hashes": report.manifest_hashes,
        "problems": [{"namespace": p.namespace, "key": p.key, "problem": p.problem, "detail": p.detail} for p in report.problems],
    }
    if get_globals(ctx).json:
        typer.echo(json_module.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        typer.echo(f"rows={report.rows} namespaces={report.namespaces} states={report.states} question_sets={report.question_sets}")
        for name, digest in report.manifest_hashes.items():
            typer.echo(f"  manifest {name}: {digest}")
        for problem in report.problems:
            typer.echo(f"  PROBLEM {problem.problem} {problem.namespace}/{problem.key[:16]}: {problem.detail}")
        typer.echo("ok" if report.ok else f"{len(report.problems)} problem(s)")
    if not report.ok:
        raise typer.Exit(int(ExitCode.ERROR))
