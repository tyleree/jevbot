"""typer root: lazily registered sub-apps, global options, the PAPER-ONLY banner, exit codes (DESIGN.md sections 14 and 16).

`jevbot = jevbot.cli.main:app` (pyproject). The root registers one sub-app per area from the static list `SUBAPPS`, plus the
extenders of `EXTENDERS`. Nothing is imported until a sub-app is actually invoked: `jevbot --help` lists every area from the static
help table without importing pandas, the SDKs or any `*_cmds` module, and a module that no work package has delivered yet prints
"not built yet" instead of breaking the CLI - so the CLI is runnable after every wave.

CONTRACT FOR THE SUB-APP MODULES (each is owned by the package that implements the area, section 16)

1. A `SUBAPPS` entry is `(name, "package.module")` or `(name, "package.module:attribute")`; the attribute (default `app`) is a
   `typer.Typer`. `jevbot.cli.jev_cmds` exposes two: `app` (the `jev` sub-app) and `cache_app` (the `cache` sub-app).
   `jevbot.cli.doctor` exposes `app` with its single command, which becomes `jevbot doctor`.
2. An `EXTENDERS` entry is `(module, function)`; `EXTENDER_TARGETS[module]` names the sub-app it extends and the commands it adds.
   The function is called as `function(sub_app)` with that sub-app's `typer.Typer` right after it was imported
   (`jevbot.cli.leakage_cmds.register(eval_app)` adds `eval leakage`). A missing extender module leaves "not built yet" placeholders.
3. GLOBAL OPTIONS - `--config PATH` (repeatable, merged in order), `-o section.key=value` (repeatable), `--data-dir PATH`,
   `--log-level info|warning|error|debug`, `--json` - are accepted BEFORE the area name (parsed by the root) and ANYWHERE AFTER it
   (`jevbot paper run --config config/paper.toml`, `jevbot cache stats --json`): the root takes them out of the argument list
   before the sub-app parses it. A command reads them with `get_globals(ctx)` and boots with ONE call,
   `loaded = get_globals(ctx).load()` -> `LoadedConfig(cfg, secrets, data_dir)`: the merged configuration, `config.load_secrets`
   (INV-02 / INV-18 checks, every secret registered for redaction), the data directory (`config.data_dir` -> `ensure_data_dir`:
   `--data-dir` / `paths.data_dir` / `$JEVBOT_DATA`, outside the repo, mode 700) and - the moment the data directory is known -
   the JSON-lines log file `<data_dir>/logs/jevbot-<UTC date>.jsonl` of section 4 / 13.1, attached at the invocation's log level
   by `configure_logging(data_dir)`. Every command that works inside the data directory (all of them but `--help`-style listings
   and `jev questions`) uses `load()`; `load_config()` alone returns the configuration only (no secrets, no data dir, no file
   log) and is for commands that need nothing else. A command must NOT declare options with these names itself; if one does,
   that option is the command's own: the root leaves it in place and does not record it. `-o` is reserved: never use it as a
   short flag. In a sub-app's own tests (invoked without the root) pass `obj=GlobalOptions(...)` to `CliRunner.invoke`;
   without it `get_globals` returns the defaults.
4. Every command that can touch the broker prints the PAPER-ONLY banner first: the root prints it (once per invocation, on
   stderr so that `--json` output stays parseable) for every sub-app of `BANNER_SUBAPPS`; any other command that talks to Alpaca
   (`doctor --online`, `data fetch news|exdiv|bars`) calls `paper_only_banner(ctx)` itself.
5. Exit codes per 2.9: a `JevbotError` that escapes a command is printed as one redacted line and mapped with
   `errors.exit_code_for`; commands that refuse (kill active, lock held, `doctor --strict`) raise `typer.Exit(ExitCode.SAFETY_REFUSED)`.
6. Cross-package conveniences import their dependency lazily INSIDE the command function and print "not built yet" on
   `ImportError` (section 16); a sub-app module never imports another work package at module level.

The root installs the redacting log configuration (INV-18) on the console before any sub-app module is imported; the JSON-lines
file handler under the data directory is attached by `GlobalOptions.load()` (item 3) as soon as that directory is known;
`--log-level debug` never surfaces SDK bodies (third-party loggers stay pinned at WARNING). Tracebacks of unexpected errors are
plain (no local variables).
"""

import importlib
import logging
import os
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Annotated, Any, Final

import typer
from typer.core import TyperCommand, TyperGroup
from typer.main import get_command as typer_get_command

from jevbot import __version__, logsetup
from jevbot.errors import ExitCode, InvariantError, JevbotError, exit_code_for

if TYPE_CHECKING:
    from jevbot.config import Config, Secrets

__all__ = [
    "BANNER_SUBAPPS",
    "EXTENDERS",
    "EXTENDER_TARGETS",
    "LOG_SUBDIR",
    "NOT_BUILT",
    "PAPER_ONLY_BANNER",
    "SUBAPPS",
    "SUBAPP_HELP",
    "GlobalOptions",
    "LazySubApp",
    "LoadedConfig",
    "RootGroup",
    "app",
    "get_globals",
    "paper_only_banner",
    "split_global_options",
]

_log = logging.getLogger(__name__)

# ======================================================================================================================
# The static registration tables (section 16)
# ======================================================================================================================

SUBAPPS: Final[list[tuple[str, str]]] = [
    ("data", "jevbot.cli.data_cmds"),
    ("jev", "jevbot.cli.jev_cmds"),
    ("cache", "jevbot.cli.jev_cmds:cache_app"),
    ("backtest", "jevbot.cli.backtest_cmds"),
    ("baselines", "jevbot.cli.baselines_cmds"),
    ("eval", "jevbot.cli.eval_cmds"),
    ("paper", "jevbot.cli.paper_cmds"),
    ("record", "jevbot.cli.record_cmds"),
    ("doctor", "jevbot.cli.doctor"),
]
EXTENDERS: Final[list[tuple[str, str]]] = [("jevbot.cli.leakage_cmds", "register")]
# extender module -> (the sub-app it extends, the commands it adds): unrelated extenders are never imported, and a missing
# extender module leaves a "not built yet" placeholder for each of its commands
EXTENDER_TARGETS: Final[dict[str, tuple[str, tuple[str, ...]]]] = {"jevbot.cli.leakage_cmds": ("eval", ("leakage",))}

# shown by `jevbot --help` WITHOUT importing the sub-app modules
SUBAPP_HELP: Final[dict[str, str]] = {
    "data": "Datasets: fetch, derive, verify, scan-candidates, manifest.",
    "jev": "Jev decider: Step 0 probes, show-request, questions.",
    "cache": "Decision cache: stats, verify.",
    "backtest": "Backtests: run, report, verify.",
    "baselines": "Baselines 1-7 and the pre-registered ablation arm against a reference run.",
    "eval": "Evaluation: prereg, calibration, leakage, power, dsr, trials, model-agreement.",
    "paper": "Alpaca PAPER trading service: run, once, status, reconcile, kill, rearm, deadman, shadow, probe.",
    "record": "Snapshot recorder without trading (read-only market-data and news calls).",
    "doctor": "Environment and safety self-check.",
}

BANNER_SUBAPPS: Final[tuple[str, ...]] = ("paper", "record")  # every command of these areas talks to Alpaca
PAPER_ONLY_BANNER: Final = (
    "==============================================================================\n"
    " jevbot is PAPER-ONLY: every broker call goes to the Alpaca PAPER trading API.\n"
    " There is no code path to a funded account (INV-01).\n"
    "=============================================================================="
)
NOT_BUILT: Final = "not built yet"

_DEFAULT_ATTRIBUTE: Final = "app"
_COMPLETION_PARAMS: Final = frozenset({"install_completion", "show_completion"})
_PASS_THROUGH: Final[dict[str, Any]] = {"allow_extra_args": True, "ignore_unknown_options": True}
_BANNER_META_KEY: Final = "jevbot.cli.paper_only_banner"

# ======================================================================================================================
# Global options
# ======================================================================================================================

OPT_CONFIG: Final = "--config"
OPT_OVERRIDE: Final = "-o"
OPT_DATA_DIR: Final = "--data-dir"
OPT_LOG_LEVEL: Final = "--log-level"
OPT_JSON: Final = "--json"
_VALUE_OPTIONS: Final = (OPT_CONFIG, OPT_OVERRIDE, OPT_DATA_DIR, OPT_LOG_LEVEL)

LOG_SUBDIR: Final = "logs"  # `$JEVBOT_DATA/logs/jevbot-<date>.jsonl` (13.1)


@dataclass(frozen=True)
class LoadedConfig:
    """What `GlobalOptions.load()` returns: everything a command needs to work inside the data directory."""

    cfg: "Config"  # the merged, validated configuration (NOT resolved: `config.resolve` is the entry point's job, before RUN_START)
    secrets: "Secrets"  # env + `.env`, INV-02 / INV-18 checked, every value registered for redaction
    data_dir: Path  # resolved, existing, mode 700, outside the repository (D1); the JSON-lines log lives in `data_dir / LOG_SUBDIR`


@dataclass(frozen=True)
class GlobalOptions:
    """The global options of one invocation (section 14), wherever they were written on the command line."""

    config_paths: tuple[Path, ...] = ()  # `--config`, in command-line order (merged in that order over default.toml)
    overrides: tuple[str, ...] = ()  # `-o section.key=value`, in command-line order
    data_dir: Path | None = None  # `--data-dir`
    log_level: str = "info"  # one of logsetup.LEVELS
    json: bool = False  # `--json`: machine-readable output on stdout

    def config_overrides(self) -> tuple[str, ...]:
        """The `-o` items, then `--data-dir` as the override `paths.data_dir=...` (the explicit option wins)."""
        if self.data_dir is None:
            return self.overrides
        return (*self.overrides, f"paths.data_dir={self.data_dir.expanduser().absolute()}")

    def load_config(self, *, flags: Sequence[str] = ()) -> "Config":
        """`default.toml` <- every `--config` file <- `-o` / `--data-dir` (section 4). ConfigError (exit 2) on any problem;
        in paper mode `-o` is refused for the sections of `config.PROTECTED_SECTIONS_PAPER`.

        The configuration ONLY: no secrets, no data directory, no file log. Commands that work inside the data directory
        (nearly all of them) call `load()` instead."""
        from jevbot.config import load_config  # lazy: the root must stay importable without pandas

        return load_config(self.config_paths or None, self.config_overrides(), flags=flags)

    def configure_logging(self, data_dir: Path) -> Path:
        """Attach the JSON-lines handler `<data_dir>/logs/jevbot-<UTC date>.jsonl` (section 4, 13.1) at this invocation's log
        level, keeping the console handler, the redaction layers and the third-party pinning (INV-18). Idempotent (a second
        call replaces the handlers, never doubles them). Returns the log directory (created mode 700, files mode 600)."""
        log_dir = Path(data_dir) / LOG_SUBDIR
        logsetup.configure(self.log_level, log_dir=log_dir)
        return log_dir

    def load(self, *, flags: Sequence[str] = (), env: Mapping[str, str] | None = None) -> "LoadedConfig":
        """The one boot call of a command (module docstring, item 3): `load_config()`, then `config.load_secrets(cfg, env)`
        (`os.environ` by default; the `.env` file of `paths.env_file`; INV-02 / INV-18 refusals raise here), then the data
        directory (`config.data_dir(cfg, secrets)` -> `config.ensure_data_dir`, created when missing) and, the moment it is
        known, `configure_logging(data_dir)` - so every command that touches the data directory writes the JSON-lines log
        with redaction applied. Errors map to the 2.9 exit codes through the root."""
        from jevbot import config  # lazy: the root must stay importable without pandas

        cfg = self.load_config(flags=flags)
        secrets = config.load_secrets(cfg, os.environ if env is None else env)
        data_dir = config.ensure_data_dir(config.data_dir(cfg, secrets))
        self.configure_logging(data_dir)
        return LoadedConfig(cfg=cfg, secrets=secrets, data_dir=data_dir)


def get_globals(ctx: typer.Context) -> GlobalOptions:
    """The invocation's global options; the defaults when the command runs without the root (a sub-app's own tests)."""
    for candidate in (ctx.find_root().obj, ctx.obj):
        if isinstance(candidate, GlobalOptions):
            return candidate
    return GlobalOptions()


def _with_options(base: GlobalOptions, found: Sequence[tuple[str, str | None]]) -> GlobalOptions:
    """`base` plus the `(option, value)` pairs in command-line order; usage errors (exit 2) for invalid values."""
    config_paths = list(base.config_paths)
    overrides = list(base.overrides)
    data_dir, log_level, json_output = base.data_dir, base.log_level, base.json
    for option, value in found:
        if option == OPT_JSON:
            json_output = True
            continue
        if value is None or not value.strip():
            raise typer.BadParameter("requires a value", param_hint=option)
        if option == OPT_CONFIG:
            path = Path(value).expanduser()
            if not path.is_file():
                raise typer.BadParameter(f"config file {value!r} does not exist", param_hint=option)
            config_paths.append(path)
        elif option == OPT_OVERRIDE:
            overrides.append(value)
        elif option == OPT_DATA_DIR:
            data_dir = Path(value)
        elif option == OPT_LOG_LEVEL:
            level = value.strip().lower()
            if level not in logsetup.LEVELS:
                raise typer.BadParameter(f"{value!r} is not one of {', '.join(logsetup.LEVELS)}", param_hint=option)
            log_level = level
        else:
            raise InvariantError(f"jevbot.cli.main: unknown global option {option!r}")
    return GlobalOptions(
        config_paths=tuple(config_paths), overrides=tuple(overrides), data_dir=data_dir, log_level=log_level, json=json_output
    )


def _command_path(command: Any, args: Sequence[str]) -> list[Any]:
    """The commands `args` will walk through, from the sub-app's own command down to the leaf (first matching name per level)."""
    path = [command]
    current = command
    for token in args:
        if token == "--":
            break
        children = getattr(current, "commands", None)
        if not children:
            break
        if token in children:
            current = children[token]
            path.append(current)
    return path


def _declared_options(path: Sequence[Any]) -> dict[str, int]:
    """Every option string the commands on `path` declare themselves -> how many following tokens it consumes (0 for a flag)."""
    declared: dict[str, int] = {}
    for command in path:
        for param in getattr(command, "params", ()):
            is_flag = bool(getattr(param, "is_flag", False) or getattr(param, "count", False))
            consumes = 0 if is_flag else max(int(getattr(param, "nargs", 1)), 0)
            for opt in (*getattr(param, "opts", ()), *getattr(param, "secondary_opts", ())):
                if opt.startswith("-"):  # a positional argument lists its bare name here
                    declared[opt] = consumes
    return declared


def split_global_options(args: Sequence[str], declared: Mapping[str, int] | None = None) -> tuple[list[str], list[tuple[str, str | None]]]:
    """Take the global options out of a sub-app's argument list: `(remaining args, [(option, value)])` in command-line order.

    Recognised forms: `--json`; `--config X`, `--config=X` (same for `--data-dir`, `--log-level`); `-o X`, `-oX`. Everything after
    a literal `--` is left alone. `declared` maps the options that the target command DECLARES itself to the number of values
    they take: such an option belongs to that command - it is copied together with its value(s), so neither the option nor a
    value that merely looks like a global option (`--note "-o was wrong"`) is removed or recorded.
    """
    own = declared or {}
    rest: list[str] = []
    found: list[tuple[str, str | None]] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--":
            rest.extend(args[i:])
            break
        name, eq, attached = token.partition("=")
        if name in own:
            rest.append(token)
            if not eq:
                rest.extend(args[i + 1 : i + 1 + own[name]])
                i += own[name]
        elif token == OPT_JSON:
            found.append((OPT_JSON, None))
        elif token in _VALUE_OPTIONS:
            found.append((token, args[i + 1] if i + 1 < len(args) else None))
            i += 1
        elif eq and name in _VALUE_OPTIONS and name.startswith("--"):
            found.append((name, attached))
        elif token.startswith(OPT_OVERRIDE) and not token.startswith("--") and len(token) > len(OPT_OVERRIDE) and OPT_OVERRIDE not in own:
            found.append((OPT_OVERRIDE, token[len(OPT_OVERRIDE) :]))
        else:
            rest.append(token)
        i += 1
    return rest, found


# ======================================================================================================================
# The PAPER-ONLY banner
# ======================================================================================================================


def paper_only_banner(ctx: typer.Context | None = None) -> None:
    """Print the PAPER-ONLY banner on stderr - at most once per invocation when `ctx` is given (section 14, 11.1)."""
    if ctx is not None:
        meta = ctx.find_root().meta
        if meta.get(_BANNER_META_KEY):
            return
        meta[_BANNER_META_KEY] = True
    typer.echo(PAPER_ONLY_BANNER, err=True)


# ======================================================================================================================
# Lazy registration
# ======================================================================================================================


def _split_target(target: str) -> tuple[str, str]:
    module, _, attribute = target.partition(":")
    return module, attribute or _DEFAULT_ATTRIBUTE


def _import_or_none(module: str) -> ModuleType | None:
    """The module, or None when THAT module (or a package on its path) does not exist yet. Any other import failure - a missing
    third-party package, a missing sibling module, a bug inside the module - propagates: it must never be masked as "not built"."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name is not None and (exc.name == module or module.startswith(exc.name + ".")):
            return None
        raise


def _not_built(what: str, missing: str) -> None:
    typer.echo(f"jevbot {what}: {NOT_BUILT} (module {missing} is not part of this checkout)")
    raise typer.Exit(int(ExitCode.ERROR))


def _not_built_command(name: str, missing: str) -> TyperCommand:
    def callback() -> None:
        _not_built(name, missing)

    return TyperCommand(name, callback=callback, help=f"({NOT_BUILT})", add_help_option=False, context_settings=dict(_PASS_THROUGH))


# sub-app Typer -> extender modules already applied to it (Typer objects are module singletons; the root may be built many times)
_EXTENDED: Final["weakref.WeakKeyDictionary[typer.Typer, set[str]]"] = weakref.WeakKeyDictionary()


def _has_command(sub: typer.Typer, name: str) -> bool:
    return any((info.name or getattr(info.callback, "__name__", "")) == name for info in sub.registered_commands)


def _add_placeholder(sub: typer.Typer, area: str, command: str, missing: str) -> None:
    def placeholder() -> None:
        _not_built(f"{area} {command}", missing)

    sub.command(command, help=f"({NOT_BUILT})", add_help_option=False, context_settings=dict(_PASS_THROUGH))(placeholder)


def _apply_extenders(name: str, sub: typer.Typer) -> None:
    for module_name, function_name in EXTENDERS:
        if module_name not in EXTENDER_TARGETS:
            raise InvariantError(f"jevbot.cli.main: EXTENDERS entry {module_name!r} has no EXTENDER_TARGETS entry")
        target, commands = EXTENDER_TARGETS[module_name]
        applied = _EXTENDED.setdefault(sub, set())
        if target != name or module_name in applied:
            continue
        module = _import_or_none(module_name)
        if module is None:
            for command in commands:
                if not _has_command(sub, command):
                    _add_placeholder(sub, name, command, module_name)
            continue
        register = getattr(module, function_name, None)
        if not callable(register):
            raise InvariantError(f"jevbot.cli.main: {module_name}.{function_name} is not callable (the extender contract)")
        register(sub)
        applied.add(module_name)


def _load_subapp(name: str, target: str) -> Any:
    """Import the sub-app, apply its extenders and turn it into a command - or the "not built yet" command."""
    module_name, attribute = _split_target(target)
    module = _import_or_none(module_name)
    if module is None:
        return _not_built_command(name, module_name)
    sub = getattr(module, attribute, None)
    if not isinstance(sub, typer.Typer):
        raise InvariantError(f"jevbot.cli.main: {module_name}:{attribute} is not a typer.Typer (the sub-app contract)")
    _apply_extenders(name, sub)
    try:
        command = typer_get_command(sub)
    except RuntimeError:
        raise InvariantError(f"jevbot.cli.main: {module_name}:{attribute} registers no command") from None
    command.name = name
    command.params = [param for param in command.params if param.name not in _COMPLETION_PARAMS]
    return command


class LazySubApp(TyperCommand):
    """Stands in for one `SUBAPPS` entry: static help for listings; the real module is imported only when the area is invoked
    (`make_context` is the first thing the root group asks of a sub-command, and the context it returns carries the REAL command)."""

    def __init__(self, name: str, target: str) -> None:
        super().__init__(name, help=SUBAPP_HELP.get(name, ""), add_help_option=False, context_settings=dict(_PASS_THROUGH))
        self.target = target

    def make_context(self, info_name: str | None, args: list[str], parent: Any = None, **extra: Any) -> Any:
        name = self.name or ""
        resilient = bool(extra.get("resilient_parsing"))
        if name in BANNER_SUBAPPS and not resilient:
            paper_only_banner(parent)
        command = _load_subapp(name, self.target)

        rest, found = split_global_options(args, _declared_options(_command_path(command, args)))
        base = parent.obj if parent is not None and isinstance(parent.obj, GlobalOptions) else GlobalOptions()
        merged = _with_options(base, found) if found else base
        if parent is not None:
            parent.obj = merged
        if merged.log_level != base.log_level and not resilient:
            logsetup.configure(merged.log_level)
        extra.setdefault("obj", merged)
        return command.make_context(info_name, rest, parent=parent, **extra)


class RootGroup(TyperGroup):
    """The root command group: one `LazySubApp` per `SUBAPPS` entry, and the 2.9 exit-code mapping for every command."""

    def __init__(self, **attrs: Any) -> None:
        super().__init__(**attrs)
        for name, target in SUBAPPS:
            self.add_command(LazySubApp(name, target), name)

    def invoke(self, ctx: Any) -> Any:
        try:
            return super().invoke(ctx)
        except JevbotError as exc:
            _log.debug("command failed", exc_info=True)
            typer.echo(f"error: {type(exc).__name__}: {logsetup.redact(str(exc))}", err=True)
            raise typer.Exit(int(exit_code_for(exc))) from None


# ======================================================================================================================
# The root app
# ======================================================================================================================

app = typer.Typer(
    name="jevbot",
    cls=RootGroup,
    help=(
        "jevbot - options trading bot: historical backtest + Alpaca paper trading. PAPER-ONLY: there is no live-trading code path. "
        "The global options below are accepted before the command or anywhere after it."
    ),
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,  # plain tracebacks: a rich traceback would print local variables (INV-18)
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"jevbot {__version__}")
        raise typer.Exit(int(ExitCode.OK))


@app.callback()
def root(
    ctx: typer.Context,
    config: Annotated[
        list[str] | None,
        typer.Option(OPT_CONFIG, metavar="PATH", help="Config file merged over config/default.toml (repeatable, in order)."),
    ] = None,
    override: Annotated[
        list[str] | None,
        typer.Option(
            OPT_OVERRIDE,
            metavar="SECTION.KEY=VALUE",
            help="Config override (repeatable). Refused in paper mode for risk / kill / health / orders / dte / exits.",
        ),
    ] = None,
    data_dir: Annotated[
        str | None, typer.Option(OPT_DATA_DIR, metavar="PATH", help="Data directory (else paths.data_dir, else $JEVBOT_DATA).")
    ] = None,
    log_level: Annotated[
        str, typer.Option(OPT_LOG_LEVEL, metavar="LEVEL", help="info | warning | error | debug (debug never surfaces SDK bodies).")
    ] = "info",
    json_output: Annotated[bool, typer.Option(OPT_JSON, help="Machine-readable output on stdout.")] = False,
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True, help="Print the version and exit.")
    ] = False,
) -> None:
    """Parse the global options given BEFORE the area name and install the redacting log configuration."""
    if ctx.resilient_parsing:
        return
    found: list[tuple[str, str | None]] = [(OPT_CONFIG, path) for path in config or ()]
    found += [(OPT_OVERRIDE, item) for item in override or ()]
    if data_dir is not None:
        found.append((OPT_DATA_DIR, data_dir))
    found.append((OPT_LOG_LEVEL, log_level))
    if json_output:
        found.append((OPT_JSON, None))
    options = _with_options(GlobalOptions(), found)
    ctx.obj = options
    # INV-18: the redacting configuration is in place before any sub-app module (and any secret) is loaded
    logsetup.configure(options.log_level)
    ctx.call_on_close(logsetup.reset)


if __name__ == "__main__":
    # `python -m jevbot.cli.main`: run the canonical module instance, the one the sub-app modules import `get_globals` from
    from jevbot.cli.main import app as _app

    _app()
