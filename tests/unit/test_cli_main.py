"""Tests for src/jevbot/cli/main.py: the lazy typer root (DESIGN.md sections 14 and 16; WP00).

Sub-app modules are faked through `sys.modules` and a patched `SUBAPPS` table, so these tests keep working whichever real
`*_cmds` modules the later work packages have delivered.
"""

import importlib.util
import json
import logging
import stat
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest
import typer
from typer.testing import CliRunner

import jevbot
from jevbot import logsetup
from jevbot.cli import main
from jevbot.cli.main import (
    EXTENDER_TARGETS,
    EXTENDERS,
    NOT_BUILT,
    PAPER_ONLY_BANNER,
    SUBAPP_HELP,
    SUBAPPS,
    GlobalOptions,
    app,
    get_globals,
    paper_only_banner,
    split_global_options,
)
from jevbot.errors import (
    BrokerRejected,
    CacheMissError,
    ConfigError,
    DataUnavailable,
    InvariantError,
    JevbotError,
    LedgerCorrupt,
    ModelMismatchError,
    PaperGuardError,
    PitViolation,
    SpendLimitError,
)

REPO = Path(__file__).resolve().parents[2]
FAKE = "jevbot.cli._t_fake_cmds"
GHOST = "jevbot.cli._t_module_that_does_not_exist"

runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean_logging() -> Iterator[None]:
    yield
    logsetup.reset()
    logging.getLogger().setLevel(logging.WARNING)


def install_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: object) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def fake_area() -> typer.Typer:
    """A sub-app with two commands; `show` prints what it received (its own arguments and the global options)."""
    sub = typer.Typer(help="fake area")

    @sub.command()
    def show(
        ctx: typer.Context,
        name: str = typer.Option("anon", "--name"),
        extra: list[str] = typer.Argument(None),
    ) -> None:
        options = get_globals(ctx)
        typer.echo(
            json.dumps(
                {
                    "name": name,
                    "extra": list(extra or ()),
                    "config": [str(p) for p in options.config_paths],
                    "overrides": list(options.overrides),
                    "data_dir": None if options.data_dir is None else str(options.data_dir),
                    "log_level": options.log_level,
                    "json": options.json,
                    "root_logger_level": logging.getLogger().level,
                    "sdk_logger_level": logging.getLogger("typesafe_sdk").getEffectiveLevel(),
                }
            )
        )

    @sub.command()
    def other() -> None:
        typer.echo("other ran")

    return sub


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> typer.Typer:
    sub = fake_area()
    install_module(monkeypatch, FAKE, app=sub)
    monkeypatch.setattr(main, "SUBAPPS", [("fake", FAKE)])
    return sub


def shown(result: object) -> dict:
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    return json.loads(result.stdout)  # type: ignore[attr-defined]


# ======================================================================================================================
# The static tables (section 16)
# ======================================================================================================================


def test_subapps_is_the_static_list_of_section_14() -> None:
    assert [name for name, _ in SUBAPPS] == ["data", "jev", "cache", "backtest", "baselines", "eval", "paper", "record", "doctor"]
    assert dict(SUBAPPS) == {
        "data": "jevbot.cli.data_cmds",
        "jev": "jevbot.cli.jev_cmds",
        "cache": "jevbot.cli.jev_cmds:cache_app",  # jev_cmds.py carries the `jev` AND the `cache` sub-app (section 16, WP03)
        "backtest": "jevbot.cli.backtest_cmds",
        "baselines": "jevbot.cli.baselines_cmds",
        "eval": "jevbot.cli.eval_cmds",
        "paper": "jevbot.cli.paper_cmds",
        "record": "jevbot.cli.record_cmds",
        "doctor": "jevbot.cli.doctor",
    }
    assert EXTENDERS == [("jevbot.cli.leakage_cmds", "register")]
    assert EXTENDER_TARGETS == {"jevbot.cli.leakage_cmds": ("eval", ("leakage",))}
    assert set(SUBAPP_HELP) == {name for name, _ in SUBAPPS}
    assert all(target in dict(SUBAPPS) for target, _ in EXTENDER_TARGETS.values())


def test_every_cli_module_of_the_section_1_tree_is_registered() -> None:
    # cli/doctor.py, data_cmds, jev_cmds (jev + cache), backtest_cmds, eval_cmds, baselines_cmds, leakage_cmds (extends eval),
    # paper_cmds, record_cmds
    registered = {target.partition(":")[0] for _, target in SUBAPPS} | {module for module, _ in EXTENDERS}
    assert registered == {
        f"jevbot.cli.{stem}"
        for stem in (
            "doctor",
            "data_cmds",
            "jev_cmds",
            "backtest_cmds",
            "eval_cmds",
            "baselines_cmds",
            "leakage_cmds",
            "paper_cmds",
            "record_cmds",
        )
    }


def test_console_script_points_at_the_root_app() -> None:
    import tomllib

    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["jevbot"] == "jevbot.cli.main:app"
    assert isinstance(app, typer.Typer)


# ======================================================================================================================
# Root help, version, usage errors
# ======================================================================================================================


def test_root_help_lists_every_area_and_the_global_options() -> None:
    before = set(sys.modules)
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for name, _ in SUBAPPS:
        assert name in result.output
    for option in ("--config", "-o", "--data-dir", "--log-level", "--json", "--version"):
        assert option in result.output
    assert "PAPER-ONLY" in result.output
    assert "--install-completion" not in result.output
    imported = {name for name in set(sys.modules) - before if name.startswith("jevbot.cli.")}
    assert imported == set(), f"root --help imported sub-app modules: {sorted(imported)}"


def test_root_stays_light_nothing_heavy_is_imported_for_help() -> None:
    code = textwrap.dedent(
        """
        import sys
        from typer.testing import CliRunner
        import jevbot.cli.main as m
        result = CliRunner().invoke(m.app, ["--help"])
        assert result.exit_code == 0, result.output
        heavy = ("pandas", "numpy", "scipy", "pyarrow", "matplotlib", "alpaca", "typesafe_sdk", "httpx2", "exchange_calendars", "msgspec")
        print(sorted({name.split(".")[0] for name in sys.modules} & set(heavy)))
        print(sorted(name for name in sys.modules if name.startswith("jevbot.")))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO, timeout=120
    ).stdout.splitlines()
    assert out[0] == "[]", f"heavy imports at root --help: {out[0]}"
    assert out[1] == str(["jevbot.cli", "jevbot.cli.main", "jevbot.errors", "jevbot.logsetup"])


def test_python_dash_m_entry_point_and_version() -> None:
    done = subprocess.run([sys.executable, "-m", "jevbot.cli.main", "--version"], capture_output=True, text=True, cwd=REPO, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == f"jevbot {jevbot.__version__}"
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"jevbot {jevbot.__version__}"


def test_no_arguments_prints_help() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code in (0, 2)
    assert "Usage" in result.output and "paper" in result.output


def test_unknown_command_is_a_usage_error() -> None:
    result = runner.invoke(app, ["no-such-area"])
    assert result.exit_code == 2
    assert "No such command" in result.output


# ======================================================================================================================
# Lazy registration: missing modules, import bugs, attribute contract
# ======================================================================================================================


def test_missing_module_prints_not_built_yet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "SUBAPPS", [("ghost", GHOST), ("ghost2", f"{GHOST}:other_app")])
    for args in (["ghost"], ["ghost", "--help"], ["ghost", "fetch", "mirror", "--force", "--json"], ["ghost2", "stats"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 1, result.output
        assert NOT_BUILT in result.stdout
        assert GHOST in result.stdout
    listing = runner.invoke(app, ["--help"])
    assert listing.exit_code == 0 and "ghost" in listing.output  # the CLI is runnable although the area is missing


def test_real_areas_that_are_not_delivered_yet_say_not_built_yet() -> None:
    assert NOT_BUILT == "not built yet"
    for name, target in SUBAPPS:
        module = target.partition(":")[0]
        if importlib.util.find_spec(module) is None:
            result = runner.invoke(app, [name, "--help"])
            assert result.exit_code == 1, (name, result.output)
            assert NOT_BUILT in result.stdout and module in result.stdout


def test_an_import_bug_inside_a_subapp_is_never_masked_as_not_built(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "t_broken_cmds.py").write_text("import t_dependency_that_is_missing\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(main, "SUBAPPS", [("broken", "t_broken_cmds")])
    result = runner.invoke(app, ["broken", "run"])
    assert result.exit_code == 1
    assert isinstance(result.exception, ModuleNotFoundError)
    assert result.exception.name == "t_dependency_that_is_missing"
    assert NOT_BUILT not in result.output
    sys.modules.pop("t_broken_cmds", None)


def test_the_attribute_must_be_a_typer_app(monkeypatch: pytest.MonkeyPatch) -> None:
    install_module(monkeypatch, FAKE, app="not a typer app", empty=typer.Typer())
    monkeypatch.setattr(main, "SUBAPPS", [("fake", FAKE), ("nope", f"{FAKE}:missing"), ("empty", f"{FAKE}:empty")])
    for area, fragment in (("fake", "is not a typer.Typer"), ("nope", "is not a typer.Typer"), ("empty", "registers no command")):
        result = runner.invoke(app, [area, "x"])
        assert result.exit_code == 1
        assert "error: InvariantError" in result.stderr and fragment in result.stderr


def test_module_colon_attribute_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    first, second = fake_area(), typer.Typer()

    @second.command()
    def stats() -> None:
        typer.echo("cache stats ran")

    @second.command()
    def verify() -> None:
        typer.echo("cache verify ran")

    install_module(monkeypatch, FAKE, app=first, cache_app=second)
    monkeypatch.setattr(main, "SUBAPPS", [("jev", FAKE), ("cache", f"{FAKE}:cache_app")])
    assert runner.invoke(app, ["jev", "other"]).stdout.strip() == "other ran"
    assert runner.invoke(app, ["cache", "stats"]).stdout.strip() == "cache stats ran"
    assert runner.invoke(app, ["cache", "verify"]).stdout.strip() == "cache verify ran"


def test_single_command_module_becomes_the_command_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    # `jevbot doctor [--strict]`: a Typer with ONE command is the command, not a group with a `doctor doctor` level
    sub = typer.Typer()

    @sub.command()
    def doctor(ctx: typer.Context, strict: bool = typer.Option(False, "--strict")) -> None:
        typer.echo(f"doctor strict={strict} json={get_globals(ctx).json}")

    install_module(monkeypatch, FAKE, app=sub)
    monkeypatch.setattr(main, "SUBAPPS", [("doctor", FAKE)])
    assert runner.invoke(app, ["doctor"]).stdout.strip() == "doctor strict=False json=False"
    assert runner.invoke(app, ["doctor", "--strict", "--json"]).stdout.strip() == "doctor strict=True json=True"
    help_text = runner.invoke(app, ["doctor", "--help"])
    assert help_text.exit_code == 0 and "--strict" in help_text.output and "jevbot doctor" in help_text.output


def test_subapp_help_shows_its_commands_without_completion_options(fake: typer.Typer) -> None:
    result = runner.invoke(app, ["fake", "--help"])
    assert result.exit_code == 0, result.output
    assert "show" in result.output and "other" in result.output
    assert "jevbot fake" in result.output
    assert "--install-completion" not in result.output and "--show-completion" not in result.output
    leaf = runner.invoke(app, ["fake", "show", "--help"])
    assert leaf.exit_code == 0 and "--name" in leaf.output and "jevbot fake show" in leaf.output


# ======================================================================================================================
# Global options: before the area, anywhere after it, merged in command-line order
# ======================================================================================================================


def test_global_options_before_and_after_the_command(fake: typer.Typer, tmp_path: Path) -> None:
    files = [tmp_path / f"c{i}.toml" for i in range(3)]
    for file in files:
        file.write_text("", encoding="utf-8")
    args = ["--config", str(files[0]), "-o", "risk.max_new_per_day=1", "fake", "--config", str(files[1]), "show", "--name", "bob"]
    args += [
        "-o",
        "dte.target=30",
        "--json",
        "--data-dir",
        "/tmp/jb-data",
        "left",
        f"--config={files[2]}",
        "-ocadence.x=2",
        "--log-level",
        "DEBUG",
        "right",
    ]
    got = shown(runner.invoke(app, args))
    assert got["name"] == "bob"
    assert got["extra"] == ["left", "right"]  # the command's own arguments are untouched, in order
    assert got["config"] == [str(f) for f in files]  # merged in command-line order
    assert got["overrides"] == ["risk.max_new_per_day=1", "dte.target=30", "cadence.x=2"]
    assert got["data_dir"] == "/tmp/jb-data"
    assert got["log_level"] == "debug"
    assert got["json"] is True


def test_every_global_option_is_accepted_before_the_area_name(fake: typer.Typer, tmp_path: Path) -> None:
    file = tmp_path / "c.toml"
    file.write_text("", encoding="utf-8")
    args = ["--config", str(file), "-o", "dte.target=30", "--data-dir", "/tmp/jb-data", "--log-level", "warning", "--json", "fake", "show"]
    got = shown(runner.invoke(app, args))
    assert (got["config"], got["overrides"], got["data_dir"], got["log_level"], got["json"]) == (
        [str(file)],
        ["dte.target=30"],
        "/tmp/jb-data",
        "warning",
        True,
    )
    assert got["root_logger_level"] == logging.WARNING


def test_resilient_parsing_configures_nothing(fake: typer.Typer) -> None:
    # shell-completion style parsing must have no side effects: no logging setup, no banner, no global-option object
    command = typer.main.get_command(app)
    with command.make_context("jevbot", ["--log-level", "debug", "fake", "show"], resilient_parsing=True) as ctx:
        assert ctx.obj is None
        assert [h for h in logging.getLogger().handlers if getattr(h, "_jevbot_handler", False)] == []


def test_a_lazy_subapp_works_without_a_parent_context(fake: typer.Typer) -> None:
    lazy = main.LazySubApp("fake", FAKE)
    assert lazy.help == SUBAPP_HELP.get("fake", "")
    with lazy.make_context("fake", ["show", "--json", "--name", "solo"]) as ctx:
        assert isinstance(ctx.obj, GlobalOptions) and ctx.obj.json is True
        assert ctx.command.name == "fake"  # the context carries the REAL command, not the lazy stand-in
        assert ctx.args == ["--name", "solo"]  # `--json` was taken out before the sub-app parsed its arguments


def test_defaults_when_no_global_option_is_given(fake: typer.Typer) -> None:
    got = shown(runner.invoke(app, ["fake", "show"]))
    assert got == {
        "name": "anon",
        "extra": [],
        "config": [],
        "overrides": [],
        "data_dir": None,
        "log_level": "info",
        "json": False,
        "root_logger_level": logging.INFO,
        "sdk_logger_level": logging.WARNING,
    }


def test_the_systemd_command_line_of_11_10(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # ExecStart=.../jevbot paper run --config config/paper.toml  - the global option comes AFTER the command
    sub = typer.Typer()

    @sub.command()
    def run(ctx: typer.Context) -> None:
        typer.echo(",".join(p.name for p in get_globals(ctx).config_paths))

    @sub.command()
    def status() -> None:
        typer.echo("status")

    install_module(monkeypatch, FAKE, app=sub)
    monkeypatch.setattr(main, "SUBAPPS", [("paper", FAKE)])
    paper_toml = tmp_path / "paper.toml"
    paper_toml.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["paper", "run", "--config", str(paper_toml)])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "paper.toml"


def test_a_value_of_the_commands_own_option_is_never_mistaken_for_a_global_option(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = typer.Typer()

    @sub.command()
    def rearm(ctx: typer.Context, note: str = typer.Option(..., "--note"), reset_peak: bool = typer.Option(False, "--reset-peak")) -> None:
        options = get_globals(ctx)
        typer.echo(json.dumps({"note": note, "reset_peak": reset_peak, "json": options.json, "overrides": list(options.overrides)}))

    @sub.command()
    def status() -> None:
        typer.echo("status")

    install_module(monkeypatch, FAKE, app=sub)
    monkeypatch.setattr(main, "SUBAPPS", [("paper", FAKE)])
    got = json.loads(runner.invoke(app, ["paper", "rearm", "--note", "-o was wrong", "--reset-peak", "--json"]).stdout)
    assert got == {"note": "-o was wrong", "reset_peak": True, "json": True, "overrides": []}
    got = json.loads(runner.invoke(app, ["paper", "rearm", "--note=--json", "-o", "dte.target=30"]).stdout)
    assert got == {"note": "--json", "reset_peak": False, "json": False, "overrides": ["dte.target=30"]}
    # the unit-level view of the same rule: `--note` takes one value, `--reset-peak` none
    declared = {"--note": 1, "--reset-peak": 0}
    assert split_global_options(["rearm", "--note", "--json", "--reset-peak", "--json"], declared) == (
        ["rearm", "--note", "--json", "--reset-peak"],
        [("--json", None)],
    )


def test_an_option_the_command_declares_itself_belongs_to_the_command(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = typer.Typer()

    @sub.command()
    def trials(ctx: typer.Context, as_json: bool = typer.Option(False, "--json"), out: str = typer.Option("", "-o", "--out")) -> None:
        typer.echo(
            json.dumps(
                {"local_json": as_json, "out": out, "global_json": get_globals(ctx).json, "overrides": list(get_globals(ctx).overrides)}
            )
        )

    @sub.command()
    def plain(ctx: typer.Context) -> None:
        typer.echo(json.dumps({"global_json": get_globals(ctx).json}))

    install_module(monkeypatch, FAKE, app=sub)
    monkeypatch.setattr(main, "SUBAPPS", [("eval", FAKE)])
    monkeypatch.setattr(main, "EXTENDERS", [])
    own = json.loads(runner.invoke(app, ["eval", "trials", "--json", "-o", "file.csv"]).stdout)
    assert own == {"local_json": True, "out": "file.csv", "global_json": False, "overrides": []}
    # the sibling command declares nothing: there the same token is the global option
    assert json.loads(runner.invoke(app, ["eval", "plain", "--json"]).stdout) == {"global_json": True}
    # and before the area name it is always the root's
    before = json.loads(runner.invoke(app, ["--json", "eval", "trials"]).stdout)
    assert before == {"local_json": False, "out": "", "global_json": True, "overrides": []}


def test_everything_after_a_double_dash_is_left_alone(fake: typer.Typer) -> None:
    got = shown(runner.invoke(app, ["fake", "show", "--", "--json", "-o", "x.y=1"]))
    assert got["extra"] == ["--json", "-o", "x.y=1"]
    assert got["json"] is False and got["overrides"] == []


@pytest.mark.parametrize(
    "args",
    [
        ["--log-level", "loud", "fake", "show"],
        ["fake", "show", "--log-level", "loud"],
        ["fake", "show", "--log-level=trace"],
        ["--config", "/nonexistent/jevbot.toml", "fake", "show"],
        ["fake", "show", "--config", "/nonexistent/jevbot.toml"],
        ["fake", "show", "--config"],  # value missing
        ["fake", "show", "-o"],
        ["fake", "show", "--data-dir="],
    ],
)
def test_invalid_global_option_values_are_usage_errors(fake: typer.Typer, args: list[str]) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 2, result.output
    assert result.stdout == ""  # the command never ran


def test_split_global_options_forms_and_order() -> None:
    rest, found = split_global_options(
        ["run", "--config", "a.toml", "-x", "--json", "pos", "--data-dir=/d", "-orisk.a=1", "-o", "b.c=2", "--log-level=debug"]
    )
    assert rest == ["run", "-x", "pos"]
    assert found == [
        ("--config", "a.toml"),
        ("--json", None),
        ("--data-dir", "/d"),
        ("-o", "risk.a=1"),
        ("-o", "b.c=2"),
        ("--log-level", "debug"),
    ]
    # look-alikes are not global options
    rest, found = split_global_options(["--config-hash", "abc", "--jsonl", "--out", "x", "-only"])
    assert (rest, found) == (["--config-hash", "abc", "--jsonl", "--out", "x"], [("-o", "nly")])
    # a trailing option without its value is reported with value None (a usage error later)
    assert split_global_options(["run", "--config"]) == (["run"], [("--config", None)])
    # `--` ends the scan
    assert split_global_options(["a", "--", "--json"]) == (["a", "--", "--json"], [])
    # declared options stay where they are and are not recorded
    declared = {"--json": 0, "-o": 1, "--config": 1}
    args = ["t", "--json", "-o", "f", "-of", "--config", "c", "--config=c", "--log-level", "error"]
    assert split_global_options(args, declared) == (
        ["t", "--json", "-o", "f", "-of", "--config", "c", "--config=c"],
        [("--log-level", "error")],
    )


def test_get_globals_without_the_root() -> None:
    # a sub-app's own tests invoke it directly: defaults, or whatever the test passes as obj
    sub = fake_area()
    assert shown(runner.invoke(sub, ["show"]))["json"] is False
    got = shown(runner.invoke(sub, ["show"], obj=GlobalOptions(json=True, overrides=("a.b=1",))))
    assert got["json"] is True and got["overrides"] == ["a.b=1"]


# ======================================================================================================================
# GlobalOptions.load_config: merge order, --data-dir, the paper-mode refusal (section 4)
# ======================================================================================================================


def config_area() -> typer.Typer:
    sub = typer.Typer()

    @sub.command()
    def show(ctx: typer.Context) -> None:
        cfg = get_globals(ctx).load_config()
        typer.echo(json.dumps({"max_new_per_day": cfg.risk.max_new_per_day, "data_dir": cfg.paths.data_dir, "target": cfg.dte.target}))

    @sub.command()
    def other() -> None:
        typer.echo("other")

    return sub


def test_load_config_merges_files_then_overrides_then_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_module(monkeypatch, FAKE, app=config_area())
    monkeypatch.setattr(main, "SUBAPPS", [("fake", FAKE)])
    first, second = tmp_path / "a.toml", tmp_path / "b.toml"
    first.write_text("[risk]\nmax_new_per_day = 1\n[dte]\ntarget = 33\n", encoding="utf-8")
    second.write_text("[dte]\ntarget = 36\n", encoding="utf-8")
    data = tmp_path / "data"

    plain = json.loads(runner.invoke(app, ["fake", "show"]).stdout)
    assert plain["max_new_per_day"] == 2 and plain["target"] == 35  # the shipped defaults

    result = runner.invoke(app, ["--config", str(first), "fake", "show", "--config", str(second), "--data-dir", str(data)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"max_new_per_day": 1, "data_dir": str(data), "target": 36}  # later file wins

    override = runner.invoke(
        app, ["fake", "show", "--config", str(first), "-o", "dte.target=30", "-o", f"paths.data_dir={tmp_path}/ignored"]
    )
    assert json.loads(override.stdout)["target"] == 30
    both = runner.invoke(app, ["fake", "show", "-o", f"paths.data_dir={tmp_path}/ignored", "--data-dir", str(data)])
    assert json.loads(both.stdout)["data_dir"] == str(data)  # the explicit option wins over -o paths.data_dir


def test_a_relative_data_dir_is_made_absolute() -> None:
    options = GlobalOptions(data_dir=Path("some/dir"), overrides=("a.b=1",))
    assert options.config_overrides() == ("a.b=1", f"paths.data_dir={Path.cwd() / 'some/dir'}")
    assert GlobalOptions(overrides=("a.b=1",)).config_overrides() == ("a.b=1",)


def test_config_errors_exit_2_and_paper_mode_refuses_protected_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    install_module(monkeypatch, FAKE, app=config_area())
    monkeypatch.setattr(main, "SUBAPPS", [("fake", FAKE)])
    typo = runner.invoke(app, ["fake", "show", "-o", "risk.no_such_key=1"])
    assert typo.exit_code == 2 and "error: ConfigError" in typo.stderr and typo.stdout == ""
    malformed = runner.invoke(app, ["fake", "show", "-o", "not-an-override"])
    assert malformed.exit_code == 2 and "error: ConfigError" in malformed.stderr
    refused = runner.invoke(app, ["fake", "show", "-o", "run.mode=paper", "-o", "risk.max_new_per_day=5"])
    assert refused.exit_code == 2
    assert "refused in paper mode" in refused.stderr


# ======================================================================================================================
# GlobalOptions.load(): config + secrets + data dir + the JSON-lines log file under it (section 4, 13.1; INV-02, INV-18)
# ======================================================================================================================

SECRET = "ts-secret-7c1e9b2a4d6f"  # the TypeSafe key of these tests: it must never reach a log line
JSONL_HANDLERS = "jevbot_jsonl_handlers"


def boot_area() -> typer.Typer:
    """A command that boots the way every data-dir command must: `get_globals(ctx).load()`, then logs at every level."""
    sub = typer.Typer()

    @sub.command()
    def boot(ctx: typer.Context, secret_in_log: bool = typer.Option(False, "--secret-in-log")) -> None:
        loaded = get_globals(ctx).load()
        log = logging.getLogger("jevbot.test.boot")
        log.debug("boot debug")
        log.info("boot info")
        log.warning("boot warning")
        if secret_in_log:
            log.info("authenticating with key %s", loaded.secrets.typesafe_api_key)  # the redaction layer must catch this
        logging.getLogger("typesafe_sdk").info("a sub-WARNING third-party record: never written")
        jsonl = [h for h in logging.getLogger().handlers if isinstance(h, logsetup.DailyJsonlHandler)]
        typer.echo(
            json.dumps(
                {
                    "data_dir": str(loaded.data_dir),
                    "config_data_dir": loaded.cfg.paths.data_dir,
                    "has_typesafe_key": loaded.secrets.has_typesafe_key,
                    "jsonl_handlers": len(jsonl),
                    "log_dirs": sorted({str(h.path_for(datetime.now(UTC).date()).parent) for h in jsonl}),
                    "console_handlers": len([h for h in logging.getLogger().handlers if getattr(h, "_jevbot_handler", False)]) - len(jsonl),
                    "root_level": logging.getLogger().level,
                    "secrets_registered": logsetup.registered_secret_count(),
                }
            )
        )

    @sub.command()
    def other() -> None:
        typer.echo("other")

    return sub


def no_env_file(tmp_path: Path) -> str:
    """The override that keeps the operator's real (git-ignored) `.env` out of a test: an absent file beside the test's tmp dir."""
    return f"paths.env_file={tmp_path / 'absent.env'}"


@pytest.fixture
def boot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Installs the boot area and returns the argument prefix every invocation starts with."""
    install_module(monkeypatch, FAKE, app=boot_area())
    monkeypatch.setattr(main, "SUBAPPS", [("fake", FAKE)])
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    return ["-o", no_env_file(tmp_path)]


def jsonl_records(data_dir: Path) -> list[dict[str, object]]:
    files = sorted((data_dir / main.LOG_SUBDIR).glob("jevbot-*.jsonl"))
    assert len(files) == 1, files
    assert files[0].name == f"jevbot-{datetime.now(UTC).date().isoformat()}.jsonl"
    return [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]


def test_load_boots_the_command_and_writes_the_jsonl_log_under_the_data_dir(boot: list[str], data_dir: Path) -> None:
    result = runner.invoke(app, [*boot, "fake", "boot"])
    got = shown(result)
    assert got["data_dir"] == str(data_dir.resolve()) and got["config_data_dir"] == ""  # $JEVBOT_DATA, as paths.data_dir is ""
    assert got["has_typesafe_key"] is True and got["secrets_registered"] >= 1
    assert got["jsonl_handlers"] == 1 and got["log_dirs"] == [str(data_dir.resolve() / "logs")]
    assert got["console_handlers"] == 1 and got["root_level"] == logging.INFO  # the console handler survives, the level is the invocation's
    assert main.LOG_SUBDIR == "logs"
    log_dir = data_dir / "logs"
    assert stat.S_IMODE(log_dir.stat().st_mode) == 0o700
    records = jsonl_records(data_dir)
    messages = [r["msg"] for r in records if r["logger"] == "jevbot.test.boot"]
    assert messages == ["boot info", "boot warning"]  # info level: the debug record is not written
    assert all(r["logger"] != "typesafe_sdk" for r in records)  # INV-18: pinned third-party namespaces stay silent below WARNING
    for record in records:
        assert set(record) >= {"ts", "level", "logger", "msg"} and str(record["ts"]).endswith("Z")
    for path in log_dir.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    # the console still gets the human-readable line (stderr), stdout stays machine-readable
    assert "boot warning" in result.stderr
    # when the command ends the root removes its handlers again (nothing leaks into the next invocation)
    assert [h for h in logging.getLogger().handlers if getattr(h, "_jevbot_handler", False)] == []


def test_the_jsonl_log_is_redacted(boot: list[str], data_dir: Path) -> None:
    result = runner.invoke(app, [*boot, "fake", "boot", "--secret-in-log"])
    assert result.exit_code == 0, result.output
    text = (data_dir / "logs" / f"jevbot-{datetime.now(UTC).date().isoformat()}.jsonl").read_text(encoding="utf-8")
    assert SECRET not in text and SECRET not in result.output
    redacted = [r for r in jsonl_records(data_dir) if "authenticating" in str(r["msg"])]
    assert len(redacted) == 1 and redacted[0]["msg"] == f"authenticating with key {logsetup.REDACTED}"


def test_load_honours_data_dir_and_log_level(boot: list[str], tmp_path: Path) -> None:
    override = tmp_path / "elsewhere"
    got = shown(runner.invoke(app, ["--log-level", "debug", *boot, "fake", "boot", "--data-dir", str(override)]))
    assert got["data_dir"] == str(override.resolve()) and got["config_data_dir"] == str(override.absolute())
    assert got["log_dirs"] == [str(override.resolve() / "logs")] and got["root_level"] == logging.DEBUG
    assert stat.S_IMODE(override.stat().st_mode) == 0o700  # created by ensure_data_dir (D1)
    messages = [r["msg"] for r in jsonl_records(override) if r["logger"] == "jevbot.test.boot"]
    assert messages == ["boot debug", "boot info", "boot warning"]
    quiet = shown(runner.invoke(app, [*boot, "fake", "boot", "--data-dir", str(tmp_path / "quiet"), "--log-level", "error"]))
    assert quiet["root_level"] == logging.ERROR and quiet["jsonl_handlers"] == 1
    # nothing at or above ERROR was logged: the handler is attached (its directory exists) but the day file is opened on first use
    assert (tmp_path / "quiet" / "logs").is_dir() and list((tmp_path / "quiet" / "logs").iterdir()) == []


def test_load_refuses_what_secrets_refuses_with_the_2_9_exit_codes(
    boot: list[str], monkeypatch: pytest.MonkeyPatch, data_dir: Path
) -> None:
    monkeypatch.setenv("APCA_API_KEY_ID", "x")  # ambiguous credentials (INV-02)
    result = runner.invoke(app, [*boot, "fake", "boot"])
    assert result.exit_code == 4 and "error: PaperGuardError" in result.stderr and result.stdout == ""
    monkeypatch.delenv("APCA_API_KEY_ID")
    monkeypatch.setenv("_".join(("TYPESAFE", "LOG", "LEVEL")), " debug")  # INV-18: refused before any SDK import
    result = runner.invoke(app, [*boot, "fake", "boot"])
    assert result.exit_code == 2 and "error: ConfigError" in result.stderr
    assert not (data_dir / "logs").exists()  # nothing was booted, so no log file was opened


def test_load_refuses_a_data_dir_inside_the_repository(boot: list[str]) -> None:
    result = runner.invoke(app, [*boot, "fake", "boot", "--data-dir", str(REPO / "tests" / "_t_data_inside_repo")])
    assert result.exit_code == 2 and "inside the git repository" in result.stderr
    assert not (REPO / "tests" / "_t_data_inside_repo").exists()


def test_load_without_the_root_uses_the_process_environment(data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # a sub-app's own tests boot without the root: GlobalOptions(...).load() reads os.environ (the offline_env fixture) ...
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    options = GlobalOptions(log_level="warning", overrides=(no_env_file(tmp_path),))
    loaded = options.load()
    try:
        assert isinstance(loaded, main.LoadedConfig)
        assert loaded.data_dir == data_dir.resolve() and loaded.secrets.typesafe_api_key == SECRET and loaded.cfg.risk.max_new_per_day == 2
        assert logging.getLogger().level == logging.WARNING
        assert [type(h).__name__ for h in logging.getLogger().handlers if getattr(h, "_jevbot_handler", False)] == [
            "StreamHandler",
            "DailyJsonlHandler",
        ]
        # ... or an explicit mapping
        explicit = options.load(env={"JEVBOT_DATA": str(data_dir)})
        assert explicit.secrets.typesafe_api_key is None and explicit.data_dir == data_dir.resolve()
        # configure_logging is idempotent: handlers are replaced, never doubled
        again = options.configure_logging(data_dir)
        assert again == data_dir / "logs" and len([h for h in logging.getLogger().handlers if getattr(h, "_jevbot_handler", False)]) == 2
    finally:
        logsetup.reset()


# ======================================================================================================================
# Exit codes per 2.9, one redacted line, no traceback
# ======================================================================================================================

ERRORS = [
    (ConfigError("bad config"), 2),
    (PaperGuardError("not the paper host"), 4),
    (CacheMissError("replay miss"), 5),
    (ModelMismatchError("jev-1.14.0 != jev-1.13.0"), 6),
    (SpendLimitError("day ceiling"), 7),
    (DataUnavailable("no snapshot"), 8),
    (PitViolation("future row"), 8),
    (LedgerCorrupt("hash chain broken at seq 17"), 9),
    (InvariantError("bug"), 1),
    (BrokerRejected("rejected", status=403, reject_code=40310000), 1),
    (JevbotError("generic"), 1),
]


@pytest.mark.parametrize(("error", "code"), ERRORS, ids=[type(e).__name__ for e, _ in ERRORS])
def test_jevbot_errors_map_to_the_exit_codes_of_2_9(monkeypatch: pytest.MonkeyPatch, error: JevbotError, code: int) -> None:
    sub = typer.Typer()

    @sub.command()
    def boom() -> None:
        raise error

    @sub.command()
    def other() -> None:
        typer.echo("other")

    install_module(monkeypatch, FAKE, app=sub)
    monkeypatch.setattr(main, "SUBAPPS", [("fake", FAKE)])
    result = runner.invoke(app, ["fake", "boom"])
    assert result.exit_code == code
    assert f"error: {type(error).__name__}: " in result.stderr
    assert "Traceback" not in result.output
    assert result.stdout == ""


def test_commands_can_refuse_with_exit_3_and_unexpected_errors_are_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = typer.Typer()

    @sub.command()
    def refuse() -> None:
        typer.echo("kill switch active", err=True)
        raise typer.Exit(3)

    @sub.command()
    def crash() -> None:
        raise ZeroDivisionError("a plain bug")

    install_module(monkeypatch, FAKE, app=sub)
    monkeypatch.setattr(main, "SUBAPPS", [("fake", FAKE)])
    assert runner.invoke(app, ["fake", "refuse"]).exit_code == 3
    crashed = runner.invoke(app, ["fake", "crash"])
    assert crashed.exit_code == 1 and isinstance(crashed.exception, ZeroDivisionError)


def test_the_error_line_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "PKTESTSECRET-0123456789abcdef"
    sub = typer.Typer()

    @sub.command()
    def boom() -> None:
        logsetup.register_secret(secret)
        raise ConfigError(f"could not authenticate with key {secret} at the vendor")

    @sub.command()
    def other() -> None:
        typer.echo("other")

    install_module(monkeypatch, FAKE, app=sub)
    monkeypatch.setattr(main, "SUBAPPS", [("fake", FAKE)])
    saved = set(logsetup._secrets)
    try:
        result = runner.invoke(app, ["--log-level", "debug", "fake", "boom"])
    finally:
        logsetup.clear_secrets()
        for value in saved:
            logsetup.register_secret(value)
    assert result.exit_code == 2
    assert secret not in result.output
    assert logsetup.REDACTED in result.stderr


# ======================================================================================================================
# Logging (INV-18): configured by the root before the sub-app is imported, debug never unpins the SDK loggers
# ======================================================================================================================


def test_log_level_is_applied_and_third_party_loggers_stay_pinned(fake: typer.Typer) -> None:
    before = shown(runner.invoke(app, ["--log-level", "debug", "fake", "show"]))
    assert before["root_logger_level"] == logging.DEBUG
    assert before["sdk_logger_level"] >= logging.WARNING
    after = shown(runner.invoke(app, ["fake", "show", "--log-level", "error"]))  # given after the command: re-applied at dispatch
    assert after["root_logger_level"] == logging.ERROR
    assert after["sdk_logger_level"] >= logging.WARNING


def test_the_root_removes_its_log_handlers_when_the_command_ends(fake: typer.Typer) -> None:
    runner.invoke(app, ["fake", "show"])
    assert [h for h in logging.getLogger().handlers if getattr(h, "_jevbot_handler", False)] == []


def test_logging_is_configured_before_the_subapp_module_is_imported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "t_lazy_cmds.py").write_text(
        textwrap.dedent(
            """
            import logging
            import typer
            HANDLERS_AT_IMPORT = [h for h in logging.getLogger().handlers if getattr(h, "_jevbot_handler", False)]
            app = typer.Typer()

            @app.command()
            def one() -> None:
                typer.echo(f"handlers_at_import={len(HANDLERS_AT_IMPORT)}")

            @app.command()
            def two() -> None:
                pass
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(main, "SUBAPPS", [("lazy", "t_lazy_cmds")])
    assert "t_lazy_cmds" not in sys.modules
    assert runner.invoke(app, ["--help"]).exit_code == 0
    assert "t_lazy_cmds" not in sys.modules  # listing the areas does not import them
    result = runner.invoke(app, ["lazy", "one"])
    assert result.stdout.strip() == "handlers_at_import=1"
    assert "t_lazy_cmds" in sys.modules
    sys.modules.pop("t_lazy_cmds", None)


# ======================================================================================================================
# The PAPER-ONLY banner
# ======================================================================================================================


def test_banner_text() -> None:
    assert "PAPER-ONLY" in PAPER_ONLY_BANNER
    assert "PAPER" in PAPER_ONLY_BANNER and "INV-01" in PAPER_ONLY_BANNER
    assert main.BANNER_SUBAPPS == ("paper", "record")


@pytest.mark.parametrize("area", ["paper", "record"])
def test_broker_areas_print_the_banner_first_once_and_on_stderr(monkeypatch: pytest.MonkeyPatch, area: str) -> None:
    sub = typer.Typer()

    @sub.command()
    def status(ctx: typer.Context) -> None:
        paper_only_banner(ctx)  # a command that prints it itself does not double it
        typer.echo(json.dumps({"ok": True}))

    @sub.command()
    def other() -> None:
        typer.echo("other")

    install_module(monkeypatch, FAKE, app=sub)
    monkeypatch.setattr(main, "SUBAPPS", [(area, FAKE)])
    result = runner.invoke(app, [area, "status", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"ok": True}  # stdout stays machine-readable
    assert result.stderr.count("PAPER-ONLY") == 1
    assert result.output.index("PAPER-ONLY") < result.output.index('{"ok"')  # printed FIRST


def test_other_areas_do_not_print_the_banner_unless_the_command_asks(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = typer.Typer()

    @sub.command()
    def offline() -> None:
        typer.echo("offline")

    @sub.command()
    def online(ctx: typer.Context) -> None:
        paper_only_banner(ctx)
        paper_only_banner(ctx)
        typer.echo("online")

    install_module(monkeypatch, FAKE, app=sub)
    monkeypatch.setattr(main, "SUBAPPS", [("doctor", FAKE)])
    assert "PAPER-ONLY" not in runner.invoke(app, ["doctor", "offline"]).output
    assert runner.invoke(app, ["doctor", "online"]).stderr.count("PAPER-ONLY") == 1


def test_banner_is_printed_even_when_the_paper_area_is_not_built(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "SUBAPPS", [("paper", GHOST)])
    result = runner.invoke(app, ["paper", "run"])
    assert result.exit_code == 1
    assert "PAPER-ONLY" in result.stderr and NOT_BUILT in result.stdout


def test_banner_without_a_context_always_prints(capsys: pytest.CaptureFixture[str]) -> None:
    paper_only_banner()
    paper_only_banner()
    assert capsys.readouterr().err.count("PAPER-ONLY") == 2


# ======================================================================================================================
# Extenders (`eval leakage`)
# ======================================================================================================================

EXT = "jevbot.cli._t_fake_extender"


def eval_area() -> typer.Typer:
    sub = typer.Typer()

    @sub.command()
    def power() -> None:
        typer.echo("power ran")

    @sub.command()
    def trials() -> None:
        typer.echo("trials ran")

    return sub


def test_an_extender_adds_commands_to_its_target_area_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[typer.Typer] = []

    def register(sub: typer.Typer) -> None:
        calls.append(sub)

        @sub.command()
        def leakage(ctx: typer.Context, which: str = typer.Argument("all")) -> None:
            typer.echo(f"leakage {which} json={get_globals(ctx).json}")

    eval_app, other_app = eval_area(), fake_area()
    install_module(monkeypatch, FAKE, app=eval_app, other=other_app)
    install_module(monkeypatch, EXT, register=register)
    monkeypatch.setattr(main, "SUBAPPS", [("eval", FAKE), ("fake", f"{FAKE}:other")])
    monkeypatch.setattr(main, "EXTENDERS", [(EXT, "register")])
    monkeypatch.setattr(main, "EXTENDER_TARGETS", {EXT: ("eval", ("leakage",))})

    assert runner.invoke(app, ["fake", "other"]).stdout.strip() == "other ran"
    assert calls == []  # an unrelated area never touches the extender
    assert runner.invoke(app, ["eval", "leakage", "recall", "--json"]).stdout.strip() == "leakage recall json=True"
    assert runner.invoke(app, ["eval", "power"]).stdout.strip() == "power ran"
    assert "leakage" in runner.invoke(app, ["eval", "--help"]).output
    assert calls == [eval_app]  # applied exactly once although the area was loaded three times


def test_a_missing_extender_leaves_a_not_built_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    install_module(monkeypatch, FAKE, app=eval_area())
    monkeypatch.setattr(main, "SUBAPPS", [("eval", FAKE)])
    monkeypatch.setattr(main, "EXTENDERS", [(GHOST, "register")])
    monkeypatch.setattr(main, "EXTENDER_TARGETS", {GHOST: ("eval", ("leakage",))})
    assert runner.invoke(app, ["eval", "trials"]).stdout.strip() == "trials ran"  # the area itself works
    for _ in range(2):  # loading the area again does not pile up placeholders
        result = runner.invoke(app, ["eval", "leakage", "masked", "RUN", "--n", "5"])
        assert result.exit_code == 1
        assert NOT_BUILT in result.stdout and GHOST in result.stdout
    listing = runner.invoke(app, ["eval", "--help"]).output
    assert listing.count("leakage") == 1


def test_a_broken_extender_contract_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    install_module(monkeypatch, FAKE, app=eval_area())
    install_module(monkeypatch, EXT, register="not callable")
    monkeypatch.setattr(main, "SUBAPPS", [("eval", FAKE)])
    monkeypatch.setattr(main, "EXTENDERS", [(EXT, "register")])
    monkeypatch.setattr(main, "EXTENDER_TARGETS", {EXT: ("eval", ("leakage",))})
    result = runner.invoke(app, ["eval", "power"])
    assert result.exit_code == 1 and "is not callable" in result.stderr
    monkeypatch.setattr(main, "EXTENDER_TARGETS", {})
    result = runner.invoke(app, ["eval", "power"])
    assert result.exit_code == 1 and "has no EXTENDER_TARGETS entry" in result.stderr
