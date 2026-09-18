"""Guard: forbidden project configuration (DESIGN.md 1.1, 15.5; D26, D29, INV-18).

- the only package index is PyPI: no extra index, no find-links, no non-PyPI source - in `pyproject.toml`, in `uv.lock` and in
  any uv / pip configuration file shipped with the repository;
- the two vendor SDKs carry exact pins (`typesafe-sdk==0.6.0`, `alpaca-py==0.44.0`) and `uv.lock` pins them (and `websockets`);
  the vendor's superseded distributions (`typesafe-client`, `cooksafe`) never appear;
- nothing in the repository SETS the SDK's log-level variable: no repo file that can feed an environment (`.env.example`, TOML,
  systemd units, scripts, the Windows watchdog, CI files ...) mentions it, and no source module assigns it. At DEBUG the SDK logs
  full request and response bodies; `config.check_sdk_log_level` refuses such a value at startup, and this guard keeps the
  repository from ever shipping one.
"""

import ast
import os
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

from jevbot import config
from jevbot.errors import ConfigError

REPO = Path(__file__).resolve().parents[2]
PYPROJECT = REPO / "pyproject.toml"
UV_LOCK = REPO / "uv.lock"

PYPI = "https://pypi.org/simple"
EXACT_PINS = {"typesafe-sdk": "0.6.0", "alpaca-py": "0.44.0"}
SUPERSEDED = ("typesafe-client", "cooksafe")

# assembled from fragments so that this file never contains the token it hunts for
LOG_LEVEL_TOKEN = "_".join(("TYPESAFE", "LOG", "LEVEL"))

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "htmlcov",
    "build",
    "dist",
}
SKIP_DIRS |= {"runs", "logs", "data", "jevbot-data"}  # git-ignored run artefacts, should they ever sit inside the checkout
# prose and code are handled separately: documentation may EXPLAIN the rule, tests exercise the guard with the literal,
# and source modules are checked structurally (AST) below
SKIP_TOP_LEVEL = {"docs", "tests", "src"}
SKIP_SUFFIXES = {".md"}


def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def lock() -> dict:
    return tomllib.loads(UV_LOCK.read_text(encoding="utf-8"))


# ======================================================================================================================
# no extra index
# ======================================================================================================================


def index_problems(data: dict) -> list[str]:
    """Every way a pyproject / uv.toml can point the resolver away from PyPI."""
    problems: list[str] = []
    tool = data.get("tool", {})
    uv = tool.get("uv", {})
    for key in ("index", "index-url", "extra-index-url", "find-links", "sources", "no-index", "index-strategy"):
        if key in uv:
            problems.append(f"tool.uv.{key}")
    pip = uv.get("pip", {})
    for key in ("index-url", "extra-index-url", "find-links", "no-index", "index-strategy"):
        if key in pip:
            problems.append(f"tool.uv.pip.{key}")
    if "pip" in tool:
        problems.append("tool.pip")
    for dep in data.get("project", {}).get("dependencies", []):
        if "@" in dep or "://" in dep:
            problems.append(f"direct reference dependency: {dep}")
    return problems


def test_pyproject_configures_no_extra_index() -> None:
    assert index_problems(pyproject()) == []


def test_index_detector_catches_each_form() -> None:
    assert index_problems(tomllib.loads('[[tool.uv.index]]\nurl = "https://example.invalid/simple"\n')) == ["tool.uv.index"]
    assert index_problems(tomllib.loads('[tool.uv]\nextra-index-url = ["https://example.invalid/simple"]\n')) == ["tool.uv.extra-index-url"]
    assert index_problems(tomllib.loads('[tool.uv]\nfind-links = ["./wheels"]\n')) == ["tool.uv.find-links"]
    assert index_problems(tomllib.loads('[tool.uv.sources]\ntypesafe-sdk = { git = "https://example.invalid/sdk" }\n')) == [
        "tool.uv.sources"
    ]
    assert index_problems(tomllib.loads('[tool.uv.pip]\nextra-index-url = ["https://example.invalid/simple"]\n')) == [
        "tool.uv.pip.extra-index-url"
    ]
    assert index_problems(tomllib.loads('[project]\ndependencies = ["typesafe-sdk @ https://example.invalid/sdk.whl"]\n')) == [
        "direct reference dependency: typesafe-sdk @ https://example.invalid/sdk.whl"
    ]
    assert index_problems(tomllib.loads('[project]\ndependencies = ["numpy>=2.0"]\n[tool.uv]\ndefault-groups = ["dev"]\n')) == []


@pytest.mark.parametrize("name", ["uv.toml", "pip.conf", "pip.ini", ".pypirc"])
def test_no_side_channel_index_configuration(name: str) -> None:
    """`pyproject.toml` is the one place that could configure an index (and it configures none): no uv / pip configuration
    file sits next to it."""
    assert not (REPO / name).exists(), f"{name} must not exist in the repository root"


def test_uv_lock_resolves_from_pypi_only() -> None:
    packages = lock()["package"]
    assert len(packages) > 20
    foreign = []
    for package in packages:
        source = package.get("source", {})
        if source == {"registry": PYPI}:
            continue
        if package["name"] == "jevbot" and source in ({"editable": "."}, {"virtual": "."}):
            continue
        foreign.append((package["name"], source))
    assert foreign == [], "every locked distribution comes from PyPI (the project itself is the only local source)"
    for package in packages:
        for artefact in (package.get("sdist"), *package.get("wheels", [])):
            if artefact and "url" in artefact:
                assert artefact["url"].startswith("https://files.pythonhosted.org/"), (package["name"], artefact["url"])


# ======================================================================================================================
# exact pins
# ======================================================================================================================


def test_vendor_sdks_are_pinned_exactly_in_pyproject() -> None:
    deps = pyproject()["project"]["dependencies"]
    for name, version in EXACT_PINS.items():
        matching = [d for d in deps if d.replace(" ", "").lower().startswith(name)]
        assert matching == [f"{name}=={version}"], f"{name} must carry the exact pin =={version}"


def test_uv_lock_pins_the_vendor_sdks_and_websockets() -> None:
    by_name = {p["name"]: p for p in lock()["package"]}
    for name, version in EXACT_PINS.items():
        assert by_name[name]["version"] == version
    assert by_name["websockets"]["version"], "websockets is pinned by uv.lock (1.1)"
    requires = {d["name"]: d.get("specifier") for d in by_name["jevbot"]["metadata"]["requires-dist"]}
    for name, version in EXACT_PINS.items():
        assert requires[name] == f"=={version}"


def test_pins_agree_with_the_config_default() -> None:
    assert config.Config().jev.sdk_version == EXACT_PINS["typesafe-sdk"], "jev.sdk_version is asserted against typesafe_sdk.__version__"


@pytest.mark.parametrize("name", SUPERSEDED)
def test_superseded_vendor_distributions_are_absent(name: str) -> None:
    assert name not in {p["name"] for p in lock()["package"]}
    assert name not in UV_LOCK.read_text(encoding="utf-8")
    assert not [d for d in pyproject()["project"]["dependencies"] if name in d.lower()]


# ======================================================================================================================
# the SDK log-level variable is never set by anything in the repository (INV-18)
# ======================================================================================================================


def environment_feeding_files() -> Iterator[Path]:
    """Every non-prose, non-Python-source file of the repository: `.env.example`, TOML, lock file, systemd units, PowerShell,
    XML, shell scripts, CI definitions, dotfiles ..."""
    for root, dirs, files in os.walk(REPO):
        top = Path(root) == REPO
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not (top and d in SKIP_TOP_LEVEL))
        for name in sorted(files):
            path = Path(root) / name
            if path.is_symlink() or path.suffix.lower() in SKIP_SUFFIXES:
                continue
            if (name == ".env" or name.startswith(".env.")) and name != ".env.example":
                continue  # the operator's git-ignored secrets file is not a repository file (and is never read by a test)
            yield path


def files_mentioning(token: str, paths: Iterator[Path]) -> list[str]:
    hits = []
    for path in paths:
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue  # binary artefact: cannot feed an environment
        if token in text:
            hits.append(str(path))
    return hits


def test_no_repo_file_mentions_the_sdk_log_level_variable() -> None:
    scanned = list(environment_feeding_files())
    names = {p.relative_to(REPO).as_posix() for p in scanned}
    assert {"pyproject.toml", "uv.lock", ".env.example", "config/default.toml", "config/paper.toml"} <= names, (
        "the scan must reach the files that matter"
    )
    assert files_mentioning(LOG_LEVEL_TOKEN, iter(scanned)) == []


def test_file_scanner_detects_the_token(tmp_path: Path) -> None:
    unit = tmp_path / "jevbot-paper.service"
    unit.write_text(f"[Service]\nEnvironment={LOG_LEVEL_TOKEN}=debug\n", encoding="utf-8")
    clean = tmp_path / "clean.service"
    clean.write_text("[Service]\nEnvironmentFile=%h/jevbot/.env\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00binary")
    assert files_mentioning(LOG_LEVEL_TOKEN, iter(sorted(tmp_path.iterdir()))) == [str(unit)]


def _is_token(node: ast.expr) -> bool:
    """The literal variable name, or a reference to config's constant for it."""
    if isinstance(node, ast.Constant):
        return node.value == LOG_LEVEL_TOKEN
    name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
    return name == "ENV_" + LOG_LEVEL_TOKEN


def environment_writes(source: str) -> list[int]:
    """Line numbers where Python code SETS the variable: `environ[...] = v`, `del` / augmented assignment, `setdefault`,
    `putenv`, `setenv`, `update(NAME=...)` / `dict(environ, NAME=...)`, or a dict literal with that key handed to a call's `env=`."""
    lines: list[int] = []
    for node in ast.walk(ast.parse(source)):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AugAssign | ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = list(node.targets)
        for target in targets:
            if isinstance(target, ast.Subscript) and _is_token(target.slice):
                lines.append(node.lineno)
        if isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
            if called in {"setdefault", "putenv", "setenv"} and node.args and _is_token(node.args[0]):
                lines.append(node.lineno)
            if any(kw.arg == LOG_LEVEL_TOKEN for kw in node.keywords):
                lines.append(node.lineno)
            for kw in node.keywords:
                if kw.arg == "env" and isinstance(kw.value, ast.Dict) and any(k is not None and _is_token(k) for k in kw.value.keys):
                    lines.append(node.lineno)
    return sorted(set(lines))


def test_no_source_module_sets_the_sdk_log_level_variable() -> None:
    modules = sorted((REPO / "src").rglob("*.py"))
    assert REPO / "src" / "jevbot" / "config.py" in modules
    offenders = {str(p.relative_to(REPO)): hits for p in modules if (hits := environment_writes(p.read_text(encoding="utf-8")))}
    assert offenders == {}


def test_environment_write_detector_catches_each_form() -> None:
    name = LOG_LEVEL_TOKEN
    flagged = [
        f'import os\nos.environ["{name}"] = "debug"\n',
        f'import os\nos.environ["{name}"] += "x"\n',
        f'import os\nos.environ.setdefault("{name}", "debug")\n',
        f'import os\nos.putenv("{name}", "debug")\n',
        f'def f(monkeypatch):\n    monkeypatch.setenv("{name}", "debug")\n',
        f'import os\nos.environ.update({name}="debug")\n',
        f'import os\nenv = dict(os.environ, {name}="debug")\n',
        f'import subprocess\nsubprocess.run(["x"], env={{"{name}": "debug"}})\n',
        f"import os\nfrom jevbot import config\nos.environ[config.ENV_{name}] = 'debug'\n",
        f"import os\nfrom jevbot.config import ENV_{name}\nos.environ.setdefault(ENV_{name}, 'info')\n",
    ]
    for source in flagged:
        assert environment_writes(source), source
    harmless = [
        f'import os\nlevel = os.environ.get("{name}")\n',
        f'ENV_{name} = "{name}"\n',
        f'def check(env):\n    return (env.get("{name}") or "").strip().lower()\n',
        f'report = {{"{name}": "ok"}}\n',
        'import os\nos.environ["JEVBOT_DATA"] = "/tmp/x"\n',
    ]
    for source in harmless:
        assert environment_writes(source) == [], source


def test_the_startup_guard_behind_this_rule_exists() -> None:
    """The repository never ships the variable, and a value injected from outside is refused before the SDK is imported."""
    assert config.ENV_TYPESAFE_LOG_LEVEL == LOG_LEVEL_TOKEN
    with pytest.raises(ConfigError):
        config.check_sdk_log_level({LOG_LEVEL_TOKEN: "debug"})
    config.check_sdk_log_level({LOG_LEVEL_TOKEN: "warning"})
