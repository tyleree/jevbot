"""Guard: there is no live-trading path (DESIGN 11.1; D2, INV-01, INV-02).

Three layers, exactly as 11.1 prescribes.

(a) **Grep** of `src/` and `deploy/` for anything that could point the bot at a live account: the live trading host, the
    literal that would turn the paper flag off, the override-URL keyword, the live base-URL enum member and the legacy
    credential environment names. Every pattern is assembled from fragments inside this file, so the guard file itself
    stays clean and a naive grep of the repository still finds nothing. The one carve-out is the refusal list itself:
    `config.FORBIDDEN_ALPACA_ENV` has to spell the legacy names out in order to refuse them, so a legacy name may appear
    as a string literal inside that constant, or in a comment or docstring that explains the refusal - never in code.

(b) **AST**: a call to `TradingClient` / `TradingStream` may appear only inside `make_clients`, and must pass `paper` as
    the literal constant `True` - not a variable, not an expression.

(c) **Runtime**: a client that came back with the live URL, a key that looks live, an account below options level 3, a
    blocked account and a forbidden environment variable each raise `PaperGuardError` (exit 4).

Every detector is first proven to fire on planted source, then applied to the real tree.
"""

import ast
import io
import textwrap
import tokenize
from pathlib import Path

import pytest

from jevbot.config import FORBIDDEN_ALPACA_ENV, Secrets, secrets
from jevbot.errors import PaperGuardError
from jevbot.paper.alpaca_client import (
    PAPER_BASE_URL,
    assert_no_forbidden_env,
    assert_paper_base_url,
    assert_tradable_account,
    paper_credentials,
)
from tests.fixtures.fake_alpaca import FakeRestClient, FakeTradingClient

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
DEPLOY = REPO / "deploy"
PACKAGE = SRC / "jevbot"
CONFIG_MODULE = PACKAGE / "config.py"
CONSTRUCTION_SITE = PACKAGE / "paper" / "alpaca_client.py"
CONSTRUCTOR_FUNCTION = "make_clients"

# --- the forbidden strings, assembled from fragments so this file contains none of them whole ---------------------------

LIVE_HOST = "api" + "." + "alpaca" + "." + "markets"
LIVE_HOST_PATTERN = r"(?<!paper-)" + LIVE_HOST.replace(".", r"\.")
PAPER_OFF_PATTERN = r"\bpaper\b\s*[:=][^=\n]*\bFalse\b"
OVERRIDE_URL = "url" + "_" + "override"
LIVE_ENUM = "TRADING" + "_" + "LIVE"
LEGACY_ENV_NAMES = ("ALPACA" + "_API_KEY", "ALPACA" + "_SECRET_KEY", "AP" + "CA_")

# the patterns of layer (a) that must not appear ANYWHERE under src/ and deploy/, comments and docstrings included
FORBIDDEN_ANYWHERE: tuple[tuple[str, str], ...] = (
    ("live trading host", LIVE_HOST_PATTERN),
    ("the paper flag turned off", PAPER_OFF_PATTERN),
    ("the override-URL keyword", OVERRIDE_URL),
    ("the live base-URL enum member", LIVE_ENUM),
)

SCANNED_SUFFIXES = {".py", ".toml", ".service", ".timer", ".ps1", ".xml", ".sh", ".md", ".json", ".conf", ""}


# ======================================================================================================================
# Helpers
# ======================================================================================================================


def scanned_files(*roots: Path) -> list[Path]:
    """Every text file under the roots. A missing root (deploy/ before WP12 ships it) contributes nothing."""
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        found += [
            p for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.suffix.lower() in SCANNED_SUFFIXES
        ]
    return found


def label(path: Path) -> str:
    """The repo-relative path for a real source file, the plain path for a planted one under tmp_path."""
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def grep(pattern: str, paths: list[Path]) -> list[str]:
    """`<path>:<line>` for every line matching `pattern`, over whole files (comments and docstrings included)."""
    import re

    compiled = re.compile(pattern)
    hits: list[str] = []
    for path in paths:
        text = read(path)
        if text is None:
            continue
        hits += [
            f"{path.relative_to(REPO).as_posix()}:{number}: {line.strip()[:110]}"
            for number, line in enumerate(text.splitlines(), start=1)
            if compiled.search(line)
        ]
    return hits


def code_only(source: str) -> str:
    """The source with every string literal and comment blanked out, so a grep sees CODE only."""
    out: list[str] = []
    line, column = 1, 0
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        start_line, start_col = token.start
        end_line, end_col = token.end
        while line < start_line:
            out.append("\n")
            line, column = line + 1, 0
        out.append(" " * max(0, start_col - column))
        text = token.string
        if token.type in (tokenize.STRING, tokenize.COMMENT, tokenize.FSTRING_START, tokenize.FSTRING_MIDDLE):
            text = "".join("\n" if ch == "\n" else " " for ch in text)
        out.append(text)
        line, column = end_line, end_col
    return "".join(out)


def docstring_ids(tree: ast.AST) -> set[int]:
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                found.add(id(first.value))
    return found


def refusal_list_ids(tree: ast.AST) -> set[int]:
    """Ids of every constant inside the `FORBIDDEN_ALPACA_ENV` / legacy-prefix assignments: the one place a legacy name
    may be written out, because refusing a name requires naming it."""
    allowed: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign | ast.AnnAssign) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if any("FORBIDDEN" in name and "ALPACA" in name for name in names):
            allowed |= {id(sub) for sub in ast.walk(node.value)}
    return allowed


def legacy_name_problems(path: Path, source: str) -> list[str]:
    """Every legacy credential name in `source` that is not part of the refusal list, a comment or a docstring."""
    where = path.relative_to(REPO).as_posix()
    problems: list[str] = []

    for number, line in enumerate(code_only(source).splitlines(), start=1):
        problems += [f"{where}:{number}: {name} in CODE" for name in LEGACY_ENV_NAMES if name in line]

    tree = ast.parse(source)
    exempt = refusal_list_ids(tree) | docstring_ids(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in exempt:
            problems += [
                f"{where}:{node.lineno}: {name} in a string literal outside the refusal list"
                for name in LEGACY_ENV_NAMES
                if name in node.value
            ]
    return sorted(set(problems))


TRADING_CLIENT_CLASSES = ("TradingClient", "TradingStream")


def construction_calls(tree: ast.AST) -> list[tuple[ast.Call, str]]:
    """Every call to a trading-client class, paired with the INNERMOST enclosing function ("" at module level)."""
    found: list[tuple[ast.Call, str]] = []

    def visit(node: ast.AST, function: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                visit(child, child.name)
                continue
            if isinstance(child, ast.Call) and _called_name(child) in TRADING_CLIENT_CLASSES:
                found.append((child, function))
            visit(child, function)

    visit(tree, "")
    return found


def _called_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    return func.attr if isinstance(func, ast.Attribute) else ""


def paper_flag_problems(path: Path, source: str) -> list[str]:
    """Every trading-client construction that is not inside `make_clients` with a literal `paper=True` (INV-01)."""
    where = path.relative_to(REPO).as_posix()
    tree = ast.parse(source)
    problems: list[str] = []
    for call, function in construction_calls(tree):
        label = f"{where}:{call.lineno}: {_called_name(call)}("
        if function != CONSTRUCTOR_FUNCTION:
            problems.append(f"{label} outside {CONSTRUCTOR_FUNCTION} (in {function or '<module level>'})")
            continue
        keyword = next((k for k in call.keywords if k.arg == "paper"), None)
        if keyword is None:
            problems.append(f"{label} without the paper keyword")
        elif not (isinstance(keyword.value, ast.Constant) and keyword.value.value is True):
            problems.append(f"{label} with paper={ast.unparse(keyword.value)}, which is not the literal True")
    return sorted(set(problems))


# ======================================================================================================================
# (a) The detectors fire on planted source ...
# ======================================================================================================================


@pytest.mark.parametrize(
    ("label", "planted"),
    [
        ("live trading host", 'BASE = "https://' + LIVE_HOST + '"'),
        ("live trading host", "# talk to " + LIVE_HOST + " when we go live"),
        ("the paper flag turned off", "client = TradingClient(key, secret, paper=False)"),
        ("the paper flag turned off", "class C:\n    paper: bool = False"),
        ("the override-URL keyword", f"c = TradingClient(key, {OVERRIDE_URL}='http://localhost')"),
        ("the live base-URL enum member", f"BASE = BaseURL.{LIVE_ENUM}"),
    ],
)
def test_the_grep_layer_fires_on_planted_source(label: str, planted: str, tmp_path: Path) -> None:
    path = tmp_path / "planted.py"
    path.write_text(planted + "\n", encoding="utf-8")
    pattern = next(p for name, p in FORBIDDEN_ANYWHERE if name == label)

    assert grep(pattern, [path]), planted


def test_the_paper_host_itself_is_not_a_hit(tmp_path: Path) -> None:
    path = tmp_path / "ok.py"
    path.write_text(f'PAPER = "{PAPER_BASE_URL}"\nDATA = "https://data.alpaca.markets"\n', encoding="utf-8")

    assert grep(LIVE_HOST_PATTERN, [path]) == []


def test_a_paper_true_line_is_not_a_paper_off_hit(tmp_path: Path) -> None:
    path = tmp_path / "ok.py"
    path.write_text("c = TradingClient(api_key=k, secret_key=s, paper=True)\n", encoding="utf-8")

    assert grep(PAPER_OFF_PATTERN, [path]) == []


def test_the_legacy_name_detector_separates_code_from_the_refusal_list(tmp_path: Path) -> None:
    legacy = LEGACY_ENV_NAMES[0]
    bad = tmp_path / "bad.py"
    bad.write_text(f'import os\nKEY = os.environ["{legacy}"]\n', encoding="utf-8")
    assert legacy_name_problems(bad, bad.read_text(encoding="utf-8"))

    good = tmp_path / "good.py"
    good.write_text(
        f'"""We refuse {legacy} (INV-02)."""\n\n# {legacy} is ambiguous\nFORBIDDEN_ALPACA_ENV = ("{legacy}",)\n',
        encoding="utf-8",
    )
    assert legacy_name_problems(good, good.read_text(encoding="utf-8")) == []


def test_a_legacy_name_in_an_ordinary_string_is_a_problem(tmp_path: Path) -> None:
    legacy = LEGACY_ENV_NAMES[1]
    path = tmp_path / "sneaky.py"
    path.write_text(f'def f(env):\n    return env.get("{legacy}")\n', encoding="utf-8")

    problems = legacy_name_problems(path, path.read_text(encoding="utf-8"))

    assert problems and "string literal outside the refusal list" in problems[0]


def test_code_only_blanks_strings_and_comments_without_moving_lines() -> None:
    source = 'X = "hidden"  # also hidden\nY = visible\n'
    stripped = code_only(source)

    assert "hidden" not in stripped
    assert "visible" in stripped
    assert len(stripped.splitlines()) == len(source.splitlines())


# ======================================================================================================================
# ... and the real tree is clean
# ======================================================================================================================


def test_the_scan_actually_covers_the_package() -> None:
    names = {p.relative_to(SRC).as_posix() for p in scanned_files(SRC)}
    assert {"jevbot/config.py", "jevbot/paper/alpaca_client.py", "jevbot/paper/broker.py"} <= names


def test_deploy_is_scanned_the_moment_it_exists() -> None:
    # WP12 owns deploy/ (systemd units, the Windows watchdog); 11.1 greps it too, so this guard must not go blind on it
    if DEPLOY.exists():
        assert scanned_files(DEPLOY), f"{DEPLOY} exists but nothing under it is scanned: widen SCANNED_SUFFIXES"
    else:
        assert scanned_files(DEPLOY) == []
        assert grep(LIVE_HOST_PATTERN, scanned_files(DEPLOY)) == []


@pytest.mark.parametrize(("label", "pattern"), FORBIDDEN_ANYWHERE)
def test_src_and_deploy_are_free_of_every_live_trading_marker(label: str, pattern: str) -> None:
    hits = grep(pattern, scanned_files(SRC, DEPLOY))

    assert hits == [], f"INV-01: {label} found under src/ or deploy/:\n" + "\n".join(hits)


def test_the_legacy_credential_names_appear_only_in_the_refusal_list() -> None:
    problems: list[str] = []
    for path in scanned_files(SRC, DEPLOY):
        if path.suffix != ".py":
            continue
        source = read(path)
        if source is not None:
            problems += legacy_name_problems(path, source)

    assert problems == [], "INV-02: the legacy names may only be named in order to be refused:\n" + "\n".join(problems)


def test_the_refusal_list_really_covers_the_names_this_guard_greps_for() -> None:
    assert LEGACY_ENV_NAMES[0] in FORBIDDEN_ALPACA_ENV
    assert LEGACY_ENV_NAMES[1] in FORBIDDEN_ALPACA_ENV
    assert any(name.startswith(LEGACY_ENV_NAMES[2]) for name in FORBIDDEN_ALPACA_ENV)


# ======================================================================================================================
# (b) The construction site
# ======================================================================================================================


@pytest.mark.parametrize(
    "planted",
    [
        "def helper(k, s):\n    return TradingClient(api_key=k, secret_key=s, paper=True)\n",
        "CLIENT = TradingClient(api_key='k', secret_key='s', paper=True)\n",
        f"def {CONSTRUCTOR_FUNCTION}(k, s):\n    return TradingClient(api_key=k, secret_key=s)\n",
        f"def {CONSTRUCTOR_FUNCTION}(k, s, flag):\n    return TradingClient(api_key=k, secret_key=s, paper=flag)\n",
        f"def {CONSTRUCTOR_FUNCTION}(k, s):\n    return TradingClient(api_key=k, secret_key=s, paper=bool(1))\n",
        f"def {CONSTRUCTOR_FUNCTION}(k, s):\n    return TradingStream(api_key=k, secret_key=s, paper=True)\n"
        "def other(k, s):\n    return TradingStream(api_key=k, secret_key=s, paper=True)\n",
    ],
)
def test_the_ast_check_fires_on_planted_source(planted: str, tmp_path: Path) -> None:
    path = tmp_path / "planted.py"
    path.write_text(textwrap.dedent(planted), encoding="utf-8")

    assert paper_flag_problems(path, path.read_text(encoding="utf-8")), planted


def test_the_ast_check_accepts_the_real_shape(tmp_path: Path) -> None:
    path = tmp_path / "ok.py"
    path.write_text(
        f"def {CONSTRUCTOR_FUNCTION}(k, s):\n    return TradingClient(api_key=k, secret_key=s, paper=True)\n",
        encoding="utf-8",
    )

    assert paper_flag_problems(path, path.read_text(encoding="utf-8")) == []


def test_every_trading_client_in_src_is_built_inside_make_clients_with_the_literal_true() -> None:
    problems: list[str] = []
    for path in scanned_files(SRC, DEPLOY):
        if path.suffix != ".py":
            continue
        source = read(path)
        if source is not None:
            problems += paper_flag_problems(path, source)

    assert problems == [], "INV-01: the trading client is constructed in exactly one function:\n" + "\n".join(problems)


def test_the_construction_site_is_where_section_11_1_says_it_is() -> None:
    source = CONSTRUCTION_SITE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    sites = [function for _, function in construction_calls(tree)]

    assert sites == [CONSTRUCTOR_FUNCTION]  # exactly one construction, in exactly that function


# ======================================================================================================================
# (c) The runtime guards
# ======================================================================================================================


def test_a_client_that_came_back_with_the_live_url_is_refused() -> None:
    with pytest.raises(PaperGuardError, match="INV-01"):
        assert_paper_base_url(FakeRestClient(base_url="https://" + LIVE_HOST))


def test_a_live_looking_key_is_refused() -> None:
    with pytest.raises(PaperGuardError, match="INV-01"):
        paper_credentials(Secrets(alpaca_paper_key="AK000000000000", alpaca_paper_secret="SK"))


def test_an_options_level_below_three_is_refused() -> None:
    with pytest.raises(PaperGuardError, match="options trading level"):
        assert_tradable_account(FakeTradingClient(options_trading_level=2))


def test_a_blocked_account_is_refused() -> None:
    with pytest.raises(PaperGuardError, match="cannot trade"):
        assert_tradable_account(FakeTradingClient(trading_blocked=True))


@pytest.mark.parametrize("name", [LEGACY_ENV_NAMES[0], LEGACY_ENV_NAMES[1], LEGACY_ENV_NAMES[2] + "API_BASE_URL"])
def test_a_forbidden_environment_variable_is_refused_by_both_gates(name: str) -> None:
    with pytest.raises(PaperGuardError, match="ambiguous Alpaca credentials"):
        assert_no_forbidden_env({name: "whatever"})
    with pytest.raises(PaperGuardError, match="ambiguous Alpaca credentials"):
        secrets({name: "whatever"}, None)


def test_a_clean_environment_and_a_paper_account_pass_every_runtime_guard() -> None:
    client = FakeTradingClient()

    assert_no_forbidden_env({"ALPACA_PAPER_KEY": "PK1", "ALPACA_PAPER_SECRET": "SK1"})
    assert_paper_base_url(client)
    assert paper_credentials(Secrets(alpaca_paper_key="PK1", alpaca_paper_secret="SK1")) == ("PK1", "SK1")
    assert assert_tradable_account(client).options_level >= 3
