"""Guard: the import rules of DESIGN.md section 1 (15.5), checked on the AST of every module under `src/jevbot` at test time.

1. only `jev/live.py` and `jev/probe.py` import `typesafe_sdk` (and, 1.1, `httpx2`);
2. only `paper/*` imports `alpaca` - `data/fetch.py` receives its Alpaca-backed archive sources by injection;
3. `rules.py`, `risk.py`, `state.py`, `features.py`, `buckets.py`, `fills.py`, `portfolio.py`, `candidates.py`, `structmath.py`
   import no network library and do no IO;
4. `risk.py` never imports `jevbot.jev`;
5. nothing outside `data/`, `paper/live_data.py` and `paper/recorder.py` calls `read_parquet` / `read_csv`;
6. cross-package CLI conveniences import their dependency lazily INSIDE the command function (sections 1 and 16), and the
   typer root imports no sub-app module at module level;
7. `protocols.py` imports nothing from the package but `types`, `config` and `errors`, carries the contract blocks of sections
   3.1-3.6 verbatim, resolves every annotation (no dangling forward reference) and types `CycleContext` only with Protocols.

The scanners take the package directory as an argument: every rule is first proven to FIRE on a small planted tree (so a rule can
never pass vacuously) and then applied to the real tree, whatever modules the later work packages have added by then.
Imports are resolved from `import x`, `from x import y` (relative forms included; `from pkg import sub` counts as `pkg.sub`),
`importlib.import_module("x")` and `__import__("x")`; an import under `if TYPE_CHECKING:` counts like any other.
"""

import ast
import collections.abc
import dataclasses
import inspect
import re
import textwrap
import typing
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "src" / "jevbot"
DESIGN_MD = REPO / "docs" / "design" / "DESIGN.md"

NETWORK_LIBS = (
    "socket",
    "ssl",
    "http",
    "urllib",
    "urllib3",
    "requests",
    "httpx",
    "httpx2",
    "httpcore",
    "httpcore2",
    "aiohttp",
    "websockets",
    "websocket",
    "asyncio",
    "ftplib",
    "smtplib",
    "xmlrpc",
    "alpaca",
    "typesafe_sdk",
)
IO_LIBS = ("os", "shutil", "subprocess", "sqlite3", "tempfile", "glob", "pathlib", "pyarrow", "pickle", "shelve", "csv", "fcntl", "mmap")
IO_BUILTINS = ("open", "print", "input")
IO_METHODS = (
    "read_text",
    "write_text",
    "read_bytes",
    "write_bytes",
    "read_parquet",
    "read_csv",
    "read_pickle",
    "read_feather",
    "read_json",
    "to_parquet",
    "to_csv",
    "to_pickle",
    "to_feather",
    "mkdir",
    "unlink",
    "rmdir",
    "touch",
)
PURE_MODULES = (
    "rules.py",
    "risk.py",
    "state.py",
    "features.py",
    "buckets.py",
    "fills.py",
    "portfolio.py",
    "candidates.py",
    "structmath.py",
)
TABLE_READERS = ("read_parquet", "read_csv")

# file (relative to the package) -> modules it may import only INSIDE a function (the lazy cross-wave edges of sections 1 / 16)
LAZY_ONLY = {
    "cli/data_cmds.py": ("jevbot.paper", "jevbot.candidates"),
    "cli/jev_cmds.py": ("jevbot.cycle",),
    "cli/baselines_cmds.py": ("jevbot.backtest",),
    "cli/leakage_cmds.py": ("jevbot.backtest",),
    "baselines.py": ("jevbot.backtest",),
    "eval/leakage.py": ("jevbot.backtest",),
}
CLI_SUBAPP_MODULES = (
    "data_cmds",
    "jev_cmds",
    "backtest_cmds",
    "eval_cmds",
    "baselines_cmds",
    "leakage_cmds",
    "paper_cmds",
    "record_cmds",
    "doctor",
)


# ======================================================================================================================
# The scanner
# ======================================================================================================================


@dataclass(frozen=True)
class Ref:
    name: str  # absolute dotted module name (or module.attribute for `from m import a`)
    line: int
    lazy: bool  # inside a function body: not executed when the module is imported


def under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def module_of(path: Path, package: Path) -> tuple[str, str]:
    """(module name, the package relative imports resolve against) for a file under the `jevbot` package directory."""
    parts = [package.name, *path.relative_to(package).with_suffix("").parts]
    if parts[-1] == "__init__":
        parts.pop()
        return ".".join(parts), ".".join(parts)
    return ".".join(parts), ".".join(parts[:-1])


def import_refs(source: str, package_name: str) -> list[Ref]:
    refs: list[Ref] = []

    def visit(node: ast.AST, lazy: bool) -> None:
        if isinstance(node, ast.Import):
            refs.extend(Ref(alias.name, node.lineno, lazy) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package_name.split(".")
                anchor = anchor[: len(anchor) - (node.level - 1)]
                base = ".".join([*anchor, *([node.module] if node.module else [])])
            refs.append(Ref(base, node.lineno, lazy))
            refs.extend(Ref(f"{base}.{alias.name}", node.lineno, lazy) for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Call):
            func = node.func
            called = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if called in ("import_module", "__import__") and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    refs.append(Ref(first.value, node.lineno, lazy))
        inside = lazy or isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda)
        for child in ast.iter_child_nodes(node):
            visit(child, inside)

    visit(ast.parse(source), False)
    return refs


def call_names(source: str) -> list[tuple[str, int, bool]]:
    """(called name, line, is_attribute_call) for every call: `open(...)` -> ("open", n, False); `pd.read_csv(...)` -> ("read_csv", n, True)."""
    out: list[tuple[str, int, bool]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                out.append((node.func.id, node.lineno, False))
            elif isinstance(node.func, ast.Attribute):
                out.append((node.func.attr, node.lineno, True))
    return out


def python_files(package: Path) -> list[Path]:
    return sorted(p for p in package.rglob("*.py") if "__pycache__" not in p.parts)


def rel(path: Path, package: Path) -> str:
    return path.relative_to(package).as_posix()


# ----------------------------------------------------------------------------------------------------------------------
# The rules: each returns human-readable violations for one package directory
# ----------------------------------------------------------------------------------------------------------------------


def only_allowed_files_import(package: Path, library: str, allowed: typing.Callable[[str], bool]) -> list[str]:
    problems = []
    for path in python_files(package):
        _, pkg = module_of(path, package)
        if allowed(rel(path, package)):
            continue
        problems += [
            f"{rel(path, package)}:{ref.line} imports {ref.name}"
            for ref in import_refs(path.read_text(encoding="utf-8"), pkg)
            if under(ref.name, library)
        ]
    return problems


def pure_module_violations(package: Path) -> list[str]:
    problems = []
    for name in PURE_MODULES:
        path = package / name
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        _, pkg = module_of(path, package)
        for ref in import_refs(source, pkg):
            for library in (*NETWORK_LIBS, *IO_LIBS):
                if under(ref.name, library):
                    problems.append(f"{name}:{ref.line} imports {ref.name} (pure module: no network library, no IO)")
        for called, line, is_attribute in call_names(source):
            if (not is_attribute and called in IO_BUILTINS) or (is_attribute and called in IO_METHODS):
                problems.append(f"{name}:{line} calls {called}() (pure module: no IO)")
    return problems


def risk_imports_jev(package: Path) -> list[str]:
    path = package / "risk.py"
    if not path.exists():
        return []
    _, pkg = module_of(path, package)
    return [
        f"risk.py:{ref.line} imports {ref.name}"
        for ref in import_refs(path.read_text(encoding="utf-8"), pkg)
        if under(ref.name, "jevbot.jev")
    ]


def table_reader_violations(package: Path) -> list[str]:
    def allowed(name: str) -> bool:
        return name.startswith("data/") or name in ("paper/live_data.py", "paper/recorder.py")

    problems = []
    for path in python_files(package):
        if allowed(rel(path, package)):
            continue
        source = path.read_text(encoding="utf-8")
        problems += [f"{rel(path, package)}:{line} calls {called}()" for called, line, _ in call_names(source) if called in TABLE_READERS]
        _, pkg = module_of(path, package)
        problems += [
            f"{rel(path, package)}:{ref.line} imports {ref.name}"
            for ref in import_refs(source, pkg)
            if ref.name.rpartition(".")[2] in TABLE_READERS
        ]
    return problems


def eager_cross_wave_imports(package: Path) -> list[str]:
    problems = []
    for name, lazy_only in LAZY_ONLY.items():
        path = package / name
        if not path.exists():
            continue
        _, pkg = module_of(path, package)
        for ref in import_refs(path.read_text(encoding="utf-8"), pkg):
            if not ref.lazy and any(under(ref.name, module) for module in lazy_only):
                problems.append(f"{name}:{ref.line} imports {ref.name} at module level (must be lazy, inside the command function)")
    root = package / "cli" / "main.py"
    if root.exists():
        for ref in import_refs(root.read_text(encoding="utf-8"), "jevbot.cli"):
            if not ref.lazy and any(under(ref.name, f"jevbot.cli.{stem}") for stem in CLI_SUBAPP_MODULES):
                problems.append(f"cli/main.py:{ref.line} imports {ref.name} at module level (sub-apps are registered lazily)")
    return problems


def protocols_package_imports(package: Path) -> list[str]:
    path = package / "protocols.py"
    allowed = ("jevbot.types", "jevbot.config", "jevbot.errors")
    return [
        f"protocols.py:{ref.line} imports {ref.name}"
        for ref in import_refs(path.read_text(encoding="utf-8"), "jevbot")
        if under(ref.name, "jevbot") and not any(under(ref.name, module) for module in allowed)
    ]


def sdk_allowed(name: str) -> bool:
    return name in ("jev/live.py", "jev/probe.py")


def alpaca_allowed(name: str) -> bool:
    return name.startswith("paper/")


# ======================================================================================================================
# The scanner itself is tested first
# ======================================================================================================================


def test_import_refs_sees_every_import_form() -> None:
    source = textwrap.dedent(
        """
        import alpaca.trading.client as atc, os
        from typesafe_sdk import TypeSafeClient
        from . import live
        from .. import jev
        from ..data.view import DataView
        from jevbot import jev as decider_pkg
        import importlib
        mod = importlib.import_module("httpx2")
        other = __import__("requests.adapters")
        if TYPE_CHECKING:
            from alpaca.data import OptionHistoricalDataClient

        def command():
            from jevbot.cycle import preview_request
            import scipy.stats
            return importlib.import_module("jevbot.backtest")
        """
    )
    refs = import_refs(source, "jevbot.paper")
    eager = {r.name for r in refs if not r.lazy}
    lazy = {r.name for r in refs if r.lazy}
    assert {"alpaca.trading.client", "os", "typesafe_sdk", "typesafe_sdk.TypeSafeClient", "httpx2", "requests.adapters"} <= eager
    assert {"jevbot.paper", "jevbot.paper.live", "jevbot", "jevbot.jev", "jevbot.data.view", "jevbot.data.view.DataView"} <= eager
    assert "alpaca.data" in eager  # TYPE_CHECKING imports count
    assert lazy == {"jevbot.cycle", "jevbot.cycle.preview_request", "scipy.stats", "jevbot.backtest"}
    assert under("alpaca.trading", "alpaca") and under("alpaca", "alpaca") and not under("alpaca_shim", "alpaca")


def test_module_names_and_relative_anchors() -> None:
    package = Path("/x/src/jevbot")
    assert module_of(package / "risk.py", package) == ("jevbot.risk", "jevbot")
    assert module_of(package / "paper" / "broker.py", package) == ("jevbot.paper.broker", "jevbot.paper")
    assert module_of(package / "paper" / "__init__.py", package) == ("jevbot.paper", "jevbot.paper")
    assert module_of(package / "__init__.py", package) == ("jevbot", "jevbot")


def plant(root: Path, files: dict[str, str]) -> Path:
    package = root / "jevbot"
    for name, source in files.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return package


def test_every_rule_fires_on_a_planted_violation(tmp_path: Path) -> None:
    package = plant(
        tmp_path,
        {
            "__init__.py": "",
            "jev/__init__.py": "",
            "jev/live.py": "import typesafe_sdk\nimport httpx2\n",  # allowed
            "jev/replay.py": "from typesafe_sdk import TypeSafeClient\n",  # NOT allowed
            "jev/mock.py": "def f():\n    import httpx2\n",  # NOT allowed, even lazily
            "paper/__init__.py": "",
            "paper/broker.py": "from alpaca.trading.client import TradingClient\n",  # allowed
            "paper/recorder.py": "import pandas as pd\ndf = pd.read_parquet('x')\n",  # allowed
            "data/__init__.py": "",
            "data/fetch.py": "def fetch():\n    from alpaca.data.historical.news import NewsClient\n",  # NOT allowed
            "data/store.py": "import pandas as pd\nx = pd.read_csv('x')\n",  # allowed
            "eval/__init__.py": "",
            "eval/load.py": "import pandas as pd\nframe = pd.read_parquet('runs/x.parquet')\n",  # NOT allowed
            "eval/report.py": "from pandas import read_csv\n",  # NOT allowed
            "risk.py": "from jevbot.jev.stats import to_answers\nfrom . import jev\nimport requests\n",
            "rules.py": "import logging\nimport math\n\ndef f(p):\n    return open(p).read()\n",
            "fills.py": "from pathlib import Path\n\ndef g(df):\n    df.to_csv('x')\n    print(df)\n",
            "structmath.py": "from fractions import Fraction\nimport math\n",  # clean
            "cli/__init__.py": "",
            "cli/main.py": "from jevbot.cli import paper_cmds\n",
            "cli/data_cmds.py": "from jevbot.paper.live_data import AlpacaNews\n\ndef scan():\n    from jevbot.candidates import scan\n",
            "cli/jev_cmds.py": "def show():\n    from jevbot.cycle import preview_request\n",  # lazy: fine
            "baselines.py": "from jevbot.backtest import run_backtest\n",
        },
    )
    assert only_allowed_files_import(package, "typesafe_sdk", sdk_allowed) == [
        "jev/replay.py:1 imports typesafe_sdk",
        "jev/replay.py:1 imports typesafe_sdk.TypeSafeClient",
    ]
    assert only_allowed_files_import(package, "httpx2", sdk_allowed) == ["jev/mock.py:2 imports httpx2"]
    assert only_allowed_files_import(package, "alpaca", alpaca_allowed) == [
        "data/fetch.py:2 imports alpaca.data.historical.news",
        "data/fetch.py:2 imports alpaca.data.historical.news.NewsClient",
    ]
    assert risk_imports_jev(package) == [
        "risk.py:1 imports jevbot.jev.stats",
        "risk.py:1 imports jevbot.jev.stats.to_answers",
        "risk.py:2 imports jevbot.jev",
    ]
    assert pure_module_violations(package) == [
        "rules.py:5 calls open() (pure module: no IO)",  # `import logging` / `import math` are fine
        "risk.py:3 imports requests (pure module: no network library, no IO)",
        "fills.py:1 imports pathlib (pure module: no network library, no IO)",
        "fills.py:1 imports pathlib.Path (pure module: no network library, no IO)",
        "fills.py:4 calls to_csv() (pure module: no IO)",
        "fills.py:5 calls print() (pure module: no IO)",
    ]
    assert table_reader_violations(package) == [
        "eval/load.py:2 calls read_parquet()",
        "eval/report.py:1 imports pandas.read_csv",
    ]
    assert eager_cross_wave_imports(package) == [
        "cli/data_cmds.py:1 imports jevbot.paper.live_data at module level (must be lazy, inside the command function)",
        "cli/data_cmds.py:1 imports jevbot.paper.live_data.AlpacaNews at module level (must be lazy, inside the command function)",
        "baselines.py:1 imports jevbot.backtest at module level (must be lazy, inside the command function)",
        "baselines.py:1 imports jevbot.backtest.run_backtest at module level (must be lazy, inside the command function)",
        "cli/main.py:1 imports jevbot.cli.paper_cmds at module level (sub-apps are registered lazily)",
    ]


def test_a_clean_planted_tree_has_no_violations(tmp_path: Path) -> None:
    package = plant(
        tmp_path,
        {
            "__init__.py": "",
            "jev/__init__.py": "",
            "jev/live.py": "import typesafe_sdk\nimport httpx2\n",
            "jev/probe.py": "import typesafe_sdk\n",
            "paper/__init__.py": "",
            "paper/live_data.py": "import alpaca\nimport pandas as pd\n\ndef f():\n    return pd.read_csv('x')\n",
            "data/__init__.py": "",
            "data/fetch.py": "import urllib.request\nimport pandas as pd\n\ndef f():\n    return pd.read_parquet('x')\n",
            "risk.py": "import logging\nimport math\nfrom jevbot import structmath\nfrom jevbot.types import Position\n",
            "structmath.py": "from fractions import Fraction\n",
            "cli/__init__.py": "",
            "cli/main.py": "import importlib\n\ndef load(name):\n    return importlib.import_module(name)\n",
            "cli/data_cmds.py": "def news():\n    from jevbot.paper.live_data import AlpacaNews\n",
        },
    )
    assert only_allowed_files_import(package, "typesafe_sdk", sdk_allowed) == []
    assert only_allowed_files_import(package, "httpx2", sdk_allowed) == []
    assert only_allowed_files_import(package, "alpaca", alpaca_allowed) == []
    assert risk_imports_jev(package) == []
    assert pure_module_violations(package) == []
    assert table_reader_violations(package) == []
    assert eager_cross_wave_imports(package) == []


# ======================================================================================================================
# The real tree
# ======================================================================================================================


def test_the_scan_really_covers_the_package() -> None:
    names = {rel(path, PACKAGE) for path in python_files(PACKAGE)}
    assert {"__init__.py", "types.py", "protocols.py", "structmath.py", "config.py", "cli/main.py"} <= names
    for path in python_files(PACKAGE):
        ast.parse(path.read_text(encoding="utf-8"))  # every module parses


def test_only_jev_live_and_probe_import_the_typesafe_sdk() -> None:
    assert only_allowed_files_import(PACKAGE, "typesafe_sdk", sdk_allowed) == []


def test_only_jev_live_and_probe_import_httpx2() -> None:
    assert only_allowed_files_import(PACKAGE, "httpx2", sdk_allowed) == []


def test_only_the_paper_package_imports_alpaca() -> None:
    assert only_allowed_files_import(PACKAGE, "alpaca", alpaca_allowed) == []


def test_data_fetch_never_imports_alpaca() -> None:
    # its Alpaca-backed archive sources arrive by injection (3.2 NewsArchiveSource / CorporateActionsSource / DailyBarsSource)
    assert not alpaca_allowed("data/fetch.py")
    fetch = PACKAGE / "data" / "fetch.py"
    if fetch.exists():
        refs = import_refs(fetch.read_text(encoding="utf-8"), "jevbot.data")
        assert [r.name for r in refs if under(r.name, "alpaca") or under(r.name, "jevbot.paper")] == []


def test_pure_modules_import_no_network_library_and_do_no_io() -> None:
    assert (PACKAGE / "structmath.py").exists()
    assert pure_module_violations(PACKAGE) == []


def test_risk_never_imports_the_jev_package() -> None:
    assert risk_imports_jev(PACKAGE) == []


def test_tables_are_read_only_by_the_data_layer() -> None:
    assert table_reader_violations(PACKAGE) == []


def test_cross_wave_cli_conveniences_are_lazy_and_the_root_imports_no_subapp() -> None:
    assert eager_cross_wave_imports(PACKAGE) == []


def test_no_module_imports_the_tests() -> None:
    for path in python_files(PACKAGE):
        _, pkg = module_of(path, PACKAGE)
        refs = import_refs(path.read_text(encoding="utf-8"), pkg)
        assert [r.name for r in refs if under(r.name, "tests") or under(r.name, "pytest")] == [], rel(path, PACKAGE)


# ======================================================================================================================
# protocols.py: the contract blocks of sections 3.1-3.6, imports, forward references, CycleContext
# ======================================================================================================================


def test_protocols_imports_only_the_wp00_contract_modules() -> None:
    assert protocols_package_imports(PACKAGE) == []


def design_contract_nodes() -> list[ast.stmt]:
    """Top-level classes and assignments of the fenced python blocks of sections 3.1-3.6. Signatures that the spec prints there
    for OTHER modules (`decide_batch`, the concrete `DataView` / `PitTable`) have no body and are left out."""
    text = DESIGN_MD.read_text(encoding="utf-8")
    section = text[text.index("## 3. Interfaces") : text.index("### 3.7 ")]
    nodes: list[ast.stmt] = []
    for block in re.findall(r"```python\n(.*?)```", section, flags=re.S):
        if "(Protocol)" not in block and "CycleContext" not in block:
            continue
        kept: list[str] = []
        skipping = False
        for line in block.splitlines():
            if line.startswith("def "):  # a bodiless module-level signature (cycle.decide_batch) and its comment lines
                skipping = True
            elif skipping and line.strip() and not line.startswith((" ", "#")):
                skipping = False
            if not skipping:
                kept.append(line)
        nodes.extend(node for node in ast.parse("\n".join(kept)).body if isinstance(node, ast.ClassDef | ast.Assign))
    return nodes


def members(node: ast.ClassDef) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            out.append(("attribute", item.target.id, ast.dump(item.annotation)))
        elif isinstance(item, ast.FunctionDef):
            decorators = ",".join(ast.dump(d) for d in item.decorator_list)  # a read-only `@property` member is not a method
            out.append(("method", item.name, decorators, ast.dump(item.args), ast.dump(item.returns) if item.returns else ""))
        elif isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
            continue  # a docstring or the `...` of an empty body
        else:
            raise AssertionError(f"unexpected statement in contract class {node.name}: {ast.dump(item)[:80]}")
    return out


def test_protocols_are_the_design_contract_blocks_verbatim() -> None:
    design = design_contract_nodes()
    code = [
        node
        for node in ast.parse((PACKAGE / "protocols.py").read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) or (isinstance(node, ast.Assign) and not ast.unparse(node.targets[0]).startswith("__"))
    ]
    design_classes = [n for n in design if isinstance(n, ast.ClassDef)]
    code_classes = [n for n in code if isinstance(n, ast.ClassDef)]
    assert [n.name for n in code_classes] == [n.name for n in design_classes]  # same set, same order, nothing extra
    assert len(design_classes) == 23 and "CycleContext" in {n.name for n in design_classes}
    for spec, real in zip(design_classes, code_classes, strict=True):
        assert [ast.dump(b) for b in real.bases] == [ast.dump(b) for b in spec.bases], spec.name
        assert [ast.dump(d) for d in real.decorator_list] == [ast.dump(d) for d in spec.decorator_list], spec.name
        assert members(real) == members(spec), spec.name
    design_aliases = {ast.unparse(n.targets[0]): ast.dump(n.value) for n in design if isinstance(n, ast.Assign)}
    code_aliases = {ast.unparse(n.targets[0]): ast.dump(n.value) for n in code if isinstance(n, ast.Assign)}
    assert set(design_aliases) == {"OrderWorker"}
    assert code_aliases == design_aliases


def test_runtime_checkable_protocols_are_exactly_decider_broker_chain_provider() -> None:
    from jevbot import protocols

    checkable = {
        name
        for name, obj in vars(protocols).items()
        if inspect.isclass(obj)
        and getattr(obj, "_is_protocol", False)
        and getattr(obj, "_is_runtime_protocol", False)
        and obj.__module__ == protocols.__name__
    }
    assert checkable == {"Decider", "Broker", "ChainProvider"}


def test_every_annotation_in_protocols_resolves() -> None:
    """No unresolved forward reference (section 16 acceptance): `typing.get_type_hints` must succeed for every member."""
    from jevbot import protocols

    classes = [obj for obj in vars(protocols).values() if inspect.isclass(obj) and obj.__module__ == protocols.__name__]
    assert len(classes) == 23
    properties = 0
    for cls in classes:
        typing.get_type_hints(cls)
        for _, fn in inspect.getmembers(cls, inspect.isfunction):
            if fn.__qualname__.startswith(cls.__name__ + "."):
                typing.get_type_hints(fn)
        for _, prop in inspect.getmembers(cls, lambda o: isinstance(o, property)):
            assert prop.fget is not None and prop.fset is None and prop.fdel is None  # read-only data members (section 3)
            hints = typing.get_type_hints(prop.fget)
            assert "return" in hints
            properties += 1
    # the read-only data members of section 3: ChainProvider (2), MarketView (5), Decider (2), SpendLedger (1), Broker (1), BookP (1),
    # DecisionRulesP (1); a Protocol declares NO settable attribute (a frozen implementation would be rejected otherwise)
    assert properties == 13
    for cls in classes:
        if getattr(cls, "_is_protocol", False):
            assert not [name for name in getattr(cls, "__annotations__", {}) if not name.startswith("_")], cls.__name__
    # the one string forward reference of the module: OrderWorker's "CycleContext"
    worker = typing.get_type_hints(protocols.CycleContext)["order_worker"]
    assert typing.get_origin(worker) is collections.abc.Callable
    parameters, result = typing.get_args(worker)
    assert parameters[1] is protocols.CycleContext and parameters[2] is protocols.MarketView
    assert result in (None, type(None))
    assert sorted(protocols.__all__) == sorted([c.__name__ for c in classes] + ["OrderWorker"])


def test_cycle_context_is_typed_only_with_protocols() -> None:
    from jevbot import protocols
    from jevbot.config import Config
    from jevbot.types import RunMeta

    assert dataclasses.is_dataclass(protocols.CycleContext)
    hints = typing.get_type_hints(protocols.CycleContext)
    assert [f.name for f in dataclasses.fields(protocols.CycleContext)] == list(hints)
    plain = {"cfg": Config, "meta": RunMeta, "manage_jev": str}
    callables = {"order_worker", "health", "tier_of"}
    for name, hint in hints.items():
        if name in plain:
            assert hint is plain[name]
        elif name in callables:
            assert typing.get_origin(hint) is collections.abc.Callable
        else:
            options = [t for t in typing.get_args(hint) if t is not type(None)] or [hint]
            assert len(options) == 1 and getattr(options[0], "_is_protocol", False), f"CycleContext.{name} is not a Protocol: {hint!r}"
            assert options[0].__module__ == protocols.__name__
    assert set(plain) | callables < set(hints) and len(hints) == 19


@pytest.mark.parametrize("name", ["Calendar", "MarketView", "Ledger", "RiskEngine", "SpendLedger", "BookP"])
def test_protocol_members_named_by_the_work_package_exist(name: str) -> None:
    from jevbot import protocols

    required = {
        "Calendar": {"prev_or_same_session", "offset_from_close", "sessions_between", "next_open_after"},
        "MarketView": {"as_of", "key", "session", "calendar", "fidelity", "close", "closes", "daily", "touched"},
        "Ledger": {"put_state", "get_states", "rollback", "claim_fill", "set_meta"},
        "RiskEngine": {"approve", "budget_floor", "recheck_fill", "size_entry", "hard_exit"},
        "SpendLedger": {"scope", "reserve", "commit", "totals", "blocked"},
        "BookP": {"last_key", "apply", "state", "intent", "has_intent", "filled_qty", "open_orders", "leg_positions"},
    }[name]
    cls = getattr(protocols, name)
    present = set(vars(cls)) | set(getattr(cls, "__annotations__", {}))
    assert required <= present
    if name == "RiskEngine":
        approve = inspect.signature(cls.approve).parameters
        assert approve["now"].kind is inspect.Parameter.KEYWORD_ONLY and approve["now"].default is inspect.Parameter.empty
