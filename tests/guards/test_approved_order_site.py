"""Guard: INV-03 - `ApprovedOrder` is constructed only in `risk.py` (DESIGN.md 9.1, 15.5).

`Broker.submit` accepts only an `ApprovedOrder`, and `RiskEngine.approve()` is its ONLY constructor: that is what makes the risk
engine the last step before every broker call. The type gate is worthless if any other module can mint the type, so this AST
check walks every module under `src/jevbot` and reports, outside `risk.py`:

- S1  a call of the class:                                  `ApprovedOrder(...)`, `types.ApprovedOrder(...)`, an `import ... as` alias
- S2  the class handed to another callable that can build it: `msgspec.convert(x, type=ApprovedOrder)`, `json.decode(buf, type=...)`,
      `Decoder(ApprovedOrder)`, `functools.partial(ApprovedOrder, ...)`, `map(ApprovedOrder, ...)`
- S3  a constructor reached through an attribute or by name: `ApprovedOrder.__new__(...)`, `getattr(types, "ApprovedOrder")`

Using the NAME stays free everywhere: annotations, `isinstance` / `issubclass` / `cast`, type aliases, imports.
(`msgspec.structs.replace(order, ...)` on an existing instance cannot be recognised without type inference; the frozen struct and
code review cover that.) Every rule is proven on planted source before the real tree is checked.
"""

import ast
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "src" / "jevbot"
TYPE_NAME = "ApprovedOrder"
ALLOWED_FILE = "risk.py"
NAME_ONLY_CALLS = ("isinstance", "issubclass", "cast")  # calls that take the class without being able to build an instance
CONSTRUCTOR_ATTRIBUTES = ("__new__", "__init__", "__call__")


def called_name(call: ast.Call) -> str:
    func = call.func
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""


def construction_sites(source: str) -> list[str]:
    """`<line>: <rule> <detail>` for every place in one module that constructs, or could construct, an ApprovedOrder."""
    tree = ast.parse(source)
    names = {TYPE_NAME}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.asname for alias in node.names if alias.name == TYPE_NAME and alias.asname)
    # simple re-bindings: `AO = ApprovedOrder` / `AO = types.ApprovedOrder`
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and is_reference(node.value, names):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in names:
                        names.add(target.id)
                        changed = True

    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = called_name(node)
        if is_reference(node.func, names):
            problems.append(f"{node.lineno}: S1 constructs {TYPE_NAME}")
            continue
        if isinstance(node.func, ast.Attribute) and is_reference(node.func.value, names) and node.func.attr in CONSTRUCTOR_ATTRIBUTES:
            problems.append(f"{node.lineno}: S3 calls {TYPE_NAME}.{node.func.attr}")
            continue
        if name == "getattr" and any(isinstance(a, ast.Constant) and a.value == TYPE_NAME for a in node.args):
            problems.append(f"{node.lineno}: S3 looks {TYPE_NAME} up by name")
            continue
        if name in NAME_ONLY_CALLS:
            continue
        arguments = [*node.args, *(kw.value for kw in node.keywords)]
        if any(mentions(argument, names) for argument in arguments):
            problems.append(f"{node.lineno}: S2 hands {TYPE_NAME} to {name or 'a call'}(...)")
    return sorted(set(problems), key=lambda p: int(p.split(":")[0]))


def is_reference(node: ast.AST | None, names: set[str]) -> bool:
    return (isinstance(node, ast.Name) and node.id in names) or (isinstance(node, ast.Attribute) and node.attr == TYPE_NAME)


def mentions(node: ast.AST, names: set[str]) -> bool:
    """Does the expression mention the class, other than inside a nested isinstance / issubclass / cast (or another call, which is
    judged on its own)?"""
    if is_reference(node, names):
        return True
    if isinstance(node, ast.Call | ast.Lambda):
        return False
    return any(mentions(child, names) for child in ast.iter_child_nodes(node))


def source_files(package: Path) -> list[Path]:
    return sorted(p for p in package.rglob("*.py") if "__pycache__" not in p.parts)


def violations(package: Path) -> list[str]:
    problems = []
    for path in source_files(package):
        name = path.relative_to(package).as_posix()
        if name == ALLOWED_FILE:
            continue
        problems += [f"{name}:{site}" for site in construction_sites(path.read_text(encoding="utf-8"))]
    return problems


# ======================================================================================================================
# The detector
# ======================================================================================================================


@pytest.mark.parametrize(
    ("source", "rule"),
    [
        (
            "from jevbot.types import ApprovedOrder\norder = ApprovedOrder(intent=i, verdict_id='v', client_order_id='c', attempt=0, qty=1, limit=5, approved_at=t)\n",
            "S1",
        ),
        ("from jevbot import types\n\ndef f():\n    return types.ApprovedOrder(**fields)\n", "S1"),
        ("import jevbot.types as T\norder = T.ApprovedOrder(**fields)\n", "S1"),
        ("from jevbot.types import ApprovedOrder as AO\norder = AO(**fields)\n", "S1"),
        ("from jevbot.types import ApprovedOrder\nMaker = ApprovedOrder\nAgain = Maker\norder = Again(**fields)\n", "S1"),
        ("import msgspec\nfrom jevbot.types import ApprovedOrder\norder = msgspec.convert(raw, type=ApprovedOrder)\n", "S2"),
        ("import msgspec\nfrom jevbot.types import ApprovedOrder\norder = msgspec.json.decode(buf, type=ApprovedOrder | None)\n", "S2"),
        ("import msgspec\nfrom jevbot.types import ApprovedOrder\ndecoder = msgspec.json.Decoder(ApprovedOrder)\n", "S2"),
        ("import msgspec\nfrom jevbot.types import ApprovedOrder\ndecoder = msgspec.json.Decoder(list[ApprovedOrder])\n", "S2"),
        ("from functools import partial\nfrom jevbot.types import ApprovedOrder\nmake = partial(ApprovedOrder, attempt=0)\n", "S2"),
        ("from jevbot.types import ApprovedOrder\norders = list(map(ApprovedOrder, rows))\n", "S2"),
        ("from jevbot.types import ApprovedOrder\norder = ApprovedOrder.__new__(ApprovedOrder)\n", "S3"),
        ("from jevbot import types\ncls = getattr(types, 'ApprovedOrder')\n", "S3"),
    ],
)
def test_detector_fires(source: str, rule: str) -> None:
    sites = construction_sites(source)
    assert sites, source
    assert any(f" {rule} " in site for site in sites), sites


def test_using_the_name_is_free() -> None:
    source = textwrap.dedent(
        '''
        """Docstring that mentions ApprovedOrder( ... ) in prose."""
        from collections.abc import Callable, Sequence
        from typing import cast

        from jevbot.types import ApprovedOrder, OrderState

        __all__ = ["ApprovedOrder"]
        Worker = Callable[[Sequence[ApprovedOrder]], None]
        BY_KIND = {"approved": ApprovedOrder}


        class Broker:
            pending: list[ApprovedOrder]

            def submit(self, order: ApprovedOrder) -> OrderState:
                if not isinstance(order, ApprovedOrder):  # the type gate of INV-03
                    raise TypeError("Broker.submit accepts only an ApprovedOrder")
                assert issubclass(type(order), ApprovedOrder)
                # ApprovedOrder(...) in a comment is not code
                return self._send(cast(ApprovedOrder, order), label="ApprovedOrder(x)")

            def approved(self, orders: Sequence[ApprovedOrder] = ()) -> tuple[ApprovedOrder | None, ...]:
                return tuple(sorted(orders, key=lambda o: o.client_order_id))
        '''
    )
    assert construction_sites(source) == []


def plant(root: Path, files: dict[str, str]) -> Path:
    package = root / "jevbot"
    for name, source in files.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return package


def test_only_risk_py_may_construct_on_a_planted_tree(tmp_path: Path) -> None:
    construct = "from jevbot.types import ApprovedOrder\n\ndef build(**fields):\n    return ApprovedOrder(**fields)\n"
    package = plant(
        tmp_path,
        {
            "__init__.py": "",
            "risk.py": construct,  # THE construction site
            "cycle.py": construct,
            "paper/__init__.py": "",
            "paper/broker.py": "from jevbot.types import ApprovedOrder\n\ndef submit(order: ApprovedOrder) -> None:\n    assert isinstance(order, ApprovedOrder)\n",
            "paper/runner.py": "import msgspec\nfrom jevbot import types\n\ndef again(raw):\n    return msgspec.convert(raw, type=types.ApprovedOrder)\n",
            "killswitch.py": "from jevbot.types import ApprovedOrder as Approved\n\ndef flatten(i):\n    return Approved(intent=i)\n",
        },
    )
    assert violations(package) == [
        "cycle.py:4: S1 constructs ApprovedOrder",
        "killswitch.py:4: S1 constructs ApprovedOrder",
        "paper/runner.py:5: S2 hands ApprovedOrder to convert(...)",
    ]


# ======================================================================================================================
# The real tree
# ======================================================================================================================


def test_the_type_exists_and_the_scan_covers_the_package() -> None:
    from jevbot import types

    assert types.ApprovedOrder.__name__ == TYPE_NAME
    names = {p.relative_to(PACKAGE).as_posix() for p in source_files(PACKAGE)}
    assert {"types.py", "protocols.py", "structmath.py", "cli/main.py"} <= names


def test_approved_order_is_constructed_nowhere_but_in_risk_py() -> None:
    problems = violations(PACKAGE)
    assert problems == [], "INV-03: only risk.py (RiskEngine.approve) may construct ApprovedOrder:\n" + "\n".join(problems)


def test_risk_py_is_the_construction_site_once_it_exists() -> None:
    risk = PACKAGE / ALLOWED_FILE
    if risk.exists():  # WP05; until then the rule above already holds for the whole tree
        sites = construction_sites(risk.read_text(encoding="utf-8"))
        assert any(" S1 " in site for site in sites), "risk.py must construct ApprovedOrder (RiskEngine.approve is its only constructor)"
    else:
        assert violations(PACKAGE) == []


def test_the_contract_names_approve_as_the_only_constructor() -> None:
    # the two ends of the type gate, as frozen in protocols.py: submit() takes ONLY an ApprovedOrder, approve() returns one
    import inspect
    import typing

    from jevbot import protocols, types

    submit = typing.get_type_hints(protocols.Broker.submit)
    assert submit["order"] is types.ApprovedOrder
    approve = typing.get_type_hints(protocols.RiskEngine.approve)
    assert approve["return"] == tuple[types.RiskVerdict, types.ApprovedOrder | None]
    assert list(inspect.signature(protocols.Broker.submit).parameters) == ["self", "order"]
