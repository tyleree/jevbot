"""Guard: INV-13 - all cut-offs are offsets from that day's calendar close; there is no clock literal in `src/` (15.5).

A clock literal is a hard-coded time of day: on a 13:00 ET early close every action has to shift with the close, which only works
when each cut-off is written as `calendar.offset_from_close(session, minutes)` with the minutes coming from the config.

Two layers. (1) The grep that section 11.4 prescribes, literally: no file under `src/` contains `15:30`, `15:45`, `16:00`, `16:15`,
`time(15`, `time(16`, `hour=15` or `hour=16` - anywhere, comments and docstrings included (write "the close" / "close - 5 min"
instead of a wall-clock time). (2) An AST walk of every module under `src/jevbot` that catches the same idea in ANY form and for ANY
time of day (there comments and docstrings may explain a time, code may not contain one), plus the values of every `config/*.toml`:

- C1  a `time(...)` / `datetime.time(...)` constructed from numbers:            `time(15, 55)`, `dt.time(hour=16)`
- C2  `hour=` / `minute=` keyword arguments with a number:                       `ts.replace(hour=15, minute=30)`
- C3  `datetime(...)` / `Timestamp(...)` with a numeric hour or minute position: `datetime(2026, 11, 27, 18, 0, tzinfo=UTC)`
- C4  a string that contains a time of day:                                      `"16:00"`, `"T15:55:00"`, `between_time("09:30", ...)`
- C5  `.hour` / `.minute` compared with a number:                                `if now.hour >= 15`
- C6  a named clock constant:                                                    `CLOSE_HOUR = 16`, `cutoff_minute: int = 55`

Midnight (all zeros) is not a cut-off - it is how a UTC day or a date boundary is written - and is allowed everywhere. UTC offsets
such as `"+00:00"` are not times of day. Durations (`timedelta(minutes=...)`, `*_offset_min`, `*_per_minute`) are not clock literals.

Every rule is first proven to fire on planted source, then applied to the real tree - whatever modules exist by then.
"""

import ast
import datetime as dt
import re
import textwrap
import tomllib
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
PACKAGE = SRC / "jevbot"
CONFIG_DIR = REPO / "config"

# 11.4, literally: "tests/guards/test_no_clock_literals.py greps src/ for ..."
SPEC_TOKENS = ("15:30", "15:45", "16:00", "16:15", "time(15", "time(16", "hour=15", "hour=16")

# HH:MM or HH:MM:SS, not part of a longer number / IP:port / UTC offset (+00:00, -05:00)
TIME_OF_DAY = re.compile(r"(?<![\d:.+\-])(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?![\d:])")
CLOCK_CONTEXT = {"open", "close", "cutoff", "market", "session", "start", "end", "deadline", "at", "bell", "rth"}
CLOCK_UNIT = {"hour", "minute", "hhmm", "hh", "mm"}


def grep_spec_tokens(root: Path) -> list[str]:
    """`<file>:<line>: <token>` for every occurrence of a section-11.4 token in any text file under `root`."""
    hits: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # not a text file
        for number, line in enumerate(text.splitlines(), start=1):
            hits += [f"{path.relative_to(root).as_posix()}:{number}: {token}" for token in SPEC_TOKENS if token in line]
    return hits


def is_number(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool)


def non_zero(node: ast.AST | None) -> bool:
    return is_number(node) and node.value != 0  # type: ignore[union-attr]


def called_name(call: ast.Call) -> str:
    func = call.func
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""


def docstring_nodes(tree: ast.AST) -> set[int]:
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                found.add(id(first.value))
    return found


def times_in(text: str) -> list[str]:
    return [m.group(0) for m in TIME_OF_DAY.finditer(text) if m.group(0).strip("0:") != ""]  # 00:00 / 00:00:00 = midnight: allowed


def is_clock_name(name: str) -> bool:
    tokens = [t for t in re.split(r"[_\W]+", name.lower()) if t]
    return bool(CLOCK_UNIT & set(tokens)) and bool(CLOCK_CONTEXT & set(tokens))


def clock_literals(source: str) -> list[str]:
    """`<line>: <rule> <detail>` for every clock literal in one module's source."""
    tree = ast.parse(source)
    docstrings = docstring_nodes(tree)
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = called_name(node)
            keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            if name == "time" and (
                any(non_zero(a) for a in node.args) or any(non_zero(keywords.get(k)) for k in ("hour", "minute", "second"))
            ):
                problems.append(f"{node.lineno}: C1 time(...) built from numbers")
            elif name != "time" and any(non_zero(keywords.get(k)) for k in ("hour", "minute")):
                problems.append(f"{node.lineno}: C2 hour= / minute= given as a number in {name}(...)")
            elif name in ("datetime", "Timestamp") and any(non_zero(a) for a in node.args[3:5]):
                problems.append(f"{node.lineno}: C3 {name}(...) with a numeric hour / minute")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            for hit in times_in(node.value):
                problems.append(f"{node.lineno}: C4 string contains the time of day {hit!r}")
        elif isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            if any(isinstance(s, ast.Attribute) and s.attr in ("hour", "minute") for s in sides) and any(non_zero(s) for s in sides):
                problems.append(f"{node.lineno}: C5 .hour / .minute compared with a number")
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            values = node.value.elts if isinstance(node.value, ast.Tuple | ast.List) else [node.value]
            for target in targets:
                label = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else ""
                if label and is_clock_name(label) and any(non_zero(v) for v in values):
                    problems.append(f"{node.lineno}: C6 clock constant {label}")
    return sorted(problems, key=lambda p: int(p.split(":")[0]))


def toml_clock_literals(data: Any, path: str = "") -> list[str]:
    problems: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            where = f"{path}.{key}" if path else str(key)
            if is_clock_name(str(key)) and isinstance(value, int | float) and not isinstance(value, bool) and value != 0:
                problems.append(f"{where}: clock constant {value!r}")
            problems += toml_clock_literals(value, where)
    elif isinstance(data, list):
        for i, value in enumerate(data):
            problems += toml_clock_literals(value, f"{path}[{i}]")
    elif isinstance(data, dt.datetime):
        if data.time().replace(tzinfo=None) != dt.time(0, 0):
            problems.append(f"{path}: datetime with a time of day {data.isoformat()}")
    elif isinstance(data, dt.time):
        if data.replace(tzinfo=None) != dt.time(0, 0):
            problems.append(f"{path}: time of day {data.isoformat()}")
    elif isinstance(data, str):
        problems += [f"{path}: string contains the time of day {hit!r}" for hit in times_in(data)]
    return problems


# ======================================================================================================================
# The detector fires on every form ...
# ======================================================================================================================


@pytest.mark.parametrize(
    ("source", "rule"),
    [
        ("from datetime import time\nCUTOFF = time(15, 55)\n", "C1"),
        ("import datetime as dt\nx = dt.time(hour=16)\n", "C1"),
        ("import datetime\nx = datetime.time(9, 30, 0)\n", "C1"),
        ("def f(ts):\n    return ts.replace(hour=15, minute=30, second=0)\n", "C2"),
        ("def f(ts):\n    return ts.replace(minute=55)\n", "C2"),
        ("import pandas as pd\nx = pd.Timestamp(year=2026, month=1, day=2, hour=16)\n", "C2"),
        ("from datetime import datetime, UTC\nx = datetime(2026, 11, 27, 18, 0, tzinfo=UTC)\n", "C3"),
        ("import pandas as pd\nx = pd.Timestamp(2026, 11, 27, 13)\n", "C3"),
        ('CLOSE = "16:00"\n', "C4"),
        ('def f(df):\n    return df.between_time("09:30", "15:55")\n', "C4"),
        ('def f(d):\n    return f"{d}T15:55:00-05:00"\n', "C4"),
        ('x = "close at 1:00 pm on early-close days"\n', "C4"),
        ("def f(now):\n    return now.hour >= 15\n", "C5"),
        ("def f(now):\n    return 30 <= now.minute\n", "C5"),
        ("CLOSE_HOUR = 16\n", "C6"),
        ("class C:\n    cutoff_minute: int = 55\n", "C6"),
        ("MARKET_OPEN_HHMM = (9, 30)\n", "C6"),
        ("class C:\n    def __init__(self):\n        self.session_end_hour = 16\n", "C6"),
    ],
)
def test_detector_fires(source: str, rule: str) -> None:
    problems = clock_literals(source)
    assert problems, source
    assert all(f" {rule} " in p for p in problems), problems


@pytest.mark.parametrize(
    "source",
    [
        # offsets from the close: THE way to express a cut-off
        "def cutoff(calendar, session, cfg):\n    return calendar.offset_from_close(session, cfg.cadence.order_cutoff_offset_min)\n",
        "from datetime import timedelta\nLAG = timedelta(minutes=25)\nSTEP = timedelta(hours=1, seconds=60)\n",
        # midnight / date boundaries are not cut-offs
        "from datetime import time, datetime, UTC\nMIDNIGHT = time(0, 0)\nEPOCH = datetime(1970, 1, 1, 0, 0, tzinfo=UTC)\n",
        "def utc_day(ts):\n    return ts.replace(hour=0, minute=0, second=0, microsecond=0)\n",
        'def parse(s):\n    from datetime import datetime\n    return datetime.fromisoformat(s.replace("Z", "+00:00"))\n',
        'x = "00:00:00"\n',
        # the `time` MODULE and durations
        "import time\nstart = time.time()\nmono = time.monotonic()\ntime.sleep(15)\n",
        "max_orders_per_minute: int = 10\nrest_calls_per_minute = 180\norders_last_minute = 3\nSECONDS_PER_HOUR = 3600\n_ONE_MINUTE_S = 60\n",
        "lookback_hours = 72\nmin_session_minutes = 120\ndecide_offset_min = 25\nheartbeat_s = 15\n",
        # look-alikes that are not times of day
        'URL = "http://127.0.0.1:11434/api"\nFMT = "%H:%M:%S"\nRATIO = "3:1"\nKEY = "a:b:c"\nV = "jev-1.13.0"\nOFFSET = "-05:00"\n',
        'STAMP = "%Y%m%dT%H%M%S"\nNAMESPACE = "exp001:jev-1.13.0:g0"\n',
        # a docstring or a comment may EXPLAIN a time
        'def f():\n    """Early closes end at 13:00 ET; the normal close is 16:00 ET."""\n    return 1  # 15:55 is close - 5 min\n',
        '"""Module docstring: the recorder runs at close-25 min (15:35 ET on a normal day)."""\n',
        "def f(ts, other):\n    return ts.hour == other.hour and ts.minute == 0\n",
    ],
)
def test_detector_stays_quiet(source: str) -> None:
    assert clock_literals(source) == []


def test_docstrings_are_skipped_but_other_strings_in_the_same_function_are_not() -> None:
    source = 'def f():\n    """Closes at 16:00."""\n    label = "16:00"\n    return label\n'
    assert clock_literals(source) == ["3: C4 string contains the time of day '16:00'"]


def test_toml_detector() -> None:
    planted = tomllib.loads(
        textwrap.dedent(
            """
            [cadence]
            decide_offset_min = 25
            close_hour = 16
            cutoff = "15:55"
            local = 15:30:00
            stamp = 2026-11-27T18:00:00Z
            day = 2026-11-27
            midnight = 2026-11-27T00:00:00Z
            [orders]
            max_orders_per_minute = 10
            windows = ["09:30-16:00"]
            """
        )
    )
    assert toml_clock_literals(planted) == [
        "cadence.close_hour: clock constant 16",
        "cadence.cutoff: string contains the time of day '15:55'",
        "cadence.local: time of day 15:30:00",
        "cadence.stamp: datetime with a time of day 2026-11-27T18:00:00+00:00",
        # "-16:00" reads like a UTC offset and is not reported a second time: the string is caught by its first time of day
        "orders.windows[0]: string contains the time of day '09:30'",
    ]


# ======================================================================================================================
# ... and the real tree is clean
# ======================================================================================================================


def test_the_section_11_4_grep_fires_on_every_token_even_in_comments_and_docstrings(tmp_path: Path) -> None:
    lines = [
        '"""Orders stop at 15:45 ET."""',
        "# the close is 16:00 ET, late orders until 16:15",
        "from datetime import time",
        "A = time(15, 30)",
        "B = time(16)",
        'C = "15:30"',
        "def f(ts):",
        "    return ts.replace(hour=15), ts.replace(hour=16)",
    ]
    (tmp_path / "pkg").mkdir()
    good = ["CUTOFF_MIN = 5  # close - 5 min", "x = calendar.offset_from_close(session, CUTOFF_MIN)"]
    (tmp_path / "pkg" / "bad.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "pkg" / "good.py").write_text("\n".join(good) + "\n", encoding="utf-8")
    (tmp_path / "pkg" / "blob.bin").write_bytes(bytes([0xFF, 0xFE]) + b"16:00")  # not a text file: skipped
    hits = grep_spec_tokens(tmp_path)
    assert {hit.rsplit(": ", 1)[1] for hit in hits} == set(SPEC_TOKENS)
    assert all(hit.startswith("pkg/bad.py:") for hit in hits)
    assert "pkg/bad.py:1: 15:45" in hits and "pkg/bad.py:2: 16:00" in hits and "pkg/bad.py:8: hour=16" in hits


def test_section_11_4_grep_of_src_is_clean() -> None:
    assert SPEC_TOKENS == ("15:30", "15:45", "16:00", "16:15", "time(15", "time(16", "hour=15", "hour=16")
    hits = grep_spec_tokens(SRC)
    assert hits == [], "INV-13 (11.4 grep): write 'the close' / 'close - N min' instead of a wall-clock time:\n" + "\n".join(hits)


def source_files() -> list[Path]:
    return sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)


def test_the_scan_covers_the_package() -> None:
    names = {p.relative_to(PACKAGE).as_posix() for p in source_files()}
    assert {"types.py", "config.py", "protocols.py", "structmath.py", "cli/main.py"} <= names


def test_no_clock_literals_in_src() -> None:
    problems = [
        f"src/jevbot/{path.relative_to(PACKAGE).as_posix()}:{p}"
        for path in source_files()
        for p in clock_literals(path.read_text(encoding="utf-8"))
    ]
    assert problems == [], "INV-13: express every cut-off with calendar.offset_from_close(session, minutes):\n" + "\n".join(problems)


def test_no_clock_literals_in_the_shipped_config() -> None:
    files = sorted(CONFIG_DIR.glob("*.toml"))
    assert (CONFIG_DIR / "default.toml") in files
    problems = [f"config/{f.name}: {p}" for f in files for p in toml_clock_literals(tomllib.loads(f.read_text(encoding="utf-8")))]
    assert problems == [], "INV-13: config cut-offs are offsets in minutes from the close:\n" + "\n".join(problems)


def test_cutoffs_in_the_config_are_offsets_from_the_close() -> None:
    cadence = tomllib.loads((CONFIG_DIR / "default.toml").read_text(encoding="utf-8"))["cadence"]
    offsets = {k: v for k, v in cadence.items() if k.endswith("_offset_min")}
    assert {"decide_offset_min", "order_cutoff_offset_min", "eod_offset_min"} <= set(offsets)
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in offsets.values())
