"""Shared pytest configuration of jevbot (DESIGN.md section 15, INV-18, D26, D29).

Every test runs offline and hermetically:

* `block_network` (autouse): `socket.socket.connect` / `connect_ex` raise `NetworkBlockedError` for every address family
  except AF_UNIX (local IPC), and `socket.getaddrinfo` refuses to resolve anything but `localhost` and numeric hosts, so
  not even a DNS packet leaves the machine. A test that swallows the error still fails at teardown: an attempt was
  recorded. `@pytest.mark.expects_blocked_network` marks the few tests that exercise the guard itself.
* `offline_env` (autouse): `JEVBOT_DATA` is a fresh mode-700 temporary directory, `TYPESAFE_API_KEY=dummy`,
  `ALPACA_PAPER_KEY=PKTESTDUMMY` (plus a dummy paper secret), and every variable `config.secrets()` refuses or that could
  leak into a test from the developer's shell (`APCA_*`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `TYPESAFE_BASE_URL`,
  `TYPESAFE_DEFAULT_MODEL`, the SDK log-level variable, the alert webhook) is removed for the duration of the test.
* `isolated_logging` (autouse): the root logger, the record factory, the pinned third-party levels and the secret
  redaction registry are restored after every test (`logsetup.reset()` + `clear_secrets()`), so a test that configured
  logging or registered the dummy keys cannot change what a later test sees.
* `tiny_config` / `tiny_config_path`: the default strategy configuration scaled down for tests (a 60-session 2014 window,
  the offline synthetic provider, zero retry back-off, small evaluation reps) with `paths.data_dir` inside `$JEVBOT_DATA`
  and `paths.env_file` pointing at an absent file inside the temporary directory - the operator's real `.env` is never read.
* `--regen-goldens` / `goldens`: golden files are compared byte for byte and rewritten only by that explicit flag (15.2).
* `xnys_calendar`, `factory_chain`, `fake_view`, `memory_ledger`: the shared doubles of WP00, ready to use.

The self-tests of the fixture modules (`tests/fixtures/{chain_factory,fake_view,memory_ledger}.py`) and of this file are
collected by `pytest_collect_file` below: no test file of another package is touched, and "the doubles satisfy the
Protocols" is part of the ordinary `pytest` gate.
"""

import ipaddress
import json
import logging
import os
import socket
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pytest

if TYPE_CHECKING:
    from jevbot.cal import XnysCalendar
    from jevbot.config import Config
    from jevbot.types import ChainSnapshot
    from tests.fixtures.fake_view import FakeView
    from tests.fixtures.memory_ledger import MemoryLedger

TESTS_DIR: Final = Path(__file__).resolve().parent
REPO_ROOT: Final = TESTS_DIR.parent
GOLDEN_DIR: Final = TESTS_DIR / "fixtures" / "golden"

DUMMY_TYPESAFE_API_KEY: Final = "dummy"  # the mock transport asserts `Authorization: Bearer dummy` (15.3)
DUMMY_ALPACA_PAPER_KEY: Final = "PKTESTDUMMY"  # a paper key id starts with "PK" (INV-02)
DUMMY_ALPACA_PAPER_SECRET: Final = "SKTESTDUMMYSECRET"

# environment names are constants of config.py (section 4); repeated here as literals so that a broken config module
# cannot take the whole conftest down. The SDK log-level token is assembled from fragments, as the guard test does.
ENV_JEVBOT_DATA: Final = "JEVBOT_DATA"
ENV_TYPESAFE_API_KEY: Final = "TYPESAFE_API_KEY"
ENV_ALPACA_PAPER_KEY: Final = "ALPACA_PAPER_KEY"
ENV_ALPACA_PAPER_SECRET: Final = "ALPACA_PAPER_SECRET"
ENV_SDK_LOG_LEVEL: Final = "_".join(("TYPESAFE", "LOG", "LEVEL"))
SCRUBBED_ENV_EXACT: Final[frozenset[str]] = frozenset(
    {"ALPACA_API_KEY", "ALPACA_SECRET_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL", ENV_SDK_LOG_LEVEL, "JEVBOT_ALERT_WEBHOOK"}
)
SCRUBBED_ENV_PREFIXES: Final[tuple[str, ...]] = ("APCA_",)

# files whose `test_*` functions are collected although their names do not match `python_files` (module docstring)
SELF_TEST_FILES: Final[frozenset[str]] = frozenset(
    {"conftest.py", "fixtures/chain_factory.py", "fixtures/fake_view.py", "fixtures/memory_ledger.py"}
)

_LOCAL_HOSTS: Final[frozenset[str]] = frozenset({"", "localhost", "localhost.localdomain"})


# ======================================================================================================================
# pytest hooks
# ======================================================================================================================


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--regen-goldens",
        action="store_true",
        default=False,
        help="rewrite golden fixture files from the current outputs instead of comparing against them (DESIGN 15.2)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "expects_blocked_network: the test deliberately triggers the network guard and must not fail at teardown"
    )


def pytest_collect_file(file_path: Path, parent: pytest.Collector) -> pytest.Module | None:
    """Collect the self-tests embedded in the shared fixture modules and in this file (see the module docstring)."""
    try:
        relative = file_path.resolve().relative_to(TESTS_DIR).as_posix()
    except ValueError:
        return None
    if relative not in SELF_TEST_FILES or parent.session.isinitpath(file_path):
        return None  # an explicit `pytest tests/fixtures/x.py` argument is collected by pytest itself
    return pytest.Module.from_parent(parent, path=file_path)


# ======================================================================================================================
# Network guard
# ======================================================================================================================


class NetworkBlockedError(RuntimeError):
    """Raised instead of any real network operation. A RuntimeError on purpose: HTTP clients map OSError subclasses to
    'transient' errors that fail-closed code would swallow; this one propagates."""


def _is_numeric_host(host: object) -> bool:
    text = host.decode("ascii", "replace") if isinstance(host, bytes) else host
    if not isinstance(text, str):
        return False
    try:
        ipaddress.ip_address(text.split("%", 1)[0])
    except ValueError:
        return False
    return True


def _is_local_name(host: object) -> bool:
    text = host.decode("ascii", "replace") if isinstance(host, bytes) else host
    return isinstance(text, str) and text.lower() in _LOCAL_HOSTS


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Iterator[list[str]]:
    """Block every non-AF_UNIX socket connection and every remote name resolution; fail the test if one was attempted."""
    attempts: list[str] = []
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo
    af_unix = getattr(socket, "AF_UNIX", None)

    def refuse(what: str) -> NetworkBlockedError:
        attempts.append(what)
        return NetworkBlockedError(f"network access is blocked in tests (DESIGN 15): {what}")

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if af_unix is not None and self.family == af_unix:
            real_connect(self, address)
            return
        self.close()  # helpers such as socket.create_connection close only on OSError: no dangling descriptor, no ResourceWarning
        raise refuse(f"connect({address!r})")

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        if af_unix is not None and self.family == af_unix:
            result: int = real_connect_ex(self, address)
            return result
        self.close()
        raise refuse(f"connect_ex({address!r})")

    def guarded_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        if host is None or _is_local_name(host) or _is_numeric_host(host):
            return real_getaddrinfo(host, port, *args, **kwargs)
        raise refuse(f"getaddrinfo({host!r})")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    yield attempts
    if attempts and request.node.get_closest_marker("expects_blocked_network") is None:
        pytest.fail("the test attempted network access (blocked, DESIGN 15): " + "; ".join(attempts[:5]), pytrace=False)


# ======================================================================================================================
# Environment: tmp $JEVBOT_DATA, dummy keys, scrubbed variables
# ======================================================================================================================


def scrub_environment(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Remove every variable `config.secrets()` refuses (INV-02, INV-18) or that must never reach a test; returns the names."""
    removed = sorted(name for name in os.environ if name in SCRUBBED_ENV_EXACT or name.startswith(SCRUBBED_ENV_PREFIXES))
    for name in removed:
        monkeypatch.delenv(name, raising=False)
    return removed


@pytest.fixture(autouse=True)
def offline_env(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """`JEVBOT_DATA` = a fresh mode-700 temporary directory, dummy keys set, forbidden variables scrubbed. Returns the
    variables it set (pass it as the `env` mapping of `config.secrets()` when `os.environ` is not wanted).

    The directory is a sibling of the test's `tmp_path` (not inside it), so a test that enumerates its own `tmp_path`
    sees only what it created."""
    data_dir = tmp_path_factory.mktemp("jevbot-data", numbered=True)
    data_dir.chmod(0o700)  # the umask must not widen it: config.ensure_data_dir insists on 700 (D1)
    scrub_environment(monkeypatch)
    values = {
        ENV_JEVBOT_DATA: str(data_dir),
        ENV_TYPESAFE_API_KEY: DUMMY_TYPESAFE_API_KEY,
        ENV_ALPACA_PAPER_KEY: DUMMY_ALPACA_PAPER_KEY,
        ENV_ALPACA_PAPER_SECRET: DUMMY_ALPACA_PAPER_SECRET,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return values


@pytest.fixture
def data_dir(offline_env: dict[str, str]) -> Path:
    """The temporary `$JEVBOT_DATA` of this test (mode 700, outside the repository)."""
    return Path(offline_env[ENV_JEVBOT_DATA])


# ======================================================================================================================
# Logging isolation (INV-18: registered secrets and handlers never leak between tests)
# ======================================================================================================================


def _is_pytest_handler(handler: logging.Handler) -> bool:
    return type(handler).__module__.startswith("_pytest")


def snapshot_logging() -> dict[str, Any]:
    root = logging.getLogger()
    try:
        from jevbot import logsetup
    except Exception:  # a broken logsetup module must not take every test down; its own tests will fail loudly
        pinned: dict[str, int] = {}
    else:
        pinned = {name: logging.getLogger(name).level for name in logsetup.PINNED_LOGGERS}
    return {
        "level": root.level,
        "handlers": list(root.handlers),
        "filters": list(root.filters),
        "factory": logging.getLogRecordFactory(),
        "pinned": pinned,
    }


def restore_logging(saved: Mapping[str, Any]) -> None:
    root = logging.getLogger()
    try:
        from jevbot import logsetup
    except Exception:
        pass
    else:
        logsetup.reset()
        logsetup.clear_secrets()
    for handler in list(root.handlers):
        if handler not in saved["handlers"] and not _is_pytest_handler(handler):
            root.removeHandler(handler)
            handler.close()
    for handler in saved["handlers"]:
        if handler not in root.handlers and not _is_pytest_handler(handler):
            root.addHandler(handler)
    root.filters[:] = list(saved["filters"])
    root.setLevel(saved["level"])
    logging.setLogRecordFactory(saved["factory"])
    for name, level in saved["pinned"].items():
        logging.getLogger(name).setLevel(level)


@pytest.fixture(autouse=True)
def isolated_logging() -> Iterator[None]:
    """Every test gives back the logging state it found: handlers, filters, level, record factory, pinned levels; the
    secret redaction registry is emptied (the dummy keys a test registered must not redact a later test's output)."""
    saved = snapshot_logging()
    yield
    restore_logging(saved)


# ======================================================================================================================
# Tiny config
# ======================================================================================================================

TINY_CONFIG_TEMPLATE: Final = """\
# jevbot tiny test configuration (tests/conftest.py): the default strategy parameters, scaled down for offline tests.
# Everything not listed here keeps its config/default.toml value.

[run]
experiment = "tinytest"
seed = 7
start = 2014-04-09                  # the 60-session mini window of DESIGN 15.2 (a Saturday-dated monthly, a Good-Friday
end = 2014-07-03                    # week, two FOMC meetings and an early close)

[paths]
data_dir = {data_dir}
env_file = {env_file}               # absent on purpose: the operator's real .env is never read by a test

[data]
provider = "synthetic"              # the offline provider of 5.10: no input files
min_history_sessions = 20           # percentile / rank features become available inside the short window

[jev]
retry_backoff_s = [0.0, 0.0]        # our retry loop never sleeps in tests

[eval]
bootstrap_reps = 200
random_baseline_seeds = 5
null_sim_reps = 20
base_rate_min_events = 20
recal_min_events = 20
"""


def write_tiny_config(path: Path, data_dir: Path) -> Path:
    """Write the tiny configuration TOML to `path` with `paths.*` inside `data_dir`'s parent; returns `path`."""
    env_file = data_dir.parent / f"{data_dir.name}.env"  # a per-test path beside the data dir, never created
    text = TINY_CONFIG_TEMPLATE.format(data_dir=json.dumps(str(data_dir)), env_file=json.dumps(str(env_file)))
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def tiny_config_path(tmp_path: Path, data_dir: Path) -> Path:
    """Path of the tiny TOML (pass it as `--config` / `load_config(path)`)."""
    return write_tiny_config(tmp_path / "tiny.toml", data_dir)


@pytest.fixture
def tiny_config(tiny_config_path: Path) -> "Config":
    """The tiny `Config`, loaded and validated through `config.load_config`."""
    from jevbot.config import load_config

    return load_config(tiny_config_path)


# ======================================================================================================================
# Goldens (15.2): compared byte for byte, rewritten only by `pytest --regen-goldens`
# ======================================================================================================================


class Goldens:
    """`check_text(name, actual)` compares `actual` with `<root>/<name>`; with `--regen-goldens` it writes the file
    instead (and the test passes); a missing golden without the flag fails with the command to run."""

    def __init__(self, root: Path, regen: bool) -> None:
        self.root = root
        self.regen = regen
        self.written: list[Path] = []

    def path(self, name: str) -> Path:
        return self.root / name

    def check_text(self, name: str, actual: str) -> None:
        path = self.path(name)
        if self.regen:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(actual, encoding="utf-8", newline="")
            tmp.replace(path)
            self.written.append(path)
            return
        if not path.exists():
            pytest.fail(f"golden {path.relative_to(self.root)} is missing: run `pytest --regen-goldens` and review the diff", pytrace=False)
        expected = path.read_bytes().decode("utf-8")  # bytes, not text mode: newlines are part of the golden
        assert actual == expected, (
            f"golden {path.relative_to(self.root)} differs (regenerate with `pytest --regen-goldens` and review the diff)"
        )


@pytest.fixture(scope="session")
def regen_goldens(pytestconfig: pytest.Config) -> bool:
    return bool(pytestconfig.getoption("--regen-goldens"))


@pytest.fixture
def goldens(regen_goldens: bool) -> Goldens:
    return Goldens(GOLDEN_DIR, regen_goldens)


# ======================================================================================================================
# The shared doubles
# ======================================================================================================================


@pytest.fixture(scope="session")
def xnys_calendar() -> "XnysCalendar":
    from jevbot.cal import XnysCalendar

    return XnysCalendar()


@pytest.fixture
def factory_chain() -> "ChainSnapshot":
    """The default factory chain (SPY, $450, Friday 2024-05-17, weekly Fridays + monthlies)."""
    from tests.fixtures.chain_factory import make_chain

    return make_chain()


@pytest.fixture
def fake_view() -> "FakeView":
    """A complete FakeView world for SPY on the default session (300 sessions of history, indices, rate, one FOMC)."""
    from tests.fixtures.fake_view import make_view

    return make_view()


@pytest.fixture
def memory_ledger() -> "MemoryLedger":
    """A fresh per-append MemoryLedger."""
    from tests.fixtures.memory_ledger import MemoryLedger

    return MemoryLedger()


# ======================================================================================================================
# Self-tests of this file (collected through pytest_collect_file)
# ======================================================================================================================


@pytest.mark.expects_blocked_network
def test_tcp_connections_are_blocked(block_network: list[str]) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s, pytest.raises(NetworkBlockedError):
        s.connect(("127.0.0.1", 9))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s, pytest.raises(NetworkBlockedError):
        s.connect_ex(("127.0.0.1", 9))
    with pytest.raises(NetworkBlockedError):
        socket.create_connection(("127.0.0.1", 9), timeout=0.1)
    assert len(block_network) == 3 and all("127.0.0.1" in a for a in block_network)
    assert issubclass(NetworkBlockedError, RuntimeError) and not issubclass(NetworkBlockedError, OSError)


@pytest.mark.expects_blocked_network
def test_remote_name_resolution_is_blocked_but_local_names_resolve(block_network: list[str]) -> None:
    with pytest.raises(NetworkBlockedError):
        socket.getaddrinfo("api.typesafe.ai", 443)
    with pytest.raises(NetworkBlockedError):
        socket.getaddrinfo(b"paper-api.alpaca.markets", 443)
    assert socket.getaddrinfo("127.0.0.1", 80)[0][4][0] == "127.0.0.1"
    assert socket.getaddrinfo("::1", 80)[0][0] == socket.AF_INET6
    assert socket.getaddrinfo(None, 80)
    assert socket.getaddrinfo("localhost", 80)
    assert block_network == ["getaddrinfo('api.typesafe.ai')", "getaddrinfo(b'paper-api.alpaca.markets')"]


def test_unix_domain_sockets_still_work(tmp_path: Path) -> None:
    af_unix = getattr(socket, "AF_UNIX", None)
    if af_unix is None:
        pytest.skip("no AF_UNIX on this platform")
    path = str(tmp_path / "ipc.sock")
    with socket.socket(af_unix, socket.SOCK_STREAM) as server:
        server.bind(path)
        server.listen(1)
        with socket.socket(af_unix, socket.SOCK_STREAM) as client:
            client.connect(path)
            conn, _ = server.accept()
            with conn:
                client.sendall(b"ping")
                assert conn.recv(4) == b"ping"
        with socket.socket(af_unix, socket.SOCK_STREAM) as client:
            assert client.connect_ex(path) == 0


def test_offline_environment_is_hermetic(offline_env: dict[str, str], data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import stat

    from jevbot import config
    from jevbot.errors import PaperGuardError

    assert os.environ[ENV_JEVBOT_DATA] == str(data_dir) and data_dir.is_dir()
    assert stat.S_IMODE(data_dir.stat().st_mode) == 0o700 and REPO_ROOT not in data_dir.resolve().parents
    assert os.environ[ENV_TYPESAFE_API_KEY] == "dummy" and os.environ[ENV_ALPACA_PAPER_KEY] == "PKTESTDUMMY"
    for name in SCRUBBED_ENV_EXACT:
        assert name not in os.environ
    assert not any(name.startswith(SCRUBBED_ENV_PREFIXES) for name in os.environ)
    secrets = config.secrets(os.environ, None)
    assert (
        secrets.has_typesafe_key and secrets.has_alpaca_key and secrets.alpaca_key_paper_hint is True and secrets.data_dir == str(data_dir)
    )
    assert config.secrets(offline_env, None) == secrets and config.ensure_data_dir(data_dir) == data_dir.resolve()
    assert config.data_dir(config.Config(), secrets) == data_dir
    # the scrubber removes what config.secrets() would refuse
    monkeypatch.setenv("APCA_API_KEY_ID", "x")
    monkeypatch.setenv("ALPACA_API_KEY", "x")
    monkeypatch.setenv(ENV_SDK_LOG_LEVEL, "debug")
    with pytest.raises(PaperGuardError):
        config.secrets(os.environ, None)
    assert scrub_environment(monkeypatch) == ["ALPACA_API_KEY", "APCA_API_KEY_ID", ENV_SDK_LOG_LEVEL]
    assert config.secrets(os.environ, None) == secrets


def test_tiny_config_loads_and_points_inside_the_temporary_directory(
    tiny_config: "Config", tiny_config_path: Path, data_dir: Path, xnys_calendar: "XnysCalendar"
) -> None:
    from datetime import date

    from jevbot import config
    from jevbot.errors import ConfigError
    from jevbot.types import RunMode

    cfg = tiny_config
    assert cfg.run.mode is RunMode.BACKTEST and cfg.run.experiment == "tinytest" and cfg.run.seed == 7
    assert (cfg.run.start, cfg.run.end) == (date(2014, 4, 9), date(2014, 7, 3))
    assert len(xnys_calendar.sessions(cfg.run.start, cfg.run.end)) == 60 and xnys_calendar.is_early_close(date(2014, 7, 3))
    assert cfg.data.provider == "synthetic" and cfg.data.min_history_sessions == 20 and cfg.jev.retry_backoff_s == (0.0, 0.0)
    assert (cfg.eval.bootstrap_reps, cfg.eval.random_baseline_seeds, cfg.eval.null_sim_reps) == (200, 5, 20)
    assert cfg.risk == config.RiskConfig() and cfg.rules == config.RulesConfig() and cfg.candidates == config.CandidatesConfig()
    assert (
        Path(cfg.paths.data_dir) == data_dir
        and Path(cfg.paths.env_file).parent == data_dir.parent
        and not Path(cfg.paths.env_file).exists()
    )
    secrets = config.load_secrets(cfg, os.environ)
    assert secrets.has_typesafe_key and config.ensure_data_dir(config.data_dir(cfg, secrets)) == data_dir.resolve()
    assert tiny_config_path.read_text(encoding="utf-8").startswith("# jevbot tiny test configuration")
    assert config.load_config(tiny_config_path) == cfg  # deterministic
    with pytest.raises(ConfigError):
        config.load_config(tiny_config_path, overrides=["run.mode=paper", "risk.max_new_per_day=1"])  # protected in paper mode


def test_goldens_helper_compares_and_regenerates_only_on_request(tmp_path: Path) -> None:
    root = tmp_path / "golden"
    strict = Goldens(root, regen=False)
    with pytest.raises(pytest.fail.Exception, match="regen-goldens"):
        strict.check_text("a/b.json", "x")
    writer = Goldens(root, regen=True)
    writer.check_text("a/b.json", '{"k": 1}\n')
    assert writer.written == [root / "a" / "b.json"] and (root / "a" / "b.json").read_bytes() == b'{"k": 1}\n'
    strict.check_text("a/b.json", '{"k": 1}\n')
    with pytest.raises(AssertionError, match="differs"):
        strict.check_text("a/b.json", '{"k": 2}\n')
    with pytest.raises(AssertionError):
        strict.check_text("a/b.json", '{"k": 1}\r\n')  # byte for byte: newlines matter
    assert GOLDEN_DIR == TESTS_DIR / "fixtures" / "golden"


def test_logging_isolation_restores_the_root_logger_and_clears_secrets() -> None:
    from jevbot import logsetup

    root = logging.getLogger()
    assert logsetup.registered_secret_count() == 0 and not [h for h in root.handlers if getattr(h, "_jevbot_handler", False)]
    saved = snapshot_logging()
    level, factory = root.level, logging.getLogRecordFactory()
    logsetup.configure("debug", console=False)
    logsetup.register_secret("hunter2")
    mine = logging.NullHandler()
    root.addHandler(mine)
    assert logsetup.registered_secret_count() == 1 and root.level == logging.DEBUG and logging.getLogRecordFactory() is not factory
    restore_logging(saved)
    assert logsetup.registered_secret_count() == 0 and root.level == level and logging.getLogRecordFactory() is factory
    assert mine not in root.handlers and not [h for h in root.handlers if getattr(h, "_jevbot_handler", False)]
    assert logsetup.redact("hunter2") == "hunter2"
    for name in logsetup.PINNED_LOGGERS:
        assert logging.getLogger(name).level == saved["pinned"][name]


def test_shared_double_fixtures(
    factory_chain: "ChainSnapshot", fake_view: "FakeView", memory_ledger: "MemoryLedger", xnys_calendar: "XnysCalendar"
) -> None:
    from datetime import date

    assert factory_chain.underlying == "SPY" and factory_chain.key.session == date(2024, 5, 17)
    assert fake_view.spot("SPY") == 45_000 and len(fake_view.closes("SPY", 260)) == 260
    assert memory_ledger.head() == (0, "0" * 64) and memory_ledger.commit_mode == "per_append"
    assert xnys_calendar.prev_or_same_session(date(2014, 4, 19)) == date(2014, 4, 17)


def test_self_tests_are_collected_through_the_hook(request: pytest.FixtureRequest) -> None:
    assert request.node.nodeid.startswith("tests/conftest.py::")  # this very test was collected by pytest_collect_file
    for name in sorted(SELF_TEST_FILES):
        assert (TESTS_DIR / name).is_file()
