"""Tests for `jevbot.logsetup` (DESIGN.md section 4, INV-18): JSON lines, secret redaction (fuzzed, incl. tracebacks),
third-party logger pinning."""

import io
import json
import logging
import random
import stat
import string
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from jevbot import config, logsetup
from jevbot.errors import ConfigError

SPEC_PINNED = ("typesafe_sdk", "httpx2", "httpcore", "urllib3", "requests", "alpaca")  # INV-18 + section 4


@pytest.fixture(autouse=True)
def clean_logging() -> Iterator[None]:
    """Every test starts from, and gives back, the logging state it found."""
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    saved_filters = list(root.filters)
    saved_factory = logging.getLogRecordFactory()
    saved_secrets = set(logsetup._secrets)
    pinned_levels = {name: logging.getLogger(name).level for name in logsetup.PINNED_LOGGERS}
    logsetup.clear_secrets()
    yield
    logsetup.reset()
    for handler in list(root.handlers):
        if handler not in saved_handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in saved_handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.filters[:] = saved_filters
    root.setLevel(saved_level)
    logging.setLogRecordFactory(saved_factory)
    for name, level in pinned_levels.items():
        logging.getLogger(name).setLevel(level)
    logsetup.clear_secrets()
    for value in saved_secrets:
        logsetup.register_secret(value)


_IDENTITY_ATTRS = ("name", "pathname", "filename", "module", "funcName")


class Capture(logging.Handler):
    """Keeps the LogRecord objects, the way pytest's caplog or any third-party handler would - plus every string the record
    held AT EMIT TIME (a later filter may still change the shared record object, so looking afterwards proves nothing)."""

    def __init__(self, *, with_extras: bool) -> None:
        super().__init__()
        self.with_extras = with_extras
        self.records: list[logging.LogRecord] = []
        self.seen: list[list[str]] = []
        self.had_exception_object: list[bool] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        fields = {k: v for k, v in record.__dict__.items() if k not in _IDENTITY_ATTRS}
        if not self.with_extras:
            fields = {k: v for k, v in fields.items() if k in logsetup._STANDARD_ATTRS}
        self.seen.append([*strings_in(fields), record.getMessage()])
        self.had_exception_object.append(record.exc_info is not None or record.args is not None)


def log_file(log_dir: Path) -> Path:
    files = sorted(log_dir.glob("jevbot-*.jsonl"))
    assert len(files) == 1, files
    return files[0]


def read_lines(log_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in log_file(log_dir).read_text(encoding="utf-8").splitlines()]


def strings_in(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, bytes | bytearray):
        yield bytes(obj).decode("utf-8", "replace")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from strings_in(key)
            yield from strings_in(value)
    elif isinstance(obj, list | tuple | set | frozenset):
        for value in obj:
            yield from strings_in(value)
    elif obj is not None and not isinstance(obj, bool | int | float):
        yield str(obj)


# ======================================================================================================================
# the secret registry and redact()
# ======================================================================================================================


def test_redact_replaces_every_occurrence() -> None:
    assert logsetup.redact("nothing registered: tok-12345") == "nothing registered: tok-12345"
    logsetup.register_secret("tok-12345")
    assert logsetup.redact("Bearer tok-12345, again tok-12345.") == "Bearer [REDACTED], again [REDACTED]."
    assert logsetup.redact("") == ""
    assert logsetup.REDACTED == "[REDACTED]"
    assert logsetup.registered_secret_count() == 1
    logsetup.register_secret("tok-12345")
    assert logsetup.registered_secret_count() == 1
    logsetup.clear_secrets()
    assert logsetup.redact("tok-12345") == "tok-12345" and logsetup.registered_secret_count() == 0


@pytest.mark.parametrize("blank", [None, "", " ", "\t\n"])
def test_blank_values_are_never_registered(blank: str | None) -> None:
    logsetup.register_secret(blank)
    assert logsetup.registered_secret_count() == 0
    assert logsetup.redact("a b\tc") == "a b\tc"


def test_padded_value_is_registered_stripped() -> None:
    logsetup.register_secret("  padded-secret-77 \n")
    assert logsetup.redact("x padded-secret-77 y") == "x [REDACTED] y"
    assert logsetup.redact("x  padded-secret-77 \n y") == "x  [REDACTED] \n y"


def test_longer_secret_wins_over_a_contained_shorter_one() -> None:
    logsetup.register_secret("abc123")
    logsetup.register_secret("abc123xyz789")
    assert logsetup.redact("key=abc123xyz789;") == "key=[REDACTED];"
    assert logsetup.redact("key=abc123;") == "key=[REDACTED];"


def test_escaped_spellings_are_redacted_too() -> None:
    secret = 'pa"ss' + chr(0x5C) + "wo rd/" + chr(0xE9) + "+&"  # a double quote, a backslash, a blank, a non-ASCII letter
    logsetup.register_secret(secret)
    for spelling in (
        secret,
        json.dumps(secret)[1:-1],
        json.dumps(secret, ensure_ascii=False)[1:-1],
        repr(secret)[1:-1],
        quote(secret, safe=""),
        repr(secret.encode())[2:-1],
    ):
        assert logsetup.redact(f"<{spelling}>") == "<[REDACTED]>", spelling


def test_redact_obj_walks_containers() -> None:
    logsetup.register_secret("s3cr3t-value")

    class Opaque:
        def __str__(self) -> str:
            return "opaque s3cr3t-value"

    got = logsetup.redact_obj(
        {"a": "x s3cr3t-value", "s3cr3t-value": [1, 2.5, True, None, ("s3cr3t-value",)], "b": b"raw s3cr3t-value", "o": Opaque()}
    )
    assert got == {
        "a": "x [REDACTED]",
        "[REDACTED]": [1, 2.5, True, None, ("[REDACTED]",)],
        "b": "raw [REDACTED]",
        "o": "opaque [REDACTED]",
    }
    assert logsetup.redact_obj({"s3cr3t-value", "plain"}) == ["plain", "[REDACTED]"]  # a set becomes a list in a stable order
    assert logsetup.redact_obj(12) == 12 and logsetup.redact_obj(None) is None and logsetup.redact_obj(0.5) == 0.5


# ======================================================================================================================
# configure(): JSON lines file, idempotence, levels, rollover
# ======================================================================================================================


def test_configure_writes_json_lines(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    logsetup.configure("info", log_dir=log_dir, console=False)
    log = logging.getLogger("jevbot.test.lines")
    log.info("cycle %s started for %d underlyings", "2026-09-17", 3, extra={"phase": "DECIDE", "n": 3})
    log.debug("below the application level")
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("cycle failed")

    path = log_file(log_dir)
    assert path.name == f"jevbot-{datetime.now(UTC).date().isoformat()}.jsonl"
    assert stat.S_IMODE(log_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    first, second = read_lines(log_dir)
    assert first["level"] == "INFO" and first["logger"] == "jevbot.test.lines"
    assert first["msg"] == "cycle 2026-09-17 started for 3 underlyings"
    assert first["extra"] == {"phase": "DECIDE", "n": 3}
    stamp = datetime.fromisoformat(first["ts"].replace("Z", "+00:00"))
    assert first["ts"].endswith("Z") and stamp.tzinfo is not None and abs((datetime.now(UTC) - stamp).total_seconds()) < 60
    assert set(first) == {"ts", "level", "logger", "msg", "extra"}
    assert second["level"] == "ERROR" and second["msg"] == "cycle failed"
    assert "Traceback (most recent call last)" in second["exc"] and "ValueError: boom" in second["exc"]


def test_configure_is_idempotent(tmp_path: Path) -> None:
    root = logging.getLogger()
    before = len(root.handlers)
    for _ in range(3):
        logsetup.configure("info", log_dir=tmp_path, console=True, stream=io.StringIO())
    assert len(root.handlers) == before + 2, "one console and one file handler, however often configure() runs"
    assert sum(isinstance(f, logsetup.RedactionFilter) for f in root.filters) == 1
    for handler in root.handlers:
        assert sum(isinstance(f, logsetup.RedactionFilter) for f in handler.filters) == 1
    logging.getLogger("jevbot.test.idem").warning("once")
    assert [line["msg"] for line in read_lines(tmp_path)] == ["once"]


def test_console_handler_is_human_readable_and_utc() -> None:
    stream = io.StringIO()
    logsetup.configure("warning", console=True, stream=stream)
    logging.getLogger("jevbot.test.console").warning("heads up %s", "now")
    logging.getLogger("jevbot.test.console").info("not shown")
    (line,) = stream.getvalue().splitlines()
    assert line.endswith("WARNING jevbot.test.console: heads up now")
    datetime.strptime(line.split(" ")[0], "%Y-%m-%dT%H:%M:%SZ")  # noqa: DTZ007 - format check only


def test_app_debug_level_writes_debug_records(tmp_path: Path) -> None:
    logsetup.configure("debug", log_dir=tmp_path, console=False)
    logging.getLogger("jevbot.test.debug").debug("fine detail")
    assert [(line["level"], line["msg"]) for line in read_lines(tmp_path)] == [("DEBUG", "fine detail")]


def test_daily_rollover_by_utc_date(tmp_path: Path) -> None:
    handler = logsetup.DailyJsonlHandler(tmp_path)
    try:
        for stamp, text in (
            (datetime(2026, 9, 17, 23, 59, 59, tzinfo=UTC), "late"),
            (datetime(2026, 9, 18, 0, 0, 1, tzinfo=UTC), "early"),
            (datetime(2026, 9, 18, 14, 0, tzinfo=UTC), "noon"),
        ):
            record = logging.LogRecord("jevbot.test.roll", logging.INFO, __file__, 1, text, None, None)
            record.created = stamp.timestamp()
            handler.handle(record)
    finally:
        handler.close()
    first = [json.loads(x) for x in (tmp_path / "jevbot-2026-09-17.jsonl").read_text(encoding="utf-8").splitlines()]
    second = [json.loads(x) for x in (tmp_path / "jevbot-2026-09-18.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(x["msg"], x["ts"]) for x in first] == [("late", "2026-09-17T23:59:59.000Z")]
    assert [x["msg"] for x in second] == ["early", "noon"]
    assert handler.path_for(datetime(2026, 11, 27, tzinfo=UTC).date()).name == "jevbot-2026-11-27.jsonl"


def test_parse_level() -> None:
    assert logsetup.parse_level("info") == logging.INFO
    assert logsetup.parse_level(" DEBUG ") == logging.DEBUG
    assert logsetup.parse_level("warning") == logging.WARNING
    assert logsetup.parse_level("error") == logging.ERROR
    assert logsetup.parse_level(logging.WARNING) == logging.WARNING
    assert set(logsetup.LEVELS) == {"info", "warning", "error", "debug"}  # the CLI's --log-level values (section 14)
    for bad in ("verbose", "", "warn ing", 0, -5, True):
        with pytest.raises(ConfigError, match="invalid log level"):
            logsetup.parse_level(bad)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="invalid log level"):
        logsetup.configure("chatty")


# ======================================================================================================================
# third-party pinning (INV-18): application DEBUG can never surface request / response bodies
# ======================================================================================================================


def test_pinned_names_cover_the_spec() -> None:
    assert set(SPEC_PINNED) <= set(logsetup.PINNED_LOGGERS)
    assert "httpcore2" in logsetup.PINNED_LOGGERS, "the pinned wheel's HTTP core logs under httpcore2.*"


@pytest.mark.parametrize("name", logsetup.PINNED_LOGGERS)
def test_third_party_loggers_are_pinned_at_app_level_debug(name: str, tmp_path: Path) -> None:
    logging.getLogger(name).setLevel(logging.NOTSET)  # as after import: the logger would inherit the DEBUG root
    logsetup.configure("debug", log_dir=tmp_path, console=False)
    logger = logging.getLogger(name)
    child = logging.getLogger(name + ".transport")
    assert logger.level == logging.WARNING
    assert not logger.isEnabledFor(logging.DEBUG) and not logger.isEnabledFor(logging.INFO)
    assert not child.isEnabledFor(logging.DEBUG), "children inherit the pin"
    child.debug("request body: {'state': ...}")
    logger.info("response body: {'answers': ...}")
    logger.warning("rate limited")
    assert [(line["logger"], line["msg"]) for line in read_lines(tmp_path)] == [(name, "rate limited")]


def test_pinning_keeps_stricter_levels_and_raises_chatty_children() -> None:
    logging.getLogger("typesafe_sdk").setLevel(logging.CRITICAL + 1)  # TYPESAFE_LOG_LEVEL=off
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("requests").setLevel(logging.DEBUG)
    chatty = logging.getLogger("httpx2._client")
    chatty.setLevel(logging.DEBUG)
    strict_child = logging.getLogger("alpaca.trading.stream")
    strict_child.setLevel(logging.ERROR)
    try:
        logsetup.pin_third_party_loggers()
        assert logging.getLogger("typesafe_sdk").level == logging.CRITICAL + 1
        assert logging.getLogger("urllib3").level == logging.ERROR
        assert logging.getLogger("requests").level == logging.WARNING
        assert chatty.level == logging.WARNING
        assert strict_child.level == logging.ERROR
    finally:
        chatty.setLevel(logging.NOTSET)
        strict_child.setLevel(logging.NOTSET)


def test_sub_warning_third_party_records_are_dropped_even_if_a_level_is_lowered_later(tmp_path: Path) -> None:
    logsetup.configure("debug", log_dir=tmp_path, console=False)
    sdk = logging.getLogger("typesafe_sdk")
    sdk.setLevel(logging.DEBUG)  # somebody undoes the pin after configure()
    sdk.debug("wire: %s", {"body": {"state": {"market": "..."}}})
    logging.getLogger("typesafe_sdk._core.transport").debug("wire again")
    logging.getLogger("typesafe_sdk_lookalike").debug("not a pinned namespace")
    sdk.error("request failed")
    assert [(line["logger"], line["msg"]) for line in read_lines(tmp_path)] == [
        ("typesafe_sdk_lookalike", "not a pinned namespace"),
        ("typesafe_sdk", "request failed"),
    ]
    logsetup.pin_third_party_loggers()  # LiveJev calls it again after its lazy import
    assert sdk.level == logging.WARNING


# ======================================================================================================================
# redaction inside records, files, streams - fuzzed incl. tracebacks
# ======================================================================================================================

ALPHABETS = {
    "api_key": string.ascii_letters + string.digits,
    "base64ish": string.ascii_letters + string.digits + "+/=_-",
    "punctuation": string.ascii_letters + string.digits + "!#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\" ",
    "unicode": string.ascii_letters + "".join(chr(c) for c in (0xE9, 0xDF, 0x3A9, 0x436, 0x4E2D, 0x6587, 0x1F511, 0xA0, 0x2028)),
}


def make_secret(rng: random.Random, alphabet: str) -> str:
    body = "".join(rng.choice(alphabet) for _ in range(rng.randint(8, 40)))
    return f"{rng.choice(string.ascii_letters)}{body}{rng.choice(string.ascii_letters)}"  # no leading / trailing blank: the value as an SDK would see it


class Holder:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return f"Holder<{self.value}>"


def emit_everything(log: logging.Logger, secret: str, rng: random.Random) -> int:
    """Log the secret through every channel a record has. Returns the number of records emitted."""
    calls = 0
    log.info("no-args-template " + secret + " tail")  # without args the template is never %-formatted
    log.warning("arg %s and repr %r", secret, secret)
    log.warning("mapping %(token)s", {"token": secret})
    log.error("object %s bytes %s list %s", Holder(secret), secret.encode("utf-8"), [secret, {"k": secret}])
    log.info("Authorization: Bearer %s", secret)
    log.warning("GET https://example.invalid/v1?key=%s", quote(secret, safe=""))
    log.info(
        "extras",
        extra={
            "token": secret,
            "nested": {"list": [secret, (secret, 1)], secret: "as-key"},
            "holder": Holder(secret),
            "raw": secret.encode("utf-8"),
        },
    )
    log.info("bad template %d", secret)  # %-formatting fails: the arguments are dropped, never printed raw
    log.info("stack", stack_info=True)
    calls += 9
    try:
        try:
            raise KeyError(secret)  # str(KeyError) is the repr-escaped spelling
        except KeyError as inner:
            raise RuntimeError(f"wrapper around {secret}") from inner
    except RuntimeError:
        log.exception("chained failure with %s", secret)
        calls += 1
    try:
        raise ValueError(secret, {"detail": secret}, rng.random())
    except ValueError:
        log.error("exc_info=True", exc_info=True)
        calls += 1
    return calls


@pytest.mark.parametrize("family", sorted(ALPHABETS))
def test_secret_never_appears_in_any_log_record_fuzz(family: str, tmp_path: Path) -> None:
    rng = random.Random(f"jevbot-redaction-{family}")
    stream = io.StringIO()
    foreign = Capture(with_extras=True)  # on the root logger BEFORE configure(), like a handler some other library installed
    logging.getLogger().addHandler(foreign)
    logsetup.configure("debug", log_dir=tmp_path, console=True, stream=stream)
    # attached AFTER configure() and NOT on the root logger: no filter of ours runs before it, only the record factory protects
    # it - which covers the message, the arguments, the exception and the stack (`extra=` is attached after the factory)
    late = Capture(with_extras=False)
    log = logging.getLogger(f"jevbot.test.fuzz.{family}")
    log.addHandler(late)
    try:
        secrets = [make_secret(rng, ALPHABETS[family]) for _ in range(40)]
        expected_records = 0
        for secret in secrets:
            config.Secrets(typesafe_api_key=secret)  # the production registration path
            expected_records += emit_everything(log, secret, rng)
    finally:
        log.removeHandler(late)

    lines = log_file(tmp_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == expected_records == len(foreign.records) == len(late.records)
    console = stream.getvalue()
    parsed = [json.loads(line) for line in lines]
    assert any("[REDACTED]" in line for line in lines)
    assert sum("Traceback (most recent call last)" in p.get("exc", "") for p in parsed) == 2 * len(secrets)

    for secret in secrets:
        spellings = {secret, repr(secret)[1:-1], json.dumps(secret)[1:-1], quote(secret, safe="")}
        for spelling in spellings:
            assert spelling not in console
            assert all(spelling not in line for line in lines)
        for payload in parsed:  # the DECODED json: catches a secret hiding behind JSON escapes
            assert all(secret not in text for text in strings_in(payload))
        for capture in (foreign, late):
            assert all(secret not in text for seen in capture.seen for text in seen)
    for capture in (foreign, late):
        assert not any(capture.had_exception_object), "no live exception object or raw argument ever travels with a record"
    for record in foreign.records:  # and the record objects as they are now, extras included
        assert record.exc_info is None and record.args is None
        for text in strings_in({k: v for k, v in record.__dict__.items() if k not in _IDENTITY_ATTRS}):
            assert all(secret not in text for secret in secrets)


def test_fuzz_control_an_unregistered_secret_is_visible_everywhere(tmp_path: Path) -> None:
    """The negative control of the fuzz test: the same log calls WITHOUT registration do leak - so the assertions above can fail."""
    rng = random.Random("jevbot-redaction-control")
    secret = make_secret(rng, ALPHABETS["api_key"])
    stream = io.StringIO()
    foreign = Capture(with_extras=True)
    logging.getLogger().addHandler(foreign)
    logsetup.configure("debug", log_dir=tmp_path, console=True, stream=stream)
    emitted = emit_everything(logging.getLogger("jevbot.test.fuzz.control"), secret, rng)
    lines = log_file(tmp_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == emitted == 11
    leaking = [line for line in lines if secret in line]
    clean = sorted(json.loads(line)["msg"] for line in lines if secret not in line)
    # two calls never carried the secret: the stack record, and the one whose %-arguments could not be formatted (they are dropped)
    assert clean == ["bad template %d [log arguments dropped: formatting failed]", "stack"]
    assert len(leaking) == 9
    assert secret in stream.getvalue()
    assert sum(any(secret in text for text in seen) for seen in foreign.seen) == 9


def test_traceback_text_survives_redaction(tmp_path: Path) -> None:
    logsetup.configure("info", log_dir=tmp_path, console=False)
    logsetup.register_secret("sk-live-0123456789")
    try:
        raise PermissionError("401 for key sk-live-0123456789")
    except PermissionError:
        logging.getLogger("jevbot.test.tb").exception("auth failed")
    (line,) = read_lines(tmp_path)
    assert "PermissionError: 401 for key [REDACTED]" in line["exc"]
    assert "test_traceback_text_survives_redaction" in line["exc"], "the frames are kept: only the secret is removed"


def test_secret_registered_after_a_record_was_created_is_still_caught_by_the_formatter(tmp_path: Path) -> None:
    logsetup.configure("info", log_dir=tmp_path, console=False)
    record = logging.LogRecord("jevbot.test.late", logging.INFO, __file__, 1, "token late-secret-4242", None, None)
    logsetup.register_secret("late-secret-4242")
    logging.getLogger().handle(record)
    assert read_lines(tmp_path)[0]["msg"] == "token [REDACTED]"


def test_unredactable_record_is_withheld_not_leaked(tmp_path: Path) -> None:
    class Hostile:
        def __str__(self) -> str:
            raise RuntimeError("no text form, and the traceback would mention hostile-secret-1")

    logsetup.configure("info", log_dir=tmp_path, console=False)
    logsetup.register_secret("hostile-secret-1")
    logging.getLogger("jevbot.test.hostile").info("payload hostile-secret-1", extra={"obj": Hostile()})
    (line,) = read_lines(tmp_path)
    assert line["msg"] == "log record withheld: redaction failed"
    assert "extra" not in line and "hostile-secret-1" not in json.dumps(line)


def test_structurally_damaging_secret_falls_back_to_a_valid_json_line() -> None:
    logsetup.register_secret('","')  # degenerate: every field separator of the JSON line matches
    record = logging.LogRecord("jevbot.test.json", logging.INFO, __file__, 1, "hello", None, None)
    line = logsetup.JsonLinesFormatter().format(record)
    payload = json.loads(line)  # still ONE valid JSON object
    assert payload == {"ts": payload["ts"], "level": "INFO", "logger": "jevbot.test.json", "msg": "log record withheld: redaction failed"}


def test_json_line_stays_on_one_physical_line() -> None:
    message = "a" + chr(0x0A) + "b" + chr(0x2028) + "c" + chr(0x2029) + "d" + chr(0x85) + "e" + chr(0x0D) + "f"
    record = logging.LogRecord("jevbot.test.json", logging.INFO, __file__, 1, message, None, None)
    line = logsetup.JsonLinesFormatter().format(record)
    assert line.splitlines() == [line]
    assert json.loads(line)["msg"] == message


def test_reset_undoes_configure(tmp_path: Path) -> None:
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    factory_before = logging.getLogRecordFactory()
    logsetup.configure("info", log_dir=tmp_path, console=True, stream=io.StringIO())
    assert logging.getLogRecordFactory() is not factory_before
    logsetup.reset()
    assert list(root.handlers) == handlers_before
    assert logging.getLogRecordFactory() is factory_before
    assert not any(isinstance(f, logsetup.RedactionFilter) for f in root.filters)
    assert not any(isinstance(f, logsetup.RedactionFilter) for h in root.handlers for f in h.filters)
