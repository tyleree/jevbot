"""JSON-lines logging, secret redaction and third-party logger pinning (DESIGN.md section 4; INV-18).

Three independent layers keep a secret out of every log sink:

1. a process-wide **record factory** (installed by `configure`) merges `msg % args`, formats the traceback and redacts both
   *when the record is created* - so the `LogRecord` object itself never carries a registered secret, whatever logger made
   it and whatever handler receives it (the exception object is dropped once its redacted text is cached);
2. a `RedactionFilter` on the root logger **and on every root handler** repeats that (idempotent) and also redacts the
   `extra=` attributes, which the logging module attaches only after the factory returned;
3. both formatters redact the final output line once more.

Third-party loggers (`typesafe_sdk`, `httpx2`, `httpcore`, `urllib3`, `requests`, `alpaca`, ...) are pinned to WARNING or
stricter, and our handlers drop any sub-WARNING record from those namespaces even if somebody lowers a level later: an
application log level of DEBUG can never surface request or response bodies (the SDK redacts only headers).

This module imports nothing from the package except `errors.py` and never imports `typesafe_sdk` or `alpaca`.
"""

import json
import logging
import os
import re
import sys
import threading
import traceback
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final, TextIO
from urllib.parse import quote, quote_plus

from jevbot.errors import ConfigError

__all__ = [
    "LEVELS",
    "PINNED_LOGGERS",
    "REDACTED",
    "DailyJsonlHandler",
    "JsonLinesFormatter",
    "RedactingFormatter",
    "RedactionFilter",
    "ThirdPartyFloorFilter",
    "clear_secrets",
    "configure",
    "parse_level",
    "pin_third_party_loggers",
    "redact",
    "redact_obj",
    "register_secret",
    "registered_secret_count",
    "reset",
]

REDACTED: Final = "[REDACTED]"

# INV-18 names typesafe_sdk / httpx2 / urllib3 / requests / alpaca and section 4 adds httpcore. The pinned wheel's HTTP core is
# published as `httpcore2` (its loggers are "httpcore2.*"); `httpx` and `websockets` are pinned too (alpaca-py pulls websockets in).
PINNED_LOGGERS: Final[tuple[str, ...]] = (
    "typesafe_sdk",
    "httpx2",
    "httpx",
    "httpcore2",
    "httpcore",
    "urllib3",
    "requests",
    "alpaca",
    "websockets",
)

# the values of the CLI option `--log-level info|warning|error|debug` (section 14)
LEVELS: Final[dict[str, int]] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

_WITHHELD: Final = "log record withheld: redaction failed"
_TEXT_FORMAT: Final = "%(asctime)sZ %(levelname)s %(name)s: %(message)s"

# attribute names of a plain LogRecord: everything else in record.__dict__ arrived through `extra=`
_STANDARD_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)

# ----------------------------------------------------------------------------------------------------------------------
# secret registry
# ----------------------------------------------------------------------------------------------------------------------

_lock = threading.RLock()
_secrets: set[str] = set()
_pattern: re.Pattern[str] | None = None


def _variants(value: str) -> set[str]:
    """Every spelling under which `value` can reach a log line: raw, JSON-escaped, repr-escaped, URL-quoted."""
    out = {
        value,
        json.dumps(value, ensure_ascii=False)[1:-1],
        json.dumps(value, ensure_ascii=True)[1:-1],
        repr(value)[1:-1],
        repr(value.encode("utf-8", "surrogatepass"))[2:-1],
        quote(value, safe=""),
        quote_plus(value, safe=""),
    }
    return {v for v in out if v}


def _rebuild() -> None:
    global _pattern
    spellings: set[str] = set()
    for secret in _secrets:
        spellings |= _variants(secret)
    # longest first: when one spelling contains another, the longer match wins at the same position
    ordered = sorted(spellings, key=lambda s: (-len(s), s))
    _pattern = re.compile("|".join(re.escape(s) for s in ordered)) if ordered else None


def register_secret(value: str | None) -> None:
    """Register one secret value with the redaction layer. None / blank values are ignored (they are 'absent', D29).

    The stripped value is registered: it is a substring of every padded spelling, so those are covered too.
    """
    if value is None:
        return
    stripped = value.strip()
    if not stripped:
        return
    with _lock:
        if stripped not in _secrets:
            _secrets.add(stripped)
            _rebuild()


def clear_secrets() -> None:
    """Forget every registered secret (tests only)."""
    global _pattern
    with _lock:
        _secrets.clear()
        _pattern = None


def registered_secret_count() -> int:
    """How many distinct secret values are registered (the values themselves are never exposed)."""
    with _lock:
        return len(_secrets)


def redact(text: str) -> str:
    """Replace every registered secret (in any of its spellings) by `[REDACTED]`."""
    pattern = _pattern
    if pattern is None or not text:
        return text
    return pattern.sub(REDACTED, text)


def redact_obj(obj: Any) -> Any:
    """Redact a log `extra` value: strings are redacted, containers are walked, numbers / bools / None pass, anything else
    is replaced by its redacted `str()` (an arbitrary object's text form is the only thing a formatter could print)."""
    if obj is None or isinstance(obj, bool | int | float):
        return obj
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, bytes | bytearray | memoryview):
        return redact(bytes(obj).decode("utf-8", "replace"))
    if isinstance(obj, dict):
        return {redact_obj(k) if isinstance(k, str) else k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(redact_obj(v) for v in obj)
    if isinstance(obj, set | frozenset):
        return [redact_obj(v) for v in sorted(obj, key=repr)]
    return redact(str(obj))


# ----------------------------------------------------------------------------------------------------------------------
# record sanitising (shared by the record factory and the filter)
# ----------------------------------------------------------------------------------------------------------------------


def _withhold(record: logging.LogRecord) -> None:
    """Fail closed: a record that cannot be redacted is emitted without any of its content."""
    record.msg = _WITHHELD
    record.args = None
    record.exc_info = None
    record.exc_text = None
    record.stack_info = None
    for key in [k for k in record.__dict__ if k not in _STANDARD_ATTRS]:
        del record.__dict__[key]


def _sanitise(record: logging.LogRecord) -> None:
    """Redact one record in place. Idempotent. After it the record holds no registered secret in msg / args / exception /
    stack / extras, and no live exception object (its redacted text is cached in `exc_text`, which every Formatter prints)."""
    try:
        try:
            message = record.getMessage()
        except Exception:  # bad %-arguments: keep the template, drop the arguments
            message = f"{record.msg!s} [log arguments dropped: formatting failed]"
        record.msg = redact(message)
        record.args = None
        if "message" in record.__dict__:
            record.message = record.msg

        exc_info = record.exc_info
        if exc_info and not isinstance(exc_info, bool) and exc_info[0] is not None:
            text = record.exc_text or "".join(traceback.format_exception(exc_info[0], exc_info[1], exc_info[2])).rstrip("\n")
            record.exc_text = redact(text)
        elif record.exc_text:
            record.exc_text = redact(record.exc_text)
        record.exc_info = None  # the exception object may carry the secret in its args: only the redacted text survives

        if record.stack_info:
            record.stack_info = redact(str(record.stack_info))

        for key in [k for k in record.__dict__ if k not in _STANDARD_ATTRS]:
            record.__dict__[key] = redact_obj(record.__dict__[key])
    except Exception:
        _withhold(record)


class RedactionFilter(logging.Filter):
    """Redacts every record it sees; never drops one. Attached to the root logger and to every root handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        _sanitise(record)
        return True


class ThirdPartyFloorFilter(logging.Filter):
    """Drops sub-WARNING records of the pinned third-party namespaces (INV-18), even if a level was lowered after pinning."""

    def __init__(self, names: Iterable[str] = PINNED_LOGGERS) -> None:
        super().__init__()
        self._names = tuple(names)

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        name = record.name
        return not any(name == pinned or name.startswith(pinned + ".") for pinned in self._names)


_RecordFactory = Callable[..., logging.LogRecord]
_original_factory: _RecordFactory | None = None


def _redacting_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
    base = _original_factory if _original_factory is not None else logging.LogRecord
    record = base(*args, **kwargs)
    _sanitise(record)
    return record


def _install_factory() -> None:
    global _original_factory
    current = logging.getLogRecordFactory()
    if current is _redacting_factory:
        return
    _original_factory = current
    logging.setLogRecordFactory(_redacting_factory)


def _uninstall_factory() -> None:
    global _original_factory
    if logging.getLogRecordFactory() is _redacting_factory:
        logging.setLogRecordFactory(_original_factory if _original_factory is not None else logging.LogRecord)
    _original_factory = None


# ----------------------------------------------------------------------------------------------------------------------
# formatters and handlers
# ----------------------------------------------------------------------------------------------------------------------


def _iso_utc(created: float) -> str:
    stamp = datetime.fromtimestamp(created, tz=UTC)
    return f"{stamp:%Y-%m-%dT%H:%M:%S}.{stamp.microsecond // 1000:03d}Z"


# json.dumps escapes every control character below U+0020, but with ensure_ascii=False it leaves these three line separators
# raw; escaping them keeps one record on one physical line for every reader (`str.splitlines()` splits on them).
_LINE_SEPARATORS: Final[dict[int, str]] = {0x85: r"\u0085", 0x2028: r"\u2028", 0x2029: r"\u2029"}


def _one_physical_line(encoded: str) -> str:
    return encoded.translate(_LINE_SEPARATORS)


class JsonLinesFormatter(logging.Formatter):
    """One JSON object per line: ts (UTC), level, logger, msg, and - when present - exc, stack, extra."""

    def format(self, record: logging.LogRecord) -> str:
        _sanitise(record)
        payload: dict[str, Any] = {
            "ts": _iso_utc(record.created),
            "level": record.levelname,
            "logger": record.name,
            "msg": str(record.msg),
        }
        if record.exc_text:
            payload["exc"] = record.exc_text
        if record.stack_info:
            payload["stack"] = str(record.stack_info)
        extra = {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS}
        if extra:
            payload["extra"] = extra
        line = _one_physical_line(json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":")))
        final = redact(line)
        if final != line:
            # the fields were already redacted, so this only happens for a secret that shows up through `default=str`;
            # make sure the substitution left one valid JSON object on the line
            try:
                json.loads(final)
            except ValueError:
                fallback = {"ts": payload["ts"], "level": payload["level"], "logger": payload["logger"], "msg": _WITHHELD}
                return json.dumps(fallback, ensure_ascii=False, separators=(",", ":"))
        return final


class RedactingFormatter(logging.Formatter):
    """Human-readable console format (UTC timestamps); the finished line is redacted once more."""

    def __init__(self, fmt: str = _TEXT_FORMAT) -> None:
        super().__init__(fmt=fmt)

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return datetime.fromtimestamp(record.created, tz=UTC).strftime(datefmt or "%Y-%m-%dT%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        _sanitise(record)
        return redact(super().format(record))


class DailyJsonlHandler(logging.Handler):
    """Appends JSON lines to `<log_dir>/jevbot-<UTC date>.jsonl`; a long-lived service rolls over at the UTC date change.

    The directory is created mode 700 and each file mode 600; every record is flushed (a crash loses nothing).
    """

    def __init__(self, log_dir: Path) -> None:
        super().__init__()
        self._dir = Path(log_dir)
        self._day: date | None = None
        self._stream: TextIO | None = None
        self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.setFormatter(JsonLinesFormatter())

    def path_for(self, day: date) -> Path:
        return self._dir / f"jevbot-{day.isoformat()}.jsonl"

    def _stream_for(self, day: date) -> TextIO:
        if self._stream is None or self._day != day:
            self._close_stream()
            fd = os.open(self.path_for(day), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            self._stream = os.fdopen(fd, "a", encoding="utf-8", newline="\n")
            self._day = day
        return self._stream

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            finally:
                self._stream = None
                self._day = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            stream = self._stream_for(datetime.fromtimestamp(record.created, tz=UTC).date())
            stream.write(line + "\n")
            stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        self.acquire()
        try:
            self._close_stream()
        finally:
            self.release()
        super().close()


# ----------------------------------------------------------------------------------------------------------------------
# third-party pinning and configure()
# ----------------------------------------------------------------------------------------------------------------------


def pin_third_party_loggers(names: Iterable[str] = PINNED_LOGGERS) -> None:
    """`setLevel(max(WARNING, current))` on every pinned logger and on every already-created child with an explicit lower level.

    Call it again after a lazy `import typesafe_sdk` / `import alpaca` (the SDK applies its own level once at import;
    `config.check_sdk_log_level` has already refused every value below WARNING). A logger that is stricter (ERROR, off) stays so.
    """
    pinned = tuple(names)
    for name in pinned:
        logger = logging.getLogger(name)
        logger.setLevel(max(logging.WARNING, logger.level))
    for child_name, child in list(logging.root.manager.loggerDict.items()):
        if not isinstance(child, logging.Logger):
            continue  # a PlaceHolder has no level
        if any(child_name.startswith(name + ".") for name in pinned) and logging.NOTSET < child.level < logging.WARNING:
            child.setLevel(logging.WARNING)


def parse_level(level: str | int) -> int:
    """`info | warning | error | debug` (case-insensitive) or a logging level number. Anything else is a ConfigError."""
    if isinstance(level, bool):
        raise ConfigError(f"invalid log level {level!r}: expected one of {sorted(LEVELS)}")
    if isinstance(level, int):
        if level <= logging.NOTSET:
            raise ConfigError(f"invalid log level {level!r}: must be a positive logging level")
        return level
    key = level.strip().lower()
    if key not in LEVELS:
        raise ConfigError(f"invalid log level {level!r}: expected one of {sorted(LEVELS)}")
    return LEVELS[key]


def _is_ours(handler: logging.Handler) -> bool:
    return bool(getattr(handler, "_jevbot_handler", False))


def _add_filter_once(target: logging.Logger | logging.Handler, kind: type[logging.Filter]) -> None:
    if not any(isinstance(f, kind) for f in target.filters):
        target.addFilter(kind())


def configure(level: str | int = "info", *, log_dir: Path | None = None, console: bool = True, stream: TextIO | None = None) -> None:
    """Configure process logging (idempotent; call it once per entry point, before any secret-bearing work).

    - root level = `level` (the application level; DEBUG is fine: third-party loggers stay pinned at WARNING or stricter);
    - `log_dir` given: JSON lines to `<log_dir>/jevbot-<UTC date>.jsonl` (the data dir's `logs/`, 13.1);
    - `console`: a human-readable handler on `stream` (default `sys.stderr`);
    - the redaction filter goes on the root logger and on EVERY root handler (ours and foreign ones), the redacting record
      factory is installed, and the third-party loggers are pinned (INV-18).
    """
    numeric = parse_level(level)
    root = logging.getLogger()

    for handler in [h for h in root.handlers if _is_ours(h)]:
        root.removeHandler(handler)
        handler.close()

    handlers: list[logging.Handler] = []
    if console:
        stream_handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
        stream_handler.setFormatter(RedactingFormatter())
        handlers.append(stream_handler)
    if log_dir is not None:
        handlers.append(DailyJsonlHandler(log_dir))
    for handler in handlers:
        handler._jevbot_handler = True  # type: ignore[attr-defined]
        handler.addFilter(ThirdPartyFloorFilter())
        root.addHandler(handler)

    root.setLevel(numeric)
    _add_filter_once(root, RedactionFilter)
    for handler in root.handlers:
        _add_filter_once(handler, RedactionFilter)
    _install_factory()
    pin_third_party_loggers()


def reset() -> None:
    """Undo `configure()`: remove our handlers, filters and the record factory (registered secrets are kept). Tests only."""
    root = logging.getLogger()
    for handler in [h for h in root.handlers if _is_ours(h)]:
        root.removeHandler(handler)
        handler.close()
    for target in (root, *root.handlers):
        for flt in [f for f in target.filters if isinstance(f, RedactionFilter)]:
            target.removeFilter(flt)
    _uninstall_factory()
