"""Canonical JSON (ordered / sorted), sha256 helpers, whole-state safety and the decision-cache key (DESIGN.md 3.7, 5.9).

Two canonical encodings, one formula each (deviation V1):

* `dumps_ordered` keeps dict INSERTION order. It is the Jev-facing encoding: `state_hash`, `question_hash`,
  `question_set_hash` and the cache key are taken over exactly the bytes that reach the wire (the SDK encodes with
  `msgspec.json.encode`, which preserves insertion order and escapes strings exactly like `json.dumps(ensure_ascii=False)`),
  so the D8 key-order and option-order variants are different content with different keys.
* `dumps_sorted` sorts keys. Everything else is hashed with it: ledger entries, config, manifests, ids.

Both refuse anything that has no single canonical spelling: a pre-walk rejects floats (hence NaN / inf), Decimals, datetimes,
bytes, numpy / pandas scalars and non-string dict keys BEFORE `json.dumps` could coerce or stringify them (INV-24: no float
ever reaches hashed material; probabilities are ppm ints, prices are cents).

`ensure_state_safe` is the INV-15 gate for every state sent to Jev.

`ledger_entry_hash` is THE hash-chain formula of 2.7 (INV-19, INV-24). `dumps_sorted` refuses dates and datetimes, so the
formula is only complete together with the spelling of the two time columns; that spelling is frozen here - `render_session`
(the ISO date) and `render_as_of` (the RFC 3339 UTC instant with a `Z` suffix, exactly as msgspec renders a payload
datetime) - and every `Ledger` implementation (`ledger.SqliteLedger`, the `MemoryLedger` double) hashes through this one
function, over the texts it persists, so the two can never disagree on a head hash.

This module imports nothing from the package except `errors.py`.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from enum import Enum
from functools import lru_cache
from typing import Any, Final

from jevbot.errors import StateLeak, StateTooLarge, StateTypeError

__all__ = [
    "GENESIS_HASH",
    "cache_key",
    "dumps_ordered",
    "dumps_sorted",
    "ensure_state_safe",
    "ledger_entry_hash",
    "render_as_of",
    "render_session",
    "sha256_hex",
]

# ======================================================================================================================
# Canonical JSON
# ======================================================================================================================


def _type_name(value: object) -> str:
    t = type(value)
    return t.__qualname__ if t.__module__ == "builtins" else f"{t.__module__}.{t.__qualname__}"


def _is_plain_scalar(value: object) -> bool:
    """None, or EXACTLY str / int / bool, or a str- / int-valued Enum member (json encodes it by value: StrEnum, IntEnum).

    Exact types on purpose: `numpy.float64` subclasses `float`, `numpy.str_` subclasses `str` and `numpy.int64` looks like an
    int; none of them is a builtin scalar and none may reach canonical material.
    """
    if value is None:
        return True
    t = type(value)
    if t is str or t is int or t is bool:
        return True
    return isinstance(value, Enum) and isinstance(value, str | int)


def _canon_walk(value: object, path: str, active: set[int]) -> None:
    """Raise TypeError for any node json.dumps would coerce, stringify or spell in more than one way; ValueError for a cycle."""
    if _is_plain_scalar(value):
        return
    if isinstance(value, dict):
        ident = id(value)
        if ident in active:
            raise ValueError(f"canonical JSON: circular reference at {path}")
        active.add(ident)
        for key, item in value.items():
            if not (type(key) is str or (isinstance(key, Enum) and isinstance(key, str))):
                # json.dumps would silently turn 1 / 1.5 / True / None into the strings "1" / "1.5" / "true" / "null"
                raise TypeError(f"canonical JSON: dict key of type {_type_name(key)} at {path} (keys must be str)")
            _canon_walk(item, f"{path}.{key}", active)
        active.discard(ident)
        return
    if isinstance(value, list | tuple):  # a tuple is spelt as an array: msgspec.to_builtins keeps tuples
        ident = id(value)
        if ident in active:
            raise ValueError(f"canonical JSON: circular reference at {path}")
        active.add(ident)
        for index, item in enumerate(value):
            _canon_walk(item, f"{path}[{index}]", active)
        active.discard(ident)
        return
    raise TypeError(
        f"canonical JSON: unsupported value of type {_type_name(value)} at {path} - only dict (str keys), list, tuple, str, "
        "int, bool and None are canonical; float, Decimal, datetime, bytes and numpy / pandas scalars are rejected"
    )


def dumps_ordered(obj: object) -> str:
    """`json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"))` - insertion order KEPT (Jev-facing)."""
    _canon_walk(obj, "$", set())
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def dumps_sorted(obj: object) -> str:
    """Same as `dumps_ordered` plus `sort_keys=True` (ledger, config, manifests, ids)."""
    _canon_walk(obj, "$", set())
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def sha256_hex(s: str | bytes) -> str:
    """Lower-case hex sha256; a `str` is hashed as its UTF-8 bytes (a lone surrogate raises UnicodeEncodeError)."""
    if isinstance(s, str):
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
    if isinstance(s, bytes):
        return hashlib.sha256(s).hexdigest()
    raise TypeError(f"sha256_hex takes str or bytes, not {_type_name(s)}")


# ======================================================================================================================
# The ledger hash chain (2.7; INV-19, INV-24)
# ======================================================================================================================

GENESIS_HASH: Final = "0" * 64  # `prev_hash` of seq 1; `Ledger.head()` of an empty store is (0, GENESIS_HASH)


def render_session(session: date) -> str:
    """The persisted `session` column: the ISO date (`2024-05-17`). A datetime is refused (its UTC date and the
    exchange-local session date can differ), as is anything that is not a `datetime.date`."""
    if isinstance(session, datetime) or not isinstance(session, date):
        raise TypeError(f"session must be a datetime.date, got {_type_name(session)}")
    return session.isoformat()


def render_as_of(as_of: datetime) -> str:
    """The persisted `as_of` column: the RFC 3339 UTC text of a tz-aware instant - `2024-05-17T20:00:00Z`, with six
    sub-second digits only when the microseconds are non-zero (`2024-05-20T20:05:00.250000Z`); any tz-aware instant is
    normalised to UTC first. This is byte for byte what msgspec renders for a datetime in a payload. A naive datetime is a
    ValueError (all datetimes are tz-aware UTC); anything that is not a datetime is a TypeError."""
    if not isinstance(as_of, datetime):
        raise TypeError(f"as_of must be a datetime, got {_type_name(as_of)}")
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be tz-aware (all datetimes are tz-aware UTC), got a naive datetime")
    utc = as_of.astimezone(UTC)
    text = f"{utc.year:04d}-{utc.month:02d}-{utc.day:02d}T{utc.hour:02d}:{utc.minute:02d}:{utc.second:02d}"
    if utc.microsecond:
        text += f".{utc.microsecond:06d}"
    return text + "Z"


def ledger_entry_hash(
    prev_hash: str, seq: int, kind: str, session: date | str, as_of: datetime | str, payload: Mapping[str, Any] | str
) -> str:
    """THE 2.7 formula: `sha256(prev_hash + "\\n" + dumps_sorted({"seq", "kind", "session", "as_of", "payload"}))`.

    - `kind` is the `LedgerKind` VALUE (a `LedgerKind` member is accepted and hashed by its value);
    - `session` / `as_of` are the persisted texts of `render_session` / `render_as_of`; a `date` / tz-aware `datetime` is
      rendered here (a text is taken as is - it must already be the canonical spelling, which `verify()` re-hashes);
    - `payload` is the mapping (hashed after the canonical round trip: tuples as arrays, str-enums by value; floats /
      datetimes / non-string keys are a TypeError) or the persisted `dumps_sorted` text, which must be canonical byte for
      byte (a non-canonical text cannot have been written by a conforming store: ValueError).

    Every `Ledger` implementation appends AND verifies through this function, over the texts it persists (13.4), so a
    `MemoryLedger`-tested chain and a `run.sqlite` chain built from identical entries share every hash (INV-24).
    """
    if type(prev_hash) is not str or type(seq) is not int:
        raise TypeError("ledger_entry_hash: prev_hash must be a str and seq an int")
    kind_value = kind.value if isinstance(kind, Enum) else kind
    if type(kind_value) is not str or not kind_value:
        raise TypeError(f"ledger_entry_hash: kind must be a LedgerKind value, got {_type_name(kind)}")
    session_text = session if isinstance(session, str) else render_session(session)
    as_of_text = as_of if isinstance(as_of, str) else render_as_of(as_of)
    if isinstance(payload, str):
        try:
            loaded = json.loads(payload)
        except ValueError:
            raise ValueError("ledger_entry_hash: payload text is not JSON") from None
        if not isinstance(loaded, dict) or dumps_sorted(loaded) != payload:
            raise ValueError("ledger_entry_hash: payload text is not the canonical dumps_sorted encoding of a mapping")
        canonical: dict[str, Any] = loaded
    elif isinstance(payload, Mapping):
        canonical = dict(payload)  # dumps_sorted below spells tuples as arrays and str-enums by value, and refuses the rest
    else:
        raise TypeError(f"ledger_entry_hash: payload must be a mapping or its dumps_sorted text, got {_type_name(payload)}")
    material = dumps_sorted({"seq": seq, "kind": kind_value, "session": session_text, "as_of": as_of_text, "payload": canonical})
    return sha256_hex(prev_hash + "\n" + material)


# ======================================================================================================================
# Whole-state safety (5.9, INV-15)
# ======================================================================================================================

_STATE_KEY_RE: Final = re.compile(r"[a-z][a-z0-9_]*")
_STATE_INT_LIMIT: Final = 1000  # no price level can hide in an int
# exact whole-key matches: `time_to_expiry` and `market.as_of` are legitimate keys
_FORBIDDEN_STATE_KEYS: Final[frozenset[str]] = frozenset({"uid", "uuid", "timestamp", "ts", "date", "datetime", "time", "created_at"})
_LEAK_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("iso_date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("year", re.compile(r"\b(19|20)\d{2}\b")),
    ("dollar_amount", re.compile(r"\$\s?\d")),
    ("long_number", re.compile(r"\b\d{3,}(\.\d+)?\b")),
)


@lru_cache(maxsize=64)
def _underlying_pattern(underlyings: tuple[str, ...]) -> re.Pattern[str] | None:
    if not underlyings:
        return None
    return re.compile(r"\b(" + "|".join(re.escape(u) for u in underlyings) + r")\b")


def _leak_code(text: str, ticker_re: re.Pattern[str] | None) -> str | None:
    """Name of the first 5.9 leak pattern that matches `text`, else None."""
    if ticker_re is not None and ticker_re.search(text) is not None:
        return "underlying"
    for code, pattern in _LEAK_PATTERNS:
        if pattern.search(text) is not None:
            return code
    return None


def _state_walk(value: object, path: str, masked: bool, ticker_re: re.Pattern[str] | None, active: set[int]) -> None:
    # Messages carry the path and the rule, never the offending text (it may be third-party content).
    if value is None or isinstance(value, bool):  # bool cannot be subclassed
        return
    if isinstance(value, int) and type(value) is int:
        if abs(value) > _STATE_INT_LIMIT:
            raise StateTypeError(f"state int with abs value > {_STATE_INT_LIMIT} at {path} (no price level may hide in an int)")
        return
    if isinstance(value, str) and type(value) is str:
        if masked:
            code = _leak_code(value, ticker_re)
            if code is not None:
                raise StateLeak(f"state string at {path} matches the `{code}` leak pattern")
        return
    if isinstance(value, dict) and type(value) is dict:
        ident = id(value)
        if ident in active:
            raise StateTypeError(f"circular reference in state at {path}")
        active.add(ident)
        for key, item in value.items():
            if type(key) is not str or _STATE_KEY_RE.fullmatch(key) is None:
                raise StateTypeError(f"state dict key at {path} is not a str matching ^[a-z][a-z0-9_]*$ (type {_type_name(key)})")
            if masked:
                if key in _FORBIDDEN_STATE_KEYS:
                    raise StateLeak(f"state dict key `{key}` at {path} is a forbidden identifier / timestamp key")
                code = _leak_code(key, ticker_re)
                if code is not None:
                    raise StateLeak(f"state dict key at {path} matches the `{code}` leak pattern")
            _state_walk(item, f"{path}.{key}", masked, ticker_re, active)
        active.discard(ident)
        return
    if isinstance(value, list) and type(value) is list:
        ident = id(value)
        if ident in active:
            raise StateTypeError(f"circular reference in state at {path}")
        active.add(ident)
        for index, item in enumerate(value):
            _state_walk(item, f"{path}[{index}]", masked, ticker_re, active)
        active.discard(ident)
        return
    # floats, tuples, numpy / pandas scalars, datetimes, Decimals, enums, dict / list / str subclasses, ...
    # (B1.10: msgspec would silently stringify or null some of them)
    raise StateTypeError(f"state value of type {_type_name(value)} at {path}: only dict, list, str, int, bool and None are allowed")


def ensure_state_safe(obj: object, *, masked: bool = True, underlyings: Sequence[str] = (), max_chars: int | None = None) -> None:
    """The INV-15 gate (5.9): walk `obj` (a whole state, one news item or a bare string) and raise

    * `StateTypeError` for anything that is not EXACTLY dict (keys `^[a-z][a-z0-9_]*$`), list, str, bool, int or None -
      explicitly floats, tuples, numpy / pandas scalars, datetimes, Decimals - and for an int with `abs(v) > 1000`;
    * `StateLeak`, in masked mode only, for any string (value or key) that matches `\\b(<underlyings>)\\b`, an ISO date, a
      19xx / 20xx year, a `$` amount or a number of 3+ digits, and for a dict key whose whole name is one of `uid`, `uuid`,
      `timestamp`, `ts`, `date`, `datetime`, `time`, `created_at`;
    * `StateTooLarge` when `max_chars` is given and `len(dumps_ordered(obj)) > max_chars` (`state.hard_max_chars`).

    `masked=False` (the flagged leakage-diagnostic / probe states) skips the leak rules only; the type rules always hold.
    """
    if isinstance(underlyings, str):
        raise TypeError("underlyings must be a sequence of symbols, not a single string")
    symbols = tuple(underlyings)
    for symbol in symbols:
        if type(symbol) is not str or not symbol:
            raise ValueError(f"underlyings must be non-empty strings, got {symbol!r}")
    ticker_re = _underlying_pattern(symbols) if masked else None
    _state_walk(obj, "$", masked, ticker_re, set())
    if max_chars is not None and len(dumps_ordered(obj)) > max_chars:
        raise StateTooLarge(f"state is larger than {max_chars} characters")


# ======================================================================================================================
# Cache key (D8)
# ======================================================================================================================


def cache_key(model: str, state: dict[str, Any], question_set_hash: str, question: dict[str, Any]) -> str:
    """D8, literally: sha256 over the ORDERED encoding of (version, model, state, question-set hash, question dict).

    Pure content: no question id (ids are not sent to the model), no sample index, no uid, no namespace - the namespace is
    stored beside the key (2.10). Key order of `state` and option order of `question` are part of the content (V1).
    """
    if type(model) is not str or not model:
        raise TypeError("cache_key: model must be a non-empty str")
    if type(question_set_hash) is not str or not question_set_hash:
        raise TypeError("cache_key: question_set_hash must be a non-empty str")
    if not isinstance(state, dict) or not isinstance(question, dict):
        raise TypeError("cache_key: state and question must be dicts")
    return sha256_hex(dumps_ordered({"v": 1, "model": model, "state": state, "question_set_hash": question_set_hash, "question": question}))
