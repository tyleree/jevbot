"""Identifiers - deterministic, state-independent, collision-free (DESIGN.md section 2.10, INV-07).

Every id is a pure function of ledger- or calendar-derived inputs, so a restart reproduces the same ids even if quotes moved:
no state hash, no wall clock, no run id, no strike and no price is ever part of an order id.

Collision argument (tested in tests/unit/test_ids.py): the tuple (namespace, session, decision, purpose, part, attempt) is
unique for every order the bot can emit. There is exactly one entry decision per (underlying, session) and it can emit one
`open`; there is exactly one manage decision per (position, session) and it can emit one `close`; a close re-issued on a later
session differs in session (hence in decision id); repricing differs in attempt; a kill close has its own decision id (the
`|kill` subject) AND purpose; kill per-leg fallback orders differ in part (1..4); the assigned-stock flatten uses the subject
`"EQ:<symbol>|kill"` and the pseudo position id `sha256("EQ:" + symbol)[:16]`.

Fields are joined with "|" before hashing. Every joined field except the LAST one (`subject`, `structure_id`) is refused if
it contains "|", which keeps the join injective: two different field tuples can never produce the same hashed string.
"""

import re
from datetime import date, datetime
from typing import Final, get_args

import msgspec

from jevbot import canon
from jevbot.types import DecisionKind, OrderIntent, OrderPurpose, OutcomeSpec, SnapshotKey

__all__ = [
    "ENTRY_SUBJECT",
    "ORDER_ID_PREFIX",
    "PROBE_ID_PREFIX",
    "client_order_id",
    "decision_id",
    "equity_kill_subject",
    "equity_position_id",
    "event_key",
    "fill_id",
    "forecast_id",
    "intent_id",
    "is_bot_order_id",
    "kill_subject",
    "namespace",
    "ns8",
    "position_id",
]

ORDER_ID_PREFIX: Final = "jb1-"  # every strategy / kill order; reconcile R1 treats anything else as foreign
PROBE_ID_PREFIX: Final = "jbp-"  # scripted Alpaca probes only (11.11): deliberately NOT a bot order id

ENTRY_SUBJECT: Final = "entry"  # the subject of the one entry decision of an (underlying, session)
_KILL_SUFFIX: Final = "|kill"
_EQUITY_PREFIX: Final = "EQ:"

_DECISION_KINDS: Final[tuple[str, ...]] = get_args(DecisionKind)  # ("entry", "manage")
_DECISION_ID_RE: Final = re.compile(r"[0-9a-f]{32}")
_INTENT_ID_RE: Final = re.compile(r"jb1-[0-9a-f]{8}-[0-9]{6}-[0-9a-f]{12}-(open|close|kill)-[0-9]{2}")
_CLIENT_ORDER_ID_RE: Final = re.compile(_INTENT_ID_RE.pattern + r"-[0-9]{2}")
_MAX_PART: Final = 4  # 0 = whole structure; 1..4 = per-leg kill fallback orders (a structure has at most four legs)
_MAX_ATTEMPT: Final = 99  # two digits keep the id fixed-width (<= 48 chars; Alpaca limit 128)


def _field(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty str, got {value!r}")
    if "|" in value:
        raise ValueError(f"{name} must not contain '|' (the id field separator): {value!r}")
    return str(value)


def _session(name: str, value: date) -> date:
    # a datetime IS a date: its isoformat() would put a time of day into the hashed string
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{name} must be a datetime.date (a session), got {type(value).__name__}")
    return value


def _int_in(name: str, value: int, lo: int, hi: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < lo or (hi is not None and value > hi):
        raise ValueError(f"{name} out of range: {value}")
    return value


def namespace(experiment: str, model: str, refresh_generation: int) -> str:
    """f"{experiment}:{model}:g{refresh_generation}". A forced model change or a `refresh` => new namespace (D8, D12).

    The namespace is NOT part of the cache key (pure content per D8); it is stored beside the key: PRIMARY KEY (namespace, key).
    """
    exp = _field("experiment", experiment)
    mod = _field("model", model)
    if ":" in exp:
        raise ValueError(f"experiment must not contain ':' (the namespace separator): {experiment!r}")
    gen = _int_in("refresh_generation", refresh_generation, 0)
    return f"{exp}:{mod}:g{gen}"


def ns8(namespace: str) -> str:
    """sha256(namespace)[:8]"""
    return canon.sha256_hex(_field("namespace", namespace))[:8]


def decision_id(namespace: str, session: date, underlying: str, kind: DecisionKind, subject: str) -> str:
    """sha256("|".join([namespace, session.isoformat(), underlying, kind, subject]))[:32]

    `kind` is the DECISION-level Literal["entry", "manage"], never a RequestKind: ENTRY and ENTRY_TEXT (and their variants)
    share ONE decision id and ONE DECISION entry that lists every request.
    (kind, subject) = ("entry",  "entry")                  for ENTRY and ENTRY_TEXT requests,
                      ("manage", position_id)              for MANAGE / MANAGE_TEXT requests and for code-only exits,
                      ("manage", position_id + "|kill")    for kill-switch closes (`kill_subject`);
                      ("manage", "EQ:<symbol>|kill")       for the assigned-stock flatten (`equity_kill_subject`).
    No state hash inside: a restart reproduces the same ids even if quotes moved.
    """
    if kind not in _DECISION_KINDS:
        raise ValueError(f"decision kind must be one of {_DECISION_KINDS} (the DECISION-level kind, never a RequestKind): {kind!r}")
    if not isinstance(subject, str) or not subject:  # the LAST joined field: it may contain "|" (the "|kill" subjects do)
        raise ValueError(f"subject must be a non-empty str, got {subject!r}")
    parts = [
        _field("namespace", namespace),
        _session("session", session).isoformat(),
        _field("underlying", underlying),
        str(kind),
        str(subject),
    ]
    return canon.sha256_hex("|".join(parts))[:32]


def kill_subject(position_id: str) -> str:
    """Decision subject of a kill-switch close of one position: position_id + "|kill"."""
    return _field("position_id", position_id) + _KILL_SUFFIX


def equity_kill_subject(symbol: str) -> str:
    """Decision subject of the assigned-stock flatten: "EQ:<symbol>|kill"."""
    return f"{_EQUITY_PREFIX}{_field('symbol', symbol)}{_KILL_SUFFIX}"


def position_id(namespace: str, open_session: date, structure_id: str) -> str:
    """sha256("|".join([namespace, open_session.isoformat(), structure_id]))[:16]"""
    if not isinstance(structure_id, str) or not structure_id:
        raise ValueError(f"structure_id must be a non-empty str, got {structure_id!r}")
    parts = [_field("namespace", namespace), _session("open_session", open_session).isoformat(), str(structure_id)]
    return canon.sha256_hex("|".join(parts))[:16]


def equity_position_id(symbol: str) -> str:
    """Pseudo position id of an assigned equity position: sha256("EQ:" + symbol)[:16]."""
    return canon.sha256_hex(_EQUITY_PREFIX + _field("symbol", symbol))[:16]


def intent_id(namespace: str, session: date, decision_id: str, purpose: OrderPurpose, part: int) -> str:
    """f"jb1-{ns8(namespace)}-{session:%y%m%d}-{decision_id[:12]}-{purpose.value}-{part:02d}"

    D18, literally: the id is a function of (decision_id, action, attempt). It does NOT contain position_id / structure_id, so
    an OPEN order's id does not depend on which strikes the candidate generator picked: a restarted cycle that rebuilt a
    different candidate still maps to the same broker-side id (INV-07).
    """
    if not isinstance(decision_id, str) or _DECISION_ID_RE.fullmatch(decision_id) is None:
        raise ValueError(f"decision_id must be the 32-hex-char ids.decision_id(...), got {decision_id!r}")
    action = OrderPurpose(purpose)
    leg_part = _int_in("part", part, 0, _MAX_PART)
    if leg_part != 0 and action is not OrderPurpose.KILL:
        raise ValueError(f"part {leg_part} is a per-leg kill fallback order; purpose {action.value!r} orders are always part 0")
    day = _session("session", session)
    if not 2000 <= day.year <= 2099:
        raise ValueError(f"session year outside 2000..2099 cannot be written as yymmdd: {day.isoformat()}")
    return f"{ORDER_ID_PREFIX}{ns8(namespace)}-{day:%y%m%d}-{decision_id[:12]}-{action.value}-{leg_part:02d}"


def client_order_id(intent: OrderIntent, attempt: int) -> str:
    """f"{intent.intent_id}-{attempt:02d}", e.g. "jb1-3fa91c20-260917-9c1e44a07b2d-close-00-01" (<= 48 chars; Alpaca limit 128).

    Probe orders (11.11) use the separate prefix "jbp-"; reconcile R1 treats anything that is not "jb1-" as foreign.
    """
    if _INTENT_ID_RE.fullmatch(intent.intent_id) is None:
        raise ValueError(f"intent.intent_id is not an ids.intent_id(...) value: {intent.intent_id!r}")
    return f"{intent.intent_id}-{_int_in('attempt', attempt, 0, _MAX_ATTEMPT):02d}"


def is_bot_order_id(client_order_id: str) -> bool:
    """True iff `client_order_id` has exactly the shape this module emits ("jb1-..."); probe ("jbp-") and foreign ids are False."""
    return isinstance(client_order_id, str) and _CLIENT_ORDER_ID_RE.fullmatch(client_order_id) is not None


def fill_id(client_order_id: str, cum_qty: int) -> str:
    """sha256(f"{client_order_id}|{cum_qty}")[:24] - keyed by the CUMULATIVE filled quantity, UNIQUE in the run store."""
    cid = _field("client_order_id", client_order_id)
    return canon.sha256_hex(f"{cid}|{_int_in('cum_qty', cum_qty, 1)}")[:24]


def forecast_id(decision_id: str, question_id: str, with_text: bool) -> str:
    """sha256(f"{decision_id}|{question_id}|{int(with_text)}")[:24] (2.7).

    `with_text` is IN the id: the 12 eval questions ride in both entry batches under ONE decision_id, so without it the two
    forecast sets would collide.
    """
    if not isinstance(with_text, bool):
        raise TypeError(f"with_text must be a bool, got {type(with_text).__name__}")
    return canon.sha256_hex(f"{_field('decision_id', decision_id)}|{_field('question_id', question_id)}|{int(with_text)}")[:24]


def event_key(underlying: str, key: SnapshotKey, spec: OutcomeSpec) -> str:
    """sha256(canon.dumps_sorted({"u": underlying, "session": ..., "slot": ..., "spec": to_builtins(spec)}))[:24]

    No decider information: every decider, baseline and reference asked about the same event shares the key.
    """
    material = {
        "u": _field("underlying", underlying),
        "session": _session("key.session", key.session).isoformat(),
        "slot": key.slot.value,
        "spec": msgspec.to_builtins(spec),
    }
    return canon.sha256_hex(canon.dumps_sorted(material))[:24]
