"""Property tests for canon.py over generated state-safe objects (seeded numpy generators; no hypothesis dependency)."""

import json
from typing import Any

import msgspec
import numpy as np
import pytest

from jevbot import canon
from jevbot.errors import StateLeak, StateTypeError

SEEDS = range(40)

_KEY_HEAD = "abcdefghijklmnopqrstuvwxyz"
_KEY_TAIL = _KEY_HEAD + "0123456789_"
# state-safe text: no "$", no digits (digits are added below as short numbers only), but everything JSON has to escape
_TEXT_ALPHABET = list("abcdefghij KLMNOP xyz.,:;-_()[]%/'\"\\\n\t\r\x00\x1f\x7fé€ßñ漢字😀 ​")
_FORBIDDEN_KEYS = {"uid", "uuid", "timestamp", "ts", "date", "datetime", "time", "created_at"}


def _key(rng: np.random.Generator) -> str:
    while True:
        n = int(rng.integers(1, 9))
        key = str(rng.choice(list(_KEY_HEAD))) + "".join(rng.choice(list(_KEY_TAIL), size=n - 1))
        # masked mode also runs the leak patterns over keys: a generated key may not look like a 3+ digit number or a year
        if key not in _FORBIDDEN_KEYS:
            return key


def _text(rng: np.random.Generator) -> str:
    pieces: list[str] = []
    for _ in range(int(rng.integers(0, 6))):
        if rng.random() < 0.25:
            pieces.append(str(int(rng.integers(0, 100))))  # at most two digits: never a "3+ digit number"
            pieces.append(" ")
        else:
            # picked by index: a numpy str array would silently drop the NUL character
            pieces.append("".join(_TEXT_ALPHABET[int(i)] for i in rng.integers(0, len(_TEXT_ALPHABET), size=int(rng.integers(1, 8)))))
            pieces.append(" ")
    return "".join(pieces)


def _value(rng: np.random.Generator, depth: int) -> Any:
    roll = rng.random()
    if depth >= 4 or roll < 0.45:
        kind = int(rng.integers(0, 5))
        if kind == 0:
            return _text(rng)
        if kind == 1:
            return int(rng.integers(-1000, 1001))
        if kind == 2:
            return bool(rng.integers(0, 2))
        if kind == 3:
            return None
        return _text(rng)
    if roll < 0.75:
        return _dict(rng, depth + 1)
    return [_value(rng, depth + 1) for _ in range(int(rng.integers(0, 5)))]


def _dict(rng: np.random.Generator, depth: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for _ in range(int(rng.integers(0, 6))):
        out[_key(rng)] = _value(rng, depth)
    return out


def _state(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    state = {"schema": "state.v1.entry"}
    state.update(_dict(rng, 0))
    while len(state) < 3:  # at least a few top-level keys, so that key order is observable
        state[_key(rng)] = _value(rng, 1)
    return state


def _reverse_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _reverse_keys(obj[k]) for k in reversed(list(obj))}
    if isinstance(obj, list):
        return [_reverse_keys(v) for v in obj]
    return obj


def _key_orders(obj: Any) -> list[list[str]]:
    """Key order of every dict, in document order."""
    out: list[list[str]] = []
    if isinstance(obj, dict):
        out.append(list(obj))
        for v in obj.values():
            out.extend(_key_orders(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_key_orders(v))
    return out


def _paths(obj: Any, prefix: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """Every container slot (path of keys / indexes) of the object."""
    out: list[tuple[Any, ...]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append((*prefix, k))
            out.extend(_paths(v, (*prefix, k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.append((*prefix, i))
            out.extend(_paths(v, (*prefix, i)))
    return out


def _with(obj: Any, path: tuple[Any, ...], value: Any) -> Any:
    """Deep copy of `obj` with the slot at `path` replaced by `value`."""
    if not path:
        return value
    head, rest = path[0], path[1:]
    if isinstance(obj, dict):
        return {k: (_with(v, rest, value) if k == head else _with(v, (), v)) for k, v in obj.items()}
    return [(_with(v, rest, value) if i == head else v) for i, v in enumerate(obj)]


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_states_are_state_safe_and_round_trip(seed: int) -> None:
    state = _state(seed)
    canon.ensure_state_safe(state, underlyings=("SPY", "QQQ", "IWM"))
    ordered, sorted_ = canon.dumps_ordered(state), canon.dumps_sorted(state)
    assert json.loads(ordered) == state  # loads(dumps(x)) == x
    assert json.loads(sorted_) == state
    assert _key_orders(json.loads(ordered)) == _key_orders(state)  # insertion order survives the ordered encoding
    assert all(keys == sorted(keys) for keys in _key_orders(json.loads(sorted_)))
    assert canon.dumps_ordered(json.loads(ordered)) == ordered  # idempotent
    assert canon.dumps_sorted(json.loads(ordered)) == sorted_  # the sorted form does not depend on which encoding was parsed


@pytest.mark.parametrize("seed", SEEDS)
def test_ordered_encoding_is_byte_identical_to_the_sdk_wire_encoding(seed: int) -> None:
    # 5.9: the SDK serialises with msgspec.json.encode; state_hash / cache keys must be taken over exactly those bytes
    state = _state(seed)
    assert msgspec.json.encode(state) == canon.dumps_ordered(state).encode("utf-8")


@pytest.mark.parametrize("seed", SEEDS)
def test_key_order_changes_the_ordered_hash_only(seed: int) -> None:
    state = _state(seed)
    permuted = _reverse_keys(state)
    assert permuted == state
    assert canon.dumps_sorted(permuted) == canon.dumps_sorted(state)
    assert canon.dumps_ordered(permuted) != canon.dumps_ordered(state)  # >= 3 top-level keys: the reversed order is different
    question = {"type": "noul", "instructions": "x", "criteria": {"true": "a", "false": "b"}}
    assert canon.cache_key("m", permuted, "h", question) != canon.cache_key("m", state, "h", question)
    assert canon.sha256_hex(canon.dumps_sorted(permuted)) == canon.sha256_hex(canon.dumps_sorted(state))


@pytest.mark.parametrize("seed", SEEDS)
def test_one_bad_value_anywhere_is_always_caught(seed: int) -> None:
    rng = np.random.default_rng(10_000 + seed)
    state = _state(seed)
    paths = _paths(state)
    path = paths[int(rng.integers(0, len(paths)))]

    type_poisons: list[Any] = [0.5, float("nan"), (1, 2), np.int64(7), np.float64(0.25), 1001, -45000]
    for poison in type_poisons:
        poisoned = _with(state, path, poison)
        with pytest.raises(StateTypeError):
            canon.ensure_state_safe(poisoned)
    for poison in (0.5, float("inf"), np.int64(7), np.float64(0.25), b"raw"):
        poisoned = _with(state, path, poison)
        with pytest.raises(TypeError):
            canon.dumps_ordered(poisoned)
        with pytest.raises(TypeError):
            canon.dumps_sorted(poisoned)

    for leak in ("SPY", "on 2024-03-15", "back in 2008", "$3", "level 4500"):
        poisoned = _with(state, path, leak)
        with pytest.raises(StateLeak):
            canon.ensure_state_safe(poisoned, underlyings=("SPY",))
        canon.ensure_state_safe(poisoned, masked=False, underlyings=("SPY",))
        assert json.loads(canon.dumps_ordered(poisoned)) == poisoned  # a leak is still canonical JSON


def test_the_generator_covers_what_it_claims() -> None:
    states = [_state(seed) for seed in SEEDS]
    blob = "".join(canon.dumps_ordered(s) for s in states)
    assert "\\u0000" in blob and "\\n" in blob and '\\"' in blob and "\\\\" in blob  # escapes are exercised
    assert "😀" in blob and "漢" in blob  # non-ASCII is kept verbatim (ensure_ascii=False)
    depth = max(len(p) for s in states for p in _paths(s))
    assert depth >= 4
    assert any(isinstance(v, list) and v for s in states for v in s.values())
