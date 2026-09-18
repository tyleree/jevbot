"""Integer money helpers (DESIGN.md section 3.7): ceiling division, tick tables, adverse rounding of net prices and the
limit-sign assertion.

Conventions: money is integer; option prices are CENTS PER SHARE; signed net prices follow Alpaca's mleg convention -
positive = net debit (we pay), negative = net credit (we receive). On that signed axis "more negative" is always better for us
and "more positive" is always worse, for debits and credits alike, which is what the two rounding modes are defined on.
"""

from collections.abc import Sequence
from typing import Final

from jevbot.errors import InvariantError
from jevbot.types import SHORT_PREMIUM, Cents, OrderPurpose, StructureKind

__all__ = ["assert_limit_sign", "cdiv", "round_net", "tick_cents"]

_SINGLE_LEG_KINDS: Final[frozenset[StructureKind]] = frozenset({StructureKind.LONG_CALL, StructureKind.LONG_PUT})
_NICKEL_DIME_BREAK_CENTS: Final = 300  # non-penny classes: $0.05 below $3.00, $0.10 at / above
_TICK_PENNY: Final = 1
_TICK_NICKEL: Final = 5
_TICK_DIME: Final = 10


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def cdiv(a: int, b: int) -> int:
    """Ceiling division for non-negative `a`, positive `b` (pure integer arithmetic; anything else is a ValueError)."""
    if not _is_int(a) or not _is_int(b):
        raise TypeError(f"cdiv takes ints, got {type(a).__name__} and {type(b).__name__}")
    if a < 0 or b <= 0:
        raise ValueError(f"cdiv is defined for a >= 0 and b > 0, got a={a}, b={b}")
    return -(-a // b)


def tick_cents(underlying: str, px_cents: int, penny_all: Sequence[str]) -> int:
    """Price increment in cents: 1 if `underlying` is a penny class (`universe.penny_all`); else 5 below 300, 10 at / above 300.

    `px_cents` is a LEG's premium in cents per share; its magnitude is used, so a signed single-leg price gives the same tick.
    Multi-leg nets use the finest tick among the legs (section 8) - the caller takes the min over its legs.
    """
    if isinstance(penny_all, str):
        raise TypeError("penny_all must be a sequence of symbols, not a single string")
    if not _is_int(px_cents):
        raise TypeError(f"px_cents must be an int (cents per share), got {type(px_cents).__name__}")
    if underlying in penny_all:
        return _TICK_PENNY
    return _TICK_NICKEL if abs(px_cents) < _NICKEL_DIME_BREAK_CENTS else _TICK_DIME


def round_net(net_cents: int, tick: int, *, aggressive: bool) -> int:
    """Round a signed net price (cents per share; + debit / - credit) onto the tick grid, never landing on 0.

    passive (`aggressive=False`; entries and discretionary exits): never pay more / accept less than computed - a debit rounds
        DOWN in magnitude, a credit UP in magnitude, i.e. the signed value is floored: result <= net_cents ...
    aggressive (`aggressive=True`; mandatory exits only): the inverse - the signed value is ceiled: result >= net_cents.

    0 is illegal -> +/- 1 tick. THE SUB-TICK RULE (frozen here; the 3.7 comment leaves the sign open), for the only inputs
    whose rounded value is 0 - `net_cents == 0` and `0 < |net_cents| < tick`:
    - passive keeps the SIGN of a non-zero net: a sub-tick debit becomes `+tick` (the smallest debit on the grid; this is the
      one case where the passive result is worse than computed, by less than one tick), a sub-tick credit `-tick` (the floor,
      one tick better for us), and exactly 0 becomes `-tick` (demand one tick rather than pay one). So a passive OPEN limit
      can never trip `assert_limit_sign` through rounding alone (a debit structure stays a debit), and a discretionary CLOSE
      whose natural is a few cents of debit is posted as a fillable one-tick debit instead of an unfillable credit demand.
    - aggressive always stays marketable: the result is >= net_cents for EVERY input, so a sub-tick credit (`-3` on a 5c
      grid) becomes `+tick` - we pay one tick to get a mandatory exit done (a close may legitimately cross zero; 3.7); a
      sub-tick debit becomes `+tick` and exactly 0 becomes `+tick`.
    Consequences (the property tests pin them): `passive <= aggressive` always; `net_cents <= aggressive` always;
    `passive <= net_cents` except for `0 < net_cents < tick`, where `passive == tick`; a non-zero on-grid net is unchanged
    in both modes; both modes are idempotent. With real quotes a multi-leg natural is always on the finest-tick grid (every leg
    quote is a multiple of its own tick), so the sub-tick band is reached only by `net_cents == 0`, by ladder rungs between
    mid and natural (11.6) and by off-grid synthetic data. The result is an integer number of cents (<= 2 decimals in dollars).

    SINGLE-LEG callers: an order is sent as abs(limit) with the leg's side, so its limit must carry the sign of that side
    (selling: negative). A zero-bid long leg has net 0; give it `-tick` yourself instead of the aggressive `+tick`.
    """
    if not _is_int(net_cents) or not _is_int(tick):
        raise TypeError(f"round_net takes ints, got {type(net_cents).__name__} and {type(tick).__name__}")
    if tick <= 0:
        raise ValueError(f"tick must be a positive number of cents, got {tick}")
    floored = (net_cents // tick) * tick  # floor division rounds toward -infinity for either sign
    if aggressive:
        rounded = floored if floored == net_cents else floored + tick  # the ceiling
        return rounded if rounded != 0 else tick  # 0 -> +tick: still >= net_cents, still marketable
    if floored != 0:
        return floored
    return tick if net_cents > 0 else -tick  # 0 -> the sign of the net (a sub-tick debit stays a debit); exactly 0 -> -tick


def assert_limit_sign(purpose: OrderPurpose, kind: StructureKind | None, limit: int | None, *, width: Cents, pad: Cents) -> None:
    """Guard the Level-3 condor sign trap (critique corr. 6): a violation is an `InvariantError` (a bug, never a soft reject).

    OPEN credit structure: limit < 0; OPEN debit structure (incl. single legs): limit > 0.
    CLOSE / KILL of a MULTI-LEG structure: the opposite sign is EXPECTED but a close may legitimately cross zero, so closes assert
        only |limit| <= width + pad, where width = Structure.width (max wing) and pad = the largest cushion a mandatory / kill
        close may ever carry: pad = ceil(kill.cushion_max_frac_width * width), computed by the caller (discretionary closes pass 0).
    SINGLE-LEG orders (kind LONG_CALL / LONG_PUT, or kind None = a per-leg kill fallback order): width is 0, so the width bound is
        SKIPPED; they assert only limit != 0 with the sign of the leg's side - selling a long leg to close: limit < 0. For
        kind None the side is not known here (a fallback leg may be bought back, limit > 0, or sold, limit < 0): only limit != 0.
    Market orders (limit None: the equity flatten, the last-resort kill leg) are not checked.
    Single-leg orders are sent to Alpaca as abs(limit) with the leg's side.
    """
    if limit is None:
        return
    if not _is_int(limit):
        raise InvariantError(f"limit must be an int (signed cents per share) or None, got {type(limit).__name__}")
    if not isinstance(purpose, OrderPurpose):
        raise InvariantError(f"purpose must be an OrderPurpose, got {purpose!r}")
    if kind is not None and not isinstance(kind, StructureKind):
        raise InvariantError(f"kind must be a StructureKind or None, got {kind!r}")
    if not _is_int(width) or not _is_int(pad) or width < 0 or pad < 0:
        raise InvariantError(f"width and pad must be non-negative ints (cents), got width={width!r}, pad={pad!r}")

    label = f"{purpose.value} {kind.value if kind is not None else 'single leg (kill fallback)'}"
    if purpose is OrderPurpose.OPEN:
        if kind is None:
            raise InvariantError("an OPEN order always has a structure kind (kind None is a per-leg kill fallback order)")
        if kind in SHORT_PREMIUM:
            if limit >= 0:
                raise InvariantError(f"{label}: a credit structure opens for a net credit, limit must be < 0, got {limit}")
        elif limit <= 0:
            raise InvariantError(f"{label}: a debit structure opens for a net debit, limit must be > 0, got {limit}")
        return

    # CLOSE / KILL
    if kind is None or kind in _SINGLE_LEG_KINDS:
        if limit == 0:
            raise InvariantError(f"{label}: a limit of 0 is illegal")
        if kind is not None and limit > 0:
            raise InvariantError(f"{label}: closing a long single leg SELLS it, limit must be < 0, got {limit}")
        return
    if width <= 0:
        raise InvariantError(f"{label}: a multi-leg structure has a positive width, got {width}")
    if abs(limit) > width + pad:
        raise InvariantError(f"{label}: |limit| = {abs(limit)} exceeds width + pad = {width} + {pad}")
