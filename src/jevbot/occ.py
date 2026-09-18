"""OCC option symbol format / parse (DESIGN.md section 2.2).

`format_occ` writes the unpadded form Alpaca and the mirror use (`SPY261016C00600000`); `parse_occ` also accepts the 21-character
space-padded OSI form (`SPY   261016C00600000`).

The expiry inside a symbol is the LISTED OCC date: contract identity only. It need not be a session - standard monthlies listed
before February 2015 carry Saturday dates - and it is never used for time arithmetic (that is always
`calendar.prev_or_same_session(expiry)`, Conventions / INV-11). This module therefore never consults a calendar.
"""

import re
from datetime import date, datetime
from typing import Final

from jevbot.types import OptionContract, Right

__all__ = ["format_occ", "is_occ", "parse_occ"]

# root (1-6 capital letters; adjusted roots containing digits are rejected in v1), optional OSI space padding, yymmdd, right,
# strike * 1000. ASCII classes are spelt out on purpose: `\d` would also accept non-ASCII digits (which `int()` parses) and `\s`
# a tab or a newline inside a symbol.
_OCC_RE: Final = re.compile(r"([A-Z]{1,6}) *([0-9]{6})([CP])([0-9]{8})")
_ROOT_RE: Final = re.compile(r"[A-Z]{1,6}")
_STRIKE_MILLI_LIMIT: Final = 10**8
_CENTURY: Final = 2000


def format_occ(c: OptionContract) -> str:
    """f"{root}{yy}{mm}{dd}{C|P}{strike_milli:08d}", root unpadded (Alpaca + mirror form), e.g. "SPY261016C00600000"."""
    if _ROOT_RE.fullmatch(c.underlying) is None:
        raise ValueError(f"invalid OCC root {c.underlying!r}: 1-6 capital letters (adjusted roots are rejected in v1)")
    if isinstance(c.expiry, datetime) or not isinstance(c.expiry, date):
        raise ValueError(f"expiry must be a datetime.date, got {type(c.expiry).__name__}")
    if not _CENTURY <= c.expiry.year < _CENTURY + 100:
        raise ValueError(f"expiry year outside 2000..2099 cannot be written as an OCC yymmdd date: {c.expiry.isoformat()}")
    if isinstance(c.strike_milli, bool) or not isinstance(c.strike_milli, int) or not 0 < c.strike_milli < _STRIKE_MILLI_LIMIT:
        raise ValueError(f"strike_milli out of range (0 < x < 10**8): {c.strike_milli!r}")
    right = Right(c.right)
    return f"{c.underlying}{c.expiry:%y%m%d}{right.value}{c.strike_milli:08d}"


def parse_occ(sym: str) -> OptionContract:
    """Regex ^([A-Z]{1,6}) *([0-9]{6})([CP])([0-9]{8})$ over the WHOLE string; accepts the 21-char space-padded OSI form.

    Century 20yy. An invalid calendar date, a root with digits (adjusted root), a strike of 0, lower case, leading / trailing
    whitespace (a trailing newline included), non-ASCII digits or anything else that does not match -> ValueError.
    """
    if not isinstance(sym, str):
        raise ValueError(f"OCC symbol must be a str, got {type(sym).__name__}")
    match = _OCC_RE.fullmatch(sym)
    if match is None:
        raise ValueError(f"not an OCC option symbol: {sym!r}")
    root, yymmdd, right, strike = match.groups()
    try:
        expiry = date(_CENTURY + int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6]))
    except ValueError as exc:
        raise ValueError(f"invalid expiry date in OCC symbol {sym!r}: {exc}") from None
    strike_milli = int(strike)
    if strike_milli <= 0:
        raise ValueError(f"zero strike in OCC symbol {sym!r}")
    return OptionContract(underlying=root, expiry=expiry, right=Right(right), strike_milli=strike_milli)


def is_occ(sym: str) -> bool:
    """True iff `parse_occ(sym)` succeeds (e.g. to tell an option leg from an assigned equity position)."""
    try:
        parse_occ(sym)
    except ValueError:
        return False
    return True
