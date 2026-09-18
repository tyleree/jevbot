"""XNYS calendar, simulated clock and the two single-sourced clocks of the Conventions (DESIGN.md sections 3.1 and 3.7).

* `XnysCalendar` implements the `Calendar` Protocol on `exchange_calendars` "XNYS". Every cut-off in the package is an offset
  from a session's calendar close (`offset_from_close`, INV-13): this module contains no time-of-day value at all - opens and
  closes, early closes included, come from the exchange calendar.
* `SimClock` implements the `Clock` Protocol for backtests (`set(ts)` by the backtest loop).
* `year_fraction` is THE calendar day-count (calendar / 365): discounting, forwards, annualised IV levels, `T_E`.
* `trading_time` is THE variance clock, in sessions (V13): it allocates total variance to a horizon below the first usable
  expiry or between two expiries. Single-sourced here so that expected moves (5.3) and implied probabilities (6.4) can never
  disagree.
* `last_session(calendar, expiry)` = `calendar.prev_or_same_session(expiry)`: `expiry` is contract identity only and may be a
  Saturday (monthlies listed before February 2015) or follow a holiday (Good-Friday weeks); every time computation uses
  `close(last_session)`.

All datetimes are tz-aware UTC; a naive datetime is refused. A `session` is a `datetime.date` (exchange-local trading day); a
`datetime` is refused where a date is expected (its UTC date and its exchange-local date can differ).
"""

import warnings
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from typing import TYPE_CHECKING, Final, Protocol

import numpy as np

from jevbot.errors import InvariantError
from jevbot.types import ClockReading

__all__ = ["SimClock", "XnysCalendar", "last_session", "trading_time", "year_fraction"]

_EXCHANGE: Final = "XNYS"
# Default bounds. The start covers the longest history any table carries (the Cboe index files begin in 1990); the end is a
# fixed far horizon rather than "today + n": it keeps the wall clock out of this module and already-listed long-dated expiries
# inside the calendar. Holidays beyond the library's known special closes are rule-based projections.
_DEFAULT_START: Final = date(1990, 1, 1)
_DEFAULT_END: Final = date(2050, 12, 31)
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_SECONDS_PER_YEAR: Final = 365 * 86400
_ONE_DAY: Final = timedelta(days=1)


class _SessionTimes(Protocol):
    """The slice of the `Calendar` Protocol (3.1) that `trading_time` / `last_session` need; any `Calendar` satisfies it."""

    def sessions(self, start: date, end: date) -> list[date]: ...

    def open_close(self, session: date) -> tuple[datetime, datetime]: ...

    def prev_or_same_session(self, d: date) -> date: ...


@dataclass(frozen=True)
class _Tables:
    dates: tuple[date, ...]  # every session, ascending
    ordinals: tuple[int, ...]  # date.toordinal() of `dates` (bisect keys)
    opens: tuple[datetime, ...]  # UTC
    closes: tuple[datetime, ...]  # UTC
    early: frozenset[date]  # sessions that close before the regular close


@lru_cache(maxsize=8)
def _tables(start: date, end: date) -> _Tables:
    """Session tables of XNYS in [start, end]; immutable and shared by every XnysCalendar with the same bounds.

    The ONLY place that touches exchange_calendars: it is imported here, once, and everything the package needs is copied into
    plain tuples. The library's own pandas / numpy deprecation noise is silenced for exactly that window, so that a run with
    warnings promoted to errors can still build its calendar.
    """

    def utc_datetimes(series: object) -> tuple[datetime, ...]:
        # session opens / closes are whole minutes: integer seconds since the epoch lose nothing
        seconds = np.asarray(series, dtype="datetime64[s]").astype(np.int64)
        return tuple(_EPOCH + timedelta(seconds=int(v)) for v in seconds)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        import exchange_calendars

        xcal = exchange_calendars.get_calendar(_EXCHANGE, start=start.isoformat(), end=end.isoformat())
        dates = tuple(date(ts.year, ts.month, ts.day) for ts in xcal.sessions)
        opens = utc_datetimes(xcal.opens.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy())
        closes = utc_datetimes(xcal.closes.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy())
        early = frozenset(date(ts.year, ts.month, ts.day) for ts in xcal.early_closes)
    if not dates or not (len(dates) == len(opens) == len(closes)):
        raise InvariantError("exchange_calendars returned inconsistent XNYS session tables")
    for i, (o, c) in enumerate(zip(opens, closes, strict=True)):
        if not o < c or (i > 0 and not closes[i - 1] < o):
            raise InvariantError(f"XNYS session tables are not strictly ordered at {dates[i].isoformat()}")
    return _Tables(dates=dates, ordinals=tuple(d.toordinal() for d in dates), opens=opens, closes=closes, early=early)


def _as_date(name: str, value: date) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{name} must be a datetime.date, got {type(value).__name__}")
    return value


def _as_utc(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be tz-aware (all datetimes are tz-aware UTC), got a naive datetime")
    return value.astimezone(UTC)


def _as_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


class XnysCalendar:
    """`Calendar` (3.1) on exchange_calendars "XNYS". Immutable; lookups are binary searches over precomputed tables.

    A date outside `bounds` raises ValueError - never a silent "not a session": beyond the horizon the true answer is unknown.
    """

    def __init__(self, start: date | None = None, end: date | None = None) -> None:
        lo = _as_date("start", start) if start is not None else _DEFAULT_START
        hi = _as_date("end", end) if end is not None else _DEFAULT_END
        if not lo < hi:
            raise ValueError(f"calendar start {lo.isoformat()} must be before end {hi.isoformat()}")
        self._t: Final = _tables(lo, hi)
        self._lo: Final = lo
        self._hi: Final = hi

    # --- helpers --------------------------------------------------------------------------------------------------------

    @property
    def bounds(self) -> tuple[date, date]:
        """(first date, last date) this calendar can answer for, inclusive."""
        return (self._lo, self._hi)

    def _ordinal(self, name: str, d: date) -> int:
        day = _as_date(name, d)
        if not self._lo <= day <= self._hi:
            raise ValueError(f"{name} {day.isoformat()} is outside the calendar bounds {self._lo.isoformat()}..{self._hi.isoformat()}")
        return day.toordinal()

    def _index(self, session: date) -> int:
        ordinal = self._ordinal("session", session)
        i = bisect_left(self._t.ordinals, ordinal)
        if i == len(self._t.ordinals) or self._t.ordinals[i] != ordinal:
            raise ValueError(f"{session.isoformat()} is not an XNYS session")
        return i

    def _at(self, i: int, what: str) -> date:
        if not 0 <= i < len(self._t.dates):
            raise ValueError(f"{what} lies outside the calendar bounds {self._lo.isoformat()}..{self._hi.isoformat()}")
        return self._t.dates[i]

    # --- Calendar protocol ----------------------------------------------------------------------------------------------

    def is_session(self, d: date) -> bool:
        ordinal = self._ordinal("d", d)
        i = bisect_left(self._t.ordinals, ordinal)
        return i < len(self._t.ordinals) and self._t.ordinals[i] == ordinal

    def sessions(self, start: date, end: date) -> list[date]:
        """Sessions in [start, end], inclusive; [] when end < start."""
        lo = bisect_left(self._t.ordinals, self._ordinal("start", start))
        hi = bisect_right(self._t.ordinals, self._ordinal("end", end))
        return list(self._t.dates[lo:hi])

    def open_close(self, session: date) -> tuple[datetime, datetime]:
        """(open, close) of a session, UTC; early closes honoured (D4, G4). ValueError when `session` is not a session."""
        i = self._index(session)
        return (self._t.opens[i], self._t.closes[i])

    def is_early_close(self, session: date) -> bool:
        return self._t.dates[self._index(session)] in self._t.early

    def next_session(self, d: date, n: int = 1) -> date:
        """The n-th session strictly after `d` (any calendar date)."""
        i = bisect_right(self._t.ordinals, self._ordinal("d", d)) + _as_count("n", n) - 1
        return self._at(i, f"session {n} after {d.isoformat()}")

    def prev_session(self, d: date, n: int = 1) -> date:
        """The n-th session strictly before `d` (any calendar date)."""
        i = bisect_left(self._t.ordinals, self._ordinal("d", d)) - _as_count("n", n)
        return self._at(i, f"session {n} before {d.isoformat()}")

    def prev_or_same_session(self, d: date) -> date:
        """`d` if it is a session, else the last session before it. THE map expiry -> last_session: `d` may be ANY calendar date
        (a Saturday-dated monthly, a holiday)."""
        i = bisect_right(self._t.ordinals, self._ordinal("d", d)) - 1
        return self._at(i, f"session on or before {d.isoformat()}")

    def sessions_between(self, a: date, b: date) -> int:
        """Number of sessions in (a, b]; 0 when b <= a."""
        lo = bisect_right(self._t.ordinals, self._ordinal("a", a))
        hi = bisect_right(self._t.ordinals, self._ordinal("b", b))
        return max(0, hi - lo)

    def session_of(self, ts: datetime) -> date | None:
        """The session whose [open, close] contains `ts` (both ends inclusive), else None."""
        when = _as_utc("ts", ts)
        self._check_instant(when)
        i = bisect_left(self._t.closes, when)  # first session that closes at or after ts
        if i < len(self._t.closes) and self._t.opens[i] <= when:
            return self._t.dates[i]
        return None

    def offset_from_close(self, session: date, minutes_before: int) -> datetime:
        """close(session) - minutes_before: THE way to express a cut-off (INV-13); negative = after the close."""
        if isinstance(minutes_before, bool) or not isinstance(minutes_before, int):
            raise TypeError(f"minutes_before must be an int, got {type(minutes_before).__name__}")
        return self._t.closes[self._index(session)] - timedelta(minutes=minutes_before)

    def next_open_after(self, ts: datetime) -> datetime:
        """The first session open strictly after `ts` (knowable_at of EOD series)."""
        when = _as_utc("ts", ts)
        self._check_instant(when)
        i = bisect_right(self._t.opens, when)
        if i >= len(self._t.opens):
            raise ValueError(f"no session opens after {when.isoformat()} inside the calendar bounds")
        return self._t.opens[i]

    def _check_instant(self, when: datetime) -> None:
        # one day of slack on each side: an instant's UTC date can differ from the exchange-local date
        if not self._lo - _ONE_DAY <= when.date() <= self._hi + _ONE_DAY:
            raise ValueError(f"{when.isoformat()} is outside the calendar bounds {self._lo.isoformat()}..{self._hi.isoformat()}")


class SimClock:
    """`Clock` (3.1) for backtests and tests: `now()` is whatever the loop last `set()`. It never reads the wall clock."""

    def __init__(self, ts: datetime | None = None) -> None:
        self._now: datetime | None = _as_utc("ts", ts) if ts is not None else None

    def set(self, ts: datetime) -> None:
        self._now = _as_utc("ts", ts)

    def now(self) -> datetime:
        if self._now is None:
            raise InvariantError("SimClock.now() was called before SimClock.set(ts)")
        return self._now

    def reading(self) -> ClockReading | None:
        return None  # there is no broker clock in a simulation

    def sync(self) -> ClockReading | None:
        return None  # no-op


def year_fraction(a: datetime, b: datetime) -> float:
    """(b - a).total_seconds() / (365 * 86400) - THE CALENDAR day-count: discounting, forwards, annualised IV levels
    (iv30 / iv90 / per-expiry IV), T_E. Signed: negative when b is before a."""
    return (_as_utc("b", b) - _as_utc("a", a)).total_seconds() / _SECONDS_PER_YEAR


def trading_time(calendar: _SessionTimes, a: datetime, b: datetime) -> float:
    """THE VARIANCE CLOCK, in sessions: every whole regular session in (a, b] counts 1.0 (an early-close session too); a partial
    session counts elapsed_minutes / that session's own length; 0.0 when b <= a.

    Used ONLY to allocate total variance to a horizon that lies below the first usable expiry or between two expiries (5.3
    expected moves, 6.4 implied probabilities). Nights, weekends and holidays carry no trading time: close(D) -> close(D + 1)
    is 1.0 on a Tuesday and across a weekend alike (V13).
    """
    start, end = _as_utc("a", a), _as_utc("b", b)
    if end <= start:
        return 0.0
    # one day of slack on each side: a session's exchange-local date can differ from the UTC dates of its open and close
    days = calendar.sessions(start.date() - _ONE_DAY, end.date() + _ONE_DAY)
    total = 0.0
    lo, hi = 0, len(days)
    while lo < hi:  # head: sessions that open before `a` (wholly or partly outside the interval)
        opened, closed = calendar.open_close(days[lo])
        if opened >= start:
            break
        total += _overlap_share(opened, closed, start, end)
        lo += 1
    while hi > lo:  # tail: sessions that close after `b`
        opened, closed = calendar.open_close(days[hi - 1])
        if closed <= end:
            break
        total += _overlap_share(opened, closed, start, end)
        hi -= 1
    return total + float(hi - lo)  # what is left opens at / after `a` and closes at / before `b`: whole sessions


def _overlap_share(opened: datetime, closed: datetime, start: datetime, end: datetime) -> float:
    """Share of the session [opened, closed] that lies inside [start, end], by elapsed time over the session's OWN length."""
    overlap = (min(closed, end) - max(opened, start)).total_seconds()
    if overlap <= 0.0:
        return 0.0
    return overlap / (closed - opened).total_seconds()


def last_session(calendar: _SessionTimes, expiry: date) -> date:
    """calendar.prev_or_same_session(expiry) (Conventions); a convenience wrapper."""
    return calendar.prev_or_same_session(expiry)


if TYPE_CHECKING:
    # Static proof, checked by the mypy gate: the two classes implement the Protocols of section 3.1. Import-free at runtime, so
    # this module depends on nothing but errors.py and types.py.
    from jevbot.protocols import Calendar, Clock

    _XNYS_CALENDAR_IS_A_CALENDAR: Calendar = XnysCalendar()
    _SIM_CLOCK_IS_A_CLOCK: Clock = SimClock()
