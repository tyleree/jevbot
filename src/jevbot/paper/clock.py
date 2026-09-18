"""`BrokerClock` (round-trip compensated, CLOCK_BOOTTIME bridged) and `AlpacaCalendar` (DESIGN.md 11.4, 3.1; D25, G4; INV-12).

Two facts drive this module.

**Paper trading time is the broker's, not the host's** (INV-12). A WSL guest clock drifts, and it jumps whenever the Windows
host sleeps. So `now()` never reads the wall clock: it is the broker timestamp of the last sync plus the elapsed
`CLOCK_BOOTTIME` since that sync. `CLOCK_MONOTONIC` would be wrong here - it is *paused* while the VM is suspended, so a
two-hour host sleep would look like no time at all and the clock would silently stay behind. `CLOCK_BOOTTIME` keeps counting,
which is exactly how a wake from sleep is detected (`needs_sync`).

**The measured skew must not be an artefact of a slow read.** `sync()` brackets `get_clock()` between two boottime readings
and compares the broker timestamp with the local UTC time at the MIDPOINT of the round trip, so a 400 ms request contributes
0 ms of apparent skew instead of 200 ms (D25: skew > 5 s blocks *opening* orders; V3: closes continue on broker time, so a
fake skew must never strand a must-exit position). The broker stamp refers to that same midpoint, so `now()` extrapolates
from it and **not** from the end of the round trip: anchoring at `t1` would leave `now()` a constant `rtt / 2` behind true
broker time - the unsafe direction next to the close - 5 min submission cut-off (11.4).

`AlpacaCalendar` turns `get_calendar()` rows into the `Calendar` protocol of 3.1. The vendor model parses `open` / `close`
as NAIVE datetimes built from the row's date and a "%H:%M" exchange-local string, so they are localised with
`ZoneInfo("America/New_York")` and converted to UTC - which is what makes a DST boundary come out right. At boot the rows are
cross-checked against XNYS: **the earlier close wins**, the session set is the **union** of the two calendars, and every
disagreement raises an alert (G4). Keeping a session only the cross-check knows is deliberate: a day this object called a
non-session would get no cycle at all, so a mandatory-exit day could silently disappear (INV-11, INV-21). All cut-offs are
offsets from that day's close (`offset_from_close`); this module contains no time of day (INV-13).
"""

import time
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, date, datetime, timedelta
from datetime import time as DayTime
from typing import Any, Final, Protocol
from zoneinfo import ZoneInfo

from jevbot.errors import BrokerError, InvariantError
from jevbot.protocols import Calendar
from jevbot.types import ClockReading

__all__ = [
    "EXCHANGE_TZ",
    "AlpacaCalendar",
    "BrokerClock",
    "CalendarRow",
    "RawClock",
    "boottime",
    "to_utc",
]

EXCHANGE_TZ: Final = ZoneInfo("America/New_York")
"""The exchange-local zone the vendor's naive calendar strings are expressed in (G4)."""

DEFAULT_RESYNC_AFTER_S: Final = 60.0
"""11.4: sync every 60 s, at every phase boundary and before every order."""

DEFAULT_BOOTTIME_GAP_S: Final = 90.0
"""11.4: a boottime gap larger than this between two readings means a host sleep - re-sync and reconcile before anything else."""

_MS: Final = 1000.0


class RawClock(Protocol):
    """The shape of `alpaca.trading.models.Clock` (verified against the pinned wheel in `alpaca_client`)."""

    @property
    def timestamp(self) -> datetime: ...
    @property
    def is_open(self) -> bool: ...
    @property
    def next_open(self) -> datetime: ...
    @property
    def next_close(self) -> datetime: ...


class CalendarRow(Protocol):
    """The shape of `alpaca.trading.models.Calendar`: a session date plus NAIVE exchange-local open / close datetimes."""

    @property
    def date(self) -> date: ...
    @property
    def open(self) -> datetime: ...
    @property
    def close(self) -> datetime: ...


def boottime() -> float:
    """Seconds on `CLOCK_BOOTTIME`: monotonic AND advancing while the machine (or the WSL VM) is suspended.

    Falls back to `CLOCK_MONOTONIC` where the constant does not exist (non-Linux); the sleep detection of 11.4 then only
    catches gaps the process was awake for, which is why the paper service is Linux-only (11.10).
    """
    return time.clock_gettime(getattr(time, "CLOCK_BOOTTIME", time.CLOCK_MONOTONIC))


def to_utc(value: object, what: str) -> datetime:
    """A vendor datetime -> tz-aware UTC. A NAIVE value is refused: silently assuming a zone is how skew bugs are born."""
    if not isinstance(value, datetime):
        raise BrokerError(f"the broker reported {what}={value!r}, which is not a datetime")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise BrokerError(f"the broker reported a naive {what} ({value.isoformat()}): the offset is needed to measure skew")
    return value.astimezone(UTC)


# ======================================================================================================================
# BrokerClock
# ======================================================================================================================


class BrokerClock:
    """`Clock` (3.1) on the broker's own clock, round-trip compensated and bridged over host sleep (11.4, INV-12).

    `fetch` returns one vendor clock reading; `paper/broker.py::AlpacaPaperBroker.raw_clock` supplies it so that the call
    carries the adapter's deadline, timeout and rate limit. `boottime_fn` / `utcnow` exist for tests only.

    **A detected wake from sleep is sticky** (`slept`). 11.4 makes a boottime gap "re-sync + reconcile before anything else",
    and sync runs every 60 s, at every phase boundary and before every order - so the call that first observes the gap is
    usually `sync()` itself, which then repairs the clock and would erase the evidence. `sync()` therefore records the gap in
    `slept`, which stays True until the runner calls `acknowledge_sleep()` after it has reconciled. `needs_sync` is
    deliberately NOT tied to it: a runner that loops "if needs_sync: sync()" must be able to make progress.
    """

    def __init__(
        self,
        fetch: Callable[[], RawClock],
        *,
        max_skew_ms: int,
        resync_after_s: float = DEFAULT_RESYNC_AFTER_S,
        boottime_gap_s: float = DEFAULT_BOOTTIME_GAP_S,
        boottime_fn: Callable[[], float] = boottime,
        utcnow: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._fetch = fetch
        self._max_skew_ms = int(max_skew_ms)
        self._resync_after_s = float(resync_after_s)
        self._boottime_gap_s = float(boottime_gap_s)
        self._boottime = boottime_fn
        self._utcnow = utcnow
        self._reading: ClockReading | None = None
        self._broker_ts: datetime | None = None
        self._boottime_at_sync: float = 0.0
        self._last_boottime: float = 0.0
        self._unsynced: bool = True
        self._slept: bool = False
        self._skew_high_since: float | None = None

    # --- state ---------------------------------------------------------------------------------------------------------

    @property
    def synced(self) -> bool:
        """False before the first sync and after a detected host sleep (11.4)."""
        return not self._unsynced and self._reading is not None

    @property
    def slept(self) -> bool:
        """True once a boottime gap larger than `boottime_gap_s` has been observed, by `now()` OR by `sync()` (11.4).

        Sticky on purpose: the consequence of a wake is "re-sync **and reconcile** before anything else", and only the runner
        can do the reconcile. It stays True until `acknowledge_sleep()`.
        """
        return self._slept

    def acknowledge_sleep(self) -> None:
        """Clear `slept`. The runner calls this once it has re-synced AND reconciled after a detected wake (11.4)."""
        self._slept = False

    @property
    def needs_sync(self) -> bool:
        """True when the clock is unsynced or the last sync is older than `resync_after_s` of boottime."""
        if self._unsynced or self._reading is None:
            return True
        return self._boottime() - self._boottime_at_sync >= self._resync_after_s

    @property
    def max_skew_ms(self) -> int:
        return self._max_skew_ms

    @property
    def skew_exceeded(self) -> bool:
        """True when the LAST reading's skew is over `health.max_clock_skew_ms` (risk check 8 blocks opening orders)."""
        return self._reading is not None and self._reading.skew_ms > self._max_skew_ms

    def skew_high_for_s(self) -> float:
        """Seconds of boottime the skew has been over the threshold without interruption; 0.0 when it is not.

        The kill switch escalates `CLOCK_SKEW` only beyond `health.clock_skew_kill_after_s` during market hours (V2, V3).
        """
        if self._skew_high_since is None:
            return 0.0
        return max(0.0, self._boottime() - self._skew_high_since)

    # --- Clock protocol ------------------------------------------------------------------------------------------------

    def now(self) -> datetime:
        """`broker_ts + (boottime_now - boottime_at_sync)`, tz-aware UTC. Never the host wall clock (INV-12).

        A boottime jump larger than `boottime_gap_s` since the previous reading marks the clock unsynced (a host sleep); the
        extrapolation is still returned, because a risk-reducing close must never be blocked by a clock problem (INV-21).
        The caller checks `needs_sync` at every phase boundary and before every order.
        """
        if self._broker_ts is None:
            raise InvariantError("BrokerClock.now() before the first sync(): boot step B5 syncs before anything reads the clock")
        current = self._boottime()
        if current - self._last_boottime > self._boottime_gap_s:
            self._unsynced = True
            self._slept = True  # 11.4: a wake means re-sync AND reconcile; `sync()` alone must not clear the second half
        self._last_boottime = current
        return self._broker_ts + timedelta(seconds=current - self._boottime_at_sync)

    def reading(self) -> ClockReading | None:
        """The last sync's reading; `None` before the first sync."""
        return self._reading

    def sync(self) -> ClockReading:
        """Force a broker re-sync. Round-trip compensated: the skew is measured against local UTC at the midpoint of the call.

        The same midpoint anchors `now()`: `broker_ts` is the broker's time at `(t0 + t1) / 2`, so extrapolating from `t1`
        would put every reading `rtt / 2` in the past (11.4).
        """
        t0 = self._boottime()
        local0 = self._utcnow()
        if self._reading is not None and t0 - self._last_boottime > self._boottime_gap_s:
            self._slept = True  # 11.4: sync() is usually the FIRST call after a wake; the gap must not die here
        raw = self._fetch()
        t1 = self._boottime()

        rtt_s = max(0.0, t1 - t0)
        broker_ts = to_utc(raw.timestamp, "clock timestamp")
        local_mid = local0 + timedelta(seconds=rtt_s / 2)
        skew_ms = round(abs((local_mid - broker_ts).total_seconds()) * _MS)

        reading = ClockReading(
            broker_ts=broker_ts,
            local_ts=local_mid,
            rtt_ms=round(rtt_s * _MS),
            skew_ms=skew_ms,
            is_open=bool(raw.is_open),
            next_open=to_utc(raw.next_open, "clock next_open"),
            next_close=to_utc(raw.next_close, "clock next_close"),
        )
        self._reading = reading
        self._broker_ts = broker_ts
        self._boottime_at_sync = t0 + rtt_s / 2  # the instant `broker_ts` refers to, NOT the end of the round trip
        self._last_boottime = t1
        self._unsynced = False
        if skew_ms > self._max_skew_ms:
            self._skew_high_since = self._boottime_at_sync if self._skew_high_since is None else self._skew_high_since
        else:
            self._skew_high_since = None
        return reading


# ======================================================================================================================
# AlpacaCalendar
# ======================================================================================================================


class _Session:
    __slots__ = ("close", "day", "open", "ordinal")

    def __init__(self, day: date, open_ts: datetime, close_ts: datetime) -> None:
        self.day = day
        self.open = open_ts
        self.close = close_ts
        self.ordinal = day.toordinal()


class AlpacaCalendar:
    """`Calendar` (3.1) over the broker's own `get_calendar()` rows, cross-checked against XNYS (11.4, G4).

    The close is the EARLIER of the vendor's and XNYS's whenever both know the session, the session set is the UNION of the
    two calendars, and every disagreement (a session only one calendar has, or a differing close) is reported through
    `alerts` and to the injected `on_alert` hook. The union is the safe direction in both senses: a day only the broker
    lists is a day the account really can trade, and a day only the exchange calendar lists still gets its cycle - so a
    mandatory-exit day (INV-11) or a hard-exit deadline can never vanish because one calendar was wrong or stale (INV-21).

    A date outside the loaded range raises `ValueError`, never a silent "not a session": beyond the horizon the true answer
    is unknown (the same contract as `cal.XnysCalendar`).
    """

    def __init__(
        self,
        rows: Iterable[CalendarRow],
        *,
        cross_check: Calendar | None = None,
        on_alert: Callable[[str], None] | None = None,
    ) -> None:
        sessions = [self._session_of_row(row) for row in rows]
        if not sessions:
            raise BrokerError("the broker returned an empty trading calendar")
        sessions.sort(key=lambda s: s.ordinal)
        days = [s.day for s in sessions]
        if len(set(days)) != len(days):
            duplicates = sorted({d.isoformat() for d in days if days.count(d) > 1})
            raise BrokerError(f"the broker calendar repeats {', '.join(duplicates)}")

        self._alerts: list[str] = []
        if cross_check is not None:
            sessions = self._apply_cross_check(sessions, cross_check)
        self._sessions: Final[tuple[_Session, ...]] = tuple(sessions)
        self._ordinals: Final[tuple[int, ...]] = tuple(s.ordinal for s in self._sessions)
        self._opens: Final[tuple[datetime, ...]] = tuple(s.open for s in self._sessions)
        self._closes: Final[tuple[datetime, ...]] = tuple(s.close for s in self._sessions)
        self._lo: Final[date] = self._sessions[0].day
        self._hi: Final[date] = self._sessions[-1].day
        self._regular_close: Final[DayTime] = self._longest_close()
        for alert in self._alerts:
            if on_alert is not None:
                on_alert(alert)

    # --- construction --------------------------------------------------------------------------------------------------

    @staticmethod
    def _session_of_row(row: CalendarRow) -> _Session:
        day, open_naive, close_naive = row.date, row.open, row.close
        if not isinstance(day, date) or isinstance(day, datetime):
            raise BrokerError(f"the broker calendar row has date={day!r}, which is not a calendar date")
        open_ts = AlpacaCalendar._localise(open_naive, day, "open")
        close_ts = AlpacaCalendar._localise(close_naive, day, "close")
        if not open_ts < close_ts:
            raise BrokerError(f"the broker calendar row for {day.isoformat()} closes at or before it opens")
        return _Session(day, open_ts, close_ts)

    @staticmethod
    def _localise(value: datetime, day: date, what: str) -> datetime:
        """The vendor parses "<date> <HH:MM>" into a NAIVE exchange-local datetime; attach America/New_York, then go UTC."""
        if not isinstance(value, datetime):
            raise BrokerError(f"the broker calendar row for {day.isoformat()} has {what}={value!r}, which is not a datetime")
        if value.tzinfo is not None and value.tzinfo.utcoffset(value) is not None:
            return value.astimezone(UTC)
        return value.replace(tzinfo=EXCHANGE_TZ).astimezone(UTC)

    def _apply_cross_check(self, sessions: list[_Session], other: Calendar) -> list[_Session]:
        """Earlier close wins, the session set is the union, every disagreement is an alert (G4)."""
        checked: list[_Session] = []
        for session in sessions:
            try:
                theirs_open, theirs_close = other.open_close(session.day)
            except ValueError:
                self._alerts.append(
                    f"calendar disagreement on {session.day.isoformat()}: the broker lists a session the cross-check does not"
                )
                checked.append(session)
                continue
            if theirs_close != session.close:
                winner = min(theirs_close, session.close)
                self._alerts.append(
                    f"calendar disagreement on {session.day.isoformat()}: broker close {session.close.isoformat()} vs cross-check "
                    f"{theirs_close.isoformat()}; the earlier close {winner.isoformat()} wins"
                )
                session = _Session(session.day, session.open, winner)
            if theirs_open != session.open:
                self._alerts.append(
                    f"calendar disagreement on {session.day.isoformat()}: broker open {session.open.isoformat()} vs cross-check "
                    f"{theirs_open.isoformat()}"
                )
            checked.append(session)
        known = {s.day for s in sessions}
        for missing in self._cross_check_only(other, known):
            self._alerts.append(
                f"calendar disagreement on {missing.isoformat()}: the cross-check lists a session the broker does not; "
                "it is KEPT as a session, because a day we call a non-session gets no cycle at all (INV-11, INV-21)"
            )
            checked.append(self._session_from_cross_check(other, missing))
        checked.sort(key=lambda s: s.ordinal)
        return checked

    @staticmethod
    def _session_from_cross_check(other: Calendar, day: date) -> _Session:
        """A session only the cross-check knows, on the cross-check's own (tz-aware UTC) open and close."""
        try:
            open_ts, close_ts = other.open_close(day)
        except ValueError as exc:
            raise BrokerError(f"the cross-check calendar lists {day.isoformat()} as a session but cannot give its hours: {exc}") from exc
        for what, value in (("open", open_ts), ("close", close_ts)):
            if not isinstance(value, datetime) or value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
                raise BrokerError(f"the cross-check calendar returned a naive {what} for {day.isoformat()}")
        if not open_ts < close_ts:
            raise BrokerError(f"the cross-check calendar has {day.isoformat()} closing at or before it opens")
        return _Session(day, open_ts.astimezone(UTC), close_ts.astimezone(UTC))

    @staticmethod
    def _cross_check_only(other: Calendar, known: set[date]) -> list[date]:
        if not known:
            return []
        try:
            theirs = other.sessions(min(known), max(known))
        except ValueError:
            return []
        return [d for d in theirs if d not in known]

    def _longest_close(self) -> DayTime:
        """The exchange-local time of day of the LONGEST session in the window: the regular close (no literal, INV-13).

        A full day is always the longest session of a window, so anything closing earlier is a half day. The most COMMON
        close would invert on a window that happens to hold more early closes than regular ones (a two-row December window
        around the 24th is enough), and would report genuine half days as ordinary sessions.
        """
        return max(_local_close(s.close) for s in self._sessions)

    # --- helpers -------------------------------------------------------------------------------------------------------

    @property
    def alerts(self) -> tuple[str, ...]:
        """Every cross-check disagreement found at construction (boot step B5 logs and alerts on them)."""
        return tuple(self._alerts)

    @property
    def bounds(self) -> tuple[date, date]:
        """(first session, last session) this calendar can answer for, inclusive."""
        return (self._lo, self._hi)

    def _ordinal(self, name: str, d: date) -> int:
        if not isinstance(d, date) or isinstance(d, datetime):
            raise ValueError(f"{name} must be a datetime.date, got {type(d).__name__}")
        if not self._lo <= d <= self._hi:
            raise ValueError(f"{name} {d.isoformat()} is outside the broker calendar {self._lo.isoformat()}..{self._hi.isoformat()}")
        return d.toordinal()

    def _index(self, session: date) -> int:
        ordinal = self._ordinal("session", session)
        i = bisect_left(self._ordinals, ordinal)
        if i == len(self._ordinals) or self._ordinals[i] != ordinal:
            raise ValueError(f"{session.isoformat()} is not a session of the broker calendar")
        return i

    def _at(self, i: int, what: str) -> date:
        if not 0 <= i < len(self._sessions):
            raise ValueError(f"{what} lies outside the broker calendar {self._lo.isoformat()}..{self._hi.isoformat()}")
        return self._sessions[i].day

    # --- Calendar protocol ---------------------------------------------------------------------------------------------

    def is_session(self, d: date) -> bool:
        ordinal = self._ordinal("d", d)
        i = bisect_left(self._ordinals, ordinal)
        return i < len(self._ordinals) and self._ordinals[i] == ordinal

    def sessions(self, start: date, end: date) -> list[date]:
        """Every session in [start, end], inclusive."""
        lo = bisect_left(self._ordinals, self._ordinal("start", start))
        hi = bisect_right(self._ordinals, self._ordinal("end", end))
        return [s.day for s in self._sessions[lo:hi]]

    def open_close(self, session: date) -> tuple[datetime, datetime]:
        """(open, close) of a session, UTC; early closes and the earlier-close-wins cross-check are already applied."""
        s = self._sessions[self._index(session)]
        return (s.open, s.close)

    def is_early_close(self, session: date) -> bool:
        """True when the session closes before the calendar's regular exchange-local close (a half day)."""
        return _local_close(self._sessions[self._index(session)].close) < self._regular_close

    def next_session(self, d: date, n: int = 1) -> date:
        """The n-th session strictly after `d` (any calendar date)."""
        i = bisect_right(self._ordinals, self._ordinal("d", d)) + _as_count("n", n) - 1
        return self._at(i, f"session {n} after {d.isoformat()}")

    def prev_session(self, d: date, n: int = 1) -> date:
        """The n-th session strictly before `d` (any calendar date)."""
        i = bisect_left(self._ordinals, self._ordinal("d", d)) - _as_count("n", n)
        return self._at(i, f"session {n} before {d.isoformat()}")

    def prev_or_same_session(self, d: date) -> date:
        """THE map expiry -> last trading day (Conventions, INV-11): `d` itself when it is a session, else the one before."""
        return d if self.is_session(d) else self.prev_session(d)

    def sessions_between(self, a: date, b: date) -> int:
        """The number of sessions in (a, b]; 0 when b <= a (the `cal.XnysCalendar` contract, 3.1)."""
        lo = bisect_right(self._ordinals, self._ordinal("a", a))
        hi = bisect_right(self._ordinals, self._ordinal("b", b))
        return max(0, hi - lo)

    def session_of(self, ts: datetime) -> date | None:
        """The session whose [open, close] contains `ts` (both ends inclusive), else `None`.

        Unlike the date lookups this accepts an instant outside the loaded window and answers `None`: a rolling broker
        calendar is fetched per run, and "before the first session I know about" is not a programming error.
        """
        moment = _as_utc("ts", ts)
        i = bisect_left(self._closes, moment)  # first session that closes at or after ts
        if i < len(self._closes) and self._opens[i] <= moment:
            return self._sessions[i].day
        return None

    def offset_from_close(self, session: date, minutes_before: int) -> datetime:
        """THE way to express a cut-off (INV-13): `close - minutes_before`; a negative value lands after the close."""
        _, close = self.open_close(session)
        return close - timedelta(minutes=minutes_before)

    def next_open_after(self, ts: datetime) -> datetime:
        """The first session open strictly after `ts` (the `knowable_at` of an EOD series)."""
        moment = _as_utc("ts", ts)
        i = bisect_right(self._opens, moment)
        if i >= len(self._opens):
            raise ValueError(f"no session opens after {moment.isoformat()} inside the broker calendar")
        return self._opens[i]


def _as_count(name: str, n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"{name} must be an int >= 1, got {n!r}")
    return n


def _as_utc(name: str, ts: datetime) -> datetime:
    if not isinstance(ts, datetime) or ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        raise ValueError(f"{name} must be a tz-aware datetime, got {ts!r}")
    return ts.astimezone(UTC)


def _local_close(close: datetime) -> DayTime:
    """The exchange-local time of day a session closes at, as a naive `time` (comparable across DST)."""
    return close.astimezone(EXCHANGE_TZ).timetz().replace(tzinfo=None)


def calendar_rows(client: Any, start: date, end: date) -> Sequence[CalendarRow]:
    """`get_calendar(GetCalendarRequest(start, end))` on a trading client, as the raw vendor rows.

    Kept here so that `AlpacaCalendar` itself stays a pure function of its rows and can be built from a recorded day.
    """
    from alpaca.trading.requests import GetCalendarRequest

    rows = client.get_calendar(GetCalendarRequest(start=start, end=end))
    return list(rows)
