"""cal.py (DESIGN.md 3.1, 3.7, Conventions): XnysCalendar, SimClock, year_fraction, trading_time, last_session.

Expected dates and times are calendar facts stated by hand (NYSE holidays, early closes, daylight saving), not values read
back from the code under test.
"""

import inspect
import re
import subprocess
import sys
import warnings
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from jevbot import cal
from jevbot.cal import SimClock, XnysCalendar, last_session, trading_time, year_fraction
from jevbot.errors import InvariantError

REPO = Path(__file__).resolve().parents[2]
DESIGN_MD = REPO / "docs" / "design" / "DESIGN.md"
NY = ZoneInfo("America/New_York")


def utc(y: int, m: int, d: int, hh: int = 0, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=UTC)


@pytest.fixture(scope="module")
def xnys() -> XnysCalendar:
    return XnysCalendar()


# ======================================================================================================================
# sessions, opens and closes
# ======================================================================================================================


def test_regular_session_times_follow_daylight_saving(xnys: XnysCalendar) -> None:
    # 09:30-16:00 New York = 13:30-20:00 UTC in summer (EDT), 14:30-21:00 UTC in winter (EST)
    assert xnys.open_close(date(2026, 9, 15)) == (utc(2026, 9, 15, 13, 30), utc(2026, 9, 15, 20, 0))
    assert xnys.open_close(date(2026, 1, 15)) == (utc(2026, 1, 15, 14, 30), utc(2026, 1, 15, 21, 0))
    for session in (date(2026, 9, 15), date(2026, 1, 15), date(2014, 4, 17), date(2012, 3, 16)):
        opened, closed = xnys.open_close(session)
        assert opened.tzinfo is not None and opened.utcoffset() == timedelta(0) and closed.utcoffset() == timedelta(0)
        assert (opened.astimezone(NY).hour, opened.astimezone(NY).minute) == (9, 30)
        assert (closed.astimezone(NY).hour, closed.astimezone(NY).minute) == (16, 0)
        assert closed - opened == timedelta(minutes=390)
        assert not xnys.is_early_close(session)


@pytest.mark.parametrize("session", [date(2026, 11, 27), date(2026, 12, 24)])
def test_early_closes_close_at_18_00_utc(xnys: XnysCalendar, session: date) -> None:
    # the day after Thanksgiving and Christmas Eve 2026: 13:00 New York (EST) = 18:00 UTC (D4, G4)
    assert xnys.is_session(session)
    assert xnys.is_early_close(session)
    opened, closed = xnys.open_close(session)
    assert closed == datetime(session.year, session.month, session.day, 18, 0, tzinfo=UTC)
    assert opened == datetime(session.year, session.month, session.day, 14, 30, tzinfo=UTC)
    assert closed - opened == timedelta(minutes=210)


def test_other_early_closes_and_neighbours(xnys: XnysCalendar) -> None:
    assert xnys.is_early_close(date(2014, 7, 3)) and xnys.open_close(date(2014, 7, 3))[1] == utc(2014, 7, 3, 17, 0)  # EDT
    assert xnys.is_early_close(date(2025, 11, 28)) and xnys.is_early_close(date(2025, 12, 24))
    assert not xnys.is_early_close(date(2026, 11, 25)) and not xnys.is_early_close(date(2026, 12, 23))
    assert not xnys.is_session(date(2026, 11, 26))  # Thanksgiving
    assert not xnys.is_session(date(2026, 12, 25))


def test_holidays_and_weekends(xnys: XnysCalendar) -> None:
    closed_days = [
        date(2026, 9, 19),  # Saturday
        date(2026, 9, 20),  # Sunday
        date(2026, 1, 1),  # New Year's Day
        date(2026, 1, 19),  # Martin Luther King Jr. Day
        date(2026, 4, 3),  # Good Friday
        date(2026, 6, 19),  # Juneteenth
        date(2026, 7, 3),  # Independence Day observed (4 July is a Saturday)
        date(2026, 9, 7),  # Labor Day
        date(2014, 4, 18),  # Good Friday 2014
        date(2012, 10, 29),  # hurricane Sandy: unscheduled closes are history too
        date(2012, 10, 30),
        date(2001, 9, 11),
    ]
    for d in closed_days:
        assert not xnys.is_session(d), d
    for d in (date(2026, 9, 17), date(2026, 9, 18), date(2014, 4, 17), date(2012, 3, 16), date(2012, 10, 31)):
        assert xnys.is_session(d), d


def test_sessions_range_is_inclusive(xnys: XnysCalendar) -> None:
    week = [date(2026, 9, 14), date(2026, 9, 15), date(2026, 9, 16), date(2026, 9, 17), date(2026, 9, 18)]
    assert xnys.sessions(date(2026, 9, 14), date(2026, 9, 18)) == week
    assert xnys.sessions(date(2026, 9, 12), date(2026, 9, 20)) == week  # weekend bounds
    assert xnys.sessions(date(2026, 9, 16), date(2026, 9, 16)) == [date(2026, 9, 16)]
    assert xnys.sessions(date(2026, 9, 19), date(2026, 9, 20)) == []
    assert xnys.sessions(date(2026, 9, 18), date(2026, 9, 14)) == []
    assert xnys.sessions(date(2014, 4, 14), date(2014, 4, 21)) == [
        date(2014, 4, 14),
        date(2014, 4, 15),
        date(2014, 4, 16),
        date(2014, 4, 17),
        date(2014, 4, 21),
    ]  # the Good-Friday week has four sessions
    assert len(xnys.sessions(date(2025, 1, 1), date(2025, 12, 31))) == 250  # 9 Jan 2025 (day of mourning) was closed too
    got = xnys.sessions(date(2026, 9, 14), date(2026, 9, 18))
    got.clear()  # the returned list is the caller's: mutating it cannot corrupt the calendar
    assert xnys.sessions(date(2026, 9, 14), date(2026, 9, 18)) == week


def test_next_and_prev_session(xnys: XnysCalendar) -> None:
    friday, saturday, monday = date(2026, 9, 18), date(2026, 9, 19), date(2026, 9, 21)
    assert xnys.next_session(friday) == monday  # strictly after
    assert xnys.next_session(saturday) == monday
    assert xnys.next_session(monday) == date(2026, 9, 22)
    assert xnys.prev_session(monday) == friday  # strictly before
    assert xnys.prev_session(saturday) == friday
    assert xnys.prev_session(friday) == date(2026, 9, 17)
    assert xnys.next_session(friday, 5) == date(2026, 9, 25)
    assert xnys.next_session(date(2026, 11, 25), 2) == date(2026, 11, 30)  # over Thanksgiving, the half day and a weekend
    assert xnys.prev_session(date(2026, 11, 30), 2) == date(2026, 11, 25)
    for n in (0, -1):
        with pytest.raises(ValueError):
            xnys.next_session(friday, n)
        with pytest.raises(ValueError):
            xnys.prev_session(friday, n)
    for bad in (1.0, True, "1"):
        with pytest.raises(TypeError):
            xnys.next_session(friday, bad)  # type: ignore[arg-type]


def test_prev_session_from_an_expiry(xnys: XnysCalendar) -> None:
    # `prev_session(expiry, n)`: the hard-exit arithmetic counts back from the expiry, which need not be a session
    saturday_expiry = date(2012, 3, 17)
    assert xnys.prev_session(saturday_expiry, 1) == date(2012, 3, 16)
    assert xnys.prev_session(saturday_expiry, 3) == date(2012, 3, 14)
    assert xnys.prev_session(saturday_expiry, 4) == date(2012, 3, 13)
    good_friday_week_expiry = date(2014, 4, 19)
    assert xnys.prev_session(good_friday_week_expiry, 1) == date(2014, 4, 17)
    assert xnys.prev_session(good_friday_week_expiry, 4) == date(2014, 4, 14)
    friday_expiry = date(2026, 10, 16)
    assert xnys.prev_session(friday_expiry, 1) == date(2026, 10, 15)  # strictly before, even for a session
    assert xnys.prev_session(friday_expiry, 3) == date(2026, 10, 13)


def test_prev_or_same_session_maps_expiry_to_last_session(xnys: XnysCalendar) -> None:
    # a Saturday-dated monthly (listed before February 2015): the last trading day is the Friday
    assert xnys.prev_or_same_session(date(2012, 3, 17)) == date(2012, 3, 16)
    # a Good-Friday week: Saturday 2014-04-19, Friday 04-18 is a holiday -> Thursday 04-17
    assert xnys.prev_or_same_session(date(2014, 4, 19)) == date(2014, 4, 17)
    assert xnys.prev_or_same_session(date(2014, 4, 18)) == date(2014, 4, 17)  # a holiday-dated expiry
    # a session date maps to itself
    assert xnys.prev_or_same_session(date(2026, 10, 16)) == date(2026, 10, 16)
    assert xnys.prev_or_same_session(date(2014, 4, 17)) == date(2014, 4, 17)
    assert xnys.prev_or_same_session(date(2026, 9, 20)) == date(2026, 9, 18)  # Sunday
    # the convenience wrapper of the Conventions
    assert last_session(xnys, date(2012, 3, 17)) == date(2012, 3, 16)
    assert last_session(xnys, date(2014, 4, 19)) == date(2014, 4, 17)
    assert last_session(xnys, date(2026, 10, 16)) == date(2026, 10, 16)
    for d in xnys.sessions(date(2014, 4, 1), date(2014, 7, 3)):
        assert xnys.prev_or_same_session(d) == d


def test_the_mini_mirror_window_facts_of_section_15_2(xnys: XnysCalendar) -> None:
    window = xnys.sessions(date(2014, 4, 9), date(2014, 7, 3))
    assert len(window) == 60 and window[0] == date(2014, 4, 9) and window[-1] == date(2014, 7, 3)
    assert [xnys.prev_or_same_session(d) for d in (date(2014, 4, 19), date(2014, 5, 17), date(2014, 6, 21))] == [
        date(2014, 4, 17),
        date(2014, 5, 16),
        date(2014, 6, 20),
    ]
    assert xnys.is_session(date(2014, 4, 30)) and xnys.is_session(date(2014, 6, 18))  # the FOMC decision days
    assert [d for d in window if xnys.is_early_close(d)] == [date(2014, 7, 3)]


def test_sessions_between_counts_the_half_open_interval(xnys: XnysCalendar) -> None:
    monday, friday = date(2026, 9, 14), date(2026, 9, 18)
    assert xnys.sessions_between(monday, friday) == 4  # (a, b]: Tue, Wed, Thu, Fri
    assert xnys.sessions_between(monday, monday) == 0
    assert xnys.sessions_between(friday, date(2026, 9, 21)) == 1  # across the weekend
    assert xnys.sessions_between(friday, date(2026, 9, 20)) == 0  # Friday -> Sunday: nothing in between
    assert xnys.sessions_between(date(2026, 9, 19), date(2026, 9, 21)) == 1  # a non-session start
    assert xnys.sessions_between(friday, monday) == 0  # b <= a
    # sessions_to_expiry, measured to last_session (V7): decision day 2012-03-13 -> Friday 2012-03-16 = 3
    assert xnys.sessions_between(date(2012, 3, 13), xnys.prev_or_same_session(date(2012, 3, 17))) == 3
    assert xnys.sessions_between(date(2014, 4, 14), xnys.prev_or_same_session(date(2014, 4, 19))) == 3
    # consistent with next_session: the n-th next session is exactly n sessions away
    for n in (1, 2, 5, 20):
        assert xnys.sessions_between(monday, xnys.next_session(monday, n)) == n
    assert xnys.sessions_between(date(2026, 1, 1), date(2026, 12, 31)) == len(xnys.sessions(date(2026, 1, 2), date(2026, 12, 31)))


def test_session_of(xnys: XnysCalendar) -> None:
    d = date(2026, 9, 15)
    opened, closed = xnys.open_close(d)
    assert xnys.session_of(opened) == d and xnys.session_of(closed) == d  # [open, close], both ends inclusive
    assert xnys.session_of(opened + timedelta(hours=3)) == d
    assert xnys.session_of(opened - timedelta(seconds=1)) is None
    assert xnys.session_of(closed + timedelta(seconds=1)) is None
    assert xnys.session_of(utc(2026, 9, 19, 15, 0)) is None  # Saturday
    assert xnys.session_of(utc(2026, 11, 26, 16, 0)) is None  # Thanksgiving
    # an early close ends the session early
    assert xnys.session_of(utc(2026, 11, 27, 18, 0)) == date(2026, 11, 27)
    assert xnys.session_of(utc(2026, 11, 27, 18, 1)) is None
    # any tz-aware datetime is accepted and converted
    assert xnys.session_of(datetime(2026, 9, 15, 10, 0, tzinfo=NY)) == d
    assert xnys.session_of(datetime(2026, 9, 15, 23, 0, tzinfo=timezone(timedelta(hours=9)))) == d  # 14:00 UTC


def test_offset_from_close_on_normal_and_early_days(xnys: XnysCalendar) -> None:
    # INV-13: every cut-off is an offset from THAT day's close, so it moves with an early close
    assert xnys.offset_from_close(date(2026, 9, 15), 25) == utc(2026, 9, 15, 19, 35)
    assert xnys.offset_from_close(date(2026, 9, 15), 0) == utc(2026, 9, 15, 20, 0)
    assert xnys.offset_from_close(date(2026, 9, 15), -2) == utc(2026, 9, 15, 20, 2)  # negative = after the close
    assert xnys.offset_from_close(date(2026, 1, 15), 25) == utc(2026, 1, 15, 20, 35)  # winter time
    assert xnys.offset_from_close(date(2026, 11, 27), 25) == utc(2026, 11, 27, 17, 35)  # 13:00 close: exactly 3 h earlier
    assert xnys.offset_from_close(date(2026, 11, 27), -2) == utc(2026, 11, 27, 18, 2)
    assert xnys.offset_from_close(date(2026, 11, 25), 25) - xnys.offset_from_close(date(2026, 11, 27), 25) == timedelta(days=-2, hours=3)
    assert xnys.offset_from_close(date(2026, 9, 15), 390) == xnys.open_close(date(2026, 9, 15))[0]
    with pytest.raises(ValueError):
        xnys.offset_from_close(date(2026, 9, 19), 25)  # not a session
    with pytest.raises(TypeError):
        xnys.offset_from_close(date(2026, 9, 15), 2.5)  # type: ignore[arg-type]


def test_next_open_after(xnys: XnysCalendar) -> None:
    tuesday_open, tuesday_close = xnys.open_close(date(2026, 9, 15))
    wednesday_open = xnys.open_close(date(2026, 9, 16))[0]
    assert xnys.next_open_after(tuesday_close) == wednesday_open  # knowable_at of an EOD value dated Tuesday
    assert xnys.next_open_after(tuesday_open) == wednesday_open  # strictly after
    assert xnys.next_open_after(tuesday_open - timedelta(seconds=1)) == tuesday_open
    friday_close = xnys.open_close(date(2026, 9, 18))[1]
    assert xnys.next_open_after(friday_close) == utc(2026, 9, 21, 13, 30)  # over the weekend
    assert xnys.next_open_after(utc(2026, 11, 25, 21, 0)) == utc(2026, 11, 27, 14, 30)  # over Thanksgiving


def test_inputs_are_validated(xnys: XnysCalendar) -> None:
    stamp = utc(2026, 9, 15, 15, 0)
    for call in (
        lambda: xnys.is_session(stamp),  # a datetime is not a session date (its UTC date may differ from the exchange date)
        lambda: xnys.open_close(stamp),
        lambda: xnys.next_session(stamp),
        lambda: xnys.prev_or_same_session(stamp),
        lambda: xnys.sessions(stamp, stamp),
        lambda: xnys.sessions_between(date(2026, 9, 15), stamp),
        lambda: xnys.is_session("2026-09-15"),  # type: ignore[arg-type]
        lambda: xnys.session_of(date(2026, 9, 15)),  # type: ignore[arg-type]
    ):
        with pytest.raises(TypeError):
            call()
    naive = datetime(2026, 9, 15, 15, 0)  # noqa: DTZ001 - the point of the test
    with pytest.raises(ValueError, match="tz-aware"):
        xnys.session_of(naive)
    with pytest.raises(ValueError, match="tz-aware"):
        xnys.next_open_after(naive)
    for call in (lambda: xnys.open_close(date(2026, 9, 19)), lambda: xnys.is_early_close(date(2026, 11, 26))):
        with pytest.raises(ValueError, match="not an XNYS session"):
            call()


def test_dates_beyond_the_horizon_raise_instead_of_answering(xnys: XnysCalendar) -> None:
    lo, hi = xnys.bounds
    assert lo <= date(1990, 1, 2) and hi >= date(2035, 12, 31)  # the Cboe history start and long-dated listings are inside
    for call in (
        lambda: xnys.is_session(lo - timedelta(days=1)),
        lambda: xnys.is_session(hi + timedelta(days=1)),
        lambda: xnys.prev_or_same_session(hi + timedelta(days=30)),  # never "the last session we happen to know"
        lambda: xnys.next_session(hi),
        lambda: xnys.prev_session(lo),
        lambda: xnys.sessions(lo - timedelta(days=5), lo + timedelta(days=5)),
        lambda: xnys.sessions_between(hi - timedelta(days=5), hi + timedelta(days=5)),
        lambda: xnys.session_of(datetime(hi.year + 1, 6, 1, 15, 0, tzinfo=UTC)),
        lambda: xnys.next_open_after(datetime(hi.year, 12, 31, 23, 0, tzinfo=UTC)),
    ):
        with pytest.raises(ValueError):
            call()


def test_custom_bounds_and_shared_tables() -> None:
    small = XnysCalendar(date(2026, 1, 1), date(2026, 12, 31))
    assert small.bounds == (date(2026, 1, 1), date(2026, 12, 31))
    full = XnysCalendar()
    assert small.sessions(date(2026, 1, 1), date(2026, 12, 31)) == full.sessions(date(2026, 1, 1), date(2026, 12, 31))
    assert small.open_close(date(2026, 11, 27)) == full.open_close(date(2026, 11, 27))
    with pytest.raises(ValueError):
        small.is_session(date(2027, 1, 4))
    with pytest.raises(ValueError):
        XnysCalendar(date(2026, 6, 1), date(2026, 1, 1))
    with pytest.raises(TypeError):
        XnysCalendar(utc(2026, 1, 1), date(2026, 12, 31))
    assert XnysCalendar().sessions(date(2026, 9, 14), date(2026, 9, 18)) == full.sessions(date(2026, 9, 14), date(2026, 9, 18))


def test_the_calendar_builds_with_warnings_promoted_to_errors() -> None:
    # the calendar library emits pandas / numpy deprecation noise at import and while building; cal.py contains it
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fresh = XnysCalendar(date(2019, 1, 1), date(2021, 12, 31))  # bounds no other test uses: a real build, not a cache hit
    assert fresh.prev_or_same_session(date(2020, 7, 4)) == date(2020, 7, 2)  # 3 July 2020 was the observed holiday
    code = "from datetime import date; from jevbot.cal import XnysCalendar; print(XnysCalendar().is_session(date(2026, 9, 17)))"
    done = subprocess.run([sys.executable, "-W", "error", "-c", code], capture_output=True, text=True, cwd=REPO, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "True"


def test_no_time_of_day_literal_in_the_module() -> None:
    # INV-13 / tests/guards/test_no_clock_literals.py greps src/ for these; cal.py is where they would be most tempting
    source = Path(inspect.getsourcefile(cal) or "").read_text(encoding="utf-8")
    for needle in ("15:30", "15:45", "16:00", "16:15", "time(15", "time(16", "hour=15", "hour=16"):
        assert needle not in source


# ======================================================================================================================
# the Protocols of section 3.1
# ======================================================================================================================


def _protocol_methods(name: str) -> dict[str, list[str]]:
    """{method: [parameter names]} of `class <name>(Protocol)` in the fenced contract block of DESIGN.md section 3.1."""
    text = DESIGN_MD.read_text(encoding="utf-8")
    section = text[text.index("### 3.1 Time") : text.index("### 3.2 Data")]
    body = section[section.index(f"class {name}(Protocol):") :]
    methods: dict[str, list[str]] = {}
    for line in body.splitlines()[1:]:
        if not line.startswith("    "):
            break
        m = re.match(r"\s+def (\w+)\(self(?:, )?([^)]*)\)", line)
        if m is not None:
            methods[m.group(1)] = [p.split(":")[0].strip() for p in m.group(2).split(",") if p.strip()]
    return methods


def test_xnys_calendar_implements_every_calendar_protocol_method() -> None:
    methods = _protocol_methods("Calendar")
    assert set(methods) == {
        "is_session",
        "sessions",
        "open_close",
        "is_early_close",
        "next_session",
        "prev_session",
        "prev_or_same_session",
        "sessions_between",
        "session_of",
        "offset_from_close",
        "next_open_after",
    }
    for name, params in methods.items():
        signature = inspect.signature(getattr(XnysCalendar, name))
        assert list(signature.parameters)[1:] == params, name
    assert inspect.signature(XnysCalendar.next_session).parameters["n"].default == 1
    assert inspect.signature(XnysCalendar.prev_session).parameters["n"].default == 1


def test_sim_clock_implements_the_clock_protocol() -> None:
    methods = _protocol_methods("Clock")
    assert set(methods) == {"now", "reading", "sync"}
    for name, params in methods.items():
        assert list(inspect.signature(getattr(SimClock, name)).parameters)[1:] == params, name


# ======================================================================================================================
# SimClock
# ======================================================================================================================


def test_sim_clock() -> None:
    clock = SimClock()
    with pytest.raises(InvariantError):
        clock.now()  # a backtest must never read an unset clock
    close = utc(2026, 9, 15, 20, 0)
    clock.set(close)
    assert clock.now() == close and clock.now().utcoffset() == timedelta(0)
    assert clock.reading() is None and clock.sync() is None  # no broker clock in a simulation
    assert clock.now() == close  # sync() is a no-op
    clock.set(datetime(2026, 9, 16, 16, 0, tzinfo=NY))  # any aware datetime; normalised to UTC
    assert clock.now() == utc(2026, 9, 16, 20, 0) and clock.now().tzinfo is UTC
    clock.set(close)  # the loop owns the clock: a replay may set it back
    assert clock.now() == close
    assert SimClock(close).now() == close
    with pytest.raises(ValueError, match="tz-aware"):
        clock.set(datetime(2026, 9, 15, 20, 0))  # noqa: DTZ001
    with pytest.raises(TypeError):
        clock.set(date(2026, 9, 15))  # type: ignore[arg-type]
    assert clock.now() == close  # a refused set() changes nothing


# ======================================================================================================================
# year_fraction: THE calendar day-count
# ======================================================================================================================


def test_year_fraction() -> None:
    a = utc(2026, 9, 15, 20, 0)
    assert year_fraction(a, a + timedelta(days=365)) == 1.0
    assert year_fraction(a, a + timedelta(days=30)) == pytest.approx(30 / 365, abs=1e-15)
    assert year_fraction(a, a + timedelta(days=1)) == pytest.approx(1 / 365, abs=1e-15)
    assert year_fraction(a, a + timedelta(minutes=25)) == pytest.approx(25 * 60 / (365 * 86400), abs=1e-18)
    assert year_fraction(a, a) == 0.0
    assert year_fraction(a + timedelta(days=73), a) == pytest.approx(-0.2, abs=1e-15)  # signed
    assert year_fraction(utc(2024, 1, 1), utc(2025, 1, 1)) == pytest.approx(366 / 365, abs=1e-15)  # ACT/365: a leap year is > 1
    # time zones do not matter, instants do
    assert year_fraction(datetime(2026, 9, 15, 16, 0, tzinfo=NY), a + timedelta(days=365)) == 1.0
    with pytest.raises(ValueError, match="tz-aware"):
        year_fraction(datetime(2026, 9, 15, 20, 0), a)  # noqa: DTZ001
    with pytest.raises(TypeError):
        year_fraction(date(2026, 9, 15), a)  # type: ignore[arg-type]


def test_t_e_runs_to_the_close_of_last_session_not_of_the_listed_expiry(xnys: XnysCalendar) -> None:
    # Conventions: T_E = year_fraction(ts, close(last_session(E))); a Saturday-dated monthly has no close of its own
    ts = xnys.open_close(date(2012, 3, 7))[1]
    expiry = date(2012, 3, 17)
    with pytest.raises(ValueError):
        xnys.open_close(expiry)
    t_e = year_fraction(ts, xnys.open_close(last_session(xnys, expiry))[1])
    # 7 March (EST, 21:00 UTC close) -> Friday 16 March (EDT since 11 March, 20:00 UTC close): nine days less one hour
    assert t_e == pytest.approx((9 * 86400 - 3600) / (365 * 86400), abs=1e-15)


# ======================================================================================================================
# trading_time: THE variance clock (V13)
# ======================================================================================================================


def _close(c: XnysCalendar, d: date) -> datetime:
    return c.open_close(d)[1]


def test_trading_time_close_to_close_is_one_session_on_a_tuesday_and_across_a_weekend(xnys: XnysCalendar) -> None:
    assert trading_time(xnys, _close(xnys, date(2026, 9, 15)), _close(xnys, date(2026, 9, 16))) == 1.0  # Tuesday -> Wednesday
    assert trading_time(xnys, _close(xnys, date(2026, 9, 18)), _close(xnys, date(2026, 9, 21))) == 1.0  # Friday -> Monday
    assert trading_time(xnys, _close(xnys, date(2026, 11, 25)), _close(xnys, date(2026, 11, 30))) == 2.0  # holiday + half day + weekend
    assert trading_time(xnys, _close(xnys, date(2014, 4, 17)), _close(xnys, date(2014, 4, 21))) == 1.0  # over Good Friday
    # calendar time, by contrast, is three times longer over the weekend - which is exactly what V13 refuses to allocate variance by
    weekday = year_fraction(_close(xnys, date(2026, 9, 15)), _close(xnys, date(2026, 9, 16)))
    weekend = year_fraction(_close(xnys, date(2026, 9, 18)), _close(xnys, date(2026, 9, 21)))
    assert weekend == pytest.approx(3 * weekday)


def test_trading_time_from_the_decision_time_to_the_next_close(xnys: XnysCalendar) -> None:
    decision = xnys.offset_from_close(date(2026, 9, 15), 25)
    assert trading_time(xnys, decision, _close(xnys, date(2026, 9, 16))) == pytest.approx(1 + 25 / 390, abs=1e-15)
    assert trading_time(xnys, decision, _close(xnys, date(2026, 9, 15))) == pytest.approx(25 / 390, abs=1e-15)
    friday_decision = xnys.offset_from_close(date(2026, 9, 18), 25)
    assert trading_time(xnys, friday_decision, _close(xnys, date(2026, 9, 21))) == pytest.approx(1 + 25 / 390, abs=1e-15)
    assert trading_time(xnys, friday_decision, _close(xnys, date(2026, 9, 25))) == pytest.approx(5 + 25 / 390, abs=1e-15)


def test_trading_time_early_close_session_counts_one(xnys: XnysCalendar) -> None:
    half_day = date(2026, 11, 27)
    assert xnys.is_early_close(half_day)
    assert trading_time(xnys, _close(xnys, date(2026, 11, 25)), _close(xnys, half_day)) == 1.0  # a whole early-close session = 1.0
    opened, closed = xnys.open_close(half_day)
    assert trading_time(xnys, opened, closed) == 1.0
    # a PARTIAL early-close session counts elapsed minutes over ITS OWN length (210 minutes)
    assert trading_time(xnys, closed - timedelta(minutes=25), closed) == pytest.approx(25 / 210, abs=1e-15)
    assert trading_time(xnys, xnys.offset_from_close(half_day, 25), _close(xnys, date(2026, 11, 30))) == pytest.approx(
        1 + 25 / 210, abs=1e-15
    )
    assert trading_time(xnys, opened, opened + timedelta(minutes=105)) == pytest.approx(0.5, abs=1e-15)


def test_trading_time_is_zero_when_b_is_not_after_a(xnys: XnysCalendar) -> None:
    a = _close(xnys, date(2026, 9, 15))
    assert trading_time(xnys, a, a) == 0.0
    assert trading_time(xnys, a, a - timedelta(days=3)) == 0.0
    assert trading_time(xnys, _close(xnys, date(2026, 9, 21)), _close(xnys, date(2026, 9, 18))) == 0.0


def test_trading_time_outside_sessions_is_zero_and_intraday_is_proportional(xnys: XnysCalendar) -> None:
    friday_close = _close(xnys, date(2026, 9, 18))
    monday_open = xnys.open_close(date(2026, 9, 21))[0]
    assert trading_time(xnys, friday_close, monday_open) == 0.0  # a whole weekend, no trading time
    assert trading_time(xnys, friday_close + timedelta(hours=1), monday_open - timedelta(hours=1)) == 0.0
    assert trading_time(xnys, utc(2026, 11, 26, 0, 0), utc(2026, 11, 27, 0, 0)) == 0.0  # Thanksgiving
    opened, closed = xnys.open_close(date(2026, 9, 15))
    assert trading_time(xnys, opened + timedelta(minutes=60), opened + timedelta(minutes=138)) == pytest.approx(78 / 390, abs=1e-15)
    assert trading_time(xnys, opened - timedelta(hours=5), closed + timedelta(hours=5)) == 1.0  # overnight padding adds nothing
    assert trading_time(xnys, opened - timedelta(hours=5), opened + timedelta(minutes=195)) == pytest.approx(0.5, abs=1e-15)


def test_trading_time_is_additive_and_matches_the_session_count(xnys: XnysCalendar) -> None:
    a = xnys.offset_from_close(date(2026, 11, 20), 25)
    m = xnys.open_close(date(2026, 11, 27))[0] + timedelta(minutes=100)  # inside the half day
    b = xnys.offset_from_close(date(2026, 12, 4), 40)
    assert trading_time(xnys, a, m) + trading_time(xnys, m, b) == pytest.approx(trading_time(xnys, a, b), abs=1e-12)
    # close to close it IS the session count, however long the span
    first, last = date(2026, 9, 15), date(2026, 12, 18)
    assert trading_time(xnys, _close(xnys, first), _close(xnys, last)) == float(xnys.sessions_between(first, last)) == 67.0
    # tt_sessions of a Saturday-dated monthly runs to the Friday close
    ts = _close(xnys, date(2012, 3, 7))
    assert trading_time(xnys, ts, _close(xnys, last_session(xnys, date(2012, 3, 17)))) == 7.0


def test_trading_time_accepts_any_aware_datetime_and_refuses_naive_ones(xnys: XnysCalendar) -> None:
    a = datetime(2026, 9, 15, 16, 0, tzinfo=NY)
    b = datetime(2026, 9, 16, 16, 0, tzinfo=NY)
    assert trading_time(xnys, a, b) == 1.0
    with pytest.raises(ValueError, match="tz-aware"):
        trading_time(xnys, datetime(2026, 9, 15, 20, 0), b)  # noqa: DTZ001


class _HalfDayCalendar:
    """A minimal non-XNYS calendar: `trading_time` must work from the Protocol methods alone (e.g. the broker calendar)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def sessions(self, start: date, end: date) -> list[date]:
        self.calls.append("sessions")
        days = [date(2030, 1, 7), date(2030, 1, 8), date(2030, 1, 9)]
        return [d for d in days if start <= d <= end]

    def open_close(self, session: date) -> tuple[datetime, datetime]:
        self.calls.append("open_close")
        opened = datetime(session.year, session.month, session.day, 10, 0, tzinfo=UTC)
        return opened, opened + timedelta(minutes=100 if session == date(2030, 1, 8) else 200)

    def prev_or_same_session(self, d: date) -> date:
        return max(s for s in (date(2030, 1, 7), date(2030, 1, 8), date(2030, 1, 9)) if s <= d)


def test_trading_time_uses_only_the_protocol() -> None:
    fake = _HalfDayCalendar()
    a = datetime(2030, 1, 7, 12, 30, tzinfo=UTC)  # 150 of 200 minutes into the first session
    b = datetime(2030, 1, 9, 10, 50, tzinfo=UTC)  # 50 of 200 minutes into the third
    assert trading_time(fake, a, b) == pytest.approx(50 / 200 + 1.0 + 50 / 200, abs=1e-15)
    assert trading_time(fake, datetime(2030, 1, 1, tzinfo=UTC), datetime(2030, 2, 1, tzinfo=UTC)) == 3.0
    assert last_session(fake, date(2030, 1, 12)) == date(2030, 1, 9)
    assert set(fake.calls) == {"sessions", "open_close"}
