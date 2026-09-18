"""`paper/clock.py`: the RTT-compensated broker clock and the broker calendar (DESIGN 11.4, 3.1; D25, G4; INV-12, INV-13).

Every UTC instant asserted here is computed by hand from the exchange-local time and the offset in force on that date
(EST = UTC-5, EDT = UTC-4), never by re-running the module's own arithmetic.
"""

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta

import pytest

from jevbot.cal import XnysCalendar
from jevbot.errors import BrokerError, InvariantError
from jevbot.paper.clock import AlpacaCalendar, BrokerClock, boottime, to_utc
from tests.fixtures.fake_alpaca import FakeTradingClient, calendar_models

MAX_SKEW_MS = 5_000  # health.max_clock_skew_ms (D25)


class FakeClockSource:
    """One vendor clock reading, with a scripted `timestamp`. `boottime` is driven by an explicit list of readings."""

    def __init__(self, timestamps: Sequence[datetime], *, is_open: bool = True) -> None:
        self.timestamps = list(timestamps)
        self.is_open = is_open
        self.reads = 0

    def __call__(self) -> object:
        stamp = self.timestamps[min(self.reads, len(self.timestamps) - 1)]
        self.reads += 1
        return _Reading(stamp, self.is_open)


class _Reading:
    def __init__(self, stamp: datetime, is_open: bool) -> None:
        self.timestamp = stamp
        self.is_open = is_open
        self.next_open = stamp + timedelta(hours=18)
        self.next_close = stamp + timedelta(hours=24)


class FakeBoottime:
    """A scripted `CLOCK_BOOTTIME`: each call returns the next value, repeating the last one forever."""

    def __init__(self, values: Sequence[float]) -> None:
        self.values = list(values)
        self.index = 0

    def __call__(self) -> float:
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        return value

    def push(self, *values: float) -> None:
        self.values += list(values)


def make_clock(
    *,
    broker_ts: datetime,
    boottimes: Sequence[float],
    local: datetime,
    max_skew_ms: int = MAX_SKEW_MS,
    **kwargs: object,
) -> tuple[BrokerClock, FakeBoottime, FakeClockSource]:
    source = FakeClockSource([broker_ts])
    ticks = FakeBoottime(boottimes)
    clock = BrokerClock(
        source,  # type: ignore[arg-type]
        max_skew_ms=max_skew_ms,
        boottime_fn=ticks,
        utcnow=lambda: local,
        **kwargs,  # type: ignore[arg-type]
    )
    return clock, ticks, source


T0 = datetime(2026, 9, 17, 19, 45, 0, tzinfo=UTC)


# ======================================================================================================================
# BrokerClock: round-trip compensation (D25)
# ======================================================================================================================


def test_the_skew_is_measured_at_the_midpoint_of_the_round_trip() -> None:
    # the call is bracketed by boottime 100.0 and 100.4, so local UTC "at the same instant as the broker stamp" is T0 + 200 ms
    clock, _, _ = make_clock(broker_ts=T0 + timedelta(milliseconds=200), boottimes=[100.0, 100.4], local=T0)

    reading = clock.sync()

    assert reading.rtt_ms == 400
    assert reading.skew_ms == 0  # a 400 ms read contributes NO apparent skew
    assert reading.broker_ts == T0 + timedelta(milliseconds=200)
    assert reading.local_ts == T0 + timedelta(milliseconds=200)


def test_without_compensation_the_same_reading_would_look_skewed() -> None:
    # the broker stamp equals local time at the START of the call: half the round trip is the apparent skew
    clock, _, _ = make_clock(broker_ts=T0, boottimes=[100.0, 100.4], local=T0)

    assert clock.sync().skew_ms == 200


def test_a_real_skew_survives_the_compensation_and_is_reported_absolute() -> None:
    clock, _, _ = make_clock(broker_ts=T0 + timedelta(seconds=7), boottimes=[100.0, 100.0], local=T0)

    reading = clock.sync()

    assert reading.skew_ms == 7_000
    assert clock.skew_exceeded is True


def test_a_skew_behind_the_broker_is_reported_with_the_same_magnitude() -> None:
    clock, _, _ = make_clock(broker_ts=T0 - timedelta(seconds=7), boottimes=[100.0, 100.0], local=T0)

    assert clock.sync().skew_ms == 7_000


def test_a_skew_inside_the_threshold_does_not_trip_anything() -> None:
    clock, _, _ = make_clock(broker_ts=T0 + timedelta(milliseconds=4_999), boottimes=[100.0, 100.0], local=T0)

    assert clock.sync().skew_ms == 4_999
    assert clock.skew_exceeded is False
    assert clock.skew_high_for_s() == 0.0


def test_a_persistent_skew_is_timed_from_the_first_breach() -> None:
    source = FakeClockSource([T0 + timedelta(seconds=7)])
    ticks = FakeBoottime([100.0, 100.0])
    clock = BrokerClock(source, max_skew_ms=MAX_SKEW_MS, boottime_fn=ticks, utcnow=lambda: T0)  # type: ignore[arg-type]

    clock.sync()
    ticks.push(400.0, 400.0)  # a second sync 300 s later, still skewed
    clock.sync()
    ticks.push(400.0)

    assert clock.skew_high_for_s() == pytest.approx(300.0)


def test_the_persistence_timer_resets_once_the_skew_clears() -> None:
    source = FakeClockSource([T0 + timedelta(seconds=7), T0])
    ticks = FakeBoottime([100.0, 100.0])
    locals_ = [T0, T0]
    clock = BrokerClock(source, max_skew_ms=MAX_SKEW_MS, boottime_fn=ticks, utcnow=lambda: locals_[0])  # type: ignore[arg-type]

    clock.sync()
    ticks.push(400.0, 400.0)
    clock.sync()

    assert clock.skew_exceeded is False
    assert clock.skew_high_for_s() == 0.0


# ======================================================================================================================
# BrokerClock: now() over CLOCK_BOOTTIME (INV-12)
# ======================================================================================================================


def test_now_is_the_broker_stamp_plus_elapsed_boottime_never_the_host_clock() -> None:
    clock, ticks, _ = make_clock(broker_ts=T0, boottimes=[100.0, 100.0], local=T0 + timedelta(hours=3))
    clock.sync()
    ticks.push(112.5)

    assert clock.now() == T0 + timedelta(seconds=12.5)  # the three-hour host drift is invisible


def test_now_before_the_first_sync_is_a_bug_not_a_guess() -> None:
    clock, _, _ = make_clock(broker_ts=T0, boottimes=[100.0], local=T0)

    assert clock.reading() is None
    assert clock.needs_sync is True
    with pytest.raises(InvariantError, match="before the first sync"):
        clock.now()


def test_a_boottime_gap_marks_the_clock_unsynced_but_still_answers() -> None:
    clock, ticks, _ = make_clock(broker_ts=T0, boottimes=[100.0, 100.0], local=T0)
    clock.sync()
    assert clock.synced is True

    ticks.push(130.0)  # 30 s: an ordinary tick
    clock.now()
    assert clock.synced is True

    ticks.push(400.0)  # a 270 s jump: the WSL VM was suspended (CLOCK_MONOTONIC would not have noticed)
    bridged = clock.now()

    assert bridged == T0 + timedelta(seconds=300)  # the sleep IS counted: boottime kept running
    assert clock.synced is False
    assert clock.needs_sync is True


def test_a_resync_after_a_gap_clears_the_flag() -> None:
    source = FakeClockSource([T0, T0 + timedelta(seconds=300)])
    ticks = FakeBoottime([100.0, 100.0])
    clock = BrokerClock(source, max_skew_ms=MAX_SKEW_MS, boottime_fn=ticks, utcnow=lambda: T0 + timedelta(seconds=300))  # type: ignore[arg-type]
    clock.sync()
    ticks.push(400.0)
    clock.now()
    assert clock.needs_sync is True

    ticks.push(400.0, 400.0)
    clock.sync()

    assert clock.synced is True
    assert clock.needs_sync is False
    assert source.reads == 2


def test_the_sync_falls_due_again_after_the_resync_interval() -> None:
    clock, ticks, _ = make_clock(broker_ts=T0, boottimes=[100.0, 100.0], local=T0, resync_after_s=60.0)
    clock.sync()

    ticks.push(159.0)
    assert clock.needs_sync is False
    ticks.push(161.0)
    assert clock.needs_sync is True


def test_a_naive_broker_timestamp_is_refused() -> None:
    source = FakeClockSource([datetime(2026, 9, 17, 19, 45)])  # noqa: DTZ001 - the hostile input under test
    clock = BrokerClock(source, max_skew_ms=MAX_SKEW_MS, boottime_fn=FakeBoottime([1.0]), utcnow=lambda: T0)  # type: ignore[arg-type]

    with pytest.raises(BrokerError, match="naive"):
        clock.sync()


def test_the_clock_reading_carries_the_session_bounds_the_runner_needs() -> None:
    clock, _, _ = make_clock(broker_ts=T0, boottimes=[100.0, 100.0], local=T0)

    reading = clock.sync()

    assert reading.is_open is True
    assert reading.next_open == T0 + timedelta(hours=18)
    assert reading.next_close == T0 + timedelta(hours=24)
    assert clock.reading() is reading


def test_boottime_advances_and_is_monotonic() -> None:
    first = boottime()
    second = boottime()
    assert second >= first > 0.0


def test_to_utc_normalises_an_offset_timestamp() -> None:
    eastern = datetime.fromisoformat("2026-09-17T15:45:00-04:00")
    assert to_utc(eastern, "x") == datetime(2026, 9, 17, 19, 45, tzinfo=UTC)
    with pytest.raises(BrokerError, match="not a datetime"):
        to_utc("2026-09-17", "x")


def test_the_clock_reads_through_the_real_vendor_model() -> None:
    client = FakeTradingClient(clock_timestamp=T0)
    clock = BrokerClock(client.get_clock, max_skew_ms=MAX_SKEW_MS, boottime_fn=FakeBoottime([1.0, 1.0]), utcnow=lambda: T0)

    assert clock.sync().broker_ts == T0


# ======================================================================================================================
# AlpacaCalendar: naive-Eastern parsing across DST (G4)
# ======================================================================================================================

# DST 2026: forward on Sunday 8 March, back on Sunday 1 November.
DST_ROWS = (
    ("2026-03-06", "09:30", "16:00"),  # Friday, EST = UTC-5
    ("2026-03-09", "09:30", "16:00"),  # Monday, EDT = UTC-4
    ("2026-10-30", "09:30", "16:00"),  # Friday, EDT
    ("2026-11-02", "09:30", "16:00"),  # Monday, EST
)
THANKSGIVING_ROWS = (
    ("2026-11-25", "09:30", "16:00"),
    ("2026-11-27", "09:30", "13:00"),  # the half day after Thanksgiving
    ("2026-11-30", "09:30", "16:00"),
)


def build(rows: Sequence[tuple[str, str, str]], **kwargs: object) -> AlpacaCalendar:
    return AlpacaCalendar(calendar_models(rows), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("session", "open_utc", "close_utc"),
    [
        (date(2026, 3, 6), datetime(2026, 3, 6, 14, 30, tzinfo=UTC), datetime(2026, 3, 6, 21, 0, tzinfo=UTC)),
        (date(2026, 3, 9), datetime(2026, 3, 9, 13, 30, tzinfo=UTC), datetime(2026, 3, 9, 20, 0, tzinfo=UTC)),
        (date(2026, 10, 30), datetime(2026, 10, 30, 13, 30, tzinfo=UTC), datetime(2026, 10, 30, 20, 0, tzinfo=UTC)),
        (date(2026, 11, 2), datetime(2026, 11, 2, 14, 30, tzinfo=UTC), datetime(2026, 11, 2, 21, 0, tzinfo=UTC)),
    ],
)
def test_naive_eastern_rows_localise_correctly_across_both_dst_boundaries(session: date, open_utc: datetime, close_utc: datetime) -> None:
    assert build(DST_ROWS).open_close(session) == (open_utc, close_utc)


def test_the_same_local_string_is_one_hour_apart_in_utc_across_the_spring_change() -> None:
    calendar = build(DST_ROWS)
    before = calendar.open_close(date(2026, 3, 6))[0]
    after = calendar.open_close(date(2026, 3, 9))[0]
    assert (before.time().hour - after.time().hour) == 1


def test_it_agrees_with_xnys_on_the_real_sessions() -> None:
    xnys = XnysCalendar(date(2026, 1, 1), date(2026, 12, 31))
    calendar = build(DST_ROWS)
    for row in DST_ROWS:
        session = date.fromisoformat(row[0])
        assert calendar.open_close(session) == xnys.open_close(session)


# ======================================================================================================================
# AlpacaCalendar: early closes and cut-offs (INV-13)
# ======================================================================================================================


def test_an_early_close_is_detected_against_the_calendar_s_own_regular_close() -> None:
    calendar = build(THANKSGIVING_ROWS)

    assert calendar.is_early_close(date(2026, 11, 27)) is True
    assert calendar.is_early_close(date(2026, 11, 25)) is False
    assert calendar.open_close(date(2026, 11, 27))[1] == datetime(2026, 11, 27, 18, 0, tzinfo=UTC)


def test_every_cut_off_shifts_by_exactly_three_hours_on_a_thirteen_hundred_close() -> None:
    """The same session, run twice: once with the regular close and once as a half day (11.3, INV-13)."""
    session = date(2026, 11, 27)
    full_day = build([("2026-11-27", "09:30", "16:00")])
    half_day = build([("2026-11-27", "09:30", "13:00")])

    # the section-11.3 day: pre-flight, decision snapshot, execute, order cut-off, cancel-all, the EOD mark after the close
    for minutes in (40, 25, 20, 5, 4, -2, -10):
        assert full_day.offset_from_close(session, minutes) - half_day.offset_from_close(session, minutes) == timedelta(hours=3)

    assert half_day.offset_from_close(session, 25) == datetime(2026, 11, 27, 17, 35, tzinfo=UTC)
    assert half_day.offset_from_close(session, -2) == datetime(2026, 11, 27, 18, 2, tzinfo=UTC)  # close + 2 min
    assert full_day.offset_from_close(session, 25) == datetime(2026, 11, 27, 20, 35, tzinfo=UTC)


# ======================================================================================================================
# AlpacaCalendar: the cross-check - the earlier close wins (G4)
# ======================================================================================================================


class StubCalendar:
    """A minimal `Calendar` cross-check with hand-set closes."""

    def __init__(self, rows: dict[date, tuple[datetime, datetime]]) -> None:
        self.rows = rows

    def open_close(self, session: date) -> tuple[datetime, datetime]:
        if session not in self.rows:
            raise ValueError(f"{session} is not a session")
        return self.rows[session]

    def sessions(self, start: date, end: date) -> list[date]:
        return sorted(d for d in self.rows if start <= d <= end)


def test_the_earlier_close_wins_when_the_broker_is_late() -> None:
    session = date(2026, 11, 27)
    stub = StubCalendar({session: (datetime(2026, 11, 27, 14, 30, tzinfo=UTC), datetime(2026, 11, 27, 18, 0, tzinfo=UTC))})
    alerts: list[str] = []

    calendar = AlpacaCalendar(
        calendar_models([("2026-11-27", "09:30", "16:00")]),  # the broker still thinks it is a full day
        cross_check=stub,  # type: ignore[arg-type]
        on_alert=alerts.append,
    )

    assert calendar.open_close(session)[1] == datetime(2026, 11, 27, 18, 0, tzinfo=UTC)
    assert calendar.is_early_close(session) is False  # a one-row calendar has no earlier close to compare against
    assert len(alerts) == 1 and "earlier close" in alerts[0]


def test_the_earlier_close_wins_when_the_cross_check_is_late() -> None:
    session = date(2026, 11, 27)
    stub = StubCalendar({session: (datetime(2026, 11, 27, 14, 30, tzinfo=UTC), datetime(2026, 11, 27, 21, 0, tzinfo=UTC))})
    alerts: list[str] = []

    calendar = AlpacaCalendar(
        calendar_models([("2026-11-27", "09:30", "13:00")]),  # the broker has the half day
        cross_check=stub,  # type: ignore[arg-type]
        on_alert=alerts.append,
    )

    assert calendar.open_close(session)[1] == datetime(2026, 11, 27, 18, 0, tzinfo=UTC)
    assert len(alerts) == 1 and "earlier close" in alerts[0]


def test_agreement_raises_no_alert() -> None:
    xnys = XnysCalendar(date(2026, 1, 1), date(2026, 12, 31))
    calendar = AlpacaCalendar(calendar_models(THANKSGIVING_ROWS), cross_check=xnys)

    assert calendar.alerts == ()
    assert calendar.open_close(date(2026, 11, 27))[1] == datetime(2026, 11, 27, 18, 0, tzinfo=UTC)


def test_a_session_only_one_calendar_knows_is_alerted_in_both_directions() -> None:
    broker_only = date(2026, 11, 26)  # Thanksgiving: the cross-check says the exchange is shut
    stub = StubCalendar(
        {
            date(2026, 11, 25): (datetime(2026, 11, 25, 14, 30, tzinfo=UTC), datetime(2026, 11, 25, 21, 0, tzinfo=UTC)),
            date(2026, 11, 27): (datetime(2026, 11, 27, 14, 30, tzinfo=UTC), datetime(2026, 11, 27, 18, 0, tzinfo=UTC)),
            date(2026, 11, 30): (datetime(2026, 11, 30, 14, 30, tzinfo=UTC), datetime(2026, 11, 30, 21, 0, tzinfo=UTC)),
        }
    )
    alerts: list[str] = []

    calendar = AlpacaCalendar(
        calendar_models([("2026-11-25", "09:30", "16:00"), ("2026-11-26", "09:30", "16:00"), ("2026-11-30", "09:30", "16:00")]),
        cross_check=stub,  # type: ignore[arg-type]
        on_alert=alerts.append,
    )

    # The session set is the UNION: a day either calendar lists is kept, because a day we call a non-session gets no
    # cycle at all, so a mandatory expiry exit on it would silently never happen (INV-11, INV-21). Both directions alert.
    assert calendar.is_session(broker_only) is True
    assert calendar.is_session(date(2026, 11, 27)) is True
    assert calendar.open_close(date(2026, 11, 27))[1] == datetime(2026, 11, 27, 18, 0, tzinfo=UTC)  # the cross-check's 13:00 ET close
    assert [a for a in alerts if "the cross-check does not" in a and "2026-11-26" in a]
    assert [a for a in alerts if "the broker does not" in a and "2026-11-27" in a]
    assert calendar.alerts == tuple(alerts)


# ======================================================================================================================
# AlpacaCalendar: the rest of the 3.1 protocol
# ======================================================================================================================


def test_session_navigation() -> None:
    calendar = build(THANKSGIVING_ROWS)
    monday, friday, wednesday = date(2026, 11, 30), date(2026, 11, 27), date(2026, 11, 25)

    assert calendar.is_session(wednesday) is True
    assert calendar.is_session(date(2026, 11, 26)) is False
    assert calendar.sessions(wednesday, monday) == [wednesday, friday, monday]
    assert calendar.next_session(wednesday) == friday
    assert calendar.next_session(wednesday, 2) == monday
    assert calendar.prev_session(monday) == friday
    assert calendar.prev_session(monday, 2) == wednesday
    assert calendar.prev_or_same_session(friday) == friday
    assert calendar.prev_or_same_session(date(2026, 11, 26)) == wednesday  # Thanksgiving -> the Wednesday before
    assert calendar.sessions_between(wednesday, monday) == 2  # (a, b]
    assert calendar.sessions_between(monday, wednesday) == 0
    assert calendar.bounds == (wednesday, monday)


def test_session_of_and_next_open_after() -> None:
    calendar = build(THANKSGIVING_ROWS)

    assert calendar.session_of(datetime(2026, 11, 27, 17, 0, tzinfo=UTC)) == date(2026, 11, 27)
    assert calendar.session_of(datetime(2026, 11, 27, 18, 0, tzinfo=UTC)) == date(2026, 11, 27)  # the close is inclusive
    assert calendar.session_of(datetime(2026, 11, 27, 19, 0, tzinfo=UTC)) is None  # after the early close
    assert calendar.session_of(datetime(2026, 11, 26, 17, 0, tzinfo=UTC)) is None  # a holiday
    assert calendar.next_open_after(datetime(2026, 11, 27, 19, 0, tzinfo=UTC)) == datetime(2026, 11, 30, 14, 30, tzinfo=UTC)
    assert calendar.next_open_after(datetime(2026, 11, 27, 14, 30, tzinfo=UTC)) == datetime(2026, 11, 30, 14, 30, tzinfo=UTC)


def test_out_of_range_dates_raise_rather_than_answering_false() -> None:
    calendar = build(THANKSGIVING_ROWS)

    with pytest.raises(ValueError, match="outside the broker calendar"):
        calendar.is_session(date(2027, 1, 4))
    with pytest.raises(ValueError, match="not a session"):
        calendar.open_close(date(2026, 11, 26))
    with pytest.raises(ValueError, match="no session opens after"):
        calendar.next_open_after(datetime(2026, 12, 1, tzinfo=UTC))


@pytest.mark.parametrize(
    ("rows", "needle"),
    [
        ((), "empty trading calendar"),
        ((("2026-11-25", "09:30", "16:00"), ("2026-11-25", "09:30", "13:00")), "repeats 2026-11-25"),
        ((("2026-11-25", "16:00", "09:30"),), "closes at or before it opens"),
    ],
)
def test_a_malformed_calendar_is_refused(rows: Sequence[tuple[str, str, str]], needle: str) -> None:
    with pytest.raises(BrokerError, match=needle):
        build(rows)


def test_a_naive_timestamp_is_refused_by_the_instant_lookups() -> None:
    calendar = build(THANKSGIVING_ROWS)
    with pytest.raises(ValueError, match="tz-aware"):
        calendar.session_of(datetime(2026, 11, 27, 17, 0))  # noqa: DTZ001 - the hostile input under test


# ======================================================================================================================
# Protocol conformance and the defensive edges
# ======================================================================================================================


def protocol_signature(protocol: type) -> dict[str, list[str]]:
    import inspect

    return {
        name: [p for p in inspect.signature(getattr(protocol, name)).parameters if p != "self"]
        for name in dir(protocol)
        if not name.startswith("_") and callable(getattr(protocol, name, None))
    }


def test_the_broker_clock_matches_the_clock_protocol() -> None:
    from jevbot.protocols import Clock

    clock, _, _ = make_clock(broker_ts=T0, boottimes=[1.0, 1.0], local=T0)

    for name, parameters in protocol_signature(Clock).items():
        assert callable(getattr(clock, name, None)), name
        import inspect

        assert list(inspect.signature(getattr(clock, name)).parameters) == parameters, name


def test_the_alpaca_calendar_matches_the_calendar_protocol() -> None:
    from jevbot.protocols import Calendar

    calendar = build(THANKSGIVING_ROWS)

    for name, parameters in protocol_signature(Calendar).items():
        import inspect

        assert callable(getattr(calendar, name, None)), name
        assert list(inspect.signature(getattr(calendar, name)).parameters) == parameters, name


def test_an_already_localised_calendar_row_is_taken_as_it_stands() -> None:
    class Row:
        date = date(2026, 11, 27)
        open = datetime(2026, 11, 27, 14, 30, tzinfo=UTC)
        close = datetime(2026, 11, 27, 18, 0, tzinfo=UTC)

    calendar = AlpacaCalendar([Row()])  # type: ignore[list-item]

    assert calendar.open_close(date(2026, 11, 27)) == (Row.open, Row.close)


@pytest.mark.parametrize(
    ("attributes", "needle"),
    [
        ({"date": "2026-11-27", "open": datetime(2026, 11, 27, 9, 30), "close": datetime(2026, 11, 27, 16, 0)}, "not a calendar date"),  # noqa: DTZ001
        ({"date": date(2026, 11, 27), "open": "09:30", "close": datetime(2026, 11, 27, 16, 0)}, "not a datetime"),  # noqa: DTZ001
    ],
)
def test_a_malformed_calendar_row_is_refused(attributes: dict[str, object], needle: str) -> None:
    row = type("Row", (), attributes)()

    with pytest.raises(BrokerError, match=needle):
        AlpacaCalendar([row])  # type: ignore[list-item]


def test_a_cross_check_that_cannot_answer_the_range_is_tolerated() -> None:
    class Blind(StubCalendar):
        def sessions(self, start: date, end: date) -> list[date]:
            raise ValueError("outside my bounds")

    stub = Blind({date(2026, 11, 25): (datetime(2026, 11, 25, 14, 30, tzinfo=UTC), datetime(2026, 11, 25, 21, 0, tzinfo=UTC))})

    calendar = AlpacaCalendar(calendar_models([("2026-11-25", "09:30", "16:00")]), cross_check=stub)  # type: ignore[arg-type]

    assert calendar.alerts == ()
    assert calendar.is_session(date(2026, 11, 25)) is True


def test_a_cross_check_that_disagrees_about_the_open_is_alerted_without_changing_it() -> None:
    session = date(2026, 11, 25)
    stub = StubCalendar({session: (datetime(2026, 11, 25, 15, 0, tzinfo=UTC), datetime(2026, 11, 25, 21, 0, tzinfo=UTC))})
    alerts: list[str] = []

    calendar = AlpacaCalendar(
        calendar_models([("2026-11-25", "09:30", "16:00")]),
        cross_check=stub,  # type: ignore[arg-type]
        on_alert=alerts.append,
    )

    assert calendar.open_close(session)[0] == datetime(2026, 11, 25, 14, 30, tzinfo=UTC)  # the venue's own open stands
    assert any("broker open" in a for a in alerts)


def test_navigation_beyond_the_loaded_window_raises() -> None:
    calendar = build(THANKSGIVING_ROWS)

    with pytest.raises(ValueError, match="lies outside"):
        calendar.next_session(date(2026, 11, 30))
    with pytest.raises(ValueError, match="lies outside"):
        calendar.prev_session(date(2026, 11, 25))
    with pytest.raises(ValueError, match="must be an int >= 1"):
        calendar.next_session(date(2026, 11, 25), 0)
    with pytest.raises(ValueError, match=r"must be a datetime\.date"):
        calendar.is_session(datetime(2026, 11, 25, tzinfo=UTC))  # type: ignore[arg-type]


def test_calendar_rows_reads_through_a_trading_client() -> None:
    from jevbot.paper.clock import calendar_rows

    client = FakeTradingClient(calendar=[("2026-11-25", "09:30", "16:00"), ("2026-11-27", "09:30", "13:00")])

    rows = calendar_rows(client, date(2026, 11, 26), date(2026, 11, 30))

    assert [row.date for row in rows] == [date(2026, 11, 27)]  # the request window is honoured
    assert AlpacaCalendar(rows).is_early_close(date(2026, 11, 27)) is False  # one row: nothing to be early against
