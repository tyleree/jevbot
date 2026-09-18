"""`data/series.py`: `PitTable` row and column gating, composite keys, and the table-backed news / event sources
(DESIGN.md 3.2, 5.1, 5.4, 15.1)."""

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from jevbot.cal import XnysCalendar
from jevbot.data.series import EVENT_KINDS, NullNewsSource, PitTable, TableEventSource, TableNewsSource
from jevbot.data.store import BARS_COLUMN_KNOWABLE, DAILY_COLUMN_KNOWABLE, DAILY_KEY
from jevbot.errors import DataError, DataUnavailable, PitViolation
from jevbot.types import NewsItem, ScheduledEvent, Slot

SESSIONS: list[date] = [date(2024, 5, 15), date(2024, 5, 16), date(2024, 5, 17)]
# deliberately naive (built from a tz-aware instant so the DTZ lint stays on): every gated read must refuse it
NAIVE: datetime = datetime(2024, 5, 17, 20, 0, tzinfo=UTC).replace(tzinfo=None)


@pytest.fixture(scope="module")
def cal() -> XnysCalendar:
    return XnysCalendar()


def opens(cal: XnysCalendar, session: date) -> datetime:
    return cal.open_close(session)[0]


def closes(cal: XnysCalendar, session: date) -> datetime:
    return cal.open_close(session)[1]


def bars_frame(cal: XnysCalendar) -> pd.DataFrame:
    """Three daily bars with the 3.2 gating: the ROW appears 60 s after the open, high/low/close/volume at the NEXT open."""
    row_gate = [opens(cal, s) + timedelta(seconds=60) for s in SESSIONS]
    hlcv_gate = [cal.next_open_after(closes(cal, s)) for s in SESSIONS]
    return pd.DataFrame(
        {
            "session": pd.to_datetime(SESSIONS),
            "open": [448.0, 449.0, 450.5],
            "high": [452.0, 453.0, 454.0],
            "low": [447.0, 448.0, 449.0],
            "close": [449.5, 450.0, 451.25],
            "volume": [50_000_000, 51_000_000, 52_000_000],
            "knowable_at": pd.to_datetime(row_gate, utc=True),
            "open_knowable_at": pd.to_datetime(row_gate, utc=True),
            "hlcv_knowable_at": pd.to_datetime(hlcv_gate, utc=True),
        }
    )


def bars_table(cal: XnysCalendar) -> PitTable:
    return PitTable(
        bars_frame(cal), name="bars:SPY", key="session", column_knowable=BARS_COLUMN_KNOWABLE, paths=["pq/bars/alpaca/SPY.parquet"]
    )


def daily_frame(cal: XnysCalendar, *, close_gate: str = "snapshot") -> pd.DataFrame:
    """Two sessions x two slots: `close_c` lives on the `eod` row only (5.4).

    `close_gate` is when the close becomes knowable: `snapshot` (mirror: the snapshot time IS the close), `next_open`
    (paper: `record_close` fills it the next morning) or `none` (not recorded yet).
    """
    rows: list[dict[str, object]] = []
    for session in SESSIONS[:2]:
        for slot in (Slot.DEC, Slot.EOD):
            eod = slot is Slot.EOD
            gate = {"snapshot": closes(cal, session), "next_open": cal.next_open_after(closes(cal, session)), "none": None}[close_gate]
            rows.append(
                {
                    "session": pd.Timestamp(session),
                    "slot": slot.value,
                    "px_c": 45_000 if eod else 44_990,
                    "close_c": 45_000 if eod else None,
                    "close_knowable_at": gate if eod else None,
                    "knowable_at": closes(cal, session) - timedelta(minutes=0 if eod else 25),
                }
            )
    frame = pd.DataFrame(rows)
    frame["close_c"] = frame["close_c"].astype("Int64")
    frame["close_knowable_at"] = pd.to_datetime(frame["close_knowable_at"], utc=True)
    frame["knowable_at"] = pd.to_datetime(frame["knowable_at"], utc=True)
    return frame


def daily_table(cal: XnysCalendar, *, close_gate: str = "snapshot") -> PitTable:
    return PitTable(daily_frame(cal, close_gate=close_gate), name="daily:SPY", key=DAILY_KEY, column_knowable=DAILY_COLUMN_KNOWABLE)


# ======================================================================================================================
# Construction
# ======================================================================================================================


def test_a_table_needs_a_tz_aware_row_gate_and_a_unique_key(cal: XnysCalendar) -> None:
    frame = bars_frame(cal)
    with pytest.raises(DataError, match="missing required columns"):
        PitTable(frame.drop(columns=["knowable_at"]), name="bars:SPY")
    with pytest.raises(DataError, match="missing required columns"):
        PitTable(frame, name="bars:SPY", key="slot")
    naive = frame.assign(knowable_at=pd.to_datetime([closes(cal, s) for s in SESSIONS]).tz_localize(None))
    with pytest.raises(DataError, match="tz-aware datetime column"):
        PitTable(naive, name="bars:SPY")
    holed = frame.copy()
    holed.loc[1, "knowable_at"] = pd.NaT
    with pytest.raises(DataError, match="may never be null"):
        PitTable(holed, name="bars:SPY")
    with pytest.raises(DataError, match="duplicate key"):
        PitTable(pd.concat([frame, frame.head(1)], ignore_index=True), name="bars:SPY")
    with pytest.raises(DataError, match="unknown value column"):
        PitTable(frame, name="bars:SPY", column_knowable={"nope": "hlcv_knowable_at"})
    with pytest.raises(DataError, match="unknown timestamp column"):
        PitTable(frame, name="bars:SPY", column_knowable={"high": "nope"})
    with pytest.raises(DataError, match="at least one column"):
        PitTable(frame, name="bars:SPY", key=())


def test_the_table_reports_its_shape_and_a_content_digest(cal: XnysCalendar) -> None:
    table = bars_table(cal)
    assert len(table) == 3 and table.key == ("session",) and table.name == "bars:SPY"
    assert table.paths == ("pq/bars/alpaca/SPY.parquet",) and "hlcv_knowable_at" in table.columns
    assert repr(table).startswith("PitTable('bars:SPY', rows=3")
    assert table.sha256() == bars_table(cal).sha256()
    changed = bars_frame(cal)
    changed.loc[0, "open"] = 448.01
    assert PitTable(changed, name="bars:SPY", column_knowable=BARS_COLUMN_KNOWABLE).sha256() != table.sha256()
    assert table.frame().equals(bars_table(cal).frame())  # frame() is an ungated copy for `data verify` / writers


# ======================================================================================================================
# Row gating and column gating (3.2) - the tests/guards/test_pit.py sentence, at the table level
# ======================================================================================================================


def test_todays_bar_row_gives_open_but_never_high_low_close_or_volume(cal: XnysCalendar) -> None:
    """The strict-parity rule of 0.1 item 12 / 5.1, enforced by ONE column_knowable mapping."""
    table = bars_table(cal)
    today = SESSIONS[-1]
    as_of = closes(cal, today)  # an EOD decision on the last session

    visible = table.asof(as_of)
    assert list(pd.to_datetime(visible["session"]).dt.date) == SESSIONS  # all three rows are visible ...
    today_row = visible[visible["session"] == pd.Timestamp(today)].iloc[0]
    assert today_row["open"] == 450.5  # ... today's OPEN is readable ...
    for column in ("high", "low", "close", "volume"):
        assert pd.isna(today_row[column]), column  # ... and today's high / low / close / volume are NULL
    done = visible[visible["session"] < pd.Timestamp(today)]
    assert list(done["close"]) == [449.5, 450.0]  # completed sessions keep theirs

    row = table.row(today, as_of)
    assert row["open"] == 450.5 and all(pd.isna(row[c]) for c in ("high", "low", "close", "volume"))
    assert table.value(today, "open", as_of) == 450.5
    for column in ("high", "low", "close", "volume"):
        with pytest.raises(PitViolation, match="is knowable at"):
            table.value(today, column, as_of)
    # after the next open every column of that row is readable
    later = cal.next_open_after(as_of)
    assert table.value(today, "close", later) == 451.25
    assert not pd.isna(table.asof(later).iloc[-1]["close"])


def test_a_row_is_invisible_before_its_own_gate(cal: XnysCalendar) -> None:
    table = bars_table(cal)
    before_today = opens(cal, SESSIONS[-1]) + timedelta(seconds=59)
    visible = table.asof(before_today)
    assert list(pd.to_datetime(visible["session"]).dt.date) == SESSIONS[:2]  # range read FILTERS
    with pytest.raises(PitViolation, match="row \\(2024-05-17"):  # keyed read RAISES
        table.row(SESSIONS[-1], before_today)
    with pytest.raises(PitViolation):
        table.value(SESSIONS[-1], "open", before_today)
    assert table.asof(opens(cal, SESSIONS[0])).empty  # nothing knowable yet: an EMPTY frame, not an error


def test_missing_rows_and_columns_are_named(cal: XnysCalendar) -> None:
    table = bars_table(cal)
    as_of = closes(cal, SESSIONS[-1])
    with pytest.raises(DataUnavailable, match="no row for key"):
        table.row(date(2024, 5, 20), as_of)
    with pytest.raises(DataUnavailable, match="no row for key"):
        table.value(date(2024, 5, 20), "open", as_of)
    with pytest.raises(DataError, match="no column 'nope'"):
        table.value(SESSIONS[0], "nope", as_of)
    with pytest.raises(DataError, match="needs 1"):
        table.row((SESSIONS[0], "eod"), as_of)
    assert table.has(SESSIONS[0]) and not table.has(date(2024, 5, 20))


def test_a_composite_session_slot_key_accepts_dates_timestamps_and_slots(cal: XnysCalendar) -> None:
    table = daily_table(cal)
    as_of = closes(cal, SESSIONS[1])
    assert table.key == ("session", "slot")
    row = table.row((SESSIONS[1], Slot.EOD), as_of)
    assert int(row["px_c"]) == 45_000
    assert table.row((pd.Timestamp(SESSIONS[1]), "eod"), as_of).equals(row)
    assert table.row((datetime(2024, 5, 16, tzinfo=UTC).date(), Slot.EOD.value), as_of).equals(row)
    assert table.has((SESSIONS[0], Slot.DEC)) and not table.has((SESSIONS[0], Slot.EXEC))
    assert table.value((SESSIONS[0], Slot.EOD), "close_c", as_of) == 45_000
    assert isinstance(table.value((SESSIONS[0], Slot.EOD), "close_c", as_of), int)


def test_close_c_is_gated_by_its_own_timestamp_and_only_exists_on_eod_rows(cal: XnysCalendar) -> None:
    """`MarketView.close()` IS this call (5.4): the close of session D is unreadable until `close_knowable_at`."""
    table = daily_table(cal)
    session = SESSIONS[1]
    decision = closes(cal, session) - timedelta(minutes=25)  # the `dec` slot, 25 minutes before the close
    with pytest.raises(PitViolation, match="close_c"):
        table.value((session, Slot.EOD), "close_c", decision)
    assert table.value((session, Slot.EOD), "close_c", closes(cal, session)) == 45_000
    # the dec row of the same session carries no close at all - unavailable, never a PIT violation
    with pytest.raises(DataUnavailable, match="has not been recorded yet"):
        table.value((session, Slot.DEC), "close_c", closes(cal, session))
    # the PREVIOUS session's close is knowable at this decision
    assert not pd.isna(table.row((SESSIONS[0], Slot.EOD), decision)["close_c"])
    assert table.value((SESSIONS[0], Slot.EOD), "close_c", decision) == 45_000


def test_the_column_gate_hides_a_close_that_the_row_itself_already_carries(cal: XnysCalendar) -> None:
    """Paper (5.4): the `eod` snapshot row appears at close + 2 min, its official `close_c` only the next morning. The row
    is readable, the column is not - which is exactly what a second, column-level gate buys."""
    table = daily_table(cal, close_gate="next_open")
    session = SESSIONS[1]
    at_close = closes(cal, session)
    row = table.row((session, Slot.EOD), at_close)
    assert int(row["px_c"]) == 45_000 and pd.isna(row["close_c"])  # the row is visible, the close is not
    visible = table.asof(at_close)
    eod_today = visible[(visible["session"] == pd.Timestamp(session)) & (visible["slot"] == Slot.EOD.value)]
    assert len(eod_today) == 1 and eod_today["close_c"].isna().all()
    with pytest.raises(PitViolation, match="close_c"):
        table.value((session, Slot.EOD), "close_c", at_close)
    assert table.value((session, Slot.EOD), "close_c", cal.next_open_after(at_close)) == 45_000


def test_a_null_value_behind_a_satisfied_gate_is_unavailable(cal: XnysCalendar) -> None:
    """A row that exists and is knowable, with a hole in the value itself (a vendor gap): `DataUnavailable`, never NaN."""
    frame = bars_frame(cal)
    frame.loc[0, "open"] = None
    table = PitTable(frame, name="bars:SPY", column_knowable=BARS_COLUMN_KNOWABLE)
    with pytest.raises(DataUnavailable, match="is null"):
        table.value(SESSIONS[0], "open", closes(cal, SESSIONS[-1]))
    assert table.value(SESSIONS[1], "open", closes(cal, SESSIONS[-1])) == 449.0


def test_a_close_that_was_never_recorded_is_unavailable_not_a_violation(cal: XnysCalendar) -> None:
    """Paper: the `eod` row exists from the snapshot, `record_close` fills the close next morning (5.4)."""
    table = daily_table(cal, close_gate="none")
    with pytest.raises(DataUnavailable, match="has not been recorded yet"):
        table.value((SESSIONS[1], Slot.EOD), "close_c", closes(cal, SESSIONS[1]) + timedelta(days=30))
    assert table.asof(closes(cal, SESSIONS[1]))["close_c"].isna().all()


def test_knowable_at_reports_the_effective_gate_without_checking_it(cal: XnysCalendar) -> None:
    table = daily_table(cal)
    key = (SESSIONS[1], Slot.EOD)
    assert table.knowable_at(key) == closes(cal, SESSIONS[1])
    assert table.knowable_at(key, "close_c") == closes(cal, SESSIONS[1])
    assert daily_table(cal, close_gate="none").knowable_at(key, "close_c") is None
    assert bars_table(cal).knowable_at(SESSIONS[0], "close") == cal.next_open_after(closes(cal, SESSIONS[0]))


def test_asof_needs_a_tz_aware_instant(cal: XnysCalendar) -> None:
    table = bars_table(cal)
    naive = NAIVE
    for call in (lambda: table.asof(naive), lambda: table.row(SESSIONS[0], naive), lambda: table.value(SESSIONS[0], "open", naive)):
        with pytest.raises(ValueError, match="tz-aware"):
            call()


# ======================================================================================================================
# News
# ======================================================================================================================


def item(ident: str, created: datetime, *, updated: datetime | None = None, symbols: tuple[str, ...] = ("SPY",)) -> NewsItem:
    return NewsItem(
        id=ident,
        created_at=created,
        updated_at=updated if updated is not None else created,
        received_at=None,
        knowable_at=created + timedelta(seconds=60),
        headline=f"headline {ident}",
        summary=f"summary {ident}",
        source="benzinga",
        symbols=symbols,
    )


def news_source() -> TableNewsSource:
    base = datetime(2024, 5, 17, 18, 0, tzinfo=UTC)
    items = [
        item("a", base - timedelta(hours=30)),  # outside a 24 h look-back
        item("b", base - timedelta(hours=2)),
        item("c", base - timedelta(minutes=30), updated=base + timedelta(hours=1)),  # revised after the decision
        item("d", base + timedelta(hours=2)),  # not knowable yet
        item("e", base - timedelta(hours=1), symbols=("QQQ",)),
    ]
    coverage = [("SPY", date(2024, 5, 1), date(2024, 5, 17)), ("QQQ", date(2024, 5, 1), date(2024, 5, 10))]
    return TableNewsSource.from_items(items, coverage)


def test_news_is_newest_first_gated_and_loses_a_revised_summary() -> None:
    source = news_source()
    as_of = datetime(2024, 5, 17, 18, 0, tzinfo=UTC)
    got = source.items("SPY", as_of, 24)
    assert [i.id for i in got] == ["c", "b"]  # newest first; "a" is outside the window, "d" is not knowable, "e" is QQQ
    assert got[0].summary is None and got[0].headline == "headline c"  # B6.3 rule 4: the revised text is not ours to see
    assert got[1].summary == "summary b"
    assert [i.id for i in source.items("SPY", as_of, 48)] == ["c", "b", "a"]
    assert source.items("SPY", as_of, 0) == ()
    assert [i.id for i in source.items("QQQ", as_of, 24)] == ["e"]
    later = as_of + timedelta(hours=3)
    assert [i.id for i in source.items("SPY", later, 24)] == ["d", "c", "b"]
    assert source.items("SPY", later, 24)[1].summary == "summary c"  # once the revision is public it is readable
    with pytest.raises(ValueError, match="lookback_hours"):
        source.items("SPY", as_of, -1)
    with pytest.raises(ValueError, match="tz-aware"):
        source.items("SPY", NAIVE, 24)


def test_news_coverage_separates_no_archive_from_no_news() -> None:
    source = news_source()
    assert source.covered("SPY", date(2024, 5, 17)) and not source.covered("SPY", date(2024, 5, 20))
    assert source.covered("QQQ", date(2024, 5, 9)) and not source.covered("QQQ", date(2024, 5, 17))
    assert not source.covered("IWM", date(2024, 5, 9))
    assert len(source) == 5 and source.coverage()[0][0] == "SPY"


def test_a_news_frame_is_read_like_the_archive_parquet() -> None:
    created = datetime(2024, 5, 17, 16, 0, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "id": ["1"],
            "created_at": pd.to_datetime([created]),
            "updated_at": pd.to_datetime([created]),
            "received_at": pd.to_datetime([None], utc=True),
            "knowable_at": pd.to_datetime([created]),
            "headline": ["a headline"],
            "summary": [None],
            "source": ["benzinga"],
            "symbols": [["SPY", "QQQ"]],
        }
    )
    coverage = pd.DataFrame({"underlying": ["SPY"], "start": ["2024-05-01"], "end": ["2024-05-31"], "fetched_at": [created]})
    source = TableNewsSource(frame, coverage)
    got = source.items("SPY", datetime(2024, 5, 17, 20, 0, tzinfo=UTC), 24)
    assert len(got) == 1 and got[0].symbols == ("SPY", "QQQ") and got[0].summary is None and got[0].received_at is None
    assert got[0].knowable_at == created and source.covered("SPY", date(2024, 5, 17))
    with pytest.raises(DataError, match="missing columns"):
        TableNewsSource(frame.drop(columns=["headline"]))


def test_the_null_news_source_is_never_covered() -> None:
    source = NullNewsSource()
    assert source.items("SPY", datetime(2024, 5, 17, 20, 0, tzinfo=UTC), 24) == ()
    assert not source.covered("SPY", date(2024, 5, 17))
    with pytest.raises(ValueError, match="tz-aware"):
        source.items("SPY", NAIVE, 24)


# ======================================================================================================================
# Scheduled events (5.1)
# ======================================================================================================================


def fomc(event_date: date, *, scheduled: bool = True, cancelled: bool = False) -> ScheduledEvent:
    """A historical scheduled meeting is knowable `data.fomc_knowable_days` (45) ahead at 00:00 UTC (5.1)."""
    return ScheduledEvent(
        kind="fomc_decision",
        event_date=event_date,
        underlying=None,
        amount_cents=None,
        scheduled=scheduled,
        cancelled=cancelled,
        knowable_at=datetime.combine(event_date - timedelta(days=45), datetime.min.time(), tzinfo=UTC),
        knowable_rule="scheduled_minus_45d_assumption",
        source_url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def exdiv(underlying: str, ex_date: date, amount: int) -> ScheduledEvent:
    return ScheduledEvent(
        kind="ex_dividend",
        event_date=ex_date,
        underlying=underlying,
        amount_cents=amount,
        scheduled=True,
        knowable_at=datetime.combine(ex_date - timedelta(days=14), datetime.min.time(), tzinfo=UTC),
        knowable_rule="exdiv_minus_14d_assumption",
        source_url="alpaca:corporate-actions",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def event_source() -> TableEventSource:
    return TableEventSource.from_events(
        [
            fomc(date(2015, 6, 17)),
            fomc(date(2015, 7, 29)),
            fomc(date(2015, 9, 17), cancelled=True),  # a meeting the source marks cancelled: stored, never served
            fomc(date(2008, 10, 8), scheduled=False),  # the emergency inter-meeting cut: stored, never served
            exdiv("SPY", date(2015, 6, 19), 110),
            exdiv("QQQ", date(2015, 6, 19), 30),
        ]
    )


def test_a_scheduled_meeting_appears_45_days_ahead_and_not_before() -> None:
    source = event_source()
    meeting = date(2015, 6, 17)
    thirty_before = datetime.combine(meeting - timedelta(days=30), datetime.min.time(), tzinfo=UTC)
    sixty_before = datetime.combine(meeting - timedelta(days=60), datetime.min.time(), tzinfo=UTC)
    assert [e.event_date for e in source.events(thirty_before, meeting - timedelta(days=30), meeting)] == [meeting]
    assert source.events(sixty_before, meeting - timedelta(days=60), meeting) == ()


def test_unscheduled_and_cancelled_rows_are_stored_but_never_served() -> None:
    source = event_source()
    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    served = source.events(as_of, date(2000, 1, 1), date(2030, 1, 1))
    assert [(e.kind, e.event_date) for e in served] == [
        ("fomc_decision", date(2015, 6, 17)),
        ("ex_dividend", date(2015, 6, 19)),
        ("ex_dividend", date(2015, 6, 19)),
        ("fomc_decision", date(2015, 7, 29)),
    ]
    assert date(2008, 10, 8) not in [e.event_date for e in served]  # 5.1: never in any events() result
    assert date(2015, 9, 17) not in [e.event_date for e in served]
    stored = {(e.event_date, e.scheduled, e.cancelled) for e in source.all_events()}
    assert (date(2008, 10, 8), False, False) in stored and (date(2015, 9, 17), True, True) in stored
    assert len(source) == 6


def test_an_underlying_sees_the_market_wide_rows_plus_only_its_own() -> None:
    source = event_source()
    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    window = (date(2015, 6, 1), date(2015, 6, 30))
    spy = source.events(as_of, *window, "SPY")
    assert [(e.kind, e.underlying) for e in spy] == [("fomc_decision", None), ("ex_dividend", "SPY")]
    qqq = source.events(as_of, *window, "QQQ")
    assert [(e.kind, e.underlying) for e in qqq] == [("fomc_decision", None), ("ex_dividend", "QQQ")]
    everything = source.events(as_of, *window, None)
    assert len(everything) == 3  # underlying=None serves every row
    assert source.events(as_of, date(2015, 6, 18), date(2015, 6, 30), "SPY")[0].kind == "ex_dividend"


def test_coverage_lists_only_verified_kinds_in_the_canonical_order() -> None:
    assert event_source().coverage() == ("fomc_decision", "ex_dividend")
    assert TableEventSource.from_events([]).coverage() == ()
    only_cancelled = TableEventSource.from_events([fomc(date(2015, 9, 17), cancelled=True)])
    assert only_cancelled.coverage() == ()  # a cancelled row is not coverage
    assert EVENT_KINDS == ("fomc_decision", "cpi", "nfp", "ex_dividend")


def test_an_events_frame_is_read_like_the_events_csv() -> None:
    """The 13.1 CSV round trip: string booleans, ISO dates and an empty underlying / amount."""
    frame = pd.DataFrame(
        {
            "kind": ["fomc_decision", "ex_dividend"],
            "event_date": ["2015-06-17", "2015-06-19"],
            "underlying": [None, "SPY"],
            "amount_cents": [None, 110],
            "scheduled": ["true", "true"],
            "cancelled": ["false", "false"],
            "knowable_at": ["2015-05-03T00:00:00Z", "2015-06-05T00:00:00Z"],
            "knowable_rule": ["scheduled_minus_45d_assumption", "exdiv_minus_14d_assumption"],
            "source_url": ["https://federalreserve.gov", "alpaca"],
            "fetched_at": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
        }
    )
    source = TableEventSource(frame)
    served = source.events(datetime(2026, 1, 1, tzinfo=UTC), date(2015, 1, 1), date(2015, 12, 31), "SPY")
    assert [(e.kind, e.underlying, e.amount_cents) for e in served] == [("fomc_decision", None, None), ("ex_dividend", "SPY", 110)]
    assert served[0].knowable_at == datetime(2015, 5, 3, tzinfo=UTC) and served[0].event_date == date(2015, 6, 17)
    with pytest.raises(DataError, match="missing columns"):
        TableEventSource(frame.drop(columns=["knowable_rule"]))
    with pytest.raises(DataError, match="must be a boolean"):
        TableEventSource(frame.assign(scheduled=["maybe", "true"]))


def test_event_reads_need_a_tz_aware_as_of() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        event_source().events(NAIVE, date(2015, 1, 1), date(2015, 12, 31))
