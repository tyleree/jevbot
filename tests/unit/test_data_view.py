"""`data/view.py`: the single point-in-time gate (DESIGN.md 3.2, 5.1, 5.4; INV-14; 15.1).

The reference for the read semantics is `tests/fixtures/fake_view.FakeView`, WP00's in-memory double of this very class: the
first test drives both over the SAME tables and demands identical answers, so the double every other package is unit-tested
against cannot drift from the implementation the engine runs.
"""

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from jevbot.cal import XnysCalendar
from jevbot.data.series import NullNewsSource, PitTable, TableEventSource, TableNewsSource
from jevbot.data.store import BARS_COLUMN_KNOWABLE, DAILY_COLUMN_KNOWABLE, DAILY_COLUMNS, DAILY_KEY
from jevbot.data.view import DataView, bars_table_name, daily_table_name, volidx_table_name
from jevbot.errors import DataUnavailable, PitViolation
from jevbot.types import ChainSnapshot, Fidelity, ScheduledEvent, Slot, SnapshotKey
from tests.fixtures.chain_factory import make_chain, xnys
from tests.fixtures.fake_view import (
    FakeView,
    History,
    fomc_event,
    make_history,
    make_rates,
    make_view,
    make_vol_indices,
    news_item,
)

SESSION: date = date(2024, 5, 17)


class StubChainProvider:
    """`ChainProvider` over a dict of snapshots. `opened_partitions` is the optional 13.1 hook a real provider implements."""

    def __init__(self, chains: list[ChainSnapshot], *, source: str = "mirror", partitions: tuple[str, ...] = ()) -> None:
        self._chains = {(c.underlying, c.key): c for c in chains}
        self._source = source
        self._partitions = partitions
        self.calls: list[tuple[str, SnapshotKey]] = []

    @property
    def fidelity(self) -> Fidelity:
        return next(iter(self._chains.values())).fidelity

    @property
    def source(self) -> str:
        return self._source

    def underlyings(self) -> tuple[str, ...]:
        return tuple(sorted({u for u, _ in self._chains}))

    def keys(self, underlying: str, start: date, end: date) -> list[SnapshotKey]:
        return sorted(k for (u, k) in self._chains if u == underlying and start <= k.session <= end)

    def get_chain(self, underlying: str, key: SnapshotKey) -> ChainSnapshot | None:
        self.calls.append((underlying, key))
        return self._chains.get((underlying, key))

    def manifest_hash(self) -> str:
        return "stub-manifest"

    def opened_partitions(self) -> tuple[str, ...]:
        return self._partitions


def daily_pit(frame: pd.DataFrame, underlying: str = "SPY", paths: tuple[str, ...] = ()) -> PitTable:
    return PitTable(frame, name=daily_table_name(underlying), key=DAILY_KEY, column_knowable=DAILY_COLUMN_KNOWABLE, paths=paths)


def bars_pit(frame: pd.DataFrame, underlying: str = "SPY", paths: tuple[str, ...] = ()) -> PitTable:
    return PitTable(frame, name=bars_table_name(underlying), key="session", column_knowable=BARS_COLUMN_KNOWABLE, paths=paths)


def build_view(
    *,
    chains: list[ChainSnapshot],
    histories: dict[str, History],
    key: SnapshotKey,
    as_of: datetime,
    vol: dict[str, pd.DataFrame] | None = None,
    rates: pd.DataFrame | None = None,
    events: tuple[ScheduledEvent, ...] = (),
    news: TableNewsSource | NullNewsSource | None = None,
    calendar: XnysCalendar | None = None,
    partitions: tuple[str, ...] = (),
    daily_paths: tuple[str, ...] = (),
) -> DataView:
    tables: dict[str, PitTable] = {}
    for underlying, history in histories.items():
        tables[daily_table_name(underlying)] = daily_pit(history.daily, underlying, daily_paths)
        tables[bars_table_name(underlying)] = bars_pit(history.bars, underlying)
    for name, frame in (vol or {}).items():
        tables[volidx_table_name(name)] = PitTable(frame, name=volidx_table_name(name))
    if rates is not None:
        tables["rates"] = PitTable(rates, name="rates")
    return DataView(
        key=key,
        as_of=as_of,
        calendar=calendar if calendar is not None else xnys(),
        chains=StubChainProvider(chains, partitions=partitions),
        tables=tables,
        news=news if news is not None else NullNewsSource(),
        events=TableEventSource.from_events(events),
    )


def world(
    *,
    slot: Slot = Slot.EOD,
    fidelity: Fidelity = Fidelity.EOD_QUOTES,
    n_sessions: int = 40,
    slots: tuple[Slot, ...] | None = None,
    events: tuple[ScheduledEvent, ...] | None = None,
    news: TableNewsSource | NullNewsSource | None = None,
) -> tuple[DataView, FakeView, ChainSnapshot]:
    """The same world twice: once behind `DataView` + `PitTable`s, once behind the `FakeView` double."""
    cal = xnys()
    history_slots = slots if slots is not None else ((Slot.EOD,) if slot is Slot.EOD else (slot, Slot.EOD))
    chain = make_chain("SPY", session=SESSION, slot=slot, fidelity=fidelity, calendar=cal)
    history = make_history(chain, n_sessions=n_sessions, slots=history_slots, calendar=cal)
    vol = make_vol_indices(history.iv30_bp, calendar=cal)
    rates = make_rates([pd.Timestamp(s).date() for s in history.iv30_bp.index], chain.rate, calendar=cal)
    scheduled = (fomc_event(cal.next_session(SESSION, 12)),) if events is None else events
    real = build_view(
        chains=[chain],
        histories={"SPY": history},
        key=chain.key,
        as_of=chain.ts,
        vol=vol,
        rates=rates,
        events=scheduled,
        news=news,
        calendar=cal,
    )
    double = FakeView(
        key=chain.key,
        as_of=chain.ts,
        calendar=cal,
        fidelity=fidelity,
        chains=[chain],
        daily={"SPY": history.daily},
        bars={"SPY": history.bars},
        vol_indices=vol,
        rates=rates,
        events=scheduled,
    )
    return real, double, chain


# ======================================================================================================================
# DataView and the WP00 double answer identically
# ======================================================================================================================


def test_data_view_answers_exactly_like_the_market_view_double() -> None:
    real, double, chain = world()
    assert real.key == double.key and real.session == double.session == SESSION
    assert real.as_of == double.as_of == chain.ts and real.fidelity is double.fidelity
    assert real.chain("SPY").content_hash == double.chain("SPY").content_hash
    assert real.spot("SPY") == double.spot("SPY") == chain.spot
    pd.testing.assert_series_equal(real.closes("SPY", 30), double.closes("SPY", 30))
    pd.testing.assert_frame_equal(real.daily("SPY", 30), double.daily("SPY", 30))
    pd.testing.assert_frame_equal(real.bars("SPY", 15), double.bars("SPY", 15))
    pd.testing.assert_series_equal(real.vol_index("VIX", 20), double.vol_index("VIX", 20))
    assert real.rate() == double.rate() == chain.rate
    assert real.today_open_ratio("SPY") == double.today_open_ratio("SPY")
    previous = xnys().prev_session(SESSION)
    assert real.close("SPY", previous) == double.close("SPY", previous)
    assert real.events(SESSION, SESSION + timedelta(days=60)) == double.events(SESSION, SESSION + timedelta(days=60))
    assert real.event_coverage() == double.event_coverage() == ("fomc_decision",)


def test_data_view_is_a_market_view_at_runtime() -> None:
    real, _double, _chain = world()
    for name in ("as_of", "key", "session", "calendar", "fidelity"):
        assert hasattr(real, name)
    for name in (
        "chain",
        "spot",
        "closes",
        "close",
        "bars",
        "today_open_ratio",
        "daily",
        "vol_index",
        "rate",
        "events",
        "event_coverage",
        "news",
        "news_covered",
        "touched",
    ):
        assert callable(getattr(real, name)), name
    assert repr(real).startswith("DataView(2024-05-17 eod")


# ======================================================================================================================
# 5.4: one row per session, the designated-slot rule
# ======================================================================================================================


def test_daily_returns_one_row_per_session_with_the_own_row_last() -> None:
    real, _double, chain = world(n_sessions=12)
    frame = real.daily("SPY", 5)
    assert list(frame.columns) == list(DAILY_COLUMNS) and len(frame) == 5
    sessions = [pd.Timestamp(s).date() for s in frame["session"]]
    assert sessions == sorted(sessions) and len(set(sessions)) == 5 and sessions[-1] == SESSION
    assert (frame["slot"] == Slot.EOD.value).all()
    assert int(frame["px_c"].iloc[-1]) == chain.spot  # the snapshot's own row
    assert real.daily("SPY", 1)["session"].iloc[0] == pd.Timestamp(SESSION)  # n = 1 is the own row alone


def test_a_three_slot_archive_and_the_same_data_collapsed_give_equal_daily_frames() -> None:
    """DESIGN 5.4 / 15.1: the read must be identical whether the archive holds 1 or 3 slots per session, so that every
    look-back window of 5.3 (`iv_rank`, `iv_chg_1w`, `skew_pctile`, `rv20_pctile`) indexes by SESSION in every mode."""
    cal = xnys()
    chain = make_chain("SPY", session=SESSION, slot=Slot.DEC, fidelity=Fidelity.RECORDED_INDICATIVE, calendar=cal)
    history = make_history(chain, n_sessions=30, slots=(Slot.DEC, Slot.EXEC, Slot.EOD), calendar=cal)
    assert set(history.daily["slot"]) == {"dec", "exec", "eod"} and len(history.daily) == 90

    three = build_view(chains=[chain], histories={"SPY": history}, key=chain.key, as_of=chain.ts).daily("SPY", 30)
    assert len(three) == 30 and (three["slot"] == "dec").all()
    assert three["close_c"].isna().all()  # a dec row never carries a close (5.4)

    collapsed = History(
        underlying="SPY",
        daily=history.daily[history.daily["slot"] == "dec"].reset_index(drop=True),
        bars=history.bars,
        iv30_bp=history.iv30_bp,
    )
    flat = build_view(chains=[chain], histories={"SPY": collapsed}, key=chain.key, as_of=chain.ts).daily("SPY", 30)
    pd.testing.assert_frame_equal(flat, three)


def test_a_mirror_era_session_falls_back_to_its_eod_row_seen_from_a_dec_view() -> None:
    """The real archive is mixed: mirror sessions have only an `eod` row, recorded days have three. A `dec` decision takes
    the `dec` row where it exists and that session's `eod` row otherwise; a session with neither is skipped (5.4)."""
    cal = xnys()
    chain = make_chain("SPY", session=SESSION, slot=Slot.DEC, fidelity=Fidelity.RECORDED_INDICATIVE, calendar=cal)
    history = make_history(chain, n_sessions=10, slots=(Slot.DEC, Slot.EXEC, Slot.EOD), calendar=cal)
    sessions = sorted({pd.Timestamp(s).date() for s in history.daily["session"]})
    mirror_era, recorded_era = sessions[:6], sessions[6:]
    is_mirror = history.daily["session"].isin([pd.Timestamp(s) for s in mirror_era])
    mixed = history.daily[~(is_mirror & history.daily["slot"].isin(["dec", "exec"]))].reset_index(drop=True)

    frame = build_view(
        chains=[chain], histories={"SPY": History("SPY", mixed, history.bars, history.iv30_bp)}, key=chain.key, as_of=chain.ts
    ).daily("SPY", 10)
    assert [pd.Timestamp(s).date() for s in frame["session"]] == sessions
    assert list(frame["slot"]) == ["eod"] * len(mirror_era) + ["dec"] * len(recorded_era)
    # the mirror-era rows bring their close with them; the dec rows do not
    assert frame["close_c"].iloc[: len(mirror_era)].notna().all() and frame["close_c"].iloc[len(mirror_era) :].isna().all()

    # a session with NEITHER a dec nor an eod row is skipped entirely
    dropped = sessions[2]
    thinner = mixed[mixed["session"] != pd.Timestamp(dropped)].reset_index(drop=True)
    frame = build_view(
        chains=[chain], histories={"SPY": History("SPY", thinner, history.bars, history.iv30_bp)}, key=chain.key, as_of=chain.ts
    ).daily("SPY", 10)
    assert dropped not in [pd.Timestamp(s).date() for s in frame["session"]] and len(frame) == len(sessions) - 1


def test_daily_needs_the_snapshots_own_row_and_refuses_a_future_one() -> None:
    real, _double, chain = world(n_sessions=6)
    with pytest.raises(PitViolation, match="is knowable at"):
        real_at(real, chain.ts - timedelta(seconds=1)).daily("SPY", 3)
    other = SnapshotKey(session=SESSION, slot=Slot.DEC)
    view = build_view(chains=[chain], histories={"SPY": _history_of(real)}, key=other, as_of=chain.ts)
    with pytest.raises(DataUnavailable, match="no row for key"):
        view.daily("SPY", 3)
    with pytest.raises(ValueError, match="n must be >= 1"):
        real.daily("SPY", 0)


# ======================================================================================================================
# closes() / close(): completed sessions, one value per session, keyed reads raise
# ======================================================================================================================


def test_closes_stop_before_this_session_and_carry_one_value_per_session() -> None:
    real, _double, chain = world(n_sessions=12)
    series = real.closes("SPY", 260)
    assert series.name == "close_c" and str(series.dtype) == "int64" and series.index.name == "session"
    sessions = [pd.Timestamp(s).date() for s in series.index]
    assert len(sessions) == len(set(sessions)) == 11 and max(sessions) < SESSION  # COMPLETED sessions only
    assert list(real.closes("SPY", 3)) == list(series)[-3:]
    assert real.closes("SPY", 0).empty and str(real.closes("SPY", 0).dtype) == "int64"
    with pytest.raises(ValueError, match="n must be >= 0"):
        real.closes("SPY", -1)
    assert real.close("SPY", sessions[-1]) == int(series.iloc[-1])
    # the mirror's own eod snapshot IS the close of its session: knowable exactly at as_of, never before
    assert real.close("SPY", SESSION) == chain.spot
    with pytest.raises(PitViolation):
        real_at(real, chain.ts - timedelta(seconds=1)).close("SPY", SESSION)
    with pytest.raises(DataUnavailable):
        real.close("SPY", date(2024, 5, 18))


def test_a_close_that_is_not_recorded_yet_is_unavailable_and_absent_from_closes() -> None:
    real, _double, chain = world(n_sessions=8)
    daily = _history_of(real).daily.copy()
    last = daily["session"].max()
    daily.loc[daily["session"] == last, ["close_c", "close_knowable_at"]] = [pd.NA, pd.NaT]
    view = build_view(
        chains=[chain], histories={"SPY": History("SPY", daily, _history_of(real).bars, _history_of(real).iv30_bp)}, key=chain.key, as_of=chain.ts
    )
    with pytest.raises(DataUnavailable, match="has not been recorded yet"):
        view.close("SPY", SESSION)
    assert SESSION not in [pd.Timestamp(s).date() for s in view.closes("SPY", 260).index]
    assert pd.isna(view.daily("SPY", 3)["close_c"].iloc[-1])


# ======================================================================================================================
# Bars: today's open only (5.1, 0.1 item 12)
# ======================================================================================================================


def test_todays_bar_contributes_only_its_open() -> None:
    real, _double, chain = world(n_sessions=10)
    frame = real.bars("SPY", 15)
    assert list(frame.columns) == ["session", "open", "high", "low", "close"]
    assert [pd.Timestamp(s).date() for s in frame["session"]][-1] < SESSION  # today's bar is NOT a completed bar
    assert frame.notna().all().all() and str(frame["close"].dtype) == "float64"
    ratio = real.today_open_ratio("SPY")
    assert ratio is not None
    history = _history_of(real)
    today = history.bars[history.bars["session"] == pd.Timestamp(SESSION)].iloc[0]
    previous = history.bars[history.bars["session"] < pd.Timestamp(SESSION)].iloc[-1]
    assert ratio == pytest.approx(float(today["open"]) / float(previous["close"]))
    assert real.bars("SPY", 0).empty
    with pytest.raises(ValueError, match="n must be >= 0"):
        real.bars("SPY", -1)


def test_today_open_ratio_is_none_before_the_open_plus_sixty_seconds() -> None:
    cal = xnys()
    real, _double, _chain = world(n_sessions=10)
    early = cal.open_close(SESSION)[0] + timedelta(seconds=30)
    assert real_at(real, early).today_open_ratio("SPY") is None
    assert real_at(real, cal.open_close(SESSION)[0] + timedelta(seconds=90)).today_open_ratio("SPY") is not None


# ======================================================================================================================
# Chains, side tables, events and news
# ======================================================================================================================


def test_a_chain_that_is_not_knowable_yet_raises_even_if_the_provider_offers_it() -> None:
    """INV-14: `DataView` re-checks the provider, so a provider bug raises instead of leaking a future snapshot."""
    cal = xnys()
    chain = make_chain("SPY", session=SESSION, slot=Slot.EOD, calendar=cal)
    provider = StubChainProvider([chain])
    view = DataView(
        key=chain.key,
        as_of=chain.ts - timedelta(seconds=1),
        calendar=cal,
        chains=provider,
        tables={},
        news=NullNewsSource(),
        events=TableEventSource.from_events([]),
    )
    with pytest.raises(PitViolation, match="is knowable at"):
        view.chain("SPY")
    with pytest.raises(PitViolation):
        view.spot("SPY")
    missing = DataView(
        key=SnapshotKey(session=date(2024, 5, 20), slot=Slot.EOD),
        as_of=chain.ts + timedelta(days=3),
        calendar=cal,
        chains=provider,
        tables={},
        news=NullNewsSource(),
        events=TableEventSource.from_events([]),
    )
    with pytest.raises(DataUnavailable, match="no chain snapshot"):
        missing.chain("SPY")


def test_a_missing_table_is_unavailable_and_an_empty_one_is_empty() -> None:
    cal = xnys()
    chain = make_chain("SPY", session=SESSION, calendar=cal)
    view = DataView(
        key=chain.key,
        as_of=chain.ts,
        calendar=cal,
        chains=StubChainProvider([chain]),
        tables={},
        news=NullNewsSource(),
        events=TableEventSource.from_events([]),
    )
    for call in (
        lambda: view.closes("SPY", 5),
        lambda: view.close("SPY", SESSION),
        lambda: view.daily("SPY", 5),
        lambda: view.bars("SPY", 5),
        lambda: view.today_open_ratio("SPY"),
        lambda: view.vol_index("VIX", 5),
        lambda: view.rate(),
    ):
        with pytest.raises(DataUnavailable):
            call()
    # an existing table with nothing knowable yet: an empty frame / Series, and a rate is genuinely unavailable
    real, _double, chain = world(n_sessions=8)
    before_everything = xnys().open_close(date(2024, 1, 2))[1]
    early = real_at(real, before_everything)
    assert early.closes("SPY", 5).empty and early.bars("SPY", 5).empty and early.vol_index("VIX", 5).empty
    with pytest.raises(DataUnavailable, match="no bill rate knowable"):
        early.rate()


def test_vol_index_and_rate_use_the_newest_knowable_value() -> None:
    real, _double, _chain = world(n_sessions=20)
    series = real.vol_index("VIX", 5)
    assert series.name == "VIX" and str(series.dtype) == "float64" and len(series) == 5
    # D22: the newest knowable index close at an EOD decision is the PREVIOUS session's (its gate is the next open)
    assert [pd.Timestamp(s).date() for s in series.index][-1] == xnys().prev_session(SESSION)
    assert real.rate() == pytest.approx(0.04)
    with pytest.raises(DataUnavailable, match="no volidx:NOPE"):
        real.vol_index("NOPE", 5)
    with pytest.raises(ValueError, match="n must be >= 0"):
        real.vol_index("VIX", -1)


def test_events_and_news_are_served_at_this_views_as_of() -> None:
    cal = xnys()
    meeting = cal.next_session(SESSION, 12)
    far = fomc_event(cal.next_session(SESSION, 120))  # knowable only 45 days ahead (5.1)
    covered = TableNewsSource.from_items(
        [news_item("n1", created_at=cal.open_close(SESSION)[1] - timedelta(hours=2), symbols=("SPY",))],
        [("SPY", date(2024, 5, 1), date(2024, 5, 31))],
    )
    real, _double, _chain = world(n_sessions=8, events=(fomc_event(meeting), far), news=covered)
    window = (SESSION, SESSION + timedelta(days=365))
    assert [e.event_date for e in real.events(*window)] == [meeting]  # the far meeting is not knowable yet
    assert real.events(SESSION, SESSION + timedelta(days=1)) == ()
    assert [e.event_date for e in real.events(*window, "SPY")] == [meeting]
    assert real.event_coverage() == ("fomc_decision",)
    assert [i.id for i in real.news("SPY", 24)] == ["n1"] and real.news_covered("SPY")
    assert real.news("QQQ", 24) == () and not real.news_covered("QQQ")


def test_news_off_is_distinguishable_from_a_quiet_day() -> None:
    real, _double, _chain = world(n_sessions=6)  # NullNewsSource
    assert real.news("SPY", 24) == () and real.news_covered("SPY") is False


# ======================================================================================================================
# Provenance and partition logging
# ======================================================================================================================


def test_every_read_is_recorded_once_with_the_gate_that_allowed_it() -> None:
    real, _double, chain = world(n_sessions=20)
    assert real.touched() == ()
    real.chain("SPY")
    real.closes("SPY", 5)
    real.closes("SPY", 5)  # the identical read is recorded once
    real.daily("SPY", 5)
    real.bars("SPY", 5)
    real.today_open_ratio("SPY")
    real.vol_index("VIX", 5)
    real.rate()
    real.events(SESSION, SESSION + timedelta(days=60))
    fields = [p.field for p in real.touched()]
    assert fields == [
        "chain:SPY",
        "closes:SPY",
        "daily:SPY",
        "bars:SPY",
        "bars_open:SPY",
        "volidx:VIX",
        "rates",
        f"events:{SESSION.isoformat()}:{(SESSION + timedelta(days=60)).isoformat()}:*",
    ]
    by_field = {p.field: p for p in real.touched()}
    assert by_field["chain:SPY"].source == "mirror.chain" and by_field["chain:SPY"].event_time == chain.ts
    assert by_field["chain:SPY"].knowable_at == chain.knowable_at
    assert by_field["daily:SPY"].source == "derived.daily" and by_field["volidx:VIX"].source == "cboe.VIX"
    assert by_field["rates"].source == "treasury.tbill_13w"
    assert all(p.knowable_at <= real.as_of for p in real.touched())  # INV-14: nothing was read before it was knowable
    assert len({p.payload_sha256 for p in real.touched()}) == len(real.touched())
    previous = xnys().prev_session(SESSION)
    real.close("SPY", previous)
    keyed = real.touched()[-1]
    assert keyed.field == f"close:SPY:{previous.isoformat()}" and keyed.knowable_at == xnys().open_close(previous)[1]


def test_opened_partitions_lists_the_files_the_view_actually_read() -> None:
    cal = xnys()
    chain = make_chain("SPY", session=SESSION, calendar=cal)
    history = make_history(chain, n_sessions=8, calendar=cal)
    view = build_view(
        chains=[chain],
        histories={"SPY": history},
        key=chain.key,
        as_of=chain.ts,
        partitions=("pq/enriched/mirror/SPY/year=2024.parquet",),
        daily_paths=("pq/daily/mirror/SPY.parquet",),
    )
    assert view.opened_partitions() == ()
    view.closes("SPY", 5)
    assert view.opened_partitions() == ("pq/daily/mirror/SPY.parquet",)
    view.chain("SPY")
    view.chain("SPY")
    assert view.opened_partitions() == ("pq/daily/mirror/SPY.parquet", "pq/enriched/mirror/SPY/year=2024.parquet")


def test_a_naive_as_of_is_refused() -> None:
    cal = xnys()
    chain = make_chain("SPY", session=SESSION, calendar=cal)
    with pytest.raises(ValueError, match="tz-aware"):
        DataView(
            key=chain.key,
            as_of=datetime(2024, 5, 17, 20, 0),
            calendar=cal,
            chains=StubChainProvider([chain]),
            tables={},
            news=NullNewsSource(),
            events=TableEventSource.from_events([]),
        )


def test_table_names_are_the_documented_keys() -> None:
    assert daily_table_name("SPY") == "daily:SPY" and bars_table_name("SPY") == "bars:SPY"
    assert volidx_table_name("VIX3M") == "volidx:VIX3M"
    # the FakeView world of WP00 is built on the same names
    assert make_view(("SPY",), n_sessions=6).spot("SPY") == 45_000


# ======================================================================================================================
# helpers
# ======================================================================================================================

_HISTORIES: dict[int, History] = {}


def _history_of(view: DataView) -> History:
    return _HISTORIES[id(view)]


def real_at(view: DataView, as_of: datetime) -> DataView:
    """Another `DataView` over the same tables at another `as_of` (truncation-invariance tests)."""
    clone = DataView(
        key=view.key,
        as_of=as_of,
        calendar=view.calendar,
        chains=view._chains,  # noqa: SLF001 - the test rebuilds the same view at another instant
        tables=view._tables,  # noqa: SLF001
        news=view._news,  # noqa: SLF001
        events=view._events,  # noqa: SLF001
    )
    if id(view) in _HISTORIES:
        _HISTORIES[id(clone)] = _HISTORIES[id(view)]
    return clone


_ORIGINAL_WORLD = world


def world_recording(**kwargs: object) -> tuple[DataView, FakeView, ChainSnapshot]:
    raise NotImplementedError


assert datetime(2024, 5, 17, tzinfo=UTC).tzinfo is UTC
