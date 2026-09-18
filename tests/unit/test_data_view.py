"""`data/view.py`: the single point-in-time gate (DESIGN.md 3.2, 5.1, 5.4; INV-14; 15.1).

The reference for the read semantics is `tests/fixtures/fake_view.FakeView`, WP00's in-memory double of this very class: the
first test drives both over the SAME tables and demands identical answers, so the double every other package is unit-tested
against cannot drift from the implementation the engine runs.
"""

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from jevbot.cal import XnysCalendar
from jevbot.data.series import NullNewsSource, PitTable, TableEventSource, TableNewsSource
from jevbot.data.store import BARS_COLUMN_KNOWABLE, DAILY_COLUMN_KNOWABLE, DAILY_COLUMNS, DAILY_KEY
from jevbot.data.view import DataView, bars_table_name, daily_table_name, volidx_table_name
from jevbot.errors import DataError, DataUnavailable, PitViolation
from jevbot.types import ChainSnapshot, Fidelity, NewsItem, ScheduledEvent, Slot, SnapshotKey
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
NewsLike = TableNewsSource | NullNewsSource
# deliberately naive (built from a tz-aware instant so the DTZ lint stays on): the view must refuse it
NAIVE: datetime = datetime(2024, 5, 17, 20, 0, tzinfo=UTC).replace(tzinfo=None)


class StubChainProvider:
    """`ChainProvider` over a dict of snapshots. `opened_partitions` is the optional 13.1 hook a real provider implements."""

    def __init__(self, chains: list[ChainSnapshot], *, source: str = "mirror", partitions: tuple[str, ...] = ()) -> None:
        self._chains = {(c.underlying, c.key): c for c in chains}
        self._source = source
        self._partitions = partitions

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
        return self._chains.get((underlying, key))

    def manifest_hash(self) -> str:
        return "stub-manifest"

    def opened_partitions(self) -> tuple[str, ...]:
        return self._partitions


@dataclass(frozen=True)
class World:
    """One coherent data world, readable through `DataView` (`view`) and through the WP00 double (`double`)."""

    chain: ChainSnapshot
    history: History
    key: SnapshotKey
    as_of: datetime
    vol: dict[str, pd.DataFrame]
    rates: pd.DataFrame
    events: tuple[ScheduledEvent, ...]
    news: NewsLike
    cal: XnysCalendar
    partitions: tuple[str, ...] = ()
    daily_paths: tuple[str, ...] = ()

    @property
    def view(self) -> DataView:
        tables: dict[str, PitTable] = {
            daily_table_name(self.history.underlying): PitTable(
                self.history.daily,
                name=daily_table_name(self.history.underlying),
                key=DAILY_KEY,
                column_knowable=DAILY_COLUMN_KNOWABLE,
                paths=self.daily_paths,
            ),
            bars_table_name(self.history.underlying): PitTable(
                self.history.bars, name=bars_table_name(self.history.underlying), key="session", column_knowable=BARS_COLUMN_KNOWABLE
            ),
        }
        for name, frame in self.vol.items():
            tables[volidx_table_name(name)] = PitTable(frame, name=volidx_table_name(name))
        tables["rates"] = PitTable(self.rates, name="rates")
        return DataView(
            key=self.key,
            as_of=self.as_of,
            calendar=self.cal,
            chains=StubChainProvider([self.chain], partitions=self.partitions),
            tables=tables,
            news=self.news,
            events=TableEventSource.from_events(self.events),
        )

    @property
    def double(self) -> FakeView:
        return FakeView(
            key=self.key,
            as_of=self.as_of,
            calendar=self.cal,
            fidelity=self.chain.fidelity,
            chains=[self.chain],
            daily={self.history.underlying: self.history.daily},
            bars={self.history.underlying: self.history.bars},
            vol_indices=self.vol,
            rates=self.rates,
            events=self.events,
        )

    def at(self, as_of: datetime) -> DataView:
        """The same tables at another instant (truncation-invariance and gating tests)."""
        return replace(self, as_of=as_of).view

    def with_daily(self, daily: pd.DataFrame) -> "World":
        return replace(self, history=History(self.history.underlying, daily, self.history.bars, self.history.iv30_bp))


def world(
    *,
    slot: Slot = Slot.EOD,
    fidelity: Fidelity = Fidelity.EOD_QUOTES,
    n_sessions: int = 40,
    slots: tuple[Slot, ...] | None = None,
    events: tuple[ScheduledEvent, ...] | None = None,
    news: NewsLike | None = None,
    partitions: tuple[str, ...] = (),
    daily_paths: tuple[str, ...] = (),
) -> World:
    cal = xnys()
    history_slots = slots if slots is not None else ((Slot.EOD,) if slot is Slot.EOD else (slot, Slot.EOD))
    chain = make_chain("SPY", session=SESSION, slot=slot, fidelity=fidelity, calendar=cal)
    history = make_history(chain, n_sessions=n_sessions, slots=history_slots, calendar=cal)
    vol = make_vol_indices(history.iv30_bp, calendar=cal)
    rates = make_rates([pd.Timestamp(s).date() for s in history.iv30_bp.index], chain.rate, calendar=cal)
    return World(
        chain=chain,
        history=history,
        key=chain.key,
        as_of=chain.ts,
        vol=vol,
        rates=rates,
        events=(fomc_event(cal.next_session(SESSION, 12)),) if events is None else events,
        news=news if news is not None else NullNewsSource(),
        cal=cal,
        partitions=partitions,
        daily_paths=daily_paths,
    )


def bare_view(chain: ChainSnapshot, as_of: datetime, key: SnapshotKey | None = None) -> DataView:
    """A view with a chain provider and NO tables at all."""
    return DataView(
        key=key if key is not None else chain.key,
        as_of=as_of,
        calendar=xnys(),
        chains=StubChainProvider([chain]),
        tables={},
        news=NullNewsSource(),
        events=TableEventSource.from_events([]),
    )


# ======================================================================================================================
# DataView and the WP00 double answer identically
# ======================================================================================================================


def test_data_view_answers_exactly_like_the_market_view_double() -> None:
    w = world()
    real, double = w.view, w.double
    assert real.key == double.key and real.session == double.session == SESSION
    assert real.as_of == double.as_of == w.chain.ts and real.fidelity is double.fidelity
    assert real.chain("SPY").content_hash == double.chain("SPY").content_hash
    assert real.spot("SPY") == double.spot("SPY") == w.chain.spot
    pd.testing.assert_series_equal(real.closes("SPY", 30), double.closes("SPY", 30))
    pd.testing.assert_frame_equal(real.daily("SPY", 30), double.daily("SPY", 30))
    pd.testing.assert_frame_equal(real.bars("SPY", 15), double.bars("SPY", 15))
    pd.testing.assert_series_equal(real.vol_index("VIX", 20), double.vol_index("VIX", 20))
    assert real.rate() == double.rate() == w.chain.rate
    assert real.today_open_ratio("SPY") == double.today_open_ratio("SPY")
    previous = w.cal.prev_session(SESSION)
    assert real.close("SPY", previous) == double.close("SPY", previous)
    window = (SESSION, SESSION + timedelta(days=60))
    assert real.events(*window) == double.events(*window)
    assert real.event_coverage() == double.event_coverage() == ("fomc_decision",)


def test_data_view_is_a_market_view_at_runtime() -> None:
    real = world().view
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
    w = world(n_sessions=12)
    frame = w.view.daily("SPY", 5)
    assert list(frame.columns) == list(DAILY_COLUMNS) and len(frame) == 5
    sessions = [pd.Timestamp(s).date() for s in frame["session"]]
    assert sessions == sorted(sessions) and len(set(sessions)) == 5 and sessions[-1] == SESSION
    assert (frame["slot"] == Slot.EOD.value).all()
    assert int(frame["px_c"].iloc[-1]) == w.chain.spot  # the snapshot's own row is last
    assert w.view.daily("SPY", 1)["session"].iloc[0] == pd.Timestamp(SESSION)  # n = 1 is the own row alone
    assert len(w.view.daily("SPY", 500)) == 12  # more than the archive holds is simply the archive


def test_a_three_slot_archive_and_the_same_data_collapsed_give_equal_daily_frames() -> None:
    """DESIGN 5.4 / 15.1: the read must be identical whether the archive holds 1 or 3 slots per session, so that every
    look-back window of 5.3 (`iv_rank`, `iv_chg_1w`, `skew_pctile`, `rv20_pctile`) indexes by SESSION in every mode."""
    w = world(slot=Slot.DEC, fidelity=Fidelity.RECORDED_INDICATIVE, n_sessions=30, slots=(Slot.DEC, Slot.EXEC, Slot.EOD))
    assert set(w.history.daily["slot"]) == {"dec", "exec", "eod"} and len(w.history.daily) == 90

    three = w.view.daily("SPY", 30)
    assert len(three) == 30 and (three["slot"] == "dec").all()
    assert three["close_c"].isna().all()  # a dec row never carries a close (5.4)

    collapsed = w.history.daily[w.history.daily["slot"] == "dec"].reset_index(drop=True)
    pd.testing.assert_frame_equal(w.with_daily(collapsed).view.daily("SPY", 30), three)


def test_a_mirror_era_session_falls_back_to_its_eod_row_seen_from_a_dec_view() -> None:
    """The real archive is mixed: mirror sessions have only an `eod` row, recorded days have three. A `dec` decision takes
    the `dec` row where it exists and that session's `eod` row otherwise; a session with neither is skipped (5.4)."""
    w = world(slot=Slot.DEC, fidelity=Fidelity.RECORDED_INDICATIVE, n_sessions=10, slots=(Slot.DEC, Slot.EXEC, Slot.EOD))
    daily = w.history.daily
    sessions = sorted({pd.Timestamp(s).date() for s in daily["session"]})
    mirror_era, recorded_era = sessions[:6], sessions[6:]
    is_mirror = daily["session"].isin([pd.Timestamp(s) for s in mirror_era])
    mixed = daily[~(is_mirror & daily["slot"].isin(["dec", "exec"]))].reset_index(drop=True)

    frame = w.with_daily(mixed).view.daily("SPY", 10)
    assert [pd.Timestamp(s).date() for s in frame["session"]] == sessions
    assert list(frame["slot"]) == ["eod"] * len(mirror_era) + ["dec"] * len(recorded_era)
    # the mirror-era rows bring their close with them; the dec rows do not
    assert frame["close_c"].iloc[: len(mirror_era)].notna().all() and frame["close_c"].iloc[len(mirror_era) :].isna().all()

    # a session with NEITHER a dec nor an eod row is skipped entirely
    dropped = sessions[2]
    thinner = mixed[mixed["session"] != pd.Timestamp(dropped)].reset_index(drop=True)
    frame = w.with_daily(thinner).view.daily("SPY", 10)
    assert dropped not in [pd.Timestamp(s).date() for s in frame["session"]] and len(frame) == len(sessions) - 1


def test_daily_needs_the_snapshots_own_row_and_refuses_a_future_one() -> None:
    w = world(n_sessions=6)
    with pytest.raises(PitViolation, match="is knowable at"):
        w.at(w.chain.ts - timedelta(seconds=1)).daily("SPY", 3)
    other = replace(w, key=SnapshotKey(session=SESSION, slot=Slot.DEC))
    with pytest.raises(DataUnavailable, match="no row for key"):
        other.view.daily("SPY", 3)
    with pytest.raises(ValueError, match="n must be >= 1"):
        w.view.daily("SPY", 0)


# ======================================================================================================================
# closes() / close(): completed sessions, one value per session, keyed reads raise
# ======================================================================================================================


def test_closes_stop_before_this_session_and_carry_one_value_per_session() -> None:
    w = world(n_sessions=12)
    real = w.view
    series = real.closes("SPY", 260)
    assert series.name == "close_c" and str(series.dtype) == "int64" and series.index.name == "session"
    sessions = [pd.Timestamp(s).date() for s in series.index]
    assert len(sessions) == len(set(sessions)) == 11 and max(sessions) < SESSION  # COMPLETED sessions only
    assert list(real.closes("SPY", 3)) == list(series)[-3:]
    assert real.closes("SPY", 0).empty and str(real.closes("SPY", 0).dtype) == "int64"
    with pytest.raises(ValueError, match="n must be >= 0"):
        real.closes("SPY", -1)
    assert real.close("SPY", sessions[-1]) == int(series.iloc[-1])
    # the mirror's own eod snapshot IS the close of its session: knowable exactly at as_of, never a moment before
    assert real.close("SPY", SESSION) == w.chain.spot
    with pytest.raises(PitViolation):
        w.at(w.chain.ts - timedelta(seconds=1)).close("SPY", SESSION)
    with pytest.raises(DataUnavailable):
        real.close("SPY", date(2024, 5, 18))


def test_a_three_slot_archive_still_yields_exactly_one_close_per_session() -> None:
    """`close_c` lives on the `eod` row only, so `closes()` cannot double-count a session with three snapshots (5.4)."""
    w = world(slot=Slot.DEC, fidelity=Fidelity.RECORDED_INDICATIVE, n_sessions=20, slots=(Slot.DEC, Slot.EXEC, Slot.EOD))
    series = w.view.closes("SPY", 260)
    sessions = [pd.Timestamp(s).date() for s in series.index]
    assert len(sessions) == len(set(sessions)) == 19 and max(sessions) < SESSION


def test_a_close_that_is_not_recorded_yet_is_unavailable_and_absent_from_closes() -> None:
    w = world(n_sessions=8)
    daily = w.history.daily.copy()
    daily.loc[daily["session"] == daily["session"].max(), ["close_c", "close_knowable_at"]] = [pd.NA, pd.NaT]
    view = w.with_daily(daily).view
    with pytest.raises(DataUnavailable, match="has not been recorded yet"):
        view.close("SPY", SESSION)
    assert SESSION not in [pd.Timestamp(s).date() for s in view.closes("SPY", 260).index]
    assert pd.isna(view.daily("SPY", 3)["close_c"].iloc[-1])


# ======================================================================================================================
# Bars: today's open only (5.1, 0.1 item 12)
# ======================================================================================================================


def test_todays_bar_contributes_only_its_open() -> None:
    w = world(n_sessions=10)
    real = w.view
    frame = real.bars("SPY", 15)
    assert list(frame.columns) == ["session", "open", "high", "low", "close"]
    assert [pd.Timestamp(s).date() for s in frame["session"]][-1] < SESSION  # today's bar is NOT a completed bar
    assert frame.notna().all().all() and str(frame["close"].dtype) == "float64"
    ratio = real.today_open_ratio("SPY")
    assert ratio is not None
    today = w.history.bars[w.history.bars["session"] == pd.Timestamp(SESSION)].iloc[0]
    previous = w.history.bars[w.history.bars["session"] < pd.Timestamp(SESSION)].iloc[-1]
    assert ratio == pytest.approx(float(today["open"]) / float(previous["close"]))
    assert real.bars("SPY", 0).empty
    with pytest.raises(ValueError, match="n must be >= 0"):
        real.bars("SPY", -1)


def test_today_open_ratio_is_none_before_the_open_plus_sixty_seconds() -> None:
    w = world(n_sessions=10)
    opened = w.cal.open_close(SESSION)[0]
    assert w.at(opened + timedelta(seconds=30)).today_open_ratio("SPY") is None
    assert w.at(opened + timedelta(seconds=90)).today_open_ratio("SPY") is not None


# ======================================================================================================================
# Chains, side tables, events and news
# ======================================================================================================================


def test_a_chain_that_is_not_knowable_yet_raises_even_if_the_provider_offers_it() -> None:
    """INV-14: `DataView` re-checks the provider, so a provider bug raises instead of leaking a future snapshot."""
    chain = make_chain("SPY", session=SESSION, slot=Slot.EOD, calendar=xnys())
    view = bare_view(chain, chain.ts - timedelta(seconds=1))
    with pytest.raises(PitViolation, match="is knowable at"):
        view.chain("SPY")
    with pytest.raises(PitViolation):
        view.spot("SPY")
    absent = bare_view(chain, chain.ts + timedelta(days=3), SnapshotKey(session=date(2024, 5, 20), slot=Slot.EOD))
    with pytest.raises(DataUnavailable, match="no chain snapshot"):
        absent.chain("SPY")


def test_a_missing_table_is_unavailable_and_an_empty_one_is_empty() -> None:
    chain = make_chain("SPY", session=SESSION, calendar=xnys())
    view = bare_view(chain, chain.ts)
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
    # an existing table with nothing knowable yet: an empty frame / Series, and the rate is genuinely unavailable
    w = world(n_sessions=8)
    early = w.at(w.cal.open_close(date(2024, 1, 2))[1])
    assert early.closes("SPY", 5).empty and early.bars("SPY", 5).empty and early.vol_index("VIX", 5).empty
    with pytest.raises(DataUnavailable, match="no bill rate knowable"):
        early.rate()


def test_vol_index_and_rate_use_the_newest_knowable_value() -> None:
    w = world(n_sessions=20)
    series = w.view.vol_index("VIX", 5)
    assert series.name == "VIX" and str(series.dtype) == "float64" and len(series) == 5
    # D22: the newest knowable index close at an EOD decision is the PREVIOUS session's (its gate is the next open)
    assert [pd.Timestamp(s).date() for s in series.index][-1] == w.cal.prev_session(SESSION)
    assert w.view.rate() == pytest.approx(0.04)
    with pytest.raises(DataUnavailable, match="no `volidx:NOPE` table"):
        w.view.vol_index("NOPE", 5)
    with pytest.raises(ValueError, match="n must be >= 0"):
        w.view.vol_index("VIX", -1)


def test_events_and_news_are_served_at_this_views_as_of() -> None:
    cal = xnys()
    meeting = cal.next_session(SESSION, 12)
    far = fomc_event(cal.next_session(SESSION, 120))  # a scheduled meeting is knowable only 45 days ahead (5.1)
    item: NewsItem = news_item("n1", "ETF flows steady", created_at=cal.open_close(SESSION)[1] - timedelta(hours=2), symbols=("SPY",))
    covered = TableNewsSource.from_items([item], [("SPY", date(2024, 5, 1), date(2024, 5, 31))])
    real = world(n_sessions=8, events=(fomc_event(meeting), far), news=covered).view
    window = (SESSION, SESSION + timedelta(days=365))
    assert [e.event_date for e in real.events(*window)] == [meeting]  # the far meeting is not knowable yet
    assert real.events(SESSION, SESSION + timedelta(days=1)) == ()
    assert [e.event_date for e in real.events(*window, "SPY")] == [meeting]
    assert real.event_coverage() == ("fomc_decision",)
    assert [i.id for i in real.news("SPY", 24)] == ["n1"] and real.news_covered("SPY")
    assert real.news("QQQ", 24) == () and not real.news_covered("QQQ")


def test_news_off_is_distinguishable_from_a_quiet_day() -> None:
    real = world(n_sessions=6).view  # NullNewsSource
    assert real.news("SPY", 24) == () and real.news_covered("SPY") is False


# ======================================================================================================================
# Provenance and partition logging
# ======================================================================================================================


def test_every_read_is_recorded_once_with_the_gate_that_allowed_it() -> None:
    w = world(n_sessions=20)
    real = w.view
    assert real.touched() == ()
    real.chain("SPY")
    real.closes("SPY", 5)
    real.closes("SPY", 5)  # the identical read is recorded once
    real.daily("SPY", 5)
    real.bars("SPY", 5)
    real.today_open_ratio("SPY")
    real.vol_index("VIX", 5)
    real.rate()
    window = (SESSION, SESSION + timedelta(days=60))
    real.events(*window)
    assert [p.field for p in real.touched()] == [
        "chain:SPY",
        "closes:SPY",
        "daily:SPY",
        "bars:SPY",
        "bars_open:SPY",
        "volidx:VIX",
        "rates",
        f"events:{window[0].isoformat()}:{window[1].isoformat()}:*",
    ]
    by_field = {p.field: p for p in real.touched()}
    assert by_field["chain:SPY"].source == "mirror.chain" and by_field["chain:SPY"].event_time == w.chain.ts
    assert by_field["chain:SPY"].knowable_at == w.chain.knowable_at
    assert by_field["daily:SPY"].source == "derived.daily" and by_field["volidx:VIX"].source == "cboe.VIX"
    assert by_field["rates"].source == "treasury.tbill_13w" and by_field["bars:SPY"].source == "raw.bars"
    assert all(p.knowable_at <= real.as_of for p in real.touched())  # INV-14: nothing was read before it was knowable
    assert len({p.payload_sha256 for p in real.touched()}) == len(real.touched())
    previous = w.cal.prev_session(SESSION)
    real.close("SPY", previous)
    keyed = real.touched()[-1]
    assert keyed.field == f"close:SPY:{previous.isoformat()}" and keyed.knowable_at == w.cal.open_close(previous)[1]


def test_opened_partitions_lists_the_files_the_view_actually_read() -> None:
    w = world(
        n_sessions=8,
        partitions=("pq/enriched/mirror/SPY/year=2024.parquet",),
        daily_paths=("pq/daily/mirror/SPY.parquet",),
    )
    view = w.view
    assert view.opened_partitions() == ()
    view.closes("SPY", 5)
    assert view.opened_partitions() == ("pq/daily/mirror/SPY.parquet",)
    view.chain("SPY")
    view.chain("SPY")
    assert view.opened_partitions() == ("pq/daily/mirror/SPY.parquet", "pq/enriched/mirror/SPY/year=2024.parquet")


def test_a_provider_without_the_partition_hook_still_works() -> None:
    """`opened_partitions` is optional on a `ChainProvider` (the synthetic provider has no files at all)."""

    class Minimal:
        def __init__(self, chain: ChainSnapshot) -> None:
            self._chain = chain

        @property
        def fidelity(self) -> Fidelity:
            return self._chain.fidelity

        @property
        def source(self) -> str:
            return "synthetic"

        def underlyings(self) -> tuple[str, ...]:
            return (self._chain.underlying,)

        def keys(self, underlying: str, start: date, end: date) -> list[SnapshotKey]:
            return [self._chain.key]

        def get_chain(self, underlying: str, key: SnapshotKey) -> ChainSnapshot | None:
            return self._chain if (underlying, key) == (self._chain.underlying, self._chain.key) else None

        def manifest_hash(self) -> str:
            return "minimal"

    chain = make_chain("SPY", session=SESSION, calendar=xnys())
    view = DataView(
        key=chain.key,
        as_of=chain.ts,
        calendar=xnys(),
        chains=Minimal(chain),
        tables={},
        news=NullNewsSource(),
        events=TableEventSource.from_events([]),
    )
    assert view.chain("SPY").content_hash == chain.content_hash
    assert view.opened_partitions() == ()
    assert view.touched()[0].source == "synthetic.chain"


def test_a_gap_in_the_previous_close_gives_no_open_ratio() -> None:
    w = world(n_sessions=8)
    bars = w.history.bars.copy()
    bars.loc[bars["session"] < pd.Timestamp(SESSION), "close"] = 0.0  # a corrupt vendor row, not a price
    broken = replace(w, history=History("SPY", w.history.daily, bars, w.history.iv30_bp))
    assert broken.view.today_open_ratio("SPY") is None


def test_a_mis_typed_close_column_is_refused_rather_than_rounded() -> None:
    """`close_c` is integer cents (Conventions). A float column in the parquet is a data error, never a silent int()."""
    w = world(n_sessions=8)
    daily = w.history.daily.copy()
    daily["close_c"] = daily["close_c"].astype("float64")
    with pytest.raises(DataError, match="not an integer number of cents"):
        w.with_daily(daily).view.close("SPY", w.cal.prev_session(SESSION))


def test_a_naive_as_of_is_refused() -> None:
    chain = make_chain("SPY", session=SESSION, calendar=xnys())
    with pytest.raises(ValueError, match="tz-aware"):
        bare_view(chain, NAIVE)


def test_table_names_are_the_documented_keys() -> None:
    assert daily_table_name("SPY") == "daily:SPY" and bars_table_name("SPY") == "bars:SPY"
    assert volidx_table_name("VIX3M") == "volidx:VIX3M"
    assert make_view(("SPY",), n_sessions=6).spot("SPY") == 45_000  # the WP00 world uses the same names
