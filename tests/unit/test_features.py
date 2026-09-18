"""`features.py` (DESIGN.md 5.3): hand-computed goldens, session-indexed windows, robust IV rank, trading-time moves.

The fixture world is built here, not generated: 260 sessions whose closes are a GEOMETRIC series with a constant log
return `K`, so every 5.3 formula has a closed form that this file computes independently of `features.py`:

    rv20 = sqrt(252) * K            sigma_d = K            trend_z = |20K| / (K * sqrt(20)) = sqrt(20)
    move_sigma = (e^K - 1) / K      dd_52w = 0 (today is the high)      atr_pct = 0.01 by construction

Point-in-time facts the same world proves: today's high / low / close / volume are never read (poisoning them changes
nothing), a vol-index value dated today is invisible (D22), and a future `daily` row cannot reach any feature.
"""

import json
import math
from datetime import UTC, date, datetime, timedelta
from typing import Any, Final

import pandas as pd
import pytest

from jevbot.cal import XnysCalendar, trading_time, year_fraction
from jevbot.config import DataConfig
from jevbot.errors import InvariantError
from jevbot.features import (
    FeatureSet,
    compute_features,
    expected_move,
    parse_atm_term,
    pctile,
    total_variance_at,
)
from jevbot.types import Fidelity, Slot, SnapshotKey
from tests.fixtures.chain_factory import DEFAULT_SESSION, make_chain
from tests.fixtures.fake_view import BARS_COLUMNS, DAILY_COLUMNS, FakeView, fomc_event

N_SESSIONS: Final = 260
K: Final = 0.002  # constant daily log return of the generated closes
REF: Final = 45_000  # today's reference price in cents
ATM_IV_BP: Final = 1600
HOLD: Final = 20
SESSIONS_PER_YEAR: Final = 252


# ======================================================================================================================
# The hand-built world
# ======================================================================================================================


def _fidelity(slot: Slot) -> Fidelity:
    """EOD data has only the `eod` slot (Conventions); a three-slot archive is a recorder's."""
    return Fidelity.EOD_QUOTES if slot is Slot.EOD else Fidelity.RECORDED_INDICATIVE


def _closes(n: int = N_SESSIONS) -> list[int]:
    """`closes[-1] == REF`; every step is exactly `K` in log terms (up to the cent rounding)."""
    return [round(REF * math.exp(K * (index - (n - 1)))) for index in range(n)]


def _atm_term(calendar: XnysCalendar, as_of: datetime, session: date, horizons: tuple[int, ...] = (5, 10, 20)) -> list[list[float]]:
    nodes: list[list[float]] = []
    for horizon in horizons:
        close = calendar.open_close(calendar.next_session(session, horizon))[1]
        nodes.append([year_fraction(as_of, close), trading_time(calendar, as_of, close), ATM_IV_BP, ATM_IV_BP, REF])
    return nodes


def _world(
    *,
    iv30: list[int] | None = None,
    poison_today_bar: bool = False,
    future_rows: bool = False,
    slots: tuple[Slot, ...] = (Slot.EOD,),
    slot: Slot = Slot.EOD,
) -> FakeView:
    calendar = XnysCalendar()
    session = DEFAULT_SESSION
    sessions = calendar.sessions(calendar.prev_session(session, N_SESSIONS - 1), session)
    assert len(sessions) == N_SESSIONS and sessions[-1] == session
    closes = _closes()
    as_of = calendar.open_close(session)[1]
    closes_at = [calendar.open_close(day)[1] for day in sessions]
    next_opens = [calendar.next_open_after(stamp) for stamp in closes_at]
    iv30_bp = iv30 if iv30 is not None else [1000 + 4 * index for index in range(N_SESSIONS)]
    term = _atm_term(calendar, as_of, session)

    rows: list[dict[str, Any]] = []
    for index, day in enumerate(sessions):
        for row_slot in slots:
            own = day == session and row_slot is slot
            eod = row_slot is Slot.EOD
            rows.append(
                {
                    "session": day,
                    "slot": row_slot.value,
                    "px_c": closes[index],
                    "close_c": closes[index] if eod else None,
                    "close_knowable_at": closes_at[index] if eod else None,
                    "iv30_bp": iv30_bp[index],
                    "iv30_2s_bp": iv30_bp[index],
                    "iv90_bp": round(iv30_bp[index] * 1.1),
                    "skew25_bp": 100 + index,
                    "atm_term_json": json.dumps(term) if own else None,
                    "rv20_bp": 2000,
                    "spot_measure": "parity",
                    "div_unmodelled": False,
                    "basis_suspect": False,
                    "source": "mirror",
                    "knowable_at": closes_at[index],
                }
            )
    if future_rows:  # a session AFTER this one, knowable only at its own close: no feature may ever see it
        future = calendar.next_session(session)
        rows.append(
            {
                "session": future,
                "slot": Slot.EOD.value,
                "px_c": 999_999,
                "close_c": 999_999,
                "close_knowable_at": calendar.open_close(future)[1],
                "iv30_bp": 99_999,
                "iv30_2s_bp": 99_999,
                "iv90_bp": 99_999,
                "skew25_bp": 99_999,
                "atm_term_json": None,
                "rv20_bp": 99_999,
                "spot_measure": "parity",
                "div_unmodelled": False,
                "basis_suspect": False,
                "source": "mirror",
                "knowable_at": calendar.open_close(future)[1],
            }
        )
    daily = pd.DataFrame(rows, columns=list(DAILY_COLUMNS))

    dollars = [value / 100.0 for value in closes]
    prev = [dollars[0] / math.exp(K), *dollars[:-1]]
    bar_rows: list[dict[str, Any]] = []
    for index, day in enumerate(sessions):
        today = day == session
        bar_rows.append(
            {
                "session": day,
                "open": round(prev[index] * 1.001, 4),
                "high": 9_999.0 if (today and poison_today_bar) else round(dollars[index] * 1.005, 4),
                "low": 0.01 if (today and poison_today_bar) else round(dollars[index] * 0.995, 4),
                "close": 9_999.0 if (today and poison_today_bar) else dollars[index],
                "volume": 10**12 if (today and poison_today_bar) else 1_000_000,
                "knowable_at": calendar.open_close(day)[0] + timedelta(seconds=60),
                "open_knowable_at": calendar.open_close(day)[0] + timedelta(seconds=60),
                "hlcv_knowable_at": next_opens[index],
            }
        )
    bars = pd.DataFrame(bar_rows, columns=list(BARS_COLUMNS))

    vix = [15.0 + 0.01 * index for index in range(N_SESSIONS)]
    indices = {
        "VIX": vix,
        "VIX3M": [value * 1.06 for value in vix],
        "VIX9D": [value * 0.97 for value in vix],
        "VVIX": [90.0] * N_SESSIONS,
        "SKEW": [125.0] * N_SESSIONS,
    }
    vol_indices = {
        name: pd.DataFrame({"session": sessions, "close": values, "knowable_at": next_opens}) for name, values in indices.items()
    }
    rates = pd.DataFrame({"session": sessions, "rate_bp": 400, "knowable_at": next_opens})
    chain = make_chain("SPY", session=session, slot=slot, spot=REF, calendar=calendar, ts=as_of, fidelity=_fidelity(slot))
    return FakeView(
        key=SnapshotKey(session=session, slot=slot),
        as_of=as_of,
        calendar=calendar,
        fidelity=_fidelity(slot),
        chains=[chain],
        daily={"SPY": daily},
        bars={"SPY": bars},
        vol_indices=vol_indices,
        rates=rates,
        events=(fomc_event(calendar.next_session(session, 12)),),
    )


@pytest.fixture
def world() -> FakeView:
    return _world()


@pytest.fixture
def features(world: FakeView) -> FeatureSet:
    return compute_features(world, "SPY", HOLD)


# ======================================================================================================================
# The underlying block: closed forms
# ======================================================================================================================


def test_the_series_is_completed_closes_plus_ref(world: FakeView, features: FeatureSet) -> None:
    closes = list(world.closes("SPY", 260))
    assert len(closes) == N_SESSIONS - 1  # today's own close is NOT a completed session
    assert features.n_closes == N_SESSIONS and features.ref == REF
    assert closes[-1] == _closes()[-2]


def test_moving_averages_use_the_documented_windows(features: FeatureSet) -> None:
    series = [*_closes()[:-1], REF]
    assert features.ma20 == pytest.approx(sum(series[-20:]) / 20)
    assert features.ma50 == pytest.approx(sum(series[-50:]) / 50)
    assert features.ma20_prev == pytest.approx(sum(series[-25:-5]) / 20)  # the 20 closes ending 5 sessions ago
    assert features.ma20 is not None and features.ma20_prev is not None and features.ma20 > features.ma20_prev


def test_realised_vol_trend_and_move_match_their_closed_forms(features: FeatureSet) -> None:
    assert features.rv20 == pytest.approx(math.sqrt(SESSIONS_PER_YEAR) * K, rel=2e-3)
    assert features.rv5 == pytest.approx(math.sqrt(SESSIONS_PER_YEAR) * K, rel=2e-3)
    assert features.sigma_d == pytest.approx(K, rel=2e-3)
    assert features.trend_z == pytest.approx(math.sqrt(20.0), rel=2e-3)  # |20K| / (K sqrt(20))
    assert features.move_sigma == pytest.approx((math.exp(K) - 1.0) / K, rel=2e-3)
    assert features.rv_change == pytest.approx(1.0, rel=5e-3)
    assert features.dd_52w == pytest.approx(0.0, abs=1e-9)  # today IS the 1-year high
    assert features.streak == N_SESSIONS - 1  # every close is higher than the one before


def test_atr_gap_and_distance_are_hand_computable(features: FeatureSet) -> None:
    # every bar has high = 1.005 c, low = 0.995 c, so the true range is exactly 1% of the bar's own close
    assert features.atr_pct == pytest.approx(0.01, rel=1e-3)
    assert features.atr == pytest.approx(0.01 * REF, rel=1e-3)
    series = [*_closes()[:-1], REF]
    ma20 = sum(series[-20:]) / 20
    assert features.dist_ma20_atr == pytest.approx((REF - ma20) / (0.01 * REF), rel=2e-3)
    assert features.gap_sigma == pytest.approx(0.001 / K, rel=2e-3)  # today's open is 1.001 x the prior close


def test_percentile_rank_formula() -> None:
    assert pctile(5.0, [1.0, 2.0, 3.0, 4.0, 5.0], min_len=5) == 100
    assert pctile(1.0, [1.0, 2.0, 3.0, 4.0, 5.0], min_len=5) == 0
    assert pctile(3.0, [1.0, 2.0, 3.0, 4.0, 5.0], min_len=5) == 50
    assert pctile(3.0, [3.0] * 5, min_len=5) == 100  # ties count as "at or below"
    assert pctile(3.0, [1.0, 2.0], min_len=5) is None


def test_the_daily_series_drives_the_percentile_features(features: FeatureSet) -> None:
    assert features.rv20_pctile == 100  # a constant rv20_bp history: every value is <= today's
    assert features.skew_pctile == 100  # strictly rising
    assert features.vix_pctile == 100
    assert features.vvix_pctile == 100 and features.skewidx_pctile == 100


def test_vol_index_ratios_and_the_d22_lag(world: FakeView, features: FeatureSet) -> None:
    visible = list(world.vol_index("VIX", 252))
    newest_session = world.vol_index("VIX", 252).index[-1].date()
    assert newest_session == world.calendar.prev_session(world.session)  # the value dated today is knowable only at the NEXT open (D22)
    assert len(visible) == 252
    assert features.vix_term == pytest.approx(1.0 / 1.06)
    assert features.vix_near == pytest.approx(0.97)
    assert features.vix_chg_1w == pytest.approx(visible[-1] / visible[-6] - 1.0)


def test_a_missing_index_only_nulls_its_own_feature(world: FakeView) -> None:
    del world.tables.vol_indices["VVIX"]
    features = compute_features(world, "SPY", HOLD)
    assert features.vvix_pctile is None
    assert features.vix_pctile == 100 and features.required_ok


# ======================================================================================================================
# Surface: IV rank (robust range), term, change
# ======================================================================================================================


def test_surface_features_come_from_the_daily_row(features: FeatureSet) -> None:
    iv30_bp = 1000 + 4 * (N_SESSIONS - 1)
    assert features.iv30_bp == iv30_bp and features.iv30 == pytest.approx(iv30_bp / 1e4)
    assert features.iv90 == pytest.approx(round(iv30_bp * 1.1) / 1e4)
    assert features.iv_term == pytest.approx(iv30_bp / round(iv30_bp * 1.1))
    assert features.iv_chg_1w == pytest.approx(iv30_bp / (1000 + 4 * (N_SESSIONS - 6)) - 1.0)
    assert features.iv_rv == pytest.approx((iv30_bp / 1e4) / (features.rv20 or 1.0))
    assert features.skew25 == pytest.approx((100 + N_SESSIONS - 1) / 1e4)


def test_iv_rank_is_clipped_to_the_robust_range(features: FeatureSet) -> None:
    assert features.iv_rank == 100  # today is above the 98th percentile of the trailing window


def test_one_absurd_outlier_moves_iv_rank_by_less_than_three_points() -> None:
    """5.3: the 2nd / 98th percentiles are the range, so one bad snapshot cannot pin the year's low or high."""
    base = [1500 + (index % 7) * 10 for index in range(N_SESSIONS)]
    base[-1] = 1550
    clean = compute_features(_world(iv30=base), "SPY", HOLD).iv_rank
    for position in (-2, -30, -120):
        poisoned = list(base)
        poisoned[position] = 250_000  # 2500% implied vol: one corrupt snapshot
        moved = compute_features(_world(iv30=poisoned), "SPY", HOLD).iv_rank
        assert clean is not None and moved is not None
        assert abs(moved - clean) < 3, f"outlier at {position} moved iv_rank from {clean} to {moved}"
    low = list(base)
    low[-30] = 1  # and a collapse to nothing
    moved_low = compute_features(_world(iv30=low), "SPY", HOLD).iv_rank
    assert clean is not None and moved_low is not None and abs(moved_low - clean) < 3


def test_iv_rank_is_fifty_when_the_history_is_flat() -> None:
    features = compute_features(_world(iv30=[1500] * N_SESSIONS), "SPY", HOLD)
    assert features.iv_rank == 50


def test_percentile_features_need_min_history_sessions() -> None:
    long_history = DataConfig(min_history_sessions=N_SESSIONS + 50)
    features = compute_features(_world(), "SPY", HOLD, data=long_history)
    assert features.iv_rank is None and features.rv20_pctile is None and features.vix_pctile is None
    assert not features.required_ok and "iv_rank" in features.missing_required()


# ======================================================================================================================
# Expected moves: total variance in TRADING time (V13)
# ======================================================================================================================


def test_expected_moves_have_a_closed_form_on_the_fixture(world: FakeView, features: FeatureSet) -> None:
    calendar = world.calendar
    as_of = world.as_of
    first_close = calendar.open_close(calendar.next_session(world.session, 5))[1]
    tau_1 = year_fraction(as_of, first_close)
    w_first = (ATM_IV_BP / 1e4) ** 2 * tau_1
    # one session is BELOW the first node: w = w_1 * tt / tt_1 with tt = 1 session
    assert features.em_1 == pytest.approx(math.sqrt(w_first / 5.0), rel=1e-9)
    assert features.em_1_tenths == max(1, round(1000 * math.sqrt(w_first / 5.0)))
    # five sessions IS the first node: the market's own total variance to that close
    assert features.em_5 == pytest.approx(math.sqrt(w_first), rel=1e-9)
    assert features.iv_var_5 == pytest.approx(w_first, rel=1e-9)
    # the holding window is the last node (20 sessions)
    last_close = calendar.open_close(calendar.next_session(world.session, HOLD))[1]
    w_last = (ATM_IV_BP / 1e4) ** 2 * year_fraction(as_of, last_close)
    assert features.em_hold == pytest.approx(math.sqrt(w_last), rel=1e-9)


def test_total_variance_is_linear_in_trading_time_between_nodes_and_proportional_outside() -> None:
    term = ((0.02, 5.0, 1600, 1600, 45_000), (0.04, 10.0, 1600, 1600, 45_000))
    w5 = (0.16**2) * 0.02
    w10 = (0.16**2) * 0.04
    assert total_variance_at(term, 5.0) == (pytest.approx(w5), "interpolated")
    assert total_variance_at(term, 10.0) == (pytest.approx(w10), "interpolated")
    middle = total_variance_at(term, 7.5)
    assert middle is not None and middle[0] == pytest.approx((w5 + w10) / 2) and middle[1] == "interpolated"
    below = total_variance_at(term, 1.0)
    assert below is not None and below[0] == pytest.approx(w5 / 5.0) and below[1] == "extrapolated"
    beyond = total_variance_at(term, 20.0)
    assert beyond is not None and beyond[0] == pytest.approx(w10 * 2.0) and beyond[1] == "extrapolated"
    assert total_variance_at((), 3.0) is None
    assert total_variance_at(term, 0.0) == (0.0, "interpolated")


def _weekly_term(calendar: XnysCalendar, session: date, *, iv_bp: int, weeks: int = 6, proportional: bool = False) -> list[list[float]]:
    """Weekly Friday expiries priced at a CONSTANT annualised IV (`proportional=False`), or at an IV whose total
    variance is proportional to trading time (`proportional=True`)."""
    as_of = calendar.open_close(session)[1]
    nodes: list[list[float]] = []
    day = session
    for _ in range(weeks):
        day += timedelta(days=1)
        while day.weekday() != 4:  # the next Friday
            day += timedelta(days=1)
        expiry = calendar.prev_or_same_session(day)
        if expiry <= session:
            continue
        close = calendar.open_close(expiry)[1]
        tau = year_fraction(as_of, close)
        tt = trading_time(calendar, as_of, close)
        if proportional:  # w = k * tt  =>  iv = sqrt(k * tt / tau)
            k = (iv_bp / 1e4) ** 2 / SESSIONS_PER_YEAR * 365.0 / 5.0 * 5.0 / 365.0 * SESSIONS_PER_YEAR
            iv = math.sqrt(k * tt / tau / SESSIONS_PER_YEAR * SESSIONS_PER_YEAR)
            node_bp = round(iv * 1e4)
        else:
            node_bp = iv_bp
        nodes.append([tau, tt, node_bp, node_bp, 45_000])
    return nodes


def test_em_1_on_a_friday_equals_a_tuesday_on_a_session_proportional_fixture() -> None:
    """V13: with total variance proportional to SESSIONS, a Friday's one-session move is not inflated by the weekend."""
    calendar = XnysCalendar()
    friday, tuesday = date(2024, 5, 17), date(2024, 5, 14)
    assert friday.weekday() == 4 and tuesday.weekday() == 1
    values: dict[str, float] = {}
    integers: dict[str, int] = {}
    for name, session in (("friday", friday), ("tuesday", tuesday)):
        term = parse_atm_term(_weekly_term(calendar, session, iv_bp=4000, proportional=True))
        as_of = calendar.open_close(session)[1]
        tt = trading_time(calendar, as_of, calendar.open_close(calendar.next_session(session))[1])
        assert tt == pytest.approx(1.0)
        found = expected_move(term, tt)
        assert found is not None
        values[name], integers[name] = found[0], found[1]
    assert abs(values["friday"] / values["tuesday"] - 1.0) <= 0.02
    assert abs(integers["friday"] - integers["tuesday"]) <= 1


def test_em_1_friday_to_tuesday_ratio_stays_under_1_20_on_a_calendar_flat_fixture() -> None:
    """The same fixture under CALENDAR scaling would give sqrt(3) = 1.73 (the reason V13 exists)."""
    calendar = XnysCalendar()
    friday, tuesday = date(2024, 5, 17), date(2024, 5, 14)
    moves: dict[str, tuple[float, int]] = {}
    for name, session in (("friday", friday), ("tuesday", tuesday)):
        term = parse_atm_term(_weekly_term(calendar, session, iv_bp=4000))
        as_of = calendar.open_close(session)[1]
        tt = trading_time(calendar, as_of, calendar.open_close(calendar.next_session(session))[1])
        found = expected_move(term, tt)
        assert found is not None
        moves[name] = (found[0], found[1])
    ratio = moves["friday"][0] / moves["tuesday"][0]
    assert 1.0 < ratio <= 1.20
    assert moves["friday"][1] / moves["tuesday"][1] <= 1.20
    # the counterfactual: allocating the same total variance in CALENDAR days (3 days over the weekend, 1 on a Tuesday)
    assert math.sqrt(3.0) > 1.70


def test_atm_term_parsing_rejects_a_malformed_payload() -> None:
    assert parse_atm_term(None) == () and parse_atm_term("") == () and parse_atm_term(float("nan")) == ()
    assert parse_atm_term("[[0.1, 25.0, 1600, 1600, 45173]]") == ((0.1, 25.0, 1600, 1600, 45173),)
    with pytest.raises(InvariantError):
        parse_atm_term("not json")
    with pytest.raises(InvariantError):
        parse_atm_term("[[0.1, 25.0, 1600]]")
    with pytest.raises(InvariantError):
        parse_atm_term('{"a": 1}')
    # nodes come back ordered by trading time whatever the file order
    unordered = parse_atm_term([[0.2, 10.0, 1, 1, 1], [0.1, 5.0, 2, 2, 2]])
    assert [node[1] for node in unordered] == [5.0, 10.0]


def test_expected_move_integer_is_at_least_one_tenth() -> None:
    term = ((1e-6, 1.0, 1, 1, 45_000),)
    found = expected_move(term, 1.0)
    assert found is not None and found[1] == 1  # max(1, round(...)): a zero expected move is never shown


# ======================================================================================================================
# Point-in-time guarantees
# ======================================================================================================================


def test_todays_high_low_close_and_volume_are_provably_unused() -> None:
    clean = compute_features(_world(), "SPY", HOLD)
    poisoned = compute_features(_world(poison_today_bar=True), "SPY", HOLD)
    assert clean == poisoned


def test_a_future_daily_row_reaches_nothing() -> None:
    clean = compute_features(_world(), "SPY", HOLD)
    with_future = compute_features(_world(future_rows=True), "SPY", HOLD)
    assert clean == with_future


def test_windows_index_by_session_whether_the_archive_holds_one_slot_or_three() -> None:
    """5.4: a `dec` view reads past `dec` rows where they exist and the session's `eod` row otherwise - same numbers."""
    collapsed = compute_features(_world(), "SPY", HOLD)
    three_slots = compute_features(_world(slots=(Slot.DEC, Slot.EXEC, Slot.EOD), slot=Slot.DEC), "SPY", HOLD)
    for name in ("iv_rank", "iv_chg_1w", "skew_pctile", "rv20_pctile", "iv30_bp", "em_1_tenths", "em_5_tenths"):
        assert getattr(collapsed, name) == getattr(three_slots, name), name


def test_required_features_and_the_missing_list() -> None:
    features = compute_features(_world(), "SPY", HOLD)
    assert features.required_ok and features.missing_required() == ()
    empty = FeatureSet(underlying="SPY", hold_sessions=HOLD, n_closes=0, ref=None)
    assert not empty.required_ok
    assert set(empty.missing_required()) >= {"ref", "ma20", "rv20", "iv30", "iv_rank", "em_1_tenths"}
    assert empty.raw()["ref"] == "None" and "underlying" not in empty.raw()


def test_a_short_history_leaves_the_underlying_block_empty() -> None:
    world = _world()
    short = world.at(as_of=world.as_of)
    short.tables.daily["SPY"] = short.tables.daily["SPY"].tail(40).reset_index(drop=True)
    features = compute_features(short, "SPY", HOLD)
    assert features.ma20 is None and features.rv20 is None and not features.required_ok


def test_compute_features_refuses_a_non_positive_holding_window(world: FakeView) -> None:
    with pytest.raises(InvariantError):
        compute_features(world, "SPY", 0)


def test_features_are_pure(world: FakeView) -> None:
    first = compute_features(world, "SPY", HOLD)
    second = compute_features(world, "SPY", HOLD)
    assert first == second
    assert datetime.now(UTC).year >= 2024  # the fixture carries no wall clock: the two calls above are identical
