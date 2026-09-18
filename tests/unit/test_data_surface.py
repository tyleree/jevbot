"""`data/surface.py`: parity forwards and spot, enrichment, the ATM term, the smile fit and the implied digital
(DESIGN.md 3.7, 5.2, 5.3, 6.4; 15.1).

Every number that matters is checked against an INDEPENDENT hand computation - put-call parity written out for one strike,
the normal cdf from `math.erfc`, the planted smile of the chain factory, the total-variance interpolation spelt from the
spec text - never by re-running the function under test.
"""

import math
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from jevbot.cal import XnysCalendar, trading_time, year_fraction
from jevbot.config import DataConfig
from jevbot.data import surface
from jevbot.data.surface import SurfaceNode, TermNode
from jevbot.errors import DataError, DataUnavailable
from jevbot.types import CHAIN_COLUMNS, ChainSnapshot, ScheduledEvent
from tests.fixtures.chain_factory import (
    GOOD_FRIDAY_SESSION,
    SATURDAY_MONTHLY_SESSION,
    Smile,
    good_friday_chain,
    make_chain,
    saturday_monthly_chain,
    set_quote,
    xnys,
)

FLAT = Smile(skew=0.0, curvature=0.0)  # a smile with no skew: `dw/dk = 0`, so the digital must be exactly N(d2)


@pytest.fixture(scope="module")
def cal() -> XnysCalendar:
    return xnys()


def phi(x: float) -> float:
    """The standard normal cdf from `math.erfc` - an implementation independent of `scipy` and of `bs.py`."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def tau_of(chain: ChainSnapshot, expiry: date, cal: XnysCalendar) -> float:
    return year_fraction(chain.ts, cal.open_close(cal.prev_or_same_session(expiry))[1])


def raw_table(chain: ChainSnapshot) -> pd.DataFrame:
    """The chain table as a RAW (unenriched) one: what `data derive` hands to `parity_forwards` / `enrich`."""
    return chain.table.drop(columns=["iv", "delta", "vega", "fwd", "last_session", "dte"]).reset_index(drop=True)


def exdiv(underlying: str, ex_date: date, amount: int, *, knowable: datetime | None = None) -> ScheduledEvent:
    return ScheduledEvent(
        kind="ex_dividend",
        event_date=ex_date,
        underlying=underlying,
        amount_cents=amount,
        scheduled=True,
        knowable_at=knowable if knowable is not None else datetime(2000, 1, 1, tzinfo=XnysCalendar().open_close(ex_date)[1].tzinfo),
        knowable_rule="exdiv_minus_14d_assumption",
        source_url="alpaca:corporate-actions",
        fetched_at=datetime(2026, 1, 1, tzinfo=XnysCalendar().open_close(ex_date)[1].tzinfo),
    )


# ======================================================================================================================
# 5.2 Parity forwards
# ======================================================================================================================


def test_parity_forwards_recover_the_planted_forward_of_every_expiry(cal: XnysCalendar) -> None:
    chain = make_chain()
    forwards = surface.parity_forwards(raw_table(chain), chain.rate, chain.ts, cal)
    assert set(forwards) == set(chain.expiries())
    for expiry in chain.expiries():
        assert abs(forwards[expiry] - chain.forward(expiry)) <= 1, expiry  # within the cent that rounding can cost


def test_one_strike_of_put_call_parity_written_out_by_hand(cal: XnysCalendar) -> None:
    """`F = K + exp(r T_E) (C_mid - P_mid)` with `T_E = year_fraction(ts, close(last_session(E)))`, from the raw quotes."""
    chain = make_chain()
    expiry = chain.expiries()[2]
    table = raw_table(chain)
    block = table[pd.to_datetime(table["expiry"]) == pd.Timestamp(expiry)]
    forward = chain.forward(expiry)
    nearest = block.iloc[(block["strike_milli"] / 10 - forward).abs().argmin()]
    strike_milli = int(nearest["strike_milli"])
    call = block[(block["strike_milli"] == strike_milli) & (block["right"] == "C")].iloc[0]
    put = block[(block["strike_milli"] == strike_milli) & (block["right"] == "P")].iloc[0]
    tau = tau_of(chain, expiry, cal)
    by_hand = strike_milli / 10 + math.exp(chain.rate * tau) * ((int(call["bid"]) + int(call["ask"])) - (int(put["bid"]) + int(put["ask"]))) / 2
    assert abs(by_hand - forward) <= 1.0
    assert abs(surface.parity_forwards(table, chain.rate, chain.ts, cal)[expiry] - by_hand) <= 1.0


def test_a_saturday_dated_monthly_discounts_to_the_friday_close(cal: XnysCalendar) -> None:
    """Conventions / INV-11: `expiry` is identity only. `close(2014-06-21)` does not exist; `T_E` runs to Friday 06-20."""
    chain = saturday_monthly_chain()
    listed, last = date(2014, 6, 21), date(2014, 6, 20)
    assert not cal.is_session(listed) and cal.prev_or_same_session(listed) == last
    assert chain.last_session(listed) == last and (last - SATURDAY_MONTHLY_SESSION).days == 35

    tau = tau_of(chain, listed, cal)
    naive_tau = year_fraction(chain.ts, chain.ts + timedelta(days=(listed - SATURDAY_MONTHLY_SESSION).days))
    assert naive_tau > tau and (naive_tau - tau) * 365 == pytest.approx(1.0, abs=0.01)  # a whole day too long

    forwards = surface.parity_forwards(raw_table(chain), chain.rate, chain.ts, cal)
    assert abs(forwards[listed] - chain.forward(listed)) <= 1
    # the same forward computed with the naive date-based tau is materially different
    naive = chain.spot * math.exp(chain.rate * naive_tau)
    assert abs(naive - forwards[listed]) > 4  # cents, at 4% over one extra day on a $450 underlying


def test_a_good_friday_week_expiry_maps_to_the_thursday(cal: XnysCalendar) -> None:
    chain = good_friday_chain()
    listed, last = date(2014, 4, 19), date(2014, 4, 17)
    assert not cal.is_session(date(2014, 4, 18))  # Good Friday
    assert chain.last_session(listed) == last and (last - GOOD_FRIDAY_SESSION).days == 35
    forwards = surface.parity_forwards(raw_table(chain), chain.rate, chain.ts, cal)
    assert abs(forwards[listed] - chain.forward(listed)) <= 1
    nodes = {n.expiry: n for n in surface.chain_nodes(chain, cal)}
    assert nodes[listed].last_session == last and nodes[listed].dte == 35
    assert nodes[listed].tau_years == pytest.approx(year_fraction(chain.ts, cal.open_close(last)[1]))
    assert nodes[listed].tt_sessions == pytest.approx(trading_time(cal, chain.ts, cal.open_close(last)[1]))


def test_an_expiry_without_three_two_sided_pairs_gets_no_forward(cal: XnysCalendar) -> None:
    chain = make_chain()
    expiry = chain.expiries()[0]
    table = raw_table(chain)
    stamps = pd.to_datetime(table["expiry"])
    keep = stamps != pd.Timestamp(expiry)
    block = table[~keep].copy()
    # leave exactly two strikes two-sided on both sides; zero the bids everywhere else in this expiry
    two_sided_strikes = sorted(set(block["strike_milli"]))[len(set(block["strike_milli"])) // 2 :][:2]
    block.loc[~block["strike_milli"].isin(two_sided_strikes), "bid"] = 0
    crippled = pd.concat([table[keep], block], ignore_index=True)
    forwards = surface.parity_forwards(crippled, chain.rate, chain.ts, cal)
    assert expiry not in forwards and len(forwards) == len(chain.expiries()) - 1

    enriched = surface.enrich(crippled, forwards, chain.rate, chain.ts, cal, session=chain.key.session)
    orphans = enriched[pd.to_datetime(enriched["expiry"]) == pd.Timestamp(expiry)]
    assert (orphans["fwd"] == 0).all() and orphans["iv"].isna().all()  # 5.2: no forward => iv NaN, and the row survives
    assert orphans["last_session"].notna().all() and (orphans["dte"] > 0).all()
    assert surface.fit_smile(enriched, expiry, chain.ts, cal) is None
    assert expiry not in {n.expiry for n in surface.surface_nodes(enriched, {}, chain.ts, cal, session=chain.key.session)}


def test_parity_forwards_need_the_raw_quote_columns(cal: XnysCalendar) -> None:
    chain = make_chain()
    with pytest.raises(DataError, match="needs a `bid` column"):
        surface.parity_forwards(raw_table(chain).drop(columns=["bid"]), chain.rate, chain.ts, cal)
    with pytest.raises(ValueError, match="tz-aware"):
        surface.parity_forwards(raw_table(chain), chain.rate, datetime(2024, 5, 17, 20, 0), cal)


# ======================================================================================================================
# 5.2 The parity spot
# ======================================================================================================================


def test_the_parity_spot_recovers_the_underlying_and_names_the_front_expiry(cal: XnysCalendar) -> None:
    chain = make_chain()
    forwards = surface.parity_forwards(raw_table(chain), chain.rate, chain.ts, cal)
    spot, unmodelled, front = surface.parity_spot(forwards, chain.rate, chain.ts, (), calendar=cal, session=chain.key.session)
    assert abs(spot - chain.spot) <= 1
    assert front == chain.expiries()[0] and (cal.prev_or_same_session(front) - chain.key.session).days >= 2
    assert unmodelled is True  # nothing was verified: the spot could not include a dividend PV (5.2)
    assert surface.parity_spot(forwards, chain.rate, chain.ts, (), calendar=cal, covered=True)[1] is False
    # by hand: S = F * exp(-r T)
    tau = tau_of(chain, front, cal)
    assert spot == math.floor(forwards[front] * math.exp(-chain.rate * tau) + 0.5)


def test_the_parity_spot_adds_the_present_value_of_a_verified_dividend(cal: XnysCalendar) -> None:
    """`S = F e^{-r T} + sum(d_j e^{-r t_j})` over verified ex-dates in `(session, last_session(E1)]` (5.2)."""
    chain = make_chain()
    forwards = surface.parity_forwards(raw_table(chain), chain.rate, chain.ts, cal)
    front = chain.expiries()[0]
    last = cal.prev_or_same_session(front)
    ex_date = cal.next_session(chain.key.session, 2)
    assert chain.key.session < ex_date <= last
    dividend = exdiv("SPY", ex_date, 150)

    with_div = surface.parity_spot(
        forwards, chain.rate, chain.ts, [dividend], calendar=cal, session=chain.key.session, covered=True
    )
    tau = year_fraction(chain.ts, cal.open_close(last)[1])
    t_j = year_fraction(chain.ts, cal.open_close(ex_date)[1])
    by_hand = forwards[front] * math.exp(-chain.rate * tau) + 150 * math.exp(-chain.rate * t_j)
    assert with_div == (math.floor(by_hand + 0.5), False, front)
    bare = surface.parity_spot(forwards, chain.rate, chain.ts, (), calendar=cal, session=chain.key.session, covered=True)
    assert with_div[0] - bare[0] == pytest.approx(150, abs=1)  # a $1.50 dividend lifts the implied spot by ~$1.50


def test_the_parity_spot_ignores_dividends_outside_the_window_or_not_yet_knowable(cal: XnysCalendar) -> None:
    chain = make_chain()
    forwards = surface.parity_forwards(raw_table(chain), chain.rate, chain.ts, cal)
    front = chain.expiries()[0]
    last = cal.prev_or_same_session(front)
    base = surface.parity_spot(forwards, chain.rate, chain.ts, (), calendar=cal, session=chain.key.session, covered=True)[0]
    outside = [
        exdiv("SPY", chain.key.session, 150),  # today: already gone ex at the decision
        exdiv("SPY", cal.next_session(last), 150),  # after the front expiry's last session
        exdiv("SPY", cal.next_session(chain.key.session), 150, knowable=chain.ts + timedelta(days=1)),  # not knowable yet
    ]
    for event in outside:
        got = surface.parity_spot(forwards, chain.rate, chain.ts, [event], calendar=cal, session=chain.key.session, covered=True)
        assert got[0] == base, event.event_date
    cancelled = ScheduledEvent(
        kind="ex_dividend",
        event_date=cal.next_session(chain.key.session),
        underlying="SPY",
        amount_cents=150,
        scheduled=True,
        cancelled=True,
        knowable_at=chain.ts - timedelta(days=30),
        knowable_rule="exdiv_minus_14d_assumption",
        source_url="x",
        fetched_at=chain.ts,
    )
    assert surface.parity_spot(forwards, chain.rate, chain.ts, [cancelled], calendar=cal, session=chain.key.session)[0] == base


def test_the_front_expiry_needs_two_days_and_a_forward(cal: XnysCalendar) -> None:
    chain = make_chain(session=date(2024, 5, 20))  # a Monday, so the next session is one calendar day away
    forwards = surface.parity_forwards(raw_table(chain), chain.rate, chain.ts, cal)
    tomorrow = cal.next_session(chain.key.session)
    assert (tomorrow - chain.key.session).days == 1  # dte 1 < 2: not eligible as the front expiry
    near = {**forwards, tomorrow: 45_000}
    _spot, _unmodelled, front = surface.parity_spot(near, chain.rate, chain.ts, (), calendar=cal, session=chain.key.session)
    assert front != tomorrow and front == chain.expiries()[0]
    with pytest.raises(DataUnavailable, match="dte >= 2"):
        surface.parity_spot({tomorrow: 45_000}, chain.rate, chain.ts, (), calendar=cal, session=chain.key.session)
    with pytest.raises(DataUnavailable):
        surface.parity_spot({}, chain.rate, chain.ts, (), calendar=cal, session=chain.key.session)
    assert surface.parity_spot({**forwards, tomorrow: 0}, chain.rate, chain.ts, (), calendar=cal)[2] == front  # fwd 0 = no forward
    with pytest.raises(ValueError, match="not an XNYS session"):
        surface.parity_spot(forwards, chain.rate, chain.ts, (), calendar=cal, session=date(2024, 5, 18))


# ======================================================================================================================
# 5.3 Enrichment
# ======================================================================================================================


def test_enrichment_reproduces_the_factory_greeks_exactly(cal: XnysCalendar) -> None:
    """The factory solves iv / delta / vega the way 5.3 prescribes; `enrich` must agree bit for bit on the same quotes."""
    chain = make_chain()
    forwards = surface.parity_forwards(raw_table(chain), chain.rate, chain.ts, cal)
    enriched = surface.enrich(raw_table(chain), forwards, chain.rate, chain.ts, cal, session=chain.key.session)
    for column in ("iv", "delta", "vega"):
        expected = chain.table[column].to_numpy(dtype=np.float64)
        got = enriched[column].to_numpy(dtype=np.float64)
        assert np.array_equal(np.isnan(expected), np.isnan(got)), column
        np.testing.assert_allclose(got[~np.isnan(got)], expected[~np.isnan(expected)], rtol=0, atol=0)
    assert (enriched["fwd"].to_numpy() == chain.table["fwd"].to_numpy()).all()
    assert (enriched["dte"].to_numpy() == chain.table["dte"].to_numpy()).all()
    assert (enriched["last_session"].to_numpy() == chain.table["last_session"].to_numpy()).all()
    assert set(CHAIN_COLUMNS) <= set(enriched.columns)


def test_enrichment_solves_own_iv_only_from_two_sided_quotes_inside_the_band(cal: XnysCalendar) -> None:
    """D21 / 5.3: own IV from the MID of a `valid()` quote, accepted iff `0.02 < iv < 5.0`; otherwise NaN, and the greeks
    with it."""
    chain = make_chain()
    expiry = chain.expiries()[3]
    strike = int(chain.side(expiry, chain.table["right"].iloc[0] and __import__("jevbot.types", fromlist=["Right"]).Right.CALL).iloc[5]["strike_milli"])
    from jevbot.types import OptionContract, Right

    contract = OptionContract(underlying="SPY", expiry=expiry, right=Right.CALL, strike_milli=strike)
    zero_bid = set_quote(chain, contract, bid=0)
    forwards = surface.parity_forwards(raw_table(zero_bid), zero_bid.rate, zero_bid.ts, cal)
    enriched = surface.enrich(raw_table(zero_bid), forwards, zero_bid.rate, zero_bid.ts, cal, session=zero_bid.key.session)
    row = enriched[enriched["occ"] == contract.occ].iloc[0]
    assert pd.isna(row["iv"]) and pd.isna(row["delta"]) and pd.isna(row["vega"])  # one-sided: no own IV
    assert int(row["fwd"]) > 0 and int(row["dte"]) > 0  # but the row keeps its forward and its calendar facts
    # a deep-OTM quote whose mid implies a vol outside the band is refused too
    assert surface.IV_MIN == 0.02 and surface.IV_MAX == 5.0
    solvable = enriched[enriched["iv"].notna()]
    assert ((solvable["iv"] > surface.IV_MIN) & (solvable["iv"] < surface.IV_MAX)).all()


def test_enrichment_is_vectorised_and_inside_the_speed_budget(cal: XnysCalendar) -> None:
    """DESIGN 1.1: own IV is solved ONCE, vectorised, in `data derive`; no per-row Python solver runs in a backtest."""
    chain = make_chain()
    table = raw_table(chain)
    big = pd.concat([table] * 6, ignore_index=True)
    big["strike_milli"] = big["strike_milli"] + (big.index // len(table)).to_numpy() * 0  # same strikes, 6x the rows
    forwards = surface.parity_forwards(table, chain.rate, chain.ts, cal)
    assert len(big) >= 10_000
    started = time.perf_counter()
    enriched = surface.enrich(big, forwards, chain.rate, chain.ts, cal, session=chain.key.session)
    elapsed = time.perf_counter() - started
    assert len(enriched) == len(big) and enriched["iv"].notna().sum() > 0
    assert elapsed < 2.0, f"enrich took {elapsed:.3f}s for {len(big)} rows: the solver is no longer vectorised"


def test_enrich_needs_a_session_that_is_a_session(cal: XnysCalendar) -> None:
    chain = make_chain()
    forwards = surface.parity_forwards(raw_table(chain), chain.rate, chain.ts, cal)
    assert len(surface.enrich(raw_table(chain), forwards, chain.rate, chain.ts, cal)) == len(chain.table)  # session = ts.date()
    with pytest.raises(ValueError, match="not an XNYS session"):
        surface.enrich(raw_table(chain), forwards, chain.rate, chain.ts, cal, session=date(2024, 5, 18))
    with pytest.raises(DataError, match="needs a `right` column"):
        surface.enrich(raw_table(chain).drop(columns=["right"]), forwards, chain.rate, chain.ts, cal)


# ======================================================================================================================
# 6.4 step 1: the smile fit
# ======================================================================================================================


def test_the_fit_recovers_the_planted_quadratic_smile(cal: XnysCalendar) -> None:
    """The factory plants `w(k) = a (1 + skew k + curvature k^2)`, a quadratic in TOTAL VARIANCE; the fit must find it."""
    smile = Smile()
    chain = make_chain(smile=smile)
    for expiry in chain.expiries():
        fit = surface.fit_smile(chain.table, expiry, chain.ts, cal)
        assert fit is not None, expiry
        tau = tau_of(chain, expiry, cal)
        dte = (cal.prev_or_same_session(expiry) - chain.key.session).days
        a, b, c = smile.coefficients(dte, tau)
        assert fit.a == pytest.approx(a, rel=0.01)
        assert fit.b == pytest.approx(b, rel=0.05)
        assert fit.n_points >= surface.MIN_FIT_POINTS and fit.tau_years == pytest.approx(tau)
        assert fit.last_session == cal.prev_or_same_session(expiry) and fit.fwd == chain.forward(expiry)
        assert fit.k_lo < 0.0 < fit.k_hi
        # the fitted CURVE, not just its coefficients, matches the planted one across the whole fitted range
        for k in np.linspace(fit.k_lo, fit.k_hi, 25):
            planted = a + b * k + c * k * k
            assert fit.w(float(k)) == pytest.approx(planted, rel=0.02, abs=0.02 * a)
        # the analytic derivative is the derivative of the fitted curve
        step = 1e-6
        assert fit.dw_dk(0.03) == pytest.approx((fit.w(0.03 + step) - fit.w(0.03 - step)) / (2 * step), rel=1e-4)


def test_a_flat_smile_is_recovered_as_a_flat_smile(cal: XnysCalendar) -> None:
    chain = make_chain(smile=FLAT)
    expiry = chain.expiries()[4]
    fit = surface.fit_smile(chain.table, expiry, chain.ts, cal)
    assert fit is not None
    tau = tau_of(chain, expiry, cal)
    dte = (cal.prev_or_same_session(expiry) - chain.key.session).days
    assert fit.a == pytest.approx(FLAT.atm(dte) ** 2 * tau, rel=0.005)
    assert abs(fit.b) < 0.02 * fit.a and abs(fit.c) < 0.5 * fit.a  # no slope, no curvature worth the name
    assert abs(fit.dw_dk(0.0)) < 0.02 * fit.a


def test_the_fit_is_refused_without_enough_points_or_a_positive_total_variance(cal: XnysCalendar) -> None:
    chain = make_chain()
    expiry = chain.expiries()[0]
    table = chain.table.copy()
    stamps = pd.to_datetime(table["expiry"])
    # widen one expiry down to three usable quotes
    block = table[stamps == pd.Timestamp(expiry)]
    doomed = block.index[3:]
    table.loc[doomed, "bid"] = 0
    assert surface.fit_smile(table, expiry, chain.ts, cal) is None
    # an expiry that is not listed at all, and one whose last session has passed
    assert surface.fit_smile(chain.table, date(2024, 5, 23), chain.ts, cal) is None
    stale = make_chain(session=date(2024, 5, 17), expiries=[date(2024, 5, 24)])
    assert surface.fit_smile(stale.table, date(2024, 5, 24), cal.open_close(date(2024, 5, 24))[1], cal) is None
    with pytest.raises(DataError, match="needs a `vega` column"):
        surface.fit_smile(chain.table.drop(columns=["vega"]), chain.expiries()[1], chain.ts, cal)


def test_fit_smiles_covers_every_expiry_that_admits_a_fit(cal: XnysCalendar) -> None:
    chain = make_chain()
    fits = surface.fit_smiles(chain.table, chain.ts, cal)
    assert set(fits) == set(chain.expiries())
    assert set(surface.fit_smiles(chain.table, chain.ts, cal, expiries=chain.expiries()[:2])) == set(chain.expiries()[:2])


# ======================================================================================================================
# 5.3 The ATM term structure
# ======================================================================================================================


def test_the_two_strike_atm_iv_is_the_hand_interpolation_between_the_bracketing_strikes(cal: XnysCalendar) -> None:
    from jevbot.types import Right

    chain = make_chain()
    expiry = chain.expiries()[3]
    forward = chain.forward(expiry)
    calls = chain.side(expiry, Right.CALL).set_index("strike_milli")["iv"]
    puts = chain.side(expiry, Right.PUT).set_index("strike_milli")["iv"]
    strikes = sorted(set(calls.index) | set(puts.index))
    k1 = max(s for s in strikes if s / 10 <= forward)
    k2 = min(s for s in strikes if s / 10 > forward)
    weight = (forward - k1 / 10) / ((k2 - k1) / 10)
    by_hand = float(puts[k1]) + weight * (float(calls[k2]) - float(puts[k1]))  # put below the forward, call above
    assert surface.two_strike_atm_iv(chain.table, expiry, forward) == pytest.approx(by_hand)
    assert surface.two_strike_atm_iv(chain.table, expiry, 0) is None
    assert surface.two_strike_atm_iv(chain.table, expiry, 10_000_000) is None  # no strike above the forward


def test_the_atm_iv_is_the_fit_at_k_zero_with_the_two_strike_value_stored_beside_it(cal: XnysCalendar) -> None:
    chain = make_chain()
    fits = surface.fit_smiles(chain.table, chain.ts, cal)
    nodes = surface.surface_nodes(chain.table, fits, chain.ts, cal, session=chain.key.session)
    assert [n.expiry for n in nodes] == list(chain.expiries())
    for node in nodes:
        fit = fits[node.expiry]
        assert node.atm_iv == pytest.approx(math.sqrt(fit.w(0.0) / node.tau_years))  # the FIT at k = 0
        assert node.atm_iv_2s == pytest.approx(surface.two_strike_atm_iv(chain.table, node.expiry, node.fwd))
        assert node.atm_iv == pytest.approx(node.atm_iv_2s, rel=DataConfig().atm_iv_tolerance)
        assert node.dte >= 1 and node.fwd == chain.forward(node.expiry)
    # without a fit the node falls back to the two-strike value and keeps contributing
    bare = surface.surface_nodes(chain.table, {}, chain.ts, cal, session=chain.key.session)
    assert [n.expiry for n in bare] == [n.expiry for n in nodes]
    assert all(n.fit is None and n.atm_iv == pytest.approx(n.atm_iv_2s) for n in bare)


def test_a_distorted_pair_of_atm_quotes_diverges_from_the_fit(cal: XnysCalendar) -> None:
    """5.3: the two-strike level rests on two mid quotes and is fragile - which is why it is only the QC twin. A 30% shift
    of the two strikes around the forward must move it past `data.atm_iv_tolerance` while the fit barely moves."""
    from jevbot.types import OptionContract, Right

    chain = make_chain()
    expiry = chain.expiries()[5]
    forward = chain.forward(expiry)
    strikes = sorted(set(chain.side(expiry, Right.CALL)["strike_milli"]))
    k1 = max(s for s in strikes if s / 10 <= forward)
    k2 = min(s for s in strikes if s / 10 > forward)
    distorted = chain
    for strike, right in ((k1, Right.PUT), (k2, Right.CALL)):
        contract = OptionContract(underlying="SPY", expiry=expiry, right=right, strike_milli=strike)
        quote = chain.quote(contract)
        assert quote is not None
        distorted = set_quote(distorted, contract, bid=int(quote.bid * 1.35), ask=int(quote.ask * 1.35))
    fits = surface.fit_smiles(distorted.table, distorted.ts, cal)
    node = next(n for n in surface.surface_nodes(distorted.table, fits, distorted.ts, cal, session=chain.key.session) if n.expiry == expiry)
    divergence = abs(node.atm_iv / node.atm_iv_2s - 1.0)
    assert divergence > DataConfig().atm_iv_tolerance  # `data derive` raises ANOMALY atm_iv_divergence here
    clean = next(n for n in surface.chain_nodes(chain, cal) if n.expiry == expiry)
    assert node.atm_iv == pytest.approx(clean.atm_iv, rel=0.03) and node.atm_iv_2s > clean.atm_iv_2s * 1.1


def test_atm_term_renders_the_nodes_as_the_3_7_tuples(cal: XnysCalendar) -> None:
    chain = make_chain()
    fits = surface.fit_smiles(chain.table, chain.ts, cal)
    nodes = surface.surface_nodes(chain.table, fits, chain.ts, cal, session=chain.key.session)
    term = surface.atm_term(chain.table, fits, chain.ts, cal, session=chain.key.session)
    assert term == surface.term_of(nodes)
    for node, (tau, tt, bp, twin_bp, fwd) in zip(nodes, term, strict=True):
        assert (tau, tt, fwd) == (node.tau_years, node.tt_sessions, node.fwd)
        assert bp == round(node.atm_iv * 1e4) and twin_bp == round(node.atm_iv_2s * 1e4)
    assert [t[0] for t in term] == sorted(t[0] for t in term)  # ordered by last_session
    assert surface.chain_nodes(chain, cal) == nodes


# ======================================================================================================================
# V13: total variance in TRADING time
# ======================================================================================================================


def flat_term(nodes: list[tuple[float, float, float]]) -> list[TermNode]:
    """`(tau_years, tt_sessions, iv)` -> the 3.7 tuples."""
    return [(tau, tt, round(iv * 1e4), round(iv * 1e4), 45_000) for tau, tt, iv in nodes]


def test_total_variance_is_node_exact_linear_between_nodes_and_proportional_outside() -> None:
    # two hand-made nodes: w1 = 0.04 at 5 sessions, w2 = 0.09 at 10 sessions
    tau1, tau2 = 7 / 365, 14 / 365
    iv1, iv2 = math.sqrt(0.04 / tau1), math.sqrt(0.09 / tau2)
    term = flat_term([(tau1, 5.0, iv1), (tau2, 10.0, iv2)])
    w1 = (round(iv1 * 1e4) / 1e4) ** 2 * tau1
    w2 = (round(iv2 * 1e4) / 1e4) ** 2 * tau2

    assert surface.total_variance_at(term, 5.0) == (pytest.approx(w1), "interpolated")  # node-exact
    assert surface.total_variance_at(term, 10.0) == (pytest.approx(w2), "interpolated")
    midpoint = surface.total_variance_at(term, 7.5)
    assert midpoint == (pytest.approx(0.5 * (w1 + w2)), "interpolated")  # LINEAR in trading time
    assert surface.total_variance_at(term, 6.0)[0] == pytest.approx(w1 + 0.2 * (w2 - w1))
    assert surface.total_variance_at(term, 1.0) == (pytest.approx(w1 / 5.0), "extrapolated")  # below the first node
    assert surface.total_variance_at(term, 20.0) == (pytest.approx(2.0 * w2), "extrapolated")  # beyond the last
    with pytest.raises(DataUnavailable):
        surface.total_variance_at([], 1.0)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="positive number of sessions"):
            surface.total_variance_at(term, bad)


def test_two_expiries_that_share_a_last_session_collapse_to_one_node() -> None:
    """A Saturday-dated monthly and that week's Friday weekly have the SAME last session, hence the same trading time."""
    tau = 7 / 365
    iv = 0.16
    doubled = flat_term([(tau, 5.0, iv), (tau, 5.0, iv * 2)])
    assert surface.total_variance_at(doubled, 5.0)[0] == pytest.approx((round(iv * 1e4) / 1e4) ** 2 * tau)


def test_a_one_session_horizon_is_not_inflated_by_the_weekend() -> None:
    """V13 / 5.3, the reason the variance clock exists, on the two fixtures the spec names.

    The horizon is ONE session: Friday close -> Monday close (3 calendar days) against Tuesday close -> Wednesday close
    (1 calendar day). The Friday's front weekly is 7 calendar days / 5 sessions out, the Tuesday's 3 / 3.
    """
    per_session = 0.0001
    # (a) total variance proportional to SESSIONS: the two one-session moves must agree within 2%
    friday = flat_term([(7 / 365, 5.0, math.sqrt(per_session * 5.0 / (7 / 365)))])
    tuesday = flat_term([(3 / 365, 3.0, math.sqrt(per_session * 3.0 / (3 / 365)))])
    w_friday, _q = surface.total_variance_at(friday, 1.0)
    w_tuesday, _q2 = surface.total_variance_at(tuesday, 1.0)
    assert w_friday == pytest.approx(per_session, rel=1e-3) and w_tuesday == pytest.approx(per_session, rel=1e-3)
    assert math.sqrt(w_friday) / math.sqrt(w_tuesday) == pytest.approx(1.0, rel=0.02)

    # (b) a calendar-FLAT-vol weekly-expiry fixture: the same annualised IV at both front expiries
    iv = 0.16
    flat_friday = flat_term([(7 / 365, 5.0, iv)])
    flat_tuesday = flat_term([(3 / 365, 3.0, iv)])
    em_friday = math.sqrt(surface.total_variance_at(flat_friday, 1.0)[0])
    em_tuesday = math.sqrt(surface.total_variance_at(flat_tuesday, 1.0)[0])
    assert em_friday / em_tuesday == pytest.approx(math.sqrt(1.4), rel=1e-3)
    assert em_friday / em_tuesday <= 1.20  # the spec's bound for this fixture
    # the calendar-time rule the spec rejects: allocate by tau instead of by sessions
    calendar_friday = math.sqrt(iv**2 * (7 / 365) * (3 / 7))  # Friday -> Monday spans 3 calendar days
    calendar_tuesday = math.sqrt(iv**2 * (3 / 365) * (1 / 3))
    assert calendar_friday / calendar_tuesday == pytest.approx(math.sqrt(3.0), rel=1e-9)  # the 1.73 the spec calls out


def test_const_maturity_iv_interpolates_total_variance_in_calendar_time() -> None:
    tau1, tau2 = 21 / 365, 49 / 365
    iv1, iv2 = 0.15, 0.18
    term = flat_term([(tau1, 15.0, iv1), (tau2, 35.0, iv2)])
    target = 30 / 365
    w = iv1**2 * tau1 + (iv2**2 * tau2 - iv1**2 * tau1) * (target - tau1) / (tau2 - tau1)
    value, quality = surface.const_maturity_iv(term, 30)
    assert quality == "interpolated" and value == pytest.approx(math.sqrt(w / target), rel=1e-6)
    assert iv1 < value < iv2
    # one-sided: the nearest node, flagged extrapolated
    assert surface.const_maturity_iv(term, 90) == (pytest.approx(iv2), "extrapolated")
    assert surface.const_maturity_iv(term, 7) == (pytest.approx(iv1), "extrapolated")
    # the dte >= 7 filter of 5.3 applies here and ONLY here: a 2-day node is not allowed to anchor iv30
    short_and_long = flat_term([(2 / 365, 2.0, 0.60), (tau2, 35.0, iv2)])
    assert surface.const_maturity_iv(short_and_long, 30) == (pytest.approx(iv2), "extrapolated")  # only the 49-day node counts
    loose_value, loose_quality = surface.const_maturity_iv(short_and_long, 30, min_dte=1)
    assert loose_quality == "interpolated" and loose_value > iv2  # the 60-vol front week would drag iv30 up
    with pytest.raises(DataUnavailable, match="dte >= 7"):
        surface.const_maturity_iv(flat_term([(2 / 365, 2.0, 0.6)]), 30)
    with pytest.raises(ValueError, match="days must be >= 1"):
        surface.const_maturity_iv(term, 0)


def test_const_maturity_iv_on_the_factory_chain_sits_between_its_neighbours(cal: XnysCalendar) -> None:
    chain = make_chain()
    term = surface.term_of(surface.chain_nodes(chain, cal))
    iv30, quality = surface.const_maturity_iv(term, 30)
    assert quality == "interpolated"
    neighbours = [bp / 1e4 for tau, _tt, bp, _twin, _fwd in term if 21 / 365 <= tau <= 42 / 365]
    assert min(neighbours) <= iv30 <= max(neighbours)
    assert surface.const_maturity_iv(term, 90)[0] > iv30  # the factory's default term structure is in contango


# ======================================================================================================================
# 6.4 The skew-consistent implied digital
# ======================================================================================================================


def node_at_resolve(chain: ChainSnapshot, cal: XnysCalendar, expiry: date) -> tuple[datetime, float]:
    last = cal.prev_or_same_session(expiry)
    resolve = cal.open_close(last)[1]
    return resolve, year_fraction(chain.ts, resolve)


def test_a_flat_smile_gives_exactly_the_hand_computed_n_d2(cal: XnysCalendar) -> None:
    """With `dw/dk = 0` the digital collapses to `N(d2)` - computed here from `math.erfc`, independently of `bs.py`."""
    chain = make_chain(smile=FLAT)
    expiry = chain.expiries()[2]
    resolve, tau = node_at_resolve(chain, cal, expiry)
    fit = surface.fit_smile(chain.table, expiry, chain.ts, cal)
    assert fit is not None
    for strike in (int(chain.spot * 0.97), chain.spot, int(chain.spot * 1.03)):
        got = surface.implied_prob_above(chain, strike, resolve, cal)
        assert got is not None
        p, method, quality = got
        assert (method, quality) == ("smile_digital", "interpolated")  # the node's last session IS the resolve session
        forward = chain.spot * math.exp(chain.rate * tau)
        k = math.log(strike / forward)
        sigma = math.sqrt(fit.w(k) / tau)
        total_vol = sigma * math.sqrt(tau)
        d2 = math.log(forward / strike) / total_vol - 0.5 * total_vol
        assert p == pytest.approx(phi(d2), abs=2e-3)


def test_the_skew_term_is_exactly_the_vega_times_the_smile_slope(cal: XnysCalendar) -> None:
    """`PA = N(d2) - F sqrt(tau) n(d1) dsigma/dK` (6.4 step 3): the correction is checked against the fit's OWN slope, and
    an index smirk must push the digital ABOVE the plain N(d2)."""
    chain = make_chain()  # the default index-like smirk: negative skew
    expiry = chain.expiries()[4]
    resolve, tau = node_at_resolve(chain, cal, expiry)
    fit = surface.fit_smile(chain.table, expiry, chain.ts, cal)
    assert fit is not None
    strike = int(chain.spot * 0.98)
    forward = chain.spot * math.exp(chain.rate * tau)
    k = math.log(strike / forward)
    w, dw = fit.w(k), fit.dw_dk(k)
    sigma = math.sqrt(w / tau)
    dsigma_dk = dw / (2.0 * sigma * tau * strike)
    total_vol = sigma * math.sqrt(tau)
    d1 = math.log(forward / strike) / total_vol + 0.5 * total_vol
    d2 = d1 - total_vol
    by_hand = phi(d2) - forward * math.sqrt(tau) * npdf(d1) * dsigma_dk

    got = surface.implied_prob_above(chain, strike, resolve, cal)
    assert got is not None and got[1:] == ("smile_digital", "interpolated")
    assert got[0] == pytest.approx(by_hand, abs=1e-9)
    assert dsigma_dk < 0.0 and by_hand > phi(d2)  # a negative skew makes the down-side digital RICHER than N(d2)
    assert abs(by_hand - phi(d2)) > 1e-3  # and the correction is material, not noise


def test_the_plain_fallback_uses_the_shared_node_set(cal: XnysCalendar) -> None:
    """`nd2_plain` drops the slope term and prices off `total_variance_at` over the SAME nodes (6.4 step 3)."""
    chain = make_chain()
    expiry = chain.expiries()[4]
    resolve, tau = node_at_resolve(chain, cal, expiry)
    term = surface.term_of(surface.chain_nodes(chain, cal))
    tt = trading_time(cal, chain.ts, resolve)
    w, quality = surface.total_variance_at(term, tt)
    strike = int(chain.spot * 1.01)
    forward = chain.spot * math.exp(chain.rate * tau)
    sigma = math.sqrt(w / tau)
    total_vol = sigma * math.sqrt(tau)
    d2 = math.log(forward / strike) / total_vol - 0.5 * total_vol

    got = surface.implied_prob_above(chain, strike, resolve, cal, force_plain=True)
    assert got == (pytest.approx(phi(d2), abs=1e-9), "nd2_plain", quality)
    smile = surface.implied_prob_above(chain, strike, resolve, cal)
    assert smile is not None and abs(smile[0] - got[0]) > 1e-3  # the two methods really differ under skew


def test_a_horizon_below_the_first_expiry_uses_the_trading_time_ratio(cal: XnysCalendar) -> None:
    """6.4 step 2: below the first node, `rho = tt / tt_1`, `k' = k / sqrt(rho)`, `w = rho w_1(k')`. On a flat smile
    `w_1(k') = w_1`, so the whole rule reduces to `w = rho w_1` - hand-checkable end to end."""
    chain = make_chain(smile=FLAT)
    resolve = cal.open_close(cal.next_session(chain.key.session))[1]  # one session ahead; the first expiry is 5 away
    tau = year_fraction(chain.ts, resolve)
    tt = trading_time(cal, chain.ts, resolve)
    front = chain.expiries()[0]
    fit = surface.fit_smile(chain.table, front, chain.ts, cal)
    assert fit is not None
    rho = tt / fit.tt_sessions
    assert tt == pytest.approx(1.0) and fit.tt_sessions == pytest.approx(5.0) and rho == pytest.approx(0.2)

    strike = int(chain.spot * 1.005)
    forward = chain.spot * math.exp(chain.rate * tau)
    k = math.log(strike / forward)
    w = rho * fit.w(k / math.sqrt(rho))
    sigma = math.sqrt(w / tau)
    total_vol = sigma * math.sqrt(tau)
    d2 = math.log(forward / strike) / total_vol - 0.5 * total_vol

    got = surface.implied_prob_above(chain, strike, resolve, cal)
    assert got is not None and got[1:] == ("smile_digital", "extrapolated")
    assert got[0] == pytest.approx(phi(d2), abs=1e-9)


def test_the_horizon_between_two_nodes_is_blended_linearly_in_trading_time(cal: XnysCalendar) -> None:
    chain = make_chain(smile=FLAT)
    nodes = surface.chain_nodes(chain, cal)
    lower, upper = nodes[0], nodes[1]
    # a session strictly between the two expiries' last sessions
    middle = cal.next_session(lower.last_session, 2)
    assert lower.last_session < middle < upper.last_session
    resolve = cal.open_close(middle)[1]
    tau = year_fraction(chain.ts, resolve)
    tt = trading_time(cal, chain.ts, resolve)
    theta = (tt - lower.tt_sessions) / (upper.tt_sessions - lower.tt_sessions)
    assert 0.0 < theta < 1.0

    strike = int(chain.spot * 0.99)
    forward = chain.spot * math.exp(chain.rate * tau)
    k = math.log(strike / forward)
    assert lower.fit is not None and upper.fit is not None
    w = lower.fit.w(k) + theta * (upper.fit.w(k) - lower.fit.w(k))
    sigma = math.sqrt(w / tau)
    total_vol = sigma * math.sqrt(tau)
    d2 = math.log(forward / strike) / total_vol - 0.5 * total_vol

    got = surface.implied_prob_above(chain, strike, resolve, cal)
    assert got is not None and got[1:] == ("smile_digital", "interpolated")
    assert got[0] == pytest.approx(phi(d2), abs=1e-9)


def test_the_expected_move_and_the_digital_share_one_node_set(cal: XnysCalendar) -> None:
    """The acceptance sentence of 6.4: the threshold (`em`, from `total_variance_at`) and its reference (`PA`) are built
    from the SAME nodes and the SAME trading-time rule - every expiry with dte >= 1, no dte >= 7 filter."""
    chain = make_chain()
    nodes = surface.chain_nodes(chain, cal)
    term = surface.term_of(nodes)
    assert [n.dte for n in nodes] == sorted(n.dte for n in nodes) and min(n.dte for n in nodes) < 7  # the front week is IN
    resolve = cal.open_close(cal.next_session(chain.key.session, 5))[1]
    tt = trading_time(cal, chain.ts, resolve)
    w, quality = surface.total_variance_at(term, tt)
    em = math.sqrt(w)
    em_tenths = max(1, round(1000 * em))
    ref = chain.spot
    hi = (ref * (1000 + em_tenths) + 500) // 1000  # the 6.4 threshold arithmetic, on the same integer em
    got = surface.implied_prob_above(chain, int(hi), resolve, cal)
    assert got is not None and got[2] == quality  # the same horizon classification on both sides
    assert 0.05 < got[0] < 0.45  # a one-expected-move up-move is an unlikely-but-real event
    # passing the nodes in explicitly must not change anything
    assert surface.implied_prob_above(chain, int(hi), resolve, cal, nodes=nodes) == got


def test_the_digital_is_monotone_in_the_strike_and_clipped(cal: XnysCalendar) -> None:
    chain = make_chain()
    resolve = cal.open_close(cal.next_session(chain.key.session, 5))[1]
    probabilities = []
    for strike in range(int(chain.spot * 0.94), int(chain.spot * 1.06), 50):
        got = surface.implied_prob_above(chain, strike, resolve, cal)
        assert got is not None
        probabilities.append(got[0])
    assert all(later <= earlier for earlier, later in zip(probabilities, probabilities[1:], strict=False))
    assert all(0.001 <= p <= 0.999 for p in probabilities)
    far = surface.implied_prob_above(chain, int(chain.spot * 3), resolve, cal)
    assert far is not None and far[0] == 0.001


def test_the_digital_is_none_when_there_is_no_usable_horizon(cal: XnysCalendar) -> None:
    chain = make_chain()
    assert surface.implied_prob_above(chain, chain.spot, chain.ts, cal) is None  # a zero horizon
    assert surface.implied_prob_above(chain, chain.spot, chain.ts - timedelta(days=1), cal) is None
    resolve = cal.open_close(cal.next_session(chain.key.session, 5))[1]
    assert surface.implied_prob_above(chain, chain.spot, resolve, cal, nodes=[]) is None
    with pytest.raises(ValueError, match="strike_c must be positive"):
        surface.implied_prob_above(chain, 0, resolve, cal)
    with pytest.raises(ValueError, match="tz-aware"):
        surface.implied_prob_above(chain, chain.spot, datetime(2024, 5, 24, 20, 0), cal)


def test_a_node_without_a_fit_contributes_a_flat_smile(cal: XnysCalendar) -> None:
    """6.4 step 1: an expiry that cannot be fitted still participates, at its ATM total variance with `dw/dk = 0`, so the
    node set never differs from `atm_term`'s."""
    chain = make_chain()
    nodes = surface.chain_nodes(chain, cal)
    flat_nodes = [SurfaceNode(**{**n.__dict__, "fit": None}) for n in nodes]
    assert all(n.fit is None for n in flat_nodes)
    resolve = cal.open_close(nodes[2].last_session)[1]
    strike = int(chain.spot * 0.98)
    got = surface.implied_prob_above(chain, strike, resolve, cal, nodes=flat_nodes)
    plain = surface.implied_prob_above(chain, strike, resolve, cal, nodes=nodes, force_plain=True)
    assert got is not None and plain is not None
    assert got[1] == "nd2_plain" and got[0] == pytest.approx(plain[0], abs=1e-9)  # no fit anywhere => the plain fallback
    node = flat_nodes[2]
    assert node.w_atm == pytest.approx((round(node.atm_iv * 1e4) / 1e4) ** 2 * node.tau_years)
