"""bs.py (DESIGN.md 3.7): Black-76 price / delta / vega, vectorised implied vol, skew-consistent digital.

Reference values are independent of the code under test: hand-computed closed forms from tabulated standard-normal values, a
test-local formula on `scipy.stats.norm`, a numerical integral of the payoff under the lognormal density, put-call parity and
finite differences.
"""

import math
from itertools import pairwise

import numpy as np
import pytest
from scipy.stats import norm

from jevbot import bs

# tabulated standard-normal values (checked against scipy below, so a typo here cannot hide)
N_01, N_05, N_10 = 0.5398278372770290, 0.6914624612740131, 0.8413447460685429
PDF_01, PDF_05, PDF_10 = 0.3969525474770118, 0.3520653267642995, 0.24197072451914337


def ref_price(f: float, k: float, tau: float, sigma: float, df: float, is_call: bool) -> float:
    """Test-local Black-76 on scipy.stats.norm (scalar)."""
    v = sigma * math.sqrt(tau)
    d1 = (math.log(f / k) + 0.5 * sigma * sigma * tau) / v
    d2 = d1 - v
    if is_call:
        return float(df * (f * norm.cdf(d1) - k * norm.cdf(d2)))
    return float(df * (k * norm.cdf(-d2) - f * norm.cdf(-d1)))


def integral_price(f: float, k: float, tau: float, sigma: float, df: float, is_call: bool) -> float:
    """df * E[payoff] with S_T = F exp(-v^2/2 + v z), z ~ N(0, 1): trapezoid rule over z, no closed form involved."""
    v = sigma * math.sqrt(tau)
    z = np.linspace(-12.0, 12.0, 480_001)
    s = f * np.exp(-0.5 * v * v + v * z)
    payoff = np.maximum(s - k, 0.0) if is_call else np.maximum(k - s, 0.0)
    density = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    y = payoff * density
    return float(df * np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(z)))


def test_tabulated_normal_values() -> None:
    assert norm.cdf([0.1, 0.5, 1.0]) == pytest.approx([N_01, N_05, N_10], abs=1e-15)
    assert norm.pdf([0.1, 0.5, 1.0]) == pytest.approx([PDF_01, PDF_05, PDF_10], abs=1e-15)


# ======================================================================================================================
# price / delta / vega
# ======================================================================================================================


def test_at_the_money_forward_hand_values() -> None:
    # F = K = 100, tau = 1, sigma = 0.2, df = 1: d1 = 0.1, d2 = -0.1
    expected = 100.0 * (N_01 - (1.0 - N_01))  # = 100 (2 N(0.1) - 1) = 7.96556745540580
    assert expected == pytest.approx(7.9655674554058, abs=1e-12)
    assert float(bs.b76_price(100.0, 100.0, 1.0, 0.2, 1.0, True)) == pytest.approx(expected, abs=1e-12)
    assert float(bs.b76_price(100.0, 100.0, 1.0, 0.2, 1.0, False)) == pytest.approx(expected, abs=1e-12)  # ATM forward: C == P
    assert float(bs.b76_delta(100.0, 100.0, 1.0, 0.2, 1.0, True)) == pytest.approx(N_01, abs=1e-14)
    assert float(bs.b76_delta(100.0, 100.0, 1.0, 0.2, 1.0, False)) == pytest.approx(N_01 - 1.0, abs=1e-14)
    assert float(bs.b76_vega(100.0, 100.0, 1.0, 0.2, 1.0)) == pytest.approx(100.0 * PDF_01, abs=1e-12)


def test_d1_one_d2_half_hand_values() -> None:
    # sigma = 0.5, tau = 1, ln(F/K) = 0.375  =>  d1 = (0.375 + 0.125) / 0.5 = 1.0, d2 = 0.5; discounted at 5%
    f, sigma, tau, df = 100.0, 0.5, 1.0, math.exp(-0.05)
    k = f * math.exp(-0.375)
    call = df * (f * N_10 - k * N_05)
    put = df * (k * (1.0 - N_05) - f * (1.0 - N_10))
    assert float(bs.b76_price(f, k, tau, sigma, df, True)) == pytest.approx(call, abs=1e-12)
    assert float(bs.b76_price(f, k, tau, sigma, df, False)) == pytest.approx(put, abs=1e-12)
    assert call - put == pytest.approx(df * (f - k), abs=1e-12)  # the hand values themselves satisfy parity
    assert float(bs.b76_delta(f, k, tau, sigma, df, True)) == pytest.approx(df * N_10, abs=1e-14)
    assert float(bs.b76_delta(f, k, tau, sigma, df, False)) == pytest.approx(-df * (1.0 - N_10), abs=1e-14)
    assert float(bs.b76_vega(f, k, tau, sigma, df)) == pytest.approx(df * f * PDF_10, abs=1e-12)
    # the same d1 / d2 with sigma = 1.0 and tau = 0.25: only sigma sqrt(tau) matters for the price; vega scales with sqrt(tau)
    assert float(bs.b76_price(f, k, 0.25, 1.0, df, True)) == pytest.approx(call, abs=1e-12)
    assert float(bs.b76_vega(f, k, 0.25, 1.0, df)) == pytest.approx(df * f * PDF_10 * 0.5, abs=1e-12)


@pytest.mark.parametrize(
    ("f", "k", "tau", "sigma", "df"),
    [
        (45000.0, 46000.0, 30 / 365, 0.18, math.exp(-0.04 * 30 / 365)),  # cents, one month, OTM call / ITM put
        (45000.0, 41000.0, 45 / 365, 0.27, math.exp(-0.05 * 45 / 365)),  # OTM put / ITM call
        (66012.0, 66000.0, 1 / 365, 0.12, 1.0),  # one day
        (100.0, 35.0, 2.0, 0.9, 0.9),  # far ITM call, long-dated, high vol
        (100.0, 260.0, 0.5, 0.6, 0.97),
    ],
)
def test_price_against_independent_references(f: float, k: float, tau: float, sigma: float, df: float) -> None:
    for is_call in (True, False):
        got = float(bs.b76_price(f, k, tau, sigma, df, is_call))
        assert got == pytest.approx(ref_price(f, k, tau, sigma, df, is_call), rel=1e-12, abs=1e-12)
        assert got == pytest.approx(integral_price(f, k, tau, sigma, df, is_call), rel=1e-7, abs=1e-7 * f)


def test_put_call_parity_on_the_forward() -> None:
    f = np.array([100.0, 100.0, 45000.0, 45000.0, 66012.0])
    k = np.array([80.0, 125.0, 45000.0, 47250.0, 60000.0])
    tau = np.array([0.1, 2.0, 30 / 365, 7 / 365, 1.0])
    sigma = np.array([0.15, 0.6, 0.2, 0.35, 0.25])
    df = np.exp(-0.045 * tau)
    call = bs.b76_price(f, k, tau, sigma, df, True)
    put = bs.b76_price(f, k, tau, sigma, df, False)
    np.testing.assert_allclose((call - put) / f, df * (f - k) / f, rtol=0, atol=1e-13)  # C - P = df (F - K), per unit of forward
    # ... and for the deltas: delta_C - delta_P = df
    np.testing.assert_allclose(bs.b76_delta(f, k, tau, sigma, df, True) - bs.b76_delta(f, k, tau, sigma, df, False), df, atol=1e-14)


def test_delta_and_vega_are_the_price_derivatives() -> None:
    f, k, tau, sigma, df = 45000.0, 46000.0, 30 / 365, 0.18, 0.9967
    for is_call in (True, False):
        h = 0.5
        fd_delta = (ref_price(f + h, k, tau, sigma, df, is_call) - ref_price(f - h, k, tau, sigma, df, is_call)) / (2 * h)
        assert float(bs.b76_delta(f, k, tau, sigma, df, is_call)) == pytest.approx(fd_delta, abs=1e-7)
        e = 1e-5
        fd_vega = (ref_price(f, k, tau, sigma + e, df, is_call) - ref_price(f, k, tau, sigma - e, df, is_call)) / (2 * e)
        assert float(bs.b76_vega(f, k, tau, sigma, df)) == pytest.approx(fd_vega, rel=1e-7)


def test_delta_bounds_and_signs() -> None:
    k = np.linspace(20.0, 500.0, 97)
    for df in (1.0, 0.93):
        call = bs.b76_delta(100.0, k, 0.75, 0.4, df, True)
        put = bs.b76_delta(100.0, k, 0.75, 0.4, df, False)
        assert np.all((call > 0.0) & (call < df))  # forward delta * discount
        assert np.all((put < 0.0) & (put > -df))
        assert np.all(np.diff(call) < 0.0) and np.all(np.diff(put) < 0.0)  # both fall as the strike rises
    assert np.all(bs.b76_vega(100.0, k, 0.75, 0.4, 1.0) > 0.0)


def test_vectorisation_and_broadcasting() -> None:
    k = np.array([[90.0, 100.0, 110.0], [95.0, 105.0, 115.0]])
    is_call = np.array([True, False, True])
    out = bs.b76_price(100.0, k, 0.5, 0.3, 0.98, is_call)
    assert isinstance(out, np.ndarray) and out.dtype == np.float64 and out.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            assert out[i, j] == pytest.approx(ref_price(100.0, float(k[i, j]), 0.5, 0.3, 0.98, bool(is_call[j])), rel=1e-13)
    scalar = bs.b76_price(100, 100, 1, 0.2, 1, True)  # ints and 0-d results are fine
    assert isinstance(scalar, np.ndarray) and scalar.shape == ()
    assert bs.b76_delta(100.0, k, 0.5, 0.3, 0.98, is_call).shape == (2, 3)
    assert bs.b76_vega(100.0, k, 0.5, 0.3, 0.98).shape == (2, 3)
    assert bs.b76_implied_vol(out, 100.0, k, 0.5, 0.98, is_call).shape == (2, 3)


def test_zero_total_volatility_is_the_discounted_intrinsic_value() -> None:
    f, df = 100.0, 0.95
    k = np.array([80.0, 100.0, 120.0])
    for tau, sigma in ((0.0, 0.2), (-0.01, 0.2), (0.5, 0.0), (0.5, -0.1)):
        np.testing.assert_allclose(bs.b76_price(f, k, tau, sigma, df, True), df * np.array([20.0, 0.0, 0.0]), atol=1e-15)
        np.testing.assert_allclose(bs.b76_price(f, k, tau, sigma, df, False), df * np.array([0.0, 0.0, 20.0]), atol=1e-15)
        np.testing.assert_allclose(bs.b76_delta(f, k, tau, sigma, df, True), df * np.array([1.0, 0.0, 0.0]), atol=1e-15)
        np.testing.assert_allclose(bs.b76_delta(f, k, tau, sigma, df, False), df * np.array([0.0, 0.0, -1.0]), atol=1e-15)
        np.testing.assert_array_equal(bs.b76_vega(f, k, tau, sigma, df), np.zeros(3))
    # the live formula converges to that limit
    assert float(bs.b76_price(f, 80.0, 1e-10, 0.2, df, True)) == pytest.approx(df * 20.0, abs=1e-9)


def test_invalid_forward_or_strike_is_nan_without_warnings() -> None:
    with np.errstate(all="raise"):  # the module silences its own floating-point noise; nothing may leak as a warning / error
        for f, k in ((0.0, 100.0), (-5.0, 100.0), (100.0, 0.0), (100.0, -1.0), (float("nan"), 100.0)):
            assert math.isnan(float(bs.b76_price(f, k, 0.5, 0.2, 1.0, True)))
            assert math.isnan(float(bs.b76_delta(f, k, 0.5, 0.2, 1.0, False)))
            assert math.isnan(float(bs.b76_vega(f, k, 0.5, 0.2, 1.0)))
            assert math.isnan(float(bs.b76_implied_vol(5.0, f, k, 0.5, 1.0, True)))


def test_nan_inputs_give_nan_never_an_intrinsic_value() -> None:
    nan = float("nan")
    for tau, sigma, df in ((nan, 0.2, 1.0), (0.5, nan, 1.0), (0.5, 0.2, nan)):
        assert math.isnan(float(bs.b76_price(100.0, 80.0, tau, sigma, df, True)))  # NOT df * 20: an unknown vol is not a zero vol
        assert math.isnan(float(bs.b76_delta(100.0, 80.0, tau, sigma, df, True)))
        assert math.isnan(float(bs.b76_vega(100.0, 80.0, tau, sigma, df)))
    mixed = bs.b76_price(100.0, 80.0, np.array([0.5, nan, 0.0]), 0.2, 1.0, True)
    assert mixed[0] > 20.0 and math.isnan(mixed[1]) and mixed[2] == 20.0


# ======================================================================================================================
# implied vol
# ======================================================================================================================


@pytest.mark.parametrize("sigma", [0.0101, 0.05, 0.2, 0.75, 2.5, 4.99])
@pytest.mark.parametrize("moneyness", [0.85, 1.0, 1.2])
def test_implied_vol_round_trip(sigma: float, moneyness: float) -> None:
    f, tau, df = 45000.0, 60 / 365, 0.993
    k = f * moneyness ** (sigma / 0.2)  # keep the strike within reach of the vol level
    for is_call in (True, False):
        price = ref_price(f, k, tau, sigma, df, is_call)
        assert float(bs.b76_implied_vol(price, f, k, tau, df, is_call)) == pytest.approx(sigma, abs=1e-9)


def test_implied_vol_in_the_money_quotes_use_the_otm_twin_consistently() -> None:
    # an ITM call and the OTM put at the same strike carry the same vol (parity); both must come back
    f, k, tau, df, sigma = 45000.0, 40000.0, 30 / 365, 0.9967, 0.31
    call = ref_price(f, k, tau, sigma, df, True)  # ~ 5000 of intrinsic value plus a little time value
    put = ref_price(f, k, tau, sigma, df, False)
    assert call > df * (f - k) and put < 0.05 * call  # the call is almost all intrinsic value, the put all time value
    assert float(bs.b76_implied_vol(call, f, k, tau, df, True)) == pytest.approx(sigma, abs=1e-8)
    assert float(bs.b76_implied_vol(put, f, k, tau, df, False)) == pytest.approx(sigma, abs=1e-10)


def test_implied_vol_is_nan_outside_the_no_arbitrage_bounds() -> None:
    f, k, tau, df = 100.0, 90.0, 0.5, 0.97
    intrinsic_call = df * (f - k)
    nan_cases = [
        (intrinsic_call, True),  # at the lower bound: no time value
        (intrinsic_call - 0.5, True),  # below intrinsic
        (df * f, True),  # at the upper bound of a call
        (df * f + 1.0, True),
        (0.0, False),  # a put worth nothing (its intrinsic value is 0 here)
        (-1.0, False),
        (df * k, False),  # at the upper bound of a put
        (float("nan"), True),
        (float("inf"), True),
    ]
    for price, is_call in nan_cases:
        assert math.isnan(float(bs.b76_implied_vol(price, f, k, tau, df, is_call))), (price, is_call)
    ok = ref_price(f, k, tau, 0.25, df, True)
    for bad_tau in (0.0, -0.1, float("nan")):
        assert math.isnan(float(bs.b76_implied_vol(ok, f, k, bad_tau, df, True)))
    assert math.isnan(float(bs.b76_implied_vol(ok, f, k, tau, 0.0, True)))
    assert float(bs.b76_implied_vol(ok, f, k, tau, df, True)) == pytest.approx(0.25, abs=1e-10)


def test_implied_vol_outside_the_search_range_is_nan_never_a_clamped_bound() -> None:
    f, k, tau, df = 100.0, 100.0, 1.0, 1.0
    too_low = ref_price(f, k, tau, 0.005, df, True)  # a real price, but sigma < lo = 0.01
    too_high = ref_price(f, k, tau, 6.0, df, True)  # sigma > hi = 5.0
    assert math.isnan(float(bs.b76_implied_vol(too_low, f, k, tau, df, True)))
    assert math.isnan(float(bs.b76_implied_vol(too_high, f, k, tau, df, True)))
    # widening the bracket finds them
    assert float(bs.b76_implied_vol(too_low, f, k, tau, df, True, lo=0.001)) == pytest.approx(0.005, abs=1e-10)
    assert float(bs.b76_implied_vol(too_high, f, k, tau, df, True, hi=8.0)) == pytest.approx(6.0, abs=1e-9)


def test_implied_vol_mixed_vector() -> None:
    f, tau, df = 100.0, 0.25, 0.99
    k = np.array([90.0, 100.0, 110.0, 100.0, 100.0])
    is_call = np.array([False, True, True, True, False])
    price = bs.b76_price(f, k, tau, 0.3, df, is_call)
    price[3] = -1.0  # garbage row
    price[4] = np.nan
    iv = bs.b76_implied_vol(price, f, k, tau, df, is_call)
    np.testing.assert_allclose(iv[:3], 0.3, atol=1e-10)
    assert np.isnan(iv[3]) and np.isnan(iv[4])


def test_implied_vol_iterations_and_arguments() -> None:
    price = ref_price(100.0, 105.0, 0.5, 0.2234, 1.0, True)
    coarse = float(bs.b76_implied_vol(price, 100.0, 105.0, 0.5, 1.0, True, iters=10))
    fine = float(bs.b76_implied_vol(price, 100.0, 105.0, 0.5, 1.0, True))
    assert abs(coarse - 0.2234) <= (5.0 - 0.01) / 2**10  # bisection: the bracket halves every step
    assert abs(fine - 0.2234) < 1e-12
    for lo, hi, iters in ((0.0, 5.0, 48), (0.5, 0.5, 48), (2.0, 1.0, 48), (0.01, 5.0, 0)):
        with pytest.raises(ValueError):
            bs.b76_implied_vol(price, 100.0, 105.0, 0.5, 1.0, True, lo=lo, hi=hi, iters=iters)


# ======================================================================================================================
# prob_above: the skew-consistent digital
# ======================================================================================================================


def test_prob_above_without_skew_is_n_d2() -> None:
    # F = K = 100, tau = 1, sigma = 0.2: d2 = -0.1
    assert bs.prob_above(100.0, 100.0, 1.0, 0.2, 0.0) == pytest.approx(1.0 - N_01, abs=1e-15)
    # d1 = 1, d2 = 0.5
    assert bs.prob_above(100.0, 100.0 * math.exp(-0.375), 1.0, 0.5, 0.0) == pytest.approx(N_05, abs=1e-15)
    for k in (70.0, 95.0, 100.0, 104.0, 140.0):
        v = 0.3 * math.sqrt(0.4)
        d2 = math.log(100.0 / k) / v - 0.5 * v
        assert bs.prob_above(100.0, k, 0.4, 0.3, 0.0) == pytest.approx(float(norm.cdf(d2)), abs=1e-14)


def test_prob_above_skew_term_hand_value() -> None:
    # F = K = 100, tau = 1, sigma = 0.2, dsigma/dK = -0.001: N(-0.1) - 100 * 1 * n(0.1) * (-0.001) = 0.46017216 + 0.03969525
    expected = (1.0 - N_01) + 100.0 * PDF_01 * 0.001
    assert expected == pytest.approx(0.4998674174706722, abs=1e-14)
    assert bs.prob_above(100.0, 100.0, 1.0, 0.2, -0.001) == pytest.approx(expected, abs=1e-15)
    assert bs.prob_above(100.0, 100.0, 1.0, 0.2, 0.001) == pytest.approx((1.0 - N_01) - 100.0 * PDF_01 * 0.001, abs=1e-15)


def _smile_sigma(f: float, k: float) -> float:
    x = math.log(k / f)
    return 0.20 - 0.10 * x + 0.05 * x * x  # an index-like downward skew with a little curvature


def _smile_slope(f: float, k: float) -> float:
    x = math.log(k / f)
    return (-0.10 + 0.10 * x) / k  # d sigma / dK


@pytest.mark.parametrize("k", [88.0, 95.0, 100.0, 103.0, 110.0])
def test_skew_term_against_a_finite_difference_call_spread(k: float) -> None:
    # P(S_T > K) is minus the TOTAL strike derivative of the undiscounted call along the smile: a tight call spread measures it
    f, tau, h = 100.0, 45 / 365, 0.01
    spread = -(
        ref_price(f, k + h, tau, _smile_sigma(f, k + h), 1.0, True) - ref_price(f, k - h, tau, _smile_sigma(f, k - h), 1.0, True)
    ) / (2 * h)
    got = bs.prob_above(f, k, tau, _smile_sigma(f, k), _smile_slope(f, k))
    assert got == pytest.approx(spread, abs=2e-7)
    plain = bs.prob_above(f, k, tau, _smile_sigma(f, k), 0.0)
    # the plain N(d2) omits the smile-slope term (6.4): off by more than a point near the money, less in the wings (vega decays)
    assert abs(plain - spread) > (0.01 if 95.0 <= k <= 103.0 else 0.003)
    assert got > plain  # a downward skew puts MORE mass above the strike than flat-vol N(d2) says


def test_skew_term_with_the_total_variance_recipe_of_section_6_4() -> None:
    # 6.4 step 3: w(k) = a + b k + c k^2 (total variance, k = ln(K/F)); sigma = sqrt(w / tau); dsigma/dK = (dw/dk) / (2 sigma tau K)
    f, tau = 45000.0, 20 / 365
    a, b, c = 0.0022, -0.0040, 0.0150

    def sigma_at(strike: float) -> float:
        x = math.log(strike / f)
        return math.sqrt((a + b * x + c * x * x) / tau)

    for k in (43500.0, 45000.0, 46200.0):
        x = math.log(k / f)
        sigma = sigma_at(k)
        dsigma_dk = (b + 2.0 * c * x) / (2.0 * sigma * tau * k)
        h = 2.0
        spread = -(ref_price(f, k + h, tau, sigma_at(k + h), 1.0, True) - ref_price(f, k - h, tau, sigma_at(k - h), 1.0, True)) / (2 * h)
        assert bs.prob_above(f, k, tau, sigma, dsigma_dk) == pytest.approx(spread, abs=1e-7)


def test_prob_above_is_monotone_in_the_strike() -> None:
    f, tau = 100.0, 45 / 365
    strikes = np.linspace(70.0, 135.0, 261)
    flat = [bs.prob_above(f, float(k), tau, 0.22, 0.0) for k in strikes]
    skewed = [bs.prob_above(f, float(k), tau, _smile_sigma(f, float(k)), _smile_slope(f, float(k))) for k in strikes]
    for probs in (flat, skewed):
        assert all(p1 >= p2 for p1, p2 in pairwise(probs))  # non-increasing (flat inside the clip)
        assert probs[0] > 0.99 and probs[-1] < 0.01
        inner = [p for p in probs if 0.001 < p < 0.999]
        assert len(inner) > 100 and all(p1 > p2 for p1, p2 in pairwise(inner))  # strictly decreasing inside


def test_prob_above_is_clipped() -> None:
    assert bs.prob_above(100.0, 10.0, 0.1, 0.2, 0.0) == 0.999
    assert bs.prob_above(100.0, 1000.0, 0.1, 0.2, 0.0) == 0.001
    assert bs.prob_above(100.0, 100.0, 1.0, 0.2, 0.5) == 0.001  # an absurd positive slope cannot produce a negative probability
    assert bs.prob_above(100.0, 100.0, 1.0, 0.2, -0.5) == 0.999
    assert isinstance(bs.prob_above(100, 100, 1, 0.2, 0), float)  # ints are fine


@pytest.mark.parametrize(
    "args",
    [
        (0.0, 100.0, 1.0, 0.2, 0.0),
        (-1.0, 100.0, 1.0, 0.2, 0.0),
        (100.0, 0.0, 1.0, 0.2, 0.0),
        (100.0, 100.0, 0.0, 0.2, 0.0),
        (100.0, 100.0, -1.0, 0.2, 0.0),
        (100.0, 100.0, 1.0, 0.0, 0.0),
        (100.0, 100.0, 1.0, float("nan"), 0.0),
        (float("inf"), 100.0, 1.0, 0.2, 0.0),
        (100.0, 100.0, 1.0, 0.2, float("nan")),
        (100.0, 100.0, 1.0, 0.2, float("inf")),
    ],
)
def test_prob_above_refuses_garbage(args: tuple[float, float, float, float, float]) -> None:
    with pytest.raises(ValueError, match="prob_above"):
        bs.prob_above(*args)
