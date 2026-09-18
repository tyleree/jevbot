"""Property tests for bs.py on seeded random inputs (numpy generators): IV round trip at scale, parity, bounds, monotonicity.

The reference implementation below is test-local (scipy.stats.norm), never the code under test.
"""

import time
from itertools import pairwise

import numpy as np
import numpy.typing as npt
import pytest
from scipy.stats import norm

from jevbot import bs

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
Sample = tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, BoolArray]


def _sample(seed: int, n: int, *, max_std_moneyness: float = 2.0) -> Sample:
    """Forwards in cents, 5-400 day maturities, 5%-150% vols, strikes within `max_std_moneyness` standard deviations."""
    rng = np.random.default_rng(seed)
    f = rng.uniform(5_000.0, 70_000.0, n)
    tau = rng.uniform(5.0, 400.0, n) / 365.0
    sigma = rng.uniform(0.05, 1.5, n)
    z = rng.uniform(-max_std_moneyness, max_std_moneyness, n)
    k = f * np.exp(z * sigma * np.sqrt(tau))
    df = np.exp(-rng.uniform(0.0, 0.06, n) * tau)
    is_call = rng.random(n) < 0.5
    return f, k, tau, sigma, df, is_call


def _ref_price(f: FloatArray, k: FloatArray, tau: FloatArray, sigma: FloatArray, df: FloatArray, is_call: BoolArray) -> FloatArray:
    v = sigma * np.sqrt(tau)
    d1 = (np.log(f / k) + 0.5 * sigma**2 * tau) / v
    d2 = d1 - v
    call = df * (f * norm.cdf(d1) - k * norm.cdf(d2))
    put = df * (k * norm.cdf(-d2) - f * norm.cdf(-d1))
    out: FloatArray = np.where(is_call, call, put)
    return out


def test_implied_vol_round_trip_1e5_rows_under_one_second() -> None:
    f, k, tau, sigma, df, is_call = _sample(1, 100_000)
    price = _ref_price(f, k, tau, sigma, df, is_call)  # priced by the REFERENCE, inverted by the code under test
    bs.b76_implied_vol(price[:100], f[:100], k[:100], tau[:100], df[:100], is_call[:100])  # warm-up (imports, caches)
    best = float("inf")
    for _ in range(3):  # best of three: a scheduling hiccup on a shared machine is not a regression
        start = time.perf_counter()
        iv = bs.b76_implied_vol(price, f, k, tau, df, is_call)
        best = min(best, time.perf_counter() - start)
    assert iv.shape == (100_000,) and iv.dtype == np.float64
    assert not np.isnan(iv).any()
    np.testing.assert_allclose(iv, sigma, rtol=0, atol=1e-9)
    assert best < 1.0, f"vectorised implied vol took {best:.3f}s for 1e5 rows"


@pytest.mark.parametrize("seed", range(5))
def test_price_matches_the_reference_and_round_trips(seed: int) -> None:
    f, k, tau, sigma, df, is_call = _sample(10 + seed, 20_000, max_std_moneyness=3.0)
    price = bs.b76_price(f, k, tau, sigma, df, is_call)
    np.testing.assert_allclose(price, _ref_price(f, k, tau, sigma, df, is_call), rtol=1e-10, atol=1e-9)
    iv = bs.b76_implied_vol(price, f, k, tau, df, is_call)
    np.testing.assert_allclose(iv, sigma, rtol=0, atol=1e-7)


@pytest.mark.parametrize("seed", range(5))
def test_put_call_parity(seed: int) -> None:
    f, k, tau, sigma, df, _ = _sample(20 + seed, 20_000, max_std_moneyness=4.0)
    call = bs.b76_price(f, k, tau, sigma, df, True)
    put = bs.b76_price(f, k, tau, sigma, df, False)
    np.testing.assert_allclose(call - put, df * (f - k), rtol=0, atol=1e-9 * f.max())
    delta_gap = bs.b76_delta(f, k, tau, sigma, df, True) - bs.b76_delta(f, k, tau, sigma, df, False)
    np.testing.assert_allclose(delta_gap, df, rtol=0, atol=1e-13)


@pytest.mark.parametrize("seed", range(5))
def test_bounds(seed: int) -> None:
    f, k, tau, sigma, df, is_call = _sample(30 + seed, 20_000, max_std_moneyness=4.0)
    price = bs.b76_price(f, k, tau, sigma, df, is_call)
    intrinsic = df * np.where(is_call, np.maximum(f - k, 0.0), np.maximum(k - f, 0.0))
    upper = df * np.where(is_call, f, k)
    assert np.all(price >= intrinsic - 1e-9) and np.all(price <= upper)  # no-arbitrage bounds
    delta = bs.b76_delta(f, k, tau, sigma, df, is_call)
    assert np.all(np.where(is_call, (delta >= 0.0) & (delta <= df), (delta <= 0.0) & (delta >= -df)))  # delta bounds
    vega = bs.b76_vega(f, k, tau, sigma, df)
    assert np.all(vega > 0.0)
    assert np.all(np.isfinite(price)) and np.all(np.isfinite(delta)) and np.all(np.isfinite(vega))


@pytest.mark.parametrize("seed", range(5))
def test_monotonicity(seed: int) -> None:
    f, k, tau, sigma, df, is_call = _sample(40 + seed, 20_000)
    base = bs.b76_price(f, k, tau, sigma, df, is_call)
    assert np.all(bs.b76_price(f, k, tau, sigma * 1.05, df, is_call) > base)  # more vol, more value (vega > 0)
    assert np.all(bs.b76_price(f, k, tau * 1.05, sigma, df, is_call) > base)  # more time at the same df, more value
    higher_strike = bs.b76_price(f, k * 1.01, tau, sigma, df, is_call)
    assert np.all(np.where(is_call, higher_strike < base, higher_strike > base))  # calls fall, puts rise with the strike
    higher_forward = bs.b76_price(f * 1.01, k, tau, sigma, df, is_call)
    assert np.all(np.where(is_call, higher_forward > base, higher_forward < base))


@pytest.mark.parametrize("seed", range(5))
def test_vega_is_the_sigma_derivative(seed: int) -> None:
    f, k, tau, sigma, df, is_call = _sample(50 + seed, 5_000)
    e = 1e-5
    fd = (_ref_price(f, k, tau, sigma + e, df, is_call) - _ref_price(f, k, tau, sigma - e, df, is_call)) / (2 * e)
    np.testing.assert_allclose(bs.b76_vega(f, k, tau, sigma, df), fd, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_garbage_prices_are_nan_never_a_number(seed: int) -> None:
    f, k, tau, sigma, df, is_call = _sample(60 + seed, 10_000)
    rng = np.random.default_rng(600 + seed)
    intrinsic = df * np.where(is_call, np.maximum(f - k, 0.0), np.maximum(k - f, 0.0))
    upper = df * np.where(is_call, f, k)
    below = intrinsic - rng.uniform(0.0, 50.0, f.size)  # at or below the lower bound
    above = upper + rng.uniform(0.0, 50.0, f.size)  # at or above the upper bound
    assert np.isnan(bs.b76_implied_vol(below, f, k, tau, df, is_call)).all()
    assert np.isnan(bs.b76_implied_vol(above, f, k, tau, df, is_call)).all()
    good = _ref_price(f, k, tau, sigma, df, is_call)
    assert np.isnan(bs.b76_implied_vol(good, f, k, -tau, df, is_call)).all()  # tau <= 0
    assert np.isnan(bs.b76_implied_vol(good, f, k, np.zeros_like(tau), df, is_call)).all()
    # whatever comes back for an arbitrary positive price is either NaN or a vol that REPRODUCES the price
    wild = rng.uniform(0.0, 1.2, f.size) * upper
    iv = bs.b76_implied_vol(wild, f, k, tau, df, is_call)
    solved = ~np.isnan(iv)
    assert solved.any() and (~solved).any()
    assert np.all((iv[solved] >= 0.01) & (iv[solved] <= 5.0))
    repriced = _ref_price(f[solved], k[solved], tau[solved], iv[solved], df[solved], is_call[solved])
    np.testing.assert_allclose(repriced, wild[solved], rtol=1e-9, atol=1e-7)


@pytest.mark.parametrize("seed", range(5))
def test_prob_above_properties(seed: int) -> None:
    rng = np.random.default_rng(70 + seed)
    for _ in range(300):
        f = float(rng.uniform(5_000.0, 70_000.0))
        tau = float(rng.uniform(1.0, 120.0)) / 365.0
        sigma = float(rng.uniform(0.08, 0.9))
        v = sigma * np.sqrt(tau)
        strikes = f * np.exp(np.sort(rng.uniform(-3.0, 3.0, 12)) * v)
        flat = [bs.prob_above(f, float(k), tau, sigma, 0.0) for k in strikes]
        # with dsigma_dk = 0 it IS N(d2) (clipped) ...
        d2 = np.log(f / strikes) / v - 0.5 * v
        np.testing.assert_allclose(flat, np.clip(norm.cdf(d2), 0.001, 0.999), rtol=0, atol=1e-13)
        # ... monotone non-increasing in K and always a probability inside the clip
        assert all(a >= b for a, b in pairwise(flat))
        assert all(0.001 <= p <= 0.999 for p in flat)
        # a negative slope (downward skew) can only ADD mass above the strike, a positive slope only remove it
        k0 = float(strikes[5])
        slope = float(rng.uniform(0.0, 0.3)) / k0
        assert bs.prob_above(f, k0, tau, sigma, -slope) >= bs.prob_above(f, k0, tau, sigma, 0.0) >= bs.prob_above(f, k0, tau, sigma, slope)
