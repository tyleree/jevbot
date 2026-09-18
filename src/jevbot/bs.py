"""Black-76 on the parity forward (DESIGN.md section 3.7): price / delta / vega, vectorised implied vol, skew-consistent digital.

All vols are annualised on calendar/365 (`cal.year_fraction`); `tau` is that year fraction, `df` the discount factor
exp(-r * tau), `fwd` and `strike` share one price unit (the result of `b76_price` is in that unit). Everything is
numpy-vectorised with ordinary broadcasting and returns float64 arrays (0-d for scalar inputs); the normal cdf is
`scipy.special.ndtr`.

    d1 = (ln(F / K) + 0.5 sigma^2 tau) / (sigma sqrt(tau)),   d2 = d1 - sigma sqrt(tau)
    call = df (F N(d1) - K N(d2)),   put = df (K N(-d2) - F N(-d1))
    delta_call = df N(d1),   delta_put = -df N(-d1)            (forward delta * discount, signed)
    vega = df F n(d1) sqrt(tau)                                (per 1.00 of sigma)

Degenerate inputs: where `tau <= 0` or `sigma <= 0` the total volatility is zero and the functions return the limit values
(discounted intrinsic value, step delta, zero vega); a non-positive forward or strike, or a NaN anywhere, gives NaN. American early-exercise premium is
ignored (5.2: stated approximation). This module is pure: no IO, no clock, no package imports.
"""

import math
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

__all__ = ["b76_delta", "b76_implied_vol", "b76_price", "b76_vega", "prob_above"]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

_INV_SQRT_2PI: Final = 1.0 / math.sqrt(2.0 * math.pi)
_PROB_FLOOR: Final = 0.001
_PROB_CAP: Final = 0.999


def _f64(x: npt.ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def _cdf(x: FloatArray) -> FloatArray:
    out: FloatArray = np.asarray(ndtr(x), dtype=np.float64)
    return out


def _pdf(x: FloatArray) -> FloatArray:
    out: FloatArray = _INV_SQRT_2PI * np.exp(-0.5 * x * x)
    return out


def _d1_d2(log_fk: FloatArray, total_vol: FloatArray) -> tuple[FloatArray, FloatArray]:
    """(d1, d2) from ln(F/K) and the total volatility v = sigma sqrt(tau); callers mask v <= 0 themselves."""
    d1 = log_fk / total_vol + 0.5 * total_vol
    return d1, d1 - total_vol


def _total_vol(tau: FloatArray, sigma: FloatArray) -> FloatArray:
    """sigma sqrt(tau) where both are positive; 0.0 (the intrinsic-value limit) where tau <= 0 or sigma <= 0; NaN where either is NaN."""
    unknown = np.isnan(tau) | np.isnan(sigma)
    live = (tau > 0.0) & (sigma > 0.0)
    out: FloatArray = np.where(live, sigma * np.sqrt(np.maximum(tau, 0.0)), np.where(unknown, np.nan, 0.0))
    return out


def _by_regime(total_vol: FloatArray, live: FloatArray, limit: FloatArray | float) -> FloatArray:
    """`live` where the total volatility is positive, `limit` where it is exactly zero, NaN where it is unknown (NaN)."""
    out: FloatArray = np.where(total_vol > 0.0, live, np.where(total_vol == 0.0, limit, np.nan))
    return out


def _price_from_total_vol(
    f: FloatArray, k: FloatArray, log_fk: FloatArray, total_vol: FloatArray, d: FloatArray, call: BoolArray
) -> FloatArray:
    d1, d2 = _d1_d2(log_fk, total_vol)
    call_value = f * _cdf(d1) - k * _cdf(d2)
    put_value = k * _cdf(-d2) - f * _cdf(-d1)
    intrinsic = np.where(call, np.maximum(f - k, 0.0), np.maximum(k - f, 0.0))
    value = _by_regime(total_vol, np.where(call, call_value, put_value), intrinsic)
    out: FloatArray = np.where((f > 0.0) & (k > 0.0), d * value, np.nan)
    return out


def b76_price(
    fwd: npt.ArrayLike, strike: npt.ArrayLike, tau: npt.ArrayLike, sigma: npt.ArrayLike, df: npt.ArrayLike, is_call: npt.ArrayLike
) -> FloatArray:
    """Discounted Black-76 option value; `is_call` selects call (True) or put (False) per element."""
    f, k, d = _f64(fwd), _f64(strike), _f64(df)
    call: BoolArray = np.asarray(is_call, dtype=np.bool_)
    with np.errstate(all="ignore"):
        return _price_from_total_vol(f, k, np.log(f / k), _total_vol(_f64(tau), _f64(sigma)), d, call)


def b76_delta(
    fwd: npt.ArrayLike, strike: npt.ArrayLike, tau: npt.ArrayLike, sigma: npt.ArrayLike, df: npt.ArrayLike, is_call: npt.ArrayLike
) -> FloatArray:
    """Signed Black-76 delta (forward delta * discount): call df N(d1) in [0, df], put -df N(-d1) in [-df, 0]."""
    f, k, d = _f64(fwd), _f64(strike), _f64(df)
    call: BoolArray = np.asarray(is_call, dtype=np.bool_)
    with np.errstate(all="ignore"):
        total_vol = _total_vol(_f64(tau), _f64(sigma))
        d1, _ = _d1_d2(np.log(f / k), total_vol)
        live = np.where(call, _cdf(d1), -_cdf(-d1))
        # zero total volatility: the option is its intrinsic value, so delta is a step at the strike (0 exactly at the money)
        expired = np.where(call, np.where(f > k, 1.0, 0.0), np.where(f < k, -1.0, 0.0))
        out: FloatArray = np.where((f > 0.0) & (k > 0.0), d * _by_regime(total_vol, live, expired), np.nan)
    return out


def b76_vega(fwd: npt.ArrayLike, strike: npt.ArrayLike, tau: npt.ArrayLike, sigma: npt.ArrayLike, df: npt.ArrayLike) -> FloatArray:
    """Black-76 vega per 1.00 of sigma (the same for calls and puts): df F n(d1) sqrt(tau); 0 where tau <= 0 or sigma <= 0."""
    f, k, d, t = _f64(fwd), _f64(strike), _f64(df), _f64(tau)
    with np.errstate(all="ignore"):
        total_vol = _total_vol(t, _f64(sigma))
        d1, _ = _d1_d2(np.log(f / k), total_vol)
        live = f * _pdf(d1) * np.sqrt(np.maximum(t, 0.0))
        out: FloatArray = np.where((f > 0.0) & (k > 0.0), d * _by_regime(total_vol, live, 0.0), np.nan)
    return out


def b76_implied_vol(
    price: npt.ArrayLike,
    fwd: npt.ArrayLike,
    strike: npt.ArrayLike,
    tau: npt.ArrayLike,
    df: npt.ArrayLike,
    is_call: npt.ArrayLike,
    lo: float = 0.01,
    hi: float = 5.0,
    iters: int = 48,
) -> FloatArray:
    """Vectorised bisection for sigma in [lo, hi]; NaN where the price is outside the no-arbitrage bounds or tau <= 0.

    No-arbitrage bounds (exclusive): df * intrinsic(F, K) < price < df * F (call) or df * K (put). A price inside those bounds
    that no sigma in [lo, hi] reproduces (below the `lo` price or above the `hi` price) is NaN too - never a clamped `lo` / `hi`.
    The solve runs on the option's OUT-OF-THE-MONEY twin (put-call parity: call - put = df (F - K)), whose value is all time
    value: better conditioned for in-the-money quotes and half the normal-cdf evaluations. The twin's value is increasing in
    sigma, so the bracket [lo, hi] halves `iters` times: 48 steps resolve sigma to about 2e-14.
    """
    if not (0.0 < lo < hi) or iters < 1:
        raise ValueError(f"b76_implied_vol needs 0 < lo < hi and iters >= 1, got lo={lo}, hi={hi}, iters={iters}")
    p, f, k, t, d, call_raw = np.broadcast_arrays(
        _f64(price), _f64(fwd), _f64(strike), _f64(tau), _f64(df), np.asarray(is_call, dtype=np.bool_)
    )
    call: BoolArray = np.asarray(call_raw, dtype=np.bool_)
    with np.errstate(all="ignore"):
        log_fk = np.log(f / k)
        sqrt_tau = np.sqrt(np.maximum(t, 0.0))
        parity = d * (f - k)  # call - put
        intrinsic = np.where(call, np.maximum(parity, 0.0), np.maximum(-parity, 0.0))
        upper = d * np.where(call, f, k)
        in_bounds = (t > 0.0) & (f > 0.0) & (k > 0.0) & (d > 0.0) & (p > intrinsic) & (p < upper)

        twin_is_call = k >= f
        sign = np.where(twin_is_call, 1.0, -1.0)
        target = np.where(call == twin_is_call, p, np.where(call, p - parity, p + parity))

        def twin_value(sig: FloatArray | float) -> FloatArray:
            d1, d2 = _d1_d2(log_fk, sig * sqrt_tau)
            out: FloatArray = d * sign * (f * _cdf(sign * d1) - k * _cdf(sign * d2))
            return out

        solvable = in_bounds & (target >= twin_value(lo)) & (target <= twin_value(hi))
        low = np.full(p.shape, lo, dtype=np.float64)
        high = np.full(p.shape, hi, dtype=np.float64)
        for _ in range(iters):
            mid = 0.5 * (low + high)
            below = twin_value(mid) < target  # model value too low => sigma is higher
            low = np.where(below, mid, low)
            high = np.where(below, high, mid)
        result: FloatArray = np.where(solvable, 0.5 * (low + high), np.nan)
    return result


def prob_above(fwd: float, strike: float, tau: float, sigma: float, dsigma_dk: float) -> float:
    """Skew-consistent digital: P(S_T > K) = N(d2) - fwd * sqrt(tau) * n(d1) * dsigma_dk   (undiscounted; dsigma_dk = d sigma / dK).

    It is minus the TOTAL strike derivative of the undiscounted call C(K, sigma(K)): -dC/dK = N(d2) - vega * dsigma/dK with the
    undiscounted vega fwd * sqrt(tau) * n(d1); the plain N(d2) omits the smile-slope term (6.4).
    d1 = (ln(fwd/K) + 0.5 sigma^2 tau) / (sigma sqrt(tau)), d2 = d1 - sigma sqrt(tau). Result clipped to [0.001, 0.999].
    Non-finite or non-positive fwd / strike / tau / sigma, or a non-finite slope, raise ValueError: a probability that gets frozen
    in the ledger is never computed from garbage.
    """
    values = {"fwd": fwd, "strike": strike, "tau": tau, "sigma": sigma}
    for name, value in values.items():
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"prob_above: {name} must be finite and > 0, got {value!r}")
    if not math.isfinite(dsigma_dk):
        raise ValueError(f"prob_above: dsigma_dk must be finite, got {dsigma_dk!r}")
    total_vol = sigma * math.sqrt(tau)
    d1 = math.log(fwd / strike) / total_vol + 0.5 * total_vol
    d2 = d1 - total_vol
    pdf_d1 = _INV_SQRT_2PI * math.exp(-0.5 * d1 * d1)
    p = float(ndtr(d2)) - fwd * math.sqrt(tau) * pdf_d1 * dsigma_dk
    return min(max(p, _PROB_FLOOR), _PROB_CAP)
