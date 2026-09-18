"""Probabilistic / Deflated Sharpe ratio and the Minimum Track Record Length (DESIGN.md 12.6; WP07).

The four formulas of 12.6, verbatim::

    PSR(SR*) = Phi( (SR - SR*) * sqrt(T - 1) / sqrt(1 - skew*SR + (kurt - 1)/4 * SR^2) )   # per-period (daily) SR
    SR0      = sqrt(Var(SR_n)) * ((1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N*e)))               # g = Euler-Mascheroni
    DSR      = PSR(SR0)
    MinTRL   = 1 + (1 - skew*SR + (kurt - 1)/4 * SR^2) * (Z(conf) / (SR - SR_ref))^2       # observations

`kurt` is the **Pearson (non-excess) kurtosis**, Normal = 3: the `(kurt - 1)/4` term requires it (for a Normal sample
it gives `1 + SR^2/2`); feeding *excess* kurtosis would silently mis-scale PSR and MinTRL.  This module therefore
computes `skew` and `kurt` itself from the daily return series (`m3 / m2^1.5`, `m4 / m2^2`) and never accepts an
"excess" flavour.  `trial_results.kurt` (13.5) stores exactly this quantity.

Guards (12.6): `N < 2` or an undefined / zero `Var(SR_n)` => `SR0 = 0`, `DSR = PSR(0)`, flagged `dsr_trials<2` (never
`Z(0) = -inf`); a negative radicand or `SR <= SR_ref` => DSR / MinTRL reported as `n/a` (`None` here).

Scope of `N` (12.6) is the strategy-selection trial count and nothing else; `counts_as_selection_trial` is the pure
predicate the trial registry applies to its rows.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr
from scipy.stats import norm

from jevbot.errors import EvalError

__all__ = [
    "DERIVED_FAMILY_SUFFIXES",
    "EULER_MASCHERONI",
    "EXCLUDED_FLAGS",
    "EXCLUDED_FLAG_PREFIXES",
    "SELECTION_PURPOSES",
    "DsrResult",
    "SampleMoments",
    "counts_as_selection_trial",
    "deflated_sharpe",
    "min_trl",
    "moments",
    "psr",
    "psr_radicand",
    "sample_kurtosis",
    "sample_skew",
    "sharpe",
    "sr0",
    "windows_overlap",
]

FloatArray = npt.NDArray[np.float64]

EULER_MASCHERONI: Final = 0.5772156649  # `g` of 12.6, to the digits the spec prints
TRADING_DAYS_PER_YEAR: Final = 252  # the 252-day convention of 12.2 (rf = 0, V8)

#: 12.6: `N` counts registered trials with one of these purposes - completed, failed AND abandoned alike.
SELECTION_PURPOSES: Final[frozenset[str]] = frozenset({"tune", "validate", "final"})
#: 12.6: a trial carrying any of these flags is not strategy selection and never enters `N` or `Var(SR_n)`.
EXCLUDED_FLAGS: Final[frozenset[str]] = frozenset(
    {"placebo", "unmasked", "diagnostic", "shadow", "reference_history", "model_overlap"}
)
EXCLUDED_FLAG_PREFIXES: Final[tuple[str, ...]] = ("baseline:", "ablation:")
#: 12.6: runs that are not strategy selection register under these derived families.
DERIVED_FAMILY_SUFFIXES: Final[tuple[str, ...]] = ("#baseline", "#shadow", "#reference", "#diagnostic", "#ablation")

_EPS: Final = 1e-15


# ======================================================================================================================
# sample moments
# ======================================================================================================================


@dataclass(frozen=True, slots=True)
class SampleMoments:
    """Moments of a per-period (daily) return series.  `kurt` is Pearson (Normal = 3), never excess (12.6)."""

    n: int
    mean: float
    sd: float
    skew: float
    kurt: float


def _returns(values: npt.ArrayLike) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size and not bool(np.isfinite(arr).all()):
        raise EvalError("the return series contains non-finite values")
    return arr


def sample_skew(returns: npt.ArrayLike) -> float:
    """`m3 / m2^1.5` - the population (biased) skewness the PSR formula of 12.6 uses."""
    return moments(returns).skew


def sample_kurtosis(returns: npt.ArrayLike) -> float:
    """`m4 / m2^2` - the **Pearson** kurtosis (Normal = 3), the quantity `(kurt - 1)/4` expects (12.6)."""
    return moments(returns).kurt


def moments(returns: npt.ArrayLike) -> SampleMoments:
    """`n`, mean, standard deviation, `m3 / m2^1.5` and `m4 / m2^2` of a daily return series (12.6)."""
    arr = _returns(returns)
    n = int(arr.size)
    if n == 0:
        return SampleMoments(0, float("nan"), float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    centred = arr - mean
    m2 = float(np.mean(centred**2))
    if m2 <= _EPS:  # a constant series has no Sharpe and no shape
        return SampleMoments(n, mean, 0.0, float("nan"), float("nan"))
    m3 = float(np.mean(centred**3))
    m4 = float(np.mean(centred**4))
    return SampleMoments(n=n, mean=mean, sd=math.sqrt(m2), skew=m3 / m2**1.5, kurt=m4 / m2**2)


def sharpe(returns: npt.ArrayLike) -> float:
    """Per-period (daily) Sharpe with `rf = 0` (V8): `mean / sd`.  NaN for a constant or empty series."""
    stats = moments(returns)
    if stats.n == 0 or not math.isfinite(stats.sd) or stats.sd <= _EPS:
        return float("nan")
    return float(stats.mean / stats.sd)


# ======================================================================================================================
# PSR / DSR / MinTRL
# ======================================================================================================================


def psr_radicand(sr: float, *, skew: float, kurt: float) -> float:
    """`1 - skew*SR + (kurt - 1)/4 * SR^2` - the variance term shared by PSR and MinTRL (12.6).

    For a Normal sample (`skew = 0`, `kurt = 3`) it collapses to `1 + SR^2/2`, which is the standard result; feeding an
    *excess* kurtosis of 0 would give `1 - SR^2/4` and silently mis-scale both statistics.
    """
    return float(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr)


def psr(sr: float, sr_star: float, *, t: int, skew: float, kurt: float) -> float | None:
    """`Phi((SR - SR*) sqrt(T - 1) / sqrt(1 - skew SR + (kurt - 1)/4 SR^2))` (12.6).

    `None` = the `n/a` of 12.6: fewer than two observations or a negative radicand.
    """
    if t < 2:
        return None
    if not all(math.isfinite(v) for v in (sr, sr_star, skew, kurt)):
        return None
    radicand = psr_radicand(sr, skew=skew, kurt=kurt)
    if radicand <= 0.0:
        return None
    z = (sr - sr_star) * math.sqrt(t - 1) / math.sqrt(radicand)
    return float(ndtr(z))


def sr0(var_sr: float, n_trials: int) -> float:
    """`sqrt(Var(SR_n)) * ((1 - g) Z(1 - 1/N) + g Z(1 - 1/(N e)))` - the expected maximum Sharpe of `N` trials (12.6).

    Guard (12.6): `N < 2`, or an undefined / non-positive `Var(SR_n)`, gives `SR0 = 0` - never `Z(0) = -inf`.
    """
    if n_trials < 2 or not math.isfinite(var_sr) or var_sr <= 0.0:
        return 0.0
    g = EULER_MASCHERONI
    z1 = float(norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return float(math.sqrt(var_sr) * ((1.0 - g) * z1 + g * z2))


def min_trl(sr: float, sr_ref: float, *, skew: float, kurt: float, conf: float) -> float | None:
    """Minimum Track Record Length in **observations**: `1 + radicand * (Z(conf) / (SR - SR_ref))^2` (12.6, G1).

    `None` = `n/a`: `SR <= SR_ref` (the track record can never establish a difference that is not there) or a negative
    radicand.  Worked example of Bailey & Lopez de Prado, reproduced by `test_eval_dsr`: an annual Sharpe of 2 against
    a benchmark of 1, at 95% confidence, on Normal daily returns, needs about 2.73 years (688 observations).
    """
    if not 0.0 < conf < 1.0:
        raise EvalError(f"conf must be in (0, 1): {conf!r}")
    if not all(math.isfinite(v) for v in (sr, sr_ref, skew, kurt)):
        return None
    if sr <= sr_ref:
        return None
    radicand = psr_radicand(sr, skew=skew, kurt=kurt)
    if radicand <= 0.0:
        return None
    z = float(norm.ppf(conf))
    return float(1.0 + radicand * (z / (sr - sr_ref)) ** 2)


@dataclass(frozen=True, slots=True)
class DsrResult:
    """Everything the report prints for a band (12.6): SR, SR0, DSR, N and its composition, T, skew, kurt, MinTRL."""

    sr: float
    sr0: float
    dsr: float | None
    psr_zero: float | None  # PSR against SR* = 0, the `t > 3` rule-of-thumb companion
    min_trl: float | None  # observations
    min_trl_years: float | None
    t: int
    skew: float
    kurt: float
    var_sr: float
    n_trials: int
    n_completed: int
    n_failed: int
    n_abandoned: int
    flags: tuple[str, ...] = field(default=())

    @property
    def significant_rule_of_thumb(self) -> bool:
        """The `t > 3` rule of thumb of 12.6: `SR * sqrt(T) > 3`."""
        return math.isfinite(self.sr) and self.t > 0 and self.sr * math.sqrt(self.t) > 3.0


def deflated_sharpe(
    returns: npt.ArrayLike,
    *,
    n_trials: int,
    completed_sharpes: Sequence[float],
    n_completed: int | None = None,
    n_failed: int = 0,
    n_abandoned: int = 0,
    sr_ref: float = 0.0,
    conf: float = 0.95,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> DsrResult:
    """SR, SR0, DSR and MinTRL for one band's daily return series (12.6).

    `n_trials` is `N` **as scoped by 12.6**: registered selection trials of the same family and namespace whose window
    overlaps this run's - `completed`, `failed` and `abandoned` alike (a crashed or spend-stopped attempt was still a
    look at the data).  `completed_sharpes` are the daily Sharpes of the **completed** subset of exactly those trials;
    their variance is `Var(SR_n)`.  `eval/registry.py` does the scoping with `counts_as_selection_trial`.
    """
    if n_trials < 0:
        raise EvalError(f"n_trials must be >= 0: {n_trials!r}")
    stats = moments(returns)
    sr = sharpe(returns)
    flags: list[str] = []

    usable = np.asarray([s for s in completed_sharpes if math.isfinite(s)], dtype=np.float64)
    var_sr = float(usable.var(ddof=1)) if usable.size >= 2 else float("nan")
    if n_trials < 2:
        flags.append("dsr_trials<2")
    elif not math.isfinite(var_sr) or var_sr <= 0.0:
        flags.append("dsr_var_sr_undefined")
    deflator = sr0(var_sr, n_trials)

    dsr = psr(sr, deflator, t=stats.n, skew=stats.skew, kurt=stats.kurt)
    psr0 = psr(sr, 0.0, t=stats.n, skew=stats.skew, kurt=stats.kurt)
    trl = min_trl(sr, sr_ref, skew=stats.skew, kurt=stats.kurt, conf=conf)
    if dsr is None:
        flags.append("dsr_na")
    if trl is None:
        flags.append("min_trl_na")

    return DsrResult(
        sr=sr,
        sr0=deflator,
        dsr=dsr,
        psr_zero=psr0,
        min_trl=trl,
        min_trl_years=None if trl is None else trl / periods_per_year,
        t=stats.n,
        skew=stats.skew,
        kurt=stats.kurt,
        var_sr=var_sr,
        n_trials=n_trials,
        n_completed=len(list(completed_sharpes)) if n_completed is None else n_completed,
        n_failed=n_failed,
        n_abandoned=n_abandoned,
        flags=tuple(flags),
    )


# ======================================================================================================================
# the scope of N (12.6) - a pure predicate over registry rows
# ======================================================================================================================


def windows_overlap(a_start: object, a_end: object, b_start: object, b_end: object) -> bool:
    """Inclusive overlap of two `[start, end]` windows of comparable values (dates or ISO date strings)."""
    return not (a_end < b_start or b_end < a_start)  # type: ignore[operator]


def counts_as_selection_trial(
    *,
    purpose: str,
    flags: Sequence[str],
    family: str,
    namespace: str,
    reported_family: str,
    reported_namespace: str,
) -> bool:
    """The 12.6 scope of `N`: is this registry row a strategy-selection trial of the reported run's family?

    A row counts when its `purpose` is `tune` / `validate` / `final` (whatever its *status*: `completed`, `failed` and
    `abandoned` alike), it carries no `baseline:*`, `placebo`, `unmasked`, `diagnostic`, `shadow`, `reference_history`,
    `model_overlap` or `ablation:*` flag, its family is **exactly** the reported family (never a derived `#baseline` /
    `#shadow` / `#reference` / `#diagnostic` / `#ablation` family) and its namespace matches.  The caller adds the
    window-overlap condition with `windows_overlap`.
    """
    if purpose not in SELECTION_PURPOSES:
        return False
    for flag in flags:
        if flag in EXCLUDED_FLAGS or flag.startswith(EXCLUDED_FLAG_PREFIXES):
            return False
    if family != reported_family or namespace != reported_namespace:
        return False
    return not family.endswith(DERIVED_FAMILY_SUFFIXES)
