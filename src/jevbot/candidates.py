"""CandidateGenerator and `scan()` (DESIGN.md section 8): the one concrete structure the rules' `StructureKind` becomes.

Deterministic and Jev-free. Runs after `DecisionRules` chose a kind: exactly one candidate per (underlying, session). It
depends on the portfolio through **one number only**, `budget_floor = RiskEngine.budget_floor(pf)` (9.3), and never on
positions - so tier-matched baselines trade the same structure.

Pipeline (section 8, in this order):

1. **Expiry.** Listed expiries with `dte.min_entry <= dte <= dte.max_entry`, `sessions_to_expiry > dte.hard_exit_sessions +
   dte.min_sessions_beyond_hard_exit`, a parity forward and at least `candidates.min_two_sided_frac` of the strikes within
   +/- 2 expected moves two-sided; `argmin |dte - dte.target|`, ties => the later `last_session`. Every time computation
   runs to the expiry's **`last_session`**, never to the listed `expiry` (Conventions, INV-11).
2. **Delta.** "Nearest" = `argmin |abs(delta) - target|` among rows that pass the per-leg liquidity filter, ties => the
   strike further OTM. `candidates.delta_tolerance` guards the SHORT legs and a long single's starting strike; the long leg
   of a spread has no tolerance test (its target is a preference and the outer bound of the budget fit). A credit short
   additionally steps further OTM until it is `candidates.credit_short_min_em` expected moves from spot.
3. **Width rules.** `width = abs(K_long - K_short)` capped at `candidates.max_width_pct_spot * spot`; the legs never share
   a strike (`candidates.min_width_strikes` eligible strikes apart). Condor wings are clamped independently.
4. **Budget fit.** While `max_loss_pc` (worst band, `fee_rt` included) exceeds `budget_floor`, the LONG leg moves one
   eligible strike toward the short (a long single: one strike further OTM, never below `candidates.long_min_delta`); the
   condor narrows its wider wing first. The short leg never moves. Nothing left to narrow => `exceeds_risk_budget`.
5. **Pricing and sanity bounds** at the ORATS band, then the per-leg liquidity re-check and the ex-dividend entry block.

Early failures - no expiry, an unreachable short delta, no width, `exceeds_risk_budget` - have no legs or prices and return
a `CandidateReject`. A structure that could be built and priced but fails liquidity or economics is a `Candidate` with a
non-empty `rejects` (only `rejects == ()` is tradable).

Every 9.2 formula and the per-leg liquidity filter come from `structmath` (WP00); none of them is re-implemented here.
Pure: no IO, no clock, no network, no randomness.
"""

import bisect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from fractions import Fraction
from typing import Any, Final, Protocol

import numpy as np
import pandas as pd

from jevbot import structmath, vocab
from jevbot.cal import year_fraction
from jevbot.config import Config
from jevbot.errors import DataError, DataUnavailable, InvariantError
from jevbot.money import round_net, tick_cents
from jevbot.protocols import Calendar, ChainProvider, FillModel, MarketView
from jevbot.types import (
    Band,
    BandPrices,
    Candidate,
    CandidateReject,
    Cents,
    ChainSnapshot,
    FillRule,
    Leg,
    LegFill,
    OptionContract,
    OrderLeg,
    PositionIntent,
    Quote,
    Right,
    ScheduledEvent,
    Side,
    Slot,
    SnapshotKey,
    Structure,
    StructureKind,
)

__all__ = [
    "SCAN_TIERS",
    "CandidateGenerator",
    "budget_floor_at",
    "natural_limit",
    "scan",
    "scan_columns",
    "structure_tick",
]

# --- reject codes (every literal is checked against vocab.CANDIDATE_REJECTS at import) ----------------------------------
_NO_EXPIRY: Final = "no_expiry_in_window"
_WIDTH: Final = "width"
_EXCEEDS_BUDGET: Final = "exceeds_risk_budget"
_CREDIT_TO_WIDTH: Final = "credit_to_width"
_DEBIT_TO_WIDTH: Final = "debit_to_width"
_ECONOMICS_INVALID: Final = "economics_invalid"
_EXDIV_SHORT_CALL: Final = "exdiv_short_call"
_LEG_SHORT: Final = "short"
_LEG_LONG: Final = "long"
_LEG_SHORT_PUT: Final = "short_put"
_LEG_SHORT_CALL: Final = "short_call"

_SINGLES: Final[frozenset[StructureKind]] = frozenset({StructureKind.LONG_CALL, StructureKind.LONG_PUT})
_CREDIT_VERTICALS: Final[frozenset[StructureKind]] = frozenset({StructureKind.CALL_CREDIT, StructureKind.PUT_CREDIT})
_DEBIT_VERTICALS: Final[frozenset[StructureKind]] = frozenset({StructureKind.CALL_DEBIT, StructureKind.PUT_DEBIT})
# the right a vertical is written in, and whether the LONG leg sits further OTM than the short (credit) or closer to the money (debit)
_VERTICAL_RIGHT: Final[Mapping[StructureKind, Right]] = {
    StructureKind.CALL_CREDIT: Right.CALL,
    StructureKind.CALL_DEBIT: Right.CALL,
    StructureKind.PUT_CREDIT: Right.PUT,
    StructureKind.PUT_DEBIT: Right.PUT,
}
_SINGLE_RIGHT: Final[Mapping[StructureKind, Right]] = {StructureKind.LONG_CALL: Right.CALL, StructureKind.LONG_PUT: Right.PUT}
_TWO_EXPECTED_MOVES: Final = 2.0  # the +/- 2 expected-move band of the two-sided coverage test (section 8)
_MILLI_PER_CENT: Final = 10
_DELTA_TIE: Final = 1e-12  # two |delta| distances this close are a tie, broken by "further OTM" (section 8)

SCAN_TIERS: Final[tuple[int, ...]] = (500_000, 750_000, 1_000_000)  # the non-zero sizing tiers of 7.6, in ppm

_REJECT_ORDER: Final[Mapping[str, int]] = {code: i for i, code in enumerate(vocab.CANDIDATE_REJECTS)}
for _code in (
    _NO_EXPIRY,
    _WIDTH,
    _EXCEEDS_BUDGET,
    _CREDIT_TO_WIDTH,
    _DEBIT_TO_WIDTH,
    _ECONOMICS_INVALID,
    _EXDIV_SHORT_CALL,
    *(f"{vocab.DELTA_UNREACHABLE_PREFIX}{leg}" for leg in (_LEG_SHORT, _LEG_LONG, _LEG_SHORT_PUT, _LEG_SHORT_CALL)),
):
    if _code not in _REJECT_ORDER:  # a typo here would emit a code no RISK_VERDICT may carry (7.9)
        raise InvariantError(f"candidates.py emits {_code!r}, which is not a vocab.CANDIDATE_REJECTS code")
del _code


# ======================================================================================================================
# small pure helpers
# ======================================================================================================================


class _View(Protocol):
    """The part of `MarketView` the generator reads (every `MarketView` satisfies it; `scan()` supplies its own)."""

    @property
    def key(self) -> SnapshotKey: ...
    @property
    def session(self) -> date: ...
    @property
    def calendar(self) -> Calendar: ...
    def chain(self, underlying: str) -> ChainSnapshot: ...
    def events(self, start: date, end: date, underlying: str | None = None) -> tuple[ScheduledEvent, ...]: ...


def _exact(value: float, name: str) -> Fraction:
    """A config number as the exact decimal the operator wrote (`structmath` does the same): 0.03 -> 3/100, never binary noise."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvariantError(f"candidates: {name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value) or value < 0:
        raise InvariantError(f"candidates: {name} must be finite and >= 0, got {value!r}")
    return Fraction(repr(value))


def _opt_int(value: Any) -> int | None:
    return None if pd.isna(value) else int(value)


def _opt_float(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


def _opt_ts(value: Any) -> datetime | None:
    if pd.isna(value):
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise DataError(f"candidates: tz-naive quote_ts {value!r}")
    out: datetime = stamp.tz_convert("UTC").to_pydatetime(warn=False)
    return out


def _strike_cents(strike_milli: int) -> Cents:
    """A strike distance in milli-dollars as cents/share, rounded UP - exactly `types._milli_to_cents`, so a width computed
    here equals `Structure.width`."""
    return -(-abs(strike_milli) // _MILLI_PER_CENT)


def _band_price(fill: LegFill, band: Band) -> Cents:
    if band is Band.ORATS:
        return fill.orats
    if band is Band.WORST:
        return fill.worst
    return fill.mid


def _headline_band(cfg: Config) -> Band:
    """Conventions: `orats` under `next_snapshot`, `worst` under `same_snapshot_worst`."""
    return Band.ORATS if cfg.cadence.fill_rule is FillRule.NEXT_SNAPSHOT else Band.WORST


def _otm_step(right: Right) -> int:
    """+1 / -1 in ascending-strike order: further OTM is a higher strike for calls, a lower one for puts."""
    return 1 if right is Right.CALL else -1


def _order_rejects(codes: Sequence[str]) -> tuple[str, ...]:
    """Deduplicated, in the fixed order of `vocab.CANDIDATE_REJECTS`."""
    return tuple(sorted(set(codes), key=lambda code: _REJECT_ORDER[code]))


def structure_tick(cfg: Config, structure: Structure, quotes: Sequence[Quote]) -> int:
    """The price increment of a multi-leg net (section 8): the FINEST tick among the legs.

    SPY / QQQ / IWM are penny classes at any price (`universe.penny_all`); the generic 5c / 10c rule exists only because the
    universe is user-tunable. Each leg's tick is taken at its own ORATS-independent quote mid, so the tick never depends on
    which band a later limit is computed from.
    """
    if len(quotes) != len(structure.legs):
        raise InvariantError(f"structure_tick: {len(quotes)} quotes for {len(structure.legs)} legs")
    penny = cfg.universe.penny_all
    return min(tick_cents(structure.underlying, quote.mid2 // 2, penny) for quote in quotes)


def natural_limit(cfg: Config, structure: Structure, quotes: Sequence[Quote], net_worst: int) -> int:
    """`limit_natural` of section 8: `round_net(net.worst, tick, aggressive=False)` - the passive (never pay more / accept
    less) rounding of the natural price onto the finest leg tick. Integer cents, so a 3-decimal price cannot be produced."""
    return round_net(int(net_worst), structure_tick(cfg, structure, quotes), aggressive=False)


def budget_floor_at(cfg: Config, equity_cents: Cents) -> Cents:
    """The 9.3 `budget_floor` formula for an explicit equity: `floor(risk.max_loss_per_trade_pct * equity * min(non-zero tier))`.

    `RiskEngine.budget_floor(pf)` is the runtime source (it knows `risk.equity_basis`); this function exists because `scan()`
    reports its rates at `run.initial_equity_usd`, before any portfolio exists. Same arithmetic, exact decimals, floored.
    """
    tiers = [*(b for _a, b in cfg.rules.tiers.score), *(b for _a, b in cfg.rules.tiers.peakedness), *cfg.rules.tiers.environment]
    non_zero = [_exact(t, "rules.tiers") for t in tiers if t > 0.0]
    if not non_zero:
        raise InvariantError("candidates.budget_floor_at: [rules.tiers] has no non-zero tier")
    value = _exact(cfg.risk.max_loss_per_trade_pct, "risk.max_loss_per_trade_pct") * int(equity_cents) * min(non_zero)
    return int(value.numerator // value.denominator)


# ======================================================================================================================
# eligible rows and the chosen expiry
# ======================================================================================================================


@dataclass(frozen=True)
class _Row:
    """One eligible chain row of a single (expiry, right), with its materialised contract and quote."""

    contract: OptionContract
    quote: Quote
    strike_milli: int
    delta: float

    @property
    def abs_delta(self) -> float:
        return abs(self.delta)


@dataclass(frozen=True)
class _Expiry:
    """The chosen expiry and everything derived from it once."""

    chain: ChainSnapshot
    expiry: date
    last_session: date
    dte: int
    sessions_to_expiry: int
    spot: Cents
    tau: float
    atm_iv: float
    expected_move: float  # EM_T = sigma_atm * sqrt(tau) * spot, in cents


def _quote_from_row(underlying: str, expiry: date, right: Right, row: Any) -> Quote:
    contract = OptionContract(underlying=underlying, expiry=expiry, right=right, strike_milli=int(row.strike_milli))
    return Quote(
        contract=contract,
        bid=int(row.bid),
        ask=int(row.ask),
        bid_size=_opt_int(row.bid_size),
        ask_size=_opt_int(row.ask_size),
        oi_prev=_opt_int(row.oi_prev),
        iv=_opt_float(row.iv),
        delta=_opt_float(row.delta),
        vega=_opt_float(row.vega),
        quote_ts=_opt_ts(row.quote_ts),
    )


def _atm_iv(chain: ChainSnapshot, expiry: date) -> float | None:
    """Sigma at the money of one expiry: the two-strike interpolation of our own IVs at the parity forward.

    Per strike the call and put IV are averaged (they agree up to the rounding of the quotes); the value at the forward is
    linear in strike between the two bracketing strikes, and the nearest strike's IV outside the listed range.
    """
    rows = chain.table[(chain.table["expiry"] == pd.Timestamp(expiry)) & chain.table["iv"].notna()]
    if rows.empty:
        return None
    grouped = rows.groupby("strike_milli")["iv"].mean().sort_index()
    strikes = grouped.index.to_numpy(dtype=np.float64) / _MILLI_PER_CENT
    ivs = grouped.to_numpy(dtype=np.float64)
    fwd = float(chain.forward(expiry))
    if fwd <= strikes[0]:
        return float(ivs[0])
    if fwd >= strikes[-1]:
        return float(ivs[-1])
    j = int(np.searchsorted(strikes, fwd))
    lo, hi = strikes[j - 1], strikes[j]
    if hi <= lo:
        return float(ivs[j])
    weight = (fwd - lo) / (hi - lo)
    return float(ivs[j - 1] * (1.0 - weight) + ivs[j] * weight)


def _two_sided_frac(chain: ChainSnapshot, expiry: date, spot: Cents, expected_move: float) -> float:
    """Share of the strikes within +/- 2 expected moves of spot whose quote is two-sided (`Quote.valid()`)."""
    rows = chain.table[chain.table["expiry"] == pd.Timestamp(expiry)]
    if rows.empty:
        return 0.0
    strike_c = rows["strike_milli"].to_numpy(dtype=np.float64) / _MILLI_PER_CENT
    band = rows[(strike_c >= spot - _TWO_EXPECTED_MOVES * expected_move) & (strike_c <= spot + _TWO_EXPECTED_MOVES * expected_move)]
    if band.empty:
        return 0.0
    two_sided = (band["bid"] > 0) & (band["ask"] > band["bid"])
    return float(two_sided.sum()) / float(len(band))


# ======================================================================================================================
# the long leg's search path (delta target -> width clamp -> budget fit)
# ======================================================================================================================


@dataclass(frozen=True)
class _Wing:
    """One long/short pair (a vertical, or one wing of the condor) or a long single's lone leg.

    `path` holds the LONG leg's candidate row indices in fit order: the delta-target strike first (already width-clamped),
    then one eligible strike at a time toward the short leg (a single: further OTM). The short leg never moves.
    """

    short: _Row | None
    rows: tuple[_Row, ...]
    path: tuple[int, ...]

    def long(self, position: int) -> _Row:
        return self.rows[self.path[position]]

    def width(self, position: int) -> Cents:
        if self.short is None:
            return 0
        return _strike_cents(self.long(position).strike_milli - self.short.strike_milli)

    def legs(self, position: int) -> tuple[Leg, ...]:
        long_leg = Leg(contract=self.long(position).contract, side=Side.BUY)
        if self.short is None:
            return (long_leg,)
        return (Leg(contract=self.short.contract, side=Side.SELL), long_leg)


def _nearest(rows: Sequence[_Row], target: float, otm_step: int) -> int | None:
    """`argmin |abs(delta) - target|`; ties => the strike further OTM (section 8). None when `rows` is empty."""
    if not rows:
        return None
    best = min(abs(row.abs_delta - target) for row in rows)
    tied = [i for i, row in enumerate(rows) if abs(row.abs_delta - target) <= best + _DELTA_TIE]
    return max(tied) if otm_step > 0 else min(tied)


def _step_to_min_em(rows: Sequence[_Row], start: int, spot: Cents, expected_move: float, min_em: float, otm_step: int) -> int | None:
    """Step further OTM from `start` until the strike is at least `min_em` expected moves from spot (section 8, credit shorts)."""
    i = start
    while 0 <= i < len(rows):
        if abs(rows[i].strike_milli / _MILLI_PER_CENT - spot) >= min_em * expected_move:
            return i
        i += otm_step
    return None


def _short_index(rows: Sequence[_Row], short: _Row, toward_long: int) -> int:
    """Position of the short's strike inside the LONG leg's rows, so that `toward_long * (i - j) >= min_width_strikes`
    counts eligible strikes strictly between the legs even when the short's own strike is not buy-eligible."""
    strikes = [row.strike_milli for row in rows]
    j = bisect.bisect_left(strikes, short.strike_milli)
    if j < len(strikes) and strikes[j] == short.strike_milli:
        return j
    return j - 1 if toward_long > 0 else j


class CandidateGenerator:
    """Implements `protocols.CandidateGeneratorP` (section 8). Stateless apart from its configuration and fill model."""

    def __init__(self, cfg: Config, fill_model: FillModel) -> None:
        self._cfg = cfg
        self._fill = fill_model
        self._headline = _headline_band(cfg)

    @property
    def cfg(self) -> Config:
        return self._cfg

    @property
    def headline(self) -> Band:
        return self._headline

    # ------------------------------------------------------------------------------------------------------------------
    # the contract
    # ------------------------------------------------------------------------------------------------------------------

    def build(self, kind: StructureKind, view: MarketView, underlying: str, *, budget_floor: Cents) -> Candidate | CandidateReject:
        """The one candidate for `kind` on this snapshot, or a `CandidateReject` when nothing could be built or priced."""
        return self._build(StructureKind(kind), view, underlying, budget_floor)

    def eligible(self, chain: ChainSnapshot, *, sold: bool) -> pd.DataFrame:
        """The chain rows passing `structmath.leg_liquidity_rejects` for a leg we sell (`sold=True`) or buy - THE per-leg
        liquidity filter of section 8, the same implementation `RiskEngine.approve` check 9 and the paper gate call."""
        table = chain.table
        if table.empty:
            return table.copy()
        cfg = self._cfg.liquidity
        keep = [
            not structmath.leg_liquidity_rejects(
                _quote_from_row(chain.underlying, pd.Timestamp(row.expiry).date(), Right(row.right), row), sold=sold, cfg=cfg
            )
            for row in table.itertuples()
        ]
        return table[pd.Series(keep, index=table.index)].copy()

    # ------------------------------------------------------------------------------------------------------------------
    # eligible rows of one (expiry, right)
    # ------------------------------------------------------------------------------------------------------------------

    def _rows(self, chain: ChainSnapshot, expiry: date, right: Right, *, sold: bool) -> tuple[_Row, ...]:
        """Eligible rows of one (expiry, right), ascending by strike: liquidity filter passed and our own delta solvable."""
        table = chain.side(expiry, right)
        cfg = self._cfg.liquidity
        out: list[_Row] = []
        for row in table.itertuples():
            delta = _opt_float(row.delta)
            if delta is None:
                continue
            quote = _quote_from_row(chain.underlying, expiry, right, row)
            if structmath.leg_liquidity_rejects(quote, sold=sold, cfg=cfg):
                continue
            out.append(_Row(contract=quote.contract, quote=quote, strike_milli=int(row.strike_milli), delta=delta))
        return tuple(out)

    # ------------------------------------------------------------------------------------------------------------------
    # 1. expiry choice
    # ------------------------------------------------------------------------------------------------------------------

    def _choose_expiry(self, chain: ChainSnapshot, view: _View) -> _Expiry | None:
        cfg = self._cfg
        spot = chain.spot
        best: _Expiry | None = None
        best_rank: tuple[int, int] | None = None
        for expiry in chain.expiries():
            try:
                last = chain.last_session(expiry)
                dte = int(chain.table.loc[chain.table["expiry"] == pd.Timestamp(expiry), "dte"].iloc[0])
                forward = chain.forward(expiry)
            except (DataError, DataUnavailable, IndexError):
                continue
            if forward <= 0 or not cfg.dte.min_entry <= dte <= cfg.dte.max_entry:
                continue
            if view.calendar.sessions_between(view.session, last) <= cfg.dte.hard_exit_sessions + cfg.dte.min_sessions_beyond_hard_exit:
                continue
            sigma = _atm_iv(chain, expiry)
            if sigma is None or sigma <= 0.0:
                continue
            tau = year_fraction(chain.ts, view.calendar.open_close(last)[1])
            if tau <= 0.0:
                continue
            expected_move = sigma * math.sqrt(tau) * spot
            if expected_move <= 0.0:
                continue
            if _two_sided_frac(chain, expiry, spot, expected_move) < cfg.candidates.min_two_sided_frac:
                continue
            rank = (abs(dte - cfg.dte.target), -last.toordinal())  # argmin |dte - target|; ties => the LATER last_session
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best = _Expiry(
                    chain=chain,
                    expiry=expiry,
                    last_session=last,
                    dte=dte,
                    sessions_to_expiry=view.calendar.sessions_between(view.session, last),
                    spot=spot,
                    tau=tau,
                    atm_iv=sigma,
                    expected_move=expected_move,
                )
        return best

    # ------------------------------------------------------------------------------------------------------------------
    # 2-3. leg selection, width rules
    # ------------------------------------------------------------------------------------------------------------------

    def _short_leg(self, ctx: _Expiry, right: Right, target: float, *, min_em: float | None) -> _Row | str:
        """The short leg: nearest `target` delta inside `candidates.delta_tolerance`, then stepped further OTM until it is
        `min_em` expected moves from spot. Returns the row or the leg name of a `delta_target_unreachable:<leg>` reject."""
        cfg = self._cfg.candidates
        rows = self._rows(ctx.chain, ctx.expiry, right, sold=True)
        step = _otm_step(right)
        start = _nearest(rows, target, step)
        if start is None or abs(rows[start].abs_delta - target) > cfg.delta_tolerance:
            return "unreachable"
        if min_em is None:
            return rows[start]
        moved = _step_to_min_em(rows, start, ctx.spot, ctx.expected_move, min_em, step)
        if moved is None:
            return "unreachable"
        return rows[moved]

    def _wing(self, ctx: _Expiry, right: Right, short: _Row, long_target: float, *, long_further_otm: bool) -> _Wing | str:
        """A vertical (or one condor wing): the long leg's search path from its delta target toward the short leg, with the
        width cap already applied. Returns the wing or the reject code `width` when no eligible long strike is left."""
        cfg = self._cfg.candidates
        rows = self._rows(ctx.chain, ctx.expiry, right, sold=False)
        if not rows:
            return _WIDTH
        toward_long = _otm_step(right) * (1 if long_further_otm else -1)
        j = _short_index(rows, short, toward_long)
        domain = [i for i in range(len(rows)) if toward_long * (i - j) >= cfg.min_width_strikes]
        start = _nearest([rows[i] for i in domain], long_target, _otm_step(right))
        if start is None:
            return _WIDTH
        path: list[int] = []
        i = domain[start]
        while 0 <= i < len(rows) and toward_long * (i - j) >= cfg.min_width_strikes:
            path.append(i)
            i -= toward_long  # one eligible strike toward the short leg
        cap = _exact(cfg.max_width_pct_spot, "candidates.max_width_pct_spot") * int(ctx.spot)
        cap_c = int(cap.numerator // cap.denominator)  # floored: the cap is never widened by rounding
        clamped = [i for i in path if _strike_cents(rows[i].strike_milli - short.strike_milli) <= cap_c]
        if not clamped:
            return _WIDTH
        return _Wing(short=short, rows=rows, path=tuple(clamped))

    def _single_wing(self, ctx: _Expiry, right: Right) -> _Wing | str:
        """A long single: the 0.35-delta starting strike (tolerance-tested), then one strike further OTM at a time, never
        below `candidates.long_min_delta`."""
        cfg = self._cfg.candidates
        rows = self._rows(ctx.chain, ctx.expiry, right, sold=False)
        step = _otm_step(right)
        start = _nearest(rows, cfg.long_delta, step)
        if start is None or abs(rows[start].abs_delta - cfg.long_delta) > cfg.delta_tolerance:
            return "unreachable"
        path: list[int] = []
        i = start
        while 0 <= i < len(rows) and rows[i].abs_delta >= cfg.long_min_delta:
            path.append(i)
            i += step
        if not path:
            return "unreachable"
        return _Wing(short=None, rows=rows, path=tuple(path))

    def _plan(self, kind: StructureKind, ctx: _Expiry) -> list[_Wing] | tuple[str, ...]:
        """The wings of `kind` with their long-leg paths, or the `CandidateReject` codes of an early failure."""
        cfg = self._cfg.candidates
        if kind in _SINGLES:
            wing = self._single_wing(ctx, _SINGLE_RIGHT[kind])
            if isinstance(wing, str):
                return (f"{vocab.DELTA_UNREACHABLE_PREFIX}{_LEG_LONG}",)
            return [wing]
        if kind is StructureKind.IRON_CONDOR:
            wings: list[_Wing] = []
            for right, leg_name in ((Right.PUT, _LEG_SHORT_PUT), (Right.CALL, _LEG_SHORT_CALL)):
                short = self._short_leg(ctx, right, cfg.condor_short_delta, min_em=cfg.credit_short_min_em)
                if isinstance(short, str):
                    return (f"{vocab.DELTA_UNREACHABLE_PREFIX}{leg_name}",)
                wing = self._wing(ctx, right, short, cfg.condor_long_delta, long_further_otm=True)
                if isinstance(wing, str):
                    return (wing,)
                wings.append(wing)
            return wings
        right = _VERTICAL_RIGHT[kind]
        credit = kind in _CREDIT_VERTICALS
        short_target = cfg.credit_short_delta if credit else cfg.debit_short_delta
        short = self._short_leg(ctx, right, short_target, min_em=cfg.credit_short_min_em if credit else None)
        if isinstance(short, str):
            return (f"{vocab.DELTA_UNREACHABLE_PREFIX}{_LEG_SHORT}",)
        wing = self._wing(ctx, right, short, cfg.credit_long_delta if credit else cfg.debit_long_delta, long_further_otm=credit)
        if isinstance(wing, str):
            return (wing,)
        return [wing]

    # ------------------------------------------------------------------------------------------------------------------
    # 4-5. budget fit, pricing, the finished candidate
    # ------------------------------------------------------------------------------------------------------------------

    def _structure(self, kind: StructureKind, ctx: _Expiry, wings: Sequence[_Wing], at: Sequence[int]) -> Structure:
        legs: list[Leg] = []
        for wing, position in zip(wings, at, strict=True):
            legs.extend(wing.legs(position))
        ordered = tuple(sorted(legs, key=lambda leg: (leg.contract.right is Right.CALL, leg.contract.strike_milli)))
        return Structure(kind=kind, underlying=ctx.chain.underlying, expiry=ctx.expiry, last_session=ctx.last_session, legs=ordered)

    def _price(self, structure: Structure, chain: ChainSnapshot) -> tuple[BandPrices, Cents]:
        """`(net, fee_rt)`: the same `FillModel.price` the brokers use, and the 10.7 round-trip fee at the headline prices."""
        legs = tuple(
            OrderLeg(
                contract=leg.contract,
                side=leg.side,
                position_intent=PositionIntent.BTO if leg.side is Side.BUY else PositionIntent.STO,
                ratio=leg.ratio,
            )
            for leg in structure.legs
        )
        net, leg_fills, _quality = self._fill.price(legs, chain, mandatory=False)
        prices = [_band_price(fill, self._headline) for fill in leg_fills]
        n_sell = sum(1 for leg in structure.legs if leg.side is Side.SELL)
        return net, structmath.fee_round_trip(len(structure.legs), n_sell, prices, self._cfg.fees)

    def _max_loss(self, structure: Structure, chain: ChainSnapshot) -> tuple[BandPrices, Cents, Cents]:
        net, fee_rt = self._price(structure, chain)
        return net, fee_rt, structmath.max_loss_pc(structure.kind, structure.wing_widths, net.worst, fee_rt)

    def _fit(
        self, kind: StructureKind, ctx: _Expiry, wings: Sequence[_Wing], budget_floor: Cents
    ) -> tuple[Structure, BandPrices, Cents, Cents] | None:
        """The budget fit of section 8: narrow the long leg(s) until `max_loss_pc <= budget_floor`; the short never moves.

        The condor narrows its WIDER wing first (its max loss is driven by the wider wing). Returns None when even the
        narrowest reachable structure exceeds the budget (`exceeds_risk_budget`).
        """
        at = [0] * len(wings)
        while True:
            structure = self._structure(kind, ctx, wings, at)
            net, fee_rt, max_loss = self._max_loss(structure, ctx.chain)
            if max_loss <= budget_floor:
                return structure, net, fee_rt, max_loss
            order = sorted(range(len(wings)), key=lambda i: -wings[i].width(at[i]))
            for i in order:
                if at[i] + 1 < len(wings[i].path):
                    at[i] += 1
                    break
            else:
                return None

    def _build(self, kind: StructureKind, view: _View, underlying: str, budget_floor: Cents) -> Candidate | CandidateReject:
        cfg = self._cfg
        if isinstance(budget_floor, bool) or not isinstance(budget_floor, int) or budget_floor < 0:
            raise InvariantError(f"candidates.build: budget_floor must be a non-negative integer of cents, got {budget_floor!r}")
        chain = view.chain(underlying)

        def rejected(*codes: str) -> CandidateReject:
            return CandidateReject(underlying=underlying, kind=kind, key=view.key, rejects=_order_rejects(codes))

        ctx = self._choose_expiry(chain, view)
        if ctx is None:
            return rejected(_NO_EXPIRY)
        plan = self._plan(kind, ctx)
        if isinstance(plan, tuple):
            return rejected(*plan)
        fitted = self._fit(kind, ctx, plan, budget_floor)
        if fitted is None:
            return rejected(_EXCEEDS_BUDGET)
        structure, net, fee_rt, max_loss = fitted

        quotes_by_occ = self._quotes_of(structure, ctx.chain)
        quotes = tuple(quotes_by_occ[leg.contract.occ] for leg in structure.legs)

        rejects: list[str] = []
        for leg in structure.legs:
            rejects.extend(structmath.leg_liquidity_rejects(quotes_by_occ[leg.contract.occ], sold=leg.side is Side.SELL, cfg=cfg.liquidity))
        rejects.extend(self._economics(structure, net, max_loss))
        rejects.extend(self._exdiv_block(structure, view, ctx))

        widths = structure.wing_widths
        return Candidate(
            structure=structure,
            key=view.key,
            dte=ctx.dte,
            sessions_to_expiry=ctx.sessions_to_expiry,
            quotes=quotes,
            net=net,
            budget_floor=budget_floor,
            max_loss_per_contract=max_loss,
            max_profit_per_contract=structmath.max_profit_pc(kind, widths, net.get(self._headline)),
            bp_required_per_contract=structmath.bp_required_pc(kind, widths, net.worst, fee_rt, cfg.risk),
            breakevens=structmath.breakevens(kind, structure.legs, net.worst),
            short_distance_em=self._short_distance_em(structure, ctx),
            net_delta=sum(
                (1.0 if leg.side is Side.BUY else -1.0) * (quotes_by_occ[leg.contract.occ].delta or 0.0) for leg in structure.legs
            ),
            net_vega=sum((1.0 if leg.side is Side.BUY else -1.0) * (quotes_by_occ[leg.contract.occ].vega or 0.0) for leg in structure.legs),
            rejects=_order_rejects(rejects),
        )

    def _quotes_of(self, structure: Structure, chain: ChainSnapshot) -> dict[str, Quote]:
        out: dict[str, Quote] = {}
        for leg in structure.legs:
            quote = chain.quote(leg.contract)
            if quote is None:  # the legs were picked from this very snapshot
                raise InvariantError(f"candidates: {leg.contract.occ} vanished from the {chain.key.session} chain")
            out[leg.contract.occ] = quote
        return out

    def _short_distance_em(self, structure: Structure, ctx: _Expiry) -> float | None:
        """The nearest short strike's distance from spot in expected moves to expiry; None without a short leg."""
        shorts = structure.short_legs
        if not shorts:
            return None
        return min(abs(leg.contract.strike_milli / _MILLI_PER_CENT - ctx.spot) for leg in shorts) / ctx.expected_move

    def _economics(self, structure: Structure, net: BandPrices, max_loss: Cents) -> list[str]:
        """The sanity bounds of section 8, at the ORATS band: credit / debit to width, and the basic economics."""
        cfg = self._cfg.candidates
        kind = structure.kind
        width = structure.width
        signed = net.orats
        out: list[str] = []
        if kind in _SINGLES:
            if max_loss <= 0 or signed <= 0:
                out.append(_ECONOMICS_INVALID)
            return out
        credit_kind = kind in _CREDIT_VERTICALS or kind is StructureKind.IRON_CONDOR
        magnitude = -signed if credit_kind else signed
        if max_loss <= 0 or width <= 0 or not 0 < magnitude < width:
            out.append(_ECONOMICS_INVALID)
        if width <= 0:
            return out
        if credit_kind:
            condor = kind is StructureKind.IRON_CONDOR
            lo = _exact(cfg.condor_min_credit_to_width if condor else cfg.min_credit_to_width, "candidates.min_credit_to_width")
            hi = _exact(cfg.condor_max_credit_to_width if condor else cfg.max_credit_to_width, "candidates.max_credit_to_width")
            if magnitude < lo * width or magnitude > hi * width:
                out.append(_CREDIT_TO_WIDTH)
        elif magnitude > _exact(cfg.max_debit_to_width, "candidates.max_debit_to_width") * width:
            out.append(_DEBIT_TO_WIDTH)
        return out

    def _exdiv_block(self, structure: Structure, view: _View, ctx: _Expiry) -> list[str]:
        """Ex-dividend entry block (section 8, also RiskEngine check 11): a short call whose strike is below `spot + dividend`
        while a verified ex-date lies inside `[session, last_session]` would be ITM or near ITM across the ex-date."""
        short_calls = [leg for leg in structure.short_legs if leg.contract.right is Right.CALL]
        if not short_calls:
            return []
        events = view.events(view.session, ctx.last_session, structure.underlying)
        blocked = False
        for event in events:
            if event.kind != "ex_dividend" or event.underlying != structure.underlying:
                continue
            threshold = ctx.spot + (event.amount_cents or 0)
            if any(_strike_cents(leg.contract.strike_milli) < threshold for leg in short_calls):
                blocked = True
        return [_EXDIV_SHORT_CALL] if blocked else []


# ======================================================================================================================
# scan() - feasibility of the frozen [candidates] / [liquidity] / [risk] numbers (section 8)
# ======================================================================================================================


class _ScanView:
    """The read surface `CandidateGenerator` needs over one already-fetched snapshot. `scan()` walks a `ChainProvider`
    directly (it is an offline analysis over whole years, not a decision), so it has no event source: the ex-dividend entry
    block is inert here and `reject:exdiv_short_call` is always 0 in a scan."""

    def __init__(self, key: SnapshotKey, calendar: Calendar, chain: ChainSnapshot) -> None:
        self.key = key
        self.session = key.session
        self.calendar = calendar
        self._chain = chain

    def chain(self, underlying: str) -> ChainSnapshot:
        if underlying != self._chain.underlying:
            raise DataUnavailable(f"scan: no chain for {underlying} at {self.key.session}")
        return self._chain

    def events(self, start: date, end: date, underlying: str | None = None) -> tuple[ScheduledEvent, ...]:
        return ()


def _decision_keys(keys: Sequence[SnapshotKey]) -> list[SnapshotKey]:
    """One key per session: the `dec` snapshot when the archive has one, else `eod`, else the first slot of that session."""
    by_session: dict[date, list[SnapshotKey]] = {}
    for key in keys:
        by_session.setdefault(key.session, []).append(key)
    out: list[SnapshotKey] = []
    for session in sorted(by_session):
        slots = {key.slot: key for key in by_session[session]}
        out.append(slots.get(Slot.DEC) or slots.get(Slot.EOD) or by_session[session][0])
    return out


def scan_columns() -> tuple[str, ...]:
    """The exact column order of the frame `scan()` returns."""
    return (
        "underlying",
        "kind",
        "year",
        "sessions",
        "priced",
        "tradable",
        *(f"reject:{code}" for code in vocab.CANDIDATE_REJECTS),
        "exceeds_risk_budget_rate",
        *(f"size_zero_{tier // 10_000}" for tier in SCAN_TIERS),
        *(f"unsizeable_rate_{tier // 10_000}" for tier in SCAN_TIERS),
    )


def scan(
    cfg: Config,
    provider: ChainProvider,
    calendar: Calendar,
    start: date,
    end: date,
    *,
    fill_model: FillModel | None = None,
) -> pd.DataFrame:
    """Per (underlying, kind, year): candidates built, every reject code's count and - at `run.initial_equity_usd` - the
    share of sessions that are `exceeds_risk_budget` or would be `risk:size_zero` at EACH tier (0.5 / 0.75 / 1.0).

    Run BEFORE freezing `[candidates]` / `[liquidity]` / `[risk]` (CLI `data scan-candidates`): `register_trial` refuses
    `purpose = "final"` and the paper boot refuses to start when any enabled (underlying, kind) is unsizeable at the lowest
    non-zero tier more often than `candidates.max_unsizeable_rate` in the latest scanned year.

    `fill_model` defaults to the run's `fills.BandFillModel` (imported lazily so that this pure module keeps no import-time
    dependency on it); tests inject their own.
    """
    if fill_model is None:
        from jevbot.fills import BandFillModel  # lazy: scan() is an offline CLI path, not part of the decision cycle

        fill_model = BandFillModel(cfg)
    generator = CandidateGenerator(cfg, fill_model)
    equity_cents = int(cfg.run.initial_equity_usd) * 100
    floor = budget_floor_at(cfg, equity_cents)
    budgets = {tier: _budget_at_tier(cfg, equity_cents, tier) for tier in SCAN_TIERS}
    max_contracts = cfg.risk.max_contracts_per_trade
    kinds = tuple(StructureKind(k) for k in cfg.structures.enabled)

    rows: list[dict[str, Any]] = []
    counters: dict[tuple[str, StructureKind, int], dict[str, int]] = {}
    for underlying in provider.underlyings():
        for key in _decision_keys(provider.keys(underlying, start, end)):
            chain = provider.get_chain(underlying, key)
            if chain is None:
                continue
            view = _ScanView(key, calendar, chain)
            for kind in kinds:
                cell = counters.setdefault((underlying, kind, key.session.year), _new_cell())
                cell["sessions"] += 1
                # `_ScanView` is this module's own narrow view; `build()` is typed with the public `MarketView` contract
                result = generator._build(kind, view, underlying, floor)
                for code in result.rejects:
                    cell[f"reject:{code}"] += 1
                if isinstance(result, CandidateReject):
                    if _EXCEEDS_BUDGET in result.rejects:
                        cell["exceeds"] += 1
                    continue
                cell["priced"] += 1
                if result.rejects:
                    continue
                cell["tradable"] += 1
                # 9.3: `qty_requested >= 1` for every non-zero tier BY CONSTRUCTION of the budget fit (`budget_floor` is the
                # LOWEST tier's budget), so these counters stay 0 on a healthy configuration - which is exactly what makes a
                # non-zero one worth reporting: it means the fit and the sizing arithmetic disagree.
                for tier in SCAN_TIERS:
                    qty = min(budgets[tier] // result.max_loss_per_contract, max_contracts) if result.max_loss_per_contract > 0 else 0
                    if qty < 1:
                        cell[f"size_zero_{tier // 10_000}"] += 1

    for (underlying, kind, year), cell in sorted(counters.items(), key=lambda item: (item[0][0], item[0][1].value, item[0][2])):
        sessions = cell["sessions"]
        row: dict[str, Any] = {"underlying": underlying, "kind": kind.value, "year": year, "sessions": sessions}
        row["priced"] = cell["priced"]
        row["tradable"] = cell["tradable"]
        for code in vocab.CANDIDATE_REJECTS:
            row[f"reject:{code}"] = cell[f"reject:{code}"]
        row["exceeds_risk_budget_rate"] = cell["exceeds"] / sessions if sessions else 0.0
        for tier in SCAN_TIERS:
            name = f"size_zero_{tier // 10_000}"
            row[name] = cell[name]
            row[f"unsizeable_rate_{tier // 10_000}"] = (cell["exceeds"] + cell[name]) / sessions if sessions else 0.0
        rows.append(row)
    frame = pd.DataFrame(rows, columns=list(scan_columns()))
    return frame


def _new_cell() -> dict[str, int]:
    cell = {"sessions": 0, "priced": 0, "tradable": 0, "exceeds": 0}
    cell.update({f"reject:{code}": 0 for code in vocab.CANDIDATE_REJECTS})
    cell.update({f"size_zero_{tier // 10_000}": 0 for tier in SCAN_TIERS})
    return cell


def _budget_at_tier(cfg: Config, equity_cents: Cents, tier_ppm: int) -> Cents:
    """`budget = floor(risk.max_loss_per_trade_pct * equity_basis * tier_ppm / 1_000_000)` (9.3)."""
    value = _exact(cfg.risk.max_loss_per_trade_pct, "risk.max_loss_per_trade_pct") * int(equity_cents) * Fraction(tier_ppm, 1_000_000)
    return int(value.numerator // value.denominator)
