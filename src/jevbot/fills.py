"""`BandFillModel` - the three fill bands, the band-independent rejection rules, forced fills, liquidation marks and fees
(DESIGN.md 10.3-10.5, 10.7; implements `protocols.FillModel`, 3.4).

Integer arithmetic throughout, rounding always against us (money is integer cents per share; nothing here returns a float).
Every number this module produces ends up in a hashed FILL / MARK payload, so none of it may be a float (INV-24): config
fractions are read as the exact decimals the operator wrote (`Fraction(repr(x))`) and every division is a ceiling or floor.

* 10.3 bands per leg, with `p_bp = orats_p[min(n_legs, 4) - 1]` in basis points (7500 / 6600 / 5600 / 5300 by default):
      BUY   orats = bid + cdiv((ask - bid) * p_bp, 10000)    worst = ask    mid = cdiv(bid + ask, 2)
      SELL  orats = ask - cdiv((ask - bid) * p_bp, 10000)    worst = bid    mid = (bid + ask) // 2
  `net[band] = sum(buy prices) - sum(sell prices)` (signed: + debit / - credit). Never the last price, never a forward-filled
  quote. Property (`tests/property/test_prop_fills.py`): for a buy `bid <= mid <= orats <= worst = ask`; for a sell the reverse.
* 10.4 usability is PER SIDE. `no_quote` = a BUY leg without an ask, or a SELL leg of an OPEN order with `bid <= 0`. A SELL leg
  of a CLOSE / KILL order (`sell_to_close`) with `bid == 0 < ask` is NOT a reject: it is sold at 0 on all three bands (the ORATS
  interpolation is deliberately not used there: nobody pays inside a zero-bid market). Every rule of 10.4 is evaluated on its own
  and `check` returns EVERY code that matches, so a BUY leg quoted `100 x 0` is both `no_quote` (rule (a): no ask) and
  `crossed_or_locked`. Rejection never looks at a band, so the trade list is identical across bands. Forced fills (mandatory
  exits, kill) are never rejected: every usable leg is priced at the worst band plus
  `max(1 tick, cdiv(spread * forced_penalty_frac_spread))`; a leg without a usable quote is priced at the no-quote fallback
  (SELL `max(intrinsic - pad, 0)`, BUY `max(intrinsic, last mark) + pad`) and the fill is `degraded`.
* 10.5 marks: `liq_value = sum(ask of short legs) - sum(bid of long legs)` (what closing would COST now), `mid_value` likewise at
  mids. A long leg with `bid == 0` is a VALID mark of 0. A leg is unusable only when its quote is missing, a short leg has no ask,
  or the quote is crossed / locked with a positive bid; then the previous values are kept (`stale = True`).
* Ratios. `Leg.ratio` / `OrderLeg.ratio` is always 1 in v1 (2.3, 2.4; risk check 4 of 9.1 admits no other structure) and `LegFill`
  carries no ratio, so a signed net per band could not be replayed from the ledger for a ratio leg: an order leg with any other
  ratio is refused here rather than half-priced. `liquidation` works on a `Structure` and weights each leg by its ratio anyway,
  so a mark stays correct if ratios are ever admitted.
* 10.7 fees per fill through `structmath.fill_fees_micro` (the ONE fee arithmetic); the end-of-day charge is `cdiv(accrued, 10_000)`.

Import rules (section 1): no network library, no IO; `fills.py` imports only `config`, `errors`, `money`, `structmath`, `types`, `vocab`.
"""

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Final

from jevbot.config import Config, FeesConfig, FillsConfig, HealthConfig, LiquidityConfig, headline_band
from jevbot.errors import InvariantError
from jevbot.money import cdiv, tick_cents
from jevbot.structmath import MULTIPLIER, fill_fees_micro
from jevbot.types import (
    Band,
    BandPrices,
    Cents,
    ChainSnapshot,
    Fidelity,
    LegFill,
    Micros,
    OptionContract,
    OrderLeg,
    PositionIntent,
    Quote,
    Right,
    Side,
    Structure,
)
from jevbot.vocab import FILL_REJECTS

__all__ = [
    "BP_DENOMINATOR",
    "MICROS_PER_CENT",
    "QUALITY_DEGRADED",
    "QUALITY_OK",
    "BandFillModel",
    "end_of_day_fee_cents",
    "intrinsic_cents",
    "leg_band_prices",
    "net_of",
    "orats_p_bp",
]

BP_DENOMINATOR: Final = 10_000  # `p_bp` is in basis points of the spread
MICROS_PER_CENT: Final = 10_000  # fees accrue in micro-dollars; 10_000 micro-dollars = 1 cent
QUALITY_OK: Final = "ok"
QUALITY_DEGRADED: Final = "degraded"  # priced on an invalid / missing quote (2.4 `Fill.quality`)

_MILLI_PER_CENT: Final = 10  # strikes are milli-dollars (450.5 -> 450500)
_ORATS_LEG_CLASSES: Final = 4  # ORATS classes: 1, 2, 3, 4+ legs
_TIMED_FIDELITIES: Final = frozenset({Fidelity.RECORDED_INDICATIVE, Fidelity.LIVE_INDICATIVE})  # the only data with quote ages

(
    _REJECT_NO_QUOTE,
    _REJECT_CROSSED,
    _REJECT_WIDE_SPREAD,
    _REJECT_SIZE,
    _REJECT_OI,
    _REJECT_STALE,
    _REJECT_MISSING,
) = FILL_REJECTS
_REJECT_ORDER: Final[Mapping[str, int]] = {code: i for i, code in enumerate(FILL_REJECTS)}


# ======================================================================================================================
# input hygiene
# ======================================================================================================================


def _int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvariantError(f"fills: {name} must be an int (money is integer cents), got {type(value).__name__}: {value!r}")
    return value


def _exact(value: float, name: str) -> Fraction:
    """A config number as the exact decimal the operator wrote (0.75 -> 3/4), never binary noise."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvariantError(f"fills: {name} must be a number, got {type(value).__name__}")
    if value != value or value in (float("inf"), float("-inf")) or value < 0:  # `value != value` is the NaN test (no math import)
        raise InvariantError(f"fills: {name} must be finite and >= 0, got {value!r}")
    return Fraction(repr(value))


def _ceil(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


# ======================================================================================================================
# pure helpers (exposed so the tests and other packages can compute the same numbers)
# ======================================================================================================================


def orats_p_bp(n_legs: int, orats_p: Sequence[float] = FillsConfig().orats_p) -> int:
    """`p_bp = orats_p[min(n_legs, 4) - 1]` in basis points (10.3): 7500 / 6600 / 5600 / 5300 by default. A fraction that is not a
    whole number of basis points is rounded UP (against us: a higher share of the spread is worse on either side)."""
    n = _int(n_legs, "n_legs")
    if n < 1:
        raise InvariantError(f"fills: n_legs must be >= 1, got {n}")
    if len(orats_p) != _ORATS_LEG_CLASSES:
        raise InvariantError(f"fills: orats_p must hold {_ORATS_LEG_CLASSES} fractions (1, 2, 3, 4+ legs), got {len(orats_p)}")
    p = _exact(orats_p[min(n, _ORATS_LEG_CLASSES) - 1], "fills.orats_p")
    if p > 1:
        raise InvariantError(f"fills: orats_p entries must be <= 1, got {p}")
    return _ceil(p * BP_DENOMINATOR)


def leg_band_prices(side: Side, bid: Cents, ask: Cents, p_bp: int) -> tuple[Cents, Cents, Cents]:
    """The 10.3 row for one leg with a usable quote: `(orats, worst, mid)` cents/share as traded.

    BUY: `bid + cdiv((ask - bid) * p_bp, 10000)`, `ask`, `cdiv(bid + ask, 2)`; SELL: `ask - cdiv(...)`, `bid`, `(bid + ask) // 2`.
    Requires `0 <= bid <= ask`, `ask > 0` (a usable quote on the side in question; the zero-bid sell-to-close rule is applied by
    the caller, which never interpolates a zero-bid market).
    """
    b, a, p = _int(bid, "bid"), _int(ask, "ask"), _int(p_bp, "p_bp")
    if not 0 <= b <= a or a <= 0:
        raise InvariantError(f"fills: leg_band_prices needs 0 <= bid <= ask with ask > 0, got bid={b}, ask={a}")
    if not 0 <= p <= BP_DENOMINATOR:
        raise InvariantError(f"fills: p_bp must be in 0..{BP_DENOMINATOR}, got {p}")
    share = cdiv((a - b) * p, BP_DENOMINATOR)
    if Side(side) is Side.BUY:
        return (b + share, a, cdiv(b + a, 2))
    return (a - share, b, (b + a) // 2)


def net_of(leg_fills: Sequence[LegFill]) -> BandPrices:
    """`net[band] = sum(buy prices) - sum(sell prices)`, signed cents/share (+ debit / - credit), per band (10.3)."""
    orats = worst = mid = 0
    for leg in leg_fills:
        sign = 1 if leg.side is Side.BUY else -1
        orats += sign * _int(leg.orats, "LegFill.orats")
        worst += sign * _int(leg.worst, "LegFill.worst")
        mid += sign * _int(leg.mid, "LegFill.mid")
    return BandPrices(orats=orats, worst=worst, mid=mid)


def intrinsic_cents(contract: OptionContract, spot: Cents, *, up: bool) -> Cents:
    """Intrinsic value in cents/share against `spot` (cents): call `max(spot - K, 0)`, put `max(K - spot, 0)`. A strike that is
    not a whole cent rounds against us: `up=True` for a price we pay (BUY), `up=False` for one we receive (SELL)."""
    s = _int(spot, "spot")
    milli = s * _MILLI_PER_CENT - contract.strike_milli if contract.right is Right.CALL else contract.strike_milli - s * _MILLI_PER_CENT
    if milli <= 0:
        return 0
    return cdiv(milli, _MILLI_PER_CENT) if up else milli // _MILLI_PER_CENT


def end_of_day_fee_cents(accrued_micro: Micros) -> Cents:
    """The session-end charge (10.7): `cdiv(accrued, 10_000)` - Alpaca charges at end of day, rounded UP to the cent."""
    accrued = _int(accrued_micro, "accrued_micro")
    if accrued < 0:
        raise InvariantError(f"fills: accrued fees cannot be negative, got {accrued}")
    return cdiv(accrued, MICROS_PER_CENT)


def _is_crossed(q: Quote) -> bool:
    """`ask <= bid` with a positive bid: crossed or locked (`crossed_or_locked`; unusable on both sides)."""
    return q.bid > 0 and q.ask <= q.bid


def _usable_on_side(leg: OrderLeg, q: Quote) -> bool:
    """Per-side usability of 10.4 for PRICING: BUY needs an ask (`usable_buy`); SELL-to-open needs a bid too; SELL-to-close
    needs only an ask. A crossed / locked row is unusable on both sides (`usable_*` require `ask > bid`)."""
    if leg.side is Side.BUY:
        return q.usable_buy()
    if leg.position_intent is PositionIntent.STO:
        return q.bid > 0 and q.usable_sell_close()  # we never open by selling into a zero bid
    return q.usable_sell_close()


def _is_no_quote(leg: OrderLeg, q: Quote) -> bool:
    """The `no_quote` rule of 10.4, evaluated on its own (never as the `else` of the crossed rule).

    10.4 lists (a) any BUY leg without an ask (`ask <= 0`) and (b) a SELL leg of an OPEN order with `bid <= 0`. A market with no
    ask at all is (a)'s mirror on the sell side: the zero-bid sell-to-close exemption is stated for `bid == 0` **and `ask > 0`**,
    so a `0 x 0` row is no quote whichever side we are on. Both rules ignore crossing, so a BUY quoted `100 x 0` yields
    `no_quote` AND `crossed_or_locked` - the funnel must not attribute a genuinely absent ask to the wrong reason.
    """
    if q.ask <= 0:
        return True
    return leg.side is Side.SELL and leg.position_intent is PositionIntent.STO and q.bid <= 0


def _is_zero_bid_close(leg: OrderLeg, q: Quote) -> bool:
    """The condor-wing case: a sell-to-close leg quoted `0 x ask` is sold at 0 (10.4), never interpolated, never spread-checked."""
    return leg.side is Side.SELL and leg.position_intent is not PositionIntent.STO and q.bid == 0 < q.ask


# ======================================================================================================================
# BandFillModel
# ======================================================================================================================


class BandFillModel:
    """Implements `protocols.FillModel` (module docstring). Pure: holds config only, reads no clock and does no IO."""

    def __init__(self, cfg: Config) -> None:
        self._fills: FillsConfig = cfg.fills
        self._liq: LiquidityConfig = cfg.liquidity
        self._health: HealthConfig = cfg.health
        self._fees: FeesConfig = cfg.fees
        self._penny_all: tuple[str, ...] = tuple(cfg.universe.penny_all)
        self._headline: Band = headline_band(cfg)
        self._p_bp: tuple[int, ...] = tuple(orats_p_bp(n, cfg.fills.orats_p) for n in range(1, _ORATS_LEG_CLASSES + 1))
        self._max_rel_spread = _exact(cfg.fills.max_rel_spread, "fills.max_rel_spread")
        self._max_abs_spread = _int(cfg.fills.max_abs_spread_cents, "fills.max_abs_spread_cents")
        self._penalty_frac = _exact(cfg.fills.forced_penalty_frac_spread, "fills.forced_penalty_frac_spread")
        self._no_quote_pad = _int(cfg.fills.forced_no_quote_pad_cents, "fills.forced_no_quote_pad_cents")
        self._max_pct_size = _exact(cfg.liquidity.max_pct_displayed_size, "liquidity.max_pct_displayed_size")
        self._max_pct_oi = _exact(cfg.liquidity.max_pct_open_interest, "liquidity.max_pct_open_interest")
        self._max_quote_age_s = _int(cfg.health.max_quote_age_s, "health.max_quote_age_s")
        if self._max_abs_spread < 0 or self._no_quote_pad < 0 or self._max_quote_age_s < 0:
            raise InvariantError("fills: max_abs_spread_cents, forced_no_quote_pad_cents and max_quote_age_s must be >= 0")

    @property
    def headline(self) -> Band:
        """The headline band (Conventions): `orats` under `next_snapshot`, `worst` under `same_snapshot_worst`."""
        return self._headline

    def p_bp(self, n_legs: int) -> int:
        """`orats_p[min(n_legs, 4) - 1]` in basis points (10.3)."""
        n = _int(n_legs, "n_legs")
        if n < 1:
            raise InvariantError(f"fills: n_legs must be >= 1, got {n}")
        return self._p_bp[min(n, _ORATS_LEG_CLASSES) - 1]

    # --- shared plumbing ------------------------------------------------------------------------------------------------

    @staticmethod
    def _unit_ratio(leg: OrderLeg) -> None:
        """v1 order legs have `ratio == 1` (2.3 / 2.4; risk check 4 of 9.1 admits no other structure).

        `LegFill` (2.4) carries no ratio, so `net_of` could not weight a ratio leg and a ledger replay could not recover it:
        such a leg is refused outright rather than counted once in the net and twice in the mark and the fee.
        """
        if isinstance(leg.ratio, bool) or leg.ratio != 1:
            raise InvariantError(
                f"fills: order leg {leg.contract.occ} has ratio {leg.ratio!r}; v1 ratios are 1 (2.4) and LegFill carries no ratio"
            )

    @classmethod
    def _legs(cls, legs: Sequence[OrderLeg], chain: ChainSnapshot) -> list[OrderLeg]:
        checked = list(legs)
        if not checked:
            raise InvariantError("fills: an order has at least one leg")
        for leg in checked:
            if not isinstance(leg, OrderLeg):
                raise InvariantError(f"fills: legs must be OrderLeg, got {type(leg).__name__}")
            if leg.contract.underlying != chain.underlying:
                raise InvariantError(f"fills: leg {leg.contract.occ} priced against a {chain.underlying} chain (a wiring bug)")
            cls._unit_ratio(leg)
        return checked

    def _tick(self, underlying: str, px: Cents) -> int:
        return tick_cents(underlying, px, self._penny_all)

    def _too_wide(self, q: Quote) -> bool:
        """`spread > max(max_rel_spread * mid, max_abs_spread_cents)` with `mid = (bid + ask) / 2`, in exact rationals (10.4)."""
        spread = q.ask - q.bid
        if spread <= self._max_abs_spread:
            return False
        return 2 * spread > self._max_rel_spread * (q.bid + q.ask)

    def _is_stale(self, q: Quote, chain: ChainSnapshot) -> bool:
        """Recorded / live data only: the feed timestamp is older than `health.max_quote_age_s` at the snapshot time (10.4).
        A missing timestamp on timed data is an unknown age and counts as stale (fail closed)."""
        if chain.fidelity not in _TIMED_FIDELITIES:
            return False
        if q.quote_ts is None:
            return True
        return (chain.ts - q.quote_ts).total_seconds() > self._max_quote_age_s

    def _no_quote_price(self, leg: OrderLeg, q: Quote | None, chain: ChainSnapshot, last_mark: int | None) -> Cents:
        """The no-usable-quote fallback of 10.4: SELL at `max(intrinsic - pad, 0)`, BUY at `max(intrinsic, last leg mark) + pad`.

        `last_mark` is the caller's last per-leg mark when it has one; otherwise the most conservative price visible on the row
        itself (`max(bid, ask)` of a crossed / one-sided quote, 0 when the contract is missing) stands in for it.
        """
        pad = self._no_quote_pad
        if leg.side is Side.SELL:
            return max(intrinsic_cents(leg.contract, chain.spot, up=False) - pad, 0)
        seen = 0 if q is None else max(q.bid, q.ask, 0)
        last = seen if last_mark is None else max(_int(last_mark, "last_marks[occ]"), 0)
        return max(intrinsic_cents(leg.contract, chain.spot, up=True), last) + pad

    def _forced_price(self, leg: OrderLeg, q: Quote, worst: Cents) -> Cents:
        """Worst band plus `max(1 tick, cdiv(spread * forced_penalty_frac_spread))` against us; a sale never goes below 0 (10.4)."""
        penalty = max(self._tick(leg.contract.underlying, worst), _ceil(self._penalty_frac * (q.ask - q.bid)))
        return worst + penalty if leg.side is Side.BUY else max(worst - penalty, 0)

    # --- FillModel: check ----------------------------------------------------------------------------------------------

    def check(self, legs: Sequence[OrderLeg], qty: int, chain: ChainSnapshot, *, mandatory: bool) -> tuple[str, ...]:
        """`vocab.FILL_REJECTS` codes in their fixed order, deduplicated across legs; `()` = fillable. Band-independent (10.4).

        Every rule is evaluated independently, so one leg can carry several codes (a BUY quoted `100 x 0` is `no_quote` -
        rule (a), no ask - and `crossed_or_locked`); `check` returns a tuple precisely so each reason reaches the funnel.
        `mandatory=True` always returns `()`: a mandatory exit / kill order is never rejected (it is force-priced instead).
        """
        if mandatory:
            return ()
        n = _int(qty, "qty")
        if n < 1:
            raise InvariantError(f"fills: qty must be >= 1, got {n}")
        codes: set[str] = set()
        for leg in self._legs(legs, chain):
            q = chain.quote(leg.contract)
            if q is None:
                codes.add(_REJECT_MISSING)
                continue
            contracts = n  # `_legs` has pinned every ratio to 1, so `contracts = qty * n_legs` counts each leg once (10.7)
            if _is_no_quote(leg, q):
                codes.add(_REJECT_NO_QUOTE)  # a BUY without an ask, or a SELL-to-open into a zero bid (never a zero-bid sell-to-close)
            if _is_crossed(q):
                codes.add(_REJECT_CROSSED)
            if not _is_zero_bid_close(leg, q) and self._too_wide(q):
                codes.add(_REJECT_WIDE_SPREAD)
            size = q.ask_size if leg.side is Side.BUY else q.bid_size  # the side we hit
            if size is not None and size > 0 and contracts > self._max_pct_size * size:
                codes.add(_REJECT_SIZE)
            if q.oi_prev is not None and contracts > self._max_pct_oi * q.oi_prev:
                codes.add(_REJECT_OI)
            if self._is_stale(q, chain):
                codes.add(_REJECT_STALE)
        return tuple(sorted(codes, key=_REJECT_ORDER.__getitem__))

    # --- FillModel: price ----------------------------------------------------------------------------------------------

    def price(
        self,
        legs: Sequence[OrderLeg],
        chain: ChainSnapshot,
        *,
        mandatory: bool,
        last_marks: Mapping[str, int] | None = None,
    ) -> tuple[BandPrices, tuple[LegFill, ...], str]:
        """`(signed net cents/share per band, per-leg detail, quality)` (10.3 / 10.4).

        Non-mandatory: every usable leg gets the three bands of 10.3; a sell-to-close leg quoted `0 x ask` is sold at 0 on all
        bands. Mandatory (forced fill): every usable leg takes the worst band plus the penalty on all three bands. On either path a
        leg without a usable quote (missing, crossed / locked, no ask, sell-to-open into a zero bid) takes the no-quote fallback on
        all three bands and the fill is `"degraded"`; otherwise `"ok"`. `last_marks` (occ -> last per-leg mark, cents) is the
        optional "last leg mark" of the BUY fallback; without it the row's own prices stand in (see `_no_quote_price`).
        """
        checked = self._legs(legs, chain)
        p_bp = self.p_bp(len(checked))  # 10.3's leg class is the NUMBER of legs (every ratio is 1, `_legs` refuses the rest)
        fills: list[LegFill] = []
        degraded = False
        for leg in checked:
            q = chain.quote(leg.contract)
            occ = leg.contract.occ
            if q is None or not _usable_on_side(leg, q):
                degraded = True
                px = self._no_quote_price(leg, q, chain, None if last_marks is None else last_marks.get(occ))
                bid, ask = (0, 0) if q is None else (q.bid, q.ask)
                fills.append(LegFill(occ=occ, side=leg.side, bid=bid, ask=ask, orats=px, worst=px, mid=px))
                continue
            if _is_zero_bid_close(leg, q):
                orats, worst, mid = 0, 0, 0  # sold at 0 on all three bands: nobody pays inside a zero-bid market
            else:
                orats, worst, mid = leg_band_prices(leg.side, q.bid, q.ask, p_bp)
            if mandatory:
                orats = worst = mid = self._forced_price(leg, q, worst)
            fills.append(LegFill(occ=occ, side=leg.side, bid=q.bid, ask=q.ask, orats=orats, worst=worst, mid=mid))
        leg_fills = tuple(fills)
        return net_of(leg_fills), leg_fills, QUALITY_DEGRADED if degraded else QUALITY_OK

    # --- FillModel: liquidation ----------------------------------------------------------------------------------------

    def liquidation(
        self,
        structure: Structure,
        chain: ChainSnapshot,
        last: tuple[int, int] | None,
        *,
        last_asks: Mapping[str, int] | None = None,
    ) -> tuple[int, int, bool]:
        """`(liq_value, mid_value, stale)` in signed cents/share: what closing would COST now (negative = we would receive).

        `liq_value = sum(ask of short legs) - sum(bid of long legs)`; `mid_value` at mids, rounded against us (a short leg's mid
        rounds up, a long leg's down: a zero-bid long wing is `ask // 2`). A long leg with `bid == 0` is a VALID mark of 0. A leg
        is unusable only when its quote is missing, a short leg has no ask, or the quote is crossed / locked with a positive bid;
        then `stale = True` and the previous `last = (liq_value, mid_value)` is returned unchanged. With `last = None` (no previous
        mark, or the caller has retired one after `health.max_stale_mark_sessions`) the fallback bound of 10.5 prices the unusable
        legs instead - long at intrinsic, short at `max(intrinsic, last ask)` - while usable legs keep their real quotes.

        `last_asks` (occ -> the last ask we saw for that leg, cents) is 10.5's "last ask"; it is an extension of the `FillModel`
        Protocol (3.4 passes only the structure-level `last`), so without it the most conservative price visible on the row itself
        stands in - which is the BID when a short leg is quoted `bid x 0`, and therefore possibly OPTIMISTIC about what closing
        that leg costs. Each leg is weighted by `Leg.ratio` (always 1 in v1), so the mark stays correct if ratios are admitted.
        """
        if not isinstance(structure, Structure) or not structure.legs:
            raise InvariantError("fills: liquidation needs a Structure with at least one leg")
        if structure.underlying != chain.underlying:
            raise InvariantError(f"fills: structure {structure.structure_id} marked against a {chain.underlying} chain (a wiring bug)")
        if last is not None:
            last = (_int(last[0], "last[0]"), _int(last[1], "last[1]"))
        liq = mid = 0
        stale = False
        for leg in structure.legs:
            q = chain.quote(leg.contract)
            short = leg.side is Side.SELL
            usable = q is not None and not _is_crossed(q) and (not short or q.ask > 0)
            ratio = leg.ratio
            if usable and q is not None:
                if short:
                    liq += q.ask * ratio  # buying the short back costs the ask
                    mid += cdiv(q.bid + q.ask, 2) * ratio
                else:
                    liq -= q.bid * ratio  # selling the long fetches the bid (0 for a zero-bid wing: a valid mark)
                    mid -= ((q.bid + q.ask) // 2) * ratio
                continue
            stale = True
            if last is not None:
                continue
            if short:  # the fallback bound of 10.5: short = max(intrinsic, last ask); the row's own prices stand in without one
                seen = 0 if q is None else max(q.bid, q.ask, 0)
                prior = 0 if last_asks is None else max(_int(last_asks.get(leg.contract.occ, 0), "last_asks[occ]"), 0)
                bound = max(intrinsic_cents(leg.contract, chain.spot, up=True), prior, seen)
                liq += bound * ratio
                mid += bound * ratio
            else:  # long = intrinsic
                bound = intrinsic_cents(leg.contract, chain.spot, up=False)
                liq -= bound * ratio
                mid -= bound * ratio
        if stale and last is not None:
            return (last[0], last[1], True)
        return (liq, mid, stale)

    # --- FillModel: fees -----------------------------------------------------------------------------------------------

    def fees_micro(self, legs: Sequence[OrderLeg], qty: int, leg_fills: Sequence[LegFill]) -> Micros:
        """The per-fill fee of 10.7 in micro-dollars through `structmath.fill_fees_micro`:

        `contracts = qty * n_legs`, `sold = qty * n_sell_legs`, `sell_notional_cents = sum(headline leg price * 100 * qty)` over
        the SELL legs (the headline band: `orats` under `next_snapshot`, `worst` under `same_snapshot_worst`). `leg_fills` must be
        aligned with `legs` (same occ and side): a mismatch is a wiring bug. Every leg counts once - v1 ratios are 1 and any other
        ratio is refused here exactly as it is in `price` / `check`, so the fee can never count a leg the net does not.
        """
        n = _int(qty, "qty")
        if n < 1:
            raise InvariantError(f"fills: qty must be >= 1, got {n}")
        order_legs = list(legs)
        fills = list(leg_fills)
        if not order_legs or len(order_legs) != len(fills):
            raise InvariantError(f"fills: {len(fills)} leg fills for {len(order_legs)} order legs")
        contracts = sold = notional = 0
        for leg, fill in zip(order_legs, fills, strict=True):
            if fill.occ != leg.contract.occ or fill.side is not leg.side:
                raise InvariantError(
                    f"fills: leg fill {fill.occ}/{fill.side.value} does not match order leg {leg.contract.occ}/{leg.side.value}"
                )
            self._unit_ratio(leg)
            count = n
            contracts += count
            if leg.side is Side.SELL:
                sold += count
                price = fill.orats if self._headline is Band.ORATS else fill.worst
                notional += max(_int(price, "LegFill price"), 0) * MULTIPLIER * count
        return fill_fees_micro(contracts, sold, notional, self._fees)


if TYPE_CHECKING:
    # Static proof, checked by mypy: BandFillModel implements the FillModel Protocol of 3.4.
    from jevbot.protocols import FillModel

    _BAND_FILL_MODEL_IS_A_FILL_MODEL: FillModel = BandFillModel(Config())
