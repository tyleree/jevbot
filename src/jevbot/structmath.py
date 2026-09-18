"""THE structure maths (DESIGN.md 9.2) and THE per-leg liquidity filter (section 8). Pure, integer, no IO.

`candidates.py`, `risk.py`, `portfolio.py`, `state.py` (the PNL bucket's `max_*_mid` terms) and the paper pre-submission gate MUST
call these functions; none of them may re-implement a formula (a second implementation anywhere is a review reject).

Units (DESIGN "Conventions"): option prices and widths are integer **cents per share**, the contract multiplier is 100, so every
per-contract result is integer **cents**; strikes are milli-dollars; fees accrue in micro-dollars. `net` is the signed net price of
the structure in Alpaca's mleg convention: **positive = net debit (we pay), negative = net credit (we receive)**. `widths` is
`Structure.wing_widths` = `(put wing, call wing)`, 0 where the wing is absent.

The 9.2 table (per contract; `w` = width, `n` = signed net, credit = `-n`):

    kind                       max_loss_pc                          max_profit_pc     breakeven(s)                bp_required_pc
    long_call                  n * 100 + fee_rt                     None              K + n                       n * 100
    long_put                   n * 100 + fee_rt                     None (reported)   K - n                       n * 100
    call / put debit spread    n * 100 + fee_rt                     (w - n) * 100     K_long +/- n                n * 100
    call / put credit spread   (w - credit) * 100 + fee_rt          credit * 100      K_short -/+ credit          (w - credit) * 100 * bp_haircut_mult
    iron_condor                (max(w_put, w_call) - credit) * 100  credit * 100      K_sp - credit, K_sc + credit  sum_wings: ((w_put + w_call) - credit) * 100 * mult
                               + fee_rt                                                                           max_wing:  (max(w_put, w_call) - credit) * 100 * mult

Every formula is `-(min P&L at expiry)` resp. `max P&L at expiry` of the payoff and therefore stays meaningful for a wrong-signed or
oversized `net`: the functions do NOT raise on bad economics - a result `<= 0` is how `candidates.py` detects `economics_invalid`
(section 8). What DOES raise `InvariantError` (a bug, never a soft fail) is malformed input: non-integer money, negative widths, a
wing of a right the kind does not have, legs that do not match the kind's template in `breakevens`.

Rounding is always against us: config floats enter as the exact decimal the operator wrote (`Fraction(repr(x))`, never binary
noise) and every result that is not an integer is rounded UP (fees, buying power). Nothing here returns or stores a float.
"""

import math
import operator
from collections.abc import Sequence
from fractions import Fraction
from typing import Final, SupportsIndex

from jevbot.config import FeesConfig, LiquidityConfig, RiskConfig
from jevbot.errors import InvariantError
from jevbot.types import Cents, Leg, Micros, Quote, Right, Side, StructureKind
from jevbot.vocab import LIQUIDITY_REJECTS

__all__ = [
    "MULTIPLIER",
    "bp_required_pc",
    "breakevens",
    "defined_risk_ok",
    "fee_round_trip",
    "fill_fees_micro",
    "leg_liquidity_rejects",
    "max_loss_pc",
    "max_profit_pc",
]

MULTIPLIER: Final = 100  # shares per contract: one contract at price p (cents/share) is worth p * 100 cents
_MICROS_PER_USD: Final = 1_000_000
_MICROS_PER_CENT: Final = 10_000
_MILLI_PER_CENT: Final = 10  # strikes are milli-dollars (450.5 -> 450500)
_REL_SPREAD_MIN_MID2: Final = 100  # the relative-spread rule applies when mid >= 50 cents, i.e. bid + ask >= 100 (section 8)

_LIQ_BID, _LIQ_CROSSED, _LIQ_SPREAD, _LIQ_OI = LIQUIDITY_REJECTS

_SINGLES: Final = frozenset({StructureKind.LONG_CALL, StructureKind.LONG_PUT})
_DEBIT_VERTICALS: Final = frozenset({StructureKind.CALL_DEBIT, StructureKind.PUT_DEBIT})
_PUT_VERTICALS: Final = frozenset({StructureKind.PUT_DEBIT, StructureKind.PUT_CREDIT})
_CALL_VERTICALS: Final = frozenset({StructureKind.CALL_DEBIT, StructureKind.CALL_CREDIT})

# The leg template of every kind (section 8 table): role -> (side when OPENING, right). BUY = long leg, SELL = short leg.
_TEMPLATES: Final[dict[StructureKind, dict[str, tuple[Side, Right]]]] = {
    StructureKind.LONG_CALL: {"long": (Side.BUY, Right.CALL)},
    StructureKind.LONG_PUT: {"long": (Side.BUY, Right.PUT)},
    StructureKind.CALL_DEBIT: {"long": (Side.BUY, Right.CALL), "short": (Side.SELL, Right.CALL)},
    StructureKind.PUT_DEBIT: {"long": (Side.BUY, Right.PUT), "short": (Side.SELL, Right.PUT)},
    StructureKind.CALL_CREDIT: {"short": (Side.SELL, Right.CALL), "long": (Side.BUY, Right.CALL)},
    StructureKind.PUT_CREDIT: {"short": (Side.SELL, Right.PUT), "long": (Side.BUY, Right.PUT)},
    StructureKind.IRON_CONDOR: {
        "long_put": (Side.BUY, Right.PUT),
        "short_put": (Side.SELL, Right.PUT),
        "short_call": (Side.SELL, Right.CALL),
        "long_call": (Side.BUY, Right.CALL),
    },
}


# ======================================================================================================================
# input hygiene
# ======================================================================================================================


def _int(value: object, name: str) -> int:
    """Money is integer (Conventions). Accepts any true integer (numpy ints included) as a Python int; bool / float / Decimal raise."""
    if isinstance(value, bool) or not isinstance(value, SupportsIndex):
        raise InvariantError(f"structmath: {name} must be an integer (money is integer cents), got {type(value).__name__}: {value!r}")
    return operator.index(value)


def _non_negative(value: object, name: str) -> int:
    number = _int(value, name)
    if number < 0:
        raise InvariantError(f"structmath: {name} must be >= 0, got {number}")
    return number


def _exact(value: float, name: str) -> Fraction:
    """A config number as the exact decimal the operator wrote: 0.15 -> 3/20, 0.0000206 -> 103/5000000 (never binary noise)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvariantError(f"structmath: {name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value) or value < 0:
        raise InvariantError(f"structmath: {name} must be finite and >= 0, got {value!r}")
    return Fraction(repr(value))


def _ceil(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def _kind(kind: StructureKind) -> StructureKind:
    try:
        return StructureKind(kind)
    except ValueError:
        raise InvariantError(f"structmath: unknown structure kind {kind!r}") from None


def _wings(kind: StructureKind, widths: tuple[Cents, Cents]) -> tuple[int, int]:
    """`Structure.wing_widths` = (put wing, call wing), validated against the kind: a wing of a right the kind does not have is a bug."""
    if not isinstance(widths, tuple | list) or len(widths) != 2:
        raise InvariantError(f"structmath: widths must be (put wing, call wing), got {widths!r}")
    w_put = _non_negative(widths[0], "widths[0] (put wing)")
    w_call = _non_negative(widths[1], "widths[1] (call wing)")
    if kind in _SINGLES and (w_put or w_call):
        raise InvariantError(f"structmath: {kind.value} is a single leg, widths must be (0, 0), got {(w_put, w_call)}")
    if kind in _PUT_VERTICALS and w_call:
        raise InvariantError(f"structmath: {kind.value} has no call wing, got widths {(w_put, w_call)}")
    if kind in _CALL_VERTICALS and w_put:
        raise InvariantError(f"structmath: {kind.value} has no put wing, got widths {(w_put, w_call)}")
    return w_put, w_call


# ======================================================================================================================
# 9.2 formulas
# ======================================================================================================================


def max_loss_pc(kind: StructureKind, widths: tuple[Cents, Cents], net: int, fee_rt: Cents) -> Cents:
    """Max loss per contract in cents, `fee_rt` (entry + estimated exit fees, `fee_round_trip`) included.

    `net` = signed cents/share at the band in question (pre-trade: the WORST band; `Position.max_loss`: the actual worst-band
    fill; the manage state's PNL bucket: `open_mid_at_decision` with `fee_rt = 0`). A result `<= 0` means invalid economics
    (a credit >= the width, a debit structure priced at a credit): callers reject it, this function does not raise.
    """
    kind = _kind(kind)
    w_put, w_call = _wings(kind, widths)
    n = _int(net, "net")
    fee = _non_negative(fee_rt, "fee_rt")
    if kind in _SINGLES or kind in _DEBIT_VERTICALS:
        return n * MULTIPLIER + fee  # the debit paid is all that can be lost
    # credit verticals and the condor: (w - credit) * 100 + fee_rt with credit = -n; the condor loses on ONE wing only, the wider one
    return (max(w_put, w_call) + n) * MULTIPLIER + fee


def max_profit_pc(kind: StructureKind, widths: tuple[Cents, Cents], net: int) -> Cents | None:
    """Max profit per contract in cents at `net` (pre-trade: the HEADLINE band), fees excluded. None = unbounded.

    A long call is unbounded; a long put is REPORTED as None too (9.2; its internal bound `(K - n) * 100` needs the strike and is
    not part of this contract).
    """
    kind = _kind(kind)
    w_put, w_call = _wings(kind, widths)
    n = _int(net, "net")
    if kind in _SINGLES:
        return None
    if kind in _DEBIT_VERTICALS:
        return (max(w_put, w_call) - n) * MULTIPLIER
    return -n * MULTIPLIER  # credit verticals and the condor keep the credit


def bp_required_pc(kind: StructureKind, widths: tuple[Cents, Cents], net: int, fee_rt: Cents, cfg: RiskConfig) -> Cents:
    """Buying power one contract reserves, in cents (Cboe minimums, B7.5), rounded UP; never negative.

    Debit structures pay the net debit; credit verticals post `(w - credit) * 100 * bp_haircut_mult`; the condor posts the sum of
    its wings or its wider wing less the credit, by `cfg.condor_bp_mode` (conservative `sum_wings` until probe P-ALP-5).

    `fee_rt` is part of the 3.7 signature but NO cell of the 9.2 buying-power column carries a fee term: this models the BROKER's
    requirement (it is cross-checked against `broker_options_bp`, 9.3, and calibrated by probe P-ALP-5), and the broker reserves
    no fees. It is validated and deliberately not added; fees are in `max_loss_pc`, which is what sizing is limited by.
    """
    kind = _kind(kind)
    w_put, w_call = _wings(kind, widths)
    n = _int(net, "net")
    _non_negative(fee_rt, "fee_rt")
    if kind in _SINGLES or kind in _DEBIT_VERTICALS:
        return max(n * MULTIPLIER, 0)
    mult = _exact(cfg.bp_haircut_mult, "risk.bp_haircut_mult")
    if kind is StructureKind.IRON_CONDOR:
        if cfg.condor_bp_mode == "sum_wings":
            width = w_put + w_call
        elif cfg.condor_bp_mode == "max_wing":
            width = max(w_put, w_call)
        else:
            raise InvariantError(f"structmath: unknown risk.condor_bp_mode {cfg.condor_bp_mode!r}")
    else:
        width = max(w_put, w_call)
    return max(_ceil((width + n) * MULTIPLIER * mult), 0)


# ----------------------------------------------------------------------------------------------------------------------
# leg templates: breakevens and the defined-risk rule
# ----------------------------------------------------------------------------------------------------------------------


def _roles(kind: StructureKind, legs: Sequence[Leg]) -> dict[str, Leg] | None:
    """The legs by template role, or None when count / rights / sides do not match the kind's template exactly."""
    template = _TEMPLATES[kind]
    if len(legs) != len(template):
        return None
    roles: dict[str, Leg] = {}
    for role, (side, right) in template.items():
        matches = [leg for leg in legs if leg.side == side and leg.contract.right == right]
        if len(matches) != 1:
            return None
        roles[role] = matches[0]
    return roles


def _strike_cents(leg: Leg, *, up: bool) -> int:
    """Strike in cents/share. A strike that is not a whole cent (sub-cent OCC strikes) rounds AGAINST us: a breakeven we must
    rise above rounds up, one we must stay below rounds down."""
    milli = leg.contract.strike_milli
    return -(-milli // _MILLI_PER_CENT) if up else milli // _MILLI_PER_CENT


def breakevens(kind: StructureKind, legs: Sequence[Leg], net: int) -> tuple[Cents, ...]:
    """Underlying price levels (cents) where the expiry P&L before fees is zero, ascending: one level, two for the condor.

    long call `K + n`; long put `K - n`; debit verticals `K_long +/- n`; credit verticals `K_short -/+ credit`; condor
    `(K_sp - credit, K_sc + credit)`. Legs that do not match the kind's template are a bug: `InvariantError`.
    """
    kind = _kind(kind)
    n = _int(net, "net")
    roles = _roles(kind, legs)
    if roles is None:
        raise InvariantError(f"structmath.breakevens: the legs do not match the {kind.value} template")
    if kind is StructureKind.LONG_CALL or kind is StructureKind.CALL_DEBIT:
        return (_strike_cents(roles["long"], up=True) + n,)  # profit above K_long + debit
    if kind is StructureKind.LONG_PUT or kind is StructureKind.PUT_DEBIT:
        return (_strike_cents(roles["long"], up=False) - n,)  # profit below K_long - debit
    if kind is StructureKind.PUT_CREDIT:
        return (_strike_cents(roles["short"], up=True) + n,)  # profit above K_short - credit
    if kind is StructureKind.CALL_CREDIT:
        return (_strike_cents(roles["short"], up=False) - n,)  # profit below K_short + credit
    return (_strike_cents(roles["short_put"], up=True) + n, _strike_cents(roles["short_call"], up=False) - n)


def defined_risk_ok(kind: StructureKind, legs: Sequence[Leg]) -> bool:
    """The leg-pairing rule of 9.1 check 5, for credit AND debit kinds. True iff `max_loss_pc` bounds the expiry payoff of `legs`.

    Every SELL leg is paired 1:1 with a BUY leg of the same right, expiry and ratio (1 in v1):
    - credit kinds (call / put credit spread, each condor wing): the BUY leg is FURTHER OTM than the SELL leg (calls: higher
      strike, puts: lower strike);
    - debit kinds (call / put debit spread): the BUY leg is CLOSER to the money (calls: lower strike, puts: higher strike) - a
      debit vertical is defined-risk *because* the long leg sits at the better strike;
    - long singles have no SELL leg;
    - the condor's short put is not above its short call (otherwise both wings can lose at once and `max(w) - credit` is no bound).
    The legs must also be the kind's exact template (count, rights, sides), one underlying, one expiry: a covered short of the
    wrong shape is not what the 9.2 formulas price. The caller (RiskEngine check 5) turns False into `InvariantError`.
    """
    try:
        kind = StructureKind(kind)
    except ValueError:
        return False
    roles = _roles(kind, legs)
    if roles is None:
        return False
    if any(isinstance(leg.ratio, bool) or leg.ratio != 1 for leg in legs):
        return False
    first = legs[0].contract
    if any(leg.contract.underlying != first.underlying or leg.contract.expiry != first.expiry for leg in legs):
        return False

    def strike(role: str) -> int:
        return roles[role].contract.strike_milli

    if kind in _SINGLES:
        return True
    if kind is StructureKind.CALL_DEBIT:
        return strike("long") < strike("short")
    if kind is StructureKind.PUT_DEBIT:
        return strike("long") > strike("short")
    if kind is StructureKind.CALL_CREDIT:
        return strike("long") > strike("short")
    if kind is StructureKind.PUT_CREDIT:
        return strike("long") < strike("short")
    return strike("long_put") < strike("short_put") <= strike("short_call") < strike("long_call")


# ======================================================================================================================
# fees (10.7) - the per-fill formula and its round-trip estimate
# ======================================================================================================================


def fill_fees_micro(contracts: int, sold: int, sell_notional_cents: Cents, fees: FeesConfig) -> Micros:
    """THE per-fill fee formula of 10.7 in micro-dollars:

        contracts * (orf + occ + cat + commission) * 1e6  +  sold * taf_sell * 1e6  +  ceil(sec_sell_rate * sell_notional_cents * 1e4)

    `contracts = qty * n_legs`; `sold = qty * n_sell_legs`; `sell_notional_cents = sum(headline leg price * 100 * qty)` over the
    legs SOLD in this fill. Exact rational arithmetic; the per-contract part and the SEC part are each rounded UP to a whole
    micro-dollar (at the shipped rates - 15,000 / 25,000 / 300 / 0 and 3,290 micros - the per-contract part is whole already).
    Exposed so that `fills.py` and `fee_round_trip` share one arithmetic.
    """
    n_contracts = _non_negative(contracts, "contracts")
    n_sold = _non_negative(sold, "sold")
    notional = _non_negative(sell_notional_cents, "sell_notional_cents")
    if n_sold > n_contracts:
        raise InvariantError(f"structmath: sold ({n_sold}) cannot exceed contracts ({n_contracts})")
    per_contract = sum((_exact(getattr(fees, name), f"fees.{name}") for name in ("orf", "occ", "cat", "commission")), Fraction(0))
    per_sold = _exact(fees.taf_sell, "fees.taf_sell")
    fixed = _ceil((n_contracts * per_contract + n_sold * per_sold) * _MICROS_PER_USD)
    sec = _ceil(_exact(fees.sec_sell_rate, "fees.sec_sell_rate") * notional * _MICROS_PER_CENT)
    return fixed + sec


def fee_round_trip(n_legs: int, n_sell_legs_open: int, open_leg_prices: Sequence[Cents], fees: FeesConfig) -> Cents:
    """`fee_rt` of 9.2: the 10.7 formula for ONE contract, one open plus one close at the current prices, rounded UP to the cent.

    `open_leg_prices` holds the current per-leg price (cents/share) of EVERY leg, in any order. Over a round trip each leg is sold
    exactly once - the `n_sell_legs_open` SELL legs at the open, the other `n_legs - n_sell_legs_open` (the long legs) at the
    close, estimated at today's prices - so the sell-side terms (TAF, SEC) cover all legs once and the per-contract terms (ORF,
    OCC, CAT, commission) cover all legs twice; the result does not depend on which price belongs to which side.

    10.7 rounds the SEC fee up PER FILL, and which legs a fill sells is not part of this signature. Each leg's SEC fee is therefore
    rounded up on its own: `sum(ceil(x_i)) >= ceil(sum of any subset) + ceil(sum of the rest)`, so `fee_rt` is never below the two
    fills' fees at these prices, whichever legs open short (rounding is against us; the excess is below `n_legs` micro-dollars).
    """
    legs = _int(n_legs, "n_legs")
    sells_open = _int(n_sell_legs_open, "n_sell_legs_open")
    if legs < 1 or not 0 <= sells_open <= legs:
        raise InvariantError(f"structmath.fee_round_trip: need n_legs >= 1 and 0 <= n_sell_legs_open <= n_legs, got {legs}, {sells_open}")
    prices = [_non_negative(price, "open_leg_prices[i]") for price in open_leg_prices]
    if len(prices) != legs:
        raise InvariantError(f"structmath.fee_round_trip: {len(prices)} leg prices for {legs} legs")
    sells_close = legs - sells_open
    per_contract_and_taf = fill_fees_micro(2 * legs, sells_open + sells_close, 0, fees)
    sec = sum(fill_fees_micro(0, 0, price * MULTIPLIER, fees) for price in prices)
    return -(-(per_contract_and_taf + sec) // _MICROS_PER_CENT)


# ======================================================================================================================
# section 8: THE per-leg liquidity filter (entry side only; closes are never blocked by it, 10.4)
# ======================================================================================================================


def leg_liquidity_rejects(quote: Quote, *, sold: bool, cfg: LiquidityConfig) -> tuple[str, ...]:
    """`vocab.LIQUIDITY_REJECTS` codes in their fixed order; `()` = the leg is eligible.

    - `liq:bid`     bid below `min_bid_cents_sold` (10c) on a leg we sell, `min_bid_cents_bought` (1c) on a leg we buy;
    - `liq:crossed` not `ask > bid` (crossed, locked or no ask);
    - `liq:spread`  `(ask - bid) / mid > max_rel_spread` when `mid >= 50c`, else `ask - bid > max_abs_spread_cents`;
    - `liq:oi`      `oi_prev < min_open_interest`; a missing `oi_prev` passes only with `allow_missing_open_interest`.
    No same-day volume anywhere (0.1 item 12). Integer / exact-rational comparisons: a spread exactly on a limit passes.
    """
    bid = _int(quote.bid, "quote.bid")
    ask = _int(quote.ask, "quote.ask")
    rejects: list[str] = []

    if bid < (cfg.min_bid_cents_sold if sold else cfg.min_bid_cents_bought):
        rejects.append(_LIQ_BID)
    if not ask > bid:
        rejects.append(_LIQ_CROSSED)

    spread, mid2 = ask - bid, ask + bid
    if mid2 >= _REL_SPREAD_MIN_MID2:
        # (ask - bid) / mid <= r   <=>   2 * (ask - bid) <= r * (bid + ask), mid = (bid + ask) / 2
        too_wide = 2 * spread > _exact(cfg.max_rel_spread, "liquidity.max_rel_spread") * mid2
    else:
        too_wide = spread > cfg.max_abs_spread_cents
    if too_wide:
        rejects.append(_LIQ_SPREAD)

    if quote.oi_prev is None:
        if not cfg.allow_missing_open_interest:
            rejects.append(_LIQ_OI)
    elif _int(quote.oi_prev, "quote.oi_prev") < cfg.min_open_interest:
        rejects.append(_LIQ_OI)
    return tuple(rejects)
