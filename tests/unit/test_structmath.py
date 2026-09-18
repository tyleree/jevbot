"""Unit tests for src/jevbot/structmath.py: THE 9.2 formulas and THE per-leg liquidity filter of section 8 (WP00).

Every expected number below was computed BY HAND from the 9.2 table / the 10.7 fee formula (the arithmetic is written out in the
comments) - never by calling the code under test. The payoff checks use an independent expiry-payoff function defined in this file.
"""

from datetime import date
from itertools import pairwise

import msgspec
import numpy as np
import pytest

from jevbot import structmath
from jevbot.config import FeesConfig, LiquidityConfig, RiskConfig
from jevbot.errors import InvariantError
from jevbot.structmath import (
    bp_required_pc,
    breakevens,
    defined_risk_ok,
    fee_round_trip,
    fill_fees_micro,
    leg_liquidity_rejects,
    max_loss_pc,
    max_profit_pc,
)
from jevbot.types import Leg, OptionContract, Quote, Right, Side, Structure, StructureKind
from jevbot.vocab import LIQUIDITY_REJECTS

EXPIRY = date(2026, 10, 16)
RISK = RiskConfig()
FEES = FeesConfig()
LIQ = LiquidityConfig()

K = StructureKind


def leg(right: Right, strike_usd: float, side: Side, *, expiry: date = EXPIRY, underlying: str = "SPY", ratio: int = 1) -> Leg:
    contract = OptionContract(underlying=underlying, expiry=expiry, right=right, strike_milli=round(strike_usd * 1000))
    return Leg(contract=contract, side=side, ratio=ratio)


def call(strike_usd: float, side: Side, **kw: object) -> Leg:
    return leg(Right.CALL, strike_usd, side, **kw)  # type: ignore[arg-type]


def put(strike_usd: float, side: Side, **kw: object) -> Leg:
    return leg(Right.PUT, strike_usd, side, **kw)  # type: ignore[arg-type]


BUY, SELL = Side.BUY, Side.SELL

# One well-formed example per kind: (legs, widths = (put wing, call wing) in cents/share)
LEGS: dict[StructureKind, tuple[Leg, ...]] = {
    K.LONG_CALL: (call(450, BUY),),
    K.LONG_PUT: (put(440, BUY),),
    K.CALL_DEBIT: (call(450, BUY), call(460, SELL)),
    K.PUT_DEBIT: (put(440, SELL), put(450, BUY)),
    K.PUT_CREDIT: (put(435, BUY), put(440, SELL)),
    K.CALL_CREDIT: (call(460, SELL), call(466, BUY)),
    K.IRON_CONDOR: (put(430, BUY), put(435, SELL), call(465, SELL), call(472, BUY)),
}
WIDTHS: dict[StructureKind, tuple[int, int]] = {
    K.LONG_CALL: (0, 0),
    K.LONG_PUT: (0, 0),
    K.CALL_DEBIT: (0, 1000),
    K.PUT_DEBIT: (1000, 0),
    K.PUT_CREDIT: (500, 0),
    K.CALL_CREDIT: (0, 600),
    K.IRON_CONDOR: (500, 700),
}


def expiry_pnl_pc(legs: tuple[Leg, ...], net: int, s_cents: int) -> int:
    """Independent reference: P&L per contract in cents at expiry with the underlying at `s_cents`, before fees.
    Long legs earn their intrinsic value, short legs owe it; `net` (signed, + = debit) was paid at the open."""
    value = 0
    for one in legs:
        assert one.contract.strike_milli % 10 == 0, "the reference payoff works in whole cents"
        strike = one.contract.strike_milli // 10
        intrinsic = max(s_cents - strike, 0) if one.contract.right is Right.CALL else max(strike - s_cents, 0)
        value += intrinsic if one.side is Side.BUY else -intrinsic
    return (value - net) * 100


def price_grid(legs: tuple[Leg, ...]) -> list[int]:
    strikes = sorted(one.contract.strike_milli // 10 for one in legs)
    points = {0, 1, 2 * strikes[-1], 3 * strikes[-1]}
    for strike in strikes:
        points.update((strike - 1, strike, strike + 1))
    for lo, hi in pairwise(strikes):
        points.add((lo + hi) // 2)
    points.update(range(0, 2 * strikes[-1], 250))
    return sorted(points)


# ======================================================================================================================
# The fixtures above agree with types.Structure (widths are what callers will pass)
# ======================================================================================================================


@pytest.mark.parametrize("kind", list(K))
def test_example_widths_are_structure_wing_widths(kind: StructureKind) -> None:
    ordered = tuple(sorted(LEGS[kind], key=lambda one: (one.contract.right is Right.CALL, one.contract.strike_milli)))
    structure = Structure(kind=kind, underlying="SPY", expiry=EXPIRY, last_session=EXPIRY, legs=ordered)
    assert structure.wing_widths == WIDTHS[kind]


# ======================================================================================================================
# 9.2 goldens per kind (hand-computed)
# ======================================================================================================================

# kind, net (worst band), net (headline band), fee_rt, max_loss, max_profit at the headline net, breakevens at the worst net, bp
GOLDENS = [
    # long call 450 @ 3.20 debit: loss 320*100 + 10 = 32,010; unbounded profit; BE 45,000 + 320; BP = the debit 32,000
    (K.LONG_CALL, 320, 315, 10, 32_010, None, (45_320,), 32_000),
    # long put 440 @ 2.75: loss 275*100 + 10 = 27,510; profit REPORTED as None; BE 44,000 - 275; BP 27,500
    (K.LONG_PUT, 275, 270, 10, 27_510, None, (43_725,), 27_500),
    # call debit 450/460 (w = 1,000) @ 4.10 worst, 4.00 headline: loss 410*100 + 18 = 41,018; profit (1000 - 400)*100 = 60,000;
    # BE = K_long + n = 45,000 + 410; BP 41,000
    (K.CALL_DEBIT, 410, 400, 18, 41_018, 60_000, (45_410,), 41_000),
    # put debit 450/440 (w = 1,000) @ 3.80 worst, 3.70 headline: loss 38,018; profit (1000 - 370)*100 = 63,000; BE 45,000 - 380; BP 38,000
    (K.PUT_DEBIT, 380, 370, 18, 38_018, 63_000, (44_620,), 38_000),
    # put credit 440/435 (w = 500) @ 1.00 credit worst, 1.06 headline - the 9.3 worked example: (500 - 100)*100 + 20 = 40,020;
    # profit 106*100 = 10,600; BE = K_short - credit = 44,000 - 100; BP (500 - 100)*100 * 1.00 = 40,000
    (K.PUT_CREDIT, -100, -106, 20, 40_020, 10_600, (43_900,), 40_000),
    # call credit 460/466 (w = 600) @ 1.25 credit - the other 9.3 example: (600 - 125)*100 + 20 = 47,520; profit 13,000 at 1.30;
    # BE = K_short + credit = 46,000 + 125; BP 47,500
    (K.CALL_CREDIT, -125, -130, 20, 47_520, 13_000, (46_125,), 47_500),
    # condor 430/435 puts (w 500), 465/472 calls (w 700) @ 2.10 credit worst, 2.20 headline: (max(500, 700) - 210)*100 + 35 = 49,035;
    # profit 22,000; BEs 43,500 - 210 and 46,500 + 210; BP sum_wings ((500 + 700) - 210)*100 = 99,000
    (K.IRON_CONDOR, -210, -220, 35, 49_035, 22_000, (43_290, 46_710), 99_000),
]


@pytest.mark.parametrize(("kind", "net_worst", "net_headline", "fee_rt", "loss", "profit", "bes", "bp"), GOLDENS)
def test_formula_goldens_per_kind(
    kind: StructureKind, net_worst: int, net_headline: int, fee_rt: int, loss: int, profit: int | None, bes: tuple[int, ...], bp: int
) -> None:
    assert max_loss_pc(kind, WIDTHS[kind], net_worst, fee_rt) == loss
    assert max_profit_pc(kind, WIDTHS[kind], net_headline) == profit
    assert breakevens(kind, LEGS[kind], net_worst) == bes
    assert bp_required_pc(kind, WIDTHS[kind], net_worst, fee_rt, RISK) == bp


def test_goldens_cover_all_seven_kinds() -> None:
    assert {row[0] for row in GOLDENS} == set(K)
    assert len(GOLDENS) == 7


@pytest.mark.parametrize(("kind", "net_worst", "net_headline", "fee_rt", "loss", "profit", "bes", "bp"), GOLDENS)
def test_payoff_grid_maximum_equals_max_loss_for_all_seven_kinds(
    kind: StructureKind, net_worst: int, net_headline: int, fee_rt: int, loss: int, profit: int | None, bes: tuple[int, ...], bp: int
) -> None:
    legs = LEGS[kind]
    pnls = [expiry_pnl_pc(legs, net_worst, s) for s in price_grid(legs)]
    # fees aside the grid maximum of the loss EQUALS max_loss_pc; with fees the loss never exceeds it
    assert max(-pnl for pnl in pnls) == max_loss_pc(kind, WIDTHS[kind], net_worst, 0)
    assert max(-pnl for pnl in pnls) + fee_rt == loss
    # bounded kinds: the grid maximum of the profit equals max_profit_pc at the same net
    bounded = max_profit_pc(kind, WIDTHS[kind], net_worst)
    if bounded is not None:
        assert max(pnls) == bounded
    # the P&L is exactly zero AT every breakeven and changes sign across it
    for level in bes:
        assert expiry_pnl_pc(legs, net_worst, level) == 0
        assert expiry_pnl_pc(legs, net_worst, level - 1) * expiry_pnl_pc(legs, net_worst, level + 1) < 0


def test_long_put_internal_bound_is_strike_minus_debit() -> None:
    # 9.2: "None (reported); (K - n) * 100 internally" - the payoff at S = 0 is that bound, the function reports None
    assert max_profit_pc(K.LONG_PUT, (0, 0), 275) is None
    assert expiry_pnl_pc(LEGS[K.LONG_PUT], 275, 0) == (44_000 - 275) * 100


def test_pnl_bucket_usage_mid_price_without_fees() -> None:
    # state.py evaluates both bounds at open_mid_at_decision with fee_rt = 0 (5.5 PNL bucket)
    assert max_loss_pc(K.PUT_CREDIT, (500, 0), -103, 0) == 39_700
    assert max_profit_pc(K.PUT_CREDIT, (500, 0), -103) == 10_300
    assert max_loss_pc(K.LONG_CALL, (0, 0), 318, 0) == 31_800


def test_results_are_plain_python_ints_never_floats() -> None:
    risk = msgspec.structs.replace(RISK, bp_haircut_mult=1.37)
    for kind, net_worst, net_headline, fee_rt, *_ in GOLDENS:
        assert type(max_loss_pc(kind, WIDTHS[kind], net_worst, fee_rt)) is int
        assert type(bp_required_pc(kind, WIDTHS[kind], net_worst, fee_rt, risk)) is int
        profit = max_profit_pc(kind, WIDTHS[kind], net_headline)
        assert profit is None or type(profit) is int
        assert all(type(level) is int for level in breakevens(kind, LEGS[kind], net_worst))


def test_kind_may_be_given_as_its_string_value() -> None:
    assert max_loss_pc("put_credit_spread", (500, 0), -100, 20) == 40_020  # type: ignore[arg-type]
    with pytest.raises(InvariantError, match="unknown structure kind"):
        max_loss_pc("butterfly", (500, 0), -100, 20)  # type: ignore[arg-type]


# ======================================================================================================================
# Invalid economics are REPORTED (<= 0), malformed input RAISES
# ======================================================================================================================


def test_invalid_economics_do_not_raise_they_return_non_positive_numbers() -> None:
    # section 8: `max_loss_per_contract > 0 and credit < width and debit < width else economics_invalid` - the caller decides
    assert max_loss_pc(K.PUT_CREDIT, (500, 0), -500, 0) == 0  # credit == width
    assert max_loss_pc(K.PUT_CREDIT, (500, 0), -520, 0) == -2_000  # credit > width: "loss" negative = arbitrage-looking quote
    assert max_loss_pc(K.CALL_DEBIT, (0, 1000), -5, 0) == -500  # a debit spread quoted at a credit
    assert max_profit_pc(K.CALL_DEBIT, (0, 1000), 1_050) == -5_000  # debit > width: can never profit
    # a credit structure quoted at a DEBIT still gets its true bound: the width plus what was paid
    assert max_loss_pc(K.CALL_CREDIT, (0, 600), 10, 0) == 61_000
    assert expiry_pnl_pc(LEGS[K.CALL_CREDIT], 10, 100_000) == -61_000


def test_buying_power_is_never_negative() -> None:
    assert bp_required_pc(K.PUT_CREDIT, (500, 0), -520, 0, RISK) == 0
    assert bp_required_pc(K.LONG_CALL, (0, 0), -3, 0, RISK) == 0
    assert bp_required_pc(K.IRON_CONDOR, (100, 100), -300, 0, RISK) == 0


@pytest.mark.parametrize(
    ("kind", "widths"),
    [
        (K.LONG_CALL, (0, 500)),  # a single leg has no wing
        (K.LONG_PUT, (500, 0)),
        (K.PUT_CREDIT, (0, 500)),  # wrong wing
        (K.PUT_CREDIT, (500, 500)),  # a wing the kind does not have
        (K.PUT_DEBIT, (500, 1)),
        (K.CALL_CREDIT, (500, 0)),
        (K.CALL_DEBIT, (1, 500)),
        (K.IRON_CONDOR, (-500, 700)),  # negative width
        (K.PUT_CREDIT, (500,)),  # not a pair
        (K.PUT_CREDIT, (500, 0, 0)),
        (K.PUT_CREDIT, 500),
    ],
)
def test_malformed_widths_raise(kind: StructureKind, widths: object) -> None:
    with pytest.raises(InvariantError):
        max_loss_pc(kind, widths, -100, 0)  # type: ignore[arg-type]
    with pytest.raises(InvariantError):
        max_profit_pc(kind, widths, -100)  # type: ignore[arg-type]
    with pytest.raises(InvariantError):
        bp_required_pc(kind, widths, -100, 0, RISK)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [1.0, 100.5, "100", None, True, float("nan")])
def test_money_must_be_integer(bad: object) -> None:
    with pytest.raises(InvariantError, match="integer"):
        max_loss_pc(K.PUT_CREDIT, (500, 0), bad, 0)  # type: ignore[arg-type]
    with pytest.raises(InvariantError, match="integer"):
        max_loss_pc(K.PUT_CREDIT, (500, 0), -100, bad)  # type: ignore[arg-type]
    with pytest.raises(InvariantError, match="integer"):
        max_profit_pc(K.PUT_CREDIT, (500, 0), bad)  # type: ignore[arg-type]
    with pytest.raises(InvariantError, match="integer"):
        breakevens(K.PUT_CREDIT, LEGS[K.PUT_CREDIT], bad)  # type: ignore[arg-type]
    with pytest.raises(InvariantError, match="integer"):
        max_loss_pc(K.PUT_CREDIT, (bad, 0), -100, 0)  # type: ignore[arg-type]


def test_negative_fee_is_refused() -> None:
    with pytest.raises(InvariantError, match=">= 0"):
        max_loss_pc(K.PUT_CREDIT, (500, 0), -100, -1)
    with pytest.raises(InvariantError, match=">= 0"):
        bp_required_pc(K.PUT_CREDIT, (500, 0), -100, -1, RISK)


def test_numpy_integers_are_accepted_and_come_back_as_python_ints() -> None:
    result = max_loss_pc(K.PUT_CREDIT, (np.int64(500), np.int64(0)), np.int64(-100), np.int32(20))  # type: ignore[arg-type]
    assert result == 40_020
    assert type(result) is int
    with pytest.raises(InvariantError, match="integer"):
        max_loss_pc(K.PUT_CREDIT, (500, 0), np.float64(-100.0), 0)  # type: ignore[arg-type]


# ======================================================================================================================
# Buying power: haircut, rounding up, condor modes; fee_rt is validated but is no part of the requirement
# ======================================================================================================================


def test_bp_haircut_multiplies_credit_structures_only_and_rounds_up() -> None:
    risk = msgspec.structs.replace(RISK, bp_haircut_mult=1.25)
    assert bp_required_pc(K.PUT_CREDIT, (500, 0), -100, 0, risk) == 50_000  # 40,000 * 1.25
    assert bp_required_pc(K.CALL_DEBIT, (0, 1000), 410, 0, risk) == 41_000  # debit structures pay the net debit, no haircut
    assert bp_required_pc(K.LONG_PUT, (0, 0), 275, 0, risk) == 27_500
    # (500 - 101) * 100 = 39,900; * 1.001 = 39,939.9 -> 39,940 (rounded UP)
    assert bp_required_pc(K.PUT_CREDIT, (500, 0), -101, 0, msgspec.structs.replace(RISK, bp_haircut_mult=1.001)) == 39_940
    # 1.1 is not exactly representable in binary: 40,000 * 1.1 must be 44,000, not 44,001 (float noise would round up)
    assert bp_required_pc(K.PUT_CREDIT, (500, 0), -100, 0, msgspec.structs.replace(RISK, bp_haircut_mult=1.1)) == 44_000
    # 39,900 * 1.15 = 45,885 exactly
    assert bp_required_pc(K.PUT_CREDIT, (500, 0), -101, 0, msgspec.structs.replace(RISK, bp_haircut_mult=1.15)) == 45_885


def test_condor_bp_modes() -> None:
    widths, net = (500, 700), -210
    assert bp_required_pc(K.IRON_CONDOR, widths, net, 0, RISK) == 99_000  # default sum_wings: (1200 - 210) * 100
    max_wing = msgspec.structs.replace(RISK, condor_bp_mode="max_wing")
    assert bp_required_pc(K.IRON_CONDOR, widths, net, 0, max_wing) == 49_000  # (700 - 210) * 100
    both = msgspec.structs.replace(RISK, condor_bp_mode="max_wing", bp_haircut_mult=1.5)
    assert bp_required_pc(K.IRON_CONDOR, widths, net, 0, both) == 73_500
    # the condor's max LOSS is driven by the wider wing in either BP mode
    assert max_loss_pc(K.IRON_CONDOR, widths, net, 0) == 49_000
    assert max_loss_pc(K.IRON_CONDOR, (700, 500), net, 0) == 49_000


def test_unknown_condor_bp_mode_is_a_bug() -> None:
    class Cfg:
        bp_haircut_mult = 1.0
        condor_bp_mode = "average"

    with pytest.raises(InvariantError, match="condor_bp_mode"):
        bp_required_pc(K.IRON_CONDOR, (500, 700), -210, 0, Cfg())  # type: ignore[arg-type]


@pytest.mark.parametrize("mult", [float("nan"), float("inf"), -1.0])
def test_non_finite_or_negative_haircut_is_a_bug(mult: float) -> None:
    with pytest.raises(InvariantError, match="bp_haircut_mult"):
        bp_required_pc(K.PUT_CREDIT, (500, 0), -100, 0, msgspec.structs.replace(RISK, bp_haircut_mult=mult))


def test_a_haircut_that_is_not_a_number_is_a_bug() -> None:
    for bad in ("1.0", None, True):
        with pytest.raises(InvariantError, match="bp_haircut_mult"):
            bp_required_pc(K.PUT_CREDIT, (500, 0), -100, 0, msgspec.structs.replace(RISK, bp_haircut_mult=bad))
    # an integer haircut (TOML `bp_haircut_mult = 2`) is a number
    assert bp_required_pc(K.PUT_CREDIT, (500, 0), -100, 0, msgspec.structs.replace(RISK, bp_haircut_mult=2)) == 80_000


@pytest.mark.parametrize(("kind", "net_worst", "net_headline", "fee_rt", "loss", "profit", "bes", "bp"), GOLDENS)
def test_bp_column_of_9_2_has_no_fee_term(
    kind: StructureKind, net_worst: int, net_headline: int, fee_rt: int, loss: int, profit: int | None, bes: tuple[int, ...], bp: int
) -> None:
    # the broker reserves no fees: the requirement is the same whatever fee_rt is (fees live in max_loss_pc)
    assert bp_required_pc(kind, WIDTHS[kind], net_worst, 0, RISK) == bp
    assert bp_required_pc(kind, WIDTHS[kind], net_worst, 5_000, RISK) == bp
    assert max_loss_pc(kind, WIDTHS[kind], net_worst, 5_000) == loss - fee_rt + 5_000


# ======================================================================================================================
# Breakevens
# ======================================================================================================================


def test_breakevens_do_not_depend_on_leg_order() -> None:
    legs = LEGS[K.IRON_CONDOR]
    assert breakevens(K.IRON_CONDOR, tuple(reversed(legs)), -210) == (43_290, 46_710)
    assert breakevens(K.PUT_DEBIT, tuple(reversed(LEGS[K.PUT_DEBIT])), 380) == (44_620,)


def test_sub_cent_strikes_round_against_us() -> None:
    # a 22.505 strike is 2250.5 cents: a level we must rise above rounds UP, a level we must stay below rounds DOWN
    assert breakevens(K.LONG_CALL, (call(22.505, BUY),), 40) == (2_251 + 40,)
    assert breakevens(K.LONG_PUT, (put(22.505, BUY),), 40) == (2_250 - 40,)
    assert breakevens(K.PUT_CREDIT, (put(20, BUY), put(22.505, SELL)), -30) == (2_251 - 30,)
    assert breakevens(K.CALL_CREDIT, (call(22.505, SELL), call(25, BUY)), -30) == (2_250 + 30,)


@pytest.mark.parametrize(
    ("kind", "legs"),
    [
        (K.LONG_CALL, (put(440, BUY),)),  # wrong right
        (K.LONG_CALL, (call(450, SELL),)),  # wrong side
        (K.LONG_CALL, LEGS[K.CALL_DEBIT]),  # wrong count
        (K.PUT_CREDIT, LEGS[K.CALL_CREDIT]),  # the other right
        (K.PUT_CREDIT, (put(435, BUY), put(440, BUY))),  # no short leg
        (K.IRON_CONDOR, LEGS[K.PUT_CREDIT]),
        (K.IRON_CONDOR, (*LEGS[K.PUT_CREDIT], call(465, SELL), call(470, SELL))),  # two short calls
        (K.CALL_DEBIT, ()),
    ],
)
def test_breakevens_raise_when_legs_do_not_match_the_template(kind: StructureKind, legs: tuple[Leg, ...]) -> None:
    with pytest.raises(InvariantError, match="template"):
        breakevens(kind, legs, 100)


# ======================================================================================================================
# defined_risk_ok: the leg-pairing rule of 9.1 check 5, credit AND debit kinds
# ======================================================================================================================


@pytest.mark.parametrize("kind", list(K))
def test_defined_risk_accepts_every_well_formed_kind(kind: StructureKind) -> None:
    assert defined_risk_ok(kind, LEGS[kind]) is True
    assert defined_risk_ok(kind, tuple(reversed(LEGS[kind]))) is True  # order of the legs is irrelevant


def test_defined_risk_accepts_debit_verticals_whose_long_leg_is_closer_to_the_money() -> None:
    # QNT-2: demanding "BUY leg further OTM" would reject 2 of the 7 D3 structures
    assert defined_risk_ok(K.CALL_DEBIT, (call(450, BUY), call(460, SELL)))  # long call at the LOWER strike
    assert defined_risk_ok(K.PUT_DEBIT, (put(450, BUY), put(440, SELL)))  # long put at the HIGHER strike
    assert defined_risk_ok(K.CALL_CREDIT, (call(460, SELL), call(466, BUY)))  # credit kinds: long leg further OTM
    assert defined_risk_ok(K.PUT_CREDIT, (put(440, SELL), put(435, BUY)))


@pytest.mark.parametrize(
    ("kind", "legs", "why"),
    [
        (K.PUT_CREDIT, (put(440, BUY), put(435, SELL)), "mis-ordered: the long put is ABOVE the short put (that is a debit spread)"),
        (K.CALL_CREDIT, (call(466, SELL), call(460, BUY)), "mis-ordered: the long call is BELOW the short call"),
        (K.CALL_DEBIT, (call(460, BUY), call(450, SELL)), "mis-ordered debit: long call above the short call"),
        (K.PUT_DEBIT, (put(440, BUY), put(450, SELL)), "mis-ordered debit: long put below the short put"),
        (K.PUT_CREDIT, (put(440, SELL), put(440, BUY)), "zero width: both legs on one strike"),
        (K.PUT_CREDIT, (put(440, SELL),), "uncovered short put"),
        (K.CALL_CREDIT, (call(460, SELL),), "uncovered short call"),
        (K.PUT_CREDIT, (put(440, SELL), call(466, BUY)), "short put 'covered' by a call: different right"),
        (K.PUT_CREDIT, (put(440, SELL), put(435, BUY, expiry=date(2026, 11, 20))), "cover in another expiry (a calendar)"),
        (K.PUT_CREDIT, (put(440, SELL), put(435, BUY, underlying="QQQ")), "cover on another underlying"),
        (K.PUT_CREDIT, (put(440, SELL, ratio=2), put(435, BUY)), "ratio 2 short against ratio 1 long"),
        (K.PUT_CREDIT, (put(440, SELL, ratio=2), put(435, BUY, ratio=2)), "ratios other than 1 are not v1 structures"),
        (K.PUT_CREDIT, (put(440, SELL), put(435, BUY), put(430, SELL)), "an extra, uncovered short"),
        (K.PUT_CREDIT, LEGS[K.CALL_CREDIT], "legs of another kind"),
        (K.LONG_CALL, (call(450, SELL),), "a 'long' single that is short"),
        (K.LONG_CALL, (put(450, BUY),), "wrong right"),
        (K.LONG_PUT, LEGS[K.PUT_DEBIT], "too many legs"),
        (K.IRON_CONDOR, (put(430, BUY), put(435, SELL), call(465, SELL)), "condor without its long call"),
        (K.IRON_CONDOR, (put(435, BUY), put(430, SELL), call(465, SELL), call(472, BUY)), "condor put wing mis-ordered"),
        (K.IRON_CONDOR, (put(430, BUY), put(435, SELL), call(472, SELL), call(465, BUY)), "condor call wing mis-ordered"),
        (K.IRON_CONDOR, (put(460, BUY), put(470, SELL), call(450, SELL), call(480, BUY)), "inverted: short put above short call"),
        (K.IRON_CONDOR, (), "no legs"),
    ],
)
def test_defined_risk_rejects(kind: StructureKind, legs: tuple[Leg, ...], why: str) -> None:
    assert defined_risk_ok(kind, legs) is False, why


def test_inverted_condor_really_breaks_the_bound_which_is_why_it_is_rejected() -> None:
    # both wings 10.00 wide, but the short put (470) sits ABOVE the short call (450): every leg pair is "covered", yet at
    # S = 460 the put wing has lost its full 1,000 AND the call wing its full 1,000 - both at once
    legs = (put(460, BUY), put(470, SELL), call(450, SELL), call(460, BUY))
    assert defined_risk_ok(K.IRON_CONDOR, legs) is False
    assert expiry_pnl_pc(legs, 0, 46_000) == -200_000
    worst = max(-expiry_pnl_pc(legs, 0, s) for s in price_grid(legs))
    assert worst == 200_000
    assert max_loss_pc(K.IRON_CONDOR, (1000, 1000), 0, 0) == 100_000  # the 9.2 formula would understate the risk by half


def test_iron_butterfly_shorts_on_one_strike_keeps_the_bound() -> None:
    legs = (put(440, BUY), put(450, SELL), call(450, SELL), call(465, BUY))
    assert defined_risk_ok(K.IRON_CONDOR, legs)
    worst = max(-expiry_pnl_pc(legs, -600, s) for s in price_grid(legs))
    assert worst == max_loss_pc(K.IRON_CONDOR, (1000, 1500), -600, 0) == 90_000


def test_defined_risk_unknown_kind_is_false() -> None:
    assert defined_risk_ok("butterfly", LEGS[K.PUT_CREDIT]) is False  # type: ignore[arg-type]


# ======================================================================================================================
# Fees: the 10.7 formula per fill and fee_rt (one open + one close, per contract, rounded up to the cent)
# ======================================================================================================================


def test_default_fee_rates_are_the_ones_these_goldens_assume() -> None:
    assert (FEES.orf, FEES.occ, FEES.taf_sell, FEES.cat, FEES.sec_sell_rate, FEES.commission) == (
        0.015,
        0.025,
        0.00329,
        0.0003,
        0.0000206,
        0.0,
    )


def test_fill_fees_micro_golden() -> None:
    # 3 lots of a 2-leg vertical, one leg sold at 2.50: contracts = 6, sold = 3, sell notional = 250 * 100 * 3 = 75,000 cents = $750
    # per contract: (0.015 + 0.025 + 0.0003 + 0) USD = 40,300 micros -> 6 * 40,300 = 241,800
    # TAF: 3 * 0.00329 USD = 3 * 3,290 = 9,870
    # SEC: 0.0000206 * $750 = $0.01545 = 15,450 micros
    assert fill_fees_micro(6, 3, 75_000, FEES) == 241_800 + 9_870 + 15_450 == 267_120
    # a buy-only fill pays no TAF and no SEC fee
    assert fill_fees_micro(2, 0, 0, FEES) == 80_600
    # the SEC term rounds UP to the next micro-dollar: 0.0000206 * 1 cent * 1e4 = 0.206 micros -> 1
    assert fill_fees_micro(0, 0, 1, FEES) == 1
    assert fill_fees_micro(0, 0, 0, FEES) == 0


def test_fill_fees_micro_validation() -> None:
    with pytest.raises(InvariantError, match="sold"):
        fill_fees_micro(2, 3, 0, FEES)
    with pytest.raises(InvariantError):
        fill_fees_micro(-1, 0, 0, FEES)
    with pytest.raises(InvariantError):
        fill_fees_micro(2, 1, 100.0, FEES)  # type: ignore[arg-type]
    with pytest.raises(InvariantError, match=r"fees\.orf"):
        fill_fees_micro(2, 1, 100, msgspec.structs.replace(FEES, orf=float("nan")))


def test_fee_round_trip_goldens() -> None:
    # single leg at 3.20: contracts 2 * 40,300 = 80,600; the leg is sold once (at the close): TAF 3,290;
    # SEC 0.0000206 * $320 = 6,592 micros; total 90,482 micros = 9.0482 cents -> 10
    assert fee_round_trip(1, 0, [320], FEES) == 10
    # 2-leg vertical, prices 2.50 and 1.25: contracts 4 * 40,300 = 161,200; each leg sold once: TAF 2 * 3,290 = 6,580;
    # SEC 0.0000206 * $375 = 7,725; total 175,505 micros = 17.5505 cents -> 18
    assert fee_round_trip(2, 1, [250, 125], FEES) == 18
    # 4-leg condor, prices 0.60 / 1.50 / 1.40 / 0.50: 8 * 40,300 = 322,400; TAF 4 * 3,290 = 13,160; SEC 0.0000206 * $400 = 8,240;
    # total 343,800 micros = 34.38 cents -> 35
    assert fee_round_trip(4, 2, [60, 150, 140, 50], FEES) == 35


def test_fee_round_trip_is_independent_of_price_order_and_of_which_legs_open_short() -> None:
    # over a round trip EVERY leg is sold exactly once, so the split between the opening and the closing fill cancels
    assert fee_round_trip(2, 1, [125, 250], FEES) == fee_round_trip(2, 1, [250, 125], FEES) == 18
    assert fee_round_trip(2, 0, [250, 125], FEES) == fee_round_trip(2, 2, [250, 125], FEES) == 18
    assert fee_round_trip(4, 2, [50, 140, 150, 60], FEES) == 35


def test_fee_round_trip_equals_open_fill_plus_close_fill_of_the_per_fill_formula() -> None:
    # open a put credit spread (sell 2.50, buy 1.25), close it at the same prices (sell the long at 1.25, buy back the short)
    open_fill = fill_fees_micro(2, 1, 250 * 100, FEES)
    close_fill = fill_fees_micro(2, 1, 125 * 100, FEES)
    assert open_fill == 80_600 + 3_290 + 5_150
    assert close_fill == 80_600 + 3_290 + 2_575
    assert open_fill + close_fill == 175_505
    assert fee_round_trip(2, 1, [250, 125], FEES) == -(-(open_fill + close_fill) // 10_000)


def test_fee_round_trip_rounds_up_to_the_cent_and_keeps_exact_cents() -> None:
    flat = FeesConfig(orf=0.01, occ=0.0, taf_sell=0.0, cat=0.0, sec_sell_rate=0.0, commission=0.0)
    assert fee_round_trip(1, 0, [320], flat) == 2  # 2 * 10,000 micros = exactly 2 cents: NOT rounded to 3
    assert fee_round_trip(3, 1, [10, 20, 30], flat) == 6
    one_micro_more = FeesConfig(orf=0.010001, occ=0.0, taf_sell=0.0, cat=0.0, sec_sell_rate=0.0, commission=0.0)
    assert fee_round_trip(1, 0, [320], one_micro_more) == 3  # 20,002 micros -> 3 cents
    free = FeesConfig(orf=0.0, occ=0.0, taf_sell=0.0, cat=0.0, sec_sell_rate=0.0, commission=0.0)
    assert fee_round_trip(4, 2, [60, 150, 140, 50], free) == 0
    # a commission is charged per contract and side like the other per-contract fees: 4 * 0.65 USD = 260 cents on top of 17.5505
    with_commission = msgspec.structs.replace(FEES, commission=0.65)
    assert fee_round_trip(2, 1, [250, 125], with_commission) == 278


def test_sub_micro_rates_round_up() -> None:
    tiny = FeesConfig(orf=0.0000004, occ=0.0, taf_sell=0.0, cat=0.0, sec_sell_rate=0.0, commission=0.0)
    assert fill_fees_micro(1, 0, 0, tiny) == 1  # 0.4 micro-dollars per contract is charged as 1


@pytest.mark.parametrize(
    ("n_legs", "n_sell", "prices"),
    [
        (0, 0, []),
        (2, 3, [250, 125]),
        (2, -1, [250, 125]),
        (2, 1, [250]),
        (2, 1, [250, 125, 10]),
        (2, 1, [250, -125]),
        (2, 1, [250, 12.5]),
        (2.0, 1, [250, 125]),
    ],
)
def test_fee_round_trip_validation(n_legs: int, n_sell: int, prices: list[int]) -> None:
    with pytest.raises(InvariantError):
        fee_round_trip(n_legs, n_sell, prices, FEES)


def test_the_9_3_worked_examples() -> None:
    # "$6 wide at $1.25 natural credit: max_loss_pc = (600 - 125) * 100 + fee_rt = about 47,520c"
    fee_rt = fee_round_trip(2, 1, [200, 75], FEES)
    # contracts 161,200 + TAF 6,580 + SEC 0.0000206 * $275 = 5,665 -> 173,445 micros -> 18 cents
    assert fee_rt == 18
    assert max_loss_pc(K.PUT_CREDIT, (600, 0), -125, fee_rt) == 47_518
    # budgets at the defaults: $1,000 / $750 / $500 -> 2 / 1 / 1 contracts
    assert [budget // 47_518 for budget in (100_000, 75_000, 50_000)] == [2, 1, 1]


# ======================================================================================================================
# Section 8: the per-leg liquidity filter
# ======================================================================================================================


def quote(bid: int, ask: int, oi_prev: int | None = 500) -> Quote:
    contract = OptionContract(underlying="SPY", expiry=EXPIRY, right=Right.PUT, strike_milli=440_000)
    return Quote(
        contract=contract, bid=bid, ask=ask, bid_size=10, ask_size=10, oi_prev=oi_prev, iv=0.2, delta=-0.25, vega=0.5, quote_ts=None
    )


def test_default_liquidity_limits_are_the_ones_these_cases_assume() -> None:
    assert (LIQ.min_bid_cents_sold, LIQ.min_bid_cents_bought, LIQ.max_rel_spread, LIQ.max_abs_spread_cents) == (10, 1, 0.15, 10)
    assert (LIQ.min_open_interest, LIQ.allow_missing_open_interest) == (100, True)


@pytest.mark.parametrize(
    ("bid", "ask", "oi", "sold", "expected"),
    [
        (250, 260, 500, True, ()),  # a healthy quote
        (250, 260, 500, False, ()),
        # --- liq:bid: >= 10c on legs we sell, >= 1c on legs we buy
        (9, 12, 500, True, ("liq:bid",)),
        (10, 13, 500, True, ()),
        (9, 12, 500, False, ()),
        (0, 5, 500, False, ("liq:bid",)),
        (1, 5, 500, False, ()),
        # --- liq:crossed: ask > bid is required (locked and crossed both fail); a negative spread is not ALSO a wide spread
        (50, 50, 500, False, ("liq:crossed",)),
        (60, 50, 500, False, ("liq:crossed",)),
        (0, 0, 500, True, ("liq:bid", "liq:crossed")),  # no quote at all
        (250, 0, 500, True, ("liq:crossed",)),  # no ask
        # --- liq:spread, relative rule (mid >= 50c): spread / mid <= 0.15
        (185, 215, 500, True, ()),  # 30 / 200 = 0.15 exactly: passes
        (185, 216, 500, True, ("liq:spread",)),  # 31 / 200.5 = 0.1546
        (1000, 1160, 500, True, ()),  # 160 / 1080 = 0.1481
        (1000, 1163, 500, True, ("liq:spread",)),  # 163 / 1081.5 = 0.1507
        # --- the 50c switch: mid = 50c exactly uses the RELATIVE rule, below it the absolute 10c rule
        (45, 55, 500, True, ("liq:spread",)),  # mid 50.0: 10 / 50 = 0.20 > 0.15
        (47, 54, 500, True, ()),  # mid 50.5: 7 / 50.5 = 0.1386
        (45, 54, 500, True, ()),  # mid 49.5: absolute rule, 9 <= 10 (relative would be 0.18 and fail)
        (44, 55, 500, True, ("liq:spread",)),  # mid 49.5: absolute rule, 11 > 10
        (20, 30, 500, True, ()),  # 10c wide at a 25c mid: passes the absolute rule
        (20, 31, 500, True, ("liq:spread",)),
        # --- liq:oi
        (250, 260, 100, True, ()),
        (250, 260, 99, True, ("liq:oi",)),
        (250, 260, 0, True, ("liq:oi",)),
        (250, 260, None, True, ()),  # missing OI passes with allow_missing_open_interest = true (the default)
        # --- several at once, always in the vocab order
        (5, 200, 1, True, ("liq:bid", "liq:spread", "liq:oi")),
        (0, 0, 1, True, ("liq:bid", "liq:crossed", "liq:oi")),
    ],
)
def test_leg_liquidity_rejects(bid: int, ask: int, oi: int | None, sold: bool, expected: tuple[str, ...]) -> None:
    assert leg_liquidity_rejects(quote(bid, ask, oi), sold=sold, cfg=LIQ) == expected


def test_missing_open_interest_fails_when_the_config_says_so() -> None:
    strict = msgspec.structs.replace(LIQ, allow_missing_open_interest=False)
    assert leg_liquidity_rejects(quote(250, 260, None), sold=True, cfg=strict) == ("liq:oi",)
    assert leg_liquidity_rejects(quote(250, 260, 100), sold=True, cfg=strict) == ()


def test_liquidity_limits_come_from_the_config() -> None:
    cfg = LiquidityConfig(
        min_bid_cents_sold=25, min_bid_cents_bought=5, max_rel_spread=0.05, max_abs_spread_cents=3, min_open_interest=1000
    )
    assert leg_liquidity_rejects(quote(24, 26, 1000), sold=True, cfg=cfg) == ("liq:bid",)
    assert leg_liquidity_rejects(quote(4, 6, 1000), sold=False, cfg=cfg) == ("liq:bid",)
    assert leg_liquidity_rejects(quote(195, 205, 1000), sold=True, cfg=cfg) == ()  # 10 / 200 = 0.05 exactly
    assert leg_liquidity_rejects(quote(195, 206, 1000), sold=True, cfg=cfg) == ("liq:spread",)
    assert leg_liquidity_rejects(quote(30, 34, 1000), sold=True, cfg=cfg) == ("liq:spread",)  # 4 > 3 under the 50c switch
    assert leg_liquidity_rejects(quote(250, 255, 999), sold=True, cfg=cfg) == ("liq:oi",)


def test_liquidity_codes_are_the_vocab_codes_in_vocab_order() -> None:
    assert LIQUIDITY_REJECTS == ("liq:bid", "liq:crossed", "liq:spread", "liq:oi")
    everything = leg_liquidity_rejects(quote(0, 0, 0), sold=True, cfg=LIQ)
    assert set(everything) <= set(LIQUIDITY_REJECTS)
    assert list(everything) == sorted(everything, key=LIQUIDITY_REJECTS.index)


def test_liquidity_filter_never_reads_same_day_volume_or_sizes() -> None:
    # 0.1 item 12: Quote has no volume field at all; sizes are fill-time inputs (10.4), not part of THIS filter
    thin = msgspec.structs.replace(quote(250, 260), bid_size=None, ask_size=0)
    assert leg_liquidity_rejects(thin, sold=True, cfg=LIQ) == ()
    assert "volume" not in Quote.__struct_fields__


# ======================================================================================================================
# The module is the single, pure implementation
# ======================================================================================================================


def test_public_api_is_the_3_7_contract() -> None:
    contract = {
        "max_loss_pc",
        "max_profit_pc",
        "breakevens",
        "bp_required_pc",
        "fee_round_trip",
        "defined_risk_ok",
        "leg_liquidity_rejects",
    }
    assert contract <= set(structmath.__all__)
    assert set(structmath.__all__) - contract == {"fill_fees_micro", "MULTIPLIER"}  # the shared 10.7 per-fill arithmetic; 100
    assert structmath.MULTIPLIER == 100


def test_signatures_match_section_3_7() -> None:
    import inspect

    def params(fn: object) -> list[str]:
        return [str(p) for p in inspect.signature(fn).parameters.values()]  # type: ignore[arg-type]

    assert params(max_loss_pc) == ["kind: jevbot.types.StructureKind", "widths: tuple[int, int]", "net: int", "fee_rt: int"]
    assert params(max_profit_pc) == ["kind: jevbot.types.StructureKind", "widths: tuple[int, int]", "net: int"]
    assert [p.split(":")[0] for p in params(breakevens)] == ["kind", "legs", "net"]
    assert [p.split(":")[0] for p in params(bp_required_pc)] == ["kind", "widths", "net", "fee_rt", "cfg"]
    assert [p.split(":")[0] for p in params(fee_round_trip)] == ["n_legs", "n_sell_legs_open", "open_leg_prices", "fees"]
    assert [p.split(":")[0] for p in params(defined_risk_ok)] == ["kind", "legs"]
    liq = inspect.signature(leg_liquidity_rejects).parameters
    assert list(liq) == ["quote", "sold", "cfg"]
    assert liq["sold"].kind is inspect.Parameter.KEYWORD_ONLY and liq["cfg"].kind is inspect.Parameter.KEYWORD_ONLY
