"""Property tests for src/jevbot/structmath.py (WP00). Seeded numpy generators - no hypothesis dependency (DESIGN 1.1).

The references in this file are written independently of the code under test: the expiry payoff is summed leg by leg, the 10.7
fee formula and the section 8 spread rule are evaluated with `decimal.Decimal` exactly as the prose states them (division and
all), whereas structmath works with cross-multiplied integers and `Fraction`.
"""

import hashlib
import math
from datetime import date
from decimal import ROUND_CEILING, Decimal, getcontext

import msgspec
import numpy as np
import pytest

from jevbot.config import FeesConfig, LiquidityConfig, RiskConfig
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

getcontext().prec = 60

EXPIRY = date(2026, 10, 16)
K = StructureKind
N_STRUCTURES = 400  # per kind
RISK = RiskConfig()
FEES = FeesConfig()
LIQ = LiquidityConfig()


def rng_for(tag: str) -> np.random.Generator:
    # a stable seed per test (Python's hash() is salted per process); draws never depend on test order (10.9)
    return np.random.Generator(np.random.PCG64(int(hashlib.sha256(tag.encode()).hexdigest()[:16], 16)))


def make_leg(right: Right, strike_cents: int, side: Side) -> Leg:
    contract = OptionContract(underlying="SPY", expiry=EXPIRY, right=right, strike_milli=strike_cents * 10)
    return Leg(contract=contract, side=side)


def strikes(rng: np.random.Generator, n: int) -> list[int]:
    """n distinct ascending strikes in cents on a 0.50 grid between 50.00 and 800.00."""
    picks = rng.choice(np.arange(100, 1601), size=n, replace=False)
    return sorted(int(p) * 50 for p in picks)


def random_structure(rng: np.random.Generator, kind: StructureKind) -> tuple[tuple[Leg, ...], tuple[int, int], int]:
    """(legs, (put wing, call wing) in cents, signed net) with VALID economics: 0 < debit < width, 0 < credit < narrower wing."""
    if kind is K.LONG_CALL:
        (k1,) = strikes(rng, 1)
        return (make_leg(Right.CALL, k1, Side.BUY),), (0, 0), int(rng.integers(1, 5_000))
    if kind is K.LONG_PUT:
        (k1,) = strikes(rng, 1)
        return (make_leg(Right.PUT, k1, Side.BUY),), (0, 0), int(rng.integers(1, min(5_000, k1)))
    if kind in (K.CALL_DEBIT, K.PUT_DEBIT, K.CALL_CREDIT, K.PUT_CREDIT):
        lo, hi = strikes(rng, 2)
        width = hi - lo
        amount = int(rng.integers(1, width))  # 1 .. width - 1
        if kind is K.CALL_DEBIT:
            return (make_leg(Right.CALL, lo, Side.BUY), make_leg(Right.CALL, hi, Side.SELL)), (0, width), amount
        if kind is K.PUT_DEBIT:
            return (make_leg(Right.PUT, lo, Side.SELL), make_leg(Right.PUT, hi, Side.BUY)), (width, 0), amount
        if kind is K.CALL_CREDIT:
            return (make_leg(Right.CALL, lo, Side.SELL), make_leg(Right.CALL, hi, Side.BUY)), (0, width), -amount
        return (make_leg(Right.PUT, lo, Side.BUY), make_leg(Right.PUT, hi, Side.SELL)), (width, 0), -amount
    k_lp, k_sp, k_sc, k_lc = strikes(rng, 4)
    if rng.random() < 0.1:
        k_sc = k_sp  # an iron butterfly now and then: the shorts share a strike
    w_put, w_call = k_sp - k_lp, k_lc - k_sc
    credit = int(rng.integers(1, min(w_put, w_call)))
    legs = (
        make_leg(Right.PUT, k_lp, Side.BUY),
        make_leg(Right.PUT, k_sp, Side.SELL),
        make_leg(Right.CALL, k_sc, Side.SELL),
        make_leg(Right.CALL, k_lc, Side.BUY),
    )
    return legs, (w_put, w_call), -credit


def expiry_pnl_pc(legs: tuple[Leg, ...], net: int, s_cents: int) -> int:
    value = 0
    for one in legs:
        strike = one.contract.strike_milli // 10
        intrinsic = max(s_cents - strike, 0) if one.contract.right is Right.CALL else max(strike - s_cents, 0)
        value += intrinsic if one.side is Side.BUY else -intrinsic
    return (value - net) * 100


def terminal_prices(rng: np.random.Generator, legs: tuple[Leg, ...]) -> list[int]:
    ks = sorted(one.contract.strike_milli // 10 for one in legs)
    points = {0, 1, 2 * ks[-1], 5 * ks[-1]}
    for k in ks:
        points.update((k - 1, k, k + 1))
    points.update(int(x) for x in rng.integers(0, 2 * ks[-1], size=150))
    return sorted(points)


# ======================================================================================================================
# 9.2: the payoff never loses more than max_loss_pc, and (fees aside) the maximum EQUALS it - all seven kinds
# ======================================================================================================================


@pytest.mark.parametrize("kind", list(K))
def test_expiry_loss_never_exceeds_max_loss_and_the_grid_maximum_equals_it(kind: StructureKind) -> None:
    rng = rng_for(f"maxloss:{kind.value}")
    for _ in range(N_STRUCTURES):
        legs, widths, net = random_structure(rng, kind)
        fee_rt = int(rng.integers(0, 60))
        bound = max_loss_pc(kind, widths, net, fee_rt)
        losses = [-expiry_pnl_pc(legs, net, s) for s in terminal_prices(rng, legs)]
        assert max(losses) + fee_rt == bound
        assert all(loss + fee_rt <= bound for loss in losses)
        assert bound > 0
        assert defined_risk_ok(kind, legs)


@pytest.mark.parametrize("kind", list(K))
def test_widths_passed_by_callers_are_structure_wing_widths(kind: StructureKind) -> None:
    rng = rng_for(f"widths:{kind.value}")
    for _ in range(50):
        legs, widths, _ = random_structure(rng, kind)
        ordered = tuple(sorted(legs, key=lambda one: (one.contract.right is Right.CALL, one.contract.strike_milli)))
        structure = Structure(kind=kind, underlying="SPY", expiry=EXPIRY, last_session=EXPIRY, legs=ordered)
        assert structure.wing_widths == widths


@pytest.mark.parametrize("kind", list(K))
def test_max_profit_is_the_payoff_maximum_for_bounded_kinds(kind: StructureKind) -> None:
    rng = rng_for(f"maxprofit:{kind.value}")
    for _ in range(N_STRUCTURES):
        legs, widths, net = random_structure(rng, kind)
        best = max(expiry_pnl_pc(legs, net, s) for s in terminal_prices(rng, legs))
        profit = max_profit_pc(kind, widths, net)
        if kind is K.LONG_CALL:
            assert profit is None
            top = 5 * legs[0].contract.strike_milli // 10
            assert best == expiry_pnl_pc(legs, net, top)  # still rising at the top of the grid: unbounded
        elif kind is K.LONG_PUT:
            assert profit is None  # REPORTED as None; internally (K - n) * 100, reached at S = 0
            assert best == (legs[0].contract.strike_milli // 10 - net) * 100
        else:
            assert profit == best
            assert profit is not None and profit > 0


@pytest.mark.parametrize("kind", list(K))
def test_pnl_is_zero_at_every_breakeven_and_changes_sign_across_it(kind: StructureKind) -> None:
    rng = rng_for(f"breakeven:{kind.value}")
    for _ in range(N_STRUCTURES):
        legs, _, net = random_structure(rng, kind)
        levels = breakevens(kind, legs, net)
        assert len(levels) == (2 if kind is K.IRON_CONDOR else 1)
        assert list(levels) == sorted(levels)
        for level in levels:
            assert expiry_pnl_pc(legs, net, level) == 0
            assert expiry_pnl_pc(legs, net, level - 1) * expiry_pnl_pc(legs, net, level + 1) < 0
        shuffled = tuple(legs[i] for i in rng.permutation(len(legs)))
        assert breakevens(kind, shuffled, net) == levels


@pytest.mark.parametrize("kind", [K.CALL_DEBIT, K.PUT_DEBIT, K.CALL_CREDIT, K.PUT_CREDIT])
def test_swapping_the_sides_of_a_vertical_is_never_defined_risk_under_the_same_kind(kind: StructureKind) -> None:
    rng = rng_for(f"swap:{kind.value}")
    for _ in range(N_STRUCTURES):
        legs, _, _ = random_structure(rng, kind)
        flipped = tuple(Leg(contract=one.contract, side=Side.SELL if one.side is Side.BUY else Side.BUY) for one in legs)
        assert defined_risk_ok(kind, legs)
        assert not defined_risk_ok(kind, flipped)  # the mis-ordered version: long leg on the wrong strike
        assert not defined_risk_ok(kind, legs[:1])
        assert not defined_risk_ok(kind, tuple(one for one in legs if one.side is Side.SELL))  # the uncovered short alone


def test_defined_risk_is_exactly_the_payoff_bound_for_random_four_leg_condors() -> None:
    """For ANY strike order of a put pair + call pair under kind iron_condor: defined_risk_ok implies the 9.2 bound holds
    on the payoff grid, and every arrangement where the bound FAILS is rejected."""
    rng = rng_for("condor-any-order")
    accepted = violated = 0
    for _ in range(3_000):
        ks = [int(k) * 50 for k in rng.integers(100, 140, size=4)]  # narrow range: plenty of overlaps and inversions
        legs = (
            make_leg(Right.PUT, ks[0], Side.BUY),
            make_leg(Right.PUT, ks[1], Side.SELL),
            make_leg(Right.CALL, ks[2], Side.SELL),
            make_leg(Right.CALL, ks[3], Side.BUY),
        )
        widths = (abs(ks[1] - ks[0]), abs(ks[3] - ks[2]))
        worst = max(-expiry_pnl_pc(legs, 0, s) for s in terminal_prices(rng, legs))
        ok = defined_risk_ok(K.IRON_CONDOR, legs)
        if ok:
            accepted += 1
            assert worst == max_loss_pc(K.IRON_CONDOR, widths, 0, 0)
        if worst > max_loss_pc(K.IRON_CONDOR, widths, 0, 0):
            violated += 1
            assert not ok
    assert accepted > 50 and violated > 50  # both branches were really exercised


# ======================================================================================================================
# Buying power
# ======================================================================================================================


@pytest.mark.parametrize("kind", list(K))
def test_buying_power_relations(kind: StructureKind) -> None:
    rng = rng_for(f"bp:{kind.value}")
    max_wing = msgspec.structs.replace(RISK, condor_bp_mode="max_wing")
    for _ in range(N_STRUCTURES):
        _, widths, net = random_structure(rng, kind)
        loss_no_fee = max_loss_pc(kind, widths, net, 0)
        bp = bp_required_pc(kind, widths, net, int(rng.integers(0, 60)), RISK)
        assert bp >= 0 and type(bp) is int
        if kind is K.IRON_CONDOR:
            # max_wing reserves exactly the max loss; sum_wings adds the narrower wing on top
            assert bp_required_pc(kind, widths, net, 0, max_wing) == loss_no_fee
            assert bp == loss_no_fee + min(widths) * 100
        else:
            assert bp == loss_no_fee  # at the 1.00 haircut the requirement IS the fee-free max loss (Cboe minimums)
        # the haircut never lowers the requirement, is exact-decimal and rounds up
        mult = round(1 + float(rng.integers(0, 2_000)) / 1_000, 3)
        hair = bp_required_pc(kind, widths, net, 0, msgspec.structs.replace(RISK, bp_haircut_mult=mult))
        if kind in (K.LONG_CALL, K.LONG_PUT, K.CALL_DEBIT, K.PUT_DEBIT):
            assert hair == bp  # debit structures pay the net debit
        else:
            exact = (Decimal(bp) * Decimal(str(mult))).to_integral_value(rounding=ROUND_CEILING)
            assert hair == int(exact) >= bp


# ======================================================================================================================
# Fees
# ======================================================================================================================


def reference_fill_fees_micro(contracts: int, sold: int, sell_notional_cents: int, fees: FeesConfig) -> int:
    """10.7, literally, in Decimal: contracts * (orf + occ + cat + commission) * 1e6 + sold * taf_sell * 1e6
    + ceil(sec_sell_rate * sell_notional_cents * 1e4)."""

    def d(x: float) -> Decimal:
        return Decimal(repr(x))

    per_contract = (d(fees.orf) + d(fees.occ) + d(fees.cat) + d(fees.commission)) * contracts * Decimal(10**6)
    taf = d(fees.taf_sell) * sold * Decimal(10**6)
    sec = (d(fees.sec_sell_rate) * sell_notional_cents * Decimal(10**4)).to_integral_value(rounding=ROUND_CEILING)
    return int((per_contract + taf).to_integral_value(rounding=ROUND_CEILING) + sec)


def test_fill_fees_micro_matches_the_decimal_reference() -> None:
    rng = rng_for("fill-fees")
    for _ in range(3_000):
        n_legs = int(rng.integers(1, 5))
        qty = int(rng.integers(1, 11))
        n_sell = int(rng.integers(0, n_legs + 1))
        notional = int(rng.integers(0, 3_000, size=n_sell).sum()) * 100 * qty
        assert fill_fees_micro(qty * n_legs, qty * n_sell, notional, FEES) == reference_fill_fees_micro(
            qty * n_legs, qty * n_sell, notional, FEES
        )


def test_fee_round_trip_bounds_the_two_fills_from_above_within_one_cent() -> None:
    rng = rng_for("fee-rt")
    equal = 0
    for _ in range(3_000):
        n_legs = int(rng.integers(1, 5))
        prices = [int(p) for p in rng.integers(0, 3_000, size=n_legs)]
        sell_open = [bool(b) for b in rng.integers(0, 2, size=n_legs)]
        open_fill = reference_fill_fees_micro(
            n_legs, sum(sell_open), sum(p for p, s in zip(prices, sell_open, strict=True) if s) * 100, FEES
        )
        close_fill = reference_fill_fees_micro(
            n_legs, n_legs - sum(sell_open), sum(p for p, s in zip(prices, sell_open, strict=True) if not s) * 100, FEES
        )
        two_fills_cents = -(-(open_fill + close_fill) // 10_000)
        fee_rt = fee_round_trip(n_legs, sum(sell_open), prices, FEES)
        assert two_fills_cents <= fee_rt <= two_fills_cents + 1  # never understated; the per-leg rounding costs < n_legs micros
        equal += fee_rt == two_fills_cents
        # order of the prices and the open-side count do not matter
        assert fee_round_trip(n_legs, 0, list(reversed(prices)), FEES) == fee_rt
        # a dearer leg never lowers the estimate
        bumped = [prices[0] + int(rng.integers(1, 500)), *prices[1:]]
        assert fee_round_trip(n_legs, sum(sell_open), bumped, FEES) >= fee_rt
    assert equal > 2_900  # the one-cent slack is a boundary effect, not the norm


def test_fee_round_trip_grows_with_the_number_of_legs() -> None:
    fees = [fee_round_trip(n, 0, [100] * n, FEES) for n in (1, 2, 3, 4)]
    assert fees == sorted(fees) and len(set(fees)) == 4


# ======================================================================================================================
# Section 8 liquidity filter against the prose rule in Decimal
# ======================================================================================================================


def reference_liquidity(bid: int, ask: int, oi: int | None, *, sold: bool, cfg: LiquidityConfig) -> tuple[str, ...]:
    out = []
    if bid < (cfg.min_bid_cents_sold if sold else cfg.min_bid_cents_bought):
        out.append("liq:bid")
    if not ask > bid:
        out.append("liq:crossed")
    mid = Decimal(bid + ask) / 2
    if mid >= 50:
        wide = Decimal(ask - bid) / mid > Decimal(repr(cfg.max_rel_spread))
    else:
        wide = ask - bid > cfg.max_abs_spread_cents
    if wide:
        out.append("liq:spread")
    if (oi is None and not cfg.allow_missing_open_interest) or (oi is not None and oi < cfg.min_open_interest):
        out.append("liq:oi")
    return tuple(out)


def make_quote(bid: int, ask: int, oi: int | None) -> Quote:
    contract = OptionContract(underlying="SPY", expiry=EXPIRY, right=Right.CALL, strike_milli=450_000)
    return Quote(
        contract=contract, bid=bid, ask=ask, bid_size=None, ask_size=None, oi_prev=oi, iv=None, delta=None, vega=None, quote_ts=None
    )


@pytest.mark.parametrize(
    "cfg",
    [
        LIQ,
        msgspec.structs.replace(LIQ, allow_missing_open_interest=False),
        LiquidityConfig(min_bid_cents_sold=25, min_bid_cents_bought=5, max_rel_spread=0.1, max_abs_spread_cents=4, min_open_interest=10),
        LiquidityConfig(max_rel_spread=0.3333, max_abs_spread_cents=1),
    ],
    ids=["default", "oi-required", "tight", "odd"],
)
def test_liquidity_filter_matches_the_prose_rule(cfg: LiquidityConfig) -> None:
    rng = rng_for(f"liq:{cfg.max_rel_spread}:{cfg.allow_missing_open_interest}")
    seen: set[str] = set()
    for _ in range(6_000):
        bid = int(rng.integers(0, 130)) if rng.random() < 0.7 else int(rng.integers(0, 3_000))
        ask = max(0, bid + int(rng.integers(-5, 40)))
        oi = None if rng.random() < 0.2 else int(rng.integers(0, 300))
        for sold in (True, False):
            got = leg_liquidity_rejects(make_quote(bid, ask, oi), sold=sold, cfg=cfg)
            assert got == reference_liquidity(bid, ask, oi, sold=sold, cfg=cfg), (bid, ask, oi, sold)
            assert list(got) == sorted(got, key=LIQUIDITY_REJECTS.index)  # vocab order
            seen.update(got)
    assert seen == set(LIQUIDITY_REJECTS)  # every code was produced


def test_every_spread_exactly_on_the_relative_limit_passes() -> None:
    # spread / mid == 0.15 exactly  <=>  spread = 3t, bid + ask = 40t: bid = 18.5t, ask = 21.5t -> t even
    for t in range(6, 400, 2):
        bid, ask = 37 * t // 2, 43 * t // 2
        assert math.isclose((ask - bid) / ((ask + bid) / 2), 0.15)
        assert leg_liquidity_rejects(make_quote(bid, ask, 500), sold=True, cfg=LIQ) == ()
        assert leg_liquidity_rejects(make_quote(bid, ask + 1, 500), sold=True, cfg=LIQ) == ("liq:spread",)


def test_a_leg_we_sell_is_never_easier_to_pass_than_a_leg_we_buy() -> None:
    rng = rng_for("sold-vs-bought")
    for _ in range(3_000):
        bid = int(rng.integers(0, 60))
        ask = bid + int(rng.integers(-3, 30))
        q = make_quote(bid, max(ask, 0), int(rng.integers(0, 300)))
        assert set(leg_liquidity_rejects(q, sold=False, cfg=LIQ)) <= set(leg_liquidity_rejects(q, sold=True, cfg=LIQ))
