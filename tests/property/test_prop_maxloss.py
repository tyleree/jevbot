"""Max-loss payoff property (DESIGN.md section 8, 9.1 check 5, 9.2).

For random valid structures of **all seven kinds** and every terminal underlying price on a grid, the expiry payoff loss
never exceeds `max_loss_per_contract`, and - fees aside - the grid maximum **equals** it. That is the property
`structmath.defined_risk_ok` relies on, so a structure that passes the leg-pairing rule is bounded by the 9.2 formula.

The payoff is computed here from first principles (intrinsic value per leg, signed by side), never from the formula under
test. Seeded `numpy` generators only - the project takes no hypothesis dependency (DESIGN 1.1).
"""

from collections.abc import Iterator, Sequence
from datetime import date
from typing import Final

import numpy as np
import pytest

from jevbot import structmath
from jevbot.candidates import CandidateGenerator
from jevbot.config import Config
from jevbot.types import Candidate, Cents, Leg, OptionContract, Right, Side, Structure, StructureKind
from tests.fixtures.chain_factory import make_chain, six_hundred_dollar_chain, xnys
from tests.fixtures.fake_view import FakeView
from tests.unit.test_candidates import StubFillModel

SEED: Final = 20260917
DRAWS: Final = 40  # per kind; each draw sweeps a whole terminal-price grid
UNDERLYING: Final = "SPY"
EXPIRY: Final = date(2024, 6, 21)
LAST_SESSION: Final = date(2024, 6, 21)
DOLLAR: Final = 1_000  # strike_milli of $1.00
ALL_KINDS: Final[tuple[StructureKind, ...]] = tuple(StructureKind)


# ======================================================================================================================
# an independent expiry payoff
# ======================================================================================================================


def intrinsic(leg: Leg, terminal_c: int) -> int:
    """Intrinsic value of one leg at expiry, cents per share (strikes are whole cents in every generated structure)."""
    strike_c, remainder = divmod(leg.contract.strike_milli, 10)
    assert remainder == 0, "the generated structures use whole-cent strikes"
    if leg.contract.right is Right.CALL:
        return max(terminal_c - strike_c, 0)
    return max(strike_c - terminal_c, 0)


def pnl_cents(structure: Structure, net: int, terminal_c: int) -> int:
    """Per-contract P&L at expiry in cents, fees excluded: `100 * (value of the legs - the signed net paid)`."""
    value = sum((1 if leg.side is Side.BUY else -1) * intrinsic(leg, terminal_c) for leg in structure.legs)
    return 100 * (value - net)


def price_grid(structure: Structure) -> list[int]:
    """Every strike, both neighbours of every strike, 0 and a price far above the top strike (cents)."""
    strikes = sorted(leg.contract.strike_milli // 10 for leg in structure.legs)
    grid = {0, strikes[0] // 2, strikes[-1] * 2}
    for strike in strikes:
        grid.update({strike - 100, strike - 1, strike, strike + 1, strike + 100})
    span = strikes[-1] - strikes[0]
    step = max(span // 16, 1)
    grid.update(range(max(strikes[0] - span - 100, 0), strikes[-1] + span + 101, step))
    return sorted(x for x in grid if x >= 0)


# ======================================================================================================================
# random structure generation
# ======================================================================================================================


def leg(right: Right, side: Side, dollars: int) -> Leg:
    return Leg(contract=OptionContract(underlying=UNDERLYING, expiry=EXPIRY, right=right, strike_milli=dollars * DOLLAR), side=side)


def structure_of(kind: StructureKind, legs: Sequence[Leg]) -> Structure:
    ordered = tuple(sorted(legs, key=lambda item: (item.contract.right is Right.CALL, item.contract.strike_milli)))
    return Structure(kind=kind, underlying=UNDERLYING, expiry=EXPIRY, last_session=LAST_SESSION, legs=ordered)


def draw(kind: StructureKind, rng: np.random.Generator) -> tuple[Structure, int]:
    """A random defined-risk structure of `kind` and a plausible signed net (cents/share, + debit / - credit)."""
    anchor = int(rng.integers(60, 700))
    width = int(rng.integers(1, 26))
    if kind is StructureKind.LONG_CALL:
        structure = structure_of(kind, [leg(Right.CALL, Side.BUY, anchor)])
    elif kind is StructureKind.LONG_PUT:
        structure = structure_of(kind, [leg(Right.PUT, Side.BUY, anchor)])
    elif kind is StructureKind.CALL_DEBIT:
        structure = structure_of(kind, [leg(Right.CALL, Side.BUY, anchor), leg(Right.CALL, Side.SELL, anchor + width)])
    elif kind is StructureKind.PUT_DEBIT:
        structure = structure_of(kind, [leg(Right.PUT, Side.BUY, anchor), leg(Right.PUT, Side.SELL, anchor - width)])
    elif kind is StructureKind.CALL_CREDIT:
        structure = structure_of(kind, [leg(Right.CALL, Side.SELL, anchor), leg(Right.CALL, Side.BUY, anchor + width)])
    elif kind is StructureKind.PUT_CREDIT:
        structure = structure_of(kind, [leg(Right.PUT, Side.SELL, anchor), leg(Right.PUT, Side.BUY, anchor - width)])
    else:
        gap = int(rng.integers(2, 40))
        call_width = int(rng.integers(1, 26))
        structure = structure_of(
            kind,
            [
                leg(Right.PUT, Side.SELL, anchor),
                leg(Right.PUT, Side.BUY, anchor - width),
                leg(Right.CALL, Side.SELL, anchor + gap),
                leg(Right.CALL, Side.BUY, anchor + gap + call_width),
            ],
        )
    # a credit below EVERY wing and a debit below the width: the economically valid range of section 8's sanity bounds
    # (a condor credit above its narrower wing would be a free lunch, and its 9.2 breakeven would lie past the long wing)
    bound = min(w for w in structure.wing_widths if w > 0) if any(structure.wing_widths) else 0
    if kind in (StructureKind.LONG_CALL, StructureKind.LONG_PUT):
        net = int(rng.integers(1, 3_000))
    elif kind in (StructureKind.CALL_DEBIT, StructureKind.PUT_DEBIT):
        net = int(rng.integers(1, max(bound, 2)))
    else:
        net = -int(rng.integers(1, max(bound, 2)))
    return structure, net


def cases() -> Iterator[tuple[StructureKind, Structure, int]]:
    rng = np.random.default_rng(SEED)
    for kind in ALL_KINDS:
        for _ in range(DRAWS):
            structure, net = draw(kind, rng)
            yield kind, structure, net


ALL_CASES: Final[list[tuple[StructureKind, Structure, int]]] = list(cases())


# ======================================================================================================================
# the properties
# ======================================================================================================================


def test_the_generated_structures_are_defined_risk() -> None:
    assert len(ALL_CASES) == len(ALL_KINDS) * DRAWS
    for kind, structure, _net in ALL_CASES:
        assert structmath.defined_risk_ok(kind, structure.legs), structure
        assert structure.kind is kind


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_the_payoff_grid_maximum_loss_equals_max_loss_pc(kind: StructureKind) -> None:
    """9.2 / 9.1 check 5: `max_loss_pc` is exactly `-(min P&L at expiry)`, for every one of the seven kinds."""
    for case_kind, structure, net in ALL_CASES:
        if case_kind is not kind:
            continue
        widths = structure.wing_widths
        bound = structmath.max_loss_pc(kind, widths, net, 0)
        losses = [-pnl_cents(structure, net, price) for price in price_grid(structure)]
        assert max(losses) == bound, (structure, net, max(losses), bound)
        assert bound > 0


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_the_loss_never_exceeds_max_loss_with_fees(kind: StructureKind) -> None:
    fees = Config().fees
    for case_kind, structure, net in ALL_CASES:
        if case_kind is not kind:
            continue
        n_legs = len(structure.legs)
        n_sell = sum(1 for item in structure.legs if item.side is Side.SELL)
        fee_rt = structmath.fee_round_trip(n_legs, n_sell, [max(abs(net), 1)] * n_legs, fees)
        bound = structmath.max_loss_pc(kind, structure.wing_widths, net, fee_rt)
        assert fee_rt > 0 and bound == structmath.max_loss_pc(kind, structure.wing_widths, net, 0) + fee_rt
        for price in price_grid(structure):
            assert -pnl_cents(structure, net, price) <= bound


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_max_profit_and_breakevens_agree_with_the_payoff(kind: StructureKind) -> None:
    """The 9.2 profit column and the breakeven levels, read off the same independent payoff."""
    for case_kind, structure, net in ALL_CASES:
        if case_kind is not kind:
            continue
        grid = price_grid(structure)
        best = max(pnl_cents(structure, net, price) for price in grid)
        bound = structmath.max_profit_pc(kind, structure.wing_widths, net)
        if bound is None:
            assert kind in (StructureKind.LONG_CALL, StructureKind.LONG_PUT)
        else:
            assert best == bound, (structure, net, best, bound)
        for level in structmath.breakevens(kind, structure.legs, net):
            assert pnl_cents(structure, net, level) == 0, (structure, net, level)


# ======================================================================================================================
# the same property on real candidates
# ======================================================================================================================


def _candidates() -> Iterator[tuple[str, StructureKind, Structure, int, Cents]]:
    cfg = Config()
    generator = CandidateGenerator(cfg, StubFillModel(cfg))
    for label, chain in (("450", make_chain()), ("600", six_hundred_dollar_chain())):
        view = FakeView(key=chain.key, as_of=chain.ts, calendar=xnys(), chains=[chain])
        for kind in ALL_KINDS:
            result = generator.build(kind, view, "SPY", budget_floor=50_000)
            assert isinstance(result, Candidate), (label, kind, result)
            yield label, kind, result.structure, result.net.worst, result.max_loss_per_contract


BUILT: Final[list[tuple[str, StructureKind, Structure, int, Cents]]] = list(_candidates())


def test_every_built_candidate_is_bounded_by_its_own_max_loss() -> None:
    assert len(BUILT) == 2 * len(ALL_KINDS)
    for label, kind, structure, net_worst, max_loss in BUILT:
        assert structmath.defined_risk_ok(kind, structure.legs)
        losses = [-pnl_cents(structure, net_worst, price) for price in price_grid(structure)]
        # the candidate's bound carries `fee_rt` on top of the pure payoff maximum, so it is never below it
        assert max(losses) <= max_loss, (label, kind)
        assert max_loss - max(losses) < 100, (label, kind)  # the fee term is cents, not dollars
        assert max_loss > 0
