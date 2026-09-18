"""CandidateGenerator and `scan()` (DESIGN.md section 8, test plan 15.1 row `candidates`).

Everything runs on the deterministic chain factory. `StubFillModel` below is this package's own `FillModel` double: the
10.3 band table written out in plain integer arithmetic, so the candidate numbers are checked against an implementation
that is not the one under test, and the suite never depends on WP06's `fills.py`.

The golden strikes and nets of the first tests are hand-computed from the raw factory quotes, which the test prints in its
own comments; the budget-fit tests then prove the three properties the acceptance column names: every kind on a
$600-priced chain ends at `max_loss_per_contract <= budget_floor` or `exceeds_risk_budget`, the fitted long strike is the
furthest-OTM one that fits, and the short leg never moves.
"""

from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any, Final

import pandas as pd
import pytest

from jevbot import structmath, vocab
from jevbot.candidates import (
    SCAN_TIERS,
    CandidateGenerator,
    budget_floor_at,
    natural_limit,
    scan,
    scan_columns,
    structure_tick,
)
from jevbot.config import CadenceConfig, CandidatesConfig, Config, DteConfig, RiskConfig, UniverseConfig
from jevbot.errors import InvariantError
from jevbot.money import cdiv
from jevbot.types import (
    Band,
    BandPrices,
    Candidate,
    CandidateReject,
    Cents,
    ChainSnapshot,
    Fidelity,
    FillRule,
    LegFill,
    Micros,
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
from tests.fixtures.chain_factory import (
    contract_at,
    crossed_quote,
    make_chain,
    nearest_delta,
    quote_of,
    six_hundred_dollar_chain,
    strikes,
    target_expiry,
    thin_oi,
    widened_spread,
    xnys,
    zero_bid,
)
from tests.fixtures.fake_view import FakeView, ex_dividend_event

ALL_KINDS: Final[tuple[StructureKind, ...]] = tuple(StructureKind)
CREDIT_KINDS: Final[tuple[StructureKind, ...]] = (StructureKind.PUT_CREDIT, StructureKind.CALL_CREDIT, StructureKind.IRON_CONDOR)
SINGLES: Final[tuple[StructureKind, ...]] = (StructureKind.LONG_CALL, StructureKind.LONG_PUT)
DEFAULT_FLOOR: Final[Cents] = 50_000  # floor(0.01 * $100k * 0.5) = $500, the 9.3 worked example


# ======================================================================================================================
# the local FillModel double (DESIGN 10.3 / 10.4 / 10.5), independent of src/jevbot/fills.py
# ======================================================================================================================


class StubFillModel:
    """`protocols.FillModel` over the 10.3 band table: BUY `bid + cdiv((ask-bid)*p, 10000)` / `ask` / `cdiv(bid+ask, 2)`,
    SELL `ask - cdiv((ask-bid)*p, 10000)` / `bid` / `(bid+ask)//2`, with `p_bp = [7500, 6600, 5600, 5300]` by leg count.
    A zero-bid `sell_to_close` leg prices at 0 on every band (10.4). Integer arithmetic only, rounding against us."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._p_bp = tuple(round(p * 10_000) for p in cfg.fills.orats_p)
        self._headline = Band.ORATS if cfg.cadence.fill_rule is FillRule.NEXT_SNAPSHOT else Band.WORST

    def _p(self, n_legs: int) -> int:
        return self._p_bp[min(n_legs, len(self._p_bp)) - 1]

    def _leg(self, leg: OrderLeg, quote: Quote, n_legs: int) -> LegFill:
        bid, ask = quote.bid, quote.ask
        step = cdiv((ask - bid) * self._p(n_legs), 10_000) if ask > bid else 0
        if leg.side is Side.BUY:
            orats, worst, mid = bid + step, ask, cdiv(bid + ask, 2)
        elif bid <= 0 and leg.position_intent is PositionIntent.STC:
            orats = worst = mid = 0  # a zero-bid wing sold to close is sold at 0 on all three bands (10.4)
        else:
            orats, worst, mid = ask - step, bid, (bid + ask) // 2
        return LegFill(occ=leg.contract.occ, side=leg.side, bid=bid, ask=ask, orats=orats, worst=worst, mid=mid)

    def check(self, legs: Sequence[OrderLeg], qty: int, chain: ChainSnapshot, *, mandatory: bool) -> tuple[str, ...]:
        if mandatory:
            return ()
        codes: list[str] = []
        for leg in legs:
            quote = chain.quote(leg.contract)
            if quote is None:
                codes.append("missing_contract")
                continue
            selling_to_close = leg.side is Side.SELL and leg.position_intent is PositionIntent.STC
            if quote.ask <= 0 or (leg.side is Side.SELL and quote.bid <= 0 and not selling_to_close):
                codes.append("no_quote")
                continue
            if quote.ask <= quote.bid and quote.bid > 0:
                codes.append("crossed_or_locked")
                continue
            if quote.bid > 0:
                spread = quote.ask - quote.bid
                limit = max(self._cfg.fills.max_rel_spread * (quote.mid2 / 2.0), float(self._cfg.fills.max_abs_spread_cents))
                if spread > limit:
                    codes.append("wide_spread")
        return tuple(sorted(set(codes), key=vocab.FILL_REJECTS.index))

    def price(self, legs: Sequence[OrderLeg], chain: ChainSnapshot, *, mandatory: bool) -> tuple[BandPrices, tuple[LegFill, ...], str]:
        fills: list[LegFill] = []
        for leg in legs:
            quote = chain.quote(leg.contract)
            if quote is None:
                raise AssertionError(f"StubFillModel: {leg.contract.occ} is not in the snapshot")
            fills.append(self._leg(leg, quote, len(legs)))
        sign = {Side.BUY: 1, Side.SELL: -1}
        net = BandPrices(
            orats=sum(sign[f.side] * f.orats for f in fills),
            worst=sum(sign[f.side] * f.worst for f in fills),
            mid=sum(sign[f.side] * f.mid for f in fills),
        )
        return net, tuple(fills), "ok"

    def liquidation(self, structure: Structure, chain: ChainSnapshot, last: tuple[int, int] | None) -> tuple[int, int, bool]:
        liq = mid = 0
        for leg in structure.legs:
            quote = chain.quote(leg.contract)
            if quote is None or (leg.side is Side.SELL and quote.ask <= 0) or (quote.bid > 0 and quote.ask <= quote.bid):
                return (*(last if last is not None else (0, 0)), True)
            if leg.side is Side.SELL:
                liq += quote.ask
                mid += cdiv(quote.mid2, 2)
            else:
                liq -= quote.bid
                mid -= quote.mid2 // 2
        return liq, mid, False

    def fees_micro(self, legs: Sequence[OrderLeg], qty: int, leg_fills: Sequence[LegFill]) -> Micros:
        sells = [f for f in leg_fills if f.side is Side.SELL]
        notional = sum(getattr(f, self._headline.value) * 100 * qty for f in sells)
        return structmath.fill_fees_micro(qty * len(legs), qty * len(sells), notional, self._cfg.fees)


# ======================================================================================================================
# helpers
# ======================================================================================================================


def cfg_with(**sections: Any) -> Config:
    return Config(**sections)


def generator(cfg: Config | None = None) -> CandidateGenerator:
    config = cfg if cfg is not None else Config()
    return CandidateGenerator(config, StubFillModel(config))


def view_of(chain: ChainSnapshot, *, events: Sequence[ScheduledEvent] = ()) -> FakeView:
    return FakeView(key=chain.key, as_of=chain.ts, calendar=xnys(), chains=[chain], events=events)


def built(result: Candidate | CandidateReject) -> Candidate:
    assert isinstance(result, Candidate), f"expected a Candidate, got {result}"
    return result


def open_legs_of(structure: Structure) -> tuple[OrderLeg, ...]:
    return tuple(
        OrderLeg(
            contract=leg.contract,
            side=leg.side,
            position_intent=PositionIntent.BTO if leg.side is Side.BUY else PositionIntent.STO,
            ratio=leg.ratio,
        )
        for leg in structure.legs
    )


def max_loss_of(cfg: Config, fill: StubFillModel, structure: Structure, chain: ChainSnapshot) -> Cents:
    """`max_loss_pc` of an arbitrary structure at the worst band, fees included - the 9.2 arithmetic, via `structmath`."""
    net, fills, _ = fill.price(open_legs_of(structure), chain, mandatory=False)
    prices = [f.orats for f in fills]
    n_sell = sum(1 for leg in structure.legs if leg.side is Side.SELL)
    fee_rt = structmath.fee_round_trip(len(structure.legs), n_sell, prices, cfg.fees)
    return structmath.max_loss_pc(structure.kind, structure.wing_widths, net.worst, fee_rt)


def with_long_strike(structure: Structure, right: Right, strike_milli: int, chain: ChainSnapshot) -> Structure:
    """The same structure with the LONG leg of `right` moved to another strike (used to prove the fit is tight)."""
    from jevbot.types import Leg

    legs = []
    for leg in structure.legs:
        if leg.side is Side.BUY and leg.contract.right is right:
            legs.append(Leg(contract=contract_at(chain, structure.expiry, right, strike_milli), side=Side.BUY))
        else:
            legs.append(leg)
    ordered = tuple(sorted(legs, key=lambda leg: (leg.contract.right is Right.CALL, leg.contract.strike_milli)))
    return Structure(
        kind=structure.kind, underlying=structure.underlying, expiry=structure.expiry, last_session=structure.last_session, legs=ordered
    )


def long_leg(structure: Structure, right: Right) -> OptionContract:
    matches = [leg.contract for leg in structure.legs if leg.side is Side.BUY and leg.contract.right is right]
    assert len(matches) == 1
    return matches[0]


def short_strikes(structure: Structure) -> tuple[int, ...]:
    return tuple(sorted(leg.contract.strike_milli for leg in structure.legs if leg.side is Side.SELL))


# ======================================================================================================================
# the fill-model double is itself pinned to hand-computed 10.3 numbers
# ======================================================================================================================


def test_the_stub_fill_model_reproduces_the_10_3_band_table() -> None:
    cfg = Config()
    chain = make_chain()
    expiry = target_expiry(chain, 35)
    short = contract_at(chain, expiry, Right.PUT, 432_000)  # bid 310, ask 314
    long = contract_at(chain, expiry, Right.PUT, 427_000)  # bid 234, ask 238
    assert (quote_of(chain, short).bid, quote_of(chain, short).ask) == (310, 314)
    assert (quote_of(chain, long).bid, quote_of(chain, long).ask) == (234, 238)
    legs = (
        OrderLeg(contract=short, side=Side.SELL, position_intent=PositionIntent.STO),
        OrderLeg(contract=long, side=Side.BUY, position_intent=PositionIntent.BTO),
    )
    net, fills, quality = StubFillModel(cfg).price(legs, chain, mandatory=False)
    # two legs => p_bp 6600; step = cdiv(4 * 6600, 10000) = 3
    assert [(f.orats, f.worst, f.mid) for f in fills] == [(311, 310, 312), (237, 238, 236)]
    assert (net.orats, net.worst, net.mid) == (237 - 311, 238 - 310, 236 - 312) == (-74, -72, -76)
    assert quality == "ok"


# ======================================================================================================================
# 1. expiry choice (`last_session`, never the listed expiry)
# ======================================================================================================================


def test_the_target_expiry_is_chosen_by_dte_to_its_last_session() -> None:
    chain = make_chain()
    candidate = built(generator().build(StructureKind.PUT_CREDIT, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR))
    assert candidate.structure.expiry == date(2024, 6, 21)
    assert candidate.structure.last_session == date(2024, 6, 21)
    assert candidate.dte == DteConfig().target == 35
    assert candidate.sessions_to_expiry == xnys().sessions_between(date(2024, 5, 17), date(2024, 6, 21)) == 23
    assert candidate.key == chain.key
    assert DteConfig().min_entry <= candidate.dte <= DteConfig().max_entry


def test_a_saturday_dated_monthly_is_measured_to_its_friday() -> None:
    from tests.fixtures.chain_factory import saturday_monthly_chain

    chain = saturday_monthly_chain()
    candidate = built(generator().build(StructureKind.PUT_CREDIT, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR))
    assert candidate.structure.expiry == date(2024 - 10, 6, 21)  # 2014-06-21, a Saturday
    assert not xnys().is_session(candidate.structure.expiry)
    assert candidate.structure.last_session == date(2014, 6, 20)
    assert candidate.dte == 35 == (candidate.structure.last_session - chain.key.session).days
    assert candidate.sessions_to_expiry == xnys().sessions_between(chain.key.session, date(2014, 6, 20))


def test_a_good_friday_week_is_measured_to_the_thursday() -> None:
    from tests.fixtures.chain_factory import good_friday_chain

    chain = good_friday_chain()
    candidate = built(generator().build(StructureKind.PUT_CREDIT, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR))
    assert candidate.structure.expiry == date(2014, 4, 19)
    assert candidate.structure.last_session == date(2014, 4, 17)  # Good Friday 04-18 is a holiday
    assert candidate.dte == 35


def test_no_expiry_in_the_entry_window_is_an_early_reject() -> None:
    chain = make_chain(expiries=[date(2024, 5, 24), date(2024, 5, 31)])  # dte 7 and 14, far below dte.min_entry
    result = generator().build(StructureKind.PUT_CREDIT, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR)
    assert isinstance(result, CandidateReject)
    assert result.rejects == ("no_expiry_in_window",) and result.kind is StructureKind.PUT_CREDIT
    assert result.underlying == "SPY" and result.key == chain.key


def test_an_expiry_too_close_to_the_hard_exit_is_refused() -> None:
    chain = make_chain()
    tight = cfg_with(dte=DteConfig(min_sessions_beyond_hard_exit=30))  # needs sessions_to_expiry > 33; the 35-dte expiry has 23
    result = generator(tight).build(StructureKind.PUT_CREDIT, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR)
    assert isinstance(result, CandidateReject) and result.rejects == ("no_expiry_in_window",)
    assert isinstance(generator().build(StructureKind.PUT_CREDIT, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR), Candidate)


def test_an_expiry_without_enough_two_sided_strikes_is_refused() -> None:
    expiry = date(2024, 6, 21)
    chain = make_chain(expiries=[expiry])
    assert isinstance(generator().build(StructureKind.PUT_CREDIT, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR), Candidate)
    strict = cfg_with(candidates=CandidatesConfig(min_two_sided_frac=1.0))
    assert isinstance(generator(strict).build(StructureKind.PUT_CREDIT, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR), Candidate)
    holed = zero_bid(chain, contract_at(chain, expiry, Right.PUT, 44_500 * 10))  # one at-the-money strike loses its bid
    result = generator(strict).build(StructureKind.PUT_CREDIT, view_of(holed), "SPY", budget_floor=DEFAULT_FLOOR)
    assert isinstance(result, CandidateReject) and result.rejects == ("no_expiry_in_window",)


# ======================================================================================================================
# 2. delta rules
# ======================================================================================================================


def test_the_put_credit_spread_golden_on_the_factory_chain() -> None:
    """Hand-computed from the raw quotes: P432 bid 310 / ask 314, P427 bid 234 / ask 238 (2024-06-21, spot $450)."""
    chain = make_chain()
    candidate = built(generator().build(StructureKind.PUT_CREDIT, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR))
    structure = candidate.structure
    assert [(leg.side.value, leg.contract.right.value, leg.contract.strike_milli) for leg in structure.legs] == [
        ("buy", "P", 427_000),
        ("sell", "P", 432_000),
    ]
    assert structure.wing_widths == (500, 0) and structure.width == 500  # $5.00 wide
    assert (candidate.net.orats, candidate.net.worst, candidate.net.mid) == (-74, -72, -76)
    # fee_rt = ceil((4*40300 + 2*3290 + ceil(.0000206*23700*1e4) + ceil(.0000206*31100*1e4)) / 10000)
    #        = ceil((161200 + 6580 + 4883 + 6407) / 10000) = ceil(17.907) = 18 cents
    assert candidate.max_loss_per_contract == (500 - 72) * 100 + 18 == 42_818
    assert candidate.max_profit_per_contract == 74 * 100 == 7_400  # the credit at the HEADLINE (orats) band
    assert candidate.bp_required_per_contract == (500 - 72) * 100 == 42_800
    assert candidate.breakevens == (432_000 // 10 - 72,) == (43_128,)
    assert candidate.budget_floor == DEFAULT_FLOOR and candidate.rejects == ()
    # net_delta / net_vega are the signed per-share sums over the legs
    quotes = {q.contract.strike_milli: q for q in candidate.quotes}
    assert candidate.net_delta == pytest.approx(quotes[427_000].delta - quotes[432_000].delta)
    assert candidate.net_vega == pytest.approx(quotes[427_000].vega - quotes[432_000].vega)
    assert [q.contract.occ for q in candidate.quotes] == [leg.contract.occ for leg in structure.legs]


def test_the_long_call_golden_and_its_budget_step() -> None:
    """C460 is the nearest 0.35 delta (0.3581) but costs $5.10 -> max_loss 51,010 > $500; the fit steps one strike OTM."""
    chain = make_chain()
    expiry = target_expiry(chain, 35)
    assert nearest_delta(chain, expiry, Right.CALL, 0.35).strike_milli == 460_000
    candidate = built(generator().build(StructureKind.LONG_CALL, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR))
    assert [leg.contract.strike_milli for leg in candidate.structure.legs] == [461_000]
    assert (candidate.net.orats, candidate.net.worst, candidate.net.mid) == (470, 471, 469)  # bid 467 / ask 471, p_bp 7500
    assert candidate.max_loss_per_contract == 471 * 100 + 10 == 47_110
    assert candidate.max_profit_per_contract is None and candidate.short_distance_em is None
    assert candidate.bp_required_per_contract == 471 * 100 and candidate.breakevens == (46_100 + 471,)
    assert candidate.structure.width == 0 and candidate.rejects == ()
    # with a budget that fits the 0.35-delta strike the fit does not move at all
    roomy = built(generator().build(StructureKind.LONG_CALL, view_of(chain), "SPY", budget_floor=100_000))
    assert [leg.contract.strike_milli for leg in roomy.structure.legs] == [460_000]


def test_the_iron_condor_golden() -> None:
    chain = make_chain()
    candidate = built(generator().build(StructureKind.IRON_CONDOR, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR))
    assert [(leg.side.value, leg.contract.right.value, leg.contract.strike_milli) for leg in candidate.structure.legs] == [
        ("buy", "P", 421_000),
        ("sell", "P", 427_000),
        ("sell", "C", 472_000),
        ("buy", "C", 478_000),
    ]
    assert candidate.structure.wing_widths == (600, 600) and candidate.structure.width == 600
    assert (candidate.net.orats, candidate.net.worst) == (-149, -146)
    assert candidate.max_loss_per_contract == (600 - 146) * 100 + 35 == 45_435
    assert candidate.max_profit_per_contract == 149 * 100
    # condor_bp_mode "sum_wings": ((600 + 600) - 146) * 100
    assert candidate.bp_required_per_contract == (1200 - 146) * 100 == 105_400
    assert candidate.breakevens == (42_700 - 146, 47_200 + 146)
    assert candidate.rejects == ()
    assert structmath.defined_risk_ok(StructureKind.IRON_CONDOR, candidate.structure.legs)


# The whole factory-chain golden table: legs (side, right, strike_milli), width in cents, (orats, worst, mid) net and the
# worst-band max loss with `fee_rt`. Every row is reproducible by hand from the raw quotes and the 10.3 / 9.2 arithmetic,
# e.g. long_put: ask 486 -> 486 * 100 + fee_rt 10 = 48,610; call_credit: (600 - 115) * 100 + fee_rt 18 = 48,518.
FACTORY_GOLDENS: Final[dict[StructureKind, tuple[tuple[tuple[str, str, int], ...], int, tuple[int, int, int], int]]] = {
    StructureKind.LONG_CALL: ((("buy", "C", 461_000),), 0, (470, 471, 469), 47_110),
    StructureKind.LONG_PUT: ((("buy", "P", 440_000),), 0, (485, 486, 484), 48_610),
    StructureKind.CALL_DEBIT: ((("buy", "C", 454_000), ("sell", "C", 466_000)), 1_200, (473, 476, 471), 47_620),
    StructureKind.PUT_DEBIT: ((("sell", "P", 436_000), ("buy", "P", 449_000)), 1_300, (393, 396, 391), 39_620),
    StructureKind.CALL_CREDIT: ((("sell", "C", 468_000), ("buy", "C", 474_000)), 600, (-117, -115, -119), 48_518),
    StructureKind.PUT_CREDIT: ((("buy", "P", 427_000), ("sell", "P", 432_000)), 500, (-74, -72, -76), 42_818),
    StructureKind.IRON_CONDOR: (
        (("buy", "P", 421_000), ("sell", "P", 427_000), ("sell", "C", 472_000), ("buy", "C", 478_000)),
        600,
        (-149, -146, -153),
        45_435,
    ),
}


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_the_factory_chain_golden_of_every_kind(kind: StructureKind) -> None:
    """Expected strikes, widths and nets for all seven structures on the default $450 chain (test plan 15.1)."""
    legs, width, net, max_loss = FACTORY_GOLDENS[kind]
    candidate = built(generator().build(kind, view_of(make_chain()), "SPY", budget_floor=DEFAULT_FLOOR))
    assert tuple((leg.side.value, leg.contract.right.value, leg.contract.strike_milli) for leg in candidate.structure.legs) == legs
    assert candidate.structure.width == width
    assert (candidate.net.orats, candidate.net.worst, candidate.net.mid) == net
    assert candidate.max_loss_per_contract == max_loss <= DEFAULT_FLOOR
    assert candidate.net.mid <= candidate.net.orats <= candidate.net.worst  # mid is the best case, worst the natural
    assert candidate.rejects == ()


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_every_kind_is_defined_risk_and_canonically_ordered(kind: StructureKind) -> None:
    chain = make_chain()
    candidate = built(generator().build(kind, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR))
    legs = candidate.structure.legs
    assert structmath.defined_risk_ok(kind, legs)
    keys = [(leg.contract.right is Right.CALL, leg.contract.strike_milli) for leg in legs]
    assert keys == sorted(keys)  # puts before calls, then ascending strike
    assert {leg.contract.expiry for leg in legs} == {candidate.structure.expiry}
    assert all(leg.ratio == 1 for leg in legs)
    assert candidate.rejects == ()


def test_the_delta_tolerance_guards_the_short_leg() -> None:
    """Every put whose |delta| is anywhere near 0.25 loses its open interest: the short target becomes unreachable."""
    chain = make_chain()
    expiry = target_expiry(chain, 35)
    table = chain.table
    illiquid = chain
    for row in table[(table["expiry"] == pd.Timestamp(expiry)) & (table["right"] == "P")].itertuples():
        if not pd.isna(row.delta) and 0.14 <= abs(float(row.delta)) <= 0.45:
            illiquid = thin_oi(illiquid, contract_at(illiquid, expiry, Right.PUT, int(row.strike_milli)))
    result = generator().build(StructureKind.PUT_CREDIT, view_of(illiquid), "SPY", budget_floor=DEFAULT_FLOOR)
    assert isinstance(result, CandidateReject)
    assert result.rejects == ("delta_target_unreachable:short",)
    assert result.rejects[0] in vocab.CANDIDATE_REJECTS


def test_the_long_leg_of_a_spread_has_no_tolerance_test() -> None:
    """Section 8: the long leg's delta target is a preference and an outer bound, never a reject reason."""
    chain = make_chain()
    unreachable_long = cfg_with(candidates=CandidatesConfig(credit_long_delta=0.0001))
    candidate = built(generator(unreachable_long).build(StructureKind.PUT_CREDIT, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR))
    assert candidate.rejects == ()
    assert short_strikes(candidate.structure) == (432_000,)  # the short is exactly where the default config puts it


def test_a_long_single_needs_its_starting_strike_within_tolerance() -> None:
    chain = make_chain()
    impossible = cfg_with(candidates=CandidatesConfig(long_delta=0.999, delta_tolerance=0.001))
    result = generator(impossible).build(StructureKind.LONG_CALL, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR)
    assert isinstance(result, CandidateReject) and result.rejects == ("delta_target_unreachable:long",)


@pytest.mark.parametrize("kind", CREDIT_KINDS)
def test_credit_shorts_stand_at_least_0_8_expected_moves_from_spot(kind: StructureKind) -> None:
    chain = make_chain()
    candidate = built(generator().build(kind, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR))
    assert candidate.short_distance_em is not None
    assert candidate.short_distance_em >= CandidatesConfig().credit_short_min_em
    # without the guard the short sits on the raw nearest-delta strike instead
    unguarded = cfg_with(candidates=CandidatesConfig(credit_short_min_em=0.0))
    loose = built(generator(unguarded).build(kind, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR))
    assert loose.short_distance_em is not None
    expiry = target_expiry(chain, 35)
    raw = {
        StructureKind.PUT_CREDIT: (nearest_delta(chain, expiry, Right.PUT, 0.25, min_bid=10).strike_milli,),
        StructureKind.CALL_CREDIT: (nearest_delta(chain, expiry, Right.CALL, 0.25, min_bid=10).strike_milli,),
        StructureKind.IRON_CONDOR: tuple(
            sorted(nearest_delta(chain, expiry, right, 0.16, min_bid=10).strike_milli for right in (Right.PUT, Right.CALL))
        ),
    }[kind]
    assert short_strikes(loose.structure) == raw
    if kind is StructureKind.IRON_CONDOR:
        # the 0.16-delta short is about one expected move out already, so the guard is inert and moves nothing
        assert loose.short_distance_em == candidate.short_distance_em >= 0.8
        assert short_strikes(candidate.structure) == raw == (427_000, 472_000)
    else:
        assert loose.short_distance_em < 0.8 <= candidate.short_distance_em
        assert short_strikes(loose.structure) != short_strikes(candidate.structure)
    if kind is StructureKind.PUT_CREDIT:
        assert short_strikes(loose.structure) == (436_000,) and short_strikes(candidate.structure) == (432_000,)


def test_a_debit_spread_short_leg_has_no_expected_move_guard() -> None:
    chain = make_chain()
    candidate = built(generator().build(StructureKind.CALL_DEBIT, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR))
    assert short_strikes(candidate.structure) == (
        nearest_delta(chain, target_expiry(chain, 35), Right.CALL, 0.25, min_bid=10).strike_milli,
    )
    assert short_strikes(candidate.structure) == (466_000,)


# ======================================================================================================================
# 3. width rules
# ======================================================================================================================


def test_the_width_cap_moves_the_long_leg_toward_the_short() -> None:
    chain = make_chain()
    wide = built(generator().build(StructureKind.PUT_CREDIT, view_of(chain), "SPY", budget_floor=1_000_000))
    assert wide.structure.width == 432_000 // 10 - 421_000 // 10 == 1_100  # the 0.12-delta long, $11 wide
    narrow_cfg = cfg_with(candidates=CandidatesConfig(max_width_pct_spot=0.005))  # cap = floor(0.005 * 45000) = 225 cents
    narrow = built(generator(narrow_cfg).build(StructureKind.PUT_CREDIT, view_of(chain), "SPY", budget_floor=1_000_000))
    assert narrow.structure.width == 200 <= 225
    assert short_strikes(narrow.structure) == short_strikes(wide.structure)  # the short never moves


def test_no_eligible_strike_inside_the_width_cap_is_a_width_reject() -> None:
    chain = make_chain()
    impossible = cfg_with(candidates=CandidatesConfig(max_width_pct_spot=0.001))  # cap 45 cents, below one $1 strike step
    result = generator(impossible).build(StructureKind.PUT_CREDIT, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR)
    assert isinstance(result, CandidateReject) and result.rejects == ("width",)


def test_the_legs_of_a_spread_never_share_a_strike() -> None:
    chain = make_chain()
    for kind in ALL_KINDS:
        candidate = built(generator().build(kind, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR))
        strikes_by_right: dict[Right, list[int]] = {}
        for leg in candidate.structure.legs:
            strikes_by_right.setdefault(leg.contract.right, []).append(leg.contract.strike_milli)
        for right, values in strikes_by_right.items():
            assert len(set(values)) == len(values), (kind, right, values)
    tight = cfg_with(candidates=CandidatesConfig(min_width_strikes=4))
    candidate = built(generator(tight).build(StructureKind.PUT_CREDIT, view_of(chain), "SPY", budget_floor=1_000_000))
    assert candidate.structure.width >= 400  # at least four $1 strikes apart


# ======================================================================================================================
# 4. the budget fit (the acceptance case)
# ======================================================================================================================


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_budget_fit_on_the_600_dollar_chain(kind: StructureKind) -> None:
    """Section 8: every kind ends with `max_loss_per_contract <= budget_floor` or `exceeds_risk_budget`."""
    chain = six_hundred_dollar_chain()
    result = generator().build(kind, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR)
    if isinstance(result, CandidateReject):
        assert result.rejects == ("exceeds_risk_budget",)
        return
    assert result.max_loss_per_contract <= DEFAULT_FLOOR
    assert result.budget_floor == DEFAULT_FLOOR
    assert structmath.defined_risk_ok(kind, result.structure.legs)


@pytest.mark.parametrize("kind", CREDIT_KINDS)
def test_the_fitted_long_strike_is_the_furthest_otm_one_that_fits(kind: StructureKind) -> None:
    """One eligible strike further OTM (a wider wing) must already exceed the budget."""
    cfg = Config()
    fill = StubFillModel(cfg)
    chain = six_hundred_dollar_chain()
    candidate = built(generator(cfg).build(kind, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR))
    expiry = candidate.structure.expiry
    for right in (Right.PUT, Right.CALL):
        legs = [leg for leg in candidate.structure.legs if leg.contract.right is right and leg.side is Side.BUY]
        if not legs:
            continue
        listed = strikes(chain, expiry, right)
        position = listed.index(legs[0].contract.strike_milli)
        wider = position - 1 if right is Right.PUT else position + 1  # one step further OTM
        assert 0 <= wider < len(listed)
        widened = with_long_strike(candidate.structure, right, listed[wider], chain)
        assert max_loss_of(cfg, fill, widened, chain) > DEFAULT_FLOOR, (kind, right)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_the_short_leg_never_moves_with_the_budget(kind: StructureKind) -> None:
    chain = six_hundred_dollar_chain()
    tight = generator().build(kind, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR)
    roomy = built(generator().build(kind, view_of(chain), "SPY", budget_floor=100_000_000))
    if isinstance(tight, CandidateReject):
        pytest.skip(f"{kind.value} is unsizeable at the default floor on this chain")
    assert short_strikes(tight.structure) == short_strikes(roomy.structure)
    if short_strikes(tight.structure):
        assert tight.short_distance_em == roomy.short_distance_em


@pytest.mark.parametrize("kind", SINGLES)
def test_long_singles_stop_at_long_min_delta(kind: StructureKind) -> None:
    chain = six_hundred_dollar_chain()
    cfg = Config()
    candidate = built(generator(cfg).build(kind, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR))
    quote = candidate.quotes[0]
    assert quote.delta is not None and abs(quote.delta) >= cfg.candidates.long_min_delta
    # a budget so small that even the `long_min_delta` strike is too dear ends in exceeds_risk_budget, never below the floor
    result = generator(cfg).build(kind, view_of(chain), "SPY", budget_floor=100)
    assert isinstance(result, CandidateReject) and result.rejects == ("exceeds_risk_budget",)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_a_tiny_budget_exceeds_the_risk_budget_for_every_kind(kind: StructureKind) -> None:
    chain = make_chain()
    result = generator().build(kind, view_of(chain), "SPY", budget_floor=1_000)
    assert isinstance(result, CandidateReject) and result.rejects == ("exceeds_risk_budget",)


def test_early_failures_never_return_a_half_built_candidate() -> None:
    chain = make_chain()
    cases = [
        (Config(), make_chain(expiries=[date(2024, 5, 24)]), "no_expiry_in_window"),
        (cfg_with(candidates=CandidatesConfig(max_width_pct_spot=0.001)), chain, "width"),
        (cfg_with(candidates=CandidatesConfig(long_delta=0.999, delta_tolerance=0.0001)), chain, "delta_target_unreachable:long"),
    ]
    for cfg, snapshot, expected in cases:
        kind = StructureKind.LONG_CALL if expected.endswith(":long") else StructureKind.PUT_CREDIT
        result = generator(cfg).build(kind, view_of(snapshot), "SPY", budget_floor=DEFAULT_FLOOR)
        assert isinstance(result, CandidateReject) and result.rejects == (expected,)
    budget = generator().build(StructureKind.IRON_CONDOR, view_of(chain), "SPY", budget_floor=10)
    assert isinstance(budget, CandidateReject) and budget.rejects == ("exceeds_risk_budget",)


def test_budget_floor_at_follows_the_9_3_formula() -> None:
    cfg = Config()
    assert budget_floor_at(cfg, 100_000 * 100) == 50_000  # floor(0.01 * 10,000,000c * 0.5)
    assert budget_floor_at(cfg, 37_500 * 100) == 18_750
    assert budget_floor_at(cfg_with(risk=RiskConfig(max_loss_per_trade_pct=0.0075)), 100_000 * 100) == 37_500
    with pytest.raises(InvariantError):
        generator().build(StructureKind.PUT_CREDIT, view_of(make_chain()), "SPY", budget_floor=-1)


# ======================================================================================================================
# 5. liquidity, pricing sanity bounds, ex-dividend
# ======================================================================================================================


def test_eligible_is_the_structmath_filter() -> None:
    cfg = Config()
    gen = generator(cfg)
    chain = make_chain()
    expiry = target_expiry(chain, 35)
    crossed = contract_at(chain, expiry, Right.PUT, 430_000)
    thin = contract_at(chain, expiry, Right.PUT, 431_000)
    wide = contract_at(chain, expiry, Right.PUT, 433_000)
    hurt = widened_spread(thin_oi(crossed_quote(chain, crossed), thin), wide, 200)
    for contract, codes in (
        (crossed, ("liq:crossed",)),  # the sides swapped: ask < bid, so the relative-spread test is vacuous
        (thin, ("liq:oi",)),
        (wide, ("liq:spread",)),
    ):
        assert structmath.leg_liquidity_rejects(quote_of(hurt, contract), sold=True, cfg=cfg.liquidity) == codes
    eligible = gen.eligible(hurt, sold=True)
    kept = set(eligible["occ"])
    assert crossed.occ not in kept and thin.occ not in kept and wide.occ not in kept
    assert contract_at(hurt, expiry, Right.PUT, 432_000).occ in kept
    assert list(eligible.columns) == list(chain.table.columns)
    # the kept set of one expiry, recomputed here row by row straight from `structmath`
    rows = hurt.table[hurt.table["expiry"] == pd.Timestamp(expiry)]
    expected = {
        str(row.occ)
        for row in rows.itertuples()
        if not structmath.leg_liquidity_rejects(
            quote_of(hurt, contract_at(hurt, expiry, Right(row.right), int(row.strike_milli))), sold=True, cfg=cfg.liquidity
        )
    }
    assert set(eligible[eligible["expiry"] == pd.Timestamp(expiry)]["occ"]) == expected
    assert 0 < len(expected) < len(rows)  # far-OTM rows below the 10-cent sold-bid floor are dropped
    # a bought leg accepts a 1-cent bid, a sold leg needs 10
    cheap = chain.table[chain.table["bid"].between(1, 9)]
    if not cheap.empty:
        assert len(gen.eligible(chain, sold=False)) > len(gen.eligible(chain, sold=True))


def test_an_illiquid_short_strike_is_stepped_over_not_traded() -> None:
    chain = make_chain()
    expiry = target_expiry(chain, 35)
    hurt = thin_oi(chain, contract_at(chain, expiry, Right.PUT, 432_000))
    candidate = built(generator().build(StructureKind.PUT_CREDIT, view_of(hurt), "SPY", budget_floor=DEFAULT_FLOOR))
    assert short_strikes(candidate.structure) != (432_000,)
    assert candidate.rejects == ()  # the traded legs are always drawn from eligible rows


def test_the_default_credit_and_debit_floors_are_feasible_on_the_factory_chains() -> None:
    """Section 8: the shipped `[candidates]` ratio bounds must not reject what the delta targets produce on the two
    factory chains the test plan names ($450 and the $600-priced budget-fit chain)."""
    for chain in (make_chain(), six_hundred_dollar_chain()):
        for kind in ALL_KINDS:
            result = generator().build(kind, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR)
            assert isinstance(result, Candidate), (chain.spot, kind, result)
            assert result.rejects == (), (chain.spot, kind, result.rejects)
            assert result.max_loss_per_contract <= DEFAULT_FLOOR


def test_a_cheap_underlying_surfaces_the_credit_floor_instead_of_trading_it() -> None:
    """On a $100 chain the 0.25 / 0.12 wing is one strike wide and prices below `min_credit_to_width`: the structure is
    priced and REJECTED (`credit_to_width`), which is exactly what `data scan-candidates` is there to reveal (section 8)."""
    chain = make_chain("IWM", spot=10_000)
    result = built(generator().build(StructureKind.PUT_CREDIT, view_of(chain), "IWM", budget_floor=DEFAULT_FLOOR))
    assert result.structure.width == 100  # one $1 strike
    assert abs(result.net.orats) * 100 < CandidatesConfig().min_credit_to_width * result.structure.width * 100
    assert result.rejects == ("credit_to_width",)
    assert result.max_loss_per_contract > 0  # the economics are valid, only the floor is missed


@pytest.mark.parametrize(
    ("kind", "overrides", "expected"),
    [
        (StructureKind.PUT_CREDIT, {"min_credit_to_width": 0.40}, "credit_to_width"),
        (StructureKind.PUT_CREDIT, {"max_credit_to_width": 0.05}, "credit_to_width"),
        (StructureKind.IRON_CONDOR, {"condor_min_credit_to_width": 0.50}, "credit_to_width"),
        (StructureKind.CALL_DEBIT, {"max_debit_to_width": 0.05}, "debit_to_width"),
    ],
)
def test_the_sanity_bounds_reject_a_priced_structure(kind: StructureKind, overrides: dict[str, float], expected: str) -> None:
    chain = make_chain()
    cfg = cfg_with(candidates=CandidatesConfig(**overrides))
    result = generator(cfg).build(kind, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR)
    assert isinstance(result, Candidate)  # priced, so a Candidate with rejects - not a CandidateReject
    assert expected in result.rejects and all(code in vocab.CANDIDATE_REJECTS for code in result.rejects)


def test_the_ex_dividend_entry_block() -> None:
    """Section 8 / risk check 11: a short call below `spot + dividend` across a verified ex-date is rejected."""
    chain = make_chain()
    itm_short = cfg_with(candidates=CandidatesConfig(credit_short_delta=0.70, credit_short_min_em=0.0, max_width_pct_spot=0.20))
    ex_date = date(2024, 6, 14)
    known_at = datetime(2024, 5, 1, tzinfo=UTC)  # verified well before the decision snapshot (INV-14)
    event = ex_dividend_event("SPY", ex_date, 150, knowable_at=known_at)
    assert chain.key.session <= ex_date <= date(2024, 6, 21)
    blocked = built(generator(itm_short).build(StructureKind.CALL_CREDIT, view_of(chain, events=[event]), "SPY", budget_floor=10_000_000))
    short_call = short_strikes(blocked.structure)[0]
    assert short_call // 10 < chain.spot + 150  # an ITM short call
    assert "exdiv_short_call" in blocked.rejects
    clean = built(generator(itm_short).build(StructureKind.CALL_CREDIT, view_of(chain), "SPY", budget_floor=10_000_000))
    assert "exdiv_short_call" not in clean.rejects
    puts = built(generator().build(StructureKind.PUT_CREDIT, view_of(chain, events=[event]), "SPY", budget_floor=DEFAULT_FLOOR))
    assert puts.rejects == ()  # no short call, no block
    otm = built(generator().build(StructureKind.CALL_CREDIT, view_of(chain, events=[event]), "SPY", budget_floor=DEFAULT_FLOOR))
    assert otm.rejects == ()  # the 0.25-delta short call is far above spot + dividend
    far = ex_dividend_event("SPY", date(2024, 7, 12), 150, knowable_at=known_at)  # beyond last_session
    assert (
        "exdiv_short_call"
        not in built(
            generator(itm_short).build(StructureKind.CALL_CREDIT, view_of(chain, events=[far]), "SPY", budget_floor=10_000_000)
        ).rejects
    )


# ======================================================================================================================
# price increments (section 8)
# ======================================================================================================================


def test_tick_rounding_is_always_against_us() -> None:
    chain = make_chain()
    candidate = built(generator().build(StructureKind.PUT_CREDIT, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR))
    cfg = Config()
    assert structure_tick(cfg, candidate.structure, candidate.quotes) == 1  # SPY is a penny class at any price
    assert natural_limit(cfg, candidate.structure, candidate.quotes, candidate.net.worst) == -72
    # a non-penny class: leg mids are 236 and 312 cents -> ticks 5 and 10 -> the FINEST tick is 5
    generic = cfg_with(universe=UniverseConfig(penny_all=()))
    assert structure_tick(generic, candidate.structure, candidate.quotes) == 5
    # passive rounding of a credit keeps the sign and never accepts less: floor(-72 / 5) * 5 = -75
    assert natural_limit(generic, candidate.structure, candidate.quotes, candidate.net.worst) == -75
    debit = built(generator().build(StructureKind.LONG_CALL, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR))
    assert structure_tick(generic, debit.structure, debit.quotes) == 10  # a $4.69 mid is at or above $3.00
    assert natural_limit(generic, debit.structure, debit.quotes, debit.net.worst) == 470  # floor(471 / 10) * 10
    assert natural_limit(cfg, debit.structure, debit.quotes, debit.net.worst) == 471
    with pytest.raises(InvariantError):
        structure_tick(cfg, candidate.structure, candidate.quotes[:1])


def test_max_profit_follows_the_headline_band() -> None:
    chain = make_chain()
    headline = built(generator().build(StructureKind.PUT_CREDIT, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR))
    worst_cfg = cfg_with(cadence=CadenceConfig(fill_rule=FillRule.SAME_SNAPSHOT_WORST))
    worst = built(generator(worst_cfg).build(StructureKind.PUT_CREDIT, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR))
    assert headline.max_profit_per_contract == 74 * 100
    assert worst.max_profit_per_contract == 72 * 100  # the worst-band credit
    assert headline.max_loss_per_contract == worst.max_loss_per_contract  # max loss is always the WORST band


def test_build_is_deterministic_and_does_not_mutate_the_snapshot() -> None:
    chain = make_chain()
    before = chain.content_hash
    gen = generator()
    first = built(gen.build(StructureKind.IRON_CONDOR, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR))
    second = built(gen.build(StructureKind.IRON_CONDOR, view_of(chain), "SPY", budget_floor=DEFAULT_FLOOR))
    assert first == second
    assert chain.content_hash == before
    assert first.structure.structure_id == second.structure.structure_id


def test_a_missing_chain_propagates_the_data_error() -> None:
    from jevbot.errors import DataUnavailable

    chain = make_chain()
    with pytest.raises(DataUnavailable):
        generator().build(StructureKind.PUT_CREDIT, view_of(chain), "QQQ", budget_floor=DEFAULT_FLOOR)


# ======================================================================================================================
# scan()
# ======================================================================================================================


class StubProvider:
    """A `ChainProvider` over a handful of pre-built snapshots (no IO, no parquet)."""

    fidelity = Fidelity.EOD_QUOTES
    source = "stub"

    def __init__(self, chains: Sequence[ChainSnapshot]) -> None:
        self._chains: dict[tuple[str, SnapshotKey], ChainSnapshot] = {(c.underlying, c.key): c for c in chains}

    def underlyings(self) -> tuple[str, ...]:
        return tuple(sorted({u for u, _ in self._chains}))

    def keys(self, underlying: str, start: date, end: date) -> list[SnapshotKey]:
        return sorted(key for (u, key) in self._chains if u == underlying and start <= key.session <= end)

    def get_chain(self, underlying: str, key: SnapshotKey) -> ChainSnapshot | None:
        return self._chains.get((underlying, key))

    def manifest_hash(self) -> str:
        return "0" * 64


def _scan_sessions() -> list[date]:
    calendar = xnys()
    sessions = [date(2024, 5, 17)]
    for _ in range(2):
        sessions.append(calendar.next_session(sessions[-1]))
    return sessions


def test_scan_reports_counts_and_unsizeable_rates_per_tier() -> None:
    cfg = cfg_with(run=Config().run, structures=Config().structures)
    sessions = _scan_sessions()
    chains = [make_chain("SPY", session=s) for s in sessions] + [six_hundred_dollar_chain("QQQ", session=s) for s in sessions]
    frame = scan(cfg, StubProvider(chains), xnys(), sessions[0], sessions[-1], fill_model=StubFillModel(cfg))
    assert list(frame.columns) == list(scan_columns())
    assert set(frame["underlying"]) == {"SPY", "QQQ"}
    assert set(frame["kind"]) == {kind.value for kind in cfg.structures.enabled}
    assert set(frame["year"]) == {2024}
    assert (frame["sessions"] == len(sessions)).all()
    assert len(frame) == 2 * len(cfg.structures.enabled)
    for column in scan_columns():
        assert column in frame.columns
    spy_condor = frame[(frame["underlying"] == "SPY") & (frame["kind"] == "iron_condor")].iloc[0]
    assert spy_condor["priced"] == spy_condor["tradable"] == len(sessions)
    assert spy_condor["exceeds_risk_budget_rate"] == 0.0
    assert spy_condor["reject:exceeds_risk_budget"] == 0
    # the $500 budget floor buys exactly one 45,435c condor at every tier: floor(budget / max_loss) >= 1
    assert spy_condor["unsizeable_rate_50"] == 0.0
    for tier in SCAN_TIERS:
        assert spy_condor[f"size_zero_{tier // 10_000}"] == 0
        assert 0.0 <= spy_condor[f"unsizeable_rate_{tier // 10_000}"] <= 1.0
    assert frame["reject:exdiv_short_call"].sum() == 0  # scan() has no event source
    totals = frame[[f"reject:{code}" for code in vocab.CANDIDATE_REJECTS]].sum(axis=1)
    assert (totals >= 0).all()


def test_scan_counts_the_unsizeable_sessions_of_a_tiny_account() -> None:
    cfg = cfg_with(run=Config().run.__class__(initial_equity_usd=2_000))  # budget_floor = floor(0.01 * 200,000c * 0.5) = $10
    sessions = _scan_sessions()[:1]
    chains = [make_chain("SPY", session=s) for s in sessions]
    frame = scan(cfg, StubProvider(chains), xnys(), sessions[0], sessions[-1], fill_model=StubFillModel(cfg))
    assert budget_floor_at(cfg, 2_000 * 100) == 1_000
    row = frame[frame["kind"] == "put_credit_spread"].iloc[0]
    assert row["priced"] == 0 and row["tradable"] == 0
    assert row["reject:exceeds_risk_budget"] == 1 and row["exceeds_risk_budget_rate"] == 1.0
    for tier in SCAN_TIERS:
        assert row[f"unsizeable_rate_{tier // 10_000}"] == 1.0
    assert row["sessions"] == 1


def test_scan_uses_one_key_per_session() -> None:
    cfg = Config()
    session = date(2024, 5, 17)
    dec = make_chain("SPY", session=session, slot=Slot.DEC, fidelity=Fidelity.RECORDED_INDICATIVE)
    eod = make_chain("SPY", session=session, slot=Slot.EOD, fidelity=Fidelity.RECORDED_INDICATIVE)
    frame = scan(cfg, StubProvider([dec, eod]), xnys(), session, session, fill_model=StubFillModel(cfg))
    assert (frame["sessions"] == 1).all()


def test_scan_on_an_empty_window_returns_the_empty_frame() -> None:
    cfg = Config()
    frame = scan(cfg, StubProvider([]), xnys(), date(2024, 5, 1), date(2024, 5, 2), fill_model=StubFillModel(cfg))
    assert frame.empty and list(frame.columns) == list(scan_columns())


def test_scan_tiers_are_the_non_zero_sizing_tiers_of_7_6() -> None:
    cfg = Config()
    assert SCAN_TIERS == (500_000, 750_000, 1_000_000)
    assert {round(t * 1_000_000) for t in cfg.rules.tiers.environment if t > 0} == set(SCAN_TIERS)
    assert {round(b * 1_000_000) for _a, b in cfg.rules.tiers.score} == set(SCAN_TIERS)
    assert budget_floor_at(cfg, 100_000 * 100) * 2 == 100_000  # the lowest non-zero tier is 0.5
