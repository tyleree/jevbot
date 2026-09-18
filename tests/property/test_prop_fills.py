"""Property tests for src/jevbot/fills.py (WP06). Seeded numpy generators - no hypothesis dependency (DESIGN 1.1).

The references here are written independently of the code under test: the 10.3 band prices are evaluated with `decimal`
exactly as the prose states them (`bid + ceil((ask - bid) * p_bp / 10000)`, division and all), the marks of 10.5 are summed
leg by leg from the raw quotes, and the fee of 10.7 is re-derived from the fee table. The named property of 10.3 -
"for a buy `bid <= mid <= orats <= worst = ask`; for a sell the reverse" - is `test_leg_band_ordering`.
"""

import hashlib
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, getcontext
from typing import Final

import msgspec
import numpy as np
import pytest

from jevbot.config import CadenceConfig, Config, FillsConfig
from jevbot.errors import InvariantError
from jevbot.fills import BandFillModel, leg_band_prices, net_of, orats_p_bp
from jevbot.types import (
    Band,
    ChainSnapshot,
    FillRule,
    Leg,
    OptionContract,
    OrderLeg,
    PositionIntent,
    Quote,
    Right,
    Side,
    Structure,
    StructureKind,
)
from tests.fixtures import chain_factory as cf

getcontext().prec = 60

CONFIG: Final = Config()
WORST_HEADLINE: Final = msgspec.structs.replace(Config(), cadence=CadenceConfig(fill_rule=FillRule.SAME_SNAPSHOT_WORST))
OTHER_SHARES: Final = msgspec.structs.replace(Config(), fills=FillsConfig(orats_p=(0.5, 0.5, 0.5, 0.5)))
MODEL: Final = BandFillModel(CONFIG)

N_MARKETS: Final = 60  # random markets per test (each rebuilds four quotes on the factory chain)
N_QUOTES: Final = 500  # random quotes per pure-arithmetic property

PER_CONTRACT_MICRO: Final = 40_300  # orf + occ + cat + commission, in micro-dollars
TAF_MICRO: Final = 3_290  # taf_sell per sold contract
SEC_RATE: Final = Decimal("0.0000206")


def rng_for(tag: str) -> np.random.Generator:
    # a stable seed per test (Python's hash() is salted per process); draws never depend on test order (10.9)
    return np.random.Generator(np.random.PCG64(int(hashlib.sha256(tag.encode()).hexdigest()[:16], 16)))


def ceil_of(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def floor_of(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_FLOOR))


# ======================================================================================================================
# The shared market: four contracts on the factory chain whose quotes each draw overwrites
# ======================================================================================================================


def _build_market() -> tuple[ChainSnapshot, tuple[OptionContract, ...]]:
    """One narrow chain (a single ~35 DTE expiry, strikes around the spot) shared by every property below: each draw
    replaces the four quotes on a copy of it, and a small table keeps those copies cheap."""
    expiry = cf.target_expiry(cf.make_chain())
    chain = cf.make_chain(expiries=(expiry,), moneyness_window=0.06)
    contracts = (
        cf.contract_at(chain, expiry, Right.PUT, 440_000),
        cf.contract_at(chain, expiry, Right.PUT, 445_000),
        cf.contract_at(chain, expiry, Right.CALL, 455_000),
        cf.contract_at(chain, expiry, Right.CALL, 460_000),
    )
    return chain, contracts


_MARKET: Final = _build_market()


def market() -> tuple[ChainSnapshot, tuple[OptionContract, ...]]:
    return _MARKET


def draw_quote(rng: np.random.Generator) -> tuple[int, int]:
    """A quote of one of the shapes real chains carry: two-sided, zero-bid, absent, crossed, locked or very wide."""
    shape = int(rng.integers(0, 10))
    if shape == 0:
        return (0, 0)  # no quote at all
    if shape == 1:
        return (0, int(rng.integers(1, 40)))  # a zero-bid wing
    if shape == 2:
        low = int(rng.integers(1, 500))
        return (low + int(rng.integers(1, 30)), low)  # crossed
    if shape == 3:
        locked = int(rng.integers(1, 500))
        return (locked, locked)
    if shape == 4:
        bid = int(rng.integers(1, 500))
        return (bid, bid + int(rng.integers(40, 400)))  # very wide
    bid = int(rng.integers(0, 2_000))
    return (bid, bid + int(rng.integers(1, 12)))


def plant(chain: ChainSnapshot, quotes: dict[OptionContract, tuple[int, int]]) -> ChainSnapshot:
    for contract, (bid, ask) in quotes.items():
        chain = cf.set_quote(chain, contract, bid=bid, ask=ask)
    return chain


def draw_legs(rng: np.random.Generator, contracts: tuple[OptionContract, ...]) -> list[OrderLeg]:
    """An order over a random non-empty subset of the contracts, with random sides and position intents."""
    count = int(rng.integers(1, len(contracts) + 1))
    chosen = rng.permutation(len(contracts))[:count]
    legs: list[OrderLeg] = []
    for index in sorted(int(i) for i in chosen):
        side = Side.BUY if rng.integers(0, 2) else Side.SELL
        if side is Side.BUY:
            intent = PositionIntent.BTO if rng.integers(0, 2) else PositionIntent.BTC
        else:
            intent = PositionIntent.STO if rng.integers(0, 2) else PositionIntent.STC
        legs.append(OrderLeg(contract=contracts[index], side=side, position_intent=intent))
    return legs


def usable(leg: OrderLeg, quote: Quote) -> bool:
    """10.4 per-side usability, restated: a BUY needs an ask; a SELL-to-OPEN needs a bid as well; a SELL-to-CLOSE does not."""
    two_sided = quote.ask > 0 and quote.ask > quote.bid
    if leg.side is Side.BUY:
        return two_sided
    if leg.position_intent is PositionIntent.STO:
        return two_sided and quote.bid > 0
    return two_sided


# ======================================================================================================================
# 10.3 the per-leg bands
# ======================================================================================================================


def test_leg_band_ordering() -> None:
    """DESIGN 10.3: for a buy `bid <= mid <= orats <= worst = ask`; for a sell the reverse. With the 10.3 prices themselves
    re-derived from the prose with exact decimals."""
    rng = rng_for("leg-bands")
    for _ in range(N_QUOTES):
        bid = int(rng.integers(0, 5_000))
        ask = bid + int(rng.integers(1, 500))
        for n_legs in (1, 2, 3, 4, 5, 9):
            p_bp = orats_p_bp(n_legs)
            share = ceil_of(Decimal(ask - bid) * Decimal(p_bp) / Decimal(10_000))
            buy = leg_band_prices(Side.BUY, bid, ask, p_bp)
            sell = leg_band_prices(Side.SELL, bid, ask, p_bp)
            assert buy == (bid + share, ask, ceil_of(Decimal(bid + ask) / 2))
            assert sell == (ask - share, bid, floor_of(Decimal(bid + ask) / 2))
            assert bid <= buy[2] <= buy[0] <= buy[1] == ask  # the named property, for a buy
            assert ask >= sell[2] >= sell[0] >= sell[1] == bid  # ... and for a sell
            assert buy[0] + sell[0] == bid + ask  # the two ORATS prices are symmetric around the quote


def test_the_mid_band_is_always_the_best_case_and_the_worst_band_the_worst() -> None:
    """D13 / 10.3: `net[mid] <= net[orats] <= net[worst]` for EVERY order, in any market - mid is reported as the best case
    only. It holds leg by leg: a buy's bands are ordered upwards, a sell's downwards, and the sign flips in the net."""
    rng = rng_for("net-order")
    chain, contracts = market()
    for _ in range(N_MARKETS):
        planted = plant(chain, {contract: draw_quote(rng) for contract in contracts})
        legs = draw_legs(rng, contracts)
        net, leg_fills, _ = MODEL.price(legs, planted, mandatory=False)
        assert net.mid <= net.orats <= net.worst
        assert net == net_of(leg_fills)
        for leg, fill in zip(legs, leg_fills, strict=True):
            if leg.side is Side.BUY:
                assert fill.mid <= fill.orats <= fill.worst
            else:
                assert fill.mid >= fill.orats >= fill.worst
            assert fill.orats >= 0 and fill.worst >= 0 and fill.mid >= 0  # nobody pays or receives a negative premium


def test_pricing_never_leaves_the_quoted_market_when_it_is_usable() -> None:
    rng = rng_for("inside-market")
    chain, contracts = market()
    for _ in range(N_MARKETS):
        quotes = {contract: draw_quote(rng) for contract in contracts}
        planted = plant(chain, quotes)
        legs = draw_legs(rng, contracts)
        _, leg_fills, quality = MODEL.price(legs, planted, mandatory=False)
        degraded = False
        for leg, fill in zip(legs, leg_fills, strict=True):
            bid, ask = quotes[leg.contract]
            quote = cf.quote_of(planted, leg.contract)
            if not usable(leg, quote):
                degraded = True
                continue
            if leg.side is Side.SELL and leg.position_intent is not PositionIntent.STO and bid == 0:
                assert (fill.orats, fill.worst, fill.mid) == (0, 0, 0)  # the zero-bid sell-to-close rule of 10.4
                continue
            for band in (fill.orats, fill.worst, fill.mid):
                assert bid <= band <= ask  # never outside the quoted market, never a last price
        assert quality == ("degraded" if degraded else "ok")  # exactly the legs without a usable quote degrade a fill


# ======================================================================================================================
# 10.4 rejection and forced fills
# ======================================================================================================================


def test_rejection_is_band_independent() -> None:
    """10.4: `check` never looks at a band or at the ORATS shares, so the trade list is identical in all three balances."""
    rng = rng_for("band-independent")
    chain, contracts = market()
    worst_model, other_model = BandFillModel(WORST_HEADLINE), BandFillModel(OTHER_SHARES)
    assert MODEL.headline is Band.ORATS and worst_model.headline is Band.WORST
    seen: set[tuple[str, ...]] = set()
    for _ in range(N_MARKETS):
        planted = plant(chain, {contract: draw_quote(rng) for contract in contracts})
        legs = draw_legs(rng, contracts)
        qty = int(rng.integers(1, 150))  # wide enough to trip the displayed-size and open-interest caps as well
        codes = MODEL.check(legs, qty, planted, mandatory=False)
        assert worst_model.check(legs, qty, planted, mandatory=False) == codes
        assert other_model.check(legs, qty, planted, mandatory=False) == codes
        assert MODEL.check(legs, qty, planted, mandatory=True) == ()  # a forced fill is never rejected
        seen.add(codes)
    assert len({code for codes in seen for code in codes}) >= 4  # the draw really exercises several rejection reasons


def test_a_forced_fill_is_never_better_than_the_worst_band() -> None:
    rng = rng_for("forced")
    chain, contracts = market()
    for _ in range(N_MARKETS):
        planted = plant(chain, {contract: draw_quote(rng) for contract in contracts})
        legs = draw_legs(rng, contracts)
        unforced, _, _ = MODEL.price(legs, planted, mandatory=False)
        forced, forced_fills, _ = MODEL.price(legs, planted, mandatory=True)
        assert forced.orats == forced.worst == forced.mid  # a forced fill takes one price on all three bands
        assert forced.worst >= unforced.worst >= unforced.orats >= unforced.mid
        for leg, fill in zip(legs, forced_fills, strict=True):
            quote = cf.quote_of(planted, leg.contract)
            if not usable(leg, quote):
                continue
            if leg.side is Side.BUY:
                assert fill.worst >= quote.ask  # we pay at least the ask, plus the penalty
            else:
                assert fill.worst <= max(quote.bid, 0)  # we receive at most the bid, minus the penalty, never below 0


def test_a_zero_bid_sell_to_close_leg_is_never_rejected_and_always_sold_at_zero() -> None:
    rng = rng_for("zero-bid")
    chain, contracts = market()
    for _ in range(N_MARKETS):
        contract = contracts[int(rng.integers(0, len(contracts)))]
        ask = int(rng.integers(1, 800))  # any ask at all: even an absurdly wide zero-bid market
        planted = plant(chain, {contract: (0, ask)})
        close = OrderLeg(contract=contract, side=Side.SELL, position_intent=PositionIntent.STC)
        open_leg = OrderLeg(contract=contract, side=Side.SELL, position_intent=PositionIntent.STO)
        assert MODEL.check([close], 1, planted, mandatory=False) == ()  # never rejected, whatever the spread
        assert "no_quote" in MODEL.check([open_leg], 1, planted, mandatory=False)  # ... but never OPEN into a zero bid
        for mandatory in (False, True):
            net, fills, quality = MODEL.price([close], planted, mandatory=mandatory)
            assert (fills[0].orats, fills[0].worst, fills[0].mid) == (0, 0, 0)
            assert net.orats == net.worst == net.mid == 0 and quality == "ok"


# ======================================================================================================================
# 10.5 marks
# ======================================================================================================================


def condor_of(chain: ChainSnapshot, contracts: tuple[OptionContract, ...]) -> Structure:
    return Structure(
        kind=StructureKind.IRON_CONDOR,
        underlying=chain.underlying,
        expiry=contracts[0].expiry,
        last_session=chain.last_session(contracts[0].expiry),
        legs=(
            Leg(contract=contracts[0], side=Side.BUY),
            Leg(contract=contracts[1], side=Side.SELL),
            Leg(contract=contracts[2], side=Side.SELL),
            Leg(contract=contracts[3], side=Side.BUY),
        ),
    )


def test_marks_follow_the_10_5_formula_and_only_stale_on_an_unusable_leg() -> None:
    rng = rng_for("marks")
    chain, contracts = market()
    structure = condor_of(chain, contracts)
    last = (4_242, 4_040)
    for _ in range(N_MARKETS):
        quotes = {contract: draw_quote(rng) for contract in contracts}
        planted = plant(chain, quotes)
        liq, mid, stale = MODEL.liquidation(structure, planted, last)
        # the reference: liq = ask(shorts) - bid(longs); a long leg's zero bid is a mark of 0, never a stale quote
        want_liq = want_mid = 0
        want_stale = False
        for leg in structure.legs:
            bid, ask = quotes[leg.contract]
            short = leg.side is Side.SELL
            crossed = bid > 0 and ask <= bid
            if crossed or (short and ask <= 0):
                want_stale = True
                continue
            if short:
                want_liq += ask
                want_mid += ceil_of(Decimal(bid + ask) / 2)
            else:
                want_liq -= bid
                want_mid -= floor_of(Decimal(bid + ask) / 2)
        assert stale is want_stale
        assert (liq, mid) == (last if want_stale else (want_liq, want_mid))
        if not want_stale:  # closing costs at most the sum of the short asks
            assert liq <= sum(quotes[leg.contract][1] for leg in structure.legs if leg.side is Side.SELL)


def test_a_zero_bid_long_leg_never_makes_a_mark_stale() -> None:
    rng = rng_for("zero-bid-mark")
    chain, contracts = market()
    structure = condor_of(chain, contracts)
    for _ in range(N_MARKETS):
        quotes = {
            contracts[0]: (0, int(rng.integers(1, 20))),  # both long wings quoted 0 x ask
            contracts[3]: (0, int(rng.integers(1, 20))),
            contracts[1]: (200, 206),
            contracts[2]: (190, 196),
        }
        liq, mid, stale = MODEL.liquidation(structure, plant(chain, quotes), (1, 1))
        assert stale is False
        assert liq == 206 + 196  # the long legs fetch their zero bids: nothing is subtracted
        assert mid == 203 + 193 - (quotes[contracts[0]][1] // 2 + quotes[contracts[3]][1] // 2)


# ======================================================================================================================
# 10.7 fees
# ======================================================================================================================


def test_fees_follow_the_10_7_table_and_grow_with_size() -> None:
    rng = rng_for("fees")
    chain, contracts = market()
    for _ in range(N_MARKETS):
        quotes: dict[OptionContract, tuple[int, int]] = {}
        for contract in contracts:
            bid = int(rng.integers(20, 900))
            quotes[contract] = (bid, bid + int(rng.integers(1, 10)))
        planted = plant(chain, quotes)
        legs = draw_legs(rng, contracts)
        leg_fills = MODEL.price(legs, planted, mandatory=False)[1]
        previous = 0
        for qty in (1, 2, 5, 13):
            fees = MODEL.fees_micro(legs, qty, leg_fills)
            contracts_traded = qty * len(legs)
            sold = qty * sum(1 for leg in legs if leg.side is Side.SELL)
            notional = sum(fill.orats * 100 * qty for leg, fill in zip(legs, leg_fills, strict=True) if leg.side is Side.SELL)
            want = contracts_traded * PER_CONTRACT_MICRO + sold * TAF_MICRO + ceil_of(SEC_RATE * Decimal(notional) * Decimal(10_000))
            assert fees == want
            assert fees > previous  # more contracts always cost more
            previous = fees


def test_fees_refuse_a_leg_list_that_does_not_match_the_fill() -> None:
    chain, contracts = market()
    planted = plant(chain, {contract: (100, 106) for contract in contracts})
    legs = [
        OrderLeg(contract=contracts[0], side=Side.BUY, position_intent=PositionIntent.BTO),
        OrderLeg(contract=contracts[1], side=Side.SELL, position_intent=PositionIntent.STO),
    ]
    leg_fills = MODEL.price(legs, planted, mandatory=False)[1]
    with pytest.raises(InvariantError):
        MODEL.fees_micro(list(reversed(legs)), 1, leg_fills)
