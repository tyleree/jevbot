"""Unit tests for src/jevbot/fills.py: the three fill bands, the band-independent rejection rules, forced fills,
liquidation marks and the per-fill fee (DESIGN.md 10.3-10.5, 10.7; WP06).

Every expected number below was computed BY HAND from the 10.3 table / the 10.7 formula - the arithmetic is written out in the
comment beside it - and never by calling the code under test. Quotes are planted on the shared `chain_factory` snapshot with
`set_quote`, so each golden is a small, fully specified market.
"""

import inspect
from datetime import date

import msgspec
import pytest

from jevbot.config import CadenceConfig, Config, FeesConfig, FillsConfig
from jevbot.errors import InvariantError
from jevbot.fills import (
    BandFillModel,
    end_of_day_fee_cents,
    intrinsic_cents,
    leg_band_prices,
    net_of,
    orats_p_bp,
)
from jevbot.structmath import fill_fees_micro
from jevbot.types import (
    Band,
    BandPrices,
    ChainSnapshot,
    Fidelity,
    FillRule,
    Leg,
    LegFill,
    OptionContract,
    OrderLeg,
    PositionIntent,
    Right,
    Side,
    Slot,
    Structure,
    StructureKind,
)
from jevbot.vocab import FILL_REJECTS
from tests.fixtures import chain_factory as cf

SPOT = cf.DEFAULT_SPOT  # 45_000 cents = $450.00
CONFIG = Config()
WORST_HEADLINE = msgspec.structs.replace(Config(), cadence=CadenceConfig(fill_rule=FillRule.SAME_SNAPSHOT_WORST))

NO_QUOTE, CROSSED, WIDE_SPREAD, SIZE, OPEN_INTEREST, STALE_QUOTE, MISSING_CONTRACT = FILL_REJECTS


# ======================================================================================================================
# Helpers: a planted market on the shared factory chain
# ======================================================================================================================


def base_chain(*, slot: Slot = Slot.EOD, fidelity: Fidelity = Fidelity.EOD_QUOTES) -> tuple[ChainSnapshot, date]:
    """The factory chain and the ~35 DTE expiry every golden below is priced on."""
    chain = cf.make_chain(slot=slot, fidelity=fidelity)
    return chain, cf.target_expiry(chain)


def plant(chain: ChainSnapshot, quotes: dict[OptionContract, tuple[int, int]]) -> ChainSnapshot:
    """A copy of `chain` in which each contract is quoted exactly `(bid, ask)`."""
    for contract, (bid, ask) in quotes.items():
        chain = cf.set_quote(chain, contract, bid=bid, ask=ask)
    return chain


def at(chain: ChainSnapshot, expiry: date, right: Right, strike_usd: int) -> OptionContract:
    return cf.contract_at(chain, expiry, right, strike_usd * 1000)


def order_leg(c: OptionContract, side: Side, intent: PositionIntent, ratio: int = 1) -> OrderLeg:
    return OrderLeg(contract=c, side=side, position_intent=intent, ratio=ratio)


def buy_to_open(c: OptionContract) -> OrderLeg:
    return order_leg(c, Side.BUY, PositionIntent.BTO)


def sell_to_open(c: OptionContract) -> OrderLeg:
    return order_leg(c, Side.SELL, PositionIntent.STO)


def sell_to_close(c: OptionContract) -> OrderLeg:
    return order_leg(c, Side.SELL, PositionIntent.STC)


def buy_to_close(c: OptionContract) -> OrderLeg:
    return order_leg(c, Side.BUY, PositionIntent.BTC)


CONDOR_QUOTES: dict[str, tuple[int, int]] = {
    "long_put": (60, 66),
    "short_put": (150, 158),
    "short_call": (140, 146),
    "long_call": (55, 59),
}


def condor(chain: ChainSnapshot, expiry: date) -> tuple[Structure, dict[str, OptionContract]]:
    """A 440/445 put x 455/460 call iron condor in canonical leg order (puts before calls, ascending strike)."""
    legs = {
        "long_put": at(chain, expiry, Right.PUT, 440),
        "short_put": at(chain, expiry, Right.PUT, 445),
        "short_call": at(chain, expiry, Right.CALL, 455),
        "long_call": at(chain, expiry, Right.CALL, 460),
    }
    structure = Structure(
        kind=StructureKind.IRON_CONDOR,
        underlying=chain.underlying,
        expiry=expiry,
        last_session=chain.last_session(expiry),
        legs=(
            Leg(contract=legs["long_put"], side=Side.BUY),
            Leg(contract=legs["short_put"], side=Side.SELL),
            Leg(contract=legs["short_call"], side=Side.SELL),
            Leg(contract=legs["long_call"], side=Side.BUY),
        ),
    )
    return structure, legs


def planted_condor() -> tuple[BandFillModel, ChainSnapshot, Structure, dict[str, OptionContract]]:
    chain, expiry = base_chain()
    structure, legs = condor(chain, expiry)
    return (BandFillModel(CONFIG), plant(chain, {legs[name]: quote for name, quote in CONDOR_QUOTES.items()}), structure, legs)


def condor_open_legs(legs: dict[str, OptionContract]) -> list[OrderLeg]:
    return [
        buy_to_open(legs["long_put"]),
        sell_to_open(legs["short_put"]),
        sell_to_open(legs["short_call"]),
        buy_to_open(legs["long_call"]),
    ]


# ======================================================================================================================
# 10.3 p_bp and the per-leg band arithmetic
# ======================================================================================================================


def test_orats_p_bp_is_the_leg_class_table_of_10_3() -> None:
    assert [orats_p_bp(n) for n in (1, 2, 3, 4)] == [7500, 6600, 5600, 5300]  # 0.75 / 0.66 / 0.56 / 0.53
    assert orats_p_bp(5) == orats_p_bp(9) == 5300  # the "4+" class
    model = BandFillModel(CONFIG)
    assert [model.p_bp(n) for n in (1, 2, 3, 4, 7)] == [7500, 6600, 5600, 5300, 5300]
    # a fraction that is not a whole number of basis points is rounded UP (a larger share of the spread is worse on either side)
    assert orats_p_bp(1, (0.123456, 0.5, 0.5, 0.5)) == 1235  # 1234.56 bp -> 1235
    assert orats_p_bp(1, (1.0, 1.0, 1.0, 1.0)) == 10_000
    for bad in (0, -1, True):
        with pytest.raises(InvariantError):
            orats_p_bp(bad)  # type: ignore[arg-type]
    with pytest.raises(InvariantError):
        orats_p_bp(1, (0.75, 0.66, 0.56))  # four leg classes or nothing
    with pytest.raises(InvariantError):
        orats_p_bp(1, (1.5, 0.66, 0.56, 0.53))  # a share above the whole spread
    with pytest.raises(InvariantError):
        orats_p_bp(1, (float("nan"), 0.66, 0.56, 0.53))
    with pytest.raises(InvariantError):
        BandFillModel(CONFIG).p_bp(0)


def test_leg_band_prices_goldens_and_ordering() -> None:
    # bid 100, ask 110, p_bp 7500: share = ceil(10 * 7500 / 10000) = ceil(7.5) = 8
    assert leg_band_prices(Side.BUY, 100, 110, 7500) == (108, 110, 105)  # bid + 8, ask, ceil(210 / 2)
    assert leg_band_prices(Side.SELL, 100, 110, 7500) == (102, 100, 105)  # ask - 8, bid, 210 // 2
    # an odd mid rounds against us: the buyer pays up, the seller receives down
    assert leg_band_prices(Side.BUY, 100, 111, 6600) == (108, 111, 106)  # share = ceil(11 * 0.66) = ceil(7.26) = 8
    assert leg_band_prices(Side.SELL, 100, 111, 6600) == (103, 100, 105)  # 111 - 8 ; 211 // 2 = 105
    # a zero bid is a perfectly good BUY quote (10.4)
    assert leg_band_prices(Side.BUY, 0, 4, 5300) == (3, 4, 2)  # share = ceil(4 * 0.53) = ceil(2.12) = 3 ; ceil(4 / 2) = 2
    # every band is ordered: bid <= mid <= orats <= worst = ask for a buy, and the reverse for a sell
    for bid, ask, p_bp in ((100, 110, 7500), (0, 4, 5300), (7, 8, 5600)):
        buy = leg_band_prices(Side.BUY, bid, ask, p_bp)
        sell = leg_band_prices(Side.SELL, bid, ask, p_bp)
        assert bid <= buy[2] <= buy[0] <= buy[1] == ask
        assert ask >= sell[2] >= sell[0] >= sell[1] == bid
    # a locked / crossed / absent quote never reaches this function
    for bad_bid, bad_ask in ((0, 0), (110, 100), (-1, 10)):
        with pytest.raises(InvariantError):
            leg_band_prices(Side.BUY, bad_bid, bad_ask, 7500)
    with pytest.raises(InvariantError):
        leg_band_prices(Side.BUY, 100, 110, 10_001)


def test_net_of_signs_and_intrinsic_rounding() -> None:
    legs = (
        LegFill(occ="A", side=Side.BUY, bid=100, ask=110, orats=108, worst=110, mid=105),
        LegFill(occ="B", side=Side.SELL, bid=40, ask=50, orats=43, worst=40, mid=45),
    )
    assert net_of(legs) == BandPrices(orats=65, worst=70, mid=60)  # 108-43, 110-40, 105-45: positive = debit
    assert net_of(()) == BandPrices(orats=0, worst=0, mid=0)
    expiry = date(2024, 6, 21)
    call = OptionContract(underlying="SPY", expiry=expiry, right=Right.CALL, strike_milli=445_500)
    put = OptionContract(underlying="SPY", expiry=expiry, right=Right.PUT, strike_milli=455_500)
    # spot 45_000 cents = 450_000 milli: call intrinsic 4_500 milli = 450 cents, put intrinsic 5_500 milli = 550 cents
    assert intrinsic_cents(call, SPOT, up=True) == intrinsic_cents(call, SPOT, up=False) == 450
    assert intrinsic_cents(put, SPOT, up=True) == 550
    half = OptionContract(underlying="SPY", expiry=expiry, right=Right.CALL, strike_milli=449_995)
    assert intrinsic_cents(half, SPOT, up=True) == 1 and intrinsic_cents(half, SPOT, up=False) == 0  # 5 milli = half a cent
    otm = OptionContract(underlying="SPY", expiry=expiry, right=Right.CALL, strike_milli=460_000)
    assert intrinsic_cents(otm, SPOT, up=True) == 0


# ======================================================================================================================
# 10.3 band goldens for 1, 2, 3 and 4 legs
# ======================================================================================================================


def test_band_golden_one_leg() -> None:
    chain, expiry = base_chain()
    c = at(chain, expiry, Right.CALL, 455)
    chain = plant(chain, {c: (250, 256)})
    net, legs, quality = BandFillModel(CONFIG).price([buy_to_open(c)], chain, mandatory=False)
    # one leg: p_bp 7500 ; spread 6 -> share = ceil(6 * 0.75) = ceil(4.5) = 5
    assert legs == (LegFill(occ=c.occ, side=Side.BUY, bid=250, ask=256, orats=255, worst=256, mid=253),)
    assert net == BandPrices(orats=255, worst=256, mid=253) and quality == "ok"
    assert net.mid < net.orats < net.worst  # a debit: mid is the best case, worst the worst


def test_band_golden_two_legs_call_debit() -> None:
    chain, expiry = base_chain()
    long_call = at(chain, expiry, Right.CALL, 450)
    short_call = at(chain, expiry, Right.CALL, 455)
    chain = plant(chain, {long_call: (300, 308), short_call: (120, 126)})
    net, legs, quality = BandFillModel(CONFIG).price([buy_to_open(long_call), sell_to_open(short_call)], chain, mandatory=False)
    # two legs: p_bp 6600. BUY  spread 8 -> ceil(8 * 0.66) = ceil(5.28) = 6 -> (306, 308, ceil(608 / 2) = 304)
    #                      SELL spread 6 -> ceil(6 * 0.66) = ceil(3.96) = 4 -> (122, 120, 246 // 2 = 123)
    assert legs[0] == LegFill(occ=long_call.occ, side=Side.BUY, bid=300, ask=308, orats=306, worst=308, mid=304)
    assert legs[1] == LegFill(occ=short_call.occ, side=Side.SELL, bid=120, ask=126, orats=122, worst=120, mid=123)
    assert net == BandPrices(orats=184, worst=188, mid=181) and quality == "ok"  # 306-122, 308-120, 304-123


def test_band_golden_three_legs() -> None:
    chain, expiry = base_chain()
    long_call = at(chain, expiry, Right.CALL, 450)
    short_call = at(chain, expiry, Right.CALL, 455)
    short_put = at(chain, expiry, Right.PUT, 440)
    chain = plant(chain, {long_call: (100, 104), short_call: (60, 66), short_put: (30, 34)})
    legs_in = [buy_to_open(long_call), sell_to_open(short_call), sell_to_open(short_put)]
    net, legs, _ = BandFillModel(CONFIG).price(legs_in, chain, mandatory=False)
    # three legs: p_bp 5600. BUY  spread 4 -> ceil(2.24) = 3 -> (103, 104, ceil(204 / 2) = 102)
    #   SELL spread 6 -> ceil(3.36) = 4 -> (62, 60, 126 // 2 = 63) ; SELL spread 4 -> 3 -> (31, 30, 64 // 2 = 32)
    assert [(leg.orats, leg.worst, leg.mid) for leg in legs] == [(103, 104, 102), (62, 60, 63), (31, 30, 32)]
    assert net == BandPrices(orats=10, worst=14, mid=7)  # 103-62-31, 104-60-30, 102-63-32


def test_band_golden_four_legs_iron_condor() -> None:
    model, chain, _, legs = planted_condor()
    net, leg_fills, quality = model.price(condor_open_legs(legs), chain, mandatory=False)
    # four legs: p_bp 5300.  long put   spread 6 -> ceil(3.18) = 4 -> (64, 66, ceil(126 / 2) = 63)
    #   short put  spread 8 -> ceil(4.24) = 5 -> (153, 150, 308 // 2 = 154)
    #   short call spread 6 -> ceil(3.18) = 4 -> (142, 140, 286 // 2 = 143)
    #   long call  spread 4 -> ceil(2.12) = 3 -> (58, 59, ceil(114 / 2) = 57)
    assert [(f.orats, f.worst, f.mid) for f in leg_fills] == [(64, 66, 63), (153, 150, 154), (142, 140, 143), (58, 59, 57)]
    # net = buys - sells: (64+58) - (153+142) = -173 ; (66+59) - (150+140) = -165 ; (63+57) - (154+143) = -177
    assert net == BandPrices(orats=-173, worst=-165, mid=-177) and quality == "ok"
    assert net.mid < net.orats < net.worst  # a credit: mid receives the most, worst the least


def test_a_ratio_leg_is_refused_wherever_it_could_be_half_counted() -> None:
    """2.4: v1 ratios are 1 and `LegFill` carries no ratio field.

    A ratio-2 leg would be counted ONCE in the signed net of 10.3 (`net_of` sums the per-leg prices) and TWICE in the fee of
    10.7 and the contract counts of 10.4, so `Book.apply`'s `cash[b] -= net[b] * 100 * qty` would book half the cash the mark
    and the fee assume - and no replay could recover the ratio from the ledger. Every entry point refuses it instead.
    """
    chain, expiry = base_chain()
    c = at(chain, expiry, Right.CALL, 455)
    chain = plant(chain, {c: (100, 110)})
    model = BandFillModel(CONFIG)
    unit = order_leg(c, Side.BUY, PositionIntent.BTO)
    net, leg_fills, _ = model.price([unit], chain, mandatory=False)
    assert (leg_fills[0].orats, net.orats) == (108, 108)  # one leg, one leg class: 100 + ceil(10 * 0.75) = 108
    assert model.fees_micro([unit], 1, leg_fills) == 40_300  # ... and exactly one contract's fee: ceil(0.0403 * 1e6)
    for ratio in (2, 3, 0, -1, True):
        odd = order_leg(c, Side.BUY, PositionIntent.BTO, ratio=ratio)
        with pytest.raises(InvariantError, match="ratio"):
            model.price([odd], chain, mandatory=False)
        with pytest.raises(InvariantError, match="ratio"):
            model.price([odd], chain, mandatory=True)
        with pytest.raises(InvariantError, match="ratio"):
            model.check([odd], 1, chain, mandatory=False)
        with pytest.raises(InvariantError, match="ratio"):
            model.fees_micro([odd], 1, leg_fills)  # the leg lists still line up: it is the ratio that is refused
    # the 10.3 leg class is the NUMBER of legs, not the sum of their ratios
    other = at(chain, expiry, Right.PUT, 440)
    two = model.price([unit, sell_to_open(other)], plant(chain, {other: (100, 110)}), mandatory=False)[1]
    assert two[0].orats == 107  # two legs: 100 + ceil(10 * 0.66) = 107


def test_price_refuses_an_empty_or_foreign_order() -> None:
    chain, expiry = base_chain()
    c = at(chain, expiry, Right.CALL, 455)
    model = BandFillModel(CONFIG)
    with pytest.raises(InvariantError):
        model.price([], chain, mandatory=False)
    foreign = OptionContract(underlying="QQQ", expiry=c.expiry, right=Right.CALL, strike_milli=c.strike_milli)
    with pytest.raises(InvariantError):
        model.price([buy_to_open(foreign)], chain, mandatory=False)


# ======================================================================================================================
# 10.4 rejection rules: every code, band-independent
# ======================================================================================================================


def test_each_rejection_code_of_10_4() -> None:
    chain, expiry = base_chain()
    c = at(chain, expiry, Right.CALL, 455)
    other = at(chain, expiry, Right.PUT, 440)
    model = BandFillModel(CONFIG)
    healthy = plant(chain, {c: (100, 106)})
    assert model.check([buy_to_open(c)], 1, healthy, mandatory=False) == ()

    # no_quote (a): a BUY leg without an ask
    assert model.check([buy_to_open(c)], 1, plant(chain, {c: (0, 0)}), mandatory=False) == (NO_QUOTE,)
    # no_quote (b): a SELL leg of an OPEN order into a zero bid - we never open by selling into a zero-bid market
    assert model.check([sell_to_open(c)], 1, plant(chain, {c: (0, 4)}), mandatory=False) == (NO_QUOTE,)
    # ... and the same leg SOLD TO CLOSE is not a reject at all (the condor-wing case of 10.4)
    assert model.check([sell_to_close(c)], 1, plant(chain, {c: (0, 4)}), mandatory=False) == ()

    # crossed_or_locked: ask <= bid with bid > 0
    assert model.check([buy_to_open(c)], 1, plant(chain, {c: (110, 100)}), mandatory=False) == (CROSSED,)
    assert model.check([sell_to_close(c)], 1, plant(chain, {c: (100, 100)}), mandatory=False) == (CROSSED,)

    # wide_spread: spread > max(0.25 * mid, 15). bid 100 ask 160 -> 60 > max(32.5, 15)
    assert model.check([buy_to_open(c)], 1, plant(chain, {c: (100, 160)}), mandatory=False) == (WIDE_SPREAD,)
    assert model.check([buy_to_open(c)], 1, plant(chain, {c: (10, 24)}), mandatory=False) == ()  # 14 <= the 15-cent allowance
    assert model.check([buy_to_open(c)], 1, plant(chain, {c: (10, 26)}), mandatory=False) == (WIDE_SPREAD,)  # 16 > max(4.5, 15)
    # a zero-bid sell-to-close leg is never spread-checked (10.4), while buying the same row is far too wide
    wide_zero_bid = plant(chain, {c: (0, 400)})
    assert model.check([sell_to_close(c)], 1, wide_zero_bid, mandatory=False) == ()
    assert model.check([buy_to_open(c)], 1, wide_zero_bid, mandatory=False) == (WIDE_SPREAD,)

    # size: qty > 0.50 * displayed size on the side we hit (the ask size for a buy, the bid size for a sell)
    sized = cf.set_quote(healthy, c, bid_size=40, ask_size=10)
    assert model.check([buy_to_open(c)], 5, sized, mandatory=False) == ()  # 5 > 5.0 is false
    assert model.check([buy_to_open(c)], 6, sized, mandatory=False) == (SIZE,)
    assert model.check([sell_to_close(c)], 20, sized, mandatory=False) == ()  # the bid side is deeper
    assert model.check([sell_to_close(c)], 21, sized, mandatory=False) == (SIZE,)
    assert model.check([buy_to_open(c)], 99, cf.set_quote(healthy, c, bid_size=None, ask_size=None), mandatory=False) == ()

    # open_interest: qty > 0.05 * oi_prev
    thin = cf.thin_oi(healthy, c, oi_prev=100)
    assert model.check([buy_to_open(c)], 5, thin, mandatory=False) == ()
    assert model.check([buy_to_open(c)], 6, thin, mandatory=False) == (OPEN_INTEREST,)
    sizeless = cf.set_quote(healthy, c, bid_size=None, ask_size=None)
    assert model.check([buy_to_open(c)], 99, cf.missing_oi(sizeless, c), mandatory=False) == ()  # neither limit is knowable

    # missing_contract
    assert model.check([buy_to_open(c)], 1, cf.drop_contract(healthy, c), mandatory=False) == (MISSING_CONTRACT,)

    # several codes over several legs, deduplicated and in the frozen vocabulary order
    messy = plant(chain, {c: (110, 100), other: (0, 0)})
    assert model.check([buy_to_open(c), buy_to_open(other)], 1, messy, mandatory=False) == (NO_QUOTE, CROSSED)


def test_stale_quotes_are_only_a_reject_on_timed_data() -> None:
    recorded, expiry = base_chain(slot=Slot.EXEC, fidelity=Fidelity.RECORDED_INDICATIVE)
    c = at(recorded, expiry, Right.CALL, 455)
    model = BandFillModel(CONFIG)
    fresh = plant(recorded, {c: (100, 106)})
    assert model.check([buy_to_open(c)], 1, fresh, mandatory=False) == ()  # 5 s old; the allowance is 1200 s
    assert model.check([buy_to_open(c)], 1, cf.stale_quote(fresh, c, age_s=1201), mandatory=False) == (STALE_QUOTE,)
    assert model.check([buy_to_open(c)], 1, cf.stale_quote(fresh, c, age_s=1200), mandatory=False) == ()  # not OLDER than
    assert model.check([buy_to_open(c)], 1, cf.set_quote(fresh, c, quote_ts=None), mandatory=False) == (STALE_QUOTE,)  # fail closed
    # EOD data carries no feed timestamp and is never stale-checked
    eod, eod_expiry = base_chain()
    eod_contract = at(eod, eod_expiry, Right.CALL, 455)
    assert model.check([buy_to_open(eod_contract)], 1, plant(eod, {eod_contract: (100, 106)}), mandatory=False) == ()


def test_rejection_is_identical_across_bands_and_leg_classes() -> None:
    """10.4: `check` never looks at a band, so the trade list is the same in all three cash balances."""
    chain, expiry = base_chain()
    c = at(chain, expiry, Right.CALL, 455)
    other = at(chain, expiry, Right.PUT, 440)
    headline_orats = BandFillModel(CONFIG)
    headline_worst = BandFillModel(WORST_HEADLINE)
    other_shares = BandFillModel(msgspec.structs.replace(Config(), fills=FillsConfig(orats_p=(0.5, 0.5, 0.5, 0.5))))
    assert headline_orats.headline is Band.ORATS and headline_worst.headline is Band.WORST
    for bid, ask in ((100, 106), (0, 0), (0, 4), (110, 100), (100, 160), (10, 24)):
        planted = plant(chain, {c: (bid, ask), other: (bid, ask)})
        for legs in ([buy_to_open(c)], [sell_to_open(c)], [sell_to_close(c)], [buy_to_close(c), sell_to_close(other)]):
            expected = headline_orats.check(legs, 3, planted, mandatory=False)
            assert headline_worst.check(legs, 3, planted, mandatory=False) == expected
            assert other_shares.check(legs, 3, planted, mandatory=False) == expected


def test_check_refuses_a_nonsense_quantity_and_accepts_every_mandatory_order() -> None:
    chain, expiry = base_chain()
    c = at(chain, expiry, Right.CALL, 455)
    model = BandFillModel(CONFIG)
    broken = cf.drop_contract(plant(chain, {c: (110, 100)}), at(chain, expiry, Right.PUT, 440))
    assert model.check([buy_to_open(c)], 1, broken, mandatory=False) == (CROSSED,)
    assert model.check([buy_to_open(c)], 1, broken, mandatory=True) == ()  # a forced fill is never rejected
    for bad in (0, -3):
        with pytest.raises(InvariantError):
            model.check([buy_to_open(c)], bad, broken, mandatory=False)


# ======================================================================================================================
# 10.4 / 10.5 the zero-bid wing on a winning condor (the 15.2 fixture case)
# ======================================================================================================================


def test_zero_bid_wing_is_marked_at_zero_and_never_goes_stale() -> None:
    scenario = cf.winning_condor()
    model = BandFillModel(CONFIG)
    liq, mid, stale = model.liquidation(scenario.structure, scenario.later_chain, None)
    assert stale is False  # a worthless wing must not freeze the structure's mark (10.5): `stale_marks` stays 0
    assert liq == scenario.liquidation_cost
    hand_liq = hand_mid = 0
    for leg in scenario.structure.legs:  # the 10.5 formula, written out leg by leg
        q = cf.quote_of(scenario.later_chain, leg.contract)
        if leg.side is Side.SELL:
            hand_liq += q.ask
            hand_mid += -(-(q.bid + q.ask) // 2)
        else:
            hand_liq -= q.bid
            hand_mid -= (q.bid + q.ask) // 2
    assert (liq, mid) == (hand_liq, hand_mid)
    for wing in scenario.wings:
        wing_quote = cf.quote_of(scenario.later_chain, wing)
        assert wing_quote.bid == 0 < wing_quote.ask and not wing_quote.valid()
        assert wing_quote.usable_sell_close() and wing_quote.usable_buy()
    assert 2 * liq <= scenario.open_credit_worst  # closing costs at most half the credit: the profit target fires


def test_profit_target_and_time_exit_closes_fill_with_the_wing_sold_at_zero() -> None:
    scenario = cf.winning_condor()
    model = BandFillModel(CONFIG)
    close = cf.close_legs(scenario.structure)
    wing_occs = {wing.occ for wing in scenario.wings}
    assert model.check(close, 1, scenario.later_chain, mandatory=False) == ()  # neither exit is blocked
    for mandatory in (False, True):  # the discretionary profit-target close and the mandatory time exit
        net, leg_fills, quality = model.price(close, scenario.later_chain, mandatory=mandatory)
        sold_wings = [f for f in leg_fills if f.occ in wing_occs]
        assert len(sold_wings) == 2
        for fill in sold_wings:
            assert fill.side is Side.SELL and fill.bid == 0 < fill.ask
            assert (fill.orats, fill.worst, fill.mid) == (0, 0, 0)  # sold at 0 on all three bands, never interpolated
        assert quality == "ok"  # every leg had a usable quote: not degraded (and, in the caller, forced = False)
        assert net.orats > 0 and net.worst > 0 and net.mid > 0  # closing a credit condor is a debit
    unforced = model.price(close, scenario.later_chain, mandatory=False)[0]
    forced = model.price(close, scenario.later_chain, mandatory=True)[0]
    assert forced.worst >= unforced.worst  # the forced fill pays the penalty on top


def test_a_zero_bid_sell_leg_of_an_open_is_still_no_quote() -> None:
    scenario = cf.winning_condor()
    model = BandFillModel(CONFIG)
    wing = scenario.wings[0]
    assert model.check([sell_to_open(wing)], 1, scenario.later_chain, mandatory=False) == (NO_QUOTE,)
    assert model.check([sell_to_close(wing)], 1, scenario.later_chain, mandatory=False) == ()
    # priced anyway (paper books the fill and records what the model WOULD have rejected): the fallback, degraded
    net, legs, quality = model.price([sell_to_open(wing)], scenario.later_chain, mandatory=False)
    assert quality == "degraded" and legs[0].orats == legs[0].worst == legs[0].mid
    assert net.orats == -legs[0].orats


# ======================================================================================================================
# 10.4 forced fills
# ======================================================================================================================


def test_forced_fill_pays_the_penalty_on_every_band() -> None:
    chain, expiry = base_chain()
    long_call = at(chain, expiry, Right.CALL, 450)
    short_call = at(chain, expiry, Right.CALL, 455)
    chain = plant(chain, {long_call: (300, 308), short_call: (120, 126)})
    legs = [sell_to_close(long_call), buy_to_close(short_call)]  # closing the debit spread
    net, leg_fills, quality = BandFillModel(CONFIG).price(legs, chain, mandatory=True)
    # SPY is a penny class (tick 1). SELL: worst = bid = 300, penalty = max(1, ceil(0.10 * 8)) = 1 -> 299
    #                                BUY : worst = ask = 126, penalty = max(1, ceil(0.10 * 6)) = 1 -> 127
    assert [(f.orats, f.worst, f.mid) for f in leg_fills] == [(299, 299, 299), (127, 127, 127)]
    assert net == BandPrices(orats=-172, worst=-172, mid=-172)  # 127 - 299 on every band
    assert quality == "ok"  # both quotes were usable; only the price is worse


def test_forced_penalty_uses_the_spread_when_it_exceeds_one_tick() -> None:
    chain, expiry = base_chain()
    c = at(chain, expiry, Right.CALL, 455)
    chain = plant(chain, {c: (100, 141)})  # spread 41 -> penalty = max(1 tick, ceil(4.1)) = 5
    bought = BandFillModel(CONFIG).price([buy_to_close(c)], chain, mandatory=True)[1]
    assert (bought[0].orats, bought[0].worst, bought[0].mid) == (146, 146, 146)  # ask 141 + 5
    sold = BandFillModel(CONFIG).price([sell_to_close(c)], chain, mandatory=True)[1]
    assert (sold[0].orats, sold[0].worst, sold[0].mid) == (95, 95, 95)  # bid 100 - 5


def test_forced_fill_of_a_leg_without_a_usable_quote_is_degraded() -> None:
    chain, expiry = base_chain()
    itm_call = at(chain, expiry, Right.CALL, 445)  # intrinsic 500 cents at spot 450.00
    otm_put = at(chain, expiry, Right.PUT, 440)  # intrinsic 0
    model = BandFillModel(CONFIG)

    # SELL with no usable quote: max(intrinsic - pad, 0) = max(500 - 5, 0) = 495
    sold = model.price([sell_to_close(itm_call)], plant(chain, {itm_call: (0, 0)}), mandatory=True)
    assert [(f.orats, f.worst, f.mid) for f in sold[1]] == [(495, 495, 495)] and sold[2] == "degraded"
    worthless = model.price([sell_to_close(otm_put)], plant(chain, {otm_put: (0, 0)}), mandatory=True)
    assert [(f.orats, f.worst, f.mid) for f in worthless[1]] == [(0, 0, 0)] and worthless[2] == "degraded"  # never below 0

    # BUY with no usable quote: max(intrinsic, last mark) + pad; on a crossed row the visible max(bid, ask) stands in
    bought = model.price([buy_to_close(itm_call)], plant(chain, {itm_call: (700, 600)}), mandatory=True)
    assert [(f.orats, f.worst, f.mid) for f in bought[1]] == [(705, 705, 705)]  # max(500, 700) + 5
    assert bought[2] == "degraded"
    marked = model.price([buy_to_close(itm_call)], plant(chain, {itm_call: (0, 0)}), mandatory=True, last_marks={itm_call.occ: 900})
    assert [(f.orats, f.worst, f.mid) for f in marked[1]] == [(905, 905, 905)]  # max(500, 900) + 5
    bare = model.price([buy_to_close(itm_call)], cf.drop_contract(chain, itm_call), mandatory=True)
    assert [(f.orats, f.worst, f.mid) for f in bare[1]] == [(505, 505, 505)]  # nothing visible: intrinsic + pad
    assert bare[1][0].bid == bare[1][0].ask == 0 and bare[2] == "degraded"


def test_an_unusable_leg_is_degraded_on_the_ordinary_path_too() -> None:
    chain, expiry = base_chain()
    c = at(chain, expiry, Right.CALL, 455)
    model = BandFillModel(CONFIG)
    usable = model.price([buy_to_open(c)], plant(chain, {c: (100, 106)}), mandatory=False)
    assert usable[2] == "ok"
    unusable = model.price([buy_to_open(c)], plant(chain, {c: (0, 0)}), mandatory=False)
    assert unusable[2] == "degraded"


def test_a_forced_fill_never_improves_on_the_unforced_one() -> None:
    model, chain, structure, _ = planted_condor()
    close = cf.close_legs(structure)
    unforced = model.price(close, chain, mandatory=False)[0]
    forced = model.price(close, chain, mandatory=True)[0]
    assert unforced.mid < unforced.orats < unforced.worst  # closing the condor is a debit
    for band in Band:
        assert forced.get(band) >= unforced.worst


# ======================================================================================================================
# 10.5 marks
# ======================================================================================================================


def test_liquidation_goldens() -> None:
    model, chain, structure, legs = planted_condor()
    # liq = ask(short legs) - bid(long legs) = (158 + 146) - (60 + 55) = 304 - 115 = 189
    # mid: shorts at ceil((bid + ask) / 2) = 154 + 143 = 297 ; longs at (bid + ask) // 2 = 63 + 57 = 120 -> 177
    assert model.liquidation(structure, chain, None) == (189, 177, False)
    # a zero-bid long wing is a VALID mark of 0 (its mid is ask // 2 = 2)
    zero_bid = plant(chain, {legs["long_call"]: (0, 4)})
    assert model.liquidation(structure, zero_bid, None) == (304 - 60, 297 - (63 + 2), False)
    # marks are band-independent
    assert BandFillModel(WORST_HEADLINE).liquidation(structure, chain, None) == (189, 177, False)
    # a previous mark is irrelevant while every leg is usable
    assert model.liquidation(structure, chain, (999, 999)) == (189, 177, False)


def test_a_leg_is_unusable_only_when_missing_crossed_or_a_short_without_an_ask() -> None:
    model, chain, structure, legs = planted_condor()
    last = (1234, 1200)
    for broken in (
        cf.drop_contract(chain, legs["short_call"]),  # a missing quote
        plant(chain, {legs["short_call"]: (0, 0)}),  # a short leg with no ask
        plant(chain, {legs["short_call"]: (158, 140)}),  # crossed with a positive bid
        plant(chain, {legs["short_call"]: (140, 140)}),  # locked
        cf.drop_contract(chain, legs["long_put"]),  # a missing LONG leg is unusable too
    ):
        assert model.liquidation(structure, broken, last) == (1234, 1200, True)  # the previous values are kept
    assert model.liquidation(structure, plant(chain, {legs["long_put"]: (0, 3)}), last)[2] is False  # 0 x ask is usable


def test_the_fallback_bound_prices_unusable_legs_when_no_previous_mark_exists() -> None:
    model, chain, structure, legs = planted_condor()
    # the short 445 put is out of the money at spot 450.00 and the dropped row shows nothing: bound = max(0, 0) = 0
    assert model.liquidation(structure, cf.drop_contract(chain, legs["short_put"]), None) == (146 - 115, 143 - 120, True)

    expiry = cf.target_expiry(chain)
    short_call = at(chain, expiry, Right.CALL, 445)  # intrinsic 500 cents
    long_call = at(chain, expiry, Right.CALL, 450)  # intrinsic 0
    spread = Structure(
        kind=StructureKind.CALL_CREDIT,
        underlying=chain.underlying,
        expiry=expiry,
        last_session=chain.last_session(expiry),
        legs=(Leg(contract=short_call, side=Side.SELL), Leg(contract=long_call, side=Side.BUY)),
    )
    # a crossed short leg: bound = max(intrinsic 500, the most conservative visible price 700) = 700; the long keeps its quote
    crossed = plant(chain, {short_call: (700, 600), long_call: (300, 308)})
    assert model.liquidation(spread, crossed, None) == (700 - 300, 700 - 304, True)  # long mid = 608 // 2 = 304
    # a missing LONG leg falls back to its intrinsic (0 for the 450 call at spot 450.00)
    missing_long = cf.drop_contract(plant(chain, {short_call: (700, 706)}), long_call)
    assert model.liquidation(spread, missing_long, None) == (706, 703, True)  # 706 - 0 ; ceil(1406 / 2) - 0


def test_liquidation_refuses_a_foreign_or_empty_structure() -> None:
    model, chain, structure, _ = planted_condor()
    with pytest.raises(InvariantError):
        model.liquidation(msgspec.structs.replace(structure, underlying="QQQ"), chain, None)
    with pytest.raises(InvariantError):
        model.liquidation(msgspec.structs.replace(structure, legs=()), chain, None)
    with pytest.raises(InvariantError):
        model.liquidation(structure, chain, (1, 1.5))  # type: ignore[arg-type]


# ======================================================================================================================
# 10.7 fees
# ======================================================================================================================


def test_fee_goldens_for_a_condor_fill() -> None:
    model, chain, _, legs = planted_condor()
    open_legs = condor_open_legs(legs)
    leg_fills = model.price(open_legs, chain, mandatory=False)[1]
    fees = model.fees_micro(open_legs, 2, leg_fills)
    # contracts = 2 * 4 = 8 ; sold = 2 * 2 = 4
    # fixed = ceil((8 * (0.015 + 0.025 + 0.0003 + 0.0) + 4 * 0.00329) * 1e6) = ceil((0.3224 + 0.01316) * 1e6) = 335_560
    # headline = orats: sell notional = (153 + 142) * 100 * 2 = 59_000 cents
    # sec   = ceil(0.0000206 * 59_000 * 1e4) = ceil(12_154.0) = 12_154
    assert fees == 335_560 + 12_154 == 347_714
    assert end_of_day_fee_cents(fees) == 35  # 34.7714 cents, rounded UP at the end of the day

    # under `same_snapshot_worst` the headline band is `worst`: the sold legs are the bid, not the ORATS price
    worst_model = BandFillModel(WORST_HEADLINE)
    worst_fills = worst_model.price(open_legs, chain, mandatory=False)[1]
    worst_fees = worst_model.fees_micro(open_legs, 2, worst_fills)
    # sell notional = (150 + 140) * 100 * 2 = 58_000 -> sec = ceil(0.0000206 * 58_000 * 1e4) = 11_948
    assert worst_fees == 335_560 + 11_948 == 347_508 and end_of_day_fee_cents(worst_fees) == 35
    assert worst_fees < fees  # a lower sell notional is a lower SEC fee


def test_fee_golden_for_one_sold_leg_with_a_fractional_sec_fee() -> None:
    chain, expiry = base_chain()
    c = at(chain, expiry, Right.CALL, 455)
    planted = plant(chain, {c: (150, 158)})
    model = BandFillModel(CONFIG)
    legs = [sell_to_open(c)]
    leg_fills = model.price(legs, planted, mandatory=False)[1]
    assert leg_fills[0].orats == 152  # one leg: 158 - ceil(8 * 0.75) = 158 - 6
    # fixed = ceil((0.0403 + 0.00329) * 1e6) = 43_590 ; notional = 152 * 100 = 15_200
    # sec   = ceil(0.0000206 * 15_200 * 1e4) = ceil(3_131.2) = 3_132
    assert model.fees_micro(legs, 1, leg_fills) == 43_590 + 3_132 == 46_722
    assert end_of_day_fee_cents(46_722) == 5  # 4.6722 -> 5
    # three contracts: fixed = ceil((3 * 0.0403 + 3 * 0.00329) * 1e6) = 130_770 ;
    #                  sec = ceil(0.0000206 * 45_600 * 1e4) = ceil(9_393.6) = 9_394
    assert model.fees_micro(legs, 3, leg_fills) == 130_770 + 9_394 == 140_164
    # a BUY-only fill pays no TAF and no SEC fee
    bought = model.price([buy_to_open(c)], planted, mandatory=False)[1]
    assert model.fees_micro([buy_to_open(c)], 1, bought) == 40_300  # ceil(0.0403 * 1e6)


def test_fee_arithmetic_is_the_one_structmath_formula() -> None:
    chain, expiry = base_chain()
    c = at(chain, expiry, Right.CALL, 455)
    planted = plant(chain, {c: (150, 158)})
    model = BandFillModel(CONFIG)
    legs = [sell_to_open(c)]
    leg_fills = model.price(legs, planted, mandatory=False)[1]
    for qty in (1, 2, 3, 17):
        assert model.fees_micro(legs, qty, leg_fills) == fill_fees_micro(qty, qty, 152 * 100 * qty, FeesConfig())


def test_fees_refuse_a_misaligned_leg_list() -> None:
    chain, expiry = base_chain()
    c = at(chain, expiry, Right.CALL, 455)
    other = at(chain, expiry, Right.PUT, 440)
    planted = plant(chain, {c: (150, 158), other: (60, 66)})
    model = BandFillModel(CONFIG)
    legs = [sell_to_open(c), buy_to_open(other)]
    leg_fills = model.price(legs, planted, mandatory=False)[1]
    assert model.fees_micro(legs, 1, leg_fills) > 0
    with pytest.raises(InvariantError):
        model.fees_micro(legs, 1, leg_fills[:1])
    with pytest.raises(InvariantError):
        model.fees_micro(list(reversed(legs)), 1, leg_fills)
    with pytest.raises(InvariantError):
        model.fees_micro(legs, 0, leg_fills)


def test_end_of_day_fee_rounds_up_to_the_cent() -> None:
    assert end_of_day_fee_cents(0) == 0
    assert end_of_day_fee_cents(1) == 1  # any accrual at all costs a cent
    assert end_of_day_fee_cents(10_000) == 1
    assert end_of_day_fee_cents(10_001) == 2
    assert end_of_day_fee_cents(347_714) == 35
    with pytest.raises(InvariantError):
        end_of_day_fee_cents(-1)
    with pytest.raises(InvariantError):
        end_of_day_fee_cents(1.5)  # type: ignore[arg-type]


# ======================================================================================================================
# The model as a whole
# ======================================================================================================================


def test_band_fill_model_matches_the_fill_model_protocol() -> None:
    from jevbot.protocols import FillModel

    for name, member in vars(FillModel).items():
        if name.startswith("_") or not callable(member):
            continue
        want = [(p.name, p.kind) for p in inspect.signature(member).parameters.values()][1:]
        parameters = list(inspect.signature(getattr(BandFillModel, name)).parameters.values())[1:]
        got = [(p.name, p.kind) for p in parameters]
        assert got[: len(want)] == want, name
        for extra in parameters[len(want) :]:  # anything beyond the Protocol is optional (`price(last_marks=...)`)
            assert extra.default is not inspect.Parameter.empty, f"{name}.{extra.name}"


def test_the_model_is_pure() -> None:
    model, chain, structure, legs = planted_condor()
    open_legs = condor_open_legs(legs)
    first = model.price(open_legs, chain, mandatory=False)
    marks = model.liquidation(structure, chain, None)
    assert model.price(open_legs, chain, mandatory=False) == first  # the same inputs, the same bytes
    assert model.liquidation(structure, chain, None) == marks
    assert model.check(open_legs, 1, chain, mandatory=False) == ()
