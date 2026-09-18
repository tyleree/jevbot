"""Property tests for money.py (seeded numpy generators): ceiling division, tick rounding directions, the limit-sign bound."""

from fractions import Fraction
from math import ceil

import numpy as np
import pytest

from jevbot.errors import InvariantError
from jevbot.money import assert_limit_sign, cdiv, round_net, tick_cents
from jevbot.types import SHORT_PREMIUM, OrderPurpose, StructureKind

TICKS = (1, 5, 10)
MULTI_LEG = [k for k in StructureKind if k not in (StructureKind.LONG_CALL, StructureKind.LONG_PUT)]


@pytest.mark.parametrize("seed", range(5))
def test_cdiv_is_the_exact_ceiling(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(4000):
        a = int(rng.integers(0, 10**12)) * int(rng.integers(0, 10**6))  # beyond 2**53: floats could not do this
        b = int(rng.integers(1, 10**9))
        q = cdiv(a, b)
        assert q == ceil(Fraction(a, b))
        assert b * (q - 1) < a <= b * q or (a == 0 and q == 0)
        assert cdiv(q * b, b) == q  # exact multiples are not rounded


@pytest.mark.parametrize("seed", range(5))
def test_round_net_properties(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    for _ in range(6000):
        net = int(rng.integers(-5000, 5001))
        tick = int(rng.choice(TICKS))
        passive = round_net(net, tick, aggressive=False)
        aggressive = round_net(net, tick, aggressive=True)
        sub_tick_debit = 0 < net < tick  # the one band where passive concedes (less than one tick) to stay a fillable debit
        sub_tick_credit = -tick < net < 0  # its mirror image: aggressive crosses zero to +tick to stay marketable

        # aggressive is marketable for EVERY input; passive never pays more / accepts less than computed, except that a
        # sub-tick debit becomes the one-tick debit (the sub-tick rule of money.round_net)
        assert net <= aggressive
        assert passive <= aggressive
        if sub_tick_debit:
            assert passive == tick == aggressive
        else:
            assert passive <= net
        # on the grid, never zero, always an int number of cents (<= 2 decimals in dollars)
        for value in (passive, aggressive):
            assert type(value) is int and value % tick == 0 and value != 0
        # the sign: passive keeps the sign of a non-zero net, and turns exactly 0 into a one-tick credit demand
        assert (passive > 0) == (net > 0)
        # never further than one tick from the computed price, except across the illegal zero (one extra tick)
        assert abs(net - passive) < tick or (net == 0 and passive == -tick)
        assert aggressive - net < (tick if aggressive != tick or net > 0 else 2 * tick)
        # a price on the grid is left alone (unless it is the illegal zero)
        if net % tick == 0 and net != 0:
            assert passive == net == aggressive
        elif sub_tick_debit:
            assert aggressive == passive
        else:
            assert aggressive - passive in (tick, 2 * tick)
        # idempotent
        assert round_net(passive, tick, aggressive=False) == passive
        assert round_net(aggressive, tick, aggressive=True) == aggressive
        # mirror symmetry of the two modes: flipping the sign of the price swaps them - everywhere but in the sub-tick band, where
        # the rule is deliberately asymmetric: passive keeps a sub-tick debit a debit (+tick) while aggressive takes a sub-tick
        # credit across zero (+tick), both landing on the same one-tick debit
        if sub_tick_debit:
            assert round_net(-net, tick, aggressive=True) == tick == passive  # the mirrored sub-tick credit, closed aggressively
        else:
            assert round_net(-net, tick, aggressive=True) == -passive
        if sub_tick_credit:
            assert round_net(-net, tick, aggressive=False) == tick == aggressive  # the mirrored sub-tick debit, posted passively
        else:
            assert round_net(-net, tick, aggressive=False) == -aggressive
        # magnitudes, as the contract words it
        if net > 0 and not sub_tick_debit:
            assert abs(passive) <= abs(net) <= abs(aggressive)  # debit: passive DOWN in magnitude, aggressive UP
        if net < 0 and aggressive < 0:
            assert abs(passive) >= abs(net) >= abs(aggressive)  # credit: passive UP in magnitude, aggressive DOWN


@pytest.mark.parametrize("seed", range(3))
def test_tick_table(seed: int) -> None:
    rng = np.random.default_rng(200 + seed)
    penny = ("SPY", "QQQ", "IWM")
    for _ in range(3000):
        px = int(rng.integers(0, 20000))
        assert tick_cents(str(rng.choice(penny)), px, penny) == 1
        tick = tick_cents("XLF", px, penny)
        assert tick == (5 if px < 300 else 10)
        # a price rounded onto its own tick grid keeps that tick class unless it crosses the 3.00 break upwards
        rounded = round_net(max(px, 1), tick, aggressive=False)
        assert rounded % tick == 0


@pytest.mark.parametrize("seed", range(3))
def test_limit_sign_bound_for_multi_leg_closes(seed: int) -> None:
    rng = np.random.default_rng(300 + seed)
    for _ in range(3000):
        kind = MULTI_LEG[int(rng.integers(0, len(MULTI_LEG)))]
        purpose = OrderPurpose.CLOSE if rng.random() < 0.5 else OrderPurpose.KILL
        width = int(rng.integers(1, 3000))
        pad = int(rng.integers(0, 500))
        limit = int(rng.integers(-4000, 4001))
        if abs(limit) <= width + pad:
            assert_limit_sign(purpose, kind, limit, width=width, pad=pad)
        else:
            with pytest.raises(InvariantError):
                assert_limit_sign(purpose, kind, limit, width=width, pad=pad)


@pytest.mark.parametrize("seed", range(3))
def test_open_sign_rule_for_every_kind(seed: int) -> None:
    rng = np.random.default_rng(400 + seed)
    kinds = list(StructureKind)
    for _ in range(3000):
        kind = kinds[int(rng.integers(0, len(kinds)))]
        limit = int(rng.integers(-3000, 3001))
        width = 0 if kind in (StructureKind.LONG_CALL, StructureKind.LONG_PUT) else int(rng.integers(1, 3000))
        ok = limit < 0 if kind in SHORT_PREMIUM else limit > 0
        if ok:
            assert_limit_sign(OrderPurpose.OPEN, kind, limit, width=width, pad=0)
        else:
            with pytest.raises(InvariantError):
                assert_limit_sign(OrderPurpose.OPEN, kind, limit, width=width, pad=0)
