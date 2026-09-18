"""money.py (DESIGN.md 3.7): cdiv, tick tables, adverse rounding of net prices, the limit-sign assertion.

Sign convention (Conventions): positive = net debit (we pay), negative = net credit (we receive).
"""

import inspect
from decimal import Decimal
from typing import Any

import pytest

from jevbot.errors import InvariantError
from jevbot.money import assert_limit_sign, cdiv, round_net, tick_cents
from jevbot.types import SHORT_PREMIUM, OrderPurpose, StructureKind

OPEN, CLOSE, KILL = OrderPurpose.OPEN, OrderPurpose.CLOSE, OrderPurpose.KILL
PENNY = ("SPY", "QQQ", "IWM")

CREDIT_KINDS = (StructureKind.CALL_CREDIT, StructureKind.PUT_CREDIT, StructureKind.IRON_CONDOR)
DEBIT_VERTICALS = (StructureKind.CALL_DEBIT, StructureKind.PUT_DEBIT)
SINGLES = (StructureKind.LONG_CALL, StructureKind.LONG_PUT)
MULTI_LEG = CREDIT_KINDS + DEBIT_VERTICALS

# ======================================================================================================================
# cdiv
# ======================================================================================================================


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (0, 1, 0),
        (0, 7, 0),
        (1, 1, 1),
        (1, 2, 1),
        (2, 2, 1),
        (3, 2, 2),
        (7, 7, 1),
        (8, 7, 2),
        (37500, 10000, 4),  # 10.3: a 5-cent spread at 75% of the way -> 3.75 -> 4 cents (against us)
        (26400, 10000, 3),  # 4-cent spread, two legs (66%): 2.64 -> 3
        (30000, 10000, 3),  # exact: no rounding
        (123456, 10_000, 13),  # 10.7: fee micro-dollars to cents, 12.3456 -> 13
        (10**30 + 1, 10**15, 10**15 + 1),  # exact integer arithmetic, no float round-off
    ],
)
def test_cdiv(a: int, b: int, expected: int) -> None:
    assert cdiv(a, b) == expected


def test_cdiv_domain() -> None:
    for a, b in ((-1, 2), (1, 0), (1, -2), (0, 0)):
        with pytest.raises(ValueError):
            cdiv(a, b)
    not_ints: list[tuple[Any, Any]] = [(1.0, 2), (1, 2.0), (True, 2), ("1", 2)]
    for bad_a, bad_b in not_ints:
        with pytest.raises(TypeError):
            cdiv(bad_a, bad_b)


# ======================================================================================================================
# tick_cents
# ======================================================================================================================


@pytest.mark.parametrize("underlying", PENNY)
@pytest.mark.parametrize("px", [0, 1, 299, 300, 301, 5000])
def test_penny_classes_tick_one_cent_at_any_price(underlying: str, px: int) -> None:
    assert tick_cents(underlying, px, PENNY) == 1


@pytest.mark.parametrize(("px", "tick"), [(0, 5), (1, 5), (295, 5), (299, 5), (300, 10), (301, 10), (12345, 10)])
def test_non_penny_classes_tick_a_nickel_below_three_dollars_and_a_dime_from_there(px: int, tick: int) -> None:
    assert tick_cents("XLE", px, PENNY) == tick
    assert tick_cents("XLE", -px, PENNY) == tick  # a signed single-leg price has the tick of its magnitude
    assert tick_cents("SPY", px, ()) == tick  # penny status comes from the configured list only
    assert tick_cents("SPY", px, ["QQQ"]) == tick


def test_tick_cents_argument_checks() -> None:
    with pytest.raises(TypeError):
        tick_cents("SPY", 100, "SPY")  # a bare string would make "S" in "SPY" a penny class
    with pytest.raises(TypeError):
        tick_cents("SPY", 1.5, PENNY)  # type: ignore[arg-type]
    assert list(inspect.signature(tick_cents).parameters) == ["underlying", "px_cents", "penny_all"]


# ======================================================================================================================
# round_net
# ======================================================================================================================


@pytest.mark.parametrize(
    ("net", "tick", "passive", "aggressive"),
    [
        # debits: passive pays LESS (down in magnitude), aggressive pays MORE
        (123, 5, 120, 125),
        (123, 10, 120, 130),
        (120, 5, 120, 120),  # on the grid: unchanged in both modes
        (301, 10, 300, 310),
        (7, 1, 7, 7),
        # sub-tick debits (0 < net < tick): the floor would be 0 -> the smallest DEBIT on the grid in BOTH modes (the sign of the
        # net is kept: passive pays < 1 tick more than computed rather than demanding a credit that can never fill)
        (4, 5, 5, 5),
        (1, 5, 5, 5),
        (3, 10, 10, 10),
        (9, 10, 10, 10),
        # credits: passive demands MORE (up in magnitude), aggressive accepts LESS
        (-123, 5, -125, -120),
        (-123, 10, -130, -120),
        (-125, 5, -125, -125),
        (-7, 1, -7, -7),
        # sub-tick credits (-tick < net < 0): passive floors to -tick (one tick better for us); the aggressive ceiling would be 0
        # -> +tick: a mandatory exit stays marketable (>= net) even if that means paying one tick
        (-4, 5, -5, 5),
        (-1, 5, -5, 5),
        (-9, 10, -10, 10),
        # exactly 0 is illegal -> one tick in the mode's direction: passive demands a tick, aggressive pays one
        (0, 1, -1, 1),
        (0, 5, -5, 5),
        (0, 10, -10, 10),
    ],
)
def test_round_net_goldens(net: int, tick: int, passive: int, aggressive: int) -> None:
    assert round_net(net, tick, aggressive=False) == passive
    assert round_net(net, tick, aggressive=True) == aggressive


def test_round_net_directions_in_words() -> None:
    # passive (entries, discretionary exits): never pay more / accept less than computed
    assert round_net(123, 5, aggressive=False) <= 123  # a debit is rounded DOWN in magnitude
    assert abs(round_net(-123, 5, aggressive=False)) >= 123  # a credit is rounded UP in magnitude
    # aggressive (mandatory exits only): the inverse
    assert round_net(123, 5, aggressive=True) >= 123
    assert abs(round_net(-123, 5, aggressive=True)) <= 123


def test_sub_tick_rule_a_passive_net_keeps_its_sign_and_an_aggressive_net_stays_marketable() -> None:
    """Regression for the frozen sub-tick rule (3.7 says only "0 is illegal -> +/- 1 tick").

    Non-penny class (tick 5): a discretionary CLOSE whose natural is a 4c debit must be posted as a fillable 5c debit, not as
    a 5c credit demand that no counterparty can hit (it would hang until the mandatory L-3 exit); a mandatory CLOSE that would
    be sold for 3c must stay marketable (pay 5c) rather than demand 5c."""
    for tick in (5, 10):
        for net in range(1, tick):
            passive, aggressive = round_net(net, tick, aggressive=False), round_net(net, tick, aggressive=True)
            assert passive == aggressive == tick  # a sub-tick debit is the one-tick debit in both modes
            assert passive > 0, "passive never flips the sign of a non-zero net"
            assert passive - net < tick, "the only passive concession there is: less than one tick"
            assert round_net(-net, tick, aggressive=False) == -tick  # a sub-tick credit: passive floors (better for us)
            assert round_net(-net, tick, aggressive=True) == tick >= -net  # aggressive: >= net, crosses zero, marketable
    # penny classes: no sub-tick band exists besides 0 itself
    assert [round_net(n, 1, aggressive=False) for n in (-2, -1, 0, 1, 2)] == [-2, -1, -1, 1, 2]
    assert [round_net(n, 1, aggressive=True) for n in (-2, -1, 0, 1, 2)] == [-2, -1, 1, 1, 2]
    # the rounding alone can never turn an OPEN limit into a sign violation (check 19 must not raise on a data condition)
    for kind in DEBIT_VERTICALS + SINGLES:
        for net in (1, 4, 6, 123):
            assert_limit_sign(OPEN, kind, round_net(net, 5, aggressive=False), width=0 if kind in SINGLES else 500, pad=0)
    for kind in CREDIT_KINDS:
        for net in (-1, -4, -6, -123, 0):
            assert_limit_sign(OPEN, kind, round_net(net, 5, aggressive=False), width=500, pad=0)
    # a sub-tick natural CLOSE in a nickel class: limit_natural (passive, section 8) is a fillable debit, and the mandatory
    # aggressive price never sits below the computed net
    natural = round_net(4, 5, aggressive=False)
    assert natural == 5 and natural >= 4  # marketable against a 4c ask-side cost
    assert_limit_sign(CLOSE, StructureKind.PUT_CREDIT, natural, width=500, pad=0)
    assert round_net(-3, 5, aggressive=True) == 5 >= -3
    assert_limit_sign(CLOSE, StructureKind.PUT_CREDIT, round_net(-3, 5, aggressive=True), width=500, pad=75)


def test_round_net_never_produces_a_third_decimal() -> None:
    for net in range(-1200, 1201, 7):
        for tick in (1, 5, 10):
            for aggressive in (False, True):
                cents = round_net(net, tick, aggressive=aggressive)
                assert isinstance(cents, int) and not isinstance(cents, bool)
                dollars = Decimal(cents) / Decimal(100)
                assert dollars == dollars.quantize(Decimal("0.01"))
                assert cents % tick == 0 and cents != 0


def test_round_net_argument_checks() -> None:
    assert inspect.signature(round_net).parameters["aggressive"].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        round_net(100, 5)  # type: ignore[call-arg]  # the mode is never implicit
    for tick in (0, -5):
        with pytest.raises(ValueError):
            round_net(100, tick, aggressive=False)
    not_ints: list[tuple[Any, Any]] = [(100.0, 5), (100, 5.0), (True, 5)]
    for bad_net, bad_tick in not_ints:
        with pytest.raises(TypeError):
            round_net(bad_net, bad_tick, aggressive=False)


# ======================================================================================================================
# assert_limit_sign(purpose, kind, limit, *, width, pad)
# ======================================================================================================================


def test_the_condor_sign_trap() -> None:
    # critique corr. 6: an iron condor OPENS for a credit; a positive limit would be a debit order for a credit structure
    assert_limit_sign(OPEN, StructureKind.IRON_CONDOR, -140, width=500, pad=0)
    with pytest.raises(InvariantError, match="credit"):
        assert_limit_sign(OPEN, StructureKind.IRON_CONDOR, 140, width=500, pad=0)
    with pytest.raises(InvariantError):
        assert_limit_sign(OPEN, StructureKind.IRON_CONDOR, 0, width=500, pad=0)


@pytest.mark.parametrize("kind", CREDIT_KINDS)
def test_open_credit_structures_need_a_negative_limit(kind: StructureKind) -> None:
    assert kind in SHORT_PREMIUM
    assert_limit_sign(OPEN, kind, -1, width=500, pad=0)
    for limit in (0, 1, 95):
        with pytest.raises(InvariantError):
            assert_limit_sign(OPEN, kind, limit, width=500, pad=0)


@pytest.mark.parametrize("kind", DEBIT_VERTICALS + SINGLES)
def test_open_debit_structures_and_single_legs_need_a_positive_limit(kind: StructureKind) -> None:
    width = 0 if kind in SINGLES else 500
    assert_limit_sign(OPEN, kind, 1, width=width, pad=0)
    assert_limit_sign(OPEN, kind, 2500, width=width, pad=0)  # OPEN asserts the sign only
    for limit in (0, -1, -95):
        with pytest.raises(InvariantError, match="debit"):
            assert_limit_sign(OPEN, kind, limit, width=width, pad=0)


@pytest.mark.parametrize("purpose", [CLOSE, KILL])
@pytest.mark.parametrize("kind", MULTI_LEG)
def test_multi_leg_close_within_and_beyond_width_plus_pad(purpose: OrderPurpose, kind: StructureKind) -> None:
    width, pad = 500, 75  # pad = ceil(kill.cushion_max_frac_width 0.15 * 500)
    # a close may legitimately cross zero: both signs pass, only |limit| <= width + pad is asserted
    for limit in (575, 500, 120, 1, 0, -1, -120, -575):
        assert_limit_sign(purpose, kind, limit, width=width, pad=pad)
    for limit in (576, -576, 5000, -5000):
        with pytest.raises(InvariantError, match="width"):
            assert_limit_sign(purpose, kind, limit, width=width, pad=pad)
    # discretionary closes pass pad = 0
    assert_limit_sign(purpose, kind, 500, width=width, pad=0)
    assert_limit_sign(purpose, kind, -500, width=width, pad=0)
    with pytest.raises(InvariantError):
        assert_limit_sign(purpose, kind, 501, width=width, pad=0)


@pytest.mark.parametrize("purpose", [CLOSE, KILL])
@pytest.mark.parametrize("kind", SINGLES)
def test_a_single_leg_close_far_above_any_pad_passes(purpose: OrderPurpose, kind: StructureKind) -> None:
    # CON-13: a long option bought for 3.00 and now worth 25.00 is SOLD to close at -2500; width is 0, so a width bound would
    # refuse every single-leg close. The bound is skipped; only sign / non-zero are asserted.
    assert_limit_sign(purpose, kind, -2500, width=0, pad=0)
    assert_limit_sign(purpose, kind, -1, width=0, pad=0)
    assert_limit_sign(purpose, kind, -2500, width=0, pad=75)


@pytest.mark.parametrize("purpose", [CLOSE, KILL])
@pytest.mark.parametrize("kind", SINGLES)
def test_single_leg_close_sign_checks(purpose: OrderPurpose, kind: StructureKind) -> None:
    with pytest.raises(InvariantError, match="0 is illegal"):
        assert_limit_sign(purpose, kind, 0, width=0, pad=0)
    with pytest.raises(InvariantError, match="SELLS"):  # selling a long leg to close: limit < 0
        assert_limit_sign(purpose, kind, 2500, width=0, pad=0)


def test_per_leg_kill_fallback_orders() -> None:
    # kind None = one leg of a broken-up structure: a short leg is bought back (limit > 0), a long leg is sold (limit < 0)
    assert_limit_sign(KILL, None, 830, width=0, pad=0)
    assert_limit_sign(KILL, None, -2500, width=0, pad=0)
    assert_limit_sign(KILL, None, 830, width=500, pad=75)  # the width bound is skipped whatever the caller passes
    with pytest.raises(InvariantError, match="0 is illegal"):
        assert_limit_sign(KILL, None, 0, width=0, pad=0)
    with pytest.raises(InvariantError, match="OPEN"):
        assert_limit_sign(OPEN, None, 100, width=0, pad=0)  # an OPEN always has a structure kind


@pytest.mark.parametrize("purpose", [OPEN, CLOSE, KILL])
@pytest.mark.parametrize("kind", [None, *StructureKind])
def test_market_orders_are_skipped(purpose: OrderPurpose, kind: StructureKind | None) -> None:
    # limit None = market order (the equity flatten, the last-resort kill leg): not checked
    assert_limit_sign(purpose, kind, None, width=0, pad=0)
    assert_limit_sign(purpose, kind, None, width=500, pad=75)


def test_bug_class_arguments_raise_invariant_error() -> None:
    with pytest.raises(InvariantError, match="positive width"):
        assert_limit_sign(CLOSE, StructureKind.PUT_CREDIT, 100, width=0, pad=0)  # a vertical without a width is a bug
    for width, pad in ((-1, 0), (500, -1), (500.0, 0), (500, 7.5), (True, 0)):
        with pytest.raises(InvariantError):
            assert_limit_sign(CLOSE, StructureKind.PUT_CREDIT, 100, width=width, pad=pad)  # type: ignore[arg-type]
    with pytest.raises(InvariantError):
        assert_limit_sign(CLOSE, StructureKind.PUT_CREDIT, 100.0, width=500, pad=0)  # type: ignore[arg-type]
    with pytest.raises(InvariantError):
        assert_limit_sign("close", StructureKind.PUT_CREDIT, 100, width=500, pad=0)  # type: ignore[arg-type]
    with pytest.raises(InvariantError):
        assert_limit_sign(CLOSE, "put_credit_spread", 100, width=500, pad=0)  # type: ignore[arg-type]


def test_signature_is_the_contract() -> None:
    params = inspect.signature(assert_limit_sign).parameters
    assert list(params) == ["purpose", "kind", "limit", "width", "pad"]
    assert params["width"].kind is inspect.Parameter.KEYWORD_ONLY and params["pad"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["width"].default is inspect.Parameter.empty and params["pad"].default is inspect.Parameter.empty
