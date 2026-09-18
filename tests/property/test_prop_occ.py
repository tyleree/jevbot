"""Property: parse_occ(format_occ(c)) == c for generated contracts, incl. Saturday-dated expiries (seeded numpy generators)."""

from datetime import date, timedelta

import numpy as np
import pytest

from jevbot.occ import format_occ, is_occ, parse_occ
from jevbot.types import OptionContract, Right

_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_FIRST = date(2000, 1, 1)
_DAYS = (date(2099, 12, 31) - _FIRST).days + 1  # every calendar date an OCC yymmdd can express
_STRIKE_EDGES = (1, 5, 999, 1000, 450500, 9999999, 10000000, 99999999)


def _contract(rng: np.random.Generator) -> OptionContract:
    root = "".join(rng.choice(_LETTERS, size=int(rng.integers(1, 7))))
    expiry = _FIRST + timedelta(days=int(rng.integers(0, _DAYS)))
    right = Right.CALL if rng.random() < 0.5 else Right.PUT
    strike = int(rng.choice(_STRIKE_EDGES)) if rng.random() < 0.2 else int(rng.integers(1, 10**8))
    return OptionContract(underlying=root, expiry=expiry, right=right, strike_milli=strike)


@pytest.mark.parametrize("seed", range(8))
def test_round_trip(seed: int) -> None:
    rng = np.random.default_rng(seed)
    weekdays: set[int] = set()
    for _ in range(2500):
        c = _contract(rng)
        weekdays.add(c.expiry.weekday())
        symbol = format_occ(c)
        assert symbol == c.occ
        assert len(symbol) == len(c.underlying) + 15 and symbol.isascii()
        assert parse_occ(symbol) == c  # contract -> symbol -> contract
        assert format_occ(parse_occ(symbol)) == symbol  # symbol -> contract -> symbol
        assert is_occ(symbol)
        osi = f"{c.underlying:<6}{symbol[len(c.underlying) :]}"  # the 21-character space-padded OSI form
        assert len(osi) == 21
        assert parse_occ(osi) == c
    assert weekdays == set(range(7))  # expiries are identity only: Saturdays and Sundays round-trip like any other date


def test_every_saturday_monthly_before_the_2015_change_round_trips() -> None:
    # standard monthlies listed before February 2015 expire on the SATURDAY after the third Friday
    n = 0
    for year in range(2005, 2015):
        for month in range(1, 13):
            first = date(year, month, 1)
            third_friday = first + timedelta(days=(4 - first.weekday()) % 7 + 14)
            saturday = third_friday + timedelta(days=1)
            assert third_friday.weekday() == 4 and saturday.weekday() == 5
            for right in (Right.CALL, Right.PUT):
                c = OptionContract(underlying="SPY", expiry=saturday, right=right, strike_milli=100000 + 500 * month)
                assert parse_occ(format_occ(c)) == c
                assert parse_occ(format_occ(c)).expiry.weekday() == 5
                n += 1
    assert n == 240


@pytest.mark.parametrize("seed", range(4))
def test_distinct_contracts_have_distinct_symbols(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    contracts = {_contract(rng) for _ in range(3000)}
    assert len({format_occ(c) for c in contracts}) == len(contracts)


@pytest.mark.parametrize("seed", range(4))
def test_single_character_corruption_never_parses_to_the_same_contract(seed: int) -> None:
    rng = np.random.default_rng(200 + seed)
    alphabet = list("ABCXYZcp0123456789 .-")
    for _ in range(1500):
        c = _contract(rng)
        symbol = format_occ(c)
        i = int(rng.integers(0, len(symbol)))
        replacement = str(rng.choice(alphabet))
        if replacement == symbol[i]:
            continue
        corrupted = symbol[:i] + replacement + symbol[i + 1 :]
        if is_occ(corrupted):
            assert parse_occ(corrupted) != c  # a different, well-formed symbol is a DIFFERENT contract
        else:
            with pytest.raises(ValueError):
                parse_occ(corrupted)
