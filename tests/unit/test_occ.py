"""occ.py (DESIGN.md 2.2): OCC symbol format / parse. The expiry is contract identity only - it may be a Saturday."""

from datetime import UTC, date, datetime

import msgspec
import pytest

from jevbot.occ import format_occ, is_occ, parse_occ
from jevbot.types import OptionContract, Right


def _c(root: str, expiry: date, right: Right, strike_milli: int) -> OptionContract:
    return OptionContract(underlying=root, expiry=expiry, right=right, strike_milli=strike_milli)


@pytest.mark.parametrize(
    ("contract", "symbol"),
    [
        (_c("SPY", date(2026, 10, 16), Right.CALL, 600000), "SPY261016C00600000"),  # the example of section 2.2
        (_c("SPY", date(2026, 10, 16), Right.PUT, 450500), "SPY261016P00450500"),  # 450.5 -> 450500 (Conventions)
        (_c("QQQ", date(2012, 3, 17), Right.PUT, 65000), "QQQ120317P00065000"),  # a SATURDAY-dated monthly (pre-2015)
        (_c("IWM", date(2014, 4, 19), Right.CALL, 112500), "IWM140419C00112500"),  # Saturday after Good Friday
        (_c("A", date(2000, 1, 1), Right.CALL, 1), "A000101C00000001"),  # smallest root, year, strike
        (_c("GOOGLX", date(2099, 12, 31), Right.PUT, 99999999), "GOOGLX991231P99999999"),  # largest root, year, strike
        (_c("SPY", date(2028, 2, 29), Right.CALL, 1234567), "SPY280229C01234567"),  # a leap day
    ],
)
def test_format_and_parse_goldens(contract: OptionContract, symbol: str) -> None:
    assert format_occ(contract) == symbol
    assert contract.occ == symbol  # the struct property is the same formula (types.py cannot import occ.py)
    assert parse_occ(symbol) == contract
    assert is_occ(symbol)


def test_saturday_dated_expiries_round_trip_without_any_calendar() -> None:
    saturday = date(2012, 3, 17)
    assert saturday.weekday() == 5
    contract = _c("SPY", saturday, Right.CALL, 140000)
    parsed = parse_occ(format_occ(contract))
    assert parsed == contract and parsed.expiry == saturday  # identity only: never moved to the Friday


@pytest.mark.parametrize(
    ("padded", "unpadded"),
    [
        ("SPY   261016C00600000", "SPY261016C00600000"),  # the 21-character OSI form
        ("A     000101C00000001", "A000101C00000001"),
        ("GOOGLX991231P99999999", "GOOGLX991231P99999999"),  # a six-letter root has no padding
        ("QQQ 120317P00065000", "QQQ120317P00065000"),  # the documented regex accepts any number of pad spaces
    ],
)
def test_padded_osi_form_is_accepted(padded: str, unpadded: str) -> None:
    assert parse_occ(padded) == parse_occ(unpadded)
    assert format_occ(parse_occ(padded)) == unpadded  # we always WRITE the unpadded form
    assert is_occ(padded)


def test_the_osi_form_is_21_characters_with_the_root_padded_to_six() -> None:
    for root in ("A", "SPY", "GOOGL", "GOOGLX"):
        osi = f"{root:<6}261016C00600000"
        assert len(osi) == 21
        assert parse_occ(osi) == _c(root, date(2026, 10, 16), Right.CALL, 600000)


@pytest.mark.parametrize(
    "symbol",
    [
        "SPY1  261016C00600000",  # adjusted root (digit in the root), padded
        "SPY1261016C00600000",  # adjusted root, unpadded (19 chars would shift every field)
        "BRK.B261016C00600000",  # punctuation in the root
        "BRKB7 261016C00600000",
        "2SPY261016C00600000",
    ],
)
def test_adjusted_roots_are_rejected(symbol: str) -> None:
    with pytest.raises(ValueError):
        parse_occ(symbol)
    assert not is_occ(symbol)


@pytest.mark.parametrize(
    "symbol",
    [
        "",
        "SPY",  # an equity ticker (an assigned position): not an option symbol
        "spy261016C00600000",  # lower-case root
        "SPY261016c00600000",  # lower-case right
        "SPY261016X00600000",  # unknown right
        "SPY261016C0060000",  # 7-digit strike
        "SPY261016C006000000",  # 9-digit strike
        "SPY26101C00600000",  # 5-digit date
        "SPY261016C00000000",  # strike 0
        "SPY261316C00600000",  # month 13
        "SPY260230C00600000",  # 30 February
        "SPY270229C00600000",  # 29 February in a non-leap year
        "SPY260931C00600000",  # 31 September
        "SPY260000C00600000",  # month 0 / day 0
        "TOOLONGX261016C00600000",  # 8-letter root
        " SPY261016C00600000",  # leading whitespace
        "SPY261016C00600000 ",  # trailing whitespace
        "SPY261016C00600000\n",  # trailing newline (a `$` anchor would let this through)
        "SPY\t261016C00600000",  # pad characters are spaces only
        "SPY\n261016C00600000",
        "SPY261016C 00600000",
        "SPY261016C00600000X",
        "SPY261016C-0600000",
        "SPY261016C+0600000",
        "SPY261016C006000.0",
        "SPY２６1016C00600000",  # full-width digits: `\d` and int() would accept them
        "SPY٢٦1016C00600000",  # Arabic-Indic digits
        "ＳＰＹ261016C00600000",  # full-width letters
        "O:SPY261016C00600000",  # vendor prefix
    ],
)
def test_malformed_symbols_are_rejected(symbol: str) -> None:
    with pytest.raises(ValueError):
        parse_occ(symbol)
    assert not is_occ(symbol)


def test_non_string_input() -> None:
    for bad in (None, 123, b"SPY261016C00600000", ["SPY261016C00600000"]):
        with pytest.raises(ValueError):
            parse_occ(bad)  # type: ignore[arg-type]
        assert not is_occ(bad)  # type: ignore[arg-type]


def test_century_is_always_20yy() -> None:
    assert parse_occ("SPY000101C00100000").expiry == date(2000, 1, 1)
    assert parse_occ("SPY691231C00100000").expiry == date(2069, 12, 31)  # no strptime-style pivot to 1969
    assert parse_occ("SPY991231C00100000").expiry == date(2099, 12, 31)


def test_format_refuses_contracts_that_cannot_round_trip() -> None:
    with pytest.raises(ValueError, match="year"):
        format_occ(_c("SPY", date(1999, 12, 31), Right.CALL, 100000))  # yymmdd "991231" would parse back as 2099
    with pytest.raises(ValueError, match="year"):
        format_occ(_c("SPY", date(2100, 1, 1), Right.CALL, 100000))
    with pytest.raises(ValueError, match="date"):
        # a datetime IS a date and would format, but parse_occ(format_occ(c)) == c could never hold
        format_occ(_c("SPY", datetime(2026, 10, 16, 20, 0, tzinfo=UTC), Right.CALL, 100000))


def test_format_rechecks_identity_on_a_tampered_struct() -> None:
    # OptionContract validates in __post_init__; format_occ does not rely on it (a symbol is what reaches the broker)
    tampered = _c("SPY", date(2026, 10, 16), Right.CALL, 600000)
    msgspec.structs.force_setattr(tampered, "underlying", "SPY1")
    with pytest.raises(ValueError, match="root"):
        format_occ(tampered)
    for strike in (0, -5, 10**8, 600000.0, True):
        tampered = _c("SPY", date(2026, 10, 16), Right.CALL, 600000)
        msgspec.structs.force_setattr(tampered, "strike_milli", strike)
        with pytest.raises(ValueError, match="strike_milli"):
            format_occ(tampered)


def test_contract_identity_invariants_come_from_the_struct() -> None:
    # OptionContract itself refuses what would corrupt a symbol, so format_occ can never see it
    with pytest.raises(ValueError):
        _c("SPY1", date(2026, 10, 16), Right.CALL, 100000)
    with pytest.raises(ValueError):
        _c("SPY", date(2026, 10, 16), Right.CALL, 0)
    with pytest.raises(ValueError):
        _c("SPY", date(2026, 10, 16), Right.CALL, 10**8)


def test_parsed_contracts_sort_like_their_fields() -> None:
    symbols = ["SPY261016P00450000", "SPY261016C00600000", "SPY260918C00600000", "QQQ261016C00500000"]
    parsed = sorted(parse_occ(s) for s in symbols)
    assert [c.occ for c in parsed] == ["QQQ261016C00500000", "SPY260918C00600000", "SPY261016C00600000", "SPY261016P00450000"]
