"""Bucket tables (DESIGN.md 5.5): the exact thresholds and label strings, as data, plus one generic `bucketize()`.

Every label is `"<code>: <meaning>"`. The code (the text before the first colon) is the stable machine token that
`vocab.py`, MockJev, the rules cross-checks and `EntryContext.entry_codes` read; the meaning is prose for Jev. A `None`
feature renders as the bare string `vocab.UNAVAILABLE` (`"unavailable"`).

A table is a tuple of `Cut(upper, closed, label)` in ASCENDING value order plus the label of everything above the last
cut. `bucketize` walks the cuts in order and returns the first label whose bound the value satisfies (`x < upper`, or
`x <= upper` when the cut is `closed`). Most 5.5 intervals are half-open `[lo, hi)` - `closed=False` - but several are
not (`PCTL3`'s `[30,70]`, `CHANGE5`'s `[-0.05,0.05]`, `TERM4`'s `[0.95,1.05]`, `NEAR3`, `RV_CHANGE`, `SIGMA5`, `HOLD`,
`DTE4`, `HELD4`, `STREAK`, `MOVE_SINCE_ENTRY`), so a cut carries its inclusivity flag: the spec's `list[tuple[float, str]]`
shape cannot spell `[30,70]` and `(70,100]` apart. Pure data and pure functions: no IO, no clock, no RNG.

`bucket_spec_hash` (a constant, also callable as `bucket_spec_hash()`) is `sha256(dumps_sorted(tables))` of 5.5. The
thresholds are floats and `canon.dumps_sorted` refuses floats (INV-24), so the hashed material spells every bound with
`repr()` - the shortest exact decimal - which is what the operator wrote and is stable across runs and machines.

Two 5.5 tables have two wordings each and are therefore two tables here: `PCTL5` (`percentile` for the two percentile
fields, `range` for `vol_surface.iv_rank_1y`) and `TERM4` (`3-month` for `market.vol_term_structure`, `90-day` for
`vol_surface.term_structure`). `TREND_DIR` (an ordered rule table, not a threshold table) and `PNL` (two-sided, relative
to the structure's own maximum profit / loss) have their own functions.
"""

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, NamedTuple

from jevbot import vocab
from jevbot.canon import dumps_sorted, sha256_hex
from jevbot.errors import InvariantError

__all__ = [
    "BUCKET_SPEC_HASH",
    "TABLES",
    "BucketTable",
    "Cut",
    "bucket_spec_hash",
    "bucketize",
    "code_of",
    "pnl_bucket",
    "trend_direction",
]


class Cut(NamedTuple):
    """One threshold of a bucket table: `label` applies while `x < upper` (`x <= upper` when `closed`)."""

    upper: float
    closed: bool
    label: str


class BucketTable(NamedTuple):
    """A 5.5 table: `name` is its `vocab.BUCKET_CODES` key, `cuts` are ascending, `last` covers everything above them."""

    name: str
    cuts: tuple[Cut, ...]
    last: str

    @property
    def labels(self) -> tuple[str, ...]:
        """Every label of the table, in ascending value order."""
        return (*(cut.label for cut in self.cuts), self.last)

    @property
    def codes(self) -> tuple[str, ...]:
        """Every machine code of the table, in ascending value order."""
        return tuple(vocab.bucket_code(label) for label in self.labels)


def _table(name: str, *pairs: tuple[float, bool, str], last: str) -> BucketTable:
    cuts = tuple(Cut(upper, closed, label) for upper, closed, label in pairs)
    bounds = [cut.upper for cut in cuts]
    if bounds != sorted(bounds) or len(set(bounds)) != len(bounds):
        raise InvariantError(f"buckets: table {name} has non-ascending thresholds {bounds}")
    return BucketTable(name=name, cuts=cuts, last=last)


# ======================================================================================================================
# 5.5 tables - market and underlying
# ======================================================================================================================

_PCTL5_RANGE_TAIL: Final = "percent of the way from the past year's low to its high"

PCTL5_PCTILE: Final = _table(
    "PCTL5",
    (20.0, False, "very_low: bottom fifth of the past year"),
    (40.0, False, "low: 20th to 40th percentile of the past year"),
    (60.0, False, "middle: 40th to 60th percentile of the past year"),
    (80.0, False, "upper_middle: 60th to 80th percentile of the past year"),
    last="high: top fifth of the past year",
)
PCTL5_RANGE: Final = _table(  # vol_surface.iv_rank_1y: the same codes with range wording
    "PCTL5",
    (20.0, False, "very_low: within the bottom fifth of the past year's range"),
    (40.0, False, f"low: 20 to 40 {_PCTL5_RANGE_TAIL}"),
    (60.0, False, f"middle: 40 to 60 {_PCTL5_RANGE_TAIL}"),
    (80.0, False, f"upper_middle: 60 to 80 {_PCTL5_RANGE_TAIL}"),
    last="high: within the top fifth of the past year's range",
)
PCTL3: Final = _table(  # market.vol_of_vol, market.tail_skew_index
    "PCTL3",
    (30.0, False, "subdued: bottom 30 percent of the past year"),
    (70.0, True, "normal: middle of the past year's range"),  # [30,70] is CLOSED; (70,100] is elevated
    last="elevated: top 30 percent of the past year",
)
PCTL3_SKEW: Final = _table(  # vol_surface.skew: the same thresholds, its own wording
    "PCTL3_SKEW",
    (30.0, False, "flat: downside puts cheap relative to the past year"),
    (70.0, True, "normal: downside puts moderately bid"),
    last="steep: downside puts heavily bid relative to the past year",
)
CHANGE5: Final = _table(
    "CHANGE5",
    (-0.15, False, "falling_sharply: down more than 15 percent"),
    (-0.05, False, "falling: down 5 to 15 percent"),
    (0.05, True, "steady: little change"),  # [-0.05, 0.05] closed at both ends
    (0.15, True, "rising: up 5 to 15 percent"),
    last="rising_sharply: up more than 15 percent",
)
TERM4_MARKET: Final = _table(  # market.vol_term_structure on vix_term ("3-month" wording)
    "TERM4",
    (0.85, False, "steep_contango: 30-day implied volatility far below 3-month implied volatility"),
    (0.95, False, "contango: 30-day implied volatility below 3-month implied volatility"),
    (1.05, True, "flat: 30-day and 3-month implied volatility about equal"),
    last="backwardation: 30-day implied volatility above 3-month implied volatility",
)
TERM4_SURFACE: Final = _table(  # vol_surface.term_structure on iv_term ("90-day" wording)
    "TERM4",
    (0.85, False, "steep_contango: 30-day implied volatility far below 90-day implied volatility"),
    (0.95, False, "contango: 30-day implied volatility below 90-day implied volatility"),
    (1.05, True, "flat: 30-day and 90-day implied volatility about equal"),
    last="backwardation: 30-day implied volatility above 90-day implied volatility",
)
NEAR3: Final = _table(
    "NEAR3",
    (0.95, False, "calm: 9-day implied volatility below 30-day implied volatility"),
    (1.10, True, "neutral: 9-day and 30-day implied volatility about equal"),
    last="stressed: 9-day implied volatility well above 30-day implied volatility",
)
TREND_STRENGTH: Final = _table(
    "TREND_STRENGTH",
    (0.5, False, "weak: the 20-day move is small relative to normal daily swings"),
    (1.5, False, "moderate: the 20-day move is ordinary relative to normal daily swings"),
    last="strong: the 20-day move is large relative to normal daily swings",
)
DIST_ATR: Final = _table(
    "DIST_ATR",
    (-2.5, False, "stretched_far_below: price far below its 20-day average"),
    (-1.0, False, "extended_below: price moderately below its 20-day average"),
    (1.0, True, "near_average: price close to its 20-day average"),
    (2.5, True, "extended_above: price moderately above its 20-day average"),
    last="stretched_far_above: price far above its 20-day average",
)
STREAK: Final = _table(
    "STREAK",
    (-4.0, True, "long_down_streak: four or more lower closes in a row"),  # <= -4
    (-2.0, True, "short_down_streak: two or three lower closes in a row"),  # -3, -2
    (1.0, True, "no_streak: no run of closes in one direction"),  # -1, 0, 1
    (3.0, True, "short_up_streak: two or three higher closes in a row"),  # 2, 3
    last="long_up_streak: four or more higher closes in a row",  # >= 4
)
RV_CHANGE: Final = _table(
    "RV_CHANGE",
    (0.85, False, "contracting: last week's swings smaller than the past month's"),
    (1.15, True, "stable: last week's swings similar to the past month's"),
    (1.5, True, "expanding: last week's swings larger than the past month's"),
    last="expanding_sharply: last week's swings much larger than the past month's",
)
SIGMA5_GAP: Final = _table(
    "SIGMA5_GAP",
    (-1.5, False, "large_gap_down: opened far below the prior close"),
    (-0.5, False, "gap_down: opened below the prior close"),
    (0.5, True, "none: opened near the prior close"),
    (1.5, True, "gap_up: opened above the prior close"),
    last="large_gap_up: opened far above the prior close",
)
SIGMA5_MOVE: Final = _table(
    "SIGMA5_MOVE",
    (-1.5, False, "large_decline: fell far more than a normal day"),
    (-0.5, False, "decline: fell about a normal day's move"),
    (0.5, True, "quiet: little net change today"),
    (1.5, True, "advance: rose about a normal day's move"),
    last="large_advance: rose far more than a normal day",
)
DD_52W: Final = _table(
    "DD_52W",
    (-0.20, False, "deep_drawdown: more than 20 percent below the 1-year high"),
    (-0.10, False, "correction: 10 to 20 percent below the 1-year high"),
    (-0.05, False, "pullback: 5 to 10 percent below the 1-year high"),
    (-0.02, False, "close_to_high: 2 to 5 percent below the 1-year high"),
    last="near_high: within 2 percent of the 1-year high",
)
IV_RV: Final = _table(
    "IV_RV",
    (0.9, False, "iv_cheap: implied volatility below recent realized volatility"),
    (1.1, False, "iv_fair: implied volatility about equal to recent realized volatility"),
    (1.4, False, "iv_rich: implied volatility clearly above recent realized volatility"),
    last="iv_very_rich: implied volatility far above recent realized volatility",
)


def _em_label(code: str) -> str:
    """5.5 EM: the meaning is `": one expected move over this horizon is about <the code's words with spaces>"`."""
    return f"{code}: one expected move over this horizon is about {code.replace('_', ' ')}"


EM: Final = _table(  # value = tenths of a percent
    "EM",
    (5.0, False, _em_label("under_half_percent")),
    (10.0, False, _em_label("half_to_1_percent")),
    (20.0, False, _em_label("1_to_2_percent")),
    (40.0, False, _em_label("2_to_4_percent")),
    (60.0, False, _em_label("4_to_6_percent")),
    (100.0, False, _em_label("6_to_10_percent")),
    last=_em_label("10_percent_or_more"),
)
HOLD: Final = _table(  # context.holding_window_sessions
    "HOLD",
    (5.0, True, "about_one_week: roughly five trading sessions"),
    (10.0, True, "about_two_weeks: roughly ten trading sessions"),
    (17.0, True, "about_three_weeks: roughly fifteen trading sessions"),
    (25.0, True, "about_four_weeks: roughly twenty trading sessions"),
    last="more_than_a_month: more than a month of trading sessions",
)

# ======================================================================================================================
# 5.5 position tables (manage state)
# ======================================================================================================================

DTE4: Final = _table(  # value = calendar days from the session to pos.structure.last_session (never the listed expiry)
    "DTE4",
    (7.0, True, "one_week_or_less: about one week or less until expiry"),
    (14.0, False, "one_to_two_weeks: about one to two weeks until expiry"),
    (28.0, False, "two_to_four_weeks: about two to four weeks until expiry"),
    last="four_weeks_or_more: about four weeks or more until expiry",
)
HELD4: Final = _table(  # value = sessions held, counted from the decision session
    "HELD4",
    (2.0, True, "just_opened: held for at most two sessions"),
    (7.0, True, "about_one_week: held about one week"),
    (15.0, True, "two_to_three_weeks: held about two to three weeks"),
    last="more_than_three_weeks: held more than three weeks",
)
PNL_LOSS: Final = _table(  # b = -pnl_mid / max_loss_mid, evaluated when pnl_mid < 0
    "PNL",
    (0.05, False, "flat: about break-even"),
    (0.25, False, "small_loss: under one quarter of the maximum loss"),
    (0.5, False, "moderate_loss: one quarter to one half of the maximum loss"),
    (0.75, False, "loss: one half to three quarters of the maximum loss"),
    last="large_loss: more than three quarters of the maximum loss",
)
PNL_GAIN: Final = _table(  # g = pnl_mid / max_profit_mid, evaluated when pnl_mid >= 0
    "PNL",
    (0.05, False, "flat: about break-even"),
    (0.25, False, "small_gain: under one quarter of the maximum profit"),
    (0.5, False, "gain: one quarter to one half of the maximum profit"),
    last="large_gain: more than one half of the maximum profit",
)
PNL_GAIN_LONG: Final = _table(  # long options have no maximum profit: the ratio is to the premium paid (5.5)
    "PNL",
    (0.05, False, "flat: about break-even"),
    (0.25, False, "small_gain: under one quarter of the premium paid"),
    (0.5, False, "gain: one quarter to one half of the premium paid"),
    last="large_gain: more than one half of the premium paid",
)
SHORT_DIST: Final = _table(  # d = ln distance to the nearest short strike in expected moves to that expiry
    "SHORT_DIST",
    (0.0, False, "breached: price is beyond the short strike"),
    (0.25, False, "at_strike: at the short strike"),
    (0.75, False, "close: within three quarters of an expected move of the short strike"),
    (1.5, False, "about_one_move: about one expected move from the short strike"),
    last="far: more than one and a half expected moves from the short strike",
)
BREAKEVEN: Final = _table(  # e = signed distance past the mid-price breakeven, in expected moves to that expiry
    "BREAKEVEN",
    (-1.5, False, "far_short: far short of breakeven"),
    (-0.5, False, "short: about one expected move short of breakeven"),
    (0.0, False, "just_short: slightly short of breakeven"),
    (0.5, False, "just_beyond: slightly past breakeven"),
    last="well_beyond: more than half an expected move past breakeven",
)
MOVE_SINCE_ENTRY: Final = _table(  # x = ln(ref / entry_spot) in entry-time holding-window expected moves, signed so + favours us
    "MOVE_SINCE_ENTRY",
    (-1.0, True, "strongly_adverse: price has moved strongly against the position since entry"),
    (-0.25, True, "adverse: price has moved against the position since entry"),
    (0.25, False, "little_change: price has moved little relative to the position since entry"),
    (1.0, False, "favourable: price has moved in favour of the position since entry"),
    last="strongly_favourable: price has moved strongly in favour of the position since entry",
)

# ======================================================================================================================
# TREND_DIR (5.5): an ORDERED rule table, not a threshold table
# ======================================================================================================================

TREND_UP: Final = "up: price above rising 20-day and 50-day averages"
TREND_DOWN: Final = "down: price below falling 20-day and 50-day averages"
TREND_FLAT: Final = "flat: price near flat 20-day and 50-day averages"
TREND_MIXED: Final = "mixed: price and averages are not aligned"
TREND_DIR_LABELS: Final[tuple[str, ...]] = (TREND_UP, TREND_DOWN, TREND_FLAT, TREND_MIXED)
_TREND_FLAT_TOLERANCE: Final = 0.01


def trend_direction(ref: float | None, ma20: float | None, ma50: float | None, ma20_prev: float | None) -> str:
    """The 5.5 TREND_DIR label, evaluated in the documented order (`c = ref`); `unavailable` when an input is missing."""
    if ref is None or ma20 is None or ma50 is None or ma20_prev is None:
        return vocab.UNAVAILABLE
    for name, value in (("ref", ref), ("ma20", ma20), ("ma50", ma50), ("ma20_prev", ma20_prev)):
        if not math.isfinite(value):
            raise InvariantError(f"buckets.trend_direction: {name} is not finite ({value!r})")
    if ma20 <= 0.0 or ma50 <= 0.0:
        raise InvariantError(f"buckets.trend_direction: moving averages must be positive, got ma20={ma20!r}, ma50={ma50!r}")
    if ref > ma20 > ma50 and ma20 > ma20_prev:
        return TREND_UP
    if ref < ma20 < ma50 and ma20 < ma20_prev:
        return TREND_DOWN
    if abs(ref / ma20 - 1.0) < _TREND_FLAT_TOLERANCE and abs(ma20 / ma50 - 1.0) < _TREND_FLAT_TOLERANCE:
        return TREND_FLAT
    return TREND_MIXED


# ======================================================================================================================
# The generic bucketiser
# ======================================================================================================================


def bucketize(value: float | None, table: BucketTable) -> str:
    """The 5.5 label of `value` in `table`; `vocab.UNAVAILABLE` when `value` is None (a missing feature, 5.5).

    A non-finite value is an `InvariantError`: NaN compares False against every bound and would silently fall into the
    table's top bucket (`features.py` renders an unavailable feature as None, never as NaN).
    """
    if value is None:
        return vocab.UNAVAILABLE
    x = float(value)
    if not math.isfinite(x):
        raise InvariantError(f"buckets.bucketize: {table.name} received a non-finite value ({value!r})")
    for cut in table.cuts:
        if x <= cut.upper if cut.closed else x < cut.upper:
            return cut.label
    return table.last


def code_of(label: str) -> str:
    """The machine code of a label (`vocab.bucket_code`): the text before the first colon."""
    return vocab.bucket_code(label)


def _ratio_bucket(numerator: int, base: int | None, table: BucketTable) -> str:
    if base is None or base <= 0:  # invalid economics (a credit >= the width, a debit priced at a credit): 9.2
        return vocab.UNAVAILABLE
    return bucketize(numerator / base, table)


def pnl_bucket(*, pnl_mid: int, gain_base: int | None, loss_base: int | None, long_premium: bool) -> str:
    """The 5.5 PNL label from the path-independent mid-to-mid P&L of ONE contract (cents).

    `pnl_mid = (-entry.open_mid_at_decision - mid_value) * 100`. `gain_base` is `structmath.max_profit_pc` evaluated at
    `open_mid_at_decision` with `fee_rt = 0` - for a long single (`long_premium`) that is unbounded, so the caller passes
    the debit paid and the labels say "of the premium paid". `loss_base` is `structmath.max_loss_pc` there.
    `g = pnl_mid / gain_base` when `pnl_mid >= 0` (`g < 0.05` is `flat`), else `b = -pnl_mid / loss_base`. A missing or
    non-positive base is invalid economics and renders `unavailable` - never a silently wrong bucket.
    """
    if pnl_mid >= 0:
        return _ratio_bucket(pnl_mid, gain_base, PNL_GAIN_LONG if long_premium else PNL_GAIN)
    return _ratio_bucket(-pnl_mid, loss_base, PNL_LOSS)


TABLES: Final[Mapping[str, BucketTable]] = MappingProxyType(
    {
        "PCTL5_PCTILE": PCTL5_PCTILE,
        "PCTL5_RANGE": PCTL5_RANGE,
        "PCTL3": PCTL3,
        "PCTL3_SKEW": PCTL3_SKEW,
        "CHANGE5": CHANGE5,
        "TERM4_MARKET": TERM4_MARKET,
        "TERM4_SURFACE": TERM4_SURFACE,
        "NEAR3": NEAR3,
        "TREND_STRENGTH": TREND_STRENGTH,
        "DIST_ATR": DIST_ATR,
        "STREAK": STREAK,
        "RV_CHANGE": RV_CHANGE,
        "SIGMA5_GAP": SIGMA5_GAP,
        "SIGMA5_MOVE": SIGMA5_MOVE,
        "DD_52W": DD_52W,
        "IV_RV": IV_RV,
        "EM": EM,
        "HOLD": HOLD,
        "DTE4": DTE4,
        "HELD4": HELD4,
        "PNL_LOSS": PNL_LOSS,
        "PNL_GAIN": PNL_GAIN,
        "PNL_GAIN_LONG": PNL_GAIN_LONG,
        "SHORT_DIST": SHORT_DIST,
        "BREAKEVEN": BREAKEVEN,
        "MOVE_SINCE_ENTRY": MOVE_SINCE_ENTRY,
    }
)


def _spec_material() -> dict[str, object]:
    """The hashed 5.5 material: every table plus the TREND_DIR rule labels, floats spelt with `repr` (canon refuses floats)."""
    tables: dict[str, object] = {
        key: {
            "table": table.name,
            "cuts": [[repr(cut.upper), cut.closed, cut.label] for cut in table.cuts],
            "last": table.last,
        }
        for key, table in TABLES.items()
    }
    tables["TREND_DIR"] = {
        "table": "TREND_DIR",
        "labels": list(TREND_DIR_LABELS),
        "flat_tolerance": repr(_TREND_FLAT_TOLERANCE),
    }
    return tables


BUCKET_SPEC_HASH: Final[str] = sha256_hex(dumps_sorted(_spec_material()))


def bucket_spec_hash() -> str:
    """`sha256(dumps_sorted(tables))` of 5.5 - the callable spelling of `BUCKET_SPEC_HASH`, which
    `config.state_config_hash(cfg, mask_version=..., bucket_spec_hash=...)` takes as a string."""
    return BUCKET_SPEC_HASH
