"""`buckets.py` (DESIGN.md 5.5): every threshold, every label string, the TREND_DIR rule and the PNL bucket.

The boundary table below is written out by hand from 5.5 - never derived from `buckets.py`'s own data - so a moved
threshold or a flipped inclusivity fails here.
"""

import math

import pytest

from jevbot import buckets, vocab
from jevbot.errors import InvariantError

# (table, [(value, expected code), ...]) - hand-transcribed from DESIGN 5.5, including every boundary value
BOUNDARY_CASES: list[tuple[buckets.BucketTable, list[tuple[float, str]]]] = [
    (
        buckets.PCTL5_PCTILE,
        [
            (0, "very_low"),
            (19.9, "very_low"),
            (20, "low"),
            (39.9, "low"),
            (40, "middle"),
            (59.9, "middle"),
            (60, "upper_middle"),
            (79.9, "upper_middle"),
            (80, "high"),
            (100, "high"),
        ],
    ),
    (
        buckets.PCTL5_RANGE,
        [(0, "very_low"), (20, "low"), (40, "middle"), (60, "upper_middle"), (80, "high"), (100, "high")],
    ),
    (buckets.PCTL3, [(0, "subdued"), (29.9, "subdued"), (30, "normal"), (70, "normal"), (70.1, "elevated"), (100, "elevated")]),
    (buckets.PCTL3_SKEW, [(0, "flat"), (29.9, "flat"), (30, "normal"), (70, "normal"), (70.1, "steep"), (100, "steep")]),
    (
        buckets.CHANGE5,
        [
            (-0.16, "falling_sharply"),
            (-0.15, "falling"),
            (-0.051, "falling"),
            (-0.05, "steady"),
            (0.0, "steady"),
            (0.05, "steady"),
            (0.051, "rising"),
            (0.15, "rising"),
            (0.151, "rising_sharply"),
        ],
    ),
    (
        buckets.TERM4_MARKET,
        [(0.84, "steep_contango"), (0.85, "contango"), (0.949, "contango"), (0.95, "flat"), (1.05, "flat"), (1.051, "backwardation")],
    ),
    (
        buckets.TERM4_SURFACE,
        [(0.84, "steep_contango"), (0.85, "contango"), (0.95, "flat"), (1.05, "flat"), (1.051, "backwardation")],
    ),
    (buckets.NEAR3, [(0.94, "calm"), (0.95, "neutral"), (1.10, "neutral"), (1.11, "stressed")]),
    (buckets.TREND_STRENGTH, [(0.49, "weak"), (0.5, "moderate"), (1.49, "moderate"), (1.5, "strong"), (9.0, "strong")]),
    (
        buckets.DIST_ATR,
        [
            (-3.0, "stretched_far_below"),
            (-2.5, "extended_below"),
            (-1.01, "extended_below"),
            (-1.0, "near_average"),
            (0.0, "near_average"),
            (1.0, "near_average"),
            (1.01, "extended_above"),
            (2.5, "extended_above"),
            (2.51, "stretched_far_above"),
        ],
    ),
    (
        buckets.STREAK,
        [
            (-9, "long_down_streak"),
            (-4, "long_down_streak"),
            (-3, "short_down_streak"),
            (-2, "short_down_streak"),
            (-1, "no_streak"),
            (0, "no_streak"),
            (1, "no_streak"),
            (2, "short_up_streak"),
            (3, "short_up_streak"),
            (4, "long_up_streak"),
            (20, "long_up_streak"),
        ],
    ),
    (
        buckets.RV_CHANGE,
        [(0.84, "contracting"), (0.85, "stable"), (1.15, "stable"), (1.16, "expanding"), (1.5, "expanding"), (1.51, "expanding_sharply")],
    ),
    (
        buckets.SIGMA5_GAP,
        [
            (-1.6, "large_gap_down"),
            (-1.5, "gap_down"),
            (-0.51, "gap_down"),
            (-0.5, "none"),
            (0.5, "none"),
            (0.51, "gap_up"),
            (1.5, "gap_up"),
            (1.51, "large_gap_up"),
        ],
    ),
    (
        buckets.SIGMA5_MOVE,
        [
            (-1.6, "large_decline"),
            (-1.5, "decline"),
            (-0.5, "quiet"),
            (0.5, "quiet"),
            (0.51, "advance"),
            (1.5, "advance"),
            (1.51, "large_advance"),
        ],
    ),
    (
        buckets.DD_52W,
        [
            (-0.21, "deep_drawdown"),
            (-0.20, "correction"),
            (-0.11, "correction"),
            (-0.10, "pullback"),
            (-0.051, "pullback"),
            (-0.05, "close_to_high"),
            (-0.021, "close_to_high"),
            (-0.02, "near_high"),
            (0.0, "near_high"),
        ],
    ),
    (buckets.IV_RV, [(0.89, "iv_cheap"), (0.9, "iv_fair"), (1.09, "iv_fair"), (1.1, "iv_rich"), (1.39, "iv_rich"), (1.4, "iv_very_rich")]),
    (
        buckets.EM,
        [
            (1, "under_half_percent"),
            (4, "under_half_percent"),
            (5, "half_to_1_percent"),
            (9, "half_to_1_percent"),
            (10, "1_to_2_percent"),
            (19, "1_to_2_percent"),
            (20, "2_to_4_percent"),
            (39, "2_to_4_percent"),
            (40, "4_to_6_percent"),
            (59, "4_to_6_percent"),
            (60, "6_to_10_percent"),
            (99, "6_to_10_percent"),
            (100, "10_percent_or_more"),
            (250, "10_percent_or_more"),
        ],
    ),
    (
        buckets.HOLD,
        [
            (1, "about_one_week"),
            (5, "about_one_week"),
            (6, "about_two_weeks"),
            (10, "about_two_weeks"),
            (11, "about_three_weeks"),
            (17, "about_three_weeks"),
            (18, "about_four_weeks"),
            (25, "about_four_weeks"),
            (26, "more_than_a_month"),
        ],
    ),
    (
        buckets.DTE4,
        [
            (0, "one_week_or_less"),
            (7, "one_week_or_less"),
            (8, "one_to_two_weeks"),
            (13, "one_to_two_weeks"),
            (14, "two_to_four_weeks"),
            (27, "two_to_four_weeks"),
            (28, "four_weeks_or_more"),
            (120, "four_weeks_or_more"),
        ],
    ),
    (
        buckets.HELD4,
        [
            (0, "just_opened"),
            (2, "just_opened"),
            (3, "about_one_week"),
            (7, "about_one_week"),
            (8, "two_to_three_weeks"),
            (15, "two_to_three_weeks"),
            (16, "more_than_three_weeks"),
        ],
    ),
    (
        buckets.SHORT_DIST,
        [
            (-0.1, "breached"),
            (0.0, "at_strike"),
            (0.24, "at_strike"),
            (0.25, "close"),
            (0.74, "close"),
            (0.75, "about_one_move"),
            (1.49, "about_one_move"),
            (1.5, "far"),
        ],
    ),
    (
        buckets.BREAKEVEN,
        [
            (-1.6, "far_short"),
            (-1.5, "short"),
            (-0.51, "short"),
            (-0.5, "just_short"),
            (-0.01, "just_short"),
            (0.0, "just_beyond"),
            (0.49, "just_beyond"),
            (0.5, "well_beyond"),
        ],
    ),
    (
        buckets.MOVE_SINCE_ENTRY,
        [
            (-1.1, "strongly_adverse"),
            (-1.0, "strongly_adverse"),
            (-0.99, "adverse"),
            (-0.25, "adverse"),
            (-0.24, "little_change"),
            (0.0, "little_change"),
            (0.24, "little_change"),
            (0.25, "favourable"),
            (0.99, "favourable"),
            (1.0, "strongly_favourable"),
        ],
    ),
    (
        buckets.PNL_LOSS,
        [
            (0.0, "flat"),
            (0.049, "flat"),
            (0.05, "small_loss"),
            (0.249, "small_loss"),
            (0.25, "moderate_loss"),
            (0.49, "moderate_loss"),
            (0.5, "loss"),
            (0.749, "loss"),
            (0.75, "large_loss"),
        ],
    ),
    (buckets.PNL_GAIN, [(0.0, "flat"), (0.049, "flat"), (0.05, "small_gain"), (0.25, "gain"), (0.49, "gain"), (0.5, "large_gain")]),
    (buckets.PNL_GAIN_LONG, [(0.049, "flat"), (0.05, "small_gain"), (0.25, "gain"), (0.5, "large_gain")]),
]


@pytest.mark.parametrize(("table", "cases"), [(table, cases) for table, cases in BOUNDARY_CASES], ids=[t.name for t, _ in BOUNDARY_CASES])
def test_every_documented_boundary_lands_in_the_documented_bucket(table: buckets.BucketTable, cases: list[tuple[float, str]]) -> None:
    for value, code in cases:
        assert buckets.code_of(buckets.bucketize(value, table)) == code, f"{table.name} at {value}"


def test_every_table_covers_exactly_the_vocab_codes_in_ascending_order() -> None:
    # PNL is split into its loss / gain halves (5.5 is two-sided), so the three PNL tables are checked against subsets
    for key, table in buckets.TABLES.items():
        codes = table.codes
        assert len(set(codes)) == len(codes), key
        assert all(label.split(":", 1)[0] == code for label, code in zip(table.labels, codes, strict=True)), key
        expected = vocab.BUCKET_CODES[table.name]
        if key.startswith("PNL"):
            assert set(codes) <= set(expected), key
        else:
            assert set(codes) == set(expected), key
    assert set(buckets.PNL_LOSS.codes) | set(buckets.PNL_GAIN.codes) == set(vocab.PNL)
    assert set(buckets.PNL_GAIN_LONG.codes) == set(buckets.PNL_GAIN.codes)


def test_ascending_code_order_is_pinned() -> None:
    assert buckets.PCTL5_PCTILE.codes == ("very_low", "low", "middle", "upper_middle", "high")
    assert buckets.DTE4.codes == ("one_week_or_less", "one_to_two_weeks", "two_to_four_weeks", "four_weeks_or_more")
    assert buckets.HELD4.codes == ("just_opened", "about_one_week", "two_to_three_weeks", "more_than_three_weeks")
    assert buckets.SHORT_DIST.codes == ("breached", "at_strike", "close", "about_one_move", "far")
    assert buckets.BREAKEVEN.codes == ("far_short", "short", "just_short", "just_beyond", "well_beyond")
    assert buckets.MOVE_SINCE_ENTRY.codes == ("strongly_adverse", "adverse", "little_change", "favourable", "strongly_favourable")


def test_every_label_is_code_colon_meaning_and_the_two_wordings_differ() -> None:
    for table in buckets.TABLES.values():
        for label in table.labels:
            code, sep, meaning = label.partition(": ")
            assert sep == ": " and meaning.strip(), label
            assert code in vocab.ALL_BUCKET_CODES, label
    assert "percentile of the past year" in buckets.PCTL5_PCTILE.labels[1]
    assert "of the way from the past year's low to its high" in buckets.PCTL5_RANGE.labels[1]
    assert "3-month" in buckets.TERM4_MARKET.labels[1] and "90-day" in buckets.TERM4_SURFACE.labels[1]
    assert "of the maximum profit" in buckets.PNL_GAIN.labels[-1] and "of the premium paid" in buckets.PNL_GAIN_LONG.labels[-1]


def test_the_five_labels_quoted_verbatim_in_the_spec() -> None:
    # DESIGN 5.5 / 5.6 / 5.7 print these strings; they are part of the state bytes
    assert buckets.bucketize(62, buckets.PCTL5_RANGE) == "upper_middle: 60 to 80 percent of the way from the past year's low to its high"
    assert buckets.bucketize(21, buckets.EM) == "2_to_4_percent: one expected move over this horizon is about 2 to 4 percent"
    assert buckets.bucketize(20, buckets.HOLD) == "about_four_weeks: roughly twenty trading sessions"
    assert buckets.bucketize(27, buckets.DTE4) == "two_to_four_weeks: about two to four weeks until expiry"
    assert buckets.bucketize(6, buckets.HELD4) == "about_one_week: held about one week"
    assert buckets.bucketize(0.0, buckets.MOVE_SINCE_ENTRY) == "little_change: price has moved little relative to the position since entry"


def test_em_meaning_is_the_code_words_with_spaces() -> None:
    for label in buckets.EM.labels:
        code, _, meaning = label.partition(": ")
        assert meaning == f"one expected move over this horizon is about {code.replace('_', ' ')}"


def test_a_missing_feature_renders_unavailable_and_a_nan_is_a_bug() -> None:
    for table in buckets.TABLES.values():
        assert buckets.bucketize(None, table) == vocab.UNAVAILABLE
        with pytest.raises(InvariantError):
            buckets.bucketize(math.nan, table)
        with pytest.raises(InvariantError):
            buckets.bucketize(math.inf, table)
    assert buckets.code_of(vocab.UNAVAILABLE) == vocab.UNAVAILABLE


# ======================================================================================================================
# TREND_DIR (an ordered rule table) and PNL (two-sided)
# ======================================================================================================================


def test_trend_direction_follows_the_documented_order() -> None:
    up = buckets.trend_direction(110.0, 105.0, 100.0, 104.0)
    down = buckets.trend_direction(90.0, 95.0, 100.0, 96.0)
    flat = buckets.trend_direction(100.4, 100.0, 100.5, 100.0)
    mixed = buckets.trend_direction(110.0, 105.0, 100.0, 106.0)  # above both averages but the 20-day is FALLING
    assert (buckets.code_of(up), buckets.code_of(down), buckets.code_of(flat), buckets.code_of(mixed)) == ("up", "down", "flat", "mixed")
    # the flat rule is strict (< 1%) on BOTH ratios
    assert buckets.code_of(buckets.trend_direction(101.0, 100.0, 100.0, 100.0)) == "mixed"
    assert buckets.code_of(buckets.trend_direction(100.9, 100.0, 100.5, 100.0)) == "flat"
    # `up` wins over `flat` when both could apply (evaluated in order)
    assert buckets.code_of(buckets.trend_direction(100.5, 100.4, 100.3, 100.2)) == "up"
    assert buckets.trend_direction(None, 1.0, 1.0, 1.0) == vocab.UNAVAILABLE
    with pytest.raises(InvariantError):
        buckets.trend_direction(100.0, 0.0, 100.0, 100.0)


def test_pnl_bucket_hand_computed_for_a_credit_spread_and_a_long_call() -> None:
    # put credit spread, width 200c, opened at a mid credit of 60c: max profit 6000c, max loss (200-60)*100 = 14000c
    gain_base, loss_base = 6000, 14000
    assert buckets.code_of(buckets.pnl_bucket(pnl_mid=0, gain_base=gain_base, loss_base=loss_base, long_premium=False)) == "flat"
    assert buckets.code_of(buckets.pnl_bucket(pnl_mid=299, gain_base=gain_base, loss_base=loss_base, long_premium=False)) == "flat"
    assert buckets.code_of(buckets.pnl_bucket(pnl_mid=300, gain_base=gain_base, loss_base=loss_base, long_premium=False)) == "small_gain"
    assert buckets.code_of(buckets.pnl_bucket(pnl_mid=3000, gain_base=gain_base, loss_base=loss_base, long_premium=False)) == "large_gain"
    assert buckets.code_of(buckets.pnl_bucket(pnl_mid=-700, gain_base=gain_base, loss_base=loss_base, long_premium=False)) == "small_loss"
    assert buckets.code_of(buckets.pnl_bucket(pnl_mid=-699, gain_base=gain_base, loss_base=loss_base, long_premium=False)) == "flat"
    assert buckets.code_of(buckets.pnl_bucket(pnl_mid=-10500, gain_base=gain_base, loss_base=loss_base, long_premium=False)) == "large_loss"
    # a long call: profit is unbounded, so the gain base is the premium paid and the wording changes
    long_gain = buckets.pnl_bucket(pnl_mid=200, gain_base=400, loss_base=400, long_premium=True)
    assert long_gain == "large_gain: more than one half of the premium paid"
    # invalid economics (a credit >= the width) can never produce a bucket
    assert buckets.pnl_bucket(pnl_mid=10, gain_base=0, loss_base=100, long_premium=False) == vocab.UNAVAILABLE
    assert buckets.pnl_bucket(pnl_mid=-10, gain_base=100, loss_base=-5, long_premium=False) == vocab.UNAVAILABLE
    assert buckets.pnl_bucket(pnl_mid=10, gain_base=None, loss_base=100, long_premium=True) == vocab.UNAVAILABLE


# ======================================================================================================================
# bucket_spec_hash (5.5): it enters state_config_hash
# ======================================================================================================================


def test_bucket_spec_hash_is_stable_and_moves_with_a_threshold_or_a_label() -> None:
    from jevbot.canon import dumps_sorted, sha256_hex

    assert buckets.bucket_spec_hash() == buckets.BUCKET_SPEC_HASH
    assert len(buckets.BUCKET_SPEC_HASH) == 64 and set(buckets.BUCKET_SPEC_HASH) <= set("0123456789abcdef")
    material = buckets._spec_material()
    assert sha256_hex(dumps_sorted(material)) == buckets.BUCKET_SPEC_HASH  # no float reaches the hashed material

    moved = buckets._spec_material()
    moved["PCTL5_PCTILE"]["cuts"][0][0] = repr(21.0)  # type: ignore[index]
    assert sha256_hex(dumps_sorted(moved)) != buckets.BUCKET_SPEC_HASH
    reworded = buckets._spec_material()
    reworded["IV_RV"]["last"] = "iv_very_rich: implied volatility far above realized volatility"  # type: ignore[index]
    assert sha256_hex(dumps_sorted(reworded)) != buckets.BUCKET_SPEC_HASH


def test_the_config_hash_accepts_it() -> None:
    from jevbot.config import Config, state_config_hash

    cfg = Config()
    digest = state_config_hash(cfg, mask_version="textmask.v1abc", bucket_spec_hash=buckets.BUCKET_SPEC_HASH)
    other = state_config_hash(cfg, mask_version="textmask.v1abc", bucket_spec_hash="0" * 64)
    assert digest != other and len(digest) == 64
