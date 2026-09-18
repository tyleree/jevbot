"""Property tests of the bucket tables (DESIGN.md 5.5, 15.1 `features / buckets`).

Seeded numpy generators only (no hypothesis dependency, 1.1). The three properties are the ones the work-package
acceptance names: every float maps to exactly one bucket, the boundaries are half-open exactly as documented, and every
label's code is a `vocab` code.
"""

import math

import numpy as np
import pytest

from jevbot import buckets, vocab

SEED = 20260917
DRAWS = 4000


def _generator(purpose: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(abs(hash((SEED, purpose))) % (2**63)))


def _span(table: buckets.BucketTable) -> tuple[float, float]:
    bounds = [cut.upper for cut in table.cuts]
    lo, hi = min(bounds), max(bounds)
    pad = max(1.0, (hi - lo) * 0.5)
    return lo - pad, hi + pad


TABLES = list(buckets.TABLES.items())


@pytest.mark.parametrize(("key", "table"), TABLES, ids=[key for key, _ in TABLES])
def test_the_table_is_a_partition_and_is_monotone_in_the_value(key: str, table: buckets.BucketTable) -> None:
    """Exactly one label per value, always one of the table's own, and never decreasing as the value rises."""
    lo, hi = _span(table)
    values = np.sort(_generator(key).uniform(lo, hi, DRAWS))
    labels = table.labels
    previous = -1
    for value in values:
        label = buckets.bucketize(float(value), table)
        assert labels.count(label) == 1, (key, value)
        index = labels.index(label)
        assert index >= previous, f"{key}: {value} fell back to an earlier bucket"
        previous = index
    assert buckets.bucketize(lo - 1e9, table) == labels[0]
    assert buckets.bucketize(hi + 1e9, table) == labels[-1]


@pytest.mark.parametrize(("key", "table"), TABLES, ids=[key for key, _ in TABLES])
def test_every_cut_is_exactly_as_inclusive_as_documented(key: str, table: buckets.BucketTable) -> None:
    """`x < upper` for a half-open cut, `x <= upper` for a closed one - checked at the bound and one ulp on each side."""
    for index, cut in enumerate(table.cuts):
        below = math.nextafter(cut.upper, -math.inf)
        above = math.nextafter(cut.upper, math.inf)
        assert buckets.bucketize(below, table) == cut.label, (key, cut.upper)
        at = buckets.bucketize(cut.upper, table)
        after = table.labels[index + 1]
        assert at == (cut.label if cut.closed else after), (key, cut.upper)
        assert buckets.bucketize(above, table) == after, (key, cut.upper)


@pytest.mark.parametrize(("key", "table"), TABLES, ids=[key for key, _ in TABLES])
def test_every_label_carries_a_vocab_code_and_a_meaning(key: str, table: buckets.BucketTable) -> None:
    for label in table.labels:
        code, sep, meaning = label.partition(": ")
        assert sep == ": " and meaning and code in vocab.ALL_BUCKET_CODES, (key, label)
    assert buckets.bucketize(None, table) == vocab.UNAVAILABLE


def test_the_trend_rule_returns_one_of_its_four_labels_for_any_positive_inputs() -> None:
    rng = _generator("trend")
    seen: set[str] = set()
    for _ in range(DRAWS):
        ref, ma20, ma50, ma20_prev = (float(x) for x in rng.uniform(80.0, 120.0, 4))
        label = buckets.trend_direction(ref, ma20, ma50, ma20_prev)
        assert label in buckets.TREND_DIR_LABELS
        seen.add(buckets.code_of(label))
    assert seen == set(vocab.TREND_DIR)  # all four branches are reachable


def test_the_pnl_bucket_is_monotone_in_the_profit_and_in_the_loss() -> None:
    gain_base, loss_base = 6000, 14000
    gains = [buckets.pnl_bucket(pnl_mid=value, gain_base=gain_base, loss_base=loss_base, long_premium=False) for value in range(0, 6000, 7)]
    losses = [buckets.pnl_bucket(pnl_mid=-value, gain_base=gain_base, loss_base=loss_base, long_premium=False) for value in range(0, 14000, 13)]
    gain_order = [buckets.PNL_GAIN.labels.index(label) for label in gains]
    loss_order = [buckets.PNL_LOSS.labels.index(label) for label in losses]
    assert gain_order == sorted(gain_order) and loss_order == sorted(loss_order)
    assert gains[0] == losses[0] == "flat: about break-even"  # the zero P&L is one bucket, reached from both sides
