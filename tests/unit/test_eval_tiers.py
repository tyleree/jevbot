"""Evidence tiers, banners and the pooling guard (DESIGN.md 12.1, INV-22; test plan 15.1 `eval/*`).

Three properties are asserted here and nowhere else:

* the **tier table** - every branch of `evidence_tier`, including the two boundaries that decide whether forward data is
  evidence at all (`session == model_release_date`, and a paper decision logged too late);
* **banner enforcement** - a report that omits the banner of a tier it contains raises `TierViolation`, so no rendering
  path can quietly drop "TIER C - NOT EVIDENCE OF MODEL SKILL";
* **pooling raises** - tiers, namespaces, price measures and the with-text / without-text forecast sets can never be
  averaged into one number.
"""

from datetime import UTC, date, datetime, timedelta

import pytest

from jevbot.errors import TierViolation
from jevbot.eval import tiers
from jevbot.types import EvidenceTier, Fidelity, RunMode

RELEASE = date(2026, 9, 15)


@pytest.mark.parametrize(
    ("fidelity", "session", "mode", "decided_live", "expected"),
    [
        # synthetic prices are never evidence, whatever the dates or the mode say
        (Fidelity.SYNTHETIC, date(2012, 1, 3), RunMode.BACKTEST, False, EvidenceTier.NONE),
        (Fidelity.SYNTHETIC, date(2027, 1, 4), RunMode.PAPER, True, EvidenceTier.NONE),
        # decision dates before the model's release may be memorised outcomes
        (Fidelity.EOD_QUOTES, date(2012, 1, 3), RunMode.BACKTEST, False, EvidenceTier.C),
        (Fidelity.LIVE_INDICATIVE, RELEASE - timedelta(days=1), RunMode.PAPER, True, EvidenceTier.C),
        # the boundary: the release day itself is NOT before the release
        (Fidelity.EOD_QUOTES, RELEASE, RunMode.BACKTEST, False, EvidenceTier.B),
        # forward paper decisions logged at decision time
        (Fidelity.LIVE_INDICATIVE, date(2026, 10, 1), RunMode.PAPER, True, EvidenceTier.A),
        # a recorded paper day replayed later is Tier B, not Tier A
        (Fidelity.RECORDED_INDICATIVE, date(2026, 10, 1), RunMode.PAPER, False, EvidenceTier.B),
        # a post-release backtest is Tier B however good its data is
        (Fidelity.EOD_QUOTES, date(2026, 10, 1), RunMode.BACKTEST, True, EvidenceTier.B),
    ],
)
def test_evidence_tier_table(fidelity: Fidelity, session: date, mode: RunMode, decided_live: bool, expected: EvidenceTier) -> None:
    assert (
        tiers.evidence_tier(
            fidelity=fidelity,
            session=session,
            model_release_date=RELEASE,
            mode=mode,
            decided_live=decided_live,
        )
        is expected
    )


def test_decided_live_uses_the_log_delay_budget() -> None:
    as_of = datetime(2026, 10, 1, 19, 35, tzinfo=UTC)
    assert tiers.decided_live(as_of=as_of, ledgered_wall=as_of, max_log_delay_s=600)
    assert tiers.decided_live(as_of=as_of, ledgered_wall=as_of.replace(minute=45), max_log_delay_s=600)  # 600 s exactly still counts
    assert not tiers.decided_live(
        as_of=as_of, ledgered_wall=as_of.replace(minute=46), max_log_delay_s=600
    )  # 660 s: replayed after the fact
    # an entry ledgered BEFORE its own as_of is not a live decision either - it is a clock or a bug
    assert not tiers.decided_live(as_of=as_of, ledgered_wall=as_of.replace(minute=30), max_log_delay_s=600)


def test_decided_live_refuses_naive_datetimes() -> None:
    naive = datetime(2026, 10, 1, 19, 35)  # noqa: DTZ001 - the point of the test
    with pytest.raises(ValueError, match="tz-aware"):
        tiers.decided_live(as_of=naive, ledgered_wall=naive, max_log_delay_s=600)


# ======================================================================================================================
# Banners
# ======================================================================================================================


def test_tier_c_banner_names_the_release_date_and_is_required_for_it() -> None:
    text = tiers.banner(EvidenceTier.C, model_release_date=RELEASE)
    assert text.startswith("TIER C - NOT EVIDENCE OF MODEL SKILL.")
    assert "2026-09-15" in text
    with pytest.raises(ValueError, match="release date"):
        tiers.banner(EvidenceTier.C)


def test_required_banners_cover_every_present_tier_and_always_the_private_footer() -> None:
    required = tiers.required_banners(tiers=[EvidenceTier.A, EvidenceTier.B], model_release_date=RELEASE)
    assert required == (tiers.TIER_A_BANNER, tiers.TIER_B_BANNER, tiers.PRIVATE_FOOTER)

    # SYNTHETIC data demands the SMOKE banner even when the rows carry some other tier label
    with_synthetic = tiers.required_banners(
        tiers=[EvidenceTier.B], fidelities=[Fidelity.EOD_QUOTES, Fidelity.SYNTHETIC], model_release_date=RELEASE
    )
    assert tiers.SMOKE_BANNER in with_synthetic
    assert with_synthetic[-1] == tiers.PRIVATE_FOOTER


def test_assert_banners_raises_when_a_rendered_report_drops_one() -> None:
    complete = "\n".join(tiers.required_banners(tiers=[EvidenceTier.B, EvidenceTier.C], model_release_date=RELEASE))
    tiers.assert_banners(complete, tiers=[EvidenceTier.B, EvidenceTier.C], model_release_date=RELEASE)

    without_c = complete.replace(tiers.banner(EvidenceTier.C, model_release_date=RELEASE), "")
    with pytest.raises(TierViolation) as excinfo:
        tiers.assert_banners(without_c, tiers=[EvidenceTier.B, EvidenceTier.C], model_release_date=RELEASE)
    assert "TIER C" in str(excinfo.value)

    without_footer = complete.replace(tiers.PRIVATE_FOOTER, "")
    with pytest.raises(TierViolation, match="PRIVATE"):
        tiers.assert_banners(without_footer, tiers=[EvidenceTier.B, EvidenceTier.C], model_release_date=RELEASE)


def test_only_tier_a_and_b_may_enter_a_go_no_go_table() -> None:
    assert tiers.REPORTABLE_TIERS == frozenset({EvidenceTier.A, EvidenceTier.B})
    assert EvidenceTier.C not in tiers.REPORTABLE_TIERS
    assert EvidenceTier.NONE not in tiers.REPORTABLE_TIERS


# ======================================================================================================================
# Pooling
# ======================================================================================================================


def test_assert_poolable_accepts_one_homogeneous_slice() -> None:
    tiers.assert_poolable(
        {
            "tier": ["A", "A", "A"],
            "namespace": ["exp:model:g0"] * 3,
            "price_measure": ["parity_forward"] * 3,
            "with_text": [True, True, True],
        },
        what="the pre-registered Brier score",
    )


@pytest.mark.parametrize(
    ("key", "values"),
    [
        ("tier", ["A", "B"]),
        ("namespace", ["exp:m1:g0", "exp:m2:g0"]),
        ("price_measure", ["parity_forward", "live_mid"]),
        ("with_text", [True, False]),
    ],
)
def test_assert_poolable_raises_on_every_forbidden_mix(key: str, values: list[object]) -> None:
    with pytest.raises(TierViolation) as excinfo:
        tiers.assert_poolable({key: values}, what="a pooled skill number")
    message = str(excinfo.value)
    assert key in message
    assert "a pooled skill number" in message


def test_assert_poolable_normalises_enum_and_numpy_values() -> None:
    numpy = pytest.importorskip("numpy")
    # EvidenceTier.A and the plain string "A" are the same tier: normalising must not invent a violation
    tiers.assert_poolable({"tier": [EvidenceTier.A, "A"]}, what="one tier")
    tiers.assert_poolable({"with_text": [numpy.bool_(True), True]}, what="one forecast set")
    with pytest.raises(TierViolation):
        tiers.assert_poolable({"tier": [EvidenceTier.A, EvidenceTier.B]}, what="two tiers")


def test_assert_poolable_ignores_keys_the_caller_does_not_have() -> None:
    # a caller that cannot know the price measure passes what it has; the guard checks exactly those columns
    tiers.assert_poolable({"tier": ["A", "A"]}, what="a slice without a measure column")
