"""Evidence tiers, report banners and the pooling guard (DESIGN.md 12.1; INV-22).

Three things live here, and nothing else:

* `evidence_tier()` - the per-decision tier function of 12.1, copied from the specification. It is evaluated once per
  decision and stamped on every DECISION and FORECAST entry, so a report never has to re-derive a tier from dates.
* the banners - fixed texts that a report renders first and cannot disable. `required_banners()` says which ones a set of
  tiers / fidelities demands; `assert_banners()` raises `TierViolation` when a rendered report is missing one (INV-22).
* `assert_poolable()` - the guard that refuses to put two tiers, two namespaces, two price measures or the with-text and
  without-text forecast sets into one number. Exactly two code paths are exempt, both named in 12.1: the reference
  history (training pairs only, never scored) and `eval/agreement.py` (it compares namespaces, it never pools them).

`TierViolation` is re-exported here because 12.1 / section 16 name this module as its home; the class itself belongs to
the frozen `errors.py` hierarchy.
"""

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from typing import Final

from jevbot.errors import TierViolation
from jevbot.types import EvidenceTier, Fidelity, RunMode

__all__ = [
    "BANNERS",
    "POOLING_KEYS",
    "PRIVATE_FOOTER",
    "REPORTABLE_TIERS",
    "SMOKE_BANNER",
    "TIER_A_BANNER",
    "TIER_B_BANNER",
    "TIER_C_BANNER_TEMPLATE",
    "TierViolation",
    "assert_banners",
    "assert_poolable",
    "banner",
    "decided_live",
    "evidence_tier",
    "required_banners",
]


# ======================================================================================================================
# The tier function (12.1, verbatim)
# ======================================================================================================================


def evidence_tier(*, fidelity: Fidelity, session: date, model_release_date: date, mode: RunMode, decided_live: bool) -> EvidenceTier:
    """The evidence tier of ONE decision (12.1).

    `model_release_date` is a property of the namespace - the one tier-boundary key, stored in the cache `namespaces`
    table and in RUN_START, so it can never drift between two places. A forced model change creates a new namespace with
    its own release date, and forward data collected earlier becomes Tier C *for that namespace* (G1).
    """
    if fidelity is Fidelity.SYNTHETIC:
        return EvidenceTier.NONE  # never evidence; banner SMOKE
    if session < model_release_date:
        return EvidenceTier.C
    if mode is RunMode.PAPER and decided_live:
        return EvidenceTier.A
    return EvidenceTier.B


def decided_live(*, as_of: datetime, ledgered_wall: datetime, max_log_delay_s: int) -> bool:
    """`decided_live` of 12.1: the entry was appended by the paper runner within `evidence.tier_a_max_log_delay_s` of its
    `as_of` (600 s by default). A recorded paper day replayed later has a far larger delta and is Tier B."""
    if as_of.tzinfo is None or ledgered_wall.tzinfo is None:
        raise ValueError("as_of and ledgered_wall must be tz-aware UTC datetimes")
    delay = (ledgered_wall - as_of).total_seconds()
    return 0.0 <= delay <= float(max_log_delay_s)


# Only Tier A and B rows can enter a go / no-go table (12.1).
REPORTABLE_TIERS: Final[frozenset[EvidenceTier]] = frozenset({EvidenceTier.A, EvidenceTier.B})


# ======================================================================================================================
# Banners (12.1) - rendered first, cannot be disabled
# ======================================================================================================================

TIER_C_BANNER_TEMPLATE: Final[str] = (
    "TIER C - NOT EVIDENCE OF MODEL SKILL. Decision dates precede the model's release ({model_release_date}); its "
    "training cutoff is undisclosed, so answers may reflect memorised outcomes. Valid only for engine, risk and fill "
    "validation and for leakage diagnostics."
)
TIER_B_BANNER: Final[str] = (
    "TIER B - post-release dates replayed after the fact. Clean of training leakage but not logged at decision time."
)
TIER_A_BANNER: Final[str] = (
    "TIER A - forward paper decisions logged at decision time. P&L is from the SHADOW replay (own fill model on "
    "recorded quotes); Alpaca paper fills are a plumbing test only."
)
SMOKE_BANNER: Final[str] = "SMOKE RUN - SYNTHETIC prices. Not evidence of anything."
PRIVATE_FOOTER: Final[str] = "PRIVATE - TypeSafe MCA 2.3(f): do not publish or share performance information about the service."

# Tier -> banner, with Tier C's text still carrying its `{model_release_date}` placeholder.
BANNERS: Final[Mapping[EvidenceTier, str]] = {
    EvidenceTier.A: TIER_A_BANNER,
    EvidenceTier.B: TIER_B_BANNER,
    EvidenceTier.C: TIER_C_BANNER_TEMPLATE,
    EvidenceTier.NONE: SMOKE_BANNER,
}


def banner(tier: EvidenceTier, *, model_release_date: date | None = None) -> str:
    """The banner text of one tier. Tier C names the model's release date, so it is required for that tier."""
    text = BANNERS[tier]
    if tier is EvidenceTier.C:
        if model_release_date is None:
            raise ValueError("the Tier C banner names the model's release date: pass model_release_date")
        return text.format(model_release_date=model_release_date.isoformat())
    return text


def required_banners(
    *,
    tiers: Iterable[EvidenceTier],
    fidelities: Iterable[Fidelity] = (),
    model_release_date: date | None = None,
) -> tuple[str, ...]:
    """Every banner a report over these tiers / fidelities must render, in the fixed order A, B, C, SMOKE, PRIVATE.

    SYNTHETIC data demands the SMOKE banner even when its rows were stamped with a tier by some other path; the PRIVATE
    footer is on every report (12.1).
    """
    present = set(tiers)
    if any(fidelity is Fidelity.SYNTHETIC for fidelity in fidelities):
        present.add(EvidenceTier.NONE)
    out = [
        banner(tier, model_release_date=model_release_date)
        for tier in (EvidenceTier.A, EvidenceTier.B, EvidenceTier.C, EvidenceTier.NONE)
        if tier in present
    ]
    out.append(PRIVATE_FOOTER)
    return tuple(out)


def assert_banners(
    text: str,
    *,
    tiers: Iterable[EvidenceTier],
    fidelities: Iterable[Fidelity] = (),
    model_release_date: date | None = None,
) -> None:
    """Raise `TierViolation` unless `text` carries every banner the present tiers / fidelities demand (INV-22).

    The report writer calls this on its own rendered output: a banner that a template dropped, a caller silenced or a
    later edit removed can then never reach a reader.
    """
    missing = [
        required
        for required in required_banners(tiers=tiers, fidelities=fidelities, model_release_date=model_release_date)
        if required not in text
    ]
    if missing:
        raise TierViolation("report is missing its mandatory banner(s): " + " | ".join(missing))


# ======================================================================================================================
# The pooling guard (12.1, INV-22)
# ======================================================================================================================

# The four things that may never be pooled into one number.
POOLING_KEYS: Final[tuple[str, ...]] = ("tier", "namespace", "price_measure", "with_text")


def assert_poolable(columns: Mapping[str, Iterable[object]], *, what: str, keys: Sequence[str] = POOLING_KEYS) -> None:
    """Raise `TierViolation` when `what` would mix two tiers, namespaces, price measures or forecast sets (INV-22).

    `columns` maps a key of `keys` to the values of the rows that would enter the number (a pandas Series, a list, any
    iterable). A key that is absent is not checked - a caller that cannot know the price measure of its rows must not be
    able to *silently* pass the guard, so the report passes every column it has.
    """
    mixed: list[str] = []
    for key in keys:
        values = columns.get(key)
        if values is None:
            continue
        distinct = {_hashable(value) for value in values}
        if len(distinct) > 1:
            shown = ", ".join(sorted(repr(value) for value in distinct)[:6])
            mixed.append(f"{key}: {{{shown}}}")
    if mixed:
        raise TierViolation(
            f"refusing to pool {what} across " + "; ".join(mixed) + " (12.1: tiers, namespaces, price measures and the "
            "with-text / without-text forecast sets are never pooled)"
        )


def _hashable(value: object) -> object:
    """Values arrive from pandas as numpy scalars and from msgspec as enums; both compare by value once normalised."""
    if isinstance(value, bool | int | float | str | bytes) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _hashable(item())
        except (TypeError, ValueError):  # pragma: no cover - numpy scalars always convert
            return str(value)
    return str(value)
