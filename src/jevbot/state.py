"""`StateBuilder`: the four states Jev ever sees, their variants and their provenance (DESIGN.md 5, 5.6-5.9).

It implements `protocols.StateBuilderP`:

    entry(view, u)              `state.v1.entry`       text-free; None when a REQUIRED feature is unavailable (5.3)
    entry_text(view, u, base)   `state.v1.entry_text`  the SAME object plus `news_status` and `news`
    manage(view, pos)           `state.v1.manage`      everything position-specific from `pos.structure` and `pos.entry`
    manage_text(view, pos, b)   `state.v1.manage_text` plus `news_since_entry`; None when that block would be empty
    variant(state, v)           the KEY_PERM / BUCKET_ONLY renderers (OPT_PERM permutes questions, not states)

Pure and deterministic: same `MarketView` contents => same bytes => same hash (D24). No clock, no RNG, no environment,
no IO. Dict insertion order IS the wire order (V1: the SDK encodes with `msgspec.json.encode`), so the key order below
is exactly the order printed in 5.6 / 5.7 and is part of the contract.

Three properties the tests pin, and the reasons they hold by construction:

* **Text is isolated** (D11, INV-16, V6). `entry` and `manage` never read `view.news`; only the `_text` builders do,
  and they only ever APPEND to a finished base state, so no news byte can move a gate, a size or a rank.
* **The manage state is path-independent.** Nothing in it comes from our fills or our equity: the P&L bucket is
  mid-to-mid from `pos.entry.open_mid_at_decision` (the DECISION-snapshot mid) through `structmath`'s 9.2 formulas, and
  every other position field comes from `pos.structure` / `pos.entry`, which reach the `Position` through the ledger.
  A book replayed from the ledger therefore rebuilds byte-identical manage states.
* **Recency is decided in code** (5.6). `news.since_previous_session` holds the items newer than the previous session's
  decision time; the pending-event questions read that list only, so a stale "decision due tomorrow" cannot be read as
  still pending.

`Provenance.decision_id`, `.evidence_tier` and `.ledgered_wall` are the three sidecar fields a builder cannot know: the
cycle owns the decision id (2.10), the evidence tier is computed per decision (12.1) and the wall clock belongs to the
ledger writer. They are emitted as `""`, the fidelity's conservative tier and `view.as_of`, and `cycle.py` replaces them
with `msgspec.structs.replace` before the sidecar is written. Nothing hashed depends on them (INV-24).
"""

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from types import MappingProxyType
from typing import Any, Final

import msgspec

from jevbot import buckets, structmath, textmask, vocab
from jevbot.cal import trading_time
from jevbot.canon import dumps_ordered, ensure_state_safe, sha256_hex
from jevbot.config import Config, headline_band
from jevbot.errors import DataUnavailable, InvariantError
from jevbot.features import FeatureSet, compute_features, parse_atm_term, total_variance_at
from jevbot.protocols import Calendar, MarketView
from jevbot.types import (
    SHORT_PREMIUM,
    STRUCTURE_DIRECTION,
    Bp,
    BuiltState,
    Cents,
    Direction,
    EntryFacts,
    EvidenceTier,
    Fidelity,
    ManageFacts,
    MaskTerms,
    Position,
    Provenance,
    Right,
    ScheduledEvent,
    Structure,
    StructureKind,
    Variant,
)

__all__ = [
    "DECISION_SESSION_TEXT",
    "DIRECTIONAL_EXPOSURE",
    "EARNINGS_TEXT",
    "EVENT_COVERAGE_TEXT",
    "EVENT_TEXT",
    "MARKET_AS_OF_TEXT",
    "VOL_EXPOSURE",
    "StateBuilder",
]

DECISION_SESSION_TEXT: Final = "near the close of the regular trading session"
MARKET_AS_OF_TEXT: Final = "prior session close"  # the vol indices are the prior session's values (D22)
EARNINGS_TEXT: Final = "not_applicable_index_etf"  # the v1 universe is index ETFs (section 4 `universe`)
EM_UNIT_TEXT: Final = "tenths of a percent"

# 5.7: fixed strings per StructureKind - never derived from the legs, never from our fills
DIRECTIONAL_EXPOSURE: Final[Mapping[StructureKind, str]] = MappingProxyType(
    {
        StructureKind.LONG_CALL: "bullish: profits if price rises well above the strike before expiry",
        StructureKind.LONG_PUT: "bearish: profits if price falls well below the strike before expiry",
        StructureKind.CALL_DEBIT: "moderately bullish: profits if price rises toward or above the short call strike",
        StructureKind.PUT_DEBIT: "moderately bearish: profits if price falls toward or below the short put strike",
        StructureKind.PUT_CREDIT: "neutral to bullish: profits if price stays above the short put strike",
        StructureKind.CALL_CREDIT: "neutral to bearish: profits if price stays below the short call strike",
        StructureKind.IRON_CONDOR: "range-bound: profits if price stays between the short put and short call strikes",
    }
)
_LONG_PREMIUM_TEXT: Final = "long premium: loses from time passing, gains from rising implied volatility"
_SHORT_PREMIUM_TEXT: Final = "short premium: profits from time passing and falling implied volatility"
_LIMITED_TEXT: Final = "limited: small net sensitivity to implied volatility"
VOL_EXPOSURE: Final[Mapping[StructureKind, str]] = MappingProxyType(
    {
        StructureKind.LONG_CALL: _LONG_PREMIUM_TEXT,
        StructureKind.LONG_PUT: _LONG_PREMIUM_TEXT,
        StructureKind.CALL_DEBIT: _LIMITED_TEXT,
        StructureKind.PUT_DEBIT: _LIMITED_TEXT,
        StructureKind.PUT_CREDIT: _SHORT_PREMIUM_TEXT,
        StructureKind.CALL_CREDIT: _SHORT_PREMIUM_TEXT,
        StructureKind.IRON_CONDOR: _SHORT_PREMIUM_TEXT,
    }
)

# 5.6: event strings are CODE-GENERATED - they are not third-party text and may appear in the text-free state
EVENT_TEXT: Final[Mapping[str, str]] = MappingProxyType(
    {
        "fomc_decision": "major central-bank rate decision",
        "cpi": "major consumer-inflation data release",
        "nfp": "major monthly employment data release",
        "ex_dividend": "ex-dividend date for the underlying",
    }
)
EVENT_COVERAGE_TEXT: Final[Mapping[str, str]] = MappingProxyType(
    {
        "fomc_decision": "central-bank rate decisions",
        "cpi": "consumer-inflation data releases",
        "nfp": "monthly employment data releases",
        "ex_dividend": "ex-dividend dates for the underlying",
    }
)
_COVERAGE_PREFIX: Final = "scheduled events tracked: "
_LONG_SINGLES: Final[frozenset[StructureKind]] = frozenset({StructureKind.LONG_CALL, StructureKind.LONG_PUT})
_DEBIT_KINDS: Final[frozenset[StructureKind]] = frozenset(
    {StructureKind.LONG_CALL, StructureKind.LONG_PUT, StructureKind.CALL_DEBIT, StructureKind.PUT_DEBIT}
)
_MILLI_PER_CENT: Final = 10
_PPM: Final = 1_000_000

ValueBucket = dict[str, Any] | str


# ======================================================================================================================
# Small pure renderers
# ======================================================================================================================


def _vb(value: int | None, label: str) -> ValueBucket:
    """`{"value": ..., "bucket": ...}`, or the bare string `"unavailable"` when the feature is missing (5.5)."""
    if value is None or label == vocab.UNAVAILABLE:
        return vocab.UNAVAILABLE
    return {"value": value, "bucket": label}


def _em(tenths: int | None) -> ValueBucket:
    """An expected-move field: `{"value", "unit", "bucket"}`. Its integer is the one the outcome resolver uses (6.4)."""
    if tenths is None:
        return vocab.UNAVAILABLE
    return {"value": tenths, "unit": EM_UNIT_TEXT, "bucket": buckets.bucketize(tenths, buckets.EM)}


def _clip(value: float | None, lo: int, hi: int) -> int | None:
    return None if value is None else max(lo, min(hi, round(value)))


def _event_phrase(kind: str, sessions: int) -> str:
    text = EVENT_TEXT.get(kind, kind.replace("_", " "))
    return f"{text} in {sessions} session" + ("" if sessions == 1 else "s")


def _copy(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _copy(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_copy(item) for item in obj]
    return obj


def _reverse_keys(obj: Any) -> Any:
    """KEY_PERM (5.9): every dict rebuilt with REVERSED key order at every level; list ORDER is untouched."""
    if isinstance(obj, dict):
        return {key: _reverse_keys(value) for key, value in reversed(list(obj.items()))}
    if isinstance(obj, list):
        return [_reverse_keys(item) for item in obj]
    return obj


def _bucket_only(obj: Any, path: str) -> Any:
    """BUCKET_ONLY (5.9): every dict holding both `value` and `bucket` becomes its bucket string - except the
    `vol_surface.expected_move_*` dicts, whose integers define the evaluation thresholds the questions quote."""
    if isinstance(obj, dict):
        if "value" in obj and "bucket" in obj and not path.startswith(vocab.EXPECTED_MOVE_PREFIX):
            bucket = obj["bucket"]
            return bucket if isinstance(bucket, str) else _copy(bucket)
        return {key: _bucket_only(value, f"{path}.{key}" if path else key) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_bucket_only(item, path) for item in obj]
    return obj


# ======================================================================================================================
# StateBuilder
# ======================================================================================================================


class StateBuilder:
    """Implements `protocols.StateBuilderP` (3.6). One instance per run; it holds no mutable state."""

    def __init__(self, cfg: Config, mask_terms: MaskTerms, *, news_resolved: bool) -> None:
        self.cfg = cfg
        self.mask_terms = mask_terms
        self.news_resolved = news_resolved
        self._masked = not cfg.state.unmasked  # `state.unmasked` is the leakage diagnostic only; it can never trade

    # --- entry --------------------------------------------------------------------------------------------------------

    def entry(self, view: MarketView, underlying: str) -> BuiltState | None:
        """`state.v1.entry` (5.6), or None when a required feature is unavailable (`dq:insufficient`, 5.3)."""
        hold = self.cfg.dte.hold_horizon_sessions
        features = compute_features(view, underlying, hold, data=self.cfg.data)
        if not features.required_ok:
            return None
        events, inside = self._events_block(view, underlying, hold)
        trend = buckets.trend_direction(features.ref, features.ma20, features.ma50, features.ma20_prev)
        iv_rank_label = buckets.bucketize(features.iv_rank, buckets.PCTL5_RANGE)
        iv_rv_label = buckets.bucketize(features.iv_rv, buckets.IV_RV)
        dist_label = buckets.bucketize(features.dist_ma20_atr, buckets.DIST_ATR)
        state: dict[str, Any] = {
            "schema": vocab.STATE_SCHEMA["entry"],
            "context": {
                "underlying_alias": self._alias(underlying),
                "underlying_kind": self._kind(underlying),
                "decision_session": DECISION_SESSION_TEXT,
                "holding_window_sessions": _vb(hold, buckets.bucketize(hold, buckets.HOLD)),
            },
            "market": self._market_block(features),
            "underlying": self._underlying_block(features, trend),
            "vol_surface": self._vol_surface_block(features, iv_rank_label, iv_rv_label, expected_moves=True),
            "events": events,
        }
        facts = EntryFacts(
            trend_code=buckets.code_of(trend),
            iv_rank_code=buckets.code_of(iv_rank_label),
            iv_rv_code=buckets.code_of(iv_rv_label),
            dist_code=buckets.code_of(dist_label),
            news_enabled=self._news_on(view, underlying),
            news_count=0,
            news_recent_count=0,
            spot=_require(features.ref, "ref"),
            iv30_bp=_require(features.iv30_bp, "iv30_bp"),
            em_hold_tenths=_require(features.em_hold_tenths, "em_hold_tenths"),
            events_in_window=inside,
            thesis=_thesis(trend, iv_rv_label, iv_rank_label, inside),
        )
        return self._finish(state, view, underlying, facts, ref=features.ref, source=features)

    def entry_text(self, view: MarketView, underlying: str, base: BuiltState) -> BuiltState | None:
        """`state.v1.entry_text` = the base object with `news_status` and `news` appended (5.6).

        None when news is resolved off or the archive does not cover this session - then no `entry_text` request exists
        at all and the DECISION payload records `text: "off"` / `"no_archive"`.
        """
        if not isinstance(base.facts, EntryFacts):
            raise InvariantError("state.entry_text: `base` must be the entry state of the same underlying")
        if not self._news_on(view, underlying):
            return None
        lists, stats = self._news(view, underlying, since=None)
        state = _append_after_events(
            base.state,
            vocab.STATE_SCHEMA["entry_text"],
            {
                "news_status": "present" if stats.kept else "none_in_window",
                "news": dict(lists),
            },
        )
        state = self._trim_to_max_chars(state, lists, "news")
        facts = msgspec.structs.replace(base.facts, news_enabled=True, news_count=stats.kept, news_recent_count=stats.kept_recent)
        return self._finish(state, view, underlying, facts, ref=base.facts.spot, source=base.provenance, stats=stats)

    # --- manage -------------------------------------------------------------------------------------------------------

    def manage(self, view: MarketView, pos: Position) -> BuiltState:
        """`state.v1.manage` (5.7). Never None: a position must be manageable even on a poor data day."""
        underlying = pos.structure.underlying
        features = compute_features(view, underlying, self.cfg.dte.hold_horizon_sessions, data=self.cfg.data)
        horizon = _time_exit_horizon(view.calendar, view.session, pos.structure, self.cfg)
        events, _inside = self._events_block(view, underlying, horizon)
        trend = buckets.trend_direction(features.ref, features.ma20, features.ma50, features.ma20_prev)
        iv_rank_label = buckets.bucketize(features.iv_rank, buckets.PCTL5_RANGE)
        iv_rv_label = buckets.bucketize(features.iv_rv, buckets.IV_RV)
        sigma = _expiry_sigma(view, underlying, pos.structure)
        held = view.calendar.sessions_between(pos.open_key.session, view.session)
        dte = (pos.structure.last_session - view.session).days
        pnl_label = _pnl_label(pos)
        short_label = _short_distance(pos.structure, features.ref, sigma)
        breakeven_label = _breakeven_distance(pos.structure, pos.entry.open_mid_at_decision, features.ref, sigma)
        move_value, move_label = _move_since_entry(pos, features.ref)
        state: dict[str, Any] = {
            "schema": vocab.STATE_SCHEMA["manage"],
            "context": {
                "underlying_alias": self._alias(underlying),
                "underlying_kind": self._kind(underlying),
                "decision_session": DECISION_SESSION_TEXT,
            },
            "position": {
                "structure": pos.structure.kind.value,
                "directional_exposure": DIRECTIONAL_EXPOSURE[pos.structure.kind],
                "vol_exposure": VOL_EXPOSURE[pos.structure.kind],
                "entry_thesis": pos.entry.entry_thesis,
                "sessions_held": _vb(held, buckets.bucketize(held, buckets.HELD4)),
                "time_to_expiry": _vb(dte, buckets.bucketize(dte, buckets.DTE4)),
                "pnl": pnl_label,
                "short_strike_distance": short_label,
                "price_vs_breakeven": breakeven_label,
            },
            "changes_since_entry": {
                "trend_at_entry": pos.entry.entry_codes.get("trend", vocab.UNAVAILABLE),
                "trend_now": buckets.code_of(trend),
                "iv_vs_realized_at_entry": pos.entry.entry_codes.get("iv_vs_realized", vocab.UNAVAILABLE),
                "iv_vs_realized_now": buckets.code_of(iv_rv_label),
                "iv_change_since_entry": _iv_change_since_entry(features.iv30_bp, pos.entry.entry_iv30_bp),
                "underlying_move_since_entry": _vb(move_value, move_label),
            },
            "market": self._market_block(features),
            "underlying": self._underlying_block(features, trend),
            "vol_surface": self._vol_surface_block(features, iv_rank_label, iv_rv_label, expected_moves=False),
            "events": events,
        }
        facts = _manage_facts(pos, self.cfg, move_label, short_label, news_count=0, news_recent_count=0)
        return self._finish(state, view, underlying, facts, ref=features.ref, source=features)

    def manage_text(self, view: MarketView, pos: Position, base: BuiltState) -> BuiltState | None:
        """`state.v1.manage_text` = the base object with `news_since_entry` appended (5.7).

        None when news is off, the archive does not cover this session, or no item since the opening session survives.
        """
        if not isinstance(base.facts, ManageFacts):
            raise InvariantError("state.manage_text: `base` must be the manage state of the same position")
        underlying = pos.structure.underlying
        if not self._news_on(view, underlying):
            return None
        open_ts = view.calendar.open_close(pos.open_key.session)[0]
        lists, stats = self._news(view, underlying, since=open_ts)
        if not stats.kept:
            return None
        state = _append_after_events(base.state, vocab.STATE_SCHEMA["manage_text"], {"news_since_entry": dict(lists)})
        state = self._trim_to_max_chars(state, lists, "news_since_entry")
        facts = msgspec.structs.replace(base.facts, news_count=stats.kept, news_recent_count=stats.kept_recent)
        return self._finish(state, view, underlying, facts, ref=_identity_ref(base), source=base.provenance, stats=stats)

    # --- variants -----------------------------------------------------------------------------------------------------

    def variant(self, state: dict[str, Any], v: Variant) -> dict[str, Any]:
        """The 5.9 state renderers. `OPT_PERM` permutes a question's options, not the state, so it is a copy here."""
        if v is Variant.KEY_PERM:
            rendered = _reverse_keys(state)
        elif v is Variant.BUCKET_ONLY:
            rendered = _bucket_only(state, "")
        elif v in (Variant.BASE, Variant.OPT_PERM):
            rendered = _copy(state)
        else:  # pragma: no cover - Variant is a closed enum
            raise InvariantError(f"state.variant: unknown variant {v!r}")
        if not isinstance(rendered, dict):  # pragma: no cover - a state is always a dict
            raise InvariantError("state.variant: a state must be a dict")
        return rendered

    # --- blocks -------------------------------------------------------------------------------------------------------

    def _alias(self, underlying: str) -> str:
        alias = self.cfg.universe.alias.get(underlying)
        if not alias:
            raise InvariantError(f"state: no `universe.alias` for {underlying!r} - the raw ticker may never reach a state (INV-15)")
        return alias

    def _kind(self, underlying: str) -> str:
        kind = self.cfg.universe.kind.get(underlying)
        if not kind:
            raise InvariantError(f"state: no `universe.kind` for {underlying!r}")
        return kind

    def _market_block(self, f: FeatureSet) -> dict[str, Any]:
        return {
            "as_of": MARKET_AS_OF_TEXT,
            "vol_index_pctile_1y": _vb(f.vix_pctile, buckets.bucketize(f.vix_pctile, buckets.PCTL5_PCTILE)),
            "vol_index_change_1w": buckets.bucketize(f.vix_chg_1w, buckets.CHANGE5),
            "vol_term_structure": buckets.bucketize(f.vix_term, buckets.TERM4_MARKET),
            "near_term_stress": buckets.bucketize(f.vix_near, buckets.NEAR3),
            "vol_of_vol": buckets.bucketize(f.vvix_pctile, buckets.PCTL3),
            "tail_skew_index": buckets.bucketize(f.skewidx_pctile, buckets.PCTL3),
        }

    def _underlying_block(self, f: FeatureSet, trend: str) -> dict[str, Any]:
        return {
            "trend": {
                "direction": trend,
                "strength": buckets.bucketize(f.trend_z, buckets.TREND_STRENGTH),
            },
            "momentum": {
                "distance_from_20d_avg_in_atr": _vb(_clip(f.dist_ma20_atr, -9, 9), buckets.bucketize(f.dist_ma20_atr, buckets.DIST_ATR)),
                "consecutive_closes": _vb(_clip(f.streak, -20, 20), buckets.bucketize(f.streak, buckets.STREAK)),
            },
            "range": {
                "realized_vol_20d_pctile_1y": _vb(f.rv20_pctile, buckets.bucketize(f.rv20_pctile, buckets.PCTL5_PCTILE)),
                "realized_vol_change": buckets.bucketize(f.rv_change, buckets.RV_CHANGE),
                "gap_today": buckets.bucketize(f.gap_sigma, buckets.SIGMA5_GAP),
                "move_today": buckets.bucketize(f.move_sigma, buckets.SIGMA5_MOVE),
            },
            "levels": {"distance_from_52w_high": buckets.bucketize(f.dd_52w, buckets.DD_52W)},
        }

    def _vol_surface_block(self, f: FeatureSet, iv_rank_label: str, iv_rv_label: str, *, expected_moves: bool) -> dict[str, Any]:
        block: dict[str, Any] = {
            "iv_rank_1y": _vb(f.iv_rank, iv_rank_label),
            "iv_vs_realized": iv_rv_label,
            "iv_change_1w": buckets.bucketize(f.iv_chg_1w, buckets.CHANGE5),
            "term_structure": buckets.bucketize(f.iv_term, buckets.TERM4_SURFACE),
            "skew": buckets.bucketize(f.skew_pctile, buckets.PCTL3_SKEW),
        }
        if expected_moves:
            block["expected_move_1_session"] = _em(f.em_1_tenths)
            block["expected_move_5_sessions"] = _em(f.em_5_tenths)
            block["expected_move_holding_window"] = _em(f.em_hold_tenths)
        return block

    def _events_block(self, view: MarketView, underlying: str, horizon_sessions: int) -> tuple[dict[str, Any], int]:
        """The 5.6 events block plus the count that becomes `EntryFacts.events_in_window`."""
        calendar = view.calendar
        session = view.session
        inside: list[str] = []
        next_session: list[str] = []
        if horizon_sessions >= 1:
            start = calendar.next_session(session, 1)
            end = calendar.next_session(session, horizon_sessions)
            for event in _ordered(view.events(start, end, underlying)):
                inside.append(_event_phrase(event.kind, calendar.sessions_between(session, event.event_date)))
            next_session = [_event_phrase(event.kind, 1) for event in _ordered(view.events(start, start, underlying))]
        block = {
            "coverage": _coverage_text(view.event_coverage()),
            "inside_holding_window": inside,
            "next_session": next_session,
            "earnings": EARNINGS_TEXT,
        }
        return block, len(inside)

    # --- news ---------------------------------------------------------------------------------------------------------

    def _news_on(self, view: MarketView, underlying: str) -> bool:
        """The resolved news flag AND archive coverage for this session (`EntryFacts.news_enabled`, 5.6)."""
        return self.news_resolved and view.news_covered(underlying)

    def _cutoff(self, view: MarketView) -> datetime:
        """The previous session's decision time (5.6): the `dec` offset from its close, or its close for an `eod` view."""
        previous = view.calendar.prev_session(view.session)
        if view.key.slot.value == "eod":
            return view.calendar.open_close(previous)[1]
        return view.calendar.offset_from_close(previous, self.cfg.cadence.decide_offset_min)

    def _news(
        self, view: MarketView, underlying: str, *, since: datetime | None
    ) -> tuple[dict[str, list[dict[str, Any]]], textmask.NewsStats]:
        """Run the 5.8 pipeline over the items this view serves. `since` = the manage block's opening-session open."""
        lookback = self.cfg.news.lookback_hours
        if since is not None:
            elapsed = math.ceil((view.as_of - since).total_seconds() / 3600.0)
            lookback = max(lookback, elapsed)
        items = tuple(view.news(underlying, lookback))
        if since is not None:
            items = tuple(item for item in items if item.created_at >= since)
        return textmask.prepare_news(items, view.as_of, self._cutoff(view), self.cfg.news, self.mask_terms, self.cfg.universe.underlyings)

    def _trim_to_max_chars(self, state: dict[str, Any], lists: Mapping[str, list[dict[str, Any]]], key: str) -> dict[str, Any]:
        """5.8 step 6: after the block fits `news.max_total_chars`, drop oldest until the WHOLE state fits `state.max_chars`."""
        while len(dumps_ordered(state)) > self.cfg.state.max_chars and textmask.drop_oldest(lists):
            state[key] = dict(lists)
            if "news_status" in state:
                state["news_status"] = "present" if (lists[textmask.RECENT_KEY] or lists[textmask.EARLIER_KEY]) else "none_in_window"
        return state

    # --- assembly -----------------------------------------------------------------------------------------------------

    def _finish(
        self,
        state: dict[str, Any],
        view: MarketView,
        underlying: str,
        facts: EntryFacts | ManageFacts,
        *,
        ref: Cents | None,
        source: FeatureSet | Provenance,
        stats: textmask.NewsStats | None = None,
    ) -> BuiltState:
        """Render (`state.render`), gate (INV-15 / 5.9), hash and wrap with the provenance sidecar.

        `source` carries the two audit-only provenance members: the freshly computed `FeatureSet` for a base state, the
        base state's own `Provenance` for a `_text` state (the features are identical - the text builders only append).
        """
        if self.cfg.state.render == "bucket_only":
            state = self.variant(state, Variant.BUCKET_ONLY)
        if self.cfg.state.unmasked:  # leakage diagnostic only: the run is flagged and can never trade (5.6)
            state = _with_identity(state, underlying, view.session, ref)
        ensure_state_safe(
            state,
            masked=self._masked,
            underlyings=self.cfg.universe.underlyings,
            max_chars=self.cfg.state.hard_max_chars,
        )
        state_hash = sha256_hex(dumps_ordered(state))
        provenance = Provenance(
            decision_id="",  # the cycle stamps it (2.10); nothing hashed depends on it
            as_of=view.as_of,
            real_symbol=underlying,
            state_sha256=state_hash,
            evidence_tier=EvidenceTier.NONE if view.fidelity is Fidelity.SYNTHETIC else EvidenceTier.B,
            data_fidelity=view.fidelity,
            inputs=view.touched(),
            news_ids=() if stats is None else stats.ids,
            news_dropped=0 if stats is None else stats.dropped,
            news_hostile_dropped=0 if stats is None else stats.hostile_dropped,
            mask_version=self.mask_terms.version,
            iv_hist_proxy_pct=source.iv_hist_proxy_pct,
            spot_measure=_spot_measure(view, underlying),
            raw_features=source.raw() if isinstance(source, FeatureSet) else dict(source.raw_features),
            request_id=None,
            input_tokens=None,
            latency_ms=None,
            ledgered_wall=view.as_of,  # placeholder: the ledger writer stamps the real wall clock (never hashed)
        )
        return BuiltState(state=state, state_hash=state_hash, provenance=provenance, facts=facts)


# ======================================================================================================================
# Module-level helpers (pure)
# ======================================================================================================================


def _identity_ref(base: BuiltState) -> Cents | None:
    """The `ref` the base state's identity block used (leakage diagnostic only; None when it was unavailable)."""
    identity = base.state.get("identity")
    if not isinstance(identity, dict):
        return None
    spot = identity.get("spot")
    if not isinstance(spot, str) or not spot.endswith(" dollars"):
        return None
    dollars, _, cents = spot[: -len(" dollars")].partition(".")
    return int(dollars) * 100 + int(cents)


def _require(value: int | None, name: str) -> int:
    if value is None:  # pragma: no cover - `required_ok` was checked first
        raise InvariantError(f"state: required feature {name} is unavailable")
    return value


def _ordered(events: Sequence[ScheduledEvent]) -> list[ScheduledEvent]:
    """Deterministic order regardless of the provider: by date, then kind, then underlying."""
    return sorted(events, key=lambda event: (event.event_date, event.kind, event.underlying or ""))


def _coverage_text(kinds: Sequence[str]) -> str:
    if not kinds:
        return f"{_COVERAGE_PREFIX}none"
    phrases = [EVENT_COVERAGE_TEXT.get(kind, kind.replace("_", " ")) for kind in kinds]
    if len(phrases) == 1:
        return f"{_COVERAGE_PREFIX}{phrases[0]} only"
    return _COVERAGE_PREFIX + ", ".join(phrases)


def _thesis(trend: str, iv_rv_label: str, iv_rank_label: str, events_in_window: int) -> str:
    """5.7 `entry_thesis`, written once at entry from bucket CODES only (never from prices, dates or text)."""
    phrase = (
        "no tracked event inside the holding window"
        if events_in_window == 0
        else f"{events_in_window} tracked event(s) inside the holding window"
    )
    return (
        f"opened with trend {buckets.code_of(trend)}, implied volatility {buckets.code_of(iv_rv_label)} versus realized, "
        f"iv rank {buckets.code_of(iv_rank_label)}, {phrase}"
    )


def _append_after_events(base: Mapping[str, Any], schema: str, extra: Mapping[str, Any]) -> dict[str, Any]:
    """The `_text` states are the SAME object with their schema string changed and the news key(s) appended (5.6 / 5.7).

    The optional `identity` block of the leakage diagnostic stays last, so the news keys follow `events` in every case.
    """
    out: dict[str, Any] = {}
    identity = None
    for key, value in base.items():
        if key == "identity":
            identity = _copy(value)
            continue
        out[key] = schema if key == "schema" else _copy(value)
    out.update({key: _copy(value) for key, value in extra.items()})
    if identity is not None:
        out["identity"] = identity
    return out


def _with_identity(state: dict[str, Any], underlying: str, session: date, ref: Cents | None) -> dict[str, Any]:
    """5.6: `state.unmasked = true` appends the identity block; `ensure_state_safe(masked=False)` lets it through."""
    out = dict(state)
    out["identity"] = {
        "ticker": underlying,
        "date": session.isoformat(),
        "spot": "unavailable" if ref is None else f"{ref // 100}.{ref % 100:02d} dollars",
    }
    return out


def _spot_measure(view: MarketView, underlying: str) -> str:
    try:
        return view.chain(underlying).spot_measure
    except DataUnavailable:
        return "unknown"


def _time_exit_horizon(calendar: Calendar, session: date, structure: Structure, cfg: Config) -> int:
    """Sessions from `session` until the code-side time exit of 9.4 row 4 (the manage state's event window, 5.7)."""
    days = cfg.dte.time_exit_short_premium if structure.kind in SHORT_PREMIUM else cfg.dte.time_exit_long_premium
    target = structure.last_session - timedelta(days=days)
    exit_session = target if calendar.is_session(target) else calendar.next_session(target)
    if exit_session > structure.last_session:
        exit_session = structure.last_session
    return max(0, calendar.sessions_between(session, exit_session))


def _expiry_sigma(view: MarketView, underlying: str, structure: Structure) -> float | None:
    """`sqrt(total variance to the structure's last trading day)` - the "expected move" both position distances use (5.5)."""
    try:
        daily = view.daily(underlying, 2)
    except DataUnavailable:
        return None
    if daily.empty or "atm_term_json" not in daily.columns:
        return None
    term = parse_atm_term(daily["atm_term_json"].iloc[-1])
    close = view.calendar.open_close(structure.last_session)[1]
    found = total_variance_at(term, trading_time(view.calendar, view.as_of, close))
    if found is None or found[0] <= 0.0:
        return None
    return math.sqrt(found[0])


def _strike_cents(strike_milli: int) -> float:
    return strike_milli / _MILLI_PER_CENT


def _short_distance(structure: Structure, ref: Cents | None, sigma: float | None) -> str | None:
    """SHORT_DIST (5.5): the NEAREST short strike's log distance on its OTM side, in expected moves to that expiry."""
    shorts = structure.short_legs
    if not shorts:
        return None  # a long single or a debit structure with no short leg: the field is null
    if ref is None or ref <= 0 or sigma is None or sigma <= 0.0:
        return vocab.UNAVAILABLE
    distances: list[float] = []
    for leg in shorts:
        strike = _strike_cents(leg.contract.strike_milli)
        if strike <= 0:
            continue
        # a short put is OTM BELOW the spot, a short call ABOVE it: positive = the strike is still out of the money
        moneyness = math.log(ref / strike) if leg.contract.right is Right.PUT else math.log(strike / ref)
        distances.append(moneyness / sigma)
    if not distances:
        return vocab.UNAVAILABLE
    return buckets.bucketize(min(distances), buckets.SHORT_DIST)


def _breakeven_distance(structure: Structure, open_mid: int, ref: Cents | None, sigma: float | None) -> str | None:
    """BREAKEVEN (5.5): signed distance past the MID-price breakeven in the profitable direction; null for credits."""
    if structure.kind not in _DEBIT_KINDS:
        return None
    if ref is None or ref <= 0 or sigma is None or sigma <= 0.0:
        return vocab.UNAVAILABLE
    levels = structmath.breakevens(structure.kind, structure.legs, open_mid)
    if not levels or levels[0] <= 0:
        return vocab.UNAVAILABLE
    breakeven = levels[0]
    bullish = STRUCTURE_DIRECTION[structure.kind] is Direction.BULLISH
    signed = math.log(ref / breakeven) if bullish else math.log(breakeven / ref)
    return buckets.bucketize(signed / sigma, buckets.BREAKEVEN)


def _pnl_label(pos: Position) -> str:
    """PNL (5.5): mid-to-mid and path-independent - `open_mid_at_decision` and the current mid, never our fill."""
    open_mid = pos.entry.open_mid_at_decision
    pnl_mid = (-open_mid - pos.mid_value) * structmath.MULTIPLIER
    widths = pos.structure.wing_widths
    loss_base = structmath.max_loss_pc(pos.structure.kind, widths, open_mid, 0)
    long_premium = pos.structure.kind in _LONG_SINGLES
    profit = structmath.max_profit_pc(pos.structure.kind, widths, open_mid)
    gain_base = open_mid * structmath.MULTIPLIER if profit is None else profit
    return buckets.pnl_bucket(pnl_mid=pnl_mid, gain_base=gain_base, loss_base=loss_base, long_premium=long_premium)


def _iv_change_since_entry(iv30_bp: Bp | None, entry_iv30_bp: Bp) -> str:
    if iv30_bp is None or entry_iv30_bp <= 0:
        return vocab.UNAVAILABLE
    return buckets.bucketize(iv30_bp / entry_iv30_bp - 1.0, buckets.CHANGE5)


def _move_since_entry(pos: Position, ref: Cents | None) -> tuple[int | None, str]:
    """MOVE_SINCE_ENTRY (5.5): `ln(ref / entry_spot)` in entry-time holding-window expected moves, signed so that
    positive favours the position; the condor (neutral) uses `-abs(x)`."""
    entry_spot = pos.entry.entry_spot
    em = pos.entry.entry_em_hold_tenths / 1000.0
    if ref is None or ref <= 0 or entry_spot <= 0 or em <= 0.0:
        return None, vocab.UNAVAILABLE
    x = math.log(ref / entry_spot) / em
    direction = STRUCTURE_DIRECTION[pos.structure.kind]
    if direction is Direction.BEARISH:
        x = -x
    elif direction is Direction.NEUTRAL:
        x = -abs(x)
    return round(x), buckets.bucketize(x, buckets.MOVE_SINCE_ENTRY)


def _manage_facts(
    pos: Position, cfg: Config, move_label: str, short_label: str | None, *, news_count: int, news_recent_count: int
) -> ManageFacts:
    """`ManageFacts` (2.6): the code-side facts of 7.7. Unlike the STATE they use the actual fill (the rules compare
    them with our own risk limits); the state itself stays path-independent."""
    band = headline_band(cfg)
    pnl_headline = (-pos.open_net.get(band) - pos.liq_value) * structmath.MULTIPLIER * pos.qty
    frac_loss = 0
    if pnl_headline < 0 and pos.max_loss > 0:
        frac_loss = min(_PPM, round(_PPM * (-pnl_headline) / pos.max_loss))
    return ManageFacts(
        pnl_headline=pnl_headline,
        pnl_frac_loss_ppm=frac_loss,
        move_code=buckets.code_of(move_label),
        short_dist_code=None if short_label is None else buckets.code_of(short_label),
        news_count=news_count,
        news_recent_count=news_recent_count,
    )
