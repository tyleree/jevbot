"""DecisionRules (DESIGN.md section 7): gates, cross-checks, vetoes, composite, tiers, hysteresis, perturbation policy.

Pure functions of their arguments and of `[rules]`: they can only say **no**, pick from closed enums, and choose one of four size
tiers. They read only question ids in `vocab.RULES_READABLE_IDS` (`TRADING_IDS | TEXT_IDS | MANAGE_IDS | MANAGE_TEXT_IDS`);
evaluation answers are invisible here (INV-23, `tests/guards/test_no_eval_in_rules.py`). No IO, no clock, no network.

Entry (7.2): the steps run in order; the FIRST failing step is the decision's reason for the funnel of 12.2, and every later step
that can still be evaluated is evaluated and logged too, so `EntryDecision.reasons` carries every failing code in evaluation order.
Steps 0 (cycle-wide decider health) and 1 (data quality) happen before an answer exists: the cycle builds those decisions with
`no_trade()` below. Text (7.4, 7.5, INV-16): text answers appear only as vetoes and in the rank term; nothing here lets a text
answer relax a gate, raise `S_core`, raise a tier or alone yield `enter`.

Management (7.7): the code-side hard exit always wins; a decider outage falls back to the code default (`code_default`); the
hysteresis latch is driven by the text-free exit pressure alone; a text reading can close a position ONLY with code-side
market-data confirmation (`text_confirmed`) - unconfirmed adverse text only advances the `watch_text` alert counter.

A `raw_sum` outside `[RAW_SUM_LO, RAW_SUM_HI]` (7.1) makes a Choice read as its no-match label (gate failed), a Score read
conservatively (`risk.environment`: the hostile level; `pos.short_strike_threat`: UNCERTAIN, i.e. the discretionary zone), and is
never silently accepted.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from jevbot import config as _config
from jevbot import vocab
from jevbot.config import Config, GateConfig, RulesConfig
from jevbot.errors import DeciderError, InvariantError
from jevbot.types import (
    MAPPING,
    SHORT_PREMIUM,
    STRUCTURE_DIRECTION,
    STRUCTURE_STANCE,
    Answer,
    ChoiceAns,
    DecisionResult,
    Direction,
    EntryDecision,
    EntryFacts,
    ExitReason,
    ManageDecision,
    ManageFacts,
    NoulAns,
    Position,
    Ppm,
    ScoreAns,
    StructureKind,
    Tri,
    Variant,
    VolStance,
)

__all__ = [
    "ANOMALY_TEXT_VETO_ON_EMPTY_NEWS",
    "PPM",
    "RAW_SUM_HI",
    "RAW_SUM_LO",
    "DecisionRules",
    "entry_anomalies",
    "no_trade",
    "text_watch_alert",
    "tri_band",
]

PPM: Final = 1_000_000
RAW_SUM_LO: Final = 0.98  # 7.1: a renormalised answer whose pre-normalisation sum lies outside this band is not trusted
RAW_SUM_HI: Final = 1.02
ANOMALY_TEXT_VETO_ON_EMPTY_NEWS: Final = "text_veto_nonclear_on_empty_news"  # 7.4 (vocab.KNOWN_ANOMALY_TYPES)

# --- question ids this module reads (every literal is asserted below to be in vocab.RULES_READABLE_IDS; INV-23) -----------------
_Q_REGIME: Final = "regime.market"
_Q_DIRECTION: Final = "under.direction"
_Q_STRETCHED: Final = "under.stretched"
_Q_VOL_STANCE: Final = "vol.stance"
_Q_VOL_EVENT: Final = "vol.explained_by_event"
_Q_FIT: Final = "fit.structure_family"
_Q_ENV: Final = "risk.environment"
_Q_TEXT_MATERIAL: Final = "text.material_present"
_Q_TEXT_NEGATIVE: Final = "text.clearly_negative"
_Q_TEXT_POSITIVE: Final = "text.clearly_positive"
_Q_TEXT_PENDING: Final = "text.pending_binary"
_Q_TEXT_STRESS: Final = "text.market_stress"
_Q_THESIS: Final = "pos.thesis_invalidated"
_Q_THREAT: Final = "pos.short_strike_threat"
_Q_ACTION: Final = "pos.action"
_Q_ADVERSE: Final = "pos.adverse_news_since_entry"
_Q_PENDING_SINCE: Final = "pos.pending_binary_since_entry"

_READ_IDS: Final[frozenset[str]] = frozenset(
    {
        _Q_REGIME,
        _Q_DIRECTION,
        _Q_STRETCHED,
        _Q_VOL_STANCE,
        _Q_VOL_EVENT,
        _Q_FIT,
        _Q_ENV,
        _Q_TEXT_MATERIAL,
        _Q_TEXT_NEGATIVE,
        _Q_TEXT_POSITIVE,
        _Q_TEXT_PENDING,
        _Q_TEXT_STRESS,
        _Q_THESIS,
        _Q_THREAT,
        _Q_ACTION,
        _Q_ADVERSE,
        _Q_PENDING_SINCE,
    }
)
if not _READ_IDS <= vocab.RULES_READABLE_IDS:  # INV-23, checked at import
    raise InvariantError(f"rules.py reads question ids outside the trading vocabulary: {sorted(_READ_IDS - vocab.RULES_READABLE_IDS)}")

# --- labels (vocab.CHOICE_LABELS) --------------------------------------------------------------------------------------------
_REGIME_NO_MATCH: Final = vocab.NO_MATCH_LABEL[_Q_REGIME]  # unclear_or_transition
_DIRECTION_NO_MATCH: Final = vocab.NO_MATCH_LABEL[_Q_DIRECTION]  # conflicting_signals
_VOL_NO_MATCH: Final = vocab.NO_MATCH_LABEL[_Q_VOL_STANCE]  # unclear
_FIT_NO_MATCH: Final = vocab.NO_MATCH_LABEL[_Q_FIT]  # no_trade
_ACTION_NO_MATCH: Final = vocab.NO_MATCH_LABEL[_Q_ACTION]  # unclear
_ACTION_HOLD: Final = "hold"
_ACTION_TAKE_PROFIT: Final = "take_profit"
_ACTION_CUT_LOSS: Final = "close_to_cut_loss"
_DIRECTION_VALUES: Final[frozenset[str]] = frozenset(d.value for d in Direction)
_STANCE_VALUES: Final[frozenset[str]] = frozenset(s.value for s in VolStance)

# 7.5 REGIME_OK
_REGIME_OK: Final[Mapping[Direction, tuple[str, ...]]] = {
    Direction.BULLISH: ("trending_up_calm", "trending_up_volatile"),
    Direction.BEARISH: ("orderly_downtrend", "disorderly_selloff"),
    Direction.NEUTRAL: ("range_bound_calm", "range_bound_volatile"),
}
# 7.3 code-side cross-checks on the state's own bucket codes
_TREND_UP: Final = "up"
_TREND_DOWN: Final = "down"
_TREND_RANGE: Final[frozenset[str]] = frozenset({"flat", "mixed"})
_IV_RICH: Final[frozenset[str]] = frozenset({"iv_rich", "iv_very_rich"})
_IV_CHEAP_RANK: Final[frozenset[str]] = frozenset({"very_low", "low"})
_LONG_SINGLES: Final[frozenset[StructureKind]] = frozenset({StructureKind.LONG_CALL, StructureKind.LONG_PUT})
# 7.7 market-data confirmation of an adverse text reading
_ADVERSE_MOVES: Final[frozenset[str]] = frozenset({"adverse", "strongly_adverse"})
_TEXT_CONFIRM_LOSS_PPM: Final = 250_000
_CODE_DEFAULT_CLOSE_CODES: Final[frozenset[str]] = frozenset({"breached", "at_strike"})
_ENV_LEVELS: Final = 4  # risk.environment has levels 0..3 (vocab.SCORE_LEVELS)

_ACTION_ENTER: Final = "enter"
_ACTION_NO_TRADE: Final = "no_trade"
_MANAGE_HOLD: Final = "hold"
_MANAGE_CLOSE: Final = "close"
_SOURCE_HARD: Final = "hard_exit"
_SOURCE_JEV: Final = "jev"
_SOURCE_CODE_DEFAULT: Final = "code_default"


# ======================================================================================================================
# small pure helpers
# ======================================================================================================================


def _ppm(x: float) -> Ppm:
    """Probabilities and scores enter the ledger as integer parts-per-million (Conventions)."""
    return round(x * PPM)


def _threshold_ppm(x: float) -> Ppm:
    return round(x * PPM)


def _in_band(raw_sum: float) -> bool:
    return RAW_SUM_LO <= raw_sum <= RAW_SUM_HI


def _round_half_up(x: float) -> int:
    return int(x + 0.5) if x >= 0.0 else -int(-x + 0.5)


def tri_band(p: float, cfg: RulesConfig) -> Tri:
    """7.4: for a bad-is-TRUE Noul, `p > veto_hi` = VETO; `veto_lo <= p <= veto_hi` = UNCERTAIN; `p < veto_lo` = CLEAR."""
    if p > cfg.veto_hi:
        return Tri.VETO
    if p >= cfg.veto_lo:
        return Tri.UNCERTAIN
    return Tri.CLEAR


def _answer(result: DecisionResult, qid: str) -> Answer:
    if qid not in _READ_IDS:  # INV-23: a code path can never read anything but the trading ids
        raise InvariantError(f"rules.py may not read question {qid!r}")
    try:
        return result.answers[qid]
    except KeyError:
        raise InvariantError(f"decision {result.decision_id}: {result.kind.value} result has no answer for {qid!r}") from None


def _choice(result: DecisionResult, qid: str) -> ChoiceAns:
    ans = _answer(result, qid)
    if not isinstance(ans, ChoiceAns):
        raise InvariantError(f"{qid!r} must be a Choice answer, got {type(ans).__name__}")
    return ans


def _noul(result: DecisionResult, qid: str) -> NoulAns:
    ans = _answer(result, qid)
    if not isinstance(ans, NoulAns):
        raise InvariantError(f"{qid!r} must be a Noul answer, got {type(ans).__name__}")
    return ans


def _score(result: DecisionResult, qid: str) -> ScoreAns:
    ans = _answer(result, qid)
    if not isinstance(ans, ScoreAns):
        raise InvariantError(f"{qid!r} must be a Score answer, got {type(ans).__name__}")
    return ans


def _prob(ans: ChoiceAns, label: str) -> float:
    try:
        return ans.probs[label]
    except KeyError:
        raise InvariantError(f"Choice answer has no probability for label {label!r}") from None


def _tier_lookup(x: float, table: Sequence[tuple[float, float]]) -> float:
    """7.6: the first `b` whose `a <= x`, else 0 (the tables are validated descending at config load)."""
    for a, b in table:
        if x >= a:
            return b
    return 0.0


def _tier_lookup_ppm(x_ppm: Ppm, table: Sequence[tuple[float, float]]) -> float:
    for a, b in table:
        if x_ppm >= _threshold_ppm(a):
            return b
    return 0.0


def _rank_ppm(score_core_ppm: Ppm, news_align_ppm: Ppm, w: float) -> Ppm:
    """7.5: `S_rank = (1 - w) * S_core + w * news_align` on the ledgered ppm values (ranking only)."""
    return round((1.0 - w) * score_core_ppm + w * news_align_ppm)


def _gate_codes(ans: ChoiceAns, gate: GateConfig, prefix: str, no_match: str, no_match_code: str, valid: frozenset[str]) -> list[str]:
    """7.2 steps 3 / 4 / 7: every failing sub-condition of a Choice gate, in the order the reason vocabulary lists them."""
    codes: list[str] = []
    if ans.top == no_match or ans.top not in valid or not _in_band(ans.raw_sum):
        codes.append(f"{prefix}:{no_match_code}")
    if ans.p_top < gate.p_top:
        codes.append(f"{prefix}:p_top")
    if ans.margin < gate.margin:
        codes.append(f"{prefix}:margin")
    return codes


# ======================================================================================================================
# 7.2-7.6: one evaluation of a text-free result (+ the text result) against the facts
# ======================================================================================================================


@dataclass
class _Eval:
    reasons: list[str] = field(default_factory=list)
    kind: StructureKind | None = None
    score_core_ppm: Ppm = 0
    news_align_ppm: Ppm = PPM // 2
    tier_ppm: Ppm = 0
    features_ppm: dict[str, Ppm] = field(default_factory=dict)

    @property
    def action(self) -> str:
        return _ACTION_NO_TRADE if self.reasons else _ACTION_ENTER


def entry_anomalies(text: DecisionResult | None, facts: EntryFacts) -> tuple[str, ...]:
    """7.4 diagnostics: `text_veto_nonclear_on_empty_news` once per text veto that reads non-CLEAR although `news_count == 0`
    (the veto itself is skipped: an empty list is known exactly in code). The cycle ledgers these as ANOMALY entries."""
    if text is None or facts.news_count > 0:
        return ()
    cfg = RulesConfig()  # the band edges are the [rules] defaults' meaning of "non-CLEAR"; overridden by DecisionRules.anomalies
    return _text_anomalies(text, cfg)


def _text_anomalies(text: DecisionResult, cfg: RulesConfig) -> tuple[str, ...]:
    out: list[str] = []
    for qid in (_Q_TEXT_PENDING, _Q_TEXT_STRESS, _Q_TEXT_NEGATIVE, _Q_TEXT_POSITIVE):
        if tri_band(_noul(text, qid).p, cfg) is not Tri.CLEAR:
            out.append(ANOMALY_TEXT_VETO_ON_EMPTY_NEWS)
    return tuple(out)


def no_trade(underlying: str, decision_id: str, *reasons: str) -> EntryDecision:
    """The `no_trade` decision of 7.2 steps 0 / 1 (and 7.9 `state_rejected`), built by the cycle before any answer exists."""
    if not reasons:
        raise InvariantError("a no_trade decision needs at least one reason code")
    for code in reasons:
        if not vocab.is_reason(code):
            raise InvariantError(f"{code!r} is not a vocab.REASONS code")
    return EntryDecision(
        underlying=underlying,
        decision_id=decision_id,
        action=_ACTION_NO_TRADE,
        kind=None,
        score_core_ppm=0,
        score_rank_ppm=0,
        tier_ppm=0,
        reasons=tuple(reasons),
        features_ppm={},
        variant_agreement={},
    )


def text_watch_alert(decision: ManageDecision, cfg: RulesConfig) -> bool:
    """7.7: True exactly when the unconfirmed-text counter has just REACHED `rules.text_watch_alert_sessions` - the one cycle in
    which `RISK_EVENT{text_watch}` is appended and the alert raised (once per position). Never a reason to close (INV-16)."""
    return decision.watch_text == cfg.text_watch_alert_sessions


class DecisionRules:
    """Implements `protocols.DecisionRulesP` (section 7). Stateless apart from its configuration."""

    def __init__(self, cfg: RulesConfig, enabled: Sequence[StructureKind]) -> None:
        self._cfg = cfg
        self._enabled: frozenset[StructureKind] = frozenset(StructureKind(k) for k in enabled)
        # the section-4 `rules_hash` of the [rules] section: config.rules_hash over a Config carrying exactly this section
        self._rules_hash = _config.rules_hash(Config(rules=cfg))

    @property
    def rules_hash(self) -> str:
        return self._rules_hash

    @property
    def cfg(self) -> RulesConfig:
        return self._cfg

    @property
    def enabled(self) -> frozenset[StructureKind]:
        return self._enabled

    # ------------------------------------------------------------------------------------------------------------------
    # entry
    # ------------------------------------------------------------------------------------------------------------------

    def anomalies(self, text: DecisionResult | None, facts: EntryFacts) -> tuple[str, ...]:
        """`entry_anomalies` with this instance's veto bands."""
        if text is None or facts.news_count > 0:
            return ()
        return _text_anomalies(text, self._cfg)

    def decide_entry(
        self, underlying: str, decision_id: str, core: DecisionResult, text: DecisionResult | None, facts: EntryFacts
    ) -> EntryDecision:
        """Phase 1 (base variant): steps 2-10 of 7.2. `text` is None when news is off / not covered (absent by design)."""
        ev = self._evaluate(core, text, facts)
        return EntryDecision(
            underlying=underlying,
            decision_id=decision_id,
            action=ev.action,
            kind=ev.kind,
            score_core_ppm=ev.score_core_ppm,
            score_rank_ppm=_rank_ppm(ev.score_core_ppm, ev.news_align_ppm, self._cfg.text_rank_weight),
            tier_ppm=ev.tier_ppm,
            reasons=tuple(ev.reasons),
            features_ppm=dict(ev.features_ppm),
            variant_agreement={},
        )

    def confirm_entry(
        self,
        base: EntryDecision,
        core_variants: Mapping[Variant, DecisionResult | DeciderError],
        text: DecisionResult | None,
        facts: EntryFacts,
    ) -> EntryDecision:
        """Phase 2 (7.8): re-run steps 2-10 on every perturbation variant with the same text result and facts. Every variant must
        return `enter` with the same structure; then `tier = min(tiers)`, `S_core = min(S_core)`. An errored variant is
        representable and counts as disagreement (`perturb:decider_failed`)."""
        if base.action != _ACTION_ENTER:
            return base  # only called for entering decisions; a no_trade base has nothing to confirm
        reasons = list(base.reasons)
        agreement: dict[str, bool] = {}
        tiers = [base.tier_ppm]
        scores = [base.score_core_ppm]
        for variant, result in core_variants.items():
            name = Variant(variant).value
            if isinstance(result, DeciderError):
                agreement[name] = False
                reasons.append("perturb:decider_failed")
                continue
            ev = self._evaluate(result, text, facts)
            tiers.append(ev.tier_ppm)
            scores.append(ev.score_core_ppm)
            if ev.action == _ACTION_ENTER and ev.kind == base.kind:
                agreement[name] = True
                continue
            agreement[name] = False
            detail = ev.reasons[0] if ev.reasons else f"kind:{ev.kind.value if ev.kind is not None else 'none'}"
            code = f"{vocab.PERTURB_DISAGREE_PREFIX}{name}:{detail}"
            if not vocab.is_reason(code):
                raise InvariantError(f"{code!r} is not a vocab.REASONS code")
            reasons.append(code)
        score_core = min(scores)
        return EntryDecision(
            underlying=base.underlying,
            decision_id=base.decision_id,
            action=_ACTION_ENTER if not reasons else _ACTION_NO_TRADE,
            kind=base.kind,
            score_core_ppm=score_core,
            score_rank_ppm=_rank_ppm(score_core, base.features_ppm.get("news_align", PPM // 2), self._cfg.text_rank_weight),
            tier_ppm=min(tiers),
            reasons=tuple(reasons),
            features_ppm=dict(base.features_ppm),
            variant_agreement=agreement,
        )

    def rank(self, entries: Sequence[EntryDecision], order: Sequence[str]) -> list[EntryDecision]:
        """By `score_rank_ppm` descending, ties by universe order (unknown underlyings after the universe, by name)."""
        position = {u: i for i, u in enumerate(order)}
        n = len(order)
        return sorted(entries, key=lambda e: (-e.score_rank_ppm, position.get(e.underlying, n), e.underlying))

    # --- the evaluation ------------------------------------------------------------------------------------------------

    def _evaluate(self, core: DecisionResult, text: DecisionResult | None, facts: EntryFacts) -> _Eval:
        cfg = self._cfg
        ev = _Eval()
        regime = _choice(core, _Q_REGIME)
        direction = _choice(core, _Q_DIRECTION)
        stance = _choice(core, _Q_VOL_STANCE)
        fit = _choice(core, _Q_FIT)
        stretched = _noul(core, _Q_STRETCHED)
        vol_event = _noul(core, _Q_VOL_EVENT)
        env = _score(core, _Q_ENV)

        # step 2: regime veto (an out-of-band R counts as unclear_or_transition)
        r_top = regime.top if _in_band(regime.raw_sum) else _REGIME_NO_MATCH
        if r_top in vocab.REGIME_VETO_LABELS:
            ev.reasons.append(f"veto:regime:{r_top}")

        # steps 3 / 4: direction and vol-stance gates
        ev.reasons += _gate_codes(direction, cfg.gates.direction, "gate:direction", _DIRECTION_NO_MATCH, "conflicting", _DIRECTION_VALUES)
        ev.reasons += _gate_codes(stance, cfg.gates.vol_stance, "gate:vol_stance", _VOL_NO_MATCH, "unclear", _STANCE_VALUES)

        # step 5: deterministic mapping (7.3)
        d_top = direction.top if _in_band(direction.raw_sum) else _DIRECTION_NO_MATCH
        v_top = stance.top if _in_band(stance.raw_sum) else _VOL_NO_MATCH
        kind: StructureKind | None = None
        if d_top in _DIRECTION_VALUES and v_top in _STANCE_VALUES:
            kind = MAPPING[(Direction(d_top), VolStance(v_top))]
        if kind is None:
            ev.reasons.append("map:no_structure")
        elif kind not in self._enabled:
            ev.reasons.append("map:disabled")
        ev.kind = kind

        # step 6: code-side cross-checks on the state's own bucket codes (7.3)
        if kind is not None and cfg.code_crosschecks:
            ev.reasons += [f"crosscheck:{name}" for name in _crosscheck_failures(kind, facts)]

        # step 7: structure cross-check (fit.structure_family can only confirm what the table implies)
        f_top = fit.top if _in_band(fit.raw_sum) else _FIT_NO_MATCH
        if f_top == _FIT_NO_MATCH:
            ev.reasons.append("fit:no_trade")
        elif kind is not None and f_top != kind.value:
            ev.reasons.append("fit:disagrees_with_mapping")
        if fit.p_top < cfg.gates.structure.p_top:
            ev.reasons.append("fit:p_top")
        if fit.margin < cfg.gates.structure.margin:
            ev.reasons.append("fit:margin")

        # step 8: three-valued vetoes (7.4), max-style; text vetoes only when a text result exists and the list is non-empty
        vetoes: dict[str, float] = {}
        if kind in SHORT_PREMIUM:
            vetoes[_Q_VOL_EVENT] = vol_event.p
        text_live = text is not None and facts.news_count > 0
        if text is not None and text_live:
            if facts.news_recent_count > 0:
                vetoes[_Q_TEXT_PENDING] = _noul(text, _Q_TEXT_PENDING).p
            vetoes[_Q_TEXT_STRESS] = _noul(text, _Q_TEXT_STRESS).p
            if kind is not None:
                d_k = STRUCTURE_DIRECTION[kind]
                if d_k is Direction.BULLISH or kind is StructureKind.IRON_CONDOR:
                    vetoes[_Q_TEXT_NEGATIVE] = _noul(text, _Q_TEXT_NEGATIVE).p
                if d_k is Direction.BEARISH or kind is StructureKind.IRON_CONDOR:
                    vetoes[_Q_TEXT_POSITIVE] = _noul(text, _Q_TEXT_POSITIVE).p
        for qid in vocab.VETO_IDS:  # fixed order
            if qid not in vetoes:
                continue
            band = tri_band(vetoes[qid], cfg)
            if band is Tri.VETO:
                ev.reasons.append(f"veto:{qid}:hard")
            elif band is Tri.UNCERTAIN:
                ev.reasons.append(f"veto:{qid}:uncertain")

        # step 9: composite S_core (7.5) - TEXT-FREE; needs the mapped structure
        tier_s = 0.0
        if kind is not None:
            d_k = STRUCTURE_DIRECTION[kind]
            s_k = STRUCTURE_STANCE[kind]
            align = _prob(direction, d_k.value)
            volfit = _prob(stance, s_k.value)
            fit_x = _prob(fit, kind.value)
            regimefit = sum(_prob(regime, label) for label in _REGIME_OK[d_k])
            calm = 1.0 - stretched.p
            w = cfg.weights
            s_core = w.align * align + w.volfit * volfit + w.fit * fit_x + w.regimefit * regimefit + w.calm * calm
            ev.score_core_ppm = _ppm(s_core)
            ev.features_ppm = {
                "align": _ppm(align),
                "volfit": _ppm(volfit),
                "fit": _ppm(fit_x),
                "regimefit": _ppm(regimefit),
                "calm": _ppm(calm),
            }
            # the rank term (ordering only): tone from the text result when it exists, the list is non-empty and material
            tone = 0.0
            if text is not None and text_live and _noul(text, _Q_TEXT_MATERIAL).p >= 0.5:
                tone = _noul(text, _Q_TEXT_POSITIVE).p - _noul(text, _Q_TEXT_NEGATIVE).p
            if d_k is Direction.BULLISH:
                news_align = 0.5 + 0.5 * tone
            elif d_k is Direction.BEARISH:
                news_align = 0.5 - 0.5 * tone
            else:
                news_align = 1.0 - abs(tone)
            ev.news_align_ppm = _ppm(news_align)
            ev.features_ppm["news_align"] = ev.news_align_ppm
            if ev.score_core_ppm < _threshold_ppm(cfg.min_score):
                ev.reasons.append("score:below_min")
            tier_s = _tier_lookup_ppm(ev.score_core_ppm, cfg.tiers.score)

        # step 10: sizing tier (7.6) - the minimum of three tiers, never a product
        tier_peak = _tier_lookup(min(direction.p_top, stance.p_top), cfg.tiers.peakedness)
        if _in_band(env.raw_sum):
            env_level = max(_round_half_up(env.mean), env.top)
        else:
            env_level = _ENV_LEVELS - 1  # an untrusted environment score reads as the hostile level (fail closed)
        env_level = min(max(env_level, 0), _ENV_LEVELS - 1)
        tier_env = cfg.tiers.environment[env_level]
        tier = min(tier_s, tier_peak, tier_env)
        ev.tier_ppm = _ppm(tier)
        if tier <= 0.0:
            if kind is not None and tier_s <= 0.0:
                ev.reasons.append("tier:zero:score")
            if tier_peak <= 0.0:
                ev.reasons.append("tier:zero:peak")
            if tier_env <= 0.0:
                ev.reasons.append("tier:zero:env")
            if kind is None and tier_peak > 0.0 and tier_env > 0.0:
                pass  # no structure: the score tier is undefined and the gates already carry the reason
        return ev

    # ------------------------------------------------------------------------------------------------------------------
    # management (7.7)
    # ------------------------------------------------------------------------------------------------------------------

    def decide_manage(
        self,
        pos: Position,
        decision_id: str,
        hard: ExitReason | None,
        core: DecisionResult | None,
        text: DecisionResult | None,
        facts: ManageFacts,
    ) -> ManageDecision:
        cfg = self._cfg
        # 1. the code-side hard exit always wins (9.4); Jev is not consulted for that position
        if hard is not None:
            return ManageDecision(
                position_id=pos.position_id,
                decision_id=decision_id,
                action=_MANAGE_CLOSE,
                reason=ExitReason(hard).value,
                source=_SOURCE_HARD,
                pressure_ppm=None,
                exit_latch=pos.exit_latch,
                watch_text=pos.watch_text,
                reasons=(),
            )
        # 2. decider down / not asked => the code default (D19): close only on a breached or at-strike short, else hold
        if core is None:
            close = facts.short_dist_code in _CODE_DEFAULT_CLOSE_CODES
            return ManageDecision(
                position_id=pos.position_id,
                decision_id=decision_id,
                action=_MANAGE_CLOSE if close else _MANAGE_HOLD,
                reason=ExitReason.CODE_DEFAULT.value if close else _MANAGE_HOLD,
                source=_SOURCE_CODE_DEFAULT,
                pressure_ppm=None,
                exit_latch=pos.exit_latch,
                watch_text=0,
                reasons=() if close else (_MANAGE_HOLD,),
            )
        # 3. core exit pressure X (text-free) and the hysteresis latch
        thesis = _noul(core, _Q_THESIS)
        threat = _score(core, _Q_THREAT)
        has_short = bool(pos.structure.short_legs)
        threat_norm = min(max(threat.norm, 0.0), 1.0) if _in_band(threat.raw_sum) else 0.5  # untrusted => UNCERTAIN
        pressure = max(thesis.p, threat_norm if has_short else 0.0)
        latch = pos.exit_latch
        close = False
        reasons: tuple[str, ...] = (_MANAGE_HOLD,)
        if latch:
            if pressure < cfg.exit_pressure_release:
                latch = False
                reasons = ("hysteresis:released",)
            else:
                close = True  # keeps trying on later sessions if the close does not fill
        elif pressure >= cfg.exit_pressure_enter:
            latch = True
            close = True
        elif pressure >= cfg.exit_pressure_release:
            close = self._discretionary_close(_choice(core, _Q_ACTION), facts)
        reason = ExitReason.JEV.value if close else _MANAGE_HOLD
        if close:
            reasons = ()
        # 4. the text rule: a text reading closes ONLY with code-side market-data confirmation (INV-16)
        watch = 0
        if text is not None:
            pending = 0.0
            if pos.structure.kind in SHORT_PREMIUM and facts.news_recent_count > 0:
                pending = _noul(text, _Q_PENDING_SINCE).p
            t_value = max(_noul(text, _Q_ADVERSE).p, pending)
            if t_value > cfg.veto_hi:
                confirmed = facts.move_code in _ADVERSE_MOVES or facts.pnl_frac_loss_ppm >= _TEXT_CONFIRM_LOSS_PPM
                if confirmed:
                    if not close:
                        close = True
                        reason = ExitReason.TEXT_CONFIRMED.value
                        reasons = ()
                else:
                    watch = pos.watch_text + 1  # an alert counter; it never closes
        return ManageDecision(
            position_id=pos.position_id,
            decision_id=decision_id,
            action=_MANAGE_CLOSE if close else _MANAGE_HOLD,
            reason=reason,
            source=_SOURCE_JEV,
            pressure_ppm=_ppm(pressure),
            exit_latch=latch,
            watch_text=watch,
            reasons=reasons,
        )

    def _discretionary_close(self, action: ChoiceAns, facts: ManageFacts) -> bool:
        """7.7 discretionary zone: the gated `pos.action` label, else the risk-reducing default (close when losing)."""
        gate = self._cfg.gates.action
        gated = _in_band(action.raw_sum) and action.p_top >= gate.p_top and action.margin >= gate.margin
        if gated:
            if action.top == _ACTION_HOLD:
                return False
            if action.top == _ACTION_TAKE_PROFIT and facts.pnl_headline > 0:
                return True
            if action.top == _ACTION_CUT_LOSS and facts.pnl_headline < 0:
                return True
        return facts.pnl_headline < 0  # unclear, gate failed or a label inconsistent with the sign of the P&L


def _crosscheck_failures(kind: StructureKind, facts: EntryFacts) -> list[str]:
    """7.3, in the order of vocab.CROSSCHECK_NAMES."""
    out: list[str] = []
    direction = STRUCTURE_DIRECTION[kind]
    if (direction is Direction.BULLISH and facts.trend_code == _TREND_DOWN) or (
        direction is Direction.BEARISH and facts.trend_code == _TREND_UP
    ):
        out.append("trend_not_opposed")
    if kind is StructureKind.IRON_CONDOR and facts.trend_code not in _TREND_RANGE:
        out.append("range_needs_no_trend")
    if kind in SHORT_PREMIUM and facts.iv_rv_code not in _IV_RICH:
        out.append("sell_needs_rich")
    if kind in _LONG_SINGLES and facts.iv_rank_code not in _IV_CHEAP_RANK:
        out.append("long_single_needs_cheap")
    return out
