"""Frozen vocabulary: question ids, Choice labels, bucket codes, state paths, reason / reject codes (DESIGN.md 2.11).

Sources, all in DESIGN.md: 5.5 (bucket tables), 5.6 / 5.7 (state shapes), 6.1-6.6 (question ids, option labels, the probe state),
7.2 / 7.3 / 7.9 (reasons, gate codes), 8 (candidate rejects), 9.1 / 9.3 (risk codes), 10.4 (fill rejects), 2.11 (ledger vocabularies).
`tests/unit/test_vocab.py` parses DESIGN.md at test time and asserts that the constants below equal what is written there.

Plain data and pure helpers only: this module imports nothing from the package, so every module may import it. `rules.py` may
read only `TRADING_IDS | TEXT_IDS | MANAGE_IDS | MANAGE_TEXT_IDS` (INV-23).
"""

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any, Final

# ======================================================================================================================
# Question ids (6.1-6.6). The `*_ORDER` tuples keep the AUTHORED order (it is bound into `question_set_hash`, 5.9);
# the `*_IDS` frozensets are the disjoint sets the contract names.
# ======================================================================================================================

TRADING_ORDER: Final[tuple[str, ...]] = (  # 6.1 entry.v1 trading questions (7), text-free state
    "regime.market",
    "under.direction",
    "under.stretched",
    "vol.stance",
    "vol.explained_by_event",
    "fit.structure_family",
    "risk.environment",
)
EVAL_ORDER: Final[tuple[str, ...]] = (  # 6.2 evaluation-only questions (12): appended, byte-identical, to BOTH entry sets
    "eval.up_1s",
    "eval.down_1em_1s",
    "eval.up_1em_1s",
    "eval.inside_1em_1s",
    "eval.up_5s",
    "eval.down_1em_5s",
    "eval.up_1em_5s",
    "eval.inside_1em_5s",
    "eval.rv_gt_iv_5s",
    "eval.down_1em_hold",
    "eval.up_1em_hold",
    "eval.inside_1em_hold",
)
TEXT_ORDER: Final[tuple[str, ...]] = (  # 6.3 entry_text.v1 text questions (5)
    "text.material_present",
    "text.clearly_negative",
    "text.clearly_positive",
    "text.pending_binary",
    "text.market_stress",
)
MANAGE_ORDER: Final[tuple[str, ...]] = (  # 6.5 manage.v1
    "pos.thesis_invalidated",
    "pos.short_strike_threat",
    "pos.action",
)
MANAGE_TEXT_ORDER: Final[tuple[str, ...]] = (  # 6.5 manage_text.v1
    "pos.adverse_news_since_entry",
    "pos.pending_binary_since_entry",
)
PROBE_ORDER: Final[tuple[str, ...]] = ("probe.closed_higher_5s",)  # 6.6 probe.recall.v1

TRADING_IDS: Final[frozenset[str]] = frozenset(TRADING_ORDER)
TEXT_IDS: Final[frozenset[str]] = frozenset(TEXT_ORDER)
EVAL_IDS: Final[frozenset[str]] = frozenset(EVAL_ORDER)  # invisible to rules.py (INV-23)
MANAGE_IDS: Final[frozenset[str]] = frozenset(MANAGE_ORDER)
MANAGE_TEXT_IDS: Final[frozenset[str]] = frozenset(MANAGE_TEXT_ORDER)
PROBE_IDS: Final[frozenset[str]] = frozenset(PROBE_ORDER)  # diagnostic namespace that can never trade

RULES_READABLE_IDS: Final[frozenset[str]] = TRADING_IDS | TEXT_IDS | MANAGE_IDS | MANAGE_TEXT_IDS
ALL_QUESTION_IDS: Final[frozenset[str]] = RULES_READABLE_IDS | EVAL_IDS | PROBE_IDS

# `under.direction` is additionally scored (secondary) as three derived Nouls (6.4); these are FORECAST question ids, never asked.
DERIVED_EVAL_IDS: Final[frozenset[str]] = frozenset({"under.direction#bullish", "under.direction#bearish", "under.direction#neutral_range"})

# request kind (types.RequestKind value) -> question set id, and question set id -> ids in batch order
QUESTION_SET_ID: Final[Mapping[str, str]] = MappingProxyType(
    {
        "entry": "entry.v1",
        "entry_text": "entry_text.v1",
        "manage": "manage.v1",
        "manage_text": "manage_text.v1",
        "probe": "probe.recall.v1",
    }
)
QUESTION_SETS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "entry.v1": TRADING_ORDER + EVAL_ORDER,
        "entry_text.v1": TEXT_ORDER + EVAL_ORDER,
        "manage.v1": MANAGE_ORDER,
        "manage_text.v1": MANAGE_TEXT_ORDER,
        "probe.recall.v1": PROBE_ORDER,
    }
)

# ======================================================================================================================
# Answer labels (6.1, 6.5). Choice labels in AUTHORED option order; every Choice has a no-match option.
# ======================================================================================================================

NOUL_LABELS: Final[tuple[str, str]] = ("true", "false")

CHOICE_LABELS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "regime.market": (
            "trending_up_calm",
            "trending_up_volatile",
            "range_bound_calm",
            "range_bound_volatile",
            "orderly_downtrend",
            "disorderly_selloff",
            "unclear_or_transition",
        ),
        "under.direction": ("bullish", "bearish", "neutral_range", "conflicting_signals"),
        "vol.stance": ("sell_premium", "buy_premium", "limit_vol_exposure", "unclear"),
        "fit.structure_family": (
            "long_call",
            "long_put",
            "call_debit_spread",
            "put_debit_spread",
            "put_credit_spread",
            "call_credit_spread",
            "iron_condor",
            "no_trade",
        ),
        "pos.action": ("hold", "take_profit", "close_to_cut_loss", "unclear"),
    }
)
NO_MATCH_LABEL: Final[Mapping[str, str]] = MappingProxyType(
    {
        "regime.market": "unclear_or_transition",
        "under.direction": "conflicting_signals",
        "vol.stance": "unclear",
        "fit.structure_family": "no_trade",
        "pos.action": "unclear",
    }
)
# Score questions -> number of levels K (wire keys "0".."K-1"; norm = mean / (K-1))
SCORE_LEVELS: Final[Mapping[str, int]] = MappingProxyType({"risk.environment": 4, "pos.short_strike_threat": 4})

QUESTION_TYPES: Final[Mapping[str, str]] = MappingProxyType(
    {
        qid: ("choice" if qid in CHOICE_LABELS else "score" if qid in SCORE_LEVELS else "noul")
        for qid in (*TRADING_ORDER, *EVAL_ORDER, *TEXT_ORDER, *MANAGE_ORDER, *MANAGE_TEXT_ORDER, *PROBE_ORDER)
    }
)

# ======================================================================================================================
# Bucket codes (5.5). Every label is "<code>: <meaning>"; the code (text before the first colon) is the stable machine token.
# Codes are listed in the order of the 5.5 tables. A None feature renders as the string "unavailable".
# ======================================================================================================================

UNAVAILABLE: Final[str] = "unavailable"

PCTL5: Final[tuple[str, ...]] = ("very_low", "low", "middle", "upper_middle", "high")
PCTL3: Final[tuple[str, ...]] = ("subdued", "normal", "elevated")  # market.vol_of_vol, market.tail_skew_index
PCTL3_SKEW: Final[tuple[str, ...]] = ("flat", "normal", "steep")  # vol_surface.skew (same thresholds, its own wording)
CHANGE5: Final[tuple[str, ...]] = ("falling_sharply", "falling", "steady", "rising", "rising_sharply")
TERM4: Final[tuple[str, ...]] = ("steep_contango", "contango", "flat", "backwardation")
NEAR3: Final[tuple[str, ...]] = ("calm", "neutral", "stressed")
TREND_DIR: Final[tuple[str, ...]] = ("up", "down", "flat", "mixed")
TREND_STRENGTH: Final[tuple[str, ...]] = ("weak", "moderate", "strong")
DIST_ATR: Final[tuple[str, ...]] = ("stretched_far_below", "extended_below", "near_average", "extended_above", "stretched_far_above")
STREAK: Final[tuple[str, ...]] = ("long_down_streak", "short_down_streak", "no_streak", "short_up_streak", "long_up_streak")
RV_CHANGE: Final[tuple[str, ...]] = ("contracting", "stable", "expanding", "expanding_sharply")
SIGMA5_GAP: Final[tuple[str, ...]] = ("large_gap_down", "gap_down", "none", "gap_up", "large_gap_up")  # underlying.range.gap_today
SIGMA5_MOVE: Final[tuple[str, ...]] = ("large_decline", "decline", "quiet", "advance", "large_advance")  # underlying.range.move_today
DD_52W: Final[tuple[str, ...]] = ("near_high", "close_to_high", "pullback", "correction", "deep_drawdown")
IV_RV: Final[tuple[str, ...]] = ("iv_cheap", "iv_fair", "iv_rich", "iv_very_rich")
EM: Final[tuple[str, ...]] = (
    "under_half_percent",
    "half_to_1_percent",
    "1_to_2_percent",
    "2_to_4_percent",
    "4_to_6_percent",
    "6_to_10_percent",
    "10_percent_or_more",
)
HOLD: Final[tuple[str, ...]] = ("about_one_week", "about_two_weeks", "about_three_weeks", "about_four_weeks", "more_than_a_month")
# position buckets (manage state)
DTE4: Final[tuple[str, ...]] = ("four_weeks_or_more", "two_to_four_weeks", "one_to_two_weeks", "one_week_or_less")
HELD4: Final[tuple[str, ...]] = ("just_opened", "about_one_week", "two_to_three_weeks", "more_than_three_weeks")
PNL: Final[tuple[str, ...]] = ("large_loss", "loss", "moderate_loss", "small_loss", "flat", "small_gain", "gain", "large_gain")
SHORT_DIST: Final[tuple[str, ...]] = ("far", "about_one_move", "close", "at_strike", "breached")
BREAKEVEN: Final[tuple[str, ...]] = ("well_beyond", "just_beyond", "just_short", "short", "far_short")
MOVE_SINCE_ENTRY: Final[tuple[str, ...]] = ("strongly_favourable", "favourable", "little_change", "adverse", "strongly_adverse")

BUCKET_CODES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "PCTL5": PCTL5,
        "PCTL3": PCTL3,
        "PCTL3_SKEW": PCTL3_SKEW,
        "CHANGE5": CHANGE5,
        "TERM4": TERM4,
        "NEAR3": NEAR3,
        "TREND_DIR": TREND_DIR,
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
        "PNL": PNL,
        "SHORT_DIST": SHORT_DIST,
        "BREAKEVEN": BREAKEVEN,
        "MOVE_SINCE_ENTRY": MOVE_SINCE_ENTRY,
    }
)
ALL_BUCKET_CODES: Final[frozenset[str]] = frozenset(code for codes in BUCKET_CODES.values() for code in codes)

# State path of every bucketed field -> its bucket table. The four `changes_since_entry.*_at_entry` / `*_now` fields carry the
# BARE code (no meaning text, 5.7); every other field carries the full label, directly or under its `bucket` key.
BUCKET_TABLE_BY_PATH: Final[Mapping[str, str]] = MappingProxyType(
    {
        "context.holding_window_sessions": "HOLD",
        "market.vol_index_pctile_1y": "PCTL5",
        "market.vol_index_change_1w": "CHANGE5",
        "market.vol_term_structure": "TERM4",
        "market.near_term_stress": "NEAR3",
        "market.vol_of_vol": "PCTL3",
        "market.tail_skew_index": "PCTL3",
        "underlying.trend.direction": "TREND_DIR",
        "underlying.trend.strength": "TREND_STRENGTH",
        "underlying.momentum.distance_from_20d_avg_in_atr": "DIST_ATR",
        "underlying.momentum.consecutive_closes": "STREAK",
        "underlying.range.realized_vol_20d_pctile_1y": "PCTL5",
        "underlying.range.realized_vol_change": "RV_CHANGE",
        "underlying.range.gap_today": "SIGMA5_GAP",
        "underlying.range.move_today": "SIGMA5_MOVE",
        "underlying.levels.distance_from_52w_high": "DD_52W",
        "vol_surface.iv_rank_1y": "PCTL5",
        "vol_surface.iv_vs_realized": "IV_RV",
        "vol_surface.iv_change_1w": "CHANGE5",
        "vol_surface.term_structure": "TERM4",
        "vol_surface.skew": "PCTL3_SKEW",
        "vol_surface.expected_move_1_session": "EM",
        "vol_surface.expected_move_5_sessions": "EM",
        "vol_surface.expected_move_holding_window": "EM",
        "position.sessions_held": "HELD4",
        "position.time_to_expiry": "DTE4",
        "position.pnl": "PNL",
        "position.short_strike_distance": "SHORT_DIST",
        "position.price_vs_breakeven": "BREAKEVEN",
        "changes_since_entry.trend_at_entry": "TREND_DIR",
        "changes_since_entry.trend_now": "TREND_DIR",
        "changes_since_entry.iv_vs_realized_at_entry": "IV_RV",
        "changes_since_entry.iv_vs_realized_now": "IV_RV",
        "changes_since_entry.iv_change_since_entry": "CHANGE5",
        "changes_since_entry.underlying_move_since_entry": "MOVE_SINCE_ENTRY",
    }
)
BARE_CODE_PATHS: Final[frozenset[str]] = frozenset(
    {
        "changes_since_entry.trend_at_entry",
        "changes_since_entry.trend_now",
        "changes_since_entry.iv_vs_realized_at_entry",
        "changes_since_entry.iv_vs_realized_now",
    }
)


def bucket_code(label: str) -> str:
    """The machine code of a bucket label: the text before the first colon (`"unavailable"` and bare codes map to themselves)."""
    return label.split(":", 1)[0]


# Closed state enums that are not bucket tables (5.6)
NEWS_STATUS: Final[tuple[str, ...]] = ("present", "none_in_window")
NEWS_SOURCE_TYPES: Final[tuple[str, ...]] = ("newswire", "press_release", "other")

# ======================================================================================================================
# State shapes and paths (5.6, 5.7, 6.6). A shape is the state's nested KEY ORDER with None leaves; lists are leaves.
# STATE_PATHS[kind] = every dotted key path of that shape (inner nodes included): every state path a question may name.
# ======================================================================================================================

STATE_SCHEMA: Final[Mapping[str, str]] = MappingProxyType(
    {
        "entry": "state.v1.entry",
        "entry_text": "state.v1.entry_text",
        "manage": "state.v1.manage",
        "manage_text": "state.v1.manage_text",
        "probe": "state.v1.probe_recall",
    }
)

EXPECTED_MOVE_PREFIX: Final[str] = (
    "vol_surface.expected_move_"  # BUCKET_ONLY keeps these dicts: their integers define the eval thresholds (5.9)
)


def _vb() -> dict[str, Any]:
    return {"value": None, "bucket": None}


def _vub() -> dict[str, Any]:
    return {"value": None, "unit": None, "bucket": None}


def _market() -> dict[str, Any]:
    return {
        "as_of": None,
        "vol_index_pctile_1y": _vb(),
        "vol_index_change_1w": None,
        "vol_term_structure": None,
        "near_term_stress": None,
        "vol_of_vol": None,
        "tail_skew_index": None,
    }


def _underlying() -> dict[str, Any]:
    return {
        "trend": {"direction": None, "strength": None},
        "momentum": {"distance_from_20d_avg_in_atr": _vb(), "consecutive_closes": _vb()},
        "range": {
            "realized_vol_20d_pctile_1y": _vb(),
            "realized_vol_change": None,
            "gap_today": None,
            "move_today": None,
        },
        "levels": {"distance_from_52w_high": None},
    }


def _vol_surface(*, expected_moves: bool) -> dict[str, Any]:
    shape: dict[str, Any] = {
        "iv_rank_1y": _vb(),
        "iv_vs_realized": None,
        "iv_change_1w": None,
        "term_structure": None,
        "skew": None,
    }
    if expected_moves:
        shape["expected_move_1_session"] = _vub()
        shape["expected_move_5_sessions"] = _vub()
        shape["expected_move_holding_window"] = _vub()
    return shape


def _events() -> dict[str, Any]:
    return {"coverage": None, "inside_holding_window": None, "next_session": None, "earnings": None}


def _news() -> dict[str, Any]:
    return {"since_previous_session": None, "earlier": None}


def _entry_shape(*, text: bool) -> dict[str, Any]:
    shape: dict[str, Any] = {
        "schema": None,
        "context": {
            "underlying_alias": None,
            "underlying_kind": None,
            "decision_session": None,
            "holding_window_sessions": _vb(),
        },
        "market": _market(),
        "underlying": _underlying(),
        "vol_surface": _vol_surface(expected_moves=True),
        "events": _events(),
    }
    if text:  # state.v1.entry_text = the same object with two keys appended after `events`
        shape["news_status"] = None
        shape["news"] = _news()
    return shape


def _manage_shape(*, text: bool) -> dict[str, Any]:
    shape: dict[str, Any] = {
        "schema": None,
        "context": {"underlying_alias": None, "underlying_kind": None, "decision_session": None},
        "position": {
            "structure": None,
            "directional_exposure": None,
            "vol_exposure": None,
            "entry_thesis": None,
            "sessions_held": _vb(),
            "time_to_expiry": _vb(),
            "pnl": None,
            "short_strike_distance": None,
            "price_vs_breakeven": None,
        },
        "changes_since_entry": {
            "trend_at_entry": None,
            "trend_now": None,
            "iv_vs_realized_at_entry": None,
            "iv_vs_realized_now": None,
            "iv_change_since_entry": None,
            "underlying_move_since_entry": _vb(),
        },
        "market": _market(),
        "underlying": _underlying(),
        "vol_surface": _vol_surface(expected_moves=False),  # same block as entry, without the three expected_move fields
        "events": _events(),
    }
    if text:  # state.v1.manage_text = the same object with `news_since_entry` appended
        shape["news_since_entry"] = _news()
    return shape


def _freeze(shape: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: (_freeze(value) if isinstance(value, Mapping) else value) for key, value in shape.items()})


STATE_SHAPES: Final[Mapping[str, Mapping[str, Any]]] = MappingProxyType(
    {
        "entry": _freeze(_entry_shape(text=False)),
        "entry_text": _freeze(_entry_shape(text=True)),
        "manage": _freeze(_manage_shape(text=False)),
        "manage_text": _freeze(_manage_shape(text=True)),
        "probe": _freeze({"schema": None, "ticker": None, "date": None}),
    }
)


def _walk(obj: Mapping[str, Any], prefix: str, *, bucket_only: bool) -> Iterator[str]:
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else key
        yield path
        if not isinstance(value, Mapping):
            continue
        if bucket_only and "value" in value and "bucket" in value and not path.startswith(EXPECTED_MOVE_PREFIX):
            continue  # 5.9 BUCKET_ONLY: the dict is replaced by its bucket string
        yield from _walk(value, path, bucket_only=bucket_only)


def state_paths(state: Mapping[str, Any]) -> frozenset[str]:
    """Every dotted key path of a state object, inner nodes included. Lists (event strings, news items) are leaves.

    `state_paths(built_state) <= STATE_PATHS[kind]` for every state a StateBuilder may emit; equality holds when no
    value/bucket field collapsed to the plain "unavailable" string.
    """
    return frozenset(_walk(state, "", bucket_only=False))


STATE_PATHS: Final[Mapping[str, frozenset[str]]] = MappingProxyType({kind: state_paths(shape) for kind, shape in STATE_SHAPES.items()})
# the path set of the BUCKET_ONLY variant / `state.render = "bucket_only"` (5.9)
STATE_PATHS_BUCKET_ONLY: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {kind: frozenset(_walk(shape, "", bucket_only=True)) for kind, shape in STATE_SHAPES.items()}
)
STATE_KEYS: Final[frozenset[str]] = frozenset(segment for paths in STATE_PATHS.values() for path in paths for segment in path.split("."))

# 5.8 step 3: any underscore-containing identifier that equals a question option label or a state key marks a news item as hostile
UNDERSCORE_IDENTIFIERS: Final[frozenset[str]] = frozenset(
    token for token in (STATE_KEYS | {label for labels in CHOICE_LABELS.values() for label in labels}) if "_" in token
)

# ======================================================================================================================
# Reasons (7.2, 7.3, 7.9): EntryDecision.reasons / ManageDecision.reasons - a pure function of answers + facts.
# ======================================================================================================================

DECIDER_ERROR_CLASSES: Final[tuple[str, ...]] = (  # errors.DeciderError and its subclasses, by class name
    "DeciderError",
    "DeciderTransportError",
    "DeciderResponseError",
    "DeciderConfigError",
    "SpendLimitError",
)
REGIME_VETO_LABELS: Final[tuple[str, ...]] = ("disorderly_selloff", "unclear_or_transition")  # 7.2 step 2
CROSSCHECK_NAMES: Final[tuple[str, ...]] = (  # 7.3
    "trend_not_opposed",
    "range_needs_no_trend",
    "sell_needs_rich",
    "long_single_needs_cheap",
)
VETO_IDS: Final[tuple[str, ...]] = (  # 7.4: bad-is-TRUE Nouls
    "vol.explained_by_event",
    "text.pending_binary",
    "text.market_stress",
    "text.clearly_negative",
    "text.clearly_positive",
)
TIER_ZERO_WHICH: Final[tuple[str, ...]] = ("score", "peak", "env")  # 7.6: tier_S, tier_peak, tier_env
PERTURB_VARIANTS: Final[tuple[str, ...]] = ("opt_perm", "key_perm", "bucket_only")  # 7.8

# the closed family list of 7.9, verbatim
REASON_FAMILIES: Final[tuple[str, ...]] = (
    "decider_failed_cycle:*",
    "dq:insufficient",
    "news_pipeline_error",
    "state_rejected",
    "veto:regime:*",
    "gate:direction:*",
    "gate:vol_stance:*",
    "map:*",
    "crosscheck:*",
    "fit:*",
    "veto:<qid>:hard|uncertain",
    "score:below_min",
    "tier:zero:*",
    "perturb:*",
    "hysteresis:released",
    "hold",
)

# every concrete code of those families, in the evaluation order of 7.2 ...
REASON_CODES: Final[tuple[str, ...]] = (
    *(f"decider_failed_cycle:{name}" for name in DECIDER_ERROR_CLASSES),  # step 0 (INV-05)
    "dq:insufficient",  # step 1
    "news_pipeline_error",
    "state_rejected",  # 7.9: state rejected / too large
    *(f"veto:regime:{label}" for label in REGIME_VETO_LABELS),  # step 2
    "gate:direction:conflicting",  # step 3
    "gate:direction:p_top",
    "gate:direction:margin",
    "gate:vol_stance:unclear",  # step 4
    "gate:vol_stance:p_top",
    "gate:vol_stance:margin",
    "map:no_structure",  # step 5
    "map:disabled",
    *(f"crosscheck:{name}" for name in CROSSCHECK_NAMES),  # step 6
    "fit:no_trade",  # step 7
    "fit:disagrees_with_mapping",
    "fit:p_top",
    "fit:margin",
    *(f"veto:{qid}:{band}" for qid in VETO_IDS for band in ("hard", "uncertain")),  # step 8
    "score:below_min",  # step 9
    *(f"tier:zero:{which}" for which in TIER_ZERO_WHICH),  # step 10
    "perturb:decider_failed",  # step 11
    "hysteresis:released",  # 7.7
    "hold",
)
REASONS: Final[frozenset[str]] = frozenset(REASON_CODES)
# ... plus ONE open family: `perturb:disagree:<variant>:<reason>` (7.2 step 11), <variant> in PERTURB_VARIANTS, <reason> non-empty
PERTURB_DISAGREE_PREFIX: Final[str] = "perturb:disagree:"


def is_reason(code: str) -> bool:
    """True iff `code` belongs to the closed reason vocabulary of 7.9."""
    if code in REASONS:
        return True
    if code.startswith(PERTURB_DISAGREE_PREFIX):
        variant, sep, reason = code[len(PERTURB_DISAGREE_PREFIX) :].partition(":")
        return variant in PERTURB_VARIANTS and sep == ":" and reason != ""
    return False


# ======================================================================================================================
# Post-DECISION outcomes: RISK_VERDICT.reject_codes = GATE_CODES | "candidate:<CANDIDATE_REJECTS code>" | "risk:<RISK_CODES code>"
# ======================================================================================================================

GATE_CODES: Final[tuple[str, ...]] = ("gate:kill_active", "gate:halt_entries", "gate:deadline_missed")  # 7.9

# 9.1: the codes of each ordered check of approve(); index i = check i + 1
RISK_CHECKS: Final[tuple[tuple[str, ...], ...]] = (
    ("kill_active", "halt_entries"),  # 1
    ("diagnostic_run",),  # 2
    ("underlying_not_allowed",),  # 3
    ("structure_not_allowed",),  # 4
    ("not_defined_risk",),  # 5
    ("close_only_reduces",),  # 6
    ("past_order_cutoff",),  # 7
    ("clock_skew",),  # 8
    ("stale_quote", "crossed_quote"),  # 9
    ("dte_window", "expiry_policy"),  # 10
    ("event_blackout", "exdiv_short_call"),  # 11
    ("dup_underlying_direction", "reentry_cooldown", "same_direction_cap"),  # 12
    ("max_open_structures",),  # 13
    ("max_new_per_day",),  # 14
    ("max_loss_per_trade",),  # 15
    ("agg_max_loss",),  # 16
    ("buying_power",),  # 17
    ("qty_cap", "notional_cap"),  # 18
    ("price_increment", "limit_sign", "beyond_natural"),  # 19
    ("adverse_drift",),  # 20
    ("order_rate", "attempt_cap"),  # 21
)
SIZE_ZERO: Final[str] = "size_zero"  # check 15 / size_entry: qty reduced to 0 ("risk:size_zero")
RISK_CODES: Final[tuple[str, ...]] = (*(code for check in RISK_CHECKS for code in check), SIZE_ZERO)
# the delayed-fill recheck (9.3): "recheck_failed:<code>", <code> = the failing check's RISK_CODES code
RECHECK_FAILED_PREFIX: Final[str] = "recheck_failed:"


def is_risk_code(code: str) -> bool:
    """True iff `code` is a RiskCheck.code / the part after "risk:" of a reject code (9.1, 9.3)."""
    if code.startswith(RECHECK_FAILED_PREFIX):
        return code[len(RECHECK_FAILED_PREFIX) :] in RISK_CODES
    return code in RISK_CODES


LIQUIDITY_REJECTS: Final[tuple[str, ...]] = ("liq:bid", "liq:crossed", "liq:spread", "liq:oi")  # structmath.leg_liquidity_rejects (8)
DELTA_UNREACHABLE_PREFIX: Final[str] = "delta_target_unreachable:"
DELTA_UNREACHABLE_LEGS: Final[tuple[str, ...]] = ("short", "long", "short_put", "short_call")  # verticals / long singles / condor wings
CANDIDATE_REJECTS: Final[tuple[str, ...]] = (  # section 8
    "no_expiry_in_window",
    *(f"{DELTA_UNREACHABLE_PREFIX}{leg}" for leg in DELTA_UNREACHABLE_LEGS),
    "width",
    "exceeds_risk_budget",
    *LIQUIDITY_REJECTS,
    "credit_to_width",
    "debit_to_width",
    "economics_invalid",
    "exdiv_short_call",
)

FILL_REJECTS: Final[tuple[str, ...]] = (  # 10.4: FillModel.check codes; band-independent
    "no_quote",
    "crossed_or_locked",
    "wide_spread",
    "size",
    "open_interest",
    "stale_quote",
    "missing_contract",
)


def is_reject_code(code: str) -> bool:
    """True iff `code` may appear in RiskVerdict.reject_codes (7.9): a gate code, `candidate:<code>` or `risk:<code>`."""
    if code in GATE_CODES:
        return True
    if code.startswith("candidate:"):
        return code[len("candidate:") :] in CANDIDATE_REJECTS
    if code.startswith("risk:"):
        return is_risk_code(code[len("risk:") :])
    return False


# ======================================================================================================================
# Smaller closed vocabularies of section 2 (ledger payloads, forecasts, run metadata)
# ======================================================================================================================

KILL_STEPS: Final[tuple[str, ...]] = (  # 2.11 KILL.step
    "tripped",
    "cancelled",
    "close_submitted",
    "fallback_legs",
    "flat_verified",
    "suspended",
    "not_flat",
    "locked",
)
DECISION_TEXT_MODES: Final[tuple[str, ...]] = ("on", "off", "no_archive")  # 2.11 DECISION.text (5.6)
# Forecast.missing_reason (2.7): the DeciderError class name, "state_rejected" or "requests_suppressed"
MISSING_REASONS: Final[tuple[str, ...]] = (*DECIDER_ERROR_CLASSES, "state_rejected", "requests_suppressed")
NEWS_REASONS: Final[tuple[str, ...]] = (  # RunMeta.news_reason (2.8, section 4)
    "explicit_on",
    "explicit_off",
    "auto_keys_present",
    "auto_no_keys",
    "text_probe_pending",
)
# OPEN lists ("..." in 2.11): the types the spec names; new types may be added by their writers
KNOWN_RISK_EVENT_TYPES: Final[tuple[str, ...]] = (
    "daily_loss_halt",
    "halt_set",
    "halt_cleared",
    "cycle_started",
    "entries_done",
    "trigger_seen",
    "bp_model_drift",
    "deadline_missed",
    "requests_suppressed",
    "text_watch",
)
KNOWN_ANOMALY_TYPES: Final[tuple[str, ...]] = (
    "stale_mark",
    "anomaly_settlement",
    "assignment_sim",
    "text_veto_nonclear_on_empty_news",
    "parity_basis_suspect",
    "atm_iv_divergence",
)
