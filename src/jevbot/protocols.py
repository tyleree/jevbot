"""Every `typing.Protocol` of the project plus `CycleContext` (DESIGN.md section 3) - a frozen contract.

All are `typing.Protocol` (structural). `@runtime_checkable` on `Decider`, `Broker`, `ChainProvider` (used by `doctor` and the
wave-1 contract test). No Protocol method takes or returns a vendor SDK type. Data members (`MarketView.as_of`, `Decider.name`,
`BookP.last_key`, ...) are read-only `@property` members: no consumer assigns to them, and a settable-attribute declaration would make
mypy reject every frozen Struct / frozen dataclass / `@property` implementation. An implementation may satisfy them with a plain
instance or class attribute, a frozen field or a property.

The classes below are the fenced `python` contract blocks of sections 3.1-3.6, copied verbatim: names, member order, signatures and
the comment text that carries the semantics. `tests/guards/test_import_rules.py` parses DESIGN.md at test time and compares every
member with this module. Layout convention (the spec prints the comments to the right of or below a signature; here, so that the
formatter keeps them stable): **a comment block documents the member that FOLLOWS it.**

Two signatures that the spec prints inside those blocks belong to other modules and appear here as comments only:
`cycle.decide_batch` (3.3) and the concrete `data.view.DataView` / `data.series.PitTable` (3.2).

`CycleContext` is typed ONLY with Protocols (plus `Config` and `RunMeta`), so this module passes `mypy --strict` before any
implementation exists and wave-2 packages can stub every collaborator structurally. This module therefore imports nothing from
the package except `types`, `config` and `errors` - never an implementation module.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from jevbot.config import Config
from jevbot.errors import DeciderError
from jevbot.types import (
    AccountSnapshot,
    ApprovedOrder,
    BandPrices,
    BrokerActivity,
    BrokerPosition,
    BuiltState,
    CachedAnswer,
    Candidate,
    CandidateReject,
    Cents,
    ChainSnapshot,
    ClockReading,
    CycleGate,
    DecisionRequest,
    DecisionResult,
    EntryDecision,
    EntryFacts,
    EvidenceTier,
    ExitReason,
    Fidelity,
    HealthSnapshot,
    KillState,
    KillTrigger,
    LedgerEntry,
    LedgerKind,
    LegFill,
    ManageDecision,
    ManageFacts,
    Micros,
    NewsItem,
    OrderIntent,
    OrderLeg,
    OrderState,
    PortfolioState,
    Position,
    Ppm,
    ProvenanceInput,
    RiskVerdict,
    RunMeta,
    ScheduledEvent,
    Slot,
    SnapshotKey,
    Structure,
    StructureKind,
    TriggerAction,
    Variant,
)

__all__ = [
    "BookP",
    "Broker",
    "Calendar",
    "CandidateGeneratorP",
    "ChainProvider",
    "Clock",
    "CorporateActionsSource",
    "CycleContext",
    "DailyBarsSource",
    "Decider",
    "DecisionCache",
    "DecisionRulesP",
    "EventSource",
    "FillModel",
    "KillSwitch",
    "Ledger",
    "MarketView",
    "NewsArchiveSource",
    "NewsSource",
    "OrderWorker",
    "RiskEngine",
    "SnapshotSource",
    "SpendLedger",
    "StateBuilderP",
]

# ======================================================================================================================
# 3.1 Time
# ======================================================================================================================


class Calendar(Protocol):
    def is_session(self, d: date) -> bool: ...

    # inclusive
    def sessions(self, start: date, end: date) -> list[date]: ...

    # UTC; early closes honoured (D4, G4)
    def open_close(self, session: date) -> tuple[datetime, datetime]: ...
    def is_early_close(self, session: date) -> bool: ...

    # strictly after d
    def next_session(self, d: date, n: int = 1) -> date: ...

    # strictly before d
    def prev_session(self, d: date, n: int = 1) -> date: ...

    # d if is_session(d) else prev_session(d). THE map expiry -> last_session:
    # d may be ANY calendar date (a Saturday-dated monthly, a holiday)
    def prev_or_same_session(self, d: date) -> date: ...

    # number of sessions in (a, b]
    def sessions_between(self, a: date, b: date) -> int: ...

    # session whose [open, close] contains ts
    def session_of(self, ts: datetime) -> date | None: ...

    # THE way to express a cut-off (INV-13); negative = after the close
    def offset_from_close(self, session: date, minutes_before: int) -> datetime: ...

    # knowable_at of EOD series
    def next_open_after(self, ts: datetime) -> datetime: ...


class Clock(Protocol):
    # tz-aware UTC. SimClock (backtest) or BrokerClock (paper)
    def now(self) -> datetime: ...

    # None for SimClock
    def reading(self) -> ClockReading | None: ...

    # force a broker re-sync (no-op for SimClock)
    def sync(self) -> ClockReading | None: ...


# Implementations: `cal.XnysCalendar` (exchange_calendars "XNYS"), `cal.SimClock` (`set(ts)` by the backtest loop),
# `paper.clock.AlpacaCalendar` (`get_calendar()` rows; naive-Eastern `open`/`close` localised with `zoneinfo("America/New_York")`;
# cross-checked against XNYS at boot: the earlier close wins and an alert is raised), `paper.clock.BrokerClock` (11.4).

# ======================================================================================================================
# 3.2 Data
# ======================================================================================================================


@runtime_checkable
class ChainProvider(Protocol):
    @property
    def fidelity(self) -> Fidelity: ...
    @property
    def source(self) -> str: ...

    def underlyings(self) -> tuple[str, ...]: ...

    # ordered; snapshots that exist
    def keys(self, underlying: str, start: date, end: date) -> list[SnapshotKey]: ...

    # ENRICHED (own iv/delta/fwd, spot); no PIT logic here
    def get_chain(self, underlying: str, key: SnapshotKey) -> ChainSnapshot | None: ...
    def manifest_hash(self) -> str: ...


class NewsSource(Protocol):
    # only knowable_at <= as_of, newest first; summary=None when updated_at > as_of (B6.3 rule 4)
    def items(self, underlying: str, as_of: datetime, lookback_hours: int) -> tuple[NewsItem, ...]: ...

    # True iff the archive's fetched ranges cover this session ("no archive" must be distinguishable from "no news")
    def covered(self, underlying: str, session: date) -> bool: ...


class EventSource(Protocol):
    # only rows with scheduled == True and cancelled == False, knowable_at <= as_of and start <= event_date <= end
    def events(self, as_of: datetime, start: date, end: date, underlying: str | None = None) -> tuple[ScheduledEvent, ...]: ...

    # event kinds that have at least one verified row
    def coverage(self) -> tuple[str, ...]: ...


# Archive sources: what `data fetch news|exdiv|bars` needs from Alpaca. data/fetch.py (WP01) takes them BY INJECTION and is tested with fakes;
# the adapters (paper/live_data.py: AlpacaNews, AlpacaCorporateActions, AlpacaBars) belong to WP10. No vendor type crosses these signatures.
class NewsArchiveSource(Protocol):
    # padded-end rule of B6.3 applied by the adapter
    def fetch_news(self, symbols: Sequence[str], start: datetime, end: datetime) -> Iterator[NewsItem]: ...


class CorporateActionsSource(Protocol):
    # kind "ex_dividend"; knowable_at = fetched_at
    def cash_dividends(self, symbols: Sequence[str], start: date, end: date) -> tuple[ScheduledEvent, ...]: ...


class DailyBarsSource(Protocol):
    # session, open, high, low, close, volume; UNADJUSTED
    def raw_daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame: ...


class MarketView(Protocol):
    """The read surface of DataView. Every consumer (features, state, candidates, risk, outcomes, cycle, SimBroker, fills)
    is typed against MarketView so it can be unit-tested with tests/fixtures/fake_view.py."""

    @property
    def as_of(self) -> datetime: ...
    @property
    def key(self) -> SnapshotKey: ...
    @property
    def session(self) -> date: ...
    @property
    def calendar(self) -> Calendar: ...
    @property
    def fidelity(self) -> Fidelity: ...

    # FRAME CONVENTIONS (the double and DataView agree on them; consumers rely on them): every `session` column / index is
    # datetime64[ns] (like the chain's `expiry` / `last_session`); a MISSING table (`daily`, `bars`, `volidx:<NAME>`, `rates`)
    # raises DataUnavailable, an existing table with no knowable rows returns an empty frame / Series.

    # DataUnavailable if none; PitViolation if knowable_at > as_of
    def chain(self, underlying: str) -> ChainSnapshot: ...

    # chain(underlying).spot  (the decision-time reference price `ref`)
    def spot(self, underlying: str) -> Cents: ...

    # close_c of the last n COMPLETED sessions (< session), oldest first, int cents:
    # exactly ONE value per session (close_c lives on the session's `eod` row only, 5.4).
    # An int64 Series named "close_c" indexed by session (DatetimeIndex "session")
    def closes(self, underlying: str, n: int) -> pd.Series: ...

    # keyed read of that session's close_c; PitViolation if its close_knowable_at > as_of;
    # DataUnavailable when the eod row exists but carries no recorded close yet (or has no row)
    def close(self, underlying: str, session: date) -> Cents: ...

    # last n COMPLETED sessions: columns session, open, high, low, close (float64), oldest first.
    # WITHIN-BAR RATIOS ONLY (5.3)
    def bars(self, underlying: str, n: int) -> pd.DataFrame: ...

    # open(today) / file_close(prev session); today's OPEN column is knowable at open + 60 s
    # while today's high / low / close / volume stay null until the next open (column gating, below)
    def today_open_ratio(self, underlying: str) -> float | None: ...

    # derived daily series (13.2): exactly ONE ROW PER SESSION - for each of the last n-1 past
    # sessions the row of the DESIGNATED SLOT (the slot equal to this view's key.slot, else that
    # session's `eod` row), then this snapshot's own row last. Every look-back window in 5.3 is
    # therefore indexed by SESSION, in every mode, whether the archive holds 1 or 3 slots per session.
    # The 13.2 columns; `close_c` is nulled where `close_knowable_at` is null or > as_of (column gating)
    def daily(self, underlying: str, n: int) -> pd.DataFrame: ...

    # last n closes with knowable_at <= as_of (newest is normally session-1, D22); a float64 Series indexed by session
    def vol_index(self, name: str, n: int) -> pd.Series: ...

    # 13-week bill coupon-equivalent, decimal; last knowable
    def rate(self) -> float: ...

    # knowable (knowable_at <= as_of), scheduled, non-cancelled rows with start <= event_date <= end. underlying=None serves
    # EVERY such row; underlying=U serves the market-wide rows (underlying None: fomc_decision / cpi / nfp) PLUS U's own rows
    # (ex_dividend) - never another underlying's. The entry state's events block, risk check 11 and the ex-dividend entry
    # block all read it with underlying=U.
    def events(self, start: date, end: date, underlying: str | None = None) -> tuple[ScheduledEvent, ...]: ...
    def event_coverage(self) -> tuple[str, ...]: ...
    def news(self, underlying: str, lookback_hours: int) -> tuple[NewsItem, ...]: ...
    def news_covered(self, underlying: str) -> bool: ...

    # every read is logged for the provenance sidecar
    def touched(self) -> tuple[ProvenanceInput, ...]: ...


# `data.view.DataView` is the one concrete implementation (WP01; signatures printed in 3.2, reproduced here as a comment):
#
#   class DataView:                                          # implements MarketView
#       def __init__(self, *, key: SnapshotKey, as_of: datetime, calendar: Calendar, chains: ChainProvider,
#                    tables: Mapping[str, PitTable], news: NewsSource, events: EventSource) -> None
#
#   class PitTable:                                          # data/series.py - every non-chain dataset is one of these
#       def __init__(self, df: pd.DataFrame, *, name: str, key: str | tuple[str, ...] = "session",
#                    column_knowable: Mapping[str, str] | None = None) -> None
#           # requires a tz-aware ROW-level `knowable_at` column (the earliest moment ANY part of the row may be seen), else DataError; `key` must be unique.
#           # column_knowable maps a VALUE column to its own knowable-at column for values that become visible LATER than the row.
#       def asof(self, as_of: datetime) -> pd.DataFrame      # rows with knowable_at <= as_of (range reads FILTER, then ASSERT max(knowable_at) <= as_of);
#                                                            # every column_knowable value whose own timestamp is > as_of (or null) is returned as NULL
#       def row(self, key: object, as_of: datetime) -> pd.Series   # keyed reads RAISE PitViolation when the ROW's knowable_at > as_of; DataUnavailable when absent;
#                                                            # gated columns that are not yet knowable come back NULL
#       def value(self, key: object, column: str, as_of: datetime) -> object   # keyed read of ONE column: RAISES PitViolation when that column's knowable-at
#                                                            # (its column_knowable timestamp, else the row's) is > as_of. MarketView.close() is this call.
#       def sha256(self) -> str
#
# PIT semantics (D24): range reads filter then assert; keyed reads raise; `DataView.chain()` re-checks `knowable_at <= as_of`, so a
# buggy provider also raises. `StateBuilder`, `RiskEngine`, `SimBroker`, `CandidateGenerator`, `FillModel` and the outcome resolver
# receive only a `MarketView`. Tables and their gating (the only per-column rules in the project):
#
#   table                       key                              row `knowable_at`                     `column_knowable`
#   `bars:<UND>`                `session`                        `open_knowable_at` = open(D) + 60 s   `high`, `low`, `close`, `volume` -> `hlcv_knowable_at` (= next session open)
#   `daily:<UND>`               `(session, slot)`                that snapshot's `knowable_at`         `close_c` -> `close_knowable_at` (non-null on `eod` rows only)
#   `volidx:<NAME>`, `rates`    `session`                        next session open                     none
#   `events`                    `(kind, event_date, underlying)` per row (5.1)                         none

# ======================================================================================================================
# 3.3 Decider, cache, spend
# ======================================================================================================================


@runtime_checkable
class Decider(Protocol):
    # "live_jev" | "replay_jev" | "mock_jev" | "baseline:<...>"
    @property
    def name(self) -> str: ...
    # "jev-1.13.0" | "mock-1" | "baseline"
    @property
    def model(self) -> str: ...

    # (There is NO needs_state switch: states are ALWAYS built. DecisionRequest.state / state_hash, EntryFacts for the cross-checks and the
    #  expected-move integers of the forecasts all come from the BuiltState, for every decider including the random and always-enter baselines.)
    #
    # decide(): RAISES DeciderError subclasses (fail closed), CacheMissError (abort), ModelMismatchError (abort / kill).
    # Must be thread-safe: decide_batch() calls it from a small thread pool.
    def decide(self, req: DecisionRequest) -> DecisionResult: ...
    def close(self) -> None: ...


# cycle.py (WP09) - printed inside the 3.3 contract block, implemented there:
#
#   def decide_batch(decider: Decider, reqs: Sequence[DecisionRequest], max_workers: int, *, mode: RunMode) -> list[DecisionResult | DeciderError]
#       # cycle.py. Order-preserving. ONE uniform convention: DeciderError instances are RETURNED in place (per-request fail-closed);
#       # CacheMissError and ModelMismatchError are RE-RAISED after the pool drains - and so is SpendLimitError when mode is RunMode.BACKTEST
#       # (D9 hard stop: a record run must not grind on, committing session after session of MISSING forecasts). In RunMode.PAPER a
#       # SpendLimitError is returned in place like any other DeciderError (entries halt, the service lives on). Nothing else may escape.


class DecisionCache(Protocol):
    def get_many(self, namespace: str, keys: Sequence[str]) -> dict[str, CachedAnswer]: ...

    # ONE transaction, all-or-nothing, INSERT OR IGNORE (first write wins, never evicted). index = (session, underlying, request_kind, variant).
    # question_sets.question_set_id is taken from rows[0].question_set_id (all rows of one request share it; asserted).
    # An existing row whose answer differs is KEPT and the divergence is recorded in table `nondeterminism` (free evidence for unknown U1).
    def put_request(
        self, namespace: str, rows: Sequence[CachedAnswer], state_json: str, questions_json: str, index: tuple[date, str, str, str]
    ) -> None: ...

    # all keys present (used by the shadow replay, 11.8)
    def has_request(self, namespace: str, keys: Sequence[str]) -> bool: ...

    # refresh=True requires that the namespace has NO rows yet (ConfigError otherwise). diagnostic=True sets namespaces.diagnostic = 1
    # (probe / leakage namespaces); an existing namespace's flag can never be changed (ConfigError on a mismatch).
    def ensure_namespace(
        self, namespace: str, model: str, model_release_date: date, *, refresh: bool, diagnostic: bool = False
    ) -> None: ...

    # read by the run-start guard (a diagnostic namespace can only back runs with purpose "diagnostic") and surfaced to risk check 2 through
    # the RUN_START flags: run_backtest / the paper boot add the flag "diagnostic" when it is True, and check 2 rejects every order of such a run.
    def is_diagnostic(self, namespace: str) -> bool: ...
    def stats(self, namespace: str | None = None) -> dict[str, int]: ...

    # sha256 over "key:sha256(answer_json)\n" for all rows of the namespace ORDER BY key
    def manifest_hash(self, namespace: str) -> str: ...


# jev/spend.py, backed by $JEVBOT_DATA/state/spend.sqlite (shared by EVERY entry point)
class SpendLedger(Protocol):
    # "paper" (the paper service) | "batch" (backtests, probes, leakage, baselines). One guard
    # instance is bound to one scope; counters and the sticky block are PER SCOPE, so a
    # backtest that hits its ceiling can never halt the paper service's entries.
    @property
    def scope(self) -> str: ...

    # returns a reservation id; SpendLimitError when the run total (batch scope only) or the
    # scope's UTC-day total would exceed its ceiling (paper: jev.spend.paper_max_input_tokens_per_day;
    # batch: jev.spend.max_input_tokens_per_day)
    def reserve(self, run_id: str, tokens: int) -> int: ...
    def commit(self, reservation_id: int, tokens: int, *, estimated: bool) -> None: ...

    # (run_total, utc_day_total of THIS scope across all its runs)
    def totals(self, run_id: str | None = None) -> tuple[int, int]: ...

    # sticky for the rest of the UTC day once THIS scope's ceiling was hit
    def blocked(self) -> bool: ...


# ======================================================================================================================
# 3.4 Execution
# ======================================================================================================================


@runtime_checkable
class Broker(Protocol):
    # "sim" | "alpaca_paper" | "fake"
    @property
    def name(self) -> str: ...

    def account(self) -> AccountSnapshot: ...

    # leg level, exactly as the broker reports, INCLUDING equity
    def positions(self) -> tuple[BrokerPosition, ...]: ...
    def open_orders(self) -> tuple[OrderState, ...]: ...

    # idempotent on client_order_id; raises BrokerRejected / BrokerAmbiguous.
    # LEDGER-FREE: a Broker has no Ledger and no Book. ORDER_STATUS entries are written only by
    # reconcile.record_order_status (9.6), which the order workers call around submit()
    def submit(self, order: ApprovedOrder) -> OrderState: ...

    # by CLIENT id (D18); None = broker has no such order
    def get_order(self, client_order_id: str) -> OrderState | None: ...

    # returns the post-cancel state; waits for a terminal status up to a deadline
    def cancel(self, client_order_id: str) -> OrderState: ...
    def cancel_all(self) -> int: ...

    # OPASN / OPEXC / OPEXP / OPTRD; () for sim
    def activities(self, since: date) -> tuple[BrokerActivity, ...]: ...

    # Alpaca suspend_trade (needs the full config object); flag in sim
    def set_suspended(self, suspended: bool) -> None: ...

    # SimBroker: price queued orders, update order states. Alpaca: no-op.
    def on_snapshot(self, view: MarketView) -> None: ...


class FillModel(Protocol):
    # vocab.FILL_REJECTS codes; () = fillable. Band-independent. mandatory=True always returns ().
    # Usability is PER SIDE (10.4): a leg whose position_intent is sell_to_close with bid == 0 and ask > 0 is NOT a reject (it is sold at 0).
    def check(self, legs: Sequence[OrderLeg], qty: int, chain: ChainSnapshot, *, mandatory: bool) -> tuple[str, ...]: ...

    # (signed net cents/share per band, per-leg detail, quality "ok" | "degraded"); mandatory applies the forced-fill penalty (10.4)
    def price(self, legs: Sequence[OrderLeg], chain: ChainSnapshot, *, mandatory: bool) -> tuple[BandPrices, tuple[LegFill, ...], str]: ...

    # (liq_value: longs at bid / shorts at ask, mid_value, stale); a long leg with bid == 0 and ask > 0 is a VALID mark of 0;
    # `last` is carried only when a quote is MISSING or a short leg has no ask (10.5)
    def liquidation(self, structure: Structure, chain: ChainSnapshot, last: tuple[int, int] | None) -> tuple[int, int, bool]: ...
    def fees_micro(self, legs: Sequence[OrderLeg], qty: int, leg_fills: Sequence[LegFill]) -> Micros: ...


class RiskEngine(Protocol):
    def pre_cycle(
        self, pf: PortfolioState, health: HealthSnapshot
    ) -> tuple[CycleGate, tuple[tuple[KillTrigger, TriggerAction, str], ...]]: ...

    # daily-loss halt (vs pf.day_start_equity = the PREVIOUS
    # session's end equity), drawdown kill; evaluated at the decision snapshot
    def on_mark(self, pf: PortfolioState) -> tuple[tuple[KillTrigger, TriggerAction, str], ...]: ...

    # code-only exits; run first and always win
    def hard_exit(self, pos: Position, view: MarketView) -> ExitReason | None: ...

    # floor(max_loss_per_trade_pct * equity_basis * LOWEST non-zero tier):
    # the per-contract max-loss budget handed to CandidateGenerator.build (8, 9.3)
    def budget_floor(self, pf: PortfolioState) -> Cents: ...

    # requested qty (9.3); 0 = no trade
    def size_entry(self, cand: Candidate, tier_ppm: Ppm, pf: PortfolioState, approved_so_far: Sequence[ApprovedOrder]) -> int: ...

    # the ONLY constructor of ApprovedOrder. Called for EVERY attempt of EVERY order, including closes and kill orders.
    # now (REQUIRED) is the caller's time: backtest passes view.as_of, paper passes ctx.clock.now(). approve() NEVER reads a clock itself;
    # `now` feeds check 7 (and approved_at). `clock` is only the broker ClockReading for check 8 (skew); None in backtests.
    # limit=None means "use intent.limit_start"; market=True (KILL last resort, market hours only) yields ApprovedOrder.limit = None.
    # view may be the last available view when the kill switch runs outside a cycle; checks that need quotes are skipped for KILL.
    def approve(
        self,
        intent: OrderIntent,
        pf: PortfolioState,
        view: MarketView,
        *,
        now: datetime,
        attempt: int = 0,
        limit: int | None = None,
        cand: Candidate | None = None,
        approved_so_far: Sequence[ApprovedOrder] = (),
        clock: ClockReading | None = None,
        market: bool = False,
    ) -> tuple[RiskVerdict, ApprovedOrder | None]: ...

    # SimBroker only, at the delayed fill snapshot `view`: (qty, reject code). qty still satisfies per-trade / aggregate / BP limits at the ACTUAL
    # fill price at 1.0x (never looser); checks 10 (dte window) and 11 (event blackout, ex-dividend short call) are RE-EVALUATED as of the fill
    # snapshot. qty == 0 => cancel with `risk:recheck_failed:<code>` (9.3).
    def recheck_fill(self, intent: OrderIntent, net: BandPrices, pf: PortfolioState, view: MarketView) -> tuple[int, str | None]: ...


class KillSwitch(Protocol):
    def state(self) -> KillState: ...
    def event_id(self) -> str | None: ...

    # idempotent; persists (file fsync + ledger) BEFORE returning
    def trip(self, trigger: KillTrigger, detail: str) -> None: ...

    # advance the flatten sequence one round (9.5). The
    # implementation is constructed with the run's Ledger, BookP, RiskEngine, Clock and FillModel: it ledgers KILL steps and ORDER_INTENTs itself and
    # submits through the same protocol as the order workers (reconcile.record_order_status around broker.submit; approve(now=clock.now()))
    def step(self, broker: Broker, view: MarketView | None, market_open: bool) -> KillState: ...
    def rearm(self, rearm_file_text: str, *, reset_peak: bool, note: str) -> None: ...


class Ledger(Protocol):
    # every implementation computes `LedgerEntry.hash` with `canon.ledger_entry_hash` (THE 2.7 formula, incl. the frozen spelling of the
    # `session` / `as_of` columns: `canon.render_session` / `canon.render_as_of`), on append AND in verify() over the persisted texts (13.4),
    # so that SqliteLedger and the MemoryLedger double give the same head hash for the same entries (INV-19, INV-24). A payload with a
    # float / datetime / non-string key is a TypeError BEFORE anything is appended.
    def append(
        self, kind: LedgerKind, session: date, as_of: datetime, payload: Mapping[str, Any], *, sidecar: Mapping[str, Any] | None = None
    ) -> LedgerEntry: ...

    # backtest: once per session; paper: after every append (synchronous=FULL)
    def commit(self) -> None: ...

    # per-session mode: drop the uncommitted session (abort paths of run_backtest, 10.1) - the entries AND the fill claims, states and meta
    # written since the last commit (one SQLite transaction); no-op in per-append mode
    def rollback(self) -> None: ...

    # (seq, hash); (0, "0"*64) when empty
    def head(self) -> tuple[int, str]: ...

    # entries with seq > since_seq, ascending
    def entries(self, kind: LedgerKind | None = None, since_seq: int = 0) -> Iterator[LedgerEntry]: ...

    # recompute the chain; raises LedgerCorrupt
    def verify(self, from_seq: int = 1) -> None: ...

    # INSERT into the UNIQUE `fill_ids` table; False = already booked (the dedupe of the ONE fill path)
    def claim_fill(self, fill_id: str) -> bool: ...

    # unhashed `states` + `state_index` tables (13.4): every built state of the run, indexed by (session, underlying, request_kind, variant).
    # The same (session, underlying, request_kind, variant) with the SAME state_hash is an idempotent no-op (the 10.1 restart case); with a
    # DIFFERENT state_hash it raises InvariantError (a rebuilt state must be byte-identical: a determinism bug, never silently overwritten)
    def put_state(self, state_hash: str, state_json: str, *, session: date, underlying: str, request_kind: str, variant: str) -> None: ...

    # (session, underlying, state_json), ordered by (session, underlying). THE source of baseline 6's recorded states (12.4) and of the
    # probe suites' `run:RUN_ID` state source (6.8)
    def get_states(self, request_kind: str, variant: str = "base") -> Iterator[tuple[date, str, str]]: ...
    def get_meta(self, key: str) -> str | None: ...

    # write-once keys: the same value again is a no-op; a DIFFERENT value for an existing key raises InvariantError
    def set_meta(self, key: str, value: str) -> None: ...


# ======================================================================================================================
# 3.5 Paper-only seams
# ======================================================================================================================


# paper/runner.py depends on this, not on alpaca
class SnapshotSource(Protocol):
    # fetch + record chains / underlying / news for every underlying; returns the recorded key
    def take(self, slot: Slot) -> SnapshotKey: ...

    # official closes (raw / unadjusted) -> daily series
    def record_close(self, session: date) -> None: ...

    # DataView over the bytes just recorded (so a later replay sees identical inputs)
    def view(self, key: SnapshotKey) -> MarketView: ...


#   backtest (backtest.sim_order_worker): approve(attempt 0, now=view.as_of) + submit for each; fills arrive at the next snapshot (or at once under same_snapshot_worst)
#   paper    (paper.runner.paper_order_worker): the price ladder of 11.6 (approve -> submit -> wait -> cancel -> approve(attempt+1) ...), exits before entries
#   BOTH follow the same submit protocol (9.6): record_order_status(SUBMITTING) -> broker.submit -> record_order_status(SUBMITTED | REJECTED | UNKNOWN)
OrderWorker = Callable[[Sequence[tuple[OrderIntent, Candidate | None]], "CycleContext", MarketView], None]

# ======================================================================================================================
# 3.6 Builder / book Protocols and cycle wiring
#
# The four Protocols below restate the signatures of sections 5, 7, 8 and 10.8 (those sections describe behaviour; THESE are
# the contract).
# ======================================================================================================================


# implemented by portfolio.Book (10.8)
class BookP(Protocol):
    # the last snapshot key folded in (None before the first SESSION_START); resume starts after it (10.1)
    @property
    def last_key(self) -> SnapshotKey | None: ...

    def apply(self, entry: LedgerEntry) -> None: ...
    def state(self) -> PortfolioState: ...
    def intent(self, intent_id: str) -> OrderIntent: ...
    def has_intent(self, intent_id: str) -> bool: ...
    def filled_qty(self, client_order_id: str) -> int: ...
    def open_orders(self) -> tuple[tuple[OrderIntent, OrderState], ...]: ...
    def leg_positions(self) -> dict[str, int]: ...


# implemented by state.StateBuilder (section 5)
class StateBuilderP(Protocol):
    def entry(self, view: MarketView, underlying: str) -> BuiltState | None: ...
    def entry_text(self, view: MarketView, underlying: str, base: BuiltState) -> BuiltState | None: ...
    def manage(self, view: MarketView, pos: Position) -> BuiltState: ...
    def manage_text(self, view: MarketView, pos: Position, base: BuiltState) -> BuiltState | None: ...
    def variant(self, state: dict[str, Any], v: Variant) -> dict[str, Any]: ...


# implemented by rules.DecisionRules (section 7)
class DecisionRulesP(Protocol):
    # the section-4 `rules_hash` of the [rules] section this instance was built from
    @property
    def rules_hash(self) -> str: ...

    def decide_entry(
        self, underlying: str, decision_id: str, core: DecisionResult, text: DecisionResult | None, facts: EntryFacts
    ) -> EntryDecision: ...
    def confirm_entry(
        self,
        base: EntryDecision,
        core_variants: Mapping[Variant, DecisionResult | DeciderError],
        text: DecisionResult | None,
        facts: EntryFacts,
    ) -> EntryDecision: ...
    def decide_manage(
        self,
        pos: Position,
        decision_id: str,
        hard: ExitReason | None,
        core: DecisionResult | None,
        text: DecisionResult | None,
        facts: ManageFacts,
    ) -> ManageDecision: ...
    def rank(self, entries: Sequence[EntryDecision], order: Sequence[str]) -> list[EntryDecision]: ...


# implemented by candidates.CandidateGenerator (section 8)
class CandidateGeneratorP(Protocol):
    def build(self, kind: StructureKind, view: MarketView, underlying: str, *, budget_floor: Cents) -> Candidate | CandidateReject: ...


@dataclass
class CycleContext:
    cfg: Config
    meta: RunMeta
    calendar: Calendar
    clock: Clock
    broker: Broker
    decider: Decider
    cache: DecisionCache | None
    risk: RiskEngine
    kill: KillSwitch
    fill_model: FillModel
    ledger: Ledger
    book: BookP
    state_builder: StateBuilderP
    rules: DecisionRulesP
    candidates: CandidateGeneratorP
    order_worker: OrderWorker
    health: Callable[[MarketView], HealthSnapshot]
    # (decision_id, session) -> tier; the shadow replay injects the LIVE ledger's tiers (11.8)
    tier_of: Callable[[str, date], EvidenceTier]
    # "on" | "off" | "cached_only"  (cached_only: ask only when cache.has_request(...), else code-only)
    manage_jev: str
