"""Every enum and msgspec Struct shared across modules (DESIGN.md section 2) - a frozen contract.

All structs are `msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True` unless noted.
Enums are `enum.StrEnum`; the value is what is persisted. `Cents`, `Micros`, `Ppm`, `Bp` are plain `int` aliases.

Conventions (DESIGN "Conventions"): money is integer; option prices are cents per share (contract multiplier 100); strikes are
milli-dollars as in the OCC symbol (450.5 -> 450500); fees accrue in micro-dollars; signed net prices follow Alpaca's mleg
convention (positive = net debit, negative = net credit); probabilities inside hashed payloads are integer ppm; every datetime is
tz-aware UTC; `session` is an XNYS trading day. `expiry` is contract identity only - every time computation uses `last_session`.

This module depends on nothing inside the package except `errors.py` (occ.py, canon.py, ids.py, ... all import it). The two
identity strings that the spec phrases through those modules - `OptionContract.occ` (= `occ.format_occ`) and
`Structure.structure_id` (= sha256 of `canon.dumps_sorted(...)`) - are therefore computed here with the identical formulas.
"""

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final, Literal

import msgspec
import pandas as pd

from jevbot.errors import DataError, DataUnavailable

Cents = int
Micros = int
Ppm = int
Bp = int

# ======================================================================================================================
# 2.1 Enums and static tables
# ======================================================================================================================


class Fidelity(StrEnum):
    EOD_QUOTES = "EOD_QUOTES"
    RECORDED_INDICATIVE = "RECORDED_INDICATIVE"
    LIVE_INDICATIVE = "LIVE_INDICATIVE"
    SYNTHETIC = "SYNTHETIC"


class EvidenceTier(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    NONE = "NONE"  # NONE = synthetic data (never evidence)


class RunMode(StrEnum):
    BACKTEST = "backtest"
    PAPER = "paper"


class Slot(StrEnum):
    EOD = "eod"
    DEC = "dec"
    EXEC = "exec"


class Phase(StrEnum):  # which steps run_cycle executes (10.1)
    FULL = "full"
    DECIDE = "decide"
    SETTLE = "settle"
    CLOSE_OUT = "close_out"


class Right(StrEnum):
    CALL = "C"
    PUT = "P"


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class PositionIntent(StrEnum):
    BTO = "buy_to_open"
    STO = "sell_to_open"
    BTC = "buy_to_close"
    STC = "sell_to_close"


class StructureKind(StrEnum):
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    CALL_DEBIT = "call_debit_spread"
    PUT_DEBIT = "put_debit_spread"
    CALL_CREDIT = "call_credit_spread"
    PUT_CREDIT = "put_credit_spread"
    IRON_CONDOR = "iron_condor"


class Direction(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral_range"


class VolStance(StrEnum):
    SELL = "sell_premium"
    BUY = "buy_premium"
    LIMIT = "limit_vol_exposure"


class Band(StrEnum):
    ORATS = "orats"
    WORST = "worst"
    MID = "mid"


class OrderPurpose(StrEnum):
    OPEN = "open"
    CLOSE = "close"
    KILL = "kill"


class OrderStatus(StrEnum):
    INTENT = "intent"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    PARTIAL = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


TERMINAL_STATUSES: Final[frozenset[OrderStatus]] = frozenset(
    {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
)


class ExitReason(StrEnum):
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    TIME_EXIT = "time_exit"
    FORCE_EXPIRY = "force_exit_expiry"
    EX_DIVIDEND = "ex_dividend"
    ASSIGNMENT_RISK = "assignment_risk"
    JEV = "jev_discretionary"
    TEXT_CONFIRMED = "text_confirmed"
    CODE_DEFAULT = "code_default"
    KILL = "kill_switch"
    ANOMALY = "anomaly_settlement"
    ASSIGNMENT_SIM = "assignment_sim"
    # CODE_DEFAULT = the decider-down close of 7.7 step 2. There is NO text-only exit reason (INV-16).


# forced fills allowed; may go past natural by the pad
MANDATORY_EXITS: Final[frozenset[ExitReason]] = frozenset(
    {ExitReason.FORCE_EXPIRY, ExitReason.EX_DIVIDEND, ExitReason.ASSIGNMENT_RISK, ExitReason.KILL}
)


class RequestKind(StrEnum):
    ENTRY = "entry"
    ENTRY_TEXT = "entry_text"
    MANAGE = "manage"
    MANAGE_TEXT = "manage_text"
    PROBE = "probe"


# DECISION-level kind (2.10, 2.11): ONE decision covers the text-free and the text request of its subject
DecisionKind = Literal["entry", "manage"]


class Variant(StrEnum):
    BASE = "base"
    OPT_PERM = "opt_perm"
    KEY_PERM = "key_perm"
    BUCKET_ONLY = "bucket_only"


class CacheMode(StrEnum):
    RECORD = "record"
    REPLAY = "replay"
    REFRESH = "refresh"


class FillRule(StrEnum):
    NEXT_SNAPSHOT = "next_snapshot"
    SAME_SNAPSHOT_WORST = "same_snapshot_worst"


class QuestionRole(StrEnum):
    GATE = "gate"
    VETO = "veto"
    COMPOSITE = "composite"
    SIZING = "sizing"
    RANK = "rank"
    EVAL = "eval"


class InfoClass(StrEnum):  # G9 attribution in reports
    RESTATE = "restate"
    JUDGEMENT = "judgement"
    TEXT = "text"
    FORECAST = "forecast"


class Tri(StrEnum):
    CLEAR = "clear"
    UNCERTAIN = "uncertain"
    VETO = "veto"


class KillState(StrEnum):
    ARMED = "armed"
    TRIPPED = "tripped"
    FLATTENING = "flattening"
    NOT_FLAT = "not_flat"
    LOCKED = "locked"


class KillTrigger(StrEnum):
    OPERATOR = "operator"
    DRAWDOWN = "drawdown"
    RECONCILE_MISMATCH = "reconcile_mismatch"
    MODEL_MISMATCH = "model_mismatch"
    EXPIRY_VIOLATION = "expiry_violation"
    ASSIGNMENT = "assignment"
    LEDGER_CORRUPT = "ledger_corrupt"
    ORDER_RATE = "order_rate"
    CLOCK_SKEW = "clock_skew"
    STALE_QUOTES = "stale_quotes"
    JEV_ERRORS = "jev_errors"
    BROKER_ERRORS = "broker_errors"


class TriggerAction(StrEnum):
    KILL = "kill"
    HALT = "halt"
    HALT_THEN_KILL = "halt_then_kill"


class LedgerKind(StrEnum):
    RUN_START = "run_start"
    SESSION_START = "session_start"
    DECISION = "decision"
    FORECAST = "forecast"
    OUTCOME = "outcome"
    RISK_VERDICT = "risk_verdict"
    ORDER_INTENT = "order_intent"
    ORDER_STATUS = "order_status"
    FILL = "fill"
    BROKER_FILL = "broker_fill"
    MARK = "mark"
    FEE = "fee"
    RISK_EVENT = "risk_event"
    RECONCILE = "reconcile"
    KILL = "kill"
    REARM = "rearm"
    SESSION_END = "session_end"
    ANOMALY = "anomaly"


# Static lookup tables (module constants)

STRUCTURE_DIRECTION: Final[Mapping[StructureKind, Direction]] = MappingProxyType(
    {
        StructureKind.LONG_CALL: Direction.BULLISH,
        StructureKind.CALL_DEBIT: Direction.BULLISH,
        StructureKind.PUT_CREDIT: Direction.BULLISH,
        StructureKind.LONG_PUT: Direction.BEARISH,
        StructureKind.PUT_DEBIT: Direction.BEARISH,
        StructureKind.CALL_CREDIT: Direction.BEARISH,
        StructureKind.IRON_CONDOR: Direction.NEUTRAL,
    }
)
STRUCTURE_STANCE: Final[Mapping[StructureKind, VolStance]] = MappingProxyType(
    {
        StructureKind.LONG_CALL: VolStance.BUY,
        StructureKind.LONG_PUT: VolStance.BUY,
        StructureKind.CALL_DEBIT: VolStance.LIMIT,
        StructureKind.PUT_DEBIT: VolStance.LIMIT,
        StructureKind.CALL_CREDIT: VolStance.SELL,
        StructureKind.PUT_CREDIT: VolStance.SELL,
        StructureKind.IRON_CONDOR: VolStance.SELL,
    }
)
# the deterministic direction x vol-stance table (7.3)
MAPPING: Final[Mapping[tuple[Direction, VolStance], StructureKind | None]] = MappingProxyType(
    {
        (Direction.BULLISH, VolStance.SELL): StructureKind.PUT_CREDIT,
        (Direction.BULLISH, VolStance.BUY): StructureKind.LONG_CALL,
        (Direction.BULLISH, VolStance.LIMIT): StructureKind.CALL_DEBIT,
        (Direction.BEARISH, VolStance.SELL): StructureKind.CALL_CREDIT,
        (Direction.BEARISH, VolStance.BUY): StructureKind.LONG_PUT,
        (Direction.BEARISH, VolStance.LIMIT): StructureKind.PUT_DEBIT,
        (Direction.NEUTRAL, VolStance.SELL): StructureKind.IRON_CONDOR,
        (Direction.NEUTRAL, VolStance.BUY): None,
        (Direction.NEUTRAL, VolStance.LIMIT): None,
    }
)
SHORT_PREMIUM: Final[frozenset[StructureKind]] = frozenset({StructureKind.CALL_CREDIT, StructureKind.PUT_CREDIT, StructureKind.IRON_CONDOR})

# ======================================================================================================================
# 2.2 Contracts, quotes, chains
# ======================================================================================================================

_OCC_ROOT_RE: Final = re.compile(r"[A-Z]{1,6}")
_STRIKE_MILLI_LIMIT: Final = 10**8


class OptionContract(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True, order=True):
    underlying: str  # "SPY" (OCC root; adjusted roots containing digits are rejected in v1)
    expiry: date  # the LISTED OCC date: identity only, may be a Saturday (pre-2015 monthlies) or follow a holiday. NEVER used for time
    # arithmetic - that is always calendar.prev_or_same_session(expiry), carried as `last_session` (Conventions, INV-11)
    right: Right
    strike_milli: int  # strike * 1000, e.g. 450.5 -> 450500; 0 < x < 10**8

    def __post_init__(self) -> None:
        # identity invariants the OCC symbol depends on (a 9-digit strike or a digit in the root would corrupt the symbol)
        if _OCC_ROOT_RE.fullmatch(self.underlying) is None:
            raise ValueError(f"invalid OCC root {self.underlying!r}: 1-6 capital letters (adjusted roots are rejected in v1)")
        if not 0 < self.strike_milli < _STRIKE_MILLI_LIMIT:
            raise ValueError(f"strike_milli out of range (0 < x < 10**8): {self.strike_milli}")

    @property
    def occ(self) -> str:
        # occ.format_occ(self): f"{root}{yy}{mm}{dd}{C|P}{strike_milli:08d}", root unpadded (Alpaca + mirror form), e.g. "SPY261016C00600000"
        return f"{self.underlying}{self.expiry:%y%m%d}{self.right.value}{self.strike_milli:08d}"


class SnapshotKey(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True, order=True):
    session: date
    slot: Slot


class Quote(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    contract: OptionContract
    bid: Cents  # cents/share; 0 = no bid
    ask: Cents  # cents/share; 0 = no ask
    bid_size: int | None
    ask_size: int | None
    oi_prev: int | None  # open interest knowable at this snapshot: PREVIOUS session's value (mirror) / as reported with its date (live)
    iv: float | None  # OUR Black-76 IV from mid on the parity forward (annualised, calendar/365); None if not solvable
    delta: float | None  # OUR signed Black-76 delta (forward delta * discount); None if iv is None
    vega: float | None
    quote_ts: datetime | None  # feed timestamp when available (live); None for EOD data

    @property
    def mid2(self) -> int:
        # bid + ask (twice the mid, keeps integers)
        return self.bid + self.ask

    def valid(self) -> bool:
        # bid > 0 and ask > bid : a TWO-SIDED quote (IV, forwards, smile fits, entry liquidity use only these)
        return self.bid > 0 and self.ask > self.bid

    def usable_buy(self) -> bool:
        # ask > 0 and ask > bid : we can price BUYING this leg (a zero bid is fine)
        return self.ask > 0 and self.ask > self.bid

    def usable_sell_close(self) -> bool:
        # ask > 0 and ask > bid : we can price SELLING this leg TO CLOSE; bid == 0 is a legitimate price of 0 (10.4, 10.5)
        return self.ask > 0 and self.ask > self.bid


CHAIN_COLUMNS: Final[list[str]] = [
    "occ",
    "expiry",
    "last_session",
    "right",
    "strike_milli",
    "dte",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "oi_prev",
    "iv",
    "delta",
    "vega",
    "fwd",
    "iv_vendor",
    "quote_ts",
]
# dtypes: occ str; expiry, last_session datetime64[ns] (dates; last_session = calendar.prev_or_same_session(expiry));
# dte int64 = (last_session - session).days; right "C"/"P"; strike_milli,bid,ask int64; sizes/oi_prev Int64 (nullable);
# iv,delta,vega,iv_vendor float64 (nullable); fwd int64 (parity forward of the row's expiry, cents); quote_ts datetime64[ns, UTC] (nullable).
# There is deliberately NO same-day volume column (0.1 item 12). iv_vendor is kept for QC only and is never read by features.


def _opt_int(value: Any) -> int | None:
    return None if pd.isna(value) else int(value)


def _opt_float(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


@dataclass(frozen=True, eq=False)  # plain dataclass: holds a DataFrame
class ChainSnapshot:
    underlying: str
    key: SnapshotKey
    ts: datetime  # snapshot time (EOD data: that session's calendar close)
    knowable_at: datetime  # EOD_QUOTES / SYNTHETIC: ts; recorded / live: local received_at
    spot: Cents  # decision-time reference price under the run's price measure (5.2)
    spot_measure: str  # "parity" | "file_close" | "live_mid" | "synthetic"
    div_unmodelled: bool  # True when the parity spot could not include a verified dividend PV (5.2)
    rate: float  # 13-week bill rate used for forwards / discounting (decimal, last knowable)
    table: pd.DataFrame  # CHAIN_COLUMNS, sorted (expiry, right, strike_milli), filtered to 1 <= dte <= data.max_dte and |ln(K/fwd)| <= 0.35
    fidelity: Fidelity
    source: str  # "mirror" | "synthetic" | "alpaca_recorded" | "alpaca_live"
    content_hash: str  # sha256 of the table's canonical CSV bytes

    def _rows(self, expiry: date) -> pd.DataFrame:
        return self.table[self.table["expiry"] == pd.Timestamp(expiry)]

    def _single(self, expiry: date, column: str) -> Any:
        rows = self._rows(expiry)
        if rows.empty:
            raise DataUnavailable(f"{self.underlying} {self.key.session} {self.key.slot.value}: expiry {expiry} is not listed")
        values = rows[column].unique()
        if len(values) != 1:
            raise DataError(f"{self.underlying} {self.key.session}: expiry {expiry} has {len(values)} distinct `{column}` values")
        return values[0]

    def quote(self, c: OptionContract) -> Quote | None:
        if c.underlying != self.underlying:
            return None
        t = self.table
        rows = t[(t["strike_milli"] == c.strike_milli) & (t["right"] == c.right.value) & (t["expiry"] == pd.Timestamp(c.expiry))]
        if rows.empty:
            return None
        if len(rows) != 1:
            raise DataError(f"{self.underlying} {self.key.session}: {len(rows)} chain rows for contract {c.occ}")
        row = rows.iloc[0]
        quote_ts: datetime | None = None
        if not pd.isna(row["quote_ts"]):
            stamp = pd.Timestamp(row["quote_ts"])
            if stamp.tzinfo is None:
                raise DataError(f"{self.underlying} {self.key.session}: tz-naive quote_ts for {c.occ}")
            quote_ts = stamp.tz_convert("UTC").to_pydatetime(warn=False)
        return Quote(
            contract=c,
            bid=int(row["bid"]),
            ask=int(row["ask"]),
            bid_size=_opt_int(row["bid_size"]),
            ask_size=_opt_int(row["ask_size"]),
            oi_prev=_opt_int(row["oi_prev"]),
            iv=_opt_float(row["iv"]),
            delta=_opt_float(row["delta"]),
            vega=_opt_float(row["vega"]),
            quote_ts=quote_ts,
        )

    def expiries(self) -> tuple[date, ...]:
        # listed dates, ordered by (last_session, expiry)
        pairs = self.table[["last_session", "expiry"]].drop_duplicates().sort_values(["last_session", "expiry"])
        return tuple(pd.Timestamp(e).date() for e in pairs["expiry"])

    def last_session(self, expiry: date) -> date:
        # the row's last_session column (DataUnavailable if the expiry is not listed)
        session: date = pd.Timestamp(self._single(expiry, "last_session")).date()
        return session

    def side(self, expiry: date, right: Right) -> pd.DataFrame:
        # one expiry, one right, strike-sorted
        rows = self._rows(expiry)
        return rows[rows["right"] == right.value].sort_values("strike_milli", kind="stable").reset_index(drop=True)

    def forward(self, expiry: date) -> Cents:
        return int(self._single(expiry, "fwd"))


class SmileFit(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # data/surface.fit_smile output; consumed by implied_prob_above and const-maturity / ATM IV (5.3, 6.4)
    expiry: date
    last_session: date
    tau_years: float  # year_fraction(ts, close(last_session))
    tt_sessions: float  # cal.trading_time(ts, close(last_session))
    fwd: Cents
    a: float
    b: float
    c: float  # total variance w(k) = a + b*k + c*k^2, k = ln(K / fwd)
    k_lo: float
    k_hi: float  # fitted range
    n_points: int

    def w(self, k: float) -> float:
        return self.a + self.b * k + self.c * k * k

    def dw_dk(self, k: float) -> float:
        return self.b + 2.0 * self.c * k


# The chain is a DataFrame (vectorised features, fast); `Quote` objects are materialised only for the handful of legs we price.

# ======================================================================================================================
# 2.3 Legs, structures, candidates
# ======================================================================================================================


def _milli_to_cents(milli: int) -> Cents:
    # strike distance in milli-dollars -> cents/share, rounded UP: a width is never understated (it bounds max loss, 9.2)
    return -(-milli // 10)


class Leg(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    contract: OptionContract
    side: Side  # orientation when OPENING: BUY = long leg, SELL = short leg
    ratio: int = 1  # always 1 in v1 (mleg ratios must have GCD 1)


class Structure(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    kind: StructureKind
    underlying: str
    expiry: date  # single LISTED expiry (no calendars in v1); identity only
    last_session: (
        date  # calendar.prev_or_same_session(expiry), copied from the chain row by the CandidateGenerator. ALL time arithmetic on a
    )
    # structure / position (dte, sessions_to_expiry, exits, INV-11) uses this field; it is ledgered with the ORDER_INTENT
    legs: tuple[Leg, ...]  # canonical order: puts before calls, then ascending strike

    @property
    def structure_id(self) -> str:
        # sha256(canon.dumps_sorted([kind, [leg.contract.occ + ":" + leg.side for legs]]))[:16]
        payload = [self.kind.value, [f"{leg.contract.occ}:{leg.side.value}" for leg in self.legs]]
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    @property
    def width(self) -> Cents:
        # cents/share; max wing width; 0 for single legs
        return max(self.wing_widths)

    @property
    def wing_widths(self) -> tuple[Cents, Cents]:
        # (put wing, call wing); 0 where absent
        def wing(right: Right) -> Cents:
            strikes = [leg.contract.strike_milli for leg in self.legs if leg.contract.right is right]
            return _milli_to_cents(max(strikes) - min(strikes)) if len(strikes) >= 2 else 0

        return (wing(Right.PUT), wing(Right.CALL))

    @property
    def direction(self) -> Direction:
        return STRUCTURE_DIRECTION[self.kind]

    @property
    def short_legs(self) -> tuple[Leg, ...]:
        return tuple(leg for leg in self.legs if leg.side is Side.SELL)


class BandPrices(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    orats: int
    worst: int
    mid: int

    def get(self, band: Band) -> int:
        if band is Band.ORATS:
            return self.orats
        if band is Band.WORST:
            return self.worst
        if band is Band.MID:
            return self.mid
        raise ValueError(f"unknown band {band!r}")


class Candidate(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # a PRICED structure. Early failures that have no legs or prices are a CandidateReject instead (below)
    structure: Structure
    key: SnapshotKey
    dte: int  # calendar days to structure.last_session
    sessions_to_expiry: int  # calendar.sessions_between(session, structure.last_session)
    quotes: tuple[Quote, ...]  # aligned with structure.legs, from the decision snapshot
    net: BandPrices  # signed cents/share to OPEN (+debit / -credit)
    budget_floor: Cents  # the per-contract max-loss budget the long leg was fitted to (section 8); audit only
    max_loss_per_contract: Cents  # cents, at the WORST band, fees for entry + estimated exit included (9.2)
    max_profit_per_contract: Cents | None  # cents at the headline band; None = unbounded (long call; long put reported as None)
    bp_required_per_contract: Cents  # 9.2
    breakevens: tuple[Cents, ...]  # underlying price levels
    short_distance_em: float | None  # nearest short strike's distance from spot, in expected moves to expiry; None without a short leg
    net_delta: float  # per contract, shares-equivalent / 100
    net_vega: float
    rejects: tuple[str, ...] = ()  # liquidity / pricing reject codes of a PRICED structure (vocab.CANDIDATE_REJECTS); empty = tradable


class CandidateReject(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # CandidateGenerator.build() result when no structure could be built or priced at all
    underlying: str
    kind: StructureKind
    key: SnapshotKey
    rejects: tuple[str, ...]  # non-empty; e.g. ("no_expiry_in_window",), ("delta_target_unreachable:short",), ("exceeds_risk_budget",)


# `build()` returns `Candidate | CandidateReject`. The cycle never constructs an `OrderIntent` for a `CandidateReject`, for a `Candidate` with
# non-empty `rejects`, or when sizing returns 0 (`OrderIntent.qty >= 1` holds by construction); it ledgers a no-intent RISK_VERDICT instead (10.1).

# ======================================================================================================================
# 2.4 Orders, fills, positions, portfolio
# ======================================================================================================================


class OrderLeg(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    contract: OptionContract
    side: Side
    position_intent: PositionIntent
    ratio: int = 1


# EntryContext = everything the manage state needs from the ENTRY decision. It is carried on the OPEN OrderIntent, so the ORDER_INTENT ledger entry
# holds it and Book.replay can rebuild the Position from the ledger alone (resume = replay, 10.8).
class EntryContext(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    entry_thesis: str  # = EntryFacts.thesis: written by code at entry from bucket codes only (5.7)
    entry_codes: dict[
        str, str
    ]  # bucket codes at entry, keys "trend", "iv_vs_realized", "iv_rank" (= EntryFacts.trend_code / iv_rv_code / iv_rank_code)
    entry_spot: Cents  # = EntryFacts.spot (ref at the decision snapshot)
    entry_iv30_bp: Bp  # = EntryFacts.iv30_bp
    entry_em_hold_tenths: int  # = EntryFacts.em_hold_tenths (the holding-window expected-move integer shown to Jev)
    open_mid_at_decision: (
        int  # = Candidate.net.mid: signed mid net at the DECISION snapshot (path-independent P&L bucket of the manage state)
    )


class OrderIntent(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # produced by the cycle / kill switch; NOT submittable
    intent_id: str  # ids.intent_id(...): client_order_id without the attempt suffix (2.10)
    decision_id: str  # ALWAYS set: code-only exits and kill closes have deterministic decision ids too (2.10)
    position_id: str  # ids.position_id(...); for OPEN it is the id the position will get
    purpose: OrderPurpose
    part: int  # 0 = whole structure (one mleg or the single leg); 1..4 = per-leg fallback order (kill only)
    underlying: str
    legs: tuple[
        OrderLeg, ...
    ]  # OPEN: BUY->BTO, SELL->STO.  CLOSE/KILL: sides flipped, long leg -> (SELL, STC), short leg -> (BUY, BTC). () for the equity flatten
    qty: int  # whole contracts >= 1 (kill closes use the BROKER's actual leg quantity); 0 ONLY for the equity flatten (shares are in equity_qty)
    limit_start: int  # signed cents/share: first rung of the ladder (11.6)
    limit_natural: int  # signed cents/share: the natural (worst-band) price at intent time
    reason: str  # "entry" or an ExitReason value
    mandatory: bool  # True for MANDATORY_EXITS: fill model may not reject; may go past natural by the pad
    session: date
    key: SnapshotKey  # snapshot the intent was priced on
    tier_ppm: int = 0  # OPEN only: sizing tier used (0 | 500000 | 750000 | 1000000)
    structure: Structure | None = None  # None only for per-leg fallback and equity flatten orders
    entry_ctx: EntryContext | None = None  # REQUIRED for purpose OPEN, None otherwise; Book.apply(FILL) copies it onto the new Position
    equity_symbol: str | None = (
        None  # the three equity_* fields are set ONLY for the assigned-stock flatten order inside the kill sequence (K4):
    )
    equity_side: Side | None = None  #   SELL to flatten a long share position, BUY to cover a short one
    equity_qty: int | None = None  #   whole shares, <= abs(the broker's share position)


class ApprovedOrder(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # the ONLY type Broker.submit accepts (INV-03); constructed only in risk.py
    intent: OrderIntent
    verdict_id: str
    client_order_id: str  # ids.client_order_id(intent, attempt)
    attempt: int
    qty: int  # <= intent.qty (RiskEngine may only reduce)
    limit: int | None  # signed cents/share, tick-rounded adversely; None = market (kill last resort, market hours only)
    approved_at: datetime


class OrderState(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    client_order_id: str
    broker_order_id: str | None
    status: OrderStatus
    qty: int
    filled_qty: int  # cumulative
    filled_net: int | None  # signed cents/share as reported by the broker (plumbing only). Single-leg: sign derived from side.
    reject_code: int | None  # broker numeric code, e.g. 40310000
    message: str | None  # diagnostics only; never branched on except coarse DIAGNOSTIC tags (critique corr. 4)
    updated_at: datetime


class LegFill(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    occ: str
    side: Side
    bid: Cents
    ask: Cents
    orats: Cents
    worst: Cents
    mid: Cents  # all cents/share, per leg, as traded (buy or sell)


class Fill(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    fill_id: str  # ids.fill_id(client_order_id, cum_qty) = sha256(f"{cid}|{cum_qty}")[:24]; UNIQUE in the run store
    client_order_id: str
    intent_id: str
    decision_id: str
    position_id: str
    purpose: OrderPurpose
    structure_id: str | None
    qty: int  # this delta, not cumulative
    key: SnapshotKey  # snapshot whose quotes priced the fill
    ts: datetime
    net: BandPrices  # signed cents/share
    legs: tuple[LegFill, ...]
    fees_micro: Micros
    forced: bool  # mandatory exit priced with the forced-fill penalty model (10.4)
    model_reject: tuple[str, ...]  # paper only: reject codes our fill model WOULD have raised; the fill is still booked (worst band)
    quality: str  # "ok" | "degraded" (priced on an invalid / missing quote)
    source: str  # "sim" | "paper"
    broker_order_id: str | None
    broker_net: int | None  # Alpaca's reported price (plumbing only; never in any P&L)


class Position(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    position_id: str
    structure: Structure
    qty: int
    open_key: SnapshotKey  # DECISION snapshot of the entry
    open_decision_id: str
    open_net: BandPrices  # signed cents/share (actual fill, per band)
    max_loss: Cents  # total, worst band, fixed at the actual entry fill, fees included
    max_profit: Cents | None  # total, headline band
    bp_reserved: Cents  # total
    entry: EntryContext  # copied from the OPEN intent's entry_ctx by Book.apply(FILL) - every field is therefore ledger-sourced
    exit_latch: bool = False  # hysteresis latch (7.7)
    watch_text: int = 0  # consecutive sessions with an unconfirmed adverse-text reading: ALERT counter only, never closes (7.7, INV-16)
    liq_value: int = 0  # last conservative liquidation value, signed cents/share (what closing would COST; negative = receive)
    mid_value: int = 0  # last mid-to-mid value, same sign convention
    stale_marks: int = 0


class PortfolioState(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    key: SnapshotKey
    cash: BandPrices  # cents
    equity: BandPrices  # cash - sum(liq_value * 100 * qty)
    positions: tuple[Position, ...]  # sorted by position_id
    working: tuple[OrderIntent, ...]  # intents whose latest status is not terminal
    peak_equity: Cents  # headline band; reset only by REARM{reset_peak:true}
    day_start_equity: Cents  # headline-band equity of the PREVIOUS session's SESSION_END entry (initial cash on the first session);
    # carried by Book.replay. NOT "the first mark of today": that mark is the one the halt is evaluated on (9.5)
    opened_today: int
    fees_accrued_micro: Micros
    halt_entries: bool
    halt_reasons: tuple[str, ...]
    kill_state: KillState
    kill_event_id: str | None
    cooldowns: tuple[tuple[str, str, date], ...]  # (underlying, direction, first session entries are allowed again)
    jev_fail_sessions: int  # consecutive sessions with a transient decider failure
    stale_sessions: int  # consecutive sessions with a stale decision snapshot WHILE at least one position was open (resets otherwise)
    broker_fail_streak: int  # consecutive failed broker calls on the exit path
    orders_last_minute: int
    broker_equity: Cents | None = None  # paper only
    broker_prev_equity: Cents | None = (
        None  # paper only: the broker's prior-session closing equity (AccountSnapshot.last_equity); daily-loss halt (9.5)
    )
    broker_options_bp: Cents | None = None


class AccountSnapshot(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    equity: Cents
    cash: Cents
    options_buying_power: Cents
    last_equity: (
        Cents | None
    )  # equity as of the previous session's close as reported by the broker (Alpaca `last_equity`; name verified in 11.1)
    options_level: int
    trading_blocked: bool
    account_blocked: bool
    suspended: bool  # suspend_trade flag
    ts: datetime


class BrokerPosition(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    symbol: str  # OCC symbol, or an equity ticker after an assignment
    qty: int  # signed: + long / - short (contracts, or shares for equity)
    is_option: bool


class BrokerActivity(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    activity_type: str  # "OPASN" | "OPEXC" | "OPEXP" | "OPTRD" | other
    symbol: str
    qty: int
    day: date
    raw_id: str


# ======================================================================================================================
# 2.5 Decision types
# ======================================================================================================================


class NoulAns(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True, tag="noul"):
    p: float  # P(yes), clipped to [0,1]; NaN -> DeciderResponseError


class ChoiceAns(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True, tag="choice"):
    probs: dict[str, float]  # renormalised to sum 1, in the question's AUTHORED option order
    top: str  # argmax; ties broken by authored option order
    p_top: float
    margin: float  # p1 - p2
    entropy: float  # -sum p ln p / ln K, in [0,1]
    raw_sum: float  # pre-normalisation sum; outside [0.98, 1.02] => the question is treated as UNCERTAIN / gate failed
    server_choice: str  # logged only
    server_confidence: float  # logged only (B1.4 rule: never gated on)


class ScoreAns(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True, tag="score"):
    probs: tuple[float, ...]  # renormalised, index = level
    mean: float  # sum i * p_i  (ours, from renormalised probs)
    norm: float  # mean / (K-1)
    top: int
    p_top: float
    margin: float
    entropy: float
    raw_sum: float
    server_score: float
    server_confidence: float


Answer = NoulAns | ChoiceAns | ScoreAns


class DecisionRequest(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    kind: RequestKind
    variant: Variant
    question_set_id: str  # "entry.v1" | "entry_text.v1" | "manage.v1" | "manage_text.v1" | "probe.recall.v1"
    state: dict[str, Any]  # passed ensure_state_safe(); insertion-ordered; exactly what goes on the wire
    questions: dict[str, dict[str, Any]]  # FULL batch, raw-dict form, insertion-ordered (D8)
    state_hash: str  # sha256(canon.dumps_ordered(state))
    question_set_hash: str  # sha256(canon.dumps_ordered(list(questions.values())))   (ids excluded: not sent to the model)
    # provenance only - never serialised into the wire request:
    namespace: str
    subject: (
        str  # the literal "entry" (ENTRY and ENTRY_TEXT) or the position_id (MANAGE and MANAGE_TEXT) - exactly ids.decision_id's subject
    )
    underlying: str
    session: date
    key: SnapshotKey
    decision_id: str  # ids.decision_id(namespace, session, underlying, "entry" | "manage", subject) - the DECISION-level kind, NOT
    # self.kind: an entry request and its entry_text sibling (and all their variants) carry the SAME decision_id.
    # No state hash inside (crash-stable).


class DecisionResult(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    decision_id: str
    kind: RequestKind
    variant: Variant
    state_hash: str
    question_set_hash: str
    model: str  # versioned id that answered
    answers: dict[str, Answer]  # by question id; complete (every question answered) or the decider raised
    cache_keys: dict[str, str]  # by question id
    source: str  # "live" | "cache" | "mock" | "baseline"
    request_id: str | None  # sidecar only (never hashed)
    input_tokens: int | None  # sidecar only
    latency_ms: int | None  # sidecar only


class CachedAnswer(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    key: str
    namespace: str
    requested_model: str
    response_model: str
    question_set_id: str  # e.g. "entry.v1"; fills question_sets.question_set_id (13.3)
    question_set_hash: str
    state_hash: str
    question_hash: str
    question_id: str
    request_kind: str
    variant: str
    answer_json: str  # canon.dumps_sorted(raw_http_response.json()["answers"][qid])  - canonical re-encoding of the parsed sub-object
    request_id: str | None
    input_tokens: int | None  # tokens of the whole request (repeated on each row)
    latency_ms: int | None
    sdk_version: str
    created_at: datetime


class QuestionMeta(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # registry metadata; never on the wire
    qid: str
    roles: tuple[
        QuestionRole, ...
    ]  # one or more, primary first: e.g. (GATE, COMPOSITE) for under.direction, (VETO, RANK) for text.clearly_negative
    info_class: InfoClass
    outcome: str | None  # key into questions.OUTCOME_SPECS when EVAL is in roles, else None


# ======================================================================================================================
# 2.6 Rules / risk outputs
# ======================================================================================================================


class EntryFacts(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # code-side facts the rules cross-check against (never from Jev)
    trend_code: str  # vocab TREND_DIR code: up | down | flat | mixed
    iv_rank_code: str  # PCTL5 code
    iv_rv_code: str  # IV_RV code
    dist_code: str  # DIST_ATR code
    news_enabled: bool  # resolved news flag AND archive coverage for this session
    news_count: int  # items in the text state, both lists (0 => text vetoes skipped, counted as anomaly if non-clear)
    news_recent_count: int  # items in `news.since_previous_session` (0 => text.pending_binary skipped, 7.4)
    # the five fields below are the raw material of EntryContext (2.4); they are ledgered in the DECISION payload and never sent to Jev as such
    spot: Cents  # ref at the decision snapshot
    iv30_bp: Bp  # iv30 at the decision snapshot
    em_hold_tenths: int  # the holding-window expected-move integer shown to Jev
    events_in_window: int  # tracked scheduled events inside the holding window
    thesis: str  # the code-written entry thesis of 5.7 (bucket codes + events phrase only)


class EntryDecision(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    underlying: str
    decision_id: str
    action: str  # "enter" | "no_trade"
    kind: StructureKind | None
    score_core_ppm: Ppm  # composite S_core (text-free); floor and tier use THIS
    score_rank_ppm: Ppm  # S_rank (adds the text rank term); used ONLY to order underlyings
    tier_ppm: Ppm  # 0 | 500000 | 750000 | 1000000
    reasons: tuple[str, ...]  # vocab.REASONS codes, in evaluation order. A PURE function of answers + facts (path-independent):
    # whatever happens AFTER the decision (kill / halt gate, candidate, risk, deadline) is NOT here - see RISK_VERDICT (7.9)
    features_ppm: dict[str, Ppm]  # the x_i that fed S (weight sweeps without inference)
    variant_agreement: dict[str, bool]  # variant name -> agreed; {} when not run


class ManageFacts(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    pnl_headline: Cents  # conservative mark, headline band (hard exits and defaults)
    pnl_frac_loss_ppm: Ppm  # loss as a fraction of max loss (0 when in profit)
    move_code: str  # MOVE_SINCE_ENTRY code
    short_dist_code: str | None  # SHORT_DIST code
    news_count: int  # items in `news_since_entry`, both lists
    news_recent_count: int  # items in `news_since_entry.since_previous_session` (0 => pos.pending_binary_since_entry ignored, 7.7)


class ManageDecision(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    position_id: str
    decision_id: str  # always set (2.10)
    action: str  # "hold" | "close"
    reason: (
        str  # ExitReason value or "hold". Decider-down close => "code_default"; discretionary-zone fallback close => "jev_discretionary"
    )
    source: str  # "hard_exit" | "jev" | "code_default"
    pressure_ppm: Ppm | None  # exit pressure X (7.7)
    exit_latch: bool  # new latch value
    watch_text: int  # new ALERT counter value (never a reason to close)
    reasons: tuple[str, ...]


class RiskCheck(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    code: str  # vocab.RISK_CODES
    passed: bool
    observed: int | None
    limit: int | None
    detail: str = ""


class RiskVerdict(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    verdict_id: str  # sha256(canon([intent_id or decision_id, attempt, limit, reject_codes, portfolio digest, risk_config_hash]))[:24]
    decision_id: str
    intent_id: str | None  # None = a NO-INTENT verdict: the entry died after the DECISION but before an OrderIntent existed
    # (reject_codes "gate:kill_active" | "gate:halt_entries" | "gate:deadline_missed" | "candidate:<code>" | "risk:size_zero"); built by
    # cycle.no_intent_verdict(), approved = False, qty_approved = 0, checks = ()
    approved: bool
    qty_approved: int  # never above the requested qty
    checks: tuple[RiskCheck, ...]  # every check that ran, in the fixed order of 9.1
    reject_codes: tuple[str, ...]
    max_loss: Cents  # total for qty_approved, at the limit of THIS attempt
    bp_required: Cents


class CycleGate(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    allow_entries: bool
    allow_manage_jev: bool  # False => code-only management
    kill: bool
    reasons: tuple[str, ...]


class HealthSnapshot(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    clock_skew_ms: int | None  # None in backtest
    chain_age_s: dict[str, int]  # per underlying: as_of - snapshot received_at
    two_sided_frac_ppm: dict[str, Ppm]  # per underlying: share of near-the-money quotes that are two-sided
    stale_quote_frac_ppm: dict[str, Ppm]  # share of needed quotes older than health.max_quote_age_s (live only)
    reconcile_ok: bool
    model_ok: bool
    ledger_ok: bool
    spend_blocked: bool
    expiry_violation: bool  # a broker/ledger leg expires today
    assignment_seen: bool  # OPASN/OPEXC activity or an equity position


class ClockReading(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    broker_ts: datetime
    local_ts: datetime
    rtt_ms: int
    skew_ms: int
    is_open: bool
    next_open: datetime
    next_close: datetime


# ======================================================================================================================
# 2.7 Ledger, provenance, forecasts, events, news
# ======================================================================================================================


class LedgerEntry(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    seq: int  # 1..n, gapless
    kind: LedgerKind
    session: date
    as_of: datetime  # simulated time in backtest, broker clock in paper
    payload: dict[str, Any]  # builtins only: int/str/bool/None/list/dict. NO floats (probabilities as ppm, prices as cents)
    prev_hash: str  # "0"*64 for seq 1 (canon.GENESIS_HASH)
    # sha256(prev_hash + "\n" + canon.dumps_sorted({"seq","kind","session","as_of","payload"})) = canon.ledger_entry_hash(...), THE
    # single-sourced formula (3.7): kind by VALUE, session as canon.render_session (ISO date), as_of as canon.render_as_of (RFC 3339 UTC
    # "Z", six sub-second digits only when non-zero), payload after the canonical round trip. Every Ledger appends and verifies through it.
    hash: str


# run_id, trial_id, wall-clock time, request ids, token counts and latency are NEVER part of hashed material (INV-24);
# they live in the run store's `meta` and `sidecar` tables.


class ProvenanceInput(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    field: str  # state path, e.g. "vol_surface.iv_rank_1y"
    source: str  # e.g. "mirror.chain", "derived.daily", "cboe.VIX"
    event_time: datetime | None
    knowable_at: datetime
    payload_sha256: str


class Provenance(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # SIDECAR (unhashed table keyed by ledger seq); NEVER sent to Jev (D27)
    decision_id: str
    as_of: datetime
    real_symbol: str
    state_sha256: str
    evidence_tier: EvidenceTier
    data_fidelity: Fidelity
    inputs: tuple[ProvenanceInput, ...]
    news_ids: tuple[str, ...]
    news_dropped: int
    news_hostile_dropped: int
    mask_version: str
    iv_hist_proxy_pct: int  # share of the trailing IV-history window that is proxy-filled (5.4)
    spot_measure: str
    raw_features: dict[str, str]  # unbucketed feature values rendered as strings (audit only)
    request_id: str | None
    input_tokens: int | None
    latency_ms: int | None
    ledgered_wall: datetime


class BuiltState(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # StateBuilder output (section 5); shared by state.py, cycle.py, jev/probe.py, paper/runner.py
    # In-memory only: it encodes (to_builtins / json.encode) but msgspec cannot DECODE it, because `facts` is a union of two untagged
    # Structs (tagging them would add a key to the hashed DECISION `facts` payload). Its parts are persisted separately (2.11, 13.4).
    state: dict[str, Any]
    state_hash: str
    provenance: Provenance
    facts: EntryFacts | ManageFacts  # code-side facts for the rules cross-checks (never sent to Jev)


class OutcomeSpec(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # frozen at forecast time (I5): integer thresholds, calendar-computed resolve session
    kind: str  # "close_gt" | "close_lt" | "close_inside" | "rv_gt_iv"
    horizon_sessions: int
    resolve_on: date
    ref: Cents
    lo: Cents | None = None  # close_lt: y = close < lo ; close_inside: lo <= close <= hi
    hi: Cents | None = None  # close_gt: y = close > hi
    iv_var_ppm: int | None = None  # rv_gt_iv: implied TOTAL variance to the resolve close, * 1e6


class OutcomeTemplate(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # questions.OUTCOME_SPECS values (6.4): how outcomes.build_forecasts instantiates an OutcomeSpec for an eval question
    kind: str  # OutcomeSpec.kind
    horizon: str  # "1" | "5" | "hold"  (hold = dte.hold_horizon_sessions)
    band: str | None  # None | "lo" | "hi" | "inside" | "lo_half" | "hi_half" | "inside_half": which expected-move threshold(s) apply
    implied: str | None  # "PA(ref)" | "1-PA(lo)" | "PA(hi)" | "PA(lo)-PA(hi)" | None: the p_implied recipe of the 6.4 table


class Forecast(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    forecast_id: str  # sha256(f"{decision_id}|{question_id}|{int(with_text)}")[:24] - with_text is IN the id: the 12 eval questions ride
    # in both entry batches under ONE decision_id, so without it the two forecast sets would collide
    event_key: str  # ids.event_key(underlying, key, spec): shared by every decider and reference asked about the same event
    decision_id: str
    question_id: str
    question_hash: str
    with_text: bool  # True for forecasts from entry_text requests
    underlying: str
    key: SnapshotKey
    p_ppm: (
        Ppm | None
    )  # decider's P(yes), base variant. None = MISSING: the request's state was built but the request failed / was suppressed
    missing_reason: str | None = None  # set iff p_ppm is None: the DeciderError class name, "state_rejected" or "requests_suppressed"
    p_abstain_ppm: Ppm | None = (
        None  # only on the derived `under.direction#*` forecasts: the mass on conflicting_signals before renormalisation (6.4)
    )
    p_implied_ppm: Ppm | None  # option-implied comparison probability, frozen now (6.4)
    implied_method: str | None  # "smile_digital" | "nd2_plain" | None
    implied_quality: str | None  # "interpolated" | "extrapolated"
    p_implied_spread_ppm: Ppm | None  # call-spread cross-check when an expiry lands on the resolve session
    spec: OutcomeSpec
    tier: EvidenceTier
    fidelity: Fidelity
    iv_history: str  # "own" | "proxy_mixed"
    prereg: bool  # tagged PREREGISTERED (12.1)


class Outcome(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    event_key: str
    resolved_on: date
    y: int | None  # 1 | 0 | None = void (data gap) - voids are reported, never dropped silently
    observed: dict[str, int]  # e.g. {"close": 45310} or {"rv_var_ppm": 412}
    div_in_window: str  # "yes" | "no" | "unknown"
    price_measure: str  # measure used for the close (must equal the forecast's spot_measure family, 5.2)


class ScheduledEvent(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    kind: str  # "fomc_decision" | "cpi" | "nfp" | "ex_dividend"
    event_date: date  # fomc_decision: the LAST calendar day of the listed meeting range = the statement / decision day (5.1)
    underlying: str | None  # only for ex_dividend
    amount_cents: Cents | None  # only for ex_dividend
    scheduled: bool  # False = unscheduled / emergency (stored for audit, NEVER served as an upcoming event)
    cancelled: bool = False  # True = a scheduled meeting the source marks cancelled (stored and listed by `data verify`, NEVER served)
    knowable_at: datetime
    knowable_rule: str  # "fetched_at" | "scheduled_minus_45d_assumption" | "exdiv_minus_14d_assumption"
    source_url: str
    fetched_at: datetime


class NewsItem(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    id: str
    created_at: datetime
    updated_at: datetime
    received_at: datetime | None  # local clock at fetch (forward recordings only)
    knowable_at: datetime  # received_at if present else created_at + news.lag_s
    headline: str
    summary: str | None
    source: str
    symbols: tuple[str, ...]


class MaskTerms(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # the parsed config/mask_terms.toml (5.8); a StateBuilder constructor argument that cycle / runner / probes must build
    # sha256(RULES_VERSION + file bytes)[:12] = mask_version; RULES_VERSION IS config.MASK_RULES_VERSION - the one constant, bumped
    # whenever a 5.8 sanitiser / masking rule changes (textmask.py defines no second one)
    version: str
    # group name ("funds", "indices", "central_banks", "agencies", "releases", "people", "companies", "geo_events") -> terms;
    # always all eight keys of config.MASK_GROUPS (an omitted group is ())
    groups: dict[str, tuple[str, ...]]
    replacements: dict[str, str]  # group name -> replacement phrase; always all eight (5.8 defaults unless the file overrides one)


# def load_mask_terms(path: Path) -> MaskTerms     # config.py (WP00): msgspec.toml decode + validation; ConfigError on unknown groups / empty terms

# ======================================================================================================================
# 2.8 Run metadata
# ======================================================================================================================


class RunMeta(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # stored in the run store `meta` table; NOT hashed into the ledger
    run_id: str  # backtest: f"{utc:%Y%m%dT%H%M%S}-{config_hash[:8]}"; paper: "paper-<experiment>";
    # shadow stores (11.8): "shadow-<experiment>", "shadow-<experiment>-codeonly", "shadow-<experiment>-same"
    trial_id: int | None
    experiment: str
    family: str  # trial family for DSR (12.6): run.family, default = run.experiment; derived suffixes "#baseline", "#shadow",
    # "#reference", "#diagnostic", "#ablation" are appended by the code that registers such runs
    namespace: str  # ids.namespace(experiment, model, refresh_generation)
    mode: RunMode
    decider: str  # "live_jev" | "replay_jev" | "mock_jev" | "baseline:<...>"
    model: str
    model_release_date: date
    fidelity: Fidelity
    fill_rule: FillRule
    spot_measure: str
    news_resolved: bool  # news.enabled resolved ONCE at run start; recorded; part of the resolved-config hash
    news_reason: str  # "explicit_on" | "explicit_off" | "auto_keys_present" | "auto_no_keys" | "text_probe_pending"  (section 4)
    config_hash: str
    state_config_hash: str
    rules_hash: str
    risk_config_hash: str
    entry_qset_hash: str
    entry_text_qset_hash: str
    manage_qset_hash: str
    manage_text_qset_hash: str
    git_commit: str | None
    data_manifest_hash: str
    cache_manifest_hash: str | None  # filled at report time
    start: date
    end: date | None
    purpose: str  # "validate" | "tune" | "diagnostic" | "final" | "reference" | "paper" | "shadow"
    flags: tuple[str, ...]  # e.g. ("placebo",), ("unmasked",), ("news_off",), ("baseline:4:seed=17",), ("shadow",), ("reference_history",),
    # ("ablation:buckets_only",), ("model_overlap",), ("diagnostic",)


class ProbeRecord(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # one Step 0 suite result (6.8): written by jev/probe.py, read by config.probe_status, registry.sync_step0, doctor
    suite: str  # "meta" | "determinism" | "batch" | "order" | "text"
    model: str
    sdk_version: str
    entry_qset_hash: str
    entry_text_qset_hash: str  # THE key: a record never satisfies another model / wording / SDK
    verdict: dict[str, Any]  # suite-specific summary (e.g. {"deterministic": true, "max_noul_std_ppm": 9000})
    run_dir: str
    recorded_at: datetime


# ======================================================================================================================
# 2.11 Hashed ledger payload fields per `LedgerKind` (frozen by WP00; all ints/strs/bools/None/lists/dicts)
# ======================================================================================================================
# The TOP-LEVEL keys of each kind's hashed payload. "<Struct> builtins" = msgspec.to_builtins(struct): its field names.
# Descriptive cells of the 2.11 table are frozen here as literal keys: "equity per band" / "cash per band" -> `equity` / `cash`
# (BandPrices builtins), "per-position {...}" -> `positions` (position_id -> {liq_value, mid_value, stale}), "positions digest" ->
# `positions_digest`, "qset hashes (4)" -> the four RunMeta names, "candidate summary" -> `candidate`, "realised P&L per band" ->
# `realised_pnl` (BandPrices builtins, cents).

DECISION_REQUEST_FIELDS: Final[tuple[str, ...]] = (
    "request_kind",
    "variant",
    "state_hash",
    "question_set_id",
    "question_set_hash",
    "question_hashes",  # {qid: hash}
    "cache_keys",  # {qid: key}
    "model",
    "source",
    "answers",  # {qid: ppm ints}
    "error",  # class name or null
)
CANDIDATE_SUMMARY_FIELDS: Final[tuple[str, ...]] = (
    "structure_id",
    "legs",  # [occ, side]
    "net",  # per band
    "max_loss_per_contract",
    "budget_floor",
    "rejects",
)
MARK_POSITION_FIELDS: Final[tuple[str, ...]] = ("liq_value", "mid_value", "stale")

LEDGER_PAYLOAD_FIELDS: Final[Mapping[LedgerKind, tuple[str, ...]]] = MappingProxyType(
    {
        LedgerKind.RUN_START: (
            "mode",
            "namespace",
            "model",
            "model_release_date",
            "decider",
            "fidelity",
            "fill_rule",
            "spot_measure",
            "news_resolved",
            "news_reason",
            "config_hash",
            "state_config_hash",
            "rules_hash",
            "risk_config_hash",
            "entry_qset_hash",
            "entry_text_qset_hash",
            "manage_qset_hash",
            "manage_text_qset_hash",
            "data_manifest_hash",  # computed UP FRONT from the selected partitions, 13.1
            "git_commit",
            "purpose",
            "flags",
            "initial_cash",
        ),
        LedgerKind.SESSION_START: ("session", "slot", "phase"),
        # Entry DECISIONs are a pure function of answers + facts; what happened afterwards is in RISK_VERDICT
        LedgerKind.DECISION: (
            "decision_id",
            "kind",  # DecisionKind: "entry" / "manage"
            "subject_alias",
            "text",  # "on" / "off" / "no_archive"
            "requests",  # [DECISION_REQUEST_FIELDS]
            "rules",  # EntryDecision or ManageDecision builtins
            "facts",  # EntryFacts / ManageFacts builtins
            "tier",
        ),
        # `p_ppm = null` + `missing_reason` for a MISSING forecast; spec and `p_implied` are still computed from the view
        LedgerKind.FORECAST: Forecast.__struct_fields__,
        LedgerKind.OUTCOME: Outcome.__struct_fields__,
        # `intent_id = null` marks a no-intent verdict (2.6). The post-DECISION funnel lives here: `reject_codes` holds
        # `gate:*`, `candidate:<code>`, `risk:<code>`. `candidate` = CANDIDATE_SUMMARY_FIELDS, null when no structure was built
        LedgerKind.RISK_VERDICT: (*RiskVerdict.__struct_fields__, "candidate"),
        # for OPEN this includes `entry_ctx`, the ledger source of `Position.entry`
        LedgerKind.ORDER_INTENT: OrderIntent.__struct_fields__,
        LedgerKind.ORDER_STATUS: (
            "client_order_id",
            "intent_id",
            "attempt",
            "status",
            "qty",
            "filled_qty",
            "limit",
            "broker_order_id",
            "reject_code",
            "tag",  # coarse diagnostic tag
        ),
        LedgerKind.FILL: Fill.__struct_fields__,
        LedgerKind.BROKER_FILL: ("client_order_id", "cum_qty", "broker_net", "ts"),  # plumbing evidence only
        LedgerKind.MARK: (
            "equity",
            "cash",
            "open_max_loss",
            "bp_used",
            "bp_utilisation_ppm",
            "positions",  # {position_id: MARK_POSITION_FIELDS}
            "net_delta_milli",
            "net_vega_milli",
        ),
        LedgerKind.FEE: ("fee_cents", "accrued_micro_before"),  # fee_cents charged to all three bands
        LedgerKind.RISK_EVENT: ("type", "trigger", "detail", "counters"),
        LedgerKind.RECONCILE: ("ok", "order_actions", "diff", "foreign_orders", "activities"),  # diff{symbol: [ledger, broker]}
        LedgerKind.KILL: ("event_id", "step", "trigger", "detail"),
        LedgerKind.REARM: ("event_id", "reset_peak", "operator_note", "ledger_head_at_rearm"),
        # the headline `equity` value is the NEXT session's `day_start_equity` (9.5)
        LedgerKind.SESSION_END: (
            "equity",
            "positions_digest",
            "n_decisions",
            "n_forecasts",
            "n_intents",
            "invariant_no_expiry_risk",
        ),
        LedgerKind.ANOMALY: ("type", "detail"),
    }
)

# Keys that are present only in some entries of a kind.
LEDGER_PAYLOAD_OPTIONAL_FIELDS: Final[Mapping[LedgerKind, tuple[str, ...]]] = MappingProxyType(
    {
        LedgerKind.SESSION_START: ("clock_skew_ms",),  # paper adds `clock_skew_ms`
        LedgerKind.FILL: ("realised_pnl",),  # + realised P&L per band on closing fills
        # paper adds broker_equity, broker_prev_equity (the broker's prior-session closing equity), broker_options_bp
        # (the Book's only source for them)
        LedgerKind.MARK: ("broker_equity", "broker_prev_equity", "broker_options_bp"),
    }
)
