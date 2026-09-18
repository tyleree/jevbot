"""`DefaultRiskEngine`: the last step before any broker call (DESIGN.md section 9, INV-03, INV-21, D19).

Pure: it never calls a broker, never reads a clock (`approve(now=...)` is handed the caller's time) and never imports
`jevbot.jev`. It runs after the decider and after the rules, for every attempt of every order, in every mode. It can approve,
reduce a quantity or reject; it can never increase a quantity, loosen a price or change legs. `approve()` is the ONLY
constructor of `ApprovedOrder` (`tests/guards/test_approved_order_site.py`).

What lives here:

* `approve()` - the 21 ordered checks of 9.1. EVERY check is recorded in `RiskVerdict.checks`, in the fixed order of the table
  and one `RiskCheck` per code; a check that does not apply to the order's purpose is recorded as `passed = True` with an
  `n/a: ...` detail (the "applies to" column is what keeps closes and kill orders from ever being blocked by entry-side
  conditions, INV-21). Any failure => not approved.
* `size_entry()` / `budget_floor()` - the 9.3 sizing arithmetic.
* `recheck_fill()` - the delayed D+1 fill of `SimBroker`: checks 10 / 11 re-evaluated AS OF the fill snapshot plus checks
  15-17 at the actual fill price, at 1.0x.
* `hard_exit()` - the six code-only exits of 9.4, first match wins, on conservative headline-band marks.
* `pre_cycle()` / `on_mark()` / `daily_loss()` - the trigger evaluation of 9.5 with the proportionate actions of deviation V2.

Two seams the 3.4 signatures do not carry:

* `daily_loss(pf)`. The daily-loss halt of 9.5 is explicitly **not** a kill trigger and `KillTrigger` has no member for it, so it
  cannot travel through `on_mark()`'s `(KillTrigger, TriggerAction, str)` tuples. `cycle.py` calls this method at step 2 and
  appends `RISK_EVENT{daily_loss_halt}` (which `Book.apply` turns into `halt_entries`).
* `set_kill_holdings(mapping)`. Check 6 measures a KILL order against **the broker's** held quantity (9.1), and a pure engine
  may not call a broker. The kill-switch flatten driver, which has just fetched `broker.positions()`, hands that mapping in
  before it approves the round's closers and clears it afterwards. Without it, check 6 falls back to the book's own legs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from fractions import Fraction
from typing import TYPE_CHECKING, Final

import msgspec

from jevbot import canon, ids, money, structmath
from jevbot.config import Config, risk_config_hash
from jevbot.errors import DataError, InvariantError
from jevbot.structmath import MULTIPLIER
from jevbot.types import (
    SHORT_PREMIUM,
    ApprovedOrder,
    Band,
    BandPrices,
    Candidate,
    Cents,
    ChainSnapshot,
    ClockReading,
    CycleGate,
    ExitReason,
    FillRule,
    HealthSnapshot,
    KillState,
    KillTrigger,
    Leg,
    OrderIntent,
    OrderLeg,
    OrderPurpose,
    PortfolioState,
    Position,
    PositionIntent,
    Ppm,
    Quote,
    Right,
    RiskCheck,
    RiskVerdict,
    RunMeta,
    RunMode,
    ScheduledEvent,
    Side,
    Slot,
    Structure,
    StructureKind,
    TriggerAction,
)
from jevbot.vocab import RISK_CHECKS, SIZE_ZERO

if TYPE_CHECKING:
    from jevbot.protocols import MarketView

__all__ = ["DefaultRiskEngine", "headline_band"]

_PPM: Final = 1_000_000
_CLOSE_INTENTS: Final[frozenset[PositionIntent]] = frozenset({PositionIntent.BTC, PositionIntent.STC})
_FOMC: Final = "fomc_decision"
_EX_DIVIDEND: Final = "ex_dividend"
_MILLI_PER_CENT: Final = 10

# 9.1 check 4: the exact leg template of each kind, as a sorted multiset of (side, right)
_TEMPLATES: Final[Mapping[StructureKind, tuple[tuple[Side, Right], ...]]] = {
    StructureKind.LONG_CALL: ((Side.BUY, Right.CALL),),
    StructureKind.LONG_PUT: ((Side.BUY, Right.PUT),),
    StructureKind.CALL_DEBIT: ((Side.BUY, Right.CALL), (Side.SELL, Right.CALL)),
    StructureKind.PUT_DEBIT: ((Side.BUY, Right.PUT), (Side.SELL, Right.PUT)),
    StructureKind.CALL_CREDIT: ((Side.BUY, Right.CALL), (Side.SELL, Right.CALL)),
    StructureKind.PUT_CREDIT: ((Side.BUY, Right.PUT), (Side.SELL, Right.PUT)),
    StructureKind.IRON_CONDOR: (
        (Side.BUY, Right.CALL),
        (Side.BUY, Right.PUT),
        (Side.SELL, Right.CALL),
        (Side.SELL, Right.PUT),
    ),
}

# codes, by check number (1-based), taken from the frozen vocabulary so the two can never drift
_C = RISK_CHECKS


def headline_band(cfg: Config) -> Band:
    """ "Headline band" of the Conventions: `orats` under `next_snapshot`, `worst` under `same_snapshot_worst`."""
    return Band.WORST if cfg.cadence.fill_rule is FillRule.SAME_SNAPSHOT_WORST else Band.ORATS


def _frac(value: float, name: str) -> Fraction:
    """A config number as the exact decimal the operator wrote (0.01 -> 1/100), never binary noise."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvariantError(f"risk: {name} must be a number, got {type(value).__name__}")
    return Fraction(repr(float(value)))


def _floor(value: Fraction) -> int:
    return value.numerator // value.denominator


def _strike_cents(strike_milli: int) -> Cents:
    return strike_milli // _MILLI_PER_CENT


def _intrinsic(right: Right, strike_milli: int, spot: Cents) -> Cents:
    strike = _strike_cents(strike_milli)
    return max(spot - strike, 0) if right is Right.CALL else max(strike - spot, 0)


def _leg_signature(legs: Sequence[Leg]) -> tuple[tuple[Side, Right], ...]:
    return tuple(sorted(((leg.side, leg.contract.right) for leg in legs), key=lambda p: (p[0].value, p[1].value)))


def _matches_template(structure: Structure) -> bool:
    """9.1 check 4: count, rights, sides, one underlying, a single expiry, ratio 1. (Strike ORDER is check 5's business.)"""
    template = _TEMPLATES.get(structure.kind)
    if template is None or len(structure.legs) != len(template):
        return False
    want = tuple(sorted(template, key=lambda p: (p[0].value, p[1].value)))
    if _leg_signature(structure.legs) != want:
        return False
    if any(isinstance(leg.ratio, bool) or leg.ratio != 1 for leg in structure.legs):
        return False
    first = structure.legs[0].contract
    return all(leg.contract.underlying == structure.underlying and leg.contract.expiry == first.expiry for leg in structure.legs)


def _worst_net(legs: Sequence[OrderLeg], quotes: Mapping[str, Quote | None]) -> int | None:
    """The worst-band natural of an order: buy at the ask, sell at the bid (10.3). None when a leg has no quote."""
    net = 0
    for leg in legs:
        quote = quotes.get(leg.contract.occ)
        if quote is None:
            return None
        if leg.side is Side.BUY:
            net += quote.ask * leg.ratio
        else:
            net -= quote.bid * leg.ratio
    return net


@dataclass
class _Eval:
    """Everything the 21 checks of one `approve()` call share."""

    intent: OrderIntent
    pf: PortfolioState
    view: MarketView
    now: datetime
    attempt: int
    limit: int | None
    cand: Candidate | None
    approved_so_far: tuple[ApprovedOrder, ...]
    clock: ClockReading | None
    market: bool
    qty: int
    chain: ChainSnapshot | None
    quotes: dict[str, Quote | None]
    checks: list[RiskCheck] = field(default_factory=list)
    rejects: list[str] = field(default_factory=list)

    def record(self, code: str, passed: bool, *, observed: int | None = None, limit: int | None = None, detail: str = "") -> None:
        self.checks.append(RiskCheck(code=code, passed=passed, observed=observed, limit=limit, detail=detail))
        if not passed:
            self.rejects.append(f"risk:{code}")

    def na(self, code: str, why: str) -> None:
        self.record(code, True, detail=f"n/a: {why}")

    @property
    def purpose(self) -> OrderPurpose:
        return self.intent.purpose

    @property
    def is_open(self) -> bool:
        return self.intent.purpose is OrderPurpose.OPEN

    @property
    def is_kill(self) -> bool:
        return self.intent.purpose is OrderPurpose.KILL

    @property
    def is_equity(self) -> bool:
        return self.intent.equity_symbol is not None

    @property
    def structure(self) -> Structure | None:
        return self.intent.structure


class DefaultRiskEngine:
    """Implements `protocols.RiskEngine` (3.4) - see the module docstring."""

    def __init__(self, cfg: Config, meta: RunMeta | None = None) -> None:
        self._cfg: Final = cfg
        self._meta: Final = meta
        self._mode: Final[RunMode] = meta.mode if meta is not None else cfg.run.mode
        self._flags: Final[frozenset[str]] = frozenset(meta.flags if meta is not None else ())
        self._risk_hash: Final[str] = meta.risk_config_hash if meta is not None else risk_config_hash(cfg)
        self._headline: Final[Band] = headline_band(cfg)
        self._kill_holdings: Mapping[str, int] | None = None

    # ------------------------------------------------------------------------------------------------------------------
    # seams
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def headline(self) -> Band:
        return self._headline

    def set_kill_holdings(self, holdings: Mapping[str, int] | None) -> None:
        """The BROKER's signed holdings (OCC symbol / equity ticker -> qty) that check 6 measures a KILL order against (9.1).

        The kill-switch flatten driver sets this from `broker.positions()` before approving a round and clears it afterwards;
        the engine itself never calls a broker. `None` falls back to the book's own legs.
        """
        self._kill_holdings = None if holdings is None else dict(holdings)

    # ------------------------------------------------------------------------------------------------------------------
    # 9.3 sizing
    # ------------------------------------------------------------------------------------------------------------------

    def equity_basis(self, pf: PortfolioState) -> Cents:
        """Headline-band book equity in a backtest; `min(broker equity, book headline equity)` in paper (9.1)."""
        book = pf.equity.get(self._headline)
        if self._mode is RunMode.PAPER and pf.broker_equity is not None and self._cfg.risk.equity_basis == "min_broker_book":
            return min(pf.broker_equity, book)
        return book

    def lowest_tier(self) -> Fraction:
        """The lowest NON-ZERO sizing tier the rules can produce (9.3); 0.5 at the shipped defaults."""
        tiers = self._cfg.rules.tiers
        values = [value for _, value in tiers.score]
        values += [value for _, value in tiers.peakedness]
        values += list(tiers.environment)
        non_zero = [_frac(v, "rules.tiers") for v in values if v > 0]
        if not non_zero:
            raise InvariantError("risk: [rules.tiers] defines no non-zero sizing tier")
        return min(non_zero)

    def budget_floor(self, pf: PortfolioState) -> Cents:
        """`floor(risk.max_loss_per_trade_pct * equity_basis * lowest non-zero tier)` - the per-contract max-loss budget the
        CandidateGenerator fits the long leg to (8, 9.3). $500 at the defaults."""
        pct = _frac(self._cfg.risk.max_loss_per_trade_pct, "risk.max_loss_per_trade_pct")
        return _floor(pct * self.equity_basis(pf) * self.lowest_tier())

    def size_entry(self, cand: Candidate, tier_ppm: Ppm, pf: PortfolioState, approved_so_far: Sequence[ApprovedOrder]) -> int:
        """`min(floor(budget / max_loss_per_contract), risk.max_contracts_per_trade)` (9.3); 0 = no trade.

        Aggregate loss and buying power are enforced by checks 16 / 17 of `approve()`, which may reduce further.
        """
        del approved_so_far  # checks 16 / 17 of approve() carry the already-approved orders
        if tier_ppm <= 0 or cand.max_loss_per_contract <= 0:
            return 0
        pct = _frac(self._cfg.risk.max_loss_per_trade_pct, "risk.max_loss_per_trade_pct")
        budget = _floor(pct * self.equity_basis(pf) * Fraction(int(tier_ppm), _PPM))
        return max(0, min(budget // cand.max_loss_per_contract, self._cfg.risk.max_contracts_per_trade))

    # ------------------------------------------------------------------------------------------------------------------
    # 9.1 approve()
    # ------------------------------------------------------------------------------------------------------------------

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
    ) -> tuple[RiskVerdict, ApprovedOrder | None]:
        if now.tzinfo is None:
            raise InvariantError("approve(now=...) must be a tz-aware UTC instant (approve never reads a clock)")
        limit_value = None if market else (intent.limit_start if limit is None else limit)
        chain, quotes = self._quotes(intent, view)
        ev = _Eval(
            intent=intent,
            pf=pf,
            view=view,
            now=now,
            attempt=attempt,
            limit=limit_value,
            cand=cand,
            approved_so_far=tuple(approved_so_far),
            clock=clock,
            market=market,
            qty=intent.qty,
            chain=chain,
            quotes=quotes,
        )
        self._check_01_kill(ev)
        self._check_02_diagnostic(ev)
        self._check_03_underlying(ev)
        self._check_04_structure(ev)
        self._check_05_defined_risk(ev)
        self._check_06_close_reduces(ev)
        self._check_07_cutoff(ev)
        self._check_08_clock_skew(ev)
        self._check_09_quotes(ev)
        self._check_10_dte(ev)
        self._check_11_events(ev)
        self._check_12_exposure(ev)
        self._check_13_max_open(ev)
        self._check_14_max_new(ev)
        max_loss_pc, bp_pc = self._check_15_to_18_sizing(ev)
        self._check_19_price(ev)
        self._check_20_drift(ev)
        self._check_21_rate(ev)

        qty = ev.qty
        if ev.is_open and qty <= 0 and f"risk:{SIZE_ZERO}" not in ev.rejects:
            ev.rejects.append(f"risk:{SIZE_ZERO}")
        if ev.is_equity:
            approved = not ev.rejects and (intent.equity_qty or 0) > 0
        else:
            approved = not ev.rejects and qty > 0
        verdict = RiskVerdict(
            verdict_id=self._verdict_id(ev),
            decision_id=intent.decision_id,
            intent_id=intent.intent_id,
            approved=approved,
            qty_approved=qty if approved else 0,
            checks=tuple(ev.checks),
            reject_codes=tuple(ev.rejects),
            max_loss=max_loss_pc * qty if approved else 0,
            bp_required=bp_pc * qty if approved else 0,
        )
        if not approved:
            return (verdict, None)
        order = ApprovedOrder(
            intent=intent,
            verdict_id=verdict.verdict_id,
            client_order_id=ids.client_order_id(intent, attempt),
            attempt=attempt,
            qty=qty,
            limit=limit_value,
            approved_at=now,
        )
        return (verdict, order)

    # --- check 1 -------------------------------------------------------------------------------------------------------

    def _check_01_kill(self, ev: _Eval) -> None:
        kill_code, halt_code = _C[0]
        state = ev.pf.kill_state
        if ev.is_open:
            ev.record(kill_code, state is KillState.ARMED, detail=f"kill_state={state.value}")
            ev.record(halt_code, not ev.pf.halt_entries, detail=",".join(ev.pf.halt_reasons))
            return
        if ev.is_kill:
            ev.record(kill_code, state is not KillState.LOCKED, detail=f"kill_state={state.value}")
            ev.na(halt_code, "kill orders are never blocked by an entry halt (INV-21)")
            return
        ev.na(kill_code, "a close is allowed in every kill state (INV-21)")
        ev.na(halt_code, "a close is allowed under an entry halt (INV-21)")

    # --- check 2 -------------------------------------------------------------------------------------------------------

    def _check_02_diagnostic(self, ev: _Eval) -> None:
        (code,) = _C[1]
        bad = sorted(self._flags & {"diagnostic", "unmasked"})
        ev.record(code, not bad, detail=",".join(bad))

    # --- check 3 -------------------------------------------------------------------------------------------------------

    def _check_03_underlying(self, ev: _Eval) -> None:
        (code,) = _C[2]
        whitelist = self._cfg.universe.underlyings
        symbol = ev.intent.equity_symbol
        if symbol is not None:
            ev.record(code, symbol in whitelist, detail=f"equity {symbol}")
            return
        roots = {leg.contract.underlying for leg in ev.intent.legs}
        ok = ev.intent.underlying in whitelist and roots <= {ev.intent.underlying}
        ev.record(code, ok, detail=f"{ev.intent.underlying} legs={sorted(roots)}")

    # --- check 4 -------------------------------------------------------------------------------------------------------

    def _check_04_structure(self, ev: _Eval) -> None:
        (code,) = _C[3]
        if not ev.is_open:
            ev.na(code, f"{ev.purpose.value} order")
            return
        structure = ev.structure
        if structure is None:
            ev.record(code, False, detail="an OPEN order carries no structure")
            return
        enabled = structure.kind in self._cfg.structures.enabled
        ev.record(code, enabled and _matches_template(structure), detail=f"{structure.kind.value} enabled={enabled}")

    # --- check 5 -------------------------------------------------------------------------------------------------------

    def _check_05_defined_risk(self, ev: _Eval) -> None:
        (code,) = _C[4]
        if not ev.is_open:
            ev.na(code, f"{ev.purpose.value} order")
            return
        structure = ev.structure
        if structure is None:
            raise InvariantError("risk check 5: an OPEN order reached the RiskEngine without a structure (2.4)")
        if not structmath.defined_risk_ok(structure.kind, structure.legs):
            raise InvariantError(
                f"risk check 5: {structure.kind.value} {structure.structure_id} is not defined risk - an uncovered or "
                "mis-ordered short reached the RiskEngine (9.1 check 5)"
            )
        max_loss = structmath.max_loss_pc(structure.kind, structure.wing_widths, self._risk_net(ev), self._fee_rt(ev))
        if max_loss <= 0:
            raise InvariantError(
                f"risk check 5: {structure.kind.value} prices to max_loss_pc={max_loss} (<= 0): invalid economics reached "
                "the RiskEngine (9.1 check 5, 9.2)"
            )
        ev.record(code, True, observed=max_loss, detail=f"max_loss_pc={max_loss}")

    # --- check 6 -------------------------------------------------------------------------------------------------------

    def _check_06_close_reduces(self, ev: _Eval) -> None:
        (code,) = _C[5]
        if ev.is_open:
            ev.na(code, "open order")
            return
        held = self._held(ev)
        if ev.is_equity:
            symbol = ev.intent.equity_symbol or ""
            shares = held.get(symbol, 0)
            want_side = Side.SELL if shares > 0 else Side.BUY
            qty = ev.intent.equity_qty or 0
            ok = shares != 0 and ev.intent.equity_side is want_side and 0 < qty <= abs(shares)
            ev.record(code, ok, observed=qty, limit=abs(shares), detail=f"{symbol} shares={shares}")
            return
        problems: list[str] = []
        for leg in ev.intent.legs:
            occ = leg.contract.occ
            position = held.get(occ, 0)
            if leg.position_intent not in _CLOSE_INTENTS:
                problems.append(f"{occ}:{leg.position_intent.value}")
                continue
            if position == 0:
                problems.append(f"{occ}:not_held")
                continue
            want = Side.SELL if position > 0 else Side.BUY
            if leg.side is not want:
                problems.append(f"{occ}:side={leg.side.value}")
            if ev.qty * leg.ratio > abs(position):
                problems.append(f"{occ}:qty={ev.qty * leg.ratio}>{abs(position)}")
        ev.record(code, not problems, observed=ev.qty, detail=",".join(problems))

    def _held(self, ev: _Eval) -> Mapping[str, int]:
        """Signed holdings check 6 measures against: the BROKER's for a KILL (9.1), the book's legs otherwise."""
        if ev.is_kill and self._kill_holdings is not None:
            return self._kill_holdings
        legs: dict[str, int] = {}
        for pos in ev.pf.positions:
            for leg in pos.structure.legs:
                signed = pos.qty * leg.ratio * (1 if leg.side is Side.BUY else -1)
                legs[leg.contract.occ] = legs.get(leg.contract.occ, 0) + signed
        return legs

    # --- check 7 -------------------------------------------------------------------------------------------------------

    def _check_07_cutoff(self, ev: _Eval) -> None:
        (code,) = _C[6]
        if ev.is_kill:
            ev.na(code, "KILL is exempt; market-closed handling is the kill switch's (9.5)")
            return
        if ev.view.key.slot is Slot.EOD:
            ev.na(code, "eod decision snapshot")
            return
        session = ev.view.key.session
        calendar = ev.view.calendar
        cutoff = calendar.offset_from_close(session, self._cfg.cadence.order_cutoff_offset_min)
        opened, closed = calendar.open_close(session)
        inside = opened <= ev.now <= closed
        ok = ev.now < cutoff and inside
        detail = f"now={ev.now.isoformat()} cutoff={cutoff.isoformat()}" + ("" if inside else " outside the session")
        ev.record(code, ok, observed=int((cutoff - ev.now).total_seconds()), limit=0, detail=detail)

    # --- check 8 -------------------------------------------------------------------------------------------------------

    def _check_08_clock_skew(self, ev: _Eval) -> None:
        (code,) = _C[7]
        if not ev.is_open:
            ev.na(code, "a close is timed by the broker clock and never blocked by skew (V3, INV-21)")
            return
        if ev.clock is None:
            ev.na(code, "no broker clock (backtest)")
            return
        cap = self._cfg.health.max_clock_skew_ms
        ev.record(code, abs(ev.clock.skew_ms) <= cap, observed=abs(ev.clock.skew_ms), limit=cap)

    # --- check 9 -------------------------------------------------------------------------------------------------------

    def _check_09_quotes(self, ev: _Eval) -> None:
        stale_code, crossed_code = _C[8]
        crossed = [
            leg.contract.occ
            for leg in ev.intent.legs
            if (quote := ev.quotes.get(leg.contract.occ)) is not None and quote.bid > 0 and quote.ask <= quote.bid
        ]
        if not ev.is_open:
            if ev.intent.mandatory or ev.is_kill:
                ev.na(stale_code, "a mandatory / KILL close is never blocked by quotes (INV-21)")
                ev.na(crossed_code, "a mandatory / KILL close is never blocked by quotes (INV-21)")
                return
            ev.na(stale_code, "only crossed quotes delay a discretionary close (9.1)")
            ev.record(crossed_code, not crossed, detail=",".join(crossed))
            return
        if ev.chain is None:
            ev.record(stale_code, False, detail="no chain snapshot for the order's underlying")
            ev.record(crossed_code, False, detail="no chain snapshot for the order's underlying")
            return
        problems: list[str] = []
        for leg in ev.intent.legs:
            quote = ev.quotes.get(leg.contract.occ)
            if quote is None:
                problems.append(f"{leg.contract.occ}:missing")
                continue
            codes = structmath.leg_liquidity_rejects(quote, sold=leg.side is Side.SELL, cfg=self._cfg.liquidity)
            problems += [f"{leg.contract.occ}:{c}" for c in codes if c != "liq:crossed"]
            if quote.quote_ts is not None:
                age = int((ev.view.as_of - quote.quote_ts).total_seconds())
                if age > self._cfg.health.max_quote_age_s:
                    problems.append(f"{leg.contract.occ}:quote_age={age}")
        chain_age = int((ev.view.as_of - ev.chain.knowable_at).total_seconds())
        if chain_age > self._cfg.health.max_chain_age_s:
            problems.append(f"chain_age={chain_age}")
        ev.record(stale_code, not problems, observed=chain_age, limit=self._cfg.health.max_chain_age_s, detail=",".join(problems))
        ev.record(crossed_code, not crossed, detail=",".join(crossed))

    # --- check 10 ------------------------------------------------------------------------------------------------------

    def _check_10_dte(self, ev: _Eval) -> None:
        window_code, policy_code = _C[9]
        if not ev.is_open:
            ev.na(window_code, f"{ev.purpose.value} order")
            ev.na(policy_code, f"{ev.purpose.value} order")
            return
        structure = ev.structure
        if structure is None:
            ev.record(window_code, False, detail="no structure")
            ev.record(policy_code, False, detail="no structure")
            return
        window_ok, window_detail, dte = self._dte_window(structure, ev.view.key.session)
        policy_ok, policy_detail, sessions = self._expiry_policy(structure, ev.view)
        cfg = self._cfg.dte
        ev.record(window_code, window_ok, observed=dte, limit=cfg.max_entry, detail=window_detail)
        ev.record(
            policy_code,
            policy_ok,
            observed=sessions,
            limit=cfg.hard_exit_sessions + cfg.min_sessions_beyond_hard_exit,
            detail=policy_detail,
        )

    def _dte_window(self, structure: Structure, session: date) -> tuple[bool, str, int]:
        dte = (structure.last_session - session).days
        cfg = self._cfg.dte
        return (cfg.min_entry <= dte <= cfg.max_entry, f"dte={dte} to last_session {structure.last_session.isoformat()}", dte)

    def _expiry_policy(self, structure: Structure, view: MarketView) -> tuple[bool, str, int]:
        sessions = view.calendar.sessions_between(view.key.session, structure.last_session)
        floor = self._cfg.dte.hard_exit_sessions + self._cfg.dte.min_sessions_beyond_hard_exit
        return (sessions > floor, f"sessions_to_expiry={sessions} (to last_session)", sessions)

    # --- check 11 ------------------------------------------------------------------------------------------------------

    def _check_11_events(self, ev: _Eval) -> None:
        blackout_code, exdiv_code = _C[10]
        if not ev.is_open:
            ev.na(blackout_code, f"{ev.purpose.value} order")
            ev.na(exdiv_code, f"{ev.purpose.value} order")
            return
        structure = ev.structure
        if structure is None:
            ev.record(blackout_code, False, detail="no structure")
            ev.record(exdiv_code, False, detail="no structure")
            return
        blackout_ok, blackout_detail = self._event_blackout(structure, ev.view)
        exdiv_ok, exdiv_detail = self._exdiv_short_call(structure, ev.view)
        ev.record(blackout_code, blackout_ok, limit=self._cfg.risk.event_blackout_sessions, detail=blackout_detail)
        ev.record(exdiv_code, exdiv_ok, detail=exdiv_detail)

    def _event_blackout(self, structure: Structure, view: MarketView) -> tuple[bool, str]:
        if structure.kind not in SHORT_PREMIUM:
            return (True, "not short premium")
        sessions = self._cfg.risk.event_blackout_sessions
        session = view.key.session
        horizon = view.calendar.next_session(session, sessions) if sessions > 0 else session
        hits = [e.event_date.isoformat() for e in _events(view, session, horizon, None) if e.kind == _FOMC]
        return (not hits, f"fomc within {sessions} sessions: {','.join(hits)}" if hits else f"no fomc by {horizon.isoformat()}")

    def _exdiv_short_call(self, structure: Structure, view: MarketView) -> tuple[bool, str]:
        short_calls = [leg for leg in structure.legs if leg.side is Side.SELL and leg.contract.right is Right.CALL]
        if not short_calls or not self._cfg.exits.ex_dividend_guard:
            return (True, "no short call" if not short_calls else "ex-dividend guard off")
        session = view.key.session
        events = [e for e in _events(view, session, structure.last_session, structure.underlying) if e.kind == _EX_DIVIDEND]
        if not events:
            return (True, "no verified ex-date in [session, last_session]")
        spot = _spot(view, structure.underlying)
        if spot is None:
            return (True, "no spot to measure the short call against")
        hits: list[str] = []
        for event in events:
            threshold = spot + (event.amount_cents or 0)
            hits += [
                f"{leg.contract.occ}@{event.event_date.isoformat()}"
                for leg in short_calls
                if _strike_cents(leg.contract.strike_milli) < threshold
            ]
        return (not hits, ",".join(hits) if hits else "every short call is above spot + dividend")

    # --- check 12 ------------------------------------------------------------------------------------------------------

    def _check_12_exposure(self, ev: _Eval) -> None:
        dup_code, cooldown_code, cap_code = _C[11]
        if not ev.is_open:
            ev.na(dup_code, f"{ev.purpose.value} order")
            ev.na(cooldown_code, f"{ev.purpose.value} order")
            ev.na(cap_code, f"{ev.purpose.value} order")
            return
        structure = ev.structure
        if structure is None:
            ev.record(dup_code, False, detail="no structure")
            ev.record(cooldown_code, False, detail="no structure")
            ev.record(cap_code, False, detail="no structure")
            return
        direction = structure.direction
        underlying = structure.underlying
        pending = self._pending_opens(ev)
        same_pair = [s for s in pending if s.underlying == underlying and s.direction is direction]
        same_pair += [p.structure for p in ev.pf.positions if p.structure.underlying == underlying and p.structure.direction is direction]
        if self._cfg.risk.one_per_underlying_direction:
            ev.record(dup_code, not same_pair, observed=len(same_pair), limit=0, detail=f"{underlying} {direction.value}")
        else:
            ev.na(dup_code, "risk.one_per_underlying_direction is off")
        until = next(
            (when for u, d, when in ev.pf.cooldowns if u == underlying and d == direction.value),
            None,
        )
        active = until is not None and ev.view.key.session < until
        ev.record(cooldown_code, not active, detail="" if until is None else f"entries allowed again on {until.isoformat()}")
        same_direction = sum(1 for s in pending if s.direction is direction)
        same_direction += sum(1 for p in ev.pf.positions if p.structure.direction is direction)
        cap = self._cfg.risk.max_same_direction_structures
        ev.record(cap_code, same_direction < cap, observed=same_direction, limit=cap, detail=direction.value)

    def _pending_opens(self, ev: _Eval) -> list[Structure]:
        """Structures of the working OPEN intents and of the entries already approved this cycle."""
        out = [i.structure for i in ev.pf.working if i.purpose is OrderPurpose.OPEN and i.structure is not None]
        out += [o.intent.structure for o in ev.approved_so_far if o.intent.purpose is OrderPurpose.OPEN and o.intent.structure is not None]
        return out

    # --- checks 13 / 14 ------------------------------------------------------------------------------------------------

    def _check_13_max_open(self, ev: _Eval) -> None:
        (code,) = _C[12]
        if not ev.is_open:
            ev.na(code, f"{ev.purpose.value} order")
            return
        count = len(ev.pf.positions) + len(self._pending_opens(ev))
        cap = self._cfg.risk.max_open_structures
        ev.record(code, count < cap, observed=count, limit=cap)

    def _check_14_max_new(self, ev: _Eval) -> None:
        (code,) = _C[13]
        if not ev.is_open:
            ev.na(code, f"{ev.purpose.value} order")
            return
        count = ev.pf.opened_today + len(self._pending_opens(ev))
        cap = self._cfg.risk.max_new_per_day
        ev.record(code, count < cap, observed=count, limit=cap)

    # --- checks 15-18 --------------------------------------------------------------------------------------------------

    def _check_15_to_18_sizing(self, ev: _Eval) -> tuple[Cents, Cents]:
        """The four quantity checks. Returns (max_loss_per_contract, bp_required_per_contract) at THIS attempt's limit."""
        codes = (_C[14][0], _C[15][0], _C[16][0], *_C[17])
        structure = ev.structure
        if not ev.is_open or structure is None:
            for code in codes:
                ev.na(code, f"{ev.purpose.value} order")
            return (0, 0)
        widths = structure.wing_widths
        net = self._risk_net(ev)
        fee_rt = self._fee_rt(ev)
        max_loss_pc = structmath.max_loss_pc(structure.kind, widths, net, fee_rt)
        bp_pc = structmath.bp_required_pc(structure.kind, widths, net, fee_rt, self._cfg.risk)
        basis = self.equity_basis(ev.pf)

        # 15 - per trade, at 1.0x, at the limit of THIS attempt
        cap = _floor(_frac(self._cfg.risk.max_loss_per_trade_pct, "risk.max_loss_per_trade_pct") * basis)
        ev.qty = min(ev.qty, cap // max_loss_pc if max_loss_pc > 0 else 0)
        ev.record(codes[0], ev.qty > 0, observed=ev.qty * max_loss_pc, limit=cap, detail=f"max_loss_pc={max_loss_pc}")

        # 16 - aggregate open loss
        held = sum(p.max_loss for p in ev.pf.positions) + self._pending_max_loss(ev)
        agg_cap = _floor(_frac(self._cfg.risk.max_aggregate_open_loss_pct, "risk.max_aggregate_open_loss_pct") * basis)
        room = agg_cap - held
        ev.qty = min(ev.qty, max(room // max_loss_pc, 0) if max_loss_pc > 0 else 0)
        ev.record(codes[1], ev.qty > 0, observed=held + ev.qty * max_loss_pc, limit=agg_cap, detail=f"open+working={held}")

        # 17 - buying power (9.3)
        reserved = sum(p.bp_reserved for p in ev.pf.positions) + self._pending_bp(ev)
        internal = _floor(_frac(self._cfg.risk.max_bp_utilisation, "risk.max_bp_utilisation") * basis) - reserved
        available = internal if ev.pf.broker_options_bp is None else min(internal, ev.pf.broker_options_bp)
        ev.qty = min(ev.qty, max(available // bp_pc, 0) if bp_pc > 0 else ev.qty)
        ev.record(codes[2], ev.qty > 0, observed=reserved + ev.qty * bp_pc, limit=reserved + max(available, 0), detail=f"bp_pc={bp_pc}")

        # 18 - hard caps
        qty_cap = self._cfg.risk.max_contracts_per_trade
        ev.qty = min(ev.qty, qty_cap)
        ev.record(codes[3], ev.qty > 0, observed=ev.qty, limit=qty_cap)
        notional_cap = self._cfg.risk.max_order_notional_usd * MULTIPLIER
        if ev.limit is None:
            ev.na(codes[4], "market order")
        else:
            per_contract = abs(ev.limit) * MULTIPLIER
            ev.qty = min(ev.qty, notional_cap // per_contract if per_contract > 0 else ev.qty)
            ev.record(codes[4], ev.qty > 0, observed=ev.qty * per_contract, limit=notional_cap)
        return (max_loss_pc, bp_pc)

    def _pending_max_loss(self, ev: _Eval) -> Cents:
        total = 0
        for intent in ev.pf.working:
            if intent.purpose is OrderPurpose.OPEN and intent.structure is not None:
                total += structmath.max_loss_pc(intent.structure.kind, intent.structure.wing_widths, intent.limit_natural, 0) * intent.qty
        for order in ev.approved_so_far:
            structure = order.intent.structure
            if order.intent.purpose is OrderPurpose.OPEN and structure is not None:
                net = order.limit if order.limit is not None else order.intent.limit_natural
                total += structmath.max_loss_pc(structure.kind, structure.wing_widths, net, 0) * order.qty
        return total

    def _pending_bp(self, ev: _Eval) -> Cents:
        total = 0
        risk = self._cfg.risk
        for intent in ev.pf.working:
            if intent.purpose is OrderPurpose.OPEN and intent.structure is not None:
                total += (
                    structmath.bp_required_pc(intent.structure.kind, intent.structure.wing_widths, intent.limit_natural, 0, risk)
                    * intent.qty
                )
        for order in ev.approved_so_far:
            structure = order.intent.structure
            if order.intent.purpose is OrderPurpose.OPEN and structure is not None:
                net = order.limit if order.limit is not None else order.intent.limit_natural
                total += structmath.bp_required_pc(structure.kind, structure.wing_widths, net, 0, risk) * order.qty
        return total

    # --- check 19 ------------------------------------------------------------------------------------------------------

    def _check_19_price(self, ev: _Eval) -> None:
        increment_code, sign_code, natural_code = _C[18]
        structure = ev.structure
        kind = structure.kind if structure is not None else None
        width = structure.width if structure is not None else 0
        pad = self._cushion(ev, width)
        if ev.limit is None:
            ev.na(increment_code, "market order")
            ev.na(sign_code, "market order")
            ev.na(natural_code, "market order")
            money.assert_limit_sign(ev.purpose, kind, None, width=width, pad=pad)
            return
        tick = self._tick(ev)
        ev.record(increment_code, ev.limit % tick == 0, observed=ev.limit, limit=tick, detail=f"tick={tick}")
        money.assert_limit_sign(ev.purpose, kind, ev.limit, width=width, pad=pad)
        ev.record(sign_code, True, observed=ev.limit, detail=f"width={width} pad={pad}")
        allowance = pad if (ev.is_kill or ev.intent.mandatory) else 0
        beyond = ev.limit - ev.intent.limit_natural
        ev.record(
            natural_code,
            beyond <= allowance,
            observed=beyond,
            limit=allowance,
            detail=f"limit={ev.limit} natural={ev.intent.limit_natural}",
        )

    def _cushion(self, ev: _Eval, width: Cents) -> Cents:
        """`pad` of `money.assert_limit_sign`: the largest cushion a mandatory / KILL close may ever carry (9.1 check 19)."""
        if not (ev.is_kill or ev.intent.mandatory):
            return 0
        kill = self._cfg.kill
        if width > 0:
            return _ceil_frac(_frac(kill.cushion_max_frac_width, "kill.cushion_max_frac_width") * width)
        # a single leg (or a per-leg kill fallback) has no width: the ladder's own cap bounds the cushion instead
        return kill.cushion_ticks * kill.flatten_attempts * self._tick(ev)

    def _tick(self, ev: _Eval) -> int:
        penny = self._cfg.universe.penny_all
        prices = [quote.mid2 // 2 for quote in ev.quotes.values() if quote is not None]
        if not prices:
            prices = [abs(ev.limit or ev.intent.limit_natural)]
        return min(money.tick_cents(ev.intent.underlying, price, penny) for price in prices)

    # --- check 20 ------------------------------------------------------------------------------------------------------

    def _check_20_drift(self, ev: _Eval) -> None:
        (code,) = _C[19]
        if not ev.is_open:
            ev.na(code, f"{ev.purpose.value} order")
            return
        natural_now = _worst_net(ev.intent.legs, ev.quotes)
        if natural_now is None:
            ev.record(code, False, detail="a leg has no quote on the current view")
            return
        drift = natural_now - ev.intent.limit_natural
        allowance = _ceil_frac(_frac(self._cfg.risk.max_adverse_drift, "risk.max_adverse_drift") * abs(ev.intent.limit_natural))
        ev.record(code, drift <= allowance, observed=drift, limit=allowance, detail=f"natural now={natural_now}")

    # --- check 21 ------------------------------------------------------------------------------------------------------

    def _check_21_rate(self, ev: _Eval) -> None:
        rate_code, attempt_code = _C[20]
        if ev.is_kill or ev.intent.mandatory:
            ev.na(rate_code, "mandatory closes and KILL orders are exempt (the kill path must not throttle itself)")
            ev.na(attempt_code, "mandatory closes and KILL orders are exempt")
            return
        cap = self._cfg.risk.max_orders_per_minute
        ev.record(rate_code, ev.pf.orders_last_minute < cap, observed=ev.pf.orders_last_minute, limit=cap)
        attempts = self._cfg.risk.max_order_attempts
        ev.record(attempt_code, ev.attempt < attempts, observed=ev.attempt, limit=attempts)

    # --- shared helpers ------------------------------------------------------------------------------------------------

    def _quotes(self, intent: OrderIntent, view: MarketView) -> tuple[ChainSnapshot | None, dict[str, Quote | None]]:
        """The order's leg quotes on the CURRENT view. A missing snapshot is not fatal: the kill switch may run outside a
        cycle with the last available view, and checks that need quotes are skipped for KILL (3.4)."""
        if not intent.legs:
            return (None, {})
        try:
            chain = view.chain(intent.underlying)
        except DataError:
            return (None, {leg.contract.occ: None for leg in intent.legs})
        return (chain, {leg.contract.occ: chain.quote(leg.contract) for leg in intent.legs})

    def _risk_net(self, ev: _Eval) -> int:
        """The signed net the sizing formulas price this attempt at: the limit of THIS attempt (9.1 check 15)."""
        if ev.limit is not None:
            return ev.limit
        if ev.cand is not None:
            return ev.cand.net.worst
        return ev.intent.limit_natural

    def _fee_rt(self, ev: _Eval) -> Cents:
        """`fee_rt` (entry + estimated exit fees per contract, 9.2). Recovered exactly from the candidate when there is one -
        `max_loss_per_contract - max_loss_pc(..., fee_rt = 0)` - and from the current leg mids otherwise."""
        structure = ev.structure
        if structure is None:
            return 0
        if ev.cand is not None:
            base = structmath.max_loss_pc(structure.kind, structure.wing_widths, ev.cand.net.worst, 0)
            return max(ev.cand.max_loss_per_contract - base, 0)
        return _fee_round_trip_from_quotes(structure, ev.quotes, self._cfg)

    def _verdict_id(self, ev: _Eval) -> str:
        material = [
            ev.intent.intent_id or ev.intent.decision_id,
            ev.attempt,
            ev.limit,
            list(ev.rejects),
            canon.sha256_hex(msgspec.json.encode(ev.pf)),
            self._risk_hash,
        ]
        return canon.sha256_hex(canon.dumps_sorted(material))[:24]

    # ------------------------------------------------------------------------------------------------------------------
    # 9.3 recheck_fill
    # ------------------------------------------------------------------------------------------------------------------

    def recheck_fill(self, intent: OrderIntent, net: BandPrices, pf: PortfolioState, view: MarketView) -> tuple[int, str | None]:
        """The delayed D+1 fill (`SimBroker` only): `(qty, reject code)` at the FILL snapshot `view` (9.3).

        (a) checks 10 and 11 are re-evaluated as of the fill snapshot - an entry decided two sessions before an FOMC decision
        passes the blackout at D but would fill one session before it, inside the blackout; (b) `max_loss_pc` is recomputed at
        the ACTUAL worst-band fill price and the largest `qty <= intent.qty` that still satisfies checks 15-17 at 1.0x is
        returned. `qty == 0` cancels the order with `risk:recheck_failed:<code>`.
        """
        structure = intent.structure
        if intent.purpose is not OrderPurpose.OPEN or structure is None:
            return (intent.qty, None)
        window_ok, _, _ = self._dte_window(structure, view.key.session)
        if not window_ok:
            return (0, _C[9][0])
        policy_ok, _, _ = self._expiry_policy(structure, view)
        if not policy_ok:
            return (0, _C[9][1])
        blackout_ok, _ = self._event_blackout(structure, view)
        if not blackout_ok:
            return (0, _C[10][0])
        exdiv_ok, _ = self._exdiv_short_call(structure, view)
        if not exdiv_ok:
            return (0, _C[10][1])

        _, quotes = self._quotes(intent, view)
        fee_rt = _fee_round_trip_from_quotes(structure, quotes, self._cfg)
        widths = structure.wing_widths
        max_loss_pc = structmath.max_loss_pc(structure.kind, widths, net.worst, fee_rt)
        bp_pc = structmath.bp_required_pc(structure.kind, widths, net.worst, fee_rt, self._cfg.risk)
        basis = self.equity_basis(pf)
        if max_loss_pc <= 0:
            return (0, _C[14][0])
        qty = min(intent.qty, _floor(_frac(self._cfg.risk.max_loss_per_trade_pct, "risk.max_loss_per_trade_pct") * basis) // max_loss_pc)
        if qty <= 0:
            return (0, _C[14][0])
        held = sum(p.max_loss for p in pf.positions)
        room = _floor(_frac(self._cfg.risk.max_aggregate_open_loss_pct, "risk.max_aggregate_open_loss_pct") * basis) - held
        qty = min(qty, max(room // max_loss_pc, 0))
        if qty <= 0:
            return (0, _C[15][0])
        if bp_pc > 0:
            reserved = sum(p.bp_reserved for p in pf.positions)
            internal = _floor(_frac(self._cfg.risk.max_bp_utilisation, "risk.max_bp_utilisation") * basis) - reserved
            available = internal if pf.broker_options_bp is None else min(internal, pf.broker_options_bp)
            qty = min(qty, max(available // bp_pc, 0))
        if qty <= 0:
            return (0, _C[16][0])
        return (qty, None)

    # ------------------------------------------------------------------------------------------------------------------
    # 9.4 hard exits
    # ------------------------------------------------------------------------------------------------------------------

    def hard_exit(self, pos: Position, view: MarketView) -> ExitReason | None:
        """The six code-only exits of 9.4, first match wins, on conservative headline-band marks. Never blocked by exposure
        limits, stale quotes, clock skew, order-rate limits, a spend stop or a decider failure (INV-21)."""
        session = view.key.session
        last = pos.structure.last_session
        calendar = view.calendar
        if calendar.sessions_between(session, last) <= self._cfg.dte.hard_exit_sessions:
            return ExitReason.FORCE_EXPIRY

        spot = _spot(view, pos.structure.underlying)
        quotes = self._position_quotes(pos, view)
        if self._ex_dividend_exit(pos, view, spot, quotes):
            return ExitReason.EX_DIVIDEND
        if self._assignment_risk(pos, spot, quotes):
            return ExitReason.ASSIGNMENT_RISK

        dte = (last - session).days
        limit = self._cfg.dte.time_exit_short_premium if pos.structure.kind in SHORT_PREMIUM else self._cfg.dte.time_exit_long_premium
        if dte <= limit:
            return ExitReason.TIME_EXIT

        pnl = (-pos.open_net.get(self._headline) - pos.liq_value) * MULTIPLIER * pos.qty
        if pos.max_loss > 0 and -pnl >= _frac(self._cfg.exits.stop_loss_frac, "exits.stop_loss_frac") * pos.max_loss:
            return ExitReason.STOP_LOSS
        target_frac = _frac(self._cfg.exits.profit_target_frac, "exits.profit_target_frac")
        if pos.max_profit is not None:
            if pos.max_profit > 0 and pnl >= target_frac * pos.max_profit:
                return ExitReason.PROFIT_TARGET
        else:
            debit_paid = pos.open_net.get(self._headline) * MULTIPLIER * pos.qty
            if debit_paid > 0 and pnl >= target_frac * debit_paid:
                return ExitReason.PROFIT_TARGET
        return None

    def _position_quotes(self, pos: Position, view: MarketView) -> dict[str, Quote | None]:
        try:
            chain = view.chain(pos.structure.underlying)
        except DataError:
            return {}
        return {leg.contract.occ: chain.quote(leg.contract) for leg in pos.structure.legs}

    def _ex_dividend_exit(self, pos: Position, view: MarketView, spot: Cents | None, quotes: Mapping[str, Quote | None]) -> bool:
        if not self._cfg.exits.ex_dividend_guard or spot is None:
            return False
        session = view.key.session
        horizon = view.calendar.next_session(session, self._cfg.exits.ex_div_exit_sessions)
        events = [e for e in _events(view, session, horizon, pos.structure.underlying) if e.kind == _EX_DIVIDEND]
        if not events:
            return False
        for leg in pos.structure.legs:
            if leg.side is not Side.SELL or leg.contract.right is not Right.CALL:
                continue
            intrinsic = _intrinsic(Right.CALL, leg.contract.strike_milli, spot)
            if intrinsic <= 0:
                continue
            quote = quotes.get(leg.contract.occ)
            for event in events:
                if event.amount_cents is None:
                    return True  # any ITM short call when the amount is unknown (9.4 rule 2)
                if quote is None or quote.ask - intrinsic < event.amount_cents:
                    return True
        return False

    def _assignment_risk(self, pos: Position, spot: Cents | None, quotes: Mapping[str, Quote | None]) -> bool:
        if spot is None:
            return False
        floor = self._cfg.exits.assignment_extrinsic_floor_cents
        for leg in pos.structure.legs:
            if leg.side is not Side.SELL:
                continue
            intrinsic = _intrinsic(leg.contract.right, leg.contract.strike_milli, spot)
            if intrinsic <= 0:
                continue
            quote = quotes.get(leg.contract.occ)
            if quote is None or quote.ask <= 0:
                continue
            if quote.ask - intrinsic < floor:
                return True
        return False

    # ------------------------------------------------------------------------------------------------------------------
    # 9.5 triggers
    # ------------------------------------------------------------------------------------------------------------------

    def _action(self, trigger: KillTrigger, *, escalate: bool) -> TriggerAction:
        """The proportionate action of deviation V2: a `halt_then_kill` trigger escalates only past its persistence
        threshold; an operator who downgraded it to `halt` never escalates."""
        configured = self._cfg.kill.actions.for_trigger(trigger)
        if configured is TriggerAction.HALT_THEN_KILL:
            return TriggerAction.KILL if escalate else TriggerAction.HALT
        return configured

    def pre_cycle(self, pf: PortfolioState, health: HealthSnapshot) -> tuple[CycleGate, tuple[tuple[KillTrigger, TriggerAction, str], ...]]:
        cfg, hcfg = self._cfg, self._cfg.health
        triggers: list[tuple[KillTrigger, TriggerAction, str]] = []
        if not health.reconcile_ok:
            triggers.append((KillTrigger.RECONCILE_MISMATCH, self._action(KillTrigger.RECONCILE_MISMATCH, escalate=True), "reconcile"))
        if not health.model_ok:
            triggers.append((KillTrigger.MODEL_MISMATCH, self._action(KillTrigger.MODEL_MISMATCH, escalate=True), "model"))
        if not health.ledger_ok:
            triggers.append((KillTrigger.LEDGER_CORRUPT, self._action(KillTrigger.LEDGER_CORRUPT, escalate=True), "verify"))
        if health.expiry_violation:
            triggers.append(
                (KillTrigger.EXPIRY_VIOLATION, self._action(KillTrigger.EXPIRY_VIOLATION, escalate=True), "last_session<=today")
            )
        if health.assignment_seen:
            triggers.append((KillTrigger.ASSIGNMENT, self._action(KillTrigger.ASSIGNMENT, escalate=True), "assignment"))
        if health.clock_skew_ms is not None and abs(health.clock_skew_ms) > hcfg.max_clock_skew_ms:
            detail = f"skew_ms={health.clock_skew_ms}"
            triggers.append((KillTrigger.CLOCK_SKEW, self._action(KillTrigger.CLOCK_SKEW, escalate=False), detail))
        stale = self._stale_underlyings(health)
        if stale:
            # kill only after `stale_kill_after_sessions` consecutive stale sessions DURING WHICH a position was open:
            # with a flat book there is nothing to flatten (9.5)
            escalate = bool(pf.positions) and pf.stale_sessions + 1 >= hcfg.stale_kill_after_sessions
            triggers.append((KillTrigger.STALE_QUOTES, self._action(KillTrigger.STALE_QUOTES, escalate=escalate), ",".join(stale)))
        if pf.jev_fail_sessions > 0:
            escalate = pf.jev_fail_sessions >= hcfg.jev_error_kill_after_sessions
            triggers.append(
                (KillTrigger.JEV_ERRORS, self._action(KillTrigger.JEV_ERRORS, escalate=escalate), f"sessions={pf.jev_fail_sessions}")
            )
        if pf.broker_fail_streak >= hcfg.broker_error_halt_after:
            escalate = pf.broker_fail_streak >= hcfg.broker_error_kill_after
            triggers.append(
                (KillTrigger.BROKER_ERRORS, self._action(KillTrigger.BROKER_ERRORS, escalate=escalate), f"streak={pf.broker_fail_streak}")
            )
        if pf.orders_last_minute >= cfg.risk.max_orders_per_minute:
            triggers.append(
                (KillTrigger.ORDER_RATE, self._action(KillTrigger.ORDER_RATE, escalate=True), f"orders={pf.orders_last_minute}")
            )

        kill = any(action is TriggerAction.KILL for _, action, _ in triggers)
        halt = any(action is TriggerAction.HALT for _, action, _ in triggers)
        reasons = tuple(f"{trigger.value}:{action.value}" for trigger, action, _ in triggers)
        if health.spend_blocked:
            reasons = (*reasons, "spend_blocked")  # INV-17: a spend stop halts entries and is never a kill input
        if pf.halt_entries:
            reasons = (*reasons, *pf.halt_reasons)
        allow_entries = not kill and not halt and not health.spend_blocked and not pf.halt_entries and pf.kill_state is KillState.ARMED
        allow_manage_jev = not kill and not health.spend_blocked and health.model_ok and pf.jev_fail_sessions == 0
        gate = CycleGate(allow_entries=allow_entries, allow_manage_jev=allow_manage_jev, kill=kill, reasons=reasons)
        return (gate, tuple(triggers))

    def _stale_underlyings(self, health: HealthSnapshot) -> list[str]:
        hcfg = self._cfg.health
        two_sided_floor = _frac(hcfg.min_two_sided_frac, "health.min_two_sided_frac") * _PPM
        stale_cap = _frac(hcfg.max_stale_quote_frac, "health.max_stale_quote_frac") * _PPM
        bad: set[str] = {u for u, age in health.chain_age_s.items() if age > hcfg.max_chain_age_s}
        bad |= {u for u, share in health.two_sided_frac_ppm.items() if share < two_sided_floor}
        bad |= {u for u, share in health.stale_quote_frac_ppm.items() if share > stale_cap}
        return sorted(bad)

    def on_mark(self, pf: PortfolioState) -> tuple[tuple[KillTrigger, TriggerAction, str], ...]:
        """The drawdown kill, evaluated at the DECISION-snapshot mark (9.5) so a breach flattens the same day.

        The daily-loss halt of the same table is not a kill trigger; it is `daily_loss()` below.
        """
        if pf.peak_equity <= 0:
            return ()
        cap = _frac(self._cfg.risk.drawdown_kill_pct, "risk.drawdown_kill_pct")
        equity = pf.equity.get(self._headline)
        losses = [Fraction(pf.peak_equity - equity, pf.peak_equity)]
        if pf.broker_equity is not None:
            losses.append(Fraction(pf.peak_equity - pf.broker_equity, pf.peak_equity))
        worst = max(losses)
        if worst < cap:
            return ()
        detail = f"drawdown_ppm={_floor(worst * _PPM)} peak={pf.peak_equity} equity={equity}"
        return ((KillTrigger.DRAWDOWN, self._action(KillTrigger.DRAWDOWN, escalate=True), detail),)

    def daily_loss(self, pf: PortfolioState) -> str | None:
        """The daily-loss halt of 9.5, measured against the PREVIOUS session's SESSION_END equity (`pf.day_start_equity`),
        never against today's first mark. Paper additionally compares the broker's equity with the broker's prior-session
        closing equity and takes the WORSE of the two relative losses. Returns the RISK_EVENT detail, or None.

        Not a kill trigger (9.5), so it cannot travel through `on_mark()`'s `KillTrigger` tuples: `cycle.py` calls this at
        step 2 and appends `RISK_EVENT{daily_loss_halt}`, which `Book.apply` turns into `halt_entries`.
        """
        cap = _frac(self._cfg.risk.daily_loss_halt_pct, "risk.daily_loss_halt_pct")
        losses: list[tuple[str, Fraction]] = []
        if pf.day_start_equity > 0:
            losses.append(("book", Fraction(pf.day_start_equity - pf.equity.get(self._headline), pf.day_start_equity)))
        if pf.broker_prev_equity is not None and pf.broker_equity is not None and pf.broker_prev_equity > 0:
            losses.append(("broker", Fraction(pf.broker_prev_equity - pf.broker_equity, pf.broker_prev_equity)))
        if not losses:
            return None
        source, worst = max(losses, key=lambda pair: pair[1])
        if worst < cap:
            return None
        return f"{source} loss_ppm={_floor(worst * _PPM)} limit_ppm={_floor(cap * _PPM)}"


# ======================================================================================================================
# module helpers
# ======================================================================================================================


def _ceil_frac(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def _events(view: MarketView, start: date, end: date, underlying: str | None) -> tuple[ScheduledEvent, ...]:
    if end < start:
        return ()
    try:
        return view.events(start, end, underlying)
    except DataError:
        return ()


def _spot(view: MarketView, underlying: str) -> Cents | None:
    try:
        return view.spot(underlying)
    except DataError:
        return None


def _fee_round_trip_from_quotes(structure: Structure, quotes: Mapping[str, Quote | None], cfg: Config) -> Cents:
    """`fee_rt` from the current leg mids when no candidate carries it (the SEC term needs a price per leg, 9.2 / 10.7)."""
    prices: list[Cents] = []
    for leg in structure.legs:
        quote = quotes.get(leg.contract.occ)
        prices.append(0 if quote is None else max(quote.mid2 // 2, 0))
    sells = sum(1 for leg in structure.legs if leg.side is Side.SELL)
    return structmath.fee_round_trip(len(structure.legs), sells, prices, cfg.fees)
