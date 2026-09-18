"""Lean per-session decision logic for the minimal backtest engine (`backtest.py` drives the session loop).

Builds entry / manage states, calls the `Decider`, applies `DecisionRules`, builds candidates, sizes and approves
orders - and returns ready-to-append `(LedgerKind, payload)` pairs plus the freshly approved orders, so
`backtest.py`'s loop stays a thin `book.apply(ledger.append(kind, session, as_of, payload))` walk.

SKIPPED relative to DESIGN.md section 10 (see `backtest.py`'s module docstring for the complete list): perturbation
variants (`confirm_entry` / `rank` variant-agreement), `entry_text` / `manage_text` requests (news is never sent to
Jev here - `text` is always `"off"`), forecasts / outcomes / calibration, `ledger.put_state`, the kill switch and
`risk.pre_cycle` / `on_mark` / `daily_loss` gating, reconcile, resume.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Final

import msgspec

from jevbot import canon, ids
from jevbot import questions as questions_module
from jevbot.candidates import CandidateGenerator, natural_limit
from jevbot.config import Config
from jevbot.errors import DataError, DeciderError
from jevbot.protocols import Decider, MarketView
from jevbot.risk import DefaultRiskEngine
from jevbot.rules import DecisionRules
from jevbot.state import StateBuilder
from jevbot.types import (
    MANDATORY_EXITS,
    ApprovedOrder,
    Band,
    BuiltState,
    Candidate,
    CandidateReject,
    ChoiceAns,
    DecisionRequest,
    DecisionResult,
    EntryContext,
    EntryDecision,
    ExitReason,
    LedgerKind,
    Leg,
    ManageDecision,
    NoulAns,
    OrderIntent,
    OrderLeg,
    OrderPurpose,
    Position,
    PositionIntent,
    RequestKind,
    RiskVerdict,
    ScoreAns,
    Side,
    SnapshotKey,
    Structure,
    Variant,
)

__all__ = ["LedgerWrite", "decide_entries", "decide_manages", "namespace_for"]

LedgerWrite = tuple[LedgerKind, dict[str, Any]]

_PPM: Final = 1_000_000


def namespace_for(cfg: Config, decider: Decider) -> str:
    return ids.namespace(cfg.run.experiment, decider.model, cfg.jev.refresh_generation)


def _headline_band(cfg: Config) -> Band:
    from jevbot.fills import headline_band as _hb

    return _hb(cfg)


def _open_legs(structure: Structure) -> tuple[OrderLeg, ...]:
    return tuple(
        OrderLeg(
            contract=leg.contract,
            side=leg.side,
            position_intent=PositionIntent.BTO if leg.side is Side.BUY else PositionIntent.STO,
        )
        for leg in structure.legs
    )


def _close_legs(structure: Structure) -> tuple[OrderLeg, ...]:
    return tuple(
        OrderLeg(
            contract=leg.contract,
            side=Side.SELL if leg.side is Side.BUY else Side.BUY,
            position_intent=PositionIntent.STC if leg.side is Side.BUY else PositionIntent.BTC,
        )
        for leg in structure.legs
    )


def _decide(decider: Decider, req: DecisionRequest) -> DecisionResult | DeciderError:
    try:
        return decider.decide(req)
    except DeciderError as exc:
        return exc


def _build_request(
    *, kind: RequestKind, decision_kind: str, question_set_id: str, namespace: str, subject: str, underlying: str, session: date, key: SnapshotKey, built: BuiltState
) -> DecisionRequest:
    questions = questions_module.question_set(question_set_id)
    return DecisionRequest(
        kind=kind,
        variant=Variant.BASE,
        question_set_id=question_set_id,
        state=built.state,
        questions=questions,
        state_hash=built.state_hash,
        question_set_hash=questions_module.QUESTION_SET_HASHES[question_set_id],
        namespace=namespace,
        subject=subject,
        underlying=underlying,
        session=session,
        key=key,
        decision_id=ids.decision_id(namespace, session, underlying, decision_kind, subject),  # type: ignore[arg-type]
    )


def _answer_ppm(answer: object) -> int:
    if isinstance(answer, NoulAns):
        return round(answer.p * _PPM)
    if isinstance(answer, ChoiceAns):
        return round(answer.p_top * _PPM)
    if isinstance(answer, ScoreAns):
        return round(answer.norm * _PPM)
    return 0


def _request_record(req: DecisionRequest, result: DecisionResult | DeciderError | None) -> dict[str, Any]:
    question_hashes = {qid: questions_module.QUESTION_HASHES[qid] for qid in req.questions}
    if isinstance(result, DecisionResult):
        return {
            "request_kind": req.kind.value,
            "variant": req.variant.value,
            "state_hash": req.state_hash,
            "question_set_id": req.question_set_id,
            "question_set_hash": req.question_set_hash,
            "question_hashes": question_hashes,
            "cache_keys": dict(result.cache_keys),
            "model": result.model,
            "source": result.source,
            "answers": {qid: _answer_ppm(ans) for qid, ans in result.answers.items()},
            "error": None,
        }
    return {
        "request_kind": req.kind.value,
        "variant": req.variant.value,
        "state_hash": req.state_hash,
        "question_set_id": req.question_set_id,
        "question_set_hash": req.question_set_hash,
        "question_hashes": question_hashes,
        "cache_keys": {},
        "model": "",
        "source": "",
        "answers": {},
        "error": None if result is None else type(result).__name__,
    }


def _no_intent_verdict(decision_id: str, codes: tuple[str, ...]) -> RiskVerdict:
    verdict_id = canon.sha256_hex(canon.dumps_sorted([decision_id, list(codes)]))[:24]
    return RiskVerdict(
        verdict_id=verdict_id, decision_id=decision_id, intent_id=None, approved=False, qty_approved=0, checks=(), reject_codes=codes, max_loss=0, bp_required=0
    )


def _candidate_summary(cand: Candidate | CandidateReject | None) -> dict[str, Any] | None:
    if not isinstance(cand, Candidate):
        return None
    return {
        "structure_id": cand.structure.structure_id,
        "legs": [[leg.contract.occ, leg.side.value] for leg in cand.structure.legs],
        "net": msgspec.to_builtins(cand.net),
        "max_loss_per_contract": cand.max_loss_per_contract,
        "budget_floor": cand.budget_floor,
        "rejects": list(cand.rejects),
    }


def _verdict_write(verdict: RiskVerdict, cand: Candidate | CandidateReject | None) -> LedgerWrite:
    payload = {**msgspec.to_builtins(verdict), "candidate": _candidate_summary(cand)}
    return (LedgerKind.RISK_VERDICT, payload)


# ======================================================================================================================
# manage
# ======================================================================================================================


def decide_manages(
    *,
    cfg: Config,
    namespace: str,
    session: date,
    as_of: datetime,
    view: MarketView,
    decider: Decider,
    rules: DecisionRules,
    state_builder: StateBuilder,
    risk: DefaultRiskEngine,
    positions: Sequence[Position],
    pf_state: Any,
) -> tuple[list[LedgerWrite], list[tuple[OrderIntent, ApprovedOrder]]]:
    """Hard exits first (`risk.hard_exit`), else `manage.v1` -> `DecisionRules.decide_manage`; closes are priced,
    sized at the position's full quantity and approved exactly like an entry (`risk.approve`, always last)."""
    writes: list[LedgerWrite] = []
    orders: list[tuple[OrderIntent, ApprovedOrder]] = []
    for pos in positions:
        built = state_builder.manage(view, pos)
        decision_id = ids.decision_id(namespace, session, pos.structure.underlying, "manage", pos.position_id)
        hard = risk.hard_exit(pos, view)
        core: DecisionResult | DeciderError | None = None
        req: DecisionRequest | None = None
        if hard is None:
            req = _build_request(
                kind=RequestKind.MANAGE,
                decision_kind="manage",
                question_set_id="manage.v1",
                namespace=namespace,
                subject=pos.position_id,
                underlying=pos.structure.underlying,
                session=session,
                key=view.key,
                built=built,
            )
            core = _decide(decider, req)
        core_result = core if isinstance(core, DecisionResult) else None
        md: ManageDecision = rules.decide_manage(pos, decision_id, hard, core_result, None, built.facts)
        requests = [] if req is None else [_request_record(req, core)]
        writes.append(
            (
                LedgerKind.DECISION,
                {
                    "decision_id": decision_id,
                    "kind": "manage",
                    "subject_alias": pos.position_id,
                    "text": "off",
                    "requests": requests,
                    "rules": msgspec.to_builtins(md),
                    "facts": msgspec.to_builtins(built.facts),
                    "tier": built.provenance.evidence_tier.value,
                },
            )
        )
        if md.action != "close":
            continue
        try:
            chain = view.chain(pos.structure.underlying)
        except DataError:
            continue  # nothing to price a close against this session; the latch/watch state is already ledgered
        quotes = [chain.quote(leg.contract) for leg in pos.structure.legs]
        if any(q is None for q in quotes):
            continue
        mandatory = hard is not None and ExitReason(hard) in MANDATORY_EXITS
        from jevbot.fills import BandFillModel  # local: avoids a module-level cycle with candidates.py's own import

        fill_model = BandFillModel(cfg)
        close_net, _fills, _quality = fill_model.price(_close_legs(pos.structure), chain, mandatory=mandatory)
        natural = natural_limit(cfg, pos.structure, quotes, close_net.worst)
        intent_id = ids.intent_id(namespace, session, decision_id, OrderPurpose.CLOSE, 0)
        intent = OrderIntent(
            intent_id=intent_id,
            decision_id=decision_id,
            position_id=pos.position_id,
            purpose=OrderPurpose.CLOSE,
            part=0,
            underlying=pos.structure.underlying,
            legs=_close_legs(pos.structure),
            qty=pos.qty,
            limit_start=natural,
            limit_natural=natural,
            reason=md.reason,
            mandatory=mandatory,
            session=session,
            key=view.key,
            structure=pos.structure,
        )
        verdict, order = risk.approve(intent, pf_state(), view, now=as_of, attempt=0, limit=natural, cand=None, approved_so_far=(), clock=None, market=False)
        writes.append(_verdict_write(verdict, None))
        if order is not None:
            writes.append((LedgerKind.ORDER_INTENT, msgspec.to_builtins(intent)))
            orders.append((intent, order))
    return writes, orders


# ======================================================================================================================
# entries
# ======================================================================================================================


@dataclass
class _Alias:
    alias: dict[str, str]


def decide_entries(
    *,
    cfg: Config,
    namespace: str,
    session: date,
    as_of: datetime,
    view: MarketView,
    decider: Decider,
    rules: DecisionRules,
    state_builder: StateBuilder,
    candidates: CandidateGenerator,
    risk: DefaultRiskEngine,
    pf_state: Any,
) -> tuple[list[LedgerWrite], list[tuple[OrderIntent, ApprovedOrder]]]:
    """`entry.v1` for every underlying -> `DecisionRules.decide_entry` -> `rank` -> `CandidateGenerator.build` ->
    `risk.size_entry` -> `risk.approve` (always last). No perturbation confirmation (`confirm_entry` is skipped)."""
    writes: list[LedgerWrite] = []
    entered: list[EntryDecision] = []
    built_by_u: dict[str, BuiltState] = {}
    for underlying in cfg.universe.underlyings:
        decision_id = ids.decision_id(namespace, session, underlying, "entry", ids.ENTRY_SUBJECT)
        built = state_builder.entry(view, underlying)
        if built is None:
            ed = rules.no_trade(underlying, decision_id, "dq:insufficient")
            writes.append(
                (
                    LedgerKind.DECISION,
                    {
                        "decision_id": decision_id,
                        "kind": "entry",
                        "subject_alias": cfg.universe.alias.get(underlying, underlying),
                        "text": "off",
                        "requests": [],
                        "rules": msgspec.to_builtins(ed),
                        "facts": {},
                        "tier": "NONE",
                    },
                )
            )
            continue
        built_by_u[underlying] = built
        req = _build_request(
            kind=RequestKind.ENTRY,
            decision_kind="entry",
            question_set_id="entry.v1",
            namespace=namespace,
            subject=ids.ENTRY_SUBJECT,
            underlying=underlying,
            session=session,
            key=view.key,
            built=built,
        )
        result = _decide(decider, req)
        if isinstance(result, DeciderError):
            ed = rules.no_trade(underlying, decision_id, "decider_failed_cycle")
        else:
            ed = rules.decide_entry(underlying, decision_id, result, None, built.facts)
        writes.append(
            (
                LedgerKind.DECISION,
                {
                    "decision_id": decision_id,
                    "kind": "entry",
                    "subject_alias": cfg.universe.alias.get(underlying, underlying),
                    "text": "off",
                    "requests": [_request_record(req, result)],
                    "rules": msgspec.to_builtins(ed),
                    "facts": msgspec.to_builtins(built.facts),
                    "tier": built.provenance.evidence_tier.value,
                },
            )
        )
        if ed.action == "enter":
            entered.append(ed)

    ranked = rules.rank(entered, cfg.universe.underlyings)
    approved: list[ApprovedOrder] = []
    orders: list[tuple[OrderIntent, ApprovedOrder]] = []
    for ed in ranked:
        cand = candidates.build(ed.kind, view, ed.underlying, budget_floor=risk.budget_floor(pf_state()))
        if isinstance(cand, CandidateReject) or cand.rejects:
            codes = tuple(f"candidate:{c}" for c in cand.rejects)
            writes.append(_verdict_write(_no_intent_verdict(ed.decision_id, codes), cand))
            continue
        qty = risk.size_entry(cand, ed.tier_ppm, pf_state(), approved)
        if qty == 0:
            writes.append(_verdict_write(_no_intent_verdict(ed.decision_id, ("risk:size_zero",)), cand))
            continue
        facts = built_by_u[ed.underlying].facts
        entry_ctx = EntryContext(
            entry_thesis=facts.thesis,
            entry_codes={"trend": facts.trend_code, "iv_vs_realized": facts.iv_rv_code, "iv_rank": facts.iv_rank_code},
            entry_spot=facts.spot,
            entry_iv30_bp=facts.iv30_bp,
            entry_em_hold_tenths=facts.em_hold_tenths,
            open_mid_at_decision=cand.net.mid,
        )
        natural = natural_limit(cfg, cand.structure, cand.quotes, cand.net.worst)
        intent_id = ids.intent_id(namespace, session, ed.decision_id, OrderPurpose.OPEN, 0)
        position_id = ids.position_id(namespace, session, cand.structure.structure_id)
        intent = OrderIntent(
            intent_id=intent_id,
            decision_id=ed.decision_id,
            position_id=position_id,
            purpose=OrderPurpose.OPEN,
            part=0,
            underlying=ed.underlying,
            legs=_open_legs(cand.structure),
            qty=qty,
            limit_start=natural,
            limit_natural=natural,
            reason="entry",
            mandatory=False,
            session=session,
            key=view.key,
            tier_ppm=ed.tier_ppm,
            structure=cand.structure,
            entry_ctx=entry_ctx,
        )
        verdict, order = risk.approve(intent, pf_state(), view, now=as_of, attempt=0, limit=natural, cand=cand, approved_so_far=tuple(approved), clock=None, market=False)
        writes.append(_verdict_write(verdict, cand))
        if order is not None:
            approved.append(order)
            writes.append((LedgerKind.ORDER_INTENT, msgspec.to_builtins(intent)))
            orders.append((intent, order))
    return writes, orders
