"""Property: the accounting identity of `portfolio.Book` holds on EVERY band under random open / partial / close sequences.

DESIGN.md 10.8 / 15.1 (`portfolio` row). Two identities, both tracked independently of the Book while the sequence is built:

    cash[b]   == initial_cash - sum over every FILL of net[b] * 100 * qty - sum of the FEE entries
    equity[b] == cash[b] - sum over open positions of liq_value * 100 * qty
              == initial_cash + realised P&L[b] + unrealised P&L[b] - fees

with `realised = (-open_net[b] - close_net[b]) * 100 * qty_closed` and `unrealised = (-open_net[b] - liq_value) * 100 * qty`.
Partial OPEN fills of one position share a net, so the against-us blending of 10.8 is exact here and the decomposition is an
equality, not a bound. Seeded numpy generators, no hypothesis (1.1).
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Any

import msgspec
import numpy as np

from jevbot.cal import XnysCalendar
from jevbot.config import Config
from jevbot.portfolio import Book
from jevbot.types import (
    Band,
    BandPrices,
    EntryContext,
    Fill,
    LedgerKind,
    Leg,
    LegFill,
    OptionContract,
    OrderIntent,
    OrderLeg,
    OrderPurpose,
    PositionIntent,
    Right,
    Side,
    Slot,
    SnapshotKey,
    Structure,
    StructureKind,
)
from tests.fixtures.memory_ledger import MemoryLedger

CAL = XnysCalendar()
CFG = Config()
SESSION = date(2024, 5, 17)
AS_OF = CAL.open_close(SESSION)[1]
EXPIRY = date(2024, 6, 21)
KEY = SnapshotKey(session=SESSION, slot=Slot.EOD)
INITIAL_CASH = 10_000_000
BANDS = (Band.ORATS, Band.WORST, Band.MID)
MULTIPLIER = 100

ENTRY_CTX = EntryContext(
    entry_thesis="t",
    entry_codes={"trend": "up", "iv_vs_realized": "rich", "iv_rank": "p40_p60"},
    entry_spot=45_000,
    entry_iv30_bp=1800,
    entry_em_hold_tenths=40,
    open_mid_at_decision=-100,
)


def rng(seed: int, purpose: str) -> np.random.Generator:
    """The 10.9 seeding recipe: draws never depend on call order."""
    digest = hashlib.sha256(f"{seed}|{purpose}".encode()).hexdigest()
    return np.random.Generator(np.random.PCG64(int(digest[:16], 16)))


def spread(index: int) -> Structure:
    short = OptionContract(underlying="SPY", expiry=EXPIRY, right=Right.PUT, strike_milli=400_000 + index * 1_000)
    long = OptionContract(underlying="SPY", expiry=EXPIRY, right=Right.PUT, strike_milli=395_000 + index * 1_000)
    return Structure(
        kind=StructureKind.PUT_CREDIT,
        underlying="SPY",
        expiry=EXPIRY,
        last_session=EXPIRY,
        legs=(Leg(contract=long, side=Side.BUY), Leg(contract=short, side=Side.SELL)),
    )


def order_legs(structure: Structure, *, closing: bool) -> tuple[OrderLeg, ...]:
    out = []
    for leg in structure.legs:
        if closing:
            side = Side.SELL if leg.side is Side.BUY else Side.BUY
            intent = PositionIntent.STC if leg.side is Side.BUY else PositionIntent.BTC
        else:
            side = leg.side
            intent = PositionIntent.BTO if leg.side is Side.BUY else PositionIntent.STO
        out.append(OrderLeg(contract=leg.contract, side=side, position_intent=intent))
    return tuple(out)


def intent_of(structure: Structure, index: int, *, purpose: OrderPurpose, qty: int, tag: str) -> OrderIntent:
    return OrderIntent(
        intent_id=f"jb1-prop-{index:04d}-{tag}",
        decision_id=f"{index:032d}",
        position_id=f"pos{index:04d}",
        purpose=purpose,
        part=0,
        underlying="SPY",
        legs=order_legs(structure, closing=purpose is not OrderPurpose.OPEN),
        qty=qty,
        limit_start=-100,
        limit_natural=-95,
        reason="entry" if purpose is OrderPurpose.OPEN else "profit_target",
        mandatory=False,
        session=SESSION,
        key=KEY,
        tier_ppm=1_000_000 if purpose is OrderPurpose.OPEN else 0,
        structure=structure,
        entry_ctx=ENTRY_CTX if purpose is OrderPurpose.OPEN else None,
    )


def fill_of(intent: OrderIntent, *, qty: int, cum: int, net: BandPrices, tag: str) -> dict[str, Any]:
    fill = Fill(
        fill_id=f"{intent.intent_id}-{tag}-{cum}",
        client_order_id=f"{intent.intent_id}-00",
        intent_id=intent.intent_id,
        decision_id=intent.decision_id,
        position_id=intent.position_id,
        purpose=intent.purpose,
        structure_id=intent.structure.structure_id if intent.structure else None,
        qty=qty,
        key=KEY,
        ts=AS_OF,
        net=net,
        legs=tuple(LegFill(occ=leg.contract.occ, side=leg.side, bid=98, ask=102, orats=100, worst=100, mid=100) for leg in intent.legs),
        fees_micro=0,
        forced=False,
        model_reject=(),
        quality="ok",
        source="sim",
        broker_order_id=None,
        broker_net=None,
    )
    return msgspec.to_builtins(fill)


def add(book: Book, ledger: MemoryLedger, kind: LedgerKind, payload: dict[str, Any]) -> None:
    book.apply(ledger.append(kind, SESSION, AS_OF, payload))


def mark_payload(marks: dict[str, int]) -> dict[str, Any]:
    zero = msgspec.to_builtins(BandPrices(orats=0, worst=0, mid=0))
    return {
        "equity": zero,
        "cash": zero,
        "open_max_loss": 0,
        "bp_used": 0,
        "bp_utilisation_ppm": 0,
        "positions": {pid: {"liq_value": liq, "mid_value": liq, "stale": False} for pid, liq in marks.items()},
        "net_delta_milli": 0,
        "net_vega_milli": 0,
    }


def test_cash_equity_and_pnl_identities_hold_on_every_band() -> None:
    for seed in range(25):
        draw = rng(seed, "portfolio-identity")
        ledger = MemoryLedger(strict=True)
        book = Book(initial_cash=INITIAL_CASH, headline=Band.ORATS, cfg=CFG, calendar=CAL)

        add(book, ledger, LedgerKind.SESSION_START, {"session": SESSION.isoformat(), "slot": "eod", "phase": "full"})

        expected_cash = dict.fromkeys(BANDS, INITIAL_CASH)
        fees_total = 0
        realised = dict.fromkeys(BANDS, 0)
        live: dict[str, dict[str, Any]] = {}  # position_id -> {"qty", "open_net", "liq"}
        marks: dict[str, int] = {}

        for index in range(int(draw.integers(2, 7))):
            structure = spread(index)
            opening = intent_of(structure, index, purpose=OrderPurpose.OPEN, qty=int(draw.integers(1, 5)), tag="open")
            add(book, ledger, LedgerKind.ORDER_INTENT, msgspec.to_builtins(opening))
            open_net = BandPrices(
                orats=int(draw.integers(-300, -50)),
                worst=int(draw.integers(-300, -50)),
                mid=int(draw.integers(-300, -50)),
            )
            # one or two partials, at the SAME net (10.8 blends against us; equal inputs blend exactly)
            chunks = [opening.qty] if opening.qty == 1 or draw.random() < 0.5 else [1, opening.qty - 1]
            cum = 0
            for part, chunk in enumerate(chunks):
                cum += chunk
                add(book, ledger, LedgerKind.FILL, fill_of(opening, qty=chunk, cum=cum, net=open_net, tag=f"o{part}"))
                for band in BANDS:
                    expected_cash[band] -= open_net.get(band) * MULTIPLIER * chunk
            live[opening.position_id] = {"qty": opening.qty, "open_net": open_net}

            liq = int(draw.integers(-50, 400))
            marks[opening.position_id] = liq
            add(book, ledger, LedgerKind.MARK, mark_payload(marks))

            if draw.random() < 0.6:  # close all or part of it
                closed = opening.qty if draw.random() < 0.6 else int(draw.integers(1, opening.qty + 1))
                closing = intent_of(structure, index, purpose=OrderPurpose.CLOSE, qty=closed, tag="close")
                close_net = BandPrices(
                    orats=int(draw.integers(-200, 400)),
                    worst=int(draw.integers(-200, 400)),
                    mid=int(draw.integers(-200, 400)),
                )
                add(book, ledger, LedgerKind.ORDER_INTENT, msgspec.to_builtins(closing))
                add(book, ledger, LedgerKind.FILL, fill_of(closing, qty=closed, cum=closed, net=close_net, tag="c0"))
                for band in BANDS:
                    expected_cash[band] -= close_net.get(band) * MULTIPLIER * closed
                    realised[band] += (-open_net.get(band) - close_net.get(band)) * MULTIPLIER * closed
                live[opening.position_id]["qty"] -= closed
                if live[opening.position_id]["qty"] == 0:
                    del live[opening.position_id]
                    marks.pop(opening.position_id, None)

            if draw.random() < 0.3:
                fee = int(draw.integers(1, 40))
                add(book, ledger, LedgerKind.FEE, {"fee_cents": fee, "accrued_micro_before": fee * 10_000})
                fees_total += fee
                for band in BANDS:
                    expected_cash[band] -= fee

        state = book.state()
        for band in BANDS:
            assert state.cash.get(band) == expected_cash[band], (seed, band)
            held = sum(marks[pid] * MULTIPLIER * data["qty"] for pid, data in live.items())
            assert state.equity.get(band) == state.cash.get(band) - held, (seed, band)
            unrealised = sum((-data["open_net"].get(band) - marks[pid]) * MULTIPLIER * data["qty"] for pid, data in live.items())
            assert state.equity.get(band) == INITIAL_CASH + realised[band] + unrealised - fees_total, (seed, band)
        assert {p.position_id for p in state.positions} == set(live)
        assert Book.replay(ledger, initial_cash=INITIAL_CASH, headline=Band.ORATS, cfg=CFG, calendar=CAL).state() == state


def test_leg_positions_mirror_the_open_structures_exactly() -> None:
    for seed in range(10):
        draw = rng(seed, "portfolio-legs")
        ledger = MemoryLedger(strict=True)
        book = Book(initial_cash=INITIAL_CASH, headline=Band.ORATS, cfg=CFG, calendar=CAL)
        book.apply(
            ledger.append(LedgerKind.SESSION_START, SESSION, AS_OF, {"session": SESSION.isoformat(), "slot": "eod", "phase": "full"})
        )
        expected: dict[str, int] = {}
        for index in range(int(draw.integers(1, 5))):
            structure = spread(index)
            qty = int(draw.integers(1, 4))
            opening = intent_of(structure, index, purpose=OrderPurpose.OPEN, qty=qty, tag="open")
            book.apply(ledger.append(LedgerKind.ORDER_INTENT, SESSION, AS_OF, msgspec.to_builtins(opening)))
            net = BandPrices(orats=-100, worst=-90, mid=-105)
            book.apply(ledger.append(LedgerKind.FILL, SESSION, AS_OF, fill_of(opening, qty=qty, cum=qty, net=net, tag="o")))
            for leg in structure.legs:
                signed = qty * (1 if leg.side is Side.BUY else -1)
                expected[leg.contract.occ] = expected.get(leg.contract.occ, 0) + signed
        assert book.leg_positions() == {occ: qty for occ, qty in sorted(expected.items()) if qty != 0}
