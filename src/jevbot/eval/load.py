"""Run store -> pandas frames (DESIGN.md 12.2 / 12.3 / 12.8 over the schema of 13.4).

This is the ONLY door between a `run.sqlite` and the rest of `eval/`: the report, the metrics, the calibration machinery
and the model-agreement report all read frames from here and never open a store themselves. Three rules shape it:

* **The chain is verified first.** `RunStore.verify()` recomputes every hash from the persisted texts through
  `canon.ledger_entry_hash` (the ONE formula of 2.7), so a report can never be written from a tampered store (INV-19).
* **Forecasts join outcomes by `event_key`** (12.3), never by decision or session: Jev, every baseline decider and every
  reference are then scored on exactly the same events.
* **A missing forecast is a row, not a gap.** `p_ppm = NULL` in the FORECAST payload means the request's state was built
  but the request failed or was suppressed; it loads as `p = NaN` with `missing = True` and its `missing_reason`, and is
  imputed with the reference forecast by the pre-registration (12.1 `missing`). It is never dropped.

`reference_history()` returns the Jev-free training pairs of `[prereg.reference_history]` (12.1 / 12.3). It is the one
loader whose rows may come from another tier and price measure, and it is passed **only** to the two reference builders
of `eval/calibration.py`; nothing else in `eval/` accepts it, so those rows can never reach a scored table.
"""

import json
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Final

import msgspec
import numpy as np
import pandas as pd

from jevbot import canon
from jevbot.errors import DataUnavailable, LedgerCorrupt
from jevbot.types import LedgerKind, RunMeta

__all__ = [
    "BANDS",
    "CALIBRATION_COLUMNS",
    "REFERENCE_HISTORY_COLUMNS",
    "RunStore",
    "StoreArg",
    "anomalies_frame",
    "calibration_frame",
    "daily_frame",
    "decisions_frame",
    "fills_frame",
    "forecasts_frame",
    "intents_frame",
    "kill_affected_sessions",
    "kill_frame",
    "marks_frame",
    "open_stores",
    "order_status_frame",
    "outcomes_frame",
    "position_marks_frame",
    "reference_history",
    "risk_events_frame",
    "risk_verdicts_frame",
    "sidecar_frame",
    "trades_frame",
]

BANDS: Final[tuple[str, ...]] = ("orats", "worst", "mid")

CALIBRATION_COLUMNS: Final[tuple[str, ...]] = (
    "run_id",
    "namespace",
    "session",
    "underlying",
    "question_id",
    "with_text",
    "event_key",
    "forecast_id",
    "decision_id",
    "p",
    "p_ppm",
    "missing",
    "missing_reason",
    "p_abstain",
    "p_implied",
    "p_implied_ppm",
    "implied_method",
    "implied_quality",
    "horizon",
    "resolve_on",
    "resolved_on",
    "y",
    "resolved",
    "void",
    "div_in_window",
    "tier",
    "fidelity",
    "price_measure",
    "iv_history",
    "prereg",
)

REFERENCE_HISTORY_COLUMNS: Final[tuple[str, ...]] = ("question_id", "horizon", "session", "p_implied", "y", "resolved_on")


# ======================================================================================================================
# The store handle
# ======================================================================================================================


class RunStore:
    """A read-only handle on one `run.sqlite` (13.4).

    Opened read-only through SQLite's URI mode: evaluation never writes to a run store, and an accidental write is an
    error rather than a silent mutation of the evidence.
    """

    def __init__(self, path: Path | str, *, verify: bool = False) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise DataUnavailable(f"run store {self.path} does not exist")
        self.conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row
        self._meta: RunMeta | None = None
        if verify:
            self.verify()

    # --- lifecycle ------------------------------------------------------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "RunStore":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"RunStore({self.path.as_posix()!r})"

    # --- meta -----------------------------------------------------------------------------------------------------

    def meta_value(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    @property
    def meta(self) -> RunMeta:
        """The `RunMeta` stored under the `run_meta` key of the `meta` table (13.4)."""
        if self._meta is None:
            raw = self.meta_value("run_meta")
            if raw is None:
                raise DataUnavailable(f"run store {self.path} has no `run_meta` row in its meta table")
            try:
                self._meta = msgspec.json.decode(raw.encode(), type=RunMeta)
            except msgspec.MsgspecError as exc:
                raise DataUnavailable(f"run store {self.path}: `run_meta` is not a RunMeta: {exc}") from exc
        return self._meta

    @property
    def run_id(self) -> str:
        return self.meta.run_id

    # --- the ledger -----------------------------------------------------------------------------------------------

    def head(self) -> tuple[int, str]:
        """`(seq, hash)` of the last entry; `(0, GENESIS_HASH)` for an empty store."""
        row = self.conn.execute("SELECT seq, hash FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
        if row is None:
            return 0, canon.GENESIS_HASH
        return int(row["seq"]), str(row["hash"])

    def verify(self) -> None:
        """Recompute the whole hash chain from the persisted texts (INV-19). Raises `LedgerCorrupt`.

        The report verifies the chain before it writes anything (12.8), so a truncated, re-ordered or edited store is
        caught before a single number is printed.
        """
        prev = canon.GENESIS_HASH
        expected_seq = 1
        for row in self.conn.execute("SELECT seq, kind, session, as_of, payload, prev_hash, hash FROM ledger ORDER BY seq"):
            seq = int(row["seq"])
            if seq != expected_seq:
                raise LedgerCorrupt(f"{self.path}: ledger seq {seq} breaks the gapless 1..n sequence (expected {expected_seq})")
            if str(row["prev_hash"]) != prev:
                raise LedgerCorrupt(f"{self.path}: ledger entry {seq} does not chain onto its predecessor")
            try:
                recomputed = canon.ledger_entry_hash(
                    prev, seq, str(row["kind"]), str(row["session"]), str(row["as_of"]), str(row["payload"])
                )
            except (TypeError, ValueError) as exc:
                raise LedgerCorrupt(f"{self.path}: ledger entry {seq} is not canonically encoded: {exc}") from exc
            if recomputed != str(row["hash"]):
                raise LedgerCorrupt(f"{self.path}: ledger entry {seq} hash mismatch (payload or header edited)")
            prev = recomputed
            expected_seq += 1

    def rows(self, kind: LedgerKind | str | None = None) -> Iterator[dict[str, Any]]:
        """Every ledger entry (optionally of one kind) as `{seq, kind, session, as_of, payload}`, payload decoded."""
        if kind is None:
            cursor = self.conn.execute("SELECT seq, kind, session, as_of, payload FROM ledger ORDER BY seq")
        else:
            value = kind.value if isinstance(kind, LedgerKind) else str(kind)
            cursor = self.conn.execute("SELECT seq, kind, session, as_of, payload FROM ledger WHERE kind = ? ORDER BY seq", (value,))
        for row in cursor:
            yield {
                "seq": int(row["seq"]),
                "kind": str(row["kind"]),
                "session": date.fromisoformat(str(row["session"])),
                "as_of": _parse_as_of(str(row["as_of"])),
                "payload": json.loads(str(row["payload"])),
            }

    def sessions(self) -> tuple[date, ...]:
        """Every session the run ledgered, ascending - the run's own trading calendar.

        Holding periods are counted in SESSIONS (12.2), and a run store knows its own sessions exactly; deriving them
        from calendar days would count a weekend as two trading days.
        """
        rows = self.conn.execute("SELECT DISTINCT session FROM ledger ORDER BY session").fetchall()
        return tuple(date.fromisoformat(str(row["session"])) for row in rows)

    def payload_frame(self, kind: LedgerKind | str) -> pd.DataFrame:
        """`seq, session, as_of` plus one column per top-level payload key, for one ledger kind."""
        records = [{"seq": row["seq"], "session": row["session"], "as_of": row["as_of"], **row["payload"]} for row in self.rows(kind)]
        return pd.DataFrame.from_records(records)


StoreArg = RunStore | Path | str


def _store(arg: StoreArg) -> tuple[RunStore, bool]:
    """Normalise a store argument; the second element says whether the caller must close it."""
    if isinstance(arg, RunStore):
        return arg, False
    return RunStore(arg), True


def _store_args(stores: "StoreArg | Sequence[StoreArg]") -> list[StoreArg]:
    """One store or several: `calibration_frame(store)` and `calibration_frame([a, b])` both work."""
    if isinstance(stores, RunStore | Path | str):
        return [stores]
    return list(stores)


def open_stores(args: Iterable[StoreArg], *, verify: bool = False) -> list[RunStore]:
    """Open several run stores; a path becomes a `RunStore`, an open handle is passed through."""
    out: list[RunStore] = []
    for arg in args:
        if isinstance(arg, RunStore):
            if verify:
                arg.verify()
            out.append(arg)
        else:
            out.append(RunStore(arg, verify=verify))
    return out


def _parse_as_of(text: str) -> datetime:
    """Parse the persisted RFC 3339 UTC text of `canon.render_as_of` back into a tz-aware UTC datetime."""
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:  # pragma: no cover - a conforming store always writes the offset
        raise LedgerCorrupt(f"ledger as_of {text!r} is not tz-aware")
    return parsed.astimezone(UTC)


# ======================================================================================================================
# Forecasts, outcomes and the calibration frame (12.3)
# ======================================================================================================================


def forecasts_frame(store: StoreArg) -> pd.DataFrame:
    """Every FORECAST entry of one store, `p_ppm = NULL` kept as a MISSING row (13.4 `v_forecasts`)."""
    handle, owned = _store(store)
    try:
        meta = handle.meta
        records: list[dict[str, Any]] = []
        for row in handle.rows(LedgerKind.FORECAST):
            payload = row["payload"]
            spec = payload.get("spec") or {}
            p_ppm = payload.get("p_ppm")
            p_implied_ppm = payload.get("p_implied_ppm")
            p_abstain_ppm = payload.get("p_abstain_ppm")
            records.append(
                {
                    "run_id": meta.run_id,
                    "namespace": meta.namespace,
                    "session": row["session"],
                    "as_of": row["as_of"],
                    "underlying": payload.get("underlying"),
                    "question_id": payload.get("question_id"),
                    "question_hash": payload.get("question_hash"),
                    "with_text": bool(payload.get("with_text")),
                    "event_key": payload.get("event_key"),
                    "forecast_id": payload.get("forecast_id"),
                    "decision_id": payload.get("decision_id"),
                    "p_ppm": p_ppm,
                    "p": _ppm_to_p(p_ppm),
                    "missing": p_ppm is None,
                    "missing_reason": payload.get("missing_reason"),
                    "p_abstain": _ppm_to_p(p_abstain_ppm),
                    "p_implied_ppm": p_implied_ppm,
                    "p_implied": _ppm_to_p(p_implied_ppm),
                    "implied_method": payload.get("implied_method"),
                    "implied_quality": payload.get("implied_quality"),
                    "horizon": spec.get("horizon_sessions"),
                    "resolve_on": _opt_date(spec.get("resolve_on")),
                    "spec_kind": spec.get("kind"),
                    "tier": payload.get("tier"),
                    "fidelity": payload.get("fidelity"),
                    "price_measure": meta.spot_measure,
                    "iv_history": payload.get("iv_history"),
                    "prereg": bool(payload.get("prereg")),
                }
            )
        frame = pd.DataFrame.from_records(records)
        if frame.empty:
            return _empty(
                (
                    "run_id",
                    "namespace",
                    "session",
                    "as_of",
                    "underlying",
                    "question_id",
                    "question_hash",
                    "with_text",
                    "event_key",
                    "forecast_id",
                    "decision_id",
                    "p_ppm",
                    "p",
                    "missing",
                    "missing_reason",
                    "p_abstain",
                    "p_implied_ppm",
                    "p_implied",
                    "implied_method",
                    "implied_quality",
                    "horizon",
                    "resolve_on",
                    "spec_kind",
                    "tier",
                    "fidelity",
                    "price_measure",
                    "iv_history",
                    "prereg",
                )
            )
        frame["p_ppm"] = frame["p_ppm"].astype("Int64")
        frame["p_implied_ppm"] = frame["p_implied_ppm"].astype("Int64")
        frame["horizon"] = frame["horizon"].astype("Int64")
        return frame
    finally:
        if owned:
            handle.close()


def outcomes_frame(store: StoreArg) -> pd.DataFrame:
    """Every OUTCOME entry (13.4 `v_outcomes`). `y = None` is a VOID outcome: reported, never dropped silently."""
    handle, owned = _store(store)
    try:
        records = [
            {
                "event_key": row["payload"].get("event_key"),
                "resolved_on": _opt_date(row["payload"].get("resolved_on")),
                "y": row["payload"].get("y"),
                "observed": row["payload"].get("observed") or {},
                "div_in_window": row["payload"].get("div_in_window"),
                "outcome_price_measure": row["payload"].get("price_measure"),
            }
            for row in handle.rows(LedgerKind.OUTCOME)
        ]
        frame = pd.DataFrame.from_records(records)
        if frame.empty:
            return _empty(("event_key", "resolved_on", "y", "observed", "div_in_window", "outcome_price_measure"))
        frame["y"] = frame["y"].astype("Int64")
        return frame
    finally:
        if owned:
            handle.close()


def calibration_frame(stores: StoreArg | Sequence[StoreArg], *, resolved_only: bool = True) -> pd.DataFrame:
    """FORECAST joined to OUTCOME **by `event_key`** over one or more run stores (12.3).

    Every decider and every reference asked about the same event therefore lands on the same row key. Unresolved
    forecasts are dropped by default (`resolved_only`); VOID outcomes (`y` is NA) are kept so the report can list them,
    and a MISSING forecast (`p_ppm = NULL`) keeps `p = NaN` with `missing = True`.
    """
    args = _store_args(stores)
    parts: list[pd.DataFrame] = []
    for arg in args:
        handle, owned = _store(arg)
        try:
            forecasts = forecasts_frame(handle)
            outcomes = outcomes_frame(handle)
            if forecasts.empty:
                continue
            how = "inner" if resolved_only else "left"
            joined = forecasts.merge(outcomes, on="event_key", how=how)
            if "y" not in joined.columns:
                joined["y"] = pd.array([], dtype="Int64") if joined.empty else pd.NA
            joined["y"] = joined["y"].astype("Int64")
            # "resolved" = an OUTCOME entry exists; "void" = it exists and carries y = null (a data gap).
            # Voids are reported, never dropped silently (2.7); an unresolved forecast is simply still open.
            joined["resolved"] = joined["resolved_on"].notna() if "resolved_on" in joined.columns else False
            joined["void"] = joined["resolved"] & joined["y"].isna()
            parts.append(joined)
        finally:
            if owned:
                handle.close()
    if not parts:
        return _empty(CALIBRATION_COLUMNS)
    frame = pd.concat(parts, ignore_index=True)
    ordered = [column for column in CALIBRATION_COLUMNS if column in frame.columns]
    rest = [column for column in frame.columns if column not in ordered]
    return frame[ordered + rest]


def reference_history(store: StoreArg) -> pd.DataFrame:
    """The Jev-free training pairs of `[prereg.reference_history]` (12.1, 12.3).

    Columns: `question_id, horizon, session, p_implied, y, resolved_on` - and nothing else. Only rows with a real
    `p_implied` and a resolved (non-void) outcome are training pairs; the tier and price measure of this run may differ
    from the evaluated run's, which is exactly why the pooling guard exempts it *for this purpose alone* and why these
    columns carry no Jev output at all.
    """
    frame = calibration_frame(store, resolved_only=True)
    if frame.empty:
        return _empty(REFERENCE_HISTORY_COLUMNS)
    usable = frame[frame["p_implied"].notna() & frame["y"].notna()]
    out = usable.loc[:, ["question_id", "horizon", "session", "p_implied", "y", "resolved_on"]].copy()
    out["horizon"] = out["horizon"].astype("Int64")
    out["y"] = out["y"].astype("Int64")
    return out.reset_index(drop=True)


# ======================================================================================================================
# P&L frames (12.2)
# ======================================================================================================================


def daily_frame(stores: StoreArg | Sequence[StoreArg]) -> pd.DataFrame:
    """One row per session per store from the SESSION_END entries: equity and cash per band (13.4 `v_daily`).

    The headline-band equity of a session is the next session's `day_start_equity` (9.5), so this frame is the P&L
    series every metric, bootstrap and figure is computed from.
    """
    args = _store_args(stores)
    parts: list[pd.DataFrame] = []
    for arg in args:
        handle, owned = _store(arg)
        try:
            meta = handle.meta
            records: list[dict[str, Any]] = []
            for row in handle.rows(LedgerKind.SESSION_END):
                payload = row["payload"]
                equity = _band_map(payload, "equity")
                cash = _band_map(payload, "cash")
                records.append(
                    {
                        "run_id": meta.run_id,
                        "namespace": meta.namespace,
                        "session": row["session"],
                        **{f"equity_{band}": equity.get(band) for band in BANDS},
                        **{f"cash_{band}": cash.get(band) for band in BANDS},
                        "n_decisions": payload.get("n_decisions"),
                        "n_forecasts": payload.get("n_forecasts"),
                        "n_intents": payload.get("n_intents"),
                        "invariant_no_expiry_risk": payload.get("invariant_no_expiry_risk"),
                        "positions_digest": payload.get("positions_digest"),
                    }
                )
            if records:
                parts.append(pd.DataFrame.from_records(records))
        finally:
            if owned:
                handle.close()
    columns = (
        "run_id",
        "namespace",
        "session",
        *(f"equity_{band}" for band in BANDS),
        *(f"cash_{band}" for band in BANDS),
        "n_decisions",
        "n_forecasts",
        "n_intents",
        "invariant_no_expiry_risk",
        "positions_digest",
    )
    if not parts:
        return _empty(columns)
    frame = pd.concat(parts, ignore_index=True).sort_values(["run_id", "session"]).reset_index(drop=True)
    return frame


def position_marks_frame(store: StoreArg) -> pd.DataFrame:
    """The per-position part of every MARK entry: `session, position_id, liq_value, mid_value, stale` (2.11).

    `liq_value` is the conservative liquidation value in signed cents per share - what closing the structure would COST
    now, negative when it would pay us (10.5) - which is what the maximum-adverse-excursion metric of 12.2 needs.
    """
    handle, owned = _store(store)
    try:
        records: list[dict[str, Any]] = []
        for row in handle.rows(LedgerKind.MARK):
            positions = row["payload"].get("positions") or {}
            if not isinstance(positions, dict):
                continue
            for position_id, entry in positions.items():
                if not isinstance(entry, dict):
                    continue
                records.append(
                    {
                        "seq": row["seq"],
                        "session": row["session"],
                        "position_id": position_id,
                        "liq_value": entry.get("liq_value"),
                        "mid_value": entry.get("mid_value"),
                        "stale": bool(entry.get("stale")),
                    }
                )
        columns = ("seq", "session", "position_id", "liq_value", "mid_value", "stale")
        return pd.DataFrame.from_records(records) if records else _empty(columns)
    finally:
        if owned:
            handle.close()


def marks_frame(store: StoreArg) -> pd.DataFrame:
    """MARK entries: equity / cash per band, exposure and greeks (2.11)."""
    handle, owned = _store(store)
    try:
        records: list[dict[str, Any]] = []
        for row in handle.rows(LedgerKind.MARK):
            payload = row["payload"]
            equity = _band_map(payload, "equity")
            cash = _band_map(payload, "cash")
            records.append(
                {
                    "seq": row["seq"],
                    "session": row["session"],
                    "as_of": row["as_of"],
                    **{f"equity_{band}": equity.get(band) for band in BANDS},
                    **{f"cash_{band}": cash.get(band) for band in BANDS},
                    "open_max_loss": payload.get("open_max_loss"),
                    "bp_used": payload.get("bp_used"),
                    "bp_utilisation_ppm": payload.get("bp_utilisation_ppm"),
                    "net_delta_milli": payload.get("net_delta_milli"),
                    "net_vega_milli": payload.get("net_vega_milli"),
                    "n_positions": len(payload.get("positions") or {}),
                    "stale_positions": sum(
                        1 for entry in (payload.get("positions") or {}).values() if isinstance(entry, dict) and entry.get("stale")
                    ),
                }
            )
        columns = (
            "seq",
            "session",
            "as_of",
            *(f"equity_{band}" for band in BANDS),
            *(f"cash_{band}" for band in BANDS),
            "open_max_loss",
            "bp_used",
            "bp_utilisation_ppm",
            "net_delta_milli",
            "net_vega_milli",
            "n_positions",
            "stale_positions",
        )
        return pd.DataFrame.from_records(records) if records else _empty(columns)
    finally:
        if owned:
            handle.close()


def fills_frame(store: StoreArg) -> pd.DataFrame:
    """FILL entries flattened: net per band, fees, forced / quality flags, realised P&L per band on closing fills.

    `zero_bid_close_legs` counts SELL legs of a CLOSE / KILL fill whose recorded `bid` was 0 - the normal state of a
    winning short-premium wing (10.4), reported in the fills table of every report (12.2).
    """
    handle, owned = _store(store)
    try:
        records: list[dict[str, Any]] = []
        for row in handle.rows(LedgerKind.FILL):
            payload = row["payload"]
            net = _band_map(payload, "net")
            realised = _band_map(payload, "realised_pnl")
            legs = payload.get("legs") or []
            purpose = payload.get("purpose")
            zero_bid_close_legs = sum(
                1
                for leg in legs
                if isinstance(leg, dict) and leg.get("side") == "sell" and int(leg.get("bid") or 0) == 0 and purpose in ("close", "kill")
            )
            records.append(
                {
                    "seq": row["seq"],
                    "session": row["session"],
                    "as_of": row["as_of"],
                    "fill_id": payload.get("fill_id"),
                    "client_order_id": payload.get("client_order_id"),
                    "intent_id": payload.get("intent_id"),
                    "decision_id": payload.get("decision_id"),
                    "position_id": payload.get("position_id"),
                    "purpose": purpose,
                    "structure_id": payload.get("structure_id"),
                    "qty": payload.get("qty"),
                    **{f"net_{band}": net.get(band) for band in BANDS},
                    **{f"realised_{band}": realised.get(band) for band in BANDS},
                    "fees_micro": payload.get("fees_micro"),
                    "forced": bool(payload.get("forced")),
                    "quality": payload.get("quality"),
                    "source": payload.get("source"),
                    "n_legs": len(legs),
                    "zero_bid_close_legs": zero_bid_close_legs,
                    "model_reject": tuple(payload.get("model_reject") or ()),
                }
            )
        columns = (
            "seq",
            "session",
            "as_of",
            "fill_id",
            "client_order_id",
            "intent_id",
            "decision_id",
            "position_id",
            "purpose",
            "structure_id",
            "qty",
            *(f"net_{band}" for band in BANDS),
            *(f"realised_{band}" for band in BANDS),
            "fees_micro",
            "forced",
            "quality",
            "source",
            "n_legs",
            "zero_bid_close_legs",
            "model_reject",
        )
        return pd.DataFrame.from_records(records) if records else _empty(columns)
    finally:
        if owned:
            handle.close()


def intents_frame(store: StoreArg) -> pd.DataFrame:
    """ORDER_INTENT entries: the `reason` column is `"entry"` or an `ExitReason` value (2.4) - the exit-reason mix."""
    handle, owned = _store(store)
    try:
        records = [
            {
                "seq": row["seq"],
                "session": row["session"],
                "intent_id": row["payload"].get("intent_id"),
                "decision_id": row["payload"].get("decision_id"),
                "position_id": row["payload"].get("position_id"),
                "purpose": row["payload"].get("purpose"),
                "underlying": row["payload"].get("underlying"),
                "qty": row["payload"].get("qty"),
                "reason": row["payload"].get("reason"),
                "mandatory": bool(row["payload"].get("mandatory")),
                "tier_ppm": row["payload"].get("tier_ppm"),
                "kind": (row["payload"].get("structure") or {}).get("kind"),
            }
            for row in handle.rows(LedgerKind.ORDER_INTENT)
        ]
        columns = (
            "seq",
            "session",
            "intent_id",
            "decision_id",
            "position_id",
            "purpose",
            "underlying",
            "qty",
            "reason",
            "mandatory",
            "tier_ppm",
            "kind",
        )
        return pd.DataFrame.from_records(records) if records else _empty(columns)
    finally:
        if owned:
            handle.close()


def order_status_frame(store: StoreArg) -> pd.DataFrame:
    """ORDER_STATUS entries (2.11): the one writer is `reconcile.record_order_status`."""
    handle, owned = _store(store)
    try:
        records = [
            {
                "seq": row["seq"],
                "session": row["session"],
                "client_order_id": row["payload"].get("client_order_id"),
                "intent_id": row["payload"].get("intent_id"),
                "attempt": row["payload"].get("attempt"),
                "status": row["payload"].get("status"),
                "qty": row["payload"].get("qty"),
                "filled_qty": row["payload"].get("filled_qty"),
                "limit": row["payload"].get("limit"),
                "reject_code": row["payload"].get("reject_code"),
                "tag": row["payload"].get("tag"),
            }
            for row in handle.rows(LedgerKind.ORDER_STATUS)
        ]
        columns = (
            "seq",
            "session",
            "client_order_id",
            "intent_id",
            "attempt",
            "status",
            "qty",
            "filled_qty",
            "limit",
            "reject_code",
            "tag",
        )
        return pd.DataFrame.from_records(records) if records else _empty(columns)
    finally:
        if owned:
            handle.close()


def trades_frame(store: StoreArg) -> pd.DataFrame:
    """Closed trades, assembled from the FILL entries of one position (12.2 "From closed trades").

    A trade is one `position_id`: the OPEN fill gives the entry session, band nets and quantity, the CLOSE / KILL fills
    give the exit session, the realised P&L per band and - through the ORDER_INTENT they belong to - the exit reason.
    Positions still open at the end of the run are not trades and are excluded (they carry `closed = False`).
    """
    handle, owned = _store(store)
    try:
        fills = fills_frame(handle)
        intents = intents_frame(handle)
        if fills.empty:
            return _empty(
                (
                    "position_id",
                    "underlying",
                    "kind",
                    "qty",
                    "entry_session",
                    "exit_session",
                    "holding_sessions",
                    "exit_reason",
                    "forced",
                    "fees_micro",
                    "closed",
                    *(f"pnl_{band}" for band in BANDS),
                    *(f"open_net_{band}" for band in BANDS),
                    *(f"close_net_{band}" for band in BANDS),
                )
            )
        session_index = {session: index for index, session in enumerate(handle.sessions())}
        reason_by_intent = dict(zip(intents["intent_id"], intents["reason"], strict=False)) if not intents.empty else {}
        underlying_by_intent = dict(zip(intents["intent_id"], intents["underlying"], strict=False)) if not intents.empty else {}
        kind_by_intent = dict(zip(intents["intent_id"], intents["kind"], strict=False)) if not intents.empty else {}

        records: list[dict[str, Any]] = []
        for position_id, group in fills.groupby("position_id", sort=True):
            group = group.sort_values("seq")
            opens = group[group["purpose"] == "open"]
            closes = group[group["purpose"].isin(["close", "kill"])]
            if opens.empty:
                continue
            first_open = opens.iloc[0]
            last_close = None if closes.empty else closes.iloc[-1]
            closed = last_close is not None
            entry_session = first_open["session"]
            exit_session = None if last_close is None else last_close["session"]
            intent_id = first_open["intent_id"] if last_close is None else last_close["intent_id"]
            record: dict[str, Any] = {
                "position_id": position_id,
                "underlying": underlying_by_intent.get(first_open["intent_id"]),
                "kind": kind_by_intent.get(first_open["intent_id"]),
                "qty": int(first_open["qty"] or 0),
                "entry_session": entry_session,
                "exit_session": exit_session,
                "holding_sessions": (
                    None if exit_session is None else session_index.get(exit_session, 0) - session_index.get(entry_session, 0)
                ),
                "exit_reason": reason_by_intent.get(intent_id) if closed else None,
                "forced": bool(closes["forced"].any()) if closed else False,
                "fees_micro": int(group["fees_micro"].fillna(0).sum()),
                "closed": closed,
            }
            for band in BANDS:
                record[f"open_net_{band}"] = first_open[f"net_{band}"]
                record[f"close_net_{band}"] = None if last_close is None else last_close[f"net_{band}"]
                record[f"pnl_{band}"] = float(closes[f"realised_{band}"].dropna().sum()) if closed else np.nan
            records.append(record)
        return pd.DataFrame.from_records(records)
    finally:
        if owned:
            handle.close()


# ======================================================================================================================
# Decision-process frames (12.2)
# ======================================================================================================================


def decisions_frame(store: StoreArg) -> pd.DataFrame:
    """DECISION entries (2.11). `reasons` is the rules-side abstention trail, a pure function of answers and facts.

    The per-request answers, errors and variant agreements ride along as objects so the veto-rate, cross-check and
    perturbation-flip metrics of 12.2 can be computed without a second pass over the ledger.
    """
    handle, owned = _store(store)
    try:
        records: list[dict[str, Any]] = []
        for row in handle.rows(LedgerKind.DECISION):
            payload = row["payload"]
            rules = payload.get("rules") or {}
            requests = payload.get("requests") or []
            records.append(
                {
                    "seq": row["seq"],
                    "session": row["session"],
                    "as_of": row["as_of"],
                    "decision_id": payload.get("decision_id"),
                    "kind": payload.get("kind"),
                    "subject_alias": payload.get("subject_alias"),
                    "text": payload.get("text"),
                    "tier": payload.get("tier"),
                    "underlying": rules.get("underlying"),
                    "position_id": rules.get("position_id"),
                    "action": rules.get("action"),
                    "structure_kind": rules.get("kind"),
                    "score_core_ppm": rules.get("score_core_ppm"),
                    "score_rank_ppm": rules.get("score_rank_ppm"),
                    "tier_ppm": rules.get("tier_ppm"),
                    "reasons": tuple(rules.get("reasons") or ()),
                    "first_reason": (tuple(rules.get("reasons") or ()) or (None,))[0],
                    "variant_agreement": dict(rules.get("variant_agreement") or {}),
                    "facts": dict(payload.get("facts") or {}),
                    "requests": requests,
                    "n_requests": len(requests),
                    "request_errors": tuple(
                        request.get("error") for request in requests if isinstance(request, dict) and request.get("error")
                    ),
                }
            )
        columns = (
            "seq",
            "session",
            "as_of",
            "decision_id",
            "kind",
            "subject_alias",
            "text",
            "tier",
            "underlying",
            "position_id",
            "action",
            "structure_kind",
            "score_core_ppm",
            "score_rank_ppm",
            "tier_ppm",
            "reasons",
            "first_reason",
            "variant_agreement",
            "facts",
            "requests",
            "n_requests",
            "request_errors",
        )
        return pd.DataFrame.from_records(records) if records else _empty(columns)
    finally:
        if owned:
            handle.close()


def risk_verdicts_frame(store: StoreArg) -> pd.DataFrame:
    """RISK_VERDICT entries (2.11): the post-DECISION funnel lives here.

    `intent_id is None` marks a no-intent verdict - the entry died after the DECISION but before an OrderIntent existed
    (`gate:*`, `candidate:*`, `risk:size_zero`).
    """
    handle, owned = _store(store)
    try:
        records: list[dict[str, Any]] = []
        for row in handle.rows(LedgerKind.RISK_VERDICT):
            payload = row["payload"]
            checks = payload.get("checks") or []
            summary = payload.get("candidate") or payload.get("candidate_summary") or {}
            records.append(
                {
                    "seq": row["seq"],
                    "session": row["session"],
                    "verdict_id": payload.get("verdict_id"),
                    "decision_id": payload.get("decision_id"),
                    "intent_id": payload.get("intent_id"),
                    "approved": bool(payload.get("approved")),
                    "qty_approved": payload.get("qty_approved"),
                    "reject_codes": tuple(payload.get("reject_codes") or ()),
                    "first_reject_code": (tuple(payload.get("reject_codes") or ()) or (None,))[0],
                    "failed_checks": tuple(check.get("code") for check in checks if isinstance(check, dict) and not check.get("passed")),
                    "max_loss": payload.get("max_loss"),
                    "bp_required": payload.get("bp_required"),
                    "candidate_rejects": tuple((summary or {}).get("rejects") or ()),
                    "budget_floor": (summary or {}).get("budget_floor"),
                }
            )
        columns = (
            "seq",
            "session",
            "verdict_id",
            "decision_id",
            "intent_id",
            "approved",
            "qty_approved",
            "reject_codes",
            "first_reject_code",
            "failed_checks",
            "max_loss",
            "bp_required",
            "candidate_rejects",
            "budget_floor",
        )
        return pd.DataFrame.from_records(records) if records else _empty(columns)
    finally:
        if owned:
            handle.close()


def risk_events_frame(store: StoreArg) -> pd.DataFrame:
    """RISK_EVENT entries (halts, daily-loss halts, suppressed requests, ...)."""
    return _simple_frame(store, LedgerKind.RISK_EVENT, ("type", "trigger", "detail", "counters"))


def anomalies_frame(store: StoreArg) -> pd.DataFrame:
    """ANOMALY entries (stale marks, assignment simulations, parity basis, ...)."""
    return _simple_frame(store, LedgerKind.ANOMALY, ("type", "detail"))


def kill_frame(store: StoreArg) -> pd.DataFrame:
    """KILL entries (2.11): one row per step of the kill sequence."""
    return _simple_frame(store, LedgerKind.KILL, ("event_id", "step", "trigger", "detail"))


def kill_affected_sessions(store: StoreArg) -> set[date]:
    """Sessions touched by a kill or an entry halt - the `KILL-AFFECTED WINDOWS` header flag and the report slice (12.2).

    A session counts when it carries a KILL entry, or a RISK_EVENT whose type halts entries (`halt_set`,
    `daily_loss_halt`, `requests_suppressed`). The slice exists so that a window in which the bot could not trade is
    never silently averaged into a performance number.
    """
    handle, owned = _store(store)
    try:
        sessions = {row["session"] for row in handle.rows(LedgerKind.KILL)}
        halting = {"halt_set", "daily_loss_halt", "requests_suppressed"}
        sessions |= {row["session"] for row in handle.rows(LedgerKind.RISK_EVENT) if row["payload"].get("type") in halting}
        return sessions
    finally:
        if owned:
            handle.close()


def sidecar_frame(store: StoreArg) -> pd.DataFrame:
    """The unhashed sidecar (13.4): wall-clock creation time, provenance and diagnostics per ledger seq.

    Jev usage (request counts, input tokens, latency) is read from here - never from the hashed ledger, which carries no
    wall clock, request id or token count (2.7, INV-24).
    """
    handle, owned = _store(store)
    try:
        records: list[dict[str, Any]] = []
        cursor = handle.conn.execute(
            "SELECT s.seq AS seq, l.kind AS kind, l.session AS session, s.wall_created_at AS wall_created_at,"
            " s.provenance AS provenance, s.diagnostics AS diagnostics"
            " FROM sidecar s JOIN ledger l ON l.seq = s.seq ORDER BY s.seq"
        )
        for row in cursor:
            provenance = json.loads(str(row["provenance"])) if row["provenance"] else {}
            diagnostics = json.loads(str(row["diagnostics"])) if row["diagnostics"] else {}
            records.append(
                {
                    "seq": int(row["seq"]),
                    "kind": str(row["kind"]),
                    "session": date.fromisoformat(str(row["session"])),
                    "wall_created_at": _parse_as_of(str(row["wall_created_at"])),
                    "request_id": provenance.get("request_id"),
                    "input_tokens": provenance.get("input_tokens"),
                    "latency_ms": provenance.get("latency_ms"),
                    "evidence_tier": provenance.get("evidence_tier"),
                    "data_fidelity": provenance.get("data_fidelity"),
                    "cache_hit": diagnostics.get("cache_hit"),
                    "diagnostics": diagnostics,
                }
            )
        columns = (
            "seq",
            "kind",
            "session",
            "wall_created_at",
            "request_id",
            "input_tokens",
            "latency_ms",
            "evidence_tier",
            "data_fidelity",
            "cache_hit",
            "diagnostics",
        )
        if not records:
            return _empty(columns)
        frame = pd.DataFrame.from_records(records)
        frame["input_tokens"] = frame["input_tokens"].astype("Int64")
        frame["latency_ms"] = frame["latency_ms"].astype("Int64")
        return frame
    finally:
        if owned:
            handle.close()


# ======================================================================================================================
# helpers
# ======================================================================================================================


def _simple_frame(store: StoreArg, kind: LedgerKind, keys: Sequence[str]) -> pd.DataFrame:
    handle, owned = _store(store)
    try:
        records = [
            {
                "seq": row["seq"],
                "session": row["session"],
                "as_of": row["as_of"],
                **{key: row["payload"].get(key) for key in keys},
            }
            for row in handle.rows(kind)
        ]
        return pd.DataFrame.from_records(records) if records else _empty(("seq", "session", "as_of", *keys))
    finally:
        if owned:
            handle.close()


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _ppm_to_p(ppm: Any) -> float:
    """Parts per million -> probability; a missing value is NaN, never 0 (12.1: MISSING is not "certainly no")."""
    if ppm is None:
        return float("nan")
    return float(ppm) / 1_000_000.0


def _opt_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _band_map(payload: dict[str, Any], name: str) -> dict[str, int]:
    """Read a per-band payload field written either as `{"equity": {"orats": ...}}` or as flat `equity_orats` keys.

    2.11 spells these fields as "equity per band" without fixing the JSON shape; both spellings are accepted here so a
    run store written by either convention loads, and neither is silently read as zero.
    """
    nested = payload.get(name)
    out: dict[str, int] = {}
    if isinstance(nested, dict):
        for band in BANDS:
            value = nested.get(band)
            if value is not None:
                out[band] = int(value)
        return out
    for band in BANDS:
        value = payload.get(f"{name}_{band}")
        if value is not None:
            out[band] = int(value)
    return out
