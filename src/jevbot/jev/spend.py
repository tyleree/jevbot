"""The spend guard and the token bucket (DESIGN.md 3.3, 6.7, 13.5; INV-17, D9).

`SpendGuard` is the `SpendLedger` of 3.3 over `$JEVBOT_DATA/state/spend.sqlite`, the ONE file every entry point shares. It
reserves the worst case **before** each HTTP attempt and commits the real `usage.input_tokens` after it, so a retried (and
therefore billed) attempt is always metered.

INV-17, literally: counters and the sticky block are **per scope**.

* `"paper"` - the paper service; ceiling `jev.spend.paper_max_input_tokens_per_day`, no per-run ceiling (the service's run
  lives for months);
* `"batch"` - backtests, probes, leakage runs, baselines; ceilings `jev.spend.max_input_tokens_per_run` and
  `jev.spend.max_input_tokens_per_day`.

A guard instance is bound to one scope, so a batch run that hits its ceiling can never halt the paper service's entries. Once
a **day** ceiling is reached the scope is blocked for the rest of the UTC day (`spend_block`, sticky and persisted, so a
restart does not un-block it); a **run** ceiling stops that run only. Either way `reserve` raises `SpendLimitError`, which is
a `DeciderError`: paper halts entries and lives on, a backtest stops with exit 7 and resumes on the next UTC day - free for
everything already cached (6.7).

Accounting is conservative: a reservation that was never committed counts at its reserved size (the attempt may well have
been billed), and a failed attempt is committed with `estimated=True` at its estimate.

`TokenBucket` is the client-side rate limiter of `jev.max_rps` / `jev.max_tokens_per_s`; it is what the design calls
`TokenBucket.acquire(estimate)` in 6.8 step 3, and it never crosses a process boundary (the persisted ceilings do).
"""

import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from jevbot.config import JevSpendConfig
from jevbot.errors import ConfigError, InvariantError, SpendLimitError

__all__ = ["SCOPES", "SCOPE_BATCH", "SCOPE_PAPER", "SPEND_SCHEMA_SQL", "SpendGuard", "TokenBucket"]

SCOPE_PAPER: Final = "paper"
SCOPE_BATCH: Final = "batch"
SCOPES: Final[tuple[str, ...]] = (SCOPE_PAPER, SCOPE_BATCH)

SPEND_SCHEMA_SQL: Final = """
CREATE TABLE IF NOT EXISTS spend (id INTEGER PRIMARY KEY, ts TEXT NOT NULL, utc_day TEXT NOT NULL, scope TEXT NOT NULL,
  run_id TEXT NOT NULL, reserved INTEGER NOT NULL, committed INTEGER, estimated INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS spend_by_day ON spend(scope, utc_day);
CREATE INDEX IF NOT EXISTS spend_by_run ON spend(scope, run_id);
CREATE TABLE IF NOT EXISTS spend_block (utc_day TEXT NOT NULL, scope TEXT NOT NULL, reason TEXT NOT NULL, at TEXT NOT NULL,
  PRIMARY KEY (utc_day, scope));
"""

# tokens accounted to a row: the committed value once it is known, the reservation until then (conservative)
_ACCOUNTED: Final = "COALESCE(committed, reserved)"


def _utc_now() -> datetime:
    return datetime.now(UTC)


class TokenBucket:
    """A thread-safe token bucket: `rate_per_s` tokens per second, bursting up to `capacity` (default: one second).

    `acquire(n)` returns the seconds it waited. A request larger than the capacity is served at the capacity (it waits for a
    full bucket) instead of deadlocking. `monotonic` / `sleep` are injectable so tests measure the wait without spending it.
    """

    def __init__(
        self,
        rate_per_s: float,
        capacity: float | None = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not rate_per_s > 0:
            raise ConfigError(f"TokenBucket needs a positive rate, got {rate_per_s}")
        self.rate_per_s = float(rate_per_s)
        self.capacity = float(self.rate_per_s if capacity is None else capacity)
        if not self.capacity > 0:
            raise ConfigError(f"TokenBucket needs a positive capacity, got {capacity}")
        self._monotonic = monotonic
        self._sleep = sleep
        self._lock = threading.Lock()
        self._tokens = self.capacity
        self._updated = monotonic()

    def acquire(self, tokens: int) -> float:
        """Take `tokens` from the bucket, waiting if it is not full enough; returns the seconds waited."""
        need = min(max(float(tokens), 0.0), self.capacity)
        with self._lock:
            now = self._monotonic()
            self._tokens = min(self.capacity, self._tokens + max(0.0, now - self._updated) * self.rate_per_s)
            self._updated = now
            if self._tokens >= need:
                self._tokens -= need
                return 0.0
            wait = (need - self._tokens) / self.rate_per_s
            self._tokens = 0.0
            self._updated = now + wait  # the wait is booked here, so concurrent callers queue instead of racing
        self._sleep(wait)
        return wait


class SpendGuard:
    """`SpendLedger` (3.3) over `state/spend.sqlite`, bound to ONE scope (`"paper"` or `"batch"`; INV-17)."""

    def __init__(self, path: Path | str, *, scope: str, cfg: JevSpendConfig, now: Callable[[], datetime] = _utc_now) -> None:
        if scope not in SCOPES:
            raise ConfigError(f"spend scope must be one of {SCOPES}, got {scope!r}")
        self.path = Path(path)
        self._scope = scope
        self.cfg = cfg
        self._now = now
        self._lock = threading.RLock()
        self._run_blocked: set[str] = set()  # runs that hit the per-run ceiling (the day block is persisted instead)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
        self._db.row_factory = sqlite3.Row
        self._db.isolation_level = None
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        with self._lock:
            self._db.executescript(SPEND_SCHEMA_SQL)

    @property
    def scope(self) -> str:
        return self._scope

    @property
    def day_ceiling(self) -> int:
        """This scope's UTC-day ceiling (`paper_max_input_tokens_per_day` / `max_input_tokens_per_day`)."""
        return self.cfg.paper_max_input_tokens_per_day if self._scope == SCOPE_PAPER else self.cfg.max_input_tokens_per_day

    @property
    def run_ceiling(self) -> int | None:
        """The per-run ceiling - batch scope only; the paper service's run has none (6.7)."""
        return None if self._scope == SCOPE_PAPER else self.cfg.max_input_tokens_per_run

    def utc_day(self) -> str:
        return self._now().astimezone(UTC).date().isoformat()

    # ------------------------------------------------------------------ the guard

    def reserve(self, run_id: str, tokens: int) -> int:
        """Book `tokens` for `run_id` BEFORE the HTTP attempt; returns the reservation id.

        Raises `SpendLimitError` - without booking anything and therefore before any HTTP call - when this scope is already
        blocked for the UTC day, when the run total would exceed the per-run ceiling (batch scope), or when the scope's
        UTC-day total would exceed its day ceiling. The day case writes the sticky `spend_block` row first.
        """
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
            raise InvariantError(f"reserve() takes a non-negative integer token estimate, got {tokens!r}")
        if not run_id:
            raise InvariantError("reserve() needs a run id")
        day = self.utc_day()
        with self._lock:
            if self._is_blocked(day):
                raise SpendLimitError(
                    f"the {self._scope} spend scope is blocked for the UTC day {day}: its "
                    f"{self.day_ceiling} input-token ceiling was reached (INV-17)"
                )
            if run_id in self._run_blocked:
                raise SpendLimitError(f"run {run_id} reached its {self.run_ceiling} input-token ceiling (D9)")
            run_total, day_total = self._totals(run_id, day)
            ceiling = self.run_ceiling
            if ceiling is not None and run_total + tokens > ceiling:
                self._run_blocked.add(run_id)
                raise SpendLimitError(
                    f"run {run_id} would exceed its input-token ceiling: {run_total} + {tokens} > {ceiling} (D9; resume with --resume)"
                )
            if day_total + tokens > self.day_ceiling:
                self._block(day, f"day ceiling {self.day_ceiling} reached ({day_total} + {tokens})")
                raise SpendLimitError(
                    f"the {self._scope} scope would exceed its UTC-day input-token ceiling: "
                    f"{day_total} + {tokens} > {self.day_ceiling}; it is blocked for the rest of {day} (INV-17)"
                )
            cursor = self._db.execute(
                "INSERT INTO spend (ts, utc_day, scope, run_id, reserved, committed, estimated) VALUES (?, ?, ?, ?, ?, NULL, 0)",
                (self._now().astimezone(UTC).isoformat(), day, self._scope, run_id, tokens),
            )
            reservation_id = cursor.lastrowid
            if reservation_id is None:  # pragma: no cover - sqlite3 always reports a rowid for this table
                raise InvariantError("spend: the reservation row has no id")
            return int(reservation_id)

    def commit(self, reservation_id: int, tokens: int, *, estimated: bool) -> None:
        """Settle a reservation with the real `usage.input_tokens` (or the estimate again, `estimated=True`, after a failure).

        Committing a settled or unknown reservation is a bug (`InvariantError`). When the settled total crosses the day
        ceiling the scope is blocked at once - the next `reserve` then fails before any HTTP call.
        """
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
            raise InvariantError(f"commit() takes a non-negative integer token count, got {tokens!r}")
        with self._lock:
            row = self._db.execute("SELECT utc_day, scope, committed FROM spend WHERE id = ?", (reservation_id,)).fetchone()
            if row is None or row["scope"] != self._scope:
                raise InvariantError(f"spend: reservation {reservation_id} does not belong to the {self._scope} scope")
            if row["committed"] is not None:
                raise InvariantError(f"spend: reservation {reservation_id} was already committed")
            self._db.execute("UPDATE spend SET committed = ?, estimated = ? WHERE id = ?", (tokens, int(bool(estimated)), reservation_id))
            day = str(row["utc_day"])
            if self._day_total(day) > self.day_ceiling and not self._is_blocked(day):
                self._block(day, f"day ceiling {self.day_ceiling} passed by the committed totals")

    def totals(self, run_id: str | None = None) -> tuple[int, int]:
        """`(tokens accounted to run_id, this scope's UTC-day total across all its runs)`; 0 for the run when `run_id` is None."""
        with self._lock:
            return self._totals(run_id, self.utc_day())

    def blocked(self) -> bool:
        """True once this scope hit its day ceiling; sticky for the rest of the UTC day, across processes."""
        with self._lock:
            return self._is_blocked(self.utc_day())

    def block_reason(self) -> str | None:
        """Why this scope is blocked today, or None."""
        with self._lock:
            row = self._db.execute(
                "SELECT reason FROM spend_block WHERE utc_day = ? AND scope = ?", (self.utc_day(), self._scope)
            ).fetchone()
        return str(row["reason"]) if row is not None else None

    def by_day(self, limit: int = 14) -> dict[str, int]:
        """This scope's accounted tokens per UTC day, newest first (what `cache stats` prints as per-day spend)."""
        with self._lock:
            cursor = self._db.execute(
                f"SELECT utc_day, COALESCE(SUM({_ACCOUNTED}), 0) AS tokens FROM spend WHERE scope = ? "
                "GROUP BY utc_day ORDER BY utc_day DESC LIMIT ?",
                (self._scope, int(limit)),
            )
            return {str(row["utc_day"]): int(row["tokens"]) for row in cursor.fetchall()}

    def reservations(self, run_id: str | None = None) -> int:
        """How many reservations this scope booked (for `run_id` when given) - the per-attempt metering counter."""
        with self._lock:
            if run_id is None:
                return int(self._db.execute("SELECT COUNT(*) FROM spend WHERE scope = ?", (self._scope,)).fetchone()[0])
            return int(self._db.execute("SELECT COUNT(*) FROM spend WHERE scope = ? AND run_id = ?", (self._scope, run_id)).fetchone()[0])

    # ------------------------------------------------------------------ plumbing

    def _totals(self, run_id: str | None, day: str) -> tuple[int, int]:
        run_total = 0
        if run_id is not None:
            run_total = int(
                self._db.execute(
                    f"SELECT COALESCE(SUM({_ACCOUNTED}), 0) FROM spend WHERE scope = ? AND run_id = ?", (self._scope, run_id)
                ).fetchone()[0]
            )
        return run_total, self._day_total(day)

    def _day_total(self, day: str) -> int:
        return int(
            self._db.execute(
                f"SELECT COALESCE(SUM({_ACCOUNTED}), 0) FROM spend WHERE scope = ? AND utc_day = ?", (self._scope, day)
            ).fetchone()[0]
        )

    def _is_blocked(self, day: str) -> bool:
        row = self._db.execute("SELECT 1 FROM spend_block WHERE utc_day = ? AND scope = ?", (day, self._scope)).fetchone()
        return row is not None

    def _block(self, day: str, reason: str) -> None:
        self._db.execute(
            "INSERT OR IGNORE INTO spend_block (utc_day, scope, reason, at) VALUES (?, ?, ?, ?)",
            (day, self._scope, reason, self._now().astimezone(UTC).isoformat()),
        )

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> "SpendGuard":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
