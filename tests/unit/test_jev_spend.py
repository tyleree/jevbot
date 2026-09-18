"""`jev/spend.py`: the per-scope spend guard and the token bucket (DESIGN.md 3.3, 13.5; INV-17, D9).

The guard is the only thing standing between a scripted backtest and an unbounded bill, so the tests pin the exact
behaviour INV-17 promises: a reservation per HTTP attempt, per-run and per-day ceilings shared across instances and
processes, a UTC-day rollover that clears the counters, a block that is sticky for the rest of the day - and, above all,
**two independent scopes**: a batch stop can never halt the paper service.
"""

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from jevbot.config import JevSpendConfig
from jevbot.errors import ConfigError, InvariantError, SpendLimitError
from jevbot.jev.spend import SCOPE_BATCH, SCOPE_PAPER, SpendGuard, TokenBucket

# the day ceiling binds first here; RUN_TIGHT is the mirror image (a run ceiling below the day ceiling)
SMALL = JevSpendConfig(max_input_tokens_per_run=2000, max_input_tokens_per_day=1500, paper_max_input_tokens_per_day=800)
RUN_TIGHT = JevSpendConfig(max_input_tokens_per_run=1000, max_input_tokens_per_day=10_000, paper_max_input_tokens_per_day=800)


class _Clock:
    """A settable UTC clock (no wall clock in a test that asserts on UTC days)."""

    def __init__(self, at: datetime) -> None:
        self.at = at

    def __call__(self) -> datetime:
        return self.at


def guard(tmp_path: Path, scope: str = SCOPE_BATCH, *, cfg: JevSpendConfig = SMALL, clock: _Clock | None = None) -> SpendGuard:
    return SpendGuard(tmp_path / "state" / "spend.sqlite", scope=scope, cfg=cfg, now=clock or _Clock(datetime(2026, 9, 17, 12, tzinfo=UTC)))


# ======================================================================================================================
# Reservations
# ======================================================================================================================


def test_a_reservation_counts_before_it_is_committed(tmp_path: Path) -> None:
    spend = guard(tmp_path)
    reservation = spend.reserve("run-1", 400)
    assert spend.totals("run-1") == (400, 400), "an uncommitted reservation counts at its reserved size (conservative)"
    spend.commit(reservation, 250, estimated=False)
    assert spend.totals("run-1") == (250, 250)
    assert spend.reservations("run-1") == 1 and spend.reservations() == 1
    spend.close()


def test_every_attempt_is_metered_and_a_failed_attempt_commits_the_estimate(tmp_path: Path) -> None:
    spend = guard(tmp_path)
    for _ in range(3):  # three HTTP attempts of one logical request (6.8 step 3)
        spend.commit(spend.reserve("run-1", 200), 200, estimated=True)
    assert spend.totals("run-1") == (600, 600) and spend.reservations("run-1") == 3
    spend.close()


def test_committing_twice_or_an_unknown_reservation_is_a_bug(tmp_path: Path) -> None:
    spend = guard(tmp_path)
    reservation = spend.reserve("run-1", 10)
    spend.commit(reservation, 10, estimated=False)
    with pytest.raises(InvariantError, match="already committed"):
        spend.commit(reservation, 10, estimated=False)
    with pytest.raises(InvariantError, match="does not belong"):
        spend.commit(reservation + 999, 10, estimated=False)
    with pytest.raises(InvariantError, match="non-negative"):
        spend.reserve("run-1", -1)
    with pytest.raises(InvariantError, match="run id"):
        spend.reserve("", 1)
    spend.close()


# ======================================================================================================================
# Ceilings
# ======================================================================================================================


def test_the_per_run_ceiling_stops_that_run_only(tmp_path: Path) -> None:
    spend = guard(tmp_path, cfg=RUN_TIGHT)
    spend.commit(spend.reserve("run-1", 900), 900, estimated=False)
    with pytest.raises(SpendLimitError, match="run-1 would exceed"):
        spend.reserve("run-1", 200)  # 900 + 200 > 1000
    assert spend.totals("run-1") == (900, 900), "a refused reservation books nothing"
    assert not spend.blocked(), "a per-run ceiling is not a day block"
    other = spend.reserve("run-2", 400)  # a different run of the same scope may continue
    spend.commit(other, 400, estimated=False)
    assert spend.totals("run-2") == (400, 1300)  # the DAY total carries both runs
    spend.close()


def test_the_day_ceiling_blocks_the_scope_stickily(tmp_path: Path) -> None:
    spend = guard(tmp_path)
    spend.commit(spend.reserve("run-1", 900), 900, estimated=False)
    spend.commit(spend.reserve("run-2", 500), 500, estimated=False)
    assert spend.totals()[1] == 1400 and not spend.blocked()
    with pytest.raises(SpendLimitError, match="UTC-day"):
        spend.reserve("run-3", 200)  # 1400 + 200 > 1500
    assert spend.blocked() and "day ceiling" in (spend.block_reason() or "")
    with pytest.raises(SpendLimitError, match="blocked for the UTC day"):
        spend.reserve("run-3", 1)  # sticky: even a one-token request is refused
    spend.close()


def test_the_block_survives_a_new_guard_instance_and_a_new_process(tmp_path: Path) -> None:
    clock = _Clock(datetime(2026, 9, 17, 12, tzinfo=UTC))
    spend = guard(tmp_path, clock=clock)
    spend.commit(spend.reserve("run-1", 1400), 1400, estimated=False)
    with pytest.raises(SpendLimitError):
        spend.reserve("run-1", 200)
    spend.close()
    again = guard(tmp_path, clock=_Clock(clock.at))
    assert again.blocked() and again.totals("run-1") == (1400, 1400)
    again.close()
    # a second PROCESS sees the same counters and the same block (the file is the shared ledger of INV-17)
    script = (
        "import datetime, sys;"
        "from jevbot.config import JevSpendConfig;"
        "from jevbot.jev.spend import SpendGuard;"
        f"g = SpendGuard({str(tmp_path / 'state' / 'spend.sqlite')!r}, scope='batch',"
        " cfg=JevSpendConfig(max_input_tokens_per_run=2000, max_input_tokens_per_day=1500, paper_max_input_tokens_per_day=800),"
        " now=lambda: datetime.datetime(2026, 9, 17, 12, tzinfo=datetime.UTC));"
        "print(g.blocked(), g.totals('run-1')[1])"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env)
    assert out.stdout.strip() == "True 1400"


def test_a_utc_day_rollover_clears_the_counters_and_the_block(tmp_path: Path) -> None:
    clock = _Clock(datetime(2026, 9, 17, 23, 30, tzinfo=UTC))
    spend = guard(tmp_path, clock=clock)
    spend.commit(spend.reserve("run-1", 1400), 1400, estimated=False)
    with pytest.raises(SpendLimitError):
        spend.reserve("run-1", 200)
    assert spend.blocked()
    clock.at += timedelta(hours=1)  # 2026-09-18 00:30 UTC
    assert not spend.blocked(), "the block is per UTC day"
    assert spend.totals("run-1") == (1400, 0), "the run total is cumulative; the day total starts again"
    with pytest.raises(SpendLimitError, match="would exceed its input-token ceiling"):
        spend.reserve("run-1", 700)  # 1400 + 700 > 2000: the RUN ceiling still holds across the day boundary
    spend.commit(spend.reserve("run-2", 200), 200, estimated=False)
    assert spend.by_day() == {"2026-09-18": 200, "2026-09-17": 1400}
    spend.close()


def test_a_committed_overrun_blocks_the_scope_immediately(tmp_path: Path) -> None:
    spend = guard(tmp_path)
    reservation = spend.reserve("run-1", 100)
    spend.commit(reservation, 1600, estimated=False)  # the real bill was far above the estimate
    assert spend.blocked(), "the guard blocks as soon as the settled totals cross the ceiling"
    spend.close()


# ======================================================================================================================
# Scopes (INV-17)
# ======================================================================================================================


def test_a_batch_block_does_not_block_the_paper_scope(tmp_path: Path) -> None:
    batch = guard(tmp_path, SCOPE_BATCH)
    paper = guard(tmp_path, SCOPE_PAPER)
    assert (batch.scope, paper.scope) == ("batch", "paper")
    assert (batch.day_ceiling, batch.run_ceiling) == (1500, 2000)
    assert (paper.day_ceiling, paper.run_ceiling) == (800, None), "the paper service's run has no per-run ceiling"
    batch.commit(batch.reserve("backtest", 1400), 1400, estimated=False)
    with pytest.raises(SpendLimitError):
        batch.reserve("backtest", 500)
    assert batch.blocked() and not paper.blocked()
    paper.commit(paper.reserve("paper-exp001", 700), 700, estimated=False)  # the service keeps trading
    assert paper.totals("paper-exp001") == (700, 700) and batch.totals("backtest") == (1400, 1400)
    with pytest.raises(SpendLimitError, match="paper scope"):
        paper.reserve("paper-exp001", 200)  # 700 + 200 > 800
    assert paper.blocked()
    batch.close()
    paper.close()


def test_an_unknown_scope_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="spend scope"):
        SpendGuard(tmp_path / "spend.sqlite", scope="probe", cfg=SMALL)


# ======================================================================================================================
# TokenBucket
# ======================================================================================================================


class _FakeTime:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_the_token_bucket_only_waits_when_it_has_to() -> None:
    clock = _FakeTime()
    bucket = TokenBucket(100.0, monotonic=clock.monotonic, sleep=clock.sleep)
    assert bucket.acquire(60) == 0.0  # the bucket starts full (capacity = one second of rate)
    assert bucket.acquire(40) == 0.0
    assert clock.slept == []
    assert bucket.acquire(50) == pytest.approx(0.5)  # 50 tokens at 100/s
    assert clock.slept == [pytest.approx(0.5)]
    clock.now += 1.0  # a second of idling refills the bucket, but never past its capacity
    assert bucket.acquire(100) == 0.0
    assert bucket.acquire(1) == pytest.approx(0.01)


def test_the_token_bucket_clamps_an_oversized_request_instead_of_deadlocking() -> None:
    clock = _FakeTime()
    bucket = TokenBucket(10.0, monotonic=clock.monotonic, sleep=clock.sleep)
    assert bucket.acquire(10_000) == 0.0  # clamped to the capacity: it waits for a full bucket, not forever
    assert bucket.acquire(10_000) == pytest.approx(1.0)
    assert bucket.acquire(0) == 0.0


def test_the_token_bucket_refuses_a_non_positive_rate_or_capacity() -> None:
    with pytest.raises(ConfigError, match="positive rate"):
        TokenBucket(0)
    with pytest.raises(ConfigError, match="positive capacity"):
        TokenBucket(10, 0)
