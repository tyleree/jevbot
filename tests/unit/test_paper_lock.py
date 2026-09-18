"""`paper/lock.py`: one process per data directory (DESIGN 11.2 step B1; INV-20).

Exclusivity is proven twice: within one process (two open file descriptions on the same path, which is what the dead-man
check relies on) and across processes (a real child interpreter, which is the case that actually matters).
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from jevbot.paper.lock import LOCK_FILE_NAME, InstanceLock, LockHeld, acquire, lock_path, read_holder

CHILD = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {repo!r})
    from jevbot.paper.lock import LockHeld, acquire
    try:
        lock = acquire({data_dir!r})
    except LockHeld as exc:
        print("HELD", exc)
        raise SystemExit(3)
    print("TOOK")
    raise SystemExit(0)
    """
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_child(data_dir: Path) -> subprocess.CompletedProcess[str]:
    """A real second instance: `flock` is per open file description, so only a separate process proves cross-process."""
    source = CHILD.format(repo=str(REPO_ROOT / "src"), data_dir=str(data_dir))
    return subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=60, check=False)


# ======================================================================================================================
# Paths and the file itself
# ======================================================================================================================


def test_the_lock_lives_where_section_13_1_puts_it(tmp_path: Path) -> None:
    assert lock_path(tmp_path) == tmp_path / "state" / LOCK_FILE_NAME
    assert LOCK_FILE_NAME == "jevbot.lock"


def test_acquire_creates_the_state_directory_private(tmp_path: Path) -> None:
    lock = acquire(tmp_path)
    try:
        state = tmp_path / "state"
        assert state.is_dir()
        assert (os.stat(state).st_mode & 0o077) == 0  # no group / other access (13.1)
        assert lock.path.exists()
    finally:
        lock.release()


def test_the_holder_stamp_is_written_under_the_lock(tmp_path: Path) -> None:
    with acquire(tmp_path) as lock:
        holder = read_holder(lock.path)
        assert holder is not None
        assert holder["pid"] == os.getpid()
        assert isinstance(holder["host"], str) and holder["host"]
        assert isinstance(holder["acquired_at"], str) and holder["acquired_at"].endswith("+00:00")


def test_read_holder_survives_an_absent_or_corrupt_file(tmp_path: Path) -> None:
    assert read_holder(tmp_path / "nothing-here") is None
    broken = tmp_path / "broken.lock"
    broken.write_text("not json at all", encoding="utf-8")
    assert read_holder(broken) is None
    listy = tmp_path / "listy.lock"
    listy.write_text("[1, 2]", encoding="utf-8")
    assert read_holder(listy) is None


# ======================================================================================================================
# Exclusivity
# ======================================================================================================================


def test_a_second_lock_object_in_this_process_is_refused(tmp_path: Path) -> None:
    first = acquire(tmp_path)
    try:
        second = InstanceLock(lock_path(tmp_path))
        with pytest.raises(LockHeld) as caught:
            second.acquire()
        assert second.held is False
        assert caught.value.holder["pid"] == os.getpid()
        assert "INV-20" in str(caught.value)
    finally:
        first.release()


def test_the_lock_is_free_again_after_release(tmp_path: Path) -> None:
    acquire(tmp_path).release()
    again = acquire(tmp_path)
    assert again.held is True
    again.release()


def test_release_is_idempotent_and_the_context_manager_releases(tmp_path: Path) -> None:
    lock = acquire(tmp_path)
    lock.release()
    lock.release()
    assert lock.held is False and lock.fileno is None

    with InstanceLock(lock_path(tmp_path)) as held:
        assert held.held is True and isinstance(held.fileno, int)
    assert held.held is False


def test_acquiring_twice_on_the_same_object_is_a_no_op(tmp_path: Path) -> None:
    lock = acquire(tmp_path)
    try:
        fd = lock.fileno
        assert lock.acquire() is lock
        assert lock.fileno == fd
    finally:
        lock.release()


def test_a_second_process_is_refused_while_we_hold_it(tmp_path: Path) -> None:
    lock = acquire(tmp_path)
    try:
        result = run_child(tmp_path)
    finally:
        lock.release()

    assert result.returncode == 3, result.stderr
    assert result.stdout.startswith("HELD")
    assert str(os.getpid()) in result.stdout  # the refusal names who holds it


def test_a_second_process_takes_the_lock_once_we_let_go(tmp_path: Path) -> None:
    acquire(tmp_path).release()

    result = run_child(tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "TOOK"
    # the child died holding the lock; the kernel released it, so no stale lock is left behind
    holder = read_holder(lock_path(tmp_path))
    assert holder is not None and holder["pid"] != os.getpid()
    reclaimed = acquire(tmp_path)
    assert reclaimed.held is True
    reclaimed.release()


def test_the_stamp_is_replaced_not_appended(tmp_path: Path) -> None:
    acquire(tmp_path).release()
    run_child(tmp_path)  # a second owner writes its own pid over the first one's

    text = lock_path(tmp_path).read_text(encoding="utf-8")
    assert text.count("\n") == 1
    assert isinstance(json.loads(text), dict)


def test_repr_says_whether_it_is_held(tmp_path: Path) -> None:
    lock = InstanceLock(lock_path(tmp_path))
    assert "held=False" in repr(lock)
    lock.acquire()
    try:
        assert "held=True" in repr(lock)
    finally:
        lock.release()
