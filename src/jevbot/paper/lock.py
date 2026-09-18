"""One process per data directory: the `flock` single-instance guard of boot step B1 (DESIGN.md 11.2; INV-20).

Two jevbot processes sharing one `$JEVBOT_DATA` would fight over the run store, the heartbeat and - far worse - the broker:
each would reconcile the other's orders as foreign and cancel them. `acquire()` therefore takes an exclusive, non-blocking
`flock` on `$JEVBOT_DATA/state/jevbot.lock` and holds it for the lifetime of the process; a second instance raises
`LockHeld`, which the CLI reports and turns into exit 3 (`ExitCode.SAFETY_REFUSED`).

`flock` is the right primitive here: the kernel releases it when the process dies, however it dies, so a crashed runner never
leaves a stale lock behind. The lock belongs to the open file description, so a second `acquire()` in the SAME process on a
second descriptor is refused as well - the dead-man check (11.9, WP12) relies on exactly that to tell "the service is running"
from "nobody owns this data directory, flatten".

The file's contents (pid, host, an ISO-8601 UTC stamp) are diagnostics only. They are written AFTER the lock is taken and are
never trusted for the decision: a pid file races, a `flock` does not.
"""

import errno
import fcntl
import json
import os
import socket
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Final

from jevbot.errors import JevbotError

__all__ = ["LOCK_FILE_NAME", "LOCK_RELATIVE_PATH", "InstanceLock", "LockHeld", "acquire", "lock_path", "read_holder"]

LOCK_FILE_NAME: Final = "jevbot.lock"
LOCK_RELATIVE_PATH: Final = ("state", LOCK_FILE_NAME)
"""`$JEVBOT_DATA/state/jevbot.lock` (13.1)."""

_STATE_DIR_MODE: Final = 0o700
_LOCK_FILE_MODE: Final = 0o600


class LockHeld(JevbotError):
    """Another process already owns this data directory (INV-20).

    `errors.ExitCode.SAFETY_REFUSED` (3) has no exception class of its own - refusals exit with it directly - so the paper
    CLI catches this and exits 3 with `holder` in the message.
    """

    def __init__(self, path: Path, holder: dict[str, object] | None) -> None:
        self.path = path
        self.holder = holder or {}
        who = f"pid {self.holder['pid']} on {self.holder.get('host', '?')}" if "pid" in self.holder else "another process"
        super().__init__(
            f"{path} is locked by {who}: one jevbot process per data directory (INV-20). Stop it first "
            f"(`systemctl --user stop jevbot-paper`) or point $JEVBOT_DATA somewhere else."
        )


def lock_path(data_dir: Path | str) -> Path:
    """`<data_dir>/state/jevbot.lock`."""
    return Path(data_dir).joinpath(*LOCK_RELATIVE_PATH)


def read_holder(path: Path | str) -> dict[str, object] | None:
    """The diagnostics the holder wrote, or `None` when the file is absent, empty or not ours to parse.

    Never a liveness test: the `flock` is. `jevbot paper status` prints this next to the refusal.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        holder = json.loads(text)
    except json.JSONDecodeError:
        return None
    return holder if isinstance(holder, dict) else None


class InstanceLock:
    """An exclusive `flock` held for the lifetime of the process (boot step B1). Usable as a context manager."""

    def __init__(self, path: Path | str) -> None:
        self.path: Final = Path(path)
        self._fd: int | None = None

    # --- state ---------------------------------------------------------------------------------------------------------

    @property
    def held(self) -> bool:
        return self._fd is not None

    @property
    def fileno(self) -> int | None:
        """The locked descriptor, for a caller that wants to keep it across an exec; `None` while not held."""
        return self._fd

    # --- acquire / release ---------------------------------------------------------------------------------------------

    def acquire(self) -> "InstanceLock":
        """Take the lock, then stamp the file with our pid. Raises `LockHeld` when someone else has it."""
        if self._fd is not None:
            return self
        self.path.parent.mkdir(mode=_STATE_DIR_MODE, parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, _LOCK_FILE_MODE)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise LockHeld(self.path, read_holder(self.path)) from None
            raise
        self._fd = fd
        self._stamp(fd)
        return self

    @staticmethod
    def _stamp(fd: int) -> None:
        """Diagnostics only, written under the lock: pid, host, and when this owner took it."""
        payload = json.dumps(
            {"pid": os.getpid(), "host": socket.gethostname(), "acquired_at": datetime.now(UTC).isoformat()},
            separators=(",", ":"),
        )
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, payload.encode("utf-8") + b"\n")
        os.fsync(fd)

    def release(self) -> None:
        """Drop the lock. Idempotent. The file itself is left in place: unlinking it would race a second instance that
        already holds a descriptor to the same inode, and both would then believe they own the directory."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    # --- context manager -----------------------------------------------------------------------------------------------

    def __enter__(self) -> "InstanceLock":
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"InstanceLock({str(self.path)!r}, held={self.held})"


def acquire(data_dir: Path | str) -> InstanceLock:
    """Boot step B1: `acquire($JEVBOT_DATA)` -> the held lock, or `LockHeld` (exit 3)."""
    return InstanceLock(lock_path(data_dir)).acquire()
