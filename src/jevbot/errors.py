"""Exception hierarchy and process exit codes (DESIGN.md section 2.9).

The tree is a frozen contract::

    JevbotError
     +- ConfigError
     +- DataError
     |    +- DataUnavailable          no snapshot / table row for the request
     |    +- PitViolation             a keyed read asked for knowable_at > as_of, or a provider returned such a row
     |    +- ManifestMismatch
     +- StateError
     |    +- StateTypeError           a non str/int/bool/None/list/dict value in state
     |    +- StateLeak                ticker / absolute date / year / price-like value detected in masked state
     |    +- StateTooLarge
     +- DeciderError                  FAIL-CLOSED class: RETURNED per request by decide_batch (never raised through it)
     |    +- DeciderTransportError    transient: connection / timeout / 429 / 5xx             (counts toward jev_fail_sessions)
     |    +- DeciderResponseError     validation errors, missing answers, unknown labels      (counts toward jev_fail_sessions)
     |    +- DeciderConfigError       401/403/400/404/422 - bad key or our bug                (halts entries + alert; never counts toward kill)
     |    +- SpendLimitError          ceiling reached (D9)
     +- CacheMissError                replay miss: ALWAYS RAISED, aborts the run (D7). Deliberately NOT a DeciderError
     +- ModelMismatchError            resp.model != pinned: ALWAYS RAISED
     +- BrokerError
     |    +- BrokerAmbiguous          outcome unknown -> lookup by client id (D18)
     |    +- BrokerRejected           definitive 4xx; attributes status, reject_code, message, tag
     |    +- PaperGuardError          any paper-only assertion failed (D2)
     +- ReconcileMismatch
     +- LedgerCorrupt
     +- InvariantError                a bug-class violation
     +- EvalError
          +- TierViolation            pooling tiers or namespaces in one number
          +- HoldoutViolation         tuning on post-release (Tier A/B) sessions
          +- PreregError

This module imports nothing from the package: every other module may import it.
"""

from enum import IntEnum
from typing import Any

__all__ = [
    "BrokerAmbiguous",
    "BrokerError",
    "BrokerRejected",
    "CacheMissError",
    "ConfigError",
    "DataError",
    "DataUnavailable",
    "DeciderConfigError",
    "DeciderError",
    "DeciderResponseError",
    "DeciderTransportError",
    "EvalError",
    "ExitCode",
    "HoldoutViolation",
    "InvariantError",
    "JevbotError",
    "LedgerCorrupt",
    "ManifestMismatch",
    "ModelMismatchError",
    "PaperGuardError",
    "PitViolation",
    "PreregError",
    "ReconcileMismatch",
    "SpendLimitError",
    "StateError",
    "StateLeak",
    "StateTooLarge",
    "StateTypeError",
    "TierViolation",
    "exit_code_for",
]


class JevbotError(Exception):
    """Root of every error the bot raises on purpose."""


# --- configuration ------------------------------------------------------------------------------------------------------


class ConfigError(JevbotError):
    """Invalid configuration, forbidden environment, bad CLI usage (exit 2)."""


# --- data ---------------------------------------------------------------------------------------------------------------


class DataError(JevbotError):
    """Data / manifest error (exit 8)."""


class DataUnavailable(DataError):
    """No snapshot / table row for the request."""


class PitViolation(DataError):
    """A keyed read asked for knowable_at > as_of, or a provider returned such a row (INV-14)."""


class ManifestMismatch(DataError):
    """A dataset no longer matches its recorded manifest."""


# --- state sent to Jev --------------------------------------------------------------------------------------------------


class StateError(JevbotError):
    """The state object failed `canon.ensure_state_safe` (INV-15)."""


class StateTypeError(StateError):
    """A non str/int/bool/None/list/dict value in state."""


class StateLeak(StateError):
    """Ticker / absolute date / year / price-like value detected in masked state."""


class StateTooLarge(StateError):
    """The serialised state exceeds `state.hard_max_chars`."""


# --- decider ------------------------------------------------------------------------------------------------------------


class DeciderError(JevbotError):
    """FAIL-CLOSED class: RETURNED per request by decide_batch (never raised through it). INV-05."""


class DeciderTransportError(DeciderError):
    """Transient: connection / timeout / 429 / 5xx (counts toward jev_fail_sessions)."""


class DeciderResponseError(DeciderError):
    """Validation errors, missing answers, unknown labels (counts toward jev_fail_sessions)."""


class DeciderConfigError(DeciderError):
    """401/403/400/404/422 - bad key or our bug (halts entries + alert; never counts toward kill)."""


class SpendLimitError(DeciderError):
    """Ceiling reached (D9).

    PAPER: returned like any DeciderError (halts entries; never counts toward kill).
    BACKTEST: RE-RAISED by decide_batch exactly like CacheMissError - the hard stop of D9: the uncommitted session is
    rolled back, the trial is marked failed:spend_limit, exit 7, resumable with --resume (3.3, 7.9).
    """


class CacheMissError(JevbotError):
    """Replay miss: ALWAYS RAISED, aborts the run (D7). Deliberately NOT a DeciderError. No config knob disables it (exit 5)."""


class ModelMismatchError(JevbotError):
    """resp.model != pinned: ALWAYS RAISED; aborts a backtest, trips the kill switch in paper (D6, D17, INV-06; exit 6)."""


# --- broker -------------------------------------------------------------------------------------------------------------


class BrokerError(JevbotError):
    """Any failure of a broker call."""


class BrokerAmbiguous(BrokerError):
    """Timeout / connection / 5xx / 429 / deadline: outcome unknown -> lookup by client id (D18)."""


class BrokerRejected(BrokerError):
    """Definitive 4xx (403/422). The order worker ledgers `status`, `reject_code`, `message` and `tag`.

    `status` is the HTTP status, `reject_code` the broker's numeric code (e.g. 40310000), `message` the broker text
    (diagnostics only, never branched on) and `tag` a coarse DIAGNOSTIC tag (critique corr. 4).
    """

    def __init__(
        self,
        message: str = "",
        *,
        status: int | None = None,
        reject_code: int | None = None,
        tag: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.reject_code = reject_code
        self.message = message
        self.tag = tag

    def __reduce__(self) -> tuple[Any, ...]:
        # keyword-only attributes must survive pickling (process pools) and copy.copy
        return (_rebuild_broker_rejected, (type(self), self.message, self.status, self.reject_code, self.tag))

    def __str__(self) -> str:
        parts = [self.message or "order rejected"]
        if self.status is not None:
            parts.append(f"status={self.status}")
        if self.reject_code is not None:
            parts.append(f"reject_code={self.reject_code}")
        if self.tag is not None:
            parts.append(f"tag={self.tag}")
        return " ".join(parts)


def _rebuild_broker_rejected(
    cls: type[BrokerRejected], message: str, status: int | None, reject_code: int | None, tag: str | None
) -> BrokerRejected:
    return cls(message, status=status, reject_code=reject_code, tag=tag)


class PaperGuardError(BrokerError):
    """Any paper-only assertion failed (D2, INV-01, INV-02; exit 4)."""


# --- run integrity ------------------------------------------------------------------------------------------------------


class ReconcileMismatch(JevbotError):
    """Residual ledger / broker difference after order catch-up (INV-10)."""


class LedgerCorrupt(JevbotError):
    """The hash chain does not verify (INV-19; exit 9)."""


class InvariantError(JevbotError):
    """A bug-class violation (e.g. undefined-risk structure reached the RiskEngine)."""


# --- evaluation ---------------------------------------------------------------------------------------------------------


class EvalError(JevbotError):
    """Evaluation / reporting error."""


class TierViolation(EvalError):
    """Pooling tiers or namespaces in one number (INV-22)."""


class HoldoutViolation(EvalError):
    """Tuning on post-release (Tier A/B) sessions."""


class PreregError(EvalError):
    """Pre-registration missing, stale, unverifiable or not evaluable."""


# --- process exit codes -------------------------------------------------------------------------------------------------


class ExitCode(IntEnum):
    """Process exit codes (2.9).

    The paper service does NOT exit on a kill or on a spend stop: on a kill it stays alive in NOT_FLAT retry or idles in
    LOCKED (9.5), still ledgering forecasts every session (10.1 step 6a).
    """

    OK = 0
    ERROR = 1  # generic error
    CONFIG = 2  # config / usage
    SAFETY_REFUSED = 3  # safety guard refused (kill active for a trading command, lock held)
    PAPER_GUARD = 4  # paper guard failure
    CACHE_MISS = 5  # replay cache miss
    MODEL_MISMATCH = 6  # model mismatch
    SPEND_LIMIT = 7  # spend limit (any batch entry point in record mode); resumable with --resume
    DATA = 8  # data / manifest error
    LEDGER_CORRUPT = 9  # ledger corrupt


# Most specific class first: the first isinstance match wins.
_EXIT_CODE_BY_CLASS: tuple[tuple[type[BaseException], ExitCode], ...] = (
    (ConfigError, ExitCode.CONFIG),
    (PaperGuardError, ExitCode.PAPER_GUARD),
    (CacheMissError, ExitCode.CACHE_MISS),
    (ModelMismatchError, ExitCode.MODEL_MISMATCH),
    (SpendLimitError, ExitCode.SPEND_LIMIT),
    (DataError, ExitCode.DATA),
    (LedgerCorrupt, ExitCode.LEDGER_CORRUPT),
)


def exit_code_for(exc: BaseException) -> ExitCode:
    """The process exit code of 2.9 for an exception that ends a command; anything unlisted is the generic 1.

    `ExitCode.SAFETY_REFUSED` (3) has no exception class: commands that refuse (kill active, lock held, `doctor --strict`
    warnings) exit with it directly.
    """
    for cls, code in _EXIT_CODE_BY_CLASS:
        if isinstance(exc, cls):
            return code
    return ExitCode.ERROR
