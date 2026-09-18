"""LiveJev - the ONLY runtime module besides `jev/probe.py` that imports `typesafe_sdk` (DESIGN.md 6.8; D6, D7).

The SDK stays behind the `Decider` protocol, and the import itself is **lazy and guarded**: `config.check_sdk_log_level` runs
BEFORE `import typesafe_sdk`, because the SDK applies `TYPESAFE_LOG_LEVEL` once at import and logs full request and response
**bodies** at DEBUG (INV-18). A `debug` / `info` / typo value is a `ConfigError` and the SDK is never imported at all; after
the import its logger is pinned to WARNING, so an application-level `--log-level debug` can never surface a body
(`tests/guards/test_no_bodies_logged.py`).

`decide(req)` follows 6.8 step by step:

1. one content cache key per question of the FULL batch (`canon.cache_key`, D8);
2. all keys present in the run's namespace => `source = "cache"`, **zero HTTP calls**, the model still verified (INV-06);
3. any key missing => the **full batch** is sent, never only the missing questions: batch composition is part of the
   experiment (D8 / G3). Our own retry loop books a spend reservation per HTTP ATTEMPT, so a retried - and therefore billed -
   attempt is always metered, and the reservation is taken BEFORE the call (INV-17: a spend stop blocks before any HTTP);
4. `resp.model != cfg.model` => `ModelMismatchError` and **nothing is cached** (INV-06);
5. the answers are validated (`stats.to_answers`) before anything is written;
6. one cache transaction, then the result is built from the just-stored rows - the same code path as a hit;
7. `request_id` comes from the header (`resp.request_id` raises when the header is absent).

The SDK is always called with `RetryPolicy(max_retries=0)`: OUR loop is the only retry, so every attempt is metered and the
back-off is ours. `extra_body` / `extra_headers` are never used and the model is passed explicitly on every call (never the
`jev-latest` default).

Only the exception class, the HTTP status and the request id are ever logged - never a body. The `detail` of a 422 is kept in
`last_response_detail` for the caller's ledger sidecar (6.8) and is never logged.
"""

import logging
import math
import os
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

import msgspec

from jevbot import canon, config
from jevbot.config import JevConfig
from jevbot.errors import (
    ConfigError,
    DeciderConfigError,
    DeciderError,
    DeciderResponseError,
    DeciderTransportError,
    InvariantError,
    ModelMismatchError,
)
from jevbot.jev.common import cache_keys_for, check_request_hashes, result_from_rows
from jevbot.jev.spend import TokenBucket
from jevbot.jev.stats import to_answers
from jevbot.protocols import DecisionCache, SpendLedger
from jevbot.types import CachedAnswer, CacheMode, DecisionRequest, DecisionResult

if TYPE_CHECKING:  # the SDK is imported lazily inside __init__ (INV-18), so its types are only names here
    import httpx2

__all__ = ["LIVE_NAME", "REQUEST_ID_HEADER", "LiveJev"]

_log = logging.getLogger(__name__)

LIVE_NAME: Final = "live_jev"
REQUEST_ID_HEADER: Final = "x-typesafe-request-id"
_MAX_RETRY_AFTER_S: Final = 5.0  # `retry_after_ms` is honoured ONCE and capped here (6.8 step 3)
_MAX_DETAIL_CHARS: Final = 500


class LiveJev:
    """The live `Decider` (`name = "live_jev"`), bound to one namespace-model, one run id and one spend scope."""

    def __init__(
        self,
        cfg: JevConfig,
        cache: DecisionCache,
        spend: SpendLedger,
        *,
        api_key: str,
        run_id: str,
        mode: CacheMode,
        transport: "httpx2.BaseTransport | None" = None,
        sleep: Any = time.sleep,
    ) -> None:
        # (1) a blank key is never passed to an SDK (D29)
        if not isinstance(api_key, str) or not api_key.strip():
            raise ConfigError("LiveJev needs a non-empty TYPESAFE_API_KEY: an empty key is never sent to the SDK")
        if mode is CacheMode.REPLAY:
            raise ConfigError('cache mode "replay" never constructs a network client: use ReplayJev (6.8)')
        # (2) INV-18: refuse a body-logging level BEFORE the SDK is imported - it reads the variable once, at import
        config.check_sdk_log_level(os.environ)
        # (3) the lazy, guarded import
        import typesafe_sdk

        if typesafe_sdk.__version__ != cfg.sdk_version:
            raise ConfigError(
                f"typesafe_sdk {typesafe_sdk.__version__} is installed but the configuration pins {cfg.sdk_version}: "
                "Step 0 records and cached answers are keyed by the SDK version (6.8)"
            )
        logging.getLogger("typesafe_sdk").setLevel(logging.WARNING)
        # 5.9: the wire order we hash must be the order msgspec encodes (the SDK serialises with msgspec.json.encode)
        if msgspec.json.encode({"b": 1, "a": 2}) != b'{"b":1,"a":2}':  # pragma: no cover - a msgspec regression
            raise InvariantError("msgspec.json.encode no longer preserves insertion order: the D8 key-order variants would collide")
        # (4) our retry loop is the only retry: the SDK never retries, so every billed attempt is metered
        self._client = typesafe_sdk.TypeSafeClient(
            api_key=api_key,
            model=cfg.model,
            timeout=cfg.timeout_s,
            retry=typesafe_sdk.RetryPolicy(max_retries=0),
            transport=transport,
        )
        self.cfg = cfg
        self.cache = cache
        self.spend = spend
        self.run_id = run_id
        self.mode = mode
        self.last_response_detail: str | None = None  # the 422 `detail`, for the caller's sidecar; NEVER logged
        self._sleep = sleep
        self._lock = threading.Lock()
        self._requests = TokenBucket(cfg.max_rps)
        self._tokens = TokenBucket(cfg.max_tokens_per_s)
        self._errors = _ErrorMap(typesafe_sdk)

    @property
    def name(self) -> str:
        return LIVE_NAME

    @property
    def model(self) -> str:
        return self.cfg.model

    # ------------------------------------------------------------------ the decision

    def decide(self, req: DecisionRequest) -> DecisionResult:
        """Answer one request from the cache, or send the full batch and cache it (6.8). Thread-safe."""
        check_request_hashes(req)
        keys = cache_keys_for(self.cfg.model, req)
        rows = self.cache.get_many(req.namespace, list(keys.values()))
        if len(rows) == len(keys) and all(key in rows for key in keys.values()):
            return result_from_rows(req, self.cfg.model, keys, rows, source="cache")  # zero HTTP calls
        resp, request_id, input_tokens, latency_ms = self._send(req)
        if resp.model != self.cfg.model:
            raise ModelMismatchError(
                f"the response was produced by model {resp.model!r}, not the pinned {self.cfg.model!r}: nothing is cached (INV-06)"
            )
        body = resp.raw_http_response.json()
        wire = body.get("answers") if isinstance(body, Mapping) else None
        if not isinstance(wire, Mapping):
            raise DeciderResponseError("the response body has no 'answers' object")
        to_answers(req.questions, wire)  # validate completeness, types and label sets BEFORE anything is cached
        now = datetime.now(UTC)
        stored = [
            CachedAnswer(
                key=keys[qid],
                namespace=req.namespace,
                requested_model=self.cfg.model,
                response_model=resp.model,
                question_set_id=req.question_set_id,
                question_set_hash=req.question_set_hash,
                state_hash=req.state_hash,
                question_hash=canon.sha256_hex(canon.dumps_ordered(dict(question))),
                question_id=qid,
                request_kind=req.kind.value,
                variant=req.variant.value,
                # "exact wire bytes" are not recoverable from .json() and are not needed: a canonical re-encoding is
                answer_json=canon.dumps_sorted(wire[qid]),
                request_id=request_id,
                input_tokens=input_tokens,
                latency_ms=latency_ms,
                sdk_version=self.cfg.sdk_version,
                created_at=now,
            )
            for qid, question in req.questions.items()
        ]
        self.cache.put_request(
            req.namespace,
            stored,
            canon.dumps_ordered(req.state),
            canon.dumps_ordered(req.questions),
            (req.session, req.underlying, req.kind.value, req.variant.value),
        )
        written = self.cache.get_many(req.namespace, list(keys.values()))
        if len(written) != len(keys):  # pragma: no cover - put_request is all-or-nothing
            raise InvariantError("the decision cache did not store the whole request")
        return result_from_rows(req, self.cfg.model, keys, written, source="live")

    # ------------------------------------------------------------------ the metered HTTP loop

    def _send(self, req: DecisionRequest) -> tuple[Any, str | None, int | None, int | None]:
        estimate = self.estimate_tokens(req)
        honoured_retry_after = False
        for attempt in range(self.cfg.retry_max + 1):
            # INV-17: the reservation is taken BEFORE the HTTP attempt - a spend stop never reaches the transport
            reservation = self.spend.reserve(self.run_id, estimate)
            self._requests.acquire(1)
            self._tokens.acquire(estimate)
            started = time.monotonic()
            try:
                resp = self._client.system_one(req.state, req.questions, model=self.cfg.model)
            except BaseException as exc:
                self.spend.commit(reservation, estimate, estimated=True)  # conservative: assume the attempt was billed
                mapped, retryable, retry_after_ms = self._errors.map(exc)
                if mapped is None:
                    raise
                self._log_failure(req, attempt, mapped, exc)
                if not retryable or attempt >= self.cfg.retry_max:
                    raise mapped from None
                delay = self._backoff(attempt)
                if retry_after_ms is not None and not honoured_retry_after:
                    honoured_retry_after = True
                    delay = min(retry_after_ms / 1000.0, _MAX_RETRY_AFTER_S)
                self._sleep(delay)
                continue
            latency_ms = int((time.monotonic() - started) * 1000)
            reported = resp.usage.input_tokens if resp.usage is not None else None
            self.spend.commit(reservation, estimate if reported is None else int(reported), estimated=reported is None)
            headers = resp.raw_http_response.headers
            request_id = headers.get(REQUEST_ID_HEADER)  # `resp.request_id` RAISES when the header is absent
            _log.info(
                "jev answered: kind=%s variant=%s questions=%d request_id=%s",
                req.kind.value,
                req.variant.value,
                len(req.questions),
                request_id or "-",
            )
            return resp, request_id, (None if reported is None else int(reported)), latency_ms
        raise InvariantError("the retry loop ended without a response or an error")  # pragma: no cover - unreachable

    def estimate_tokens(self, req: DecisionRequest) -> int:
        """`ceil(len(request_json) / estimate_chars_per_token) + estimate_overhead_tokens` (6.8 step 3).

        `request_json` is the body the SDK sends - `{"state", "model", "questions"}` in that order - so the worst-case
        estimate the guard reserves grows with the batch, exactly like the bill.
        """
        spend_cfg = self.cfg.spend
        body = canon.dumps_ordered({"state": req.state, "model": self.cfg.model, "questions": req.questions})
        return math.ceil(len(body) / spend_cfg.estimate_chars_per_token) + spend_cfg.estimate_overhead_tokens

    def _backoff(self, attempt: int) -> float:
        table = self.cfg.retry_backoff_s
        if not table:
            return 0.0
        return float(table[min(attempt, len(table) - 1)])

    def _log_failure(self, req: DecisionRequest, attempt: int, mapped: DeciderError, exc: BaseException) -> None:
        """Log the exception CLASS, the status and the request id - never a body, never a header, never the state."""
        status = getattr(exc, "status", None)
        headers = getattr(exc, "headers", None)
        request_id = headers.get(REQUEST_ID_HEADER) if hasattr(headers, "get") else None
        if status == 422:
            self._remember_detail(exc)
        _log.warning(
            "jev request failed: kind=%s attempt=%d error=%s mapped=%s status=%s request_id=%s",
            req.kind.value,
            attempt,
            type(exc).__name__,
            type(mapped).__name__,
            status if status is not None else "-",
            request_id or "-",
        )

    def _remember_detail(self, exc: BaseException) -> None:
        """Keep a 422's `detail` for the caller's ledger sidecar (6.8). It is never logged and never cached."""
        body = getattr(exc, "body", None)
        detail: Any = body.get("detail") if isinstance(body, Mapping) else None
        text = None if detail is None else str(detail)[:_MAX_DETAIL_CHARS]
        with self._lock:
            self.last_response_detail = text

    def close(self) -> None:
        """Release the HTTP client (and its transport)."""
        self._client.close()


class _ErrorMap:
    """The 6.8 catch order, built once from the imported SDK module (B1.10)."""

    def __init__(self, sdk: Any) -> None:
        self.validation = sdk.TypeSafeAPIResponseValidationError
        self.rate_limit = sdk.TypeSafeRateLimitError
        self.retryable = (
            sdk.TypeSafeRateLimitError,
            sdk.TypeSafeInternalServerError,
            sdk.TypeSafeAPITimeoutError,
            sdk.TypeSafeAPIConnectionError,
        )
        self.config = (
            sdk.TypeSafeAuthenticationError,
            sdk.TypeSafePermissionDeniedError,
            sdk.TypeSafeBadRequestError,
            sdk.TypeSafeNotFoundError,
            sdk.TypeSafeUnprocessableEntityError,
        )
        self.base = sdk.TypeSafeError

    def map(self, exc: BaseException) -> tuple[DeciderError | None, bool, float | None]:
        """`(our exception, retryable, retry_after_ms)`; `(None, ...)` for anything that is not an SDK error (it propagates)."""
        # order matters: TypeSafeAPIResponseValidationError IS a TypeSafeAPIError, and a 200 body is never retried
        if isinstance(exc, self.validation):
            return DeciderResponseError(f"the response failed schema validation at {getattr(exc, 'field_path', '?')!r}"), False, None
        if isinstance(exc, self.config):
            return (
                DeciderConfigError(f"{type(exc).__name__} (status {getattr(exc, 'status', '?')}): bad key, bad request or our bug"),
                False,
                None,
            )
        if isinstance(exc, self.retryable):
            retry_after = getattr(exc, "retry_after_ms", None) if isinstance(exc, self.rate_limit) else None
            status = getattr(exc, "status", None)
            return (
                DeciderTransportError(f"{type(exc).__name__} (status {status if status is not None else '-'})"),
                True,
                None if retry_after is None else float(retry_after),
            )
        if isinstance(exc, self.base):
            # any other TypeSafeAPIError / TypeSafeError: transient by 6.8's table, and retried like one
            return DeciderTransportError(f"{type(exc).__name__}"), True, None
        return None, False, None
