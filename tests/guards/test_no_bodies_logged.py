"""Guard: `LiveJev` never logs a request or response body, or a secret, even at application DEBUG (INV-18, 15.5).

The SDK logs FULL request and response bodies at DEBUG (`typesafe_sdk._core.transport::_log_wire`; only headers are
redacted), and it takes its level from `TYPESAFE_LOG_LEVEL` **once at import**. jevbot therefore does two things, both
guarded here: it refuses a body-logging value of that variable before the SDK is imported at all, and it pins the
`typesafe_sdk` logger to WARNING so that `jevbot --log-level debug` cannot surface a body.

The full-cycle variant of this guard is WP13's `tests/integration/test_no_bodies_logged_cycle.py`.
"""

import logging
from datetime import date
from pathlib import Path

import msgspec
import pytest

from jevbot import logsetup
from jevbot.config import Config
from jevbot.errors import ConfigError, DeciderConfigError
from jevbot.jev.cache import SqliteDecisionCache
from jevbot.jev.live import LiveJev
from jevbot.jev.spend import SCOPE_BATCH, SpendGuard
from jevbot.types import CacheMode
from tests.fixtures.jev_transport import NAMESPACE, Fault, make_jev_transport, make_request

SECRET_KEY = "sk-live-do-not-log-me"

# fragments that may only ever appear in a BODY, never in a log line
BODY_FRAGMENTS = (
    "Which description best fits the current market regime",  # a question's instructions
    "underlying_alias",  # a state key
    "broad US large-cap equity index ETF",  # a state value
    "probabilities",  # an answer field
    "trending_up_calm",  # an option label
    SECRET_KEY,
    "Bearer",
)


def _rig(tmp_path: Path, *, faults: tuple[Fault, ...] = ()) -> tuple[LiveJev, SqliteDecisionCache, SpendGuard]:
    cfg = msgspec.structs.replace(Config().jev, retry_backoff_s=(0.0, 0.0))
    cache = SqliteDecisionCache(tmp_path / "cache" / "decisions.sqlite")
    cache.ensure_namespace(NAMESPACE, cfg.model, date(2026, 9, 15), refresh=False)
    spend = SpendGuard(tmp_path / "state" / "spend.sqlite", scope=SCOPE_BATCH, cfg=cfg.spend)
    jev = LiveJev(
        cfg,
        cache,
        spend,
        api_key=SECRET_KEY,
        run_id="guard",
        mode=CacheMode.RECORD,
        transport=make_jev_transport(faults=faults, api_key=SECRET_KEY),
        sleep=lambda _seconds: None,
    )
    return jev, cache, spend


def test_a_successful_request_at_debug_logs_no_body_and_no_secret(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    logsetup.register_secret(SECRET_KEY)
    logsetup.configure("debug", console=False)  # the application asks for DEBUG everywhere
    jev, cache, spend = _rig(tmp_path)
    try:
        with caplog.at_level(logging.DEBUG):
            result = jev.decide(make_request())
        assert result.source == "live"
        text = caplog.text
        assert text, "the run did log something (otherwise this guard would pass vacuously)"
        assert "jev answered" in text
        for fragment in BODY_FRAGMENTS:
            assert fragment not in text, f"a DEBUG log line leaked {fragment!r}"
        assert logging.getLogger("typesafe_sdk").level == logging.WARNING, "INV-18: the SDK logger is pinned"
    finally:
        jev.close()
        cache.close()
        spend.close()


def test_a_failing_request_at_debug_logs_only_the_class_status_and_request_id(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    logsetup.register_secret(SECRET_KEY)
    logsetup.configure("debug", console=False)
    jev, cache, spend = _rig(tmp_path, faults=(Fault.http_422("questions.0.criteria: field required"),))
    try:
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(DeciderConfigError):
                jev.decide(make_request())
        text = caplog.text
        assert "TypeSafeUnprocessableEntityError" in text and "status=422" in text
        assert "field required" not in text, "not even the 422 detail is logged (it goes to the sidecar)"
        for fragment in BODY_FRAGMENTS:
            assert fragment not in text, f"a DEBUG log line leaked {fragment!r}"
    finally:
        jev.close()
        cache.close()
        spend.close()


def test_the_sdk_logger_stays_pinned_when_the_application_reconfigures_logging(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    jev, cache, spend = _rig(tmp_path)
    try:
        logsetup.configure("debug", console=False)  # a later reconfiguration must not un-pin the SDK
        assert logging.getLogger("typesafe_sdk").level == logging.WARNING
        with caplog.at_level(logging.DEBUG):
            logging.getLogger("typesafe_sdk").debug("POST /v1/systemone -> body=%r", {"state": "underlying_alias"})
            jev.decide(make_request())
        assert "underlying_alias" not in caplog.text
    finally:
        jev.close()
        cache.close()
        spend.close()


def test_a_body_logging_sdk_log_level_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # every spelling the SDK would accept as "debug" (it normalises with .strip().lower()) is refused by LiveJev
    for value in ("debug", " debug", "DEBUG ", "info", "Info", "trace"):
        monkeypatch.setenv("TYPESAFE_LOG_LEVEL", value)
        with pytest.raises(ConfigError, match="TYPESAFE_LOG_LEVEL"):
            _rig(tmp_path)
    for value in ("", "warn", "WARNING ", "error", "off"):
        monkeypatch.setenv("TYPESAFE_LOG_LEVEL", value)
        jev, cache, spend = _rig(tmp_path)
        jev.close()
        cache.close()
        spend.close()
