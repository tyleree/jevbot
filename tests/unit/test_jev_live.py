"""`jev/live.py` through the REAL SDK over `httpx2.MockTransport` (DESIGN.md 6.8, 15.3; INV-06, INV-17, INV-18).

`LiveJev(..., api_key="dummy", transport=make_jev_transport())` is the only way these tests touch `typesafe_sdk`: the SDK's
own encoder, headers, error classes and response decoder all run; only the socket is replaced (and `tests/conftest.py`
blocks the network anyway). What is pinned here is what 6.8 promises:

* the FULL batch is always sent, with wire bytes equal to `canon.dumps_ordered` of what we hashed;
* a hit makes ZERO HTTP calls and still verifies the model;
* a model mismatch caches nothing;
* every SDK error class maps to its documented exception, and the retry loop meters EVERY attempt;
* a spend stop raises before the transport is touched at all.
"""

import json
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import msgspec
import pytest

from jevbot import canon, questions as questions_module
from jevbot.config import Config, JevConfig, JevSpendConfig
from jevbot.errors import (
    ConfigError,
    DeciderConfigError,
    DeciderResponseError,
    DeciderTransportError,
    ModelMismatchError,
    SpendLimitError,
)
from jevbot.jev.cache import SqliteDecisionCache
from jevbot.jev.live import LiveJev
from jevbot.jev.spend import SCOPE_BATCH, SpendGuard
from jevbot.types import CacheMode, DecisionRequest, RequestKind, Variant
from tests.fixtures.jev_transport import DEFAULT_MODEL, NAMESPACE, REQUEST_ID, Fault, make_jev_transport, make_request

RELEASE = date(2026, 9, 15)


@dataclass
class Rig:
    jev: LiveJev
    cache: SqliteDecisionCache
    spend: SpendGuard
    calls: list[dict[str, Any]] = field(default_factory=list)
    slept: list[float] = field(default_factory=list)

    def close(self) -> None:
        self.jev.close()
        self.cache.close()
        self.spend.close()


def make_rig(
    tmp_path: Path,
    *,
    faults: Sequence[Fault] = (),
    model: str = DEFAULT_MODEL,
    usage: dict[str, int] | None = None,
    omit_request_id: bool = False,
    spend_cfg: JevSpendConfig | None = None,
    jev_cfg: JevConfig | None = None,
    api_key: str = "dummy",
    known_models: frozenset[str] | None = None,
) -> Rig:
    cfg = jev_cfg or msgspec.structs.replace(Config().jev, retry_backoff_s=(0.0, 0.0))
    if spend_cfg is not None:
        cfg = msgspec.structs.replace(cfg, spend=spend_cfg)
    cache = SqliteDecisionCache(tmp_path / "cache" / "decisions.sqlite")
    cache.ensure_namespace(NAMESPACE, cfg.model, RELEASE, refresh=False)
    spend = SpendGuard(tmp_path / "state" / "spend.sqlite", scope=SCOPE_BATCH, cfg=cfg.spend)
    calls: list[dict[str, Any]] = []
    slept: list[float] = []
    transport = make_jev_transport(
        faults=faults,
        model=model,
        calls=calls,
        usage=usage if usage is not None else {"input_tokens": 1234, "output_tokens": 20},
        omit_request_id=omit_request_id,
        known_models=known_models,
    )
    jev = LiveJev(
        cfg,
        cache,
        spend,
        api_key=api_key,
        run_id="run-1",
        mode=CacheMode.RECORD,
        transport=transport,
        sleep=slept.append,
    )
    return Rig(jev=jev, cache=cache, spend=spend, calls=calls, slept=slept)


def _max_attempts(stop: Any) -> int | None:
    """The SDK's tenacity stop tree may be a combination; find the attempt cap inside it."""
    if hasattr(stop, "max_attempt_number"):
        return int(stop.max_attempt_number)
    for child in getattr(stop, "stops", ()):
        found = _max_attempts(child)
        if found is not None:
            return found
    return None


# ======================================================================================================================
# The happy path
# ======================================================================================================================


def test_the_full_batch_is_sent_with_the_bytes_we_hashed(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    try:
        req = make_request()
        result = rig.jev.decide(req)
        assert len(rig.calls) == 1
        body = json.loads(rig.calls[0]["raw"])
        assert list(body) == ["state", "model", "questions"]
        assert body["model"] == rig.jev.model
        assert list(body["questions"]) == list(req.questions), "the FULL batch, in its authored order (D8 / G3)"
        assert len(body["questions"]) == 19
        raw_text = rig.calls[0]["raw"].decode()
        # 5.9: the SDK serialises with msgspec, which keeps insertion order, so the wire bytes ARE dumps_ordered's
        assert canon.dumps_ordered(req.state) in raw_text
        assert canon.dumps_ordered(req.questions) in raw_text
        assert canon.sha256_hex(canon.dumps_ordered(body["state"])) == req.state_hash
        assert result.source == "live" and result.model == DEFAULT_MODEL
        assert set(result.answers) == set(req.questions) and set(result.cache_keys) == set(req.questions)
        assert result.request_id == REQUEST_ID and result.input_tokens == 1234 and result.latency_ms is not None
        assert result.decision_id == req.decision_id and result.state_hash == req.state_hash
    finally:
        rig.close()


def test_a_cache_hit_makes_zero_http_calls_and_still_verifies_the_model(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    try:
        req = make_request()
        first = rig.jev.decide(req)
        second = rig.jev.decide(req)
        assert len(rig.calls) == 1, "the second decide() is answered from the cache"
        assert second.source == "cache"
        assert {qid: answer for qid, answer in second.answers.items()} == first.answers
        assert second.cache_keys == first.cache_keys and second.request_id == REQUEST_ID
        # INV-06 on the hit path: a namespace whose rows came from another model can never answer for this one
        other = msgspec.structs.replace(rig.jev.cfg, model="jev-1.14.0")
        rig.jev.cfg = other
        with pytest.raises(ModelMismatchError):
            rig.jev.decide(make_request())
    finally:
        rig.close()


def test_one_missing_key_resends_the_whole_batch(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    try:
        req = make_request()
        rig.jev.decide(req)
        # drop one question from the batch: a different question set, so every key differs and the FULL batch is resent
        shorter = {qid: q for qid, q in req.questions.items() if qid != "risk.environment"}
        smaller = make_request(questions=shorter)
        rig.jev.decide(smaller)
        assert len(rig.calls) == 2
        assert len(json.loads(rig.calls[1]["raw"])["questions"]) == 18
        assert rig.cache.stats()["answers"] == 19 + 18
    finally:
        rig.close()


def test_a_missing_request_id_header_is_tolerated(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, omit_request_id=True)
    try:
        result = rig.jev.decide(make_request())
        assert result.request_id is None  # `resp.request_id` would RAISE; 6.8 step 7 reads the header
        assert result.source == "live" and len(result.answers) == 19
    finally:
        rig.close()


def test_usage_none_falls_back_to_the_estimate(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, faults=[Fault.usage_none()])
    try:
        req = make_request()
        estimate = rig.jev.estimate_tokens(req)
        result = rig.jev.decide(req)
        assert result.input_tokens is None
        assert rig.spend.totals("run-1") == (estimate, estimate), "an unreported bill is metered at the estimate"
        body = canon.dumps_ordered({"state": req.state, "model": rig.jev.model, "questions": req.questions})
        assert estimate == -(-len(body) // 3) + 300  # ceil(chars / 3.0) + 300, hand-computed from the shipped defaults
    finally:
        rig.close()


# ======================================================================================================================
# INV-06
# ======================================================================================================================


def test_a_model_mismatch_raises_and_caches_nothing(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, model="jev-1.14.0")
    try:
        with pytest.raises(ModelMismatchError, match="jev-1.14.0"):
            rig.jev.decide(make_request())
        assert rig.cache.stats()["answers"] == 0
        assert len(rig.calls) == 1 and rig.spend.totals("run-1")[0] == 1234, "the answered request is still billed"
    finally:
        rig.close()


def test_the_wrong_model_fault_is_caught_on_the_first_attempt(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, faults=[Fault.wrong_model("jev-1.14.0")])
    try:
        with pytest.raises(ModelMismatchError):
            rig.jev.decide(make_request())
        assert rig.cache.stats()["answers"] == 0
    finally:
        rig.close()


# ======================================================================================================================
# Error mapping (6.8's table)
# ======================================================================================================================


@pytest.mark.parametrize(
    ("fault", "expected", "attempts"),
    [
        (Fault.http_401(), DeciderConfigError, 1),
        (Fault.http_403(), DeciderConfigError, 1),
        (Fault.http_422(), DeciderConfigError, 1),
        (Fault.http_429(), DeciderTransportError, 3),
        (Fault.http_529(), DeciderTransportError, 3),
        (Fault.timeout(), DeciderTransportError, 3),
        (Fault.connect_error(), DeciderTransportError, 3),
        (Fault.missing_field("answers.x.probabilities"), DeciderResponseError, 1),
        (Fault.unknown_answer_type(), DeciderResponseError, 1),
        (Fault.unknown_label(), DeciderResponseError, 1),
    ],
)
def test_sdk_errors_map_per_the_6_8_table(tmp_path: Path, fault: Fault, expected: type[Exception], attempts: int) -> None:
    # the retryable faults are queued once per attempt so that the whole loop runs into them
    faults = [fault] * attempts
    rig = make_rig(tmp_path, faults=faults)
    try:
        with pytest.raises(expected):
            rig.jev.decide(make_request())
        assert len(rig.calls) == attempts, "config errors are never retried; transport errors use our retry_max = 2"
        assert rig.cache.stats()["answers"] == 0
        assert rig.spend.reservations("run-1") == attempts, "EVERY attempt is metered (it may have been billed)"
    finally:
        rig.close()


def test_a_retryable_failure_that_clears_succeeds_on_the_next_attempt(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, faults=[Fault.http_529()])
    try:
        result = rig.jev.decide(make_request())
        assert result.source == "live" and len(rig.calls) == 2
        assert rig.spend.reservations("run-1") == 2
        assert rig.slept == [0.0], "the first back-off of the configured table"
    finally:
        rig.close()


def test_retry_after_is_honoured_once_and_capped(tmp_path: Path) -> None:
    cfg = msgspec.structs.replace(Config().jev, retry_backoff_s=(0.25, 0.25), retry_max=3)
    rig = make_rig(tmp_path, faults=[Fault.http_429(20_000), Fault.http_429(20_000)], jev_cfg=cfg)
    try:
        rig.jev.decide(make_request())
        assert rig.slept == [5.0, 0.25], "the first retry-after is honoured and capped at 5 s; the second uses the table"
    finally:
        rig.close()


def test_a_422_detail_is_kept_for_the_sidecar_but_never_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    rig = make_rig(tmp_path, faults=[Fault.http_422("questions.0.criteria: field required")])
    try:
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(DeciderConfigError):
                rig.jev.decide(make_request())
        assert rig.jev.last_response_detail == "questions.0.criteria: field required"
        assert "field required" not in caplog.text, "the 422 detail goes to the sidecar, never to the log"
        assert "TypeSafeUnprocessableEntityError" in caplog.text and "status=422" in caplog.text
    finally:
        rig.close()


def test_a_probability_sum_outside_the_band_is_kept_not_refused(tmp_path: Path) -> None:
    # 7.1: `raw_sum` outside [0.98, 1.02] makes the question read UNCERTAIN in rules.py - it is not a decider error
    rig = make_rig(tmp_path, faults=[Fault.prob_sum(0.96)])
    try:
        result = rig.jev.decide(make_request())
        answer = result.answers["regime.market"]
        assert getattr(answer, "raw_sum") == pytest.approx(0.96)
    finally:
        rig.close()


# ======================================================================================================================
# INV-17: the reservation happens BEFORE the HTTP call
# ======================================================================================================================


def test_a_spend_stop_blocks_before_the_transport_is_touched(tmp_path: Path) -> None:
    # the DAY ceiling binds: the scope is blocked stickily and nothing reaches the transport
    tight = JevSpendConfig(max_input_tokens_per_run=10_000_000, max_input_tokens_per_day=10, paper_max_input_tokens_per_day=10)
    rig = make_rig(tmp_path, spend_cfg=tight)
    try:
        with pytest.raises(SpendLimitError, match="UTC-day"):
            rig.jev.decide(make_request())
        assert rig.calls == [], "transport call count is 0: the guard stops the request before any HTTP attempt"
        assert rig.cache.stats()["answers"] == 0
        assert rig.spend.blocked()
    finally:
        rig.close()


def test_a_run_ceiling_also_stops_before_the_transport(tmp_path: Path) -> None:
    # the RUN ceiling binds: this run stops (exit 7, resumable) but the scope's day is not blocked
    tight = JevSpendConfig(max_input_tokens_per_run=10, max_input_tokens_per_day=10_000_000, paper_max_input_tokens_per_day=10)
    rig = make_rig(tmp_path, spend_cfg=tight)
    try:
        with pytest.raises(SpendLimitError, match="would exceed its input-token ceiling"):
            rig.jev.decide(make_request())
        assert rig.calls == [] and not rig.spend.blocked()
    finally:
        rig.close()


def test_the_estimate_is_reserved_before_and_the_real_usage_committed_after(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    try:
        req = make_request()
        estimate = rig.jev.estimate_tokens(req)
        assert estimate > 1234, "the worst case is reserved first; the reported bill is smaller here"
        rig.jev.decide(req)
        assert rig.spend.totals("run-1") == (1234, 1234)
        assert rig.spend.reservations("run-1") == 1
    finally:
        rig.close()


# ======================================================================================================================
# Construction guards
# ======================================================================================================================


def test_the_sdk_is_always_called_with_retries_disabled(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    try:
        client = rig.jev._client
        assert _max_attempts(client._retry.stop) == 1, "RetryPolicy(max_retries=0): OUR loop is the only retry"
    finally:
        rig.close()


@pytest.mark.parametrize("api_key", ["", "   ", "\t"])
def test_a_blank_api_key_is_refused(tmp_path: Path, api_key: str) -> None:
    with pytest.raises(ConfigError, match="non-empty"):
        make_rig(tmp_path, api_key=api_key)


def test_replay_mode_never_constructs_a_client(tmp_path: Path) -> None:
    cfg = Config().jev
    cache = SqliteDecisionCache(tmp_path / "cache" / "decisions.sqlite")
    spend = SpendGuard(tmp_path / "state" / "spend.sqlite", scope=SCOPE_BATCH, cfg=cfg.spend)
    try:
        with pytest.raises(ConfigError, match="ReplayJev"):
            LiveJev(cfg, cache, spend, api_key="dummy", run_id="r", mode=CacheMode.REPLAY)
    finally:
        cache.close()
        spend.close()


def test_a_pinned_sdk_version_mismatch_is_refused(tmp_path: Path) -> None:
    cfg = msgspec.structs.replace(Config().jev, sdk_version="0.5.0")
    with pytest.raises(ConfigError, match="0.5.0"):
        make_rig(tmp_path, jev_cfg=cfg)


def test_a_body_logging_sdk_log_level_is_refused_before_the_sdk_is_imported(tmp_path: Path) -> None:
    """INV-18: the SDK applies TYPESAFE_LOG_LEVEL once AT IMPORT, so the check has to come first - proven in a fresh process."""
    script = (
        "import sys;"
        "from jevbot.config import Config;"
        "from jevbot.jev.live import LiveJev;"
        "from jevbot.types import CacheMode;"
        "print('imported_at_module_level', 'typesafe_sdk' in sys.modules);"
        "err = None\n"
        "try:\n"
        "    LiveJev(Config().jev, None, None, api_key='dummy', run_id='r', mode=CacheMode.RECORD)\n"
        "except Exception as exc:\n"
        "    err = type(exc).__name__\n"
        "print('error', err, 'imported_after', 'typesafe_sdk' in sys.modules)"
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        "TYPESAFE_LOG_LEVEL": "debug",
    }
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env)
    lines = out.stdout.split()
    assert "imported_at_module_level False" in out.stdout, "jev/live.py must not import the SDK at module level"
    assert "error ConfigError imported_after False" in out.stdout, out.stdout
    assert lines  # the subprocess really ran


def test_the_state_and_question_hashes_of_a_request_are_checked_before_anything_is_sent(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    try:
        req = make_request()
        broken = DecisionRequest(
            **{**{f: getattr(req, f) for f in req.__struct_fields__}, "state_hash": "0" * 64}  # type: ignore[arg-type]
        )
        with pytest.raises(Exception, match="state_hash"):
            rig.jev.decide(broken)
        assert rig.calls == []
    finally:
        rig.close()


def test_every_request_kind_and_variant_round_trips(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    try:
        for kind in (RequestKind.ENTRY, RequestKind.ENTRY_TEXT, RequestKind.MANAGE, RequestKind.MANAGE_TEXT):
            for variant in (Variant.BASE, Variant.OPT_PERM):
                questions = (
                    questions_module.opt_perm(questions_module.QUESTION_SETS[questions_module.vocab.QUESTION_SET_ID[kind.value]])
                    if variant is Variant.OPT_PERM
                    else None
                )
                req = make_request(kind, variant, questions=questions)
                result = rig.jev.decide(req)
                assert set(result.answers) == set(req.questions)
                assert result.kind is kind and result.variant is variant
        # `entry_text.v1` and `manage_text.v1` are Noul-only sets: OPT_PERM leaves them byte-identical, so their
        # permuted variant has the same content keys and is answered from the cache (D8: the key IS the content)
        assert len(rig.calls) == 6
        assert rig.cache.stats()["requests"] == 6
    finally:
        rig.close()
