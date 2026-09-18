"""`jev/replay.py`: record once, replay for ever, with the network blocked (DESIGN.md 6.8, D7, D8, G12).

Replay is the mode every threshold sweep, baseline and ablation arm runs in, so two properties decide whether the evidence
store is trustworthy: a miss must ABORT (never fall back to a model call), and the module must not even be able to reach the
network - `ReplayJev` imports nothing from `typesafe_sdk`, which a fresh subprocess proves here.
"""

import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import msgspec
import pytest

from jevbot import questions as questions_module
from jevbot.config import Config
from jevbot.errors import CacheMissError, ModelMismatchError
from jevbot.jev.cache import SqliteDecisionCache
from jevbot.jev.live import LiveJev
from jevbot.jev.replay import REPLAY_NAME, ReplayJev
from jevbot.jev.spend import SCOPE_BATCH, SpendGuard
from jevbot.types import CacheMode, RequestKind, Variant
from tests.fixtures.jev_transport import NAMESPACE, make_jev_transport, make_request

RELEASE = date(2026, 9, 15)


def record(tmp_path: Path, kinds: tuple[RequestKind, ...] = (RequestKind.ENTRY,)) -> tuple[SqliteDecisionCache, int]:
    """Record one request per kind with LiveJev over the mock transport, then close the writer."""
    cfg = msgspec.structs.replace(Config().jev, retry_backoff_s=(0.0, 0.0))
    cache = SqliteDecisionCache(tmp_path / "cache" / "decisions.sqlite")
    cache.ensure_namespace(NAMESPACE, cfg.model, RELEASE, refresh=False)
    spend = SpendGuard(tmp_path / "state" / "spend.sqlite", scope=SCOPE_BATCH, cfg=cfg.spend)
    calls: list[dict[str, object]] = []
    jev = LiveJev(
        cfg, cache, spend, api_key="dummy", run_id="rec", mode=CacheMode.RECORD, transport=make_jev_transport(calls=calls)
    )
    for kind in kinds:
        jev.decide(make_request(kind))
    jev.close()
    spend.close()
    cache.close()
    return SqliteDecisionCache(tmp_path / "cache" / "decisions.sqlite", read_only=True), len(calls)


def test_record_then_replay_reproduces_every_answer(tmp_path: Path) -> None:
    kinds = (RequestKind.ENTRY, RequestKind.ENTRY_TEXT, RequestKind.MANAGE)
    cache, recorded = record(tmp_path, kinds)
    try:
        assert recorded == len(kinds)
        replay = ReplayJev(Config().jev, cache)
        assert replay.name == REPLAY_NAME == "replay_jev" and replay.model == "jev-1.13.0"
        for kind in kinds:
            req = make_request(kind)
            result = replay.decide(req)
            assert result.source == "cache" and set(result.answers) == set(req.questions)
            assert result.decision_id == req.decision_id and result.model == replay.model
            assert result.request_id == "req-test-0001" and result.input_tokens == 1234
        replay.close()
    finally:
        cache.close()


def test_a_replay_miss_raises_and_names_what_is_missing(tmp_path: Path) -> None:
    cache, _ = record(tmp_path)
    try:
        replay = ReplayJev(Config().jev, cache)
        # a state that was never recorded: every key of the batch misses
        other = make_request()
        other.state["market"]["as_of"] = "two sessions ago"
        missing = make_request(state=other.state)
        with pytest.raises(CacheMissError) as caught:
            replay.decide(missing)
        assert missing.state_hash[:12] in str(caught.value)
        assert "19 of 19 questions missing" in str(caught.value)
        # a PARTIAL hit is a miss too: 13.3's hit rule is all-or-nothing
        shorter = {qid: q for qid, q in questions_module.ENTRY_V1.items() if qid != "risk.environment"}
        with pytest.raises(CacheMissError, match="18 of 18"):
            replay.decide(make_request(questions=shorter))
    finally:
        cache.close()


def test_a_variant_that_was_not_recorded_misses(tmp_path: Path) -> None:
    cache, _ = record(tmp_path)
    try:
        replay = ReplayJev(Config().jev, cache)
        permuted = questions_module.opt_perm(questions_module.ENTRY_V1)
        with pytest.raises(CacheMissError):
            replay.decide(make_request(Variant.OPT_PERM and RequestKind.ENTRY, Variant.OPT_PERM, questions=permuted))
    finally:
        cache.close()


def test_the_model_is_verified_on_hits(tmp_path: Path) -> None:
    cache, _ = record(tmp_path)
    try:
        replay = ReplayJev(msgspec.structs.replace(Config().jev, model="jev-1.14.0"), cache)
        with pytest.raises(ModelMismatchError, match="INV-06"):
            # the keys are model-dependent, so a foreign model normally MISSES; force the hit path with the recorded keys
            replay.cache = _ModelBlindCache(cache)  # type: ignore[assignment]
            replay.decide(make_request())
    finally:
        cache.close()


class _ModelBlindCache:
    """A cache double that answers every key from the recorded row set - the only way to reach the hit path with a
    foreign model id (the real keys embed the model). It proves the row-level model check of `result_from_rows`."""

    def __init__(self, inner: SqliteDecisionCache) -> None:
        self._rows = list(inner.get_many(NAMESPACE, [row for row in _all_keys(inner)]).values())

    def get_many(self, namespace: str, keys: list[str]) -> dict[str, object]:
        return {key: row for key, row in zip(keys, self._rows, strict=False)}


def _all_keys(cache: SqliteDecisionCache) -> list[str]:
    questions = questions_module.ENTRY_V1
    req = make_request()
    from jevbot import canon

    return [canon.cache_key("jev-1.13.0", req.state, req.question_set_hash, q) for q in questions.values()]


def test_replay_jev_never_imports_the_sdk(tmp_path: Path) -> None:
    """The import rules of section 1: only `jev/live.py` and `jev/probe.py` may import `typesafe_sdk`."""
    script = (
        "import sys;"
        "from jevbot.jev.replay import ReplayJev;"
        "from jevbot.jev.cache import SqliteDecisionCache;"
        "from jevbot.config import Config;"
        "print('sdk', 'typesafe_sdk' in sys.modules, 'httpx2', 'httpx2' in sys.modules);"
        "print('name', ReplayJev(Config().jev, None).name)"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env)
    assert out.stdout.splitlines()[0] == "sdk False httpx2 False"
    assert out.stdout.splitlines()[1] == "name replay_jev"


def test_replay_opens_the_cache_read_only_and_writes_nothing(tmp_path: Path) -> None:
    cache, _ = record(tmp_path)
    try:
        assert cache.read_only is True
        replay = ReplayJev(Config().jev, cache)
        before = cache.manifest_hash(NAMESPACE)
        replay.decide(make_request())
        assert cache.manifest_hash(NAMESPACE) == before
        assert cache.stats()["answers"] == 19
    finally:
        cache.close()
