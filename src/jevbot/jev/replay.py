"""ReplayJev - answers from the decision cache only; it imports NOTHING from `typesafe_sdk` (DESIGN.md 6.8, D7, D8).

Cache mode `replay` is the default for backtests (G12): a miss is not a fallback, it is a `CacheMissError` that aborts the
run (exit 5). No config knob disables it, and no network client is ever constructed - `tests/guards/test_import_rules.py`
and `tests/unit/test_jev_replay.py` both assert that importing this module leaves `typesafe_sdk` out of `sys.modules`.

That is what makes every rules / risk / fill sweep, every baseline, the placebo and the D11 ablation arm free and exact: the
entry-type request set of a session is a pure function of the market data (structural choice 3), so a recorded run replays
with zero entry-type misses, and the answers are the recorded bytes - not a re-run of the model.

The model is verified on hits too (INV-06): a namespace's rows can only answer for the model they were recorded with.
"""

from typing import Final

from jevbot.config import JevConfig
from jevbot.errors import CacheMissError
from jevbot.jev.common import cache_keys_for, check_request_hashes, result_from_rows
from jevbot.protocols import DecisionCache
from jevbot.types import DecisionRequest, DecisionResult

__all__ = ["REPLAY_NAME", "ReplayJev"]

REPLAY_NAME: Final = "replay_jev"


class ReplayJev:
    """The replay `Decider` (`name = "replay_jev"`) over a recorded decision cache (opened `mode=ro` by the caller)."""

    def __init__(self, cfg: JevConfig, cache: DecisionCache) -> None:
        self.cfg = cfg
        self.cache = cache

    @property
    def name(self) -> str:
        return REPLAY_NAME

    @property
    def model(self) -> str:
        return self.cfg.model

    def decide(self, req: DecisionRequest) -> DecisionResult:
        """Answer from the cache; ANY missing key raises `CacheMissError(state_hash, missing question ids)`."""
        check_request_hashes(req)
        keys = cache_keys_for(self.cfg.model, req)
        rows = self.cache.get_many(req.namespace, list(keys.values()))
        missing = [qid for qid, key in keys.items() if key not in rows]
        if missing:
            raise CacheMissError(
                f"replay miss in namespace {req.namespace!r} for state {req.state_hash[:12]} "
                f"({req.kind.value}/{req.variant.value}, {len(missing)} of {len(keys)} questions missing: {sorted(missing)})"
            )
        return result_from_rows(req, self.cfg.model, keys, rows, source="cache")

    def close(self) -> None:
        """Nothing to release: the cache belongs to the caller."""
