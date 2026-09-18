# BUILD BRIEF: Jev-driven options bot (historical backtest + Alpaca PAPER only)

Date: 2026-09-17. Target: Python 3.12, WSL2 Ubuntu, uv. No live money is in scope anywhere.

Evidence tags used throughout:
- **[V]** verified: read from the installed `typesafe-sdk` 0.6.0 code or executed offline by the SDK probe, or a primary source confirmed by the adversarial check.
- **[D]** documented by the vendor but not verified live. No Jev or Alpaca API key was used anywhere in this research.
- **[I]** inferred, or our own design recommendation.
- **[U]** unknown.

Where docs and installed SDK code disagree, the installed code wins.

---

## 0. Read this first

1. **[V]** The only inference call is `client.system_one(state, questions, ...)` in `typesafe-sdk==0.6.0`. There is no `ask()` or `evaluate()`. The HTTP layer is `httpx2` and models are `msgspec` structs; `httpx` and `pydantic` are not used.
2. **[V]** Jev's training cutoff is not published anywhere. Jev launched on 2026-09-15, so every historical backtest date may be contaminated. Forward paper trading is the only clean test of Jev's skill. Historical backtests validate the engine, risk and fill logic and provide leakage diagnostics only.
3. **[V]** The vendor documents Jev as weak at arithmetic, counting, numeric comparison and date ordering or windows. Large irrelevant state reduces accuracy, and adversarial text in state can steer answers. All options maths, date logic, sizing and risk limits therefore stay in code, and Jev receives pre-computed named buckets.
4. **[V]** There is no seed or temperature parameter. Run-to-run noise is documented, with a caveat:
   - Noise on byte-identical input was measured only on jev-1.12.
   - The jev-1.13.0 tests added a random `uid` field to the state on each call.
   - Identical-input noise on jev-1.13.0 is therefore unmeasured.
   - A content-addressed decision cache is the only way to make backtests reproducible.
5. **[V]** API access is waitlisted. Everything up to live inference can be built and tested offline with `httpx2.MockTransport` and a dummy key.
6. **[V]** Recommended broker: Alpaca paper with `alpaca-py==0.44.0`. Do not go below 0.43.5; `get_account()` breaks on older versions after Alpaca's 2026-07-06 API change.
7. **[V]** No free source has historical intraday option quotes. The best $0 source of end-of-day (EOD) chains is a preservation mirror of uncertain provenance.
   - It covers SPY and IWM 2008 to 2025-12, and QQQ 2011 to 2025-12.
   - ThetaData Free EOD is the official cross-check and the 2026 gap-fill.
8. **[V]** Legal terms in the TypeSafe Master Customer Agreement (MCA):
   - 2.3(f) forbids publishing benchmarks or performance information about the service.
   - 2.3(b) forbids distilling or imitating outputs, so no local surrogate model trained on Jev outputs.
   - No clause restricting trading or financial use was found.
9. **[V]** FRED's legal terms appear to prohibit using the FRED API for ML/LLM development and for caching or archiving. Avoid FRED in v1 (see sections 6 and 9).
10. **[I]** Frame Jev as a converter from text and state to calibrated judgements, plus a verifier. It is not a price forecaster, and nothing in the vendor docs claims predictive market skill.
    - On TypeSafe's own non-financial evals, Jev agrees with a frontier-model consensus 67.8% of the time.

---

## 1. Jev API cheat-sheet

### 1.1 Package

- **[V]** Install with `uv add "typesafe-sdk==0.6.0"` and import as `typesafe_sdk`. It is MIT-licensed, needs Python >=3.10, and was released 2026-09-15.
- **[V]** Dependencies are `httpx2>=2.0.0`, `msgspec>=0.21.1`, `tenacity>=9.0.0` and `typing-extensions>=4.13.0`.
- **[V]** Pin the version exactly. It is a 0.x package, and 0.6.0 shipped a breaking change one day after the first public release: `Score.criteria` is now an ordered list and no longer an int-keyed dict.
- **[V]** A probe venv already exists at `/home/tyler/venvs/jev-probe` (CPython 3.12.3). The wheel sha256 matched PyPI, and all probing was offline.
- **[V]** Do not install `typesafe-client`. It is the dead preview client, and `/preview/evaluation` no longer works.
- **[V]** Do not install `cooksafe`. It is a cookbook helper that needs `--extra-index-url https://pypi.typesafe.ai/` and pulls in `typesafe-client`. The extra index is an avoidable supply-chain risk, so write your own cache.
- **[V]** Custom transports, clients and timeouts must be `httpx2` types: `httpx2.Client`, `httpx2.AsyncClient`, `httpx2.BaseTransport`, `httpx2.Timeout`, `httpx2.MockTransport`.

### 1.2 Client and method signatures [V]

```python
class TypeSafeClient:
    def __init__(self, *, api_key: str | None = None, model: str | None = None,
                 retry: RetryPolicy | None = None, timeout: float | httpx2.Timeout | None = None,
                 headers: Mapping[str, str] | None = None, transport: httpx2.BaseTransport | None = None,
                 http_client: httpx2.Client | None = None, base_url: str | None = None) -> None
    def system_one(self, state: JSONContent, questions: Mapping[str, Question], *,
                   model: str | None = None, retry: RetryPolicy | None = None,
                   timeout: float | httpx2.Timeout | None = None,
                   extra_headers: Mapping[str, str] | None = None,
                   extra_body: Mapping[str, JSONValue | None] | None = None) -> SystemOneResponse
    # .models.list(*, retry=None, timeout=None, extra_headers=None) -> ListModelsResponse ; .close() ; context manager
# AsyncTypeSafeClient: identical, with httpx2.AsyncBaseTransport / httpx2.AsyncClient, `await client.system_one(...)`, `await client.aclose()`, `async with`.
```

- All `__init__` arguments are keyword-only. In `system_one`, `state` and `questions` may be positional.
- Passing both `transport` and `http_client` raises `ValueError`. `close()` also closes an `http_client` you supplied.
- A missing key raises `TypeSafeError("No API key was provided. ...")` at construction.
- Pitfall: `api_key=""` does not raise. The client sends `Bearer ` and fails at request time. Never write `os.environ.get("TYPESAFE_API_KEY", "")`. The same applies to `model=""`.
- `extra_body` is shallow-merged last and can override `state`, `model` and `questions`.
  - `beam_width` and `weight` in the docs are illustrative only and are not in the OpenAPI schema. Do not use them.

Environment variables and constants (`typesafe_sdk.constants`) **[V]**:

| Env var | Default / constant | Notes |
|---|---|---|
| `TYPESAFE_API_KEY` | required, no default | Explicit arguments beat env vars; blank env values are ignored |
| `TYPESAFE_BASE_URL` | `https://api.typesafe.ai` | |
| `TYPESAFE_DEFAULT_MODEL` | `jev-latest` | Always pass the pinned model explicitly instead |
| `TYPESAFE_LOG_LEVEL` | one of debug, info, warn, warning, error, off | Applied once at import |
| — | `DEFAULT_TIMEOUT = 10.0` | Seconds per HTTP operation |

`debug` logs full request and response bodies unredacted, which would include positions and account state. Never use it where logs are shared.

### 1.3 Question construction [V]

```python
Noul(instructions=..., criteria={"true": "...", "false": "..."})    # criteria optional
Choice(instructions=..., criteria={"label": "description" | None, ...})  # criteria required
Score(instructions=..., criteria=["level0 description", "level1", ...])  # ordered list, level index from 0
```

- These are keyword-only msgspec structs. `Noul("text")` raises `TypeError`.
- `instructions` and every criteria value accept a string, object or array.
- An equivalent raw-dict form exists: `{"type": "noul"|"choice"|"score", "instructions": ..., "criteria": ...}`.
  - It must be a real `dict`, not another `Mapping`.
  - Dicts and question objects can be mixed in one request.
- **[I]** Use the raw-dict form in our code so that questions serialise canonically for the cache key.
- Client-side validation is minimal:
  - It rejects empty questions.
  - It rejects a Score with empty criteria.
  - It rejects a dict question without `type`, or a choice or score dict without `criteria`.
  - Nothing else is validated at construction. `Score(criteria={0: "a"})` constructs without error and fails only at the server.
- The question key (your id) is not sent to the model **[V, api.md]**. All meaning must be in `instructions` and `criteria`.
- Always pass `instructions`. api.md marks it required, while OpenAPI and the SDK make it optional. Server behaviour without it is **[U]**.
- Documented limits **[D]** are enforced server-side, not by the SDK:
  - A Choice takes at most 255 options.
  - A Score takes 2 to 10 levels. A cookbook reports that 11 levels returns a server error.

### 1.4 Reading the response: probabilities versus confidence [V]

| Type | Access | Fields |
|---|---|---|
| Noul | `resp.nouls[k]` | `.noul`, a float P(yes). It has no confidence and no probabilities. |
| Choice | `resp.choices[k]` | `.choice` (the highest-probability label), `.confidence` (float), `.probabilities` (`dict[str, float]` over all labels). |
| Score | `resp.scores[k]` | `.score` (probability-weighted mean of level indices, can be fractional), `.confidence`, `.probabilities` (`dict[int, float]`), `.legend` (`dict[int, ...]`). |

- Score maps have int keys in the SDK. `probabilities["2"]` raises `KeyError`; use `[2]`. The wire format has string keys.
- Other response fields:
  - `resp.answers[k]` is the untyped union.
  - `resp.model` is the versioned id that answered.
  - `resp.usage.input_tokens` and `.output_tokens` are `int | None`, so tolerate `None`.
  - `resp.raw_http_response` is the underlying `httpx2.Response`.
  - `resp.request_id` raises `TypeSafeError` if the `x-typesafe-request-id` header is absent. On errors, `error.request_id` returns `None` instead.
- An answer of an unknown type is dropped from `resp.answers` with a warning log.
- A missing required field raises `TypeSafeAPIResponseValidationError` with `.field_path`, for example `answers.frustration.probabilities`.
- The quickstart's sample response omits the Score `probabilities` field and would fail decoding. Do not build test fixtures from it.
- **[V, OpenAPI]** Probabilities "sum to approximately 1". Renormalise before doing statistics, and never assert an exact sum.
- **[V]** `confidence` is a server-computed statistic of the distribution.
  - Its formula is undocumented.
  - It changed at v1; the migration guide calls it "a new computation". The preview version was 1 minus normalised entropy.
  - **[I]** TypeSafe's LLM adapter repo uses two formulas:
    - For Choice it uses `(p_max - 1/K)/(1 - 1/K)`.
    - For Score it uses a distance-from-mode formula.
    - These fit some docs examples to within 0.01 but not the examples on api.md or the quickstart.
- **Rule:** always persist the full `probabilities`.
  - Gate on statistics we compute ourselves: `p_top`, `margin = p1 - p2`, and normalised entropy.
  - Log the server `confidence` for analysis only.
- For typed access use `resp.choices[...]` and `resp.scores[...]`. `resp.answers[k].confidence` works at runtime but fails strict type checkers.
- To persist a response:
  - `resp.raw_http_response.json()` keeps the wire-format string keys and is the most stable across SDK versions.
  - `msgspec.to_builtins(resp)` also works.
  - Responses are picklable from 0.6.0.

### 1.5 Wire format [V: payload captured offline, matches api.md]

Send `POST {base_url}/v1/systemone` with headers `Authorization: Bearer <key>` and `Content-Type: application/json`. `GET /v1/models` lists models. An OpenAPI 3.1 spec is at `https://api.typesafe.ai/openapi.json`.

```json
{"state": "<str|object|array>", "model": "jev-1.13.0",
 "questions": {"<key>": {"type": "noul", "instructions": "...", "criteria": {"true": "...", "false": null}},
               "<key>": {"type": "choice", "criteria": {"<label>": "desc or null"}, "instructions": "..."},
               "<key>": {"type": "score", "criteria": ["level0", "level1"], "instructions": "..."}}}
```
```json
{"model": "jev-latest",
 "answers": {"a": {"type": "noul", "noul": 0.92},
             "b": {"type": "choice", "choice": "technical", "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07}, "confidence": 0.82},
             "c": {"type": "score", "score": 1.6, "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"}, "probabilities": {"0": 0.05, "1": 0.3, "2": 0.65}, "confidence": 0.78}},
 "usage": {"input_tokens": 312, "output_tokens": 48}}
```

State cannot be `None`, although inner values may be. State is text only. Each request carries exactly one state.

### 1.6 Sync example [V: executed end to end offline against `httpx2.MockTransport`, not against the live API]

This example demonstrates the API shape only. Its question wording and its state break the rules in section 2: it has a raw symbol, raw numbers, vague Score levels and a date-comparison Noul. Do not copy the wording.

```python
import os
from typesafe_sdk import (
    Choice, Noul, Score, RetryPolicy, SystemOneResponse, TypeSafeClient,
    TypeSafeAPIConnectionError, TypeSafeAPIError, TypeSafeError, TypeSafeRateLimitError,
)

QUESTIONS = {
    "stance": Choice(
        instructions="Given this market snapshot, which options stance fits best?",
        criteria={
            "bullish": "Conditions favor long calls or bull spreads",
            "bearish": "Conditions favor long puts or bear spreads",
            "neutral": "No directional edge; stay flat",
        },
    ),
    "conviction": Score(
        instructions="How strong is the setup?",
        criteria=["weak or conflicting signals", "moderate setup", "strong, aligned signals"],  # ordered list, level 0..2
    ),
    "event_risk": Noul(instructions="Is there a scheduled earnings or macro event before expiry?"),
}
STATE = {"symbol": "SPY", "iv_rank": 0.62, "trend_20d": "up", "days_to_expiry": 21}  # str | dict | list

def read(resp: SystemOneResponse) -> dict:
    stance = resp.choices["stance"]          # ChoiceAnswer
    conviction = resp.scores["conviction"]   # ScoreAnswer
    event = resp.nouls["event_risk"]         # NoulAnswer - has ONLY .noul
    return {
        "model": resp.model,                                 # versioned id that answered
        "stance": stance.choice,                             # str
        "stance_conf": stance.confidence,                    # float 0..1
        "p_bullish": stance.probabilities["bullish"],        # dict[str, float]
        "conviction": conviction.score,                      # float, 0..2, may be fractional
        "conviction_conf": conviction.confidence,
        "p_top_level": conviction.probabilities[2],          # dict[int, float] -> INT keys, not "2"
        "legend_top": conviction.legend[2],
        "p_event": event.noul,                               # P(yes); no confidence on Noul
        "tokens_in": resp.usage.input_tokens,                # int | None
    }

with TypeSafeClient(                       # all args keyword-only; api_key falls back to env TYPESAFE_API_KEY
    model="jev-1.13.0",                    # pin for reproducible backtests; default is "jev-latest"
    timeout=10.0,                          # == DEFAULT_TIMEOUT, per HTTP operation
    retry=RetryPolicy(max_retries=2, timeout=30.0),   # == defaults; RetryPolicy(max_retries=0) disables
) as client:
    try:
        decision = read(client.system_one(state=STATE, questions=QUESTIONS))
    except TypeSafeRateLimitError as e:        # 429 after retries
        print("rate limited; retry_after_ms =", e.retry_after_ms)
    except TypeSafeAPIError as e:              # any other non-2xx, or 2xx with invalid body
        print("api error", e.status, e.request_id, e.body)
    except TypeSafeAPIConnectionError as e:    # network; includes TypeSafeAPITimeoutError
        print("network/timeout", e)
    except TypeSafeError as e:                 # missing key, empty questions, bad timeout, unencodable body
        print("sdk error", e)
```

### 1.7 Async example [V: executed offline; reuses QUESTIONS, STATE and read()]

```python
import asyncio
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeError

async def decide(client: AsyncTypeSafeClient, state: dict) -> dict:
    resp = await client.system_one(        # state, questions positional-or-keyword; rest keyword-only
        state=state,
        questions=QUESTIONS,
        model="jev-1.13.0",                # per-call override of client model
        timeout=5.0,                       # per-call override, seconds or httpx2.Timeout
        # retry=RetryPolicy(max_retries=0),          # per-call override
        # extra_headers={"X-Run-Id": "bt-001"},      # cannot override Authorization/Accept/User-Agent/X-TypeSafe-*
    )
    return read(resp)

async def main() -> None:
    async with AsyncTypeSafeClient() as client:      # env TYPESAFE_API_KEY
        snapshots = [STATE, {**STATE, "symbol": "QQQ"}]
        results = await asyncio.gather(*(decide(client, s) for s in snapshots), return_exceptions=True)
        for r in results:
            print(r if not isinstance(r, TypeSafeError) else f"failed: {r}")
        models = await client.models.list()          # ListModelsResponse
        for m in models.models:                      # tuple[ModelMetadata, ...]: name, description, release_date
            print(m.name, m.release_date, m.description)

asyncio.run(main())
```

**[I]** The SDK has no rate limiter. Bound concurrency yourself with an `asyncio.Semaphore` and a token bucket below 20 requests per second.

### 1.8 Offline test seam

The `transport=httpx2.MockTransport(...)` pattern is **[V]**; it is the seam the SDK's own tests use. The fixture below is composed and was not executed.

```python
import httpx2
from typesafe_sdk import TypeSafeClient

FAKE = {"model": "jev-1.13.0",
        "answers": {"stance": {"type": "choice", "choice": "neutral",
                               "probabilities": {"bullish": 0.2, "bearish": 0.1, "neutral": 0.7}, "confidence": 0.55},
                    "conviction": {"type": "score", "score": 0.4, "legend": {"0": "a", "1": "b", "2": "c"},
                                   "probabilities": {"0": 0.7, "1": 0.2, "2": 0.1}, "confidence": 0.6},   # probabilities REQUIRED
                    "event_risk": {"type": "noul", "noul": 0.12}},
        "usage": {"input_tokens": 300, "output_tokens": 20}}
client = TypeSafeClient(api_key="dummy", transport=httpx2.MockTransport(
    lambda req: httpx2.Response(200, json=FAKE, headers={"x-typesafe-request-id": "req_test"})))
```

If the mock omits the request-id header, `resp.request_id` raises. CI without secrets must pass a dummy key, because construction raises without one.

### 1.9 Batching, limits, pricing and rate limits

- **[V docs]** Put every question that shares a state into one request.
  - Questions are evaluated in parallel and independently, and one answer never becomes context for another.
  - Extra questions add almost no latency.
  - There is no multi-state endpoint.
  - A second request is justified only when code cannot build it without the first answer.
  - The batch-independence measurement (one call of 13 questions versus 13 single calls) was done on jev-1.12, where batching was about 12x cheaper.
- **Token budget: the docs conflict [V]**
  - primitives.md says about 32,000 tokens are shared by state plus questions.
  - The jev-1.13 jaggedness page says 64k in total, and 32k for the state plus the longest question.
  - Design to at most 32k in total, and target a state of at most about 4k tokens.
  - There is no tokenizer or token-count endpoint, and the over-limit error is **[U]**.
  - Calibrate characters per token from `usage.input_tokens`.
  - **[I]** Fixed overhead is about 250 to 300 tokens per request, plus about 50 to 70 per question.
- **Pricing [V docs]**
  - $0.042 per million input tokens; output tokens are free.
  - A 2k-token request costs about $0.000084, so 100k decisions cost about $8.
  - Prepaid credits are non-refundable and expire after 12 months.
- **Rate limits [V docs]**
  - 1,200 requests per minute and 250,000 tokens per second; these "can change without notice".
  - Exceeding either returns 429; overload returns 529.
- **Latency [D]**
  - About 100 ms typical; 70 to 500 ms claimed.
  - The service is hosted on the US West Coast, and no percentiles are published.

### 1.10 Errors, retries and timeouts [V]

```
TypeSafeError
 ├─ TypeSafeAPIError(.status .body .headers .endpoint .request_id)
 │    ├─ TypeSafeBadRequestError 400 · TypeSafeAuthenticationError 401 · TypeSafePermissionDeniedError 403
 │    ├─ TypeSafeNotFoundError 404 · TypeSafeUnprocessableEntityError 422
 │    ├─ TypeSafeRateLimitError 429 (.retry_after_ms) · TypeSafeInternalServerError >=500 (incl. 529)
 │    └─ TypeSafeAPIResponseValidationError (2xx body fails schema; .field_path)
 └─ TypeSafeAPIConnectionError (NOT a TypeSafeAPIError)
      └─ TypeSafeAPITimeoutError (.timeout)
```

- Catch in this order:
  1. RateLimit.
  2. Authentication and PermissionDenied, which are fatal configuration errors.
  3. UnprocessableEntity, which means our bug; log `error.body`.
  4. ResponseValidation.
  5. APIError.
  6. APITimeout.
  7. APIConnection.
  8. TypeSafeError.
- Always log `error.status` and `error.request_id`.
- Observed live on 2026-09-17:
  - No Authorization header returns 403 with `{"detail":{"error_type":"authentication_error","message":...}}`.
  - An invalid key returns 401.
  - The SDK always sends `Bearer`, so a bad key surfaces as `TypeSafeAuthenticationError`.
  - What a valid but waitlisted key returns is **[U]**.
  - The 422 body is FastAPI-style: `detail: [{loc, msg, type}]`.
- `RetryPolicy` defaults:

| Field | Default |
|---|---|
| `max_retries` | 2 |
| `backoff_initial` | 0.5 |
| `backoff_max` | 5.0 |
| `backoff_jitter` | 0.25 |
| `http_statuses` | {408, 429, 500-599} |
| `respect_retry_after` | True |
| `api_connection_error` | True |
| `api_timeout_error` | True |
| `timeout` | 30.0 (total budget per SDK call) |

- The SDK honours `retry-after-ms` first, then `retry-after`. The original exception is re-raised, not a tenacity `RetryError`.
- **[I from source]** The 30-second budget does not cancel an in-flight attempt. With a long per-request timeout, such as the cookbooks' 120 seconds, a slow first attempt gets no retry. Keep `timeout` and `RetryPolicy(timeout=)` consistent.
- For the trading loop, use short explicit timeouts and treat any error as "no decision", which means no new trade.
- **Encoding [V correction]**
  - msgspec silently encodes `Decimal` and `datetime` as strings.
  - It silently encodes NaN and infinity as `null`.
  - numpy and pandas scalars are expected to raise `TypeSafeError` **[I]**.
  - Sanitise state to plain `str`, `int`, `bool`, `None`, `list` and `dict`, and reject non-finite numbers ourselves.
  - `json.dumps(..., allow_nan=False)` in the cache canonicaliser doubles as a guard.

### 1.11 Model ids [V docs]

- The versioned id is `jev-1.13.0`. The aliases `jev-latest` (the SDK default) and `jev-preview` both currently point to it.
- Aliases move when a new release ships. Pin `jev-1.13.0` and fail the run if `resp.model != "jev-1.13.0"`.
  - Cookbook logs show that the response carries the versioned id even when an alias is requested.
  - The api.md examples show the alias echoed back instead. Verify with the first live call.
- `jev`, `jev-1.13` and `jev-1.12` appear in docs examples but are not documented ids. Do not use them.
- `GET /v1/models` currently lists aliases only. Its `release_date` field may be the alias's date.
- No deprecation policy exists. jev-1.12 was de-listed within weeks.

---

## 2. Question-design rules and jagged edges that matter for trading state

### 2.1 Rules (all [V] from vendor docs unless tagged)

1. **Ask atomic questions.** Each question is one narrow judgement that a knowledgeable person could make in seconds.
   - Split multi-factor judgements into several questions and recombine them in code.
   - "Analyse and decide the best action" is the canonical bad question.
   - A bounded action selection as a Choice is allowed, because atomic does not mean one fact. Cross-check it in code.
2. **Questions are independent.** One answer never feeds another. Fan out every possibly-needed question in one call, including conditional ones. Phrase conditional premises explicitly, for example "If this is X, which kind is it?". Code ignores the branches that do not apply.
3. **Write the whole question in `instructions`.**
   - Reference state with backticked paths such as `` `vol_surface.iv_vs_realized` ``.
   - Word questions literally and directly, with no double negatives and no multi-hop phrasing.
   - Jev answers the question you wrote, not the one you meant, so put boundary cases in the criteria.
4. **Noul.**
   - Phrase it so that a high value means yes.
   - Keep `criteria.true` and `criteria.false` aligned with the instruction, because contradictions reduce accuracy.
   - A value of 0.5 means equal odds, not "medium".
   - For multi-label situations use one Noul per label.
   - Vague adjectives such as "strong" make the probability hard to interpret. Define the condition precisely, or use a Score.
5. **Choice.**
   - Both option names and descriptions are sent to the model.
   - Always include a no-match option (`unclear`, `none` or `no_trade`).
   - Use option keys that code can consume directly.
   - Add a companion "is this stated at all?" Noul when an argument is optional. Without it the Choice will confidently name something.
6. **Score.**
   - Measure one dimension per Score.
   - Every level is judged in isolation; the model sees neither level numbers nor neighbouring levels.
   - Levels must therefore describe standalone situations, not degrees.
   - Number-only levels fail: the docs example scored 0.57 at confidence 0.35 with numeric levels, against 0.0 at confidence 1.0 with descriptive ones.
   - Normalise with `score / (len(criteria) - 1)`.
   - Never interpolate a magnitude from a Score, because the levels "are weak in numerical calibration". Threshold checks and ranking are fine.
   - The SDK reference examples (`["low", "medium", "high"]`) are anti-patterns and should not be copied.
7. **Structured instructions and criteria** may be objects with fields such as `question`, `focus`, `what`, `not_for`, `examples`, `signals`, `compare` and `inspect`.
   - Use structure only where two options are being confused.
   - Examples help only if they resemble real inputs.
   - Keep atomic questions short.
8. **Composition.**
   - Use weighted sums for preferences that compensate one another.
   - Use separate any-flag gates, combined with max and not mean, for vetoes.
   - For the confidence of a multi-judgement call, take the minimum over the judgements, not the product.
   - Changing weights does not require re-running inference, so cached answers can be reused for weight sweeps.
9. **Thresholds.**
   - Scale them to the stakes, tune them on our own data, and start conservative.
   - Pin the model version.
   - Most cookbook thresholds were measured on jev-1.12, so treat them as examples.
10. **[I]** Keep all questions and thresholds in one reviewable module. A wording change is a new experiment and a new cache namespace.

### 2.2 Jagged edges and what they mean for trading state

| Documented weakness [V] | Consequence for the bot |
|---|---|
| "Jev is not a calculator"; it counts unreliably; it "cannot reliably judge whether two values are near each other" | Compute greeks, IV rank, IV minus realised vol, expected move, P&L percentages, distances to strikes, sizing and margin in code. Pass "either the computed number or a named bucket". |
| "reads dates as text, not as ordered quantities"; it is worse with "quarters, settlement windows" | Never ask whether an event falls before expiry. Compute days to expiry, days to event and window membership in code, and pass facts such as `"inside_holding_window": [...]`. Use relative ages only. |
| Context rot: "Accuracy falls as the state grows with content unrelated to the decision" | Keep a lean state of about 2 to 4k tokens. Use separate entry and position-management requests with different states. Cap news items. |
| Adversarial text in state "can move the answer" | News and headline text is an injection surface. Text-derived signals may veto or rank but never raise limits. Sanitise the text. No single text-derived answer triggers a trade without confirmation from market-data rules in code. |
| Literal reading; indirection reduces accuracy | Use single-hop questions with exact conditions in the criteria. |
| Score levels are weak in numerical calibration | Use Score only as a rank, a threshold or a sizing tier. Never map it to an expected move size. |
| No rationale output | The audit trail is state, questions and probabilities only. Log everything. |
| Small run-to-run noise. jev-1.13.0 Noul probability std is about 0.010, with one question spanning 0.43 to 0.53. On Choices the top label flipped on 2 of 8 borderline questions. | Use abstention bands and hysteresis, never a bare 0.5 cut. Keep thresholds off the 0.01 grid **[I: outputs look quantised to about 0.01, and Nouls look clipped to 0.01–0.99]**. |
| Calibration claim covers answer correctness, "across groups of predictions". Nothing is claimed for market forecasting, and no ECE or Brier figures are published. | "Jev is calibrated for market outcomes" is a hypothesis we must test (section 7.7). |
| The docs say nothing on key order, option-order bias, or tabular and time-series state | Do not send OHLC arrays or chains; summarise them in code. Test option-order and field-order sensitivity ourselves. |

---

## 3. Recommended decision architecture [I, consistent with the vendor's "Keep control flow, deterministic rules, and side effects in code"]

### 3.1 Pipeline

This one codebase serves both backtest and paper trading. Only the `Broker` adapter differs.

```
Clock → DataView(as_of) [hard point-in-time gate]
      → CandidateGenerator (code: liquidity filters, allowed structures, strikes/expiries from chain)
      → StateBuilder (pure function → canonical JSON; buckets computed in code)
      → JevDecider (adapter over SDK; decision cache; record/replay/refresh)
      → DecisionRules (code: composite score, veto gates, abstention, hysteresis)
      → RiskEngine (code: max loss, BP/margin, exposure caps, event blackout, expiry policy, kill switch) ← always runs AFTER Jev; nothing Jev says can bypass it
      → OrderIntents → Broker {SimBroker | AlpacaPaperBroker}
      → Ledger (hash-chained) + provenance sidecar + outcome resolver (for calibration)
```

Jev never outputs a quantity, strike, price or limit. Its outputs do only two things:
- They select from closed enums that code already implements.
- They provide probabilities that pass or fail thresholds owned by code.

Hard exits always run first and always win. These are the profit target, the stop, the days-to-expiry exit and the event exit.

If Jev fails, fail closed:
- Trigger on any `TypeSafeError`, a timeout, a model-id mismatch, or a cache miss in replay mode.
- Open no new positions.
- Keep managing existing risk purely in code.
- Do not switch to a different model silently.

### 3.2 `JevDecider` adapter

- Hide the SDK behind `decide(state: dict, questions: dict[str, dict]) -> DecisionResult`.
- `DecisionResult` is a set of plain dataclasses holding:
  - Noul values;
  - Choice probabilities;
  - Score probabilities;
  - the server confidence;
  - our own statistics;
  - the model, request id, input tokens and latency.
- Implementations:
  - `LiveJev`;
  - `ReplayJev`, which reads the cache only;
  - `MockJev`, a deterministic or rule-based stand-in that doubles as the Jev-off baseline;
  - optionally `AdapterLLM`, using `typesafe-ai/system-one-adapter-python` (MIT).
    - It is a drop-in replacement backed by OpenAI or Anthropic.
    - It is useful as a control arm with a known training cutoff.
    - It is never a silent fallback for live decisions.

### 3.3 Two request types, one call each

- **Entry request.** The state has no `position` block. It fans out all the regime, volatility, fit, veto and sizing questions.
- **Management request.** The state has `position` and `news.since_entry` but no `candidate`.

Starter question set for v1:
- Give every question full `instructions` with backticked state paths.
- Every Score level must describe a situation.
- Every Choice must include a no-match option.

| id | type | purpose | used as |
|---|---|---|---|
| `regime.market` | Choice (trending_up_calm, trending_up_volatile, range_bound_calm, range_bound_volatile, orderly_downtrend, disorderly_selloff, unclear_or_transition) | Gates which strategy families are allowed | gate |
| `under.trend_strength` | Score, 4 levels | Trend strength regardless of direction | composite |
| `under.direction` | Choice (bullish, bearish, neutral_range, conflicting_signals) | `P(bull) - P(bear)` as a signed feature; require a margin | composite + gate |
| `under.vol_expanding` | Noul | Realised volatility is expanding | composite |
| `under.stretched` | Noul | Mean-reversion stretch | composite |
| `vol.richness` | Score, 4 levels (cheap / fair / rich / extreme, written as situations) | Rank or threshold only | composite |
| `vol.explained_by_event` | Noul | Whether an event listed in state explains the elevated IV | veto for premium selling |
| `vol.stance` | Choice (sell_premium, buy_premium, avoid_vol_exposure, no_trade) | Volatility stance | gate |
| `fit.structure_family` | Choice over the structure names implemented in code, plus `no_trade` | The least atomic question; cross-check against a deterministic mapping of direction × vol.stance, and treat disagreement as no_trade | gate |
| `fit.trend_supports_candidate`, `fit.vol_supports_candidate` | Noul | Consistency of the code-built candidate with the state | veto, inverted |
| `risk.environment` | Score, 4 levels (benign to hostile) | Rounded level maps to a size multiplier of 1.0, 0.75, 0.5 or 0 | sizing tier |
| `risk.pending_binary_in_news`, `risk.adverse_news` | Noul, bad = TRUE | News vetoes. Jev is never asked whether timing falls before expiry. | veto |
| `risk.correlated_exposure` | Noul | Fuzzy same-theme exposure. Exact ticker duplicates are checked in code. | veto |
| `dq.state_sufficient` | Noul | Data-quality gate | gate |
| mgmt: `pos.thesis_invalidated`, `pos.adverse_news_since_entry` | Noul | Management signals | composite |
| mgmt: `pos.short_strike_threat` | Score, 4 levels | Distances to the strike are computed in code and bucketed | composite |
| mgmt: `pos.action` | Choice (hold, take_profit, close_to_cut_loss, roll_out_in_time, reduce_size, unclear) | Used only inside the discretionary zone. `unclear` or low peakedness resolves to the risk-reducing default. | gate |

### 3.4 Composite scoring, vetoes and abstention

All thresholds below are starting points to be tuned walk-forward. None of them is a finding.

- **Opportunity score (ranking only).** `S = Σ w_i·x_i`.
  - Each `x_i` is a normalised Score (`score/(K-1)`), a Noul, or a signed Choice feature.
  - Weights live in a versioned config file.
  - Use S to rank candidates and pick the top K. Never use it to size continuously.
- **Risk veto (max-style, any flag).**
  - Each veto Noul is three-valued.
    - A value above 0.70 is a hard veto.
    - A value from 0.30 to 0.70 inclusive is uncertain, which also blocks new entries.
    - Only a value below 0.30 clears.
  - Never average vetoes.
  - Holistic "is this OK?" questions give mushy answers and are not used for gating. In the vendor's cascade example the holistic judge scored 0.56 while the per-field checks scored 0.95.
- **Choice gates.**
  - Compute `p_top` and `margin = p1 - p2` from renormalised probabilities.
  - To enter, require `p_top ≥ 0.60` and `margin ≥ 0.25`.
    - The 0.60 rule is from the vendor cookbook, where it raised repeat agreement from 90.8% to 99.2% at the cost of 25.8% abstentions.
    - The 0.25 margin is ours.
  - Set thresholds per question, because any peakedness statistic depends on the number of options.
- **Risk asymmetry.**
  - Opening a new position needs the strictest bar.
  - Holding needs a looser one.
  - Closing to reduce risk needs the loosest.
  - Anything below the floor becomes `no_trade` plus a journal entry.
- **Hysteresis for held positions.** For example, enter above 0.70 and flip only below 0.45, so that a probability hovering at a boundary cannot churn the position.
- **Sizing tier.**
  - Take the minimum of three things: the tier from S, the tier from the peakedness of direction and vol.stance, and the multiplier from `risk.environment`.
  - Hard caps in RiskEngine then apply.
  - Never ask "how big should I trade?".

### 3.5 Self-consistency

- The vendor's prescription is a single call plus an abstention band, not majority voting **[V]**.
- **[I] Step 0, mandatory before choosing thresholds.**
  - On jev-1.13.0, send at least 20 byte-identical repeats over a few hundred representative states.
  - Report the per-question probability standard deviation and the decision flip rate.
  - Also test sensitivity to option order, state-field order and an irrelevant field.
  - Keep this as a regression suite and rerun it on any model change.
- **[I] Runtime policy, identical in backtest and paper.**
  - For entries that pass on sample 0, draw K = 3 samples with `sample_index` 0 to 2. The cost is negligible.
  - Require all K samples to land on the same side of every gate; otherwise do not trade.
  - Log the dispersion as a metric.
  - Every sample is cached, so replays are bit-reproducible.

### 3.6 Decision cache (the experiment's immutable log)

- **Key.**
  - `sha256(canonical_json({"v":1,"model":"jev-1.13.0","state":…,"question":{type,instructions,criteria},"sample":i}))`
  - One entry per question. This is valid because questions are evaluated independently and the question id is not sent to the model **[V docs]**; exclude the id from the key.
  - On a miss, send only the missing questions in one request.
- **Canonical JSON.**
  - `json.dumps(sort_keys=True, separators=(",",":"), ensure_ascii=False, allow_nan=False)`.
  - State contains only strings, ints, bools, null, lists and dicts. Floats are excluded by pre-bucketing or rounding to int.
  - State contains no wall-clock time, uid or random fields, because an irrelevant field may itself shift answers.
- **Value.**
  - The raw answer JSON, from `raw_http_response.json()["answers"][k]`.
  - `requested_model`, `response.model`, `request_id` and `usage`.
  - Latency, SDK version and `created_at`.
- **Storage.**
  - SQLite in WAL mode, or append-only Parquet.
  - Writes are atomic so runs can resume.
  - Never evict entries.
  - Put a manifest hash of the cache in every run report.
- **Modes.**
  - `record` calls the API on a miss.
  - `replay` treats a miss as a hard error and makes no network calls.
  - `refresh` starts a new namespace.
- Do not copy the cookbook's cache pattern. Its fingerprint is a truncated 12-hex hash of the text only, and its `JsonCache` rewrites the whole file on each miss.
- Paper-trading decisions go through the same cache, and each day's snapshot is archived so the day can be replayed in the backtester.

---

## 4. Broker (paper): Alpaca with `alpaca-py==0.44.0`

### 4.1 Why Alpaca, and why not the others [V]

**Alpaca**
- The paper-only account is free, global and needs only an email. It has a $100k default balance.
- Options are enabled by default on paper. The level granted is **[U]**, so check it at startup.
- Access is plain REST and WebSocket with API keys, which is simple on headless WSL.
- Level 3 allows `order_class="mleg"` with 2 to 4 legs, all covered, filled as a unit.
- One call returns the chain with quote, IV and greeks.

**Tradier sandbox**
- It needs a brokerage account.
- Greeks are "Not Available" in the sandbox.
- Quotes are delayed 15 minutes, there is no streaming, and the limit is 60 requests per minute.

**IBKR paper**
- It needs a funded Pro account.
- Login is through a GUI, and headless operation is unsupported.
- It needs daily restarts and a weekly 2FA login.
- The paper simulator lists "Limited combo trading" and no penny option fills.
- The IBC login automation project was retired and archived on 2026-09-01.

**Pinning**
- Pin `alpaca-py==0.44.0` exactly.
- It needs Python >=3.10,<4, and versions below 0.43.5 break.
- Also pin `websockets` in the lockfile, because a breaking streaming change is pending upstream.

### 4.2 Constraints to design around [V unless tagged]

**Order rules**
- Time in force is `day` or `gtc` only.
- Quantities are whole numbers; no notional amounts and no extended hours.
- Single-leg orders can be market, limit, stop or stop_limit.
- Multi-leg orders can be market or limit only.
- Market orders are accepted only from 9:30 to 16:00 ET.
- No uncovered short options and no equity-plus-option combos.
- Multi-leg ratios must have a GCD of 1.
- In multi-leg orders a positive limit price is a net debit and a negative one is a net credit.
- Send `position_intent` on every leg, because the server error table requires it.

**Order management**
- Cancel or replace the parent order only, never a leg. Match errors on the code and HTTP status (42210000 / 422), never on the message text.

**Price increments**
- Most contracts: $0.05 below $3 and $0.10 at or above $3.
- Penny-program classes: $0.01 below $3 and $0.05 at or above $3.
- SPY, QQQ and IWM quote in pennies at any price.
- A price with 3 or more decimals is rejected.
- The `ppind` flag is not in the SDK model, so read it from raw REST.

**v0.44.0 client-side validation**
- A multi-leg order needs `qty`, 2 to 4 unique legs, and type market or limit.
- A non-multi-leg order needs `symbol` and `side`.
- `LimitOrderRequest` needs `limit_price`.

**Data on the free Basic plan**
- Options quotes are an indicative feed: "not actual OPRA quotes", with trades delayed 15 minutes.
- Limits are 200 historical calls per minute and 200 WebSocket quote subscriptions.
- "Most users" get one stream connection per endpoint, so only one process may hold the option stream.
- Real OPRA data needs Algo Trader Plus at $99 per month. That plan does not add historical quotes.
- Greeks and IV can be `None`. They are absent for 0DTE, zero bid or ask, or deep-OTM non-convergence. Treat them as optional and keep our own Black-Scholes fallback.
- `get_option_chain` auto-paginates at 1000 per page. An unfiltered SPY chain costs several calls, so filter by expiry and strike.
- The SDK's `OptionsSnapshot` drops bars.
- `get_option_contracts` does not auto-paginate. Its default `expiration_date_lte` is the next weekend, so always pass explicit expiration bounds.

**Paper fills are optimistic**
- Orders are matched against "NBBO" with no size check.
- A limit order fills as soon as it is marketable.
- 10% of fills are random partial fills.
- There is no modelling of queue position, slippage, fees or dividends.
- Which quote source drives option paper fills is **[U]**, so test it empirically.
- Log the quote at submission against the fill price, and apply our own cost haircut to paper P&L.

**Expiry**
- From 3:30 pm ET on expiry day Alpaca evaluates expiring positions and stops accepting opening orders for them.
- Contracts at least $0.01 in the money are auto-exercised.
- Positions lacking buying power are sold out within 1 hour before expiry.
- Slightly out-of-the-money positions may also be liquidated.
- A user report (alpaca-py issue #774, unconfirmed by Alpaca) says the risk process closed only the short leg of a spread at 3:30 pm. Policy: close or roll everything well before 15:30 ET on the last trading day.
- Manual exercise is `POST /v2/positions/{id}/exercise`, exposed as `exercise_options_position`.
- The API reference now documents `POST /v2/positions/{symbol_or_contract_id}/do-not-exercise`, but the guide page still says to contact support. There is no SDK wrapper; verify it live.
- Assignments never arrive over the WebSocket, and paper non-trade activities (OPEXC, OPASN, OPEXP, OPTRD) sync the next day. Poll REST each morning and reconcile.

**Universe**
- US equity and ETF options only, all American-style. SPX and XSP index options are a Broker-API product enabled through a Customer Success Manager; assume they are unavailable.
- Every paper-tradable contract therefore carries early-assignment and dividend risk.

**Rate limit and retries**
- The Trading API limit is about 200 requests per minute per account (2022 source).
- SDK retry defaults are fixed: 3 attempts, 3 seconds, on 429 and 504. They are not configurable through the public constructors.
- Add our own token bucket at about 180 per minute.

**Streams**
- `TradingStream.run()` and `OptionDataStream.run()` each block, so run them in separate threads or tasks.
- In 0.44.0 stale-socket detection is opt-in; pass `data_timeout=60`.

### 4.3 Code

**Clients, chain and streams.** The names are **[V]** against the v0.44.0 source. The official notebook passes `url_override=` variables that are `None`; they are dropped here, and nothing was executed.

```python
import os
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.live.option import OptionDataStream
from alpaca.data.requests import OptionChainRequest, OptionSnapshotRequest

api_key, secret_key = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]   # PAPER keys only
trade_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)       # hard-code paper=True
option_historical_data_client = OptionHistoricalDataClient(api_key, secret_key)

acct = trade_client.get_account()   # has options_buying_power, options_approved_level, options_trading_level
option_historical_data_client.get_option_chain(OptionChainRequest(underlying_symbol="SPY"))  # Dict[str, OptionsSnapshot]

trade_stream_client = TradingStream(api_key, secret_key, paper=True)
async def trade_updates_handler(data): print(data)
trade_stream_client.subscribe_trade_updates(trade_updates_handler)
# trade_stream_client.run()   # blocks
```

**End-to-end script.** Composed by the researcher. It statically passes all v0.44.0 validators but was not executed. It is illustrative only:
- Steps 3 and 4 both buy `long_sym`.
- Steps 6a and 6b close it twice.

```python
# uv add "alpaca-py==0.44.0"
import os
from datetime import date, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, PositionIntent, ContractType
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest
from alpaca.data.enums import OptionsFeed

KEY, SECRET = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]  # PAPER keys

# 1) authenticate: paper=True -> https://paper-api.alpaca.markets ; data -> https://data.alpaca.markets (v1beta1)
trade = TradingClient(api_key=KEY, secret_key=SECRET, paper=True)
data = OptionHistoricalDataClient(api_key=KEY, secret_key=SECRET)
acct = trade.get_account()
assert (acct.options_trading_level or 0) >= 3, acct.options_trading_level

# 2) option chain with greeks/IV -> Dict[str, OptionsSnapshot]; greeks / implied_volatility / latest_quote may be None
chain = data.get_option_chain(OptionChainRequest(
    underlying_symbol="SPY",
    feed=OptionsFeed.INDICATIVE,            # free Basic plan; OptionsFeed.OPRA needs Algo Trader Plus
    type=ContractType.CALL,
    expiration_date_gte=date.today() + timedelta(days=21),
    expiration_date_lte=date.today() + timedelta(days=45),
))
rows = [(sym, s.latest_quote.bid_price, s.latest_quote.ask_price, s.implied_volatility, s.greeks.delta)
        for sym, s in chain.items() if s.latest_quote and s.greeks]

long_sym, short_sym = "SPY261016C00600000", "SPY261016C00610000"   # placeholders: choose from `chain`

# 3) single-leg LIMIT order (side is REQUIRED for non-mleg in 0.44.0; limit_price REQUIRED)
o1 = trade.submit_order(LimitOrderRequest(
    symbol=long_sym, qty=1, side=OrderSide.BUY, limit_price=5.00,
    time_in_force=TimeInForce.DAY, position_intent=PositionIntent.BUY_TO_OPEN,
))

# 4) multi-leg vertical (bull call debit spread) LIMIT order; limit_price > 0 = net debit, < 0 = net credit
o2 = trade.submit_order(LimitOrderRequest(
    qty=1, order_class=OrderClass.MLEG, time_in_force=TimeInForce.DAY, limit_price=3.00,
    legs=[
        OptionLegRequest(symbol=long_sym,  ratio_qty=1, side=OrderSide.BUY,  position_intent=PositionIntent.BUY_TO_OPEN),
        OptionLegRequest(symbol=short_sym, ratio_qty=1, side=OrderSide.SELL, position_intent=PositionIntent.SELL_TO_OPEN),
    ],
))

# 5) list positions (options have asset_class us_option; short legs have side short)
for p in trade.get_all_positions():
    print(p.symbol, p.asset_class, p.side, p.qty, p.avg_entry_price, p.unrealized_pl)

# 6a) close a single-leg position (DELETE /v2/positions/{symbol}; qty is a str) - unpriced close; market hours only
trade.close_position(long_sym, close_options=ClosePositionRequest(qty="1"))

# 6b) close the vertical atomically: one mleg order with *_to_close intents, net credit => negative limit
trade.submit_order(LimitOrderRequest(
    qty=1, order_class=OrderClass.MLEG, time_in_force=TimeInForce.DAY, limit_price=-2.50,
    legs=[
        OptionLegRequest(symbol=long_sym,  ratio_qty=1, side=OrderSide.SELL, position_intent=PositionIntent.SELL_TO_CLOSE),
        OptionLegRequest(symbol=short_sym, ratio_qty=1, side=OrderSide.BUY,  position_intent=PositionIntent.BUY_TO_CLOSE),
    ],
))
# replace/cancel PARENT only: trade.replace_order_by_id(o2.id, ReplaceOrderRequest(qty=...)); trade.cancel_order_by_id(o2.id)
```

**Raw REST payload for a vertical [V, verbatim from docs]:**

```json
{"order_class": "mleg", "qty": "1", "type": "limit", "limit_price": "1.00", "time_in_force": "day",
 "legs": [{"symbol": "AAPL250117C00190000", "ratio_qty": "1", "side": "buy",  "position_intent": "buy_to_open"},
          {"symbol": "AAPL250117C00210000", "ratio_qty": "1", "side": "sell", "position_intent": "sell_to_open"}]}
```

**Policies [I]**
- Prefer closing a spread as one multi-leg order. The support article describes it as one way to avoid insufficient-buying-power errors.
  - If legs must be closed separately, close the short leg first.
- Add a startup guard that refuses to run unless the client is on the paper base URL and `options_trading_level >= 3`.
- Wrap the broker in a `BrokerAdapter` interface.
  - Its methods are `get_chain`, `submit_single`, `submit_spread`, `positions`, `close_position`, `close_spread`, `cancel` and `reconcile`.
  - OCC symbols are the canonical contract id.
- v1 structures are defined-risk only: long calls and puts, debit and credit verticals, and iron condors.
  - There are no calendars or diagonals in v1.

---

## 5. Historical data plan ($0 start, upgrade path)

### 5.1 Tiers

Tag every backtest run with its `fidelity`.

| Tier | Source | Coverage | What it has | Compromises |
|---|---|---|---|---|
| EOD_QUOTES (primary) | `anahatsingh-ui/options-dataset-hist` (MIT preservation mirror, about 1.6 GB of plain parquet) **[V: footers read]** | SPY and IWM from 2008, QQQ from 2011. Ends 2025-12-12 for SPY and 2025-12-16 for QQQ. No 2026 data. | bid/ask with sizes, mark, last, volume, open interest, IV, greeks, OCC `contract_id`; separate `underlying_prices.parquet` | See the mirror notes below the table. |
| EOD_QUOTES (official cross-check and 2026 gap-fill) | ThetaData FREE `/v3/option/history/eod` **[V docs]** | From 2023-06-01 per the tier table; the same page also says "1 year". 20 or 30 requests per minute. | Closing NBBO bid/ask from the 17:15 ET report, OHLC, volume | No greeks, IV or open interest, so compute IV yourself. The 17:15 snapshot differs from the mirror's 16:00 one; do not mix them within a trade. It needs a free account plus the Java 21 Theta Terminal. Whether the terminal-less `thetadata` library accepts a free account is **[U]**. |
| IV surface (sanity check and IV-rank history) | DoltHub `post-no-preference/options` (CC BY-SA 4.0) **[V live]** | 2019-02-09 to yesterday. Weekly in 2019, Mon/Wed/Fri in 2024, daily in 2026. | `volatility_history` (iv_current, 52-week high and low, realised vol) for 2,330 symbols. `option_chain` has bid/ask, IV (the column named `vol`) and greeks. | See the DoltHub notes below the table. |
| INTRADAY_TRADES_ONLY | Alpaca Basic option bars and trades **[V docs]** | From Feb 2024. 200 calls per minute, 100 symbols per call. | Trade-derived OHLCV | No historical quotes, IV, greeks or open interest. No bar is produced when there is no trade. It needs a spread model calibrated from the EOD tiers. |
| Forward-recorded (start now) | Alpaca `get_option_chain` snapshots taken on a schedule, the news WebSocket with a local `received_at`, and daily forward earnings calendars | From today | Our own intraday quote, greek and news history with provable point-in-time timestamps | The feed is indicative. These cannot be reconstructed later. |
| SYNTHETIC (smoke tests only) | Black-Scholes pricing from underlying bars, VIX or VXN, and the Treasury 13-week rate | VIX from 1990 | — | VIX is not the IV of your expiry. There is no skew, no bid/ask and no early exercise. Never report P&L from this tier as evidence. |

**Mirror dataset notes**
- There is one 16:00 ET snapshot per day.
- The vendor is unnamed. **[I, strongly supported]** It was probably derived from Alpha Vantage, because the field order is identical.
  - The original repo and its host went offline in 2026.
  - Treat the mirror as private use only, and clone and checksum it now.
- `date` and `expiration` are strings.
- QQQ files have extra `id` and `created_at` columns.
- IV is capped to 0.015–9.995.
- The greeks are Black-Scholes, which is wrong for American exercise and dividends.
- Option rows carry no underlying price, so join the underlying file.

**DoltHub notes**
- It is a re-sampled surface, not a contract history.
- Strikes and the 3 to 4 expirations change daily.
- The nearest expiry is always 2 or more weeks out.
- It has no volume, open interest or underlying price.
- Use it only by fitting a smile for each day.
- Queries returning more than 1000 rows come back with status `RowLimit`, and broad scans time out. Query by exact date and symbol, or `dolt clone` the repo.

**Sources that do not fill the gap [V]**

| Source | Why not |
|---|---|
| Massive (formerly Polygon) Basic, $0 | 5 calls per minute, 2 years of trade aggregates. No quotes or greeks; no flat files on Basic. |
| Alpha Vantage `HISTORICAL_OPTIONS` | Premium only. |
| yfinance | Current chain only, personal use. |
| OptionsDX | Data to 2023 only, manual download, unclear licence. |

### 5.2 Upgrade path, cheapest first [V prices]

1. ThetaData Value, $40 per month. Historical 1-minute NBBO, OHLC and open interest from 2020. This removes the spread model and is the single biggest accuracy gain.
2. ThetaData Standard, $80 per month. Tick data plus historical IV and first-order greeks from 2016.
3. Massive Starter, $29 per month. Flat-file aggregates for 2 years, still with no quotes. Massive Advanced, $199 per month, adds quotes from 2022-03-07.
4. Databento. Spend the $125 signup credit on a targeted OPRA `cbbo-1m` slice (history from 2013-04-01) for final validation, and check the cost API first.
5. Alpaca Algo Trader Plus, $99 per month. Real OPRA data for paper and live snapshots only.
6. Cboe DataShop, only if official 15:45 ET calculations are needed.

### 5.3 Code

```bash
git clone https://github.com/anahatsingh-ui/options-dataset-hist   # plain blobs, no LFS; then sha256sum every file and store the manifest
```
```python
import pandas as pd
df = pd.read_parquet("spy/options_2024.parquet")                     # [V README]; parse df["date"], df["expiration"] with pd.to_datetime

import duckdb                                                        # corrected vs README; NOT executed
duckdb.sql("""SELECT * FROM read_parquet('spy/options_*.parquet', union_by_name=true)
              WHERE expiration = '2024-06-21'""")                    -- expiration is VARCHAR; or CAST(expiration AS DATE)
```
```
# ThetaData FREE whole-chain EOD, one request per symbol-day (verbatim pattern from docs; Theta Terminal v3 on :25503)
http://127.0.0.1:25503/v3/option/history/eod?symbol=AAPL&expiration=*&start_date=20241104&end_date=20241104
```
```python
# thetadata 1.0.10 (Python >= 3.12): real signature puts start_date/end_date FIRST, so use keywords. Free-account auth [U].
from datetime import date
from thetadata import ThetaClient
client = ThetaClient()   # THETADATA_API_KEY
eod = client.option_history_eod(symbol="SPY", expiration="*", start_date=date(2026, 1, 5), end_date=date(2026, 1, 5))
```
```python
# Alpaca option bars (names [V] v0.44.0; field is `end`, not `end_date`)
from alpaca.data.requests import OptionBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
req = OptionBarsRequest(symbol_or_symbols=["SPY260918C00650000"], timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Hour), start=start, end=end)
option_historical_data_client.get_option_bars(req).df
```
```python
# Cboe vol indices - researcher's loader, TESTED LIVE 2026-09-17. OHLC files: VIX, VIX9D, VIX3M, VIX6M, VIX1Y, VXN, VIX1D, RVX.
# Two-column files (DATE,<SYM>): VVIX, SKEW, GVZ, OVX.
import csv, io, urllib.request
from datetime import date, datetime
UA = {"User-Agent": "Mozilla/5.0"}
def cboe_index_history(sym: str) -> dict[date, float]:
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
    txt = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read().decode()
    out = {}
    for row in csv.DictReader(io.StringIO(txt)):
        d = datetime.strptime(row["DATE"], "%m/%d/%Y").date()
        out[d] = float(row.get("CLOSE") or row[sym])
    return out
def last_close_before(series: dict[date, float], decision_day: date) -> tuple[date, float]:
    k = max(x for x in series if x < decision_day)     # PIT rule: EOD value for D usable from D+1
    return k, series[k]
```
```bash
# DoltHub IV history (TESTED LIVE): IV rank = (iv_current - iv_year_low)/(iv_year_high - iv_year_low); handle nulls
curl -sL --get --data-urlencode "q=SELECT * FROM volatility_history WHERE act_symbol='SPY' ORDER BY date DESC LIMIT 1" \
  "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
```

The risk-free rate comes from the US Treasury daily bill-rates CSV, which returned HTTP 200. The URL is
`https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/2026/all?type=daily_treasury_bill_rates&field_tdr_date_value=2026&page&_format=csv` and the column is "13 WEEKS COUPON EQUIVALENT".

Put everything behind a `ChainProvider` interface:
- `get_chain(underlying, asof)` and `get_bars(occ, start, end)`;
- a `fidelity` enum;
- a local parquet cache keyed by (source, underlying, date).

Keep all vendor data outside git.

---

## 6. State bundle

### 6.1 Principles [I, reconciling the leakage and Jev-accuracy findings]

- State is a small JSON object with descriptive named sections, which the docs recommend.
  - Each value is either a scale-free quantity or a named bucket computed in code.
- State contains no tickers, company names, absolute dates or raw price levels. Raw index levels would let a model reconstruct the date.
  - State uses relative ages only, for example "6h ago" or "in 2 sessions".
  - This serves leakage control, and it also works around Jev's documented weakness on dates and numbers.
- State contains only ints and strings, with no floats, no uid and no timestamps.
  - This keeps the hash stable and the noise low.
- Each scale-free numeric is sent as `{"value": <int>, "bucket": "<label with meaning>"}`.
  - The docs endorse "either the computed number or a named bucket".
  - A/B test sending the bucket alone against sending the value plus the bucket.
- News text is the only free-text block.
  - It is entity-masked, capped in size and controlled by a config flag.
  - It is both the main leakage vector and the main injection surface, and it is also where Jev is strongest.
  - Its presence must always be ablated (section 7.2).

### 6.2 `state.v1.entry` sketch

```json
{
  "schema": "state.v1.entry",
  "context": {
    "underlying_alias": "UNDERLYING_A",
    "underlying_kind": "broad US large-cap equity index ETF",
    "decision_session": "near-close",
    "holding_window_sessions": {"value": 15, "bucket": "about three weeks"}
  },
  "market": {
    "index_trend_20d": "up: price above rising 20-day and 50-day averages",
    "implied_vol_index_level": {"value": 22, "bucket": "low: 20th-40th percentile of the past year"},
    "implied_vol_term_structure": "contango: 30-day implied vol below 3-month implied vol",
    "vol_of_vol": "normal",
    "tail_skew_index": "elevated: upper quartile of the past year",
    "equity_put_call_ratio": "normal"
  },
  "underlying": {
    "trend": {"direction": "up", "description": "price holding above aligned, rising 20/50-day averages; shallow pullbacks"},
    "momentum": {"distance_from_20d_avg_in_atr": {"value": 1, "bucket": "mildly extended above average"}, "consecutive_up_closes": {"value": 3, "bucket": "short streak"}},
    "range": {"realized_vol_20d_vs_1y": "low: daily ranges contracting vs recent weeks", "gap_today": "none"},
    "levels": {"distance_from_52w_high": "near: within 2 percent", "volume_vs_avg": "normal"},
    "intraday": "orderly: no gap, no acceleration"
  },
  "vol_surface": {
    "iv_rank_1y": {"value": 62, "bucket": "upper-middle: 60th-80th percentile of the past year"},
    "iv_vs_realized": "iv_rich: implied clearly above recent realized",
    "iv_change_1w": "rising",
    "term_structure": "contango: front expiry implied vol below next expiry",
    "skew": "normal: downside puts moderately bid",
    "expected_move_to_expiry": "2-4 percent"
  },
  "events": {
    "inside_holding_window": ["major central-bank rate decision in 2 sessions"],
    "inside_front_expiry": [],
    "earnings": {"status": "not_applicable_index_etf"}
  },
  "candidate": {
    "structure": "put_credit_spread",
    "directional_assumption": "neutral to bullish: profits if price stays above the short strike",
    "vol_exposure": "short premium",
    "short_strike_distance": "about one expected move below spot",
    "defined_risk": true,
    "tenor": "30-45 days"
  },
  "portfolio": {
    "open_positions": [{"alias": "UNDERLYING_B", "kind": "US small-cap index ETF", "directional_exposure": "bullish", "structure": "put_credit_spread"}],
    "risk_budget_used": "low: under 25 percent"
  },
  "news": [
    {"age": "6h", "source_type": "newswire", "headline": "<entity-masked headline>", "summary": "<entity-masked summary or null>"}
  ]
}
```

The `state.v1.manage` bundle drops `candidate`. It adds a `position` block with these fields:
- `entry_thesis`, a short structured string written by code at entry;
- `directional_exposure`;
- `short_leg.distance_bucket`;
- `pnl_vs_max_bucket`;
- `dte_bucket`.

It also adds `news_since_entry`.

**Provenance sidecar.** This is logged next to every Jev call and is never sent to Jev.
```json
{"decision_id": "…", "as_of_utc": "…", "real_symbol": "SPY", "state_sha256": "…", "evidence_tier": "C", "data_fidelity": "EOD_QUOTES",
 "inputs": [{"field": "vol_surface.iv_rank_1y", "source": "dolthub.volatility_history", "event_time": "…", "knowable_at_utc": "…", "fetched_at_utc": "…", "payload_sha256": "…"}]}
```

### 6.3 Point-in-time guarantees, enforced in code

1. One `DataView(as_of)` object gates all market data.
   - Any read with `knowable_at > as_of` raises.
   - StateBuilder, RiskEngine and SimBroker receive only this view.
2. `knowable_at` rules by source:

| Source | `knowable_at` rule |
|---|---|
| Alpaca news | `created_at + 60s` |
| EDGAR filings | `acceptanceDateTime + 120s`. The submissions JSON gives true UTC **[V]**. The SGML header and index page give Eastern time with no marker, so never mix the two. |
| Any EOD series (Cboe indices, put/call, mirror chains, DoltHub, daily bars) | Next session open. For DoltHub, use the commit time. |
| Intraday bars | Bar timestamp plus the interval. Alpaca's `t` is the left edge of the bar. |
| Macro values | Vintage date, available the next session |

3. **Bars.**
   - Use Alpaca historical bars with `feed=sip`, which is free if `end` is at least 15 minutes old. History starts in 2016.
   - Use `adjustment=raw` and apply splits ourselves as of T.
   - Do not use yfinance as a primary source.
4. **News [V plus correction].**
   - The Alpaca index key is `updated_at`, and the API serves the current, possibly revised, text.
   - Fetch with a padded `end` and filter locally on `created_at <= T - lag`.
   - If `updated_at > T`, keep the headline only, or drop the item.
   - Strip HTML and prefer headline plus summary.
   - Keep articles with 3 or fewer `symbols`.
   - Start the news WebSocket now and log a local `received_at`.

```python
# COMPOSED + corrected; UNTESTED (no Alpaca key). Names verified vs alpaca-py 0.44.0.
from datetime import datetime, timedelta
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
def news_known_at(client: NewsClient, symbol: str, as_of: datetime, lookback_h=72, lag_s=60, pad_days=7) -> list[dict]:
    req = NewsRequest(symbols=symbol, start=as_of - timedelta(hours=lookback_h),
                      end=as_of + timedelta(days=pad_days), sort="desc", include_content=False)   # padded end: index key is updated_at
    out = []
    for n in client.get_news(req).data["news"]:              # SDK auto-paginates, 50/page
        if n.created_at > as_of - timedelta(seconds=lag_s):   # not knowable yet
            continue
        revised_later = n.updated_at > as_of                  # API returns CURRENT text
        out.append({"id": n.id, "created_at": n.created_at, "headline": n.headline,
                    "summary": None if revised_later else n.summary, "symbols": n.symbols})
    return out
```

5. **Earnings** (only if the universe extends beyond index ETFs).
   - Ground truth is the EDGAR 8-K Item 2.02 `acceptanceDateTime`, which is free and accurate to the second in UTC.
   - Classify before-open or after-close in code.
   - Page through `filings.files`. For high-volume filers the `recent` block spans only about 1 year, and the older pages are flat columnar dicts.
   - Look-ahead trap: actual report dates were not known weeks ahead.
     - Expose `upcoming_confirmed` only within about 14 days **[I assumption]**; otherwise expose `upcoming_estimated`.
   - Never ingest post-event fields such as epsActual, surprise or Nasdaq's current market cap.
   - Archive forward calendars daily from now.
   - SEC access needs a declared User-Agent and at most 10 requests per second.
6. **Macro events** are turned into "event in N sessions" facts in code.
   - FOMC dates can be scraped from federalreserve.gov, which works with curl but lists no clock times.
   - CPI and NFP release dates:
     - FRED `release/dates` is technically the source.
     - Its terms are a problem; see section 8.
     - BLS blocks bots.
   - **[I]** Maintain a small hand-curated CSV of public release dates instead.
7. StateBuilder is a pure, deterministic function of `DataView`, so the same inputs give the same bytes and the same hash.

### 6.4 v2 wishlist

- 8-K EX-99.1 press-release and guidance excerpts. EDGAR full-text search hits are exhibit-level; join on `adsh` to get timestamps.
- Earnings-call transcripts.
- FOMC statement and minutes text.
- Form 4 insider filings with acceptance times.
- Diffs of 10-Q and 10-K risk factors.
- Full historical chains with open interest and volume, from ThetaData or Massive, for unusual-activity and per-symbol put/call features.
- A second news source.
- Our own forward-archived datasets.
- An offline loop in the style of the vendor's feature-discovery cookbook.
  - Jev's Score and Noul answers become features.
  - For Scores, use the mean plus the spread computed from `probabilities`.
  - A classical model predicts forward labels that matter for options.
  - Validation uses purged chronological splits.
- A Noul relevance pre-filter for each headline.
- The single-name equity universe.

---

## 7. Backtest methodology requirements

### 7.1 Evidence tiers (hard rule in code and in reports)

- **Tier C** is any date before Jev's release on 2026-09-15.
  - It is contaminated, because the training cutoff is undisclosed **[V]**.
  - The vendor describes Jev as post-trained from pretrained language models, so it carries world knowledge.
  - Tier C is usable only for engine, risk and execution validation and for leakage diagnostics.
  - Every Tier C report carries a "not evidence of model skill" banner.
- **Tier B** is post-release dates replayed historically. This is clean data, but there is almost none of it today.
- **Tier A** is forward paper trading with decisions logged at decision time.
- Only Tier A and Tier B numbers feed a go/no-go decision.
- The post-release window is a one-shot holdout, and nothing may be tuned on it.
- Pre-register the paper-trading evaluation plan before starting:
  - the metrics;
  - the minimum duration and trade count, from Minimum Track Record Length (Bailey and Lopez de Prado);
  - the stop criteria.
- The literature is consistent **[V abstracts]**:
  - A prompt that tells the model to "pretend it is date X" does not work.
  - Masking identifiers is only partial, because models reconstruct entities and dates from minimal context.
  - Masking also costs signal.
  - Post-cutoff recall disappears.
- State design (section 6) is therefore a mitigation, not a cure.

### 7.2 Leakage diagnostics (run on Tier C before trusting anything)

- **(a) Masked versus unmasked ablation** on the same dates. Also compute the "anonymisation gap", which is the decision-disagreement rate and needs no outcome data.
- **(b) Placebo.** Shuffle state across dates. Performance must collapse to the random baseline.
- **(c) Recall probe using Nouls.** Give Jev only the ticker and date, and ask whether it closed higher 5 sessions later. Accuracy above chance before the release date proves that recall exists.
- **(d) Counterfactual perturbation.** Flip a driver feature; the decision must move in the economically sensible direction.
- **(e) Tier C versus Tier A gap** in hit rate and Brier score. This gap is the contamination estimate.
- **(f) News on versus off.**
- **(g) Optional control arm** with the LLM adapter, using a model whose cutoff is known.

### 7.3 Fill and spread model

- Never fill at mid or last price.
- Use quotes from a pre-close snapshot:
  - Cboe uses 15:45 ET, and ORATS uses 14 minutes before the close.
  - Closing prints are avoided because quote quality degrades at the close.
- **ORATS formula [V primary source].** Buy at `Bid + (Ask − Bid)·p` and sell at `Ask − (Ask − Bid)·p`.
  - p is 0.75 for 1 leg, 0.66 for 2 legs, 0.56 for 3 legs and 0.53 for 4 or more legs.
- Always also report a worst-case band (buy at ask, sell at bid) and a best-case band (mid).
  - A strategy that is profitable only at mid is rejected.
- Do not copy optopsy's `per_leg` numbers.
  - Its model penalises extra legs, which is the opposite of ORATS.
  - Its parameter docs contradict its own code.
- Orders decided at snapshot t fill no earlier than snapshot t+1. This follows LEAN's rule of never filling on data carrying the order's timestamp.
  - **[I]** With EOD-only data the headline is "decide on day D, fill on day D+1".
  - Report a same-snapshot fill with harsher slippage as a sensitivity.
  - Keep the paper-trading cadence consistent with whichever rule is the headline.
- Reject a fill when:
  - the bid is 0, or the quote is crossed or locked;
  - the spread exceeds X% of mid or $Y;
  - the quote is older than Z;
  - the order size exceeds k% of the displayed size or the day's volume.
- Add per-contract commissions and regulatory fees.
- Round prices to valid increments.
- Mark to market from real quotes on the conservative side (longs at bid, shorts at ask), never from a constant-IV reprice.
- With trades-only intraday data, use half-spread = max(tick floor, k% of price).
  - Calibrate k by moneyness and days-to-expiry bucket from the EOD tiers.
  - Make no forward-filled fills on bars that had no trade.

### 7.4 Liquidity filters

- These live in the shared RiskEngine, so backtest and paper behave identically.
- The filters are:
  - minimum open interest;
  - minimum daily volume;
  - maximum relative spread;
  - minimum bid;
  - minimum days to expiry;
  - an underlying whitelist.
- **[I]** Use SPY, QQQ and IWM for v1.
- Log every rejection with its reason.

### 7.5 Expiry, assignment and margin (one policy for backtest and paper)

- Alpaca paper trades American-style equity and ETF options only, so the full policy applies:
  1. Mandatory close of all positions by a fixed time on the last trading day, well before 15:30 ET.
     - This removes pin risk, because holders can exercise until 5:30 pm ET.
     - It also avoids the broker's auto-liquidation.
  2. Close short in-the-money calls before the ex-dividend date when the remaining extrinsic value is less than the dividend.
  3. A LEAN-style heuristic assigns short legs that are at least 5% in the money within 4 days of expiry.
  4. Handle assignment afterwards: the resulting stock position, the margin impact, and liquidation at the next open with slippage.
- OCC auto-exercises options that are at least $0.01 in the money **[V]**.
- **Margin [V Cboe minimums].**
  - Debit structures pay the net debit.
  - Credit verticals and condors post (width − credit) × 100 × quantity.
  - No naked shorts in v1.
  - Apply a broker haircut multiplier.
  - Track buying-power utilisation, the percentage of equity at risk per trade, and the aggregate.
- Flag positions that span scheduled events and report them as a separate slice.

### 7.6 Baselines

All baselines run through the identical engine, fills, filters and risk limits:
1. Cash.
2. Buy and hold the underlying.
3. An unconditional always-enter version of the same structure on the same schedule. This isolates the structure's risk premium from Jev's timing.
4. Random-entry Monte Carlo with at least 1000 seeds, the same frequency and the same structure mix. Report Jev's percentile in that distribution.
5. A transparent rule, such as trend plus IV rank.
6. Shuffled-state Jev.
7. Jev switched off (`MockJev`).

- Headline numbers are the paired difference versus baseline 3 and the percentile versus baseline 4. Absolute P&L is never the headline.
- Add attribution of delta and vega exposure so that market beta is not mistaken for skill.
- Cross-check the always-enter baseline against optopsy (AGPL, private use only) on the same data.

### 7.7 Calibration analysis

- Every probabilistic question gets a machine-resolvable outcome with a fixed horizon, for example "the underlying closes above the short strike at expiry".
- Log `(decision_id, question_hash, p, resolve_at, outcome)`.
- Report:
  - the Brier score;
  - the Brier skill score against the base rate and against the option-implied probability (delta or N(d2)), which is the real bar to beat;
  - log-loss;
  - a reliability curve from `sklearn.calibration.calibration_curve(y_true, y_prob, n_bins=…, strategy="quantile")`;
    - Note that `brier_score_loss` names the same argument `y_proba`.
  - expected calibration error (ECE);
  - the Murphy decomposition;
  - a sharpness histogram;
  - for Choice and Score, a plot of server `confidence` against realised accuracy.
- Give all of these cluster-bootstrap confidence intervals and break them down by regime.
- If the forecasts have resolution but poor reliability, fit isotonic or Platt recalibration walk-forward on resolved outcomes only.

### 7.8 Walk-forward, statistics and metrics

- The model is frozen, so walk-forward applies only to thresholds, recalibration maps, sizing and exits.
  - Choose on window k and apply on window k+1.
  - Use a purge gap of at least the longest outcome horizon.
  - Question wording is frozen for each experiment.
- Use stationary-bootstrap confidence intervals (`arch.bootstrap.StationaryBootstrap` with `optimal_block_length`) on daily P&L and on paired differences.
  - Trade-level statistics are resampled by entry-date cluster.
- Keep an append-only trial registry of every config, threshold and prompt variant tried.
  - Report the Deflated Sharpe Ratio from it. It needs the trial count, the variance of the trial Sharpes, the sample length, skew and kurtosis.
  - Use t > 3 as a rule of thumb.
- Metrics, all reported against the baselines and with confidence intervals:
  - return and CAGR; Sharpe and Sortino; maximum drawdown and Calmar;
  - hit rate, average win and loss, profit factor, expectancy per trade;
  - CVaR 5% and the worst trade;
  - buying-power utilisation and exposure; turnover;
  - slippage paid relative to gross edge;
  - abstention and veto rates; decision flip rate across K samples;
  - Jev latency, tokens and cost;
  - slices by regime, event window and data fidelity.
- Hash-chain the decision ledger:
  - the previous hash;
  - the state hash;
  - the question hashes;
  - the cache keys;
  - the risk verdict;
  - the order ids.
- Any result can then be recomputed from the cache, the data snapshot and the git commit.

### 7.9 Engine build versus buy [V facts, I recommendation]

- Write a small purpose-built backtester driven by snapshots and events, sharing its modules with the paper bot.
- The existing engines are rejected for these reasons:

| Engine | Why rejected |
|---|---|
| backtrader | No options support; stale since 2023. |
| vectorbt | Vectorised, no options, Commons Clause licence. |
| zipline-reloaded | Equities and futures only. |
| LEAN | Best options reality models, but it is C# under Docker, and its CLI needs a paid-tier organisation. |
| nautilus_trader | Options support is in a v2 release candidate (`--pre`). Early assignment and spread margin are undocumented. Revisit only for intraday or IB. |

- Use optopsy as a cross-check only.

---

## 8. Risks and blockers

**API access (the top blocker) [V]**
- Jev is in "early access" with a waitlist.
- Keys are created at `https://console.typesafe.ai/settings/keys`.
- It is **[U]** whether a fresh login can create keys immediately and whether any free credits exist.

**What works without a TypeSafe key**
- The entire SDK offline, through `httpx2.MockTransport` and a dummy key: construction, encoding, decoding, the error paths and the retry logic.
- The docs and the public OpenAPI spec.
- All of our own code: state builder, cache, rules, risk engine, backtester and broker adapter, with `MockJev` and `ReplayJev`.
- All market data sources in sections 5 and 6, and Alpaca paper with its own free keys.
- The LLM-backed `system-one-adapter-python`, which needs an OpenAI or Anthropic key, as a stand-in for the interface and as a control arm.
- `GET /v1/models` without a key returns 403, and no inference is possible.
- The Vercel AI Gateway route (`typesafe-ai/jev`):
  - It is JavaScript AI SDK 7 only.
  - It exposes the unversioned id only, so the version cannot be pinned.
  - It renames `noul` to `boolean`.
  - Whether it bypasses the waitlist is **[U]**.
  - It is not recommended.

**Terms of service [V, MCA and Terms read]**
- No clause restricting financial or trading use was found, and no acceptable-use policy exists.
- MCA 2.3(f) forbids publishing benchmarks or performance information.
  - MCA 14.1 makes TypeSafe's Documentation, pricing information and the agreement terms confidential.
  - Breach of 2.3 is outside the liability cap and is grounds for immediate suspension.
  - Keep backtest and paper results private, or get written permission.
- MCA 2.3(b) forbids distilling or imitating outputs.
  - Using Jev probabilities as features for a model that predicts something else is a pattern the vendor endorses.
- The service is provided "AS IS", with no SLA.
  - Liability is capped at the greater of 12 months' fees or $50, and lost profits are excluded.
- Data handling:
  - A contractual no-training clause covers Customer Data.
  - "Telemetry" (logs, hashes, summary statistics and classifications) may be used without restriction.
  - The retention period is unspecified.
  - Zero data retention is enterprise-only.
  - Keep account ids, broker credentials and personal data out of state.
- The MCA is written for an "entity", so individual use needs checking.
- The site Terms forbid automated scraping of typesafe.ai. The bot must never scrape TypeSafe web properties at runtime.

**Determinism**
- There is no seed or temperature control.
- The vendor's FAQ sidesteps the question: "Determinism … is less valuable than consistency".
- Mitigations are the cache, abstention bands, K-sample agreement and the Step 0 measurement.

**Rate limits and availability [V]**
- 1,200 requests per minute and 250k tokens per second, which change "without notice".
- The vendor's status page shows 99.858% API uptime over 90 days.
  - That is 21 days with downtime, 173 minutes in total, including one 59-minute outage.
  - No incident write-ups are published.
- The service is hosted on the US West Coast.
- The backtester needs bounded concurrency, resumable cache writes and handling of 429 and 529.

**Model and SDK churn**
- Aliases move, and jev-1.12 has already been de-listed.
- MCA 2.5 permits incompatible API changes.
- The `confidence` formula changed at v1.
- The SDK is 0.x and made a breaking change after 1 day.
- Mitigations: pin the model and the SDK, log `resp.model`, persist the full probabilities, and treat any version change as a new experiment.

**Token budget**
- The docs give conflicting limits, and the over-limit error is unknown. Design to 32k tokens.

**Data**
- The provenance and legal status of the mirror dataset are uncertain, and it could vanish. Clone and checksum it now.
- Alpaca's free options feed is indicative.
- ThetaData, Massive and yfinance terms restrict use to personal use and forbid redistribution.
- DoltHub's licence is share-alike.
- The Cboe daily-stats JSON, the EDGAR full-text-search JSON and the Nasdaq calendar are undocumented endpoints. Cache their results and expect breakage.
- **FRED [V via WebFetch summary]**
  - The legal terms say the FRED API may not be used "in connection with the development or training of any … machine learning, including … large language models".
  - They also forbid "storing, caching, or archiving" FRED content.
  - Avoid the FRED API in v1 unless the user reads and accepts the terms.

**Broker**
- Paper fills are optimistic.
- The expiry-day short-leg behaviour in issue #774 is unconfirmed.
- Assignments must be polled.
- Only one option-stream connection is allowed.

**Security and hygiene**
- Never set `TYPESAFE_LOG_LEVEL=debug`.
- Never add the `pypi.typesafe.ai` index.
- Never pass `api_key=""`.
- Treat news text as hostile input.

**Agent-directed text in sources**
- No hostile prompt injection was found by any reader or verifier.
- Several docs contain text addressed to agents:
  - "Fetch the documentation index …" banners on docs.typesafe.ai, docs.alpaca.markets, docs.tradier.com and the IBKR docs.
  - Text in the TypeSafe quickstart and agent-skill pages reading "Install the TypeSafe skill … `claude plugin marketplace add typesafe-ai/skills`".
  - A DoltHub `AGENT.md`.
  - MCP marketing in the Alpha Vantage docs.
- None of it was followed, and nothing was installed except `typesafe-sdk` in the isolated probe venv. Installing the TypeSafe agent skill is the user's decision and is not needed for the build.

---

## 9. Open questions for the user (each needs a human decision)

1. **Jev access.**
   - Do you already have a TypeSafe API key, or a console account off the waitlist?
   - If not:
     - Join the waitlist now.
     - Build against the mock transport plus the cache.
     - Optionally use the LLM adapter, which needs an OpenAI or Anthropic key, as a stand-in and control arm.
2. **MCA acceptance.**
   - Are you comfortable with these terms, as an individual?
     - No publishing results attributed to Jev.
     - No distilling or surrogate models.
     - Unrestricted "Telemetry" use.
     - Prepaid credits that are non-refundable and expire after 12 months.
   - Do you want written permission from TypeSafe to publish results?
3. **Evidence standard.**
   - Do you accept that historical backtests will be labelled "not evidence of Jev skill", and that go/no-go rests on forward paper trading?
   - What minimum paper-trading duration and trade count will you pre-register?
4. **Universe and structures for v1.** The recommendation is SPY, QQQ and IWM only, defined-risk structures only, and no naked shorts.
   - Do you agree?
   - What are the maximum risk per trade and the aggregate cap?
   - What is the decision cadence: once daily near 15:45 ET, or intraday?
5. **Mirror dataset.**
   - Are you willing to use `anahatsingh-ui/options-dataset-hist` privately?
   - Its upstream rights are uncertain and it was probably derived from Alpha Vantage.
   - If not, the $0 history shrinks to ThetaData Free EOD (2023-06 onward) plus DoltHub surfaces.
6. **Budget.**
   - Do you approve ThetaData Value at $40 per month once the pipeline works? It is the biggest accuracy gain.
   - Is Alpaca Algo Trader Plus at $99 per month wanted for OPRA-quality paper quotes?
   - Do you want small Jev credit purchases only?
7. **FRED.**
   - Given its terms, should v1 skip the FRED API?
   - The alternative is to use:
     - the Treasury CSV for rates;
     - the Fed's own site for FOMC dates;
     - a hand-maintained CPI and NFP date CSV.
8. **News text in the v1 state.**
   - Should v1 include entity-masked headlines? That plays to Jev's strength but adds leakage and injection surface, and it would always be ablated.
   - Or should v1 start with buckets only?
9. **TypeSafe agent skill or plugin.**
   - Do you want it installed in your Claude Code setup?
   - It is not required, and it was deliberately not installed during research.
10. **Questions to put to TypeSafe** on Discord or by email, if you are willing:
    - What is the training cutoff of jev-1.13.0?
    - Is there any determinism control?
    - What is the deprecation policy for versioned model ids?
    - Which token limit is authoritative?

---

## Appendix: local artefacts (raw fetched sources)

- Probe venv: `/home/tyler/venvs/jev-probe`, with the SDK source at `/home/tyler/venvs/jev-probe/lib/python3.12/site-packages/typesafe_sdk/`.
- Scratch root: `C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin\fd29dd75-f797-4ad7-8155-3e4dbab012da\scratchpad`
  - The root holds `probe.py`, `probe2.py`, `example_check.py`, `sigs.py`, `docs\*.md`, `pypi_*.json` and `typesafe_sdk-0.6.0.whl`.
  - `jev\`, `jev_patterns_agent\` and `verify-jev-primitives\` hold Jev docs, the OpenAPI spec and SDK sources.
  - `ts\`, `ts\ext\` and `ts\arx\` hold TypeSafe pages, third-party pages and arXiv PDFs.
  - `v\docs`, `v\sup`, `v\tr` and `v\ib` hold broker sources.
  - `opt\` holds options-data sources.
  - `bt\` holds backtest-methodology sources.