"""The offline Jev transport (DESIGN.md 15.3): an `httpx2.MockTransport` that serves the REAL SDK.

`LiveJev(..., api_key="dummy", transport=make_jev_transport())` is the only way a test touches `typesafe_sdk`: the request
walks the SDK's own encoder, headers, error mapping and response decoder, and only the socket is replaced. Tests never reach
the network (`tests/conftest.py` blocks it anyway).

The handler

* asserts `POST /v1/systemone` and `Authorization: Bearer <api_key>` - a request that is neither is a bug, not a 404;
* appends the decoded body **and the raw bytes** to `calls`, so key-order tests can compare the wire bytes with
  `canon.dumps_ordered` of what we hashed (5.9);
* asserts the state carries no float (INV-15 would have caught it earlier; this is the wire-level backstop);
* pops the next `Fault` if one is queued, else answers 200 with `{"model", "answers", "usage"}` and the
  `x-typesafe-request-id` header.

Score answers always carry `probabilities` AND `legend` with string keys: the vendor quickstart sample omits them and would
fail the SDK's decoding (B1.4). `answer_fn` defaults to `jev.mock.mock_wire_answers`, so the fixture and MockJev answer
identically and a recorded fixture run replays byte for byte.
"""

import copy
import json
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import httpx2

from jevbot.jev.mock import mock_wire_answers

__all__ = ["DEFAULT_MODEL", "DEFAULT_USAGE", "REQUEST_ID", "SYSTEM_ONE_PATH", "Fault", "make_jev_transport"]

SYSTEM_ONE_PATH: Final = "/v1/systemone"
DEFAULT_MODEL: Final = "jev-1.13.0"
DEFAULT_USAGE: Final[Mapping[str, int]] = {"input_tokens": 1234, "output_tokens": 20}
REQUEST_ID: Final = "req-test-0001"

AnswerFn = Callable[[Any, dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class Fault:
    """One scripted failure, popped by the next call. Build them with the class methods, never by hand."""

    kind: str
    arg: Any = None

    @classmethod
    def http_429(cls, retry_after_ms: int = 250) -> "Fault":
        """A rate limit carrying `retry-after-ms` (the SDK parses it; our loop honours it once, capped at 5 s)."""
        return cls("http_429", retry_after_ms)

    @classmethod
    def http_529(cls) -> "Fault":
        """The vendor's overloaded status; anything >= 500 decodes as `TypeSafeInternalServerError`."""
        return cls("http_529")

    @classmethod
    def http_401(cls) -> "Fault":
        return cls("http_401")

    @classmethod
    def http_403(cls) -> "Fault":
        return cls("http_403")

    @classmethod
    def http_422(cls, detail: Any = "questions.0.criteria: field required") -> "Fault":
        return cls("http_422", detail)

    @classmethod
    def timeout(cls) -> "Fault":
        return cls("timeout")

    @classmethod
    def connect_error(cls) -> "Fault":
        return cls("connect_error")

    @classmethod
    def wrong_model(cls, model: str = "jev-1.14.0") -> "Fault":
        """A 200 answered by another model id: INV-06 must fail closed and cache nothing."""
        return cls("wrong_model", model)

    @classmethod
    def missing_field(cls, path: str = "answers.x.probabilities") -> "Fault":
        """Drop one field from the 200 body. `x` (or `*`) as a segment means "the first key of that object"."""
        return cls("missing_field", path)

    @classmethod
    def unknown_answer_type(cls, type_name: str = "quantum") -> "Fault":
        """An answer type this SDK version does not model (it drops the answer; we must still fail closed)."""
        return cls("unknown_answer_type", type_name)

    @classmethod
    def unknown_label(cls, label: str = "not_an_option") -> "Fault":
        """Rename one Choice label, so the label set no longer equals the question's criteria keys."""
        return cls("unknown_label", label)

    @classmethod
    def prob_sum(cls, total: float = 0.96) -> "Fault":
        """Scale the first Choice answer's probabilities so they sum to `total` (the 7.1 `raw_sum` band)."""
        return cls("prob_sum", total)

    @classmethod
    def usage_none(cls) -> "Fault":
        """A 200 that reports no token counts: the spend guard must fall back to the estimate."""
        return cls("usage_none")


def _no_floats(node: Any, path: str = "state") -> None:
    if isinstance(node, float):
        raise AssertionError(f"the state sent to Jev contains a float at {path} (INV-15)")
    if isinstance(node, Mapping):
        for key, value in node.items():
            _no_floats(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _no_floats(value, f"{path}[{index}]")


def _first_key(node: Mapping[str, Any]) -> str:
    return next(iter(node))


def _drop(payload: MutableMapping[str, Any], path: str) -> None:
    node: Any = payload
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(node, MutableMapping):
            raise AssertionError(f"missing_field: {path} does not address an object")
        key = part if part in node else (_first_key(node) if part in {"x", "*"} and node else part)
        if key not in node:
            raise AssertionError(f"missing_field: {path} is not in the response body")
        node = node[key]
    if not isinstance(node, MutableMapping):
        raise AssertionError(f"missing_field: {path} does not address an object")
    leaf = parts[-1] if parts[-1] in node else (_first_key(node) if parts[-1] in {"x", "*"} and node else parts[-1])
    node.pop(leaf, None)


def _first_of_type(answers: MutableMapping[str, Any], wanted: str) -> MutableMapping[str, Any]:
    for answer in answers.values():
        if isinstance(answer, MutableMapping) and answer.get("type") == wanted:
            return answer
    raise AssertionError(f"the batch has no {wanted} answer to corrupt")


def _apply(fault: Fault, body: dict[str, Any], request: httpx2.Request) -> httpx2.Response | None:
    """Return the faulted response, or None when the fault only edited `body` (the caller then sends the 200)."""
    kind = fault.kind
    if kind == "http_429":
        return httpx2.Response(429, json={"error": "rate limited"}, headers={"retry-after-ms": str(int(fault.arg))}, request=request)
    if kind == "http_529":
        return httpx2.Response(529, json={"error": "overloaded"}, request=request)
    if kind == "http_401":
        return httpx2.Response(401, json={"error": "invalid api key"}, request=request)
    if kind == "http_403":
        return httpx2.Response(403, json={"error": "forbidden"}, request=request)
    if kind == "http_422":
        return httpx2.Response(422, json={"detail": fault.arg}, request=request)
    if kind == "timeout":
        raise httpx2.ReadTimeout("scripted timeout", request=request)
    if kind == "connect_error":
        raise httpx2.ConnectError("scripted connection error", request=request)
    if kind == "wrong_model":
        body["model"] = str(fault.arg)
        return None
    if kind == "missing_field":
        _drop(body, str(fault.arg))
        return None
    if kind == "unknown_answer_type":
        _first_key(body["answers"])
        body["answers"][_first_key(body["answers"])]["type"] = str(fault.arg)
        return None
    if kind == "unknown_label":
        answer = _first_of_type(body["answers"], "choice")
        probabilities = answer["probabilities"]
        answer["probabilities"] = {(str(fault.arg) if index == 0 else key): value for index, (key, value) in enumerate(probabilities.items())}
        return None
    if kind == "prob_sum":
        answer = _first_of_type(body["answers"], "choice")
        total = sum(answer["probabilities"].values()) or 1.0
        scale = float(fault.arg) / total
        answer["probabilities"] = {key: value * scale for key, value in answer["probabilities"].items()}
        return None
    if kind == "usage_none":
        body["usage"] = {}
        return None
    raise AssertionError(f"unknown fault kind {kind!r}")


def make_jev_transport(
    *,
    answer_fn: AnswerFn = mock_wire_answers,
    model: str = DEFAULT_MODEL,
    faults: Sequence[Fault] = (),
    omit_request_id: bool = False,
    usage: Mapping[str, int] | None = DEFAULT_USAGE,
    calls: list[dict[str, Any]] | None = None,
    api_key: str = "dummy",
) -> httpx2.MockTransport:
    """The offline transport of 15.3. `calls` (when given) collects one record per request, in order."""
    queue = list(faults)
    reported_usage: dict[str, Any] = {} if usage is None else dict(usage)

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.method == "POST", f"the SDK sent {request.method}, not POST"
        assert request.url.path == SYSTEM_ONE_PATH, f"the SDK called {request.url.path}, not {SYSTEM_ONE_PATH}"
        assert request.headers.get("Authorization") == f"Bearer {api_key}", "the request carries no / the wrong bearer token"
        raw = request.content
        decoded = json.loads(raw)
        state = decoded.get("state")
        questions = decoded.get("questions")
        assert isinstance(questions, dict) and questions, "the request carries no questions object"
        _no_floats(state)
        if calls is not None:
            calls.append(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "raw": bytes(raw),  # key-order tests compare these bytes with canon.dumps_ordered
                    "body": decoded,
                    "state": state,
                    "questions": questions,
                    "model": decoded.get("model"),
                    "headers": dict(request.headers),
                }
            )
        body: dict[str, Any] = {
            "model": model,
            "answers": copy.deepcopy(answer_fn(state, questions)),
            "usage": dict(reported_usage),
        }
        if queue:
            faulted = _apply(queue.pop(0), body, request)
            if faulted is not None:
                return faulted
        headers = {} if omit_request_id else {"x-typesafe-request-id": REQUEST_ID}
        return httpx2.Response(200, json=body, headers=headers, request=request)

    return httpx2.MockTransport(handler)
