# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from tenacity import wait_none

from elastic_evals.api import (
    EvaluateEvaluatorConfig,
    EvaluateRequest,
    EvaluateResult,
    EvaluationInstrumentation,
    EvaluationSubject,
    EvaluationTrace,
    EvaluatorError,
    EvaluatorIdentity,
    EvidenceProbe,
    InstrumentationProfile,
    KibanaEvaluatorsClient,
    KibanaEvaluatorsError,
    ValidateEvaluatorConfig,
    ValidateEvaluatorsRequest,
)
from elastic_evals.api.retry import is_retryable_status_code


class _RecordingAsyncClient:
    responses: list[httpx.Response | BaseException] = []
    requests: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.timeout = kwargs.get("timeout")

    async def __aenter__(self) -> _RecordingAsyncClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    async def get(self, url: str, *, headers: dict[str, str]) -> httpx.Response:
        return self._record("GET", url, headers=headers)

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any] | None,
        headers: dict[str, str],
    ) -> httpx.Response:
        return self._record("POST", url, json=json, headers=headers)

    def _record(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        self.requests.append({"method": method, "url": url, "timeout": self.timeout, **kwargs})
        outcome = self.responses.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @classmethod
    def configure(cls, outcomes: Sequence[httpx.Response | BaseException]) -> None:
        cls.responses = list(outcomes)
        cls.requests = []


@pytest.fixture(autouse=True)
def _mock_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("elastic_evals.api.evaluators_client.httpx.AsyncClient", _RecordingAsyncClient)
    for operation in (
        KibanaEvaluatorsClient.list_evaluators,
        KibanaEvaluatorsClient.resolve_instrumentation,
        KibanaEvaluatorsClient.validate,
        KibanaEvaluatorsClient.evaluate,
    ):
        monkeypatch.setattr(operation.retry, "wait", wait_none())


def _subject(*, profile: InstrumentationProfile | None = None) -> EvaluationSubject:
    instrumentation = EvaluationInstrumentation(profile=profile) if profile else None
    return EvaluationSubject(
        traces=[
            EvaluationTrace(
                trace_id="0123456789abcdef0123456789abcdef",
                reference_data={"expected": "Paris"},
            )
        ],
        instrumentation=instrumentation,
    )


def _list_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "evaluators": [
                {
                    "name": "correctness",
                    "version": "1.0.0",
                    "kind": "llm",
                    "description": "Checks correctness",
                    "reference_data_schema": {"type": "object", "custom": [1, {"nested": True}]},
                    "evidence_schema": {"required": ["input", "response"]},
                },
                {
                    "name": "latency",
                    "version": "1.0.0",
                    "kind": "code",
                    "description": "Measures latency",
                },
            ]
        },
    )


@pytest.mark.asyncio
async def test_resolve_instrumentation_posts_trace_and_parses_all_evidence_states() -> None:
    _RecordingAsyncClient.configure(
        [
            httpx.Response(
                200,
                json={
                    "profiles": [
                        {
                            "profile": "elastic-inference",
                            "evidence": {
                                "user_query": {"status": "found", "field": "input", "sample": "hello"},
                                "agent_response": {"status": "content_redacted", "field": "response"},
                                "tool_calls": {"status": "not_found"},
                            },
                        },
                        {
                            "profile": "otel-genai-events",
                            "evidence": {
                                "user_query": {"status": "not_found"},
                                "agent_response": {"status": "not_found"},
                                "tool_calls": {"status": "not_found"},
                            },
                        },
                    ],
                    "recommended_instrumentation": None,
                },
            )
        ]
    )

    result = await KibanaEvaluatorsClient("http://kibana:5601").resolve_instrumentation("trace-1")

    assert result.profiles[0].evidence.user_query.status == "found"
    assert result.profiles[0].evidence.agent_response.status == "content_redacted"
    assert result.recommended_instrumentation is None
    request = _RecordingAsyncClient.requests[0]
    assert request["method"] == "POST"
    assert request["url"] == "http://kibana:5601/internal/evals/traces/_resolve_instrumentation"
    assert request["json"] == {"trace_id": "trace-1"}
    assert "Authorization" not in request["headers"]


@pytest.mark.asyncio
async def test_validate_posts_typed_payload_and_parses_ready_and_unready_results() -> None:
    _RecordingAsyncClient.configure(
        [
            httpx.Response(
                200,
                json={
                    "evaluators": [
                        {"name": "latency", "version": "1.0.0", "ready": True, "unmet": []},
                        {
                            "name": "groundedness",
                            "version": "1.0.0",
                            "ready": False,
                            "unmet": ["steps"],
                            "remediation": "enable includeToolDetails",
                        },
                    ]
                },
            )
        ]
    )
    payload = ValidateEvaluatorsRequest(
        subject=_subject(profile="claude-code"),
        evaluators=[
            ValidateEvaluatorConfig(name="latency"),
            ValidateEvaluatorConfig(name="groundedness", version="1.0.0"),
        ],
    )

    result = await KibanaEvaluatorsClient("http://kibana:5601").validate(payload)

    assert result.evaluators[0].ready is True
    assert result.evaluators[1].unmet == ["steps"]
    assert result.evaluators[1].remediation == "enable includeToolDetails"
    request = _RecordingAsyncClient.requests[0]
    assert request["method"] == "POST"
    assert request["url"] == "http://kibana:5601/internal/evals/evaluators/_validate"
    assert request["json"] == {
        "subject": {
            "mode": "single-turn",
            "traces": [
                {
                    "trace_id": "0123456789abcdef0123456789abcdef",
                    "reference_data": {"expected": "Paris"},
                }
            ],
            "instrumentation": {"profile": "claude-code"},
        },
        "evaluators": [
            {"name": "latency"},
            {"name": "groundedness", "version": "1.0.0"},
        ],
    }


@pytest.mark.asyncio
async def test_validate_surfaces_non_retryable_payload_error() -> None:
    _RecordingAsyncClient.configure([httpx.Response(400, json={"message": "invalid evaluator configuration"})])
    payload = ValidateEvaluatorsRequest(
        subject=_subject(),
        evaluators=[ValidateEvaluatorConfig(name="missing-evaluator")],
    )

    with pytest.raises(KibanaEvaluatorsError) as exc_info:
        await KibanaEvaluatorsClient("http://kibana:5601").validate(payload)

    assert exc_info.value.status_code == 400
    assert exc_info.value.retryable is False
    assert exc_info.value.body == {"message": "invalid evaluator configuration"}
    assert len(_RecordingAsyncClient.requests) == 1


@pytest.mark.asyncio
async def test_evaluate_preserves_scores_unavailable_values_and_mixed_errors() -> None:
    _RecordingAsyncClient.configure(
        [
            httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "status": "ok",
                            "evaluator": {"name": "correctness", "version": "1.0.0", "kind": "llm"},
                            "scores": [
                                {"name": "factuality", "score": 1.0, "label": "pass"},
                                {"name": "relevance", "score": 0.8, "explanation": "Mostly relevant"},
                                {"name": "sequence_accuracy", "score": 0.6, "metadata": {"analysis": "..."}},
                            ],
                        },
                        {
                            "status": "ok",
                            "evaluator": {"name": "input_tokens", "version": "1.0.0", "kind": "code"},
                            "scores": [{"name": "input_tokens", "score": None, "label": "unavailable"}],
                        },
                        {
                            "status": "error",
                            "evaluator": {"name": "groundedness", "version": "1.0.0", "kind": "llm"},
                            "error": {"code": "evidence_unmet", "message": "steps are missing"},
                        },
                        {
                            "status": "error",
                            "evaluator": {"name": "latency", "version": "1.0.0", "kind": "code"},
                            "error": {"message": "runtime failure"},
                        },
                    ]
                },
            )
        ]
    )
    payload = EvaluateRequest(
        subject=_subject(profile="otel-genai-attributes"),
        evaluators=[
            EvaluateEvaluatorConfig(name="correctness", connector_id="connector-1"),
            EvaluateEvaluatorConfig(name="input_tokens"),
        ],
    )

    result = await KibanaEvaluatorsClient("http://kibana:5601").evaluate(payload)

    assert [score.name for score in result.results[0].scores or []] == [
        "factuality",
        "relevance",
        "sequence_accuracy",
    ]
    assert (result.results[1].scores or [])[0].score is None
    assert result.results[2].error is not None
    assert result.results[2].error.code == "evidence_unmet"
    assert result.results[3].error is not None
    assert result.results[3].error.code is None
    request = _RecordingAsyncClient.requests[0]
    assert request["method"] == "POST"
    assert request["url"] == "http://kibana:5601/internal/evals/_evaluate"
    assert request["json"]["evaluators"] == [
        {"name": "correctness", "connector_id": "connector-1"},
        {"name": "input_tokens"},
    ]


@pytest.mark.asyncio
async def test_evaluate_retries_transient_post_with_same_payload() -> None:
    _RecordingAsyncClient.configure(
        [
            httpx.Response(503, json={"message": "temporarily unavailable"}),
            httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "status": "ok",
                            "evaluator": {
                                "name": "latency",
                                "version": "1.0.0",
                                "kind": "code",
                            },
                            "scores": [{"name": "latency", "score": 1.25}],
                        }
                    ]
                },
            ),
        ]
    )
    payload = EvaluateRequest(
        subject=_subject(),
        evaluators=[EvaluateEvaluatorConfig(name="latency")],
    )

    result = await KibanaEvaluatorsClient("http://kibana:5601").evaluate(payload)

    assert (result.results[0].scores or [])[0].score == 1.25
    assert len(_RecordingAsyncClient.requests) == 2
    assert _RecordingAsyncClient.requests[0]["json"] == _RecordingAsyncClient.requests[1]["json"]
    assert _RecordingAsyncClient.requests[0]["json"] == payload.model_dump(exclude_none=True)


@pytest.mark.asyncio
async def test_list_evaluators_rejects_malformed_json_response() -> None:
    _RecordingAsyncClient.configure([httpx.Response(200, text="{not-json")])

    with pytest.raises(json.JSONDecodeError):
        await KibanaEvaluatorsClient("http://kibana:5601").list_evaluators()


@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
@pytest.mark.asyncio
async def test_non_retryable_errors_preserve_json_or_text_body(status_code: int) -> None:
    response = (
        httpx.Response(status_code, json={"message": "denied"})
        if status_code % 2 == 0
        else httpx.Response(status_code, text="denied")
    )
    _RecordingAsyncClient.configure([response])

    with pytest.raises(KibanaEvaluatorsError) as exc_info:
        await KibanaEvaluatorsClient("http://kibana:5601").list_evaluators()

    error = exc_info.value
    assert error.status_code == status_code
    assert error.retryable is False
    assert error.body == ({"message": "denied"} if status_code % 2 == 0 else "denied")
    assert "denied" in str(error)
    assert len(_RecordingAsyncClient.requests) == 1


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(400, False), (408, True), (429, True), (499, False), (500, True), (503, True)],
)
def test_retryable_status_classification(status_code: int, expected: bool) -> None:
    assert is_retryable_status_code(status_code) is expected


@pytest.mark.asyncio
async def test_transient_status_retries_then_succeeds() -> None:
    _RecordingAsyncClient.configure([httpx.Response(503, text="temporary"), _list_response()])

    result = await KibanaEvaluatorsClient("http://kibana:5601").list_evaluators()

    assert result.evaluators[0].name == "correctness"
    assert len(_RecordingAsyncClient.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [408, 429, 500])
async def test_retryable_status_exhausts_after_three_attempts(status_code: int) -> None:
    _RecordingAsyncClient.configure([httpx.Response(status_code, text="busy") for _ in range(3)])

    with pytest.raises(KibanaEvaluatorsError) as exc_info:
        await KibanaEvaluatorsClient("http://kibana:5601").list_evaluators()

    assert exc_info.value.status_code == status_code
    assert exc_info.value.retryable is True
    assert len(_RecordingAsyncClient.requests) == 3


@pytest.mark.parametrize(
    "profile",
    ["elastic-inference", "otel-genai-events", "otel-genai-attributes", "claude-code"],
)
def test_all_instrumentation_profiles_are_accepted(profile: InstrumentationProfile) -> None:
    assert EvaluationInstrumentation(profile=profile).profile == profile


@pytest.mark.parametrize("kind", ["LLM", "custom"])
def test_evaluator_identity_rejects_unknown_kind(kind: str) -> None:
    with pytest.raises(ValidationError):
        EvaluatorIdentity(name="example", version="1.0.0", kind=kind)  # type: ignore[arg-type]


def test_response_models_reject_unknown_enums() -> None:
    with pytest.raises(ValidationError):
        EvidenceProbe(status="missing")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        EvaluatorError(code="runtime_error", message="failed")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        EvaluateResult(
            status="partial",  # type: ignore[arg-type]
            evaluator=EvaluatorIdentity(name="latency", version="1.0.0", kind="code"),
        )
