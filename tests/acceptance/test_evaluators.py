# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import pytest

from elastic_evals.api import (
    EvaluateEvaluatorConfig,
    EvaluateRequest,
    EvaluateResponse,
    EvaluationInstrumentation,
    EvaluationSubject,
    EvaluationTrace,
    InstrumentationProfile,
    KibanaEvaluatorsClient,
    ValidateEvaluatorConfig,
    ValidateEvaluatorsRequest,
)

_API_KEY = "mock-api-key"
_TRACE_ID = "0123456789abcdef0123456789abcdef"


def _definition(name: str, kind: str) -> dict[str, Any]:
    return {
        "name": name,
        "version": "1.0.0",
        "kind": kind,
        "description": f"Mock {name} evaluator",
    }


def _profile_evidence(
    *,
    user_query: str = "found",
    agent_response: str = "found",
    tool_calls: str = "found",
) -> dict[str, Any]:
    return {
        "user_query": {"status": user_query},
        "agent_response": {"status": agent_response},
        "tool_calls": {"status": tool_calls},
    }


class _WorkflowScenario:
    _expected_routes = [
        ("GET", "/internal/evals/evaluators"),
        ("POST", "/internal/evals/traces/_resolve_instrumentation"),
        ("POST", "/internal/evals/evaluators/_validate"),
        ("POST", "/internal/evals/_evaluate"),
    ]

    def __init__(
        self,
        *,
        definitions: list[dict[str, Any]],
        instrumentation: dict[str, Any],
        validation: dict[str, Any],
        evaluation: dict[str, Any],
    ) -> None:
        self.definitions = definitions
        self.instrumentation = instrumentation
        self.validation = validation
        self.evaluation = evaluation
        self.stage = 0

    def respond(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        headers: Any,
    ) -> tuple[int, dict[str, Any]]:
        if not _has_expected_headers(headers):
            return 401, {"message": "missing required headers"}
        if self.stage >= len(self._expected_routes):
            return 409, {"message": "workflow already completed"}
        if (method, path) != self._expected_routes[self.stage]:
            return 409, {"message": "unexpected workflow operation"}
        if not self._valid_body(body):
            return 400, {"message": "unexpected request body"}

        responses = [
            {"evaluators": self.definitions},
            self.instrumentation,
            self.validation,
            self.evaluation,
        ]
        response = responses[self.stage]
        self.stage += 1
        return 200, response

    def _valid_body(self, body: dict[str, Any] | None) -> bool:
        if self.stage == 0:
            return body is None
        if self.stage == 1:
            return body == {"trace_id": _TRACE_ID}

        expected_names = [definition["name"] for definition in self.definitions]
        if not isinstance(body, dict):
            return False
        evaluator_configs = body.get("evaluators")
        if not isinstance(evaluator_configs, list):
            return False
        if [config.get("name") for config in evaluator_configs] != expected_names:
            return False

        if self.stage == 2:
            expected_configs = [
                {
                    "name": definition["name"],
                    "version": definition["version"],
                }
                for definition in self.definitions
            ]
            return evaluator_configs == expected_configs

        expected_configs = [
            {
                "name": definition["name"],
                "version": definition["version"],
                **({"connector_id": "mock-connector"} if definition["kind"] == "llm" else {}),
            }
            for definition in self.definitions
        ]
        return evaluator_configs == expected_configs


class _RetryScenario:
    def __init__(self) -> None:
        self.first_body: dict[str, Any] | None = None
        self.attempt = 0

    def respond(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        headers: Any,
    ) -> tuple[int, dict[str, Any]]:
        if not _has_expected_headers(headers):
            return 401, {"message": "missing required headers"}
        if method != "POST" or path != "/internal/evals/_evaluate":
            return 404, {"message": "not found"}

        self.attempt += 1
        if self.attempt == 1:
            self.first_body = body
            return 503, {"message": "temporarily unavailable"}
        if body != self.first_body:
            return 400, {"message": "retry payload changed"}
        return 200, {
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
        }


def _has_expected_headers(headers: Any) -> bool:
    return (
        headers.get("Authorization") == f"ApiKey {_API_KEY}"
        and headers.get("kbn-xsrf") == "true"
        and headers.get("Elastic-Api-Version") == "1"
    )


class _MockEvaluatorsAPI:
    def __init__(self, scenario: _WorkflowScenario | _RetryScenario) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                self._handle("GET")

            def do_POST(self) -> None:  # noqa: N802
                self._handle("POST")

            def _handle(self, method: str) -> None:
                content_length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(content_length)) if content_length else None
                status, response_body = scenario.respond(
                    method,
                    self.path,
                    body,
                    self.headers,
                )
                payload = json.dumps(response_body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, message_format: str, *args: Any) -> None:
                return None

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self) -> _MockEvaluatorsAPI:
        self.thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


async def _run_workflow(client: KibanaEvaluatorsClient) -> EvaluateResponse:
    definitions = (await client.list_evaluators()).evaluators
    instrumentation = await client.resolve_instrumentation(_TRACE_ID)
    recommended = instrumentation.recommended_instrumentation
    profile = cast(
        InstrumentationProfile,
        recommended.profile if recommended is not None else "elastic-inference",
    )
    subject = EvaluationSubject(
        traces=[
            EvaluationTrace(
                trace_id=_TRACE_ID,
                reference_data={"expected": "Paris"},
            )
        ],
        instrumentation=EvaluationInstrumentation(profile=profile),
    )

    await client.validate(
        ValidateEvaluatorsRequest(
            subject=subject,
            evaluators=[
                ValidateEvaluatorConfig(
                    name=definition.name,
                    version=definition.version,
                )
                for definition in definitions
            ],
        )
    )
    return await client.evaluate(
        EvaluateRequest(
            subject=subject,
            evaluators=[
                EvaluateEvaluatorConfig(
                    name=definition.name,
                    version=definition.version,
                    connector_id=("mock-connector" if definition.kind == "llm" else None),
                )
                for definition in definitions
            ],
        )
    )


@pytest.mark.asyncio
async def test_complete_evaluator_workflow_returns_all_scores() -> None:
    definitions = [
        _definition("correctness", "llm"),
        _definition("groundedness", "llm"),
        _definition("latency", "code"),
        _definition("input_tokens", "code"),
        _definition("output_tokens", "code"),
        _definition("tool_calls", "code"),
    ]
    validation = {
        "evaluators": [
            {
                "name": definition["name"],
                "version": definition["version"],
                "ready": True,
                "unmet": [],
            }
            for definition in definitions
        ]
    }
    evaluation = {
        "results": [
            {
                "status": "ok",
                "evaluator": {
                    "name": definition["name"],
                    "version": definition["version"],
                    "kind": definition["kind"],
                },
                "scores": [
                    {
                        "name": definition["name"],
                        "score": score,
                    }
                ],
            }
            for definition, score in zip(
                definitions,
                [1.0, 0.9, 1.25, 120.0, 42.0, 2.0],
                strict=True,
            )
        ]
    }
    scenario = _WorkflowScenario(
        definitions=definitions,
        instrumentation={
            "profiles": [
                {
                    "profile": "elastic-inference",
                    "evidence": _profile_evidence(),
                }
            ],
            "recommended_instrumentation": {"profile": "elastic-inference"},
        },
        validation=validation,
        evaluation=evaluation,
    )

    with _MockEvaluatorsAPI(scenario) as api:
        result = await _run_workflow(KibanaEvaluatorsClient(api.url, api_key=_API_KEY))

    assert {
        score.name: score.score for evaluation_result in result.results for score in evaluation_result.scores or []
    } == {
        "correctness": 1.0,
        "groundedness": 0.9,
        "latency": 1.25,
        "input_tokens": 120.0,
        "output_tokens": 42.0,
        "tool_calls": 2.0,
    }


@pytest.mark.asyncio
async def test_partial_evidence_workflow_preserves_mixed_results() -> None:
    definitions = [
        _definition("correctness", "llm"),
        _definition("groundedness", "llm"),
        _definition("input_tokens", "code"),
    ]
    scenario = _WorkflowScenario(
        definitions=definitions,
        instrumentation={
            "profiles": [
                {
                    "profile": "elastic-inference",
                    "evidence": _profile_evidence(tool_calls="not_found"),
                }
            ],
            "recommended_instrumentation": None,
        },
        validation={
            "evaluators": [
                {
                    "name": "correctness",
                    "version": "1.0.0",
                    "ready": True,
                    "unmet": [],
                },
                {
                    "name": "groundedness",
                    "version": "1.0.0",
                    "ready": False,
                    "unmet": ["tool_calls"],
                    "remediation": "capture tool-call details",
                },
                {
                    "name": "input_tokens",
                    "version": "1.0.0",
                    "ready": True,
                    "unmet": [],
                },
            ]
        },
        evaluation={
            "results": [
                {
                    "status": "ok",
                    "evaluator": {
                        "name": "correctness",
                        "version": "1.0.0",
                        "kind": "llm",
                    },
                    "scores": [{"name": "correctness", "score": 1.0}],
                },
                {
                    "status": "error",
                    "evaluator": {
                        "name": "groundedness",
                        "version": "1.0.0",
                        "kind": "llm",
                    },
                    "error": {
                        "code": "evidence_unmet",
                        "message": "tool calls are missing",
                    },
                },
                {
                    "status": "ok",
                    "evaluator": {
                        "name": "input_tokens",
                        "version": "1.0.0",
                        "kind": "code",
                    },
                    "scores": [
                        {
                            "name": "input_tokens",
                            "score": None,
                            "label": "unavailable",
                        }
                    ],
                },
            ]
        },
    )

    with _MockEvaluatorsAPI(scenario) as api:
        result = await _run_workflow(KibanaEvaluatorsClient(api.url, api_key=_API_KEY))

    assert (result.results[0].scores or [])[0].score == 1.0
    assert result.results[1].error is not None
    assert result.results[1].error.code == "evidence_unmet"
    assert (result.results[2].scores or [])[0].label == "unavailable"


@pytest.mark.asyncio
async def test_transient_evaluate_failure_returns_retried_result() -> None:
    scenario = _RetryScenario()
    payload = EvaluateRequest(
        subject=EvaluationSubject(
            traces=[EvaluationTrace(trace_id=_TRACE_ID)],
        ),
        evaluators=[EvaluateEvaluatorConfig(name="latency", version="1.0.0")],
    )

    with _MockEvaluatorsAPI(scenario) as api:
        result = await KibanaEvaluatorsClient(
            api.url,
            api_key=_API_KEY,
        ).evaluate(payload)

    assert (result.results[0].scores or [])[0].score == 1.25
