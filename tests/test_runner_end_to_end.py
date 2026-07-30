# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from elastic_evals.api import compute_dataset_id
from elastic_evals.config import ElasticEvalsConfig
from elastic_evals.evaluators import (
    KibanaEvaluatorConfig,
    kibana_evaluators,
)
from elastic_evals.evaluators.base import SimpleEvaluator
from elastic_evals.executor import ElasticEvalsClient
from elastic_evals.tracing import TracingConfig
from elastic_evals.types import (
    EvaluationDataset,
    EvaluationResult,
    EvaluatorParams,
    Example,
)

_REAL_ASYNC_CLIENT = httpx.AsyncClient


class _TransportBackedAsyncClient:
    transport: httpx.AsyncBaseTransport | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if self.transport is None:
            raise RuntimeError("transport is not configured")
        self._client = _REAL_ASYNC_CLIENT(
            transport=self.transport,
            timeout=kwargs.get("timeout"),
        )

    async def __aenter__(self) -> _TransportBackedAsyncClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self._client.aclose()

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        return await self._client.post(url, json=json, headers=headers)

    async def get(self, url: str, *, headers: dict[str, str]) -> httpx.Response:
        return await self._client.get(url, headers=headers)


@pytest.fixture
def fake_kibana(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[dict[str, list[dict[str, Any]]]]:
    requests: dict[str, list[dict[str, Any]]] = {
        "upsert": [],
        "get": [],
        "scores": [],
        "evaluations": [],
    }
    dataset_id = compute_dataset_id("tiny")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        headers = dict(request.headers)

        if request.method == "POST" and url.endswith("/internal/evals/datasets/_upsert"):
            body = json.loads(request.content.decode("utf-8"))
            requests["upsert"].append({"url": url, "headers": headers, "body": body})
            return httpx.Response(
                status_code=200,
                json={
                    "dataset_id": dataset_id,
                    "added": 2,
                    "removed": 0,
                    "unchanged": 0,
                },
            )

        if request.method == "GET" and url.endswith(f"/internal/evals/datasets/{dataset_id}"):
            requests["get"].append({"url": url, "headers": headers})
            return httpx.Response(
                status_code=200,
                json={
                    "id": dataset_id,
                    "name": "tiny",
                    "description": "tiny dataset",
                    "examples": [
                        {
                            "id": "upstream-example-1",
                            "input": {"q": "ONE-UPSTREAM"},
                            "output": {"gold": "g1"},
                            "metadata": {"source": "sync"},
                            "created_at": "2026-01-01T00:00:00Z",
                            "updated_at": "2026-01-01T00:00:00Z",
                        },
                        {
                            "id": "upstream-example-2",
                            "input": {"q": "TWO-UPSTREAM"},
                            "output": {"gold": "g2"},
                            "metadata": {"source": "sync"},
                            "created_at": "2026-01-01T00:00:00Z",
                            "updated_at": "2026-01-01T00:00:00Z",
                        },
                    ],
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                },
            )

        if request.method == "POST" and url.endswith("/internal/evals/scores"):
            body = json.loads(request.content.decode("utf-8"))
            requests["scores"].append({"url": url, "headers": headers, "body": body})
            return httpx.Response(status_code=200, json={"ingested": 1, "conflicted": 0, "failed": []})

        if request.method == "POST" and url.endswith("/internal/evals/_evaluate"):
            body = json.loads(request.content.decode("utf-8"))
            requests["evaluations"].append({"url": url, "headers": headers, "body": body})
            scores = {
                "latency": 2.5,
                "input_tokens": 123,
                "output_tokens": 45,
                "tool_calls": 3,
            }
            return httpx.Response(
                status_code=200,
                json={
                    "results": [
                        {
                            "status": "ok",
                            "evaluator": {
                                "name": evaluator["name"],
                                "version": "1.0.0",
                                "kind": "code",
                            },
                            "scores": [
                                {
                                    "name": evaluator["name"],
                                    "score": scores[evaluator["name"]],
                                }
                            ],
                        }
                        for evaluator in body["evaluators"]
                    ]
                },
            )

        return httpx.Response(status_code=404, json={"message": "unexpected request"})

    _TransportBackedAsyncClient.transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "elastic_evals.api.datasets_client.httpx.AsyncClient",
        _TransportBackedAsyncClient,
    )
    monkeypatch.setattr(
        "elastic_evals.api.scores_client.httpx.AsyncClient",
        _TransportBackedAsyncClient,
    )
    monkeypatch.setattr(
        "elastic_evals.api.evaluators_client.httpx.AsyncClient",
        _TransportBackedAsyncClient,
    )

    yield requests


@pytest.mark.asyncio
async def test_runner_end_to_end(
    fake_kibana: dict[str, list[dict[str, Any]]],
) -> None:
    dataset: EvaluationDataset[Example[dict[str, str], None, None]] = EvaluationDataset(
        name="tiny",
        description="tiny dataset",
        examples=[
            Example(input={"q": "one"}),
            Example(input={"q": "two"}),
        ],
    )

    config = ElasticEvalsConfig(
        run_id="run-123",
        connector_id="test-connector",
        kibana_url="http://kibana:5601",
        kibana_api_key="secret-key",
        repetitions=1,
        concurrency=1,
        tracing=TracingConfig(enabled=False),
    )
    client = ElasticEvalsClient(config=config)
    seen_task_inputs: list[dict[str, Any]] = []
    seen_eval_inputs: list[dict[str, Any]] = []
    seen_eval_trace_ids: list[str | None] = []

    async def task(example: Example) -> dict[str, str]:
        seen_task_inputs.append(example.input)
        trace_id = "1" * 32 if example.input["q"].startswith("ONE") else "2" * 32
        return {"answer": example.input["q"], "_interaction_trace_id": trace_id}

    async def evaluate(params: EvaluatorParams) -> EvaluationResult:
        seen_eval_inputs.append(params.input)
        seen_eval_trace_ids.append(params.trace_id)
        return EvaluationResult(score=1.0)

    evaluator = SimpleEvaluator(name="echo", kind="CODE", evaluate=evaluate)
    result = await client.run_experiment(
        dataset=dataset,
        task=task,
        evaluators=[evaluator],
        experiment_name="Tiny named experiment",
    )

    assert len(result.evaluation_runs) == 2
    assert all(run.result is not None for run in result.evaluation_runs)
    assert all(run.result and run.result.score == 1.0 for run in result.evaluation_runs)
    assert all(run.experiment_run_id for run in result.evaluation_runs)
    assert result.dataset_id == compute_dataset_id("tiny")
    assert seen_task_inputs == [{"q": "ONE-UPSTREAM"}, {"q": "TWO-UPSTREAM"}]
    assert seen_eval_inputs == [{"q": "ONE-UPSTREAM"}, {"q": "TWO-UPSTREAM"}]
    assert seen_eval_trace_ids == ["1" * 32, "2" * 32]

    assert len(fake_kibana["upsert"]) == 1
    assert fake_kibana["upsert"][0]["body"] == {
        "name": "tiny",
        "description": "tiny dataset",
        "examples": [{"input": {"q": "one"}}, {"input": {"q": "two"}}],
    }
    assert fake_kibana["upsert"][0]["headers"]["kbn-xsrf"] == "true"
    assert fake_kibana["upsert"][0]["headers"]["x-elastic-internal-origin"] == "true"
    assert fake_kibana["upsert"][0]["headers"]["elastic-api-version"] == "1"
    assert fake_kibana["upsert"][0]["headers"]["authorization"] == "ApiKey secret-key"

    assert len(fake_kibana["get"]) == 1
    assert fake_kibana["get"][0]["url"].endswith(f"/internal/evals/datasets/{compute_dataset_id('tiny')}")

    assert len(fake_kibana["scores"]) == 2
    first_score = fake_kibana["scores"][0]["body"]
    second_score = fake_kibana["scores"][1]["body"]

    assert first_score["metadata"]["execution_id"] == "run-123"
    assert first_score["experiment_name"] == "Tiny named experiment"
    assert first_score["scores"][0]["example"]["dataset"]["id"] == compute_dataset_id("tiny")
    assert first_score["scores"][0]["example"]["id"] == "upstream-example-1"
    assert first_score["scores"][0]["example"]["index"] == 0
    assert first_score["scores"][0]["task"]["repetition_index"] == 0
    assert first_score["scores"][0]["task"]["output"] == {"answer": "ONE-UPSTREAM"}
    assert first_score["scores"][0]["evaluator"]["name"] == "echo"
    assert first_score["scores"][0]["evaluator"]["score"] == 1.0

    assert second_score["scores"][0]["example"]["dataset"]["id"] == compute_dataset_id("tiny")
    assert second_score["scores"][0]["example"]["id"] == "upstream-example-2"
    assert second_score["scores"][0]["example"]["index"] == 1
    assert second_score["scores"][0]["task"]["output"] == {"answer": "TWO-UPSTREAM"}


@pytest.mark.asyncio
async def test_runner_executes_registered_trace_metrics_through_kibana(
    fake_kibana: dict[str, list[dict[str, Any]]],
) -> None:
    dataset = EvaluationDataset(
        name="tiny",
        description="tiny dataset",
        examples=[Example(input={"q": "one"}), Example(input={"q": "two"})],
    )
    config = ElasticEvalsConfig(
        run_id="trace-metrics-run",
        connector_id="test-connector",
        kibana_url="http://kibana:5601",
        kibana_api_key="secret-key",
        repetitions=1,
        concurrency=1,
        tracing=TracingConfig(enabled=False),
    )
    client = ElasticEvalsClient(config=config)
    evaluators = kibana_evaluators(
        [
            KibanaEvaluatorConfig(name="latency", kind="CODE"),
            KibanaEvaluatorConfig(name="input_tokens", kind="CODE"),
            KibanaEvaluatorConfig(name="output_tokens", kind="CODE"),
            KibanaEvaluatorConfig(name="tool_calls", kind="CODE"),
        ],
        client=client.get_evaluators_client(),
    )

    async def task(example: Example) -> dict[str, str]:
        trace_id = "1" * 32 if example.input["q"].startswith("ONE") else "2" * 32
        return {"answer": example.input["q"], "_interaction_trace_id": trace_id}

    result = await client.run_experiment(dataset=dataset, task=task, evaluators=evaluators)

    assert len(result.evaluation_runs) == 8
    assert len(fake_kibana["evaluations"]) == 2
    assert len(fake_kibana["scores"]) == 8
    assert all(
        request["url"] == "http://kibana:5601/internal/evals/_evaluate" for request in fake_kibana["evaluations"]
    )
    assert all(request["headers"]["authorization"] == "ApiKey secret-key" for request in fake_kibana["evaluations"])
    assert {request["body"]["subject"]["traces"][0]["trace_id"] for request in fake_kibana["evaluations"]} == {
        "1" * 32,
        "2" * 32,
    }
    assert all(
        [evaluator["name"] for evaluator in request["body"]["evaluators"]]
        == ["latency", "input_tokens", "output_tokens", "tool_calls"]
        for request in fake_kibana["evaluations"]
    )
    assert {run.name for run in result.evaluation_runs} == {
        "latency",
        "input_tokens",
        "output_tokens",
        "tool_calls",
    }
