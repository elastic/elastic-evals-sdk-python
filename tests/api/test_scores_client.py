# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from elastic_evals.api import (
    Dataset,
    IngestEvaluator,
    IngestExample,
    IngestMetadata,
    IngestScoreItem,
    IngestScoresError,
    IngestScoresRequest,
    IngestScoresResponse,
    IngestTask,
    Model,
)
from elastic_evals.api.scores_client import KibanaScoresClient


def _build_payload() -> IngestScoresRequest:
    return IngestScoresRequest(
        experiment_id="exp-1",
        task_model=Model(id="gpt-4o"),
        evaluator_model=Model(id="judge-1"),
        metadata=IngestMetadata(
            execution_id="run-1",
            suite_id="suite-1",
            total_repetitions=1,
            hostname="localhost",
        ),
        scores=[
            IngestScoreItem(
                example=IngestExample(
                    id="example-1",
                    index=0,
                    dataset=Dataset(id="dataset-1", name="dataset"),
                    input={"question": "hello"},
                ),
                task=IngestTask(repetition_index=0, output={"answer": "world"}),
                evaluator=IngestEvaluator(name="exact_match", score=1.0),
            )
        ],
    )


class _RecordingAsyncClient:
    responses: list[httpx.Response | BaseException] = []
    requests: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.timeout = kwargs.get("timeout")

    async def __aenter__(self) -> _RecordingAsyncClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        self.requests.append({"url": url, "json": json, "headers": headers})
        outcome = self.responses.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @classmethod
    def configure(cls, outcomes: Sequence[httpx.Response | BaseException]) -> None:
        cls.responses = list(outcomes)
        cls.requests = []


@pytest.mark.asyncio
async def test_ingest_scores_200_returns_parsed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _build_payload()
    _RecordingAsyncClient.configure(
        [
            httpx.Response(
                status_code=200, json={"ingested": 1, "conflicted": 0, "failed": []}
            )
        ]
    )
    monkeypatch.setattr(
        "elastic_evals.api.scores_client.httpx.AsyncClient", _RecordingAsyncClient
    )

    client = KibanaScoresClient("http://kibana:5601", api_key="key-123")
    result = await client.ingest_scores(payload)

    assert result == IngestScoresResponse(ingested=1, conflicted=0, failed=[])
    assert (
        _RecordingAsyncClient.requests[0]["url"]
        == "http://kibana:5601/internal/evals/scores"
    )
    assert _RecordingAsyncClient.requests[0]["json"] == payload.model_dump(
        exclude_none=True
    )
    assert _RecordingAsyncClient.requests[0]["headers"]["kbn-xsrf"] == "true"
    assert (
        _RecordingAsyncClient.requests[0]["headers"]["x-elastic-internal-origin"]
        == "true"
    )
    assert _RecordingAsyncClient.requests[0]["headers"]["Elastic-Api-Version"] == "1"
    assert (
        _RecordingAsyncClient.requests[0]["headers"]["Authorization"]
        == "ApiKey key-123"
    )


@pytest.mark.asyncio
async def test_ingest_scores_207_logs_failures_and_returns_response(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    payload = _build_payload()
    _RecordingAsyncClient.configure(
        [
            httpx.Response(
                status_code=207,
                json={
                    "ingested": 1,
                    "conflicted": 0,
                    "failed": [
                        {"index": 3, "status": 500, "reason": "oops"},
                        {"index": 4, "status": 400, "reason": "bad payload"},
                    ],
                },
            )
        ]
    )
    monkeypatch.setattr(
        "elastic_evals.api.scores_client.httpx.AsyncClient", _RecordingAsyncClient
    )

    client = KibanaScoresClient("http://kibana:5601")
    result = await client.ingest_scores(payload)

    assert result.ingested == 1
    assert "score ingest item 3 failed: status=500 reason='oops'" in caplog.text
    assert "score ingest item 4 failed: status=400 reason='bad payload'" in caplog.text


@pytest.mark.asyncio
async def test_ingest_scores_400_raises_non_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _build_payload()
    _RecordingAsyncClient.configure(
        [httpx.Response(status_code=400, json={"message": "invalid payload"})]
    )
    monkeypatch.setattr(
        "elastic_evals.api.scores_client.httpx.AsyncClient", _RecordingAsyncClient
    )

    client = KibanaScoresClient("http://kibana:5601")

    with pytest.raises(IngestScoresError) as exc_info:
        await client.ingest_scores(payload)

    assert exc_info.value.status_code == 400
    assert exc_info.value.retryable is False
    assert len(_RecordingAsyncClient.requests) == 1


@pytest.mark.asyncio
async def test_ingest_scores_429_retries_three_times_then_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _build_payload()
    _RecordingAsyncClient.configure(
        [
            httpx.Response(status_code=429, json={"message": "rate limited"}),
            httpx.Response(status_code=429, json={"message": "rate limited"}),
            httpx.Response(status_code=429, json={"message": "rate limited"}),
        ]
    )
    monkeypatch.setattr(
        "elastic_evals.api.scores_client.httpx.AsyncClient", _RecordingAsyncClient
    )

    client = KibanaScoresClient("http://kibana:5601")

    with pytest.raises(IngestScoresError) as exc_info:
        await client.ingest_scores(payload)

    assert exc_info.value.status_code == 429
    assert exc_info.value.retryable is True
    assert len(_RecordingAsyncClient.requests) == 3


@pytest.mark.asyncio
async def test_ingest_scores_503_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _build_payload()
    _RecordingAsyncClient.configure(
        [
            httpx.Response(
                status_code=503, json={"message": "temporarily unavailable"}
            ),
            httpx.Response(
                status_code=200, json={"ingested": 1, "conflicted": 0, "failed": []}
            ),
        ]
    )
    monkeypatch.setattr(
        "elastic_evals.api.scores_client.httpx.AsyncClient", _RecordingAsyncClient
    )

    client = KibanaScoresClient("http://kibana:5601")
    result = await client.ingest_scores(payload)

    assert result.ingested == 1
    assert len(_RecordingAsyncClient.requests) == 2


@pytest.mark.asyncio
async def test_ingest_scores_without_api_key_omits_authorization_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _build_payload()
    _RecordingAsyncClient.configure(
        [
            httpx.Response(
                status_code=200, json={"ingested": 1, "conflicted": 0, "failed": []}
            )
        ]
    )
    monkeypatch.setattr(
        "elastic_evals.api.scores_client.httpx.AsyncClient", _RecordingAsyncClient
    )

    client = KibanaScoresClient("http://kibana:5601", api_key=None)
    await client.ingest_scores(payload)

    assert "Authorization" not in _RecordingAsyncClient.requests[0]["headers"]
