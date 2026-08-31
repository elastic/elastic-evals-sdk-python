# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import json as _json
import uuid
from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from elastic_evals.api import (
    DATASET_UUID_NAMESPACE,
    DatasetSyncError,
    GetDatasetResponse,
    UpsertDatasetExamplePayload,
    UpsertDatasetResponse,
)
from elastic_evals.api.datasets_client import KibanaDatasetsClient, compute_dataset_id


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
        self.requests.append({"method": "POST", "url": url, "json": json, "headers": headers})
        outcome = self.responses.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def get(self, url: str, *, headers: dict[str, str]) -> httpx.Response:
        self.requests.append({"method": "GET", "url": url, "headers": headers})
        outcome = self.responses.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @classmethod
    def configure(cls, outcomes: Sequence[httpx.Response | BaseException]) -> None:
        cls.responses = list(outcomes)
        cls.requests = []


@pytest.mark.asyncio
async def test_upsert_posts_expected_body_and_returns_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = [
        UpsertDatasetExamplePayload(input={"question": "hi"}, metadata=None),
        UpsertDatasetExamplePayload(output={"answer": "there"}, metadata={"k": "v"}),
    ]
    _RecordingAsyncClient.configure(
        [
            httpx.Response(
                status_code=200,
                json={
                    "dataset_id": "dataset-1",
                    "added": 2,
                    "removed": 1,
                    "unchanged": 0,
                },
            )
        ]
    )
    monkeypatch.setattr("elastic_evals.api.datasets_client.httpx.AsyncClient", _RecordingAsyncClient)

    client = KibanaDatasetsClient("http://kibana:5601", api_key="key-123")
    response = await client.upsert("demo", "desc", examples)

    assert response == UpsertDatasetResponse(dataset_id="dataset-1", added=2, removed=1, unchanged=0)
    assert _RecordingAsyncClient.requests[0]["method"] == "POST"
    assert _RecordingAsyncClient.requests[0]["url"] == "http://kibana:5601/internal/evals/datasets/_upsert"
    assert _RecordingAsyncClient.requests[0]["json"] == {
        "name": "demo",
        "description": "desc",
        "examples": [
            {"input": {"question": "hi"}},
            {"output": {"answer": "there"}, "metadata": {"k": "v"}},
        ],
    }
    assert _RecordingAsyncClient.requests[0]["headers"]["kbn-xsrf"] == "true"
    assert _RecordingAsyncClient.requests[0]["headers"]["x-elastic-internal-origin"] == "true"
    assert _RecordingAsyncClient.requests[0]["headers"]["Elastic-Api-Version"] == "1"
    assert _RecordingAsyncClient.requests[0]["headers"]["Authorization"] == "ApiKey key-123"


@pytest.mark.asyncio
async def test_get_parses_dataset_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingAsyncClient.configure(
        [
            httpx.Response(
                status_code=200,
                json={
                    "id": "dataset-1",
                    "name": "demo",
                    "description": "desc",
                    "examples": [
                        {
                            "id": "example-1",
                            "input": {"a": 1},
                            "output": {"b": 2},
                            "metadata": {"k": "v"},
                            "created_at": "2026-01-01T00:00:00Z",
                            "updated_at": "2026-01-01T00:00:00Z",
                        }
                    ],
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                },
            )
        ]
    )
    monkeypatch.setattr("elastic_evals.api.datasets_client.httpx.AsyncClient", _RecordingAsyncClient)

    client = KibanaDatasetsClient("http://kibana:5601")
    response = await client.get("dataset-1")

    assert response == GetDatasetResponse(
        id="dataset-1",
        name="demo",
        description="desc",
        examples=[
            {
                "id": "example-1",
                "input": {"a": 1},
                "output": {"b": 2},
                "metadata": {"k": "v"},
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        ],
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )
    assert _RecordingAsyncClient.requests[0]["method"] == "GET"
    assert _RecordingAsyncClient.requests[0]["url"] == "http://kibana:5601/internal/evals/datasets/dataset-1"
    assert "Authorization" not in _RecordingAsyncClient.requests[0]["headers"]


@pytest.mark.asyncio
async def test_get_404_raises_non_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingAsyncClient.configure([httpx.Response(status_code=404, json={"message": "not found"})])
    monkeypatch.setattr("elastic_evals.api.datasets_client.httpx.AsyncClient", _RecordingAsyncClient)

    client = KibanaDatasetsClient("http://kibana:5601")
    with pytest.raises(DatasetSyncError) as exc_info:
        await client.get("missing-dataset")

    assert exc_info.value.status_code == 404
    assert exc_info.value.retryable is False
    assert len(_RecordingAsyncClient.requests) == 1


@pytest.mark.asyncio
async def test_get_503_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingAsyncClient.configure(
        [
            httpx.Response(status_code=503, json={"message": "temporary"}),
            httpx.Response(
                status_code=200,
                json={
                    "id": "dataset-1",
                    "name": "demo",
                    "description": "desc",
                    "examples": [],
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                },
            ),
        ]
    )
    monkeypatch.setattr("elastic_evals.api.datasets_client.httpx.AsyncClient", _RecordingAsyncClient)

    client = KibanaDatasetsClient("http://kibana:5601")
    response = await client.get("dataset-1")

    assert response.id == "dataset-1"
    assert len(_RecordingAsyncClient.requests) == 2


def test_dataset_id_matches_uuid5_namespace() -> None:
    expected = str(uuid.uuid5(DATASET_UUID_NAMESPACE, "foo"))
    assert compute_dataset_id("foo") == expected


def test_dataset_id_default_space_explicit_matches_implicit() -> None:
    assert compute_dataset_id("foo", "default") == compute_dataset_id("foo")


def test_dataset_id_non_default_space_differs_from_default() -> None:
    default_id = compute_dataset_id("foo")
    space_id = compute_dataset_id("foo", "my-space")
    assert default_id != space_id


def test_dataset_id_non_default_space_matches_kibana_formula() -> None:

    # Mirrors kbn-evals-common: uuidv5(JSON.stringify([spaceId, name]), NAMESPACE)
    expected = str(uuid.uuid5(DATASET_UUID_NAMESPACE, _json.dumps(["my-space", "foo"], separators=(",", ":"))))
    assert compute_dataset_id("foo", "my-space") == expected


@pytest.mark.parametrize(
    "url, expected_space",
    [
        ("http://localhost:5601", "default"),
        ("http://localhost:5601/", "default"),
        ("https://host/s/my-space", "my-space"),
        ("https://host/s/my-space/", "my-space"),
        ("http://kibana:5601/s/staging", "staging"),
    ],
)
def test_kibana_datasets_client_extracts_space_from_url(url: str, expected_space: str) -> None:
    client = KibanaDatasetsClient(url)
    assert client.space_id == expected_space
