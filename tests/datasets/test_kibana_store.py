# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from elastic_evals.api import (
    DatasetExample,
    GetDatasetResponse,
    KibanaDatasetsClient,
    UpsertDatasetExamplePayload,
    UpsertDatasetResponse,
    compute_dataset_id,
)
from elastic_evals.datasets import KibanaDatasetStore
from elastic_evals.types import EvaluationDataset, Example


def _dataset() -> EvaluationDataset:
    return EvaluationDataset(
        name="greetings",
        description="two greetings",
        examples=[
            Example(input={"q": "hi"}, output={"a": "hello"}, metadata={"lang": "en"}),
            Example(input={"q": "hola"}, output="not a dict", metadata=None),
        ],
    )


def _client() -> AsyncMock:
    client = AsyncMock(spec=KibanaDatasetsClient)
    dataset_id = compute_dataset_id("greetings")
    client.upsert.return_value = UpsertDatasetResponse(dataset_id=dataset_id, added=2, removed=0, unchanged=0)
    client.get.return_value = GetDatasetResponse(
        id=dataset_id,
        name="greetings",
        description="two greetings",
        examples=[
            DatasetExample(id="kbn-1", input={"q": "hi"}, output={"a": "hello"}, created_at="t", updated_at="t"),
            DatasetExample(id="kbn-2", input=None, output=None, created_at="t", updated_at="t"),
        ],
        created_at="t",
        updated_at="t",
    )
    return client


@pytest.mark.asyncio
async def test_resolve_upserts_dataset_dropping_non_dict_fields() -> None:
    client = _client()

    await KibanaDatasetStore(client).resolve(_dataset())

    client.upsert.assert_awaited_once_with(
        "greetings",
        "two greetings",
        [
            UpsertDatasetExamplePayload(input={"q": "hi"}, output={"a": "hello"}, metadata={"lang": "en"}),
            UpsertDatasetExamplePayload(input={"q": "hola"}, output=None, metadata=None),
        ],
    )


@pytest.mark.asyncio
async def test_resolve_fetches_by_computed_dataset_id_and_returns_kibana_ids() -> None:
    client = _client()

    resolved = await KibanaDatasetStore(client).resolve(_dataset())

    client.get.assert_awaited_once_with(compute_dataset_id("greetings"))
    assert [example.id for example in resolved] == ["kbn-1", "kbn-2"]
    assert resolved[0].output == {"a": "hello"}


@pytest.mark.asyncio
async def test_resolve_normalises_missing_upstream_input_to_empty_dict() -> None:
    resolved = await KibanaDatasetStore(_client()).resolve(_dataset())

    assert resolved[1].input == {}
    assert resolved[1].output is None
