# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Dataset store backed by the Kibana evals dataset API."""

from __future__ import annotations

from typing import Any

from elastic_evals.api.datasets_client import KibanaDatasetsClient, compute_dataset_id
from elastic_evals.api.datasets_models import UpsertDatasetExamplePayload
from elastic_evals.types import EvaluationDataset, ExampleWithId


class KibanaDatasetStore:
    """Replaces the dataset on the Kibana side, then reads back the examples with Kibana's ids.

    The upsert is a full replacement, so the examples the task runs on are exactly the ones
    in `dataset`, but with server-assigned ids. Only dict-shaped `output` and `metadata`
    survive the round-trip; anything else is stored as `None`.
    """

    def __init__(self, client: KibanaDatasetsClient) -> None:
        self._client = client

    async def resolve(self, dataset: EvaluationDataset) -> list[ExampleWithId]:
        await self._client.upsert(
            dataset.name,
            dataset.description,
            [
                UpsertDatasetExamplePayload(
                    input=_dict_or_none(example.input),
                    output=_dict_or_none(example.output),
                    metadata=_dict_or_none(example.metadata),
                )
                for example in dataset.examples
            ],
        )
        upstream = await self._client.get(compute_dataset_id(dataset.name))
        return [
            ExampleWithId(
                id=example.id,
                input=example.input or {},
                output=example.output,
                metadata=example.metadata,
            )
            for example in upstream.examples
        ]


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None
