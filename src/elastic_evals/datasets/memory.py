# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Dataset store that runs on the in-memory examples without any remote call."""

from __future__ import annotations

import uuid

from elastic_evals.api.datasets_client import compute_dataset_id
from elastic_evals.types import EvaluationDataset, ExampleWithId


class InMemoryDatasetStore:
    """Hands the runner the user's own examples, creating an id for each one.

    Kibana normally creates example ids on upsert. Without Kibana, this store has to do it.
    Ids must be strings, unique within a dataset, and stable across reruns of the same
    dataset so results from two local runs can be compared example by example.
    """

    async def resolve(self, dataset: EvaluationDataset) -> list[ExampleWithId]:
        namespace = uuid.UUID(compute_dataset_id(dataset.name))
        return [
            ExampleWithId(
                id=str(uuid.uuid5(namespace, str(index))),
                input=example.input,
                output=example.output,
                metadata=example.metadata,
            )
            for index, example in enumerate(dataset.examples)
        ]
