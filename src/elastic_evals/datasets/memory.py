# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Dataset store that runs on the in-memory examples without any remote call."""

from __future__ import annotations

import json
import uuid

from elastic_evals.api.datasets_client import compute_dataset_id
from elastic_evals.types import EvaluationDataset, Example, ExampleWithId


class InMemoryDatasetStore:
    """Hands the runner the user's own examples, with an id derived from each example's content.

    Same rule as Kibana: identical content within a dataset gives the same id, regardless of position.
    """

    async def resolve(self, dataset: EvaluationDataset) -> list[ExampleWithId]:
        namespace = uuid.UUID(compute_dataset_id(dataset.name))
        resolved: list[ExampleWithId] = []
        first_index_by_id: dict[str, int] = {}
        for index, example in enumerate(dataset.examples):
            example_id = str(uuid.uuid5(namespace, _content_key(example)))
            if example_id in first_index_by_id:
                raise ValueError(
                    f"Dataset {dataset.name!r} has a duplicate example: index {index} repeats index "
                    f"{first_index_by_id[example_id]}. Kibana rejects duplicates on upload; remove one."
                )
            first_index_by_id[example_id] = index
            resolved.append(
                ExampleWithId(id=example_id, input=example.input, output=example.output, metadata=example.metadata)
            )
        return resolved


def _content_key(example: Example) -> str:
    content = example.model_dump(mode="json", include={"input", "output", "metadata"}, exclude_none=True)
    return json.dumps(content, sort_keys=True, separators=(",", ":"))
