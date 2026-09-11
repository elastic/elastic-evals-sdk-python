# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import pytest

from elastic_evals.datasets import InMemoryDatasetStore
from elastic_evals.types import EvaluationDataset, Example


def _dataset(name: str = "greetings") -> EvaluationDataset:
    return EvaluationDataset(
        name=name,
        description="two greetings",
        examples=[
            Example(input={"q": "hi"}, output="hello", metadata={"lang": "en"}),
            Example(input={"q": "hola"}, output=["hello", "hi"], metadata=None),
        ],
    )


@pytest.mark.asyncio
async def test_resolve_returns_one_example_with_id_per_input_example() -> None:
    resolved = await InMemoryDatasetStore().resolve(_dataset())

    assert len(resolved) == 2
    assert all(isinstance(example.id, str) and example.id for example in resolved)
    assert len({example.id for example in resolved}) == 2


@pytest.mark.asyncio
async def test_resolve_passes_input_output_and_metadata_through_unchanged() -> None:
    resolved = await InMemoryDatasetStore().resolve(_dataset())

    assert resolved[0].input == {"q": "hi"}
    assert resolved[0].output == "hello"
    assert resolved[0].metadata == {"lang": "en"}
    assert resolved[1].output == ["hello", "hi"]
    assert resolved[1].metadata is None


@pytest.mark.asyncio
async def test_ids_are_stable_across_reruns_of_the_same_dataset() -> None:
    first = await InMemoryDatasetStore().resolve(_dataset())
    second = await InMemoryDatasetStore().resolve(_dataset())

    assert [example.id for example in first] == [example.id for example in second]


@pytest.mark.asyncio
async def test_ids_differ_between_datasets_with_different_names() -> None:
    greetings = await InMemoryDatasetStore().resolve(_dataset("greetings"))
    farewells = await InMemoryDatasetStore().resolve(_dataset("farewells"))

    assert {example.id for example in greetings}.isdisjoint({example.id for example in farewells})
