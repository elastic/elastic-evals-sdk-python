# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Score sink that keeps results in memory instead of sending them anywhere."""

from __future__ import annotations

from elastic_evals.types import ExampleResult


class InMemoryScoreSink:
    """Collects every finished example in `results`, in the order they complete.

    Each entry keeps its `example_index` and `repetition`, so results can be re-sorted
    or matched back to the dataset after a concurrent run.
    """

    def __init__(self) -> None:
        self.results: list[ExampleResult] = []

    async def write(self, result: ExampleResult) -> None:
        self.results.append(result)
