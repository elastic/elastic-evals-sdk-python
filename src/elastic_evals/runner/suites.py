# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Discovery helpers for evaluation suites."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import metadata
from typing import Awaitable, Callable, Iterable

SuiteRunner = Callable[[], Awaitable[None] | None]


@dataclass(frozen=True)
class EvaluationSuite:
    id: str
    description: str
    run: SuiteRunner


@dataclass(frozen=True)
class SuiteDiscoveryResult:
    suites: list[EvaluationSuite]
    errors: list[str]


def _normalize_suite(entry_name: str, value: object) -> EvaluationSuite:
    if isinstance(value, EvaluationSuite):
        suite = value
    elif callable(value):
        suite = value()
        if not isinstance(suite, EvaluationSuite):
            raise TypeError("Entry point callable must return EvaluationSuite.")
    else:
        raise TypeError("Entry point must be EvaluationSuite or callable returning it.")

    suite_id = suite.id or entry_name
    return EvaluationSuite(id=suite_id, description=suite.description, run=suite.run)


def discover_suites() -> SuiteDiscoveryResult:
    suites: list[EvaluationSuite] = []
    errors: list[str] = []
    entry_points: Iterable[metadata.EntryPoint]

    try:
        entry_points = metadata.entry_points(group="elastic_evals.suites")
    except TypeError:
        entry_points = metadata.entry_points().get("elastic_evals.suites", ())

    for entry in entry_points:
        try:
            suites.append(_normalize_suite(entry.name, entry.load()))
        except Exception as exc:
            errors.append(f"{entry.name}: {exc}")

    suites = sorted(suites, key=lambda suite: suite.id)
    return SuiteDiscoveryResult(suites=suites, errors=errors)


def get_suite(suite_id: str) -> EvaluationSuite | None:
    result = discover_suites()
    for suite in result.suites:
        if suite.id == suite_id:
            return suite
    return None


def is_async_runner(runner: SuiteRunner) -> bool:
    return inspect.iscoroutinefunction(runner)
