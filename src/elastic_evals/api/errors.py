# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Typed API errors for Kibana evals clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class IngestScoresError(RuntimeError):
    message: str
    status_code: int | None = None
    body: Any = None
    retryable: bool = False

    def __str__(self) -> str:
        return self.message


@dataclass
class DatasetSyncError(RuntimeError):
    message: str
    status_code: int | None = None
    body: Any = None
    retryable: bool = False

    def __str__(self) -> str:
        return self.message
