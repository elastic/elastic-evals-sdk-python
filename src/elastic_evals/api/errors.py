# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Typed API errors for Kibana evals clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class KibanaAPIError(RuntimeError):
    """Base error raised for unsuccessful Kibana evals API requests."""

    message: str
    status_code: int | None = None
    body: Any = None
    retryable: bool = False

    def __str__(self) -> str:
        return self.message


class IngestScoresError(KibanaAPIError):
    """Error raised by the Kibana score ingestion API."""


class DatasetSyncError(KibanaAPIError):
    """Error raised by the Kibana dataset sync APIs."""


class KibanaEvaluatorsError(KibanaAPIError):
    """Error raised by the Kibana evaluators APIs."""
