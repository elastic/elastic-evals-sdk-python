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
