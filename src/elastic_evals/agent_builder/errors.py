"""Typed errors for the Kibana Agent Builder client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentBuilderError(RuntimeError):
    message: str
    status_code: int | None = None
    body: Any = None
    retryable: bool = False

    def __str__(self) -> str:
        return self.message
