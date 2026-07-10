# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.

"""DTOs for Kibana evals dataset sync APIs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class UpsertDatasetExamplePayload(BaseModel):
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class UpsertDatasetRequest(BaseModel):
    name: str
    description: str
    examples: list[UpsertDatasetExamplePayload]


class UpsertDatasetResponse(BaseModel):
    dataset_id: str
    added: int
    removed: int
    unchanged: int


class DatasetExample(BaseModel):
    id: str
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    created_at: str
    updated_at: str


class GetDatasetResponse(BaseModel):
    id: str
    name: str
    description: str
    examples: list[DatasetExample]
    created_at: str
    updated_at: str
