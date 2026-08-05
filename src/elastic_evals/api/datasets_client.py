# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Client for Kibana dataset sync APIs."""

from __future__ import annotations

import uuid

import httpx

from elastic_evals.api.constants import (
    DATASET_UUID_NAMESPACE,
    EVALS_DATASET_UPSERT_URL,
    EVALS_DATASET_URL,
)
from elastic_evals.api.datasets_models import (
    GetDatasetResponse,
    UpsertDatasetExamplePayload,
    UpsertDatasetResponse,
)
from elastic_evals.api.errors import DatasetSyncError
from elastic_evals.api.headers import build_kibana_headers
from elastic_evals.api.response import raise_kibana_error
from elastic_evals.api.retry import retry_kibana_api_call


def compute_dataset_id(name: str) -> str:
    """Compute Kibana dataset ID from dataset name."""
    return str(uuid.uuid5(DATASET_UUID_NAMESPACE, name))


class KibanaDatasetsClient:
    def __init__(
        self,
        kibana_url: str,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.kibana_url = kibana_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @retry_kibana_api_call
    async def upsert(
        self,
        name: str,
        description: str,
        examples: list[UpsertDatasetExamplePayload],
    ) -> UpsertDatasetResponse:
        """Upsert dataset examples as a full replacement on the server."""
        url = f"{self.kibana_url}{EVALS_DATASET_UPSERT_URL}"
        request_body = {
            "name": name,
            "description": description,
            "examples": [example.model_dump(exclude_none=True) for example in examples],
        }
        headers = build_kibana_headers(self.api_key)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=request_body, headers=headers)

        if response.status_code == 200:
            return UpsertDatasetResponse.model_validate(response.json())

        raise_kibana_error(
            response,
            error_cls=DatasetSyncError,
            context="Kibana dataset sync request",
        )

    @retry_kibana_api_call
    async def get(self, dataset_id: str) -> GetDatasetResponse:
        url = f"{self.kibana_url}{EVALS_DATASET_URL.format(dataset_id=dataset_id)}"
        headers = build_kibana_headers(self.api_key)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, headers=headers)

        if response.status_code == 200:
            return GetDatasetResponse.model_validate(response.json())
        if response.status_code == 404:
            raise DatasetSyncError(
                message="dataset not found",
                status_code=404,
                body=response.text,
                retryable=False,
            )

        raise_kibana_error(
            response,
            error_cls=DatasetSyncError,
            context="Kibana dataset sync request",
        )
