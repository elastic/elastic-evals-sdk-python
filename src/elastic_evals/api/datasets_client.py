# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Client for Kibana dataset sync APIs."""

from __future__ import annotations

import json
import uuid
from typing import Any, NoReturn

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

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


def _is_retryable_status_code(status_code: int) -> bool:
    return status_code in {408, 429} or status_code >= 500


def _is_retryable_error(error: BaseException) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return _is_retryable_status_code(error.response.status_code)
    if isinstance(error, httpx.HTTPError):
        return True
    if isinstance(error, DatasetSyncError):
        return error.retryable
    return False


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

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception(_is_retryable_error),
    )
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

        self._raise_dataset_sync_error(response)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception(_is_retryable_error),
    )
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

        self._raise_dataset_sync_error(response)

    def _raise_dataset_sync_error(self, response: httpx.Response) -> NoReturn:
        status_code = response.status_code
        body: Any
        try:
            body = response.json()
            body_text = json.dumps(body, ensure_ascii=True)
        except (ValueError, json.JSONDecodeError):
            body = response.text
            body_text = response.text

        message = f"Kibana dataset sync request failed with {status_code}"
        if body_text:
            message = f"{message}: {body_text}"

        raise DatasetSyncError(
            message=message,
            status_code=status_code,
            body=body,
            retryable=_is_retryable_status_code(status_code),
        )
