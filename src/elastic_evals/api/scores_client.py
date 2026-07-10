# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Client for Kibana score ingestion API."""

from __future__ import annotations

import json
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from elastic_evals.api.constants import EVALS_SCORES_URL
from elastic_evals.api.errors import IngestScoresError
from elastic_evals.api.headers import build_kibana_headers
from elastic_evals.api.scores_models import IngestScoresRequest, IngestScoresResponse
from elastic_evals.utils.logging import log

logger = log.getChild(__name__)


def _is_retryable_status_code(status_code: int) -> bool:
    return status_code in {408, 429} or status_code >= 500


def _is_retryable_error(error: BaseException) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return _is_retryable_status_code(error.response.status_code)
    if isinstance(error, httpx.HTTPError):
        return True
    if isinstance(error, IngestScoresError):
        return error.retryable
    return False


class KibanaScoresClient:
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
    async def ingest_scores(self, payload: IngestScoresRequest) -> IngestScoresResponse:
        url = f"{self.kibana_url}{EVALS_SCORES_URL}"
        request_body = payload.model_dump(exclude_none=True)
        headers = build_kibana_headers(self.api_key)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=request_body, headers=headers)
            except httpx.HTTPError:
                raise

        if response.status_code in {200, 207}:
            parsed = IngestScoresResponse.model_validate(response.json())
            if response.status_code == 207:
                for failure in parsed.failed:
                    logger.warning(
                        "score ingest item %s failed: status=%s reason=%r",
                        failure.index,
                        failure.status,
                        failure.reason,
                    )
            return parsed

        status_code = response.status_code
        body: Any
        try:
            body = response.json()
            body_text = json.dumps(body, ensure_ascii=True)
        except (ValueError, json.JSONDecodeError):
            body = response.text
            body_text = response.text

        message = f"Kibana score ingest request failed with {status_code}"
        if body_text:
            message = f"{message}: {body_text}"

        raise IngestScoresError(
            message=message,
            status_code=status_code,
            body=body,
            retryable=_is_retryable_status_code(status_code),
        )
