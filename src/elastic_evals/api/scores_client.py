# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Client for Kibana score ingestion API."""

from __future__ import annotations

import httpx

from elastic_evals.api.constants import EVALS_SCORES_URL
from elastic_evals.api.errors import IngestScoresError
from elastic_evals.api.headers import build_kibana_headers
from elastic_evals.api.response import parse_error_body
from elastic_evals.api.retry import is_retryable_status_code, retry_kibana_api_call
from elastic_evals.api.scores_models import IngestScoresRequest, IngestScoresResponse
from elastic_evals.utils.logging import log

logger = log.getChild(__name__)


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

    @retry_kibana_api_call
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
        body, body_text = parse_error_body(response)

        message = f"Kibana score ingest request failed with {status_code}"
        if body_text:
            message = f"{message}: {body_text}"

        raise IngestScoresError(
            message=message,
            status_code=status_code,
            body=body,
            retryable=is_retryable_status_code(status_code),
        )
