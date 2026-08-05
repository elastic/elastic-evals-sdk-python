# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Shared retry policy for Kibana evals API clients."""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from elastic_evals.api.errors import KibanaAPIError


def is_retryable_status_code(status_code: int) -> bool:
    """Return whether an HTTP status is safe to retry under the SDK policy."""
    return status_code in {408, 429} or status_code >= 500


def is_retryable_error(error: BaseException) -> bool:
    """Return whether a transport or typed Kibana API error should be retried."""
    if isinstance(error, httpx.HTTPStatusError):
        return is_retryable_status_code(error.response.status_code)
    if isinstance(error, httpx.HTTPError):
        return True
    if isinstance(error, KibanaAPIError):
        return error.retryable
    return False


retry_kibana_api_call = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception(is_retryable_error),
)
