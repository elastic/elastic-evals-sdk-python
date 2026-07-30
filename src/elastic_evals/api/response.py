# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Shared response helpers for Kibana evals API clients."""

from __future__ import annotations

import json
import logging
from typing import Any, NoReturn

import httpx

from elastic_evals.api.errors import KibanaAPIError
from elastic_evals.api.retry import is_retryable_status_code


def parse_error_body(response: httpx.Response) -> tuple[Any, str]:
    """Parse an error body while preserving non-JSON response text."""
    try:
        body = response.json()
        return body, json.dumps(body, ensure_ascii=True)
    except (ValueError, json.JSONDecodeError):
        return response.text, response.text


def raise_kibana_error(
    response: httpx.Response,
    *,
    error_cls: type[KibanaAPIError],
    context: str,
    operation: str | None = None,
    logger: logging.Logger | None = None,
) -> NoReturn:
    """Raise a typed Kibana API error from an unsuccessful response."""
    body, body_text = parse_error_body(response)
    message = f"{context} failed"
    if operation is not None:
        message = f"{message} ({operation})"
    message = f"{message} with {response.status_code}"
    if body_text:
        message = f"{message}: {body_text}"
    if logger is not None:
        logger.error(message)
    raise error_cls(
        message=message,
        status_code=response.status_code,
        body=body,
        retryable=is_retryable_status_code(response.status_code),
    )
