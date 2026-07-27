# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Shared response helpers for Kibana evals API clients."""

from __future__ import annotations

import json
from typing import Any

import httpx


def parse_error_body(response: httpx.Response) -> tuple[Any, str]:
    """Parse an error body while preserving non-JSON response text."""
    try:
        body = response.json()
        return body, json.dumps(body, ensure_ascii=True)
    except (ValueError, json.JSONDecodeError):
        return response.text, response.text
