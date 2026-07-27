# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Tracing utilities for elastic-evals."""

from .client import ElasticsearchTraceClient, EsqlColumn, EsqlResponse
from .config import TracingConfig, init_tracing
from .spans import (
    get_current_trace_id,
    propagated_headers,
    with_evaluator_span,
    with_task_span,
)

__all__ = [
    "ElasticsearchTraceClient",
    "EsqlColumn",
    "EsqlResponse",
    "TracingConfig",
    "get_current_trace_id",
    "init_tracing",
    "propagated_headers",
    "with_evaluator_span",
    "with_task_span",
]
