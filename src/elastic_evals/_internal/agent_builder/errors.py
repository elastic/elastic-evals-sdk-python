# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Errors for the Kibana Agent Builder client."""

from __future__ import annotations

from elastic_evals.api.errors import KibanaAPIError


class AgentBuilderError(KibanaAPIError):
    """Error raised by the Kibana Agent Builder API."""
