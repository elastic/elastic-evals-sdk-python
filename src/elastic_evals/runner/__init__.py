# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Runner package for suite discovery and CLI integration."""

from elastic_evals.runner.suites import (
    EvaluationSuite,
    SuiteDiscoveryResult,
    discover_suites,
    get_suite,
)

__all__ = ["EvaluationSuite", "SuiteDiscoveryResult", "discover_suites", "get_suite"]
