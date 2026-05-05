"""Runner package for suite discovery and CLI integration."""

from elastic_evals.runner.suites import (
    EvaluationSuite,
    SuiteDiscoveryResult,
    discover_suites,
    get_suite,
)

__all__ = ["EvaluationSuite", "SuiteDiscoveryResult", "discover_suites", "get_suite"]
