# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""CI metadata helpers."""

from __future__ import annotations

import os

from elastic_evals.types import BuildkiteMetadata

# `BuildkiteMetadata` field -> the Buildkite job variable that fills it.
_BUILDKITE_VARIABLES = {
    "build_id": "BUILDKITE_BUILD_ID",
    "job_id": "BUILDKITE_JOB_ID",
    "build_url": "BUILDKITE_BUILD_URL",
    "pipeline_slug": "BUILDKITE_PIPELINE_SLUG",
    "pull_request": "BUILDKITE_PULL_REQUEST",
    "branch": "BUILDKITE_BRANCH",
    "commit": "BUILDKITE_COMMIT",
}


def get_buildkite_metadata() -> BuildkiteMetadata | None:
    """The Buildkite build this process runs in, or ``None`` outside Buildkite.

    Same rules as kbn-evals: Buildkite sets ``BUILDKITE_PULL_REQUEST`` to ``false`` on a
    build that is not for a pull request, which counts as unset.
    """
    values = {field: value for field, name in _BUILDKITE_VARIABLES.items() if (value := os.environ.get(name))}
    if values.get("pull_request") == "false":
        del values["pull_request"]
    return BuildkiteMetadata(**values) if values else None
