# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Run an evaluation script in its own process with the SDK's logging opted in."""

from __future__ import annotations

import os
import runpy
import sys

from elastic_evals.utils.logging import setup_logging


def main() -> None:
    setup_logging(os.environ.get("ELASTIC_EVALS_LOG_LEVEL", "INFO"))
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
