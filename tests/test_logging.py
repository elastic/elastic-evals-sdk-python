# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import logging
import subprocess
import sys

from rich.logging import RichHandler

from elastic_evals.utils.logging import setup_logging


def test_importing_the_sdk_configures_only_its_own_logger() -> None:
    """Checked in a fresh interpreter: pytest itself attaches handlers and the conftest
    fixture re-enables propagation so `caplog` can see SDK lines."""
    script = (
        "import logging\n"
        "from rich.logging import RichHandler\n"
        "from elastic_evals.config import ElasticEvalsConfig\n"
        "root, sdk = logging.getLogger(), logging.getLogger('elastic_evals')\n"
        "print(len(root.handlers), root.level == logging.WARNING, sdk.propagate,"
        " sum(isinstance(h, RichHandler) for h in sdk.handlers))\n"
    )

    output = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True).stdout

    assert output.split() == ["0", "True", "False", "1"]


def test_setup_logging_does_not_stack_handlers_when_called_again() -> None:
    logger = setup_logging()
    before = sum(isinstance(handler, RichHandler) for handler in logger.handlers)

    setup_logging("DEBUG")

    assert sum(isinstance(handler, RichHandler) for handler in logger.handlers) == before == 1
    assert logger.level == logging.DEBUG
