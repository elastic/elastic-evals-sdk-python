# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Iterator

import pytest
from rich.logging import RichHandler

from elastic_evals.utils.logging import setup_logging


@pytest.fixture
def sdk_logger() -> Iterator[logging.Logger]:
    logger = logging.getLogger("elastic_evals")
    saved = list(logger.handlers)
    yield logger
    for handler in logger.handlers[:]:
        if handler not in saved:
            logger.removeHandler(handler)


def test_importing_the_sdk_installs_no_log_handlers() -> None:
    """Checked in a fresh interpreter, because pytest attaches handlers of its own."""
    script = (
        "import logging\n"
        "from elastic_evals.config import ElasticEvalsConfig\n"
        "from elastic_evals.executor import ElasticEvalsClient\n"
        "root, sdk = logging.getLogger(), logging.getLogger('elastic_evals')\n"
        "print(len(root.handlers), len(sdk.handlers), sdk.propagate)\n"
    )

    output = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True).stdout

    assert output.split() == ["0", "0", "True"]


def test_setup_logging_attaches_one_rich_handler_to_the_sdk_logger(sdk_logger: logging.Logger) -> None:
    setup_logging()
    setup_logging("DEBUG")

    assert sum(isinstance(handler, RichHandler) for handler in sdk_logger.handlers) == 1
    assert sdk_logger.level == logging.DEBUG


def test_setup_logging_accepts_a_lowercase_level(sdk_logger: logging.Logger) -> None:
    assert setup_logging("debug").level == logging.DEBUG
