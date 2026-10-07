# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Git metadata helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from subprocess import PIPE, CalledProcessError, run


@dataclass(frozen=True)
class GitMetadata:
    branch: str | None
    commit_sha: str | None


def _try_git_command(path: str | os.PathLike[str] | None, *args: str) -> str | None:
    command = ["git", *(["-C", os.fspath(path)] if path is not None else []), *args]
    try:
        result = run(command, check=True, stdout=PIPE, stderr=PIPE, text=True)
        return result.stdout.strip() or None
    except (CalledProcessError, OSError):
        return None


def get_git_metadata(path: str | os.PathLike[str] | None = None) -> GitMetadata:
    """Branch and commit of the checkout at ``path``, or of the working directory when omitted."""
    return GitMetadata(
        branch=_try_git_command(path, "rev-parse", "--abbrev-ref", "HEAD"),
        commit_sha=_try_git_command(path, "rev-parse", "HEAD"),
    )
