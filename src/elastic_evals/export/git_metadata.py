# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.

"""Git metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from subprocess import PIPE, CalledProcessError, run


@dataclass(frozen=True)
class GitMetadata:
    branch: str | None
    commit_sha: str | None


def _try_git_command(command: str) -> str | None:
    try:
        result = run(
            command, shell=True, check=True, stdout=PIPE, stderr=PIPE, text=True
        )
        return result.stdout.strip() or None
    except CalledProcessError:
        return None


def get_git_metadata() -> GitMetadata:
    return GitMetadata(
        branch=_try_git_command("git rev-parse --abbrev-ref HEAD"),
        commit_sha=_try_git_command("git rev-parse HEAD"),
    )
