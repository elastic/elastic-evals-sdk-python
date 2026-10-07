# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import subprocess
from pathlib import Path

from elastic_evals.export import GitMetadata, get_git_metadata


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def test_reads_the_checkout_at_the_given_path(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q", "-b", "feature")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "--allow-empty", "-m", "one")

    assert get_git_metadata(tmp_path) == GitMetadata(branch="feature", commit_sha=_git(tmp_path, "rev-parse", "HEAD"))


def test_reports_nothing_outside_a_checkout(tmp_path: Path) -> None:
    assert get_git_metadata(tmp_path) == GitMetadata(branch=None, commit_sha=None)
