# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Short id table helper."""

from __future__ import annotations

import random
from typing import Dict

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
MAX_ATTEMPTS_AT_LENGTH = 100


def _generate_short_id(size: int) -> str:
    return "".join(random.choice(ALPHABET) for _ in range(size))


class ShortIdTable:
    def __init__(self) -> None:
        self._by_short_id: Dict[str, str] = {}
        self._by_original_id: Dict[str, str] = {}

    def take(self, original_id: str) -> str:
        if original_id in self._by_original_id:
            return self._by_original_id[original_id]

        unique_id: str | None = None
        attempts_at_length = 0
        length = 4
        while unique_id is None:
            next_id = _generate_short_id(length)
            attempts_at_length += 1
            if next_id not in self._by_short_id:
                unique_id = next_id
            elif attempts_at_length >= MAX_ATTEMPTS_AT_LENGTH:
                attempts_at_length = 0
                length += 1

        self._by_short_id[unique_id] = original_id
        self._by_original_id[original_id] = unique_id
        return unique_id

    def lookup(self, short_id: str) -> str | None:
        return self._by_short_id.get(short_id)
