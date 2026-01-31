"""Trace-based evaluator helpers."""

from __future__ import annotations

import re

_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)


def is_valid_trace_id(trace_id: str | None) -> bool:
    if not trace_id or not isinstance(trace_id, str):
        return False
    if not _TRACE_ID_RE.fullmatch(trace_id):
        return False
    return trace_id.lower() != "0" * 32
