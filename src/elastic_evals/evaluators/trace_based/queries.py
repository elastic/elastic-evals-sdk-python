"""ES|QL query builders for trace-based evaluators."""

from __future__ import annotations


def build_input_tokens_query(trace_id: str) -> str:
    return f"""
FROM traces-*
| WHERE trace.id == "{trace_id}"
| STATS input_tokens = SUM(attributes.gen_ai.usage.input_tokens)
""".strip()


def build_output_tokens_query(trace_id: str) -> str:
    return f"""
FROM traces-*
| WHERE trace.id == "{trace_id}"
| STATS output_tokens = SUM(attributes.gen_ai.usage.output_tokens)
""".strip()


def build_cached_tokens_query(trace_id: str) -> str:
    return f"""
FROM traces-*
| WHERE trace.id == "{trace_id}"
| STATS cached_tokens = SUM(attributes.gen_ai.usage.cached_input_tokens)
""".strip()


def build_latency_query(trace_id: str) -> str:
    return f"""
FROM traces-*
| WHERE trace.id == "{trace_id}"
| STATS total_duration_ns = MAX(duration)
| EVAL latency_seconds = TO_DOUBLE(total_duration_ns) / 1000000000
| KEEP latency_seconds
""".strip()


def build_tool_calls_query(trace_id: str) -> str:
    return f"""
FROM traces-*
| WHERE trace.id == "{trace_id}" AND attributes.elastic.inference.span.kind == "TOOL"
| STATS tool_calls = COUNT(*)
""".strip()


def build_span_latency_query(trace_id: str, span_name: str) -> str:
    return f"""
FROM traces-*
| WHERE trace.id == "{trace_id}" AND name == "{span_name}"
| EVAL latency_seconds = TO_DOUBLE(duration) / 1000000000
| KEEP latency_seconds
""".strip()
