# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Claude Code CLI task implementation."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import httpx

from elastic_evals.config import ElasticEvalsConfig
from elastic_evals.tracing import propagated_headers
from elastic_evals.types import Example

EDOT_ENDPOINT = os.environ.get("EDOT_ENDPOINT", "http://localhost:4318")


async def _fetch_interaction_trace_id(
    es_url: str,
    run_id: str,
    example_id: str,
    *,
    api_key: str | None = None,
    max_attempts: int = 15,
    poll_interval_s: float = 2.0,
) -> str | None:
    """Poll Elasticsearch for the claude_code.interaction span for this example.

    Traces are batched (OTEL_TRACES_EXPORT_INTERVAL) so they arrive shortly after
    the subprocess exits. We retry until the span lands or we give up.
    """
    query = {
        "size": 1,
        "query": {
            "bool": {
                "must": [
                    {"term": {"name": "claude_code.interaction"}},
                    {"term": {"resource.attributes.elastic.evals.run_id": run_id}},
                    {"term": {"resource.attributes.elastic.evals.example_id": example_id}},
                ]
            }
        },
        "_source": ["trace_id"],
    }
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"
    async with httpx.AsyncClient() as client:
        for _ in range(max_attempts):
            await asyncio.sleep(poll_interval_s)
            try:
                resp = await client.post(
                    f"{es_url}/.ds-traces-generic.otel-default-*/_search",
                    json=query,
                    headers=headers,
                    timeout=10.0,
                )
                hits = resp.json().get("hits", {}).get("hits", [])
                if hits:
                    return hits[0]["_source"].get("trace_id")
            except Exception:
                pass
    return None


# Safety ceiling: most tasks complete in under 2 minutes.
_SUBPROCESS_TIMEOUT_S = 120


async def claude_code_task(example: Example, config: ElasticEvalsConfig) -> dict[str, Any]:
    """Run a prompt through the Claude Code CLI and return output with telemetry metadata.

    Environment variables injected into the subprocess:
    - CLAUDE_CODE_ENABLE_TELEMETRY / OTEL_* — forward Claude Code traces, metrics + logs to EDOT
    - OTEL_RESOURCE_ATTRIBUTES — tags every span with run_id + example_id for correlation
    - TRACEPARENT — W3C trace context from the current eval span; Claude Code spans appear as
      children in APM if its Node.js OTel SDK reads this env var (requires the env propagator
      to be wired up in Claude Code — see README for details)

    After the subprocess exits, if ELASTICSEARCH_URL is configured this function polls Elasticsearch
    for the claude_code.interaction span and returns its trace_id as "_interaction_trace_id" so
    the executor can substitute it for the eval harness trace_id, linking the UI to the actual
    LLM interaction trace.
    """
    prompt = example.input.get("prompt", "")
    example_id: str | None = getattr(example, "id", None)

    env = {**os.environ}
    env.update(
        {
            "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
            "OTEL_METRICS_EXPORTER": "otlp",
            "OTEL_LOGS_EXPORTER": "otlp",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
            "OTEL_EXPORTER_OTLP_ENDPOINT": EDOT_ENDPOINT,
            # Trace export uses a separate non-standard env var; OTEL_TRACES_EXPORTER has no effect.
            "BETA_TRACING_ENDPOINT": EDOT_ENDPOINT,
            "ENABLE_BETA_TRACING_DETAILED": "1",
            "OTEL_METRIC_EXPORT_INTERVAL": "10000",
            "OTEL_LOGS_EXPORT_INTERVAL": "5000",
            "OTEL_TRACES_EXPORT_INTERVAL": "2000",
            "OTEL_SERVICE_NAME": "claude-code",
            # Include message/response content in log records (correlated to traces via trace ID).
            # Without these, API bodies and prompts are redacted or omitted from telemetry.
            "OTEL_LOG_RAW_API_BODIES": "1",
            "OTEL_LOG_TOOL_CONTENT": "1",
            "OTEL_LOG_TOOL_DETAILS": "1",
            "OTEL_LOG_USER_PROMPTS": "1",
            # Correlate Claude Code spans with this eval run + example via resource attributes.
            "OTEL_RESOURCE_ATTRIBUTES": (
                f"elastic.evals.run_id={config.run_id}"
                + (f",elastic.evals.example_id={example_id}" if example_id else "")
            ),
        }
    )

    # Inject W3C trace context so Claude Code subprocess spans link back to the eval span.
    headers = propagated_headers()
    traceparent = headers.get("traceparent")
    if traceparent:
        env["TRACEPARENT"] = traceparent
        tracestate = headers.get("tracestate", "")
        if tracestate:
            env["TRACESTATE"] = tracestate

    start = time.monotonic()

    proc = await asyncio.create_subprocess_exec(
        "claude",
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",  # required by Claude Code when output-format=stream-json
        "--dangerously-skip-permissions",
        prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=_SUBPROCESS_TIMEOUT_S)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return {
            "response": "",
            "tool_calls": [],
            "tool_call_count": 0,
            "latency_ms": round((time.monotonic() - start) * 1000),
            "exit_code": -1,
            "error": "timeout",
        }

    latency_ms = round((time.monotonic() - start) * 1000)

    tool_calls: list[str] = []
    response_text = ""
    num_turns: int = 0
    claude_duration_ms: int | None = None
    total_cost_usd: float | None = None
    is_error: bool = False

    for raw_line in stdout_bytes.decode(errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type")

        if event_type == "assistant":
            message = event.get("message", {})
            for block in message.get("content", []):
                if block.get("type") == "tool_use":
                    tool_calls.append(block.get("name", "unknown"))

        elif event_type == "result":
            response_text = event.get("result", "")
            num_turns = event.get("num_turns", 0)
            claude_duration_ms = event.get("duration_ms")
            total_cost_usd = event.get("total_cost_usd")
            is_error = bool(event.get("is_error", False))

    stderr_text = stderr_bytes.decode(errors="replace").strip()

    result: dict[str, Any] = {
        "response": response_text,
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "latency_ms": latency_ms,
        "claude_duration_ms": claude_duration_ms,
        "num_turns": num_turns,
        "total_cost_usd": total_cost_usd,
        "exit_code": proc.returncode,
    }
    if is_error or proc.returncode != 0:
        result["error"] = stderr_text or response_text

    if config.elasticsearch_url and example_id:
        interaction_trace_id = await _fetch_interaction_trace_id(
            config.elasticsearch_url,
            config.run_id,
            example_id,
            api_key=config.elasticsearch_api_key,
        )
        if interaction_trace_id:
            result["_interaction_trace_id"] = interaction_trace_id

    return result
