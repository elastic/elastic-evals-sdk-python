# Claude Code Eval

Evaluates the Claude Code CLI as a task target. Each dataset example is a prompt sent
to `claude --print --output-format stream-json`. The eval harness captures:

- **Latency** — wall-clock time for the subprocess to complete
- **Tool use** — which tools Claude Code actually called vs. expected
- **Response quality** — optional LLM criteria scoring (requires a Kibana connector)

Claude Code's own telemetry (metrics + logs) is forwarded to a local EDOT collector
so you can observe both sides — the eval harness and Claude Code itself — in Kibana APM.

## Prerequisites

1. Running Elasticsearch and Kibana (local stack is fine).
2. EDOT collector listening at `http://localhost:4318` (HTTP/protobuf).
3. `claude` CLI installed and authenticated (`claude --version`).
4. A Kibana LLM connector for dataset storage and optional criteria scoring.

## Run the eval

```bash
CONNECTOR_ID="your-connector-id" \
KIBANA_URL="http://elastic:changeme@localhost:5601" \
EDOT_ENDPOINT="http://localhost:4318" \
uv run elastic-evals run --suite claude-code-eval
```

Or run directly:

```bash
  CONNECTOR_ID="azure-gpt4_1" \
  KIBANA_URL="http://elastic:changeme@host.docker.internal:5601/dev" \
  EDOT_ENDPOINT="http://kibana-edot-collector:4318" \
  ELASTIC_EVALS_TRACING_ENDPOINT="http://kibana-edot-collector:4318/v1/traces" \
  TRACE_ES_URL="http://elastic:changeme@host.docker.internal:9200" \
  uv run elastic-evals run --suite claude-code-eval
```

## Telemetry flow

```
elastic-evals harness
  │  spans → OTLP/HTTP → localhost:4318 (EDOT)
  │
  └─ spawns: claude --print --output-format stream-json
               │  metrics → OTLP/HTTP → localhost:4318 (EDOT)
               │  logs    → OTLP/HTTP → localhost:4318 (EDOT)
               └─ (spans if trace propagation works — see below)
```

Both services share the same `elastic.evals.run_id` resource attribute, so you can
filter by run ID in APM / Discover to correlate all signals from a single eval run.

## Trace propagation

The harness injects the current OTel span context as `TRACEPARENT` and `TRACESTATE`
environment variables before spawning each Claude Code subprocess.

**Current status**: Claude Code's Node.js OTel SDK reads trace context from HTTP headers,
not from environment variables, so strict parent-child linking does not work out of the box.
The `TRACEPARENT` env var is set anyway — if a future Claude Code release adds an env-var
propagator, spans will automatically appear as children in the eval trace.

**What works today**: correlation via `elastic.evals.run_id`. In APM, filter for:

```
resource.attributes.elastic.evals.run_id: "<your-run-id>"
```

This surfaces spans from both `elastic-evals` and `claude-code` services for the same run.

## Evaluators

| Name | Kind | What it scores |
|------|------|----------------|
| `Latency` | CODE | Wall-clock ms → FAST / GOOD / OK / SLOW / VERY_SLOW |
| `ToolUse` | CODE | Overlap between expected and actual tool names |
| `criteria` | LLM | Per-example criteria from dataset metadata (optional) |

## Tuning

| Env var | Default | Purpose |
|---------|---------|---------|
| `EDOT_ENDPOINT` | `http://localhost:4318` | OTLP endpoint for both harness and Claude Code |
| `ELASTIC_EVALS_REPETITIONS` | `1` | Number of times to repeat each example |
| `ELASTIC_EVALS_CONCURRENCY` | `5` | Parallel task slots |
| `EVALUATION_CONNECTOR_ID` | — | Separate connector for LLM scoring |
