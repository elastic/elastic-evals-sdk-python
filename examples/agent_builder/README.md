# Agent Builder Evaluation Example

This example runs an evaluation against the Agent Builder API using ambiguous queries.

## Prerequisites

1. Running Kibana instance.
2. A configured LLM connector in Kibana.
3. Optional: `KIBANA_API_KEY` for secured Kibana deployments.

## Running

```bash
uv run elastic-evals run --suite agent-builder \
  --connector-id "your-connector-id" \
  --evaluation-connector-id "your-evaluator-connector-id" \
  --kibana-url "http://elastic:changeme@localhost:5620" \
  --tracing-endpoint "http://localhost:4318"
```

## Expected Output

The evaluation will:

1. Send dataset examples as user prompts to Agent Builder's converse API
2. Run correctness and groundedness evaluators.
3. Ingest scores via Kibana's `/internal/evals/scores` API.
4. Print a summary report.

Dataset sync note: this example uses `run_experiment()` dataset upsert behavior, which is
destructive. If an example is removed locally and re-run, that example is removed from
Kibana dataset storage.
