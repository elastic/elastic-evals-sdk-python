# Agent Builder Evaluation Example

This example runs an evaluation against the Agent Builder API using ambiguous queries.

## Prerequisites

1. Running Kibana and Elasticsearch instance.
2. A configured LLM connector in Kibana.

## Running

```bash
uv run elastic-evals run --suite agent-builder \
  --connector-id "your-connector-id" \
  --evaluation-connector-id "your-evaluator-connector-id" \
  --evaluations-es-url "http://elastic:changeme@localhost:9200" \
  --kibana-url "http://elastic:changeme@localhost:5620" \
  --tracing-endpoint "http://localhost:4318"
```

## Expected Output

The evaluation will:

1. Send dataset examples as user prompts to Agent Builder's converse API
2. Run correctness and groundedness evaluators.
3. Export results to Elasticsearch.
4. Print a summary report.
