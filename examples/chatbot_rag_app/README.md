# Chatbot RAG App Evaluation Example

This example evaluates Elastic's published `chatbot-rag-app` Docker image and exports
scores/traces through the same OTLP pipeline used by `elastic-evals`.

## Prerequisites

1. Running Elasticsearch and Kibana.
2. Running EDOT collector reachable at `http://localhost:4318`.
3. A configured LLM connector in Kibana for the task model.
4. A configured LLM connector in Kibana for evaluator scoring.

## Setup

From `examples/chatbot_rag_app/`:

```bash
cp .env.example .env
```

Then edit `.env` and set:

- `LLM_TYPE` and the corresponding provider keys.
- `ELASTICSEARCH_URL`, `ELASTICSEARCH_USER`, and `ELASTICSEARCH_PASSWORD` for your stack.

## Start the app

```bash
./start_app.sh
```

Or run compose directly:

```bash
docker compose --env-file .env up -d --wait
```

## Run the eval

From the repository root:

```bash
uv run elastic-evals run --suite chatbot-rag-eval-example \
  --connector-id "your-connector-id" \
  --evaluation-connector-id "your-evaluator-connector-id" \
  --evaluations-es-url "http://elastic:changeme@localhost:9220" \
  --kibana-url "http://elastic:changeme@localhost:5620" \
  --tracing-endpoint "http://localhost:4318"
```

## Cleanup

From `examples/chatbot_rag_app/`:

```bash
docker compose down
```

## Kibana check

- In APM, confirm traces appear for both `chatbot-rag-app` and `elastic-evals`.
- In Discover, use the `.kibana-evaluations` data view and confirm new score docs:
  - one `criteria` result per example
  - one `SourceCitation` result per example

## Notes

This example intentionally omits groundedness evaluation because the chatbot output does not
include the tool-call evidence shape that groundedness expects.
