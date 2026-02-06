# Phoenix Evaluation Example

This example demonstrates how to:

1. Load a dataset from Arize Phoenix
2. Run RAG metrics evaluators (Precision@K, Recall@K, F1@K)
3. Run groundedness evaluator (LLM-based)
4. Run trace-based evaluators (latency, tokens, tool calls)
5. Export evaluation results as Phoenix experiments

## Quick start

**1. Install dependencies**

From source (recommended for development):

```bash
uv sync --extra phoenix
```

Or from PyPI:

```bash
uv add elastic-evals[phoenix] python-dotenv
```

**2. Configure**

Copy the example `.env` and fill in the required values:

```bash
cp examples/phoenix_eval/.env.example examples/phoenix_eval/.env
```

At a minimum, set these in the `.env` file:

```env
CONNECTOR_ID=your-connector-id
KIBANA_URL=http://elastic:changeme@localhost:5601
PHOENIX_BASE_URL=http://localhost:6006
PHOENIX_DATASET_NAME=my-rag-dataset
```

**3. Run**

Using the CLI (registered as a suite plugin):

```bash
uv run elastic-evals run --suite phoenix-eval
```

Or run the script directly:

```bash
uv run examples/phoenix_eval/run.py
```

Both approaches load your `.env` automatically. The script loads the Phoenix dataset, runs all configured evaluators against the Kibana Agent Builder API, exports scores to Elasticsearch, and prints a summary report.

You can also override specific settings via CLI flags without editing the `.env`:

```bash
uv run elastic-evals run --suite phoenix-eval \
  --connector-id gemini-25-flash \
  --kibana-url http://elastic:changeme@localhost:5620 \
  --trace-es-url http://elastic:changeme@localhost:9220 \
  --evaluations-es-url http://elastic:changeme@localhost:9220 \
  --tracing-exporter otlp \
  --tracing-endpoint http://localhost:4320
```

CLI flags take precedence over values in the `.env` file. Phoenix-specific settings (like `PHOENIX_BASE_URL` and `PHOENIX_DATASET_NAME`) are always read from the `.env`.

## Configuration reference

The `.env.example` file documents every available option. Below are the key groups.

### Required

| Variable               | Description                                |
| ---------------------- | ------------------------------------------ |
| `CONNECTOR_ID`         | Kibana connector ID for LLM inference      |
| `KIBANA_URL`           | Kibana URL (include credentials if needed) |
| `PHOENIX_BASE_URL`     | Phoenix server URL                         |
| `PHOENIX_DATASET_NAME` | Name of the dataset to load from Phoenix   |

### Phoenix Cloud

For Phoenix Cloud instead of a local server, set:

```bash
PHOENIX_API_KEY=your-api-key
PHOENIX_BASE_URL=https://app.phoenix.arize.com/s/your-space
```

### Trace-based evaluators (optional)

Trace-based evaluators (latency, tokens, tool calls) require:

1. An OpenTelemetry collector (e.g., [EDOT Collector](https://www.elastic.co/docs/reference/edot-collector)) to receive traces
2. Elasticsearch with APM data for querying traces

```bash
TRACE_ES_URL=http://localhost:9200
ELASTIC_EVALS_TRACING_ENDPOINT=http://localhost:4318/v1/traces
```

Without `TRACE_ES_URL`, trace-based evaluators are automatically skipped.

### Multi-exporter tracing (optional)

Send traces to both Elasticsearch and Phoenix:

```bash
ELASTIC_EVALS_TRACING_TARGETS=otlp,phoenix
ELASTIC_EVALS_TRACING_ENDPOINT=http://localhost:4318
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
PHOENIX_PROJECT_NAME=my-evals
```

## Dataset format

The Phoenix dataset should have examples with the following structure:

### Input

```json
{
  "question": "What is the capital of France?"
}
```

### Expected output (for RAG evaluation)

```json
{
  "ground_truth": {
    "documents": {
      "doc-id-1": 1.0,
      "doc-id-2": 0.8
    }
  }
}
```

Or simplified format:

```json
{
  "relevant_docs": ["doc-id-1", "doc-id-2"],
  "index": "documents"
}
```

## Customization

### Modifying the Task

Edit the `rag_task()` function in `run.py` to implement your actual RAG pipeline:

```python
async def rag_task(example, config, inference_client):
    question = example.input.get("question")

    # 1. Retrieve documents from your vector store
    retrieved_docs = await your_retrieval_function(question)

    # 2. Generate response using LLM
    response = await inference_client.prompt(
        prompt="Answer based on context...",
        input_data={"question": question, "context": retrieved_docs},
    )

    return {
        "answer": response.content,
        "retrieved_docs": retrieved_docs,
        "messages": [{"message": response.content}],
        "traceId": get_current_trace_id(),
    }
```

### Modifying Ground Truth Extraction

Edit `extract_ground_truth()` to match your dataset's expected output format:

```python
def extract_ground_truth(expected):
    # Your custom extraction logic
    return {
        "index-name": {
            "doc-id": relevance_score,
            ...
        }
    }
```

## Evaluators

| Evaluator     | Type  | Description                               |
| ------------- | ----- | ----------------------------------------- |
| Precision@K   | CODE  | Precision of top-K retrieved docs         |
| Recall@K      | CODE  | Recall of top-K retrieved docs            |
| F1@K          | CODE  | F1 score combining precision and recall   |
| Groundedness  | LLM   | Verifies response is grounded in evidence |
| Latency       | Trace | End-to-end latency from traces            |
| Input Tokens  | Trace | Total input tokens from traces            |
| Output Tokens | Trace | Total output tokens from traces           |
| Tool Calls    | Trace | Number of tool calls from traces          |
