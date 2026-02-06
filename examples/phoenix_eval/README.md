# Phoenix Evaluation Example

This example demonstrates how to:

1. Load a dataset from Arize Phoenix
2. Run RAG metrics evaluators (Precision@K, Recall@K, F1@K)
3. Run groundedness evaluator (LLM-based)
4. Run trace-based evaluators (latency, tokens, tool calls)
5. Export evaluation results as Phoenix experiments

## Prerequisites

Install elastic-evals with Phoenix support:

```bash
pip install elastic-evals[phoenix] python-dotenv
```

## Configuration

### Option 1: Using .env file (Recommended)

Copy the example configuration and edit it:

```bash
cd examples/phoenix_eval
cp .env.example .env
# Edit .env with your values
```

The `.env` file contains all configuration options with documentation:

```env
# Required
CONNECTOR_ID=your-connector-id
KIBANA_URL=http://elastic:changeme@localhost:5601

# Phoenix
PHOENIX_BASE_URL=http://localhost:6006
PHOENIX_DATASET_NAME=my-rag-dataset

# Optional: trace-based evaluators
TRACE_ES_URL=http://localhost:9200

# Optional: multi-exporter tracing
ELASTIC_EVALS_TRACING_TARGETS=otlp,phoenix
```

### Option 2: Environment Variables

```bash
# Kibana connector for LLM inference
export CONNECTOR_ID="your-connector-id"
export KIBANA_URL="http://elastic:changeme@localhost:5601"

# Phoenix server
export PHOENIX_BASE_URL="http://localhost:6006"
export PHOENIX_DATASET_NAME="my-rag-dataset"

# For Phoenix Cloud
export PHOENIX_API_KEY="your-api-key"
export PHOENIX_BASE_URL="https://app.phoenix.arize.com/s/your-space"
```

### Trace-based Evaluators (Optional)

Trace-based evaluators (latency, tokens, tool calls) require:
1. An OpenTelemetry collector (e.g., [EDOT Collector](https://www.elastic.co/docs/reference/edot-collector)) to receive traces
2. Elasticsearch with APM data for querying traces

```bash
# Elasticsearch URL for querying trace data
TRACE_ES_URL=http://localhost:9200

# OTLP endpoint (where your OTel collector receives traces)
ELASTIC_EVALS_TRACING_ENDPOINT=http://localhost:4318/v1/traces
```

### Multi-Exporter Tracing (Optional)

Send traces to both Elasticsearch and Phoenix:

```bash
ELASTIC_EVALS_TRACING_TARGETS=otlp,phoenix
ELASTIC_EVALS_TRACING_ENDPOINT=http://localhost:4318
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
PHOENIX_PROJECT_NAME=my-evals
```

## Dataset Format

The Phoenix dataset should have examples with the following structure:

### Input

```json
{
  "question": "What is the capital of France?"
}
```

### Expected Output (for RAG evaluation)

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

## Running the Example

```bash
python examples/phoenix_eval/run.py
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

| Evaluator | Type | Description |
|-----------|------|-------------|
| Precision@K | CODE | Precision of top-K retrieved docs |
| Recall@K | CODE | Recall of top-K retrieved docs |
| F1@K | CODE | F1 score combining precision and recall |
| Groundedness | LLM | Verifies response is grounded in evidence |
| Latency | Trace | End-to-end latency from traces |
| Input Tokens | Trace | Total input tokens from traces |
| Output Tokens | Trace | Total output tokens from traces |
| Tool Calls | Trace | Number of tool calls from traces |
