# elastic-evals

[![License](https://img.shields.io/badge/License-Elastic%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

Python SDK for running offline LLM evaluations with Kibana connectors, Elasticsearch exports,
and OpenTelemetry tracing. This mirrors the `kbn-evals` framework used in Kibana, adapted
to a Python-first workflow.

## Features

- **Experiment runner** with configurable concurrency and repetitions
- **Built-in evaluators** for correctness, groundedness, criteria, RAG metrics, and trace data
- **Elasticsearch export** to `.kibana-evaluations` datastreams
- **OpenTelemetry tracing** with OTLP/HTTP exporter support
- **Terminal reporting** with Rich tables
- **CLI** for running suites, comparisons, and diagnostics
- **Suite discovery plugins** via Python entry points

## Installation

### Using pip

```bash
pip install elastic-evals
```

### With Phoenix support

To use Arize Phoenix for trace export or dataset loading:

```bash
pip install elastic-evals[phoenix]
```

### Using uv (recommended)

```bash
uv add elastic-evals
# With Phoenix support:
uv add elastic-evals[phoenix]
```

### From source

```bash
git clone https://github.com/elastic/elastic-evals.git
cd elastic-evals
uv sync
```

## Quick start

Set required environment variables:

```bash
export CONNECTOR_ID="your-connector-id"
export KIBANA_AUTH="base64-user-pass"
export KIBANA_URL="http://localhost:5620"
```

Minimal example with a custom evaluator:

```python
import asyncio

from elastic_evals.config import ElasticEvalsConfig
from elastic_evals.evaluators.base import SimpleEvaluator
from elastic_evals.executor import ElasticEvalsClient
from elastic_evals.types import EvaluationDataset, EvaluationResult, EvaluatorParams, Example

dataset = EvaluationDataset(
    name="example-eval",
    description="Quick start example",
    examples=[
        Example(input={"question": "What is 2 + 2?"}, output={"expected": "4"}),
    ],
)

async def task(example: Example) -> dict:
    return {"answer": "4"}

async def evaluator(params: EvaluatorParams) -> EvaluationResult:
    expected = (params.expected or {}).get("expected")
    answer = (params.output or {}).get("answer")
    return EvaluationResult(score=1.0 if answer == expected else 0.0)

async def main() -> None:
    config = ElasticEvalsConfig.from_env()
    client = ElasticEvalsClient(config)
    result = await client.run_experiment(
        dataset=dataset,
        task=task,
        evaluators=[SimpleEvaluator(name="ExactMatch", kind="CODE", evaluate=evaluator)],
    )
    print(f"Finished experiment {result.id} with {len(result.evaluation_runs)} runs")

asyncio.run(main())
```

## Core concepts

### Dataset
An `EvaluationDataset` is a collection of `Example` items with input, expected output,
and optional metadata. Each dataset is hashed into a stable ID for exports.

### Task
A task is an async function that consumes an `Example` and returns a task output
that evaluators will score.

### Evaluator
Evaluators implement `evaluate()` and return an `EvaluationResult`. They can be:

- **LLM** (LLM-as-judge): correctness, groundedness, criteria
- **CODE** (deterministic): RAG metrics like Precision@K, Recall@K, F1@K
- **Trace-based**: token usage, latency, tool calls (requires trace export + ES access)

### Experiment
An experiment is a full evaluation run combining a dataset, task, and evaluators.
Results are stored in `RanExperiment`.

## Configuration

### Environment variables

| Variable | Description | Required | Default |
| --- | --- | --- | --- |
| `KIBANA_URL` | Kibana base URL | No | `http://localhost:5601` |
| `CONNECTOR_ID` | Kibana connector ID for tasks | Yes | - |
| `EVALUATION_CONNECTOR_ID` | Connector ID for evaluator LLMs | No | - |
| `KIBANA_AUTH` | Base64 credentials (`user:pass`) | Yes | - |
| `EVALUATIONS_ES_URL` | Elasticsearch URL for scores | No | - |
| `TRACE_ES_URL` | Elasticsearch URL for traces | No | - |
| `ELASTIC_EVALS_RUN_ID` | Override run ID | No | UUID |
| `ELASTIC_EVALS_REPETITIONS` | Number of repetitions | No | `3` |
| `ELASTIC_EVALS_CONCURRENCY` | Concurrency level | No | `5` |
| `ELASTIC_EVALS_LOG_LEVEL` | Log level | No | `INFO` |
| `ELASTIC_EVALS_MODEL` | JSON model metadata override | No | - |
| `ELASTIC_EVALS_TRACING_ENABLED` | Enable tracing (`true`/`false`) | No | `true` |
| `ELASTIC_EVALS_TRACING_EXPORTER` | Tracing exporter (`otlp`, `phoenix`, `console`, `none`) | No | `otlp` |
| `ELASTIC_EVALS_TRACING_ENDPOINT` | OTLP/HTTP endpoint | No | `http://localhost:4318/v1/traces` |
| `ELASTIC_EVALS_TRACING_SERVICE_NAME` | Tracing service name | No | `elastic-evals` |
| `ELASTIC_EVALS_TRACING_TARGETS` | Comma-separated exporters for multi-export | No | - |
| `ELASTIC_EVALS_TRACING_EXPORTERS` | JSON array of exporter configs | No | - |
| `PHOENIX_COLLECTOR_ENDPOINT` | Phoenix server URL | No | `http://localhost:6006` |
| `PHOENIX_PROJECT_NAME` | Phoenix project name | No | - |
| `PHOENIX_API_KEY` | Phoenix Cloud API key | No | - |
| `ELASTIC_EVALS_PHOENIX_USE_GRPC` | Use gRPC for Phoenix (`true`/`false`) | No | `false` |
| `ELASTIC_EVALS_PHOENIX_EXPERIMENT_EXPORT` | Export experiments to Phoenix (`true`/`false`) | No | `false` |
| `SELECTED_EVALUATORS` | Comma-separated evaluator names | No | - |
| `RAG_EVAL_K` | Override RAG K | No | - |
| `INDEX_FOCUSED_RAG_EVAL` | Restrict to ground-truth indices (`true`) | No | - |

## Multi-exporter tracing

Send traces to multiple destinations simultaneously (e.g., both Elasticsearch APM and Arize Phoenix).

### Using environment variables

```bash
# Option 1: Comma-separated targets (simple)
export ELASTIC_EVALS_TRACING_TARGETS=otlp,phoenix
export ELASTIC_EVALS_TRACING_ENDPOINT=http://localhost:4318
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
export PHOENIX_PROJECT_NAME=my-evals

# Option 2: Full JSON configuration (flexible)
export ELASTIC_EVALS_TRACING_EXPORTERS='[
  {"type":"otlp","endpoint":"http://localhost:4318"},
  {"type":"phoenix","endpoint":"http://localhost:6006","project_name":"my-evals"}
]'
```

### Using Python code

```python
from elastic_evals.tracing import TracingConfig, ExporterConfig, init_tracing

config = TracingConfig(
    exporters=[
        ExporterConfig(type="otlp", endpoint="http://localhost:4318"),
        ExporterConfig(
            type="phoenix",
            endpoint="http://localhost:6006",
            project_name="my-evals",
            api_key="your-api-key",  # Optional, for Phoenix Cloud
        ),
    ],
    service_name="elastic-evals",
)
init_tracing(config)
```

### Phoenix Cloud

For Phoenix Cloud, set your API key and endpoint:

```bash
export PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/your-space
export PHOENIX_API_KEY=your-api-key
export ELASTIC_EVALS_TRACING_EXPORTER=phoenix
```

## Loading datasets from Phoenix

Load datasets directly from Arize Phoenix for use in evaluations.

### Installation

```bash
pip install elastic-evals[phoenix]
```

### Using environment variables

```bash
export PHOENIX_BASE_URL=http://localhost:6006
export PHOENIX_API_KEY=your-api-key  # Optional, for Phoenix Cloud
```

### Loading datasets

```python
from elastic_evals.datasets import load_dataset_from_phoenix, list_phoenix_datasets

# List available datasets
datasets = list_phoenix_datasets()
for ds in datasets:
    print(f"{ds['name']}: {ds['example_count']} examples")

# Load a specific dataset
dataset = load_dataset_from_phoenix("customer-support-qa")

# Use in evaluation
result = await client.run_experiment(
    dataset=dataset,
    task=my_task,
    evaluators=my_evaluators,
)
```

### With custom configuration

```python
from elastic_evals.datasets import load_dataset_from_phoenix, PhoenixDatasetConfig

config = PhoenixDatasetConfig(
    base_url="https://app.phoenix.arize.com/s/your-space",
    api_key="your-api-key",
)

dataset = load_dataset_from_phoenix("my-dataset", config=config)
```

## Exporting experiments to Phoenix

After running an evaluation, you can export the results to Phoenix Experiments for
persistence, visualization, and comparison in the Phoenix UI.

### Using environment variables

```bash
# Enable Phoenix experiment export
export ELASTIC_EVALS_PHOENIX_EXPERIMENT_EXPORT=true
export PHOENIX_BASE_URL=http://localhost:6006
```

### Using Python code

```python
from elastic_evals.datasets import load_dataset_from_phoenix
from elastic_evals.export.phoenix_experiments import export_experiment_to_phoenix

# Load dataset from Phoenix (preserves Phoenix dataset ID)
dataset = load_dataset_from_phoenix("my-dataset")

# Run your evaluation
experiment = await client.run_experiment(
    dataset=dataset,
    task=my_task,
    evaluators=my_evaluators,
)

# Export results to Phoenix Experiments
result = await export_experiment_to_phoenix(
    experiment=experiment,
    dataset=dataset,
    experiment_name="My Evaluation Run",
)

print(f"Phoenix Experiment ID: {result.experiment_id}")
print(f"View at: {result.experiment_url}")
```

This syncs your evaluation results to Phoenix without re-running the experiment.
The Phoenix Experiments UI will show all task outputs and evaluator scores.

## Evaluators reference

### LLM evaluators
- **Correctness analysis**: structured judgment of factuality, relevance, and sequence accuracy
- **Groundedness analysis**: verifies claims against tool-call evidence
- **Criteria evaluator**: custom PASS/FAIL/N/A criteria for arbitrary checks

### CODE evaluators
- **RAG metrics**: Precision@K, Recall@K, and F1@K for retrieval quality

### Trace-based evaluators
- **Input/Output/Cached tokens**
- **Latency** (overall and span-level)
- **Tool call counts**

Trace-based evaluators require traces exported via OTLP and access to trace data in ES
(set `TRACE_ES_URL`).

## Elasticsearch export

Exports target the `.kibana-evaluations` data stream. The schema includes:

- `@timestamp`
- `run_id`, `experiment_id`
- `example` (id, index, input_hash, dataset)
- `task` (trace_id, repetition_index, model)
- `evaluator` (name, score, label, explanation, metadata, model)
- `run_metadata` (git_branch, git_commit_sha, total_repetitions)
- `environment` (hostname)

To export, build documents with `build_flattened_score_documents()` and send them via
`EvaluationScoreRepository` (see `examples/agent_builder/run.py` for a reference).

## CLI

```bash
elastic-evals run <script.py>        # Run evaluation script
elastic-evals run --suite <suite>    # Run a registered suite plugin
elastic-evals list                   # List suites and example scripts
elastic-evals compare <id1> <id2>    # Compare runs with paired t-tests
elastic-evals doctor                 # Check prerequisites
elastic-evals env                    # Show supported env vars
```

### Suite discovery plugins

Suites are discovered via Python entry points:

```toml
[project.entry-points."elastic_evals.suites"]
my-suite = "my_package.my_suite:get_suite"
```

Entry points return an `EvaluationSuite` instance (sync or async):

```python
from elastic_evals.suites import EvaluationSuite

def get_suite() -> EvaluationSuite:
    return EvaluationSuite(
        id="my-suite",
        description="My evaluation suite.",
        run=run_suite,
    )
```

### Built-in example: Agent Builder

The repo registers a suite plugin at `examples/agent_builder/suite_plugin.py`:

```toml
[project.entry-points."elastic_evals.suites"]
agent-builder = "examples.agent_builder.suite_plugin:get_suite"
```

Run it with:

```bash
elastic-evals run --suite agent-builder \
  --connector-id "<connector-id>" \
  --evaluation-connector-id "<evaluator-connector-id>" \
  --evaluations-es-url "http://elastic:changeme@localhost:9220" \
  --kibana-url "http://localhost:5620" \
  --kibana-auth "ZWxhc3RpYzpjaGFuZ2VtZQ==" \
  --tracing-exporter "otlp" \
  --tracing-endpoint "http://localhost:4320"
```

## Examples

- [Agent Builder](examples/agent_builder/) - Evaluate Agent Builder responses
- [Phoenix Evaluation](examples/phoenix_eval/) - Load datasets from Phoenix and run RAG, groundedness, and trace-based evaluators

## License

Elastic License 2.0