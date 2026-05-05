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

### Using uv (recommended)

```bash
uv add elastic-evals
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
export KIBANA_URL="http://elastic:changeme@localhost:5620"
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

| Variable                                  | Description                                             | Required | Default                           |
| ----------------------------------------- | ------------------------------------------------------- | -------- | --------------------------------- |
| `KIBANA_URL`                              | Kibana base URL                                         | No       | `http://localhost:5601`           |
| `CONNECTOR_ID`                            | Kibana connector ID for tasks                           | Yes      | -                                 |
| `EVALUATION_CONNECTOR_ID`                 | Connector ID for evaluator LLMs                         | No       | -                                 |
| `EVALUATIONS_ES_URL`                      | Elasticsearch URL for scores                            | Yes      | -                                 |
| `TRACE_ES_URL`                            | Elasticsearch URL for traces                            | No       | -                                 |
| `ELASTIC_EVALS_RUN_ID`                    | Override run ID                                         | No       | UUID                              |
| `ELASTIC_EVALS_REPETITIONS`               | Number of repetitions                                   | No       | `3`                               |
| `ELASTIC_EVALS_CONCURRENCY`               | Concurrency level                                       | No       | `5`                               |
| `ELASTIC_EVALS_LOG_LEVEL`                 | Log level                                               | No       | `INFO`                            |
| `ELASTIC_EVALS_MODEL`                     | JSON model metadata override                            | No       | -                                 |
| `ELASTIC_EVALS_TRACING_ENABLED`           | Enable tracing (`true`/`false`)                         | No       | `true`                            |
| `ELASTIC_EVALS_TRACING_EXPORTER`          | Tracing exporter (`otlp`, `console`, `none`)            | No       | `otlp`                            |
| `ELASTIC_EVALS_TRACING_ENDPOINT`          | OTLP/HTTP endpoint                                      | No       | `http://localhost:4318/v1/traces` |
| `ELASTIC_EVALS_TRACING_SERVICE_NAME`      | Tracing service name                                    | No       | `elastic-evals`                   |
| `ELASTIC_EVALS_TRACING_TARGETS`           | Comma-separated exporters for multi-export              | No       | -                                 |
| `ELASTIC_EVALS_TRACING_EXPORTERS`         | JSON array of exporter configs                          | No       | -                                 |
| `SELECTED_EVALUATORS`                     | Comma-separated evaluator names                         | No       | -                                 |
| `RAG_EVAL_K`                              | Override RAG K                                          | No       | -                                 |
| `INDEX_FOCUSED_RAG_EVAL`                  | Restrict to ground-truth indices (`true`)               | No       | -                                 |

## Multi-exporter tracing

Send traces to multiple destinations simultaneously using either:

- `ELASTIC_EVALS_TRACING_TARGETS` for a simple comma-separated list of targets.
- `ELASTIC_EVALS_TRACING_EXPORTERS` for full JSON exporter configuration.

### Using environment variables

```bash
# Option 1: Comma-separated targets (simple)
export ELASTIC_EVALS_TRACING_TARGETS=otlp,console
export ELASTIC_EVALS_TRACING_ENDPOINT=http://localhost:4318

# Option 2: Full JSON configuration (flexible)
export ELASTIC_EVALS_TRACING_EXPORTERS='[
  {"type":"otlp","endpoint":"http://localhost:4318"},
  {"type":"otlp","endpoint":"http://localhost:14318/v1/traces"}
]'
```

### Using Python code

```python
from elastic_evals.tracing import TracingConfig, ExporterConfig, init_tracing

config = TracingConfig(
    exporters=[
        ExporterConfig(type="otlp", endpoint="http://localhost:4318"),
        ExporterConfig(type="console"),
    ],
    service_name="elastic-evals",
)
init_tracing(config)
```

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

Trace-based evaluators require:

1. An OpenTelemetry collector (e.g., [EDOT Collector](https://www.elastic.co/docs/reference/edot-collector)) running to receive traces
2. Tracing enabled with `ELASTIC_EVALS_TRACING_ENDPOINT` pointing to the collector
3. Elasticsearch with APM data accessible via `TRACE_ES_URL`

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
  --kibana-url "http://elastic:changeme@localhost:5620" \
  --tracing-exporter "otlp" \
  --tracing-endpoint "http://localhost:4320"
```

## Examples

- [Agent Builder](examples/agent_builder/) - Evaluate Agent Builder responses

## Development

Install dev dependencies and set up pre-commit hooks for automatic linting and formatting:

```bash
uv sync --extra dev
uv run pre-commit install
```

Pre-commit runs [ruff](https://docs.astral.sh/ruff/) for linting and formatting on every commit. You can also run them manually:

```bash
uv run ruff check src/       # lint
uv run ruff format src/      # format
uv run pre-commit run --all-files  # run all hooks
```

## License

Elastic License 2.0
