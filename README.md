# elastic-evals

[![License](https://img.shields.io/badge/License-Elastic%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

Python SDK for running offline LLM evaluations with Kibana connectors, Kibana evals API ingestion,
and OpenTelemetry tracing. This mirrors the `kbn-evals` framework used in Kibana, adapted
to a Python-first workflow.

## Features

- **Experiment runner** with configurable concurrency and repetitions
- **Built-in evaluators** for correctness, groundedness, and criteria
- **Score ingestion** through Kibana evals plugin and its APIs
- **OpenTelemetry tracing** with OTLP/HTTP endpoint configuration
- **Optional CLI** (`elastic-evals[runner]`) for `run` and `list`
- **Suite discovery plugins** via Python entry points in the runner package

## Installation

This package is not yet published to PyPI. Install directly from GitHub:

### Using uv (recommended)

Existing environment (notebooks, a venv that already exists):

```bash
# Core SDK
uv pip install "git+https://github.com/elastic/elastic-evals-sdk-python.git"

# SDK + CLI runner
uv pip install "elastic-evals[runner] @ git+https://github.com/elastic/elastic-evals-sdk-python.git"
```

Google Colab (caps OpenTelemetry so it stays compatible with preinstalled `google-adk`):

```bash
uv pip install "elastic-evals[colab] @ git+https://github.com/elastic/elastic-evals-sdk-python.git"
```

For a uv project (a directory that already has pyproject.toml):

```bash
# Core SDK
uv add "git+https://github.com/elastic/elastic-evals-sdk-python.git"

# SDK + CLI runner
uv add "elastic-evals[runner] @ git+https://github.com/elastic/elastic-evals-sdk-python.git"
```

### Using pip

```bash
# Core SDK
pip install "git+https://github.com/elastic/elastic-evals-sdk-python.git"

# SDK + CLI runner
pip install "elastic-evals[runner] @ git+https://github.com/elastic/elastic-evals-sdk-python.git"

# Google Colab
pip install "elastic-evals[colab] @ git+https://github.com/elastic/elastic-evals-sdk-python.git"
```

### From source

```bash
git clone https://github.com/elastic/elastic-evals-sdk-python.git
cd elastic-evals-sdk-python
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
and optional metadata. Before each run, the dataset is upserted to Kibana and refreshed
from Kibana's canonical dataset examples.

### Task

A task is an async function that consumes an `Example` and returns a task output
that evaluators will score.

### Evaluator

Evaluators implement `evaluate()` and return an `EvaluationResult`. They can be:

- **LLM** (LLM-as-judge): correctness, groundedness, criteria
- **CODE** (deterministic): custom evaluators via `SimpleEvaluator`

### Experiment

An experiment is a full evaluation run combining a dataset, task, and evaluators.
Results are stored in `RanExperiment`.

## Configuration

### Environment variables

| Variable                             | Description                                              | Required | Default                           |
| ------------------------------------ | -------------------------------------------------------- | -------- | --------------------------------- |
| `KIBANA_URL`                         | Kibana base URL                                          | No       | `http://localhost:5601`           |
| `KIBANA_API_KEY`                     | API key with `evals` plugin privilege for secured Kibana | No       | -                                 |
| `CONNECTOR_ID`                       | Kibana connector ID for tasks                            | Yes      | -                                 |
| `EVALUATION_CONNECTOR_ID`            | Connector ID for evaluator LLMs                          | No       | -                                 |
| `ELASTICSEARCH_URL`                  | Elasticsearch URL for trace lookup                       | No       | -                                 |
| `ELASTICSEARCH_API_KEY`              | API key for Elasticsearch trace lookup                   | No       | -                                 |
| `ELASTIC_EVALS_RUN_ID`               | Override run ID                                          | No       | UUID                              |
| `ELASTIC_EVALS_REPETITIONS`          | Number of repetitions                                    | No       | `1`                               |
| `ELASTIC_EVALS_CONCURRENCY`          | Concurrency level                                        | No       | `5`                               |
| `ELASTIC_EVALS_LOG_LEVEL`            | Log level                                                | No       | `INFO`                            |
| `ELASTIC_EVALS_MODEL`                | JSON model metadata override                             | No       | -                                 |
| `ELASTIC_EVALS_TRACING_ENABLED`      | Enable tracing (`true`/`false`)                          | No       | `true`                            |
| `ELASTIC_EVALS_TRACING_EXPORTER`     | Tracing exporter (`otlp`, `console`, `none`)             | No       | `otlp`                            |
| `ELASTIC_OTLP_ENDPOINT`              | OTLP/HTTP base endpoint                                  | No       | `http://localhost:4318`           |
| `ELASTIC_OTLP_API_KEY`              | API key used for OTLP Authorization header               | No       | -                                 |
| `ELASTIC_EVALS_TRACING_SERVICE_NAME` | Tracing service name                                     | No       | `elastic-evals`                   |

## Evaluators reference

### LLM evaluators

- **Correctness analysis**: structured judgment of factuality, relevance, and sequence accuracy
- **Groundedness analysis**: verifies claims against tool-call evidence
- **Criteria evaluator**: custom PASS/FAIL/N/A criteria for arbitrary checks

## Score ingestion

`elastic-evals` posts scores to `POST /internal/evals/scores` once per evaluator result.
The SDK no longer writes score documents directly to Elasticsearch.

## Datasets

`run_experiment()` syncs datasets to Kibana before each run by calling
`POST /internal/evals/datasets/_upsert`, then reads canonical examples from
`GET /internal/evals/datasets/{dataset_id}`.

**This sync is destructive**: examples missing from a later run's dataset payload are deleted
from Kibana dataset storage.

## Evaluators API

`KibanaEvaluatorsClient` exposes the four raw Kibana evaluator endpoints: list evaluators,
resolve trace instrumentation, validate evaluator evidence, and evaluate a trace. It returns
per-evaluator errors from a successful evaluation as typed result items rather than raising them.

The following manual smoke test requires an indexed trace and a Kibana connector. The API key
needs `read_evals` for listing and `manage_evals` for the other operations.

```python
import asyncio
import os

from elastic_evals.api import (
    EvaluateEvaluatorConfig,
    EvaluateRequest,
    EvaluationInstrumentation,
    EvaluationSubject,
    EvaluationTrace,
    KibanaEvaluatorsClient,
    ValidateEvaluatorConfig,
    ValidateEvaluatorsRequest,
)


async def main() -> None:
    trace_id = os.environ["TRACE_ID"]
    connector_id = os.environ["EVALUATION_CONNECTOR_ID"]
    client = KibanaEvaluatorsClient(
        os.environ.get("KIBANA_URL", "http://localhost:5601"),
        api_key=os.environ.get("KIBANA_API_KEY"),
    )

    available = await client.list_evaluators()
    names = {evaluator.name for evaluator in available.evaluators}
    expected_names = {
        "groundedness",
        "correctness",
        "latency",
        "input_tokens",
        "output_tokens",
        "tool_calls",
    }
    print("evaluators:", sorted(names))
    assert expected_names <= names

    resolved = await client.resolve_instrumentation(trace_id)
    if resolved.recommended_instrumentation is None:
        raise RuntimeError("No instrumentation profile found for this trace")
    profile = resolved.recommended_instrumentation.profile
    print("instrumentation:", profile)

    subject = EvaluationSubject(
        traces=[
            EvaluationTrace(
                trace_id=trace_id,
                reference_data={"expected": "The expected answer for this trace"},
            )
        ],
        instrumentation=EvaluationInstrumentation(profile=profile),
    )
    validation = await client.validate(
        ValidateEvaluatorsRequest(
            subject=subject,
            evaluators=[
                ValidateEvaluatorConfig(name="latency"),
                ValidateEvaluatorConfig(name="correctness"),
            ],
        )
    )
    for evaluator in validation.evaluators:
        print("validation:", evaluator.name, evaluator.ready, evaluator.unmet)

    evaluation = await client.evaluate(
        EvaluateRequest(
            subject=subject,
            evaluators=[
                EvaluateEvaluatorConfig(name="latency"),
                EvaluateEvaluatorConfig(name="correctness", connector_id=connector_id),
            ],
        )
    )
    for result in evaluation.results:
        if result.status == "ok":
            for score in result.scores or []:
                print("score:", result.evaluator.name, score.name, score.score, score.label)
        else:
            print("error:", result.evaluator.name, result.error)


asyncio.run(main())
```

Correctness returns three sub-scores: `factuality`, `relevance`, and `sequence_accuracy`.
The client retries transient failures, so an ambiguous `_evaluate` failure can repeat LLM
inference work and cost.

## CLI

```bash
elastic-evals run <script.py>        # Run evaluation script
elastic-evals run --suite <suite>    # Run a registered suite plugin
elastic-evals list                   # List suites and example scripts
```

### Suite discovery plugins

Suites are discovered via Python entry points:

```toml
[project.entry-points."elastic_evals.suites"]
my-suite = "my_package.my_suite:get_suite"
```

Entry points return an `EvaluationSuite` instance (sync or async):

```python
from elastic_evals.runner.suites import EvaluationSuite

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
  --kibana-url "http://elastic:changeme@localhost:5620" \
  --tracing-exporter "otlp" \
  --tracing-endpoint "http://localhost:4320"
```

## Examples

- [Agent Builder](examples/agent_builder/) - Evaluate Agent Builder responses
- [Chatbot RAG App](examples/chatbot_rag_app/) - Evaluate the Dockerized chatbot-rag-app with criteria and source-citation scoring

## Development

Install dev dependencies and set up pre-commit hooks for automatic linting and formatting:

```bash
uv sync
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
