# Instructions

This PoC demonstrates how Agent Builder can use the `kbn/evals` Python SDK for dataset
management, experiment tracing, and evaluator score ingestion—capabilities previously
handled through Opik—and how to build custom evaluators.

## 1. Dependencies

`run2.py` uses external Orca evaluators. Clone the `orca` repo as a sibling of
`elastic-evals-sdk-python/` so the layout is:

```
── elastic-evals-sdk-python/
── orca/
```

## 2. Python environment

From the SDK repo root:

```bash
uv sync --group dev --extra runner --extra poc
source .venv/bin/activate
uv pip install --editable ../orca
```

Register the venv as a Jupyter kernel. Only needed if your notebook/IDE doesn't pick up `.venv` automatically:

```bash
uv run --no-sync python -m ipykernel install --user --name elastic-evals-poc
```

## 3. Secrets

Create `.env` next to `.env.example`:

```bash
cp examples/opik_vs_elastic/.env.example examples/opik_vs_elastic/.env
```

Set the local URLs, `ELASTICSEARCH_API_KEY`, `KIBANA_API_KEY`, `CONNECTOR_ID`,
and `EVALUATION_CONNECTOR_ID`. The Opik variables are used when `run2.py` runs
the tracked external Orca evaluators. Retrieve internal credentials from Vault
when needed:

```bash
VAULT_ADDR=https://secrets.elastic.co:8200 vault login --method oidc
```

The public Hugging Face dataset does not require an API key.

## 4. Data source and sample size

Set these values near the top of the script before running it:

```python
USE_ENTIRE_DATASET = False
DATASET_SAMPLE_SIZE = 10
USE_GCP = False
```

With `USE_GCP = False`, the scripts load the public `Wix/WixQA` dataset and
knowledge base from Hugging Face. Set `USE_GCP = True` to use the internal GCS
files. `DATASET_SAMPLE_SIZE` is ignored when `USE_ENTIRE_DATASET` is `True`.
The entire knowledge base is always indexed.

The available examples and their order may differ between Hugging Face and GCS.
`run.py` defaults to 10 examples and `run2.py` defaults to 3.

## 5. GCP access

Only needed when `USE_GCP = True`. Authenticate with your `@elastic.co` account:

```bash
gcloud auth application-default login
```

## 6. Local stack

Use a separate terminal for each service and leave it running.

In `kibana/config/kibana.dev.yml`, enable evals and OTLP tracing:

```yaml
server.basePath: /dev
xpack.evals.enabled: true
elastic.apm.active: false
elastic.apm.contextPropagationOnly: false
telemetry.enabled: true
telemetry.tracing.enabled: true
telemetry.tracing.sample_rate: 1
telemetry.tracing.exporters:
  - http:
      url: http://localhost:4318/v1/traces
uiSettings:
  overrides:
    agentBuilder:experimentalFeatures: true
```

### Elasticsearch

```bash
cd /Users/mafaldasavelho/Documents/work-repos/kibana-fork/kibana
nvm use
yarn es snapshot --license trial
```

Check Elasticsearch:

```bash
curl --user elastic:changeme \
  --max-time 10 \
  'http://localhost:9200/_cluster/health?wait_for_status=yellow&timeout=60s'
```

### EDOT collector

Start Docker Desktop, then run:

```bash
cd /Users/mafaldasavelho/Documents/work-repos/kibana-fork/kibana
nvm use
node scripts/edot_collector.js
```

Check EDOT:

```bash
docker ps \
  --filter name=kibana-edot-collector \
  --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
```

### Kibana

```bash
cd /Users/mafaldasavelho/Documents/work-repos/kibana-fork/kibana
nvm use
node scripts/kibana --dev --verbose
```

Check Kibana:

```bash
curl --silent --show-error \
  --output /dev/null \
  --write-out '%{http_code}\n' \
  --max-time 15 \
  http://localhost:5601/dev/api/status
```

## 7. Elasticsearch API key

Both scripts authenticate to Elasticsearch and Kibana. Create an API key against
the local cluster and paste the `encoded` field into both
`ELASTICSEARCH_API_KEY` and `KIBANA_API_KEY` in `.env`:

```bash
curl -u elastic:changeme -XPOST http://localhost:9200/_security/api_key \
  -H 'Content-Type: application/json' -d '{"name":"evals-poc"}'
```

This step is optional if both variables already contain a valid key for the
current cluster. API keys are cluster-specific, so a key from a previous local
Elasticsearch snapshot returns `401`.

## 8. Run the PoC

### `run.py`: managed workflow

Demonstrates the higher-level workflow. It uses
`ElasticEvalsClient.run_experiment()` to execute the selected WixQA examples,
run SDK-side and custom evaluators, and ingest their scores.

```bash
cd /Users/mafaldasavelho/Documents/work-repos/kibana-fork/evals-python-sdk/elastic-evals-sdk-python
uv run --no-sync python -m examples.opik_vs_elastic.run
```

### `run2.py`: granular workflow

Demonstrates the lower-level workflow without `run_experiment()`. It directly
coordinates the Dataset, Evaluators, and Score Ingestion APIs, runs the custom
Document Recall evaluator, and attaches external Orca scores.

```bash
uv run --no-sync python -m examples.opik_vs_elastic.run2
```
