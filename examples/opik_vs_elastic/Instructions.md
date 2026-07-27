# Instructions

Setup and run the Opik vs elastic-evals-sdk-python PoC.

## 1. Python environment

From the SDK repo root:

```bash
cd elastic-evals-sdk-python
uv sync --extra dev --extra runner --extra poc
source .venv/bin/activate
```

## 2. Dependencies

```bash
uv add --optional poc datasets pandas python-dotenv ipykernel elasticsearch opik loguru
uv sync --extra poc
```

Orca must be importable. Clone the `orca` repo as a sibling of `elastic-evals-sdk-python/` so the layout is:

```
── elastic-evals-sdk-python/
── orca/
```

Then install it editable and register it under the `poc` extra:

```bash
uv add --optional poc --editable ../orca
```

Register the venv as a Jupyter kernel. Only needed if your notebook/IDE doesn't pick up `.venv` automatically:

```bash
uv run python -m ipykernel install --user --name elastic-evals-poc
```

## 3. Secrets

Retrieve the API keys (Opik, OpenRouter, HuggingFace) from Vault, then fill them into `.env`:

```bash
VAULT_ADDR=https://secrets.elastic.co:8200 vault login --method oidc
```

## 4. GCP access

Only needed if you import data from a GCP bucket (e.g. `gs://agent-builder-data-science-datasets/...`). Authenticate with your `@elastic.co` account:

```bash
gcloud auth application-default login
```

## 5. Local stack

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

## 6. Elasticsearch API key

`run.py` authenticates to Elasticsearch with `ELASTICSEARCH_API_KEY` (and reuses the
same key for Kibana, which validates Elasticsearch API keys). Create one against your
cluster and paste the `encoded` field into `ELASTICSEARCH_API_KEY` in `.env`:

```bash
curl -u elastic:changeme -XPOST http://localhost:9200/_security/api_key \
  -H 'Content-Type: application/json' -d '{"name":"evals-poc"}'
```

This step is **optional** — skip it if you already have a valid `ELASTICSEARCH_API_KEY`
for the cluster in `ES_URL`. It's required when the key is missing or invalid, e.g.
after starting a fresh local Elasticsearch (`yarn es snapshot`): API keys are
cluster-specific, so a key from a previous cluster returns `401`.

## 7. Run the PoC

```bash
cd /Users/mafaldasavelho/Documents/work-repos/kibana-fork/evals-python-sdk/elastic-evals-sdk-python
uv run --extra poc python examples/opik_vs_elastic/run.py
```
