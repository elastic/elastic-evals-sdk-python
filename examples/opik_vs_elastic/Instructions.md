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
uv add --optional poc datasets pandas python-dotenv ipykernel elasticsearch opik
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

## 5. Elasticsearch API key

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

## 6. Run the PoC

_To be added._
