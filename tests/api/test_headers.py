from elastic_evals.api import build_kibana_headers


def test_build_kibana_headers_without_api_key() -> None:
    headers = build_kibana_headers(None)

    assert headers["Content-Type"] == "application/json"
    assert headers["kbn-xsrf"] == "true"
    assert headers["x-elastic-internal-origin"] == "true"
    assert headers["Elastic-Api-Version"] == "1"
    assert "Authorization" not in headers


def test_build_kibana_headers_with_api_key() -> None:
    headers = build_kibana_headers("my-key")

    assert headers["Content-Type"] == "application/json"
    assert headers["kbn-xsrf"] == "true"
    assert headers["x-elastic-internal-origin"] == "true"
    assert headers["Elastic-Api-Version"] == "1"
    assert headers["Authorization"] == "ApiKey my-key"


def test_build_kibana_headers_with_empty_api_key() -> None:
    headers = build_kibana_headers("")

    assert headers["Content-Type"] == "application/json"
    assert headers["kbn-xsrf"] == "true"
    assert headers["x-elastic-internal-origin"] == "true"
    assert headers["Elastic-Api-Version"] == "1"
    assert "Authorization" not in headers
