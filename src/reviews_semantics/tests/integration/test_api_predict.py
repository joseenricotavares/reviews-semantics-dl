from fastapi.testclient import TestClient
from reviews_semantics.api.app import create_app


def _client(bilstm_bundle_dir, monkeypatch) -> TestClient:
    monkeypatch.setenv("MODEL_SOURCE", "local")
    monkeypatch.setenv("MODEL_PATH", str(bilstm_bundle_dir))
    return TestClient(create_app())


def test_healthz_never_depends_on_the_model(bilstm_bundle_dir, monkeypatch):
    with _client(bilstm_bundle_dir, monkeypatch) as client:
        assert client.get("/healthz").json() == {"status": "ok"}


def test_readyz_is_ready_once_the_model_has_loaded(bilstm_bundle_dir, monkeypatch):
    with _client(bilstm_bundle_dir, monkeypatch) as client:
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}


def test_predict_one_returns_a_valid_star_score(bilstm_bundle_dir, monkeypatch):
    with _client(bilstm_bundle_dir, monkeypatch) as client:
        response = client.post("/v1/predictions", json={"text": "produto otimo chegou rapido"})
        assert response.status_code == 200
        assert response.json()["score"] in range(1, 6)


def test_predict_batch_returns_one_score_per_input(bilstm_bundle_dir, monkeypatch):
    with _client(bilstm_bundle_dir, monkeypatch) as client:
        response = client.post("/v1/predictions/batch", json={"texts": ["produto otimo", "produto ruim"]})
        assert response.status_code == 200
        assert len(response.json()["scores"]) == 2


def test_model_info_exposes_the_bundle_manifest(bilstm_bundle_dir, monkeypatch):
    with _client(bilstm_bundle_dir, monkeypatch) as client:
        response = client.get("/v1/model/info")
        assert response.status_code == 200
        body = response.json()
        assert body["flavor"] == "bilstm"
        assert body["classes"] == [1, 2, 3, 4, 5]


def test_predict_rejects_empty_text(bilstm_bundle_dir, monkeypatch):
    with _client(bilstm_bundle_dir, monkeypatch) as client:
        response = client.post("/v1/predictions", json={"text": ""})
        assert response.status_code == 422


def test_predict_batch_rejects_oversized_payload(bilstm_bundle_dir, monkeypatch):
    with _client(bilstm_bundle_dir, monkeypatch) as client:
        response = client.post("/v1/predictions/batch", json={"texts": ["texto"] * 300})
        assert response.status_code == 422
