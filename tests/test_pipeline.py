import json

import pytest
from fastapi.testclient import TestClient

from src.config import FEATURES, MONITOR_SUMMARY, RETRAIN_SUMMARY
from src.data_gen import generate
from tests.conftest import run


def test_drift_knob_changes_distribution():
    base, shifted = generate(1000, seed=1, drift=0.0), generate(1000, seed=1, drift=1.0)
    assert shifted["fever"].mean() > base["fever"].mean() + 0.15
    assert shifted["loss_of_taste_smell"].mean() < base["loss_of_taste_smell"].mean()


@pytest.fixture()
def client():
    from src.api import app
    with TestClient(app) as c:
        yield c


def test_health_and_model_info(client):
    assert client.get("/health").json()["model_loaded"] is True
    assert client.get("/model-info").json()["version"] == "1"


def test_predict_and_feedback_roundtrip(client):
    r = client.post("/predict", json={"fever": 1, "cough": 1, "shortness_of_breath": 1, "loss_of_taste_smell": 1})
    body = r.json()
    assert r.status_code == 200 and body["success"]
    assert 0 <= body["confidence"] <= 100 and body["prediction"] and body["prediction_id"]
    ok = client.post("/feedback", json={"prediction_id": body["prediction_id"], "true_label": "COVID-19"})
    assert ok.status_code == 200
    bad = client.post("/feedback", json={"prediction_id": body["prediction_id"], "true_label": "Nonsense"})
    assert bad.status_code == 400


def test_predict_rejects_bad_input(client):
    assert client.post("/predict", json={"fever": 7}).status_code == 400


def test_monitor_retrain_loop():
    # healthy traffic -> no drift, retrain refuses to run
    assert run("src.simulate_traffic", "--n", "200", "--drift", "0", "--reset").returncode == 0
    assert run("src.monitor").returncode == 0
    ok = json.loads(MONITOR_SUMMARY.read_text())
    assert ok["retrain_required"] is False
    run("src.retrain")
    assert json.loads(RETRAIN_SUMMARY.read_text())["retrained"] is False

    # shifted traffic -> drift flagged, retraining runs and the gate decides
    assert run("src.simulate_traffic", "--n", "300", "--drift", "1.0", "--reset").returncode == 0
    assert run("src.monitor").returncode == 0
    bad = json.loads(MONITOR_SUMMARY.read_text())
    assert bad["retrain_required"] is True and bad["drift"]["share_of_features_drifted"] >= 0.3
    assert run("src.retrain").returncode == 0
    out = json.loads(RETRAIN_SUMMARY.read_text())
    assert out["retrained"] is True
    assert out["promoted"] == (out["f1_gain"] >= out["required_gain"])
