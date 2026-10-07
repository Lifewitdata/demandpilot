"""Smoke tests for the forecasting service (endpoint behavior + cross-layer consistency)."""

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.data import ROOT

REGISTRY = ROOT / "models" / "registry.json"
pytestmark = pytest.mark.skipif(
    not REGISTRY.exists(), reason="models/registry.json missing -- run src.train first"
)

SKU = "85123A"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == "v1"
    assert body["horizon"] == 8
    assert 0 < body["holdout"]["wape"] < 1


def test_skus_endpoint(client):
    body = client.get("/skus").json()
    assert body["n_skus"] == 100
    assert SKU in body["skus"]


def test_forecast_shape_and_bands(client):
    resp = client.get(f"/forecast/{SKU}")
    assert resp.status_code == 200
    body = resp.json()
    weeks = body["forecast"]
    assert [w["horizon"] for w in weeks] == list(range(1, 9))
    for w in weeks:
        assert w["week_index"] == body["origin_week_index"] + w["horizon"]
        assert 0 <= w["p05"] <= w["p95"]
        assert w["units"] >= 0
        assert w["p05"] <= w["units"] + 1e-9 <= w["p95"] + 1e-9
    starts = [w["week_start"] for w in weeks]
    assert starts == sorted(starts)
    last_origin_week = body["origin_week_index"]
    assert last_origin_week > 100  # default origin is the latest stored week


def test_forecast_accepts_historical_origin(client):
    body = client.get(f"/forecast/{SKU}?origin_week=90").json()
    assert body["origin_week_index"] == 90
    assert len(body["forecast"]) == 8


def test_forecast_unknown_sku_and_origin_404(client):
    assert client.get("/forecast/NOPE").status_code == 404
    assert client.get(f"/forecast/{SKU}?origin_week=99999").status_code == 404


def test_inventory_policy_endpoint(client):
    body = client.get(f"/inventory/{SKU}").json()
    assert body["stock_code"] == SKU
    assert body["opt_cost"] <= body["naive_cost"]
    assert body["opt_safety_stock"] >= 0
    assert client.get("/inventory/NOPE").status_code == 404


def test_inventory_protection_units_match_forecast_sum(client):
    """The API forecast and the inventory layer must agree numerically."""
    foreseen = client.get(f"/forecast/{SKU}").json()["forecast"][:6]  # h=1..6
    policy = client.get(f"/inventory/{SKU}").json()
    assert abs(sum(w["units"] for w in foreseen) - policy["forecast_protection_units"]) <= 1.0
