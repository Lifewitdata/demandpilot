"""Always-on service contracts using deterministic, generated artifacts."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
import src.api as api
from src.features import FEATURE_COLUMNS

SKU = '85123A'

@pytest.fixture
def client(monkeypatch):
    codes = [SKU] + [f'S{i:03}' for i in range(99)]
    features = pd.DataFrame([
        {'stock_code': sku, 'week_index': week, 'week_start': pd.Timestamp('2011-12-05') + pd.Timedelta(weeks=week-105),
         **{c: 1.0 for c in FEATURE_COLUMNS}}
        for sku in codes for week in (90, 105)])
    artifact = {'model': SimpleNamespace(predict=lambda X: np.full(len(X),100.0)),
                'feature_columns': [*FEATURE_COLUMNS, 'h'], 'bias_factors': {h: 1.0 for h in range(1,9)},
                'horizon': 8, 'trained_through_week_index': 105}
    registry = {'version': 'test-fixture', 'horizon':8, 'n_train_pairs': 100,
                'created_at': '2026-10-28T00:00:00Z', 'holdout': {'wape': 0.5}}
    policy = pd.DataFrame([{'stock_code': SKU, 'forecast_protection_units':600.0,
      'opt_service_level':0.95, 'opt_safety_stock':100.0, 'opt_cost':12.0,
      'target_safety_stock':100.0, 'target_cost':12.0, 'naive_safety_stock':200.0,
      'naive_cost':24.0, 'naive_achieved_sl':0.8, 'savings_vs_naive_pct':50.0}])
    monkeypatch.setattr(api,'_registry',lambda:registry)
    monkeypatch.setattr(api,'_artifact',lambda:artifact)
    monkeypatch.setattr(api,'_intervals',lambda:{'per_horizon':{str(h):{'q05':-0.5,'q95':0.5} for h in range(1,9)}})
    monkeypatch.setattr(api,'_features',lambda:features)
    monkeypatch.setattr(api,'_policy',lambda:policy)
    return TestClient(api.app)


def test_health_endpoint(client):
    r=client.get('/health'); assert r.status_code==200
    assert r.json()['model_version']=='test-fixture'


def test_skus_endpoint(client):
    b=client.get('/skus').json(); assert b['n_skus']==100 and SKU in b['skus']


def test_forecast_shape_and_bands(client):
    r=client.get(f'/forecast/{SKU}'); assert r.status_code==200
    b=r.json(); assert b['origin_week_index']==105
    assert [w['horizon'] for w in b['forecast']]==list(range(1,9))
    for w in b['forecast']:
        assert w['week_index']==105+w['horizon']
        assert 0<=w['p05']<=w['units']<=w['p95']


def test_forecast_rejects_origin_before_training_cutoff(client):
    assert client.get(f'/forecast/{SKU}?origin_week=90').status_code==422


def test_unknown_sku_and_origin(client):
    assert client.get('/forecast/NOPE').status_code==404
    assert client.get(f'/forecast/{SKU}?origin_week=9999').status_code==404
    assert client.get('/inventory/NOPE').status_code==404


def test_inventory_matches_forecast(client):
    f=client.get(f'/forecast/{SKU}').json()['forecast']
    p=client.get(f'/inventory/{SKU}').json()
    assert sum(w['units'] for w in f[:6])==p['forecast_protection_units']
    assert p['opt_cost']<=p['naive_cost']


def test_missing_artifact_returns_503(client,monkeypatch):
    def missing(): raise FileNotFoundError('test artifact missing')
    monkeypatch.setattr(api,'_artifact',missing)
    assert client.get('/health').status_code==503
    assert client.get(f'/forecast/{SKU}').status_code==503
