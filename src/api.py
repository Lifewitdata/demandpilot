"""FastAPI serving layer for DemandPilot forecasts and inventory policies.

Everything is served from the artifacts the pipeline already produced:

- models/registry.json + models/demand_hgb_v1.joblib  -> final model
- data/processed/features.parquet                     -> feature store
    (origin-week rows, leakage-safe by construction in src/features.py)
- models/intervals.json                               -> bias factors and
    empirical per-horizon residual quantiles (from validation folds only)
- reports/inventory_policy.csv                        -> safety-stock policy

Forecast path: take the SKU's feature row at the requested origin week
(default: latest), append the horizon h (the model is horizon-aware, one
row per (origin, h)), predict, clip at zero, apply the per-horizon bias
factor, and convert the fold residual quantiles into a p05/p95 band.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data import ROOT, load_config

app = FastAPI(
    title="DemandPilot",
    version="1.0",
    description="Weekly SKU demand forecasting and inventory policy service",
)


class ForecastWeek(BaseModel):
    horizon: int
    week_index: int
    week_start: date
    units: float
    p05: float
    p95: float


class ForecastResponse(BaseModel):
    stock_code: str
    model_version: str
    origin_week_index: int
    origin_week_start: date
    forecast: list[ForecastWeek]


class InventoryPolicy(BaseModel):
    stock_code: str
    forecast_protection_units: float
    opt_service_level: float
    opt_safety_stock: float
    opt_cost: float
    target_safety_stock: float
    target_cost: float
    naive_safety_stock: float
    naive_cost: float
    naive_achieved_sl: float
    savings_vs_naive_pct: float | None


@lru_cache
def _registry() -> dict:
    path = ROOT / "models" / "registry.json"
    if not path.exists():
        raise FileNotFoundError("models/registry.json not found -- run python -m src.train first")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache
def _artifact() -> dict:
    return joblib.load(ROOT / "models" / _registry()["artifact"])


@lru_cache
def _intervals() -> dict:
    path = ROOT / "models" / "intervals.json"
    if not path.exists():
        raise FileNotFoundError("models/intervals.json not found -- run python -m src.train first")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache
def _features() -> pd.DataFrame:
    cfg = load_config()
    path = ROOT / cfg["data"]["processed_dir"] / "features.parquet"
    if not path.exists():
        raise FileNotFoundError("features.parquet not found -- run python -m src.features first")
    return pd.read_parquet(path)


@lru_cache
def _policy() -> pd.DataFrame:
    path = ROOT / "reports" / "inventory_policy.csv"
    if not path.exists():
        raise FileNotFoundError("reports/inventory_policy.csv not found -- run python -m src.inventory first")
    return pd.read_csv(path)


def _origin_row(stock_code: str, origin_week: int | None) -> pd.Series:
    sub = _features()
    sub = sub[sub["stock_code"] == stock_code]
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"unknown stock_code {stock_code!r} -- see GET /skus")
    if origin_week is not None:
        sub = sub[sub["week_index"] == origin_week]
        if sub.empty:
            raise HTTPException(
                status_code=404,
                detail=f"no feature row for {stock_code!r} at origin_week={origin_week}",
            )
    return sub.sort_values("week_index").iloc[-1]


def _forecast_weeks(row: pd.Series) -> list[ForecastWeek]:
    artifact = _artifact()
    if int(row['week_index']) < int(artifact['trained_through_week_index']):
        raise HTTPException(status_code=422, detail='Origin predates this model training cutoff; use stored out-of-sample evaluation, not a model fitted on later targets.')
    intervals = _intervals()
    model = artifact["model"]
    base_cols = [c for c in artifact["feature_columns"] if c != "h"]
    X = row[base_cols].to_numpy(dtype=float).reshape(1, -1)

    bias_factors = artifact["bias_factors"]
    origin_start = pd.Timestamp(row["week_start"])

    weeks = []
    for h in range(1, int(artifact["horizon"]) + 1):
        X_h = np.hstack([X, np.full((1, 1), float(h))])
        yhat = float(np.clip(model.predict(X_h), 0, None)[0])
        yhat *= float(bias_factors[h])
        band = intervals["per_horizon"][str(h)]
        weeks.append(
            ForecastWeek(
                horizon=h,
                week_index=int(row["week_index"]) + h,
                week_start=(origin_start + timedelta(weeks=h)).date(),
                units=round(yhat, 2),
                p05=round(max(0.0, yhat * (1 + band["q05"])), 2),
                p95=round(max(0.0, yhat * (1 + band["q95"])), 2),
            )
        )
    return weeks


@app.get("/health")
def health() -> dict:
    try:
        registry = _registry()
        artifact = _artifact()
        _intervals()
        _features()
        _policy()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "model_version": registry["version"],
        "horizon": registry["horizon"],
        "n_train_pairs": registry["n_train_pairs"],
        "trained_at_utc": registry["created_at"],
        "holdout": registry["holdout"],
    }


@app.get("/skus")
def skus() -> dict:
    try:
        features = _features()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    codes = sorted(features["stock_code"].unique().tolist())
    return {"n_skus": len(codes), "skus": codes}


@app.get("/forecast/{stock_code}", response_model=ForecastResponse)
def forecast(stock_code: str, origin_week: int | None = None) -> ForecastResponse:
    try:
        row = _origin_row(stock_code, origin_week)
        weeks = _forecast_weeks(row)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ForecastResponse(
        stock_code=stock_code,
        model_version=_registry()["version"],
        origin_week_index=int(row["week_index"]),
        origin_week_start=pd.Timestamp(row["week_start"]).date(),
        forecast=weeks,
    )


@app.get("/inventory/{stock_code}", response_model=InventoryPolicy)
def inventory(stock_code: str) -> InventoryPolicy:
    try:
        policy = _policy()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    match = policy[policy["stock_code"] == stock_code]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"no inventory policy for {stock_code!r} -- see GET /skus")
    row = match.iloc[0]
    savings = float(row["savings_vs_naive_pct"])
    return InventoryPolicy(
        stock_code=stock_code,
        forecast_protection_units=float(row["forecast_protection_units"]),
        opt_service_level=float(row["opt_service_level"]),
        opt_safety_stock=float(row["opt_safety_stock"]),
        opt_cost=float(row["opt_cost"]),
        target_safety_stock=float(row["target_safety_stock"]),
        target_cost=float(row["target_cost"]),
        naive_safety_stock=float(row["naive_safety_stock"]),
        naive_cost=float(row["naive_cost"]),
        naive_achieved_sl=float(row["naive_achieved_sl"]),
        savings_vs_naive_pct=None if np.isnan(savings) else savings,
    )


def main() -> None:
    cfg = load_config()["api"]
    uvicorn.run("src.api:app", host=cfg["host"], port=cfg["port"])


if __name__ == "__main__":
    main()
