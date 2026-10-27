"""Leakage-safe weekly forecasting features for the top-N SKUs.

Row identity is (stock_code, week_start) = the *origin* week. Every feature
for an origin week uses data at or before that origin only: the origin is the
last fully observed week, and calendar terms are known in
advance. Targets are formed by joining `units` at origin + horizon in the
backtest/train modules. tests/test_features.py enforces the no-future rule.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data import ROOT, load_config

TARGET = "units"

FEATURE_COLUMNS = [
    # recent demand
    "units_lag1",
    "units_lag2",
    "units_lag3",
    "units_lag4",
    "units_lag8",
    "roll4_mean",
    "roll4_std",
    "roll12_mean",
    "roll12_std",
    # order behaviour
    "orders_lag1",
    "orders_roll4_mean",
    # price signal
    "price_lag1",
    "price_exp_mean",
    # recency / level
    "weeks_since_last_sale",
    "sku_exp_mean",
    # calendar (known in advance)
    "woy_sin1",
    "woy_cos1",
    "woy_sin2",
    "woy_cos2",
    "trend",
]


def select_skus(abc: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    eligible = abc[abc["active_weeks"] >= cfg["forecast"]["min_history_weeks"]]
    return (
        eligible.sort_values("total_units", ascending=False)
        .head(cfg["forecast"]["top_n_skus"])
        .reset_index(drop=True)
    )


def training_catalogue(weekly: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Freeze the cohort before the earliest validation window, not on all sales."""
    weekly = weekly.copy()
    weekly['week_start'] = pd.to_datetime(weekly['week_start'])
    weeks = pd.date_range(weekly['week_start'].min(), weekly['week_start'].max(), freq='7D')
    first_eval = len(weeks) - (cfg['forecast']['folds'] + 1) * cfg['forecast']['horizon']
    if first_eval <= 0:
        raise ValueError('Insufficient weeks for validation and holdout')
    cutoff = weeks[first_eval]
    past = weekly[weekly['week_start'] < cutoff]
    catalogue = past.groupby('stock_code', as_index=False).agg(
        total_units=('units', 'sum'), total_revenue=('revenue', 'sum'),
        active_weeks=('week_start', 'nunique'))
    return catalogue, cutoff


def build_grid(weekly: pd.DataFrame, skus: list[str]) -> pd.DataFrame:
    weekly = weekly.copy()
    weekly['week_start'] = pd.to_datetime(weekly['week_start'])
    weeks = pd.date_range(
        weekly["week_start"].min(), weekly["week_start"].max(), freq="7D"
    )
    sub = weekly[weekly["stock_code"].isin(skus)]
    frames = {}
    for value in ("units", "orders", "revenue"):
        wide = (
            sub.pivot_table(
                index="week_start", columns="stock_code", values=value, aggfunc="sum"
            )
            .reindex(index=weeks, columns=skus)
            .fillna(0.0)
        )
        frames[value] = wide.stack().rename(value)
    grid = pd.concat(frames.values(), axis=1).reset_index()
    grid.columns = ["week_start", "stock_code", "units", "orders", "revenue"]
    grid["week_index"] = grid["week_start"].rank(method="dense").astype(int) - 1
    grid["price"] = np.where(
        grid["units"] > 0, grid["revenue"] / grid["units"].replace(0, np.nan), np.nan
    )
    return grid.sort_values(["stock_code", "week_index"]).reset_index(drop=True)


def _rolling(series: pd.Series, keys: pd.Series, window: int, how: str) -> pd.Series:
    rolled = series.groupby(keys).rolling(window, min_periods=window)
    return getattr(rolled, how)().reset_index(level=0, drop=True)


def add_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    keys = df["stock_code"]
    g = df.groupby("stock_code", group_keys=False)

    for k in (1, 2, 3, 4, 8):
        df[f"units_lag{k}"] = g["units"].shift(k - 1)

    shifted_units = df['units']
    df["roll4_mean"] = _rolling(shifted_units, keys, 4, "mean")
    df["roll4_std"] = _rolling(shifted_units, keys, 4, "std")
    df["roll12_mean"] = _rolling(shifted_units, keys, 12, "mean")
    df["roll12_std"] = _rolling(shifted_units, keys, 12, "std")

    shifted_orders = df['orders']
    df["orders_lag1"] = shifted_orders
    df["orders_roll4_mean"] = _rolling(shifted_orders, keys, 4, "mean")

    shifted_price = df['price']
    df["price_lag1"] = shifted_price
    df["price_exp_mean"] = (
        shifted_price.groupby(keys).expanding(min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )

    df["sku_exp_mean"] = (
        shifted_units.groupby(keys).expanding(min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )

    sale_week = df["week_index"].where(df["units"] > 0)
    last_sale = sale_week.groupby(keys).ffill()
    df["weeks_since_last_sale"] = (df["week_index"] - last_sale).fillna(999).astype(float)

    woy = df["week_start"].dt.isocalendar().week.astype(float)
    period = cfg["forecast"]["seasonality"]
    for k in (1, 2):
        df[f"woy_sin{k}"] = np.sin(2 * np.pi * k * woy / period)
        df[f"woy_cos{k}"] = np.cos(2 * np.pi * k * woy / period)
    df["trend"] = df["week_index"].astype(float)

    return df


def build_features(cfg: dict) -> tuple[pd.DataFrame, dict]:
    processed = ROOT / cfg["data"]["processed_dir"]
    weekly = pd.read_parquet(processed / cfg["warehouse"]["weekly_file"])
    abc, selection_cutoff = training_catalogue(weekly, cfg)
    top = select_skus(abc, cfg)
    if len(top) != cfg['forecast']['top_n_skus']:
        raise ValueError('Not enough SKUs with training-only minimum history')
    skus = top["stock_code"].tolist()
    revenue_share = top["total_revenue"].sum() / abc["total_revenue"].sum()

    grid = build_grid(weekly, skus)
    features = add_features(grid, cfg)

    schema = {
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "horizon": cfg["forecast"]["horizon"],
        "n_skus": len(skus),
        "skus": skus,
        "revenue_share_of_selected": round(float(revenue_share), 4),
        "selection": {
            "data_before": str(selection_cutoff.date()),
            "scope": "cohort and revenue share frozen using earliest training window only",
        },
        "weeks": {
            "min": str(grid["week_start"].min().date()),
            "max": str(grid["week_start"].max().date()),
            "count": int(grid["week_index"].max()) + 1,
        },
    }

    out = processed / "features.parquet"
    features.to_parquet(out, index=False)
    (ROOT / "reports" / "feature_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )

    print(f"SKUs selected: {len(skus)} (revenue coverage {revenue_share:.1%})")
    print(
        f"weeks: {schema['weeks']['count']} "
        f"({schema['weeks']['min']} -> {schema['weeks']['max']})"
    )
    print(f"{len(FEATURE_COLUMNS)} features -> {out}")
    return features, schema


def main() -> None:
    build_features(load_config())


if __name__ == "__main__":
    main()
