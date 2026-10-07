"""Leakage contract for src/features.py.

The strongest test is truncation invariance: features for origin weeks <= T
must be identical whether the grid ends at T or extends beyond it. Any feature
secretly reading future weeks would fail it. The remaining tests pin exact
values on hand-built grids (lags, rolling means, recency) so an accidental
off-by-one in the shift logic cannot hide behind "invariance".
"""

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, add_features

CFG = {"forecast": {"seasonality": 52}}
PRICE = 2.5


def make_grid(n_weeks: int = 80, skus: tuple[str, ...] = ("A", "B")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    weeks = pd.date_range("2010-01-04", periods=n_weeks, freq="7D")
    rows = []
    for sku in skus:
        for i, week in enumerate(weeks):
            units = float(rng.poisson(5 + i % 7))
            rows.append(
                {
                    "stock_code": sku,
                    "week_start": week,
                    "week_index": i,
                    "units": units,
                    "orders": float(rng.integers(0, 4)),
                    "revenue": units * PRICE,
                    "price": PRICE,
                }
            )
    return pd.DataFrame(rows).reset_index(drop=True)


def make_tiny(units: list[float]) -> pd.DataFrame:
    n = len(units)
    return pd.DataFrame(
        {
            "stock_code": ["A"] * n,
            "week_start": pd.date_range("2010-01-04", periods=n, freq="7D"),
            "week_index": range(n),
            "units": [float(u) for u in units],
            "orders": [1.0] * n,
            "revenue": [float(u) * PRICE for u in units],
            "price": [PRICE] * n,
        }
    )


def test_features_independent_of_future_weeks():
    grid = make_grid(80)
    full = add_features(grid, CFG)[["stock_code", "week_index", *FEATURE_COLUMNS]]
    cut = add_features(grid[grid["week_index"] <= 60], CFG)[
        ["stock_code", "week_index", *FEATURE_COLUMNS]
    ]
    merged = full.merge(cut, on=["stock_code", "week_index"], suffixes=("_full", "_cut"))
    assert len(merged) == len(cut)
    for col in FEATURE_COLUMNS:
        assert np.allclose(
            merged[f"{col}_full"], merged[f"{col}_cut"], equal_nan=True
        ), f"{col} changes when future weeks are removed"


def test_lags_and_rolling_use_past_only():
    f = add_features(make_tiny([1, 2, 3, 4, 5, 6]), CFG).set_index("week_index")
    row = f.loc[5]
    assert row["units_lag1"] == 5
    assert row["units_lag2"] == 4
    assert row["units_lag3"] == 3
    assert row["units_lag4"] == 2
    assert np.isnan(row["units_lag8"])
    assert row["roll4_mean"] == (2 + 3 + 4 + 5) / 4
    assert np.isnan(row["roll12_mean"])
    assert row["orders_lag1"] == 1
    assert row["sku_exp_mean"] == (1 + 2 + 3 + 4 + 5) / 5
    assert row["weeks_since_last_sale"] == 1


def test_first_weeks_have_no_history():
    f = add_features(make_tiny(list(range(1, 11))), CFG).set_index("week_index")
    for lag in (1, 2, 3, 4, 8):
        assert np.isnan(f.loc[lag - 1, f"units_lag{lag}"])


def test_weeks_since_last_sale_counts_gap():
    f = add_features(make_tiny([0, 0, 5, 0, 0, 0, 0, 0]), CFG).set_index("week_index")
    assert f.loc[0, "weeks_since_last_sale"] == 999
    assert f.loc[2, "weeks_since_last_sale"] == 999
    assert f.loc[3, "weeks_since_last_sale"] == 1
    assert f.loc[7, "weeks_since_last_sale"] == 5


def test_calendar_features_are_known_in_advance():
    f = add_features(make_grid(60), CFG)
    weeks = pd.DatetimeIndex(f["week_start"]).isocalendar().week.astype(float)
    expected = np.sin(2 * np.pi * weeks / 52)
    assert np.allclose(f["woy_sin1"], expected)
