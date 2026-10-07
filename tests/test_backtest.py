"""Contracts for the backtest harness: pair construction, metrics, baselines.

Plus artifact-level checks on reports/fold_residuals.parquet and
reports/backtest.csv so a regression in the pipeline (or a skipped re-run)
is caught by the test suite, not by reading the report by hand.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    PAIR_FEATURES,
    REL_CLIP,
    baseline_predictions,
    build_pairs,
    eval_metrics,
    sku_scales,
)
from src.data import ROOT
from src.features import FEATURE_COLUMNS

RESID_PATH = ROOT / "reports" / "fold_residuals.parquet"
BACKTEST_PATH = ROOT / "reports" / "backtest.csv"


def make_features(skus=("A", "B"), n_weeks=12, horizon_pad=True) -> pd.DataFrame:
    rows = []
    for sku in skus:
        for wk in range(n_weeks):
            rows.append(
                {
                    "stock_code": sku,
                    "week_index": wk,
                    "units": float(10 * (wk + 1)),
                    **{col: 0.0 for col in FEATURE_COLUMNS},
                }
            )
    return pd.DataFrame(rows)


def test_pair_features_include_horizon():
    assert PAIR_FEATURES == [*FEATURE_COLUMNS, "h"]


def test_build_pairs_aligns_targets_and_stops_at_grid_end():
    features = make_features(n_weeks=12)
    pairs = build_pairs(features, horizon=3)
    assert set(pairs["h"]) == {1, 2, 3}
    assert pairs["target_week_index"].max() == 11
    sample = pairs[
        (pairs["stock_code"] == "A")
        & (pairs["week_index"] == 5)
        & (pairs["h"] == 2)
    ].iloc[0]
    assert sample["target_week_index"] == 7
    assert sample["y"] == 80.0  # units at week 7 = 10 * (7 + 1)


def test_perfect_forecast_scores_zero():
    ev = pd.DataFrame(
        {
            "stock_code": ["A", "B", "A", "B"],
            "h": [1, 1, 2, 2],
            "y": [10.0, 20.0, 12.0, 24.0],
        }
    )
    rows = eval_metrics(
        ev,
        ev["y"].to_numpy(float),
        "perfect",
        "test",
        scales=np.ones(2),
        sku_pos=pd.Series([0, 1], index=["A", "B"]),
        horizon=2,
    )
    assert {r["horizon"] for r in rows} == {1, 2, "all"}
    for r in rows:
        assert r["wape"] == 0.0
        assert r["mae"] == 0.0
        assert r["bias"] == 0.0
        assert r["mase_macro"] == 0.0


def test_baselines_compute_expected_values():
    # weeks x skus; A rises 10 -> 20, B rises 4 -> 14
    U = np.array(
        [[10, 4], [12, 6], [14, 8], [16, 10], [18, 12], [20, 14]], dtype=float
    )
    pairs = pd.DataFrame(
        {"stock_code": ["A", "A", "B", "B"], "week_index": [4, 4, 4, 4], "h": [1, 2, 1, 2]}
    )
    preds = baseline_predictions(pairs, ["A", "B"], U, season=4)
    # naive: last observed weekly value at the origin
    assert list(preds["naive"]) == [16, 16, 10, 10]
    # season 4: same week one "season" ago (target week o+h-4)
    assert list(preds["snaive"]) == [12, 14, 6, 8]
    # moving average of the four weeks before the origin
    assert list(preds["ma4"]) == [13, 13, 7, 7]


def test_sku_scales_positive_and_zero_demand_fallback():
    U = np.array(
        [[10, 0], [12, 0], [14, 0], [16, 0], [18, 0], [20, 0]], dtype=float
    )
    scales = sku_scales(U, upto=4, season=2)
    assert scales[0] > 0
    assert scales[1] == 1.0  # zero-demand SKU cannot have zero MAE scale


pytestmark_artifacts = pytest.mark.skipif(
    not RESID_PATH.exists(),
    reason="reports/fold_residuals.parquet missing -- run src.backtest first",
)


@pytestmark_artifacts
def test_fold_residuals_respect_clip_bounds():
    resid = pd.read_parquet(RESID_PATH)
    assert len(resid) > 0
    assert resid["rel_resid"].between(*REL_CLIP).all()
    assert (resid["yhat"] >= 0).all()


@pytestmark_artifacts
def test_backtest_csv_has_model_and_baselines_on_every_fold():
    if not BACKTEST_PATH.exists():
        pytest.skip("reports/backtest.csv missing -- run src.backtest first")
    results = pd.read_csv(BACKTEST_PATH)
    for split in ["fold0", "fold1", "fold2", "holdout"]:
        models = set(results.loc[results["split"] == split, "model"])
        assert {"hgb", "naive", "snaive", "ma4"} <= models
    pooled = results[results["horizon"].astype(str) == "all"]
    naive = pooled[pooled["model"] == "naive"].set_index("split")["wape"]
    for model in ("hgb", "hgb_bias_corrected"):
        sub = pooled[pooled["model"] == model].set_index("split")["wape"]
        common = sub.index.intersection(naive.index)
        assert (sub.loc[common] < naive.loc[common]).all(), (
            f"{model} must beat naive in every window it was evaluated on"
        )
