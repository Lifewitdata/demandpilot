"""Fit the final forecast model, evaluate the holdout, persist artifacts.

The final model trains on every pair whose target week precedes the holdout
window (the last `horizon` weeks). Bias factors (sum actual / sum predicted
per horizon) and prediction-interval quantiles are estimated on the *validation
folds* only (reports/fold_residuals.parquet), never the holdout, so holdout
metrics remain honest. Artifacts: models/demand_hgb_{version}.joblib,
models/intervals.json, models/registry.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from src.backtest import (
    MODEL_PARAMS,
    PAIR_FEATURES,
    REL_CLIP,
    baseline_predictions,
    build_pairs,
    eval_metrics,
    new_model,
    pivot_units,
    sku_scales,
    split_at_origin,
)
from src.data import ROOT, load_config

VERSION = "v1"
BIAS_CLIP = (0.8, 1.5)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def estimate_bias_factors(fold_resid: pd.DataFrame, horizon: int) -> dict[int, float]:
    g = fold_resid.groupby("h")[["y", "yhat"]].sum()
    factors = (g["y"] / g["yhat"]).clip(*BIAS_CLIP)
    return {int(h): float(factors.get(h, 1.0)) for h in range(1, horizon + 1)}


def adjusted_residuals(
    fold_resid: pd.DataFrame, bias_factors: dict[int, float]
) -> pd.DataFrame:
    adj = fold_resid.copy()
    adj["yhat_c"] = adj["yhat"] * adj["h"].map(bias_factors)
    adj["rel"] = (
        (adj["y"] - adj["yhat_c"]) / np.clip(adj["yhat_c"], 1, None)
    ).clip(*REL_CLIP)
    return adj


def intervals_from_resid(adj: pd.DataFrame) -> dict:
    per_horizon = {}
    for h, grp in adj.groupby("h"):
        qs = grp["rel"].quantile(list(QUANTILES))
        per_horizon[str(int(h))] = {
            **{f"q{int(q * 100):02d}": float(v) for q, v in qs.items()},
            "mean": float(grp["rel"].mean()),
            "std": float(grp["rel"].std()),
            "n": int(len(grp)),
        }
    return per_horizon


def train(cfg: dict) -> dict:
    features = pd.read_parquet(
        ROOT / cfg["data"]["processed_dir"] / "features.parquet"
    )
    horizon = cfg["forecast"]["horizon"]
    weeks_axis, skus, U = pivot_units(features)
    n_weeks = int(weeks_axis.max()) + 1
    holdout_start = n_weeks - horizon

    pairs = build_pairs(features, horizon)
    tr, ev = split_at_origin(pairs, holdout_start - 1, horizon)
    ev = ev.copy()

    model = new_model(cfg).fit(
        tr[PAIR_FEATURES].to_numpy(float), tr["y"].to_numpy(float)
    )
    yhat = np.clip(model.predict(ev[PAIR_FEATURES].to_numpy(float)), 0, None)

    fold_resid = pd.read_parquet(ROOT / "reports" / "fold_residuals.parquet")
    bias_factors = estimate_bias_factors(fold_resid, horizon)
    adj_resid = adjusted_residuals(fold_resid, bias_factors)
    yhat_corrected = yhat * ev["h"].map(bias_factors).to_numpy()

    sku_pos = pd.Series(np.arange(len(skus)), index=skus)
    scales = sku_scales(U, holdout_start)
    baselines = baseline_predictions(ev, skus, U)
    results: list[dict] = []
    predictions = {
        "hgb": yhat,
        "hgb_bias_corrected": yhat_corrected,
        **baselines,
    }
    for name, pred in predictions.items():
        results += eval_metrics(ev, pred, name, "holdout", scales, sku_pos, horizon)
    results_df = pd.DataFrame(results)

    week_map = features[["week_index", "week_start"]].drop_duplicates()
    holdout = ev[["stock_code", "week_index", "target_week_index", "h", "y"]].copy()
    holdout['train_target_max'] = int(tr['target_week_index'].max())
    holdout = holdout.merge(
        week_map.rename(columns={"week_index": "target_week_index"}), how="left"
    )
    holdout["yhat"] = yhat
    holdout["yhat_bias_corrected"] = yhat_corrected
    for name, pred in baselines.items():
        holdout[f"yhat_{name}"] = pred
    holdout.to_parquet(ROOT / "reports" / "holdout_predictions.parquet", index=False)

    backtest_csv = ROOT / "reports" / "backtest.csv"
    existing = pd.read_csv(backtest_csv)
    combined = pd.concat(
        [existing[existing["split"] != "holdout"], results_df], ignore_index=True
    )
    combined.to_csv(backtest_csv, index=False)

    intervals = {
        "version": VERSION,
        "source": "validation_folds",
        "bias_factors": bias_factors,
        "per_horizon": intervals_from_resid(adj_resid),
    }
    (ROOT / "models" / "intervals.json").write_text(
        json.dumps(intervals, indent=2), encoding="utf-8"
    )

    draws = {
        f"h{int(h)}": grp["rel"].to_numpy(dtype=float)
        for h, grp in adj_resid.groupby("h")
    }
    np.savez_compressed(ROOT / "models" / f"residual_draws_{VERSION}.npz", **draws)

    # Only after the holdout is scored, refit the frozen model configuration on
    # all observed targets for serving. Calibration still uses validation only.
    final_model = new_model(cfg).fit(pairs[PAIR_FEATURES].to_numpy(float), pairs['y'].to_numpy(float))
    artifact = {
        "model": final_model,
        "feature_columns": PAIR_FEATURES,
        "bias_factors": bias_factors,
        "horizon": horizon,
        "params": MODEL_PARAMS,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_pairs": int(len(pairs)),
        "trained_through_week_index": int(pairs['target_week_index'].max()),
        "holdout_start_week_index": int(holdout_start),
    }
    artifact_path = ROOT / "models" / f"demand_hgb_{VERSION}.joblib"
    joblib.dump(artifact, artifact_path)

    headline = results_df[
        (results_df["horizon"] == "all") & (results_df["model"] == "hgb_bias_corrected")
    ].iloc[0]
    registry = {
        "version": VERSION,
        "artifact": artifact_path.name,
        "created_at": artifact["trained_at"],
        "horizon": horizon,
        "n_features": len(PAIR_FEATURES),
        "n_train_pairs": int(len(pairs)),
        "evaluation_train_pairs": int(len(tr)),
        "trained_through_week_index": artifact['trained_through_week_index'],
        "evaluation_protocol": "fixed origin, eight unseen target weeks; cohort frozen before all validation",
        "holdout": {
            "wape": round(float(headline["wape"]), 4),
            "mase_macro": round(float(headline["mase_macro"]), 4),
            "bias": round(float(headline["bias"]), 4),
        },
    }
    (ROOT / "models" / "registry.json").write_text(
        json.dumps(registry, indent=2), encoding="utf-8"
    )

    print(f"train pairs {len(tr):,} | holdout pairs {len(ev):,}")
    print(f"bias factors: {{{', '.join(f'h{h}: {c:.3f}' for h, c in bias_factors.items())}}}")
    summary = (
        results_df[results_df["horizon"] == "all"]
        .set_index("model")[["wape", "mase_macro", "bias"]]
        .round(3)
    )
    print("\nHoldout (all horizons pooled):")
    print(summary.to_string())
    print(f"\nSaved {artifact_path.name}, models/intervals.json, models/registry.json")
    return registry


def main() -> None:
    train(load_config())


if __name__ == "__main__":
    main()
