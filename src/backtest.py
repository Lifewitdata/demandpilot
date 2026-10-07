"""Rolling-origin backtest: direct multi-step gradient boosting vs naive baselines.

Design: origins are weekly. For each origin o and horizon h the target is the
units sold in week o+h, so the model for horizon h only ever sees information
available at o (all features in src/features.py are shifted). `h` itself is an
input feature (PAIR_FEATURES), so one model predicts every step of the horizon
with horizon-aware accuracy. Three validation folds of 8 weeks each sit
immediately before the holdout (the last 8 weeks, evaluated in src/train.py).
A fold trains only on pairs whose *target week* precedes the fold window, so no
evaluation target is ever seen during fitting.

Metrics: WAPE (volume-weighted abs error), MAE, bias (sum error / sum actual)
and macro MASE scaled by each SKU's training-window seasonal-naive MAE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data import ROOT, load_config
from src.features import FEATURE_COLUMNS, TARGET

REL_CLIP = (-1.0, 3.0)

PAIR_FEATURES = [*FEATURE_COLUMNS, "h"]

MODEL_PARAMS = dict(
    max_iter=300,
    learning_rate=0.06,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=False,
)
BASELINES = ("naive", "snaive", "ma4")


def new_model(cfg: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        random_state=cfg["project"]["seed"], **MODEL_PARAMS
    )


def pivot_units(features: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    wide = features.pivot(index="week_index", columns="stock_code", values=TARGET)
    return wide.index.to_numpy(), list(wide.columns), wide.to_numpy(dtype=float)


def build_pairs(features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    units = features[["stock_code", "week_index", TARGET]].rename(
        columns={TARGET: "y", "week_index": "target_week_index"}
    )
    frames = []
    for h in range(1, horizon + 1):
        p = features[["stock_code", "week_index", *FEATURE_COLUMNS]].copy()
        p["h"] = h
        p["target_week_index"] = p["week_index"] + h
        frames.append(
            p.merge(units, on=["stock_code", "target_week_index"], how="inner")
        )
    pairs = pd.concat(frames, ignore_index=True)
    return pairs.dropna(subset=[*FEATURE_COLUMNS, "y"]).reset_index(drop=True)


def baseline_predictions(
    pairs: pd.DataFrame, skus: list[str], U: np.ndarray, season: int = 52
) -> dict[str, np.ndarray]:
    o = pairs["week_index"].to_numpy()
    h = pairs["h"].to_numpy()
    s = pd.Series(np.arange(len(skus)), index=skus).reindex(
        pairs["stock_code"].to_numpy()
    ).to_numpy()
    n = len(pairs)

    preds: dict[str, np.ndarray] = {}

    p = np.full(n, np.nan)
    idx = o - 1
    v = idx >= 0
    p[v] = U[idx[v], s[v]]
    preds["naive"] = p

    p = np.full(n, np.nan)
    idx = o + h - season
    v = idx >= 0
    p[v] = U[idx[v], s[v]]
    preds["snaive"] = p

    p = np.full(n, np.nan)
    v = o >= 4
    C = np.vstack([np.zeros((1, U.shape[1])), np.cumsum(U, axis=0)])
    p[v] = (C[o[v], s[v]] - C[o[v] - 4, s[v]]) / 4.0
    preds["ma4"] = p

    return {k: np.clip(vals, 0, None) for k, vals in preds.items()}


def sku_scales(U: np.ndarray, upto: int, season: int = 52) -> np.ndarray:
    """Per-SKU seasonal-naive MAE over weeks strictly before `upto`."""
    if upto > season:
        diffs = np.abs(U[season:upto] - U[: upto - season])
    elif upto > 1:
        diffs = np.abs(np.diff(U[:upto], axis=0))
    else:
        return np.ones(U.shape[1])
    scales = diffs.mean(axis=0)
    return np.where(scales == 0, 1.0, scales)


def eval_metrics(
    ev: pd.DataFrame,
    yhat: np.ndarray,
    model: str,
    split: str,
    scales: np.ndarray,
    sku_pos: pd.Series,
    horizon: int,
) -> list[dict]:
    y = ev["y"].to_numpy(float)
    err = yhat - y
    s_idx = sku_pos.reindex(ev["stock_code"].to_numpy()).to_numpy()
    h_all = ev["h"].to_numpy()
    rows = []
    for hz in [*range(1, horizon + 1), "all"]:
        mask = np.ones(len(ev), bool) if hz == "all" else h_all == hz
        if mask.sum() == 0:
            continue
        yh, eh = y[mask], err[mask]
        ae = pd.DataFrame({"s": s_idx[mask], "ae": np.abs(eh)})
        mae_sku = ae.groupby("s")["ae"].mean()
        mase = float((mae_sku / scales[mae_sku.index.to_numpy()]).mean())
        denom = max(yh.sum(), 1e-9)
        rows.append(
            {
                "split": split,
                "model": model,
                "horizon": hz,
                "n_pairs": int(mask.sum()),
                "wape": float(np.abs(eh).sum() / denom),
                "mae": float(np.abs(eh).mean()),
                "bias": float(eh.sum() / denom),
                "mase_macro": mase,
            }
        )
    return rows


def run_backtest(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(
        ROOT / cfg["data"]["processed_dir"] / "features.parquet"
    )
    horizon = cfg["forecast"]["horizon"]
    folds = cfg["forecast"]["folds"]
    weeks_axis, skus, U = pivot_units(features)
    n_weeks = int(weeks_axis.max()) + 1
    holdout_start = n_weeks - horizon

    pairs = build_pairs(features, horizon)
    sku_pos = pd.Series(np.arange(len(skus)), index=skus)

    results: list[dict] = []
    residuals: list[pd.DataFrame] = []

    for f in range(folds):
        eval_start = holdout_start - (f + 1) * horizon
        eval_end = eval_start + horizon - 1
        split = f"fold{f}"
        tr = pairs[pairs["target_week_index"] < eval_start]
        ev = pairs[
            (pairs["target_week_index"] >= eval_start)
            & (pairs["target_week_index"] <= eval_end)
        ]

        scales = sku_scales(U, eval_start)
        model = new_model(cfg).fit(
            tr[PAIR_FEATURES].to_numpy(float), tr["y"].to_numpy(float)
        )
        yhat = np.clip(model.predict(ev[PAIR_FEATURES].to_numpy(float)), 0, None)

        predictions = {"hgb": yhat, **baseline_predictions(ev, skus, U)}
        for name, pred in predictions.items():
            results += eval_metrics(ev, pred, name, split, scales, sku_pos, horizon)

        resid = ev[["stock_code", "target_week_index", "h", "y"]].copy()
        resid["yhat"] = yhat
        resid["rel_resid"] = (
            (resid["y"] - resid["yhat"]) / np.clip(resid["yhat"], 1, None)
        ).clip(*REL_CLIP)
        residuals.append(resid)

        print(
            f"{split}: eval weeks {eval_start}-{eval_end}, "
            f"train pairs {len(tr):,}, eval pairs {len(ev):,}"
        )

    results_df = pd.DataFrame(results)
    residuals_df = pd.concat(residuals, ignore_index=True)

    out_csv = ROOT / "reports" / "backtest.csv"
    out_resid = ROOT / "reports" / "fold_residuals.parquet"
    results_df.to_csv(out_csv, index=False)
    residuals_df.to_parquet(out_resid, index=False)

    summary = (
        results_df[results_df["horizon"] == "all"]
        .pivot_table(index="model", columns="split", values="wape")
        .round(3)
    )
    print("\nWAPE by model x fold (all horizons pooled):")
    print(summary.to_string())
    print(f"\nSaved {out_csv} and {out_resid}")
    return results_df, residuals_df


def main() -> None:
    run_backtest(load_config())


if __name__ == "__main__":
    main()
