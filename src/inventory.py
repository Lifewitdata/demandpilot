"""Turn forecasts into inventory decisions: safety stock and a cost curve.

For each SKU we forecast the next `lead_time + review_period` weeks from the
last observed week (all from that single origin, using the direct multi-step
model and the validation-fold bias factors). Demand uncertainty is simulated
by Monte Carlo: for each horizon we resample the model's relative residuals
(stratified by horizon, drawn from validation folds only) so no distributional
shape is assumed. Residuals are bounded to [-100%, +300%] (src/backtest.REL_CLIP):
demand cannot fall below zero, and unbounded spikes from near-zero forecasts
would otherwise dominate the simulation. Per horizon draws are independent --
a stated limitation.

Safety stock at service level s is SS = Q_s(D) - E[D] with D the simulated
protection-period demand. Cost per protection cycle = holding cost on SS plus
stockout cost on E[max(D - E[D] - SS, 0)], with asymmetric costs mirroring the
business reality (a stockout costs ~100x a unit-week of holding). The naive
policy holds `naive_weeks_of_cover` weeks of average forecast as safety stock;
both policies are scored on the same simulated demand.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.data import ROOT, load_config


def next_window_forecasts(
    artifact: dict, features: pd.DataFrame, weeks: int
) -> tuple[list[str], np.ndarray]:
    model = artifact["model"]
    cols = [c for c in artifact["feature_columns"] if c != "h"]
    bias = artifact["bias_factors"]
    last = int(features["week_index"].max())
    origin = features[features["week_index"] == last]
    X = origin[cols].to_numpy(float)
    rows = []
    for h in range(1, weeks + 1):
        X_h = np.hstack([X, np.full((len(X), 1), float(h))])
        rows.append(np.clip(model.predict(X_h), 0, None) * bias[h])
    return origin["stock_code"].tolist(), np.vstack(rows)


def simulate_demand(F: np.ndarray, draws: dict, mc_draws: int, rng) -> np.ndarray:
    total = np.zeros(mc_draws)
    for h_idx in range(F.shape[0]):
        pool = draws[f"h{h_idx + 1}"]
        rel = rng.choice(pool, size=mc_draws, replace=True)
        total += F[h_idx] * (1 + rel)
    return total


def policy_cost(
    D: np.ndarray, mu: float, ss: float, inv_cfg: dict, protection: int
) -> float:
    holding = inv_cfg["holding_cost_per_unit_week"] * protection * max(ss, 0.0)
    shortfall = np.maximum(D - (mu + ss), 0).mean()
    return float(holding + inv_cfg["stockout_cost_per_unit"] * shortfall)


def run_inventory(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    inv_cfg = cfg["inventory"]
    registry = pd.read_json(ROOT / "models" / "registry.json", typ="series")
    version = registry["version"]
    artifact = joblib.load(ROOT / "models" / f"demand_hgb_{version}.joblib")
    draws = dict(
        np.load(ROOT / "models" / f"residual_draws_{version}.npz")
    )

    features = pd.read_parquet(
        ROOT / cfg["data"]["processed_dir"] / "features.parquet"
    )
    protection = int(inv_cfg["lead_time_weeks"] + inv_cfg["review_period_weeks"])
    skus, F = next_window_forecasts(artifact, features, protection)

    rng = np.random.default_rng(cfg["project"]["seed"])
    service_levels = list(inv_cfg["service_levels"])
    target_sl = inv_cfg["target_service_level"]
    mc_draws = int(inv_cfg["mc_draws"])

    rows = []
    curve_ss = {sl: 0.0 for sl in service_levels}
    curve_cost = {sl: 0.0 for sl in service_levels}
    for i, sku in enumerate(skus):
        D = simulate_demand(F[:, i], draws, mc_draws, rng)
        mu = float(F[:, i].sum())

        ss_cost = {}
        for sl in service_levels:
            ss = max(0.0, float(np.quantile(D, sl)) - mu)
            cost = policy_cost(D, mu, ss, inv_cfg, protection)
            ss_cost[sl] = (ss, cost)
            curve_ss[sl] += ss
            curve_cost[sl] += cost
        best_sl = min(ss_cost, key=lambda s: ss_cost[s][1])
        ss_opt, cost_opt = ss_cost[best_sl]
        ss_target, cost_target = ss_cost[target_sl]

        ss_naive = inv_cfg["naive_weeks_of_cover"] * mu / protection
        cost_naive = policy_cost(D, mu, ss_naive, inv_cfg, protection)
        achieved_naive = float((D <= mu + ss_naive).mean())

        rows.append(
            {
                "stock_code": sku,
                "forecast_protection_units": round(mu, 1),
                "opt_service_level": best_sl,
                "opt_safety_stock": round(ss_opt, 1),
                "opt_cost": round(cost_opt, 2),
                "target_safety_stock": round(ss_target, 1),
                "target_cost": round(cost_target, 2),
                "naive_safety_stock": round(ss_naive, 1),
                "naive_cost": round(cost_naive, 2),
                "naive_achieved_sl": round(achieved_naive, 3),
                "savings_vs_naive_pct": (
                    round(100 * (cost_naive - cost_opt) / cost_naive, 1)
                    if cost_naive > 0
                    else np.nan
                ),
            }
        )

    policy = pd.DataFrame(rows).sort_values("naive_cost", ascending=False)

    curve = pd.DataFrame(
        {
            "service_level": service_levels,
            "total_safety_stock": [curve_ss[sl] for sl in service_levels],
            "total_cost": [curve_cost[sl] for sl in service_levels],
        }
    )

    total_naive = float(policy["naive_cost"].sum())
    total_opt = float(policy["opt_cost"].sum())
    total_target = float(policy["target_cost"].sum())
    print(f"SKUs: {len(policy)}, protection window: {protection} weeks")
    print(
        f"pooled cost/cycle: naive {total_naive:,.0f} | "
        f"target-{target_sl} {total_target:,.0f} "
        f"({100 * (total_naive - total_target) / total_naive:.1f}% cheaper) | "
        f"per-SKU optimum {total_opt:,.0f} "
        f"({100 * (total_naive - total_opt) / total_naive:.1f}% cheaper)"
    )
    print(
        f"naive policy achieved service level (pooled mean): "
        f"{policy['naive_achieved_sl'].mean():.3f}"
    )

    policy.to_csv(ROOT / "reports" / "inventory_policy.csv", index=False)
    curve.to_csv(ROOT / "reports" / "service_level_curve.csv", index=False)
    print("\nTop SKUs by naive cost:")
    print(
        policy.head(8)[
            [
                "stock_code",
                "forecast_protection_units",
                "opt_service_level",
                "opt_safety_stock",
                "naive_safety_stock",
                "naive_achieved_sl",
                "savings_vs_naive_pct",
            ]
        ].to_string(index=False)
    )
    return policy, curve


def main() -> None:
    run_inventory(load_config())


if __name__ == "__main__":
    main()
