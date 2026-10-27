"""Generate the forecast evaluation report and figures.

Consumes reports/backtest.csv (folds + holdout), reports/holdout_predictions.parquet
and the two inventory CSVs; writes reports/forecast_report.md and PNGs under
reports/figures/.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.data import ROOT, load_config

FIG_DIR = ROOT / "reports" / "figures"
MODEL_ORDER = ["hgb_bias_corrected", "hgb", "ma4", "naive", "snaive"]


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in df.iterrows():
        cells = [
            format(v, floatfmt) if isinstance(v, float) else str(v) for v in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_sku_wape(holdout: pd.DataFrame, pred_col: str) -> pd.Series:
    err = (holdout["y"] - holdout[pred_col]).abs()
    return err.groupby(holdout["stock_code"]).sum() / holdout["y"].groupby(
        holdout["stock_code"]
    ).sum().replace(0, np.nan)


def fig_wape_by_horizon(folds: pd.DataFrame) -> str:
    by_h = (
        folds[folds["horizon"] != "all"]
        .groupby(["model", "horizon"])["wape"]
        .mean()
        .unstack()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    for model in MODEL_ORDER:
        if model in by_h.index:
            ax.plot(by_h.columns, by_h.loc[model], marker="o", label=model)
    ax.set_xlabel("horizon (weeks ahead)")
    ax.set_ylabel("WAPE (mean of validation folds)")
    ax.set_title("Validation error by forecast horizon -- model vs baselines")
    ax.grid(alpha=0.3)
    ax.legend()
    path = FIG_DIR / "wape_by_horizon.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "figures/wape_by_horizon.png"


def fig_holdout_weekly(holdout: pd.DataFrame) -> str:
    if holdout.duplicated(['stock_code', 'target_week_index']).any():
        raise ValueError('Holdout chart requires one prediction per SKU/target week')
    weekly = holdout.groupby("target_week_index", as_index=False).agg(
        week_start=('week_start', 'first'),
        actual=("y", "sum"),
        forecast=("yhat_bias_corrected", "sum"),
        naive=("yhat_naive", "sum"),
        moving_average=('yhat_ma4', 'sum'),
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(weekly['week_start'], weekly["actual"], marker="o", label="actual")
    ax.plot(
        weekly['week_start'],
        weekly["forecast"],
        marker="s",
        label="forecast (bias-corrected)",
    )
    ax.plot(
        weekly['week_start'],
        weekly["naive"],
        marker="^",
        linestyle="--",
        label="naive",
    )
    ax.plot(weekly['week_start'], weekly['moving_average'], linestyle=':', label='moving average (4 weeks)')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.set_xlabel("holdout week")
    fig.autofmt_xdate()
    ax.set_ylabel("units (top 100 SKUs)")
    ax.set_title("Holdout weeks: total demand vs forecasts")
    ax.grid(alpha=0.3)
    ax.legend()
    path = FIG_DIR / "holdout_weekly.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "figures/holdout_weekly.png"


def fig_service_level(curve: pd.DataFrame, naive_cost: float) -> str:
    best = curve.loc[curve["total_cost"].idxmin()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        curve["service_level"],
        curve["total_cost"],
        marker="o",
        label="forecast-driven policy",
    )
    ax.axhline(naive_cost, color="tab:red", linestyle="--", label="naive policy")
    ax.axvline(best["service_level"], color="gray", linestyle=":", alpha=0.7)
    ax.annotate(
        f"min at {best['service_level']:.2f}",
        xy=(best["service_level"], best["total_cost"]),
        xytext=(6, 14),
        textcoords="offset points",
    )
    ax.set_xlabel("service level")
    ax.set_ylabel("total cost per protection cycle")
    ax.set_title("Safety-stock cost curve over service levels (100 SKUs)")
    ax.grid(alpha=0.3)
    ax.legend()
    path = FIG_DIR / "service_level_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "figures/service_level_curve.png"


def fig_sku_wape(holdout: pd.DataFrame) -> tuple[str, float]:
    w_model = per_sku_wape(holdout, "yhat_bias_corrected")
    w_naive = per_sku_wape(holdout, "yhat_naive")
    valid = w_model.notna() & w_naive.notna()
    w_model, w_naive = w_model[valid], w_naive[valid]
    if w_model.empty:
        raise ValueError('Per-SKU WAPE is undefined: all holdout demand is zero')
    win_rate = float((w_model < w_naive).mean())
    lim = max(w_model.max(), w_naive.max()) * 1.05
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(w_naive, w_model, s=14, alpha=0.6)
    ax.plot([0, lim], [0, lim], color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("naive WAPE per SKU (holdout)")
    ax.set_ylabel("model WAPE per SKU (holdout)")
    ax.set_title(f"Per-SKU accuracy, model wins on {win_rate:.0%} of SKUs")
    ax.grid(alpha=0.3)
    path = FIG_DIR / "sku_wape.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "figures/sku_wape.png", win_rate


def main() -> None:
    cfg = load_config()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    backtest = pd.read_csv(ROOT / "reports" / "backtest.csv")
    holdout = pd.read_parquet(ROOT / "reports" / "holdout_predictions.parquet")
    policy = pd.read_csv(ROOT / "reports" / "inventory_policy.csv")
    curve = pd.read_csv(ROOT / "reports" / "service_level_curve.csv")
    schema = json.loads(
        (ROOT / "reports" / "feature_schema.json").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (ROOT / "models" / "registry.json").read_text(encoding="utf-8")
    )

    folds = backtest[backtest["split"] != "holdout"]
    holdout_metrics = (
        backtest[(backtest["split"] == "holdout") & (backtest["horizon"] == "all")]
        .set_index("model")[["wape", "mase_macro", "bias"]]
        .round(3)
        .reindex([m for m in MODEL_ORDER if m in set(backtest["model"])])
        .reset_index()
    )
    fold_wape = (
        folds[folds["horizon"] == "all"]
        .pivot_table(index="model", columns="split", values="wape")
        .round(3)
        .reindex([m for m in MODEL_ORDER if m in set(folds["model"])])
        .reset_index()
    )

    f1 = fig_wape_by_horizon(folds)
    f2 = fig_holdout_weekly(holdout)
    f3 = fig_service_level(curve, float(policy["naive_cost"].sum()))
    f4, win_rate = fig_sku_wape(holdout)

    total_naive = float(policy["naive_cost"].sum())
    total_opt = float(policy["opt_cost"].sum())
    total_target = float(policy["target_cost"].sum())
    best_sl = float(curve.loc[curve["total_cost"].idxmin(), "service_level"])
    inv_summary = pd.DataFrame(
        [
            {
                "policy": "naive 2-weeks-of-cover",
                "safety stock (units)": round(float(policy["naive_safety_stock"].sum())),
                "cost per cycle": round(total_naive),
                "vs naive": "-",
            },
            {
                "policy": f"forecast-driven @ target SL {cfg['inventory']['target_service_level']}",
                "safety stock (units)": round(
                    float(policy["target_safety_stock"].sum())
                ),
                "cost per cycle": round(total_target),
                "vs naive": f"{100 * (total_naive - total_target) / total_naive:.1f}% cheaper",
            },
            {
                "policy": f"forecast-driven @ per-SKU optimum (min cost at SL {best_sl:.2f})",
                "safety stock (units)": round(float(policy["opt_safety_stock"].sum())),
                "cost per cycle": round(total_opt),
                "vs naive": f"{100 * (total_naive - total_opt) / total_naive:.1f}% cheaper",
            },
        ]
    )

    corrected = holdout_metrics.set_index('model').loc['hgb_bias_corrected']
    raw = holdout_metrics.set_index('model').loc['hgb']
    worse_folds = [str(f) for f in sorted(folds['split'].unique())
                   if fold_wape.set_index('model').loc['hgb', f] >
                   fold_wape.set_index('model').loc[list(('naive','snaive','ma4')), f].min()]
    interval_data = json.loads((ROOT / 'models/intervals.json').read_text())
    low = holdout.apply(lambda r: max(0, r['yhat_bias_corrected'] *
            (1 + interval_data['per_horizon'][str(int(r['h']))]['q05'])), axis=1)
    high = holdout.apply(lambda r: max(0, r['yhat_bias_corrected'] *
            (1 + interval_data['per_horizon'][str(int(r['h']))]['q95'])), axis=1)
    coverage = float(((holdout['y'] >= low) & (holdout['y'] <= high)).mean())
    report = f"""# DemandPilot -- forecast and inventory evaluation

Generated by `python -m src.evaluate`. Data: UCI Online Retail II,
{schema['weeks']['min']} to {schema['weeks']['max']} ({schema['weeks']['count']} weeks).
The {schema['n_skus']} SKUs and their {schema['revenue_share_of_selected']:.1%} revenue share
are selected using data strictly before {schema['selection']['data_before']}.
Model: HistGradientBoosting,
direct multi-step, {schema['horizon']}-week horizon, {len(schema['feature_columns'])} leakage-safe features.
Artifact: `{registry['artifact']}` (version {registry['version']}).

## Holdout (last {schema['horizon']} weeks, excluded from evaluation-model fitting and calibration)

Each validation fold and the holdout forecast all eight target weeks from one
fixed origin. Training targets end on or before that origin; no evaluation-window
observations update the features. The holdout has {len(holdout):,} unique SKU/week
predictions. The serving model is refitted on all observed targets only after
holdout scoring; the API rejects origins before its training cutoff.

{md_table(holdout_metrics)}

WAPE = volume-weighted absolute error; MASE uses each SKU's seasonal-naive training MAE.
The corrected forecast beats the last-week baseline on {win_rate:.0%} of SKUs.
This per-SKU comparison excludes {int(holdout.groupby('stock_code')['y'].sum().eq(0).sum())}
SKU(s) with zero holdout demand, whose WAPE is undefined; pooled metrics retain them.
Bias correction changes signed bias from {raw['bias']:.3f} to {corrected['bias']:.3f}
and WAPE from {raw['wape']:.3f} to {corrected['wape']:.3f}.
Validation-calibrated 90% prediction bands cover {coverage:.1%} of holdout outcomes;
this empirical coverage is reported rather than assuming nominal calibration.

![per-SKU WAPE]({f4})

## Rolling-origin validation folds (WAPE, all horizons pooled)

{md_table(fold_wape)}

Folds where at least one baseline has lower WAPE than raw HGB:
{', '.join(worse_folds) if worse_folds else 'none'}. Compare all models in the table;
no universal model superiority is assumed. The untouched holdout is a test, not a
model-tuning set: compare its moving-average baseline with HGB before considering
deployment. These results do not justify preferring ML in every setting.

![WAPE by horizon]({f1})

## Holdout: actual vs forecast

![holdout weekly]({f2})

## Inventory policy (lead time {cfg['inventory']['lead_time_weeks']}w +
review {cfg['inventory']['review_period_weeks']}w = {cfg['inventory']['lead_time_weeks'] + cfg['inventory']['review_period_weeks']}-week protection window)

Monte Carlo ({cfg['inventory']['mc_draws']:,} draws/SKU) over validation-fold
residuals; holding cost {cfg['inventory']['holding_cost_per_unit_week']}/unit-week vs
stockout cost {cfg['inventory']['stockout_cost_per_unit']}/unit. The naive policy
achieves a {policy['naive_achieved_sl'].mean():.0%} mean per-SKU cycle service level:

{md_table(inv_summary)}

The empirical grid minimum is at service level {best_sl:.2f}. For this simplified
linear safety-stock holding cost plus expected shortage objective, the continuous
optimum is F(Q) = 1 - holding_cost_per_unit_week * protection / stockout_cost_per_unit
= {1 - cfg['inventory']['holding_cost_per_unit_week'] * (cfg['inventory']['lead_time_weeks'] + cfg['inventory']['review_period_weeks']) / cfg['inventory']['stockout_cost_per_unit']:.3f},
when interior. This is a simulation consistency check, not observed savings.

![service level curve]({f3})

## Limitations

- Metrics refer to one historical holdout and a cohort frozen before validation.
  They do not establish future accuracy or causal savings.
- Partial first/last source weeks and nonpositive-price lines are excluded from the
  warehouse. Unobserved internal weeks are assumed zero; source completeness is not proven.
- The Monte Carlo treats horizon residuals as independent across weeks; demand
  dependence is unmodeled. Positive correlation may increase inventory risk, but
  no universal lower-bound guarantee is claimed.
- Relative residuals are clipped to [-1, 3]; extreme demand tails are truncated.
  Costs and service levels are simulated on this bounded residual model, not
  independently validated inventory outcomes.
- A real catalogue requires an authoritative product master and stock-availability
  history; high unit volume is not proof that an item is a valid retail product.
"""

    out = ROOT / "reports" / "forecast_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Figures: {f1}, {f2}, {f3}, {f4}")
    print(f"model win rate vs naive: {win_rate:.0%}")
    print(f"holdout wape (bias-corrected): {registry['holdout']['wape']}")


if __name__ == "__main__":
    main()
