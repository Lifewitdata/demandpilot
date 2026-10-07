# DemandPilot — SKU Demand Forecasting & Inventory Planning

Forecast weekly demand for the top 100 selling SKUs and turn those forecasts into
safety-stock decisions with an explicit cost model — built as a production-style,
reproducible ML pipeline.

**Business framing:** stockouts lose revenue and holding costs burn working capital.
This project forecasts 8 weeks ahead per SKU, quantifies uncertainty with Monte Carlo
simulation over model residuals, and chooses safety stock that minimizes a holding-vs-
stockout cost objective — beating a rule-of-thumb policy by **~34%** on simulated cost.

**Dataset:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) —
real wholesale transactions (Dec 2009–Dec 2011), 1,067,371 cleaned lines, 4,813 SKUs.

## Architecture

```
data.py → warehouse.py → features.py → backtest.py → train.py → inventory.py → evaluate.py → api.py
   │           │              │              │            │            │             │          │
 download   DuckDB SQL     20 leakage-   rolling-      final fit    Monte Carlo   report +   FastAPI
 + clean    weekly agg     safe weekly   origin 3      + bias(per-  safety stock  figures    /health
            + ABC Pareto   features      folds + 8-wk  horizon) +   + cost curve             /skus
            (window fns)                 holdout       intervals                             /forecast
                                                                                            /inventory
```

## Build roadmap

- [x] 1. Repo scaffold, config, dependencies
- [x] 2. Data acquisition + cleaning module
- [x] 3. DuckDB warehouse: weekly demand + ABC Pareto (window functions)
- [x] 4. Leakage-safe forecasting feature engineering
- [x] 5. Rolling-origin backtest harness (3 folds + holdout, 3 baselines)
- [x] 6. Final model training with bias correction + prediction intervals
- [x] 7. Inventory decision layer (Monte Carlo safety stock + cost curve)
- [x] 8. Evaluation report + figures
- [x] 9. FastAPI forecast service
- [x] 10. Pytest suite (24 tests) + README

## Data preparation

`python -m src.warehouse` — from `data/processed/weekly_demand.parquet` + `abc_classes.parquet`:

| Fact | Value |
|---|---|
| Cleaned transaction lines | 1,067,371 |
| SKUs in catalogue | 4,813 |
| Active weeks | 104 (2009-11-30 → 2011-12-05) |
| Regular weekly grid | 106 weeks (two all-zero post-Christmas shutdown weeks kept as true zeros) |
| ABC classes (Pareto by revenue) | A 967 / B 1,188 / C 2,658 SKUs |
| Modelling scope | top 100 SKUs by units = **20.1% of revenue**, each with ≥52 weeks history |

**Why DuckDB SQL:** weekly aggregation and the Pareto ABC classification
(`sql/02_abc_analysis.sql`: `ROW_NUMBER()` + running `SUM() OVER (ORDER BY ...)`)
are expressible as plain SQL over parquet — the warehouse is the reproducible
source of truth, pandas only reads its outputs.

Non-product codes (`M` manual adjustments, `DOT` postage, `POST`, `AMAZONFEE`, … —
40 codes, 3.4% of revenue) fall out naturally of the volume-based top-100 selection
(the largest of them ranks #239 by units); a production catalogue would map them
explicitly (see Limitations).

## Forecasting approach

- **Target / identity:** every training row is `(SKU, origin week, h)` — the units sold
  `h` weeks after the origin. Features describe the world as of the origin week only;
  `h` (`1..8`) is itself an input feature, so **one model forecasts every step** with
  horizon-aware accuracy and no recursive error accumulation (direct multi-step).
- **20 leakage-safe features** (`src/features.py`): lags 1/2/3/4/8, rolling 4- and 12-week
  mean/std (all shifted ≥1 week), order counts, price level and trend, weeks-since-last-sale,
  per-SKU expanding means, and Fourier terms for 52-week seasonality. `tests/test_features.py`
  proves leakage-safety by truncation: removing all future weeks leaves origin-week features
  bit-identical.
- **Validation:** 3 rolling-origin folds of 8 weeks plus a final 8-week holdout
  (Oct–Dec 2011). A fold trains only on targets strictly before its evaluation window;
  the holdout is touched exactly once, after all choices are frozen.
- **Model:** scikit-learn `HistGradientBoostingRegressor` (`max_iter=300`, `lr=0.06`,
  `min_samples_leaf=20`, `l2=1.0`), predictions clipped at zero.
- **Bias correction + intervals:** per-horizon factors `c_h = Σy / Σŷ` and residual
  quantiles are estimated on the validation folds (never the holdout), clipped to sane
  bounds, and persisted in `models/intervals.json` — the API applies both.

## Results

Holdout = last 8 weeks, never used for fitting or calibration (`reports/forecast_report.md`):

| Model | WAPE | MASE | Bias |
|---|---|---|---|
| **HGB + bias correction** | **0.566** | **0.865** | **-0.033** |
| HGB (raw) | 0.553 | 0.848 | -0.095 |
| Moving-average-4 | 0.591 | 0.911 | -0.135 |
| Naive (last week) | 0.686 | 1.050 | -0.098 |
| Seasonal naive (t−52) | 0.972 | 1.339 | 0.484 |

Rolling-origin folds (WAPE, pooled horizons): HGB 0.625 / 0.570 / 0.626 vs naive
0.729 / 0.752 / 0.775 — the model beats every baseline in every window, and beats
naive on **84% of SKUs** individually. WAPE ~0.55 is honest for intermittent SKU-level
demand; the product decision (safety stock) is designed around exactly that.

![WAPE by horizon](reports/figures/wape_by_horizon.png)
![Holdout weekly](reports/figures/holdout_weekly.png)

**Inventory decision layer** (`src/inventory.py`, 6-week protection window =
4-week lead time + 2-week review): 20,000 Monte Carlo draws per SKU resample the
model's per-horizon residuals (no distributional assumption), and safety stock is
chosen per service level:

| Policy | Safety stock (units) | Cost / cycle | vs naive |
|---|---|---|---|
| Naive: 2 weeks of cover | 34,577 | 18,290 | — (achieves only SL 0.76) |
| Forecast-driven @ target SL 0.95 | 81,254 | 11,989 | **34.4% cheaper** |
| Forecast-driven @ per-SKU cost optimum (SL 0.93) | 72,996 | 11,982 | **34.5% cheaper** |

The empirical cost minimum at SL ≈ 0.93 matches the newsvendor critical ratio implied
by the cost parameters (2.0 / (2.0 + 0.02×6) = 0.94) — a sanity check that the
simulation and optimizer are coherent.

![Service level curve](reports/figures/service_level_curve.png)

## Serving

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000

curl http://127.0.0.1:8000/health          # model version, horizon, holdout metrics
curl http://127.0.0.1:8000/skus            # the 100 forecastable SKUs
curl http://127.0.0.1:8000/forecast/85123A # 8 weekly forecasts with p05/p95 bands
curl http://127.0.0.1:8000/inventory/85123A# safety stock + cost vs naive policy
```

```
$ curl "http://127.0.0.1:8000/forecast/85123A?origin_week=105"
{"stock_code": "85123A", "model_version": "v1", "origin_week_start": "2011-12-05",
 "forecast": [{"horizon": 1, "week_start": "2011-12-12", "units": 890.82,
               "p05": 95.56, "p95": 2886.43}, ... 8 weeks total]}
```

The service loads the versioned artifact once (lru cache), applies the same bias
factors and quantiles the pipeline calibrated, and returns 404 for unknown SKUs or
origin weeks, 503 when artifacts are missing. The inventory endpoint is served from
the same policy file the report quotes — `tests/test_api.py` asserts the API's h=1..6
forecast sum equals the inventory layer's protection-window demand to within rounding.

## Project structure

```
config.yaml             # paths, horizon, folds, inventory costs, API host/port
sql/
  01_weekly_demand.sql  # SKU x week aggregation (units, orders, revenue)
  02_abc_analysis.sql   # Pareto ABC classification via window functions
src/
  data.py               # download (UCI zip) + clean -> parquet
  warehouse.py          # DuckDB executes SQL -> weekly_demand + abc parquet
  features.py           # 20 leakage-safe origin-week features
  backtest.py           # rolling-origin folds, 3 baselines, WAPE/MAE/bias/MASE
  train.py              # final model, bias factors, intervals, versioned artifact
  inventory.py          # Monte Carlo safety stock + service-level cost curve
  evaluate.py           # forecast_report.md + 4 figures
  api.py                # FastAPI /health /skus /forecast /inventory
tests/                  # 24 tests: leakage invariants, metric math, baselines,
                        #   policy sanity, API contract + cross-layer consistency
reports/                # generated: backtest.csv, forecast_report.md, figures, ...
models/                 # generated: demand_hgb_v1.joblib + registry (gitignored)
```

## Evidence map

How each resume claim is backed by an artifact in this repo (interview-defensible):

| Claim | Where to point |
|---|---|
| SQL analytics on a real warehouse | `sql/01_weekly_demand.sql`, `sql/02_abc_analysis.sql` — DuckDB, window functions for Pareto ABC |
| Feature engineering with leakage discipline | `src/features.py`; `tests/test_features.py` truncation-invariance proof |
| Rigorous evaluation, not a single lucky split | `src/backtest.py` — 3 rolling-origin folds + untouched holdout, naive/snaive/MA4 baselines, WAPE/MAE/bias/MASE (macro) |
| pandas/NumPy/sklearn end-to-end pipeline | `src/` modules driven by `config.yaml`, seed 42, artifacts regenerable from a clean clone |
| Productionizing a model | `src/api.py` FastAPI service; versioned `models/registry.json`; bias factors + intervals persisted with the model |
| Business translation (forecast → decision) | `src/inventory.py` — asymmetric cost model (stockout 2.0 vs holding 0.02×6), newsvendor agreement at SL 0.94, naive-policy comparison |
| Communicating results with visuals | `reports/forecast_report.md` + `reports/figures/` (WAPE by horizon, holdout weekly, service-level curve, per-SKU WAPE) |

## Quickstart

```bash
pip install -r requirements.txt

python -m src.data        # download + clean -> data/interim/transactions.parquet
python -m src.warehouse   # DuckDB SQL -> weekly_demand + ABC parquet
python -m src.features    # features -> data/processed/features.parquet
python -m src.backtest    # folds -> reports/backtest.csv + fold residuals
python -m src.train       # final model -> models/demand_hgb_v1.joblib + registry
python -m src.inventory   # safety stock + cost curve -> reports/*.csv
python -m src.evaluate    # report + figures -> reports/

pytest -q                 # 24 tests
```

Reproducible from a clean clone: `python -m src.data --force-download` re-fetches the
UCI archive; every downstream step reads only `data/` and `models/` artifacts.

## Limitations

- Weekly SKU demand is intermittent; WAPE ≈ 0.55 is high in absolute terms and is why
  reorder decisions, not point forecasts, are the product.
- The holdout window (Oct–Dec 2011) is the holiday ramp; all models under-forecast it,
  and the fold-calibrated bias correction halves the model's bias at a 0.004 WAPE cost.
  Production would recalibrate bias factors on a rolling window.
- Monte Carlo draws residuals independently across weeks; demand autocorrelation would
  fatten the tails, so computed safety stock is a lower bound under that correlation.
- Non-product codes (3.4% of revenue) are naturally excluded from the top-100 selection
  but a production catalogue needs explicit mapping.
- The API serves forecasts from the latest *stored* origin week per SKU; a live
  deployment would refresh the feature store from the warehouse on request.
