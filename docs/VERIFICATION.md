# Review corrections and reproduction

## What changed

- **Evaluation timing:** one origin produces each eight-week validation/holdout
  forecast. Training targets end no later than that origin. Reports preserve
  `week_index` and `train_target_max` as audit columns.
- **Selection timing:** the cohort and minimum-history test use sales before the
  earliest validation window (25 April 2011), never full-period ABC rankings.
- **Origin convention:** origin is the last observed week. Features include that
  week's available sales; targets are weeks origin+1 through origin+8. Baselines
  use the same information cutoff.
- **Chart aggregation:** the holdout contains one actual per SKU/target week.
  Duplicate rows now raise an error instead of silently inflating weekly totals.
- **Comparisons:** reports state where baselines outperform HGB. Bias and error
  changes are calculated from saved outputs rather than hard-coded narrative.
- **API tests:** deterministic fixtures replace the previous seven model-file
  skips. These always-on contracts do not require a source download.
- **Serving safety:** the example serving model is refitted only after the holdout
  is evaluated, and historical origins before its cutoff are rejected.
- **Input edge cases:** partial boundary weeks and nonpositive prices are excluded
  from weekly demand; missing/nonfinite/fractional quantities are rejected during
  cleaning. Missing prices in zero-sale weeks no longer drop demand targets.
- **Zero-demand evaluation:** per-SKU WAPE is undefined for zero actual demand;
  one such SKU is excluded from the individual comparison, retained in pooled error.

## Reproduction

Install `requirements.txt`, then run:

```bash
python -m src.pipeline
python -m src.verify
```

The first command downloads/cleans source data, builds the SQL warehouse, selects
SKUs, generates features, backtests, evaluates the holdout, refits serving artifacts,
simulates inventory and regenerates figures/report. The second runs the test suite
and checks real regenerated artifacts through FastAPI's local TestClient, including
forecast/inventory agreement and historical-origin rejection. It writes
`reports/verification.json` with source SHA-256, dependency versions and outcomes.
This is not a verification of an externally hosted API.

On large multi-core hosts, set OMP_NUM_THREADS=2 and OPENBLAS_NUM_THREADS=2 for a
predictable local runtime. They are performance controls, not model parameters.
Tests in a clean clone run without data/model downloads. Integration verification
requires the pipeline artifacts. A running API should be restarted after rebuilds.

## Interpretation

The corrected holdout gives bias-corrected HGB WAPE 0.690, versus 0.959 for last-week
and 0.609 for moving-average-4. The latter wins this holdout. No parameter changes
were made to chase that outcome. Earlier 0.566 WAPE and 34.4% savings are superseded.
The current target-0.95 policy is 20.1% cheaper than naive safety stock under the
same simulated demand model, not an observed saving or a comparison against a
moving-average-driven inventory policy. Residual clipping and independent draws
remain explicit modeling assumptions. Production readiness is not claimed.
