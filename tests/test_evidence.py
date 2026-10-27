"""Committed report evidence must agree with the evaluation protocol and metrics."""
import numpy as np
import pandas as pd
from src.data import ROOT


def test_holdout_is_one_frozen_origin_and_one_actual_per_sku_week():
    h=pd.read_parquet(ROOT/'reports/holdout_predictions.parquet')
    assert len(h)==800
    assert h.week_index.nunique()==1
    assert not h.duplicated(['stock_code','target_week_index']).any()
    assert (h.train_target_max<=h.week_index).all()
    assert (h.target_week_index>h.week_index).all()
    assert h.groupby('stock_code').h.nunique().eq(8).all()


def test_validation_folds_obey_origins():
    f=pd.read_parquet(ROOT/'reports/fold_residuals.parquet')
    assert len(f)==2400
    assert f.groupby('split').week_index.nunique().eq(1).all()
    assert not f.duplicated(['split','stock_code','target_week_index']).any()
    assert (f.train_target_max<=f.week_index).all()
    assert (f.target_week_index>f.week_index).all()


def test_saved_holdout_metrics_reconcile_with_predictions():
    h=pd.read_parquet(ROOT/'reports/holdout_predictions.parquet')
    b=pd.read_csv(ROOT/'reports/backtest.csv')
    pooled=b[(b.split=='holdout')&(b.horizon=='all')].set_index('model')
    for model,col in [('hgb','yhat'),('hgb_bias_corrected','yhat_bias_corrected'),
                      ('naive','yhat_naive'),('snaive','yhat_snaive'),('ma4','yhat_ma4')]:
        assert np.isclose((h.y-h[col]).abs().sum()/h.y.sum(),pooled.loc[model,'wape'])
