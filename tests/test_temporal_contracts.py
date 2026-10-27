"""Regressions for the temporal leakage and duplicated-actual review findings."""
import numpy as np
import pandas as pd
import pytest
from src.features import FEATURE_COLUMNS, training_catalogue, select_skus
from src.backtest import build_pairs, split_at_origin
from src.data import clean


def test_cohort_ignores_future_sales_and_future_activity():
    cfg={'forecast':{'folds':2,'horizon':2,'min_history_weeks':4,'top_n_skus':1}}
    weeks=pd.date_range('2020-01-06',periods=20,freq='7D')
    w=pd.DataFrame([{'stock_code':sku,'week_start':date,'units':units,'revenue':units*2}
                    for sku,units in [('A',10),('B',1)] for date in weeks])
    before,cutoff=training_catalogue(w,cfg)
    altered=w.copy(); mask=(altered.stock_code=='B')&(altered.week_start>=cutoff)
    altered.loc[mask,['units','revenue']]=1e12
    after,new_cutoff=training_catalogue(altered,cfg)
    pd.testing.assert_frame_equal(before,after)
    assert cutoff==new_cutoff
    assert select_skus(before,cfg).stock_code.tolist()==['A']
    assert select_skus(after,cfg).stock_code.tolist()==['A']


def test_single_origin_split_cannot_fit_targets_after_forecast_date():
    features=pd.DataFrame([{'stock_code':sku,'week_index':w,'units':float(w+1),
                           **{c:0.0 for c in FEATURE_COLUMNS}}
                          for sku in ['A','B'] for w in range(20)])
    pairs=build_pairs(features,3); train,evaluation=split_at_origin(pairs,10,3)
    assert train.target_week_index.max()<=10
    assert set(evaluation.week_index)=={10}
    assert set(evaluation.target_week_index)=={11,12,13}
    assert len(evaluation)==6
    assert not evaluation.duplicated(['stock_code','target_week_index']).any()


def test_missing_prices_keep_zero_demand_examples():
    features=pd.DataFrame([{'stock_code':'A','week_index':w,'units':0.0,
                           **{c:0.0 for c in FEATURE_COLUMNS}}
                          for w in range(20)])
    features['price_lag1']=np.nan;features['price_exp_mean']=np.nan
    pairs=build_pairs(features,3)
    _,evaluation=split_at_origin(pairs,10,3)
    assert len(evaluation)==3 and evaluation.y.eq(0).all()


def test_chart_rejects_overlapping_horizons():
    from src.evaluate import fig_holdout_weekly
    h=pd.DataFrame({'stock_code':['A','A'],'target_week_index':[10,10],
                    'y':[4.,4.],'yhat_bias_corrected':[3.,5.],'yhat_naive':[3.,3.]})
    with pytest.raises(ValueError,match='one prediction'):
        fig_holdout_weekly(h)


def test_clean_rejects_missing_nonfinite_and_fractional_measures():
    base={'Invoice':' 123 ','StockCode':' A ','Description':'test','InvoiceDate':'2011-01-03',
          'Quantity':'1','Price':'2.0','Customer ID':None,'Country':'UK'}
    rows=[base,{**base,'Quantity':None},{**base,'Price':'inf'},
          {**base,'Quantity':'1.5'},{**base,'StockCode':' '}]
    result,stats=clean(pd.DataFrame(rows))
    assert len(result)==1 and result.stock_code.iloc[0]=='A'
    assert stats['rows_dropped_missing_ids']==1
    assert stats['rows_dropped_unparseable']==3


def test_warehouse_excludes_partial_boundary_weeks_and_nonpositive_prices():
    import duckdb
    from src.data import ROOT
    dates=['2011-01-05','2011-01-11','2011-01-12','2011-01-18','2011-01-26']
    transactions=pd.DataFrame({'stock_code':['A']*5,'invoice_date':pd.to_datetime(dates),
        'quantity':[1,2,3,4,5],'unit_price':[2.,2.,0.,2.,2.],
        'invoice_no':list('abcde'),'is_cancellation':[False]*5,'line_amount':[2.,4.,0.,8.,10.]})
    con=duckdb.connect()
    try:
        con.register('transactions',transactions)
        con.execute((ROOT/'sql/01_weekly_demand.sql').read_text())
        result=con.execute('SELECT * FROM weekly_demand ORDER BY week_start').df()
        assert list(result.units)==[2,4]
        assert list(result.revenue)==[4.,8.]
    finally:
        con.close()


def test_zero_demand_sku_wape_is_undefined_not_infinity():
    from src.evaluate import per_sku_wape
    h=pd.DataFrame({'stock_code':['A','B'],'y':[0.,10.],'prediction':[2.,8.]})
    w=per_sku_wape(h,'prediction')
    assert np.isnan(w['A']) and w['B']==pytest.approx(.2)
