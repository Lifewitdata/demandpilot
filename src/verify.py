"""Verify regenerated artifacts and real local API responses, then save evidence."""
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
import pandas as pd
from fastapi.testclient import TestClient
from src.api import app
from src.data import ROOT


def main():
    result=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,capture_output=True,text=True,check=True)
    if ' skipped' in result.stdout:
        raise ValueError('Verification requires no skipped tests')
    client=TestClient(app)
    health=client.get('/health'); assert health.status_code==200,health.text
    skus=client.get('/skus'); assert skus.status_code==200 and skus.json()['n_skus']==100
    sku=skus.json()['skus'][0]
    forecast=client.get(f'/forecast/{sku}'); assert forecast.status_code==200,forecast.text
    body=forecast.json(); assert len(body['forecast'])==8
    policy=client.get(f'/inventory/{sku}'); assert policy.status_code==200,policy.text
    assert abs(sum(w['units'] for w in body['forecast'][:6])-policy.json()['forecast_protection_units'])<=0.1
    assert client.get(f'/forecast/{sku}?origin_week=0').status_code==422
    assert client.get('/forecast/UNKNOWN-SKU').status_code==404
    assert client.get(f'/forecast/{sku}?origin_week=10000').status_code==404
    h=pd.read_parquet(ROOT/'reports/holdout_predictions.parquet')
    assert not h.duplicated(['stock_code','target_week_index']).any()
    evidence={
      'verified_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
      'test_output':result.stdout.strip(),
      'api_verification':'real regenerated artifacts through local in-process FastAPI TestClient; not an external deployment',
      'api_statuses':{'health':200,'skus':200,'forecast':200,'inventory':200,'historical_origin':422,'unknown_sku':404,'unknown_origin':404},
      'holdout_rows':len(h),'unique_sku_week_rows':len(h[['stock_code','target_week_index']].drop_duplicates()),
      'origin_week_index':int(h.week_index.iloc[0]),'max_evaluation_training_target':int(h.train_target_max.max()),
      'source_xlsx_sha256':hashlib.sha256((ROOT/'data/raw/online_retail_II.xlsx').read_bytes()).hexdigest(),
      'versions':{n:importlib.metadata.version(n) for n in ['pandas','numpy','scikit-learn','duckdb','pyarrow','fastapi','pytest']},
    }
    (ROOT/'reports/verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(result.stdout.strip())
    print('Real-artifact API checks passed; saved reports/verification.json')


if __name__=='__main__':
    main()
