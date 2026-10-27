"""Rebuild data, fixed-origin evaluation, serving artifacts and reports in order."""
from src.data import load_config, build_interim
from src.warehouse import build_warehouse
from src.features import build_features
from src.backtest import run_backtest
from src.train import train
from src.inventory import run_inventory
from src.evaluate import main as evaluate


def main():
    cfg=load_config()
    build_interim(cfg)
    build_warehouse(cfg)
    build_features(cfg)
    run_backtest(cfg)
    train(cfg)
    run_inventory(cfg)
    evaluate()


if __name__=='__main__':
    main()
