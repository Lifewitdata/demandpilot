"""Contracts for the inventory layer: simulation, cost trade-off, policy sanity.

The simulation test uses a degenerate residual pool (all draws equal) so the
expected result is exact arithmetic. The cost tests pin the two structural
properties the optimizer relies on: holding cost is linear in safety stock,
and with asymmetric costs the minimum sits strictly inside the horizon of
candidate service levels (newsvendor critical ratio > 0).
"""

import numpy as np
import pandas as pd
import pytest

from src.data import ROOT
from src.inventory import policy_cost, simulate_demand

INV_CFG = {
    "holding_cost_per_unit_week": 0.02,
    "stockout_cost_per_unit": 2.0,
    "naive_weeks_of_cover": 2,
}
PROTECTION = 6

POLICY_PATH = ROOT / "reports" / "inventory_policy.csv"
CURVE_PATH = ROOT / "reports" / "service_level_curve.csv"


def test_simulate_demand_applies_bias_corrected_forecast_and_residuals():
    draws = {"h1": np.array([0.0]), "h2": np.array([0.5])}
    F = np.array([100.0, 200.0])
    rng = np.random.default_rng(0)
    D = simulate_demand(F, draws, 500, rng)
    assert D.shape == (500,)
    assert np.allclose(D, 100.0 * 1.0 + 200.0 * 1.5)


def test_holding_cost_is_linear_when_shortfall_eliminated():
    rng = np.random.default_rng(1)
    D = np.clip(rng.normal(100, 20, 5000), 0, None)
    mu = float(D.mean())
    ss = float(D.max()) - mu + 1.0  # mu + ss > max demand -> zero shortfall
    expected = INV_CFG["holding_cost_per_unit_week"] * PROTECTION * ss
    assert policy_cost(D, mu, ss, INV_CFG, PROTECTION) == pytest.approx(expected)


def test_cost_curve_minimum_is_interior():
    rng = np.random.default_rng(2)
    D = np.clip(rng.normal(100, 20, 20000), 0, None)
    mu = 100.0
    grid = np.linspace(0.0, D.max() - mu, 300)
    costs = np.array([policy_cost(D, mu, ss, INV_CFG, PROTECTION) for ss in grid])
    best = grid[int(np.argmin(costs))]
    assert 0 < best < grid[-1], "asymmetric costs put the optimum strictly inside"
    assert costs.min() < costs[0] and costs.min() < costs[-1]


policy_artifacts = pytest.mark.skipif(
    not POLICY_PATH.exists(),
    reason="reports/inventory_policy.csv missing -- run src.inventory first",
)


@policy_artifacts
def test_shipped_policy_beats_naive_on_every_sku():
    policy = pd.read_csv(POLICY_PATH)
    assert len(policy) == 100
    scored = policy[policy["naive_cost"] > 0]
    assert (scored["opt_cost"] <= scored["naive_cost"] + 1e-6).all()
    assert policy["naive_achieved_sl"].mean() < 0.90
    assert (policy["forecast_protection_units"] > 0).all()


@policy_artifacts
def test_service_level_curve_has_interior_minimum():
    if not CURVE_PATH.exists():
        pytest.skip("reports/service_level_curve.csv missing -- run src.inventory first")
    curve = pd.read_csv(CURVE_PATH)
    assert (curve["total_safety_stock"].diff().dropna() > 0).all()
    best = curve.loc[curve["total_cost"].idxmin(), "service_level"]
    assert best not in (curve["service_level"].iloc[0], curve["service_level"].iloc[-1])
