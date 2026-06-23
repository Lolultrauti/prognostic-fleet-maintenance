"""
Unit tests for the cost-aware metrics (src/evaluation.py).

The whole project hinges on the late-vs-early asymmetry, so these tests pin down
that asymmetry numerically rather than just smoke-testing that the functions run.
"""

import numpy as np
import pytest

from evaluation import (
    FAILURE_COST,
    SCHEDULED_COST,
    cost_sensitive_mse,
    fleet_business_cost,
    nasa_score,
    rmse,
)


def test_perfect_prediction_scores_zero_everywhere():
    y = [10, 50, 125]
    assert rmse(y, y) == 0.0
    assert nasa_score(y, y) == 0.0
    assert cost_sensitive_mse(y, y) == 0.0


def test_rmse_is_symmetric():
    # Same magnitude error either direction -> identical RMSE.
    assert rmse([50], [40]) == rmse([50], [60]) == 10.0


def test_nasa_score_punishes_late_more_than_early():
    # 20-cycle error: late (over-predict) must cost strictly more than early.
    early = nasa_score([50], [30])   # under-predict
    late = nasa_score([50], [70])    # over-predict
    assert late > early > 0
    # Exact PHM08 values: exp(20/13)-1 vs exp(20/10)-1.
    assert early == pytest.approx(np.exp(20 / 13) - 1)
    assert late == pytest.approx(np.exp(20 / 10) - 1)


def test_cost_sensitive_mse_weights_late_ten_times():
    # Same squared error; late side must be exactly late_penalty (10x) the early.
    early = cost_sensitive_mse([50], [40])   # d=-10, weight 1  -> 100
    late = cost_sensitive_mse([50], [60])    # d=+10, weight 10 -> 1000
    assert early == pytest.approx(100.0)
    assert late == pytest.approx(1000.0)
    assert late == pytest.approx(10 * early)


def test_fleet_cost_charges_failure_for_a_missed_engine():
    # One engine truly failing (true RUL 8) but predicted healthy (35 >= 30):
    # not flagged -> missed -> FAILURE_COST.
    cost = fleet_business_cost([8], [35], threshold=30)
    assert cost == FAILURE_COST


def test_fleet_cost_charges_scheduled_for_a_flagged_engine():
    # Predicted below threshold -> we schedule maintenance -> SCHEDULED_COST,
    # and because it was truly failing there is no failure charge.
    cost = fleet_business_cost([8], [10], threshold=30)
    assert cost == SCHEDULED_COST


def test_fleet_cost_is_zero_for_correct_healthy_engine():
    # Healthy and predicted healthy -> no action, no cost.
    assert fleet_business_cost([90], [88], threshold=30) == 0.0


def test_fleet_breakdown_counts_outcomes():
    yt = [10, 12, 8, 90, 45, 20]
    yp = [15, 9, 35, 88, 50, 5]
    bd = fleet_business_cost(yt, yp, return_breakdown=True)
    assert bd["n_engines"] == 6
    assert bd["n_missed_failures"] == 1          # engine #3 (true 8, pred 35)
    assert bd["n_scheduled"] == 3                # engines 1, 2, 6 flagged
    assert bd["total_cost"] == 3 * SCHEDULED_COST + 1 * FAILURE_COST
