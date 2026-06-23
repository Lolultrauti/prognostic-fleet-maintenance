"""
src/evaluation.py
=================
Evaluation metrics for RUL prediction — defined BEFORE any model is trained.

Why metrics first? In predictive maintenance the *business* asymmetry is the
whole point: the two ways to be wrong are not equally bad.

    * Predict RUL too LOW  (pred < true) -> "early" / pessimistic.
      Consequence: you service an engine sooner than strictly necessary. You
      lose a little remaining useful life. Cost: a wasted maintenance slot.

    * Predict RUL too HIGH (pred > true) -> "late" / optimistic.
      Consequence: you believe the engine is healthier than it is, skip
      maintenance, and it can FAIL IN OPERATION. Cost: catastrophic — unplanned
      downtime, secondary damage, safety.

So late (over-)predictions must be punished far harder than early ones. This
module encodes that asymmetry three ways, then exposes sklearn scorers so the
Phase 6 model is *tuned* to minimise business cost, not plain RMSE.

Sign convention used everywhere:
    error  d = y_pred - y_true
    d < 0  -> early / under-predict (safe-ish)
    d > 0  -> late  / over-predict  (dangerous)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import make_scorer


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to flat float arrays so the metrics accept lists/Series."""
    return (
        np.asarray(y_true, dtype=float).ravel(),
        np.asarray(y_pred, dtype=float).ravel(),
    )


# --------------------------------------------------------------------------- #
# 1. RMSE — the symmetric baseline metric (for comparison / sanity)
# --------------------------------------------------------------------------- #
def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error. Symmetric: treats early and late errors alike."""
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


# --------------------------------------------------------------------------- #
# 2. NASA / PHM08 score — the standard prognostics competition metric
# --------------------------------------------------------------------------- #
# Reference: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation
# Modeling for Aircraft Engine Run-to-Failure Simulation," PHM08, 2008.
#
# For each engine i, with d_i = pred_i - true_i:
#
#       s_i = exp(-d_i / 13) - 1     if d_i < 0   (early prediction)
#       s_i = exp( d_i / 10) - 1     if d_i >= 0  (late  prediction)
#
#       NASA_score = sum_i s_i
#
# The two different time-constants are what make it ASYMMETRIC: the late branch
# divides by 10 and the early branch by 13, so e^(d/10) grows faster than
# e^(|d|/13). A late error of 20 cycles costs far more than an early error of 20.
# Lower is better; a perfect prediction scores 0.
# --------------------------------------------------------------------------- #
def nasa_score(y_true, y_pred) -> float:
    """Asymmetric PHM08 prognostics score (lower is better, 0 = perfect)."""
    yt, yp = _as_arrays(y_true, y_pred)
    d = yp - yt
    # Branch per-sample on the sign of the error.
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s))


# --------------------------------------------------------------------------- #
# 3. Cost-sensitive MSE — asymmetric squared error, late penalised ~10x
# --------------------------------------------------------------------------- #
def cost_sensitive_mse(y_true, y_pred, late_penalty: float = 10.0) -> float:
    """Mean squared error with late (over-)predictions weighted `late_penalty`x.

    error d = pred - true.
        d > 0 (late / over-predict, dangerous) -> weight = late_penalty (10)
        d <= 0 (early / under-predict, safe)    -> weight = 1

    Unlike the NASA score this stays in squared-error units, so it slots in as a
    drop-in, differentiable-style replacement for MSE during model selection.
    Lower is better.
    """
    yt, yp = _as_arrays(y_true, y_pred)
    d = yp - yt
    weights = np.where(d > 0, late_penalty, 1.0)
    return float(np.mean(weights * d**2))


# --------------------------------------------------------------------------- #
# 4. Fleet-level business cost — dollars, the metric a manager cares about
# --------------------------------------------------------------------------- #
# Operating policy modelled:
#   * If predicted RUL < `threshold` (default 30 cycles), we proactively schedule
#     maintenance -> a small, planned SCHEDULED_COST.
#   * If an engine is actually near failure (true RUL < threshold) but we did
#     NOT flag it, it runs to an unplanned, in-service FAILURE -> a large
#     FAILURE_COST.
#
#   Four outcomes per engine (evaluated at its last observed cycle):
#     flagged & truly-failing      -> correct preventive maintenance  : SCHEDULED
#     flagged & not failing        -> false alarm, early maintenance   : SCHEDULED
#     not flagged & truly-failing  -> MISSED failure (catastrophe)     : FAILURE
#     not flagged & not failing    -> correct no-op                    : 0
# --------------------------------------------------------------------------- #
SCHEDULED_COST = 5_000     # planned maintenance event ($)
FAILURE_COST = 100_000     # unplanned in-service failure ($) — ~20x worse


def fleet_business_cost(
    y_true,
    y_pred,
    threshold: int = 30,
    scheduled_cost: float = SCHEDULED_COST,
    failure_cost: float = FAILURE_COST,
    return_breakdown: bool = False,
):
    """Total maintenance + failure cost over a fleet (one prediction per engine).

    Pass the LAST-cycle prediction for each engine (see
    targets.last_cycle_per_engine). Lower is better. Set return_breakdown=True
    to get a dict of counts + per-bucket costs for the README business case.
    """
    yt, yp = _as_arrays(y_true, y_pred)

    flagged = yp < threshold          # we scheduled maintenance
    failing = yt < threshold          # engine truly near failure

    scheduled = flagged                       # any flag -> a scheduled event
    missed = (~flagged) & failing             # not flagged but actually failing

    n_scheduled = int(scheduled.sum())
    n_missed = int(missed.sum())
    total = n_scheduled * scheduled_cost + n_missed * failure_cost

    if not return_breakdown:
        return float(total)

    correct_preventive = int((flagged & failing).sum())
    false_alarm = int((flagged & ~failing).sum())
    correct_noop = int((~flagged & ~failing).sum())
    return {
        "total_cost": float(total),
        "n_engines": int(yt.size),
        "n_scheduled": n_scheduled,
        "n_missed_failures": n_missed,
        "correct_preventive": correct_preventive,
        "false_alarms": false_alarm,
        "correct_noops": correct_noop,
        "scheduled_cost_total": n_scheduled * scheduled_cost,
        "failure_cost_total": n_missed * failure_cost,
    }


# --------------------------------------------------------------------------- #
# 5. sklearn scorers — plug straight into GridSearchCV / cross_val_score
# --------------------------------------------------------------------------- #
# All three are COSTS (lower better), so greater_is_better=False -> sklearn
# negates them internally and still "maximises", picking the lowest-cost model.
nasa_scorer = make_scorer(nasa_score, greater_is_better=False)
cost_mse_scorer = make_scorer(cost_sensitive_mse, greater_is_better=False)
fleet_cost_scorer = make_scorer(fleet_business_cost, greater_is_better=False)


# --------------------------------------------------------------------------- #
# Phase-5 visible output — demonstrate the asymmetry on synthetic predictions
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 76)
    print("Phase 5 — cost-aware evaluation metrics (asymmetry demo)")
    print("=" * 76)

    # Same magnitude of error, opposite direction, on a true RUL of 50.
    true = np.array([50.0])
    early = np.array([30.0])   # under-predict by 20 (safe)
    late = np.array([70.0])    # over-predict  by 20 (dangerous)

    print("\nTrue RUL = 50. Compare a 20-cycle EARLY vs a 20-cycle LATE error:")
    print(f"  {'metric':<22}{'early (pred 30)':>18}{'late (pred 70)':>18}")
    for name, fn in [("RMSE", rmse), ("NASA/PHM08 score", nasa_score),
                     ("cost_sensitive_mse", cost_sensitive_mse)]:
        e = fn(true, early)
        l = fn(true, late)
        print(f"  {name:<22}{e:>18.2f}{l:>18.2f}")
    print("  -> RMSE is identical both ways; NASA & cost_mse punish LATE far more.")

    # Fleet business-cost example on 6 engines (last-cycle predictions).
    print("\nFleet business cost on 6 example engines:")
    yt = np.array([10, 12,  8, 90, 45, 20])   # true RUL at last cycle
    yp = np.array([15,  9, 35, 88, 50,  5])   # model predictions
    bd = fleet_business_cost(yt, yp, return_breakdown=True)
    for k, v in bd.items():
        print(f"  {k:<22}: {v:,.0f}" if isinstance(v, float) else f"  {k:<22}: {v}")
    # Engine index 2: true=8 (failing) but pred=35 (>=30, not flagged) -> MISSED.
    print("  -> engine #3 (true 8, pred 35) is a MISSED failure -> $100k dominates.")
