"""
src/model.py
============
XGBoost baseline for RUL prediction, tuned on the COST metric (not RMSE).

Pipeline:
  1. Build the engineered feature matrix (Phase 4) for train + test.
  2. Randomised hyper-parameter search with **GroupKFold by engine** — every
     cross-validation fold keeps each engine entirely in train OR validation, so
     rolling/derivative features never leak a single engine across the split.
  3. The search is scored with ``cost_sensitive_mse`` (Phase 5), so the selected
     model is the one that minimises *business* cost, not symmetric error.
  4. Evaluate on the held-out test set at each engine's last observed cycle —
     the realistic "predict RUL right now" scenario — reporting RMSE, the
     PHM08 NASA score, and total fleet dollar cost.

Predictions are clipped to [0, RUL_CAP]: negative remaining life is meaningless,
and we never trained the target above the 125 cap, so predictions outside that
range are not credible.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from xgboost import XGBRegressor

from evaluation import (
    cost_mse_scorer,
    cost_sensitive_mse,
    fleet_business_cost,
    nasa_score,
    rmse,
)
from features import build_train_test_features
from targets import RUL_CAP, last_cycle_per_engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_feature_data():
    """Return (train_feat, test_feat, feature_cols) from the Phase 4 pipeline."""
    return build_train_test_features(include_lags=False, save=True)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_baseline(X, y, groups, n_iter: int = 20, n_splits: int = 3):
    """Randomised search over XGBoost hyper-parameters, scored on cost_mse.

    GroupKFold(groups=engine id) prevents the same engine appearing in both the
    train and validation side of a fold. Returns the fitted RandomizedSearchCV.
    """
    base = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",      # fast histogram algorithm
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Modest but meaningful search space (kept small for a 2-day project).
    param_dist = {
        "n_estimators": [200, 300, 500, 800],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "reg_lambda": [1.0, 3.0, 5.0],
    }

    cv = GroupKFold(n_splits=n_splits)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=cost_mse_scorer,   # minimise business-aligned cost
        cv=cv,
        n_jobs=1,                  # xgb already uses all cores; avoid oversubscription
        random_state=RANDOM_STATE,
        verbose=0,
        refit=True,
    )
    search.fit(X, y, groups=groups)
    return search


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def predict_clipped(model, X) -> np.ndarray:
    """Predict and clip to the valid RUL range [0, RUL_CAP]."""
    return np.clip(model.predict(X), 0.0, RUL_CAP)


def evaluate(model, X, y_true) -> dict:
    """Compute the full metric suite for a set of predictions."""
    y_pred = predict_clipped(model, X)
    return {
        "rmse": rmse(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "cost_sensitive_mse": cost_sensitive_mse(y_true, y_pred),
        "fleet_cost": fleet_business_cost(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Phase-6 visible output
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 76)
    print("Phase 6 — XGBoost baseline (tuned on cost_sensitive_mse)")
    print("=" * 76)

    train_feat, test_feat, feature_cols = load_feature_data()
    print(f"\nfeatures: {len(feature_cols)} | train rows: {len(train_feat)} | "
          f"test rows: {len(test_feat)}")

    X_train = train_feat[feature_cols]
    y_train = train_feat["RUL"]
    groups = train_feat["unit_number"]

    # Test set: one prediction per engine, at its last observed cycle.
    test_last = last_cycle_per_engine(test_feat)
    X_test = test_last[feature_cols]
    y_test = test_last["RUL"].to_numpy()

    # --- Reference: an untuned default XGBoost, for an honest "did tuning help?" --
    print("\n[1/2] Default XGBoost (no tuning) ...")
    default = XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    default.fit(X_train, y_train)
    default_metrics = evaluate(default, X_test, y_test)

    # --- Tuned search ---------------------------------------------------------
    print("[2/2] RandomizedSearchCV (GroupKFold by engine, scoring=cost_mse) ...")
    search = train_baseline(X_train, y_train, groups, n_iter=20, n_splits=3)
    best = search.best_estimator_
    tuned_metrics = evaluate(best, X_test, y_test)

    # --- Report ---------------------------------------------------------------
    print("\nBest hyper-parameters:")
    for k, v in search.best_params_.items():
        print(f"  {k:<18}: {v}")
    print(f"  best CV cost_mse  : {-search.best_score_:.2f}")

    print("\nHELD-OUT TEST METRICS (100 engines, last-cycle prediction):")
    print(f"  {'metric':<22}{'default':>14}{'tuned':>14}")
    for k in ["rmse", "nasa_score", "cost_sensitive_mse", "fleet_cost"]:
        print(f"  {k:<22}{default_metrics[k]:>14.2f}{tuned_metrics[k]:>14.2f}")

    # Fleet cost breakdown for the winner (lower fleet_cost).
    winner = best if tuned_metrics["fleet_cost"] <= default_metrics["fleet_cost"] else default
    bd = fleet_business_cost(y_test, predict_clipped(winner, X_test), return_breakdown=True)
    print("\nFleet business-cost breakdown (best model):")
    for k, v in bd.items():
        print(f"  {k:<22}: {v:,.0f}" if isinstance(v, float) else f"  {k:<22}: {v}")

    # --- Persist the tuned model + feature list for Phases 8/9 ----------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best, "feature_cols": feature_cols},
                MODEL_DIR / "xgb_baseline.joblib")
    print(f"\nSaved -> models/xgb_baseline.joblib")
    print("[model] Phase 6 baseline complete.")
