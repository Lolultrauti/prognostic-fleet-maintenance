"""
src/explain.py
==============
SHAP explainability for the XGBoost RUL model (Phase 6).

Two questions every reviewer / maintenance engineer asks:

  1. GLOBAL — "Across the whole fleet, which sensors/features drive the model's
     RUL predictions?" Answered with a SHAP summary (beeswarm) + mean-|SHAP| bar
     chart over the test engines.

  2. LOCAL  — "For THIS specific engine that's flagged as near-failure, WHY did
     the model say so?" Answered with a SHAP waterfall for that one prediction,
     showing how each feature pushed the RUL up or down from the baseline.

SHAP values are additive: base_value + sum(shap_values) = model prediction, so a
waterfall is an exact, per-feature decomposition of one prediction — ideal for
explaining a maintenance decision.

Functions here are reused by the Phase 9 Streamlit app (one explainer, per-engine
waterfalls on demand).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # non-interactive backend: render straight to PNG
import matplotlib.pyplot as plt
import numpy as np
import shap

from features import build_train_test_features
from targets import last_cycle_per_engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_baseline.joblib"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_model():
    """Load the tuned XGBoost model + its feature column list from Phase 6."""
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_cols"]


def load_test_eval(feature_cols):
    """Return the per-engine last-cycle test frame and its feature matrix."""
    _, test_feat, _ = build_train_test_features(include_lags=False, save=False)
    test_last = last_cycle_per_engine(test_feat)
    X = test_last[feature_cols]
    return test_last, X


def build_explainer(model):
    """A TreeExplainer is exact and fast for tree models like XGBoost."""
    return shap.TreeExplainer(model)


# --------------------------------------------------------------------------- #
# Reusable: SHAP explanation for a single engine (used by Streamlit too)
# --------------------------------------------------------------------------- #
def explain_engine(explainer, X, test_last, unit_number: int):
    """Return a single-row SHAP Explanation for the given engine's prediction."""
    row_pos = test_last.index[test_last["unit_number"] == unit_number][0]
    iloc = test_last.index.get_loc(row_pos)
    expl = explainer(X)            # Explanation over all engines
    return expl[iloc]


# --------------------------------------------------------------------------- #
# Phase-8 visible output
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 76)
    print("Phase 8 — SHAP explainability (XGBoost)")
    print("=" * 76)

    model, feature_cols = load_model()
    test_last, X = load_test_eval(feature_cols)
    print(f"\nmodel features: {len(feature_cols)} | test engines: {len(X)}")

    explainer = build_explainer(model)
    explanation = explainer(X)     # SHAP Explanation for all 100 engines

    # ---- GLOBAL 1: beeswarm summary -----------------------------------------
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title("Global SHAP summary — feature impact on predicted RUL (fleet)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_global_summary.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ---- GLOBAL 2: mean|SHAP| bar -------------------------------------------
    plt.figure()
    shap.plots.bar(explanation, max_display=15, show=False)
    plt.title("Global feature importance — mean |SHAP| (fleet)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_global_bar.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Print the top features by mean|SHAP| for the README / console.
    mean_abs = np.abs(explanation.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:12]
    print("\nTop 12 features by mean |SHAP| (global importance):")
    for rank, j in enumerate(order, 1):
        print(f"  {rank:2d}. {feature_cols[j]:<28} {mean_abs[j]:.3f}")

    # ---- LOCAL: waterfall for the most urgent (lowest true RUL) engine ------
    urgent_pos = int(np.argmin(test_last["RUL"].to_numpy()))
    urgent_engine = int(test_last.iloc[urgent_pos]["unit_number"])
    true_rul = float(test_last.iloc[urgent_pos]["RUL"])
    pred_rul = float(np.clip(model.predict(X.iloc[[urgent_pos]]), 0, 125)[0])
    print(f"\nMost urgent engine: #{urgent_engine}  (true RUL={true_rul:.0f}, "
          f"pred RUL={pred_rul:.1f})")

    plt.figure()
    shap.plots.waterfall(explanation[urgent_pos], max_display=14, show=False)
    plt.title(f"Local SHAP — why engine #{urgent_engine} is predicted near failure")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_waterfall_engine{urgent_engine}.png",
                dpi=120, bbox_inches="tight")
    plt.close()

    print(f"\nSaved figures -> {FIG_DIR}")
    for f in ["shap_global_summary.png", "shap_global_bar.png",
              f"shap_waterfall_engine{urgent_engine}.png"]:
        print(f"  - {f}")
    print("[explain] Phase 8 complete.")
