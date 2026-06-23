"""
src/features.py
===============
Feature engineering for NASA C-MAPSS FD001.

The raw data gives us one row per (engine, cycle) with 21 sensor readings. A
single reading at one instant says little about *degradation*, which is a
trajectory: how a sensor is trending, how fast it is moving, and how that
compares to where the engine sits in its life. This module turns each raw row
into a much richer feature vector that captures that temporal context, so a
plain tabular model (XGBoost, Phase 6) can "see" trends without being a
sequence model.

THE ONE HARD RULE
-----------------
Every temporal operation (rolling window, derivative, lag) is computed
**per engine** via ``groupby("unit_number")``. If we rolled over the flat frame,
the window at the start of engine 5 would average in the *failure* rows at the
end of engine 4 — leaking one engine's end-of-life into another's healthy start
and corrupting the label/feature relationship. Grouping by engine keeps every
window strictly inside one engine's own history.

Feature families
----------------
1. Rolling stats (mean/std/min/max) over windows 10, 20, 50  -> smoothed level,
   volatility, and recent extremes of each sensor.
2. First & second derivatives  -> rate of change and acceleration of degradation.
3. normalized_life_position    -> how far through its life the engine is.
4. Lag(t-1) features           -> previous-cycle value. Built but LSTM-oriented;
   EXCLUDED from the default XGBoost feature set (see add_lag_features).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the canonical column definitions + loaders from earlier phases.
from data import SENSOR_COLS, SETTING_COLS, download_data, load_rul, load_test, load_train
from targets import add_test_rul, prepare_train_targets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Columns that are never model inputs (ids and the prediction target).
NON_FEATURE_COLS = {"unit_number", "time_cycles", "RUL"}


# --------------------------------------------------------------------------- #
# 1. Which sensors actually carry signal
# --------------------------------------------------------------------------- #
def informative_sensor_cols(df: pd.DataFrame, threshold: float = 1e-6) -> list[str]:
    """Return the sensors whose values actually vary (std > threshold).

    In FD001 seven sensors are constant; a constant column has zero predictive
    value and only adds noise/compute. We detect them by standard deviation
    rather than hardcoding the list, so the same rule transfers to other
    subsets (FD002–FD004). This is the single source of truth for "useful
    sensors" used everywhere downstream.
    """
    stds = df[SENSOR_COLS].std()
    return stds[stds > threshold].index.tolist()


def constant_raw_cols(df: pd.DataFrame, threshold: float = 1e-6) -> list[str]:
    """Return the raw sensor + operational-setting columns that are constant.

    These zero-variance columns (e.g. op_setting_3, sensor_1/5/10/16/18/19 in
    FD001) are excluded from the model feature set: they cannot help a model and
    only add clutter. We never engineer features from them either.
    """
    candidate = SETTING_COLS + SENSOR_COLS
    stds = df[candidate].std()
    return stds[stds <= threshold].index.tolist()


# --------------------------------------------------------------------------- #
# 2. Rolling-window statistics
# --------------------------------------------------------------------------- #
def add_rolling_features(
    df: pd.DataFrame,
    cols: list[str],
    windows: tuple[int, ...] = (10, 20, 50),
    stats: tuple[str, ...] = ("mean", "std", "min", "max"),
) -> pd.DataFrame:
    """Add per-engine rolling mean/std/min/max for each sensor and window size.

    For each engine we take a trailing window of the last ``w`` cycles and
    summarise it. ``min_periods=1`` means early cycles (fewer than ``w`` rows
    available) still get a value computed from whatever history exists, instead
    of NaN. The std of a single-row window is undefined (NaN) — we fill those
    with 0 (no variability observed yet).

    New columns are named ``{sensor}_roll{w}_{stat}``.
    """
    out = df.copy()
    grouped = out.groupby("unit_number")[cols]

    new_frames = []
    for w in windows:
        # One rolling object per window, reused across all requested stats.
        roller = grouped.rolling(window=w, min_periods=1)
        for stat in stats:
            rolled = getattr(roller, stat)()
            # groupby().rolling() returns a frame indexed by (unit_number, orig_index);
            # drop the engine level so it realigns to the original row index.
            rolled = rolled.reset_index(level=0, drop=True)
            rolled.columns = [f"{c}_roll{w}_{stat}" for c in cols]
            new_frames.append(rolled)

    feat = pd.concat(new_frames, axis=1)
    # std of a 1-row window is NaN -> 0 (no spread seen yet).
    feat = feat.fillna(0.0)
    return pd.concat([out, feat], axis=1)


# --------------------------------------------------------------------------- #
# 3. Derivatives (rate of change & acceleration)
# --------------------------------------------------------------------------- #
def add_derivative_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add per-engine 1st and 2nd derivatives for each sensor.

    * 1st derivative (``_d1``) = cycle-over-cycle change = how fast the sensor is
      moving. A healthy engine sits flat (~0); degradation shows up as a drift.
    * 2nd derivative (``_d2``) = change of the change = whether degradation is
      *accelerating* (the classic sign of approaching failure).

    The first cycle of each engine has no prior row (and the second has no prior
    d1), so those diffs are NaN — we fill with 0 ("no change observed yet").
    """
    out = df.copy()
    grouped = out.groupby("unit_number")[cols]

    d1 = grouped.diff()                       # first difference
    d2 = grouped.diff().diff()                # difference of the difference
    d1.columns = [f"{c}_d1" for c in cols]
    d2.columns = [f"{c}_d2" for c in cols]

    deriv = pd.concat([d1, d2], axis=1).fillna(0.0)
    return pd.concat([out, deriv], axis=1)


# --------------------------------------------------------------------------- #
# 4. Normalized life position
# --------------------------------------------------------------------------- #
def add_normalized_life_position(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``normalized_life_position`` = current_cycle / max_cycle_for_engine.

    Gives the model an explicit sense of "how far through its life" the engine
    is, on a 0..1 scale.

    NOTE on TRAIN vs TEST:
      * TRAIN engines run to failure, so max_cycle = the failure cycle and this
        is the true fraction of life elapsed.
      * TEST engines are truncated before failure, so max_cycle is only the last
        *observed* cycle — the denominator is smaller than the true lifetime and
        the value is therefore an over-estimate of true life fraction. This is a
        known, accepted approximation (we cannot know the true lifetime of a
        still-running engine); it is consistent within each test engine and the
        model treats it as just another monotone-increasing feature.
    """
    out = df.copy()
    max_cycle = out.groupby("unit_number")["time_cycles"].transform("max")
    out["normalized_life_position"] = out["time_cycles"] / max_cycle
    return out


# --------------------------------------------------------------------------- #
# 5. Lag features (LSTM-oriented; excluded from XGBoost by default)
# --------------------------------------------------------------------------- #
def add_lag_features(
    df: pd.DataFrame, cols: list[str], lags: tuple[int, ...] = (1,)
) -> pd.DataFrame:
    """Add per-engine lagged sensor values (default t-1).

    These are kept available primarily for the Phase 7 CNN-LSTM, which consumes
    raw sequential context naturally. They are deliberately EXCLUDED from the
    default XGBoost feature set (see ``build_features(include_lags=False)``):
    the rolling-window stats already encode recent history, and raw lags are
    highly collinear with the current value, which adds noise to a tree model
    without adding information.

    The first row of each engine has no prior cycle -> NaN -> back-filled with
    the engine's own first observed value (a stand-in for "same as now").
    """
    out = df.copy()
    grouped = out.groupby("unit_number")[cols]
    for lag in lags:
        lagged = grouped.shift(lag)
        lagged.columns = [f"{c}_lag{lag}" for c in cols]
        out = pd.concat([out, lagged], axis=1)

    # Back-fill the leading NaN within each engine so no NaN reaches a model.
    lag_cols = [f"{c}_lag{lag}" for lag in lags for c in cols]
    out[lag_cols] = out.groupby("unit_number")[lag_cols].bfill()
    return out


# --------------------------------------------------------------------------- #
# 6. Orchestrator
# --------------------------------------------------------------------------- #
def build_features(
    df: pd.DataFrame, include_lags: bool = False
) -> tuple[pd.DataFrame, list[str]]:
    """Run the full feature pipeline and return (engineered_df, feature_cols).

    Order: detect informative sensors -> rolling stats -> derivatives ->
    life position -> (optional) lags. ``feature_cols`` is the list of columns a
    model should actually train on (excludes ids, the raw target, and — when
    ``include_lags`` is False — the lag columns).

    Asserts the returned feature matrix contains no NaN/inf, so downstream
    phases can trust it.
    """
    sensors = informative_sensor_cols(df)
    dropped = set(constant_raw_cols(df))  # raw zero-variance cols never used

    out = add_rolling_features(df, sensors)
    out = add_derivative_features(out, sensors)
    out = add_normalized_life_position(out)
    if include_lags:
        out = add_lag_features(out, sensors)

    # Model features = everything engineered, minus ids/target, the raw constant
    # columns, and (when disabled) the lag columns.
    feature_cols = [
        c
        for c in out.columns
        if c not in NON_FEATURE_COLS
        and c not in dropped
        and (include_lags or "_lag" not in c)
    ]

    # Guard: no NaN/inf may reach the model.
    block = out[feature_cols]
    n_nan = int(block.isna().to_numpy().sum())
    n_inf = int(np.isinf(block.to_numpy()).sum())
    assert n_nan == 0, f"{n_nan} NaNs in feature matrix"
    assert n_inf == 0, f"{n_inf} infs in feature matrix"

    return out, feature_cols


# --------------------------------------------------------------------------- #
# Convenience: build + persist train/test feature frames
# --------------------------------------------------------------------------- #
def build_train_test_features(
    include_lags: bool = False, save: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build features for both train and test (with RUL labels attached).

    Returns (train_feat, test_feat, feature_cols). Optionally caches the frames
    to ``data/processed/*.parquet`` so later phases load instantly.
    """
    download_data()
    train = prepare_train_targets(load_train())
    test = add_test_rul(load_test(), load_rul())

    train_feat, feature_cols = build_features(train, include_lags=include_lags)
    test_feat, _ = build_features(test, include_lags=include_lags)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        train_feat.to_parquet(PROCESSED_DIR / "train_features.parquet", index=False)
        test_feat.to_parquet(PROCESSED_DIR / "test_features.parquet", index=False)

    return train_feat, test_feat, feature_cols


# --------------------------------------------------------------------------- #
# Phase-4 visible output
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print("=" * 76)
    print("NASA C-MAPSS FD001 — Phase 4 feature engineering")
    print("=" * 76)

    download_data()
    train = prepare_train_targets(load_train())

    sensors = informative_sensor_cols(train)
    print(f"\nInformative sensors ({len(sensors)}): {sensors}")

    # ---- Eyeball on a tiny slice first (engine 1, first cycles) --------------
    eng1 = train[train["unit_number"] == 1].copy()
    eng1_feat = add_derivative_features(
        add_rolling_features(eng1, sensors), sensors
    )
    peek = ["time_cycles", "sensor_4", "sensor_4_roll10_mean", "sensor_4_d1", "sensor_4_d2"]
    print("\n--- Engine 1, first 5 cycles (sensor_4 raw vs engineered) ---")
    print(eng1_feat[peek].head().to_string(index=False))

    # ---- Full build ----------------------------------------------------------
    train_feat, feature_cols = build_features(train, include_lags=False)
    print(f"\nRaw train shape       : {train.shape}")
    print(f"Engineered train shape: {train_feat.shape}")
    print(f"Model feature columns : {len(feature_cols)}")
    print(f"First 12 features     : {feature_cols[:12]}")

    # ---- NaN/inf assertion (already enforced inside build_features) ----------
    n_nan = int(train_feat[feature_cols].isna().to_numpy().sum())
    print(f"\nNaN count in feature matrix = {n_nan}  (assertion passed)")

    # ---- Persist train + test feature frames ---------------------------------
    tr, te, fc = build_train_test_features(include_lags=False, save=True)
    print(f"\nSaved -> data/processed/train_features.parquet  {tr.shape}")
    print(f"Saved -> data/processed/test_features.parquet   {te.shape}")
    print("\n[features] Phase 4 feature engineering complete.")
