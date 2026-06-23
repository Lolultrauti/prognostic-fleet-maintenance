"""
Unit tests for feature engineering (src/features.py).

Synthetic data only (no dataset download) so the tests are fast and hermetic.
The important guarantees: no NaN reaches the model, temporal features do not
leak across engine boundaries, constant sensors are dropped, and the leaky
normalized_life_position is kept out of the model feature set.
"""

import numpy as np
import pandas as pd
import pytest

from data import SENSOR_COLS, SETTING_COLS
from features import (
    add_derivative_features,
    build_features,
    constant_raw_cols,
    informative_sensor_cols,
)


def _toy_frame():
    """Two engines with all 26 raw columns; sensor_1 is constant, others vary."""
    rows = []
    for unit, n in [(1, 6), (2, 5)]:
        for t in range(1, n + 1):
            row = {"unit_number": unit, "time_cycles": t}
            for c in SETTING_COLS:
                row[c] = 0.5 + t * 0.01            # varies
            for c in SENSOR_COLS:
                row[c] = float(t) + unit           # varies with cycle
            row["sensor_1"] = 42.0                  # deliberately constant
            rows.append(row)
    df = pd.DataFrame(rows)
    df["RUL"] = 100  # dummy target column
    return df


def test_constant_sensor_is_detected_and_excluded():
    df = _toy_frame()
    assert "sensor_1" in constant_raw_cols(df)
    assert "sensor_1" not in informative_sensor_cols(df)


def test_build_features_has_no_nan_or_inf():
    df = _toy_frame()
    feat, cols = build_features(df)
    block = feat[cols]
    assert int(block.isna().to_numpy().sum()) == 0
    assert not np.isinf(block.to_numpy()).any()


def test_ids_target_and_leaky_feature_excluded_from_model_cols():
    df = _toy_frame()
    _, cols = build_features(df)
    for forbidden in ["unit_number", "time_cycles", "RUL",
                      "normalized_life_position", "sensor_1"]:
        assert forbidden not in cols


def test_lags_excluded_by_default_included_when_requested():
    df = _toy_frame()
    _, no_lag = build_features(df, include_lags=False)
    _, with_lag = build_features(df, include_lags=True)
    assert not any("_lag" in c for c in no_lag)
    assert any("_lag" in c for c in with_lag)


def test_derivatives_do_not_leak_across_engines():
    # The first row of engine 2 must have derivative 0 (no prior cycle within
    # engine 2) — it must NOT diff against engine 1's last row.
    df = _toy_frame()
    sensors = informative_sensor_cols(df)
    out = add_derivative_features(df, sensors)
    eng2_first = out[out["unit_number"] == 2].iloc[0]
    assert eng2_first["sensor_2_d1"] == 0.0
    assert eng2_first["sensor_2_d2"] == 0.0


def test_rolling_mean_at_engine_start_uses_only_that_engine():
    # First cycle's rolling mean must equal that engine's own first reading,
    # proving the window did not reach back into the previous engine.
    df = _toy_frame()
    feat, _ = build_features(df)
    eng2_first = feat[feat["unit_number"] == 2].iloc[0]
    # sensor_2 for engine 2 at t=1 is (1 + 2) = 3.0 in our toy frame.
    assert eng2_first["sensor_2_roll10_mean"] == pytest.approx(3.0)
