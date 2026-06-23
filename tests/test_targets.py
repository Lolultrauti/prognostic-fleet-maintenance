"""
Unit tests for RUL target engineering (src/targets.py).

These check the two pieces of logic an interviewer is most likely to probe:
the linear RUL countdown and the 125-cycle piecewise-linear clip, plus the
slightly subtle test-set RUL reconstruction.
"""

import pandas as pd
import pytest

from targets import (
    RUL_CAP,
    add_linear_rul,
    add_test_rul,
    clip_rul,
    last_cycle_per_engine,
    prepare_train_targets,
)


def _toy_train():
    """Two engines: #1 runs 5 cycles, #2 runs 3 cycles (last cycle = failure)."""
    return pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 1, 1, 2, 2, 2],
            "time_cycles": [1, 2, 3, 4, 5, 1, 2, 3],
        }
    )


def test_linear_rul_counts_down_to_zero_per_engine():
    df = add_linear_rul(_toy_train())
    eng1 = df[df["unit_number"] == 1]["RUL"].tolist()
    eng2 = df[df["unit_number"] == 2]["RUL"].tolist()
    # max_cycle - current_cycle, so the failure row is exactly 0.
    assert eng1 == [4, 3, 2, 1, 0]
    assert eng2 == [2, 1, 0]


def test_clip_caps_at_125_and_leaves_small_values():
    df = pd.DataFrame({"RUL": [0, 50, 125, 200, 361]})
    out = clip_rul(df, cap=125)["RUL"].tolist()
    assert out == [0, 50, 125, 125, 125]


def test_clip_is_a_noop_when_all_below_cap():
    df = pd.DataFrame({"RUL": [0, 10, 124]})
    assert clip_rul(df)["RUL"].tolist() == [0, 10, 124]


def test_prepare_train_targets_never_exceeds_cap():
    # An engine living 300 cycles must have early rows flattened to the cap.
    long_engine = pd.DataFrame(
        {"unit_number": [1] * 300, "time_cycles": list(range(1, 301))}
    )
    out = prepare_train_targets(long_engine)
    assert out["RUL"].max() == RUL_CAP
    assert out["RUL"].min() == 0
    # The first row had linear RUL 299 -> must be clipped to the cap.
    assert out.iloc[0]["RUL"] == RUL_CAP


def test_add_test_rul_reconstructs_from_end_value():
    # Engine truncated at cycle 3; true RUL at that last cycle is 10.
    test = pd.DataFrame({"unit_number": [1, 1, 1], "time_cycles": [1, 2, 3]})
    rul = pd.DataFrame({"unit_number": [1], "RUL": [10]})
    out = add_test_rul(test, rul)
    # RUL = end_rul + (last_cycle - current_cycle): 12, 11, 10.
    assert out["RUL"].tolist() == [12, 11, 10]


def test_add_test_rul_applies_same_cap():
    test = pd.DataFrame({"unit_number": [1, 1], "time_cycles": [1, 2]})
    rul = pd.DataFrame({"unit_number": [1], "RUL": [124]})
    out = add_test_rul(test, rul)
    # cycle 1 would be 125, cycle 2 = 124; cap is 125 so nothing truncates here.
    assert out["RUL"].tolist() == [125, 124]


def test_last_cycle_per_engine_picks_the_final_row():
    df = pd.DataFrame(
        {"unit_number": [1, 1, 2], "time_cycles": [1, 2, 1], "RUL": [9, 8, 5]}
    )
    last = last_cycle_per_engine(df).sort_values("unit_number")
    assert last["time_cycles"].tolist() == [2, 1]
    assert last["RUL"].tolist() == [8, 5]
