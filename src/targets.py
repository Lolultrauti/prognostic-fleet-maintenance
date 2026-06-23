"""
src/targets.py
==============
RUL (Remaining Useful Life) target engineering for NASA C-MAPSS FD001.

This module turns the raw cycle counters into the regression LABEL our models
predict: how many operational cycles remain before the engine fails.

Two ideas live here:

1. Linear RUL
   In the TRAIN set every engine runs to failure, so for any row we know the
   true remaining life exactly:

       RUL(row) = (last cycle of that engine) - (current cycle)

   The last row of each engine therefore has RUL = 0 (failure), and RUL counts
   down linearly as cycles increase.

2. Piecewise-linear RUL clipping (cap at 125)  -- THE KEY MODELLING CHOICE
   A purely linear target says a brand-new engine at cycle 1 with 300 cycles
   left is "300 units unhealthy", and asks the model to tell 300-left apart from
   299-left. That is both unrealistic and harmful:

     * Physically, an engine shows almost no measurable degradation when it is
       far from failure. Its sensor signature at RUL=300 looks essentially
       identical to RUL=200 -- there is no signal there to learn. Only as it
       approaches end-of-life do sensors drift in a detectable way.
     * Forcing the model to fit huge early RUL values wastes capacity on a
       region we cannot predict anyway, and inflates the loss with errors that
       do not matter for maintenance decisions.

   The standard fix (Heimes 2008; used throughout the C-MAPSS literature) is a
   piecewise-linear target: clip RUL to a constant ceiling R_early (commonly
   125 for FD001). Health is treated as "effectively full" until the engine
   gets within 125 cycles of failure, after which RUL decreases linearly to 0:

       RUL_clipped = min(RUL_linear, 125)

   The model then only has to learn the part of the curve that is both
   predictable and operationally relevant.
"""

from __future__ import annotations

import pandas as pd

# Standard early-RUL ceiling for FD001 in the prognostics literature.
RUL_CAP = 125


# --------------------------------------------------------------------------- #
# Train targets
# --------------------------------------------------------------------------- #
def add_linear_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    """Add an uncapped, linear ``RUL`` column to a run-to-failure TRAIN frame.

    For each engine, RUL = max_cycle_of_engine - current_cycle, so the failure
    row gets RUL = 0. Returns a new DataFrame (does not mutate the input).
    """
    df = train_df.copy()

    # Per-engine final cycle = the cycle at which that engine failed.
    max_cycle = df.groupby("unit_number")["time_cycles"].transform("max")

    # Cycles remaining until that failure.
    df["RUL"] = max_cycle - df["time_cycles"]
    return df


def clip_rul(df: pd.DataFrame, cap: int = RUL_CAP, col: str = "RUL") -> pd.DataFrame:
    """Apply piecewise-linear clipping: RUL = min(RUL, cap).

    See the module docstring for WHY (no learnable signal far from failure).
    Returns a new DataFrame.
    """
    out = df.copy()
    out[col] = out[col].clip(upper=cap)
    return out


def prepare_train_targets(train_df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Convenience: linear RUL then piecewise-linear clipping, in one call."""
    return clip_rul(add_linear_rul(train_df), cap=cap)


# --------------------------------------------------------------------------- #
# Test targets
# --------------------------------------------------------------------------- #
def add_test_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Build per-row RUL labels for the TRUNCATED test set.

    Unlike train, test engines stop BEFORE failure. ``RUL_FD001.txt`` gives the
    true RUL only at each engine's LAST observed cycle. We reconstruct RUL for
    every earlier row by adding how far that row is from the engine's last
    observed cycle:

        RUL(row) = true_RUL_at_last_cycle + (last_cycle - current_cycle)

    Then apply the same 125 cap so the test labels match the train target space.

    Most evaluation only scores the final row of each engine (one prediction per
    engine), but having full per-row labels is handy for plots and the CNN-LSTM
    windows later.
    """
    df = test_df.copy()

    # Last observed cycle per engine in the (truncated) test set.
    last_cycle = df.groupby("unit_number")["time_cycles"].transform("max")

    # Map each engine's true end-of-life RUL from the RUL file.
    true_end_rul = df["unit_number"].map(
        rul_df.set_index("unit_number")["RUL"]
    )

    # Reconstruct per-row RUL, then clip identically to train.
    df["RUL"] = true_end_rul + (last_cycle - df["time_cycles"])
    df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def last_cycle_per_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the final row of each engine (the standard test-eval point)."""
    idx = df.groupby("unit_number")["time_cycles"].idxmax()
    return df.loc[idx].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Phase-2 visible output
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from data import download_data, load_test, load_train, load_rul

    print("=" * 76)
    print("NASA C-MAPSS FD001 — Phase 2 RUL target engineering")
    print("=" * 76)

    if not download_data():
        raise SystemExit(1)

    train = load_train()
    test = load_test()
    rul = load_rul()

    # ---- Train ---------------------------------------------------------------
    train_lin = add_linear_rul(train)
    train_cap = clip_rul(train_lin, cap=RUL_CAP)

    print("\n--- TRAIN engine 1, first & last 3 cycles (linear vs capped) ---")
    eng1 = train_cap[train_cap["unit_number"] == 1]
    cols = ["unit_number", "time_cycles", "RUL"]
    lin1 = train_lin[train_lin["unit_number"] == 1][cols].rename(columns={"RUL": "RUL_linear"})
    show = lin1.merge(train_cap[train_cap["unit_number"] == 1][cols], on=["unit_number", "time_cycles"])
    show = show.rename(columns={"RUL": "RUL_capped"})
    print(pd.concat([show.head(3), show.tail(3)]).to_string(index=False))

    print(f"\nLinear RUL range : {train_lin['RUL'].min()} .. {train_lin['RUL'].max()}")
    print(f"Capped RUL range : {train_cap['RUL'].min()} .. {train_cap['RUL'].max()} (cap={RUL_CAP})")
    pct_capped = (train_lin["RUL"] > RUL_CAP).mean() * 100
    print(f"Rows hitting the cap: {pct_capped:.1f}% of train rows are flattened to {RUL_CAP}")

    # ---- Test ----------------------------------------------------------------
    test_rul = add_test_rul(test, rul, cap=RUL_CAP)
    test_last = last_cycle_per_engine(test_rul)

    print("\n--- TEST last-cycle RUL (model is scored here), first 5 engines ---")
    print(test_last[["unit_number", "time_cycles", "RUL"]].head().to_string(index=False))

    # Sanity check: last-cycle RUL must equal the capped RUL file value.
    expected = rul.copy()
    expected["RUL_capped_truth"] = expected["RUL"].clip(upper=RUL_CAP)
    merged = test_last.merge(expected[["unit_number", "RUL_capped_truth"]], on="unit_number")
    ok = (merged["RUL"] == merged["RUL_capped_truth"]).all()
    print(f"\nSanity check — test last-cycle RUL matches capped RUL file: {ok}")

    print("\n[targets] Phase 2 RUL targets built successfully.")
