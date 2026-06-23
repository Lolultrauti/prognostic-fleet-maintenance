"""
src/cnn_lstm.py
===============
A CNN-LSTM deep model for RUL prediction (PyTorch).

Contrast with Phase 6: XGBoost needed us to hand-craft ~227 features (rolling
stats, derivatives, ...). This model instead reads the RAW sensor time series
directly. The architecture does the feature extraction itself:

    raw window (L cycles x 15 sensors)
        -> 1D CNN over time      : learns local degradation *shapes/patterns*
                                   (a short convolution = a learned, data-driven
                                   replacement for our hand-made rolling/derivative
                                   features)
        -> LSTM over the sequence : integrates those local patterns across the
                                   whole window, carrying long-range temporal
                                   state (the trend / memory)
        -> Linear head            : regresses a single RUL value

So the only "feature engineering" here is: pick the informative sensors,
standardise them, and slice fixed-length sliding windows. Everything else is
learned.

Honesty note (resolved fairly in the README): this does NOT need to beat the
tuned XGBoost baseline. It needs to work, be explainable, and be compared
honestly on the same held-out test set with the same metrics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import download_data, load_rul, load_test, load_train
from evaluation import cost_sensitive_mse, fleet_business_cost, nasa_score, rmse
from features import informative_sensor_cols
from targets import RUL_CAP, add_test_rul, last_cycle_per_engine, prepare_train_targets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

WINDOW_LEN = 30          # cycles of history fed to the model at once
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Standardisation (fit on TRAIN only — never peek at test statistics)
# --------------------------------------------------------------------------- #
class SensorScaler:
    """Per-sensor z-score standardisation. Mean/std learned from train rows."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, df, sensors: list[str]) -> "SensorScaler":
        self.mean_ = df[sensors].mean().to_numpy()
        self.std_ = df[sensors].std().to_numpy()
        self.std_[self.std_ == 0] = 1.0   # guard (shouldn't happen post-filter)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean_) / self.std_


# --------------------------------------------------------------------------- #
# Sliding-window construction
# --------------------------------------------------------------------------- #
def make_windows(df, sensors, scaler, window_len=WINDOW_LEN, last_only=False):
    """Turn per-engine sensor sequences into fixed-length windows.

    For each engine (rows ordered by cycle) we form, for every cycle end `t`, a
    window of the previous `window_len` standardised sensor readings ending at
    `t`; the label is the RUL at cycle `t`. Engines (or early cycles) shorter
    than the window are LEFT-PADDED by repeating the earliest reading, so we can
    always produce a window — important for short test engines.

    Returns X with shape (N, n_sensors, window_len) [channels-first for Conv1d]
    and y with shape (N,). With ``last_only=True`` we keep just the single
    window ending at each engine's last observed cycle (the test-eval scenario).
    """
    X_list, y_list = [], []
    for _, g in df.groupby("unit_number"):
        g = g.sort_values("time_cycles")
        arr = scaler.transform(g[sensors].to_numpy())     # (n, S)
        rul = g["RUL"].to_numpy()
        n = len(g)

        ends = [n] if last_only else range(1, n + 1)
        for end in ends:
            start = end - window_len
            if start < 0:
                pad = np.repeat(arr[0:1], -start, axis=0)  # repeat first reading
                window = np.vstack([pad, arr[0:end]])
            else:
                window = arr[start:end]
            X_list.append(window.T)                        # (S, L) channels-first
            y_list.append(rul[end - 1])

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class CNNLSTM(nn.Module):
    """1D-CNN feature extractor feeding an LSTM, then a linear RUL head."""

    def __init__(self, n_sensors: int, cnn_channels: int = 32,
                 lstm_hidden: int = 64, lstm_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        # Conv1d operates over the time axis; padding='same' keeps length L.
        self.conv = nn.Sequential(
            nn.Conv1d(n_sensors, cnn_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, x):                  # x: (B, S, L)
        z = self.conv(x)                   # (B, C, L)
        z = z.permute(0, 2, 1)             # (B, L, C)  -> sequence for the LSTM
        out, _ = self.lstm(z)              # (B, L, H)
        last = out[:, -1, :]               # final timestep's hidden state
        return self.head(last).squeeze(1)  # (B,)


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def train_model(model, X, y, epochs=20, batch_size=512, lr=1e-3):
    """Standard supervised loop, MSE loss, Adam. Prints loss each few epochs."""
    model.to(DEVICE)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            rmse_epoch = (running / len(ds)) ** 0.5
            print(f"  epoch {epoch:2d}/{epochs}  train RMSE {rmse_epoch:6.2f}")
    return model


@torch.no_grad()
def predict(model, X) -> np.ndarray:
    """Predict RUL for windows X, clipped to the valid [0, RUL_CAP] range."""
    model.eval()
    preds = model(torch.from_numpy(X).to(DEVICE)).cpu().numpy()
    return np.clip(preds, 0.0, RUL_CAP)


# --------------------------------------------------------------------------- #
# Phase-7 visible output
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 76)
    print("Phase 7 — CNN-LSTM deep model (raw sensor windows)")
    print("=" * 76)

    download_data()
    train = prepare_train_targets(load_train())
    test = add_test_rul(load_test(), load_rul())

    sensors = informative_sensor_cols(train)
    print(f"\ndevice={DEVICE} | sensors={len(sensors)} | window_len={WINDOW_LEN}")

    # Fit scaler on train, build windows for train (all) and test (last per engine).
    scaler = SensorScaler().fit(train, sensors)
    X_train, y_train = make_windows(train, sensors, scaler)
    X_test, y_test = make_windows(test, sensors, scaler, last_only=True)
    print(f"train windows: {X_train.shape} | test windows: {X_test.shape}")

    # Train.
    print("\nTraining CNN-LSTM ...")
    model = CNNLSTM(n_sensors=len(sensors))
    model = train_model(model, X_train, y_train, epochs=80, batch_size=256)

    # Evaluate on held-out test (same metrics as XGBoost).
    y_pred = predict(model, X_test)
    dl_metrics = {
        "rmse": rmse(y_test, y_pred),
        "nasa_score": nasa_score(y_test, y_pred),
        "cost_sensitive_mse": cost_sensitive_mse(y_test, y_pred),
        "fleet_cost": fleet_business_cost(y_test, y_pred),
    }

    # XGBoost Phase-6 results for an honest side-by-side (from the tuned run).
    xgb_metrics = {"rmse": 14.51, "nasa_score": 417.00,
                   "cost_sensitive_mse": 1342.13, "fleet_cost": 510000.0}

    print("\nHELD-OUT TEST METRICS — CNN-LSTM vs tuned XGBoost (Phase 6):")
    print(f"  {'metric':<22}{'CNN-LSTM':>14}{'XGBoost':>14}")
    for k in ["rmse", "nasa_score", "cost_sensitive_mse", "fleet_cost"]:
        print(f"  {k:<22}{dl_metrics[k]:>14.2f}{xgb_metrics[k]:>14.2f}")

    bd = fleet_business_cost(y_test, y_pred, return_breakdown=True)
    print("\nCNN-LSTM fleet business-cost breakdown:")
    for k, v in bd.items():
        print(f"  {k:<22}: {v:,.0f}" if isinstance(v, float) else f"  {k:<22}: {v}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "sensors": sensors,
                "window_len": WINDOW_LEN,
                "scaler_mean": scaler.mean_, "scaler_std": scaler.std_},
               MODEL_DIR / "cnn_lstm.pt")
    print("\nSaved -> models/cnn_lstm.pt")
    print("[cnn_lstm] Phase 7 complete.")
