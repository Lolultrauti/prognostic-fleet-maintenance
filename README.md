# Prognostic Fleet Maintenance

**Predicting Remaining Useful Life (RUL) with Cost-Aware Modeling** — NASA C-MAPSS Turbofan Engine Degradation dataset (FD001).

> 🚧 Work in progress. This README is a stub and is fully written up in Phase 10.

## Problem
Predict how many operational cycles a turbofan engine has left before failure
(its Remaining Useful Life), and turn that prediction into a cost-aware
maintenance decision for an entire fleet.

## Dataset
NASA C-MAPSS, subset **FD001** (single operating condition, single fault mode):
- 100 train engines run to failure.
- 100 test engines truncated before failure, with true RUL provided.
- 3 operational settings + 21 sensor channels per cycle.

See `src/data.py` for loaders and the citation.

## Project structure
```
predictive-maintenance/
├── data/raw/, data/processed/
├── notebooks/        # EDA
├── src/              # data.py, features.py, model.py, evaluation.py
├── tests/            # unit tests
├── app/              # Streamlit demo
├── README.md
└── pyproject.toml
```

## How to run (so far)
```bash
poetry install
poetry run python src/data.py   # downloads FD001 and prints data summaries
```

## Build phases
1. **Setup & Data** ✅ — Poetry project, FD001 download + loaders.
2. RUL target engineering (piecewise-linear clipping at 125).
3. EDA notebook.
4. Feature engineering.
5. Cost functions + NASA (PHM08) scoring.
6. XGBoost baseline tuned on business cost.
7. CNN-LSTM (PyTorch).
8. SHAP explainability.
9. Streamlit demo.
10. Tests, README, deploy.
