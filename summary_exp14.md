# Experiment 14: Honest Validation (Time Split)
**Date**: 2026-02-17 03:44
**Goal**: Establish a validation strategy that correlates with the Leaderboard by using Time-based Split.

## 1. Methodology
- **Split Strategy**: Train (~2023.03) / Valid (2023.04~2023.06).
- **Model**: LightGBM (Exp12-v2 Features + Log Target).
- **Rationale**: Shuffle Split leaked future information within clusters. Time split mimics the forecasting task.

## 2. Results
- **CV RMSE**: ~13,670 (Measured on 2023.04-06 data).
- **Interpretation**: The large gap between previous CV (~7,000) and LB (~16,000) is addressed. This CV score is a realistic estimate of model performance on unseen future data.
## 3. Conclusion
- We now have a trustworthy compass. Improving this CV score should directly improve LB score.
