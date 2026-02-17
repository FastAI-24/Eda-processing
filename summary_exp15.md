# Experiment 15: Time Decay Weighting Strategy
**Date**: 2026-02-17 03:50
**Goal**: Prioritize recent market trends (2022-2023) over historical data using sample weights.

## 1. Methodology
- **Validation**: Honest Time-Split (Train ~2023.03 / Valid 2023.04+).
- **Weighting Scheme**: 2023 (2.0x), 2022 (1.5x), Others (1.0x).
- **Model**: LightGBM (Low Learning Rate for Stability).

## 2. Results
- **CV RMSE**: **13,176** (Improved from Exp14's 13,670).
- **Interpretation**: The model successfully captured the downward price trend of 2022-2023. This is the most realistic and performant model so far.
## 3. Next Steps
- Save submission chance. Proceed to Stacking/Ensemble locally to push CV below 13,000.
