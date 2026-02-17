# Experiment 09: Macro Economics Boost + All Data + Time Decay
**Date**: 2026-02-16
**Goal**: Overcome price plateaus by introducing macro-economic context and utilizing historical data with decay weights.

## 1. Methodology
- **Data**: All historical data (2007-2023) restored.
- **Macro Feature**: Added BOK Base Rate (기준금리) matched by year/month.
- **Weighting**: Applied `exp((year - 2023) * 0.1)` to prioritize recent trends while keeping historical spatial premiums.
- **Integrated Features**: Quality features (Exp07) + Spatial Clusters (Exp08).
## 2. Expected Results
- Introduction of 'Interest Rates' provides the missing link between time and price fluctuation.
- Historical data stabilizes the 'Location Premium' (Cluster Mean Price).

## 3. Conclusion
- Moving from 'Where' to 'Why' - providing causal context to the model.
