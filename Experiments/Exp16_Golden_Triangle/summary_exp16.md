# Experiment 16: Golden Triangle Strategy
**Date**: 2026-02-18 16:42
**Goal**: Expand location features by adding distances to all 3 Major Business Districts (GBD, CBD, YBD).

## 1. Methodology
- **Added Features**: `dist_cbd` (Gwanghwamun), `dist_ybd` (Yeouido), `min_dist_to_job`.
- **Validation**: Honest Time-Split (Train ~2023.03 / Valid 2023.04+).
- **Weights**: Exponential Time-Weighting (Same as Exp15).
- **Data**: Recent Data Only (2017+).

## 2. Results
- **CV RMSE**: **13,101** (Improved from Exp15: 13,176).
- **Key Insight**: `min_dist_to_job` ranked very high in feature importance, proving that accessibility to all major job centers is crucial, not just Gangnam.
## 3. Next Steps
- Implement OOF Target Encoding for High-Cardinality categories (Apt Name, Road Name) to capture brand/neighborhood premiums.
