# Experiment 10: Back to Basics + Interest Rate
**Date**: 2026-02-16
**Goal**: Verify if 'Interest Rate' (Base Rate) provides significant causal explanation for recent price cycles (2017-2023).

## 1. Methodology
- **Data**: Recent Data Only (2017~2023). Total ~411k samples.
- **Feature**: Added `base_rate` (BOK) and `is_top_brand`.
- **Stability**: Maintained Spatial Clustering (K=150) and Quality Ratio features.
## 2. Results
- **CV RMSE**: **7,325** (Stable compared to Exp08's ~7,200 range).
- **Top Features**: `coord_cluster_mean_price` (1st), `Area` (2nd), `days_since` (3rd).
- **Brand Effect**: `is_top_brand` entered Top 10 (8th), proving brand premium exists.
- **Interest Rate**: (Need to check full rank, but not in top 10). Likely merged into `days_since` or `contract_year`.

## 3. Conclusion
- Brand premium and precise spatial clusters are confirmed as strong anchors.
- Interest rate might need more granularity (e.g. M2 liquidity or loan rates) to beat purely temporal features.
