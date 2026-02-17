# Experiment 11: Top Brand Isolation Proof
**Date**: 2026-02-17
**Goal**: Prove the standalone benefit of `is_top_brand` feature by adding it to the Exp08 baseline and removing interest rate context.

## 1. Methodology
- **Baseline**: Exp08 (Recent Data + Clusters + Quality).
- **Change**: Added `is_top_brand` (10 major construction companies).
- **Isolation**: No Macro features, No weighting additions.
## 2. Results
- **CV RMSE**: **7,303** (Exp08 CV was in the same range).
- **Status**: Successfully isolated the brand feature. Needs LB verification to confirm improvement over 15,642.

## 3. Conclusion
- The model has been cleaned of experimental noise. This submission will purely prove if 'Brand' is a winning feature.
