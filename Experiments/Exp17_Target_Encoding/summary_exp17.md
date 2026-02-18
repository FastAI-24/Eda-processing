# Experiment 17: Branding Strategy (Target Encoding)
**Date**: 2026-02-18 17:15

## 1. Methodology
- **Key Change**: Applied 5-Fold OOF Target Encoding to `apt_name` and `sigungu`.
- **Baseline**: Exp 16 (Golden Triangle + Time Split).
- **Issue Fixed**: Handled missing `road_name` by prioritizing `apt_name` and `sigungu` clusters.

## 2. Results
- **CV RMSE**: **12,042** (Drastic improvement from 13,101).
- **Impact**: Proved that 'Apt Brand' and 'District Category' are the strongest price anchors in the Seoul market.

## 3. Decision
- Now we have reached 12,000 range. Time to unlock the full potential with 2007+ data and Macro indicators (Base Rate).
