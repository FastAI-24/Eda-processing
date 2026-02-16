# Experiment 04: True Baseline - TimeSeriesSplit (시계열 베이스라인 구축)

**날짜**: 2026-02-16
**목표**: 데이터 누수(Leakage) 없는 현실적인 검증 체계를 위해 TimeSeriesSplit을 도입하고 정규 베이스라인을 수립한다.

## 1. 방법론 (Methodology)

- **검증 전략**: `TimeSeriesSplit` (5 Splits). 과거 데이터로 학습하여 미래의 일정 구간을 예측하는 방식.
- **주요 변수**: `Building Age` (Exp03에서 확인된 핵심 변수 포함).
- **모델**: LightGBM (Exp03과 동일한 파라미터 유지).

## 2. 결과 (Results)

- **평균 CV RMSE**: 약 **18,000** (Shuffle Split 방식의 5,372보다 높게 나오지만, 실제 리더보드 점수와 더 유사한 경향을 보임).
- **리더보드 예상**: 검증 방식이 견고해졌으므로, 실제 제출 시 Exp03(16,313)보다 안정적인 성능을 보일 것으로 기대됨.
- **결론**: 시계열 데이터 특성상 단순 Shuffle 방식은 성능을 심각하게 낙관적으로 평가함(과적합 유도). **TimeSeriesSplit**이 훨씬 더 정교한 평가 지표를 제공함.

## 3. 향후 계획 (Next Step - Exp05)

- 구축된 시계열 베이스라인 위에 **교통 관련 변수**(`dist_to_subway`, `subway_line` 등)를 추가로 투입.
- 목표: 평균 RMSE를 18,000 수준에서 14,000 이하로 단축.
