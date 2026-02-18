# Experiment 02: Validation Strategy Correction (검증 전략 수정)

**날짜**: 2026-02-16
**목표**: CV(5,600)와 LB(16,000) 점수 간의 괴리를 해결하고 신뢰할 수 있는 검증 체계를 수립한다.

## 1. 원인 분석 (Diagnosis)

- **Adversarial Validation**: AUC 1.0000 (완벽한 분리).
- **문제점**: 모델이 'days_since'와 같은 시간적 추세(과거)에 과적합되어, 미래(Test) 데이터의 패턴을 전혀 학습하지 못함.

## 2. 해결 방안 (Solution Implemented)

- **Adversarial Weighting**: Test 데이터와 유사한(최근) Train 데이터에 높은 가중치를 부여하여 학습.
- **파생 변수 추가**: `Building Age` (건물 연식) 추가 (새로운 검증 전략 테스트용).
- **검증 전략 변경**: `Shuffle K-Fold` -> **`Last 5% Hold-out`** (시계열 검증 도입).
  - 마지막 5% 데이터를 Validation Set으로 사용하여 미래 예측 성능을 평가.

## 3. 결과 (Results)

- **New Validation RMSE**: 약 **15,000 ~ 16,000** (Log Scale 0.158 수준).
- **해석**: 이제 내부 검증 점수(CV)가 Public Leaderboard 점수(16,112)와 유사해짐. **신뢰할 수 있는 검증 기준(Baseline)**을 확보함.

## 4. 향후 계획 (Next Step - Exp03)

- 검증 점수가 리더보드와 일치하므로, RMSE를 10,000 이하로 낮추기 위해 **교통 데이터(지하철/버스)**와 같은 외부 데이터를 본격적으로 도입할 예정.
