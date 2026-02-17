# Experiment 13: CatBoost Revolutions (GPU) - 범주형 변수 최적화 및 그룹 검증

**날짜**: 2026-02-17
**목표**: LightGBM의 한계를 넘기 위해 범주형 변수 처리에 특화된 CatBoost를 도입하고, 단지별 과적합을 방지하기 위해 Group K-Fold를 적용한다.

## 1. 방법론 (Methodology)

- **모델**: CatBoost (GPU 가속 활용).
- **검증 전략**: Group K-Fold (아파트 단지별로 그룹을 나누어 동일 단지가 학습과 검증에 동시에 들어가지 않도록 통제).
- **특징**: 범주형 변수(Label Encoding 대신)를 CatBoost의 내장 기능을 통해 직접 처리.

## 2. 결과 (Results)

- **CV RMSE (Group K-Fold)**: **20,779**
- **LB RMSE**: **25,140** (참혹한 실패)
- **해석**: 시계열 데이터에서 시간축을 고려하지 않은 `GroupKFold`는 강력한 데이터 드리프트(Drift)를 유발함. "한 번도 보지 못한 아파트를 맞히라"는 너무 어려운 과제와 CatBoost의 복잡한 연산이 노이즈에 과적합됨.

## 3. 결론 (Conclusion)

- 단지별 정보를 외워서 맞추는 것이 아닌, 미학습 단지에도 대응 가능한 일반화 성능을 확보하는 과정임.
