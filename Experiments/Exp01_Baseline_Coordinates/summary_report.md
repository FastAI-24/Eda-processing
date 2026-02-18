# Experiment 01: Baseline Model with Coordinates (좌표 기반 베이스라인)

**날짜**: 2026-02-16
**목표**: 결측된 좌표(78%)를 복원하고, 이를 포함한 베이스라인 모델을 수립하여 성능을 확인한다.

## 주요 결과 (Key Metrics)

- **CV RMSE**: 5,601 (Shuffle K-Fold 사용 시)
- **Leaderboard RMSE**: 16,112
- **Adversarial Validation AUC**: 1.0000 (심각한 분포 차이 발생)

## 핵심 결론 (Key Conclusion)

1. **과적합(Overfitting)**: Shuffle Split을 사용한 검증 방식이 시계열 데이터의 특성을 무시하여, 모델이 'days_since'와 같은 과거 추세에 과도하게 의존하게 만듦. 결과적으로 CV와 실제 리더보드 점수 간의 괴리가 큼.
2. **좌표 데이터의 한계**: 좌표 정보만으로는 성능 향상에 한계가 있음.
3. **향후 계획 (Next Steps)**:
    - **검증 전략 수정**: 미래 데이터를 예측하는 과제이므로, Time-based Split 또는 시계열 교차 검증 도입 필요.
    - **외부 데이터 활용**: 지하철/버스 등 교통 편의성(Transport) 데이터를 추가하여 지리적 특성을 더 구체화해야 함 (Exp05 예정).
