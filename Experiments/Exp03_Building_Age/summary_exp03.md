# Experiment 03: Feature Verification - Building Age (건물 연식 효과 검증)

**날짜**: 2026-02-16
**목표**: 복잡한 가중치 적용 없이 '건물 연식' 변수가 모델 성능에 미치는 순수 효과를 검증한다.

## 1. 방법론 (Methodology)

- **추가 변수**: `Building Age` = `계약년도` - `건축년도`.
- **검증 전략**: Shuffle K-Fold (Exp01 베이스라인과 동일한 조건에서 비교하기 위함).
- **모델**: LightGBM (최적화된 하이퍼파라미터 적용).

## 2. 결과 (Results)

- **CV RMSE**: **5,372** (Exp01 베이스라인 5,601 대비 약 230 개선).
- **주요 변수 Importance**: `Dong_encoded`, `Area`, `days_since`, **`building_age`**.
- **결론**: 건물 연식은 **강력한 예측 변수**임이 확인됨. 노후 건물의 감가상각과 신축 건물의 프리미엄을 모델이 효과적으로 학습함.

## 3. 향후 계획 (Next Step - Exp04)

- CV 점수는 개선되었으나, Shuffle Split을 사용했기 때문에 리더보드(LB) 점수는 여전히 16,000점 근처에 머물 것으로 예상됨.
- 다음 실험(**Exp04**)에서는 외부 데이터를 추가하기 전, 시계열 데이터에 적합한 **TimeSeriesSplit**을 사용하여 현실적인 베이스라인을 재구축할 예정.
