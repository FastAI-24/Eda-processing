# 아파트 실거래가 예측: EDA 및 데이터 전처리 상세 보고서

**작성일**: 2026-02-16
**목적**: 팀원 공유 및 재현 가능한 분석 파이프라인 구축

---

## 1. 개요 (Overview)

본 보고서는 서울시 아파트 실거래가 예측 프로젝트의 **탐색적 데이터 분석(EDA)** 및 **전처리 과정**을 단계별로 상세히 기술합니다.

### 핵심 문제 및 해결

| 문제 | 규모 | 해결 방법 |
|:---|:--:|:---|
| **좌표 결측치** | 87만 건 (78%) | API 지오코딩 + Spatial Median Imputation |
| **좌표 이상치** | 3,137건 (0.3%) | 서울 범위 검증 + 동별 중앙값 대체 |
| **주차대수 결측치** | 약 30% | RandomForest 기반 예측 모델 (R² 0.86) |
| **불필요 컬럼** | 26개 | 상관분석 + 결측률 기반 제거 |

---

## 2. 초기 데이터 탐색 (Initial EDA)

### 2.1. 데이터 로딩 및 기본 통계

**데이터셋 구조**:

- Train: 1,119,088 rows × 48 columns
- Test: 267,627 rows × 47 columns (타겟 제외)

**타겟 변수(`실거래가`) 분포**:

![Target Distribution](./eda_output/target_distribution.png)

**주요 발견**:

- 가격 분포가 오른쪽으로 긴 꼬리(Right-skewed) 형태
- 로그 변환 필요성 확인 → 정규분포에 가깝게 변환

![Log Target Distribution](./eda_output/log_target_distribution.png)

**변환 후 효과**:

- 왜도(Skewness) 감소: 2.3 → 0.4
- 회귀 모델 성능 향상에 기여

### 2.2. 결측치 현황

| 컬럼명 | 결측 개수 | 결측률 |
|:---|--:|--:|
| `좌표X`, `좌표Y` | 869,670 | **77.7%** |
| `주차대수` | ~330,000 | **29.5%** |
| `k-관리비` | 1,095,167 | **97.9%** |
| `k-전화번호` | 1,091,234 | **97.5%** |

**전략 수립**:

1. 결측률 80% 이상 컬럼 → 삭제
2. 좌표 결측 → API + 통계적 보간
3. 주차대수 결측 → ML 기반 예측

---

## 3. 상관관계 분석 (Correlation Analysis)

### 3.1. 수치형 변수와 타겟 상관계수

![Correlation Matrix](./eda_output/correlation_matrix.png)

**상위 10개 양의 상관관계**:

- `전용면적(㎡)`: **0.577** ← (가장 강력한 예측 변수)
- `주차대수`: 0.348
- `계약년월`: 0.345 (시계열 트렌드)
- `k-연면적`: 0.344
- `k-주거전용면적`: 0.334
- `k-관리비부과면적`: 0.317
- `k-85㎡~135㎡이하`: 0.257
- `k-전체동수`: 0.234
- `k-전용면적별세대현황(60㎡~85㎡이하)`: 0.230
- `좌표X`: 0.135

**음의 상관관계**:

- `좌표Y`: **-0.312** (남북 방향, 강북 < 강남)

![Top Correlations](./eda_output/top_numeric_correlations.png)

### 3.2. 컬럼 제거 기준

**기준 1: 높은 결측률 (≥ 80%)**

- `k-관리비`, `k-전화번호`, `k-팩스번호`, `중개사소재지`, `거래유형`

**기준 2: 예측 무관 메타데이터**

- `본번`, `부번`, `등기신청일자`, `관리사무소팩스`

**기준 3: 다중공선성 (Multicollinearity)**

- `k-연면적` vs `k-주거전용면적` (상관계수 0.98) → 하나 제거
- `k-전체세대수` vs `k-전체동수` (상관계수 0.87) → 세대수 유지

**최종 제거 컬럼**: 26개

![Deleted Columns Viz](./eda_output/deleted_columns_viz_summary.png)

---

## 4. 이상치 탐지 및 처리 (Outlier Detection)

### 4.1. 전용면적 이상치

**탐지 기준**: > 300㎡ (초대형 평형)

**발견**: 11건

- 최대값: 424.32㎡ (타워팰리스 등 초고가 아파트)
- **조치**: 유지 (실제 거래 데이터이므로 삭제하지 않음)

### 4.2. 건축년도 이상치

**탐지 기준**: < 1960년 또는 > 2026년

**발견**: 0건 (모두 정상 범위)

### 4.3. 좌표 이상치 (Critical)

**서울시 정상 범위**:

- 경도(X): 126.7 ~ 127.3
- 위도(Y): 37.4 ~ 37.7

**이상치 발견**:

- X축 이탈: 3,137건
- Y축 이탈: 3,430건

**원인 분석**:

```
예시: 서울 강남구 삼성동 → 좌표 (35.99, 126.98)
실제 위치: 전라북도 익산시 삼성동 (동명 오류)
```

**API가 서울이 아닌 전라북도를 반환한 사례**

**조치**:

1. 이상치를 `NaN`으로 변환
2. 해당 법정동의 정상 데이터 중앙값으로 대체

![Coordinate Outliers](./eda_output/outliers/coordinate_outliers_map.png)

---

## 5. 지오코딩 프로세스 (Geocoding - Priority)

### 5.1. 문제 정의

**핵심 이슈**: 전체 데이터의 **78% (869,670건)** 가 좌표 결측

**단순 삭제 불가 이유**:

- 데이터의 80%를 잃게 됨
- Test 데이터에도 48건 결측 존재 → 제출 불가

### 5.2. 해결 전략: 3단계 하이브리드 접근법

**Phase 1: 중복 제거**

```python
# 효율화: 87만 건 → 8,200개 고유 주소로 압축
unique_addresses = df.groupby(['시군구', '번지']).size()
# 결과: 8,200개만 API 호출
```

**Phase 2: API 순차 호출**

1. **Vworld (국토부)** - 최우선 (지적도 기반, 정확도 최고)
2. **Naver Maps** - 2순위 (상업 POI 강점)
3. **Kakao Local** - 3순위 (주소 검색 보완)

**성공률**: 8,200개 중 8,072개 (98.4%) 성공

**Phase 3: Spatial Median Imputation**

API 실패 128개 + 이상치 3,137개 = **총 3,265건 보간 필요**

**가설**:
> "같은 법정동(Dong)의 아파트들은 지리적으로 매우 인접해 있다.  
> 따라서 해당 동의 정상 데이터 중앙값(Median)으로 대체해도 오차는 수백 미터 이내"

**구현 로직**:

```python
# 1. 서울 범위 밖 좌표를 NaN 처리
seoul_mask = (df['좌표X'] < 126.7) | (df['좌표X'] > 127.3) | \
             (df['좌표Y'] < 37.4) | (df['좌표Y'] > 37.7)
df.loc[seoul_mask, ['좌표X', '좌표Y']] = np.nan

# 2. 동(Dong)별 중앙값 계산
medians = df.groupby(['Gu', 'Dong'])[['좌표X', '좌표Y']].median()

# 3. 결측치 매핑
df['좌표X'] = df['좌표X'].fillna(df['Dong'].map(medians['좌표X']))
df['좌표Y'] = df['좌표Y'].fillna(df['Dong'].map(medians['좌표Y']))
```

### 5.3. 검증 결과

| 항목 | Before | After |
|:---|--:|--:|
| 좌표 결측치 | 869,670건 | **0건** |
| 서울 밖 이상치 | 3,137건 | **0건** |
| 복원율 | 22.3% | **100%** |

**시각적 검증**:

- Before: 전북 익산(35.99, 126.98) 등 전국 산재
- After: 모든 좌표가 서울시 범위 내 집중

---

## 6. 주차대수 결측치 처리 (Parking Imputation)

### 6.1. 결측치 현황

- Train: 약 330,000건 (29.5%)
- 주차대수는 아파트 편의성의 핵심 지표

### 6.2. 방법론 비교 (Method Comparison)

| 방법 | 장점 | 단점 | R² Score |
|:---|:---|:---|--:|
| **평균 대체** | 간단 | 단지 규모 무시 | 0.12 |
| **중앙값 대체** | 이상치 강건 | 단지 특성 무시 | 0.15 |
| **KNN** | 지역성 반영 | 연식, 면적 가중치 낮음 | 0.64 |
| **RandomForest** | 비선형 관계 포착 | 계산 비용 높음 | **0.86** |

### 6.3. 선택한 방법: RandomForest Regressor

**선택 근거**:

1. **높은 설명력**: R² 0.86 (분산의 86% 설명)
2. **변수 간 상호작용 포착**: 세대수 × 면적 등
3. **이상치 강건성**: Tree 기반 모델의 특성

**학습 변수**:

```python
features = ['전용면적(㎡)', '건축년도', '좌표X', '좌표Y', 'Gu_encoded', '전체세대수']
```

**구현 코드**:

```python
from sklearn.ensemble import RandomForestRegressor

# 학습 데이터: 주차대수가 존재하는 정상 데이터
train_clean = train[train['주차대수'].notnull()]

# 모델 학습
rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
rf.fit(train_clean[features], train_clean['주차대수'])

# 예측 및 대체
missing_mask = train['주차대수'].isnull()
train.loc[missing_mask, '주차대수'] = rf.predict(train.loc[missing_mask, features])
```

### 6.4. 검증

**Feature Importance**:

- `전용면적(㎡)`: 45.2%
- `전체세대수`: 28.7%
- `건축년도`: 12.3%
- `좌표X/Y`: 9.8%
- `Gu_encoded`: 4.0%

**예측 분포 비교**:

![Parking Impact](./eda_output/parking_info_price_impact.png)

**결과**:

- 원본 데이터와 예측 데이터의 분포 일치도 확인
- 평균 오차: 약 15대 (전체 평균 500대 대비 3%)

---

## 7. 범주형 변수 처리 (Categorical Features)

### 7.1. k-컬럼 분석

![k-단지분류](./eda_output/cat_k-단지분류(아파트,주상복합등등).png)

**유지 결정 변수**:

- `k-단지분류`: 아파트, 주상복합 등 구분
- `k-복도유형`: 복도식, 계단식
- `k-난방방식`: 개별, 지역, 중앙

![k-난방방식](./eda_output/cat_k-난방방식.png)

**인코딩 전략**:

- High Cardinality (Gu, Dong) → Target Encoding (K-Fold)
- Low Cardinality → Label Encoding

---

## 8. 취소 거래 제거 (Cancelled Transactions)

**컬럼**: `해제사유발생일`

**논리**:

- 해제된 거래는 실제 실거래가가 아님
- 학습 시 노이즈로 작용

![Cancelled Transaction Boxbox](./eda_output/cancelled_transactions_price_boxplot.png)

**조치**:

- Train: 해제 거래 전체 삭제
- Test: 해당 컬럼 자체가 없으므로 영향 없음

---

## 9. 최종 데이터 품질 (Final Quality Check)

| 지표 | Before | After |
|:---|--:|--:|
| 총 컬럼 수 | 48개 | 22개 |
| 결측치 | 87만 건 | **0건** |
| 이상치 | 3,137건 | **0건** |
| Train Shape | (1,119,088, 48) | (1,112,839, 22) |
| Test Shape | (267,627, 47) | (267,627, 22) |

**저장 파일**:

- `data/processed/train_final.csv`
- `data/processed/test_final.csv`

---

## 10. 결론 (Conclusion)

**주요 성과**:

1. 78% 좌표 결측 → **100% 복원** (API + Spatial Median)
2. 3,137건 좌표 이상치 → **0건** (서울 범위 검증)
3. 30% 주차대수 결측 → **ML 기반 정밀 예측** (R² 0.86)
4. 26개 불필요 컬럼 제거 → **모델 효율성 향상**

**재현 가능성**:

- 모든 단계가 스크립트로 자동화됨
- `preprocessing_pipeline.py` 실행 시 동일 결과 보장

**다음 단계**:

- 파생변수 생성 (교통 접근성, 건물 나이 등)
- 모델링 및 하이퍼파라미터 튜닝
