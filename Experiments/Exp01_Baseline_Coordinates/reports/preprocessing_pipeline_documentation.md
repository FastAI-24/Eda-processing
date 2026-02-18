# 아파트 실거래가 예측 프로젝트: 전처리 파이프라인 (Data Pipeline Documentation)

**작성일**: 2026-02-15
**목표**: 서울시 아파트 실거래가(Target) 예측 (RMSE 최소화)
**현재 최고 성적**: CV RMSE 5,601 / Public LB 16,112

---

## 1. 데이터 전처리 히스토리 (Preprocessing History)

이 프로젝트는 Raw Data의 품질 문제(결측치, 이상치, 불필요 변수)를 단계적으로 해결하고, 변수 특성에 맞는 인코딩 방식을 적용하는 **정공법**으로 진행되었습니다.

### **Step 1: 데이터 정제 (Row/Column Cleaning)**

- **스크립트**: `preprocess_step1_cancellation.py`
- **논리**:
    1. **해제사유발생일(Cancellation Date)**: 거래가 취소된 건은 '실거래가'가 아니므로 학습에 노이즈를 줌.
        - **Action**: `해제사유발생일`이 존재하는 행 **전체 삭제**. (단, Test 데이터는 삭제 불가하므로 원본 유지)

### **Step 2: 불필요 컬럼 삭제 (Feature Selection)**

- **스크립트**: `preprocess_step2_columns.py`
- **논리**: 데이터의 50% 이상이 결측이거나, 가격 결정과 무관한 메타데이터 삭제.
  - **삭제 대상 (26개)**:
    - `거래유형`, `중개사소재지`: 97%가 결측('-').
    - `k-전화번호`, `k-팩스번호`: 가격과 무관.
    - `등기신청일자`: 거래 후 미래의 정보(Leakage).
    - `k-시행사`: 결측이 많고 중요도 낮음 (시공사는 보존하여 파생변수화).
    - 기타 관리비/단지신청일 관련 정보 삭제.

### **Step 3: 좌표 무결성 확보 (Geocoding & Correction)**

- **스크립트**: `impute_coordinates_api.py`, `correct_coordinate_outliers.py`
- **문제**: 전체 데이터의 78%가 좌표(`좌표X/Y`) 결측.
- **해결 전략**:
    1. **API 활용**: Vworld → Naver → Kakao 순차 호출로 8,200개 유니크 주소 지오코딩 완료.
    2. **이상치 교정**:
        - 문제: 일부 좌표가 서울 범위를 벗어남 (전북 익산 등 잘못 매핑).
        - Action: 서울(`126.7~127.3`, `37.4~37.7`) 벗어나는 좌표는 `NaN` 처리 후, **같은 동(Dong)의 정상 데이터 중앙값**으로 강제 보정.
    3. **잔여 결측**: Test 데이터의 48개 결측도 **동 중앙값**으로 완벽 채움.
                    (test.csv : 결측치 0건, 정확도 100%)

### **Step 4: 특성 공학 및 결측치 복원 (Feature Eng & Imputation)**

- **스크립트**: `final_step_preprocessing.py`
- **핵심 로직**:
    1. **파생변수 생성**:
        - `days_since`: 시계열 반영 (2007-01-01 기준 경과일 수).
        - `is_top_brand`: 아파트명에서 10대 브랜드(래미안, 자이, 힐스테이트 등) 추출 (0/1).
    2. **주차대수 복원 (Model-based Imputation)**:
        - `주차대수` 결측치를 평균/중앙값으로 채우면 왜곡 심함.
        - **RandomForest Regressor** 학습 (`전용면적`, `건축년도`, `좌표`, `구` -> `주차대수` 예측).
        - R2 Score 0.86 수준으로 정밀 복원 완료.
    3. **방어적 프로그래밍 (Defensive One)**:
        - `전용면적 <= 0`, `건축년도 < 1900` 등 물리적 오류값 사전 차단.

---

## 5. 변수별 전처리 상세 (Detailed Feature Processing)

### **A. 연속형 변수 (Continuous Variables)**

| 변수명 | 처리 방식 | 논리 |
|:---:|:---|:---|
| **Target (실거래가)** | **Log Transform (`np.log1p`)** | 가격 분포가 오른쪽으로 긴 꼬리(Skewed) 형태이므로, 로그 변환을 통해 정규분포에 가깝게 만들어 회귀 성능 향상. |
| **전용면적 (Area)** | **Log Transform (`np.log1p`)** | 면적 역시 소형~초대형 간 격차가 크므로 로그 변환 적용. |
| **주차대수** | **Imputation (RandomForest)** | 단순 평균 대신, 단지 특성(면적, 연식, 위치)을 고려한 머신러닝 예측값으로 채움. |
| **좌표 (X, Y)** | **Correction & Normalization** | 이상치 교정 완료. 모델에는 그대로 입력되거나, 추후 클러스터링/거리 계산에 활용됨. |
| **건축년도/계약년월** | **Feature Generation** | `계약년도 - 건축년도` = `건물 나이(Building Age)` 변수 생성 예정 (구축/신축 구분). (V2 반영 예정) |

### **B. 범주형 변수 (Categorical Variables)**

| 변수명 | 인코딩 방식 | 논리 |
|:---:|:---:|:---|
| **구(Gu), 동(Dong)** | **Target Encoding (K-Fold)** | 범주 개수가 너무 많음(High Cardinality). 원-핫 인코딩 시 차원이 폭발하므로, **"해당 지역의 평균 가격"** 정보를 수치화하여 입력. 과적합 방지를 위해 K-Fold 적용. |
| **k-단지분류** | **Label Encoding** | 아파트, 주상복합 등 범주 개수가 적고 명확하여 정수(0, 1, 2...)로 변환. |
| **k-복도유형** | **Label Encoding** | 복도식/계단식 등 주거 형태를 구분하는 단순 범주이므로 레이블 인코딩 적용. |
| **k-난방방식** | **Label Encoding** | 개별난방/지역난방 등 단순 범주. |
| **브랜드(Top Brand)** | **Binary Encoding (0/1)** | 10대 브랜드 포함 여부만 중요하므로 1(포함) 또는 0(미포함)으로 변환. |

---

## 3. 현재 성적표 (Performance)

| 단계 | 모델 | Validation RMSE | Leaderboard RMSE | 비고 |
|:---:|:---:|:---:|:---:|:---|
| Step 4 | Tuned LGBM | **5,601** (Real) | **16,112** | 지하철/버스 데이터 미포함, Shuffle K-Fold 적용 |

**분석**:

- 내부 데이터만으로는 한계 도달 (CV 5,600 선).
- 리더보드 점수와의 갭(Gap)을 줄이려면 **'역세권' 같은 외부 입지 정보**가 필수적임.

---

## 4. Next Step (To-Do)

1. **교통 파생변수 추가 (최우선)**:
    - `data/raw/subway_feature.csv`, `bus_feature.csv` 활용.
    - `dist_to_subway` (최단 지하철 거리).
    - `subway_count_1km` (역세권 밀도).
2. **건물 나이(Building Age) 추가**:
    - `계약년도 - 건축년도`로 물리적 노후도 명시화.
3. **앙상블 (Ensemble)**:
    - XGBoost, CatBoost 모델 추가 학습 후 `LGBM`과 평균(Weighted Blending).

---
*이 문서는 프로젝트 진행에 따라 지속적으로 업데이트됩니다.*
