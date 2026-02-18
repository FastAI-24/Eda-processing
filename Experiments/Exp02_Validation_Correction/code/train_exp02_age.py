
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------------------
# Exp02: Validation Strategy Correction (검증 전략 수정 및 Adversarial Weighting)
# --------------------------------------------------------------------------------
# 이 스크립트는 Exp01의 과적합 문제를 해결하기 위해 다음을 수행합니다.
# 1. Adversarial Validation 가중치 적용 (Test 데이터와 유사한 Train 데이터 강조)
# 2. 시계열 검증 도입 (Last 5% Hold-out) - 과거 데이터를 보고 미래를 예측하는 구조 반영
# 3. 추가 변수 테스트 (건물 연식 - Building Age)
# --------------------------------------------------------------------------------

# 시각화 설정 (한글 폰트)
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
input_dir = 'data/processed'
output_dir = 'logs'
submission_dir = 'submissions'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(submission_dir):
    os.makedirs(submission_dir)

train_path = os.path.join(input_dir, 'train_final_adv_weighted.csv')
test_path = os.path.join(input_dir, 'test_final.csv')

# --------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------

def minimal_clean_cols(df):
    """LGBM 호환을 위해 컬럼명의 특수문자를 정리합니다."""
    new_cols = []
    for col in df.columns:
        c = re.sub(r'[(),\[\]\s]', '_', col)
        new_cols.append(c)
    df.columns = new_cols
    return df

# --------------------------------------------------------------------------------
# Main Training Function
# --------------------------------------------------------------------------------

def train_with_building_age():
    print("========== [Exp02] Validation Strategy Correction ==========")
    
    # 1. 데이터 로드 (Adversarial Weight가 포함된 데이터셋 사용)
    print("\n[Step 1] Loading Weighted Data...")
    try:
        train = pd.read_csv(train_path, low_memory=False)
        test = pd.read_csv(test_path, low_memory=False)
        print(f"Train Size: {train.shape}, Test Size: {test.shape}")
    except FileNotFoundError:
        print(f"Error: {train_path} not found. Please run adversarial validation script first.")
        return

    # 2. 파생 변수 추가: 건물 연식 (Building Age)
    print("\n[Step 2] Adding Feature: Building Age...")
    
    # '계약년월'과 '건축년도'를 이용해 연식 계산
    if '건축년도' in train.columns and '계약년월' in train.columns:
        # 계약년도 추출 (YYYYMM -> YYYY)
        train['contract_year'] = train['계약년월'].astype(str).str[:4].astype(int)
        test['contract_year'] = test['계약년월'].astype(str).str[:4].astype(int)
        
        # 연식 = 계약년도 - 건축년도
        train['building_age'] = train['contract_year'] - train['건축년도']
        test['building_age'] = test['contract_year'] - test['건축년도']
        
        # 재건축, 선분양 등으로 인해 음수가 나올 경우 0으로 처리 (Clipping)
        train['building_age'] = train['building_age'].clip(lower=0)
        test['building_age'] = test['building_age'].clip(lower=0)
        print(">> Created 'building_age' feature successfully.")
    else:
        print(">> Warning: '건축년도' or '계약년월' missing. Skipping Age Feature.")

    # 3. 전처리 (Preprocessing)
    print("\n[Step 3] Preprocessing (Log Transform & Cleaning)...")
    
    # 컬럼명 정리
    train = minimal_clean_cols(train)
    test = minimal_clean_cols(test)
    
    # 로그 변환 (Target) - 왜도(Skewness) 완화
    if train['target'].max() > 100: # 값이 100보다 크면 원본 가격이라고 가정
        print(">> Applying Log Transform to Target...")
        train['target'] = np.log1p(train['target'])

    # 로그 변환 (면적)
    if '전용면적_㎡_' in train.columns:
         if train['전용면적_㎡_'].max() > 100:
             print(">> Applying Log Transform to Area...")
             train['전용면적_㎡_'] = np.log1p(train['전용면적_㎡_'])
             test['전용면적_㎡_'] = np.log1p(test['전용면적_㎡_'])
    
    # 4. 불필요한 텍스트 컬럼 제거
    print("\n[Step 4] Dropping Explicit Text Columns...")
    drop_texts = ['아파트명', '시군구', '도로명', '번지', '본번', '부번', '계약년월', '계약일']
    drop_texts_cleaned = [re.sub(r'[(),\[\]\s]', '_', c) for c in drop_texts]
    
    train.drop(columns=[c for c in drop_texts_cleaned if c in train.columns], inplace=True)
    test.drop(columns=[c for c in drop_texts_cleaned if c in test.columns], inplace=True)
    
    # 5. 인코딩 (Label Encoding)
    # Exp01의 Target Encoding 대신, Exp02에서는 보다 보수적인 Label Encoding 위주로 진행
    print("\n[Step 5] Label Encoding Categorical Features...")
    le = LabelEncoder()
    obj_cols = train.select_dtypes(include=['object']).columns
    print(f"Object columns to encode: {obj_cols.tolist()}")
    
    for col in obj_cols:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    # 6. 검증 전략: Last 5% Hold-out (Time-based Split)
    # Exp01의 실패 원인인 Shuffle Split을 대체하여, 시간에 따른 데이터 분포 변화를 반영
    print("\n[Step 6] Validation Strategy: Last 5% Hold-out (Time Series)")
    
    if 'days_since' in train.columns:
        train = train.sort_values('days_since') # 시간순 정렬 필수
        
        # 전체 데이터의 마지막 5%를 검증 셋으로 사용 (약 6개월~1년 분량 예상)
        valid_size = int(len(train) * 0.05)
        train_set = train.iloc[:-valid_size]
        valid_set = train.iloc[-valid_size:]
        
        print(f"Train Size: {len(train_set)} (Past Data)")
        print(f"Valid Size: {len(valid_set)} (Future Data - approx last 5%)")
    else:
        print("Error: 'days_since' column missing. Cannot perform Time-based Split.")
        return

    # 7. 모델 학습 (Weighted Training)
    print("\n[Step 7] Final Training with Adversarial Weights...")
    
    # 학습에 사용할 Feature 및 Target, Weight 컬럼 지정
    features = [c for c in train.columns if c not in ['target', 'adversarial_weight']]
    target_col = 'target'
    weight_col = 'adversarial_weight' # Adversarial Validation을 통해 계산된 가중치
    
    X_train = train_set[features]
    y_train = train_set[target_col]
    
    if weight_col in train_set.columns:
        print(">> Using Adversarial Weights for Training.")
        w_train = train_set[weight_col]
    else:
        print(">> Warning: Weights not found. Training without weights.")
        w_train = None
    
    X_valid = valid_set[features]
    y_valid = valid_set[target_col]
    # Validation에서는 Weight를 사용하지 않음 (실제 성능 평가 목적)
    
    dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.03,
        'num_leaves': 256,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'n_jobs': -1,
        'seed': 42,
        'verbose': -1
    }
    
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        valid_sets=[dtrain, dvalid],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
    )
    
    # 8. 평가 (Evaluation)
    print("\n[Step 8] Evaluation...")
    valid_pred = model.predict(X_valid)
    
    # RMSE 계산 (원래 스케일로 변환)
    rmse = np.sqrt(mean_squared_error(np.expm1(y_valid), np.expm1(valid_pred)))
    print(f"\n>> Final Validation RMSE (Real Scale): {rmse:,.0f}")
    
    # Feature Importance 출력
    imp = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importance(importance_type='gain')
    }).sort_values('Importance', ascending=False)
    
    print("\n[Top 10 Important Features]")
    print(imp.head(10))
    
    # 9. 제출 (Submission)
    print("\n[Step 9] Generating Submission...")
    test_pred = np.expm1(model.predict(test[features]))
    sub = pd.DataFrame({'target': test_pred})
    sub_path = os.path.join(submission_dir, 'submission_exp02_age.csv')
    sub.to_csv(sub_path, index=False)
    print(f"Submission Saved to {sub_path}")

if __name__ == "__main__":
    train_with_building_age()
