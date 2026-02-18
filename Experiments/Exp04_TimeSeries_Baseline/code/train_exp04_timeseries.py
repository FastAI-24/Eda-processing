
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------------------
# Exp04: True Baseline - TimeSeriesSplit (시계열 베이스라인 구축)
# --------------------------------------------------------------------------------
# 이 스크립트는 시계열 데이터의 특성을 고려하여 과거 데이터로 미래를 예측하는 
# TimeSeriesSplit 검증 전략을 수립합니다. 데이터 누수(Leakage)를 방지하고 
# 실제 리더보드 점수와 연동되는 신뢰할 수 있는 베이스라인을 구축하는 것이 목표입니다.
# --------------------------------------------------------------------------------

# 시각화 설정 (한글 폰트)
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
input_dir = 'data/processed'
submission_dir = 'submissions'

if not os.path.exists(submission_dir):
    os.makedirs(submission_dir)

train_path = os.path.join(input_dir, 'train_final.csv')
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

def train_exp04_timeseries():
    print("========== [Exp04] TimeSeries Baseline Training Start ==========")
    
    # 1. 데이터 로드
    print("\n[Step 1] Loading Data...")
    try:
        train = pd.read_csv(train_path, low_memory=False)
        test = pd.read_csv(test_path, low_memory=False)
        print(f"Original Train Shape: {train.shape}")
    except FileNotFoundError:
        print(f"Error: {train_path} not found.")
        return

    # 2. 파생 변수 생성: 건물 연식 (Exp03 변수 재활용)
    print("\n[Step 2] Adding Feature: Building Age...")
    if '건축년도' in train.columns and '계약년월' in train.columns:
        train['contract_year'] = train['계약년월'].astype(str).str[:4].astype(int)
        test['contract_year'] = test['계약년월'].astype(str).str[:4].astype(int)
        
        train['building_age'] = train['contract_year'] - train['건축년도']
        test['building_age'] = test['contract_year'] - test['건축년도']
        
        train['building_age'] = train['building_age'].clip(lower=0)
        test['building_age'] = test['building_age'].clip(lower=0)
        print(">> Created 'building_age' feature successfully.")
    else:
        print(">> Warning: Required columns missing for Age feature.")

    # 3. 전처리 (Preprocessing)
    print("\n[Step 3] Preprocessing (Cleansing & Log Transform)...")
    train = minimal_clean_cols(train)
    test = minimal_clean_cols(test)
    
    # Target 로그 변환
    if train['target'].max() > 100:
        print(">> Applying Log Transform to Target...")
        train['target'] = np.log1p(train['target'])

    # 면적 로그 변환
    area_col = [c for c in train.columns if '전용면적' in c]
    if area_col:
        col_name = area_col[0]
        if train[col_name].max() > 100:
            print(f">> Applying Log Transform to {col_name}...")
            train[col_name] = np.log1p(train[col_name])
            test[col_name] = np.log1p(test[col_name])
    
    # 4. 불필요한 텍스트 컬럼 제거
    print("\n[Step 4] Dropping Text Columns...")
    drop_texts = ['아파트명', '시군구', '도로명', '번지', '본번', '부번', '계약년월', '계약일']
    drop_texts_cleaned = [re.sub(r'[(),\[\]\s]', '_', c) for c in drop_texts]
    train.drop(columns=[c for c in drop_texts_cleaned if c in train.columns], inplace=True)
    test.drop(columns=[c for c in drop_texts_cleaned if c in test.columns], inplace=True)
    
    # 5. 인코딩 (Label Encoding)
    print("\n[Step 5] Label Encoding...")
    le = LabelEncoder()
    obj_cols = train.select_dtypes(include=['object']).columns
    for col in obj_cols:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    # 6. 검증 전략 구성: TimeSeriesSplit (5 Splits)
    # 시계열 기반의 성능 측정을 위해 데이터를 시간순(days_since)으로 정렬합니다.
    print("\n[Step 6] Validation Strategy: TimeSeriesSplit (5 Splits)")
    
    if 'days_since' not in train.columns:
        print("Error: 'days_since' column missing. Required for sorting.")
        return
        
    # 시간순 정렬
    train = train.sort_values('days_since').reset_index(drop=True)
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    features = [c for c in train.columns if c != 'target']
    X = train[features]
    y = train['target']
    
    # 하이퍼파라미터 설정
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
    
    rmse_scores = []
    test_preds = np.zeros(len(test))

    # Cross Validation Loop (Expanding Window 방식)
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=10000,
            valid_sets=[dtrain, dvalid],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
        )
        
        # 검증 평가
        val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_pred)))
        rmse_scores.append(rmse)
        print(f"Fold {fold+1} (Train Data: {len(train_idx)}, Valid Data: {len(val_idx)}) RMSE: {rmse:,.0f}")
        
        # Test 데이터 예측 누적 (가중치 없이 단순 평균 사용)
        test_preds += model.predict(test[features]) / 5
        
    # 최종 평균 성능
    avg_rmse = np.mean(rmse_scores)
    print(f"\n>> Final Average TimeSeries RMSE: {avg_rmse:,.0f}")
    
    # 7. 제출 파일 저장
    sub_path = os.path.join(submission_dir, 'submission_exp04_timeseries.csv')
    pd.DataFrame({'target': np.expm1(test_preds)}).to_csv(sub_path, index=False)
    print(f"\n[Step 7] Submission file saved: {sub_path}")

if __name__ == "__main__":
    train_exp04_timeseries()
