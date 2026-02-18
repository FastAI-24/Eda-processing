
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KDTree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------------------
# Exp07: Quality Features (아파트 단지 품질 및 스펙 기반 정교화)
# --------------------------------------------------------------------------------
# 이 스크립트는 단순한 절대 수치(예: 총 주차대수)를 상대적인 품질 지표(예: 세대당 주차대수)
# 로 변환하여 모델이 아파트 단지의 '실제 품질'을 학습하도록 유도합니다.
# 또한 '아파트명'과 같은 과적합 위험 변수를 제거하여 일반화 성능을 극대화합니다.
# --------------------------------------------------------------------------------

# 시각화 설정 (한글 폰트)
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
input_dir = 'data/processed'
raw_dir = 'data/raw'
submission_dir = 'submissions'

if not os.path.exists(submission_dir):
    os.makedirs(submission_dir)

train_path = os.path.join(input_dir, 'train_final.csv')
test_path = os.path.join(input_dir, 'test_final.csv')

# --------------------------------------------------------------------------------
# Helper Functions & Quality Feature Engineering
# --------------------------------------------------------------------------------

def minimal_clean_cols(df):
    """LGBM 호환을 위해 컬럼명의 특수문자를 정리합니다."""
    new_cols = []
    for col in df.columns:
        c = re.sub(r'[(),\[\]\s]', '_', col)
        new_cols.append(c)
    df.columns = new_cols
    return df

def feature_engineering_quality(train, test):
    """단지의 품질을 나타내는 상대 지표 생성 및 노이즈 변수 제거"""
    print("\n[Step 2] Quality Feature Engineering Start...")
    
    # 처리를 위해 Train/Test 결합
    train_len = len(train)
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # 1. 시계열 기본 변수 (계약년월 분해 및 건물 연식)
    if '계약년월' in df.columns:
        df['contract_year'] = df['계약년월'].astype(str).str[:4].astype(int)
        df['contract_month'] = df['계약년월'].astype(str).str[4:6].astype(int)
        
    if '건축년도' in df.columns:
        df['building_age'] = df['contract_year'] - df['건축년도']
        df['building_age'] = df['building_age'].clip(lower=0)

    # 2. 품질 지표 생성 (Quality Metrics - Ratio)
    # 0으로 나누기 방지를 위한 보정
    df['k-전체세대수'] = df['k-전체세대수'].replace(0, np.nan)
    df['k-전체동수'] = df['k-전체동수'].replace(0, np.nan)
    
    print("  - Calculating Quality Ratios (Parking, Area, Density)...")
    # A. 세대당 주차대수: 주차 공간의 여유로운 정도는 핵심적인 품질 지표임
    if '주차대수' in df.columns:
        df['parking_per_unit'] = df['주차대수'] / df['k-전체세대수']
    
    # B. 평균 가구 면적: 단지가 전반적으로 대형 평수 위주인지 파악 (부촌 여부)
    if 'k-연면적' in df.columns:
        df['unit_area_avg'] = df['k-연면적'] / df['k-전체세대수']
        
    # C. 단지 밀도: 동당 세대수가 적을수록 쾌적한 환경으로 판단
    if 'k-전체동수' in df.columns:
        df['complex_density'] = df['k-전체세대수'] / df['k-전체동수']

    # 3. 로그 변환 (로그 스케일링을 통한 정규화)
    # 절대 수치가 큰 변수들이 모델에 과도한 영향을 주지 않도록 변환
    if 'k-전체세대수' in df.columns:
        df['log_total_units'] = np.log1p(df['k-전체세대수'])
    if 'k-연면적' in df.columns:
        df['log_total_area'] = np.log1p(df['k-연면적'])

    # 4. 교통 관련 변수 통합 (지하철/버스)
    print("  - Adding Transport Features via KDTree...")
    sub_path = os.path.join(raw_dir, 'subway_feature.csv')
    bus_path = os.path.join(raw_dir, 'bus_feature.csv')
    
    try:
        sub_df = pd.read_csv(sub_path)
        tree_sub = KDTree(sub_df[['위도', '경도']].values, metric='euclidean')
        coords = df[['좌표Y', '좌표X']].values
        dist, idx = tree_sub.query(coords, k=1)
        df['dist_to_subway'] = dist * 100
        line_col = '호선' if '호선' in sub_df.columns else ('노선명' if '노선명' in sub_df.columns else sub_df.columns[2])
        df['subway_line'] = sub_df.iloc[idx.flatten()][line_col].values
    except Exception as e:
        print(f"  >> Subway info error (skipped): {e}")

    try:
        bus_df = pd.read_csv(bus_path) 
        if 'Y' in bus_df.columns and 'X' in bus_df.columns:
            tree_bus = KDTree(bus_df[['Y', 'X']].values, metric='euclidean')
            dist, _ = tree_bus.query(df[['좌표Y', '좌표X']].values, k=1)
            df['dist_to_bus'] = dist * 100
            df['bus_count_500'] = tree_bus.query_radius(df[['좌표Y', '좌표X']].values, r=0.005, count_only=True)
    except Exception as e:
        print(f"  >> Bus info error (skipped): {e}")

    # 5. 변수 다이어트 (Feature Diet - 과적합 및 노이즈 변수 제거)
    # 아파트명, 지번, 계약일 등 모델이 '이름'을 외우게 만드는 변수들을 과감히 삭제합니다.
    print("  - Feature Diet: Dropping noisy and high-cardinality columns...")
    drop_cols = ['주차대수', 'k-전체세대수', 'k-연면적', 'k-전체동수', '아파트명', '계약일', '계약년월', 
                 '시군구', '도로명', '번지', '본번', '부번']
    
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')
    
    # Train/Test 다시 분리
    train = df.iloc[:train_len].copy()
    test = df.iloc[train_len:].copy()
    
    return train, test

# --------------------------------------------------------------------------------
# Main Training Function
# --------------------------------------------------------------------------------

def train_exp07_quality():
    print("========== [Exp07] Quality Features & Feature Diet Start ==========")
    
    # 1. 데이터 로드 및 품질 변수 생성
    print("\n[Step 1] Loading and Engineering Quality Data...")
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    
    train, test = feature_engineering_quality(train, test)
    
    # 2. 최신 데이터(2017+) 필터링 전략 유지 (Exp06)
    print("\n[Step 2] Filtering Recent Data (2017~2023)...")
    initial_len = len(train)
    train = train[train['contract_year'] >= 2017].reset_index(drop=True)
    print(f"Data Filtered: {initial_len} -> {len(train)} (Remaining: {len(train)/initial_len:.1%})")
    
    # 3. 전처리 (Cleansing & Log Transform)
    print("\n[Step 3] Preprocessing...")
    train = minimal_clean_cols(train)
    test = minimal_clean_cols(test)
    
    # Target 로그 변환
    if train['target'].max() > 100:
        print(">> Applying Log Transform to Target...")
        train['target'] = np.log1p(train['target'])

    # 범주형 변수 인코딩
    print("\n[Step 4] Label Encoding Categorical Features...")
    le = LabelEncoder()
    obj_cols = train.select_dtypes(include=['object']).columns
    for col in obj_cols:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    # 4. 검증 전략 (Shuffle K-Fold)
    print("\n[Step 5] Validation Strategy: Shuffle K-Fold (5 Splits)")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    features = [c for c in train.columns if c != 'target']
    
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
    
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    # Cross Validation Loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(train[features], train['target'])):
        X_tr, X_val = train.iloc[train_idx][features], train.iloc[val_idx][features]
        y_tr, y_val = train.iloc[train_idx]['target'], train.iloc[val_idx]['target']
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=10000,
            valid_sets=[dtrain, dvalid],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
        )
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(test[features]) / 5
        
        rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_pred)))
        print(f"Fold {fold+1} RMSE: {rmse:,.0f}")
        
    # 최종 성능 확인
    total_rmse = np.sqrt(mean_squared_error(np.expm1(train['target']), np.expm1(oof_preds)))
    print(f"\n>> Final Exp07 CV RMSE: {total_rmse:,.0f}")
    
    # 5. 변수 중요도 확인
    imp = pd.DataFrame({'Feature': features, 'Gain': model.feature_importance('gain')}).sort_values('Gain', ascending=False)
    print("\n[Top 10 Features (Gain)]")
    print(imp.head(10))
    
    # 6. 제출 파일 생성
    sub_path = os.path.join(submission_dir, 'submission_exp07_quality.csv')
    pd.DataFrame({'target': np.expm1(test_preds)}).to_csv(sub_path, index=False)
    print(f"\n[Step 7] Submission file saved: {sub_path}")

if __name__ == "__main__":
    train_exp07_quality()
