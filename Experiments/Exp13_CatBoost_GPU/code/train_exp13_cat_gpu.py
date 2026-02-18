
import pandas as pd
import numpy as np
import os
import re
import datetime
import shutil
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

# =============================================================================
# [Exp13] CatBoost Revolution (GPU Accelerated + Group K-Fold)
# 목표:
#   1. CatBoost를 도입하여 범주형 변수의 잠재력을 폭발시킨다. (1등과의 격차 해소)
#   2. Group K-Fold (아파트 기준)를 통해 진짜 실력(CV)을 검증한다.
#   3. GPU 가속을 활용하여 학습 시간을 단축한다.
# =============================================================================

# 1. 하버사인 거리 계산 함수
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# 2. 경로 및 폴더 설정
EXP_NAME = "Exp13_CatBoost_GPU"
BASE_DIR = 'experiments'
TARGET_DIR = os.path.join(BASE_DIR, EXP_NAME)
for sub in ['code', 'results', 'data']:
    path = os.path.join(TARGET_DIR, sub)
    if not os.path.exists(path): os.makedirs(path)

input_dir = 'data/processed'
raw_dir = 'data/raw'
train_path = os.path.join(input_dir, 'train_final.csv')
test_path = os.path.join(input_dir, 'test_final.csv')

def train_exp13_catboost():
    print(f"[{datetime.datetime.now()}] Loading Data for Exp13...")
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    
    # 아파트명 미리 백업 (GroupKFold용)
    train_apartments = train['아파트명'].astype(str)
    
    # 2017년 이후 데이터 필터링
    temp_year = train['계약년월'].astype(str).str[:4].astype(int)
    train = train[temp_year >= 2017].reset_index(drop=True)
    train_apartments = train_apartments[temp_year >= 2017].reset_index(drop=True)
    
    print("\n[Step 1] Feature Engineering (Exp12-v2 Set + CatBoost Optimization)")
    train_len = len(train)
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # 피처 엔지니어링 (Exp12-v2 복원)
    df['contract_year'] = df['계약년월'].astype(str).str[:4].astype(int)
    df['contract_month'] = df['계약년월'].astype(str).str[4:6].astype(int)
    df['building_age'] = (df['contract_year'] - df['건축년도']).clip(lower=0)
    
    # 품질 및 공간 정보
    df['parking_per_unit'] = df['주차대수'] / df['k-전체세대수'].replace(0, np.nan)
    df['dist_gbd'] = haversine_distance(df['좌표Y'], df['좌표X'], 37.496, 127.027)
    
    # 교통 (Subway/Bus)
    try:
        sub_df = pd.read_csv(os.path.join(raw_dir, 'subway_feature.csv'))
        sub_tree = KDTree(sub_df[['위도', '경도']].values)
        dist, _ = sub_tree.query(df[['좌표Y', '좌표X']].values, k=1)
        df['dist_to_subway'] = dist * 100
    except: pass

    # 공간 클러스터링 (CatBoost용 범주형으로 활용 가능)
    coords = df[['좌표Y', '좌표X']].values
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=150, random_state=42, n_init=10)
    df['coord_cluster'] = kmeans.fit_predict(scaler.fit_transform(coords))

    # [핵심] CatBoost를 위해 범주형 변수를 자동으로 식별합니다.
    # numeric 이 아닌 모든 컬럼을 cat_features로 지정
    drop_cols = ['주차대수', 'k-전체세대수', 'k-연면적', 'k-전체동수', '아파트명', '계약일', '계약년월', 
                 '시군구', '도로명', '번지', '본번', '부번', 'k-건설사(시공사)'] # 건설사는 너무 종류가 많아 일단 제외 권장 (필요시 포함)
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

    cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # coord_cluster도 범주형으로 취급
    if 'coord_cluster' in df.columns:
        cat_features.append('coord_cluster')
    
    print(f"Cat Features: {cat_features}")
    
    train = df.iloc[:train_len].copy()
    test = df.iloc[train_len:].copy()
    
    # Target Log
    train['target'] = np.log1p(train['target'])
    
    # [Step 2] Training with GroupKFold
    print(f"\n[Step 2] Training on GPU with GroupKFold (by Apartment Name)...")
    gkf = GroupKFold(n_splits=5)
    features = [c for c in train.columns if c != 'target']
    
    params = {
        'iterations': 10000,
        'learning_rate': 0.05,
        'depth': 8,
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'task_type': 'GPU',  # GPU 사용 설정
        'devices': '0',      # 첫 번째 GPU 사용
        'early_stopping_rounds': 200,
        'verbose': 500
    }
    
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(train[features], train['target'], groups=train_apartments)):
        X_tr, X_val = train.iloc[tr_idx][features], train.iloc[val_idx][features]
        y_tr, y_val = train.iloc[tr_idx]['target'], train.iloc[val_idx]['target']
        
        # 범주형 변수 처리 (CatBoost 전용)
        for col in cat_features:
            X_tr[col] = X_tr[col].fillna('NAN').astype(str)
            X_val[col] = X_val[col].fillna('NAN').astype(str)
            test[col] = test[col].fillna('NAN').astype(str)

        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool)
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(test[features]) / 5
        
        rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_pred)))
        print(f"Fold {fold+1} Ground-Truth RMSE: {rmse:,.0f}")
        
    total_rmse = np.sqrt(mean_squared_error(np.expm1(train['target']), np.expm1(oof_preds)))
    print(f"\n>> Overall Exp13 CatBoost CV RMSE (GroupKFold): {total_rmse:,.0f}")
    
    # [Step 3] Finalize & Save
    sub_path = os.path.join(TARGET_DIR, 'results', 'submission_exp13_cat_int.csv')
    pd.DataFrame({'target': np.expm1(test_preds).astype(int)}).to_csv(sub_path, index=False)
    
    # Copy code for archive
    current_script = 'train_exp13_cat_gpu.py' # 이 이름으로 저장 예정
    if os.path.exists(current_script):
        shutil.copy2(current_script, os.path.join(TARGET_DIR, 'code'))
        
    # Create Report
    report_path = os.path.join(TARGET_DIR, 'summary_exp13.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Experiment 13: CatBoost Revolutions (GPU)\n")
        f.write(f"- CV RMSE (GroupKFold): {total_rmse:,.0f}\n")
        f.write("- Model: CatBoost (GPU Accelerated)\n")
        f.write("- Strategy: 범주형 변수 직접 사용 + 아파트 단지별 그룹 검증\n")
        
    print(f"\nExperiment Finalized: {TARGET_DIR}")

if __name__ == "__main__":
    train_exp13_catboost()
