
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import re
from sklearn.neighbors import KDTree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import datetime

# =============================================================================
# [Exp17] The Branding - Fixed Columns Strategy
# =============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def minimal_clean_cols(df):
    new_cols = []
    for col in df.columns:
        # 특수문자 및 공백 제거
        c = re.sub(r'[(),\[\]\s\-]', '_', col)
        new_cols.append(c)
    df.columns = new_cols
    return df

def feature_engineering_exp17_final(train, test):
    print("\n[Step 1] Feature Engineering (Job Centers + Quality)...")
    train_len = len(train)
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # 기초 날짜 피처 (이미 있을 수 있으나 확실히 재생성)
    df['contract_year'] = df['계약년월'].astype(str).str[:4].astype(int)
    df['contract_month'] = df['계약년월'].astype(str).str[4:6].astype(int)
    df['building_age'] = (df['contract_year'] - df['건축년도']).clip(lower=0)
    
    # Golden Triangle (Exp16)
    centers = {'gbd': (37.496, 127.027), 'cbd': (37.566, 126.978), 'ybd': (37.521, 126.924)}
    for name, pos in centers.items():
        if '좌표Y' in df.columns:
            df[f'dist_{name}'] = haversine_distance(df['좌표Y'], df['좌표X'], pos[0], pos[1])
    
    dist_cols = [f'dist_{n}' for n in centers.keys() if f'dist_{n}' in df.columns]
    if dist_cols:
        df['min_dist_to_job'] = df[dist_cols].min(axis=1)
    
    # Cluster & Quality
    if '주차대수' in df.columns and 'k-전체세대수' in df.columns:
        df['parking_per_unit'] = df['주차대수'] / df['k-전체세대수'].replace(0, np.nan)
    
    if '좌표Y' in df.columns:
        coords = df[['좌표Y', '좌표X']].values
        scaler = StandardScaler()
        df['coord_cluster'] = KMeans(n_clusters=150, random_state=42, n_init=10).fit_predict(scaler.fit_transform(coords))

    # Encoding Target Backups (존재하는 컬럼만)
    df['apt_name_tmp'] = df['아파트명'].astype(str) if '아파트명' in df.columns else 'Unknown'
    df['sigungu_tmp'] = df['시군구'].astype(str) if '시군구' in df.columns else 'Unknown'

    drop_cols = ['주차대수', 'k-전체세대수', 'k-연면적', 'k-전체동수', '아파트명', '도로명', '계약일', '계약년월', '번지', '본번', '부번', '좌표X', '좌표Y', '시군구']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    train = df.iloc[:train_len].copy()
    test = df.iloc[train_len:].copy()
    return train, test

def apply_target_encoding(train, test, cat_cols):
    print(f"[Step 2] OOF Target Encoding for: {cat_cols}")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for col in cat_cols:
        target_name = f'{col}_mean_price'
        train[target_name] = np.nan
        for tr_idx, val_idx in kf.split(train):
            means = train.iloc[tr_idx].groupby(col)['target'].mean()
            train.loc[val_idx, target_name] = train.iloc[val_idx][col].map(means)
        
        full_means = train.groupby(col)['target'].mean()
        test[target_name] = test[col].map(full_means)
        global_mean = train['target'].mean()
        train[target_name].fillna(global_mean, inplace=True)
        test[target_name].fillna(global_mean, inplace=True)
    
    # 보조 컬럼 제거
    train.drop(columns=cat_cols, inplace=True)
    test.drop(columns=cat_cols, inplace=True)
    return train, test

def train_exp17():
    print(f"[{datetime.datetime.now()}] Starting Exp17 (No Road_Name Mode)...")
    train = pd.read_csv('data/processed/train_final.csv', low_memory=False)
    test = pd.read_csv('data/processed/test_final.csv', low_memory=False)
    
    # 2017+ 필터 유지
    temp_year = train['계약년월'].astype(str).str[:4].astype(int)
    train = train[temp_year >= 2017].reset_index(drop=True)

    # 1. FE
    train, test = feature_engineering_exp17_final(train, test)
    if train['target'].max() > 100: train['target'] = np.log1p(train['target'])

    # 2. Target Encoding ('도로명' 제외)
    train, test = apply_target_encoding(train, test, ['apt_name_tmp', 'sigungu_tmp', 'coord_cluster'])

    # 3. Label Encoding (타입 안전성 확보)
    print("[Step 3] Robust Label Encoding...")
    le = LabelEncoder()
    cat_cols = train.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        combined = pd.concat([train[col].astype(str), test[col].astype(str)])
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

    # 4. Cleaning Features
    train = minimal_clean_cols(train)
    test = minimal_clean_cols(test)

    # 5. Split & Weights
    val_condition = (train['contract_year'] == 2023) & (train['contract_month'] >= 4)
    X_tr = train[~val_condition].reset_index(drop=True)
    X_val = train[val_condition].reset_index(drop=True)
    
    tr_weights = np.ones(len(X_tr))
    tr_weights[X_tr['contract_year'] == 2023] = 2.0
    tr_weights[X_tr['contract_year'] == 2022] = 1.5

    features = [c for c in X_tr.columns if c != 'target']
    
    # 6. Train
    print(f"Training with {len(features)} features...")
    params = {'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.05, 
              'num_leaves': 128, 'colsample_bytree': 0.8, 'subsample': 0.8, 'n_jobs': -1, 'seed': 42}
    
    dtrain = lgb.Dataset(X_tr[features], label=X_tr['target'], weight=tr_weights)
    dval = lgb.Dataset(X_val[features], label=X_val['target'], reference=dtrain)
    
    model = lgb.train(params, dtrain, num_boost_round=10000, valid_sets=[dval], 
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)])
    
    val_pred = model.predict(X_val[features])
    rmse = np.sqrt(mean_squared_error(np.expm1(X_val['target']), np.expm1(val_pred)))
    print(f"\n>> Final Exp17 CV RMSE: {rmse:,.0f}")

    # 7. Submission
    test_preds = model.predict(test[features])
    pd.DataFrame({'target': np.expm1(test_preds).astype(int)}).to_csv('submissions/submission_exp17_target_encoding.csv', index=False)

if __name__ == "__main__":
    train_exp17()
