
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

# =============================================================================
# [Exp09-Step1] Only All Data + Time Decay Weighting
# 목표: 금리 등 외부 변수 없이, "전체 데이터 + 시간 가중치"만으로 성능 변화를 측정.
# 이전 실패(16,803)의 원인을 파악하기 위해 변수 추가를 배제하고 가중치 효과만 검증함.
# =============================================================================

# 1. 환경 설정 및 경로
input_dir = 'data/processed'
raw_dir = 'data/raw'
submission_dir = 'submissions'

if not os.path.exists(submission_dir):
    os.makedirs(submission_dir)

train_path = os.path.join(input_dir, 'train_final.csv')
test_path = os.path.join(input_dir, 'test_final.csv')

def minimal_clean_cols(df):
    new_cols = []
    for col in df.columns:
        c = re.sub(r'[(),\[\]\s]', '_', col)
        new_cols.append(c)
    df.columns = new_cols
    return df

# 2. 피처 엔지니어링 (Exp08 수준 유지)
def feature_engineering_step1(train, test):
    print("\n[Exp09-Step1] Engineering (Inherit Exp08 Features)...")
    train_len = len(train)
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # 시간 관련
    if '계약년월' in df.columns:
        df['contract_year'] = df['계약년월'].astype(str).str[:4].astype(int)
        df['contract_month'] = df['계약년월'].astype(str).str[4:6].astype(int)
    if '건축년도' in df.columns:
        df['building_age'] = df['contract_year'] - df['건축년도']
        df['building_age'] = df['building_age'].clip(lower=0)

    # 품질 변수 (Exp07)
    df['k-전체세대수'] = df['k-전체세대수'].replace(0, np.nan)
    df['k-전체동수'] = df['k-전체동수'].replace(0, np.nan)
    if '주차대수' in df.columns: df['parking_per_unit'] = df['주차대수'] / df['k-전체세대수']
    if 'k-연면적' in df.columns: df['unit_area_avg'] = df['k-연면적'] / df['k-전체세대수']
    if 'k-전체동수' in df.columns: df['complex_density'] = df['k-전체세대수'] / df['k-전체동수']
    if 'k-전체세대수' in df.columns: df['log_total_units'] = np.log1p(df['k-전체세대수'])
    if 'k-연면적' in df.columns: df['log_total_area'] = np.log1p(df['k-연면적'])

    # 교통 (Inherit)
    try:
        sub_path = os.path.join(raw_dir, 'subway_feature.csv')
        sub_df = pd.read_csv(sub_path)
        sub_coords = sub_df[['위도', '경도']].values
        tree = KDTree(sub_coords, metric='euclidean')
        coords = df[['좌표Y', '좌표X']].values
        dist, idx = tree.query(coords, k=1)
        df['dist_to_subway'] = dist * 100
        nearest_indices = idx.flatten()
        if '호선' in sub_df.columns: line_col = '호선'
        elif '노선명' in sub_df.columns: line_col = '노선명'
        else: line_col = sub_df.columns[2]
        df['subway_line'] = sub_df.iloc[nearest_indices][line_col].values
    except: pass

    # 공간 클러스터링 (Exp08)
    coords = df[['좌표Y', '좌표X']].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    kmeans = KMeans(n_clusters=150, random_state=42, n_init=10)
    df['coord_cluster'] = kmeans.fit_predict(coords_scaled)
    
    # 가중치 계산 (여기가 유일한 핵심 변화)
    # 2023년은 1.0, 점차 줄어들어 2007년은 약 0.2가 되도록 설정
    df['sample_weight'] = np.exp((df['contract_year'] - 2023) * 0.1)
    
    # 다이어트
    drop_cols = ['주차대수', 'k-전체세대수', 'k-연면적', 'k-전체동수', '아파트명', '계약일', '계약년월', 
                 '시군구', '도로명', '번지', '본번', '부번']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')
    
    train = df.iloc[:train_len].copy()
    test = df.iloc[train_len:].copy()
    return train, test

def target_encoding_kfold(train, test, cluster_col='coord_cluster', target_col='target', n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train[f'{cluster_col}_mean_price'] = np.nan
    test[f'{cluster_col}_mean_price'] = np.nan
    for tr_idx, val_idx in kf.split(train):
        X_tr = train.iloc[tr_idx]
        means = X_tr.groupby(cluster_col)[target_col].mean()
        train.loc[val_idx, f'{cluster_col}_mean_price'] = train.iloc[val_idx][cluster_col].map(means)
    global_mean = train[target_col].mean()
    train[f'{cluster_col}_mean_price'].fillna(global_mean, inplace=True)
    global_means = train.groupby(cluster_col)[target_col].mean()
    test[f'{cluster_col}_mean_price'] = test[cluster_col].map(global_means)
    test[f'{cluster_col}_mean_price'].fillna(global_mean, inplace=True)
    return train, test

# 3. 메인 학습
def train_exp09_step1():
    print("Loading Data...")
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    
    # Feature Engineering (전체 데이터 활용 + 가중치)
    train, test = feature_engineering_step1(train, test)
    
    train = minimal_clean_cols(train)
    test = minimal_clean_cols(test)
    if train['target'].max() > 100: train['target'] = np.log1p(train['target'])

    train, test = target_encoding_kfold(train, test)
    
    le = LabelEncoder()
    for col in train.select_dtypes(include=['object']).columns:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    print("\n[Validation Strategy] Shuffle K-Fold with All Data + Time Decay Weight")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    features = [c for c in train.columns if c not in ['target', 'sample_weight']]
    
    params = {
        'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.03, 'num_leaves': 256,
        'colsample_bytree': 0.8, 'subsample': 0.8, 'n_jobs': -1, 'seed': 42, 'verbose': -1
    }
    
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train[features], train['target'])):
        X_tr, X_val = train.iloc[train_idx][features], train.iloc[val_idx][features]
        y_tr, y_val = train.iloc[train_idx]['target'], train.iloc[val_idx]['target']
        w_tr = train.iloc[train_idx]['sample_weight'] # 가중치 적용
        
        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(params, dtrain, num_boost_round=10000, valid_sets=[dtrain, dvalid],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(1000)])
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(test[features]) / 5
        
        rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_pred)))
        print(f"Fold {fold+1} RMSE: {rmse:,.0f}")
        
    total_rmse = np.sqrt(mean_squared_error(np.expm1(train['target']), np.expm1(oof_preds)))
    print(f"\n>> Overall Exp09-Step1 CV RMSE: {total_rmse:,.0f}")
    
    sub_path = os.path.join(submission_dir, 'submission_exp09_step1_weight.csv')
    pd.DataFrame({'target': np.expm1(test_preds)}).to_csv(sub_path, index=False)
    print(f"Submission Saved: {sub_path}")

if __name__ == "__main__":
    train_exp09_step1()
