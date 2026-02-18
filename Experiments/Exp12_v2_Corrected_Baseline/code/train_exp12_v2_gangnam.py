
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
# [Exp12-v2] Corrected Baseline + Gangnam Distance (Haversine)
# 목표: 
#   1. Exp08(15,642점)의 모든 피처(교통, 품질, 클러스터)를 완벽히 복원.
#   2. '강남역 거리(dist_gbd)' 피처 하나만 하버사인 공식으로 추가하여 증명.
#   3. 최신 데이터(2017+) 필터링 유지.
# =============================================================================

# 1. 환경 설정 및 경로
input_dir = 'data/processed'
raw_dir = 'data/raw'
submission_dir = 'submissions'

if not os.path.exists(submission_dir):
    os.makedirs(submission_dir)

train_path = os.path.join(input_dir, 'train_final.csv')
test_path = os.path.join(input_dir, 'test_final.csv')

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 # 지구 반지름 (km)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def minimal_clean_cols(df):
    new_cols = []
    for col in df.columns:
        c = re.sub(r'[(),\[\]\s]', '_', col)
        new_cols.append(c)
    df.columns = new_cols
    return df

def feature_engineering_exp12_v2(train, test):
    print("\n[Exp12-v2] Restoring Exp08 Features + Adding Gangnam Distance...")
    train_len = len(train)
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # 1. 시점 변수
    if '계약년월' in df.columns:
        df['contract_year'] = df['계약년월'].astype(str).str[:4].astype(int)
        df['contract_month'] = df['계약년월'].astype(str).str[4:6].astype(int)
    
    # 2. 건물 연식
    if '건축년도' in df.columns:
        df['building_age'] = df['contract_year'] - df['건축년도']
        df['building_age'] = df['building_age'].clip(lower=0)

    # 3. 품질 지표 (Exp07)
    df['k-전체세대수'] = df['k-전체세대수'].replace(0, np.nan)
    df['k-전체동수'] = df['k-전체동수'].replace(0, np.nan)
    if '주차대수' in df.columns:
        df['parking_per_unit'] = df['주차대수'] / df['k-전체세대수']
    if 'k-연면적' in df.columns:
        df['unit_area_avg'] = df['k-연면적'] / df['k-전체세대수']
    if 'k-전체동수' in df.columns:
        df['complex_density'] = df['k-전체세대수'] / df['k-전체동수']
    if 'k-전체세대수' in df.columns:
        df['log_total_units'] = np.log1p(df['k-전체세대수'])
    if 'k-연면적' in df.columns:
        df['log_total_area'] = np.log1p(df['k-연면적'])

    # 4. 교통 지표 (Exp05/08 복원) - 이 부분이 이전 Exp10~12에서 누락되었습니다.
    print("  -> Restoring Transport Features (Subway/Bus)...")
    try:
        sub_df = pd.read_csv(os.path.join(raw_dir, 'subway_feature.csv'))
        sub_coords = sub_df[['위도', '경도']].values
        tree = KDTree(sub_coords, metric='euclidean')
        coords = df[['좌표Y', '좌표X']].values
        dist, idx = tree.query(coords, k=1)
        df['dist_to_subway'] = dist * 100
        nearest_indices = idx.flatten()
        line_col = '호선' if '호선' in sub_df.columns else sub_df.columns[2]
        df['subway_line'] = sub_df.iloc[nearest_indices][line_col].values
    except Exception as e:
        print(f"Warning: Subway features failed: {e}")

    try:
        bus_df = pd.read_csv(os.path.join(raw_dir, 'bus_feature.csv')) 
        if 'Y' in bus_df.columns:
            bus_coords = bus_df[['Y', 'X']].values
            tree_bus = KDTree(bus_coords, metric='euclidean')
            coords = df[['좌표Y', '좌표X']].values
            dist, _ = tree_bus.query(coords, k=1)
            df['dist_to_bus'] = dist * 100
            counts = tree_bus.query_radius(coords, r=0.005, count_only=True)
            df['bus_count_500'] = counts
    except Exception as e:
        print(f"Warning: Bus features failed: {e}")

    # 5. [Exp12 New] 강남역 거리 (Single Variable Proof)
    print("  -> Adding Gangnam Distance (Haversine)...")
    # 강남역: 37.496, 127.027
    df['dist_gbd'] = haversine_distance(df['좌표Y'], df['좌표X'], 37.496, 127.027)

    # 6. 공간 클러스터링 (Exp08, K=150)
    print("  -> Generating Spatial Clusters (K=150)...")
    coords = df[['좌표Y', '좌표X']].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    kmeans = KMeans(n_clusters=150, random_state=42, n_init=10)
    df['coord_cluster'] = kmeans.fit_predict(coords_scaled)
    
    # 7. 다이어트
    drop_cols = ['주차대수', 'k-전체세대수', 'k-연면적', 'k-전체동수', '아파트명', '계약일', '계약년월', 
                 '시군구', '도로명', '번지', '본번', '부번']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')
    
    train = df.iloc[:train_len].copy()
    test = df.iloc[train_len:].copy()
    return train, test

def target_encoding_kfold(train, test, cluster_col='coord_cluster', target_col='target', n_splits=5):
    print(f"  -> Applying K-Fold Target Encoding for Clusters...")
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

def train_exp12_v2():
    print("Loading Data...")
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    
    # 2017년 이후 데이터만 필터링 (Exp08 베이스라인)
    temp_year = train['계약년월'].astype(str).str[:4].astype(int)
    train = train[temp_year >= 2017].reset_index(drop=True)
    print(f"Data Filtered (2017+): {len(train)} samples.")

    # 1. Feature Engineering
    train, test = feature_engineering_exp12_v2(train, test)
    
    # 2. Preprocessing
    train = minimal_clean_cols(train)
    test = minimal_clean_cols(test)
    if train['target'].max() > 100: train['target'] = np.log1p(train['target'])

    # 타겟 인코딩 (Log 변환 후)
    train, test = target_encoding_kfold(train, test)
    
    # Label Encoding
    le = LabelEncoder()
    col_list = train.select_dtypes(include=['object']).columns
    for col in col_list:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    # 3. 학습
    print("\n[Validation Strategy] Shuffle K-Fold (5 Splits)")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    features = [c for c in train.columns if c != 'target']
    print(f"Final Features ({len(features)}): {features}")
    
    params = {
        'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.03, 'num_leaves': 256,
        'colsample_bytree': 0.8, 'subsample': 0.8, 'n_jobs': -1, 'seed': 42, 'verbose': -1
    }
    
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train[features], train['target'])):
        X_tr, X_val = train.iloc[train_idx][features], train.iloc[val_idx][features]
        y_tr, y_val = train.iloc[train_idx]['target'], train.iloc[val_idx]['target']
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(params, dtrain, num_boost_round=10000, valid_sets=[dtrain, dvalid],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(1000)])
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(test[features]) / 5
        
        rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_pred)))
        print(f"Fold {fold+1} RMSE: {rmse:,.0f}")
        
    total_rmse = np.sqrt(mean_squared_error(np.expm1(train['target']), np.expm1(oof_preds)))
    print(f"\n>> Overall Exp12-v2 CV RMSE: {total_rmse:,.0f}")
    
    # 4. 저장
    sub_path = os.path.join(submission_dir, 'submission_exp12_v2_final.csv')
    pd.DataFrame({'target': np.expm1(test_preds)}).to_csv(sub_path, index=False)
    print(f"Submission Saved: {sub_path}")

if __name__ == "__main__":
    train_exp12_v2()
