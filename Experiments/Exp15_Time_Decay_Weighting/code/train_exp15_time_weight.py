
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import re
from sklearn.neighbors import KDTree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import datetime

# =============================================================================
# [Exp15] Time Decay Weighting + Honest Validation
# 목표:
#   1. Exp14(Honest CV) 체계 위에서 최신 데이터(2022~2023)에 가중치를 부여한다.
#   2. 과거 데이터보다 최근 시장 트렌드(금리 인상기 등)를 더 강하게 학습시킨다.
#   3. Learning Rate를 낮춰 가중치 학습의 안정성을 확보한다.
# =============================================================================

# 1. 하버사인 거리 계산
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# 2. 환경 설정
input_dir = 'data/processed'
raw_dir = 'data/raw'
submission_dir = 'submissions'
if not os.path.exists(submission_dir): os.makedirs(submission_dir)

train_path = os.path.join(input_dir, 'train_final.csv')
test_path = os.path.join(input_dir, 'test_final.csv')

def minimal_clean_cols(df):
    new_cols = []
    for col in df.columns:
        c = re.sub(r'[(),\[\]\s]', '_', col)
        new_cols.append(c)
    df.columns = new_cols
    return df

# 3. Feature Engineering (Exp12-v2/Exp14 Same)
def feature_engineering_exp15(train, test):
    print("\n[Exp15] Engineering: Exp12-v2 Features + Time Weighting Prep...")
    train_len = len(train)
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # 날짜 및 연식
    df['contract_year'] = df['계약년월'].astype(str).str[:4].astype(int)
    df['contract_month'] = df['계약년월'].astype(str).str[4:6].astype(int)
    df['building_age'] = (df['contract_year'] - df['건축년도']).clip(lower=0)

    # 품질 지표
    df['k-전체세대수'] = df['k-전체세대수'].replace(0, np.nan)
    df['parking_per_unit'] = df['주차대수'] / df['k-전체세대수']
    df['log_total_units'] = np.log1p(df['k-전체세대수'])

    # 교통 및 강남 거리
    print("  -> Calculating Distances (Subway & Gangnam)...")
    try:
        sub_df = pd.read_csv(os.path.join(raw_dir, 'subway_feature.csv'))
        sub_tree = KDTree(sub_df[['위도', '경도']].values)
        dist_s, _ = sub_tree.query(df[['좌표Y', '좌표X']].values, k=1)
        df['dist_to_subway'] = dist_s * 100
    except: pass
    
    df['dist_gbd'] = haversine_distance(df['좌표Y'], df['좌표X'], 37.496, 127.027)

    # 공간 클러스터링
    print("  -> Generating Spatial Clusters...")
    coords = df[['좌표Y', '좌표X']].values
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=150, random_state=42, n_init=10)
    df['coord_cluster'] = kmeans.fit_predict(scaler.fit_transform(coords))
    
    # 다이어트
    drop_cols = ['주차대수', 'k-전체세대수', 'k-연면적', 'k-전체동수', '아파트명', '계약일', '계약년월', 
                 '시군구', '도로명', '번지', '본번', '부번']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')
    
    train = df.iloc[:train_len].copy()
    test = df.iloc[train_len:].copy()
    return train, test

def target_encoding_exp15(train, test, cluster_col='coord_cluster'):
    means = train.groupby(cluster_col)['target'].mean()
    train[f'{cluster_col}_mean_price'] = train[cluster_col].map(means)
    test[f'{cluster_col}_mean_price'] = test[cluster_col].map(means)
    
    global_mean = train['target'].mean()
    train[f'{cluster_col}_mean_price'].fillna(global_mean, inplace=True)
    test[f'{cluster_col}_mean_price'].fillna(global_mean, inplace=True)
    return train, test

# 4. 메인 학습 루틴
def train_exp15_weighting():
    print(f"[{datetime.datetime.now()}] Loading Data...")
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    
    # 최신 데이터 필터링 (2017+)
    temp_year = train['계약년월'].astype(str).str[:4].astype(int)
    train = train[temp_year >= 2017].reset_index(drop=True)

    # 1. Feature Engineering
    train, test = feature_engineering_exp15(train, test)
    
    if train['target'].max() > 100:
        train['target'] = np.log1p(train['target'])

    # 2. Time-based Split & Weight Assignment
    # 검증셋: 2023년 4월 ~ 6월 (최근 3개월)
    val_condition = (train['contract_year'] == 2023) & (train['contract_month'] >= 4)
    X_train_full = train[~val_condition].reset_index(drop=True)
    X_val_full = train[val_condition].reset_index(drop=True)
    
    print(f"\n[Time-Split] Train: {len(X_train_full)} samples")
    print(f"[Time-Split] Valid: {len(X_val_full)} samples (Strictly 2023.04+)")

    # [핵심] 가중치 부여 (Step Weighting)
    # 학습 데이터에만 적용
    # 2023년: 2.0 / 2022년: 1.5 / 그외: 1.0
    weights = np.ones(len(X_train_full))
    weights[X_train_full['contract_year'] == 2023] = 2.0
    weights[X_train_full['contract_year'] == 2022] = 1.5
    
    print(f"  -> Applied Time Weights: 2023(2.0), 2022(1.5), Others(1.0)")

    # 타겟 인코딩
    X_train_full, X_val_full = target_encoding_exp15(X_train_full, X_val_full)
    _, test = target_encoding_exp15(X_train_full, test)

    # 3. 전처리 & 인코딩
    X_train_full = minimal_clean_cols(X_train_full)
    X_val_full = minimal_clean_cols(X_val_full)
    test = minimal_clean_cols(test)
    
    features = [c for c in X_train_full.columns if c != 'target']
    cat_cols = X_train_full.select_dtypes(include=['object']).columns
    
    le = LabelEncoder()
    for col in cat_cols:
        combined = pd.concat([X_train_full[col], X_val_full[col], test[col]]).astype(str)
        le.fit(combined)
        X_train_full[col] = le.transform(X_train_full[col].astype(str))
        X_val_full[col] = le.transform(X_val_full[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

    # 4. 학습 (Low LR for Stability)
    params = {
        'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.03, # Lower LR
        'num_leaves': 128, 'colsample_bytree': 0.8, 'subsample': 0.8,
        'n_jobs': -1, 'seed': 42, 'verbose': -1
    }
    
    # Weight 전달
    dtrain = lgb.Dataset(X_train_full[features], label=X_train_full['target'], weight=weights)
    dvalid = lgb.Dataset(X_val_full[features], label=X_val_full['target'], reference=dtrain)
    
    print("\nTraining LightGBM with Time Decay Weights...")
    model = lgb.train(params, dtrain, num_boost_round=10000, valid_sets=[dtrain, dvalid],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(1000)])
    
    # 5. 검증
    val_pred = model.predict(X_val_full[features])
    rmse = np.sqrt(mean_squared_error(np.expm1(X_val_full['target']), np.expm1(val_pred)))
    print(f"\n>> Final Exp15 Weighted CV RMSE: {rmse:,.0f}")
    
    # 6. 리더보드용 재학습 (전체 데이터 + 가중치 적용)
    print("Re-training on full data with weights...")
    full_train = pd.concat([X_train_full, X_val_full])
    full_weights = np.ones(len(full_train))
    full_weights[full_train['contract_year'] == 2023] = 2.0
    full_weights[full_train['contract_year'] == 2022] = 1.5
    
    dtrain_final = lgb.Dataset(full_train[features], label=full_train['target'], weight=full_weights)
    final_model = lgb.train(params, dtrain_final, num_boost_round=model.best_iteration)
    
    test_preds = final_model.predict(test[features])
    sub_path = os.path.join(submission_dir, 'submission_exp15_weighted.csv')
    pd.DataFrame({'target': np.expm1(test_preds).astype(int)}).to_csv(sub_path, index=False)
    print(f"Submission Saved: {sub_path}")

if __name__ == "__main__":
    train_exp15_weighting()
