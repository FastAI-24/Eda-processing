
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
# [Exp16] Golden Triangle Strategy - 업무지구 접근성 완성
# 목표:
#   1. 강남(GBD)에만 국한된 입지 성능을 여의도(YBD), 한양도성(CBD)까지 확장한다.
#   2. 직주근접 논리를 강화하여 입지 피처의 설명력을 높인다.
#   3. Exp15(가중치 적용)를 베이스라인으로 하여 개선 폭을 엄격히 검증한다.
# =============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def feature_engineering_exp16(train, test):
    print("\n[Exp16] Engineering: Completing the Golden Triangle (CBD, YBD)...")
    train_len = len(train)
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # 1. 날짜 및 연식
    df['contract_year'] = df['계약년월'].astype(str).str[:4].astype(int)
    df['contract_month'] = df['계약년월'].astype(str).str[4:6].astype(int)
    df['building_age'] = (df['contract_year'] - df['건축년도']).clip(lower=0)

    # 2. 입지: 3대 업무지구(Golden Triangle) 거리
    centers = {
        'gbd': (37.496, 127.027),  # 강남역
        'cbd': (37.566, 126.978),  # 서울시청
        'ybd': (37.521, 126.924)   # 여의도역
    }
    for name, pos in centers.items():
        df[f'dist_{name}'] = haversine_distance(df['좌표Y'], df['좌표X'], pos[0], pos[1])
    
    # 파생 변수: 가장 가까운 주요 업무지구까지의 거리
    df['min_dist_to_job'] = df[['dist_gbd', 'dist_cbd', 'dist_ybd']].min(axis=1)

    # 3. 교통 및 기타 품질 (Exp15 동일 유지)
    try:
        sub_df = pd.read_csv('data/raw/subway_feature.csv')
        sub_tree = KDTree(sub_df[['위도', '경도']].values)
        dist_s, _ = sub_tree.query(df[['좌표Y', '좌표X']].values, k=1)
        df['dist_to_subway'] = dist_s * 100
    except: pass
    
    df['parking_per_unit'] = df['주차대수'] / df['k-전체세대수'].replace(0, np.nan)
    
    # 4. 공간 클러스터링
    coords = df[['좌표Y', '좌표X']].values
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=150, random_state=42, n_init=10)
    df['coord_cluster'] = kmeans.fit_predict(scaler.fit_transform(coords))

    drop_cols = ['주차대수', 'k-전체세대수', 'k-연면적', 'k-전체동수', '아파트명', '계약일', '계약년월', 
                 '시군구', '도로명', '번지', '본번', '부번', '좌표X', '좌표Y']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    train = df.iloc[:train_len].copy()
    test = df.iloc[train_len:].copy()
    return train, test

def target_encoding_exp16(train, test):
    # Cluster 기반 평균가 인코딩
    means = train.groupby('coord_cluster')['target'].mean()
    train['cluster_mean_price'] = train['coord_cluster'].map(means)
    test['cluster_mean_price'] = test['coord_cluster'].map(means)
    
    global_mean = train['target'].mean()
    train['cluster_mean_price'].fillna(global_mean, inplace=True)
    test['cluster_mean_price'].fillna(global_mean, inplace=True)
    return train, test

def minimal_clean_cols(df):
    new_cols = []
    for col in df.columns:
        c = re.sub(r'[(),\[\]\s]', '_', col)
        new_cols.append(c)
    df.columns = new_cols
    return df

def train_exp16():
    print(f"[{datetime.datetime.now()}] Loading Data for Exp16...")
    train = pd.read_csv('data/processed/train_final.csv', low_memory=False)
    test = pd.read_csv('data/processed/test_final.csv', low_memory=False)
    
    # 2017+ 필터 유지 (Exp15와 비교를 위해)
    temp_year = train['계약년월'].astype(str).str[:4].astype(int)
    train = train[temp_year >= 2017].reset_index(drop=True)

    # 1. Feature Engineering
    train, test = feature_engineering_exp16(train, test)
    
    if train['target'].max() > 100:
        train['target'] = np.log1p(train['target'])

    # 컬럼명 클리닝 (LightGBM Error 방지)
    train = minimal_clean_cols(train)
    test = minimal_clean_cols(test)

    # 2. Time-based Split & Weights
    val_condition = (train['contract_year'] == 2023) & (train['contract_month'] >= 4)
    X_train_full = train[~val_condition].reset_index(drop=True)
    X_val_full = train[val_condition].reset_index(drop=True)
    
    # Exp 15의 시간 가중치 설정 유지
    weights = np.ones(len(X_train_full))
    weights[X_train_full['contract_year'] == 2023] = 2.0
    weights[X_train_full['contract_year'] == 2022] = 1.5

    # 3. Encoding
    X_train_full, X_val_full = target_encoding_exp16(X_train_full, X_val_full)
    _, test = target_encoding_exp16(X_train_full, test)

    le = LabelEncoder()
    cat_cols = X_train_full.select_dtypes(include=['object']).columns
    for col in cat_cols:
        combined = pd.concat([X_train_full[col], X_val_full[col], test[col]]).astype(str)
        le.fit(combined)
        X_train_full[col] = le.transform(X_train_full[col].astype(str))
        X_val_full[col] = le.transform(X_val_full[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

    # 4. 학습 (GBDT for faster local validation)
    features = [c for c in X_train_full.columns if c != 'target']
    params = {
        'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.05,
        'num_leaves': 128, 'colsample_bytree': 0.8, 'subsample': 0.8,
        'n_jobs': -1, 'seed': 42, 'verbose': -1
    }
    
    dtrain = lgb.Dataset(X_train_full[features], label=X_train_full['target'], weight=weights)
    dvalid = lgb.Dataset(X_val_full[features], label=X_val_full['target'], reference=dtrain)
    
    print("\nTraining LightGBM with Golden Triangle Features...")
    model = lgb.train(params, dtrain, num_boost_round=10000, valid_sets=[dtrain, dvalid],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(1000)])
    
    # 5. 검증 결과
    val_pred = model.predict(X_val_full[features])
    rmse = np.sqrt(mean_squared_error(np.expm1(X_val_full['target']), np.expm1(val_pred)))
    print(f"\n>> Final Exp16 (Golden Triangle) CV RMSE: {rmse:,.0f}")
    
    # 6. 최종 제출 파일 저장
    # 전체 데이터 재학습 (Option)
    full_train = pd.concat([X_train_full, X_val_full])
    full_weights = np.ones(len(full_train))
    full_weights[full_train['contract_year'] == 2023] = 2.0
    full_weights[full_train['contract_year'] == 2022] = 1.5
    
    dtrain_final = lgb.Dataset(full_train[features], label=full_train['target'], weight=full_weights)
    final_model = lgb.train(params, dtrain_final, num_boost_round=model.best_iteration)
    
    test_preds = final_model.predict(test[features])
    sub_path = 'submissions/submission_exp16_golden_triangle.csv'
    pd.DataFrame({'target': np.expm1(test_preds).astype(int)}).to_csv(sub_path, index=False)
    
    # Feature Importance
    imp = pd.DataFrame({'Feature': features, 'Gain': final_model.feature_importance('gain')}).sort_values('Gain', ascending=False)
    print("\n[Exp16 Feature Importance (Top 10)]")
    print(imp.head(10))

if __name__ == "__main__":
    train_exp16()
