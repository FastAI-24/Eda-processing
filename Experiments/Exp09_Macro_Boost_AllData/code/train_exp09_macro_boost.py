
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
# [Exp09] Macro Economics Boost + All Data + Time Decay Weighting
# 목표: 
#   1. 기준금리(Base Rate)를 추가하여 거시적 경제 흐름 반영
#   2. 2007~2023 전체 데이터를 사용하되, 가중치를 활용해 최근 데이터 강조
#   3. 기존의 품질 변수(Exp07) 및 공간 클러스터링(Exp08) 통합
# =============================================================================

# 1. 환경 설정 및 경로
# -----------------
input_dir = 'data/processed'
raw_dir = 'data/raw'
submission_dir = 'submissions'

if not os.path.exists(submission_dir):
    os.makedirs(submission_dir)

train_path = os.path.join(input_dir, 'train_final.csv')
test_path = os.path.join(input_dir, 'test_final.csv')

# 2. 한국은행 기준금리 매핑 데이터 (2007-2023)
# ----------------------------------------
def get_base_rate_map():
    # 기준금리 변동 내역 (주요 시점 기준 월별 매핑)
    # 데이터 출처: 한국은행
    base_rate_dict = {
        2007: {1: 4.5, 2: 4.5, 3: 4.5, 4: 4.5, 5: 4.5, 6: 4.5, 7: 4.75, 8: 5.0, 9: 5.0, 10: 5.0, 11: 5.0, 12: 5.0},
        2008: {1: 5.0, 2: 5.0, 3: 5.0, 4: 5.0, 5: 5.0, 6: 5.0, 7: 5.0, 8: 5.25, 9: 5.25, 10: 4.25, 11: 4.0, 12: 3.0},
        2009: {1: 2.50, 2: 2.00, 3: 2.00, 4: 2.00, 5: 2.00, 6: 2.00, 7: 2.00, 8: 2.00, 9: 2.00, 10: 2.00, 11: 2.00, 12: 2.00},
        2010: {1: 2.00, 2: 2.00, 3: 2.00, 4: 2.00, 5: 2.00, 6: 2.00, 7: 2.25, 8: 2.25, 9: 2.25, 10: 2.25, 11: 2.50, 12: 2.50},
        2011: {1: 2.75, 2: 2.75, 3: 3.00, 4: 3.00, 5: 3.00, 6: 3.25, 7: 3.25, 8: 3.25, 9: 3.25, 10: 3.25, 11: 3.25, 12: 3.25},
        2012: {1: 3.25, 2: 3.25, 3: 3.25, 4: 3.25, 5: 3.25, 6: 3.25, 7: 3.00, 8: 3.00, 9: 3.00, 10: 2.75, 11: 2.75, 12: 2.75},
        2013: {1: 2.75, 2: 2.75, 3: 2.75, 4: 2.75, 5: 2.50, 6: 2.50, 7: 2.50, 8: 2.50, 9: 2.50, 10: 2.50, 11: 2.50, 12: 2.50},
        2014: {1: 2.50, 2: 2.50, 3: 2.50, 4: 2.50, 5: 2.50, 6: 2.50, 7: 2.50, 8: 2.25, 9: 2.25, 10: 2.00, 11: 2.00, 12: 2.00},
        2015: {1: 2.00, 2: 2.00, 3: 1.75, 4: 1.75, 5: 1.75, 6: 1.50, 7: 1.50, 8: 1.50, 9: 1.50, 10: 1.50, 11: 1.50, 12: 1.50},
        2016: {1: 1.50, 2: 1.50, 3: 1.50, 4: 1.50, 5: 1.50, 6: 1.25, 7: 1.25, 8: 1.25, 9: 1.25, 10: 1.25, 11: 1.25, 12: 1.25},
        2017: {1: 1.25, 2: 1.25, 3: 1.25, 4: 1.25, 5: 1.25, 6: 1.25, 7: 1.25, 8: 1.25, 9: 1.25, 10: 1.25, 11: 1.50, 12: 1.50},
        2018: {1: 1.50, 2: 1.50, 3: 1.50, 4: 1.50, 5: 1.50, 6: 1.50, 7: 1.50, 8: 1.50, 9: 1.50, 10: 1.50, 11: 1.75, 12: 1.75},
        2019: {1: 1.75, 2: 1.75, 3: 1.75, 4: 1.75, 5: 1.75, 6: 1.75, 7: 1.50, 8: 1.50, 9: 1.50, 10: 1.25, 11: 1.25, 12: 1.25},
        2020: {1: 1.25, 2: 1.25, 3: 0.75, 4: 0.75, 5: 0.50, 6: 0.50, 7: 0.50, 8: 0.50, 9: 0.50, 10: 0.50, 11: 0.50, 12: 0.50},
        2021: {1: 0.50, 2: 0.50, 3: 0.50, 4: 0.50, 5: 0.50, 6: 0.50, 7: 0.50, 8: 0.75, 9: 0.75, 10: 0.75, 11: 1.00, 12: 1.00},
        2022: {1: 1.25, 2: 1.25, 3: 1.25, 4: 1.50, 5: 1.75, 6: 1.75, 7: 2.25, 8: 2.50, 9: 2.50, 10: 3.00, 11: 3.25, 12: 3.25},
        2023: {1: 3.50, 2: 3.50, 3: 3.50, 4: 3.50, 5: 3.50, 6: 3.50, 7: 3.50, 8: 3.50, 9: 3.50, 10: 3.50, 11: 3.50, 12: 3.50},
    }
    return base_rate_dict

# 3. 유틸리티 함수
# --------------
def minimal_clean_cols(df):
    new_cols = []
    for col in df.columns:
        c = re.sub(r'[(),\[\]\s]', '_', col)
        new_cols.append(c)
    df.columns = new_cols
    return df

# 4. 피처 엔지니어링 (핵심)
# ----------------------
def feature_engineering_macro(train, test):
    print("\n[Exp09] Feature Engineering (Macro + Quality + Cluster)...")
    
    train_len = len(train)
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # A. 날짜 및 연식 변수
    if '계약년월' in df.columns:
        df['contract_year'] = df['계약년월'].astype(str).str[:4].astype(int)
        df['contract_month'] = df['계약년월'].astype(str).str[4:6].astype(int)
    if '건축년도' in df.columns:
        df['building_age'] = df['contract_year'] - df['건축년도']
        df['building_age'] = df['building_age'].clip(lower=0)

    # B. 거시경제 변수: 기준금리(Base Rate) 추가
    print("  -> Adding Macro Feature: Base Rate...")
    br_map = get_base_rate_map()
    df['base_rate'] = df.apply(lambda x: br_map.get(x['contract_year'], {}).get(x['contract_month'], 2.0), axis=1)

    # C. 품질 변수 (Exp07)
    print("  -> Generating Quality Features (Parking, Density, Area)...")
    df['k-전체세대수'] = df['k-전체세대수'].replace(0, np.nan)
    df['k-전체동수'] = df['k-전체동수'].replace(0, np.nan)
    
    if '주차대수' in df.columns:
        df['parking_per_unit'] = df['주차대수'] / df['k-전체세대수']
    if 'k-연면적' in df.columns:
        df['unit_area_avg'] = df['k-연면적'] / df['k-전체세대수']
    if 'k-전체동수' in df.columns:
        df['complex_density'] = df['k-전체세대수'] / df['k-전체동수']

    # 로그 변환 (규모 변수)
    if 'k-전체세대수' in df.columns:
        df['log_total_units'] = np.log1p(df['k-전체세대수'])
    if 'k-연면적' in df.columns:
        df['log_total_area'] = np.log1p(df['k-연면적'])

    # D. 교통 변수 (KDTree 활용)
    print("  -> Adding Transport Features (Subway, Bus)...")
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

    try:
        bus_path = os.path.join(raw_dir, 'bus_feature.csv')
        bus_df = pd.read_csv(bus_path) 
        if 'Y' in bus_df.columns:
            bus_coords = bus_df[['Y', 'X']].values
            tree_bus = KDTree(bus_coords, metric='euclidean')
            coords = df[['좌표Y', '좌표X']].values
            dist, _ = tree_bus.query(coords, k=1)
            df['dist_to_bus'] = dist * 100
            counts = tree_bus.query_radius(coords, r=0.005, count_only=True)
            df['bus_count_500'] = counts
    except: pass

    # E. 공간 클러스터링 (Exp08)
    print("  -> Running Spatial Clustering (K=150)...")
    coords = df[['좌표Y', '좌표X']].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    kmeans = KMeans(n_clusters=150, random_state=42, n_init=10)
    df['coord_cluster'] = kmeans.fit_predict(coords_scaled)
    
    # 불필요한 원본 컬럼 삭제 (Data Diet)
    drop_cols = ['주차대수', 'k-전체세대수', 'k-연면적', 'k-전체동수', '아파트명', '계약일', '계약년월', 
                 '시군구', '도로명', '번지', '본번', '부번']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')
    
    # 5. 시간 가중치 계산 (Sample Weights)
    # 최근 2023년 데이터에 더 큰 비중을 둠 (alpha 조절 가능)
    # weight = exp((year - 2023) * 0.1) -> 2023: 1.0, 2017: 0.55, 2007: 0.2
    print("  -> Calculating Time Decay Weights...")
    df['sample_weight'] = np.exp((df['contract_year'] - 2023) * 0.1)
    
    # 분리
    train = df.iloc[:train_len].copy()
    test = df.iloc[train_len:].copy()
    
    return train, test

# 5. 타겟 인코딩 (OOF 방식)
# ------------------------
def target_encoding_kfold(train, test, cluster_col='coord_cluster', target_col='target', n_splits=5):
    print(f"  -> Applying K-Fold Target Encoding for {cluster_col}...")
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

# 6. 메인 학습 루프
# ---------------
def train_exp09_macro():
    print("Loading Data...")
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    
    # 1. Feature Engineering (전체 데이터 사용)
    train, test = feature_engineering_macro(train, test)
    
    # 2. 전처리
    train = minimal_clean_cols(train)
    test = minimal_clean_cols(test)
    
    if train['target'].max() > 100:
        train['target'] = np.log1p(train['target'])

    # 타겟 인코딩 (클러스터)
    train, test = target_encoding_kfold(train, test)
    
    # 카테고리 인코딩
    le = LabelEncoder()
    col_list = train.select_dtypes(include=['object']).columns
    for col in col_list:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    # 3. Validation & Training
    print("\n[Validation Strategy] Shuffle K-Fold (5 Splits) with All Data + Weights")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 'sample_weight'는 피처가 아니므로 제외
    features = [c for c in train.columns if c not in ['target', 'sample_weight']]
    print(f"Final Features: {features}")
    
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
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train[features], train['target'])):
        X_tr, X_val = train.iloc[train_idx][features], train.iloc[val_idx][features]
        y_tr, y_val = train.iloc[train_idx]['target'], train.iloc[val_idx]['target']
        w_tr = train.iloc[train_idx]['sample_weight'] # 학습 폴드 가중치 적용
        
        # Dataset 생성 시 weight 파라미터 전달
        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=10000,
            valid_sets=[dtrain, dvalid],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(1000)]
        )
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(test[features]) / 5
        
        rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_pred)))
        print(f"Fold {fold+1} RMSE: {rmse:,.0f}")
        
    total_rmse = np.sqrt(mean_squared_error(np.expm1(train['target']), np.expm1(oof_preds)))
    print(f"\n>> Overall Exp09 CV RMSE (All Data Weighted): {total_rmse:,.0f}")
    
    # Feature Importance 확인
    imp = pd.DataFrame({'Feature': features, 'Gain': model.feature_importance('gain')}).sort_values('Gain', ascending=False)
    print("\n[Top 10 Features - Exp09]")
    print(imp.head(10))
    
    # 4. 결과 저장
    sub_path = os.path.join(submission_dir, 'submission_exp09_macro.csv')
    pd.DataFrame({'target': np.expm1(test_preds)}).to_csv(sub_path, index=False)
    print(f"Submission Saved: {sub_path}")

if __name__ == "__main__":
    train_exp09_macro()
