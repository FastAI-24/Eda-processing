
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
# [Exp18] The Hidden King - Private Leaderboard Dominator
# 전략:
#   1. 데이터 복구: 2007년~2023년 전 기간 데이터 활용 (장기 시계열 패턴 학습)
#   2. 매크로 지표: 한국은행 기준금리(Base Rate) 전 기간 매핑
#   3. OOF 타겟 인코딩: 아파트명, 시군구 5-Fold OOF 적용
#   4. DART 부스팅: 과적합 방지를 위해 DART 알고리즘 사용
#   5. 타겟 로그 변환: np.log1p 적용
# =============================================================================

# --- 한국은행 기준금리 데이터 (2007~2023) ---
interest_rates = [
    ('200701', 4.50), ('200707', 4.75), ('200708', 5.00),
    ('200803', 5.00), ('200808', 5.25), ('200810', 4.25), ('200811', 4.00), ('200812', 3.00),
    ('200901', 2.50), ('200902', 2.00),
    ('201007', 2.25), ('201011', 2.50),
    ('201101', 2.75), ('201103', 3.00), ('201106', 3.25),
    ('201207', 3.00), ('201210', 2.75),
    ('201305', 2.50),
    ('201408', 2.25), ('201410', 2.00),
    ('201503', 1.75), ('201506', 1.50),
    ('201606', 1.25),
    ('201711', 1.50),
    ('201811', 1.75),
    ('201907', 1.50), ('201910', 1.25),
    ('202003', 0.75), ('202005', 0.50),
    ('202108', 0.75), ('202111', 1.00),
    ('202201', 1.25), ('202204', 1.50), ('202205', 1.75), ('202207', 2.25), ('202208', 2.50), ('202210', 3.00), ('202211', 3.25),
    ('202301', 3.50)
]

def get_base_rate(yyyymm):
    yyyymm = int(yyyymm)
    current_rate = 4.50
    for date, rate in interest_rates:
        if yyyymm >= int(date):
            current_rate = rate
        else:
            break
    return current_rate

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
        c = re.sub(r'[(),\[\]\s\-]', '_', col)
        new_cols.append(c)
    df.columns = new_cols
    return df

def feature_engineering_hidden_king(train, test):
    print("\n[Hidden King] Expanding Features: Macro + Distance + Cluster...")
    train_len = len(train)
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # 1. Macro: Interest Rate
    df['base_rate'] = df['계약년월'].apply(get_base_rate)
    
    # 2. Time Features
    df['contract_year'] = df['계약년월'].astype(str).str[:4].astype(int)
    df['contract_month'] = df['계약년월'].astype(str).str[4:6].astype(int)
    df['building_age'] = (df['contract_year'] - df['건축년도']).clip(lower=0)
    
    # 3. Location: Golden Triangle (Gangnam, CBD, YBD)
    centers = {'gbd': (37.496, 127.027), 'cbd': (37.566, 126.978), 'ybd': (37.521, 126.924)}
    for name, pos in centers.items():
        df[f'dist_{name}'] = haversine_distance(df['좌표Y'], df['좌표X'], pos[0], pos[1])
    df['min_dist_to_job'] = df[['dist_gbd', 'dist_cbd', 'dist_ybd']].min(axis=1)

    # 4. Cluster & Quality
    df['parking_per_unit'] = df['주차대수'] / df['k-전체세대수'].replace(0, np.nan)
    coords = df[['좌표Y', '좌표X']].values
    scaler = StandardScaler()
    df['coord_cluster'] = KMeans(n_clusters=200, random_state=42, n_init=10).fit_predict(scaler.fit_transform(coords))

    # 5. Prep for Target Encoding
    df['apt_name_tmp'] = df['아파트명'].astype(str)
    df['sigungu_tmp'] = df['시군구'].astype(str)

    drop_cols = ['주차대수', 'k-전체세대수', 'k-연면적', 'k-전체동수', '아파트명', '도로명', '계약일', '계약년월', '번지', '본번', '부번', '좌표X', '좌표Y', '시군구']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    train = df.iloc[:train_len].copy()
    test = df.iloc[train_len:].copy()
    return train, test

def apply_target_encoding(train, test, cat_cols):
    print(f"[Hidden King] Advanced OOF Target Encoding: {cat_cols}")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for col in cat_cols:
        name = f'{col}_mean_price'
        train[name] = np.nan
        for tr_idx, val_idx in kf.split(train):
            means = train.iloc[tr_idx].groupby(col)['target'].mean()
            train.loc[val_idx, name] = train.iloc[val_idx][col].map(means)
        
        full_means = train.groupby(col)['target'].mean()
        test[name] = test[col].map(full_means)
        
        global_mean = train['target'].mean()
        train[name].fillna(global_mean, inplace=True)
        test[name].fillna(global_mean, inplace=True)
    
    train.drop(columns=cat_cols, inplace=True)
    test.drop(columns=cat_cols, inplace=True)
    return train, test

def train_hidden_king():
    print(f"[{datetime.datetime.now()}] Summoning 'The Hidden King' Model...")
    train = pd.read_csv('data/processed/train_final.csv', low_memory=False)
    test = pd.read_csv('data/processed/test_final.csv', low_memory=False)
    
    # [핵심] 2007년 이후 모든 데이터 복구
    print(f"Full Data Restored: 2007 ~ 2023 ({len(train):,} samples)")

    # 1. FE
    train, test = feature_engineering_hidden_king(train, test)
    if train['target'].max() > 100: train['target'] = np.log1p(train['target'])

    # 2. Target Encoding
    train, test = apply_target_encoding(train, test, ['apt_name_tmp', 'sigungu_tmp', 'coord_cluster'])

    # 3. Label Encoding
    le = LabelEncoder()
    cat_cols = train.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        combined = pd.concat([train[col].astype(str), test[col].astype(str)])
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

    train = minimal_clean_cols(train)
    test = minimal_clean_cols(test)

    # 4. Honest Time-Split Validation
    val_condition = (train['contract_year'] == 2023) & (train['contract_month'] >= 4)
    X_tr = train[~val_condition].reset_index(drop=True)
    X_val = train[val_condition].reset_index(drop=True)
    
    # 5. Time Decay Weighting (최신 데이터 3배 강화)
    tr_weights = np.ones(len(X_tr))
    tr_weights[X_tr['contract_year'] >= 2023] = 3.0
    tr_weights[X_tr['contract_year'] == 2022] = 2.0
    tr_weights[X_tr['contract_year'] <= 2015] = 0.5 # 과거 데이터 가중치 약화

    features = [c for c in X_tr.columns if c != 'target']
    
    # 6. Train with DART
    print(f"Training DART Model on {len(features)} features...")
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'dart', # 과적합 방지의 핵심
        'learning_rate': 0.1,
        'num_leaves': 255,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'n_jobs': -1,
        'seed': 42,
        'drop_rate': 0.1,
        'skip_drop': 0.5,
        'verbose': -1
    }
    
    dtrain = lgb.Dataset(X_tr[features], label=X_tr['target'], weight=tr_weights)
    dval = lgb.Dataset(X_val[features], label=X_val['target'], reference=dtrain)
    
    # 학습 (DART는 Early Stopping 비권장 - Loss가 튀기 때문)
    # 안정적인 학습을 위해 고정된 3000 라운드 수행
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,       
        valid_sets=[dtrain, dval],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.log_evaluation(period=100)
        ]
    )
    
    # DART의 경우 best_iteration이 0이거나 기록되지 않을 수 있음 -> 전체 트리 사용
    best_iter = model.best_iteration
    if best_iter is None or best_iter <= 0:
        print("Warning: Best iteration not found or 0. Using all trees for prediction.")
        best_iter = model.num_trees()

    val_pred = model.predict(X_val[features], num_iteration=best_iter)
    rmse = np.sqrt(mean_squared_error(np.expm1(X_val['target']), np.expm1(val_pred)))
    print(f"\n>> 'The Hidden King' CV RMSE: {rmse:,.0f}")

    # 7. Final Submission (Full Train)
    full_weights = np.ones(len(train))
    full_weights[train['contract_year'] >= 2023] = 3.0
    full_weights[train['contract_year'] == 2022] = 2.0
    
    dtrain_full = lgb.Dataset(train[features], label=train['target'], weight=full_weights)
    final_model = lgb.train(params, dtrain_full, num_boost_round=best_iter)
    
    test_preds = final_model.predict(test[features])
    pd.DataFrame({'target': np.expm1(test_preds).astype(int)}).to_csv('submissions/submission_exp18_hidden_king.csv', index=False)

if __name__ == "__main__":
    train_hidden_king()
