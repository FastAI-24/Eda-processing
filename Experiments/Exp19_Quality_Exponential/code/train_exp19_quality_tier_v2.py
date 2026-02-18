
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import re
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import datetime

# =============================================================================
# [Exp19] The Quality & Tier (v2: Exponential Decay)
# 전략:
#   1. 관리 품질(Quality): is_managed, k_info_count, has_parking_info
#   2. 층수 계급(Tier): floor_score (데이터 기반 구간별 가격 점수)
#   3. Weighting: Exponential Time Decay (반감기 2년 = 730일) 적용
#      -> 임의적 가중치 배제, 수학적 감쇄 모델 도입
#   4. Validation: 2023년 상반기(1~6월) 전체 검증
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

# 3. Feature Engineering
def feature_engineering_exp19_v2(train, test):
    print("\n[Exp19 v2] Engineering: Quality & Tier Features + Exp Decay Weight Prep...")
    train_len = len(train)
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # ── [1] Time Features ──
    df['contract_year'] = df['계약년월'].astype(str).str[:4].astype(int)
    df['contract_month'] = df['계약년월'].astype(str).str[4:6].astype(int)
    # days_since (이미 있음)
    df['building_age'] = (df['contract_year'] - df['건축년도']).clip(lower=0)

    # ── [2] Quality Features (k-정보 활용) ──
    k_cols = ['k-전체세대수', 'k-전체동수', 'k-연면적', 
              'k-복도유형', 'k-난방방식', 'k-단지분류(아파트,주상복합등등)']
    
    # A. 관리 여부 (0/1)
    df['is_managed'] = df['k-전체세대수'].notnull().astype(int)
    
    # B. 정보 밀도 (0~6)
    df['k_info_count'] = df[k_cols].notnull().sum(axis=1)
    
    # C. 주차 정보 신뢰도 (RF추정 vs 실제)
    df['has_parking_info'] = df['is_managed'] 
    
    # D. 세대당 주차대수 (NaN 처리 없음 -> LGBM에 맡김)
    df['parking_per_unit'] = df['주차대수'] / df['k-전체세대수']

    # ── [3] Tier Features (층수 구간화) ──
    # 데이터 기반 변곡점 (15, 25, 37)
    # Floor Score: 구간별 가중치 부여 (가설 검증 결과 기반 대략적 비율)
    conditions = [
        (df['층'] <= 15),
        (df['층'] > 15) & (df['층'] <= 25),
        (df['층'] > 25) & (df['층'] <= 37),
        (df['층'] > 37)
    ]
    # 구간별 매핑 점수 (임의가 아닌 데이터 trend 반영)
    scores = [1.0, 1.2, 1.5, 2.0] 
    df['floor_score'] = np.select(conditions, scores, default=1.0)
    
    # ── [4] Location Features ──
    print("  -> Calculating Distances...")
    try:
        sub_df = pd.read_csv(os.path.join(raw_dir, 'subway_feature.csv'))
        sub_tree = KDTree(sub_df[['위도', '경도']].values)
        dist_s, _ = sub_tree.query(df[['좌표Y', '좌표X']].values, k=1)
        df['dist_to_subway'] = dist_s * 100
    except: pass
    
    df['dist_gbd'] = haversine_distance(df['좌표Y'], df['좌표X'], 37.496, 127.027)

    # ── [5] Spatial Clustering ──
    print("  -> Generating Spatial Clusters...")
    coords = df[['좌표Y', '좌표X']].values
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=150, random_state=42, n_init=10)
    df['coord_cluster'] = kmeans.fit_predict(scaler.fit_transform(coords))
    
    # ── [6] Diet (불필요 컬럼 제거) ──
    df['log_total_units'] = np.log1p(df['k-전체세대수'].fillna(0))
    
    drop_cols = ['k-전체세대수', 'k-연면적', 'k-전체동수', '아파트명', '계약일', '계약년월', 
                 '시군구', '도로명', '번지', '본번', '부번', '주차대수']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')
    
    train = df.iloc[:train_len].copy()
    test = df.iloc[train_len:].copy()
    return train, test

def target_encoding_exp19(train, test, cluster_col='coord_cluster'):
    means = train.groupby(cluster_col)['target'].mean()
    global_mean = train['target'].mean()
    train[f'{cluster_col}_mean_price'] = train[cluster_col].map(means)
    test[f'{cluster_col}_mean_price'] = test[cluster_col].map(means)
    train[f'{cluster_col}_mean_price'].fillna(global_mean, inplace=True)
    test[f'{cluster_col}_mean_price'].fillna(global_mean, inplace=True)
    return train, test

# 4. 학습 루틴 (v2: Exp Decay)
def train_exp19_v2():
    print(f"[{datetime.datetime.now()}] Exp19 v2: Quality & Tier + Exp Decay Start")
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    
    # 2017년 이후 (Exp15 Base)
    temp_year = train['계약년월'].astype(str).str[:4].astype(int)
    train = train[temp_year >= 2017].reset_index(drop=True)

    # Feature Engineering
    train, test = feature_engineering_exp19_v2(train, test)
    
    if train['target'].max() > 100:
        train['target'] = np.log1p(train['target'])

    # Validation Strategy (Extended Window: 2023.01 ~ 2023.06)
    val_condition = (train['contract_year'] == 2023) & (train['contract_month'] >= 1)
    X_train_full = train[~val_condition].reset_index(drop=True)
    X_val_full = train[val_condition].reset_index(drop=True)
    
    print(f"\n[Validation] Train: {len(X_train_full)} samples")
    print(f"[Validation] Valid: {len(X_val_full)} samples (2023.01 ~ 2023.06)")

    # ─────────────────────────────────────────────────────────────────────────────
    # [Weighting Strategy] Exponential Time Decay (Half-Life: 2 Years = 730 Days)
    # ─────────────────────────────────────────────────────────────────────────────
    # W = exp( (days - max_days) * lambda )
    # lambda = ln(2) / Half_Life
    
    def get_exp_decay_weights(df, half_life_days=730):
        # 기준점: 현재 데이터셋의 가장 최근 날짜 (Test 포함 전체 기준이 좋으나, Train 내 상대적 중요도이므로 Train Max 사용)
        # 하지만 여기서는 Generalization을 위해 Global Max (2023.06 말) 기준으로 잡아도 됨.
        # 편의상 해당 DF의 max days_since 사용
        max_days = df['days_since'].max()
        decay_constant = np.log(2) / half_life_days
        
        # 날짜 차이 (음수)
        diff = df['days_since'] - max_days
        weights = np.exp(diff * decay_constant)
        
        return weights

    weights = get_exp_decay_weights(X_train_full, half_life_days=730)
    print(f"  -> Applied Exponential Decay Weights (Half-Life: 2 Years)")
    print(f"     Min Weight: {weights.min():.4f}, Max Weight: {weights.max():.4f}")

    # Encoding & Clean
    X_train_full, X_val_full = target_encoding_exp19(X_train_full, X_val_full)
    _, test = target_encoding_exp19(X_train_full, test)

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

    # LGBM Training
    params = {
        'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.02,
        'num_leaves': 128, 'colsample_bytree': 0.8, 'subsample': 0.8,
        'n_jobs': -1, 'seed': 42, 'verbose': -1
    }
    
    dtrain = lgb.Dataset(X_train_full[features], label=X_train_full['target'], weight=weights)
    dvalid = lgb.Dataset(X_val_full[features], label=X_val_full['target'], reference=dtrain)
    
    print(f"\nTraining Model on {len(features)} features...")
    model = lgb.train(params, dtrain, num_boost_round=15000, valid_sets=[dtrain, dvalid],
                      callbacks=[lgb.early_stopping(300), lgb.log_evaluation(1000)])
    
    val_pred = model.predict(X_val_full[features])
    rmse = np.sqrt(mean_squared_error(np.expm1(X_val_full['target']), np.expm1(val_pred)))
    print(f"\n>> Final Exp19 v2 CV RMSE: {rmse:,.0f}")
    
    # Final Re-training (Full Data)
    print("Re-training on FULL data for Submission...")
    full_train = pd.concat([X_train_full, X_val_full])
    
    # 전체 데이터 기준 Weight 재산출 (반감기 동일)
    full_weights = get_exp_decay_weights(full_train, half_life_days=730)
    
    dtrain_final = lgb.Dataset(full_train[features], label=full_train['target'], weight=full_weights)
    final_model = lgb.train(params, dtrain_final, num_boost_round=model.best_iteration)
    
    test_preds = final_model.predict(test[features])
    sub_path = os.path.join(submission_dir, 'submission_exp19_quality_tier_v2.csv')
    pd.DataFrame({'target': np.expm1(test_preds).astype(int)}).to_csv(sub_path, index=False)
    print(f"Submission Saved: {sub_path}")

    # Feature Importance
    imp_df = pd.DataFrame({'feature': features, 'gain': final_model.feature_importance(importance_type='gain')})
    imp_df = imp_df.sort_values('gain', ascending=False)
    imp_df.to_csv(os.path.join(submission_dir, 'feature_importance_exp19_v2.csv'), index=False)
    print("Feature Importance saved.")

if __name__ == "__main__":
    train_exp19_v2()
