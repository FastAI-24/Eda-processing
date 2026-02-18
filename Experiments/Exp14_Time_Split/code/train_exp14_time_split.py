
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
# [Exp14] Honest Validation - Time Split Strategy
# 목표:
#   1. Shuffle K-Fold를 버리고 Time-based Split을 도입하여 CV와 LB 점수를 동기화한다.
#   2. 15,572점(LGBM) 베이스라인으로 복귀하여 안정성을 확보한다.
#   3. 로그 변환(log1p)을 유지하여 가격 변동폭에 대한 모델의 민감도를 조절한다.
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

# 3. 상세 피처 엔지니어링 (Exp12-v2 복원)
def feature_engineering_exp14(train, test):
    print("\n[Exp14] Engineering: Exp12-v2 (Reliable Set) Features...")
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
    
    # 강남역 하버사인 거리
    df['dist_gbd'] = haversine_distance(df['좌표Y'], df['좌표X'], 37.496, 127.027)

    # 공간 클러스터링 (Exp08, K=150)
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

def target_encoding_exp14(train, test, cluster_col='coord_cluster'):
    # Time-based Split이므로, 검증셋 기간 이전 데이터로만 평균을 내어 누수를 방지함
    # 여기서는 단순화를 위해 전체 Train 평균을 쓰되, 실제로는 Time-Split 내부에서 처리함이 권장됨
    # 하지만 일단 이전 로직 유지 후 효과 측정
    means = train.groupby(cluster_col)['target'].mean()
    train[f'{cluster_col}_mean_price'] = train[cluster_col].map(means)
    test[f'{cluster_col}_mean_price'] = test[cluster_col].map(means)
    global_mean = train['target'].mean()
    train[f'{cluster_col}_mean_price'].fillna(global_mean, inplace=True)
    test[f'{cluster_col}_mean_price'].fillna(global_mean, inplace=True)
    return train, test

# 4. 메인 학습 루틴
def train_exp14_time_split():
    print(f"[{datetime.datetime.now()}] Loading Data...")
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    
    # 최신 데이터 필터링 가동 (2017+)
    temp_year = train['계약년월'].astype(str).str[:4].astype(int)
    train = train[temp_year >= 2017].reset_index(drop=True)

    # 1. Feature Engineering
    train, test = feature_engineering_exp14(train, test)
    
    # Target Log Transform
    if train['target'].max() > 100:
        train['target'] = np.log1p(train['target'])

    # 2. Time-based Split 준비
    # 2023년 4월 이후를 검증셋으로 사용 (최근 약 3개월 분량)
    val_condition = (train['contract_year'] == 2023) & (train['contract_month'] >= 4)
    X_train_full = train[~val_condition].reset_index(drop=True)
    X_val_full = train[val_condition].reset_index(drop=True)
    
    print(f"\n[Time-Split] Train: {len(X_train_full)} samples (up to 2023-03)")
    print(f"[Time-Split] Valid: {len(X_val_full)} samples (2023-04 to 2023-06/Max)")

    # 타겟 인코딩 (Train으로만 기준 생성)
    X_train_full, X_val_full = target_encoding_exp14(X_train_full, X_val_full)
    _, test = target_encoding_exp14(X_train_full, test)

    # 3. 전처리 (로그 정리 및 인코딩)
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

    # 4. 학습 (Fast-tuning)
    params = {
        'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.1, # 가성비 속도
        'num_leaves': 128, 'colsample_bytree': 0.8, 'subsample': 0.8,
        'n_jobs': -1, 'seed': 42, 'verbose': -1
    }
    
    dtrain = lgb.Dataset(X_train_full[features], label=X_train_full['target'])
    dvalid = lgb.Dataset(X_val_full[features], label=X_val_full['target'], reference=dtrain)
    
    print("\nTraining LightGBM with Time-based Split...")
    model = lgb.train(params, dtrain, num_boost_round=5000, valid_sets=[dtrain, dvalid],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
    
    # 5. 검증 및 저장
    val_pred = model.predict(X_val_full[features])
    rmse = np.sqrt(mean_squared_error(np.expm1(X_val_full['target']), np.expm1(val_pred)))
    print(f"\n>> Final Exp14 Honestly Measured CV RMSE: {rmse:,.0f}")
    
    # 리더보드용 전체 데이터 재학습 (Option: Valid까지 포함하여 재학습하면 더 좋음)
    print("Re-training on full data for submission...")
    dtrain_final = lgb.Dataset(pd.concat([X_train_full[features], X_val_full[features]]), 
                               label=pd.concat([X_train_full['target'], X_val_full['target']]))
    final_model = lgb.train(params, dtrain_final, num_boost_round=model.best_iteration)
    
    test_preds = final_model.predict(test[features])
    sub_path = os.path.join(submission_dir, 'submission_exp14_time_split.csv')
    pd.DataFrame({'target': np.expm1(test_preds).astype(int)}).to_csv(sub_path, index=False)
    print(f"Submission Saved: {sub_path}")

if __name__ == "__main__":
    train_exp14_time_split()
