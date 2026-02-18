
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KDTree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# --------------------------------------------------------------------------------
# Exp08: Spatial Clustering (좌표 기반 공간 클러스터링 및 타겟 인코딩)
# --------------------------------------------------------------------------------
# 이 스크립트는 단순 행정구역(동)의 경계를 넘어, K-Means 알고리즘을 통해 실제 
# 좌표 기반의 '생활권 클러스터'를 생성합니다. 또한 클러스터별 평균 가격을 
# K-Fold Target Encoding 방식으로 변환하여 모델의 핵심 예측 변수로 활용합니다.
# --------------------------------------------------------------------------------

# 시각화 설정 (한글 폰트)
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
input_dir = 'data/processed'
raw_dir = 'data/raw'
submission_dir = 'submissions'

if not os.path.exists(submission_dir):
    os.makedirs(submission_dir)

train_path = os.path.join(input_dir, 'train_final.csv')
test_path = os.path.join(input_dir, 'test_final.csv')

# --------------------------------------------------------------------------------
# Helper Functions & Clustering Engineering
# --------------------------------------------------------------------------------

def minimal_clean_cols(df):
    """LGBM 호환을 위해 컬럼명의 특수문자를 정리합니다."""
    new_cols = []
    for col in df.columns:
        c = re.sub(r'[(),\[\]\s]', '_', col)
        new_cols.append(c)
    df.columns = new_cols
    return df

def feature_engineering_cluster(train, test):
    """공간 클러스터링 기반 파생 변수 생성 및 품질 변수 계승"""
    print("\n[Step 2] Feature Engineering (Clustering) Start...")
    
    # 처리를 위해 Train/Test 결합
    train_len = len(train)
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # 1. 공통 시계열 변수
    if '계약년월' in df.columns:
        df['contract_year'] = df['계약년월'].astype(str).str[:4].astype(int)
        df['contract_month'] = df['계약년월'].astype(str).str[4:6].astype(int)
    if '건축년도' in df.columns:
        df['building_age'] = df['contract_year'] - df['건축년도']
        df['building_age'] = df['building_age'].clip(lower=0)

    # 2. 품질 지표 (Exp07 계승)
    df['k-전체세대수'] = df['k-전체세대수'].replace(0, np.nan)
    if '주차대수' in df.columns:
        df['parking_per_unit'] = df['주차대수'] / df['k-전체세대수']
    if 'k-연면적' in df.columns:
        df['unit_area_avg'] = df['k-연면적'] / df['k-전체세대수']
    if 'k-전체세대수' in df.columns:
        df['log_total_units'] = np.log1p(df['k-전체세대수'])

    # 3. 교통 데이터 (KDTree 활용)
    sub_path = os.path.join(raw_dir, 'subway_feature.csv')
    try:
        sub_df = pd.read_csv(sub_path)
        tree = KDTree(sub_df[['위도', '경도']].values, metric='euclidean')
        dist, idx = tree.query(df[['좌표Y', '좌표X']].values, k=1)
        df['dist_to_subway'] = dist * 100
        line_col = '호선' if '호선' in sub_df.columns else ('노선명' if '노선명' in sub_df.columns else sub_df.columns[2])
        df['subway_line'] = sub_df.iloc[idx.flatten()][line_col].values
    except: pass

    # 4. 핵심: 공간 클러스터링 (K-Means)
    # 좌표(위도, 경도)를 표준화한 후 150개의 클러스터로 분할합니다.
    print("  - Running K-Means (K=150) to build living-area clusters...")
    coords = df[['좌표Y', '좌표X']].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    kmeans = KMeans(n_clusters=150, random_state=42, n_init=10)
    df['coord_cluster'] = kmeans.fit_predict(coords_scaled)
    
    # 5. 데이터 분리 및 다이어트
    train_res = df.iloc[:train_len].copy()
    test_res = df.iloc[train_len:].copy()
    
    drop_cols = ['주차대수', 'k-전체세대수', 'k-연면적', 'k-전체동수', '아파트명', '계약일', '계약년월', 
                 '시군구', '도로명', '번지', '본번', '부번']
    train_res.drop(columns=[c for c in drop_cols if c in train_res.columns], inplace=True, errors='ignore')
    test_res.drop(columns=[c for c in drop_cols if c in test_res.columns], inplace=True, errors='ignore')

    return train_res, test_res

def target_encoding_kfold(train, test, cluster_col='coord_cluster', target_col='target', n_splits=5):
    """과적합 방지를 위한 K-Fold Target Encoding"""
    print(f"\n[Step 4] Applying K-Fold Target Encoding (Cluster: {cluster_col})...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    mean_col = f'{cluster_col}_mean_price'
    train[mean_col] = np.nan
    
    # Train 데이터: Out-of-Fold 평균값 적용 (Leakage 방지)
    for tr_idx, val_idx in kf.split(train):
        X_tr = train.iloc[tr_idx]
        group_means = X_tr.groupby(cluster_col)[target_col].mean()
        train.loc[val_idx, mean_col] = train.loc[val_idx, cluster_col].map(group_means)
        
    # 데이터가 없는 클러스터는 전역 평균으로 보전
    global_mean = train[target_col].mean()
    train[mean_col] = train[mean_col].fillna(global_mean)
    
    # Test 데이터: 전체 Train 데이터의 평균값 적용
    test_means = train.groupby(cluster_col)[target_col].mean()
    test[mean_col] = test[cluster_col].map(test_means).fillna(global_mean)
    
    return train, test

# --------------------------------------------------------------------------------
# Main Training Function
# --------------------------------------------------------------------------------

def train_exp08_cluster():
    print("========== [Exp08] Spatial Clustering Training Start ==========")
    
    # 1. 데이터 로드 및 변수 생성
    print("\n[Step 1] Loading and Clustering Data...")
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    
    train, test = feature_engineering_cluster(train, test)
    
    # 2. 최신 데이터 필터링 (2017+)
    print("\n[Step 2] Filtering Recent Data (2017~)...")
    initial_len = len(train)
    train = train[train['contract_year'] >= 2017].reset_index(drop=True)
    print(f"Data Filtered: {initial_len} -> {len(train)}")

    # 3. 전처리 및 타겟 로그 변환
    print("\n[Step 3] Preprocessing and Log Transform...")
    train = minimal_clean_cols(train)
    test = minimal_clean_cols(test)
    
    if train['target'].max() > 100:
        train['target'] = np.log1p(train['target'])

    # 4. 타겟 인코딩 적용 (반드시 로그 변환 후 수행)
    train, test = target_encoding_kfold(train, test, cluster_col='coord_cluster', target_col='target')
    
    # 범주형 변수 인코딩
    le = LabelEncoder()
    obj_cols = train.select_dtypes(include=['object']).columns
    for col in obj_cols:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    # 5. 검증 전략 (Shuffle K-Fold)
    print("\n[Step 6] Validation Strategy: Shuffle K-Fold (5 Splits)")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    features = [c for c in train.columns if c != 'target']
    
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
    
    # Cross Validation Loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(train[features], train['target'])):
        X_tr, X_val = train.iloc[train_idx][features], train.iloc[val_idx][features]
        y_tr, y_val = train.iloc[train_idx]['target'], train.iloc[val_idx]['target']
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=10000,
            valid_sets=[dtrain, dvalid],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
        )
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(test[features]) / 5
        
        rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_pred)))
        print(f"Fold {fold+1} RMSE: {rmse:,.0f}")
        
    # 최종 결과 출력
    total_rmse = np.sqrt(mean_squared_error(np.expm1(train['target']), np.expm1(oof_preds)))
    print(f"\n>> Final Exp08 CV RMSE: {total_rmse:,.0f}")
    
    # 6. 변수 중요도 확인
    imp = pd.DataFrame({'Feature': features, 'Gain': model.feature_importance('gain')}).sort_values('Gain', ascending=False)
    print("\n[Top 10 Features (Gain)]")
    print(imp.head(10))
    
    # 7. 제출 파일 생성
    sub_path = os.path.join(submission_dir, 'submission_exp08_spatial.csv')
    pd.DataFrame({'target': np.expm1(test_preds)}).to_csv(sub_path, index=False)
    print(f"\n[Step 7] Submission file saved: {sub_path}")

if __name__ == "__main__":
    train_exp08_cluster()
