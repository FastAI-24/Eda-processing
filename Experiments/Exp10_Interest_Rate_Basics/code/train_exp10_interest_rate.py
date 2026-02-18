
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
# [Exp10] Back to Basics + Macro Strategy
# 목표:
#   1. 최신 데이터(2017~2023)만 사용하여 과거 데이터 노이즈 차단. (Regime Shift 대응)
#   2. '기준금리(Base Rate)' 변수를 추가하여 시계열 변동성(유동성) 설명력 확보.
#   3. 검증된 공간 피처(Cluster)와 품질 피처(Parking, Density) 통합.
#   4. '상위 브랜드(Top Brand)' 여부 추가.
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

# 2. 한국은행 기준금리 매핑 데이터 (2017-2023Focus)
# ---------------------------------------------
def get_base_rate_map():
    base_rate_dict = {
        2017: {1: 1.25, 2: 1.25, 3: 1.25, 4: 1.25, 5: 1.25, 6: 1.25, 7: 1.25, 8: 1.25, 9: 1.25, 10: 1.25, 11: 1.50, 12: 1.50},
        2018: {1: 1.50, 2: 1.50, 3: 1.50, 4: 1.50, 5: 1.50, 6: 1.50, 7: 1.50, 8: 1.50, 9: 1.50, 10: 1.50, 11: 1.75, 12: 1.75},
        2019: {1: 1.75, 2: 1.75, 3: 1.75, 4: 1.75, 5: 1.75, 6: 1.75, 7: 1.50, 8: 1.50, 9: 1.50, 10: 1.25, 11: 1.25, 12: 1.25},
        2020: {1: 1.25, 2: 1.25, 3: 0.75, 4: 0.75, 5: 0.50, 6: 0.50, 7: 0.50, 8: 0.50, 9: 0.50, 10: 0.50, 11: 0.50, 12: 0.50},
        2021: {1: 0.50, 2: 0.50, 3: 0.50, 4: 0.50, 5: 0.50, 6: 0.50, 7: 0.50, 8: 0.75, 9: 0.75, 10: 0.75, 11: 1.00, 12: 1.00},
        2022: {1: 1.25, 2: 1.25, 3: 1.25, 4: 1.50, 5: 1.75, 6: 1.75, 7: 2.25, 8: 2.50, 9: 2.50, 10: 3.00, 11: 3.25, 12: 3.25},
        2023: {1: 3.50, 2: 3.50, 3: 3.50, 4: 3.50, 5: 3.50, 6: 3.50, 7: 3.50, 8: 3.50, 9: 3.50, 10: 3.50, 11: 3.50, 12: 3.50},
    }
    return base_rate_dict

def minimal_clean_cols(df):
    new_cols = []
    for col in df.columns:
        c = re.sub(r'[(),\[\]\s]', '_', col)
        new_cols.append(c)
    df.columns = new_cols
    return df

# 3. 상세 피처 엔지니어링
# --------------------
def feature_engineering_exp10(train, test):
    print("\n[Exp10] Applying Interest Rate + Quality + Cluster Features (Recent Data Only)...")
    
    train_len = len(train)
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # A. 날짜 변수 추출
    if '계약년월' in df.columns:
        df['contract_year'] = df['계약년월'].astype(str).str[:4].astype(int)
        df['contract_month'] = df['계약년월'].astype(str).str[4:6].astype(int)
    
    # B. 거시경제: 기준금리(Base Rate) 추가
    print("  -> Adding Base Rate (Interest Rate)...")
    br_map = get_base_rate_map()
    df['base_rate'] = df.apply(lambda x: br_map.get(x['contract_year'], {}).get(x['contract_month'], 2.0), axis=1)

    # C. 품질 변수 (Exp07)
    print("  -> Adding Quality Ratio Features (Parking, Area, Density)...")
    df['k-전체세대수'] = df['k-전체세대수'].replace(0, np.nan)
    df['k-전체동수'] = df['k-전체동수'].replace(0, np.nan)
    
    if '주차대수' in df.columns:
        df['parking_per_unit'] = df['주차대수'] / df['k-전체세대수']
    if 'k-연면적' in df.columns:
        df['unit_area_avg'] = df['k-연면적'] / df['k-전체세대수']
    if 'k-전체동수' in df.columns:
        df['complex_density'] = df['k-전체세대수'] / df['k-전체동수']

    # 상위 브랜드(Top Brand) 여부 추가
    # 도약, 자이, 래미안, 힐스테이트, 푸르지오, 롯데캐슬, 아이파크, e편한세상, 더샵, SK뷰
    top_brands = ['래미안', '자이', '힐스테이트', '아이파크', '푸르지오', 'e편한세상', '롯데캐슬', '더샵', 'SK뷰', '포레나']
    if '아파트명' in df.columns:
        df['is_top_brand'] = df['아파트명'].apply(lambda x: 1 if any(brand in str(x) for brand in top_brands) else 0)

    # D. 공간 클러스터링 (Exp08, K=150)
    print("  -> Generating Spatial Clusters (K=150)...")
    coords = df[['좌표Y', '좌표X']].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    kmeans = KMeans(n_clusters=150, random_state=42, n_init=10)
    df['coord_cluster'] = kmeans.fit_predict(coords_scaled)
    
    # E. 다이어트: 불필요한 고차원 범주형/원본 변수 제거
    drop_cols = ['주차대수', 'k-전체세대수', 'k-연면적', 'k-전체동수', '아파트명', '계약일', '계약년월', 
                 '시군구', '도로명', '번지', '본번', '부번']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')
    
    train = df.iloc[:train_len].copy()
    test = df.iloc[train_len:].copy()
    return train, test

def target_encoding_kfold(train, test, cluster_col='coord_cluster', target_col='target', n_splits=5):
    print(f"  -> Applying K-Fold Target Encoding (Cluster Base Value)...")
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

# 4. 메인 학습 루틴
# ---------------
def train_exp10_interest():
    print("Loading Data...")
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    
    # [중요] 최신 데이터 필터링 (2017년 이후만 사용)
    # Regime Shift 대응을 위해 노이지한 과거 데이터는 과감히 삭제
    if '계약년월' not in train.columns:
        train['contract_year'] = train['계약년월'].astype(str).str[:4].astype(int)
    else:
        # 이미 있다면 년도 추출은 feature engineering에서 수행됨
        pass
    
    # 임시 년도 추출 후 필터링
    temp_year = train['계약년월'].astype(str).str[:4].astype(int)
    train = train[temp_year >= 2017].reset_index(drop=True)
    print(f"Recent Data Filter (2017+): {len(train)} samples remained.")

    # 1. Feature Engineering
    train, test = feature_engineering_exp10(train, test)
    
    # 2. 전처리
    train = minimal_clean_cols(train)
    test = minimal_clean_cols(test)
    if train['target'].max() > 100: train['target'] = np.log1p(train['target'])

    # 타겟 인코딩
    train, test = target_encoding_kfold(train, test)
    
    # 범주형 인코딩
    le = LabelEncoder()
    col_list = train.select_dtypes(include=['object']).columns
    for col in col_list:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    # 3. 학습 및 검증
    print("\n[Validation Strategy] Shuffle K-Fold (5 Splits) on Recent Data")
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
    print(f"\n>> Overall Exp10 CV RMSE: {total_rmse:,.0f}")
    
    # Feature Importance
    imp = pd.DataFrame({'Feature': features, 'Gain': model.feature_importance('gain')}).sort_values('Gain', ascending=False)
    print("\n[Exp10 Feature Importance - Top 10]")
    print(imp.head(10))
    
    # 4. 결과 저장
    sub_path = os.path.join(submission_dir, 'submission_exp10_interest.csv')
    pd.DataFrame({'target': np.expm1(test_preds)}).to_csv(sub_path, index=False)
    print(f"Submission Saved: {sub_path}")

if __name__ == "__main__":
    train_exp10_interest()
