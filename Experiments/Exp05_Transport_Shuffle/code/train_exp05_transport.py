
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
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------------------
# Exp05: Transport Features (지하철 및 버스 정보 기반 입지 가치 분석)
# --------------------------------------------------------------------------------
# 이 스크립트는 지하철역과의 거리, 인근 버스 정류장 수 등 교통 편의성 데이터를 
# 파생 변수로 추가하여 모델의 예측력을 높입니다. 공간적 검색을 위해 KDTree 알고리즘을 
# 사용하며, 입지 가치가 아파트 가격에 미치는 영향을 집중적으로 학습합니다.
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
# Helper Functions & Feature Engineering
# --------------------------------------------------------------------------------

def minimal_clean_cols(df):
    """LGBM 호환을 위해 컬럼명의 특수문자를 정리합니다."""
    new_cols = []
    for col in df.columns:
        c = re.sub(r'[(),\[\]\s]', '_', col)
        new_cols.append(c)
    df.columns = new_cols
    return df

def feature_engineering_advanced(train, test):
    """고급 파생 변수 생성 (건물 연식 및 교통 정보)"""
    print("\n[Step 2] Advanced Feature Engineering Start...")
    
    # 1. 건물 연식 (Ref. Exp03)
    if '건축년도' in train.columns and '계약년월' in train.columns:
        print("  - Adding Building Age...")
        train['contract_year'] = train['계약년월'].astype(str).str[:4].astype(int)
        test['contract_year'] = test['계약년월'].astype(str).str[:4].astype(int)
        train['building_age'] = train['contract_year'] - train['건축년도']
        test['building_age'] = test['contract_year'] - test['건축년도']
        train['building_age'] = train['building_age'].clip(lower=0)
        test['building_age'] = test['building_age'].clip(lower=0)
        
    # 2. 교통 변수 (지하철/버스)
    print("  - Integrating Transport Data (Subway/Bus)...")
    sub_path = os.path.join(raw_dir, 'subway_feature.csv')
    bus_path = os.path.join(raw_dir, 'bus_feature.csv')
    
    try:
        sub_df = pd.read_csv(sub_path)
        # 버스 데이터는 인코딩 이슈가 있을 수 있으므로 예외 처리
        try:
            bus_df = pd.read_csv(bus_path, encoding='cp949')
        except:
            bus_df = pd.read_csv(bus_path) 
    except Exception as e:
        print(f"Error loading transport data: {e}")
        return train, test

    # --- 지하철 관련 변수 (KDTree 활용) ---
    # 지하철 좌표 데이터 구축
    sub_coords = sub_df[['위도', '경도']].values
    tree_sub = KDTree(sub_coords, metric='euclidean')
    
    def add_subway_features(src_df):
        # 아파트 좌표 (Y:위도, X:경도)
        src_coords = src_df[['좌표Y', '좌표X']].values
        dist, idx = tree_sub.query(src_coords, k=1) # 가장 가까운 역 1개 찾기
        
        # 최단 거리 (좌표 단위를 km로 근사하기 위해 * 100 적용)
        src_df['dist_to_subway'] = dist * 100 
        
        # 해당 지하철역의 호선 정보 가져오기
        nearest_indices = idx.flatten()
        line_col = '호선' if '호선' in sub_df.columns else ('노선명' if '노선명' in sub_df.columns else sub_df.columns[2])
        src_df['subway_line'] = sub_df.iloc[nearest_indices][line_col].values
        return src_df
        
    train = add_subway_features(train)
    test = add_subway_features(test)
    print("  >> Subway distance and line features added.")
    
    # --- 버스 관련 변수 ---
    if 'Y' in bus_df.columns and 'X' in bus_df.columns:
        bus_coords = bus_df[['Y', 'X']].values
        tree_bus = KDTree(bus_coords, metric='euclidean')
        
        def add_bus_features(src_df):
            src_coords = src_df[['좌표Y', '좌표X']].values
            
            # (1) 가장 가까운 버스 정류장까지의 거리
            dist, _ = tree_bus.query(src_coords, k=1)
            src_df['dist_to_bus'] = dist * 100
            
            # (2) 반경 500m 이내 버스 정류장 개수 (임계값 0.005 적용)
            counts = tree_bus.query_radius(src_coords, r=0.005, count_only=True)
            src_df['bus_count_500'] = counts
            return src_df
            
        train = add_bus_features(train)
        test = add_bus_features(test)
        print("  >> Bus distance and density features added.")
    else:
        print("  >> Warning: Bus coordinate columns (X, Y) not found.")

    return train, test

# --------------------------------------------------------------------------------
# Main Training Function
# --------------------------------------------------------------------------------

def train_exp05_transport():
    print("========== [Exp05] Transport Feature Training Start ==========")
    
    # 1. 데이터 로드 및 변수 생성
    print("\n[Step 1] Loading Data...")
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    
    train, test = feature_engineering_advanced(train, test)
    
    # 3. 전처리 (Cleansing & Log Transform)
    print("\n[Step 3] Preprocessing...")
    train = minimal_clean_cols(train)
    test = minimal_clean_cols(test)
    
    # Target 로그 변환
    if train['target'].max() > 100:
        print(">> Applying Log Transform to Target...")
        train['target'] = np.log1p(train['target'])

    # 불필요한 텍스트 컬럼 제거
    print("\n[Step 4] Dropping Text Columns...")
    drop_texts = ['아파트명', '시군구', '도로명', '번지', '본번', '부번', '계약년월', '계약일']
    drop_texts_cleaned = [re.sub(r'[(),\[\]\s]', '_', c) for c in drop_texts]
    train.drop(columns=[c for c in drop_texts_cleaned if c in train.columns], inplace=True)
    test.drop(columns=[c for c in drop_texts_cleaned if c in test.columns], inplace=True)
    
    # 범주형 변수 인코딩 (Label Encoding)
    print("\n[Step 5] Label Encoding Categorical Features...")
    le = LabelEncoder()
    obj_cols = train.select_dtypes(include=['object']).columns
    print(f"Object columns: {list(obj_cols)}")
    
    for col in obj_cols:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    # 4. 검증 전략: Shuffle K-Fold (5 Splits)
    # 지리적 특성에 따른 가격 레벨을 광범위하게 학습하기 위해 셔플 방식을 채택합니다.
    print("\n[Step 6] Validation Strategy: Shuffle K-Fold (5 Splits)")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    features = [c for c in train.columns if c != 'target']
    
    # 하이퍼파라미터 설정
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
        
    # 최종 성능 산출
    total_rmse = np.sqrt(mean_squared_error(np.expm1(train['target']), np.expm1(oof_preds)))
    print(f"\n>> Final Exp05 CV RMSE (Real Scale): {total_rmse:,.0f}")
    
    # 5. 변수 중요도 (Feature Importance) 시각화 및 출력
    imp = pd.DataFrame({'Feature': features, 'Gain': model.feature_importance('gain')}).sort_values('Gain', ascending=False)
    print("\n[Top 10 Features (Gain)]")
    print(imp.head(10))
    
    # 6. 제출 파일 생성
    sub_path = os.path.join(submission_dir, 'submission_exp05_transport.csv')
    pd.DataFrame({'target': np.expm1(test_preds)}).to_csv(sub_path, index=False)
    print(f"\n[Step 7] Submission file saved: {sub_path}")

if __name__ == "__main__":
    train_exp05_transport()
