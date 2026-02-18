
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------------------
# Exp01: Baseline Model with Coordinates (좌표 기반 베이스라인)
# --------------------------------------------------------------------------------
# 이 스크립트는 Exp01의 학습 및 추론 코드를 정리한 버전입니다.
# 주요 기능:
# 1. 데이터 로드 및 전처리 (브랜드명 추출, 불필요한 컬럼 제거)
# 2. 로그 변환 (Target, Area)
# 3. Target Encoding (구, 동)
# 4. LightGBM 모델 학습 (Shuffle Split 검증 사용 - Exp01 당시 전략)
# 5. 결과 저장 및 제출 파일 생성
# --------------------------------------------------------------------------------

# 시각화 설정 (한글 폰트)
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
input_dir = 'data/processed'
output_dir = 'logs'
submission_dir = 'submissions'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(submission_dir):
    os.makedirs(submission_dir)

train_path = os.path.join(input_dir, 'train_final.csv')
test_path = os.path.join(input_dir, 'test_final.csv')

# --------------------------------------------------------------------------------
# Helper Functions (보조 함수 정의)
# --------------------------------------------------------------------------------

def minimal_clean_cols(df):
    """
    컬럼명에서 LightGBM이 처리하기 어려운 특수문자만 최소한으로 제거합니다.
    (), [], 공백, 콤마 등을 언더스코어(_)로 변환합니다.
    """
    new_cols = []
    for col in df.columns:
        # (), [], ,, 공백 등을 _로 치환
        c = re.sub(r'[(),\[\]\s]', '_', col)
        new_cols.append(c)
    df.columns = new_cols
    return df

def target_encode(train_df, test_df, col, target='target', n_splits=5):
    """
    Target Encoding을 수행하는 함수입니다.
    K-Fold 방식을 사용하여 Data Leakage를 방지합니다.
    """
    if col not in train_df.columns: return train_df, test_df
    
    # 새로운 컬럼 생성
    train_df[f'{col}_target'] = np.nan
    test_df[f'{col}_target'] = np.nan
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # K-Fold 기반 Encoding (Train)
    for tr_idx, val_idx in kf.split(train_df):
        X_tr, X_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        means = X_tr.groupby(col)[target].mean()
        train_df.loc[val_idx, f'{col}_target'] = X_val[col].map(means)
        
    # 결측치(Fold에서 보지 못한 값)는 전체 평균으로 대체
    global_mean = train_df[target].mean()
    train_df[f'{col}_target'].fillna(global_mean, inplace=True)
    
    # Test 데이터 Encoding (전체 Train 평균 사용)
    global_means = train_df.groupby(col)[target].mean()
    test_df[f'{col}_target'] = test_df[col].map(global_means)
    test_df[f'{col}_target'].fillna(global_mean, inplace=True)
    
    return train_df, test_df

# 상위 브랜드 아파트 리스트
top_brands = ['래미안', '자이', '힐스테이트', '아이파크', '푸르지오', '롯데캐슬', '더샵', 'e편한세상', '이편한세상', '아크로', '디에이치', '꿈의숲']

def check_brand(name):
    """아파트명에 상위 브랜드 키워드가 포함되어 있는지 확인합니다."""
    if pd.isna(name): return 0
    for brand in top_brands:
        if brand in str(name):
            return 1
    return 0

# --------------------------------------------------------------------------------
# Main Training Function
# --------------------------------------------------------------------------------

def train_refined_baseline_final():
    print("========== [Exp01] Baseline Model Training Start ==========")
    
    # 1. 데이터 로드
    print("\n[Step 1] Loading Data...")
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    print(f"Train Shape: {train.shape}, Test Shape: {test.shape}")

    # 2. 파생 변수 생성 (Top Brand)
    # '아파트명' 컬럼이 존재할 때만 수행
    print("\n[Step 2] Creating Features (Top Brand)...")
    if '아파트명' in train.columns:
        train['is_top_brand'] = train['아파트명'].apply(check_brand)
        test['is_top_brand'] = test['아파트명'].apply(check_brand)
        
    # 3. 불필요한 텍스트 컬럼 제거
    # 모델 학습에 직접 사용하기 어렵거나, 카디널리티가 너무 높은 텍스트 컬럼 제거
    print("\n[Step 3] Dropping Explicit Text Columns...")
    drop_texts = ['아파트명', '시군구', '도로명', '번지', '본번', '부번', '계약년월', '계약일']
    train.drop(columns=[c for c in drop_texts if c in train.columns], inplace=True)
    test.drop(columns=[c for c in drop_texts if c in test.columns], inplace=True)
    
    # 4. 컬럼명 정리 (LightGBM 호환성)
    print("\n[Step 4] Cleaning Column Names...")
    train = minimal_clean_cols(train)
    test = minimal_clean_cols(test)
    
    # 5. 로그 변환 (Log Transformation)
    # Target 값과 면적(Area)은 분포가 치우쳐 있을 가능성이 높으므로 로그 변환 수행
    print("\n[Step 5] Log Transformation (Target & Area)...")
    train['target'] = np.log1p(train['target'])
    
    # 컬럼명이 한글일 경우 '_㎡_' 등으로 변환되었을 수 있음. 확인 후 변환.
    area_col = [c for c in train.columns if '전용면적' in c]
    if area_col:
        col_name = area_col[0] # '전용면적_㎡_' 예상
        print(f"Log transforming area column: {col_name}")
        train[col_name] = np.log1p(train[col_name])
        test[col_name] = np.log1p(test[col_name])
        
    # 6. Target Encoding
    # 구, 동 정보는 범주형 변수이므로 Target Encoding을 통해 수치형으로 변환
    print("\n[Step 6] Target Encoding (Gu, Dong)...")
    if 'Gu_encoded' in train.columns:
        train, test = target_encode(train, test, 'Gu_encoded')
    if 'Dong_encoded' in train.columns:
        train, test = target_encode(train, test, 'Dong_encoded')
        
    # 7. 나머지 범주형 변수 Label Encoding
    # 위에서 처리되지 않은 나머지 object 타입 컬럼들을 Label Encoding
    print("\n[Step 7] Label Encoding Remaining Objects...")
    obj_cols = train.select_dtypes(include=['object']).columns.tolist()
    print(f"Object Cols to Encode: {obj_cols}")
    
    le = LabelEncoder()
    for col in obj_cols:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        
    # 8. 모델 학습 (LightGBM)
    print("\n[Step 8] Final Training (LightGBM)...")
    features = [c for c in train.columns if c != 'target']
    X = train[features]
    y = train['target']
    X_test = test[features]

    # Validation Strategy: Exp01에서는 Shuffle Split (Random Split) 사용
    # 주의: 시계열 데이터에서는 적절하지 않을 수 있으나, 베이스라인 구축을 위해 사용함.
    if 'days_since' in X.columns:
        # 시간순 정렬 후 마지막 20%를 검증 데이터로 사용 (Time-based Split 시도 흔적)
        # 하지만 Exp01의 주요 이슈는 Shuffle이었음. 여기서는 코드에 있는 대로 Time-based Split 흉내(sort 후 split)를 유지.
        X = X.sort_values('days_since')
        y = y.loc[X.index]
        valid_size = int(len(X) * 0.2)
        X_train, X_valid = X.iloc[:-valid_size], X.iloc[-valid_size:]
        y_train, y_valid = y.iloc[:-valid_size], y.iloc[-valid_size:]
    else:
        # days_since가 없다면 일반적인 random split
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 128,
        'n_jobs': -1,
        'seed': 42,
        'verbose': -1
    }
    
    model = lgb.train(
        params, 
        dtrain, 
        num_boost_round=10000, 
        valid_sets=[dtrain, dvalid],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
    )
    
    # 9. 결과 평가 및 시각화
    print("\n[Step 9] Evaluation & Visualization...")
    valid_pred = model.predict(X_valid)
    # Log 변환된 Target을 다시 원래 스케일로 복원 (expm1)하여 RMSE 계산
    rmse = np.sqrt(mean_squared_error(np.expm1(y_valid), np.expm1(valid_pred)))
    print(f"\nFinal Validation RMSE (Real Scale): {rmse:,.2f}")
    
    # Feature Importance 저장
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importance(importance_type='gain')
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance.head(20))
    plt.title(f'Feature Importance (Exp01) - RMSE: {rmse:,.0f}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_final.png'))
    print(f"Feature Importance saved to {output_dir}")
    
    # 10. 제출 파일 생성
    print("\n[Step 10] Generating Submission...")
    test_pred = np.expm1(model.predict(X_test))
    sub = pd.DataFrame({'target': test_pred})
    sub.to_csv(os.path.join(submission_dir, 'submission_final_lgbm.csv'), index=False)
    print("Submission Saved.")

if __name__ == "__main__":
    train_refined_baseline_final()
