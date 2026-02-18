
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import os
import re

# =============================================================================
# [Exp01] Baseline Model
# 목표: 결측된 좌표(78%)를 복원한 데이터로 가장 기본적인 LightGBM 모델 학습
# 특징: 
#   1. 복잡한 파생변수 없이 기본 컬럼(면적, 층, 건축년도 등)만 사용
#   2. Dong_encoded(동별 평균가) 사용 (가장 강력한 베이스라인 변수)
#   3. Shuffle K-Fold (5-Fold) 사용 -> 시계열 특성 무시 (과적합 위험 존재)
# =============================================================================

# 1. 데이터 로드 설멍
# -----------------
input_dir = 'data/processed'
train_path = os.path.join(input_dir, 'train_final.csv') 
test_path = os.path.join(input_dir, 'test_final.csv')

def minimal_clean_cols(df):
    """
    LightGBM은 특수문자(:, /, space 등)가 포함된 컬럼명을 싫어하므로
    이를 _(언더바)로 치환해주는 함수.
    """
    new_cols = []
    for col in df.columns:
        c = re.sub(r'[(),\[\]\s]', '_', col)
        new_cols.append(c)
    df.columns = new_cols
    return df

def train_exp01_baseline():
    print("Loading Data...")
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    
    # 2. 전처리 (Preprocessing)
    # -----------------------
    # 컬럼명 정리
    train = minimal_clean_cols(train)
    test = minimal_clean_cols(test)
    
    # Target(실거래가) 로그 변환
    # 부동산 가격은 왜도(Skewness)가 심하므로 정규분포에 가깝게 만들기 위해 log1p 사용
    if train['target'].max() > 100: # 이미 로그변환 안되어 있다면
        train['target'] = np.log1p(train['target'])

    # 불필요한 텍스트 컬럼 삭제
    # 모델이 학습하기 어려운 비정형 텍스트나, 정보가 중복된 상세 주소 제거
    drop_texts = ['아파트명', '시군구', '도로명', '번지', '본번', '부번', '계약년월', '계약일']
    drop_texts = [re.sub(r'[(),\[\]\s]', '_', c) for c in drop_texts] # 컬럼명 클리닝 반영
    
    train.drop(columns=[c for c in drop_texts if c in train.columns], inplace=True, errors='ignore')
    test.drop(columns=[c for c in drop_texts if c in test.columns], inplace=True, errors='ignore')
    
    # 범주형 변수(Categorical) 인코딩
    # Label Encoding: '강남구' -> 0, '서초구' -> 1 식으로 변환
    le = LabelEncoder()
    col_list = train.select_dtypes(include=['object']).columns
    
    for col in col_list:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        # Train과 Test의 모든 카테고리를 합쳐서 학습 (Unknown 방지)
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    # 3. 모델 학습 (Training)
    # ---------------------
    print("\n[Validation Strategy] Shuffle K-Fold (5 Splits)")
    # 데이터를 무작위로 섞어서 5등분함. (시계열 고려 X)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    features = [c for c in train.columns if c != 'target']
    
    # LightGBM 하이퍼파라미터 (기본값 위주)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,  # 학습률
        'num_leaves': 128,      # 트리의 복잡도
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'n_jobs': -1,
        'seed': 42,
        'verbose': -1
    }
    
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train[features], train['target'])):
        # Train / Validation Split
        X_tr, X_val = train.iloc[train_idx][features], train.iloc[val_idx][features]
        y_tr, y_val = train.iloc[train_idx]['target'], train.iloc[val_idx]['target']
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        # 학습 수행
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dtrain, dvalid],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(1000)]
        )
        
        # OOF 예측 및 Test 예측
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(test[features]) / 5 # 5개 폴드 평균
        
        # RMSE 계산 (로그 변환 풀어서 실제 가격 차이 확인)
        rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_pred)))
        print(f"Fold {fold+1} RMSE: {rmse:,.0f}")
        
    total_rmse = np.sqrt(mean_squared_error(np.expm1(train['target']), np.expm1(oof_preds)))
    print(f"\n>> Overall Exp01 CV RMSE: {total_rmse:,.0f}")
    
    # 4. 결과 저장
    # -----------
    sub_path = 'submissions/submission_exp01_baseline.csv'
    if not os.path.exists('submissions'): os.makedirs('submissions')
    
    # 정수로 변환하여 저장
    pd.DataFrame({'target': np.expm1(test_preds)}).to_csv(sub_path, index=False)
    print(f"Submission Saved: {sub_path}")

if __name__ == "__main__":
    train_exp01_baseline()
