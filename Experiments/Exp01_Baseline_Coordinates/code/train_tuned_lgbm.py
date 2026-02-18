
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import re
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import time

# Try importing optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
    print("Optuna is available. Tuning will be performed.")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not found. Switching to Random Search / Manual Tuning.")

# Paths
input_dir = 'data/processed'
output_dir = 'logs'
submission_dir = 'submissions'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(submission_dir):
    os.makedirs(submission_dir)

train_path = os.path.join(input_dir, 'train_final.csv')
test_path = os.path.join(input_dir, 'test_final.csv')

# --- Helper Functions (From previous valid steps) ---
def minimal_clean_cols(df):
    new_cols = []
    for col in df.columns:
        c = re.sub(r'[(),\[\]\s]', '_', col)
        new_cols.append(c)
    df.columns = new_cols
    return df

def check_brand(name):
    top_brands = ['래미안', '자이', '힐스테이트', '아이파크', '푸르지오', '롯데캐슬', '더샵', 'e편한세상', '이편한세상', '아크로', '디에이치', '꿈의숲']
    if pd.isna(name): return 0
    for brand in top_brands:
        if brand in str(name):
            return 1
    return 0

def target_encode(train_df, test_df, col, target='target', n_splits=5):
    if col not in train_df.columns: return train_df, test_df
    
    train_df[f'{col}_target'] = np.nan
    test_df[f'{col}_target'] = np.nan
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for tr_idx, val_idx in kf.split(train_df):
        X_tr, X_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        means = X_tr.groupby(col)[target].mean()
        train_df.loc[val_idx, f'{col}_target'] = X_val[col].map(means)
        
    global_mean = train_df[target].mean()
    train_df[f'{col}_target'].fillna(global_mean, inplace=True)
    
    global_means = train_df.groupby(col)[target].mean()
    test_df[f'{col}_target'] = test_df[col].map(global_means)
    test_df[f'{col}_target'].fillna(global_mean, inplace=True)
    
    return train_df, test_df

# --- Load & Preprocess ---
def load_and_preprocess():
    print("Loading Final Data...")
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    
    # 1. Top Brand
    print("Creating Top Brand Feature...")
    if '아파트명' in train.columns:
        train['is_top_brand'] = train['아파트명'].apply(check_brand)
        test['is_top_brand'] = test['아파트명'].apply(check_brand)
        
    # 2. Drop Text Cols
    drop_texts = ['아파트명', '시군구', '도로명', '번지', '본번', '부번', '계약년월', '계약일']
    train.drop(columns=[c for c in drop_texts if c in train.columns], inplace=True)
    test.drop(columns=[c for c in drop_texts if c in test.columns], inplace=True)
    
    # 3. Rename Cols
    train = minimal_clean_cols(train)
    test = minimal_clean_cols(test)
    
    # 4. Log Transform
    print("Applying Log Transformation...")
    train['target'] = np.log1p(train['target'])
    
    area_col = '전용면적_㎡_' if '전용면적_㎡_' in train.columns else 'Area' # Check actual name
    if '전용면적_㎡_' in train.columns: area_col = '전용면적_㎡_'
    
    if area_col in train.columns:
        train[area_col] = np.log1p(train[area_col])
        test[area_col] = np.log1p(test[area_col])
        
    # 5. Target Encode (using Log Target)
    print("Target Encoding...")
    if 'Gu_encoded' in train.columns:
        train, test = target_encode(train, test, 'Gu_encoded')
    if 'Dong_encoded' in train.columns:
        train, test = target_encode(train, test, 'Dong_encoded')
        
    # 6. Label Encode Objects
    le = LabelEncoder()
    obj_cols = train.select_dtypes(include=['object']).columns.tolist()
    for col in obj_cols:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        
    return train, test

# --- Optimization ---
def tune_and_train():
    train, test = load_and_preprocess()
    
    features = [c for c in train.columns if c != 'target']
    X = train[features]
    y = train['target']
    X_test = test[features]
    
    print(f"Training shapes: X={X.shape}, y={y.shape}")
    
    # Objective for Optuna
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'seed': 42,
            'n_jobs': -1,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 32, 512),
            'max_depth': trial.suggest_int('max_depth', 6, 16),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        }
        
        # Fast CV with KFold (Shuffle=True for robust estimate)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Use LightGBM built-in CV
        dtrain = lgb.Dataset(X, label=y)
        
        history = lgb.cv(
            params,
            dtrain,
            num_boost_round=5000,
            folds=cv,
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        # Returns: {'rmse-mean': [scores...], 'rmse-stdv': ...}
        best_rmse = history['valid rmse-mean'][-1]
        return best_rmse

    best_params = {}
    
    if OPTUNA_AVAILABLE:
        print("\nStarting Optuna Tuning (15 Trials to save time)...")
        # Increase trials if you have more time/resources.
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=15) 
        
        print("\nBest Trial:")
        print(f"  Value (Log RMSE): {study.best_value:.5f}")
        print(f"  Params: {study.best_params}")
        
        best_params = study.best_params
    else:
        print("\nUsing Manual Strong Params...")
        best_params = {
            'learning_rate': 0.03,
            'num_leaves': 256,
            'max_depth': 12,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }

    # Final Train with Best Params
    print("\nTraining Final Model with Best Params...")
    
    # Add constant params
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1
    })
    
    # Use 5-Fold Splitting manually to get OOF predictions (for accurate CV score check)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            best_params,
            dtrain,
            num_boost_round=10000,
            valid_sets=[dtrain, dval],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)] # Silent
        )
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        
        # Predict Test (Average later)
        test_preds += model.predict(X_test) / kf.get_n_splits()
        
        rmse_fold = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"Fold {fold+1} Log RMSE: {rmse_fold:.5f}")

    # Calculate Overall CV RMSE
    cv_log_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\nOverall CV Log RMSE: {cv_log_rmse:.5f}")
    
    # Calculate Real Scale RMSE
    y_real = np.expm1(y)
    oof_real = np.expm1(oof_preds)
    cv_real_rmse = np.sqrt(mean_squared_error(y_real, oof_real))
    print(f"Overall CV Real RMSE: {cv_real_rmse:,.2f}")
    
    # Submission
    final_test_preds = np.expm1(test_preds)
    sub = pd.DataFrame({'target': final_test_preds})
    sub_path = os.path.join(submission_dir, 'submission_tuned_lgbm.csv')
    sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

if __name__ == "__main__":
    tune_and_train()
