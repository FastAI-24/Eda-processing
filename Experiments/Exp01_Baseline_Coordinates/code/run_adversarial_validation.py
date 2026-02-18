
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Set Plotting Style
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Paths
input_dir = 'data/processed'
output_dir = 'viz'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

train_path = os.path.join(input_dir, 'train_final.csv')
test_path = os.path.join(input_dir, 'test_final.csv')

def run_adversarial_validation():
    print("Loading Data...")
    try:
        train = pd.read_csv(train_path, low_memory=False)
        test = pd.read_csv(test_path, low_memory=False)
    except FileNotFoundError:
        print("Data not found.")
        return

    # --- Preprocessing for ADV (Simplified) ---
    # We need to mimic the features used in the regressor as closely as possible
    
    # 1. Drop unnecessary columns
    drop_cols = ['아파트명', '시군구', '도로명', '번지', '본번', '부번', '계약년월', '계약일', 'target'] 
    
    # Prepare Train
    # Note: 'target' is price, we drop it.
    X_train = train.drop(columns=[c for c in drop_cols if c in train.columns], errors='ignore')
    
    # Prepare Test
    X_test = test.drop(columns=[c for c in drop_cols if c in test.columns], errors='ignore')
    
    # Ensure columns match
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    print(f"Features used for Adversarial Validation ({len(common_cols)}): {common_cols}")

    # 2. Add Adversarial Target
    X_train['is_test'] = 0
    X_test['is_test'] = 1
    
    # 3. Combine
    data_adv = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y_adv = data_adv['is_test']
    X_adv = data_adv.drop('is_test', axis=1)

    # 4. Handle Categorical / Object columns
    # Re-apply minimal cleaning if needed
    def clean_cols(df):
        new_cols = []
        for col in df.columns:
            c = re.sub(r'[(),\[\]\s]', '_', col)
            new_cols.append(c)
        df.columns = new_cols
        return df

    X_adv = clean_cols(X_adv)
    
    # Label Encode objects
    le = LabelEncoder()
    for col in X_adv.select_dtypes(include=['object']).columns:
        X_adv[col] = X_adv[col].astype(str)
        X_adv[col] = le.fit_transform(X_adv[col])
        
    # 5. Train LGBM Classifier
    print("\nTraining Adversarial Classifier...")
    
    clf = lgb.LGBMClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # Stratified CV for stable AUC
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    
    # Just fit once on full for feature importance, or Loop?
    # Let's do 5-fold to get accurate AUC score first
    oof_preds = np.zeros(len(X_adv))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_adv, y_adv)):
        X_tr, X_val = X_adv.iloc[train_idx], X_adv.iloc[val_idx]
        y_tr, y_val = y_adv.iloc[train_idx], y_adv.iloc[val_idx]
        
        clf.fit(X_tr, y_tr)
        val_pred = clf.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        
        score = roc_auc_score(y_val, val_pred)
        auc_scores.append(score)
        print(f"Fold {fold+1} AUC: {score:.4f}")
        
    avg_auc = np.mean(auc_scores)
    print(f"\nOverall Adversarial AUC Score: {avg_auc:.4f}")
    
    # Interpretation
    if avg_auc > 0.7:
        print(">> Warning: Significant difference between Train and Test distributions!")
    else:
        print(">> Good: Train and Test distributions are relatively similar.")

    # 6. Feature Importance (Train on full data one last time)
    clf.fit(X_adv, y_adv)
    importances = pd.DataFrame({
        'feature': X_adv.columns,
        'importance': clf.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    print("\n[Top 10 Features causing Distribution Shift]")
    print(importances.head(10))
    
    # Visualize
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=importances.head(20))
    plt.title(f'Adversarial Validation Feature Importance (AUC: {avg_auc:.4f})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adversarial_importance.png'))
    print(f"\nFeature importance plot saved to {output_dir}/adversarial_importance.png")

if __name__ == "__main__":
    run_adversarial_validation()
