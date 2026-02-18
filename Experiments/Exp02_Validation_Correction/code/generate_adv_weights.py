
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Set Plotting Style
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Paths
input_dir = 'data/processed'
output_dir = 'data/processed' # Save weight enhanced train file here

train_path = os.path.join(input_dir, 'train_final.csv')
test_path = os.path.join(input_dir, 'test_final.csv')
train_adv_path = os.path.join(output_dir, 'train_final_adv_weighted.csv')

def generate_adversarial_weights():
    print("Loading Data...")
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    
    # 1. Prepare Data for Classification
    # Use features that are relevant but not strictly Time Series (days_since is tricky)
    # If we include days_since, AUC -> 1.0. Model learns "Future is Test".
    # BUT, we want to weight "Recent" data higher.
    # So including days_since is actually OK for weighting purpose, because it will naturally weight recent data higher.
    # However, to find OTHER distributional shifts (e.g., specific location over-representation),
    # we should check if removing days_since helps finding other issues.
    # For "Weighting", using all features is generally preferred to make Train resemble Test in ALL aspects.
    
    # Define features to use
    drop_cols = ['아파트명', '시군구', '도로명', '번지', '본번', '부번', '계약년월', '계약일', 'target']
    
    X_train = train.drop(columns=[c for c in drop_cols if c in train.columns], errors='ignore')
    X_test = test.drop(columns=[c for c in drop_cols if c in test.columns], errors='ignore')
    
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    print(f"Features for Weighting ({len(common_cols)}): {common_cols}")
    
    # 2. Labeling
    X_train['is_test'] = 0
    X_test['is_test'] = 1
    
    # 3. Concat
    data_adv = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y_adv = data_adv['is_test']
    X_adv = data_adv.drop('is_test', axis=1)
    
    # 4. Encoding (Simple Label)
    le = LabelEncoder()
    def clean_cols(df): # Clean for LGBM
        new_cols = []
        for col in df.columns:
            c = re.sub(r'[(),\[\]\s]', '_', col)
            new_cols.append(c)
        df.columns = new_cols
        return df

    X_adv = clean_cols(X_adv)
    
    for col in X_adv.select_dtypes(include=['object']).columns:
        X_adv[col] = X_adv[col].astype(str)
        X_adv[col] = le.fit_transform(X_adv[col])
        
    # 5. Train Classifier (Cross-Validated Probability)
    print("\nTraining Adversarial Classifier for Weights...")
    clf = lgb.LGBMClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # We want probability for TRAIN set.
    # We can use predictions from CV.
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Array to store proba for ALL data
    # (But we really only need it for Train part)
    # Actually, to get 'unbiased' probability for Train, we should use OOF predictions.
    # If we train on all and predict on all, it might overfit.
    # OOF represents "How much does this validation fold look like Test?" (trained on other folds + Test)
    # Wait, standard Adv Val is Train vs Test.
    # To get weight for Train instance i:
    # Train on (Train_others + Test), Predict on (Train_i).
    # This is slightly complex.
    
    # Simpler approach often used:
    # Train on (Train + Test). Predict on (Train).
    # If model is not too complex, it's fine.
    # But usually 5-fold OOF is better.
    
    # Let's do 5-fold OOF for Train part. Test part is always 1 (target).
    
    train_indices = data_adv[data_adv['is_test'] == 0].index
    test_indices = data_adv[data_adv['is_test'] == 1].index
    
    # We need predictions for train_indices.
    # X_adv has all rows.
    
    train_probs = np.zeros(len(train_indices))
    
    # Split TRAIN into 5 folds. Test is always used in training to provide "Class 1" examples.
    # Fold 1: Train = (Train_Fold_2345 + Test), Valid = (Train_Fold_1)
    # This way Valid (Train_Fold_1) is unseen by the model trained to distinguish Class 0 (Other Train) vs Class 1 (Test).
    
    # However, 'is_test' label is fixed.
    # If we just do StratifiedKFold on FULL data, we get OOF for everyone.
    
    oof_preds_full = np.zeros(len(data_adv))
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_adv, y_adv)):
        X_tr, X_val = X_adv.iloc[trn_idx], X_adv.iloc[val_idx]
        y_tr, y_val = y_adv.iloc[trn_idx], y_adv.iloc[val_idx]
        
        clf.fit(X_tr, y_tr)
        val_prob = clf.predict_proba(X_val)[:, 1]
        oof_preds_full[val_idx] = val_prob
        print(f"Fold {fold+1} Finished.")
        
    # Extract Train Probs
    train_probs = oof_preds_full[train_indices]
    
    # 6. Calculate Weights
    # Weight = p / (1 - p)
    # If p (probability of being Test) is high -> Weight is high.
    # If p is close to 1, weight explodes. Clip it.
    
    # Clip prob to avoid division by zero (max 0.95?)
    train_probs = np.clip(train_probs, 0.0, 0.99) # Allow high weights for very recent data
    
    weights = train_probs / (1.0 - train_probs)
    
    # Normalize weights (optional, but good for stability)
    # Mean weight = 1.0
    weights = weights / weights.mean()
    
    print(f"\nWeight Stats:")
    print(f"Min: {weights.min():.4f}")
    print(f"Max: {weights.max():.4f}")
    print(f"Mean: {weights.mean():.4f}")
    
    # 7. Viz histogram of weights
    plt.figure(figsize=(10, 6))
    sns.histplot(weights, bins=50)
    plt.title('Distribution of Adversarial Weights (Train Data)')
    plt.xlabel('Weight (Likelihood of being Test-like)')
    plt.savefig('viz/adversarial_weights_dist.png')
    print("Weight distribution plot saved.")
    
    # 8. Save Weights
    train['adversarial_weight'] = weights
    
    # Save new Train file
    train.to_csv(train_adv_path, index=False)
    print(f"Saved Weighted Train Data: {train_adv_path}")

if __name__ == "__main__":
    generate_adversarial_weights()
