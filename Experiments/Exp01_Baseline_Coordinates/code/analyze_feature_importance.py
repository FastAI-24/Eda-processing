
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

# Set Plotting Style
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Paths
input_dir = 'data/processed'
output_dir = 'logs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

train_path = os.path.join(input_dir, 'train_final.csv')

def analyze_importance():
    print("Loading Data for Analysis...")
    try:
        train = pd.read_csv(train_path, low_memory=False)
    except:
        print("Data not found.")
        return

    # --- Reproduce Preprocessing (Simplified for Feature Extraction) ---
    print("Preprocessing...")
    
    # 1. Top Brand
    def check_brand(name):
        top_brands = ['래미안', '자이', '힐스테이트', '아이파크', '푸르지오', '롯데캐슬', '더샵', 'e편한세상', '이편한세상', '아크로', '디에이치', '꿈의숲']
        if pd.isna(name): return 0
        for brand in top_brands:
            if brand in str(name): return 1
        return 0
    
    if '아파트명' in train.columns:
        train['is_top_brand'] = train['아파트명'].apply(check_brand)
    
    # 2. Drop Text
    drop_texts = ['아파트명', '시군구', '도로명', '번지', '본번', '부번', '계약년월', '계약일']
    train.drop(columns=[c for c in drop_texts if c in train.columns], inplace=True)
    
    # 3. Clean Cols
    new_cols = []
    for col in train.columns:
        c = re.sub(r'[(),\[\]\s]', '_', col) # same as before
        new_cols.append(c)
    train.columns = new_cols
    
    # 4. Log Transform
    train['target'] = np.log1p(train['target'])
    if '전용면적_㎡_' in train.columns:
        train['전용면적_㎡_'] = np.log1p(train['전용면적_㎡_'])
        
    # 5. Target Encode (Gu/Dong) - Quick approx for importance
    # Just global mean map for speed, or simple k-fold
    # Let's do simple map to save logic overhead in this analysis script
    for col in ['Gu_encoded', 'Dong_encoded']:
        if col in train.columns:
            means = train.groupby(col)['target'].mean()
            train[f'{col}_target'] = train[col].map(means)
            
    # 6. Label Encode Objects
    le = LabelEncoder()
    obj_cols = train.select_dtypes(include=['object']).columns.tolist()
    for col in obj_cols:
        train[col] = train[col].astype(str)
        train[col] = le.fit_transform(train[col])
        
    # --- Train LightGBM (Fast) ---
    print("Training LightGBM (Fast mode for Feature Importance)...")
    
    features = [c for c in train.columns if c != 'target']
    X = train[features]
    y = train['target']
    
    # Params (Using the Strong ones from tuning)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 128,
        'max_depth': 12,
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # Train on all data (or subset) just to get importance
    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(params, dtrain, num_boost_round=500)
    
    # --- Extract Importance ---
    importance = pd.DataFrame({
        'Feature': features,
        'Importance (Gain)': model.feature_importance(importance_type='gain'),
        'Importance (Split)': model.feature_importance(importance_type='split')
    }).sort_values('Importance (Gain)', ascending=False)
    
    # Normalizing for better readability
    importance['Importance (Gain)'] = importance['Importance (Gain)'] / importance['Importance (Gain)'].sum() * 100
    
    print("\n[Top 20 Important Features]")
    print(importance.head(20).to_string(index=False, formatters={'Importance (Gain)': '{:.2f}%'.format}))
    
    # Save Text Report
    importance.head(30).to_csv(os.path.join(output_dir, 'feature_importance_report.csv'), index=False)
    
    # Visualize
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance (Gain)', y='Feature', data=importance.head(20))
    plt.title('LightGBM Feature Importance (Top 20)')
    plt.xlabel('Importance (%)')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'feature_importance_viz.png')
    plt.savefig(plot_path)
    print(f"\nAnalysis Complete. Visualization saved to {plot_path}")

if __name__ == "__main__":
    analyze_importance()
