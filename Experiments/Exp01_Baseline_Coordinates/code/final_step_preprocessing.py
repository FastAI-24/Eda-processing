
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Paths
input_dir = 'data/processed'
output_dir = 'data/processed'
train_file = os.path.join(input_dir, 'train_step3_corrected.csv')
test_file = os.path.join(input_dir, 'test_step3_corrected.csv')

def final_preprocessing():
    print("Loading Corrected Data...")
    try:
        train = pd.read_csv(train_file, low_memory=False)
        test = pd.read_csv(test_file, low_memory=False)
    except FileNotFoundError:
        print("Corrected data not found.")
        return

    # 1. [Safety Check] Basic Outliers (Defensive)
    # Area: <= 0 or > 500 (Physical logic)
    # Year: < 1900 or > 2025 (Physical logic)
    print("\n[Step 1] Basic Outlier Defense...")
    
    # Train
    mask_area = (train['전용면적(㎡)'] <= 0) | (train['전용면적(㎡)'] > 500)
    if mask_area.sum() > 0:
        print(f"  Found {mask_area.sum()} invalid areas in Train. Replacing with NaN.")
        train.loc[mask_area, '전용면적(㎡)'] = np.nan
        
    mask_year = (train['건축년도'] < 1900) | (train['건축년도'] > 2025)
    if mask_year.sum() > 0:
        print(f"  Found {mask_year.sum()} invalid years in Train. Replacing with NaN.")
        train.loc[mask_year, '건축년도'] = np.nan
        
    # Test (Must fill if NaN)
    mask_area_test = (test['전용면적(㎡)'] <= 0) | (test['전용면적(㎡)'] > 500)
    if mask_area_test.sum() > 0:
        test.loc[mask_area_test, '전용면적(㎡)'] = np.nan
        
    mask_year_test = (test['건축년도'] < 1900) | (test['건축년도'] > 2025)
    if mask_year_test.sum() > 0:
        test.loc[mask_year_test, '건축년도'] = np.nan

    # Fill NaNs from Step 1 (if any created)
    # Use Median of entire Train
    if train['전용면적(㎡)'].isnull().sum() > 0:
        train['전용면적(㎡)'].fillna(train['전용면적(㎡)'].median(), inplace=True)
    if train['건축년도'].isnull().sum() > 0:
        train['건축년도'].fillna(train['건축년도'].median(), inplace=True)
        
    if test['전용면적(㎡)'].isnull().sum() > 0:
        test['전용면적(㎡)'].fillna(train['전용면적(㎡)'].median(), inplace=True)
    if test['건축년도'].isnull().sum() > 0:
        test['건축년도'].fillna(train['건축년도'].median(), inplace=True)

    # 2. Extract Location (Gu, Dong) & Encode
    print("\n[Step 2] Location Features & Encoding...")
    
    # Create Gu/Dong
    for df in [train, test]:
        df['Gu'] = df['시군구'].str.split(' ').str[1]
        df['Dong'] = df['시군구'].str.split(' ').str[2]
        # Fill missing Dong if split failed
        df['Gu'] = df['Gu'].fillna('unknown')
        df['Dong'] = df['Dong'].fillna('unknown')

    # Label Encode (Fit on All)
    le_gu = LabelEncoder()
    le_dong = LabelEncoder()
    
    all_gu = pd.concat([train['Gu'], test['Gu']]).astype(str).unique()
    all_dong = pd.concat([train['Dong'], test['Dong']]).astype(str).unique()
    
    le_gu.fit(all_gu)
    le_dong.fit(all_dong)
    
    train['Gu_encoded'] = le_gu.transform(train['Gu'].astype(str))
    test['Gu_encoded'] = le_gu.transform(test['Gu'].astype(str))
    
    train['Dong_encoded'] = le_dong.transform(train['Dong'].astype(str))
    test['Dong_encoded'] = le_dong.transform(test['Dong'].astype(str))

    # 3. [Safety Check] Fill Remaining Missing Coordinates (Test 48 rows)
    print("\n[Step 3] Coordinate Final Fill (Dong Median)...")
    
    # Compute medians from TRAIN
    dong_medians = train.groupby('Dong_encoded')[['좌표X', '좌표Y']].median()
    gu_medians = train.groupby('Gu_encoded')[['좌표X', '좌표Y']].median()
    
    def fill_remaining_coords(df):
        # 1. Dong Median
        if df['좌표X'].isnull().sum() > 0:
            df_merged = pd.merge(df, dong_medians, left_on='Dong_encoded', right_index=True, how='left', suffixes=('', '_med'))
            df['좌표X'] = df['좌표X'].fillna(df_merged['좌표X_med'])
            df['좌표Y'] = df['좌표Y'].fillna(df_merged['좌표Y_med'])
            
        # 2. Gu Median (Fallback)
        if df['좌표X'].isnull().sum() > 0:
             df_merged = pd.merge(df, gu_medians, left_on='Gu_encoded', right_index=True, how='left', suffixes=('', '_gu'))
             df['좌표X'] = df['좌표X'].fillna(df_merged['좌표X_gu'])
             df['좌표Y'] = df['좌표Y'].fillna(df_merged['좌표Y_gu'])
        return df

    train = fill_remaining_coords(train)
    test = fill_remaining_coords(test)
    
    print(f"  Missing Coords - Train: {train['좌표X'].isnull().sum()}, Test: {test['좌표X'].isnull().sum()}")

    # 4. [Drop] Unnecessary Columns
    cols_drop = ['도로명', '번지', '본번', '부번', 'k-관리방식'] 
    # Note: 'Gu', 'Dong' text cols kept or dropped? Let's drop text cols, keep encoded.
    # Actually, keep text for EDA/Human check if needed, but for model drop them?
    # Let's keep text for now (Step 4 final usually has features ready for model).
    # Model needs numbers. Let's drop raw address parts.
    
    print(f"\n[Step 4] Dropping Columns: {cols_drop}")
    train.drop(columns=[c for c in cols_drop if c in train.columns], inplace=True)
    test.drop(columns=[c for c in cols_drop if c in test.columns], inplace=True)

    # 5. [Feature] Time
    print("\n[Step 5] Time Features...")
    def process_time(df):
        df['contract_date'] = pd.to_datetime(df['계약년월'].astype(str) + df['계약일'].astype(str).str.zfill(2), format='%Y%m%d', errors='coerce')
        base_date = pd.to_datetime('2007-01-01')
        df['days_since'] = (df['contract_date'] - base_date).dt.days
        df['contract_year'] = df['contract_date'].dt.year
        df['contract_month'] = df['contract_date'].dt.month
        df.drop(columns=['contract_date'], inplace=True)
        return df
        
    train = process_time(train)
    test = process_time(test)

    # 6. [Imputation] k- Categorical 'unknown'
    k_cols = ['k-복도유형', 'k-난방방식', 'k-단지분류(아파트,주상복합등등)']
    for col in k_cols:
        # Fill unknown
        train[col] = train[col].fillna('unknown')
        test[col] = test[col].fillna('unknown')
        # Label Encode
        le = LabelEncoder()
        all_vals = pd.concat([train[col], test[col]]).astype(str)
        le.fit(all_vals)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

    # 7. [Parking Imputation] Model-based (RF)
    print("\n[Step 7] Parking Imputation (Model-based)...")
    
    # Filter Training Data for RF (Clean only)
    # Condition: Parking is known AND Area > 0
    # Add Defensive: Parking cars < 5000 (Physical limit per complex?) -> Optional
    train_parking_clean = train[
        (train['주차대수'].notnull()) & 
        (train['전용면적(㎡)'] > 0)
    ].copy()
    
    # Feature Selection for Parking Model
    pk_features = ['전용면적(㎡)', '건축년도', '좌표X', '좌표Y', 'Gu_encoded']
    
    X_train_pk = train_parking_clean[pk_features]
    y_train_pk = train_parking_clean['주차대수']
    
    # Prepare Predict sets (Missing Parking)
    mask_miss_train = train['주차대수'].isnull()
    mask_miss_test = test['주차대수'].isnull()
    
    X_pred_train = train.loc[mask_miss_train, pk_features]
    X_pred_test = test.loc[mask_miss_test, pk_features]
    
    print(f"  Training RF with {len(X_train_pk)} samples.")
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train_pk, y_train_pk)
    print(f"  Parking Model R2 Score: {rf.score(X_train_pk, y_train_pk):.4f}")
    
    # Predict & Fill
    if len(X_pred_train) > 0:
        train.loc[mask_miss_train, '주차대수'] = rf.predict(X_pred_train)
        
    if len(X_pred_test) > 0:
        test.loc[mask_miss_test, '주차대수'] = rf.predict(X_pred_test)

    # 8. Final Export
    # Drop temp text cols if you want purely numeric?
    # Let's keep 'Gu', 'Dong', 'k-...' text if needed, but for now models need encoded.
    # We have 'Gu_encoded', 'Dong_encoded'.
    # We Label Encoded k-cols in place.
    # So 'Gu', 'Dong' text cols are redundant. Drop them to be clean.
    train.drop(columns=['Gu', 'Dong'], inplace=True)
    test.drop(columns=['Gu', 'Dong'], inplace=True)
    
    train.to_csv('data/processed/train_final.csv', index=False, encoding='utf-8-sig')
    test.to_csv('data/processed/test_final.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n[Done] Saved final datasets to {output_dir}")
    print(f"Train Final Shape: {train.shape}")
    print(f"Test Final Shape: {test.shape}")
    print("Columns:", train.columns.tolist())

if __name__ == "__main__":
    final_preprocessing()
