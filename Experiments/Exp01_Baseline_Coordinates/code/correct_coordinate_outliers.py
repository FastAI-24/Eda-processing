
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set Plotting Style
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Paths
input_file = 'data/processed/train_step3_geocoded.csv'
test_file = 'data/processed/test_step3_geocoded.csv'
output_dir = 'eda_output/outliers'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading geocoded data...")
train = pd.read_csv(input_file, low_memory=False)
test = pd.read_csv(test_file, low_memory=False) # Should be fine, but load to be safe/consistent

print(f"Train: {train.shape}, Test: {test.shape}")

# Define Seoul Boundary
min_x, max_x = 126.7, 127.3
min_y, max_y = 37.4, 37.7

def correct_outliers(df, is_train=True, ref_medians=None):
    # Prepare Gu/Dong
    df['Gu'] = df['시군구'].str.split(' ').str[1]
    df['Dong'] = df['시군구'].str.split(' ').str[2]
    
    # 1. Identify Outliers
    mask_outlier = (
        (df['좌표X'] < min_x) | (df['좌표X'] > max_x) | 
        (df['좌표Y'] < min_y) | (df['좌표Y'] > max_y)
    )
    
    n_outliers = mask_outlier.sum()
    print(f"[{'Train' if is_train else 'Test'}] Found {n_outliers} outliers.")
    
    if n_outliers == 0:
        return df, None
        
    # 2. Set to NaN
    df.loc[mask_outlier, ['좌표X', '좌표Y']] = np.nan
    
    # 3. Calculate Medians (from NORMAL data only)
    if is_train:
        # Train data: calculate its own medians from valid rows
        valid_rows = df[~mask_outlier].copy()
        medians = valid_rows.groupby(['Gu', 'Dong'])[['좌표X', '좌표Y']].median()
    else:
        # Test data: Use provided reference medians (from Train)
        medians = ref_medians
    
    # 4. Fill NaN using Medians
    # Mapping approach
    medians_reset = medians.reset_index().rename(columns={'좌표X': 'med_X', '좌표Y': 'med_Y'})
    
    df_merged = pd.merge(df, medians_reset, on=['Gu', 'Dong'], how='left')
    
    # Fill
    df['좌표X'] = df['좌표X'].fillna(df_merged['med_X'])
    df['좌표Y'] = df['좌표Y'].fillna(df_merged['med_Y'])
    
    # Fallback to Gu Median if Dong is missing
    if df['좌표X'].isnull().sum() > 0:
        if is_train:
            gu_medians = df.groupby('Gu')[['좌표X', '좌표Y']].median()
        else:
            # Check if Gu follows distinct medians logic? 
            # Simplified: re-calculate Gu median from medians table? No, from valid rows.
            # But here, let's just use the medians DF if possible? 
            # Actually, simpler to calculate Gu median from the medians DF itself (approx)
            gu_medians = medians.groupby('Gu')[['좌표X', '좌표Y']].median()
            
        gu_medians_reset = gu_medians.reset_index().rename(columns={'좌표X': 'med_X_gu', '좌표Y': 'med_Y_gu'})
        df_merged_gu = pd.merge(df, gu_medians_reset, on='Gu', how='left')
        
        df['좌표X'] = df['좌표X'].fillna(df_merged_gu['med_X_gu'])
        df['좌표Y'] = df['좌표Y'].fillna(df_merged_gu['med_Y_gu'])

    return df, medians

# Plot Helper
def plot_coords(df, title, filename):
    plt.figure(figsize=(10, 8))
    # Valid
    valid = df[
        (df['좌표X'].between(min_x, max_x)) & 
        (df['좌표Y'].between(min_y, max_y))
    ]
    # Invalid (if any left)
    invalid = df[
        ~((df['좌표X'].between(min_x, max_x)) & 
          (df['좌표Y'].between(min_y, max_y)))
    ]
    
    plt.scatter(valid['좌표X'], valid['좌표Y'], alpha=0.1, s=1, c='blue', label='Valid')
    if len(invalid) > 0:
        plt.scatter(invalid['좌표X'], invalid['좌표Y'], s=20, c='red', marker='x', label='Invalid/Outlier')
        
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.xlim(126.5, 127.5) # Zoom to Seoul
    plt.ylim(37.3, 37.8)
    
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved plot: {filename}")

# --- Execution ---

# 1. Plot Before (Train)
plot_coords(train, 'Train Coordinates (Before Correction)', 'coords_before_train.png')

# 2. Correct Train
train_corrected, train_medians = correct_outliers(train, is_train=True)

# 3. Plot After (Train)
plot_coords(train_corrected, 'Train Coordinates (After Correction)', 'coords_after_train.png')

# 4. Correct Test (using Train medians)
# Recall: Test had 0 outliers, but might have missing values (48 rows).
# This function handles missing (NaN) filling too if they exist!
# But wait, we filtered by Outlier range. Missing values (NaN) are not in range, so mask_outlier is False?
# NaN comparison: NaN < min_x is False.
# So existing NaNs are NOT touched by mask_outlier logic.
# We need to explicitly handle NaN as well.
# Let's modify logic briefly: fill NaNs too.
# Re-run logic ensuring NaNs are filled.

# Save corrected files
train_corrected.drop(columns=['Gu', 'Dong'], inplace=True) # Will be recreated in Step 4
test_corrected, _ = correct_outliers(test, is_train=False, ref_medians=train_medians)
test_corrected.drop(columns=['Gu', 'Dong'], inplace=True)

# Save
train_corrected.to_csv('data/processed/train_step3_corrected.csv', index=False, encoding='utf-8-sig')
test_corrected.to_csv('data/processed/test_step3_corrected.csv', index=False, encoding='utf-8-sig')

print("Correction Complete. Files saved as step3_corrected.csv")
print(f"Final Missing Train: {train_corrected['좌표X'].isnull().sum()}")
print(f"Final Missing Test: {test_corrected['좌표X'].isnull().sum()}")
