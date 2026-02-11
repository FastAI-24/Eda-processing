
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set Plot Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def analyze_and_clean_step1():
    print("🚀 [Step 1] Loading Raw Data...")
    
    # 1. Load Data
    raw_path = '../../data/raw/train.csv'
    if not os.path.exists(raw_path):
        print(f"❌ Error: Data not found at {raw_path}")
        return

    df = pd.read_csv(raw_path)
    print(f"✅ Initial Shape: {df.shape}")
    
    # 2. Analyze '해제사유발생일' (Cancellation Date)
    print("\n🔍 Analyzing Cancelled Transactions ('해제사유발생일')...")
    
    if '해제사유발생일' in df.columns:
        # Check non-null count
        n_cancelled = df['해제사유발생일'].notnull().sum()
        total_rows = len(df)
        ratio = (n_cancelled / total_rows) * 100
        
        print(f" -> Total Rows: {total_rows}")
        print(f" -> Cancelled Transactions (Non-null '해제사유발생일'): {n_cancelled} ({ratio:.2f}%)")
        
        if n_cancelled > 0:
            # Check price distribution of cancelled vs valid
            cancelled_prices = df[df['해제사유발생일'].notnull()]['target'] # target is 'transaction_real_price' usually but check raw
            # Raw data usually has 'target' as the last column, but let's check column names if needed.
            # Assuming 'target' exists as per previous scripts. If raw, it might be 'k-...' or something else.
            # Let's check columns first.
            
            # Actually, raw/train.csv usually has 'target' column in this competition setting.
            
            print("\n[Decision Point] These are cancelled transactions.")
            print("They represent 'failed' market prices or potentially manipulative signals.")
            print("❌ Action: Removing these rows to keep only valid transactions.")
            
            # Remove rows
            df_clean = df[df['해제사유발생일'].isnull()].copy()
            print(f"✅ Dropped {n_cancelled} rows.")
            print(f" -> New Shape: {df_clean.shape}")
            
            # Now we can drop the column itself since it's all null
            df_clean = df_clean.drop(columns=['해제사유발생일'])
            print("✅ Dropped column '해제사유발생일' (All remaining values are NaN)")
            
        else:
            print("No cancelled transactions found.")
            df_clean = df.copy()
    else:
        print("Column '해제사유발생일' not found.")
        df_clean = df.copy()

    # 3. Save Intermediate Data
    output_dir = '../../data/analysis_steps'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'step1_valid_transactions.csv')
    df_clean.to_csv(output_path, index=False)
    print(f"\n💾 Saved valid transactions to: {output_path}")

    return df_clean

if __name__ == "__main__":
    analyze_and_clean_step1()
