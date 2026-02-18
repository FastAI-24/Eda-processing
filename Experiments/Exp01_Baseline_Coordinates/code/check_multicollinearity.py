
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

# Set Plotting Style
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

input_path = 'data/processed/train_final.csv'
output_dir = 'viz'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def check_multicollinearity():
    print("Loading Final Data...")
    try:
        df = pd.read_csv(input_path, low_memory=False)
    except FileNotFoundError:
        print("File not found.")
        return

    # Check for text columns to drop for correlation analysis
    # We want to analyze numerical correlations.
    # Identify numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Also, we might have some object columns that are actually numeric but read as object?
    # Let's trust select_dtypes for now, but maybe some 'k-' columns are relevant if they were encoded?
    # In train_final.csv, 'Gu_encoded', 'Dong_encoded' should be there.
    
    print(f"Analyzing {len(numeric_df.columns)} numerical columns: {numeric_df.columns.tolist()}")
    
    # Calculate Correlation Matrix
    corr_matrix = numeric_df.corr()
    
    # Filter for high correlation > 0.9
    high_corr_pairs = []
    
    # Iterate over the correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                colname_i = corr_matrix.columns[i]
                colname_j = corr_matrix.columns[j]
                correlation_value = corr_matrix.iloc[i, j]
                high_corr_pairs.append((colname_i, colname_j, correlation_value))

    print("\n[High Correlation Pairs (> 0.9)]")
    if not high_corr_pairs:
        print("No pairs found with correlation > 0.9")
    else:
        for pair in high_corr_pairs:
            print(f"{pair[0]} <-> {pair[1]}: {pair[2]:.4f}")

    # Visualize
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix (Final Data)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_correlation_matrix.png'))
    print(f"\nCorrelation matrix saved to {output_dir}/final_correlation_matrix.png")

if __name__ == "__main__":
    check_multicollinearity()
