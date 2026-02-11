
"""
Step 3: Feature Importance 분석 및 시각화
이 스크립트는 수치형 및 범주형 변수가 타겟(집값)에 미치는 영향을 분석합니다.
분석 결과는 추후 '변수 선택(Feature Selection)' 단계에서 불필요한 변수를 제거하는 근거로 활용됩니다.

[분석 방법]
1. 수치형 변수: 피어슨 상관계수(Pearson Correlation)를 계산하여 타겟과의 선형 관계를 파악합니다.
2. 범주형 변수: ANOVA (분산분석) F-Test를 통해 그룹(범주) 간의 평균 차이가 통계적으로 유의미한지 검증합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# 설정
INPUT_PATH = '../../data/analysis_steps/step2_converted_dtypes.csv'
OUTPUT_DIR = '../../analysis/01_integrity_check/feature_selection_reports'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 시각화 스타일 설정 (한글 폰트 지원 등)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def analyze_feature_importance():
    print("[Step 3] Analyzing Feature Importance (Numeric & Categorical)...")
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input data not found at {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded Data Shape: {df.shape}")
    
    # 1. 타겟 컬럼 식별
    # 일반적인 이름('target', 'transaction_real_price' 등) 중 하나를 찾아서 타겟으로 설정합니다.
    target_col = 'target'
    if target_col not in df.columns:
        possible_targets = ['target', 'transaction_real_price', '물건금액(만원)']
        for t in possible_targets:
            if t in df.columns:
                target_col = t
                break
    
    print(f"Target Variable: '{target_col}'")
    
    # 타겟값이 없는 행은 분석에서 제외 (안전장치)
    df = df.dropna(subset=[target_col])
    
    # ---------------------------------------------------------
    # 1. 수치형 변수 분석 (상관계수)
    # ---------------------------------------------------------
    print("\n1. Numeric Feature Analysis (Correlation w/ Target)")
    
    numeric_df = df.select_dtypes(include=[np.number])
    # 타겟과의 상관계수 계산 (타겟 자체와의 상관관계 1.0은 제외)
    correlations = numeric_df.corr()[target_col].drop(target_col)
    
    # 절대값 기준으로 내림차순 정렬 (높은 상관관계 우선)
    correlations_abs = correlations.abs().sort_values(ascending=False)
    
    # 낮은 중요도 임계값 설정
    THRESHOLD_CORR = 0.05
    
    low_corr_cols = correlations_abs[correlations_abs < THRESHOLD_CORR].index.tolist()
    
    print(f"High Correlation Top 5:")
    print(correlations_abs.head(5))
    print(f"Low Correlation (< {THRESHOLD_CORR}) Candidates to Drop: {len(low_corr_cols)}")
    print(f"    {low_corr_cols}")
    
    # 리포트 저장
    with open(f'{OUTPUT_DIR}/numeric_feature_report.txt', 'w', encoding='utf-8') as f:
        f.write("Numeric Feature Importance (Pearson Correlation)\n")
        f.write(f"Target: {target_col}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Feature':<40} | {'Correlation':<10}\n")
        f.write("-" * 60 + "\n")
        for idx, val in correlations_abs.items():
            f.write(f"{idx:<40} | {val:.4f}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Drop Candidates (Corr < {THRESHOLD_CORR}):\n")
        f.write(", ".join(low_corr_cols))

    # ---------------------------------------------------------
    # 2. 범주형 변수 분석 (ANOVA)
    # ---------------------------------------------------------
    print("\n2. Categorical Feature Analysis (ANOVA F-Test)")
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    anova_results = []
    
    for col in categorical_cols:
        # 카테고리가 너무 많은 경우(예: 아파트명, 지번주소)는 일원분산분석에 부적절하므로 스킵
        n_unique = df[col].nunique()
        if n_unique > 50: 
            print(f"Skipping '{col}' (Too many categories: {n_unique})")
            continue
            
        # 각 범주별로 타겟값(집값) 그룹핑 (데이터 수가 적은 희귀 범주는 노이즈가 되므로 제외)
        groups = [group[target_col].values for name, group in df.groupby(col) if len(group) > 10]
        
        # 비교할 그룹이 2개 미만이면 분석 불가
        if len(groups) < 2:
            continue
            
        # ANOVA Test 수행
        # F-val이 클수록 그룹 간 차이가 큼, P-val이 작을수록(통상 0.05 미만) 통계적으로 유의미함
        f_val, p_val = stats.f_oneway(*groups)
        
        influence = "Low"
        if p_val < 0.05: influence = "High" # 유의수준 5% 이내에서 차이가 있음
        
        anova_results.append({
            'Feature': col,
            'F-Score': f_val,
            'P-Value': p_val,
            'Unique Categories': n_unique,
            'Influence': influence
        })
        
        # 박스플롯(Box Plot) 시각화 저장
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=col, y=target_col, data=df)
        plt.title(f'Price Distribution by {col} (ANOVA p={p_val:.2e})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        # 파일명 안전하게 변환
        safe_col_name = col.replace("/", "_").replace("=", "_").replace(" ", "_")
        plt.savefig(f'{OUTPUT_DIR}/boxplot_{safe_col_name}.png')
        plt.close()

    # F-Score 기준으로 정렬 (영향력이 큰 순서)
    anova_df = pd.DataFrame(anova_results).sort_values(by='F-Score', ascending=False)
    
    # 영향력이 낮은(통계적으로 유의미하지 않은) 컬럼 식별
    low_influence_cols = anova_df[anova_df['P-Value'] > 0.05]['Feature'].tolist()
    
    print(f"Categorical Features Analysis Complete.")
    print(f"Low Influence (Not Significant) Candidates: {low_influence_cols}")
    
    # 범주형 변수 분석 리포트 저장
    with open(f'{OUTPUT_DIR}/categorical_feature_report.txt', 'w', encoding='utf-8') as f:
        f.write("Categorical Feature Importance (ANOVA F-Test)\n")
        f.write("Higher F-Score = Stronger relationship with Price.\n")
        f.write("P-Value < 0.05 = Statistically Significant.\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Feature':<30} | {'F-Score':<10} | {'P-Value':<10} | {'Categories':<5} | {'Influence'}\n")
        f.write("-" * 100 + "\n")
        
        for _, row in anova_df.iterrows():
            f.write(f"{row['Feature']:<30} | {row['F-Score']:<10.2f} | {row['P-Value']:<10.2e} | {row['Unique Categories']:<5} | {row['Influence']}\n")
            
        f.write("-" * 100 + "\n")
        f.write(f"Drop Candidates (P-Value > 0.05):\n")
        f.write(", ".join(low_influence_cols))
        
    print(f"Reports saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    analyze_feature_importance()
