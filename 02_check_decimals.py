
"""
Step 2-2: Float 컬럼의 소수점 데이터 무결성 검사
이 스크립트는 데이터셋 내의 모든 float 타입 컬럼을 검사하여,
실제로 소수점 이하 값이 존재하는지 확인합니다.

[기능]
1. Float 컬럼 식별: 데이터셋에서 부동소수점(float) 타입의 컬럼을 모두 찾습니다.
2. 소수점 값 검사: 각 값에서 정수부를 뺐을 때 0이 아닌 값이 있는지(즉, 소수점이 유의미한지) 확인합니다.
3. 리포트 생성: 
    - 소수점 비율이 0%인 컬럼 -> Integer(정수)로 변환해도 안전함.
    - 소수점 비율이 있는 컬럼 -> Float 유지 필요.

[출력]
- 'float_decimal_check_report.txt': 분석 결과 리포트 파일
"""

import pandas as pd
import numpy as np

def check_decimals_in_float_cols():
    print("[Step 2-2] Inspecting Decimals in Float Columns...")
    
    # 정제된 데이터 로드 (Step 1 결과물)
    data_path = '../../data/analysis_steps/step1_valid_transactions.csv'
    df = pd.read_csv(data_path)
    
    # Float 타입 컬럼만 선택
    float_cols = df.select_dtypes(include=['float']).columns
    print(f"Checking {len(float_cols)} float columns for decimal values...\n")
    
    results = []
    
    for col in float_cols:
        # 결측치(NaN)는 소수점 검사에서 제외합니다.
        series = df[col].dropna()
        if len(series) == 0: continue
        
        # 소수점 이하 값 존재 여부 확인 (값 % 1 != 0)
        # 예: 12.34 % 1 = 0.34 (True), 12.0 % 1 = 0.0 (False)
        has_decimal = (series % 1 != 0)
        n_decimals = has_decimal.sum()
        ratio_decimals = (n_decimals / len(series)) * 100
        
        # 소수점이 있는 실제 예시 값 몇 개를 추출하여 리포트에 포함시킵니다.
        examples = series[has_decimal].head(3).tolist()
        
        results.append({
            'Column': col,
            'Total Non-Null': len(series),
            'Decimal Count': n_decimals,
            'Decimal Ratio (%)': round(ratio_decimals, 4),
            'Sample Decimals': examples if examples else "None"
        })

    # 소수점 비율이 낮은 순서대로 정렬합니다. (0%인 경우 Integer 변환 후보 1순위)
    results.sort(key=lambda x: x['Decimal Ratio (%)'])
    
    # 분석 결과를 텍스트 파일로 저장합니다.
    with open('float_decimal_check_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"Decimal Value Analysis ({len(results)} columns)\n")
        f.write("Columns with 0.0000% decimal ratio are safe to convert to Int.\n")
        f.write("-" * 110 + "\n")
        f.write(f"{'Column Name':<30} | {'Total':<8} | {'Decimals':<8} | {'Ratio(%)':<10} | {'Examples'}\n")
        f.write("-" * 110 + "\n")
        
        for r in results:
            f.write(f"{r['Column']:<30} | {r['Total Non-Null']:<8} | {r['Decimal Count']:<8} | {r['Decimal Ratio (%)']:<10} | {r['Sample Decimals']}\n")
            
    print("Analysis Complete. Report saved to 'float_decimal_check_report.txt'")

if __name__ == "__main__":
    check_decimals_in_float_cols()
