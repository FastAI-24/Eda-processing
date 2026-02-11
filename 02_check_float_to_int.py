
"""
Step 2: Float -> Integer 변환 가능성 진단
데이터셋의 Float 컬럼들 중, 실제로는 정수값만 가지고 있는 컬럼(예: 12.0, 500.0)을 식별합니다.
이러한 컬럼들은 불필요하게 Float 타입을 사용하여 메모리를 낭비하고 있으므로,
Int 타입(혹은 결측치가 있다면 Int64)으로 변환하는 것이 좋습니다.

[기능]
1. 결측치 확인: NaN이 포함되어 있는지 확인합니다 (일반 Int 변환 불가, Int64 사용 필요).
2. 정수 여부 검사: 모든 유효한 값이 정수 형태(x.0)인지 확인합니다.
3. 리포트 생성: 변환 가능한 후보 컬럼 목록을 파일로 저장합니다.
"""

import pandas as pd
import numpy as np

def identify_float_to_int_candidates():
    print("[Step 2] Identifying Float Columns that should be Integers...")
    
    # 데이터 로드 (Step 1 결과)
    data_path = '../../data/analysis_steps/step1_valid_transactions.csv'
    df = pd.read_csv(data_path)
    print(f"Loaded Data Shape: {df.shape}")
    
    # Float 컬럼 추출
    float_cols = df.select_dtypes(include=['float']).columns
    print(f"Checking {len(float_cols)} float columns...\n")
    
    candidates = []
    
    for col in float_cols:
        # Check 1: 결측치(NaN)가 있는지 확인
        # (결측치가 있으면 일반 'int' 타입으로는 변환할 수 없고, pandas의 'Int64' 타입을 써야 함)
        has_na = df[col].isnull().sum() > 0
        
        # Check 2: 결측치를 제외한 모든 값이 정수인지 확인
        # (예: 3.0은 정수 취급, 3.5는 실수 취급)
        non_na_values = df[col].dropna()
        if len(non_na_values) == 0:
            continue
            
        is_all_integer = (non_na_values % 1 == 0).all()
        
        # 모든 값이 정수라면 후보 리스트에 추가
        if is_all_integer:
            candidates.append({
                'Column': col,
                'Missing Count': df[col].isnull().sum(),
                'Sample Values': df[col].dropna().unique()[:5].tolist() # 샘플 값 5개 확인
            })

    # 분석 결과를 리포트 파일로 저장
    with open('float_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"Found {len(candidates)} candidates for Integer conversion:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Column Name':<30} | {'Missing':<13} | {'Sample Values'}\n")
        f.write("-" * 80 + "\n")
        
        for c in candidates:
            # 샘플 값은 5개까지만 보여줌
            samples = str(c['Sample Values'][:5])
            f.write(f"{c['Column']:<30} | {c['Missing Count']:<13} | {samples}\n")
            
        f.write("-" * 80 + "\n")
        f.write("\nTip: Columns with missing values cannot be standard 'int' type.")
        f.write("        We must either fill NaNs or use pandas 'Int64' (nullable int) type.\n")

    print("Analysis Complete. Report saved to 'float_analysis_report.txt'")

if __name__ == "__main__":
    identify_float_to_int_candidates()
