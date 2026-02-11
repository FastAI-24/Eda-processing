
"""
Step 2: 데이터 타입 변환 실행 (Float -> Int64)
이전 단계에서 식별된 '정수형 데이터를 가진 Float 컬럼'들을 실제로 변환합니다.
Pandas의 'Int64' (Nullable Integer) 타입을 사용하여, NaN 값을 유지하면서도 정수형으로 변환합니다.
이 과정은 메모리 사용량을 줄이고 데이터의 의미를 명확히 합니다.

[작업 내용]
1. 변환 대상 컬럼 정의: 이전 분석 결과(리포트)를 바탕으로 변환할 컬럼 리스트 확정.
2. 타입 변환: Float -> Round(반올림) -> Int64 변환 수행.
   - 반올림을 하는 이유는 부동소수점 오차(예: 3.0000000001)를 안전하게 처리하기 위함입니다.
3. 저장: 변환된 데이터를 다음 단계(Step 2) 파일로 저장합니다.
"""

import pandas as pd
import numpy as np
import os

def convert_to_nullable_int():
    print("[Step 2] Converting Float Columns to Nullable Integer (Int64)...")
    
    # 입력/출력 경로 설정
    input_path = '../../data/analysis_steps/step1_valid_transactions.csv'
    output_path = '../../data/analysis_steps/step2_converted_dtypes.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: Previous step data not found at {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"Loaded Data Shape: {df.shape}")
    
    # 변환 대상 컬럼 목록 정의
    # (이전 스크립트 분석 결과를 바탕으로 Int로 변환해도 안전하다고 판단된 컬럼들)
    # 주의: '단지소개기존clob' 같은 컬럼이 포함되어 있다면 텍스트 여부를 확인해야 하지만, 
    # 분석 결과 수치형으로 판명되면 포함합니다.
    
    target_cols = [
        '본번', '부번',
        'k-전체동수', 'k-전체세대수', '주차대수',
        'k-연면적', 'k-주거전용면적', 'k-관리비부과면적',
        'k-전용면적별세대현황(60㎡이하)', 
        'k-전용면적별세대현황(60㎡~85㎡이하)', 
        'k-85㎡~135㎡이하', 
        'k-135㎡초과'
    ]
    
    # 실제 데이터셋에 존재하는 컬럼만 필터링
    existing_cols = [c for c in target_cols if c in df.columns]
    
    print(f"Converting {len(existing_cols)} columns to Int64 (Nullable Int)...")
    
    for col in existing_cols:
        try:
            # 원본 타입 확인
            original_type = df[col].dtype
            
            # Int64 (대문자 I)로 변환하여 결측치(NaN) 허용 정수형으로 변경
            # round()를 먼저 호출하여 미세한 소수점 오차 제거
            df[col] = df[col].round().astype('Int64')
            
            print(f" -> Converted '{col}': {original_type} -> {df[col].dtype}")
            
        except Exception as e:
            print(f"Failed to convert '{col}': {e}")

    # 결과 저장
    df.to_csv(output_path, index=False)
    print(f"\nSaved type-converted data to: {output_path}")

if __name__ == "__main__":
    convert_to_nullable_int()
