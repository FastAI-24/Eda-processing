
"""
Step 6: 컬럼명 최종 정리 및 분석 데이터셋 확정
이전 단계까지의 정제 작업을 마친 데이터셋에서, 분석에 최종적으로 사용할 컬럼들을 확정합니다.
코드의 가독성을 위해 한글 컬럼명을 직관적으로 변경하고, 중복되거나 불필요한 파생 전 원본 컬럼들을 제거합니다.

[주요 작업]
1. 추가 삭제(Drops): 이미 파생변수가 만들어져 필요 없어진 컬럼(예: 시군구, 번지)을 삭제합니다.
2. 검증(Validation): 핵심 컬럼(전용면적 등)이 존재하는지 확인합니다.
3. 재정렬(Reorder): 분석가가 보기 편하도록 중요 컬럼(구, 동, 아파트명 등)을 앞으로 배치합니다.
4. 리스트 출력: 최종 컬럼 목록을 출력하여 사용자가 확인할 수 있게 합니다.
"""

import pandas as pd
import numpy as np
import os

def finalize_columns_and_list():
    print("[Step 6] Finalizing Columns & Listing them...")
    
    input_path = '../../data/analysis_steps/step4_cleaned_categorical.csv'
    output_path = '../../data/analysis_steps/step5_column_finalized.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"Input Shape: {df.shape}")
    
    # ---------------------------------------------------------
    # 1. 불필요 컬럼 추가 삭제
    # ---------------------------------------------------------
    # - '번지', '시군구' -> '지번주소', '구', '동'으로 대체됨
    # - '계약년월' -> '계약일자'로 대체됨 (필요시 남길 수도 있음)
    # - 'k-주거전용면적' -> '전용면적'과 거의 동일하며 결측이 많아 삭제
    
    drop_cols = ['계약년월', 'k-주거전용면적', '번지', '시군구']
    existing_drop = [c for c in drop_cols if c in df.columns]
    
    if existing_drop:
        df.drop(columns=existing_drop, inplace=True)
        print(f"Dropped Additional Columns: {existing_drop}")
        
    # 핵심 컬럼 존재 여부 확인
    if '전용면적(㎡)' in df.columns:
        print("Confirmed '전용면적(㎡)' exists.")
    else:
        print("Warning: '전용면적(㎡)' not found! Please check columns.")

    # ---------------------------------------------------------
    # 2. 컬럼 순서 재배치 (가독성 향상)
    # ---------------------------------------------------------
    # 사람이 식별하기 쉬운 지리/이름 정보를 맨 앞으로 이동
    priority_cols = ['구', '동', '지번주소', '아파트명', '계약일자']
    
    existing_priority = [c for c in priority_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_priority]
    
    # 우선순위 컬럼 + 나머지 컬럼 순으로 결합
    new_order = existing_priority + remaining_cols
    df = df[new_order]
    print(f"Reordered Columns: {existing_priority} came to front.")

    # ---------------------------------------------------------
    # 3. 최종 컬럼 리스트 출력
    # ---------------------------------------------------------
    cols = df.columns.tolist()
    print(f"\nCurrent Total Columns: {len(cols)}")
    print("-" * 60)
    for idx, col in enumerate(cols):
        dtype = df[col].dtype
        n_missing = df[col].isnull().sum()
        print(f"{idx+1:02d}. {col:<40} ({dtype}, Missing: {n_missing})")
    print("-" * 60)

    # 저장
    df.to_csv(output_path, index=False)
    print(f"\nSaved finalized data to: {output_path}")

if __name__ == "__main__":
    finalize_columns_and_list()
