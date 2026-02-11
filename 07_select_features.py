
"""
Step 7: 최종 피처 선택 (Feature Selection)
앞선 분석(상관관계 분석, ANOVA 등)과 도메인 지식을 바탕으로 모델 학습에 사용할 최종 피처를 선별합니다.
정보량이 낮거나(Low Variance), 중복되거나(Redundant), 미래 정보(Data Leakage)를 담고 있는 변수들을 과감히 제거합니다.

[주요 작업]
1. 제거 리스트 적용: 중개사 소재지(지역 중복), 등기신청일자(미래 정보), 시행사(브랜드와 중복) 등 제거.
2. 중요 변수 상태 점검: '복도유형', '난방방식' 등 결측치가 있지만 중요한 편의시설 정보들의 상태를 확인합니다.
3. 최종 저장: 선별된 컬럼들만 남긴 데이터셋을 저장하고, 컬럼 목록을 별도 파일로 기록합니다.
"""

import pandas as pd
import numpy as np
import os

def select_final_features():
    print("[Step 7] Final Feature Selection (Dropping Low Impact/Redundant)...")
    
    input_path = '../../data/analysis_steps/step5_column_finalized.csv'
    output_path = '../../data/analysis_steps/step7_features_selected.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"Input Shape: {df.shape}")
    
    # ---------------------------------------------------------
    # 1. 제거 리스트 적용 (사용자 확인 및 분석 결과 기반)
    # ---------------------------------------------------------
    drop_targets = [
        '중개사소재지',      # '구', '동' 정보와 중복됨
        '등기신청일자',      # 거래 이후에 발생하는 정보이므로 예측 시점에 알 수 없음 (Data Leakage)
        #'거래유형',        # (주석처리됨) 직거래/중개거래 여부는 가격에 영향을 줄 수 있어 보류
        '도로명',           # 지번주소와 거의 동일한 정보
        'k-시행사',         # 건설사(브랜드) 정보가 더 중요함
        'k-단지분류(아파트,주상복합등등)', # 결측치가 많고 아파트 데이터셋이라 변별력 낮음
        'k-관리방식',       # 결측치 많음
        
        # 전용면적별 세대수는 'k-전체세대수'와 상관관계가 높거나 너무 세분화되어 있어 제거
        'k-전용면적별세대현황(60㎡이하)',
        'k-전용면적별세대현황(60㎡~85㎡이하)',
        'k-85㎡~135㎡이하',
        'k-135㎡초과'
    ]
    
    existing_drop = [c for c in drop_targets if c in df.columns]
    
    if existing_drop:
        df.drop(columns=existing_drop, inplace=True)
        print(f"Dropped {len(existing_drop)} columns: {existing_drop}")

    # ---------------------------------------------------------
    # 2. 주요 결측 변수 상태 점검 (Rescue Targets)
    # ---------------------------------------------------------
    # 이 변수들은 결측이 있더라도 추후에 채워서 살려볼 가치가 있는 변수들입니다.
    rescue_targets = ['k-복도유형', 'k-난방방식']
    print(f"\nCheck Rescue Targets Status:")
    for col in rescue_targets:
        if col in df.columns:
            n_missing = df[col].isnull().sum()
            ratio = (n_missing / len(df)) * 100
            print(f"   - {col}: Missing {n_missing} ({ratio:.2f}%)")
            
            # 값의 분포 확인 (채울 때 최빈값 등을 사용할지 판단하기 위함)
            print(f"     Values: {df[col].dropna().unique()[:5]}")

    # ---------------------------------------------------------
    # 3. 최종 저장 및 목록 생성
    # ---------------------------------------------------------
    print(f"\nSaved selected features to: {output_path}")
    print(f"   Final Shape: {df.shape}")
    
    df.to_csv(output_path, index=False)
    
    # 컬럼 목록을 텍스트 파일로 저장하여 한눈에 파악 가능하게 함
    with open('columns_step7.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(df.columns.tolist()))

if __name__ == "__main__":
    select_final_features()
