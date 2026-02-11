
"""
Step 5: 범주형 변수 정제 (불필요 컬럼 제거 및 이름 변경)
이전 단계(결측치 진단)를 바탕으로 범주형 변수들을 정리합니다.
분석에 도움이 되지 않거나(노이즈), 결측치가 너무 많거나, 행정적인 정보(전화번호 등)를 담고 있는 컬럼을 제거합니다.

[주요 작업]
1. 컬럼 삭제: '등록일자', '홈페이지', '전화번호' 등 집값 예측과 무관한 관리용 데이터 제거.
2. 컬럼명 변경: 분석가가 이해하기 쉬운 직관적인 이름으로 변경 (예: 복잡한 한글명 -> 단순화).
3. 상태 리포트: 정제 후 남은 컬럼 목록과 여전히 결측치가 있는 컬럼을 출력하여 다음 단계 작업을 안내.
"""

import pandas as pd
import numpy as np
import os

def cleanup_categorical_features(df_path, output_path):
    print(f"[Step 5] Categorical Feature Cleanup for {df_path}...")
    
    if not os.path.exists(df_path):
        print(f"Error: Input file not found: {df_path}")
        return

    df = pd.read_csv(df_path)
    print(f"Input Shape: {df.shape}")
    
    # ---------------------------------------------------------
    # 1. 불필요 범주형 컬럼 제거
    # ---------------------------------------------------------
    # 제거 사유:
    # - 정보 가치 없음 (관리번호, 전화번호, 홈페이지)
    # - 결측치 과다 (단지신청일, 승인일 등)
    # - 중복 정보 혹은 의미 불명 (기타/임의 등)
    
    drop_cols = [
        'k-등록일자', 'k-홈페이지', 'k-수정일자', 'k-전화번호', 'k-팩스번호',
        '고용보험관리번호', '단지신청일', '단지승인일',
        '사용허가여부', '관리비 업로드', 
        '기타/의무/임대/임의=1/2/3/4'
    ]
    
    existing_drop = [c for c in drop_cols if c in df.columns]
    
    if existing_drop:
        df.drop(columns=existing_drop, inplace=True)
        print(f"Dropped {len(existing_drop)} columns: {existing_drop}")

    # ---------------------------------------------------------
    # 2. 주요 컬럼명 표준화
    # ---------------------------------------------------------
    rename_map = {
        'k-사용검사일-사용승인일': '사용검사일'
    }
    
    df.rename(columns=rename_map, inplace=True)
    if '사용검사일' in df.columns:
        print(f"Renamed 'k-사용검사일-사용승인일' -> '사용검사일'")

    # ---------------------------------------------------------
    # 3. 최종 상태 리포트
    # ---------------------------------------------------------
    print(f"\nSaved cleanup data to: {output_path}")
    print(f"   Final Shape: {df.shape}")
    print(f"   Remaining Columns ({len(df.columns)}):")
    print(df.columns.tolist())
    
    # 향후 결측치 처리가 필요한 컬럼(Imputation Candidates) 식별
    missing_cats = df.select_dtypes(include=['object']).isnull().sum()
    missing_cats = missing_cats[missing_cats > 0]
    if not missing_cats.empty:
        print("\nCategorical columns with missing values (To be handled later):")
        print(missing_cats.index.tolist())
        
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    cleanup_categorical_features(
        '../../data/analysis_steps/step3_feature_engineered.csv',
        '../../data/analysis_steps/step4_cleaned_categorical.csv'
    )
