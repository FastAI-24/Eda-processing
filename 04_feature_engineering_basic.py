
"""
Step 4: 기본 피처 엔지니어링 (날짜 및 주소)
원천 데이터의 날짜와 주소 정보를 파싱하여 분석 모델이 이해하기 쉬운 형태의 파생 변수를 생성합니다.

[주요 작업]
1. 날짜 처리: 
   - '계약년월' + '계약일' -> '계약일자' (YYYYMMDD 정수형) 생성.
   - 시계열 분석이나 정렬을 위해 단일 정수형 날짜 코드가 필요합니다.
2. 주소 분리:
   - '시군구' 문자열을 분해하여 '시', '구', '동' 컬럼을 각각 생성합니다.
   - 예: "서울특별시 강남구 역삼동" -> 시:서울, 구:강남구, 동:역삼동
3. 지번주소 생성:
   - 외부 API(Geocoding) 연동을 위해 정확한 '지번주소' 포맷을 생성합니다.
   - 본번, 부번 정보를 결합하여 '강남구 역삼동 123-4' 형태의 주소를 만듭니다.
"""

import pandas as pd
import numpy as np
import os

def process_address_and_date(df_path, output_path, is_train=True):
    print(f"[Step 4] Processing Address & Date Features for {df_path}...")
    
    if not os.path.exists(df_path):
        print(f"Error: Input file not found: {df_path}")
        return

    df = pd.read_csv(df_path)
    print(f"Loaded Shape: {df.shape}")
    
    # ---------------------------------------------------------
    # 1. Date Feature Engineering
    # ---------------------------------------------------------
    print("Standardizing Dates...")
    
    # '계약년월'(YYYYMM)과 '계약일'(D)이 모두 존재할 경우 처리
    if '계약년월' in df.columns and '계약일' in df.columns:
        # 문자열로 변환하여 결합한 뒤, 다시 정수형(Int64)으로 변환
        # zfill(2)는 한 자리수 날짜(1일)를 두 자리(01일)로 맞춰줌
        df['계약일자'] = (
            df['계약년월'].astype(str) + 
            df['계약일'].astype(str).str.zfill(2)
        ).astype('Int64')
        print(f" -> Created '계약일자' from '계약년월' + '계약일'")
        
        # '계약일' 컬럼은 이제 중복 정보이므로 제거 (계약년월은 그룹핑 용도로 남길 수도 있음)
        df.drop(columns=['계약일'], inplace=True, errors='ignore')
        print(f" -> Dropped raw '계약일' column")

    # ---------------------------------------------------------
    # 2. Address Feature Engineering
    # ---------------------------------------------------------
    print("Parsing Addresses...")
    
    if '시군구' in df.columns:
        # 공백을 기준으로 문자열 분할
        # format: "City(0) Gu(1) Dong(2)"
        split_addr = df['시군구'].str.split(' ', expand=True)
        
        # 분할된 결과가 있는지 확인 후 할당
        if split_addr.shape[1] >= 2:
            df['구'] = split_addr[1]
            print(f" -> Created '구' (e.g., 강남구)")
            
        if split_addr.shape[1] >= 3:
            df['동'] = split_addr[2]
            print(f" -> Created '동' (e.g., 개포동)")
            
    # 지번주소(Lot Address) 생성
    # Kakao Map API 등에서 좌표를 찾으려면 '동 + 번지' 조합이 필요함
    if '본번' in df.columns and '부번' in df.columns:
        # 결측치 처리 후 문자열 변환
        bonbun = df['본번'].fillna(0).astype(int).astype(str)
        bubun = df['부번'].fillna(0).astype(int)
        
        # 부번이 0이면 "123", 0보다 크면 "123-4" 형태로 만듦
        bubun_str = bubun.astype(str)
        suffix = np.where(bubun > 0, "-" + bubun_str, "")
        
        # 최종 주소 조합
        df['지번주소'] = df['시군구'] + " " + bonbun + suffix
        print(f" -> Created '지번주소' (e.g., 서울특별시 강남구 개포동 658-1)")
        
        # 원본 숫자 컬럼은 이제 불필요하므로 제거
        df.drop(columns=['본번', '부번'], inplace=True)
        print(" -> Dropped '본번', '부번' columns")
        
    # ---------------------------------------------------------
    # 3. Drop Useless Columns (Based on Analysis)
    # ---------------------------------------------------------
    # 이전 분석에서 유용성이 없거나 결측치가 너무 많다고 판단된 컬럼 제거
    drop_cols = ['단지소개기존clob', '건축면적']
    existing_drop = [c for c in drop_cols if c in df.columns]
    
    if existing_drop:
        df.drop(columns=existing_drop, inplace=True)
        print(f"Dropped Useless Columns: {existing_drop}")

    # 4. Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved processed data to: {output_path}")
    print(f"   Final Shape: {df.shape}")

if __name__ == "__main__":
    # 파이프라인 실행: Step 2 -> Step 3
    process_address_and_date(
        '../../data/analysis_steps/step2_converted_dtypes.csv',
        '../../data/analysis_steps/step3_feature_engineered.csv'
    )
