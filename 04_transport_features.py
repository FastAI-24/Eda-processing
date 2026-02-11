"""
교통 피처 엔지니어링 (Step 10)
- KDTree를 사용하여 거리 기반 피처 생성
- Train/Test 데이터 모두 적용
"""
import pandas as pd
import numpy as np
import os
from scipy.spatial import cKDTree
from tqdm import tqdm

print("=" * 80)
print("Step 10: 교통 피처 엔지니어링")
print("=" * 80)

base_path = r'c:\Users\romeo\Desktop\house prediction'

# 1. 데이터 로드
print("\n[1] 데이터 로드")
try:
    train = pd.read_csv(os.path.join(base_path, 'data/analysis_steps/step9_3_cleaned_final.csv'), encoding='utf-8')
    test = pd.read_csv(os.path.join(base_path, 'data/analysis_steps/test_step9_3_cleaned.csv'), encoding='utf-8')
except:
    train = pd.read_csv(os.path.join(base_path, 'data/analysis_steps/step9_3_cleaned_final.csv'), encoding='cp949')
    test = pd.read_csv(os.path.join(base_path, 'data/analysis_steps/test_step9_3_cleaned.csv'), encoding='cp949')

subway = pd.read_csv(os.path.join(base_path, 'data/raw/subway_feature.csv'), encoding='utf-8')
bus = pd.read_csv(os.path.join(base_path, 'data/raw/bus_feature.csv'), encoding='utf-8')

print(f"  - Train: {train.shape}")
print(f"  - Test: {test.shape}")
print(f"  - Subway: {subway.shape}")
print(f"  - Bus: {bus.shape}")

# 2. 좌표 데이터 준비 (KDTree)
print("\n[2] KDTree 구축")

# 좌표 전처리 함수
def clean_coords(df, x_col, y_col):
    # 결측치 제거
    df[x_col] = df[x_col].fillna(0)
    df[y_col] = df[y_col].fillna(0)
    
    # Inf 제거
    df[x_col] = df[x_col].replace([np.inf, -np.inf], 0)
    df[y_col] = df[y_col].replace([np.inf, -np.inf], 0)
    
    # 타입 변환
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce').fillna(0)
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce').fillna(0)
    return df

# 전처리 적용
print("  - Cleaning coordinates...")
train = clean_coords(train, '좌표X', '좌표Y')
test = clean_coords(test, '좌표X', '좌표Y')
subway = clean_coords(subway, '경도', '위도')
bus = clean_coords(bus, 'X좌표', 'Y좌표')

# KDTree는 (x, y) 순서 중요하지 않지만 통일해야 함. 여기선 (위도, 경도) 순서로 사용
# 위도 = Y, 경도 = X
subway_coords = subway[['위도', '경도']].values
bus_coords = bus[['Y좌표', 'X좌표']].values

subway_tree = cKDTree(subway_coords)
bus_tree = cKDTree(bus_coords)

# 강남역 좌표 (위도, 경도)
gangnam_debug = np.array([[37.498095, 127.027610]])

def add_transport_features(df, name="Data"):
    print(f"\n  Processing {name} ({len(df)} rows)...")
    
    # 아파트 좌표 (위도, 경도)
    house_coords = df[['좌표Y', '좌표X']].values
    
    # 1. 가장 가까운 지하철역 거리
    # k=1: 가장 가까운 1개 점
    # distance는 도(degree) 단위이므로 미터(m)로 대략 변환 필요
    # 위도 1도 ≈ 111km, 경도 1도 ≈ 88km (서울 기준) -> 평균 100km로 근사하거나, 
    # 정확하게 하려면 Haversine 써야 하지만, 여기선 상대적 비교가 중요하므로 degree * 111000 사용
    
    print("    - Calculating nearest subway distance...")
    dists, _ = subway_tree.query(house_coords, k=1)
    df['nearest_subway_dist'] = dists * 111000  # meter 변환 (약식)
    
    # 2. 가장 가까운 버스 정류장 거리
    print("    - Calculating nearest bus distance...")
    dists, _ = bus_tree.query(house_coords, k=1)
    df['nearest_bus_dist'] = dists * 111000
    
    # 3. 반경 내 역세권 여부
    # 1km = 0.009 degree (대략)
    # 500m = 0.0045 degree (대략)
    
    print("    - Counting nearby stations (KDTree query_ball_point takes time)...")
    # query_ball_point는 리스트를 반환하므로 len()을 적용해야 함 -> 속도 느릴 수 있음
    # 속도 개선을 위해 apply 대신 list comprehension 사용
    
    # 지하철 1km (0.009)
    # df['subway_count_1km'] = [len(x) for x in subway_tree.query_ball_point(house_coords, 0.009)]
    # -> 너무 느릴 수 있으니 제외하고 거리만 먼저 사용
    
    # 대신 역세권 여부 (500m 이내)
    df['is_subway_500m'] = (df['nearest_subway_dist'] <= 500).astype(int)
    df['is_bus_300m'] = (df['nearest_bus_dist'] <= 300).astype(int)
    
    # 4. 강남역 거리
    # 단순 유클리드 거리 (degree) -> 미터 변환
    print("    - Calculating Gangnam distance...")
    diff = house_coords - gangnam_debug
    # diff[:, 0] = 위도 차이, diff[:, 1] = 경도 차이
    # 간단히 유클리드 거리 계산
    dist_deg = np.sqrt(np.sum(diff**2, axis=1))
    df['gangnam_dist'] = dist_deg * 111000
    
    # 5. 종합 교통 점수 (임의 가중치)
    # 거리가 가까울수록 점수가 높아야 함 (Log 변환 등 고려)
    # 여기선 간단히 1000m 기준 역수 점수
    df['transport_score'] = (
        (1000 / (df['nearest_subway_dist'] + 10)) * 0.7 + 
        (500 / (df['nearest_bus_dist'] + 10)) * 0.3
    )
    
    return df

# 적용
train = add_transport_features(train, "Train")
test = add_transport_features(test, "Test")

# 3. 저장
print("\n[3] 결과 저장")
save_dir_train = os.path.join(base_path, 'data/analysis_steps/step10_transport_features.csv')
save_dir_test = os.path.join(base_path, 'data/analysis_steps/test_step10_transport_features.csv')

train.to_csv(save_dir_train, index=False, encoding='utf-8-sig') # 한글 깨짐 방지 utf-8-sig
test.to_csv(save_dir_test, index=False, encoding='utf-8-sig')

print(f"  - Saved Train: {save_dir_train}")
print(f"  - Saved Test: {save_dir_test}")

# 컬럼 확인
print("\n[New Features]")
print(train[['nearest_subway_dist', 'nearest_bus_dist', 'gangnam_dist', 'transport_score']].head())

print("\nDone.")
