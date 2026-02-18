
import os
import re
import requests
import json
import logging
import pandas as pd
import math
from dotenv import load_dotenv
from time import sleep

# Load API keys from .env
load_dotenv()
VWORLD_KEY = os.getenv("VWORLD_API_KEY")
NAVER_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_SECRET = os.getenv("NAVER_CLIENT_SECRET")
KAKAO_KEY = os.getenv("KAKAO_API_KEY")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Input/Output paths (Wait, we need to load Step 2 data)
INPUT_DIR = 'data/processed'
OUTPUT_DIR = 'data/processed'

# 1. Address Refinement function
def refine_address(address):
    """주소에서 괄호와 상세 주소를 제거하는 정제 함수"""
    if pd.isna(address): return ""
    # 1. 괄호와 그 안의 내용 제거
    address = re.sub(r'\(.*\)', '', str(address))
    # 2. 층, 호, 동(건물번호 외) 정보 제거 (간단하게 번지수 뒤를 자름 - 숫자-숫자 또는 숫자 만 남김)
    # Ex: "개포동 658-1 501호" -> "개포동 658-1"
    # Find patterns like "123-45" or "123" and cut strictly after that? 
    # Or just keep everything before the first digit sequence unless it's Jibun?
    # Let's use the provided regex logic:
    match = re.search(r'(.+?\d+(?:-\d+)?)', address)
    if match:
        return match.group(1).strip()
    return address.strip()

# 2. API Call Functions
def get_coords_from_vworld(address, key):
    url = f"http://api.vworld.kr/req/address?service=address&request=getcoord&version=2.0&crs=epsg:4326&address={address}&refine=true&simple=false&format=json&type=PARCEL&key={key}"
    try:
        if not key: return None, None
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['response']['status'] == 'OK':
                pt = data['response']['result']['point']
                return float(pt['x']), float(pt['y']) # X=Lon, Y=Lat
    except Exception as e:
        logging.error(f"Vworld Error for {address}: {e}")
    return None, None

def get_coords_from_naver(address, client_id, client_secret):
    url = f"https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode?query={address}"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret
    }
    try:
        if not client_id or not client_secret: return None, None
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'OK' and data['addresses']:
                pt = data['addresses'][0]
                return float(pt['x']), float(pt['y'])
    except Exception as e:
        logging.error(f"Naver Error for {address}: {e}")
    return None, None

def get_coords_from_kakao(address, key):
    url = f"https://dapi.kakao.com/v2/local/search/address.json?query={address}"
    headers = {"Authorization": f"KakaoAK {key}"}
    try:
        if not key: return None, None
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['documents']:
                pt = data['documents'][0]
                return float(pt['x']), float(pt['y'])
    except Exception as e:
        logging.error(f"Kakao Error for {address}: {e}")
    return None, None

# 3. Master Coordination Function
def get_coords_master(address):
    # [준비] 주소 정제
    clean_addr = refine_address(address)
    if not clean_addr: return None, None, "Fail"
    
    # [Step 1] 브이월드 (Vworld)
    x, y = get_coords_from_vworld(clean_addr, VWORLD_KEY)
    if x: return x, y, "Vworld"
    
    # [Step 2] 네이버 (Naver)
    x, y = get_coords_from_naver(clean_addr, NAVER_ID, NAVER_SECRET)
    if x: return x, y, "Naver"
    
    # [Step 3] 카카오 (Kakao) 
    # Use clean_addr for Kakao too
    x, y = get_coords_from_kakao(clean_addr, KAKAO_KEY)
    if x: return x, y, "Kakao"
    
    # [최후의 수단] 주소 단순화 (동 + 지번) -> 네이버 재시도
    # 예: '역삼동 123-4' 형태만 남기기 (뒤에서 2단어)
    parts = clean_addr.split()
    if len(parts) >= 2:
        very_short_addr = " ".join(parts[-2:])
        # Try Naver again
        x, y = get_coords_from_naver(very_short_addr, NAVER_ID, NAVER_SECRET)
        if x: return x, y, "Final_Naver"
        
        # Try Kakao again
        x, y = get_coords_from_kakao(very_short_addr, KAKAO_KEY)
        if x: return x, y, "Final_Kakao"

    return None, None, "Fail"

# --- Main Execution ---
if __name__ == "__main__":
    
    # 1. Load Data
    logging.info("Loading Step 2 Data...")
    try:
        train = pd.read_csv(os.path.join(INPUT_DIR, 'train_step2.csv'), low_memory=False)
        test = pd.read_csv(os.path.join(INPUT_DIR, 'test_step2.csv'), low_memory=False)
    except FileNotFoundError:
        logging.error("Step 2 data not found. Run previous step.")
        exit()

    # Combine for processing (to create a comprehensive map)
    # We only care about rows missing coordinates: '좌표X', '좌표Y'
    # Check column names (Step 2 might have dropped some, but user said '도로명', '번지' exist. '좌표X' too?)
    # 'imputation_test_log.txt' showed coordinates were missing.
    # Check if '좌표X' exists and if it has nulls.
    cols_to_check = ['좌표X', '좌표Y']
    if '좌표X' not in train.columns: # It might have been dropped in Step 2 if user approved it?
        # NO. Step 2 dropped 26 cols. '좌표X' was NOT in that list.
        # But wait, did Step 2 accidentally drop them?
        pass

    # Identify rows with missing coordinates
    train['is_train'] = True
    test['is_train'] = False
    all_df = pd.concat([train, test], ignore_index=True)
    
    # Filter rows with missing coords
    missing_mask = all_df['좌표X'].isnull() | all_df['좌표Y'].isnull()
    target_rows = all_df[missing_mask].copy()
    
    logging.info(f"Total rows with missing coordinates: {len(target_rows)}")
    
    if len(target_rows) == 0:
        logging.info("No missing coordinates found! Skipping API calls.")
        exit()

    # Create unique address list for efficiency
    # Use '시군구' + '번지' (most precise Jibun address)
    # If '번지' is missing, fallback to '도로명'?
    # But usually '번지' is key for Jibun.
    # Let's combine: 시군구 + " " + 번지
    # Handle NaN in '번지'
    target_rows['full_jibun'] = target_rows['시군구'].astype(str) + " " + target_rows['번지'].fillna("").astype(str)
    
    unique_addrs = target_rows['full_jibun'].unique()
    logging.info(f"Unique addresses to geocode: {len(unique_addrs)}")
    
    # Check for existing cache/map file to resume (Optional)
    map_file = 'address_coordinate_map.csv'
    if os.path.exists(map_file):
        existing_map = pd.read_csv(map_file)
        # Filter out already processed
        processed_addrs = set(existing_map['address'].unique())
        addrs_to_process = [addr for addr in unique_addrs if addr not in processed_addrs]
        logging.info(f"Resuming... {len(addrs_to_process)} addresses left.")
        coord_map = existing_map.to_dict('records') # List of dicts
        # Re-convert to dictionary for fast lookup later? No, stick to list append
    else:
        addrs_to_process = unique_addrs
        coord_map = []

    # Batch Processing
    batch_size = 100
    save_interval = 500
    total_processed = 0

    import time
    
    for i, addr in enumerate(addrs_to_process):
        try:
            x, y, source = get_coords_master(addr)
            
            coord_map.append({
                'address': addr,
                'new_X': x,
                'new_Y': y,
                'source': source
            })
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(addrs_to_process)}: {addr} -> {source}", end='\r')
            
            # Save intermediate
            if (i + 1) % save_interval == 0:
                pd.DataFrame(coord_map).to_csv(map_file, index=False)
                logging.info(f"Saved checkpoint at {i+1}")
            
            # Rate limiting (kindness to API)
            time.sleep(0.05) 
            
        except Exception as e:
            logging.error(f"Error processing {addr}: {e}")
            
    # Final Save of Map
    map_df = pd.DataFrame(coord_map)
    map_df.to_csv(map_file, index=False)
    logging.info("Geocoding Complete. Map Saved.")
    
    # 4. Apply to Train/Test Data
    logging.info("Applying coordinates to dataset...")
    
    # Convert map to dictionary for mapping
    # Note: we used 'full_jibun' as key
    # Filter only successful geocodes
    valid_map = map_df[map_df['source'] != 'Fail'].set_index('address')
    
    mapper_x = valid_map['new_X'].to_dict()
    mapper_y = valid_map['new_Y'].to_dict()
    
    # Apply to Train
    train['full_jibun'] = train['시군구'].astype(str) + " " + train['번지'].fillna("").astype(str)
    
    # Update only where missing
    train_mask = train['좌표X'].isnull()
    train.loc[train_mask, '좌표X'] = train.loc[train_mask, 'full_jibun'].map(mapper_x)
    train.loc[train_mask, '좌표Y'] = train.loc[train_mask, 'full_jibun'].map(mapper_y)
    
    # Apply to Test
    test['full_jibun'] = test['시군구'].astype(str) + " " + test['번지'].fillna("").astype(str)
    
    test_mask = test['좌표X'].isnull()
    test.loc[test_mask, '좌표X'] = test.loc[test_mask, 'full_jibun'].map(mapper_x)
    test.loc[test_mask, '좌표Y'] = test.loc[test_mask, 'full_jibun'].map(mapper_y)
    
    # Drop temp col
    train.drop(columns=['full_jibun', 'is_train'], inplace=True)
    test.drop(columns=['full_jibun', 'is_train'], inplace=True)
    
    # Check remaining missing
    rem_train = train['좌표X'].isnull().sum()
    rem_test = test['좌표X'].isnull().sum()
    logging.info(f"Remaining Missing Coords - Train: {rem_train}, Test: {rem_test}")
    
    # Save Step 3 (Geocoded)
    # We name it 'train_step3.csv' as requested (Step 3 is now Geocoding)
    # But wait, previously Step 3 was feature engineering.
    # User's new plan: Step 3 is Geocoding -> Step 4 is Feature Engineering (Drop/Impute/Etc)
    # Actually User said: "Step 1. Geocoding ... Step 2. Drop ... Step 3. Feature ..."
    # But we are modifying the existing pipeline.
    # Let's save as 'train_step3_geocoded.csv' to be safe and clear.
    train.to_csv(os.path.join(OUTPUT_DIR, 'train_step3_geocoded.csv'), index=False, encoding='utf-8-sig')
    test.to_csv(os.path.join(OUTPUT_DIR, 'test_step3_geocoded.csv'), index=False, encoding='utf-8-sig')
    
    logging.info(f"Saved geocoded data to {OUTPUT_DIR}/train_step3_geocoded.csv")
