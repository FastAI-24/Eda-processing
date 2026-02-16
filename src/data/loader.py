"""데이터 로드 함수"""
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import sys

from src.config import DATA_DIR


def _check_data_file(filepath: Path, filename: str):
    """데이터 파일 존재 여부 확인 및 친절한 에러 메시지 제공"""
    if not filepath.exists():
        print("\n" + "=" * 70)
        print(f"오류: {filename}을 찾을 수 없습니다.")
        print("=" * 70)
        print(f"예상 경로: {filepath}")
        print(f"\n데이터 디렉토리: {DATA_DIR}")
        
        if not DATA_DIR.exists():
            print(f"\n경고: 데이터 디렉토리가 존재하지 않습니다.")
            print(f"다음 명령어로 디렉토리를 생성하세요:")
            print(f"  mkdir -p {DATA_DIR}")
        
        print("\n데이터 다운로드:")
        print("  데이터 URL: https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000420/data/data.tar")
        print("  다운로드 후 압축 해제하여 다음 위치에 배치하세요:")
        print(f"    {DATA_DIR}/")
        print("\n필요한 파일:")
        print("  - train.csv")
        print("  - test.csv")
        print("  - bus_feature.csv")
        print("  - subway_feature.csv")
        print("  - sample_submission.csv")
        print("=" * 70 + "\n")
        raise FileNotFoundError(f"{filename}을 찾을 수 없습니다: {filepath}")


def load_train_data() -> pd.DataFrame:
    """학습 데이터 로드"""
    train_path = DATA_DIR / "train.csv"
    _check_data_file(train_path, "train.csv")
    return pd.read_csv(train_path, low_memory=False)


def load_test_data() -> pd.DataFrame:
    """테스트 데이터 로드"""
    test_path = DATA_DIR / "test.csv"
    _check_data_file(test_path, "test.csv")
    return pd.read_csv(test_path)


def load_bus_features() -> pd.DataFrame:
    """버스 정류소 피처 로드"""
    bus_path = DATA_DIR / "bus_feature.csv"
    _check_data_file(bus_path, "bus_feature.csv")
    return pd.read_csv(bus_path)


def load_subway_features() -> pd.DataFrame:
    """지하철역 피처 로드"""
    subway_path = DATA_DIR / "subway_feature.csv"
    _check_data_file(subway_path, "subway_feature.csv")
    return pd.read_csv(subway_path)


def load_sample_submission() -> pd.DataFrame:
    """샘플 제출 파일 로드"""
    sample_path = DATA_DIR / "sample_submission.csv"
    _check_data_file(sample_path, "sample_submission.csv")
    return pd.read_csv(sample_path)


def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """모든 데이터 로드"""
    train = load_train_data()
    test = load_test_data()
    bus = load_bus_features()
    subway = load_subway_features()
    return train, test, bus, subway
