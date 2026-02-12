"""
전처리 실행 엔트리포인트

실행 방법:
    uv run python run.py
    uv run python run.py --skip-eda
    uv run python run.py --no-save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from preprocessing.run_preprocessing import main  # noqa: E402

# src/ 를 모듈 검색 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="전처리 파이프라인 실행")
    parser.add_argument("--skip-eda", action="store_true", help="EDA 시각화 건너뜀")
    parser.add_argument("--no-save", action="store_true", help="CSV 저장 건너뜀")
    args = parser.parse_args()

    main(skip_eda=args.skip_eda, save_csv=not args.no_save)
