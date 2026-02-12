"""
LightGBM 학습 실행 엔트리포인트

실행 방법:
    uv run python run_train.py                  # LightGBM 5-Fold CV
    uv run python run_train.py --n-splits 10    # 10-Fold CV
    uv run python run_train.py --save-submission # submission.csv 생성
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# src/ 를 모듈 검색 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from modeling.run_training import main  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBM 학습 및 RMSE 평가")
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="K-Fold 분할 수 (기본: 5)",
    )
    parser.add_argument(
        "--save-submission",
        action="store_true",
        help="submission.csv 저장",
    )
    args = parser.parse_args()

    main(
        n_splits=args.n_splits,
        save_submission=args.save_submission,
    )
