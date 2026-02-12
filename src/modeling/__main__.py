"""모델 학습 패키지를 모듈로 실행할 수 있게 합니다.

사용법:
    uv run python -m modeling
    uv run python -m modeling --n-splits 10
    uv run python -m modeling --save-submission
"""

from .run_training import main

if __name__ == "__main__":
    import argparse

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
