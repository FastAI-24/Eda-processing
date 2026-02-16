"""
Optuna 하이퍼파라미터 튜닝 실행 엔트리포인트

각 모델(LightGBM, XGBoost, CatBoost)에 대해 Optuna로 최적 하이퍼파라미터를 탐색합니다.

실행 방법:
    cd house-price-prediction
    uv run python run_tuning.py                              # 전체 모델 튜닝
    uv run python run_tuning.py --model lightgbm             # LightGBM만
    uv run python run_tuning.py --model xgboost --trials 50  # XGBoost 50회
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# src/ 를 모듈 검색 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from modeling.config import ModelConfig  # noqa: E402
from modeling.tuner import HyperparameterTuner  # noqa: E402


def _load_data(config: ModelConfig):
    data_dir = config.data_dir
    print(f"데이터 디렉토리: {data_dir}")

    X_train = pd.read_csv(data_dir / "X_train_preprocessed.csv")
    y_train = pd.read_csv(data_dir / "y_train_preprocessed.csv")["target"]

    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    return X_train, y_train


def main(
    models: list[str] | None = None,
    n_trials: int = 100,
    n_splits: int = 3,
) -> None:
    total_start = time.time()

    config = ModelConfig(n_splits=n_splits)
    X_train, y_train = _load_data(config)

    tuner = HyperparameterTuner(config)

    all_models = models or ["lightgbm", "xgboost", "catboost"]
    all_results = {}

    for model_name in all_models:
        try:
            best_params = tuner.tune(model_name, X_train, y_train, n_trials=n_trials)
            all_results[model_name] = best_params
        except Exception as e:
            print(f"[ERROR] {model_name} 튜닝 실패: {e}")

    # 결과 저장
    output_path = config.output_dir / "optuna_best_params.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n최적 파라미터 저장: {output_path}")

    total_elapsed = time.time() - total_start
    print(f"\n전체 소요 시간: {total_elapsed:.1f}초 ({total_elapsed / 60:.1f}분)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna 하이퍼파라미터 튜닝")
    parser.add_argument(
        "--model",
        nargs="+",
        default=None,
        choices=["lightgbm", "xgboost", "catboost"],
        help="튜닝할 모델 (기본: 전체)",
    )
    parser.add_argument("--trials", type=int, default=100, help="Optuna trial 수")
    parser.add_argument("--n-splits", type=int, default=3, help="CV fold 수 (튜닝은 3이 효율적)")
    args = parser.parse_args()
    main(models=args.model, n_trials=args.trials, n_splits=args.n_splits)
