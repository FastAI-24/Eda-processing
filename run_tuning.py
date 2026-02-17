"""
Optuna 하이퍼파라미터 튜닝 실행 엔트리포인트

각 모델(LightGBM, XGBoost, CatBoost)에 대해 Optuna로 최적 하이퍼파라미터를 탐색합니다.
Coarse → Fine 2단계 탐색을 지원합니다.

실행 방법:
    cd house-price-prediction

    # Coarse 탐색 (빠르게)
    uv run python run_tuning.py --trials 50 --n-splits 3

    # Fine 탐색 (정밀하게)
    uv run python run_tuning.py --trials 100 --n-splits 5

    # 특정 모델만
    uv run python run_tuning.py --model lightgbm --trials 50
    uv run python run_tuning.py --model xgboost catboost --trials 100

    # 기존 결과에 추가 (이미 저장된 파라미터를 유지하면서 특정 모델만 재튜닝)
    uv run python run_tuning.py --model catboost --trials 200 --append
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
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
    append: bool = False,
) -> None:
    total_start = time.time()

    config = ModelConfig(n_splits=n_splits)
    X_train, y_train = _load_data(config)

    tuner = HyperparameterTuner(config)

    all_models = models or ["lightgbm", "xgboost", "catboost"]
    output_path = config.output_dir / "optuna_best_params.json"

    # --append 모드: 기존 결과를 유지하면서 특정 모델만 업데이트
    if append and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"기존 파라미터 로드: {output_path}")
        print(f"  기존 모델: {[k for k in all_results if k != 'tuning_meta']}")
    else:
        all_results = {}

    for model_name in all_models:
        try:
            best_params = tuner.tune(model_name, X_train, y_train, n_trials=n_trials)
            all_results[model_name] = best_params
        except Exception as e:
            print(f"[ERROR] {model_name} 튜닝 실패: {e}")

    total_elapsed = time.time() - total_start

    # 메타 정보 추가
    all_results["tuning_meta"] = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_trials": n_trials,
        "n_splits": n_splits,
        "cv_strategy": config.cv_strategy,
        "data_shape": list(X_train.shape),
        "elapsed_seconds": round(total_elapsed, 1),
        "models_tuned": [m for m in all_models if m in all_results],
    }

    # 결과 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n최적 파라미터 저장: {output_path}")

    # 요약 출력
    print(f"\n{'='*60}")
    print("튜닝 결과 요약")
    print(f"{'='*60}")
    for model_name in all_models:
        if model_name in all_results:
            params = all_results[model_name]
            cv_rmse = params.get("cv_rmse", "N/A")
            n_params = len([k for k in params if k != "cv_rmse"])
            print(f"  {model_name:<12s}  CV RMSE={cv_rmse}  ({n_params}개 파라미터)")
    print(f"{'='*60}")
    print(f"전체 소요 시간: {total_elapsed:.1f}초 ({total_elapsed / 60:.1f}분)")


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
    parser.add_argument("--append", action="store_true", help="기존 결과에 추가 (특정 모델만 재튜닝)")
    args = parser.parse_args()
    main(models=args.model, n_trials=args.trials, n_splits=args.n_splits, append=args.append)
