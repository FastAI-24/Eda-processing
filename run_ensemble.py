"""
앙상블 학습 실행 엔트리포인트

LightGBM + XGBoost + CatBoost 가중 평균 앙상블을 실행합니다.

실행 방법:
    cd house-price-prediction
    uv run python run_ensemble.py                    # 앙상블 학습
    uv run python run_ensemble.py --save-submission  # submission.csv 생성
    uv run python run_ensemble.py --n-splits 10      # 10-Fold CV
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# src/ 를 모듈 검색 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from modeling.config import ModelConfig  # noqa: E402
from modeling.ensemble import EnsembleTrainer  # noqa: E402


def _load_data(config: ModelConfig) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """전처리된 데이터를 로드합니다."""
    data_dir = config.data_dir

    x_train_path = data_dir / "X_train_preprocessed.csv"
    y_train_path = data_dir / "y_train_preprocessed.csv"
    x_test_path = data_dir / "X_test_preprocessed.csv"

    for path in [x_train_path, y_train_path, x_test_path]:
        if not path.exists():
            print(f"\n  ❌ 파일이 존재하지 않습니다: {path}")
            print("  전처리를 먼저 실행해주세요: python run.py --skip-eda")
            sys.exit(1)

    files = [
        ("X_train", x_train_path),
        ("y_train", y_train_path),
        ("X_test", x_test_path),
    ]

    data = {}
    for name, path in tqdm(
        files,
        desc="  📂 데이터 로딩",
        bar_format="  {l_bar}{bar:30}{r_bar}",
        ncols=100,
    ):
        data[name] = pd.read_csv(path)

    X_train = data["X_train"]
    y_train = data["y_train"]["target"]
    X_test = data["X_test"]

    print(f"  X_train: {X_train.shape}  |  y_train: {y_train.shape}  |  X_test: {X_test.shape}")
    return X_train, y_train, X_test


def _save_submission(
    predictions: np.ndarray,
    output_dir: Path,
    use_log_target: bool,
) -> None:
    """submission.csv를 저장합니다."""
    if use_log_target:
        predictions_original = np.expm1(predictions)
    else:
        predictions_original = predictions

    # 음수 클리핑 + 정수형 반올림
    predictions_original = np.maximum(predictions_original, 0)
    predictions_int = np.round(predictions_original).astype(np.int64)

    # 평가 시스템이 pred[["ID", "target"]]로 접근하므로 ID 컬럼 포함
    submission = pd.DataFrame({
        "ID": range(len(predictions_int)),
        "target": predictions_int,
    })
    submission_path = output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\n  📄 Submission 저장: {submission_path}")
    print(f"     Shape: {submission.shape}")
    print(f"     예측값 범위: [{predictions_int.min():,} ~ {predictions_int.max():,}] 만원")
    print(f"     예측값 평균: {predictions_int.mean():,.0f} 만원")


def main(
    n_splits: int = 5,
    save_submission: bool = False,
    models: list[str] | None = None,
) -> None:
    """앙상블 학습 파이프라인을 실행합니다."""
    total_start = time.time()

    config = ModelConfig(n_splits=n_splits)
    if models:
        config.ensemble_models = models

    print(f"\n{'━'*60}")
    print(f"  🚀 House Price Prediction — 앙상블 학습 파이프라인")
    print(f"{'━'*60}")

    # ── 데이터 로드 ──
    X_train, y_train, X_test = _load_data(config)

    # ── 앙상블 학습 ──
    ensemble_trainer = EnsembleTrainer(config)
    ensemble_result = ensemble_trainer.train_ensemble(X_train, y_train, X_test)

    # ── 개별 모델 피처 중요도 저장 ──
    for model_name, result in ensemble_result["results"].items():
        if result.feature_importances is not None:
            imp_path = config.output_dir / f"feature_importance_{model_name}.csv"
            result.feature_importances.to_csv(imp_path, index=False)
            tqdm.write(f"  💾 피처 중요도 저장: {imp_path.name}")

    # ── Submission 저장 ──
    if save_submission and ensemble_result["ensemble_test_predictions"] is not None:
        _save_submission(
            ensemble_result["ensemble_test_predictions"],
            config.output_dir,
            config.use_log_target,
        )

    total_elapsed = time.time() - total_start
    print(f"\n  ⏱  전체 소요 시간: {total_elapsed:.1f}초 ({total_elapsed / 60:.1f}분)")
    print(f"{'━'*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="앙상블 학습 (LightGBM + XGBoost + CatBoost)")
    parser.add_argument("--n-splits", type=int, default=5, help="K-Fold 분할 수")
    parser.add_argument("--save-submission", action="store_true", help="submission.csv 저장")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=["lightgbm", "xgboost", "catboost"],
        help="사용할 모델 (기본: 전체)",
    )
    args = parser.parse_args()
    main(n_splits=args.n_splits, save_submission=args.save_submission, models=args.models)
