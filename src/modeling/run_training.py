"""
모델 학습 실행 엔트리포인트

전처리된 데이터를 로드하고, LightGBM으로 학습하고, RMSE를 평가합니다.

실행 방법:
    cd house-price-prediction
    uv run python run_train.py                  # LightGBM 5-Fold CV
    uv run python run_train.py --n-splits 10    # 10-Fold CV
    uv run python run_train.py --save-submission # submission.csv 생성
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ModelConfig
from .models import LightGBMModel
from .trainer import Trainer


def _load_data(config: ModelConfig) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """전처리된 데이터를 로드합니다."""
    data_dir = config.data_dir

    print(f"데이터 디렉토리: {data_dir}")

    x_train_path = data_dir / "X_train_preprocessed.csv"
    y_train_path = data_dir / "y_train_preprocessed.csv"
    x_test_path = data_dir / "X_test_preprocessed.csv"

    for path in [x_train_path, y_train_path, x_test_path]:
        if not path.exists():
            print(f"\n[ERROR] 파일이 존재하지 않습니다: {path}")
            print("전처리를 먼저 실행해주세요: uv run python run.py --skip-eda")
            sys.exit(1)

    print("데이터 로딩 중...")
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)["target"]
    X_test = pd.read_csv(x_test_path)

    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}")

    return X_train, y_train, X_test


def _save_submission(
    predictions: np.ndarray,
    n_test: int,
    output_dir: Path,
    use_log_target: bool,
) -> None:
    """submission.csv를 저장합니다.

    후처리:
        - log 역변환 (use_log_target일 때)
        - 음수 예측 클리핑 (0 이상)
        - 극단값 보정 (학습 데이터 범위 기반)
        - 정수형 반올림
    """
    if use_log_target:
        predictions_original = np.expm1(predictions)
    else:
        predictions_original = predictions

    # 음수 클리핑 (가격은 0 이상)
    n_negative = (predictions_original < 0).sum()
    if n_negative > 0:
        print(f"  [후처리] 음수 예측 {n_negative}건 → 0으로 클리핑")
    predictions_original = np.maximum(predictions_original, 0)

    # 정수형으로 반올림
    predictions_int = np.round(predictions_original).astype(np.int64)

    # 평가 시스템이 pred[["ID", "target"]]로 접근하므로 ID 컬럼 포함
    submission = pd.DataFrame({
        "ID": range(len(predictions_int)),
        "target": predictions_int,
    })

    submission_path = output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission 저장: {submission_path}")
    print(f"  Shape: {submission.shape}")
    print(f"  예측값 범위: [{predictions_int.min():,}, {predictions_int.max():,}] 만원")
    print(f"  예측값 평균: {predictions_int.mean():,.0f} 만원")


def _save_feature_importance(result, output_dir: Path) -> None:
    """피처 중요도를 저장합니다."""
    if result.feature_importances is not None:
        imp_path = output_dir / "feature_importance_lightgbm.csv"
        result.feature_importances.to_csv(imp_path, index=False)
        print(f"\n피처 중요도 저장: {imp_path}")
        print("  Top 15 피처:")
        for _, row in result.feature_importances.head(15).iterrows():
            print(f"    {row['feature']:<40s} {row['importance']:>10.1f}")


def main(
    n_splits: int = 5,
    save_submission: bool = False,
    config: ModelConfig | None = None,
) -> None:
    """LightGBM 학습 파이프라인을 실행합니다.

    Args:
        n_splits: K-Fold 분할 수
        save_submission: True면 submission.csv 저장
        config: 모델 설정 (None이면 기본값)
    """
    total_start = time.time()

    if config is None:
        config = ModelConfig(n_splits=n_splits)
    else:
        config.n_splits = n_splits

    # ── 데이터 로드 ──
    X_train, y_train, X_test = _load_data(config)

    # ── Trainer 생성 ──
    trainer = Trainer(config)

    # ── LightGBM 학습 ──
    result = trainer.train_with_cv(LightGBMModel, X_train, y_train, X_test)

    # ── 결과 출력 ──
    print(f"\n{'='*60}")
    print("최종 결과")
    print(f"{'='*60}")
    print(result.summary())

    # ── 피처 중요도 저장 ──
    _save_feature_importance(result, config.output_dir)

    # ── Submission 저장 ──
    if save_submission and result.test_predictions is not None:
        _save_submission(
            result.test_predictions,
            len(X_test),
            config.output_dir,
            config.use_log_target,
        )

    total_elapsed = time.time() - total_start
    print(f"\n전체 소요 시간: {total_elapsed:.1f}초 ({total_elapsed / 60:.1f}분)")


if __name__ == "__main__":
    main()
