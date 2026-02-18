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
    y_train: np.ndarray | None = None,
) -> None:
    """submission.csv를 저장합니다 (성능 최적화 Phase 4: 예측값 클리핑)."""
    if use_log_target:
        predictions_original = np.expm1(predictions)
    else:
        predictions_original = predictions.copy()

    # 음수 클리핑
    predictions_original = np.maximum(predictions_original, 0)
    # 학습 target 범위 기반 클리핑 (극단값 보정)
    if y_train is not None and len(y_train) > 0:
        y_min, y_max = float(y_train.min()), float(y_train.max())
        clip_lo = max(0, y_min * 0.5)
        clip_hi = y_max * 1.5
        predictions_original = np.clip(predictions_original, clip_lo, clip_hi)
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
    use_stacking: bool = False,
    use_multi_seed: bool = False,
    use_pseudo_labeling: bool = False,
    use_quantile: bool = False,
    use_mlp: bool = False,
    optimized: bool = False,
    cv_strategy: str = "kfold",
    no_tuned_params: bool = False,
) -> None:
    """앙상블 학습 파이프라인을 실행합니다."""
    total_start = time.time()

    config = ModelConfig(n_splits=n_splits)
    config.cv_strategy = cv_strategy
    if models:
        config.ensemble_models = models
    else:
        base = ["lightgbm", "xgboost", "catboost"]
        if use_quantile:
            base.append("lightgbm_quantile")
        if use_mlp:
            base.append("mlp")
        config.ensemble_models = base
    if use_stacking or optimized:
        config.ensemble_strategy = "stacking"
    if use_multi_seed or optimized:
        config.ensemble_use_multi_seed = True
    if optimized:
        config.use_fold_time_lag = True
    if use_pseudo_labeling:
        config.use_pseudo_labeling = True

    # 개별 모델 성능 향상: Optuna 튜닝 결과 자동 적용 (--no-tuned-params 시 건너뜀)
    if not no_tuned_params:
        config.apply_tuned_params()

    print(f"\n{'━'*60}")
    print(f"  🚀 House Price Prediction — 앙상블 학습 파이프라인")
    print(f"{'━'*60}")

    # ── 데이터 로드 ──
    X_train, y_train, X_test = _load_data(config)

    # ── 앙상블 학습 ──
    ensemble_trainer = EnsembleTrainer(config)
    ensemble_result = ensemble_trainer.train_ensemble(X_train, y_train, X_test)

    # ── Pseudo Labeling (Exp10) ──
    if config.use_pseudo_labeling and ensemble_result["ensemble_test_predictions"] is not None:
        ratio = config.pseudo_label_ratio
        n_pseudo = max(1, int(len(X_test) * ratio))
        pred_log = ensemble_result["ensemble_test_predictions"]
        median = np.median(pred_log)
        dist = np.abs(pred_log - median)
        idx = np.argsort(dist)[:n_pseudo]
        X_pseudo = X_test.iloc[idx].reset_index(drop=True)
        # 모델은 log1p(y) 학습 → pseudo도 원본 스케일로 변환 (만원)
        y_pseudo_original = np.expm1(pred_log[idx])
        y_pseudo = pd.Series(y_pseudo_original, index=X_pseudo.index)
        X_train_aug = pd.concat([X_train, X_pseudo], ignore_index=True)
        y_train_aug = pd.concat([y_train, y_pseudo], ignore_index=True)
        print(f"\n  📌 Pseudo Labeling: {n_pseudo}건 추가 후 재학습")
        ensemble_result = ensemble_trainer.train_ensemble(X_train_aug, y_train_aug, X_test)

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
            y_train=y_train.values if hasattr(y_train, "values") else y_train,
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
        choices=["lightgbm", "lightgbm_quantile", "xgboost", "catboost", "mlp"],
        help="사용할 모델 (기본: lightgbm, xgboost, catboost)",
    )
    parser.add_argument(
        "--stacking",
        action="store_true",
        help="Stacking 앙상블 사용 (Ridge 메타 학습기)",
    )
    parser.add_argument(
        "--multi-seed",
        action="store_true",
        help="Multi-seed 앙상블 사용 (5개 시드 평균)",
    )
    parser.add_argument(
        "--pseudo-labeling",
        action="store_true",
        help="Pseudo Labeling 사용 (테스트 데이터 반복 학습)",
    )
    parser.add_argument(
        "--quantile",
        action="store_true",
        help="LightGBM Quantile Regression 모델 추가",
    )
    parser.add_argument(
        "--mlp",
        action="store_true",
        help="MLP 신경망 모델 추가",
    )
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="최적화 모드: multi-seed + stacking + time-lag 한번에 활성화",
    )
    parser.add_argument(
        "--cv-strategy",
        type=str,
        default="kfold",
        choices=["kfold", "timeseries"],
        help="CV 전략 (기본: kfold)",
    )
    parser.add_argument(
        "--no-tuned-params",
        action="store_true",
        help="Optuna 튜닝 파라미터 미적용",
    )
    args = parser.parse_args()
    main(
        n_splits=args.n_splits,
        save_submission=args.save_submission,
        models=args.models,
        use_stacking=args.stacking,
        use_multi_seed=args.multi_seed,
        use_pseudo_labeling=args.pseudo_labeling,
        use_quantile=args.quantile,
        use_mlp=args.mlp,
        optimized=args.optimized,
        cv_strategy=args.cv_strategy,
        no_tuned_params=args.no_tuned_params,
    )
