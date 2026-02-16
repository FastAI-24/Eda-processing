"""
전처리 실행 엔트리포인트

노트북(preprocessing.ipynb)과 동일한 결과를 재현합니다.

실행 방법:
    cd house-price-prediction
    uv run python src/preprocessing/run_preprocessing.py

    # 또는 모듈 실행
    uv run python -m preprocessing
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .config import PreprocessingConfig
from .pipeline import PreprocessingPipeline
from .submission import create_baseline_submission
from .visualizer import (
    plot_correlation_heatmap,
    plot_missing_ratio,
    plot_numeric_features,
    plot_target_distribution,
    plot_train_test_comparison,
    setup_plot_style,
)


def main(
    config: PreprocessingConfig | None = None,
    skip_eda: bool = False,
    save_csv: bool = True,
) -> None:
    """전처리 파이프라인을 실행합니다.

    Args:
        config: 전처리 설정 (None이면 기본값)
        skip_eda: True면 EDA 시각화 건너뜀
        save_csv: True면 전처리 결과를 CSV로 저장
    """
    if config is None:
        config = PreprocessingConfig()

    # ── 시각화 스타일 설정 ──
    setup_plot_style()

    # ── 데이터 로드 ──
    data_dir = config.data_dir
    print(f"프로젝트 루트: {config.project_root}")
    print(f"데이터 디렉토리: {data_dir}")
    print(f"데이터 존재 여부: {data_dir.exists()}")

    if not data_dir.exists():
        print(f"\n[ERROR] 데이터 디렉토리가 존재하지 않습니다: {data_dir}")
        sys.exit(1)

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    print(f"\n학습 데이터: {train_df.shape}")
    print(f"테스트 데이터: {test_df.shape}")

    # ── EDA 시각화 (선택) ──
    if not skip_eda:
        print("\n" + "=" * 60)
        print("EDA (탐색적 데이터 분석)")
        print("=" * 60)

        # 기초 정보
        print("\n학습 데이터 기초 정보:")
        train_df.info(verbose=True, show_counts=True)

        # Target 분포
        plot_target_distribution(train_df["target"])

        # 결측 비율
        plot_missing_ratio(train_df)

        # 수치형 피처
        plot_numeric_features(train_df)

        # 상관관계 히트맵
        plot_correlation_heatmap(train_df)

        # Train/Test 비교
        plot_train_test_comparison(train_df, test_df)

    # ── 전처리 파이프라인 실행 ──
    pipeline = PreprocessingPipeline.create_default(config)
    ctx = pipeline.run(train_df, test_df)

    # ── CSV 저장 ──
    if save_csv:
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / "X_train_preprocessed.csv"
        ctx.X_train.to_csv(train_path, index=False, encoding="utf-8-sig")
        print(f"\n학습 피처 저장: {train_path}")
        print(f"  shape: {ctx.X_train.shape}")

        test_path = output_dir / "X_test_preprocessed.csv"
        ctx.X_test.to_csv(test_path, index=False, encoding="utf-8-sig")
        print(f"테스트 피처 저장: {test_path}")
        print(f"  shape: {ctx.X_test.shape}")

        target_path = output_dir / "y_train_preprocessed.csv"
        ctx.y_train.to_csv(target_path, index=False, encoding="utf-8-sig", header=["target"])
        print(f"학습 타겟 저장: {target_path}")
        print(f"  shape: {ctx.y_train.shape}")

        # 제출용 파일 (baseline: y_train median)
        submission_df = create_baseline_submission(
            ctx.y_train, n_test=len(ctx.X_test), id_offset=1
        )
        submission_path = output_dir / "submission.csv"
        submission_df.to_csv(submission_path, index=False, encoding="utf-8-sig")
        print(f"제출용 파일 저장: {submission_path}")
        print(f"  shape: {submission_df.shape} (baseline: median={ctx.y_train.median():,.0f} 만원)")

        print(f"\n전처리 결과 CSV 저장 완료 ({output_dir})")


if __name__ == "__main__":
    main()
