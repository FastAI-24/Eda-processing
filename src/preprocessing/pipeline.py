"""
전처리 파이프라인 (Chain of Responsibility Pattern)

각 PreprocessingStep을 순차적으로 실행하는 오케스트레이터입니다.

디자인 패턴:
    - Chain of Responsibility: 각 Step이 Context를 받아 처리 후 다음 Step으로 전달
    - Factory Method: create_default_pipeline()으로 기본 파이프라인 생성

사용 예:
    from preprocessing.pipeline import PreprocessingPipeline

    # 기본 파이프라인 사용
    pipeline = PreprocessingPipeline.create_default()
    ctx = pipeline.run(train_df, test_df)

    # 커스텀 파이프라인 (특정 Step 추가)
    pipeline = PreprocessingPipeline()
    pipeline.add_step(FilterCancelledTransactionsStep())
    pipeline.add_step(TargetSeparationStep())
    pipeline.add_step(MyCustomStep())  # 팀원이 만든 커스텀 Step
    ctx = pipeline.run(train_df, test_df)
"""

from __future__ import annotations

import pandas as pd
import numpy as np


from .base import PreprocessingContext, PreprocessingStep
from .config import PreprocessingConfig
from .steps import (
    CoordinateInterpolationStep,
    DateAddressFeaturesStep,
    FilterCancelledTransactionsStep,
    FloatToIntConversionStep,
    IdentifyCategoricalColumnsStep,
    MissingIndicatorStep,
    MissingValueImputerStep,
    OutlierClippingStep,
    ParkingPerHouseholdStep,
    RemoveHighMissingColumnsStep,
    SanitizeColumnNamesStep,
    TargetLogTransformStep,
    TargetSeparationStep,
)


class PreprocessingPipeline:
    """전처리 단계들을 순차적으로 실행하는 파이프라인.

    Chain of Responsibility 패턴으로 각 Step이 Context를 처리합니다.
    """

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self._steps: list[PreprocessingStep] = []
        self._config = config or PreprocessingConfig()

    # ── Step 관리 ──
    def add_step(self, step: PreprocessingStep) -> "PreprocessingPipeline":
        """Step을 파이프라인 끝에 추가합니다 (Builder 패턴 체이닝)."""
        self._steps.append(step)
        return self

    def insert_step(self, index: int, step: PreprocessingStep) -> "PreprocessingPipeline":
        """특정 위치에 Step을 삽입합니다."""
        self._steps.insert(index, step)
        return self

    def remove_step(self, step_type: type) -> "PreprocessingPipeline":
        """특정 타입의 Step을 제거합니다."""
        self._steps = [s for s in self._steps if not isinstance(s, step_type)]
        return self

    @property
    def steps(self) -> list[PreprocessingStep]:
        return list(self._steps)

    # ── 실행 ──
    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> PreprocessingContext:
        """파이프라인의 모든 Step을 순차적으로 실행합니다.

        Args:
            train_df: 원본 학습 데이터
            test_df: 원본 테스트 데이터

        Returns:
            모든 전처리가 완료된 PreprocessingContext
        """
        ctx = PreprocessingContext(
            raw_train_df=train_df.copy(),
            raw_test_df=test_df.copy(),
            config=self._config,
        )

        total = len(self._steps)
        print("=" * 60)
        print(f"전처리 파이프라인 시작 (총 {total}단계)")
        print("=" * 60)

        for i, step in enumerate(self._steps, 1):
            print(f"\n{'─' * 60}")
            print(f"[{i}/{total}] {step.name}")
            print("─" * 60)
            ctx = step.execute(ctx)

        print(f"\n{'=' * 60}")
        print("전처리 파이프라인 완료!")
        print("=" * 60)
        self._print_summary(ctx)

        return ctx

    @staticmethod
    def _print_summary(ctx: PreprocessingContext) -> None:
        """전처리 결과 요약을 출력합니다."""
        print(f"\n[데이터 형상]")
        print(f"  X_train:     {ctx.X_train.shape}")
        if ctx.y_train_log is not None:
            print(f"  y_train_log: {ctx.y_train_log.shape}")
        print(f"  X_test:      {ctx.X_test.shape}")


        train_na = ctx.X_train.isnull().sum().sum()
        test_na = ctx.X_test.isnull().sum().sum()
        print(f"\n[잔여 결측]")
        print(f"  X_train: {train_na:,}건")
        print(f"  X_test:  {test_na:,}건")

        num_cols = ctx.X_train.select_dtypes(include=[np.number]).columns
        cat_cols_final = [c for c in ctx.categorical_cols if c in ctx.X_train.columns]
        print(f"\n[컬럼 구성]")
        print(f"  수치형: {len(num_cols)}개")
        print(f"  범주형: {len(cat_cols_final)}개")
        print(f"  전체:   {len(ctx.X_train.columns)}개")

        # Train/Test 컬럼 일치 확인
        train_cols = set(ctx.X_train.columns)
        test_cols = set(ctx.X_test.columns)
        only_train = train_cols - test_cols
        only_test = test_cols - train_cols
        print(f"\n[컬럼 일치]")
        print(f"  공통 컬럼: {len(train_cols & test_cols)}개")
        if only_train:
            print(f"  학습에만 있는 컬럼: {sorted(only_train)}")
        if only_test:
            print(f"  테스트에만 있는 컬럼: {sorted(only_test)}")
        if not only_train and not only_test:
            print("  학습/테스트 컬럼 완벽 일치!")

    # ── Factory Method ──
    @classmethod
    def create_default(
        cls, config: PreprocessingConfig | None = None,
    ) -> "PreprocessingPipeline":
        """노트북과 동일한 순서의 기본 파이프라인을 생성합니다.

        Pipeline:
            Step 0   → 취소 거래 필터링
            Step 1   → Target 분리
            Step 2   → 고결측 컬럼 제거
            Step 2.5 → Float→Int64 타입 변환
            Step 3   → 컬럼명 정리
            Step 3.5 → 날짜/주소 파생 피처
            Step 4   → 범주형 컬럼 식별
            Step 5   → 결측 지표 피처
            Step 6   → 좌표 보간
            Step 7   → 결측값 대체 (KNN Imputer)
            Step 7.5 → 세대당 주차대수
            Step 8   → 이상치 클리핑
            Step 9   → Target 로그 변환
        """
        pipeline = cls(config=config)
        pipeline.add_step(FilterCancelledTransactionsStep())
        pipeline.add_step(TargetSeparationStep())
        pipeline.add_step(RemoveHighMissingColumnsStep())
        pipeline.add_step(FloatToIntConversionStep())
        pipeline.add_step(SanitizeColumnNamesStep())
        pipeline.add_step(DateAddressFeaturesStep())
        pipeline.add_step(IdentifyCategoricalColumnsStep())
        pipeline.add_step(MissingIndicatorStep())
        pipeline.add_step(CoordinateInterpolationStep())
        pipeline.add_step(MissingValueImputerStep())
        pipeline.add_step(ParkingPerHouseholdStep())
        pipeline.add_step(OutlierClippingStep())
        pipeline.add_step(TargetLogTransformStep())
        return pipeline
