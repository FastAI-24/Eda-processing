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
    AdversarialValidationStep,
    BrandFeatureStep,
    CoordinateInterpolationStep,
    CoordinateOutlierDetectionStep,
    DateAddressFeaturesStep,
    FeatureSelectionStep,
    FilterCancelledTransactionsStep,
    FloatToIntConversionStep,
    HanRiverDistanceStep,
    IdentifyCategoricalColumnsStep,
    InteractionFeaturesStep,
    LabelEncodingStep,
    LowImportanceFeatureRemovalStep,
    MissingIndicatorStep,
    MissingValueImputerStep,
    MulticollinearityRemovalStep,
    OutlierClippingStep,
    ParkingPerHouseholdStep,
    ParkingPredictionStep,
    QualityFeaturesStep,
    RecentDataFilterStep,
    RemoveHighMissingColumnsStep,
    SanitizeColumnNamesStep,
    SpatialClusteringStep,
    SpatialFeaturesStep,
    TargetEncodingStep,
    TargetLogTransformStep,
    TargetSeparationStep,
    TemporalFeaturesStep,
    TimeSeriesFeaturesStep,
    TransitFeaturesStep,
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
        """최적화된 기본 파이프라인을 생성합니다.

        docs/PREPROCESSING_PIPELINE.md 기준 21단계 파이프라인:

        [핵심 15단계]
            1.  취소 거래 제거 (FilterCancelledTransactions)
            2.  불필요 컬럼 제거 (RemoveHighMissingColumns)
            3.  다중공선성 제거 (MulticollinearityRemoval)
            4.  타겟 변수 분리 (TargetSeparation)
            5.  이상치 탐지: 전용면적 — Step 8에서 통합 처리
            6.  이상치 탐지: 좌표 (CoordinateOutlierDetection)
            7~9. 지오코딩 (CoordinateInterpolation — 주소 압축+API+Spatial Median)
            10. 결측치 모델링: 주차대수 (ParkingPrediction — RandomForest)
            11. 파생변수 생성: 시간 (TemporalFeatures — days_since 포함)
            12. 파생변수 생성: 브랜드 (BrandFeature — is_top_brand)
            13. 범주형 인코딩: High Cardinality — Fold 내 TE (trainer.py)
            14. 범주형 인코딩: Low Cardinality (LabelEncoding)
            15. 타겟 로그 변환 (TargetLogTransform)

        [확장 단계]
            + 공간 파생 피처 (Golden Triangle min_dist_to_job)
            + 교통 거리 피처 (BallTree)
            + K-Means 좌표 클러스터링
            + 단지 품질 / 교호작용 / 도메인 피처
            + 이상치 클리핑 + Feature Diet

        Note:
            TargetEncodingStep(High Cardinality)은 CV 누수 방지를 위해
            전처리에서 제외. trainer.py에서 각 Fold 내부에서 수행합니다.
            coord_cluster는 Fold 내 TE 대상으로 ModelConfig에 등록됩니다.
        """
        pipeline = cls(config=config)
        # ── 핵심 파이프라인 (21단계 기반) ──
        pipeline.add_step(FilterCancelledTransactionsStep())       # 1. 취소 거래 제거
        pipeline.add_step(TargetSeparationStep())                  # 4. 타겟 변수 분리 (X_train/X_test 생성)
        pipeline.add_step(RemoveHighMissingColumnsStep())          # 2. 불필요 컬럼 제거
        pipeline.add_step(MulticollinearityRemovalStep())          # 3. 다중공선성 제거
        pipeline.add_step(FloatToIntConversionStep())              #    타입 정리
        pipeline.add_step(SanitizeColumnNamesStep())               #    컬럼명 정리 (LightGBM 호환)
        pipeline.add_step(DateAddressFeaturesStep())               #    날짜/주소 파생 피처
        pipeline.add_step(TemporalFeaturesStep())                  # 11. 시간 파생 (days_since 포함)
        pipeline.add_step(RecentDataFilterStep())                  #    Exp06: 2017+ 필터링
        pipeline.add_step(IdentifyCategoricalColumnsStep())        #    범주형 컬럼 식별
        pipeline.add_step(MissingIndicatorStep())                  #    결측 지표 피처
        pipeline.add_step(CoordinateOutlierDetectionStep())        # 6. 좌표 이상치 탐지 (서울 경계)
        pipeline.add_step(CoordinateInterpolationStep())           # 7~9. 지오코딩 (API+보간)
        pipeline.add_step(SpatialFeaturesStep())                   #    Golden Triangle + min_dist_to_job
        pipeline.add_step(TransitFeaturesStep())                   #    교통 거리 피처 (BallTree)
        pipeline.add_step(HanRiverDistanceStep())                  #    한강 거리
        pipeline.add_step(SpatialClusteringStep())                 #    K-Means 좌표 클러스터링
        pipeline.add_step(ParkingPredictionStep())                 # 10. 주차대수 RF 예측
        pipeline.add_step(MissingValueImputerStep())               #    나머지 결측값 Median 대체
        pipeline.add_step(ParkingPerHouseholdStep())               #    세대당 주차대수
        pipeline.add_step(QualityFeaturesStep())                   #    단지 품질 피처
        pipeline.add_step(BrandFeatureStep())                      # 12. is_top_brand
        pipeline.add_step(InteractionFeaturesStep())               #    교호작용 + 도메인 피처
        pipeline.add_step(TimeSeriesFeaturesStep())                #    시계열 피처
        pipeline.add_step(OutlierClippingStep())                   # 5+8. 이상치 클리핑 (IQR)
        pipeline.add_step(LabelEncodingStep())                     # 14. Low Cardinality Label Encoding
        pipeline.add_step(AdversarialValidationStep())             #    AV 기반 피처 제거
        pipeline.add_step(FeatureSelectionStep())                  #    Permutation/SHAP 피처 선택
        pipeline.add_step(LowImportanceFeatureRemovalStep())       #    Feature Diet (Exp07)
        pipeline.add_step(TargetLogTransformStep())                # 15. 타겟 로그 변환
        return pipeline
