"""
전처리 패키지 (house-price-prediction)

디자인 패턴:
    - Strategy Pattern: 각 Step이 동일한 인터페이스(PreprocessingStep)를 구현
    - Chain of Responsibility: Pipeline이 Step들을 순차 실행
    - Factory Method: PreprocessingPipeline.create_default()로 기본 파이프라인 생성
    - Builder Pattern: pipeline.add_step()으로 커스텀 파이프라인 구성

사용 예:
    from preprocessing import PreprocessingPipeline, PreprocessingConfig

    config = PreprocessingConfig()
    pipeline = PreprocessingPipeline.create_default(config)
    ctx = pipeline.run(train_df, test_df)

    X_train = ctx.X_train
    y_train_log = ctx.y_train_log
    X_test = ctx.X_test
"""

from .base import PreprocessingContext, PreprocessingStep
from .config import PreprocessingConfig
from .pipeline import PreprocessingPipeline
from .submission import create_baseline_submission, create_submission
from .steps import (
    CoordinateInterpolationStep,
    DateAddressFeaturesStep,
    FilterCancelledTransactionsStep,
    FloatToIntConversionStep,
    IdentifyCategoricalColumnsStep,
    InteractionFeaturesStep,
    MissingIndicatorStep,
    MissingValueImputerStep,
    OutlierClippingStep,
    ParkingPerHouseholdStep,
    RemoveHighMissingColumnsStep,
    SanitizeColumnNamesStep,
    SpatialFeaturesStep,
    TargetEncodingStep,
    TargetLogTransformStep,
    TargetSeparationStep,
    TemporalFeaturesStep,
)

__all__ = [
    # 핵심 클래스
    "PreprocessingConfig",
    "PreprocessingContext",
    "PreprocessingPipeline",
    "PreprocessingStep",
    # 개별 Step (커스텀 파이프라인 구성용)
    "FilterCancelledTransactionsStep",
    "TargetSeparationStep",
    "RemoveHighMissingColumnsStep",
    "FloatToIntConversionStep",
    "SanitizeColumnNamesStep",
    "DateAddressFeaturesStep",
    "TemporalFeaturesStep",
    "IdentifyCategoricalColumnsStep",
    "MissingIndicatorStep",
    "CoordinateInterpolationStep",
    "SpatialFeaturesStep",
    "MissingValueImputerStep",
    "ParkingPerHouseholdStep",
    "InteractionFeaturesStep",
    "TargetEncodingStep",
    "OutlierClippingStep",
    "TargetLogTransformStep",
    "create_submission",
    "create_baseline_submission",
]
