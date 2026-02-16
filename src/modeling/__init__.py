"""
모델 학습 패키지 (house-price-prediction)

LightGBM 기반 K-Fold 교차 검증 학습 및 RMSE 평가를 수행합니다.

디자인 패턴:
    - Strategy Pattern: BaseModel 인터페이스를 통해 모델 교체 가능
    - Template Method: Trainer가 학습/평가 흐름을 제어
    - Factory Method: 모델 클래스를 Trainer에 전달하여 Fold별 인스턴스 생성

사용 예:
    from modeling import Trainer, LightGBMModel, ModelConfig

    config = ModelConfig(n_splits=5)
    trainer = Trainer(config)

    result = trainer.train_with_cv(LightGBMModel, X_train, y_train, X_test)
    print(result.summary())
"""

from .base import BaseModel, TrainingResult
from .config import ModelConfig
from .ensemble import EnsembleTrainer
from .models import CatBoostModel, LightGBMModel, XGBoostModel
from .trainer import Trainer
from .tuner import HyperparameterTuner

__all__ = [
    "ModelConfig",
    "BaseModel",
    "TrainingResult",
    "Trainer",
    "LightGBMModel",
    "XGBoostModel",
    "CatBoostModel",
    "EnsembleTrainer",
    "HyperparameterTuner",
]
