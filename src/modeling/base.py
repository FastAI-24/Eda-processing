"""
모델 추상 기본 클래스 (Strategy Pattern)

각 모델을 독립적인 클래스로 캡슐화하여,
팀원이 특정 모델만 수정/교체/추가할 수 있게 합니다.

디자인 패턴:
    - Strategy Pattern: 각 Model이 동일한 인터페이스(train, predict)를 구현
    - Template Method: 공통 로깅/검증 로직을 기본 클래스에서 제공
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TrainingResult:
    """학습 결과를 담는 데이터 클래스.

    각 Fold 또는 전체 학습의 결과를 구조화합니다.
    """

    model_name: str = ""
    fold_scores: list[float] = field(default_factory=list)
    mean_rmse: float = 0.0
    std_rmse: float = 0.0
    oof_predictions: np.ndarray | None = None
    test_predictions: np.ndarray | None = None
    feature_importances: pd.DataFrame | None = None
    trained_models: list[Any] = field(default_factory=list)

    def summary(self) -> str:
        """결과 요약 문자열을 반환합니다."""
        lines = [
            f"모델: {self.model_name}",
            f"평균 RMSE: {self.mean_rmse:.6f} (±{self.std_rmse:.6f})",
        ]
        if self.fold_scores:
            lines.append("Fold별 RMSE:")
            for i, score in enumerate(self.fold_scores, 1):
                lines.append(f"  Fold {i}: {score:.6f}")
        return "\n".join(lines)


class BaseModel(ABC):
    """모델의 추상 기본 클래스.

    모든 모델은 이 클래스를 상속하고 train(), predict()를 구현합니다.

    사용 예:
        class MyModel(BaseModel):
            @property
            def name(self) -> str:
                return "My Custom Model"

            def train(self, X_train, y_train, X_val, y_val, ...):
                # 학습 로직
                ...

            def predict(self, X):
                return self.model.predict(X)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """모델 이름 (로그 출력용)."""
        ...

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        categorical_features: list[str] | None = None,
    ) -> Any:
        """모델을 학습합니다.

        Args:
            X_train: 학습 피처
            y_train: 학습 타겟
            X_val: 검증 피처 (early stopping용)
            y_val: 검증 타겟
            categorical_features: 범주형 피처 목록

        Returns:
            학습된 모델 객체
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측을 수행합니다.

        Args:
            X: 예측할 피처 데이터

        Returns:
            예측값 배열
        """
        ...

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame | None:
        """피처 중요도를 반환합니다.

        Returns:
            feature, importance 컬럼을 가진 DataFrame (없으면 None)
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
