"""
전처리 단계 추상 기본 클래스 (Strategy Pattern)

각 전처리 단계를 독립적인 클래스로 캡슐화하여,
팀원이 특정 단계만 수정/교체/추가할 수 있게 합니다.

디자인 패턴:
    - Strategy Pattern: 각 Step이 동일한 인터페이스(execute)를 구현
    - Template Method: 공통 로깅/검증 로직을 기본 클래스에서 제공
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .config import PreprocessingConfig


@dataclass
class PreprocessingContext:
    """전처리 파이프라인에서 공유되는 컨텍스트 객체.

    각 Step 간 데이터를 전달하는 Mediator 역할을 합니다.
    """

    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: pd.Series | None = None
    y_train_log: pd.Series | None = None
    categorical_cols: list[str] = field(default_factory=list)
    train_stats: dict[str, Any] = field(default_factory=dict)
    config: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    # 원본 데이터 (시각화/검증용)
    raw_train_df: pd.DataFrame | None = None
    raw_test_df: pd.DataFrame | None = None


class PreprocessingStep(ABC):
    """전처리 단계의 추상 기본 클래스.

    모든 전처리 단계는 이 클래스를 상속하고 execute()를 구현합니다.

    사용 예:
        class MyCustomStep(PreprocessingStep):
            @property
            def name(self) -> str:
                return "커스텀 단계"

            def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
                # 전처리 로직 구현
                return ctx
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """단계 이름 (로그 출력용)."""
        ...

    @abstractmethod
    def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
        """전처리 로직을 실행하고 컨텍스트를 반환합니다."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
