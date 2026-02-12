"""
LightGBM 모델 구현 (Concrete Strategy Class)

BaseModel을 상속하며 독립적으로 테스트/교체 가능합니다.
LightGBM의 native categorical feature 지원을 활용합니다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseModel
from .config import ModelConfig


class LightGBMModel(BaseModel):
    """LightGBM 기반 회귀 모델.

    LightGBM의 native categorical feature 지원을 활용합니다.

    사용 예:
        model = LightGBMModel(config)
        model.train(X_train, y_train, X_val, y_val, categorical_features=cat_cols)
        predictions = model.predict(X_test)
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self._config = config or ModelConfig()
        self._model: Any = None
        self._params = dict(self._config.lgbm_params)
        self._feature_names: list[str] = []

    @property
    def name(self) -> str:
        return "LightGBM"

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        categorical_features: list[str] | None = None,
    ) -> Any:
        import lightgbm as lgb

        self._feature_names = list(X_train.columns)

        # 범주형 피처 처리: LightGBM native categorical
        cat_features = categorical_features or []
        cat_features = [c for c in cat_features if c in X_train.columns]

        # 모든 문자열/object 컬럼도 범주형에 포함
        for col in X_train.columns:
            if pd.api.types.is_string_dtype(X_train[col]) and col not in cat_features:
                cat_features.append(col)

        # 범주형 컬럼을 category dtype으로 변환 (결측값 처리 포함)
        X_train = X_train.copy()
        for col in cat_features:
            X_train[col] = X_train[col].fillna("__MISSING__").astype("category")

        callbacks = [
            lgb.log_evaluation(period=200),
            lgb.early_stopping(
                stopping_rounds=self._config.early_stopping_rounds,
                verbose=True,
            ),
        ]

        if X_val is not None and y_val is not None:
            X_val = X_val.copy()
            for col in cat_features:
                if col in X_val.columns:
                    X_val[col] = X_val[col].fillna("__MISSING__").astype("category")

            self._model = lgb.LGBMRegressor(**self._params)
            self._model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=callbacks,
                categorical_feature=cat_features if cat_features else "auto",
            )
        else:
            # Validation 없이 학습
            params_no_es = {k: v for k, v in self._params.items()}
            params_no_es["n_estimators"] = 1000  # Early stopping 없으므로 줄임
            self._model = lgb.LGBMRegressor(**params_no_es)
            self._model.fit(
                X_train,
                y_train,
                categorical_feature=cat_features if cat_features else "auto",
            )

        print(f"    Best iteration: {self._model.best_iteration_}")
        return self._model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")

        X = X.copy()
        # 학습 시 사용한 범주형 컬럼을 동일하게 변환
        for col in X.columns:
            if pd.api.types.is_string_dtype(X[col]) or X[col].dtype == object:
                X[col] = X[col].fillna("__MISSING__").astype("category")

        return self._model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame | None:
        if self._model is None:
            return None
        importance = self._model.feature_importances_
        return pd.DataFrame({
            "feature": self._feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
