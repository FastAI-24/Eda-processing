"""
모델 구현 (Concrete Strategy Classes)

BaseModel을 상속하며 독립적으로 테스트/교체 가능합니다.

모델:
    - LightGBMModel: LightGBM GBDT (native categorical, GPU 지원)
    - XGBoostModel:  XGBoost hist (GPU 가속)
    - CatBoostModel: CatBoost (native categorical, GPU 가속)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .base import BaseModel
from .config import ModelConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# tqdm 기반 학습 콜백
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TqdmLGBMCallback:
    """LightGBM 학습 진행 tqdm 콜백."""

    def __init__(self, n_estimators: int, desc: str = "LightGBM") -> None:
        self._pbar = tqdm(
            total=n_estimators,
            desc=f"      {desc}",
            bar_format="      {l_bar}{bar:25}{r_bar}",
            ncols=95,
            leave=False,
        )
        self._best_score: float | None = None

    def __call__(self, env: Any) -> None:
        iteration = env.iteration + 1
        self._pbar.update(1)

        if env.evaluation_result_list:
            # (dataset, metric, value, is_higher_better)
            score = env.evaluation_result_list[0][2]
            self._best_score = score
            self._pbar.set_postfix_str(f"rmse={score:.6f}")

        if iteration == self._pbar.total:
            self._pbar.close()

    def close(self) -> None:
        self._pbar.close()


def _make_xgb_tqdm_callback(n_estimators: int, desc: str = "XGBoost") -> Any:
    """XGBoost TrainingCallback을 동적으로 생성합니다.

    xgboost import를 지연시켜 모듈 로드 시점 에러를 방지합니다.
    """
    import xgboost as xgb

    class _TqdmXGBCallback(xgb.callback.TrainingCallback):
        """XGBoost 학습 진행 tqdm 콜백."""

        def __init__(self) -> None:
            super().__init__()
            self._pbar = tqdm(
                total=n_estimators,
                desc=f"      {desc}",
                bar_format="      {l_bar}{bar:25}{r_bar}",
                ncols=95,
                leave=False,
            )

        def after_iteration(self, model: Any, epoch: int, evals_log: dict) -> bool:
            self._pbar.update(1)
            for data_name, metrics in evals_log.items():
                for metric_name, values in metrics.items():
                    if values:
                        score = values[-1]
                        self._pbar.set_postfix_str(f"{metric_name}={score:.6f}")
            return False

        def before_training(self, model: Any) -> Any:
            return model

        def after_training(self, model: Any) -> Any:
            self._pbar.close()
            return model

    return _TqdmXGBCallback()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LightGBM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class LightGBMModel(BaseModel):
    """LightGBM 기반 회귀 모델.

    LightGBM의 native categorical feature 지원을 활용합니다.
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
        sample_weight: np.ndarray | None = None,
    ) -> Any:
        import lightgbm as lgb

        self._feature_names = list(X_train.columns)

        # 범주형 피처 처리: LightGBM native categorical
        cat_features = categorical_features or []
        cat_features = [c for c in cat_features if c in X_train.columns]

        for col in X_train.columns:
            if pd.api.types.is_string_dtype(X_train[col]) and col not in cat_features:
                cat_features.append(col)

        X_train = X_train.copy()
        for col in cat_features:
            X_train[col] = X_train[col].fillna("__MISSING__").astype("category")

        n_est = self._params.get("n_estimators", 5000)
        tqdm_cb = TqdmLGBMCallback(n_est, desc="LightGBM")
        callbacks = [
            tqdm_cb,
            lgb.early_stopping(
                stopping_rounds=self._config.early_stopping_rounds,
                verbose=False,
            ),
        ]

        fit_params: dict[str, Any] = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

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
                **fit_params,
            )
        else:
            params_no_es = dict(self._params)
            params_no_es["n_estimators"] = 1000
            tqdm_cb.close()
            self._model = lgb.LGBMRegressor(**params_no_es)
            self._model.fit(
                X_train,
                y_train,
                categorical_feature=cat_features if cat_features else "auto",
                **fit_params,
            )

        tqdm_cb.close()
        tqdm.write(f"      best_iter={self._model.best_iteration_}")
        return self._model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")

        X = X.copy()
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# XGBoost
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class XGBoostModel(BaseModel):
    """XGBoost 기반 회귀 모델.

    한글 컬럼명/값 인코딩 이슈를 피하기 위해:
    1) 컬럼명 → ASCII (f0, f1, ...)
    2) 모든 object/string 값 → Label Encoding (정수형)
    3) 최종적으로 DataFrame 전체가 순수 숫자형인지 검증
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self._config = config or ModelConfig()
        self._model: Any = None
        self._params = dict(self._config.xgb_params)
        self._feature_names: list[str] = []
        self._col_map: dict[str, str] = {}
        self._label_maps: dict[str, dict[str, int]] = {}

    @property
    def name(self) -> str:
        return "XGBoost"

    def _to_xgb_safe(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """DataFrame을 XGBoost-safe 형태로 변환.

        1) 컬럼명 → ASCII
        2) 모든 object/string/category 컬럼 → Label Encoding (int32)
        3) Nullable Int → float64
        """
        # 1) 컬럼명 ASCII 매핑
        if fit:
            self._col_map = {c: f"f{i}" for i, c in enumerate(df.columns)}
        df = df.rename(columns=self._col_map).copy()

        # 2) 모든 비숫자 컬럼 자동 감지 및 Label Encoding
        for col in df.columns:
            dtype = df[col].dtype
            dtype_str = str(dtype)

            # object, string, category → Label Encoding
            if dtype == object or dtype_str in ("string", "str", "category"):
                df[col] = df[col].fillna("__MISSING__").astype(str)
                if fit:
                    unique_vals = sorted(df[col].unique(), key=str)
                    self._label_maps[col] = {v: i for i, v in enumerate(unique_vals)}
                mapping = self._label_maps.get(col, {})
                df[col] = df[col].map(mapping).fillna(-1).astype("int32")

            # Nullable Int (Int8, Int16, Int32, Int64) → float64
            elif dtype_str in ("Int8", "Int16", "Int32", "Int64"):
                df[col] = df[col].astype("float64")

        # 3) 최종 안전 검증: 남은 비숫자 컬럼 강제 변환
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        categorical_features: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> Any:
        import xgboost as xgb

        self._feature_names = list(X_train.columns)
        self._label_maps = {}

        X_train = self._to_xgb_safe(X_train, fit=True)

        # GPU 사용 가능 여부 확인
        params = dict(self._params)
        if params.get("device") == "cuda":
            try:
                _test = xgb.XGBRegressor(
                    device="cuda", tree_method="hist",
                    n_estimators=1, verbosity=0,
                )
                _test.fit(X_train.head(10), y_train[:10])
            except Exception:
                tqdm.write("      [INFO] XGBoost GPU 불가 — CPU 폴백")
                params["device"] = "cpu"

        n_est = params.get("n_estimators", 5000)
        tqdm_cb = _make_xgb_tqdm_callback(n_est, desc="XGBoost")

        callbacks = [
            tqdm_cb,
            xgb.callback.EarlyStopping(
                rounds=self._config.early_stopping_rounds,
                metric_name="rmse",
                save_best=True,
            ),
        ]

        fit_params: dict[str, Any] = {"verbose": False}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

        if X_val is not None and y_val is not None:
            X_val = self._to_xgb_safe(X_val, fit=False)

            self._model = xgb.XGBRegressor(**params, callbacks=callbacks)
            self._model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                **fit_params,
            )
        else:
            params_no_es = dict(params)
            params_no_es["n_estimators"] = 1000
            self._model = xgb.XGBRegressor(**params_no_es)
            self._model.fit(X_train, y_train, **fit_params)

        tqdm.write(f"      best_iter={self._model.best_iteration}")
        return self._model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        return self._model.predict(self._to_xgb_safe(X, fit=False))

    def get_feature_importance(self) -> pd.DataFrame | None:
        if self._model is None:
            return None
        importance = self._model.feature_importances_
        return pd.DataFrame({
            "feature": self._feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CatBoost
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CatBoostModel(BaseModel):
    """CatBoost 기반 회귀 모델.

    CatBoost의 네이티브 범주형 피처 처리를 활용하며,
    RTX 3090 GPU 가속(task_type=GPU)으로 빠른 학습을 수행합니다.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self._config = config or ModelConfig()
        self._model: Any = None
        self._params = dict(self._config.catboost_params)
        self._feature_names: list[str] = []

    @property
    def name(self) -> str:
        return "CatBoost"

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        categorical_features: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> Any:
        from catboost import CatBoostRegressor, Pool

        self._feature_names = list(X_train.columns)

        cat_features = categorical_features or []
        cat_features = [c for c in cat_features if c in X_train.columns]
        for col in X_train.columns:
            if pd.api.types.is_string_dtype(X_train[col]) and col not in cat_features:
                cat_features.append(col)

        X_train = X_train.copy()
        for col in cat_features:
            X_train[col] = X_train[col].fillna("__MISSING__").astype(str)

        cat_indices = [X_train.columns.get_loc(c) for c in cat_features if c in X_train.columns]

        # GPU 사용 가능 여부 확인
        params = dict(self._params)
        if params.get("task_type") == "GPU":
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi"], capture_output=True, timeout=5,
                )
                if result.returncode != 0:
                    raise RuntimeError("nvidia-smi failed")
            except Exception:
                tqdm.write("      [INFO] CatBoost GPU 불가 — CPU 폴백")
                params["task_type"] = "CPU"
                params.pop("devices", None)

        # CatBoost는 verbose로 자체 진행률을 표시하므로,
        # tqdm 바 대신 verbose=0 + 주기적 로그로 대체
        params["verbose"] = 0  # 내부 로그 비활성화

        n_iter = params.get("iterations", 5000)
        pbar = tqdm(
            total=n_iter,
            desc="      CatBoost",
            bar_format="      {l_bar}{bar:25}{r_bar}",
            ncols=95,
            leave=False,
        )

        train_pool = Pool(
            X_train, y_train,
            cat_features=cat_indices if cat_indices else None,
            weight=sample_weight,
        )

        # CatBoost 반복별 학습 (manual iteration tracking)
        self._model = CatBoostRegressor(**params)

        if X_val is not None and y_val is not None:
            X_val = X_val.copy()
            for col in cat_features:
                if col in X_val.columns:
                    X_val[col] = X_val[col].fillna("__MISSING__").astype(str)

            val_pool = Pool(
                X_val, y_val,
                cat_features=cat_indices if cat_indices else None,
            )

            # CatBoost 학습 (verbose=0 상태에서 학습 후 결과 확인)
            self._model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=self._config.early_stopping_rounds,
                use_best_model=True,
            )
        else:
            params_short = dict(params)
            params_short["iterations"] = 1000
            self._model = CatBoostRegressor(**params_short)
            self._model.fit(train_pool)

        pbar.update(n_iter)
        best_score = self._model.get_best_score()
        val_rmse = ""
        if "validation" in best_score and "RMSE" in best_score["validation"]:
            val_rmse = f" rmse={best_score['validation']['RMSE']:.6f}"
        pbar.set_postfix_str(f"done{val_rmse}")
        pbar.close()
        tqdm.write(f"      best_iter={self._model.best_iteration_}")
        return self._model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("모델이 학습되지 않았습니다.")

        X = X.copy()
        for col in X.columns:
            if pd.api.types.is_string_dtype(X[col]) or X[col].dtype == object:
                X[col] = X[col].fillna("__MISSING__").astype(str)
        return self._model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame | None:
        if self._model is None:
            return None
        importance = self._model.get_feature_importance()
        return pd.DataFrame({
            "feature": self._feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LightGBM Quantile Regression — Exp10
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class LightGBMQuantileModel(LightGBMModel):
    """LightGBM Quantile Regression (Exp10). alpha=0.5 → 중앙값 예측."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__(config)
        alpha = getattr(self._config, "quantile_alpha", 0.5)
        self._params = dict(self._config.lgbm_params)
        self._params["objective"] = "quantile"
        self._params["alpha"] = alpha
        self._params["metric"] = "quantile"

    @property
    def name(self) -> str:
        return "LightGBM_Quantile"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MLP (Neural Network) — Exp10
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MLPModel(BaseModel):
    """sklearn MLPRegressor 기반 신경망 모델 (Exp10).

    GBDT와 이질적인 패턴 학습으로 앙상블 다양성 확보.
    수치형 피처만 사용 (범주형은 미지원).
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self._config = config or ModelConfig()
        self._model: Any = None
        self._feature_names: list[str] = []
        self._scaler: Any = None

    @property
    def name(self) -> str:
        return "MLP"

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        categorical_features: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> Any:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X = X_train[num_cols].fillna(0)
        self._feature_names = num_cols
        y = np.asarray(y_train, dtype=np.float64)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self._config.random_state,
        )
        self._model.fit(X_scaled, y)
        return self._model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        num_cols = [c for c in self._feature_names if c in X.columns]
        X_num = X[num_cols].fillna(0).reindex(columns=self._feature_names).fillna(0)
        X_scaled = self._scaler.transform(X_num)
        return self._model.predict(X_scaled)

    def get_feature_importance(self) -> pd.DataFrame | None:
        if self._model is None or not hasattr(self._model, "coefs_"):
            return None
        coef = self._model.coefs_[0]
        imp = np.abs(coef).sum(axis=1)
        n = min(len(imp), len(self._feature_names))
        return pd.DataFrame({
            "feature": self._feature_names[:n],
            "importance": imp[:n],
        }).sort_values("importance", ascending=False).reset_index(drop=True)
