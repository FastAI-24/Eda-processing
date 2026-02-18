"""
Optuna 기반 하이퍼파라미터 자동 탐색 (Tuner)

LightGBM, XGBoost, CatBoost 각각에 대해 Optuna trial을 정의하고,
최적 하이퍼파라미터를 탐색합니다.

원격 서버(64코어, RTX 3090) 활용:
    - n_jobs=-1로 병렬 학습
    - GPU 가속 자동 감지

사용 예:
    from modeling.tuner import HyperparameterTuner
    from modeling.config import ModelConfig

    config = ModelConfig(n_splits=3)
    tuner = HyperparameterTuner(config)
    best_params = tuner.tune("lightgbm", X_train, y_train, n_trials=100)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, TimeSeriesSplit

from .config import ModelConfig
from .trainer import compute_fold_target_encoding, compute_sample_weight


class HyperparameterTuner:
    """Optuna 기반 하이퍼파라미터 탐색기."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self._config = config or ModelConfig()

    def _get_cv_splitter(self, X: pd.DataFrame):
        cfg = self._config
        if cfg.cv_strategy == "timeseries":
            return TimeSeriesSplit(n_splits=cfg.n_splits)
        return KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)

    def _detect_cat_features(self, X: pd.DataFrame) -> list[str]:
        cat_cols = []
        for col in X.columns:
            if (
                X[col].dtype == object
                or pd.api.types.is_string_dtype(X[col])
            ):
                cat_cols.append(col)
        return cat_cols

    def _objective_lightgbm(self, trial, X, y_transformed, cat_features):
        import lightgbm as lgb

        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 5000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),  # 512→127 과적합 방지
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "subsample_freq": trial.suggest_int("subsample_freq", 0, 5),
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
            "feature_fraction_bynode": trial.suggest_float("feature_fraction_bynode", 0.5, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        cv = self._get_cv_splitter(X)
        fold_scores = []

        for train_idx, val_idx in cv.split(X):
            X_tr = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()
            y_tr = y_transformed[train_idx]
            y_val = y_transformed[val_idx]

            # Fold 내 TE
            if self._config.use_fold_target_encoding:
                X_tr, X_val, _ = compute_fold_target_encoding(
                    X_tr, y_tr, X_val, None,
                    self._config.target_encode_cols,
                    self._config.target_encode_smoothing,
                )

            for col in cat_features:
                if col in X_tr.columns:
                    X_tr[col] = X_tr[col].fillna("__MISSING__").astype("category")
                if col in X_val.columns:
                    X_val[col] = X_val[col].fillna("__MISSING__").astype("category")

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
                categorical_feature=[c for c in cat_features if c in X_tr.columns] or "auto",
            )
            pred = model.predict(X_val)
            fold_scores.append(np.sqrt(mean_squared_error(y_val, pred)))

        return np.mean(fold_scores)

    def _objective_xgboost(self, trial, X, y_transformed, cat_features):
        import xgboost as xgb

        params = {
            "objective": "reg:squarederror",
            "n_estimators": 5000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

        cv = self._get_cv_splitter(X)
        fold_scores = []

        for train_idx, val_idx in cv.split(X):
            X_tr = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()
            y_tr = y_transformed[train_idx]
            y_val = y_transformed[val_idx]

            if self._config.use_fold_target_encoding:
                X_tr, X_val, _ = compute_fold_target_encoding(
                    X_tr, y_tr, X_val, None,
                    self._config.target_encode_cols,
                    self._config.target_encode_smoothing,
                )

            for col in cat_features:
                if col in X_tr.columns:
                    X_tr[col] = X_tr[col].fillna("__MISSING__").astype("category")
                if col in X_val.columns:
                    X_val[col] = X_val[col].fillna("__MISSING__").astype("category")

            model = xgb.XGBRegressor(
                **params,
                enable_categorical=True,
                callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            pred = model.predict(X_val)
            fold_scores.append(np.sqrt(mean_squared_error(y_val, pred)))

        return np.mean(fold_scores)

    def _objective_catboost(self, trial, X, y_transformed, cat_features):
        from catboost import CatBoostRegressor, Pool

        params = {
            "iterations": 5000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "random_seed": 42,
            "verbose": 0,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
        }

        # GPU 사용 시도
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            if result.returncode == 0:
                params["task_type"] = "GPU"
                params["devices"] = "0"
                # GPU 모드에서 border_count > 254 시 OOM 가능
                if params["border_count"] > 254:
                    params["border_count"] = 254
        except Exception:
            pass

        cat_indices = [X.columns.get_loc(c) for c in cat_features if c in X.columns]

        cv = self._get_cv_splitter(X)
        fold_scores = []

        for train_idx, val_idx in cv.split(X):
            X_tr = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()
            y_tr = y_transformed[train_idx]
            y_val = y_transformed[val_idx]

            if self._config.use_fold_target_encoding:
                X_tr, X_val, _ = compute_fold_target_encoding(
                    X_tr, y_tr, X_val, None,
                    self._config.target_encode_cols,
                    self._config.target_encode_smoothing,
                )
                # TE 후 cat_indices 재계산
                cat_indices_fold = [
                    X_tr.columns.get_loc(c)
                    for c in cat_features if c in X_tr.columns
                ]
            else:
                cat_indices_fold = cat_indices

            for col in cat_features:
                if col in X_tr.columns:
                    X_tr[col] = X_tr[col].fillna("__MISSING__").astype(str)
                if col in X_val.columns:
                    X_val[col] = X_val[col].fillna("__MISSING__").astype(str)

            train_pool = Pool(X_tr, y_tr, cat_features=cat_indices_fold or None)
            val_pool = Pool(X_val, y_val, cat_features=cat_indices_fold or None)

            model = CatBoostRegressor(**params)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
            pred = model.predict(X_val)
            fold_scores.append(np.sqrt(mean_squared_error(y_val, pred)))

        return np.mean(fold_scores)

    def tune(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        n_trials: int = 100,
    ) -> dict:
        """Optuna로 하이퍼파라미터를 탐색합니다.

        Args:
            model_name: "lightgbm" | "xgboost" | "catboost"
            X: 학습 피처
            y: 학습 타겟 (원본 스케일)
            n_trials: 탐색 횟수

        Returns:
            best_params: 최적 하이퍼파라미터 딕셔너리 (cv_rmse 포함)
        """
        import optuna
        from optuna.samplers import TPESampler

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        cfg = self._config

        if cfg.use_log_target:
            y_transformed = np.log1p(y)
        else:
            y_transformed = np.asarray(y, dtype=np.float64)

        if isinstance(y_transformed, pd.Series):
            y_transformed = y_transformed.values

        cat_features = self._detect_cat_features(X)

        objective_map = {
            "lightgbm": self._objective_lightgbm,
            "xgboost": self._objective_xgboost,
            "catboost": self._objective_catboost,
        }

        if model_name not in objective_map:
            raise ValueError(f"지원하지 않는 모델: {model_name}. (lightgbm, xgboost, catboost)")

        objective_fn = objective_map[model_name]

        print(f"\n{'='*60}")
        print(f"Optuna 하이퍼파라미터 탐색: {model_name}")
        print(f"  n_trials={n_trials}, n_splits={cfg.n_splits}, cv={cfg.cv_strategy}")
        print(f"  Sampler: TPE (seed={cfg.random_state})")
        print(f"  Pruner: MedianPruner (startup=10)")
        print(f"{'='*60}")

        sampler = TPESampler(
            seed=cfg.random_state,
            multivariate=True,
        )
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=100,
            interval_steps=10,
        )

        start = time.time()
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            study_name=f"{model_name}_tuning",
        )
        study.optimize(
            lambda trial: objective_fn(trial, X, y_transformed, cat_features),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        elapsed = time.time() - start

        print(f"\n탐색 완료 ({elapsed:.1f}초)")
        print(f"  최적 RMSE: {study.best_value:.6f}")
        print(f"  완료/전체: {len(study.trials)}/{n_trials} trials")
        print(f"  최적 파라미터:")
        for key, val in study.best_params.items():
            print(f"    {key}: {val}")

        result = dict(study.best_params)
        result["cv_rmse"] = study.best_value
        return result
