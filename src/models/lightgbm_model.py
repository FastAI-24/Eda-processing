"""LightGBM 모델 학습, 예측, 튜닝"""
import platform
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
import warnings

import lightgbm as lgb
import optuna
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from src.config import (
    RANDOM_SEED,
    N_FOLDS,
    LGBM_DEFAULT_PARAMS,
    OPTUNA_SEARCH_SPACE,
    OPTUNA_N_TRIALS,
    MODELS_DIR,
    MULTI_SEED_LIST,
)


def _detect_gpu_config() -> Dict:
    """플랫폼에 따른 GPU 설정 자동 탐지"""
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin" and machine == "arm64":
        # Apple Silicon (M1/M2/M3) → OpenCL GPU
        return {
            "device": "gpu",
            "gpu_use_dp": False,  # Apple GPU는 FP32 사용
        }
    elif system == "Linux":
        # Linux (NVIDIA CUDA) → GPU
        return {
            "device": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
        }
    else:
        return {"device": "gpu"}


class LightGBMModel:
    """LightGBM 회귀 모델 래퍼 클래스"""

    def __init__(self, use_gpu: bool = False, **kwargs):
        self.use_gpu = use_gpu
        self.gpu_available = False
        self.model: Optional[lgb.LGBMRegressor] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.cat_feature_names: List[str] = []
        self.feature_names: List[str] = []

        params = LGBM_DEFAULT_PARAMS.copy()
        params.update(kwargs)

        if use_gpu:
            gpu_config = _detect_gpu_config()
            params.update(gpu_config)
            system = platform.system()
            machine = platform.machine()
            if system == "Darwin" and machine == "arm64":
                tqdm.write("  GPU 설정: Apple Silicon (OpenCL)")
            elif system == "Linux":
                tqdm.write("  GPU 설정: NVIDIA CUDA (OpenCL)")
            else:
                tqdm.write(f"  GPU 설정: {system}/{machine}")

        self.params = params

    # ── 범주형 인코딩 ──

    def _encode_categoricals(
        self, df: pd.DataFrame, cat_features: List[str], fit: bool = False
    ) -> pd.DataFrame:
        df = df.copy()
        for col in cat_features:
            if col not in df.columns:
                continue
            df[col] = df[col].astype(str)
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is None:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                else:
                    # 딕셔너리 매핑 (벡터화) - 처음 보는 값은 -1
                    mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
                    df[col] = df[col].map(mapping).fillna(-1).astype(int)
        return df

    # ── 학습 / 예측 / 저장 / 로드 ──

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cat_features: Optional[List[str]] = None,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "LightGBMModel":
        if cat_features is None:
            cat_features = []
        self.cat_feature_names = [c for c in cat_features if c in X.columns]
        self.feature_names = list(X.columns)

        X_enc = self._encode_categoricals(X, self.cat_feature_names, fit=True)

        eval_data = None
        callbacks = [lgb.log_evaluation(period=100)]
        if eval_set is not None:
            X_eval, y_eval = eval_set
            X_eval_enc = self._encode_categoricals(X_eval, self.cat_feature_names, fit=False)
            eval_data = [(X_eval_enc, y_eval)]
            callbacks.insert(0, lgb.early_stopping(stopping_rounds=200, verbose=True))

        self.model = lgb.LGBMRegressor(**self.params)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(
                    X_enc, y,
                    eval_set=eval_data,
                    callbacks=callbacks,
                    sample_weight=sample_weight,
                )
            if self.use_gpu:
                self.gpu_available = True
        except lgb.basic.LightGBMError as e:
            if self.use_gpu and "gpu" in str(e).lower():
                tqdm.write(f"  ⚠ GPU 사용 실패: {e}")
                tqdm.write("  → CPU 모드로 전환합니다.")
                cpu_params = {
                    k: v for k, v in self.params.items()
                    if k not in ("device", "gpu_platform_id", "gpu_device_id", "gpu_use_dp")
                }
                self.params = cpu_params
                self.model = lgb.LGBMRegressor(**self.params)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.model.fit(
                        X_enc, y,
                        eval_set=eval_data,
                        callbacks=callbacks,
                        sample_weight=sample_weight,
                    )
                self.gpu_available = False
            else:
                raise
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")
        X_enc = self._encode_categoricals(X, self.cat_feature_names, fit=False)
        return self.model.predict(X_enc)

    def save(self, filepath: Path):
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(filepath))
        meta = {
            "label_encoders": self.label_encoders,
            "cat_feature_names": self.cat_feature_names,
            "feature_names": self.feature_names,
            "use_gpu": self.use_gpu,
            "params": self.params,
        }
        with open(filepath.with_suffix(".pkl"), "wb") as f:
            pickle.dump(meta, f)

    def load(self, filepath: Path):
        filepath = Path(filepath)
        booster = lgb.Booster(model_file=str(filepath))
        self.model = lgb.LGBMRegressor()
        self.model._Booster = booster
        self.model.fitted_ = True
        meta_path = filepath.with_suffix(".pkl")
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.label_encoders = meta.get("label_encoders", {})
            self.cat_feature_names = meta.get("cat_feature_names", [])
            self.feature_names = meta.get("feature_names", [])
            self.use_gpu = meta.get("use_gpu", False)


# ════════════════════════════════════════════════
# 교차 검증
# ════════════════════════════════════════════════


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: List[str],
    use_gpu: bool = False,
    n_folds: int = N_FOLDS,
    **model_params,
) -> Dict:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    cv_scores: List[float] = []

    fold_bar = tqdm(
        enumerate(kf.split(X), 1),
        total=n_folds,
        desc="K-Fold CV (LightGBM)",
        unit="fold",
        leave=True,
        colour="cyan",
    )

    for fold, (train_idx, val_idx) in fold_bar:
        fold_start = time.time()

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LightGBMModel(use_gpu=use_gpu, **model_params)
        model.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_val, y_val))

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        cv_scores.append(rmse)

        elapsed = time.time() - fold_start
        fold_bar.set_postfix({
            "RMSE": f"{rmse:.4f}",
            "평균": f"{np.mean(cv_scores):.4f}",
            "소요": f"{elapsed:.0f}s",
        })

    fold_bar.close()

    mean_rmse = np.mean(cv_scores)
    std_rmse = np.std(cv_scores)

    print(f"\n{'='*50}")
    print("교차 검증 결과")
    print(f"{'='*50}")
    print(f"평균 RMSE: {mean_rmse:.4f} (+/- {std_rmse:.4f})")
    print(f"각 Fold RMSE: {[f'{s:.4f}' for s in cv_scores]}")

    return {"mean_rmse": mean_rmse, "std_rmse": std_rmse, "fold_scores": cv_scores}


# ════════════════════════════════════════════════
# K-Fold OOF (Out-of-Fold) 앙상블 예측
# ════════════════════════════════════════════════


def cross_validate_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    cat_features: List[str],
    use_gpu: bool = False,
    n_folds: int = N_FOLDS,
    sample_weight: Optional[np.ndarray] = None,
    **model_params,
) -> Tuple[Dict, np.ndarray]:
    """K-Fold 교차 검증 + OOF 앙상블 예측

    K개 fold로 나누어 각각 학습하고, test 예측을 평균내어 반환합니다.
    단일 모델보다 안정적인 예측을 제공합니다.

    Args:
        sample_weight: 학습 데이터 샘플별 가중치 (시간 기반 등)

    Returns:
        cv_results: 교차 검증 결과 (mean_rmse, std_rmse, fold_scores, oof_predictions)
        test_preds: K개 모델의 test 예측 평균
    """
    # KFold 분할은 항상 동일한 시드 사용 (다중 시드 앙상블 시 동일 분할 보장)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    cv_scores: List[float] = []
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    seed_label = model_params.get("random_state", RANDOM_SEED)
    fold_bar = tqdm(
        enumerate(kf.split(X_train), 1),
        total=n_folds,
        desc=f"K-Fold OOF (seed={seed_label})",
        unit="fold",
        leave=True,
        colour="cyan",
    )

    for fold, (train_idx, val_idx) in fold_bar:
        fold_start = time.time()

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        w_tr = sample_weight[train_idx] if sample_weight is not None else None

        model = LightGBMModel(use_gpu=use_gpu, **model_params)
        model.fit(
            X_tr, y_tr,
            cat_features=cat_features,
            eval_set=(X_val, y_val),
            sample_weight=w_tr,
        )

        # OOF validation predictions
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds

        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        cv_scores.append(rmse)

        # Test predictions: 각 fold 모델의 예측을 평균
        test_preds += model.predict(X_test) / n_folds

        elapsed = time.time() - fold_start
        fold_bar.set_postfix({
            "RMSE": f"{rmse:.4f}",
            "평균": f"{np.mean(cv_scores):.4f}",
            "소요": f"{elapsed:.0f}s",
        })

    fold_bar.close()

    mean_rmse = np.mean(cv_scores)
    std_rmse = np.std(cv_scores)

    # OOF 전체 RMSE
    oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))

    print(f"\n{'='*50}")
    print(f"K-Fold OOF 예측 결과 (seed={seed_label})")
    print(f"{'='*50}")
    print(f"평균 Fold RMSE: {mean_rmse:.4f} (+/- {std_rmse:.4f})")
    print(f"전체 OOF RMSE: {oof_rmse:.4f}")
    print(f"각 Fold RMSE: {[f'{s:.4f}' for s in cv_scores]}")

    cv_results = {
        "mean_rmse": mean_rmse,
        "std_rmse": std_rmse,
        "oof_rmse": oof_rmse,
        "fold_scores": cv_scores,
        "oof_predictions": oof_preds,
    }

    return cv_results, test_preds


# ════════════════════════════════════════════════
# 다중 시드 앙상블
# ════════════════════════════════════════════════


def multi_seed_cross_validate_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    cat_features: List[str],
    use_gpu: bool = False,
    n_folds: int = N_FOLDS,
    seeds: Optional[List[int]] = None,
    sample_weight: Optional[np.ndarray] = None,
    **model_params,
) -> Tuple[Dict, np.ndarray]:
    """다중 시드 앙상블: 여러 random_state로 학습 후 예측 평균

    동일한 K-Fold 분할에서 서로 다른 모델 시드를 사용하여
    모델 분산을 줄이고 안정적인 예측을 제공합니다.
    """
    if seeds is None:
        seeds = MULTI_SEED_LIST

    all_test_preds: List[np.ndarray] = []
    all_cv_results: List[Dict] = []

    for i, seed in enumerate(seeds):
        tqdm.write(f"\n{'─'*40}")
        tqdm.write(f"  다중 시드 앙상블: Seed {seed} ({i+1}/{len(seeds)})")
        tqdm.write(f"{'─'*40}")

        params = {**model_params, "random_state": seed}
        cv_results, test_preds = cross_validate_and_predict(
            X_train, y_train, X_test, cat_features,
            use_gpu=use_gpu, n_folds=n_folds,
            sample_weight=sample_weight,
            **params,
        )
        all_test_preds.append(test_preds)
        all_cv_results.append(cv_results)

    # 예측 평균
    final_preds = np.mean(all_test_preds, axis=0)

    mean_rmse = np.mean([r["mean_rmse"] for r in all_cv_results])
    mean_oof = np.mean([r["oof_rmse"] for r in all_cv_results])
    std_rmse = np.mean([r["std_rmse"] for r in all_cv_results])

    print(f"\n{'='*50}")
    print(f"다중 시드 앙상블 결과 ({len(seeds)} seeds)")
    print(f"{'='*50}")
    for seed, res in zip(seeds, all_cv_results):
        print(f"  Seed {seed}: OOF={res['oof_rmse']:.4f}, Mean Fold={res['mean_rmse']:.4f}")
    print(f"  앙상블 평균 OOF RMSE: {mean_oof:.4f}")

    combined = {
        "mean_rmse": mean_rmse,
        "std_rmse": std_rmse,
        "oof_rmse": mean_oof,
        "seed_results": all_cv_results,
    }

    return combined, final_preds


# ════════════════════════════════════════════════
# Optuna 하이퍼파라미터 최적화
# ════════════════════════════════════════════════


def optimize_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: List[str],
    use_gpu: bool = False,
    n_trials: int = OPTUNA_N_TRIALS,
    timeout: Optional[int] = None,
) -> Dict:

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    trial_bar = tqdm(total=n_trials, desc="Optuna 튜닝", unit="trial", colour="yellow")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *OPTUNA_SEARCH_SPACE["n_estimators"]),
            "learning_rate": trial.suggest_float("learning_rate", *OPTUNA_SEARCH_SPACE["learning_rate"], log=True),
            "num_leaves": trial.suggest_int("num_leaves", *OPTUNA_SEARCH_SPACE["num_leaves"]),
            "max_depth": trial.suggest_int("max_depth", *OPTUNA_SEARCH_SPACE["max_depth"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *OPTUNA_SEARCH_SPACE["reg_alpha"]),
            "reg_lambda": trial.suggest_float("reg_lambda", *OPTUNA_SEARCH_SPACE["reg_lambda"]),
            "min_child_samples": trial.suggest_int("min_child_samples", *OPTUNA_SEARCH_SPACE["min_child_samples"]),
            "subsample": trial.suggest_float("subsample", *OPTUNA_SEARCH_SPACE["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *OPTUNA_SEARCH_SPACE["colsample_bytree"]),
            "min_child_weight": trial.suggest_float("min_child_weight", *OPTUNA_SEARCH_SPACE["min_child_weight"], log=True),
            "path_smooth": trial.suggest_float("path_smooth", *OPTUNA_SEARCH_SPACE["path_smooth"]),
            "subsample_freq": 1,
            "random_state": RANDOM_SEED,
            "verbose": -1,
        }

        model = LightGBMModel(use_gpu=use_gpu, **params)
        model.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_val, y_val))

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        trial_bar.update(1)
        trial_bar.set_postfix({"RMSE": f"{rmse:.4f}", "best": f"{trial.study.best_value:.4f}" if trial.study.best_value < float("inf") else "N/A"})
        return rmse

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", study_name="lightgbm_optimization")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    trial_bar.close()

    print(f"\n{'='*50}")
    print("최적 하이퍼파라미터:")
    print(f"{'='*50}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"\n최적 RMSE: {study.best_value:.4f}")

    return study.best_params


# ════════════════════════════════════════════════
# 최종 모델 학습
# ════════════════════════════════════════════════


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: List[str],
    use_gpu: bool = False,
    model_params: Optional[Dict] = None,
    save_path: Optional[Path] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> LightGBMModel:
    if model_params is None:
        model_params = {}

    print(f"\n{'='*50}")
    print("최종 모델 학습 시작 (전체 데이터 사용)")
    print(f"{'='*50}")

    model = LightGBMModel(use_gpu=use_gpu, **model_params)
    model.fit(X, y, cat_features=cat_features, sample_weight=sample_weight)

    if save_path:
        model.save(save_path)
        print(f"모델 저장 완료: {save_path}")

    return model
