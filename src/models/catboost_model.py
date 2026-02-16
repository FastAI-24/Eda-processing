"""CatBoost 모델 학습, 예측, 튜닝"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from src.config import (
    RANDOM_SEED,
    N_FOLDS,
    CATBOOST_DEFAULT_PARAMS,
    OPTUNA_SEARCH_SPACE,
    OPTUNA_N_TRIALS,
    MODELS_DIR,
)


class CatBoostModel:
    """CatBoost 회귀 모델 래퍼 클래스"""
    
    def __init__(self, use_gpu: bool = False, **kwargs):
        """
        Args:
            use_gpu: GPU 사용 여부
            **kwargs: CatBoost 하이퍼파라미터
        """
        self.use_gpu = use_gpu
        self.model = None
        self.cat_features = None  # 범주형 피처 인덱스
        self.cat_feature_names = None  # 범주형 피처 이름
        
        # 기본 파라미터에 kwargs 병합
        params = CATBOOST_DEFAULT_PARAMS.copy()
        params.update(kwargs)
        
        # GPU 설정
        if use_gpu:
            params["task_type"] = "GPU"
            params["devices"] = "0"  # 첫 번째 GPU 사용
        
        self.params = params
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cat_features: List[str] = None,
        eval_set: Tuple[pd.DataFrame, pd.Series] = None,
    ):
        """모델 학습"""
        # 범주형 피처 인덱스 변환
        if cat_features is None:
            cat_features = []
        
        cat_indices = [X.columns.get_loc(col) for col in cat_features if col in X.columns]
        self.cat_features = cat_indices
        self.cat_feature_names = [col for col in cat_features if col in X.columns]
        
        # Pool 객체 생성
        train_pool = Pool(
            data=X,
            label=y,
            cat_features=cat_indices,
        )
        
        # 검증 세트가 있으면 Pool 생성
        eval_pool = None
        if eval_set is not None:
            X_eval, y_eval = eval_set
            eval_pool = Pool(
                data=X_eval,
                label=y_eval,
                cat_features=cat_indices,
            )
        
        # 모델 학습
        self.model = CatBoostRegressor(**self.params)
        self.model.fit(
            train_pool,
            eval_set=eval_pool,
            verbose=self.params.get("verbose", 100),
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")
        
        test_pool = Pool(
            data=X,
            cat_features=self.cat_features,
        )
        
        return self.model.predict(test_pool)
    
    def save(self, filepath: Path):
        """모델 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(filepath))
        
        # 범주형 피처 정보도 저장
        metadata = {
            "cat_features": self.cat_features,
            "cat_feature_names": self.cat_feature_names,
            "use_gpu": self.use_gpu,
        }
        metadata_path = filepath.with_suffix(".pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
    
    def load(self, filepath: Path):
        """모델 로드"""
        self.model = CatBoostRegressor()
        self.model.load_model(str(filepath))
        
        # 메타데이터 로드
        metadata_path = filepath.with_suffix(".pkl")
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
                self.cat_features = metadata.get("cat_features", [])
                self.cat_feature_names = metadata.get("cat_feature_names", [])
                self.use_gpu = metadata.get("use_gpu", False)


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: List[str],
    use_gpu: bool = False,
    n_folds: int = N_FOLDS,
    **model_params,
) -> Dict[str, float]:
    """K-Fold 교차 검증"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = []
    
    print(f"\n{'='*50}")
    print(f"{n_folds}-Fold 교차 검증 시작")
    print(f"{'='*50}")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}/{n_folds}")
        
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # 모델 학습
        model = CatBoostModel(use_gpu=use_gpu, **model_params)
        model.fit(
            X_train_fold,
            y_train_fold,
            cat_features=cat_features,
            eval_set=(X_val_fold, y_val_fold),
        )
        
        # 검증 세트 예측
        y_pred_fold = model.predict(X_val_fold)
        
        # RMSE 계산 (로그 변환된 값이므로 그대로 계산)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
        cv_scores.append(rmse)
        
        print(f"Fold {fold} RMSE: {rmse:.4f}")
    
    mean_rmse = np.mean(cv_scores)
    std_rmse = np.std(cv_scores)
    
    print(f"\n{'='*50}")
    print(f"교차 검증 결과")
    print(f"{'='*50}")
    print(f"평균 RMSE: {mean_rmse:.4f} (+/- {std_rmse:.4f})")
    print(f"각 Fold RMSE: {[f'{s:.4f}' for s in cv_scores]}")
    
    return {
        "mean_rmse": mean_rmse,
        "std_rmse": std_rmse,
        "fold_scores": cv_scores,
    }


def optimize_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: List[str],
    use_gpu: bool = False,
    n_trials: int = OPTUNA_N_TRIALS,
    timeout: int = None,
) -> Dict:
    """Optuna를 사용한 하이퍼파라미터 최적화"""
    
    def objective(trial):
        # 하이퍼파라미터 샘플링
        params = {
            "iterations": trial.suggest_int("iterations", *OPTUNA_SEARCH_SPACE["iterations"]),
            "learning_rate": trial.suggest_float("learning_rate", *OPTUNA_SEARCH_SPACE["learning_rate"], log=True),
            "depth": trial.suggest_int("depth", *OPTUNA_SEARCH_SPACE["depth"]),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *OPTUNA_SEARCH_SPACE["l2_leaf_reg"]),
            "bagging_temperature": trial.suggest_float("bagging_temperature", *OPTUNA_SEARCH_SPACE["bagging_temperature"]),
            "random_seed": RANDOM_SEED,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "early_stopping_rounds": 200,
            "verbose": False,
        }
        
        if use_gpu:
            params["task_type"] = "GPU"
            params["devices"] = "0"
        
        # 간단한 검증 (전체 CV는 시간이 오래 걸리므로 간단한 분할 사용)
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )
        
        model = CatBoostModel(use_gpu=use_gpu, **params)
        model.fit(
            X_train,
            y_train,
            cat_features=cat_features,
            eval_set=(X_val, y_val),
        )
        
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        return rmse
    
    print(f"\n{'='*50}")
    print(f"Optuna 하이퍼파라미터 최적화 시작 (n_trials={n_trials})")
    print(f"{'='*50}")
    
    study = optuna.create_study(direction="minimize", study_name="catboost_optimization")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    
    print(f"\n{'='*50}")
    print(f"최적 하이퍼파라미터:")
    print(f"{'='*50}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\n최적 RMSE: {study.best_value:.4f}")
    
    return study.best_params


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: List[str],
    use_gpu: bool = False,
    model_params: Dict = None,
    save_path: Path = None,
) -> CatBoostModel:
    """최종 모델 학습 (전체 데이터 사용)"""
    if model_params is None:
        model_params = {}
    
    print(f"\n{'='*50}")
    print("최종 모델 학습 시작 (전체 데이터 사용)")
    print(f"{'='*50}")
    
    model = CatBoostModel(use_gpu=use_gpu, **model_params)
    model.fit(X, y, cat_features=cat_features)
    
    if save_path:
        model.save(save_path)
        print(f"모델 저장 완료: {save_path}")
    
    return model
