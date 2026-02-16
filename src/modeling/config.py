"""
모델 학습 설정 (Configuration)

모든 모델 학습/평가에서 사용하는 설정값을 중앙 관리합니다.
팀원이 설정만 변경하면 전체 학습 파이프라인에 반영됩니다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _resolve_project_root() -> Path:
    """프로젝트 루트 경로를 안정적으로 찾습니다."""
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    candidate = Path(__file__).resolve().parent.parent.parent
    if (candidate / "assets" / "data").exists() or (candidate / "src").exists():
        return candidate

    cwd = Path(os.getcwd())
    if cwd.name == "notebooks":
        return cwd.parent
    return cwd


@dataclass
class ModelConfig:
    """모델 학습 파이프라인의 모든 설정을 담는 데이터 클래스.

    Attributes:
        project_root: 프로젝트 루트 경로
        n_splits: K-Fold 분할 수
        random_state: 재현성을 위한 시드
        use_log_target: log1p 변환된 타겟 사용 여부
        cv_strategy: 교차 검증 전략 ("kfold" | "timeseries")
        use_sample_weight: 시간 기반 샘플 가중치 사용 여부
    """

    # ── 경로 설정 ──
    project_root: Path = field(default_factory=_resolve_project_root)

    @property
    def data_dir(self) -> Path:
        """전처리된 데이터 디렉토리."""
        return self.project_root / "notebooks" / "data"

    @property
    def output_dir(self) -> Path:
        """모델 출력 디렉토리 (모델, 예측, submission)."""
        d = self.project_root / "outputs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── 교차 검증 설정 ──
    n_splits: int = 5
    random_state: int = 42
    cv_strategy: str = "kfold"  # "kfold" | "timeseries"

    # ── 타겟 변환 ──
    use_log_target: bool = True

    # ── 시간 기반 Sample Weight ──
    use_sample_weight: bool = True
    sample_weight_decay: float = 0.05  # 지수 감쇠 계수

    # ── Fold 내 Target Encoding ──
    use_fold_target_encoding: bool = True
    target_encode_cols: list[str] = field(
        default_factory=lambda: [
            "아파트명", "도로명", "번지", "시군구", "구", "동",
            "coord_cluster",
        ]
    )
    target_encode_smoothing: int = 100

    # ── LightGBM 하이퍼파라미터 ──
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    })

    # ── XGBoost 하이퍼파라미터 ──
    xgb_params: dict = field(default_factory=lambda: {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "device": "cuda",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    })

    # ── CatBoost 하이퍼파라미터 ──
    catboost_params: dict = field(default_factory=lambda: {
        "iterations": 5000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "random_strength": 0.5,
        "bagging_temperature": 0.8,
        "random_seed": 42,
        "task_type": "GPU",
        "devices": "0",
        "verbose": 200,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
    })

    # ── Early Stopping ──
    early_stopping_rounds: int = 100

    # ── 범주형 피처 (자동 감지 또는 수동 지정) ──
    categorical_features: list[str] | None = None

    # ── 앙상블 설정 ──
    # 3개 모델 앙상블: 모델 다양성이 높을수록 일반화 성능 향상
    ensemble_models: list[str] = field(
        default_factory=lambda: ["lightgbm", "xgboost", "catboost"]
    )
