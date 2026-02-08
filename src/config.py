"""프로젝트 설정 및 상수 정의"""
import os
import platform
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

# ── 환경 자동 탐지 ──
IS_REMOTE = os.path.exists("/data/ephemeral/home/data")
IS_APPLE_SILICON = (platform.system() == "Darwin" and platform.machine() == "arm64")

# 데이터 디렉토리 경로 (원격 서버 또는 로컬)
if IS_REMOTE:
    DATA_DIR = Path("/data/ephemeral/home/data")
else:
    DATA_DIR = PROJECT_ROOT / "assets" / "data"

# 모델 저장 디렉토리
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# 제출 파일 디렉토리
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_DIR.mkdir(exist_ok=True)

# 랜덤 시드
RANDOM_SEED = 42

# K-Fold CV 설정
N_FOLDS = 5

# 다중 시드 앙상블
MULTI_SEED_LIST = [42, 123, 456]

# 결측 임계값 (80%: k-* 피처 보존)
MISSING_THRESHOLD = 0.80

# 샘플 가중치 (시간 기반)
SAMPLE_WEIGHT_MIN = 0.3   # 가장 오래된 데이터 가중치
SAMPLE_WEIGHT_MAX = 1.0   # 가장 최신 데이터 가중치

# ── LightGBM 기본 하이퍼파라미터 ──
# 원격 서버(고사양): 무거운 파라미터 / 로컬: 경량 파라미터
if IS_REMOTE:
    LGBM_DEFAULT_PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "num_leaves": 255,
        "max_depth": 8,
        "reg_alpha": 0.1,
        "reg_lambda": 3.0,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,        # 배깅 활성화 (subsample < 1.0 시 필수)
        "colsample_bytree": 0.8,
        "min_child_weight": 0.001,
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "n_jobs": -1,
    }
else:
    LGBM_DEFAULT_PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 1000,       # 로컬: 1000 (원격: 5000)
        "learning_rate": 0.1,       # 로컬: 0.1  (원격: 0.05) - 빠른 수렴
        "num_leaves": 127,          # 로컬: 127  (원격: 255)
        "max_depth": 7,             # 로컬: 7    (원격: 8)
        "reg_alpha": 0.1,
        "reg_lambda": 3.0,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,        # 배깅 활성화
        "colsample_bytree": 0.8,
        "min_child_weight": 0.001,
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "n_jobs": -1,
    }

# Optuna 튜닝 범위 (LightGBM)
OPTUNA_SEARCH_SPACE = {
    "n_estimators": (500, 5000) if not IS_REMOTE else (1000, 10000),
    "learning_rate": (0.01, 0.1),
    "num_leaves": (31, 512),
    "max_depth": (4, 12),
    "reg_alpha": (0.0, 10.0),
    "reg_lambda": (0.0, 10.0),
    "min_child_samples": (5, 100),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "min_child_weight": (1e-3, 10.0),
    "path_smooth": (0.0, 10.0),
}

# Optuna 튜닝 설정
OPTUNA_N_TRIALS = 20 if not IS_REMOTE else 50
OPTUNA_TIMEOUT = 3600  # 1시간
