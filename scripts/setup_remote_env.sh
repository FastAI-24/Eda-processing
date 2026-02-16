#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# 원격 GPU 서버 conda 환경 구축 스크립트
#
# 서버 스펙:
#   - AMD Threadripper PRO 3975WX 32코어 / 64스레드
#   - RAM 251 GB
#   - NVIDIA RTX 3090 24 GB (CUDA 12.2)
#   - Ubuntu 20.04, Conda 23.9.0
#
# 사용법:
#   로컬에서 원격 실행:
#     ssh -i tyler_server.pem -p 30694 root@10.196.197.31 'bash -s' < scripts/setup_remote_env.sh
#   또는 원격 접속 후 직접 실행:
#     bash scripts/setup_remote_env.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

ENV_NAME="hpp"
PYTHON_VERSION="3.12"

echo "============================================"
echo " 원격 GPU 서버 환경 구축 시작"
echo "============================================"

# ── conda 활성화 ──
source /opt/conda/etc/profile.d/conda.sh

# ── 기존 환경 확인 ──
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] 기존 '${ENV_NAME}' 환경이 존재합니다. 삭제 후 재생성합니다."
    conda deactivate 2>/dev/null || true
    conda env remove -n "${ENV_NAME}" -y
fi

# ── conda 환경 생성 ──
echo ""
echo "[1/4] conda 환경 생성: ${ENV_NAME} (Python ${PYTHON_VERSION})"
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

# ── 환경 활성화 ──
conda activate "${ENV_NAME}"
echo "  Python: $(python --version)"
echo "  pip:    $(pip --version)"

# ── 핵심 ML 패키지 설치 ──
echo ""
echo "[2/4] ML 패키지 설치"
pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    lightgbm \
    xgboost \
    catboost \
    optuna \
    tqdm \
    python-dotenv \
    matplotlib \
    seaborn

# ── GPU 지원 확인 ──
echo ""
echo "[3/4] GPU 지원 확인"

python -c "
import lightgbm as lgb
print(f'  LightGBM {lgb.__version__}')

import xgboost as xgb
print(f'  XGBoost  {xgb.__version__}')

import catboost as cb
print(f'  CatBoost {cb.__version__}')

try:
    import torch
    print(f'  PyTorch  {torch.__version__}  CUDA: {torch.cuda.is_available()}')
except ImportError:
    print('  PyTorch  미설치 (GPU 부스팅에는 불필요)')
"

# ── 프로젝트 디렉토리 확인 ──
echo ""
echo "[4/4] 데이터 디렉토리 확인"
DATA_DIR="/data/ephemeral/home/data"
if [ -d "${DATA_DIR}" ]; then
    echo "  데이터 디렉토리: ${DATA_DIR}"
    ls -lh "${DATA_DIR}"/*.csv 2>/dev/null || echo "  CSV 파일 없음"
else
    echo "  [WARN] 데이터 디렉토리가 존재하지 않습니다: ${DATA_DIR}"
fi

echo ""
echo "============================================"
echo " 환경 구축 완료!"
echo " 사용법: conda activate ${ENV_NAME}"
echo "============================================"
