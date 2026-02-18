#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# 원격 GPU 서버 실행 자동화 스크립트
#
# 로컬에서 코드 업로드 → 원격 실행 → 결과 다운로드를 자동화합니다.
#
# 사용법:
#   ./scripts/run_remote.sh                    # 전처리 + 학습 (기본)
#   ./scripts/run_remote.sh --preprocess-only  # 전처리만
#   ./scripts/run_remote.sh --train-only       # 학습만 (전처리된 데이터 필요)
#   ./scripts/run_remote.sh --ensemble         # 앙상블 학습
#   ./scripts/run_remote.sh --tune             # Optuna 하이퍼파라미터 튜닝
#   ./scripts/run_remote.sh --setup            # 환경 설치만
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ── 프로젝트 루트 (이 스크립트 기준) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── 원격 서버 접속 정보 ──
SSH_KEY="${HOME}/.ssh/tyler_server.pem"
SSH_HOST="10.196.197.31"
SSH_PORT="30694"
SSH_USER="root"
REMOTE_BASE="/data/ephemeral/home"
REMOTE_PROJECT="${REMOTE_BASE}/house-price-prediction"
CONDA_ENV="hpp"

# ── SSH / SCP 공통 옵션 ──
SSH_OPTS="-i ${SSH_KEY} -p ${SSH_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=10"
SCP_OPTS="-i ${SSH_KEY} -P ${SSH_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=10"
SSH_CMD="ssh ${SSH_OPTS} ${SSH_USER}@${SSH_HOST}"
SCP_CMD="scp ${SCP_OPTS}"

# ── 색상 출력 ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ── 인자 파싱 ──
MODE="full"
case "${1:-}" in
    --preprocess-only) MODE="preprocess" ;;
    --train-only)      MODE="train" ;;
    --ensemble)        MODE="ensemble" ;;
    --tune)            MODE="tune" ;;
    --setup)           MODE="setup" ;;
    "")                MODE="full" ;;
    *)
        log_error "알 수 없는 옵션: $1"
        echo "사용법: $0 [--preprocess-only|--train-only|--ensemble|--tune|--setup]"
        exit 1
        ;;
esac

echo "============================================"
echo " 원격 GPU 서버 실행 (모드: ${MODE})"
echo "============================================"

# ── Step 1: SSH 연결 테스트 ──
log_info "원격 서버 연결 테스트..."
if ! ${SSH_CMD} "echo 'OK'" >/dev/null 2>&1; then
    log_error "원격 서버에 연결할 수 없습니다: ${SSH_USER}@${SSH_HOST}:${SSH_PORT}"
    exit 1
fi
log_info "연결 성공!"

# ── Step 2: 환경 설치만 (--setup) ──
if [ "${MODE}" = "setup" ]; then
    log_info "원격 서버 환경 설치 중..."
    ${SSH_CMD} 'bash -s' < "${SCRIPT_DIR}/setup_remote_env.sh"
    log_info "환경 설치 완료!"
    exit 0
fi

# ── Step 3: 코드 업로드 ──
log_info "코드 업로드 중..."

# 원격 프로젝트 디렉토리 생성
${SSH_CMD} "mkdir -p ${REMOTE_PROJECT}/src/preprocessing ${REMOTE_PROJECT}/src/modeling ${REMOTE_PROJECT}/outputs ${REMOTE_PROJECT}/notebooks/data ${REMOTE_PROJECT}/scripts"

# 소스 코드 업로드
${SCP_CMD} -r "${PROJECT_ROOT}/src/preprocessing/"*.py "${SSH_USER}@${SSH_HOST}:${REMOTE_PROJECT}/src/preprocessing/"
${SCP_CMD} -r "${PROJECT_ROOT}/src/modeling/"*.py "${SSH_USER}@${SSH_HOST}:${REMOTE_PROJECT}/src/modeling/"

# 엔트리포인트 업로드
for f in run.py run_train.py main.py pyproject.toml; do
    if [ -f "${PROJECT_ROOT}/${f}" ]; then
        ${SCP_CMD} "${PROJECT_ROOT}/${f}" "${SSH_USER}@${SSH_HOST}:${REMOTE_PROJECT}/"
    fi
done

# 앙상블/튜닝 엔트리포인트 업로드
for f in run_ensemble.py run_tuning.py; do
    if [ -f "${PROJECT_ROOT}/${f}" ]; then
        ${SCP_CMD} "${PROJECT_ROOT}/${f}" "${SSH_USER}@${SSH_HOST}:${REMOTE_PROJECT}/"
    fi
done

log_info "코드 업로드 완료!"

# ── Step 4: 원격 실행 ──
log_info "원격 실행 시작 (모드: ${MODE})..."

REMOTE_CMD="
    source /opt/conda/etc/profile.d/conda.sh
    conda activate ${CONDA_ENV}
    cd ${REMOTE_PROJECT}
    export PROJECT_ROOT=${REMOTE_PROJECT}
"

case "${MODE}" in
    preprocess)
        REMOTE_CMD+="
    echo '── 전처리 실행 ──'
    python -u run.py --skip-eda 2>&1 | tee outputs/preprocess.log
"
        ;;
    train)
        REMOTE_CMD+="
    echo '── 학습 실행 ──'
    python -u run_train.py --save-submission 2>&1 | tee outputs/train.log
"
        ;;
    ensemble)
        REMOTE_CMD+="
    echo '── 앙상블 실행 (optimized) ──'
    rm -f outputs/optuna_best_params.json
    python -u run_ensemble.py --optimized --no-tuned-params --save-submission 2>&1 | tee outputs/ensemble.log
"
        ;;
    tune)
        REMOTE_CMD+="
    echo '── Optuna 튜닝 실행 ──'
    python -u run_tuning.py 2>&1 | tee outputs/tuning.log
"
        ;;
    full)
        REMOTE_CMD+="
    echo '── 전처리 실행 ──'
    python -u run.py --skip-eda 2>&1 | tee outputs/preprocess.log
    echo ''
    echo '── 학습 실행 ──'
    python -u run_train.py --save-submission 2>&1 | tee outputs/train.log
"
        ;;
esac

${SSH_CMD} "${REMOTE_CMD}"

log_info "원격 실행 완료!"

# ── Step 5: 결과 다운로드 ──
log_info "결과 다운로드 중..."

LOCAL_OUTPUT="${PROJECT_ROOT}/outputs"
mkdir -p "${LOCAL_OUTPUT}"

# submission 및 로그 다운로드
for f in submission.csv feature_importance_lightgbm.csv feature_importance_xgboost.csv feature_importance_catboost.csv preprocess.log train.log ensemble.log tuning.log; do
    ${SCP_CMD} "${SSH_USER}@${SSH_HOST}:${REMOTE_PROJECT}/outputs/${f}" "${LOCAL_OUTPUT}/" 2>/dev/null || true
done

# 전처리된 데이터 다운로드 (선택적)
for f in X_train_preprocessed.csv X_test_preprocessed.csv y_train_preprocessed.csv; do
    if ${SSH_CMD} "test -f ${REMOTE_PROJECT}/notebooks/data/${f}" 2>/dev/null; then
        log_info "전처리 데이터 다운로드: ${f}"
        ${SCP_CMD} "${SSH_USER}@${SSH_HOST}:${REMOTE_PROJECT}/notebooks/data/${f}" "${PROJECT_ROOT}/notebooks/data/" 2>/dev/null || true
    fi
done

log_info "결과 다운로드 완료!"

echo ""
echo "============================================"
echo " 전체 작업 완료!"
echo "============================================"
echo ""
echo "결과 파일:"
ls -lh "${LOCAL_OUTPUT}/"*.csv 2>/dev/null || echo "  CSV 파일 없음"
ls -lh "${LOCAL_OUTPUT}/"*.log 2>/dev/null || echo "  로그 파일 없음"
