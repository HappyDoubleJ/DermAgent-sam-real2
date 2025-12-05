#!/usr/bin/env bash
# DermAgent conda 환경 설정 스크립트
# 사용법: bash setup_conda_env.sh

set -euo pipefail

ENV_NAME="${ENV_NAME:-dermAgent}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  DermAgent Conda 환경 설정${NC}"
echo -e "${BLUE}================================================${NC}"

if ! command -v conda >/dev/null 2>&1; then
  echo -e "${RED}conda를 찾을 수 없습니다. Miniconda/Anaconda를 먼저 설치하세요.${NC}"
  exit 1
fi

if ! [ -f "${REQ_FILE}" ]; then
  echo -e "${RED}requirements.txt를 찾을 수 없습니다: ${REQ_FILE}${NC}"
  exit 1
fi

if conda env list | grep -q "^${ENV_NAME} "; then
  echo -e "${YELLOW}'${ENV_NAME}' 환경이 이미 있습니다. 재생성하려면 삭제 후 진행하세요.${NC}"
  read -p "기존 환경을 삭제하고 새로 만드시겠습니까? (y/N): " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    conda env remove -n "${ENV_NAME}" -y
  else
    echo "종료합니다."
    exit 0
  fi
fi

echo -e "${BLUE}Python ${PYTHON_VERSION} 환경 생성 중...${NC}"
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo -e "${GREEN}✓ 환경 활성화: ${ENV_NAME}${NC}"

pip install --upgrade pip

echo -e "${BLUE}PyTorch 설치 대상 선택${NC}"
echo "1) CUDA 12.1 (추천, 최신 GPU)"
echo "2) CUDA 11.8"
echo "3) CPU only"
read -p "선택 (1-3) [1]: " -r cuda_choice
cuda_choice=${cuda_choice:-1}

case "${cuda_choice}" in
  1) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
  2) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
  3) TORCH_INDEX="https://download.pytorch.org/whl/cpu" ;;
  *) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
esac

echo -e "${BLUE}PyTorch/torchvision 설치 중...${NC}"
pip install torch torchvision --index-url "${TORCH_INDEX}"

echo -e "${BLUE}나머지 의존성 설치 중...${NC}"
tmp_req="$(mktemp)"
grep -v -E '^(torch|torchvision)' "${REQ_FILE}" > "${tmp_req}"
pip install -r "${tmp_req}"
rm -f "${tmp_req}"

echo -e "${GREEN}✓ 설치 완료${NC}"
echo ""
cat <<'INFO'
환경 활성화: conda activate dermAgent
비활성화:   conda deactivate

GPU 확인 예시:
python - <<'PY'
import torch
print('cuda available:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
print('device count:', torch.cuda.device_count())
PY
INFO
