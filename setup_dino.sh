#!/usr/bin/env bash
set -euo pipefail

# — 환경 설정 —
ENV_NAME="dino"
PYTHON_VERSION="3.10"

# — Miniconda 설치 경로 —
CONDA_BASE="${HOME}/miniconda3"
CONDA_BIN="${CONDA_BASE}/bin/conda"

# 1. conda 실행 파일 확인
if [ ! -x "${CONDA_BIN}" ]; then
  echo "Error: ${CONDA_BIN} not found. Miniconda/Anaconda가 설치되지 않았거나 경로가 다릅니다."
  exit 1
fi

# 2. conda 초기화 (bash 전용)
eval "$( ${CONDA_BIN} shell.bash hook )"

# 3. 환경 생성 (이미 존재하면 건너뜀)
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' already exists. Skipping creation."
else
  conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# 4. 환경 활성화
conda activate "${ENV_NAME}"

# 5. pip 최신화
pip install --upgrade pip

# 6. 로컬 requirements.txt 확인 및 설치
if [ ! -f "requirements.txt" ]; then
  echo "Error: requirements.txt not found in current directory."
  exit 1
fi

pip install -r requirements.txt

echo "Conda environment '${ENV_NAME}' is set up and dependencies are installed."
