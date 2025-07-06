#!/bin/bash

# 원하는 환경 이름 지정
ENV_NAME=vit_attention_env
PYTHON_VERSION=3.9

# 1. Conda 가상환경 생성
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# 2. Conda 환경 활성화
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 3. pip 최신화 (가상환경 내)
python -m pip install --upgrade pip

# 4. requirements.txt에 따라 패키지 설치
pip install -r requirements.txt

# 5. 코드 실행 예시 (적절히 수정)
# --pretrained_weights, --image_path 등 인자 필요 시 아래에 추가
# 예: python your_script.py --pretrained_weights path/to/weights.pth --image_path path/to/image.jpg

echo "패키지 설치 및 환경 준비 완료."
echo "다음과 같이 코드를 실행하세요:"
echo "conda activate $ENV_NAME"
echo "python visualize_attention.py --pretrained_weights path/to/weights.pth --image_path path/to/image.jpg"
