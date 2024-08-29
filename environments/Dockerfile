# CUDA 11.0がインストールされたPython 3.6.2ベースのイメージを使用
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

# 環境変数でタイムゾーン設定を非対話モードに設定
ENV DEBIAN_FRONTEND=noninteractive

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    python3.7 \
    python3.7-dev \
    python3-pip \
    python3.7-venv \
    wget \
    nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# pipをアップグレードし、Poetryをインストール
RUN python3.7 -m pip install --upgrade pip && \
    python3.7 -m pip install poetry==1.1.15

# pyproject.tomlとpoetry.lockをコピー
COPY ./environments/pyproject.toml ./
# COPY ./environments/poetry.lock ./

# Poetryで依存関係をインストール
RUN poetry install --no-root
