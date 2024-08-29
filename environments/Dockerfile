FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.7 \
    python3.7-dev \
    python3-pip \
    python3.7-venv \
    wget \
    nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3.7 -m pip install --upgrade pip && \
    python3.7 -m pip install poetry==1.1.15

COPY ./environments/pyproject.toml ./

RUN poetry install --no-root
