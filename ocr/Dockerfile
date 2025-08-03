# Используем официальный образ с CUDA 11.8 и Python 3.10
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Устанавливаем системные зависимости и Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    python3.11-dev \
    git \
    libgl1 \
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app
RUN git clone https://github.com/Vasilevykh-M/doc_classifier.git
RUN cd doc_classifier && \
    git fetch origin && \
    git checkout origin/ocr && \
    cd ocr && \
    python3.11 -m pip install --no-cache-dir -r requirements.txt
CMD ["tail", "-f", "/dev/null"]