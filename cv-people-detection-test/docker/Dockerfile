FROM python:3.11-slim

# Метаданные
LABEL maintainer="crowd-detection@example.com"
LABEL description="Crowd Detection - People detection on video using YOLOv8"

# Установка системных зависимостей для OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копирование requirements
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY src/ ./src/
COPY app/ ./app/
COPY run.py .

# Создание директорий для данных
RUN mkdir -p data/input data/output data/reports models logs

# Порт Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501', timeout=5)"

# Запуск приложения
CMD ["python", "run.py"]
