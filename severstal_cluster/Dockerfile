# Используем официальный образ Python 3.9
FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование файла зависимостей
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY . .

# Создание непривилегированного пользователя
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Экспозиция порта
EXPOSE 8501

# Команда запуска
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]