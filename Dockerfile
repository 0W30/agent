# Многоэтапная сборка для оптимизации размера образа
FROM python:3.11-slim as builder

# Устанавливаем системные зависимости для сборки
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY pyproject.toml ./

# Устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Финальный образ
FROM python:3.11-slim

# Устанавливаем только runtime зависимости
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создаём пользователя для безопасности
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /app/logs /app/vector_store && \
    chown -R appuser:appuser /app

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем установленные пакеты из builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копируем код приложения
COPY --chown=appuser:appuser . .

# Переключаемся на непривилегированного пользователя
USER appuser

# Открываем порт
EXPOSE 8000

# Переменные окружения по умолчанию
ENV LOG_LEVEL=INFO
ENV VECTOR_STORE_PATH=/app/vector_store
ENV LOG_FILE=/app/logs/app.log

# Команда запуска
CMD ["python", "api.py"]

