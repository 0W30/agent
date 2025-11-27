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
COPY requirements.txt ./

# Устанавливаем зависимости через pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Финальный образ
FROM python:3.11-slim

# Устанавливаем только runtime зависимости
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создаём пользователя для безопасности
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /app/logs /app/vector_store /app/cloned_repos /home/appuser/.ssh && \
    chown -R appuser:appuser /app /home/appuser/.ssh && \
    chmod -R 755 /app/logs && \
    chmod -R 755 /app/vector_store && \
    chmod -R 755 /app/cloned_repos && \
    chmod 700 /home/appuser/.ssh

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем установленные пакеты из builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копируем код приложения
COPY --chown=appuser:appuser . .

# Копируем entrypoint скрипт
COPY --chown=appuser:appuser entrypoint.sh /tmp/entrypoint.sh
RUN chmod +x /tmp/entrypoint.sh

# Переключаемся на непривилегированного пользователя
USER appuser

# Создаём known_hosts в /tmp (не будет перезаписан монтированием)
RUN mkdir -p /tmp/ssh && \
    ssh-keyscan -H github.com >> /tmp/ssh/known_hosts 2>/dev/null || true && \
    ssh-keyscan -H gitlab.com >> /tmp/ssh/known_hosts 2>/dev/null || true && \
    chmod 644 /tmp/ssh/known_hosts

# Открываем порт
EXPOSE 8000

# Переменные окружения по умолчанию
ENV LOG_LEVEL=INFO
ENV VECTOR_STORE_PATH=/app/vector_store
ENV LOG_FILE=/app/logs/app.log
# GIT_SSH_COMMAND будет установлен в entrypoint скрипте после генерации ключа

# Запускаем приложение через entrypoint
CMD ["/tmp/entrypoint.sh"]
