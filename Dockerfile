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

# Создаём entrypoint скрипт для генерации SSH ключа при запуске
RUN echo '#!/bin/sh\n\
set -e\n\
\n\
# Создаём директорию для SSH ключей контейнера (стандартное место)\n\
SSH_DIR="/home/appuser/.ssh"\n\
mkdir -p "$SSH_DIR"\n\
chmod 700 "$SSH_DIR"\n\
\n\
# Генерируем SSH ключ если его нет\n\
if [ ! -f "$SSH_DIR/id_rsa" ]; then\n\
    echo "========================================"\n\
    echo "Генерация SSH ключа для контейнера..."\n\
    echo "========================================"\n\
    ssh-keygen -t rsa -b 4096 -f "$SSH_DIR/id_rsa" -N "" -C "container-generated-key-$(hostname)"\n\
    chmod 600 "$SSH_DIR/id_rsa"\n\
    chmod 644 "$SSH_DIR/id_rsa.pub"\n\
    echo ""\n\
    echo "✅ SSH ключ сгенерирован!"\n\
    echo ""\n\
    echo "📋 Публичный ключ (добавьте его в GitHub/GitLab):"\n\
    echo "----------------------------------------"\n\
    cat "$SSH_DIR/id_rsa.pub"\n\
    echo "----------------------------------------"\n\
    echo ""\n\
    echo "💡 Добавьте этот ключ:"\n\
    echo "   - В GitHub: Settings → SSH and GPG keys → New SSH key"\n\
    echo "   - В GitLab: Preferences → SSH Keys"\n\
    echo "   - Или как Deploy Key для конкретного репозитория"\n\
    echo ""\n\
fi\n\
\n\
# Создаём SSH config файл для автоматического использования ключа\n\
cat > "$SSH_DIR/config" << EOF\n\
Host github.com\n\
    HostName github.com\n\
    User git\n\
    IdentityFile $SSH_DIR/id_rsa\n\
    StrictHostKeyChecking accept-new\n\
    UserKnownHostsFile /tmp/ssh/known_hosts\n\
\n\
Host gitlab.com\n\
    HostName gitlab.com\n\
    User git\n\
    IdentityFile $SSH_DIR/id_rsa\n\
    StrictHostKeyChecking accept-new\n\
    UserKnownHostsFile /tmp/ssh/known_hosts\n\
EOF\n\
chmod 600 "$SSH_DIR/config"\n\
\n\
# Настраиваем GIT_SSH_COMMAND для GitPython\n\
export GIT_SSH_COMMAND="ssh -F $SSH_DIR/config -o UserKnownHostsFile=/tmp/ssh/known_hosts -o StrictHostKeyChecking=accept-new"\n\
\n\
# Экспортируем переменную для всех дочерних процессов\n\
export SSH_DIR\n\
\n\
# Запускаем приложение\n\
exec python main.py' > /tmp/entrypoint.sh && \
    chmod +x /tmp/entrypoint.sh

# Запускаем приложение через entrypoint
CMD ["/tmp/entrypoint.sh"]
