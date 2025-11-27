#!/bin/sh
set -e

# Создаём директорию для SSH ключей контейнера (стандартное место)
SSH_DIR="/home/appuser/.ssh"
mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

# Генерируем SSH ключ если его нет
if [ ! -f "$SSH_DIR/id_rsa" ]; then
    echo "========================================"
    echo "Генерация SSH ключа для контейнера..."
    echo "========================================"
    ssh-keygen -t rsa -b 4096 -f "$SSH_DIR/id_rsa" -N "" -C "container-generated-key-$(hostname)"
    chmod 600 "$SSH_DIR/id_rsa"
    chmod 644 "$SSH_DIR/id_rsa.pub"
    echo ""
    echo "✅ SSH ключ сгенерирован!"
    echo ""
    echo "📋 Публичный ключ (добавьте его в GitHub/GitLab):"
    echo "----------------------------------------"
    cat "$SSH_DIR/id_rsa.pub"
    echo "----------------------------------------"
    echo ""
    echo "💡 Добавьте этот ключ:"
    echo "   - В GitHub: Settings → SSH and GPG keys → New SSH key"
    echo "   - В GitLab: Preferences → SSH Keys"
    echo "   - Или как Deploy Key для конкретного репозитория"
    echo ""
fi

# Создаём SSH config файл для автоматического использования ключа
cat > "$SSH_DIR/config" << EOF
Host github.com
    HostName github.com
    User git
    IdentityFile $SSH_DIR/id_rsa
    StrictHostKeyChecking accept-new
    UserKnownHostsFile /tmp/ssh/known_hosts

Host gitlab.com
    HostName gitlab.com
    User git
    IdentityFile $SSH_DIR/id_rsa
    StrictHostKeyChecking accept-new
    UserKnownHostsFile /tmp/ssh/known_hosts
EOF
chmod 600 "$SSH_DIR/config"

# Создаём директории с правильными правами (для named volumes это не требуется, но на всякий случай)
for dir in /app/cloned_repos /app/vector_store /app/logs; do
    mkdir -p "$dir" 2>/dev/null || true
done

# Настраиваем GIT_SSH_COMMAND для GitPython
export GIT_SSH_COMMAND="ssh -F $SSH_DIR/config -o UserKnownHostsFile=/tmp/ssh/known_hosts -o StrictHostKeyChecking=accept-new"

# Экспортируем переменную для всех дочерних процессов
export SSH_DIR

# Запускаем приложение
exec python main.py

