#!/usr/bin/env python3
"""Точка входа для запуска API сервера."""
from agent.api import app
import uvicorn
import os
from pathlib import Path

# Настраиваем SSH для GitPython при запуске приложения
ssh_dir = Path.home() / ".ssh"
ssh_key = ssh_dir / "id_rsa"
ssh_config = ssh_dir / "config"

if ssh_key.exists():
    if ssh_config.exists():
        os.environ["GIT_SSH_COMMAND"] = f"ssh -F {ssh_config} -o UserKnownHostsFile=/tmp/ssh/known_hosts -o StrictHostKeyChecking=accept-new"
    else:
        os.environ["GIT_SSH_COMMAND"] = f"ssh -i {ssh_key} -o UserKnownHostsFile=/tmp/ssh/known_hosts -o StrictHostKeyChecking=accept-new"

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )

