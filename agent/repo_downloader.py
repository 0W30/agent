"""Модуль для скачивания Git-репозиториев по SSH."""
import os
from pathlib import Path

from git import Repo, GitCommandError, InvalidGitRepositoryError

from agent.logger_config import get_logger

logger = get_logger(__name__)


def clone_repo(ssh_url: str, branch: str, target_dir: str) -> str:
    """
    Клонирует Git-репозиторий по SSH в указанную директорию.
    Если директория уже существует и является git-репозиторием, выполняет git pull.
    
    Args:
        ssh_url: SSH URL репозитория (например, git@github.com:user/repo.git)
        branch: Ветка для клонирования или обновления
        target_dir: Путь для локального клонирования
        
    Returns:
        str: Абсолютный путь к клонированному репозиторию
        
    Raises:
        GitCommandError: Если произошла ошибка при клонировании или обновлении
        ValueError: Если target_dir существует, но не является git-репозиторием
    """
    logger.info(f"Начало клонирования репозитория: {ssh_url}, ветка: {branch}, целевая директория: {target_dir}")
    target_dir = Path(target_dir).resolve()
    
    # Создаём родительскую директорию, если её нет
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Проверяем, существует ли директория
    if target_dir.exists():
        logger.info(f"Директория {target_dir} уже существует, проверяем наличие git-репозитория")
        try:
            # Пытаемся открыть существующий репозиторий
            repo = Repo(str(target_dir))
            
            # Проверяем, что это валидный git-репозиторий
            if not repo.bare:
                # Получаем текущую ветку или переключаемся на нужную
                try:
                    current_branch = repo.active_branch.name
                    if current_branch != branch:
                        # Переключаемся на нужную ветку
                        repo.git.checkout(branch)
                except (AttributeError, TypeError):
                    # Если нет активной ветки (detached HEAD), переключаемся на нужную
                    repo.git.checkout(branch)
                
                # Выполняем git pull
                logger.info(f"Выполняем git pull для обновления репозитория")
                repo.git.pull()
                logger.info(f"Репозиторий успешно обновлён: {target_dir}")
                return str(target_dir)
            else:
                raise ValueError(f"Директория {target_dir} существует, но является bare репозиторием")
                
        except InvalidGitRepositoryError:
            raise ValueError(
                f"Директория {target_dir} существует, но не является git-репозиторием. "
                "Удалите её вручную или укажите другой путь."
            )
        except GitCommandError as e:
            raise GitCommandError(f"Ошибка при обновлении репозитория: {e}")
    else:
        # Директория не существует - клонируем репозиторий
        try:
            logger.info(f"Клонируем репозиторий в {target_dir}")
            repo = Repo.clone_from(ssh_url, str(target_dir), branch=branch)
            logger.info(f"Репозиторий успешно клонирован: {target_dir}")
            return str(target_dir)
        except GitCommandError as e:
            logger.error(f"Ошибка при клонировании репозитория: {e}")
            raise GitCommandError(f"Ошибка при клонировании репозитория: {e}")

