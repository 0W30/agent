"""Модуль для скачивания Git-репозиториев."""
import os
from pathlib import Path

from git import Repo, GitCommandError, InvalidGitRepositoryError

from agent.logger_config import get_logger

logger = get_logger(__name__)


def clone_repo(url: str, branch: str, target_dir: str) -> str:
    """
    Клонирует Git-репозиторий по URL (SSH или HTTPS) в указанную директорию.
    Если директория уже существует и является git-репозиторием, выполняет git pull.
    
    Args:
        url: URL репозитория (SSH или HTTPS)
        branch: Ветка для клонирования или обновления
        target_dir: Путь для локального клонирования
        
    Returns:
        str: Абсолютный путь к клонированному репозиторию
        
    Raises:
        GitCommandError: Если произошла ошибка при клонировании или обновлении
        ValueError: Если target_dir существует, но не является git-репозиторием
    """
    logger.info(f"Клонирование репозитория: {url}, ветка: {branch}, директория: {target_dir}")
    target_dir = Path(target_dir).resolve()
    target_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    
    try:
        os.chmod(target_dir.parent, 0o755)
    except (OSError, PermissionError):
        pass
    
    if target_dir.exists():
        logger.info(f"Директория существует, проверяем git-репозиторий: {target_dir}")
        try:
            repo = Repo(str(target_dir))
            
            if not repo.bare:
                # Проверяем существование ветки перед переключением
                try:
                    branches = [ref.name for ref in repo.refs if ref.name.startswith('origin/')]
                    local_branches = [ref.name for ref in repo.heads]
                    
                    # Проверяем, существует ли ветка локально или в origin
                    branch_exists = branch in local_branches or f"origin/{branch}" in branches
                    
                    if branch_exists:
                        try:
                            current_branch = repo.active_branch.name
                            if current_branch != branch:
                                logger.info(f"Переключение на ветку {branch}")
                                repo.git.checkout(branch)
                        except (AttributeError, TypeError):
                            logger.info(f"Переключение на ветку {branch}")
                            repo.git.checkout(branch)
                    else:
                        logger.warning(
                            f"Ветка {branch} не найдена в репозитории. "
                            f"Доступные ветки: {', '.join(local_branches) if local_branches else 'нет локальных веток'}. "
                            f"Продолжаем с текущей веткой."
                        )
                except Exception as e:
                    logger.warning(f"Не удалось проверить ветки: {e}. Продолжаем обновление.")
                
                # Обновляем репозиторий: сначала fetch, потом pull
                try:
                    logger.info("Получение изменений через git fetch")
                    repo.git.fetch()
                    
                    logger.info("Обновление репозитория через git pull")
                    repo.git.pull()
                    logger.info(f"Репозиторий обновлён: {target_dir}")
                except GitCommandError as pull_error:
                    error_msg = str(pull_error)
                    if "cannot lock ref" in error_msg or "reference already exists" in error_msg:
                        logger.error(f"Ошибка блокировки при обновлении: {error_msg}")
                        raise GitCommandError(
                            f"Ошибка блокировки репозитория. Репозиторий может быть в некорректном состоянии. "
                            f"Попробуйте удалить директорию {target_dir} и клонировать заново, или удалите директорию вручную."
                        )
                    else:
                        raise
                
                return str(target_dir)
            else:
                raise ValueError(f"Директория {target_dir} является bare репозиторием")
                
        except InvalidGitRepositoryError:
            raise ValueError(
                f"Директория {target_dir} существует, но не является git-репозиторием. "
                "Удалите её вручную или укажите другой путь."
            )
        except GitCommandError as e:
            error_msg = str(e)
            if "pathspec" in error_msg and "did not match" in error_msg:
                raise GitCommandError(
                    f"Ветка '{branch}' не найдена в репозитории. "
                    f"Проверьте правильность имени ветки или используйте существующую ветку."
                )
            elif "cannot lock ref" in error_msg or "reference already exists" in error_msg:
                raise GitCommandError(
                    f"Ошибка блокировки репозитория. Репозиторий может быть в некорректном состоянии. "
                    f"Попробуйте удалить директорию {target_dir} и клонировать заново."
                )
            raise GitCommandError(f"Ошибка при обновлении репозитория: {error_msg}")
    else:
        try:
            logger.info(f"Клонирование репозитория в {target_dir}")
            repo = Repo.clone_from(url, str(target_dir), branch=branch)
            logger.info(f"Репозиторий клонирован: {target_dir}")
            return str(target_dir)
        except GitCommandError as e:
            error_msg = str(e)
            logger.error(f"Ошибка при клонировании: {error_msg}")
            
            if "500" in error_msg or "Internal Server Error" in error_msg:
                raise GitCommandError(
                    f"GitHub вернул ошибку 500. Это может быть временная проблема. "
                    f"Попробуйте позже или проверьте доступность репозитория {url}."
                )
            elif "404" in error_msg or "not found" in error_msg.lower():
                raise GitCommandError(f"Репозиторий {url} не найден. Проверьте правильность URL.")
            elif "403" in error_msg or "forbidden" in error_msg.lower():
                raise GitCommandError(
                    f"Доступ к репозиторию {url} запрещён. "
                    f"Убедитесь, что репозиторий публичный или у вас есть права доступа."
                )
            elif "Permission denied" in error_msg:
                raise GitCommandError(
                    f"Ошибка прав доступа при клонировании в {target_dir}. Проверьте права на директорию."
                )
            else:
                raise GitCommandError(f"Ошибка при клонировании репозитория: {error_msg}")
