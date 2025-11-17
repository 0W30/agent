"""Модуль для индексации Python-файлов из репозитория."""
from pathlib import Path

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from agent.logger_config import get_logger

logger = get_logger(__name__)


# Директории для игнорирования
IGNORE_DIRS = {
    ".git",
    "node_modules",
    "venv",
    ".idea",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "dist",
    "*.egg-info"
}


def should_ignore_path(path: Path) -> bool:
    """
    Проверяет, нужно ли игнорировать путь.
    
    Args:
        path: Путь для проверки
        
    Returns:
        bool: True, если путь нужно игнорировать
    """
    parts = path.parts
    for part in parts:
        if part in IGNORE_DIRS:
            return True
    return False


def extract_python_files(repo_path: str) -> list[Document]:
    """
    Рекурсивно обходит репозиторий и извлекает все Python-файлы.
    
    Args:
        repo_path: Путь к корню репозитория
        
    Returns:
        list[Document]: Список документов LangChain с содержимым файлов
    """
    logger.info(f"Начало индексации Python-файлов из {repo_path}")
    repo_path = Path(repo_path)
    documents = []
    
    if not repo_path.exists():
        logger.error(f"Путь {repo_path} не существует")
        raise ValueError(f"Путь {repo_path} не существует")
    
    if not repo_path.is_dir():
        logger.error(f"Путь {repo_path} не является директорией")
        raise ValueError(f"Путь {repo_path} не является директорией")
    
    # Рекурсивно обходим все файлы
    total_files = 0
    ignored_files = 0
    for file_path in repo_path.rglob("*.py"):
        total_files += 1
        # Проверяем, нужно ли игнорировать файл
        if should_ignore_path(file_path):
            ignored_files += 1
            logger.debug(f"Игнорируем файл: {file_path}")
            continue
        
        try:
            # Читаем содержимое файла
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            # Создаём документ LangChain с именем файла в metadata
            filename = file_path.name
            doc = Document(
                page_content=content,
                metadata={"path": filename}
            )
            documents.append(doc)
            logger.debug(f"Проиндексирован файл: {file_path.name}")
        except Exception as e:
            logger.warning(f"Ошибка при чтении файла {file_path}: {e}")
            continue
    
    logger.info(f"Индексация завершена: найдено {total_files} Python-файлов, "
                f"проиндексировано {len(documents)}, проигнорировано {ignored_files}")
    return documents
