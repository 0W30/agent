"""Модуль для индексации файлов из репозитория."""
from pathlib import Path
from typing import List

# Импорты с поддержкой разных версий LangChain
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from agent.logger_config import get_logger

logger = get_logger(__name__)

# Параметры для chunking
CHUNK_SIZE = 500  # Количество строк в одном чанке
CHUNK_OVERLAP = 50  # Количество строк перекрытия между чанками


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


def split_into_chunks(content: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[tuple[int, int, str]]:
    """
    Разбивает содержимое файла на чанки с перекрытием.
    
    Args:
        content: Содержимое файла
        chunk_size: Размер чанка в строках
        chunk_overlap: Количество строк перекрытия между чанками
        
    Returns:
        List[tuple[int, int, str]]: Список кортежей (start_line, end_line, chunk_content)
    """
    lines = content.split('\n')
    total_lines = len(lines)
    
    if total_lines <= chunk_size:
        # Файл маленький, возвращаем целиком
        return [(1, total_lines, content)]
    
    chunks = []
    start = 0
    
    while start < total_lines:
        end = min(start + chunk_size, total_lines)
        chunk_lines = lines[start:end]
        chunk_content = '\n'.join(chunk_lines)
        
        chunks.append((start + 1, end, chunk_content))  # +1 потому что строки нумеруются с 1
        
        # Переходим к следующему чанку с учётом перекрытия
        if end >= total_lines:
            break
        start = end - chunk_overlap
    
    return chunks


def extract_python_files(repo_path: str) -> list[Document]:
    """
    Рекурсивно обходит репозиторий и извлекает все файлы.
    
    Args:
        repo_path: Путь к корню репозитория
        
    Returns:
        list[Document]: Список документов LangChain с содержимым файлов
    """
    logger.info(f"Начало индексации файлов из {repo_path}")
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
    skipped_files = 0
    for file_path in repo_path.rglob("*"):
        # Пропускаем директории
        if file_path.is_dir():
            continue
        
        total_files += 1
        # Проверяем, нужно ли игнорировать файл
        if should_ignore_path(file_path):
            ignored_files += 1
            logger.debug(f"Игнорируем файл: {file_path}")
            continue
        
        try:
            # Пытаемся прочитать файл как UTF-8 текст
            try:
                with open(file_path, "r", encoding="utf-8", errors="strict") as f:
                    content = f.read()
            except (UnicodeDecodeError, UnicodeError):
                # Если не удалось прочитать как UTF-8, пропускаем файл
                skipped_files += 1
                logger.debug(f"Пропущен файл (не UTF-8): {file_path}")
                continue
            
            # Создаём документ LangChain с именем файла и путём к проекту в metadata
            filename = file_path.name
            # Сохраняем относительный путь от корня репозитория
            try:
                relative_path = file_path.relative_to(repo_path)
            except ValueError:
                # Если не удалось вычислить относительный путь, используем только имя файла
                relative_path = filename

            # Разбиваем файл на чанки
            chunks = split_into_chunks(content, CHUNK_SIZE, CHUNK_OVERLAP)
            
            # Создаём отдельный документ для каждого чанка
            file_extension = file_path.suffix  # Расширение файла (.py, .js, .md и т.д.)
            for chunk_idx, (start_line, end_line, chunk_content) in enumerate(chunks):
                # Формируем metadata с информацией о чанке
                chunk_metadata = {
                    "path": filename,
                    "file_path": str(relative_path),  # Относительный путь к файлу
                    "file_extension": file_extension,  # Расширение файла
                    "project_path": str(repo_path),  # Полный путь к проекту
                    "chunk_index": chunk_idx,  # Индекс чанка в файле
                    "start_line": start_line,  # Начальная строка чанка
                    "end_line": end_line,  # Конечная строка чанка
                    "total_chunks": len(chunks),  # Общее количество чанков в файле
                }
                
                doc = Document(
                    page_content=chunk_content,
                    metadata=chunk_metadata
                )
                documents.append(doc)
            
            if len(chunks) > 1:
                logger.debug(f"Проиндексирован файл: {file_path.name} (разбит на {len(chunks)} чанков)")
            else:
                logger.debug(f"Проиндексирован файл: {file_path.name}")
        except Exception as e:
            logger.warning(f"Ошибка при чтении файла {file_path}: {e}")
            continue
    
    logger.info(f"Индексация завершена: найдено {total_files} файлов, "
                f"проиндексировано {len(documents)}, проигнорировано {ignored_files}, пропущено {skipped_files}")
    return documents
