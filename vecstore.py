"""Модуль для создания и управления векторной базой данных FAISS."""
import os
from pathlib import Path
from typing import Optional

from logger_config import get_logger

logger = get_logger(__name__)

# Импорты с поддержкой разных версий LangChain
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    try:
        from langchain.embeddings import OpenAIEmbeddings
    except ImportError:
        from langchain_community.embeddings import OpenAIEmbeddings

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

import faiss


def create_openrouter_embeddings():
    """
    Создаёт объект эмбеддингов для OpenRouter API.
    
    Returns:
        OpenAIEmbeddings: Объект эмбеддингов, настроенный на OpenRouter
    """
    # Получаем API ключ OpenRouter
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY не установлен")
        raise ValueError("Не указан OPENROUTER_API_KEY. Установите переменную окружения.")
    
    # Получаем модель эмбеддингов (по умолчанию text-embedding-ada-002)
    model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "text-embedding-ada-002")
    logger.info(f"Создание эмбеддингов через OpenRouter API, модель: {model}")
    
    # OpenRouter использует OpenAI-совместимый API
    # Настраиваем base_url на OpenRouter
    try:
        # Для новых версий LangChain
        embeddings = OpenAIEmbeddings(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model=model
        )
    except TypeError:
        # Для старых версий LangChain
        embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            model=model
        )
    
    return embeddings


def create_vector_store(docs: list[Document], path: str = "./vector_store") -> FAISS:
    """
    Создаёт векторную базу данных FAISS из списка документов.
    
    Args:
        docs: Список документов LangChain для индексации
        path: Путь для сохранения векторной базы
        
    Returns:
        FAISS: Объект векторной базы данных
    """
    if not docs:
        logger.error("Попытка создать векторную базу из пустого списка документов")
        raise ValueError("Список документов не может быть пустым")
    
    logger.info(f"Начало создания векторной базы из {len(docs)} документов")
    # Создаём эмбеддинги через OpenRouter API
    embeddings = create_openrouter_embeddings()
    
    # Создаём векторную базу из документов
    logger.info("Генерация эмбеддингов для документов...")
    vector_store = FAISS.from_documents(docs, embeddings)
    logger.info("Эмбеддинги успешно созданы")
    
    # Сохраняем векторную базу
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем через стандартный метод LangChain (который использует faiss.write_index внутри)
    logger.info(f"Сохранение векторной базы в {path}")
    vector_store.save_local(str(path))
    
    # Также сохраняем индекс напрямую через faiss.write_index для совместимости
    index_path = path / "index.faiss"
    if hasattr(vector_store, 'index'):
        faiss.write_index(vector_store.index, str(index_path))
        logger.debug(f"Индекс также сохранён в {index_path}")
    
    logger.info(f"Векторная база успешно сохранена в {path}")
    
    return vector_store


def load_vector_store(path: str = "./vector_store") -> FAISS:
    """
    Загружает существующую векторную базу данных FAISS.
    
    Args:
        path: Путь к сохранённой векторной базе
        
    Returns:
        FAISS: Объект векторной базы данных
    """
    path = Path(path)
    
    if not path.exists():
        logger.error(f"Векторная база не найдена в {path}")
        raise ValueError(f"Векторная база не найдена в {path}")
    
    logger.info(f"Загрузка векторной базы из {path}")
    # Создаём эмбеддинги через OpenRouter API (должны совпадать с теми, что использовались при создании)
    embeddings = create_openrouter_embeddings()
    
    # Загружаем векторную базу через стандартный метод LangChain
    logger.info("Загрузка индекса FAISS...")
    vector_store = FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    logger.info("Векторная база успешно загружена")
    
    return vector_store
