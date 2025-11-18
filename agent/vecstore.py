"""Модуль для создания и управления векторной базой данных FAISS."""
import os
import requests
from pathlib import Path
from typing import Optional, List

from agent.logger_config import get_logger

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
    
    # Логируем первые и последние символы ключа для отладки (безопасно)
    if len(api_key) > 10:
        logger.debug(f"API ключ получен (длина: {len(api_key)}, начало: {api_key[:5]}...)")
    else:
        logger.warning("API ключ слишком короткий, возможно ошибка")
    
    # Получаем модель эмбеддингов
    # OpenRouter требует формат "openai/text-embedding-ada-002" (с префиксом)
    default_model = "openai/text-embedding-ada-002"
    model = os.getenv("OPENROUTER_EMBEDDING_MODEL", default_model)
    
    # Если модель указана без префикса, добавляем префикс openai/
    if not model.startswith(("openai/", "anthropic/", "cohere/", "huggingface/")):
        if "embedding" in model.lower() or "ada" in model.lower():
            model = f"openai/{model}"
            logger.info(f"Модель автоматически дополнена префиксом: {model}")
    
    logger.info(f"Создание эмбеддингов через OpenRouter API, модель: {model}")
    
    # OpenRouter использует OpenAI-совместимый API
    # Важно: модель должна быть с префиксом "openai/" для OpenRouter
    # Пробуем разные варианты инициализации для совместимости
    embeddings = None
    last_error = None
    
    try:
        # Вариант 1: Для новых версий LangChain (langchain-openai)
        # Используем model с префиксом openai/
        embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            model=model  # Модель с префиксом openai/text-embedding-ada-002
        )
        logger.debug("Эмбеддинги созданы через openai_api_key параметр")
    except (TypeError, Exception) as e:
        last_error = e
        logger.debug(f"Попытка 1 не удалась: {e}, пробуем другой вариант")
        try:
            # Вариант 2: Альтернативный вариант для новых версий
            embeddings = OpenAIEmbeddings(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                model=model
            )
            logger.debug("Эмбеддинги созданы через api_key параметр")
        except (TypeError, Exception) as e2:
            last_error = e2
            logger.debug(f"Попытка 2 не удалась: {e2}, пробуем старый вариант")
            try:
                # Вариант 3: Для старых версий LangChain
                embeddings = OpenAIEmbeddings(
                    openai_api_key=api_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    model_name=model
                )
                logger.debug("Эмбеддинги созданы через model_name параметр")
            except (TypeError, Exception) as e3:
                last_error = e3
                logger.error(f"Все попытки создания эмбеддингов не удались. Последняя ошибка: {e3}")
                raise ValueError(f"Не удалось создать объект эмбеддингов: {e3}")
    
    if embeddings is None:
        raise ValueError(f"Не удалось создать объект эмбеддингов. Последняя ошибка: {last_error}")
    
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
    try:
        embeddings = create_openrouter_embeddings()
    except Exception as e:
        logger.error(f"Ошибка при создании объекта эмбеддингов: {e}")
        raise
    
    # Создаём векторную базу из документов
    logger.info("Генерация эмбеддингов для документов...")
    try:
        # Пробуем создать векторную базу с обработкой ошибок
        vector_store = FAISS.from_documents(docs, embeddings)
    except ValueError as e:
        error_msg = str(e)
        if "No embedding data received" in error_msg:
            used_model = os.getenv('OPENROUTER_EMBEDDING_MODEL', 'text-embedding-ada-002')
            logger.error(
                f"OpenRouter API не вернул данные эмбеддингов. "
                f"Используемая модель: {used_model}\n"
                "Возможные причины:\n"
                "1. Модель не поддерживается OpenRouter для эмбеддингов\n"
                "2. Проблемы с API ключом или авторизацией\n"
                "3. OpenRouter требует специальные заголовки или формат запроса\n"
                "4. Превышен лимит запросов к API\n"
                "Рекомендация: Проверьте документацию OpenRouter для списка поддерживаемых моделей эмбеддингов"
            )
        raise ValueError(f"Ошибка при создании векторной базы: {error_msg}") from e
    except Exception as e:
        logger.error(f"Неожиданная ошибка при создании векторной базы: {e}", exc_info=True)
        raise
    logger.info("Эмбеддинги успешно созданы")
    
    # Сохраняем векторную базу
    path = Path(path)
    
    # Создаём директорию, если не существует
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True, mode=0o777)
            logger.debug(f"Директория {path} создана")
        except (OSError, PermissionError) as e:
            logger.warning(f"Не удалось создать директорию {path}: {e}, пробуем продолжить")
    
    # Проверяем права доступа на запись и пытаемся исправить
    if path.exists():
        # Пытаемся установить максимальные права для записи
        try:
            os.chmod(path, 0o777)
            logger.debug(f"Права доступа для {path} установлены в 777")
        except (OSError, PermissionError) as e:
            logger.debug(f"Не удалось изменить права для {path}: {e}")
        
        # Проверяем, можем ли мы записать
        if not os.access(path, os.W_OK):
            logger.warning(f"Нет прав на запись в {path} после попытки исправления")
        else:
            logger.debug(f"Права на запись в {path} подтверждены")
    
    # Сохраняем через стандартный метод LangChain (который использует faiss.write_index внутри)
    logger.info(f"Сохранение векторной базы в {path}")
    try:
        vector_store.save_local(str(path))
    except (OSError, PermissionError, RuntimeError) as e:
        error_msg = str(e)
        logger.error(f"Ошибка при сохранении векторной базы в {path}: {error_msg}")
        
        # Если ошибка связана с правами доступа, пытаемся исправить
        if "Permission denied" in error_msg or "PermissionError" in error_msg:
            logger.warning("Обнаружена ошибка прав доступа, проверяем директорию...")
            # Проверяем родительскую директорию
            parent = path.parent
            if parent.exists() and not os.access(parent, os.W_OK):
                logger.error(f"Нет прав на запись в родительскую директорию {parent}")
                raise PermissionError(f"Нет прав на запись в {parent}. Проверьте права доступа к volume.")
            
            # Пытаемся создать директорию заново с максимальными правами
            try:
                if path.exists():
                    # Проверяем, можем ли мы записать в существующую директорию
                    test_file = path / ".test_write"
                    try:
                        test_file.touch()
                        test_file.unlink()
                        logger.debug("Тест записи в директорию успешен")
                    except (OSError, PermissionError):
                        logger.error(f"Не удалось записать тестовый файл в {path}")
                        raise PermissionError(
                            f"Нет прав на запись в {path}. "
                            f"Убедитесь, что volume имеет правильные права доступа. "
                            f"Попробуйте: chmod -R 777 ./vector_store на хосте"
                        )
                else:
                    path.mkdir(parents=True, exist_ok=True, mode=0o777)
                    logger.info(f"Директория {path} пересоздана с правами 777")
            except Exception as e2:
                logger.error(f"Не удалось исправить права доступа: {e2}")
                raise PermissionError(
                    f"Критическая ошибка прав доступа: {e2}. "
                    f"Проверьте права на volume ./vector_store на хосте."
                )
            
            # Повторная попытка сохранения
            logger.info("Повторная попытка сохранения векторной базы...")
            vector_store.save_local(str(path))
        else:
            # Другая ошибка - пробрасываем дальше
            raise
    
    # Также сохраняем индекс напрямую через faiss.write_index для совместимости
    index_path = path / "index.faiss"
    if hasattr(vector_store, 'index'):
        faiss.write_index(vector_store.index, str(index_path))
        logger.debug(f"Индекс также сохранён в {index_path}")
    
    logger.info(f"Векторная база успешно сохранена в {path}")
    
    return vector_store


def add_documents_to_vector_store(docs: list[Document], vector_store: FAISS, path: str = "./vector_store") -> FAISS:
    """
    Добавляет документы к существующей векторной базе данных FAISS.
    
    Args:
        docs: Список документов LangChain для добавления
        vector_store: Существующая векторная база данных FAISS
        path: Путь для сохранения обновлённой векторной базы
        
    Returns:
        FAISS: Обновлённый объект векторной базы данных
    """
    if not docs:
        logger.warning("Список документов для добавления пуст")
        return vector_store
    
    logger.info(f"Добавление {len(docs)} документов к существующей векторной базе")
    
    try:
        # Добавляем документы к существующей базе
        vector_store.add_documents(docs)
        logger.info(f"Документы успешно добавлены к векторной базе")
        
        # Сохраняем обновлённую векторную базу
        path = Path(path)
        if path.exists():
            try:
                os.chmod(path, 0o777)
            except (OSError, PermissionError):
                pass
        
        try:
            vector_store.save_local(str(path))
            logger.info(f"Обновлённая векторная база сохранена в {path}")
        except (OSError, PermissionError, RuntimeError) as e:
            logger.error(f"Ошибка при сохранении обновлённой векторной базы: {e}")
            raise
        
        return vector_store
    except Exception as e:
        logger.error(f"Ошибка при добавлении документов к векторной базе: {e}", exc_info=True)
        raise


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
