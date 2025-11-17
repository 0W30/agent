"""Модуль для обработки stack trace и разрешения ошибок."""
import os
import re
from typing import List, Dict, Any
from pathlib import Path

from logger_config import get_logger

logger = get_logger(__name__)

# Импорты с поддержкой разных версий LangChain
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
    except ImportError:
        from langchain_community.chat_models import ChatOpenAI

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS


def parse_stack_trace(trace: str) -> List[Dict[str, Any]]:
    """
    Парсит stack trace и извлекает пути файлов и номера строк.
    
    Args:
        trace: Сырая строка stack trace
        
    Returns:
        List[Dict[str, any]]: Список словарей с ключами 'file' (str) и 'line' (int)
    """
    # Паттерн для поиска путей файлов и номеров строк
    # Примеры: File "/path/to/file.py", line 42
    #          File "/path/to/file.py", line 42, in function_name
    #          File "/path/to/file.py", line 42
    logger.debug("Начало парсинга stack trace")
    pattern = r'File\s+["\']([^"\']+)["\']\s*,\s*line\s+(\d+)'
    
    matches = re.findall(pattern, trace)
    logger.debug(f"Найдено {len(matches)} совпадений в stack trace")
    
    extracted_info = []
    for file_path, line_num in matches:
        # Извлекаем имя файла из пути
        file_path_obj = Path(file_path)
        file_name = file_path_obj.name
        
        extracted_info.append({
            "file": file_name,
            "line": int(line_num)
        })
        logger.debug(f"Извлечён файл: {file_name}, строка: {line_num}")
    
    logger.info(f"Парсинг завершён: извлечено {len(extracted_info)} файлов")
    return extracted_info


def get_relevant_docs(stack_info: List[Dict[str, Any]], vector_store: FAISS) -> List[Document]:
    """
    Получает релевантные документы из векторной базы по каждому файлу из stack trace.
    
    Args:
        stack_info: Результат parse_stack_trace - список словарей с 'file' и 'line'
        vector_store: Векторная база данных FAISS
        
    Returns:
        List[Document]: Объединённый список релевантных документов без дублей
    """
    logger.info(f"Поиск релевантных документов для {len(stack_info)} файлов")
    all_documents = []
    seen_docs = set()  # Для отслеживания дублей по содержимому
    
    for info in stack_info:
        file_name = info["file"]
        logger.debug(f"Поиск документов для файла: {file_name}")
        
        # Ищем top_k=5 документов по similarity_search
        results = vector_store.similarity_search(file_name, k=5)
        logger.debug(f"Найдено {len(results)} документов для {file_name}")
        
        # Добавляем документы, избегая дублей
        for doc in results:
            # Создаём уникальный идентификатор на основе содержимого и пути
            doc_id = (doc.metadata.get("path", ""), doc.page_content[:100])  # Первые 100 символов для идентификации
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                all_documents.append(doc)
    
    logger.info(f"Всего найдено {len(all_documents)} уникальных документов")
    return all_documents


def build_context(docs: List[Document], max_tokens: int = 150000) -> str:
    """
    Склеивает содержимое файлов в один текст с ограничением по токенам.
    
    Args:
        docs: Список документов для объединения
        max_tokens: Максимальное количество токенов (примерно 4 символа = 1 токен)
        
    Returns:
        str: Объединённый контекст
    """
    logger.info(f"Построение контекста из {len(docs)} документов, максимум токенов: {max_tokens}")
    max_chars = max_tokens * 4  # Примерная оценка: 4 символа = 1 токен
    
    combined = []
    current_length = 0
    
    for doc in docs:
        # Формируем строку с содержимым файла
        file_path = doc.metadata.get("path", "unknown")
        content = f"=== Файл: {file_path} ===\n{doc.page_content}\n\n"
        content_length = len(content)
        
        # Проверяем, не превысим ли лимит
        if current_length + content_length > max_chars:
            # Если превышаем, добавляем только часть, которая влезает
            remaining_chars = max_chars - current_length
            if remaining_chars > 100:  # Добавляем только если осталось достаточно места
                truncated_content = doc.page_content[:remaining_chars - 50]  # Оставляем запас
                content = f"=== Файл: {file_path} ===\n{truncated_content}\n\n[файл обрезан]\n\n"
                combined.append(content)
            break
        
        combined.append(content)
        current_length += content_length
    
    result = "\n".join(combined)
    logger.info(f"Контекст построен: {len(result)} символов (~{len(result) // 4} токенов)")
    return result


def resolve_error(trace: str, vector_store: FAISS) -> str:
    """
    Полностью решает задачу разрешения ошибки из stack trace.
    
    Выполняет:
    а) parse_stack_trace - парсинг stack trace
    б) get_relevant_docs - поиск релевантных документов
    в) build_context - построение контекста
    г) LLM.generate - генерация ответа через LLM
    
    Args:
        trace: Сырая строка stack trace
        vector_store: Векторная база данных FAISS
        
    Returns:
        str: Объяснение ошибки и предложение исправления
    """
    logger.info("Начало разрешения ошибки из stack trace")
    # а) Парсим stack trace
    stack_info = parse_stack_trace(trace)
    
    if not stack_info:
        logger.warning("Не удалось извлечь информацию о файлах из stack trace")
        return "Не удалось извлечь информацию о файлах из stack trace."
    
    # б) Получаем релевантные документы
    relevant_docs = get_relevant_docs(stack_info, vector_store)
    
    if not relevant_docs:
        logger.warning("Не найдено релевантных файлов в векторной базе данных")
        return "Не найдено релевантных файлов в векторной базе данных."
    
    # в) Строим контекст
    context = build_context(relevant_docs, max_tokens=150000)
    
    # Формируем промпт
    prompt = f"""Вот ошибка из stack trace:

{trace}

Вот релевантные файлы из кодовой базы:

{context}

Проанализируй ошибку и:
1. Объясни причину ошибки
2. Предложи конкретное исправление с указанием файла и строки
3. Если возможно, предоставь исправленный код

Ответ должен быть на русском языке."""

    # г) Генерируем ответ через LLM через OpenRouter API
    logger.info("Генерация ответа через LLM (OpenRouter)...")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY не установлен")
        raise ValueError("Не указан OPENROUTER_API_KEY. Установите переменную окружения.")
    
    # Получаем модель LLM (по умолчанию gpt-3.5-turbo)
    model = os.getenv("OPENROUTER_LLM_MODEL", "openai/gpt-3.5-turbo")
    
    # Создаём LLM через OpenRouter API
    try:
        # Для новых версий LangChain
        llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model=model,
            temperature=0
        )
    except TypeError:
        # Для старых версий LangChain
        llm = ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            model_name=model,
            temperature=0
        )
    
    logger.debug("Отправка запроса в LLM...")
    # Генерируем ответ
    try:
        # Для новых версий LangChain используем invoke
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        if hasattr(response, 'content'):
            logger.info("Ответ от LLM получен")
            return response.content
        logger.info("Ответ от LLM получен")
        return str(response)
    except (ImportError, AttributeError):
        try:
            # Для средних версий используем predict
            response = llm.predict(prompt)
            logger.info("Ответ от LLM получен")
            return response
        except AttributeError:
            # Для старых версий вызываем напрямую
            response = llm(prompt)
            logger.info("Ответ от LLM получен")
            return response
