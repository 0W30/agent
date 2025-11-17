"""Модуль для обработки stack trace и разрешения ошибок."""
import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

from agent.logger_config import get_logger

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
        List[Dict[str, any]]: Список словарей с ключами 'file' (str) и 'line' (int или None)
    """
    # Паттерн для поиска путей файлов и номеров строк
    # Примеры: File "/path/to/file.py", line 42
    #          File "/path/to/file.py", line 42, in function_name
    #          File "/path/to/file.py", line ?
    #          File "/path/to/file.py", line 42
    logger.debug("Начало парсинга stack trace")
    
    # Паттерн 1: File "path", line 42 или line ?
    pattern1 = r'File\s+["\']([^"\']+)["\']\s*,\s*line\s+(\d+|\?)'
    
    # Паттерн 2: File "path", line 42, in function (более строгий)
    pattern2 = r'File\s+["\']([^"\']+)["\']\s*,\s*line\s+(\d+|\?)\s*,'
    
    # Паттерн 3: Просто путь к файлу без "File" (для других форматов)
    pattern3 = r'["\']([^"\']+\.py)["\']'
    
    extracted_info = []
    seen_files = set()
    
    # Сначала ищем по основным паттернам
    for pattern in [pattern1, pattern2]:
        matches = re.findall(pattern, trace, re.MULTILINE)
        logger.debug(f"Паттерн {pattern} нашёл {len(matches)} совпадений")
        for file_path, line_num in matches:
            file_path_obj = Path(file_path)
            file_name = file_path_obj.name
            if file_name not in seen_files:
                seen_files.add(file_name)
                # Преобразуем номер строки: ? -> None, иначе int
                line = None if line_num == '?' else int(line_num)
                extracted_info.append({
                    "file": file_name,
                    "line": line
                })
                logger.debug(f"Извлечён файл: {file_name}, строка: {line_num}")
    
    # Если ничего не нашли, пробуем найти просто файлы .py
    if not extracted_info:
        logger.debug("Основные паттерны не нашли совпадений, пробуем найти .py файлы")
        py_files = re.findall(pattern3, trace)
        for file_path in py_files:
            file_path_obj = Path(file_path)
            file_name = file_path_obj.name
            if file_name not in seen_files and file_name.endswith('.py'):
                seen_files.add(file_name)
                extracted_info.append({
                    "file": file_name,
                    "line": None
                })
                logger.debug(f"Извлечён файл (без номера строки): {file_name}")
    
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


def resolve_error(trace: str, vector_store: FAISS, custom_prompt: Optional[str] = None) -> str:
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
    
    # Системный промпт с инструкциями для агента
    # Если передан кастомный промпт, используем его вместо стандартного
    if custom_prompt:
        logger.info("Используется кастомный промпт")
        system_prompt = custom_prompt
    else:
        system_prompt = """Ты - опытный Python-разработчик и эксперт по отладке кода. Твоя задача - анализировать stack trace ошибок и предоставлять детальные объяснения с конкретными исправлениями.

ИНСТРУКЦИИ ПО АНАЛИЗУ ОШИБОК:

1. АНАЛИЗ ОШИБКИ:
   - Внимательно изучи stack trace и определи тип ошибки (TypeError, AttributeError, ImportError, и т.д.)
   - Найдите точное место в коде, где произошла ошибка (файл и строка)
   - Определи причину ошибки: неправильный тип данных, отсутствующий атрибут, неправильный импорт, логическая ошибка и т.д.

2. ПОИСК В КОНТЕКСТЕ:
   - Изучи предоставленные файлы из кодовой базы
   - Найди связанные функции, классы и модули
   - Определи зависимости и связи между компонентами
   - Обрати внимание на импорты и использование переменных

3. СТРУКТУРА ОТВЕТА:
   Ответ должен быть структурирован следующим образом:

   ## Анализ ошибки
   [Краткое описание типа ошибки и её причины]

   ## Причина
   [Детальное объяснение, почему произошла ошибка. Укажи конкретные факторы: неправильные типы данных, отсутствующие атрибуты, неправильная логика и т.д.]

   ## Местоположение
   - Файл: [имя файла]
   - Строка: [номер строки]
   - Функция/Класс: [если применимо]

   ## Решение
   [Конкретное описание того, что нужно исправить]

   ## Исправленный код
   ```python
   [Покажи исправленный фрагмент кода с контекстом (минимум 5-10 строк до и после исправления)]
   ```

   ## Дополнительные рекомендации
   [Если есть связанные проблемы или улучшения, которые стоит учесть]

4. ТРЕБОВАНИЯ К КАЧЕСТВУ:
   - Будь конкретным: указывай точные имена файлов, строки, переменные
   - Предоставляй рабочий код: исправления должны быть готовы к использованию
   - Объясняй причину: не просто показывай исправление, но объясняй почему оно работает
   - Учитывай контекст: анализируй весь связанный код, а не только проблемную строку
   - Предупреждай о побочных эффектах: если исправление может повлиять на другие части кода

5. ОСОБЫЕ СЛУЧАИ:
   - Если ошибка связана с импортами: проверь структуру проекта и пути импорта
   - Если ошибка связана с типами данных: проверь все места, где используются эти данные
   - Если ошибка связана с отсутствующими атрибутами: проверь инициализацию объектов
   - Если ошибка связана с логикой: предложи альтернативные подходы

ВАЖНО: Все ответы должны быть на русском языке. Будь точным, профессиональным и полезным."""
    
    # Если кастомный промпт не передан, используем стандартный системный промпт

    # Пользовательский промпт с данными об ошибке
    user_prompt = f"""Вот stack trace ошибки:

{trace}

Вот релевантные файлы из кодовой базы:

{context}

Проанализируй эту ошибку согласно инструкциям и предоставь структурированный ответ."""

    # г) Генерируем ответ через LLM через OpenRouter API
    logger.info("Генерация ответа через LLM (OpenRouter)...")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY не установлен")
        raise ValueError("Не указан OPENROUTER_API_KEY. Установите переменную окружения.")
    
    # Получаем модель LLM (по умолчанию gpt-3.5-turbo)
    model = os.getenv("OPENROUTER_LLM_MODEL", "openai/gpt-3.5-turbo")
    
    # Автоматически добавляем префикс для моделей Qwen, если его нет
    if model.startswith("qwen") and not model.startswith("qwen/"):
        model = f"qwen/{model}"
        logger.info(f"Автоматически добавлен префикс для модели Qwen: {model}")
    elif "qwen" in model.lower() and "/" not in model:
        # Если модель содержит "qwen" но нет префикса, добавляем qwen/
        model = f"qwen/{model}"
        logger.info(f"Автоматически добавлен префикс для модели Qwen: {model}")
    
    logger.info(f"Используется модель LLM: {model}")
    
    # Создаём LLM через OpenRouter API
    # Пробуем разные варианты инициализации для совместимости
    try:
        # Для новых версий LangChain (langchain-openai)
        llm = ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            model=model,
            temperature=0
        )
        logger.debug("LLM создан через openai_api_key параметр")
    except (TypeError, Exception) as e:
        logger.debug(f"Попытка 1 не удалась: {e}, пробуем другой вариант")
        try:
            # Альтернативный вариант для новых версий
            llm = ChatOpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                model=model,
                temperature=0
            )
            logger.debug("LLM создан через api_key параметр")
        except (TypeError, Exception) as e2:
            logger.debug(f"Попытка 2 не удалась: {e2}, пробуем старый вариант")
            # Для старых версий LangChain
            llm = ChatOpenAI(
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                model_name=model,
                temperature=0
            )
            logger.debug("LLM создан через model_name параметр")
    
    logger.debug("Отправка запроса в LLM...")
    # Генерируем ответ с использованием системного промпта
    try:
        # Для новых версий LangChain используем SystemMessage и HumanMessage
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = llm.invoke(messages)
            if hasattr(response, 'content'):
                logger.info("Ответ от LLM получен")
                return response.content
            logger.info("Ответ от LLM получен")
            return str(response)
        except ImportError:
            # Если SystemMessage недоступен, используем только HumanMessage с объединённым промптом
            from langchain_core.messages import HumanMessage
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = llm.invoke([HumanMessage(content=combined_prompt)])
            if hasattr(response, 'content'):
                logger.info("Ответ от LLM получен")
                return response.content
            logger.info("Ответ от LLM получен")
            return str(response)
    except (ImportError, AttributeError):
        try:
            # Для средних версий используем predict с объединённым промптом
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = llm.predict(combined_prompt)
            logger.info("Ответ от LLM получен")
            return response
        except AttributeError:
            # Для старых версий вызываем напрямую с объединённым промптом
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = llm(combined_prompt)
            logger.info("Ответ от LLM получен")
            return response
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Ошибка при вызове LLM: {error_msg}")
        
        # Проверяем, является ли это ошибкой неверной модели
        if "not a valid model ID" in error_msg or "model" in error_msg.lower() and "400" in error_msg:
            logger.error(
                f"Модель '{model}' не найдена в OpenRouter. "
                f"Проверьте доступные модели Qwen на https://openrouter.ai/models "
                f"или используйте другую модель, например: qwen/qwen-2.5-72b-instruct"
            )
            raise ValueError(
                f"Модель '{model}' не найдена в OpenRouter. "
                f"Проверьте правильность имени модели. "
                f"Для Qwen используйте формат: qwen/model-name"
            ) from e
        
        # Для других ошибок просто пробрасываем
        raise
