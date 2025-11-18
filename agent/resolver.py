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
    logger.debug("Начало парсинга stack trace")

    # Паттерн для поиска путей файлов и номеров строк
    # Примеры: File "/path/to/file.py", line 42
    #          File "/path/to/file.py", line 42, in function_name
    #          File "/path/to/file.py", line ?
    pattern1 = r'File\s+["\']([^"\']+)["\']\s*,\s*line\s+(\d+|\?)'
    pattern2 = r'File\s+["\']([^"\']+)["\']\s*,\s*line\s+(\d+|\?)\s*,'
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
                # Сохраняем полный путь для точного сопоставления
                extracted_info.append({
                    "file": file_name,
                    "line": line,
                    "full_path": file_path,  # Полный путь из stack trace
                    "file_path": str(file_path_obj)  # Нормализованный путь
                })
                logger.debug(f"Извлечён файл: {file_name}, строка: {line_num}, путь: {file_path}")

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
                    "line": None,
                    "full_path": file_path,  # Полный путь из stack trace
                    "file_path": str(file_path_obj)  # Нормализованный путь
                })
                logger.debug(f"Извлечён файл (без номера строки): {file_name}, путь: {file_path}")

    logger.info(f"Парсинг завершён: извлечено {len(extracted_info)} файлов")
    return extracted_info


def get_relevant_docs(stack_info: List[Dict[str, Any]], vector_store: FAISS) -> List[Document]:
    """
    Получает релевантные документы из векторной базы проекта по каждому файлу из stack trace.
    Использует гибридный подход: точное сопоставление по пути + semantic search.

    Args:
        stack_info: Результат parse_stack_trace - список словарей с 'file', 'line', 'full_path'
        vector_store: Векторная база данных FAISS для конкретного проекта

    Returns:
        List[Document]: Объединённый список релевантных документов без дублей, отсортированный по релевантности
    """
    logger.info(f"Поиск релевантных документов для {len(stack_info)} файлов в проекте")

    all_documents = []
    seen_doc_paths = set()  # Для отслеживания дублей по пути файла
    exact_matches = []  # Точные совпадения (приоритет)
    semantic_matches = []  # Semantic search результаты (дополнительно)

    # Собираем все уникальные имена файлов и путей для оптимизации поиска
    file_names = [info.get("file", "") for info in stack_info]
    file_paths_map = {info.get("file", ""): info.get("full_path", "") for info in stack_info}
    
    # Шаг 1: Точное сопоставление по пути файла
    # Делаем один большой поиск для всех файлов сразу (оптимизация)
    all_candidates = []  # Используем список вместо set, так как Document не хешируемы
    seen_candidate_ids = set()  # Отслеживаем дубли по пути файла
    search_query = " ".join(set(file_names))  # Объединяем все имена файлов для одного поиска
    if search_query:
        candidates = vector_store.similarity_search(search_query, k=min(100, len(file_names) * 10))
        for candidate in candidates:
            # Используем путь файла как уникальный идентификатор
            candidate_id = candidate.metadata.get("file_path", candidate.metadata.get("path", ""))
            if candidate_id not in seen_candidate_ids:
                seen_candidate_ids.add(candidate_id)
                all_candidates.append(candidate)
    
    # Фильтруем кандидатов по точному совпадению пути
    for info in stack_info:
        file_name = info.get("file", "")
        full_path = info.get("full_path", "")
        
        logger.debug(f"Точный поиск для файла: {file_name} (путь: {full_path})")
        
        # Ищем точное совпадение среди кандидатов
        found_exact = False
        for doc in all_candidates:
            doc_path = doc.metadata.get("file_path", "")
            doc_name = doc.metadata.get("path", "")
            
            # Проверяем точное совпадение по разным вариантам пути
            is_exact_match = False
            
            if full_path:
                # Сравниваем по полному пути (нормализуем для сравнения)
                full_path_normalized = str(Path(full_path)).replace("\\", "/").lower()
                doc_path_normalized = str(Path(doc_path)).replace("\\", "/").lower()
                # Проверяем, заканчивается ли один путь другим
                if (full_path_normalized.endswith(doc_path_normalized) or 
                    doc_path_normalized.endswith(full_path_normalized) or
                    full_path_normalized in doc_path_normalized or
                    doc_path_normalized in full_path_normalized):
                    is_exact_match = True
            
            # Также проверяем по имени файла
            if not is_exact_match:
                if doc_name == file_name:
                    is_exact_match = True
                # Проверяем, заканчивается ли путь файлом
                elif doc_path and file_name:
                    doc_path_normalized = str(Path(doc_path)).replace("\\", "/").lower()
                    if doc_path_normalized.endswith(file_name.lower()):
                        is_exact_match = True
            
            if is_exact_match:
                doc_id = doc.metadata.get("file_path", doc.metadata.get("path", ""))
                if doc_id not in seen_doc_paths:
                    seen_doc_paths.add(doc_id)
                    exact_matches.append((doc, 1.0))  # Максимальная релевантность для точных совпадений
                    found_exact = True
                    logger.debug(f"Точное совпадение: {doc_path}")
        
        # Если не нашли точного совпадения, используем semantic search для этого конкретного файла
        if not found_exact:
            logger.debug(f"Точное совпадение не найдено для {file_name}, используем semantic search")
            try:
                semantic_results = vector_store.similarity_search_with_score(file_name, k=3)
                for doc, score in semantic_results:
                    doc_id = doc.metadata.get("file_path", doc.metadata.get("path", ""))
                    if doc_id not in seen_doc_paths:
                        seen_doc_paths.add(doc_id)
                        # Нормализуем score (меньше = лучше в FAISS, инвертируем)
                        normalized_score = 1.0 / (1.0 + score) if score > 0 else 0.5
                        semantic_matches.append((doc, normalized_score))
                        logger.debug(f"Semantic match: {doc.metadata.get('file_path')} (score: {score:.3f})")
            except Exception as e:
                logger.warning(f"Ошибка при semantic search для {file_name}: {e}")
                # Fallback на обычный similarity_search
                fallback_results = vector_store.similarity_search(file_name, k=3)
                for doc in fallback_results:
                    doc_id = doc.metadata.get("file_path", doc.metadata.get("path", ""))
                    if doc_id not in seen_doc_paths:
                        seen_doc_paths.add(doc_id)
                        semantic_matches.append((doc, 0.3))  # Низкая релевантность для fallback

    # Объединяем результаты: сначала точные совпадения, потом semantic
    all_documents = [doc for doc, _ in sorted(exact_matches + semantic_matches, key=lambda x: x[1], reverse=True)]
    
    logger.info(f"Найдено {len(exact_matches)} точных совпадений и {len(semantic_matches)} semantic совпадений")
    logger.info(f"Всего найдено {len(all_documents)} уникальных документов")
    
    return all_documents


def build_context(docs: List[Document], stack_info: List[Dict[str, Any]] = None, max_tokens: int = 150000) -> str:
    """
    Склеивает содержимое файлов в один текст с ограничением по токенам.
    Учитывает номера строк из stack trace для выделения релевантных участков кода.
    
    Args:
        docs: Список документов для объединения
        stack_info: Информация о файлах и строках из stack trace (опционально)
        max_tokens: Максимальное количество токенов (примерно 4 символа = 1 токен)
        
    Returns:
        str: Объединённый контекст
    """
    logger.info(f"Построение контекста из {len(docs)} документов, максимум токенов: {max_tokens}")
    max_chars = max_tokens * 4  # Примерная оценка: 4 символа = 1 токен
    
    # Создаём словарь для быстрого поиска номеров строк по именам файлов
    file_lines_map = {}
    if stack_info:
        for info in stack_info:
            file_name = info.get("file")
            line_num = info.get("line")
            if file_name and line_num is not None:
                if file_name not in file_lines_map:
                    file_lines_map[file_name] = []
                file_lines_map[file_name].append(line_num)
    
    combined = []
    current_length = 0
    
    for doc in docs:
        # Получаем имя файла из metadata (может быть полный путь или только имя)
        file_path_meta = doc.metadata.get("path", "unknown")
        file_path_relative = doc.metadata.get("file_path", file_path_meta)
        file_name = Path(file_path_meta).name
        file_name_relative = Path(file_path_relative).name
        
        # Проверяем, является ли документ чанком
        is_chunk = "chunk_index" in doc.metadata
        chunk_start = doc.metadata.get("start_line", None)
        chunk_end = doc.metadata.get("end_line", None)
        
        content_lines = doc.page_content.split('\n')
        
        # Проверяем, есть ли информация о строках для этого файла
        # Пробуем найти по имени файла или по относительному пути
        relevant_lines = file_lines_map.get(file_name, [])
        if not relevant_lines:
            relevant_lines = file_lines_map.get(file_name_relative, [])
        
        # Также проверяем по полному пути из stack trace (если есть)
        if not relevant_lines and stack_info:
            for info in stack_info:
                full_path = info.get("full_path", "")
                if full_path:
                    full_path_name = Path(full_path).name
                    if full_path_name == file_name or full_path_name == file_name_relative:
                        line_num = info.get("line")
                        if line_num is not None:
                            if not relevant_lines:
                                relevant_lines = []
                            relevant_lines.append(line_num)
        
        # Если это чанк и есть номера строк, проверяем, попадает ли нужная строка в этот чанк
        if is_chunk and relevant_lines and chunk_start and chunk_end:
            # Фильтруем только те строки, которые попадают в диапазон чанка
            relevant_lines_in_chunk = [
                line_num for line_num in relevant_lines 
                if chunk_start <= line_num <= chunk_end
            ]
            if relevant_lines_in_chunk:
                # Нужная строка в этом чанке - используем весь чанк
                relevant_lines = relevant_lines_in_chunk
            else:
                # Нужная строка не в этом чанке, но чанк может быть релевантным через semantic search
                # Показываем его как дополнительный контекст
                relevant_lines = []
        
        if relevant_lines:
            # Если это чанк и нужная строка в нём, показываем весь чанк с выделением проблемных строк
            if is_chunk and chunk_start and chunk_end:
                # Показываем весь чанк, но выделяем проблемные строки
                numbered_lines = []
                for i, line in enumerate(content_lines):
                    actual_line = chunk_start + i  # Номер строки в файле
                    marker = ">>> " if actual_line in relevant_lines else "    "
                    numbered_lines.append(f"{marker}{actual_line:4d} | {line}")
                
                display_path = file_path_relative if file_path_relative != "unknown" else file_path_meta
                chunk_info = f" [чанк {doc.metadata.get('chunk_index', 0) + 1}/{doc.metadata.get('total_chunks', 1)}, строки {chunk_start}-{chunk_end}]"
                content = f"=== Файл: {display_path}{chunk_info} (строки из stack trace: {', '.join(map(str, relevant_lines))}) ===\n" + "\n".join(numbered_lines) + "\n\n"
            else:
                # Если это не чанк или старый формат, выделяем контекст вокруг проблемных строк
                context_ranges = []
                for line_num in relevant_lines:
                    # Берём контекст: 20 строк до и 20 строк после проблемной строки
                    # line_num - это номер строки (начинается с 1), индексы начинаются с 0
                    start = max(0, line_num - 1 - 20)  # line_num-1 для индексации с 0, -20 для контекста
                    end = min(len(content_lines), line_num - 1 + 21)  # line_num-1 + 20 строк после + сама строка
                    context_ranges.append((start, end, line_num))
                
                # Объединяем перекрывающиеся диапазоны
                if context_ranges:
                    context_ranges.sort()
                    merged_ranges = [context_ranges[0]]
                    for start, end, line_num in context_ranges[1:]:
                        last_start, last_end, _ = merged_ranges[-1]
                        if start <= last_end:
                            # Перекрываются - объединяем
                            merged_ranges[-1] = (last_start, max(last_end, end), line_num)
                        else:
                            merged_ranges.append((start, end, line_num))
                    
                    # Формируем контекст с выделением проблемных строк
                    content_parts = []
                    for start, end, line_num in merged_ranges:
                        context_lines = content_lines[start:end]
                        # Добавляем номера строк и выделяем проблемную строку
                        numbered_lines = []
                        for i, line in enumerate(context_lines):
                            actual_line = start + i + 1  # +1 потому что строки нумеруются с 1
                            marker = ">>> " if actual_line == line_num else "    "
                            numbered_lines.append(f"{marker}{actual_line:4d} | {line}")
                        
                        content_parts.append("\n".join(numbered_lines))
                    
                    display_path = file_path_relative if file_path_relative != "unknown" else file_path_meta
                    content = f"=== Файл: {display_path} (строки из stack trace: {', '.join(map(str, relevant_lines))}) ===\n" + "\n".join(content_parts) + "\n\n"
                else:
                    # context_ranges пуст (не удалось создать диапазоны) - показываем весь файл
                    display_path = file_path_relative if file_path_relative != "unknown" else file_path_meta
                    content = f"=== Файл: {display_path} ===\n{doc.page_content}\n\n"
        else:
            # Файл не упомянут в stack trace напрямую, но найден через similarity search
            # Если это чанк, показываем весь чанк, иначе первые 100 строк
            if is_chunk and chunk_start and chunk_end:
                # Показываем весь чанк
                chunk_info = f" [чанк {doc.metadata.get('chunk_index', 0) + 1}/{doc.metadata.get('total_chunks', 1)}, строки {chunk_start}-{chunk_end}]"
                display_path = file_path_relative if file_path_relative != "unknown" else file_path_meta
                content = f"=== Файл: {display_path}{chunk_info} (найден через similarity search) ===\n{doc.page_content}\n\n"
            else:
                # Показываем начало файла (первые 100 строк)
                preview_lines = content_lines[:100]
                if len(content_lines) > 100:
                    preview_lines.append(f"\n... (файл обрезан, всего {len(content_lines)} строк)")
                display_path = file_path_relative if file_path_relative != "unknown" else file_path_meta
                content = f"=== Файл: {display_path} (найден через similarity search, показаны первые 100 строк) ===\n" + "\n".join(preview_lines) + "\n\n"
        
        content_length = len(content)
        
        # Проверяем, не превысим ли лимит
        if current_length + content_length > max_chars:
            remaining_chars = max_chars - current_length
            if remaining_chars > 100:
                content = content[:remaining_chars - 50] + "\n\n[контекст обрезан]\n\n"
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
    
    # б) Получаем релевантные документы (теперь векторная база уже отфильтрована по проекту)
    relevant_docs = get_relevant_docs(stack_info, vector_store)
    
    if not relevant_docs:
        logger.warning("Не найдено релевантных файлов в векторной базе данных")
        return "Не найдено релевантных файлов в векторной базе данных."
    
    # в) Строим контекст с учётом номеров строк из stack trace
    context = build_context(relevant_docs, stack_info=stack_info, max_tokens=150000)
    
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
