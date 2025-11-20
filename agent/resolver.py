"""Модуль для обработки stack trace и разрешения ошибок."""
import os
from typing import Optional

from agent.logger_config import get_logger
from agent.context_builder import parse_stack_trace, get_relevant_docs, build_context

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
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    SystemMessage = None
    HumanMessage = None

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS


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
    
    # Получаем модель LLM
    model = os.getenv("OPENROUTER_LLM_MODEL", "qwen/qwen-2.5-72b-instruct")
    
    logger.info(f"Используется модель LLM: {model}")
    
    # Создаём LLM через OpenRouter API
    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model=model,
        temperature=0
    )
    
    logger.debug("Отправка запроса в LLM...")
    
    # Генерируем ответ с использованием системного промпта
    try:
        if SystemMessage and HumanMessage:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = llm.invoke(messages)
        else:
            # Fallback: объединяем промпты
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            if HumanMessage:
                response = llm.invoke([HumanMessage(content=combined_prompt)])
            else:
                response = llm.invoke(combined_prompt)
        
        # Извлекаем контент из ответа
        if hasattr(response, 'content'):
            logger.info("Ответ от LLM получен")
            return response.content
        logger.info("Ответ от LLM получен")
        return str(response)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Ошибка при вызове LLM: {error_msg}")
        raise
