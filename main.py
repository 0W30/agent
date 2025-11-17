"""Основной модуль для демонстрации работы системы разрешения ошибок."""
import os
import sys
from pathlib import Path

from logger_config import setup_logging, get_logger
from repo_downloader import clone_repo
from indexer import extract_python_files
from vecstore import create_vector_store, load_vector_store
from resolver import resolve_error

# Настраиваем логирование
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", "logs/main.log")
)

logger = get_logger(__name__)


def main():
    """Основная функция для демонстрации работы системы."""
    logger.info("=" * 60)
    logger.info("Система разрешения ошибок из stack trace")
    logger.info("=" * 60)
    
    # Проверяем наличие необходимых переменных окружения
    required_vars = ["OPENROUTER_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Отсутствуют необходимые переменные окружения: {', '.join(missing_vars)}")
        logger.info("\nДля использования системы необходимо установить:")
        logger.info("1. OPENROUTER_API_KEY - для создания эмбеддингов и генерации ответов через LLM")
        logger.info("\nПример:")
        logger.info("export OPENROUTER_API_KEY='your-openrouter-key'")
        return
    
    logger.info("Все необходимые переменные окружения установлены")
    
    # Пример полного workflow:
    logger.info("\n" + "=" * 60)
    logger.info("Пример использования системы:")
    logger.info("=" * 60)
    
    example_workflow = """
    # 1. Скачивание репозитория
    ssh_url = "git@github.com:user/repo.git"
    branch = "main"
    target_dir = "./cloned_repo"
    repo_path = clone_repo(ssh_url, branch, target_dir)
    
    # 2. Индексация Python-файлов
    documents = extract_python_files(repo_path)
    
    # 3. Создание векторной базы данных
    vector_store = create_vector_store(documents, path="./vector_store")
    
    # 4. Разрешение ошибки из stack trace
    stack_trace = '''
    Traceback (most recent call last):
      File "/path/to/file.py", line 42, in function_name
        result = some_function()
      File "/path/to/other_file.py", line 10, in some_function
        return value / 0
    ZeroDivisionError: division by zero
    '''
    
    answer = resolve_error(trace=stack_trace, vector_store=vector_store)
    print(answer)
    """
    
    logger.info(example_workflow)
    logger.info("\n" + "=" * 60)
    logger.info("Для запуска API сервера используйте: python api.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Прервано пользователем")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)
