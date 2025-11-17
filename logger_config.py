"""Конфигурация логирования для проекта."""
import logging
import sys
from pathlib import Path


def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    log_format: str = None
) -> logging.Logger:
    """
    Настраивает логирование для приложения.
    
    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Путь к файлу для записи логов (опционально)
        log_format: Формат логов (опционально)
        
    Returns:
        logging.Logger: Настроенный логгер
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Преобразуем строку уровня в константу
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Настраиваем форматтер
    formatter = logging.Formatter(log_format)
    
    # Настраиваем обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    
    # Создаём корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()  # Очищаем существующие обработчики
    root_logger.addHandler(console_handler)
    
    # Если указан файл, добавляем файловый обработчик
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Настраиваем логирование для внешних библиотек
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Получает логгер с указанным именем.
    
    Args:
        name: Имя логгера (обычно __name__ модуля)
        
    Returns:
        logging.Logger: Логгер
    """
    return logging.getLogger(name)

