"""FastAPI приложение для обработки stack trace."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from logger_config import setup_logging, get_logger
from vecstore import load_vector_store
from resolver import resolve_error

# Настраиваем логирование при импорте модуля
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", "logs/app.log")
)

logger = get_logger(__name__)


# Глобальная переменная для векторной базы
_vector_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружает векторную базу данных при старте приложения."""
    global _vector_store
    
    # Загружаем векторную базу при старте
    vector_store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    
    logger.info("Запуск приложения, загрузка векторной базы данных...")
    try:
        _vector_store = load_vector_store(path=vector_store_path)
        logger.info(f"Векторная база данных успешно загружена из {vector_store_path}")
    except Exception as e:
        logger.error(f"Не удалось загрузить векторную базу данных: {e}", exc_info=True)
        logger.warning("Приложение запущено, но векторная база не загружена.")
        _vector_store = None
    
    yield
    
    # Очистка при завершении (если нужно)
    logger.info("Завершение работы приложения")
    _vector_store = None


app = FastAPI(
    title="Stack Trace Resolver API",
    version="1.0.0",
    lifespan=lifespan
)


class StackTraceRequest(BaseModel):
    """Модель запроса для обработки stack trace."""
    trace: str


class StackTraceResponse(BaseModel):
    """Модель ответа с решением ошибки."""
    answer: str


@app.post("/resolve", response_model=StackTraceResponse)
async def resolve_stack_trace(request: StackTraceRequest):
    """
    Обрабатывает stack trace и возвращает объяснение ошибки с предложением исправления.
    
    Args:
        request: Запрос с stack trace в поле "trace"
        
    Returns:
        StackTraceResponse: Ответ с объяснением и предложением исправления от LLM
    """
    logger.info("Получен запрос на обработку stack trace")
    if not request.trace:
        logger.warning("Получен запрос с пустым полем 'trace'")
        raise HTTPException(status_code=400, detail="Поле 'trace' не может быть пустым")
    
    # Проверяем, что векторная база загружена
    if _vector_store is None:
        logger.error("Попытка обработать запрос при незагруженной векторной базе")
        raise HTTPException(
            status_code=500,
            detail="Векторная база данных не загружена. Проверьте логи при старте приложения."
        )
    
    try:
        # Используем resolve_error для обработки stack trace
        logger.debug(f"Длина stack trace: {len(request.trace)} символов")
        answer = resolve_error(trace=request.trace, vector_store=_vector_store)
        logger.info("Stack trace успешно обработан")
        return StackTraceResponse(answer=answer)
    except Exception as e:
        logger.error(f"Ошибка при обработке stack trace: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке stack trace: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Проверка здоровья API."""
    return {
        "status": "ok",
        "vector_store_loaded": _vector_store is not None
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    logger.info(f"Запуск API сервера на {host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
