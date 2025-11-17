"""FastAPI приложение для обработки stack trace."""
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.logger_config import setup_logging, get_logger
from agent.vecstore import load_vector_store, create_vector_store
from agent.repo_downloader import clone_repo
from agent.indexer import extract_python_files
from agent.resolver import resolve_error

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
    """Загружает векторную базу данных при старте приложения (если существует)."""
    global _vector_store
    
    # Пытаемся загрузить векторную базу при старте (если существует)
    vector_store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    
    logger.info("Запуск приложения, попытка загрузки векторной базы данных...")
    try:
        _vector_store = load_vector_store(path=vector_store_path)
        logger.info(f"Векторная база данных успешно загружена из {vector_store_path}")
    except Exception as e:
        logger.warning(f"Векторная база данных не найдена или не может быть загружена: {e}")
        logger.info("Используйте endpoint /clone для создания векторной базы из репозитория")
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
    """Модель запроса для обработки stack trace.
    
    Принимает объект с полями:
    - stacktrace: обязательное поле со stack trace
    - message: опциональное поле с сообщением об ошибке
    - exception_type: опциональное поле с типом исключения
    - exception_value: опциональное поле со значением исключения
    """
    stacktrace: str = Field(..., description="Stack trace в виде строки")
    message: Optional[str] = Field(None, description="Сообщение об ошибке")
    exception_type: Optional[str] = Field(None, description="Тип исключения")
    exception_value: Optional[str] = Field(None, description="Значение исключения")
    
    class Config:
        json_schema_extra = {
            "example": {
                "stacktrace": "File \"test.py\", line 42, in function_name\n    code here",
                "message": "TypeError: unsupported operand type",
                "exception_type": "TypeError",
                "exception_value": "unsupported operand type"
            }
        }


class StackTraceResponse(BaseModel):
    """Модель ответа с решением ошибки."""
    answer: str


class CloneRepoRequest(BaseModel):
    """Модель запроса для клонирования репозитория."""
    ssh_url: str
    branch: str = "main"
    target_dir: Optional[str] = None  # Если не указан, будет сгенерирован автоматически


class CloneRepoResponse(BaseModel):
    """Модель ответа на клонирование репозитория."""
    success: bool
    message: str
    repo_path: str = None
    files_indexed: int = 0
    vector_store_path: str = None


class PromptRequest(BaseModel):
    """Модель запроса для обработки с кастомным промптом."""
    trace: str
    prompt: Optional[str] = None  # Опциональный кастомный промпт


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
    
    if not request.stacktrace or not request.stacktrace.strip():
        logger.warning("Получен запрос с пустым полем 'stacktrace'")
        raise HTTPException(status_code=400, detail="Поле 'stacktrace' не может быть пустым")
    
    # Формируем полный trace с информацией об ошибке
    trace_parts = []
    
    # Добавляем информацию об ошибке, если есть
    if request.exception_type and request.exception_value:
        trace_parts.append(f"{request.exception_type}: {request.exception_value}")
    elif request.message:
        trace_parts.append(request.message)
    elif request.exception_type:
        trace_parts.append(request.exception_type)
    
    # Добавляем stacktrace
    trace_parts.append(request.stacktrace)
    
    # Объединяем все части
    full_trace = "\n".join(trace_parts)
    
    # Проверяем, что векторная база загружена
    if _vector_store is None:
        logger.error("Попытка обработать запрос при незагруженной векторной базе")
        raise HTTPException(
            status_code=500,
            detail="Векторная база данных не загружена. Используйте /clone для её создания."
        )
    
    try:
        # Используем resolve_error для обработки stack trace
        logger.debug(f"Длина stack trace: {len(full_trace)} символов")
        answer = resolve_error(trace=full_trace, vector_store=_vector_store)
        logger.info("Stack trace успешно обработан")
        return StackTraceResponse(answer=answer)
    except Exception as e:
        logger.error(f"Ошибка при обработке stack trace: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке stack trace: {str(e)}"
        )


@app.post("/clone", response_model=CloneRepoResponse)
async def clone_repository(request: CloneRepoRequest):
    """
    Клонирует Git-репозиторий по SSH, индексирует Python-файлы и создаёт векторную базу данных.
    
    Args:
        request: Запрос с SSH URL, веткой и опциональным путём для клонирования
        
    Returns:
        CloneRepoResponse: Результат операции клонирования и индексации
    """
    global _vector_store
    
    logger.info(f"Получен запрос на клонирование репозитория: {request.ssh_url}, ветка: {request.branch}")
    
    if not request.ssh_url:
        logger.warning("Получен запрос с пустым SSH URL")
        raise HTTPException(status_code=400, detail="Поле 'ssh_url' не может быть пустым")
    
    try:
        # Генерируем путь для клонирования, если не указан
        if not request.target_dir:
            # Используем имя репозитория из SSH URL
            repo_name = request.ssh_url.split("/")[-1].replace(".git", "")
            request.target_dir = f"./cloned_repos/{repo_name}"
        
        # Клонируем репозиторий
        logger.info(f"Клонирование репозитория в {request.target_dir}")
        repo_path = clone_repo(
            ssh_url=request.ssh_url,
            branch=request.branch,
            target_dir=request.target_dir
        )
        logger.info(f"Репозиторий успешно клонирован: {repo_path}")
        
        # Индексируем Python-файлы
        logger.info("Начало индексации Python-файлов...")
        documents = extract_python_files(repo_path)
        files_count = len(documents)
        logger.info(f"Проиндексировано {files_count} Python-файлов")
        
        if files_count == 0:
            return CloneRepoResponse(
                success=False,
                message="В репозитории не найдено Python-файлов для индексации",
                repo_path=repo_path,
                files_indexed=0
            )
        
        # Создаём векторную базу
        vector_store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store")
        logger.info(f"Создание векторной базы в {vector_store_path}")
        _vector_store = create_vector_store(docs=documents, path=vector_store_path)
        logger.info("Векторная база успешно создана")
        
        return CloneRepoResponse(
            success=True,
            message=f"Репозиторий успешно клонирован и проиндексирован. Создана векторная база из {files_count} файлов.",
            repo_path=repo_path,
            files_indexed=files_count,
            vector_store_path=vector_store_path
        )
        
    except Exception as e:
        logger.error(f"Ошибка при клонировании репозитория: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при клонировании репозитория: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Проверка здоровья API."""
    return {
        "status": "ok",
        "vector_store_loaded": _vector_store is not None
    }


# Точка входа вынесена в main.py в корне проекта
