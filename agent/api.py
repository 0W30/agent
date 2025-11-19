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
from agent.yandex_tracker import YandexTrackerClient, create_tracker_client

try:
    from yandex_tracker_client.exceptions import TrackerError, NotFound
except ImportError:
    TrackerError = Exception
    NotFound = Exception

# Настраиваем логирование при импорте модуля
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", "logs/app.log")
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация приложения."""
    logger.info("Запуск приложения Stack Trace Resolver...")
    logger.info("Используйте endpoint /clone для создания векторных баз по проектам")
    
    # Проверяем настройку Яндекс Трекера
    tracker_client = create_tracker_client()
    if tracker_client:
        logger.info("Яндекс Трекер настроен и готов к использованию")
    else:
        logger.info("Яндекс Трекер не настроен (установите YANDEX_TRACKER_TOKEN и YANDEX_TRACKER_ORG_ID)")

    yield

    logger.info("Завершение работы приложения")


app = FastAPI(
    title="Stack Trace Resolver API",
    version="1.0.0",
    lifespan=lifespan
)


class StackTraceRequest(BaseModel):
    """Модель запроса для обработки stack trace.

    Принимает объект с полями:
    - stacktrace: обязательное поле со stack trace
    - project_name: обязательное поле с именем проекта
    - message: опциональное поле с сообщением об ошибке
    - exception_type: опциональное поле с типом исключения
    - exception_value: опциональное поле со значением исключения
    - send_to_tracker: опциональное поле для автоматической отправки в Яндекс Трекер
    - tracker_queue: опциональное поле с ключом очереди в Трекере
    """
    stacktrace: str = Field(..., description="Stack trace в виде строки")
    project_name: str = Field(..., description="Имя проекта для поиска релевантных документов")
    message: Optional[str] = Field(None, description="Сообщение об ошибке")
    exception_type: Optional[str] = Field(None, description="Тип исключения")
    exception_value: Optional[str] = Field(None, description="Значение исключения")
    send_to_tracker: Optional[bool] = Field(False, description="Автоматически отправить решение в Яндекс Трекер")
    tracker_queue: Optional[str] = Field(None, description="Ключ очереди в Яндекс Трекере (обязательно, если send_to_tracker=True)")

    class Config:
        json_schema_extra = {
            "example": {
                "stacktrace": "File \"test.py\", line 42, in function_name\n    code here",
                "project_name": "my-project",
                "message": "TypeError: unsupported operand type",
                "exception_type": "TypeError",
                "exception_value": "unsupported operand type",
                "send_to_tracker": False,
                "tracker_queue": "TEST"
            }
        }


class StackTraceResponse(BaseModel):
    """Модель ответа с решением ошибки."""
    answer: str
    tracker_issue_key: Optional[str] = Field(None, description="Ключ созданной задачи в Яндекс Трекере (если была отправка)")


class CloneRepoRequest(BaseModel):
    """Модель запроса для клонирования репозитория."""
    url: str = Field(..., description="SSH или HTTPS URL репозитория")
    branch: str = "main"
    project_name: str = Field(..., description="Уникальное имя проекта для группировки векторов")
    target_dir: Optional[str] = None  # Если не указан, будет сгенерирован автоматически

    class Config:
        json_schema_extra = {
            "example": {
                "url": "git@github.com:user/repo.git",
                "branch": "main",
                "project_name": "my-project"
            }
        }


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

    if not request.project_name:
        logger.warning("Получен запрос с пустым полем 'project_name'")
        raise HTTPException(status_code=400, detail="Поле 'project_name' не может быть пустым")
    
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
    
    # Загружаем векторную базу для указанного проекта
    project_vector_store_path = f"./vector_store/{request.project_name}"
    try:
        project_vector_store = load_vector_store(path=project_vector_store_path)
        logger.info(f"Загружена векторная база проекта '{request.project_name}'")
    except (ValueError, FileNotFoundError):
        logger.error(f"Векторная база для проекта '{request.project_name}' не найдена")
        raise HTTPException(
            status_code=404,
            detail=f"Векторная база для проекта '{request.project_name}' не найдена. Сначала клонируйте проект с помощью /clone."
        )
    
    try:
        # Используем resolve_error для обработки stack trace
        logger.debug(f"Длина stack trace: {len(full_trace)} символов")
        answer = resolve_error(trace=full_trace, vector_store=project_vector_store)
        logger.info("Stack trace успешно обработан")
        
        # Отправка в Яндекс Трекер, если запрошено
        tracker_issue_key = None
        if request.send_to_tracker:
            if not request.tracker_queue:
                raise HTTPException(
                    status_code=400,
                    detail="Для отправки в Яндекс Трекер необходимо указать tracker_queue"
                )
            
            try:
                tracker_client = create_tracker_client()
                if not tracker_client:
                    logger.warning("Яндекс Трекер не настроен, пропускаем отправку")
                else:
                    # Формируем описание для задачи
                    issue_summary = f"{request.exception_type or 'Ошибка'}: {request.message or 'Ошибка в коде'}"
                    if len(issue_summary) > 255:
                        issue_summary = issue_summary[:252] + "..."
                    
                    issue_description = f"""**Ошибка:** {request.exception_type or 'Неизвестная ошибка'}

**Сообщение:** {request.message or 'Нет сообщения'}

**Проект:** {request.project_name}

**Stack Trace:**
```
{request.stacktrace[:5000]}
```

**Предложенное решение:**
{answer[:10000]}
"""
                    
                    result = tracker_client.create_issue(
                        queue=request.tracker_queue,
                        summary=issue_summary,
                        description=issue_description,
                        tags=["auto-generated", "stack-trace", request.project_name]
                    )
                    tracker_issue_key = result.get("key")
                    logger.info(f"Решение отправлено в Яндекс Трекер: {tracker_issue_key}")
            except NotFound as tracker_error:
                logger.error(f"Задача не найдена при отправке в Яндекс Трекер: {tracker_error}")
                # Не прерываем выполнение, просто логируем ошибку
            except TrackerError as tracker_error:
                logger.error(f"Ошибка Яндекс Трекера при отправке: {tracker_error}")
                # Не прерываем выполнение, просто логируем ошибку
            except Exception as tracker_error:
                logger.error(f"Неожиданная ошибка при отправке в Яндекс Трекер: {tracker_error}", exc_info=True)
                # Не прерываем выполнение, просто логируем ошибку
        
        return StackTraceResponse(answer=answer, tracker_issue_key=tracker_issue_key)
    except HTTPException:
        raise
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
    
    logger.info(f"Получен запрос на клонирование репозитория: {request.url}, ветка: {request.branch}")

    if not request.url:
        logger.warning("Получен запрос с пустым URL")
        raise HTTPException(status_code=400, detail="Поле 'url' не может быть пустым")

    if not request.project_name:
        logger.warning("Получен запрос с пустым именем проекта")
        raise HTTPException(status_code=400, detail="Поле 'project_name' не может быть пустым")

    try:
        # Генерируем путь для клонирования, если не указан
        if not request.target_dir:
            # Извлекаем имя репозитория из URL (работает для SSH и HTTPS)
            if request.url.startswith('git@'):
                # SSH URL: git@github.com:user/repo.git -> repo
                repo_name = request.url.split(':')[-1].replace('.git', '')
            else:
                # HTTPS URL: https://github.com/user/repo.git -> repo
                repo_name = request.url.split('/')[-1].replace('.git', '')
            request.target_dir = f"./cloned_repos/{repo_name}"

        # Клонируем репозиторий
        logger.info(f"Клонирование репозитория в {request.target_dir}")
        repo_path = clone_repo(
            url=request.url,
            branch=request.branch,
            target_dir=request.target_dir
        )
        logger.info(f"Репозиторий успешно клонирован: {repo_path}")
        
        # Индексируем файлы
        logger.info("Начало индексации файлов...")
        documents = extract_python_files(repo_path)
        files_count = len(documents)
        logger.info(f"Проиндексировано {files_count} файлов")
        
        if files_count == 0:
            return CloneRepoResponse(
                success=False,
                message="В репозитории не найдено файлов для индексации",
                repo_path=repo_path,
                files_indexed=0
            )
        
        # Создаём векторную базу для этого проекта
        base_path = os.getenv("VECTOR_STORE_PATH", "./vector_store")
        vector_store_path = f"{base_path}/{request.project_name}"
        logger.info(f"Создание векторной базы для проекта '{request.project_name}' в {vector_store_path}")
        project_vector_store = create_vector_store(docs=documents, path=vector_store_path)
        logger.info(f"Векторная база проекта '{request.project_name}' успешно создана")
        
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
    return {"status": "ok"}


# Точка входа вынесена в main.py в корне проекта
