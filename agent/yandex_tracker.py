"""Модуль для работы с API Яндекс Трекера через официальную библиотеку."""
import os
from typing import Optional, Dict, Any

try:
    from yandex_tracker_client import TrackerClient
    from yandex_tracker_client.exceptions import TrackerError, NotFound
except ImportError:
    TrackerClient = None
    TrackerError = Exception
    NotFound = Exception

from agent.logger_config import get_logger

logger = get_logger(__name__)


class YandexTrackerClient:
    """Обёртка над официальным клиентом Яндекс Трекера."""
    
    def __init__(
        self,
        oauth_token: Optional[str] = None,
        org_id: Optional[str] = None,
        cloud_org_id: Optional[str] = None,
        iam_token: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Инициализация клиента Яндекс Трекера.
        
        Поддерживает два режима:
        1. Обычный Яндекс Трекер: oauth_token + org_id
        2. Яндекс 360 для бизнеса: iam_token + cloud_org_id
        
        Args:
            oauth_token: OAuth токен для авторизации (можно получить из переменной окружения YANDEX_TRACKER_TOKEN)
            org_id: ID организации в Яндекс Трекере (можно получить из переменной окружения YANDEX_TRACKER_ORG_ID)
            cloud_org_id: ID организации в Яндекс 360 для бизнеса (можно получить из переменной окружения YANDEX_TRACKER_CLOUD_ORG_ID)
            iam_token: IAM токен для Яндекс 360 (можно получить из переменной окружения YANDEX_TRACKER_IAM_TOKEN)
            base_url: Базовый URL API (не используется, оставлен для совместимости)
        """
        if TrackerClient is None:
            raise ImportError(
                "Библиотека yandex_tracker_client не установлена. "
                "Установите её: pip install yandex-tracker-client"
            )
        
        self.oauth_token = oauth_token or os.getenv("YANDEX_TRACKER_TOKEN")
        self.org_id = org_id or os.getenv("YANDEX_TRACKER_ORG_ID")
        self.cloud_org_id = cloud_org_id or os.getenv("YANDEX_TRACKER_CLOUD_ORG_ID")
        self.iam_token = iam_token or os.getenv("YANDEX_TRACKER_IAM_TOKEN")
        
        # Определяем режим работы
        # OAuth токен может использоваться с cloud_org_id или org_id
        # IAM токен используется только с cloud_org_id
        self.is_iam_mode = bool(self.cloud_org_id and self.iam_token)
        self.is_oauth_with_cloud = bool(self.oauth_token and self.cloud_org_id)
        self.is_oauth_with_org = bool(self.oauth_token and self.org_id and not self.cloud_org_id)
        
        # Детальное логирование для диагностики
        if self.is_iam_mode:
            logger.debug(f"Режим IAM токена: cloud_org_id={self.cloud_org_id}")
        elif self.is_oauth_with_cloud:
            logger.debug(f"Режим OAuth токена с cloud_org_id: cloud_org_id={self.cloud_org_id}")
        elif self.is_oauth_with_org:
            logger.debug(f"Режим OAuth токена с org_id: org_id={self.org_id}")
        else:
            if not self.oauth_token and not self.iam_token:
                logger.warning("Токен не установлен. Работа с Трекером будет недоступна.")
            if not self.org_id and not self.cloud_org_id:
                logger.warning("ID организации не установлен. Работа с Трекером будет недоступна.")
            logger.debug("Для OAuth токена используйте YANDEX_TRACKER_TOKEN и YANDEX_TRACKER_CLOUD_ORG_ID (или YANDEX_TRACKER_ORG_ID)")
            logger.debug("Для IAM токена используйте YANDEX_TRACKER_IAM_TOKEN и YANDEX_TRACKER_CLOUD_ORG_ID")
        
        # Инициализируем официальный клиент
        if self.is_iam_mode:
            try:
                self._client = TrackerClient(iam_token=self.iam_token, cloud_org_id=self.cloud_org_id)
                logger.debug("Клиент Яндекс Трекера (IAM токен) успешно инициализирован")
            except Exception as e:
                logger.error(f"Ошибка при инициализации клиента Яндекс Трекера (IAM токен): {e}")
                self._client = None
        elif self.is_oauth_with_cloud:
            try:
                self._client = TrackerClient(token=self.oauth_token, cloud_org_id=self.cloud_org_id)
                logger.debug("Клиент Яндекс Трекера (OAuth токен с cloud_org_id) успешно инициализирован")
            except Exception as e:
                logger.error(f"Ошибка при инициализации клиента Яндекс Трекера (OAuth токен с cloud_org_id): {e}")
                self._client = None
        elif self.is_oauth_with_org:
            try:
                self._client = TrackerClient(token=self.oauth_token, org_id=self.org_id)
                logger.debug("Клиент Яндекс Трекера (OAuth токен с org_id) успешно инициализирован")
            except Exception as e:
                logger.error(f"Ошибка при инициализации клиента Яндекс Трекера (OAuth токен с org_id): {e}")
                self._client = None
        else:
            self._client = None
    
    def create_issue(
        self,
        queue: str,
        summary: str,
        description: str,
        assignee: Optional[str] = None,
        priority: Optional[str] = None,
        tags: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Создаёт задачу в Яндекс Трекере.
        
        Args:
            queue: Ключ очереди (обязательно)
            summary: Краткое описание задачи (обязательно)
            description: Подробное описание задачи (обязательно)
            assignee: Логин исполнителя (опционально)
            priority: Приоритет задачи (опционально, например: "critical", "high", "normal", "low")
            tags: Список тегов (опционально)
            **kwargs: Дополнительные поля задачи (например: "type", "components", и т.д.)
            
        Returns:
            Dict[str, Any]: Информация о созданной задаче (содержит ключ "key")
            
        Raises:
            ValueError: Если не установлены токен или org_id
            TrackerError: При ошибке запроса к API
        """
        if not self._client:
            if self.is_iam_mode:
                raise ValueError("Клиент Яндекс Трекера не инициализирован. Проверьте YANDEX_TRACKER_IAM_TOKEN и YANDEX_TRACKER_CLOUD_ORG_ID.")
            elif self.is_oauth_with_cloud:
                raise ValueError("Клиент Яндекс Трекера не инициализирован. Проверьте YANDEX_TRACKER_TOKEN и YANDEX_TRACKER_CLOUD_ORG_ID.")
            elif self.is_oauth_with_org:
                raise ValueError("Клиент Яндекс Трекера не инициализирован. Проверьте YANDEX_TRACKER_TOKEN и YANDEX_TRACKER_ORG_ID.")
            else:
                raise ValueError("Клиент Яндекс Трекера не инициализирован. Установите необходимые переменные окружения.")
        
        logger.info(f"Создание задачи в очереди '{queue}' с темой: {summary[:50]}...")
        
        try:
            # Формируем параметры для создания задачи
            issue_params = {
                "queue": queue,
                "summary": summary,
                "description": description
            }
            
            # Добавляем опциональные поля
            if assignee:
                issue_params["assignee"] = assignee
            
            if priority:
                # Приоритет передаём как строку (например: "critical", "high", "normal", "low")
                # или как словарь, если передан словарь
                issue_params["priority"] = priority
            
            if tags:
                issue_params["tags"] = tags
            
            # Добавляем дополнительные поля из kwargs
            issue_params.update(kwargs)
            
            # Создаём задачу через официальный клиент
            issue = self._client.issues.create(**issue_params)
            
            issue_key = issue.key
            logger.info(f"Задача успешно создана: {issue_key}")
            
            # Возвращаем словарь с информацией о задаче (для совместимости)
            return {
                "key": issue_key,
                "id": issue.id,
                "summary": issue.summary,
                "status": issue.status.get("key") if hasattr(issue.status, "get") else str(issue.status),
            }
            
        except TrackerError as e:
            error_msg = f"Ошибка при создании задачи в Яндекс Трекере: {e}"
            logger.error(error_msg)
            raise TrackerError(error_msg) from e
        except Exception as e:
            error_msg = f"Неожиданная ошибка при создании задачи: {e}"
            logger.error(error_msg, exc_info=True)
            raise TrackerError(error_msg) from e
    
    def add_comment(self, issue_key: str, text: str) -> Dict[str, Any]:
        """
        Добавляет комментарий к задаче.
        
        Args:
            issue_key: Ключ задачи (например, "TEST-123")
            text: Текст комментария
            
        Returns:
            Dict[str, Any]: Информация о созданном комментарии
            
        Raises:
            ValueError: Если не установлены токен или org_id
            TrackerError: При ошибке запроса к API
            NotFound: Если задача не найдена
        """
        if not self._client:
            if self.is_iam_mode:
                raise ValueError("Клиент Яндекс Трекера не инициализирован. Проверьте YANDEX_TRACKER_IAM_TOKEN и YANDEX_TRACKER_CLOUD_ORG_ID.")
            elif self.is_oauth_with_cloud:
                raise ValueError("Клиент Яндекс Трекера не инициализирован. Проверьте YANDEX_TRACKER_TOKEN и YANDEX_TRACKER_CLOUD_ORG_ID.")
            elif self.is_oauth_with_org:
                raise ValueError("Клиент Яндекс Трекера не инициализирован. Проверьте YANDEX_TRACKER_TOKEN и YANDEX_TRACKER_ORG_ID.")
            else:
                raise ValueError("Клиент Яндекс Трекера не инициализирован. Установите необходимые переменные окружения.")
        
        logger.info(f"Добавление комментария к задаче {issue_key}")
        
        try:
            # Получаем задачу
            issue = self._client.issues[issue_key]
            
            # Добавляем комментарий
            comment = issue.comments.create(text=text)
            
            logger.info(f"Комментарий успешно добавлен к задаче {issue_key}")
            
            # Возвращаем словарь с информацией о комментарии (для совместимости)
            return {
                "id": comment.id,
                "text": comment.text,
                "createdAt": str(comment.createdAt) if hasattr(comment, "createdAt") else None,
            }
            
        except NotFound as e:
            error_msg = f"Задача {issue_key} не найдена: {e}"
            logger.error(error_msg)
            raise NotFound(error_msg) from e
        except TrackerError as e:
            error_msg = f"Ошибка при добавлении комментария: {e}"
            logger.error(error_msg)
            raise TrackerError(error_msg) from e
        except Exception as e:
            error_msg = f"Неожиданная ошибка при добавлении комментария: {e}"
            logger.error(error_msg, exc_info=True)
            raise TrackerError(error_msg) from e


def create_tracker_client() -> Optional[YandexTrackerClient]:
    """
    Создаёт клиент Яндекс Трекера, если настроены необходимые переменные окружения.
    
    Поддерживает три режима:
    1. OAuth токен с cloud_org_id: YANDEX_TRACKER_TOKEN + YANDEX_TRACKER_CLOUD_ORG_ID
    2. OAuth токен с org_id: YANDEX_TRACKER_TOKEN + YANDEX_TRACKER_ORG_ID
    3. IAM токен с cloud_org_id: YANDEX_TRACKER_IAM_TOKEN + YANDEX_TRACKER_CLOUD_ORG_ID
    
    Returns:
        Optional[YandexTrackerClient]: Клиент или None, если не настроен
    """
    oauth_token = os.getenv("YANDEX_TRACKER_TOKEN")
    org_id = os.getenv("YANDEX_TRACKER_ORG_ID")
    iam_token = os.getenv("YANDEX_TRACKER_IAM_TOKEN")
    cloud_org_id = os.getenv("YANDEX_TRACKER_CLOUD_ORG_ID")
    
    # Проверяем режим IAM токена (приоритет)
    if iam_token and cloud_org_id:
        try:
            logger.debug("Создание клиента Яндекс Трекера с IAM токеном")
            return YandexTrackerClient()
        except Exception as e:
            logger.warning(f"Не удалось создать клиент Яндекс Трекера (IAM токен): {e}")
            return None
    # Проверяем OAuth токен с cloud_org_id
    elif oauth_token and cloud_org_id:
        try:
            logger.debug("Создание клиента Яндекс Трекера с OAuth токеном и cloud_org_id")
            return YandexTrackerClient()
        except Exception as e:
            logger.warning(f"Не удалось создать клиент Яндекс Трекера (OAuth + cloud_org_id): {e}")
            return None
    # Проверяем OAuth токен с org_id
    elif oauth_token and org_id:
        try:
            logger.debug("Создание клиента Яндекс Трекера с OAuth токеном и org_id")
            return YandexTrackerClient()
        except Exception as e:
            logger.warning(f"Не удалось создать клиент Яндекс Трекера (OAuth + org_id): {e}")
            return None
    else:
        logger.debug(
            "Яндекс Трекер не настроен. "
            "Для OAuth токена используйте YANDEX_TRACKER_TOKEN и YANDEX_TRACKER_CLOUD_ORG_ID (или YANDEX_TRACKER_ORG_ID). "
            "Для IAM токена используйте YANDEX_TRACKER_IAM_TOKEN и YANDEX_TRACKER_CLOUD_ORG_ID"
        )
        return None
