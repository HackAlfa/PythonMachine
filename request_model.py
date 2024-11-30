
from pydantic import BaseModel, Field
from typing import List, Optional

class Signatures(BaseModel):
    common: dict[str, int]
    special: dict[str, int]

class RequestModel(BaseModel):
    clientId: str = Field(..., description="ИД пользователя")
    organizationId: str = Field(..., description="ИД организации")
    segment: str = Field(..., description="Сегмент организации: \"Малый бизнес\", \"Средний бизнес\", \"Крупный бизнес\"")
    role: str = Field(..., description="Роль уполномоченного лица: \"ЕИО\", \"Сотрудник\"")
    organizations: int = Field(..., ge=1, le=300, description="Общее количество организаций у уполномоченного лица")
    currentMethod: str = Field(..., description="Действующий способ подписания: \"SMS\", \"PayControl\", \"КЭП на токене\", \"КЭП в приложении\"")
    mobileApp: bool = Field(..., description="Наличие мобильного приложения")
    signatures: Signatures = Field(..., description="Подписанные ранее типы документов")
    availableMethods: List[str] = Field(..., description="Уже подключенные способы подписания")
    claims: int = Field(..., description="Наличие обращений в банк по причине проблем с использованием СМС")
    context: str = Field(..., description="Контекст")
