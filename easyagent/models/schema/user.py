from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UserCreate(BaseModel):
    external_user_id: str
    user_name: str | None = None
    email: str | None = None
    user_context: dict[str, Any] = Field(default_factory=dict)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    external_user_id: str
    user_name: str | None
    email: str | None
    created_at: datetime
    user_context: dict[str, Any]
