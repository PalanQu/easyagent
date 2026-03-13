from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SessionCreate(BaseModel):
    user_id: int
    thread_id: str | None = None
    session_context: dict[str, Any] = Field(default_factory=dict)


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    thread_id: str | None
    created_at: datetime
    session_context: dict[str, Any]


class SessionCreateForCurrentUser(BaseModel):
    thread_id: str | None = None
    session_context: dict[str, Any] = Field(default_factory=dict)


class SessionUpdate(BaseModel):
    session_context: dict[str, Any]


class SessionMessageOut(BaseModel):
    role: str
    content: str
