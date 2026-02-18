from abc import ABC, abstractmethod
from typing import Any

from fastapi import Request
from pydantic import BaseModel


class AuthUser(BaseModel):
    user_id: str
    user_name: str | None = None
    email: str | None = None
    metadata: dict[str, Any] = {}


class AuthProvider(ABC):
    @abstractmethod
    async def authenticate(self, request: Request) -> AuthUser:
        ...
