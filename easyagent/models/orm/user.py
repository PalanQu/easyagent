from typing import (
    List, TYPE_CHECKING, Optional, Dict, Any
)

from datetime import datetime, UTC

from sqlmodel import (
    SQLModel, Field, Relationship, JSON
)

if TYPE_CHECKING:
    from .session import Session


class User(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    external_user_id: str = Field(index=True, unique=True)
    user_name: Optional[str] = Field(default=None)
    email: Optional[str] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    user_context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        sa_type=JSON,
    )
    sessions: List["Session"] = Relationship(back_populates="user")
