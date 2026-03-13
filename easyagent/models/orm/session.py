from typing import (
    List, TYPE_CHECKING, Optional, Dict, Any
)

from datetime import datetime, UTC

from sqlmodel import (
    SQLModel, Field, Relationship, JSON
)

if TYPE_CHECKING:
    from .user import User

class Session(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    thread_id: Optional[str] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    session_context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        sa_type=JSON,
        description="A JSON field to store session context data"
    )
    user: "User" = Relationship(back_populates="sessions")
