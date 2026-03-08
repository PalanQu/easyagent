from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


AGUIEventType = Literal[
    "RUN_STARTED",
    "RUN_FINISHED",
    "RUN_ERROR",
    "TEXT_MESSAGE_START",
    "TEXT_MESSAGE_CONTENT",
    "TEXT_MESSAGE_END",
    "TOOL_CALL_START",
    "TOOL_CALL_ARGS",
    "TOOL_CALL_END",
    "TOOL_CALL_RESULT",
    "STATE_DELTA",
    "STATE_SNAPSHOT",
    "CUSTOM",
]


class AGUIEvent(BaseModel):
    type: AGUIEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    runId: str
    threadId: str | None = None
    messageId: str | None = None
    toolCallId: str | None = None
    toolCallName: str | None = None
    delta: str | None = None
    args: str | None = None
    result: Any = None
    state: dict[str, Any] | None = None
    name: str | None = None
    value: Any = None
    error: str | None = None

