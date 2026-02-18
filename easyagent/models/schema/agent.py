from typing import Any

from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    """Request schema for agent invocation (runtime parameters only)."""

    input: str = Field(default="", description="User input message.")
    files: dict[str, str] | None = Field(default=None, description="Optional virtual files for StateBackend.")
    thread_id: str | None = Field(default=None, description="Optional thread id for checkpointing.")
    user_id: str | None = Field(default=None, description="Optional user id in graph configurable.")
    invoke_input: dict[str, Any] | None = Field(
        default=None,
        description="Optional raw invoke input. If set, takes precedence over `input`.",
    )
    invoke_config: dict[str, Any] | None = Field(
        default=None,
        description="Optional raw invoke config merged with thread/user configurable.",
    )


class AgentRunResponse(BaseModel):
    final_output: str | None = Field(default=None, description="Extracted final assistant text.")
    state: dict[str, Any] = Field(default_factory=dict, description="Raw graph state (JSON-safe).")
