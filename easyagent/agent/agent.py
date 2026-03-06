from collections.abc import Callable, Sequence
import logging
import os
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
from fastapi.encoders import jsonable_encoder
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langchain.agents.middleware import InterruptOnConfig
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.cache.base import BaseCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from easyagent.models.schema.agent import AgentRunRequest, AgentRunResponse
from easyagent.utils.settings import Settings


logger = logging.getLogger(__name__)


class InvocationLifecycleLogger(BaseCallbackHandler):
    def __init__(self, *, user_id: str | None, thread_id: str | None) -> None:
        self._user_id = user_id
        self._thread_id = thread_id
        self._llm_run_name: dict[str, str] = {}
        self._tool_run_name: dict[str, str] = {}

    def _extra(self) -> dict[str, str]:
        return {
            "user_id": self._user_id or "-",
            "thread_id": self._thread_id or "-",
        }

    def _resolve_name(self, serialized: dict[str, Any] | None, *, fallback: str) -> str:
        if not isinstance(serialized, dict):
            return fallback
        name = serialized.get("name")
        if isinstance(name, str) and name.strip():
            return name

        serialized_id = serialized.get("id")
        if isinstance(serialized_id, list):
            for candidate in reversed(serialized_id):
                if isinstance(candidate, str) and candidate.strip():
                    return candidate
        elif isinstance(serialized_id, str) and serialized_id.strip():
            return serialized_id

        return fallback

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        **kwargs: Any,
    ) -> Any:
        self.on_llm_start(serialized, prompts=[], **kwargs)

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> Any:
        llm_name = self._resolve_name(serialized, fallback="unknown_llm")
        run_id = kwargs.get("run_id")
        if run_id is not None:
            self._llm_run_name[str(run_id)] = llm_name
        logger.info("llm call started: %s", llm_name, extra=self._extra())

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        run_id = kwargs.get("run_id")
        llm_name = self._llm_run_name.pop(str(run_id), "unknown_llm") if run_id is not None else "unknown_llm"
        logger.info("llm call finished: %s", llm_name, extra=self._extra())

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        tool_name = self._resolve_name(serialized, fallback="unknown_tool")
        run_id = kwargs.get("run_id")
        if run_id is not None:
            self._tool_run_name[str(run_id)] = tool_name
        logger.info("tool call started: %s", tool_name, extra=self._extra())

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        run_id = kwargs.get("run_id")
        tool_name = self._tool_run_name.pop(str(run_id), "unknown_tool") if run_id is not None else "unknown_tool"
        logger.info("tool call finished: %s", tool_name, extra=self._extra())


def _create_chat_model(settings: Settings) -> ChatOpenAI:
    """Create ChatOpenAI model (supports any OpenAI API compatible service)."""
    return ChatOpenAI(
        api_key=settings.model_key,
        base_url=settings.model_base_url,
        model=settings.model_name,
    )


class LocalModeRuntimeFactory:
    """Factory for creating local mode runtime components."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def create_runtime_kwargs(self) -> dict:
        if not self.settings.local_mode:
            raise NotImplementedError(
                "Only local mode is supported now. Please set local_mode=True"
            )

        self._ensure_dirs()
        return {
            "checkpointer": InMemorySaver(),
            "store": InMemoryStore(),
            "backend": self._create_backend(),
        }

    def _ensure_dirs(self) -> None:
        for path in (
            self.settings.base_path,
            self.settings.skills_path,
            self.settings.memories_path,
            self.settings.tmp_path,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)

    def _create_backend(self) -> BackendProtocol:
        return CompositeBackend(
            default=LocalShellBackend(
                root_dir=self.settings.base_path,
                virtual_mode=True,
                inherit_env=True,
                timeout=30.0,
            ),
            routes={
                "/skills/": FilesystemBackend(root_dir=self.settings.skills_path, virtual_mode=True),
                "/memories/": FilesystemBackend(root_dir=self.settings.memories_path, virtual_mode=True),
                "/tmp/": FilesystemBackend(root_dir=self.settings.tmp_path, virtual_mode=True),
            },
        )


class DeepAgentRunner:
    """
    Precompiled agent runner. The agent is created once at initialization
    and reused for all subsequent run() calls.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        system_prompt: str | None = None,
        skills: list[str] | None = None,
        memory: list[str] | None = None,
        tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
        middleware: Sequence[AgentMiddleware] = (),
        subagents: list[SubAgent | CompiledSubAgent] | None = None,
        response_format: ResponseFormat | None = None,
        context_schema: type[Any] | None = None,
        interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
        cache: BaseCache | None = None,
    ) -> None:
        # Build agent kwargs
        kwargs: dict[str, Any] = {
            "model": _create_chat_model(settings),
        }

        # Local mode runtime (checkpointer, store, backend)
        if settings.local_mode:
            runtime_kwargs = LocalModeRuntimeFactory(settings).create_runtime_kwargs()
            kwargs.update(runtime_kwargs)

        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        if skills is not None:
            kwargs["skills"] = skills
        if memory is not None:
            kwargs["memory"] = memory
        if tools is not None:
            kwargs["tools"] = tools
        if middleware:
            kwargs["middleware"] = middleware
        if subagents is not None:
            kwargs["subagents"] = subagents
        if response_format is not None:
            kwargs["response_format"] = response_format
        if context_schema is not None:
            kwargs["context_schema"] = context_schema
        if interrupt_on is not None:
            kwargs["interrupt_on"] = interrupt_on
        if cache is not None:
            kwargs["cache"] = cache

        self._langfuse_enabled = self._initialize_langfuse()

        # Pre-create the agent (compiled once, reused for all requests)
        self._agent = create_deep_agent(**kwargs)

    def run(self, payload: AgentRunRequest) -> AgentRunResponse:
        """Execute the precompiled agent with the given runtime input."""
        # Build invoke input
        invoke_input: dict
        if payload.invoke_input is not None:
            invoke_input = dict(payload.invoke_input)
        else:
            if not payload.input.strip():
                raise ValueError("`input` is required when `invoke_input` is not provided")
            invoke_input = {"messages": [{"role": "user", "content": payload.input}]}

        if payload.files and "files" not in invoke_input:
            invoke_input["files"] = payload.files

        # Build invoke config
        config: dict = dict(payload.invoke_config or {})
        configurable: dict = dict(config.get("configurable") or {})
        if payload.thread_id:
            configurable["thread_id"] = payload.thread_id
        if payload.user_id:
            configurable["user_id"] = payload.user_id
        if configurable:
            config["configurable"] = configurable

        if self._langfuse_enabled:
            handler = self._build_langfuse_handler()
            if handler is not None:
                callbacks = list(config.get("callbacks") or [])
                callbacks.append(handler)
                config["callbacks"] = callbacks

                metadata = dict(config.get("metadata") or {})
                if payload.user_id and "langfuse_user_id" not in metadata:
                    metadata["langfuse_user_id"] = payload.user_id
                if payload.thread_id and "langfuse_session_id" not in metadata:
                    metadata["langfuse_session_id"] = payload.thread_id
                if metadata:
                    config["metadata"] = metadata

        callbacks = list(config.get("callbacks") or [])
        callbacks.append(
            InvocationLifecycleLogger(
                user_id=payload.user_id,
                thread_id=payload.thread_id,
            )
        )
        config["callbacks"] = callbacks

        # Invoke the precompiled agent
        state = self._agent.invoke(invoke_input, config=config if config else None)
        final_output = self._extract_final_output(state)
        safe_state = self._safe_jsonable(state)
        return AgentRunResponse(final_output=final_output, state=safe_state)

    def _initialize_langfuse(self) -> bool:
        if not os.getenv("LANGFUSE_BASE_URL"):
            return False

        get_client()
        return True

    def _build_langfuse_handler(self) -> object | None:
        return CallbackHandler()

    def _extract_final_output(self, state: object) -> str | None:
        if not isinstance(state, dict):
            return None
        messages = state.get("messages")
        if not isinstance(messages, list):
            return None

        for message in reversed(messages):
            content = None
            if isinstance(message, dict):
                role = message.get("role") or message.get("type")
                if role in {"assistant", "ai"}:
                    content = message.get("content")
            else:
                msg_type = getattr(message, "type", None)
                if msg_type == "ai":
                    content = getattr(message, "content", None)

            text = self._coerce_text(content)
            if text:
                return text
        return None

    def _coerce_text(self, content: object) -> str | None:
        if content is None:
            return None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    texts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        texts.append(text)
            joined = "\n".join(t for t in texts if t.strip())
            return joined or None
        return str(content)

    def _safe_jsonable(self, value: object) -> dict:
        encoded = jsonable_encoder(value)
        if isinstance(encoded, dict):
            return encoded
        return {"result": encoded}
