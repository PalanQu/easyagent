from collections.abc import Callable, Iterator, Sequence
import logging
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend, StoreBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
from fastapi.encoders import jsonable_encoder
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langchain.agents.middleware import InterruptOnConfig
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.cache.base import BaseCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from easyagent.models.schema.agent import AgentRunRequest, AgentRunResponse
from easyagent.models.schema.ag_ui import AGUIEvent
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


def _sanitize_namespace_component(value: object, *, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        return fallback
    sanitized = re.sub(r"[^A-Za-z0-9\-_.@+:~]", "_", text)
    return sanitized or fallback


def _extract_configurable(context: Any) -> dict[str, Any]:
    runtime = getattr(context, "runtime", None)
    config = getattr(runtime, "config", {}) if runtime is not None else {}
    if not isinstance(config, dict):
        return {}
    raw_configurable = config.get("configurable")
    if not isinstance(raw_configurable, dict):
        return {}
    return raw_configurable

class ClusterModeRuntimeFactory:
    """Factory for creating cluster mode runtime components."""

    def __init__(
        self,
        settings: Settings,
        *,
        sandbox: BackendProtocol | Callable[[Any], BackendProtocol] | None = None,
    ) -> None:
        self.settings = settings
        self.sandbox = sandbox
        self._pool: ConnectionPool | None = None

    def get_cleanup_callbacks(self) -> list[Callable[[], None]]:
        return [self.close]

    def create_runtime_kwargs(self) -> dict:
        if self.settings.local_mode:
            raise NotImplementedError("Cluster mode requires local_mode=False")
        if not self._is_postgres_url(self.settings.database_url):
            raise ValueError(
                "Cluster mode requires a Postgres `db_url` "
                "(e.g. postgresql://user:pass@host:5432/dbname)"
            )

        pool_conninfo = self._to_pool_conninfo(self.settings.database_url)
        pool = ConnectionPool(
            pool_conninfo,
            min_size=self.settings.cluster_pg_pool_min_size,
            max_size=self.settings.cluster_pg_pool_max_size,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        )

        checkpointer = PostgresSaver(pool)
        store = PostgresStore(pool)
        checkpointer.setup()
        store.setup()
        self._ensure_dirs()

        self._pool = pool
        return {
            "checkpointer": checkpointer,
            "store": store,
            "backend": self._create_backend_factory(),
        }

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def _create_backend_factory(self) -> Callable[[Any], BackendProtocol]:
        def _factory(runtime: Any) -> BackendProtocol:
            default_backend = self._resolve_default_backend(runtime)
            return CompositeBackend(
                default=default_backend,
                routes={
                    "/memory/": StoreBackend(runtime=runtime, namespace=self._memory_namespace_factory),
                },
            )

        return _factory

    def _memory_namespace_factory(self, context: Any) -> tuple[str, ...]:
        configurable = _extract_configurable(context)
        user_id = configurable.get("user_id")
        if user_id is None or not str(user_id).strip():
            return ("memory", "global")
        user_component = _sanitize_namespace_component(user_id, fallback="anonymous")
        return ("memory", user_component)

    def _resolve_default_backend(self, runtime: Any) -> BackendProtocol:
        if self.sandbox is not None:
            if callable(self.sandbox):
                return self.sandbox(runtime)
            return self.sandbox
        return LocalShellBackend(
            root_dir=self.settings.base_path,
            virtual_mode=True,
            inherit_env=True,
        )

    def _ensure_dirs(self) -> None:
        for path in (
            self.settings.base_path,
            self.settings.skills_path,
            self.settings.tmp_path,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _is_postgres_url(url: str) -> bool:
        return (
            url.startswith("postgres://")
            or url.startswith("postgresql://")
            or url.startswith("postgres+psycopg://")
            or url.startswith("postgresql+psycopg://")
        )

    @staticmethod
    def _to_pool_conninfo(url: str) -> str:
        if url.startswith("postgresql+psycopg://"):
            return "postgresql://" + url[len("postgresql+psycopg://") :]
        if url.startswith("postgres+psycopg://"):
            return "postgres://" + url[len("postgres+psycopg://") :]
        return url


class LocalModeRuntimeFactory:
    """Factory for creating local mode runtime components."""

    def __init__(
        self,
        settings: Settings,
        *,
        sandbox: BackendProtocol | Callable[[Any], BackendProtocol] | None = None,
    ) -> None:
        self.settings = settings
        self.sandbox = sandbox

    def get_cleanup_callbacks(self) -> list[Callable[[], None]]:
        return []

    def create_runtime_kwargs(self) -> dict:
        if not self.settings.local_mode:
            raise NotImplementedError(
                "Only local mode is supported now. Please set local_mode=True"
            )

        self._ensure_dirs()
        return {
            "checkpointer": InMemorySaver(),
            "store": InMemoryStore(),
            "backend": self._create_backend_factory(),
        }

    def _ensure_dirs(self) -> None:
        for path in (
            self.settings.base_path,
            self.settings.skills_path,
            self.settings.memories_path,
            self.settings.tmp_path,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)

    def _create_backend_factory(self) -> Callable[[Any], BackendProtocol]:
        def _factory(runtime: Any) -> BackendProtocol:
            default_backend = self._resolve_default_backend(runtime)
            memory_backend = self._resolve_memory_backend(runtime)
            return CompositeBackend(
                default=default_backend,
                routes={
                    "/memory/": memory_backend,
                },
            )

        return _factory

    def _resolve_memory_backend(self, runtime: Any) -> BackendProtocol:
        configurable = runtime.config.get("configurable", {}) if hasattr(runtime, "config") else {}
        user_id = configurable.get("user_id") if isinstance(configurable, dict) else None
        if user_id is None or not str(user_id).strip():
            root_dir = self.settings.memories_path
        else:
            user_component = _sanitize_namespace_component(user_id, fallback="anonymous")
            root_dir = self.settings.memories_path / user_component
        root_dir.mkdir(parents=True, exist_ok=True)
        return FilesystemBackend(root_dir=root_dir, virtual_mode=True)

    def _resolve_default_backend(self, runtime: Any) -> BackendProtocol:
        if self.sandbox is not None:
            if callable(self.sandbox):
                return self.sandbox(runtime)
            return self.sandbox
        return LocalShellBackend(
            root_dir=self.settings.base_path,
            virtual_mode=True,
            inherit_env=True,
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
        sandbox: BackendProtocol | Callable[[Any], BackendProtocol] | None = None,
        model: BaseChatModel | None = None,
    ) -> None:
        self._cluster_mode = not settings.local_mode
        self._cleanup_callbacks: list[Callable[[], None]] = []

        # Build agent kwargs
        kwargs: dict[str, Any] = {
            "model": model or _create_chat_model(settings),
        }

        # Runtime (checkpointer, store, backend)
        runtime_factory: LocalModeRuntimeFactory | ClusterModeRuntimeFactory
        if settings.local_mode:
            runtime_factory = LocalModeRuntimeFactory(settings, sandbox=sandbox)
        else:
            runtime_factory = ClusterModeRuntimeFactory(settings, sandbox=sandbox)
        runtime_kwargs = runtime_factory.create_runtime_kwargs()
        kwargs.update(runtime_kwargs)
        self._cleanup_callbacks.extend(runtime_factory.get_cleanup_callbacks())
        if sandbox is not None and callable(sandbox):
            close_cb = getattr(sandbox, "close", None)
            if callable(close_cb):
                self._cleanup_callbacks.append(close_cb)

        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        if skills is not None:
            kwargs["skills"] = skills
        kwargs["memory"] = memory if memory is not None else ["/memory/AGENTS.md"]
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

        self._agent = create_deep_agent(**kwargs)

    def run(self, payload: AgentRunRequest) -> AgentRunResponse:
        """Execute the precompiled agent with the given runtime input."""
        invoke_input, config = self._build_invoke_input_and_config(payload)

        # Invoke the precompiled agent
        state = self._agent.invoke(invoke_input, config=config if config else None)
        final_output = self._extract_final_output(state)
        safe_state = self._safe_jsonable(state)
        return AgentRunResponse(final_output=final_output, state=safe_state)

    def run_stream(self, payload: AgentRunRequest) -> list[AGUIEvent]:
        """Collect AG-UI events for callers that need a non-streaming list."""
        return list(self.iter_ag_ui_events(payload))

    def iter_ag_ui_events(self, payload: AgentRunRequest) -> Iterator[AGUIEvent]:
        """
        Execute one run and yield AG-UI compatible events.

        When the compiled agent supports `.stream()`, token/tool/state events are
        emitted incrementally. Otherwise this falls back to a single-shot run.
        """
        run_id = payload.run_id or f"run_{uuid4().hex}"
        thread_id = payload.thread_id

        yield AGUIEvent(type="RUN_STARTED", runId=run_id, threadId=thread_id)

        if not hasattr(self._agent, "stream"):
            yield from self._emit_non_streaming_events(payload, run_id=run_id, thread_id=thread_id)
            return

        try:
            invoke_input, config = self._build_invoke_input_and_config(payload)
            current_message_id: str | None = None
            assistant_message_started = False
            open_tool_call_ids: set[str] = set()
            final_state: dict[str, Any] | None = None

            stream = self._agent.stream(
                invoke_input,
                config=config if config else None,
                stream_mode=["messages", "updates", "custom", "values"],
                subgraphs=True,
            )
            for stream_item in stream:
                namespace, mode, chunk = self._unpack_stream_item(stream_item)
                if mode == "messages":
                    token, metadata = chunk
                    if not assistant_message_started:
                        current_message_id = self._resolve_message_id(metadata) or f"msg_{uuid4().hex}"
                        yield AGUIEvent(
                            type="TEXT_MESSAGE_START",
                            runId=run_id,
                            threadId=thread_id,
                            messageId=current_message_id,
                        )
                        assistant_message_started = True

                    text_delta = self._extract_text_delta(token)
                    if text_delta:
                        yield AGUIEvent(
                            type="TEXT_MESSAGE_CONTENT",
                            runId=run_id,
                            threadId=thread_id,
                            messageId=current_message_id,
                            delta=text_delta,
                        )

                    for tool_event in self._extract_tool_call_events(
                        token=token,
                        run_id=run_id,
                        thread_id=thread_id,
                        open_tool_call_ids=open_tool_call_ids,
                    ):
                        yield tool_event
                    continue

                if mode == "updates":
                    delta = chunk if isinstance(chunk, dict) else {"value": chunk}
                    yield AGUIEvent(
                        type="STATE_DELTA",
                        runId=run_id,
                        threadId=thread_id,
                        state={"namespace": namespace, "delta": delta},
                    )
                    continue

                if mode == "values":
                    if isinstance(chunk, dict):
                        final_state = chunk
                    continue

                if mode == "custom":
                    custom_value = chunk if isinstance(chunk, dict) else {"value": chunk}
                    event_name = "a2ui" if self._looks_like_a2ui_payload(custom_value) else "custom"
                    yield AGUIEvent(
                        type="CUSTOM",
                        runId=run_id,
                        threadId=thread_id,
                        name=event_name,
                        value={"namespace": namespace, "payload": custom_value},
                    )
                    continue

            if current_message_id:
                yield AGUIEvent(
                    type="TEXT_MESSAGE_END",
                    runId=run_id,
                    threadId=thread_id,
                    messageId=current_message_id,
                )

            for tool_call_id in list(open_tool_call_ids):
                yield AGUIEvent(
                    type="TOOL_CALL_END",
                    runId=run_id,
                    threadId=thread_id,
                    toolCallId=tool_call_id,
                )

            if final_state is not None:
                yield AGUIEvent(
                    type="STATE_SNAPSHOT",
                    runId=run_id,
                    threadId=thread_id,
                    state=final_state,
                )
                if "a2ui" in final_state:
                    yield AGUIEvent(
                        type="CUSTOM",
                        runId=run_id,
                        threadId=thread_id,
                        name="a2ui",
                        value=final_state["a2ui"],
                    )

            yield AGUIEvent(type="RUN_FINISHED", runId=run_id, threadId=thread_id)
        except Exception as exc:  # noqa: BLE001
            yield AGUIEvent(
                type="RUN_ERROR",
                runId=run_id,
                threadId=thread_id,
                error=str(exc),
            )

    def close(self) -> None:
        for callback in reversed(self._cleanup_callbacks):
            try:
                callback()
            except Exception:  # noqa: BLE001
                logger.exception("failed to cleanup runner runtime resource")

    def _initialize_langfuse(self) -> bool:
        if not os.getenv("LANGFUSE_BASE_URL"):
            return False

        get_client()
        return True

    def _build_langfuse_handler(self) -> BaseCallbackHandler:
        return CallbackHandler()

    def _build_invoke_input_and_config(self, payload: AgentRunRequest) -> tuple[dict, dict]:
        if self._cluster_mode and not (payload.thread_id and payload.thread_id.strip()):
            raise ValueError("`thread_id` is required in cluster mode for persistent checkpointing")

        invoke_input: dict
        if payload.invoke_input is not None:
            invoke_input = dict(payload.invoke_input)
        else:
            if not payload.input.strip():
                raise ValueError("`input` is required when `invoke_input` is not provided")
            invoke_input = {"messages": [{"role": "user", "content": payload.input}]}

        if payload.files and "files" not in invoke_input:
            invoke_input["files"] = payload.files

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
        return invoke_input, config

    def _emit_non_streaming_events(
        self,
        payload: AgentRunRequest,
        *,
        run_id: str,
        thread_id: str | None,
    ) -> Iterator[AGUIEvent]:
        try:
            response = self.run(payload)
            message_id = f"msg_{uuid4().hex}"
            yield AGUIEvent(type="TEXT_MESSAGE_START", runId=run_id, threadId=thread_id, messageId=message_id)
            if response.final_output:
                yield AGUIEvent(
                    type="TEXT_MESSAGE_CONTENT",
                    runId=run_id,
                    threadId=thread_id,
                    messageId=message_id,
                    delta=response.final_output,
                )
            yield AGUIEvent(type="TEXT_MESSAGE_END", runId=run_id, threadId=thread_id, messageId=message_id)
            yield AGUIEvent(type="STATE_SNAPSHOT", runId=run_id, threadId=thread_id, state=response.state)
            if "a2ui" in response.state:
                yield AGUIEvent(
                    type="CUSTOM",
                    runId=run_id,
                    threadId=thread_id,
                    name="a2ui",
                    value=response.state["a2ui"],
                )
            yield AGUIEvent(type="RUN_FINISHED", runId=run_id, threadId=thread_id)
        except Exception as exc:  # noqa: BLE001
            yield AGUIEvent(type="RUN_ERROR", runId=run_id, threadId=thread_id, error=str(exc))

    def _unpack_stream_item(self, stream_item: object) -> tuple[str, str, Any]:
        namespace = ""
        payload = stream_item
        if isinstance(stream_item, tuple) and len(stream_item) == 2:
            namespace = str(stream_item[0])
            payload = stream_item[1]

        if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], str):
            return namespace, payload[0], payload[1]
        return namespace, "custom", payload

    def _resolve_message_id(self, metadata: object) -> str | None:
        if not isinstance(metadata, dict):
            return None
        for key in ("message_id", "messageId", "id"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    def _extract_text_delta(self, token: object) -> str | None:
        content = getattr(token, "content", None)
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            joined = "".join(parts)
            return joined if joined else None
        return None

    def _extract_tool_call_events(
        self,
        *,
        token: object,
        run_id: str,
        thread_id: str | None,
        open_tool_call_ids: set[str],
    ) -> list[AGUIEvent]:
        events: list[AGUIEvent] = []
        tool_chunks = getattr(token, "tool_call_chunks", None)
        if isinstance(tool_chunks, list):
            for chunk in tool_chunks:
                if isinstance(chunk, dict):
                    tool_call_id = chunk.get("id")
                    name = chunk.get("name")
                    args = chunk.get("args")
                else:
                    tool_call_id = getattr(chunk, "id", None)
                    name = getattr(chunk, "name", None)
                    args = getattr(chunk, "args", None)
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    continue
                if tool_call_id not in open_tool_call_ids:
                    events.append(
                        AGUIEvent(
                            type="TOOL_CALL_START",
                            runId=run_id,
                            threadId=thread_id,
                            toolCallId=tool_call_id,
                            toolCallName=name if isinstance(name, str) else None,
                        )
                    )
                    open_tool_call_ids.add(tool_call_id)
                if isinstance(args, str) and args:
                    events.append(
                        AGUIEvent(
                            type="TOOL_CALL_ARGS",
                            runId=run_id,
                            threadId=thread_id,
                            toolCallId=tool_call_id,
                            args=args,
                        )
                    )

        token_type = getattr(token, "type", None)
        if token_type == "tool":
            tool_call_id = getattr(token, "tool_call_id", None)
            if isinstance(tool_call_id, str) and tool_call_id:
                events.append(
                    AGUIEvent(
                        type="TOOL_CALL_RESULT",
                        runId=run_id,
                        threadId=thread_id,
                        toolCallId=tool_call_id,
                        result=getattr(token, "content", None),
                    )
                )
                if tool_call_id in open_tool_call_ids:
                    events.append(
                        AGUIEvent(
                            type="TOOL_CALL_END",
                            runId=run_id,
                            threadId=thread_id,
                            toolCallId=tool_call_id,
                        )
                    )
                    open_tool_call_ids.remove(tool_call_id)
        return events

    def _looks_like_a2ui_payload(self, value: dict[str, Any]) -> bool:
        marker = value.get("kind")
        if marker == "a2ui":
            return True
        mime = value.get("mimeType")
        if isinstance(mime, str) and "a2ui" in mime.lower():
            return True
        if "surface" in value and isinstance(value.get("surface"), dict):
            return True
        if "component" in value and isinstance(value.get("component"), dict):
            return True
        return False

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
