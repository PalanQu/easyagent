from collections.abc import Callable, Generator, Sequence
import logging
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from sqlmodel import Session

from easyagent.adapters.fastapi import build_easyagent_router
from easyagent.adapters.fastapi.middleware import build_logging_context_middleware
from easyagent.agent.agent import DeepAgentRunner
from easyagent.auth import AuthProvider, AuthUser, NoopAuthProvider
from easyagent.repos.factory import build_session_repo, build_user_repo
from easyagent.services.session_service import SessionService
from easyagent.services.user_service import UserService
from easyagent.utils.db import Database
from easyagent.utils.logging import setup_logging
from easyagent.utils.settings import Settings

from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
from langchain.agents.middleware import InterruptOnConfig
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache

__all__ = [
    "EasyagentSDK",
    "Settings",
    "AuthProvider",
    "AuthUser",
    "SubAgent",
    "CompiledSubAgent",
    "AgentMiddleware",
    "ResponseFormat",
    "InterruptOnConfig",
    "BaseTool",
    "BaseCache",
]

logger = logging.getLogger(__name__)


class EasyagentSDK:
    """
    Example:
        settings = Settings(
            model_key="sk-xxx",
            model_base_url="https://api.deepseek.com/v1",
            model_name="deepseek-chat",
        )
        sdk = EasyagentSDK(settings, system_prompt="You are a helpful assistant.")
        app = sdk.create_app()

    Advanced Example with custom tools and subagents:
        from langchain_core.tools import tool

        @tool
        def my_tool(query: str) -> str:
            '''My custom tool.'''
            return f"Result for {query}"

        sdk = EasyagentSDK(
            settings,
            system_prompt="You are a helpful assistant.",
            tools=[my_tool],
            subagents=[
                {
                    "name": "researcher",
                    "description": "A subagent that specializes in research",
                    "prompt": "You are a research expert.",
                }
            ],
            interrupt_on={"write_file": True},  # Pause before writing files
        )
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
        auth_provider: AuthProvider | None = None,
        title: str = "Easyagent API",
        version: str = "0.1.0",
    ) -> None:
        """
        Initialize the Easyagent SDK.

        Args:
            settings: SDK configuration, including model API key, base URL, and related options.
            system_prompt: Custom system prompt.
            skills: List of skill resource paths (e.g., ["/skills/user/", "/skills/project/"]).
            memory: List of memory file paths (e.g., ["/memory/AGENTS.md"]).
            tools: Custom tools list; each item can be a BaseTool, callable, or dict.
            middleware: Additional middleware sequence, applied after the standard middleware stack.
            subagents: List of subagents, each containing fields such as name, description, and prompt.
            response_format: Structured output response format.
            context_schema: Context schema for the deep agent.
            interrupt_on: Mapping of tool names to interrupt configurations, used to pause execution on specific tool calls for human approval.
            cache: Cache used by the agent.
            auth_provider: Custom authentication provider used to resolve user identity for tenant isolation.
            title: FastAPI application title.
            version: FastAPI application version.
        """
        setup_logging()
        self.settings = settings
        self.database = Database(settings)
        self.database.create_tables()
        logger.info(
            "EasyagentSDK initialized"
        )

        self.title = title
        self.version = version
        self.auth_provider = auth_provider or NoopAuthProvider()

        self.agent_runner = DeepAgentRunner(
            settings,
            system_prompt=system_prompt,
            skills=skills,
            memory=memory,
            tools=tools,
            middleware=middleware,
            subagents=subagents,
            response_format=response_format,
            context_schema=context_schema,
            interrupt_on=interrupt_on,
            cache=cache,
        )

        self._router: APIRouter | None = None

    @property
    def _db_backend(self) -> str:
        return "sqlite" if self.settings.local_mode else "postgres"

    def _db_session(self) -> Generator[Session, None, None]:
        yield from self.database.session()

    def _build_user_service(self, db_session: Session) -> UserService:
        user_repo = build_user_repo(db_session, self._db_backend)
        return UserService(user_repo)

    def _build_session_service(self, db_session: Session) -> SessionService:
        user_repo = build_user_repo(db_session, self._db_backend)
        session_repo = build_session_repo(db_session, self._db_backend)
        return SessionService(session_repo=session_repo, user_repo=user_repo)

    async def get_current_user(self, request: Request) -> AuthUser:
        return await self.auth_provider.authenticate(request)

    def router(self) -> APIRouter:
        if self._router is not None:
            return self._router
        self._router = build_easyagent_router(self)
        return self._router

    def mount_fastapi(self, app: FastAPI, prefix: str = "") -> None:
        app.include_router(self.router(), prefix=prefix)

    def create_app(self, prefix: str = "") -> FastAPI:
        app = FastAPI(title=self.title, version=self.version)
        app.middleware("http")(build_logging_context_middleware(self.get_current_user))
        self.mount_fastapi(app, prefix=prefix)
        return app
