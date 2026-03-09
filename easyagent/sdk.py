from collections.abc import Callable, Generator, Sequence
from contextlib import asynccontextmanager
import logging
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from sqlmodel import Session

from easyagent.adapters.a2a.server import A2AServerConfig
from easyagent.adapters.fastapi import build_easyagent_router, mount_copilotkit_routes
from easyagent.adapters.fastapi.middleware import build_logging_context_middleware
from easyagent.agent.agent import DeepAgentRunner
from easyagent.agent.discovery import discover_subagents_from_gateway
from easyagent.adapters.a2a import mount_a2a_routes
from easyagent.agent.opensandbox import OpenSandboxBackend, OpenSandboxThreadBackendFactory
from easyagent.auth import AuthProvider, AuthUser, NoopAuthProvider
from easyagent.repos.factory import build_session_repo, build_user_repo
from easyagent.services.session_service import SessionService
from easyagent.services.user_service import UserService
from easyagent.utils.db import Database
from easyagent.utils.logging import setup_logging
from easyagent.utils.settings import Settings

from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
from deepagents.backends.protocol import BackendProtocol
from langchain.agents.middleware import InterruptOnConfig
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain_core.language_models import BaseChatModel
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
    "BackendProtocol",
    "OpenSandboxBackend",
    "OpenSandboxThreadBackendFactory",
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
        sandbox: BackendProtocol | Callable[[Any], BackendProtocol] | None = None,
        model: BaseChatModel | None = None,
        auth_provider: AuthProvider | None = None,
        a2a_enabled: bool = True,
        a2a_public_base_url: str = "http://127.0.0.1:8000",
        a2a_rpc_path: str = "/a2a",
        a2a_agent_name: str = "Easyagent",
        a2a_agent_description: str = "Easyagent compatible A2A agent.",
        a2a_gateway_url: str | None = None,
        a2a_gateway_agents_path: str = "/agents",
        a2a_gateway_timeout_seconds: float = 10.0,
        a2a_gateway_subagent_name_prefix: str = "remote_",
        a2a_gateway_fail_fast: bool = False,
        copilotkit_enabled: bool = True,
        copilotkit_path: str = "/copilotkit",
        copilotkit_agent_name: str = "easyagent",
        copilotkit_agent_description: str = "EasyAgent LangGraph endpoint for CopilotKit AG-UI.",
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
            sandbox: Optional sandbox backend (or backend factory). If provided, it is used as default backend.
            model: Optional injected chat model. If omitted, a ChatOpenAI model is created from `settings`.
            auth_provider: Custom authentication provider used to resolve user identity for tenant isolation.
            a2a_enabled: Whether to expose A2A protocol endpoints.
            a2a_public_base_url: Public base URL used to build AgentCard.url (e.g. "http://127.0.0.1:8000").
            a2a_rpc_path: JSON-RPC path for A2A endpoint, mounted on the same FastAPI app.
            a2a_agent_name: Agent name shown in A2A AgentCard.
            a2a_agent_description: Agent description shown in A2A AgentCard.
            a2a_gateway_url: Gateway base URL used to discover remote A2A agents.
            a2a_gateway_agents_path: Path on gateway used to list registered agent addresses.
            a2a_gateway_timeout_seconds: Timeout used when calling gateway and remote card endpoints.
            a2a_gateway_subagent_name_prefix: Prefix added to discovered subagent names.
            a2a_gateway_fail_fast: Raise initialization error when gateway discovery fails.
            copilotkit_enabled: Whether to expose a CopilotKit LangGraph AG-UI endpoint.
            copilotkit_path: Route prefix used for the CopilotKit endpoint.
            copilotkit_agent_name: Agent name shown to CopilotKit clients.
            copilotkit_agent_description: Agent description shown to CopilotKit clients.
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
        self.a2a_enabled = a2a_enabled
        self.copilotkit_enabled = copilotkit_enabled
        self.copilotkit_path = copilotkit_path
        self.copilotkit_agent_name = copilotkit_agent_name
        self.copilotkit_agent_description = copilotkit_agent_description
        self.a2a_config = A2AServerConfig(
            public_base_url=a2a_public_base_url,
            rpc_path=a2a_rpc_path,
            agent_name=a2a_agent_name,
            agent_description=a2a_agent_description,
            version=version,
        )

        resolved_subagents = list(subagents or [])
        if a2a_gateway_url:
            try:
                discovered = discover_subagents_from_gateway(
                    gateway_url=a2a_gateway_url,
                    agents_path=a2a_gateway_agents_path,
                    timeout_seconds=a2a_gateway_timeout_seconds,
                    name_prefix=a2a_gateway_subagent_name_prefix,
                )
                resolved_subagents.extend(discovered)
                logger.info(
                    "Discovered %d remote subagents from gateway %s",
                    len(discovered),
                    a2a_gateway_url,
                )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to discover remote subagents from gateway: %s", a2a_gateway_url)
                if a2a_gateway_fail_fast:
                    raise

        self.agent_runner = DeepAgentRunner(
            settings,
            system_prompt=system_prompt,
            skills=skills,
            memory=memory,
            tools=tools,
            middleware=middleware,
            subagents=resolved_subagents if resolved_subagents else None,
            response_format=response_format,
            context_schema=context_schema,
            interrupt_on=interrupt_on,
            cache=cache,
            sandbox=sandbox,
            model=model,
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
        if self.a2a_enabled:
            mount_a2a_routes(app=app, runner=self.agent_runner, config=self.a2a_config)
        if self.copilotkit_enabled:
            mount_copilotkit_routes(
                app=app,
                graph=self.agent_runner._agent,
                path=self.copilotkit_path,
                name=self.copilotkit_agent_name,
                description=self.copilotkit_agent_description,
            )

    def create_app(self, prefix: str = "") -> FastAPI:
        @asynccontextmanager
        async def lifespan(_: FastAPI):
            try:
                yield
            finally:
                self.agent_runner.close()

        app = FastAPI(title=self.title, version=self.version, lifespan=lifespan)
        app.middleware("http")(build_logging_context_middleware(self.get_current_user))
        self.mount_fastapi(app, prefix=prefix)
        return app
