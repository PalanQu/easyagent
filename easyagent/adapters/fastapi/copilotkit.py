from collections.abc import Awaitable, Callable
import warnings

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langgraph.graph.state import CompiledStateGraph

from easyagent.models.schema.auth import AuthUser

# Suppress noisy third-party schema warnings emitted by ag_ui/cpk with pydantic v2.
warnings.filterwarnings(
    "ignore",
    message=r"The '.*' attribute with value .* was provided to the `Field\(\)` function, which has no effect.*",
    module=r"pydantic\._internal\._generate_schema",
)


def mount_copilotkit_routes(
    app: FastAPI,
    *,
    graph: CompiledStateGraph,
    path: str,
    name: str,
    description: str,
    authenticate: Callable[[Request], Awaitable[AuthUser]],
    ensure_thread_session: Callable[[AuthUser, str], object] | None = None,
) -> None:
    try:
        # Some versions of ag-ui/cpk model schemas emit noisy Pydantic
        # UnsupportedFieldAttributeWarning (alias on union internals).
        # Keep logs clean while importing these third-party modules.
        from pydantic.warnings import UnsupportedFieldAttributeWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnsupportedFieldAttributeWarning)
            from ag_ui.core.types import RunAgentInput
            from ag_ui.encoder import EventEncoder
            from copilotkit import LangGraphAGUIAgent
    except ImportError as exc:
        raise RuntimeError(
            "CopilotKit AG-UI integration requires both `copilotkit` and `ag-ui-langgraph` to be installed."
        ) from exc

    async def _authenticate_request(request: Request) -> AuthUser:
        cached_user = getattr(request.state, "auth_user", None)
        if isinstance(cached_user, AuthUser):
            return cached_user
        return await authenticate(request)

    @app.post(path)
    async def copilotkit_endpoint(input_data: RunAgentInput, request: Request):
        current_user = await _authenticate_request(request)
        if ensure_thread_session and input_data.thread_id:
            ensure_thread_session(current_user, input_data.thread_id)
        accept_header = request.headers.get("accept")
        encoder = EventEncoder(accept=accept_header)

        config: dict[str, object] = {
            "configurable": {
                "user_id": current_user.user_id,
            }
        }
        if input_data.thread_id:
            config["metadata"] = {
                "langfuse_user_id": current_user.user_id,
                "langfuse_session_id": input_data.thread_id,
            }

        agent = LangGraphAGUIAgent(
            name=name,
            description=description,
            graph=graph,
            config=config,
        )

        async def event_generator():
            async for event in agent.run(input_data):
                yield encoder.encode(event)

        return StreamingResponse(
            event_generator(),
            media_type=encoder.get_content_type(),
        )

    @app.get(f"{path}/health")
    def copilotkit_health() -> dict[str, object]:
        return {
            "status": "ok",
            "agent": {
                "name": name,
            },
        }
