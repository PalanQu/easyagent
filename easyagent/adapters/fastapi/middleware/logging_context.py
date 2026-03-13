from collections.abc import Awaitable, Callable

from fastapi import Request

from easyagent.models.schema.auth import AuthUser
from easyagent.utils.logging import get_request_logger, set_request_log_context


async def _resolve_thread_id(request: Request) -> str:
    try:
        body = await request.json()
    except Exception:
        return "-"

    if isinstance(body, dict):
        thread_id = body.get("thread_id") or body.get("threadId")
        if thread_id:
            return str(thread_id)

    return "-"


def _resolve_user_id(request: Request, auth_user: AuthUser | None) -> str:
    header_user_id = request.headers.get("X-User-ID")
    if auth_user is None:
        return header_user_id or "-"

    if auth_user.user_id and auth_user.user_id != "anonymous":
        return auth_user.user_id

    return header_user_id or auth_user.user_id or "-"


def build_logging_context_middleware(
    authenticate: Callable[[Request], Awaitable[AuthUser]],
):
    async def logging_context_middleware(request: Request, call_next):
        thread_id = await _resolve_thread_id(request)
        user_id = "-"
        auth_user: AuthUser | None = None

        try:
            auth_user = await authenticate(request)
            request.state.auth_user = auth_user
        except Exception:
            pass
        user_id = _resolve_user_id(request, auth_user)
        set_request_log_context(request, user_id=user_id, thread_id=thread_id)
        request_logger = get_request_logger(request, __name__)

        request_logger.debug("request logging context initialized")
        return await call_next(request)

    return logging_context_middleware
