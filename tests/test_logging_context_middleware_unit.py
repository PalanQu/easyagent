import unittest
from types import SimpleNamespace

from starlette.responses import Response

from easyagent.adapters.fastapi.middleware.logging_context import (
    _resolve_thread_id,
    _resolve_user_id,
    build_logging_context_middleware,
)
from easyagent.models.schema.auth import AuthUser


class _FakeRequest:
    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        json_body=None,  # noqa: ANN001
        json_error: Exception | None = None,
    ):
        self.headers = headers or {}
        self.state = SimpleNamespace()
        self._json_body = json_body
        self._json_error = json_error

    async def json(self):  # noqa: ANN201
        if self._json_error is not None:
            raise self._json_error
        return self._json_body


class TestLoggingContextMiddlewareUnit(unittest.IsolatedAsyncioTestCase):
    async def test_resolve_thread_id_happy_and_fallback(self) -> None:
        req_with_thread = _FakeRequest(json_body={"thread_id": 123})
        req_with_camel_thread = _FakeRequest(json_body={"threadId": "camel-thread"})
        req_bad_json = _FakeRequest(json_error=ValueError("bad json"))

        self.assertEqual(await _resolve_thread_id(req_with_thread), "123")
        self.assertEqual(await _resolve_thread_id(req_with_camel_thread), "camel-thread")
        self.assertEqual(await _resolve_thread_id(req_bad_json), "-")

    async def test_resolve_user_id_priority(self) -> None:
        request = _FakeRequest(headers={"X-User-ID": "header-user"})

        self.assertEqual(_resolve_user_id(request, None), "header-user")
        self.assertEqual(_resolve_user_id(request, AuthUser(user_id="auth-user")), "auth-user")
        self.assertEqual(_resolve_user_id(request, AuthUser(user_id="anonymous")), "header-user")
        self.assertEqual(_resolve_user_id(_FakeRequest(headers={}), AuthUser(user_id="anonymous")), "anonymous")

    async def test_middleware_sets_auth_user_and_log_context(self) -> None:
        request = _FakeRequest(headers={"X-User-ID": "header-user"}, json_body={"thread_id": "t-1"})

        async def _authenticate(_request):  # noqa: ANN001
            return AuthUser(user_id="auth-user")

        async def _call_next(_request):  # noqa: ANN001
            return Response("ok", status_code=200)

        middleware = build_logging_context_middleware(_authenticate)
        response = await middleware(request, _call_next)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(request.state.auth_user.user_id, "auth-user")
        self.assertEqual(request.state.log_context, {"user_id": "auth-user", "thread_id": "t-1"})

    async def test_middleware_swallow_auth_error_and_continue(self) -> None:
        request = _FakeRequest(headers={"X-User-ID": "header-user"}, json_body={"thread_id": "t-2"})

        async def _authenticate(_request):  # noqa: ANN001
            raise RuntimeError("auth failed")

        async def _call_next(_request):  # noqa: ANN001
            return Response("ok", status_code=200)

        middleware = build_logging_context_middleware(_authenticate)
        response = await middleware(request, _call_next)

        self.assertEqual(response.status_code, 200)
        self.assertFalse(hasattr(request.state, "auth_user"))
        self.assertEqual(request.state.log_context, {"user_id": "header-user", "thread_id": "t-2"})


if __name__ == "__main__":
    unittest.main()
