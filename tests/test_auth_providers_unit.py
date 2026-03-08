import unittest
from types import SimpleNamespace

from fastapi import HTTPException

from easyagent.auth.providers import CallableAuthProvider, HeaderAuthProvider
from easyagent.models.schema.auth import AuthUser


class _FakeRequest:
    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}
        self.state = SimpleNamespace()


class TestAuthProvidersUnit(unittest.IsolatedAsyncioTestCase):
    async def test_header_auth_provider_required_true_raises(self) -> None:
        provider = HeaderAuthProvider(required=True)
        request = _FakeRequest(headers={})

        with self.assertRaises(HTTPException) as ctx:
            await provider.authenticate(request)
        self.assertEqual(ctx.exception.status_code, 401)

    async def test_header_auth_provider_required_false_uses_anonymous(self) -> None:
        provider = HeaderAuthProvider(required=False)
        request = _FakeRequest(headers={})

        user = await provider.authenticate(request)
        self.assertEqual(user.user_id, "anonymous")

    async def test_header_auth_provider_supports_custom_header_names(self) -> None:
        provider = HeaderAuthProvider(
            user_id_header="X-Custom-User",
            user_name_header="X-Custom-Name",
            email_header="X-Custom-Email",
            required=True,
        )
        request = _FakeRequest(
            headers={
                "X-Custom-User": "u-1",
                "X-Custom-Name": "Alice",
                "X-Custom-Email": "alice@example.com",
            }
        )

        user = await provider.authenticate(request)
        self.assertEqual(user.user_id, "u-1")
        self.assertEqual(user.user_name, "Alice")
        self.assertEqual(user.email, "alice@example.com")

    async def test_callable_auth_provider_passthrough(self) -> None:
        request = _FakeRequest(headers={"X-User-ID": "u-2"})

        async def _auth(req):  # noqa: ANN001
            self.assertIs(req, request)
            return AuthUser(user_id="u-2", user_name="Bob")

        provider = CallableAuthProvider(_auth)
        user = await provider.authenticate(request)
        self.assertEqual(user.user_id, "u-2")
        self.assertEqual(user.user_name, "Bob")


if __name__ == "__main__":
    unittest.main()
