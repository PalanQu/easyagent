from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request

from easyagent.models.schema.auth import AuthProvider, AuthUser


class NoopAuthProvider(AuthProvider):
    def __init__(self, default_user_id: str = "anonymous"):
        self.default_user_id = default_user_id

    async def authenticate(self, request: Request) -> AuthUser:
        return AuthUser(user_id=self.default_user_id)


class HeaderAuthProvider(AuthProvider):
    def __init__(
        self,
        user_id_header: str = "X-User-ID",
        user_name_header: str = "X-User-Name",
        email_header: str = "X-User-Email",
        required: bool = True,
    ):
        self.user_id_header = user_id_header
        self.user_name_header = user_name_header
        self.email_header = email_header
        self.required = required

    async def authenticate(self, request: Request) -> AuthUser:
        user_id = request.headers.get(self.user_id_header)

        if not user_id:
            if self.required:
                raise HTTPException(status_code=401, detail="Missing user identification")
            user_id = "anonymous"

        return AuthUser(
            user_id=user_id,
            user_name=request.headers.get(self.user_name_header),
            email=request.headers.get(self.email_header),
        )


class CallableAuthProvider(AuthProvider):
    def __init__(
        self,
        auth_func: Callable[[Request], Awaitable[AuthUser]],
    ):
        self._auth_func = auth_func

    async def authenticate(self, request: Request) -> AuthUser:
        return await self._auth_func(request)