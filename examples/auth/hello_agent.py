import uvicorn
from dotenv import load_dotenv
from fastapi import HTTPException, Request

from easyagent.auth import AuthUser, CallableAuthProvider
from easyagent.sdk import EasyagentSDK
from easyagent.utils.settings import Settings

# curl -X POST "http://127.0.0.1:8000/agent/run" \
#  -H "Content-Type: application/json" \
#  -H "X-User-ID: user_001" \
#  -H "X-User-Name: Alice" \
#  -H "X-User-Email: alice@example.com" \
#  -d '{
#    "input": "hi","thread_id": "thread_001"
#  }'


async def authenticate_user(request: Request) -> AuthUser:
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user identification")

    return AuthUser(
        user_id=user_id,
        user_name=request.headers.get("X-User-Name"),
        email=request.headers.get("X-User-Email"),
    )


load_dotenv(override=False)
settings = Settings.from_env()

auth_provider = CallableAuthProvider(authenticate_user)
sdk = EasyagentSDK(
    settings=settings,
    system_prompt="You are a helpful assistant.",
    auth_provider=auth_provider,
)
app = sdk.create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_config=None)
