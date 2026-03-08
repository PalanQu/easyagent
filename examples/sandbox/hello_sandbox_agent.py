import os

import uvicorn
from dotenv import load_dotenv

from easyagent.sdk import EasyagentSDK, OpenSandboxBackend
from easyagent.utils.settings import Settings


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


load_dotenv(override=False)
settings = Settings.from_env()

sandbox = OpenSandboxBackend.create(
    image=os.getenv("OPEN_SANDBOX_IMAGE", "python:3.13"),
    api_key=os.getenv("OPEN_SANDBOX_API_KEY"),
    domain=os.getenv("OPEN_SANDBOX_DOMAIN"),
    protocol=os.getenv("OPEN_SANDBOX_PROTOCOL", "http"),
    use_server_proxy=_parse_bool(os.getenv("OPEN_SANDBOX_USE_SERVER_PROXY"), True),
)

sdk = EasyagentSDK(
    settings=settings,
    system_prompt="You are a helpful assistant.",
    sandbox=sandbox,
)
app = sdk.create_app()


if __name__ == "__main__":
    # Request example:
    # curl -X POST "http://127.0.0.1:8000/agent/run" \
    #   -H "Content-Type: application/json" \
    #   -d '{"input":"list files","thread_id":"thread_001","user_id":"user_001"}'
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_config=None)
