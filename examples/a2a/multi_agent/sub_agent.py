import json
import os
from urllib import error as urllib_error
from urllib import request as urllib_request

import uvicorn
from dotenv import load_dotenv
from langchain_core.tools import tool

from easyagent.sdk import EasyagentSDK, Settings


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("division by zero is not allowed")
    return a / b


def register_to_gateway(gateway_url: str, agent_base_url: str) -> None:
    payload = {"url": agent_base_url}
    req = urllib_request.Request(
        f"{gateway_url.rstrip('/')}/agents",
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib_request.urlopen(req, timeout=10) as response:
        response.read()


# curl -X POST "http://127.0.0.1:8101/agent/run" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "input": "calculate 23.5 + 18.2 - 10",
#     "thread_id": "thread_sub_001"
#   }'


load_dotenv(override=False)
settings = Settings.from_env()

host = os.getenv("SUB_AGENT_HOST", "127.0.0.1")
port = int(os.getenv("SUB_AGENT_PORT", "8101"))
public_base_url = os.getenv("SUB_AGENT_PUBLIC_BASE_URL", f"http://{host}:{port}")
gateway_url = os.getenv("GATEWAY_URL", "http://127.0.0.1:8010")
register_gateway = os.getenv("REGISTER_TO_GATEWAY", "true").lower() == "true"

sdk = EasyagentSDK(
    settings=settings,
    system_prompt=(
        "You are the math subagent and only handle arithmetic calculations. "
        "Use these tools for calculation: add, subtract, multiply, divide. "
        "If a request is outside arithmetic, clearly state your limitations."
    ),
    tools=[add, subtract, multiply, divide],
    a2a_public_base_url=public_base_url,
    a2a_agent_name="remote_math_agent",
    a2a_agent_description="Math agent exposed by A2A for gateway discovery.",
)
app = sdk.create_app()


if __name__ == "__main__":
    if register_gateway:
        try:
            register_to_gateway(gateway_url=gateway_url, agent_base_url=public_base_url)
            print(f"registered to gateway: {gateway_url} -> {public_base_url}")
        except urllib_error.HTTPError as exc:
            # 409 means already registered; keep service startup idempotent.
            if exc.code != 409:
                raise
            print(f"already registered in gateway: {public_base_url}")
        except urllib_error.URLError as exc:
            print(f"gateway register failed: {exc}")

    uvicorn.run(app, host=host, port=port, reload=False, log_config=None)
