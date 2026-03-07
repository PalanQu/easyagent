import os

import uvicorn
from dotenv import load_dotenv

from easyagent.sdk import EasyagentSDK, Settings

# Start order:
# 1) gateway service on http://127.0.0.1:9000
# 2) this folder's sub_agent.py (it registers itself to gateway)
# 3) this master_agent.py
#
# curl -X POST "http://127.0.0.1:8000/agent/run" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "input": "Ask remote_math_agent to calculate 23.5 plus 18.2, then subtract 10",
#     "thread_id": "thread_gateway_001"
#   }'


load_dotenv(override=False)
settings = Settings.from_env()

host = os.getenv("MASTER_AGENT_HOST", "0.0.0.0")
port = int(os.getenv("MASTER_AGENT_PORT", "8000"))
gateway_url = os.getenv("GATEWAY_URL", "http://127.0.0.1:8010")

sdk = EasyagentSDK(
    settings=settings,
    system_prompt=(
        "You are a coordinator agent. "
        "When arithmetic is involved, delegate to remote_math_agent via subagent tools. "
        "Return concise final answers in English."
    ),
    a2a_gateway_url=gateway_url,
)
app = sdk.create_app()


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port, reload=False, log_config=None)
