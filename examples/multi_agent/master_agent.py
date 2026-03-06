import uvicorn
from dotenv import load_dotenv

from easyagent.sdk import EasyagentSDK, Settings

from sub_agent import build_math_subagent

#  curl -X POST "http://127.0.0.1:8000/agent/run" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "input": "Ask math_agent to calculate 23.5 plus 18.2, then subtract 10",
#     "thread_id": "thread_multi_001"
#   }'


load_dotenv(override=False)
settings = Settings.from_env()

sdk = EasyagentSDK(
	settings=settings,
	system_prompt=(
		"You are a coordinator agent. "
		"When a request involves arithmetic, call the math_agent subagent. "
		"Return a concise final answer in English."
	),
	subagents=[build_math_subagent(settings)],
)
app = sdk.create_app()


if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_config=None)
