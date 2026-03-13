import uvicorn
from copilotkit import CopilotKitMiddleware
from dotenv import load_dotenv

from easyagent.sdk import EasyagentSDK
from easyagent.utils.settings import Settings

#  curl -N -X POST "http://127.0.0.1:8000/copilotkit" \
#    -H "Content-Type: application/json" \
#    -H "Accept: text/event-stream" \
#    -d '{
#      "threadId": "thread_agui_001",
#      "runId": "run_001",
#      "state": {},
#      "messages": [
#        {
#          "id": "msg_001",
#          "role": "user",
#          "content": "what is the weather in Shanghai?"
#        }
#      ],
#      "tools": [],
#      "context": [],
#      "forwardedProps": {}
#    }'



def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"The weather in {location} is sunny."


load_dotenv(override=False)
settings = Settings.from_env()

sdk = EasyagentSDK(
    settings=settings,
    system_prompt="You are a helpful research assistant.",
    tools=[get_weather],
    middleware=[CopilotKitMiddleware()],
    copilotkit_enabled=True,
    copilotkit_path="/copilotkit",
    agent_name="sample_agent",
    agent_description=(
        "An example agent to use as a starting point for your own agent."
    ),
)
app = sdk.create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_config=None)
