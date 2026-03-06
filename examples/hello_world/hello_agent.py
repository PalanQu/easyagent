import uvicorn
from dotenv import load_dotenv

from easyagent.sdk import EasyagentSDK

from easyagent.utils.settings import Settings

# curl -X POST "http://127.0.0.1:8000/agent/run" \
#  -H "Content-Type: application/json" \
#  -d '{
#    "input": "hi","thread_id": "thread_001"
#  }'


load_dotenv(override=False)
settings = Settings.from_env()

sdk = EasyagentSDK(settings=settings, system_prompt="You are a helpful assistant.")
app = sdk.create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_config=None)
