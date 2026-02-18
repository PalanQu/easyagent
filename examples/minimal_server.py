import uvicorn

from easyagent.sdk import EasyagentSDK

from easyagent.utils.settings import Settings

settings = Settings(
    model_key="",
    model_base_url="https://api.deepseek.com/v1",
    model_name="deepseek-chat",
    local_mode=True,
)

sdk = EasyagentSDK(settings=settings, system_prompt="You are a helpful assistant.")
app = sdk.create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
