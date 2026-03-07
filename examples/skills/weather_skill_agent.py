from pathlib import Path
from urllib.request import urlopen

import uvicorn
from dotenv import load_dotenv

from easyagent.sdk import EasyagentSDK
from easyagent.utils.settings import Settings

DEFAULT_WEATHER_SKILL_URL = (
    "https://raw.githubusercontent.com/openclaw/skills/refs/heads/main/"
    "skills/steipete/weather/SKILL.md"
)
SKILL_NAME = "weather"


# curl -X POST "http://127.0.0.1:8000/agent/run" \
#  -H "Content-Type: application/json" \
#  -d '{
#    "input": "What is the weather in Shanghai today?", "thread_id": "thread_weather_001"
#  }'


def install_weather_skill(settings: Settings, url: str = DEFAULT_WEATHER_SKILL_URL) -> Path:
    skill_dir = settings.skills_path / SKILL_NAME
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = skill_dir / "SKILL.md"
    with urlopen(url, timeout=20) as response:  # nosec B310
        content = response.read()

    skill_md.write_bytes(content)
    return skill_md


load_dotenv(override=False)
settings = Settings.from_env()

installed_skill = install_weather_skill(settings)
print(f"Installed weather skill to: {installed_skill}")

sdk = EasyagentSDK(
    settings=settings,
    system_prompt=(
        "You are a weather skill demo assistant. "
    ),
    skills=["/skills/"],
)
app = sdk.create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_config=None)
