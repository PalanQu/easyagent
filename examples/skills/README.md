uv run python examples/skills/weather_skill_agent.py

curl -X POST "http://127.0.0.1:8000/agent/run" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "今天上海的天气怎么样",
    "thread_id": "thread_weather_005"
  }'
