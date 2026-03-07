uv run python gateway/main.py
uv run python examples/a2a/multi_agent/sub_agent.py
uv run python examples/a2a/multi_agent/master_agent.py

```
curl -X POST "http://127.0.0.1:8000/agent/run" \
-H "Content-Type: application/json" \
-H "X-User-ID: user_001" \
-H "X-User-Name: Alice" \
-H "X-User-Email: alice@example.com" \
-d '{
  "input": "Ask math_agent to calculate 23.5 plus 18.2, then subtract 10","thread_id": "thread_003"
}'

```