import json

import httpx


def main() -> None:
    url = "http://127.0.0.1:8000/ag-ui"
    payload = {
        "input": "Give me a short status update and propose a tiny dashboard UI.",
        "thread_id": "thread_ag_ui_demo_001",
        "run_id": "run_ag_ui_demo_001",
    }

    with httpx.stream("POST", url, json=payload, timeout=120.0) as response:
        response.raise_for_status()
        print("SSE status:", response.status_code)
        print("Streaming events...\n")

        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith("event:"):
                print(f"[event] {line.split(':', 1)[1].strip()}")
                continue
            if not line.startswith("data:"):
                continue

            raw = line.split(":", 1)[1].strip()
            event = json.loads(raw)
            event_type = event.get("type")

            if event_type == "TEXT_MESSAGE_CONTENT":
                print(event.get("delta", ""), end="", flush=True)
                continue

            if event_type == "TOOL_CALL_START":
                print(f"\n[tool:start] {event.get('toolCallName')} ({event.get('toolCallId')})")
                continue

            if event_type == "TOOL_CALL_RESULT":
                print(f"[tool:result] {event.get('toolCallId')} -> {event.get('result')}")
                continue

            if event_type == "CUSTOM" and event.get("name") == "a2ui":
                print(f"\n[a2ui] payload: {json.dumps(event.get('value'), ensure_ascii=False)}")
                continue

            if event_type in {"RUN_FINISHED", "RUN_ERROR"}:
                print(f"\n[{event_type}]")


if __name__ == "__main__":
    main()
