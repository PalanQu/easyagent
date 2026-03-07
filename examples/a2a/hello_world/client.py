import json
import uuid

import httpx


def main() -> None:
    base_url = "http://127.0.0.1:8000"
    rpc_url = f"{base_url}/a2a"
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "messageId": str(uuid.uuid4()),
                "role": "user",
                "parts": [{"kind": "text", "text": "hi"}],
            }
        },
    }

    with httpx.Client(timeout=60) as client:
        response = client.post(rpc_url, json=payload)
        response.raise_for_status()
        data = response.json()

    print("status:", response.status_code)
    print("response:")
    print(json.dumps(data, ensure_ascii=False, indent=2))

    parts = (
        data.get("result", {})
        .get("status", {})
        .get("message", {})
        .get("parts", [])
    )
    texts = [p.get("text") for p in parts if isinstance(p, dict) and p.get("kind") == "text"]
    if texts:
        print("\nassistant:", "\n".join(texts))


if __name__ == "__main__":
    main()
