from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
import logging
import re
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
import uuid

from a2a.types import AgentCard
from deepagents.middleware.subagents import CompiledSubAgent
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

logger = logging.getLogger(__name__)


def _join_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def _load_json(url: str, *, timeout_seconds: float) -> Any:
    with urllib_request.urlopen(url, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def _post_json(url: str, payload: dict[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
    req = urllib_request.Request(
        url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib_request.urlopen(req, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def _extract_agent_urls(records: Any) -> list[str]:
    if not isinstance(records, list):
        return []
    urls: list[str] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if isinstance(url, str) and url.strip():
            urls.append(url.strip())
    return urls


def _try_get_agent_card(url: str, *, timeout_seconds: float) -> AgentCard | None:
    candidates = [url]
    parsed = urllib_parse.urlparse(url)
    if parsed.path.endswith("/a2a"):
        base_url = urllib_parse.urlunparse(parsed._replace(path=parsed.path[: -len("/a2a")] or "/", query="", fragment=""))
        candidates.append(base_url.rstrip("/"))

    for base in candidates:
        card_url = _join_url(base, "/.well-known/agent-card.json")
        try:
            data = _load_json(card_url, timeout_seconds=timeout_seconds)
            return AgentCard.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed fetching AgentCard from %s: %s", card_url, exc)
    return None


def _sanitize_subagent_name(name: str, *, used: set[str], prefix: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")
    if not normalized:
        normalized = "agent"
    candidate = f"{prefix}{normalized}" if prefix else normalized
    if candidate not in used:
        used.add(candidate)
        return candidate

    i = 2
    while True:
        numbered = f"{candidate}_{i}"
        if numbered not in used:
            used.add(numbered)
            return numbered
        i += 1


def _extract_text_parts(parts: Any) -> list[str]:
    if not isinstance(parts, list):
        return []

    texts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("kind") == "text":
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text)
    return texts


def _extract_output_text(data: dict[str, Any]) -> str:
    result = data.get("result")
    if isinstance(result, dict):
        direct = _extract_text_parts(result.get("parts"))
        if direct:
            return "\n".join(direct)

        status = result.get("status")
        if isinstance(status, dict):
            message = status.get("message")
            if isinstance(message, dict):
                texts = _extract_text_parts(message.get("parts"))
                if texts:
                    return "\n".join(texts)

    error_data = data.get("error")
    if isinstance(error_data, dict):
        message = error_data.get("message")
        if isinstance(message, str) and message.strip():
            raise RuntimeError(message.strip())

    if result is not None:
        return json.dumps(result, ensure_ascii=False)
    return "(empty response)"


def _last_user_text(messages: Any) -> str:
    if not isinstance(messages, Iterable):
        return ""

    last_text = ""
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            last_text = content
            continue

        if isinstance(msg, dict):
            candidate = msg.get("content")
            if isinstance(candidate, str) and candidate.strip():
                last_text = candidate
    return last_text


@dataclass(slots=True)
class _RemoteA2AInvoker:
    rpc_url: str
    timeout_seconds: float

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        text = _last_user_text(state.get("messages", []))
        if not text.strip():
            text = "Please continue based on prior context."

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": str(uuid.uuid4()),
                    "role": "user",
                    "parts": [{"kind": "text", "text": text}],
                }
            },
        }
        response = _post_json(self.rpc_url, payload, timeout_seconds=self.timeout_seconds)
        output = _extract_output_text(response)
        return {"messages": [AIMessage(content=output)]}


def discover_subagents_from_gateway(
    *,
    gateway_url: str,
    agents_path: str = "/agents",
    timeout_seconds: float = 10.0,
    name_prefix: str = "remote_",
) -> list[CompiledSubAgent]:
    records_url = _join_url(gateway_url, agents_path)
    data = _load_json(records_url, timeout_seconds=timeout_seconds)
    agent_urls = _extract_agent_urls(data)
    if not agent_urls:
        return []

    used_names: set[str] = set()
    subagents: list[CompiledSubAgent] = []
    for url in agent_urls:
        try:
            card = _try_get_agent_card(url, timeout_seconds=timeout_seconds)
            if card is None:
                logger.warning("Skip gateway agent %s: cannot resolve AgentCard.", url)
                continue

            name = _sanitize_subagent_name(card.name or "agent", used=used_names, prefix=name_prefix)
            description = card.description or f"Remote A2A agent at {card.url}"
            runnable = RunnableLambda(_RemoteA2AInvoker(rpc_url=card.url, timeout_seconds=timeout_seconds))
            subagents.append({"name": name, "description": description, "runnable": runnable})
        except urllib_error.URLError as exc:
            logger.warning("Skip gateway agent %s due to network error: %s", url, exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skip gateway agent %s due to unexpected error: %s", url, exc)

    return subagents
