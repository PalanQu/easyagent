from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult


@dataclass
class MockRule:
    when_contains: str
    answer: str = ""
    call_tool: bool = False
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None


class ScriptedMockChatModel(BaseChatModel):
    rules: list[MockRule]
    default_answer: str = "mocked: default response"

    def __init__(self, *, rules: list[MockRule], default_answer: str = "mocked: default response"):
        super().__init__(rules=rules, default_answer=default_answer)
        self._bound_tools: set[str] = set()

    @property
    def _llm_type(self) -> str:
        return "scripted-mock-chat-model"

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> "ScriptedMockChatModel":
        bound = ScriptedMockChatModel(rules=self.rules, default_answer=self.default_answer)
        bound._bound_tools = self._extract_tool_names(tools)
        return bound

    def _generate(  # noqa: PLR0913
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        user_text = self._last_user_text(messages)
        rule = self._match_rule(user_text)
        if rule and rule.call_tool:
            tool_name = rule.tool_name
            if not tool_name:
                raise ValueError("tool call rule requires `tool_name`")
            if self._bound_tools and tool_name not in self._bound_tools:
                raise ValueError(f"tool `{tool_name}` is not bound on model")
            tool_call = {
                "id": f"call_{uuid4().hex}",
                "name": tool_name,
                "args": rule.tool_args or {},
                "type": "tool_call",
            }
            message = AIMessage(content=rule.answer, tool_calls=[tool_call])
            return ChatResult(generations=[ChatGeneration(message=message)])

        answer = rule.answer if rule else self.default_answer
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])

    @staticmethod
    def _extract_tool_names(tools: Any) -> set[str]:
        names: set[str] = set()
        if not isinstance(tools, list):
            return names
        for tool in tools:
            if isinstance(tool, dict):
                name = tool.get("name")
                if isinstance(name, str) and name:
                    names.add(name)
                continue
            name = getattr(tool, "name", None)
            if isinstance(name, str) and name:
                names.add(name)
        return names

    @staticmethod
    def _last_user_text(messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                content = message.content
                if isinstance(content, str):
                    return content
                return str(content)
        return ""

    def _match_rule(self, user_text: str) -> MockRule | None:
        lowered = user_text.lower()
        for rule in self.rules:
            if rule.when_contains.lower() in lowered:
                return rule
        return None
