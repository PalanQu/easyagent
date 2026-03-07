import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from easyagent.models.schema.agent import AgentRunRequest
from easyagent.sdk import EasyagentSDK, Settings
from examples.multi_agent.sub_agent import add, divide, multiply, subtract


class _MasterDelegationMockModel(BaseChatModel):
    def __init__(self, *, force_unknown_subagent: bool = False):
        super().__init__()
        self._bound_tools: set[str] = set()
        self._force_unknown_subagent = force_unknown_subagent

    @property
    def _llm_type(self) -> str:
        return "master-delegation-mock-model"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # noqa: ANN001, ANN003, ANN201
        bound = _MasterDelegationMockModel(force_unknown_subagent=self._force_unknown_subagent)
        names: set[str] = set()
        if isinstance(tools, list):
            for item in tools:
                if isinstance(item, dict):
                    name = item.get("name")
                else:
                    name = getattr(item, "name", None)
                if isinstance(name, str) and name:
                    names.add(name)
        bound._bound_tools = names
        return bound

    def _generate(  # noqa: PLR0913
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,  # noqa: ANN001
        **kwargs,  # noqa: ANN003
    ) -> ChatResult:
        user_text = ""
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                user_text = str(message.content)
                break

        tool_content = None
        for message in reversed(messages):
            if getattr(message, "type", None) == "tool":
                tool_content = str(getattr(message, "content", ""))
                break

        if tool_content is not None:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"Final answer: {tool_content}"))])

        if "math" in user_text.lower() or "+" in user_text or "subtract" in user_text.lower():
            if self._bound_tools and "task" not in self._bound_tools:
                raise ValueError("tool `task` is not bound on model")
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="Delegating arithmetic to subagent.",
                            tool_calls=[
                                {
                                    "id": f"call_{uuid4().hex}",
                                    "name": "task",
                                    "args": {
                                        "description": (
                                            "Calculate 23.5 + 18.2, then subtract 10. "
                                            "Return only the numeric result."
                                        ),
                                        "subagent_type": "unknown_math_agent" if self._force_unknown_subagent else "math_agent",
                                    },
                                    "type": "tool_call",
                                }
                            ],
                        )
                    )
                ]
            )

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="No arithmetic delegation needed."))])


class _MathSubagentMockModel(BaseChatModel):
    def __init__(self):
        super().__init__()
        self._bound_tools: set[str] = set()

    @property
    def _llm_type(self) -> str:
        return "math-subagent-mock-model"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # noqa: ANN001, ANN003, ANN201
        bound = _MathSubagentMockModel()
        names: set[str] = set()
        if isinstance(tools, list):
            for item in tools:
                if isinstance(item, dict):
                    name = item.get("name")
                else:
                    name = getattr(item, "name", None)
                if isinstance(name, str) and name:
                    names.add(name)
        bound._bound_tools = names
        return bound

    def _generate(  # noqa: PLR0913
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,  # noqa: ANN001
        **kwargs,  # noqa: ANN003
    ) -> ChatResult:
        user_text = ""
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                user_text = str(message.content)
                break

        tool_messages = [message for message in messages if getattr(message, "type", None) == "tool"]

        if "23.5 + 18.2" in user_text and "subtract 10" in user_text:
            if len(tool_messages) == 0:
                self._assert_tool_is_bound("add")
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="First add two numbers.",
                                tool_calls=[
                                    {
                                        "id": f"call_{uuid4().hex}",
                                        "name": "add",
                                        "args": {"a": 23.5, "b": 18.2},
                                        "type": "tool_call",
                                    }
                                ],
                            )
                        )
                    ]
                )

            if len(tool_messages) == 1:
                self._assert_tool_is_bound("subtract")
                previous = float(str(getattr(tool_messages[-1], "content", "0")))
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="Then subtract 10.",
                                tool_calls=[
                                    {
                                        "id": f"call_{uuid4().hex}",
                                        "name": "subtract",
                                        "args": {"a": previous, "b": 10.0},
                                        "type": "tool_call",
                                    }
                                ],
                            )
                        )
                    ]
                )

            result = float(str(getattr(tool_messages[-1], "content", "0")))
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"{result:.1f}"))])

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="I only handle arithmetic."))])

    def _assert_tool_is_bound(self, tool_name: str) -> None:
        if self._bound_tools and tool_name not in self._bound_tools:
            raise ValueError(f"tool `{tool_name}` is not bound on model")


class TestMultiAgentExample(unittest.TestCase):
    @staticmethod
    def _close_sdk(sdk: EasyagentSDK) -> None:
        sdk.agent_runner.close()
        sdk.database.engine.dispose()

    def test_math_tools_basic_and_division_by_zero(self) -> None:
        self.assertEqual(add.invoke({"a": 2, "b": 3}), 5)
        self.assertEqual(subtract.invoke({"a": 7, "b": 4}), 3)
        self.assertEqual(multiply.invoke({"a": -5, "b": 0}), 0)
        self.assertAlmostEqual(divide.invoke({"a": 9, "b": 2}), 4.5)

        with self.assertRaisesRegex(ValueError, "division by zero is not allowed"):
            divide.invoke({"a": 1, "b": 0})

    def test_master_delegates_to_math_subagent_and_returns_result(self) -> None:
        with TemporaryDirectory() as tmpdir:
            settings = Settings(
                model_key="dummy-key",
                model_base_url="https://example.com/v1",
                model_name="real-model-not-used",
                base_path=Path(tmpdir),
                local_mode=True,
            )

            sub_sdk = EasyagentSDK(
                settings=settings,
                system_prompt="Math-only subagent.",
                tools=[add, subtract, multiply, divide],
                model=_MathSubagentMockModel(),
            )
            compiled_subagent = {
                "name": "math_agent",
                "description": "Performs arithmetic calculations",
                "runnable": sub_sdk.agent_runner._agent,
            }

            master_sdk = EasyagentSDK(
                settings=settings,
                system_prompt="Delegate arithmetic to math_agent.",
                subagents=[compiled_subagent],
                model=_MasterDelegationMockModel(),
            )

            response = master_sdk.agent_runner.run(
                AgentRunRequest(
                    input="Ask math agent to calculate 23.5 plus 18.2, then subtract 10.",
                    thread_id="thread_multi_agent_success_001",
                    user_id="user_multi_agent_success_001",
                )
            )

            self.assertIn("31.7", (response.final_output or ""))
            self._close_sdk(master_sdk)
            self._close_sdk(sub_sdk)

    def test_master_returns_direct_response_for_non_arithmetic(self) -> None:
        with TemporaryDirectory() as tmpdir:
            settings = Settings(
                model_key="dummy-key",
                model_base_url="https://example.com/v1",
                model_name="real-model-not-used",
                base_path=Path(tmpdir),
                local_mode=True,
            )
            sub_sdk = EasyagentSDK(
                settings=settings,
                system_prompt="Math-only subagent.",
                tools=[add, subtract, multiply, divide],
                model=_MathSubagentMockModel(),
            )
            sdk = EasyagentSDK(
                settings=settings,
                system_prompt="Delegate arithmetic when needed.",
                subagents=[
                    {
                        "name": "math_agent",
                        "description": "Performs arithmetic calculations",
                        "runnable": sub_sdk.agent_runner._agent,
                    }
                ],
                model=_MasterDelegationMockModel(),
            )
            response = sdk.agent_runner.run(
                AgentRunRequest(
                    input="Write me a short greeting.",
                    thread_id="thread_multi_agent_non_math_001",
                )
            )
            self.assertIn("no arithmetic delegation needed", (response.final_output or "").lower())
            self._close_sdk(sdk)
            self._close_sdk(sub_sdk)

    def test_master_surfaces_unknown_subagent_type_error(self) -> None:
        with TemporaryDirectory() as tmpdir:
            settings = Settings(
                model_key="dummy-key",
                model_base_url="https://example.com/v1",
                model_name="real-model-not-used",
                base_path=Path(tmpdir),
                local_mode=True,
            )

            sub_sdk = EasyagentSDK(
                settings=settings,
                system_prompt="Math-only subagent.",
                tools=[add, subtract, multiply, divide],
                model=_MathSubagentMockModel(),
            )
            compiled_subagent = {
                "name": "math_agent",
                "description": "Performs arithmetic calculations",
                "runnable": sub_sdk.agent_runner._agent,
            }

            master_sdk = EasyagentSDK(
                settings=settings,
                system_prompt="Delegate arithmetic to math_agent.",
                subagents=[compiled_subagent],
                model=_MasterDelegationMockModel(force_unknown_subagent=True),
            )
            response = master_sdk.agent_runner.run(
                AgentRunRequest(
                    input="Ask math agent to calculate 23.5 plus 18.2, then subtract 10.",
                    thread_id="thread_multi_agent_bad_subagent_001",
                )
            )
            self.assertIn("does not exist", (response.final_output or "").lower())
            self.assertIn("math_agent", (response.final_output or "").lower())
            self._close_sdk(master_sdk)
            self._close_sdk(sub_sdk)


if __name__ == "__main__":
    unittest.main()
