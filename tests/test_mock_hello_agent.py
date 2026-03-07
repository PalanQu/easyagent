import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from easyagent.sdk import EasyagentSDK, Settings
from easyagent.models.schema.agent import AgentRunRequest
from tests.tools.mock_chat_model import MockRule, ScriptedMockChatModel


class TestMockHelloAgent(unittest.TestCase):
    @staticmethod
    def _extract_tool_names(sdk: EasyagentSDK) -> set[str]:
        tools_node = sdk.agent_runner._agent.nodes["tools"]
        tool_node = tools_node.bound
        return set(tool_node._tools_by_name.keys())

    def test_mock_agent_returns_fake_response(self) -> None:
        with TemporaryDirectory() as tmpdir:
            settings = Settings(
                model_key="dummy-key",
                model_base_url="https://example.com/v1",
                model_name="real-model-not-used",
                base_path=Path(tmpdir),
                local_mode=True,
            )
            sdk = EasyagentSDK(
                settings=settings,
                system_prompt="You are a helpful assistant.",
                model=ScriptedMockChatModel(
                    rules=[
                        MockRule(
                            when_contains="hi",
                            answer="mocked: hello from fake llm",
                            call_tool=False,
                        )
                    ],
                    default_answer="mocked: fallback",
                ),
            )
            response = sdk.agent_runner.run(
                AgentRunRequest(
                    input="say hi",
                    thread_id="thread_test_mock_hello_001",
                )
            )

            self.assertEqual(response.final_output, "mocked: hello from fake llm")
            sdk.agent_runner.close()

    def test_default_tools_count_is_9(self) -> None:
        with TemporaryDirectory() as tmpdir:
            settings = Settings(
                model_key="dummy-key",
                model_base_url="https://example.com/v1",
                model_name="real-model-not-used",
                base_path=Path(tmpdir),
                local_mode=True,
            )
            sdk = EasyagentSDK(
                settings=settings,
                system_prompt="You are a helpful assistant.",
                model=ScriptedMockChatModel(
                    rules=[MockRule(when_contains="hi", answer="mocked: hello from fake llm")]
                ),
            )
            tool_names = self._extract_tool_names(sdk)
            self.assertEqual(len(tool_names), 9)
            self.assertEqual(
                tool_names,
                {"write_todos", "ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute", "task"},
            )
            sdk.agent_runner.close()

    def test_long_term_memory_same_user_remembers_and_cross_user_isolated(self) -> None:
        class _MemoryToolMockModel(BaseChatModel):
            def __init__(self):
                super().__init__()
                self._bound_tools: set[str] = set()

            @property
            def _llm_type(self) -> str:
                return "memory-tool-mock-model"

            def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # noqa: ANN001, ANN003, ANN201
                bound = _MemoryToolMockModel()
                names: set[str] = set()
                if isinstance(tools, list):
                    for tool in tools:
                        if isinstance(tool, dict):
                            name = tool.get("name")
                            if isinstance(name, str) and name:
                                names.add(name)
                            continue
                        name = getattr(tool, "name", None)
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

                if "remember" in user_text.lower() and "soccer" in user_text.lower():
                    if tool_content is None:
                        if self._bound_tools and "write_file" not in self._bound_tools:
                            raise ValueError("tool `write_file` is not bound on model")
                        return ChatResult(
                            generations=[
                                ChatGeneration(
                                    message=AIMessage(
                                        content="I will remember that.",
                                        tool_calls=[
                                            {
                                                "id": f"call_{uuid4().hex}",
                                                "name": "write_file",
                                                "args": {
                                                    "file_path": "/memory/user_profile.md",
                                                    "content": "hobby: soccer\n",
                                                },
                                                "type": "tool_call",
                                            }
                                        ],
                                    )
                                )
                            ]
                        )
                    return ChatResult(
                        generations=[ChatGeneration(message=AIMessage(content="Got it, I remember you like soccer."))]
                    )

                if "what do i like" in user_text.lower():
                    if tool_content is None:
                        if self._bound_tools and "read_file" not in self._bound_tools:
                            raise ValueError("tool `read_file` is not bound on model")
                        return ChatResult(
                            generations=[
                                ChatGeneration(
                                    message=AIMessage(
                                        content="Let me check memory.",
                                        tool_calls=[
                                            {
                                                "id": f"call_{uuid4().hex}",
                                                "name": "read_file",
                                                "args": {
                                                    "file_path": "/memory/user_profile.md",
                                                },
                                                "type": "tool_call",
                                            }
                                        ],
                                    )
                                )
                            ]
                        )
                    if "soccer" in tool_content.lower():
                        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="You like soccer."))])
                    return ChatResult(
                        generations=[ChatGeneration(message=AIMessage(content="I do not know your hobby yet."))]
                    )

                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="mocked: fallback"))])

        with TemporaryDirectory() as tmpdir:
            settings = Settings(
                model_key="dummy-key",
                model_base_url="https://example.com/v1",
                model_name="real-model-not-used",
                base_path=Path(tmpdir),
                memories_path=Path(tmpdir) / "memory",
                local_mode=True,
            )
            sdk = EasyagentSDK(
                settings=settings,
                system_prompt="You are a helpful assistant.",
                model=_MemoryToolMockModel(),
            )

            remember_response = sdk.agent_runner.run(
                AgentRunRequest(
                    input="Remember my hobby is soccer.",
                    thread_id="thread_memory_write_001",
                    user_id="user_a",
                )
            )
            self.assertIn("remember", (remember_response.final_output or "").lower())

            same_user_response = sdk.agent_runner.run(
                AgentRunRequest(
                    input="What do I like?",
                    thread_id="thread_memory_read_001",
                    user_id="user_a",
                )
            )
            self.assertIn("soccer", (same_user_response.final_output or "").lower())

            other_user_response = sdk.agent_runner.run(
                AgentRunRequest(
                    input="What do I like?",
                    thread_id="thread_memory_read_002",
                    user_id="user_b",
                )
            )
            self.assertIn("do not know", (other_user_response.final_output or "").lower())
            sdk.agent_runner.close()


if __name__ == "__main__":
    unittest.main()
