import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from easyagent.models.schema.agent import AgentRunRequest
from easyagent.repos.session_repo import PostgresSessionRepo
from easyagent.repos.user_repo import PostgresUserRepo
from easyagent.sdk import EasyagentSDK, Settings


class _DummyCompiledAgent:
    def invoke(self, invoke_input, config=None):  # noqa: ANN001, ANN201
        return {"messages": [{"role": "assistant", "content": "cluster mocked ok"}]}


class _FakeDatabase:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.create_tables_called = 0

    def create_tables(self) -> None:
        self.create_tables_called += 1

    def session(self):  # noqa: ANN201
        yield object()


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
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Got it, I remember you like soccer."))])

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
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="I do not know your hobby yet."))])

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="mocked: fallback"))])


class TestClusterModeMockInjection(unittest.TestCase):
    def test_cluster_mode_can_be_tested_without_real_postgres(self) -> None:
        with TemporaryDirectory() as tmpdir:
            settings = Settings(
                model_key="dummy-key",
                model_base_url="https://example.com/v1",
                model_name="real-model-name",
                base_path=Path(tmpdir),
                local_mode=False,
                db_backend="postgres",
                db_url="postgresql://user:pass@localhost:5432/easyagent_test",
            )
            fake_model = GenericFakeChatModel(messages=iter(["unused"]))

            with (
                patch("easyagent.sdk.Database", side_effect=lambda s: _FakeDatabase(s)) as mocked_db,
                patch(
                    "easyagent.agent.agent.ClusterModeRuntimeFactory.create_runtime_kwargs",
                    return_value={
                        "checkpointer": object(),
                        "store": object(),
                        "backend": lambda _: object(),
                    },
                ),
                patch("easyagent.agent.agent.create_deep_agent", return_value=_DummyCompiledAgent()) as create_agent,
                patch("easyagent.agent.agent._create_chat_model") as create_chat_model,
            ):
                sdk = EasyagentSDK(
                    settings=settings,
                    system_prompt="You are a helpful assistant.",
                    model=fake_model,
                )

            self.assertEqual(mocked_db.call_count, 1)
            self.assertEqual(sdk._db_backend, "postgres")

            kwargs = create_agent.call_args.kwargs
            self.assertIs(kwargs["model"], fake_model)
            create_chat_model.assert_not_called()

            user_service = sdk._build_user_service(object())
            self.assertIsInstance(user_service.user_repo, PostgresUserRepo)

            session_service = sdk._build_session_service(object())
            self.assertIsInstance(session_service.user_repo, PostgresUserRepo)
            self.assertIsInstance(session_service.session_repo, PostgresSessionRepo)

            response = sdk.agent_runner.run(
                AgentRunRequest(
                    input="say hi",
                    thread_id="thread_cluster_mock_001",
                    user_id="user_cluster_mock_001",
                )
            )
            self.assertEqual(response.final_output, "cluster mocked ok")
            sdk.agent_runner.close()

    def test_cluster_mode_long_term_memory_can_be_mocked(self) -> None:
        def _mock_cluster_runtime_kwargs(factory_self):  # noqa: ANN001
            factory_self._ensure_dirs()
            return {
                "checkpointer": InMemorySaver(),
                "store": InMemoryStore(),
                "backend": factory_self._create_backend_factory(),
            }

        with TemporaryDirectory() as tmpdir:
            settings = Settings(
                model_key="dummy-key",
                model_base_url="https://example.com/v1",
                model_name="real-model-name",
                base_path=Path(tmpdir),
                local_mode=False,
                db_backend="postgres",
                db_url="postgresql://user:pass@localhost:5432/easyagent_test",
            )

            with (
                patch("easyagent.sdk.Database", side_effect=lambda s: _FakeDatabase(s)),
                patch(
                    "easyagent.agent.agent.ClusterModeRuntimeFactory.create_runtime_kwargs",
                    new=_mock_cluster_runtime_kwargs,
                ),
            ):
                sdk = EasyagentSDK(
                    settings=settings,
                    system_prompt="You are a helpful assistant.",
                    model=_MemoryToolMockModel(),
                )

            remember_response = sdk.agent_runner.run(
                AgentRunRequest(
                    input="Remember my hobby is soccer.",
                    thread_id="thread_cluster_memory_write_001",
                    user_id="user_a",
                )
            )
            self.assertIn("remember", (remember_response.final_output or "").lower())

            same_user_response = sdk.agent_runner.run(
                AgentRunRequest(
                    input="What do I like?",
                    thread_id="thread_cluster_memory_read_001",
                    user_id="user_a",
                )
            )
            self.assertIn("soccer", (same_user_response.final_output or "").lower())

            other_user_response = sdk.agent_runner.run(
                AgentRunRequest(
                    input="What do I like?",
                    thread_id="thread_cluster_memory_read_002",
                    user_id="user_b",
                )
            )
            self.assertIn("do not know", (other_user_response.final_output or "").lower())
            sdk.agent_runner.close()


if __name__ == "__main__":
    unittest.main()
