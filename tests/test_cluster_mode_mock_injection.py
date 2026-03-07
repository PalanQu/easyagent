import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

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


if __name__ == "__main__":
    unittest.main()
