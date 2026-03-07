import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from easyagent.agent.agent import DeepAgentRunner
from easyagent.utils.settings import Settings


class _DummyCompiledAgent:
    def invoke(self, invoke_input, config=None):  # noqa: ANN001, ANN201
        return {"messages": [{"role": "assistant", "content": "ok"}]}


class TestModelInjection(unittest.TestCase):
    def test_deep_agent_runner_uses_injected_model(self) -> None:
        with TemporaryDirectory() as tmpdir:
            settings = Settings(
                model_key="dummy-key",
                model_base_url="https://example.com/v1",
                model_name="real-model-name",
                base_path=Path(tmpdir),
                local_mode=True,
            )
            fake_model = GenericFakeChatModel(messages=iter(["fake response"]))

            with (
                patch("easyagent.agent.agent.create_deep_agent", return_value=_DummyCompiledAgent()) as create_agent,
                patch("easyagent.agent.agent._create_chat_model") as create_chat_model,
            ):
                runner = DeepAgentRunner(settings, model=fake_model)

            kwargs = create_agent.call_args.kwargs
            self.assertIs(kwargs["model"], fake_model)
            create_chat_model.assert_not_called()
            runner.close()


if __name__ == "__main__":
    unittest.main()
