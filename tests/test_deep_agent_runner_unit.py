import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from types import MethodType
from unittest.mock import patch

from easyagent.agent.agent import DeepAgentRunner, InvocationLifecycleLogger
from easyagent.models.schema.agent import AgentRunRequest
from easyagent.utils.settings import Settings


class _RecordingCompiledAgent:
    def __init__(self, state):
        self.state = state
        self.last_invoke_input = None
        self.last_config = None

    def invoke(self, invoke_input, config=None):  # noqa: ANN001, ANN201
        self.last_invoke_input = invoke_input
        self.last_config = config
        return self.state


class _StreamingCompiledAgent:
    def invoke(self, invoke_input, config=None):  # noqa: ANN001, ANN201
        return {"messages": [{"role": "assistant", "content": "fallback"}]}

    def stream(self, invoke_input, config=None, stream_mode=None, subgraphs=False):  # noqa: ANN001, ANN201
        yield (
            (),
            (
                "messages",
                (
                    SimpleNamespace(content="Hel", tool_call_chunks=[]),
                    {"message_id": "msg_1"},
                ),
            ),
        )
        yield (
            (),
            (
                "messages",
                (
                    SimpleNamespace(
                        content="lo",
                        tool_call_chunks=[{"id": "call_1", "name": "search", "args": '{"q":"weather"}'}],
                    ),
                    {"message_id": "msg_1"},
                ),
            ),
        )
        yield (
            (),
            (
                "messages",
                (
                    SimpleNamespace(type="tool", tool_call_id="call_1", content="result"),
                    {"message_id": "msg_1"},
                ),
            ),
        )
        yield ((), ("updates", {"model": {"step": 1}}))
        yield ((), ("custom", {"kind": "a2ui", "surface": {"component": "card"}}))
        yield ((), ("values", {"messages": [{"role": "assistant", "content": "Hello"}], "a2ui": {"surface": {}}}))


class TestDeepAgentRunnerUnit(unittest.TestCase):
    @staticmethod
    def _local_settings(tmpdir: str) -> Settings:
        return Settings(
            model_key="dummy-key",
            model_base_url="https://example.com/v1",
            model_name="dummy-model",
            base_path=Path(tmpdir),
            local_mode=True,
        )

    @staticmethod
    def _cluster_settings(tmpdir: str) -> Settings:
        return Settings(
            model_key="dummy-key",
            model_base_url="https://example.com/v1",
            model_name="dummy-model",
            base_path=Path(tmpdir),
            local_mode=False,
            db_backend="postgres",
            db_url="postgresql://user:pass@localhost:5432/easyagent_test",
        )

    def test_run_builds_invoke_input_and_merges_config(self) -> None:
        with TemporaryDirectory() as tmpdir:
            compiled = _RecordingCompiledAgent(state={"messages": [{"role": "assistant", "content": "ok"}]})
            with patch("easyagent.agent.agent.create_deep_agent", return_value=compiled):
                runner = DeepAgentRunner(self._local_settings(tmpdir), model=object())

            payload = AgentRunRequest(
                input="hello",
                files={"a.txt": "A"},
                thread_id="thread_001",
                user_id="user_001",
                invoke_config={"configurable": {"existing": "value"}},
            )
            response = runner.run(payload)

            self.assertEqual(response.final_output, "ok")
            self.assertEqual(
                compiled.last_invoke_input,
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "files": {"a.txt": "A"},
                },
            )
            self.assertEqual(compiled.last_config["configurable"]["existing"], "value")
            self.assertEqual(compiled.last_config["configurable"]["thread_id"], "thread_001")
            self.assertEqual(compiled.last_config["configurable"]["user_id"], "user_001")
            callbacks = compiled.last_config["callbacks"]
            self.assertIsInstance(callbacks[-1], InvocationLifecycleLogger)
            runner.close()

    def test_run_uses_invoke_input_precedence_and_preserves_existing_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            compiled = _RecordingCompiledAgent(state={"messages": [{"role": "assistant", "content": "ok"}]})
            with patch("easyagent.agent.agent.create_deep_agent", return_value=compiled):
                runner = DeepAgentRunner(self._local_settings(tmpdir), model=object())

            payload = AgentRunRequest(
                input="ignored",
                files={"new.txt": "new"},
                invoke_input={"messages": [{"role": "user", "content": "from invoke_input"}], "files": {"old.txt": "old"}},
            )
            runner.run(payload)

            self.assertEqual(compiled.last_invoke_input["messages"][0]["content"], "from invoke_input")
            self.assertEqual(compiled.last_invoke_input["files"], {"old.txt": "old"})
            runner.close()

    def test_run_requires_thread_id_in_cluster_mode(self) -> None:
        with TemporaryDirectory() as tmpdir:
            compiled = _RecordingCompiledAgent(state={"messages": [{"role": "assistant", "content": "ok"}]})
            with (
                patch(
                    "easyagent.agent.agent.ClusterModeRuntimeFactory.create_runtime_kwargs",
                    return_value={"checkpointer": object(), "store": object(), "backend": lambda _: object()},
                ),
                patch("easyagent.agent.agent.create_deep_agent", return_value=compiled),
            ):
                runner = DeepAgentRunner(self._cluster_settings(tmpdir), model=object())

            with self.assertRaisesRegex(ValueError, "thread_id"):
                runner.run(AgentRunRequest(input="hello"))
            runner.close()

    def test_run_requires_input_when_invoke_input_missing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            compiled = _RecordingCompiledAgent(state={"messages": [{"role": "assistant", "content": "ok"}]})
            with patch("easyagent.agent.agent.create_deep_agent", return_value=compiled):
                runner = DeepAgentRunner(self._local_settings(tmpdir), model=object())

            with self.assertRaisesRegex(ValueError, "`input` is required"):
                runner.run(AgentRunRequest(input="   "))
            runner.close()

    def test_run_adds_langfuse_handler_and_metadata_when_enabled(self) -> None:
        with TemporaryDirectory() as tmpdir:
            compiled = _RecordingCompiledAgent(state={"messages": [{"role": "assistant", "content": "ok"}]})
            with patch("easyagent.agent.agent.create_deep_agent", return_value=compiled):
                runner = DeepAgentRunner(self._local_settings(tmpdir), model=object())

            runner._langfuse_enabled = True
            sentinel_handler = object()

            def _build_handler(_self):  # noqa: ANN001
                return sentinel_handler

            runner._build_langfuse_handler = MethodType(_build_handler, runner)

            payload = AgentRunRequest(
                input="hello",
                thread_id="thread_001",
                user_id="user_001",
                invoke_config={"metadata": {"keep": "yes"}},
            )
            runner.run(payload)

            callbacks = compiled.last_config["callbacks"]
            self.assertIn(sentinel_handler, callbacks)
            self.assertIsInstance(callbacks[-1], InvocationLifecycleLogger)
            metadata = compiled.last_config["metadata"]
            self.assertEqual(metadata["keep"], "yes")
            self.assertEqual(metadata["langfuse_user_id"], "user_001")
            self.assertEqual(metadata["langfuse_session_id"], "thread_001")
            runner.close()

    def test_extract_final_output_and_coerce_text_variants(self) -> None:
        with TemporaryDirectory() as tmpdir:
            compiled = _RecordingCompiledAgent(state={"messages": []})
            with patch("easyagent.agent.agent.create_deep_agent", return_value=compiled):
                runner = DeepAgentRunner(self._local_settings(tmpdir), model=object())

            state = {
                "messages": [
                    {"role": "assistant", "content": [{"type": "text", "text": "line1"}, "line2"]},
                ]
            }
            self.assertEqual(runner._extract_final_output(state), "line1\nline2")
            self.assertEqual(runner._coerce_text(123), "123")
            self.assertIsNone(runner._coerce_text(None))
            self.assertEqual(runner._safe_jsonable(["x"]), {"result": ["x"]})
            runner.close()

    def test_run_stream_emits_ag_ui_lifecycle_events(self) -> None:
        with TemporaryDirectory() as tmpdir:
            compiled = _RecordingCompiledAgent(state={"messages": [{"role": "assistant", "content": "ok"}]})
            with patch("easyagent.agent.agent.create_deep_agent", return_value=compiled):
                runner = DeepAgentRunner(self._local_settings(tmpdir), model=object())

            events = runner.run_stream(AgentRunRequest(input="hello", thread_id="thread_001"))
            self.assertEqual(events[0].type, "RUN_STARTED")
            self.assertEqual(events[-1].type, "RUN_FINISHED")
            self.assertTrue(any(event.type == "TEXT_MESSAGE_CONTENT" for event in events))
            self.assertTrue(any(event.type == "STATE_SNAPSHOT" for event in events))
            runner.close()

    def test_run_stream_emits_run_error_when_run_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            compiled = _RecordingCompiledAgent(state={"messages": [{"role": "assistant", "content": "ok"}]})
            with patch("easyagent.agent.agent.create_deep_agent", return_value=compiled):
                runner = DeepAgentRunner(self._cluster_settings(tmpdir), model=object())

            events = runner.run_stream(AgentRunRequest(input="hello"))
            self.assertEqual(events[0].type, "RUN_STARTED")
            self.assertEqual(events[-1].type, "RUN_ERROR")
            self.assertIn("thread_id", events[-1].error or "")
            runner.close()

    def test_iter_ag_ui_events_streams_messages_tools_and_custom(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with patch("easyagent.agent.agent.create_deep_agent", return_value=_StreamingCompiledAgent()):
                runner = DeepAgentRunner(self._local_settings(tmpdir), model=object())

            events = list(runner.iter_ag_ui_events(AgentRunRequest(input="hello", thread_id="thread_001")))
            types = [event.type for event in events]
            self.assertEqual(types[0], "RUN_STARTED")
            self.assertIn("TEXT_MESSAGE_START", types)
            self.assertGreaterEqual(types.count("TEXT_MESSAGE_CONTENT"), 2)
            self.assertIn("TOOL_CALL_START", types)
            self.assertIn("TOOL_CALL_ARGS", types)
            self.assertIn("TOOL_CALL_RESULT", types)
            self.assertIn("STATE_DELTA", types)
            self.assertIn("STATE_SNAPSHOT", types)
            self.assertEqual(types[-1], "RUN_FINISHED")
            custom_names = [event.name for event in events if event.type == "CUSTOM"]
            self.assertIn("a2ui", custom_names)
            runner.close()


if __name__ == "__main__":
    unittest.main()
