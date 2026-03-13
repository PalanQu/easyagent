import types
import unittest
import builtins
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from ag_ui.core import EventType, RunFinishedEvent, RunStartedEvent

from easyagent.auth import AuthUser
from easyagent.adapters.fastapi.copilotkit import mount_copilotkit_routes
from easyagent.sdk import EasyagentSDK, Settings


class TestCopilotKitIntegrationUnit(unittest.TestCase):
    @staticmethod
    def _local_settings(tmpdir: str) -> Settings:
        return Settings(
            model_key="dummy-key",
            model_base_url="https://example.com/v1",
            model_name="dummy-model",
            base_path=Path(tmpdir),
            local_mode=True,
        )

    def test_mount_copilotkit_routes_raises_when_dependency_missing(self) -> None:
        app = FastAPI()
        real_import = builtins.__import__

        def _raising_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001, ANN201
            if name in {"copilotkit", "ag_ui_langgraph"}:
                raise ImportError(f"missing {name}")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=_raising_import):
            with self.assertRaisesRegex(RuntimeError, "requires both `copilotkit` and `ag-ui-langgraph`"):
                mount_copilotkit_routes(
                    app=app,
                    graph=object(),
                    path="/copilotkit",
                    name="easyagent",
                    description="desc",
                    authenticate=lambda _request: None,
                )

    def test_mount_copilotkit_routes_returns_401_when_authentication_fails(self) -> None:
        async def _authenticate(_request: Request) -> AuthUser:
            raise HTTPException(status_code=401, detail="Missing user identification")

        app = FastAPI()
        mount_copilotkit_routes(
            app=app,
            graph=object(),
            path="/copilotkit",
            name="easyagent",
            description="desc",
            authenticate=_authenticate,
        )

        with TestClient(app) as client:
            response = client.post(
                "/copilotkit",
                json={
                    "threadId": "thread-1",
                    "runId": "run-1",
                    "state": {},
                    "messages": [],
                    "tools": [],
                    "context": [],
                    "forwardedProps": {},
                },
            )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"detail": "Missing user identification"})

    def test_mount_copilotkit_routes_registers_endpoint_and_injects_user_id(self) -> None:
        fake_copilotkit = types.ModuleType("copilotkit")
        calls: dict[str, object] = {}
        real_import = builtins.__import__

        class _FakeLangGraphAGUIAgent:
            def __init__(self, *, name, graph, description=None, config=None):  # noqa: ANN001
                self.name = name
                self.graph = graph
                self.description = description
                self.config = config
                calls["agent"] = self

            async def run(self, input_data):  # noqa: ANN001
                yield RunStartedEvent(
                    type=EventType.RUN_STARTED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                )
                yield RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                )

        fake_copilotkit.LangGraphAGUIAgent = _FakeLangGraphAGUIAgent

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001, ANN201
            if name == "copilotkit":
                return fake_copilotkit
            return real_import(name, globals, locals, fromlist, level)

        async def _authenticate(_request: Request) -> AuthUser:
            return AuthUser(user_id="user_001", user_name="Alice", email="alice@example.com")

        with patch("builtins.__import__", side_effect=_fake_import):
            app = FastAPI()
            graph = object()
            mount_copilotkit_routes(
                app=app,
                graph=graph,
                path="/copilotkit",
                name="easyagent",
                description="desc",
                authenticate=_authenticate,
            )

        with TestClient(app) as client:
            response = client.post(
                "/copilotkit",
                json={
                    "threadId": "thread-1",
                    "runId": "run-1",
                    "state": {},
                    "messages": [],
                    "tools": [],
                    "context": [],
                    "forwardedProps": {},
                },
                headers={"accept": "text/event-stream"},
            )

        self.assertEqual(response.status_code, 200)
        agent = calls["agent"]
        self.assertIsInstance(agent, _FakeLangGraphAGUIAgent)
        self.assertEqual(agent.name, "easyagent")
        self.assertEqual(agent.description, "desc")
        self.assertIs(agent.graph, graph)
        self.assertEqual(
            agent.config,
            {
                "configurable": {"user_id": "user_001"},
                "metadata": {
                    "langfuse_user_id": "user_001",
                    "langfuse_session_id": "thread-1",
                },
            },
        )

    def test_mount_fastapi_mounts_copilotkit_when_enabled(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with patch("easyagent.agent.agent.create_deep_agent", return_value=object()):
                sdk = EasyagentSDK(
                    self._local_settings(tmpdir),
                    model=object(),
                    a2a_enabled=False,
                    copilotkit_enabled=True,
                    copilotkit_path="/agui",
                    agent_name="sample_agent",
                    agent_description="sample desc",
                )

            app = FastAPI()
            with patch("easyagent.sdk.mount_copilotkit_routes") as mount_mock:
                sdk.mount_fastapi(app)

            mount_mock.assert_called_once_with(
                app=app,
                graph=sdk.agent_runner._agent,
                path="/agui",
                name="sample_agent",
                description="sample desc",
                authenticate=sdk.get_current_user,
            )
            sdk.agent_runner.close()
