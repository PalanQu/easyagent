import sys
import types
import unittest
import builtins
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import FastAPI

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
            if name == "ag_ui_langgraph":
                raise ImportError("missing ag_ui_langgraph")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=_raising_import):
            with self.assertRaisesRegex(RuntimeError, "AG-UI integration requires"):
                mount_copilotkit_routes(
                    app=app,
                    graph=object(),
                    path="/copilotkit",
                    name="easyagent",
                    description="desc",
                )

    def test_mount_copilotkit_routes_registers_endpoint(self) -> None:
        fake_ag_ui_langgraph = types.ModuleType("ag_ui_langgraph")
        calls: dict[str, object] = {}

        def _add_langgraph_fastapi_endpoint(app, agent, path):  # noqa: ANN001, ANN201
            calls["app"] = app
            calls["agent"] = agent
            calls["path"] = path

        fake_ag_ui_langgraph.add_langgraph_fastapi_endpoint = _add_langgraph_fastapi_endpoint

        with patch.dict(
            sys.modules,
            {
                "ag_ui_langgraph": fake_ag_ui_langgraph,
            },
        ):
            app = FastAPI()
            graph = object()
            mount_copilotkit_routes(
                app=app,
                graph=graph,
                path="/copilotkit",
                name="easyagent",
                description="desc",
            )

        self.assertIs(calls["app"], app)
        self.assertEqual(calls["path"], "/copilotkit")
        self.assertIs(calls["agent"], graph)

    def test_mount_fastapi_mounts_copilotkit_when_enabled(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with patch("easyagent.agent.agent.create_deep_agent", return_value=object()):
                sdk = EasyagentSDK(
                    self._local_settings(tmpdir),
                    model=object(),
                    a2a_enabled=False,
                    copilotkit_enabled=True,
                    copilotkit_path="/agui",
                    copilotkit_agent_name="sample_agent",
                    copilotkit_agent_description="sample desc",
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
            )
            sdk.agent_runner.close()
