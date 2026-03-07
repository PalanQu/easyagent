import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from easyagent.auth import AuthUser, CallableAuthProvider
from easyagent.sdk import EasyagentSDK, Settings
from tests.tools import MockRule, ScriptedMockChatModel


async def authenticate_user(request: Request) -> AuthUser:
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user identification")

    return AuthUser(
        user_id=user_id,
        user_name=request.headers.get("X-User-Name"),
        email=request.headers.get("X-User-Email"),
    )


class TestAuthAgent(unittest.TestCase):
    @staticmethod
    def _build_sdk(tmpdir: str) -> EasyagentSDK:
        settings = Settings(
            model_key="dummy-key",
            model_base_url="https://example.com/v1",
            model_name="real-model-not-used",
            base_path=Path(tmpdir),
            local_mode=True,
        )
        return EasyagentSDK(
            settings=settings,
            system_prompt="You are a helpful assistant.",
            auth_provider=CallableAuthProvider(authenticate_user),
            model=ScriptedMockChatModel(
                rules=[MockRule(when_contains="hi", answer="mocked: hello from fake llm")]
            ),
        )

    @staticmethod
    def _close_sdk(sdk: EasyagentSDK) -> None:
        sdk.agent_runner.close()
        sdk.database.engine.dispose()

    def test_agent_run_injects_user_id_from_auth_header(self) -> None:
        with TemporaryDirectory() as tmpdir:
            sdk = self._build_sdk(tmpdir)
            app = sdk.create_app()

            with TestClient(app) as client, patch.object(sdk.agent_runner, "run", wraps=sdk.agent_runner.run) as run_spy:
                response = client.post(
                    "/agent/run",
                    json={
                        "input": "hi",
                        "thread_id": "thread_auth_001",
                        "user_id": "spoofed_user_id",
                    },
                    headers={
                        "X-User-ID": "user_001",
                        "X-User-Name": "Alice",
                        "X-User-Email": "alice@example.com",
                    },
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json().get("final_output"), "mocked: hello from fake llm")
            self.assertEqual(run_spy.call_count, 1)

            called_payload = run_spy.call_args.args[0]
            self.assertEqual(called_payload.user_id, "user_001")

            self._close_sdk(sdk)

    def test_agent_run_returns_401_when_missing_user_header(self) -> None:
        with TemporaryDirectory() as tmpdir:
            sdk = self._build_sdk(tmpdir)
            app = sdk.create_app()

            with TestClient(app) as client:
                response = client.post(
                    "/agent/run",
                    json={
                        "input": "hi",
                        "thread_id": "thread_auth_unauthorized_001",
                    },
                )

            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.json(), {"detail": "Missing user identification"})

            self._close_sdk(sdk)


if __name__ == "__main__":
    unittest.main()
