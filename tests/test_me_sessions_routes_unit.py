import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from easyagent.auth import AuthUser, CallableAuthProvider
from easyagent.models.schema.agent import AgentRunResponse
from easyagent.sdk import EasyagentSDK, Settings


async def _authenticate_user(request: Request) -> AuthUser:
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user identification")
    return AuthUser(
        user_id=user_id,
        user_name=request.headers.get("X-User-Name"),
        email=request.headers.get("X-User-Email"),
    )


class TestMeSessionsRoutesUnit(unittest.TestCase):
    @staticmethod
    def _build_sdk(tmpdir: str) -> EasyagentSDK:
        settings = Settings(
            model_key="dummy-key",
            model_base_url="https://example.com/v1",
            model_name="dummy-model",
            base_path=Path(tmpdir),
            local_mode=True,
        )
        with patch("easyagent.agent.agent.create_deep_agent", return_value=object()):
            return EasyagentSDK(
                settings=settings,
                auth_provider=CallableAuthProvider(_authenticate_user),
                model=object(),
                a2a_enabled=False,
                copilotkit_enabled=False,
            )

    @staticmethod
    def _close_sdk(sdk: EasyagentSDK) -> None:
        sdk.agent_runner.close()
        sdk.database.engine.dispose()

    def test_me_sessions_crud_and_scope(self) -> None:
        with TemporaryDirectory() as tmpdir:
            sdk = self._build_sdk(tmpdir)
            app = sdk.create_app()

            with TestClient(app) as client:
                unauth_response = client.get("/me/sessions")
                self.assertEqual(unauth_response.status_code, 401)

                create_response = client.post(
                    "/me/sessions",
                    json={"session_context": {"title": "first thread"}},
                    headers={"X-User-ID": "alice"},
                )
                self.assertEqual(create_response.status_code, 201)
                created = create_response.json()
                self.assertEqual(created["session_context"], {"title": "first thread"})

                session_id = created["id"]

                list_response = client.get("/me/sessions", headers={"X-User-ID": "alice"})
                self.assertEqual(list_response.status_code, 200)
                sessions = list_response.json()
                self.assertEqual(len(sessions), 1)
                self.assertEqual(sessions[0]["id"], session_id)

                get_response = client.get(f"/me/sessions/{session_id}", headers={"X-User-ID": "alice"})
                self.assertEqual(get_response.status_code, 200)
                self.assertEqual(get_response.json()["id"], session_id)

                patch_response = client.patch(
                    f"/me/sessions/{session_id}",
                    json={"session_context": {"title": "renamed"}},
                    headers={"X-User-ID": "alice"},
                )
                self.assertEqual(patch_response.status_code, 200)
                self.assertEqual(patch_response.json()["session_context"], {"title": "renamed"})

                user_response = client.post(
                    "/users",
                    json={"external_user_id": "bob"},
                )
                self.assertEqual(user_response.status_code, 201)
                bob_id = user_response.json()["id"]
                other_session = client.post(
                    "/sessions",
                    json={"user_id": bob_id, "session_context": {"title": "bob thread"}},
                )
                self.assertEqual(other_session.status_code, 201)
                other_session_id = other_session.json()["id"]

                forbidden_get = client.get(
                    f"/me/sessions/{other_session_id}",
                    headers={"X-User-ID": "alice"},
                )
                self.assertEqual(forbidden_get.status_code, 404)

            self._close_sdk(sdk)

    def test_agent_run_auto_binds_thread_to_current_user_session(self) -> None:
        with TemporaryDirectory() as tmpdir:
            sdk = self._build_sdk(tmpdir)
            app = sdk.create_app()

            with patch.object(
                sdk.agent_runner,
                "run",
                return_value=AgentRunResponse(final_output="ok", state={}),
            ):
                with TestClient(app) as client:
                    response = client.post(
                        "/agent/run",
                        json={"input": "hello", "thread_id": "thread-auto-1"},
                        headers={"X-User-ID": "alice"},
                    )
                    self.assertEqual(response.status_code, 200)

                    sessions = client.get("/me/sessions", headers={"X-User-ID": "alice"})
                    self.assertEqual(sessions.status_code, 200)
                    body = sessions.json()
                    self.assertEqual(len(body), 1)
                    self.assertEqual(body[0]["thread_id"], "thread-auto-1")

            self._close_sdk(sdk)

    def test_list_messages_from_thread_state(self) -> None:
        with TemporaryDirectory() as tmpdir:
            sdk = self._build_sdk(tmpdir)
            app = sdk.create_app()

            with TestClient(app) as client:
                created = client.post(
                    "/me/sessions",
                    json={"thread_id": "thread-msg-1", "session_context": {}},
                    headers={"X-User-ID": "alice"},
                )
                self.assertEqual(created.status_code, 201)
                session_id = created.json()["id"]

                with patch.object(
                    sdk.agent_runner,
                    "get_thread_state",
                    return_value={
                        "messages": [
                            {"type": "human", "content": "hi"},
                            {"type": "ai", "content": [{"type": "text", "text": "hello"}]},
                        ]
                    },
                ):
                    response = client.get(
                        f"/me/sessions/{session_id}/messages",
                        headers={"X-User-ID": "alice"},
                    )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.json(),
                    [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hello"},
                    ],
                )

            self._close_sdk(sdk)


if __name__ == "__main__":
    unittest.main()
