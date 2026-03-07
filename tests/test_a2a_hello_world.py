import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from fastapi.testclient import TestClient

from easyagent.sdk import EasyagentSDK, Settings
from tests.tools import MockRule, ScriptedMockChatModel


class TestA2AHelloWorld(unittest.TestCase):
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
            model=ScriptedMockChatModel(
                rules=[MockRule(when_contains="hi", answer="mocked: hello from fake llm")]
            ),
        )

    @staticmethod
    def _build_payload(text: str) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": str(uuid4()),
                    "role": "user",
                    "parts": [{"kind": "text", "text": text}],
                }
            },
        }

    @staticmethod
    def _extract_result_texts(data: dict) -> list[str]:
        parts = data.get("result", {}).get("status", {}).get("message", {}).get("parts", [])
        if not isinstance(parts, list):
            return []
        return [
            str(part.get("text"))
            for part in parts
            if isinstance(part, dict) and part.get("kind") == "text" and part.get("text") is not None
        ]

    @staticmethod
    def _close_sdk(sdk: EasyagentSDK) -> None:
        sdk.agent_runner.close()
        sdk.database.engine.dispose()

    def test_a2a_message_send_returns_agent_text(self) -> None:
        with TemporaryDirectory() as tmpdir:
            sdk = self._build_sdk(tmpdir)
            app = sdk.create_app()

            with TestClient(app) as client:
                response = client.post("/a2a", json=self._build_payload("hi"))

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body.get("jsonrpc"), "2.0")
            self.assertIsNotNone(body.get("result"))
            texts = self._extract_result_texts(body)
            self.assertTrue(any("mocked: hello from fake llm" in text for text in texts))

            self._close_sdk(sdk)

    def test_a2a_message_send_rejects_empty_text(self) -> None:
        with TemporaryDirectory() as tmpdir:
            sdk = self._build_sdk(tmpdir)
            app = sdk.create_app()

            with TestClient(app) as client:
                response = client.post("/a2a", json=self._build_payload("   "))

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body.get("jsonrpc"), "2.0")
            self.assertIsNotNone(body.get("result"))
            texts = self._extract_result_texts(body)
            self.assertTrue(any("A2A request requires text input in message.parts." in text for text in texts))

            self._close_sdk(sdk)


if __name__ == "__main__":
    unittest.main()
