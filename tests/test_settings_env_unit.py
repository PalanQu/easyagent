import os
import unittest
from pathlib import Path
from unittest.mock import patch

from easyagent.utils.settings import Settings


class TestSettingsFromEnvUnit(unittest.TestCase):
    def test_from_env_raises_when_required_env_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "missing required env vars"):
                Settings.from_env()

    def test_from_env_raises_on_invalid_bool(self) -> None:
        with patch.dict(
            os.environ,
            {
                "EASYAGENT_MODEL_KEY": "k",
                "EASYAGENT_MODEL_BASE_URL": "https://example.com/v1",
                "EASYAGENT_MODEL_NAME": "m",
                "EASYAGENT_LOCAL_MODE": "not_bool",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "invalid bool value"):
                Settings.from_env()

    def test_from_env_local_mode_defaults_derived_path(self) -> None:
        with patch.dict(
            os.environ,
            {
                "EASYAGENT_MODEL_KEY": "k",
                "EASYAGENT_MODEL_BASE_URL": "https://example.com/v1",
                "EASYAGENT_MODEL_NAME": "m",
                "EASYAGENT_LOCAL_MODE": "true",
            },
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertTrue(settings.local_mode)
        self.assertEqual(settings.memories_path, settings.base_path / "memory")
        self.assertEqual(settings.skills_path, settings.base_path / "skills")
        self.assertEqual(settings.tmp_path, settings.base_path / "tmp")

    def test_from_env_postgres_requires_db_url(self) -> None:
        with patch.dict(
            os.environ,
            {
                "EASYAGENT_MODEL_KEY": "k",
                "EASYAGENT_MODEL_BASE_URL": "https://example.com/v1",
                "EASYAGENT_MODEL_NAME": "m",
                "EASYAGENT_DB_BACKEND": "postgres",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "db_url is required"):
                Settings.from_env()


if __name__ == "__main__":
    unittest.main()
