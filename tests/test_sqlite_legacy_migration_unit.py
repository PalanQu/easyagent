import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from easyagent.utils.db import Database
from easyagent.utils.settings import Settings


class TestSqliteLegacyMigrationUnit(unittest.TestCase):
    def test_auto_adds_legacy_columns_for_sqlite(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "legacy.db"
            settings = Settings(
                model_key="dummy-key",
                model_base_url="https://example.com/v1",
                model_name="dummy-model",
                base_path=Path(tmpdir),
                local_mode=True,
                db_backend="sqlite",
                db_url=f"sqlite:///{db_file}",
            )
            database = Database(settings)

            with database.engine.begin() as conn:
                conn.exec_driver_sql(
                    """
                    CREATE TABLE "user" (
                        id INTEGER PRIMARY KEY,
                        user_id TEXT,
                        created_at TEXT,
                        user_context TEXT
                    )
                    """
                )
                conn.exec_driver_sql(
                    """
                    CREATE TABLE "session" (
                        id INTEGER PRIMARY KEY,
                        user_id INTEGER,
                        created_at TEXT,
                        session_context TEXT
                    )
                    """
                )
                conn.exec_driver_sql(
                    """
                    INSERT INTO "user" (id, user_id, created_at, user_context)
                    VALUES (1, 'legacy_alice', '2026-03-13T00:00:00Z', '{}')
                    """
                )

            database.create_tables()

            with database.engine.begin() as conn:
                user_columns = {
                    str(row[1]) for row in conn.exec_driver_sql('PRAGMA table_info("user")').fetchall()
                }
                session_columns = {
                    str(row[1]) for row in conn.exec_driver_sql('PRAGMA table_info("session")').fetchall()
                }
                self.assertIn("external_user_id", user_columns)
                self.assertIn("thread_id", session_columns)

                row = conn.exec_driver_sql('SELECT id FROM "user" WHERE id = 1').fetchone()
                self.assertIsNotNone(row)


if __name__ == "__main__":
    unittest.main()
