from collections.abc import Generator
from contextlib import contextmanager

from sqlmodel import Session, SQLModel, create_engine

from easyagent.utils.settings import Settings

# Ensure ORM models are imported before metadata operations.
from easyagent.models.orm import session as _session_model  # noqa: F401
from easyagent.models.orm import user as _user_model  # noqa: F401


class Database:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = self._create_engine()

    def _create_engine(self):
        connect_args: dict[str, bool] = {}
        database_url = self.settings.database_url
        if database_url.startswith("postgresql://") or database_url.startswith("postgres://"):
            database_url = self._to_sqlalchemy_postgres_url(database_url)
        if database_url.startswith("sqlite:") or self.settings.db_backend == "sqlite":
            connect_args["check_same_thread"] = False
        return create_engine(
            database_url,
            connect_args=connect_args,
            echo=self.settings.db_echo,
        )

    @staticmethod
    def _to_sqlalchemy_postgres_url(url: str) -> str:
        if url.startswith("postgresql://"):
            return "postgresql+psycopg://" + url[len("postgresql://") :]
        if url.startswith("postgres://"):
            return "postgresql+psycopg://" + url[len("postgres://") :]
        return url

    def create_tables(self) -> None:
        SQLModel.metadata.create_all(self.engine)
        self._run_sqlite_legacy_migrations()

    def _run_sqlite_legacy_migrations(self) -> None:
        database_url = self.settings.database_url
        if not (database_url.startswith("sqlite:") or self.settings.db_backend == "sqlite"):
            return

        with self.engine.begin() as conn:
            self._ensure_sqlite_column(conn, "user", "external_user_id", "VARCHAR")
            self._ensure_sqlite_column(conn, "user", "user_name", "VARCHAR")
            self._ensure_sqlite_column(conn, "user", "email", "VARCHAR")
            self._ensure_sqlite_column(conn, "session", "thread_id", "VARCHAR")

            user_columns = self._get_sqlite_columns(conn, "user")
            if "external_user_id" in user_columns:
                conn.exec_driver_sql(
                    'CREATE UNIQUE INDEX IF NOT EXISTS ix_user_external_user_id ON "user" (external_user_id)'
                )

            session_columns = self._get_sqlite_columns(conn, "session")
            if "thread_id" in session_columns:
                conn.exec_driver_sql('CREATE INDEX IF NOT EXISTS ix_session_thread_id ON "session" (thread_id)')

    @staticmethod
    def _get_sqlite_columns(conn, table_name: str) -> set[str]:
        rows = conn.exec_driver_sql(f'PRAGMA table_info("{table_name}")').fetchall()
        if not rows:
            return set()
        return {str(row[1]) for row in rows}

    def _ensure_sqlite_column(self, conn, table_name: str, column_name: str, column_type: str) -> None:
        columns = self._get_sqlite_columns(conn, table_name)
        if not columns:
            return
        if column_name in columns:
            return
        conn.exec_driver_sql(f'ALTER TABLE "{table_name}" ADD COLUMN {column_name} {column_type}')

    def session(self) -> Generator[Session, None, None]:
        with Session(self.engine) as session:
            yield session

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        with Session(self.engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise


def create_default_database() -> Database:
    return Database(Settings.from_env())
