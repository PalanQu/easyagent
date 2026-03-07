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
