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
        if self.settings.db_backend == "sqlite":
            connect_args["check_same_thread"] = False
        return create_engine(
            self.settings.database_url,
            connect_args=connect_args,
            echo=self.settings.db_echo,
        )

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
