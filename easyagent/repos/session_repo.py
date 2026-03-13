from sqlmodel import Session, select

from easyagent.models.orm.session import Session as SessionModel
from easyagent.repos.ports import SessionRepoPort


class SQLModelSessionRepo(SessionRepoPort):
    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, session_id: int) -> SessionModel | None:
        return self.session.get(SessionModel, session_id)

    def get_by_user_id(self, user_id: int) -> list[SessionModel]:
        statement = select(SessionModel).where(SessionModel.user_id == user_id)
        return list(self.session.exec(statement).all())

    def get_by_user_id_and_thread_id(self, user_id: int, thread_id: str) -> SessionModel | None:
        statement = select(SessionModel).where(
            SessionModel.user_id == user_id,
            SessionModel.thread_id == thread_id,
        )
        return self.session.exec(statement).first()

    def create(
        self,
        user_id: int,
        thread_id: str | None = None,
        session_context: dict | None = None,
    ) -> SessionModel:
        db_session = SessionModel(
            user_id=user_id,
            thread_id=thread_id,
            session_context=session_context or {},
        )
        self.session.add(db_session)
        self.session.commit()
        self.session.refresh(db_session)
        return db_session

    def update_context(self, session_id: int, session_context: dict) -> SessionModel | None:
        db_session = self.get_by_id(session_id)
        if db_session is None:
            return None
        db_session.session_context = session_context
        self.session.add(db_session)
        self.session.commit()
        self.session.refresh(db_session)
        return db_session


class SqliteSessionRepo(SQLModelSessionRepo):
    """SQLite implementation entrypoint."""


class PostgresSessionRepo(SQLModelSessionRepo):
    """PostgreSQL implementation entrypoint."""
