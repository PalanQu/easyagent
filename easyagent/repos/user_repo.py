from sqlmodel import Session, select

from easyagent.models.orm.user import User
from easyagent.repos.ports import UserRepoPort


class SQLModelUserRepo(UserRepoPort):
    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, user_id: int) -> User | None:
        return self.session.get(User, user_id)

    def get_by_external_user_id(self, external_user_id: str) -> User | None:
        statement = select(User).where(User.external_user_id == external_user_id)
        return self.session.exec(statement).first()

    def get_by_email(self, email: str) -> User | None:
        statement = select(User).where(User.email == email)
        return self.session.exec(statement).first()

    def create(
        self,
        external_user_id: str,
        user_name: str | None = None,
        email: str | None = None,
        user_context: dict | None = None,
    ) -> User:
        user = User(
            external_user_id=external_user_id,
            user_name=user_name,
            email=email,
            user_context=user_context or {},
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user


class SqliteUserRepo(SQLModelUserRepo):
    """SQLite implementation entrypoint."""


class PostgresUserRepo(SQLModelUserRepo):
    """PostgreSQL implementation entrypoint."""
