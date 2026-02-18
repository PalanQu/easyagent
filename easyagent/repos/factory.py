from sqlmodel import Session

from easyagent.repos.ports import SessionRepoPort, UserRepoPort
from easyagent.repos.session_repo import PostgresSessionRepo, SqliteSessionRepo
from easyagent.repos.user_repo import PostgresUserRepo, SqliteUserRepo


def build_user_repo(db_session: Session, backend: str) -> UserRepoPort:
    if backend == "postgres":
        return PostgresUserRepo(db_session)
    if backend == "sqlite":
        return SqliteUserRepo(db_session)
    raise ValueError(f"unsupported db backend: {backend}")


def build_session_repo(db_session: Session, backend: str) -> SessionRepoPort:
    if backend == "postgres":
        return PostgresSessionRepo(db_session)
    if backend == "sqlite":
        return SqliteSessionRepo(db_session)
    raise ValueError(f"unsupported db backend: {backend}")
