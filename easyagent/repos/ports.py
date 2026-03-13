from typing import Protocol

from easyagent.models.orm.session import Session
from easyagent.models.orm.user import User


class UserRepoPort(Protocol):
    def get_by_id(self, user_id: int) -> User | None:
        ...

    def get_by_external_user_id(self, external_user_id: str) -> User | None:
        ...

    def get_by_email(self, email: str) -> User | None:
        ...

    def create(
        self,
        external_user_id: str,
        user_name: str | None = None,
        email: str | None = None,
        user_context: dict | None = None,
    ) -> User:
        ...


class SessionRepoPort(Protocol):
    def get_by_id(self, session_id: int) -> Session | None:
        ...

    def get_by_user_id(self, user_id: int) -> list[Session]:
        ...

    def get_by_user_id_and_thread_id(self, user_id: int, thread_id: str) -> Session | None:
        ...

    def create(
        self,
        user_id: int,
        thread_id: str | None = None,
        session_context: dict | None = None,
    ) -> Session:
        ...

    def update_context(self, session_id: int, session_context: dict) -> Session | None:
        ...
