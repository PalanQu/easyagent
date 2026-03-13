from easyagent.models.orm.session import Session
from easyagent.models.schema.session import SessionCreate
from easyagent.repos.ports import SessionRepoPort, UserRepoPort


class SessionService:
    def __init__(self, session_repo: SessionRepoPort, user_repo: UserRepoPort):
        self.session_repo = session_repo
        self.user_repo = user_repo

    def create_session(self, payload: SessionCreate) -> Session:
        user = self.user_repo.get_by_id(payload.user_id)
        if user is None:
            raise ValueError(f"user not found: {payload.user_id}")
        return self.session_repo.create(
            user_id=payload.user_id,
            thread_id=payload.thread_id,
            session_context=payload.session_context,
        )

    def get_session_by_id(self, session_id: int) -> Session | None:
        return self.session_repo.get_by_id(session_id)

    def list_sessions_by_user_id(self, user_id: int) -> list[Session]:
        return self.session_repo.get_by_user_id(user_id)

    def get_session_by_thread_id_for_user(self, thread_id: str, user_id: int) -> Session | None:
        return self.session_repo.get_by_user_id_and_thread_id(user_id=user_id, thread_id=thread_id)

    def get_session_by_id_for_user(self, session_id: int, user_id: int) -> Session | None:
        session = self.session_repo.get_by_id(session_id)
        if session is None:
            return None
        if session.user_id != user_id:
            return None
        return session

    def update_session_context_for_user(
        self,
        session_id: int,
        user_id: int,
        session_context: dict,
    ) -> Session | None:
        session = self.get_session_by_id_for_user(session_id, user_id)
        if session is None:
            return None
        return self.session_repo.update_context(session_id, session_context)
