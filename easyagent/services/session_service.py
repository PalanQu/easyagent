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
            session_context=payload.session_context,
        )

    def get_session_by_id(self, session_id: int) -> Session | None:
        return self.session_repo.get_by_id(session_id)

    def list_sessions_by_user_id(self, user_id: int) -> list[Session]:
        return self.session_repo.get_by_user_id(user_id)
