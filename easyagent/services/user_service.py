from easyagent.models.orm.user import User
from easyagent.models.schema.user import UserCreate
from easyagent.repos.ports import UserRepoPort


class UserService:
    def __init__(self, user_repo: UserRepoPort):
        self.user_repo = user_repo

    def register_user(self, payload: UserCreate) -> User:
        existed = self.user_repo.get_by_external_user_id(payload.external_user_id)
        if existed:
            raise ValueError(f"user already exists: {payload.external_user_id}")

        normalized_email = payload.email.strip().lower() if payload.email else None
        return self.user_repo.create(
            external_user_id=payload.external_user_id,
            user_name=payload.user_name,
            email=normalized_email,
            user_context=payload.user_context,
        )

    def get_user_by_id(self, user_id: int) -> User | None:
        return self.user_repo.get_by_id(user_id)

    def get_user_by_external_user_id(self, external_user_id: str) -> User | None:
        return self.user_repo.get_by_external_user_id(external_user_id)

    def get_user_by_email(self, email: str) -> User | None:
        normalized_email = email.strip().lower()
        return self.user_repo.get_by_email(normalized_email)

    def get_or_create_user(
        self,
        external_user_id: str,
        user_name: str | None = None,
        email: str | None = None,
        user_context: dict | None = None,
    ) -> User:
        user = self.user_repo.get_by_external_user_id(external_user_id)
        if user:
            return user

        normalized_email = email.strip().lower() if email else None
        return self.user_repo.create(
            external_user_id=external_user_id,
            user_name=user_name,
            email=normalized_email,
            user_context=user_context,
        )
