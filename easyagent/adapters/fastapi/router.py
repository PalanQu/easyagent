from typing import Protocol

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlmodel import Session

from easyagent.models.schema.agent import AgentRunRequest, AgentRunResponse
from easyagent.models.schema.auth import AuthUser
from easyagent.models.schema.session import SessionCreate, SessionOut
from easyagent.models.schema.user import UserCreate, UserOut
from easyagent.utils.logging import get_request_logger


class EasyagentSDKRouterProtocol(Protocol):
    def _db_session(self): ...

    def _build_user_service(self, db_session: Session): ...

    def _build_session_service(self, db_session: Session): ...

    @property
    def agent_runner(self): ...


def build_easyagent_router(sdk: EasyagentSDKRouterProtocol) -> APIRouter:
    router = APIRouter(tags=["easyagent"])

    async def get_current_user(request: Request) -> AuthUser:
        cached_user = getattr(request.state, "auth_user", None)
        if isinstance(cached_user, AuthUser):
            return cached_user
        raise HTTPException(status_code=401, detail="Missing user identification")

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.post("/agent/run", response_model=AgentRunResponse)
    async def run_agent(
        request: Request,
        payload: AgentRunRequest,
        current_user: AuthUser = Depends(get_current_user),
    ) -> AgentRunResponse:
        get_request_logger(request, __name__).info("agent run request received")
        payload.user_id = current_user.user_id
        try:
            return sdk.agent_runner.run(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"agent run failed: {exc}") from exc

    @router.post("/users", response_model=UserOut, status_code=201)
    def create_user(
        payload: UserCreate,
        db_session: Session = Depends(sdk._db_session),
    ) -> UserOut:
        try:
            return sdk._build_user_service(db_session).register_user(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/users/by-external-id", response_model=UserOut)
    def get_user_by_external_id(
        external_user_id: str = Query(..., description="External user ID"),
        db_session: Session = Depends(sdk._db_session),
    ) -> UserOut:
        user = sdk._build_user_service(db_session).get_user_by_external_user_id(external_user_id)
        if user is None:
            raise HTTPException(status_code=404, detail=f"user not found: {external_user_id}")
        return user

    @router.post("/sessions", response_model=SessionOut, status_code=201)
    def create_session(
        payload: SessionCreate,
        db_session: Session = Depends(sdk._db_session),
    ) -> SessionOut:
        try:
            return sdk._build_session_service(db_session).create_session(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/sessions/{session_id}", response_model=SessionOut)
    def get_session_by_id(
        session_id: int,
        db_session: Session = Depends(sdk._db_session),
    ) -> SessionOut:
        session = sdk._build_session_service(db_session).get_session_by_id(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"session not found: {session_id}")
        return session

    @router.get("/users/{user_id}/sessions", response_model=list[SessionOut])
    def list_user_sessions(
        user_id: int,
        db_session: Session = Depends(sdk._db_session),
    ) -> list[SessionOut]:
        return sdk._build_session_service(db_session).list_sessions_by_user_id(user_id)

    return router
