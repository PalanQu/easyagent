import logging
from typing import Protocol

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlmodel import Session

from easyagent.models.schema.agent import AgentRunRequest, AgentRunResponse
from easyagent.models.schema.auth import AuthUser
from easyagent.models.schema.session import (
    SessionCreate,
    SessionCreateForCurrentUser,
    SessionMessageOut,
    SessionOut,
    SessionUpdate,
)
from easyagent.models.schema.user import UserCreate, UserOut
from easyagent.utils.logging import get_request_logger


class EasyagentSDKRouterProtocol(Protocol):
    def _db_session(self): ...

    def _build_user_service(self, db_session: Session): ...

    def _build_session_service(self, db_session: Session): ...

    @property
    def agent_runner(self): ...

    def ensure_session_for_user_thread(self, auth_user: AuthUser, thread_id: str) -> None: ...


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
        if payload.thread_id:
            sdk.ensure_session_for_user_thread(auth_user=current_user, thread_id=payload.thread_id)
        try:
            return sdk.agent_runner.run(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logging.getLogger(__name__).error(f"agent run failed: {exc}", exc_info=True)
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
        except Exception as exc:
            logging.getLogger(__name__).error(f"user creation failed: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"user creation failed: {exc}") from exc

    @router.get("/users/by-external-id", response_model=UserOut)
    def get_user_by_external_id(
        external_user_id: str = Query(..., description="External user ID"),
        db_session: Session = Depends(sdk._db_session),
    ) -> UserOut:
        try:
            user = sdk._build_user_service(db_session).get_user_by_external_user_id(external_user_id)
            if user is None:
                raise HTTPException(status_code=404, detail=f"user not found: {external_user_id}")
            return user
        except Exception as exc:
            logging.getLogger(__name__).error(f"get user by external ID failed: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"get user by external ID failed: {exc}") from exc

    @router.post("/sessions", response_model=SessionOut, status_code=201)
    def create_session(
        payload: SessionCreate,
        db_session: Session = Depends(sdk._db_session),
    ) -> SessionOut:
        try:
            return sdk._build_session_service(db_session).create_session(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _resolve_or_create_current_db_user(request: Request, db_session: Session):
        auth_user = request.state.auth_user
        user_service = sdk._build_user_service(db_session)
        return user_service.get_or_create_user(
            external_user_id=auth_user.user_id,
            user_name=auth_user.user_name,
            email=auth_user.email,
        )

    @router.post("/me/sessions", response_model=SessionOut, status_code=201)
    def create_current_user_session(
        request: Request,
        payload: SessionCreateForCurrentUser,
        current_user: AuthUser = Depends(get_current_user),
        db_session: Session = Depends(sdk._db_session),
    ) -> SessionOut:
        _ = current_user
        user = _resolve_or_create_current_db_user(request, db_session)
        session_payload = SessionCreate(
            user_id=user.id,
            thread_id=payload.thread_id,
            session_context=payload.session_context,
        )
        return sdk._build_session_service(db_session).create_session(session_payload)

    @router.get("/me/sessions", response_model=list[SessionOut])
    def list_current_user_sessions(
        request: Request,
        current_user: AuthUser = Depends(get_current_user),
        db_session: Session = Depends(sdk._db_session),
    ) -> list[SessionOut]:
        _ = current_user
        user = _resolve_or_create_current_db_user(request, db_session)
        return sdk._build_session_service(db_session).list_sessions_by_user_id(user.id)

    @router.get("/me/sessions/{session_id}", response_model=SessionOut)
    def get_current_user_session(
        session_id: int,
        request: Request,
        current_user: AuthUser = Depends(get_current_user),
        db_session: Session = Depends(sdk._db_session),
    ) -> SessionOut:
        _ = current_user
        user = _resolve_or_create_current_db_user(request, db_session)
        session = sdk._build_session_service(db_session).get_session_by_id_for_user(session_id, user.id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"session not found: {session_id}")
        return session

    @router.patch("/me/sessions/{session_id}", response_model=SessionOut)
    def update_current_user_session(
        session_id: int,
        payload: SessionUpdate,
        request: Request,
        current_user: AuthUser = Depends(get_current_user),
        db_session: Session = Depends(sdk._db_session),
    ) -> SessionOut:
        _ = current_user
        user = _resolve_or_create_current_db_user(request, db_session)
        session = sdk._build_session_service(db_session).update_session_context_for_user(
            session_id=session_id,
            user_id=user.id,
            session_context=payload.session_context,
        )
        if session is None:
            raise HTTPException(status_code=404, detail=f"session not found: {session_id}")
        return session

    def _extract_text_from_message_content(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part.strip())
        if content is None:
            return ""
        return str(content)

    @router.get("/me/sessions/{session_id}/messages", response_model=list[SessionMessageOut])
    def list_current_user_session_messages(
        session_id: int,
        request: Request,
        current_user: AuthUser = Depends(get_current_user),
        db_session: Session = Depends(sdk._db_session),
    ) -> list[SessionMessageOut]:
        _ = current_user
        user = _resolve_or_create_current_db_user(request, db_session)
        session = sdk._build_session_service(db_session).get_session_by_id_for_user(session_id, user.id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"session not found: {session_id}")
        if not session.thread_id:
            return []

        try:
            state = sdk.agent_runner.get_thread_state(
                thread_id=session.thread_id,
                user_id=request.state.auth_user.user_id,
            )
        except Exception:
            return []
        raw_messages = state.get("messages")
        if not isinstance(raw_messages, list):
            return []

        messages: list[SessionMessageOut] = []
        for raw in raw_messages:
            if not isinstance(raw, dict):
                continue
            role = raw.get("role") or raw.get("type")
            if role not in {"human", "user", "ai", "assistant", "tool"}:
                continue
            content = _extract_text_from_message_content(raw.get("content"))
            if not content.strip():
                continue
            normalized_role = "assistant" if role in {"ai"} else ("user" if role == "human" else str(role))
            messages.append(SessionMessageOut(role=normalized_role, content=content))
        return messages

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
