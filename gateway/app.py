from collections.abc import Iterable
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, HttpUrl


class RegisterAgentRequest(BaseModel):
    url: HttpUrl


class AgentRecord(BaseModel):
    url: HttpUrl
    registered_at: datetime


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, AgentRecord] = {}

    def register(self, url: HttpUrl) -> AgentRecord:
        key = str(url)
        if key in self._agents:
            raise ValueError(f"agent already registered: {key}")
        record = AgentRecord(url=url, registered_at=datetime.now(UTC))
        self._agents[key] = record
        return record

    def list_all(self) -> list[AgentRecord]:
        return list(self._agents.values())

    def __iter__(self) -> Iterable[AgentRecord]:
        return iter(self._agents.values())


def create_app() -> FastAPI:
    app = FastAPI(title="Agent Gateway", version="0.1.0")
    registry = AgentRegistry()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/agents", response_model=AgentRecord, status_code=status.HTTP_201_CREATED)
    def register_agent(payload: RegisterAgentRequest) -> AgentRecord:
        try:
            return registry.register(payload.url)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/agents", response_model=list[AgentRecord])
    def list_agents() -> list[AgentRecord]:
        return registry.list_all()

    return app


app = create_app()
