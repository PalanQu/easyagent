import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def _parse_bool(value: str | bool | None, default: bool) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False

    raise ValueError(f"invalid bool value: {value}")


class Settings(BaseModel):
    model_key: str
    model_base_url: str
    model_name: str

    base_path: Path = Field(default_factory=lambda: Path.home() / ".easyagent")
    local_mode: bool = True

    db_backend: Literal["sqlite", "postgres"] = "sqlite"
    db_url: str | None = None
    db_echo: bool = False
    db_sqlite_filename: str = "easyagent.db"
    cluster_pg_pool_min_size: int = 1
    cluster_pg_pool_max_size: int = 20

    @model_validator(mode="after")
    def _fill_defaults(self) -> "Settings":
        self.base_path = self.base_path.expanduser().resolve()

        if self.db_url:
            return self
        if self.db_backend == "sqlite":
            self.db_url = f"sqlite:///{self.base_path / self.db_sqlite_filename}"
            return self
        raise ValueError("db_url is required when db_backend='postgres'")

    @property
    def database_url(self) -> str:
        if self.db_url is None:
            raise ValueError("db_url is not initialized")
        return self.db_url

    @property
    def skills_path(self) -> Path:
        return self.base_path / "skills"

    @property
    def memories_path(self) -> Path:
        return self.base_path / "memory"

    @property
    def tmp_path(self) -> Path:
        return self.base_path / "tmp"

    @classmethod
    def from_env(
        cls,
        *,
        env_prefix: str = "EASYAGENT_",
    ) -> "Settings":
        def env(name: str) -> str | None:
            return os.getenv(f"{env_prefix}{name}")

        model_key = env("MODEL_KEY")
        model_base_url = env("MODEL_BASE_URL")
        model_name = env("MODEL_NAME")
        if not model_key or not model_base_url or not model_name:
            raise ValueError(
                "missing required env vars: "
                f"{env_prefix}MODEL_KEY, "
                f"{env_prefix}MODEL_BASE_URL, "
                f"{env_prefix}MODEL_NAME"
            )

        return cls(
            model_key=model_key,
            model_base_url=model_base_url,
            model_name=model_name,
            base_path=Path(env("BASE_PATH")).expanduser() if env("BASE_PATH") else Path.home() / ".easyagent",
            local_mode=_parse_bool(env("LOCAL_MODE"), True),
            db_backend=env("DB_BACKEND") or "sqlite",
            db_url=env("DB_URL"),
            db_echo=_parse_bool(env("DB_ECHO"), False),
            db_sqlite_filename=env("DB_SQLITE_FILENAME") or "easyagent.db",
            cluster_pg_pool_min_size=int(env("CLUSTER_PG_POOL_MIN_SIZE") or "1"),
            cluster_pg_pool_max_size=int(env("CLUSTER_PG_POOL_MAX_SIZE") or "20"),
        )
