import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
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

    skills_path: Path | None = None
    memories_path: Path | None = None
    tmp_path: Path | None = None

    db_backend: Literal["sqlite", "postgres"] = "sqlite"
    db_url: str | None = None
    db_echo: bool = False
    db_sqlite_filename: str = "easyagent.db"
    cluster_pg_pool_min_size: int = 1
    cluster_pg_pool_max_size: int = 20

    @model_validator(mode="after")
    def _fill_defaults(self) -> "Settings":
        self.base_path = self.base_path.expanduser().resolve()
        self.skills_path = self.skills_path or self.base_path / "skills"
        if self.memories_path is None:
            if self.local_mode:
                self.memories_path = Path("/tmp/.easyagent/memory")
            else:
                self.memories_path = self.base_path / "memory"
        self.tmp_path = self.tmp_path or self.base_path / "tmp"

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

    @classmethod
    def from_env(
        cls,
        *,
        env_prefix: str = "EASYAGENT_",
        env_file: str | Path | None = ".env",
    ) -> "Settings":
        if env_file is not None:
            load_dotenv(env_file, override=False)

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
            skills_path=Path(env("SKILLS_PATH")).expanduser() if env("SKILLS_PATH") else None,
            memories_path=Path(env("MEMORIES_PATH")).expanduser() if env("MEMORIES_PATH") else None,
            tmp_path=Path(env("TMP_PATH")).expanduser() if env("TMP_PATH") else None,
            db_backend=env("DB_BACKEND") or "sqlite",
            db_url=env("DB_URL"),
            db_echo=_parse_bool(env("DB_ECHO"), False),
            db_sqlite_filename=env("DB_SQLITE_FILENAME") or "easyagent.db",
            cluster_pg_pool_min_size=int(env("CLUSTER_PG_POOL_MIN_SIZE") or "1"),
            cluster_pg_pool_max_size=int(env("CLUSTER_PG_POOL_MAX_SIZE") or "20"),
        )
