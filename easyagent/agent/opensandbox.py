from __future__ import annotations

from datetime import timedelta
import logging
import threading
from typing import Callable
from typing import Any

from deepagents.backends.protocol import (
    BackendProtocol,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from psycopg_pool import ConnectionPool

from opensandbox import SandboxSync
from opensandbox.config.connection_sync import ConnectionConfigSync
from opensandbox.models.execd import RunCommandOpts

logger = logging.getLogger(__name__)


class OpenSandboxBackend(BaseSandbox):
    """DeepAgents sandbox backend powered by OpenSandbox."""

    def __init__(
        self,
        sandbox: SandboxSync,
        *,
        max_output_bytes: int = 100_000,
        command_timeout_seconds: float | None = 120.0,
        on_activity: Callable[[], None] | None = None,
    ) -> None:
        if not hasattr(sandbox, "commands") or not hasattr(sandbox, "files") or not hasattr(sandbox, "id"):
            raise TypeError("sandbox must be an opensandbox.SandboxSync-like instance")
        self._sandbox = sandbox
        self._max_output_bytes = max_output_bytes
        self._command_timeout_seconds = command_timeout_seconds
        self._on_activity = on_activity

    @classmethod
    def create(
        cls,
        *,
        image: str,
        api_key: str | None = None,
        domain: str | None = None,
        protocol: str = "https",
        use_server_proxy: bool = True,
        timeout_seconds: float = 600.0,
        ready_timeout_seconds: float = 30.0,
        env: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        resource: dict[str, str] | None = None,
        extensions: dict[str, str] | None = None,
        entrypoint: list[str] | None = None,
        max_output_bytes: int = 100_000,
        command_timeout_seconds: float | None = 120.0,
        on_activity: Callable[[], None] | None = None,
    ) -> OpenSandboxBackend:
        config = ConnectionConfigSync(
            api_key=api_key,
            domain=domain,
            protocol=protocol,
            use_server_proxy=use_server_proxy,
        )
        sandbox = SandboxSync.create(
            image=image,
            timeout=timedelta(seconds=timeout_seconds),
            ready_timeout=timedelta(seconds=ready_timeout_seconds),
            env=env,
            metadata=metadata,
            resource=resource,
            extensions=extensions,
            entrypoint=entrypoint,
            connection_config=config,
        )
        return cls(
            sandbox,
            max_output_bytes=max_output_bytes,
            command_timeout_seconds=command_timeout_seconds,
            on_activity=on_activity,
        )

    @classmethod
    def connect(
        cls,
        sandbox_id: str,
        *,
        api_key: str | None = None,
        domain: str | None = None,
        protocol: str = "https",
        use_server_proxy: bool = True,
        connect_timeout_seconds: float = 30.0,
        max_output_bytes: int = 100_000,
        command_timeout_seconds: float | None = 120.0,
        on_activity: Callable[[], None] | None = None,
    ) -> OpenSandboxBackend:
        config = ConnectionConfigSync(
            api_key=api_key,
            domain=domain,
            protocol=protocol,
            use_server_proxy=use_server_proxy,
        )
        sandbox = SandboxSync.connect(
            sandbox_id=sandbox_id,
            connection_config=config,
            connect_timeout=timedelta(seconds=connect_timeout_seconds),
        )
        return cls(
            sandbox,
            max_output_bytes=max_output_bytes,
            command_timeout_seconds=command_timeout_seconds,
            on_activity=on_activity,
        )

    @property
    def id(self) -> str:
        return str(self._sandbox.id)

    def execute(self, command: str) -> ExecuteResponse:
        if not command.strip():
            return ExecuteResponse(output="Error: Command must be a non-empty string.", exit_code=1, truncated=False)
        self._mark_activity()
        try:
            opts = self._build_run_opts()
            execution = self._sandbox.commands.run(command, opts=opts)
            exit_code = self._resolve_exit_code(getattr(execution, "id", None), getattr(execution, "error", None))

            output_parts: list[str] = []
            logs = getattr(execution, "logs", None)
            stdout_logs = getattr(logs, "stdout", []) if logs is not None else []
            stderr_logs = getattr(logs, "stderr", []) if logs is not None else []

            output_parts.extend(_safe_text(item) for item in stdout_logs if _safe_text(item))
            output_parts.extend(f"[stderr] {text}" for text in (_safe_text(item) for item in stderr_logs) if text)

            if getattr(execution, "error", None) is not None:
                err = execution.error
                name = getattr(err, "name", "") or "ExecutionError"
                value = getattr(err, "value", "") or "unknown failure"
                output_parts.append(f"[stderr] {name}: {value}")

            output = "\n".join(output_parts).strip() or "<no output>"

            truncated = False
            output_bytes = output.encode("utf-8")
            if len(output_bytes) > self._max_output_bytes:
                output = output_bytes[: self._max_output_bytes].decode("utf-8", errors="ignore")
                output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
                truncated = True

            if exit_code not in (None, 0):
                output = f"{output.rstrip()}\n\nExit code: {exit_code}"

            return ExecuteResponse(
                output=output,
                exit_code=exit_code,
                truncated=truncated,
            )
        except Exception as exc:  # noqa: BLE001
            return ExecuteResponse(
                output=f"Error executing command: {exc}",
                exit_code=1,
                truncated=False,
            )
        finally:
            self._mark_activity()

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        self._mark_activity()
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                self._sandbox.files.write_file(path, content)
                responses.append(FileUploadResponse(path=path, error=None))
            except Exception as exc:  # noqa: BLE001
                responses.append(FileUploadResponse(path=path, error=_map_upload_error(str(exc))))
        self._mark_activity()
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        self._mark_activity()
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                content = self._sandbox.files.read_bytes(path)
                responses.append(FileDownloadResponse(path=path, content=content, error=None))
            except Exception as exc:  # noqa: BLE001
                responses.append(FileDownloadResponse(path=path, content=None, error=_map_download_error(str(exc))))
        self._mark_activity()
        return responses

    def renew(self, timeout_seconds: float) -> Any:
        return self._sandbox.renew(timedelta(seconds=timeout_seconds))

    def close(self) -> None:
        self._sandbox.close()

    def kill(self) -> None:
        self._sandbox.kill()

    def _resolve_exit_code(self, execution_id: str | None, execution_error: Any) -> int | None:
        if execution_id:
            try:
                status = self._sandbox.commands.get_command_status(execution_id)
                if getattr(status, "exit_code", None) is not None:
                    return status.exit_code
            except Exception:
                pass
        if execution_error is not None:
            return 1
        return 0

    def _build_run_opts(self) -> Any:
        if self._command_timeout_seconds is None:
            return None
        return RunCommandOpts(timeout=timedelta(seconds=self._command_timeout_seconds))

    def _mark_activity(self) -> None:
        if callable(self._on_activity):
            try:
                self._on_activity()
            except Exception:
                logger.debug("failed to record sandbox activity", exc_info=True)


def _safe_text(item: Any) -> str:
    value = getattr(item, "text", None)
    if isinstance(value, str):
        return value
    return ""


def _map_upload_error(message: str) -> str:
    text = message.lower()
    if "permission" in text or "denied" in text:
        return "permission_denied"
    if "invalid" in text and "path" in text:
        return "invalid_path"
    return "invalid_path"


def _map_download_error(message: str) -> str:
    text = message.lower()
    if "not found" in text or "no such file" in text:
        return "file_not_found"
    if "permission" in text or "denied" in text:
        return "permission_denied"
    if "is a directory" in text:
        return "is_directory"
    if "invalid" in text and "path" in text:
        return "invalid_path"
    return "invalid_path"


class OpenSandboxThreadBackendFactory:
    """Map `(user_id, thread_id)` to a stable OpenSandbox backend across instances."""

    def __init__(
        self,
        *,
        db_url: str,
        image: str,
        api_key: str | None = None,
        domain: str | None = None,
        protocol: str = "https",
        use_server_proxy: bool = True,
        sandbox_timeout_seconds: float = 600.0,
        ready_timeout_seconds: float = 30.0,
        connect_timeout_seconds: float = 30.0,
        max_output_bytes: int = 100_000,
        command_timeout_seconds: float | None = 120.0,
        idle_timeout_seconds: float = 3600.0,
        sweeper_interval_seconds: float = 60.0,
        pg_pool_min_size: int = 1,
        pg_pool_max_size: int = 20,
    ) -> None:
        self._image = image
        self._api_key = api_key
        self._domain = domain
        self._protocol = protocol
        self._use_server_proxy = use_server_proxy
        self._sandbox_timeout_seconds = sandbox_timeout_seconds
        self._ready_timeout_seconds = ready_timeout_seconds
        self._connect_timeout_seconds = connect_timeout_seconds
        self._max_output_bytes = max_output_bytes
        self._command_timeout_seconds = command_timeout_seconds
        self._idle_timeout_seconds = idle_timeout_seconds
        self._sweeper_interval_seconds = sweeper_interval_seconds
        self._cache: dict[tuple[str, str], OpenSandboxBackend] = {}
        self._cache_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sweeper_thread: threading.Thread | None = None

        self._pool = ConnectionPool(
            self._normalize_postgres_url(db_url),
            min_size=pg_pool_min_size,
            max_size=pg_pool_max_size,
            kwargs={"autocommit": False},
        )
        self._ensure_bindings_table()
        if self._idle_timeout_seconds > 0:
            self._start_sweeper()

    def __call__(self, runtime: Any) -> BackendProtocol:
        user_id, thread_id = self._extract_ids(runtime)
        cache_key = (user_id, thread_id)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        backend = self._load_or_create_backend(user_id=user_id, thread_id=thread_id)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                backend.close()
                return cached
            self._cache[cache_key] = backend
            return backend

    def close(self) -> None:
        self._stop_event.set()
        if self._sweeper_thread is not None:
            self._sweeper_thread.join(timeout=3.0)
        with self._cache_lock:
            backends = list(self._cache.values())
            self._cache.clear()
        for backend in backends:
            try:
                backend.close()
            except Exception:
                pass
        self._pool.close()

    def _load_or_create_backend(self, *, user_id: str, thread_id: str) -> OpenSandboxBackend:
        key = f"{user_id}:{thread_id}"
        with self._pool.connection() as conn:
            with conn.transaction():
                conn.execute("SELECT pg_advisory_xact_lock((('x' || substr(md5(%s),1,16))::bit(64)::bigint))", (key,))
                row = conn.execute(
                    "SELECT sandbox_id FROM easyagent_sandbox_bindings WHERE user_id = %s AND thread_id = %s",
                    (user_id, thread_id),
                ).fetchone()
                if row is not None and row[0]:
                    sandbox_id = str(row[0])
                    try:
                        self._touch_binding(conn=conn, user_id=user_id, thread_id=thread_id)
                        return self._connect_backend(sandbox_id, user_id=user_id, thread_id=thread_id)
                    except Exception:
                        pass

                backend = self._create_backend(user_id=user_id, thread_id=thread_id)
                conn.execute(
                    """
                    INSERT INTO easyagent_sandbox_bindings (user_id, thread_id, sandbox_id, last_active_at)
                    VALUES (%s, %s, %s, now())
                    ON CONFLICT (user_id, thread_id)
                    DO UPDATE SET sandbox_id = EXCLUDED.sandbox_id, last_active_at = now(), updated_at = now()
                    """,
                    (user_id, thread_id, backend.id),
                )
                return backend

    def _connect_backend(self, sandbox_id: str, *, user_id: str, thread_id: str) -> OpenSandboxBackend:
        return OpenSandboxBackend.connect(
            sandbox_id=sandbox_id,
            api_key=self._api_key,
            domain=self._domain,
            protocol=self._protocol,
            use_server_proxy=self._use_server_proxy,
            connect_timeout_seconds=self._connect_timeout_seconds,
            max_output_bytes=self._max_output_bytes,
            command_timeout_seconds=self._command_timeout_seconds,
            on_activity=lambda: self._touch_binding_now(user_id=user_id, thread_id=thread_id),
        )

    def _create_backend(self, *, user_id: str, thread_id: str) -> OpenSandboxBackend:
        return OpenSandboxBackend.create(
            image=self._image,
            api_key=self._api_key,
            domain=self._domain,
            protocol=self._protocol,
            use_server_proxy=self._use_server_proxy,
            timeout_seconds=self._sandbox_timeout_seconds,
            ready_timeout_seconds=self._ready_timeout_seconds,
            max_output_bytes=self._max_output_bytes,
            command_timeout_seconds=self._command_timeout_seconds,
            on_activity=lambda: self._touch_binding_now(user_id=user_id, thread_id=thread_id),
        )

    def _ensure_bindings_table(self) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS easyagent_sandbox_bindings (
                    user_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    sandbox_id TEXT NOT NULL,
                    last_active_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (user_id, thread_id)
                )
                """
            )
            conn.execute(
                "ALTER TABLE easyagent_sandbox_bindings ADD COLUMN IF NOT EXISTS last_active_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            )
            conn.commit()

    @staticmethod
    def _extract_ids(runtime: Any) -> tuple[str, str]:
        config = getattr(runtime, "config", {})
        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        user_id = configurable.get("user_id") if isinstance(configurable, dict) else None
        thread_id = configurable.get("thread_id") if isinstance(configurable, dict) else None
        user = str(user_id).strip() if user_id is not None else ""
        thread = str(thread_id).strip() if thread_id is not None else ""
        if not user:
            raise ValueError("`user_id` is required for OpenSandboxThreadBackendFactory")
        if not thread:
            raise ValueError("`thread_id` is required for OpenSandboxThreadBackendFactory")
        return user, thread

    @staticmethod
    def _normalize_postgres_url(url: str) -> str:
        if url.startswith("postgresql+psycopg://"):
            return "postgresql://" + url[len("postgresql+psycopg://") :]
        if url.startswith("postgres+psycopg://"):
            return "postgres://" + url[len("postgres+psycopg://") :]
        if url.startswith("postgresql://") or url.startswith("postgres://"):
            return url
        raise ValueError("OpenSandboxThreadBackendFactory requires a Postgres db_url")

    def _touch_binding(self, *, conn: Any, user_id: str, thread_id: str) -> None:
        conn.execute(
            """
            UPDATE easyagent_sandbox_bindings
            SET last_active_at = now(), updated_at = now()
            WHERE user_id = %s AND thread_id = %s
            """,
            (user_id, thread_id),
        )

    def _touch_binding_now(self, *, user_id: str, thread_id: str) -> None:
        with self._pool.connection() as conn:
            with conn.transaction():
                self._touch_binding(conn=conn, user_id=user_id, thread_id=thread_id)

    def _start_sweeper(self) -> None:
        self._sweeper_thread = threading.Thread(
            target=self._sweeper_loop,
            name="opensandbox-idle-sweeper",
            daemon=True,
        )
        self._sweeper_thread.start()

    def _sweeper_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._sweep_idle_once()
            except Exception:
                logger.exception("opensandbox idle sweep failed")
            self._stop_event.wait(self._sweeper_interval_seconds)

    def _sweep_idle_once(self) -> None:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT user_id, thread_id, sandbox_id
                FROM easyagent_sandbox_bindings
                WHERE last_active_at < now() - make_interval(secs => %s)
                LIMIT 100
                """,
                (self._idle_timeout_seconds,),
            ).fetchall()

        for row in rows:
            user_id = str(row[0])
            thread_id = str(row[1])
            sandbox_id = str(row[2])
            self._retire_if_still_idle(user_id=user_id, thread_id=thread_id, sandbox_id=sandbox_id)

    def _retire_if_still_idle(self, *, user_id: str, thread_id: str, sandbox_id: str) -> None:
        lock_key = f"{user_id}:{thread_id}"
        with self._pool.connection() as conn:
            with conn.transaction():
                got_lock = conn.execute(
                    "SELECT pg_try_advisory_xact_lock((('x' || substr(md5(%s),1,16))::bit(64)::bigint))",
                    (lock_key,),
                ).fetchone()
                if not got_lock or not got_lock[0]:
                    return

                row = conn.execute(
                    """
                    SELECT sandbox_id
                    FROM easyagent_sandbox_bindings
                    WHERE user_id = %s AND thread_id = %s
                      AND last_active_at < now() - make_interval(secs => %s)
                    """,
                    (user_id, thread_id, self._idle_timeout_seconds),
                ).fetchone()
                if row is None:
                    return
                current_sandbox_id = str(row[0])

                self._kill_sandbox_best_effort(current_sandbox_id, user_id=user_id, thread_id=thread_id)
                conn.execute(
                    "DELETE FROM easyagent_sandbox_bindings WHERE user_id = %s AND thread_id = %s",
                    (user_id, thread_id),
                )

    def _kill_sandbox_best_effort(self, sandbox_id: str, *, user_id: str, thread_id: str) -> None:
        try:
            backend = OpenSandboxBackend.connect(
                sandbox_id=sandbox_id,
                api_key=self._api_key,
                domain=self._domain,
                protocol=self._protocol,
                use_server_proxy=self._use_server_proxy,
                connect_timeout_seconds=self._connect_timeout_seconds,
            )
            try:
                backend.kill()
            finally:
                backend.close()
        except Exception:
            logger.debug("failed to kill idle sandbox %s", sandbox_id, exc_info=True)

        with self._cache_lock:
            cached = self._cache.pop((user_id, thread_id), None)
        if cached is not None:
            try:
                cached.close()
            except Exception:
                pass
