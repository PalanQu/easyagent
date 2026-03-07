import logging
import os
import sys
from typing import Any

from fastapi import Request

_DEFAULT_LOG_FORMAT = (
    "%(asctime)s %(levelname)s [%(name)s]%(request_context)s "
    "%(filename)s:%(lineno)d - %(message)s"
)
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_CONTEXT_VALUE = "-"
_DEFAULT_CONTEXT = {
    "user_id": _DEFAULT_CONTEXT_VALUE,
    "thread_id": _DEFAULT_CONTEXT_VALUE,
}


def _to_context_value(value: str | int | None) -> str:
    if value is None:
        return _DEFAULT_CONTEXT_VALUE
    return str(value)


def set_request_log_context(
    request: Request,
    *,
    user_id: str | int | None = None,
    thread_id: str | int | None = None,
) -> None:
    """Store request-scoped logging context on request.state."""
    context: dict[str, str] = {}
    if user_id not in (None, "", _DEFAULT_CONTEXT_VALUE):
        context["user_id"] = _to_context_value(user_id)
    if thread_id not in (None, "", _DEFAULT_CONTEXT_VALUE):
        context["thread_id"] = _to_context_value(thread_id)
    request.state.log_context = context


class RequestContextFilter(logging.Filter):
    """Inject default request context fields into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.user_id = _to_context_value(getattr(record, "user_id", _DEFAULT_CONTEXT_VALUE))
        record.thread_id = _to_context_value(getattr(record, "thread_id", _DEFAULT_CONTEXT_VALUE))

        context_parts: list[str] = []
        if record.user_id != _DEFAULT_CONTEXT_VALUE:
            context_parts.append(f"user_id={record.user_id}")
        if record.thread_id != _DEFAULT_CONTEXT_VALUE:
            context_parts.append(f"thread_id={record.thread_id}")

        record.request_context = f" [{' '.join(context_parts)}]" if context_parts else ""
        return True


class ContextLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that merges explicit extra fields with default context."""

    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        extra = dict(_DEFAULT_CONTEXT)
        extra.update(self.extra)
        message_extra = kwargs.get("extra")
        if isinstance(message_extra, dict):
            extra.update(message_extra)
        kwargs["extra"] = extra
        return msg, kwargs


def get_request_logger(request: Request, name: str) -> ContextLoggerAdapter:
    """Return a logger adapter populated from request.state log context."""
    extra = dict(_DEFAULT_CONTEXT)
    context = getattr(request.state, "log_context", None)
    if isinstance(context, dict):
        for key in ("user_id", "thread_id"):
            value = context.get(key)
            if value not in (None, "", _DEFAULT_CONTEXT_VALUE):
                extra[key] = _to_context_value(value)
    return ContextLoggerAdapter(logging.getLogger(name), extra)


def _parse_level(level: str | int | None) -> int:
    if level is None:
        level = os.getenv("EASYAGENT_LOG_LEVEL", "INFO")

    if isinstance(level, int):
        return level

    parsed = logging.getLevelName(level.upper())
    if isinstance(parsed, int):
        return parsed

    raise ValueError(f"invalid log level: {level}")


def setup_logging(
    *,
    level: str | int | None = None,
    logger_name: str = "",
) -> logging.Logger:
    """Configure process logging with a production-safe text format.

    The output includes file and line information, e.g. `agent.py:42`.
    """
    target_logger = logging.getLogger(logger_name)
    target_level = _parse_level(level)

    formatter = logging.Formatter(_DEFAULT_LOG_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(RequestContextFilter())

    if target_logger.handlers:
        target_logger.handlers.clear()
    target_logger.addHandler(handler)
    target_logger.setLevel(target_level)
    target_logger.propagate = False

    for uvicorn_logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(uvicorn_logger_name)
        if uvicorn_logger.handlers:
            uvicorn_logger.handlers.clear()
        uvicorn_logger.addHandler(handler)
        uvicorn_logger.setLevel(target_level)
        uvicorn_logger.propagate = False

    return target_logger
