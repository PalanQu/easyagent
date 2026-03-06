import logging
import os
import sys

_DEFAULT_LOG_FORMAT = (
    "%(asctime)s %(levelname)s [%(name)s] %(filename)s:%(lineno)d - %(message)s"
)
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S%z"


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

    target_logger.addHandler(handler)
    target_logger.setLevel(target_level)
    target_logger.propagate = False
    return target_logger
