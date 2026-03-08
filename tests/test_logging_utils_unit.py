import logging
import unittest
from types import SimpleNamespace

from easyagent.utils.logging import (
    ContextLoggerAdapter,
    RequestContextFilter,
    _parse_level,
    get_request_logger,
)


class _FakeRequest:
    def __init__(self, log_context=None):  # noqa: ANN001
        self.state = SimpleNamespace(log_context=log_context)


class TestLoggingUtilsUnit(unittest.TestCase):
    def test_parse_level(self) -> None:
        self.assertEqual(_parse_level(logging.DEBUG), logging.DEBUG)
        self.assertEqual(_parse_level("warning"), logging.WARNING)
        with self.assertRaisesRegex(ValueError, "invalid log level"):
            _parse_level("NOT_A_LEVEL")

    def test_request_context_filter_formats_context(self) -> None:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        record.user_id = "u1"
        record.thread_id = "t1"
        RequestContextFilter().filter(record)
        self.assertEqual(record.request_context, " [user_id=u1 thread_id=t1]")

    def test_context_logger_adapter_merges_extra(self) -> None:
        logger = logging.getLogger("test_context_logger_adapter_merges_extra")
        adapter = ContextLoggerAdapter(logger, {"user_id": "adapter-user"})
        _, kwargs = adapter.process("msg", {"extra": {"thread_id": "thread-x"}})
        self.assertEqual(kwargs["extra"]["user_id"], "adapter-user")
        self.assertEqual(kwargs["extra"]["thread_id"], "thread-x")

    def test_get_request_logger_uses_request_state_context(self) -> None:
        request = _FakeRequest(log_context={"user_id": "u2", "thread_id": "t2"})
        adapter = get_request_logger(request, "test_logger")
        self.assertEqual(adapter.extra["user_id"], "u2")
        self.assertEqual(adapter.extra["thread_id"], "t2")


if __name__ == "__main__":
    unittest.main()
