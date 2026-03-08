import unittest
import threading
from types import SimpleNamespace

from easyagent.agent.opensandbox import OpenSandboxBackend, OpenSandboxThreadBackendFactory


class _FakeCommands:
    def __init__(self) -> None:
        self._execution = None
        self._status = None
        self._error: Exception | None = None

    def set_result(self, execution, status=None) -> None:  # noqa: ANN001
        self._execution = execution
        self._status = status
        self._error = None

    def set_error(self, error: Exception) -> None:
        self._error = error

    def run(self, command: str, *, opts=None):  # noqa: ANN001, ARG002
        if self._error is not None:
            raise self._error
        return self._execution

    def get_command_status(self, execution_id: str):  # noqa: ANN001, ARG002
        return self._status


class _FakeFiles:
    def __init__(self) -> None:
        self._content_by_path: dict[str, bytes] = {}

    def write_file(self, path: str, data: bytes) -> None:
        if path == "/forbidden.txt":
            raise RuntimeError("permission denied")
        self._content_by_path[path] = data

    def read_bytes(self, path: str) -> bytes:
        if path not in self._content_by_path:
            raise RuntimeError("file not found")
        return self._content_by_path[path]


class _FakeSandbox:
    def __init__(self, sandbox_id: str = "sb-1") -> None:
        self.id = sandbox_id
        self.commands = _FakeCommands()
        self.files = _FakeFiles()
        self.closed = False
        self.killed = False

    def close(self) -> None:
        self.closed = True

    def kill(self) -> None:
        self.killed = True


class TestOpenSandboxBackendUnit(unittest.TestCase):
    def test_execute_merges_stdout_and_stderr(self) -> None:
        sandbox = _FakeSandbox()
        sandbox.commands.set_result(
            execution=SimpleNamespace(
                id="exec-1",
                logs=SimpleNamespace(
                    stdout=[SimpleNamespace(text="hello")],
                    stderr=[SimpleNamespace(text="oops")],
                ),
                error=None,
            ),
            status=SimpleNamespace(exit_code=0),
        )
        backend = OpenSandboxBackend(sandbox)

        result = backend.execute("echo hello")

        self.assertEqual(result.exit_code, 0)
        self.assertIn("hello", result.output)
        self.assertIn("[stderr] oops", result.output)
        self.assertFalse(result.truncated)

    def test_execute_non_zero_exit_code_appends_exit_line(self) -> None:
        sandbox = _FakeSandbox()
        sandbox.commands.set_result(
            execution=SimpleNamespace(
                id="exec-2",
                logs=SimpleNamespace(stdout=[SimpleNamespace(text="bad")], stderr=[]),
                error=None,
            ),
            status=SimpleNamespace(exit_code=2),
        )
        backend = OpenSandboxBackend(sandbox)

        result = backend.execute("false")

        self.assertEqual(result.exit_code, 2)
        self.assertIn("Exit code: 2", result.output)

    def test_execute_handles_sdk_exception(self) -> None:
        sandbox = _FakeSandbox()
        sandbox.commands.set_error(RuntimeError("network down"))
        backend = OpenSandboxBackend(sandbox)

        result = backend.execute("echo test")

        self.assertEqual(result.exit_code, 1)
        self.assertIn("Error executing command", result.output)

    def test_upload_and_download_files(self) -> None:
        sandbox = _FakeSandbox()
        backend = OpenSandboxBackend(sandbox)

        upload_res = backend.upload_files(
            [
                ("/ok.txt", b"ok"),
                ("/forbidden.txt", b"x"),
            ]
        )
        download_res = backend.download_files(["/ok.txt", "/missing.txt"])

        self.assertIsNone(upload_res[0].error)
        self.assertEqual(upload_res[1].error, "permission_denied")
        self.assertEqual(download_res[0].content, b"ok")
        self.assertEqual(download_res[1].error, "file_not_found")

    def test_activity_hook_called_for_operations(self) -> None:
        sandbox = _FakeSandbox()
        sandbox.commands.set_result(
            execution=SimpleNamespace(
                id="exec-3",
                logs=SimpleNamespace(stdout=[SimpleNamespace(text="ok")], stderr=[]),
                error=None,
            ),
            status=SimpleNamespace(exit_code=0),
        )
        counter = {"n": 0}

        def _on_activity() -> None:
            counter["n"] += 1

        backend = OpenSandboxBackend(sandbox, on_activity=_on_activity)
        backend.execute("echo ok")
        backend.upload_files([("/a.txt", b"a")])
        backend.download_files(["/a.txt"])

        self.assertGreaterEqual(counter["n"], 4)

    def test_thread_factory_reuses_same_user_and_thread(self) -> None:
        factory = OpenSandboxThreadBackendFactory.__new__(OpenSandboxThreadBackendFactory)
        factory._cache = {}
        factory._cache_lock = threading.Lock()
        created: list[OpenSandboxBackend] = []

        def _loader(*, user_id: str, thread_id: str) -> OpenSandboxBackend:
            backend = OpenSandboxBackend(_FakeSandbox(f"{user_id}:{thread_id}:{len(created)}"))
            created.append(backend)
            return backend

        factory._load_or_create_backend = _loader  # type: ignore[method-assign]

        runtime = SimpleNamespace(config={"configurable": {"user_id": "u1", "thread_id": "t1"}})
        a = factory(runtime)
        b = factory(runtime)

        self.assertIs(a, b)
        self.assertEqual(len(created), 1)

    def test_thread_factory_splits_different_threads_for_same_user(self) -> None:
        factory = OpenSandboxThreadBackendFactory.__new__(OpenSandboxThreadBackendFactory)
        factory._cache = {}
        factory._cache_lock = threading.Lock()
        created: list[OpenSandboxBackend] = []

        def _loader(*, user_id: str, thread_id: str) -> OpenSandboxBackend:
            backend = OpenSandboxBackend(_FakeSandbox(f"{user_id}:{thread_id}:{len(created)}"))
            created.append(backend)
            return backend

        factory._load_or_create_backend = _loader  # type: ignore[method-assign]

        rt1 = SimpleNamespace(config={"configurable": {"user_id": "u1", "thread_id": "t1"}})
        rt2 = SimpleNamespace(config={"configurable": {"user_id": "u1", "thread_id": "t2"}})

        b1 = factory(rt1)
        b2 = factory(rt2)

        self.assertIsNot(b1, b2)
        self.assertEqual(len(created), 2)

    def test_thread_factory_requires_user_and_thread(self) -> None:
        with self.assertRaisesRegex(ValueError, "user_id"):
            OpenSandboxThreadBackendFactory._extract_ids(SimpleNamespace(config={"configurable": {"thread_id": "t"}}))
        with self.assertRaisesRegex(ValueError, "thread_id"):
            OpenSandboxThreadBackendFactory._extract_ids(SimpleNamespace(config={"configurable": {"user_id": "u"}}))


if __name__ == "__main__":
    unittest.main()
