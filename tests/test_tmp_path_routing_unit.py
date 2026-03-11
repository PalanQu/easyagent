import shlex
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from deepagents.backends import LocalShellBackend

from easyagent.agent.agent import ClusterModeRuntimeFactory, LocalModeRuntimeFactory
from easyagent.utils.files import save_tmp_file
from easyagent.utils.settings import Settings


class TestTmpPathRoutingUnit(unittest.TestCase):
    @staticmethod
    def _local_settings(tmpdir: str) -> Settings:
        return Settings(
            model_key="dummy-key",
            model_base_url="https://example.com/v1",
            model_name="dummy-model",
            base_path=Path(tmpdir),
            local_mode=True,
        )

    @staticmethod
    def _cluster_settings(tmpdir: str) -> Settings:
        return Settings(
            model_key="dummy-key",
            model_base_url="https://example.com/v1",
            model_name="dummy-model",
            base_path=Path(tmpdir),
            local_mode=False,
            db_backend="postgres",
            db_url="postgresql://user:pass@localhost:5432/easyagent_test",
        )

    @staticmethod
    def _runtime() -> SimpleNamespace:
        return SimpleNamespace(config={"configurable": {"user_id": "user-1", "thread_id": "thread-1"}})

    def test_save_tmp_file_path_is_consistent_for_read_and_execute_in_local_and_cluster_modes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cases = [
                (
                    "local",
                    LocalModeRuntimeFactory(self._local_settings(tmpdir)),
                ),
                (
                    "cluster",
                    ClusterModeRuntimeFactory(
                        self._cluster_settings(tmpdir),
                        sandbox=LocalShellBackend(
                            root_dir=Path(tmpdir),
                            virtual_mode=True,
                            inherit_env=True,
                        ),
                    ),
                ),
            ]

            for mode_name, factory in cases:
                with self.subTest(mode=mode_name):
                    backend = factory._create_backend_factory()(self._runtime())
                    file_path = save_tmp_file(
                        str(factory.settings.tmp_path),
                        f"{mode_name}/nested/example.txt",
                        f"{mode_name} tmp content\nsecond line",
                    )

                    read_result = backend.read(file_path)
                    execute_result = backend.execute(f"cat {shlex.quote(file_path)}")

                    self.assertEqual(file_path, str(factory.settings.tmp_path / f"{mode_name}/nested/example.txt"))
                    self.assertIn(f"{mode_name} tmp content", read_result)
                    self.assertIn("second line", read_result)
                    self.assertEqual(execute_result.exit_code, 0)
                    self.assertIn(f"{mode_name} tmp content", execute_result.output)
                    self.assertIn("second line", execute_result.output)


if __name__ == "__main__":
    unittest.main()
