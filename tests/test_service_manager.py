import unittest
from pathlib import Path
from unittest.mock import patch

from detector import service_manager


class TestServiceManager(unittest.TestCase):
    def test_generate_plist(self):
        plist = service_manager.generate_plist()
        self.assertIn("<?xml", plist)
        self.assertIn(service_manager.SERVICE_LABEL, plist)
        # Should contain standard invocation
        self.assertIn("--service", plist)
        self.assertIn("--startup", plist)

    def test_log_paths(self):
        out_p, err_p = service_manager.get_log_paths()
        self.assertEqual(out_p.name, "detector.log")
        self.assertEqual(err_p.name, "detector.err.log")

    def test_status_dataclass(self):
        status = service_manager.ServiceStatus(
            running=False,
            pid=None,
            uptime=None,
            endpoints=["tcp://0.0.0.0:5555"],
            startup_enabled=False,
            log_path=Path("log"),
            err_log_path=Path("err"),
            models_dir=Path("models"),
            status_label="Stopped",
            debug=False,
        )
        self.assertFalse(status.running)
        self.assertEqual(status.endpoints, ["tcp://0.0.0.0:5555"])

    def test_is_cli_installed_true(self):
        """is_cli_installed() returns True when a symlink exists in ~/.local/bin/."""
        fake_path = service_manager.CLI_INSTALL_DIR / "detector"
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "is_symlink", return_value=True),
        ):
            # Patch the specific path check
            with patch("detector.service_manager.CLI_INSTALL_DIR", new=fake_path.parent):
                result = service_manager.is_cli_installed()
                self.assertTrue(result)

    def test_is_cli_installed_false(self):
        """is_cli_installed() returns False when no symlink exists."""
        with (
            patch.object(Path, "exists", return_value=False),
            patch.object(Path, "is_symlink", return_value=False),
        ):
            result = service_manager.is_cli_installed()
            self.assertFalse(result)

    def test_uninstall_cli_returns_tuple(self):
        """uninstall_cli() always returns a (bool, str) tuple."""
        with patch.object(Path, "is_symlink", return_value=False):
            result = service_manager.uninstall_cli()
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            ok, msg = result
            self.assertIsInstance(ok, bool)
            self.assertIsInstance(msg, str)

    def test_uninstall_cli_removes_local_symlink(self):
        """uninstall_cli() removes local symlink and reports success."""
        with (
            patch.object(Path, "is_symlink", side_effect=lambda p=None: True),
            patch.object(Path, "unlink"),
        ):
            ok, msg = service_manager.uninstall_cli()
            self.assertTrue(ok)
            self.assertIn("CLI removed from:", msg)


if __name__ == "__main__":
    unittest.main()
