from __future__ import annotations

import importlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


class TestScannerRefactor(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp())
        (self.test_dir / "file1.txt").write_text("content", encoding="utf-8")
        (self.test_dir / "file2.md").write_text("# Markdown", encoding="utf-8")
        (self.test_dir / "ignore_me.pyc").write_text("", encoding="utf-8")
        (self.test_dir / ".hidden").write_text("", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_scanner_import(self) -> None:
        scanner = importlib.import_module("core.data_pipeline.scanner")
        self.assertTrue(hasattr(scanner, "run_scan"))
        self.assertTrue(hasattr(scanner, "ScanConfig"))

    def test_scanner_functionality(self) -> None:
        scanner = importlib.import_module("core.data_pipeline.scanner")
        cfg = scanner.ScanConfig(roots=[self.test_dir], exts=[".txt", ".md"], allow_hash=False)
        results = scanner.run_scan(cfg)
        paths = [row.path.name for row in results]
        self.assertIn("file1.txt", paths)
        self.assertIn("file2.md", paths)
        self.assertNotIn("ignore_me.pyc", paths)
        self.assertNotIn(".hidden", paths)

    def test_cli_scan_import(self) -> None:
        try:
            importlib.import_module("scripts.pipeline.infopilot_cli.scan")
        except ImportError as exc:
            self.fail(f"Failed to import infopilot_cli.scan: {exc}")
        except ModuleNotFoundError as exc:
            self.fail(f"Module not found: {exc}")


if __name__ == "__main__":
    unittest.main()
