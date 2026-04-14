from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks.mvp.features import _preflight_video_root


class MVPExtractWarningTests(unittest.TestCase):
    def test_missing_videos_root_writes_warning_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            missing_root = tmp_path / "videos" / "missing"
            warning_log = tmp_path / "artifacts" / "features" / "mvp" / "extract_warnings.log"

            with self.assertRaises(FileNotFoundError):
                _preflight_video_root(missing_root, warning_log)

            self.assertTrue(warning_log.exists())
            content = warning_log.read_text(encoding="utf-8")
            self.assertIn("missing_videos_root", content)


if __name__ == "__main__":
    unittest.main()
