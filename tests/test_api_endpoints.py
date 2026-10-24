"""Regressions for comma-separated endpoint configuration and the actual CLI."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import generate_cusd_dataset as generator

FIRST = "http://127.0.0.1:30000/v1"
SECOND = "http://127.0.0.1:30001/v1"


class EndpointTests(unittest.TestCase):
    def test_comma_separated_urls(self):
        expected = [{"api_base": FIRST, "max_workers": 4},
                    {"api_base": SECOND, "max_workers": 4}]
        for values in [f"{FIRST},{SECOND}", [f"{FIRST},{SECOND}"], [FIRST, SECOND]]:
            with self.subTest(values=values):
                self.assertEqual(generator.build_api_endpoints(values, 4), expected)

    def test_single_url(self):
        self.assertEqual(generator.build_api_endpoints(FIRST, 2),
                         [{"api_base": FIRST, "max_workers": 2}])

    def test_whitespace_and_empty_segments(self):
        result = generator.build_api_endpoints(f" , {FIRST}, , {SECOND}, ")
        self.assertEqual([item["api_base"] for item in result], [FIRST, SECOND])

    def test_environment_default_and_cli_precedence(self):
        with patch.object(generator, "DEFAULT_API_BASES", f"{FIRST},{SECOND}"):
            self.assertEqual(len(generator.build_api_endpoints()), 2)
            self.assertEqual(generator.build_api_endpoints(SECOND, 1),
                             [{"api_base": SECOND, "max_workers": 1}])

    def test_empty_configuration_is_rejected(self):
        for values in ["", " , ", [], [" , "]]:
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "at least one"):
                generator.build_api_endpoints(values)
        with patch.object(generator, "DEFAULT_API_BASES", ""), self.assertRaises(ValueError):
            generator.build_api_endpoints()

    def test_actual_cli_initializes_two_endpoints_without_api_calls(self):
        # No candidate prompts means no API calls; this exercises argparse and queue setup.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            source = path / "empty.json"
            source.write_text('{"users": []}', encoding="utf-8")
            output = path / "output.json"
            command = [sys.executable, "-B", str(ROOT / "src/generate_cusd_dataset.py"),
                       "--input_path", str(source), "--output_file", str(output),
                       "--api_base", f"{FIRST},{SECOND}", "--max_workers", "2"]
            env = dict(os.environ, CUSD_API_BASES="http://127.0.0.1:1/v1", CUSD_MAX_WORKERS="1")
            result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Initialized 2 API endpoint(s) with 4 total worker(s).", result.stdout)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), [])
            command[command.index("--api_base") + 1] = " , "
            result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=30)
            self.assertEqual(result.returncode, 2)
            self.assertIn("at least one API base URL", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
