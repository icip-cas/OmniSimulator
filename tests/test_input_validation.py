"""Exercise malformed inputs and temporal leakage without loading model libraries."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_input import validate_path, validate_prepared


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.example = json.loads((ROOT / "dataset/example_input.json").read_text(encoding="utf-8"))

    def test_prepared_file_and_list_forms(self):
        self.assertEqual(validate_prepared(self.example), (1, 1))
        self.assertEqual(validate_prepared(self.example["users"]), (1, 1))

    def test_malformed_events(self):
        for field, value in [("timestamp", "2025-02-30 09:30:00"),
                             ("timestamp", None), ("type", "video browsing"),
                             ("type", []), ("context", []), ("action", {}),
                             ("action", [None])]:
            with self.subTest(field=field, value=value):
                data = copy.deepcopy(self.example)
                data["users"][0]["test_action"][field] = value
                with self.assertRaises(ValueError):
                    validate_prepared(data)

    def test_future_and_equal_time_history_rejected(self):
        for timestamp in ["2025-11-02 09:30:00", "2025-11-03 09:30:00"]:
            with self.subTest(timestamp=timestamp):
                data = copy.deepcopy(self.example)
                data["users"][0]["action_history"][0]["timestamp"] = timestamp
                with self.assertRaisesRegex(ValueError, "strictly precede"):
                    validate_prepared(data)

    def windowed_input(self):
        user = copy.deepcopy(self.example["users"][0])
        history = user.pop("action_history")
        target = user.pop("test_action")
        user.update(base_history=history, test_time_all_actions=[target],
                    test_actions=[{"action": copy.deepcopy(target), "test_time_index": 0}])
        return {"users": [user]}

    def test_windowed_input(self):
        self.assertEqual(validate_prepared(self.windowed_input()), (1, 1))

    def test_invalid_window_indices(self):
        for position in [-1, 1, True, "0", None]:
            with self.subTest(position=position):
                data = self.windowed_input()
                data["users"][0]["test_actions"][0]["test_time_index"] = position
                with self.assertRaisesRegex(ValueError, "zero-based index"):
                    validate_prepared(data)

    def test_mismatched_and_duplicate_targets(self):
        data = self.windowed_input()
        data["users"][0]["test_actions"][0]["action"]["context"] = {}
        with self.assertRaisesRegex(ValueError, "must match"):
            validate_prepared(data)
        data = self.windowed_input()
        data["users"][0]["test_actions"] *= 2
        with self.assertRaisesRegex(ValueError, "duplicate target"):
            validate_prepared(data)

    def test_raw_directory_and_duplicate_users(self):
        event = self.example["users"][0]["test_action"]
        history = []
        for day in range(20, 0, -1):
            item = copy.deepcopy(event)
            item["timestamp"] = f"2025-10-{day:02d} 09:30:00"
            history.append(item)
        raw = {"example_user": {"user_profile": "", "action_history": history}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "one.json").write_text(json.dumps(raw), encoding="utf-8")
            self.assertIn("1 users with at least 20 events", validate_path(path))
            (path / "two.json").write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate user ID"):
                validate_path(path)

    def test_empty_and_ineligible_inputs(self):
        for data in [{}, [], {"users": []}, {"users": [None]}]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate_prepared(data)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with self.assertRaisesRegex(ValueError, "no JSON"):
                validate_path(path)
            (path / "data.json").write_text('{"user": {"action_history": []}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "20 events"):
                validate_path(path)

    def test_cli_error_is_nonzero_and_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            original = b'{"users": ['
            path.write_bytes(original)
            result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/validate_input.py"), str(path)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1)
            self.assertIn("Invalid input:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
