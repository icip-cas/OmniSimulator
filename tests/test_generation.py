"""Offline regressions for selection, causal history, and exported training records."""

import copy
import io
import json
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import generate_cusd_dataset as generator


def response(answer):
    return f"<think>Offline fixture reasoning.</think>\n{answer}"


class SelectionTests(unittest.TestCase):
    def test_binary_ignores_answers_inside_reasoning(self):
        candidate = "<think>No is an alternative.</think>Yes"
        result = generator.filter_responses_by_gt([candidate, response("No"), "Yes"], 1, "binary")
        self.assertEqual(result["best_response"], candidate)
        self.assertFalse(result["all_failed"])

    def test_no_matching_binary_answer(self):
        result = generator.filter_responses_by_gt([response("No"), "Yes"], 1, "binary")
        self.assertTrue(result["all_failed"])
        self.assertIsNone(result["best_response"])

    def test_continuous_selects_closest_numeric_answer(self):
        result = generator.filter_responses_by_gt(
            [response("unknown"), response("10"), response("29"), response("50")], 30, "continuous")
        self.assertEqual(result["best_response"], response("29"))

    def test_text_requires_match(self):
        result = generator.filter_responses_by_gt(
            [response("Breakfast recipes"), response("  BREAKFAST RECIPE  ")], "breakfast recipe", "text")
        self.assertEqual(result["best_response"], response("  BREAKFAST RECIPE  "))

    def test_invalid_response_formats_are_rejected(self):
        for task, truth in [("binary", 1), ("continuous", 30), ("text", "recipe")]:
            with self.subTest(task=task):
                result = generator.filter_responses_by_gt(["Yes", "30", "recipe"], truth, task)
                self.assertTrue(result["all_failed"])


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.example = json.loads((ROOT / "dataset/example_input.json").read_text(encoding="utf-8"))
        self.random_state = random.getstate()
        self.addCleanup(random.setstate, self.random_state)

    def run_generation(self, data, responses=None, capture_history=None):
        original_builder = generator.build_forward_prompt

        def build(target, question, history, profile):
            if capture_history is not None:
                capture_history.append(copy.deepcopy(history))
            return original_builder(target, question, history, profile)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            input_path = path / "input.json"
            input_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            output_path = path / "outputs" / "traces.json"
            with patch.object(generator, "_call_llm_api_batch",
                              return_value=responses if responses is not None else
                              [response("Yes"), response("No"), response("30")]) as call, \
                    patch.object(generator, "OpenAI", side_effect=AssertionError("Network access forbidden in tests")), \
                    patch.object(generator, "build_forward_prompt", side_effect=build), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                generator.generate_cusd_dataset(
                    str(input_path), str(output_path), num_n_samples=3,
                    api_endpoints=[{"api_base": "http://127.0.0.1:1/v1", "max_workers": 1}],
                    api_key="offline-test-key", model_name="offline-test-model")
            outputs = [json.loads((output_path.parent / name).read_text(encoding="utf-8"))
                       for name in ["traces.json", "traces_sft_ready.json", "traces_binary_all_failed.json"]]
            self.assertTrue((output_path.parent / "debug_discard_reasons.log").is_file())
            return outputs, call.call_count

    def test_example_exports_paired_teacher_and_student_records(self):
        (teacher, student, failed), calls = self.run_generation(self.example)
        self.assertEqual((len(teacher), len(student), len(failed), calls), (7, 7, 0, 7))
        by_field = {item["field"]: item for item in teacher}
        for item in student:
            trace = by_field[item["field"]]
            self.assertEqual([entry["from"] for entry in item["conversations"]], ["human", "gpt"])
            self.assertEqual(item["conversations"][1], trace["conversations"][1])
            self.assertEqual(item["ground_truth"], trace["ground_truth"])
            if item["task_type"] == "binary":
                self.assertEqual(item["conversations"][0], trace["conversations"][0])
            else:
                self.assertNotEqual(item["conversations"][0], trace["conversations"][0])

    def test_failures_are_separate_from_sft_data(self):
        (teacher, student, failed), calls = self.run_generation(self.example, responses=["invalid format"])
        self.assertEqual((teacher, student, len(failed), calls), ([], [], 6, 7))
        self.assertTrue(all(item["task_type"] == "binary" for item in failed))

    def test_windowed_history_excludes_target_and_future_events(self):
        user = copy.deepcopy(self.example["users"][0])
        base = user.pop("action_history")
        target = user.pop("test_action")
        previous = copy.deepcopy(target)
        previous["timestamp"] = "2025-11-02 09:00:00"
        future = copy.deepcopy(target)
        future["timestamp"] = "2025-11-02 10:00:00"
        user.update(base_history=base, test_time_all_actions=[previous, target, future],
                    test_actions=[{"action": target, "test_time_index": 1}])
        histories = []
        self.run_generation({"users": [user]}, capture_history=histories)
        self.assertEqual(len(histories), 7)
        self.assertTrue(all(history == base + [previous] for history in histories))

    def test_ground_truth_hints_never_enter_student_text_prompt(self):
        target = self.example["users"][0]["test_action"]
        sentinel = "GROUND_TRUTH_SENTINEL_84729"
        student, teacher = generator.build_forward_prompt(
            target, {"type": "text", "field": "test_text", "true_value": sentinel}, [])
        self.assertNotIn(sentinel, student)
        self.assertIn(sentinel, teacher)


if __name__ == "__main__":
    unittest.main()
