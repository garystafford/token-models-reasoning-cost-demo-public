import json
import tempfile
import unittest
from pathlib import Path

from reasoning_benchmark_evaluation import (
    evaluate_answer,
    evaluate_recoverable_answer,
    load_expected_answers,
)


class EvaluationTests(unittest.TestCase):
    def test_integer_contract_rejects_decimal_spelling(self) -> None:
        result = evaluate_answer('{"value": 2.0}', {"value": 2})
        self.assertFalse(result["correct"])
        self.assertIn("expected int", result["detail"])

    def test_fractional_contract_accepts_integer_spelling(self) -> None:
        result = evaluate_answer('{"value": 2}', {"value": 2.0})
        self.assertTrue(result["correct"])

    def test_non_standard_json_constants_are_rejected(self) -> None:
        result = evaluate_answer('{"value": NaN}', {"value": 1})
        self.assertEqual(result["status"], "invalid_json")

    def test_single_json_code_fence_is_recoverable(self) -> None:
        result = evaluate_recoverable_answer(
            '```json\n{"value": 1}\n```',
            {"value": 1},
        )
        self.assertTrue(result["correct"])
        self.assertEqual(result["response_format"], "json_code_fence")

    def test_answer_key_must_be_a_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answers.json"
            path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_expected_answers(["scenario"], path)


if __name__ == "__main__":
    unittest.main()
