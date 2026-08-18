"""Shared answer-key loading and strict JSON response evaluation."""

import json
import re
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from typing import Any, NoReturn

FENCED_JSON_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
OUTCOME_ORDER = (
    "strict",
    "format_only",
    "semantic_error",
    "policy_refusal",
    "truncated",
    "malformed",
    "endpoint_error",
)
OUTCOME_LABELS = {
    "strict": "CORRECT_JSON",
    "format_only": "FORMAT_ONLY",
    "semantic_error": "WRONG_ANSWER",
    "policy_refusal": "POLICY_REFUSAL",
    "truncated": "TRUNCATED",
    "malformed": "MALFORMED",
    "endpoint_error": "ENDPOINT_ERROR",
}


def load_expected_answers(
    scenario_ids: Iterable[str], answer_key_path: Path
) -> dict[str, dict[str, Any]]:
    """Load an answer key and ensure it covers the shared prompt suite exactly."""
    try:
        # Keep fractional answer-key values exact. Integers remain ``int`` so
        # they continue to express the stricter <integer> output contract.
        answer_key = _load_json(answer_key_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load answer key: {answer_key_path}") from exc

    if not isinstance(answer_key, dict):
        raise ValueError("Answer key must be a JSON object.")

    if answer_key.get("version") != 2:
        raise ValueError("Answer key must declare version 2.")

    answers = answer_key.get("answers")
    if not isinstance(answers, dict) or not answers:
        raise ValueError("Answer key must contain a non-empty answers object.")

    expected_ids = set(scenario_ids)
    answer_ids = set(answers)
    if answer_ids != expected_ids:
        missing = sorted(expected_ids - answer_ids)
        unexpected = sorted(answer_ids - expected_ids)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise ValueError(
            f"Answer key scenario IDs do not match prompt suite ({'; '.join(details)})."
        )

    if not all(isinstance(answer, dict) for answer in answers.values()):
        raise ValueError("Each answer-key entry must be a JSON object.")

    return answers


def _is_json_number(value: Any) -> bool:
    """Return whether a value is a JSON number, excluding JSON booleans."""
    return type(value) in (int, float, Decimal)


def _reject_non_json_constant(value: str) -> NoReturn:
    """Reject Python's non-standard NaN and Infinity JSON extensions."""
    raise json.JSONDecodeError(f"Invalid JSON constant: {value}", value, 0)


def _load_json(text: str) -> Any:
    """Parse strict JSON while preserving exact fractional values."""
    return json.loads(
        text,
        parse_float=Decimal,
        parse_constant=_reject_non_json_constant,
    )


def _first_json_mismatch(actual: Any, expected: Any, path: str = "$") -> str | None:
    """Return the first JSON-contract mismatch, or None when values are equal.

    Answer-key integers represent <integer> fields and therefore retain their
    exact JSON type. Fractional answer-key values represent <number> fields;
    those accept numerically equal integer or decimal JSON spellings, such as
    ``2`` and ``2.0``.
    """
    if type(expected) in (float, Decimal) and _is_json_number(actual):
        if Decimal(str(actual)) != Decimal(str(expected)):
            return f"{path}: expected {expected!r}, got {actual!r}"
        return None

    if type(actual) is not type(expected):
        return (
            f"{path}: expected {type(expected).__name__}, "
            f"got {type(actual).__name__}"
        )

    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)
            details = []
            if missing:
                details.append(f"missing keys: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected keys: {', '.join(unexpected)}")
            return f"{path}: {'; '.join(details)}"
        for key, expected_value in expected.items():
            mismatch = _first_json_mismatch(
                actual[key], expected_value, f"{path}.{key}"
            )
            if mismatch is not None:
                return mismatch
        return None

    if isinstance(expected, list):
        if len(actual) != len(expected):
            return f"{path}: expected {len(expected)} items, got {len(actual)}"
        for index, expected_value in enumerate(expected):
            mismatch = _first_json_mismatch(
                actual[index], expected_value, f"{path}[{index}]"
            )
            if mismatch is not None:
                return mismatch
        return None

    if actual != expected:
        return f"{path}: expected {expected!r}, got {actual!r}"
    return None


def evaluate_answer(
    answer_text: str, expected_answer: dict[str, Any]
) -> dict[str, Any]:
    """Strictly compare one required-JSON response to its expected answer."""
    try:
        actual_answer = _load_json(answer_text)
    except json.JSONDecodeError as exc:
        return {
            "status": "invalid_json",
            "correct": False,
            "detail": f"Invalid JSON: {exc.msg}",
        }

    mismatch = _first_json_mismatch(actual_answer, expected_answer)
    if mismatch is None:
        return {"status": "correct", "correct": True, "detail": None}
    return {"status": "incorrect", "correct": False, "detail": mismatch}


def evaluate_recoverable_answer(
    answer_text: str, expected_answer: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate a JSON answer after one unambiguous fenced-JSON extraction.

    This metric distinguishes underlying task correctness from a bare-JSON
    output-contract failure. It accepts either direct JSON or exactly one
    ```json fenced block. It is not a permissive parser for arbitrary prose.
    """
    try:
        actual_answer = _load_json(answer_text)
        response_format = "bare_json"
    except json.JSONDecodeError:
        fenced_blocks = FENCED_JSON_PATTERN.findall(answer_text)
        if len(fenced_blocks) != 1:
            return {
                "status": "not_recoverable",
                "correct": False,
                "detail": "Response is neither bare JSON nor exactly one JSON code fence.",
                "response_format": None,
            }
        try:
            actual_answer = _load_json(fenced_blocks[0])
        except json.JSONDecodeError as exc:
            return {
                "status": "not_recoverable",
                "correct": False,
                "detail": f"Fenced JSON is invalid: {exc.msg}",
                "response_format": "json_code_fence",
            }
        response_format = "json_code_fence"

    mismatch = _first_json_mismatch(actual_answer, expected_answer)
    if mismatch is None:
        return {
            "status": "correct",
            "correct": True,
            "detail": None,
            "response_format": response_format,
        }
    return {
        "status": "incorrect",
        "correct": False,
        "detail": mismatch,
        "response_format": response_format,
    }


def classify_response_outcome(
    evaluation: dict[str, Any],
    recoverable_evaluation: dict[str, Any],
    *,
    provider_refusal: bool = False,
    provider_truncated: bool = False,
) -> str:
    """Classify response correctness, formatting, and provider availability."""
    if provider_refusal:
        return "policy_refusal"
    if evaluation["correct"]:
        return "strict"
    if recoverable_evaluation["correct"]:
        return "format_only"
    if provider_truncated:
        return "truncated"
    if recoverable_evaluation["status"] == "incorrect":
        return "semantic_error"
    return "malformed"


def outcome_label(outcome: str) -> str:
    """Return the stable terminal label for a response outcome."""
    return OUTCOME_LABELS.get(outcome, "UNKNOWN")


def count_outcomes(results: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Count stable outcome categories across result records."""
    counts = {outcome: 0 for outcome in OUTCOME_ORDER}
    for result in results:
        outcome = result.get("outcome")
        if outcome not in counts:
            raise ValueError(f"Unknown or missing result outcome: {outcome!r}")
        counts[outcome] += 1
    return counts


def evaluation_label(evaluation: dict[str, Any]) -> str:
    """Return a compact terminal label for an evaluation result."""
    return {
        "correct": "PASS",
        "incorrect": "FAIL",
        "invalid_json": "INVALID",
        "not_recoverable": "UNRECOVERABLE",
        "error": "ERROR",
    }.get(evaluation["status"], "UNKNOWN")
