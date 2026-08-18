"""Shared configuration and file-handling utilities for benchmark runners."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reasoning_benchmark_evaluation import load_expected_answers

AnswerKeyVerifier = Callable[[], Mapping[str, Any]]
Result = dict[str, Any]


@dataclass(frozen=True, slots=True)
class SuiteConfig:
    """Immutable inputs that define one benchmark suite."""

    variant: str
    results_basename: str
    prompt_suite_path: Path
    answer_key_path: Path
    system_prompt: str
    verify_answer_key: AnswerKeyVerifier
    results_dir: Path
    repair_results_basename: str | None = None


def load_prompts(prompt_suite_path: Path) -> tuple[tuple[str, str], ...]:
    """Load and validate a version-two prompt suite."""
    try:
        suite = json.loads(prompt_suite_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load prompt suite: {prompt_suite_path}") from exc

    if suite.get("version") != 2:
        raise ValueError("Prompt suite must declare version 2.")

    scenarios = suite.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Prompt suite must contain at least one scenario.")

    prompts: list[tuple[str, str]] = []
    scenario_ids: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("Each prompt-suite scenario must be an object.")

        scenario_id = scenario.get("id")
        difficulty = scenario.get("difficulty")
        prompt = scenario.get("prompt")
        if (
            not isinstance(scenario_id, str)
            or not scenario_id
            or not isinstance(difficulty, str)
            or not difficulty
            or not isinstance(prompt, str)
            or not prompt
        ):
            raise ValueError(
                "Each scenario requires non-empty id, difficulty, and prompt fields."
            )
        if scenario_id in scenario_ids:
            raise ValueError(f"Duplicate prompt-suite scenario ID: {scenario_id}")

        scenario_ids.add(scenario_id)
        prompts.append((scenario_id, prompt))

    return tuple(prompts)


def load_suite(
    config: SuiteConfig,
) -> tuple[tuple[tuple[str, str], ...], Mapping[str, Any]]:
    """Load a suite's prompts and expected answers from its explicit config."""
    prompts = load_prompts(config.prompt_suite_path)
    expected_answers = load_expected_answers(
        (scenario_id for scenario_id, _ in prompts),
        config.answer_key_path,
    )
    return prompts, expected_answers


def create_run_metadata(
    config: SuiteConfig,
    *,
    provider_metadata: Mapping[str, Any],
    results_basename: str | None = None,
    selected_model: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Create run identity, artifact hashes, and a non-overwriting result path."""
    from datetime import datetime, timezone

    started_at = datetime.now(timezone.utc)
    timestamp = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    run_id = (
        f"{results_basename or config.results_basename}_{config.variant}_{timestamp}"
    )
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "run_started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
        "benchmark_variant": config.variant,
        **provider_metadata,
        "prompt_suite_sha256": hashlib.sha256(
            config.prompt_suite_path.read_bytes()
        ).hexdigest(),
        "answer_key_sha256": hashlib.sha256(
            config.answer_key_path.read_bytes()
        ).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(
            config.system_prompt.encode("utf-8")
        ).hexdigest(),
    }
    if selected_model is not None:
        metadata["selected_model"] = selected_model
    return metadata, config.results_dir / f"{run_id}.json"


def write_results_checkpoint(results_path: Path, results: list[Result]) -> None:
    """Atomically preserve completed calls without waiting for completion."""
    temporary_path = results_path.with_suffix(f"{results_path.suffix}.tmp")
    temporary_path.write_text(
        f"{json.dumps(results, indent=2)}\n",
        encoding="utf-8",
    )
    temporary_path.replace(results_path)
