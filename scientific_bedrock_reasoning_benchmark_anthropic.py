#!/usr/bin/env python3
"""Run the scientific field-research suite across Anthropic models, cold-cache."""

from pathlib import Path

import operations_bedrock_reasoning_benchmark_anthropic as benchmark
from reasoning_benchmark_evaluation import load_expected_answers
from scientific_verify_answer_key import verify_answer_key


ROOT = Path(__file__).resolve().parent
PROMPT_SUITE_PATH = ROOT / "scientific_reasoning_benchmark_prompts.json"
ANSWER_KEY_PATH = ROOT / "scientific_reasoning_benchmark_expected_answers.json"


def configure_scientific_suite() -> None:
    """Point the shared Anthropic runner at the independently verified suite."""
    benchmark.BENCHMARK_VARIANT = "scientific_field_research_v1"
    benchmark.RESULTS_BASENAME = "scientific_bedrock_reasoning_benchmark_anthropic"
    benchmark.REPAIR_RESULTS_BASENAME = "scientific_bedrock_reasoning_repair_anthropic"
    benchmark.PROMPT_SUITE_PATH = PROMPT_SUITE_PATH
    benchmark.ANSWER_KEY_PATH = ANSWER_KEY_PATH
    benchmark.PROMPTS = benchmark.load_prompts()
    benchmark.EXPECTED_ANSWERS = load_expected_answers(
        (scenario_id for scenario_id, _ in benchmark.PROMPTS),
        ANSWER_KEY_PATH,
    )
    benchmark.verify_answer_key = verify_answer_key


if __name__ == "__main__":
    configure_scientific_suite()
    benchmark.main()
