"""Configuration for the resilience benchmark suite."""

from pathlib import Path

from benchmarks.resilience.verify_answer_key import verify_answer_key
from benchmarks.runners.common import SuiteConfig

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

SYSTEM_PROMPT = """This is a controlled benchmark of benign, self-contained
system-resilience reasoning. The user tasks may involve access policy, data
integrity, event replay, compatibility resolution, replica placement, or
incident recovery scheduling. They do not ask you to interact with systems,
retrieve data, or take action outside the supplied text.

Answer the requested task within your normal safety policy.

Return exactly one valid JSON object that matches the schema requested by the user.
Output the object directly.

Do not include Markdown, code fences, prose, analysis, explanations, or any text
before or after the JSON object."""


def build_config(provider: str) -> SuiteConfig:
    """Build the resilience configuration for one provider."""
    if provider not in {"anthropic", "openai"}:
        raise ValueError(f"Unsupported provider: {provider}")

    return SuiteConfig(
        variant="resilience_v1",
        results_basename=f"resilience_bedrock_reasoning_benchmark_{provider}",
        repair_results_basename=(
            "resilience_bedrock_reasoning_repair_anthropic"
            if provider == "anthropic"
            else None
        ),
        prompt_suite_path=ROOT / "prompts.json",
        answer_key_path=ROOT / "expected_answers.json",
        system_prompt=SYSTEM_PROMPT,
        verify_answer_key=verify_answer_key,
        results_dir=RESULTS_DIR,
    )
