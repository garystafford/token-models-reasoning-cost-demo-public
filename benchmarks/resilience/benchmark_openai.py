"""Run the resilience suite across OpenAI models."""

from benchmarks.runners import openai
from benchmarks.resilience.config import build_config


def main() -> None:
    """Run the resilience suite using the OpenAI provider."""
    openai.main(build_config("openai"))


if __name__ == "__main__":
    main()
