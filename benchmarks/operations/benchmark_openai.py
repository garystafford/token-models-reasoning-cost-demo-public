"""Run the business-operations suite across OpenAI models."""

from benchmarks.operations.config import build_config
from benchmarks.runners import openai


def main() -> None:
    """Run the operations suite using the OpenAI provider."""
    openai.main(build_config("openai"))


if __name__ == "__main__":
    main()
