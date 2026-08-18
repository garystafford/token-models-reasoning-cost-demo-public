"""Run the scientific field-research suite across OpenAI models."""

from benchmarks.runners import openai
from benchmarks.scientific.config import build_config


def main() -> None:
    """Run the scientific suite using the OpenAI provider."""
    openai.main(build_config("openai"))


if __name__ == "__main__":
    main()
