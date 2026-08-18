"""Run the scientific field-research suite across Anthropic models."""

from benchmarks.runners import anthropic
from benchmarks.scientific.config import build_config


def main(argv: list[str] | None = None) -> None:
    """Run the scientific suite using the Anthropic provider."""
    anthropic.main(build_config("anthropic"), argv)


if __name__ == "__main__":
    main()
