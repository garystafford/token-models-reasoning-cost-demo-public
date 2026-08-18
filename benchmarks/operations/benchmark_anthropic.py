"""Run the business-operations suite across Anthropic models."""

from benchmarks.operations.config import build_config
from benchmarks.runners import anthropic


def main(argv: list[str] | None = None) -> None:
    """Run the operations suite using the Anthropic provider."""
    anthropic.main(build_config("anthropic"), argv)


if __name__ == "__main__":
    main()
