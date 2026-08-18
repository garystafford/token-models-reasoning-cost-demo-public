"""Run the resilience suite across Anthropic models."""

from benchmarks.resilience.config import build_config
from benchmarks.runners import anthropic


def main(argv: list[str] | None = None) -> None:
    """Run the resilience suite using the Anthropic provider."""
    anthropic.main(build_config("anthropic"), argv)


if __name__ == "__main__":
    main()
